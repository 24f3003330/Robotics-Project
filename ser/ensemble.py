"""Combining the per-dataset models' calibrated probabilities.

Three combiners, all decided on VALIDATION data. Nothing here is hand-tuned:
the requirement is that no weight is hardcoded, so the weighted average solves
for its weights and the stacker solves for its coefficients, both against a
held-out validation set built from speakers none of the members trained on.

  average   unweighted mean of member probabilities. The honest baseline: it
            has no fitted parameters, so it cannot overfit the validation set,
            and any fancier combiner has to beat it to be worth shipping.

  weighted  mean with per-member weights on the simplex, fitted by minimising
            validation NLL. Learns "MSP-Podcast is worth more than RAVDESS here"
            without being told.

  stacking  multinomial logistic regression over every member's log-probability
            vector. Strictly more expressive - it can learn per-CLASS trust
            ("believe CREMA-D about anger, believe IEMOCAP about sadness") -
            and correspondingly the easiest to overfit, which is why it is
            L2-regularised and why selection is by validation macro-F1.

`select_best` picks whichever wins on validation and records the full
comparison, so a stacker that lost to plain averaging is visible in the report
rather than quietly shipped.
"""

from __future__ import annotations

import numpy as np
import torch

from .labels import CANONICAL_EMOTIONS
from .metrics import evaluate

N_CLASSES = len(CANONICAL_EMOTIONS)
_EPS = 1e-9


def stack_members(member_probs, member_names):
    """{name: (n, C)} -> (M, n, C) in a fixed, recorded member order."""
    arrays = [np.asarray(member_probs[name], dtype=np.float64) for name in member_names]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"members disagree on shape: "
                         f"{ {n: a.shape for n, a in zip(member_names, arrays)} }")
    return np.stack(arrays, axis=0)


class Combiner:
    kind = "base"

    def combine(self, member_probs):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @property
    def weight_report(self):
        return {}


class AverageCombiner(Combiner):
    kind = "average"

    def __init__(self, member_names):
        self.member_names = list(member_names)

    def combine(self, member_probs):
        probs = stack_members(member_probs, self.member_names).mean(axis=0)
        return probs / probs.sum(axis=-1, keepdims=True)

    def to_dict(self):
        return {"kind": self.kind, "members": self.member_names}

    @property
    def weight_report(self):
        n = len(self.member_names)
        return {name: round(1.0 / n, 4) for name in self.member_names}


class WeightedAverageCombiner(Combiner):
    kind = "weighted"

    def __init__(self, member_names, weights):
        self.member_names = list(member_names)
        w = np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()

    @classmethod
    def fit(cls, member_probs, y_true, member_names, steps=600, lr=0.05):
        """Weights on the simplex minimising validation NLL.

        Parameterised as softmax(theta) so the weights stay non-negative and
        sum to one by construction - no projection step, no chance of a
        negative weight quietly inverting a member's vote.
        """
        stacked = stack_members(member_probs, member_names)
        x = torch.as_tensor(np.log(np.clip(stacked, _EPS, 1.0)), dtype=torch.float64)
        probs = torch.exp(x)
        y = torch.as_tensor(np.asarray(y_true), dtype=torch.long)
        theta = torch.zeros(len(member_names), dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps):
            optimiser.zero_grad()
            w = torch.softmax(theta, dim=0).view(-1, 1, 1)
            mixed = (w * probs).sum(dim=0)
            mixed = mixed / mixed.sum(dim=-1, keepdim=True)
            loss = torch.nn.functional.nll_loss(torch.log(mixed.clamp_min(_EPS)), y)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            weights = torch.softmax(theta, dim=0).numpy()
        return cls(member_names, weights)

    def combine(self, member_probs):
        stacked = stack_members(member_probs, self.member_names)
        mixed = (self.weights.reshape(-1, 1, 1) * stacked).sum(axis=0)
        return mixed / mixed.sum(axis=-1, keepdims=True)

    def to_dict(self):
        return {"kind": self.kind, "members": self.member_names,
                "weights": [round(float(w), 6) for w in self.weights]}

    @property
    def weight_report(self):
        return {n: round(float(w), 4) for n, w in zip(self.member_names, self.weights)}


class StackingCombiner(Combiner):
    kind = "stacking"

    def __init__(self, member_names, coef, intercept):
        self.member_names = list(member_names)
        self.coef = np.asarray(coef, dtype=np.float64)          # (M*C, C)
        self.intercept = np.asarray(intercept, dtype=np.float64)  # (C,)

    @staticmethod
    def _features(stacked):
        """(M, n, C) -> (n, M*C) of log-probabilities.

        Log-probs rather than probs: a linear model over log-probs can express
        the (weighted) product-of-experts rule as well as an additive one, so
        the stacker strictly contains the averaging family.
        """
        m, n, c = stacked.shape
        return np.log(np.clip(stacked, _EPS, 1.0)).transpose(1, 0, 2).reshape(n, m * c)

    @classmethod
    def fit(cls, member_probs, y_true, member_names, l2=1.0, max_iter=300):
        stacked = stack_members(member_probs, member_names)
        feats = cls._features(stacked)
        x = torch.as_tensor(feats, dtype=torch.float64)
        y = torch.as_tensor(np.asarray(y_true), dtype=torch.long)
        coef = torch.zeros((feats.shape[1], N_CLASSES), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros(N_CLASSES, dtype=torch.float64, requires_grad=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.LBFGS([coef, bias], lr=0.5, max_iter=max_iter)

        def closure():
            optimiser.zero_grad()
            loss = loss_fn(x @ coef + bias, y) + l2 / len(y) * (coef ** 2).sum()
            loss.backward()
            return loss

        optimiser.step(closure)
        return cls(member_names, coef.detach().numpy(), bias.detach().numpy())

    def combine(self, member_probs):
        stacked = stack_members(member_probs, self.member_names)
        logits = self._features(stacked) @ self.coef + self.intercept
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def to_dict(self):
        return {"kind": self.kind, "members": self.member_names,
                "coef": self.coef.tolist(), "intercept": self.intercept.tolist()}

    @property
    def weight_report(self):
        """Total absolute influence of each member, normalised - a rough 'trust' readout."""
        block = np.abs(self.coef).reshape(len(self.member_names), N_CLASSES, N_CLASSES).sum(axis=(1, 2))
        total = block.sum() or 1.0
        return {n: round(float(v / total), 4) for n, v in zip(self.member_names, block)}


def combiner_from_dict(spec):
    kind = spec["kind"]
    if kind == "average":
        return AverageCombiner(spec["members"])
    if kind == "weighted":
        return WeightedAverageCombiner(spec["members"], spec["weights"])
    if kind == "stacking":
        return StackingCombiner(spec["members"], spec["coef"], spec["intercept"])
    raise ValueError(f"unknown combiner kind {kind!r}")


def fit_all(val_member_probs, y_val, member_names, l2=1.0):
    """Every combiner, fitted on validation. Returns {kind: combiner}."""
    return {
        "average": AverageCombiner(member_names),
        "weighted": WeightedAverageCombiner.fit(val_member_probs, y_val, member_names),
        "stacking": StackingCombiner.fit(val_member_probs, y_val, member_names, l2=l2),
    }


def select_best(combiners, val_member_probs, y_val, metric="macro_f1"):
    """Pick the combiner with the best VALIDATION metric; return (name, combiner, report).

    Ties go to the simpler combiner (average < weighted < stacking), because a
    fitted combiner that only matches plain averaging on validation is carrying
    parameters that can only hurt on test.
    """
    simplicity = {"average": 0, "weighted": 1, "stacking": 2}
    report = {}
    for name, combiner in combiners.items():
        probs = combiner.combine(val_member_probs)
        result = evaluate(probs, y_val, label=f"ensemble:{name} (val)")
        report[name] = {"val_metrics": result, "weights": combiner.weight_report}
    best = min(report, key=lambda n: (-round(report[n]["val_metrics"][metric], 4), simplicity[n]))
    return best, combiners[best], report

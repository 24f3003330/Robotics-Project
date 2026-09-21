"""Probability calibration, fitted on validation logits only.

A fine-tuned HuBERT head is over-confident: it reports 0.95 for windows it gets
right barely 0.7 of the time. That matters more here than usual, because the
ensemble combines PROBABILITIES. Averaging an over-confident member with a
well-behaved one lets the over-confident one win every disagreement regardless
of which is actually right, so calibration is what makes averaging meaningful
rather than arbitrary.

Two scalers, both fitted by L-BFGS on validation NLL:

  temperature  logits / T                 - 1 parameter, cannot change the
                                            model's ranking, so accuracy is
                                            unchanged and only confidence moves.
  vector       logits * w + b (per class)  - 8 parameters, can also correct a
                                            per-class bias (useful for the
                                            neutral-heavy corpora), at a higher
                                            risk of overfitting a small val set.

The method is CHOSEN by cross-validated NLL and only then refitted on the whole
validation split, with "no calibration at all" always in the running.

That indirection is load-bearing. Scoring a calibrator on the same rows it was
fitted to is not a measurement: vector scaling has 2C free parameters, and on a
small validation split it will happily drive in-sample NLL to literally 0.0 by
separating the points it was handed - which is exactly what happened the first
time this was run on a 24-utterance split. Out-of-fold NLL exposes that, so the
small corpora fall back to temperature scaling or to nothing, while the large
ones can still earn the extra parameters.

Fitting on TEST would be cheating; fitting on TRAIN does nothing, because the
model is over-confident precisely on the data it was trained on.
"""

from __future__ import annotations

import numpy as np
import torch

from .labels import CANONICAL_EMOTIONS
from .metrics import expected_calibration_error, negative_log_likelihood


def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _fit(logits, labels, parameters, transform, max_iter=200):
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.LBFGS(parameters, lr=0.1, max_iter=max_iter)

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(transform(x), y)
        loss.backward()
        return loss

    optimiser.step(closure)
    with torch.no_grad():
        return float(loss_fn(transform(x), y))


# On a linearly separable fold, nothing stops the optimiser driving T towards 0
# (infinitely confident, zero training loss). Clamping keeps a diverged fit from
# producing inf/NaN downstream, where it would silently poison the comparison
# instead of simply losing it.
TEMPERATURE_BOUNDS = (0.05, 100.0)


def fit_temperature(logits, labels):
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T = exp(log_t) > 0
    _fit(logits, labels, [log_t], lambda x: x / torch.exp(log_t))
    temperature = float(torch.exp(log_t).item())
    if not np.isfinite(temperature):
        temperature = 1.0
    temperature = float(np.clip(temperature, *TEMPERATURE_BOUNDS))
    return {"kind": "temperature", "temperature": temperature}


def fit_vector_scaling(logits, labels):
    n = np.asarray(logits).shape[1]
    w = torch.ones(n, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    _fit(logits, labels, [w, b], lambda x: x * w + b)
    weight, bias = w.detach().numpy(), b.detach().numpy()
    if not (np.all(np.isfinite(weight)) and np.all(np.isfinite(bias))):
        return {"kind": "none"}          # diverged; caller scores it and moves on
    return {"kind": "vector", "weight": weight.tolist(), "bias": bias.tolist()}


def apply_calibration(logits, calibration):
    """Calibrated probabilities. `calibration=None` means plain softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    if not calibration or calibration.get("kind") in (None, "none", "identity"):
        return softmax_np(logits)
    kind = calibration["kind"]
    if kind == "temperature":
        return softmax_np(logits / max(float(calibration["temperature"]), 1e-6))
    if kind == "vector":
        w = np.asarray(calibration["weight"], dtype=np.float64)
        b = np.asarray(calibration["bias"], dtype=np.float64)
        return softmax_np(logits * w + b)
    raise ValueError(f"unknown calibration kind {kind!r}")


FITTERS = {"temperature": fit_temperature, "vector": fit_vector_scaling}

# Rough sample budget per free parameter. Temperature has 1 parameter, vector
# scaling 2C (=8). Below this a method is not even offered, because the
# cross-validation folds would themselves be too small to mean anything.
_SAMPLES_PER_PARAMETER = 15
_N_PARAMETERS = {"temperature": 1, "vector": 2 * len(CANONICAL_EMOTIONS)}


def _folds(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.concatenate([idx[j::k] for j in range(k) if j != i]), idx[i::k])
            for i in range(k)]


def cross_validated_nll(logits, labels, method, k=5, seed=0):
    """Mean out-of-fold NLL of a calibration method. Lower is better.

    This is the number that decides whether a method is worth using. A method
    that fits its own fold perfectly and generalises badly scores badly here,
    which is the entire point.
    """
    scores = []
    for train_idx, test_idx in _folds(len(labels), k, seed):
        if len(test_idx) == 0 or len(np.unique(labels[train_idx])) < 2:
            continue
        try:
            params = FITTERS[method](logits[train_idx], labels[train_idx])
        except Exception:
            return float("inf")
        probs = apply_calibration(logits[test_idx], params)
        score = negative_log_likelihood(probs, labels[test_idx])
        if not np.isfinite(score):
            return float("inf")          # a fold that blew up disqualifies the method
        scores.append(score)
    if not scores:
        return float("inf")
    mean = float(np.mean(scores))
    return mean if np.isfinite(mean) else float("inf")


def calibrate(val_logits, val_labels, methods=("temperature", "vector"), k=5, seed=0):
    """Choose a calibration method by cross-validated NLL, then refit it on all of val.

    Returns (calibration, report). "none" is always a candidate and wins unless
    a method beats it out of fold, so a calibrator that does not actually help
    is never applied. The report keeps every candidate's CV score next to its
    in-sample score, which is what makes an overfitted fit visible instead of
    persuasive.
    """
    val_logits = np.asarray(val_logits, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=int)
    baseline = softmax_np(val_logits)
    uncalibrated = {"nll": round(negative_log_likelihood(baseline, val_labels), 4),
                    "ece": round(expected_calibration_error(baseline, val_labels), 4)}
    report = {
        "n_val": int(len(val_labels)),
        "classes": list(CANONICAL_EMOTIONS),
        "selection": f"{k}-fold cross-validated NLL on the validation split",
        "uncalibrated": uncalibrated,
        "candidates": {},
    }
    if len(val_labels) < 2 * k:
        report["note"] = (f"only {len(val_labels)} validation utterances - too few for "
                          f"{k}-fold selection, left uncalibrated")
        report["chosen"] = {"kind": "none"}
        return {"kind": "none"}, report

    best, best_cv = {"kind": "none"}, float(negative_log_likelihood(baseline, val_labels))
    report["candidates"]["none"] = {"cv_nll": round(best_cv, 4), **uncalibrated}

    for name in methods:
        needed = _N_PARAMETERS[name] * _SAMPLES_PER_PARAMETER
        if len(val_labels) < needed:
            report["candidates"][name] = {
                "skipped": f"needs about {needed} validation utterances for "
                           f"{_N_PARAMETERS[name]} parameters, have {len(val_labels)}"}
            continue
        cv_nll = cross_validated_nll(val_logits, val_labels, name, k=k, seed=seed)
        try:
            params = FITTERS[name](val_logits, val_labels)
        except Exception as exc:
            report["candidates"][name] = {"error": str(exc)}
            continue
        probs = apply_calibration(val_logits, params)
        report["candidates"][name] = {
            "params": params,
            "cv_nll": round(cv_nll, 4),
            "in_sample_nll": round(negative_log_likelihood(probs, val_labels), 4),
            "in_sample_ece": round(expected_calibration_error(probs, val_labels), 4),
        }
        if cv_nll < best_cv - 1e-4:
            best, best_cv = params, cv_nll

    chosen_probs = apply_calibration(val_logits, best)
    report["chosen"] = best
    report["chosen_cv_nll"] = round(best_cv, 4)
    report["chosen_in_sample"] = {
        "nll": round(negative_log_likelihood(chosen_probs, val_labels), 4),
        "ece": round(expected_calibration_error(chosen_probs, val_labels), 4),
    }
    return best, report

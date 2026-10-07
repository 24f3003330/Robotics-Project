"""
voice_age/model.py - lightweight age-group classifier heads on frozen embeddings.

  LogRegHead  StandardScaler + class-balanced multinomial logistic regression
  MLPHead     StandardScaler + 1 hidden layer MLP (PyTorch), class-weighted loss,
              early stopping on validation macro F1, best checkpoint kept

Both expose fit / predict_proba / state() and are rebuilt from state() at
inference (no pickled code objects). Metrics live here too.
"""

import copy

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"n": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                              "support": int(s[i])} for i, c in enumerate(classes)},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def majority_baseline(y_train, y_eval, classes):
    m = int(np.bincount(y_train, minlength=len(classes)).argmax())
    out = metrics(y_eval, np.full(len(y_eval), m), classes)
    out["predicts"] = classes[m]
    return out


def adjacent_accuracy(y_true, y_pred):
    """Share of predictions at most one age group away (groups are ordered)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------
class _Scaler:
    def fit(self, X):
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


class LogRegHead:
    kind = "logreg"

    def __init__(self, C=1.0, seed=config.RANDOM_SEED):
        self.C, self.seed = C, seed

    def fit(self, X, y, X_val=None, y_val=None, n_classes=None):
        from sklearn.linear_model import LogisticRegression
        self.scaler = _Scaler().fit(X)
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, random_state=self.seed)
        self.clf.fit(self.scaler.transform(X), y)
        self.n_classes = n_classes or int(y.max()) + 1
        return self

    def predict_proba(self, X):
        p = self.clf.predict_proba(self.scaler.transform(X))
        out = np.zeros((len(X), self.n_classes))
        out[:, self.clf.classes_] = p
        return out

    def state(self):
        return {"kind": self.kind, "C": self.C, "mean": self.scaler.mean, "std": self.scaler.std,
                "coef": self.clf.coef_, "intercept": self.clf.intercept_, "classes": self.clf.classes_,
                "n_classes": self.n_classes}

    @staticmethod
    def from_state(st):
        h = _LinearState(st)
        return h


class _LinearState:
    """Logistic-regression head rebuilt from saved arrays (inference only)."""

    def __init__(self, st):
        self.st = st

    def predict_proba(self, X):
        z = ((X - self.st["mean"]) / self.st["std"]) @ np.asarray(self.st["coef"]).T + self.st["intercept"]
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        out = np.zeros((len(X), self.st["n_classes"]))
        out[:, np.asarray(self.st["classes"])] = p
        return out


class MLPHead:
    kind = "mlp"

    def __init__(self, hidden=256, dropout=0.3, lr=1e-3, weight_decay=1e-3, epochs=60, batch_size=64,
                 patience=8, seed=config.RANDOM_SEED, log=None):
        self.hp = dict(hidden=hidden, dropout=dropout, lr=lr, weight_decay=weight_decay, epochs=epochs,
                       batch_size=batch_size, patience=patience)
        self.seed = seed
        self.log = log or (lambda *_: None)

    def _net(self, d_in, n_out):
        import torch.nn as nn
        return nn.Sequential(nn.Linear(d_in, self.hp["hidden"]), nn.ReLU(), nn.Dropout(self.hp["dropout"]),
                             nn.Linear(self.hp["hidden"], n_out))

    def fit(self, X, y, X_val, y_val, n_classes):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.scaler = _Scaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        Xv = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32)
        counts = np.bincount(y, minlength=n_classes).astype(float)
        weights = torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
        self.net = self._net(X.shape[1], n_classes)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        best, best_state, bad, self.history = -1.0, None, 0, []
        for epoch in range(self.hp["epochs"]):
            self.net.train()
            perm = rng.permutation(len(y))
            for i in range(0, len(y), self.hp["batch_size"]):
                b = perm[i:i + self.hp["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                pred = self.net(Xv).argmax(1).numpy()
            f1 = metrics(y_val, pred, list(range(n_classes)))["macro_f1"]
            self.history.append({"epoch": epoch + 1, "val_macro_f1": f1})
            if f1 > best + 1e-4:
                best, best_state, bad, self.best_epoch = f1, copy.deepcopy(self.net.state_dict()), 0, epoch + 1
            else:
                bad += 1
                if bad >= self.hp["patience"]:
                    break
        self.net.load_state_dict(best_state)            # checkpoint with the best validation macro F1
        self.net.eval()
        self.n_classes = n_classes
        self.log(f"      mlp: best epoch {self.best_epoch} of {len(self.history)} (val macro F1 {best:.3f})")
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            z = self.net(torch.tensor(self.scaler.transform(X), dtype=torch.float32))
            return torch.softmax(z, 1).numpy()

    def state(self):
        return {"kind": self.kind, "hp": self.hp, "mean": self.scaler.mean, "std": self.scaler.std,
                "weights": {k: v.cpu().numpy() for k, v in self.net.state_dict().items()},
                "n_classes": self.n_classes, "best_epoch": self.best_epoch}

    @staticmethod
    def from_state(st, d_in):
        import torch
        h = MLPHead(**{k: st["hp"][k] for k in ("hidden", "dropout")})
        h.scaler = _Scaler()
        h.scaler.mean, h.scaler.std = st["mean"], st["std"]
        h.net = h._net(d_in, st["n_classes"])
        h.net.load_state_dict({k: torch.tensor(v) for k, v in st["weights"].items()})
        h.net.eval()
        h.n_classes = st["n_classes"]
        return h


def head_from_state(st, d_in):
    if st["kind"] == "logreg":
        return LogRegHead.from_state(st)
    if st["kind"] == "mlp":
        return MLPHead.from_state(st, d_in)
    raise ValueError(f"unknown head kind {st['kind']!r}")


def count_parameters(st):
    if st["kind"] == "logreg":
        return int(np.asarray(st["coef"]).size + np.asarray(st["intercept"]).size)
    return int(sum(np.asarray(v).size for v in st["weights"].values()))

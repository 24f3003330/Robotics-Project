"""Metrics, in numpy so the evaluation path has no scikit-learn dependency.

Reported for every single-dataset model and for every ensemble combiner:
accuracy, macro-F1, per-class precision/recall/F1/support, and the confusion
matrix. Macro-F1 is the headline number rather than accuracy - these corpora
are class-imbalanced (MSP-Podcast is dominated by neutral), so accuracy alone
rewards a model that mostly predicts the majority class.
"""

from __future__ import annotations

import numpy as np

from .labels import CANONICAL_EMOTIONS


def confusion_matrix(y_true, y_pred, n_classes=len(CANONICAL_EMOTIONS)):
    """cm[i][j] = number of true-class-i samples predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        cm[t, p] += 1
    return cm


def per_class_prf(cm):
    """Precision / recall / F1 / support per class, from the confusion matrix."""
    out = {}
    for i, name in enumerate(CANONICAL_EMOTIONS):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4), "support": int(cm[i, :].sum())}
    return out


def expected_calibration_error(probs, y_true, n_bins=15):
    """ECE: mean |confidence - accuracy| over equal-width confidence bins.

    This is the number calibration is judged on. An ensemble that averages
    probabilities is only meaningful if those probabilities mean the same thing
    across members, so each member's ECE is reported before and after scaling.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    if probs.size == 0:
        return 0.0
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def negative_log_likelihood(probs, y_true, eps=1e-12):
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    if probs.size == 0:
        return 0.0
    return float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)).mean())


def evaluate(probs, y_true, label=""):
    """Full metric bundle for one model or combiner."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    per_class = per_class_prf(cm)
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
    accuracy = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    # Unweighted average recall = "balanced accuracy", the standard SER headline.
    uar = float(np.mean([v["recall"] for v in per_class.values()]))
    return {
        "label": label,
        "n": int(len(y_true)),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "uar": round(uar, 4),
        "nll": round(negative_log_likelihood(probs, y_true), 4),
        "ece": round(expected_calibration_error(probs, y_true), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": list(CANONICAL_EMOTIONS),
    }


def format_report(result):
    """The text block printed for each model and written into the comparison report."""
    lines = []
    head = result.get("label") or "model"
    lines.append(f"{head}  (n={result['n']})")
    lines.append(f"  accuracy {result['accuracy']:.4f}   macro-F1 {result['macro_f1']:.4f}   "
                 f"UAR {result['uar']:.4f}   NLL {result['nll']:.4f}   ECE {result['ece']:.4f}")
    lines.append(f"  {'class':<9}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for name, v in result["per_class"].items():
        lines.append(f"  {name:<9}{v['precision']:>8.4f}{v['recall']:>8.4f}"
                     f"{v['f1']:>8.4f}{v['support']:>9d}")
    cm = np.asarray(result["confusion_matrix"])
    width = max(6, max(len(c) for c in result["classes"]) + 2)
    lines.append("  confusion matrix (rows = true, cols = predicted)")
    lines.append("  " + " " * 10 + "".join(f"{c:>{width}}" for c in result["classes"]))
    for i, c in enumerate(result["classes"]):
        lines.append("  " + f"{c:<10}" + "".join(f"{v:>{width}d}" for v in cm[i]))
    return "\n".join(lines)


def compare_table(results):
    """One row per model/combiner, sorted by macro-F1 - the headline comparison."""
    rows = sorted(results, key=lambda r: -r["macro_f1"])
    width = max([len(r["label"]) for r in rows] + [10])
    lines = [f"{'model':<{width}}{'acc':>9}{'macroF1':>9}{'UAR':>9}{'NLL':>9}{'ECE':>9}{'n':>8}"]
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append(f"{r['label']:<{width}}{r['accuracy']:>9.4f}{r['macro_f1']:>9.4f}"
                     f"{r['uar']:>9.4f}{r['nll']:>9.4f}{r['ece']:>9.4f}{r['n']:>8d}")
    return "\n".join(lines)

"""
python -m body_vitals.evaluate

Re-evaluates the saved model (models/body_vitals_model.pkl) on the held-out TEST
SUBJECTS listed in its metadata and (re)writes body_vitals/results/:
    metrics.json, classification_report.txt, confusion_matrix.png

All numbers are window-level metrics over windows of subjects the model never saw
during training or model selection.
"""

import argparse
import json
import os

import numpy as np

from . import config
from .dataset import load_features
from .model import compute_metrics, load_artifacts, majority_baseline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", help="feature CSV (default: the table for the model's window)")
    p.add_argument("--model", default=config.MODEL_PATH)
    p.add_argument("--metadata", default=config.METADATA_PATH)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    return p.parse_args(argv)


def classification_report_text(y_true, y_pred, title):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, labels=list(range(len(config.CLASSES))),
                                   target_names=[config.CLASS_DISPLAY[c] for c in config.CLASSES],
                                   digits=3, zero_division=0)
    return f"{title}\n\n{report}"


def plot_confusion_matrix(cm, path, title):
    """Counts with row-normalised percentages; one sequential hue (light = few, dark = many)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[evaluate] matplotlib unavailable ({exc}); confusion_matrix.png not written")
        return None
    cm = np.asarray(cm)
    row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [config.CLASS_DISPLAY[c] for c in config.CLASSES]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150)
    ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{row[i, j]:.0%}", ha="center", va="center", fontsize=11,
                    color="white" if row[i, j] > 0.55 else "#1f2937")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (WESAD condition)")
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_results(results_dir, metrics, report_text, cm, title):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as fh:
        fh.write(report_text)
    plot_confusion_matrix(cm, os.path.join(results_dir, "confusion_matrix.png"), title)


def run(args):
    from .train import prepare
    model, meta = load_artifacts(args.model, args.metadata)
    names = meta["feature_names"]
    if names != config.FEATURE_SETS.get(meta.get("feature_version")):
        raise SystemExit("Saved model's feature list is not a registered feature set; retrain it.")
    spec = {"window_sec": meta["window_size_sec"], "step_sec": meta["window_step_sec"],
            "feature_set": meta["feature_version"], "normalization": meta.get("normalization", "none")}
    path = args.features or config.features_csv(spec["window_sec"], spec["step_sec"])
    df = prepare(load_features(path, names), spec)
    test, fit_subjects = meta["test_subjects"], meta["final_fit_subjects"]
    part = df[df["subject"].isin(test)]
    y = part["label"].to_numpy()
    pred = np.asarray(model.predict_proba(part[names])).argmax(axis=1)
    metrics = compute_metrics(y, pred)
    y_fit = df[df["subject"].isin(fit_subjects)]["label"].to_numpy()
    out = {"evaluation": "held-out test subjects (window-level metrics, subject-wise split)",
           "model_type": meta["model_type"], "test_subjects": test, "test": metrics,
           "majority_class_baseline": majority_baseline(y_fit, y)}
    if os.path.isfile(os.path.join(args.results_dir, "metrics.json")):
        with open(os.path.join(args.results_dir, "metrics.json")) as fh:
            previous = json.load(fh)
        out = dict(previous, **out)        # keep training-time sections (LOSO, comparison, ...)
    title = (f"{meta['model_type']} ({meta['feature_version'][-2:]}, {spec['window_sec']} s"
             f"{', rest-calibrated' if spec['normalization'] == 'rest_calibration' else ''})\n"
             f"test subjects {', '.join(test)} - {metrics['n_windows']} windows unseen in training")
    report = classification_report_text(y, pred, title) + _summary_lines(metrics, out["majority_class_baseline"])
    write_results(args.results_dir, out, report, metrics["confusion_matrix"], title)
    print(report)
    print(f"Wrote {args.results_dir}/metrics.json, classification_report.txt, confusion_matrix.png")
    return out


def _summary_lines(m, base):
    return (f"\nbalanced accuracy {m['balanced_accuracy']:.3f}   macro F1 {m['macro_f1']:.3f}\n"
            f"majority-class baseline (always '{base['predicts']}'): accuracy {base['accuracy']:.3f}, "
            f"balanced accuracy {base['balanced_accuracy']:.3f}, macro F1 {base['macro_f1']:.3f}\n")


if __name__ == "__main__":
    run(parse_args())

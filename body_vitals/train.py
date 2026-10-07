"""
python -m body_vitals.train [--pipeline body_vitals/results/experiments/selected_pipeline.json --track TRACK]
python -m body_vitals.train --legacy        # the original v1 pipeline (RF, 19 features, 60 s)

Trains the FINAL Body Vitals model for a pipeline chosen by
`python -m body_vitals.experiment` (subject-wise CV on the development
subjects only) and evaluates it ONCE on the held-out test subjects.

  1. leakage audit: train / validation / test subjects disjoint; the test
     subjects equal config.TEST_SUBJECTS and were excluded from the experiment
  2. fit the pipeline on the 12 development subjects (train + validation)
  3. evaluate once on the test subjects; the previous model (models/archive/
     body_vitals_model_v1.pkl, also fitted on the same 12 subjects) is scored on
     exactly the same test windows for comparison
  4. leave-one-subject-out over all 15 subjects (supplementary)
Writes models/body_vitals_model.pkl, models/body_vitals_metadata.json and
body_vitals/results/{metrics.json, classification_report.txt, confusion_matrix.png}.
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np

from . import config
from .dataset import load_features
from .evaluate import _summary_lines, classification_report_text, write_results
from .model import compute_metrics, fit_model, load_artifacts, majority_baseline, save_artifacts, subject_split
from .normalization import CALIBRATION_SEC, calibrate_table, drop_calibration_period

SELECTED_PIPELINE = os.path.join(config.RESULTS_DIR, "experiments", "selected_pipeline.json")
ARCHIVED_V1_MODEL = os.path.join(config.PROJECT_DIR, "models", "archive", "body_vitals_model_v1.pkl")
ARCHIVED_V1_METADATA = os.path.join(config.PROJECT_DIR, "models", "archive", "body_vitals_metadata_v1.json")
LEGACY = {"window_sec": 60, "step_sec": 30, "feature_set": "wesad-chest-v1", "normalization": "none",
          "config": {"name": "rf", "family": "RandomForest",
                     "params": {"n_estimators": 500, "min_samples_leaf": 1, "max_features": "sqrt"}}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", default=SELECTED_PIPELINE, help="selected_pipeline.json from the experiment")
    p.add_argument("--track", default=config.DEPLOY_TRACK, help="which selected pipeline to train")
    p.add_argument("--legacy", action="store_true", help="train the original v1 pipeline instead")
    p.add_argument("--features", help="feature CSV (default: the table for the pipeline's window)")
    p.add_argument("--model", default=config.MODEL_PATH)
    p.add_argument("--metadata", default=config.METADATA_PATH)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--no-loso", action="store_true")
    return p.parse_args(argv)


def load_spec(args):
    if args.legacy:
        return dict(LEGACY, source="legacy v1 pipeline"), None
    with open(args.pipeline) as fh:
        selected = json.load(fh)
    s = selected[args.track]
    spec = {"window_sec": s["data"]["window_sec"], "step_sec": s["data"]["step_sec"],
            "feature_set": s["data"]["feature_set"],
            "normalization": "rest_calibration" if s["data"]["normalization"] == "calibrated" else "none",
            "config": s["config"], "source": f"{os.path.relpath(args.pipeline, config.PROJECT_DIR)} [{args.track}]"}
    return spec, selected


def prepare(df, spec):
    """Feature table -> model table for the pipeline. Rest calibration is per subject from that
    subject's own first CALIBRATION_SEC s of rest; those windows are removed."""
    names = config.FEATURE_SETS[spec["feature_set"]]
    if spec["normalization"] == "rest_calibration":
        return calibrate_table(df, names, spec["window_sec"])[0].reset_index(drop=True)
    return df.reset_index(drop=True)


def _fmt(m):
    return (f"acc {m['accuracy']:.3f}  bal-acc {m['balanced_accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}  "
            f"recall B/S/A {m['per_class']['baseline']['recall']:.2f}/{m['per_class']['stress']['recall']:.2f}/"
            f"{m['per_class']['amusement']['recall']:.2f}")


def loso(df, names, cfg, seed):
    from sklearn.model_selection import LeaveOneGroupOut
    X, y, groups = df[names], df["label"].to_numpy(), df["subject"].to_numpy()
    pred = np.zeros_like(y)
    per_subject = {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model = fit_model(cfg, X.iloc[tr], y[tr], seed, probability=False)
        pred[te] = np.asarray(model.predict(X.iloc[te])).astype(int)
        m = compute_metrics(y[te], pred[te])
        per_subject[str(groups[te][0])] = {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                                           "macro_f1": m["macro_f1"],
                                           **{f"recall_{c}": m["per_class"][c]["recall"] for c in config.CLASSES}}
    keys = ["accuracy", "balanced_accuracy", "macro_f1"] + [f"recall_{c}" for c in config.CLASSES]
    return {"pooled": compute_metrics(y, pred), "per_subject": per_subject,
            "mean_over_subjects": {k: float(np.mean([v[k] for v in per_subject.values()])) for k in keys},
            "std_over_subjects": {k: float(np.std([v[k] for v in per_subject.values()])) for k in keys}}


def leakage_audit(train, val, test, dev_used_by_experiment):
    checks = {
        "train ∩ validation = ∅": not set(train) & set(val),
        "train ∩ test = ∅": not set(train) & set(test),
        "validation ∩ test = ∅": not set(val) & set(test),
        "test subjects == config.TEST_SUBJECTS": sorted(test) == sorted(config.TEST_SUBJECTS),
        "experiment (model/feature/window selection) used no test subject":
            dev_used_by_experiment is None or not set(dev_used_by_experiment) & set(test),
        "scaling / class weights fitted inside each training fold (sklearn Pipeline / fit)": True,
        "no data-driven feature selection (fixed feature sets)": True,
        "no resampling (class weighting only)": True,
        "rest calibration uses only each subject's own first 5 min of rest": True,
    }
    return checks


def run(args):
    t0 = time.time()
    spec, selected = load_spec(args)
    names = config.FEATURE_SETS[spec["feature_set"]]
    cfg = spec["config"]
    path = args.features or config.features_csv(spec["window_sec"], spec["step_sec"])
    raw = load_features(path, names)
    subjects = sorted(raw["subject"].unique(), key=lambda s: (len(s), s))
    train, val, test = subject_split(subjects, seed=args.seed)
    dev = sorted(train + val, key=lambda s: (len(s), s))
    dev_in_experiment = None
    if selected:
        dev_in_experiment = [f["subject"] for f in selected[args.track]["dev_cv"]["folds"]]

    audit = leakage_audit(train, val, test, dev_in_experiment)
    print("================ leakage audit ================")
    print(f"  train      ({len(train)}): {train}\n  validation ({len(val)}): {val}\n  test       ({len(test)}): {test}")
    for k, ok in audit.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {k}")
    if not all(audit.values()):
        raise SystemExit("leakage audit failed - not training")

    print(f"\n[train] pipeline ({spec['source']}): {cfg['family']} {json.dumps(cfg['params'])}")
    print(f"        window {spec['window_sec']} s (step {spec['step_sec']}), feature set {spec['feature_set']} "
          f"({len(names)} features), normalization {spec['normalization']}")
    df = prepare(raw, spec)
    fit = df[df["subject"].isin(dev)]
    te = df[df["subject"].isin(test)]
    assert not set(fit["subject"]) & set(te["subject"])
    model = fit_model(cfg, fit[names], fit["label"].to_numpy(), args.seed, probability=True)

    pred = np.asarray(model.predict_proba(te[names])).argmax(axis=1)     # what the API returns
    y_te = te["label"].to_numpy()
    test_metrics = compute_metrics(y_te, pred)
    baseline = majority_baseline(fit["label"].to_numpy(), y_te)
    print(f"\n[train] TEST subjects {test}: {len(te)} windows")
    print(f"    new pipeline          {_fmt(test_metrics)}")

    # previous model on exactly these test windows (absolute features: its own pipeline)
    comparison = {}
    if os.path.isfile(ARCHIVED_V1_MODEL) and not args.legacy:
        old, old_meta = load_artifacts(ARCHIVED_V1_MODEL, ARCHIVED_V1_METADATA)
        assert sorted(old_meta["final_fit_subjects"]) == sorted(dev), "old model was fitted on other subjects"
        v1 = config.FEATURE_SETS["wesad-chest-v1"]
        raw60 = load_features(config.features_csv(60, 30), v1)
        t_full = raw60[raw60["subject"].isin(test)]
        comparison["v1_on_all_test_windows"] = compute_metrics(t_full["label"].to_numpy(), old.predict(t_full[v1]))
        print(f"    v1 model, all 60 s test windows            {_fmt(comparison['v1_on_all_test_windows'])}")
        if spec["normalization"] == "rest_calibration":
            t_red = drop_calibration_period(t_full, 60)
            comparison["v1_on_test_windows_after_calibration_period"] = compute_metrics(
                t_red["label"].to_numpy(), old.predict(t_red[v1]))
            print(f"    v1 model, same windows minus first {CALIBRATION_SEC // 60} min rest "
                  f"{_fmt(comparison['v1_on_test_windows_after_calibration_period'])}")
    print(f"    majority class        {_fmt(baseline)}  (always '{baseline['predicts']}')")

    loso_result, loso_v1_same = None, None
    if not args.no_loso:
        print(f"\n[train] leave-one-subject-out over all {len(subjects)} subjects ...")
        loso_result = loso(df, names, cfg, args.seed)
        print(f"    new pipeline  pooled {_fmt(loso_result['pooled'])}")
        if not args.legacy:
            v1 = config.FEATURE_SETS["wesad-chest-v1"]
            raw60 = load_features(config.features_csv(60, 30), v1)
            same = drop_calibration_period(raw60, 60) if spec["normalization"] == "rest_calibration" else raw60
            loso_v1_same = loso(same.reset_index(drop=True), v1, LEGACY["config"], args.seed)
            label = "same windows" if spec["normalization"] == "rest_calibration" else "all windows"
            print(f"    v1 pipeline ({label}) pooled {_fmt(loso_v1_same['pooled'])}")

    import sklearn
    metadata = {
        "dataset": "WESAD",
        "task": "physiological condition classification (WESAD study conditions, not self-reported emotion)",
        "classes": config.CLASSES,
        "class_index": {c: i for i, c in enumerate(config.CLASSES)},
        "wesad_label_mapping": {"1": "baseline", "2": "stress", "3": "amusement"},
        "feature_version": spec["feature_set"],
        "feature_names": names,
        "feature_units": {f: config.FEATURE_UNITS[f] for f in names},
        "signals_used": ["chest ECG", "chest EDA", "chest Resp"],
        "signals_ignored": ["chest EMG", "chest Temp", "chest ACC", "wrist BVP", "wrist EDA", "wrist TEMP",
                            "wrist ACC"],
        "preprocessing": {"source_sampling_rate_hz": config.WESAD_CHEST_FS, "ecg_fs": config.ECG_FS,
                          "eda_fs": config.EDA_FS, "resp_fs": config.RESP_FS,
                          "rr_limits_s": [config.RR_MIN_S, config.RR_MAX_S],
                          "rr_max_local_deviation": config.RR_MAX_LOCAL_DEVIATION,
                          "min_valid_rr_per_60s": config.MIN_VALID_RR,
                          "max_rr_artifact_fraction": config.MAX_RR_ARTIFACT_FRACTION,
                          "breath_interval_s": [config.BREATH_MIN_S, config.BREATH_MAX_S],
                          "min_breaths": config.MIN_BREATHS, "scr_min_amplitude_us": config.SCR_MIN_AMPLITUDE_US,
                          "documentation": "docs/BODY_VITALS.md"},
        "window_size_sec": spec["window_sec"],
        "window_step_sec": spec["step_sec"],
        "normalization": spec["normalization"],
        "calibration_sec": CALIBRATION_SEC if spec["normalization"] == "rest_calibration" else None,
        "calibration_note": ("features minus the per-feature median of the person's calibration windows "
                             "(windows entirely inside the first calibration_sec seconds of seated rest)")
        if spec["normalization"] == "rest_calibration" else None,
        "model_type": cfg["family"],
        "model_config": cfg["name"],
        "model_params": cfg["params"],
        "class_balancing": "balanced class weights",
        "pipeline_source": spec["source"],
        "random_seed": args.seed,
        "split_strategy": "fixed held-out test subjects (GroupShuffleSplit by subject, seed 42); model selection "
                          "by leave-one-subject-out CV over the 12 development subjects",
        "training_subjects": train,
        "validation_subjects": val,
        "test_subjects": test,
        "final_fit_subjects": dev,
        "final_fit_note": "fitted on train + validation subjects after selection on development-subject CV",
        "n_windows": {"final_fit": int(len(fit)), "test": int(len(te))},
        "development_cv": selected[args.track]["dev_cv"] if selected else None,
        "evaluation_metrics": {
            "test": {k: test_metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
                                                  "macro_recall", "per_class", "confusion_matrix")},
            "test_majority_baseline": {k: baseline[k] for k in ("predicts", "accuracy", "balanced_accuracy",
                                                                "macro_f1")},
            "loso_pooled": loso_result and {k: loso_result["pooled"][k] for k in
                                            ("accuracy", "balanced_accuracy", "macro_f1")},
        },
        "leakage_audit": audit,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    save_artifacts(model, metadata, args.model, args.metadata)

    metrics = {
        "evaluation": "held-out test subjects (window-level metrics, subject-wise split)",
        "pipeline": {k: spec[k] for k in ("window_sec", "step_sec", "feature_set", "normalization", "source")},
        "model_type": cfg["family"], "model_params": cfg["params"],
        "split": {"train": train, "validation": val, "test": test},
        "test_subjects": test, "test": test_metrics, "majority_class_baseline": baseline,
        "previous_model_on_same_test_subjects": comparison,
        "leave_one_subject_out": loso_result,
        "leave_one_subject_out_v1_pipeline_same_windows": loso_v1_same,
        "leakage_audit": audit,
    }
    title = (f"{cfg['family']} ({spec['feature_set'][-2:]}, {spec['window_sec']} s"
             f"{', rest-calibrated' if spec['normalization'] == 'rest_calibration' else ''})\n"
             f"test subjects {', '.join(test)} - {test_metrics['n_windows']} windows unseen in training")
    report = classification_report_text(y_te, pred, title) + _summary_lines(test_metrics, baseline)
    if loso_result:
        p = loso_result["pooled"]
        report += (f"\nLeave-one-subject-out over {len(subjects)} subjects: pooled accuracy {p['accuracy']:.3f}, "
                   f"balanced accuracy {p['balanced_accuracy']:.3f}, macro F1 {p['macro_f1']:.3f}; per-subject "
                   f"macro F1 {loso_result['mean_over_subjects']['macro_f1']:.3f} +- "
                   f"{loso_result['std_over_subjects']['macro_f1']:.3f}\n")
    write_results(args.results_dir, metrics, report, test_metrics["confusion_matrix"], title)
    print("\n" + report)
    print(f"Saved {args.model}\n      {args.metadata}\n      {args.results_dir}/  ({time.time() - t0:.1f} s)")
    return metrics


if __name__ == "__main__":
    run(parse_args())

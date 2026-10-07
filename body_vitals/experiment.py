"""
python -m body_vitals.experiment

Model / feature / windowing / normalisation study for the Body Vitals model,
run on the 12 DEVELOPMENT subjects only. The held-out test subjects
(config.TEST_SUBJECTS) are removed from every table as soon as it is loaded and
are never seen here - the final, single test evaluation is done by
`python -m body_vitals.train --pipeline body_vitals/results/experiments/selected_pipeline.json`.

Every score is subject-wise cross-validation: leave-one-subject-out over the
development subjects (12 folds; GroupKFold with one subject per fold). Each fold
fits on 11 subjects and scores the 12th, so no subject's windows are ever in
both the training and the scoring data. Scaling (LR, SVM) and class weighting
are fitted inside each training fold. There is no data-driven feature
selection; feature SETS are fixed lists compared as whole pipelines.

Stages
  A  window {30, 60, 90 s} x feature set {v1, v2} x normalisation {none, rest
     calibration} x 5 model families with default settings
  B  small hyper-parameter grids per family on the best data configuration,
     plus a class-weighting check
  C  selection by mean macro F1 over folds (ties: balanced accuracy); amusement
     analysis; the fixed train->validation split as a secondary check

Writes body_vitals/results/experiments/{stage_a.json, stage_b.json,
selected_pipeline.json, amusement_analysis.json, summary.txt, cv_confusion_*.png}.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import config
from .model import compute_metrics, default_configs, fit_model, xgboost_available
from .normalization import CALIBRATION_SEC, calibrate_table, drop_calibration_period

OUT_DIR = os.path.join(config.RESULTS_DIR, "experiments")
WINDOWS = [(30, 15), (60, 30), (90, 45)]
VARIANTS = [("full", "none"), ("cal", "none"), ("cal", "calibrated")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--quick", action="store_true", help="fewer trees / configs (smoke test)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_dev(window, step):
    df = pd.read_csv(config.features_csv(window, step))
    df = df[~df["subject"].isin(config.TEST_SUBJECTS)].reset_index(drop=True)
    assert not set(df["subject"]) & set(config.TEST_SUBJECTS), "test subject in development data"
    return df


def make_variant(df, window, protocol, norm, feature_set):
    names = config.FEATURE_SETS[feature_set]
    if protocol == "full":
        assert norm == "none"
        return df.reset_index(drop=True)
    if norm == "none":
        return drop_calibration_period(df, window).reset_index(drop=True)
    out, _ = calibrate_table(df, names, window)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# subject-wise CV
# ---------------------------------------------------------------------------
def loso_cv(df, feature_set, cfg, seed):
    """Leave-one-subject-out over the subjects in df. Returns fold metrics + pooled predictions."""
    from sklearn.model_selection import LeaveOneGroupOut
    names = config.FEATURE_SETS[feature_set]
    X, y, groups = df[names], df["label"].to_numpy(), df["subject"].to_numpy()
    pred = np.full(len(y), -1)
    folds = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        assert not set(groups[tr]) & set(groups[te])
        model = fit_model(cfg, X.iloc[tr], y[tr], seed, probability=False)
        pred[te] = np.asarray(model.predict(X.iloc[te])).astype(int)
        m = compute_metrics(y[te], pred[te])
        folds.append({"subject": str(groups[te][0]), "accuracy": m["accuracy"],
                      "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"],
                      **{f"recall_{c}": m["per_class"][c]["recall"] for c in config.CLASSES}})
    return summarise(folds, y, pred), pred


def summarise(folds, y, pred):
    keys = ["accuracy", "balanced_accuracy", "macro_f1"] + [f"recall_{c}" for c in config.CLASSES]
    out = {k: float(np.mean([f[k] for f in folds])) for k in keys}
    out.update({f"{k}_std": float(np.std([f[k] for f in folds])) for k in keys})
    out["pooled"] = compute_metrics(y, pred)
    out["folds"] = folds
    return out


def fmt(r):
    return (f"F1 {r['macro_f1']:.3f}±{r['macro_f1_std']:.3f}  bal {r['balanced_accuracy']:.3f}  "
            f"acc {r['accuracy']:.3f}  rec B/S/A {r['recall_baseline']:.2f}/{r['recall_stress']:.2f}/"
            f"{r['recall_amusement']:.2f}")


def key(r):
    return (round(r["macro_f1"], 6), round(r["balanced_accuracy"], 6))


# ---------------------------------------------------------------------------
# hyper-parameter grids (stage B) - deliberately small: 12 subjects cannot
# support a large search without overfitting the CV itself
# ---------------------------------------------------------------------------
def grids(quick=False):
    trees = 200 if quick else 500
    g = {
        "RandomForest": [{"n_estimators": trees, "min_samples_leaf": l, "max_features": mf, "max_depth": d}
                         for l in (1, 3, 8) for mf in ("sqrt", 0.5) for d in (None, 10)],
        "HistGradientBoosting": [{"max_iter": it, "learning_rate": lr, "max_leaf_nodes": ln,
                                  "l2_regularization": 1.0} for it in (150, 300) for lr in (0.03, 0.1)
                                 for ln in (7, 15)],
        "LogisticRegression": [{"C": c} for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)],
        "SVM": [{"C": c, "gamma": gm} for c in (0.3, 1.0, 3.0, 10.0) for gm in ("scale", 0.01)],
    }
    if xgboost_available():
        g["XGBoost"] = [{"n_estimators": n, "max_depth": d, "learning_rate": 0.05, "subsample": 0.8,
                         "colsample_bytree": 0.8, "min_child_weight": mcw}
                        for n in (200, 400) for d in (2, 3, 5) for mcw in (1, 5)]
    if quick:
        g = {k: v[:2] for k, v in g.items()}
    return g


def plot_cv_confusion(cm, path, title):
    from .evaluate import plot_confusion_matrix
    plot_confusion_matrix(cm, path, title)


# ---------------------------------------------------------------------------
def run(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    defaults = default_configs()
    if args.quick:
        for c in defaults:
            if "n_estimators" in c["params"]:
                c["params"]["n_estimators"] = 100
    dev_subjects = sorted(load_dev(60, 30)["subject"].unique(), key=lambda s: (len(s), s))
    print(f"[experiment] development subjects ({len(dev_subjects)}): {dev_subjects}")
    print(f"[experiment] held-out test subjects {config.TEST_SUBJECTS} are excluded from everything below")
    print("[experiment] protocol: leave-one-subject-out CV over the development subjects\n")

    # ---------------- stage A
    stage_a = []
    data_cache = {}
    for window, step in WINDOWS:
        df = load_dev(window, step)
        for protocol, norm in VARIANTS:
            for fset in config.FEATURE_SETS:
                d = make_variant(df, window, protocol, norm, fset)
                data_cache[(window, protocol, norm, fset)] = d
                for cfg in defaults:
                    r, _ = loso_cv(d, fset, cfg, args.seed)
                    row = {"window_sec": window, "step_sec": step, "protocol": protocol, "normalization": norm,
                           "feature_set": fset, "config": cfg, "n_windows": int(len(d)),
                           **{k: v for k, v in r.items() if k not in ("pooled", "folds")},
                           "folds": r["folds"]}
                    stage_a.append(row)
                    print(f"  A w{window:<3} {protocol:<4} {norm:<10} {fset[-2:]} {cfg['family']:<21} {fmt(r)}",
                          flush=True)
    with open(os.path.join(args.out, "stage_a.json"), "w") as fh:
        json.dump(stage_a, fh, indent=1)

    # reference = the current pipeline (v1, 60 s, no normalisation, RF defaults) in the same CV
    reference = next(r for r in stage_a if r["window_sec"] == 60 and r["protocol"] == "full"
                     and r["feature_set"] == "wesad-chest-v1" and r["config"]["family"] == "RandomForest")

    # Choice of data configuration. Calibrated and uncalibrated pipelines are compared on the
    # SAME windows ("cal" protocol); the "full" protocol only serves as the comparison with the
    # existing model. For each (window, protocol, normalisation, feature set) take the best family.
    def best_of(pred):
        rows = [r for r in stage_a if pred(r)]
        return max(rows, key=key) if rows else None

    summary = {"reference_current_pipeline_dev_cv": _slim(reference)}
    best_uncal = best_of(lambda r: r["normalization"] == "none" and r["protocol"] == "full")
    best_cal_none = best_of(lambda r: r["protocol"] == "cal" and r["normalization"] == "none")
    best_cal = best_of(lambda r: r["normalization"] == "calibrated")
    print(f"\n[A] best uncalibrated (all windows):  {_desc(best_uncal)}  {fmt(best_uncal)}")
    print(f"[A] best uncalibrated (cal. windows): {_desc(best_cal_none)}  {fmt(best_cal_none)}")
    print(f"[A] best rest-calibrated:             {_desc(best_cal)}  {fmt(best_cal)}")
    print(f"[A] reference (current pipeline):     {_desc(reference)}  {fmt(reference)}")

    # ---------------- stage B on both tracks: deployable-without-calibration and calibrated
    stage_b = {}
    tracks = {"uncalibrated": best_uncal, "calibrated": best_cal}
    for track, base in tracks.items():
        window, fset = base["window_sec"], base["feature_set"]
        d = data_cache[(window, base["protocol"], base["normalization"], fset)]
        rows = []
        for fam, grid in grids(args.quick).items():
            for i, params in enumerate(grid):
                cfg = {"name": f"{fam}-{i}", "family": fam, "params": params}
                r, _ = loso_cv(d, fset, cfg, args.seed)
                rows.append({"config": cfg, **{k: v for k, v in r.items() if k not in ("pooled", "folds")},
                             "folds": r["folds"]})
                print(f"  B {track:<12} {fam:<21} {json.dumps(params):<110} {fmt(r)}", flush=True)
        best = max(rows, key=key)
        # class-imbalance check on the winning configuration: no class weighting
        cfg_nw = {"name": best["config"]["name"] + "-unweighted", "family": best["config"]["family"],
                  "params": dict(best["config"]["params"], class_weight=None)}
        r_nw, _ = loso_cv(d, fset, cfg_nw, args.seed)
        print(f"  B {track:<12} class_weight=None for the best config:  {fmt(r_nw)}")
        stage_b[track] = {"data": {"window_sec": window, "step_sec": base["step_sec"], "feature_set": fset,
                                   "protocol": base["protocol"], "normalization": base["normalization"]},
                          "rows": rows, "best": _slim(best),
                          "best_without_class_weight": {k: v for k, v in r_nw.items() if k not in ("pooled",)}}
    with open(os.path.join(args.out, "stage_b.json"), "w") as fh:
        json.dump(stage_b, fh, indent=1)

    # ---------------- stage C: selection + amusement analysis
    selected = {}
    for track in stage_b:
        sb = stage_b[track]
        best_cfg = sb["best"]["config"]
        d = data_cache[(sb["data"]["window_sec"], sb["data"]["protocol"], sb["data"]["normalization"],
                        sb["data"]["feature_set"])]
        r, pred = loso_cv(d, sb["data"]["feature_set"], best_cfg, args.seed)
        selected[track] = {"data": sb["data"], "config": best_cfg, "dev_cv": _slim(r),
                           "dev_cv_pooled": r["pooled"]}
        plot_cv_confusion(r["pooled"]["confusion_matrix"], os.path.join(args.out, f"cv_confusion_{track}.png"),
                          f"{track}: {best_cfg['family']} w{sb['data']['window_sec']} "
                          f"{sb['data']['feature_set'][-2:]}\npooled dev LOSO CV (12 subjects)")
    # the current pipeline's pooled CV confusion, for the amusement comparison
    d_ref = data_cache[(60, "full", "none", "wesad-chest-v1")]
    r_ref, _ = loso_cv(d_ref, "wesad-chest-v1", reference["config"], args.seed)
    plot_cv_confusion(r_ref["pooled"]["confusion_matrix"], os.path.join(args.out, "cv_confusion_reference.png"),
                      "current pipeline: RF w60 v1\npooled dev LOSO CV (12 subjects)")

    amusement = {"reference": _confusions(r_ref["pooled"]),
                 **{t: _confusions(selected[t]["dev_cv_pooled"]) for t in selected}}
    # does a calibrated model have to be compared on the calibrated window set? yes: give the
    # uncalibrated selected config's score on the same windows
    sb = stage_b["calibrated"]["data"]
    d_same = data_cache[(sb["window_sec"], "cal", "none", sb["feature_set"])]
    r_same, _ = loso_cv(d_same, sb["feature_set"], selected["uncalibrated"]["config"], args.seed)
    selected["uncalibrated_on_calibration_windows"] = _slim(r_same)
    with open(os.path.join(args.out, "amusement_analysis.json"), "w") as fh:
        json.dump(amusement, fh, indent=2)

    # secondary check: the original fixed split (9 train -> 3 validation subjects)
    from .model import subject_split
    all_subjects = dev_subjects + config.TEST_SUBJECTS
    train_s, val_s, test_s = subject_split(all_subjects, seed=args.seed)
    assert sorted(test_s) == sorted(config.TEST_SUBJECTS), "split no longer reproduces the fixed test subjects"
    fixed = {}
    for name, (data, fset, cfg) in {
        "reference": ((60, "full", "none"), "wesad-chest-v1", reference["config"]),
        **{t: ((s["data"]["window_sec"], s["data"]["protocol"], s["data"]["normalization"]),
               s["data"]["feature_set"], s["config"]) for t, s in selected.items() if "data" in s},
    }.items():
        d = data_cache[(*data, fset)]
        tr, va = d[d["subject"].isin(train_s)], d[d["subject"].isin(val_s)]
        model = fit_model(cfg, tr[config.FEATURE_SETS[fset]], tr["label"].to_numpy(), args.seed, probability=False)
        m = compute_metrics(va["label"].to_numpy(), np.asarray(model.predict(va[config.FEATURE_SETS[fset]])).astype(int))
        fixed[name] = {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")} | \
            {f"recall_{c}": m["per_class"][c]["recall"] for c in config.CLASSES}
    selected["fixed_train_validation_split"] = {"train": train_s, "validation": val_s, "results": fixed}

    selected["reference_current_pipeline"] = _slim(reference)
    selected["protocol"] = ("leave-one-subject-out CV over development subjects "
                            f"{dev_subjects}; test subjects {config.TEST_SUBJECTS} excluded")
    with open(os.path.join(args.out, "selected_pipeline.json"), "w") as fh:
        json.dump(selected, fh, indent=2)

    lines = [f"Development subjects: {dev_subjects}   (test {config.TEST_SUBJECTS} never used)",
             "Scores: leave-one-subject-out CV over the development subjects (mean ± SD over 12 folds)", "",
             f"reference (current)  {_desc(reference)}  {fmt(reference)}"]
    for t in ("uncalibrated", "calibrated"):
        s = selected[t]
        lines.append(f"{t:<20} {s['config']['family']} {json.dumps(s['config']['params'])} "
                     f"w{s['data']['window_sec']} {s['data']['feature_set']} {s['data']['normalization']}")
        lines.append(f"{'':<20} {fmt(s['dev_cv'])}")
    lines.append(f"uncalibrated selected, same windows as calibrated: {fmt(selected['uncalibrated_on_calibration_windows'])}")
    lines.append(f"fixed train->validation split: {json.dumps(fixed)}")
    lines.append(f"({time.time() - t0:.0f} s)")
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return selected


def _slim(r):
    return {k: v for k, v in r.items() if k not in ("pooled",)}


def _desc(r):
    return (f"w{r['window_sec']} {r['protocol']}/{r['normalization']} {r['feature_set'][-2:]} "
            f"{r['config']['family']}")


def _confusions(pooled):
    cm = np.asarray(pooled["confusion_matrix"])
    idx = {c: i for i, c in enumerate(config.CLASSES)}
    b, a, s = idx["baseline"], idx["amusement"], idx["stress"]
    return {"confusion_matrix": cm.tolist(),
            "amusement_predicted_as_baseline": int(cm[a, b]), "amusement_total": int(cm[a].sum()),
            "baseline_predicted_as_amusement": int(cm[b, a]), "baseline_total": int(cm[b].sum()),
            "amusement_predicted_as_stress": int(cm[a, s]),
            "amusement_recall": pooled["per_class"]["amusement"]["recall"],
            "amusement_precision": pooled["per_class"]["amusement"]["precision"]}


if __name__ == "__main__":
    run(parse_args())

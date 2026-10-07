"""
body_vitals/model.py - subject-wise splits, candidate classifiers, metrics, persistence.

Windows from one subject are strongly correlated, so every split here is BY
SUBJECT (GroupShuffleSplit / LeaveOneGroupOut with groups = subject id). A
subject never contributes windows to more than one of train / validation / test.
"""

import json
import os

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def subject_split(subjects, seed=config.RANDOM_SEED, n_test=config.N_TEST_SUBJECTS,
                  n_val=config.N_VAL_SUBJECTS):
    """Disjoint (train, val, test) lists of subject ids, reproducible from `seed`."""
    from sklearn.model_selection import GroupShuffleSplit
    subjects = np.array(sorted(set(subjects), key=lambda s: (len(s), s)))
    if len(subjects) < n_test + n_val + 2:
        raise ValueError(f"need at least {n_test + n_val + 2} subjects for a train/val/test split, "
                         f"got {len(subjects)}")
    # int test_size in GroupShuffleSplit = number of GROUPS (subjects) held out
    rest_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
                              .split(subjects, groups=subjects))
    rest = subjects[rest_idx]
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=n_val, random_state=seed + 1)
                              .split(rest, groups=rest))
    train, val, test = (sorted(map(str, x), key=_skey) for x in (rest[train_idx], rest[val_idx],
                                                                 subjects[test_idx]))
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test)), "subject leakage"
    return train, val, test


def _skey(s):
    return (len(s), s)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def xgboost_available():
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:          # not installed, or libomp missing on macOS
        return False


FAMILIES = ["RandomForest", "XGBoost", "HistGradientBoosting", "LogisticRegression", "SVM"]
SCALED_FAMILIES = {"LogisticRegression", "SVM"}       # standardised inside the pipeline (per training fold)


def default_configs():
    """One sensible default configuration per model family (all class-balanced)."""
    configs = [
        {"name": "rf", "family": "RandomForest",
         "params": {"n_estimators": 500, "min_samples_leaf": 1, "max_features": "sqrt"}},
        {"name": "hgb", "family": "HistGradientBoosting",
         "params": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0}},
        {"name": "logreg", "family": "LogisticRegression", "params": {"C": 1.0}},
        {"name": "svm", "family": "SVM", "params": {"C": 1.0, "gamma": "scale"}},
    ]
    if xgboost_available():
        configs.insert(1, {"name": "xgb", "family": "XGBoost",
                           "params": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
                                      "subsample": 0.8, "colsample_bytree": 0.8}})
    return configs


def candidate_configs():
    """Configurations compared by `train` on the validation subjects (backwards compatible)."""
    return default_configs()


def build_model(cfg, seed=config.RANDOM_SEED, probability=True):
    """sklearn estimator for a config. Class balancing: RF/HGB/LR/SVM via class_weight,
    XGBoost via per-sample weights in fit_model. Scaling (LR, SVM) lives inside the
    pipeline, so it is fitted on the training fold only."""
    fam, params = cfg["family"], dict(cfg["params"])
    balance = params.pop("class_weight", "balanced")
    if fam == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        cw = "balanced_subsample" if balance == "balanced" else balance
        return RandomForestClassifier(class_weight=cw, random_state=seed, n_jobs=-1, **params)
    if fam == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="multi:softprob", num_class=len(config.CLASSES), eval_metric="mlogloss",
                             tree_method="hist", random_state=seed, n_jobs=4, **params)
    if fam == "HistGradientBoosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(class_weight=balance, early_stopping=False, random_state=seed,
                                              **params)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if fam == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight=balance, max_iter=5000, **params))
    if fam == "SVM":
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight=balance, probability=probability,
                                                   random_state=seed, **params))
    raise ValueError(f"unknown model family {fam}")


def fit_model(cfg, X, y, seed=config.RANDOM_SEED, probability=True):
    """Fit one configuration. X must be a DataFrame whose columns are a registered
    feature set (so the model records feature_names_in_)."""
    from sklearn.utils.class_weight import compute_sample_weight
    assert list(X.columns) in list(config.FEATURE_SETS.values()), "feature columns are not a registered set"
    model = build_model(cfg, seed, probability)
    if cfg["family"] == "XGBoost" and cfg["params"].get("class_weight", "balanced") == "balanced":
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    labels = list(range(len(config.CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro",
                                                    zero_division=0)
    return {
        "n_windows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(config.CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "confusion_matrix_axes": "rows = true class, columns = predicted class, order = classes",
    }


def majority_baseline(y_train, y_eval):
    """Always predict the most frequent TRAINING class."""
    majority = int(np.bincount(np.asarray(y_train), minlength=len(config.CLASSES)).argmax())
    out = compute_metrics(y_eval, np.full(len(y_eval), majority))
    out["predicts"] = config.CLASSES[majority]
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_artifacts(model, metadata, model_path=config.MODEL_PATH, metadata_path=config.METADATA_PATH):
    import joblib
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)


def load_artifacts(model_path=config.MODEL_PATH, metadata_path=config.METADATA_PATH):
    import joblib
    with open(metadata_path) as fh:
        metadata = json.load(fh)
    return joblib.load(model_path), metadata

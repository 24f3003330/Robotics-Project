"""
body_vitals/dataset.py - WESAD subject files -> one row of features per window.

WESAD layout (as distributed):  <WESAD>/S2/S2.pkl, <WESAD>/S3/S3.pkl, ...
Each pickle (latin-1 encoded) holds
    signal.chest.{ECG, EDA, Resp, EMG, Temp, ACC}   RespiBAN, all at 700 Hz
    signal.wrist.{BVP, EDA, TEMP, ACC}              Empatica E4 (ignored here)
    label                                           700 Hz, aligned with chest
Only chest ECG / EDA / Resp and labels 1-3 are used.
"""

import csv
import json
import os
import pickle
import re
from collections import Counter

import numpy as np

from . import config
from .features import extract_window_features
from .preprocessing import SignalQualityError, label_segments, window_bounds

QUALITY_COLUMNS = ["r_peaks", "rr_valid", "rr_artifact_fraction", "scr_peaks", "breaths"]
META_COLUMNS = ["subject", "wesad_label", "label", "class", "start_s"]


class SubjectSkipped(Exception):
    """A whole subject cannot be used (missing / corrupted file, missing signal)."""


def subject_sort_key(subject):
    m = re.match(r"S(\d+)$", subject)
    return (int(m.group(1)) if m else 10 ** 6, subject)


def find_subjects(data_dir):
    """Subject ids (S2, S3, ...) present as folders in the WESAD directory."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"WESAD directory not found: {data_dir}")
    subs = [d for d in os.listdir(data_dir)
            if re.match(r"S\d+$", d) and os.path.isdir(os.path.join(data_dir, d))]
    return sorted(subs, key=subject_sort_key)


def load_subject(data_dir, subject):
    """Return (dict of 1-D chest signals, labels). Raises SubjectSkipped."""
    path = os.path.join(data_dir, subject, f"{subject}.pkl")
    if not os.path.isfile(path):
        raise SubjectSkipped(f"missing file {path}")
    try:
        with open(path, "rb") as fh:
            raw = pickle.load(fh, encoding="latin1")
        chest = raw["signal"]["chest"]
        labels = np.asarray(raw["label"]).reshape(-1)
    except Exception as exc:          # truncated download, wrong format, ...
        raise SubjectSkipped(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc
    signals = {}
    for name, key in config.WESAD_SIGNALS.items():
        if key not in chest:
            raise SubjectSkipped(f"chest signal {key} missing")
        x = np.asarray(chest[key], dtype=float).reshape(-1)
        if len(x) != len(labels):
            raise SubjectSkipped(f"{key} has {len(x)} samples but labels have {len(labels)}")
        if not np.isfinite(x).any():
            raise SubjectSkipped(f"{key} contains no finite samples")
        signals[name] = x
    del raw
    return signals, labels


def process_subject(data_dir, subject, window_sec=config.WINDOW_SEC, step_sec=config.STEP_SEC):
    """All windows of one subject. Returns (rows, summary). Raises SubjectSkipped."""
    signals, labels = load_subject(data_dir, subject)
    fs = config.WESAD_CHEST_FS
    rows, discarded, created = [], Counter(), Counter()
    discard_log = []
    for wesad_label, start, stop in label_segments(labels, config.WESAD_LABEL_TO_CLASS):
        label = config.WESAD_LABEL_TO_CLASS[wesad_label]
        cls = config.CLASSES[label]
        for a, b in window_bounds(start, stop, fs, window_sec, step_sec):
            created[cls] += 1
            try:
                feats, quality = extract_window_features(signals["ecg"][a:b], signals["eda"][a:b],
                                                         signals["resp"][a:b], fs)
            except SignalQualityError as exc:
                discarded[exc.reason] += 1
                discard_log.append({"subject": subject, "class": cls, "start_s": round(a / fs, 1),
                                    "reason": exc.reason, "detail": exc.detail})
                continue
            row = {"subject": subject, "wesad_label": wesad_label, "label": label, "class": cls,
                   "start_s": round(a / fs, 1)}
            row.update(feats)
            row.update({k: quality.get(k) for k in QUALITY_COLUMNS})
            rows.append(row)
    kept = Counter(r["class"] for r in rows)
    summary = {"subject": subject, "windows_created": dict(created), "windows_kept": dict(kept),
               "windows_discarded": sum(discarded.values()), "discard_reasons": dict(discarded),
               "discard_log": discard_log}
    return rows, summary


def write_features(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = META_COLUMNS + config.ALL_FEATURE_NAMES + QUALITY_COLUMNS
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_features(path=config.FEATURES_CSV, feature_names=None):
    """Feature table as a pandas DataFrame (raises with the command to run if absent).
    `feature_names` (default config.FEATURE_NAMES) must all be present."""
    import pandas as pd
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found - run `python -m body_vitals.prepare_wesad "
                                f"--data /path/to/WESAD` first")
    df = pd.read_csv(path)
    missing = [c for c in META_COLUMNS + list(feature_names or config.FEATURE_NAMES) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; re-run prepare_wesad")
    return df


def write_report(report, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

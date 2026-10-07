"""
body_vitals/normalization.py - personal-baseline ("rest calibration") features.

Physiology differs far more BETWEEN people than between conditions within a
person (e.g. 93 % of EDA-level variance in WESAD is between subjects), so the
same absolute value can be "rest" for one person and "aroused" for another.

Rest calibration expresses every feature relative to the person's own resting
reference:

    x_calibrated = x - median(x over the person's calibration windows)

Calibration windows = the windows that lie entirely inside the first
CALIBRATION_SEC seconds of the person's resting period. In deployment this
is a short seated rest recorded when a session starts (the app instructs the
user to sit still), which is available before any prediction is made.

In WESAD the resting period is the baseline condition, which is always the first
condition of the protocol. Its label is used ONLY to locate the first 5 minutes
of the recording's rest period - the equivalent of "the user was told to rest
now" - never to normalise with statistics of stress or amusement windows. All
windows that overlap the calibration period are removed from training AND
evaluation (they are used only as the reference), so a model is never scored on
the very windows that defined its reference.

Caveat (documented in docs/BODY_VITALS.md): WESAD's protocol order is fixed
(baseline first), so calibrated features also partly encode slow drift over
the session.
"""

import numpy as np


CALIBRATION_SEC = 300


def calibration_masks(df, window_sec, calibration_sec=CALIBRATION_SEC):
    """Boolean masks over the rows of a WESAD feature table (needs `subject`, `class`,
    `start_s`): (is_reference, overlaps_calibration)."""
    ref = np.zeros(len(df), bool)
    overlap = np.zeros(len(df), bool)
    subjects = df["subject"].to_numpy()
    classes = df["class"].to_numpy()
    starts = df["start_s"].to_numpy(dtype=float)
    for subject in np.unique(subjects):
        rows = subjects == subject
        base = rows & (classes == "baseline")
        if not base.any():
            continue
        start = starts[base].min()        # first baseline window starts at the segment start
        end = start + calibration_sec
        ref |= base & (starts >= start - 1e-6) & (starts + window_sec <= end + 1e-6)
        overlap |= rows & (starts < end - 1e-6) & (starts + window_sec > start + 1e-6)
    return ref, overlap


def reference_from_windows(feature_rows, feature_names):
    """Per-feature median of calibration windows (a DataFrame or list of dicts)."""
    import pandas as pd
    rows = pd.DataFrame(list(feature_rows) if not hasattr(feature_rows, "columns") else feature_rows)
    if rows.empty:
        raise ValueError("no calibration windows")
    return {f: float(rows[f].median()) for f in feature_names}


def apply_reference(features, reference, feature_names):
    """One window (dict) -> calibrated dict."""
    return {f: float(features[f]) - reference[f] for f in feature_names}


def calibrate_table(df, feature_names, window_sec, calibration_sec=CALIBRATION_SEC):
    """WESAD table -> (calibrated table without calibration-period windows, references by subject)."""
    ref_mask, overlap = calibration_masks(df, window_sec, calibration_sec)
    out = df.loc[~overlap].copy()
    refs = {}
    for subject in df["subject"].unique():
        rows = df.loc[ref_mask & (df["subject"] == subject).to_numpy()]
        if rows.empty:
            raise ValueError(f"{subject}: no calibration windows in the first {calibration_sec} s of rest")
        refs[subject] = reference_from_windows(rows, feature_names)
        sel = out["subject"] == subject
        for f in feature_names:
            out.loc[sel, f] = out.loc[sel, f] - refs[subject][f]
    return out, refs


def drop_calibration_period(df, window_sec, calibration_sec=CALIBRATION_SEC):
    """Same windows as calibrate_table keeps, but features left absolute (fair comparison)."""
    _, overlap = calibration_masks(df, window_sec, calibration_sec)
    return df.loc[~overlap].copy()

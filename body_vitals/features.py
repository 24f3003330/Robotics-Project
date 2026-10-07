"""
body_vitals/features.py - one window of ECG / EDA / respiration -> feature vector.

The same function builds the WESAD training rows and the live inference input,
so a feature can never mean one thing in training and another in the app.
Every feature of every registered set is computed (config.ALL_FEATURE_NAMES);
a model uses the columns of its own feature set. Units: config.FEATURE_UNITS.
"""

import numpy as np
from scipy import signal

from . import config
from .preprocessing import (SignalQualityError, as_1d, check_finite, check_not_flat, clean_rr,
                            condition_eda, decompose_eda, detect_breaths, detect_r_peaks)


def hrv_from_rr(rr, valid):
    """SDNN and RMSSD in ms from RR intervals (s) - never from raw ECG samples.

    SDNN  = sample standard deviation of the clean RR intervals.
    RMSSD = root mean square of successive differences, using only pairs of
            ADJACENT clean intervals, so a removed beat never creates a fake jump.
    """
    rr = np.asarray(rr, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    good = rr[valid]
    if len(good) < 3:
        raise SignalQualityError("too_few_rr", f"{len(good)} clean RR intervals")
    pairs = valid[:-1] & valid[1:]
    diffs = np.diff(rr)[pairs]
    if len(diffs) < 2:
        raise SignalQualityError("too_few_rr", "no adjacent clean RR pairs for RMSSD")
    return {"hrv_sdnn": float(np.std(good, ddof=1) * 1000.0),
            "hrv_rmssd": float(np.sqrt(np.mean(diffs ** 2)) * 1000.0)}


def ecg_features(ecg, fs, min_valid_rr=config.MIN_VALID_RR):
    ecg = as_1d(ecg, "ecg")
    check_finite(ecg, "ecg")
    check_not_flat(ecg, "ecg")
    peaks = detect_r_peaks(ecg, fs)
    if len(peaks) < 3:
        raise SignalQualityError("too_few_r_peaks", f"{len(peaks)} R-peaks detected")
    rr, valid = clean_rr(peaks, fs)
    n_valid = int(valid.sum())
    artifact_fraction = 1.0 - n_valid / len(rr)
    if n_valid < min_valid_rr:
        raise SignalQualityError("too_few_rr", f"{n_valid} clean RR intervals (< {min_valid_rr})")
    if artifact_fraction > config.MAX_RR_ARTIFACT_FRACTION:
        raise SignalQualityError("abnormal_rr", f"{artifact_fraction:.0%} of RR intervals rejected")
    hr = 60.0 / rr[valid]
    feats = {"hr_mean": float(hr.mean()), "hr_std": float(hr.std(ddof=1)),
             "hr_min": float(hr.min()), "hr_max": float(hr.max())}
    feats.update(hrv_from_rr(rr, valid))
    # v2: robust centre, spread relative to level, and pNN50 (share of adjacent clean RR
    # pairs differing by more than 50 ms - a vagal-tone index like RMSSD, less outlier-driven)
    pairs = valid[:-1] & valid[1:]
    diffs = np.abs(np.diff(rr)[pairs])
    feats.update({"hr_median": float(np.median(hr)), "hr_range": float(hr.max() - hr.min()),
                  "hr_cv": float(hr.std(ddof=1) / hr.mean()),
                  "hrv_pnn50": float(np.mean(diffs > 0.050))})
    return feats, {"r_peaks": int(len(peaks)), "rr_valid": n_valid,
                   "rr_artifact_fraction": round(artifact_fraction, 4)}


def eda_features(eda, fs):
    eda = as_1d(eda, "eda")
    check_finite(eda, "eda")
    check_not_flat(eda, "eda")
    x = condition_eda(eda, fs)
    t = np.arange(len(x)) / config.EDA_FS
    slope = float(np.polyfit(t, x, 1)[0])
    _, phasic = decompose_eda(x)
    scr, props = signal.find_peaks(phasic, prominence=config.SCR_MIN_AMPLITUDE_US,
                                   distance=config.EDA_FS)      # >= 1 s apart
    minutes = len(x) / config.EDA_FS / 60.0
    amps = props["prominences"]
    return {"eda_mean": float(x.mean()), "eda_std": float(x.std(ddof=1)),
            "eda_min": float(x.min()), "eda_max": float(x.max()),
            "eda_range": float(x.max() - x.min()), "eda_slope": slope,
            "eda_phasic_std": float(phasic.std(ddof=1)),
            "eda_scr_per_min": float(len(scr) / minutes),
            # v2: variability relative to the person's conductance level, and SCR amplitudes
            # (0 when the window has no response)
            "eda_cv": float(x.std(ddof=1) / abs(x.mean())) if abs(x.mean()) > 1e-6 else 0.0,
            "eda_scr_amp_mean": float(amps.mean()) if len(amps) else 0.0,
            "eda_scr_amp_max": float(amps.max()) if len(amps) else 0.0}, {"scr_peaks": int(len(scr))}


def resp_features(resp, fs):
    resp = as_1d(resp, "resp")
    check_finite(resp, "resp")
    check_not_flat(resp, "resp")
    filt, peaks = detect_breaths(resp, fs)
    intervals = np.diff(peaks) / config.RESP_FS
    intervals = intervals[(intervals >= config.BREATH_MIN_S) & (intervals <= config.BREATH_MAX_S)]
    if len(intervals) < config.MIN_BREATHS:
        raise SignalQualityError("too_few_breaths", f"{len(intervals)} valid breath intervals")
    rate = 60.0 / intervals
    feats = {"resp_rate_mean": float(rate.mean()), "resp_rate_std": float(rate.std(ddof=1)),
             "resp_rate_min": float(rate.min()), "resp_rate_max": float(rate.max()),
             "resp_interval_cv": float(intervals.std(ddof=1) / intervals.mean())}
    feats.update(breath_shape(filt, peaks))
    return feats, {"breaths": int(len(intervals))}


def breath_shape(filt, peaks):
    """v2: depth and timing of each complete breath (trough -> inhalation peak -> trough).

    resp_amp_mean  mean peak-to-trough depth of the band-passed chest signal (sensor units)
    resp_amp_cv    breath-to-breath depth variability (speech / laughter make it irregular)
    resp_ie_ratio  mean inhalation / exhalation duration (speech and laughter shorten
                   inhalation relative to exhalation)
    """
    troughs = np.array([a + int(np.argmin(filt[a:b])) for a, b in zip(peaks[:-1], peaks[1:])])
    amps, ie = [], []
    for k in range(1, len(peaks) - 1):
        t_prev, t_next = troughs[k - 1], troughs[k]
        amps.append(filt[peaks[k]] - (filt[t_prev] + filt[t_next]) / 2.0)
        inhale, exhale = peaks[k] - t_prev, t_next - peaks[k]
        if inhale > 0 and exhale > 0:
            ie.append(inhale / exhale)
    if len(amps) < 2 or not ie:
        raise SignalQualityError("too_few_breaths", "too few complete breaths for depth / timing")
    amps = np.asarray(amps)
    return {"resp_amp_mean": float(amps.mean()),
            "resp_amp_cv": float(amps.std(ddof=1) / amps.mean()) if amps.mean() > 0 else 0.0,
            "resp_ie_ratio": float(np.mean(ie))}


def extract_window_features(ecg, eda, resp, fs_ecg, fs_eda=None, fs_resp=None, min_valid_rr=None):
    """Features for one window. Raises SignalQualityError (with .reason) if the
    window is unusable. Returns (features dict in config.ALL_FEATURE_NAMES order, quality dict)."""
    fs_eda = fs_eda or fs_ecg
    fs_resp = fs_resp or fs_ecg
    if min_valid_rr is None:
        # scale the 30-RR minimum to the window length (30 per 60 s)
        seconds = len(np.asarray(ecg).reshape(-1)) / fs_ecg
        min_valid_rr = max(5, int(round(config.MIN_VALID_RR * seconds / config.WINDOW_SEC)))
    out, quality = {}, {}
    for fn, args in ((ecg_features, (ecg, fs_ecg, min_valid_rr)), (eda_features, (eda, fs_eda)),
                     (resp_features, (resp, fs_resp))):
        feats, q = fn(*args)
        out.update(feats)
        quality.update(q)
    bad = [k for k, v in out.items() if not np.isfinite(v)]
    if bad:
        raise SignalQualityError("nonfinite_feature", ", ".join(bad))
    return {name: out[name] for name in config.ALL_FEATURE_NAMES}, quality

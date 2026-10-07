"""
body_vitals/preprocessing.py - signal conditioning, beat/breath detection and windowing.

Everything here is a pure function of numpy arrays so the same code runs on
WESAD recordings during training and on live sensor windows at inference.

    ECG   band-pass 5-15 Hz -> Pan-Tompkins-style energy envelope -> R-peaks
          (refined to the true R extremum on the band-passed ECG, polarity-aware)
    EDA   low-pass 1 Hz, resampled to 4 Hz
    Resp  resampled to 25 Hz, band-pass 0.1-0.7 Hz (6-42 breaths/min) -> one
          breath = one inhalation peak
"""

from fractions import Fraction

import numpy as np
from scipy import signal

from . import config


class SignalQualityError(ValueError):
    """A window cannot give trustworthy features. `reason` is a short, countable key."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def as_1d(x, name):
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise SignalQualityError(f"missing_{name}", f"no {name} samples")
    return arr


def check_finite(x, name):
    if not np.all(np.isfinite(x)):
        bad = int(np.sum(~np.isfinite(x)))
        raise SignalQualityError(f"nonfinite_{name}", f"{bad} NaN/inf samples in {name}")


def check_not_flat(x, name, min_std=1e-6):
    if float(np.std(x)) < min_std:
        raise SignalQualityError(f"flat_{name}", f"{name} is a flat line (sensor off?)")


def resample(x, fs_in, fs_out):
    """Polyphase resampling (includes the anti-aliasing filter)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    # padtype="line": extrapolate the trend at the edges instead of zero-padding,
    # which would make every window dip towards 0 at both ends
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, padtype="line")


def _filtfilt_sos(sos, x):
    # padlen bounded so short windows still filter (scipy's default can exceed the length)
    return signal.sosfiltfilt(sos, x, padlen=min(len(x) - 1, 3 * 50))


def bandpass(x, fs, low, high, order=3):
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return _filtfilt_sos(sos, x)


def lowpass(x, fs, cutoff, order=3):
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return _filtfilt_sos(sos, x)


# ---------------------------------------------------------------------------
# ECG -> R-peaks
# ---------------------------------------------------------------------------
def detect_r_peaks(ecg, fs):
    """Return R-peak sample indices.

    1. band-pass 5-15 Hz (QRS energy band; removes baseline wander and T waves)
    2. derivative -> square -> 150 ms moving average (Pan & Tompkins 1985 envelope)
    3. envelope peaks at least 300 ms apart (<= 200 bpm) and above 30 % of the
       window's robust envelope maximum
    4. each peak moved to the extremum of the band-passed ECG within +-75 ms, using
       the window's dominant QRS polarity (WESAD has inverted leads on some subjects),
       so every beat uses the same fiducial point and RR jitter stays small
    """
    ecg = as_1d(ecg, "ecg")
    if len(ecg) < fs * 5:
        raise SignalQualityError("short_ecg", "less than 5 s of ECG")
    filt = bandpass(ecg, fs, 5.0, 15.0)
    energy = np.gradient(filt) ** 2
    win = max(1, int(round(0.150 * fs)))
    envelope = np.convolve(energy, np.ones(win) / win, mode="same")
    height = 0.3 * np.percentile(envelope, 99)
    if height <= 0:
        raise SignalQualityError("flat_ecg", "no QRS energy in the ECG")
    cand, _ = signal.find_peaks(envelope, height=height, distance=int(config.RR_MIN_S * fs))
    if len(cand) < 2:
        return np.array([], dtype=int)

    half = int(round(0.075 * fs))
    segments = [(max(0, c - half), min(len(filt), c + half + 1)) for c in cand]
    pos = np.median([filt[a:b].max() for a, b in segments])
    neg = np.median([-filt[a:b].min() for a, b in segments])
    sign = 1.0 if pos >= neg else -1.0
    peaks = np.array([a + int(np.argmax(sign * filt[a:b])) for a, b in segments])
    peaks = np.unique(peaks)
    # refinement can pull two candidates onto (almost) the same beat: keep the larger
    keep = [peaks[0]]
    for p in peaks[1:]:
        if p - keep[-1] < config.RR_MIN_S * fs:
            if sign * filt[p] > sign * filt[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    return np.asarray(keep, dtype=int)


def clean_rr(peaks, fs):
    """RR intervals (s) from R-peaks plus a validity mask.

    An RR is invalid if outside 0.3-2.0 s (30-200 bpm) or more than 20 % away
    from the median of its 5 neighbours (missed / extra / ectopic beat).
    """
    rr = np.diff(np.asarray(peaks, dtype=float)) / fs
    valid = (rr >= config.RR_MIN_S) & (rr <= config.RR_MAX_S)
    if len(rr) >= 3:
        padded = np.pad(rr, 2, mode="edge")
        local = np.array([np.median(padded[i:i + 5]) for i in range(len(rr))])
        valid &= np.abs(rr - local) <= config.RR_MAX_LOCAL_DEVIATION * local
    return rr, valid


# ---------------------------------------------------------------------------
# Respiration -> breaths
# ---------------------------------------------------------------------------
def detect_breaths(resp, fs):
    """Return (filtered 25 Hz respiration, inhalation-peak indices at 25 Hz).

    The chest band signal is resampled to 25 Hz and band-passed 0.1-0.7 Hz
    (6-42 breaths/min). A breath is a peak at least 1.5 s after the previous one
    with a prominence of at least 30 % of the window's signal standard deviation,
    which rejects small ripples from talking or movement.
    """
    resp = as_1d(resp, "resp")
    x = resample(resp, fs, config.RESP_FS)
    if len(x) < config.RESP_FS * 10:
        raise SignalQualityError("short_resp", "less than 10 s of respiration")
    x = bandpass(x, config.RESP_FS, 0.1, 0.7, order=2)
    sd = float(np.std(x))
    if sd < 1e-9:
        raise SignalQualityError("flat_resp", "respiration is a flat line")
    peaks, _ = signal.find_peaks(x, distance=int(config.BREATH_MIN_S * config.RESP_FS),
                                 prominence=0.3 * sd)
    return x, peaks


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def condition_eda(eda, fs):
    """Resample to 4 Hz (anti-aliased) and low-pass 1 Hz (removes electrical noise)."""
    eda = resample(as_1d(eda, "eda"), fs, config.EDA_FS)
    return lowpass(eda, config.EDA_FS, 1.0, order=2)


def decompose_eda(eda4):
    """Split 4 Hz EDA into tonic (skin conductance level, < 0.05 Hz) and phasic
    (responses) parts. A simple filter decomposition, not cvxEDA / Ledalab."""
    tonic = lowpass(eda4, config.EDA_FS, 0.05, order=2)
    return tonic, eda4 - tonic


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def label_segments(labels, keep):
    """Contiguous runs of a kept label: list of (label, start, stop) sample indices."""
    labels = np.asarray(labels).reshape(-1)
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(labels)]])
    return [(int(labels[a]), int(a), int(b)) for a, b in zip(starts, stops) if int(labels[a]) in keep]


def window_bounds(start, stop, fs, window_sec=config.WINDOW_SEC, step_sec=config.STEP_SEC):
    """Fixed windows fully inside [start, stop). A tail shorter than one window is dropped."""
    win, step = int(window_sec * fs), int(step_sec * fs)
    return [(a, a + win) for a in range(start, stop - win + 1, step)]

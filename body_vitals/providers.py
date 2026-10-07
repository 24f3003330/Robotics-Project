"""
body_vitals/providers.py - where a window of physiological signals comes from.

    PhysiologicalInputProvider     abstract: read_window() -> SignalWindow
      DemoVitalsProvider           SIMULATED waveforms for testing the UI / pipeline
      ArraySignalsProvider         signals handed over by a caller (API, file, logger)

Every provider returns RAW SIGNALS, never ready-made features, so a reading from
any source goes through the same features.extract_window_features() that built
the WESAD training data.

A real sensor provider (e.g. a chest strap ECG + EDA electrodes + respiration
belt) is NOT implemented here - no hardware is assumed. To add one, subclass
PhysiologicalInputProvider with source = "sensor", is_simulated = False, and make
read_window() return the last 60 s of ECG / EDA / respiration with their real
sampling rates (see docs/BODY_VITALS.md, "Real sensors"). A laptop webcam cannot
measure ECG, skin conductance or respiration and is not offered as a source.

The demo generator is NOT WESAD data and is never used for training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class SignalWindow:
    ecg: np.ndarray
    eda: np.ndarray
    resp: np.ndarray
    fs_ecg: float
    fs_eda: float
    fs_resp: float
    source: str                 # "demo" | "sensor" | "api"
    is_simulated: bool          # True only for generated data; None if the origin is unknown
    details: dict = field(default_factory=dict)

    @property
    def seconds(self):
        return len(self.ecg) / self.fs_ecg


class PhysiologicalInputProvider(ABC):
    source = "unknown"
    is_simulated = None

    @abstractmethod
    def read_window(self) -> SignalWindow:
        """Return the most recent window (config.WINDOW_SEC seconds) of ECG, EDA and respiration."""


class ArraySignalsProvider(PhysiologicalInputProvider):
    """Wraps signals supplied by someone else. Their origin cannot be verified here,
    so is_simulated is None and the UI must not call them sensor measurements."""
    source = "api"

    def __init__(self, ecg, eda, resp, fs_ecg, fs_eda=None, fs_resp=None):
        self._window = SignalWindow(np.asarray(ecg, float), np.asarray(eda, float), np.asarray(resp, float),
                                    float(fs_ecg), float(fs_eda or fs_ecg), float(fs_resp or fs_ecg),
                                    source=self.source, is_simulated=None)

    def read_window(self):
        return self._window


# ---------------------------------------------------------------------------
# Demo / simulation
# ---------------------------------------------------------------------------
# Target physiology per demo scenario. Hand-set, in the same range as the median
# WESAD window of each condition (models/body_vitals_metadata.json ->
# class_feature_medians). They describe what to SYNTHESISE; the displayed values
# are measured back from the synthetic waveforms by the real feature extractor.
DEMO_SCENARIOS = {
    "baseline": {"hr": 72, "sdnn_ms": 60, "rmssd_ms": 45, "eda_us": 3.7, "eda_slope": 0.0,
                 "scr_per_min": 0, "scr_amp": 0.0, "resp_bpm": 16, "resp_cv": 0.12},
    "stress": {"hr": 94, "sdnn_ms": 50, "rmssd_ms": 27, "eda_us": 4.4, "eda_slope": 0.004,
               "scr_per_min": 5, "scr_amp": 0.10, "resp_bpm": 18, "resp_cv": 0.35},
    "amusement": {"hr": 75, "sdnn_ms": 70, "rmssd_ms": 42, "eda_us": 3.4, "eda_slope": 0.0,
                  "scr_per_min": 1, "scr_amp": 0.05, "resp_bpm": 18, "resp_cv": 0.22},
}
DEMO_ECG_FS = 700
DEMO_SLOW_FS = 32


class DemoVitalsProvider(PhysiologicalInputProvider):
    """Synthetic ECG / EDA / respiration for one scenario of one simulated PERSON
    (pass `seed` for a repeatable person and window). SIMULATED - not a measurement.

    People differ: each provider draws a person whose resting heart rate, HRV,
    skin conductance level and breathing rate are offset from the scenario
    targets. read_calibration() returns that same person's resting windows - the
    reference a rest-calibrated model needs."""
    source = "demo"
    is_simulated = True

    def __init__(self, scenario="baseline", seed=None, seconds=config.WINDOW_SEC, jitter=0.08,
                 person_variation=True):
        if scenario not in DEMO_SCENARIOS:
            raise ValueError(f"unknown demo scenario {scenario!r}; choose one of {list(DEMO_SCENARIOS)}")
        self.scenario = scenario
        self.seconds = seconds
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.person = ({"hr": r.normal(0, 5), "hrv": float(np.exp(r.normal(0, 0.2))),
                        "eda": float(np.exp(r.normal(0, 0.4))), "resp": r.normal(0, 1.5)}
                       if person_variation else {"hr": 0.0, "hrv": 1.0, "eda": 1.0, "resp": 0.0})

    def _vary(self, value):
        return value * (1 + self.rng.normal(0, self.jitter)) if value else value

    def _targets(self, scenario):
        p = dict(DEMO_SCENARIOS[scenario])
        p["hr"] += self.person["hr"]
        p["sdnn_ms"] *= self.person["hrv"]
        p["rmssd_ms"] *= self.person["hrv"]
        p["eda_us"] *= self.person["eda"]
        p["resp_bpm"] += self.person["resp"]
        return {k: self._vary(v) for k, v in p.items()}

    def read_calibration(self, n_windows):
        """`n_windows` resting (baseline-scenario) windows of the same simulated person."""
        return [self._window("baseline") for _ in range(n_windows)]

    def read_window(self):
        return self._window(self.scenario)

    def _window(self, scenario):
        p = self._targets(scenario)
        return SignalWindow(synth_ecg(p["hr"], p["sdnn_ms"], p["rmssd_ms"], self.seconds, DEMO_ECG_FS, self.rng),
                            synth_eda(p["eda_us"], p["eda_slope"], p["scr_per_min"], p["scr_amp"], self.seconds,
                                      DEMO_SLOW_FS, self.rng),
                            synth_resp(p["resp_bpm"], p["resp_cv"], self.seconds, DEMO_SLOW_FS, self.rng),
                            DEMO_ECG_FS, DEMO_SLOW_FS, DEMO_SLOW_FS, source=self.source, is_simulated=True,
                            details={"scenario": scenario,
                                     "targets": {k: round(float(v), 4) for k, v in p.items()}})


def synth_rr(hr, sdnn_ms, rmssd_ms, seconds, rng):
    """RR series (s): white beat-to-beat noise (sets RMSSD ~ sqrt(2) sd) plus a slow
    0.1 Hz oscillation (adds to SDNN, barely to RMSSD)."""
    mean_rr = 60.0 / hr
    sd_white = rmssd_ms / 1000.0 / np.sqrt(2)
    slow_amp = np.sqrt(max(2 * ((sdnn_ms / 1000.0) ** 2 - sd_white ** 2), 0.0))
    rr, t = [], 0.0
    while t < seconds + 2:
        r = mean_rr + slow_amp * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, sd_white)
        r = float(np.clip(r, 0.35, 1.8))
        rr.append(r)
        t += r
    return np.array(rr)


def synth_ecg(hr, sdnn_ms, rmssd_ms, seconds, fs, rng):
    """Sum-of-Gaussians P-QRS-T complex at each beat + baseline wander + noise (mV-ish)."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    ecg = np.zeros(n)
    waves = [(-0.16, 0.025, 0.10), (-0.025, 0.008, -0.12), (0.0, 0.010, 1.0), (0.025, 0.010, -0.25),
             (0.25, 0.040, 0.30)]                       # (offset s, width s, amplitude): P, Q, R, S, T
    beat = rng.uniform(0.2, 0.6)
    for r in synth_rr(hr, sdnn_ms, rmssd_ms, seconds, rng):
        for off, width, amp in waves:
            centre = beat + off
            a, b = int((centre - 4 * width) * fs), int((centre + 4 * width) * fs) + 1
            a, b = max(a, 0), min(b, n)
            if a < b:
                ecg[a:b] += amp * np.exp(-0.5 * ((t[a:b] - centre) / width) ** 2)
        beat += r
        if beat > seconds + 0.5:
            break
    ecg += 0.05 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.3)) + rng.normal(0, 0.01, n)
    return ecg


def synth_eda(level, slope, scr_per_min, scr_amp, seconds, fs, rng):
    """Tonic level + linear trend + skin conductance responses (fast rise, slow decay)."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    eda = level + slope * (t - seconds / 2)
    n_scr = rng.poisson(scr_per_min * seconds / 60.0) if scr_per_min else 0
    for onset in rng.uniform(0, seconds - 4, n_scr):
        amp = scr_amp * rng.uniform(0.6, 1.4)
        dt = np.clip(t - onset, 0, None)
        eda += amp * (1 - np.exp(-dt / 0.75)) * np.exp(-dt / 4.0) * (t >= onset) * 2.0
    return eda + rng.normal(0, 0.002, n)


def synth_resp(rate_bpm, cv, seconds, fs, rng):
    """Breath-by-breath sinusoid whose period varies with the given coefficient of variation."""
    n = int(seconds * fs)
    intervals, total = [], 0.0
    while total < seconds + 10:
        iv = float(np.clip(rng.normal(60.0 / rate_bpm, cv * 60.0 / rate_bpm), 1.7, 12.0))
        intervals.append(iv)
        total += iv
    edges = np.concatenate([[0.0], np.cumsum(intervals)]) - rng.uniform(0, intervals[0])
    t = np.arange(n) / fs
    k = np.searchsorted(edges, t, side="right") - 1
    phase = 2 * np.pi * (k + (t - edges[k]) / np.diff(edges)[k])
    return np.sin(phase - np.pi / 2) + rng.normal(0, 0.03, n)

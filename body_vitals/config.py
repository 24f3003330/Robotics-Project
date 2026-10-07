"""
body_vitals/config.py - every tunable of the Body Vitals modality in one place.

The Body Vitals model is trained on WESAD (Schmidt et al., ICMI 2018) and
predicts that dataset's own study conditions:

    0 baseline    neutral reading / sitting (WESAD label 1)
    1 stress      Trier Social Stress Test (WESAD label 2)
    2 amusement   watching funny video clips (WESAD label 3)

They are NOT renamed to Calm / Happy / Angry / Anxiety: they are the conditions
a participant was in, not self-reported emotions.

Only the chest RespiBAN signals the RM's physiological concept needs are used:
ECG (-> heart rate and HRV), EDA (galvanic skin response) and respiration.
EMG, temperature, accelerometer and every wrist (Empatica E4) signal are ignored.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Paths. The WESAD folder is never hardcoded: pass --data or set WESAD_DIR.
#   data/body_vitals/   features written by `python -m body_vitals.prepare_wesad`
#   models/             trained model + metadata written by `python -m body_vitals.train`
#   body_vitals/results metrics / report / confusion matrix from training/evaluation
# ---------------------------------------------------------------------------
WESAD_DIR_ENV = "WESAD_DIR"
FEATURES_DIR = os.path.join(PROJECT_DIR, "data", "body_vitals")


def features_csv(window_sec, step_sec):
    """Feature table for one windowing."""
    return os.path.join(FEATURES_DIR, f"wesad_features_w{window_sec:g}_s{step_sec:g}.csv")


def preprocessing_report(window_sec, step_sec):
    return os.path.join(FEATURES_DIR, f"preprocessing_report_w{window_sec:g}_s{step_sec:g}.json")


# Windowing of the deployed model (see the windowing note below for WINDOW_SEC)
MODEL_WINDOW_SEC = 90
MODEL_STEP_SEC = 45
FEATURES_CSV = features_csv(MODEL_WINDOW_SEC, MODEL_STEP_SEC)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "body_vitals_model.pkl")
METADATA_PATH = os.path.join(PROJECT_DIR, "models", "body_vitals_metadata.json")
RESULTS_DIR = os.path.join(PACKAGE_DIR, "results")

# ---------------------------------------------------------------------------
# Classes. Index = model output index. WESAD label ids: 0 transient,
# 1 baseline, 2 stress, 3 amusement, 4 meditation, 5-7 ignored. Only 1-3 kept.
# ---------------------------------------------------------------------------
CLASSES = ["baseline", "stress", "amusement"]
CLASS_DISPLAY = {"baseline": "Baseline", "stress": "Stress", "amusement": "Amusement"}
WESAD_LABEL_TO_CLASS = {1: 0, 2: 1, 3: 2}

# ---------------------------------------------------------------------------
# Signals. WESAD chest (RespiBAN) data are all sampled at 700 Hz and the label
# array is aligned sample-by-sample with them, so no cross-device alignment
# is needed. Each signal is brought to its own working rate:
#   ECG   700 Hz (kept - R-peak timing resolution 1.4 ms matters for HRV)
#   EDA   4 Hz   (EDA content is < 1 Hz)
#   Resp  25 Hz  (breathing is < 1 Hz)
# ---------------------------------------------------------------------------
WESAD_CHEST_FS = 700
WESAD_SIGNALS = {"ecg": "ECG", "eda": "EDA", "resp": "Resp"}
ECG_FS = 700
EDA_FS = 4
RESP_FS = 25

# ---------------------------------------------------------------------------
# Windowing reference: 60 s windows, 30 s step (50 % overlap), each window entirely
# inside one labelled segment. WINDOW_SEC is also the reference length for
# MIN_VALID_RR (30 clean RR per 60 s, scaled to other window lengths) - do not
# change it to switch windowing; the deployed model's windowing is MODEL_WINDOW_SEC. 60 s is the shortest span for which SDNN/RMSSD
# and a breathing rate (6-15 breaths) are reasonably stable. Overlap only
# multiplies windows WITHIN a subject; splits are by subject, so it cannot leak.
# ---------------------------------------------------------------------------
WINDOW_SEC = 60
STEP_SEC = 30

# ---------------------------------------------------------------------------
# Signal-quality limits (a window failing any of them is discarded and the
# reason is counted in the preprocessing report).
# ---------------------------------------------------------------------------
RR_MIN_S = 0.3            # 200 bpm
RR_MAX_S = 2.0            # 30 bpm
RR_MAX_LOCAL_DEVIATION = 0.20   # RR > 20 % off the local median of 5 = ectopic / missed beat
MIN_VALID_RR = 30         # at least 30 clean RR intervals per 60 s window
MAX_RR_ARTIFACT_FRACTION = 0.20
BREATH_MIN_S = 1.5        # 40 breaths/min
BREATH_MAX_S = 15.0       # 4 breaths/min
MIN_BREATHS = 4           # at least 4 breath-to-breath intervals per window
SCR_MIN_AMPLITUDE_US = 0.02    # skin conductance response threshold on the phasic component

# ---------------------------------------------------------------------------
# Features. A feature SET is an ordered list of names; the trained model's
# metadata stores its set (name + version) and inference refuses a model whose
# list differs from the set registered here. Never change a registered set -
# add a new version instead.
#
#   wesad-chest-v1  the original 19 features
#   wesad-chest-v2  the 19 + 10 additional features (see docs/BODY_VITALS.md)
#
# features.extract_window_features() always computes ALL_FEATURE_NAMES; the
# engine passes the model only the columns of its own set, in its order.
# ---------------------------------------------------------------------------
FEATURES_V1 = [
    # heart rate from ECG R-peaks (beats per minute)
    "hr_mean", "hr_std", "hr_min", "hr_max",
    # heart rate variability from clean RR intervals (milliseconds)
    "hrv_sdnn", "hrv_rmssd",
    # electrodermal activity (microsiemens; slope in microsiemens per second)
    "eda_mean", "eda_std", "eda_min", "eda_max", "eda_range", "eda_slope",
    "eda_phasic_std", "eda_scr_per_min",
    # respiration rate from breath-to-breath intervals (breaths per minute)
    "resp_rate_mean", "resp_rate_std", "resp_rate_min", "resp_rate_max", "resp_interval_cv",
]
FEATURES_V2_EXTRA = [
    "hr_median", "hr_range", "hr_cv", "hrv_pnn50",
    "eda_cv", "eda_scr_amp_mean", "eda_scr_amp_max",
    "resp_amp_mean", "resp_amp_cv", "resp_ie_ratio",
]
FEATURE_SETS = {
    "wesad-chest-v1": FEATURES_V1,
    "wesad-chest-v2": FEATURES_V1 + FEATURES_V2_EXTRA,
}
ALL_FEATURE_NAMES = FEATURE_SETS["wesad-chest-v2"]
# Feature set of the deployed model (models/body_vitals_metadata.json is authoritative;
# inference always uses the set named in the model's metadata).
FEATURE_VERSION = "wesad-chest-v2"
FEATURE_NAMES = FEATURE_SETS[FEATURE_VERSION]
FEATURE_UNITS = {
    "hr_mean": "bpm", "hr_std": "bpm", "hr_min": "bpm", "hr_max": "bpm",
    "hrv_sdnn": "ms", "hrv_rmssd": "ms",
    "eda_mean": "uS", "eda_std": "uS", "eda_min": "uS", "eda_max": "uS", "eda_range": "uS",
    "eda_slope": "uS/s", "eda_phasic_std": "uS", "eda_scr_per_min": "1/min",
    "resp_rate_mean": "breaths/min", "resp_rate_std": "breaths/min", "resp_rate_min": "breaths/min",
    "resp_rate_max": "breaths/min", "resp_interval_cv": "ratio",
    "hr_median": "bpm", "hr_range": "bpm", "hr_cv": "ratio", "hrv_pnn50": "fraction",
    "eda_cv": "ratio", "eda_scr_amp_mean": "uS", "eda_scr_amp_max": "uS",
    "resp_amp_mean": "sensor units", "resp_amp_cv": "ratio", "resp_ie_ratio": "ratio",
}

# ---------------------------------------------------------------------------
# Training. Subject-wise split of the 15 WESAD subjects (S1 and S12 are not in
# the public release): 3 test subjects, 3 validation subjects, the rest train.
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_TEST_SUBJECTS = 3
N_VAL_SUBJECTS = 3
# The held-out test subjects produced by subject_split(seed=42). Fixed so every
# experiment, analysis and model version is judged on the same unseen people;
# development work (analysis, feature engineering, model selection) drops them.
TEST_SUBJECTS = ["S4", "S6", "S10"]
# Which pipeline from body_vitals/results/experiments/selected_pipeline.json
# `python -m body_vitals.train` builds by default ("uncalibrated" | "calibrated").
# Chosen from development-subject CV ONLY, before the test subjects were scored:
# rest calibration gave the largest and most consistent CV gain (docs/BODY_VITALS.md).
DEPLOY_TRACK = "calibrated"

DEMO_NOTICE = ("SIMULATED INPUT - these vital signs were generated by the demo mode from synthetic "
               "waveforms. They are not measurements from a person or a sensor.")

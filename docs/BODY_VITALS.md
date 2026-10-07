# Body Vitals: physiological state from WESAD

The Body Vitals modality (`/body-vitals`, package `body_vitals/`) classifies a 90-second window
of **ECG, electrodermal activity (EDA/GSR) and respiration** into one of WESAD's study
conditions: **Baseline, Stress or Amusement**.

- **One prediction mechanism:** a machine-learning model trained on WESAD. There are no hand-set
  thresholds and no rule-based arousal/valence layer.
- **Personal resting reference:** each window is compared with the person's own resting reference
  ("rest calibration", section 17), recorded at the start of a session.

It is independent of the other modalities. It uses no camera, no microphone, no fusion and no
music/robot output, and it shares no code or model with camera, voice, gender or handwriting.

```
Camera        ├── Facial Emotion          (face_emotion.py)
              └── Gender                  (voice_gender.py)
Microphone    └── Voice Emotion           (emotion_detector.py)
Handwriting   ├── EMOTHAW                 (handwriting/)
              └── 4-class prototype
Body Vitals   ├── ECG → HR / HRV          (body_vitals/)
              ├── EDA / GSR
              ├── Respiration
              ├── Feature extraction (29 features)
              ├── WESAD-trained ML model (RBF SVM, rest-calibrated)
              └── Baseline / Stress / Amusement
```

```
Physiological signals ─► preprocessing ─► 29 features ─► minus personal rest reference ─► WESAD-trained SVM ─► Baseline / Stress / Amusement
```

## How to run

```bash
# 1. Features from the WESAD download (path is never hardcoded; --data or $WESAD_DIR)
#    one table per windowing; the deployed model uses 90 s windows, 45 s step
for w in 30 60 90; do
  python -m body_vitals.prepare_wesad --data /path/to/WESAD --window-sec $w --step-sec $((w/2)) \
      --out data/body_vitals/wesad_features_w${w}_s$((w/2)).csv \
      --report data/body_vitals/preprocessing_report_w${w}_s$((w/2)).json
done
# 2. descriptive analysis (development subjects only)
python -m body_vitals.analyze --features data/body_vitals/wesad_features_w60_s30.csv
# 3. model / feature / window / normalisation study (development subjects only, ~15 min)
python -m body_vitals.experiment
# 4. train the selected pipeline, evaluate ONCE on the held-out test subjects, LOSO
python -m body_vitals.train                 # --track calibrated (default) | uncalibrated ; --legacy = v1
# 5. (optional) re-evaluate the saved model on its test subjects
python -m body_vitals.evaluate
# 6. app
python app.py      # http://localhost:5001/body-vitals
# tests
python -m unittest discover tests
```

`--data` must point at the folder that contains `S2/S2.pkl`, `S3/S3.pkl`, … as WESAD is
distributed. Everything is seeded (`config.RANDOM_SEED = 42`), so re-running gives identical
results.

| Output | Written by |
|---|---|
| `data/body_vitals/wesad_features_w{30,60,90}_s*.csv`: one row per window, all 29 features | `prepare_wesad` |
| `data/body_vitals/preprocessing_report_*.json`: subjects, windows, every discarded window and why | `prepare_wesad` |
| `body_vitals/results/analysis/feature_analysis.json` | `analyze` |
| `body_vitals/results/experiments/`: `stage_a.json`, `stage_b.json`, `selected_pipeline.json`, `amusement_analysis.json`, `summary.txt`, `cv_confusion_*.png` | `experiment` |
| `models/body_vitals_model.pkl`: trained classifier (joblib) | `train` |
| `models/body_vitals_metadata.json`: classes, feature set/order/version, preprocessing, window, normalisation, hyperparameters, subjects per split, development CV, test and LOSO metrics, leakage audit, timestamp | `train` |
| `body_vitals/results/metrics.json`, `classification_report.txt`, `confusion_matrix.png` | `train`, `evaluate` |
| `models/archive/body_vitals_model_v1.pkl` (+ metadata), `body_vitals/results/v1_baseline/`: the previous model and its results, kept for comparison and rollback | (archived) |

`data/` and `models/` are gitignored. **The WESAD dataset must never be committed.**

## 1. What WESAD is

WESAD (*Wearable Stress and Affect Detection*, Schmidt et al., ICMI 2018) is a public
multimodal dataset of 15 participants (S2–S17; S1 and S12 were dropped by the authors because of
sensor faults). Each participant wore a chest device (RespiBAN, 700 Hz) and a wrist device
(Empatica E4) while going through a lab protocol:

| WESAD label | Condition | Used here as |
|---|---|---|
| 1 | Baseline: about 20 min seated, reading neutral magazines | `baseline` (class 0) |
| 2 | Stress: Trier Social Stress Test (public speech + mental arithmetic) | `stress` (class 1) |
| 3 | Amusement: watching funny video clips | `amusement` (class 2) |
| 4 | Meditation | not used |
| 0, 5–7 | transient / undefined / not to be used | not used |

## 2. Signals used

Only the **chest (RespiBAN)** signals that match the RM's physiological concept:

| Signal | Used for |
|---|---|
| **ECG** | R-peaks → heart rate and HRV |
| **EDA** | skin conductance level and responses (GSR) |
| **Resp** | breathing rate from the chest band |

All three are recorded at 700 Hz by the same device and the WESAD label array is aligned
sample-by-sample with them, so no cross-device synchronisation is needed.

## 3. Signals ignored

Chest **EMG, temperature and accelerometer**, and every **wrist** signal (BVP, wrist EDA,
temperature, accelerometer). They might improve accuracy, but they are outside the requested
HR / HRV / GSR / respiration pipeline.

## 4. Heart rate from ECG

`preprocessing.detect_r_peaks` (ECG kept at 700 Hz, so timing resolution is 1.4 ms):

1. Band-pass 5–15 Hz (3rd-order Butterworth, zero-phase). This keeps the QRS energy and removes
   baseline wander and T waves.
2. Pan–Tompkins-style envelope: derivative, squared, then a 150 ms moving average.
3. Envelope peaks at least 0.3 s apart (at most 200 bpm) and above 30 % of the window's 99th
   percentile.
4. Each candidate is moved to the extremum of the band-passed ECG within ±75 ms, using the
   window's dominant QRS polarity (some WESAD leads are inverted). Every beat therefore uses the
   same fiducial point.

RR intervals are the differences between R-peaks. An RR is **rejected** if it lies outside
0.3–2.0 s (30–200 bpm) or deviates more than 20 % from the median of its 5 neighbours (missed,
extra or ectopic beat). Heart rate = 60 / RR over the clean intervals, which gives
`hr_mean`, `hr_std`, `hr_min` and `hr_max`.

## 5. HRV

HRV is computed **from the clean RR intervals, never from raw ECG samples** (`features.hrv_from_rr`):

- `hrv_sdnn`: sample standard deviation of the clean RR intervals, in ms.
- `hrv_rmssd`: root mean square of successive RR differences, in ms, using **only pairs of
  adjacent clean intervals**. A rejected beat never creates an artificial jump.

## 6. EDA features

EDA is resampled to 4 Hz (polyphase, anti-aliased, linear edge padding) and low-pass filtered
at 1 Hz. From each window:

- `eda_mean`, `eda_std`, `eda_min`, `eda_max`, `eda_range` (μS)
- `eda_slope`: linear trend in μS/s
- **Phasic activity**, from a simple filter decomposition: tonic = 0.05 Hz low-pass, phasic =
  EDA − tonic. This gives `eda_phasic_std` and `eda_scr_per_min`, where a skin conductance
  response is a phasic peak with prominence ≥ 0.02 μS, at least 1 s apart. This is a pragmatic
  approximation, not cvxEDA or Ledalab deconvolution.

## 7. Respiration features

The chest-band signal is resampled to 25 Hz and band-passed 0.1–0.7 Hz (6–42 breaths/min). A
**breath** is an inhalation peak at least 1.5 s after the previous one, with a prominence of at
least 30 % of the window's standard deviation, which rejects small ripples from speech or
movement. Breath-to-breath intervals between 1.5 and 15 s give the instantaneous rate
60 / interval:

- `resp_rate_mean`, `resp_rate_std`, `resp_rate_min`, `resp_rate_max` (breaths/min)
- `resp_interval_cv`: coefficient of variation of breath intervals (respiration-rate variability)

On WESAD baseline windows the peak-based rate was checked against the dominant spectral
frequency and agreed within about 1 breath/min. During the stress task, breathing is
speech-driven and irregular, which `resp_rate_std` and `resp_interval_cv` capture.

### Additional features (feature set `wesad-chest-v2`)

`wesad-chest-v1` is the original 19 features above. `wesad-chest-v2` keeps those 19, computed by
the identical code (the v1 columns were checked to be bit-for-bit identical to the archived
table), and adds 10 more. Each one is computable from the WESAD chest signals in every window:

| Feature | Definition | Why it is physiologically meaningful |
|---|---|---|
| `hr_median` | median instantaneous HR | robust centre; insensitive to a few residual beat errors |
| `hr_range` | max − min HR | size of within-window heart-rate swings (a tree cannot form this from min/max in one split) |
| `hr_cv` | HR SD / HR mean | variability relative to the person's level |
| `hrv_pnn50` | share of adjacent clean RR pairs differing by > 50 ms | standard time-domain vagal index, less outlier-driven than RMSSD |
| `eda_cv` | EDA SD / EDA mean | conductance variability relative to the person's skin conductance level |
| `eda_scr_amp_mean`, `eda_scr_amp_max` | mean / largest SCR prominence (0 if none) | strength, not just count, of sympathetic skin responses |
| `resp_amp_mean` | mean peak-to-trough depth of each complete breath (band-passed signal) | breathing depth |
| `resp_amp_cv` | breath-to-breath depth variability | speech and laughter make breathing depth irregular |
| `resp_ie_ratio` | mean inhalation / exhalation duration | speech and laughter shorten inhalation relative to exhalation |

Considered but **not** added: mean/median RR and RR SD. They are monotonic transforms of mean HR
and the same quantity as SDNN, so they add no information. Frequency-domain HRV (LF/HF) was also
left out, because it needs at least 2 minutes of RR data for a stable estimate.

Feature sets are registered in `config.FEATURE_SETS`. The model's metadata names its set, and
inference refuses a model whose list or order differs from the registered set.

## 8. Windows, sampling, filtering, invalid data

| | |
|---|---|
| Window (deployed model) | **90 s**, step **45 s** (50 % overlap); 30 s and 60 s were also evaluated |
| Window placement | entirely inside one contiguous labelled segment; a tail shorter than one window is dropped |
| Source sampling | 700 Hz (all chest signals and labels) |
| Working rates | ECG 700 Hz, EDA 4 Hz, Resp 25 Hz |

HRV and breathing-rate features need at least about a minute of signal to be stable; 90 s gives
more beats and breaths per window. The minimum number of clean RR intervals scales with the
window length (30 per 60 s). Overlap only multiplies windows within a subject; because splits are
by subject, it cannot leak between train and test.

**Window-length study.** 30 s (step 15 s), 60 s (step 30 s) and 90 s (step 45 s) windows were
compared in subject-wise CV on the development subjects (section 16). 90 s was selected. Its
advantage over 60 s was small (calibrated SVM macro F1 0.784 vs 0.779), and 30 s was slightly
worse. Feature tables: 30 s 2133 windows (9 discarded: 8 too few breaths, 1 abnormal RR); 60 s
1042; 90 s 677 (0 discarded).

A window is **discarded**, with its reason counted and logged in `preprocessing_report.json`, if
it has NaN/inf samples, a flat (disconnected) signal, an empty signal, fewer than 30 clean RR
intervals, more than 20 % rejected RR intervals, or fewer than 4 valid breaths. A **subject is
skipped** with a reason if its file is missing or unreadable, a signal is missing, or a signal's
length differs from the label array. `prepare_wesad` prints a warning if more than 10 % of
windows are discarded.

**WESAD result:** 15 subjects processed, 0 skipped. WESAD's chest recordings are clean: at 60 s
the largest per-window RR artifact fraction was 14 %, below the 20 % limit.

| Windowing | Windows kept (discarded) | Baseline | Stress | Amusement |
|---|---|---|---|---|
| 30 s / 15 s | 2133 (9) | 1151 | 635 | 347 |
| 60 s / 30 s | 1042 (0) | 566 | 311 | 165 |
| **90 s / 45 s** | **677 (0)** | 371 | 201 | 105 |

## 9. Subject-wise splitting and model selection

Windows of the same person are strongly correlated, so **no subject is ever in more than one of
train / validation / test**. A random `train_test_split` over windows is never used.

| Set | Subjects | Role |
|---|---|---|
| Train | S5, S7, S8, S11, S13, S14, S15, S16, S17 | development |
| Validation | S2, S3, S9 | development |
| **Test** | **S4, S6, S10** | held out, used **once** at the end |

The test subjects come from `GroupShuffleSplit` by subject (seed 42) and are fixed in
`config.TEST_SUBJECTS`, so the old and new models are judged on the same unseen people. They are
removed from every table before analysis, feature engineering or model selection
(`analyze.py`, `experiment.py` assert this).

- **Model selection** uses **leave-one-subject-out CV over the 12 development subjects**:
  12 folds, each fitting on 11 subjects and scoring the 12th. This is more stable than the single
  9 → 3 validation split, which is also reported as a secondary check.
- **Selection metrics:** primary is mean macro F1 over folds; secondary is balanced accuracy.
  Per-class recall is monitored. Raw accuracy is not used for selection because the classes are
  imbalanced.
- **Final model:** the chosen pipeline is refitted on all 12 development subjects and evaluated
  once on the test subjects.

**Leakage audit** (printed by `train`, stored in the metadata; all checks passed):

- train ∩ validation = ∅, train ∩ test = ∅, validation ∩ test = ∅
- the experiment (model, feature-set, window and normalisation choice) used no test subject
- scaling (LR/SVM) and class weights are fitted inside each training fold (sklearn `Pipeline`)
- no data-driven feature selection: fixed feature sets are compared as whole pipelines
- no resampling: class weighting only
- rest calibration uses only each subject's own first 5 minutes of rest

## 10. ML algorithm

**Deployed: RBF-kernel SVM** (`C=1`, `gamma="scale"`, class-balanced weights) on standardised,
rest-calibrated `wesad-chest-v2` features, with 90 s windows.

- **Scaling:** the `StandardScaler` is part of the fitted pipeline.
- **Probabilities:** from scikit-learn's Platt scaling (`probability=True`). The API reports the
  class with the highest probability, and test metrics are computed the same way.

Compared families (section 16): Random Forest, XGBoost, HistGradientBoosting, logistic regression
and RBF SVM, all class-balanced. In the calibrated setting the SVM was best in every window and
feature-set combination except one.

## 11. Evaluation

### Held-out test subjects S4, S6, S10 (evaluated once)

The new pipeline is scored on the 115 test windows that remain after each test subject's 5-minute
calibration period. The previous model (archived, fitted on the same 12 development subjects) is
scored on its own 60 s windows: all of them, and the same time span minus the calibration period.

| | Old (v1 RF, 60 s) | Old, same time span | **New (calibrated SVM, 90 s)** |
|---|---|---|---|
| Accuracy | 0.809 | 0.777 | **0.913** |
| Balanced accuracy | 0.682 | 0.678 | **0.868** |
| Macro F1 | 0.709 | 0.694 | **0.881** |
| Baseline recall | 0.965 | 0.95 | **0.962** |
| Stress recall | 0.810 | 0.81 | **0.976** |
| Amusement recall | 0.273 | 0.27 | **0.667** |
| Majority-class macro F1 | 0.234 | | 0.210 |

New model per class on test:

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Baseline | 0.879 | 0.962 | 0.919 | 53 |
| Stress | 1.000 | 0.976 | 0.988 | 41 |
| Amusement | 0.824 | 0.667 | 0.737 | 21 |

Confusion matrix (rows true, columns predicted; B / S / A):
`[[51, 0, 2], [0, 40, 1], [7, 0, 14]]`, saved as `body_vitals/results/confusion_matrix.png`.

**Read with care:** the test set is only 3 people and 21 amusement windows, and these 3 subjects
are easier than average for every model tried. Development-CV macro F1 was 0.784 and LOSO is
0.827, both below the test figure. Use LOSO as the headline.

### Leave-one-subject-out over all 15 subjects

| Pooled over 15 subjects | Old pipeline, same windows | **New pipeline** |
|---|---|---|
| Accuracy | 0.735 | **0.869** |
| Balanced accuracy | 0.667 | **0.821** |
| Macro F1 | 0.665 | **0.827** |
| Baseline recall | 0.762 | **0.947** |
| Stress recall | 0.923 | **0.896** |
| Amusement recall | 0.315 | **0.619** |

The old pipeline's original LOSO over all its windows was accuracy 0.750, balanced accuracy 0.651
and macro F1 0.652.

Per-subject macro F1 for the new pipeline is **0.808 ± 0.172**. Per-subject amusement recall is
0.62 ± 0.36: two subjects (S2, S8) get 0 % amusement recall, S15 is weak overall (macro F1 0.57),
and six subjects reach 100 % amusement recall. Per-subject numbers are in `metrics.json`.

The LOSO folds of the 12 development subjects are slightly optimistic, because the configuration
was chosen on CV over those same subjects (selection bias). The test-subject folds are not.

### The uncalibrated alternative (information only)

The best pipeline **without** calibration (Random Forest, `min_samples_leaf=3`, v1 features, 90 s)
was also scored on the test subjects after the deployment choice had been fixed:

- **Test:** accuracy 0.875, balanced accuracy 0.764, macro F1 0.782, amusement recall 0.33.
- **LOSO:** macro F1 0.704 and amusement recall 0.30, against 0.652 and 0.24 for the old model.

Without a personal resting reference, the improvements are real but modest.

## 12. Limitations

- **15 participants**, mostly young, healthy, male students, in one lab protocol. Do not expect
  this to generalise to other populations, ambulatory settings or other stressors.
- Labels are **experimental conditions, not felt emotions**.
- **Rest calibration requires a 5-minute seated rest** at the start of each session. Without it
  the deployed model cannot predict; it refuses rather than guess.
- **Protocol-order confound:** in WESAD the baseline is always first. Calibrated features
  therefore also partly encode slow drift since the start of the session, which can make baseline
  look easier to separate than it would be in free-living use.
- **Selection bias:** about 180 pipeline configurations were compared on 12 development subjects,
  so development-CV scores are optimistic. The 3-subject test set and LOSO mitigate this but are
  small.
- **Amusement remains the weakest class** (recall about 0.6), with large differences between
  people: for some subjects the signals show no amusement response the model can use.
- The phasic EDA and breath detection are simple filter/peak methods.
- The model is trained on the RespiBAN chest sensor. Re-validate before trusting it on other
  hardware.
- This is **not a medical device and not a diagnosis**.

## 13. WESAD labels vs the RM's original emotion categories

The RM's prototype mapped hand-set HR/HRV/GSR thresholds onto emotion words. WESAD provides no
such labels. This module therefore uses WESAD's own terms: **Baseline, Stress, Amusement**.
They are never renamed:

- Stress is **not** anxiety, anger or depression.
- Amusement is **not** "Happy".
- Baseline is **not** "Calm" or "Neutral emotion". It is the seated reading condition.

A model for other categories would need a dataset labelled with those categories.

## 14. One prediction pipeline and the API

The WESAD-trained model is the only thing that predicts. An earlier version had a separate,
hand-written arousal/valence formula next to it. It was removed, because it was not learned or
validated and it suggested two competing prediction systems. The model was never trained on
arousal or valence ratings, so the module reports none.

API response (`POST /body-vitals/demo` and `POST /body-vitals/predict`). The contract is unchanged
from the previous version; `model` gained fields and `calibration` is new:

```json
{
  "success": true,
  "predicted_state": "Stress",
  "confidence": 0.87,
  "probabilities": {"Baseline": 0.05, "Stress": 0.87, "Amusement": 0.08},
  "model": {"type": "SVM", "dataset": "WESAD", "classes": ["Baseline", "Stress", "Amusement"],
            "feature_version": "wesad-chest-v2", "window_sec": 90, "normalization": "rest_calibration",
            "note": "..."},
  "calibration": {"used": true, "source": "demo", "is_simulated": true, "windows": 5, "seconds": 300},
  "input": {"source": "demo", "is_simulated": true, "label": "SIMULATED / DEMO VALUES",
            "vitals": {"heart_rate_bpm": 92.5, "hrv_sdnn_ms": 55.0, "hrv_rmssd_ms": 34.0,
                       "gsr_us": 4.07, "respiration_rate_bpm": 17.0},
            "features": {"hr_mean": 92.5, "...": "all 29 features (absolute values), by name"},
            "signal_quality": {"...": "..."}},
  "inference_ms": 12.3
}
```

A rest-calibrated model needs the person's resting reference with every `/body-vitals/predict`
request, sent as one of:

- `"calibration_features": [{...}, ...]`: features of the rest windows
- `"calibration_signals": [{"ecg": [...], "eda": [...], "resp": [...], "fs": 700}, ...]`: raw rest
  windows

The reference is the per-feature median over those windows, computed exactly as in training.
Without it the API returns `400` with `kind = "calibration_required"`. Other errors return
`"success": false` with `kind` = `input` (400), `signal_quality` (422) or
`model_unavailable` (503).

The inference engine reads the feature set, window length and normalisation from the model's
metadata. The archived v1 model (`models/archive/`) therefore still loads through the same API:
copy it and its metadata back to `models/body_vitals_model.pkl` / `models/body_vitals_metadata.json`
to roll back.

## 15. Demo mode and real sensors

### Demo / simulation

No sensors are connected yet, so the page offers **Baseline / Stress / Amusement Demo**
buttons. `providers.DemoVitalsProvider` simulates one **person**, with their own resting heart
rate, HRV, skin conductance level and breathing rate. For a scenario it synthesises one window of
the model's length (90 s):

- a sum-of-Gaussians ECG at a target heart rate and HRV
- EDA with a tonic level, a trend and skin conductance responses
- a breath-by-breath respiration sinusoid

For the calibrated model it also generates **that same simulated person's resting windows** as
the reference: 5 windows, the number the training reference used for a 90 s window and 45 s step.
The target values are hand-set in the range of WESAD's per-condition medians. The waveforms go
through **the same feature extractor, calibration and trained model** as real data.

- Every demo response has `input.source = "demo"`, `input.is_simulated = true` and
  `calibration.is_simulated = true`. The page shows a yellow **SIMULATED / DEMO VALUES** banner,
  dashed vital tiles, and a note that the resting reference is simulated.
- Demo data is **not WESAD data and is never used for training**.
- The Amusement demo is not always predicted as Amusement: about a third of seeds in a quick check.
  This matches the model's real amusement recall of about 0.6, and the demo was not tuned to hide it.

`POST /body-vitals/predict` accepts external `features` or raw `signals` (plus calibration). Their
origin cannot be verified, so responses say `source: "api"`, `is_simulated: null`, "values supplied
by the caller".

### Adding a real sensor

Every provider returns **raw signals**, not features, so real data goes through the identical
pipeline. To add hardware:

```python
from body_vitals.providers import PhysiologicalInputProvider, SignalWindow

class RealVitalsProvider(PhysiologicalInputProvider):
    source = "sensor"
    is_simulated = False

    def __init__(self, device, window_sec=90):
        self.device = device            # your driver / SDK object
        self.window_sec = window_sec

    def read_window(self):
        ecg, fs_ecg = self.device.last_seconds("ecg", self.window_sec)     # your device's API
        eda, fs_eda = self.device.last_seconds("eda", self.window_sec)
        resp, fs_resp = self.device.last_seconds("resp", self.window_sec)
        return SignalWindow(ecg, eda, resp, fs_ecg, fs_eda, fs_resp,
                            source=self.source, is_simulated=False)
```

Session flow for the calibrated model:

1. Ask the person to sit still for 5 minutes and record it.
2. Cut that recording into `engine.window_sec` windows every `window_sec / 2` seconds that lie
   fully inside the 5 minutes. That gives 5 windows for 90 s.
3. Call `engine.predict_window(provider.read_window(), calibration_windows)` for every new window,
   and label the response `source: "sensor"`.
4. Record a new reference for each session, because sensor placement and skin conductance change
   between sessions.

Requirements:

- ECG at ≥ 250 Hz (higher is better for HRV)
- EDA in μS
- a respiration belt or other chest-movement signal
- windows of the model's length (90 s)

Any sampling rate works because signals are resampled internally. Re-validate on recordings
from the new sensor before trusting the predictions (see Limitations).

No hardware API is faked in this repository. **A laptop webcam cannot measure ECG, skin
conductance or respiration** and is not offered as a source.

## 16. Improvement study (how the deployed pipeline was chosen)

`python -m body_vitals.experiment` uses development subjects only, scored by leave-one-subject-out
CV over 12 subjects. All scores below are mean ± SD over the 12 folds.

**Stage A** compared 3 window lengths × 2 feature sets × 3 data variants × 5 model families with
default settings (90 pipelines). The data variants were: all windows; windows after the
calibration period, uncalibrated; and the same windows, rest-calibrated. Best of each kind:

| Pipeline | Macro F1 | Balanced acc. | Amusement recall |
|---|---|---|---|
| Current pipeline (RF, v1, 60 s, all windows) | 0.620 ± 0.123 | 0.645 | 0.19 |
| Best uncalibrated, all windows (SVM, v1, 90 s) | 0.665 ± 0.121 | 0.685 | 0.48 |
| Best uncalibrated, post-calibration windows (SVM, v1, 90 s) | 0.710 ± 0.144 | 0.740 | 0.63 |
| **Best rest-calibrated (SVM, v2, 90 s)** | **0.784 ± 0.181** | **0.793** | 0.58 |

**Stage B** ran small hyperparameter grids per family for the best uncalibrated and the best
calibrated data configuration:

- Random Forest: 12 settings; XGBoost: 12; HistGradientBoosting: 8; logistic regression: 6; SVM: 8.
- Uncalibrated, best per family: RF 0.671, SVM 0.668, XGBoost 0.657, LR 0.645, HGB 0.643.
- Calibrated, best per family: SVM 0.784 (`C=1`, `gamma=scale`), RF 0.728, HGB 0.709, XGBoost
  0.705, LR 0.676.
- **Class-imbalance check:** removing class weights from the winners dropped amusement recall from
  0.33 to 0.17 (uncalibrated) and from 0.58 to 0.42 (calibrated). Class weighting is kept.
  Resampling was not used, because adjacent windows are highly correlated and duplicating them
  mostly adds copies.
- **Fixed 9 → 3 validation split (secondary):** macro F1 0.613 for the current pipeline, 0.587 for
  the uncalibrated winner and 0.716 for the calibrated winner.

**Decision:** the calibrated pipeline was committed (`config.DEPLOY_TRACK`) **before** any test
subject was scored, because it had the largest and most consistent development-CV gain and a
realistic deployment path.

What helped, from the CV numbers:

- **Personal rest calibration** gave the largest gain: +0.07 to +0.08 macro F1 over the best
  uncalibrated pipeline on the same windows.
- **Model family:** the SVM benefited most from calibrated features.
- **Window length:** 60 → 90 s gave a small gain.
- **v2 features:** helped mainly in the calibrated setting (0.784 vs 0.708 for v1 at 90 s). Without
  calibration they gave no consistent gain.

## 17. Personal baseline / rest calibration

Between-person differences dominate the signals. In the development subjects, 93 % of EDA-level
variance is **between** people (section 18), so the same absolute value can mean rest for one
person and arousal for another. Rest calibration (`body_vitals/normalization.py`) works like this:

- **Reference:** the per-feature median over the person's **calibration windows**, meaning the
  windows that lie entirely inside the first **5 minutes of seated rest**.
- **Model input:** `feature − reference` for every feature.
- **No cross-person data:** each person is calibrated only with their own rest windows, never with
  any other person's data or with their stress or amusement data.
- **Calibration windows are excluded:** windows overlapping the calibration period are removed from
  training and from evaluation, so a model is never scored on the windows that defined its
  reference.
- **In WESAD:** the label is used only to locate the start of the rest period (baseline is always
  the first condition), which matches an app telling the user "sit still now". This applies to the
  held-out subjects too. **In deployment:** the 5-minute rest is recorded at session start, before
  any prediction.

Caveat: because WESAD's baseline always comes first, calibrated features also carry some
"time since start" information. See Limitations.

## 18. Why amusement is hard (analysis)

`python -m body_vitals.analyze`, on the 12 development subjects only:

- **Pooled across people, baseline and amusement look alike.** On every feature the
  baseline-vs-amusement AUC is between 0.29 and 0.56. Stress, by contrast, separates strongly from
  both (e.g. `hr_mean` AUC 0.07 for baseline vs stress).
- **Within a person, amusement is consistently more activated than baseline:**
  - `eda_scr_per_min` and `resp_rate_std` are higher in amusement in **12 of 12** subjects;
  - `eda_range`, `resp_rate_max`, `resp_interval_cv` and `eda_scr_amp_mean` in 11 of 12.
- **The effect is small relative to between-person differences.** For EDA level, 93 % of the
  variance is between subjects; for HRV, about 50 %.

So the problem was mainly **subject variability**, not missing features or class imbalance alone.

Pooled development-CV confusions:

| | Amusement → Baseline | Baseline → Amusement | Amusement recall | Amusement precision |
|---|---|---|---|---|
| Current pipeline | 88 / 132 | 47 / 453 | 0.19 | 0.29 |
| Best uncalibrated | 45 / 84 | 33 / 297 | 0.33 | 0.41 |
| **Calibrated SVM** | **23 / 84** | **9 / 213** | **0.58** | **0.65** |

Even calibrated, about a quarter of amusement windows are still called baseline, and for some
people (S2, S8 in LOSO) the model finds no usable amusement response. With these chest signals,
amusement remains the least reliable condition.

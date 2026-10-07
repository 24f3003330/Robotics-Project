# AI Emotion Music + xArm Robot Bottle Pick Demo

A Flask app for real-time emotion recognition from a **live webcam** (face) or a **live
microphone** (tone of voice). Each modality has its own page and its own pipeline. The
detected emotion plays mood music and can drive a real **xArm** robot over BLE to pick a
colour-coded bottle that matches the emotion.

No speech-to-text is used anywhere — voice emotion comes only from the acoustic tone of
the voice, never the words.

```
/         Landing page
/camera   Webcam ──► face detection ──► face emotion ──┐
                                                        ├─► final emotion (manual override wins)
/voice    Mic ──► VAD ──► HuBERT ──► voice emotion ────┘      ├─► Mood music
                                                              └─► Robot (BLE, optional)
/handwriting  canvas strokes ──► Transformer ──► EMOTHAW / 4-class prototype   (independent)
/body-vitals  ECG/EDA/Resp ──► features − personal rest reference ──► WESAD SVM ──► baseline / stress / amusement   (independent)
```

The webcam is opened only while the `/camera` page is open, and the microphone and HuBERT
model only for `/voice`. Leaving a page stops its pipeline.

## Features

- **Camera page** (`/camera`): live webcam facial emotion recognition. OpenCV's YuNet
  detector finds the face and the FER+ model classifies it into neutral, happy, surprise,
  sad, angry, disgust, fear or contempt, smoothed over ~1 s. Both are small ONNX files run
  by OpenCV itself (no extra package); they are downloaded once into `models/face/`.
- **Voice page** (`/voice`): continuous microphone capture with `sounddevice`; Silero VAD
  gates speech, ~2–4 s overlapping speech windows are classified by a HuBERT model
  (`superb/hubert-base-superb-er`), with temporal smoothing so the displayed emotion
  doesn't flicker. Predictions keep updating until you press Stop or leave the page.
- **Handwriting page** (`/handwriting`): write on a canvas with a mouse, touchscreen or stylus.
  A small Transformer trained on the **EMOTHAW** dataset predicts **Anxiety**, **Depression** and
  **Stress** independently, each normal or elevated, with a probability. It needs no camera and
  no microphone, and it is not connected to music or the robot. It needs a trained checkpoint;
  import the dataset and train with
  `python -m handwriting.import_emothaw && python -m handwriting.train` (see
  [`data/README.md`](data/README.md)).
  A separate **4-class prototype** (Happy / Neutral / Sad / Angry) can be selected on the same
  page. It is trained on **seed data derived from graphology references**, which is **not
  scientifically validated**. It only exists to build and test the 4-class pipeline until a
  real emotion-labelled handwriting dataset replaces it
  (`python -m handwriting.generate_seed_data && python -m handwriting.train --dataset seed_4class`).
  Graphology reference images can be collected from pages you list
  (`python -m handwriting.collect_graphology`, robots.txt-respecting) and must be approved by a
  human (`python -m handwriting.review_graphology`) before they can be used.
  Real trajectories come from the **Collect samples** tab on `/handwriting`: anonymous writer
  ID, chosen label, natural writing, saved to `data/handwriting/seed_4class/human/`. Then run
  `python -m handwriting.generate_seed_data --force && python -m handwriting.train --dataset
  seed_4class --sample-source human_canvas` for a writer-independent evaluation (see
  [`data/README.md`](data/README.md), B10).
- **Voice Age Estimation** (on `/voice`): an **estimated age group from voice** (Teens, Twenties,
  Thirties, Forties, Fifties, Sixties, Seventies+), shown in its own card, separate from voice
  emotion and gender. A frozen `facebook/hubert-base-ls960` encoder plus a small classifier is
  trained on **Common Voice 17.0 English**, whose age metadata is **self-reported and
  categorical**. Speakers are split 70/15/15 so no speaker is in more than one split. The model
  uses the same single microphone stream and VAD as emotion, needs about 3 s of speech per estimate
  ("Insufficient speech" otherwise) and averages recent windows. The data is read programmatically
  from the Hugging Face Hub, transferring only metadata and the selected clips' audio:
  `python scripts/inspect_voice_age_dataset.py` then `python scripts/train_voice_age.py`.
  APIs: `GET /voice/age/status`, `POST /voice/age`, and `age` in `/voice/status`. It is an
  approximate age **group**, not an exact or verified age; see [`docs/VOICE_AGE.md`](docs/VOICE_AGE.md)
  for the numbers and limitations.
- **Body Vitals page** (`/body-vitals`): heart rate and HRV from ECG R-peaks, skin conductance
  (GSR/EDA) and respiration are turned into 29 features per 90 s window, expressed relative to the
  person's own 5-minute resting reference ("rest calibration"), and classified by an RBF-kernel SVM
  trained on the **WESAD** dataset into WESAD's own conditions **Baseline / Stress / Amusement**
  (not renamed to emotions). Chosen by subject-wise cross-validation on 12 development subjects and
  evaluated once on 3 held-out subjects (test macro F1 0.88; leave-one-subject-out macro F1 0.83,
  per-subject 0.81 ± 0.17; amusement is still the hardest class, recall about 0.6). No sensors are
  connected yet, so the page runs a **demo mode** whose inputs (and resting reference) are labelled
  as simulated. Independent of every other modality: no camera, no microphone. Build it with
  `python -m body_vitals.prepare_wesad --data /path/to/WESAD --window-sec 90 --step-sec 45 --out
  data/body_vitals/wesad_features_w90_s45.csv && python -m body_vitals.train` (full procedure in
  [`docs/BODY_VITALS.md`](docs/BODY_VITALS.md)).
- **Multi-dataset HuBERT ensemble** (optional, see [`docs/ENSEMBLE.md`](docs/ENSEMBLE.md)):
  train per-corpus models (IEMOCAP, CREMA-D, RAVDESS, MSP-IMPROV, MSP-Podcast) and
  combine them with a learned ensemble. The app runs on a single pretrained model
  until you build one — nothing here is required to run the demo.
- **Manual override** buttons and a **fusion** layer (manual > voice > face) so exactly
  one "final emotion" is ever authoritative.
- **Robot control**: scan/connect to an xArm over BLE, run taught poses per emotion,
  or run everything in simulation mode with no robot connected.
- Debug telemetry throughout: per-window voice probabilities, per-ensemble-member
  breakdowns, live audio/VAD heartbeat logs.

## Requirements

- Python 3.11–3.12
- A webcam (camera page) and/or a microphone (voice page). On macOS, allow camera and
  microphone access for the app that runs Python (Terminal, VS Code, ...) in
  System Settings > Privacy & Security.
- An xArm robot reachable over BLE (optional — everything runs in simulation without one)

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/24f3003330/Robotics-Project.git
cd Robotics-Project

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open **http://localhost:5001** in a browser.

The first time the voice page is opened, the HuBERT model (`superb/hubert-base-superb-er`,
~380 MB) downloads from Hugging Face; after that it's cached and works offline.

### Optional: build the multi-dataset ensemble

The app works out of the box on a single pretrained model. To train and use the
per-dataset ensemble instead, see [`docs/ENSEMBLE.md`](docs/ENSEMBLE.md) and
[`requirements_ser_training.txt`](requirements_ser_training.txt).

## Project layout

```
app.py                        Flask app: routes, camera service, fusion, robot control
emotion_detector.py           Live voice pipeline: sounddevice capture, VAD, HuBERT, smoothing
face_emotion.py               Face detection (YuNet) + facial emotion (FER+) for the camera page
templates/base.html           Shared layout + navigation
templates/landing.html        Landing page (/)
templates/camera.html         Camera emotion page (/camera)
templates/voice.html          Voice emotion page (/voice)
templates/handwriting.html    Handwriting page (/handwriting): Anxiety / Depression / Stress
templates/body_vitals.html    Body Vitals page (/body-vitals): Baseline / Stress / Amusement
templates/_emotion_output.html  Final emotion, music, manual trigger, robot panel (both pages)
static/css/app.css            Styles
static/js/common.js           Shared UI helpers, music player, robot/manual controls
static/js/camera.js           Camera page: webcam lifecycle, frame polling, face emotion
static/js/voice.js            Voice page: start/stop, live status polling, gender + age cards
static/js/handwriting.js      Handwriting page: pointer capture (mouse/touch/stylus), results
static/js/body_vitals.js      Body Vitals page: demo buttons, results rendering
static/music/                 Mood-music clips per emotion
xarm_emotion_poses.json        Taught robot poses per emotion
handwriting/                   EMOTHAW importer, preprocessing, Transformer, training, inference;
                               4-class prototype: seed generator, image->trajectory, real-data importer
body_vitals/                   WESAD preprocessing, features (HR/HRV/EDA/resp), training, evaluation,
                               ML inference engine, input providers, Flask blueprint
voice_age/                     Voice age-group estimation: Common Voice access (HTTP range reads), labels,
                               speaker-wise split, preprocessing, frozen encoder, heads, live tracker
scripts/                       inspect_voice_age_dataset.py, train_voice_age.py
tests/                         python -m unittest discover tests
ser/                           Multi-dataset HuBERT ensemble: training & evaluation (optional)
  datasets/                    Per-corpus label mapping + parsing (IEMOCAP, CREMA-D, ...)
  runtime.py                   Ensemble inference — the interface app.py actually loads
docs/
  ENSEMBLE.md                 How to train/evaluate the ensemble
  ARCHITECTURE_AUDIT.md       Notes on the audio pipeline design
  VOICE_AGE.md                Voice age estimation: dataset, mapping, split, model, metrics, caveats
  BODY_VITALS.md              Body Vitals: WESAD, features, splits, metrics, limitations, sensors
```

## Notes

- The WESAD dataset is never committed: keep it outside the repo and pass `--data` (or set
  `WESAD_DIR`). Extracted features go to the gitignored `data/body_vitals/`.
- `data/` (datasets, e.g. EMOTHAW) is gitignored; see [`data/README.md`](data/README.md).
- `models/` (trained weights) and `runs/` (training artefacts) are gitignored — they're
  build outputs, not source.
- Robot control is entirely optional: with no xArm connected, robot actions run in
  simulation and are logged instead of sent over BLE.

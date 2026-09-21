# AI Emotion Music + xArm Robot Bottle Pick Demo

A Flask app that detects human emotion from a **live webcam / uploaded video** (face) and
a **live microphone / uploaded audio** (tone of voice), fuses the two into a single
"final emotion," plays mood music for it, and can drive a real **xArm** robot over BLE
to pick a colour-coded bottle that matches the emotion.

No speech-to-text is used anywhere — voice emotion comes only from the acoustic tone of
the voice, never the words.

```
Face (webcam / video)  ──┐
                          ├──► Fusion (manual > voice > face) ──► Mood music
Voice (mic / audio)    ──┘                                    └─► Robot (BLE, optional)
```

## Features

- **Live camera** face-emotion detection (OpenCV Haar cascade), or use an uploaded
  video as the camera source instead of the webcam.
- **Live microphone** voice-emotion detection: Silero VAD gates speech, ~2–4 s
  overlapping windows are classified by a fine-tuned HuBERT model, with temporal
  smoothing so the displayed emotion doesn't flicker.
- **Uploaded audio/video** analysis: the same acoustic model analyses the file's own
  audio track; the result stays pinned to that file until you pick another one.
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
- A webcam and microphone for the live demo (optional — uploaded files work without
  either)
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
pip install -r requirements_no_tf_robot.txt

# 4. Run the app
python app.py
```

Then open **http://localhost:5000** in a browser.

The first time voice detection starts, the HuBERT model (`superb/hubert-base-superb-er`,
~380 MB) downloads from Hugging Face; after that it's cached and works offline.

### Optional: build the multi-dataset ensemble

The app works out of the box on a single pretrained model. To train and use the
per-dataset ensemble instead, see [`docs/ENSEMBLE.md`](docs/ENSEMBLE.md) and
[`requirements_ser_training.txt`](requirements_ser_training.txt).

## Project layout

```
app.py                        Flask app: routes, camera loop, fusion, robot control
emotion_detector.py           Live/uploaded voice pipeline: capture, VAD, HuBERT, smoothing
templates/index.html          Single-page UI
static/music/                 Mood-music clips per emotion
xarm_emotion_poses.json        Taught robot poses per emotion
ser/                           Multi-dataset HuBERT ensemble: training & evaluation (optional)
  datasets/                    Per-corpus label mapping + parsing (IEMOCAP, CREMA-D, ...)
  runtime.py                   Ensemble inference — the interface app.py actually loads
docs/
  ENSEMBLE.md                 How to train/evaluate the ensemble
  ARCHITECTURE_AUDIT.md       Notes on the audio pipeline design
```

## Notes

- `uploads/`, `models/` (trained weights) and `runs/` (training artefacts) are
  gitignored — they're either user data or build outputs, not source.
- Robot control is entirely optional: with no xArm connected, robot actions run in
  simulation and are logged instead of sent over BLE.

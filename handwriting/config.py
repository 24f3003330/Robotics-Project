"""
handwriting/config.py - every tunable of the handwriting modality in one place.

The handwriting model is trained on EMOTHAW (Likforman-Sulem et al., IEEE THMS
2017) and predicts that dataset's own three targets, each an independent
binary output derived from the participant's DASS-42 score:

    Anxiety     normal / elevated
    Depression  normal / elevated
    Stress      normal / elevated

The three co-occur (a writer can be elevated on any combination), so they are
three heads, not one softmax. They are NOT mapped to camera/voice emotions
(Happy / Sad / Angry / Neutral).

A trained checkpoint stores the heads it was trained with; inference refuses a
checkpoint whose heads are not these three.

A second, separate experiment - the 4-class Happy / Neutral / Sad / Angry
pipeline (seed_4class now, real_4class later) - is configured further below.
It has its own checkpoint and route and never shares labels with EMOTHAW.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(PROJECT_DIR, "models", "handwriting_transformer.pt")

# ---------------------------------------------------------------------------
# EMOTHAW data layout (see data/README.md)
#   data/emothaw/raw/         the dataset exactly as received - never modified
#   data/emothaw/processed/   written by `python -m handwriting.import_emothaw`
# ---------------------------------------------------------------------------
EMOTHAW_DIR = os.path.join(PROJECT_DIR, "data", "emothaw")
EMOTHAW_RAW_DIR = os.path.join(EMOTHAW_DIR, "raw")
EMOTHAW_PROCESSED_DIR = os.path.join(EMOTHAW_DIR, "processed")

# ---------------------------------------------------------------------------
# Output heads. "dass_cutoff": a DASS-42 score ABOVE the cutoff is "elevated".
# Cutoffs are the upper bounds of the "normal" band of the DASS severity table
# (Lovibond & Lovibond; Table 1 of the EMOTHAW paper): depression 0-9,
# anxiety 0-7, stress 0-14. (The paper's text says "anxiety less than 7" but its
# table and the standard DASS scoring put the normal range at 0-7.)
# "baseline" is the class that means "nothing to report".
# Order = the order shown in the API and on the page.
# ---------------------------------------------------------------------------
EMOTHAW_HEADS = [
    {"name": "anxiety", "display": "Anxiety", "classes": ["normal", "elevated"],
     "baseline": "normal", "positive": "elevated", "dass_cutoff": 7},
    {"name": "depression", "display": "Depression", "classes": ["normal", "elevated"],
     "baseline": "normal", "positive": "elevated", "dass_cutoff": 9},
    {"name": "stress", "display": "Stress", "classes": ["normal", "elevated"],
     "baseline": "normal", "positive": "elevated", "dass_cutoff": 14},
]
HANDWRITING_HEADS = EMOTHAW_HEADS
HEAD_NAMES = [h["name"] for h in EMOTHAW_HEADS]

# EMOTHAW tasks: 1 pentagons, 2 house, 3 capital-letter words, 4/5 loops,
# 6 clock, 7 cursive sentence. The web page asks people to WRITE, so the default
# model is trained on the two writing tasks.
DEFAULT_TASKS = (3, 7)
ALL_TASKS = (1, 2, 3, 4, 5, 6, 7)

# ---------------------------------------------------------------------------
# Preprocessing (shared by training and inference - never change one without
# retraining; the checkpoint records the values it was trained with).
# ---------------------------------------------------------------------------
RESAMPLE_HZ = 50.0           # EMOTHAW tablet ~125 Hz, browsers 60-240 Hz -> one common rate
MAX_LEN = 512                # tokens per Transformer window (~10 s of writing at 50 Hz)
MIN_DT = 1e-3                # s; floor for dt so velocity never divides by zero
MAX_DT = 5.0                 # s; longer pauses are clipped
FEATURE_CLIP = 6.0           # standardised features are clipped to +/- this

# Per-point features. The kinematic ones exist for every input device.
KINEMATIC_FEATURES = [
    "x", "y",                  # position in the writing's bounding box (scale-free, 0..1)
    "dx", "dy",                # displacement since previous point (same units)
    "log_dt",                  # log seconds since previous point
    "log_speed",               # log1p(path speed)
    "log_accel",               # signed log1p(|d speed / dt|)
    "sin_heading", "cos_heading",
    "turn",                    # direction change since previous move, / pi
    "pen",                     # 1 = on paper / pressed, 0 = in air
    "stroke_start",            # 1 at the first point of every pen-down stroke
    "log_stroke_len",          # log1p(length of the stroke this point belongs to)
]
# Pen-only measurements. Mouse/trackpad cannot report them; when they are
# missing at inference the training mean is used (and the response says so).
PRESSURE_FEATURES = ["pressure"]
ALTITUDE_FEATURES = ["altitude"]
# Azimuth is imported and kept in the processed data, but it is not in the
# default feature set: EMOTHAW's tablet was used rotated by 90 deg and its
# azimuth zero direction / sense is undocumented, so it cannot be aligned with
# the browser's azimuthAngle reliably. Opt in with --features pen_azimuth.
AZIMUTH_FEATURES = ["sin_azimuth", "cos_azimuth"]

FEATURE_SETS = {
    "kinematic": KINEMATIC_FEATURES,
    "pressure": KINEMATIC_FEATURES + PRESSURE_FEATURES,
    "pen": KINEMATIC_FEATURES + PRESSURE_FEATURES + ALTITUDE_FEATURES,
    "pen_azimuth": KINEMATIC_FEATURES + PRESSURE_FEATURES + ALTITUDE_FEATURES + AZIMUTH_FEATURES,
}
DEFAULT_FEATURES = "pen"

# The Wacom tablet records in-air (hover) samples, so in-air movement is kept
# ("keep"); the browser sends mouse/stylus hover points the same way.
AIR_MODE = "keep"
DATASET_DISPLAY = "EMOTHAW (Likforman-Sulem et al., 2017)"

# ---------------------------------------------------------------------------
# 4-class handwriting pipeline (Happy / Neutral / Sad / Angry) - a SEPARATE
# experiment from EMOTHAW, with its own checkpoint, route and page panel.
# Neither dataset's labels are ever converted into the other's.
#
#   seed_4class  temporary PROTOTYPE data: graphology reference examples used as
#                heuristic seed data. Graphology is not a scientifically validated
#                source of emotion labels; a model trained on it is a prototype
#                that only exercises the pipeline.
#   real_4class  the placeholder for a real, emotion-labelled handwriting dataset.
#                Same processed layout, same model, same checkpoint path, same API.
# ---------------------------------------------------------------------------
FOUR_CLASS_LABELS = ["Happy", "Neutral", "Sad", "Angry"]
FOUR_CLASS_HEADS = [{"name": "emotion", "display": "Emotion", "classes": FOUR_CLASS_LABELS}]
FOUR_CLASS_CHECKPOINT = os.path.join(PROJECT_DIR, "models", "handwriting_4class.pt")
FOUR_CLASS_ROOT = os.path.join(PROJECT_DIR, "data", "handwriting")

SEED_DISCLAIMER = ("These samples are being used as heuristic seed data based on graphology references "
                   "and will be replaced with real emotion-labelled handwriting data. The seed dataset is "
                   "not scientifically validated and is intended only to establish and test the 4-class "
                   "pipeline.")
PROTOTYPE_NOTICE = ("Prototype trained on graphology-based seed data. Performance is not representative "
                    "of scientifically validated emotion recognition.")

# Human canvas collection (Collect tab on /handwriting): participants choose one
# of the four labels and write naturally; each sample is saved as JSON under
# data/handwriting/seed_4class/human/<label>/. Anonymous writer ids only.
HUMAN_CANVAS_DIR = os.path.join(FOUR_CLASS_ROOT, "seed_4class", "human")
WRITER_ID_PATTERN = r"^writer_\d{3,6}$"
HUMAN_PROTOTYPE_NOTICE = ("Prototype trained on participant-labelled canvas samples (the participant chose the "
                          "label). Performance is not representative of scientifically validated emotion "
                          "recognition.")
# origin of each processed sample (manifest column "sample_source"); augmented
# copies are additionally marked synthetic=1
SAMPLE_SOURCES = ("graphology", "human_canvas")

# Geometric features: what a static handwriting image can support. No timing
# (dt, speed, acceleration) and no pen channels - an image records none of them.
GEOMETRIC_FEATURES = ["x", "y", "dx", "dy", "sin_heading", "cos_heading", "turn",
                      "pen", "stroke_start", "log_stroke_len"]
FEATURE_SETS["geometric"] = GEOMETRIC_FEATURES

# Arc-length resampling ("resample": "arclength"): pen-down strokes are
# resampled at a fixed spacing of ARCLENGTH_STEP x the median stroke height, so
# image-reconstructed and canvas writing get the same point density whatever
# their pixel size or writing speed.
ARCLENGTH_STEP = 0.125

FOUR_CLASS_DATASETS = {
    "seed_4class": {
        "dir": os.path.join(FOUR_CLASS_ROOT, "seed_4class"),
        "display": "4-class SEED dataset (graphology references - not validated)",
        "dataset_type": "seed", "source_type": "graphology_reference",
        "scientifically_validated": False, "model_kind": "prototype",
        # images give geometry only -> geometric features, arc-length resampling,
        # in-air movement collapsed to the lift point (images contain none)
        "features": "geometric", "resample": "arclength", "air": "collapse",
        "allow_images": True, "require_writer_ids": False, "augment": 20,
    },
    "real_4class": {
        "dir": os.path.join(FOUR_CLASS_ROOT, "real_4class"),
        "display": "4-class emotion-labelled handwriting dataset",
        "dataset_type": "real", "source_type": None,          # set by the importer (--source)
        "scientifically_validated": None,                      # set by the importer (--labels-verified)
        "model_kind": "model",
        # online pen data: the full EMOTHAW-style features (override with --features / --resample)
        "features": "pen", "resample": "time", "air": "keep",
        "allow_images": False, "require_writer_ids": True, "augment": 0,
    },
}

# Minimal input before a prediction is attempted.
MIN_PEN_DOWN_POINTS = 20
MIN_TOKENS = 16
MAX_INPUT_POINTS = 200000

# Transformer (small on purpose: ~100k parameters, fast on CPU).
MODEL_DEFAULTS = {"d_model": 64, "n_heads": 4, "n_layers": 3, "dim_feedforward": 128,
                  "dropout": 0.2}

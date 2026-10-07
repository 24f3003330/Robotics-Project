"""
voice_age/config.py - every tunable of the voice AGE-GROUP estimation pipeline.

The model estimates a speaker's AGE GROUP from the voice. It is trained on
Common Voice, whose age metadata is SELF-REPORTED and CATEGORICAL (decades),
never an exact or verified age. It is completely separate from the voice
EMOTION model (superb/hubert-base-superb-er), which it never touches.

Environment variables override the defaults marked (env).
"""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Dataset. Mozilla withdrew Common Voice from the Hugging Face Hub in October
# 2025 (mozilla-foundation/common_voice_17_0 now only holds a README pointing to
# the Mozilla Data Collective). fixie-ai/common_voice_17_0 is a public,
# ungated Parquet re-upload of Common Voice 17.0 (CC0 licence), one folder
# per locale. Only the needed columns / row groups are read over HTTP.
# ---------------------------------------------------------------------------
AGE_DATASET = os.environ.get("VOICE_AGE_DATASET", "fixie-ai/common_voice_17_0")          # (env)
AGE_DATASET_VERSION = "Common Voice 17.0"
AGE_LANGUAGE = os.environ.get("VOICE_AGE_LANGUAGE", "en")                                   # (env)
AGE_SOURCE_SPLIT = "validated"          # every community-validated clip (superset of train/dev/test)
REQUIRED_COLUMNS = ("audio", "age", "client_id")
METADATA_COLUMNS = ("client_id", "age", "gender", "path")

DATA_DIR = os.path.join(PROJECT_DIR, "data", "voice_age")
METADATA_CACHE = os.path.join(DATA_DIR, "metadata_{language}_{split}.parquet")
SELECTION_FILE = os.path.join(DATA_DIR, "selection.csv")
AUDIO_CACHE_DIR = os.path.join(DATA_DIR, "audio")            # selected clips, original MP3 bytes, one file per row group
EMBEDDING_CACHE_DIR = os.path.join(DATA_DIR, "embeddings")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "voice_age_model")
RESULTS_DIR = os.path.join(PROJECT_DIR, "voice_age", "results")

# ---------------------------------------------------------------------------
# Age labels. Common Voice's own categories (note its spelling "fourties").
# Mapping raw label -> age group; labels mapped to None are excluded. Sparse
# decades are merged (see docs/VOICE_AGE.md for the counts that motivated it).
# ---------------------------------------------------------------------------
AGE_GROUP_MAPPING = {
    "teens": "Teens",
    "twenties": "Twenties",
    "thirties": "Thirties",
    "fourties": "Forties",
    "forties": "Forties",
    "fifties": "Fifties",
    "sixties": "Sixties",
    "seventies": "Seventies+",
    "eighties": "Seventies+",
    "nineties": "Seventies+",
}
AGE_GROUPS = ["Teens", "Twenties", "Thirties", "Forties", "Fifties", "Sixties", "Seventies+"]
MIN_SPEAKERS_PER_GROUP = 60          # a group with fewer speakers is merged / refused (no near-empty classes)

# ---------------------------------------------------------------------------
# Subset size. Per age group, at most this many clips in total (all splits);
# each speaker contributes at most AGE_MAX_CLIPS_PER_SPEAKER clips so the model
# learns age, not a few prolific voices.
# ---------------------------------------------------------------------------
AGE_MAX_SAMPLES_PER_CLASS = int(os.environ.get("VOICE_AGE_MAX_SAMPLES_PER_CLASS", "1400"))  # (env)
AGE_MAX_CLIPS_PER_SPEAKER = int(os.environ.get("VOICE_AGE_MAX_CLIPS_PER_SPEAKER", "3"))     # (env)

# Speaker-wise split fractions (speakers, not clips)
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Audio preprocessing: mono, 16 kHz, float32, leading/trailing silence trimmed.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
MIN_CLIP_SECONDS = 1.0               # after trimming; shorter = skipped (too little speech)
MAX_CLIP_SECONDS = 6.0               # longer clips are cut (centre) - bounded compute
SILENCE_DBFS = -50.0                 # whole clip below this RMS = silent, skipped
TRIM_DBFS = -40.0                    # frame energy threshold for edge trimming (relative to clip peak)

# ---------------------------------------------------------------------------
# Model: frozen pretrained speech encoder -> mean-pooled hidden states of one
# layer -> lightweight classifier. The encoder is never fine-tuned.
# ---------------------------------------------------------------------------
AGE_ENCODER = os.environ.get("VOICE_AGE_ENCODER", "facebook/hubert-base-ls960")            # (env)
EMBED_BATCH_SECONDS = 60.0           # audio seconds per encoder batch while extracting embeddings

# ---------------------------------------------------------------------------
# Real-time inference (shares the /voice microphone stream and VAD).
# ---------------------------------------------------------------------------
AGE_INFERENCE_WINDOW_SECONDS = float(os.environ.get("VOICE_AGE_WINDOW_SECONDS", "3.0"))   # (env) speech per window
AGE_MIN_SPEECH_SECONDS = 2.0         # a phrase ending earlier than this is "insufficient speech"
AGE_UTTERANCE_END_SECONDS = 0.7      # silence that ends a phrase
AGE_SMOOTHING_WINDOWS = int(os.environ.get("VOICE_AGE_SMOOTHING_WINDOWS", "5"))           # (env) probability average
AGE_RESET_SILENCE_SECONDS = 8.0      # long silence: the next speaker may be someone else
AGE_DEVICE = os.environ.get("VOICE_AGE_DEVICE") or None                                   # (env) cpu / mps / cuda

UI_LABEL = "Estimated age group from voice"
DISCLAIMER = ("Approximate estimate of an age GROUP from the voice, learned from self-reported Common Voice "
              "metadata. Not an exact or verified age; affected by microphone, accent and speaking style. "
              "Do not use for identity verification or any high-stakes decision.")

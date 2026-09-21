"""Canonical 4-class label space and the explicit per-dataset mappings into it.

Nothing here guesses. Every dataset states, label by label, which of its own
emotion codes becomes which canonical class and which ones are DROPPED. A code
that is not listed is an error, not a silent drop - a new IEMOCAP release or a
misparsed filename must fail loudly rather than quietly relabel data.

The canonical order is identical to `emotion_detector.EMOTIONS`, because the
fine-tuned heads emit logits in this order and the live app indexes them by it.
"""

from __future__ import annotations

# MUST stay identical (order included) to emotion_detector.EMOTIONS.
CANONICAL_EMOTIONS = ("happy", "angry", "sad", "neutral")
EMOTION_TO_INDEX = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}

DROP = None  # an emotion we deliberately do not model


class LabelError(ValueError):
    """An emotion code that no dataset mapping accounts for."""


# ----------------------------------------------------------------- IEMOCAP
# Codes as they appear in Session*/dialog/EmoEvaluation/*.txt.
# "exc" (excitement) is merged into happy: this is the standard 4-class IEMOCAP
# protocol (Busso et al. define excitement as high-arousal positive, and nearly
# all published 4-class numbers merge it). Set merge_excitement=False to drop it
# instead - the two protocols are not comparable, so the choice is recorded in
# the split manifest.
IEMOCAP_LABELS = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",      # merged - see above
    "sad": "sad",
    "neu": "neutral",
    "fru": DROP,         # frustration: high-arousal negative, not angry
    "fea": DROP,
    "sur": DROP,
    "dis": DROP,
    "oth": DROP,
    "xxx": DROP,         # annotators did not agree
}

# ----------------------------------------------------------------- CREMA-D
# From the filename: 1001_DFA_ANG_XX.wav -> speaker 1001, emotion ANG.
CREMAD_LABELS = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "DIS": DROP,
    "FEA": DROP,
}

# ----------------------------------------------------------------- RAVDESS
# Third filename field: 03-01-05-02-01-01-12.wav -> emotion 05.
# "calm" (02) is dropped by default. It is acted low-arousal neutral-valence
# speech with no counterpart in the other four datasets; folding it into neutral
# inflates neutral recall with a voice quality nothing else in the ensemble has
# seen. Set include_calm=True to map it to neutral instead.
RAVDESS_LABELS = {
    "01": "neutral",
    "02": DROP,          # calm - see above
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": DROP,          # fearful
    "07": DROP,          # disgust
    "08": DROP,          # surprised
}

# ----------------------------------------------------------------- MSP-IMPROV
# Single-letter codes in Evalution.txt.
MSP_IMPROV_LABELS = {
    "A": "angry",
    "H": "happy",
    "S": "sad",
    "N": "neutral",
    "O": DROP,           # other
    "X": DROP,           # no agreement
}

# ----------------------------------------------------------------- MSP-Podcast
# EmoClass column of Labels/labels_consensus.csv.
MSP_PODCAST_LABELS = {
    "A": "angry",
    "H": "happy",
    "S": "sad",
    "N": "neutral",
    "U": DROP,           # surprise
    "F": DROP,           # fear
    "D": DROP,           # disgust
    "C": DROP,           # contempt
    "O": DROP,           # other
    "X": DROP,           # no agreement
}

DATASET_LABEL_MAPS = {
    "iemocap": IEMOCAP_LABELS,
    "cremad": CREMAD_LABELS,
    "ravdess": RAVDESS_LABELS,
    "msp_improv": MSP_IMPROV_LABELS,
    "msp_podcast": MSP_PODCAST_LABELS,
}


def map_label(dataset, code, overrides=None):
    """Raw dataset emotion code -> canonical class, or None if deliberately dropped.

    Raises LabelError for a code the dataset's map does not mention at all.
    """
    table = DATASET_LABEL_MAPS.get(dataset)
    if table is None:
        raise LabelError(f"no label map registered for dataset {dataset!r}")
    if overrides:
        table = dict(table, **overrides)
    key = str(code).strip()
    if key not in table:
        raise LabelError(
            f"{dataset}: emotion code {code!r} is not in the label map. Add it "
            f"explicitly (to a canonical class or to DROP) in ser/labels.py - "
            f"unknown codes are never dropped silently."
        )
    return table[key]


def describe_mapping(dataset, overrides=None):
    """Human-readable table of how one dataset maps, for the run manifest."""
    table = DATASET_LABEL_MAPS[dataset]
    if overrides:
        table = dict(table, **overrides)
    kept = {k: v for k, v in table.items() if v is not None}
    dropped = sorted(k for k, v in table.items() if v is None)
    return {"kept": kept, "dropped": dropped}

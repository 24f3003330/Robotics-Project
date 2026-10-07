"""CREMA-D: 7442 clips, 91 actors, emotion and speaker both encoded in the filename.

    AudioWAV/1001_DFA_ANG_XX.wav
            ^^^^ actor  ^^^ emotion
"""

from __future__ import annotations

import glob
import os

from ..labels import map_label
from .base import ScanError, Utterance, require_dir

NAME = "cremad"


def scan(root, **_opts):
    root = require_dir(root, NAME, "AudioWAV/ (or the .wav files directly)")
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if not files:
        raise ScanError(f"{NAME}: no .wav files under {root}")

    utterances = []
    for path in files:
        parts = os.path.splitext(os.path.basename(path))[0].split("_")
        if len(parts) < 3:
            continue  # not a CREMA-D clip name
        actor, _sentence, emo = parts[0], parts[1], parts[2].upper()
        label = map_label(NAME, emo)
        if label is None:
            continue
        utterances.append(Utterance(
            path=path, label=label, speaker=f"{NAME}:{actor}",
            dataset=NAME, raw_label=emo,
        ))
    if not utterances:
        raise ScanError(f"{NAME}: found .wav files under {root} but none had CREMA-D style names")
    return utterances

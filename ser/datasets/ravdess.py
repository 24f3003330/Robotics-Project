"""RAVDESS: 24 actors, everything encoded in a 7-field filename.

    Actor_12/03-01-05-02-01-01-12.wav
             ^^ modality  ^^ emotion    ^^ actor
              ^^ vocal channel

Only modality 03 (audio-only) is usable here; 01/02 are video. Song (vocal
channel 02) is excluded by default - sung emotion has a prosody the live
microphone will never see.
"""

from __future__ import annotations

import glob
import os

from ..labels import map_label
from .base import ScanError, Utterance, require_dir

NAME = "ravdess"


def scan(root, include_song=False, include_calm=False, **_opts):
    root = require_dir(root, NAME, "Actor_01/ ... Actor_24/")
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if not files:
        raise ScanError(f"{NAME}: no .wav files under {root}")

    overrides = {"02": "neutral"} if include_calm else None
    utterances = []
    for path in files:
        parts = os.path.splitext(os.path.basename(path))[0].split("-")
        if len(parts) != 7:
            continue
        modality, channel, emo, _intensity, _statement, _repeat, actor = parts
        if modality != "03":            # audio-only
            continue
        if channel == "02" and not include_song:
            continue
        label = map_label(NAME, emo, overrides=overrides)
        if label is None:
            continue
        utterances.append(Utterance(
            path=path, label=label, speaker=f"{NAME}:{actor}",
            dataset=NAME, raw_label=emo,
            session="song" if channel == "02" else "speech",
        ))
    if not utterances:
        raise ScanError(f"{NAME}: no audio-only RAVDESS clips with modelled emotions under {root}")
    return utterances

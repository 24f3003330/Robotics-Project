"""MSP-Podcast: a single consensus CSV, and an official speaker-disjoint split.

    Labels/labels_consensus.csv
    Audios/MSP-PODCAST_0001_0001.wav

Columns: FileName, EmoClass, EmoAct, EmoVal, EmoDom, SpkrID, Gender, Split_Set.

The release's own Train / Development / Test1 splits are already speaker
disjoint and are what published numbers are measured on, so they are honoured
by default (split_hint) instead of being re-cut. Clips whose SpkrID is
"Unknown" cannot be kept out of each other's splits, so they are dropped unless
keep_unknown_speakers=True.
"""

from __future__ import annotations

import csv
import glob
import os

from ..labels import map_label
from .base import ScanError, Utterance, require_dir

NAME = "msp_podcast"

_SPLIT_HINTS = {
    "train": "train", "development": "val", "validation": "val", "dev": "val",
    "test1": "test", "test": "test",
}


def scan(root, keep_unknown_speakers=False, **_opts):
    root = require_dir(root, NAME, "Labels/labels_consensus.csv and Audios/")
    csv_hits = sorted(glob.glob(os.path.join(root, "**", "labels_consensus.csv"), recursive=True))
    if not csv_hits:
        raise ScanError(f"{NAME}: no Labels/labels_consensus.csv under {root}")
    label_csv = csv_hits[0]

    audio_index = {}
    for wav in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        audio_index[os.path.basename(wav)] = wav
    if not audio_index:
        raise ScanError(f"{NAME}: no .wav files under {root}")

    utterances, missing_audio, unknown_speaker = [], 0, 0
    with open(label_csv, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            row = { (k or "").strip(): (v or "").strip() for k, v in row.items() }
            name = row.get("FileName")
            code = (row.get("EmoClass") or "").upper()
            if not name or not code:
                continue
            label = map_label(NAME, code)
            if label is None:
                continue
            speaker = row.get("SpkrID") or "Unknown"
            if speaker.lower() in ("unknown", "", "nan") and not keep_unknown_speakers:
                unknown_speaker += 1
                continue
            path = audio_index.get(name) or audio_index.get(os.path.basename(name))
            if path is None:
                missing_audio += 1
                continue
            split_set = (row.get("Split_Set") or "").strip().lower().replace(" ", "")
            utterances.append(Utterance(
                path=path, label=label, speaker=f"{NAME}:{speaker}",
                dataset=NAME, raw_label=code,
                split_hint=_SPLIT_HINTS.get(split_set, ""),
            ))
    if not utterances:
        raise ScanError(f"{NAME}: {label_csv} parsed but produced no usable utterance")
    if unknown_speaker:
        print(f"[{NAME}] {unknown_speaker} clips dropped: SpkrID is Unknown, so they cannot be "
              f"held speaker-disjoint (pass --keep-unknown-speakers to keep them)")
    if missing_audio:
        print(f"[{NAME}] {missing_audio} labelled clips had no audio file - skipped")
    return utterances

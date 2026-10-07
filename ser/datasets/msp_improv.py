"""MSP-IMPROV: one consensus label file for the whole corpus.

    Evalution.txt            (spelled that way in the release)
    Audio/session1/S01A/S/MSP-IMPROV-S01A-F01-S-FM01.wav

Label lines look like:

    UTD-IMPROV-S01A-F01-S-FM01.avi; N; A:3.000000; V:3.500000; D:3.000000;

Twelve speakers, F01..F06 and M01..M06, one pair per session - the speaker is
the 4th dash-field of the clip id.
"""

from __future__ import annotations

import glob
import os

from ..labels import map_label
from .base import ScanError, Utterance, require_dir

NAME = "msp_improv"


def _find_label_file(root):
    for pattern in ("Evalution.txt", "Evaluation.txt", "*/Evalution.txt", "*/Evaluation.txt"):
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[0]
    return None


def scan(root, **_opts):
    root = require_dir(root, NAME, "Evalution.txt and Audio/session*/")
    label_file = _find_label_file(root)
    if label_file is None:
        raise ScanError(f"{NAME}: no Evalution.txt / Evaluation.txt under {root}")

    wav_index = {}
    for wav in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        wav_index[os.path.splitext(os.path.basename(wav))[0]] = wav
    if not wav_index:
        raise ScanError(f"{NAME}: no .wav files under {root}")

    utterances, missing_audio = [], 0
    with open(label_file, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            fields = [f.strip() for f in line.strip().split(";") if f.strip()]
            if len(fields) < 2 or not fields[0].lower().endswith((".avi", ".wav")):
                continue
            clip_id = os.path.splitext(fields[0])[0]
            code = fields[1].upper()
            label = map_label(NAME, code)
            if label is None:
                continue
            parts = clip_id.split("-")
            if len(parts) < 5:
                raise ScanError(f"{NAME}: unexpected clip id {clip_id!r} in {label_file}")
            session, speaker = parts[2], parts[3]   # S01A, F01
            path = wav_index.get(clip_id)
            if path is None:
                # the audio release names files MSP-IMPROV-..., the labels UTD-IMPROV-...
                path = wav_index.get(clip_id.replace("UTD-IMPROV", "MSP-IMPROV"))
            if path is None:
                missing_audio += 1
                continue
            utterances.append(Utterance(
                path=path, label=label, speaker=f"{NAME}:{speaker}",
                dataset=NAME, raw_label=code, session=session,
            ))
    if not utterances:
        raise ScanError(f"{NAME}: {label_file} parsed but no labelled clip matched an audio file")
    if missing_audio:
        print(f"[{NAME}] {missing_audio} labelled clips had no audio file - skipped")
    return utterances

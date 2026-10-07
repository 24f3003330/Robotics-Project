"""IEMOCAP: categorical labels live in the per-dialog EmoEvaluation transcripts.

    Session1/dialog/EmoEvaluation/Ses01F_impro01.txt
    Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav

An EmoEvaluation line looks like:

    [6.2901 - 8.2357]\tSes01F_impro01_F000\tneu\t[2.5000, 2.5000, 2.5000]

There are exactly 10 speakers, two per session. The speaker is NOT the dialog
name: in Ses01F_impro01 the "F" is the session's lead actor, while the final
filename field (F000 / M012) says who actually speaks that turn. Using the
dialog name would put both actors of a session under one id and quietly break
speaker-disjoint splitting, so the turn's own gender character is used.
"""

from __future__ import annotations

import glob
import os
import re

from ..labels import map_label
from .base import ScanError, Utterance, require_dir

NAME = "iemocap"

_LINE = re.compile(r"^\[[\d.]+\s*-\s*([\d.]+)\]\s+(\S+)\s+(\w+)\s")
_TURN = re.compile(r"^Ses(\d+)([FM])_\w+?_([FM])\d+$")


def _speaker_of(turn_id):
    m = _TURN.match(turn_id)
    if not m:
        return None
    session, _lead, speaking = m.groups()
    return f"Ses{session}{speaking}"       # e.g. Ses01F - 10 of these in total


def scan(root, merge_excitement=True, **_opts):
    root = require_dir(root, NAME, "Session1/ ... Session5/")
    overrides = None if merge_excitement else {"exc": None}

    eval_files = sorted(glob.glob(
        os.path.join(root, "Session*", "dialog", "EmoEvaluation", "*.txt")))
    if not eval_files:
        raise ScanError(f"{NAME}: no Session*/dialog/EmoEvaluation/*.txt under {root}")

    wav_index = {}
    for wav in glob.glob(os.path.join(root, "Session*", "sentences", "wav", "*", "*.wav")):
        wav_index[os.path.splitext(os.path.basename(wav))[0]] = wav

    utterances, missing_audio = [], 0
    for eval_file in eval_files:
        session = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(eval_file))))
        with open(eval_file, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("["):
                    continue                     # per-annotator detail lines
                m = _LINE.match(line)
                if not m:
                    continue
                end_s, turn_id, code = m.groups()
                label = map_label(NAME, code, overrides=overrides)
                if label is None:
                    continue
                speaker = _speaker_of(turn_id)
                if speaker is None:
                    raise ScanError(f"{NAME}: cannot read a speaker out of turn id {turn_id!r} "
                                    f"in {eval_file}; refusing to guess.")
                path = wav_index.get(turn_id)
                if path is None:
                    missing_audio += 1
                    continue
                utterances.append(Utterance(
                    path=path, label=label, speaker=f"{NAME}:{speaker}",
                    dataset=NAME, raw_label=code, session=session,
                ))
    if not utterances:
        raise ScanError(f"{NAME}: parsed {len(eval_files)} EmoEvaluation files but matched no audio")
    if missing_audio:
        print(f"[{NAME}] {missing_audio} labelled turns had no .wav under sentences/wav - skipped")
    return utterances

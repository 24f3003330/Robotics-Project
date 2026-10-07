"""Common shape every dataset scanner returns."""

from __future__ import annotations

import dataclasses
import os
from collections import Counter


@dataclasses.dataclass(frozen=True)
class Utterance:
    """One labelled audio file, already mapped into the canonical label space."""

    path: str               # absolute path to the audio
    label: str              # canonical: happy | angry | sad | neutral
    speaker: str            # GLOBALLY unique: "<dataset>:<speaker id>"
    dataset: str
    raw_label: str          # the dataset's own code, kept for auditing
    session: str = ""       # recording session//show, where the dataset has one
    split_hint: str = ""    # official split, when the dataset ships one
    duration: float = 0.0   # filled in lazily; 0.0 = not measured yet


class ScanError(RuntimeError):
    """The dataset root does not look like the dataset we were told it is."""


def require_dir(path, dataset, expected):
    if not path or not os.path.isdir(path):
        raise ScanError(f"{dataset}: {path!r} is not a directory. Expected a root containing {expected}.")
    return os.path.abspath(path)


def summarise(utterances):
    """Counts used in logs and in the split manifest."""
    return {
        "utterances": len(utterances),
        "speakers": len({u.speaker for u in utterances}),
        "per_class": dict(sorted(Counter(u.label for u in utterances).items())),
        "per_raw_label": dict(sorted(Counter(u.raw_label for u in utterances).items())),
    }

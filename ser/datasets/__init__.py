"""Dataset scanners. Each returns a list of canonical `Utterance` records."""

from . import cremad, iemocap, msp_improv, msp_podcast, ravdess
from .base import ScanError, Utterance, summarise

SCANNERS = {
    "iemocap": iemocap.scan,
    "cremad": cremad.scan,
    "ravdess": ravdess.scan,
    "msp_improv": msp_improv.scan,
    "msp_podcast": msp_podcast.scan,
}

DATASET_NAMES = tuple(SCANNERS)


def scan(dataset, root, **opts):
    if dataset not in SCANNERS:
        raise ScanError(f"unknown dataset {dataset!r}; known: {', '.join(DATASET_NAMES)}")
    return SCANNERS[dataset](root, **opts)


__all__ = ["SCANNERS", "DATASET_NAMES", "Utterance", "ScanError", "scan", "summarise"]

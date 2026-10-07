"""Multi-dataset HuBERT speech-emotion ensemble for the robotics app.

Training/evaluation lives here; the app only ever touches `ser.runtime`, which
exposes an object with the same `predict(audio) -> {emotion: prob}` interface
the single model had.
"""

from .labels import CANONICAL_EMOTIONS

__all__ = ["CANONICAL_EMOTIONS"]
__version__ = "1.0.0"

"""
voice_age/audio.py - audio preprocessing for age estimation (training AND live).

    encoded bytes / array -> mono -> 16 kHz -> float32 -> reject silence ->
    trim leading / trailing silence -> min / max duration

Every rejection raises AudioError with a short `reason` so callers can skip and
count the clip instead of crashing.
"""

import io
from fractions import Fraction

import numpy as np

from . import config


class AudioError(ValueError):
    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def decode(data):
    """Encoded audio bytes (MP3 / WAV / FLAC / OGG) -> (float32 samples [n] or [n, ch], rate)."""
    import soundfile as sf
    if not data:
        raise AudioError("empty_audio", "no audio bytes")
    try:
        x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioError("decode_failed", f"{type(exc).__name__}: {exc}") from exc
    return x, rate


def to_mono_16k(x, rate):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)                         # stereo / multichannel -> mono
    if x.ndim != 1 or x.size == 0:
        raise AudioError("empty_audio", "no samples")
    if not np.all(np.isfinite(x)):
        raise AudioError("nonfinite_audio", "NaN / inf samples")
    if rate != config.SAMPLE_RATE:
        from scipy.signal import resample_poly
        ratio = Fraction(config.SAMPLE_RATE, int(rate)).limit_denominator(1000)
        x = resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)
    return x


def dbfs(x):
    rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
    return 20.0 * np.log10(max(rms, 1e-10))


def trim_silence(x, frame_seconds=0.02, rel_db=config.TRIM_DBFS):
    """Drop leading / trailing frames more than |rel_db| dB below the loudest frame."""
    n = int(frame_seconds * config.SAMPLE_RATE)
    if len(x) < 2 * n:
        return x
    frames = len(x) // n
    energy = np.sqrt(np.mean(x[:frames * n].reshape(frames, n) ** 2, axis=1)) + 1e-10
    loud = np.flatnonzero(20 * np.log10(energy / energy.max()) > rel_db)
    if loud.size == 0:
        return x[:0]
    return x[loud[0] * n:(loud[-1] + 1) * n]


def prepare(x, rate=config.SAMPLE_RATE, min_seconds=config.MIN_CLIP_SECONDS, max_seconds=config.MAX_CLIP_SECONDS,
            trim=True):
    """Samples -> standardized 16 kHz mono float32 clip, or AudioError."""
    x = to_mono_16k(x, rate)
    if dbfs(x) < config.SILENCE_DBFS:
        raise AudioError("silent", f"{dbfs(x):.0f} dBFS")
    if trim:
        x = trim_silence(x)
    seconds = len(x) / config.SAMPLE_RATE
    if seconds < min_seconds:
        raise AudioError("too_short", f"{seconds:.2f} s < {min_seconds} s")
    limit = int(max_seconds * config.SAMPLE_RATE) if np.isfinite(max_seconds) else len(x)
    if len(x) > limit:
        start = (len(x) - limit) // 2               # centre crop: drops onset / offset artefacts
        x = x[start:start + limit]
    return np.ascontiguousarray(x, dtype=np.float32)


def prepare_bytes(data, **kw):
    x, rate = decode(data)
    return prepare(x, rate, **kw)

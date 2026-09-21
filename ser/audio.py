"""Audio loading and the ONE window specification shared by training and runtime.

The window spec is deliberately in one place. Training on whole 12 s utterances
and then classifying 3 s live windows is a silent train/test mismatch: HuBERT's
pooled representation of 12 s of speech is not the representation of 3 s of it,
and the model's confidence calibration - which the ensemble depends on - is
fitted to whatever length it saw. So every path here cuts the same 2-4 s
windows with the same overlap.

VAD's job stops at "which samples are speech". It never sees a classifier and
the classifier never sees a 32 ms VAD frame: frames are only used to decide
which spans of audio become a window.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess

import numpy as np

SAMPLE_RATE = 16000


@dataclasses.dataclass(frozen=True)
class WindowSpec:
    """How a stretch of speech becomes model inputs.

    window_s / hop_s default to the live detector's own geometry (3 s of audio
    re-examined every 1.5 s => 50 % overlap), so a training window and a live
    window are the same kind of object.
    """

    window_s: float = 3.0
    hop_s: float = 1.5
    min_s: float = 2.0            # never hand the model less than this
    min_speech_ratio: float = 0.5  # a window must be at least this much speech

    def as_dict(self):
        return dataclasses.asdict(self)

    @property
    def window_samples(self):
        return int(round(self.window_s * SAMPLE_RATE))

    @property
    def hop_samples(self):
        return max(1, int(round(self.hop_s * SAMPLE_RATE)))

    @property
    def min_samples(self):
        return int(round(self.min_s * SAMPLE_RATE))


DEFAULT_WINDOW_SPEC = WindowSpec()


def _load_with_soundfile(path):
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(rate)


def _load_with_ffmpeg(path):
    """Decode anything (mp3/flac/m4a/avi/...) with the bundled ffmpeg, to raw 16 kHz mono."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:                       # pragma: no cover - environment dependent
        raise RuntimeError(f"cannot decode {path}: no soundfile and no ffmpeg ({exc})")
    cmd = [exe, "-v", "error", "-i", path, "-vn", "-f", "f32le",
           "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
        raise RuntimeError(f"ffmpeg could not decode {path}: {' | '.join(tail)}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), SAMPLE_RATE


def resample_to_16k(audio, rate):
    if rate == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    duration = len(audio) / float(rate)
    target_n = int(round(duration * SAMPLE_RATE))
    if target_n <= 1 or len(audio) <= 1:
        return np.zeros(max(target_n, 0), dtype=np.float32)
    src = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    dst = np.linspace(0.0, duration, num=target_n, endpoint=False)
    return np.interp(dst, src, audio).astype(np.float32)


def load_audio_16k(path):
    """Any audio/video file -> float32 mono at 16 kHz, peak-safe."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        audio, rate = _load_with_soundfile(path)
    except Exception:
        audio, rate = _load_with_ffmpeg(path)
    audio = resample_to_16k(audio, rate)
    if audio.size and not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def speech_mask(audio, vad, chunk_samples=512):
    """Per-chunk speech flags from an already-built VAD. VAD output stops here."""
    n = len(audio) // chunk_samples
    if n == 0:
        return np.zeros(0, dtype=bool)
    vad.reset()
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        chunk = audio[i * chunk_samples:(i + 1) * chunk_samples]
        try:
            flags[i] = bool(vad.is_speech(chunk))
        except Exception:
            flags[i] = False
    return flags


def window_bounds(n_samples, spec=DEFAULT_WINDOW_SPEC):
    """Sample ranges of the overlapping analysis windows covering n_samples."""
    win, hop = spec.window_samples, spec.hop_samples
    if n_samples < spec.min_samples:
        return []
    bounds = []
    start = 0
    while start < n_samples:
        end = min(start + win, n_samples)
        if end - start >= spec.min_samples:
            bounds.append((start, end))
        if end >= n_samples:
            break
        start += hop
    return bounds


def speech_windows(audio, flags, spec=DEFAULT_WINDOW_SPEC, chunk_samples=512):
    """Overlapping 2-4 s windows that contain enough speech.

    Returns [(window_audio, start_seconds, speech_ratio)]. `flags` is the
    per-chunk VAD output; an empty/None flags array means "take every window"
    (used when a caller has already gated the audio some other way).
    """
    out = []
    for start, end in window_bounds(len(audio), spec):
        if flags is None or len(flags) == 0:
            ratio = 1.0
        else:
            lo, hi = start // chunk_samples, max(start // chunk_samples + 1, end // chunk_samples)
            span = flags[lo:hi]
            ratio = float(span.mean()) if span.size else 0.0
        if ratio < spec.min_speech_ratio:
            continue
        out.append((audio[start:end], start / float(SAMPLE_RATE), ratio))
    return out


def crop_to_speech(audio, flags, spec=DEFAULT_WINDOW_SPEC, chunk_samples=512, pad_s=0.2):
    """Trim leading/trailing non-speech, keeping at least spec.min_samples."""
    if flags is None or not flags.any():
        return audio
    idx = np.flatnonzero(flags)
    pad = int(round(pad_s * SAMPLE_RATE))
    start = max(0, idx[0] * chunk_samples - pad)
    end = min(len(audio), (idx[-1] + 1) * chunk_samples + pad)
    if end - start < spec.min_samples:
        end = min(len(audio), start + spec.min_samples)
        start = max(0, end - spec.min_samples)
    return audio[start:end]

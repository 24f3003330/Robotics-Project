"""
emotion_detector.py - live ACOUSTIC speech emotion recognition.

Only the sound of the voice is analysed. There is no speech-to-text anywhere in
this module, so the words a person says never influence the result.

    microphone (sounddevice callback, 16 kHz mono)
        -> queue.Queue                     callback only copies samples
        -> worker thread
             -> Voice Activity Detection   Silero VAD (RMS energy fallback)
             -> rolling 3 s buffer         analysed every 0.5 s while speech is present
             -> HuBERT ensemble            one fine-tuned model per corpus (IEMOCAP,
                                           CREMA-D, RAVDESS, MSP-IMPROV, MSP-Podcast),
                                           calibrated, then combined - see ser/.
                                           Falls back to the single pretrained
                                           superb/hubert-base-superb-er when no
                                           ensemble has been built.
             -> temporal smoothing         average of recent probability vectors
             -> stable emotion             on_stable_emotion(emotion, confidence, probabilities)

VAD and classification stay strictly separate: the VAD works on 32 ms chunks and
its only output is "which samples are speech"; the classifier only ever sees the
2-4 s windows that gate builds. A VAD frame is never handed to HuBERT.

This module knows nothing about the robot. app.py decides what a stable emotion does.

Command line self-tests (these never move the robot):
    python emotion_detector.py --list-devices
    python emotion_detector.py --test-mic  [--device N] [--seconds 10] [--save mic_test.wav]
    python emotion_detector.py --test-file clip1.wav [clip2.wav ...] [--windows]
    python emotion_detector.py --live      [--device N]
"""

import argparse
import os
import queue
import sys
import threading
import time
import traceback
import wave
from collections import deque

import numpy as np

try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _TORCH_IMPORT_ERROR = None
except Exception as exc:  # ImportError, or OSError such as "DLL load failed" on Windows
    torch = None
    DEVICE = "cpu"
    _TORCH_IMPORT_ERROR = exc

try:
    import sounddevice as sd

    _SD_IMPORT_ERROR = None
except Exception as exc:  # ImportError, or OSError when the PortAudio library is missing
    sd = None
    _SD_IMPORT_ERROR = exc

# ============================================================
# SETTINGS
# ============================================================
AUDIO_SAMPLE_RATE = 16000        # the emotion model and Silero VAD expect 16 kHz
CHANNELS = 1
AUDIO_WINDOW_SECONDS = 3.0       # audio analysed per prediction
AUDIO_UPDATE_SECONDS = 0.5       # analyse every 0.5 s (raise to 1.0 on a slow CPU)
AUDIO_BLOCK_SECONDS = 0.1        # size of each microphone callback block

# Pretrained acoustic emotion model: HuBERT base fine-tuned on IEMOCAP with
# exactly four classes (neu / hap / ang / sad). "superb/hubert-large-superb-er"
# is a few points more accurate but ~3x slower on CPU; prefer it with a GPU.
EMOTION_MODEL_NAME = os.environ.get("VOICE_EMOTION_MODEL", "superb/hubert-base-superb-er")
TORCH_NUM_THREADS = int(os.environ.get("VOICE_TORCH_THREADS", "0"))  # 0 = auto (half the cores, max 4)

# Multi-dataset ensemble (ser/). When models/ensemble/manifest.json exists, the
# per-dataset fine-tuned HuBERT models listed in it are loaded and combined,
# and EMOTION_MODEL_NAME above becomes only the fallback. The ensemble object
# exposes the SAME predict(audio) -> {emotion: prob} interface, so nothing
# downstream of this module changes. Set VOICE_USE_ENSEMBLE=0 to force the
# single pretrained model back on (useful for an A/B against the old system).
VOICE_USE_ENSEMBLE = os.environ.get("VOICE_USE_ENSEMBLE", "1") != "0"
VOICE_ENSEMBLE_MANIFEST = os.environ.get(
    "VOICE_ENSEMBLE_MANIFEST",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ensemble", "manifest.json"))
# Per-window per-member probabilities on the console. The ensemble's own
# verdict is always logged; this adds every member's vote so a disagreement
# ("ensemble says sad, cremad says angry 0.71") can be read off the log.
VOICE_LOG_MEMBERS = os.environ.get("VOICE_LOG_MEMBERS", "1") != "0"

# Voice activity detection
VAD_BACKEND = os.environ.get("VOICE_VAD", "silero")  # "silero" or "rms"
VAD_CHUNK_SAMPLES = 512          # 32 ms - the chunk size Silero VAD requires at 16 kHz
VAD_SPEECH_THRESHOLD = 0.5       # Silero speech probability per chunk
VAD_RECENT_SECONDS = 0.5         # look-back for "is someone speaking right now?"
VAD_MIN_RECENT_RATIO = 0.2       # at least 20 % of that look-back must be speech
VAD_MIN_SPEECH_SECONDS = 1.5     # the analysis window must contain this much speech
RMS_MIN_DBFS = -45.0             # RMS fallback: absolute minimum speech level
RMS_MARGIN_DB = 10.0             # RMS fallback: dB above the tracked noise floor

# Hop between uploaded-file analysis windows. 1.5 s against a 3.0 s window is
# 50 % overlap, so an emotional peak that straddles a window boundary is still
# seen whole by the next window. The live path already overlaps far more
# aggressively (a 3 s buffer re-analysed every 0.5 s); the uploaded-file path
# used to step a full window at a time, i.e. no overlap at all.
UPLOAD_HOP_SECONDS = float(os.environ.get("VOICE_UPLOAD_HOP_SECONDS", "1.5"))

SPEECH_PAD_SECONDS = 0.2         # audio kept before the first speech chunk of the window
# Never send less audio than this to the model. The first ~1 s of an utterance is
# often ambiguous (a sad sentence tends to start out "neutral"), so very short
# windows could make the wrong emotion stable before the real one is audible.
MIN_ANALYSIS_SECONDS = 2.0

# ------------------------------------------------------------------
# EMOTION ANALYSIS WINDOW - built from SPEECH, not from wall clock.
#
# This replaces the old gate, which demanded that VAD_MIN_SPEECH_SECONDS (1.5 s)
# of speech happen to fall inside the trailing 3 s ring buffer AND that speech be
# active in the last 0.5 s, both at the same instant. Measured with the real
# Silero VAD on real speech, that gate is a cliff at about 1.7 s of phrase length:
#
#   phrase 1.5 s / pause 2.0 s -> at most 1.22 s of speech in any 3 s ring -> 0 windows
#   phrase 2.0 s / pause 1.5 s -> 1.82 s                                   -> 65 windows
#   phrase 3.0 s / pause 2.0 s -> 2.72 s                                   -> 215 windows
#
# So one long opening sentence produced a prediction and every normal-length
# phrase afterwards produced nothing at all - "it works once and then stops".
#
# The accumulator below collects SPEECH itself, so a pause postpones a window
# instead of destroying it, and every phrase gets classified when it ends.
# ------------------------------------------------------------------
EMOTION_WINDOW_MAX_SECONDS = 4.0      # never hand HuBERT more than this
EMOTION_TARGET_SPEECH_SECONDS = 2.0   # fire as soon as this much speech is collected
EMOTION_MIN_SPEECH_SECONDS = 0.8      # ...and classify a shorter phrase when it ends
EMOTION_OVERLAP_SECONDS = 1.5         # audio retained after a window -> 2-4 s overlapping
SPEECH_HANGOVER_SECONDS = 0.45        # non-speech kept inside an utterance (between words)
UTTERANCE_END_SECONDS = 0.7           # silence this long ends the phrase
MIN_INFERENCE_INTERVAL = 0.35         # floor on how often HuBERT may run
# Smoothing history is time-bounded as well as length-bounded, so a prediction
# from before a pause cannot keep steering the average after it.
EMOTION_HISTORY_SECONDS = 6.0
HEARTBEAT_SECONDS = 10.0              # "[VOICE HEARTBEAT] worker alive ..." cadence

# Temporal smoothing
EMOTION_HISTORY_SIZE = 6         # recent probability vectors that are averaged
# The confidence gate is applied to the SMOOTHED probabilities, i.e. the mean of
# the last EMOTION_HISTORY_SIZE vectors. Averaging pulls the peak down, so on a
# 4-class model a clearly happy speaker sits around 0.50-0.58, not 0.60+. A plain
# "mean >= 0.60" gate is therefore almost never satisfied and the streak can never
# grow (that was the "streak 0/3 forever" bug). The gate is an absolute floor plus
# a margin over the runner-up, which is what actually means "one class is winning".
EMOTION_CONFIDENCE_THRESHOLD = 0.45   # dominant smoothed probability must reach this
EMOTION_MARGIN_THRESHOLD = 0.10       # ...and beat the runner-up by at least this much
EMOTION_STABILITY_COUNT = 3      # (legacy) consecutive-window count, superseded below
# Stability is measured in SECONDS OF CONFIDENT SPEECH, not in windows.
#
# Windows are utterance-driven now, so their rate varies with how someone talks:
# "three consecutive confident windows" silently meant ~4.5 s of uninterrupted
# confident speech. Measured on 3 s phrases separated by pauses, the streak never
# passed 2 of 3, so the live voice layer never published anything and fusion
# always fell back to the face. Seconds of speech is the quantity that actually
# represents evidence, and it does not change meaning when the window rate does.
EMOTION_STABILITY_SECONDS = 2.0
# 6.0s (not the original 3.0s): a single confident window only needs to survive
# an ordinary mid-conversation pause (breath, end of a short phrase before the
# next one) to keep building toward EMOTION_STABILITY_COUNT. Measured live: a
# genuinely confident window (ANGRY 81%, streak 1/3) was wiped by an entirely
# normal >3s pause before ever reaching 3/3, so voice could not win fusion for
# any short test phrase. 6.0s still ends a genuinely separate, later utterance
# (so it does not carry stale momentum into an unrelated one) but tolerates
# normal conversational gaps between phrases of the same utterance.
SILENCE_RESET_SECONDS = 6.0      # this much silence ends an utterance and clears the history
# While a stable emotion keeps holding, re-announce it this often so consumers
# (app.py keeps a freshness TTL on the voice layer) know it is still current.
STABLE_REEMIT_SECONDS = 5.0

MIC_STALL_SECONDS = 3.0          # no audio for this long = something is wrong (warn)
# ...but a stall is NOT immediately fatal. Ending the session on the first 3 s
# gap is wrong on a loaded machine: the camera loop, Flask, BLE and HuBERT all
# compete for the same cores, and one scheduling hiccup would permanently stop
# voice detection until the user pressed Start again - indistinguishable, from
# the outside, from "it stopped working after the first prediction". So a stall
# first warns, then tries to REOPEN the microphone, and only a stream that
# cannot be reopened (or that stays silent after reopening) ends the session.
MIC_RECOVER_SECONDS = 8.0        # try reopening the stream after this long
MIC_FATAL_SECONDS = 25.0         # give up only after this long with no audio
MIC_MAX_REOPEN_ATTEMPTS = 3
LOG_EVERY_WINDOW = os.environ.get("VOICE_LOG_EVERY_WINDOW", "0") == "1"

EMOTIONS = ("happy", "angry", "sad", "neutral")

# Model label -> supported emotion, matched on the first three letters so that
# "hap", "happy" and "happiness" all work. Other labels (fear, disgust, ...)
# are ignored and the four supported emotions are renormalised.
_LABEL_PREFIXES = {
    "neu": "neutral",
    "hap": "happy",
    "exc": "happy",
    "joy": "happy",
    "ang": "angry",
    "sad": "sad",
}

_KEEP = object()


class MicrophoneError(RuntimeError):
    """A microphone problem whose message is meant to be shown to the user."""


def _log(message):
    print(f"[VOICE] {message}", flush=True)


def _dbfs(audio):
    if audio.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return 20.0 * np.log10(rms + 1e-9)


# ============================================================
# MICROPHONE HELPERS
# ============================================================
def parse_device(value):
    """None / "" / "default" -> system default, "3" -> 3, other text -> name substring."""
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "" or text.lower() == "default":
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def list_input_devices():
    """Every device with input channels, for the UI and --list-devices."""
    if sd is None:
        raise MicrophoneError(f"sounddevice is not available: {_SD_IMPORT_ERROR}")
    hostapis = sd.query_hostapis()
    try:
        default_input = sd.default.device[0]
    except Exception:
        default_input = -1
    devices = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "index": index,
                "name": dev["name"],
                "hostapi": hostapis[dev["hostapi"]]["name"],
                "channels": int(dev["max_input_channels"]),
                "default_samplerate": int(round(dev["default_samplerate"])),
                "is_default": index == default_input,
            })
    return devices


def _explain_audio_error(exc):
    text = str(exc)
    lower = text.lower()
    if "multiple" in lower:
        return (f"Microphone name matches several devices. Use the device index shown by "
                f"/voice/devices or 'python emotion_detector.py --list-devices'. ({text})")
    if "-9985" in text or "unavailable" in lower or "busy" in lower:
        return (f"Microphone is busy or unavailable. Close other apps that may hold it exclusively "
                f"(Teams, Zoom, Discord, a browser tab) or choose another device. ({text})")
    if "querying device" in lower or "-9996" in text or "no input device" in lower or "invalid device" in lower:
        return (f"No usable microphone found. Connect or enable a microphone, or choose a device "
                f"from the list. ({text})")
    if "-9999" in text or "host error" in lower or "denied" in lower or "access" in lower:
        return (f"The operating system refused microphone access. On Windows open Settings > Privacy & "
                f"security > Microphone and enable 'Microphone access' and 'Let desktop apps access "
                f"your microphone'. ({text})")
    return f"Could not open the microphone: {text}"


def open_input_stream(device, callback):
    """Open and start a microphone stream.

    Returns (stream, sample_rate, channels, device_label). Prefers 16 kHz mono; if the
    device refuses, retries with WASAPI auto-convert and then with the device's native
    rate, in which case the caller resamples to 16 kHz.
    """
    if sd is None:
        raise MicrophoneError(f"sounddevice is not available ({_SD_IMPORT_ERROR}). "
                              f"Install it with: pip install sounddevice")
    device = parse_device(device)
    try:
        info = sd.query_devices(device, kind="input")
    except Exception as exc:
        raise MicrophoneError(_explain_audio_error(exc))

    max_channels = int(info["max_input_channels"])
    if max_channels < 1:
        raise MicrophoneError(f"'{info['name']}' has no input channels")
    native_rate = int(round(info["default_samplerate"]))
    hostapi = sd.query_hostapis(info["hostapi"])["name"]
    device_index = info.get("index", device)

    attempts = [(AUDIO_SAMPLE_RATE, CHANNELS, None)]
    if "wasapi" in hostapi.lower():
        try:
            attempts.append((AUDIO_SAMPLE_RATE, CHANNELS, sd.WasapiSettings(auto_convert=True)))
        except TypeError:  # older sounddevice without auto_convert
            pass
    if native_rate != AUDIO_SAMPLE_RATE:
        attempts.append((native_rate, CHANNELS, None))
    if max_channels > 1:
        attempts.append((native_rate, min(2, max_channels), None))

    last_exc = None
    for rate, channels, extra in attempts:
        stream = None
        try:
            stream = sd.InputStream(
                device=device_index,
                samplerate=rate,
                channels=channels,
                dtype="float32",
                blocksize=int(rate * AUDIO_BLOCK_SECONDS),
                callback=callback,
                extra_settings=extra,
            )
            stream.start()
            return stream, rate, channels, f"{info['name']} ({hostapi})"
        except Exception as exc:
            last_exc = exc
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
    raise MicrophoneError(_explain_audio_error(last_exc))


class _StreamingResampler:
    """Block-by-block resampler, only used when a microphone refuses 16 kHz.

    Windowed-sinc low-pass (anti-aliasing) followed by linear interpolation. Filter
    history and interpolation phase carry over between blocks, so there are no clicks
    at block boundaries.
    """

    def __init__(self, in_rate, out_rate=AUDIO_SAMPLE_RATE, taps=101):
        self.passthrough = int(in_rate) == int(out_rate)
        self.step = in_rate / float(out_rate)  # input samples per output sample
        self._fir = None
        if in_rate > out_rate:
            cutoff = 0.45 * out_rate / in_rate  # cycles per input sample
            n = np.arange(taps) - (taps - 1) / 2.0
            h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n) * np.hamming(taps)
            self._fir = (h / h.sum()).astype(np.float32)
            self._fir_tail = np.zeros(taps - 1, dtype=np.float32)
        self._last = np.zeros(1, dtype=np.float32)
        self._pos = 1.0

    def process(self, block):
        block = np.asarray(block, dtype=np.float32)
        if self.passthrough or block.size == 0:
            return block
        if self._fir is not None:
            buf = np.concatenate([self._fir_tail, block])
            self._fir_tail = buf[-(len(self._fir) - 1):]
            block = np.convolve(buf, self._fir, mode="valid").astype(np.float32)
        # buf[0] is the previous block's last sample, so interpolation is continuous.
        buf = np.concatenate([self._last, block])
        end = len(buf) - 1
        positions = np.arange(self._pos, end, self.step)
        out = np.interp(positions, np.arange(len(buf)), buf).astype(np.float32)
        self._pos = (positions[-1] + self.step - end) if positions.size else (self._pos - end)
        self._last = buf[-1:]
        return out


# ============================================================
# VOICE ACTIVITY DETECTION
# ============================================================
def _load_silero_model():
    """Load the TorchScript model bundled in the silero-vad pip package.

    `import silero_vad` also imports torchaudio (only needed for its file helpers), so
    the bundled file is loaded directly. A torch/torchaudio version mismatch then
    cannot break voice activity detection.
    """
    import importlib.util
    import warnings

    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("silero-vad is not installed (pip install silero-vad)")
    for location in spec.submodule_search_locations:
        path = os.path.join(location, "data", "silero_vad.jit")
        if os.path.isfile(path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "torch.jit.load is deprecated" notice
                model = torch.jit.load(path, map_location="cpu")
            model.eval()
            return model
    from silero_vad import load_silero_vad

    return load_silero_vad()


class _SileroVAD:
    name = "silero"

    def __init__(self):
        if torch is None:
            raise RuntimeError(f"torch is not available: {_TORCH_IMPORT_ERROR}")
        self.model = _load_silero_model()

    def reset(self):
        self.model.reset_states()

    def is_speech(self, chunk):
        with torch.inference_mode():
            prob = self.model(torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)),
                              AUDIO_SAMPLE_RATE).item()
        return prob >= VAD_SPEECH_THRESHOLD


class _EnergyVAD:
    """Fallback (used only when Silero is unavailable): louder than the background.

    The noise floor is estimated from the QUIETEST recent frames, not from the
    frames this class just decided were "not speech".

    The original did the latter, adapting slowly (alpha=0.002) towards the level
    of frames it had called speech. That is a slow leak with a hard failure at
    the end of it: over sustained talking the floor climbs towards the speaker's
    own level, "floor + 10 dB" stops being satisfiable, and the VAD goes
    permanently quiet until a silence resets it. Measured on 60 s of continuous
    real speech before this change: 8 predictions in the first half of the
    session and 0 in the second.

    A minimum statistic cannot chase the speaker, because ordinary speech always
    contains low-energy frames (gaps between words) and those are what set the
    floor. NOISE_FLOOR_MAX_DBFS is a second, absolute guarantee: however loud the
    room, the floor can never rise so far that normal speech stops clearing it.
    """

    name = "rms"

    # ~8 s of level history: long enough to contain inter-word gaps even in
    # continuous speech, short enough to follow a room that genuinely changes.
    _MEMORY_CHUNKS = max(8, int(round(8.0 * AUDIO_SAMPLE_RATE / VAD_CHUNK_SAMPLES)))
    _PERCENTILE = 10.0
    NOISE_FLOOR_MAX_DBFS = -40.0

    def __init__(self):
        self.reset()

    def reset(self):
        self._levels = deque(maxlen=self._MEMORY_CHUNKS)
        self.noise_floor_db = -60.0

    def is_speech(self, chunk):
        level = _dbfs(chunk)
        self._levels.append(level)
        if len(self._levels) >= 8:
            estimate = float(np.percentile(np.fromiter(self._levels, dtype=np.float64),
                                           self._PERCENTILE))
            self.noise_floor_db = min(estimate, self.NOISE_FLOOR_MAX_DBFS)
        return level >= max(RMS_MIN_DBFS, self.noise_floor_db + RMS_MARGIN_DB)


def _make_vad(backend):
    """Returns (vad, warning_or_None)."""
    if backend == "silero":
        try:
            return _SileroVAD(), None
        except Exception as exc:
            return _EnergyVAD(), f"Silero VAD unavailable ({exc}); using RMS energy VAD"
    return _EnergyVAD(), None


# ============================================================
# EMOTION MODEL
# ============================================================
def _normalize_label(label):
    return _LABEL_PREFIXES.get(str(label).strip().lower()[:3])


class _EmotionModel:
    """Pretrained Hugging Face audio-classification model, loaded once and reused."""

    def __init__(self, model_name):
        if torch is None:
            raise RuntimeError(f"torch is not available: {_TORCH_IMPORT_ERROR}")
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        if DEVICE == "cpu":
            threads = TORCH_NUM_THREADS or max(1, min(4, (os.cpu_count() or 2) // 2))
            torch.set_num_threads(threads)  # leave cores free for Flask, camera and BLE

        try:
            # Use the local Hugging Face cache first: no network needed after the first run.
            extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForAudioClassification.from_pretrained(model_name, local_files_only=True)
        except Exception:
            _log(f"Model '{model_name}' is not cached yet - downloading (first run needs internet) ...")
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(model_name)

        self.extractor = extractor
        self.model = model.to(DEVICE).eval()
        self.model_name = model_name
        self.name = model_name
        self.labels = {int(i): str(l) for i, l in self.model.config.id2label.items()}
        self.index_to_emotion = {}
        for index, label in self.labels.items():
            emotion = _normalize_label(label)
            if emotion:
                self.index_to_emotion[index] = emotion
        missing = set(EMOTIONS) - set(self.index_to_emotion.values())
        if missing:
            raise RuntimeError(f"Model '{model_name}' has no label for {sorted(missing)} "
                               f"(labels: {list(self.labels.values())})")
        self.predict(np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32))  # warm-up

    def predict(self, audio):
        """audio: float32 mono at 16 kHz -> {"happy": p, "angry": p, "sad": p, "neutral": p}"""
        inputs = self.extractor(audio, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
        all_probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        probs = {e: 0.0 for e in EMOTIONS}
        for index, emotion in self.index_to_emotion.items():
            probs[emotion] += float(all_probs[index])
        total = sum(probs.values())
        if total <= 0:
            return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
        return {e: p / total for e, p in probs.items()}

    def predict_detailed(self, audio):
        """Same shape the ensemble returns, so callers need only one code path.

        A single model is an ensemble of one: it is its own only member, it
        always agrees with itself, and there is never a dissenting member.
        """
        t0 = time.perf_counter()
        probs = self.predict(audio)
        ms = round((time.perf_counter() - t0) * 1000.0, 1)
        emotion = max(probs, key=probs.get)
        return {
            "ensemble": probs,
            "members": {self.name: dict(probs)},
            "member_top": {self.name: emotion},
            "member_ms": {self.name: ms},
            "combiner": "single",
            "weights": {self.name: 1.0},
            "emotion": emotion,
            "agreement": 1.0,
            "dissenting": [],
            "ms": ms,
        }

    def describe(self):
        return {
            "kind": "single",
            "model": self.model_name,
            "members": [self.name],
            "combiner": "single",
            "weights": {self.name: 1.0},
            "labels": list(self.labels.values()),
            "device": str(DEVICE),
        }


def build_emotion_model(model_name=EMOTION_MODEL_NAME, manifest_path=None, use_ensemble=None):
    """The one place that decides WHAT classifies a window.

    Returns either a multi-dataset ensemble (ser.runtime.EnsembleEmotionModel)
    or the single pretrained model that shipped before it. Both expose
    predict() and predict_detailed() with identical signatures, so every caller
    - the live worker, the uploaded-file job, the CLI self-tests - is written
    once and works with whichever is present.

    Falling back is deliberate and silent-but-logged: a missing or half-built
    models/ensemble/ must leave the app exactly as capable as it was before,
    never broken.
    """
    if use_ensemble is None:
        use_ensemble = VOICE_USE_ENSEMBLE
    if use_ensemble:
        manifest_path = manifest_path or VOICE_ENSEMBLE_MANIFEST
        try:
            from ser.runtime import EnsembleEmotionModel, EnsembleUnavailable

            model = EnsembleEmotionModel(manifest_path)
            info = model.describe()
            _log(f"Ensemble loaded: {len(info['members'])} members "
                 f"({', '.join(info['members'])}), combiner={info['combiner']}, "
                 f"weights={info['weights']}")
            for skip in info.get("skipped", []):
                _log(f"Ensemble member '{skip['member']}' not loaded: {skip['reason']}")
            if info.get("combiner_note"):
                _log(f"WARNING: {info['combiner_note']}")
            return model
        except EnsembleUnavailable as exc:
            _log(f"No multi-dataset ensemble ({exc}); using the single model '{model_name}'. "
                 f"Build one with: python -m ser all --run r1 --cremad ... --ravdess ...")
        except Exception as exc:
            _log(f"WARNING: the ensemble failed to load ({exc}); falling back to the single "
                 f"model '{model_name}'.")
            traceback.print_exc()
    return _EmotionModel(model_name)


# ============================================================
# SPEECH ACCUMULATOR
# ============================================================
class _SpeechAccumulator:
    """Turns a stream of 32 ms VAD-flagged chunks into 2-4 s emotion windows.

    VAD frames never reach HuBERT. Their only job is to say which chunks are
    speech; this class decides when enough speech has been collected to be worth
    classifying, and hands over one contiguous 2-4 s stretch of audio.

    Two ways a window becomes ready:
      * "window_full"    - EMOTION_TARGET_SPEECH_SECONDS of speech collected.
                           The tail is kept afterwards, so a long utterance
                           produces overlapping windows rather than disjoint ones.
      * "utterance_end"  - the speaker stopped and at least
                           EMOTION_MIN_SPEECH_SECONDS was collected. This is the
                           case the old code could not express at all, and it is
                           what makes ordinary short phrases classifiable.

    Short non-speech gaps inside a phrase (SPEECH_HANGOVER_SECONDS) are kept, so
    the window the model sees is natural speech with its own micro-pauses rather
    than concatenated voiced fragments.
    """

    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE,
                 target_speech=EMOTION_TARGET_SPEECH_SECONDS,
                 min_speech=EMOTION_MIN_SPEECH_SECONDS,
                 max_window=EMOTION_WINDOW_MAX_SECONDS,
                 overlap=EMOTION_OVERLAP_SECONDS,
                 hangover=SPEECH_HANGOVER_SECONDS,
                 utterance_end=UTTERANCE_END_SECONDS):
        self.sample_rate = int(sample_rate)
        self.target_speech_samples = int(target_speech * sample_rate)
        self.min_speech_samples = int(min_speech * sample_rate)
        self.max_samples = int(max_window * sample_rate)
        self.overlap_samples = int(overlap * sample_rate)
        self.hangover_samples = int(hangover * sample_rate)
        self.utterance_end_samples = int(utterance_end * sample_rate)
        self.reset()

    def reset(self):
        self._chunks = deque()      # (samples, is_speech)
        self.speech_samples = 0
        self.total_samples = 0
        self.silence_samples = 0
        self.in_utterance = False
        # Speech already handed to a previous window. Windows overlap on
        # purpose, so without this the same half-second of audio would be
        # counted as fresh evidence again and again.
        self._counted_speech_samples = 0

    # ---------------------------------------------------------------- input
    def add(self, chunk, is_speech):
        """One VAD-classified chunk. Returns a transition string for logging, or None."""
        n = len(chunk)
        transition = None
        if is_speech:
            if not self.in_utterance:
                transition = "speech_started"
            self.in_utterance = True
            self.silence_samples = 0
            self._chunks.append((chunk, True))
            self.speech_samples += n
            self.total_samples += n
        elif self.in_utterance:
            self.silence_samples += n
            # Keep the short pauses that live inside a phrase; drop the rest.
            if self.silence_samples <= self.hangover_samples:
                self._chunks.append((chunk, False))
                self.total_samples += n
            elif self.silence_samples - n <= self.hangover_samples:
                transition = "speech_ended"
        self._trim(self.max_samples)
        return transition

    # --------------------------------------------------------------- output
    def ready(self):
        """Why a window should be classified now, or None."""
        if self.speech_samples >= self.target_speech_samples:
            return "window_full"
        if (self.in_utterance
                and self.silence_samples >= self.utterance_end_samples
                and self.speech_samples >= self.min_speech_samples):
            return "utterance_end"
        return None

    def take(self, reason):
        """The window to classify, plus how much of it is NEW speech.

        Returns (window, speech_seconds, new_speech_seconds). `new_speech_seconds`
        is what the stability logic accumulates - it is the speech this window
        contributes that no earlier window has already been credited for.
        """
        if not self._chunks:
            return np.zeros(0, dtype=np.float32), 0.0, 0.0
        window = np.concatenate([c for c, _ in self._chunks])
        speech_seconds = self.speech_samples / float(self.sample_rate)
        new_samples = max(0, self.speech_samples - self._counted_speech_samples)
        new_speech_seconds = new_samples / float(self.sample_rate)
        if reason == "utterance_end":
            # The phrase is over: start the next one clean rather than letting
            # this one's audio bleed into it.
            self.reset()
        else:
            # Mid-utterance: keep the tail so the next window OVERLAPS this one.
            self._trim(self.overlap_samples)
            self._counted_speech_samples = self.speech_samples
        return window, speech_seconds, new_speech_seconds

    def expired(self):
        """True when the speaker stopped and nothing is worth keeping."""
        return (self.in_utterance
                and self.silence_samples >= self.utterance_end_samples
                and self.speech_samples < self.min_speech_samples)

    # -------------------------------------------------------------- internal
    def _trim(self, limit):
        while self._chunks and self.total_samples > limit:
            chunk, was_speech = self._chunks.popleft()
            self.total_samples -= len(chunk)
            if was_speech:
                self.speech_samples -= len(chunk)
        self.speech_samples = max(0, self.speech_samples)
        self.total_samples = max(0, self.total_samples)

    @property
    def buffered_seconds(self):
        return self.total_samples / float(self.sample_rate)

    @property
    def speech_seconds(self):
        return self.speech_samples / float(self.sample_rate)

    @property
    def silence_seconds(self):
        return self.silence_samples / float(self.sample_rate)


# ============================================================
# LIVE DETECTOR
# ============================================================
class _Session:
    """One start() ... stop() run. Kept separate so a slow old worker can never touch a new run."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.user_stopped = False
        self.audio_queue = queue.Queue(maxsize=int(20 / AUDIO_BLOCK_SECONDS))  # ~20 s of audio
        self.stream = None
        self.stream_lock = threading.Lock()
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.worker = None
        self.dropped_blocks = 0
        self.overflows = 0
        self.warnings = {}
        # smoothing state (touched only by this session's worker)
        self.prob_history = None
        self.streak_emotion = None
        self.streak_count = 0.0
        self.last_emitted_emotion = None
        self.last_emitted_time = 0.0
        self.last_speech_time = time.monotonic()
        self.inference_count = 0
        self.inference_errors = 0
        self.skipped_cycles = 0
        self.callback_count = 0
        self.callback_errors = 0
        self.accumulator = None

    def callback(self, indata, frames, time_info, status):
        # PortAudio audio thread: only copy samples here, never run models.
        # An exception raised here would make PortAudio abort the stream, which
        # would end capture for the whole session - so nothing is allowed out.
        try:
            self.callback_count += 1
            if status:
                self.overflows += 1
            if indata.shape[1] == 1:
                mono = indata[:, 0].copy()
            else:
                mono = indata.mean(axis=1).astype(np.float32)
            try:
                self.audio_queue.put_nowait(mono)
            except queue.Full:
                # Never let the queue wedge: drop the OLDEST block and keep the
                # newest. A permanently full queue would stall the pipeline while
                # the microphone kept running, which looks exactly like a freeze.
                self.dropped_blocks += 1
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(mono)
                except (queue.Empty, queue.Full):
                    pass
            if self.callback_count % 200 == 0:
                print(f"[AUDIO] callback #{self.callback_count} "
                      f"queue={self.audio_queue.qsize()} dropped={self.dropped_blocks} "
                      f"overflows={self.overflows}", flush=True)
        except Exception as exc:
            self.callback_errors += 1
            if self.callback_errors <= 5:
                print(f"[AUDIO ERROR] callback failed ({type(exc).__name__}: {exc}) - "
                      f"capture continues", flush=True)
                traceback.print_exc()

    def close_stream(self):
        with self.stream_lock:
            stream, self.stream = self.stream, None
        if stream is not None:
            for action in (stream.stop, stream.close):
                try:
                    action()
                except Exception:
                    pass

    def stream_alive(self):
        stream = self.stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:
            return False


class LiveEmotionDetector:
    """Continuously estimates happy / angry / sad / neutral from the tone of voice.

    Thread-safe: start(), stop(), get_status() and get_current_emotion() may be called
    from Flask request threads while the worker thread processes audio.

    on_stable_emotion(emotion, confidence, probabilities) is called from the worker
    thread each time a new emotion becomes stable. on_update(status) is called after
    every analysis cycle. Both must return quickly.
    """

    def __init__(self, model_name=EMOTION_MODEL_NAME, input_device=None,
                 on_stable_emotion=None, on_update=None,
                 window_seconds=AUDIO_WINDOW_SECONDS, update_seconds=AUDIO_UPDATE_SECONDS,
                 confidence_threshold=EMOTION_CONFIDENCE_THRESHOLD,
                 margin_threshold=EMOTION_MARGIN_THRESHOLD,
                 stability_seconds=EMOTION_STABILITY_SECONDS, history_size=EMOTION_HISTORY_SIZE,
                 vad_backend=VAD_BACKEND, log_every_window=LOG_EVERY_WINDOW):
        self.model_name = model_name
        self.input_device = parse_device(input_device)
        self.on_stable_emotion = on_stable_emotion
        self.on_update = on_update
        self.window_seconds = float(window_seconds)
        self.update_seconds = float(update_seconds)
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.stability_seconds = float(stability_seconds)
        self.history_size = int(history_size)
        self.vad_backend = vad_backend
        self.log_every_window = log_every_window

        self._control_lock = threading.Lock()  # start / stop
        self._state_lock = threading.Lock()    # self._state
        self._model_lock = threading.Lock()    # loads the model exactly once
        self._model = None
        self._session = None

        # ------------------------------------------------------------
        # SPEAKER BASELINE - placeholder, not implemented yet.
        # Plan: during calibration the person talks for ~20-30 s in their
        # normal voice. The raw probability vectors are averaged into
        # speaker_baseline, e.g. {"angry": 0.35, "neutral": 0.40, ...} for a
        # naturally forceful speaker. _apply_speaker_baseline() can then
        # re-weight live predictions, e.g. p[e] ~ p[e] / baseline[e] ** alpha,
        # so that person's normal voice maps to neutral instead of angry.
        # ------------------------------------------------------------
        self.speaker_baseline = None
        self.calibrating = False

        self._state = {
            "running": False,
            "status": "stopped",  # stopped | starting | loading_model | listening | error
            "speech_detected": False,
            "emotion": None,
            "confidence": 0.0,
            "probabilities": {e: 0.0 for e in EMOTIONS},
            "raw_emotion": None,
            "raw_probabilities": {e: 0.0 for e in EMOTIONS},
            "stable_emotion": None,
            "stable_confidence": 0.0,
            "stability_count": 0,
            "stability_required": self.stability_seconds,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "margin": 0.0,
            "confident_window": False,
            "model_name": self.model_name,
            "model_kind": "single",
            "ensemble": None,
            # Per-member probabilities for the window just classified. This is
            # the debugging surface for disagreements: `members` is every
            # model's own distribution, `member_top` its argmax, `agreement`
            # the fraction of members backing the published emotion.
            "member_probabilities": {},
            "member_top": {},
            "member_ms": {},
            "member_agreement": None,
            "member_dissenting": [],
            "combiner": None,
            "model_loaded": False,
            "model_loading": False,
            "torch_device": DEVICE,
            "vad_backend": None,
            "input_device": None,
            "capture_sample_rate": None,
            "audio_level_db": None,
            "speech_ratio": 0.0,
            "last_inference_ms": None,
            "inference_count": 0,
            # Live-loop telemetry, so a stall is visible in the UI and the API
            # instead of having to be inferred from "nothing is changing".
            "prediction_timestamp": None,
            "prediction_index": 0,
            "buffered_seconds": 0.0,
            "buffered_speech_seconds": 0.0,
            "seconds_since_speech": None,
            "window_reason": None,
            "window_seconds": None,
            "window_speech_seconds": None,
            "smoothing_window": 0,
            "audio_callbacks": 0,
            "queue_depth": 0,
            "skipped_cycles": 0,
            "dropped_audio_blocks": 0,
            "error": None,
            "warning": None,
            "updated_at": time.time(),
        }

    # ------------------------------------------------------------------ public API
    def start(self, input_device=_KEEP):
        """Open the microphone and start the worker. Returns (ok, message)."""
        with self._control_lock:
            current = self._session
            if current is not None and not current.stop_event.is_set():
                return True, "Live voice emotion detection already running"
            if torch is None:
                return self._fail_start(f"PyTorch is not available ({_TORCH_IMPORT_ERROR}). "
                                        f"Install the requirements first.")
            if input_device is not _KEEP:
                self.input_device = parse_device(input_device)

            session = _Session()
            try:
                session.stream, session.sample_rate, _, label = open_input_stream(
                    self.input_device, session.callback)
            except MicrophoneError as exc:
                return self._fail_start(str(exc))

            self._session = session
            with self._state_lock:
                self._state.update({
                    "running": True,
                    "status": "starting",
                    "speech_detected": False,
                    "emotion": None,
                    "confidence": 0.0,
                    "probabilities": {e: 0.0 for e in EMOTIONS},
                    "raw_emotion": None,
                    "raw_probabilities": {e: 0.0 for e in EMOTIONS},
                    "stable_emotion": None,
                    "stable_confidence": 0.0,
                    "stability_count": 0,
                    "input_device": label,
                    "capture_sample_rate": session.sample_rate,
                    "audio_level_db": None,
                    "speech_ratio": 0.0,
                    "inference_count": 0,
                    "prediction_timestamp": None,
                    "prediction_index": 0,
                    "buffered_seconds": 0.0,
                    "buffered_speech_seconds": 0.0,
                    "seconds_since_speech": None,
                    "window_reason": None,
                    "smoothing_window": 0,
                    "audio_callbacks": 0,
                    "queue_depth": 0,
                    "skipped_cycles": 0,
                    "dropped_audio_blocks": 0,
                    "error": None,
                    "warning": None,
                    "updated_at": time.time(),
                })
            session.worker = threading.Thread(target=self._worker_main, args=(session,),
                                              name="VoiceEmotionWorker", daemon=True)
            session.worker.start()
            _log(f"Microphone started: {label} @ {session.sample_rate} Hz")
            return True, "Live voice emotion detection started"

    def stop(self):
        """Stop listening and release the microphone. Returns (ok, message)."""
        with self._control_lock:
            session = self._session
            if session is None or session.stop_event.is_set():
                self._update_state(None, running=False, speech_detected=False)
                return True, "Live voice emotion detection is not running"
            session.user_stopped = True
            session.stop_event.set()
            session.close_stream()  # release the microphone immediately
        worker = session.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
        self._update_state(session, running=False, speech_detected=False, status="stopped")
        _log("Microphone stopped")
        return True, "Live voice emotion detection stopped"

    def is_running(self):
        with self._state_lock:
            return self._state["running"]

    def get_current_emotion(self):
        """Smoothed dominant emotion of recent speech (None before any speech)."""
        with self._state_lock:
            return self._state["emotion"]

    def get_status(self):
        with self._state_lock:
            status = dict(self._state)
            status["probabilities"] = dict(self._state["probabilities"])
            status["raw_probabilities"] = dict(self._state["raw_probabilities"])
        return status

    def preload_model_async(self):
        """Load the model in the background (e.g. at app start-up) so Start is instant."""
        threading.Thread(target=self._ensure_model, name="VoiceEmotionModelLoader", daemon=True).start()

    # ------------------------------------------------------------------ speaker baseline (placeholder)
    def start_calibration(self):
        """Future: start collecting this speaker's neutral-voice probabilities."""
        return False, "Speaker calibration is not implemented yet"

    def stop_calibration(self):
        """Future: average the collected probabilities into self.speaker_baseline."""
        return False, "Speaker calibration is not implemented yet"

    def _apply_speaker_baseline(self, probabilities):
        # TODO: re-weight with self.speaker_baseline once calibration exists.
        return probabilities

    # ------------------------------------------------------------------ internals
    def _fail_start(self, message):
        _log(f"ERROR: {message}")
        self._update_state(None, running=False, status="error", error=message)
        return False, message

    def _update_state(self, session, **values):
        """Update shared state. With a session, only if that session is still the current one."""
        with self._state_lock:
            if session is not None and self._session is not session:
                return
            self._state.update(values)
            self._state["updated_at"] = time.time()

    def _ensure_model(self):
        """Load the emotion model once. Safe to call from several threads."""
        with self._model_lock:
            if self._model is not None:
                return True
            if torch is None:
                self._update_state(None, error=f"PyTorch is not available: {_TORCH_IMPORT_ERROR}")
                return False
            self._update_state(None, model_loading=True)
            started = time.time()
            _log(f"Loading emotion model '{self.model_name}' on {DEVICE} ...")
            try:
                self._model = build_emotion_model(self.model_name)
            except Exception as exc:
                message = (f"Could not load emotion model '{self.model_name}': {exc}. The first run needs "
                           f"internet to download the model (~380 MB); after that it loads offline "
                           f"from the Hugging Face cache.")
                _log(f"ERROR: {message}")
                self._update_state(None, model_loading=False, model_loaded=False, error=message)
                return False
            info = self._model.describe()
            _log(f"Emotion model ready in {time.time() - started:.1f} s "
                 f"(kind={info['kind']}, classes: {', '.join(EMOTIONS)})")
            with self._state_lock:
                self._state.update(model_loading=False, model_loaded=True,
                                   model_kind=info["kind"],
                                   model_name=getattr(self._model, "name", self.model_name),
                                   ensemble=info)
                if self._state["error"] and self._state["error"].startswith("Could not load emotion model"):
                    self._state["error"] = None
            return True

    def _worker_main(self, session):
        fatal = None
        try:
            if self._model is None:
                self._update_state(session, status="loading_model")
                if not self._ensure_model():
                    fatal = self.get_status()["error"] or "Emotion model failed to load"
                    return
            if not session.stop_event.is_set():
                self._process_audio(session)
        except MicrophoneError as exc:
            fatal = str(exc)
        except Exception as exc:
            fatal = f"Voice emotion worker stopped unexpectedly: {exc}"
            traceback.print_exc()
        finally:
            if session.user_stopped:
                fatal = None
            session.stop_event.set()
            session.close_stream()
            values = {"running": False, "speech_detected": False, "status": "error" if fatal else "stopped"}
            if fatal:
                values["error"] = fatal
                _log(f"ERROR: {fatal}")
            self._update_state(session, **values)
            if self.on_update is not None and self._session is session:
                try:
                    self.on_update(self.get_status())
                except Exception as exc:
                    _log(f"on_update callback failed: {exc}")

    def _set_warning(self, session, key, message):
        if message is None:
            session.warnings.pop(key, None)
        else:
            session.warnings[key] = message
        return "; ".join(session.warnings.values()) or None

    def _process_audio(self, session):
        """The live loop: drain audio -> VAD -> accumulate speech -> classify.

        This loop must survive everything. A VAD failure, an inference failure or
        a callback failure is logged with a full traceback and the loop CONTINUES;
        only a genuinely dead microphone ends it.
        """
        vad, vad_warning = _make_vad(self.vad_backend)
        vad.reset()
        if vad_warning:
            _log(vad_warning)
        self._update_state(session, status="listening", vad_backend=vad.name,
                           warning=self._set_warning(session, "vad", vad_warning))
        _log(f"Live loop started: vad={vad.name} target_speech={EMOTION_TARGET_SPEECH_SECONDS}s "
             f"min_speech={EMOTION_MIN_SPEECH_SECONDS}s overlap={EMOTION_OVERLAP_SECONDS}s "
             f"max_window={EMOTION_WINDOW_MAX_SECONDS}s")

        resampler = _StreamingResampler(session.sample_rate, AUDIO_SAMPLE_RATE)
        accumulator = _SpeechAccumulator()
        session.accumulator = accumulator
        session.prob_history = deque(maxlen=self.history_size)
        pending = np.zeros(0, dtype=np.float32)

        # Audio captured while the model was loading is stale - discard it.
        self._drain(session.audio_queue, timeout=0)
        started = last_audio = time.monotonic()
        session.last_speech_time = started
        last_inference = 0.0
        last_publish = started
        last_heartbeat = started
        heard_signal = False
        silent_mic_warned = False
        vad_errors = 0
        reopen_attempts = 0
        last_reopen = 0.0

        while not session.stop_event.is_set():
            blocks = self._drain(session.audio_queue, timeout=0.05)
            now = time.monotonic()

            if blocks:
                last_audio = now
                try:
                    audio = resampler.process(np.concatenate(blocks))
                except Exception as exc:
                    self._log_exception(session, "resampler", exc,
                                        accumulator=accumulator, extra=f"blocks={len(blocks)}")
                    audio = np.zeros(0, dtype=np.float32)
                heard_signal = heard_signal or bool(np.any(audio != 0.0))
                pending = np.concatenate([pending, audio])
                full = len(pending) // VAD_CHUNK_SAMPLES
                for i in range(full):
                    chunk = pending[i * VAD_CHUNK_SAMPLES:(i + 1) * VAD_CHUNK_SAMPLES]
                    try:
                        is_speech = bool(vad.is_speech(chunk))
                        vad_errors = 0
                    except Exception as exc:
                        # A VAD failure must never end the session. It is treated
                        # as "not speech" for this chunk only, and the VAD is
                        # rebuilt if it keeps failing - the old code would have
                        # let this propagate and kill the worker.
                        vad_errors += 1
                        self._log_exception(session, "vad", exc, accumulator=accumulator,
                                            extra=f"consecutive_vad_errors={vad_errors}")
                        is_speech = False
                        if vad_errors >= 25:
                            _log("VAD failed 25 times in a row - rebuilding it")
                            try:
                                vad, _ = _make_vad(self.vad_backend)
                                vad.reset()
                                vad_errors = 0
                                self._update_state(session, vad_backend=vad.name)
                            except Exception as rebuild_exc:
                                self._log_exception(session, "vad-rebuild", rebuild_exc,
                                                    accumulator=accumulator)
                    if is_speech:
                        session.last_speech_time = now
                    transition = accumulator.add(chunk, is_speech)
                    if transition and self.log_every_window:
                        print(f"[VAD] {transition} buffer={accumulator.buffered_seconds:.2f}s "
                              f"speech={accumulator.speech_seconds:.2f}s", flush=True)
                pending = pending[full * VAD_CHUNK_SAMPLES:]

            if session.stop_event.is_set():
                break

            # ---- microphone health ------------------------------------------------
            silent_for = now - last_audio
            stream_dead = not session.stream_alive()
            if silent_for > MIC_STALL_SECONDS or stream_dead:
                if silent_for > MIC_FATAL_SECONDS or reopen_attempts >= MIC_MAX_REOPEN_ATTEMPTS:
                    raise MicrophoneError(
                        f"No audio has arrived from the microphone for {silent_for:.0f}s "
                        f"after {reopen_attempts} attempt(s) to reopen it. It may have been "
                        f"unplugged, disabled, or taken by another application.")
                if silent_for > MIC_RECOVER_SECONDS and now - last_reopen > MIC_RECOVER_SECONDS:
                    reopen_attempts += 1
                    last_reopen = now
                    _log(f"No audio for {silent_for:.1f}s (stream_alive={not stream_dead}) - "
                         f"reopening the microphone, attempt {reopen_attempts}/"
                         f"{MIC_MAX_REOPEN_ATTEMPTS}")
                    try:
                        session.close_stream()
                        stream, rate, _, label = open_input_stream(self.input_device,
                                                                   session.callback)
                        with session.stream_lock:
                            session.stream = stream
                        session.sample_rate = rate
                        resampler = _StreamingResampler(rate, AUDIO_SAMPLE_RATE)
                        pending = np.zeros(0, dtype=np.float32)
                        last_audio = time.monotonic()
                        _log(f"Microphone reopened: {label} @ {rate} Hz - listening continues")
                        self._update_state(session, input_device=label, capture_sample_rate=rate,
                                           warning=self._set_warning(
                                               session, "stall",
                                               f"microphone stalled and was reopened "
                                               f"({reopen_attempts}x)"))
                    except Exception as exc:
                        self._log_exception(session, "mic-reopen", exc,
                                            accumulator=accumulator,
                                            extra=f"attempt={reopen_attempts}")
                else:
                    self._update_state(session, warning=self._set_warning(
                        session, "stall",
                        f"no audio from the microphone for {silent_for:.0f}s"))
            elif "stall" in session.warnings and silent_for < 1.0:
                self._update_state(session, warning=self._set_warning(session, "stall", None))
            if not heard_signal and not silent_mic_warned and now - started > MIC_STALL_SECONDS:
                silent_mic_warned = True
                message = ("Microphone delivers pure digital silence. On Windows check Settings > Privacy & "
                           "security > Microphone ('Let desktop apps access your microphone') and that the "
                           "device is not muted.")
                _log(f"WARNING: {message}")
                self._update_state(session, warning=self._set_warning(session, "silent_mic", message))
            elif heard_signal and "silent_mic" in session.warnings:
                self._update_state(session, warning=self._set_warning(session, "silent_mic", None))

            # ---- heartbeat: proves the worker is alive even when nobody speaks ----
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                last_heartbeat = now
                print(f"[VOICE HEARTBEAT] worker alive t={now - started:.0f}s "
                      f"callbacks={session.callback_count} dropped={session.dropped_blocks} "
                      f"queue={session.audio_queue.qsize()} "
                      f"buffer={accumulator.buffered_seconds:.2f}s "
                      f"speech={accumulator.speech_seconds:.2f}s "
                      f"predictions={session.inference_count} "
                      f"last_speech={now - session.last_speech_time:.1f}s ago", flush=True)

            # ---- is a window ready? ----------------------------------------------
            reason = accumulator.ready()
            if reason and (now - last_inference) >= MIN_INFERENCE_INTERVAL:
                window, speech_seconds, new_speech_seconds = accumulator.take(reason)
                last_inference = time.monotonic()
                self._classify_window(session, window, speech_seconds, new_speech_seconds,
                                      reason, accumulator)
                last_publish = time.monotonic()
                self._notify_update()
                continue

            # A phrase that ended with too little speech to classify is dropped,
            # so it cannot sit in the buffer blocking the next one.
            if accumulator.expired():
                if self.log_every_window:
                    print(f"[VAD] speech_ended discarded={accumulator.speech_seconds:.2f}s "
                          f"(under {EMOTION_MIN_SPEECH_SECONDS}s, too short to classify)", flush=True)
                accumulator.reset()

            # ---- idle publish: keep speech_detected and telemetry current --------
            if now - last_publish >= self.update_seconds:
                last_publish = now
                self._publish_idle(session, accumulator, now)
                self._notify_update()

    def _notify_update(self):
        if self.on_update is None:
            return
        try:
            self.on_update(self.get_status())
        except Exception as exc:
            _log(f"on_update callback failed: {exc}")
            traceback.print_exc()

    def _log_exception(self, session, stage, exc, accumulator=None, extra=""):
        """Full diagnostic for any failure inside the live loop. Never swallowed."""
        acc = accumulator if accumulator is not None else getattr(session, "accumulator", None)
        print(f"[VOICE ERROR] stage={stage} type={type(exc).__name__} message={exc}\n"
              f"  timestamp={time.time():.3f} "
              f"buffer={getattr(acc, 'buffered_seconds', float('nan')):.2f}s "
              f"speech={getattr(acc, 'speech_seconds', float('nan')):.2f}s "
              f"in_utterance={getattr(acc, 'in_utterance', None)} "
              f"silence={getattr(acc, 'silence_seconds', float('nan')):.2f}s\n"
              f"  queue={session.audio_queue.qsize()} callbacks={session.callback_count} "
              f"dropped={session.dropped_blocks} predictions={session.inference_count} "
              f"{extra}", flush=True)
        traceback.print_exc()

    def _publish_idle(self, session, accumulator, now):
        """State update for a cycle that did not classify anything."""
        speaking = accumulator.in_utterance and accumulator.silence_seconds < UTTERANCE_END_SECONDS
        values = {
            "speech_detected": bool(speaking),
            "buffered_seconds": round(accumulator.buffered_seconds, 2),
            "buffered_speech_seconds": round(accumulator.speech_seconds, 2),
            "seconds_since_speech": round(now - session.last_speech_time, 1),
            "speech_ratio": round(accumulator.speech_seconds /
                                  max(accumulator.buffered_seconds, 1e-6), 3)
                            if accumulator.buffered_seconds else 0.0,
            "skipped_cycles": session.skipped_cycles,
            "dropped_audio_blocks": session.dropped_blocks,
            "audio_callbacks": session.callback_count,
            "queue_depth": session.audio_queue.qsize(),
        }
        # An utterance that is well and truly over clears the per-window readout,
        # so the UI stops presenting a stale prediction as if it were current.
        if (now - session.last_speech_time > SILENCE_RESET_SECONDS
                and (session.prob_history or session.streak_count or session.last_emitted_emotion)):
            print(f"[VOICE STABILITY] silence > {SILENCE_RESET_SECONDS}s -> utterance ended, "
                  f"confident speech {session.streak_count:.1f}s/"
                  f"{self.stability_seconds:.1f}s cleared "
                  f"(last stable emotion is kept for display)", flush=True)
            session.prob_history.clear()
            session.streak_emotion = None
            session.streak_count = 0.0
            session.last_emitted_emotion = None
            session.last_emitted_time = 0.0
            values.update({
                "stability_count": 0, "confident_window": False,
                "emotion": None, "confidence": 0.0,
                "probabilities": {e: 0.0 for e in EMOTIONS},
                "raw_emotion": None, "raw_probabilities": {e: 0.0 for e in EMOTIONS},
                "margin": 0.0,
            })
        self._update_state(session, **values)

    @staticmethod
    def _drain(audio_queue, timeout):
        blocks = []
        try:
            if timeout:
                blocks.append(audio_queue.get(timeout=timeout))
            while True:
                blocks.append(audio_queue.get_nowait())
        except queue.Empty:
            pass
        return blocks

    def _classify_window(self, session, window, speech_seconds, new_speech_seconds,
                         reason, accumulator):
        """Classify one accumulated 2-4 s speech window and update the voice state.

        Everything that can fail here is contained: a bad buffer or a model error
        is logged with a traceback and the live loop carries on to the next window.
        """
        values = {
            "speech_detected": True,
            "buffered_seconds": round(accumulator.buffered_seconds, 2),
            "buffered_speech_seconds": round(accumulator.speech_seconds, 2),
            "window_reason": reason,
            "window_seconds": round(len(window) / float(AUDIO_SAMPLE_RATE), 2),
            "window_speech_seconds": round(speech_seconds, 2),
            "new_speech_seconds": round(new_speech_seconds, 2),
            "seconds_since_speech": 0.0,
            "speech_ratio": round(speech_seconds /
                                  max(len(window) / float(AUDIO_SAMPLE_RATE), 1e-6), 3),
            "skipped_cycles": session.skipped_cycles,
            "dropped_audio_blocks": session.dropped_blocks,
            "audio_callbacks": session.callback_count,
            "queue_depth": session.audio_queue.qsize(),
        }

        if window.size == 0 or not np.all(np.isfinite(window)):
            values["warning"] = self._set_warning(session, "audio", "Invalid audio buffer skipped")
            self._update_state(session, **values)
            return
        # HuBERT is unhappy with very short inputs; pad a short phrase with silence
        # rather than refusing to classify it (refusing is what used to lose every
        # normal-length phrase).
        if window.size < AUDIO_SAMPLE_RATE:
            window = np.pad(window, (0, AUDIO_SAMPLE_RATE - window.size))
        values["warning"] = self._set_warning(session, "audio", None)
        values["audio_level_db"] = round(_dbfs(window), 1)

        if self.log_every_window:
            print(f"[VOICE] inference started reason={reason} "
                  f"window={len(window) / AUDIO_SAMPLE_RATE:.2f}s "
                  f"speech={speech_seconds:.2f}s", flush=True)

        t0 = time.perf_counter()
        try:
            detail = self._model.predict_detailed(window)
            raw = detail["ensemble"]
        except Exception as exc:
            session.inference_errors += 1
            self._log_exception(session, "inference", exc, accumulator=accumulator,
                                extra=f"window_samples={window.size} reason={reason}")
            values["warning"] = self._set_warning(session, "inference",
                                                  f"Emotion inference failed: {exc}")
            self._update_state(session, **values)
            return
        inference_ms = (time.perf_counter() - t0) * 1000.0
        session.inference_errors = 0
        session.inference_count += 1
        values["warning"] = self._set_warning(session, "inference", None)

        members = detail.get("members") or {}
        values.update({
            "member_probabilities": {n: {e: round(p[e], 4) for e in EMOTIONS}
                                     for n, p in members.items()},
            "member_top": dict(detail.get("member_top") or {}),
            "member_ms": dict(detail.get("member_ms") or {}),
            "member_agreement": detail.get("agreement"),
            "member_dissenting": list(detail.get("dissenting") or []),
            "combiner": detail.get("combiner"),
        })
        if VOICE_LOG_MEMBERS and len(members) > 1:
            breakdown = " | ".join(
                f"{name}={max(probs, key=probs.get)[:3].upper()} {max(probs.values()) * 100:.0f}"
                for name, probs in members.items())
            print(f"[VOICE MEMBERS] combiner={detail.get('combiner')} "
                  f"agreement={detail.get('agreement')} -> {breakdown}", flush=True)
            if detail.get("dissenting"):
                print(f"[VOICE DISAGREE] {', '.join(detail['dissenting'])} disagree with the "
                      f"ensemble's {detail.get('emotion', '?').upper()}", flush=True)

        raw = self._apply_speaker_baseline(raw)
        now = time.monotonic()

        # ---- temporal smoothing: rolling AND time-bounded ---------------------
        # A plain deque(maxlen=N) is rolling in count but not in time: after a
        # pause the average still carried vectors from before it. Entries older
        # than EMOTION_HISTORY_SECONDS are dropped so old speech genuinely loses
        # influence instead of anchoring the session to its first prediction.
        session.prob_history.append((now, np.array([raw[e] for e in EMOTIONS], dtype=np.float64)))
        while session.prob_history and now - session.prob_history[0][0] > EMOTION_HISTORY_SECONDS:
            session.prob_history.popleft()
        mean = np.mean(np.stack([v for _, v in session.prob_history]), axis=0)
        smoothed = {e: float(mean[i]) for i, e in enumerate(EMOTIONS)}
        emotion = EMOTIONS[int(np.argmax(mean))]
        confidence = float(np.max(mean))
        raw_emotion = max(raw, key=raw.get)

        ordered = sorted(smoothed.values(), reverse=True)
        margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
        confident = (confidence >= self.confidence_threshold
                     and margin >= self.margin_threshold)

        previous_emotion = session.streak_emotion
        previous_count = session.streak_count

        # ------------------------------------------------------------------
        # Stability is accumulated in SECONDS OF CONFIDENT SPEECH.
        #
        # Counting consecutive confident windows was unreachable for ordinary
        # speech, and worse, it meant different things at different times:
        # windows are utterance-driven now, so "three in a row" is ~4.5 s of
        # talking during a long sentence but only ~2.4 s across three short
        # phrases. Measured on 3 s phrases, the count never passed 2 of 3, so the
        # live voice layer never published and fusion always fell back to the
        # face - which is exactly "voice emotion does nothing".
        #
        # Each window contributes the NEW speech it carries (take() subtracts the
        # part an earlier overlapping window was already credited for), so
        # overlap cannot inflate the score and the threshold means the same thing
        # however the speaker happens to phrase things. Debouncing is preserved:
        # a non-confident window withdraws its own worth of evidence, a different
        # emotion restarts the total, and the score is capped at twice the
        # threshold so a long confident run cannot bank unbounded credit.
        # ------------------------------------------------------------------
        credit = max(new_speech_seconds, 0.0)
        if not confident:
            # A window that fails the gate withdraws its own worth of evidence
            # instead of wiping the total - one borderline window in the middle
            # of a phrase should not erase the phrase.
            session.streak_count = max(0.0, session.streak_count - credit)
            if session.streak_count <= 0.0:
                session.streak_emotion = None
        elif emotion == session.streak_emotion:
            session.streak_count = min(session.streak_count + credit,
                                       self.stability_seconds * 2.0)
        else:
            session.streak_emotion = emotion
            session.streak_count = credit

        print(f"[VOICE] prediction #{session.inference_count} reason={reason} "
              f"window={values['window_seconds']}s speech={speech_seconds:.2f}s "
              f"{inference_ms:.0f}ms", flush=True)
        print(f"[VOICE RAW] emotion={raw_emotion.upper()} confidence={raw[raw_emotion] * 100:.0f} "
              f"| smoothed={emotion.upper()} confidence={confidence * 100:.0f} margin={margin:.2f} "
              f"| history={len(session.prob_history)}", flush=True)
        print(f"[VOICE STABILITY] previous={(previous_emotion or '-').upper()} "
              f"current={emotion.upper()} confident={confident} "
              f"confident_speech={previous_count:.1f}s/{self.stability_seconds:.1f}s -> "
              f"{session.streak_count:.1f}s/{self.stability_seconds:.1f}s "
              f"(+{credit:.2f}s new)", flush=True)

        event = None
        if session.streak_emotion and session.streak_count >= self.stability_seconds:
            stable_emotion = session.streak_emotion
            stable_confidence = float(smoothed[stable_emotion])
            changed = stable_emotion != session.last_emitted_emotion
            due = (now - session.last_emitted_time) >= STABLE_REEMIT_SECONDS
            if changed or due:
                session.last_emitted_emotion = stable_emotion
                session.last_emitted_time = now
                event = (stable_emotion, stable_confidence, dict(smoothed))
                values["stable_emotion"] = stable_emotion
                values["stable_confidence"] = round(stable_confidence, 4)
                print(f"[VOICE STABLE] emotion={stable_emotion.upper()} "
                      f"confidence={stable_confidence * 100:.0f} "
                      f"({'new' if changed else 'still holding'})", flush=True)

        values.update({
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "probabilities": {e: round(p, 4) for e, p in smoothed.items()},
            "raw_emotion": raw_emotion,
            "raw_probabilities": {e: round(p, 4) for e, p in raw.items()},
            "margin": round(margin, 4),
            "confident_window": confident,
            "stability_count": round(float(session.streak_count), 1),
            "last_inference_ms": round(inference_ms, 1),
            "inference_count": session.inference_count,
            "smoothing_window": len(session.prob_history),
            # Changes on EVERY prediction, so a frozen UI is now provable.
            "prediction_timestamp": time.time(),
            "prediction_index": session.inference_count,
        })
        self._update_state(session, **values)

        if event is not None:
            _log(f"Stable emotion: {event[0].upper()} confidence={event[1]:.2f}")
            if self.on_stable_emotion is not None:
                try:
                    self.on_stable_emotion(*event)
                except Exception as exc:
                    self._log_exception(session, "on_stable_emotion", exc,
                                        accumulator=accumulator)


# ============================================================
# COMMAND LINE SELF-TESTS (never touch the robot)
# ============================================================
def _read_wav_16k(path):
    """16-bit / 32-bit / 8-bit PCM WAV -> float32 mono at 16 kHz."""
    with wave.open(path, "rb") as wf:
        rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width * 8} bit")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return _StreamingResampler(rate, AUDIO_SAMPLE_RATE).process(data)


def _format_probs(probs):
    return "  ".join(f"{e}={probs[e]:.2f}" for e in EMOTIONS)


def _cli_list_devices():
    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return 1
    print(f"{'index':>5}  {'default':7}  {'host API':22}  {'rate':>6}  name")
    for d in devices:
        print(f"{d['index']:>5}  {'yes' if d['is_default'] else '':7}  {d['hostapi'][:22]:22}  "
              f"{d['default_samplerate']:>6}  {d['name']}")
    print("\nUse the index with --device N, or set VOICE_INPUT_DEVICE=N before starting app.py.")
    return 0


def _cli_test_mic(device, seconds, save_path):
    session = _Session()
    stream, rate, channels, label = open_input_stream(device, session.callback)
    session.stream = stream
    vad, warning = _make_vad(VAD_BACKEND)
    if warning:
        print(warning)
    resampler = _StreamingResampler(rate, AUDIO_SAMPLE_RATE)
    print(f"Recording {seconds:.0f} s from {label} at {rate} Hz, {channels} ch "
          f"(VAD: {vad.name}). Speak, then stay quiet ...\n")
    recorded, pending = [], np.zeros(0, dtype=np.float32)
    speech_chunks = total_chunks = 0
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            time.sleep(0.25)
            blocks = LiveEmotionDetector._drain(session.audio_queue, timeout=0)
            if not blocks:
                print("  (no audio received)")
                continue
            audio = resampler.process(np.concatenate(blocks))
            recorded.append(audio)
            pending = np.concatenate([pending, audio])
            full = len(pending) // VAD_CHUNK_SAMPLES
            flags = [vad.is_speech(pending[i * VAD_CHUNK_SAMPLES:(i + 1) * VAD_CHUNK_SAMPLES]) for i in range(full)]
            pending = pending[full * VAD_CHUNK_SAMPLES:]
            speech_chunks += sum(flags)
            total_chunks += len(flags)
            level = _dbfs(audio)
            bar = "#" * int(max(0.0, min(60.0, level + 60.0)) / 1.5)
            print(f"  {level:6.1f} dBFS  {'SPEECH ' if flags and any(flags) else 'silence'}  {bar}")
    finally:
        session.close_stream()

    audio = np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.float32)
    print(f"\nCaptured {len(audio) / AUDIO_SAMPLE_RATE:.1f} s, peak level {_dbfs(audio):.1f} dBFS RMS, "
          f"speech in {100.0 * speech_chunks / max(1, total_chunks):.0f}% of VAD chunks, "
          f"dropped blocks: {session.dropped_blocks}, overflows: {session.overflows}")
    if audio.size and not np.any(audio != 0):
        print("WARNING: all samples are exactly zero - microphone access is probably blocked by the OS.")
    if save_path and audio.size:
        with wave.open(save_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        print(f"Saved 16 kHz mono recording to {save_path} - play it back to check the capture.")
    return 0


def _cli_test_file(paths, show_windows):
    """Classify whole clips, printing every ensemble member's own vote."""
    model = build_emotion_model()
    info = model.describe()
    print(f"Model: {info['kind']} - {', '.join(info['members'])} "
          f"(combiner={info['combiner']}) on {DEVICE}")

    def show(prefix, detail):
        probs = detail["ensemble"]
        best = max(probs, key=probs.get)
        print(f"{prefix} {best.upper():7}  {_format_probs(probs)}")
        if len(detail.get("members") or {}) > 1:
            for name, member in detail["members"].items():
                top = max(member, key=member.get)
                mark = "  <-- disagrees" if name in detail.get("dissenting", []) else ""
                print(f"      {name:<22} {top.upper():7} {_format_probs(member)}{mark}")

    for path in paths:
        audio = _read_wav_16k(path)
        t0 = time.perf_counter()
        detail = model.predict_detailed(audio)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n{os.path.basename(path)}  ({len(audio) / AUDIO_SAMPLE_RATE:.1f} s, {ms:.0f} ms)")
        show("  whole clip ->", detail)
        if show_windows:
            # Same overlapping 2-4 s geometry the live path and the uploaded-file
            # path use, so the per-window view here is what the app really sees.
            win = int(AUDIO_WINDOW_SECONDS * AUDIO_SAMPLE_RATE)
            hop = int(UPLOAD_HOP_SECONDS * AUDIO_SAMPLE_RATE)
            for start in range(0, max(1, len(audio) - win + hop), hop):
                segment = audio[start:start + win]
                if len(segment) < MIN_ANALYSIS_SECONDS * AUDIO_SAMPLE_RATE:
                    break
                at = start / AUDIO_SAMPLE_RATE
                show(f"  {at:4.1f}-{(start + len(segment)) / AUDIO_SAMPLE_RATE:4.1f} s ->",
                     model.predict_detailed(segment))
    return 0


def _cli_live(device, seconds):
    def on_stable(emotion, confidence, probabilities):
        print(f"[ROBOT SIMULATION] Would trigger {emotion}", flush=True)

    detector = LiveEmotionDetector(input_device=device, on_stable_emotion=on_stable, log_every_window=True)
    ok, message = detector.start()
    print(message)
    if not ok:
        return 1
    print("Speak with different tones. Ctrl+C to stop.\n")
    started = time.monotonic()
    try:
        while True:
            time.sleep(0.5)
            status = detector.get_status()
            if not status["running"]:
                print(f"Detector stopped: {status['error']}")
                return 1
            if seconds and time.monotonic() - started > seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Live acoustic speech emotion recognition self-tests")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-devices", action="store_true", help="list microphones")
    mode.add_argument("--test-mic", action="store_true", help="capture audio and show level + VAD (no model)")
    mode.add_argument("--test-file", nargs="+", metavar="WAV", help="run the emotion model on WAV files")
    mode.add_argument("--live", action="store_true", help="full live pipeline, robot simulated")
    parser.add_argument("--device", default=os.environ.get("VOICE_INPUT_DEVICE"),
                        help="microphone index or name substring (default: system default)")
    parser.add_argument("--seconds", type=float, default=0, help="duration for --test-mic / --live")
    parser.add_argument("--save", metavar="WAV", help="--test-mic: save the recording")
    parser.add_argument("--windows", action="store_true", help="--test-file: also show rolling 3 s windows")
    args = parser.parse_args(argv)

    try:
        if args.list_devices:
            return _cli_list_devices()
        if args.test_mic:
            return _cli_test_mic(args.device, args.seconds or 10, args.save)
        if args.test_file:
            return _cli_test_file(args.test_file, args.windows)
        return _cli_live(args.device, args.seconds)
    except MicrophoneError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

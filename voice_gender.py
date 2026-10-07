"""
voice_gender.py - speaker gender (male / female) from the voice, for the /voice page.

Adapted from https://github.com/24f3003330/Gender-Detection (commit 466ce70):

    speech -> frozen HuBERT (facebook/hubert-base-ls960), layer 2, mean-pooled
           -> StandardScaler -> RBF SVM (predict_proba, Platt-scaled)
           -> male / female + probability

It does NOT open the microphone. emotion_detector.py already runs the only
microphone stream and Silero VAD; it hands every 32 ms VAD chunk to
LiveGenderTracker.feed(chunk, is_speech). The tracker cuts speech segments the
same way the original project's live mode does (1.5-3 s of VAD speech with
~100 ms padding, averaged over the last 3 segments, reset after 6 s of silence)
and classifies them on its own worker thread, so the emotion pipeline is never
slowed down or blocked by it. A gender failure is logged and shown as
"unavailable"; it can never stop voice emotion recognition.

The resampler, VAD model, chunk size and threshold in emotion_detector.py are the
same as in the original project, so the classifier sees the same kind of input it
was trained and evaluated on.

Model files live in models/gender/. They are copied from the original repository
and, if missing, downloaded from its pinned commit (checksums verified).
Label mapping (models/label_mapping.json): 0 = male, 1 = female.
"""

import hashlib
import json
import os
import queue
import threading
import time
import traceback
import urllib.request
import warnings
from collections import deque

import numpy as np

try:
    import torch

    _TORCH_IMPORT_ERROR = None
except Exception as exc:  # ImportError, or OSError such as "DLL load failed"
    torch = None
    _TORCH_IMPORT_ERROR = exc

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gender")

_REPO_RAW = ("https://raw.githubusercontent.com/24f3003330/Gender-Detection/"
             "466ce704df187a04dd846636d0f83d3d8aea18aa/models/")
# name -> (url, sha256)
MODEL_FILES = {
    "gender_classifier.pkl": (_REPO_RAW + "gender_classifier.pkl",
                              "20c3b6e86b2ea70458f213aee8dae87219401eac0b2813dabc2e48efbf16ac8b"),
    "scaler.pkl": (_REPO_RAW + "scaler.pkl",
                   "fc0c2140b254802c5b26644813a61107fdeb41190ac723c6ce55adb6b5bc8223"),
    "label_mapping.json": (_REPO_RAW + "label_mapping.json",
                           "52658379a54e31b935afa8988716acb929b0802d52a1379979d2d5e9560bbf18"),
    "feature_config.json": (_REPO_RAW + "feature_config.json",
                            "13c3a87e0141c147e11b6c16e72103ce34d42598393ddc624f4754827b5c53ce"),
}

SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512               # emotion_detector.py hands over 32 ms VAD chunks
MAX_SEGMENT_SECONDS = 6.0             # longest piece fed to HuBERT in one pass
MIN_SEGMENT_SECONDS = 0.5

# Live segmentation - same values as the original project's LiveGenderDetector.
VAD_PAD_CHUNKS = 3                    # keep ~100 ms of non-speech around speech
LIVE_TARGET_SPEECH_SECONDS = 3.0      # classify when this much speech is buffered
LIVE_MIN_SPEECH_SECONDS = 1.5         # ...or when a shorter phrase ends with silence
LIVE_UTTERANCE_END_SECONDS = 0.7
LIVE_RESET_SILENCE_SECONDS = 6.0      # long silence starts a fresh speaker/average
LIVE_SMOOTHING_SEGMENTS = 3           # average probabilities of the last N segments

# "mps" on Apple Silicon keeps this model off the CPU cores the emotion model
# uses. Override with VOICE_GENDER_DEVICE=cpu|mps|cuda.
GENDER_DEVICE = os.environ.get("VOICE_GENDER_DEVICE") or None


def _log(message):
    print(f"[VOICE GENDER] {message}", flush=True)


def _pick_device():
    if GENDER_DEVICE:
        return GENDER_DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_file(name):
    url, expected = MODEL_FILES[name]
    path = os.path.join(MODEL_DIR, name)
    if os.path.isfile(path) and _sha256(path) == expected:
        return path
    os.makedirs(MODEL_DIR, exist_ok=True)
    _log(f"Downloading {name} (first run needs internet) ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    actual = _sha256(tmp)
    if actual != expected:
        os.remove(tmp)
        raise RuntimeError(f"{name} failed its checksum (got {actual[:12]}..., "
                           f"expected {expected[:12]}...)")
    os.replace(tmp, path)
    return path


def split_segments(speech, max_s=MAX_SEGMENT_SECONDS, min_s=MIN_SEGMENT_SECONDS):
    """Cut speech into <= max_s pieces; a short tail is dropped unless it is the only piece."""
    size = int(max_s * SAMPLE_RATE)
    pieces = [speech[i:i + size] for i in range(0, len(speech), size)]
    kept = [p for p in pieces if len(p) >= min_s * SAMPLE_RATE]
    if not kept and len(speech) >= 0.25 * SAMPLE_RATE:
        kept = [speech]
    return kept


class GenderModel:
    """Frozen HuBERT embedder + the trained scaler/SVM. Loaded once; predict() is serialised."""

    def __init__(self):
        if torch is None:
            raise RuntimeError(f"PyTorch is not available: {_TORCH_IMPORT_ERROR}")
        import joblib
        from transformers import AutoFeatureExtractor, AutoModel

        paths = {name: _ensure_file(name) for name in MODEL_FILES}
        with open(paths["label_mapping.json"]) as f:
            self.id_to_label = {int(k): v for k, v in json.load(f)["id_to_label"].items()}
        with open(paths["feature_config.json"]) as f:
            self.config = json.load(f)
        self.layer = self.config["layer"]
        if self.config.get("pooling", "mean") != "mean":
            raise RuntimeError(f"unsupported pooling {self.config.get('pooling')!r}")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.clf = joblib.load(paths["gender_classifier.pkl"])
            self.scaler = joblib.load(paths["scaler.pkl"])
        for w in caught:
            # The pickles were written by scikit-learn 1.9.1; loading them in an
            # older release was checked to give identical probabilities, but the
            # warning is reported rather than hidden.
            _log(f"note while loading classifier: {type(w.message).__name__}: "
                 f"{str(w.message).splitlines()[0]}")

        self.device = _pick_device()
        encoder = self.config["encoder"]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(encoder)
        model = AutoModel.from_pretrained(encoder)
        if isinstance(self.layer, int):
            # Only hidden_states[layer] is used, and it depends on the first
            # `layer` transformer layers alone, so the rest are dropped (same
            # features, a fraction of the compute).
            model.encoder.layers = model.encoder.layers[:max(1, self.layer)]
        self.model = model.to(self.device).eval()
        self._lock = threading.Lock()
        _log(f"model ready: {encoder} layer={self.layer} + SVM on {self.device}; "
             f"labels {self.id_to_label}")

    def _embed(self, segments):
        inputs = self.feature_extractor(list(segments), sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt", padding=True)
        x = inputs.input_values.to(self.device)
        with torch.inference_mode():
            out = self.model(x, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=1)                      # B, L+1, T, H
            lengths = self.model._get_feat_extract_output_lengths(
                torch.tensor([len(a) for a in segments])).to(self.device)
            steps = torch.arange(hs.shape[2], device=self.device)
            mask = (steps[None, :] < lengths[:, None]).to(hs.dtype)         # B, T (ignore padding)
            pooled = (hs * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]
        pooled = pooled.float().cpu().numpy()
        if self.layer == "avg":
            return pooled.mean(axis=1)
        return pooled[:, int(self.layer), :]

    def predict(self, speech):
        """speech: float32 16 kHz VAD speech -> {"gender", "confidence", "probabilities", "speech_seconds"}."""
        segments = split_segments(np.asarray(speech, dtype=np.float32))
        if not segments:
            raise ValueError("not enough speech to classify")
        with self._lock:
            X = self.scaler.transform(self._embed(segments))
            proba = self.clf.predict_proba(X)                     # columns follow clf.classes_
        weights = np.array([len(s) for s in segments], dtype=np.float64)
        avg = (proba * weights[:, None]).sum(0) / weights.sum()
        probs = {self.id_to_label[int(c)]: float(p) for c, p in zip(self.clf.classes_, avg)}
        gender = max(probs, key=probs.get)
        return {"gender": gender, "confidence": probs[gender], "probabilities": probs,
                "speech_seconds": round(len(speech) / float(SAMPLE_RATE), 2)}


class LiveGenderTracker:
    """Builds speech segments from the voice pipeline's VAD chunks and classifies them.

    feed() runs on the voice emotion worker thread and only appends to a list;
    classification happens on this tracker's own "VoiceGenderWorker" thread.
    Thread-safe; never raises into its caller.
    """

    _CHUNK_SECONDS = VAD_CHUNK_SAMPLES / float(SAMPLE_RATE)

    def __init__(self, load_after=None):
        # load_after: blocking callable run before this model loads. transformers'
        # lazy imports are not thread-safe - two models loading at once can fail
        # with "cannot import name ..." in EITHER thread - so app.py passes the
        # voice emotion model's loader here and the two load one after the other.
        self._load_after = load_after
        self._model = None
        self._model_lock = threading.Lock()
        self._loader = None
        self._worker = None
        self._worker_lock = threading.Lock()
        self._queue = queue.Queue(maxsize=2)
        self._state_lock = threading.Lock()
        self._generation = 0          # bumped on start/stop/long silence: older segments are stale
        self._feed_errors = 0
        self._reset_segmenter()
        self._state = self._idle_state()
        self._state.update(model_loaded=False, model_loading=False, model_error=None, device=None)

    @staticmethod
    def _idle_state():
        return {"running": False, "state": "stopped", "gender": None, "confidence": None,
                "probabilities": {}, "segment_gender": None, "segment_confidence": None,
                "segments_classified": 0, "speech_seconds_buffered": 0.0, "error": None,
                "last_inference_ms": None, "updated_at": time.time()}

    def _reset_segmenter(self):
        self._pre_pad = deque(maxlen=VAD_PAD_CHUNKS)
        self._speech = []
        self._tail_pad = 0
        self._silence_s = 0.0

    def _update(self, generation=None, **values):
        with self._state_lock:
            if generation is not None and generation != self._generation:
                return False
            self._state.update(values)
            self._state["updated_at"] = time.time()
            return True

    # ------------------------------------------------------------------ public API
    def get_status(self):
        with self._state_lock:
            status = dict(self._state)
            status["probabilities"] = dict(self._state["probabilities"])
        if status["gender"]:
            status["label"] = status["gender"].capitalize()
        return status

    def preload_async(self):
        """Load the model in the background (idempotent). Called when /voice opens."""
        with self._model_lock:
            if self._model is not None or (self._loader is not None and self._loader.is_alive()):
                return False
            self._loader = threading.Thread(target=self._ensure_model,
                                            name="VoiceGenderModelLoader", daemon=True)
            self._loader.start()
        return True

    def start_session(self):
        """The microphone just started: begin a fresh speaker average."""
        self._ensure_worker()
        self.preload_async()
        with self._state_lock:
            if self._state["running"]:
                return                            # same microphone session, keep its average
            self._generation += 1
            self._reset_segmenter()
            keep = {k: self._state[k] for k in ("model_loaded", "model_loading", "model_error", "device")}
            self._state = self._idle_state()
            self._state.update(keep, running=True,
                               state="listening" if keep["model_loaded"] else "loading_model")
        self._drain_queue()

    def stop_session(self):
        """The microphone stopped: drop any buffered speech and the shown result."""
        with self._state_lock:
            if not self._state["running"]:
                return
            self._generation += 1
            self._reset_segmenter()
            keep = {k: self._state[k] for k in ("model_loaded", "model_loading", "model_error", "device")}
            self._state = self._idle_state()
            self._state.update(keep)
        self._drain_queue()

    def feed(self, chunk, is_speech):
        """One 32 ms VAD chunk from the voice pipeline. Cheap; never raises."""
        try:
            self._feed(chunk, is_speech)
        except Exception as exc:
            self._feed_errors += 1
            if self._feed_errors <= 5 or self._feed_errors % 500 == 0:
                _log(f"ERROR in feed (#{self._feed_errors}): {type(exc).__name__}: {exc}")
                traceback.print_exc()

    # ------------------------------------------------------------------ internals
    def _feed(self, chunk, is_speech):
        # Segment building copied from the original LiveGenderDetector._worker.
        with self._state_lock:
            if not self._state["running"]:
                return
            generation = self._generation
            if is_speech:
                self._speech.extend(self._pre_pad)
                self._pre_pad.clear()
                self._speech.append(chunk)
                self._tail_pad = VAD_PAD_CHUNKS
                self._silence_s = 0.0
            else:
                self._silence_s += self._CHUNK_SECONDS
                if self._tail_pad > 0 and self._speech:
                    self._speech.append(chunk)
                    self._tail_pad -= 1
                else:
                    self._pre_pad.append(chunk)
            buffered = len(self._speech) * self._CHUNK_SECONDS
            self._state["speech_seconds_buffered"] = round(buffered, 2)

            segment = None
            if buffered >= LIVE_TARGET_SPEECH_SECONDS or (
                    buffered >= LIVE_MIN_SPEECH_SECONDS
                    and self._silence_s >= LIVE_UTTERANCE_END_SECONDS):
                segment = np.concatenate(self._speech)
                self._speech = []
            elif self._silence_s >= LIVE_UTTERANCE_END_SECONDS and 0 < buffered < LIVE_MIN_SPEECH_SECONDS:
                self._speech = []                 # a short blip (cough, click) - discard it

            if (self._silence_s >= LIVE_RESET_SILENCE_SECONDS
                    and self._state["segments_classified"] and self._state["gender"] is not None):
                # Long silence: the next speaker may be someone else.
                self._generation += 1
                generation = self._generation
                self._state.update(gender=None, confidence=None, probabilities={},
                                   segment_gender=None, segment_confidence=None)
                _log(f"silence > {LIVE_RESET_SILENCE_SECONDS:.0f}s - gender average reset")

        if segment is not None:
            item = (generation, segment)
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Keep the newest speech: drop the oldest waiting segment.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    pass

    def _drain_queue(self):
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _ensure_model(self):
        """Load once. Returns the model or None; a failure is recorded, never raised."""
        with self._model_lock:
            if self._model is not None:
                return self._model
            self._update(model_loading=True, model_error=None)
            if self._load_after is not None:
                try:
                    self._load_after()
                except Exception as exc:
                    _log(f"load_after hook failed ({type(exc).__name__}: {exc}); loading anyway")
            started = time.time()
            _log("loading gender model ...")
            try:
                model = GenderModel()
            except Exception as exc:
                traceback.print_exc()
                message = (f"Voice gender model unavailable: {exc}. The first run needs internet "
                           f"to download facebook/hubert-base-ls960 (~360 MB).")
                _log(f"ERROR: {message}")
                with self._state_lock:
                    self._state.update(model_loading=False, model_loaded=False,
                                       model_error=message,
                                       state="unavailable" if self._state["running"] else "stopped")
                return None
            self._model = model
            _log(f"gender model loaded in {time.time() - started:.1f} s")
            with self._state_lock:
                self._state.update(model_loading=False, model_loaded=True, model_error=None,
                                   device=model.device)
                if self._state["running"] and self._state["state"] == "loading_model":
                    self._state["state"] = "listening"
            return model

    def _ensure_worker(self):
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._worker_main,
                                                name="VoiceGenderWorker", daemon=True)
                self._worker.start()

    def _worker_main(self):
        history = deque(maxlen=LIVE_SMOOTHING_SEGMENTS)
        history_generation = None
        while True:
            generation, segment = self._queue.get()
            try:
                if generation != self._generation:
                    continue                                   # stale: mic stopped or speaker reset
                model = self._ensure_model()
                if model is None:
                    continue
                t0 = time.perf_counter()
                result = model.predict(segment)
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                if history_generation != generation:
                    history.clear()
                    history_generation = generation
                history.append(result["probabilities"])
                smooth = {k: float(np.mean([h[k] for h in history])) for k in result["probabilities"]}
                gender = max(smooth, key=smooth.get)
                with self._state_lock:
                    if generation != self._generation:
                        continue
                    count = self._state["segments_classified"] + 1
                    self._state.update(gender=gender, confidence=smooth[gender],
                                       probabilities=smooth, segment_gender=result["gender"],
                                       segment_confidence=result["confidence"],
                                       segments_classified=count, error=None,
                                       last_inference_ms=elapsed_ms, state="listening",
                                       updated_at=time.time())
                _log(f"segment {result['speech_seconds']:.1f}s -> {result['gender']} "
                     f"{result['confidence']:.1%} | smoothed {gender} {smooth[gender]:.1%} "
                     f"({elapsed_ms:.0f} ms)")
            except Exception as exc:
                _log(f"ERROR: classification failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                self._update(generation, error=f"gender classification failed: {exc}")

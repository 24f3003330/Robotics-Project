"""
voice_age/inference.py - estimated AGE GROUP from voice, offline and live.

  AgeEstimator     models/voice_age_model/ -> predict(16 kHz speech) ->
                   {"age_group", "confidence", "probabilities"}
  LiveAgeTracker   receives the /voice page's existing VAD chunks (the same single
                   microphone stream the emotion and gender pipelines use - it never
                   opens a microphone), waits for AGE_INFERENCE_WINDOW_SECONDS of
                   speech, classifies on its own worker thread and averages the
                   probabilities of the last AGE_SMOOTHING_WINDOWS windows.

Separate from the voice EMOTION model: different encoder instance, different
weights, different thread. An age failure is reported as a state and never
affects emotion (or gender) recognition. If no model has been trained yet the
status says so; nothing crashes.
"""

import json
import os
import queue
import threading
import time
import traceback
from collections import deque

import numpy as np

from . import config
from .audio import AudioError, prepare


def _log(message):
    print(f"[VOICE AGE] {message}", flush=True)


class AgeModelUnavailable(RuntimeError):
    pass


class AgeEstimator:
    def __init__(self, model_dir=config.MODEL_DIR):
        self.model_dir = model_dir
        self.metadata_path = os.path.join(model_dir, "metadata.json")
        self.head_path = os.path.join(model_dir, "head.npz")
        self._lock = threading.Lock()
        self._loaded = None
        self.error = None

    # ---------------------------------------------------------------- status
    def available(self):
        return os.path.isfile(self.metadata_path) and os.path.isfile(self.head_path)

    def metadata(self):
        if not self.available():
            return None
        with open(self.metadata_path) as fh:
            return json.load(fh)

    def status(self):
        meta = self.metadata() if self.available() else None
        out = {"model_available": meta is not None, "model_path": os.path.relpath(self.model_dir, config.PROJECT_DIR),
               "inference_ready": self._loaded is not None, "error": self.error,
               "label": config.UI_LABEL, "disclaimer": config.DISCLAIMER}
        if meta is None:
            out["message"] = "Voice age model not trained yet. Run: python scripts/train_voice_age.py"
            return out
        test = meta.get("test_metrics", {})
        out.update({"model_version": meta.get("model_version"), "age_classes": meta["classes"],
                    "dataset": f"{meta['dataset']['version']} ({meta['dataset']['repo']}, {meta['dataset']['language']})",
                    "encoder": meta["architecture"]["encoder"], "trained_at": meta.get("trained_at"),
                    "test_accuracy": test.get("accuracy"), "test_balanced_accuracy": test.get("balanced_accuracy"),
                    "test_macro_f1": test.get("macro_f1")})
        return out

    # ---------------------------------------------------------------- loading
    def load(self, encoder_factory=None):
        """Load head + truncated frozen encoder (once). Raises AgeModelUnavailable."""
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            if not self.available():
                self.error = "Voice age model not trained yet."
                raise AgeModelUnavailable(self.error)
            try:
                meta = self.metadata()
                arch = meta["architecture"]
                state = np.load(self.head_path, allow_pickle=True)["state"].item()
                state = _unpack(state)
                if state["n_classes"] != len(meta["classes"]):
                    raise ValueError("head and metadata disagree on the number of classes")
                if encoder_factory is None:
                    from .embeddings import SpeechEncoder
                    encoder = SpeechEncoder(arch["encoder"], keep_layers=max(1, arch["layer"]))
                else:
                    encoder = encoder_factory(arch)
                from .model import head_from_state
                head = head_from_state(state, len(state["mean"]))
            except AgeModelUnavailable:
                raise
            except Exception as exc:
                self.error = f"Voice age model could not be loaded: {type(exc).__name__}: {exc}"
                raise AgeModelUnavailable(self.error) from exc
            self._loaded = {"meta": meta, "encoder": encoder, "head": head, "layer": arch["layer"],
                            "classes": meta["classes"]}
            self.error = None
            return self._loaded

    # ---------------------------------------------------------------- predict
    def predict(self, speech, rate=config.SAMPLE_RATE):
        """speech (any rate, mono or stereo float) -> age-group estimate. Raises AudioError for
        too little / silent / invalid audio and AgeModelUnavailable without a model."""
        m = self.load()
        x = prepare(speech, rate, min_seconds=config.MIN_CLIP_SECONDS,
                    max_seconds=float("inf"), trim=True)
        seg = int(config.MAX_CLIP_SECONDS * config.SAMPLE_RATE)
        pieces = [x[i:i + seg] for i in range(0, len(x), seg)]
        pieces = [p for p in pieces if len(p) >= config.MIN_CLIP_SECONDS * config.SAMPLE_RATE] or [x[:seg]]
        emb = m["encoder"].embed(pieces)[:, m["layer"], :]
        proba = m["head"].predict_proba(emb)
        w = np.array([len(p) for p in pieces], dtype=float)
        avg = (proba * w[:, None]).sum(0) / w.sum()
        return result_from_probabilities(avg, m["classes"], seconds=len(x) / config.SAMPLE_RATE)


def save_model(model_dir, head_state, metadata):
    """Write head.npz (plain arrays, no pickled code) + metadata.json."""
    def pack(st):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else pack(v) if isinstance(v, dict) else v)
                for k, v in st.items()}

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    os.makedirs(model_dir, exist_ok=True)
    np.savez(os.path.join(model_dir, "head.npz"), state=np.array(pack(head_state), dtype=object))
    with open(os.path.join(model_dir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2, default=default)


def result_from_probabilities(p, classes, seconds=None):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    i = int(np.argmax(p))
    out = {"age_group": classes[i], "confidence": float(p[i]),
           "probabilities": {c: float(v) for c, v in zip(classes, p)}}
    if seconds is not None:
        out["speech_seconds"] = round(float(seconds), 2)
    return out


def _unpack(st):
    out = {}
    for k, v in st.items():
        if isinstance(v, dict) and k != "hp":
            out[k] = _unpack(v)
        elif isinstance(v, list):
            out[k] = np.asarray(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# live tracker (fed by the voice pipeline)
# ---------------------------------------------------------------------------
VAD_CHUNK_SAMPLES = 512
_CHUNK_SECONDS = VAD_CHUNK_SAMPLES / float(config.SAMPLE_RATE)
PAD_CHUNKS = 3


class LiveAgeTracker:
    """States: model_not_trained | loading_model | listening | insufficient_speech |
    estimated | unavailable | stopped. feed() never raises into the voice pipeline."""

    def __init__(self, estimator=None, load_after=None):
        self.estimator = estimator or AgeEstimator()
        self._load_after = load_after
        self._queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._worker = None
        self._loader = None
        self._generation = 0
        self._feed_errors = 0
        self._reset_segmenter()
        self._state = self._idle()

    def _idle(self):
        return {"running": False, "state": "stopped", "age_group": None, "confidence": None,
                "probabilities": {}, "windows_classified": 0, "speech_seconds_buffered": 0.0,
                "last_window_age_group": None, "last_inference_ms": None, "error": None,
                "updated_at": time.time()}

    def _reset_segmenter(self):
        self._pre = deque(maxlen=PAD_CHUNKS)
        self._speech = []
        self._tail = 0
        self._silence = 0.0
        self._history = deque(maxlen=max(1, config.AGE_SMOOTHING_WINDOWS))

    # ------------------------------------------------------------ public
    def get_status(self):
        with self._lock:
            st = dict(self._state)
            st["probabilities"] = dict(self._state["probabilities"])
        model = self.estimator.status()
        st.update(model_available=model["model_available"], inference_ready=model["inference_ready"],
                  label=config.UI_LABEL, disclaimer=config.DISCLAIMER,
                  window_seconds=config.AGE_INFERENCE_WINDOW_SECONDS,
                  smoothing_windows=config.AGE_SMOOTHING_WINDOWS)
        if not model["model_available"]:
            st["state"] = "model_not_trained"
            st["message"] = model["message"]
        elif self.estimator.error and not model["inference_ready"]:
            st["state"] = "unavailable"
            st["message"] = self.estimator.error
        return st

    def preload_async(self):
        if not self.estimator.available() or self.estimator._loaded is not None:
            return False
        if self._loader is not None and self._loader.is_alive():
            return False
        self._loader = threading.Thread(target=self._load, name="VoiceAgeModelLoader", daemon=True)
        self._loader.start()
        return True

    def start_session(self):
        self._ensure_worker()
        self.preload_async()
        with self._lock:
            if self._state["running"]:
                return
            self._generation += 1
            self._reset_segmenter()
            self._state = self._idle()
            self._state.update(running=True, state="listening")
        self._drain()

    def stop_session(self):
        with self._lock:
            if not self._state["running"]:
                return
            self._generation += 1
            self._reset_segmenter()
            self._state = self._idle()
        self._drain()

    def feed(self, chunk, is_speech):
        """One 32 ms VAD chunk from the voice pipeline. Cheap; never raises."""
        try:
            self._feed(chunk, is_speech)
        except Exception as exc:
            self._feed_errors += 1
            if self._feed_errors <= 5 or self._feed_errors % 500 == 0:
                _log(f"ERROR in feed (#{self._feed_errors}): {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ internals
    def _feed(self, chunk, is_speech):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)     # rejects None / junk before buffering
        if chunk.size == 0 or not np.all(np.isfinite(chunk)):
            raise ValueError("empty or non-finite audio chunk")
        segment = None
        with self._lock:
            if not self._state["running"]:
                return
            if is_speech:
                self._speech.extend(self._pre)
                self._pre.clear()
                self._speech.append(chunk)
                self._tail = PAD_CHUNKS
                self._silence = 0.0
            else:
                self._silence += _CHUNK_SECONDS
                if self._tail > 0 and self._speech:
                    self._speech.append(chunk)
                    self._tail -= 1
                else:
                    self._pre.append(chunk)
            buffered = len(self._speech) * _CHUNK_SECONDS
            self._state["speech_seconds_buffered"] = round(buffered, 2)
            ended = self._silence >= config.AGE_UTTERANCE_END_SECONDS
            if buffered >= config.AGE_INFERENCE_WINDOW_SECONDS or (ended and buffered >= config.AGE_MIN_SPEECH_SECONDS):
                segment = (self._generation, np.concatenate(self._speech))
                self._speech = []
            elif ended and 0 < buffered < config.AGE_MIN_SPEECH_SECONDS:
                self._speech = []                       # too little speech for an age estimate
                if self._state["age_group"] is None:
                    self._state["state"] = "insufficient_speech"
            if self._silence >= config.AGE_RESET_SILENCE_SECONDS and self._state["age_group"] is not None:
                self._generation += 1                   # next speaker may be someone else
                self._history.clear()
                self._state.update(age_group=None, confidence=None, probabilities={}, state="listening")
        if segment is not None:
            try:
                self._queue.put_nowait(segment)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(segment)
                except queue.Full:
                    pass

    def _drain(self):
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _load(self):
        if self._load_after is not None:
            try:
                self._load_after()
            except Exception as exc:
                _log(f"load_after hook failed ({type(exc).__name__}: {exc}); loading anyway")
        t0 = time.time()
        try:
            m = self.estimator.load()
            _log(f"model ready in {time.time() - t0:.1f} s: {m['meta']['architecture']['encoder']} "
                 f"layer {m['layer']} + {m['meta']['architecture']['head']}; classes {m['classes']}")
        except AgeModelUnavailable as exc:
            _log(str(exc))

    def _ensure_worker(self):
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="VoiceAgeWorker", daemon=True)
            self._worker.start()

    def _work(self):
        while True:
            generation, segment = self._queue.get()
            try:
                if generation != self._generation:
                    continue
                if self.estimator._loaded is None:
                    if not self.estimator.available():
                        continue
                    self._load()
                    if self.estimator._loaded is None:
                        continue
                t0 = time.perf_counter()
                try:
                    r = self.estimator.predict(segment)
                except AudioError as exc:
                    with self._lock:
                        if generation == self._generation and self._state["age_group"] is None:
                            self._state.update(state="insufficient_speech", error=None)
                    _log(f"window skipped: {exc}")
                    continue
                ms = round((time.perf_counter() - t0) * 1000.0, 1)
                with self._lock:
                    if generation != self._generation:
                        continue
                    self._history.append(r["probabilities"])
                    classes = list(r["probabilities"])
                    avg = np.mean([[h[c] for c in classes] for h in self._history], axis=0)
                    smooth = result_from_probabilities(avg, classes)
                    self._state.update(age_group=smooth["age_group"], confidence=smooth["confidence"],
                                       probabilities=smooth["probabilities"], state="estimated",
                                       windows_classified=self._state["windows_classified"] + 1,
                                       last_window_age_group=r["age_group"], last_inference_ms=ms, error=None,
                                       updated_at=time.time())
                _log(f"window {r['speech_seconds']:.1f}s -> {r['age_group']} {r['confidence']:.0%} | "
                     f"smoothed over {len(self._history)}: {smooth['age_group']} {smooth['confidence']:.0%} ({ms:.0f} ms)")
            except Exception as exc:
                _log(f"ERROR: age inference failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                with self._lock:
                    self._state["error"] = f"age inference failed: {exc}"

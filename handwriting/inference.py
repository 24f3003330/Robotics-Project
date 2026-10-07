"""
handwriting/inference.py - run a trained checkpoint on canvas strokes.

The checkpoint is loaded once (lazily, or in the background when /handwriting
opens) and reused for every request. If it does not exist the analyzer reports
"not installed" - it never produces a prediction without a trained model.
Labels, features and normalisation all come from the checkpoint, which must
have been trained on EMOTHAW's three targets (Anxiety / Depression / Stress).

Response (POST /handwriting/analyze):
    predictions    {"Anxiety": "Elevated", "Depression": "Normal", "Stress": "Elevated"}
    probabilities  {"Anxiety": 0.82, ...}  - probability of "elevated" for each target
    heads          per target: label, confidence of that label, both class probabilities

The same class also serves the separate 4-class pipeline (kind="4class":
Happy / Neutral / Sad / Angry, models/handwriting_4class.pt, POST
/handwriting/4class/analyze):
    emotion, confidence, probabilities {"Happy": .., "Neutral": .., "Sad": .., "Angry": ..}
    prototype      true when the checkpoint was trained on SEED data (graphology
                   references), with the notice to show next to the prediction
The 4-class response is the same whether the checkpoint came from seed_4class or
real_4class - only "prototype" / "dataset" change.
"""

import os
import threading
import time
import traceback

import numpy as np

from . import config
from .preprocessing import HandwritingInputError, feature_matrix, from_browser_points, prepare, windows

try:
    import torch

    _TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc


# Features derived from timing: meaningless for a static image (no real timestamps).
TIMING_FEATURES = {"log_dt", "log_speed", "log_accel"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_PIXELS = 40_000_000


def _image_dimensions(data):
    """(width, height) of encoded image bytes, or None when they cannot be decoded."""
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    return None if img is None else (int(img.shape[1]), int(img.shape[0]))


class HandwritingModelUnavailable(RuntimeError):
    """No usable trained model; the message is meant for the user."""


def _log(message):
    print(f"[HANDWRITING] {message}", flush=True)


class HandwritingAnalyzer:
    def __init__(self, checkpoint_path=config.DEFAULT_CHECKPOINT, kind="emothaw"):
        """kind: "emothaw" (Anxiety / Depression / Stress) or "4class" (Happy / Neutral / Sad / Angry).
        Each kind accepts only checkpoints trained for it."""
        if kind not in ("emothaw", "4class"):
            raise ValueError(f"unknown analyzer kind {kind!r}")
        self.kind = kind
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()          # load once; one forward pass at a time
        self._loader = None
        self._model = None
        self._ckpt = None
        self._state = "not_loaded"             # not_loaded | loading | ready | not_installed | error
        self._error = None

    @property
    def relative_path(self):
        return os.path.relpath(self.checkpoint_path, config.PROJECT_DIR)

    def _missing_message(self):
        if self.kind == "4class":
            return (f"The 4-class handwriting model is not trained/installed yet. Please place the trained "
                    f"checkpoint at: {self.relative_path} (build seed data with `python -m "
                    f"handwriting.generate_seed_data`, then `python -m handwriting.train --dataset seed_4class`; "
                    f"see data/README.md).")
        return (f"Handwriting model is not trained/installed yet. Please place the trained "
                f"checkpoint at: {self.relative_path} (train it with `python -m handwriting.train`, "
                f"see data/README.md).")

    def _check_heads(self, ckpt):
        """Refuse a checkpoint trained for the other experiment."""
        if self.kind == "4class":
            ok = [(h["name"], h["classes"]) for h in ckpt["heads"]] == [("emotion", config.FOUR_CLASS_LABELS)]
            if not ok:
                raise ValueError(f"it predicts {[h['classes'] for h in ckpt['heads']]}, not the 4-class "
                                 f"{config.FOUR_CLASS_LABELS}; train it with `python -m handwriting.train "
                                 f"--dataset seed_4class` (or real_4class)")
            return
        heads = sorted(h["name"] for h in ckpt["heads"])
        if heads != sorted(config.HEAD_NAMES) or any(h["classes"] != ["normal", "elevated"] for h in ckpt["heads"]):
            raise ValueError(f"it predicts {[h['classes'] for h in ckpt['heads']]}, not EMOTHAW's "
                             f"anxiety / depression / stress (normal / elevated); retrain with "
                             f"`python -m handwriting.train`")

    # ------------------------------------------------------------------ loading
    def preload_async(self):
        with self._lock:
            if self._model is not None or (self._loader is not None and self._loader.is_alive()):
                return False
            self._loader = threading.Thread(target=self._safe_load, name="HandwritingModelLoader",
                                            daemon=True)
            self._loader.start()
            return True

    def _safe_load(self):
        try:
            self.ensure_loaded()
        except HandwritingModelUnavailable:
            pass                               # state/error already recorded

    def ensure_loaded(self):
        """Load the checkpoint exactly once. Re-checks the file after a failure, so a
        checkpoint copied in while the app runs is picked up."""
        with self._lock:
            if self._model is not None:
                return
            if torch is None:
                self._state, self._error = "error", f"PyTorch is not available: {_TORCH_IMPORT_ERROR}"
                raise HandwritingModelUnavailable(self._error)
            if not os.path.isfile(self.checkpoint_path):
                self._state, self._error = "not_installed", self._missing_message()
                raise HandwritingModelUnavailable(self._error)
            self._state = "loading"
            started = time.time()
            try:
                from .model import build_model

                ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
                if not isinstance(ckpt, dict) or ckpt.get("format") != "handwriting-transformer/1":
                    raise ValueError("not a handwriting-transformer/1 checkpoint")
                self._check_heads(ckpt)
                model = build_model(len(ckpt["feature_names"]), ckpt["heads"], ckpt["hparams"])
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
            except Exception as exc:
                traceback.print_exc()
                self._state = "error"
                self._error = f"Could not load the handwriting model ({self.relative_path}): {exc}"
                _log(f"ERROR: {self._error}")
                raise HandwritingModelUnavailable(self._error) from None
            self._ckpt = ckpt
            self._mean = np.asarray(ckpt["norm"]["mean"], dtype=np.float64)
            self._std = np.asarray(ckpt["norm"]["std"], dtype=np.float64)
            self._max_len = int(ckpt["preprocessing"]["max_len"])
            self._model = model
            self._state, self._error = "ready", None
            _log(f"model ready in {time.time() - started:.2f}s: heads "
                 f"{[h['name'] for h in ckpt['heads']]}, features={ckpt['feature_set']}")

    # ------------------------------------------------------------------ status
    def _model_info(self):
        if self._ckpt is None:
            return None
        tr = self._ckpt.get("training", {})
        test = tr.get("test_metrics", {})
        return {
            "heads": [{k: h[k] for k in ("name", "display", "classes") if k in h} for h in self._ckpt["heads"]],
            "feature_set": self._ckpt["feature_set"],
            "dataset": tr.get("dataset"),
            "dataset_key": tr.get("dataset_key", "emothaw"),
            "dataset_type": tr.get("dataset_type"),
            "scientifically_validated": tr.get("scientifically_validated"),
            "prototype": tr.get("dataset_type") == "seed",
            "notice": tr.get("notice"),
            "tasks": tr.get("tasks"),
            "trained_at": tr.get("trained_at"),
            "writer_independent": tr.get("writer_independent", True),
            "split_counts": {k: {"writers": v["writers"], "samples": v["samples"]}
                             for k, v in tr.get("split_counts", {}).items()},
            **{f"test_{m}": {h["name"]: (test.get("per_recording", {}).get(h["name"]) or {}).get(m)
                             for h in self._ckpt["heads"]}
               for m in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1")},
        }

    def status(self):
        if self._model is None and self._state in ("not_loaded", "not_installed") \
                and not os.path.isfile(self.checkpoint_path):
            self._state, self._error = "not_installed", self._missing_message()
        return {"available": self._model is not None, "state": self._state, "message": self._error,
                "checkpoint": self.relative_path, "model": self._model_info()}

    # ------------------------------------------------------------------ inference
    def preprocess(self, payload, air="keep", resample_mode="time"):
        """Validate + preprocess. Needs no model, so input errors are reported even
        before a checkpoint exists. air / resample_mode: the checkpoint's settings."""
        if not isinstance(payload, dict):
            raise HandwritingInputError("expected a JSON object with a 'points' list")
        return prepare(from_browser_points(payload.get("points")), air, resample_mode)

    def analyze(self, payload):
        t0 = time.perf_counter()
        traj = self.preprocess(payload)          # validates input even without a model
        try:
            self.ensure_loaded()
        except HandwritingModelUnavailable as exc:
            # the input itself was fine: say what was received so the UI can show it
            from .preprocessing import compute_features

            exc.input_summary = dict(compute_features(traj)[1], points_received=len(payload["points"]))
            raise
        ckpt = self._ckpt
        air = ckpt["preprocessing"].get("air", "keep")
        resample_mode = ckpt["preprocessing"].get("resample", "time")
        if air != "keep" or resample_mode != "time":
            traj = self.preprocess(payload, air, resample_mode)
        x, info = feature_matrix(traj, ckpt["feature_names"], self._mean, self._std)
        probs, spans = self._forward(x)

        if self.kind == "4class":
            return self._four_class_result(ckpt, probs, info, spans, payload, t0)
        order = {n: i for i, n in enumerate(config.HEAD_NAMES)}
        heads, elevated, predictions, p_elevated = [], [], {}, {}
        for h in sorted(ckpt["heads"], key=lambda h: order[h["name"]]):
            p = probs[h["name"]]
            k = int(np.argmax(p))
            label = h["classes"][k]
            display = h.get("display", h["name"].title())
            heads.append({"name": h["name"], "display": display,
                          "classes": h["classes"], "label": label, "confidence": float(p[k]),
                          "probabilities": {c: float(v) for c, v in zip(h["classes"], p)}})
            predictions[display] = label.title()
            p_elevated[display] = round(float(p[h["classes"].index("elevated")]), 4)
            if label != h.get("baseline", "normal"):
                elevated.append(display)
        return {"ok": True, "predictions": predictions, "probabilities": p_elevated,
                "heads": heads, "elevated": elevated,
                "input": dict(info, windows=len(spans), points_received=len(payload["points"])),
                "model": self._model_info(),
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    def _forward(self, x):
        """(N, F) standardised features -> ({head: mean class probabilities over windows}, spans)."""
        from .model import pad_batch

        spans = windows(len(x), self._max_len)
        with self._lock, torch.inference_mode():
            xb, mask = pad_batch([x[a:b] for a, b in spans])
            logits = self._model(xb, mask)
            probs = {n: torch.softmax(l, -1).mean(0).numpy() for n, l in logits.items()}
        return probs, spans

    def analyze_image(self, data):
        """4-class prototype only: an uploaded handwriting image -> the same result as
        analyze(). The image goes through image_trajectory (geometry real; stroke order
        and pen lifts estimated; no timing, pressure or tilt) and then the checkpoint's
        own preprocessing. Refused for a checkpoint that needs timing features, which a
        static image cannot provide."""
        t0 = time.perf_counter()
        if self.kind != "4class":
            raise HandwritingInputError("Image upload is only available for the 4-class prototype.")
        from .image_trajectory import image_bytes_to_trajectory, sniff_upload_format

        if not data:
            raise HandwritingInputError("The uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HandwritingInputError(f"The image is too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
        fmt = sniff_upload_format(data)
        if fmt is None:
            raise HandwritingInputError("Unsupported file type - please upload a PNG, JPG or WEBP image.")
        dims = _image_dimensions(data)
        if dims is None:
            raise HandwritingInputError("The image could not be read (corrupt or truncated file).")
        if dims[0] * dims[1] > MAX_UPLOAD_PIXELS:
            raise HandwritingInputError(f"The image is too large ({dims[0]}x{dims[1]} pixels).")
        if min(dims) < 32:
            raise HandwritingInputError(f"The image is too small ({dims[0]}x{dims[1]} pixels).")
        # Same check as the graphology collector: the reconstruction assumes dark ink on
        # light, evenly lit paper. Photos, shadows or coloured backgrounds would be traced
        # as thousands of junk "strokes", so they are refused rather than analysed.
        import cv2
        from .collect_graphology import handwriting_likeness
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if float(np.ptp(gray)) < 40 or float(ink.mean()) < 0.002:        # blank / near-uniform image
            raise HandwritingInputError("No handwriting could be detected in this image. Use a clear image of "
                                        "dark writing on light paper, cropped to the handwriting.")
        looks_ok, measured = handwriting_likeness(data, gray)
        if not looks_ok:
            raise HandwritingInputError(
                "This doesn't look like dark handwriting on light paper (for example a photo with shadows, a "
                "coloured background or a picture that isn't handwriting). Use a scan or an evenly lit photo "
                "cropped to the writing.")
        try:
            raw, rinfo = image_bytes_to_trajectory(data)
        except ValueError:
            raise HandwritingInputError("No handwriting could be detected in this image. Use a clear image of "
                                        "dark writing on light paper, cropped to the handwriting.") from None
        self.ensure_loaded()
        ckpt = self._ckpt
        needs_timing = sorted(set(ckpt["feature_names"]) & TIMING_FEATURES)
        if needs_timing:
            raise HandwritingInputError(
                f"The installed 4-class model uses writing-speed features ({', '.join(needs_timing)}) that a "
                f"static image cannot provide - analyse canvas writing instead.")
        pre = ckpt["preprocessing"]
        try:
            traj = prepare(raw, pre.get("air", "keep"), pre.get("resample", "time"))
        except HandwritingInputError:
            raise HandwritingInputError("Too little handwriting was detected in this image to analyse. Use a "
                                        "larger or clearer image of a word or sentence.") from None
        x, info = feature_matrix(traj, ckpt["feature_names"], self._mean, self._std)
        probs, spans = self._forward(x)
        result = self._four_class_result(ckpt, probs, info, spans, {"points": raw.x}, t0)
        result["input"].update(
            source_type="uploaded_image", image_format=fmt, image_size=rinfo["image_size"],
            trajectory_source="reconstructed_from_image", timing_available=False, pressure_available=False,
            stroke_order_available="estimated", reconstructed=rinfo["reconstructed"],
            unavailable=rinfo["unavailable"], strokes=rinfo["strokes"], duration_s=None,
            points_received=rinfo["points"])
        return result

    def _four_class_result(self, ckpt, probs, info, spans, payload, t0):
        h = ckpt["heads"][0]
        p = probs[h["name"]]
        k = int(np.argmax(p))
        model = self._model_info()
        head = {"name": h["name"], "display": h.get("display", "Emotion"), "classes": h["classes"],
                "label": h["classes"][k], "confidence": float(p[k]),
                "probabilities": {c: float(v) for c, v in zip(h["classes"], p)}}
        return {"ok": True, "emotion": head["label"], "confidence": round(head["confidence"], 4),
                "probabilities": {c: round(v, 4) for c, v in head["probabilities"].items()},
                "prototype": model["prototype"], "notice": model["notice"],
                "heads": [head],
                "input": dict(info, windows=len(spans), points_received=len(payload["points"])),
                "model": model,
                "inference_ms": round((time.perf_counter() - t0) * 1000.0, 1)}

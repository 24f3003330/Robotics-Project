"""
body_vitals/inference.py - BodyVitalsMLEngine: physiological features -> WESAD condition.

    engine = BodyVitalsMLEngine()
    engine.predict(features)            # dict keyed by config.FEATURE_NAMES
    -> {"predicted_state": "stress", "confidence": 0.87,
        "probabilities": {"baseline": 0.05, "stress": 0.87, "amusement": 0.08}}

The trained model makes the prediction - there are no hand-written thresholds
here. Before the first prediction the engine checks that the saved model, its
metadata and this code agree on the dataset, the classes, the feature list AND
its order, and the feature version; any mismatch refuses to load.
"""

import os
import threading
import time

import numpy as np

from . import config
from .features import extract_window_features
from .normalization import reference_from_windows


class BodyVitalsModelUnavailable(RuntimeError):
    """No trained model, or a model that does not match this code."""


class BodyVitalsInputError(ValueError):
    """The caller sent features that cannot be used (missing, extra, non-numeric)."""


class BodyVitalsCalibrationRequired(BodyVitalsInputError):
    """A rest-calibrated model was called without a resting reference."""


NORMALIZATIONS = ("none", "rest_calibration")


class BodyVitalsMLEngine:
    def __init__(self, model_path=config.MODEL_PATH, metadata_path=config.METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self._lock = threading.Lock()
        self._model = None
        self._metadata = None
        self._error = None

    # ------------------------------------------------------------------ loading
    def load(self):
        with self._lock:
            if self._model is not None:
                return self._model, self._metadata
            if not (os.path.isfile(self.model_path) and os.path.isfile(self.metadata_path)):
                self._error = ("No trained Body Vitals model. Run `python -m body_vitals.prepare_wesad "
                               "--data /path/to/WESAD` and then `python -m body_vitals.train`.")
                raise BodyVitalsModelUnavailable(self._error)
            try:
                from .model import load_artifacts
                model, metadata = load_artifacts(self.model_path, self.metadata_path)
                self._verify(model, metadata)
            except BodyVitalsModelUnavailable as exc:
                self._error = str(exc)
                raise
            except Exception as exc:
                self._error = f"Cannot load the Body Vitals model: {type(exc).__name__}: {exc}"
                raise BodyVitalsModelUnavailable(self._error) from exc
            self._model, self._metadata, self._error = model, metadata, None
            return model, metadata

    @staticmethod
    def _verify(model, metadata):
        problems = []
        if metadata.get("dataset") != "WESAD":
            problems.append(f"dataset is {metadata.get('dataset')!r}, expected 'WESAD'")
        if metadata.get("classes") != config.CLASSES:
            problems.append(f"classes {metadata.get('classes')} != {config.CLASSES}")
        version = metadata.get("feature_version")
        names = metadata.get("feature_names")
        if version not in config.FEATURE_SETS:
            problems.append(f"unknown feature version {version!r} (known: {list(config.FEATURE_SETS)})")
        elif names != config.FEATURE_SETS[version]:
            problems.append(f"feature names / order in the metadata differ from feature set {version}")
        trained_on = getattr(model, "feature_names_in_", None)
        if trained_on is not None and list(trained_on) != names:
            problems.append("the model was fitted with a different feature order")
        if names is not None and getattr(model, "n_features_in_", len(names)) != len(names):
            problems.append(f"model expects {model.n_features_in_} features, not {len(names)}")
        model_classes = [int(c) for c in getattr(model, "classes_", range(len(config.CLASSES)))]
        if model_classes != list(range(len(config.CLASSES))):
            problems.append(f"model classes {model_classes} are not 0..{len(config.CLASSES) - 1}")
        if metadata.get("normalization", "none") not in NORMALIZATIONS:
            problems.append(f"unknown normalization {metadata.get('normalization')!r}")
        if not hasattr(model, "predict_proba"):
            problems.append("model cannot output class probabilities")
        if problems:
            raise BodyVitalsModelUnavailable("Body Vitals model does not match this code: " + "; ".join(problems)
                                             + ". Retrain with `python -m body_vitals.train`.")

    def preload_async(self):
        if self._model is None:
            threading.Thread(target=self._quiet_load, daemon=True).start()

    def _quiet_load(self):
        try:
            self.load()
        except BodyVitalsModelUnavailable:
            pass

    def status(self):
        ready = self._model is not None
        if not ready and self._error is None:
            try:
                self.load()
                ready = True
            except BodyVitalsModelUnavailable:
                pass
        out = {"ready": ready, "error": self._error}
        if ready:
            m = self._metadata
            test = m.get("evaluation_metrics", {}).get("test", {})
            out["model"] = {"dataset": m["dataset"], "classes": m["classes"], "model_type": m["model_type"],
                            "feature_version": m["feature_version"], "n_features": len(m["feature_names"]),
                            "window_size_sec": m["window_size_sec"], "trained_at": m.get("trained_at"),
                            "normalization": m.get("normalization", "none"),
                            "calibration_sec": m.get("calibration_sec"),
                            "test_subjects": m.get("test_subjects"),
                            "test_accuracy": test.get("accuracy"),
                            "test_balanced_accuracy": test.get("balanced_accuracy"),
                            "test_macro_f1": test.get("macro_f1"),
                            "test_recall": {c: v.get("recall") for c, v in test.get("per_class", {}).items()}}
        return out

    # --------------------------------------------------------------- prediction
    @property
    def feature_names(self):
        return self.load()[1]["feature_names"]

    @property
    def normalization(self):
        return self.load()[1].get("normalization", "none")

    @property
    def window_sec(self):
        return float(self.load()[1]["window_size_sec"])

    @staticmethod
    def feature_vector(features, names=None):
        """dict -> 1 x n DataFrame in the model's training order. Rejects missing, unknown and
        non-finite values; known features the model does not use are ignored."""
        import pandas as pd
        names = list(names or config.FEATURE_NAMES)
        if not isinstance(features, dict):
            raise BodyVitalsInputError("features must be an object keyed by feature name")
        missing = [f for f in names if f not in features]
        unknown = [k for k in features if k not in config.ALL_FEATURE_NAMES]
        if missing:
            raise BodyVitalsInputError(f"missing features: {', '.join(missing)}")
        if unknown:
            raise BodyVitalsInputError(f"unknown features: {', '.join(unknown)}")
        try:
            row = [float(features[f]) for f in names]
        except (TypeError, ValueError):
            raise BodyVitalsInputError("every feature must be a number")
        bad = [f for f, v in zip(names, row) if not np.isfinite(v)]
        if bad:
            raise BodyVitalsInputError(f"non-finite features: {', '.join(bad)}")
        return pd.DataFrame([row], columns=names)

    def make_reference(self, calibration_features):
        """Resting reference from the features of calibration windows (list of dicts):
        per-feature median, exactly as in training (normalization.reference_from_windows)."""
        rows = [self.feature_vector(f, self.feature_names).iloc[0] for f in calibration_features]
        if not rows:
            raise BodyVitalsCalibrationRequired("no calibration windows")
        return reference_from_windows(rows, self.feature_names)

    def predict(self, features, reference=None):
        """features: dict of absolute feature values. reference: resting reference
        (dict) - required when the model was trained with rest calibration."""
        model, _ = self.load()
        names = self.feature_names
        X = self.feature_vector(features, names)
        if self.normalization == "rest_calibration":
            if reference is None:
                raise BodyVitalsCalibrationRequired(
                    "This model compares each window with the person's own resting reference: provide "
                    "calibration (a few minutes of seated rest) with the request.")
            ref = self.feature_vector(reference, names)
            X = X - ref.to_numpy()
        proba = np.asarray(model.predict_proba(X))[0]
        idx = int(np.argmax(proba))
        return {"predicted_state": config.CLASSES[idx],
                "confidence": float(proba[idx]),
                "probabilities": {c: float(proba[i]) for i, c in enumerate(config.CLASSES)}}

    def window_features(self, window):
        feats, quality = extract_window_features(window.ecg, window.eda, window.resp, window.fs_ecg,
                                                 window.fs_eda, window.fs_resp)
        return feats, quality

    def predict_window(self, window, calibration_windows=None):
        """Raw SignalWindow (+ calibration SignalWindows for a calibrated model) ->
        (features used by the model, signal quality, prediction). Raises SignalQualityError."""
        feats, quality = self.window_features(window)
        reference = None
        if self.normalization == "rest_calibration":
            if not calibration_windows:
                raise BodyVitalsCalibrationRequired("This model needs calibration windows of seated rest.")
            reference = self.make_reference([self.window_features(w)[0] for w in calibration_windows])
            quality = dict(quality, calibration_windows=len(calibration_windows))
        used = {f: feats[f] for f in self.feature_names}
        return used, quality, self.predict(feats, reference)

    def describe(self):
        _, m = self.load()
        return {"dataset": m["dataset"], "model_type": m["model_type"], "classes": m["classes"],
                "feature_version": m["feature_version"], "normalization": m.get("normalization", "none")}


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, round((time.perf_counter() - t0) * 1000, 1)

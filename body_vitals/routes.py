"""
body_vitals/routes.py - Flask blueprint for the independent Body Vitals page.

    GET  /body-vitals          page
    GET  /body-vitals/status   is a trained WESAD model loaded? (+ its test metrics)
    POST /body-vitals/demo     {"scenario": "baseline"|"stress"|"amusement", "seed"?: int}
                               SIMULATED waveforms -> features -> WESAD Random Forest
    POST /body-vitals/predict  {"features": {...}}  or
                               {"signals": {"ecg": [...], "eda": [...], "resp": [...],
                                            "fs_ecg": 700, "fs_eda": 700, "fs_resp": 700}}
                               A rest-calibrated model (metadata normalization =
                               "rest_calibration") also needs the person's resting reference:
                               "calibration_features": [{...}, ...]  (features of rest windows) or
                               "calibration_signals":  [{signals}, ...] (raw rest windows)

Independent of camera, voice and handwriting: no webcam, no microphone, no
fusion, no music or robot. The trained WESAD model is the only prediction
mechanism. Every response says where its input came from (input.source /
input.is_simulated) so simulated values are never shown as sensor measurements.

Success response:
    {"success": true, "predicted_state": "Stress", "confidence": 0.87,
     "probabilities": {"Baseline": 0.05, "Stress": 0.87, "Amusement": 0.08},
     "model": {...}, "input": {...}, "inference_ms": 12.3}
"""

import traceback

from flask import Blueprint, jsonify, render_template, request

from . import config
from .inference import (BodyVitalsCalibrationRequired, BodyVitalsInputError, BodyVitalsMLEngine,
                        BodyVitalsModelUnavailable, timed)
from .preprocessing import SignalQualityError
from .providers import DEMO_SCENARIOS, ArraySignalsProvider, DemoVitalsProvider

MAX_REQUEST_BYTES = 30 * 1024 * 1024
MAX_SIGNAL_SECONDS = 600

bp = Blueprint("body_vitals", __name__)
engine = BodyVitalsMLEngine()


def _vitals(features):
    """The headline numbers shown on the page, measured from the window."""
    return {"heart_rate_bpm": round(features["hr_mean"], 1), "hrv_sdnn_ms": round(features["hrv_sdnn"], 1),
            "hrv_rmssd_ms": round(features["hrv_rmssd"], 1), "gsr_us": round(features["eda_mean"], 2),
            "respiration_rate_bpm": round(features["resp_rate_mean"], 1)}


def _calibration_window_count(meta):
    """How many resting windows make up the calibration period - the same count the
    training reference used (windows fully inside the first calibration_sec seconds)."""
    w, step, cal = meta["window_size_sec"], meta["window_step_sec"], meta.get("calibration_sec") or 0
    return max(1, int((cal - w) // step) + 1)


def _response(features, prediction, input_info, quality=None, ms=None, calibration=None):
    _, meta = engine.load()
    display = config.CLASS_DISPLAY
    return {
        "success": True,
        "predicted_state": display[prediction["predicted_state"]],
        "confidence": prediction["confidence"],
        "probabilities": {display[c]: p for c, p in prediction["probabilities"].items()},
        "model": {"type": meta["model_type"], "dataset": meta["dataset"],
                  "classes": [display[c] for c in meta["classes"]],
                  "feature_version": meta["feature_version"], "window_sec": meta["window_size_sec"],
                  "normalization": meta.get("normalization", "none"),
                  "note": "WESAD study condition predicted by a model trained on WESAD physiological "
                          "features - not a diagnosis and not a self-reported emotion."},
        "calibration": calibration or {"used": False},
        "input": dict(input_info, vitals=_vitals(features),
                      features={k: round(v, 5) for k, v in features.items()}, signal_quality=quality),
        "inference_ms": ms,
    }


def _model_unavailable(exc):
    return jsonify({"success": False, "kind": "model_unavailable", "error": str(exc), "status": engine.status()}), 503


@bp.route("/body-vitals")
def page():
    engine.preload_async()
    return render_template("body_vitals.html", active_page="body_vitals", scenarios=list(DEMO_SCENARIOS),
                           demo_notice=config.DEMO_NOTICE, class_display=config.CLASS_DISPLAY)


@bp.route("/body-vitals/status")
def status():
    return jsonify(engine.status())


@bp.route("/body-vitals/demo", methods=["POST"])
def demo():
    payload = request.get_json(silent=True) or {}
    scenario = payload.get("scenario", "baseline")
    seed = payload.get("seed")
    if scenario not in DEMO_SCENARIOS:
        return jsonify({"success": False, "kind": "input",
                        "error": f"scenario must be one of {list(DEMO_SCENARIOS)}"}), 400
    if seed is not None and not isinstance(seed, int):
        return jsonify({"success": False, "kind": "input", "error": "seed must be an integer"}), 400
    try:
        _, meta = engine.load()
        provider = DemoVitalsProvider(scenario, seed=seed, seconds=engine.window_sec)
        window = provider.read_window()
        calibration, calib_windows = None, None
        if engine.normalization == "rest_calibration":
            calib_windows = provider.read_calibration(_calibration_window_count(meta))
            calibration = {"used": True, "source": "demo", "is_simulated": True, "windows": len(calib_windows),
                           "seconds": meta["calibration_sec"],
                           "note": "Simulated resting windows of the same simulated person, used as the "
                                   "personal resting reference."}
        (features, quality, prediction), ms = timed(engine.predict_window, window, calib_windows)
    except BodyVitalsModelUnavailable as exc:
        return _model_unavailable(exc)
    except SignalQualityError as exc:      # a synthetic window should never fail; report it if it does
        return jsonify({"success": False, "kind": "signal_quality", "reason": exc.reason, "error": str(exc)}), 422
    info = {"source": "demo", "is_simulated": True, "label": "SIMULATED / DEMO VALUES",
            "notice": config.DEMO_NOTICE, "scenario_requested": scenario,
            "window_sec": round(window.seconds, 1)}
    out = _response(features, prediction, info, quality, ms, calibration)
    print(f"[BODY VITALS demo] scenario={scenario} -> {prediction['predicted_state']} "
          f"{prediction['confidence']:.0%} ({ms} ms)", flush=True)
    return jsonify(out)


@bp.route("/body-vitals/predict", methods=["POST"])
def predict():
    if (request.content_length or 0) > MAX_REQUEST_BYTES:
        return jsonify({"success": False, "kind": "input", "error": "Request too large."}), 413
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not ({"features", "signals"} & set(payload)):
        return jsonify({"success": False, "kind": "input",
                        "error": "Expected JSON with either 'features' or 'signals'."}), 400
    info = {"source": "api", "is_simulated": None, "label": "VALUES SUPPLIED BY THE CALLER",
            "notice": "These values were sent to the API; their origin (sensor or simulation) is not verified."}
    try:
        engine.load()
        calibrated = engine.normalization == "rest_calibration"
        if "features" in payload:
            reference, calibration = None, None
            if calibrated:
                rows = payload.get("calibration_features")
                if not isinstance(rows, list) or not rows:
                    raise BodyVitalsCalibrationRequired("this model needs 'calibration_features': a list of "
                                                        "feature objects from seated-rest windows")
                reference = engine.make_reference(rows)
                calibration = {"used": True, "source": "api", "is_simulated": None, "windows": len(rows)}
            prediction, ms = timed(engine.predict, payload["features"], reference)
            features = {f: float(payload["features"][f]) for f in engine.feature_names}
            return jsonify(_response(features, prediction, dict(info, input_type="features"), None, ms,
                                     calibration))
        window = _signals_window(payload["signals"])
        calib_windows, calibration = None, None
        if calibrated:
            raw = payload.get("calibration_signals")
            if not isinstance(raw, list) or not raw:
                raise BodyVitalsCalibrationRequired("this model needs 'calibration_signals': a list of raw "
                                                    "seated-rest windows in the same format as 'signals'")
            calib_windows = [_signals_window(r) for r in raw]
            calibration = {"used": True, "source": "api", "is_simulated": None, "windows": len(calib_windows)}
        (features, quality, prediction), ms = timed(engine.predict_window, window, calib_windows)
        return jsonify(_response(features, prediction,
                                 dict(info, input_type="signals", window_sec=round(window.seconds, 1)), quality, ms,
                                 calibration))
    except BodyVitalsModelUnavailable as exc:
        return _model_unavailable(exc)
    except BodyVitalsCalibrationRequired as exc:
        return jsonify({"success": False, "kind": "calibration_required", "error": str(exc)}), 400
    except (BodyVitalsInputError, ValueError, TypeError) as exc:
        if isinstance(exc, SignalQualityError):
            return jsonify({"success": False, "kind": "signal_quality", "reason": exc.reason, "error": str(exc)}), 422
        return jsonify({"success": False, "kind": "input", "error": str(exc)}), 400
    except Exception as exc:
        print(f"[BODY VITALS ERROR] {exc}", flush=True)
        traceback.print_exc()
        return jsonify({"success": False, "kind": "error", "error": f"Body Vitals prediction failed: {exc}"}), 500


def _signals_window(sig):
    if not isinstance(sig, dict) or not all(isinstance(sig.get(k), list) for k in ("ecg", "eda", "resp")):
        raise BodyVitalsInputError("signals needs 'ecg', 'eda' and 'resp' lists")
    fs_ecg = sig.get("fs_ecg", sig.get("fs"))
    if not isinstance(fs_ecg, (int, float)) or fs_ecg <= 0:
        raise BodyVitalsInputError("signals needs a positive sampling rate 'fs_ecg' (or 'fs')")
    fs_eda, fs_resp = sig.get("fs_eda", fs_ecg), sig.get("fs_resp", fs_ecg)
    for fs in (fs_eda, fs_resp):
        if not isinstance(fs, (int, float)) or fs <= 0:
            raise BodyVitalsInputError("sampling rates must be positive numbers")
    if len(sig["ecg"]) / fs_ecg > MAX_SIGNAL_SECONDS:
        raise BodyVitalsInputError(f"at most {MAX_SIGNAL_SECONDS} s of signal per request")
    return ArraySignalsProvider(sig["ecg"], sig["eda"], sig["resp"], fs_ecg, fs_eda, fs_resp).read_window()

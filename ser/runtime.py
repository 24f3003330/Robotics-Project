"""The ensemble as the app sees it: same interface as emotion_detector._EmotionModel.

    model.predict(audio_float32_16k) -> {"happy": p, "angry": p, "sad": p, "neutral": p}

That signature is the whole integration. The live worker and the uploaded-file
job both already call exactly this, so swapping a single-model object for an
ensemble object upgrades both paths without touching capture, VAD, smoothing,
fusion or the UI.

predict_detailed() additionally returns each member's own distribution, which
is what makes a disagreement debuggable: "ensemble says sad, but cremad says
angry 0.71 and msp_podcast says neutral 0.55" is a diagnosis; "sad 0.42" is not.

Cost: N members means N forward passes per window. On CPU that is the one real
risk to live latency, so members run in a fixed order, the result is cached per
window, and SER_ENSEMBLE_MEMBERS can restrict the ensemble to a subset without
retraining anything.
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import torch

from .calibrate import apply_calibration
from .ensemble import combiner_from_dict
from .labels import CANONICAL_EMOTIONS

DEFAULT_MANIFEST = os.environ.get(
    "VOICE_ENSEMBLE_MANIFEST",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "ensemble", "manifest.json"))


class EnsembleUnavailable(RuntimeError):
    """No usable ensemble on disk - the caller should fall back to the single model."""


class _Member:
    __slots__ = ("name", "dir", "calibration", "weight", "model", "extractor", "device")

    def __init__(self, name, directory, calibration, weight, device):
        self.name = name
        self.dir = directory
        self.calibration = calibration
        self.weight = weight
        self.device = device
        self.model = None
        self.extractor = None

    def load(self):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model = AutoModelForAudioClassification.from_pretrained(self.dir)
        order = [str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)]
        if order != list(CANONICAL_EMOTIONS):
            raise EnsembleUnavailable(
                f"member '{self.name}' in {self.dir} emits classes {order}, but the app "
                f"indexes {list(CANONICAL_EMOTIONS)}. Retrain it with ser.train (which writes "
                f"the canonical order) rather than reordering at runtime.")
        self.model = model.to(self.device).eval()
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(self.dir)
        except Exception:
            self.extractor = AutoFeatureExtractor.from_pretrained(
                os.environ.get("SER_BASE_MODEL", "superb/hubert-base-superb-er"))
        return self

    @torch.inference_mode()
    def logits(self, audio, sample_rate):
        inputs = self.extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).logits[0].float().cpu().numpy()

    def probabilities(self, audio, sample_rate):
        """Calibrated probabilities - never a raw softmax (see ser/calibrate.py)."""
        raw = self.logits(audio, sample_rate)
        return apply_calibration(raw[None, :], self.calibration)[0]


class EnsembleEmotionModel:
    """Drop-in replacement for emotion_detector._EmotionModel."""

    def __init__(self, manifest_path=DEFAULT_MANIFEST, device=None, only_members=None):
        if not os.path.isfile(manifest_path):
            raise EnsembleUnavailable(f"no ensemble manifest at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        classes = manifest.get("classes") or list(CANONICAL_EMOTIONS)
        if list(classes) != list(CANONICAL_EMOTIONS):
            raise EnsembleUnavailable(
                f"manifest class order {classes} != app order {list(CANONICAL_EMOTIONS)}")

        self.manifest_path = manifest_path
        self.manifest = manifest
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = int(manifest.get("sample_rate", 16000))
        self.window_spec = manifest.get("window_spec", {})
        # Serialises inference across threads. The live worker and the
        # uploaded-file job can both call predict() at once, and letting them
        # run concurrently is WORSE for live latency, not better: each would
        # spawn its own torch intra-op threads on a box that is already running
        # Flask, the camera loop and BLE. Serialising costs the live path at
        # most one upload window (~N x member latency, printed by
        # `python -m ser check`) rather than an oversubscribed CPU.
        self._lock = threading.Lock()

        base_dir = os.path.dirname(os.path.abspath(manifest_path))
        wanted = only_members or _env_member_filter()
        members, skipped = [], []
        for spec in manifest.get("members", []):
            name = spec["name"]
            if spec.get("enabled") is False:
                skipped.append((name, "disabled in the manifest"))
                continue
            if wanted and name not in wanted:
                skipped.append((name, "excluded by SER_ENSEMBLE_MEMBERS"))
                continue
            directory = spec["dir"]
            if not os.path.isabs(directory):
                directory = os.path.normpath(os.path.join(base_dir, directory))
            if not os.path.isdir(directory):
                skipped.append((name, f"model directory missing: {directory}"))
                continue
            members.append(_Member(name, directory, spec.get("calibration"),
                                   float(spec.get("weight", 0.0)), self.device))
        if not members:
            raise EnsembleUnavailable(
                f"{manifest_path} lists no loadable member "
                f"({'; '.join(f'{n}: {why}' for n, why in skipped) or 'no members at all'})")

        for member in members:
            member.load()
        self.members = members
        self.member_names = [m.name for m in members]
        self.skipped = skipped

        combiner_spec = dict(manifest.get("combiner") or {"kind": "average",
                                                          "members": self.member_names})
        self.combiner, self.combiner_note = _fit_combiner_to_available(
            combiner_spec, self.member_names)

        self.name = f"ensemble[{self.combiner.kind}]({'+'.join(self.member_names)})"
        self.model_name = self.name
        self.labels = {i: e for i, e in enumerate(CANONICAL_EMOTIONS)}
        self.index_to_emotion = dict(self.labels)
        self.last_detail = None
        self.predict(np.zeros(self.sample_rate, dtype=np.float32))  # warm every member up

    # ------------------------------------------------------------------ inference
    def predict(self, audio):
        """The interface emotion_detector depends on: probabilities per emotion."""
        return self.predict_detailed(audio)["ensemble"]

    def predict_detailed(self, audio):
        """Ensemble result PLUS every member's own distribution and timing.

        {"ensemble": {emotion: p}, "members": {name: {emotion: p}},
         "combiner": "weighted", "weights": {...}, "agreement": 0.6,
         "disagreement": [...], "ms": 84.2}
        """
        audio = np.asarray(audio, dtype=np.float32)
        started = time.perf_counter()
        with self._lock:
            member_probs, member_ms = {}, {}
            for member in self.members:
                t0 = time.perf_counter()
                member_probs[member.name] = member.probabilities(audio, self.sample_rate)
                member_ms[member.name] = round((time.perf_counter() - t0) * 1000.0, 1)

            stacked = {name: probs[None, :] for name, probs in member_probs.items()}
            ensemble = self.combiner.combine(stacked)[0]

        members_out = {name: _as_emotion_dict(p) for name, p in member_probs.items()}
        top = {name: max(d, key=d.get) for name, d in members_out.items()}
        winner = CANONICAL_EMOTIONS[int(np.argmax(ensemble))]
        agreeing = [n for n, t in top.items() if t == winner]
        detail = {
            "ensemble": _as_emotion_dict(ensemble),
            "members": members_out,
            "member_top": top,
            "member_ms": member_ms,
            "combiner": self.combiner.kind,
            "weights": self.combiner.weight_report,
            "emotion": winner,
            "agreement": round(len(agreeing) / float(len(top)), 3) if top else 0.0,
            "dissenting": sorted(n for n in top if n not in agreeing),
            "ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
        self.last_detail = detail
        return detail

    # ------------------------------------------------------------------ introspection
    def describe(self):
        return {
            "kind": "ensemble",
            "manifest": self.manifest_path,
            "combiner": self.combiner.kind,
            "combiner_note": self.combiner_note,
            "members": self.member_names,
            "weights": self.combiner.weight_report,
            "calibration": {m.name: (m.calibration or {}).get("kind", "none") for m in self.members},
            "skipped": [{"member": n, "reason": r} for n, r in self.skipped],
            "window_spec": self.window_spec,
            "device": str(self.device),
            "trained_on": self.manifest.get("trained_on", {}),
            "test_macro_f1": (self.manifest.get("metrics", {})
                              .get("ensemble", {}).get("macro_f1")),
        }


def _as_emotion_dict(vector):
    """Full precision, deliberately unrounded.

    The old single-model predict() returned exactly-normalised probabilities and
    callers rely on that: emotion_detector averages these vectors and compares a
    dominant probability against an absolute threshold. Rounding here put the sum
    a couple of ULPs off 1.0, which is harmless for the threshold but makes
    "probabilities that sum to 1" untrue for anything that checks. Rounding for
    display happens where the display happens.
    """
    vector = np.asarray(vector, dtype=np.float64)
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0:
        return {e: 1.0 / len(CANONICAL_EMOTIONS) for e in CANONICAL_EMOTIONS}
    return {e: float(vector[i] / total) for i, e in enumerate(CANONICAL_EMOTIONS)}


def _env_member_filter():
    raw = os.environ.get("SER_ENSEMBLE_MEMBERS", "").strip()
    return {p.strip() for p in raw.split(",") if p.strip()} if raw else None


def _fit_combiner_to_available(spec, available):
    """Keep the trained combiner if all its members loaded; otherwise degrade honestly.

    A weighted average or a stacker is only valid over the exact member set it
    was fitted on. If one member's directory is missing, silently reusing the
    remaining coefficients would apply weights fitted for a different problem,
    so the ensemble falls back to an unweighted average of whatever did load
    and says so in the status.
    """
    trained_members = list(spec.get("members") or available)
    if list(trained_members) == list(available):
        return combiner_from_dict(spec), ""
    note = (f"combiner '{spec.get('kind')}' was fitted on {trained_members} but only "
            f"{available} loaded - falling back to an unweighted average")
    print(f"[ENSEMBLE] {note}")
    return combiner_from_dict({"kind": "average", "members": list(available)}), note


def load_ensemble(manifest_path=DEFAULT_MANIFEST, device=None):
    """Convenience loader used by emotion_detector.build_emotion_model."""
    return EnsembleEmotionModel(manifest_path, device=device)

"""
Tests for voice AGE-GROUP estimation (voice_age/):  python -m unittest discover tests

All audio here is SYNTHETIC (generated tones / noise) and the encoder is a tiny
deterministic stand-in, so the tests exercise the pipeline plumbing - schema
checks, label mapping, speaker-wise splitting, preprocessing, model saving /
loading, inference, the live tracker and the API - without network access or the
real model. Nothing here is a result or training data.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from voice_age import config  # noqa: E402
from voice_age.audio import AudioError, decode, prepare, prepare_bytes, trim_silence  # noqa: E402
from voice_age.dataset import SchemaError, check_schema  # noqa: E402
from voice_age.inference import (AgeEstimator, AgeModelUnavailable, LiveAgeTracker,  # noqa: E402
                                 result_from_probabilities, save_model)
from voice_age.model import LogRegHead, MLPHead, majority_baseline, metrics  # noqa: E402
from voice_age.selection import (SpeakerLeakageError, check_groups, label_speakers, map_age,  # noqa: E402
                                 select_subset, split_report, split_speakers, verify_disjoint)

SR = 16000
CLASSES = ["Teens", "Twenties", "Thirties"]


def synthetic_voice(seconds=3.0, f0=150.0, sr=SR, seed=0, amp=0.3):
    """SYNTHETIC 'voice': a harmonic tone with a little noise (test fixture only)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5))
    return (amp * x / np.max(np.abs(x)) + rng.normal(0, 0.01, len(t))).astype(np.float32)


def wav_bytes(x, sr=SR, channels=1):
    import soundfile as sf
    buf = io.BytesIO()
    data = np.stack([x] * channels, axis=1) if channels > 1 else x
    sf.write(buf, data, sr, format="WAV")
    return buf.getvalue()


class FakeEncoder:
    """Stand-in for the frozen speech encoder: 3 'layers' of simple spectral statistics."""

    def __init__(self, *_):
        self.calls = 0

    def embed(self, clips):
        self.calls += 1
        out = []
        for c in clips:
            spec = np.abs(np.fft.rfft(c[:SR]))[:400]
            centroid = float((spec * np.arange(len(spec))).sum() / (spec.sum() + 1e-9))
            feats = np.array([centroid / 100.0, np.log(np.mean(c ** 2) + 1e-9), len(c) / SR, 1.0])
            out.append(np.stack([feats, feats * 2, feats[::-1]]))
        return np.asarray(out, dtype=np.float32)


def make_model_dir(folder, kind="logreg"):
    """Tiny age model whose classes follow the synthetic f0 (Teens high, Thirties low)."""
    enc = FakeEncoder()
    X, y = [], []
    for label, f0 in enumerate((260.0, 180.0, 110.0)):
        for s in range(12):
            X.append(enc.embed([synthetic_voice(2.5, f0 + s, seed=s)])[0, 1])
            y.append(label)
    X, y = np.array(X), np.array(y)
    head = LogRegHead(C=1.0).fit(X, y, n_classes=3) if kind == "logreg" else \
        MLPHead(hidden=16, epochs=30, patience=10).fit(X, y, X, y, 3)
    meta = {"classes": CLASSES, "model_version": "test", "trained_at": "now",
            "dataset": {"repo": "synthetic", "version": "SYNTHETIC TEST FIXTURE", "language": "none"},
            "architecture": {"encoder": "fake", "layer": 1, "head": kind},
            "test_metrics": {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}}
    save_model(folder, head.state(), meta)
    est = AgeEstimator(folder)
    est.load(encoder_factory=lambda arch: FakeEncoder())
    return est


# ---------------------------------------------------------------------------
class SchemaTests(unittest.TestCase):
    def parquet(self, table):
        import pyarrow.parquet as pq
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        return pq.ParquetFile(buf)

    def test_valid_schema(self):
        import pyarrow as pa
        audio = pa.array([{"bytes": b"x", "path": "a.mp3"}], pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
        t = pa.table({"client_id": ["c"], "audio": audio, "age": ["twenties"], "gender": [""]})
        info = check_schema(self.parquet(t))
        self.assertIn("age", info["columns"])

    def test_missing_age_column(self):
        import pyarrow as pa
        audio = pa.array([{"bytes": b"x", "path": "a"}], pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
        with self.assertRaises(SchemaError):
            check_schema(self.parquet(pa.table({"client_id": ["c"], "audio": audio})))

    def test_audio_must_be_encoded_struct(self):
        import pyarrow as pa
        t = pa.table({"client_id": ["c"], "audio": ["not audio"], "age": ["teens"]})
        with self.assertRaises(SchemaError):
            check_schema(self.parquet(t))


class LabelMappingTests(unittest.TestCase):
    def test_common_voice_labels(self):
        self.assertEqual(map_age("fourties"), "Forties")          # Common Voice's spelling
        self.assertEqual(map_age("forties"), "Forties")
        self.assertEqual(map_age("Twenties"), "Twenties")
        self.assertEqual(map_age("eighties"), "Seventies+")       # sparse decades merged
        self.assertEqual(map_age("nineties"), "Seventies+")
        self.assertIsNone(map_age(""))
        self.assertIsNone(map_age("unknown"))
        self.assertEqual(sorted(set(config.AGE_GROUP_MAPPING.values()), key=config.AGE_GROUPS.index),
                         config.AGE_GROUPS)

    def test_one_label_per_speaker(self):
        import pandas as pd
        df = pd.DataFrame({"client_id": ["a", "a", "a", "b", "b", "c"],
                           "age": ["teens", "teens", "twenties", "thirties", "twenties", "sixties"],
                           "shard": "s", "row_group": 0, "row": range(6), "gender": "", "path": "p"})
        d, rep = label_speakers(df)
        self.assertEqual(sorted(d["client_id"].unique()), ["a", "c"])      # b is a tie -> dropped
        self.assertTrue((d[d["client_id"] == "a"]["group"] == "Teens").all())
        self.assertEqual(rep["clips_dropped_minority_label"], 1)
        self.assertEqual(rep["speakers_dropped_tie"], 1)

    def test_sparse_groups_detected(self):
        import pandas as pd
        d = pd.DataFrame({"client_id": [f"s{i}" for i in range(10)], "group": ["Teens"] * 9 + ["Sixties"]})
        self.assertEqual(check_groups(d, ["Teens", "Sixties"], min_speakers=5), ["Sixties"])


def speaker_table(n_per_group=40, clips=3):
    import pandas as pd
    rows = []
    for g in CLASSES:
        for s in range(n_per_group):
            for c in range(clips):
                rows.append({"client_id": f"{g}-{s}", "group": g, "age": g.lower(), "shard": f"sh{s % 4}",
                             "row_group": (s + c) % 7, "row": c, "path": f"{g}-{s}-{c}.mp3", "gender": ""})
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    def test_speaker_split_disjoint_stratified_reproducible(self):
        d = speaker_table()
        a = split_speakers(d, seed=1)
        self.assertEqual(a, split_speakers(d, seed=1))
        self.assertEqual(set(a), set(d["client_id"]))
        sets = {s: [k for k, v in a.items() if v == s] for s in ("train", "validation", "test")}
        audit = verify_disjoint(sets)
        self.assertTrue(all(v == 0 for v in audit.values()))
        for g in CLASSES:                                   # every group present in every split
            for s in sets:
                self.assertTrue(any(k.startswith(g) for k in sets[s]), (g, s))
        self.assertAlmostEqual(len(sets["train"]) / len(a), 0.70, delta=0.05)

    def test_leakage_fails_loudly(self):
        with self.assertRaises(SpeakerLeakageError):
            verify_disjoint({"train": ["a", "b"], "validation": ["c"], "test": ["b"]})

    def test_subset_quotas_and_speaker_cap(self):
        d = speaker_table(n_per_group=40, clips=5)
        a = split_speakers(d, seed=0)
        sel, info = select_subset(d, a, max_per_class=60, max_per_speaker=2, seed=0)
        self.assertLessEqual(sel.groupby("client_id").size().max(), 2)
        for g in CLASSES:
            self.assertLessEqual((sel["group"] == g).sum(), 60 + 1)
        verify_disjoint({s: sel.loc[sel["split"] == s, "client_id"].unique() for s in ("train", "validation", "test")})
        rep = split_report(sel)
        self.assertEqual(sum(r["clips"] for r in rep.values()), len(sel))
        self.assertGreater(info["row_groups_needed"], 0)


class AudioTests(unittest.TestCase):
    def test_stereo_48k_to_mono_16k_float32(self):
        x48 = synthetic_voice(2.0, sr=48000)
        out = prepare(np.stack([x48, x48], axis=1), 48000, trim=False)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)
        self.assertAlmostEqual(len(out) / SR, 2.0, delta=0.01)

    def test_decode_wav_bytes(self):
        out = prepare_bytes(wav_bytes(synthetic_voice(2.0), channels=2))
        self.assertAlmostEqual(len(out) / SR, 2.0, delta=0.05)

    def test_silent_short_corrupt_nonfinite(self):
        cases = [(np.zeros(SR * 2, np.float32), "silent"), (synthetic_voice(0.4), "too_short")]
        for x, reason in cases:
            with self.assertRaises(AudioError) as ctx:
                prepare(x, SR)
            self.assertEqual(ctx.exception.reason, reason)
        with self.assertRaises(AudioError) as ctx:
            decode(b"definitely not audio")
        self.assertEqual(ctx.exception.reason, "decode_failed")
        bad = synthetic_voice(2.0)
        bad[10] = np.nan
        with self.assertRaises(AudioError) as ctx:
            prepare(bad, SR)
        self.assertEqual(ctx.exception.reason, "nonfinite_audio")
        with self.assertRaises(AudioError):
            decode(b"")

    def test_long_clip_cropped_and_silence_trimmed(self):
        x = np.concatenate([np.zeros(SR, np.float32), synthetic_voice(9.0), np.zeros(SR, np.float32)])
        self.assertLess(len(trim_silence(x)), len(x) - SR)
        out = prepare(x, SR)
        self.assertEqual(len(out), int(config.MAX_CLIP_SECONDS * SR))


class HeadTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(m, 1, (40, 8)) for m in (0, 3, 6)])
        return X, np.repeat([0, 1, 2], 40)

    def test_logreg_roundtrip(self):
        from voice_age.inference import _unpack
        from voice_age.model import head_from_state
        X, y = self.data()
        h = LogRegHead().fit(X, y, n_classes=3)
        tmp = tempfile.mkdtemp()
        try:
            save_model(tmp, h.state(), {"classes": CLASSES})
            st = _unpack(np.load(os.path.join(tmp, "head.npz"), allow_pickle=True)["state"].item())
            np.testing.assert_allclose(head_from_state(st, 8).predict_proba(X), h.predict_proba(X), atol=1e-6)
        finally:
            shutil.rmtree(tmp)

    def test_mlp_early_stopping_keeps_best(self):
        X, y = self.data()
        h = MLPHead(hidden=16, epochs=40, patience=3).fit(X, y, X, y, 3)
        self.assertLessEqual(h.best_epoch, len(h.history))
        best = max(r["val_macro_f1"] for r in h.history)
        self.assertAlmostEqual(metrics(y, h.predict_proba(X).argmax(1), CLASSES)["macro_f1"], best, places=6)

    def test_majority_baseline(self):
        b = majority_baseline(np.array([1, 1, 0]), np.array([0, 1, 2, 1]), CLASSES)
        self.assertEqual(b["predicts"], "Twenties")
        self.assertAlmostEqual(b["accuracy"], 0.5)


class EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.est = make_model_dir(os.path.join(cls.tmp, "m"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_probabilities(self):
        r = self.est.predict(synthetic_voice(3.0, 260.0, seed=50))
        self.assertEqual(list(r["probabilities"]), CLASSES)
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=6)
        self.assertEqual(r["confidence"], r["probabilities"][r["age_group"]])
        self.assertEqual(r["age_group"], "Teens")              # high synthetic f0 -> the 'Teens' fixture class

    def test_long_speech_split_into_segments(self):
        r = self.est.predict(synthetic_voice(14.0, 110.0, seed=3))
        self.assertEqual(r["age_group"], "Thirties")
        self.assertGreater(r["speech_seconds"], 13.0)

    def test_short_or_silent_audio_refused(self):
        for x in (synthetic_voice(0.5), np.zeros(SR * 3, np.float32)):
            with self.assertRaises(AudioError):
                self.est.predict(x)

    def test_status(self):
        st = self.est.status()
        self.assertTrue(st["model_available"] and st["inference_ready"])
        self.assertEqual(st["age_classes"], CLASSES)
        self.assertIn("Estimated age group", st["label"])

    def test_missing_model(self):
        est = AgeEstimator(os.path.join(self.tmp, "nothing"))
        st = est.status()
        self.assertFalse(st["model_available"])
        self.assertIn("not trained yet", st["message"])
        with self.assertRaises(AgeModelUnavailable):
            est.predict(synthetic_voice(3.0))

    def test_result_helper(self):
        r = result_from_probabilities([1, 3], ["A", "B"])
        self.assertEqual((r["age_group"], r["confidence"]), ("B", 0.75))


class LiveTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.est = make_model_dir(os.path.join(cls.tmp, "m"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def feed_audio(self, tracker, x, speech=True):
        for i in range(0, len(x) - 511, 512):
            tracker.feed(x[i:i + 512], speech)

    def wait_for(self, tracker, predicate, timeout=10.0):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = tracker.get_status()
            if predicate(st):
                return st
            time.sleep(0.05)
        return tracker.get_status()

    def test_estimates_after_enough_speech_and_smooths(self):
        tr = LiveAgeTracker(estimator=self.est)
        with redirect_stdout(io.StringIO()):
            tr.start_session()
            self.feed_audio(tr, synthetic_voice(3.2, 110.0, seed=1))
            st = self.wait_for(tr, lambda s: s["windows_classified"] >= 1)
            self.assertEqual(st["state"], "estimated")
            self.assertEqual(st["age_group"], "Thirties")
            self.feed_audio(tr, synthetic_voice(3.2, 112.0, seed=2))
            st = self.wait_for(tr, lambda s: s["windows_classified"] >= 2)
            self.assertAlmostEqual(sum(st["probabilities"].values()), 1.0, places=6)
            tr.stop_session()
        self.assertEqual(tr.get_status()["state"], "stopped")
        self.assertIsNone(tr.get_status()["age_group"])

    def test_insufficient_speech(self):
        tr = LiveAgeTracker(estimator=self.est)
        tr.start_session()
        self.feed_audio(tr, synthetic_voice(0.8, 150.0))
        self.feed_audio(tr, np.zeros(SR, np.float32), speech=False)        # phrase ends after 0.8 s
        st = tr.get_status()
        self.assertEqual(st["state"], "insufficient_speech")
        self.assertIsNone(st["age_group"])
        tr.stop_session()

    def test_no_model_and_bad_chunks_never_raise(self):
        tr = LiveAgeTracker(estimator=AgeEstimator(os.path.join(self.tmp, "none")))
        tr.start_session()
        tr.feed(None, True)                     # malformed input: logged, never raised
        self.feed_audio(tr, synthetic_voice(3.5))
        st = tr.get_status()
        self.assertEqual(st["state"], "model_not_trained")
        self.assertIn("not trained yet", st["message"])
        tr.stop_session()

    def test_feed_ignored_when_not_running(self):
        tr = LiveAgeTracker(estimator=self.est)
        self.feed_audio(tr, synthetic_voice(4.0))
        self.assertEqual(tr.get_status()["speech_seconds_buffered"], 0.0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        try:
            with redirect_stdout(io.StringIO()):
                import app as app_module
            cls.app = app_module
            cls.client = app_module.app.test_client()
            cls.original = app_module.voice_age.estimator
        except Exception:
            cls.app = None

    @classmethod
    def tearDownClass(cls):
        if cls.app is not None:
            cls.app.voice_age.estimator = cls.original
        shutil.rmtree(cls.tmp)

    def setUp(self):
        if self.app is None:
            self.skipTest("app.py could not be imported")

    def post_audio(self, data):
        with redirect_stdout(io.StringIO()):
            return self.client.post("/voice/age", data={} if data is None else
                                    {"audio": (io.BytesIO(data), "clip.wav")}, content_type="multipart/form-data")

    def test_status_without_model(self):
        self.app.voice_age.estimator = AgeEstimator(os.path.join(self.tmp, "absent"))
        st = self.client.get("/voice/age/status").get_json()
        self.assertFalse(st["model_available"])
        self.assertIn("not trained yet", st["message"])
        self.assertEqual(self.post_audio(wav_bytes(synthetic_voice(3.0))).status_code, 503)
        self.app.voice_age.estimator = self.original

    def test_estimate_with_model(self):
        self.app.voice_age.estimator = make_model_dir(os.path.join(self.tmp, "m"))
        try:
            res = self.post_audio(wav_bytes(synthetic_voice(3.0, 260.0, seed=9)))
            self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
            data = res.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(set(data["probabilities"]), set(CLASSES))
            self.assertIn(data["age_group"], CLASSES)
            self.assertIn("Estimated age group", data["label"])
            short = self.post_audio(wav_bytes(synthetic_voice(0.3)))
            self.assertEqual(short.status_code, 422)
            self.assertEqual(short.get_json()["kind"], "insufficient_audio")
            self.assertEqual(self.post_audio(None).status_code, 400)
            st = self.client.get("/voice/age/status").get_json()
            self.assertTrue(st["model_available"])
            self.assertEqual(st["age_classes"], CLASSES)
        finally:
            self.app.voice_age.estimator = self.original

    def test_voice_status_keeps_emotion_and_adds_age(self):
        with redirect_stdout(io.StringIO()):
            st = self.client.get("/voice/status").get_json()
        for key in ("running", "emotion", "probabilities", "gender"):      # existing emotion / gender fields
            self.assertIn(key, st)
        self.assertIn("age", st)
        self.assertIn("state", st["age"])
        self.assertNotIn("age", st["probabilities"] or {})                 # emotion output untouched

    def test_voice_page_shows_separate_age_card(self):
        with redirect_stdout(io.StringIO()):
            html = self.client.get("/voice").get_data(as_text=True)
        self.assertIn("Voice Age Estimation", html)
        self.assertIn("Estimated age group from voice", html)
        self.assertIn("Live Voice Emotion", html)
        self.assertIn("Speaker Gender", html)

    def test_other_pages_still_render(self):
        for url in ("/", "/camera", "/handwriting", "/body-vitals"):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_emotion_model_untouched(self):
        import emotion_detector
        self.assertEqual(emotion_detector.EMOTION_MODEL_NAME, "superb/hubert-base-superb-er")
        self.assertEqual(emotion_detector.EMOTIONS, ("happy", "angry", "sad", "neutral"))
        self.assertIs(self.app.voice_detector.on_vad_chunk.__func__ if hasattr(self.app.voice_detector.on_vad_chunk, "__func__") else self.app.voice_detector.on_vad_chunk, self.app.on_voice_vad_chunk)


if __name__ == "__main__":
    unittest.main()

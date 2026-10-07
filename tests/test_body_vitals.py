"""
Tests for the Body Vitals modality (body_vitals/):  python -m unittest discover tests

Signals here are synthetic waveforms with known properties (heart rate, RR
series, breathing rate, EDA slope). The WESAD-format files and the tiny model
are throw-away fixtures in a temporary folder - they are not WESAD data, and
models/ is never written by these tests.
"""

import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from body_vitals import config, dataset, features, normalization, preprocessing, providers  # noqa: E402
from body_vitals.inference import (BodyVitalsInputError, BodyVitalsMLEngine,  # noqa: E402
                                   BodyVitalsModelUnavailable)
from body_vitals.model import compute_metrics, majority_baseline, subject_split  # noqa: E402
from body_vitals.preprocessing import SignalQualityError  # noqa: E402

FS = 700


def ecg_from_rr(rr, fs=FS, seconds=60, seed=0):
    """Synthetic ECG with beats exactly at the cumulative RR times."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    ecg = rng.normal(0, 0.01, n)
    beat = 0.5
    times = []
    for r in rr:
        if beat > seconds - 0.5:
            break
        times.append(beat)
        for off, width, amp in [(-0.025, 0.008, -0.12), (0.0, 0.010, 1.0), (0.025, 0.010, -0.25),
                                (0.25, 0.040, 0.30)]:
            ecg += amp * np.exp(-0.5 * ((t - beat - off) / width) ** 2)
        beat += r
    return ecg, np.array(times)


def sine_resp(rate_bpm, fs=FS, seconds=60):
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * rate_bpm / 60.0 * t)


def flat_eda(level=3.0, slope=0.0, fs=FS, seconds=60):
    t = np.arange(int(seconds * fs)) / fs
    return level + slope * t + np.random.default_rng(1).normal(0, 0.001, len(t))


def good_window(seed=0):
    rng = np.random.default_rng(seed)
    rr = 0.8 + rng.normal(0, 0.03, 120)
    ecg, _ = ecg_from_rr(rr, seed=seed)
    return ecg, flat_eda(), sine_resp(15)


def random_feature_rows(n_per_class=30, seed=0, names=None):
    """Feature rows whose class shifts hr_mean / eda_mean, for a tiny test model."""
    import pandas as pd
    names = names or config.FEATURE_NAMES
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label in range(3):
        base = rng.normal(0, 1, (n_per_class, len(names))) + 5
        base[:, names.index("hr_mean")] += label * 20
        base[:, names.index("eda_mean")] += label * 2
        rows.append(base)
        labels += [label] * n_per_class
    return pd.DataFrame(np.vstack(rows), columns=names), np.array(labels)


def write_tiny_model(folder, feature_set=None, normalization="none", **meta_overrides):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    feature_set = feature_set or config.FEATURE_VERSION
    names = config.FEATURE_SETS[feature_set]
    X, y = random_feature_rows(names=names)
    if normalization == "rest_calibration":
        X = X - 5.0                       # "relative to rest": centred features
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    meta = {"dataset": "WESAD", "classes": config.CLASSES, "feature_names": names,
            "feature_version": feature_set, "model_type": "RandomForest", "window_size_sec": 60,
            "window_step_sec": 30, "normalization": normalization,
            "calibration_sec": 300 if normalization == "rest_calibration" else None,
            "test_subjects": ["S98", "S99"], "evaluation_metrics": {"test": {
                "accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5,
                "per_class": {c: {"recall": 0.5} for c in config.CLASSES}}}}
    meta.update(meta_overrides)
    mp, jp = os.path.join(folder, "m.pkl"), os.path.join(folder, "m.json")
    joblib.dump(model, mp)
    with open(jp, "w") as fh:
        json.dump(meta, fh)
    return mp, jp


# ---------------------------------------------------------------------------
class EcgRrTests(unittest.TestCase):
    def test_r_peaks_match_known_beats(self):
        rr = np.full(100, 0.75)                                   # 80 bpm
        ecg, times = ecg_from_rr(rr)
        peaks = preprocessing.detect_r_peaks(ecg, FS)
        self.assertEqual(len(peaks), len(times))
        self.assertLess(np.max(np.abs(peaks / FS - times)), 0.005)

    def test_inverted_lead_detected(self):
        ecg, times = ecg_from_rr(np.full(100, 0.9))
        self.assertEqual(len(preprocessing.detect_r_peaks(-ecg, FS)), len(times))

    def test_heart_rate_features(self):
        ecg, _ = ecg_from_rr(np.full(100, 0.75))
        feats, quality = features.ecg_features(ecg, FS)
        self.assertAlmostEqual(feats["hr_mean"], 80.0, delta=0.5)
        self.assertLess(feats["hr_std"], 1.0)
        self.assertLessEqual(feats["hr_min"], feats["hr_mean"])
        self.assertGreaterEqual(feats["hr_max"], feats["hr_mean"])
        self.assertEqual(quality["rr_artifact_fraction"], 0.0)

    def test_missed_beat_rr_is_rejected(self):
        peaks = np.cumsum(np.full(40, 0.8 * FS)).astype(int)
        peaks = np.delete(peaks, 20)                              # one missed beat -> 1.6 s RR
        rr, valid = preprocessing.clean_rr(peaks, FS)
        self.assertFalse(valid[19])
        self.assertEqual(int((~valid).sum()), 1)

    def test_implausible_rr_rejected(self):
        rr, valid = preprocessing.clean_rr(np.array([0, 100, 2400, 2960]), FS)   # 0.14 s, 3.3 s, 0.8 s
        self.assertFalse(valid[0])
        self.assertFalse(valid[1])

    def test_noise_only_ecg_rejected(self):
        noise = np.random.default_rng(0).normal(0, 1, 60 * FS)
        with self.assertRaises(SignalQualityError):
            features.ecg_features(noise, FS)


class HrvTests(unittest.TestCase):
    def test_sdnn_rmssd_from_rr_intervals(self):
        rr = np.array([0.80, 0.85, 0.78, 0.90, 0.82, 0.88])
        out = features.hrv_from_rr(rr, np.ones(len(rr), bool))
        self.assertAlmostEqual(out["hrv_sdnn"], np.std(rr, ddof=1) * 1000, places=6)
        self.assertAlmostEqual(out["hrv_rmssd"], np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000, places=6)

    def test_rmssd_skips_pairs_across_rejected_beats(self):
        rr = np.array([0.80, 0.82, 1.60, 0.81, 0.83, 0.80])
        valid = np.array([True, True, False, True, True, True])
        out = features.hrv_from_rr(rr, valid)
        diffs = np.array([0.82 - 0.80, 0.83 - 0.81, 0.80 - 0.83])
        self.assertAlmostEqual(out["hrv_rmssd"], np.sqrt(np.mean(diffs ** 2)) * 1000, places=6)
        self.assertAlmostEqual(out["hrv_sdnn"], np.std(rr[valid], ddof=1) * 1000, places=6)

    def test_hrv_from_synthetic_ecg_matches_rr(self):
        rng = np.random.default_rng(3)
        rr = 0.8 + rng.normal(0, 0.04, 100)
        ecg, times = ecg_from_rr(rr)
        feats, _ = features.ecg_features(ecg, FS)
        true_rr = np.diff(times)
        self.assertAlmostEqual(feats["hrv_sdnn"], np.std(true_rr, ddof=1) * 1000, delta=3)
        self.assertAlmostEqual(feats["hrv_rmssd"], np.sqrt(np.mean(np.diff(true_rr) ** 2)) * 1000, delta=4)

    def test_too_few_rr(self):
        with self.assertRaises(SignalQualityError) as ctx:
            features.hrv_from_rr(np.array([0.8, 0.8]), np.array([True, True]))
        self.assertEqual(ctx.exception.reason, "too_few_rr")


class EdaTests(unittest.TestCase):
    def test_level_and_slope(self):
        feats, _ = features.eda_features(flat_eda(level=2.0, slope=0.01), FS)
        self.assertAlmostEqual(feats["eda_slope"], 0.01, delta=0.001)
        self.assertAlmostEqual(feats["eda_mean"], 2.3, delta=0.02)
        self.assertAlmostEqual(feats["eda_range"], 0.6, delta=0.05)
        self.assertEqual(feats["eda_scr_per_min"], 0.0)

    def test_skin_conductance_responses_counted(self):
        eda = providers.synth_eda(3.0, 0.0, 0, 0, 60, 32, np.random.default_rng(0))
        t = np.arange(len(eda)) / 32
        for onset in (10, 25, 40):
            dt = np.clip(t - onset, 0, None)
            eda += 0.2 * (1 - np.exp(-dt / 0.75)) * np.exp(-dt / 4.0) * (t >= onset)
        feats, quality = features.eda_features(eda, 32)
        self.assertEqual(quality["scr_peaks"], 3)
        self.assertGreater(feats["eda_phasic_std"], 0.01)

    def test_flat_eda_rejected(self):
        with self.assertRaises(SignalQualityError) as ctx:
            features.eda_features(np.full(60 * FS, 2.0), FS)
        self.assertEqual(ctx.exception.reason, "flat_eda")


class RespirationTests(unittest.TestCase):
    def test_rate_from_regular_breathing(self):
        for rate in (8, 15, 24):
            feats, _ = features.resp_features(sine_resp(rate), FS)
            self.assertAlmostEqual(feats["resp_rate_mean"], rate, delta=0.6, msg=f"{rate} bpm")
            self.assertLess(feats["resp_interval_cv"], 0.05)

    def test_irregular_breathing_has_variability(self):
        resp = providers.synth_resp(16, 0.3, 60, 32, np.random.default_rng(0))
        feats, _ = features.resp_features(resp, 32)
        self.assertGreater(feats["resp_rate_std"], 1.0)
        self.assertLess(feats["resp_rate_min"], feats["resp_rate_max"])

    def test_flat_respiration_rejected(self):
        with self.assertRaises(SignalQualityError):
            features.resp_features(np.zeros(60 * FS), FS)


class WindowingTests(unittest.TestCase):
    def test_label_segments_keep_only_requested(self):
        labels = np.array([0] * 5 + [1] * 10 + [4] * 3 + [2] * 7 + [1] * 4)
        segs = preprocessing.label_segments(labels, config.WESAD_LABEL_TO_CLASS)
        self.assertEqual(segs, [(1, 5, 15), (2, 18, 25), (1, 25, 29)])

    def test_windows_inside_segment_with_overlap(self):
        bounds = preprocessing.window_bounds(0, 150 * FS, FS, 60, 30)
        self.assertEqual([(a // FS, b // FS) for a, b in bounds], [(0, 60), (30, 90), (60, 120), (90, 150)])
        self.assertTrue(all(b <= 150 * FS for _, b in bounds))

    def test_short_segment_gives_no_window(self):
        self.assertEqual(preprocessing.window_bounds(0, 59 * FS, FS, 60, 30), [])


class MissingDataTests(unittest.TestCase):
    def test_nan_in_ecg(self):
        ecg, eda, resp = good_window()
        ecg[1000:1010] = np.nan
        with self.assertRaises(SignalQualityError) as ctx:
            features.extract_window_features(ecg, eda, resp, FS)
        self.assertEqual(ctx.exception.reason, "nonfinite_ecg")

    def test_missing_eda(self):
        ecg, _, resp = good_window()
        with self.assertRaises(SignalQualityError) as ctx:
            features.extract_window_features(ecg, [], resp, FS)
        self.assertEqual(ctx.exception.reason, "missing_eda")

    def test_nan_in_respiration(self):
        ecg, eda, resp = good_window()
        resp[:] = np.nan
        with self.assertRaises(SignalQualityError) as ctx:
            features.extract_window_features(ecg, eda, resp, FS)
        self.assertEqual(ctx.exception.reason, "nonfinite_resp")

    def test_good_window_gives_all_features_in_order(self):
        feats, quality = features.extract_window_features(*good_window(), FS)
        self.assertEqual(list(feats), config.ALL_FEATURE_NAMES)
        self.assertEqual(config.ALL_FEATURE_NAMES[:19], config.FEATURES_V1)
        self.assertTrue(all(np.isfinite(v) for v in feats.values()))
        self.assertGreater(quality["rr_valid"], 60)


class WesadLoaderTests(unittest.TestCase):
    """A small file in the WESAD pickle FORMAT (synthetic content) in a temp folder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        seconds = 200
        n = seconds * FS
        rr = 0.8 + np.random.default_rng(0).normal(0, 0.03, 400)
        ecg, _ = ecg_from_rr(rr, seconds=seconds)
        labels = np.zeros(n, dtype=int)
        labels[5 * FS:95 * FS] = 1                # 90 s baseline -> 2 windows (0-60, 30-90)
        labels[100 * FS:170 * FS] = 2             # 70 s stress -> 1 window
        labels[170 * FS:180 * FS] = 3             # 10 s amusement -> too short, no window
        eda = flat_eda(seconds=seconds)
        resp = sine_resp(15, seconds=seconds)
        ecg_bad = ecg.copy()
        ecg_bad[120 * FS:121 * FS] = np.nan       # spoils the stress window
        cls.write("S90", ecg, eda, resp, labels)
        cls.write("S91", ecg_bad, eda, resp, labels)
        os.makedirs(os.path.join(cls.tmp, "S92"))
        with open(os.path.join(cls.tmp, "S92", "S92.pkl"), "wb") as fh:
            fh.write(b"not a pickle")             # corrupted download
        cls.write("S93", ecg, eda, None, labels)  # respiration missing

    @classmethod
    def write(cls, subject, ecg, eda, resp, labels):
        chest = {"ECG": ecg[:, None], "EDA": eda[:, None]}
        if resp is not None:
            chest["Resp"] = resp[:, None]
        os.makedirs(os.path.join(cls.tmp, subject), exist_ok=True)
        with open(os.path.join(cls.tmp, subject, f"{subject}.pkl"), "wb") as fh:
            pickle.dump({"signal": {"chest": chest, "wrist": {}}, "label": labels, "subject": subject}, fh)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_finds_subjects(self):
        self.assertEqual(dataset.find_subjects(self.tmp), ["S90", "S91", "S92", "S93"])

    def test_windows_per_label(self):
        rows, summary = dataset.process_subject(self.tmp, "S90")
        self.assertEqual(summary["windows_kept"], {"baseline": 2, "stress": 1})
        self.assertEqual([r["class"] for r in rows], ["baseline", "baseline", "stress"])
        self.assertEqual(summary["windows_discarded"], 0)

    def test_bad_window_logged_with_reason(self):
        rows, summary = dataset.process_subject(self.tmp, "S91")
        self.assertEqual(summary["windows_discarded"], 1)
        self.assertEqual(summary["discard_reasons"], {"nonfinite_ecg": 1})
        self.assertEqual(summary["discard_log"][0]["class"], "stress")

    def test_corrupted_file_skipped(self):
        with self.assertRaises(dataset.SubjectSkipped):
            dataset.process_subject(self.tmp, "S92")

    def test_missing_signal_skipped(self):
        with self.assertRaises(dataset.SubjectSkipped) as ctx:
            dataset.process_subject(self.tmp, "S93")
        self.assertIn("Resp", str(ctx.exception))

    def test_prepare_command_end_to_end(self):
        from body_vitals import prepare_wesad
        out, rep = os.path.join(self.tmp, "f.csv"), os.path.join(self.tmp, "r.json")
        with redirect_stdout(io.StringIO()):
            report = prepare_wesad.run(prepare_wesad.parse_args(["--data", self.tmp, "--out", out,
                                                                 "--report", rep, "--jobs", "1",
                                                                 "--window-sec", "60", "--step-sec", "30"]))
        self.assertEqual(report["subjects_processed"], ["S90", "S91"])
        self.assertEqual(set(report["subjects_skipped"]), {"S92", "S93"})
        self.assertEqual(report["windows_created"], 6)
        self.assertEqual(report["windows_discarded"], 1)
        self.assertEqual(report["class_distribution"], {"baseline": 4, "stress": 1, "amusement": 0})
        df = dataset.load_features(out)
        self.assertEqual(list(df.columns[5:5 + len(config.FEATURE_NAMES)]), config.FEATURE_NAMES)


class SplitAndMetricTests(unittest.TestCase):
    SUBJECTS = [f"S{i}" for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)]

    def test_subject_split_disjoint_and_complete(self):
        train, val, test = subject_split(self.SUBJECTS, seed=42)
        self.assertEqual((len(train), len(val), len(test)), (9, 3, 3))
        self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        self.assertEqual(set(train) | set(val) | set(test), set(self.SUBJECTS))

    def test_subject_split_reproducible(self):
        self.assertEqual(subject_split(self.SUBJECTS, seed=7), subject_split(self.SUBJECTS, seed=7))

    def test_majority_baseline(self):
        out = majority_baseline([0, 0, 0, 1, 2], [0, 1, 2, 0])
        self.assertEqual(out["predicts"], "baseline")
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertAlmostEqual(out["balanced_accuracy"], 1 / 3)

    def test_metrics_shape(self):
        m = compute_metrics([0, 1, 2, 2], [0, 1, 2, 0])
        self.assertEqual(set(m["per_class"]), set(config.CLASSES))
        self.assertEqual(m["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.model_path, cls.meta_path = write_tiny_model(cls.tmp)
        cls.engine = BodyVitalsMLEngine(cls.model_path, cls.meta_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def features(self, label=1):
        X, y = random_feature_rows(seed=5)
        return dict(X[y == label].iloc[0])

    def test_prediction_format(self):
        out = self.engine.predict(self.features(1))
        self.assertIn(out["predicted_state"], config.CLASSES)
        self.assertEqual(list(out["probabilities"]), config.CLASSES)
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)
        self.assertEqual(out["confidence"], max(out["probabilities"].values()))

    def test_key_order_does_not_matter(self):
        feats = self.features(2)
        shuffled = dict(reversed(list(feats.items())))
        self.assertEqual(self.engine.predict(feats), self.engine.predict(shuffled))

    def test_missing_unknown_and_nan_features_rejected(self):
        feats = self.features()
        for bad in ({k: v for k, v in feats.items() if k != "hrv_rmssd"}, dict(feats, happiness=1.0),
                    dict(feats, hr_mean=float("nan")), dict(feats, hr_mean="fast"), [1, 2, 3]):
            with self.assertRaises(BodyVitalsInputError):
                self.engine.predict(bad)

    def test_missing_model_unavailable(self):
        engine = BodyVitalsMLEngine(os.path.join(self.tmp, "none.pkl"), os.path.join(self.tmp, "none.json"))
        with self.assertRaises(BodyVitalsModelUnavailable):
            engine.predict(self.features())
        self.assertFalse(engine.status()["ready"])

    def test_feature_order_mismatch_refused(self):
        folder = tempfile.mkdtemp(dir=self.tmp)
        swapped = list(config.FEATURE_NAMES)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        engine = BodyVitalsMLEngine(*write_tiny_model(folder, feature_names=swapped))
        with self.assertRaises(BodyVitalsModelUnavailable) as ctx:
            engine.load()
        self.assertIn("order", str(ctx.exception))

    def test_feature_version_and_classes_mismatch_refused(self):
        for override in ({"feature_version": "old"}, {"classes": ["calm", "happy", "angry"]},
                         {"dataset": "OTHER"}):
            folder = tempfile.mkdtemp(dir=self.tmp)
            with self.assertRaises(BodyVitalsModelUnavailable):
                BodyVitalsMLEngine(*write_tiny_model(folder, **override)).load()

    def test_status_reports_model(self):
        st = self.engine.status()
        self.assertTrue(st["ready"])
        self.assertEqual(st["model"]["dataset"], "WESAD")

    def test_predict_window_from_raw_signals(self):
        window = providers.DemoVitalsProvider("stress", seed=1).read_window()
        feats, quality, out = self.engine.predict_window(window)
        self.assertEqual(list(feats), config.FEATURE_NAMES)
        self.assertIn(out["predicted_state"], config.CLASSES)


class DemoProviderTests(unittest.TestCase):
    def test_demo_is_labelled_simulated(self):
        w = providers.DemoVitalsProvider("baseline", seed=0).read_window()
        self.assertTrue(w.is_simulated)
        self.assertEqual(w.source, "demo")
        self.assertAlmostEqual(w.seconds, config.WINDOW_SEC, places=3)

    def test_demo_reproducible_with_seed(self):
        a = providers.DemoVitalsProvider("stress", seed=4).read_window()
        b = providers.DemoVitalsProvider("stress", seed=4).read_window()
        np.testing.assert_array_equal(a.ecg, b.ecg)

    def test_demo_features_follow_scenario(self):
        base, _ = features.extract_window_features(*self.signals("baseline"))
        stress, _ = features.extract_window_features(*self.signals("stress"))
        self.assertAlmostEqual(base["hr_mean"], providers.DEMO_SCENARIOS["baseline"]["hr"], delta=10)
        self.assertGreater(stress["hr_mean"], base["hr_mean"])
        self.assertLess(stress["hrv_rmssd"], base["hrv_rmssd"])

    @staticmethod
    def signals(scenario):
        w = providers.DemoVitalsProvider(scenario, seed=0, jitter=0).read_window()
        return w.ecg, w.eda, w.resp, w.fs_ecg, w.fs_eda, w.fs_resp

    def test_unknown_scenario(self):
        with self.assertRaises(ValueError):
            providers.DemoVitalsProvider("happy")


class NoRuleBasedLayerTests(unittest.TestCase):
    def test_interpretation_module_removed(self):
        import importlib.util
        self.assertIsNone(importlib.util.find_spec("body_vitals.interpretation"))

    def test_no_arousal_valence_in_body_vitals_code(self):
        folders = [os.path.join(ROOT, "body_vitals")]
        files = [os.path.join(ROOT, "templates", "body_vitals.html"), os.path.join(ROOT, "static", "js", "body_vitals.js")]
        for folder in folders:
            files += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".py")]
        for path in files:
            with open(path, encoding="utf-8") as fh:
                text = fh.read().lower()
            for word in ("arousal", "valence", "rule-based"):
                self.assertNotIn(word, text, f"{word!r} still in {os.path.relpath(path, ROOT)}")


class NewFeatureTests(unittest.TestCase):
    """The 10 wesad-chest-v2 features, on signals with known properties."""

    def test_pnn50_and_hr_shape(self):
        rr = np.tile([0.80, 0.86], 60)                            # every successive difference = 60 ms
        ecg, _ = ecg_from_rr(rr)
        feats, _ = features.ecg_features(ecg, FS)
        self.assertGreater(feats["hrv_pnn50"], 0.9)
        self.assertAlmostEqual(feats["hr_range"], feats["hr_max"] - feats["hr_min"], places=9)
        self.assertAlmostEqual(feats["hr_cv"], feats["hr_std"] / feats["hr_mean"], places=9)
        steady, _ = features.ecg_features(ecg_from_rr(np.full(100, 0.8))[0], FS)
        self.assertEqual(steady["hrv_pnn50"], 0.0)

    def test_scr_amplitudes(self):
        eda = providers.synth_eda(3.0, 0.0, 0, 0, 60, 32, np.random.default_rng(0))
        none, _ = features.eda_features(eda, 32)
        self.assertEqual((none["eda_scr_amp_mean"], none["eda_scr_amp_max"]), (0.0, 0.0))
        t = np.arange(len(eda)) / 32
        for onset, amp in ((10, 0.1), (30, 0.3)):
            dt = np.clip(t - onset, 0, None)
            eda = eda + amp * (1 - np.exp(-dt / 0.75)) * np.exp(-dt / 4.0) * (t >= onset)
        feats, _ = features.eda_features(eda, 32)
        self.assertGreater(feats["eda_scr_amp_max"], feats["eda_scr_amp_mean"])
        self.assertGreater(feats["eda_cv"], 0)

    def test_breath_depth_and_timing(self):
        feats, _ = features.resp_features(2.0 * sine_resp(15), FS)
        self.assertAlmostEqual(feats["resp_ie_ratio"], 1.0, delta=0.1)       # symmetric breaths
        self.assertAlmostEqual(feats["resp_amp_mean"], 4.0, delta=0.6)       # peak-to-trough of 2*sin
        self.assertLess(feats["resp_amp_cv"], 0.05)

    def test_v1_values_unchanged_by_v2(self):
        """The 19 original features are computed exactly as before (same code path)."""
        feats, _ = features.extract_window_features(*good_window(seed=4), FS)
        ecg_f, _ = features.ecg_features(good_window(seed=4)[0], FS)
        for k in ("hr_mean", "hr_std", "hrv_sdnn", "hrv_rmssd"):
            self.assertEqual(feats[k], ecg_f[k])


class NormalizationTests(unittest.TestCase):
    def table(self):
        import pandas as pd
        rows = []
        for subject, offset in (("S90", 0.0), ("S91", 10.0)):
            for i, start in enumerate(range(0, 600, 30)):                  # 20 baseline windows
                rows.append({"subject": subject, "class": "baseline", "label": 0, "start_s": 100 + start,
                             "hr_mean": 70 + offset + (i >= 9) * 1.0})
            for start in range(1000, 1300, 30):
                rows.append({"subject": subject, "class": "stress", "label": 1, "start_s": start,
                             "hr_mean": 90 + offset})
        return pd.DataFrame(rows)

    def test_reference_windows_are_first_five_minutes_of_rest(self):
        df = self.table()
        ref, overlap = normalization.calibration_masks(df, 60)
        starts = df.loc[ref & (df["subject"] == "S90").to_numpy(), "start_s"].tolist()
        self.assertEqual(starts, list(range(100, 341, 30)))                # windows fully inside 100-400 s
        self.assertTrue((df.loc[overlap, "class"] == "baseline").all())

    def test_calibrated_table_subtracts_own_reference_and_drops_calibration(self):
        out, refs = normalization.calibrate_table(self.table(), ["hr_mean"], 60)
        self.assertEqual(refs["S90"]["hr_mean"], 70.0)
        self.assertEqual(refs["S91"]["hr_mean"], 80.0)                     # each subject: its own rest only
        self.assertTrue((out["start_s"] >= 400 - 60).all() | (out["class"] != "baseline").all())
        stress = out[out["class"] == "stress"]
        self.assertTrue(np.allclose(stress["hr_mean"], 20.0))              # offset between people removed

    def test_drop_calibration_period_matches(self):
        df = self.table()
        a, _ = normalization.calibrate_table(df, ["hr_mean"], 60)
        b = normalization.drop_calibration_period(df, 60)
        self.assertEqual(list(a.index), list(b.index))


class CalibratedEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.engine = BodyVitalsMLEngine(*write_tiny_model(cls.tmp, "wesad-chest-v2", "rest_calibration"))
        cls.v2 = BodyVitalsMLEngine(*write_tiny_model(tempfile.mkdtemp(dir=cls.tmp), "wesad-chest-v2"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_v2_model_uses_29_features(self):
        self.assertEqual(self.v2.feature_names, config.FEATURE_SETS["wesad-chest-v2"])
        feats, _, out = self.v2.predict_window(providers.DemoVitalsProvider("stress", seed=1).read_window())
        self.assertEqual(len(feats), 29)
        self.assertIn(out["predicted_state"], config.CLASSES)

    def test_calibration_required(self):
        from body_vitals.inference import BodyVitalsCalibrationRequired
        X, _ = random_feature_rows(names=self.engine.feature_names)
        with self.assertRaises(BodyVitalsCalibrationRequired):
            self.engine.predict(dict(X.iloc[0]))
        with self.assertRaises(BodyVitalsCalibrationRequired):
            self.engine.predict_window(providers.DemoVitalsProvider("stress", seed=1).read_window())

    def test_reference_is_subtracted(self):
        names = self.engine.feature_names
        X, y = random_feature_rows(names=names, seed=3)
        x_stress = dict(X[y == 2].iloc[0])
        rest = {f: 5.0 for f in names}
        shifted = {f: v + 100 for f, v in x_stress.items()}
        rest_shifted = {f: 105.0 for f in names}
        a = self.engine.predict(x_stress, rest)
        b = self.engine.predict(shifted, rest_shifted)       # same person-relative values -> same answer
        self.assertEqual(a, b)

    def test_make_reference_is_median(self):
        names = self.engine.feature_names
        rows = [{f: v for f in names} for v in (1.0, 2.0, 10.0)]
        self.assertEqual(self.engine.make_reference(rows)[names[0]], 2.0)

    def test_predict_window_with_calibration(self):
        p = providers.DemoVitalsProvider("stress", seed=2)
        feats, quality, out = self.engine.predict_window(p.read_window(), p.read_calibration(3))
        self.assertEqual(quality["calibration_windows"], 3)
        self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)


class ExperimentSafetyTests(unittest.TestCase):
    def test_development_data_never_contains_test_subjects(self):
        from body_vitals import experiment
        path = config.features_csv(60, 30)
        if not os.path.isfile(path):
            self.skipTest("WESAD features not prepared")
        df = experiment.load_dev(60, 30)
        self.assertFalse(set(df["subject"]) & set(config.TEST_SUBJECTS))
        self.assertEqual(len(df["subject"].unique()), 12)

    def test_loso_folds_are_subject_disjoint(self):
        import pandas as pd
        from body_vitals import experiment
        X, y = random_feature_rows(n_per_class=12)
        df = X.copy()
        df["label"] = y
        df["subject"] = [f"S{i % 4}" for i in range(len(df))]
        cfg = {"name": "lr", "family": "LogisticRegression", "params": {"C": 1.0}}
        r, pred = experiment.loso_cv(df, config.FEATURE_VERSION, cfg, 0)
        self.assertEqual(len(r["folds"]), 4)
        self.assertTrue((pred >= 0).all())
        self.assertEqual(sorted(f["subject"] for f in r["folds"]), ["S0", "S1", "S2", "S3"])

    def test_leakage_audit_flags_overlap(self):
        from body_vitals.train import leakage_audit
        ok = leakage_audit(["S2"], ["S3"], list(config.TEST_SUBJECTS), ["S2", "S3"])
        self.assertTrue(all(ok.values()))
        bad = leakage_audit(["S2", "S4"], ["S3"], list(config.TEST_SUBJECTS), ["S2", "S3", "S4"])
        self.assertFalse(bad["train ∩ test = ∅"])
        self.assertFalse(bad["experiment (model/feature/window selection) used no test subject"])

    def test_all_model_families_fit(self):
        from body_vitals.model import default_configs, fit_model
        X, y = random_feature_rows(n_per_class=10)
        for cfg in default_configs():
            cfg = dict(cfg, params=dict(cfg["params"], **({"n_estimators": 10} if "n_estimators" in cfg["params"] else {})))
            m = fit_model(cfg, X, y)
            self.assertEqual(np.asarray(m.predict_proba(X)).shape, (len(X), 3), cfg["family"])


class TrainedArtifactTests(unittest.TestCase):
    """Checks the real models/body_vitals_* artefact when it exists (skipped otherwise)."""

    def setUp(self):
        if not (os.path.isfile(config.MODEL_PATH) and os.path.isfile(config.METADATA_PATH)):
            self.skipTest("no trained Body Vitals model in models/")
        with open(config.METADATA_PATH) as fh:
            self.meta = json.load(fh)

    def test_metadata_consistent_with_code(self):
        self.assertEqual(self.meta["feature_names"], config.FEATURE_SETS[self.meta["feature_version"]])
        self.assertEqual(self.meta["classes"], config.CLASSES)
        self.assertEqual(self.meta["dataset"], "WESAD")
        BodyVitalsMLEngine().load()

    def test_subjects_disjoint(self):
        tr, va, te = (set(self.meta[k]) for k in ("training_subjects", "validation_subjects", "test_subjects"))
        self.assertFalse(tr & va or tr & te or va & te)
        self.assertFalse(set(self.meta["final_fit_subjects"]) & te)
        self.assertTrue(all(self.meta["leakage_audit"].values()))

    def test_real_model_predicts_every_demo(self):
        engine = BodyVitalsMLEngine()
        for scenario in ("baseline", "stress", "amusement"):
            p = providers.DemoVitalsProvider(scenario, seed=0, seconds=engine.window_sec)
            calib = p.read_calibration(3) if engine.normalization == "rest_calibration" else None
            feats, _, out = engine.predict_window(p.read_window(), calib)
            self.assertEqual(list(feats), self.meta["feature_names"])
            self.assertIn(out["predicted_state"], config.CLASSES)
            self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=6)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        from body_vitals import routes
        cls.routes = routes
        cls.original = routes.engine
        routes.engine = BodyVitalsMLEngine(*write_tiny_model(cls.tmp))
        try:
            with redirect_stdout(io.StringIO()):
                import app as app_module
            cls.client = app_module.app.test_client()
        except Exception:
            cls.client = None

    @classmethod
    def tearDownClass(cls):
        cls.routes.engine = cls.original
        shutil.rmtree(cls.tmp)

    def setUp(self):
        if self.client is None:
            self.skipTest("app.py could not be imported")

    def post(self, url, body):
        with redirect_stdout(io.StringIO()):
            return self.client.post(url, json=body)

    def test_page_renders_and_labels_demo(self):
        res = self.client.get("/body-vitals")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        for text in ("Body Vitals", "Baseline Demo", "Stress Demo", "Amusement Demo", "demo mode",
                     "ML predicted state", "trained on WESAD", "Predicted physiological state",
                     "Heart rate", "HRV (SDNN)", "HRV (RMSSD)", "GSR / EDA", "Respiration"):
            self.assertIn(text, html)
        for removed in ("Arousal", "Valence", "RULE-BASED", "rule-based"):
            self.assertNotIn(removed, html)

    def test_landing_links_to_page(self):
        self.assertIn("/body-vitals", self.client.get("/").get_data(as_text=True))

    def test_status(self):
        st = self.client.get("/body-vitals/status").get_json()
        self.assertTrue(st["ready"])
        self.assertEqual(st["model"]["model_type"], "RandomForest")       # the page shows this name
        self.assertEqual(set(st["model"]["test_recall"]), set(config.CLASSES))

    def assert_ml_only(self, data):
        """Only the model's prediction: display-cased classes, probabilities, nothing rule-based."""
        display = [config.CLASS_DISPLAY[c] for c in config.CLASSES]
        self.assertTrue(data["success"])
        self.assertIn(data["predicted_state"], display)
        self.assertEqual(set(data["probabilities"]), set(display))      # jsonify sorts keys; order is in model.classes
        self.assertAlmostEqual(sum(data["probabilities"].values()), 1.0, places=5)
        self.assertEqual(data["confidence"], data["probabilities"][data["predicted_state"]])
        self.assertEqual(data["confidence"], max(data["probabilities"].values()))
        self.assertEqual(data["model"]["dataset"], "WESAD")
        self.assertEqual(data["model"]["classes"], display)
        body = json.dumps(data).lower()
        for removed in ("arousal", "valence", "interpretation", "rule-based"):
            self.assertNotIn(removed, body)

    def test_demo_response(self):
        res = self.post("/body-vitals/demo", {"scenario": "stress", "seed": 3})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assert_ml_only(data)
        self.assertTrue(data["input"]["is_simulated"])
        self.assertEqual(data["input"]["source"], "demo")
        self.assertIn("SIMULATED", data["input"]["label"])
        self.assertEqual(data["input"]["scenario_requested"], "stress")
        self.assertEqual(set(data["input"]["vitals"]), {"heart_rate_bpm", "hrv_sdnn_ms", "hrv_rmssd_ms",
                                                        "gsr_us", "respiration_rate_bpm"})
        self.assertEqual(set(data["input"]["features"]), set(config.FEATURE_NAMES))

    def test_every_demo_scenario_runs_the_model(self):
        for scenario in ("baseline", "stress", "amusement"):
            with self.subTest(scenario=scenario):
                res = self.post("/body-vitals/demo", {"scenario": scenario, "seed": 11})
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assert_ml_only(data)
                self.assertTrue(data["input"]["is_simulated"])
                self.assertEqual(data["input"]["scenario_requested"], scenario)

    def test_demo_prediction_comes_from_the_model(self):
        """The response must equal what the loaded model says for the same extracted features
        (no hardcoded or rule-based answer)."""
        data = self.post("/body-vitals/demo", {"scenario": "amusement", "seed": 5}).get_json()
        direct = self.routes.engine.predict(data["input"]["features"])
        self.assertEqual(data["predicted_state"], config.CLASS_DISPLAY[direct["predicted_state"]])
        for c in config.CLASSES:
            self.assertAlmostEqual(data["probabilities"][config.CLASS_DISPLAY[c]], direct["probabilities"][c],
                                   places=3)

    def test_demo_bad_scenario(self):
        self.assertEqual(self.post("/body-vitals/demo", {"scenario": "happy"}).status_code, 400)

    def test_predict_features(self):
        X, _ = random_feature_rows()
        data = self.post("/body-vitals/predict", {"features": dict(X.iloc[0])}).get_json()
        self.assert_ml_only(data)
        self.assertIsNone(data["input"]["is_simulated"])
        self.assertEqual(data["input"]["source"], "api")

    def test_predict_signals(self):
        ecg, eda, resp = good_window()
        res = self.post("/body-vitals/predict", {"signals": {"ecg": ecg.tolist(), "eda": eda.tolist(),
                                                            "resp": resp.tolist(), "fs": FS}})
        self.assertEqual(res.status_code, 200, res.get_data(as_text=True))
        self.assert_ml_only(res.get_json())
        self.assertAlmostEqual(res.get_json()["input"]["vitals"]["heart_rate_bpm"], 75, delta=2)

    def test_predict_bad_inputs(self):
        X, _ = random_feature_rows()
        partial = {k: v for k, v in dict(X.iloc[0]).items() if k != "eda_mean"}
        self.assertEqual(self.post("/body-vitals/predict", {"features": partial}).status_code, 400)
        self.assertEqual(self.post("/body-vitals/predict", {"nothing": 1}).status_code, 400)
        flat = [0.0] * (60 * 100)
        res = self.post("/body-vitals/predict", {"signals": {"ecg": flat, "eda": flat, "resp": flat, "fs": 100}})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.get_json()["kind"], "signal_quality")

    def test_calibrated_model_demo_and_predict(self):
        original = self.routes.engine
        self.routes.engine = BodyVitalsMLEngine(*write_tiny_model(tempfile.mkdtemp(dir=self.tmp), "wesad-chest-v2",
                                                                  "rest_calibration"))
        try:
            data = self.post("/body-vitals/demo", {"scenario": "stress", "seed": 1}).get_json()
            self.assert_ml_only(data)
            self.assertTrue(data["calibration"]["used"])
            self.assertTrue(data["calibration"]["is_simulated"])
            self.assertEqual(data["calibration"]["windows"], 9)            # (300 - 60) / 30 + 1
            self.assertEqual(data["model"]["normalization"], "rest_calibration")
            X, _ = random_feature_rows(names=config.FEATURE_SETS["wesad-chest-v2"])
            res = self.post("/body-vitals/predict", {"features": dict(X.iloc[0])})
            self.assertEqual(res.status_code, 400)
            self.assertEqual(res.get_json()["kind"], "calibration_required")
            res = self.post("/body-vitals/predict", {"features": dict(X.iloc[0]),
                                                     "calibration_features": [dict(X.iloc[1]), dict(X.iloc[2])]})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json()["calibration"]["windows"], 2)
            self.assertIsNone(res.get_json()["calibration"]["is_simulated"])
        finally:
            self.routes.engine = original

    def test_uncalibrated_model_reports_no_calibration(self):
        data = self.post("/body-vitals/demo", {"scenario": "baseline", "seed": 2}).get_json()
        self.assertFalse(data["calibration"]["used"])

    def test_model_unavailable_503(self):
        self.routes.engine = BodyVitalsMLEngine(os.path.join(self.tmp, "x.pkl"), os.path.join(self.tmp, "x.json"))
        try:
            res = self.post("/body-vitals/demo", {"scenario": "baseline"})
            self.assertEqual(res.status_code, 503)
            self.assertFalse(res.get_json()["success"])
            self.assertEqual(res.get_json()["kind"], "model_unavailable")
        finally:
            self.routes.engine = BodyVitalsMLEngine(*write_tiny_model(tempfile.mkdtemp(dir=self.tmp)))


if __name__ == "__main__":
    unittest.main()

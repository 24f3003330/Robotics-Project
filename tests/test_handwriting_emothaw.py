"""
Tests for the EMOTHAW handwriting pipeline:  python -m unittest discover tests

The real EMOTHAW data is licensed and not part of the repository, so these
tests build a small throw-away dataset in a temporary directory that has the
EMOTHAW FILE FORMAT (.svc recordings + a DASS label table). It only exercises
the code paths - import, loading, splitting, training, inference. It never
touches data/ or models/, and nothing measured on it is a model result.
"""

import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from handwriting import config  # noqa: E402
from handwriting import import_emothaw, train  # noqa: E402
from handwriting.dataset import check_no_leakage, load_emothaw, writer_split  # noqa: E402
from handwriting.inference import HandwritingAnalyzer, HandwritingModelUnavailable  # noqa: E402
from handwriting.preprocessing import HandwritingInputError, read_svc_rows  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: files in the EMOTHAW format
# ---------------------------------------------------------------------------
def svc_text(rng, n_strokes=6, pressure_max=1023):
    """A .svc recording: count line, then x y t pen azimuth altitude pressure."""
    rows, t = [], 1_000_000
    for s in range(n_strokes):
        for i in range(40):                                  # pen down
            a = i / 39 * math.pi
            rows.append([5000 + 300 * s + 100 * math.cos(a) + rng.normal(0, 2),
                         8000 + 150 * math.sin(a) + rng.normal(0, 2), t, 1,
                         int(rng.integers(0, 4096)), int(rng.integers(300, 900)),
                         int(rng.integers(100, pressure_max))])
            t += 8
        for i in range(10):                                  # in air (tablet records hover)
            rows.append([rows[-1][0] + 5, rows[-1][1], t, 0, 1000, 600, 0])
            t += 8
    return f"{len(rows)}\n" + "\n".join(" ".join(str(int(round(v))) for v in r) for r in rows) + "\n"


def make_raw_dataset(raw, n_writers=24, tasks=(3, 7), label_rows=None, seed=0):
    """Writers 1..n in the EMOTHAW layout; the label table holds DASS-42 scores."""
    rng = np.random.default_rng(seed)
    for w in range(1, n_writers + 1):
        folder = os.path.join(raw, "collection", f"user{w:05d}")
        os.makedirs(folder, exist_ok=True)
        for task in tasks:
            with open(os.path.join(folder, f"u{w:05d}s00001_hw{task:05d}.svc"), "w") as f:
                f.write(svc_text(rng))
    if label_rows is None:
        # cycle through all 8 anxiety/depression/stress combinations; scores sit
        # right at the DASS bounds (7/8, 9/10, 14/15) to test the cutoffs
        label_rows = [[f"u{w:05d}", 8 if w & 1 else 7, 10 if w & 2 else 9, 15 if w & 4 else 14]
                      for w in range(1, n_writers + 1)]
    with open(os.path.join(raw, "DASS_scores.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["Subject", "Depression", "Anxiety", "Stress"]] +
                                [[r[0], r[2], r[1], r[3]] for r in label_rows])


def dir_digest(path):
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(path)):
        for f in sorted(files):
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, path).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def run_import(raw, out, *extra):
    return quiet(import_emothaw.run, import_emothaw.parse_args(["--raw", raw, "--out", out, *extra]))


def pointer_points(kind="stylus", n_strokes=8, per_stroke=40, hover=True):
    """Browser payload as static/js/handwriting.js sends it for each pointer type."""
    pts, t = [], 1000.0
    for s in range(n_strokes):
        for i in range(per_stroke):
            a = i / (per_stroke - 1) * math.pi
            p = {"x": 40 + 30 * s + 12 * math.cos(a), "y": 100 - 25 * math.sin(a), "t": t, "pen_down": True}
            if kind == "mouse":
                p.update(pressure=None, altitude=None, azimuth=None)
            elif kind == "touch":
                p.update(pressure=0.3 + 0.2 * math.sin(a), altitude=None, azimuth=None)
            else:
                p.update(pressure=0.2 + 0.5 * math.sin(a), altitude=55 + 10 * math.cos(a), azimuth=120 + 30 * a)
            pts.append(p)
            t += 8
        pts.append(dict(pts[-1], pen_down=False, pressure=None if kind == "mouse" else 0, t=t))
        t += 8
        if hover and kind != "touch":
            for i in range(5):
                pts.append(dict(pts[-1], x=pts[-1]["x"] + 3, t=t, pen_down=False))
                t += 8
    return pts


# ---------------------------------------------------------------------------
class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.raw = os.path.join(self.tmp, "raw")
        self.out = os.path.join(self.tmp, "processed")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_import_converts_and_labels_by_dass_cutoffs(self):
        make_raw_dataset(self.raw, n_writers=8)
        before = dir_digest(self.raw)
        report = run_import(self.raw, self.out)
        self.assertEqual(dir_digest(self.raw), before, "raw data was modified")
        self.assertEqual(report["recordings"], 16)
        self.assertEqual(report["writers_with_recordings_and_labels"], 8)
        with open(os.path.join(self.out, "labels.csv")) as f:
            labels = {int(r["writer_id"]): r for r in csv.DictReader(f)}
        for w in range(1, 9):     # scores at the bounds: 7/9/14 normal, 8/10/15 elevated
            self.assertEqual(labels[w]["anxiety"], "elevated" if w & 1 else "normal")
            self.assertEqual(labels[w]["depression"], "elevated" if w & 2 else "normal")
            self.assertEqual(labels[w]["stress"], "elevated" if w & 4 else "normal")
        self.assertEqual(report["labels"]["encoding"], {"anxiety": "dass", "depression": "dass", "stress": "dass"})

    def test_standard_csv_keeps_every_sample_and_pen_channels(self):
        make_raw_dataset(self.raw, n_writers=1, tasks=(7,))
        run_import(self.raw, self.out)
        src = os.path.join(self.raw, "collection", "user00001", "u00001s00001_hw00007.svc")
        raw = read_svc_rows(src)
        with open(os.path.join(self.out, "samples", "u00001", "u00001s00001_hw00007.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(raw))
        self.assertEqual(list(rows[0]), ["x", "y", "t_ms", "pen_down", "pressure", "altitude", "azimuth", "stroke"])
        r0 = rows[0]
        self.assertAlmostEqual(float(r0["x"]), -raw[0, 1])            # 90 deg tablet rotation
        self.assertAlmostEqual(float(r0["y"]), -raw[0, 0])
        self.assertAlmostEqual(float(r0["pressure"]), raw[0, 6] / 1023, places=4)
        self.assertAlmostEqual(float(r0["altitude"]), raw[0, 5] * 90 / 1023, places=3)
        self.assertAlmostEqual(float(r0["azimuth"]), raw[0, 4] * 360 / 4095, places=3)
        self.assertEqual({r["pen_down"] for r in rows}, {"0", "1"})   # in-air samples kept
        self.assertEqual(len({r["stroke"] for r in rows if r["pen_down"] == "1"}), 6)

    def test_writer_comes_from_file_name_not_folder(self):
        make_raw_dataset(self.raw, n_writers=2, tasks=(7,))
        wrong = os.path.join(self.raw, "collection", "user00099")
        os.makedirs(wrong)
        shutil.move(os.path.join(self.raw, "collection", "user00002", "u00002s00001_hw00007.svc"), wrong)
        run_import(self.raw, self.out)
        with open(os.path.join(self.out, "manifest.csv")) as f:
            writers = sorted(int(r["writer_id"]) for r in csv.DictReader(f))
        self.assertEqual(writers, [1, 2])

    def test_bad_names_and_duplicates_are_skipped_not_invented(self):
        make_raw_dataset(self.raw, n_writers=2, tasks=(7,))
        folder = os.path.join(self.raw, "collection", "user00001")
        shutil.copy(os.path.join(folder, "u00001s00001_hw00007.svc"), os.path.join(self.raw, "u00001s00001_hw00007.svc"))
        shutil.copy(os.path.join(folder, "u00001s00001_hw00007.svc"), os.path.join(folder, "notes.svc"))
        report = run_import(self.raw, self.out)
        self.assertEqual(report["recordings"], 2)
        self.assertTrue(any("identical copy" in s for s in report["skipped"]))
        self.assertTrue(any("notes.svc" in s for s in report["skipped"]))

    def test_binary_only_column_is_not_guessed(self):
        make_raw_dataset(self.raw, n_writers=4, label_rows=[[f"u{w:05d}", w % 2, w % 2, w % 2] for w in range(1, 5)])
        with self.assertRaisesRegex(ValueError, "label-encoding"):
            run_import(self.raw, self.out)
        report = run_import(self.raw, self.out, "--label-encoding", "binary")
        self.assertEqual(report["writers_with_labels"], 4)

    def test_severity_words_and_incomplete_rows(self):
        make_raw_dataset(self.raw, n_writers=3, label_rows=[
            ["u00001", "normal", "Mild", "Extremely severe"], ["u00002", "Moderate", "normal", "normal"],
            ["u00003", "normal", "", "normal"]])
        report = run_import(self.raw, self.out)
        self.assertEqual(report["labels"]["writers_with_incomplete_labels"], [3])
        with open(os.path.join(self.out, "labels.csv")) as f:
            labels = {int(r["writer_id"]): r for r in csv.DictReader(f)}
        self.assertEqual(sorted(labels), [1, 2])
        self.assertEqual((labels[1]["anxiety"], labels[1]["depression"], labels[1]["stress"]),
                         ("normal", "elevated", "elevated"))

    def test_unrecognised_or_ambiguous_columns_stop_the_import(self):
        make_raw_dataset(self.raw, n_writers=2)
        path = os.path.join(self.raw, "DASS_scores.csv")
        with open(path, "w") as f:
            f.write("Subject,D,A,S\nu00001,10,8,15\nu00002,1,2,3\n")
        with self.assertRaisesRegex(ValueError, "--anxiety-col"):
            run_import(self.raw, self.out, "--labels", path)
        report = run_import(self.raw, self.out, "--labels", path, "--anxiety-col", "A",
                            "--depression-col", "D", "--stress-col", "S")
        self.assertEqual(report["writers_with_labels"], 2)
        with open(path, "w") as f:
            f.write("Subject,ID,Anxiety,Depression,Stress\nu00001,1,8,10,15\n")
        with self.assertRaisesRegex(ValueError, "several candidate"):
            run_import(self.raw, self.out, "--labels", path, "--force")

    def test_out_of_range_values_stop_the_import(self):
        make_raw_dataset(self.raw, n_writers=2, label_rows=[["u00001", 50, 1, 2], ["u00002", 3, 4, 5]])
        with self.assertRaisesRegex(ValueError, "DASS-42"):
            run_import(self.raw, self.out)
        shutil.rmtree(self.raw)
        make_raw_dataset(self.raw, n_writers=2)
        with open(os.path.join(self.raw, "collection", "user00001", "u00001s00001_hw00003.svc"), "w") as f:
            f.write(svc_text(np.random.default_rng(1), pressure_max=2047))
        with self.assertRaises(SystemExit):
            run_import(self.raw, self.out)

    def test_raw_and_out_must_be_separate(self):
        make_raw_dataset(self.raw, n_writers=2)
        with self.assertRaises(SystemExit):
            run_import(self.raw, os.path.join(self.raw, "processed"))


# ---------------------------------------------------------------------------
class SplitTests(unittest.TestCase):
    def test_writer_split_is_disjoint_and_complete(self):
        keys = {w: (w % 2, (w // 2) % 2, (w // 4) % 2) for w in range(1, 130)}
        for seed in range(5):
            s = writer_split(keys, seed=seed)
            a, b, c = map(set, (s["train"], s["val"], s["test"]))
            self.assertFalse(a & b or a & c or b & c)
            self.assertEqual(a | b | c, set(keys))
            self.assertTrue(0.6 < len(a) / len(keys) < 0.8)

    def test_leakage_check_catches_writer_file_and_content(self):
        base = {"train": [{"writer": 1, "path": "a", "sha256": "x"}], "val": [], "test": []}
        self.assertTrue(check_no_leakage(base))
        for leak in ({"writer": 1, "path": "b", "sha256": "y"}, {"writer": 2, "path": "a", "sha256": "y"},
                     {"writer": 2, "path": "b", "sha256": "x"}):
            with self.assertRaises(AssertionError):
                check_no_leakage(dict(base, test=[leak]))


# ---------------------------------------------------------------------------
class TrainAndInferenceTests(unittest.TestCase):
    """One small end-to-end run shared by the tests below."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        raw, cls.processed = os.path.join(cls.tmp, "raw"), os.path.join(cls.tmp, "processed")
        make_raw_dataset(raw, n_writers=24)
        run_import(raw, cls.processed)
        cls.ckpt_path = os.path.join(cls.tmp, "model.pt")
        cls.log = io.StringIO()
        with redirect_stdout(cls.log):
            cls.trainer = train.Trainer(train.parse_args([
                "--data-dir", cls.processed, "--out", cls.ckpt_path, "--results-dir", os.path.join(cls.tmp, "run"),
                "--epochs", "2", "--d-model", "16", "--ff", "32", "--layers", "1", "--heads", "2"]))
            cls.ckpt = cls.trainer.run()
        with open(os.path.join(cls.tmp, "run", "results.json")) as f:
            cls.results = json.load(f)
        cls.analyzer = HandwritingAnalyzer(cls.ckpt_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_dataset_loading_and_task_filter(self):
        samples, report = load_emothaw(self.processed, config.EMOTHAW_HEADS, [7])
        self.assertEqual(len(samples), 24)
        self.assertEqual({s["task"] for s in samples}, {7})
        self.assertEqual(report["writers_with_labels_and_recordings"], 24)
        self.assertTrue(all(set(s["y"]) == set(config.HEAD_NAMES) for s in samples))

    def test_split_is_writer_independent(self):
        sets = self.trainer.sets
        writers = {k: {s["writer"] for s in v} for k, v in sets.items()}
        self.assertFalse(writers["train"] & writers["test"] or writers["train"] & writers["val"]
                         or writers["val"] & writers["test"])
        self.assertEqual(self.results["split"], {k: sorted(v) for k, v in writers.items()})
        self.assertIn("leakage check passed", self.log.getvalue())

    def test_normalisation_uses_training_writers_only(self):
        from handwriting.preprocessing import feature_matrix
        names = self.ckpt["feature_names"]
        rows = np.concatenate([feature_matrix(s["traj"], names)[0] for s in self.trainer.sets["train"]])
        np.testing.assert_allclose(self.ckpt["norm"]["mean"], rows.mean(0), rtol=1e-4, atol=1e-5)
        everyone = np.concatenate([feature_matrix(s["traj"], names)[0]
                                   for k in ("train", "val", "test") for s in self.trainer.sets[k]])
        self.assertFalse(np.allclose(self.ckpt["norm"]["mean"], everyone.mean(0), rtol=1e-6, atol=1e-7))

    def test_training_reports_per_target_test_metrics(self):
        self.assertEqual([h["name"] for h in self.ckpt["heads"]], ["anxiety", "depression", "stress"])
        test = self.results["test_metrics"]
        for level in ("per_recording", "per_writer"):
            for n in config.HEAD_NAMES:
                m = test[level][n]
                for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1",
                            "confusion_matrix", "majority_baseline_accuracy"):
                    self.assertIn(key, m)
                self.assertEqual(m["positive_class"], "elevated")
                self.assertEqual(sum(map(sum, m["confusion_matrix"]["matrix"])), m["n"])
        n_test = self.results["split_counts"]["test"]
        self.assertEqual(test["per_recording"]["anxiety"]["n"], n_test["samples"])
        self.assertEqual(test["per_writer"]["anxiety"]["n"], n_test["writers"])
        log = self.log.getvalue()
        for text in ("participants", "Anxiety", "Depression", "Stress", "[SPLIT] train", "[SPLIT] test",
                     "held-out TEST set", "confusion"):
            self.assertIn(text, log)

    def check_result(self, r):
        self.assertTrue(r["ok"])
        self.assertEqual(list(r["predictions"]), ["Anxiety", "Depression", "Stress"])
        self.assertEqual(list(r["probabilities"]), ["Anxiety", "Depression", "Stress"])
        for t, label in r["predictions"].items():
            p = r["probabilities"][t]
            self.assertTrue(0.0 <= p <= 1.0)
            self.assertIn(label, ("Normal", "Elevated"))
            self.assertEqual(label == "Elevated", p > 0.5 or (p == 0.5 and label == "Elevated"))
        self.assertEqual(r["elevated"], [t for t, v in r["predictions"].items() if v == "Elevated"])

    def test_inference_for_mouse_touch_and_stylus(self):
        stylus = self.analyzer.analyze({"points": pointer_points("stylus")})
        self.check_result(stylus)
        self.assertEqual(stylus["input"]["defaulted_features"], [])
        mouse = self.analyzer.analyze({"points": pointer_points("mouse")})
        self.check_result(mouse)
        self.assertEqual(mouse["input"]["defaulted_features"], ["pressure", "altitude"])
        touch = self.analyzer.analyze({"points": pointer_points("touch")})
        self.check_result(touch)
        self.assertEqual(touch["input"]["defaulted_features"], ["altitude"])

    def test_missing_optional_attributes_are_accepted(self):
        pts = [{k: v for k, v in p.items() if k in ("x", "y", "t", "pen_down")} for p in pointer_points()]
        r = self.analyzer.analyze({"points": pts})
        self.check_result(r)
        self.assertEqual(r["input"]["defaulted_features"], ["pressure", "altitude"])

    def test_long_trajectory_uses_several_windows(self):
        r = self.analyzer.analyze({"points": pointer_points(n_strokes=200)})
        self.check_result(r)
        self.assertGreater(r["input"]["windows"], 1)

    def test_empty_and_invalid_input(self):
        cases = [{"points": []}, {"points": "abc"}, {}, [], {"points": [{"x": 1, "y": 2, "t": 0}]},
                 {"points": [{"x": "1", "y": 2, "t": 0, "pen_down": True}]},
                 {"points": [{"x": 1, "y": 2, "t": 0, "pen_down": 1}]},
                 {"points": pointer_points(n_strokes=1, per_stroke=5)},           # too short
                 {"points": [dict(p, pen_down=False) for p in pointer_points()]}]  # nothing written
        for payload in cases:
            with self.assertRaises(HandwritingInputError, msg=str(payload)[:80]):
                self.analyzer.analyze(payload)

    def test_non_emothaw_checkpoint_is_refused(self):
        import torch
        bad = dict(self.ckpt, heads=[{"name": "emotion", "classes": ["Calm", "Happy", "Sad"]}])
        path = os.path.join(self.tmp, "calm_happy_sad.pt")
        torch.save(bad, path)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(HandwritingModelUnavailable, "anxiety"):
            HandwritingAnalyzer(path).analyze({"points": pointer_points()})

    def test_missing_checkpoint_says_not_installed(self):
        a = HandwritingAnalyzer(os.path.join(self.tmp, "nope.pt"))
        with self.assertRaises(HandwritingModelUnavailable):
            a.analyze({"points": pointer_points()})
        self.assertEqual(a.status()["state"], "not_installed")


# ---------------------------------------------------------------------------
class FlaskRouteTests(unittest.TestCase):
    """The web routes, with the analyzer pointed at a missing checkpoint. Does not
    open /voice (it starts loading HuBERT) or start the camera."""

    @classmethod
    def setUpClass(cls):
        try:
            with redirect_stdout(io.StringIO()):
                import app as app_module
        except Exception as exc:  # e.g. a missing optional system dependency
            raise unittest.SkipTest(f"app.py could not be imported: {exc}")
        cls.app_module = app_module
        cls.client = app_module.app.test_client()

    def test_pages_render(self):
        for url in ("/", "/camera", "/handwriting"):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        page = self.client.get("/handwriting").get_data(as_text=True)
        for t in ("Anxiety", "Depression", "Stress"):
            self.assertIn(f'data-target="{t}"', page)
        self.assertNotIn("Calm", page)
        # the EMOTHAW panel never shows the separate 4-class prototype's labels
        emothaw_panel = page[page.index('id="hwEmothawPanel"'):page.index('id="hw4Panel"')]
        for gone in ("Happy", "Neutral", "Sad", "Angry"):
            self.assertNotIn(gone, emothaw_panel)
        landing = self.client.get("/").get_data(as_text=True)
        self.assertIn("Anxiety, Depression and Stress", landing)

    def test_handwriting_api_status_codes(self):
        original = self.app_module.handwriting_analyzer
        self.app_module.handwriting_analyzer = HandwritingAnalyzer("/nonexistent/model.pt")
        try:
            self.assertEqual(self.client.post("/handwriting/analyze", data="x").status_code, 400)
            self.assertEqual(self.client.post("/handwriting/analyze", json={"points": []}).status_code, 400)
            r = self.client.post("/handwriting/analyze", json={"points": pointer_points()})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.get_json()["kind"], "model_unavailable")
            self.assertEqual(self.client.get("/handwriting/status").get_json()["state"], "not_installed")
        finally:
            self.app_module.handwriting_analyzer = original

    def test_other_modalities_still_respond(self):
        self.assertEqual(self.client.get("/status").status_code, 200)
        self.assertEqual(self.client.get("/voice/status").status_code, 200)


if __name__ == "__main__":
    unittest.main()

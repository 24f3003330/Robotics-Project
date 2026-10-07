"""
Tests for the separate 4-class (Happy / Neutral / Sad / Angry) handwriting
pipeline:  python -m unittest discover tests

No graphology examples are part of the repository, so these tests render
throw-away text images and trajectories into a temporary directory and give
them arbitrary labels. They exercise the code paths only (reconstruction,
seed generation, splitting, training, inference, API); nothing here touches
data/ or models/, and nothing measured here says anything about emotions.
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

from handwriting import config, four_class, generate_seed_data, import_4class, train  # noqa: E402
from handwriting.image_trajectory import image_to_trajectory, load_ink_mask, thin  # noqa: E402
from handwriting.inference import HandwritingAnalyzer, HandwritingModelUnavailable  # noqa: E402
from handwriting.preprocessing import HandwritingInputError, Trajectory, feature_matrix, prepare  # noqa: E402

WORDS = ["hello there", "quiet day", "morning walk", "a small note", "see you soon", "blue river"]


def render_text(path, text, font=None, scale=2.2, thickness=6, lines=1, slant=0.0):
    import cv2
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX if font is None else font
    img = np.full((140 * lines + 40, 900), 255, np.uint8)
    for k in range(lines):
        cv2.putText(img, text, (20, 110 + 140 * k), font, scale, 0, thickness, cv2.LINE_AA)
    if slant:
        m = np.float32([[1, slant, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderValue=255)
    cv2.imwrite(path, img)


def browser_points(kind="stylus", n_strokes=8, per_stroke=40, hover=True, scale=1.0, speed=1.0):
    pts, t = [], 1000.0
    for s in range(n_strokes):
        for i in range(per_stroke):
            a = i / (per_stroke - 1) * math.pi
            p = {"x": scale * (40 + 30 * s + 12 * math.cos(a)), "y": scale * (100 - 25 * math.sin(a)),
                 "t": t, "pen_down": True}
            if kind == "mouse":
                p.update(pressure=None, altitude=None, azimuth=None)
            elif kind == "touch":
                p.update(pressure=0.3 + 0.2 * math.sin(a), altitude=None, azimuth=None)
            else:
                p.update(pressure=0.2 + 0.5 * math.sin(a), altitude=55 + 10 * math.cos(a), azimuth=120 + 30 * a)
            pts.append(p)
            t += 8 / speed
        pts.append(dict(pts[-1], pen_down=False, t=t))
        t += 8 / speed
        if hover and kind != "touch":
            for _ in range(5):
                pts.append(dict(pts[-1], x=pts[-1]["x"] + 3, t=t, pen_down=False))
                t += 8 / speed
    return pts


def make_sources(root, per_class=3, with_json=True, writer_ids=False):
    """sources/<label>/*.png (+ one .json trajectory per class). Labels are arbitrary fixtures."""
    import cv2
    fonts = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SIMPLEX]
    rows, n = [], 0
    for ci, label in enumerate(config.FOUR_CLASS_LABELS):
        folder = os.path.join(root, label.lower())
        os.makedirs(folder, exist_ok=True)
        for k in range(per_class):
            n += 1
            rel = f"{label.lower()}/ref_{k + 1:02d}.png"
            render_text(os.path.join(root, rel), WORDS[(ci + k) % len(WORDS)], font=fonts[k % 3],
                        slant=0.1 * ((n % 3) - 1))
            rows.append([rel, label, f"w{n}", "fixture"])
        if with_json:
            n += 1
            rel = f"{label.lower()}/traced_01.json"
            with open(os.path.join(root, rel), "w") as f:
                json.dump({"points": browser_points(n_strokes=6 + ci)}, f)
            rows.append([rel, label, f"w{n}", "fixture"])
    if writer_ids:
        with open(os.path.join(root, "sources.csv"), "w", newline="") as f:
            csv.writer(f).writerows([["file", "label", "writer_id", "reference"]] + rows)


def dir_digest(path):
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(path)):
        for f in sorted(files):
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, path).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def seed(sources, out, *extra):
    with redirect_stdout(io.StringIO()):
        return generate_seed_data.run(generate_seed_data.parse_args(["--sources", sources, "--out", out, *extra]))


def read_manifest(out):
    with open(os.path.join(out, "manifest.csv")) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
class ImageReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_trajectory_follows_the_ink_and_invents_no_pen_data(self):
        path = os.path.join(self.tmp, "a.png")
        render_text(path, "hello world")
        traj, info = image_to_trajectory(path)
        ink, _ = load_ink_mask(path)
        down = traj.pen > 0
        r, c = np.round(traj.y[down]).astype(int), np.round(traj.x[down]).astype(int)
        self.assertTrue(ink[r, c].all(), "reconstructed points must lie on the ink")
        skel = thin(ink)
        covered = np.zeros_like(skel)
        covered[r, c] = True
        self.assertGreater(covered[skel].mean(), 0.9, "most of the skeleton must be traced")
        self.assertTrue(np.isnan(traj.pressure).all() and np.isnan(traj.altitude).all()
                        and np.isnan(traj.azimuth).all())
        self.assertTrue(np.all(np.diff(traj.t) > 0))
        self.assertEqual(info["t_meaning"], "sequence index (not time)")
        for missing in ("timestamps / timing", "pressure", "pen altitude", "pen azimuth"):
            self.assertIn(missing, info["unavailable"])
        self.assertIn("stroke order", info["reconstructed"])

    def test_reading_order_is_line_by_line(self):
        path = os.path.join(self.tmp, "two.png")
        render_text(path, "one line", lines=2)
        traj, _ = image_to_trajectory(path)
        down = traj.pen > 0
        line = (traj.y[down] > 140).astype(int)          # 0 = first line, 1 = second
        self.assertTrue(np.all(np.diff(line) >= 0), "the second line must come after the first")

    def test_blank_image_is_rejected(self):
        import cv2
        path = os.path.join(self.tmp, "blank.png")
        cv2.imwrite(path, np.full((100, 100), 255, np.uint8))
        with self.assertRaises(ValueError):
            image_to_trajectory(path)


# ---------------------------------------------------------------------------
class ArcLengthTests(unittest.TestCase):
    def test_geometric_features_ignore_size_and_speed(self):
        from handwriting.preprocessing import from_browser_points
        names = config.FEATURE_SETS["geometric"]
        a = feature_matrix(prepare(from_browser_points(browser_points()), "collapse", "arclength"), names)[0]
        b = feature_matrix(prepare(from_browser_points(browser_points(scale=3.0, speed=2.5)),
                                   "collapse", "arclength"), names)[0]
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_time_resampling_is_still_the_default(self):
        traj = Trajectory(x=np.arange(1000.0), y=np.sin(np.arange(1000) / 20), t=np.arange(1000) * 0.004,
                          pen=np.ones(1000))
        self.assertEqual(len(prepare(traj)), len(prepare(traj, "keep", "time")))


# ---------------------------------------------------------------------------
class SeedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "sources")
        make_sources(self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_metadata_marks_seed_data_as_unvalidated(self):
        before = dir_digest(self.src)
        out = os.path.join(self.tmp, "processed")
        meta = seed(self.src, out, "--augment", "3")
        self.assertEqual(dir_digest(self.src), before, "sources were modified")
        self.assertEqual((meta["dataset_type"], meta["source_type"], meta["scientifically_validated"]),
                         ("seed", "graphology_reference", False))
        self.assertIn("not scientifically validated", meta["statement"])
        self.assertEqual(meta["sources"], 16)
        rows = read_manifest(out)
        self.assertEqual(len(rows), 16 * 4)
        for r in rows:
            self.assertEqual((r["dataset_type"], r["source_type"], r["validated"]),
                             ("seed", "graphology_reference", "false"))
            self.assertEqual(r["synthetic"], "0" if r["sample_id"].endswith("__orig") else "1")
            self.assertTrue(r["source_sample"] and r["source_sha256"] and r["label"] in config.FOUR_CLASS_LABELS)
        images = [r for r in rows if r["input_kind"] == "image"]
        self.assertTrue(all(r["trajectory_origin"] == "reconstructed_from_image" and r["t_is_time"] == "0"
                            for r in images))
        with open(os.path.join(out, images[0]["file"])) as f:
            first = next(csv.DictReader(f))
        self.assertEqual((first["pressure"], first["altitude"], first["azimuth"]), ("", "", ""))
        traced = [r for r in rows if r["input_kind"] == "trajectory"]
        self.assertTrue(all(r["trajectory_origin"] == "recorded" for r in traced))

    def test_generation_is_reproducible(self):
        a, b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        seed(self.src, a, "--augment", "2")
        seed(self.src, b, "--augment", "2")
        self.assertEqual(dir_digest(os.path.join(a, "samples")), dir_digest(os.path.join(b, "samples")))

    def test_augmentation_does_not_depend_on_the_label(self):
        # the same source file under two different labels gets identical augmented copies
        other = os.path.join(self.tmp, "relabelled")
        for label in config.FOUR_CLASS_LABELS:
            os.makedirs(os.path.join(other, label.lower()))
        shutil.copy(os.path.join(self.src, "happy", "ref_01.png"), os.path.join(other, "sad", "x.png"))
        a, b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        seed(self.src, a, "--augment", "2")
        seed(other, b, "--augment", "2")
        for k in ("aug01", "aug02"):
            with open(os.path.join(a, "samples", "happy", f"happy_001__{k}.csv"), "rb") as f1, \
                    open(os.path.join(b, "samples", "sad", f"sad_001__{k}.csv"), "rb") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_augmentation_parameters_share_one_distribution(self):
        out = os.path.join(self.tmp, "p")
        seed(self.src, out, "--augment", "20")
        by_class = {c: [] for c in config.FOUR_CLASS_LABELS}
        for r in read_manifest(out):
            if r["augmentation"]:
                by_class[r["label"]].append(json.loads(r["augmentation"]))
        for key, (lo, hi) in four_class.AUGMENT_RANGES.items():
            if key == "time_scale":
                continue
            for c, params in by_class.items():
                vals = [p[key] for p in params]
                self.assertTrue(all(lo <= v <= hi for v in vals), (key, c))

    def test_unknown_labels_and_duplicates_are_skipped(self):
        os.makedirs(os.path.join(self.src, "calm"))
        shutil.copy(os.path.join(self.src, "happy", "ref_01.png"), os.path.join(self.src, "calm", "c.png"))
        shutil.copy(os.path.join(self.src, "happy", "ref_01.png"), os.path.join(self.src, "sad", "dup.png"))
        meta = seed(self.src, os.path.join(self.tmp, "o"), "--augment", "0")
        self.assertEqual(meta["sources"], 16)
        self.assertTrue(any("calm" in p and "not one of" in p for p in meta["skipped"]))
        self.assertTrue(any("dup.png" in p and "identical" in p for p in meta["skipped"]))

    def test_sources_and_output_must_be_separate(self):
        with self.assertRaises(SystemExit):
            seed(self.src, os.path.join(self.src, "processed"))


# ---------------------------------------------------------------------------
class RealDatasetImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.src = os.path.join(self.tmp, "sources")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_import(self, *extra):
        with redirect_stdout(io.StringIO()):
            return import_4class.run(import_4class.parse_args(
                ["--sources", self.src, "--out", os.path.join(self.tmp, "out"), "--source", "fixture", *extra]))

    def test_writer_ids_are_required(self):
        make_sources(self.src, per_class=0, with_json=True, writer_ids=False)
        with self.assertRaisesRegex(ValueError, "writer_id"):
            self.run_import()

    def test_label_map_and_metadata(self):
        os.makedirs(self.src)
        rows = [["file", "label", "writer_id"]]
        for i, lab in enumerate(["joy", "neutral", "sadness", "anger", "fear"]):
            with open(os.path.join(self.src, f"s{i}.json"), "w") as f:
                json.dump({"points": browser_points(n_strokes=6 + i)}, f)      # distinct files
            rows.append([f"s{i}.json", lab, f"p{i}"])
        with open(os.path.join(self.src, "sources.csv"), "w", newline="") as f:
            csv.writer(f).writerows(rows)
        meta = self.run_import("--label-map", "joy=Happy,neutral=Neutral,sadness=Sad,anger=Angry",
                               "--labels-verified")
        self.assertEqual((meta["dataset_type"], meta["source_type"], meta["scientifically_validated"]),
                         ("real", "fixture", True))
        self.assertEqual(meta["sources"], 4)
        self.assertTrue(any("fear" in p for p in meta["skipped"]))      # unmapped: skipped, not guessed
        self.assertEqual(meta["generator"]["augment_per_source"], 0)

    def test_images_need_explicit_opt_in(self):
        make_sources(self.src, per_class=1, with_json=False, writer_ids=True)
        with self.assertRaises(SystemExit):                     # nothing usable without --allow-images
            self.run_import()
        meta = self.run_import("--allow-images", "--force")
        self.assertEqual(meta["reconstructed_from_images"]["sources"], 4)


# ---------------------------------------------------------------------------
class SeedTrainingAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        src, cls.processed = os.path.join(cls.tmp, "sources"), os.path.join(cls.tmp, "processed")
        make_sources(src, per_class=4)
        seed(src, cls.processed, "--augment", "3")
        cls.ckpt_path = os.path.join(cls.tmp, "4class.pt")
        cls.log = io.StringIO()
        with redirect_stdout(cls.log):
            cls.trainer = train.Trainer(train.parse_args([
                "--dataset", "seed_4class", "--data-dir", cls.processed, "--out", cls.ckpt_path,
                "--results-dir", os.path.join(cls.tmp, "run"), "--epochs", "2",
                "--d-model", "16", "--ff", "32", "--layers", "1", "--heads", "2"]))
            cls.ckpt = cls.trainer.run()
        with open(os.path.join(cls.tmp, "run", "results.json")) as f:
            cls.results = json.load(f)
        cls.analyzer = HandwritingAnalyzer(cls.ckpt_path, kind="4class")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_checkpoint_is_marked_as_seed_prototype(self):
        self.assertEqual(self.ckpt["heads"], config.FOUR_CLASS_HEADS)
        self.assertEqual(self.ckpt["feature_names"], config.GEOMETRIC_FEATURES)
        self.assertEqual((self.ckpt["preprocessing"]["resample"], self.ckpt["preprocessing"]["air"]),
                         ("arclength", "collapse"))
        tr = self.ckpt["training"]
        self.assertEqual((tr["dataset_key"], tr["dataset_type"], tr["scientifically_validated"]),
                         ("seed_4class", "seed", False))
        self.assertEqual(tr["notice"], config.PROTOTYPE_NOTICE)
        log = self.log.getvalue()
        self.assertIn("PROTOTYPE / SEED MODEL", log)
        self.assertIn("NOT emotion-recognition accuracy", log)

    def test_augmented_copies_stay_with_their_source_and_are_not_evaluated(self):
        where = {}
        for name, samples in self.trainer.sets.items():
            for s in samples:
                self.assertEqual(where.setdefault(s["source"], name), name)
        for name in ("val", "test"):
            self.assertTrue(self.trainer.sets[name])
            self.assertFalse(any(s["synthetic"] for s in self.trainer.sets[name]))
        self.assertTrue(any(s["synthetic"] for s in self.trainer.sets["train"]))
        self.assertIn("leakage check passed", self.log.getvalue())

    def test_normalisation_uses_training_groups_only(self):
        rows = np.concatenate([feature_matrix(s["traj"], self.ckpt["feature_names"])[0]
                               for s in self.trainer.sets["train"]])
        np.testing.assert_allclose(self.ckpt["norm"]["mean"], rows.mean(0), rtol=1e-4, atol=1e-5)

    def test_metrics_cover_four_classes(self):
        m = self.results["test_metrics"]["per_recording"]["emotion"]
        self.assertEqual(m["confusion_matrix"]["rows_true_cols_pred"], config.FOUR_CLASS_LABELS)
        for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"):
            self.assertIn(key, m)
        self.assertEqual(self.results["evaluated_on"],
                         "original samples only (augmented copies used for training only)")

    def check(self, r):
        self.assertTrue(r["ok"])
        self.assertEqual(list(r["probabilities"]), config.FOUR_CLASS_LABELS)
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=3)
        self.assertEqual(r["emotion"], max(r["probabilities"], key=r["probabilities"].get))
        self.assertAlmostEqual(r["confidence"], r["probabilities"][r["emotion"]], places=4)
        self.assertTrue(r["prototype"])
        self.assertEqual(r["notice"], config.PROTOTYPE_NOTICE)

    def test_inference_for_mouse_touch_and_stylus(self):
        for kind in ("mouse", "touch", "stylus"):
            r = self.analyzer.analyze({"points": browser_points(kind)})
            self.check(r)
            self.assertEqual(r["input"]["defaulted_features"], [])   # geometric: no pen channels needed

    def test_long_and_invalid_input(self):
        self.check(self.analyzer.analyze({"points": browser_points(n_strokes=300)}))
        for payload in ({"points": []}, {}, {"points": [{"x": 1, "y": 2, "t": 0}]},
                        {"points": browser_points(n_strokes=1, per_stroke=5)}):
            with self.assertRaises(HandwritingInputError):
                self.analyzer.analyze(payload)

    def test_each_analyzer_refuses_the_other_experiments_checkpoint(self):
        import torch
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(HandwritingModelUnavailable, "anxiety"):
                HandwritingAnalyzer(self.ckpt_path, kind="emothaw").analyze({"points": browser_points()})
            emothaw_like = dict(self.ckpt, heads=config.EMOTHAW_HEADS)
            path = os.path.join(self.tmp, "emothaw_like.pt")
            torch.save(emothaw_like, path)
            with self.assertRaisesRegex(HandwritingModelUnavailable, "4-class"):
                HandwritingAnalyzer(path, kind="4class").analyze({"points": browser_points()})


# ---------------------------------------------------------------------------
class FourClassRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            with redirect_stdout(io.StringIO()):
                import app as app_module
        except Exception as exc:
            raise unittest.SkipTest(f"app.py could not be imported: {exc}")
        cls.app_module = app_module
        cls.client = app_module.app.test_client()

    def test_routes_without_a_model(self):
        original = self.app_module.handwriting_4class_analyzer
        self.app_module.handwriting_4class_analyzer = HandwritingAnalyzer("/nonexistent/4.pt", kind="4class")
        try:
            self.assertEqual(self.client.get("/handwriting/4class/status").get_json()["state"], "not_installed")
            self.assertEqual(self.client.post("/handwriting/4class/analyze", json={"points": []}).status_code, 400)
            r = self.client.post("/handwriting/4class/analyze", json={"points": browser_points()})
            self.assertEqual(r.status_code, 503)
            self.assertIn("generate_seed_data", r.get_json()["error"])
        finally:
            self.app_module.handwriting_4class_analyzer = original

    def test_page_offers_both_models_with_the_prototype_notice(self):
        page = self.client.get("/handwriting").get_data(as_text=True)
        self.assertIn('data-mode="emothaw"', page)
        self.assertIn('data-mode="4class"', page)
        self.assertIn(config.PROTOTYPE_NOTICE, page)
        for c in config.FOUR_CLASS_LABELS:
            self.assertIn(f'data-class="{c}"', page)


if __name__ == "__main__":
    unittest.main()

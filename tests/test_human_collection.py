"""
Tests for participant canvas collection (Collect tab) and its use in the
4-class seed pipeline:  python -m unittest discover tests

Everything runs in temporary folders; the "participants" are generated
trajectories that only exercise the code paths (saving, metadata, writer-grouped
split, training) - nothing here is data or a result.
"""

import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from handwriting import config, generate_seed_data, train  # noqa: E402
from handwriting import human_collection as hc  # noqa: E402
from handwriting.preprocessing import HandwritingInputError  # noqa: E402


def canvas_points(kind="mouse", n_strokes=6, wobble=0.0):
    pts, t = [], 1000.0
    for s in range(n_strokes):
        for i in range(30):
            a = i / 29 * math.pi * 2
            p = {"x": 40 + 45 * s + 10 * math.cos(a) + wobble * i, "y": 100 - (20 + wobble) * math.sin(a),
                 "t": t, "pen_down": True}
            if kind == "pen":
                p.update(pressure=0.4 + 0.2 * math.sin(a), altitude=60.0, azimuth=30.0)
            elif kind == "mouse":
                p.update(pressure=None, altitude=None, azimuth=None)
            pts.append(p)
            t += 10
        pts.append(dict(pts[-1], pen_down=False, t=t))
        t += 40
    return pts


class SaveSampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.human = os.path.join(self.tmp, "seed_4class", "human")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def save(self, **kw):
        payload = dict({"writer_id": "writer_001", "label": "Happy", "points": canvas_points()}, **kw)
        return hc.save_sample(payload, root=self.human)

    def test_saved_record_format(self):
        r = self.save(points=canvas_points("pen"), pointer_types=["pen", "bogus"])
        self.assertEqual((r["label"], r["number"], r["file"]), ("Happy", 1, "happy/writer_001_happy_001.json"))
        with open(os.path.join(self.human, r["file"])) as f:
            d = json.load(f)
        for k, v in (("writer_id", "writer_001"), ("label", "Happy"), ("source_type", "human_canvas"),
                     ("scientifically_validated", False), ("pointer_types", ["pen"])):
            self.assertEqual(d[k], v)
        self.assertTrue(d["timestamp"].endswith("Z"))
        self.assertEqual(set(d["points"][0]), {"x", "y", "t", "pen_down", "pressure", "altitude", "azimuth"})
        mouse = self.save(label="Sad")
        with open(os.path.join(self.human, mouse["file"])) as f:
            self.assertEqual(set(json.load(f)["points"][0]), {"x", "y", "t", "pen_down"})   # nulls dropped

    def test_numbering_counts_and_no_overwrite(self):
        for _ in range(3):
            last = self.save()
        self.assertEqual(last["number"], 3)
        self.save(writer_id="writer_002", label="Angry")
        c = hc.counts(self.human, "writer_001")
        self.assertEqual(c["totals"], {"Happy": 3, "Neutral": 0, "Sad": 0, "Angry": 1})
        self.assertEqual(c["writer"], {"Happy": 3, "Neutral": 0, "Sad": 0, "Angry": 0})
        self.assertEqual((c["writers"], c["samples"]), (2, 4))
        self.assertEqual(hc.next_writer_id(self.human), "writer_003")
        self.assertEqual(len(os.listdir(os.path.join(self.human, "happy"))), 3)

    def test_invalid_input(self):
        for bad in ({"writer_id": "Alice Smith"}, {"writer_id": "writer_1"}, {"writer_id": ""},
                    {"label": "Calm"}, {"label": None}, {"points": []},
                    {"points": canvas_points(n_strokes=1)[:5]}):
            with self.assertRaises(HandwritingInputError, msg=str(bad)[:60]):
                self.save(**bad)
        self.assertFalse(os.path.exists(self.human) and any(os.scandir(self.human)))


class HumanPipelineTests(unittest.TestCase):
    """10 generated 'writers' x 4 labels x 2 samples -> seed build -> human-only training."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        base = os.path.join(cls.tmp, "seed_4class")
        cls.human, cls.sources = os.path.join(base, "human"), os.path.join(base, "sources")
        os.makedirs(os.path.join(cls.sources, "angry"))
        with open(os.path.join(cls.sources, "angry", "graph.json"), "w") as f:     # one "graphology" source
            json.dump({"points": canvas_points(n_strokes=9)}, f)
        for w in range(1, 11):
            for li, label in enumerate(config.FOUR_CLASS_LABELS):
                for k in range(2):
                    hc.save_sample({"writer_id": f"writer_{w:03d}", "label": label,
                                    "points": canvas_points("pen" if w % 2 else "mouse", 5 + li, 0.3 * w + k)},
                                   root=cls.human)
        cls.out = os.path.join(base, "processed")
        with redirect_stdout(io.StringIO()):
            cls.meta = generate_seed_data.run(generate_seed_data.parse_args(
                ["--sources", cls.sources, "--out", cls.out, "--augment", "2"]))
        with open(os.path.join(cls.out, "manifest.csv")) as f:
            cls.manifest = list(csv.DictReader(f))
        cls.log = io.StringIO()
        with redirect_stdout(cls.log):
            cls.trainer = train.Trainer(train.parse_args([
                "--dataset", "seed_4class", "--data-dir", cls.out, "--sample-source", "human_canvas",
                "--out", os.path.join(cls.tmp, "m.pt"), "--results-dir", os.path.join(cls.tmp, "run"),
                "--epochs", "2", "--d-model", "16", "--ff", "32", "--layers", "1", "--heads", "2"]))
            cls.ckpt = cls.trainer.run()
        with open(os.path.join(cls.tmp, "run", "results.json")) as f:
            cls.results = json.load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_origins_are_kept_apart_in_metadata(self):
        origins = {(r["sample_source"], r["synthetic"]) for r in self.manifest}
        self.assertEqual(origins, {("graphology", "0"), ("graphology", "1"), ("human_canvas", "0"),
                                   ("human_canvas", "1")})
        human = [r for r in self.manifest if r["sample_source"] == "human_canvas"]
        self.assertTrue(all(r["source_type"] == "human_canvas" and r["validated"] == "false"
                            and r["writer_id"].startswith("writer_") and r["group_id"] == "writer:" + r["writer_id"]
                            and r["timing_available"] == "true" for r in human))
        self.assertEqual(self.meta["per_sample_source"]["human_canvas"],
                         {"Happy": 20, "Neutral": 20, "Sad": 20, "Angry": 20})
        self.assertEqual(self.meta["writers_per_sample_source"]["human_canvas"], 10)

    def test_writer_independent_split(self):
        sets = self.trainer.sets
        writers = {k: {s["writer"] for s in v} for k, v in sets.items()}
        self.assertFalse(writers["train"] & writers["val"] or writers["train"] & writers["test"]
                         or writers["val"] & writers["test"])
        self.assertEqual(sum(len(w) for w in writers.values()), 10)
        self.assertEqual({len(writers["val"]), len(writers["test"])} <= {1, 2}, True)
        self.assertTrue(all(s["sample_source"] == "human_canvas" for v in sets.values() for s in v))
        for name in ("val", "test"):
            self.assertFalse(any(s["synthetic"] for s in sets[name]))

    def test_metrics_and_notice(self):
        m = self.results["test_metrics"]["per_recording"]["emotion"]
        for k in ("accuracy", "balanced_accuracy", "macro_f1", "per_class", "confusion_matrix"):
            self.assertIn(k, m)
        self.assertEqual(set(m["per_class"]), set(config.FOUR_CLASS_LABELS))
        self.assertEqual(self.results["notice"], config.HUMAN_PROTOTYPE_NOTICE)
        self.assertEqual(self.results["sample_sources"], {"human_canvas": 80})
        self.assertIn("leakage check passed", self.log.getvalue())


class CollectRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            with redirect_stdout(io.StringIO()):
                import app as app_module
        except Exception as exc:
            raise unittest.SkipTest(f"app.py could not be imported: {exc}")
        cls.tmp = tempfile.mkdtemp()
        cls.saved_defaults = (hc.save_sample.__defaults__, hc.counts.__defaults__, hc.next_writer_id.__defaults__)
        hc.save_sample.__defaults__ = (cls.tmp,)
        hc.counts.__defaults__ = (cls.tmp, None)
        hc.next_writer_id.__defaults__ = (cls.tmp,)
        cls.client = app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        hc.save_sample.__defaults__, hc.counts.__defaults__, hc.next_writer_id.__defaults__ = cls.saved_defaults
        shutil.rmtree(cls.tmp)

    def test_collect_routes(self):
        s = self.client.get("/handwriting/4class/collect/status").get_json()
        self.assertEqual((s["next_writer_id"], s["samples"]), ("writer_001", 0))
        r = self.client.post("/handwriting/4class/collect",
                             json={"writer_id": "writer_001", "label": "Sad", "points": canvas_points()})
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.get_json()["label"], r.get_json()["number"]), ("Sad", 1))
        bad = self.client.post("/handwriting/4class/collect",
                               json={"writer_id": "John", "label": "Sad", "points": canvas_points()})
        self.assertEqual(bad.status_code, 400)
        s = self.client.get("/handwriting/4class/collect/status?writer_id=writer_001").get_json()
        self.assertEqual((s["writer"]["Sad"], s["next_writer_id"]), (1, "writer_002"))
        self.assertEqual(self.client.get("/handwriting/4class/collect/status?writer_id=bob").status_code, 400)

    def test_page_has_collect_tab_without_writing_instructions(self):
        page = self.client.get("/handwriting").get_data(as_text=True)
        self.assertIn('data-mode="collect"', page)
        self.assertIn("Write naturally for this sample.", page)
        for c in config.FOUR_CLASS_LABELS:
            self.assertIn(f'data-label="{c}"', page)
        panel = page[page.index('id="hwCollectPanel"'):page.index('id="hwDefaulted"')].lower()
        for hint in ("faster", "slower", "larger", "smaller", "pressure", "slant"):
            self.assertNotIn(hint, panel)


if __name__ == "__main__":
    unittest.main()

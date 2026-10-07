"""
Tests for analysing an UPLOADED handwriting image with the 4-class prototype
(POST /handwriting/4class/analyze-image):  python -m unittest discover tests

A tiny throw-away checkpoint is trained on generated fixtures in a temporary
folder; models/ is never touched and nothing here is a result.
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

from handwriting import config, generate_seed_data, train  # noqa: E402
from handwriting.image_trajectory import image_bytes_to_trajectory, image_to_trajectory  # noqa: E402
from handwriting.inference import HandwritingAnalyzer, MAX_UPLOAD_BYTES  # noqa: E402
from handwriting.preprocessing import HandwritingInputError  # noqa: E402
from test_handwriting_4class import browser_points, make_sources  # noqa: E402


def encode(ext, text="good morning to you", size=(1000, 260), photo=False):
    import cv2
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    cv2.putText(img, text, (30, size[1] // 2 + 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.4, (20, 20, 20), 6,
                cv2.LINE_AA)
    if photo:                                                  # saturated, dark, no paper
        img = np.clip(np.random.default_rng(0).normal(110, 60, img.shape), 0, 255).astype(np.uint8)
        img[:, :, 2] = 230
    ok, buf = cv2.imencode(ext, img)
    return buf.tobytes()


class UploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        src, processed = os.path.join(cls.tmp, "sources"), os.path.join(cls.tmp, "processed")
        make_sources(src, per_class=3)
        cls.ckpt = os.path.join(cls.tmp, "4class.pt")
        with redirect_stdout(io.StringIO()):
            generate_seed_data.run(generate_seed_data.parse_args(["--sources", src, "--out", processed,
                                                                  "--augment", "1"]))
            cls.checkpoint = train.Trainer(train.parse_args([
                "--dataset", "seed_4class", "--data-dir", processed, "--out", cls.ckpt,
                "--results-dir", os.path.join(cls.tmp, "run"), "--epochs", "1",
                "--d-model", "16", "--ff", "32", "--layers", "1", "--heads", "2"])).run()
        cls.analyzer = HandwritingAnalyzer(cls.ckpt, kind="4class")
        try:
            with redirect_stdout(io.StringIO()):
                import app as app_module
        except Exception:
            app_module = None
        cls.app_module = app_module
        if app_module is not None:
            cls.original = app_module.handwriting_4class_analyzer
            app_module.handwriting_4class_analyzer = cls.analyzer
            cls.client = app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        if cls.app_module is not None:
            cls.app_module.handwriting_4class_analyzer = cls.original
        shutil.rmtree(cls.tmp)

    def post(self, data, name="h.png", field="image"):
        if self.app_module is None:
            self.skipTest("app.py could not be imported")
        files = {} if data is None else {field: (io.BytesIO(data), name)}
        with redirect_stdout(io.StringIO()):
            return self.client.post("/handwriting/4class/analyze-image", data=files,
                                    content_type="multipart/form-data")

    def check_prediction(self, r):
        self.assertTrue(r["ok"])
        self.assertEqual(set(r["probabilities"]), set(config.FOUR_CLASS_LABELS))   # JSON key order is not meaningful
        self.assertAlmostEqual(sum(r["probabilities"].values()), 1.0, places=3)
        self.assertEqual(r["emotion"], max(r["probabilities"], key=r["probabilities"].get))
        self.assertTrue(r["prototype"])
        self.assertEqual(r["notice"], config.PROTOTYPE_NOTICE)
        self.assertFalse(r["model"]["scientifically_validated"])

    # ------------------------------------------------------------ conversion
    def test_bytes_and_file_give_the_same_trajectory(self):
        data = encode(".png")
        path = os.path.join(self.tmp, "same.png")
        with open(path, "wb") as f:
            f.write(data)
        a, _ = image_to_trajectory(path)
        b, info = image_bytes_to_trajectory(data)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.pen, b.pen)
        self.assertTrue(np.isnan(b.pressure).all() and np.isnan(b.altitude).all() and np.isnan(b.azimuth).all())
        self.assertEqual(info["t_meaning"], "sequence index (not time)")

    # ------------------------------------------------------------ inference + API
    def test_png_jpg_webp_uploads(self):
        for ext in (".png", ".jpg", ".webp"):
            resp = self.post(encode(ext), name="sample" + ext)
            self.assertEqual(resp.status_code, 200, (ext, resp.get_json()))
            r = resp.get_json()
            self.check_prediction(r)
            i = r["input"]
            self.assertEqual((i["source_type"], i["trajectory_source"], i["timing_available"],
                              i["pressure_available"], i["stroke_order_available"]),
                             ("uploaded_image", "reconstructed_from_image", False, False, "estimated"))
            self.assertEqual(i["image_size"], [1000, 260])
            self.assertGreater(i["strokes"], 3)
            self.assertIsNone(i["duration_s"])

    def test_invalid_uploads_are_refused_clearly(self):
        import cv2
        blank = cv2.imencode(".png", np.full((200, 400, 3), 255, np.uint8))[1].tobytes()
        cases = {
            "missing": (None, "No image"),
            "empty": (b"", "empty"),
            "not an image": (b"hello, this is text", "Unsupported"),
            "gif": (b"GIF89a" + b"\x00" * 100, "Unsupported"),
            "corrupt png": (encode(".png")[:200], "could not be read"),
            "blank page": (blank, "No handwriting"),
            "photo": (encode(".jpg", photo=True), "doesn't look like"),
            "too large": (b"\x89PNG\r\n\x1a\n" + b"\x00" * (MAX_UPLOAD_BYTES + 10), "too large"),
        }
        for name, (data, words) in cases.items():
            resp = self.post(data)
            self.assertIn(resp.status_code, (400, 413), name)
            body = resp.get_json()
            self.assertFalse(body["ok"], name)
            self.assertIn(words.lower(), body["error"].lower(), name)
        resp = self.post(encode(".png"), field="file")                     # wrong field name
        self.assertEqual(resp.status_code, 400)

    def test_canvas_inference_still_works(self):
        r = self.analyzer.analyze({"points": browser_points("mouse")})
        self.check_prediction(r)
        resp = self.client.post("/handwriting/4class/analyze", json={"points": browser_points("stylus")})
        self.assertEqual(resp.status_code, 200)
        self.check_prediction(resp.get_json())

    def test_image_is_never_sent_to_emothaw(self):
        with self.assertRaisesRegex(HandwritingInputError, "only available for the 4-class"):
            HandwritingAnalyzer(self.ckpt, kind="emothaw").analyze_image(encode(".png"))
        if self.app_module is not None:
            self.assertEqual(self.client.post("/handwriting/analyze-image").status_code, 404)

    def test_timing_models_refuse_images(self):
        import torch
        ck = dict(self.checkpoint)
        names = list(ck["feature_names"])
        names[names.index("turn")] = "log_dt"                            # same width, needs timing
        path = os.path.join(self.tmp, "timing.pt")
        torch.save(dict(ck, feature_names=names), path)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(HandwritingInputError, "writing-speed"):
            HandwritingAnalyzer(path, kind="4class").analyze_image(encode(".png"))

    def test_page_offers_upload_on_the_4class_tab(self):
        if self.app_module is None:
            self.skipTest("app.py could not be imported")
        page = self.client.get("/handwriting").get_data(as_text=True)
        self.assertIn("Upload Handwriting Image", page)
        self.assertIn('accept=".png,.jpg,.jpeg,.webp', page)
        self.assertIn("Remove Image", page)
        self.assertIn('id="hwCanvas"', page)                             # canvas still there
        bar = page[page.index('id="hwUploadBar"'):]
        self.assertFalse(bar[:bar.index(">")].rstrip().endswith("hidden"))   # visible without switching tabs
        self.assertIn("window.HW_INITIAL_MODE", page)

    def test_page_opens_on_a_tab_that_can_predict(self):
        if self.app_module is None:
            self.skipTest("app.py could not be imported")
        import handwriting.config as hc_config
        emothaw = os.path.isfile(hc_config.DEFAULT_CHECKPOINT)
        four = os.path.isfile(hc_config.FOUR_CLASS_CHECKPOINT)
        page = self.client.get("/handwriting").get_data(as_text=True)
        want = "4class" if four and not emothaw else "emothaw"
        self.assertIn(f'window.HW_INITIAL_MODE = "{want}"', page)


if __name__ == "__main__":
    unittest.main()

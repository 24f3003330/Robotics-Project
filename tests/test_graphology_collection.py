"""
Tests for graphology candidate collection and human review:
python -m unittest discover tests

A fake in-memory website stands in for the internet (no network access). The
page texts and images are fixtures made up for the tests; nothing here is
data, and nothing here is trained on.
"""

import csv
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from handwriting import collect_graphology as cg  # noqa: E402
from handwriting import generate_seed_data  # noqa: E402
from handwriting import review_graphology as rg  # noqa: E402


def png(text, scale=2.0, size=(900, 200), ext=".png", resize=1.0, font=None):
    import cv2
    img = np.full((size[1], size[0]), 255, np.uint8)
    cv2.putText(img, text, (20, size[1] // 2 + 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX if font is None else font,
                scale, 0, 6, cv2.LINE_AA)
    if resize != 1.0:
        img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
    return cv2.imencode(ext, img)[1].tobytes()


PAGE = """<html><head><title>Graphology &amp; emotions - examples</title></head><body>
<h1>Handwriting and mood</h1>
<p>This page is about happy, sad and angry handwriting.</p>
<figure><img src="/img/a.png"><figcaption>Handwriting of a happy, optimistic writer</figcaption></figure>
<p><img src="/img/b.png" alt="Example of angry, aggressive handwriting"></p>
<img src="/img/c.png" alt="">
<img src="/img/d.png" alt="This is not a happy writer">
<img src="/img/e.png" alt="happy and sad handwriting compared">
<img src="/img/dup.png" alt="A cheerful writer">
<img src="/img/near.jpg" alt="Joyful handwriting sample">
<img src="/img/bad.png" alt="sad handwriting">
<img src="/img/tiny.png" alt="sad handwriting">
<img src="/img/forbidden.png" alt="gloomy handwriting">
<div class="wp-caption"><img src="/img/f.png"><p class="wp-caption-text">A melancholy, downcast slant</p></div>
<h2>Calm writing</h2><img src="/img/g.png">
<img src="/img/icon.png" width="16" height="16" alt="happy icon">
<a rel="license" href="https://creativecommons.org/licenses/by/4.0/">CC BY</a>
</body></html>"""


class FakeWeb:
    """url -> (status, body, content-type). Records every request."""

    def __init__(self):
        self.requests = []
        a = png("hello there")
        self.routes = {
            "https://ex.org/robots.txt": (200, b"User-agent: *\nDisallow: /private/\n", "text/plain"),
            "https://ex.org/graphology.html": (200, PAGE.encode(), "text/html; charset=utf-8"),
            "https://ex.org/private/secret.html": (200, PAGE.encode(), "text/html"),
            "https://ex.org/captcha.html": (200, b"<html><div class='g-recaptcha'></div></html>", "text/html"),
            "https://ex.org/members.html": (403, b"", "text/html"),
            "https://other.org/robots.txt": (403, b"", "text/plain"),
            "https://other.org/page.html": (200, PAGE.encode(), "text/html"),
            "https://ex.org/img/a.png": (200, a, "image/png"),
            "https://ex.org/img/b.png": (200, png("quiet day"), "image/png"),
            "https://ex.org/img/c.png": (200, png("no label"), "image/png"),
            "https://ex.org/img/d.png": (200, png("negated"), "image/png"),
            "https://ex.org/img/e.png": (200, png("two labels"), "image/png"),
            "https://ex.org/img/dup.png": (200, a, "image/png"),
            "https://ex.org/img/near.jpg": (200, png("hello there", ext=".jpg", resize=0.7), "image/jpeg"),
            "https://ex.org/img/bad.png": (200, b"<html>not an image</html>", "image/png"),
            "https://ex.org/img/tiny.png": (200, png("x", size=(40, 20), scale=0.5), "image/png"),
            "https://ex.org/img/forbidden.png": (403, b"", "text/html"),
            "https://ex.org/img/f.png": (200, png("morning walk"), "image/png"),
            "https://ex.org/img/g.png": (200, png("see you soon"), "image/png"),
        }

    def get(self, url):
        self.requests.append(url)
        status, body, ctype = self.routes.get(url, (404, b"", "text/html"))
        return cg.Response(status, body, {"content-type": ctype})


def client(web):
    return cg.PoliteClient(web, delay=0.0, sleep=lambda s: None)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = cg.paths(os.path.join(self.tmp, "seed_4class"))
        os.makedirs(self.p["root"])
        self.web = FakeWeb()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def collect(self, urls=("https://ex.org/graphology.html",), **kw):
        with redirect_stdout(io.StringIO()):
            return cg.collect(self.p, client(self.web), [{"url": u, "license_or_access_note": ""} for u in urls],
                              **kw)

    def rows(self):
        return cg.read_csv(self.p["candidates_csv"])

    def by_url(self, suffix):
        return next(r for r in self.rows() if r["image_url"].endswith(suffix))


# ---------------------------------------------------------------------------
class CollectionTests(Base):
    def test_only_explicit_image_level_associations_become_candidates(self):
        s = self.collect()
        labels = {r["image_url"].rsplit("/", 1)[1]: r["candidate_label"] for r in self.rows()}
        self.assertEqual(labels, {"a.png": "Happy", "b.png": "Angry", "f.png": "Sad"})
        self.assertEqual(s["new_candidates"], 3)
        with open(self.p["log_csv"]) as f:
            log = {r["image_url"].rsplit("/", 1)[1]: r for r in csv.DictReader(f)}
        self.assertEqual(log["c.png"]["outcome"], "not_collected")          # page text never labels an image
        self.assertEqual(log["d.png"]["outcome"], "not_collected")          # negated
        self.assertIn("ambiguous", log["e.png"]["detail"])
        self.assertEqual(log["g.png"]["outcome"], "not_collected")          # heading only: not by default
        self.assertNotIn("icon.png", log)                                   # declared 16x16: ignored

    def test_section_headings_only_with_opt_in(self):
        self.collect(allow_section_headings=True)
        self.assertEqual(self.by_url("g.png")["candidate_label"], "Neutral")
        self.assertEqual(self.by_url("g.png")["evidence_source"], "section_heading")

    def test_metadata_and_source_urls(self):
        self.collect()
        r = self.by_url("a.png")
        self.assertEqual(r["source_page"], "https://ex.org/graphology.html")
        self.assertEqual(r["image_url"], "https://ex.org/img/a.png")
        self.assertEqual(r["page_title"], "Graphology & emotions - examples")
        self.assertEqual((r["status"], r["scientifically_validated"]), ("candidate", "false"))
        self.assertEqual(r["evidence_source"], "figcaption")
        self.assertIn("happy, optimistic", r["label_evidence"])
        self.assertIn("creativecommons.org/licenses/by/4.0", r["license_or_access_note"])
        self.assertEqual((r["trajectory_source"], r["timing_available"], r["pressure_available"],
                          r["stroke_order_available"]), ("reconstructed_from_image", "false", "false", "estimated"))
        for field in ("id", "filename", "candidate_label", "description", "sha256", "phash"):
            self.assertTrue(r[field], field)
        path = os.path.join(self.p["root"], r["filename"])
        self.assertTrue(r["filename"].startswith("candidates/happy/"))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), self.web.routes["https://ex.org/img/a.png"][1])   # original bytes
        self.assertEqual(self.by_url("f.png")["evidence_source"], "caption")

    def test_duplicates_exact_and_perceptual(self):
        self.collect()
        with open(self.p["log_csv"]) as f:
            log = {r["image_url"].rsplit("/", 1)[1]: r for r in csv.DictReader(f)}
        self.assertEqual(log["dup.png"]["outcome"], "duplicate")
        self.assertIn("identical", log["dup.png"]["detail"])
        self.assertEqual(log["near.jpg"]["outcome"], "duplicate")               # re-encoded + resized copy
        self.assertIn("same picture", log["near.jpg"]["detail"])
        before = len(self.rows())
        s = self.collect()                                                     # re-run: nothing new
        self.assertEqual((s["new_candidates"], len(self.rows())), (0, before))

    def test_different_samples_are_not_duplicates(self):
        import cv2
        fps = [cg.fingerprint(png(t, font=f)) for t in ("hello there", "hello thera", "thank you")
               for f in (cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX)]
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                self.assertFalse(cg.is_duplicate(fps[i], fps[j])[0], (i, j))

    def test_invalid_images_are_logged_not_saved(self):
        s = self.collect()
        with open(self.p["log_csv"]) as f:
            log = {r["image_url"].rsplit("/", 1)[1]: r for r in csv.DictReader(f)}
        self.assertEqual(log["bad.png"]["outcome"], "invalid_image")
        self.assertEqual(log["tiny.png"]["outcome"], "invalid_image")
        self.assertIn("too small", log["tiny.png"]["detail"])
        self.assertEqual(s["invalid"], 2)
        saved = [f for d, _, fs in os.walk(self.p["candidates"]) for f in fs]
        self.assertEqual(len(saved), 3)

    def test_restrictions_are_recorded_never_bypassed(self):
        s = self.collect(urls=("https://ex.org/private/secret.html", "https://ex.org/captcha.html",
                               "https://ex.org/members.html", "https://other.org/page.html",
                               "https://ex.org/graphology.html"))
        self.assertNotIn("https://ex.org/private/secret.html", self.web.requests)   # robots.txt disallows
        self.assertNotIn("https://other.org/page.html", self.web.requests)          # robots.txt 403
        with open(self.p["manual_csv"]) as f:
            manual = {r["url"]: r["reason"] for r in csv.DictReader(f)}
        self.assertIn("robots.txt", manual["https://ex.org/private/secret.html"])
        self.assertIn("CAPTCHA", manual["https://ex.org/captcha.html"])
        self.assertIn("403", manual["https://ex.org/members.html"])
        self.assertIn("disallow-all", manual["https://other.org/page.html"])
        self.assertIn("403", manual["https://ex.org/img/forbidden.png"])
        self.assertEqual(s["pages_blocked"], 4)
        self.assertEqual(self.web.requests.count("https://ex.org/robots.txt"), 1)

    def test_rate_limit_waits_between_requests_to_one_host(self):
        waits, t = [], [0.0]
        c = cg.PoliteClient(self.web, delay=2.0, sleep=lambda s: (waits.append(s), t.__setitem__(0, t[0] + s)),
                            clock=lambda: t[0])
        with redirect_stdout(io.StringIO()):
            cg.collect(self.p, c, [{"url": "https://ex.org/graphology.html"}])
        self.assertTrue(waits and all(abs(w - 2.0) < 1e-9 for w in waits))

    def test_host_answering_429_is_left_alone(self):
        self.web.routes["https://ex.org/img/a.png"] = (429, b"", "text/html")
        s = self.collect()
        after = self.web.requests[self.web.requests.index("https://ex.org/img/a.png") + 1:]
        self.assertFalse([u for u in after if u.startswith("https://ex.org/")])
        self.assertEqual(s["new_candidates"], 0)

    def test_dry_run_saves_nothing(self):
        s = self.collect(dry_run=True)
        self.assertEqual(s["new_candidates"], 3)
        self.assertFalse(os.path.exists(self.p["candidates_csv"]))


# ---------------------------------------------------------------------------
PLACEHOLDER = "data:image/gif,GIF89a%01%00%01%00%80%00%00%00%00%00%FF%FF%FF%21%F9%04%01%00%00%00%00%2C%00%00%00%00%01%00%01%00%00%02%01D%00%3B"


def photo(color=(40, 120, 200)):
    """A colourful 'stock photo' fixture: saturated, no paper background."""
    import cv2
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = color
    cv2.circle(img, (200, 150), 90, (20, 200, 60), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def gif(text):
    import cv2
    img = np.full((200, 900, 3), 255, np.uint8)             # this OpenCV writes GIF from 3 channels only
    cv2.putText(img, text, (20, 120), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
    ok, buf = cv2.imencode(".gif", img)
    if not ok:
        raise unittest.SkipTest("this OpenCV build cannot write GIF")
    return buf.tobytes()


class CollectorFixTests(Base):
    def test_lazy_loaded_urls(self):
        a = cg.real_image_url
        self.assertEqual(a({"src": PLACEHOLDER, "data-layzr": "https://x.org/a.jpg"}), "https://x.org/a.jpg")
        self.assertEqual(a({"src": "/img/blank.gif", "data-src": "/img/real.png"}), "/img/real.png")
        self.assertEqual(a({"src": PLACEHOLDER, "data-original": "/o.png"}), "/o.png")
        self.assertEqual(a({"src": PLACEHOLDER, "data-lazy-src": "/l.png"}), "/l.png")
        self.assertEqual(a({"src": PLACEHOLDER, "data-srcset": "/s-300.png 300w, /s-900.png 900w"}), "/s-900.png")
        self.assertEqual(a({"src": "/small.png", "srcset": "/small.png 1x, /big.png 2x"}), "/big.png")
        self.assertEqual(a({"src": "/plain.png"}), "/plain.png")
        self.assertEqual(a({"src": PLACEHOLDER}), "")                       # placeholder only
        self.assertEqual(a({"src": "/wp-content/uploads/site-logo.png"}), "")
        self.assertEqual(a({"data-src": "/images/icons/x.png", "src": PLACEHOLDER}), "")

    def test_lazy_loaded_page_is_collected(self):
        self.web.routes["https://ex.org/lazy.html"] = (200, f"""<html><title>Lazy</title><body>
            <img src="{PLACEHOLDER}" data-layzr="/img/lazy1.png" alt="Anger in handwriting">
            <img src="{PLACEHOLDER}" alt="happy writer">
            <img src="/img/tracker.gif" width="1" height="1" alt="angry">
            </body></html>""".encode(), "text/html")
        self.web.routes["https://ex.org/img/lazy1.png"] = (200, png("a lazy one"), "image/png")
        s = self.collect(urls=("https://ex.org/lazy.html",))
        self.assertEqual(s["new_candidates"], 1)
        self.assertEqual(self.rows()[0]["image_url"], "https://ex.org/img/lazy1.png")
        self.assertNotIn("https://ex.org/img/tracker.gif", self.web.requests)

    def test_multilingual_and_unicode_labels(self):
        cases = {"¿Cómo escribe la ALEGRÍA?": "Happy", "alegria": "Happy", "écriture en colère": "Angry",
                 "scrittura della tristezza": "Sad", "Sehr glücklich": "Happy", "öfkeli yazı": "Angry",
                 "nötr el yazısı": "Neutral", "Ärger in der Schrift": "Angry", "traurige": None,
                 "écriture neutre": "Neutral", "1alegria": "Happy", "angry_sample": "Angry",
                 "escritura sin alegría": None, "pas de joie": None, "mirage": None, "unhappy": "Sad"}
        for text, want in cases.items():
            self.assertEqual(cg.decide_label({"alt": text})[0], want, text)
        # decomposed (NFD) input is normalised too, and evidence keeps the original spelling
        nfd = "tristeza".replace("e", "e") + " y escritura ALEGRI\u0301A"
        self.assertEqual(cg.find_labels("ALEGRI\u0301A")["Happy"][0], "ALEGRÍA")
        self.assertEqual(cg.decide_label({"alt": nfd})[0], None)           # Sad + Happy: ambiguous

    def test_heuristic_trait_mappings_are_kept(self):
        for text, want in (("Aggression in handwriting", "Angry"), ("a hostile writer", "Angry"),
                           ("optimistic baseline", "Happy"), ("pessimistic writer", "Sad")):
            self.assertEqual(cg.decide_label({"caption": text})[0], want, text)

    def test_invisible_recaptcha_does_not_block_a_public_article(self):
        article = " ".join(["This article explains handwriting."] * 60)
        page = f"""<html><head><title>Article</title><style>.grecaptcha-badge{{visibility:hidden}}</style>
            <script src="https://www.google.com/recaptcha/api.js?render=KEY"></script></head>
            <body><p>{article}</p><form><div class="g-recaptcha" data-size="invisible"></div></form>
            <!-- Memberships powered by Paid Memberships Pro --></body></html>"""
        self.assertIsNone(cg.access_block(page))

    def test_real_access_controls_still_block(self):
        article = " ".join(["Some text."] * 200)
        blocked = {
            "widget": "<html><body><div class='g-recaptcha' data-sitekey='x'></div><p>Please continue.</p></body></html>",
            "phrase": f"<html><body><h1>Verify you are human</h1><p>{article}</p></body></html>",
            "login": f"<html><body><p>You must be logged in to view this content.</p><p>{article}</p></body></html>",
            "paywall": f"<html><body><p>Subscribe to continue reading.</p><p>{article}</p></body></html>",
            "challenge": "<html><head><title>Just a moment...</title></head><body></body></html>",
            "members": f"<html><body><p>This content is for members only.</p><p>{article}</p></body></html>"}
        for name, page in blocked.items():
            self.assertIsNotNone(cg.access_block(page), name)
        self.web.routes["https://ex.org/verify.html"] = (200, blocked["phrase"].encode(), "text/html")
        s = self.collect(urls=("https://ex.org/verify.html",))
        self.assertEqual(s["pages_blocked"], 1)
        with open(self.p["manual_csv"]) as f:
            self.assertIn("verify you are human", next(csv.DictReader(f))["reason"])

    def test_stock_photos_are_not_collected(self):
        self.web.routes["https://ex.org/photos.html"] = (200, b"""<html><body>
            <img src="/img/stock.jpg" alt="Angry man">
            <img src="/img/sample.png" alt="Angry handwriting"></body></html>""", "text/html")
        self.web.routes["https://ex.org/img/stock.jpg"] = (200, photo(), "image/jpeg")
        self.web.routes["https://ex.org/img/sample.png"] = (200, png("an angry note"), "image/png")
        s = self.collect(urls=("https://ex.org/photos.html",))
        self.assertEqual((s["new_candidates"], s["not_handwriting"], s["downloaded"]), (1, 1, 2))
        self.assertTrue(self.rows()[0]["image_url"].endswith("sample.png"))
        with open(self.p["log_csv"]) as f:
            log = {r["image_url"].rsplit("/", 1)[1]: r for r in csv.DictReader(f)}
        self.assertEqual(log["stock.jpg"]["outcome"], "not_handwriting_like")
        self.assertIn("saturation", log["stock.jpg"]["detail"])
        s = self.collect(urls=("https://ex.org/photos.html",), handwriting_filter=False)   # opt-out keeps it
        self.assertEqual(s["new_candidates"], 1)

    def test_gif_samples(self):
        data = gif("a gif sample")
        self.web.routes["https://ex.org/g.html"] = (200, b'<html><body><img src="/img/s.gif" alt="Angry handwriting"></body></html>',
                                                    "text/html")
        self.web.routes["https://ex.org/img/s.gif"] = (200, data, "image/gif")
        self.collect(urls=("https://ex.org/g.html",))
        r = self.rows()[0]
        self.assertTrue(r["filename"].endswith(".gif"))
        with open(os.path.join(self.p["root"], r["filename"]), "rb") as f:
            self.assertEqual(f.read(), data)                                  # original bytes kept
        from handwriting.image_trajectory import image_to_trajectory
        traj, _ = image_to_trajectory(os.path.join(self.p["root"], r["filename"]))
        self.assertGreater(len(traj), 20)                                    # usable by the seed pipeline


# ---------------------------------------------------------------------------
class ReviewTests(Base):
    def setUp(self):
        super().setUp()
        self.collect()
        self.ids = {r["image_url"].rsplit("/", 1)[1]: r["id"] for r in self.rows()}

    def test_approve_moves_file_and_keeps_provenance(self):
        rows = self.rows()
        rg.approve(self.p, rows, self.ids["a.png"])
        r = self.by_url("a.png")
        self.assertEqual((r["status"], r["scientifically_validated"]), ("approved", "false"))
        self.assertTrue(r["approved_file"].startswith("sources/happy/"))
        self.assertTrue(os.path.isfile(os.path.join(self.p["root"], r["approved_file"])))
        self.assertFalse(os.listdir(os.path.join(self.p["candidates"], "happy")))
        src = {x["file"]: x for x in cg.read_csv(os.path.join(self.p["sources"], "sources.csv"))}
        row = src[r["approved_file"][len("sources/"):]]
        self.assertEqual((row["label"], row["reference"], row["source_page"], row["candidate_id"]),
                         ("Happy", "https://ex.org/img/a.png", "https://ex.org/graphology.html", r["id"]))

    def test_reject_and_relabel(self):
        rows = self.rows()
        rg.reject(self.p, rows, self.ids["b.png"], "not handwriting")
        rb = self.by_url("b.png")
        self.assertEqual(rb["status"], "rejected")
        self.assertTrue(rb["filename"].startswith("rejected/angry/"))
        self.assertIn("not handwriting", rb["review_note"])
        rg.relabel(self.p, rows, self.ids["f.png"], "Neutral")
        rf = self.by_url("f.png")
        self.assertEqual((rf["candidate_label"], rf["original_candidate_label"], rf["status"]),
                         ("Neutral", "Sad", "candidate"))
        self.assertTrue(rf["filename"].startswith("candidates/neutral/"))
        with self.assertRaises(rg.ReviewError):
            rg.approve(self.p, rows, self.ids["b.png"])                  # rejected stays rejected
        with self.assertRaises(rg.ReviewError):
            rg.relabel(self.p, rows, self.ids["a.png"], "Calm")          # only the four classes

    def test_interactive_keys(self):
        order = [r["id"] for r in self.rows()]                           # a (Happy), b (Angry), f (Sad)
        keys = iter(["s", "n", "5", "r", "4", "y"])
        out = []
        rows = rg.interactive(self.p, ask=lambda prompt: next(keys), say=out.append)
        status = {r["id"]: (r["status"], r["candidate_label"]) for r in rows}
        self.assertEqual(status[order[0]], ("candidate", "Happy"))       # skipped
        self.assertEqual(status[order[1]], ("rejected", "Angry"))
        self.assertEqual(status[order[2]], ("approved", "Angry"))        # relabelled Sad -> Angry, approved
        self.assertIn("rejected: not handwriting", self.by_url("b.png")["review_note"])
        text = "\n".join(out)
        self.assertIn("not scientifically validated", text)
        self.assertIn("Angry:      0 candidates |   1 approved |   1 rejected", text)

    def test_contact_sheet(self):
        paths = rg.contact_sheets(self.p, self.rows())
        self.assertEqual(len(paths), 3)
        import cv2
        self.assertIsNotNone(cv2.imread(paths[0]))

    def test_candidates_cannot_enter_training(self):
        rows = self.rows()
        rg.approve(self.p, rows, self.ids["a.png"])
        rg.reject(self.p, rows, self.ids["b.png"], "irrelevant")
        # someone copies a pending and a rejected image into sources/ by hand
        for key, label in (("f.png", "sad"), ("b.png", "angry")):
            r = self.by_url(key)
            os.makedirs(os.path.join(self.p["sources"], label), exist_ok=True)
            shutil.copy(os.path.join(self.p["root"], r["filename"]), os.path.join(self.p["sources"], label, "copy.png"))
        out = os.path.join(self.tmp, "processed")
        with redirect_stdout(io.StringIO()):
            meta = generate_seed_data.run(generate_seed_data.parse_args(
                ["--sources", self.p["sources"], "--out", out, "--augment", "0"]))
        with open(os.path.join(out, "manifest.csv")) as f:
            manifest = list(csv.DictReader(f))
        self.assertEqual(len(manifest), 1)                               # only the approved image
        self.assertEqual(manifest[0]["reference"], "https://ex.org/img/a.png")
        self.assertEqual((manifest[0]["timing_available"], manifest[0]["pressure_available"],
                          manifest[0]["stroke_order_available"]), ("false", "false", "estimated"))
        self.assertEqual(sum("not approved" in s for s in meta["skipped"]), 2)
        for folder in ("candidates", "rejected"):
            with self.assertRaises(SystemExit):
                with redirect_stdout(io.StringIO()):
                    generate_seed_data.run(generate_seed_data.parse_args(
                        ["--sources", os.path.join(self.p["root"], folder), "--out", out + folder]))

    def test_auto_approve_prototype_workflow(self):
        done = rg.auto_approve_pending(self.p)
        self.assertEqual(len(done), 3)
        for r in self.rows():
            self.assertEqual((r["status"], r["scientifically_validated"], r["prototype_only"]),
                             ("auto_approved", "false", "true"))
            self.assertTrue(os.path.isfile(os.path.join(self.p["root"], r["approved_file"])))
        src = cg.read_csv(os.path.join(self.p["sources"], "sources.csv"))
        self.assertEqual({x["status"] for x in src}, {"auto_approved"})
        self.assertTrue(all(x["reference"].startswith("https://ex.org/img/") for x in src))
        self.assertEqual(sum(v["approved"] for v in rg.stats(self.rows()).values()), 3)   # auto_approved counts
        out = os.path.join(self.tmp, "processed")
        with redirect_stdout(io.StringIO()):
            meta = generate_seed_data.run(generate_seed_data.parse_args(
                ["--sources", self.p["sources"], "--out", out, "--augment", "0"]))
        self.assertEqual(meta["sources"], 3)                             # auto_approved images are usable
        self.assertEqual(meta["scientifically_validated"], False)
        # rejected / pending items are still refused by the seed generator
        self.assertEqual(rg.auto_approve_pending(self.p), [])

    def test_stats_format(self):
        text = rg.format_stats(self.rows())
        self.assertIn("Happy:      1 candidates |   0 approved |   0 rejected", text)
        self.assertEqual(len(text.splitlines()), 4)


if __name__ == "__main__":
    unittest.main()

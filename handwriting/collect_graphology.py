"""
handwriting/collect_graphology.py - collect CANDIDATE graphology reference images
for the 4-class seed dataset (Happy / Neutral / Sad / Angry).

    python -m handwriting.collect_graphology                      # pages from graphology_pages.csv
    python -m handwriting.collect_graphology --dry-run            # report what would be collected
    python -m handwriting.review_graphology                       # then: human review (approve / reject)
    python -m handwriting.generate_seed_data                      # then: seed data from APPROVED sources only

These samples are graphology-based reference/seed examples and are not
scientifically validated emotion labels. They are used only to establish and
test the prototype pipeline and are intended to be replaced by a properly
labelled research dataset.

What it does
  * reads ONLY the pages listed in data/handwriting/seed_4class/graphology_pages.csv
    (columns: url, license_or_access_note, notes) - it does not crawl links or use
    search engines
  * respects robots.txt (and its Crawl-delay), waits >= --delay seconds between
    requests to the same host, identifies itself with a descriptive User-Agent
  * never bypasses restrictions: pages or images answering 401 / 403 / 429 / 451,
    pages whose VISIBLE content is a CAPTCHA / login / paywall / bot challenge, and
    robots-disallowed URLs are written to manual_review.csv instead (invisible
    reCAPTCHA that only protects a site's forms does not block a public article)
  * an image becomes a candidate ONLY when its own context - alt text, title,
    <figcaption> or caption element - explicitly names exactly one emotion (see
    LABEL_TERMS). Page text never labels an image; the nearest section heading
    counts only with --allow-section-headings. Negated mentions ("not happy") are
    ignored; mentions of two classes make the image ambiguous (logged, not collected)
  * image URLs are read past lazy-loading placeholders (data-layzr, data-lazy-src,
    data-src, data-original, srcsets); logos / icons / tracking pixels are skipped
  * label terms are matched in English, Spanish, French, Italian, German and Turkish,
    accent-insensitively (Unicode NFKD)
  * images that do not look like ink on paper (stock photos, cartoons) are refused
    (handwriting_likeness; --no-handwriting-filter to disable)
  * candidates are saved unchanged (original bytes) to candidates/<label>/ and listed
    in candidates.csv with status=candidate, scientifically_validated=false
  * duplicates of any candidate, approved source or rejected image are not collected
    again: identical bytes (sha256), or the same picture re-encoded / resized /
    re-cropped (perceptual hash of the ink region as a prefilter, confirmed by
    normalised correlation >= NCC_DUPLICATE - see is_duplicate)
  * every image seen is logged with its outcome in collection_log.csv

Nothing here labels data as ground truth and nothing here starts training.
"""

import argparse
import base64
import csv
import datetime
import hashlib
import html
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser

import numpy as np

from . import config

SEED_DIR = config.FOUR_CLASS_DATASETS["seed_4class"]["dir"]
USER_AGENT = ("Robotics-Project-graphology-collector/1.0 (research prototype; reads only listed pages; "
              "respects robots.txt)")

DISCLAIMER = ("These samples are graphology-based reference/seed examples and are not scientifically "
              "validated emotion labels. They are used only to establish and test the prototype pipeline "
              "and are intended to be replaced by a properly labelled research dataset.")

CANDIDATE_FIELDS = [
    "id", "filename", "candidate_label", "original_candidate_label", "label_evidence", "evidence_source",
    "source_page", "image_url", "page_title", "description", "status", "scientifically_validated",
    "license_or_access_note", "sha256", "phash", "width", "height", "trajectory_source",
    "timing_available", "pressure_available", "stroke_order_available",
    "collected_at", "reviewed_at", "review_note", "approved_file", "prototype_only"]
LOG_FIELDS = ["time", "source_page", "image_url", "outcome", "detail"]
MANUAL_FIELDS = ["time", "url", "kind", "reason", "source_page"]

# Terms that, in an image's OWN context, associate it with a class. They include
# graphology's heuristic trait mappings (e.g. aggression -> Angry, optimistic ->
# Happy): these are candidate labels for the temporary SEED dataset only, never
# scientifically validated ground truth (scientifically_validated stays false).
# Clinical terms (e.g. "depression") are NOT mapped to Sad - the EMOTHAW
# experiment owns those constructs. Matching is case-insensitive and accent-
# insensitive (Unicode NFKD, combining marks removed): "alegría" == "alegria".
LABEL_TERMS = {
    "Happy": [
        # English
        "happy", "happiness", "joy", "joyful", "joyous", "cheerful", "cheerfulness", "optimistic",
        "optimism", "elated", "contented",
        # Spanish / French / Italian / German / Turkish
        "alegría", "alegre", "feliz", "felicidad",
        "joie", "joyeux", "joyeuse", "heureux", "heureuse",
        "felicità", "felice", "gioia",
        "glücklich", "freude", "fröhlich",
        "mutlu", "mutluluk"],
    "Neutral": [
        "neutral", "calm", "calmness", "emotionally balanced", "even-tempered",
        "emotionally stable", "emotional stability",
        "neutre", "neutro", "nötr"],
    "Sad": [
        "sad", "sadness", "sorrow", "sorrowful", "melancholy", "melancholic", "gloomy", "gloom",
        "unhappy", "unhappiness", "pessimistic", "pessimism", "downcast", "grief",
        "triste", "tristes", "tristeza", "tristesse", "tristezza",
        "traurig", "traurigkeit",
        "üzgün", "üzüntü"],
    "Angry": [
        "angry", "anger", "rage", "furious", "fury", "aggressive", "aggression", "aggressiveness", "hostile",
        "hostility", "irritable", "irritability",
        "temper",               # graphology trait "Temper" = tendency to become angry quickly

        "ira", "enfado", "enojo", "rabia",
        "colère",
        "rabbia",
        "wütend", "wut", "ärger",
        "öfkeli", "öfke"],
}
NEGATIONS = {"not", "no", "never", "without", "lack", "lacks", "lacking", "absence", "absent", "isn't",
             "aren't", "wasn't", "non",
             "sin", "nunca", "pas", "sans", "jamais", "senza", "mai", "nicht", "kein", "keine", "ohne",
             "degil", "yok"}
IMAGE_LEVEL_SOURCES = ("figcaption", "caption", "alt", "title")
# Duplicate detection. Handwriting images are mostly white paper, so a hash of
# the whole image collides for different samples; both measures below use only
# the ink's bounding box. Measured on rendered text: re-encoded / resized /
# re-margined copies correlate >= 0.988, different samples (even the same words
# in a near-identical font) <= 0.963.
PHASH_PREFILTER = 64                          # of 256 bits: farther apart = certainly different
NCC_DUPLICATE = 0.975
MIN_WIDTH, MIN_HEIGHT = 120, 40               # smaller images are icons / decorations
MAX_PIXELS = 40_000_000
# Access control is judged on what a READER sees, not on raw HTML: many public
# WordPress pages load invisible reCAPTCHA v3 (or hide its badge with CSS) only to
# protect comment / contact forms. A page is refused when its VISIBLE text says
# access is gated, when it is a bot-challenge interstitial, or when a visible
# CAPTCHA widget stands in for the content (widget present, almost no text).
BLOCK_PHRASES = (
    "verify you are human", "verify that you are human", "are you a robot", "i'm not a robot",
    "complete the captcha", "solve the captcha", "checking your browser", "enable javascript and cookies to continue",
    "please log in to", "please sign in to", "sign in to continue", "log in to continue", "login to continue",
    "log in to view", "login required", "you must be logged in", "members only", "for members only",
    "this content is for members", "subscribe to continue", "subscribe to read", "subscribers only",
    "create a free account to continue", "access denied", "paywall")
CHALLENGE_TITLES = ("just a moment", "attention required", "access denied", "security check",
                    "ddos protection", "are you a robot")
CAPTCHA_WIDGET_CLASSES = ("g-recaptcha", "h-captcha", "cf-turnstile", "captcha")
MIN_CONTENT_WORDS = 150


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def paths(root=SEED_DIR):
    return {"root": root, "candidates": os.path.join(root, "candidates"),
            "sources": os.path.join(root, "sources"), "rejected": os.path.join(root, "rejected"),
            "candidates_csv": os.path.join(root, "candidates.csv"),
            "pages_csv": os.path.join(root, "graphology_pages.csv"),
            "log_csv": os.path.join(root, "collection_log.csv"),
            "manual_csv": os.path.join(root, "manual_review.csv"),
            "review_dir": os.path.join(root, "review")}


# ---------------------------------------------------------------------------
# CSV helpers (atomic rewrite, so an interrupted run never leaves a half file)
# ---------------------------------------------------------------------------
def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def append_csv(path, row, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Image checks and hashing
# ---------------------------------------------------------------------------
_MAGIC = [(b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"), (b"BM", ".bmp"),
          (b"II*\x00", ".tif"), (b"MM\x00*", ".tif"), (b"GIF87a", ".gif"), (b"GIF89a", ".gif")]


def image_extension(data):
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def decode_image(data):
    """-> (grayscale uint8 array, extension). Raises ValueError for anything unusable."""
    import cv2
    ext = image_extension(data)
    if ext is None:
        raise ValueError("not a supported image format (png / jpg / gif / bmp / tif / webp)")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("image data could not be decoded (corrupt or truncated)")
    h, w = img.shape
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        raise ValueError(f"too small ({w}x{h}) - icon or decoration")
    if w * h > MAX_PIXELS:
        raise ValueError(f"too large ({w}x{h})")
    return img, ext


# Handwriting-likeness: graphology pages mix handwriting samples with stock photos
# and illustrations carrying the same emotion alt text. A handwriting sample is
# dark ink on light, mostly unsaturated paper. Calibrated on the collected pages
# (2026-09-29): the 4 real samples had paper grey >= 244, saturation p90 <= 20 and
# paper covering >= 85%; all 9 photos / cartoon stills / printed cards had
# saturation p90 >= 77 or a dark / mid-tone background. Thresholds leave a wide
# margin; tinted-paper scans could still be refused - they are logged with their
# numbers (collection_log.csv) and --no-handwriting-filter turns the check off.
HW_MAX_SAT_P90 = 60          # 0..255 HSV saturation, 90th percentile
HW_MIN_PAPER_GREY = 150      # the most common grey level must be light (paper)
HW_MIN_PAPER_FRACTION = 0.75 # pixels within +/-30 grey levels of the paper
HW_INK_FRACTION = (0.005, 0.30)


def handwriting_likeness(data, gray):
    """-> (looks like a handwriting sample?, measurements string)."""
    import cv2
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    sat_p90 = float(np.percentile(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1], 90)) if bgr is not None else 0.0
    g = gray.astype(np.int32)
    paper = int(np.bincount(g.ravel(), minlength=256).argmax())
    paper_frac = float((np.abs(g - paper) <= 30).mean())
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_frac = float(ink.mean())
    detail = (f"saturation p90 {sat_p90:.0f} (max {HW_MAX_SAT_P90}), paper grey {paper} (min {HW_MIN_PAPER_GREY}), "
              f"paper fraction {paper_frac:.2f} (min {HW_MIN_PAPER_FRACTION}), ink fraction {ink_frac:.3f}")
    ok = (sat_p90 <= HW_MAX_SAT_P90 and paper >= HW_MIN_PAPER_GREY and paper_frac >= HW_MIN_PAPER_FRACTION
          and HW_INK_FRACTION[0] <= ink_frac <= HW_INK_FRACTION[1])
    return ok, detail


def _ink_crop(gray):
    import cv2
    _, ink = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ink.mean() > 127:                                  # light ink on dark paper
        ink = 255 - ink
    ys, xs = np.nonzero(ink)
    if xs.size == 0:
        return gray
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def dhash(gray):
    """256-bit difference hash (hex) of the ink region."""
    import cv2
    small = cv2.resize(_ink_crop(gray), (17, 16), interpolation=cv2.INTER_AREA).astype(int)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):064x}"


def signature(gray):
    """(normalised 128x32 ink-region image, aspect ratio) for the correlation check."""
    import cv2
    c = _ink_crop(gray)
    v = cv2.GaussianBlur(cv2.resize(c, (128, 32), interpolation=cv2.INTER_AREA).astype(float), (5, 5), 0).ravel()
    return (v - v.mean()) / (v.std() + 1e-9), c.shape[1] / max(c.shape[0], 1)


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def is_duplicate(a, b):
    """a, b: {"sha256", "phash", "sig"}. -> (bool, detail)."""
    if a["sha256"] == b["sha256"]:
        return True, "identical file"
    if len(a["phash"]) != len(b["phash"]) or hamming(a["phash"], b["phash"]) > PHASH_PREFILTER:
        return False, ""
    (va, ra), (vb, rb) = a["sig"], b["sig"]
    if abs(np.log(ra / rb)) > 0.15:
        return False, ""
    ncc = float(np.dot(va, vb) / len(va))
    return (ncc >= NCC_DUPLICATE), f"same picture (correlation {ncc:.3f})"


def fingerprint(data, gray=None):
    gray = decode_image(data)[0] if gray is None else gray
    return {"sha256": hashlib.sha256(data).hexdigest(), "phash": dhash(gray), "sig": signature(gray)}


def find_duplicate(fp, known):
    """known: list of (fingerprint, description). -> description of the duplicate or None."""
    for k, desc in known:
        dup, detail = is_duplicate(fp, k)
        if dup:
            return f"{detail}: {desc}"
    return None


def known_images(p):
    """Fingerprints of every collected image (candidate / approved / rejected) and every
    image already in sources/ - so nothing already seen or rejected comes back."""
    known, done = [], set()
    for r in read_csv(p["candidates_csv"]):
        path = os.path.join(p["root"], r["filename"])
        try:
            with open(path, "rb") as fh:
                fp = fingerprint(fh.read())
        except Exception:                                   # file gone: exact-hash check still works
            fp = {"sha256": r["sha256"], "phash": "", "sig": (np.zeros(1), 1.0)}
        known.append((fp, f"{r['id']} ({r['status']})"))
        done.add(os.path.realpath(path))
    if os.path.isdir(p["sources"]):
        for d, _, files in os.walk(p["sources"]):
            for f in files:
                path = os.path.join(d, f)
                if os.path.realpath(path) in done:
                    continue
                try:
                    with open(path, "rb") as fh:
                        known.append((fingerprint(fh.read()), f"sources/{os.path.relpath(path, p['sources'])}"))
                except Exception:
                    continue                                # not an image (e.g. sources.csv, .json)
    return known


# ---------------------------------------------------------------------------
# HTML -> a small tree, and each image's own context
# ---------------------------------------------------------------------------
VOID = {"img", "br", "hr", "meta", "link", "input", "source", "area", "base", "col", "embed", "param",
        "track", "wbr"}
SKIP_TEXT = {"script", "style", "noscript", "template"}


class Node:
    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent, self.children = tag, dict(attrs), parent, []

    def text(self):
        out = []
        for c in self.children:
            if isinstance(c, str):
                out.append(c)
            elif c.tag not in SKIP_TEXT:
                out.append(c.text())
        return re.sub(r"\s+", " ", " ".join(out)).strip()

    def classes(self):
        return (self.attrs.get("class") or "").lower()


class TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {}, None)
        self.stack = [self.root]
        self.order = []                                   # elements in document order

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)
        self.order.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        node = Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)
        self.order.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if data.strip():
            self.stack[-1].children.append(data)


def parse_html(text):
    b = TreeBuilder()
    b.feed(text)
    b.close()
    return b


def _next_element(node):
    siblings = node.parent.children if node.parent else []
    i = siblings.index(node)
    for s in siblings[i + 1:]:
        if isinstance(s, Node):
            return s
        if s.strip():
            return None
    return None


def image_context(img, headings_before):
    """{source: text} for one <img>: alt, title, figcaption, caption, section_heading."""
    ctx = {"alt": img.attrs.get("alt") or "", "title": img.attrs.get("title") or ""}
    # <figure> ... <figcaption>
    anc, depth = img.parent, 0
    while anc is not None and depth < 4:
        if anc.tag == "figure":
            for c in anc.children:
                if isinstance(c, Node) and c.tag == "figcaption":
                    ctx["figcaption"] = c.text()
            break
        anc, depth = anc.parent, depth + 1
    # a caption container around the image (e.g. WordPress div.wp-caption) or a caption right after it
    captions = []
    anc, depth = img.parent, 0
    while anc is not None and depth < 3 and anc.tag not in ("body", "#root"):
        if "caption" in anc.classes() and anc.tag != "figure":
            captions.append(anc.text())
            break
        anc, depth = anc.parent, depth + 1
    holder = img.parent if img.parent is not None and img.parent.tag in ("a", "picture") else img
    nxt = _next_element(holder)
    if nxt is not None and (nxt.tag in ("figcaption", "caption") or "caption" in nxt.classes()):
        captions.append(nxt.text())
    ctx["caption"] = " ".join(captions)
    ctx["section_heading"] = headings_before
    return {k: html.unescape(v).strip() for k, v in ctx.items() if v and v.strip()}


def fold(text):
    """Case- and accent-insensitive form: NFKD, combining marks dropped, lower case.
    Keeps one character per character of the NFC text for ASCII-mappable letters,
    so match offsets stay meaningful for snippets."""
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFC", text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().replace("ı", "i")


# A term must not be glued to other LETTERS; digits, underscores and punctuation
# separate words, so filename-like alt text ("1alegria", "angry_writing") matches
# while "unhappy" does not match "happy".
_TERM_RE = {label: [re.compile(r"(?<![^\W\d_])(?<!-)" + re.escape(fold(t)).replace(r"\ ", r"\s+")
                               + r"(?![^\W\d_])(?!-)")
                    for t in terms] for label, terms in LABEL_TERMS.items()}


def find_labels(text):
    """-> {label: (term, snippet)} for non-negated explicit mentions (any supported language)."""
    found = {}
    original = unicodedata.normalize("NFC", text)
    text = fold(original)
    shown = original if len(original) == len(text) else text     # snippets in the page's own spelling
    for label, patterns in _TERM_RE.items():
        for pat in patterns:
            for m in pat.finditer(text):
                before = re.findall(r"[\w']+", text[:m.start()])[-3:]
                if NEGATIONS & set(before):
                    continue
                a, b = max(0, m.start() - 80), min(len(text), m.end() + 80)
                found.setdefault(label, (shown[m.start():m.end()], shown[a:b].strip()))
                break
    return found


def decide_label(ctx, allow_section_headings=False):
    """-> (label or None, evidence snippet, evidence source, reason if None)."""
    sources = IMAGE_LEVEL_SOURCES + (("section_heading",) if allow_section_headings else ())
    matches = {}
    for src in sources:
        for label, (term, snippet) in find_labels(ctx.get(src, "")).items():
            matches.setdefault(label, (snippet, src))
    if not matches:
        return None, "", "", "no explicit emotion association in the image's own context"
    if len(matches) > 1:
        return None, "", "", f"ambiguous: context names {sorted(matches)}"
    label, (snippet, src) = next(iter(matches.items()))
    return label, snippet, src, ""


# Lazy-loading scripts keep the real image URL in one of these attributes and
# put a placeholder (often a 1x1 data: GIF) in src. Checked in this order.
LAZY_ATTRS = ("data-layzr", "data-lazy-src", "data-src", "data-original", "data-orig-file", "data-lazyload")
SRCSET_ATTRS = ("data-lazy-srcset", "data-srcset", "srcset")
PLACEHOLDER_RE = re.compile(r"(^|[/_.-])(blank|spacer|placeholder|pixel|transparent|lazy|loading|1x1)"
                            r"([/_.-]|\.(gif|png|svg))", re.I)
NON_SAMPLE_RE = re.compile(r"(^|[/_.-])(logo|favicon|icon|icons|avatar|gravatar|sprite|emoji|badge|button|"
                           r"banner|header[-_]?logo)([/_.-]|$)", re.I)


def _largest_in_srcset(srcset):
    items = [c.strip().split() for c in srcset.split(",") if c.strip()]
    items = [c for c in items if c and not c[0].startswith("data:")]
    if not items:
        return ""
    return max(items, key=lambda c: float(re.sub(r"[^\d.]", "", c[1]) or 0) if len(c) > 1 else 0)[0]


def is_placeholder(url):
    if url.startswith("data:"):
        return len(url) < 2000                           # inline 1x1 / blurred placeholder
    return bool(PLACEHOLDER_RE.search(urllib.parse.urlsplit(url).path))


def real_image_url(attrs):
    """The actual image URL of an <img>, looking past lazy-loading placeholders.
    Returns "" for placeholders-only, logos, icons, avatars and similar non-samples."""
    candidates = [attrs.get(a) or "" for a in LAZY_ATTRS]
    candidates += [_largest_in_srcset(attrs.get(a) or "") for a in SRCSET_ATTRS]
    candidates.append(attrs.get("src") or "")
    for url in (html.unescape(c.strip()) for c in candidates):
        if not url or is_placeholder(url):
            continue
        if NON_SAMPLE_RE.search(urllib.parse.urlsplit(url).path):
            return ""
        return url
    return ""


def page_images(page_html, page_url):
    """-> (page title, license note, [(image url, context dict)])."""
    tree = parse_html(page_html)
    title = ""
    for n in tree.order:
        if n.tag == "title":
            title = n.text()
            break
        if n.tag == "meta" and n.attrs.get("property") == "og:title":
            title = n.attrs.get("content") or ""
    license_note = ""
    for n in tree.order:
        href = n.attrs.get("href") or ""
        if n.tag in ("a", "link") and ("license" in (n.attrs.get("rel") or "").lower()
                                       or "creativecommons.org/licenses" in href):
            license_note = f"page links a licence: {href}"
            break
    out, heading = [], ""
    for n in tree.order:
        if n.tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading = n.text()
        if n.tag != "img":
            continue
        src = real_image_url(n.attrs)
        if not src:
            continue                                      # placeholder only, or a logo / icon / pixel
        try:
            w, h = int(n.attrs.get("width") or 0), int(n.attrs.get("height") or 0)
        except ValueError:
            w = h = 0
        if (w and w < MIN_WIDTH) or (h and h < MIN_HEIGHT // 2):
            continue                                      # declared as an icon / tracking pixel
        url = src if src.startswith("data:") else urllib.parse.urljoin(page_url, html.unescape(src))
        out.append((url, image_context(n, heading)))
    return html.unescape(title).strip(), license_note, out


# ---------------------------------------------------------------------------
# Polite fetching
# ---------------------------------------------------------------------------
class Response:
    def __init__(self, status, body=b"", headers=None, error=""):
        self.status, self.body, self.headers, self.error = status, body, headers or {}, error


class UrllibFetcher:
    def __init__(self, user_agent=USER_AGENT, timeout=20, max_bytes=10 * 1024 * 1024):
        self.user_agent, self.timeout, self.max_bytes = user_agent, timeout, max_bytes

    def get(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read(self.max_bytes + 1)
                if len(body) > self.max_bytes:
                    return Response(0, error=f"larger than {self.max_bytes} bytes")
                return Response(r.status, body, {k.lower(): v for k, v in r.headers.items()})
        except urllib.error.HTTPError as e:
            return Response(e.code, headers={k.lower(): v for k, v in (e.headers or {}).items()},
                            error=str(e))
        except Exception as e:  # DNS, TLS, timeout ...
            return Response(0, error=str(e))


class PoliteClient:
    """robots.txt + per-host delay around a fetcher. Never retries a refusal."""

    def __init__(self, fetcher, delay=3.0, user_agent=USER_AGENT, sleep=time.sleep, clock=time.monotonic):
        self.fetcher, self.delay, self.user_agent = fetcher, delay, user_agent
        self.sleep, self.clock = sleep, clock
        self.robots, self.last, self.blocked_hosts = {}, {}, {}

    def _host(self, url):
        u = urllib.parse.urlsplit(url)
        return f"{u.scheme}://{u.netloc}"

    def _wait(self, host, crawl_delay):
        gap = max(self.delay, crawl_delay or 0)
        if host in self.last:
            remaining = gap - (self.clock() - self.last[host])
            if remaining > 0:
                self.sleep(remaining)
        self.last[host] = self.clock()

    def robots_for(self, host):
        if host not in self.robots:
            self._wait(host, 0)
            r = self.fetcher.get(host + "/robots.txt")
            rp = urllib.robotparser.RobotFileParser()
            if r.status == 200:
                rp.parse(r.body.decode("utf-8", "replace").splitlines())
                self.robots[host] = (rp, "")
            elif r.status in (401, 403):
                self.robots[host] = (None, f"robots.txt answered HTTP {r.status}: treated as disallow-all")
            elif 400 <= r.status < 500:
                rp.parse([])                              # no robots.txt: everything allowed
                self.robots[host] = (rp, "")
            else:
                self.robots[host] = (None, f"robots.txt unavailable ({r.status or r.error}): not fetching")
        return self.robots[host]

    def get(self, url):
        """-> (Response or None, refusal reason). None when we must not fetch."""
        if url.startswith("data:"):
            try:
                header, payload = url.split(",", 1)
                data = base64.b64decode(payload) if ";base64" in header else urllib.parse.unquote_to_bytes(payload)
                return Response(200, data), ""
            except Exception as exc:
                return None, f"bad data: URI ({exc})"
        if not url.startswith(("http://", "https://")):
            return None, "only http(s) URLs are fetched"
        host = self._host(url)
        if host in self.blocked_hosts:
            return None, self.blocked_hosts[host]
        rp, why = self.robots_for(host)
        if rp is None:
            return None, why
        if not rp.can_fetch(self.user_agent, url):
            return None, "disallowed by robots.txt"
        self._wait(host, rp.crawl_delay(self.user_agent))
        r = self.fetcher.get(url)
        if r.status == 429:
            self.blocked_hosts[host] = "host answered 429 (rate limited): stopped for this run"
            return None, self.blocked_hosts[host]
        if r.status in (401, 402, 403, 407, 451):
            return None, f"access restricted (HTTP {r.status})"
        if r.status != 200:
            return None, f"failed (HTTP {r.status or ''} {r.error})".strip()
        return r, ""


def visible_text(tree):
    """Text a reader would see (no script / style / noscript / template, no hidden elements)."""
    def walk(node):
        if node.tag in SKIP_TEXT or node.tag in ("head", "title"):
            return ""
        style = (node.attrs.get("style") or "").replace(" ", "").lower()
        if "hidden" in node.attrs or "display:none" in style or "visibility:hidden" in style \
                or node.attrs.get("aria-hidden") == "true":
            return ""
        return " ".join(c if isinstance(c, str) else walk(c) for c in node.children)
    return re.sub(r"\s+", " ", walk(tree.root)).strip()


def access_block(page_html):
    """-> reason string when the page's content is genuinely gated (CAPTCHA / login /
    paywall / bot challenge), else None. Invisible or form-only reCAPTCHA is ignored."""
    tree = parse_html(page_html)
    title = next((n.text() for n in tree.order if n.tag == "title"), "").lower()
    if any(t in title for t in CHALLENGE_TITLES):
        return f"bot-challenge page (title {title[:60]!r})"
    text = visible_text(tree)
    low = fold(text)
    phrase = next((p for p in BLOCK_PHRASES if p in low), None)
    if phrase:
        return f"page says {phrase!r} (login / paywall / CAPTCHA)"
    widget = any(any(c in n.classes() for c in CAPTCHA_WIDGET_CLASSES)
                 and n.attrs.get("data-size") != "invisible" for n in tree.order)
    if widget and len(text.split()) < MIN_CONTENT_WORDS:
        return "visible CAPTCHA widget in place of the page content"
    return None


def looks_blocked(body_text):
    """Kept for compatibility: True-ish reason when access_block() refuses the page."""
    return access_block(body_text)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def read_pages(path):
    rows = read_csv(path)
    if rows and "url" not in rows[0]:
        raise ValueError(f"{path}: needs a 'url' column")
    rows = [r for r in rows if (r.get("url") or "").strip() and not r["url"].lstrip().startswith("#")]
    if not rows:
        raise FileNotFoundError(
            f"No pages listed in {path}.\nAdd one public graphology reference page per row (columns url, "
            f"license_or_access_note, notes; rows starting with # are ignored) - see data/README.md, Part B.")
    return rows


def collect(p, client, pages, *, allow_section_headings=False, max_images_per_page=50, dry_run=False,
            handwriting_filter=True):
    """-> summary dict. Writes candidates/, candidates.csv, collection_log.csv, manual_review.csv."""
    candidates = read_csv(p["candidates_csv"])
    known = known_images(p)
    seen_urls = {r["image_url"] for r in candidates}
    summary = {"pages": 0, "pages_blocked": 0, "images_seen": 0, "new_candidates": 0, "duplicates": 0,
               "no_evidence": 0, "ambiguous": 0, "invalid": 0, "blocked_images": 0, "downloaded": 0,
               "not_handwriting": 0, "per_label": {}}

    def log(page, url, outcome, detail=""):
        if not dry_run:
            append_csv(p["log_csv"], {"time": now(), "source_page": page, "image_url": url,
                                      "outcome": outcome, "detail": detail}, LOG_FIELDS)

    def manual(url, kind, reason, page=""):
        if not dry_run:
            append_csv(p["manual_csv"], {"time": now(), "url": url, "kind": kind, "reason": reason,
                                         "source_page": page}, MANUAL_FIELDS)

    for page in pages:
        page_url = page["url"].strip()
        r, why = client.get(page_url)
        if r is None:
            summary["pages_blocked"] += 1
            manual(page_url, "page", why)
            print(f"[COLLECT] page skipped -> manual review: {page_url} ({why})")
            continue
        ctype = r.headers.get("content-type", "text/html")
        if "html" not in ctype.lower():
            summary["pages_blocked"] += 1
            manual(page_url, "page", f"not an HTML page ({ctype})")
            continue
        m = re.search(r"charset=([\w-]+)", ctype, re.I)
        try:
            text = r.body.decode(m.group(1) if m else "utf-8", "replace")
        except LookupError:
            text = r.body.decode("utf-8", "replace")
        blocked = access_block(text)
        if blocked:
            summary["pages_blocked"] += 1
            manual(page_url, "page", f"access restricted: {blocked} - not bypassed")
            print(f"[COLLECT] page looks restricted -> manual review: {page_url} ({blocked})")
            continue
        summary["pages"] += 1
        title, page_license, images = page_images(text, page_url)
        access_note = "; ".join(x for x in [(page.get("license_or_access_note") or "").strip(), page_license] if x) \
            or "unknown - check the site's terms before using these images"
        print(f"[COLLECT] {page_url}: {len(images)} image(s) - {title[:70]!r}")
        for img_url, ctx in images[:max_images_per_page]:
            summary["images_seen"] += 1
            shown = img_url if not img_url.startswith("data:") else "data:(inline image)"
            label, evidence, ev_src, reason = decide_label(ctx, allow_section_headings)
            if label is None:
                summary["ambiguous" if reason.startswith("ambiguous") else "no_evidence"] += 1
                log(page_url, shown, "not_collected", reason)
                continue
            if img_url in seen_urls:
                summary["duplicates"] += 1
                log(page_url, shown, "duplicate", "image URL already collected")
                continue
            ir, why = client.get(img_url)
            if ir is None:
                summary["blocked_images"] += 1
                manual(shown, "image", why, page_url)
                log(page_url, shown, "blocked", why)
                continue
            try:
                gray, ext = decode_image(ir.body)
            except ValueError as exc:
                summary["invalid"] += 1
                log(page_url, shown, "invalid_image", str(exc))
                continue
            summary["downloaded"] += 1
            if handwriting_filter:
                looks_ok, measured = handwriting_likeness(ir.body, gray)
                if not looks_ok:
                    summary["not_handwriting"] += 1
                    log(page_url, shown, "not_handwriting_like", f"{label} context, but {measured}")
                    continue
            fp = fingerprint(ir.body, gray)
            sha, ph = fp["sha256"], fp["phash"]
            dup = find_duplicate(fp, known)
            if dup:
                summary["duplicates"] += 1
                log(page_url, shown, "duplicate", dup)
                continue
            cid = "g" + sha[:12]
            rel = f"candidates/{label.lower()}/{cid}{ext}"
            description = " | ".join(f"{k}: {v}" for k, v in ctx.items()
                                     if k in IMAGE_LEVEL_SOURCES or (allow_section_headings and k == "section_heading"))
            row = {"id": cid, "filename": rel, "candidate_label": label, "original_candidate_label": label,
                   "label_evidence": evidence[:300], "evidence_source": ev_src, "source_page": page_url,
                   "image_url": shown, "page_title": title[:200], "description": description[:500],
                   "status": "candidate", "scientifically_validated": "false",
                   "license_or_access_note": access_note, "sha256": sha, "phash": ph,
                   "width": gray.shape[1], "height": gray.shape[0],
                   "trajectory_source": "reconstructed_from_image", "timing_available": "false",
                   "pressure_available": "false", "stroke_order_available": "estimated",
                   "collected_at": now(), "reviewed_at": "", "review_note": "", "approved_file": "",
                   "prototype_only": "true"}
            summary["new_candidates"] += 1
            summary["per_label"][label] = summary["per_label"].get(label, 0) + 1
            known.append((fp, cid))
            seen_urls.add(img_url)
            if dry_run:
                print(f"[COLLECT]   would collect {cid} as {label} ({ev_src}: {evidence[:60]!r})")
                continue
            os.makedirs(os.path.join(p["root"], os.path.dirname(rel)), exist_ok=True)
            with open(os.path.join(p["root"], rel), "wb") as f:
                f.write(ir.body)                             # original bytes, unchanged
            candidates.append(row)
            write_csv(p["candidates_csv"], candidates, CANDIDATE_FIELDS)
            log(page_url, shown, "candidate", f"{cid} -> {label}")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=SEED_DIR, help="the seed_4class folder")
    p.add_argument("--pages", help="page list CSV (default: <root>/graphology_pages.csv)")
    p.add_argument("--delay", type=float, default=3.0, help="minimum seconds between requests to one host")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--max-images-per-page", type=int, default=50)
    p.add_argument("--allow-section-headings", action="store_true",
                   help="also accept the nearest section heading as the image's label evidence")
    p.add_argument("--dry-run", action="store_true", help="fetch and report, but save nothing")
    p.add_argument("--auto-approve", action="store_true",
                   help="PROTOTYPE workflow: move every candidate that passed the filters straight into "
                        "sources/ (status=auto_approved) - no human review")
    p.add_argument("--no-handwriting-filter", action="store_true",
                   help="keep images even when they do not look like ink on paper (photos, cartoons)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    p = paths(os.path.abspath(a.root))
    try:
        pages = read_pages(a.pages or p["pages_csv"])
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    for d in ("happy", "neutral", "sad", "angry"):
        os.makedirs(os.path.join(p["candidates"], d), exist_ok=True)
    client = PoliteClient(UrllibFetcher(timeout=a.timeout), delay=a.delay)
    s = collect(p, client, pages, allow_section_headings=a.allow_section_headings,
                max_images_per_page=a.max_images_per_page, dry_run=a.dry_run,
                handwriting_filter=not a.no_handwriting_filter)
    print(f"[COLLECT] {json.dumps(s)}")
    print(f"[COLLECT] candidates -> {p['candidates_csv']} (status=candidate, scientifically_validated=false)")
    print(f"[COLLECT] NOTE: {DISCLAIMER}")
    if a.auto_approve and not a.dry_run:
        from .review_graphology import auto_approve_pending, format_stats
        done = auto_approve_pending(p)
        print(f"[COLLECT] auto-approved {len(done)} candidate(s) into sources/ (status=auto_approved, "
              f"prototype_only=true, scientifically_validated=false) - no human review")
        print(format_stats(read_csv(p["candidates_csv"])))
        print("[COLLECT] Nothing has been trained. Next: python -m handwriting.generate_seed_data --force")
    else:
        print("[COLLECT] Nothing has been approved or trained. Next: python -m handwriting.review_graphology")
    return 0


if __name__ == "__main__":
    sys.exit(main())

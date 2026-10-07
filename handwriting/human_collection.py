"""
handwriting/human_collection.py - save participant handwriting from the canvas
for the 4-class experiment (Collect tab on /handwriting).

A participant enters an anonymous writer id (writer_001, writer_002, ...),
selects Happy / Neutral / Sad / Angry and writes naturally. The page gives no
instruction on HOW to write any label. Each sample is stored unchanged as JSON:

    data/handwriting/seed_4class/human/<label>/<writer_id>_<label>_<NNN>.json
    {
      "schema": "human_canvas/1",
      "writer_id": "writer_003", "label": "Happy",
      "points": [{"x", "y", "t", "pen_down", "pressure"?, "altitude"?, "azimuth"?}, ...],
      "timestamp": "2026-09-29T21:04:11Z",
      "source_type": "human_canvas", "scientifically_validated": false,
      "pointer_types": ["pen"], "n_points": 812
    }

The label is the one the participant selected; it is not a validated emotional
state. Files are never overwritten. generate_seed_data reads this folder next to
the graphology sources and keeps the origin in the manifest (sample_source).
"""

import datetime
import json
import os
import re
import threading

from . import config
from .preprocessing import HandwritingInputError, from_browser_points, prepare

LABEL_OF = {c.lower(): c for c in config.FOUR_CLASS_LABELS}
WRITER_ID_RE = re.compile(config.WRITER_ID_PATTERN)
FILE_RE = re.compile(r"^(writer_\d{3,6})_([a-z]+)_(\d+)\.json$")
POINTER_TYPES = {"mouse", "pen", "touch"}
OPTIONAL = ("pressure", "altitude", "azimuth")
_lock = threading.Lock()


def check_writer_id(writer_id):
    w = (writer_id or "").strip().lower() if isinstance(writer_id, str) else ""
    if not WRITER_ID_RE.match(w):
        raise HandwritingInputError("Writer ID must look like writer_001 (anonymous id - no names).")
    return w


def check_label(label):
    c = LABEL_OF.get(str(label or "").strip().lower())
    if c is None:
        raise HandwritingInputError(f"Select one of {', '.join(config.FOUR_CLASS_LABELS)}.")
    return c


def clean_points(points):
    """Validate like the analyzer does, then keep exactly the canvas format:
    x, y, t, pen_down, plus pressure / altitude / azimuth only when reported."""
    prepare(from_browser_points(points))                     # raises HandwritingInputError when unusable
    out = []
    for p in points:
        q = {"x": float(p["x"]), "y": float(p["y"]), "t": float(p.get("t", p.get("timestamp"))),
             "pen_down": bool(p["pen_down"])}
        for k in OPTIONAL:
            v = p.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                q[k] = float(v)
        out.append(q)
    return out


def _files(root):
    for label in LABEL_OF:
        folder = os.path.join(root, label)
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                m = FILE_RE.match(f)
                if m and m.group(2) == label:
                    yield label, m.group(1), int(m.group(3)), os.path.join(folder, f)


def counts(root=config.HUMAN_CANVAS_DIR, writer_id=None):
    totals = {c: 0 for c in config.FOUR_CLASS_LABELS}
    mine = {c: 0 for c in config.FOUR_CLASS_LABELS}
    writers = set()
    for label, w, _, _ in _files(root):
        totals[LABEL_OF[label]] += 1
        writers.add(w)
        if writer_id and w == writer_id:
            mine[LABEL_OF[label]] += 1
    return {"totals": totals, "writer": mine if writer_id else None, "writers": len(writers),
            "samples": sum(totals.values())}


def next_writer_id(root=config.HUMAN_CANVAS_DIR):
    used = [int(w.split("_")[1]) for _, w, _, _ in _files(root)]
    return f"writer_{(max(used) + 1 if used else 1):03d}"


def save_sample(payload, root=config.HUMAN_CANVAS_DIR):
    """-> {"file", "writer_id", "label", "number", "counts"}. Raises HandwritingInputError."""
    if not isinstance(payload, dict):
        raise HandwritingInputError("expected a JSON object")
    writer_id = check_writer_id(payload.get("writer_id"))
    label = check_label(payload.get("label"))
    points = clean_points(payload.get("points"))
    pointer_types = sorted({t for t in (payload.get("pointer_types") or []) if t in POINTER_TYPES})
    record = {"schema": "human_canvas/1", "writer_id": writer_id, "label": label, "points": points,
              "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "source_type": "human_canvas", "scientifically_validated": False,
              "pointer_types": pointer_types, "n_points": len(points)}
    folder = os.path.join(root, label.lower())
    with _lock:
        os.makedirs(folder, exist_ok=True)
        n = 1 + max([k for lab, w, k, _ in _files(root) if w == writer_id and lab == label.lower()], default=0)
        while True:
            path = os.path.join(folder, f"{writer_id}_{label.lower()}_{n:03d}.json")
            try:
                with open(path, "x", encoding="utf-8") as f:     # "x": never overwrite
                    json.dump(record, f)
                break
            except FileExistsError:
                n += 1
    number = sum(1 for lab, w, _, _ in _files(root) if w == writer_id and lab == label.lower())
    return {"file": os.path.relpath(path, root), "writer_id": writer_id, "label": label, "number": number,
            "counts": counts(root, writer_id)}

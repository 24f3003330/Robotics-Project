"""
handwriting/review_graphology.py - human review of collected graphology candidates.

    python -m handwriting.review_graphology                 # interactive: one candidate at a time
    python -m handwriting.review_graphology --open          # ... and open each image in the system viewer
    python -m handwriting.review_graphology --label sad     # only one class
    python -m handwriting.review_graphology --contact-sheet # PNG grids of the pending candidates
    python -m handwriting.review_graphology --stats
    python -m handwriting.review_graphology --approve g1a2b3c4d5e6 --reject g0f9e8d7c6b5 --reason "not handwriting"
    python -m handwriting.review_graphology --relabel g1a2b3c4d5e6 Sad

Keys:  [y] approve   [n] reject   [s] skip   [r] relabel (1 Happy 2 Neutral 3 Sad 4 Angry)   [q] quit

Approve: candidates/<label>/<file> -> sources/<label>/<file>, status=approved, and a
row in sources/sources.csv (file, label, reference = image URL, source page,
candidate id) so every approved image traces back to where it came from. Only
files in sources/ are used by `python -m handwriting.generate_seed_data`.
Reject: -> rejected/<label>/<file>, status=rejected; it can never enter training
(and stays known, so the collector does not fetch it again).

Approval only means "usable as a graphology seed example". It does not make the
label scientifically valid: scientifically_validated stays false.
"""

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

from . import config
from .collect_graphology import (CANDIDATE_FIELDS, DISCLAIMER, SEED_DIR, now, paths, read_csv, write_csv)

LABELS = config.FOUR_CLASS_LABELS
REJECT_REASONS = {"1": "poor quality", "2": "ambiguous", "3": "duplicate", "4": "irrelevant",
                  "5": "not handwriting"}
SOURCES_FIELDS = ["file", "label", "writer_id", "reference", "source_page", "candidate_id", "status",
                  "prototype_only", "notes"]
APPROVED_STATUSES = ("approved", "auto_approved")


class ReviewError(ValueError):
    pass


def _find(rows, cid):
    for r in rows:
        if r["id"] == cid:
            return r
    raise ReviewError(f"no candidate with id {cid}")


def _move(root, rel_from, rel_to):
    src, dst = os.path.join(root, rel_from), os.path.join(root, rel_to)
    if not os.path.isfile(src):
        raise ReviewError(f"file missing: {rel_from}")
    if os.path.exists(dst):
        raise ReviewError(f"refusing to overwrite {rel_to}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)


def _upsert_sources_csv(p, row):
    """Add / replace this image's row in sources/sources.csv, keeping every other row and column."""
    path = os.path.join(p["sources"], "sources.csv")
    rows = read_csv(path)
    fields = list(dict.fromkeys((list(rows[0].keys()) if rows else []) + SOURCES_FIELDS))
    rows = [r for r in rows if r.get("file") != row["file"]] + [row]
    write_csv(path, rows, fields)


def approve(p, rows, cid, note="", auto=False):
    """Move a candidate into sources/. auto=True: approved by the collector's filters
    alone (prototype workflow, no human look) -> status=auto_approved."""
    r = _find(rows, cid)
    if r["status"] != "candidate":
        raise ReviewError(f"{cid} is {r['status']}, not a candidate")
    name = os.path.basename(r["filename"])
    label_dir = r["candidate_label"].lower()
    _move(p["root"], r["filename"], os.path.join("sources", label_dir, name))
    status = "auto_approved" if auto else "approved"
    r.update(status=status, filename=f"sources/{label_dir}/{name}",
             approved_file=f"sources/{label_dir}/{name}", reviewed_at=now(),
             review_note="; ".join(x for x in [r.get("review_note"), note] if x),
             scientifically_validated="false", prototype_only="true")
    write_csv(p["candidates_csv"], rows, CANDIDATE_FIELDS)
    how = ("auto-approved by the collector filters (no human review)" if auto
           else "approved by a reviewer")
    _upsert_sources_csv(p, {
        "file": f"{label_dir}/{name}", "label": r["candidate_label"], "writer_id": "",
        "reference": r["image_url"], "source_page": r["source_page"], "candidate_id": cid,
        "status": status, "prototype_only": "true",
        "notes": f"graphology seed example {how} {r['reviewed_at']}; prototype only; "
                 f"not scientifically validated"})
    return r


def auto_approve_pending(p):
    """Prototype workflow: every pending candidate (it already passed the collector's
    evidence / handwriting / duplicate / validity filters) goes to sources/. -> list of ids."""
    rows = read_csv(p["candidates_csv"])
    done = []
    for r in rows:
        if r["status"] == "candidate":
            approve(p, rows, r["id"], note="auto-approved (prototype seed data)", auto=True)
            done.append(r["id"])
    return done


def reject(p, rows, cid, reason=""):
    r = _find(rows, cid)
    if r["status"] != "candidate":
        raise ReviewError(f"{cid} is {r['status']}, not a candidate")
    rel = os.path.join("rejected", r["candidate_label"].lower(), os.path.basename(r["filename"]))
    _move(p["root"], r["filename"], rel)
    r.update(status="rejected", filename=rel.replace(os.sep, "/"), reviewed_at=now(),
             review_note="; ".join(x for x in [r.get("review_note"), f"rejected: {reason or 'no reason given'}"] if x))
    write_csv(p["candidates_csv"], rows, CANDIDATE_FIELDS)
    return r


def relabel(p, rows, cid, label):
    label = next((c for c in LABELS if c.lower() == str(label).lower()), None)
    if label is None:
        raise ReviewError(f"label must be one of {LABELS}")
    r = _find(rows, cid)
    if r["status"] != "candidate":
        raise ReviewError(f"{cid} is {r['status']}, not a candidate")
    if label == r["candidate_label"]:
        return r
    rel = f"candidates/{label.lower()}/{os.path.basename(r['filename'])}"
    _move(p["root"], r["filename"], rel)
    note = f"relabelled {r['candidate_label']} -> {label} by reviewer"
    r.update(candidate_label=label, filename=rel,
             review_note="; ".join(x for x in [r.get("review_note"), note] if x))
    write_csv(p["candidates_csv"], rows, CANDIDATE_FIELDS)
    return r


def stats(rows):
    out = {}
    for c in LABELS:
        mine = [r for r in rows if r["candidate_label"] == c]
        out[c] = {s: sum(r["status"] == s for r in mine) for s in ("candidate", "approved", "rejected")}
        out[c]["approved"] += sum(r["status"] == "auto_approved" for r in mine)
    return out


def format_stats(rows):
    lines = []
    for c, s in stats(rows).items():
        lines.append(f"{c + ':':9s}{s['candidate']:4d} candidates | {s['approved']:3d} approved | "
                     f"{s['rejected']:3d} rejected")
    return "\n".join(lines)


def contact_sheets(p, rows, status="candidate", thumb=(260, 150), cols=4):
    """One PNG per class: numbered thumbnails of the items with this status. -> list of paths."""
    import cv2
    os.makedirs(p["review_dir"], exist_ok=True)
    out = []
    for c in LABELS:
        items = [r for r in rows if r["candidate_label"] == c and r["status"] == status]
        if not items:
            continue
        tiles = []
        for i, r in enumerate(items, 1):
            path = os.path.join(p["root"], r["approved_file"] if status == "approved" else r["filename"])
            tile = np.full((thumb[1] + 26, thumb[0], 3), 255, np.uint8)
            img = cv2.imread(path)
            if img is not None:
                k = min(thumb[0] / img.shape[1], thumb[1] / img.shape[0])
                small = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))),
                                   interpolation=cv2.INTER_AREA)
                tile[:small.shape[0], :small.shape[1]] = small
            cv2.rectangle(tile, (0, 0), (thumb[0] - 1, thumb[1] + 25), (200, 200, 200), 1)
            cv2.putText(tile, f"#{i} {r['id']}", (4, thumb[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (40, 40, 40), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        path = os.path.join(p["review_dir"], f"contact_{status}_{c.lower()}.png")
        cv2.imwrite(path, sheet)
        out.append(path)
    return out


def _open_image(path):
    opener = "open" if sys.platform == "darwin" else ("xdg-open" if sys.platform.startswith("linux") else None)
    if opener:
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif hasattr(os, "startfile"):
        os.startfile(path)


def interactive(p, label=None, ask=input, say=print, open_images=False):
    """Review pending candidates one by one. ask/say are injectable for tests."""
    rows = read_csv(p["candidates_csv"])
    queue = [r["id"] for r in rows if r["status"] == "candidate"
             and (label is None or r["candidate_label"].lower() == label.lower())]
    say(DISCLAIMER)
    say(format_stats(rows))
    if not queue:
        say("No candidates waiting for review.")
        return rows
    i = 0
    while i < len(queue):
        r = _find(rows, queue[i])
        say(f"\n[{i + 1}/{len(queue)}] {r['id']}  candidate label: {r['candidate_label']}  "
            f"({r['width']}x{r['height']})")
        say(f"  evidence ({r['evidence_source']}): {r['label_evidence']}")
        say(f"  page: {r['page_title']}\n  {r['source_page']}\n  image: {r['image_url']}")
        say(f"  access: {r['license_or_access_note']}\n  file: {os.path.join(p['root'], r['filename'])}")
        if open_images:
            _open_image(os.path.join(p["root"], r["filename"]))
        key = ask("[y] approve  [n] reject  [s] skip  [r] relabel  [q] quit > ").strip().lower()
        try:
            if key == "y":
                approve(p, rows, r["id"])
                say(f"  approved -> {r['approved_file']}")
            elif key == "n":
                why = ask("  reason: 1 poor quality  2 ambiguous  3 duplicate  4 irrelevant  "
                          "5 not handwriting  (or type one / Enter) > ").strip()
                reject(p, rows, r["id"], REJECT_REASONS.get(why, why))
                say("  rejected")
            elif key == "r":
                choice = ask("  new label: 1 Happy  2 Neutral  3 Sad  4 Angry > ").strip()
                if choice in ("1", "2", "3", "4"):
                    relabel(p, rows, r["id"], LABELS[int(choice) - 1])
                    say(f"  relabelled -> {r['candidate_label']} (still a candidate: approve or reject it now)")
                else:
                    say("  unchanged")
                continue                                  # show the same item again
            elif key == "q":
                break
            elif key != "s":
                say("  unknown key")
                continue
        except ReviewError as exc:
            say(f"  {exc}")
        i += 1
    say("\n" + format_stats(rows))
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=SEED_DIR)
    p.add_argument("--label", help="review only this class")
    p.add_argument("--open", action="store_true", help="open each image in the system image viewer")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--contact-sheet", action="store_true", help="write PNG grids of the pending candidates")
    p.add_argument("--approve", nargs="+", default=[], metavar="ID")
    p.add_argument("--reject", nargs="+", default=[], metavar="ID")
    p.add_argument("--reason", default="")
    p.add_argument("--relabel", nargs=2, metavar=("ID", "LABEL"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    p = paths(os.path.abspath(a.root))
    rows = read_csv(p["candidates_csv"])
    if not rows:
        print(f"No candidates yet ({p['candidates_csv']}). Run python -m handwriting.collect_graphology first.")
        return 0
    batch = bool(a.approve or a.reject or a.relabel or a.stats or a.contact_sheet)
    try:
        if a.relabel:
            relabel(p, rows, *a.relabel)
        for cid in a.approve:
            approve(p, rows, cid)
        for cid in a.reject:
            reject(p, rows, cid, a.reason)
    except ReviewError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if a.contact_sheet:
        for path in contact_sheets(p, rows):
            print(f"[REVIEW] contact sheet -> {path}")
    if batch:
        print(format_stats(rows))
        return 0
    interactive(p, label=a.label, open_images=a.open)
    print("[REVIEW] Approved images are in sources/. Nothing was trained. When ready: "
          "python -m handwriting.generate_seed_data --force && python -m handwriting.train --dataset seed_4class")
    return 0


if __name__ == "__main__":
    sys.exit(main())

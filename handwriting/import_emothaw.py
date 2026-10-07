"""
handwriting/import_emothaw.py - convert the EMOTHAW dataset, exactly as received,
into the standard per-sample layout used for training.

    python -m handwriting.import_emothaw                         # raw/ -> processed/
    python -m handwriting.import_emothaw --labels raw/DASS.csv   # name the label file explicitly
    python -m handwriting.import_emothaw --preview 2             # also render 2 samples per task to PNG

Input (never modified): data/emothaw/raw/
  * recordings: uXXXXXsYYYYY_hwZZZZZ.svc at any depth (writer XXXXX, session
    YYYYY, task ZZZZZ). The writer id is taken from the FILE NAME, never from the
    folder: in the public EMOTHAW sample 7 of 64 folders are named after a
    different writer than the files inside them.
  * the DASS label table that comes with the dataset (.csv / .tsv / .txt, or
    .xlsx with openpyxl installed). Its columns are recognised by name (writer /
    user / subject / id; anxiety; depression; stress). Nothing is guessed: if a
    column is missing or ambiguous the import stops and asks for --writer-col /
    --anxiety-col / --depression-col / --stress-col.

Label encoding (--label-encoding, default auto):
  * dass      numeric DASS-42 scale scores; elevated = score above the normal
              band (anxiety > 7, depression > 9, stress > 14; config.EMOTHAW_HEADS)
  * severity  DASS severity words; "normal" -> normal, mild / moderate / severe /
              extremely severe -> elevated
  * binary    0 / 1 (or normal / elevated) already dichotomised; 1 = elevated
  auto picks dass for numbers above 1 and severity for words. A column that
  holds only 0 and 1 could be either scores or binary labels, so auto stops and
  asks for --label-encoding instead of choosing.

Output: data/emothaw/processed/
  samples/uXXXXX/uXXXXXsYYYYY_hwZZZZZ.csv  columns x, y (canvas convention, y down),
        t_ms, pen_down, pressure (0..1), altitude (0..90 deg), azimuth (0..360 deg),
        stroke - one row per raw tablet sample, nothing added or removed
  manifest.csv      one row per recording (writer, session, task, source file, sha256, ...)
  labels.csv        one row per writer: the three targets + the raw values they came from
  import_report.json
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from . import config
from .preprocessing import (SVC_ALTITUDE_MAX, SVC_AZIMUTH_MAX, SVC_PRESSURE_MAX, STANDARD_COLUMNS,
                            read_svc_rows, stroke_ids, svc_to_trajectory)

SVC_RE = re.compile(r"^u(\d+)s(\d+)_hw(\d+)\.svc$", re.IGNORECASE)
LABEL_EXTS = (".csv", ".tsv", ".txt", ".xlsx")
_norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

COLUMN_ALIASES = {
    "writer": ["writerid", "writer", "user", "userid", "subject", "subjectid", "participant",
               "participantid", "id", "collection", "collectionid", "soggetto", "utente"],
    "anxiety": ["anxiety", "anx", "anxietyscore", "dassanxiety", "dassa", "ansia"],
    "depression": ["depression", "dep", "depr", "depressionscore", "dassdepression", "dassd",
                   "depressione"],
    "stress": ["stress", "stressscore", "dassstress", "dasss"],
}
# DASS severity bands (English and Italian - EMOTHAW was collected in Italy).
SEVERITY_WORDS = {
    "normal": "normal", "normale": "normal",
    "mild": "elevated", "lieve": "elevated",
    "moderate": "elevated", "moderato": "elevated", "moderata": "elevated",
    "severe": "elevated", "grave": "elevated",
    "extremelysevere": "elevated", "estremamentegrave": "elevated",
}
BINARY_WORDS = {"normal": "normal", "elevated": "elevated", "0": "normal", "1": "elevated"}
DASS42_SCALE_MAX = 42


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------
def scan_svc(raw_dir):
    """-> (recordings, skipped). recordings: {"writer", "session", "task", "rel"}."""
    recs, skipped = [], []
    for d, _, files in os.walk(raw_dir):
        for f in sorted(files):
            if not f.lower().endswith(".svc"):
                continue
            rel = os.path.relpath(os.path.join(d, f), raw_dir)
            m = SVC_RE.match(f)
            if not m:
                skipped.append(f"{rel}: file name is not uXXXXXsYYYYY_hwZZZZZ.svc")
                continue
            recs.append({"writer": int(m.group(1)), "session": int(m.group(2)),
                         "task": int(m.group(3)), "rel": rel})
    recs.sort(key=lambda r: (r["writer"], r["task"], r["session"], r["rel"]))
    return recs, skipped


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def standard_rows(rows, pressure_max, azimuth_max, altitude_max):
    """Raw .svc rows (file order) -> standard CSV rows (time order), one per raw sample."""
    traj = svc_to_trajectory(rows, pressure_max, azimuth_max, altitude_max)
    order = np.argsort(traj.t, kind="stable")
    traj = traj.take(order)
    sid, _ = stroke_ids(traj.pen)
    out = []
    for i in range(len(traj)):
        out.append([float(traj.x[i]), float(traj.y[i]), round(float(traj.t[i] * 1000.0), 3),
                     int(traj.pen[i] > 0), round(float(traj.pressure[i]), 5),
                     round(float(traj.altitude[i]), 4), round(float(traj.azimuth[i]), 4), int(sid[i])])
    return out, traj


# ---------------------------------------------------------------------------
# Label table
# ---------------------------------------------------------------------------
def read_table(path):
    """-> list of rows (lists of strings)."""
    if path.lower().endswith(".xlsx"):
        try:
            import openpyxl
        except ImportError:
            raise ValueError(f"{path}: reading .xlsx needs openpyxl (pip install openpyxl), or save the "
                             f"sheet as CSV and pass that with --labels") from None
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        if len(wb.worksheets) > 1:
            print(f"[IMPORT] {os.path.basename(path)}: using the first of {len(wb.worksheets)} sheets "
                  f"({wb.worksheets[0].title!r})")
        return [["" if v is None else str(v) for v in row] for row in wb.worksheets[0].iter_rows(values_only=True)]
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [r for r in csv.reader(text.splitlines(), dialect) if any(c.strip() for c in r)]


def find_columns(header, overrides):
    """-> {"writer": idx, "anxiety": idx, ...}. Raises when a column is missing or ambiguous."""
    norm = [_norm(h) for h in header]
    found, problems = {}, []
    for key, aliases in COLUMN_ALIASES.items():
        if overrides.get(key):
            want = _norm(overrides[key])
            hits = [i for i, h in enumerate(norm) if h == want]
            if len(hits) != 1:
                problems.append(f"--{key}-col {overrides[key]!r} matches {len(hits)} columns")
                continue
        else:
            hits = [i for i, h in enumerate(norm) if h in aliases]
            if len(hits) > 1:
                problems.append(f"{key}: several candidate columns {[header[i] for i in hits]} - "
                                f"pass --{key}-col")
                continue
            if not hits:
                problems.append(f"{key}: no column recognised - pass --{key}-col")
                continue
        found[key] = hits[0]
    if problems:
        raise ValueError("cannot identify the label columns in header "
                         f"{header}:\n  " + "\n  ".join(problems))
    return found


def locate_header(rows, overrides):
    """The header may be preceded by title rows in a spreadsheet: use the first of
    the first 10 rows in which all four columns are found."""
    first_error = None
    for i, row in enumerate(rows[:10]):
        try:
            return i, find_columns(row, overrides)
        except ValueError as exc:
            first_error = first_error or exc
    raise first_error or ValueError("empty label table")


def _as_number(v):
    try:
        x = float(str(v).strip().replace(",", "."))
    except ValueError:
        return None
    return x if np.isfinite(x) else None


def resolve_encoding(values, requested, target):
    """Decide how one column is encoded. values: non-empty raw strings."""
    nums = [_as_number(v) for v in values]
    numeric = all(n is not None for n in nums)
    if requested != "auto":
        return requested
    if not numeric:
        words = {_norm(v) for v in values}
        if words <= set(SEVERITY_WORDS):
            return "severity"
        if words <= set(BINARY_WORDS):
            return "binary"
        raise ValueError(f"{target}: values {sorted(words)[:10]} are neither DASS scores nor known "
                         f"severity / normal-elevated labels")
    if set(nums) <= {0.0, 1.0}:
        raise ValueError(f"{target}: the column only holds 0 and 1, which could be binary labels or "
                         f"genuine DASS scores - pass --label-encoding binary or --label-encoding dass")
    return "dass"


def encode(value, encoding, head):
    """Raw cell -> "normal" / "elevated"."""
    if encoding == "dass":
        score = _as_number(value)
        if score is None:
            raise ValueError(f"{head['name']}: {value!r} is not a number")
        if score < 0 or score > DASS42_SCALE_MAX:
            raise ValueError(f"{head['name']}: {score} is outside the DASS-42 scale range 0..{DASS42_SCALE_MAX}")
        return "elevated" if score > head["dass_cutoff"] else "normal"
    table = SEVERITY_WORDS if encoding == "severity" else BINARY_WORDS
    key = _norm(value)
    if encoding == "binary" and _as_number(value) in (0.0, 1.0):
        key = str(int(_as_number(value)))
    if key not in table:
        raise ValueError(f"{head['name']}: {value!r} is not a {encoding} label")
    return table[key]


def load_label_table(path, overrides, requested_encoding):
    """-> (labels {writer: {head: class, head_raw: str}}, info)."""
    rows = read_table(path)
    header_row, cols = locate_header(rows, overrides)
    header = rows[header_row]
    body = rows[header_row + 1:]
    raw = defaultdict(list)
    for h in config.EMOTHAW_HEADS:
        raw[h["name"]] = [r[cols[h["name"]]].strip() for r in body
                          if len(r) > cols[h["name"]] and r[cols[h["name"]]].strip()]
    encodings = {h["name"]: resolve_encoding(raw[h["name"]], requested_encoding, h["name"])
                 for h in config.EMOTHAW_HEADS}
    labels, incomplete, problems = {}, [], []
    for line, r in enumerate(body, start=header_row + 2):
        cell = lambda k: r[cols[k]].strip() if len(r) > cols[k] else ""
        if not cell("writer"):
            continue
        digits = re.sub(r"\D", "", cell("writer"))
        if not digits:
            problems.append(f"line {line}: writer {cell('writer')!r} has no numeric id")
            continue
        writer = int(digits)
        if any(not cell(h["name"]) for h in config.EMOTHAW_HEADS):
            incomplete.append(writer)            # every target is needed; never filled in
            continue
        try:
            entry = {}
            for h in config.EMOTHAW_HEADS:
                entry[h["name"]] = encode(cell(h["name"]), encodings[h["name"]], h)
                entry[h["name"] + "_raw"] = cell(h["name"])
        except ValueError as exc:
            problems.append(f"line {line}: {exc}")
            continue
        if writer in labels and any(labels[writer][h["name"]] != entry[h["name"]] for h in config.EMOTHAW_HEADS):
            problems.append(f"line {line}: writer {writer} appears twice with different labels")
            continue
        labels[writer] = entry
    if problems:
        raise ValueError(f"{path}: label table problems:\n  " + "\n  ".join(problems[:30]))
    ranges = {}
    for h in config.EMOTHAW_HEADS:
        if encodings[h["name"]] == "dass":
            nums = [_as_number(e[h["name"] + "_raw"]) for e in labels.values()]
            ranges[h["name"]] = [min(nums), max(nums)] if nums else None
    info = {"file": path, "header_row": header_row + 1,
            "columns": {k: header[i] for k, i in cols.items()}, "encoding": encodings,
            "dass_score_ranges": ranges, "writers_with_incomplete_labels": sorted(set(incomplete))}
    return labels, info


def find_label_file(raw_dir):
    """The one table under raw/ whose header names a writer column and the three targets."""
    candidates = []
    for d, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith(LABEL_EXTS) and not f.startswith("~$"):
                path = os.path.join(d, f)
                try:
                    locate_header(read_table(path), {})
                    candidates.append(path)
                except Exception:
                    pass
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No DASS label table found under {raw_dir}. Pass it with --labels (and --*-col if its "
            f"columns are not named writer/user/subject, anxiety, depression, stress).")
    raise ValueError("several possible label tables: " + ", ".join(candidates) + " - pick one with --labels")


# ---------------------------------------------------------------------------
def _inside(a, b):
    a, b = os.path.realpath(a), os.path.realpath(b)
    return a == b or a.startswith(b + os.sep)


def run(a):
    raw_dir, out_dir = os.path.abspath(a.raw), os.path.abspath(a.out)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw EMOTHAW folder not found: {raw_dir}\nPut the dataset there exactly "
                                f"as received (see data/README.md).")
    if _inside(out_dir, raw_dir) or _inside(raw_dir, out_dir):
        raise SystemExit("--out and --raw must be separate folders (raw data is never modified)")
    manifest_path = os.path.join(out_dir, "manifest.csv")
    if os.path.exists(manifest_path) and not a.force:
        raise SystemExit(f"{manifest_path} exists; pass --force to re-import")

    labels_path = os.path.abspath(a.labels) if a.labels else find_label_file(raw_dir)
    overrides = {"writer": a.writer_col, "anxiety": a.anxiety_col, "depression": a.depression_col,
                 "stress": a.stress_col}
    labels, label_info = load_label_table(labels_path, overrides, a.label_encoding)
    print(f"[IMPORT] labels: {labels_path}\n[IMPORT]   columns {label_info['columns']}, "
          f"encoding {label_info['encoding']}, DASS score ranges {label_info['dass_score_ranges']}")

    recs, skipped = scan_svc(raw_dir)
    if not recs:
        raise FileNotFoundError(f"No uXXXXXsYYYYY_hwZZZZZ.svc files under {raw_dir}")

    # read everything first: value ranges are checked across the whole dataset
    loaded, problems, seen = [], [], {}
    for r in recs:
        path = os.path.join(raw_dir, r["rel"])
        try:
            rows = read_svc_rows(path)
        except Exception as exc:
            problems.append(f"{r['rel']}: {exc}")
            continue
        digest = _sha256(path)
        key = (r["writer"], r["session"], r["task"])
        if key in seen:
            if seen[key] == digest:
                problems.append(f"{r['rel']}: identical copy of an already imported file - skipped")
            else:
                problems.append(f"{r['rel']}: a different file with the same writer/session/task was "
                                f"already imported - skipped (check the raw data)")
            continue
        seen[key] = digest
        loaded.append(dict(r, rows=rows, sha256=digest))
    if not loaded:
        raise SystemExit("No readable recordings.\n  " + "\n  ".join(problems[:30]))

    maxima = {name: float(max(r["rows"][:, i].max() for r in loaded))
              for i, name in ((6, "pressure"), (4, "azimuth"), (5, "altitude"))}
    full_scale = {"pressure": a.pressure_max or SVC_PRESSURE_MAX, "azimuth": a.azimuth_max or SVC_AZIMUTH_MAX,
                  "altitude": a.altitude_max or SVC_ALTITUDE_MAX}
    over = [f"{k}: raw max {maxima[k]:g} > assumed full scale {full_scale[k]:g} (--{k}-max)"
            for k in full_scale if maxima[k] > full_scale[k]]
    if over:
        raise SystemExit("Pen values exceed the assumed tablet ranges; the scaling would be wrong:\n  "
                         + "\n  ".join(over))

    if os.path.exists(manifest_path):                      # --force: remove what we wrote before
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                p = os.path.join(out_dir, row["file"])
                if os.path.isfile(p) and _inside(p, out_dir):
                    os.remove(p)

    manifest = []
    for r in loaded:
        rows, traj = standard_rows(r["rows"], full_scale["pressure"], full_scale["azimuth"], full_scale["altitude"])
        down = traj.pen > 0
        if not down.any():
            problems.append(f"{r['rel']}: no pen-down samples - skipped")
            continue
        out_rel = os.path.join("samples", f"u{r['writer']:05d}",
                               f"u{r['writer']:05d}s{r['session']:05d}_hw{r['task']:05d}.csv")
        os.makedirs(os.path.dirname(os.path.join(out_dir, out_rel)), exist_ok=True)
        with open(os.path.join(out_dir, out_rel), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(STANDARD_COLUMNS)
            w.writerows(rows)
        dt = np.diff(traj.t)
        dt = dt[dt > 0]
        manifest.append({
            "file": out_rel.replace(os.sep, "/"), "writer_id": r["writer"], "session": r["session"],
            "task": r["task"], "source": r["rel"].replace(os.sep, "/"), "sha256": r["sha256"],
            "points": len(rows), "pen_down_points": int(down.sum()),
            "strokes": int(stroke_ids(traj.pen)[1].sum()),
            "duration_s": round(float(traj.t[-1] - traj.t[0]), 3),
            "median_dt_ms": round(float(np.median(dt) * 1000.0), 3) if dt.size else "",
            "has_pressure": int(np.ptp(traj.pressure[down]) > 0),
            "has_altitude": int(np.ptp(traj.altitude[down]) > 0),
            "has_azimuth": int(np.ptp(traj.azimuth[down]) > 0),
            "labelled": int(r["writer"] in labels)})
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    label_fields = (["writer_id"] + config.HEAD_NAMES + [n + "_raw" for n in config.HEAD_NAMES])
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(label_fields)
        for writer in sorted(labels):
            e = labels[writer]
            w.writerow([writer] + [e[n] for n in config.HEAD_NAMES] + [e[n + "_raw"] for n in config.HEAD_NAMES])

    file_writers = {m["writer_id"] for m in manifest}
    report = {
        "imported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "raw_dir": raw_dir, "labels": label_info,
        "recordings": len(manifest), "writers_with_recordings": len(file_writers),
        "writers_with_labels": len(labels),
        "writers_with_recordings_and_labels": len(file_writers & set(labels)),
        "writers_without_labels": sorted(file_writers - set(labels)),
        "labelled_writers_without_recordings": sorted(set(labels) - file_writers),
        "recordings_per_task": dict(sorted(Counter(m["task"] for m in manifest).items())),
        "pen_raw_maxima": maxima, "pen_full_scale": full_scale,
        "median_sample_interval_ms": float(np.median([m["median_dt_ms"] for m in manifest
                                                      if m["median_dt_ms"] != ""])),
        "skipped": skipped + problems,
    }
    with open(os.path.join(out_dir, "import_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    both = file_writers & set(labels)
    print(f"[IMPORT] {len(manifest)} recordings from {len(file_writers)} writers -> {manifest_path}")
    print(f"[IMPORT] {len(both)} writers have both recordings and labels; recordings per task "
          f"{report['recordings_per_task']}")
    for h in config.EMOTHAW_HEADS:
        c = Counter(labels[w][h["name"]] for w in both)
        print(f"[IMPORT]   {h['display']:10s}: normal {c['normal']}, elevated {c['elevated']} writers")
    print(f"[IMPORT] pen raw maxima {maxima}; median sample interval "
          f"{report['median_sample_interval_ms']} ms")
    for key, msg in (("writers_without_labels", "writers have recordings but no labels (not used for training)"),
                     ("labelled_writers_without_recordings", "labelled writers have no recordings"),):
        if report[key]:
            print(f"[IMPORT] {len(report[key])} {msg}: {report[key][:20]}")
    if label_info["writers_with_incomplete_labels"]:
        print(f"[IMPORT] {len(label_info['writers_with_incomplete_labels'])} writers lack a score for "
              f"at least one target and were left out: {label_info['writers_with_incomplete_labels'][:20]}")
    if report["skipped"]:
        print(f"[IMPORT] {len(report['skipped'])} file(s) skipped:\n  " + "\n  ".join(report["skipped"][:50]))
    if a.preview:
        render_previews(out_dir, manifest, a.preview)
    return report


def render_previews(out_dir, manifest, per_task):
    """PNG of a few recordings per task, to check orientation (writing must be upright, not mirrored)."""
    import cv2

    from .preprocessing import read_trajectory_csv

    os.makedirs(os.path.join(out_dir, "preview"), exist_ok=True)
    for task in sorted({m["task"] for m in manifest}):
        for m in [m for m in manifest if m["task"] == task][:per_task]:
            tr = read_trajectory_csv(os.path.join(out_dir, m["file"]))
            d = tr.pen > 0
            x0, y0 = tr.x[d].min(), tr.y[d].min()
            s = 760.0 / max(tr.x[d].max() - x0, tr.y[d].max() - y0, 1e-9)
            img = np.full((800, 800, 3), 255, np.uint8)
            pts = np.stack([(tr.x - x0) * s + 20, (tr.y - y0) * s + 20], 1).astype(int)
            for i in range(1, len(pts)):
                if d[i] and d[i - 1]:
                    cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), (40, 24, 16), 2)
            cv2.imwrite(os.path.join(out_dir, "preview", os.path.basename(m["file"])[:-4] + ".png"), img)
    print(f"[IMPORT] previews -> {os.path.join(out_dir, 'preview')} (check the writing is upright)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=config.EMOTHAW_RAW_DIR)
    p.add_argument("--out", default=config.EMOTHAW_PROCESSED_DIR)
    p.add_argument("--labels", help="the DASS label table (default: the one table found under --raw)")
    p.add_argument("--writer-col")
    p.add_argument("--anxiety-col")
    p.add_argument("--depression-col")
    p.add_argument("--stress-col")
    p.add_argument("--label-encoding", default="auto", choices=["auto", "dass", "severity", "binary"])
    p.add_argument("--pressure-max", type=float, help=f"raw pressure full scale (default {SVC_PRESSURE_MAX:g})")
    p.add_argument("--azimuth-max", type=float, help=f"raw azimuth full scale (default {SVC_AZIMUTH_MAX:g})")
    p.add_argument("--altitude-max", type=float, help=f"raw altitude full scale (default {SVC_ALTITUDE_MAX:g})")
    p.add_argument("--preview", type=int, default=0, help="render N recordings per task to PNG")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    try:
        run(parse_args(argv))
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

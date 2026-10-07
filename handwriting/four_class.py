"""
handwriting/four_class.py - data for the 4-class (Happy / Neutral / Sad / Angry)
handwriting pipeline. Shared by the seed generator (handwriting.generate_seed_data)
and the real-dataset importer (handwriting.import_4class), so both datasets end
up in the SAME processed layout and go through the same preprocessing, model,
checkpoint path and API:

    data/handwriting/<dataset>/
        sources/                  the examples exactly as provided (never modified)
            happy/ neutral/ sad/ angry/     one file per sample, label = folder
            sources.csv           optional: file, label, writer_id, reference, notes
        processed/                written here
            samples/<label>/<source_id>__orig.csv, __aug01.csv, ...
            manifest.csv          one row per sample, with provenance (see MANIFEST_FIELDS)
            dataset.json          dataset_type, source_type, scientifically_validated, ...

Accepted source files:
    images        .png .jpg .jpeg .bmp .tif .tiff -> trajectory RECONSTRUCTED from the ink
                  (image_trajectory.py: geometry only; order / pen lifts estimated;
                  no timing, pressure or tilt)
    trajectories  .csv (standard columns x, y, t_ms, pen_down, ...), .json (browser
                  points as POSTed by the handwriting page), .svc -> RECORDED pen data

Labels are exactly Happy / Neutral / Sad / Angry. No other label is mapped onto
them here (the real-dataset importer takes an explicit --label-map).

Augmentation is CLASS-BLIND: augment() never sees the label, every source gets
the same number of copies, and all parameters come from the same distributions,
so the model cannot learn a generation rule instead of the handwriting.
"""

import csv
import datetime
import hashlib
import json
import os
import re

import numpy as np

from . import config
from .preprocessing import (STANDARD_COLUMNS, Trajectory, from_browser_points, read_svc,
                            read_trajectory_csv, stroke_ids)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif")
TRAJECTORY_EXTS = (".csv", ".json", ".svc")
LABEL_OF = {c.lower(): c for c in config.FOUR_CLASS_LABELS}
MANIFEST_FIELDS = ["sample_id", "file", "label", "sample_source", "group_id", "writer_id", "source_sample",
                   "source_file",
                   "source_sha256", "input_kind", "trajectory_origin", "t_is_time", "timing_available",
                   "pressure_available", "stroke_order_available", "synthetic",
                   "augmentation", "dataset", "dataset_type", "source_type", "validated", "reference"]

# Augmentation parameter ranges - identical for every class.
AUGMENT_RANGES = {
    "rotation_deg": (-2.0, 2.0),       # small: baseline direction is itself a graphology cue
    "slant_shear": (-0.05, 0.05),      # ~ +/-3 deg of slant: slant is also a graphology cue
    "aspect": (0.95, 1.05),            # width/height ratio
    "jitter": (0.0, 0.03),             # smooth stroke-level displacement, x median stroke height
    "time_scale": (0.9, 1.1),          # speed variation - ONLY for recorded trajectories (real timing)
}


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_csv_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _inside(a, b):
    a, b = os.path.realpath(a), os.path.realpath(b)
    return a == b or a.startswith(b + os.sep)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def read_json_points(path):
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return from_browser_points(data.get("points") if isinstance(data, dict) else data)


def read_source(path):
    """-> (Trajectory, input_kind, trajectory_origin, t_is_time, info)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        from .image_trajectory import image_to_trajectory
        traj, info = image_to_trajectory(path)
        return traj, "image", "reconstructed_from_image", False, info
    if ext == ".csv":
        return read_trajectory_csv(path), "trajectory", "recorded", True, {}
    if ext == ".json":
        return read_json_points(path), "trajectory", "recorded", True, {}
    if ext == ".svc":
        return read_svc(path), "trajectory", "recorded", True, {}
    raise ValueError(f"unsupported file type {ext}")


def scan_sources(sources_dir, label_map=None):
    """-> (entries, problems). entries: {"rel", "label", "writer_id", "reference"}.
    Label: sources.csv row if listed, else the folder name. label_map: explicit
    {dataset label: one of FOUR_CLASS_LABELS} (real datasets); without it only the
    four names themselves are accepted."""
    listed = {}
    listing = os.path.join(sources_dir, "sources.csv")
    if os.path.isfile(listing):
        with open(listing, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"file", "label"} <= set(reader.fieldnames or []):
                raise ValueError(f"{listing}: needs at least the columns file, label")
            for row in reader:
                listed[os.path.normpath(row["file"].strip())] = row
    mapping = {k.lower(): v for k, v in (label_map or {}).items()}
    entries, problems = [], []
    for d, _, files in os.walk(sources_dir):
        for f in sorted(files):
            if not f.lower().endswith(IMAGE_EXTS + TRAJECTORY_EXTS) or f == "sources.csv" or f.startswith("."):
                continue
            rel = os.path.normpath(os.path.relpath(os.path.join(d, f), sources_dir))
            row = listed.get(rel, {})
            folder = rel.split(os.sep)[0] if os.sep in rel else ""
            raw_label = (row.get("label") or folder).strip()
            label = mapping.get(raw_label.lower()) if mapping else LABEL_OF.get(raw_label.lower())
            if label not in config.FOUR_CLASS_LABELS:
                problems.append(f"{rel}: label {raw_label!r} is not one of {config.FOUR_CLASS_LABELS}"
                                + (" (nor in --label-map)" if mapping else "") + " - skipped")
                continue
            entries.append({"rel": rel, "label": label, "raw_label": raw_label,
                            "writer_id": (row.get("writer_id") or "").strip(),
                            "reference": (row.get("reference") or "").strip()})
    missing = sorted(set(listed) - {e["rel"] for e in entries} - {p.split(":")[0] for p in problems})
    problems += [f"{m}: listed in sources.csv but not found" for m in missing]
    entries.sort(key=lambda e: (config.FOUR_CLASS_LABELS.index(e["label"]), e["rel"]))
    return entries, problems


# ---------------------------------------------------------------------------
# Class-blind augmentation
# ---------------------------------------------------------------------------
def sample_params(rng, has_timing):
    p = {k: float(rng.uniform(*v)) for k, v in AUGMENT_RANGES.items() if k != "time_scale"}
    p["time_scale"] = float(rng.uniform(*AUGMENT_RANGES["time_scale"])) if has_timing else None
    p["jitter_phases"] = [float(v) for v in rng.uniform(0, 2 * np.pi, 6)]
    p["jitter_freqs"] = [float(v) for v in rng.uniform(1.0, 4.0, 6)]
    p["jitter_weights"] = [float(v) for v in rng.normal(0, 1, 6)]
    return p


def augment(traj, params):
    """Geometric (and, for real timing, temporal) perturbation of one trajectory.
    Deliberately takes no label."""
    down = traj.pen > 0
    cx, cy = traj.x[down].mean(), traj.y[down].mean()
    x, y = traj.x - cx, traj.y - cy
    x = x + params["slant_shear"] * (-y)                        # y points down: shear the upright axis
    a = np.deg2rad(params["rotation_deg"])
    x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    x, y = x * np.sqrt(params["aspect"]), y / np.sqrt(params["aspect"])
    runs = np.split(np.arange(len(traj)), np.flatnonzero(np.diff(down.astype(int))) + 1)
    heights = [np.ptp(traj.y[r]) for r in runs if down[r[0]] and len(r) > 1]
    amp = params["jitter"] * (float(np.median(heights)) if heights else 0.0)
    u = np.linspace(0.0, 1.0, len(traj))                          # position along the whole writing
    w, f, ph = map(np.asarray, (params["jitter_weights"], params["jitter_freqs"], params["jitter_phases"]))
    x = x + amp * (np.sin(2 * np.pi * np.outer(u, f[:3]) + ph[:3]) * w[:3]).sum(1) / 3.0
    y = y + amp * (np.sin(2 * np.pi * np.outer(u, f[3:]) + ph[3:]) * w[3:]).sum(1) / 3.0
    t = traj.t if params["time_scale"] is None else traj.t[0] + (traj.t - traj.t[0]) * params["time_scale"]
    return Trajectory(x + cx, y + cy, t, traj.pen, traj.pressure, traj.altitude, traj.azimuth)


def rng_for(seed, source_sha256, k):
    """Independent, order-free RNG stream per (seed, source file CONTENT, copy).
    Keyed on the file's hash - not its name or label - so relabelling a sample
    gives byte-identical augmentations."""
    return np.random.default_rng([seed, int(source_sha256[:12], 16), k])


# ---------------------------------------------------------------------------
# Writing the processed layout
# ---------------------------------------------------------------------------
def write_standard_csv(path, traj, t_is_time):
    sid, _ = stroke_ids(traj.pen)
    fmt = lambda v, nd: "" if not np.isfinite(v) else round(float(v), nd)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(STANDARD_COLUMNS)
        for i in range(len(traj)):
            w.writerow([round(float(traj.x[i]), 3), round(float(traj.y[i]), 3), round(float(traj.t[i] * 1000.0), 3),
                        int(traj.pen[i] > 0), fmt(traj.pressure[i], 5), fmt(traj.altitude[i], 4),
                        fmt(traj.azimuth[i], 4), int(sid[i])])


def _json_meta(path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def build(profile_name, sources_dir, out_dir, *, augment_per_source, seed, force, source_type,
          validated, label_map=None, allow_images=True, require_writer_ids=False, statement=None,
          blocked=None, sample_source="graphology"):
    """Read every source, write originals + class-blind augmented copies. -> report dict.
    sources_dir: one folder, or a list of (folder, sample_source) pairs - e.g. the
    graphology sources/ and the human canvas folder; every processed row records its
    sample_source, so origins are never mixed invisibly. Missing extra folders are
    skipped. A JSON trajectory's own "writer_id" is used when sources.csv gives none.
    blocked: {sha256: reason} of files that must not be used (e.g. graphology
    candidates that were not approved); they are skipped and listed."""
    profile = config.FOUR_CLASS_DATASETS[profile_name]
    dirs = [(sources_dir, sample_source)] if isinstance(sources_dir, str) else list(sources_dir)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    problems, entries = [], []
    for d, origin in dirs:
        if _inside(out_dir, d) or _inside(d, out_dir):
            raise SystemExit("the output folder and the sources folder must be separate (sources are never modified)")
    existing = [(d, o) for d, o in dirs if os.path.isdir(d)]
    if not existing:
        raise FileNotFoundError(f"Sources folder not found: {dirs[0][0]}\nCreate it with one sub-folder per "
                                f"label ({', '.join(c.lower() for c in config.FOUR_CLASS_LABELS)}); see data/README.md.")
    if os.path.exists(manifest_path) and not force:
        raise SystemExit(f"{manifest_path} exists; pass --force to rebuild")
    for d, origin in dirs:
        if not os.path.isdir(d):
            problems.append(f"{d}: folder not found ({origin}) - nothing from it")
            continue
        found, probs = scan_sources(d, label_map)
        problems += probs
        prefix = "" if len(dirs) == 1 else os.path.basename(os.path.normpath(d)) + "/"
        for e in found:
            e.update(dir=d, sample_source=origin, shown=prefix + e["rel"].replace(os.sep, "/"))
            entries.append(e)
    if require_writer_ids:
        lacking = [e["rel"] for e in entries if not e["writer_id"]]
        if lacking:
            raise ValueError(f"{len(lacking)} source(s) have no writer_id in sources.csv (needed for a "
                             f"writer-independent split), e.g. {lacking[:5]}")
    loaded, seen = [], {}
    counters = {}
    for e in entries:
        path = os.path.join(e["dir"], e["rel"])
        if e["rel"].lower().endswith(".json"):
            meta_json = _json_meta(path)
            if meta_json.get("label") and LABEL_OF.get(str(meta_json["label"]).lower()) != e["label"]:
                problems.append(f"{e['shown']}: file says label {meta_json['label']!r} but sits under "
                                f"{e['label']} - skipped")
                continue
            if not e["writer_id"] and isinstance(meta_json.get("writer_id"), str):
                e["writer_id"] = meta_json["writer_id"].strip()
        if not allow_images and e["rel"].lower().endswith(IMAGE_EXTS):
            problems.append(f"{e['rel']}: image sources are not enabled for {profile_name} (--allow-images)")
            continue
        digest = sha256_file(path)
        if blocked and digest in blocked:
            problems.append(f"{e['rel']}: {blocked[digest]} - skipped")
            continue
        if digest in seen:
            problems.append(f"{e['rel']}: identical to {seen[digest]} - skipped (would leak across splits)")
            continue
        try:
            traj, kind, origin, t_is_time, info = read_source(path)
        except Exception as exc:
            problems.append(f"{e['rel']}: {exc}")
            continue
        if not (traj.pen > 0).any():
            problems.append(f"{e['rel']}: no pen-down points")
            continue
        seen[digest] = e["rel"]
        k = counters[e["label"]] = counters.get(e["label"], 0) + 1
        loaded.append(dict(e, source_id=f"{e['label'].lower()}_{k:03d}", sha256=digest, traj=traj, kind=kind,
                           origin=origin, t_is_time=t_is_time, info=info))
    if not loaded:
        raise SystemExit("Nothing usable in " + ", ".join(d for d, _ in dirs)
                         + ("\n  " + "\n  ".join(problems[:30]) if problems else ""))

    if os.path.exists(manifest_path):                          # --force: remove what we wrote before
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                p = os.path.join(out_dir, row["file"])
                if os.path.isfile(p) and _inside(p, out_dir):
                    os.remove(p)
    validated_str = "unknown" if validated is None else str(bool(validated)).lower()
    manifest = []
    for s in loaded:
        folder = os.path.join(out_dir, "samples", s["label"].lower())
        os.makedirs(folder, exist_ok=True)
        human = s["sample_source"] == "human_canvas"
        common = {"label": s["label"], "sample_source": s["sample_source"], "writer_id": s["writer_id"],
                  "group_id": f"writer:{s['writer_id']}" if s["writer_id"] else f"source:{s['source_id']}",
                  "source_sample": s["source_id"], "source_file": s["shown"],
                  "source_sha256": s["sha256"], "input_kind": s["kind"], "trajectory_origin": s["origin"],
                  "t_is_time": int(s["t_is_time"]),
                  "timing_available": str(bool(s["t_is_time"])).lower(),
                  "pressure_available": str(bool(np.isfinite(s["traj"].pressure).any())).lower(),
                  "stroke_order_available": "recorded" if s["origin"] == "recorded" else "estimated",
                  "dataset": profile_name,
                  "dataset_type": profile["dataset_type"],
                  "source_type": "human_canvas" if human else source_type,
                  "validated": "false" if human else validated_str, "reference": s["reference"]}
        variants = [("orig", s["traj"], None)]
        for k in range(1, augment_per_source + 1):
            params = sample_params(rng_for(seed, s["sha256"], k), s["t_is_time"])
            variants.append((f"aug{k:02d}", augment(s["traj"], params), params))
        for tag, traj, params in variants:
            rel = f"samples/{s['label'].lower()}/{s['source_id']}__{tag}.csv"
            write_standard_csv(os.path.join(out_dir, rel), traj, s["t_is_time"])
            manifest.append(dict(common, sample_id=f"{s['source_id']}__{tag}", file=rel,
                                 synthetic=int(params is not None),
                                 augmentation="" if params is None else json.dumps(
                                     {k: (round(v, 5) if isinstance(v, float) else v) for k, v in params.items()
                                      if not k.startswith("jitter_")} | {"jitter_components": 6})))
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(manifest)

    per_source = {o: {c: sum(s["label"] == c and s["sample_source"] == o for s in loaded)
                      for c in config.FOUR_CLASS_LABELS} for o in sorted({s["sample_source"] for s in loaded})}
    per_class = {c: {"sources": sum(s["label"] == c for s in loaded),
                     "samples": sum(m["label"] == c for m in manifest)} for c in config.FOUR_CLASS_LABELS}
    reconstructed = [s for s in loaded if s["kind"] == "image"]
    meta = {
        "dataset": profile_name, "dataset_type": profile["dataset_type"], "source_type": source_type,
        "scientifically_validated": validated, "model_kind": profile["model_kind"],
        "labels": config.FOUR_CLASS_LABELS, "statement": statement,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "sources_dir": sources_dir, "sources": len(loaded), "samples": len(manifest),
        "per_class": per_class, "per_sample_source": per_source,
        "writers_per_sample_source": {o: len({s["writer_id"] for s in loaded if s["sample_source"] == o
                                              and s["writer_id"]}) for o in per_source},
        "label_map": label_map,
        "writer_ids_known": sum(bool(s["writer_id"]) for s in loaded),
        "generator": {"seed": seed, "augment_per_source": augment_per_source,
                      "augmentation_ranges": AUGMENT_RANGES, "class_blind": True},
        "reconstructed_from_images": {
            "sources": len(reconstructed),
            "reconstructed": reconstructed[0]["info"]["reconstructed"] if reconstructed else [],
            "unavailable": reconstructed[0]["info"]["unavailable"] if reconstructed else [],
            "t_meaning": "sequence index (not time)" if reconstructed else None},
        "skipped": problems,
    }
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta, manifest, loaded


def render_previews(out_dir, loaded, per_class):
    """PNG of the ORIGINAL trajectories (reconstructed ones colour-coded by estimated order)."""
    import cv2

    os.makedirs(os.path.join(out_dir, "preview"), exist_ok=True)
    for c in config.FOUR_CLASS_LABELS:
        for s in [s for s in loaded if s["label"] == c][:per_class]:
            tr = s["traj"]
            d = tr.pen > 0
            x0, y0 = tr.x[d].min(), tr.y[d].min()
            k = 760.0 / max(np.ptp(tr.x[d]), np.ptp(tr.y[d]), 1e-9)
            img = np.full((800, 800, 3), 255, np.uint8)
            pts = np.stack([(tr.x - x0) * k + 20, (tr.y - y0) * k + 20], 1).astype(int)
            n = len(pts)
            for i in range(1, n):
                if d[i] and d[i - 1]:
                    g = int(255 * i / n)                    # blue -> red = estimated/recorded order
                    cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), (255 - g, 0, g), 2)
            cv2.imwrite(os.path.join(out_dir, "preview", f"{s['source_id']}.png"), img)
    print(f"[4CLASS] previews -> {os.path.join(out_dir, 'preview')} (colour blue -> red = stroke order)")


# ---------------------------------------------------------------------------
# Loading for training
# ---------------------------------------------------------------------------
def load_dataset(data_dir, heads):
    """processed layout -> (samples, report). samples: {"writer" (= split group), "path",
    "sha256" (of the SOURCE file: all copies of a source share it), "traj", "y",
    "synthetic", "label", "task"}."""
    manifest = os.path.join(data_dir, "manifest.csv")
    meta_path = os.path.join(data_dir, "dataset.json")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"{manifest} not found. Build it first: `python -m handwriting.generate_seed_data` "
                                f"(seed) or `python -m handwriting.import_4class` (real data); see data/README.md.")
    with open(meta_path) as f:
        meta = json.load(f)
    head = heads[0]
    samples, unreadable = [], []
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            if row["label"] not in head["classes"]:
                raise ValueError(f"{manifest} line {line}: label {row['label']!r} is not one of {head['classes']}")
            path = os.path.join(data_dir, row["file"])
            try:
                traj = read_trajectory_csv(path)
            except Exception as exc:
                unreadable.append(f"{path}: {exc}")
                continue
            samples.append({"writer": row["group_id"], "task": row["sample_id"], "path": path,
                            "sha256": row["source_sha256"], "traj": traj, "label": row["label"],
                            "synthetic": row["synthetic"] == "1", "source": row["source_sample"],
                            "sample_source": row.get("sample_source") or "graphology",
                            "y": {head["name"]: head["classes"].index(row["label"])}})
    report = {"dataset_type": meta.get("dataset_type"), "source_type": meta.get("source_type"),
              "scientifically_validated": meta.get("scientifically_validated"),
              "recordings": len(samples) + len(unreadable), "usable_samples": len(samples),
              "sources": len({s["source"] for s in samples}), "groups": len({s["writer"] for s in samples}),
              "synthetic_samples": sum(s["synthetic"] for s in samples),
              "writer_ids_known": meta.get("writer_ids_known"), "unreadable_files": unreadable}
    return samples, report, meta

"""
handwriting/import_4class.py - import a REAL emotion-labelled handwriting dataset
into the 4-class pipeline, replacing the seed data without touching the model,
the API or the page.

    python -m handwriting.import_4class --source "Dataset name, citation" \
        --label-map "joy=Happy,neutral=Neutral,sadness=Sad,anger=Angry" --labels-verified
    python -m handwriting.train --dataset real_4class      # -> models/handwriting_4class.pt

Input:  data/handwriting/real_4class/sources/ (never modified), laid out like the
        seed sources: one folder per label, or sources.csv with file, label,
        writer_id (writer_id is REQUIRED for every file: the split must be
        writer-independent). Trajectory files (.csv / .json / .svc); images only
        with --allow-images (their trajectories are then reconstructed and marked so).
Output: data/handwriting/real_4class/processed/ - the same layout as the seed data.

Labels must be mapped explicitly (--label-map) unless they already are Happy /
Neutral / Sad / Angry; unmapped labels are skipped and listed, never guessed.
No augmentation by default.
"""

import argparse
import os
import sys

from . import config
from .four_class import build, render_previews


def parse_label_map(text):
    if not text:
        return None
    out = {}
    for pair in text.split(","):
        if "=" not in pair:
            raise ValueError(f"--label-map entry {pair!r} is not 'dataset_label=Class'")
        k, v = (s.strip() for s in pair.split("=", 1))
        if v not in config.FOUR_CLASS_LABELS:
            raise ValueError(f"--label-map target {v!r} is not one of {config.FOUR_CLASS_LABELS}")
        out[k] = v
    return out


def run(a):
    profile = config.FOUR_CLASS_DATASETS[a.dataset]
    sources = os.path.abspath(a.sources or os.path.join(profile["dir"], "sources"))
    out = os.path.abspath(a.out or os.path.join(profile["dir"], "processed"))
    meta, manifest, loaded = build(
        a.dataset, sources, out, augment_per_source=a.augment, seed=a.seed, force=a.force,
        source_type=a.source, validated=bool(a.labels_verified), label_map=parse_label_map(a.label_map),
        allow_images=a.allow_images, require_writer_ids=True, statement=a.statement)
    print(f"[4CLASS] {meta['sources']} sources from {meta['writer_ids_known']} writer-labelled files -> "
          f"{meta['samples']} samples -> {out}")
    for c, n in meta["per_class"].items():
        print(f"[4CLASS]   {c:8s}: {n['sources']} sources, {n['samples']} samples")
    print(f"[4CLASS] labels verified: {bool(a.labels_verified)}")
    if meta["skipped"]:
        print(f"[4CLASS] {len(meta['skipped'])} file(s) skipped:\n  " + "\n  ".join(meta["skipped"][:50]))
    if a.preview:
        render_previews(out, loaded, a.preview)
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="real_4class", choices=["real_4class"])
    p.add_argument("--source", required=True, help="what the data is (name / citation) - stored as source_type")
    p.add_argument("--labels-verified", action="store_true",
                   help="the labels come from a validated protocol (stored as scientifically_validated)")
    p.add_argument("--statement", help="free-text description stored in dataset.json")
    p.add_argument("--label-map", help='e.g. "joy=Happy,neutral=Neutral,sadness=Sad,anger=Angry"')
    p.add_argument("--allow-images", action="store_true")
    p.add_argument("--augment", type=int, default=config.FOUR_CLASS_DATASETS["real_4class"]["augment"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources")
    p.add_argument("--out")
    p.add_argument("--preview", type=int, default=0)
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

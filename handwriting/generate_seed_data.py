"""
handwriting/generate_seed_data.py - build the 4-class SEED dataset
(Happy / Neutral / Sad / Angry) from graphology reference examples.

    python -m handwriting.generate_seed_data                  # sources/ -> processed/, 20 copies per source
    python -m handwriting.generate_seed_data --augment 0      # originals only
    python -m handwriting.generate_seed_data --preview 3      # render 3 originals per class to PNG

IMPORTANT: graphology is not a scientifically validated source of emotion
labels. These samples are heuristic SEED data used to build and test the
4-class pipeline; they will be replaced by a real emotion-labelled handwriting
dataset (python -m handwriting.import_4class --dataset real_4class). Every
output file is marked dataset_type=seed, source_type=graphology_reference,
scientifically_validated=false.

Input:  data/handwriting/seed_4class/sources/{happy,neutral,sad,angry}/  (never modified)
        images are converted by an explicit, documented reconstruction
        (handwriting/image_trajectory.py): geometry is real, stroke order and pen
        lifts are estimated, timing / pressure / tilt are unavailable and are
        NOT invented.
        data/handwriting/seed_4class/human/{happy,neutral,sad,angry}/  participant canvas samples
        (Collect tab on /handwriting), included by default (--include)
Output: data/handwriting/seed_4class/processed/  (manifest.csv, dataset.json, samples/)
        every row has sample_source = graphology | human_canvas, and synthetic = 0/1

Reproducible: fixed --seed; each augmented copy's RNG depends only on
(seed, source file content, copy number). Augmentation never sees the label.
"""

import argparse
import os
import sys

from . import config
from .four_class import _inside, build, read_csv_rows, render_previews

PROFILE = "seed_4class"


def run(a):
    root = config.FOUR_CLASS_DATASETS[PROFILE]["dir"]
    sources = os.path.abspath(a.sources or os.path.join(root, "sources"))
    human = os.path.abspath(a.human or os.path.join(os.path.dirname(sources), "human"))
    out = os.path.abspath(a.out or os.path.join(root, "processed"))
    blocked = unapproved_candidates(os.path.dirname(sources), sources)
    dirs = []
    if a.include in ("all", "graphology"):
        dirs.append((sources, "graphology"))
    if a.include == "human_canvas" or (a.include == "all" and os.path.isdir(human)):
        dirs.append((human, "human_canvas"))
    meta, manifest, loaded = build(
        PROFILE, dirs, out, augment_per_source=a.augment, seed=a.seed, force=a.force,
        source_type="graphology_reference", validated=False, allow_images=True, require_writer_ids=False,
        statement=config.SEED_DISCLAIMER, blocked=blocked)
    print(f"[SEED] {meta['sources']} source samples -> {meta['samples']} samples "
          f"({a.augment} augmented copies each) -> {out}")
    for c, n in meta["per_class"].items():
        print(f"[SEED]   {c:8s}: {n['sources']} sources, {n['samples']} samples")
    for origin, per in meta["per_sample_source"].items():
        print(f"[SEED]   origin {origin:12s}: {per} originals, "
              f"{meta['writers_per_sample_source'][origin]} writer id(s)")
    r = meta["reconstructed_from_images"]
    if r["sources"]:
        print(f"[SEED] {r['sources']} source(s) were images: trajectories RECONSTRUCTED "
              f"({', '.join(r['reconstructed'])} estimated; {', '.join(r['unavailable'])} unavailable)")
    if meta["writer_ids_known"] < meta["sources"]:
        print(f"[SEED] {meta['sources'] - meta['writer_ids_known']} source(s) have no writer_id: each is its own "
              f"split group, so two examples by the same writer could land in different splits")
    if blocked:
        print(f"[SEED] {len(blocked)} collected graphology image(s) are not approved and can never be used")
    if meta["skipped"]:
        print(f"[SEED] {len(meta['skipped'])} file(s) skipped:\n  " + "\n  ".join(meta["skipped"][:50]))
    print(f"[SEED] NOTE: {config.SEED_DISCLAIMER}")
    if a.preview:
        render_previews(out, loaded, a.preview)
    return meta


def unapproved_candidates(root, sources):
    """Guard: collected graphology images reach training ONLY via review approval.
    Refuses the candidates/ and rejected/ folders as --sources, and returns
    {sha256: reason} for every collected image that is not approved, so a copy
    placed in sources/ by hand is skipped."""
    for folder in ("candidates", "rejected"):
        if _inside(sources, os.path.join(root, folder)):
            raise SystemExit(f"--sources is inside {folder}/: only reviewed, approved images in sources/ may be "
                             f"used (python -m handwriting.review_graphology)")
    rows = read_csv_rows(os.path.join(root, "candidates.csv"))
    return {r["sha256"]: f"graphology candidate {r['id']} is '{r['status']}', not approved"
            for r in rows if r.get("status") not in ("approved", "auto_approved") and r.get("sha256")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", help="graphology sources (default: data/handwriting/seed_4class/sources)")
    p.add_argument("--human", help="human canvas samples (default: <sources>/../human)")
    p.add_argument("--include", default="all", choices=["all", "graphology", "human_canvas"],
                   help="which origins to build from; each row keeps its sample_source either way")
    p.add_argument("--out", help="default: data/handwriting/seed_4class/processed")
    p.add_argument("--augment", type=int, default=config.FOUR_CLASS_DATASETS[PROFILE]["augment"],
                   help="class-blind augmented copies per source sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", type=int, default=0, help="render N original samples per class to PNG")
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

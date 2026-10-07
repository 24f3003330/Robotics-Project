"""
python -m body_vitals.prepare_wesad --data /path/to/WESAD

Reads every WESAD subject, keeps baseline / stress / amusement, cuts windows
(default 90 s, 45 s step - the deployed model's; --window-sec / --step-sec for others)
inside each labelled segment, extracts all physiological features and writes
data/body_vitals/wesad_features_w<W>_s<S>.csv plus preprocessing_report_w<W>_s<S>.json
(with every discarded window and why). The WESAD path can also come from $WESAD_DIR.
"""

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from . import config
from .dataset import (SubjectSkipped, find_subjects, process_subject, subject_sort_key, write_features,
                      write_report)

DISCARD_WARN_FRACTION = 0.10


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.environ.get(config.WESAD_DIR_ENV),
                   help=f"WESAD folder containing S2/, S3/, ... (default: ${config.WESAD_DIR_ENV})")
    p.add_argument("--out", help="feature CSV to write (default: data/body_vitals/wesad_features_w<W>_s<S>.csv)")
    p.add_argument("--report", help="preprocessing report JSON (default: next to the CSV)")
    p.add_argument("--subjects", nargs="*", help="only these subject ids (default: all found)")
    p.add_argument("--window-sec", type=float, default=config.MODEL_WINDOW_SEC)
    p.add_argument("--step-sec", type=float, default=config.MODEL_STEP_SEC)
    p.add_argument("--jobs", type=int, default=2,
                   help="subjects processed in parallel (each needs ~2-3 GB RAM while loading)")
    return p.parse_args(argv)


def _work(item):
    data_dir, subject, window_sec, step_sec = item
    try:
        return subject, process_subject(data_dir, subject, window_sec, step_sec), None
    except SubjectSkipped as exc:
        return subject, None, str(exc)


def run(args):
    args.out = args.out or config.features_csv(args.window_sec, args.step_sec)
    args.report = args.report or config.preprocessing_report(args.window_sec, args.step_sec)
    if not args.data:
        sys.exit(f"Pass --data /path/to/WESAD or set {config.WESAD_DIR_ENV}.")
    subjects = args.subjects or find_subjects(args.data)
    if not subjects:
        sys.exit(f"No subject folders (S2, S3, ...) found in {args.data}")
    print(f"[prepare_wesad] {len(subjects)} subject folders in {args.data}: {', '.join(subjects)}")
    print(f"[prepare_wesad] window {args.window_sec:g} s, step {args.step_sec:g} s; "
          f"signals: chest ECG, EDA, Resp @ {config.WESAD_CHEST_FS} Hz; classes {config.CLASSES}")
    t0 = time.time()
    items = [(args.data, s, args.window_sec, args.step_sec) for s in subjects]
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for res in pool.map(_work, items):
                results.append(res)
                _print_subject(*res)
    else:
        for item in items:
            res = _work(item)
            results.append(res)
            _print_subject(*res)

    rows, summaries, skipped = [], [], {}
    for subject, out, error in sorted(results, key=lambda r: subject_sort_key(r[0])):
        if error:
            skipped[subject] = error
            continue
        rows.extend(out[0])
        summaries.append(out[1])
    if not rows:
        sys.exit("No usable windows - nothing written.")
    write_features(rows, args.out)

    created = sum(sum(s["windows_created"].values()) for s in summaries)
    discarded = sum(s["windows_discarded"] for s in summaries)
    reasons = Counter()
    for s in summaries:
        reasons.update(s["discard_reasons"])
    class_dist = Counter(r["class"] for r in rows)
    report = {
        "dataset": "WESAD", "data_dir_name": os.path.basename(os.path.normpath(args.data)),
        "feature_columns": config.ALL_FEATURE_NAMES, "feature_sets": config.FEATURE_SETS,
        "window_sec": args.window_sec, "step_sec": args.step_sec,
        "subjects_processed": [s["subject"] for s in summaries], "subjects_skipped": skipped,
        "windows_created": created, "windows_kept": len(rows), "windows_discarded": discarded,
        "discard_reasons": dict(reasons), "class_distribution": {c: class_dist.get(c, 0) for c in config.CLASSES},
        "per_subject": summaries, "seconds": round(time.time() - t0, 1),
    }
    write_report(report, args.report)

    print("\n================ WESAD preprocessing summary ================")
    print(f"Subjects processed : {len(summaries)}  ({', '.join(report['subjects_processed'])})")
    print(f"Subjects skipped   : {len(skipped)}" + "".join(f"\n    {k}: {v}" for k, v in skipped.items()))
    print(f"Windows created    : {created}")
    print(f"Windows kept       : {len(rows)}")
    print(f"Windows discarded  : {discarded} ({discarded / max(created, 1):.1%})")
    for reason, n in reasons.most_common():
        print(f"    {reason:<22} {n}")
    print("Class distribution : " + ", ".join(f"{c}={class_dist.get(c, 0)}" for c in config.CLASSES))
    print(f"Features extracted : {len(config.ALL_FEATURE_NAMES)} ({', '.join(config.ALL_FEATURE_NAMES)})")
    print(f"Wrote {args.out}\n      {args.report}  ({report['seconds']} s)")
    if discarded / max(created, 1) > DISCARD_WARN_FRACTION:
        print(f"WARNING: more than {DISCARD_WARN_FRACTION:.0%} of windows were discarded - "
              f"check the discard reasons in {args.report}")
    return report


def _print_subject(subject, out, error):
    if error:
        print(f"  {subject}: SKIPPED - {error}", flush=True)
        return
    s = out[1]
    print(f"  {subject}: kept {s['windows_kept']}, discarded {s['windows_discarded']} "
          f"{s['discard_reasons'] or ''}", flush=True)


if __name__ == "__main__":
    run(parse_args())

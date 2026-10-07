"""
handwriting/dataset.py - EMOTHAW loading and the writer-independent split.

Reads the processed layout written by `python -m handwriting.import_emothaw`
(see data/README.md):

    data/emothaw/processed/manifest.csv     one row per recording (writer_id, task, file, sha256, ...)
    data/emothaw/processed/labels.csv       one row per writer: anxiety, depression, stress
                                            ("normal" / "elevated", as converted by the importer)
    data/emothaw/processed/samples/...      standard per-recording CSVs

EMOTHAW labels belong to the WRITER (one DASS questionnaire per participant),
so every recording of a writer carries the same three labels, and the split
must keep all of a writer's recordings together.
"""

import csv
import os
import random
from collections import defaultdict

import numpy as np

from . import config
from .preprocessing import read_trajectory_csv


def load_labels(path, heads):
    """processed labels.csv -> {writer_id (int): {head_name: class_index}}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. Run `python -m handwriting.import_emothaw` first "
                                f"(see data/README.md).")
    labels = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ["writer_id"] + [h["name"] for h in heads] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; found {reader.fieldnames}")
        for line, row in enumerate(reader, start=2):
            entry = {}
            for h in heads:
                value = row[h["name"]].strip().lower()
                if value not in h["classes"]:
                    raise ValueError(f"{path} line {line}: {h['name']} = {row[h['name']]!r} is not one of "
                                     f"{h['classes']}")
                entry[h["name"]] = h["classes"].index(value)
            labels[int(row["writer_id"])] = entry
    return labels


def writer_split(writer_keys, fractions=(0.70, 0.15, 0.15), seed=0):
    """Writer-independent split. writer_keys: {writer: stratification key} - the
    writer's anxiety/depression/stress label combination.

    Every writer goes to exactly one of train / val / test, so none of their
    recordings can appear in two splits.
    """
    writers = list(writer_keys)
    groups = defaultdict(list)
    for w in sorted(writers):
        groups[writer_keys[w]].append(w)
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    carry = np.zeros(3)                      # fractional remainders spread across groups
    for key in sorted(groups):
        members = groups[key][:]
        rng.shuffle(members)
        want = np.array(fractions) * len(members) + carry
        counts = np.floor(want).astype(int)
        # hand leftover writers to the splits with the largest remainders
        for i in np.argsort(-(want - counts))[: len(members) - counts.sum()]:
            counts[i] += 1
        carry = want - counts
        split["train"] += members[: counts[0]]
        split["val"] += members[counts[0]: counts[0] + counts[1]]
        split["test"] += members[counts[0] + counts[1]:]
    for name in split:
        split[name].sort()
    a, b, c = (set(split[k]) for k in ("train", "val", "test"))
    assert not (a & b or a & c or b & c), "writer appears in more than one split"
    assert a | b | c == set(writers)
    return split


def load_emothaw(data_dir, heads, tasks):
    """-> (samples, report). samples: {"writer", "task", "session", "path", "sha256", "traj",
    "y": {head: idx}}. Recordings without labels, and labels without recordings, are reported."""
    manifest = os.path.join(data_dir, "manifest.csv")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"{manifest} not found.\nPut the EMOTHAW dataset, exactly as received, in "
            f"{config.EMOTHAW_RAW_DIR} and run `python -m handwriting.import_emothaw` (see data/README.md).")
    labels = load_labels(os.path.join(data_dir, "labels.csv"), heads)
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if int(r["task"]) in set(tasks)]
    if not rows:
        raise FileNotFoundError(f"{manifest}: no recordings for tasks {sorted(tasks)}")
    samples, unlabeled, unreadable = [], set(), []
    for r in rows:
        writer = int(r["writer_id"])
        if writer not in labels:
            unlabeled.add(writer)
            continue
        path = os.path.join(data_dir, r["file"])
        try:
            traj = read_trajectory_csv(path)
        except Exception as exc:  # a corrupt file is skipped, not fatal
            unreadable.append(f"{path}: {exc}")
            continue
        samples.append({"writer": writer, "task": int(r["task"]), "session": int(r["session"]),
                        "path": path, "sha256": r.get("sha256", ""), "traj": traj, "y": labels[writer]})
    report = {"recordings": len(rows), "usable_samples": len(samples),
              "writers_with_labels_and_recordings": len({s["writer"] for s in samples}),
              "writers_without_labels": sorted(unlabeled),
              "labeled_writers_without_recordings": sorted(set(labels) - {int(r["writer_id"]) for r in rows}),
              "unreadable_files": unreadable}
    return samples, report


def check_no_leakage(sets):
    """Raise if a writer, a file or identical recording content occurs in two splits."""
    owner = {}
    for name, samples in sets.items():
        for s in samples:
            for key in (("writer", s["writer"]), ("file", s["path"]), ("sha256", s.get("sha256"))):
                if not key[1]:
                    continue
                if owner.setdefault(key, name) != name:
                    raise AssertionError(f"data leakage: {key[0]} {key[1]} is in both "
                                         f"{owner[key]} and {name}")
    return True

"""
handwriting/train.py - train the handwriting Transformer on EMOTHAW.

    python -m handwriting.import_emothaw            # once: raw EMOTHAW -> data/emothaw/processed
    python -m handwriting.train                     # Anxiety / Depression / Stress, tasks 3 and 7
    python -m handwriting.train --tasks all         # all seven EMOTHAW tasks
    python -m handwriting.train --features kinematic   # without pressure / altitude

    python -m handwriting.train --dataset seed_4class  # 4-class PROTOTYPE on graphology seed data
    python -m handwriting.train --dataset seed_4class --sample-source human_canvas   # canvas data only
    python -m handwriting.train --dataset real_4class  # 4-class model on a real labelled dataset
    (both -> models/handwriting_4class.pt, Happy / Neutral / Sad / Angry; a
    separate experiment from EMOTHAW, never mixed with it)

Protocol
  * writer-independent split (70/15/15 of WRITERS, stratified by the writer's
    anxiety/depression/stress combination); no writer, file or identical
    recording occurs in two splits - checked and reported
  * feature mean/std computed on TRAINING writers only
  * model selection (early stopping) on validation loss only
  * the test writers are evaluated once, with the selected weights
  * metrics per target (elevated = positive class): accuracy, balanced
    accuracy, precision, recall, F1, macro F1, per-class precision / recall /
    F1, confusion matrix and the majority-class baseline - per recording (all
    windows averaged) and per writer (all of a writer's recordings averaged;
    EMOTHAW labels belong to the writer)

4-class datasets: the split groups are writers when writer ids are known,
otherwise single source samples; every augmented copy stays in its source's
group, and validation / test use ORIGINAL samples only. Seed-data results
only show that the pipeline runs - they are not emotion-recognition accuracy.

Outputs: the checkpoint (default models/handwriting_transformer.pt) and
runs/handwriting/<timestamp>/results.json. Training accuracy is logged for
diagnosis only and is never reported as performance.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from . import config, four_class
from .dataset import check_no_leakage, load_emothaw, writer_split
from .model import build_model, pad_batch
from .preprocessing import HandwritingInputError, feature_matrix, prepare, windows

CHECKPOINT_FORMAT = "handwriting-transformer/1"


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def classification_metrics(y_true, y_pred, classes, majority_class, positive=None):
    """majority_class: the most frequent class among TRAINING writers (the trivial baseline).
    positive: class name treated as positive for the binary precision / recall / F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class, recalls, f1s = {}, [], []
    for i, c in enumerate(classes):
        tp, support = cm[i, i], cm[i].sum()
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                        "f1": round(float(f1), 4), "support": int(support)}
        if support:
            recalls.append(rec)
            f1s.append(f1)
    n = len(y_true)
    binary = {}
    if positive is not None:
        pc = per_class[positive]
        binary = {"positive_class": positive, "precision": pc["precision"], "recall": pc["recall"],
                  "f1": pc["f1"]}
    return {"n": int(n),
            "accuracy": round(float((y_true == y_pred).mean()), 4) if n else None,
            "balanced_accuracy": round(float(np.mean(recalls)), 4) if recalls else None,
            **binary,
            "macro_precision": round(float(np.mean([per_class[c]["precision"] for c in classes
                                                    if per_class[c]["support"]])), 4) if recalls else None,
            "macro_recall": round(float(np.mean(recalls)), 4) if recalls else None,
            "macro_f1": round(float(np.mean(f1s)), 4) if f1s else None,
            "majority_baseline_accuracy": round(float((y_true == majority_class).mean()), 4) if n else None,
            "per_class": per_class,
            "confusion_matrix": {"rows_true_cols_pred": list(classes), "matrix": cm.tolist()}}


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        if args.dataset == "emothaw":
            self.profile = None
            self.heads = config.EMOTHAW_HEADS
            self.air = config.AIR_MODE
            self.resample = "time"
        else:
            self.profile = config.FOUR_CLASS_DATASETS[args.dataset]
            self.heads = config.FOUR_CLASS_HEADS
            self.air = self.profile["air"]
            self.resample = args.resample or self.profile["resample"]
        self.meta = {}
        self.feature_names = config.FEATURE_SETS[args.features]
        self.device = torch.device(args.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ data
    def load(self):
        a = self.args
        if self.dataset == "emothaw":
            samples, report = load_emothaw(a.data_dir, self.heads, a.tasks)
        else:
            samples, report, self.meta = four_class.load_dataset(a.data_dir, self.heads)
            if a.sample_source:                              # e.g. human_canvas only
                samples = [s for s in samples if s["sample_source"] in set(a.sample_source)]
                if not samples:
                    raise ValueError(f"no samples with sample_source in {a.sample_source}")
            report["sample_sources"] = {o: sum(s["sample_source"] == o and not s["synthetic"] for s in samples)
                                        for o in sorted({s["sample_source"] for s in samples})}
        prepared, skipped = [], []
        for s in samples:
            try:
                s["traj"] = prepare(s["traj"], self.air, self.resample)
                prepared.append(s)
            except HandwritingInputError as exc:
                skipped.append(f"writer {s['writer']} sample {s['task']}: {exc}")
        report["skipped_too_short"] = skipped
        if not prepared:
            raise ValueError("no usable recordings (see the data report above)")
        # stratify writers by their label combination (one per writer in EMOTHAW)
        label_sets = defaultdict(set)
        for s in prepared:
            label_sets[s["writer"]].add(tuple(s["y"][h["name"]] for h in self.heads))
        if self.dataset == "emothaw":
            conflicting = sorted(w for w, v in label_sets.items() if len(v) > 1)
            if conflicting:
                raise ValueError(f"writers with inconsistent labels across recordings: {conflicting}")
        # 4-class: a writer may have written in several emotions -> stratify by the set
        self.single_label_writers = all(len(v) == 1 for v in label_sets.values())
        self.split = writer_split({w: tuple(sorted(v)) for w, v in label_sets.items()}, seed=a.seed)
        where = {w: name for name, ws in self.split.items() for w in ws}
        self.sets = {name: [s for s in prepared if where[s["writer"]] == name] for name in self.split}
        if self.dataset != "emothaw":
            # augmented copies are for training only: evaluate on original samples
            for name in ("val", "test"):
                report[f"{name}_synthetic_excluded"] = sum(s["synthetic"] for s in self.sets[name])
                self.sets[name] = [s for s in self.sets[name] if not s["synthetic"]]
        self.writer_independent = check_no_leakage(self.sets)

        # normalisation statistics: training writers only
        raw = {id(s): feature_matrix(s["traj"], self.feature_names)[0] for s in prepared}
        train_rows = np.concatenate([raw[id(s)] for s in self.sets["train"]])
        self.mean = train_rows.mean(0)
        self.std = train_rows.std(0)
        self.std[self.std < 1e-6] = 1.0
        for s in prepared:
            s["x"], _ = feature_matrix(s["traj"], self.feature_names, self.mean, self.std)
        self.report = report
        self.split_counts = {
            name: {"writers": len(self.split[name]), "samples": len(self.sets[name]),
                   "samples_per_class": {h["name"]: {c: sum(1 for s in self.sets[name] if s["y"][h["name"]] == i)
                                                     for i, c in enumerate(h["classes"])}
                                         for h in self.heads},
                   "writers_per_class": {h["name"]: {c: len({s["writer"] for s in self.sets[name]
                                                             if s["y"][h["name"]] == i})
                                                     for i, c in enumerate(h["classes"])}
                                         for h in self.heads}}
            for name in self.split}
        if "pressure" in self.feature_names:
            have = sum(np.isfinite(s["traj"].pressure[s["traj"].pen > 0]).any() for s in self.sets["train"])
            report["train_samples_with_pressure"] = int(have)
            if have < 0.5 * len(self.sets["train"]):
                print(f"[DATA] WARNING: only {have}/{len(self.sets['train'])} training samples have "
                      f"pressure; consider --features kinematic", flush=True)

    def _notice(self):
        used = set(self.report.get("sample_sources") or {})
        return config.HUMAN_PROTOTYPE_NOTICE if used == {"human_canvas"} else config.PROTOTYPE_NOTICE

    # ------------------------------------------------------------------ training
    def _class_weights(self):
        weights = {}
        for h in self.heads:
            counts = np.bincount([s["y"][h["name"]] for s in self.sets["train"]],
                                 minlength=len(h["classes"])).astype(float)
            w = counts.sum() / (len(counts) * np.maximum(counts, 1.0))
            weights[h["name"]] = torch.tensor(w, dtype=torch.float32, device=self.device)
        return weights

    def _loss(self, logits, ys, weights=None):
        return sum(F.cross_entropy(logits[h["name"]], ys[h["name"]],
                                   weight=None if weights is None else weights[h["name"]])
                   for h in self.heads) / len(self.heads)

    def _batch(self, mats, samples):
        x, mask = pad_batch(mats)
        ys = {h["name"]: torch.tensor([s["y"][h["name"]] for s in samples], device=self.device)
              for h in self.heads}
        return x.to(self.device), mask.to(self.device), ys

    @torch.no_grad()
    def evaluate(self, model, samples):
        """Loss over all windows; probabilities averaged per recording and per writer."""
        model.eval()
        items = [(i, s["x"][a:b]) for i, s in enumerate(samples) for a, b in windows(len(s["x"]), self.args.max_len)]
        probs = {h["name"]: [[] for _ in samples] for h in self.heads}
        total, count = 0.0, 0
        for k in range(0, len(items), self.args.batch_size):
            chunk = items[k:k + self.args.batch_size]
            x, mask, ys = self._batch([m for _, m in chunk], [samples[i] for i, _ in chunk])
            logits = model(x, mask)
            total += float(self._loss(logits, ys)) * len(chunk)
            count += len(chunk)
            for h in self.heads:
                p = torch.softmax(logits[h["name"]], -1).cpu().numpy()
                for (i, _), row in zip(chunk, p):
                    probs[h["name"]][i].append(row)
        rec = {name: np.array([np.mean(v, 0) for v in lst]) for name, lst in probs.items()}
        metrics = {"loss": total / max(count, 1), "per_recording": {}, "per_writer": {}}
        writers = sorted({s["writer"] for s in samples})
        for h in self.heads:
            n, classes = h["name"], h["classes"]
            base = int(np.argmax(np.bincount([s["y"][n] for s in self.sets["train"]],
                                             minlength=len(classes))))
            y = [s["y"][n] for s in samples]
            pos = h.get("positive")
            metrics["per_recording"][n] = classification_metrics(y, rec[n].argmax(1), classes, base, pos)
            if not self.single_label_writers:
                metrics["per_writer"][n] = None       # writers have samples of several classes
                continue
            wp = [rec[n][[i for i, s in enumerate(samples) if s["writer"] == w]].mean(0) for w in writers]
            wy = [next(s["y"][n] for s in samples if s["writer"] == w) for w in writers]
            metrics["per_writer"][n] = classification_metrics(wy, np.argmax(wp, 1), classes, base, pos)
        ba = [metrics["per_recording"][h["name"]]["balanced_accuracy"] for h in self.heads]
        metrics["mean_balanced_accuracy"] = round(float(np.mean([b for b in ba if b is not None])), 4) if any(b is not None for b in ba) else None
        return metrics

    def train(self):
        a = self.args
        hparams = dict(config.MODEL_DEFAULTS, d_model=a.d_model, n_layers=a.layers,
                       n_heads=a.heads, dim_feedforward=a.ff, dropout=a.dropout)
        model = build_model(len(self.feature_names), self.heads, hparams).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        weights = self._class_weights()
        train = self.sets["train"]
        best, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
        print(f"[TRAIN] {n_params:,} parameters, features={a.features} ({len(self.feature_names)}), "
              f"device={self.device}", flush=True)
        for epoch in range(1, a.epochs + 1):
            model.train()
            order = np.random.permutation(len(train))
            losses, correct, seen = [], 0, 0
            for k in range(0, len(order), a.batch_size):
                batch = [train[i] for i in order[k:k + a.batch_size]]
                mats = []
                for s in batch:            # one random window per recording per epoch
                    L = len(s["x"])
                    start = np.random.randint(0, L - a.max_len + 1) if L > a.max_len else 0
                    mats.append(s["x"][start:start + a.max_len])
                x, mask, ys = self._batch(mats, batch)
                logits = model(x, mask)
                loss = self._loss(logits, ys, weights)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
                for h in self.heads:
                    correct += int((logits[h["name"]].argmax(-1) == ys[h["name"]]).sum())
                    seen += len(batch)
            val = self.evaluate(model, self.sets["val"])
            row = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                   "train_accuracy_diagnostic_only": round(correct / max(seen, 1), 4),
                   "val_loss": round(val["loss"], 4),
                   "val_mean_balanced_accuracy": val["mean_balanced_accuracy"]}
            history.append(row)
            print(f"[TRAIN] epoch {epoch:3d} train_loss={row['train_loss']:.4f} "
                  f"val_loss={row['val_loss']:.4f} val_bal_acc={row['val_mean_balanced_accuracy']}",
                  flush=True)
            if val["loss"] < best - 1e-4:
                best, best_epoch, bad = val["loss"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"[TRAIN] early stop: no val improvement for {a.patience} epochs", flush=True)
                    break
        model.load_state_dict(best_state)
        return model, hparams, history, best_epoch, n_params

    def run(self):
        a = self.args
        t0 = time.time()
        self.load()
        print("[DATA] " + json.dumps({k: v for k, v in self.report.items()
                                      if not isinstance(v, list)}), flush=True)
        all_samples = [s for n in self.sets for s in self.sets[n]]
        if self.dataset == "emothaw":
            print(f"[DATA] {len({s['writer'] for s in all_samples})} participants, {len(all_samples)} samples "
                  f"(tasks {sorted(a.tasks)})", flush=True)
        else:
            print(f"[DATA] dataset {self.dataset}: dataset_type={self.meta.get('dataset_type')}, "
                  f"source_type={self.meta.get('source_type')}, "
                  f"scientifically_validated={self.meta.get('scientifically_validated')}", flush=True)
            print(f"[DATA] {len({s['writer'] for s in all_samples})} split groups "
                  f"(writers where known, else source samples), {len({s['source'] for s in all_samples})} "
                  f"source samples, {len(all_samples)} samples used "
                  f"({sum(s['synthetic'] for s in all_samples)} augmented, training only)", flush=True)
            print(f"[DATA] originals per sample_source: {self.report.get('sample_sources')}", flush=True)
            if self.meta.get("dataset_type") == "seed":
                print(f"[DATA] PROTOTYPE / SEED MODEL: {self._notice()}", flush=True)
        for h in self.heads:
            dist = ", ".join(f"{c} {sum(s['y'][h['name']] == i for s in all_samples)} samples / "
                             f"{len({s['writer'] for s in all_samples if s['y'][h['name']] == i})} writers"
                             for i, c in enumerate(h["classes"]))
            print(f"[DATA]   {h['display']:10s}: {dist}", flush=True)
        unit = "writers" if self.dataset == "emothaw" else "groups"     # 4-class: writer or source sample
        for name in ("train", "val", "test"):
            c = self.split_counts[name]
            print(f"[SPLIT] {name:5s}: {c['writers']} {unit}, {c['samples']} samples", flush=True)
            for h in self.heads:
                print(f"[SPLIT] {name:5s}   {h['display']:10s} samples {c['samples_per_class'][h['name']]}, "
                      f"{unit} {c['writers_per_class'][h['name']]}", flush=True)
            print(f"[SPLIT] {name:5s} {unit}: {self.split[name]}", flush=True)
        print("[SPLIT] leakage check passed: no writer, file or identical recording is in two splits",
              flush=True)
        for name in ("val", "test"):
            for h in self.heads:
                empty = [c for c, k in self.split_counts[name]["writers_per_class"][h["name"]].items() if not k]
                if empty:
                    print(f"[SPLIT] WARNING: {name} has no {'/'.join(empty)} {unit} for {h['display']}; "
                          f"its {name} metrics are not meaningful (try another --seed)", flush=True)
        if not self.sets["train"] or not self.sets["val"] or not self.sets["test"]:
            raise SystemExit("Not enough labelled writers for a train/val/test split.")
        model, hparams, history, best_epoch, n_params = self.train()
        val = self.evaluate(model, self.sets["val"])
        test = self.evaluate(model, self.sets["test"])       # the one and only test evaluation
        if self.dataset == "emothaw":
            info = {"dataset": config.DATASET_DISPLAY, "dataset_key": "emothaw",
                    "labels_sha256": _sha256(os.path.join(a.data_dir, "labels.csv")), "tasks": sorted(a.tasks)}
        else:
            info = {"dataset": self.profile["display"], "dataset_key": self.dataset,
                    "dataset_type": self.meta.get("dataset_type"), "source_type": self.meta.get("source_type"),
                    "scientifically_validated": self.meta.get("scientifically_validated"),
                    "model_kind": self.profile["model_kind"],
                    "notice": self._notice() if self.meta.get("dataset_type") == "seed" else None,
                    "sample_sources": self.report.get("sample_sources"),
                    "evaluated_on": "original samples only (augmented copies used for training only)",
                    "tasks": None}
        training = dict(info, **{
            "data_dir": os.path.relpath(a.data_dir, config.PROJECT_DIR),
            "manifest_sha256": _sha256(os.path.join(a.data_dir, "manifest.csv")),
            "writer_independent": self.writer_independent,
            "split": self.split, "split_counts": self.split_counts, "data_report": self.report,
            "best_epoch": best_epoch, "epochs_run": len(history), "parameters": n_params,
            "val_metrics": val, "test_metrics": test, "history": history, "seed": a.seed,
            "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "train_seconds": round(time.time() - t0, 1),
        })
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "state_dict": model.state_dict(),
            "heads": self.heads,
            "feature_set": a.features,
            "feature_names": self.feature_names,
            "norm": {"mean": self.mean.tolist(), "std": self.std.tolist()},
            "hparams": hparams,
            "preprocessing": {"air": self.air, "resample": self.resample, "resample_hz": config.RESAMPLE_HZ,
                              "arclength_step": config.ARCLENGTH_STEP, "max_len": a.max_len,
                              "min_dt": config.MIN_DT, "max_dt": config.MAX_DT,
                              "feature_clip": config.FEATURE_CLIP},
            "training": training,
        }
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save(checkpoint, a.out)
        run_dir = a.results_dir or os.path.join(
            config.PROJECT_DIR, "runs", "handwriting",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ("" if self.dataset == "emothaw" else "-" + self.dataset))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(training, f, indent=2)
        print(f"\n[RESULT] best epoch {best_epoch}; checkpoint -> {a.out}; results -> {run_dir}/results.json")
        print(f"[RESULT] held-out TEST set: {self.split_counts['test']['writers']} {unit}, "
              f"{self.split_counts['test']['samples']} recordings (never used for training or model selection)")
        if self.meta.get("dataset_type") == "seed":
            print("[RESULT] PROTOTYPE / SEED MODEL - the numbers below only show that the pipeline runs end to "
                  "end. They are NOT emotion-recognition accuracy: the labels are graphology-based seed "
                  "labels, not validated emotional states.")
        for level in ("per_recording", "per_writer"):
            for h in self.heads:
                m = test[level][h["name"]]
                if m is None:
                    continue
                if "positive_class" in m:
                    detail = (f"precision={m['precision']} recall={m['recall']} F1={m['f1']} "
                              f"({m['positive_class']} = positive) macro_f1={m['macro_f1']}")
                else:
                    detail = (f"macro_precision={m['macro_precision']} macro_recall={m['macro_recall']} "
                              f"macro_f1={m['macro_f1']}")
                print(f"[TEST {level}] {h['display']:10s} n={m['n']} acc={m['accuracy']} "
                      f"bal_acc={m['balanced_accuracy']} {detail} "
                      f"(majority baseline acc {m['majority_baseline_accuracy']})")
                for c, pc in m["per_class"].items():
                    print(f"[TEST {level}]   {c:9s} P={pc['precision']} R={pc['recall']} "
                          f"F1={pc['f1']} n={pc['support']}")
                print(f"[TEST {level}]   confusion (rows true, cols pred {h['classes']}): "
                      f"{m['confusion_matrix']['matrix']}")
        return checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="emothaw", choices=["emothaw"] + sorted(config.FOUR_CLASS_DATASETS),
                   help="emothaw (Anxiety/Depression/Stress, default) or a 4-class Happy/Neutral/Sad/Angry dataset")
    p.add_argument("--data-dir", default=None,
                   help="default: data/emothaw/processed, or data/handwriting/<dataset>/processed")
    p.add_argument("--tasks", default=",".join(map(str, config.DEFAULT_TASKS)),
                   help='EMOTHAW only: comma-separated task numbers (1-7), or "all"')
    p.add_argument("--features", default=None, choices=sorted(config.FEATURE_SETS),
                   help="default: pen (emothaw), or the 4-class dataset profile's")
    p.add_argument("--sample-source", action="append", choices=list(config.SAMPLE_SOURCES),
                   help="4-class only: train on these origins only (repeatable), e.g. --sample-source human_canvas")
    p.add_argument("--resample", default=None, choices=["time", "arclength"],
                   help="4-class only: override the dataset profile's resampling")
    p.add_argument("--out", default=None,
                   help="default: models/handwriting_transformer.pt (emothaw) or models/handwriting_4class.pt")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--max-len", type=int, default=config.MAX_LEN)
    p.add_argument("--d-model", type=int, default=config.MODEL_DEFAULTS["d_model"])
    p.add_argument("--layers", type=int, default=config.MODEL_DEFAULTS["n_layers"])
    p.add_argument("--heads", type=int, default=config.MODEL_DEFAULTS["n_heads"])
    p.add_argument("--ff", type=int, default=config.MODEL_DEFAULTS["dim_feedforward"])
    p.add_argument("--dropout", type=float, default=config.MODEL_DEFAULTS["dropout"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    if a.dataset == "emothaw":
        a.data_dir = a.data_dir or config.EMOTHAW_PROCESSED_DIR
        a.features = a.features or config.DEFAULT_FEATURES
        a.out = a.out or config.DEFAULT_CHECKPOINT
    else:
        profile = config.FOUR_CLASS_DATASETS[a.dataset]
        a.data_dir = a.data_dir or os.path.join(profile["dir"], "processed")
        a.features = a.features or profile["features"]
        a.out = a.out or config.FOUR_CLASS_CHECKPOINT
    a.tasks = list(config.ALL_TASKS) if a.tasks == "all" else [int(t) for t in a.tasks.split(",")]
    bad = [t for t in a.tasks if t not in config.ALL_TASKS]
    if bad:
        p.error(f"unknown EMOTHAW task(s) {bad}; tasks are 1-7")
    return a


def main(argv=None):
    try:
        Trainer(parse_args(argv)).run()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

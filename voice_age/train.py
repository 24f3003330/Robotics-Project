"""
voice_age/train.py - reproducible training of the voice AGE-GROUP model.
Run via `python scripts/train_voice_age.py` (see --help).

  1. Common Voice metadata (cached scan; no audio)        dataset.scan_metadata
  2. age-group mapping, one label per speaker             selection.label_speakers
  3. speaker-wise split 70/15/15 + leakage check          selection.split_speakers / verify_disjoint
  4. manageable subset (per-class / per-speaker caps)     selection.select_subset
  5. download only the selected clips' audio              dataset.fetch_audio
  6. preprocess (mono 16 kHz float32, trim, limits)       audio.prepare_bytes   (bad clips logged + skipped)
  7. frozen encoder embeddings (cached)                   embeddings.SpeechEncoder
  8. choose encoder layer + head on VALIDATION speakers   model.LogRegHead / MLPHead
  9. evaluate ONCE on unseen TEST speakers; majority baseline
 10. save models/voice_age_model/{head.npz, metadata.json} + voice_age/results/
"""

import argparse
import json
import os
import platform
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from . import config
from .audio import AudioError, prepare_bytes
from .dataset import fetch_audio, load_clip_bytes, scan_metadata
from .model import LogRegHead, MLPHead, adjacent_accuracy, count_parameters, majority_baseline, metrics
from .selection import (check_groups, group_stats, label_speakers, select_subset, split_report, split_speakers,
                        verify_disjoint)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=config.AGE_DATASET, help="Hugging Face dataset repo (Common Voice)")
    p.add_argument("--language", default=config.AGE_LANGUAGE)
    p.add_argument("--max-samples-per-class", type=int, default=config.AGE_MAX_SAMPLES_PER_CLASS,
                   help="clips per age group over all splits")
    p.add_argument("--max-clips-per-speaker", type=int, default=config.AGE_MAX_CLIPS_PER_SPEAKER)
    p.add_argument("--encoder", default=config.AGE_ENCODER, help="frozen pretrained speech encoder")
    p.add_argument("--head", choices=["auto", "logreg", "mlp"], default="auto")
    p.add_argument("--epochs", type=int, default=60, help="max MLP epochs (early stopping)")
    p.add_argument("--batch-size", type=int, default=64, help="MLP batch size")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--output-dir", default=config.MODEL_DIR)
    p.add_argument("--results-dir", default=config.RESULTS_DIR)
    p.add_argument("--workers", type=int, default=6, help="parallel downloads")
    return p.parse_args(argv)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
def build_selection(args):
    meta = scan_metadata(args.language, config.AGE_SOURCE_SPLIT, log=log)
    log(f"[voice-age] {len(meta)} clips, {meta['client_id'].nunique()} speakers in "
        f"{config.AGE_DATASET}/{args.language}/{config.AGE_SOURCE_SPLIT}")
    d, rep = label_speakers(meta)
    log(f"[voice-age] age-labelled clips kept: {len(d)} ({d['client_id'].nunique()} speakers); "
        f"speakers with conflicting labels {rep['speakers_with_conflicting_labels']}, dropped as ties "
        f"{rep['speakers_dropped_tie']}, minority-label clips dropped {rep['clips_dropped_minority_label']}")
    stats = group_stats(d)
    log("[voice-age] available per age group (before subsetting):")
    for g, s in stats.items():
        log(f"    {g:<11} {s['clips']:>8} clips {s['speakers']:>7} speakers")
    sparse = check_groups(d)
    if sparse:
        raise SystemExit(f"age groups {sparse} have fewer than {config.MIN_SPEAKERS_PER_GROUP} speakers - "
                         f"merge them in config.AGE_GROUP_MAPPING before training")
    speaker_split = split_speakers(d, seed=args.seed)
    audit = verify_disjoint({s: [k for k, v in speaker_split.items() if v == s]
                             for s in ("train", "validation", "test")})
    sel, info = select_subset(d, speaker_split, args.max_samples_per_class, args.max_clips_per_speaker,
                              seed=args.seed)
    audit = verify_disjoint({s: sel.loc[sel["split"] == s, "client_id"].unique()
                             for s in ("train", "validation", "test")})
    return sel, {"label_report": rep, "available": stats, "leakage_audit": audit, **info}


def preprocess_and_embed(sel, encoder_name, chunk=256, flush_every=4):
    """Decode + standardize + embed the selected clips in chunks, keeping only the float16
    embeddings in memory (decoded audio of ~10k clips would need several GB on a 16 GB Mac).

    Resumable: embeddings and skip reasons are flushed to data/voice_age/embeddings/ every few
    chunks; a re-run only processes clips that are not cached yet. A malformed clip is skipped and
    its reason counted - it never stops training.
    Returns (E [N, layers+1, hidden] float32, kept DataFrame, skip counts, skip examples)."""
    os.makedirs(config.EMBEDDING_CACHE_DIR, exist_ok=True)
    stem = os.path.join(config.EMBEDDING_CACHE_DIR, encoder_name.replace("/", "__"))
    have, skipped_paths = {}, {}
    if os.path.isfile(stem + ".npz"):
        z = np.load(stem + ".npz", allow_pickle=False)
        have = dict(zip(z["paths"].tolist(), z["emb"]))
    if os.path.isfile(stem + ".skipped.json"):
        with open(stem + ".skipped.json") as fh:
            skipped_paths = json.load(fh)

    def flush():
        paths = np.array(list(have))
        if len(paths):
            np.savez(stem + ".tmp.npz", paths=paths, emb=np.stack([have[p] for p in paths]))
            os.replace(stem + ".tmp.npz", stem + ".npz")
        with open(stem + ".skipped.json", "w") as fh:
            json.dump(skipped_paths, fh)

    todo = [i for i, p in enumerate(sel["path"]) if p not in have and p not in skipped_paths]
    if todo:
        from .embeddings import SpeechEncoder
        log(f"[voice-age] preprocessing + embedding {len(todo)} clips with frozen {encoder_name} "
            f"({len(have)} cached) ...")
        enc = SpeechEncoder(encoder_name)
        log(f"    device {enc.device}, {enc.n_layers} layers, hidden {enc.hidden}")
        t0 = time.time()
        for n, k in enumerate(range(0, len(todo), chunk)):
            clips, paths = [], []
            for i in todo[k:k + chunk]:
                row = sel.iloc[i]
                try:
                    _, data = load_clip_bytes(row["shard"], row["row_group"], row["row"])
                    clips.append(prepare_bytes(data))
                    paths.append(row["path"])
                except AudioError as exc:
                    skipped_paths[row["path"]] = [exc.reason, str(exc)]
                except Exception as exc:                  # never let one malformed clip stop training
                    skipped_paths[row["path"]] = [f"error_{type(exc).__name__}", str(exc)]
            if clips:
                for p, e in zip(paths, enc.embed_batched(clips, batch_seconds=30.0)):
                    have[p] = e.astype(np.float16)
            del clips
            if enc.device == "mps":
                import torch
                torch.mps.empty_cache()
            if (n + 1) % flush_every == 0:
                flush()
            log(f"    {min(k + chunk, len(todo))}/{len(todo)} clips, {time.time() - t0:.0f} s")
        flush()
    keep = [i for i, p in enumerate(sel["path"]) if p in have]
    reasons = [skipped_paths[p] for p in sel["path"] if p in skipped_paths]
    skipped = dict(Counter(r[0] for r in reasons))
    examples = [{"path": p, "reason": skipped_paths[p][1]} for p in sel["path"] if p in skipped_paths][:20]
    kept = sel.iloc[keep].reset_index(drop=True)
    E = np.stack([have[p] for p in kept["path"]]).astype(np.float32)
    return E, kept, skipped, examples


def speaker_level(sel_part, proba):
    """Mean clip probabilities per speaker -> speaker-level labels / predictions."""
    import pandas as pd
    df = pd.DataFrame(proba)
    df["spk"] = sel_part["client_id"].to_numpy()
    df["y"] = sel_part["y"].to_numpy()
    g = df.groupby("spk")
    return g["y"].first().to_numpy(), g[list(range(proba.shape[1]))].mean().to_numpy().argmax(1)


def plot_confusion(cm, classes, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cm = np.asarray(cm)
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{row[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if row[i, j] > 0.55 else "#1f2937")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted age group")
    ax.set_ylabel("Self-reported age group")
    ax.set_title(title, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def run(args):
    t_start = time.time()
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass
    config.AGE_DATASET = args.dataset
    sel, sel_info = build_selection(args)
    import gc
    gc.collect()                         # the 1.8 M-row metadata table is no longer needed
    classes = [g for g in config.AGE_GROUPS if g in set(sel["group"])]
    log(f"[voice-age] classes: {classes}")
    log("[voice-age] leakage audit (speaker overlap between splits): "
        + ", ".join(f"{k} = {v}" for k, v in sel_info["leakage_audit"].items()))

    fetched = fetch_audio(sel, workers=args.workers, log=log)
    E, sel, skipped, skip_examples = preprocess_and_embed(sel, args.encoder)
    log(f"[voice-age] usable clips {len(sel)}; skipped {sum(skipped.values())}: {skipped}")
    lost = sum(v for k, v in skipped.items() if k.startswith("error_"))
    if lost > 0.05 * (len(sel) + sum(skipped.values())):
        raise SystemExit(f"{lost} selected clips could not be loaded (network?) - re-run to resume the download")
    sel["y"] = sel["group"].map({c: i for i, c in enumerate(classes)})
    verify_disjoint({s: sel.loc[sel["split"] == s, "client_id"].unique() for s in ("train", "validation", "test")})
    report = split_report(sel)
    for s, r in report.items():
        log(f"    {s:<10} {r['speakers']:>5} speakers {r['clips']:>6} clips  "
            + "  ".join(f"{g}:{v['clips']}/{v['speakers']}spk" for g, v in r["by_group"].items()))

    part = {s: (sel["split"] == s).to_numpy() for s in ("train", "validation", "test")}
    y = sel["y"].to_numpy()
    ytr, yva, yte = y[part["train"]], y[part["validation"]], y[part["test"]]

    # ---- model selection on VALIDATION speakers only
    log("[voice-age] layer selection (logistic regression, validation macro F1):")
    layer_scores = {}
    for layer in range(E.shape[1]):
        h = LogRegHead(C=0.1, seed=args.seed).fit(E[part["train"], layer], ytr, n_classes=len(classes))
        layer_scores[layer] = metrics(yva, h.predict_proba(E[part["validation"], layer]).argmax(1), classes)["macro_f1"]
        log(f"    layer {layer:>2}: {layer_scores[layer]:.3f}")
    best_layer = max(layer_scores, key=layer_scores.get)
    Xtr, Xva, Xte = (E[part[s], best_layer] for s in ("train", "validation", "test"))

    candidates = []
    if args.head in ("auto", "logreg"):
        for C in (0.03, 0.1, 0.3, 1.0):
            h = LogRegHead(C=C, seed=args.seed).fit(Xtr, ytr, n_classes=len(classes))
            candidates.append((metrics(yva, h.predict_proba(Xva).argmax(1), classes), h, {"C": C}))
    if args.head in ("auto", "mlp"):
        for hidden, dropout in ((256, 0.3), (512, 0.5)):
            h = MLPHead(hidden=hidden, dropout=dropout, epochs=args.epochs, batch_size=args.batch_size,
                        seed=args.seed, log=log).fit(Xtr, ytr, Xva, yva, len(classes))
            candidates.append((metrics(yva, h.predict_proba(Xva).argmax(1), classes), h,
                               {"hidden": hidden, "dropout": dropout, "best_epoch": h.best_epoch}))
    log("[voice-age] head selection (validation):")
    for m, h, hp in candidates:
        log(f"    {h.kind:<7} {json.dumps(hp):<45} macro F1 {m['macro_f1']:.3f}  bal-acc {m['balanced_accuracy']:.3f}")
    val_m, head, head_hp = max(candidates, key=lambda c: (c[0]["macro_f1"], c[0]["balanced_accuracy"]))
    log(f"[voice-age] selected: layer {best_layer}, {head.kind} {head_hp}")

    # ---- test: evaluated ONCE
    proba_te = head.predict_proba(Xte)
    pred_te = proba_te.argmax(1)
    test_m = metrics(yte, pred_te, classes)
    test_m["adjacent_accuracy"] = adjacent_accuracy(yte, pred_te)
    base = majority_baseline(ytr, yte, classes)
    spk_y, spk_pred = speaker_level(sel[part["test"]], proba_te)
    spk_m = metrics(spk_y, spk_pred, classes)
    val_m["adjacent_accuracy"] = adjacent_accuracy(yva, head.predict_proba(Xva).argmax(1))
    log(f"\n[voice-age] TEST ({part['test'].sum()} clips from {sel.loc[part['test'], 'client_id'].nunique()} "
        f"unseen speakers): acc {test_m['accuracy']:.3f}  bal-acc {test_m['balanced_accuracy']:.3f}  "
        f"macro F1 {test_m['macro_f1']:.3f}  weighted F1 {test_m['weighted_f1']:.3f}  "
        f"within one group {test_m['adjacent_accuracy']:.3f}")
    log(f"[voice-age] majority baseline (always {base['predicts']}): acc {base['accuracy']:.3f}  "
        f"bal-acc {base['balanced_accuracy']:.3f}  macro F1 {base['macro_f1']:.3f}")
    log(f"[voice-age] speaker-level (clip probabilities averaged per speaker): acc {spk_m['accuracy']:.3f}  "
        f"bal-acc {spk_m['balanced_accuracy']:.3f}  macro F1 {spk_m['macro_f1']:.3f}")
    for c, v in test_m["per_class"].items():
        log(f"    {c:<11} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  n {v['support']}")

    # ---- save
    st = head.state()
    metadata = {
        "task": "voice age-GROUP estimation (self-reported Common Voice age categories; not exact age)",
        "model_version": f"voice-age-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "dataset": {"repo": args.dataset, "version": config.AGE_DATASET_VERSION, "language": args.language,
                    "source_split": config.AGE_SOURCE_SPLIT,
                    "note": "Parquet re-upload of Common Voice 17.0 (CC0); Mozilla withdrew Common Voice from the "
                            "Hugging Face Hub in October 2025"},
        "classes": classes,
        "age_group_mapping": config.AGE_GROUP_MAPPING,
        "architecture": {"encoder": args.encoder, "encoder_frozen": True, "layer": int(best_layer),
                         "pooling": "mean over time (padding masked)", "head": head.kind,
                         "head_hyperparameters": head_hp, "trainable_parameters": count_parameters(st)},
        "preprocessing": {"sample_rate": config.SAMPLE_RATE, "mono": True, "dtype": "float32",
                          "min_clip_seconds": config.MIN_CLIP_SECONDS, "max_clip_seconds": config.MAX_CLIP_SECONDS,
                          "silence_dbfs": config.SILENCE_DBFS, "trim_relative_db": config.TRIM_DBFS},
        "subset": {"max_samples_per_class": args.max_samples_per_class,
                   "max_clips_per_speaker": args.max_clips_per_speaker,
                   "split_fractions_by_speaker": config.SPLIT_FRACTIONS, **sel_info},
        "splits": report,
        "skipped_clips": skipped,
        "random_seed": args.seed,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_metrics": val_m,
        "layer_selection_validation_macro_f1": layer_scores,
        "test_metrics": test_m,
        "test_speaker_level_metrics": spk_m,
        "test_majority_baseline": base,
        "environment": {"python": platform.python_version()},
    }
    from .inference import save_model
    save_model(args.output_dir, st, metadata)
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as fh:
        json.dump({k: metadata[k] for k in ("classes", "splits", "validation_metrics", "test_metrics",
                                            "test_speaker_level_metrics", "test_majority_baseline",
                                            "architecture", "skipped_clips")} | {"skip_examples": skip_examples,
                                                                                 "download_bytes": fetched},
                  fh, indent=2, default=_json_default)
    plot_confusion(test_m["confusion_matrix"], classes, os.path.join(args.results_dir, "confusion_matrix.png"),
                   f"Voice age group - {part['test'].sum()} test clips, "
                   f"{sel.loc[part['test'], 'client_id'].nunique()} unseen speakers")
    sel.drop(columns=["rg_key"], errors="ignore").to_csv(config.SELECTION_FILE, index=False)
    log(f"\nSaved {args.output_dir}/ (head.npz, metadata.json) and {args.results_dir}/ "
        f"({time.time() - t_start:.0f} s)")
    return metadata


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

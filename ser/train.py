"""Fine-tuning one HuBERT model per dataset (or per dataset group).

Deliberately NOT one model on the concatenation of all five corpora. The five
differ in recording chain, elicitation (acted vs improvised vs found podcast
audio) and label provenance; pooled training lets the largest corpus set the
decision boundary for all of them, and the resulting model is worse on every
individual corpus than a model trained on that corpus. Training them separately
keeps each corpus's own prior intact and gives the ensemble members that
actually disagree - which is the only reason an ensemble helps at all.

The starting point stays `superb/hubert-base-superb-er`, the model already in
the app. Its 4-class head is replaced (the checkpoint's own class order is not
the canonical one, and its label set is IEMOCAP's) and the CNN feature encoder
is frozen: it is a generic waveform front-end, fine-tuning it on a few thousand
utterances mostly memorises channel characteristics.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import DEFAULT_WINDOW_SPEC, SAMPLE_RATE
from .labels import CANONICAL_EMOTIONS
from .metrics import evaluate, format_report
from .windows import (_AudioLRU, _SHARED_VAD_BACKEND, aggregate_utterance, fetch_window,
                      get_shared_vad, set_shared_vad)

BASE_MODEL = os.environ.get("SER_BASE_MODEL", "superb/hubert-base-superb-er")


class WindowDataset(Dataset):
    def __init__(self, entries, spec=DEFAULT_WINDOW_SPEC, augment=False, seed=0,
                 vad_backend=None):
        self.entries = entries
        self.spec = spec
        self.augment = augment
        self.cache = _AudioLRU()
        self.rng = np.random.default_rng(seed)
        # Carried on the dataset, not read from a module global, so a DataLoader
        # worker started with spawn (rather than fork) can rebuild the same VAD
        # and reproduce the window offsets the index was built with.
        self.vad_backend = vad_backend or _SHARED_VAD_BACKEND[0]

    def __len__(self):
        return len(self.entries)

    def _ensure_vad(self):
        if self.vad_backend and get_shared_vad() is None:
            set_shared_vad(self.vad_backend)

    def __getitem__(self, i):
        entry = self.entries[i]
        if entry.get("cropped"):
            self._ensure_vad()
        window = fetch_window(self.cache, entry, self.spec)
        if self.augment:
            # Gain jitter only. Pitch/speed perturbation changes exactly the
            # prosodic cues the label describes, so it is not used here.
            window = window * float(self.rng.uniform(0.8, 1.2))
            peak = float(np.abs(window).max()) if window.size else 0.0
            if peak > 1.0:
                window = window / peak
        return torch.from_numpy(np.ascontiguousarray(window)), int(entry["label_idx"])


def collate(batch, target_samples):
    """Pad/trim to a fixed length so every batch is one tensor (HuBERT handles both)."""
    xs, ys = zip(*batch)
    out = torch.zeros(len(xs), target_samples, dtype=torch.float32)
    for i, x in enumerate(xs):
        n = min(len(x), target_samples)
        out[i, :n] = x[:n]
    return out, torch.tensor(ys, dtype=torch.long)


def class_weights(entries, device):
    """Inverse-frequency weights, normalised to mean 1.

    MSP-Podcast is ~60 % neutral; without this the model reaches a decent
    accuracy by rarely predicting sad at all, which is exactly the failure
    macro-F1 is meant to expose.
    """
    counts = np.zeros(len(CANONICAL_EMOTIONS), dtype=np.float64)
    for e in entries:
        counts[e["label_idx"]] += 1
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(device, base_model=BASE_MODEL, freeze_feature_encoder=True):
    from transformers import AutoConfig, AutoModelForAudioClassification

    config = AutoConfig.from_pretrained(base_model)
    config.num_labels = len(CANONICAL_EMOTIONS)
    config.id2label = {i: e for i, e in enumerate(CANONICAL_EMOTIONS)}
    config.label2id = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}
    model = AutoModelForAudioClassification.from_pretrained(
        base_model, config=config, ignore_mismatched_sizes=True)
    if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    return model.to(device)


@torch.no_grad()
def infer_logits(model, entries, spec, device, batch_size=8, num_workers=0, desc=""):
    """Raw logits for every window, in `entries` order."""
    model.eval()
    loader = DataLoader(WindowDataset(entries, spec), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        collate_fn=lambda b: collate(b, spec.window_samples))
    chunks = []
    for i, (x, _) in enumerate(loader):
        logits = model(input_values=x.to(device)).logits
        chunks.append(logits.float().cpu().numpy())
        if desc and i and i % 50 == 0:
            print(f"    [{desc}] {i * batch_size}/{len(entries)} windows", flush=True)
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(CANONICAL_EMOTIONS)))


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def train_model(name, train_entries, val_entries, out_dir, spec=DEFAULT_WINDOW_SPEC,
                epochs=4, batch_size=8, lr=2e-5, head_lr=1e-3, device=None,
                base_model=BASE_MODEL, vad_backend="silero", num_workers=0, seed=1337,
                grad_accum=2, max_train_windows=None):
    """Fine-tune, keeping the checkpoint with the best VALIDATION macro-F1.

    Selection is on utterance-level macro-F1 (not window-level loss) because
    that is the quantity the ensemble and the app care about.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    set_shared_vad(vad_backend)

    if max_train_windows and len(train_entries) > max_train_windows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(train_entries), size=max_train_windows, replace=False)
        train_entries = [train_entries[i] for i in sorted(keep.tolist())]
        print(f"[train:{name}] subsampled to {len(train_entries)} training windows")

    print(f"[train:{name}] {len(train_entries)} train windows / {len(val_entries)} val windows "
          f"on {device}, base={base_model}")
    model = build_model(device, base_model)
    weights = class_weights(train_entries, device)
    print(f"[train:{name}] class weights " +
          " ".join(f"{e}={w:.2f}" for e, w in zip(CANONICAL_EMOTIONS, weights.tolist())))

    head_names = ("classifier", "projector")
    head_params = [p for n, p in model.named_parameters()
                   if any(n.startswith(h) for h in head_names) and p.requires_grad]
    body_params = [p for n, p in model.named_parameters()
                   if not any(n.startswith(h) for h in head_names) and p.requires_grad]
    optimiser = torch.optim.AdamW(
        [{"params": body_params, "lr": lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    loader = DataLoader(WindowDataset(train_entries, spec, augment=True, seed=seed),
                        batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        collate_fn=lambda b: collate(b, spec.window_samples), drop_last=False)
    total_steps = max(1, (len(loader) // grad_accum) * epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[lr, head_lr], total_steps=total_steps, pct_start=0.1)

    history, best_f1, best_epoch = [], -1.0, -1
    step = 0
    for epoch in range(epochs):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        optimiser.zero_grad()
        for i, (x, y) in enumerate(loader):
            out = model(input_values=x.to(device))
            loss = loss_fn(out.logits, y.to(device)) / grad_accum
            loss.backward()
            running += float(loss) * grad_accum * len(y)
            seen += len(y)
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad()
                if step < total_steps:
                    scheduler.step()
                step += 1
            if i and i % 100 == 0:
                print(f"  [{name}] epoch {epoch + 1} step {i}/{len(loader)} "
                      f"loss {running / max(seen, 1):.4f}", flush=True)

        val_logits = infer_logits(model, val_entries, spec, device, batch_size, num_workers,
                                  desc=f"{name} val")
        probs, labels, _, _, _ = aggregate_utterance(softmax(val_logits), val_entries)
        result = evaluate(probs, labels, label=f"{name} val epoch {epoch + 1}")
        history.append({"epoch": epoch + 1, "train_loss": round(running / max(seen, 1), 4),
                        "val_macro_f1": result["macro_f1"], "val_accuracy": result["accuracy"]})
        print(f"[train:{name}] epoch {epoch + 1}/{epochs} loss {running / max(seen, 1):.4f} "
              f"val macro-F1 {result['macro_f1']:.4f} acc {result['accuracy']:.4f} "
              f"({time.time() - started:.0f}s)", flush=True)

        if result["macro_f1"] > best_f1:
            best_f1, best_epoch = result["macro_f1"], epoch + 1
            model.save_pretrained(out_dir)
            _save_extractor(base_model, out_dir)
            print(f"[train:{name}] new best -> saved to {out_dir}")

    meta = {
        "name": name, "base_model": base_model, "classes": list(CANONICAL_EMOTIONS),
        "window_spec": spec.as_dict(), "vad_backend": vad_backend,
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "head_lr": head_lr,
        "grad_accum": grad_accum, "seed": seed,
        "train_windows": len(train_entries), "val_windows": len(val_entries),
        "best_epoch": best_epoch, "best_val_macro_f1": round(best_f1, 4),
        "history": history, "device": str(device),
    }
    with open(os.path.join(out_dir, "training_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[train:{name}] done - best epoch {best_epoch}, val macro-F1 {best_f1:.4f}")
    return meta


def _save_extractor(base_model, out_dir):
    from transformers import AutoFeatureExtractor

    try:
        AutoFeatureExtractor.from_pretrained(base_model).save_pretrained(out_dir)
    except Exception as exc:
        print(f"[train] could not save the feature extractor ({exc}); "
              f"runtime will fall back to {base_model}")


def load_trained(out_dir, device=None):
    """A saved member model, ready for inference."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForAudioClassification.from_pretrained(out_dir).to(device).eval()
    try:
        extractor = AutoFeatureExtractor.from_pretrained(out_dir)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    return model, extractor

"""End-to-end run: scan -> speaker-disjoint split -> train -> calibrate -> ensemble -> report.

Everything a run produces lands under runs/<run-name>/ so two runs never mix,
and every decision that was made from data (splits, calibrators, ensemble
weights, the winning combiner) is written down next to the numbers it produced.

Two rules this module exists to enforce:

  * Validation decides, test only reports. Calibrators are fitted on each
    member's own validation split; ensemble weights are fitted on a pooled
    validation split; the winning combiner is CHOSEN on a second, speaker
    disjoint slice of validation. The test split is touched exactly once, at
    report time.

  * No speaker crosses a split boundary, ever. assert_no_speaker_leakage runs
    on every split map, including the pooled ones built for the ensemble.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from . import datasets as ds
from .audio import DEFAULT_WINDOW_SPEC, WindowSpec
from .calibrate import apply_calibration, calibrate
from .ensemble import fit_all, select_best
from .labels import CANONICAL_EMOTIONS, describe_mapping
from .metrics import compare_table, evaluate, format_report
from .splits import (SPLITS, assert_no_speaker_leakage, iemocap_session_split, official_split,
                     speaker_disjoint_split, split_report)
from .train import BASE_MODEL, infer_logits, softmax, train_model
from .windows import aggregate_utterance_logits, build_window_index, set_shared_vad

RUNS_DIR = "runs"
MODELS_DIR = "models"
BASELINE_NAME = "baseline_iemocap_pretrained"


# ===================================================================== stage 1
def prepare(run_dir, roots, window_spec=DEFAULT_WINDOW_SPEC, vad_backend="silero",
            seed=1337, iemocap_sessions=True, msp_official_split=True,
            max_windows_per_utterance=8, scan_opts=None):
    """Scan every configured dataset, split it by speaker, index its windows."""
    os.makedirs(run_dir, exist_ok=True)
    cache_dir = os.path.join(run_dir, "window_cache")
    scan_opts = scan_opts or {}
    manifest = {"created_at": time.time(), "window_spec": window_spec.as_dict(),
                "vad_backend": vad_backend, "seed": seed, "datasets": {}}
    indices = {}

    for name, root in roots.items():
        print(f"\n[prepare] === {name} === {root}")
        utterances = ds.scan(name, root, **scan_opts.get(name, {}))
        summary = ds.summarise(utterances)
        print(f"[prepare] {name}: {summary['utterances']} utterances, "
              f"{summary['speakers']} speakers, classes {summary['per_class']}")

        if name == "iemocap" and iemocap_sessions:
            split_map, strategy = iemocap_session_split(utterances), "session (1-3 / 4 / 5)"
        elif name == "msp_podcast" and msp_official_split and any(u.split_hint for u in utterances):
            split_map, strategy = official_split(utterances), "official release split"
        else:
            split_map, strategy = (speaker_disjoint_split(utterances, seed=seed),
                                   "speaker-disjoint 70/15/15")
        assert_no_speaker_leakage(split_map)
        report = split_report(split_map)
        print(f"[prepare] {name}: {strategy} -> " +
              ", ".join(f"{s}={report[s]['utterances']}u/{report[s]['n_speakers']}spk"
                        for s in SPLITS))

        indices[name] = {}
        for split in SPLITS:
            indices[name][split] = build_window_index(
                split_map[split], window_spec, vad_backend,
                max_windows_per_utterance=max_windows_per_utterance,
                cache_dir=cache_dir, tag=f"{name}_{split}")

        manifest["datasets"][name] = {
            "root": os.path.abspath(root), "strategy": strategy,
            "label_mapping": describe_mapping(name, None),
            "scan_summary": summary, "split": report,
            "windows": {s: len(indices[name][s]) for s in SPLITS},
        }

    index_path = os.path.join(run_dir, "window_index.json")
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(indices, fh)
    with open(os.path.join(run_dir, "data_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n[prepare] wrote {index_path} and data_manifest.json")
    return indices, manifest


def load_indices(run_dir):
    with open(os.path.join(run_dir, "window_index.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===================================================================== stage 2
def resolve_members(indices, groups=None):
    """member name -> [dataset names]. Default: one member per dataset."""
    if not groups:
        return {name: [name] for name in indices}
    members = {}
    for spec in groups:
        name, _, parts = spec.partition("=")
        names = [p.strip() for p in parts.split("+") if p.strip()]
        missing = [p for p in names if p not in indices]
        if missing:
            raise ValueError(f"group {name!r} names unknown dataset(s) {missing}; "
                             f"prepared datasets are {sorted(indices)}")
        members[name.strip()] = names
    return members


def member_entries(indices, member_datasets, split):
    out = []
    for name in member_datasets:
        out.extend(indices[name][split])
    return out


def train_members(run_dir, indices, members, window_spec=DEFAULT_WINDOW_SPEC,
                  models_dir=MODELS_DIR, **train_kwargs):
    """Fine-tune one model per member. Each sees only its own datasets."""
    metas = {}
    for name, dataset_names in members.items():
        out_dir = os.path.join(models_dir, name)
        train_entries = member_entries(indices, dataset_names, "train")
        val_entries = member_entries(indices, dataset_names, "val")
        if not train_entries or not val_entries:
            print(f"[train] skipping member {name}: no windows")
            continue
        print(f"\n[train] === {name} === datasets={dataset_names}")
        metas[name] = train_model(name, train_entries, val_entries, out_dir,
                                  spec=window_spec, **train_kwargs)
        metas[name]["datasets"] = dataset_names
    with open(os.path.join(run_dir, "training_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(metas, fh, indent=2)
    return metas


# ===================================================================== stage 3
class _PretrainedBaseline:
    """The model the app ships today: superb/hubert-base-superb-er, unchanged.

    Its own head is IEMOCAP's four classes in ITS order, which is mapped to the
    canonical order the same way emotion_detector._EmotionModel does it today -
    so this really is the current system's number, not a re-implementation of it.
    """

    name = BASELINE_NAME

    def __init__(self, device, base_model=BASE_MODEL):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        import emotion_detector

        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(base_model)
        self.model = AutoModelForAudioClassification.from_pretrained(base_model).to(device).eval()
        self.index_to_emotion = {}
        for i, raw in self.model.config.id2label.items():
            mapped = emotion_detector._normalize_label(raw)
            if mapped:
                self.index_to_emotion[int(i)] = mapped
        missing = set(CANONICAL_EMOTIONS) - set(self.index_to_emotion.values())
        if missing:
            raise RuntimeError(f"{base_model} has no label for {sorted(missing)}")

    @torch.inference_mode()
    def pseudo_logits(self, entries, spec, batch_size=8):
        """log p over the canonical classes, so it flows through the same code path."""
        from torch.utils.data import DataLoader

        from .train import WindowDataset, collate

        loader = DataLoader(WindowDataset(entries, spec), batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, spec.window_samples))
        out = []
        for x, _ in loader:
            probs = torch.softmax(self.model(input_values=x.to(self.device)).logits.float(), -1)
            probs = probs.cpu().numpy()
            mapped = np.zeros((len(probs), len(CANONICAL_EMOTIONS)), dtype=np.float64)
            for idx, emotion in self.index_to_emotion.items():
                mapped[:, CANONICAL_EMOTIONS.index(emotion)] += probs[:, idx]
            mapped = mapped / np.clip(mapped.sum(axis=1, keepdims=True), 1e-12, None)
            out.append(np.log(np.clip(mapped, 1e-12, 1.0)))
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(CANONICAL_EMOTIONS)))


def _logit_path(run_dir, model_name, dataset, split):
    return os.path.join(run_dir, "logits", f"{model_name}__{dataset}__{split}.npz")


def infer_all(run_dir, indices, members, window_spec=DEFAULT_WINDOW_SPEC,
              models_dir=MODELS_DIR, device=None, batch_size=8, vad_backend="silero",
              include_baseline=True, splits=SPLITS, num_workers=0):
    """Every model's window logits on every dataset split, cached as .npz.

    Cross-corpus by design: each member is run over EVERY dataset's splits, not
    only its own. That is what makes the per-dataset comparison table possible
    and what shows whether a member generalises or has merely memorised its own
    corpus's recording conditions.
    """
    from .train import load_trained

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(run_dir, "logits"), exist_ok=True)
    set_shared_vad(vad_backend)

    runners = {}
    for name in members:
        out_dir = os.path.join(models_dir, name)
        if not os.path.isdir(out_dir):
            print(f"[infer] member {name}: no model at {out_dir} - skipped")
            continue
        model, _ = load_trained(out_dir, device)
        runners[name] = lambda entries, m=model: infer_logits(
            m, entries, window_spec, device, batch_size, num_workers)
    if include_baseline:
        baseline = _PretrainedBaseline(device)
        runners[BASELINE_NAME] = lambda entries, b=baseline: b.pseudo_logits(
            entries, window_spec, batch_size)
    if not runners:
        raise RuntimeError("no models to run inference with")

    for model_name, run in runners.items():
        for dataset, per_split in indices.items():
            for split in splits:
                entries = per_split[split]
                path = _logit_path(run_dir, model_name, dataset, split)
                if os.path.isfile(path):
                    continue
                if not entries:
                    continue
                started = time.time()
                logits = run(entries)
                np.savez_compressed(path, logits=logits.astype(np.float32))
                print(f"[infer] {model_name} x {dataset}/{split}: {len(entries)} windows "
                      f"in {time.time() - started:.0f}s -> {os.path.basename(path)}", flush=True)
    return run_dir


def load_logits(run_dir, model_name, dataset, split):
    path = _logit_path(run_dir, model_name, dataset, split)
    if not os.path.isfile(path):
        return None
    return np.load(path)["logits"].astype(np.float64)


def utterance_level(run_dir, model_name, indices, datasets, split):
    """Pooled utterance-level logits/labels for one model over several datasets."""
    logits, labels, speakers, sources = [], [], [], []
    for dataset in datasets:
        entries = indices[dataset][split]
        raw = load_logits(run_dir, model_name, dataset, split)
        if raw is None or not entries:
            continue
        agg, y, _, spk = aggregate_utterance_logits(raw, entries)
        logits.append(agg)
        labels.append(y)
        speakers.extend(spk)
        sources.extend([dataset] * len(y))
    if not logits:
        return None
    return (np.concatenate(logits), np.concatenate(labels), speakers, sources)


# ===================================================================== stage 4
def calibrate_members(run_dir, indices, members):
    """Fit each member's calibrator on ITS OWN validation split."""
    calibrations, reports = {}, {}
    for name, dataset_names in members.items():
        bundle = utterance_level(run_dir, name, indices, dataset_names, "val")
        if bundle is None:
            print(f"[calibrate] {name}: no validation logits - left uncalibrated")
            calibrations[name] = {"kind": "none"}
            continue
        logits, labels, _, _ = bundle
        calibration, report = calibrate(logits, labels)
        calibrations[name] = calibration
        reports[name] = report
        before = report["uncalibrated"]
        after = report.get("chosen_in_sample", before)
        print(f"[calibrate] {name}: chose {calibration['kind']:<12} "
              f"(n={report['n_val']}, out-of-fold NLL {report.get('chosen_cv_nll', float('nan')):.4f}) "
              f"| val NLL {before['nll']:.4f} -> {after['nll']:.4f}, "
              f"ECE {before['ece']:.4f} -> {after['ece']:.4f}")
        for candidate, detail in report["candidates"].items():
            if "skipped" in detail:
                print(f"[calibrate]   {candidate}: not offered - {detail['skipped']}")
            elif "cv_nll" in detail and candidate != "none":
                cv = detail["cv_nll"]
                verdict = "diverged on a fold" if not np.isfinite(cv) else f"{cv:.4f}"
                print(f"[calibrate]   {candidate}: out-of-fold NLL {verdict} "
                      f"vs in-sample {detail.get('in_sample_nll', float('nan')):.4f}")
    with open(os.path.join(run_dir, "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump({"calibrations": calibrations, "reports": reports}, fh, indent=2)
    return calibrations, reports


# ===================================================================== stage 5
def _split_val_by_speaker(speakers, seed=7):
    """Halve the pooled validation set by SPEAKER: one half fits, the other selects.

    Fitting the weights and then picking the winner on the same rows would make
    the most flexible combiner (stacking) win by construction. Splitting by
    speaker rather than by row keeps the selection half genuinely unseen.
    """
    unique = sorted(set(speakers))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    fit_speakers = {unique[i] for i in order[: max(1, len(unique) // 2)]}
    fit_idx = np.array([i for i, s in enumerate(speakers) if s in fit_speakers], dtype=int)
    sel_idx = np.array([i for i, s in enumerate(speakers) if s not in fit_speakers], dtype=int)
    if len(fit_idx) == 0 or len(sel_idx) == 0:
        # Only one validation speaker (or all their utterances on one side).
        # Selecting on the fitting rows favours the most flexible combiner, so
        # say so loudly rather than producing a quietly biased choice.
        print(f"[ensemble] WARNING: the pooled validation set has only "
              f"{len(unique)} speaker(s), so it cannot be halved speaker-disjointly. "
              f"Combiners will be fitted AND selected on the same rows, which biases "
              f"the choice towards stacking. Treat the selected combiner as unverified.")
        every = np.arange(len(speakers))
        return every, every
    return fit_idx, sel_idx


def build_ensemble(run_dir, indices, members, calibrations, out_dir=None, l2=1.0, seed=7):
    """Fit all three combiners on pooled validation, select on a held-out half of it."""
    # A member counts as present if ANY dataset has cached validation logits for
    # it. Checking only the first dataset in the index would drop a member whose
    # inference happens to be missing for that one corpus.
    member_names = [n for n in members
                    if any(load_logits(run_dir, n, d, "val") is not None for d in indices)]
    if len(member_names) < 2:
        raise RuntimeError(f"an ensemble needs at least 2 members with cached logits, "
                           f"found {member_names}")
    all_datasets = sorted(indices)

    reference, labels, speakers = None, None, None
    val_probs = {}
    for name in member_names:
        bundle = utterance_level(run_dir, name, indices, all_datasets, "val")
        logits, y, spk, _ = bundle
        if reference is None:
            reference, labels, speakers = name, y, spk
        elif not np.array_equal(y, labels):
            raise RuntimeError(f"member {name} and {reference} disagree on the pooled "
                               f"validation ordering - delete runs/*/logits and re-run infer")
        val_probs[name] = apply_calibration(logits, calibrations.get(name))

    fit_idx, sel_idx = _split_val_by_speaker(speakers, seed=seed)
    print(f"[ensemble] pooled validation: {len(labels)} utterances, "
          f"{len(set(speakers))} speakers -> {len(fit_idx)} to fit, {len(sel_idx)} to select")

    fit_probs = {n: p[fit_idx] for n, p in val_probs.items()}
    sel_probs = {n: p[sel_idx] for n, p in val_probs.items()}
    combiners = fit_all(fit_probs, labels[fit_idx], member_names, l2=l2)
    best_name, best, comparison = select_best(combiners, sel_probs, labels[sel_idx])

    print("\n[ensemble] combiner comparison on the held-out validation half")
    print(compare_table([comparison[n]["val_metrics"] for n in comparison]))
    for name in comparison:
        print(f"  {name:<10} weights {comparison[name]['weights']}")
    print(f"[ensemble] selected: {best_name}")

    # Refit the winner on the WHOLE validation pool now that it has been chosen.
    if best_name == "weighted":
        from .ensemble import WeightedAverageCombiner
        best = WeightedAverageCombiner.fit(val_probs, labels, member_names)
    elif best_name == "stacking":
        from .ensemble import StackingCombiner
        best = StackingCombiner.fit(val_probs, labels, member_names, l2=l2)
    print(f"[ensemble] refitted {best_name} on all {len(labels)} validation utterances: "
          f"weights {best.weight_report}")

    out_dir = out_dir or os.path.join(MODELS_DIR, "ensemble")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "version": 1,
        "created_at": time.time(),
        "run_dir": os.path.abspath(run_dir),
        "classes": list(CANONICAL_EMOTIONS),
        "sample_rate": 16000,
        "window_spec": DEFAULT_WINDOW_SPEC.as_dict(),
        "members": [{
            "name": name,
            "dir": os.path.relpath(os.path.abspath(os.path.join(MODELS_DIR, name)), out_dir),
            "datasets": members[name],
            "calibration": calibrations.get(name, {"kind": "none"}),
            "weight": best.weight_report.get(name, 0.0),
            "enabled": True,
        } for name in member_names],
        "combiner": best.to_dict(),
        "combiner_selection": {
            "selected": best_name,
            "metric": "macro_f1 on the held-out validation half",
            "comparison": {n: {"macro_f1": comparison[n]["val_metrics"]["macro_f1"],
                               "accuracy": comparison[n]["val_metrics"]["accuracy"],
                               "weights": comparison[n]["weights"]} for n in comparison},
        },
        "alternatives": {n: c.to_dict() for n, c in combiners.items()},
        "trained_on": {n: members[n] for n in member_names},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[ensemble] wrote {os.path.join(out_dir, 'manifest.json')}")
    return best, combiners, comparison, manifest


# ===================================================================== stage 6
def report(run_dir, indices, members, calibrations, combiners, selected_name,
           include_baseline=True, out_dir=None):
    """The comparison the whole exercise is for, measured on TEST, printed and saved."""
    all_datasets = sorted(indices)
    model_names = list(members) + ([BASELINE_NAME] if include_baseline else [])

    results, pooled = {}, {}
    for name in model_names:
        bundle = utterance_level(run_dir, name, indices, all_datasets, "test")
        if bundle is None:
            continue
        logits, labels, _, sources = bundle
        probs = apply_calibration(logits, calibrations.get(name))
        pooled[name] = (probs, labels, sources)
        results[name] = evaluate(probs, labels, label=name)

    member_names = [n for n in members if n in pooled]
    if not member_names:
        raise RuntimeError(
            "no member model has cached TEST logits - run `python -m ser infer` first "
            f"(looked under {os.path.join(run_dir, 'logits')})")
    labels = pooled[member_names[0]][1]
    sources = pooled[member_names[0]][2]
    ensemble_inputs = {n: pooled[n][0] for n in member_names}
    for kind, combiner in combiners.items():
        expected = set(getattr(combiner, "member_names", member_names))
        if expected - set(member_names):
            print(f"[report] skipping ensemble:{kind} - it was fitted on "
                  f"{sorted(expected)} but only {sorted(member_names)} have test logits")
            continue
        tag = f"ensemble:{kind}" + (" (SELECTED)" if kind == selected_name else "")
        probs = combiner.combine(ensemble_inputs)
        pooled[tag] = (probs, labels, sources)
        results[tag] = evaluate(probs, labels, label=tag)

    lines = []
    lines.append("=" * 78)
    lines.append("POOLED TEST SET - every model and every combiner on the same utterances")
    lines.append("=" * 78)
    lines.append(compare_table(list(results.values())))
    lines.append("")
    for name, result in results.items():
        lines.append(format_report(result))
        lines.append("")

    lines.append("=" * 78)
    lines.append("PER-DATASET TEST MACRO-F1 (rows = model, cols = test corpus)")
    lines.append("=" * 78)
    per_dataset = defaultdict(dict)
    sources_arr = np.asarray(sources)
    for name, (probs, y, _) in pooled.items():
        for dataset in all_datasets:
            mask = sources_arr == dataset
            if not mask.any():
                continue
            per_dataset[name][dataset] = evaluate(probs[mask], y[mask], label=f"{name}@{dataset}")
    width = max(len(n) for n in per_dataset) + 2
    lines.append(f"{'model':<{width}}" + "".join(f"{d:>16}" for d in all_datasets))
    lines.append("-" * (width + 16 * len(all_datasets)))
    for name in sorted(per_dataset, key=lambda n: -results[n]["macro_f1"]):
        row = "".join(f"{per_dataset[name].get(d, {}).get('macro_f1', float('nan')):>16.4f}"
                      for d in all_datasets)
        lines.append(f"{name:<{width}}" + row)
    lines.append("")
    lines.append("Read the diagonal as in-corpus skill and the off-diagonal as transfer. A member")
    lines.append("that only wins on its own corpus is still useful to the ensemble - it is the")
    lines.append("disagreement between members that the combiner is fitted to exploit.")

    text = "\n".join(lines)
    print("\n" + text)

    out_dir = out_dir or run_dir
    with open(os.path.join(out_dir, "comparison.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    payload = {
        "pooled_test": {n: r for n, r in results.items()},
        "per_dataset_test": {n: {d: v for d, v in row.items()} for n, row in per_dataset.items()},
        "selected_combiner": selected_name,
        "baseline": BASELINE_NAME if include_baseline else None,
    }
    with open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[report] wrote {os.path.join(out_dir, 'comparison.txt')} and comparison.json")
    return payload

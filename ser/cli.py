"""Command line for the multi-dataset ensemble.

    # one command, start to finish (datasets you actually have)
    python -m ser all --run r1 --cremad ~/data/CREMA-D --ravdess ~/data/RAVDESS \
                      --iemocap ~/data/IEMOCAP_full_release

    # or stage by stage, resuming where you stopped
    python -m ser prepare   --run r1 --cremad ... --ravdess ...
    python -m ser train     --run r1 --epochs 4
    python -m ser infer     --run r1
    python -m ser calibrate --run r1
    python -m ser ensemble  --run r1
    python -m ser report    --run r1

    # what the app will actually load, and what it says about one file
    python -m ser check
    python -m ser predict clip.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ser import pipeline                                    # noqa: E402
from ser.audio import WindowSpec                            # noqa: E402
from ser.datasets import DATASET_NAMES                      # noqa: E402
from ser.ensemble import combiner_from_dict                 # noqa: E402

DATASET_FLAGS = {name: f"--{name.replace('_', '-')}" for name in DATASET_NAMES}


def _add_dataset_flags(parser):
    for name, flag in DATASET_FLAGS.items():
        parser.add_argument(flag, dest=name, default=os.environ.get(f"SER_{name.upper()}_ROOT"),
                            help=f"root directory of the {name} corpus")


def _roots(args):
    roots = {name: getattr(args, name) for name in DATASET_NAMES if getattr(args, name, None)}
    if not roots:
        raise SystemExit(
            "No dataset roots given. Pass at least two of "
            + ", ".join(DATASET_FLAGS.values())
            + " (or set SER_<NAME>_ROOT). An ensemble of one model is just a model.")
    return roots


def _window_spec(args):
    return WindowSpec(window_s=args.window_seconds, hop_s=args.hop_seconds,
                      min_s=args.min_seconds, min_speech_ratio=args.min_speech_ratio)


def _run_dir(args):
    return os.path.join(pipeline.RUNS_DIR, args.run)


def _scan_opts(args):
    return {
        "iemocap": {"merge_excitement": not args.no_merge_excitement},
        "ravdess": {"include_song": args.ravdess_song, "include_calm": args.ravdess_calm},
        "msp_podcast": {"keep_unknown_speakers": args.keep_unknown_speakers},
    }


# --------------------------------------------------------------------- stages
def cmd_prepare(args):
    indices, _ = pipeline.prepare(
        _run_dir(args), _roots(args), window_spec=_window_spec(args),
        vad_backend=args.vad, seed=args.seed,
        iemocap_sessions=not args.iemocap_speaker_split,
        msp_official_split=not args.msp_resplit,
        max_windows_per_utterance=args.max_windows_per_utterance,
        scan_opts=_scan_opts(args))
    return indices


def cmd_train(args, indices=None):
    run_dir = _run_dir(args)
    indices = indices or pipeline.load_indices(run_dir)
    members = pipeline.resolve_members(indices, args.group)
    print(f"[train] members: { {k: v for k, v in members.items()} }")
    return pipeline.train_members(
        run_dir, indices, members, window_spec=_window_spec(args),
        models_dir=args.models_dir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, head_lr=args.head_lr, vad_backend=args.vad, seed=args.seed,
        grad_accum=args.grad_accum, num_workers=args.num_workers,
        max_train_windows=args.max_train_windows)


def cmd_infer(args, indices=None):
    run_dir = _run_dir(args)
    indices = indices or pipeline.load_indices(run_dir)
    members = pipeline.resolve_members(indices, args.group)
    return pipeline.infer_all(run_dir, indices, members, window_spec=_window_spec(args),
                              models_dir=args.models_dir, batch_size=args.batch_size,
                              vad_backend=args.vad, include_baseline=not args.no_baseline,
                              num_workers=args.num_workers)


def cmd_calibrate(args, indices=None):
    run_dir = _run_dir(args)
    indices = indices or pipeline.load_indices(run_dir)
    members = pipeline.resolve_members(indices, args.group)
    return pipeline.calibrate_members(run_dir, indices, members)


def cmd_ensemble(args, indices=None, calibrations=None):
    run_dir = _run_dir(args)
    indices = indices or pipeline.load_indices(run_dir)
    members = pipeline.resolve_members(indices, args.group)
    if calibrations is None:
        with open(os.path.join(run_dir, "calibration.json"), "r", encoding="utf-8") as fh:
            calibrations = json.load(fh)["calibrations"]
    return pipeline.build_ensemble(run_dir, indices, members, calibrations,
                                   out_dir=os.path.join(args.models_dir, "ensemble"),
                                   l2=args.stacking_l2, seed=args.seed)


def cmd_report(args, indices=None, calibrations=None, combiners=None, selected=None):
    run_dir = _run_dir(args)
    indices = indices or pipeline.load_indices(run_dir)
    members = pipeline.resolve_members(indices, args.group)
    manifest_path = os.path.join(args.models_dir, "ensemble", "manifest.json")
    if calibrations is None:
        with open(os.path.join(run_dir, "calibration.json"), "r", encoding="utf-8") as fh:
            calibrations = json.load(fh)["calibrations"]
    if combiners is None:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        combiners = {k: combiner_from_dict(v) for k, v in manifest["alternatives"].items()}
        selected = manifest["combiner_selection"]["selected"]
    payload = pipeline.report(run_dir, indices, members, calibrations, combiners, selected,
                              include_baseline=not args.no_baseline)
    _stamp_metrics_into_manifest(manifest_path, payload, selected)
    return payload


def _stamp_metrics_into_manifest(manifest_path, payload, selected):
    """Record the test numbers next to the ensemble the app will load."""
    if not os.path.isfile(manifest_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    key = f"ensemble:{selected} (SELECTED)"
    manifest["metrics"] = {
        "pooled_test": {n: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                            "uar": r["uar"], "ece": r["ece"]}
                        for n, r in payload["pooled_test"].items()},
        "ensemble": payload["pooled_test"].get(key, {}),
        "baseline": payload["pooled_test"].get(payload.get("baseline") or "", {}),
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[report] test metrics stamped into {manifest_path}")


def cmd_all(args):
    indices = cmd_prepare(args)
    cmd_train(args, indices)
    cmd_infer(args, indices)
    calibrations, _ = cmd_calibrate(args, indices)
    _, combiners, _, _ = cmd_ensemble(args, indices, calibrations)
    manifest_path = os.path.join(args.models_dir, "ensemble", "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        selected = json.load(fh)["combiner_selection"]["selected"]
    return cmd_report(args, indices, calibrations, combiners, selected)


# ------------------------------------------------------------------ utilities
def cmd_check(args):
    """Load the ensemble exactly as the app will, and report what came up."""
    from ser.runtime import EnsembleEmotionModel, EnsembleUnavailable

    path = args.manifest or os.path.join(args.models_dir, "ensemble", "manifest.json")
    try:
        model = EnsembleEmotionModel(path)
    except EnsembleUnavailable as exc:
        print(f"ensemble NOT available: {exc}")
        print("The app will keep using the single pretrained model - nothing is broken.")
        return 1
    print(json.dumps(model.describe(), indent=2))
    import numpy as np

    detail = model.predict_detailed(np.zeros(16000 * 3, dtype=np.float32))
    print(f"\nwarm-up inference on 3 s of silence took {detail['ms']:.0f} ms "
          f"({', '.join(f'{k}={v}ms' for k, v in detail['member_ms'].items())})")
    print("(Silence is never classified in the app - the VAD gate runs first.)")
    return 0


def cmd_predict(args):
    """Run the ensemble over one file the way the uploaded-file path does."""
    import numpy as np

    from ser.audio import DEFAULT_WINDOW_SPEC, load_audio_16k, speech_mask, speech_windows
    from ser.runtime import EnsembleEmotionModel
    from ser.windows import _make_vad

    model = EnsembleEmotionModel(args.manifest or
                                 os.path.join(args.models_dir, "ensemble", "manifest.json"))
    audio = load_audio_16k(args.path)
    vad = _make_vad(args.vad)
    flags = speech_mask(audio, vad) if vad is not None else None
    windows = speech_windows(audio, flags, DEFAULT_WINDOW_SPEC)
    if not windows:
        print("no speech windows passed the VAD gate - nothing to classify")
        return 1
    totals = None
    for window, at, ratio in windows:
        detail = model.predict_detailed(window)
        members = "  ".join(f"{n}:{d[max(d, key=d.get)]:.2f} {max(d, key=d.get)[:3]}"
                            for n, d in detail["members"].items())
        print(f"{at:6.1f}s speech={ratio * 100:3.0f}%  "
              f"ENSEMBLE {detail['emotion'].upper():<8}"
              f"{detail['ensemble'][detail['emotion']]:.2f}  |  {members}")
        vec = np.array([detail["ensemble"][e] for e in model.labels.values()])
        totals = vec if totals is None else totals + vec
    totals = totals / totals.sum()
    order = sorted(zip(model.labels.values(), totals), key=lambda kv: -kv[1])
    print("\nfile average: " + "  ".join(f"{e}={p:.3f}" for e, p in order))
    return 0


# ---------------------------------------------------------------------- parse
def build_parser():
    parser = argparse.ArgumentParser(prog="python -m ser", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, datasets=False):
        p.add_argument("--run", default="r1", help="run name under runs/ (default: r1)")
        p.add_argument("--models-dir", default=pipeline.MODELS_DIR)
        p.add_argument("--group", action="append",
                       help="member grouping, e.g. --group msp=msp_improv+msp_podcast "
                            "(repeatable; default is one member per dataset)")
        p.add_argument("--vad", default="silero", choices=["silero", "rms", "none"])
        p.add_argument("--seed", type=int, default=1337)
        p.add_argument("--window-seconds", type=float, default=3.0)
        p.add_argument("--hop-seconds", type=float, default=1.5)
        p.add_argument("--min-seconds", type=float, default=2.0)
        p.add_argument("--min-speech-ratio", type=float, default=0.5)
        p.add_argument("--max-windows-per-utterance", type=int, default=8)
        p.add_argument("--epochs", type=int, default=4)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--grad-accum", type=int, default=2)
        p.add_argument("--num-workers", type=int, default=0)
        p.add_argument("--lr", type=float, default=2e-5, help="learning rate for the HuBERT body")
        p.add_argument("--head-lr", type=float, default=1e-3)
        p.add_argument("--max-train-windows", type=int, default=None,
                       help="cap training windows per member (useful on MSP-Podcast)")
        p.add_argument("--stacking-l2", type=float, default=1.0)
        p.add_argument("--no-baseline", action="store_true",
                       help="skip the current superb/hubert-base-superb-er comparison")
        p.add_argument("--no-merge-excitement", action="store_true",
                       help="IEMOCAP: drop 'exc' instead of merging it into happy")
        p.add_argument("--ravdess-song", action="store_true")
        p.add_argument("--ravdess-calm", action="store_true",
                       help="RAVDESS: map 'calm' to neutral instead of dropping it")
        p.add_argument("--keep-unknown-speakers", action="store_true",
                       help="MSP-Podcast: keep clips whose SpkrID is Unknown")
        p.add_argument("--iemocap-speaker-split", action="store_true",
                       help="split IEMOCAP by speaker instead of by session")
        p.add_argument("--msp-resplit", action="store_true",
                       help="ignore MSP-Podcast's official split and cut a new speaker-disjoint one")
        if datasets:
            _add_dataset_flags(p)

    for name, fn, with_datasets in (
            ("prepare", cmd_prepare, True), ("train", cmd_train, False),
            ("infer", cmd_infer, False), ("calibrate", cmd_calibrate, False),
            ("ensemble", cmd_ensemble, False), ("report", cmd_report, False),
            ("all", cmd_all, True)):
        p = sub.add_parser(name, help=fn.__doc__ or name)
        common(p, datasets=with_datasets)
        p.set_defaults(func=fn)

    p = sub.add_parser("check", help="load the ensemble the way the app does")
    p.add_argument("--models-dir", default=pipeline.MODELS_DIR)
    p.add_argument("--manifest", default=None)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("predict", help="classify one audio/video file, showing every member")
    p.add_argument("path")
    p.add_argument("--models-dir", default=pipeline.MODELS_DIR)
    p.add_argument("--manifest", default=None)
    p.add_argument("--vad", default="silero", choices=["silero", "rms", "none"])
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())

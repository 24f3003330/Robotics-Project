"""Turning labelled utterances into the 2-4 s VAD-gated windows the models see.

Built once per split and cached to JSON, because running Silero over every
utterance of five corpora is far slower than the training epoch that follows.
The cache key includes the window spec and the VAD backend, so changing either
rebuilds rather than silently reusing a mismatched index.

The window, not the utterance, is the training example. Evaluation still scores
UTTERANCES (see aggregate_utterance) - a model that is right about 9 windows of
a 30 s clip and wrong about 1 should be counted right once, and that is also
exactly what the live app does when it averages windows.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

from .audio import (DEFAULT_WINDOW_SPEC, SAMPLE_RATE, WindowSpec, crop_to_speech,
                    load_audio_16k, speech_mask, window_bounds)
from .labels import EMOTION_TO_INDEX

VAD_CHUNK_SAMPLES = 512


def _make_vad(backend):
    """Reuse the app's own VAD implementations so training gates exactly like runtime."""
    if backend in (None, "none"):
        return None
    import emotion_detector

    vad, warning = emotion_detector._make_vad(backend)
    if warning:
        print(f"[windows] {warning}")
    return vad


def _cache_key(utterances, spec, vad_backend, max_windows):
    digest = hashlib.sha1()
    digest.update(json.dumps(spec.as_dict(), sort_keys=True).encode())
    digest.update(f"|{vad_backend}|{max_windows}|{len(utterances)}".encode())
    for u in utterances[:2000]:
        digest.update(f"{u.path}|{u.label}".encode())
    return digest.hexdigest()[:16]


def build_window_index(utterances, spec=DEFAULT_WINDOW_SPEC, vad_backend="silero",
                       max_windows_per_utterance=8, cache_dir=None, tag="", verbose=True):
    """[{path, start, end, label, label_idx, speaker, dataset, utt_id}] for every kept window.

    An utterance that yields no window that passes the speech gate is reported
    and dropped - it is either silence or something the VAD does not consider
    speech, and either way it is not training data for a speech model.
    """
    key = _cache_key(utterances, spec, vad_backend, max_windows_per_utterance)
    cache_path = os.path.join(cache_dir, f"windows_{tag}_{key}.json") if cache_dir else None
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if verbose:
            print(f"[windows] {tag}: {len(cached)} windows from cache {os.path.basename(cache_path)}")
        return cached

    vad = _make_vad(vad_backend)
    index, dropped, failed = [], 0, 0
    for i, u in enumerate(utterances):
        if verbose and i and i % 500 == 0:
            print(f"[windows] {tag}: {i}/{len(utterances)} utterances, {len(index)} windows", flush=True)
        try:
            audio = load_audio_16k(u.path)
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"[windows] cannot read {u.path}: {exc}")
            continue
        flags = speech_mask(audio, vad, VAD_CHUNK_SAMPLES) if vad is not None else None
        if flags is not None and flags.any():
            audio_offset_source = crop_to_speech(audio, flags, spec, VAD_CHUNK_SAMPLES)
        else:
            audio_offset_source = audio
        if len(audio_offset_source) < spec.min_samples:
            # Short but real utterances are common in CREMA-D/RAVDESS; pad with
            # the clip's own trailing audio rather than discarding the example.
            if len(audio_offset_source) >= int(1.0 * SAMPLE_RATE):
                bounds = [(0, len(audio_offset_source))]
            else:
                dropped += 1
                continue
        else:
            bounds = window_bounds(len(audio_offset_source), spec)
        if not bounds:
            dropped += 1
            continue
        if len(bounds) > max_windows_per_utterance:
            pick = np.linspace(0, len(bounds) - 1, max_windows_per_utterance).round().astype(int)
            bounds = [bounds[i] for i in sorted(set(pick.tolist()))]
        for start, end in bounds:
            index.append({
                "path": u.path, "start": int(start), "end": int(end),
                "label": u.label, "label_idx": EMOTION_TO_INDEX[u.label],
                "speaker": u.speaker, "dataset": u.dataset,
                "utt_id": os.path.splitext(os.path.basename(u.path))[0],
                "cropped": bool(flags is not None and flags.any()),
            })
    if verbose:
        print(f"[windows] {tag}: {len(index)} windows from {len(utterances)} utterances "
              f"({dropped} utterances had no speech window, {failed} unreadable)")
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh)
    return index


class _AudioLRU:
    """Decoded-audio cache: one file usually produces several consecutive windows."""

    def __init__(self, maxsize=48):
        self.maxsize = maxsize
        self._store = OrderedDict()

    def get(self, path, spec, cropped):
        key = (path, cropped)
        hit = self._store.get(key)
        if hit is not None:
            self._store.move_to_end(key)
            return hit
        audio = load_audio_16k(path)
        if cropped:
            # The index recorded start/end against the CROPPED waveform, so a
            # reader that cannot crop would slice the wrong samples and quietly
            # train on misaligned audio. get_shared_vad() rebuilds the VAD in
            # this process rather than returning None, and if it genuinely
            # cannot, we raise instead of returning a plausible-looking window.
            vad = get_shared_vad()
            if vad is None:
                raise RuntimeError(
                    f"{path} was indexed with VAD cropping but no VAD is available in this "
                    f"process, so the recorded window offsets cannot be reproduced. Call "
                    f"set_shared_vad(backend) first, or rebuild the index with --vad none.")
            flags = speech_mask(audio, vad, VAD_CHUNK_SAMPLES)
            if flags.any():
                audio = crop_to_speech(audio, flags, spec, VAD_CHUNK_SAMPLES)
        self._store[key] = audio
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return audio


# Backend name AND instance. The name is what survives into a DataLoader worker
# process: a Silero VAD is a torch module that is not built to be shared across
# processes, so each worker lazily builds its own from the recorded name. Before
# this, a worker simply found None and skipped cropping - which did not fail, it
# just fed the model the wrong three seconds of every utterance.
_SHARED_VAD = [None]
_SHARED_VAD_BACKEND = [None]


def set_shared_vad(backend):
    _SHARED_VAD_BACKEND[0] = backend
    _SHARED_VAD[0] = _make_vad(backend)
    return _SHARED_VAD[0]


def get_shared_vad():
    """The process-local VAD, built on first use from the recorded backend name."""
    if _SHARED_VAD[0] is None and _SHARED_VAD_BACKEND[0] not in (None, "none"):
        _SHARED_VAD[0] = _make_vad(_SHARED_VAD_BACKEND[0])
    return _SHARED_VAD[0]


def fetch_window(cache, entry, spec):
    """One index entry -> the float32 window the model is fed, length-normalised."""
    audio = cache.get(entry["path"], spec, entry.get("cropped", False))
    window = audio[entry["start"]:entry["end"]]
    target = spec.window_samples
    if len(window) < spec.min_samples:
        window = np.pad(window, (0, max(0, spec.min_samples - len(window))))
    if len(window) > target:
        window = window[:target]
    peak = float(np.abs(window).max()) if window.size else 0.0
    if peak > 1.0:
        window = window / peak
    return np.asarray(window, dtype=np.float32)


def aggregate_utterance(window_probs, entries):
    """Window probabilities -> one probability vector per utterance.

    Mean over the utterance's windows, which is precisely what the uploaded-file
    path does at inference time, so validation numbers describe the behaviour
    the app actually ships.
    """
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        groups[(entry["path"], entry["utt_id"])].append(i)
    keys = sorted(groups)
    probs = np.stack([np.asarray(window_probs)[groups[k]].mean(axis=0) for k in keys])
    labels = np.asarray([entries[groups[k][0]]["label_idx"] for k in keys], dtype=int)
    speakers = [entries[groups[k][0]]["speaker"] for k in keys]
    datasets = [entries[groups[k][0]]["dataset"] for k in keys]
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return probs, labels, keys, speakers, datasets


def aggregate_utterance_logits(window_logits, entries):
    """Same grouping, but averaging LOGITS - what calibration must be fitted on."""
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        groups[(entry["path"], entry["utt_id"])].append(i)
    keys = sorted(groups)
    logits = np.stack([np.asarray(window_logits)[groups[k]].mean(axis=0) for k in keys])
    labels = np.asarray([entries[groups[k][0]]["label_idx"] for k in keys], dtype=int)
    speakers = [entries[groups[k][0]]["speaker"] for k in keys]
    return logits, labels, keys, speakers

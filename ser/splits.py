"""Speaker-disjoint train / val / test splits, and the check that proves it.

The split is over SPEAKERS, never over utterances. A random utterance split of
any of these corpora leaks: every speaker records dozens of clips, so the model
learns the voice rather than the emotion and validation accuracy becomes
meaningless. `assert_no_speaker_leakage` is run on every split that is written
to disk, and it raises rather than warns.

IEMOCAP also gets a session-level option (train on Sessions 1-3, validate on 4,
test on 5), which is the protocol most published IEMOCAP numbers use and which
is strictly stronger than speaker-disjoint: the two actors of a session share
their recording conditions and their dialogue partner, so keeping a session
whole removes that shared context too.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


class LeakageError(RuntimeError):
    """The same speaker appears in more than one split."""


def assert_no_speaker_leakage(split_map):
    """split_map: {"train": [Utterance], "val": [...], "test": [...]}. Raises on overlap."""
    speakers = {name: {u.speaker for u in utts} for name, utts in split_map.items()}
    problems = []
    names = sorted(speakers)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = speakers[a] & speakers[b]
            if shared:
                problems.append(f"{a} and {b} share {len(shared)} speaker(s): "
                                f"{', '.join(sorted(shared)[:8])}"
                                f"{' ...' if len(shared) > 8 else ''}")
    if problems:
        raise LeakageError("speaker leakage between splits -> " + "; ".join(problems))

    paths = defaultdict(list)
    for name, utts in split_map.items():
        for u in utts:
            paths[u.path].append(name)
    duplicated = {p: s for p, s in paths.items() if len(set(s)) > 1}
    if duplicated:
        raise LeakageError(f"{len(duplicated)} audio file(s) appear in more than one split, "
                           f"e.g. {next(iter(duplicated))}")
    return True


def _speaker_profiles(utterances):
    """speaker -> (count, {label: count})"""
    profile = defaultdict(Counter)
    for u in utterances:
        profile[u.speaker][u.label] += 1
    return profile


def speaker_disjoint_split(utterances, ratios=DEFAULT_RATIOS, seed=1337, min_speakers_per_split=1):
    """Assign whole speakers to train/val/test, matching `ratios` by utterance count.

    Speakers are shuffled, then placed greedily into whichever split is furthest
    below its target share. Greedy-on-deficit (rather than a plain sequential
    cut) matters for the small acted corpora: RAVDESS has 24 speakers and
    MSP-IMPROV only 12, so one unluckily large speaker can otherwise swallow a
    whole validation set.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    profile = _speaker_profiles(utterances)
    if len(profile) < len(SPLITS) * min_speakers_per_split:
        raise ValueError(
            f"only {len(profile)} distinct speaker(s) available; a speaker-disjoint "
            f"train/val/test split needs at least {len(SPLITS) * min_speakers_per_split}. "
            f"Group this dataset with another one instead of splitting it alone."
        )

    total = len(utterances)
    targets = dict(zip(SPLITS, (r * total for r in ratios)))
    assigned = {name: [] for name in SPLITS}
    counts = {name: 0 for name in SPLITS}

    speakers = sorted(profile, key=lambda s: -sum(profile[s].values()))
    rng = random.Random(seed)
    rng.shuffle(speakers)
    # Largest speakers first keeps the tail free to fine-tune the balance.
    speakers.sort(key=lambda s: -sum(profile[s].values()))

    for speaker in speakers:
        size = sum(profile[speaker].values())
        empty = [n for n in SPLITS if not assigned[n] and targets[n] > 0]
        if empty:                                   # guarantee every split is non-empty
            choice = max(empty, key=lambda n: targets[n])
        else:
            choice = max(SPLITS, key=lambda n: (targets[n] - counts[n]) / max(targets[n], 1.0))
        assigned[choice].append(speaker)
        counts[choice] += size

    by_speaker = defaultdict(list)
    for u in utterances:
        by_speaker[u.speaker].append(u)
    split_map = {name: [u for s in sorted(spk) for u in by_speaker[s]]
                 for name, spk in assigned.items()}
    assert_no_speaker_leakage(split_map)
    return split_map


def iemocap_session_split(utterances, val_session="Session4", test_session="Session5"):
    """Sessions 1-3 train, 4 val, 5 test - the standard IEMOCAP protocol."""
    split_map = {name: [] for name in SPLITS}
    for u in utterances:
        if u.session == test_session:
            split_map["test"].append(u)
        elif u.session == val_session:
            split_map["val"].append(u)
        else:
            split_map["train"].append(u)
    if not all(split_map.values()):
        empty = [n for n, v in split_map.items() if not v]
        raise ValueError(f"IEMOCAP session split left {empty} empty - are the session folders named "
                         f"Session1..Session5? (found: {sorted({u.session for u in utterances})})")
    assert_no_speaker_leakage(split_map)
    return split_map


def official_split(utterances):
    """Use the corpus's own split_hint (MSP-Podcast). Unhinted utterances are an error."""
    split_map = {name: [] for name in SPLITS}
    unhinted = 0
    for u in utterances:
        if u.split_hint in split_map:
            split_map[u.split_hint].append(u)
        else:
            unhinted += 1
    if unhinted:
        print(f"[split] {unhinted} utterance(s) had no official split and were left out")
    if not all(split_map.values()):
        empty = [n for n, v in split_map.items() if not v]
        raise ValueError(f"official split left {empty} empty")
    assert_no_speaker_leakage(split_map)
    return split_map


def split_report(split_map):
    """Per-split counts, speakers and class balance - written into the manifest."""
    report = {}
    for name in SPLITS:
        utts = split_map.get(name, [])
        report[name] = {
            "utterances": len(utts),
            "speakers": sorted({u.speaker for u in utts}),
            "n_speakers": len({u.speaker for u in utts}),
            "per_class": dict(sorted(Counter(u.label for u in utts).items())),
            "per_dataset": dict(sorted(Counter(u.dataset for u in utts).items())),
        }
    return report

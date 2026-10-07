"""
voice_age/selection.py - age-group mapping, speaker-wise split, subset selection.

  1. map raw Common Voice age labels to age groups (config.AGE_GROUP_MAPPING)
  2. one label per SPEAKER: the majority of that speaker's clip labels; clips that
     disagree with it are dropped, a tied speaker is dropped
  3. split SPEAKERS (not clips) into train / validation / test, stratified by age
     group, with a fixed seed; verify_disjoint() fails loudly on any overlap
  4. choose a manageable subset: per split and age group at most
     AGE_MAX_SAMPLES_PER_CLASS x split fraction clips, at most
     AGE_MAX_CLIPS_PER_SPEAKER clips per speaker. Row groups rich in rare age
     groups are visited first so few audio row groups have to be downloaded.

No audio, speaker id or file name is ever a model input; client_id is used only
to split.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import config


class SpeakerLeakageError(AssertionError):
    """The same speaker appears in more than one split."""


def map_age(raw, mapping=None):
    return (mapping or config.AGE_GROUP_MAPPING).get((raw or "").strip().lower())


def label_speakers(df, mapping=None):
    """Adds `group`; keeps clips whose group equals their speaker's majority group.
    Returns (clips DataFrame, report dict)."""
    d = df.copy()
    d["group"] = d["age"].map(lambda a: map_age(a, mapping))
    d = d[d["group"].notna()]
    counts = d.groupby(["client_id", "group"]).size().unstack(fill_value=0)
    top = counts.max(axis=1)
    ties = (counts.eq(top, axis=0).sum(axis=1) > 1)
    speaker_group = counts.idxmax(axis=1)[~ties]
    d = d[d["client_id"].isin(speaker_group.index)]
    before = len(d)
    d = d[d["group"].to_numpy() == speaker_group.reindex(d["client_id"]).to_numpy()]
    report = {"speakers_with_conflicting_labels": int((counts.gt(0).sum(axis=1) > 1).sum()),
              "speakers_dropped_tie": int(ties.sum()), "clips_dropped_minority_label": int(before - len(d))}
    return d.reset_index(drop=True), report


def group_stats(d, groups=None):
    groups = groups or config.AGE_GROUPS
    return {g: {"clips": int((d["group"] == g).sum()),
                "speakers": int(d.loc[d["group"] == g, "client_id"].nunique())} for g in groups}


def check_groups(d, groups=None, min_speakers=config.MIN_SPEAKERS_PER_GROUP):
    """Groups with too few speakers - the mapping must merge them before training."""
    return [g for g, s in group_stats(d, groups).items() if s["speakers"] < min_speakers]


def split_speakers(d, fractions=None, seed=config.RANDOM_SEED):
    """{speaker: split}, stratified by age group, deterministic for a seed."""
    fractions = fractions or config.SPLIT_FRACTIONS
    rng = np.random.default_rng(seed)
    spk = d.groupby("client_id")["group"].first()
    out = {}
    for g in sorted(spk.unique()):
        ids = np.array(sorted(spk.index[spk == g]))
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(n * fractions["validation"]))
        n_test = int(round(n * fractions["test"]))
        for i, s in enumerate(ids):
            out[s] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    return out


def verify_disjoint(splits_to_speakers):
    """Raises SpeakerLeakageError if any speaker is in two splits. Returns the audit dict."""
    names = list(splits_to_speakers)
    audit = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = set(splits_to_speakers[a]) & set(splits_to_speakers[b])
            audit[f"{a} ∩ {b}"] = len(inter)
            if inter:
                raise SpeakerLeakageError(f"speaker leakage: {len(inter)} speakers in both {a} and {b}, "
                                          f"e.g. {sorted(inter)[:3]}")
    return audit


def select_subset(d, speaker_split, max_per_class=config.AGE_MAX_SAMPLES_PER_CLASS,
                  max_per_speaker=config.AGE_MAX_CLIPS_PER_SPEAKER, fractions=None, seed=config.RANDOM_SEED):
    """Pick clips per (split, group) quota. Pass 1 allows one clip per speaker (as many
    different voices as possible), pass 2 tops up to max_per_speaker."""
    fractions = fractions or config.SPLIT_FRACTIONS
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["split"] = d["client_id"].map(speaker_split)
    groups = sorted(d["group"].unique())
    quota = {(s, g): int(round(max_per_class * f)) for s, f in fractions.items() for g in groups}
    avail = Counter(d["group"])
    rarity = {g: 1.0 / avail[g] for g in groups}
    d["rg_key"] = d["shard"] + "#" + d["row_group"].astype(str)
    d["_r"] = rng.random(len(d))
    score = d.groupby("rg_key")["group"].apply(lambda s: sum(rarity[g] for g in s))
    order = score.sample(frac=1.0, random_state=seed).sort_values(ascending=False, kind="stable").index
    by_rg = {k: v.sort_values("_r") for k, v in d.groupby("rg_key")}
    taken, per_speaker, chosen = Counter(), Counter(), set()
    for cap in (1, max_per_speaker):
        for key in order:
            part = by_rg[key]
            for idx, split, group, speaker in zip(part.index, part["split"], part["group"], part["client_id"]):
                q = (split, group)
                if taken[q] >= quota[q] or per_speaker[speaker] >= cap or idx in chosen:
                    continue
                taken[q] += 1
                per_speaker[speaker] += 1
                chosen.add(idx)
            if all(taken[q] >= quota[q] for q in quota):
                break
    sel = d.loc[sorted(chosen)].drop(columns=["_r"])
    return sel.reset_index(drop=True), {"quota": {f"{s}/{g}": n for (s, g), n in quota.items()},
                                        "row_groups_needed": int(sel["rg_key"].nunique())}


def split_report(sel):
    rows = {}
    for s in ("train", "validation", "test"):
        part = sel[sel["split"] == s]
        rows[s] = {"speakers": int(part["client_id"].nunique()), "clips": int(len(part)),
                   "by_group": {g: {"clips": int((part["group"] == g).sum()),
                                    "speakers": int(part.loc[part["group"] == g, "client_id"].nunique())}
                                for g in sorted(sel["group"].unique(), key=config.AGE_GROUPS.index)}}
    return rows

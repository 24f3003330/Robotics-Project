"""
python scripts/inspect_voice_age_dataset.py [--language en] [--split validated] [--max-shards N]

Inspects the Common Voice dataset used for voice AGE-GROUP estimation BEFORE any
training: verifies access and the real schema (audio / age / client_id), then
scans the metadata columns of every shard (no audio is downloaded) and prints
usable samples, age labels and their distribution, unique speakers, gender and
locale information. The full metadata scan is cached in data/voice_age/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_age import config  # noqa: E402
from voice_age.dataset import (DatasetAccessError, SchemaError, check_schema, dataset_access,  # noqa: E402
                               list_shards, open_shard, scan_metadata, speaker_age_consistency, summarize)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=config.AGE_DATASET)
    p.add_argument("--language", default=config.AGE_LANGUAGE)
    p.add_argument("--split", default=config.AGE_SOURCE_SPLIT)
    p.add_argument("--max-shards", type=int, help="scan only the first N shards (quick look, not cached)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    config.AGE_DATASET = args.dataset
    try:
        info = dataset_access(args.dataset)
        shards = list_shards(args.language, args.split, args.dataset)
        _, pf = open_shard(*shards[0], repo_id=args.dataset)
        schema = check_schema(pf)
    except (DatasetAccessError, SchemaError) as exc:
        print(f"\nCANNOT USE THE DATASET: {exc}")
        return 2
    print(f"Dataset          : {args.dataset}  ({config.AGE_DATASET_VERSION}; gated={info.gated}; "
          f"last modified {info.last_modified})")
    print(f"Config / split   : {args.language} / {args.split}: {len(shards)} Parquet shards, "
          f"{sum(s for _, s in shards) / 1e9:.1f} GB on the Hub (not downloaded)")
    cols = ", ".join("{} ({})".format(c, schema["types"][c]) for c in schema["columns"])
    print(f"Columns          : {cols}")
    print(f"Audio            : encoded bytes, declared sampling rate {schema['declared_audio_sampling_rate']} Hz")
    print(f"Required columns : {', '.join(config.REQUIRED_COLUMNS)} - present\n")

    df = scan_metadata(args.language, args.split, max_shards=args.max_shards, workers=args.workers)
    s = summarize(df)
    inconsistent, labelled_speakers = speaker_age_consistency(df)
    mapped = df[df["age"].map(lambda a: config.AGE_GROUP_MAPPING.get(a)).notna()]
    print(f"\nClips            : {s['clips']}  from {s['speakers']} unique speakers (client_id)")
    print(f"Clips with age   : {s['clips_with_age']} ({s['clips_with_age'] / max(s['clips'], 1):.1%}) "
          f"from {s['speakers_with_age']} speakers")
    print(f"Usable (mapped)  : {len(mapped)} clips, {mapped['client_id'].nunique()} speakers")
    print(f"Clips per speaker: median {s['clips_per_speaker_median']:.0f}, max {s['clips_per_speaker_max']}")
    print(f"Speakers with >1 age label: {inconsistent} of {labelled_speakers}")
    print("\nAge label (raw)   clips      speakers   -> age group")
    for age, n in s["age_clips"].items():
        print(f"  {age:<14} {n:>9} {s['age_speakers'].get(age, 0):>10}   -> {config.AGE_GROUP_MAPPING.get(age)}")
    print("\nAge group         clips      speakers")
    for g in config.AGE_GROUPS:
        part = mapped[mapped["age"].map(config.AGE_GROUP_MAPPING) == g]
        print(f"  {g:<14} {len(part):>9} {part['client_id'].nunique():>10}")
    print(f"\nGender (self-reported): {s['gender_clips']}")
    print(f"Locale: {args.language} only (one Common Voice config per language)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

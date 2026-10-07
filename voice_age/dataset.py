"""
voice_age/dataset.py - programmatic Common Voice access (no manual download).

  list_shards()        Parquet shards of <language>/<split> on the Hub (sizes included)
  check_schema()       verifies the shard schema has audio / age / client_id (fails loudly)
  scan_metadata()      reads ONLY client_id / age / gender / path of every row group of
                       every shard (~4 MB per 500 MB shard, via HTTP range requests) and
                       caches the result locally as a small Parquet table
  fetch_audio()        downloads the audio column of just the row groups that contain
                       selected clips and keeps only those clips' original MP3 bytes

Access problems (gated / private dataset, missing token, dataset removed) raise
DatasetAccessError with the exact step needed to fix them.
"""

import json
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .remote_parquet import RangeFile, column_chunk_ranges, hub_file_url, hub_headers


class DatasetAccessError(RuntimeError):
    """The dataset cannot be read; the message says what the user must do."""


class SchemaError(RuntimeError):
    """The dataset does not have the columns the age pipeline needs."""


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def dataset_access(repo_id=config.AGE_DATASET):
    """Returns the Hub info for the dataset or raises DatasetAccessError with instructions."""
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    try:
        info = _api().dataset_info(repo_id)
    except GatedRepoError as exc:
        raise DatasetAccessError(
            f"{repo_id} is gated. Accept its terms at https://huggingface.co/datasets/{repo_id} while logged in, "
            f"create a read token at https://huggingface.co/settings/tokens and run `hf auth login` "
            f"(or set HF_TOKEN) - then re-run.") from exc
    except RepositoryNotFoundError as exc:
        raise DatasetAccessError(f"{repo_id} was not found on the Hugging Face Hub (removed or private). "
                                 f"Set VOICE_AGE_DATASET to another Common Voice mirror.") from exc
    except HfHubHTTPError as exc:
        raise DatasetAccessError(f"Cannot reach the Hugging Face Hub for {repo_id}: {exc}") from exc
    if info.gated and not hub_headers():
        raise DatasetAccessError(
            f"{repo_id} is gated ({info.gated}) and no Hugging Face token is configured. Accept the dataset terms "
            f"on its page, then run `hf auth login` (or set HF_TOKEN) and re-run.")
    return info


def list_shards(language=config.AGE_LANGUAGE, split=config.AGE_SOURCE_SPLIT, repo_id=config.AGE_DATASET):
    """[(path, size_bytes)] of the Parquet shards for one language / split."""
    dataset_access(repo_id)
    files = [f for f in _api().list_repo_tree(repo_id, path_in_repo=language, repo_type="dataset", recursive=True)
             if getattr(f, "size", None) is not None and f.path.endswith(".parquet")
             and os.path.basename(f.path).startswith(split + "-")]
    if not files:
        raise DatasetAccessError(f"No '{split}' Parquet shards for language '{language}' in {repo_id}.")
    return sorted((f.path, f.size) for f in files)


def open_shard(path, size=None, repo_id=config.AGE_DATASET):
    import pyarrow.parquet as pq
    try:
        f = RangeFile(hub_file_url(repo_id, path), size=None, headers=hub_headers())
    except PermissionError as exc:
        raise DatasetAccessError(f"{exc}. Run `hf auth login` with a token that can read {repo_id}.") from exc
    return f, pq.ParquetFile(f, pre_buffer=False)


def check_schema(parquet_file, required=config.REQUIRED_COLUMNS):
    """Verify the real schema (never assumed). Returns a printable description."""
    schema = parquet_file.schema_arrow
    names = schema.names
    missing = [c for c in required if c not in names]
    if missing:
        raise SchemaError(f"dataset schema lacks required columns {missing}; columns are {names}")
    audio = schema.field("audio").type
    if not (hasattr(audio, "get_field_index") and audio.get_field_index("bytes") >= 0):
        raise SchemaError(f"'audio' column is not a struct with encoded 'bytes' (type {audio})")
    for c in ("age", "client_id"):
        if str(schema.field(c).type) not in ("string", "large_string"):
            raise SchemaError(f"'{c}' column is {schema.field(c).type}, expected a string")
    hf_meta = (schema.metadata or {}).get(b"huggingface")
    sampling_rate = None
    if hf_meta:
        try:
            sampling_rate = json.loads(hf_meta)["info"]["features"]["audio"].get("sampling_rate")
        except Exception:
            pass
    return {"columns": names, "types": {f.name: str(f.type) for f in schema},
            "declared_audio_sampling_rate": sampling_rate}


# ---------------------------------------------------------------------------
# metadata scan
# ---------------------------------------------------------------------------
def _scan_one(path, size):
    f, pf = open_shard(path, size)
    md = pf.metadata
    f.prefetch(column_chunk_ranges(md, range(md.num_row_groups), set(config.METADATA_COLUMNS)), workers=24)
    rows = []
    for rg in range(md.num_row_groups):
        t = pf.read_row_group(rg, columns=list(config.METADATA_COLUMNS)).to_pydict()
        for i in range(len(t["client_id"])):
            rows.append((path, rg, i, t["client_id"][i], t["age"][i] or "", t["gender"][i] or "", t["path"][i]))
    return rows, f.bytes_fetched


def _shard_cache(path):
    return os.path.join(config.DATA_DIR, "metadata_shards", path.replace("/", "__"))


def _scan_one_cached(path, size, attempts=4):
    """Scan one shard (resumable: finished shards are kept on disk, retried on network errors)."""
    import pandas as pd
    out = _shard_cache(path)
    if os.path.isfile(out):
        return pd.read_parquet(out).itertuples(index=False, name=None), 0
    last = None
    for attempt in range(attempts):
        try:
            rows, fetched = _scan_one(path, size)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pd.DataFrame(rows, columns=_COLUMNS).to_parquet(out + ".tmp", index=False)
            os.replace(out + ".tmp", out)
            return rows, fetched
        except (DatasetAccessError, SchemaError):
            raise
        except Exception as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    raise IOError(f"{path}: failed after {attempts} attempts: {type(last).__name__}: {last}")


_COLUMNS = ["shard", "row_group", "row", "client_id", "age", "gender", "path"]


def scan_metadata(language=config.AGE_LANGUAGE, split=config.AGE_SOURCE_SPLIT, max_shards=None, workers=6,
                  cache=True, log=print):
    """DataFrame (shard, row_group, row, client_id, age, gender, path) of every clip of the split.
    Cached in data/voice_age/; a partial scan (max_shards) is never cached."""
    import pandas as pd
    out = config.METADATA_CACHE.format(language=language, split=split)
    if cache and max_shards is None and os.path.isfile(out):
        log(f"[voice-age] metadata cache: {os.path.relpath(out, config.PROJECT_DIR)}")
        return pd.read_parquet(out)
    shards = list_shards(language, split)
    if max_shards:
        shards = shards[:max_shards]
    _, pf = open_shard(*shards[0])
    check_schema(pf)
    log(f"[voice-age] scanning metadata of {len(shards)} shards of {config.AGE_DATASET}/{language}/{split} "
        f"({sum(s for _, s in shards) / 1e9:.1f} GB on the Hub; only metadata columns are transferred)")
    t0, rows, fetched, done = time.time(), [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one_cached if cache else _scan_one, p, s): p for p, s in shards}
        for fut in as_completed(futures):
            r, b = fut.result()
            rows.extend(list(r))
            fetched += b
            done += 1
            if done % 10 == 0 or done == len(shards):
                log(f"    {done}/{len(shards)} shards, {len(rows)} clips, {fetched / 1e6:.0f} MB transferred, "
                    f"{time.time() - t0:.0f} s")
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df = df.sort_values(["shard", "row_group", "row"]).reset_index(drop=True)
    if cache and max_shards is None:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_parquet(out, index=False)
        log(f"[voice-age] cached metadata -> {os.path.relpath(out, config.PROJECT_DIR)}")
        import shutil
        shutil.rmtree(os.path.join(config.DATA_DIR, "metadata_shards"), ignore_errors=True)   # no duplicate copies
    return df


# ---------------------------------------------------------------------------
# audio fetch (selected clips only)
# ---------------------------------------------------------------------------
def audio_cache_path(shard, row_group):
    name = shard.replace("/", "__").replace(".parquet", "") + f"__rg{row_group:04d}.pkl"
    return os.path.join(config.AUDIO_CACHE_DIR, name)


def _fetch_shard(shard, groups, log):
    """groups: {row_group: [row, ...]} -> writes one small pickle per row group with
    {row: (path, mp3_bytes)} for the selected rows only. Returns bytes transferred."""
    todo = {rg: rows for rg, rows in groups.items() if not os.path.isfile(audio_cache_path(shard, rg))}
    if not todo:
        return 0
    f, pf = open_shard(shard)
    md = pf.metadata
    os.makedirs(config.AUDIO_CACHE_DIR, exist_ok=True)
    for rg in sorted(todo):
        f.prefetch(column_chunk_ranges(md, [rg], {"audio"}), workers=4)
        col = pf.read_row_group(rg, columns=["audio"]).column("audio")
        keep = {}
        for r in todo[rg]:
            item = col[r].as_py()
            keep[int(r)] = (item.get("path"), item.get("bytes"))
        tmp = audio_cache_path(shard, rg) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(keep, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, audio_cache_path(shard, rg))
        f.drop_cache()                       # never hold more than one row group of audio in memory
    return f.bytes_fetched


def fetch_audio(selection, workers=6, log=print):
    """Download the audio of the selected clips (DataFrame with shard / row_group / row)."""
    by_shard = {}
    for (shard, rg), part in selection.groupby(["shard", "row_group"]):
        by_shard.setdefault(shard, {})[int(rg)] = sorted(int(r) for r in part["row"])
    n_groups = sum(len(g) for g in by_shard.values())
    cached = sum(os.path.isfile(audio_cache_path(s, rg)) for s, g in by_shard.items() for rg in g)
    log(f"[voice-age] audio: {len(selection)} clips in {n_groups} row groups of {len(by_shard)} shards "
        f"({cached} row groups already cached)")
    t0, total, done, failed = time.time(), 0, 0, {}

    def attempt(shard, groups, tries=5):
        last = None
        for k in range(tries):
            try:
                return _fetch_shard(shard, groups, log)        # resumes: finished row groups are cached
            except (DatasetAccessError, SchemaError):
                raise
            except Exception as exc:
                last = exc
                time.sleep(5 * (k + 1))
        raise IOError(f"{type(last).__name__}: {last}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(attempt, s, g): s for s, g in by_shard.items()}
        for fut in as_completed(futures):
            try:
                total += fut.result()
            except (DatasetAccessError, SchemaError):
                raise
            except Exception as exc:
                failed[futures[fut]] = str(exc)
                log(f"    FAILED {futures[fut]}: {exc}")
            done += 1
            if done % 10 == 0 or done == len(futures):
                log(f"    {done}/{len(futures)} shards, {total / 1e6:.0f} MB transferred, {time.time() - t0:.0f} s")
    if failed:
        log(f"[voice-age] {len(failed)} shards could not be fetched; their clips are skipped "
            f"(re-run to resume): {sorted(failed)}")
    return total


def load_clip_bytes(shard, row_group, row):
    with open(audio_cache_path(shard, row_group), "rb") as fh:
        return pickle.load(fh)[int(row)]


def summarize(df):
    """Inspection numbers for a metadata table."""
    labelled = df[df["age"] != ""]
    return {
        "clips": int(len(df)), "speakers": int(df["client_id"].nunique()),
        "clips_with_age": int(len(labelled)), "speakers_with_age": int(labelled["client_id"].nunique()),
        "age_clips": dict(Counter(labelled["age"]).most_common()),
        "age_speakers": {a: int(n) for a, n in labelled.groupby("age")["client_id"].nunique()
                         .sort_values(ascending=False).items()},
        "gender_clips": dict(Counter(df["gender"].replace("", "(blank)")).most_common()),
        "clips_per_speaker_median": float(df.groupby("client_id").size().median()),
        "clips_per_speaker_max": int(df.groupby("client_id").size().max()),
    }


def speaker_age_consistency(df):
    """Speakers whose clips carry more than one age label (re-recorded years apart, or inconsistent)."""
    labelled = df[df["age"] != ""]
    n = labelled.groupby("client_id")["age"].nunique()
    return int((n > 1).sum()), int(len(n))


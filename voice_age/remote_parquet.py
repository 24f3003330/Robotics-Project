"""
voice_age/remote_parquet.py - read selected columns / row groups of a Parquet file
on the Hugging Face Hub WITHOUT downloading the file.

A Common Voice English shard is ~500 MB, almost all of it MP3 audio. The age
pipeline needs (1) the tiny metadata columns of every row group and (2) the
audio of a few selected rows. Parquet stores each column of each row group as
one contiguous "column chunk" whose byte offset and length are in the footer,
so this module:

  1. reads the footer (a few hundred KB) with HTTP Range requests,
  2. computes the exact byte ranges of the wanted column chunks,
  3. fetches those ranges in parallel and serves them to pyarrow from memory.

Nothing is written to disk; transfer is roughly the size of the wanted chunks.
"""

import bisect
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import requests


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, backed by a range cache."""

    def __init__(self, url, size=None, session=None, headers=None, timeout=60, retries=3):
        super().__init__()
        self.url = url
        self.session = session or requests.Session()
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.retries = retries
        self.pos = 0
        self._starts = []          # sorted cached range starts
        self._chunks = {}          # start -> bytes
        self._lock = threading.Lock()
        self.bytes_fetched = 0
        self.size = size if size is not None else self._head_size()

    # ----------------------------------------------------------------- HTTP
    def _head_size(self):
        import time
        for attempt in range(self.retries + 2):
            try:
                r = self.session.head(self.url, headers=self.headers, allow_redirects=True, timeout=self.timeout)
                break
            except requests.exceptions.RequestException:
                if attempt == self.retries + 1:
                    raise
                time.sleep(2 ** attempt)              # transient DNS / network failure
        if r.status_code in (401, 403):
            raise PermissionError(f"HTTP {r.status_code} for {self.url} - authentication required")
        r.raise_for_status()
        if r.url != self.url and "Authorization" not in self.headers:
            self.url = r.url          # follow the Hub redirect once (CDN URL), not on every request
        return int(r.headers["Content-Length"])

    def _get(self, start, length):
        if length <= 0:
            return b""
        last = None
        for _ in range(self.retries + 2):
            try:
                r = self.session.get(self.url, timeout=self.timeout, allow_redirects=True,
                                     headers=dict(self.headers, Range=f"bytes={start}-{start + length - 1}"))
                if r.status_code == 401 or r.status_code == 403:
                    raise PermissionError(f"HTTP {r.status_code} for {self.url} - authentication required")
                r.raise_for_status()
                data = r.content
                if r.status_code != 206 and len(data) != length:
                    data = data[start:start + length]          # server ignored Range (should not happen)
                if len(data) != length:
                    raise IOError(f"short read: wanted {length} bytes, got {len(data)}")
                with self._lock:
                    self.bytes_fetched += len(data)
                return data
            except PermissionError:
                raise
            except Exception as exc:      # transient network error: retry with backoff
                last = exc
                import time
                time.sleep(1.5 ** _)
        raise IOError(f"range request failed after {self.retries} tries: {last}")

    def prefetch(self, ranges, workers=16, merge_gap=64 * 1024):
        """Fetch many (start, length) ranges in parallel into the cache. Ranges closer
        than `merge_gap` bytes are merged into one request (fewer round trips)."""
        todo = [(s, n) for s, n in merge_ranges(ranges, merge_gap) if n > 0 and not self._cached(s, n)]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (s, _), data in zip(todo, pool.map(lambda r: self._get(*r), todo)):
                self._store(s, data)

    def _store(self, start, data):
        with self._lock:
            if start not in self._chunks:
                bisect.insort(self._starts, start)
            self._chunks[start] = data

    def _cached(self, start, length):
        with self._lock:
            i = bisect.bisect_right(self._starts, start) - 1
            if i < 0:
                return None
            s = self._starts[i]
            data = self._chunks[s]
            if start + length <= s + len(data):
                return data[start - s:start - s + length]
        return None

    def drop_cache(self):
        with self._lock:
            self._starts, self._chunks = [], {}

    # ----------------------------------------------------------------- io API
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        data = self._cached(self.pos, n)
        if data is None:
            data = self._get(self.pos, n)
            self._store(self.pos, data)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def merge_ranges(ranges, gap):
    merged = []
    for s, n in sorted(r for r in ranges if r[1] > 0):
        if merged and s <= merged[-1][0] + merged[-1][1] + gap:
            ps, pn = merged[-1]
            merged[-1] = (ps, max(pn, s + n - ps))
        else:
            merged.append((s, n))
    return merged


def column_chunk_ranges(metadata, row_groups, columns):
    """Byte ranges (start, length) of the given columns in the given row groups.
    `columns` are top-level names; nested children (e.g. audio.bytes) are included."""
    out = []
    for rg in row_groups:
        meta = metadata.row_group(rg)
        for j in range(meta.num_columns):
            col = meta.column(j)
            if col.path_in_schema.split(".")[0] not in columns:
                continue
            start = col.data_page_offset
            if col.has_dictionary_page and col.dictionary_page_offset and col.dictionary_page_offset > 0:
                start = min(start, col.dictionary_page_offset)
            out.append((start, col.total_compressed_size))
    return out


def hub_file_url(repo_id, filename, revision=None):
    from huggingface_hub import hf_hub_url
    return hf_hub_url(repo_id, filename, repo_type="dataset", revision=revision)


def hub_headers():
    """Authorization header if the user is logged in to Hugging Face (not needed for public data)."""
    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:
        token = None
    return {"Authorization": f"Bearer {token}"} if token else {}

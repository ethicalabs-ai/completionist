"""Reusable lazy streaming access to local and remote datasets."""

import itertools
import os

from datasets import load_dataset

from completionist.utils import handle_error


def _load(ref):
    """Return a streaming dataset handle for a local path or a HF name."""
    if os.path.exists(ref):
        if os.path.isdir(ref):
            return load_dataset(ref, streaming=True)
        ext = os.path.splitext(ref)[1].lower()
        if ext == ".parquet":
            return load_dataset("parquet", data_files=ref, streaming=True)
        if ext in (".jsonl", ".json"):
            return load_dataset("json", data_files=ref, streaming=True)
        handle_error(f"Unsupported format '{ext}'. Use .parquet or .jsonl.")
    return load_dataset(ref, streaming=True)


def _pick_split(split, splits):
    """Resolve the split to use, honoring an explicit value when given."""
    if split is not None:
        if splits and split not in splits:
            handle_error(f"Split '{split}' not found. Available: {', '.join(splits)}")
        return split
    for candidate in ("train", "test", "validation"):
        if candidate in splits:
            return candidate
    return splits[0] if splits else None


def _num_examples(ds, split):
    """Return the metadata sample count for a split, or None if unknown."""
    if split is None:
        return None
    info = getattr(ds, "info", None)
    if info is None or not info.splits:
        return None
    split_info = info.splits.get(split)
    return split_info.num_examples if split_info else None


def _file_row_count(ref):
    """Row count from a local parquet footer (metadata only), or None.

    Reads only the parquet footer, never the data, so this does not
    double-load the file. jsonl is skipped: counting lines would require
    reading the whole file again.
    """
    if os.path.splitext(ref)[1].lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(ref).metadata.num_rows
    return None


def _trim_total(total, index, limit):
    """Adjust a total count for --index and --limit."""
    if total is None:
        return None
    total = max(0, total - index)
    if limit is not None:
        total = min(total, limit)
    return total


class DatasetStream:
    """Lazy, switchable streaming source for a local or remote dataset.

    Handles remote HF names, local directories following the HF split layout,
    and single local parquet/jsonl files uniformly. Iteration is forward-only;
    ``switch_split()`` re-opens the stream for the next split.
    """

    def __init__(self, ref, split=None, index=0, limit=None):
        self.ref = ref
        self.index = index
        self.limit = limit
        self._requested = split
        self._loaded = _load(ref)
        if isinstance(self._loaded, dict):
            self.splits = list(self._loaded)
        else:
            info = getattr(self._loaded, "info", None)
            self.splits = list(info.splits) if info is not None and info.splits else []
        self.split = None
        self.total = None
        self.exhausted = False
        self._iterator = None
        self._open(_pick_split(self._requested, self.splits))

    def _open(self, split):
        self.split = split
        if isinstance(self._loaded, dict):
            if split is None:
                self._iterator = iter(())
                self.total = 0
                self.exhausted = True
                return
            ds = self._loaded[split]
        else:
            ds = self._loaded
        stop = None if self.limit is None else self.index + self.limit
        self._iterator = itertools.islice(iter(ds), self.index, stop)
        total = _num_examples(ds, split)
        if total is None and os.path.isfile(self.ref):
            total = _file_row_count(self.ref)
        self.total = _trim_total(total, self.index, self.limit)
        self.exhausted = False

    @property
    def cache_key(self):
        """Stable key identifying this dataset/split/view, for cache naming."""
        return f"{self.ref}|{self.split}|{self.index}|{self.limit}"

    def next(self):
        """Return the next record, raising StopIteration when exhausted."""
        try:
            return next(self._iterator)
        except StopIteration:
            self.exhausted = True
            raise

    def switch_split(self):
        """Move to the next split, returning False when there is only one."""
        if len(self.splits) < 2:
            return False
        nxt = self.splits[(self.splits.index(self.split) + 1) % len(self.splits)]
        self._open(nxt)
        return True

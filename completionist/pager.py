"""Bounded, disk-spilled read-ahead window over a forward-only source."""

import collections
import hashlib
import json
import os
import tempfile

DEFAULT_BUFFER = 200
DEFAULT_REFILL = 50


class _SpillFile:
    """Append-only spill file in /tmp, named by a hash of the source.

    Records evicted from the in-memory window are written here; the file is
    only read back when navigating backwards past the window.
    """

    def __init__(self, key):
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        self.path = os.path.join(
            tempfile.gettempdir(), f"completionist-inspect-{digest}.spill"
        )
        self._f = open(self.path, "w+b")

    def append(self, row):
        offset = self._f.tell()
        self._f.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")
        return offset

    def read(self, offset):
        self._f.flush()
        self._f.seek(offset)
        return json.loads(self._f.readline().decode("utf-8"))

    def remove(self):
        self._f.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


class Pager:
    """Bounded sliding window over a forward-only source, disk-spilled.

    Recent records stay in memory; older ones spill to disk so backward
    navigation works without holding the whole dataset in RAM. Reads ahead in
    chunks (``refill``) only when the cursor nears the window's right edge.
    """

    def __init__(
        self, source, first, buffer_size=DEFAULT_BUFFER, refill=DEFAULT_REFILL
    ):
        self.source = source
        self.buffer_size = buffer_size
        self.refill = refill
        self.buffer = collections.deque([first])
        self.spill = _SpillFile(source.cache_key)
        self.offsets = []  # absolute index -> byte offset in the spill file
        self.consumed = 1  # records fetched so far
        self.cursor = 0  # absolute index of the current row

    def _fetch_one(self):
        try:
            row = self.source.next()
        except StopIteration:
            return False
        if len(self.buffer) == self.buffer_size:
            self.offsets.append(self.spill.append(self.buffer.popleft()))
        self.buffer.append(row)
        self.consumed += 1
        return True

    def _read(self, i):
        mem_base = self.consumed - len(self.buffer)
        if i >= mem_base:
            return self.buffer[i - mem_base]
        return self.spill.read(self.offsets[i])

    def current(self):
        return self._read(self.cursor)

    def next(self):
        if self.cursor < self.consumed - 1:
            self.cursor += 1
        elif self._fetch_one():
            self.cursor += 1
        if self.consumed - 1 - self.cursor < self.refill:
            self._refill()

    def prev(self):
        if self.cursor > 0:
            self.cursor -= 1

    def _refill(self):
        for _ in range(self.refill):
            if not self._fetch_one():
                break

    @property
    def total(self):
        return self.consumed if self.source.exhausted else self.source.total

    def close(self):
        self.spill.remove()

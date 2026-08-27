import os

from completionist.pager import Pager


class _FakeSource:
    def __init__(self, data):
        self._it = iter(data)
        self.total = len(data)
        self.exhausted = False
        self.cache_key = "test-key"

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self.exhausted = True
            raise


def _pager(data, buffer_size, refill):
    """Build a Pager like inspect_cmd does: first record consumed from source."""
    src = _FakeSource(data)
    first = src.next()
    return Pager(src, first, buffer_size=buffer_size, refill=refill)


def test_forward_navigation():
    pager = _pager([{"x": i} for i in range(6)], buffer_size=3, refill=1)
    for _ in range(5):
        pager.next()
    assert pager.current() == {"x": 5}
    pager.close()


def test_backward_reads_spill():
    pager = _pager([{"x": i} for i in range(6)], buffer_size=3, refill=1)
    for _ in range(5):
        pager.next()
    for i in range(5, -1, -1):
        assert pager.current() == {"x": i}
        pager.prev()
    pager.close()


def test_total_on_exhaustion():
    pager = _pager([{"x": i} for i in range(3)], buffer_size=2, refill=2)
    prev = -1
    while pager.cursor != prev:
        prev = pager.cursor
        pager.next()
    assert pager.source.exhausted
    assert pager.total == 3
    pager.close()


def test_close_removes_spill():
    pager = _pager([{"x": i} for i in range(3)], buffer_size=2, refill=2)
    path = pager.spill.path
    assert os.path.exists(path)
    pager.close()
    assert not os.path.exists(path)


def test_prev_stays_at_start():
    pager = _pager([{"x": i} for i in range(3)], buffer_size=2, refill=2)
    pager.prev()
    assert pager.cursor == 0
    pager.close()


def test_next_stays_at_end():
    pager = _pager([{"x": i} for i in range(3)], buffer_size=2, refill=2)
    for _ in range(3):
        pager.next()
    assert pager.cursor == 2
    pager.next()
    assert pager.cursor == 2
    pager.close()


def test_buffer_stays_bounded():
    pager = _pager([{"x": i} for i in range(30)], buffer_size=10, refill=5)
    prev = -1
    while pager.cursor != prev:
        prev = pager.cursor
        pager.next()
    assert pager.source.exhausted
    assert pager.total == 30
    assert len(pager.buffer) <= 10
    pager.close()

import pytest
from datasets import Dataset

from completionist import dataset_stream
from completionist.dataset_stream import DatasetStream


def _fake_dataset(split_data):
    """Build a fake streaming load_dataset result: {split: iterable-with-info}."""

    class _SplitInfo:
        def __init__(self, n):
            self.num_examples = n

    class _Info:
        def __init__(self, name, n):
            self.splits = {name: _SplitInfo(n)}

    class _Iter:
        def __init__(self, rows, info):
            self._rows = rows
            self.info = info

        def __iter__(self):
            return iter(self._rows)

    return {k: _Iter(v, _Info(k, len(v))) for k, v in split_data.items()}


def test_load_local_parquet(tmp_path):
    path = tmp_path / "data.parquet"
    Dataset.from_list(
        [{"prompt": str(i), "completion": str(i)} for i in range(5)]
    ).to_parquet(str(path))
    src = DatasetStream(str(path))
    assert src.splits == ["train"]
    # Total comes from the parquet footer (metadata only, no data re-read).
    assert src.total == 5
    rows = []
    while True:
        try:
            rows.append(src.next())
        except StopIteration:
            break
    assert len(rows) == 5
    assert src.exhausted


def test_load_local_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    Dataset.from_list(
        [{"prompt": str(i), "completion": str(i)} for i in range(3)]
    ).to_json(str(path))
    src = DatasetStream(str(path))
    assert src.total is None
    assert src.next() == {"prompt": "0", "completion": "0"}


def test_load_local_directory_with_splits(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    Dataset.from_list([{"x": i} for i in range(4)]).to_parquet(
        str(tmp_path / "train" / "part-00000.parquet")
    )
    Dataset.from_list([{"x": i} for i in range(2)]).to_parquet(
        str(tmp_path / "test" / "part-00000.parquet")
    )
    src = DatasetStream(str(tmp_path))
    assert src.splits == ["train", "test"]
    assert src.split == "train"
    assert src.total is None
    assert src.switch_split() is True
    assert src.split == "test"
    assert src.total is None


def test_remote_splits_and_total(monkeypatch):
    monkeypatch.setattr(
        dataset_stream,
        "load_dataset",
        lambda *a, **k: _fake_dataset(
            {"train": [{"x": 1}, {"x": 2}], "test": [{"x": 3}]}
        ),
    )
    src = DatasetStream("some/dataset")
    assert src.splits == ["train", "test"]
    assert src.split == "train"
    assert src.total == 2
    assert src.next() == {"x": 1}
    assert src.next() == {"x": 2}
    assert src.switch_split() is True
    assert src.split == "test"
    assert src.total == 1
    assert src.next() == {"x": 3}


def test_remote_single_split(monkeypatch):
    monkeypatch.setattr(
        dataset_stream,
        "load_dataset",
        lambda *a, **k: _fake_dataset({"train": [{"x": 1}]}),
    )
    src = DatasetStream("some/dataset")
    assert src.switch_split() is False


def test_remote_index_and_limit(monkeypatch):
    rows = [{"x": i} for i in range(10)]
    monkeypatch.setattr(
        dataset_stream, "load_dataset", lambda *a, **k: _fake_dataset({"train": rows})
    )
    src = DatasetStream("some/dataset", index=3, limit=4)
    assert src.total == 4
    got = []
    while True:
        try:
            got.append(src.next())
        except StopIteration:
            break
    assert got == [{"x": 3}, {"x": 4}, {"x": 5}, {"x": 6}]


def test_pick_split_explicit():
    assert dataset_stream._pick_split("test", ["train", "test"]) == "test"


def test_pick_split_auto_prefers_train():
    assert dataset_stream._pick_split(None, ["test", "train"]) == "train"


def test_pick_split_auto_falls_back():
    assert dataset_stream._pick_split(None, ["validation"]) == "validation"


def test_pick_split_auto_empty():
    assert dataset_stream._pick_split(None, []) is None


def test_pick_split_explicit_missing_errors():
    with pytest.raises(SystemExit):
        dataset_stream._pick_split("nope", ["train"])


def test_trim_total():
    assert dataset_stream._trim_total(10, 2, 3) == 3
    assert dataset_stream._trim_total(10, 2, None) == 8
    assert dataset_stream._trim_total(10, 20, None) == 0
    assert dataset_stream._trim_total(None, 0, None) is None


def test_unsupported_extension_errors(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(SystemExit):
        DatasetStream(str(path))


def test_cache_key_changes_with_split(monkeypatch):
    monkeypatch.setattr(
        dataset_stream,
        "load_dataset",
        lambda *a, **k: _fake_dataset({"train": [{"x": 1}], "test": [{"x": 2}]}),
    )
    src = DatasetStream("some/dataset", index=3, limit=4)
    key = src.cache_key
    assert "some/dataset" in key
    assert "train" in key
    src.switch_split()
    assert src.cache_key != key
    assert "test" in src.cache_key


def test_num_examples_returns_none_without_info():
    class _NoInfo:
        pass

    assert dataset_stream._num_examples(_NoInfo(), "train") is None
    assert dataset_stream._num_examples(_NoInfo(), None) is None


def test_switch_split_cycles(monkeypatch):
    monkeypatch.setattr(
        dataset_stream,
        "load_dataset",
        lambda *a, **k: _fake_dataset({"a": [1], "b": [2]}),
    )
    src = DatasetStream("some/dataset")
    assert src.split == "a"
    assert src.switch_split() is True
    assert src.split == "b"
    assert src.switch_split() is True
    assert src.split == "a"

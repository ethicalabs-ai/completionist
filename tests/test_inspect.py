from click.testing import CliRunner
from datasets import Dataset

from completionist import dataset_stream as ds_module
from completionist.commands import inspect


def _conversation_row():
    return {
        "topic": "climate policy",
        "messages": [
            {"role": "user", "content": "Is degrowth viable?"},
            {"role": "assistant", "content": "Partly, but it needs a just transition."},
        ],
    }


def _prompt_completion_row():
    return {"prompt": "What is justice?", "completion": "A contested concept."}


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


def test_inspect_dumps_conversation(tmp_path):
    path = tmp_path / "conv.parquet"
    Dataset.from_list([_conversation_row()]).to_parquet(str(path))

    result = CliRunner().invoke(inspect.inspect_cmd, [str(path)])
    assert result.exit_code == 0, result.output
    assert "== conversation ==" in result.output
    assert "user: Is degrowth viable?" in result.output
    assert "assistant: Partly, but it needs a just transition." in result.output
    assert "== raw ==" in result.output


def test_inspect_dumps_prompt_completion(tmp_path):
    path = tmp_path / "pc.parquet"
    Dataset.from_list([_prompt_completion_row()]).to_parquet(str(path))

    result = CliRunner().invoke(inspect.inspect_cmd, [str(path)])
    assert result.exit_code == 0, result.output
    assert "== prompt ==" in result.output
    assert "== completion ==" in result.output
    assert "user: What is justice?" in result.output


def test_inspect_dumps_local_total_from_footer(tmp_path):
    # Total for a local parquet comes from its footer metadata (no re-read).
    path = tmp_path / "many.parquet"
    Dataset.from_list(
        [{"prompt": str(i), "completion": str(i)} for i in range(3)]
    ).to_parquet(str(path))

    result = CliRunner().invoke(inspect.inspect_cmd, [str(path)])
    assert result.exit_code == 0, result.output
    assert "[1/3]" in result.output
    assert "[3/3]" in result.output


def test_inspect_dumps_remote_total_from_metadata(monkeypatch):
    monkeypatch.setattr(
        ds_module,
        "load_dataset",
        lambda *a, **k: _fake_dataset(
            {
                "train": [
                    {"prompt": "a", "completion": "1"},
                    {"prompt": "b", "completion": "2"},
                ]
            }
        ),
    )

    result = CliRunner().invoke(inspect.inspect_cmd, ["some/dataset"])
    assert result.exit_code == 0, result.output
    assert "[1/2]" in result.output
    assert "[2/2]" in result.output


def test_inspect_dumps_with_index_and_limit(tmp_path):
    path = tmp_path / "many.parquet"
    Dataset.from_list(
        [{"prompt": str(i), "completion": str(i)} for i in range(5)]
    ).to_parquet(str(path))

    result = CliRunner().invoke(
        inspect.inspect_cmd, [str(path), "--index", "1", "--limit", "2"]
    )
    assert result.exit_code == 0, result.output
    assert '"prompt": "1"' in result.output
    assert '"prompt": "2"' in result.output
    assert '"prompt": "4"' not in result.output


def test_inspect_errors_on_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    result = CliRunner().invoke(inspect.inspect_cmd, [str(path)])
    assert result.exit_code == 1
    assert "Unsupported format" in result.output


def test_inspect_errors_on_empty_dataset(monkeypatch):
    monkeypatch.setattr(
        ds_module, "load_dataset", lambda *a, **k: _fake_dataset({"train": []})
    )
    result = CliRunner().invoke(inspect.inspect_cmd, ["some/dataset"])
    assert result.exit_code == 1
    assert "No samples found" in result.output


def test_inspect_errors_on_invalid_index(tmp_path):
    path = tmp_path / "data.parquet"
    Dataset.from_list([_prompt_completion_row()]).to_parquet(str(path))
    result = CliRunner().invoke(inspect.inspect_cmd, [str(path), "--index", "-1"])
    assert result.exit_code == 1


def test_wrap_segments_hanging_indent():
    lines = inspect._wrap_segments(
        [("user", "a long message that definitely wraps")], width=20
    )
    assert lines[0][0] == "user"
    assert lines[0][1].startswith("user: ")
    for style, text in lines[1:]:
        assert style == "user"
        assert text.startswith(" " * len("user: "))


def test_wrap_segments_raw_verbatim():
    lines = inspect._wrap_segments([("raw", "line1\nline2")], width=80)
    assert [text for _, text in lines] == ["line1", "line2"]


def test_wrap_segments_plain_wraps():
    lines = inspect._wrap_segments([("plain", "aaaa bbbb cccc")], width=5)
    assert lines == [("plain", "aaaa"), ("plain", "bbbb"), ("plain", "cccc")]

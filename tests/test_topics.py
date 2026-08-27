from completionist.commands import topics
from completionist.commands.topics import TopicList


def _call_generate_batch(monkeypatch, result):
    monkeypatch.setattr(topics, "get_completion", lambda **kwargs: result)
    return topics._generate_batch(
        model_name="test-model",
        api_url="http://localhost:11434/v1",
        system_prompt="generate {batch_size} topics about {categories}",
        batch_size=5,
        temperature=0.9,
        top_p=0.95,
        max_tokens=4096,
        reasoning=None,
        hf_api_token=None,
        openai_api_token=None,
    )


def test_load_existing_txt(tmp_path):
    f = tmp_path / "topics.txt"
    f.write_text("topic one\ntopic two\n\ntopic one\n")
    assert topics._load_existing(str(f), ".txt") == {"topic one", "topic two"}


def test_load_existing_jsonl(tmp_path):
    f = tmp_path / "topics.jsonl"
    f.write_text('{"topic": "a"}\n{"topic": "b"}\nnot-json\n')
    assert topics._load_existing(str(f), ".jsonl") == {"a", "b"}


def test_load_existing_missing_file(tmp_path):
    assert topics._load_existing(str(tmp_path / "nope.txt"), ".txt") == set()


def test_write_topics_file_txt(tmp_path):
    f = tmp_path / "topics.txt"
    topics._write_topics_file(str(f), ".txt", ["a", "b", "c"])
    assert f.read_text() == "a\nb\nc\n"


def test_write_topics_file_jsonl(tmp_path):
    f = tmp_path / "topics.jsonl"
    topics._write_topics_file(str(f), ".jsonl", ["a", "b"])
    assert f.read_text() == '{"topic": "a"}\n{"topic": "b"}\n'


def test_write_then_load_roundtrip(tmp_path):
    f = tmp_path / "topics.txt"
    topics._write_topics_file(str(f), ".txt", ["x", "y", "z"])
    assert topics._load_existing(str(f), ".txt") == {"x", "y", "z"}


def test_topic_list_schema():
    assert TopicList(topics=["a", "b"]).topics == ["a", "b"]


def test_generate_batch_returns_topics(monkeypatch):
    result = _call_generate_batch(monkeypatch, TopicList(topics=["a", "b"]))
    assert result == ["a", "b"]


def test_generate_batch_returns_empty_on_none(monkeypatch):
    result = _call_generate_batch(monkeypatch, None)
    assert result == []


def test_generate_batch_fallback_json_string(monkeypatch):
    result = _call_generate_batch(monkeypatch, '{"topics": ["a", "b"]}')
    assert result == ["a", "b"]

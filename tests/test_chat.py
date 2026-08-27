import time

from click.testing import CliRunner

from completionist.commands import chat
from completionist.commands.chat import ChatConversation, ChatMessage


def _llm_config(**overrides):
    config = {
        "model_name": "test-model",
        "api_url": "http://localhost:11434/v1",
        "system_prompt": "exactly {num_turns} messages",
        "user_prompt_template": "topic: {topic} with {num_turns} turns",
        "min_turns": 4,
        "max_turns": 6,
        "generation_config": {"temperature": 0.7, "top_p": 0.95},
        "hf_api_token": None,
        "openai_api_token": None,
        "reasoning": None,
        "max_tokens": 4096,
    }
    config.update(overrides)
    return config


def _conversation():
    return ChatConversation(
        topic="test topic",
        messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ],
    )


def test_chat_conversation_schema():
    conv = _conversation()
    assert conv.topic == "test topic"
    assert conv.messages[0].role == "user"


def test_task_handler_uses_even_turn_count(monkeypatch):
    captured = {}

    def fake_get_completion(**kwargs):
        captured.update(kwargs)
        return _conversation()

    monkeypatch.setattr(chat, "get_completion", fake_get_completion)
    result = chat.chat_task_handler("my topic", _llm_config())

    system_prompt = captured["system_prompt"]
    assert (
        "exactly 4 messages" in system_prompt or "exactly 6 messages" in system_prompt
    )
    assert "exactly 5 messages" not in system_prompt
    assert result["topic"] == "test topic"


def test_task_handler_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(chat, "get_completion", lambda **kwargs: None)
    result = chat.chat_task_handler("my topic", _llm_config())
    assert result is None


def test_task_handler_ignores_unknown_placeholder(monkeypatch):
    # A custom prompt with a placeholder the handler doesn't know must not crash.
    monkeypatch.setattr(chat, "get_completion", lambda **kwargs: _conversation())
    result = chat.chat_task_handler(
        "my topic",
        _llm_config(
            system_prompt="literal {unknown} and {num_turns}",
            user_prompt_template="{topic} and {unknown}",
        ),
    )
    assert result is not None


def test_chat_cmd_resumes_from_existing_output(tmp_path, monkeypatch):
    topics_file = tmp_path / "topics.txt"
    topics_file.write_text("topic A\n")
    output_file = tmp_path / "out.parquet"
    output_file.touch()

    existing = [
        {
            "topic": "topic A",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hey"},
            ],
        },
        {
            "topic": "topic A",
            "messages": [
                {"role": "user", "content": "hi2"},
                {"role": "assistant", "content": "hey2"},
            ],
        },
    ]

    class FakeDataset:
        def __init__(self):
            self.rows = existing

        @classmethod
        def from_parquet(cls, path):
            return cls()

        @classmethod
        def from_json(cls, path):
            return cls()

        def to_list(self):
            return self.rows

    monkeypatch.setattr(chat, "Dataset", FakeDataset)
    monkeypatch.setattr(chat, "get_token", lambda: None)

    captured = {}

    def fake_process(
        dataset_to_process,
        workers,
        resume_idx,
        task_handler,
        llm_config,
        save_callback=None,
        save_every=50,
    ):
        captured["dataset"] = list(dataset_to_process)
        captured["resume_idx"] = resume_idx
        captured["save_callback"] = save_callback
        return [
            {
                "topic": "topic A",
                "messages": [
                    {"role": "user", "content": "new"},
                    {"role": "assistant", "content": "new2"},
                ],
            }
        ]

    monkeypatch.setattr(chat, "process_samples_with_executor", fake_process)

    saved = {}

    def fake_save(completions, output_file, push_to_hub, hf_repo_id, hf_api_token):
        saved["completions"] = completions

    monkeypatch.setattr(chat, "save_and_push_dataset", fake_save)

    runner = CliRunner()
    result = runner.invoke(
        chat.chat_cmd,
        [
            "--topics-file",
            str(topics_file),
            "--num-conversations",
            "3",
            "--output-file",
            str(output_file),
            "--model-name",
            "test-model",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    # tasks = [topic A] * 3; 2 already done -> 1 remaining task.
    assert captured["resume_idx"] == 2
    assert captured["dataset"] == ["topic A"]
    # final save merges existing (2) + newly generated (1).
    assert len(saved["completions"]) == 3


def test_task_handler_retries_transient_failure(monkeypatch):
    calls = {"n": 0}

    def flaky_get_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return None
        return _conversation()

    monkeypatch.setattr(chat, "get_completion", flaky_get_completion)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = chat.chat_task_handler("my topic", _llm_config(retries=3))
    assert result is not None
    assert calls["n"] == 3


def test_task_handler_gives_up_after_retries(monkeypatch):
    calls = {"n": 0}

    def always_none(**kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(chat, "get_completion", always_none)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = chat.chat_task_handler("my topic", _llm_config(retries=2))
    assert result is None
    assert calls["n"] == 2


def test_task_handler_retries_invalid_json(monkeypatch):
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "this is not json"
        return _conversation()

    monkeypatch.setattr(chat, "get_completion", flaky)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    result = chat.chat_task_handler("my topic", _llm_config(retries=3))
    assert result is not None
    assert calls["n"] == 2

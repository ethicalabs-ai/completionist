from completionist.dataset_view import (
    detect_format,
    render_sample_text,
    row_to_tabs,
    title,
)


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


def _tools_row():
    return {
        "topic": "weather",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Weather in SF?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 20}'},
        ],
    }


def test_detect_format_conversation():
    assert detect_format(_conversation_row()) == "conversation"


def test_detect_format_prompt_completion():
    assert detect_format(_prompt_completion_row()) == "prompt_completion"


def test_detect_format_tools_with_tools_field():
    assert detect_format(_tools_row()) == "tools"


def test_detect_format_tools_without_tools_field():
    row = _conversation_row()
    row["messages"].append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        }
    )
    assert detect_format(row) == "tools"


def test_detect_format_generic():
    assert detect_format({"foo": "bar"}) == "generic"


def test_row_to_tabs_conversation():
    tabs = row_to_tabs(_conversation_row(), "conversation")
    names = [name for name, _ in tabs]
    assert names == ["conversation", "raw"]
    segments = tabs[0][1]
    assert ("user", "Is degrowth viable?") in segments
    assert ("assistant", "Partly, but it needs a just transition.") in segments


def test_row_to_tabs_prompt_completion_with_reasoning():
    row = _prompt_completion_row()
    row["reasoning"] = "thinking..."
    tabs = row_to_tabs(row, "prompt_completion")
    assert [name for name, _ in tabs] == ["prompt", "completion", "reasoning", "raw"]


def test_row_to_tabs_prompt_completion_without_reasoning():
    tabs = row_to_tabs(_prompt_completion_row(), "prompt_completion")
    assert [name for name, _ in tabs] == ["prompt", "completion", "raw"]


def test_row_to_tabs_tools():
    tabs = row_to_tabs(_tools_row(), "tools")
    names = [name for name, _ in tabs]
    assert names == ["conversation", "tools", "raw"]

    conv_text = [text for _, text in tabs[0][1]]
    assert any(text.startswith("-> get_weather(") for text in conv_text)
    assert any("tool(call_1)" in text for text in conv_text)

    tools_text = [text for _, text in tabs[1][1]]
    assert "get_weather" in tools_text
    assert any('"city"' in text for text in tools_text)


def test_title():
    assert title(_conversation_row(), "conversation") == "climate policy"
    assert title(_prompt_completion_row(), "prompt_completion") == "What is justice?"


def test_render_sample_text():
    out = render_sample_text(_conversation_row(), "conversation", idx=0, total=1)
    assert "[1/1] climate policy" in out
    assert "== conversation ==" in out
    assert "user: Is degrowth viable?" in out
    assert "assistant: Partly, but it needs a just transition." in out
    assert "== raw ==" in out


def test_render_sample_text_unknown_total():
    out = render_sample_text(
        _prompt_completion_row(), "prompt_completion", idx=2, total=None
    )
    assert "[3/?]" in out


def test_row_to_tabs_generic():
    tabs = row_to_tabs({"foo": "bar"}, "generic")
    assert [name for name, _ in tabs] == ["raw"]

"""Dataset format detection and rendering to styled segments (reusable)."""

import json

ROLE_PREFIX = {"user": "user: ", "assistant": "assistant: ", "system": "system: "}


def detect_format(row):
    """Return 'tools', 'conversation', 'prompt_completion', or 'generic'."""
    messages = row.get("messages")
    is_messages = (
        isinstance(messages, list)
        and bool(messages)
        and all(isinstance(m, dict) and "role" in m for m in messages)
    )
    has_tools = isinstance(row.get("tools"), list) and bool(row.get("tools"))
    has_tool_calls = is_messages and any(m.get("tool_calls") for m in messages)

    if is_messages and (has_tools or has_tool_calls):
        return "tools"
    if is_messages:
        return "conversation"
    if "prompt" in row and "completion" in row:
        return "prompt_completion"
    return "generic"


def _pretty_json(value):
    return json.dumps(value, indent=2, ensure_ascii=False)


def _pretty_args(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, TypeError):
            return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _role_style(role):
    if role == "assistant":
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def _conversation_segments(row):
    """Turn a conversation row into (style, text) segments."""
    segments = []
    topic = row.get("topic")
    if topic:
        segments.append(("topic", str(topic)))
        segments.append(("plain", ""))
    for msg in row.get("messages") or []:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "tool":
            if content:
                segments.append(
                    ("tool", f"tool({msg.get('tool_call_id', '?')}): {content}")
                )
        else:
            if content:
                segments.append((_role_style(role), content))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                name = fn.get("name", "?")
                args = fn.get("arguments")
                if args:
                    segments.append(("tool", f"-> {name}({_pretty_args(args)})"))
                else:
                    segments.append(("tool", f"-> {name}()"))
        segments.append(("plain", ""))
    return segments


def _tools_segments(row):
    """Turn the 'tools' field into (style, text) segments."""
    segments = []
    for tool in row.get("tools") or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        segments.append(("topic", fn.get("name", "?")))
        desc = fn.get("description")
        if desc:
            segments.append(("plain", desc))
        params = fn.get("parameters")
        if params is not None:
            segments.append(("raw", _pretty_json(params)))
        segments.append(("plain", ""))
    return segments


def row_to_tabs(row, fmt):
    """Return ``[(tab_name, [(style, text), ...]), ...]`` for a single row."""
    tabs = []
    if fmt == "conversation":
        tabs.append(("conversation", _conversation_segments(row)))
    elif fmt == "prompt_completion":
        tabs.append(("prompt", [("user", row.get("prompt") or "")]))
        tabs.append(("completion", [("assistant", row.get("completion") or "")]))
        if "reasoning" in row:
            tabs.append(("reasoning", [("topic", row.get("reasoning") or "")]))
    elif fmt == "tools":
        tabs.append(("conversation", _conversation_segments(row)))
        tabs.append(("tools", _tools_segments(row)))
    tabs.append(("raw", [("raw", _pretty_json(row))]))
    return tabs


def title(row, fmt):
    """Short one-line label for a row."""
    if fmt in ("conversation", "tools"):
        if row.get("topic"):
            return str(row["topic"])
        messages = row.get("messages") or []
        if messages:
            return str(messages[0].get("content") or "")[:60]
        return ""
    if fmt == "prompt_completion":
        return str(row.get("prompt") or "")[:60]
    return ""


def render_sample_text(row, fmt, idx, total):
    """Render one sample as plain text (used when stdout is not a TTY)."""
    header = f"[{idx + 1}/{total if total is not None else '?'}] {title(row, fmt)}"
    parts = [header]
    for tab_name, segments in row_to_tabs(row, fmt):
        parts.append("")
        parts.append(f"== {tab_name} ==")
        for style, text in segments:
            prefix = ROLE_PREFIX.get(style, "")
            parts.extend(prefix + ln for ln in text.splitlines() or [""])
    return "\n".join(parts)

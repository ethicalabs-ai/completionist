import json
import os
import sys
import traceback

import click
from huggingface_hub import get_token
from pydantic import BaseModel, Field

from completionist.llm_api import get_completion
from completionist.utils import read_file_content


class TopicList(BaseModel):
    topics: list[str] = Field(..., description="List of conversation topics")


DEFAULT_SYSTEM_PROMPT = (
    "You are a creative topic curator. "
    "Generate {batch_size} diverse conversation topics across these categories: "
    "{categories}. "
    "Each topic should be a short phrase that could seed an engaging multi-turn conversation. "
    "Avoid generic topics — make them specific and thought-provoking. "
    "Return the result as a JSON object matching the schema."
)


def _load_existing(path: str, ext: str) -> set[str]:
    """Load existing topics from a .txt or .jsonl file."""
    topics: set[str] = set()
    if not os.path.exists(path):
        return topics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ext == ".jsonl":
                try:
                    data = json.loads(line)
                    topic = data.get("topic", "")
                except json.JSONDecodeError:
                    continue
            else:
                topic = line
            if topic:
                topics.add(topic)
    return topics


def _write_topics_file(path: str, ext: str, topics: list[str]) -> None:
    """Write topics to a .txt (one per line) or .jsonl file."""
    if ext == ".jsonl":
        with open(path, "w") as f:
            for topic in topics:
                f.write(json.dumps({"topic": topic}) + "\n")
    else:
        with open(path, "w") as f:
            for topic in topics:
                f.write(topic + "\n")


def _generate_batch(
    model_name: str,
    api_url: str,
    system_prompt: str,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: str | None,
    hf_api_token: str | None,
    openai_api_token: str | None,
) -> list[str]:
    """Make a single API call and return generated topics."""
    result = get_completion(
        prompt=f"Generate {batch_size} topics.",
        model_name=model_name,
        api_url=api_url,
        system_prompt=system_prompt,
        hf_api_token=hf_api_token,
        openai_api_token=openai_api_token,
        pydantic_schema=TopicList,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        reasoning=reasoning,
    )

    if result is None:
        return []
    if isinstance(result, TopicList):
        return result.topics
    # outlines returned a raw string/dict — parse it as fallback
    raw = json.loads(result) if isinstance(result, str) else result
    return TopicList(**raw).topics


@click.command("topics")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, resolve_path=True),
    required=True,
    help="Output file path. Extension determines format: .txt (one per line) or .jsonl.",
)
@click.option(
    "--num-topics",
    "-n",
    type=int,
    required=True,
    help="Total number of unique topics to generate.",
)
@click.option(
    "--model-name",
    required=True,
    help="The name of the model to use for generation.",
)
@click.option(
    "--categories",
    required=True,
    help="Comma-separated list of topic categories to guide generation.",
)
@click.option(
    "--api-url",
    default="http://localhost:11434/v1",
    help="The API endpoint URL for the LLM.",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=50,
    help="Topics to request per API call. Multiple batches are deduplicated automatically.",
    show_default=True,
)
@click.option(
    "--temperature",
    type=float,
    default=0.9,
    help="Sampling temperature. Higher values produce more diverse topics.",
    show_default=True,
)
@click.option(
    "--top-p",
    type=float,
    default=0.95,
    help="Nucleus sampling (top-p).",
    show_default=True,
)
@click.option(
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum tokens per API call. Increase for reasoning models.",
    show_default=True,
)
@click.option(
    "--system-prompt",
    default=None,
    help="Override the built-in system prompt. Use {batch_size} and {categories} placeholders.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Path to a file overriding the built-in system prompt.",
)
@click.option(
    "--reasoning",
    type=str,
    default=None,
    help="llama.cpp reasoning mode: 'on', 'off', or 'auto'. Leave unset for model default.",
)
@click.option(
    "--max-batches",
    type=int,
    default=20,
    help="Maximum number of API calls before stopping, even if --num-topics is not reached.",
    show_default=True,
)
@click.option(
    "--append",
    is_flag=True,
    help="Append to existing output file. --num-topics specifies how many NEW topics to add.",
)
def topics_cmd(
    output,
    num_topics,
    model_name,
    categories,
    api_url,
    batch_size,
    temperature,
    top_p,
    max_tokens,
    system_prompt,
    system_prompt_file,
    reasoning,
    max_batches,
    append,
):
    """Generate diverse conversation topics via LLM.

    Output format is determined by file extension:
    \b
      .txt   — one topic per line
      .jsonl — {"topic": "..."} per line
    """
    if num_topics < 1:
        print("Error: --num-topics must be at least 1.")
        sys.exit(1)

    if batch_size < 1:
        print("Error: --batch-size must be at least 1.")
        sys.exit(1)

    # Resolve system prompt: file > inline > built-in default
    system_prompt_content = read_file_content(system_prompt_file) or system_prompt
    if not system_prompt_content:
        system_prompt_content = DEFAULT_SYSTEM_PROMPT

    system_prompt_content = system_prompt_content.replace(
        "{batch_size}", str(batch_size)
    )
    system_prompt_content = system_prompt_content.replace("{categories}", categories)

    # Determine output format from extension
    ext = os.path.splitext(output)[1].lower()
    if ext not in (".txt", ".jsonl"):
        print(f"Error: Unsupported output extension '{ext}'. Use .txt or .jsonl.")
        sys.exit(1)

    hf_api_token = get_token()
    openai_api_token = os.environ.get("OPENAI_API_TOKEN", None)

    # Seed with existing topics when appending
    existing_topics: set[str] = set()
    all_topics: set[str] = set()
    if append and os.path.exists(output):
        existing_topics = _load_existing(output, ext)
        all_topics.update(existing_topics)
        print(
            f"Appending to {output} ({len(existing_topics)} existing, adding up to {num_topics} new)"
        )

    batches_run = 0
    target = len(existing_topics) + num_topics  # existing + desired new

    def current_topics_list() -> list[str]:
        if append:
            new_topics = list(all_topics - existing_topics)[:num_topics]
            return list(existing_topics) + new_topics
        return list(all_topics)[:num_topics]

    print(
        f"Generating {num_topics} topics across categories: {categories}\n"
        f"Model: {model_name} | API: {api_url} | Batch size: {batch_size}\n"
    )

    while len(all_topics) < target and batches_run < max_batches:
        batches_run += 1

        try:
            batch = _generate_batch(
                model_name=model_name,
                api_url=api_url,
                system_prompt=system_prompt_content,
                batch_size=batch_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
                hf_api_token=hf_api_token,
                openai_api_token=openai_api_token,
            )
        except Exception:
            print(f"Batch {batches_run} failed: {traceback.format_exc()}")
            continue

        new = set(batch) - all_topics
        all_topics.update(batch)

        print(
            f"  Batch {batches_run}: {len(batch)} generated, "
            f"{len(new)} new unique, {len(all_topics)} total"
        )
        # Persist progress so an interrupted run can resume via --append.
        _write_topics_file(output, ext, current_topics_list())

    if len(all_topics) < target:
        shortfall = target - len(all_topics)
        print(
            f"\nWarning: reached --max-batches ({max_batches}) with "
            f"{len(all_topics)} unique topics — {shortfall} short of {target}."
        )

    topics_list = current_topics_list()
    _write_topics_file(output, ext, topics_list)

    print(f"\nSaved {len(topics_list)} unique topics to {output}")

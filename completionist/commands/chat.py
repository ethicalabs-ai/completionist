import os
import sys
import json
import random
import traceback
import time

import click
from pydantic import BaseModel, Field
from huggingface_hub import get_token
from datasets import Dataset

from completionist.processing import process_samples_with_executor
from completionist.dataset_io import save_and_push_dataset
from completionist.utils import read_file_content
from completionist.llm_api import get_completion


# --- Built-in schema ---


class ChatMessage(BaseModel):
    role: str = Field(..., description="Speaker role: 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatConversation(BaseModel):
    topic: str = Field(..., description="The conversation topic")
    messages: list[ChatMessage] = Field(..., description="Ordered list of messages")


# --- Built-in prompts ---

DEFAULT_SYSTEM_PROMPT = (
    "You are a creative conversational designer. "
    "Generate a realistic, engaging multi-turn conversation on the given topic. "
    "The conversation must have exactly {num_turns} messages, alternating between "
    "'user' and 'assistant' roles, starting with 'user'. "
    "Make it feel like two interesting people talking, not a Q&A or a lecture.\n"
    "- The assistant should occasionally disagree, push back, ask a question, or admit "
    "they don't know. Avoid wrapping up neatly — real conversations are messy.\n"
    "- Vary register: humor, frustration, curiosity, doubt. Not every turn should be "
    "polished and academic.\n"
    "- The user should have their own perspective, not just ask setup questions. "
    "Let them challenge the assistant, change their mind, or go off on a tangent.\n"
    "- Return the result as a JSON object matching the schema."
)

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Generate a multi-turn conversation with exactly {num_turns} messages "
    "(alternating user/assistant, starting with user).\n"
    "Return a JSON object with 'topic' and 'messages' fields."
)


# --- Task handler ---


def chat_task_handler(topic: str, llm_config: dict):
    """Task handler for generating a single multi-turn conversation for a topic."""
    num_turns = random.randrange(
        llm_config["min_turns"], llm_config["max_turns"] + 1, 2
    )

    user_prompt = (
        llm_config["user_prompt_template"]
        .replace("{topic}", topic)
        .replace("{num_turns}", str(num_turns))
    )
    system_prompt = llm_config["system_prompt"].replace("{num_turns}", str(num_turns))

    max_retries = llm_config.get("retries", 3)
    for attempt in range(1, max_retries + 1):
        try:
            result = get_completion(
                prompt=user_prompt,
                model_name=llm_config["model_name"],
                api_url=llm_config["api_url"],
                system_prompt=system_prompt,
                hf_api_token=llm_config["hf_api_token"],
                openai_api_token=llm_config["openai_api_token"],
                pydantic_schema=ChatConversation,
                temperature=llm_config["generation_config"]["temperature"],
                top_p=llm_config["generation_config"]["top_p"],
                max_tokens=llm_config.get("max_tokens", 2048),
                reasoning=llm_config.get("reasoning"),
            )

            if result is None:
                raise RuntimeError("completion returned no result")
            if isinstance(result, ChatConversation):
                return result.model_dump()
            # outlines returned a raw string — attempt JSON parse as fallback
            return ChatConversation(**json.loads(result)).model_dump()

        except Exception as exc:
            if attempt >= max_retries:
                print(
                    f"\nWarning: Failed to generate conversation for topic '{topic}' "
                    f"after {max_retries} attempts. Reason: {traceback.format_exc()}"
                )
                return None
            delay = 2 ** (attempt - 1)
            print(
                f"  Retrying topic '{topic}' (attempt {attempt}/{max_retries} failed): "
                f"{exc} — waiting {delay}s"
            )
            time.sleep(delay)

    return None


# --- CLI command ---


@click.command("chat")
@click.option(
    "--topics-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a text file with one topic per line to seed conversation generation.",
)
@click.option(
    "--num-conversations",
    type=int,
    required=True,
    help="Number of conversations to generate per topic.",
)
@click.option(
    "--min-turns",
    type=int,
    default=4,
    help="Minimum number of messages per conversation. Must be even.",
    show_default=True,
)
@click.option(
    "--max-turns",
    type=int,
    default=6,
    help="Maximum number of messages per conversation. Must be even.",
    show_default=True,
)
@click.option(
    "--output-file",
    required=True,
    help="The path to save the generated dataset (e.g., output.parquet).",
)
@click.option(
    "--model-name", required=True, help="The name of the model to use for generation."
)
@click.option(
    "--api-url",
    default="http://localhost:11434/v1",
    help="(Optional) The API endpoint URL for the LLM. Defaults to Ollama's base URL.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="(Optional) Override the built-in system prompt. The string '{num_turns}' will "
    "be replaced with the actual number.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="(Optional) Path to a file overriding the built-in system prompt.",
)
@click.option(
    "--user-prompt-template",
    default=None,
    help="(Optional) Override the built-in user prompt template. The strings "
    "'{topic}' and '{num_turns}' will be replaced.",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="(Optional) Number of concurrent requests. Defaults to 4.",
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="(Optional) Push the generated dataset to the Hugging Face Hub.",
)
@click.option(
    "--hf-repo-id",
    default=None,
    help="The Hugging Face repository ID to push the dataset to. Required if --push-to-hub is used.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for generation.",
)
@click.option(
    "--top-p", type=float, default=0.95, help="Nucleus sampling (top-p) for generation."
)
@click.option(
    "--reasoning",
    type=str,
    default=None,
    help="(Optional) llama.cpp reasoning mode: 'on', 'off', or 'auto'. Leave unset for model default.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum tokens per conversation. Increase for longer chats.",
    show_default=True,
)
@click.option(
    "--retries",
    type=int,
    default=3,
    help="Maximum attempts per conversation on failure.",
    show_default=True,
)
def chat_cmd(
    topics_file,
    num_conversations,
    min_turns,
    max_turns,
    output_file,
    model_name,
    api_url,
    system_prompt,
    system_prompt_file,
    user_prompt_template,
    workers,
    push_to_hub,
    hf_repo_id,
    temperature,
    top_p,
    reasoning,
    max_tokens,
    retries,
):
    """
    Generate multi-turn conversation datasets from a list of topics.
    Uses a built-in schema (ChatConversation) — no --schema needed.
    """
    hf_api_token = get_token()
    openai_api_token = os.environ.get("OPENAI_API_TOKEN", None)

    if push_to_hub and not hf_repo_id:
        print("Error: --hf-repo-id is required when --push-to-hub is used.")
        sys.exit(1)

    if min_turns % 2 != 0 or max_turns % 2 != 0:
        print("Error: --min-turns and --max-turns must be even numbers.")
        sys.exit(1)

    if min_turns > max_turns:
        print("Error: --min-turns cannot be greater than --max-turns.")
        sys.exit(1)

    if retries < 1:
        print("Error: --retries must be at least 1.")
        sys.exit(1)

    # Resolve prompts: file > inline > built-in default
    system_prompt_content = read_file_content(system_prompt_file) or system_prompt
    if not system_prompt_content:
        system_prompt_content = DEFAULT_SYSTEM_PROMPT

    user_prompt_template_content = user_prompt_template
    if not user_prompt_template_content:
        user_prompt_template_content = DEFAULT_USER_PROMPT_TEMPLATE

    # Load topics
    topics = [
        line for line in read_file_content(topics_file).splitlines() if line.strip()
    ]
    if not topics:
        print(
            f"Error: Topics file '{topics_file}' is empty or contains no valid lines."
        )
        sys.exit(1)

    # Prepare configuration
    llm_config = {
        "model_name": model_name,
        "api_url": api_url,
        "system_prompt": system_prompt_content,
        "user_prompt_template": user_prompt_template_content,
        "min_turns": min_turns,
        "max_turns": max_turns,
        "generation_config": {"temperature": temperature, "top_p": top_p},
        "hf_api_token": hf_api_token,
        "openai_api_token": openai_api_token,
        "reasoning": reasoning,
        "max_tokens": max_tokens,
        "retries": retries,
    }

    # Build task list: num_conversations per topic
    tasks = []
    for topic in topics:
        tasks.extend([topic] * num_conversations)

    # Resume from an existing output file if present.
    existing = []
    resume_idx = 0
    if os.path.exists(output_file):
        try:
            if os.path.splitext(output_file)[1].lower() == ".jsonl":
                existing = Dataset.from_json(output_file).to_list()
            else:
                existing = Dataset.from_parquet(output_file).to_list()
            resume_idx = len(existing)
            print(
                f"Resuming from {output_file}: {resume_idx} conversations already done."
            )
        except Exception as e:
            print(f"Could not load {output_file} ({e}). Starting fresh.")

    remaining = tasks[resume_idx:]

    if not remaining:
        print(f"All {len(tasks)} conversations already present in {output_file}.")
        return

    print(
        f"Generating {len(remaining)} conversations ({num_conversations} per topic) "
        f"across {len(topics)} topics with {workers} workers..."
    )

    def save_progress(completions):
        # Checkpoint partial results locally so interrupted runs keep their output.
        save_and_push_dataset(
            completions=existing + completions,
            output_file=output_file,
            push_to_hub=False,
            hf_repo_id=None,
            hf_api_token=None,
        )

    generated = process_samples_with_executor(
        dataset_to_process=remaining,
        workers=workers,
        resume_idx=resume_idx,
        task_handler=chat_task_handler,
        llm_config=llm_config,
        save_callback=save_progress,
        save_every=25,
    )

    save_and_push_dataset(
        completions=existing + generated,
        output_file=output_file,
        push_to_hub=push_to_hub,
        hf_repo_id=hf_repo_id,
        hf_api_token=hf_api_token,
    )

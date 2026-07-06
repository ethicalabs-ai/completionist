import os
import sys
import random
import traceback

import click
from pydantic import BaseModel, Field
from huggingface_hub import get_token

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
    "Vary the tone and depth — some turns short and casual, others longer and thoughtful. "
    "Return the result as a JSON object matching the schema."
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
    num_turns = random.randint(llm_config["min_turns"], llm_config["max_turns"])

    user_prompt = llm_config["user_prompt_template"].format(
        topic=topic, num_turns=num_turns
    )
    system_prompt = llm_config["system_prompt"].format(num_turns=num_turns)

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
        )

        if result is None:
            return None
        if isinstance(result, ChatConversation):
            return result.model_dump()
        # outlines returned a raw string — attempt JSON parse as fallback
        import json
        raw = json.loads(result)
        return ChatConversation(**raw).model_dump()

    except Exception:
        print(
            f"\nWarning: Failed to generate conversation for topic '{topic}'. "
            f"Reason: {traceback.format_exc()}"
        )
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
    default=3,
    help="Minimum number of messages per conversation.",
    show_default=True,
)
@click.option(
    "--max-turns",
    type=int,
    default=6,
    help="Maximum number of messages per conversation.",
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

    if min_turns > max_turns:
        print("Error: --min-turns cannot be greater than --max-turns.")
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
    }

    # Build task list: num_conversations per topic
    tasks = []
    for topic in topics:
        tasks.extend([topic] * num_conversations)

    print(
        f"Generating {len(tasks)} conversations ({num_conversations} per topic) "
        f"across {len(topics)} topics with {workers} workers..."
    )

    generated = process_samples_with_executor(
        dataset_to_process=tasks,
        workers=workers,
        resume_idx=0,
        task_handler=chat_task_handler,
        llm_config=llm_config,
    )

    save_and_push_dataset(
        completions=generated,
        output_file=output_file,
        push_to_hub=push_to_hub,
        hf_repo_id=hf_repo_id,
        hf_api_token=hf_api_token,
    )

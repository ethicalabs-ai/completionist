import os
import sys
import click
import importlib
import traceback

from pydantic import BaseModel
from huggingface_hub import get_token

from completionist.processing import process_samples_with_executor
from completionist.dataset_io import save_and_push_dataset
from completionist.utils import read_file_content
from completionist.llm_api import get_completion


def load_schema_from_import_path(import_path: str) -> BaseModel:
    """Dynamically loads a Pydantic BaseModel class from a Python import path."""
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        schema_class = getattr(module, class_name)
        if not issubclass(schema_class, BaseModel):
            raise TypeError(f"Class '{class_name}' is not a Pydantic BaseModel.")
        return schema_class
    except (ImportError, AttributeError):
        print(f"Error: Could not find or import schema '{import_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading schema class: {e}")
        sys.exit(1)


def build_task_handler(topic: str, llm_config: dict):
    """
    Task handler for generating a single structured data sample for a given topic.
    """
    user_prompt_template = llm_config["user_prompt_template"]

    try:
        user_prompt = user_prompt_template.format(topic=topic)

        # Call the centralized get_completion function with the schema
        result = get_completion(
            prompt=user_prompt,
            model_name=llm_config["model_name"],
            api_url=llm_config["api_url"],
            system_prompt=llm_config["system_prompt"],
            hf_api_token=llm_config["hf_api_token"],
            openai_api_token=llm_config["openai_api_token"],
            pydantic_schema=llm_config["pydantic_schema"],
            temperature=llm_config["generation_config"]["temperature"],
            top_p=llm_config["generation_config"]["top_p"],
        )

        if result:
            return result.model_dump()
        return None

    except Exception:
        print(
            f"\nWarning: Failed to generate a valid sample for topic '{topic}'. Reason: {traceback.format_exc()}"
        )
        return None


@click.command("build")
@click.option(
    "--schema",
    default="completionist.commands.default_schema.DefaultSchema",
    help="Full Python import path to the Pydantic schema class to use for generation.",
    show_default=True,
)
@click.option(
    "--topics-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a text file with one topic per line to seed generation.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a file containing the system prompt.",
)
@click.option(
    "--user-prompt-template-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to a prompt template file. Must contain a '{topic}' placeholder.",
)
@click.option(
    "--num-samples",
    type=int,
    required=True,
    help="The total number of samples to generate for each topic.",
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
    "--workers",
    type=int,
    default=4,
    help="(Optional) Number of concurrent requests to make to the API. Defaults to 4.",
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
def build_cmd(
    schema,
    topics_file,
    system_prompt_file,
    user_prompt_template_file,
    num_samples,
    output_file,
    model_name,
    api_url,
    workers,
    push_to_hub,
    hf_repo_id,
    temperature,
    top_p,
):
    """
    Generate a structured dataset from a list of topics using a Pydantic schema.
    """
    hf_api_token = get_token()
    openai_api_token = os.environ.get("OPENAI_API_TOKEN", None)

    if push_to_hub and not hf_repo_id:
        print("Error: --hf-repo-id is required when --push-to-hub is used.")
        sys.exit(1)

    # Load all external assets
    pydantic_schema = load_schema_from_import_path(schema)
    topics = [
        line for line in read_file_content(topics_file).splitlines() if line.strip()
    ]
    system_prompt = read_file_content(system_prompt_file)
    user_prompt_template = read_file_content(user_prompt_template_file)

    if not topics:
        print(
            f"Error: Topics file '{topics_file}' is empty or contains no valid lines."
        )
        sys.exit(1)
    if "{topic}" not in user_prompt_template:
        print(
            "Error: The user prompt template must contain a '{topic}' placeholder.\n"
            f"File checked: {user_prompt_template_file}\n"
            f"Content starts with: '{user_prompt_template[:200]}...'"
        )
        sys.exit(1)

    # Prepare configuration for the task handler
    llm_config = {
        "model_name": model_name,
        "api_url": api_url,
        "pydantic_schema": pydantic_schema,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
        "generation_config": {"temperature": temperature, "top_p": top_p},
        "hf_api_token": hf_api_token,
        "openai_api_token": openai_api_token,
    }

    # Create a list of all tasks to run (num_samples for each topic)
    tasks_to_run = []
    for topic in topics:
        tasks_to_run.extend([topic] * num_samples)

    total_tasks = len(tasks_to_run)
    print(
        f"Starting structured data generation for {total_tasks} samples ({num_samples} per topic) with {workers} workers..."
    )

    generated_samples = process_samples_with_executor(
        dataset_to_process=tasks_to_run,
        workers=workers,
        resume_idx=0,  # No resume functionality for build command
        task_handler=build_task_handler,
        llm_config=llm_config,
    )

    save_and_push_dataset(
        completions=generated_samples,
        output_file=output_file,
        push_to_hub=push_to_hub,
        hf_repo_id=hf_repo_id,
        hf_api_token=hf_api_token,
    )

import click
import re
from huggingface_hub import get_token

from completionist.processing import process_samples_with_executor
from completionist.dataset_io import load_and_prepare_dataset, save_and_push_dataset
from completionist.llm_api import get_completion
from completionist.utils import read_file_content, handle_error


def complete_task_handler(sample, llm_config):
    """
    Helper function to generate a completion for a single sample for the 'complete' command.
    """
    prompt = ""
    prompt_template = llm_config.get("prompt_template")

    if prompt_template:
        try:
            prompt = prompt_template.format(**sample)
        except KeyError as e:
            print(
                f"\nError: The placeholder {{{e.args[0]}}} in your prompt template was not found "
                f"as a column in the dataset. Available columns: {list(sample.keys())}"
            )
            return None
    else:
        prompt = sample[llm_config["prompt_input_field"]]

    completion = get_completion(
        prompt=prompt,
        model_name=llm_config["model_name"],
        api_url=llm_config["api_url"],
        system_prompt=llm_config["system_prompt"],
        hf_api_token=llm_config["hf_api_token"],
        max_tokens=llm_config["max_tokens"],
        temperature=llm_config["temperature"],
        top_p=llm_config["top_p"],
    )

    if completion:
        reasoning_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        cleaned_completion = re.sub(
            r"<think>.*?</think>", "", completion, flags=re.DOTALL
        ).strip()
        return {
            llm_config["prompt_output_field"]: prompt,
            llm_config["completion_output_field"]: cleaned_completion,
            "reasoning": reasoning,
        }
    return None


@click.command()
@click.option(
    "--dataset-name", required=True, help="The name of the Hugging Face dataset to use."
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
    help="(Optional) The API endpoint URL for the LLM. Defaults to Ollama's OpenAI-compatible endpoint.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="(Optional) A system prompt to prepend to each user prompt. Cannot be used with --system-prompt-file.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="(Optional) Path to a file containing the system prompt. Cannot be used with --system-prompt.",
)
@click.option(
    "--prompt-template-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="(Optional) Path to a text file containing the prompt template. If provided, it formats the prompt using dataset columns as placeholders (e.g. {column_name}).",
)
@click.option(
    "--max-tokens",
    type=int,
    default=2048,
    help="(Optional) The maximum number of tokens to generate per completion.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="(Optional) Limit the number of samples to process.",
)
@click.option(
    "--shuffle", is_flag=True, help="(Optional) Shuffle the dataset before processing."
)
@click.option(
    "--push-to-hub",
    is_flag=True,
    help="(Optional) Push the generated dataset to the Hugging Face Hub.",
)
@click.option(
    "--hf-repo-id",
    default=None,
    help="The Hugging Face repository ID to push the dataset to (e.g., 'your-user/your-dataset'). Required if --push_to_hub is used.",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="(Optional) Number of concurrent requests to make to the API. Defaults to 4.",
)
@click.option(
    "--prompt-input-field",
    required=True,
    help="The name of the field in the input dataset to use as the prompt. Also used for validation.",
)
@click.option(
    "--prompt-output-field",
    default="prompt",
    help="The name of the field to store the original prompt in the output dataset. Defaults to 'prompt'.",
)
@click.option(
    "--completion-output-field",
    default="completion",
    help="The name of the field to store the generated completion in the output dataset. Defaults to 'completion'.",
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
def complete_cmd(
    dataset_name,
    output_file,
    model_name,
    api_url,
    system_prompt,
    system_prompt_file,
    prompt_template_file,
    max_tokens,
    limit,
    shuffle,
    push_to_hub,
    hf_repo_id,
    workers,
    prompt_input_field,
    prompt_output_field,
    completion_output_field,
    temperature,
    top_p,
):
    """
    Generate text completions for a dataset using an LLM.
    """
    if system_prompt and system_prompt_file:
        raise click.UsageError(
            "Error: --system-prompt and --system-prompt-file are mutually exclusive."
        )

    hf_api_token = get_token()

    if push_to_hub and not hf_repo_id:
        handle_error("Error: --hf-repo-id is required when --push-to-hub is used.")

    system_prompt_content = (
        read_file_content(system_prompt_file) if system_prompt_file else system_prompt
    )
    prompt_template = read_file_content(prompt_template_file)

    dataset_to_process, resume_idx, total_samples_in_dataset, existing_completions = (
        load_and_prepare_dataset(
            dataset_name=dataset_name,
            output_file=output_file,
            prompt_input_field=prompt_input_field,
            shuffle=shuffle,
            limit=limit,
        )
    )

    llm_config = {
        "model_name": model_name,
        "api_url": api_url,
        "system_prompt": system_prompt_content,
        "prompt_template": prompt_template,
        "hf_api_token": hf_api_token,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_input_field": prompt_input_field,
        "prompt_output_field": prompt_output_field,
        "completion_output_field": completion_output_field,
    }

    print(
        f"Starting completion generation for {len(dataset_to_process)} samples (out of {total_samples_in_dataset}) with {workers} workers..."
    )

    new_completions = process_samples_with_executor(
        dataset_to_process=dataset_to_process,
        workers=workers,
        resume_idx=resume_idx,
        task_handler=complete_task_handler,
        llm_config=llm_config,
    )
    if len(new_completions) > 0:
        all_completions = existing_completions + new_completions
        save_and_push_dataset(
            all_completions, output_file, push_to_hub, hf_repo_id, hf_api_token
        )

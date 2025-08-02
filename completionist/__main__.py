import sys
import click
from huggingface_hub import get_token

from .processing import process_samples_with_executor
from .dataset_io import load_and_prepare_dataset, save_and_push_dataset


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
    default="http://localhost:11434/v1/chat/completions",
    help="(Optional) The API endpoint URL for the LLM. Defaults to Ollama's OpenAI-compatible endpoint.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="(Optional) A system prompt to prepend to each user prompt.",
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
    help="The name of the field in the input dataset to use as the prompt.",
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
def main(
    dataset_name,
    output_file,
    model_name,
    api_url,
    system_prompt,
    max_tokens,
    limit,
    shuffle,
    push_to_hub,
    hf_repo_id,
    workers,
    prompt_input_field,
    prompt_output_field,
    completion_output_field,
):
    """
    Generate text completions for a dataset using an LLM.
    """
    hf_api_token = get_token()

    if push_to_hub and not hf_repo_id:
        print("Error: --hf-repo-id is required when --push-to-hub is used.")
        sys.exit(1)

    dataset_to_process, resume_idx, total_samples_in_dataset = load_and_prepare_dataset(
        dataset_name,
        output_file,
        prompt_input_field,
        shuffle,
        limit,
        completion_output_field,
    )

    print(
        f"Starting completion generation for {len(dataset_to_process)} samples (out of {total_samples_in_dataset}) with {workers} workers..."
    )

    completions = process_samples_with_executor(
        dataset_to_process=dataset_to_process,
        workers=workers,
        resume_idx=resume_idx,
        total_samples_in_dataset=total_samples_in_dataset,
        prompt_input_field=prompt_input_field,
        prompt_output_field=prompt_output_field,
        completion_output_field=completion_output_field,
        model_name=model_name,
        api_url=api_url,
        system_prompt=system_prompt,
        hf_api_token=hf_api_token,
        max_tokens=max_tokens,
    )

    save_and_push_dataset(
        completions, output_file, push_to_hub, hf_repo_id, hf_api_token
    )


if __name__ == "__main__":
    main()

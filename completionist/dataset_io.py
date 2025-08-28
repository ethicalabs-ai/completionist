import os
import sys
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

from completionist.utils import handle_error


def load_and_prepare_dataset(
    dataset_name,
    output_file,
    prompt_input_field,
    shuffle,
    limit,
    split="train",
):
    """
    Loads, prepares, and handles resume logic for the dataset.
    Returns the dataset to process, the resume index, and the total dataset size.
    """
    try:
        if os.path.isfile(dataset_name):
            if dataset_name.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=dataset_name)
            elif dataset_name.endswith(".txt"):
                dataset = load_dataset("text", data_files=dataset_name)
            else:
                handle_error(
                    f"Error: Unsupported file format for '{dataset_name}'. Please use .jsonl or .txt."
                )
        else:
            dataset = load_dataset(dataset_name)

        if split not in dataset or (
            prompt_input_field and prompt_input_field not in dataset[split].features
        ):
            handle_error(
                f"Error: The dataset must have a '{split}' split and a '{prompt_input_field}' feature."
            )
        dataset = dataset[split]
    except Exception as e:
        handle_error(f"Error loading dataset: {e}")

    if shuffle:
        dataset = dataset.shuffle(seed=42)

    if limit:
        dataset = dataset.select(range(limit))

    completions = []
    resume_idx = 0
    if os.path.exists(output_file) and not shuffle:
        print(f"Resuming from existing file: {output_file}")
        try:
            existing_dataset = Dataset.from_parquet(output_file)
            completions = existing_dataset.to_list()
            resume_idx = len(completions)
            print(
                f"Found {len(completions)} existing completions. Resuming from index {resume_idx}."
            )
        except Exception as e:
            print(f"Could not load existing Parquet file: {e}. Starting from scratch.")

    dataset_to_process = dataset.select(range(resume_idx, len(dataset)))
    total_samples_in_dataset = len(dataset)

    return dataset_to_process, resume_idx, total_samples_in_dataset, completions


def save_and_push_dataset(
    completions, output_file, push_to_hub, hf_repo_id, hf_api_token
):
    """
    Saves the generated completions locally and pushes them to the Hugging Face Hub if requested.
    """
    new_dataset = Dataset.from_list(completions)
    try:
        new_dataset.to_parquet(output_file)
        print(f"Generated dataset saved locally to {output_file}")
    except Exception as e:
        handle_error(f"Error saving dataset locally: {e}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub as '{hf_repo_id}'...")
        try:
            api = HfApi(token=hf_api_token)
            try:
                api.whoami()
            except Exception:
                print(
                    "You must be logged in or have the HUGGING_FACE_HUB_TOKEN environment variable set to push a dataset."
                )
                print(
                    "Please run `huggingface-cli login` or set the environment variable and try again."
                )
                sys.exit(1)
            new_dataset.push_to_hub(hf_repo_id)
            print("Successfully pushed dataset to the Hugging Face Hub!")
        except Exception as e:
            handle_error(f"Error pushing dataset to the Hub: {e}")

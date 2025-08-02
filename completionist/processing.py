import concurrent.futures
import re
from .llm_api import get_completion


def generate_completion_for_sample(
    sample,
    prompt_input_field,
    prompt_output_field,
    completion_output_field,
    model_name,
    api_url,
    system_prompt,
    hf_api_token,
    max_tokens,
):
    """
    Helper function to generate a completion for a single sample.
    Used for concurrent processing.
    """
    prompt = sample[prompt_input_field]
    completion = get_completion(
        prompt, model_name, api_url, system_prompt, hf_api_token, max_tokens
    )

    if completion:
        reasoning_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        cleaned_completion = re.sub(
            r"<think>.*?</think>", "", completion, flags=re.DOTALL
        ).strip()
        return {
            prompt_output_field: prompt,
            completion_output_field: cleaned_completion,
            "reasoning": reasoning,
        }

    return None


def process_samples_with_executor(
    dataset_to_process,
    workers,
    resume_idx,
    total_samples_in_dataset,
    prompt_input_field,
    prompt_output_field,
    completion_output_field,
    model_name,
    api_url,
    system_prompt,
    hf_api_token,
    max_tokens,
):
    """
    Manages the concurrent execution of tasks using a ThreadPoolExecutor.
    """
    completions = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            for sample in dataset_to_process:
                future = executor.submit(
                    generate_completion_for_sample,
                    sample,
                    prompt_input_field,
                    prompt_output_field,
                    completion_output_field,
                    model_name,
                    api_url,
                    system_prompt,
                    hf_api_token,
                    max_tokens,
                )
                futures.append(future)

            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                if result:
                    completions.append(result)
                current_idx = resume_idx + i + 1
                print(
                    f"Completed {current_idx}/{total_samples_in_dataset} samples.",
                    end="\r",
                )
            print("\nAll completions processed.")
        except KeyboardInterrupt:
            print(
                "\nProcess interrupted. Attempting to shut down workers and save progress..."
            )
            executor.shutdown(wait=False, cancel_futures=True)
            print("Executor shutdown initiated.")
        finally:
            return completions

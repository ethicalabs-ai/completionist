import hashlib
import os
import click
from huggingface_hub import get_token

from completionist.processing import process_samples_with_executor
from completionist.dataset_io import load_and_prepare_dataset, save_and_push_dataset
from completionist.llm_api import get_completion
from completionist.utils import read_file_content, handle_error


def _get_cache_client(cache_url):
    """Returns a Redis client if cache_url is set, otherwise None."""
    if not cache_url:
        return None
    try:
        import redis
    except ImportError:
        handle_error(
            "Redis cache requested but 'redis' package is not installed. "
            "Run: pip install redis"
        )
    return redis.from_url(cache_url)


def _cache_key(source_text, source_lang, target_lang):
    """Deterministic cache key for a translation request."""
    fingerprint = f"{source_text}|{source_lang}|{target_lang}"
    return f"completionist:translate:{hashlib.sha256(fingerprint.encode()).hexdigest()}"


def _translate_with_cache(source_text, llm_config, cache):
    """Translate source_text, checking/writing the Redis cache if available."""
    if cache:
        key = _cache_key(
            source_text, llm_config["source_lang"], llm_config["target_lang"]
        )
        cached = cache.get(key)
        if cached:
            return cached

    completion = get_completion(
        prompt=source_text,
        model_name=llm_config["model_name"],
        api_url=llm_config["api_url"],
        system_prompt=llm_config["system_prompt"],
        hf_api_token=llm_config["hf_api_token"],
        openai_api_token=llm_config["openai_api_token"],
        temperature=llm_config["temperature"],
        top_p=llm_config["top_p"],
        max_tokens=llm_config.get("max_tokens", 2048),
        reasoning_effort=llm_config.get("reasoning_effort"),
        reasoning=llm_config.get("reasoning"),
    )

    if completion and cache:
        cache.set(key, completion["content"])

    return completion["content"] if completion else None


def translate_task_handler(sample, llm_config):
    """
    Task handler for translating one or more fields in a single sample.
    Translates each field independently and returns a dict with source_* and
    translated_* columns for every field.
    """
    cache = llm_config.get("cache")
    result = {}
    for field in llm_config["input_fields"]:
        source_text = sample.get(field)
        if source_text is None:
            continue
        if not source_text:
            result[f"source_{field}"] = source_text or ""
            result[f"translated_{field}"] = ""
            continue

        translated = _translate_with_cache(source_text, llm_config, cache)
        if translated:
            result[f"source_{field}"] = source_text
            result[f"translated_{field}"] = translated
        else:
            return None

    return result if result else None


@click.command("translate")
@click.option(
    "--dataset-name", required=True, help="The name of the Hugging Face dataset to use."
)
@click.option(
    "--input-field",
    "input_fields",
    required=True,
    multiple=True,
    help="The name of a field in the input dataset containing text to translate. "
    "Repeat for multiple fields (e.g. --input-field instruction --input-field output).",
)
@click.option(
    "--source-lang",
    required=True,
    help="The source language (e.g. 'English', 'Spanish').",
)
@click.option(
    "--target-lang",
    required=True,
    help="The target language to translate into (e.g. 'French', 'German').",
)
@click.option(
    "--output-file",
    required=True,
    help="The path to save the translated dataset (e.g., output.parquet).",
)
@click.option(
    "--model-name",
    default="translategemma-4b-it-GGUF-Q4_K_M",
    help="The name of the model to use for translation.",
    show_default=True,
)
@click.option(
    "--api-url",
    default="http://localhost:11434/v1",
    help="(Optional) The API endpoint URL for the LLM. Defaults to Ollama's OpenAI-compatible endpoint.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="(Optional) A custom system prompt for translation. Cannot be used with --system-prompt-file. "
    "Defaults to a generated translation prompt using --source-lang and --target-lang.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="(Optional) Path to a file containing the system prompt. Cannot be used with --system-prompt.",
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
    help="The Hugging Face repository ID to push the dataset to. Required if --push-to-hub is used.",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="(Optional) Number of concurrent requests to make to the API. Defaults to 4.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for generation.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum tokens to generate (including reasoning). Increase for reasoning models.",
    show_default=True,
)
@click.option(
    "--top-p", type=float, default=0.95, help="Nucleus sampling (top-p) for generation."
)
@click.option(
    "--reasoning-effort",
    type=str,
    default=None,
    help="(Optional) Reasoning effort level. Set to 'low', 'medium', 'high', or leave unset "
    "to disable reasoning (faster).",
)
@click.option(
    "--cache-url",
    type=str,
    default=None,
    help="(Optional) Redis URL for translation cache (e.g. redis://localhost:6379). "
    "Requires a running Redis container and 'pip install redis'.",
)
def translate_cmd(
    dataset_name,
    input_fields,
    source_lang,
    target_lang,
    output_file,
    model_name,
    api_url,
    system_prompt,
    system_prompt_file,
    limit,
    shuffle,
    push_to_hub,
    hf_repo_id,
    workers,
    temperature,
    max_tokens,
    top_p,
    reasoning_effort,
    cache_url,
):
    """
    Translate one or more text fields in a dataset from a source language
    to a target language using an LLM. Output columns are named
    source_{field} and translated_{field} for each input field.
    """
    if system_prompt and system_prompt_file:
        raise click.UsageError(
            "Error: --system-prompt and --system-prompt-file are mutually exclusive."
        )

    hf_api_token = get_token()
    openai_api_token = os.environ.get("OPENAI_API_TOKEN", None)

    if push_to_hub and not hf_repo_id:
        handle_error("Error: --hf-repo-id is required when --push-to-hub is used.")

    system_prompt_content = read_file_content(system_prompt_file) or system_prompt
    if not system_prompt_content:
        system_prompt_content = (
            f"You are a professional translator. "
            f"Translate the following text from {source_lang} to {target_lang}. "
            f"Return only the translated text with no additional commentary."
        )

    # Use the first field for dataset validation / resume checks
    primary_field = input_fields[0]
    dataset_to_process, resume_idx, total_samples_in_dataset, existing_translations = (
        load_and_prepare_dataset(
            dataset_name=dataset_name,
            output_file=output_file,
            prompt_input_field=primary_field,
            shuffle=shuffle,
            limit=limit,
        )
    )

    cache = _get_cache_client(cache_url)
    if cache:
        print(f"Using translation cache: {cache_url}")

    llm_config = {
        "model_name": model_name,
        "api_url": api_url,
        "system_prompt": system_prompt_content,
        "hf_api_token": hf_api_token,
        "openai_api_token": openai_api_token,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "input_fields": list(input_fields),
        "reasoning_effort": reasoning_effort,
        "reasoning": "off",
        "source_lang": source_lang,
        "target_lang": target_lang,
        "cache": cache,
    }

    fields_label = ", ".join(input_fields)
    print(
        f"Translating fields [{fields_label}] from {source_lang} to {target_lang} "
        f"for {len(dataset_to_process)} samples "
        f"(out of {total_samples_in_dataset}) with {workers} workers..."
    )

    def _save_progress(completions):
        save_and_push_dataset(
            existing_translations + completions,
            output_file,
            push_to_hub,
            hf_repo_id,
            hf_api_token,
        )

    new_translations = process_samples_with_executor(
        dataset_to_process=dataset_to_process,
        workers=workers,
        resume_idx=resume_idx,
        task_handler=translate_task_handler,
        llm_config=llm_config,
        save_callback=_save_progress,
        save_every=50,
    )

    if len(new_translations) > 0:
        _save_progress(new_translations)

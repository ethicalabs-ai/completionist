import os
import re
from dataclasses import asdict

import click
import httpx
from huggingface_hub import get_token
from openai import OpenAI as OpenAIClient

# Import the hallbayes toolkit
from hallbayes.toolkit import OpenAIBackend, OpenAIItem, OpenAIPlanner

from completionist.dataset_io import load_and_prepare_dataset, save_and_push_dataset
from completionist.llm_api import get_completion
from completionist.processing import process_samples_with_executor
from completionist.utils import handle_error, read_file_content


def run_hallucination_check(prompt, client, hallbayes_config):
    """
    Runs the hallucination check using the hallbayes toolkit.
    """
    backend = OpenAIBackend(model=hallbayes_config["model_name"], client=client)
    planner = OpenAIPlanner(backend, temperature=0.3)

    # Decide skeleton policy based on whether an evidence field is provided
    if hallbayes_config["evidence_field"]:
        item = OpenAIItem(
            prompt=prompt,
            n_samples=hallbayes_config["n_samples"],
            m=hallbayes_config["m_skeletons"],
            fields_to_erase=[hallbayes_config["evidence_field"]],
            skeleton_policy="auto",  # auto will use evidence_erase if field is present
        )
    else:
        # No evidence field, so run in closed-book mode
        item = OpenAIItem(
            prompt=prompt,
            n_samples=hallbayes_config["n_samples"],
            m=hallbayes_config["m_skeletons"],
            skeleton_policy="closed_book",
        )

    metrics = planner.run(
        [item],
        h_star=hallbayes_config["h_star"],
        isr_threshold=1.0,  # Standard ISR gate
        margin_extra_bits=0.2,
    )
    return metrics[0] if metrics else None


def complete_task_handler(sample, llm_config):
    """
    Helper function to generate a completion for a single sample for the 'complete' command.
    """
    prompt = ""
    prompt_template = llm_config.get("prompt_template")

    if prompt_template:
        try:
            # If evidence field is provided, ensure it's formatted correctly in the prompt
            if llm_config.get("hallbayes_config", {}).get("evidence_field"):
                evidence_field_name = llm_config["hallbayes_config"]["evidence_field"]
                if (
                    evidence_field_name in sample
                    and f"{{{evidence_field_name}}}" in prompt_template
                ):
                    # Add the field name prefix for hallbayes to recognize it
                    sample[evidence_field_name] = (
                        f"{evidence_field_name}: {sample[evidence_field_name]}"
                    )

            prompt = prompt_template.format(**sample)
        except KeyError as e:
            print(
                f"\nError: The placeholder {{{e.args[0]}}} in your prompt template was not found "
                f"as a column in the dataset. Available columns: {list(sample.keys())}"
            )
            return None
    else:
        prompt = sample[llm_config["prompt_input_field"]]

    # Hallucination Detection Step
    hallucination_info = None
    if llm_config.get("hallucination_detection"):
        client = llm_config["client"]
        hallbayes_config = llm_config["hallbayes_config"]
        metrics = run_hallucination_check(prompt, client, hallbayes_config)

        if metrics:
            hallucination_info = asdict(metrics)
            if not metrics.decision_answer:
                # High risk of hallucination detected
                if hallbayes_config["hallucination_action"] == "skip":
                    print(
                        f"\nSkipping sample due to high hallucination risk (ISR={metrics.isr:.2f})."
                    )
                    return None  # Skip the sample

    completion = get_completion(
        prompt=prompt,
        model_name=llm_config["model_name"],
        api_url=llm_config["api_url"],
        system_prompt=llm_config["system_prompt"],
        hf_api_token=llm_config["hf_api_token"],
        openai_api_token=llm_config["openai_api_token"],
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

        output = {
            llm_config["prompt_output_field"]: prompt,
            llm_config["completion_output_field"]: cleaned_completion,
            "reasoning": reasoning,
        }
        if hallucination_info:
            output["hallucination_info"] = hallucination_info

        return output
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
    help="(Optional) The API endpoint URL for the LLM.",
)
@click.option(
    "--system-prompt",
    default=None,
    help="(Optional) A system prompt. Cannot be used with --system-prompt-file.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="(Optional) Path to a file containing the system prompt.",
)
@click.option(
    "--prompt-template-file",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default=None,
    help="(Optional) Path to a prompt template file (e.g. {column_name}).",
)
@click.option("--max-tokens", type=int, default=2048)
@click.option("--limit", type=int, default=None)
@click.option("--shuffle", is_flag=True)
@click.option("--push-to-hub", is_flag=True)
@click.option("--hf-repo-id", default=None)
@click.option("--workers", type=int, default=4)
@click.option(
    "--prompt-input-field",
    required=True,
    help="The field from the input dataset to use as the prompt.",
)
@click.option("--prompt-output-field", default="prompt")
@click.option("--completion-output-field", default="completion")
@click.option("--temperature", type=float, default=0.7)
@click.option("--top-p", type=float, default=0.95)
# --- New Hallucination Detection Options ---
@click.option(
    "--hallucination-detection",
    is_flag=True,
    help="Enable the hallucination detection layer before generation.",
)
@click.option(
    "--evidence-field",
    default=None,
    help="[Hallucination Detection] The dataset field containing evidence for the prompt (e.g., 'passage' for BoolQ). Enables evidence-based detection.",
)
@click.option(
    "--hallucination-action",
    type=click.Choice(["flag", "skip"]),
    default="flag",
    show_default=True,
    help="[Hallucination Detection] Action to take if high risk is detected.",
)
@click.option(
    "--h-star",
    type=float,
    default=0.05,
    show_default=True,
    help="[Hallucination Detection] Target max hallucination rate (e.g., 0.05 for 5%).",
)
@click.option(
    "--n-samples",
    type=int,
    default=7,
    show_default=True,
    help="[Hallucination Detection] Number of samples for polling the model.",
)
@click.option(
    "--m-skeletons",
    type=int,
    default=6,
    show_default=True,
    help="[Hallucination Detection] Number of prompt skeletons to generate.",
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
    hallucination_detection,
    evidence_field,
    hallucination_action,
    h_star,
    n_samples,
    m_skeletons,
):
    """
    Generate text completions for a dataset using an LLM, with an optional
    hallucination detection layer.
    """
    if system_prompt and system_prompt_file:
        raise click.UsageError(
            "Error: --system-prompt and --system-prompt-file are mutually exclusive."
        )

    hf_api_token = get_token()
    openai_api_token = os.environ.get("OPENAI_API_TOKEN", None)

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

    # Create a single, shared API client
    api_token = hf_api_token if "huggingface.cloud" in api_url else openai_api_token
    client = OpenAIClient(
        base_url=api_url,
        api_key=api_token or "dummy",
        http_client=httpx.Client(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
        ),
    )

    llm_config = {
        "model_name": model_name,
        "api_url": api_url,
        "system_prompt": system_prompt_content,
        "prompt_template": prompt_template,
        "hf_api_token": hf_api_token,
        "openai_api_token": openai_api_token,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_input_field": prompt_input_field,
        "prompt_output_field": prompt_output_field,
        "completion_output_field": completion_output_field,
        "client": client,  # Pass the shared client
        "hallucination_detection": hallucination_detection,
        "hallbayes_config": {
            "model_name": model_name,
            "evidence_field": evidence_field,
            "hallucination_action": hallucination_action,
            "h_star": h_star,
            "n_samples": n_samples,
            "m_skeletons": m_skeletons,
        },
    }

    print(
        f"Starting completion generation for {len(dataset_to_process)} samples (out of {total_samples_in_dataset}) with {workers} workers..."
    )
    if hallucination_detection:
        print(
            f"Hallucination detection is ENABLED (Action: {hallucination_action}, Target RoH: {h_star * 100:.1f}%)"
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

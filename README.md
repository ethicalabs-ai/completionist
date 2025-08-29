# completionist

Command-line tool for generating new syntetic text datasets, by iterating over an existing Hugging Face dataset and using a LLM to create completions.

## 🛠️ Requirements

To run this project, you'll need:

- Python >=3.11 or a Container Engine (Podman, Docker..)
- A local Open-AI compatible API endpoint (Ollama, LM Studio, vLLM...)
- Or an Hugging Face inference endpoints. 

Default Ollama API endpoint is set as default.

Remember to pull your model from Ollama (or LM Studio) before running Completionist.

```
ollama pull hf.co/ethicalabs/Kurtis-E1.1-Qwen3-4B-GGUF:latest
```

## Basic Usage

To generate a new dataset (in this example, from `mrs83/kurtis_mental_health`) and save the output to a local Parquet file, use the following command.

```
uv run python3 -m completionist complete \
  --dataset-name mrs83/kurtis_mental_health \
  --prompt-input-field Context \
  --model-name hf.co/ethicalabs/Kurtis-E1.1-Qwen3-4B-GGUF:latest \
  --system-prompt "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai. You provide thoughtful and supportive responses to user queries" \
  --output-file generated_dataset.parquet
```

This command will:

- Use the `Context` column from the input dataset as the prompt.
- Use `hf.co/ethicalabs/Kurtis-E1.1-Qwen3-4B-GGUF:latest` for generation.
- Defines a system prompt to prepend to each user prompt.
- Store the resulting dataset in `generated_dataset.parquet` locally.

Hugging Face inference endpoints are supported as well, but please remember to use `tgi` as model name for TGI endpoints:

```
uv run python3 -m completionist complete \
  --api-url https://xxxxxxxxxxxxxxx.us-east-1.aws.endpoints.huggingface.cloud/v1 \
  --dataset-name mrs83/kurtis_mental_health \
  --prompt-input-field Context \
  --model-name tgi \
  --system-prompt "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai. You provide thoughtful and supportive responses to user queries" \
  --output-file generated_dataset.parquet
```

## Running with a Container Engine (Podman)

```
mkdir -p datasets
podman run -it -v ./datasets:/app/datasets ethicalabs/completionist:latest complete \
  --api-url http://host.containers.internal:11434/v1 \
  --dataset-name mrs83/kurtis_mental_health \
  --prompt-input-field Context \
  --model-name hf.co/ethicalabs/Kurtis-E1.1-Qwen3-4B-GGUF:latest \
  --system-prompt "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai. You provide thoughtful and supportive responses to user queries" \
  --output-file datasets/generated_dataset.parquet
```

In this example, `--api-url` is set to the Ollama HTTP server, listening on the host machine (`host.containers.internal:11434`).

## Generating Structured Datasets with build

The `build` command generates a new, structured dataset from scratch based on a dataset source. In the given example, it's a list of topics.

It uses the outlines library to enforce a specific JSON schema (defined by a `Pydantic` model schema), making it ideal for creating high-quality, structured data for tasks like instruction tuning.

The following example command uses a local LM Studio endpoint to generate a dataset with prompt, completion, and reasoning samples.

It will generate `--num-samples` for each topic defined in the `--topics-file`.

```
uv run python3 -m completionist build \                              
  --api-url http://localhost:1234/v1 \
  --model-name Kurtis-E1.1-Qwen2.5-3B-Instruct-GGUF/Kurtis-E1.1-Qwen2.5-3B-Instruct.Q4_K_S.gguf \
  --num-samples 10 \
  --output-file build_output.jsonl \
  --system-prompt-file docs/examples/computer-says-no/build/sft/system.txt \
  --user-prompt-template-file docs/examples/computer-says-no/build/sft/prompt-reasoning.txt \
  --topics-file docs/examples/computer-says-no/build/sft/tasks.txt \
  --schema completionist.default_schema.SchemaWithReasoning
```

## Future Development

This tool's functionality will be expanded in the near future to support different tasks. 

The plan is to add subcommands to handle various scenarios, such as translating text and post-processing.

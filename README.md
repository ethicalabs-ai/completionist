# completionist

Command-line tool for generating new syntetic text datasets, by iterating over an existing Hugging Face dataset and using a LLM to create completions.

## Basic Usage

To generate a new dataset (in this example, from `mrs83/kurtis_mental_health`) and save the output to a local Parquet file, use the following command.

```
uv run python3 -m completionist \
  --dataset-name mrs83/kurtis_mental_health \
  --prompt-input-field Context \
  --completion-output-field Response \
  --model-name hf.co/ethicalabs/Kurtis-E1.1-Qwen3-4B-GGUF:latest \
  --system-prompt "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai. You provide thoughtful and supportive responses to user queries" \
  --output-file generated_dataset.parquet
```

This command will:

- Use the `Context` column from the input dataset as the prompt.
- Save the LLM's response to the `Response` column in the output.
- Use `hf.co/ethicalabs/Kurtis-E1.1-Qwen3-4B-GGUF:latest` for generation.
- Defines a system prompt to prepend to each user prompt.
- Store the resulting dataset in `generated_dataset.parquet` locally.

Hugging Face inference endpoints are supported as well, but please remember to use `tgi` as model name:

```
uv run python3 -m completionist \
  --api-url https://xxxxxxxxxxxxxxx.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions \
  --dataset-name mrs83/kurtis_mental_health \
  --prompt-input-field Context \
  --completion-output-field Response \
  --model-name tgi \
  --system-prompt "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai. You provide thoughtful and supportive responses to user queries" \
  --output-file generated_dataset.parquet
```

## Future Development

This tool's functionality will be expanded in the near future to support different tasks. 

The plan is to add subcommands to handle various scenarios, such as generating datasets for DPO training, performing dataset cleanup, or translating text.

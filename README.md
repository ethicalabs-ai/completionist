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

## Generating Multi-Turn Conversations with chat

The `chat` command generates multi-turn conversation datasets from a list of topics. Each topic seeds one or more realistic dialogues between a `user` and an `assistant`, produced as structured JSON via outlines (no `--schema` flag needed).

```
uv run python3 -m completionist chat \
  --topics-file topics.txt \
  --num-conversations 3 \
  --min-turns 4 \
  --max-turns 6 \
  --model-name Gemma-4-E2B-it-GGUF \
  --api-url http://localhost:11434/v1 \
  --output-file chat_dataset.parquet
```

This generates 3 conversations for each topic in `topics.txt` (one topic per line), each with 4–6 messages.

### Key options

| Option | Description |
|--------|-------------|
| `--topics-file` | Path to a text file with one topic per line. **Required.** |
| `--num-conversations` | Number of conversations to generate per topic. **Required.** |
| `--min-turns` / `--max-turns` | Message count range. Defaults `4`/`6`. **Must be even** so the conversation ends on an `assistant` turn (user-first alternation). |
| `--output-file` | Output path (`.parquet` or `.jsonl`). **Required.** |
| `--model-name` | Model to use. **Required.** |
| `--api-url` | OpenAI-compatible endpoint. Defaults to Ollama. |
| `--system-prompt` / `--system-prompt-file` | Override the built-in conversational system prompt. |
| `--user-prompt-template` | Override the user-prompt template (`{topic}` and `{num_turns}` are substituted). |
| `--workers` | Concurrent requests. Defaults to `4`; set to `1` for single-request servers like llama.cpp. |
| `--temperature` / `--top-p` | Sampling controls. Defaults `0.7` / `0.95`. |
| `--max-tokens` | Token budget per conversation. Defaults to `4096`. |
| `--reasoning` | Reasoning mode: `on`, `off`, or `auto`. |
| `--push-to-hub` / `--hf-repo-id` | Push the result to the Hugging Face Hub. |

The output schema has a `topic` field and a `messages` list of `{role, content}` objects, with roles strictly alternating `user` → `assistant` starting with `user`.

## Generating Topics with topics

The `topics` command generates conversation topic lists with an LLM — useful for seeding the `chat` command. It requests topics in batches and deduplicates automatically across batches.

```
uv run python3 -m completionist topics \
  --num-topics 100 \
  --categories "technology, philosophy, science, ethics" \
  --model-name Gemma-4-E2B-it-GGUF \
  --api-url http://localhost:11434/v1 \
  --output topics.txt
```

Output format is determined by the file extension:

- `.txt` — one topic per line
- `.jsonl` — one `{"topic": "..."}` object per line

### Key options

| Option | Description |
|--------|-------------|
| `--output` / `-o` | Output path. `.txt` or `.jsonl` extension selects the format. **Required.** |
| `--num-topics` / `-n` | Number of unique topics to generate. **Required.** |
| `--model-name` | Model to use. **Required.** |
| `--categories` | Comma-separated category hints to guide diversity. **Required.** |
| `--api-url` | OpenAI-compatible endpoint. Defaults to Ollama. |
| `--batch-size` | Topics per API call. Defaults to `50`. |
| `--temperature` / `--top-p` | Sampling controls. Defaults `0.9` / `0.95`. |
| `--max-tokens` | Token budget per batch. Defaults to `4096`. |
| `--system-prompt` / `--system-prompt-file` | Override the built-in topic-curation prompt. |
| `--reasoning` | Reasoning mode: `on`, `off`, or `auto`. |
| `--max-batches` | Upper bound on API calls before stopping. Defaults to `20`. |
| `--append` | Append to an existing file; `--num-topics` becomes how many *new* topics to add on top of what is already there. |

Without `--append`, the output file is overwritten. With `--append`, existing topics are loaded first and only new unique topics are generated, so repeated runs accumulate a non-overlapping pool.

## Translating Datasets with translate

The `translate` command translates one or more text fields of a dataset using an LLM, with optional Redis caching to skip already-translated strings across runs.

```
uv run python3 -m completionist translate \
  --dataset-name ethicalabs/kurtis-v2-sft-mix-tiny \
  --input-field prompt \
  --input-field completion \
  --input-field reasoning \
  --source-lang English \
  --target-lang Italian \
  --model-name Gemma-4-E4B-it-GGUF \
  --api-url http://localhost:11434/v1 \
  --output-file translated.parquet
```

Repeat `--input-field` once per field to translate. For each input field the output adds two columns: `source_{field}` and `translated_{field}` (e.g. `source_prompt` / `translated_prompt`).

### Key options

| Option | Description |
|--------|-------------|
| `--dataset-name` | Hugging Face dataset name. **Required.** |
| `--input-field` | Field to translate. Repeatable for multiple fields. **Required.** |
| `--source-lang` / `--target-lang` | Language pair. **Required.** |
| `--output-file` | Output path. **Required.** |
| `--model-name` | Translation model. Defaults to `translategemma-4b-it-GGUF-Q4_K_M`. |
| `--api-url` | OpenAI-compatible endpoint. Defaults to Ollama. |
| `--system-prompt` / `--system-prompt-file` | Override the generated translation prompt (mutually exclusive). |
| `--limit` | Process only the first N samples. |
| `--shuffle` | Shuffle the dataset before processing. |
| `--workers` | Concurrent requests. Defaults to `4`. |
| `--temperature` / `--top-p` / `--max-tokens` | Sampling controls. Defaults `0.7` / `0.95` / `4096`. |
| `--cache-url` | Redis URL (e.g. `redis://localhost:6379`) to cache translations. Requires `pip install redis` and a running Redis server. |
| `--push-to-hub` / `--hf-repo-id` | Push the result to the Hugging Face Hub. |

## Reading Datasets with inspect

The `inspect` command reads any dataset completionist produces and auto-detects its shape — no flags needed to pick a view. It opens an interactive pager, or falls back to a plain-text dump when the output is piped.

```
uv run python3 -m completionist inspect datasets/chat_sft_merged.parquet
uv run python3 -m completionist inspect ethical-01.jsonl
uv run python3 -m completionist inspect ethicalabs/some-dataset --split train
```

### Detected formats

| Format | Detection | Tabs |
|--------|-----------|------|
| Conversation | `messages` (list of `{role, content}`) | `conversation`, `raw` |
| Prompt/completion | `prompt` + `completion` columns | `prompt`, `completion`, `reasoning` (if present), `raw` |
| Tools | `messages` with `tool_calls`/`tool` turns, or a `tools` field | `conversation`, `tools`, `raw` |
| Fallback | anything else | `raw` (pretty-printed JSON of every field) |

The `raw` tab is always present, so any dataset is inspectable.

### Keys

- `←` / `→` — previous / next sample
- `Tab` / `Shift-Tab` — cycle between tabs
- `s` — cycle between splits (when the dataset has several)
- `↑` / `↓` — scroll
- `PgUp` / `PgDn` — page up / down
- `Home` / `End` — jump to top / bottom
- `q` / `Esc` — quit

### Options

| Option | Description |
|--------|-------------|
| `dataset` | Local `.parquet`/`.jsonl` file, local directory (HF split layout), or a Hugging Face dataset name. **Positional.** |
| `--split` | Split to load. Auto-detected (`train` → `test` → `validation` → first) if omitted. |
| `--limit` | Load at most N samples. |
| `--index` | Skip this many samples before reading. |

Datasets are read via streaming — remote and local alike — with a bounded in-memory window; older samples spill to a `/tmp` cache so backward navigation works without holding the whole dataset in RAM.

## Future Development

This tool's functionality will be expanded in the near future to support additional dataset-processing tasks and post-processing utilities.

# Specification

## Product

**model-rs** is a command-line tool and Rust library that:

1. **Downloads** model weights from Hugging Face (mirror configurable via `MODEL_RS_MIRROR`, default in `src/config.rs`).
2. **Lists and manages** local model directories (path resolution, copy, remove, verify, metadata).
3. **Runs inference** on local weights: text generation, streaming generation, interactive chat (session file + slash commands), and **embeddings** for encoder-style checkpoints (e.g. BERT family). Core path uses **Candle**; **GGUF** and **MLX** are optional (`Cargo.toml` features).
4. **Serves HTTP**: OpenAI-ish `/v1/*` endpoints and **Ollama-compatible** `/api/*` routes for generate, chat, tags, embeddings, pull, copy, and delete where implemented.

## Non-goals (current)

- Hosted multi-tenant serving beyond what the local binary provides.
- Training or fine-tuning workflows.

## CLI

The user-facing binary name is **`model-rs`**. Top-level subcommands (from `src/cli.rs`; run `model-rs --help` for flags):

`download` (alias `pull`), `search`, `serve`, `generate`, `run`, `stop`, `chat`, `list` (alias `ls`), `deploy`, `embed`, `config`, `show`, `remove` (alias `rm`), `ps`, `copy`, `info`, `verify`, `cache` (stats / clear / enable / preload / evict / max).

**Behavior notes:**

- **`serve`** and **`deploy`** both call the same Axum server (`influencer::serve`). `deploy --detached` only changes user-facing messages; it does not fork or detach the process.
- **`run`** accepts an optional model name; if omitted, **`MODEL_RS_MODEL_PATH`** must point at a directory.
- **`chat`** requires `--model-path` (filesystem path). Interactive mode supports slash commands (`/help`, `/clear`, `/history`, `/save`, `/load`, `/set`, etc.) implemented in `src/influencer/mod.rs`.

Exact surface area and defaults are defined by `src/cli.rs` and `model-rs --help`.

## HTTP API

Implemented routes (`src/influencer/server.rs` `build_app`):

| Method(s) | Path | Purpose |
|-----------|------|---------|
| GET | `/health` | Liveness |
| POST | `/v1/generate` | JSON generate |
| POST | `/v1/generate_stream` | SSE streaming |
| POST | `/v1/generate_batch` | Batch generate (same loaded model) |
| POST | `/api/generate` | Ollama-style generate (optional stream → NDJSON) |
| POST | `/api/chat` | Ollama-style chat |
| POST | `/api/show` | Ollama-style show |
| POST | `/api/embeddings`, `/api/embed` | Embeddings |
| GET, POST | `/api/tags` | List models |
| POST, DELETE | `/api/delete` | Remove model |
| POST | `/api/copy` | Copy model |
| POST | `/api/pull` | Pull / ensure model (may trigger download) |

CORS is enabled for all origins/methods/headers (`tower-http`) for local development.

## Configuration

Runtime configuration uses environment variables with prefix **`MODEL_RS_`**. Implemented readers live in `src/config.rs`:

- `MODEL_RS_MODEL_PATH`, `MODEL_RS_OUTPUT_DIR`, `MODEL_RS_MIRROR`
- `MODEL_RS_TEMPERATURE`, `MODEL_RS_TOP_P`, `MODEL_RS_TOP_K`, `MODEL_RS_REPEAT_PENALTY`, `MODEL_RS_MAX_TOKENS`
- `MODEL_RS_DEVICE`, `MODEL_RS_DEVICE_INDEX`, `MODEL_RS_PORT`

Additional variables are documented in the `model-rs config` output in `src/lib.rs`, including **`MODEL_RS_WARMUP_TOKENS`** (decode warmup; read in `src/local/backends.rs`).

Integration tests use **`MODEL_RS_PORT`** (default **8080** when unset) to reach `model-rs serve`.

## Model storage and naming

- Cache root: `directories::ProjectDirs::from("com", "modelrs", "modelrs")` → `cache_dir()/models` (see `src/model_ops.rs`, `src/models.rs`).
- A Hugging Face id `org/model` is stored under a directory named `org--model` (slashes replaced).
- Fast listing may use **`.model_rs_index.json`** in the models directory (`src/models.rs`).

## Compatibility

- Rust edition **2024**.
- **Device** strings: `auto`, `cpu`, `metal`, `cuda`, `mlx` (`DevicePreference` in `src/local/config.rs`).
- **Model architectures** are auto-detected from `config.json` (`model_type`). Candle-based families: Llama, Mistral, Phi, Gemma, Qwen2/3, DeepSeek V2/V3, Kimi (DeepSeek-based), GLM-4, Mamba, BERT, Granite. Optional GGUF and MLX backends via Cargo features.
- API payloads aim to match common OpenAI/Ollama client expectations where routes exist; refer to `src/influencer/server.rs` for authoritative request/response types.

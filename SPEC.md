# Specification

## Product

**model-rs** is a command-line tool and Rust library that:

1. **Downloads** model weights from Hugging Face (mirror configurable via `MODEL_RS_MIRROR`, default in `src/config.rs`).
2. **Lists and manages** local model directories (path resolution, copy, remove, verify, metadata).
3. **Runs inference** on local weights: text generation, streaming generation, interactive chat (session file + slash commands), and **embeddings** for encoder-style checkpoints (e.g. BERT family). Core path uses **Candle**; **GGUF** quantized inference is built-in via candle's `quantized_llama::ModelWeights`.
4. **Serves HTTP**: OpenAI-ish `/v1/*` endpoints and **Ollama-compatible** `/api/*` routes for generate, chat, tags, embeddings, pull, copy, and delete where implemented.
5. **Verifies integrity**: SHA-256 checksum generation and verification for downloaded model files (`verify`, `generate-checksums`).
6. **Manages versions**: Version tracking, pinning, statistics, and cleanup for multiple model downloads (`versions` subcommands).
7. **Validates input**: All CLI arguments are validated with actionable error messages (`validation.rs`).
8. **Supports config files**: TOML/YAML configuration at `~/.config/model-rs/config.toml` with environment variable overrides (`config` subcommands).

## Non-goals (current)

- Hosted multi-tenant serving beyond what the local binary provides.
- Training or fine-tuning workflows.

## CLI

The user-facing binary name is **`model-rs`**. Top-level subcommands (from `src/cli.rs`; run `model-rs --help` for flags):

`download` (alias `pull`), `search`, `serve`, `generate`, `run`, `stop`, `chat`, `list` (alias `ls`), `deploy`, `embed`, `config` (init / edit / show / sources / validate / env / reset), `show`, `remove` (alias `rm`), `ps`, `copy`, `info`, `verify`, `generate-checksums`, `versions` (list / pin / unpin / stats / cleanup), `cache` (stats / clear / enable / preload / evict / max).

**Behavior notes:**

- **`serve`** and **`deploy`** both call the same Axum server (`influencer::serve`). `deploy --detached` only changes user-facing messages; it does not fork or detach the process.
- **`run`** accepts an optional model name; if omitted, **`MODEL_RS_MODEL_PATH`** must point at a directory.
- **`chat`** requires `--model-path` (filesystem path). Interactive mode supports slash commands (`/help`, `/clear`, `/history`, `/save`, `/load`, `/set`, etc.) implemented in `src/influencer/mod.rs`. Session save/load via `--session` and `--save-on-exit` flags.
- **`config`** is now a subcommand-based command: `init` creates `~/.config/model-rs/config.toml`, `edit` opens it in `$EDITOR`, `show` prints merged config, `sources` shows where each value comes from, `validate` checks config validity, `env` prints environment-derived configuration (resolved `MODEL_RS_*` values), `reset` restores defaults.
- **`versions`** manages model version tracking: `list [model]` shows versions, `pin <model> <version>` pins a version, `unpin <model> <version>` unpins, `stats` shows aggregate statistics, `cleanup [--keep-latest N] [--keep-pinned]` removes old versions.
- **`verify`** checks model integrity against stored SHA-256 checksums (`.model_checksums` file in model directory). **`generate-checksums`** creates or refreshes the checksum file.
- **`cache`** manages the in-process model cache: `--stats` shows cache status, `--clear` evicts all, `--enable <bool>` toggles caching, `--preload <model>` loads a model, `--evict <model>` removes one, `--max <n>` sets capacity.

Exact surface area and defaults are defined by `src/cli.rs` and `model-rs --help`.

## HTTP API

Implemented routes (`src/influencer/server.rs` `build_app`):

| Method(s) | Path | Purpose |
|-----------|------|---------|
| GET | `/health` | Liveness |
| POST | `/v1/generate` | JSON generate |
| POST | `/v1/generate_stream` | SSE streaming |
| POST | `/v1/generate_batch` | Batch generate (sequential per-request under the hood; see Batch notes below) |
| POST | `/api/generate` | Ollama-style generate (optional stream → NDJSON) |
| POST | `/api/chat` | Ollama-style chat |
| POST | `/api/show` | Ollama-style show |
| POST | `/api/embeddings`, `/api/embed` | Embeddings |
| GET, POST | `/api/tags` | List models |
| POST, DELETE | `/api/delete` | Remove model |
| POST | `/api/copy` | Copy model |
| POST | `/api/pull` | Pull / ensure model (may trigger download) |

CORS is enabled for all origins/methods/headers (`tower-http`) for local development.

## Batch generation

`/v1/generate_batch` accepts a JSON array of generation requests and returns an array of results. **Implementation note** (`src/local/batch.rs`): the endpoint runs each request **sequentially** through the same loaded model — it does not yet perform true padded/masked batching (TODO in source). Callers should not assume throughput scales with batch size; latency is approximately `N × single-request latency`.

## Configuration

Configuration is resolved from three sources in priority order (highest first):

1. **Environment variables** with prefix **`MODEL_RS_`** (read in `src/config.rs`)
2. **Config file** at `~/.config/model-rs/config.toml` (or `.yaml`) — managed by `src/config_file.rs` (`ConfigManager`, `AppConfig`)
3. **Built-in defaults** (`AppConfig::default`)

Environment variable readers (`src/config.rs`):

- `MODEL_RS_MODEL_PATH`, `MODEL_RS_OUTPUT_DIR`, `MODEL_RS_MIRROR`
- `MODEL_RS_TEMPERATURE`, `MODEL_RS_TOP_P`, `MODEL_RS_TOP_K`, `MODEL_RS_REPEAT_PENALTY`, `MODEL_RS_MAX_TOKENS`
- `MODEL_RS_DEVICE`, `MODEL_RS_DEVICE_INDEX`, `MODEL_RS_PORT`
- `MODEL_RS_WARMUP_TOKENS` (decode warmup; read in `src/local/backends.rs`)

The config file (`AppConfig` in `src/config_file.rs`) covers model, server, generation, device, download, cache, and logging sections. Use `model-rs config init` to create, `model-rs config show` to display merged values, `model-rs config sources` to trace origins, and `model-rs config validate` to check.

Integration tests use **`MODEL_RS_PORT`** (default **8080** when unset) to reach `model-rs serve`.

## Input validation

All CLI inputs are validated before execution (`src/validation.rs`):

- **Model names**: regex `^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?$`, must contain `/` (org/model format)
- **Paths**: no path traversal (`..`), no tilde expansion (`~/`), alphanumeric + `_-/.,@`
- **Model directories**: must exist, be a directory, and contain `config.json`
- **Ports**: 1–65535
- **Generation params**: temperature 0.0–2.0, top-p 0.0–1.0, top-k 1–1000, repeat penalty 0.0–2.0, max tokens 1–32768
- **Devices**: `auto`, `cpu`, `metal` (valid); `cuda`, `mlx` accepted as strings but will error at runtime (backends removed). Device index 0–7.

## Model integrity verification

`src/verification.rs` provides SHA-256 checksum-based integrity verification:

- **`ModelVerifier`**: calculates and stores checksums in `.model_checksums` (JSON) within each model directory.
- **`ModelIntegrityManager`**: resolves model names to paths, verifies individual or all models, generates checksums.
- **`verify <model>`**: verifies stored checksums against current files; exits with code 1 on failure.
- **`generate-checksums <model>`**: creates/refreshes `.model_checksums`, then verifies.

## Version management

`src/version_manager.rs` tracks model versions with pinning support:

- **`ModelVersion`**: stores model ID, version string, path, size, download date, checksum, pinned status, tags.
- **`ModelVersionManager`**: version index at `versions/version_index.json` within the models directory.
- **`versions list [model]`**: list all or per-model versions.
- **`versions pin <model> <version>`** / **`unpin <model> <version>`**: toggle pinned status.
- **`versions stats`**: aggregate statistics (total models, versions, size, pinned count, date range).
- **`versions cleanup [--model M] [--keep-latest N] [--keep-pinned]`**: remove old versions, keeping the N latest and optionally all pinned.

## Model storage and naming

- Cache root: `directories::ProjectDirs::from("com", "modelrs", "modelrs")` → `cache_dir()/models` (see `src/model_ops.rs`, `src/models.rs`).
- A Hugging Face id `org/model` is stored under a directory named `org--model` (slashes replaced).
- Fast listing may use **`.model_rs_index.json`** in the models directory (`src/models.rs`).

## Compatibility

- Rust edition **2024**.
- **Device** strings: `auto`, `cpu`, `metal` (supported); `cuda`, `mlx` accepted as strings but error at runtime (`DevicePreference` in `src/local/config.rs`).
- **Model architectures** are auto-detected from `config.json` (`model_type`). Candle-based families: Llama, Mistral, Phi, Gemma, Qwen2/3, DeepSeek V2/V3, Kimi (DeepSeek-based), GLM-4, Mamba, BERT, Granite. GGUF quantized inference is built-in (always available, not feature-gated).
- **Rust edition** 2024. A C compiler is required for `onig` (transitive dependency via `candle-core`'s `tokenizers`).
- API payloads aim to match common OpenAI/Ollama client expectations where routes exist; refer to `src/influencer/server.rs` for authoritative request/response types.

# Specification

## Product

**model-rs** is a command-line tool and Rust library that:

1. Downloads model weights from Hugging Face (with configurable mirror).
2. Lists and manages local model directories.
3. Runs text generation and related tasks using local weights (Candle; optional GGUF/MLX via features).
4. Serves an HTTP API for generation, streaming, embeddings, and Ollama-compatible endpoints where implemented.

## Non-goals (current)

- Hosted multi-tenant serving beyond what the local binary provides.
- Training or fine-tuning workflows.

## CLI

The user-facing binary name is **`model-rs`**. Subcommands include download, list, serve, generate, chat, search, cache, config, and operational helpers (show, remove, verify, etc.). Exact surface area is defined by `src/cli.rs` and `model-rs --help`.

## Configuration

Runtime configuration is driven by environment variables with prefix **`MODEL_RS_`** (see `src/config.rs` and `model-rs config`). Integration tests read **`MODEL_RS_PORT`** (default **8080** when unset) to match `model-rs serve`. Optional tuning includes **`MODEL_RS_WARMUP_TOKENS`** for Metal warmup passes in local inference.

## Compatibility

- Rust edition **2024**.
- API shapes aim to align with common OpenAI/Ollama client expectations where routes exist; refer to `src/influencer/` for request/response types.

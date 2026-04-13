# Architecture

## Layout

Single Cargo package (`model-rs`) with a **library** (`model_rs`) and a **binary** (`model-rs`) sharing `src/`.

| Area | Path | Role |
|------|------|------|
| CLI parsing | `src/cli.rs` | Clap commands and flags |
| Entry / dispatch | `src/lib.rs` `run()` | Loads `.env`, tracing, routes subcommands |
| Binary | `src/main.rs` | Calls `model_rs::run()` |
| Config | `src/config.rs` | `MODEL_RS_*` environment helpers |
| Download / HF | `src/download.rs`, `src/search.rs` | Pull models and search catalogs |
| Local inference | `src/local/` | Candle backends, generation, cache, device selection |
| Model index / paths | `src/models.rs`, `src/model_ops.rs` | Listing, cache dir (`directories`), resolve paths |
| HTTP API | `src/influencer/` | Axum server, OpenAI/Ollama-style routes |
| Output | `src/output.rs`, `src/format.rs` | Terminal / markdown style output |
| Errors | `src/error.rs` | `ModelError`, shared `Result` |

## Data paths

Application cache and model storage use `directories::ProjectDirs::from("com", "modelrs", "modelrs")`. Downloaded models are resolved under that cache hierarchy (see `model_ops`).

A JSON index for fast listing is stored as `.model_rs_index.json` in the models directory.

## Features

Optional acceleration and formats are gated in `Cargo.toml`: `metal` (default on macOS), `cuda`, `gguf`, `mlx`, etc.

## Tests and benches

- Integration tests in `tests/` exercise HTTP and CLI behavior.
- Criterion bench in `benches/throughput.rs`.

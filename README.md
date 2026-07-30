# model-rs

Rust **CLI** and **library** for downloading Hugging Face models, running **local inference** with [Candle](https://github.com/huggingface/candle) (pure Rust, no C/C++ dependencies), and exposing an **HTTP server** with OpenAI-style and Ollama-compatible routes.

## What it does

- **Download & search** — Pull weights through a configurable mirror (`MODEL_RS_MIRROR`, default HF mirror host) and query the Hub catalog.
- **Local generation** — `generate`, `run`, and `chat` load a model directory, run decoding on CPU or Metal (`auto` picks a backend), and print **markdown-aware** streamed output in the terminal.
- **HTTP API** — `serve` (and `deploy`, same server) bind an Axum app: `/v1/*` generation + SSE, `/api/*` Ollama-style generate, chat, tags, embeddings, pull, copy, delete, etc. See **HTTP API** below.
- **Model housekeeping** — `list`, `show`, `info`, `verify`, `generate-checksums`, `copy`, `remove`, `ps`, `stop`, plus `cache` for the in-process model cache (stats, clear, preload, evict).
- **Version management** — `versions` subcommands (`list`, `pin`, `unpin`, `stats`, `cleanup`) for tracking model versions, pinning specific versions, and cleaning up old downloads.
- **Configuration file** — `config` subcommands (`init`, `edit`, `show`, `sources`, `validate`, `reset`) for TOML/YAML-based configuration at `~/.config/model-rs/config.toml`, with environment variable overrides.
- **Input validation** — All CLI arguments are validated (model names, paths, ports, device strings, generation parameters) with actionable error messages.

## Supported models

`model-rs` auto-detects the architecture from `config.json` (`model_type`). Supported families (Candle-based):

| Family | Detected `model_type` values | Example models |
|--------|------------------------------|----------------|
| **Llama** (Llama 2/3, TinyLlama, etc.) | `llama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Mistral** | `mistral` | Mistral 7B, Mixtral |
| **Phi** (Phi-3/4) | `phi` | `microsoft/Phi-3-mini-4k-instruct` |
| **Gemma** (Gemma 2/3/4) | `gemma`, `gemma2`, `gemma3`, `gemma4` | `google/gemma-2-2b-it` |
| **Qwen2** | `qwen2`, `qwen2_moe` | `Qwen/Qwen2-7B-Instruct` |
| **Qwen3** | `qwen3`, `qwen3_moe`, `qwen3_vl` | `Qwen/Qwen3-8B` |
| **DeepSeek V2/V3** | `deepseek_v2`, `deepseek_v3`, `deepseek` | `deepseek-ai/DeepSeek-V3` |
| **Kimi** (K2.5, etc.) | `kimi`, `kimi_v1` | `moonshotai/Kimi-K2.5` |
| **GLM-4** | `glm4`, `glm4_new`, `chatglm` | `THUDM/glm-4-9b-chat` |
| **Mamba** | `mamba` | State-space models |
| **BERT** (encoder-only) | `bert`, `roberta`, `albert` | Embeddings |
| **Granite** | `granite` | IBM Granite |
| **GraniteMoeHybrid** | `granitemoehybrid` | Attention-only hybrids |

Additional backends:
- **GGUF** — always available; uses candle's pure Rust `quantized::gguf_file` + `quantized_llama::ModelWeights` for quantized inference. No C++ dependencies.
- **MLX** — removed; Metal backend (pure Rust FFI to macOS system framework) provides GPU acceleration on Apple Silicon.

## Compared to Ollama, vLLM, and SGLang

These projects overlap on “run an LLM and talk to it over HTTP,” but they optimize for different stacks and scales. **model-rs** is a **Rust** crate and binary built around **Candle**, Hugging Face–style downloads, and a **subset** of **Ollama-compatible** routes so existing clients can often be pointed here for local experiments—not a drop-in replacement for any of them.

| Topic | [Ollama](https://github.com/ollama/ollama) | [vLLM](https://github.com/vllm-project/vllm) | [SGLang](https://github.com/sgl-project/sglang) | **model-rs** |
|------|-------------|--------|---------|------------|
| **Primary focus** | Easy local models, one installer, rich desktop story | High-throughput **GPU** serving, production OpenAI-style APIs | Fast **GPU** serving, structured / multi-turn workloads, radix-style KV reuse | **Local** pull + run + small Axum server; **library + CLI** in Rust |
| **Runtime / stack** | Go + native runners (e.g. llama.cpp path) | Python, CUDA-centric | Python, CUDA-centric | **Pure Rust** (Candle; GGUF built-in; Metal GPU via system FFI) |
| **Model sources** | Ollama library / `pull` workflow | You supply model weights / HF layout for the server | Same idea—serving-oriented | **HF-oriented** download + mirror; paths under app cache |
| **API shape** | Ollama REST is the product’s contract | **OpenAI-compatible** HTTP (and ecosystem around it) | **OpenAI-compatible** + SGLang-specific features | **Partial** Ollama `/api/*` + some **`/v1/*`** (see table below); not full parity |
| **Sweet spot** | "Install and run" for developers and desktops | Clusters, many concurrent requests, PagedAttention-class serving | Heavy interactive / program-style LLM use on capable GPUs | Hackable **pure Rust** codebase, CPU/Metal options, integrated **HF** fetch, no C/C++ toolchain needed |

**When to prefer something else:** use **Ollama** for the broadest turnkey local ecosystem and Modelfile-style workflows; use **vLLM** or **SGLang** when you need serious **multi-GPU** serving, scheduling, and throughput on a Python stack. Use **model-rs** when you want a **pure Rust** tool that downloads from the Hub, runs **Candle** (with built-in GGUF support), and exposes a **compatible slice** of HTTP for local testing and embedding in other Rust projects.

## Requirements

- Rust toolchain with **edition 2024** support (recent stable).
- **No C/C++ compiler required** — the project is pure Rust. Candle crates are vendored under `vendor/` with a `fancy-regex` patch (replacing `onig` C regex).
- macOS: default build uses **Metal** (`metal` feature, pure Rust FFI to macOS system framework). Other platforms: use `--no-default-features` for CPU-only.

## Quick start

```bash
cargo build --release
./target/release/model-rs --help
./target/release/model-rs download <org>/<model>
./target/release/model-rs list
```

Downloaded models live under the app cache (see **Model storage** in [ARCHITECTURE.md](ARCHITECTURE.md)). Resolve a name like `TinyLlama/TinyLlama-1.1B-Chat-v1.0` to a path with `list` / `show`, or pass `--model-path`.

Run the API server:

```bash
export MODEL_RS_MODEL_PATH=/path/to/model-dir   # or: serve --model-path ...
./target/release/model-rs serve
# default port 8080; override with --port or MODEL_RS_PORT
```

`deploy` starts the **same** server as `serve`. The `--detached` flag only changes onboarding text in the terminal; the process still runs in the foreground (use your shell or a process supervisor for true background operation).

Other useful entry points: `run` / `chat` (interactive TUI-style loop with slash commands, session save/load), `embed` (encoder embeddings to stdout as JSON), `model-rs config show` (resolved configuration from file + env). Full surface: `model-rs --help` and [SPEC.md](SPEC.md).

## HTTP API (summary)

Base URL: `http://127.0.0.1:<port>` (default **8080**).

| Area | Methods | Paths |
|------|---------|--------|
| Health | GET | `/health` |
| OpenAI-style | POST | `/v1/generate`, `/v1/generate_stream` (SSE), `/v1/generate_batch` |
| Ollama-style | POST | `/api/generate`, `/api/chat`, `/api/show`, `/api/embeddings`, `/api/embed`, `/api/pull`, `/api/copy` |
| Ollama-style | GET, POST | `/api/tags` |
| Ollama-style | POST, DELETE | `/api/delete` |

Request and response shapes are defined in `src/influencer/server.rs` (and related types). Integration tests in `tests/integration_test.rs` cover a subset of these endpoints.

## Library

In `Cargo.toml` the package name is `model-rs`; in Rust code the library crate is imported as `model_rs`:

```rust
use model_rs::Result;

#[tokio::main]
async fn main() -> Result<()> {
    model_rs::run().await
}
```

Public modules include `cli`, `config`, `config_file`, `download`, `error`, `format`, `influencer`, `local`, `model_ops`, `models`, `output`, `search`, `validation`, `verification`, and `version_manager`. Examples live under `examples/` (see `examples/README.md`).

## Configuration

Configuration is resolved from three sources (highest priority first): **environment variables**, **config file** (`~/.config/model-rs/config.toml` or `.yaml`), and **built-in defaults**. Use `model-rs config init` to create a config file, `model-rs config show` to display the merged result, and `model-rs config sources` to see where each value comes from.

Environment variables use the **`MODEL_RS_`** prefix. Common keys: `MODEL_RS_MODEL_PATH`, `MODEL_RS_OUTPUT_DIR`, `MODEL_RS_MIRROR`, `MODEL_RS_PORT`, `MODEL_RS_DEVICE`, `MODEL_RS_DEVICE_INDEX`, generation defaults (`MODEL_RS_TEMPERATURE`, `MODEL_RS_TOP_P`, `MODEL_RS_TOP_K`, `MODEL_RS_REPEAT_PENALTY`, `MODEL_RS_MAX_TOKENS`), and optional **`MODEL_RS_WARMUP_TOKENS`** for local decode warmup. Run `model-rs config show` for the full list as interpreted in your environment.

A `.env` file in the working directory is loaded on startup (`dotenvy`).

## Tests and benchmarks

- **Unit / integration in crate:** `cargo test` — 227 tests total (159 lib + 12 API error + 33 E2E + 16 error handling + 7 integration), all passing.
- **API tests** (`tests/integration_test.rs`): require a running server; use `MODEL_RS_PORT` (default **8080** when unset).
- **API error tests** (`tests/api_error_test.rs`): error handling and edge-case coverage for HTTP routes.
- **Error handling tests** (`tests/error_handling_test.rs`): validation, model errors, and recovery paths.
- **CLI / API smoke tests** (`tests/e2e_test.rs`): see `tests/README.md`.
- **Criterion:** `cargo bench` (throughput bench in `benches/throughput.rs`).

## License

Apache-2.0 (see `Cargo.toml`).

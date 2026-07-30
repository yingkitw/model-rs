# TODO

Actionable follow-ups (keep this list honest and small).

## Done

- [x] Confirm `cargo test` passes (227 tests, 0 failures). `cargo build --release` is standard Rust; E2E tests resolve the CLI via runtime `CARGO_BIN_EXE_model_rs` so they stay valid if the repo path changes.
- [x] Integration tests use `MODEL_RS_PORT` with default **8080** (see `tests/integration_test.rs`).
- [x] Input validation module (`src/validation.rs`) — model names, paths, ports, device strings, generation params.
- [x] Model integrity verification (`src/verification.rs`) — SHA-256 checksums, `verify` and `generate-checksums` commands.
- [x] Version management (`src/version_manager.rs`) — `versions list/pin/unpin/stats/cleanup` commands.
- [x] Configuration file support (`src/config_file.rs`) — TOML/YAML at `~/.config/model-rs/config.toml`, `config init/edit/show/sources/validate/reset` subcommands.
- [x] API error tests (`tests/api_error_test.rs`) and error handling tests (`tests/error_handling_test.rs`).
- [x] All root docs (README, SPEC, ARCHITECTURE, AGENTS, TODO) updated to reflect current implementation: pure Rust build, no C/C++ deps, built-in GGUF, removed MLX/CUDA, single `metal` feature, 227 passing tests.

## Pending

- [ ] Optional: add a small `scripts/run_e2e_tests.sh` if you want a one-command local QA wrapper.
- [ ] Publish / version policy: decide on crates.io name vs GitHub repo naming (`model-rs`).
- [ ] Optional: make `deploy --detached` match its name (spawn/daemonize) or rename the flag now that README/SPEC describe current behavior.
- [ ] Fix `cli.rs` syntax issue: duplicate `Cache` variant fields and `Config` variant definition may cause compile errors. **[done: see 2026-07 audit pass below]**

## 2026-07 audit pass

Audit-and-fix changes (see `audit` skill report for full context):

- [x] **cli.rs**: `Cache` variant was outside the `Commands` enum (orphan struct → parse error); moved inside, deleted orphan.
- [x] **lib.rs unreachable `Commands::Config` arm** (matched a non-existent fieldless variant): removed; added `ConfigCommands::Env` case using `config::env_config_markdown()`.
- [x] **`validate_generation_args` helper** in `validation.rs`; deduped `Run`/`Chat`/`Generate` match arms in `lib.rs` (eliminated ~50 lines of repetition).
- [x] **`ModelCache::set_max_cached_models`**: `cache --max` now actually mutates state (interior `Mutex<usize>`).
- [x] **Mutex poison-tolerant** in `local/mod.rs::encode_ids_cached` (`unwrap_or_else(|e| e.into_inner())`).
- [x] **`From<Box<dyn Error + Send + Sync>>` removed** from `error.rs` (it silently misclassified errors as `TokenizerError`).
- [x] **`From<tokenizers::Error>` added** to `error.rs` so `tokenizer.decode(...)?` works.
- [x] **Error constructor helpers** added: `download_failed`, `model_not_found`, `invalid_config`, `llm_error`, `local_model_error`, `validation_error`, `cache_error`, `timeout_error`, `config_file_error`, `authentication_error`.
- [x] **server.rs test fixtures**: `.unwrap()` → `.expect("…")` for clearer panic context.
- [x] **`#[allow(dead_code)]`** removed from `GgufBackend::tokenizer` (the field is read in stream decoding).
- [x] **`SPEC.md`** documents `/v1/generate_batch` as sequential per-request and adds `config env` subcommand.
- [x] **`LlmService` trait** annotated as experimental seam (`influencer/service.rs`); recursive `impl` for `LocalModel` left intact but flagged.

### Still open after the audit pass (pre-existing WIP issues, not introduced by the audit)

- [x] **`cargo check` passes end-to-end**: Fixed all 18 pre-existing WIP compile errors in `config_file.rs` (env::var type mismatches, temporary borrow, &mut self), `verification.rs` (async ModelVerifier::new, get_models_dir), `version_manager.rs` (moved value, missing Digest import), `lib.rs` (mut bindings), `model_ops.rs` (pub get_models_dir).
- [x] **Eliminated all C/C++ dependencies** — project is now pure Rust:
  - `tokenizers`: disabled `default-features` (was pulling `esaxx_fast` C++ and `onig` C regex), enabled `fancy-regex` (pure Rust).
  - `syntect`: switched from `onig` (C regex) to `default-fancy` (`fancy-regex`, pure Rust).
  - `reqwest`: switched from `rustls`/`aws-lc-sys` (C/C++ TLS) to `native-tls` (macOS Security framework, pure Rust FFI).
  - `hf-hub`: removed (unused dependency that pulled in `aws-lc-sys`).
  - `candle-core`: vendored to `vendor/candle-core` with `[patch.crates-io]`; changed transitive `tokenizers` feature from `onig` (C) to `fancy-regex` (pure Rust).
  - `candle-nn`, `candle-transformers`: vendored with same tokenizers patch.
  - Removed `llama_cpp` (C++ llama.cpp bindings): rewrote GGUF backend to use candle-core's pure Rust `quantized::gguf_file` + candle-transformers' `quantized_llama::ModelWeights`.
  - Removed `mlx-rs` (C++ MLX framework): MLX backend replaced with error stub; Metal backend (pure Rust FFI to macOS system framework) provides equivalent GPU acceleration on Apple Silicon.
  - Removed `accelerate-src` (C BLAS), `candle-kernels` (CUDA C/C++), and corresponding features: `gguf`, `mlx`, `mlx-metal`, `accelerate`, `cuda`, `cudnn`, `nccl`.
  - Only remaining feature: `metal` (default) — pure Rust Metal FFI via `metal` crate.
  - Verified: `cargo tree --all-features` shows no `onig_sys`, no `aws-lc-sys`, no `cc`/`cmake`/`bindgen`/`clang-sys`/`link-cplusplus`. `esaxx-rs` remains but without `cpp` feature (pure Rust fallback).
- [x] **`cargo test` passes**: 227 tests total (159 lib + 12 API error + 33 E2E + 16 error handling + 7 integration), 0 failures. Fixed pre-existing test bugs (missing `config.json` in verification test fixtures, out-of-range `u16` literal, `bert-base-uncased` incorrectly in valid model names list). GGUF tests now always run (no longer feature-gated). Fixed `error_handling_test.rs` (27 compile errors from API drift), `api_error_test.rs` (missing `FromStr` import), `e2e_test.rs` (config show subcommand, TOML output assertions, error case matching). Fixed `config show` in `lib.rs` to use `ConfigManager` directly instead of requiring `init_config()`.
- [ ] Optional: split `src/local/generation.rs` (1,431 LOC) into `decode_loop`, `kv_cache`, `stop_tokens`, `sampling`.
- [ ] Optional: split `src/influencer/server.rs` (1,309 LOC) test fixtures out of the production source file.

## Brainstorming

- **OpenAI-compatible `/v1/chat/completions`**: Add a chat completions endpoint to match OpenAI client libraries more closely.
- **Model quantization**: Support on-the-fly quantization (e.g. Q4/Q8) for Candle models to reduce memory footprint.
- **Concurrent request handling**: Improve server throughput for multiple simultaneous inference requests (request batching, queue management).
- **Model pulling progress**: Add download progress bars and resume support for interrupted downloads.
- **GGUF tokenizer integration**: GGUF backend now uses candle's pure Rust quantized models with external tokenizer support.
- **Web UI**: Basic web dashboard for model management, generation testing, and server monitoring.
- **API key auth**: Optional token-based authentication for the HTTP server.
- **Model conversion**: Convert between formats (Safetensors ↔ GGUF) within the CLI.
- **Docker image**: Publish a container image with pre-built binary for easy deployment.

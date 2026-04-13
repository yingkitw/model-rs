# model-rs

Rust CLI and library for downloading Hugging Face models, running local inference (via [Candle](https://github.com/huggingface/candle)), and serving an HTTP API with OpenAI/Ollama-style endpoints.

## Requirements

- Rust toolchain with **edition 2024** support (recent stable).
- macOS: default build uses **Metal** (`metal` feature). Other platforms: use `--no-default-features` and enable `cuda` / CPU as needed (see `Cargo.toml` `[features]`).

## Quick start

```bash
cargo build --release
./target/release/model-rs --help
./target/release/model-rs download <org>/<model>
./target/release/model-rs list
```

Run the API server (set a model path or use config):

```bash
export MODEL_RS_MODEL_PATH=/path/to/model-dir
./target/release/model-rs serve
```

## Library

The package name on crates.io / in `Cargo.toml` is `model-rs`; in Rust code the library crate is imported as `model_rs`:

```rust
use model_rs::Result;

#[tokio::main]
async fn main() -> Result<()> {
    model_rs::run().await
}
```

Examples live under `examples/` (see `examples/README.md`).

## Configuration

Environment variables use the `MODEL_RS_` prefix (for example `MODEL_RS_MODEL_PATH`, `MODEL_RS_PORT`, `MODEL_RS_MIRROR`). Run `model-rs config` to see the full list resolved for your environment.

## Tests

- **Unit / integration in crate:** `cargo test`
- **API integration tests** (`tests/integration_test.rs`): hit `http://127.0.0.1:$MODEL_RS_PORT` (default **8080** if unset, matching `model-rs serve`).
- **End-to-end CLI/API** (`tests/e2e_test.rs`): see `tests/README.md`.

## License

Apache-2.0 (see `Cargo.toml`).

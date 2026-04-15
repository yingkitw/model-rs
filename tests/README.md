# model-rs tests

Integration and end-to-end tests for **model-rs**.

## Files

1. **`integration_test.rs`** — API checks against a running server at `http://127.0.0.1:$MODEL_RS_PORT` (default **8080** when the variable is unset).
2. **`e2e_test.rs`** — CLI smoke tests; several tests spawn `model-rs serve` themselves when a local model is available.

## Prerequisites (integration tests)

1. Build the binary and download a model if needed:

```bash
cargo build --release
./target/release/model-rs download TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

2. Start the server (adjust paths to your cache layout; `model-rs list` / `model-rs config` help locate models):

```bash
export MODEL_RS_MODEL_PATH=<path-to-model-directory>
export MODEL_RS_PORT=8080
./target/release/model-rs serve
```

3. Wait until the server is listening (for example log line showing the bound address).

## Running integration tests

From the repository root:

```bash
cargo test --test integration_test
cargo test --test integration_test test_v1_generate_endpoint
cargo test --test integration_test -- --nocapture
```

If the server is not running, many tests skip or no-op with a message instead of failing hard.

## Coverage (high level)

- **REST:** `POST /v1/generate`, streaming via `GET /v1/generate_stream` (SSE).
- **Ollama-style:** `POST /api/generate` (stream on/off), `GET /api/tags`.

## End-to-end tests (`e2e_test.rs`)

These resolve the `model-rs` executable via the runtime variable **`CARGO_BIN_EXE_model_rs`** (set by Cargo for integration tests), with a fallback path under `target/` next to the package manifest—so renaming or moving the repo directory does not require a manual path override.

**Model directory for tests that start a server:** set **`MODEL_RS_E2E_MODEL_PATH`** to an existing model folder, or install at least one model so `model-rs list` prints a `**Path:**` line. If neither is true, those tests print a hint and return without failing (so `cargo test` stays green on a fresh clone).

```bash
cargo build --release
cargo test --test e2e_test -- --test-threads=1
cargo test --test e2e_test test_cli -- --test-threads=1
cargo test --test e2e_test -- --nocapture
```

Optional ignored / heavy tests:

```bash
cargo test --test e2e_test --ignored -- --test-threads=1
```

Use `MODEL_RS_PORT` (and `MODEL_RS_MODEL_PATH` for workflow tests) to match your server.

## Troubleshooting

- **Connection refused** — Start `model-rs serve` with the same host/port the tests expect.
- **Port in use** — Set `MODEL_RS_PORT` to the same value for both `serve` and `cargo test --test integration_test`.
- **Timeouts** — Prefer a small/fast model or lower `max_tokens` in requests.

## CI example

```bash
./target/release/model-rs serve --model-path "$MODEL_PATH" &
SERVER_PID=$!
sleep 5
cargo test --test integration_test
kill "$SERVER_PID"
```

## Adding tests

- Prefer `#[tokio::test]` for async HTTP tests.
- On connection errors, treat `e.is_connect()` as “server not running” and skip or return early when appropriate.

## Notes

- Streaming tests may use long timeouts; outputs are validated structurally, not exact model text.

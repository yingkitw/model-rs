# TODO

Actionable follow-ups (keep this list honest and small).

- [x] Confirm `cargo test` passes (40 tests, 0 failures). `cargo build --release` is standard Rust; E2E tests resolve the CLI via runtime `CARGO_BIN_EXE_model_rs` so they stay valid if the repo path changes.
- [x] Integration tests use `MODEL_RS_PORT` with default **8080** (see `tests/integration_test.rs`).
- [ ] Optional: add a small `scripts/run_e2e_tests.sh` if you want a one-command local QA wrapper.
- [ ] Publish / version policy: decide on crates.io name vs GitHub repo naming (`model-rs`).
- [ ] Optional: make `deploy --detached` match its name (spawn/daemonize) or rename the flag now that README/SPEC describe current behavior.

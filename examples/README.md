# model-rs Rust examples

These examples live next to the `model-rs` crate and demonstrate **library** usage (`use model_rs::…`).

They are **not** required to use the CLI (`cargo run` / `model-rs …`).

## Build all examples

```bash
cargo build --examples
```

## Run

### `library_sampling`

Uses the public `do_sample` helper (no model weights required):

```bash
cargo run --example library_sampling
```

### `library_run_cli` (optional)

Forwards to the same entry as the binary (`model_rs::run()`), i.e. **parses real CLI args**:

```bash
cargo run --example library_run_cli -- --help
```

## Adding examples

- Place `*.rs` files in this directory.
- Cargo discovers them automatically; no `Cargo.toml` entry needed unless you need custom metadata.

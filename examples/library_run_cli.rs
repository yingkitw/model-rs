//! Same entry point as the `model-rs` binary: parses CLI args and runs [`model_rs::run`].
//!
//! Run from the repository root:
//! ```text
//! cargo run --example library_run_cli -- --help
//! ```

use model_rs::Result;

#[tokio::main]
async fn main() -> Result<()> {
    model_rs::run().await
}

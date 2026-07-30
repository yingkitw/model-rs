use crate::error::Result;

/// Trait for LLM text generation services.
///
/// This is an **extensibility seam**: it lets a future caller (test fixture,
/// mock, remote backend) implement generation against the same surface the
/// HTTP server exposes. `LocalModel` implements it today; the HTTP dispatch
/// path (`src/influencer/server.rs`) currently uses the concrete type.
///
/// **Status:** experimental. Not yet wired into `AppState`. Keep until a
/// second implementation appears or until `AppState` is switched to
/// `Box<dyn LlmService>`.
#[allow(async_fn_in_trait)]
pub trait LlmService {
    /// Generate text with the given prompt
    async fn generate_text(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String>;

    /// Generate text and stream output to stdout
    async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()>;
}

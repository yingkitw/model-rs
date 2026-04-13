use crate::error::Result;

/// Trait for LLM text generation services
///
/// This trait defines the interface for both local and remote LLM services,
/// allowing for interchangeable backends.
pub trait LlmService {
    /// Generate text with the given prompt
    async fn generate_text(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String>;

    /// Generate text and stream output to stdout
    async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()>;
}

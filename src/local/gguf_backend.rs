//! GGUF (quantized model) backend
//!
//! This module provides support for loading and running GGUF format models,
//! which offer significant memory savings through quantization.

use crate::error::{InfluenceError, Result};
use crate::local::LocalModelConfig;
use std::path::Path;
use tracing::info;
use tokenizers::Tokenizer;

/// GGUF backend for quantized model inference
#[cfg(feature = "gguf")]
pub struct GgufBackend {
    context_size: usize,
    quantization: String,
    gguf_path: std::path::PathBuf,
    model: Option<llama_cpp::LlamaModel>,
    #[allow(dead_code)]
    tokenizer: Option<Tokenizer>,
}

/// GGUF backend stub (when GGUF feature is not enabled)
#[cfg(not(feature = "gguf"))]
pub struct GgufBackend {
    _private: (),
}

#[cfg(feature = "gguf")]
impl GgufBackend {
    /// Load a GGUF model from the given path
    pub fn load(config: &LocalModelConfig, gguf_path: &Path) -> Result<Self> {
        info!("Loading GGUF model from: {}", gguf_path.display());

        // Detect quantization format from filename
        let quantization = Self::detect_quantization(gguf_path)?;
        info!("Detected quantization: {}", quantization);

        // Load tokenizer if available
        let tokenizer_path = config.model_path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            Some(Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| InfluenceError::LocalModelError(format!("Failed to load tokenizer: {}", e)))?)
        } else {
            info!("No tokenizer.json found, using model's internal tokenizer");
            None
        };

        // Load the GGUF model using llama_cpp
        let params = llama_cpp::LlamaModelParams {
            n_ctx: config.max_seq_len as u32,
            ..Default::default()
        };

        let model = llama_cpp::LlamaModel::load_from_file(gguf_path, params)
            .map_err(|e| InfluenceError::GgufError(format!("Failed to load GGUF model: {}", e)))?;

        info!("GGUF model loaded successfully (quantization: {})", quantization);

        Ok(Self {
            gguf_path: gguf_path.to_path_buf(),
            context_size: config.max_seq_len,
            quantization,
            model: Some(model),
            tokenizer,
        })
    }

    /// Detect quantization format from GGUF filename
    fn detect_quantization(path: &Path) -> Result<String> {
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| InfluenceError::GgufParsingError("Invalid filename".to_string()))?;

        let filename_lower = filename.to_lowercase();

        let quant = if filename_lower.contains("q2_k") {
            "Q2_K"
        } else if filename_lower.contains("q4_k_m") {
            "Q4_K_M"
        } else if filename_lower.contains("q4_k") {
            "Q4_K"
        } else if filename_lower.contains("q5_k_m") {
            "Q5_K_M"
        } else if filename_lower.contains("q5_k") {
            "Q5_K"
        } else if filename_lower.contains("q6_k") {
            "Q6_K"
        } else if filename_lower.contains("q8_0") {
            "Q8_0"
        } else if filename_lower.contains("f16") {
            "F16"
        } else {
            "Unknown"
        };

        Ok(quant.to_string())
    }

    /// Generate text from a prompt
    pub fn generate_text(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        _eos_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        let model = self.model.as_ref()
            .ok_or_else(|| InfluenceError::LocalModelError("GGUF model not loaded".to_string()))?;

        // Create session for generation
        let mut session = llama_cpp::LlamaSession::new(model);

        // Generate tokens
        let params = llama_cpp::LlamaPredictParams {
            n_predict: max_tokens as u32,
            temperature,
            top_p,
            top_k: top_k.unwrap_or(0) as i32,
            ..Default::default()
        };

        let mut output_tokens = Vec::new();
        let mut callback = |token: u32| {
            output_tokens.push(token);
            // Stop if we hit EOS (token_id 0 is usually EOS in llama.cpp)
            if token == 0 {
                false // Stop generation
            } else {
                true // Continue
            }
        };

        session.advance(prompt, params, Some(&mut callback))
            .map_err(|e| InfluenceError::GgufError(format!("GGUF generation failed: {}", e)))?;

        Ok(output_tokens)
    }

    /// Generate text with streaming callback
    pub fn generate_text_stream<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        let model = self.model.as_ref()
            .ok_or_else(|| InfluenceError::LocalModelError("GGUF model not loaded".to_string()))?;

        // Create session for generation
        let mut session = llama_cpp::LlamaSession::new(model);

        let params = llama_cpp::LlamaPredictParams {
            n_predict: max_tokens as u32,
            temperature,
            top_p,
            top_k: top_k.unwrap_or(0) as i32,
            ..Default::default()
        };

        // Streaming callback that decodes tokens
        let mut token_callback = |token: u32| {
            if token == 0 {
                false // Stop on EOS
            } else {
                // Decode single token to string
                if let Some(tokenizer) = &self.tokenizer {
                    let decoded = tokenizer.decode(&[token], false)
                        .unwrap_or_else(|_| format!("<token_{}>", token));
                    let _ = callback(decoded);
                } else {
                    // Fallback: just emit a placeholder
                    let _ = callback(format!("<token_{}>", token));
                }
                true // Continue
            }
        };

        session.advance(prompt, params, Some(&mut token_callback))
            .map_err(|e| InfluenceError::GgufError(format!("GGUF streaming generation failed: {}", e)))?;

        Ok(())
    }

    /// Generate embeddings for text
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.as_ref()
            .ok_or_else(|| InfluenceError::LocalModelError("GGUF model not loaded".to_string()))?;

        // Use llama.cpp embedding functionality
        let embeddings = model.embed_text(text)
            .map_err(|e| InfluenceError::GgufError(format!("GGUF embedding failed: {}", e)))?;

        Ok(embeddings)
    }

    /// Get the quantization format
    pub fn quantization(&self) -> &str {
        &self.quantization
    }

    /// Get the context size
    pub fn context_size(&self) -> usize {
        self.context_size
    }

    /// Get the GGUF file path
    pub fn path(&self) -> &Path {
        &self.gguf_path
    }
}

#[cfg(not(feature = "gguf"))]
impl GgufBackend {
    /// Load a GGUF model (stub when feature is not enabled)
    pub fn load(_config: &LocalModelConfig, _gguf_path: &Path) -> Result<Self> {
        Err(InfluenceError::InvalidConfig(
            "GGUF support not enabled. Build with --features gguf".to_string()
        ))
    }

    /// Get the quantization format (stub)
    pub fn quantization(&self) -> &str {
        "N/A"
    }

    /// Get the context size (stub)
    pub fn context_size(&self) -> usize {
        0
    }

    /// Get the GGUF file path (stub)
    pub fn path(&self) -> &Path {
        Path::new("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gguf")]
    fn test_detect_quantization() {
        // Test all supported quantization formats
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q2_k.gguf")).unwrap(),
            "Q2_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q4_k.gguf")).unwrap(),
            "Q4_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q4_k_m.gguf")).unwrap(),
            "Q4_K_M"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q5_k.gguf")).unwrap(),
            "Q5_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q5_k_m.gguf")).unwrap(),
            "Q5_K_M"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q6_k.gguf")).unwrap(),
            "Q6_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-q8_0.gguf")).unwrap(),
            "Q8_0"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model-f16.gguf")).unwrap(),
            "F16"
        );
        // Test case insensitivity
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("MODEL-Q2_K.GGUF")).unwrap(),
            "Q2_K"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("Model-Q4_K_M.GgUF")).unwrap(),
            "Q4_K_M"
        );
    }

    #[test]
    #[cfg(feature = "gguf")]
    fn test_detect_quantization_unknown() {
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model.gguf")).unwrap(),
            "Unknown"
        );
        assert_eq!(
            GgufBackend::detect_quantization(Path::new("model.bin")).unwrap(),
            "Unknown"
        );
    }

    #[test]
    #[cfg(feature = "gguf")]
    fn test_detect_quantization_invalid_path() {
        assert!(GgufBackend::detect_quantization(Path::new("")).is_err());
        assert!(GgufBackend::detect_quantization(Path::new("/")).is_err());
    }

    #[test]
    #[cfg(not(feature = "gguf"))]
    fn test_gguf_disabled() {
        let config = LocalModelConfig::default();
        let result = GgufBackend::load(&config, Path::new("test.gguf"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("GGUF support not enabled"));
    }
}

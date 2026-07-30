//! GGUF (quantized model) backend using candle's pure Rust quantized support
//!
//! This module provides support for loading and running GGUF format models,
//! which offer significant memory savings through quantization.
//! Uses candle-core's quantized GGUF reader and candle-transformers' quantized
//! model implementations — no C/C++ dependencies.

use crate::error::{ModelError, Result};
use crate::local::LocalModelConfig;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlama;
use std::path::Path;
use tracing::info;

/// GGUF backend for quantized model inference (pure Rust)
pub struct GgufBackend {
    context_size: usize,
    quantization: String,
    gguf_path: std::path::PathBuf,
    model: QuantizedLlama,
    device: Device,
}

impl GgufBackend {
    /// Load a GGUF model from the given path
    pub fn load(config: &LocalModelConfig, gguf_path: &Path) -> Result<Self> {
        info!("Loading GGUF model from: {}", gguf_path.display());

        let quantization = Self::detect_quantization(gguf_path)?;
        info!("Detected quantization: {}", quantization);

        let device = Device::Cpu;

        let file = std::fs::File::open(gguf_path)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to open GGUF file: {}", e)))?;
        let mut reader = std::io::BufReader::new(file);

        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse GGUF file: {}", e)))?;

        let model = QuantizedLlama::from_gguf(content, &mut reader, &device)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to load quantized model: {}", e)))?;

        info!("GGUF model loaded successfully (quantization: {})", quantization);

        Ok(Self {
            gguf_path: gguf_path.to_path_buf(),
            context_size: config.max_seq_len,
            quantization,
            model,
            device,
        })
    }

    /// Check if a directory contains any .gguf files
    pub fn detect_gguf(model_path: &Path) -> Result<Option<std::path::PathBuf>> {
        if !model_path.is_dir() {
            return Ok(None);
        }
        let entries = std::fs::read_dir(model_path)
            .map_err(|e| ModelError::LocalModelError(format!("Cannot read directory: {}", e)))?;
        for entry in entries.flatten() {
            if entry.path().extension().map_or(false, |ext| ext == "gguf") {
                return Ok(Some(entry.path()));
            }
        }
        Ok(None)
    }

    /// Detect quantization format from GGUF filename
    pub fn detect_quantization(path: &Path) -> Result<String> {
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| ModelError::LocalModelError("Invalid filename".to_string()))?;

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

    /// Generate text from input token IDs
    pub fn generate_text(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        eos_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();

        let prompt_tensor = Tensor::new(input_ids, &self.device)?
            .unsqueeze(0)?;
        let logits = self.model.forward(&prompt_tensor, 0)?;
        let logits = logits.to_dtype(candle_core::DType::F32)?;
        let logits_vec = logits.to_vec3::<f32>()?;
        let last_logits = &logits_vec[0][logits_vec[0].len() - 1];

        let mut next = sample_token(last_logits, temperature, top_p, top_k)?;
        generated.push(next);

        for idx in 1..max_tokens {
            if let Some(eos) = eos_token {
                if next == eos {
                    break;
                }
            }

            let tensor = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&tensor, input_ids.len() + idx - 1)?;
            let logits = logits.to_dtype(candle_core::DType::F32)?;
            let logits_vec = logits.to_vec3::<f32>()?;
            let last_logits = &logits_vec[0][0];

            next = sample_token(last_logits, temperature, top_p, top_k)?;
            generated.push(next);
        }

        Ok(generated)
    }

    /// Generate text with streaming callback
    pub fn generate_text_stream<F>(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        eos_token: Option<u32>,
        mut callback: F,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        use crate::local::tokenization::stream_piece;

        let mut started = false;

        let prompt_tensor = Tensor::new(input_ids, &self.device)?
            .unsqueeze(0)?;
        let logits = self.model.forward(&prompt_tensor, 0)?;
        let logits = logits.to_dtype(candle_core::DType::F32)?;
        let logits_vec = logits.to_vec3::<f32>()?;
        let last_logits = &logits_vec[0][logits_vec[0].len() - 1];

        let mut next = sample_token(last_logits, temperature, top_p, top_k)?;

        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            callback(piece)?;
        }

        for idx in 1..max_tokens {
            if let Some(eos) = eos_token {
                if next == eos {
                    break;
                }
            }

            let tensor = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&tensor, input_ids.len() + idx - 1)?;
            let logits = logits.to_dtype(candle_core::DType::F32)?;
            let logits_vec = logits.to_vec3::<f32>()?;
            let last_logits = &logits_vec[0][0];

            next = sample_token(last_logits, temperature, top_p, top_k)?;

            if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
                callback(piece)?;
            }
        }

        Ok(())
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

/// Simple sampling: temperature + top-p + top-k
fn sample_token(logits: &[f32], temperature: f32, top_p: f32, top_k: Option<usize>) -> Result<u32> {
    crate::local::sampling::do_sample(logits, temperature, top_p, top_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_quantization() {
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
    fn test_detect_quantization_invalid_path() {
        assert!(GgufBackend::detect_quantization(Path::new("")).is_err());
        assert!(GgufBackend::detect_quantization(Path::new("/")).is_err());
    }
}

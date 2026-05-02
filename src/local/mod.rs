//! Local LLM inference module
//!
//! This module provides functionality for loading and running local LLM models
//! with support for multiple architectures including Llama, Mistral, Mamba, and more.

use crate::error::{ModelError, Result};
use tokenizers::Tokenizer;
use std::path::Path;
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::{debug, info, warn};
use candle_core::Tensor;
use std::io::Write;

// Public sub-modules
mod backends;
mod device;
mod config;
mod sampling;
mod architecture;
mod tokenization;
mod generation;
mod model_cache;
mod batch;

#[cfg(feature = "gguf")]
mod gguf_backend;

#[cfg(feature = "gguf")]
mod cache;

#[cfg(feature = "mlx")]
mod mlx_backend;

// Public exports
pub use backends::LocalBackend;
pub use device::get_device;
pub use config::{ModelArchitecture, DevicePreference, LocalModelConfig};
pub use sampling::do_sample;
pub use architecture::detect_architecture;
pub use tokenization::{get_eos_token, stream_piece};
pub use model_cache::{global_model_cache, get_or_load_model, get_cached_model, ModelCache};
pub use batch::{BatchRequest, BatchResult, generate_batch, generate_batch_stream};

/// Local model runner with support for multiple architectures
pub struct LocalModel {
    config: LocalModelConfig,
    tokenizer: Tokenizer,
    backend: Option<LocalBackend>,
    device: candle_core::Device,
    /// Cache of tokenized prompts/text to reduce repeat tokenization cost.
    tokenization_cache: Mutex<HashMap<String, Vec<u32>>>,
    tokenization_cache_max: usize,
    /// Optional session KV cache reuse (interactive chat only).
    ///
    /// When enabled, we reuse the Llama KV cache across consecutive turns
    /// as long as the new prompt token prefix matches the cached sequence.
    session_kv_enabled: bool,
    llama_session_cache: Option<candle_transformers::models::llama::Cache>,
    llama_session_cache_input_ids: Vec<u32>,
}

impl LocalModel {
    /// Get a reference to the model configuration
    pub fn config(&self) -> &LocalModelConfig {
        &self.config
    }

    /// Get a mutable reference to the model configuration
    pub fn config_mut(&mut self) -> &mut LocalModelConfig {
        &mut self.config
    }

    /// Load a model from the given configuration
    pub async fn load(mut config: LocalModelConfig) -> Result<Self> {
        info!("Loading local model from: {}", config.model_path.display());

        if !config.model_path.exists() {
            return Err(ModelError::ModelNotFound(format!(
                "Model directory not found: {}\n\nHint: Use 'model-rs download <model>' to download a model first.\nAvailable models: 'model-rs list' to list downloaded models.",
                config.model_path.display()
            )));
        }

        let tokenizer = Self::load_tokenizer(&config.model_path)?;
        let architecture = detect_architecture(&config.model_path)?;
        config.architecture = architecture;

        info!("Detected architecture: {:?}", architecture);

        let device = get_device(config.device_preference, config.device_index)?;
        info!("Using device: {:?}", device);

        let backend = if config.device_preference == DevicePreference::Mlx {
            #[cfg(feature = "mlx")]
            {
                LocalBackend::load_mlx(&config, &device)?
            }
            #[cfg(not(feature = "mlx"))]
            {
                return Err(ModelError::InvalidConfig(
                    "MLX support not enabled. Build with --features mlx".to_string(),
                ));
            }
        } else {
            match architecture {
                ModelArchitecture::Llama => {
                    LocalBackend::load_llama(&config, &device)?
                }
                ModelArchitecture::LlamaQuantized => {
                    #[cfg(feature = "gguf")]
                    {
                        if let Some(gguf_backend) = LocalBackend::load_gguf(&config, &device)? {
                            Some(gguf_backend)
                        } else {
                            LocalBackend::load_llama(&config, &device)?
                        }
                    }
                    #[cfg(not(feature = "gguf"))]
                    {
                        LocalBackend::load_llama(&config, &device)?
                    }
                }
                ModelArchitecture::Mistral => LocalBackend::load_mistral(&config, &device)?,
                ModelArchitecture::Mamba => LocalBackend::load_mamba(&config, &device)?,
                ModelArchitecture::GraniteMoeHybrid => LocalBackend::load_granite_moe_hybrid(&config, &device)?,
                ModelArchitecture::Bert => LocalBackend::load_bert(&config, &device)?,
                ModelArchitecture::Phi => LocalBackend::load_phi3(&config, &device)?,
                ModelArchitecture::Gemma => LocalBackend::load_gemma(&config, &device)?,
                ModelArchitecture::Qwen2 => LocalBackend::load_qwen2(&config, &device)?,
                ModelArchitecture::Qwen3 => LocalBackend::load_qwen3(&config, &device)?,
                ModelArchitecture::DeepSeek2 => LocalBackend::load_deepseek2(&config, &device)?,
                ModelArchitecture::Glm4 => LocalBackend::load_glm4(&config, &device)?,
                _ => {
                    warn!("Architecture {:?} not yet fully implemented", architecture);
                    None
                }
            }
        };

        if backend.is_some() {
            info!("Model loaded successfully with full inference capability!");
        } else {
            info!("Model structure loaded (placeholder mode - no .safetensors files found)");
        }

        Ok(Self {
            config,
            tokenizer,
            backend,
            device,
            tokenization_cache: Mutex::new(HashMap::new()),
            tokenization_cache_max: 256,
            session_kv_enabled: false,
            llama_session_cache: None,
            llama_session_cache_input_ids: Vec::new(),
        })
    }

    /// Enable session-based KV cache reuse for interactive chat.
    pub fn enable_session_kv_cache(&mut self) {
        self.session_kv_enabled = true;
        self.clear_session_kv_cache();
    }

    /// Clear any session KV cache state.
    pub fn clear_session_kv_cache(&mut self) {
        self.llama_session_cache = None;
        self.llama_session_cache_input_ids.clear();
    }

    /// Tokenize `text` into token IDs, using a small in-memory cache.
    fn encode_ids_cached(&self, text: &str) -> Result<Vec<u32>> {
        // Fast path: cache hit.
        if let Some(ids) = self.tokenization_cache.lock().unwrap().get(text).cloned() {
            return Ok(ids);
        }

        let tokens = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| ModelError::LocalModelError(format!("Tokenization failed: {}", e)))?;
        let ids = tokens.get_ids().to_vec();

        // Simple bounded cache: when full, clear it.
        // This keeps the implementation small and avoids extra dependencies.
        let mut cache = self.tokenization_cache.lock().unwrap();
        if cache.len() >= self.tokenization_cache_max {
            cache.clear();
        }
        cache.insert(text.to_string(), ids.clone());
        Ok(ids)
    }

    fn decode_tokens(&self, token_ids: &[u32]) -> Result<String> {
        let mut result = String::new();
        let mut skipped_special = 0;

        for (i, &token_id) in token_ids.iter().enumerate() {
            let raw_token = self.tokenizer.id_to_token(token_id);

            if let Some(ref token) = raw_token {
                if token == "</s>" || token == "<s>" || token == "<unk>" {
                    skipped_special += 1;
                    continue;
                }
            }

            let token_str = self.tokenizer.decode(&[token_id], false)
                .map_err(|e| ModelError::LocalModelError(format!("Token decode failed: {}", e)))?;

            if let Some(raw) = raw_token {
                if raw.starts_with('▁') {
                    let actual_index = i - skipped_special;
                    if actual_index > 0 && !result.is_empty() {
                        result.push(' ');
                    }
                }
            }

            result.push_str(&token_str);
        }

        Ok(result.trim().to_string())
    }

    fn load_tokenizer(model_path: &Path) -> Result<Tokenizer> {
        let tokenizer_files = ["tokenizer.json", "tokenizer_config.json"];
        let tokenizer_path = tokenizer_files.iter()
            .find_map(|file| {
                let path = model_path.join(file);
                if path.exists() { Some(path) } else { None }
            })
            .ok_or_else(|| ModelError::InvalidConfig(
                "Tokenizer file not found".to_string()
            ))?;

        Tokenizer::from_file(tokenizer_path)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to load tokenizer: {}", e)))
    }

    /// Generate text from a prompt (non-streaming)
    pub async fn generate_text(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        info!("Generating (max_tokens={}, temp={})", max_tokens, temperature);

        let input_ids = self.encode_ids_cached(prompt)?;

        let eos_token = get_eos_token(&self.tokenizer);

        let top_p = self.config.top_p;
        let top_k = self.config.top_k;

        let backend = self.backend.as_mut().ok_or_else(|| {
            ModelError::LocalModelError(format!(
                "Model not loaded. Ensure .safetensors files are in: {}",
                self.config.model_path.display()
            ))
        })?;

        let generated = match backend {
            LocalBackend::Llama { model, config } if self.session_kv_enabled => {
                LocalModel::generate_llama_with_session_kv(
                    model,
                    config,
                    &input_ids,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    eos_token,
                    &self.device,
                    &mut self.llama_session_cache,
                    &mut self.llama_session_cache_input_ids,
                )?
            }
            _ => generation::generate_text(
                backend,
                &input_ids,
                max_tokens,
                temperature,
                top_p,
                top_k,
                eos_token,
                &self.device,
                do_sample,
            )?,
        };

        self.decode_tokens(&generated)
    }

    fn generate_llama_with_session_kv(
        model: &candle_transformers::models::llama::Llama,
        config: &candle_transformers::models::llama::Config,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        eos_token: Option<u32>,
        device: &candle_core::Device,
        session_cache: &mut Option<candle_transformers::models::llama::Cache>,
        session_cache_input_ids: &mut Vec<u32>,
    ) -> Result<generation::GenerationResult>
    {
        use candle_core::DType;
        use candle_transformers::models::llama::Cache;

        let prompt_len = input_ids.len();
        let cached_seq = std::mem::take(session_cache_input_ids);
        let cached_len = cached_seq.len();

        // Reuse the cache if it matches the prompt token prefix.
        let mut cache = if let Some(cache) = session_cache.take() {
            cache
        } else {
            Cache::new(true, DType::F32, config, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to create cache: {}", e))
            })?
        };

        let mut generated: Vec<u32>;

        if cached_len > 0 && input_ids.starts_with(&cached_seq[..]) {
            let delta = &input_ids[cached_len..];
            if !delta.is_empty() {
                let delta_tensor = Tensor::new(delta, device)?.unsqueeze(0)?;
                let logits = model.forward(&delta_tensor, cached_len, &mut cache)?;
                let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                let last_logits = &logits_vec[0];

                let mut next = do_sample(last_logits, temperature, top_p, top_k)?;
                generated = vec![next];

                for idx in 1..max_tokens {
                    if let Some(eos) = eos_token {
                        if next == eos {
                            break;
                        }
                    }

                    let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
                    let logits = model.forward(&tensor, prompt_len + idx - 1, &mut cache)?;
                    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    let token_logits = &logits_vec[0];

                    next = do_sample(token_logits, temperature, top_p, top_k)?;
                    generated.push(next);
                }
            } else {
                // Prefix matches but no new tokens to extend; fall back to fresh decode.
                cache = Cache::new(true, DType::F32, config, device)?;
                let prompt_tensor = Tensor::new(input_ids, device)?.unsqueeze(0)?;
                let logits = model.forward(&prompt_tensor, 0, &mut cache)?;
                let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                let last_logits = &logits_vec[0];
                let mut next = do_sample(last_logits, temperature, top_p, top_k)?;
                generated = vec![next];
                for idx in 1..max_tokens {
                    if let Some(eos) = eos_token {
                        if next == eos {
                            break;
                        }
                    }
                    let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
                    let logits = model.forward(&tensor, prompt_len + idx - 1, &mut cache)?;
                    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    let token_logits = &logits_vec[0];
                    next = do_sample(token_logits, temperature, top_p, top_k)?;
                    generated.push(next);
                }
            }
        } else {
            // Cache mismatch: start a fresh prefill.
            cache = Cache::new(true, DType::F32, config, device)
                .map_err(|e| ModelError::LocalModelError(format!("Failed to create cache: {}", e)))?;

            let prompt_tensor = Tensor::new(input_ids, device)?.unsqueeze(0)?;
            let logits = model.forward(&prompt_tensor, 0, &mut cache)?;
            let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let last_logits = &logits_vec[0];

            let mut next = do_sample(last_logits, temperature, top_p, top_k)?;
            generated = vec![next];

            for idx in 1..max_tokens {
                if let Some(eos) = eos_token {
                    if next == eos {
                        break;
                    }
                }

                let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
                let logits = model.forward(&tensor, prompt_len + idx - 1, &mut cache)?;
                let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                let token_logits = &logits_vec[0];

                next = do_sample(token_logits, temperature, top_p, top_k)?;
                generated.push(next);
            }
        }

        // Update session KV cache state so the next turn can reuse it.
        let mut new_seq = input_ids.to_vec();
        new_seq.extend(&generated);
        *session_cache_input_ids = new_seq;
        *session_cache = Some(cache);

        Ok(generated)
    }

    /// Generate text with streaming callback
    pub async fn generate_stream_with<F>(&mut self, prompt: &str, max_tokens: usize, temp: f32, emit: F) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        let input_ids = self.encode_ids_cached(prompt)?;

        debug!(
            "Generation timing: tokenization_ms={}, prompt_tokens={}",
            std::time::Instant::now().elapsed().as_millis(),
            input_ids.len()
        );

        let eos_token = get_eos_token(&self.tokenizer);

        let top_p = self.config.top_p;
        let top_k = self.config.top_k;

        let backend = self.backend.as_mut().ok_or_else(|| ModelError::LocalModelError(
            format!("Model not loaded. Ensure .safetensors files are in: {}", self.config.model_path.display())
        ))?;

        generation::generate_text_stream(
            backend,
            &self.tokenizer,
            &input_ids,
            max_tokens,
            temp,
            top_p,
            top_k,
            eos_token,
            &self.device,
            do_sample,
            emit,
        )
    }

    /// Generate text with stdout streaming
    pub async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()> {
        self.generate_stream_with(prompt, max_tokens, temperature, |piece| {
            print!("{}", piece);
            std::io::stdout().flush()?;
            Ok(())
        }).await
    }

    /// Generate embeddings for text (encoder-only models only)
    pub async fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let input_ids = self.encode_ids_cached(text)?;

        let backend = self.backend.as_ref().ok_or_else(|| ModelError::LocalModelError(
            format!("Model not loaded. Ensure .safetensors files are in: {}", self.config.model_path.display())
        ))?;

        match backend {
            LocalBackend::Bert { model } => {
                let seq_len = input_ids.len();
                if seq_len == 0 {
                    return Err(ModelError::LocalModelError("Empty input".to_string()));
                }

                let input = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
                let token_type = Tensor::zeros((1, seq_len), candle_core::DType::U32, &self.device)?;
                let output = model.forward(&input, &token_type, None)?;
                let pooled = output.mean(1)?; // [1, hidden]
                let pooled = pooled.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
                Ok(pooled[0].clone())
            }
            _ => Err(ModelError::LocalModelError(
                "Embeddings are only supported for encoder-only BERT models (including RoBERTa/ALBERT)".to_string(),
            )),
        }
    }

    /// Generate text for multiple prompts in batch
    ///
    /// This method processes multiple prompts together, providing improved throughput
    /// compared to sequential generation.
    ///
    /// # Arguments
    /// * `prompts` - List of prompt texts to generate from
    /// * `max_tokens` - Maximum tokens to generate per prompt
    /// * `temperature` - Temperature for sampling
    ///
    /// # Returns
    /// Vector of generated texts, one per input prompt
    ///
    /// # Example
    /// ```text
    /// // Requires a loaded `LocalModel` and an async context:
    /// let prompts = vec!["Hello", "How are you?", "Tell me a joke"];
    /// let results = model.generate_batch(prompts, 100, 0.7).await?;
    /// for (i, text) in results.iter().enumerate() {
    ///     println!("Prompt {}: {}", i, text);
    /// }
    /// ```
    pub async fn generate_batch(&mut self, prompts: Vec<&str>, max_tokens: usize, temperature: f32) -> Result<Vec<String>> {
        info!("Generating batch of {} prompts (max_tokens={}, temp={})", prompts.len(), max_tokens, temperature);

        let eos_token = get_eos_token(&self.tokenizer);
        let _top_p = self.config.top_p;
        let _top_k = self.config.top_k;

        let mut batch_requests = Vec::new();
        for prompt in prompts {
            let input_ids = self.encode_ids_cached(prompt)?;

            batch_requests.push(BatchRequest {
                prompt: prompt.to_string(),
                input_ids,
                max_tokens,
                temperature,
                eos_token,
            });
        }

        let backend = self.backend.as_mut().ok_or_else(|| ModelError::LocalModelError(
            format!("Model not loaded. Ensure .safetensors files are in: {}", self.config.model_path.display())
        ))?;

        let batch_results = generate_batch(backend, batch_requests, &self.device, do_sample)?;

        let mut results = Vec::new();
        for batch_result in batch_results {
            results.push(self.decode_tokens(&batch_result.tokens)?);
        }

        Ok(results)
    }

    /// Generate embeddings for multiple texts in batch
    ///
    /// This processes multiple texts through the encoder model at once,
    /// providing better throughput for embeddings workloads.
    ///
    /// # Arguments
    /// * `texts` - List of texts to embed
    ///
    /// # Returns
    /// Vector of embeddings, one per input text
    pub async fn embed_batch(&mut self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        info!("Generating embeddings for {} texts", texts.len());

        let backend = self.backend.as_ref().ok_or_else(|| ModelError::LocalModelError(
            format!("Model not loaded. Ensure .safetensors files are in: {}", self.config.model_path.display())
        ))?;

        match backend {
            LocalBackend::Bert { model } => {
                let mut embeddings = Vec::new();

                for text in texts {
                    let input_ids = self.encode_ids_cached(text)?;
                    let seq_len = input_ids.len();

                    if seq_len == 0 {
                        return Err(ModelError::LocalModelError("Empty input".to_string()));
                    }

                    let input = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
                    let token_type = Tensor::zeros((1, seq_len), candle_core::DType::U32, &self.device)?;
                    let output = model.forward(&input, &token_type, None)?;
                    let pooled = output.mean(1)?;
                    let pooled = pooled.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
                    embeddings.push(pooled[0].clone());
                }

                Ok(embeddings)
            }
            _ => Err(ModelError::LocalModelError(
                "Embeddings are only supported for encoder-only BERT models (including RoBERTa/ALBERT)".to_string(),
            )),
        }
    }
}

/// Implement the LlmService trait for LocalModel (used by the HTTP API in `influencer`)
impl crate::influencer::LlmService for LocalModel {
    async fn generate_text(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        self.generate_text(prompt, max_tokens, temperature).await
    }

    async fn generate_stream(&mut self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()> {
        self.generate_stream(prompt, max_tokens, temperature).await
    }
}

/// Load a model from a path using default configuration
pub async fn load_model_from_path(path: &Path) -> Result<LocalModel> {
    LocalModel::load(LocalModelConfig {
        model_path: path.to_path_buf(),
        ..Default::default()
    }).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_architecture() {
        assert!(matches!(ModelArchitecture::Llama, ModelArchitecture::Llama));
    }

    #[tokio::test]
    async fn test_config_default() {
        let cfg = LocalModelConfig::default();
        assert_eq!(cfg.max_seq_len, 4096);
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_p, 0.9);
        assert_eq!(cfg.top_k, None);
        assert_eq!(cfg.repeat_penalty, 1.1);
    }

    #[test]
    fn test_do_sample_with_temperature() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = do_sample(&logits, 1.0, 1.0, None);
        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(token < 5);

        let result = do_sample(&logits, 0.5, 1.0, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_do_sample_with_top_k() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.1, 0.2];
        let result = do_sample(&logits, 1.0, 1.0, Some(3));
        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(token >= 2 && token <= 4);
    }

    #[test]
    fn test_do_sample_with_top_p() {
        let logits = vec![0.1, 0.2, 0.3, 4.0, 5.0];
        let result = do_sample(&logits, 1.0, 0.9, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_do_sample_with_zero_temperature() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = do_sample(&logits, 0.0, 1.0, None);
        assert!(result.is_ok());
        let token = result.unwrap();
        assert_eq!(token, 4);
    }

    #[test]
    fn test_do_sample_single_token() {
        let logits = vec![5.0];
        let result = do_sample(&logits, 1.0, 1.0, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_do_sample_with_both_top_k_and_top_p() {
        let logits = vec![0.1, 0.2, 0.3, 4.0, 5.0, 0.5, 0.6, 0.7];
        let result = do_sample(&logits, 1.0, 0.8, Some(4));
        assert!(result.is_ok());
        let token = result.unwrap();
        assert!(token < 8);
    }

    #[test]
    fn test_config_getters() {
        let config = LocalModelConfig::default();
        assert_eq!(config.model_path, PathBuf::from("models"));
        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.repeat_penalty, 1.1);
    }

    #[test]
    fn test_detect_architecture_mamba() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"mamba"}"#,
        ).unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::Mamba));
    }

    #[test]
    fn test_detect_architecture_mistral() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"mistral"}"#,
        )
        .unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::Mistral));
    }

    #[test]
    fn test_detect_architecture_roberta_maps_to_bert() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"roberta"}"#,
        )
        .unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::Bert));
    }

    #[test]
    fn test_detect_architecture_phi() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"phi"}"#,
        )
        .unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::Phi));
    }

    #[test]
    fn test_detect_architecture_granite_moe_hybrid_attention_only() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"granitemoehybrid","layer_types":["attention","attention"]}"#,
        ).unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::GraniteMoeHybrid));
    }

    #[test]
    fn test_detect_architecture_granite_moe_hybrid_with_mamba_layer_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"granitemoehybrid","layer_types":["attention","mamba"]}"#,
        ).unwrap();
        let err = detect_architecture(tmp.path()).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("mamba"));
    }

    #[test]
    fn test_detect_architecture_moe_rejected() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"llama","num_experts":8}"#,
        ).unwrap();
        let err = detect_architecture(tmp.path()).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("moe"));
    }

    #[test]
    fn test_detect_architecture_qwen3() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"qwen3"}"#,
        ).unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::Qwen3));
    }

    #[test]
    fn test_detect_architecture_deepseek() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"deepseek_v2"}"#,
        ).unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::DeepSeek2));
    }

    #[test]
    fn test_detect_architecture_kimi() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"kimi"}"#,
        ).unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::DeepSeek2));
    }

    #[test]
    fn test_detect_architecture_glm4() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"glm4"}"#,
        ).unwrap();
        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::Glm4));
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn test_detect_architecture_gguf_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("model-q4_k_m.gguf"),
            b"fake gguf content",
        ).unwrap();

        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::LlamaQuantized));
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn test_detect_architecture_gguf_prioritizes_over_config() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("model.gguf"),
            b"fake gguf content",
        ).unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"llama"}"#,
        ).unwrap();

        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::LlamaQuantized));
    }

    #[cfg(feature = "gguf")]
    #[test]
    fn test_detect_architecture_multiple_gguf_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("model-q4_k_m.gguf"),
            b"fake gguf content",
        ).unwrap();
        std::fs::write(
            tmp.path().join("model-q8_0.gguf"),
            b"fake gguf content",
        ).unwrap();

        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::LlamaQuantized));
    }

    #[cfg(not(feature = "gguf"))]
    #[test]
    fn test_detect_architecture_ignores_gguf_without_feature() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join("model-q4_k_m.gguf"),
            b"fake gguf content",
        ).unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"model_type":"llama"}"#,
        ).unwrap();

        let arch = detect_architecture(tmp.path()).unwrap();
        assert!(matches!(arch, ModelArchitecture::Llama));
    }
}

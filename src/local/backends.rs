//! Model backend implementations
//!
//! This module handles loading different model architectures from disk.
//! Each architecture has a corresponding loader function that reads
//! the model weights and configuration from the model directory.

use crate::error::{ModelError, Result};
use crate::local::LocalModelConfig;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::gemma3::{Config as GemmaConfig, Model as GemmaModel};
use candle_transformers::models::granitemoehybrid::{
    GraniteMoeHybrid, GraniteMoeHybridConfig, GraniteMoeHybridInternalConfig,
};
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama};
use candle_transformers::models::mamba::{Config as MambaConfig, Model as MambaModel};
use candle_transformers::models::mistral::{Config as MistralConfig, Model as MistralModel};
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2Model};
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3Model};
use candle_transformers::models::deepseek2::{DeepSeekV2Config, DeepSeekV2};
use candle_transformers::models::glm4_new::{Config as Glm4Config, ModelForCausalLM as Glm4Model};
use std::fs;
use std::path::Path;
use std::time::Instant;
use tracing::{info, warn};

/// Local model backend supporting multiple architectures
///
/// Each variant contains the loaded model and its configuration.
/// Generation is handled by the `generation` module.
pub enum LocalBackend {
    /// Llama/Llama2/Llama3 model with cache configuration
    Llama { model: Llama, config: LlamaConfig },

    /// Mistral model with configuration
    Mistral {
        model: MistralModel,
        config: MistralConfig,
    },

    /// Mamba state-space model with configuration
    Mamba {
        model: MambaModel,
        config: MambaConfig,
    },

    /// Phi-3 model with configuration
    Phi3 {
        model: Phi3Model,
        config: Phi3Config,
    },

    /// GraniteMoeHybrid (attention-only) with internal configuration
    GraniteMoeHybrid {
        model: GraniteMoeHybrid,
        config: GraniteMoeHybridInternalConfig,
    },

    /// BERT-family encoder-only model
    Bert { model: BertModel },

    /// Gemma model (Gemma 2/3/4) with configuration
    Gemma {
        model: GemmaModel,
        config: GemmaConfig,
    },

    /// Qwen2 model with configuration
    Qwen2 {
        model: Qwen2Model,
        config: Qwen2Config,
    },

    /// Qwen3 model with configuration
    Qwen3 {
        model: Qwen3Model,
        config: Qwen3Config,
    },

    /// DeepSeek V2/V3 model with configuration
    DeepSeek2 {
        model: DeepSeekV2,
        config: DeepSeekV2Config,
    },

    /// GLM-4 model with configuration
    Glm4 {
        model: Glm4Model,
        config: Glm4Config,
    },

    /// GGUF quantized model backend (feature-gated)
    #[cfg(feature = "gguf")]
    Gguf {
        backend: super::gguf_backend::GgufBackend,
    },

    /// MLX backend for Apple Silicon GPU acceleration (feature-gated)
    #[cfg(feature = "mlx")]
    Mlx {
        backend: super::mlx_backend::MlxBackend,
    },
}

impl LocalBackend {
    /// Load a Llama model from the given path
    ///
    /// This function:
    /// 1. Reads `config.json` for model parameters
    /// 2. Loads `.safetensors` weight files
    /// 3. Performs Metal warmup if applicable
    pub fn load_llama(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Llama model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!(
                "Failed to parse config.json: {}\n\nHint: The model file may be corrupted. Try re-downloading: 'model-rs download <model>'",
                e
            )))?;

        // Extract parameters from config.json
        let vocab_size = config_json
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;

        let hidden_size = config_json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(4096) as usize;

        let intermediate_size = config_json
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize;

        let num_hidden_layers = config_json
            .get("num_hidden_layers")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let num_attention_heads = config_json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;

        let num_key_value_heads = config_json
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(num_attention_heads as u64) as usize;

        let rms_norm_eps = config_json
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-5);

        let rope_theta = config_json
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10000.0) as f32;

        // Create LlamaConfig with actual model parameters
        let llama_config = LlamaConfig {
            hidden_size,
            intermediate_size,
            vocab_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta,
            use_flash_attn: false,
            ..LlamaConfig::config_7b_v2(false)
        };

        info!(
            "Config: vocab={}, hidden={}, layers={}, heads={}",
            vocab_size, hidden_size, num_hidden_layers, num_attention_heads
        );

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = Llama::load(vb, &llama_config)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        // On Metal, the first few decode steps can be much slower due to kernel compilation.
        // Warm up a few single-token forward passes with an increasing position to reduce
        // visible latency for the first generated words.
        let warmup_tokens: usize = std::env::var("MODEL_RS_WARMUP_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3);

        if warmup_tokens > 0 {
            if matches!(device, Device::Metal(_)) {
                info!("Metal warmup: running {} decode step(s)...", warmup_tokens);
                let t_warm = Instant::now();
                let mut warm_cache =
                    Cache::new(true, DType::F32, &llama_config, device).map_err(|e| {
                        ModelError::LocalModelError(format!("Failed to create warmup cache: {}", e))
                    })?;
                let warm_token: u32 = 0;
                for pos in 0..warmup_tokens {
                    let tensor = Tensor::new(&[warm_token], device)?.unsqueeze(0)?;
                    let _ = model.forward(&tensor, pos, &mut warm_cache)?;
                }
                info!("Metal warmup: done in {} ms", t_warm.elapsed().as_millis());
            }
        }

        info!("Model initialized");
        Ok(Some(LocalBackend::Llama {
            model,
            config: llama_config,
        }))
    }

    /// Load a Mistral model from the given path
    pub fn load_mistral(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Mistral model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let mistral_cfg: MistralConfig = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = MistralModel::new(&mistral_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Mistral {
            model,
            config: mistral_cfg,
        }))
    }

    /// Load a Mamba state-space model from the given path
    pub fn load_mamba(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Mamba model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let mamba_cfg: MambaConfig = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = MambaModel::new(&mamba_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Mamba {
            model,
            config: mamba_cfg,
        }))
    }

    /// Load a Phi-3 model from the given path
    pub fn load_phi3(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Phi-3 model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let phi3_cfg: Phi3Config = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = Phi3Model::new(&phi3_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Phi3 {
            model,
            config: phi3_cfg,
        }))
    }

    /// Load a GraniteMoeHybrid model (attention-only configs) from the given path
    ///
    /// Note: Models with Mamba layers are not supported
    pub fn load_granite_moe_hybrid(
        config: &LocalModelConfig,
        device: &Device,
    ) -> Result<Option<Self>> {
        info!("Loading GraniteMoeHybrid model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let cfg: GraniteMoeHybridConfig = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;
        let internal_cfg = cfg.into_config(false);

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = GraniteMoeHybrid::load(vb, &internal_cfg)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::GraniteMoeHybrid {
            model,
            config: internal_cfg,
        }))
    }

    /// Load a BERT-family encoder-only model from the given path
    ///
    /// Supports: BERT, RoBERTa, ALBERT (for embeddings only)
    pub fn load_bert(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading BERT-family model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let bert_cfg: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = BertModel::load(vb, &bert_cfg)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Bert { model }))
    }

    pub fn load_gemma(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Gemma model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let gemma_cfg: GemmaConfig = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = GemmaModel::new(false, &gemma_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Gemma {
            model,
            config: gemma_cfg,
        }))
    }

    pub fn load_qwen2(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Qwen2 model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let qwen2_cfg: Qwen2Config = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = Qwen2Model::new(&qwen2_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Qwen2 {
            model,
            config: qwen2_cfg,
        }))
    }

    pub fn load_qwen3(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading Qwen3 model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let qwen3_cfg: Qwen3Config = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = Qwen3Model::new(&qwen3_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Qwen3 {
            model,
            config: qwen3_cfg,
        }))
    }

    pub fn load_deepseek2(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading DeepSeek V2/V3 model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let deepseek_cfg: DeepSeekV2Config = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = DeepSeekV2::new(&deepseek_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::DeepSeek2 {
            model,
            config: deepseek_cfg,
        }))
    }

    pub fn load_glm4(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        info!("Loading GLM-4 model weights...");

        let config_path = config.model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.\nUse 'model-rs download <model>' to re-download the model.",
                config.model_path.display()
            )));
        }

        let config_content = fs::read_to_string(&config_path)?;
        let glm4_cfg: Glm4Config = serde_json::from_str(&config_content)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

        let weight_files = find_weight_files(&config.model_path)?;
        if weight_files.is_empty() {
            warn!("No .safetensors files found");
            return Ok(None);
        }

        info!("Loading {} weight file(s)...", weight_files.len());
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, device).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load weights: {}", e))
            })?
        };

        let model = Glm4Model::new(&glm4_cfg, vb)
            .map_err(|e| ModelError::LocalModelError(format!("Failed to create model: {}", e)))?;

        info!("Model initialized");
        Ok(Some(LocalBackend::Glm4 {
            model,
            config: glm4_cfg,
        }))
    }

    /// Load a GGUF quantized model from the given path
    ///
    /// GGUF models offer significant memory savings through quantization.
    /// The quantization format is auto-detected from the filename.
    #[cfg(feature = "gguf")]
    pub fn load_gguf(config: &LocalModelConfig, device: &Device) -> Result<Option<Self>> {
        use std::fs;
        info!("Loading GGUF model...");

        // Find GGUF files in the model directory
        let gguf_files: Vec<_> = fs::read_dir(&config.model_path)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "gguf"))
            .collect();

        if gguf_files.is_empty() {
            return Ok(None);
        }

        if gguf_files.len() > 1 {
            warn!(
                "Multiple GGUF files found, using first: {}",
                gguf_files[0].path().display()
            );
        }

        let gguf_path = &gguf_files[0].path();
        info!("Found GGUF file: {}", gguf_path.display());

        let backend = super::gguf_backend::GgufBackend::load(config, gguf_path)?;

        info!(
            "GGUF model loaded successfully (quantization: {})",
            backend.quantization()
        );
        Ok(Some(LocalBackend::Gguf { backend }))
    }

    /// Load a GGUF model (stub when GGUF feature is not enabled)
    #[cfg(not(feature = "gguf"))]
    pub fn load_gguf(_config: &LocalModelConfig, _device: &Device) -> Result<Option<Self>> {
        Ok(None)
    }

    /// Load a model using the MLX backend (Apple Silicon GPU acceleration)
    #[cfg(feature = "mlx")]
    pub fn load_mlx(config: &LocalModelConfig, _device: &Device) -> Result<Option<Self>> {
        info!("Loading model via MLX backend...");

        let backend = super::mlx_backend::MlxBackend::load(config)?;
        info!("MLX model loaded successfully");
        Ok(Some(LocalBackend::Mlx { backend }))
    }

    /// Load a model using MLX (stub when feature is not enabled)
    #[cfg(not(feature = "mlx"))]
    pub fn load_mlx(_config: &LocalModelConfig, _device: &Device) -> Result<Option<Self>> {
        Ok(None)
    }
}

/// Find all .safetensors weight files in the model directory
///
/// Returns a sorted list of paths to `.safetensors` files.
fn find_weight_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_weight_files_empty() {
        let result = find_weight_files(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}

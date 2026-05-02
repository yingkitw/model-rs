use super::config::ModelArchitecture;
use crate::error::{ModelError, Result};
use std::fs;
use std::path::Path;
use tracing::warn;

pub fn detect_architecture(model_path: &Path) -> Result<ModelArchitecture> {
    #[cfg(feature = "gguf")]
    {
        use super::gguf_backend::GgufBackend;
        if let Ok(Some(_)) = GgufBackend::detect_gguf(model_path) {
            return Ok(ModelArchitecture::LlamaQuantized);
        }
    }

    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return Ok(ModelArchitecture::Llama);
    }

    let config_content = fs::read_to_string(&config_path)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to read config: {}", e)))?;

    let config: serde_json::Value = serde_json::from_str(&config_content)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to parse config: {}", e)))?;

    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama");

    if config.get("num_local_experts").is_some()
        || config.get("num_experts").is_some()
        || config.get("expert_capacity").is_some()
        || config.get("router_aux_loss_coef").is_some()
    {
        return Err(ModelError::LocalModelError(
            "Unsupported model architecture: Mixture-of-Experts (MoE) models are not yet supported"
                .to_string(),
        ));
    }

    if config.get("layer_types").is_some() {
        let has_mamba_layer = config
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().any(|x| {
                    x.as_str()
                        .map(|s| s.eq_ignore_ascii_case("mamba"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        if has_mamba_layer {
            return Err(ModelError::LocalModelError(
                "Unsupported GraniteMoeHybrid config: contains Mamba layers (not supported by candle-transformers yet)".to_string(),
            ));
        }

        return Ok(ModelArchitecture::GraniteMoeHybrid);
    }

    match model_type {
        "llama" => Ok(ModelArchitecture::Llama),
        "mistral" => Ok(ModelArchitecture::Mistral),
        "mamba" => Ok(ModelArchitecture::Mamba),
        "granitemoehybrid" => Ok(ModelArchitecture::GraniteMoeHybrid),
        "bert" | "roberta" | "albert" => Ok(ModelArchitecture::Bert),
        "phi" => Ok(ModelArchitecture::Phi),
        "granite" => Ok(ModelArchitecture::Granite),
        "gemma" | "gemma2" | "gemma3" | "gemma4" => Ok(ModelArchitecture::Gemma),
        "qwen2" | "qwen2_moe" => Ok(ModelArchitecture::Qwen2),
        "qwen3" | "qwen3_moe" | "qwen3_vl" => Ok(ModelArchitecture::Qwen3),
        "deepseek_v2" | "deepseek_v3" | "deepseek" => Ok(ModelArchitecture::DeepSeek2),
        "kimi" | "kimi_v1" => Ok(ModelArchitecture::DeepSeek2),
        "glm4" | "glm4_new" | "chatglm" => Ok(ModelArchitecture::Glm4),
        _ => {
            warn!("Unknown model type '{}', defaulting to Llama", model_type);
            Ok(ModelArchitecture::Llama)
        }
    }
}

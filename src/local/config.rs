use crate::error::ModelError;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub enum ModelArchitecture {
    Llama,
    LlamaQuantized,
    Mistral,
    Mamba,
    GraniteMoeHybrid,
    Bert,
    Phi,
    Granite,
    Gemma,
    Qwen2,
    Qwen3,
    DeepSeek2,
    Glm4,
    Mlx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Metal,
    Cuda,
    Mlx,
}

impl FromStr for DevicePreference {
    type Err = ModelError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            "mlx" => Ok(Self::Mlx),
            other => Err(ModelError::InvalidConfig(format!(
                "Invalid device '{}'. Use one of: auto, cpu, metal, cuda, mlx",
                other
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocalModelConfig {
    pub model_path: PathBuf,
    pub architecture: ModelArchitecture,
    pub quantized: bool,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub device_preference: DevicePreference,
    pub device_index: usize,
}

impl Default for LocalModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models"),
            architecture: ModelArchitecture::Llama,
            quantized: false,
            max_seq_len: 4096,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            device_preference: DevicePreference::Auto,
            device_index: 0,
        }
    }
}

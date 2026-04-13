use std::path::PathBuf;

/// Helper functions to read configuration from environment variables with fallbacks
pub fn get_model_path() -> Option<PathBuf> {
    std::env::var("MODEL_RS_MODEL_PATH").ok().map(PathBuf::from)
}

pub fn get_temperature() -> f32 {
    std::env::var("MODEL_RS_TEMPERATURE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7)
}

pub fn get_top_p() -> f32 {
    std::env::var("MODEL_RS_TOP_P")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.9)
}

pub fn get_top_k() -> Option<usize> {
    std::env::var("MODEL_RS_TOP_K")
        .ok()
        .and_then(|v| v.parse().ok())
}

pub fn get_repeat_penalty() -> f32 {
    std::env::var("MODEL_RS_REPEAT_PENALTY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.1)
}

pub fn get_max_tokens() -> usize {
    std::env::var("MODEL_RS_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512)
}

pub fn get_device() -> String {
    std::env::var("MODEL_RS_DEVICE").unwrap_or_else(|_| "auto".to_string())
}

pub fn get_device_index() -> usize {
    std::env::var("MODEL_RS_DEVICE_INDEX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

pub fn get_port() -> u16 {
    std::env::var("MODEL_RS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080)
}

pub fn get_mirror() -> String {
    std::env::var("MODEL_RS_MIRROR").unwrap_or_else(|_| "https://hf-mirror.com".to_string())
}

pub fn get_output_dir() -> Option<PathBuf> {
    std::env::var("MODEL_RS_OUTPUT_DIR").ok().map(PathBuf::from)
}

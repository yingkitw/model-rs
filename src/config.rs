use std::path::PathBuf;

pub const DEFAULT_MIRROR: &str = "https://hf-mirror.com";

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
    std::env::var("MODEL_RS_MIRROR").unwrap_or_else(|_| DEFAULT_MIRROR.to_string())
}

pub fn get_output_dir() -> Option<PathBuf> {
    std::env::var("MODEL_RS_OUTPUT_DIR").ok().map(PathBuf::from)
}

/// Render the resolved environment-derived configuration as a markdown string.
///
/// Used by the `model-rs config env` subcommand (and the legacy `config`
/// shortcut before subcommands were added).
pub fn env_config_markdown() -> String {
    let model_path = get_model_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "Not set".to_string());
    let output_dir = get_output_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "./models".to_string());
    let top_k = get_top_k()
        .map(|k| k.to_string())
        .unwrap_or_else(|| "Not set".to_string());

    format!(
        r#"### Model Settings
- **Model Path:** `{}`
- **Output Directory:** `{}`
- **Mirror URL:** `{}`

### Generation Parameters
- **Temperature:** `{}`
- **Top-P:** `{}`
- **Top-K:** `{}`
- **Repeat Penalty:** `{}`
- **Max Tokens:** `{}`

### Device Settings
- **Device:** `{}`
- **Device Index:** `{}`

### Server Settings
- **Port:** `{}`

### Environment Variables
Set these in your `.env` file or environment:
- `MODEL_RS_MODEL_PATH` - Default model path
- `MODEL_RS_OUTPUT_DIR` - Download output directory
- `MODEL_RS_MIRROR` - HuggingFace mirror URL
- `MODEL_RS_TEMPERATURE` - Generation temperature
- `MODEL_RS_TOP_P` - Top-p sampling threshold
- `MODEL_RS_TOP_K` - Top-k sampling limit
- `MODEL_RS_REPEAT_PENALTY` - Repetition penalty
- `MODEL_RS_MAX_TOKENS` - Maximum tokens to generate
- `MODEL_RS_DEVICE` - Compute device (auto/cpu/metal/cuda)
- `MODEL_RS_DEVICE_INDEX` - GPU device index
- `MODEL_RS_PORT` - Server port
- `MODEL_RS_WARMUP_TOKENS` - Metal decode warmup passes (optional tuning)
"#,
        model_path,
        output_dir,
        get_mirror(),
        get_temperature(),
        get_top_p(),
        top_k,
        get_repeat_penalty(),
        get_max_tokens(),
        get_device(),
        get_device_index(),
        get_port(),
    )
}

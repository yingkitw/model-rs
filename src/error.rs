use thiserror::Error;
use std::path::PathBuf;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Download failed: {0}")]
    DownloadError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Local model error: {0}")]
    LocalModelError(String),

    #[error("Candle error: {0}")]
    CandleError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("GGUF model error: {0}")]
    GgufError(String),

    #[error("Quantization format '{0}' not supported")]
    UnsupportedQuantization(String),

    #[error("GGUF file parsing failed for '{path}': {reason}")]
    GgufParsingError { path: String, reason: String },

    #[error("MLX error: {0}")]
    MlxError(String),

    #[error("Model validation failed for '{model}': {reason}")]
    ValidationError { model: String, reason: String, suggestion: String },

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Configuration file error: {0}")]
    ConfigFileError(String),

    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),
}

impl From<candle_core::Error> for ModelError {
    fn from(err: candle_core::Error) -> Self {
        ModelError::CandleError(err.to_string())
    }
}

impl From<tokenizers::Error> for ModelError {
    fn from(err: tokenizers::Error) -> Self {
        ModelError::TokenizerError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ModelError>;

impl ModelError {
    pub fn download_failed(reason: impl Into<String>) -> Self {
        ModelError::DownloadError(reason.into())
    }

    pub fn model_not_found(reason: impl Into<String>) -> Self {
        ModelError::ModelNotFound(reason.into())
    }

    pub fn model_not_found_with_path(model: impl Into<String>, path: impl Into<PathBuf>, suggestion: impl Into<String>) -> Self {
        // Wrap the richer call site into a single message; path/suggestion are surfaced via Display.
        let model = model.into();
        let path = path.into();
        let suggestion = suggestion.into();
        ModelError::ModelNotFound(format!("Model '{}' not found at '{}' ({})", model, path.display(), suggestion))
    }

    pub fn invalid_config(details: impl Into<String>) -> Self {
        ModelError::InvalidConfig(details.into())
    }

    pub fn llm_error(reason: impl Into<String>) -> Self {
        ModelError::LlmError(reason.into())
    }

    pub fn local_model_error(reason: impl Into<String>) -> Self {
        ModelError::LocalModelError(reason.into())
    }

    pub fn validation_error(model: impl Into<String>, reason: impl Into<String>, suggestion: impl Into<String>) -> Self {
        ModelError::ValidationError {
            model: model.into(),
            reason: reason.into(),
            suggestion: suggestion.into(),
        }
    }

    pub fn cache_error(reason: impl Into<String>) -> Self {
        ModelError::CacheError(reason.into())
    }

    pub fn timeout_error(reason: impl Into<String>) -> Self {
        ModelError::TimeoutError(reason.into())
    }

    pub fn config_file_error(reason: impl Into<String>) -> Self {
        ModelError::ConfigFileError(reason.into())
    }

    pub fn authentication_error(reason: impl Into<String>) -> Self {
        ModelError::AuthenticationError(reason.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_error_display() {
        let err = ModelError::download_failed("Failed to download");
        assert_eq!(err.to_string(), "Download failed: Failed to download");
    }

    #[test]
    fn test_model_not_found_display() {
        let err = ModelError::model_not_found("model not found");
        assert_eq!(err.to_string(), "Model not found: model not found");
    }

    #[test]
    fn test_invalid_config_display() {
        let err = ModelError::invalid_config("invalid config");
        assert_eq!(err.to_string(), "Invalid configuration: invalid config");
    }

    #[test]
    fn test_llm_error_display() {
        let err = ModelError::llm_error("LLM failed");
        assert_eq!(err.to_string(), "LLM error: LLM failed");
    }

    #[test]
    fn test_local_model_error_display() {
        let err = ModelError::local_model_error("Model load failed");
        assert_eq!(err.to_string(), "Local model error: Model load failed");
    }

    #[test]
    fn test_candle_error_display() {
        let err = ModelError::CandleError("Candle error".to_string());
        assert_eq!(err.to_string(), "Candle error: Candle error");
    }

    #[test]
    fn test_tokenizer_error_display() {
        let err = ModelError::TokenizerError("Tokenizer failed".to_string());
        assert_eq!(err.to_string(), "Tokenizer error: Tokenizer failed");
    }

    #[test]
    fn test_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let model_err: ModelError = io_err.into();
        assert!(matches!(model_err, ModelError::IoError(_)));
        assert!(model_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_from_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let model_err: ModelError = json_err.into();
        assert!(matches!(model_err, ModelError::JsonError(_)));
    }

    #[test]
    fn test_validation_error_struct() {
        let err = ModelError::validation_error("foo/bar", "bad", "fix it");
        assert_eq!(err.to_string(), "Model validation failed for 'foo/bar': bad");
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<String> {
            Ok("success".to_string())
        }
        fn returns_err() -> Result<String> {
            Err(ModelError::download_failed("test"))
        }

        assert!(returns_ok().is_ok());
        assert!(returns_err().is_err());
    }
}
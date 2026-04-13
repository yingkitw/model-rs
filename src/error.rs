use thiserror::Error;

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

    #[error("GGUF file parsing failed: {0}")]
    GgufParsingError(String),

    #[error("MLX error: {0}")]
    MlxError(String),
}

impl From<candle_core::Error> for ModelError {
    fn from(err: candle_core::Error) -> Self {
        ModelError::CandleError(err.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for ModelError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        ModelError::TokenizerError(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ModelError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_error_display() {
        let err = ModelError::DownloadError("Failed to download".to_string());
        assert_eq!(err.to_string(), "Download failed: Failed to download");
    }

    #[test]
    fn test_model_not_found_display() {
        let err = ModelError::ModelNotFound("model not found".to_string());
        assert_eq!(err.to_string(), "Model not found: model not found");
    }

    #[test]
    fn test_invalid_config_display() {
        let err = ModelError::InvalidConfig("invalid config".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: invalid config");
    }

    #[test]
    fn test_llm_error_display() {
        let err = ModelError::LlmError("LLM failed".to_string());
        assert_eq!(err.to_string(), "LLM error: LLM failed");
    }

    #[test]
    fn test_local_model_error_display() {
        let err = ModelError::LocalModelError("Model load failed".to_string());
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
    fn test_result_type_alias() {
        fn returns_ok() -> Result<String> {
            Ok("success".to_string())
        }
        fn returns_err() -> Result<String> {
            Err(ModelError::DownloadError("test".to_string()))
        }

        assert!(returns_ok().is_ok());
        assert!(returns_err().is_err());
    }
}

//! Input validation module for model-rs
//!
//! This module provides comprehensive validation for CLI arguments, HTTP requests,
//! and model configurations to ensure robust error handling and security.

use crate::error::{ModelError, Result};
use std::path::{Path, PathBuf};
use std::net::IpAddr;
use std::str::FromStr;
use regex::Regex;

/// Validator for model names and paths
pub struct ModelValidator {
    model_name_regex: Regex,
    path_regex: Regex,
}

impl ModelValidator {
    /// Create a new ModelValidator with compiled regex patterns
    pub fn new() -> Result<Self> {
        let model_name_regex = Regex::new(r"^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?$")
            .map_err(|e| ModelError::invalid_config(
                format!("Invalid model name regex: {}. Check model name format", e),
            ))?;

        let path_regex = Regex::new(r"^[a-zA-Z0-9_\-/.,@]+$")
            .map_err(|e| ModelError::invalid_config(
                format!("Invalid path regex: {}. Check file path format", e),
            ))?;

        Ok(Self {
            model_name_regex,
            path_regex,
        })
    }

    /// Validate a Hugging Face model name
    pub fn validate_model_name(&self, model: &str) -> Result<()> {
        if model.is_empty() {
            return Err(ModelError::validation_error(
                model,
                "Model name cannot be empty",
                "Provide a valid Hugging Face model name like 'org/model'"
            ));
        }

        if !self.model_name_regex.is_match(model) {
            return Err(ModelError::validation_error(
                model,
                "Invalid model name format",
                "Use format 'org/model' with alphanumeric characters, dots, dashes, and underscores"
            ));
        }

        if !model.contains('/') {
            return Err(ModelError::validation_error(
                model,
                "Model name must include organization",
                "Use format 'org/model' like 'meta-llama/Llama-2-7b'"
            ));
        }

        Ok(())
    }

    /// Validate a file system path
    pub fn validate_path(&self, path: &Path, path_type: &str) -> Result<()> {
        let path_str = path.to_string_lossy();
        
        if path_str.is_empty() {
            return Err(ModelError::validation_error(
                path_str.as_ref(),
                "Path cannot be empty",
                format!("Provide a valid {} path", path_type)
            ));
        }

        if !self.path_regex.is_match(&path_str) {
            return Err(ModelError::validation_error(
                path_str.as_ref(),
                "Invalid characters in path",
                format!("Use only alphanumeric characters, underscores, dashes, dots, and slashes for {} path", path_type)
            ));
        }

        // Check for potentially dangerous paths
        if path_str.contains("..") {
            return Err(ModelError::validation_error(
                path_str.as_ref(),
                "Path traversal not allowed",
                "Use absolute paths without '..' components"
            ));
        }

        if path_str.starts_with("~/") {
            return Err(ModelError::validation_error(
                path_str.as_ref(),
                "Tilde expansion not supported",
                "Use absolute paths directly"
            ));
        }

        Ok(())
    }

    /// Validate model directory structure
    pub async fn validate_model_directory(&self, model_path: &Path) -> Result<()> {
        self.validate_path(model_path, "model")?;

        // Check if path exists and is a directory
        if !tokio::fs::try_exists(model_path).await
            .map_err(|e| ModelError::validation_error(
                model_path.to_string_lossy().as_ref(),
                format!("Cannot access path: {}", e),
                "Check file permissions and path validity"
            ))? {
            return Err(ModelError::model_not_found(
                model_path.to_string_lossy().as_ref(),
            ));
        }

        if !tokio::fs::metadata(model_path).await
            .map_err(|e| ModelError::validation_error(
                model_path.to_string_lossy().as_ref(),
                format!("Cannot get metadata: {}", e),
                "Check if the path is a directory"
            ))?
            .is_dir() {
            return Err(ModelError::validation_error(
                model_path.to_string_lossy().as_ref(),
                "Path is not a directory",
                "Provide a directory path containing model files"
            ));
        }

        // Check for required files
        let config_file = model_path.join("config.json");
        if !tokio::fs::try_exists(&config_file).await
            .map_err(|e| ModelError::validation_error(
                config_file.to_string_lossy().as_ref(),
                format!("Cannot check config file: {}", e),
                "Check file permissions"
            ))? {
            return Err(ModelError::validation_error(
                model_path.to_string_lossy().as_ref(),
                "Missing config.json",
                "Ensure the model directory contains a valid config.json file"
            ));
        }

        Ok(())
    }
}

/// Validator for HTTP API parameters
pub struct HttpValidator {
    port_range: std::ops::RangeInclusive<u16>,
    max_tokens_range: std::ops::RangeInclusive<u32>,
    temperature_range: std::ops::RangeInclusive<f32>,
}

impl HttpValidator {
    /// Create a new HttpValidator with validation ranges
    pub fn new() -> Self {
        Self {
            port_range: 1..=65535,
            max_tokens_range: 1..=32768,
            temperature_range: 0.0..=2.0,
        }
    }

    /// Validate a port number
    pub fn validate_port(&self, port: u16) -> Result<()> {
        if !self.port_range.contains(&port) {
            return Err(ModelError::validation_error(
                &port.to_string(),
                "Invalid port number",
                "Use a port between 1 and 65535"
            ));
        }
        Ok(())
    }

    /// Validate max tokens parameter
    pub fn validate_max_tokens(&self, max_tokens: u32) -> Result<()> {
        if !self.max_tokens_range.contains(&max_tokens) {
            return Err(ModelError::validation_error(
                &max_tokens.to_string(),
                "Invalid max tokens value",
                "Use a value between 1 and 32768"
            ));
        }
        Ok(())
    }

    /// Validate temperature parameter
    pub fn validate_temperature(&self, temperature: f32) -> Result<()> {
        if !self.temperature_range.contains(&temperature) {
            return Err(ModelError::validation_error(
                &temperature.to_string(),
                "Invalid temperature value",
                "Use a value between 0.0 and 2.0"
            ));
        }
        Ok(())
    }

    /// Validate top_p parameter
    pub fn validate_top_p(&self, top_p: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ModelError::validation_error(
                &top_p.to_string(),
                "Invalid top_p value",
                "Use a value between 0.0 and 1.0"
            ));
        }
        Ok(())
    }

    /// Validate top_k parameter
    pub fn validate_top_k(&self, top_k: u32) -> Result<()> {
        if top_k == 0 || top_k > 1000 {
            return Err(ModelError::validation_error(
                &top_k.to_string(),
                "Invalid top_k value",
                "Use a value between 1 and 1000"
            ));
        }
        Ok(())
    }

    /// Validate repeat penalty parameter
    pub fn validate_repeat_penalty(&self, repeat_penalty: f32) -> Result<()> {
        if repeat_penalty < 0.0 || repeat_penalty > 2.0 {
            return Err(ModelError::validation_error(
                &repeat_penalty.to_string(),
                "Invalid repeat penalty value",
                "Use a value between 0.0 and 2.0"
            ));
        }
        Ok(())
    }
}

/// Validator for device and backend configurations
pub struct DeviceValidator {
    valid_devices: Vec<String>,
}

impl DeviceValidator {
    /// Create a new DeviceValidator
    pub fn new() -> Self {
        Self {
            valid_devices: vec![
                "auto".to_string(),
                "cpu".to_string(),
                "metal".to_string(),
                "cuda".to_string(),
                "mlx".to_string(),
            ],
        }
    }

    /// Validate device preference
    pub fn validate_device(&self, device: &str) -> Result<()> {
        if !self.valid_devices.contains(&device.to_string()) {
            return Err(ModelError::validation_error(
                device,
                "Invalid device preference",
                &format!("Use one of: {}", self.valid_devices.join(", "))
            ));
        }
        Ok(())
    }

    /// Validate device index
    pub fn validate_device_index(&self, device_index: Option<u32>) -> Result<()> {
        if let Some(index) = device_index {
            if index > 7 {
                return Err(ModelError::validation_error(
                    &index.to_string(),
                    "Invalid device index",
                    "Use a device index between 0 and 7"
                ));
            }
        }
        Ok(())
    }
}

/// Global validator instances (lazy initialized)
pub static MODEL_VALIDATOR: once_cell::sync::Lazy<ModelValidator> = 
    once_cell::sync::Lazy::new(|| {
        ModelValidator::new().expect("Failed to initialize model validator")
    });

pub static HTTP_VALIDATOR: once_cell::sync::Lazy<HttpValidator> = 
    once_cell::sync::Lazy::new(HttpValidator::new);

pub static DEVICE_VALIDATOR: once_cell::sync::Lazy<DeviceValidator> = 
    once_cell::sync::Lazy::new(DeviceValidator::new);

/// Convenience functions for common validations
pub fn validate_model_name(model: &str) -> Result<()> {
    MODEL_VALIDATOR.validate_model_name(model)
}

pub fn validate_path(path: &Path, path_type: &str) -> Result<()> {
    MODEL_VALIDATOR.validate_path(path, path_type)
}

pub async fn validate_model_directory(model_path: &Path) -> Result<()> {
    MODEL_VALIDATOR.validate_model_directory(model_path).await
}

pub fn validate_port(port: u16) -> Result<()> {
    HTTP_VALIDATOR.validate_port(port)
}

pub fn validate_max_tokens(max_tokens: u32) -> Result<()> {
    HTTP_VALIDATOR.validate_max_tokens(max_tokens)
}

pub fn validate_temperature(temperature: f32) -> Result<()> {
    HTTP_VALIDATOR.validate_temperature(temperature)
}

pub fn validate_top_p(top_p: f32) -> Result<()> {
    HTTP_VALIDATOR.validate_top_p(top_p)
}

pub fn validate_top_k(top_k: u32) -> Result<()> {
    HTTP_VALIDATOR.validate_top_k(top_k)
}

pub fn validate_repeat_penalty(repeat_penalty: f32) -> Result<()> {
    HTTP_VALIDATOR.validate_repeat_penalty(repeat_penalty)
}

pub fn validate_device(device: &str) -> Result<()> {
    DEVICE_VALIDATOR.validate_device(device)
}

pub fn validate_device_index(device_index: Option<u32>) -> Result<()> {
    DEVICE_VALIDATOR.validate_device_index(device_index)
}

/// Validate the standard generation-parameter bundle shared by `generate`,
/// `chat`, and `run`. Returns the first failing validation as a `ModelError`.
pub fn validate_generation_args(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repeat_penalty: f32,
    device: &str,
    device_index: usize,
) -> Result<()> {
    validate_max_tokens(max_tokens as u32)?;
    validate_temperature(temperature)?;
    validate_top_p(top_p)?;
    if let Some(k) = top_k {
        validate_top_k(k as u32)?;
    }
    validate_repeat_penalty(repeat_penalty)?;
    validate_device(device)?;
    validate_device_index(Some(device_index as u32))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tokio_test;

    #[test]
    fn test_model_validator_creation() {
        let validator = ModelValidator::new();
        assert!(validator.is_ok());
    }

    #[test]
    fn test_valid_model_name() {
        let validator = ModelValidator::new().unwrap();
        
        // Valid model names
        for valid_model in [
            "meta-llama/Llama-2-7b",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/phi-2",
        ] {
            assert!(validator.validate_model_name(valid_model).is_ok());
        }
    }

    #[test]
    fn test_invalid_model_names() {
        let validator = ModelValidator::new().unwrap();
        
        // Invalid model names
        for invalid_model in [
            "",
            "meta-llama/Llama-2-7b/",
            "meta-llama//Llama-2-7b",
            "invalid model name",
            "meta-llama\\Llama-2-7b",
            "no_slash",
        ] {
            assert!(validator.validate_model_name(invalid_model).is_err());
        }
    }

    #[test]
    fn test_path_validation() {
        let validator = ModelValidator::new().unwrap();
        
        // Valid paths
        for valid_path in [
            "/path/to/model",
            "./relative/path",
            "/Users/user/model",
            "/opt/models/test-model",
        ] {
            assert!(validator.validate_path(Path::new(valid_path), "test").is_ok());
        }
    }

    #[test]
    fn test_invalid_paths() {
        let validator = ModelValidator::new().unwrap();
        
        // Invalid paths
        for invalid_path in [
            "",
            "/path/../to/model",
            "~/model",
            "/path/to/model\x00",
        ] {
            assert!(validator.validate_path(Path::new(invalid_path), "test").is_err());
        }
    }

    #[tokio::test]
    async fn test_model_directory_validation() {
        let validator = ModelValidator::new().unwrap();
        
        // This test would need actual file system setup
        // For now, test the error cases
        let non_existent = PathBuf::from("/tmp/non_existent_model_12345");
        assert!(validator.validate_model_directory(&non_existent).await.is_err());
    }

    #[test]
    fn test_http_validation() {
        let validator = HttpValidator::new();
        
        // Test port validation
        assert!(validator.validate_port(8080).is_ok());
        assert!(validator.validate_port(0).is_err());
        assert!(validator.validate_port(u16::MAX).is_ok());
        
        // Test max tokens validation
        assert!(validator.validate_max_tokens(1000).is_ok());
        assert!(validator.validate_max_tokens(0).is_err());
        assert!(validator.validate_max_tokens(50000).is_err());
        
        // Test temperature validation
        assert!(validator.validate_temperature(1.0).is_ok());
        assert!(validator.validate_temperature(-0.1).is_err());
        assert!(validator.validate_temperature(2.1).is_err());
    }

    #[test]
    fn test_device_validation() {
        let validator = DeviceValidator::new();

        // Valid devices
        for valid_device in ["auto", "cpu", "metal", "cuda", "mlx"] {
            assert!(validator.validate_device(valid_device).is_ok());
        }

        // Invalid devices
        assert!(validator.validate_device("invalid_device").is_err());
    }

    #[test]
    fn test_validate_generation_args_happy_path() {
        let r = validate_generation_args(512, 0.7, 0.9, Some(40), 1.1, "auto", 0);
        assert!(r.is_ok(), "expected ok, got {:?}", r.err());
    }

    #[test]
    fn test_validate_generation_args_bad_temperature() {
        let r = validate_generation_args(512, 5.0, 0.9, None, 1.1, "auto", 0);
        assert!(r.is_err());
    }

    #[test]
    fn test_validate_generation_args_bad_device() {
        let r = validate_generation_args(512, 0.7, 0.9, None, 1.1, "tpu", 0);
        assert!(r.is_err());
    }
}
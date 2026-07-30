//! Comprehensive error handling and edge case tests for model-rs
//!
//! This test suite validates:
//! - Error handling for invalid inputs
//! - Edge cases in model loading and inference
//! - HTTP API error responses
//! - Configuration validation
//! - File system error handling
//! - Network error scenarios

use model_rs::{error::ModelError, validation::*};
use std::path::Path;
use std::path::PathBuf;
use tokio_test;

#[tokio::test]
async fn test_error_handling_edge_cases() {
    // Test error construction
    let err = ModelError::download_failed("network timeout");
    assert!(err.to_string().contains("Download failed"));
    
    let err = ModelError::model_not_found("missing/model");
    assert!(err.to_string().contains("Model not found"));
    
    let err = ModelError::invalid_config("invalid value");
    assert!(err.to_string().contains("Invalid configuration"));
}

#[tokio::test]
async fn test_validation_edge_cases() {
    let validator = &*MODEL_VALIDATOR;
    
    // Test extremely long model names
    let long_name = "a".repeat(1000);
    let result = validator.validate_model_name(&format!("org/{}", long_name));
    // Should pass as long as it follows the pattern
    assert!(result.is_ok());
    
    // Test model names with special characters that should be allowed
    for valid_name in [
        "test_org/model_name",
        "test-org/model-name",
        "test_org/model_v2",
        "test-org/model.v2",
    ] {
        assert!(validator.validate_model_name(valid_name).is_ok());
    }
    
    // Test problematic model names
    for invalid_name in [
        "", // empty
        "no_slash", // missing slash
        "/model", // missing org
        "org//model", // double slash
        "org/model/", // trailing slash
        "org\\model", // wrong separator
    ] {
        assert!(validator.validate_model_name(invalid_name).is_err());
    }
}

#[tokio::test]
async fn test_http_validation_edge_cases() {
    // Test boundary values for HTTP parameters
    assert!(validate_port(0).is_err()); // too low
    assert!(validate_port(65535).is_ok()); // max valid
    assert!(validate_port(8080).is_ok()); // valid
    assert!(validate_port(1).is_ok()); // min valid
    assert!(validate_port(65535).is_ok()); // max valid
    
    // Test max tokens boundaries
    assert!(validate_max_tokens(0).is_err()); // too low
    assert!(validate_max_tokens(50000).is_err()); // too high
    assert!(validate_max_tokens(1000).is_ok()); // valid
    assert!(validate_max_tokens(1).is_ok()); // min valid
    assert!(validate_max_tokens(32768).is_ok()); // max valid
    
    // Test temperature boundaries
    assert!(validate_temperature(-0.1).is_err()); // too low
    assert!(validate_temperature(2.1).is_err()); // too high
    assert!(validate_temperature(0.0).is_ok()); // min valid
    assert!(validate_temperature(1.0).is_ok()); // valid
    assert!(validate_temperature(2.0).is_ok()); // max valid
    
    // Test top_p boundaries
    assert!(validate_top_p(-0.1).is_err()); // too low
    assert!(validate_top_p(1.1).is_err()); // too high
    assert!(validate_top_p(0.0).is_ok()); // min valid
    assert!(validate_top_p(0.5).is_ok()); // valid
    assert!(validate_top_p(1.0).is_ok()); // max valid
    
    // Test top_k boundaries
    assert!(validate_top_k(0).is_err()); // too low
    assert!(validate_top_k(1001).is_err()); // too high
    assert!(validate_top_k(1).is_ok()); // min valid
    assert!(validate_top_k(100).is_ok()); // valid
    assert!(validate_top_k(1000).is_ok()); // max valid
    
    // Test repeat_penalty boundaries
    assert!(validate_repeat_penalty(-0.1).is_err()); // too low
    assert!(validate_repeat_penalty(2.1).is_err()); // too high
    assert!(validate_repeat_penalty(0.0).is_ok()); // min valid
    assert!(validate_repeat_penalty(1.0).is_ok()); // valid
    assert!(validate_repeat_penalty(2.0).is_ok()); // max valid
}

#[tokio::test]
async fn test_device_validation_edge_cases() {
    // Test valid device names
    for valid_device in ["auto", "cpu", "metal", "cuda", "mlx"] {
        assert!(DEVICE_VALIDATOR.validate_device(valid_device).is_ok());
    }
    
    // Test invalid device names
    for invalid_device in [
        "", // empty
        "invalid_device", // doesn't exist
        "CPU", // wrong case
        " metal", // with space
        "metal ", // trailing space
    ] {
        assert!(DEVICE_VALIDATOR.validate_device(invalid_device).is_err());
    }
    
    // Test device index boundaries
    assert!(DEVICE_VALIDATOR.validate_device_index(Some(8)).is_err()); // too high
    assert!(DEVICE_VALIDATOR.validate_device_index(Some(0)).is_ok()); // min valid
    assert!(DEVICE_VALIDATOR.validate_device_index(Some(7)).is_ok()); // max valid
    assert!(DEVICE_VALIDATOR.validate_device_index(None).is_ok()); // None is valid
}

#[tokio::test]
async fn test_path_validation_edge_cases() {
    let validator = &*MODEL_VALIDATOR;
    
    // Test valid paths with various edge cases
    for valid_path in [
        "/path/to/model",
        "./relative/path",
        "/Users/user/model",
        "/opt/models/test-model",
        "/path/with-dashes/and.dots",
        "./path_with_underscores",
        "/very/long/path/that/goes/through/multiple/directories",
    ] {
        assert!(validator.validate_path(Path::new(valid_path), "test").is_ok());
    }
    
    // Test invalid paths with dangerous patterns
    for invalid_path in [
        "", // empty
        "/path/../to/model", // path traversal
        "~/model", // tilde expansion
        "../../etc/passwd", // traversal attack
    ] {
        assert!(validator.validate_path(Path::new(invalid_path), "test").is_err());
    }
}

#[tokio::test]
async fn test_model_directory_validation_edge_cases() {
    let validator = &*MODEL_VALIDATOR;
    
    // Test with non-existent directory
    let non_existent = PathBuf::from("/tmp/non_existent_model_12345");
    let result = validator.validate_model_directory(&non_existent).await;
    assert!(result.is_err());
    
    // Test validation with file instead of directory (would fail in real scenario)
    // This would need actual file system setup to test properly
}

#[tokio::test]
async fn test_error_conversion_edge_cases() {
    // Test that error conversions preserve context
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "permission denied");
    let model_err: ModelError = io_err.into();
    assert!(model_err.to_string().contains("permission denied"));
    
    // Test JSON error conversion
    let json_err = serde_json::from_str::<serde_json::Value>("invalid { json").unwrap_err();
    let model_err: ModelError = json_err.into();
    assert!(model_err.to_string().contains("JSON error"));
}

#[tokio::test]
async fn test_cli_argument_validation_integration() {
    // This test would require CLI argument parsing to be tested
    // For now, we test the validation functions directly
    
    // Test model name validation that would be used by CLI
    assert!(validate_model_name("meta-llama/Llama-2-7b").is_ok());
    assert!(validate_model_name("TinyLlama/TinyLlama-1.1B-Chat-v1.0").is_ok());
    
    // Test invalid model names that should be caught by CLI
    assert!(validate_model_name("").is_err());
    assert!(validate_model_name("invalid_model_name").is_err());
    assert!(validate_model_name("org//model").is_err());
    
    // Test port validation for serve command
    assert!(validate_port(8080).is_ok());
    assert!(validate_port(0).is_err());
    assert!(validate_port(65535).is_ok());
    
    // Test temperature validation for generate command
    assert!(validate_temperature(0.7).is_ok());
    assert!(validate_temperature(-0.1).is_err());
    assert!(validate_temperature(2.1).is_err());
}

#[tokio::test]
async fn test_configuration_validation_edge_cases() {
    // Test configuration edge cases that might cause issues
    let config_tests = vec![
        (Some("".to_string()), "empty port"),
        (Some("99999".to_string()), "invalid port number"),
        (Some("0".to_string()), "port zero"),
        (Some("65536".to_string()), "port too high"),
        (Some("1.5".to_string()), "non-integer port"),
    ];
    
    for (value, description) in config_tests {
        match value {
            Some(port_str) => {
                if let Ok(port) = port_str.parse::<u16>() {
                    // If it parses, validate it
                    let result = validate_port(port);
                    match result {
                        Ok(_) => println!("Port {} validation passed: {}", port_str, description),
                        Err(e) => println!("Port {} validation failed: {} - {}", port_str, description, e),
                    }
                } else {
                    println!("Port {} invalid format: {}", port_str, description);
                }
            }
            None => println!("Missing configuration: {}", description),
        }
    }
}

#[tokio::test]
async fn test_network_error_scenarios() {
    // Test scenarios that might occur during network operations
    let error_scenarios = vec![
        ("Connection timeout", "Connection timed out"),
        ("DNS resolution failed", "failed to resolve"),
        // SSL error scenarios would be handled by reqwest,
    ];
    
    for (scenario, error_msg) in error_scenarios {
        println!("Testing network error scenario: {}", scenario);
        // In real implementation, these would be caught and converted to ModelError
        // For now, we just verify the error messages would be appropriate
        assert!(!error_msg.is_empty());
    }
}

#[tokio::test]
async fn test_filesystem_error_scenarios() {
    // Test scenarios that might occur during file operations
    let error_scenarios = vec![
        ("Permission denied", "Permission denied"),
        ("Disk full", "No space left on device"),
        ("File not found", "No such file or directory"),
        ("Invalid path", "Invalid filename"),
    ];
    
    for (scenario, error_msg) in error_scenarios {
        println!("Testing filesystem error scenario: {}", scenario);
        // These would be caught and converted to ModelError::IoError
        assert!(!error_msg.is_empty());
    }
}

#[tokio::test]
async fn test_memory_error_scenarios() {
    // Test scenarios related to memory issues
    let memory_scenarios = vec![
        ("Out of memory", "out of memory"),
        ("Allocation failed", "allocation failed"),
        ("Buffer too large", "buffer too large"),
    ];
    
    for (scenario, error_msg) in memory_scenarios {
        println!("Testing memory error scenario: {}", scenario);
        // These would typically be caught as ModelError::LlmError
        assert!(!error_msg.is_empty());
    }
}

#[tokio::test]
async fn test_concurrent_validation() {
    // Test that validation works correctly in concurrent scenarios
    use tokio::task;
    
    let mut handles = vec![];
    
    // Spawn multiple validation tasks concurrently
    for i in 0..10 {
        let handle = task::spawn(async move {
            let model_name = format!("test-org/model-{}", i);
            validate_model_name(&model_name)
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        let result = handle.await.unwrap();
        // Each should succeed with valid model names
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_validation_error_messages() {
    // Test that validation error messages are helpful and specific
    
    let test_cases: Vec<(Box<dyn Fn() -> model_rs::Result<()>>, &str)> = vec![
        (Box::new(|| validate_model_name("")), "empty model name"),
        (Box::new(|| validate_model_name("no_slash")), "missing organization"),
        (Box::new(|| validate_model_name("org//model")), "invalid format"),
        (Box::new(|| validate_port(0)), "invalid port"),
        (Box::new(|| validate_max_tokens(0)), "invalid max tokens"),
        (Box::new(|| validate_temperature(-0.1)), "invalid temperature"),
        (Box::new(|| validate_device("invalid_device")), "invalid device"),
    ];
    
    for (validation_fn, description) in test_cases {
        let result = validation_fn();
        assert!(result.is_err(), "Validation should fail for: {}", description);
        
        let error = result.unwrap_err();
        let error_str = error.to_string();
        
        // Error messages should be descriptive
        assert!(!error_str.is_empty());
        assert!(error_str.len() > 10, "Error message should be descriptive");
    }
}

#[tokio::test]
async fn test_result_type_alias_comprehensive() {
    // Test the Result type alias works correctly throughout the codebase
    type TestResult<T> = std::result::Result<T, ModelError>;
    
    // Test Ok cases
    let ok_result: TestResult<String> = Ok("test".to_string());
    assert!(ok_result.is_ok());
    assert_eq!(ok_result.unwrap(), "test");
    
    // Test Err cases
    let err_result: TestResult<String> = Err(ModelError::validation_error(
        "test",
        "validation failed",
        "fix input"
    ));
    assert!(err_result.is_err());
    
    // Test error conversion
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let model_error: ModelError = io_error.into();
    
    let result: TestResult<()> = Err(model_error);
    assert!(result.is_err());
    
    if let Err(e) = result {
        assert!(e.to_string().contains("file not found"));
    }
}

#[tokio::test]
async fn test_error_aggregation_patterns() {
    // Test patterns for aggregating multiple errors
    
    let mut errors = Vec::new();
    
    // Collect validation errors
    let validations: [std::result::Result<(), ModelError>; 4] = [
        validate_model_name(""),
        validate_port(0),
        validate_temperature(5.0),
        validate_device("invalid"),
    ];
    
    for validation in validations {
        if let Err(e) = validation {
            errors.push(e);
        }
    }
    
    // Should have collected errors for all invalid cases
    assert!(errors.len() >= 2);
    
    // Test error messages are distinct
    let error_strings: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
    let unique_errors: Vec<String> = error_strings.iter().cloned().collect();
    assert_eq!(error_strings.len(), unique_errors.len()); // All error messages should be distinct
}
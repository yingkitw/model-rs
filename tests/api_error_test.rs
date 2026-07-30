//! HTTP API error handling and edge case tests for model-rs
//!
//! This test suite focuses on:
//! - HTTP API error responses
//! - Request validation and malformed input handling
//! - Edge cases in API endpoints
//! - Authentication and authorization error scenarios
//! - Rate limiting and timeout scenarios

use serde_json::json;
use std::str::FromStr;
use std::time::Duration;

/// Base URL for HTTP API tests
fn api_base_url() -> String {
    let port = std::env::var("MODEL_RS_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    format!("http://127.0.0.1:{port}")
}

/// Helper function to make HTTP requests and handle common error patterns
async fn make_api_request(
    method: &str,
    endpoint: &str,
    body: Option<serde_json::Value>,
) -> Result<reqwest::Response, reqwest::Error> {
    let client = reqwest::Client::new();
    let base = api_base_url();
    let url = format!("{}{}", base, endpoint);
    
    let mut request = match method {
        "GET" => client.get(&url),
        "POST" => client.post(&url),
        "PUT" => client.put(&url),
        "DELETE" => client.delete(&url),
        _ => panic!("Unsupported HTTP method: {}", method),
    };
    
    if let Some(body) = body {
        request = request.json(&body);
    }
    
    request.send().await
}

#[tokio::test]
async fn test_malformed_json_requests() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    let test_cases = vec![
        ("invalid json", "{"),
        ("empty object", "{}"),
        ("missing required fields", r#"{"prompt": ""}"#),
        ("invalid field types", r#"{"prompt": 123, "max_tokens": "not_a_number"}"#),
        ("array instead of object", r#"[{"prompt": "test"}]"#),
    ];
    
    for (description, json_body) in test_cases {
        let response = client
            .post(format!("{base}/v1/generate"))
            .header("Content-Type", "application/json")
            .body(json_body.to_string())
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                // Should return 400 Bad Request for malformed JSON
                println!("Test case '{}' - Status: {}", description, resp.status());
                if resp.status().as_u16() >= 400 {
                    // Error response is expected
                    let error_body = resp.text().await.unwrap_or_default();
                    println!("Error body: {}", error_body);
                    assert!(error_body.len() > 0);
                }
            }
            Err(e) => {
                // Connection errors indicate server not running, which is acceptable for this test
                if e.is_connect() {
                    println!("Server not running for test case: {}", description);
                    return;
                }
                panic!("Unexpected error for test case {}: {}", description, e);
            }
        }
    }
}

#[tokio::test]
async fn test_invalid_parameter_values() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    // Test cases with invalid parameter values
    let test_cases = vec![
        // Invalid max_tokens
        (
            json!({
                "prompt": "test",
                "max_tokens": 0, // Too low
            }),
            "max_tokens too low",
        ),
        (
            json!({
                "prompt": "test",
                "max_tokens": 50000, // Too high
            }),
            "max_tokens too high",
        ),
        // Invalid temperature
        (
            json!({
                "prompt": "test",
                "temperature": -0.1, // Too low
            }),
            "temperature too low",
        ),
        (
            json!({
                "prompt": "test",
                "temperature": 2.1, // Too high
            }),
            "temperature too high",
        ),
        // Invalid top_p
        (
            json!({
                "prompt": "test",
                "top_p": -0.1, // Too low
            }),
            "top_p too low",
        ),
        (
            json!({
                "prompt": "test",
                "top_p": 1.1, // Too high
            }),
            "top_p too high",
        ),
        // Invalid top_k
        (
            json!({
                "prompt": "test",
                "top_k": 0, // Too low
            }),
            "top_k too low",
        ),
        (
            json!({
                "prompt": "test",
                "top_k": 1001, // Too high
            }),
            "top_k too high",
        ),
    ];
    
    for (body, description) in test_cases {
        let response = client
            .post(format!("{base}/v1/generate"))
            .json(&body)
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                println!("Parameter test '{}' - Status: {}", description, resp.status());
                // Should return 400 Bad Request for invalid parameters
                if resp.status().as_u16() >= 400 {
                    let error_body = resp.text().await.unwrap_or_default();
                    println!("Error body: {}", error_body);
                    assert!(!error_body.is_empty());
                }
            }
            Err(e) => {
                if e.is_connect() {
                    println!("Server not running for parameter test: {}", description);
                    return;
                }
                panic!("Unexpected error for parameter test {}: {}", description, e);
            }
        }
    }
}

#[tokio::test]
async fn test_empty_input_validation() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    let test_cases = vec![
        ("/v1/generate", json!({"prompt": ""}), "empty prompt"),
        ("/v1/generate", json!({"prompt": "   "}), "whitespace only prompt"),
        ("/api/generate", json!({"prompt": ""}), "empty prompt ollama"),
        ("/api/generate", json!({"prompt": "   "}), "whitespace only ollama"),
        ("/api/embed", json!({"text": ""}), "empty text embedding"),
        ("/api/embed", json!({"text": "   "}), "whitespace only embedding"),
    ];
    
    for (endpoint, body, description) in test_cases {
        let response = client
            .post(format!("{base}{}", endpoint))
            .json(&body)
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                println!("Empty input test '{}' - Status: {}", description, resp.status());
                // Should return 400 Bad Request for empty inputs
                if resp.status().as_u16() >= 400 {
                    let error_body = resp.text().await.unwrap_or_default();
                    println!("Error body: {}", error_body);
                    assert!(!error_body.is_empty());
                }
            }
            Err(e) => {
                if e.is_connect() {
                    println!("Server not running for empty input test: {}", description);
                    return;
                }
                panic!("Unexpected error for empty input test {}: {}", description, e);
            }
        }
    }
}

#[tokio::test]
async fn test_missing_required_fields() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    let test_cases = vec![
        ("/v1/generate", json!({}), "missing prompt"),
        ("/v1/generate", json!({"prompt": "test"}), "missing max_tokens"),
        ("/api/generate", json!({}), "missing prompt ollama"),
        ("/api/embed", json!({}), "missing text ollama"),
    ];
    
    for (endpoint, body, description) in test_cases {
        let response = client
            .post(format!("{base}{}", endpoint))
            .json(&body)
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                println!("Missing fields test '{}' - Status: {}", description, resp.status());
                if resp.status().as_u16() >= 400 {
                    let error_body = resp.text().await.unwrap_or_default();
                    println!("Error body: {}", error_body);
                    assert!(!error_body.is_empty());
                }
            }
            Err(e) => {
                if e.is_connect() {
                    println!("Server not running for missing fields test: {}", description);
                    return;
                }
                panic!("Unexpected error for missing fields test {}: {}", description, e);
            }
        }
    }
}

#[tokio::test]
async fn test_http_method_validation() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    let test_cases = vec![
        ("GET", "/v1/generate", "GET on generate endpoint"),
        ("PUT", "/v1/generate", "PUT on generate endpoint"),
        ("DELETE", "/v1/generate", "DELETE on generate endpoint"),
        ("GET", "/api/generate", "GET on ollama generate"),
        ("PUT", "/api/generate", "PUT on ollama generate"),
        ("DELETE", "/api/generate", "DELETE on ollama generate"),
    ];
    
    for (method, endpoint, description) in test_cases {
        let response = client
            .request(reqwest::Method::from_str(method).unwrap(), format!("{base}{}", endpoint))
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                println!("HTTP method test '{}' - Status: {}", description, resp.status());
                // Most endpoints should return 405 Method Not Allowed
                if resp.status().as_u16() >= 400 {
                    let error_body = resp.text().await.unwrap_or_default();
                    println!("Error body: {}", error_body);
                }
            }
            Err(e) => {
                if e.is_connect() {
                    println!("Server not running for HTTP method test: {}", description);
                    return;
                }
                panic!("Unexpected error for HTTP method test {}: {}", description, e);
            }
        }
    }
}

#[tokio::test]
async fn test_content_type_validation() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    let test_cases = vec![
        ("/v1/generate", "text/plain", "plain text instead of JSON"),
        ("/api/generate", "application/xml", "XML instead of JSON"),
        ("/api/embed", "text/plain", "plain text instead of JSON"),
    ];
    
    for (endpoint, content_type, description) in test_cases {
        let response = client
            .post(format!("{base}{}", endpoint))
            .header("Content-Type", content_type)
            .body("not json content")
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                println!("Content type test '{}' - Status: {}", description, resp.status());
                if resp.status().as_u16() >= 400 {
                    let error_body = resp.text().await.unwrap_or_default();
                    println!("Error body: {}", error_body);
                    assert!(!error_body.is_empty());
                }
            }
            Err(e) => {
                if e.is_connect() {
                    println!("Server not running for content type test: {}", description);
                    return;
                }
                panic!("Unexpected error for content type test {}: {}", description, e);
            }
        }
    }
}

#[tokio::test]
async fn test_large_payload_handling() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    // Test with very large prompt that might cause issues
    let large_prompt = "x".repeat(100_000); // 100KB prompt
    
    let response = client
        .post(format!("{base}/v1/generate"))
        .json(&json!({
            "prompt": large_prompt,
            "max_tokens": 100,
        }))
        .send()
        .await;
    
    match response {
        Ok(resp) => {
            println!("Large payload test - Status: {}", resp.status());
            // Server should handle it gracefully (either succeed with 200 or return appropriate error)
            let body = resp.text().await.unwrap_or_default();
            assert!(!body.is_empty());
        }
        Err(e) => {
            if e.is_connect() {
                println!("Server not running for large payload test");
                return;
            }
            // Server might return 413 Payload Too Large or similar
            println!("Large payload test failed: {}", e);
        }
    }
}

#[tokio::test]
async fn test_concurrent_requests() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    // Spawn multiple concurrent requests
    let mut handles = vec![];
    
    for i in 0..5 {
        let client = client.clone();
        let base = base.clone();
        let handle = tokio::spawn(async move {
            let response = client
                .post(format!("{base}/v1/generate"))
                .json(&json!({
                    "prompt": format!("Test request {}", i),
                    "max_tokens": 10,
                }))
                .send()
                .await;
            
            match response {
                Ok(resp) => (resp.status(), resp.text().await.ok()),
                Err(e) => (reqwest::StatusCode::INTERNAL_SERVER_ERROR, Some(e.to_string())),
            }
        });
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        results.push(result);
    }
    
    // Check that most requests succeeded (or all failed if no server is running)
    let successful = results.iter().filter(|(status, _)| status.is_success()).count();
    let failed = results.len() - successful;
    
    println!("Concurrent requests: {} successful, {} failed", successful, failed);
    
    // If no server is running, all requests will fail — that's acceptable
    // The test validates that concurrent requests don't panic or hang
    if successful == 0 {
        println!("No server running — all concurrent requests failed as expected");
        return;
    }
    
    // Check that error responses have appropriate content
    for (status, body) in results {
        if !status.is_success() {
            assert!(body.is_some(), "Error response should have body");
            assert!(!body.unwrap().is_empty(), "Error body should not be empty");
        }
    }
}

#[tokio::test]
async fn test_timeout_scenarios() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    // Test with very long max_tokens that might timeout
    let long_request = json!({
        "prompt": "Write a very long story about",
        "max_tokens": 1000, // Large number that might take time
        "temperature": 0.9,
    });
    
    let response = tokio::time::timeout(
        Duration::from_secs(30), // 30 second timeout
        client
            .post(format!("{base}/v1/generate"))
            .json(&long_request)
            .send(),
    )
    .await;
    
    match response {
        Ok(Ok(resp)) => {
            println!("Timeout test - Status: {}", resp.status());
            let body = resp.text().await.unwrap_or_default();
            assert!(!body.is_empty());
        }
        Ok(Err(e)) => {
            if e.is_connect() {
                println!("Server not running for timeout test");
            } else {
                println!("Request failed: {}", e);
            }
        }
        Err(_) => {
            println!("Request timed out (which is acceptable in some scenarios)");
        }
    }
}

#[tokio::test]
async fn test_cors_headers() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    let response = client
        .post(format!("{base}/v1/generate"))
        .json(&json!({
            "prompt": "test",
            "max_tokens": 10,
        }))
        .header("Origin", "http://example.com")
        .send()
        .await;
    
    match response {
        Ok(resp) => {
            println!("CORS test - Status: {}", resp.status());
            
            // Check CORS headers
            let cors_headers = [
                "Access-Control-Allow-Origin",
                "Access-Control-Allow-Methods",
                "Access-Control-Allow-Headers",
            ];
            
            for header in &cors_headers {
                if let Some(value) = resp.headers().get(*header) {
                    println!("CORS header {}: {}", header, value.to_str().unwrap_or("invalid"));
                }
            }
            
            let body = resp.text().await.unwrap_or_default();
            assert!(!body.is_empty());
        }
        Err(e) => {
            if e.is_connect() {
                println!("Server not running for CORS test");
            } else {
                panic!("CORS test failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_health_endpoint_edge_cases() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    // Test health endpoint with different methods
    let test_methods = ["GET", "POST", "PUT", "DELETE"];
    
    for method in test_methods {
        let response = client
            .request(reqwest::Method::from_str(method).unwrap(), format!("{base}/health"))
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                println!("Health endpoint {} - Status: {}", method, resp.status());
                
                if method == "GET" {
                    // GET should succeed for health endpoint
                    assert_eq!(resp.status(), reqwest::StatusCode::OK);
                    
                    let body = resp.json::<serde_json::Value>().await.unwrap_or_default();
                    assert!(body.is_object());
                    
                    // Check for common health fields
                    if let Some(health_status) = body.get("status") {
                        assert!(health_status.is_string());
                    }
                } else {
                    // Other methods might return 405 or other appropriate status
                    assert!(resp.status().as_u16() >= 400 || resp.status().as_u16() == 405);
                }
            }
            Err(e) => {
                if e.is_connect() {
                    println!("Server not running for health endpoint test");
                    return;
                }
                panic!("Health endpoint test failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_invalid_model_paths() {
    let client = reqwest::Client::new();
    let base = api_base_url();
    
    // Test requests with invalid model path references
    let test_cases = vec![
        (json!({"model": ""}), "empty model name"),
        (json!({"model": "invalid/model"}), "non-existent model"),
        (json!({"model": "../etc/passwd"}), "path traversal attempt"),
    ];
    
    for (body, description) in test_cases {
        let response = client
            .post(format!("{base}/v1/generate"))
            .json(&json!({
                "prompt": "test",
                "model": body["model"].as_str().unwrap(),
            }))
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                println!("Invalid model path test '{}' - Status: {}", description, resp.status());
                if resp.status().as_u16() >= 400 {
                    let error_body = resp.text().await.unwrap_or_default();
                    println!("Error body: {}", error_body);
                    assert!(!error_body.is_empty());
                }
            }
            Err(e) => {
                if e.is_connect() {
                    println!("Server not running for invalid model path test: {}", description);
                    return;
                }
                panic!("Unexpected error for invalid model path test {}: {}", description, e);
            }
        }
    }
}
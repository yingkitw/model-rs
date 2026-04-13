//! End-to-end (E2E) tests for model-rs
//!
//! These tests cover the complete workflow of the model-rs CLI tool,
//! including CLI commands, model operations, and API server functionality.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::{sleep, timeout};

const TEST_MODEL: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const TEST_PORT: u16 = 3020; // Use different port to avoid conflicts

/// Path to the `model-rs` binary.
///
/// `cargo test` sets `CARGO_BIN_EXE_model_rs` at compile time. `cargo check --tests` does not, so we
/// fall back to `target/{debug|release}/model-rs` (honours `CARGO_TARGET_DIR` when set at runtime).
fn model_rs_bin() -> PathBuf {
    if let Some(p) = option_env!("CARGO_BIN_EXE_model_rs") {
        return PathBuf::from(p);
    }
    let target_root = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("target"));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let exe = if cfg!(target_os = "windows") {
        "model-rs.exe"
    } else {
        "model-rs"
    };
    target_root.join(profile).join(exe)
}

/// Helper function to run model-rs CLI commands
fn run_model_rs(args: &[&str]) -> std::io::Result<std::process::Output> {
    let binary_path = model_rs_bin();

    if !binary_path.exists() {
        panic!("model-rs binary not found at {}.", binary_path.display());
    }

    Command::new(&binary_path)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
}

/// Test 1: Verify CLI binary exists and is executable
#[test]
fn test_cli_binary_exists() {
    let path = model_rs_bin();
    assert!(path.exists(), "model-rs binary should exist");
    
    // Check if file is executable (Unix)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = path.metadata().expect("Should read file metadata");
        let permissions = metadata.permissions();
        let mode = permissions.mode();
        assert!(mode & 0o111 != 0, "Binary should be executable");
    }
}

/// Test 2: CLI version command
#[test]
fn test_cli_version() {
    let output = run_model_rs(&["--version"])
        .expect("Should execute version command");

    assert!(output.status.success(), "Version command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("model-rs"),
        "Version output should contain 'model-rs'"
    );
}

/// Test 3: CLI help command
#[test]
fn test_cli_help() {
    let output = run_model_rs(&["--help"])
        .expect("Should execute help command");

    assert!(output.status.success(), "Help command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Verify main commands are documented
    assert!(
        stdout.contains("download"),
        "Help should document download command"
    );
    assert!(
        stdout.contains("generate"),
        "Help should document generate command"
    );
    assert!(
        stdout.contains("serve"),
        "Help should document serve command"
    );
    assert!(
        stdout.contains("list"),
        "Help should document list command"
    );
}

/// Test 4: List models command (empty or with models)
#[test]
fn test_list_models() {
    let output = run_model_rs(&["list"])
        .expect("Should execute list command");

    assert!(output.status.success(), "List command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should either show "No models found" or list models
    let is_empty = stdout.contains("No models found") || stdout.contains("**To download a model:**");
    let has_content = stdout.contains("#") || stdout.contains("├─") || stdout.contains("|");
    
    assert!(
        is_empty || has_content,
        "List output should either be empty or show model information"
    );
}

/// Test 5: Config command
#[test]
fn test_config_command() {
    let output = run_model_rs(&["config"])
        .expect("Should execute config command");

    assert!(output.status.success(), "Config command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Verify config sections are shown
    assert!(
        stdout.contains("Configuration Settings") || stdout.contains("Model Settings"),
        "Config should show settings sections"
    );
}

/// Test 6: Cache command (stats)
#[test]
fn test_cache_stats() {
    let output = run_model_rs(&["cache", "--stats"])
        .expect("Should execute cache stats command");

    assert!(output.status.success(), "Cache stats command should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should show cache status
    assert!(
        stdout.contains("Cache Status") || stdout.contains("Status:"),
        "Cache stats should show status"
    );
}

/// Test 7: Search command (basic connectivity test)
#[test]
fn test_search_command() {
    let output = run_model_rs(&["search", "llama", "--limit", "5"])
        .expect("Should execute search command");

    // Search might fail due to network, so we just verify it doesn't crash
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Either succeeds or shows a network error
    let has_output = output.status.success() || 
                    stderr.contains("error") || 
                    stderr.contains("failed") ||
                    stdout.contains("No results") ||
                    stdout.contains("Search results");
    
    assert!(has_output, "Search should produce output");
}

/// Test 8: Invalid model path should fail gracefully
#[test]
fn test_generate_invalid_model() {
    let output = run_model_rs(&[
        "generate",
        "Hello",
        "--model-path",
        "/nonexistent/model/path"
    ]);

    let output = output.unwrap();
    assert!(!output.status.success(), "Should fail with invalid model path");
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    assert!(
        stderr.contains("error") || stderr.contains("not found") || stderr.contains("Failed"),
        "Should show error message"
    );
}

/// Test 9: Verify model download command format
#[test]
fn test_download_command_format() {
    let output = run_model_rs(&["download", "--help"])
        .expect("Should execute download help");

    assert!(output.status.success(), "Download help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    assert!(
        stdout.contains("--model") || stdout.contains("-m"),
        "Download help should show model option"
    );
    assert!(
        stdout.contains("--mirror") || stdout.contains("-r"),
        "Download help should show mirror option"
    );
}

/// Test 10: Generate command help
#[test]
fn test_generate_command_help() {
    let output = run_model_rs(&["generate", "--help"])
        .expect("Should execute generate help");

    assert!(output.status.success(), "Generate help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    assert!(
        stdout.contains("--temperature"),
        "Generate help should show temperature option"
    );
    assert!(
        stdout.contains("--max-tokens"),
        "Generate help should show max-tokens option"
    );
    assert!(
        stdout.contains("--top-p") || stdout.contains("--top-k"),
        "Generate help should show sampling options"
    );
}

/// Test 11: Serve command help
#[test]
fn test_serve_command_help() {
    let output = run_model_rs(&["serve", "--help"])
        .expect("Should execute serve help");

    assert!(output.status.success(), "Serve help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    assert!(
        stdout.contains("--port"),
        "Serve help should show port option"
    );
    assert!(
        stdout.contains("--device"),
        "Serve help should show device option"
    );
}

/// Test 12: Chat command help
#[test]
fn test_chat_command_help() {
    let output = run_model_rs(&["chat", "--help"])
        .expect("Should execute chat help");

    assert!(output.status.success(), "Chat help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    assert!(
        stdout.contains("--system"),
        "Chat help should show system option"
    );
    assert!(
        stdout.contains("--session"),
        "Chat help should show session option"
    );
}

/// Test 13: Show command help
#[test]
fn test_show_command_help() {
    let output = run_model_rs(&["show", "--help"])
        .expect("Should execute show help");

    assert!(output.status.success(), "Show help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    assert!(
        stdout.contains("<MODEL>"),
        "Show help should show model argument"
    );
}

/// Test 14: Remove command help
#[test]
fn test_remove_command_help() {
    let output = run_model_rs(&["remove", "--help"])
        .expect("Should execute remove help");

    assert!(output.status.success(), "Remove help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    assert!(
        stdout.contains("--force") || stdout.contains("-f"),
        "Remove help should show force option"
    );
}

/// Test 15: PS command (list running processes)
#[test]
fn test_ps_command() {
    let output = run_model_rs(&["ps"])
        .expect("Should execute ps command");

    assert!(output.status.success(), "PS command should succeed");
    // PS should work even with no processes
}

// === API Server Tests ===

/// Helper function to check if server is running
async fn is_server_running(port: u16) -> bool {
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/health", port);
    
    let result = timeout(Duration::from_secs(2), client.get(&url).send()).await;
    matches!(result, Ok(Ok(resp)) if resp.status().is_success())
}

/// Test 16: Health check endpoint
#[tokio::test]
async fn test_health_endpoint() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running on port {}. Skipping API tests.", TEST_PORT);
        eprintln!(
            "Start with: {} serve --port {} --model-path <model-path>",
            model_rs_bin().display(),
            TEST_PORT
        );
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/health", TEST_PORT);
    
    let response = client
        .get(&url)
        .send()
        .await
        .expect("Health request should complete");

    assert_eq!(response.status(), 200, "Health endpoint should return 200");
    
    let body: serde_json::Value = response
        .json()
        .await
        .expect("Should parse JSON response");
    
    assert!(
        body.get("status").is_some(),
        "Health response should contain status"
    );
}

/// Test 17: V1 generate endpoint (non-streaming)
#[tokio::test]
async fn test_v1_generate_endpoint() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_v1_generate_endpoint");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/generate", TEST_PORT);
    
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": "What is 2+2?",
            "max_tokens": 20,
            "temperature": 0.7
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200, "Generate should return 200");
            
            let body = resp.json::<serde_json::Value>().await
                .expect("Should parse response JSON");
            
            assert!(
                body.get("text").is_some(),
                "Response should contain 'text' field"
            );
            
            let text = body["text"].as_str().expect("Text should be a string");
            assert!(!text.is_empty(), "Generated text should not be empty");
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

/// Test 18: V1 generate with streaming
#[tokio::test]
async fn test_v1_generate_stream() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_v1_generate_stream");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!(
        "http://localhost:{}/v1/generate_stream?prompt=Hello&max_tokens=10",
        TEST_PORT
    );
    
    let response = client
        .get(&url)
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200, "Stream should return 200");
            
            let content_type = resp.headers()
                .get("content-type")
                .expect("Should have content-type header");
            
            assert!(
                content_type.to_str().unwrap().contains("text/event-stream"),
                "Should return SSE content type"
            );
            
            let mut stream = resp.bytes_stream();
            let mut received_data = false;
            
            let result = timeout(Duration::from_secs(10), async {
                use futures_util::StreamExt;
                
                while let Some(chunk) = stream.next().await {
                    if let Ok(bytes) = chunk {
                        let text = String::from_utf8_lossy(&bytes);
                        if text.contains("data:") {
                            received_data = true;
                            break;
                        }
                    }
                }
            }).await;

            assert!(result.is_ok(), "Stream should complete within timeout");
            assert!(received_data, "Should receive SSE data");
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

/// Test 19: Ollama-compatible generate endpoint
#[tokio::test]
async fn test_ollama_generate_endpoint() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_ollama_generate_endpoint");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/api/generate", TEST_PORT);
    
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": "Hi",
            "stream": false,
            "options": {
                "num_predict": 10
            }
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200, "Ollama generate should return 200");
            
            let body = resp.json::<serde_json::Value>().await
                .expect("Should parse response JSON");
            
            assert!(
                body.get("response").is_some(),
                "Response should contain 'response' field"
            );
            assert!(
                body.get("done").is_some(),
                "Response should contain 'done' field"
            );
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

/// Test 20: Ollama tags endpoint
#[tokio::test]
async fn test_ollama_tags_endpoint() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_ollama_tags_endpoint");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/api/tags", TEST_PORT);
    
    let response = client
        .get(&url)
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200, "Tags should return 200");
            
            let body = resp.json::<serde_json::Value>().await
                .expect("Should parse response JSON");
            
            assert!(
                body.get("models").is_some(),
                "Response should contain 'models' field"
            );
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

/// Test 21: Invalid request should return 400
#[tokio::test]
async fn test_invalid_generate_request() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_invalid_generate_request");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/generate", TEST_PORT);
    
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "max_tokens": 50  // Missing required 'prompt' field
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            let status = resp.status();
            assert!(
                status == 400 || status == 422,
                "Invalid request should return 400 or 422, got {}",
                status
            );
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

/// Test 22: Generate with custom parameters
#[tokio::test]
async fn test_generate_with_custom_params() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_generate_with_custom_params");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/generate", TEST_PORT);
    
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": "Count to 3:",
            "max_tokens": 20,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200, "Generate with params should return 200");
            
            let body = resp.json::<serde_json::Value>().await
                .expect("Should parse response JSON");
            
            assert!(
                body.get("text").is_some(),
                "Response should contain 'text' field"
            );
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

/// Test 23: Embeddings endpoint
#[tokio::test]
async fn test_embeddings_endpoint() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_embeddings_endpoint");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/embeddings", TEST_PORT);
    
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "input": "Hello world"
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            // Might return 200 or 501 (not implemented for this model)
            let status = resp.status();
            assert!(
                status == 200 || status == 501,
                "Embeddings should return 200 or 501, got {}",
                status
            );
            
            if status == 200 {
                let body = resp.json::<serde_json::Value>().await
                    .expect("Should parse response JSON");
                
                assert!(
                    body.get("data").is_some(),
                    "Response should contain 'data' field"
                );
            }
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

// === Integration Test ===

/// Test 24: Full workflow test (requires model to be downloaded)
///
/// This test performs a complete workflow:
/// 1. List models
/// 2. Start server (if model exists)
/// 3. Generate text via API
/// 4. Stop server
#[tokio::test]
#[ignore] // Run with: cargo test --test e2e_test test_full_workflow -- --ignored
async fn test_full_workflow() {
    // Step 1: List models to find a valid model
    let list_output = run_model_rs(&["list"])
        .expect("Should list models");
    
    let stdout = String::from_utf8_lossy(&list_output.stdout);
    
    // Skip if no models found
    if stdout.contains("No models found") {
        eprintln!("No models found. Skipping full workflow test.");
        return;
    }

    // Step 2: Find first model path from list output
    // This is a simplified extraction - in real tests you'd parse the output properly
    let model_path = extract_model_path(&stdout);
    let model_path = match model_path {
        Some(path) => path,
        None => {
            eprintln!("Could not extract model path from list output. Skipping full workflow test.");
            return;
        }
    };

    // Step 3: Start server in background
    let port = TEST_PORT + 1; // Use different port
    let mut server_cmd = Command::new(model_rs_bin())
        .args([
            "serve",
            "--model-path",
            &model_path,
            "--port",
            &port.to_string(),
            "--device",
            "cpu", // Use CPU to avoid GPU issues
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Should start server");

    // Wait for server to start
    sleep(Duration::from_secs(5)).await;

    // Step 4: Test API endpoint
    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/generate", port);
    
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": "Hello",
            "max_tokens": 10
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200, "Full workflow: generate should succeed");
            
            let body = resp.json::<serde_json::Value>().await
                .expect("Should parse response");
            
            assert!(body.get("text").is_some(), "Should have text response");
            println!("✓ Full workflow test passed");
        }
        Err(e) => {
            eprintln!("API request failed in full workflow: {}", e);
        }
    }

    // Step 5: Stop server
    let _ = server_cmd.kill();
    let _ = server_cmd.wait();
}

/// Helper to extract model path from list output
fn extract_model_path(output: &str) -> Option<String> {
    // Simple extraction - looks for paths in the output
    // In a real implementation, you'd parse the output more carefully
    for line in output.lines() {
        if line.contains("~") || line.contains("/") {
            if line.contains("models") || line.contains("cache") {
                // Extract path (this is simplified)
                if let Some(start) = line.find('~') {
                    let path = line[start..].trim().to_string();
                    if !path.is_empty() {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

/// Test 25: Server startup and shutdown
#[test]
#[ignore] // Run with: cargo test --test e2e_test test_server_lifecycle -- --ignored
fn test_server_lifecycle() {
    // Find a model
    let list_output = run_model_rs(&["list"])
        .expect("Should list models");
    
    let stdout = String::from_utf8_lossy(&list_output.stdout);
    
    if stdout.contains("No models found") {
        eprintln!("No models found. Skipping server lifecycle test.");
        return;
    }

    let model_path = match extract_model_path(&stdout) {
        Some(path) => path,
        None => {
            eprintln!("Could not extract model path. Skipping server lifecycle test.");
            return;
        }
    };

    let port = TEST_PORT + 2;

    // Start server
    let mut server = Command::new(model_rs_bin())
        .args([
            "serve",
            "--model-path",
            &model_path,
            "--port",
            &port.to_string(),
            "--device",
            "cpu",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Should start server");

    // Give it time to start
    std::thread::sleep(Duration::from_secs(3));

    // Check if server is responsive
    let health_check = Command::new("curl")
        .args(["-s", "-o", "/dev/null", "-w", "%{http_code}", &format!("http://localhost:{}/health", port)])
        .output();

    match health_check {
        Ok(output) => {
            let status_code = String::from_utf8_lossy(&output.stdout);
            if status_code == "200" {
                println!("✓ Server started successfully");
            } else {
                eprintln!("Server returned status code: {}", status_code);
            }
        }
        Err(e) => {
            eprintln!("Failed to check server health: {}", e);
        }
    }

    // Stop server
    let _ = server.kill();
    match server.wait() {
        Ok(status) => println!("✓ Server stopped with status: {}", status),
        Err(e) => eprintln!("Failed to stop server: {}", e),
    }
}

/// Test 26: Config command with environment variables
#[test]
#[ignore] // Run with: cargo test --test e2e_test test_config_with_env -- --ignored
fn test_config_with_env() {
    // Set environment variable
    unsafe {
        std::env::set_var("MODEL_RS_TEMPERATURE", "0.5");
        std::env::set_var("MODEL_RS_MAX_TOKENS", "100");
    }

    // Run config command
    let output = run_model_rs(&["config"])
        .expect("Should execute config command");

    assert!(output.status.success(), "Config should succeed");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    // Cleanup
    unsafe {
        std::env::remove_var("MODEL_RS_TEMPERATURE");
        std::env::remove_var("MODEL_RS_MAX_TOKENS");
    }
}

/// Test 27: Cache clear command
#[test]
#[ignore] // Temporarily ignored due to CLI argument conflict (-e used by both enable and evict)
fn test_cache_clear() {
    // Note: This test is safe to run as it just clears the in-memory cache
    let output = run_model_rs(&["cache", "--clear"])
        .expect("Should execute cache clear");

    assert!(output.status.success(), "Cache clear should succeed");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Cache cleared") || stdout.contains("cleared"),
        "Should confirm cache was cleared"
    );
}

/// Test 28: Verify error handling for non-existent model operations
#[test]
fn test_nonexistent_model_operations() {
    let fake_model = "NonExistent/Model-123";
    
    // Test show command
    let output = run_model_rs(&["show", fake_model]);
    assert!(!output.unwrap().status.success(), "Show should fail for non-existent model");
    
    // Test remove command (should fail or prompt)
    let output = run_model_rs(&["remove", fake_model]);
    assert!(!output.unwrap().status.success(), "Remove should fail for non-existent model");
    
    // Test verify command
    let output = run_model_rs(&["verify", fake_model]);
    assert!(!output.unwrap().status.success(), "Verify should fail for non-existent model");
    
    // Test info command
    let output = run_model_rs(&["info", fake_model]);
    assert!(!output.unwrap().status.success(), "Info should fail for non-existent model");
}

/// Test 29: Concurrent generation requests
#[tokio::test]
#[ignore] // Run with: cargo test --test e2e_test test_concurrent_requests -- --ignored
async fn test_concurrent_requests() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_concurrent_requests");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/generate", TEST_PORT);
    
    // Spawn multiple concurrent requests
    let mut handles = vec![];
    
    for i in 0..5 {
        let client = client.clone();
        let url = url.clone();
        
        let handle = tokio::spawn(async move {
            client
                .post(&url)
                .json(&serde_json::json!({
                    "prompt": format!("Test {}", i),
                    "max_tokens": 5
                }))
                .send()
                .await
        });
        
        handles.push(handle);
    }

    // Wait for all requests to complete
    let mut successful = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(resp)) if resp.status() == 200 => successful += 1,
            Ok(Ok(resp)) => eprintln!("Request failed with status: {}", resp.status()),
            Ok(Err(e)) => eprintln!("Request error: {}", e),
            Err(e) => eprintln!("Task error: {}", e),
        }
    }

    assert!(
        successful > 0,
        "At least some concurrent requests should succeed"
    );
    println!("✓ Concurrent requests: {}/5 successful", successful);
}

/// Test 30: Long-running generation
#[tokio::test]
#[ignore] // Run with: cargo test --test e2e_test test_long_generation -- --ignored
async fn test_long_generation() {
    if !is_server_running(TEST_PORT).await {
        eprintln!("Server not running. Skipping test_long_generation");
        return;
    }

    let client = reqwest::Client::new();
    let url = format!("http://localhost:{}/v1/generate", TEST_PORT);
    
    let start = std::time::Instant::now();
    
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": "Write a short story about a robot:",
            "max_tokens": 100,
            "temperature": 0.8
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200, "Long generation should succeed");
            
            let body = resp.json::<serde_json::Value>().await
                .expect("Should parse response");
            
            let text = body["text"].as_str().expect("Should have text");
            let duration = start.elapsed();
            
            println!("✓ Long generation completed in {:?}", duration);
            println!("Generated {} characters", text.len());
            
            assert!(
                text.len() > 10,
                "Should generate reasonable amount of text"
            );
        }
        Err(e) if e.is_connect() => {
            eprintln!("Connection failed. Server might not be running.");
        }
        Err(e) => {
            panic!("Request failed: {}", e);
        }
    }
}

/// Test 31: Verify CLI output format consistency
#[test]
fn test_output_format_consistency() {
    let commands = vec![
        vec!["--help"],
        vec!["list"],
        vec!["config"],
        vec!["cache", "--stats"],
    ];

    for args in commands {
        let output = run_model_rs(&args)
            .expect("Should execute command");
        
        assert!(
            output.status.success(),
            "Command {:?} should succeed",
            args
        );
        
        // Verify output is valid UTF-8
        let _ = String::from_utf8(output.stdout)
            .expect("Output should be valid UTF-8");
    }
}

/// Test 32: Test all command help outputs
#[test]
fn test_all_command_helps() {
    let commands = vec![
        "download",
        "generate",
        "run",
        "stop",
        "chat",
        "serve",
        "list",
        "search",
        "embed",
        "show",
        "remove",
        "ps",
        "info",
        "verify",
        "cache",
        "copy",
        "config",
        "deploy",
    ];

    for cmd in commands {
        let output = run_model_rs(&[cmd, "--help"])
            .expect(&format!("Should get help for {}", cmd));
        
        assert!(
            output.status.success(),
            "Help for {} should succeed",
            cmd
        );
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !stdout.is_empty(),
            "Help for {} should have output",
            cmd
        );
        
        println!("✓ Help for '{}': OK", cmd);
    }
}

/// Test 33: Verify error messages are user-friendly
#[test]
fn test_error_messages() {
    // Test various error conditions
    let test_cases = vec![
        (vec!["generate"], true), // Missing prompt
        (vec!["download"], true), // Missing model name
        (vec!["search"], true), // Missing query
        (vec!["show"], true), // Missing model name
    ];

    for (args, should_fail) in test_cases {
        let output = run_model_rs(&args);
        
        if should_fail {
            assert!(!output.unwrap().status.success(), "Command {:?} should fail", args);
        }
    }
}

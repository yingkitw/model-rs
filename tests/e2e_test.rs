//! End-to-end (E2E) tests for model-rs
//!
//! These tests cover the complete workflow of the model-rs CLI tool,
//! including CLI commands, model operations, and API server functionality.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;
use tokio::time::{sleep, timeout};

/// Suggested HF id when no local weights are available (hint text only).
const TEST_MODEL: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const TEST_PORT: u16 = 3020; // Use different port to avoid conflicts

/// Path to the `model-rs` binary.
///
/// Prefer the runtime `CARGO_BIN_EXE_*` value Cargo sets when running integration tests (avoids a
/// stale absolute path baked in via `option_env!` after moving/renaming the repo). Fall back to
/// `target/{debug|release}/model-rs` (honours `CARGO_TARGET_DIR` when set).
fn model_rs_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_model_rs") {
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

/// Run `model-rs` with extra environment variables (isolates tests from the parent process).
fn run_model_rs_with_env(
    args: &[&str],
    envs: &[(&str, &str)],
) -> std::io::Result<std::process::Output> {
    let binary_path = model_rs_bin();
    if !binary_path.exists() {
        panic!("model-rs binary not found at {}.", binary_path.display());
    }
    let mut cmd = Command::new(&binary_path);
    for (key, val) in envs {
        cmd.env(key, val);
    }
    cmd.args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
}

/// Unique port per test that spawns `serve`, so parallel `cargo test` does not collide.
fn alloc_e2e_listen_port() -> u16 {
    static NEXT: AtomicU16 = AtomicU16::new(4_100);
    NEXT.fetch_add(1, Ordering::SeqCst)
}

fn first_model_path_from_list_output(output: &str) -> Option<PathBuf> {
    for raw_line in output.lines() {
        let line = raw_line.trim_start();
        let Some(rest) = line.strip_prefix("- **Path:** `") else {
            continue;
        };
        let Some(end) = rest.find('`') else {
            continue;
        };
        let p = PathBuf::from(&rest[..end]);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Local model directory for tests that need a running server.
///
/// Order: `MODEL_RS_E2E_MODEL_PATH` (must exist), else first path from `model-rs list`.
fn resolve_e2e_model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MODEL_RS_E2E_MODEL_PATH") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return Some(pb);
        }
    }
    let list_output = run_model_rs(&["list"]).ok()?;
    if !list_output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&list_output.stdout);
    first_model_path_from_list_output(&stdout)
}

async fn wait_for_health_port(port: u16, max_wait: Duration) -> bool {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = tokio::time::Instant::now() + max_wait;
    while tokio::time::Instant::now() < deadline {
        if let Ok(resp) = timeout(Duration::from_secs(2), client.get(&url).send()).await {
            if let Ok(resp) = resp {
                if resp.status().is_success() {
                    return true;
                }
            }
        }
        sleep(Duration::from_millis(200)).await;
    }
    false
}

fn spawn_serve_child(model_path: &Path, port: u16) -> std::io::Result<Child> {
    Command::new(model_rs_bin())
        .args([
            "serve",
            "--model-path",
            &model_path.display().to_string(),
            "--port",
            &port.to_string(),
            "--device",
            "cpu",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
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

/// Test 24: Full workflow test (requires a local model directory)
///
/// Uses `MODEL_RS_E2E_MODEL_PATH` or the first model from `model-rs list`. If neither is
/// available, the test returns early so `cargo test` stays green on clean checkouts.
#[tokio::test]
async fn test_full_workflow() {
    let Some(model_path) = resolve_e2e_model_path() else {
        eprintln!(
            "No local model for test_full_workflow: set MODEL_RS_E2E_MODEL_PATH or run `model-rs download {TEST_MODEL}`."
        );
        return;
    };

    let port = alloc_e2e_listen_port();
    let mut server_cmd = spawn_serve_child(&model_path, port).expect("Should start server");

    if !wait_for_health_port(port, Duration::from_secs(120)).await {
        let mut err = String::new();
        if let Some(stderr) = server_cmd.stderr() {
            let _ = stderr.read_to_string(&mut err);
        }
        let _ = server_cmd.kill();
        let _ = server_cmd.wait();
        panic!(
            "Server did not become healthy on port {port}. stderr (if any): {err}"
        );
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/generate");
    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": "Hello",
            "max_tokens": 10
        }))
        .send()
        .await
        .expect("request should complete");

    assert_eq!(
        response.status(),
        200,
        "Full workflow: generate should succeed"
    );

    let body = response
        .json::<serde_json::Value>()
        .await
        .expect("Should parse response");
    assert!(body.get("text").is_some(), "Should have text response");

    let _ = server_cmd.kill();
    let _ = server_cmd.wait();
}

/// Test 25: Server startup and health (no curl; portable)
#[tokio::test]
async fn test_server_lifecycle() {
    let Some(model_path) = resolve_e2e_model_path() else {
        eprintln!(
            "No local model for test_server_lifecycle: set MODEL_RS_E2E_MODEL_PATH or run `model-rs download {TEST_MODEL}`."
        );
        return;
    };

    let port = alloc_e2e_listen_port();
    let mut server = spawn_serve_child(&model_path, port).expect("Should start server");

    assert!(
        wait_for_health_port(port, Duration::from_secs(120)).await,
        "server health check failed on port {port}"
    );

    let _ = server.kill();
    let _ = server.wait();
}

/// Test 26: Config command with environment variables (child process env only)
#[test]
fn test_config_with_env() {
    let output = run_model_rs_with_env(
        &["config"],
        &[
            ("MODEL_RS_TEMPERATURE", "0.5"),
            ("MODEL_RS_MAX_TOKENS", "100"),
        ],
    )
    .expect("Should execute config command");

    assert!(output.status.success(), "Config should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("**Temperature:** `0.5`"),
        "config output should reflect MODEL_RS_TEMPERATURE=0.5, got: {stdout}"
    );
    assert!(
        stdout.contains("**Max Tokens:** `100`"),
        "config output should reflect MODEL_RS_MAX_TOKENS=100, got: {stdout}"
    );
}

/// Test 27: Cache clear command
#[test]
fn test_cache_clear() {
    let output = run_model_rs(&["cache", "--clear"]).expect("Should execute cache clear");

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
async fn test_concurrent_requests() {
    let Some(model_path) = resolve_e2e_model_path() else {
        eprintln!(
            "No local model for test_concurrent_requests: set MODEL_RS_E2E_MODEL_PATH or run `model-rs download {TEST_MODEL}`."
        );
        return;
    };

    let port = alloc_e2e_listen_port();
    let mut server = spawn_serve_child(&model_path, port).expect("Should start server");
    if !wait_for_health_port(port, Duration::from_secs(120)).await {
        let _ = server.kill();
        let _ = server.wait();
        panic!("Server did not become healthy on port {port}");
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/generate");

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

    let mut successful = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(resp)) if resp.status() == 200 => successful += 1,
            Ok(Ok(resp)) => eprintln!("Request failed with status: {}", resp.status()),
            Ok(Err(e)) => eprintln!("Request error: {}", e),
            Err(e) => eprintln!("Task error: {}", e),
        }
    }

    let _ = server.kill();
    let _ = server.wait();

    assert!(
        successful > 0,
        "At least some concurrent requests should succeed"
    );
}

/// Test 30: Longer generation request against a spawned server
#[tokio::test]
async fn test_long_generation() {
    let Some(model_path) = resolve_e2e_model_path() else {
        eprintln!(
            "No local model for test_long_generation: set MODEL_RS_E2E_MODEL_PATH or run `model-rs download {TEST_MODEL}`."
        );
        return;
    };

    let port = alloc_e2e_listen_port();
    let mut server = spawn_serve_child(&model_path, port).expect("Should start server");
    if !wait_for_health_port(port, Duration::from_secs(120)).await {
        let _ = server.kill();
        let _ = server.wait();
        panic!("Server did not become healthy on port {port}");
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/generate");

    let response = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": "Write a short story about a robot:",
            "max_tokens": 80,
            "temperature": 0.8
        }))
        .send()
        .await
        .expect("request should complete");

    assert_eq!(response.status(), 200, "Long generation should succeed");

    let body = response
        .json::<serde_json::Value>()
        .await
        .expect("Should parse response");
    let text = body["text"].as_str().expect("Should have text");
    assert!(
        text.len() > 10,
        "Should generate reasonable amount of text"
    );

    let _ = server.kill();
    let _ = server.wait();
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

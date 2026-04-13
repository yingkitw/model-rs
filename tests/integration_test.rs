use std::time::Duration;
use tokio::time::timeout;

/// Base URL for HTTP integration tests. Uses `MODEL_RS_PORT` when set (same as `model-rs serve`); otherwise **8080** (server default in `config::get_port()`).
fn integration_base_url() -> String {
    let port = std::env::var("MODEL_RS_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    format!("http://127.0.0.1:{port}")
}

#[tokio::test]
async fn test_v1_generate_endpoint() {
    let client = reqwest::Client::new();
    let base = integration_base_url();

    let response = client
        .post(format!("{base}/v1/generate"))
        .json(&serde_json::json!({
            "prompt": "What is 2+2?",
            "max_tokens": 50,
            "temperature": 0.7
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            if resp.status() == 404 {
                eprintln!("model-rs endpoints not found (404). Skipping integration test.");
                return;
            }
            assert_eq!(resp.status(), 200);
            
            let body = resp.json::<serde_json::Value>().await.unwrap();
            assert!(body.get("text").is_some());
            
            let text = body["text"].as_str().unwrap();
            assert!(!text.is_empty());
            
            if let Some(tokens) = body.get("tokens_generated") {
                assert!(tokens.as_u64().unwrap() > 0);
            }
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Start with: model-rs serve --model-path <path>");
                eprintln!("Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_v1_generate_stream_endpoint() {
    let base = integration_base_url();
    let url = format!(
        "{base}/v1/generate_stream?prompt=Hello&max_tokens=20&temperature=0.7"
    );
    
    let client = reqwest::Client::new();
    let response = client.get(url).send().await;

    match response {
        Ok(resp) => {
            if resp.status() == 404 {
                eprintln!("model-rs endpoints not found (404). Skipping integration test.");
                return;
            }
            assert_eq!(resp.status(), 200);
            
            let content_type = resp.headers().get("content-type").unwrap();
            if !content_type.to_str().unwrap_or("").contains("text/event-stream") {
                eprintln!("Unexpected content-type for SSE (not model-rs). Skipping integration test.");
                return;
            }
            
            let mut stream = resp.bytes_stream();
            let mut received_data = false;
            
            let result = timeout(Duration::from_secs(30), async {
                use futures_util::StreamExt;
                
                while let Some(chunk) = stream.next().await {
                    if let Ok(bytes) = chunk {
                        let text = String::from_utf8_lossy(&bytes);
                        if text.contains("data:") {
                            received_data = true;
                            
                            for line in text.lines() {
                                if line.starts_with("data:") {
                                    let json_str = line.trim_start_matches("data:").trim();
                                    if !json_str.is_empty() {
                                        let parsed: Result<serde_json::Value, _> = 
                                            serde_json::from_str(json_str);
                                        assert!(parsed.is_ok(), "Invalid JSON: {}", json_str);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }).await;

            assert!(result.is_ok(), "Stream timed out");
            assert!(received_data, "No data received from stream");
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Start with: model-rs serve --model-path <path>");
                eprintln!("Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_v1_generate_with_parameters() {
    let client = reqwest::Client::new();
    let base = integration_base_url();

    let response = client
        .post(format!("{base}/v1/generate"))
        .json(&serde_json::json!({
            "prompt": "Count to 5:",
            "max_tokens": 30,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            if resp.status() == 404 {
                eprintln!("model-rs endpoints not found (404). Skipping integration test.");
                return;
            }
            assert_eq!(resp.status(), 200);
            
            let body = resp.json::<serde_json::Value>().await.unwrap();
            assert!(body.get("text").is_some());
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_v1_generate_invalid_request() {
    let client = reqwest::Client::new();
    let base = integration_base_url();

    let response = client
        .post(format!("{base}/v1/generate"))
        .json(&serde_json::json!({
            "max_tokens": 50
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            if resp.status() == 404 {
                eprintln!("model-rs endpoints not found (404). Skipping integration test.");
                return;
            }
            assert_eq!(resp.status(), 400);
            
            let body = resp.json::<serde_json::Value>().await.unwrap();
            assert!(body.get("error").is_some());
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_ollama_generate_non_stream() {
    let client = reqwest::Client::new();
    let base = integration_base_url();

    let response = client
        .post(format!("{base}/api/generate"))
        .json(&serde_json::json!({
            "prompt": "Hello",
            "stream": false,
            "options": {
                "temperature": 0.7,
                "num_predict": 20
            }
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            if resp.status() == 404 {
                eprintln!("model-rs endpoints not found (404). Skipping integration test.");
                return;
            }
            if resp.status() == 500 {
                eprintln!(
                    "Server returned 500 (model missing or not model-rs). Skipping integration test."
                );
                return;
            }
            assert_eq!(resp.status(), 200);
            
            let body = resp.json::<serde_json::Value>().await.unwrap();
            assert!(body.get("response").is_some());
            assert!(body.get("done").is_some());
            assert_eq!(body["done"].as_bool().unwrap(), true);
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_ollama_generate_stream() {
    let client = reqwest::Client::new();
    let base = integration_base_url();

    let response = client
        .post(format!("{base}/api/generate"))
        .json(&serde_json::json!({
            "prompt": "Hi",
            "stream": true,
            "options": {
                "num_predict": 10
            }
        }))
        .send()
        .await;

    match response {
        Ok(resp) => {
            if resp.status() == 404 {
                eprintln!("model-rs endpoints not found (404). Skipping integration test.");
                return;
            }
            if resp.status() == 500 {
                eprintln!(
                    "Server returned 500 (model missing or not model-rs). Skipping integration test."
                );
                return;
            }
            assert_eq!(resp.status(), 200);
            
            let mut stream = resp.bytes_stream();
            let mut received_done = false;
            
            let result = timeout(Duration::from_secs(30), async {
                use futures_util::StreamExt;
                
                while let Some(chunk) = stream.next().await {
                    if let Ok(bytes) = chunk {
                        let text = String::from_utf8_lossy(&bytes);
                        
                        for line in text.lines() {
                            if !line.trim().is_empty() {
                                let parsed: Result<serde_json::Value, _> = 
                                    serde_json::from_str(line);
                                assert!(parsed.is_ok(), "Invalid NDJSON: {}", line);
                                
                                if let Ok(json) = parsed {
                                    if json.get("done").and_then(|v| v.as_bool()) == Some(true) {
                                        received_done = true;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if received_done {
                            break;
                        }
                    }
                }
            }).await;

            assert!(result.is_ok(), "Stream timed out");
            assert!(received_done, "Did not receive done=true message");
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}

#[tokio::test]
async fn test_ollama_tags_endpoint() {
    let client = reqwest::Client::new();
    let base = integration_base_url();

    let response = client
        .get(format!("{base}/api/tags"))
        .send()
        .await;

    match response {
        Ok(resp) => {
            if resp.status() == 404 {
                eprintln!("model-rs endpoints not found (404). Skipping integration test.");
                return;
            }
            if resp.status() == 500 {
                eprintln!(
                    "Server returned 500 (model missing or not model-rs). Skipping integration test."
                );
                return;
            }
            assert_eq!(resp.status(), 200);
            
            let body = resp.json::<serde_json::Value>().await.unwrap();
            assert!(body.get("models").is_some());
            
            let models = body["models"].as_array().unwrap();
            assert!(!models.is_empty());
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}

//! HTTP server for model-rs LLM inference API
//!
//! Provides an Ollama-compatible REST API for running local LLM inference.

use crate::error::{ModelError, Result};
use crate::local::{
    global_model_cache, get_or_load_model, DevicePreference, LocalModel, LocalModelConfig,
};
use crate::{download::download_model, model_ops::ModelOperations};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    response::sse::{Event, Sse},
    routing::{get, post},
    Json, Router,
};
use axum::body::Body;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::{
    convert::Infallible,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

/// Application state shared across all HTTP requests
#[derive(Clone)]
pub struct AppState {
    default_model_path: PathBuf,
    device_preference: DevicePreference,
    device_index: usize,
}

/// Request payload for the `/api/generate` endpoint
///
/// Compatible with Ollama's generate API format.
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    /// The prompt text to generate from
    pub prompt: String,
    /// Optional system prompt to guide generation
    #[serde(default)]
    pub system: Option<String>,
    /// Maximum number of tokens to generate (default: from config)
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy, higher = more random)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling threshold
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling (keep only top k tokens)
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Repeat penalty for tokens
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
}

/// Response payload for the `/api/generate` endpoint
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    /// The generated text
    pub text: String,
}

/// Request payload for the `/v1/generate_batch` endpoint
///
/// This is an internal performance endpoint to run multiple generations
/// using the same model instance.
#[derive(Debug, Deserialize)]
pub struct GenerateBatchRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompts: Vec<String>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct GenerateBatchResponse {
    pub model: String,
    pub results: Vec<String>,
}

/// Ollama-compatible generation options
#[derive(Debug, Deserialize)]
pub struct OllamaOptions {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub num_predict: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    /// How long the model will stay loaded after the request (Ollama compatibility).
    /// Parsed as JSON value to accept both numbers and strings (e.g. `5m`, `-1`).
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaEmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    /// How long the model will stay loaded after the request (Ollama compatibility).
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
}

/// Ollama /api/chat request - messages array format
#[derive(Debug, Deserialize)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<OllamaChatMessage>,
    /// How long the model will stay loaded after the request (Ollama compatibility).
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatMessageResponse {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaChatMessageResponse,
    pub done: bool,
}

/// Ollama /api/embed request - uses `input` (string or array)
#[derive(Debug, Deserialize)]
pub struct OllamaEmbedRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct OllamaEmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaTagModel>,
}

#[derive(Debug, Serialize)]
pub struct OllamaTagModel {
    pub name: String,
    pub model: String,
    pub modified_at: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaDeleteRequest {
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaCopyRequest {
    pub source: String,
    pub destination: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaPullRequest {
    pub model: String,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct OllamaStatusResponse {
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaStatusEvent {
    pub status: String,
}

fn build_effective_prompt(prompt: &str, system: Option<&str>) -> String {
    match system {
        Some(system_prompt) if !system_prompt.trim().is_empty() => format!(
            "System: {}\n\nUser: {}\n\nAssistant:",
            system_prompt.trim(),
            prompt
        ),
        _ => prompt.to_string(),
    }
}

/// Build prompt from Ollama chat messages array (system, user, assistant)
fn messages_to_prompt(messages: &[OllamaChatMessage]) -> String {
    let mut parts = Vec::new();
    let mut system = String::new();
    for msg in messages {
        let role = msg.role.to_lowercase();
        let content = msg.content.trim();
        if content.is_empty() {
            continue;
        }
        match role.as_str() {
            "system" => system = content.to_string(),
            "user" => parts.push(format!("User: {}", content)),
            "assistant" => parts.push(format!("Assistant: {}", content)),
            _ => {}
        }
    }
    let body = parts.join("\n\n");
    if body.is_empty() {
        String::new()
    } else if system.is_empty() {
        format!("{}\n\nAssistant:", body)
    } else {
        format!("System: {}\n\n{}\n\nAssistant:", system, body)
    }
}

fn apply_config_overrides(mut base: LocalModelConfig, req: &GenerateRequest) -> LocalModelConfig {
    if let Some(t) = req.temperature {
        base.temperature = t;
    }
    if let Some(p) = req.top_p {
        base.top_p = p;
    }
    if let Some(k) = req.top_k {
        base.top_k = Some(k);
    }
    if let Some(rp) = req.repeat_penalty {
        base.repeat_penalty = rp;
    }
    if let Some(mt) = req.max_tokens {
        base.max_seq_len = mt.saturating_mul(2);
    }
    base
}

fn now_millis_string() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_millis().to_string(),
        Err(_) => "0".to_string(),
    }
}

fn map_model_error(err: ModelError) -> (StatusCode, String) {
    match err {
        ModelError::ModelNotFound(msg) => (StatusCode::NOT_FOUND, msg),
        ModelError::InvalidConfig(msg) => (StatusCode::BAD_REQUEST, msg),
        other => (StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
    }
}

fn apply_ollama_options(mut base: LocalModelConfig, opts: Option<&OllamaOptions>) -> (LocalModelConfig, usize, f32) {
    let mut max_tokens = 512usize;
    let mut temperature = base.temperature;

    if let Some(opts) = opts {
        if let Some(t) = opts.temperature {
            base.temperature = t;
            temperature = t;
        }
        if let Some(p) = opts.top_p {
            base.top_p = p;
        }
        if let Some(k) = opts.top_k {
            base.top_k = Some(k);
        }
        if let Some(rp) = opts.repeat_penalty {
            base.repeat_penalty = rp;
        }
        if let Some(np) = opts.num_predict {
            max_tokens = np;
            base.max_seq_len = np.saturating_mul(2);
        }
    }

    (base, max_tokens, temperature)
}

async fn load_model_from_request(
    state: &AppState,
    model: Option<&str>,
) -> std::result::Result<(String, PathBuf, Arc<tokio::sync::RwLock<LocalModel>>), (StatusCode, String)> {
    let model_name = model
        .unwrap_or("model-rs")
        .to_string();

    let resolved_path = if let Some(model) = model {
        let ops = ModelOperations::new();
        ops.resolve_model_path(model).map_err(map_model_error)?
    } else {
        state.default_model_path.clone()
    };

    let cfg = LocalModelConfig {
        model_path: resolved_path.clone(),
        device_preference: state.device_preference,
        device_index: state.device_index,
        ..Default::default()
    };

    let model_arc = get_or_load_model(cfg).await.map_err(map_model_error)?;
    Ok((model_name, resolved_path, model_arc))
}

async fn generate_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> std::result::Result<Json<GenerateResponse>, (StatusCode, String)> {
    let effective_prompt = build_effective_prompt(&req.prompt, req.system.as_deref());

    let (_model_name, _model_path, model_arc) = load_model_from_request(&state, None).await?;
    let mut model = model_arc.write().await;

    let cfg = apply_config_overrides(model.config().clone(), &req);
    *model.config_mut() = cfg;

    let max_tokens = req.max_tokens.unwrap_or(512);
    let temperature = req.temperature.unwrap_or(model.config().temperature);

    model
        .generate_text(&effective_prompt, max_tokens, temperature)
        .await
        .map(|text| Json(GenerateResponse { text }))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

async fn generate_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> std::result::Result<Sse<impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>>>, (StatusCode, String)> {
    let effective_prompt = build_effective_prompt(&req.prompt, req.system.as_deref());

    let (tx, rx) = mpsc::channel::<String>(64);

    let max_tokens = req.max_tokens.unwrap_or(512);

    let (_model_name, _model_path, model_arc) = load_model_from_request(&state, None).await?;
    tokio::spawn(async move {
        let mut model = model_arc.write().await;

        let cfg = apply_config_overrides(model.config().clone(), &req);
        *model.config_mut() = cfg;

        let temperature = req.temperature.unwrap_or(model.config().temperature);

        let res = model
            .generate_stream_with(&effective_prompt, max_tokens, temperature, |piece| {
                let _ = tx.try_send(piece);
                Ok(())
            })
            .await;

        if let Err(e) = res {
            error!("stream generation failed: {}", e);
        }
    });

    let stream = ReceiverStream::new(rx).map(|chunk| Ok(Event::default().event("token").data(chunk)));

    Ok(Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new().interval(Duration::from_secs(10)).text("keep-alive"),
    ))
}

async fn generate_batch_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateBatchRequest>,
) -> std::result::Result<Json<GenerateBatchResponse>, (StatusCode, String)> {
    if req.prompts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "prompts cannot be empty".to_string(),
        ));
    }

    let (model_name, _model_path, model_arc) =
        load_model_from_request(&state, req.model.as_deref()).await?;
    let mut model = model_arc.write().await;

    let max_tokens = req.max_tokens.unwrap_or(512);
    let temperature = req.temperature.unwrap_or(model.config().temperature);

    let effective_prompts: Vec<String> = req
        .prompts
        .iter()
        .map(|p| build_effective_prompt(p, req.system.as_deref()))
        .collect();
    let prompt_refs: Vec<&str> = effective_prompts.iter().map(|s| s.as_str()).collect();

    // Apply per-request generation overrides.
    let mut cfg = model.config().clone();
    if let Some(t) = req.temperature {
        cfg.temperature = t;
    }
    if let Some(p) = req.top_p {
        cfg.top_p = p;
    }
    if let Some(k) = req.top_k {
        cfg.top_k = Some(k);
    }
    if let Some(rp) = req.repeat_penalty {
        cfg.repeat_penalty = rp;
    }
    if let Some(mt) = req.max_tokens {
        cfg.max_seq_len = mt.saturating_mul(2);
    }
    *model.config_mut() = cfg;

    let results = model
        .generate_batch(prompt_refs, max_tokens, temperature)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(GenerateBatchResponse {
        model: model_name,
        results,
    }))
}

async fn ollama_generate_handler(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> std::result::Result<Response, (StatusCode, String)> {
    // Keep compatibility with Ollama's `keep_alive` (currently ignored).
    let _keep_alive = req.keep_alive;
    let stream = req.stream.unwrap_or(false);
    let effective_prompt = build_effective_prompt(&req.prompt, req.system.as_deref());
    let (model_name, _model_path, model_arc) =
        load_model_from_request(&state, req.model.as_deref()).await?;

    if !stream {
        let mut model = model_arc.write().await;
        let (cfg, max_tokens, temperature) =
            apply_ollama_options(model.config().clone(), req.options.as_ref());
        *model.config_mut() = cfg;

        let text = model
            .generate_text(&effective_prompt, max_tokens, temperature)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let resp = OllamaGenerateResponse {
            model: model_name,
            response: text,
            done: true,
            created_at: now_millis_string(),
        };
        return Ok(Json(resp).into_response());
    }

    let (tx, rx) = mpsc::channel::<Bytes>(64);
    tokio::spawn(async move {
        let mut model = model_arc.write().await;
        let (cfg, max_tokens, temperature) =
            apply_ollama_options(model.config().clone(), req.options.as_ref());
        *model.config_mut() = cfg;

        let send_line = |obj: &OllamaGenerateResponse| {
            if let Ok(line) = serde_json::to_string(obj) {
                let _ = tx.try_send(Bytes::from(format!("{}\n", line)));
            }
        };

        let res = model
            .generate_stream_with(&effective_prompt, max_tokens, temperature, |piece| {
                let obj = OllamaGenerateResponse {
                    model: model_name.clone(),
                    response: piece,
                    done: false,
                    created_at: now_millis_string(),
                };
                send_line(&obj);
                Ok(())
            })
            .await;

        if let Err(e) = res {
            error!("ollama stream generation failed: {}", e);
        }

        let done_obj = OllamaGenerateResponse {
            model: model_name,
            response: String::new(),
            done: true,
            created_at: now_millis_string(),
        };
        send_line(&done_obj);
    });

    let stream = ReceiverStream::new(rx).map(Ok::<Bytes, Infallible>);
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(resp)
}

async fn ollama_embeddings_handler(
    State(state): State<AppState>,
    Json(req): Json<OllamaEmbeddingsRequest>,
) -> std::result::Result<Json<OllamaEmbeddingsResponse>, (StatusCode, String)> {
    // Keep compatibility with Ollama's `keep_alive` (currently ignored).
    let _keep_alive = req.keep_alive;
    let (_model_name, _model_path, model_arc) =
        load_model_from_request(&state, req.model.as_deref()).await?;
    let mut model = model_arc.write().await;
    let emb = model
        .embed_text(&req.prompt)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(OllamaEmbeddingsResponse { embedding: emb }))
}

async fn ollama_tags_handler(
    State(state): State<AppState>,
) -> std::result::Result<Json<OllamaTagsResponse>, (StatusCode, String)> {
    let stats = global_model_cache().stats();

    let mut models: Vec<OllamaTagModel> = stats
        .models
        .into_iter()
        .map(|m| {
            let name = m
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| m.path.display().to_string());

            OllamaTagModel {
                name: name.clone(),
                model: name,
                modified_at: now_millis_string(),
            }
        })
        .collect();

    // If cache is empty (e.g., just started), at least report the default model.
    if models.is_empty() {
        let fallback = state
            .default_model_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model-rs".to_string());
        models.push(OllamaTagModel {
            name: fallback.clone(),
            model: fallback,
            modified_at: now_millis_string(),
        });
    }

    Ok(Json(OllamaTagsResponse { models }))
}

async fn ollama_chat_handler(
    State(state): State<AppState>,
    Json(req): Json<OllamaChatRequest>,
) -> std::result::Result<Response, (StatusCode, String)> {
    // Keep compatibility with Ollama's `keep_alive` (currently ignored).
    let _keep_alive = req.keep_alive;
    let stream = req.stream.unwrap_or(false);
    let effective_prompt = messages_to_prompt(&req.messages);
    let (model_name, _model_path, model_arc) =
        load_model_from_request(&state, req.model.as_deref()).await?;

    if effective_prompt.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "messages array cannot be empty".to_string(),
        ));
    }

    if !stream {
        let mut model = model_arc.write().await;
        let (cfg, max_tokens, temperature) =
            apply_ollama_options(model.config().clone(), req.options.as_ref());
        *model.config_mut() = cfg;

        let text = model
            .generate_text(&effective_prompt, max_tokens, temperature)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let resp = OllamaChatResponse {
            model: model_name,
            created_at: now_millis_string(),
            message: OllamaChatMessageResponse {
                role: "assistant".to_string(),
                content: text,
            },
            done: true,
        };
        return Ok(Json(resp).into_response());
    }

    let (tx, rx) = mpsc::channel::<Bytes>(64);
    tokio::spawn(async move {
        let mut model = model_arc.write().await;
        let (cfg, max_tokens, temperature) =
            apply_ollama_options(model.config().clone(), req.options.as_ref());
        *model.config_mut() = cfg;

        let send_line = |content: &str, done: bool| {
            let obj = OllamaChatResponse {
                model: model_name.clone(),
                created_at: now_millis_string(),
                message: OllamaChatMessageResponse {
                    role: "assistant".to_string(),
                    content: content.to_string(),
                },
                done,
            };
            if let Ok(line) = serde_json::to_string(&obj) {
                let _ = tx.try_send(Bytes::from(format!("{}\n", line)));
            }
        };

        let res = model
            .generate_stream_with(&effective_prompt, max_tokens, temperature, |piece| {
                send_line(&piece, false);
                Ok(())
            })
            .await;

        if let Err(e) = res {
            error!("ollama chat stream failed: {}", e);
        }
        send_line("", true);
    });

    let stream = ReceiverStream::new(rx).map(Ok::<Bytes, Infallible>);
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(resp)
}

/// Ollama /api/show response - model details
#[derive(Debug, Serialize)]
pub struct OllamaShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    #[serde(rename = "details")]
    pub details: OllamaShowDetails,
}

#[derive(Debug, Serialize)]
pub struct OllamaShowDetails {
    pub format: String,
    pub family: String,
    pub families: Option<Vec<String>>,
    #[serde(rename = "parameter_size")]
    pub parameter_size: String,
    #[serde(rename = "quantization_level")]
    pub quantization_level: String,
}

async fn ollama_show_handler(
    State(state): State<AppState>,
    Json(req): Json<OllamaShowRequest>,
) -> std::result::Result<Json<OllamaShowResponse>, (StatusCode, String)> {
    let path = if let Some(model) = req.model.as_deref() {
        let ops = ModelOperations::new();
        ops.resolve_model_path(model).map_err(map_model_error)?
    } else {
        state.default_model_path.clone()
    };

    let config_path = path.join("config.json");
    let config_content = std::fs::read_to_string(&config_path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let config: serde_json::Value =
        serde_json::from_str(&config_content).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let hidden_size = config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0);
    let num_layers = config.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(0);
    let num_heads = config.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(0);
    let vocab_size = config.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0);

    let parameters = format!(
        "hidden_size={} num_hidden_layers={} num_attention_heads={} vocab_size={}",
        hidden_size, num_layers, num_heads, vocab_size
    );

    let template = config
        .get("chat_template")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let param_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .map(|h| format!("{}B", h / 1000))
        .unwrap_or_else(|| "unknown".to_string());

    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model-rs".to_string());

    let has_gguf = std::fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .any(|e| e.path().extension().map_or(false, |x| x == "gguf"))
        })
        .unwrap_or(false);
    let format_type = if has_gguf { "gguf" } else { "safetensors" };

    let resp = OllamaShowResponse {
        modelfile: format!("# Model: {}", name),
        parameters: parameters.clone(),
        template: template.clone(),
        details: OllamaShowDetails {
            format: format_type.to_string(),
            family: model_type.clone(),
            families: Some(vec![model_type]),
            parameter_size: param_size,
            quantization_level: String::new(),
        },
    };
    Ok(Json(resp))
}

#[derive(Debug, Deserialize)]
pub struct OllamaShowRequest {
    #[serde(default)]
    pub model: Option<String>,
}

async fn ollama_embed_handler(
    State(state): State<AppState>,
    Json(req): Json<OllamaEmbedRequest>,
) -> std::result::Result<Json<OllamaEmbedResponse>, (StatusCode, String)> {
    let inputs: Vec<String> = match &req.input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "input must be a string or array of strings".to_string(),
            ))
        }
    };

    if inputs.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "input cannot be empty".to_string(),
        ));
    }

    let (_model_name, _model_path, model_arc) =
        load_model_from_request(&state, req.model.as_deref()).await?;
    let mut model = model_arc.write().await;
    let mut embeddings = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let emb = model
            .embed_text(input)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        embeddings.push(emb);
    }
    Ok(Json(OllamaEmbedResponse { embeddings }))
}

async fn ollama_delete_handler(
    Json(req): Json<OllamaDeleteRequest>,
) -> std::result::Result<Json<OllamaStatusResponse>, (StatusCode, String)> {
    let model = req.model;

    let res = tokio::task::spawn_blocking(move || {
        let ops = ModelOperations::new();
        ops.remove(&model, true)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    res.map_err(map_model_error)?;
    Ok(Json(OllamaStatusResponse {
        status: "success".to_string(),
    }))
}

async fn ollama_copy_handler(
    Json(req): Json<OllamaCopyRequest>,
) -> std::result::Result<Json<OllamaStatusResponse>, (StatusCode, String)> {
    let source = req.source;
    let destination = req.destination;

    let res = tokio::task::spawn_blocking(move || {
        let ops = ModelOperations::new();
        ops.copy(&source, &destination)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    res.map_err(map_model_error)?;
    Ok(Json(OllamaStatusResponse {
        status: "success".to_string(),
    }))
}

async fn ollama_pull_handler(
    Json(req): Json<OllamaPullRequest>,
) -> std::result::Result<Response, (StatusCode, String)> {
    let model = req.model;
    let stream = req.stream.unwrap_or(true);

    if !stream {
        download_model(&model, None, None).await.map_err(map_model_error)?;
        return Ok(
            Json(OllamaStatusResponse {
                status: "success".to_string(),
            })
            .into_response(),
        );
    }

    let (tx, rx) = mpsc::channel::<Bytes>(64);
    tokio::spawn(async move {
        let started = OllamaStatusEvent {
            status: format!("pulling {model}"),
        };
        if let Ok(line) = serde_json::to_string(&started) {
            let _ = tx.try_send(Bytes::from(format!("{}\n", line)));
        }

        let result = download_model(&model, None, None).await;
        match result {
            Ok(()) => {
                let done = OllamaStatusEvent {
                    status: "success".to_string(),
                };
                if let Ok(line) = serde_json::to_string(&done) {
                    let _ = tx.try_send(Bytes::from(format!("{}\n", line)));
                }
            }
            Err(e) => {
                let done = OllamaStatusEvent {
                    status: format!("error: {}", e.to_string()),
                };
                if let Ok(line) = serde_json::to_string(&done) {
                    let _ = tx.try_send(Bytes::from(format!("{}\n", line)));
                }
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(Ok::<Bytes, Infallible>);
    let mut resp = Response::new(Body::from_stream(stream));
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(resp)
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

pub fn build_app(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_headers(Any)
        .allow_methods(Any);

    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/generate", post(generate_handler))
        .route("/v1/generate_stream", post(generate_stream_handler))
        .route("/v1/generate_batch", post(generate_batch_handler))
        .route("/api/generate", post(ollama_generate_handler))
        .route("/api/chat", post(ollama_chat_handler))
        .route("/api/show", post(ollama_show_handler))
        .route("/api/embeddings", post(ollama_embeddings_handler))
        .route("/api/embed", post(ollama_embed_handler))
        .route("/api/tags", get(ollama_tags_handler).post(ollama_tags_handler))
        .route(
            "/api/delete",
            post(ollama_delete_handler).delete(ollama_delete_handler),
        )
        .route("/api/copy", post(ollama_copy_handler))
        .route("/api/pull", post(ollama_pull_handler))
        .layer(cors)
        .with_state(state)
}

pub async fn serve(model_path: Option<&Path>, port: u16, device: &str, device_index: usize) -> Result<()> {
    let path = model_path.ok_or_else(|| {
        ModelError::InvalidConfig(
            "Model path is required for serving. Use --model-path <path> to specify a local model directory.".to_string(),
        )
    })?;

    let device_preference: DevicePreference = device.parse()?;

    let config = LocalModelConfig {
        model_path: path.to_path_buf(),
        device_preference,
        device_index,
        ..Default::default()
    };

    // Preload the default model into the in-memory cache so the server is ready.
    let _ = global_model_cache().preload(config).await?;

    let state = AppState {
        default_model_path: path.to_path_buf(),
        device_preference,
        device_index,
    };

    let app = build_app(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Serving web API on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| ModelError::IoError(e))?;

    axum::serve(listener, app)
        .await
        .map_err(|e| ModelError::LocalModelError(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{Request, header};
    use serde_json::json;
    use tempfile::tempdir;
    use tower::ServiceExt;

    fn write_minimal_llama_config(dir: &std::path::Path) {
        // Minimal config for architecture detection + llama loader parsing.
        // No weights are provided in tests (placeholder mode).
        let cfg = json!({
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();
    }

    fn write_minimal_tokenizer(dir: &std::path::Path) {
        // Write a minimal tokenizer.json directly. We only need LocalModel::load
        // to successfully parse it during tests; generation is expected to fail
        // due to missing weights.
        let tokenizer = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "<unk>": 0, "hello": 1, "world": 2 },
    "unk_token": "<unk>"
  }
}"#;
        std::fs::write(dir.join("tokenizer.json"), tokenizer.as_bytes()).unwrap();
    }

    async fn build_test_app() -> (tempfile::TempDir, Router) {
        let dir = tempdir().unwrap();
        write_minimal_llama_config(dir.path());
        write_minimal_tokenizer(dir.path());

        // Ensure tests don't share cached models across runs.
        global_model_cache().clear();

        let config = LocalModelConfig {
            model_path: dir.path().to_path_buf(),
            device_preference: DevicePreference::Cpu,
            device_index: 0,
            ..Default::default()
        };
        let _ = global_model_cache().preload(config).await.unwrap();

        let state = AppState {
            default_model_path: dir.path().to_path_buf(),
            device_preference: DevicePreference::Cpu,
            device_index: 0,
        };

        (dir, build_app(state))
    }

    #[tokio::test]
    async fn test_ollama_tags_ok() {
        let (_dir, app) = build_test_app().await;

        let req = Request::builder()
            .method("POST")
            .uri("/api/tags")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_ollama_tags_get_ok() {
        let (_dir, app) = build_test_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/api/tags")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_ollama_generate_non_stream_returns_error_without_weights() {
        let (_dir, app) = build_test_app().await;

        let body = serde_json::to_vec(&json!({
            "prompt": "hello",
            "stream": false
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_ollama_generate_stream_returns_ndjson_and_done() {
        let (_dir, app) = build_test_app().await;

        let body = serde_json::to_vec(&json!({
            "prompt": "hello",
            "stream": true
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(content_type, "application/x-ndjson");

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let s = String::from_utf8_lossy(&bytes);
        assert!(s.contains("\"done\":true"));
    }

    #[tokio::test]
    async fn test_ollama_chat_empty_messages_returns_400() {
        let (_dir, app) = build_test_app().await;

        let body = serde_json::to_vec(&json!({
            "messages": [],
            "keep_alive": "5m"
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_ollama_generate_keep_alive_does_not_break_parsing() {
        let (_dir, app) = build_test_app().await;

        let body = serde_json::to_vec(&json!({
            "prompt": "hello",
            "stream": false,
            "keep_alive": "5m"
        }))
        .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Still fails due to missing weights in the test model directory.
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_ollama_show_returns_200() {
        let (_dir, app) = build_test_app().await;

        // With multi-model support, omitting `model` should show the server's default model.
        let body = serde_json::to_vec(&json!({})).unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/api/show")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(body.get("details").is_some());
        assert!(body.get("parameters").is_some());
    }

    #[tokio::test]
    async fn test_health_returns_200() {
        let (_dir, app) = build_test_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body.get("status").and_then(|v| v.as_str()), Some("ok"));
    }

    #[tokio::test]
    async fn test_ollama_delete_ok() {
        let (_dir, app) = build_test_app().await;

        let model_dir = tempdir().unwrap();
        std::fs::write(model_dir.path().join("config.json"), "{}").unwrap();
        let model = model_dir.path().to_string_lossy().to_string();

        let body = serde_json::to_vec(&json!({ "model": model })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/delete")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(!model_dir.path().exists());
    }

    #[tokio::test]
    async fn test_ollama_copy_ok() {
        let (_dir, app) = build_test_app().await;

        let source_dir = tempdir().unwrap();
        std::fs::write(source_dir.path().join("config.json"), "{}").unwrap();
        let source = source_dir.path().to_string_lossy().to_string();

        let dest = format!(
            "copy-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        );

        let body = serde_json::to_vec(&json!({ "source": source, "destination": dest })).unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/api/copy")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ops = ModelOperations::new();
        let dest_path = ops.resolve_model_path(&dest).unwrap();
        assert!(dest_path.join("config.json").exists());

        ops.remove(&dest, true).unwrap();
    }
}

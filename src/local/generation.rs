//! Text generation logic for local models
//!
//! This module handles the core generation logic for different model architectures,
//! providing a unified interface for text generation and streaming.

use crate::error::{ModelError, Result};
use crate::local::backends::LocalBackend;
use crate::local::tokenization::stream_piece;
use candle_core::{DType, Device, Tensor};
use tracing::debug;

/// Result of a generation operation, containing generated token IDs
pub type GenerationResult = Vec<u32>;

/// Generate text from a prompt (non-streaming)
pub fn generate_text<F>(
    backend: &mut LocalBackend,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    match backend {
        LocalBackend::Llama { model, config } => generate_llama(
            model,
            config,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::Mistral {
            model,
            config: _config,
        } => generate_mistral(
            model,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::Phi3 { model, .. } => generate_phi3(
            model,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::Mamba { model, config } => generate_mamba(
            model,
            config,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::GraniteMoeHybrid { model, config } => generate_granite_moe_hybrid(
            model,
            config,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::Bert { .. } => Err(ModelError::LocalModelError(
            "Encoder-only models (BERT) cannot generate text. Use embeddings instead.".to_string(),
        )),
        LocalBackend::Gemma { model, .. } => generate_gemma(
            model,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::Qwen2 { model, .. } => generate_qwen2(
            model,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::Qwen3 { model, .. } => generate_qwen3(
            model,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::DeepSeek2 { model, .. } => generate_deepseek2(
            model,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        LocalBackend::Glm4 { model, .. } => generate_glm4(
            model,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
            device,
            &sample_fn,
        ),
        #[cfg(feature = "gguf")]
        LocalBackend::Gguf { backend } => generate_gguf(
            backend,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
        ),
        #[cfg(feature = "mlx")]
        LocalBackend::Mlx { backend } => generate_mlx(
            backend,
            input_ids,
            max_tokens,
            temperature,
            top_p,
            top_k,
            eos_token,
        ),
    }
}

/// Generate text with streaming callback
pub fn generate_text_stream<F, C>(
    backend: &mut LocalBackend,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: C,
    emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    match backend {
        LocalBackend::Llama { model, config } => stream_llama(
            model, config, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::Mistral {
            model,
            config: _config,
        } => stream_mistral(
            model, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::Phi3 { model, .. } => stream_phi3(
            model, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::Mamba { model, config } => stream_mamba(
            model, config, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::GraniteMoeHybrid { model, config } => stream_granite_moe_hybrid(
            model, config, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::Bert { .. } => Err(ModelError::LocalModelError(
            "Encoder-only models (BERT) cannot generate text. Use embeddings instead.".to_string(),
        )),
        LocalBackend::Gemma { model, .. } => stream_gemma(
            model, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::Qwen2 { model, .. } => stream_qwen2(
            model, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::Qwen3 { model, .. } => stream_qwen3(
            model, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::DeepSeek2 { model, .. } => stream_deepseek2(
            model, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        LocalBackend::Glm4 { model, .. } => stream_glm4(
            model, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, device,
            &sample_fn, emit,
        ),
        #[cfg(feature = "gguf")]
        LocalBackend::Gguf { backend } => stream_gguf(
            backend, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, emit,
        ),
        #[cfg(feature = "mlx")]
        LocalBackend::Mlx { backend } => stream_mlx(
            backend, tokenizer, input_ids, max_tokens, temp, top_p, top_k, eos_token, emit,
        ),
    }
}

// ============================================================================
// Llama Generation
// ============================================================================

fn generate_llama<F>(
    model: &candle_transformers::models::llama::Llama,
    config: &candle_transformers::models::llama::Config,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    use candle_transformers::models::llama::Cache;

    let mut cache = Cache::new(true, DType::F32, config, device)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to create cache: {}", e)))?;

    // Process the prompt to fill the cache
    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0, &mut cache)?;

    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let last_logits = &logits_vec[0];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    // Generate remaining tokens one at a time
    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1, &mut cache)?;

        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_llama<F, C>(
    model: &candle_transformers::models::llama::Llama,
    config: &candle_transformers::models::llama::Config,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    use candle_transformers::models::llama::Cache;
    use std::time::Instant;

    let mut cache = Cache::new(true, DType::F32, config, device)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to create cache: {}", e)))?;
    let mut started = false;

    let t_prefill = Instant::now();
    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0, &mut cache)?;
    debug!(
        "Generation timing: llama_prefill_ms={}",
        t_prefill.elapsed().as_millis()
    );

    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let last_logits = &logits_vec[0];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    let mut token_count: usize = 0;
    let mut total_token_ms: u128 = 0;
    for idx in 1..max_tokens {
        let t_step = Instant::now();
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1, &mut cache)?;

        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }

        let step_ms = t_step.elapsed().as_millis();
        token_count += 1;
        total_token_ms += step_ms;
        if idx <= 5 {
            debug!(
                "Generation timing: llama_token_idx={} step_ms={}",
                idx, step_ms
            );
        }
    }

    if token_count > 0 {
        debug!(
            "Generation timing: llama_tokens_generated={} avg_token_ms={}",
            token_count,
            (total_token_ms / token_count as u128)
        );
    }

    Ok(())
}

// ============================================================================
// Mistral Generation
// ============================================================================

fn generate_mistral<F>(
    model: &mut candle_transformers::models::mistral::Model,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    // Process the prompt
    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;

    // logits shape: [batch=1, seq_len, vocab_size] for Mistral
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let seq_len = logits_vec[0].len();
    let last_logits = &logits_vec[0][seq_len - 1];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    // Generate remaining tokens one at a time
    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx)?;

        // Single token: [batch=1, 1, vocab]
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_mistral<F, C>(
    model: &mut candle_transformers::models::mistral::Model,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    use std::time::Instant;

    let mut started = false;

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let seq_len = logits_vec[0].len();
    let last_logits = &logits_vec[0][seq_len - 1];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    let mut token_count: usize = 0;
    let mut total_token_ms: u128 = 0;
    for idx in 1..max_tokens {
        let t_step = Instant::now();
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx)?;

        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }

        let step_ms = t_step.elapsed().as_millis();
        token_count += 1;
        total_token_ms += step_ms;
        if idx <= 5 {
            debug!(
                "Generation timing: mistral_token_idx={} step_ms={}",
                idx, step_ms
            );
        }
    }

    if token_count > 0 {
        debug!(
            "Generation timing: mistral_tokens_generated={} avg_token_ms={}",
            token_count,
            (total_token_ms / token_count as u128)
        );
    }

    Ok(())
}

// ============================================================================
// Phi-3 Generation
// ============================================================================

fn generate_phi3<F>(
    model: &mut candle_transformers::models::phi3::Model,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    // Phi-3 keeps an internal KV cache, so we reset it per request.
    model.clear_kv_cache();

    // Prefill with prompt
    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let seq_len = logits_vec[0].len();
    let last_logits = &logits_vec[0][seq_len - 1];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    // Generate remaining tokens one at a time
    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        // seqlen_offset is the absolute position of this token
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;

        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        // With a single token input, logits shape is [1, 1, vocab]
        let token_logits = &logits_vec[0][0];
        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_phi3<F, C>(
    model: &mut candle_transformers::models::phi3::Model,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let mut started = false;

    // Prefill with prompt
    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let seq_len = logits_vec[0].len();
    let last_logits = &logits_vec[0][seq_len - 1];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    // Generate remaining tokens
    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// Mamba Generation
// ============================================================================

fn generate_mamba<F>(
    model: &candle_transformers::models::mamba::Model,
    config: &candle_transformers::models::mamba::Config,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    use candle_transformers::models::mamba::State as MambaState;

    let mut state = MambaState::new(1, config, DType::F32, device)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to create state: {}", e)))?;

    // Feed the prompt tokens one-by-one to build state
    let mut last_logits: Option<Vec<f32>> = None;
    for &tok in input_ids.iter() {
        let token_tensor = Tensor::new(&[tok], device)?;
        let logits = model.forward(&token_tensor, &mut state)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let row = logits_vec
            .get(0)
            .ok_or_else(|| ModelError::LocalModelError("Mamba logits were empty".to_string()))?;
        last_logits = Some(row.clone());
    }

    let last_logits =
        last_logits.ok_or_else(|| ModelError::LocalModelError("Empty prompt".to_string()))?;
    let token_logits = last_logits.as_slice();
    let mut next = sample_fn(token_logits, temperature, top_p, top_k)?;

    let mut generated = vec![next];
    for _idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let token_tensor = Tensor::new(&[next], device)?;
        let logits = model.forward(&token_tensor, &mut state)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = logits_vec
            .get(0)
            .ok_or_else(|| ModelError::LocalModelError("Mamba logits were empty".to_string()))?;
        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_mamba<F, C>(
    model: &candle_transformers::models::mamba::Model,
    config: &candle_transformers::models::mamba::Config,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    use candle_transformers::models::mamba::State as MambaState;

    let mut state = MambaState::new(1, config, DType::F32, device)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to create state: {}", e)))?;
    let mut started = false;

    let mut last_logits: Option<Vec<f32>> = None;
    for &tok in input_ids.iter() {
        let token_tensor = Tensor::new(&[tok], device)?;
        let logits = model.forward(&token_tensor, &mut state)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let row = logits_vec
            .get(0)
            .ok_or_else(|| ModelError::LocalModelError("Mamba logits were empty".to_string()))?;
        last_logits = Some(row.clone());
    }

    let last_logits =
        last_logits.ok_or_else(|| ModelError::LocalModelError("Empty prompt".to_string()))?;
    let token_logits = last_logits.as_slice();
    let mut next = sample_fn(token_logits, temp, top_p, top_k)?;

    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    for _idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let token_tensor = Tensor::new(&[next], device)?;
        let logits = model.forward(&token_tensor, &mut state)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = logits_vec
            .get(0)
            .ok_or_else(|| ModelError::LocalModelError("Mamba logits were empty".to_string()))?;
        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// GraniteMoeHybrid Generation
// ============================================================================

fn generate_granite_moe_hybrid<F>(
    model: &candle_transformers::models::granitemoehybrid::GraniteMoeHybrid,
    config: &candle_transformers::models::granitemoehybrid::GraniteMoeHybridInternalConfig,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    use candle_transformers::models::granitemoehybrid::GraniteMoeHybridCache;

    let mut cache = GraniteMoeHybridCache::new(true, DType::F32, config, device)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to create cache: {}", e)))?;

    // Process the prompt to fill the cache
    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0, &mut cache)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let token_logits = &logits_vec[0];

    let mut next = sample_fn(token_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1, &mut cache)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_granite_moe_hybrid<F, C>(
    model: &candle_transformers::models::granitemoehybrid::GraniteMoeHybrid,
    config: &candle_transformers::models::granitemoehybrid::GraniteMoeHybridInternalConfig,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    use candle_transformers::models::granitemoehybrid::GraniteMoeHybridCache;

    let mut cache = GraniteMoeHybridCache::new(true, DType::F32, config, device)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to create cache: {}", e)))?;
    let mut started = false;

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0, &mut cache)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let token_logits = &logits_vec[0];

    let mut next = sample_fn(token_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1, &mut cache)?;

        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// Gemma Generation
// ============================================================================

fn generate_gemma<F>(
    model: &mut candle_transformers::models::gemma3::Model,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let last_logits = &logits_vec[0];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_gemma<F, C>(
    model: &mut candle_transformers::models::gemma3::Model,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let mut started = false;

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let last_logits = &logits_vec[0];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// Qwen2 Generation
// ============================================================================

fn generate_qwen2<F>(
    model: &mut candle_transformers::models::qwen2::ModelForCausalLM,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let last_logits = &logits_vec[0][0];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_qwen2<F, C>(
    model: &mut candle_transformers::models::qwen2::ModelForCausalLM,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let mut started = false;

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let last_logits = &logits_vec[0][0];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// Qwen3 Generation
// ============================================================================

fn generate_qwen3<F>(
    model: &mut candle_transformers::models::qwen3::ModelForCausalLM,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let last_logits = &logits_vec[0][0];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_qwen3<F, C>(
    model: &mut candle_transformers::models::qwen3::ModelForCausalLM,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let mut started = false;

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let last_logits = &logits_vec[0][0];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// DeepSeek2 Generation
// ============================================================================

fn generate_deepseek2<F>(
    model: &mut candle_transformers::models::deepseek2::DeepSeekV2,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let last_logits = &logits_vec[0];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_deepseek2<F, C>(
    model: &mut candle_transformers::models::deepseek2::DeepSeekV2,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let mut started = false;

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let last_logits = &logits_vec[0];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let token_logits = &logits_vec[0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// Glm4 Generation
// ============================================================================

fn generate_glm4<F>(
    model: &mut candle_transformers::models::glm4_new::ModelForCausalLM,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let last_logits = &logits_vec[0][0];

    let mut next = sample_fn(last_logits, temperature, top_p, top_k)?;
    let mut generated = vec![next];

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temperature, top_p, top_k)?;
        generated.push(next);
    }

    Ok(generated)
}

fn stream_glm4<F, C>(
    model: &mut candle_transformers::models::glm4_new::ModelForCausalLM,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    device: &Device,
    sample_fn: &C,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
    C: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    model.clear_kv_cache();

    let mut started = false;

    let prompt_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, 0)?;
    let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let last_logits = &logits_vec[0][0];

    let mut next = sample_fn(last_logits, temp, top_p, top_k)?;
    if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
        emit(piece)?;
    }

    for idx in 1..max_tokens {
        if let Some(eos) = eos_token {
            if next == eos {
                break;
            }
        }

        let tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, input_ids.len() + idx - 1)?;
        let logits_vec = logits.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let token_logits = &logits_vec[0][0];

        next = sample_fn(token_logits, temp, top_p, top_k)?;
        if let Some(piece) = stream_piece(tokenizer, next, &mut started)? {
            emit(piece)?;
        }
    }

    Ok(())
}

// ============================================================================
// GGUF Generation
// ============================================================================

#[cfg(feature = "gguf")]
fn generate_gguf<F>(
    backend: &mut crate::local::gguf_backend::GgufBackend,
    _input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    _sample_fn: &F,
) -> Result<GenerationResult>
where
    F: Fn(&[f32], f32, f32, Option<usize>) -> Result<u32>,
{
    // GGUF uses its own sampling, so we ignore the sample_fn
    let prompt = ""; // GGUF should handle tokenization internally
    backend.generate_text(prompt, max_tokens, temperature, top_p, top_k, eos_token)
}

#[cfg(feature = "gguf")]
fn stream_gguf<F>(
    backend: &mut crate::local::gguf_backend::GgufBackend,
    _tokenizer: &tokenizers::Tokenizer,
    _input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    _eos_token: Option<u32>,
    emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
{
    let prompt = ""; // GGUF handles tokenization
    backend.generate_text_stream(prompt, max_tokens, temp, top_p, top_k, emit)
}

// ============================================================================
// MLX Generation
// ============================================================================

#[cfg(feature = "mlx")]
fn generate_mlx(
    backend: &mut crate::local::mlx_backend::MlxBackend,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
) -> Result<GenerationResult> {
    backend.generate_text(input_ids, max_tokens, temperature, top_p, top_k, eos_token)
}

#[cfg(feature = "mlx")]
fn stream_mlx<F>(
    backend: &mut crate::local::mlx_backend::MlxBackend,
    tokenizer: &tokenizers::Tokenizer,
    input_ids: &[u32],
    max_tokens: usize,
    temp: f32,
    top_p: f32,
    top_k: Option<usize>,
    eos_token: Option<u32>,
    emit: F,
) -> Result<()>
where
    F: FnMut(String) -> Result<()>,
{
    backend.generate_text_stream(
        input_ids, max_tokens, temp, top_p, top_k, eos_token, emit, tokenizer,
    )
}

//! MLX backend for Apple Silicon GPU acceleration
//!
//! This module provides support for running LLM inference using Apple's MLX
//! framework, which offers optimized performance on Apple Silicon via unified
//! memory and Metal GPU acceleration.
//!
//! Currently supports Llama-family models with safetensors weights.
//! Enable with `--features mlx` (CPU+Accelerate) or `--features mlx-metal` (Metal GPU).

use crate::error::{ModelError, Result};
use crate::local::LocalModelConfig;
use std::path::Path;
use tracing::info;

#[cfg(feature = "mlx")]
use mlx_rs::{transforms::eval, Array};

#[cfg(feature = "mlx")]
mod inner {
    use mlx_rs::{
        error::Exception,
        macros::{ModuleParameters, Quantizable},
        module::{Module, ModuleParametersExt},
        nn,
        quantization::MaybeQuantized,
        Array,
    };
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Debug, Clone, Deserialize)]
    pub struct ModelArgs {
        pub model_type: String,
        pub hidden_size: i32,
        pub num_hidden_layers: i32,
        pub intermediate_size: i32,
        pub num_attention_heads: i32,
        pub rms_norm_eps: f32,
        pub vocab_size: i32,
        pub num_key_value_heads: i32,
        pub max_position_embeddings: i32,
        pub rope_theta: f32,
        #[serde(default)]
        pub head_dim: i32,
        #[serde(default = "default_true")]
        pub tie_word_embeddings: bool,
        #[serde(default)]
        pub attention_bias: bool,
        #[serde(default)]
        pub mlp_bias: bool,
        pub rope_scaling: Option<HashMap<String, serde_json::Value>>,
    }

    fn default_true() -> bool {
        true
    }

    #[derive(Debug, Clone, ModuleParameters, Quantizable)]
    pub struct Attention {
        pub n_heads: i32,
        pub n_kv_heads: i32,
        pub head_dim: i32,
        pub scale: f32,

        #[quantizable]
        #[param]
        pub q_proj: MaybeQuantized<nn::Linear>,
        #[quantizable]
        #[param]
        pub k_proj: MaybeQuantized<nn::Linear>,
        #[quantizable]
        #[param]
        pub v_proj: MaybeQuantized<nn::Linear>,
        #[quantizable]
        #[param]
        pub o_proj: MaybeQuantized<nn::Linear>,
        #[param]
        pub rope: nn::Rope,
    }

    impl Attention {
        pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
            let dim = args.hidden_size;
            let n_heads = args.num_attention_heads;
            let n_kv_heads = args.num_key_value_heads;
            let head_dim = if args.head_dim > 0 {
                args.head_dim
            } else {
                dim / n_heads
            };
            let scale = 1.0 / (head_dim as f32).sqrt();

            let q_proj = nn::Linear::new(dim, n_heads * head_dim, args.attention_bias)?;
            let k_proj = nn::Linear::new(dim, n_kv_heads * head_dim, args.attention_bias)?;
            let v_proj = nn::Linear::new(dim, n_kv_heads * head_dim, args.attention_bias)?;
            let o_proj = nn::Linear::new(n_heads * head_dim, dim, args.attention_bias)?;

            let rope = nn::Rope::new(head_dim, false, args.rope_theta as f64)?;

            Ok(Self {
                n_heads,
                n_kv_heads,
                head_dim,
                scale,
                q_proj: MaybeQuantized::Original(q_proj),
                k_proj: MaybeQuantized::Original(k_proj),
                v_proj: MaybeQuantized::Original(v_proj),
                o_proj: MaybeQuantized::Original(o_proj),
                rope,
            })
        }
    }

    pub trait MlxKvCache {
        fn offset(&self) -> i32;
        fn update_and_fetch(
            &mut self,
            keys: Array,
            values: Array,
        ) -> Result<(Array, Array), Exception>;
    }

    pub struct ConcatCache {
        keys: Option<Array>,
        values: Option<Array>,
        offset: i32,
    }

    impl Default for ConcatCache {
        fn default() -> Self {
            Self {
                keys: None,
                values: None,
                offset: 0,
            }
        }
    }

    impl MlxKvCache for ConcatCache {
        fn offset(&self) -> i32 {
            self.offset
        }

        fn update_and_fetch(
            &mut self,
            keys: Array,
            values: Array,
        ) -> Result<(Array, Array), Exception> {
            self.offset += keys.shape()[2] as i32;
            let (nk, nv) = match (&self.keys, &self.values) {
                (Some(pk), Some(pv)) => {
                    let k = mlx_rs::ops::concat(&[pk, &keys], 2)?;
                    let v = mlx_rs::ops::concat(&[pv, &values], 2)?;
                    (k, v)
                }
                _ => (keys.clone(), values.clone()),
            };
            self.keys = Some(nk.clone());
            self.values = Some(nv.clone());
            Ok((nk, nv))
        }
    }

    pub fn sdpa(
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
    ) -> Result<Array, Exception> {
        let scores = queries
            .matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(&mlx_rs::array!(scale))?;
        let weights = mlx_rs::nn::softmax(&scores, -1, None)?;
        weights.matmul(values)
    }

    impl Attention {
        pub fn forward_with_cache(
            &mut self,
            x: &Array,
            mask: Option<&Array>,
            cache: Option<&mut dyn MlxKvCache>,
        ) -> Result<Array, Exception> {
            use mlx_rs::ops::indexing::NewAxis;
            let shape = x.shape();
            let b = shape[0];
            let l = shape[1];

            let mut q = self
                .q_proj
                .forward(x)?
                .reshape(&[b, l, self.n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let mut k = self
                .k_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let mut v = self
                .v_proj
                .forward(x)?
                .reshape(&[b, l, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;

            if let Some(cache) = cache {
                let qi = nn::RopeInput::new_with_offset(&q, cache.offset());
                q = self.rope.forward(qi)?;
                let ki = nn::RopeInput::new_with_offset(&k, cache.offset());
                k = self.rope.forward(ki)?;
                (k, v) = cache.update_and_fetch(k, v)?;
            } else {
                q = self.rope.forward(nn::RopeInput::new(&q))?;
                k = self.rope.forward(nn::RopeInput::new(&k))?;
            }

            let mut output = sdpa(&q, &k, &v, self.scale)?;
            if let Some(m) = mask {
                output = output.add(m)?;
            }
            output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;
            self.o_proj.forward(&output)
        }
    }

    #[derive(Debug, Clone, ModuleParameters, Quantizable)]
    pub struct Mlp {
        #[quantizable]
        #[param]
        pub gate_proj: MaybeQuantized<nn::Linear>,
        #[quantizable]
        #[param]
        pub down_proj: MaybeQuantized<nn::Linear>,
        #[quantizable]
        #[param]
        pub up_proj: MaybeQuantized<nn::Linear>,
    }

    impl Mlp {
        pub fn new(dim: i32, hidden_dim: i32, bias: bool) -> Result<Self, Exception> {
            Ok(Self {
                gate_proj: MaybeQuantized::Original(nn::Linear::new(dim, hidden_dim, bias)?),
                down_proj: MaybeQuantized::Original(nn::Linear::new(hidden_dim, dim, bias)?),
                up_proj: MaybeQuantized::Original(nn::Linear::new(dim, hidden_dim, bias)?),
            })
        }

        pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
            let gate = self.gate_proj.forward(x)?;
            let up = self.up_proj.forward(x)?;
            self.down_proj.forward(&nn::silu(gate)?.multiply(&up)?)
        }
    }

    #[derive(Debug, Clone, ModuleParameters, Quantizable)]
    pub struct TransformerBlock {
        #[quantizable]
        #[param]
        pub self_attn: Attention,
        #[quantizable]
        #[param]
        pub mlp: Mlp,
        #[param]
        pub input_layernorm: nn::RmsNorm,
        #[param]
        pub post_attention_layernorm: nn::RmsNorm,
    }

    impl TransformerBlock {
        pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
            Ok(Self {
                self_attn: Attention::new(args)?,
                mlp: Mlp::new(args.hidden_size, args.intermediate_size, args.mlp_bias)?,
                input_layernorm: nn::RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
                post_attention_layernorm: nn::RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            })
        }

        pub fn forward_with_cache(
            &mut self,
            x: &Array,
            mask: Option<&Array>,
            cache: Option<&mut dyn MlxKvCache>,
        ) -> Result<Array, Exception> {
            let r = self.self_attn.forward_with_cache(
                &self.input_layernorm.forward(x)?,
                mask,
                cache,
            )?;
            let h = x.add(&r)?;
            let r2 = self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&h)?)?;
            h.add(&r2)
        }
    }

    #[derive(Debug, Clone, ModuleParameters, Quantizable)]
    pub struct LlamaModel {
        pub vocab_size: i32,
        pub n_layers: i32,
        #[quantizable]
        #[param]
        pub embed_tokens: MaybeQuantized<nn::Embedding>,
        #[quantizable]
        #[param]
        pub layers: Vec<TransformerBlock>,
        #[param]
        pub norm: nn::RmsNorm,
    }

    impl LlamaModel {
        pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
            Ok(Self {
                vocab_size: args.vocab_size,
                n_layers: args.num_hidden_layers,
                embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                    args.vocab_size,
                    args.hidden_size,
                )?),
                layers: (0..args.num_hidden_layers)
                    .map(|_| TransformerBlock::new(args))
                    .collect::<Result<Vec<_>, _>>()?,
                norm: nn::RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            })
        }

        pub fn forward_with_caches(
            &mut self,
            input_ids: &Array,
            caches: &mut Vec<Option<ConcatCache>>,
        ) -> Result<Array, Exception> {
            let mut h = self.embed_tokens.forward(input_ids)?;

            let mask = if h.shape()[1] > 1 {
                let m = nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?;
                Some(m.as_dtype(h.dtype())?)
            } else {
                None
            };

            if caches.is_empty() {
                *caches = (0..self.layers.len())
                    .map(|_| Some(ConcatCache::default()))
                    .collect();
            }

            for (layer, c) in self.layers.iter_mut().zip(caches.iter_mut()) {
                h = layer.forward_with_cache(&h, mask.as_ref(), c.as_mut())?;
            }

            self.norm.forward(&h)
        }
    }

    #[derive(Debug, Clone, ModuleParameters, Quantizable)]
    pub struct Model {
        pub args: ModelArgs,
        #[quantizable]
        #[param]
        pub model: LlamaModel,
        #[quantizable]
        #[param]
        pub lm_head: Option<MaybeQuantized<nn::Linear>>,
    }

    impl Model {
        pub fn new(args: ModelArgs) -> Result<Self, Exception> {
            let model = LlamaModel::new(&args)?;
            let lm_head = if !args.tie_word_embeddings {
                Some(MaybeQuantized::Original(nn::Linear::new(
                    args.hidden_size,
                    args.vocab_size,
                    false,
                )?))
            } else {
                None
            };
            Ok(Self {
                args,
                model,
                lm_head,
            })
        }

        pub fn forward_with_caches(
            &mut self,
            input_ids: &Array,
            caches: &mut Vec<Option<ConcatCache>>,
        ) -> Result<Array, Exception> {
            let out = self.model.forward_with_caches(input_ids, caches)?;
            match self.lm_head.as_mut() {
                Some(head) => head.forward(&out),
                None => match &mut self.model.embed_tokens {
                    MaybeQuantized::Original(e) => e.as_linear(&out),
                    MaybeQuantized::Quantized(q) => q.as_linear(&out),
                },
            }
        }
    }
}

#[cfg(feature = "mlx")]
pub struct MlxBackend {
    model: inner::Model,
}

#[cfg(not(feature = "mlx"))]
pub struct MlxBackend {
    _private: (),
}

#[cfg(feature = "mlx")]
impl MlxBackend {
    pub fn load(config: &LocalModelConfig) -> Result<Self> {
        let model_path = &config.model_path;
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(ModelError::LocalModelError(format!(
                "config.json not found in {}\n\nHint: Ensure the model directory contains all required files.",
                model_path.display()
            )));
        }

        let config_content = std::fs::read_to_string(&config_path)?;
        let mut args: inner::ModelArgs = serde_json::from_str(&config_content).map_err(|e| {
            ModelError::LocalModelError(format!("Failed to parse config.json: {}", e))
        })?;

        if args.head_dim == 0 {
            args.head_dim = args.hidden_size / args.num_attention_heads;
        }

        info!(
            "MLX config: vocab={}, hidden={}, layers={}, heads={}",
            args.vocab_size, args.hidden_size, args.num_hidden_layers, args.num_attention_heads
        );

        let mut model = inner::Model::new(args).map_err(|e| {
            ModelError::LocalModelError(format!("Failed to create MLX model: {}", e))
        })?;

        let weight_files = find_weight_files(model_path);
        if weight_files.is_empty() {
            return Err(ModelError::LocalModelError(format!(
                "No .safetensors files found in {}",
                model_path.display()
            )));
        }

        info!("Loading {} MLX weight file(s)...", weight_files.len());
        for wf in &weight_files {
            model.load_safetensors(wf).map_err(|e| {
                ModelError::LocalModelError(format!("Failed to load MLX weights: {}", e))
            })?;
        }

        Ok(Self { model })
    }

    pub fn generate_text(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        eos_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        use mlx_rs::ops::indexing::{IndexOp, NewAxis};

        let prompt = Array::from(input_ids).index(NewAxis);
        let mut caches = Vec::new();

        let logits = self
            .model
            .forward_with_caches(&prompt, &mut caches)
            .map_err(|e| ModelError::LocalModelError(format!("MLX prefill failed: {}", e)))?;

        let last = logits.index((.., -1, ..));
        let mut y = sample_token(&last, temperature)?;
        eval([&y]).map_err(|e| ModelError::LocalModelError(format!("MLX eval failed: {}", e)))?;

        let mut generated = vec![y.item::<u32>()];

        for _ in 1..max_tokens {
            if let Some(eos) = eos_token {
                if generated.last() == Some(&eos) {
                    break;
                }
            }

            let inp = y.index((.., NewAxis));
            let logits = self
                .model
                .forward_with_caches(&inp, &mut caches)
                .map_err(|e| ModelError::LocalModelError(format!("MLX decode failed: {}", e)))?;
            y = sample_token(&logits, temperature)?;
            eval([&y])
                .map_err(|e| ModelError::LocalModelError(format!("MLX eval failed: {}", e)))?;
            generated.push(y.item::<u32>());
        }

        Ok(generated)
    }

    pub fn generate_text_stream<F>(
        &mut self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
        eos_token: Option<u32>,
        mut emit: F,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        use crate::local::tokenization::stream_piece;
        use mlx_rs::ops::indexing::{IndexOp, NewAxis};

        let prompt = Array::from(input_ids).index(NewAxis);
        let mut caches = Vec::new();
        let mut started = false;

        let logits = self
            .model
            .forward_with_caches(&prompt, &mut caches)
            .map_err(|e| ModelError::LocalModelError(format!("MLX prefill failed: {}", e)))?;

        let last = logits.index((.., -1, ..));
        let mut y = sample_token(&last, temperature)?;
        eval([&y]).map_err(|e| ModelError::LocalModelError(format!("MLX eval failed: {}", e)))?;
        let mut tid = y.item::<u32>();

        if let Some(piece) = stream_piece(tokenizer, tid, &mut started)? {
            emit(piece)?;
        }

        for _ in 1..max_tokens {
            if let Some(eos) = eos_token {
                if tid == eos {
                    break;
                }
            }

            let inp = y.index((.., NewAxis));
            let logits = self
                .model
                .forward_with_caches(&inp, &mut caches)
                .map_err(|e| ModelError::LocalModelError(format!("MLX decode failed: {}", e)))?;
            y = sample_token(&logits, temperature)?;
            eval([&y])
                .map_err(|e| ModelError::LocalModelError(format!("MLX eval failed: {}", e)))?;
            tid = y.item::<u32>();

            if let Some(piece) = stream_piece(tokenizer, tid, &mut started)? {
                emit(piece)?;
            }
        }

        Ok(())
    }

    pub fn model_type(&self) -> &str {
        &self.model.args.model_type
    }
}

#[cfg(feature = "mlx")]
fn sample_token(logits: &Array, temperature: f32) -> Result<Array> {
    if temperature == 0.0 {
        mlx_rs::ops::indexing::ArgMaxAxis::new(logits, -1, None)
            .run()
            .map_err(|e| ModelError::LocalModelError(format!("MLX argmax: {}", e)))
    } else {
        let scaled = logits
            .multiply(&mlx_rs::array!(1.0f32 / temperature))
            .map_err(|e| ModelError::LocalModelError(format!("MLX scale: {}", e)))?;
        mlx_rs::random::categorical(&scaled, None, None, None)
            .map_err(|e| ModelError::LocalModelError(format!("MLX sample: {}", e)))
    }
}

fn find_weight_files(model_path: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".safetensors") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

#[cfg(not(feature = "mlx"))]
impl MlxBackend {
    pub fn load(_config: &LocalModelConfig) -> Result<Self> {
        Err(ModelError::InvalidConfig(
            "MLX support not enabled. Build with --features mlx".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "mlx"))]
    fn test_mlx_disabled() {
        let config = LocalModelConfig::default();
        let result = MlxBackend::load(&config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("MLX support not enabled"));
    }
}

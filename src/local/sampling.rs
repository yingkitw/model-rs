use crate::error::{ModelError, Result};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn do_sample(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
) -> Result<u32> {
    let vocab_size = logits.len();

    if temperature == 0.0 {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        return Ok(max_idx as u32);
    }

    let scaled: Vec<f32> = logits.iter().map(|&logit| logit / temperature).collect();

    let top_k = top_k.unwrap_or(vocab_size);
    let mut sorted_indices: Vec<usize> = (0..vocab_size).collect();
    sorted_indices.sort_by(|&a, &b| {
        scaled[b]
            .partial_cmp(&scaled[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    sorted_indices.truncate(top_k);

    let mut probs: Vec<f32> = sorted_indices
        .iter()
        .map(|&i| {
            let logit = scaled[i];
            logit.exp()
        })
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum == 0.0 {
        return Ok(sorted_indices[0] as u32);
    }

    for prob in probs.iter_mut() {
        *prob /= sum;
    }

    if top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut cutoff_idx = sorted_indices.len();

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= top_p {
                cutoff_idx = idx + 1;
                break;
            }
        }

        sorted_indices.truncate(cutoff_idx);
        probs.truncate(cutoff_idx);

        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs.iter_mut() {
                *prob /= sum;
            }
        }
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| ModelError::LocalModelError(format!("Failed to get time: {}", e)))?
        .as_nanos() as f32;

    let random_value = (nanos % 1000000.0) / 1000000.0;
    let mut cumulative = 0.0;

    for (idx, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if random_value <= cumulative {
            return Ok(sorted_indices[idx] as u32);
        }
    }

    Ok(sorted_indices[sorted_indices.len() - 1] as u32)
}

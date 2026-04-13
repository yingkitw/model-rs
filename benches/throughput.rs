//! Criterion benchmarks for hot paths (sampling, local model index refresh).
//!
//! Run from the repository root:
//! ```text
//! cargo bench --bench throughput
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use model_rs::local::do_sample;
use model_rs::models::refresh_models_index;
use std::fs;
use tempfile::TempDir;

fn logits_vec(vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|i| ((i as f32) * 0.01).sin() * 2.0)
        .collect()
}

fn bench_do_sample_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("do_sample_greedy");
    for vocab in [1024_usize, 4096, 8192] {
        let logits = logits_vec(vocab);
        group.bench_with_input(BenchmarkId::from_parameter(vocab), &logits, |b, lg| {
            b.iter(|| {
                let _ = do_sample(black_box(lg.as_slice()), 0.0, 1.0, None).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_do_sample_sampled(c: &mut Criterion) {
    let vocab = 4096_usize;
    let logits = logits_vec(vocab);
    c.bench_function("do_sample_sampled_vocab_4096", |b| {
        b.iter(|| {
            let _ = do_sample(black_box(&logits), 0.8, 0.9, Some(50)).unwrap();
        });
    });
}

fn bench_refresh_models_index(c: &mut Criterion) {
    let tmp = TempDir::new().expect("tempdir");
    let root = tmp.path();
    for i in 0..16 {
        let d = root.join(format!("model-{i}"));
        fs::create_dir_all(&d).expect("mkdir");
        fs::write(d.join("w.safetensors"), b"x").expect("write");
        fs::write(d.join("config.json"), r#"{"model_type":"llama"}"#).expect("config");
    }
    c.bench_function("refresh_models_index_16_dirs", |b| {
        b.iter(|| {
            refresh_models_index(Some(root)).expect("refresh");
        });
    });
}

criterion_group!(
    benches,
    bench_do_sample_greedy,
    bench_do_sample_sampled,
    bench_refresh_models_index
);
criterion_main!(benches);

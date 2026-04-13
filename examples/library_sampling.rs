//! Minimal library usage: token sampling without loading a model.
//!
//! Run from the repository root:
//! ```text
//! cargo run --example library_sampling
//! ```

fn main() {
    let logits = vec![0.1_f32, 2.0, 0.5, -0.3];
    let idx = model_rs::local::do_sample(&logits, 0.0, 1.0, None).expect("greedy argmax");
    println!("Greedy token id (argmax): {idx}");
}

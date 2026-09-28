//! Layer-by-layer divergence isolation for Kimi-K3-0.40B.
//!
//! This test runs the Rust forward pass step by step, dumping the hidden state
//! after each layer, and compares against the Python reference per-layer outputs.
//!
//! Run with:
//! ```sh
//! cargo test --features kimi_k3_loader --test kimi_k3_layer_diff -- --nocapture
//! ```

#![cfg(feature = "kimi_k3_loader")]

use std::path::Path;

use katgpt_rs::kimi_k3::loader::load_kimi_k3;
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime};

fn model_dir() -> String {
    std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    })
}

fn model_path() -> String {
    format!("{}/model.safetensors", model_dir())
}

/// Load raw f32 from a binary file.
fn load_raw_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    bytemuck::cast_slice(&bytes).to_vec()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize, f64) {
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    let mut sum = 0.0f64;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
        sum += d as f64;
    }
    (max_diff, max_idx, sum / a.len() as f64)
}

#[test]
fn layer_by_layer_diff() {
    use katgpt_rs::kimi_k3::decoder_layer::kimi_decoder_layer_forward;

    let ref_dir = format!(
        "{}/scripts/kimi_ref/ref_output/ref_layers",
        env!("CARGO_MANIFEST_DIR")
    );

    if !Path::new(&ref_dir).exists() {
        eprintln!("skipping: reference layers not found at {ref_dir}");
        eprintln!("Generate with: python scripts/kimi_ref/run_reference.py");
        return;
    }

    let weights = load_kimi_k3(&model_path()).unwrap_or_else(|e| panic!("load failed: {e}"));
    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let mut runtime = KimiK3Runtime::new(&config, 64);

    let d = config.hidden_size;
    let token_id = 1u32;

    // Step 1: Embedding
    let embed_start = (token_id as usize) * d;
    let embed_end = embed_start + d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_end]);

    // Compare embedding
    let ref_embed = load_raw_f32(&format!("{ref_dir}/layer_embed.bin"));
    let (diff, idx, mean) = max_abs_diff(&runtime.hidden, &ref_embed);
    eprintln!("embed:     max_diff={diff:.4e} at [{idx}], mean={mean:.4e}");

    // Step 2: Run each layer + compare
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_rt = &mut runtime.layers[layer_idx];

        kimi_decoder_layer_forward(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
        );

        let ref_path = format!("{ref_dir}/layer_{layer_idx}_out.bin");
        let ref_out = load_raw_f32(&ref_path);
        let (diff, idx, mean) = max_abs_diff(&runtime.hidden, &ref_out);

        let layer_type = if layer_idx == 3 || layer_idx == 7 {
            "MLA"
        } else {
            "KDA"
        };
        let ffn_type = if layer_idx == 0 { "Dense" } else { "MoE" };
        eprintln!(
            "layer {layer_idx} ({layer_type}+{ffn_type}): max_diff={diff:.4e} at [{idx}], mean={mean:.4e}  \
             rust[{idx}]={:.6}, ref[{idx}]={:.6}",
            runtime.hidden[idx], ref_out[idx]
        );
    }

    // Step 3: Output attn-res
    if !runtime.block_state.is_empty() {
        use katgpt_transformer::attn_res::apply_attn_res;
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);

        let ref_norm = load_raw_f32(&format!("{ref_dir}/layer_final_norm.bin"));
        // Note: Python dumps final_norm AFTER rmsnorm, but here we're BEFORE rmsnorm.
        // The "final_norm" in Python is the output of model.norm, not pre-norm.
        // So we can't compare directly here — skip this step.
        let _ = ref_norm;
    }

    eprintln!("\nDivergence isolation complete.");
}

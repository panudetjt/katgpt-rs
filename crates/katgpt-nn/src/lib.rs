//! Shared neural network primitives — the common substrate for both the CNN
//! engine (`katgpt-moka-wasm`) and the transformer engine (`katgpt-forward`).
//!
//! # Why this crate exists
//!
//! Before this crate, the CNN and transformer engines shared exactly ONE
//! primitive (`simd_dot_f32` from `katgpt-types`). Everything else was
//! duplicated: Moka had its own `conv2d_into`, `linear_into`,
//! `global_mean_max_into`, `relu_inplace` in `moka.rs`; the transformer had
//! its own `matmul` paths in `katgpt-forward`. This crate is Path A of the
//! CNN→Transformer code-path unification (Issue 567): extract the shared NN
//! operations so both engines import from the same source.
//!
//! # Design rules (for Path B compatibility)
//!
//! All functions take `&mut [f32]` scratch buffers as parameters (caller-
//! allocated), NOT owning their own `ForwardContext`. This keeps them
//! composable for a future unified `Layer` enum dispatch (Path B) where a
//! single `forward()` routes to these primitives per layer type.
//!
//! # wasm32 compatibility
//!
//! Depends only on `katgpt-types` (the leaf SIMD crate). No `katgpt-core`
//! dependency → no `ahash`/`getrandom` wasm32 backend friction. Both the CNN
//! and transformer engines can import this crate on all targets.

use katgpt_types::simd::simd_dot_f32;

// ── Dot product helper ────────────────────────────────────────────────────

/// Bias-aware dot product: `init + dot(a, b)`.
///
/// This is the one function that directly calls the shared SIMD primitive
/// (`simd_dot_f32`). All higher-level operations (`conv2d_into`, `linear_into`)
/// route through this helper.
#[inline]
pub fn dot_lanes(a: &[f32], b: &[f32], init: f32) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    init + simd_dot_f32(a, b, a.len().min(b.len()))
}

// ── 2D Convolution ────────────────────────────────────────────────────────

/// 2D convolution with zero-padding (pad = k/2), HWC layout, weight layout
/// `[out_ch, k, k, in_ch]` flattened row-major.
///
/// `patch` is caller-allocated scratch of length `k * k * in_ch`. For k=1
/// (1×1 conv), the fast path skips patch gathering entirely — input slice IS
/// the patch.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_into(
    input: &[f32],
    h: usize,
    w: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    weight: &[f32],
    bias: &[f32],
    patch: &mut [f32],
    out: &mut [f32],
) {
    let patch_len = k * k * in_ch;
    if k == 1 {
        for pos in 0..h * w {
            let pslice = &input[pos * in_ch..pos * in_ch + in_ch];
            let obase = pos * out_ch;
            for oc in 0..out_ch {
                let wbase = oc * in_ch;
                out[obase + oc] = dot_lanes(pslice, &weight[wbase..wbase + in_ch], bias[oc]);
            }
        }
        return;
    }

    let pad = k / 2;
    for y in 0..h {
        for x in 0..w {
            patch[..patch_len].fill(0.0);
            for ky in 0..k {
                let iy = y + ky;
                if iy < pad || iy >= h + pad {
                    continue;
                }
                let iy = iy - pad;
                for kx in 0..k {
                    let ix = x + kx;
                    if ix < pad || ix >= w + pad {
                        continue;
                    }
                    let ix = ix - pad;
                    let src = (iy * w + ix) * in_ch;
                    let dst = (ky * k + kx) * in_ch;
                    patch[dst..dst + in_ch].copy_from_slice(&input[src..src + in_ch]);
                }
            }

            let obase = (y * w + x) * out_ch;
            let pslice = &patch[..patch_len];
            for oc in 0..out_ch {
                let wbase = oc * patch_len;
                out[obase + oc] = dot_lanes(pslice, &weight[wbase..wbase + patch_len], bias[oc]);
            }
        }
    }
}

/// Batched 2D convolution. `batch = K` samples, each `h×w×in_ch` HWC,
/// written sample-major into `outs` (length `batch * h * w * out_ch`).
/// `patches` is scratch of length `batch * k * k * in_ch`.
///
/// Falls back to the per-sample code structure when `batch == 1` so the
/// K=1 path has zero overhead vs `conv2d_into` (modulo the sample stride).
///
/// The key restructure vs the per-sample `conv2d_into` is that the weight
/// slice for each output channel is loaded ONCE and reused across all K
/// samples in the inner loop — the cache-locality win for batched MCTS.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_batched_into(
    inputs: &[f32],
    batch: usize,
    h: usize,
    w: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    weight: &[f32],
    bias: &[f32],
    patches: &mut [f32],
    outs: &mut [f32],
) {
    let patch_len = k * k * in_ch;
    let in_sample = h * w * in_ch;
    let out_sample = h * w * out_ch;

    if k == 1 {
        for pos in 0..h * w {
            let obase = pos * out_ch;
            for oc in 0..out_ch {
                let wbase = oc * in_ch;
                let w_slice = &weight[wbase..wbase + in_ch];
                let b = bias[oc];
                for s in 0..batch {
                    let pslice = &inputs[s * in_sample + pos * in_ch..][..in_ch];
                    outs[s * out_sample + obase + oc] = dot_lanes(pslice, w_slice, b);
                }
            }
        }
        return;
    }

    let pad = k / 2;
    for y in 0..h {
        for x in 0..w {
            for s in 0..batch {
                let input = &inputs[s * in_sample..];
                let patch = &mut patches[s * patch_len..];
                patch[..patch_len].fill(0.0);
                for ky in 0..k {
                    let iy = y + ky;
                    if iy < pad || iy >= h + pad {
                        continue;
                    }
                    let iy = iy - pad;
                    for kx in 0..k {
                        let ix = x + kx;
                        if ix < pad || ix >= w + pad {
                            continue;
                        }
                        let ix = ix - pad;
                        let src = (iy * w + ix) * in_ch;
                        let dst = (ky * k + kx) * in_ch;
                        patch[dst..dst + in_ch].copy_from_slice(&input[src..src + in_ch]);
                    }
                }
            }

            let obase = (y * w + x) * out_ch;
            for oc in 0..out_ch {
                let wbase = oc * patch_len;
                let w_slice = &weight[wbase..wbase + patch_len];
                let b = bias[oc];
                for s in 0..batch {
                    let pslice = &patches[s * patch_len..][..patch_len];
                    outs[s * out_sample + obase + oc] = dot_lanes(pslice, w_slice, b);
                }
            }
        }
    }
}

// ── Linear (fully-connected) ──────────────────────────────────────────────

/// Fully-connected layer: `out[o] = bias[o] + dot(weight[o], input)`.
/// Weight layout is row-major `[out_dim][in_dim]`.
pub fn linear_into(
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
) {
    for o in 0..out_dim {
        let base = o * in_dim;
        out[o] = dot_lanes(&input[..in_dim], &weight[base..base + in_dim], bias[o]);
    }
}

/// Batched fully-connected: `out[s][o] = bias[o] + dot(weight[o], input[s])`.
/// Weight layout is row-major `[out_dim][in_dim]`, shared across all K samples.
pub fn linear_batched_into(
    inputs: &[f32],
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
    outs: &mut [f32],
) {
    for o in 0..out_dim {
        let base = o * in_dim;
        let w_slice = &weight[base..base + in_dim];
        let b = bias[o];
        for s in 0..batch {
            outs[s * out_dim + o] = dot_lanes(&inputs[s * in_dim..][..in_dim], w_slice, b);
        }
    }
}

// ── Spatial pooling ───────────────────────────────────────────────────────

/// Global mean+max pool over the spatial dims, leaving a `[mean(ch); ch]`
/// followed by `[max(ch); ch]` vector of length `ch * 2`.
pub fn global_mean_max_into(x: &[f32], h: usize, w: usize, ch: usize, out: &mut [f32]) {
    let (mean, max) = out[..ch * 2].split_at_mut(ch);
    mean.fill(0.0);
    max.fill(f32::MIN);
    for pos in 0..h * w {
        let row = &x[pos * ch..pos * ch + ch];
        for c in 0..ch {
            let v = row[c];
            mean[c] += v;
            if v > max[c] {
                max[c] = v;
            }
        }
    }
    let n = (h * w) as f32;
    for m in mean.iter_mut() {
        *m /= n;
    }
}

/// Batched global mean+max pool over the spatial dims, leaving a per-sample
/// `[mean(ch); ch][max(ch); ch]` vector of length `ch * 2`.
pub fn global_mean_max_batched_into(
    inputs: &[f32],
    batch: usize,
    h: usize,
    w: usize,
    ch: usize,
    outs: &mut [f32],
) {
    let n = (h * w) as f32;
    for s in 0..batch {
        let input = &inputs[s * h * w * ch..];
        let out = &mut outs[s * ch * 2..];
        let (mean, max) = out[..ch * 2].split_at_mut(ch);
        mean.fill(0.0);
        max.fill(f32::MIN);
        for pos in 0..h * w {
            let row = &input[pos * ch..pos * ch + ch];
            for c in 0..ch {
                let v = row[c];
                mean[c] += v;
                if v > max[c] {
                    max[c] = v;
                }
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
    }
}

// ── Activations ───────────────────────────────────────────────────────────

/// In-place ReLU: clamps all negative values to 0.
pub fn relu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_inplace() {
        let mut x = vec![-1.0, 0.0, 1.0, -0.5, 0.5];
        relu_inplace(&mut x);
        assert_eq!(x, vec![0.0, 0.0, 1.0, 0.0, 0.5]);
    }

    #[test]
    fn test_linear_into() {
        // 2×3 weight matrix, 3-dim input → 2-dim output.
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // identity-ish
        let bias = vec![0.5, 0.5];
        let input = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0f32; 2];
        linear_into(&input, 3, 2, &weight, &bias, &mut out);
        // out[0] = dot([1,0,0], [1,2,3]) + 0.5 = 1.5
        // out[1] = dot([0,1,0], [1,2,3]) + 0.5 = 2.5
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_1x1() {
        // 1×1 conv = per-position matmul. 1×1 input, 1 in_ch, 1 out_ch.
        let input = vec![2.0];
        let weight = vec![3.0];
        let bias = vec![1.0];
        let mut patch = vec![0.0f32; 1];
        let mut out = vec![0.0f32; 1];
        conv2d_into(&input, 1, 1, 1, 1, 1, &weight, &bias, &mut patch, &mut out);
        // out = 2.0 * 3.0 + 1.0 = 7.0
        assert!((out[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_conv2d_3x3_identity() {
        // 3×3 conv with identity kernel (center weight = 1, rest = 0) on a
        // 3×3 input with 1 channel → output should equal input (zero-padded
        // edges → 0).
        let input = vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ];
        // Identity 3×3 kernel: center = 1, rest = 0.
        let weight = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let bias = vec![0.0];
        let mut patch = vec![0.0f32; 9];
        let mut out = vec![0.0f32; 9];
        conv2d_into(&input, 3, 3, 1, 1, 3, &weight, &bias, &mut patch, &mut out);
        // Center of the output = center of input = 5.0.
        // Corners/edges see zeros from padding → output = input value only
        // if the kernel center aligns (it does for all positions).
        assert!(
            (out[4] - 5.0).abs() < 1e-6,
            "center should be 5.0, got {}",
            out[4]
        );
        // Corner (0,0): patch = [0,0,0, 0,1,0, 0,0,0] (only center input=1).
        // dot with weight = 1*1 = 1.0 (center of the 3×3 window at (0,0) is
        // input[0][0] = 1.0, rest is zero-padded).
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "corner (0,0) should be 1.0, got {}",
            out[0]
        );
    }

    #[test]
    fn test_global_mean_max_into() {
        // 2×2 input, 2 channels.
        let input = vec![
            1.0, 2.0, // pos (0,0)
            3.0, 4.0, // pos (0,1)
            5.0, 6.0, // pos (1,0)
            7.0, 8.0, // pos (1,1)
        ];
        let mut out = vec![0.0f32; 4]; // 2 mean + 2 max
        global_mean_max_into(&input, 2, 2, 2, &mut out);
        // mean[0] = (1+3+5+7)/4 = 4.0, mean[1] = (2+4+6+8)/4 = 5.0
        // max[0] = 7.0, max[1] = 8.0
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
        assert!((out[2] - 7.0).abs() < 1e-6);
        assert!((out[3] - 8.0).abs() < 1e-6);
    }
}

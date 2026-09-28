//! ShortConv1D — causal depthwise 1D convolution with ring buffer.
//!
//! Implements the "short conv" primitive used by linear attention variants
//! (DeltaNet, GLA, KDA). Applied after the q/k/v linear projection and before
//! the activation (Swish), it mixes local token context into each channel.
//!
//! # Shape
//!
//! - **Depthwise**: each channel has its own `kernel_size` taps (no cross-channel mixing).
//! - **Causal**: only looks at the current + past `kernel_size - 1` samples (never the future).
//! - **Per-channel filter layout**: `weight[c * kernel_size + k]` weights the sample
//!   `k` timesteps ago for channel `c`. So `weight[c * kernel_size + 0]` is the
//!   tap for the **current** (newest) sample and `weight[c * kernel_size + kernel_size - 1]`
//!   is the tap for the **oldest** sample in the receptive field.
//!
//! # Ring buffer
//!
//! For single-token decode, a ring buffer of the last `kernel_size` inputs per
//! channel is maintained. Each forward step:
//!   1. Writes the new input at `buf_idx` (overwriting the oldest entry).
//!   2. Computes the conv output by walking the ring buffer in chronological order
//!      (newest first) and multiplying by the per-channel taps.
//!   3. Advances `buf_idx = (buf_idx + 1) % kernel_size`.
//!
//! At sequence start, the ring buffer is zero-initialized (zero-padding for the
//! first `kernel_size - 1` tokens).
//!
//! # Zero allocation
//!
//! All state (`weight`, `buf`, `buf_idx`) lives on the `ShortConv1D` struct and
//! is pre-allocated at construction. The forward path takes a `&mut self` + input
//! and output slices — no allocations in the hot path (G4 alloc-free).
//!
//! # Reference
//!
//! - Kimi Linear (arxiv 2510.26692) §4 + §5.2 ablation: kernel_size = 4.
//! - Standard short conv from the Mamba/HGRN/GLA lineage.

/// Causal depthwise 1D convolution with ring buffer.
///
/// See module docs for the full shape + causality conventions.
#[derive(Clone)]
pub struct ShortConv1D {
    /// Per-channel FIR taps, flattened: `weight[c * kernel_size + k]` is the tap
    /// for the sample `k` timesteps ago (k=0 = current/newest) on channel `c`.
    /// Shape: `[n_channels * kernel_size]`.
    pub weight: Vec<f32>,

    /// Ring buffer of recent inputs. `buf[c * kernel_size + i]` is channel `c`'s
    /// value at ring slot `i`. Slot `buf_idx` (post-advance) points at the
    /// position where the NEXT input will be written; the just-written input is
    /// at slot `(buf_idx - 1 + kernel_size) % kernel_size` (= newest).
    /// Shape: `[n_channels * kernel_size]`.
    pub buf: Vec<f32>,

    /// Current ring position (next write index). Wraps modulo `kernel_size`.
    pub buf_idx: usize,

    /// Number of channels (depthwise — each convolved independently).
    pub n_channels: usize,

    /// Convolution kernel size ( receptive field = kernel_size timesteps).
    pub kernel_size: usize,
}

impl ShortConv1D {
    /// Allocate a zeroed short conv with unit-initialized weights (identity passthrough
    /// on the current sample, zero on past samples — i.e. `weight[c*K+0] = 1.0`, rest 0).
    ///
    /// The weights are placeholders; production code overrides them from safetensors.
    pub fn new(n_channels: usize, kernel_size: usize) -> Self {
        debug_assert!(kernel_size >= 1, "kernel_size must be >= 1");
        debug_assert!(n_channels >= 1, "n_channels must be >= 1");

        let mut weight = vec![0.0f32; n_channels * kernel_size];
        // Identity initialization: weight[c*kernel_size + 0] = 1.0 (current sample passthrough).
        // This makes the conv a no-op until real weights are loaded.
        for c in 0..n_channels {
            weight[c * kernel_size] = 1.0;
        }

        Self {
            weight,
            buf: vec![0.0; n_channels * kernel_size],
            buf_idx: 0,
            n_channels,
            kernel_size,
        }
    }

    /// Reset the ring buffer to zeros (reuse allocations). Weights are NOT reset
    /// — they are model parameters and should persist across sequences.
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.buf_idx = 0;
    }

    /// Forward step: push `x` into the ring buffer, compute depthwise causal conv.
    ///
    /// - `x`: input `[n_channels]` (the new sample at the current timestep).
    /// - `out`: output `[n_channels]` (the conv result, same length as input).
    ///
    /// After this call, the ring buffer holds `x` at the newest slot and the
    /// `kernel_size - 1` previous inputs in chronological order behind it.
    ///
    /// Zero allocations in the hot path.
    #[inline]
    pub fn forward(&mut self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.n_channels, "input length mismatch");
        debug_assert_eq!(out.len(), self.n_channels, "output length mismatch");
        let ks = self.kernel_size;
        let nc = self.n_channels;

        // 1. Write the new input at the current buf_idx slot (overwriting the oldest).
        //    After this write, the slot at `buf_idx` holds the newest sample.
        for (c, &xc) in x.iter().enumerate().take(nc) {
            self.buf[c * ks + self.buf_idx] = xc;
        }
        // 2. Advance buf_idx to the NEXT write position. The just-written sample
        //    is now at slot `(buf_idx - 1 + ks) % ks` = the newest.
        let new_buf_idx = (self.buf_idx + 1) % ks;

        // 3. For each channel, walk the ring buffer newest-first and apply taps.
        //    Tap k=0 weights the newest sample, tap k=ks-1 weights the oldest.
        //    Newest slot = (new_buf_idx - 1 + ks) % ks = self.buf_idx (pre-advance).
        //    Sample k timesteps ago is at slot (self.buf_idx - k + ks) % ks.
        let newest_slot = self.buf_idx; // slot of the just-written (newest) sample
        for (c, out_slot) in out.iter_mut().enumerate().take(nc) {
            let mut acc = 0.0f32;
            let w_off = c * ks;
            let b_off = c * ks;
            for k in 0..ks {
                let tap = self.weight[w_off + k];
                if tap != 0.0 {
                    let slot = (newest_slot + ks - k) % ks;
                    acc += tap * self.buf[b_off + slot];
                }
            }
            *out_slot = acc;
        }

        self.buf_idx = new_buf_idx;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_zero_state_passthrough_with_identity_weights() {
        // With identity weights (tap 0 = 1.0, rest = 0.0) and zero ring buffer,
        // the output should equal the input exactly (conv is a no-op).
        let mut conv = ShortConv1D::new(4, 4);
        let x = [0.1, 0.2, 0.3, 0.4];
        let mut out = [0.0f32; 4];
        conv.forward(&x, &mut out);
        // tap 0 = 1.0 weights x[c]; past samples are zero → out = x.
        assert_eq!(
            out, x,
            "identity weights + zero state should pass x through"
        );
    }

    #[test]
    fn conv_causality_output_depends_only_on_past() {
        // Set weights so the conv sums the current + previous sample (tap0=tap1=1).
        // Then verify out[t] depends only on x[t] + x[t-1], not x[t+1].
        let mut conv = ShortConv1D::new(1, 4);
        // Override weights: tap 0 (current) = 1.0, tap 1 (one step back) = 1.0, rest 0.
        conv.weight = vec![1.0, 1.0, 0.0, 0.0];

        let mut out = [0.0f32];
        // Step 1: x = 5.0. State is zero. out should be 5 + 0 = 5.
        conv.forward(&[5.0], &mut out);
        assert_eq!(out[0], 5.0, "step 1: 5 + 0 (zero-padded past) = 5");

        // Step 2: x = 7.0. out should be 7 + 5 = 12.
        conv.forward(&[7.0], &mut out);
        assert_eq!(out[0], 12.0, "step 2: 7 + 5 = 12");

        // Step 3: x = 11.0. out should be 11 + 7 = 18.
        conv.forward(&[11.0], &mut out);
        assert_eq!(out[0], 18.0, "step 3: 11 + 7 = 18");
    }

    #[test]
    fn conv_ring_buffer_wrap_kernel4_holds_last_4() {
        // After 5 pushes with kernel_size=4, the oldest (1st) input is evicted.
        // Use weights that sum all 4 taps to inspect the buffer state.
        let mut conv = ShortConv1D::new(1, 4);
        // All taps = 1.0 → out = sum of last 4 inputs.
        conv.weight = vec![1.0, 1.0, 1.0, 1.0];

        let mut out = [0.0f32];
        conv.forward(&[1.0], &mut out); // state: [1], padded → out = 1
        assert_eq!(out[0], 1.0);
        conv.forward(&[2.0], &mut out); // state: [1,2], padded → out = 1+2 = 3
        assert_eq!(out[0], 3.0);
        conv.forward(&[3.0], &mut out); // state: [1,2,3], padded → out = 1+2+3 = 6
        assert_eq!(out[0], 6.0);
        conv.forward(&[4.0], &mut out); // state: [1,2,3,4], full → out = 1+2+3+4 = 10
        assert_eq!(out[0], 10.0);
        // 5th push: 1 is evicted, state becomes [2,3,4,5], out = 2+3+4+5 = 14
        conv.forward(&[5.0], &mut out);
        assert_eq!(
            out[0], 14.0,
            "5th push should evict the 1st (kernel_size=4)"
        );
        // 6th push: state becomes [3,4,5,6], out = 3+4+5+6 = 18
        conv.forward(&[6.0], &mut out);
        assert_eq!(out[0], 18.0);
    }

    #[test]
    fn conv_depthwise_independence() {
        // Each channel has its own filter; channel c's output should not depend
        // on channel c'≠c's input. Set channel 0's filter to sum past 2 samples,
        // channel 1's filter to identity, then verify they don't cross-contaminate.
        let mut conv = ShortConv1D::new(2, 4);
        // Channel 0: tap 0 = 1.0 (current), tap 1 = 1.0 (past). Channel 1: tap 0 = 2.0.
        conv.weight = vec![
            1.0, 1.0, 0.0, 0.0, // channel 0
            2.0, 0.0, 0.0, 0.0, // channel 1
        ];

        let mut out = [0.0f32; 2];
        conv.forward(&[10.0, 1.0], &mut out);
        // Channel 0: 10 + 0 (zero-padded) = 10. Channel 1: 2 * 1 = 2.
        assert_eq!(out[0], 10.0, "channel 0 step 1");
        assert_eq!(out[1], 2.0, "channel 1 step 1");

        conv.forward(&[20.0, 2.0], &mut out);
        // Channel 0: 20 + 10 = 30. Channel 1: 2 * 2 = 4.
        assert_eq!(
            out[0], 30.0,
            "channel 0 step 2 (uses channel 0's past only)"
        );
        assert_eq!(out[1], 4.0, "channel 1 step 2 (uses channel 1's past only)");
    }

    #[test]
    fn conv_reset_clears_history() {
        let mut conv = ShortConv1D::new(1, 4);
        conv.weight = vec![1.0; 4]; // sum all taps

        let mut out = [0.0f32];
        conv.forward(&[1.0], &mut out);
        conv.forward(&[2.0], &mut out);
        conv.forward(&[3.0], &mut out);
        // After reset, history is gone.
        conv.reset();
        conv.forward(&[10.0], &mut out);
        assert_eq!(
            out[0], 10.0,
            "after reset, only the current sample contributes"
        );
    }

    #[test]
    fn conv_kernel1_is_identity() {
        // kernel_size = 1 means no history; output = weight[0] * x.
        let mut conv = ShortConv1D::new(2, 1);
        conv.weight = vec![3.0, 5.0]; // channel 0 scale 3, channel 1 scale 5
        let mut out = [0.0f32; 2];
        conv.forward(&[2.0, 4.0], &mut out);
        assert_eq!(out[0], 6.0, "channel 0: 3 * 2");
        assert_eq!(out[1], 20.0, "channel 1: 5 * 4");
    }

    #[test]
    fn conv_multichannel_independent_ring_buffers() {
        // Two channels, kernel_size=4. Push 5 values per channel. Verify each
        // channel's ring buffer wraps independently (channel 0's eviction doesn't
        // affect channel 1).
        let mut conv = ShortConv1D::new(2, 4);
        // Channel 0: sum last 4. Channel 1: identity (tap 0 only).
        conv.weight = vec![
            1.0, 1.0, 1.0, 1.0, // channel 0
            1.0, 0.0, 0.0, 0.0, // channel 1
        ];

        let mut out = [0.0f32; 2];
        // Push [ch0, ch1] = [(1,10), (2,20), (3,30), (4,40), (5,50)]
        for (a, b) in [
            (1.0, 10.0),
            (2.0, 20.0),
            (3.0, 30.0),
            (4.0, 40.0),
            (5.0, 50.0),
        ] {
            conv.forward(&[a, b], &mut out);
        }
        // Channel 0: last 4 of [1,2,3,4,5] = [2,3,4,5], sum = 14.
        assert_eq!(out[0], 14.0, "channel 0 sum of last 4");
        // Channel 1: identity, out = 50.
        assert_eq!(out[1], 50.0, "channel 1 current sample only");
    }
}

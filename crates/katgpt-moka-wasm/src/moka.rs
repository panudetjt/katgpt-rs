//! Moka v1 forward pass, ported from `katgpt_pruners::go::moka_net` (Plan 563)
//! for a dependency-free WASM build (Plan 565). Same architecture, same
//! vendored weights, same kernel — only the board type changed (this crate's
//! minimal `board::Board` instead of the full `GoState` engine, so this
//! crate carries no `katgpt-core` dependency and therefore no ahash/getrandom
//! wasm32 backend friction the full engine's dependency tree runs into).
//!
//! Kept in sync by construction: this is the same forward-pass code, not a
//! reimplementation — see Plan 563/Issue 564 for the parity/equivalence
//! testing this logic already passed in its native home.

use std::collections::HashMap;

use serde::Deserialize;

use crate::board::{AREA as BOARD_AREA, Board, Cell, SIZE as BOARD_SIZE, flood_group};

pub(crate) const INPUT_PLANES: usize = 12;
pub(crate) const TRUNK_CHANNELS: usize = 32;
pub(crate) const BOTTLENECK_CHANNELS: usize = 16;
pub(crate) const NUM_BLOCKS: usize = 12;
pub(crate) const GLOBAL_BLOCK_INTERVAL: usize = 4;
pub(crate) const POLICY_CHANNELS: usize = 4;
pub const POLICY_MOVES: usize = 82;
/// Flat feature-tensor length (`9*9*12`) — the input size `wasmi_infer`'s
/// raw FFI boundary expects.
pub const INPUT_ELEMENT_COUNT: usize = BOARD_AREA * INPUT_PLANES;
pub(crate) const VALUE_CHANNELS: usize = 2;
pub(crate) const SCORE_HIDDEN_CHANNELS: usize = 32;
/// Moka's own training-time komi convention — not this crate's board
/// convention, whichever that ends up being. The feature plane must match
/// what the network was trained on.
const MOKA_KOMI: f32 = 7.0;
const KOMI_NORMALIZATION: f32 = 15.0;

pub(crate) static MANIFEST_JSON: &str =
    include_str!("../../katgpt-pruners/assets/moka/go-model.json");
pub(crate) static WEIGHTS_BIN: &[u8] =
    include_bytes!("../../katgpt-pruners/assets/moka/go-model.bin");

#[derive(Deserialize)]
pub(crate) struct Manifest {
    pub(crate) tensors: HashMap<String, TensorMeta>,
}

#[derive(Deserialize)]
pub(crate) struct TensorMeta {
    #[serde(rename = "dataOffset")]
    pub(crate) data_offset: usize,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    #[serde(rename = "scaleOffset", default)]
    pub(crate) scale_offset: Option<usize>,
}

pub(crate) fn read_f32(bytes: &[u8], offset: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let o = offset + i * 4;
            f32::from_le_bytes(bytes[o..o + 4].try_into().expect("4-byte slice"))
        })
        .collect()
}

fn load_dequantized(tensors: &HashMap<String, TensorMeta>, bytes: &[u8], name: &str) -> Vec<f32> {
    let meta = tensors
        .get(name)
        .unwrap_or_else(|| panic!("moka manifest missing tensor {name}"));
    assert_eq!(meta.dtype, "int8", "expected int8 weight tensor {name}");
    let out_channels = meta.shape[0];
    let count: usize = meta.shape.iter().product();
    let per_channel = count / out_channels;
    let scale_offset = meta
        .scale_offset
        .unwrap_or_else(|| panic!("{name} missing scaleOffset"));
    let scales = read_f32(bytes, scale_offset, out_channels);
    let mut out = Vec::with_capacity(count);
    for (oc, &scale) in scales.iter().enumerate() {
        let base = meta.data_offset + oc * per_channel;
        for k in 0..per_channel {
            out.push((bytes[base + k] as i8) as f32 * scale);
        }
    }
    out.shrink_to_fit();
    out
}

pub(crate) fn load_bias(
    tensors: &HashMap<String, TensorMeta>,
    bytes: &[u8],
    name: &str,
) -> Vec<f32> {
    let meta = tensors
        .get(name)
        .unwrap_or_else(|| panic!("moka manifest missing tensor {name}"));
    assert_eq!(meta.dtype, "float32", "expected float32 bias tensor {name}");
    let count: usize = meta.shape.iter().product();
    read_f32(bytes, meta.data_offset, count)
}

pub(crate) struct Wb {
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

pub(crate) struct GlobalBranch {
    pub(crate) hidden: Wb,
    pub(crate) output: Wb,
}

pub(crate) struct ResidualBlock {
    pub(crate) reduce: Wb,
    pub(crate) first: Wb,
    pub(crate) global: Option<GlobalBranch>,
    pub(crate) second: Wb,
    pub(crate) expand: Wb,
}

pub struct MokaWeights {
    stem: Wb,
    blocks: Vec<ResidualBlock>,
    policy_conv: Wb,
    policy_linear: Wb,
    value_conv: Wb,
    value_hidden: Wb,
    value_output: Wb,
}

impl MokaWeights {
    pub fn load() -> Self {
        let manifest: Manifest =
            serde_json::from_str(MANIFEST_JSON).expect("vendored moka manifest is valid JSON");
        let tensors = &manifest.tensors;
        let get = |prefix: &str| -> Wb {
            Wb {
                w: load_dequantized(tensors, WEIGHTS_BIN, &format!("{prefix}.weight")),
                b: load_bias(tensors, WEIGHTS_BIN, &format!("{prefix}.bias")),
            }
        };

        let mut blocks = Vec::with_capacity(NUM_BLOCKS);
        for i in 0..NUM_BLOCKS {
            let prefix = format!("residual.{i}");
            let has_global = (i + 1) % GLOBAL_BLOCK_INTERVAL == 0;
            blocks.push(ResidualBlock {
                reduce: get(&format!("{prefix}.reduce")),
                first: get(&format!("{prefix}.first")),
                global: has_global.then(|| GlobalBranch {
                    hidden: get(&format!("{prefix}.global.hidden")),
                    output: get(&format!("{prefix}.global.output")),
                }),
                second: get(&format!("{prefix}.second")),
                expand: get(&format!("{prefix}.expand")),
            });
        }

        Self {
            stem: get("stem"),
            blocks,
            policy_conv: get("policy.convolution"),
            policy_linear: get("policy.linear"),
            value_conv: get("value.convolution"),
            value_hidden: get("value.hidden"),
            value_output: get("value.output"),
        }
    }
}

// ── NN primitives now live in katgpt-nn (Issue 567 — CNN→Transformer code-path unification Path A).
// These were local functions (conv2d_into, linear_into, global_mean_max_into, relu_inplace, etc.)
// extracted into the shared `katgpt-nn` crate so both the CNN and transformer engines import
// from the same source. Moka-specific weight types + forward graph stay here.
use katgpt_nn::{
    conv2d_batched_into, conv2d_into, global_mean_max_batched_into, linear_batched_into,
    linear_into,
};
// Re-export for sibling modules (moka_int8.rs, research.rs access these via crate::moka::*).
pub(crate) use katgpt_nn::{global_mean_max_into, relu_inplace};

pub struct MokaScratch {
    trunk: Vec<f32>,
    expand: Vec<f32>,
    hidden_a: Vec<f32>,
    hidden_b: Vec<f32>,
    head4: Vec<f32>,
    head2: Vec<f32>,
    patch: Vec<f32>,
    pooled: Vec<f32>,
    gh: Vec<f32>,
    gbias: Vec<f32>,
    value_h: Vec<f32>,
    policy: Vec<f32>,
}

impl MokaScratch {
    pub fn new() -> Self {
        Self {
            trunk: vec![0.0; BOARD_AREA * TRUNK_CHANNELS],
            expand: vec![0.0; BOARD_AREA * TRUNK_CHANNELS],
            hidden_a: vec![0.0; BOARD_AREA * BOTTLENECK_CHANNELS],
            hidden_b: vec![0.0; BOARD_AREA * BOTTLENECK_CHANNELS],
            head4: vec![0.0; BOARD_AREA * POLICY_CHANNELS],
            head2: vec![0.0; BOARD_AREA * VALUE_CHANNELS],
            patch: vec![0.0; 3 * 3 * TRUNK_CHANNELS],
            pooled: vec![0.0; BOTTLENECK_CHANNELS * 2],
            gh: vec![0.0; 8],
            gbias: vec![0.0; BOTTLENECK_CHANNELS],
            value_h: vec![0.0; SCORE_HIDDEN_CHANNELS],
            policy: vec![0.0; POLICY_MOVES],
        }
    }
}

impl Default for MokaScratch {
    fn default() -> Self {
        Self::new()
    }
}

// ── Research accessors (Issue 565 / Research 463) ───────────────────────────
// Behind the `research` feature: expose each weight+bias slice so the PoC
// bench (in riir-poc) can build per-layer error matrices without
// reimplementing the forward pass. The fields stay private in the default
// build (the WASM browser path). These accessors are the minimal surface.
#[cfg(feature = "research")]
impl MokaWeights {
    #[inline]
    pub fn stem_w(&self) -> (&[f32], &[f32]) {
        (&self.stem.w, &self.stem.b)
    }
    #[inline]
    #[allow(private_interfaces)] // ResidualBlock stays pub(crate); riir-poc uses only the pub inherent methods below, never names the type
    pub fn blocks_ref(&self) -> &[ResidualBlock] {
        &self.blocks
    }
    #[inline]
    pub fn policy_conv_w(&self) -> (&[f32], &[f32]) {
        (&self.policy_conv.w, &self.policy_conv.b)
    }
    #[inline]
    pub fn policy_linear_w(&self) -> (&[f32], &[f32]) {
        (&self.policy_linear.w, &self.policy_linear.b)
    }
    #[inline]
    pub fn value_conv_w(&self) -> (&[f32], &[f32]) {
        (&self.value_conv.w, &self.value_conv.b)
    }
    #[inline]
    pub fn value_hidden_w(&self) -> (&[f32], &[f32]) {
        (&self.value_hidden.w, &self.value_hidden.b)
    }
    #[inline]
    pub fn value_output_w(&self) -> (&[f32], &[f32]) {
        (&self.value_output.w, &self.value_output.b)
    }
}

#[cfg(feature = "research")]
impl ResidualBlock {
    #[inline]
    pub fn reduce_w(&self) -> (&[f32], &[f32]) {
        (&self.reduce.w, &self.reduce.b)
    }
    #[inline]
    pub fn first_w(&self) -> (&[f32], &[f32]) {
        (&self.first.w, &self.first.b)
    }
    #[inline]
    pub fn second_w(&self) -> (&[f32], &[f32]) {
        (&self.second.w, &self.second.b)
    }
    #[inline]
    pub fn expand_w(&self) -> (&[f32], &[f32]) {
        (&self.expand.w, &self.expand.b)
    }
    #[inline]
    pub fn global_ref(&self) -> Option<&GlobalBranch> {
        self.global.as_ref()
    }
}

#[cfg(feature = "research")]
impl GlobalBranch {
    #[inline]
    pub fn hidden_w(&self) -> (&[f32], &[f32]) {
        (&self.hidden.w, &self.hidden.b)
    }
    #[inline]
    pub fn output_w(&self) -> (&[f32], &[f32]) {
        (&self.output.w, &self.output.b)
    }
}

#[cfg(feature = "research")]
impl MokaScratch {
    /// Lend out the internal scratch buffers as mutable slices. The corrected
    /// forward pass (`research::forward_corrected_with_scratch`) destructures
    /// MokaScratch by field, so it needs `pub` field access OR these
    /// accessors. We lend all buffers at once via a tuple to keep the borrow
    /// checker happy (a single &mut borrow split into sub-slices).
    #[inline]
    #[allow(clippy::type_complexity)] // 12-buffer scratch lease; see doc comment above
    pub fn lend_all(
        &mut self,
    ) -> (
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
        &mut [f32],
    ) {
        (
            &mut self.trunk,
            &mut self.expand,
            &mut self.hidden_a,
            &mut self.hidden_b,
            &mut self.head4,
            &mut self.head2,
            &mut self.patch,
            &mut self.pooled,
            &mut self.gh,
            &mut self.gbias,
            &mut self.value_h,
            &mut self.policy,
        )
    }
}

pub fn forward_with_scratch(
    weights: &MokaWeights,
    features: &[f32],
    scratch: &mut MokaScratch,
) -> ([f32; POLICY_MOVES], f32) {
    let MokaScratch {
        trunk,
        expand,
        hidden_a,
        hidden_b,
        head4,
        head2,
        patch,
        pooled,
        gh,
        gbias,
        value_h,
        policy,
    } = scratch;

    conv2d_into(
        features,
        BOARD_SIZE,
        BOARD_SIZE,
        INPUT_PLANES,
        TRUNK_CHANNELS,
        3,
        &weights.stem.w,
        &weights.stem.b,
        patch,
        trunk,
    );
    relu_inplace(&mut trunk[..BOARD_AREA * TRUNK_CHANNELS]);

    for block in &weights.blocks {
        conv2d_into(
            trunk,
            BOARD_SIZE,
            BOARD_SIZE,
            TRUNK_CHANNELS,
            BOTTLENECK_CHANNELS,
            1,
            &block.reduce.w,
            &block.reduce.b,
            patch,
            hidden_a,
        );
        relu_inplace(hidden_a);
        conv2d_into(
            hidden_a,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.first.w,
            &block.first.b,
            patch,
            hidden_b,
        );
        relu_inplace(hidden_b);

        if let Some(g) = &block.global {
            global_mean_max_into(
                hidden_b,
                BOARD_SIZE,
                BOARD_SIZE,
                BOTTLENECK_CHANNELS,
                pooled,
            );
            linear_into(
                pooled,
                BOTTLENECK_CHANNELS * 2,
                g.hidden.b.len(),
                &g.hidden.w,
                &g.hidden.b,
                gh,
            );
            relu_inplace(&mut gh[..g.hidden.b.len()]);
            linear_into(
                gh,
                g.hidden.b.len(),
                BOTTLENECK_CHANNELS,
                &g.output.w,
                &g.output.b,
                gbias,
            );
            for pos in 0..BOARD_AREA {
                let row = &mut hidden_b[pos * BOTTLENECK_CHANNELS..(pos + 1) * BOTTLENECK_CHANNELS];
                for c in 0..BOTTLENECK_CHANNELS {
                    row[c] += gbias[c];
                }
            }
        }

        conv2d_into(
            hidden_b,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.second.w,
            &block.second.b,
            patch,
            hidden_a,
        );
        relu_inplace(hidden_a);
        conv2d_into(
            hidden_a,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            TRUNK_CHANNELS,
            1,
            &block.expand.w,
            &block.expand.b,
            patch,
            expand,
        );

        for i in 0..BOARD_AREA * TRUNK_CHANNELS {
            let v = trunk[i] + expand[i];
            trunk[i] = if v < 0.0 { 0.0 } else { v };
        }
    }

    conv2d_into(
        trunk,
        BOARD_SIZE,
        BOARD_SIZE,
        TRUNK_CHANNELS,
        POLICY_CHANNELS,
        1,
        &weights.policy_conv.w,
        &weights.policy_conv.b,
        patch,
        head4,
    );
    relu_inplace(head4);
    linear_into(
        head4,
        POLICY_CHANNELS * BOARD_AREA,
        POLICY_MOVES,
        &weights.policy_linear.w,
        &weights.policy_linear.b,
        policy,
    );

    conv2d_into(
        trunk,
        BOARD_SIZE,
        BOARD_SIZE,
        TRUNK_CHANNELS,
        VALUE_CHANNELS,
        1,
        &weights.value_conv.w,
        &weights.value_conv.b,
        patch,
        head2,
    );
    relu_inplace(head2);
    let value_hidden_dim = weights.value_hidden.b.len();
    linear_into(
        head2,
        VALUE_CHANNELS * BOARD_AREA,
        value_hidden_dim,
        &weights.value_hidden.w,
        &weights.value_hidden.b,
        value_h,
    );
    relu_inplace(&mut value_h[..value_hidden_dim]);
    let mut value_out = [0f32; 1];
    linear_into(
        value_h,
        value_hidden_dim,
        1,
        &weights.value_output.w,
        &weights.value_output.b,
        &mut value_out,
    );

    let mut logits = [0f32; POLICY_MOVES];
    logits.copy_from_slice(&policy[..POLICY_MOVES]);
    (logits, value_out[0].tanh())
}

// ── Batched forward pass (Issue 205) ────────────────────────────────
//
// K-wide forward pass for batched MCTS. Same arithmetic as K sequential
// `forward_with_scratch` calls, but reuses each weight slice across all K
// samples in the inner loop — the cache-locality win on CPU (the Moka
// trunk block is ~100 KB, fits L2 not L1; sequential passes reload from
// L2 K times, batched loads once per out_channel).
//
// Layout: all buffers are sample-major (`buffer[s * sample_len..]` is
// sample s's data). This keeps pointer arithmetic simple + lets the K
// inner-loop iterations walk contiguous memory for the output writes.

/// K-wide scratch space for batched forward passes. Constructed once at
/// player init with the max K; the same buffers are reused across every
/// `select_move` call. Sample-major layout throughout.
pub struct MokaBatchScratch {
    pub batch: usize,
    trunk: Vec<f32>,
    expand: Vec<f32>,
    hidden_a: Vec<f32>,
    hidden_b: Vec<f32>,
    head4: Vec<f32>,
    head2: Vec<f32>,
    patches_3x3_trunk: Vec<f32>, // batch × (3·3·TRUNK_CHANNELS)  — stem
    patches_3x3_bottleneck: Vec<f32>, // batch × (3·3·BOTTLENECK_CHANNELS) — first/second
    pooled: Vec<f32>,            // batch × (BOTTLENECK_CHANNELS·2)
    gh: Vec<f32>,                // batch × 8
    gbias: Vec<f32>,             // batch × BOTTLENECK_CHANNELS
    value_h: Vec<f32>,           // batch × SCORE_HIDDEN_CHANNELS
    policy: Vec<f32>,            // batch × POLICY_MOVES
    value_out: Vec<f32>,         // batch × 1
}

impl MokaBatchScratch {
    pub fn new(batch: usize) -> Self {
        Self {
            batch,
            trunk: vec![0.0; batch * BOARD_AREA * TRUNK_CHANNELS],
            expand: vec![0.0; batch * BOARD_AREA * TRUNK_CHANNELS],
            hidden_a: vec![0.0; batch * BOARD_AREA * BOTTLENECK_CHANNELS],
            hidden_b: vec![0.0; batch * BOARD_AREA * BOTTLENECK_CHANNELS],
            head4: vec![0.0; batch * BOARD_AREA * POLICY_CHANNELS],
            head2: vec![0.0; batch * BOARD_AREA * VALUE_CHANNELS],
            patches_3x3_trunk: vec![0.0; batch * 3 * 3 * TRUNK_CHANNELS],
            patches_3x3_bottleneck: vec![0.0; batch * 3 * 3 * BOTTLENECK_CHANNELS],
            pooled: vec![0.0; batch * BOTTLENECK_CHANNELS * 2],
            gh: vec![0.0; batch * 8],
            gbias: vec![0.0; batch * BOTTLENECK_CHANNELS],
            value_h: vec![0.0; batch * SCORE_HIDDEN_CHANNELS],
            policy: vec![0.0; batch * POLICY_MOVES],
            value_out: vec![0.0; batch],
        }
    }
}

/// Batched forward pass. `features_batch` is K HWC feature tensors laid out
/// sample-major (`features_batch[s * INPUT_ELEMENT_COUNT..]` is sample s).
/// Writes per-sample logits into `policy_batch[s * POLICY_MOVES..]` and
/// per-sample tanh-value into `value_batch[s]`.
///
/// `policy_batch` must be length `≥ batch * POLICY_MOVES`; `value_batch`
/// must be length `≥ batch`. The caller owns these so the search loop can
/// read them without copying out of the scratch struct.
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_with_scratch(
    weights: &MokaWeights,
    features_batch: &[f32],
    batch: usize,
    scratch: &mut MokaBatchScratch,
    policy_batch: &mut [f32],
    value_batch: &mut [f32],
) {
    debug_assert!(
        scratch.batch >= batch,
        "scratch must be sized for at least this batch; scratch={}, batch={}",
        scratch.batch,
        batch
    );
    let MokaBatchScratch {
        trunk,
        expand,
        hidden_a,
        hidden_b,
        head4,
        head2,
        patches_3x3_trunk,
        patches_3x3_bottleneck,
        pooled,
        gh,
        gbias,
        value_h,
        policy,
        value_out,
        batch: _,
    } = scratch;

    let trunk_len = BOARD_AREA * TRUNK_CHANNELS;
    let bn_len = BOARD_AREA * BOTTLENECK_CHANNELS;

    // Stem: 3×3 conv, 12 → 32 channels. The patch gather happens once per
    // (sample, position); the weight slice is reused across all K samples.
    conv2d_batched_into(
        features_batch,
        batch,
        BOARD_SIZE,
        BOARD_SIZE,
        INPUT_PLANES,
        TRUNK_CHANNELS,
        3,
        &weights.stem.w,
        &weights.stem.b,
        patches_3x3_trunk,
        trunk,
    );
    for s in 0..batch {
        relu_inplace(&mut trunk[s * trunk_len..][..trunk_len]);
    }

    for block in &weights.blocks {
        // reduce: 1×1 conv, 32 → 16.
        conv2d_batched_into(
            trunk,
            batch,
            BOARD_SIZE,
            BOARD_SIZE,
            TRUNK_CHANNELS,
            BOTTLENECK_CHANNELS,
            1,
            &block.reduce.w,
            &block.reduce.b,
            patches_3x3_trunk,
            hidden_a,
        );
        for s in 0..batch {
            relu_inplace(&mut hidden_a[s * bn_len..][..bn_len]);
        }
        // first: 3×3 conv, 16 → 16.
        conv2d_batched_into(
            hidden_a,
            batch,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.first.w,
            &block.first.b,
            patches_3x3_bottleneck,
            hidden_b,
        );
        for s in 0..batch {
            relu_inplace(&mut hidden_b[s * bn_len..][..bn_len]);
        }

        if let Some(g) = &block.global {
            global_mean_max_batched_into(
                hidden_b,
                batch,
                BOARD_SIZE,
                BOARD_SIZE,
                BOTTLENECK_CHANNELS,
                pooled,
            );
            let gh_len = g.hidden.b.len();
            linear_batched_into(
                pooled,
                batch,
                BOTTLENECK_CHANNELS * 2,
                gh_len,
                &g.hidden.w,
                &g.hidden.b,
                gh,
            );
            for s in 0..batch {
                relu_inplace(&mut gh[s * gh_len..][..gh_len]);
            }
            linear_batched_into(
                gh,
                batch,
                gh_len,
                BOTTLENECK_CHANNELS,
                &g.output.w,
                &g.output.b,
                gbias,
            );
            for s in 0..batch {
                let row = &mut hidden_b[s * bn_len..];
                let g_s = &gbias[s * BOTTLENECK_CHANNELS..][..BOTTLENECK_CHANNELS];
                for pos in 0..BOARD_AREA {
                    let slot = &mut row[pos * BOTTLENECK_CHANNELS..(pos + 1) * BOTTLENECK_CHANNELS];
                    for c in 0..BOTTLENECK_CHANNELS {
                        slot[c] += g_s[c];
                    }
                }
            }
        }

        // second: 3×3 conv, 16 → 16.
        conv2d_batched_into(
            hidden_b,
            batch,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            BOTTLENECK_CHANNELS,
            3,
            &block.second.w,
            &block.second.b,
            patches_3x3_bottleneck,
            hidden_a,
        );
        for s in 0..batch {
            relu_inplace(&mut hidden_a[s * bn_len..][..bn_len]);
        }
        // expand: 1×1 conv, 16 → 32, then residual add into trunk.
        conv2d_batched_into(
            hidden_a,
            batch,
            BOARD_SIZE,
            BOARD_SIZE,
            BOTTLENECK_CHANNELS,
            TRUNK_CHANNELS,
            1,
            &block.expand.w,
            &block.expand.b,
            patches_3x3_trunk,
            expand,
        );
        for s in 0..batch {
            let t = &mut trunk[s * trunk_len..][..trunk_len];
            let e = &expand[s * trunk_len..][..trunk_len];
            for i in 0..trunk_len {
                let v = t[i] + e[i];
                t[i] = if v < 0.0 { 0.0 } else { v };
            }
        }
    }

    // Policy head.
    let head4_len = BOARD_AREA * POLICY_CHANNELS;
    conv2d_batched_into(
        trunk,
        batch,
        BOARD_SIZE,
        BOARD_SIZE,
        TRUNK_CHANNELS,
        POLICY_CHANNELS,
        1,
        &weights.policy_conv.w,
        &weights.policy_conv.b,
        patches_3x3_trunk,
        head4,
    );
    for s in 0..batch {
        relu_inplace(&mut head4[s * head4_len..][..head4_len]);
    }
    linear_batched_into(
        head4,
        batch,
        POLICY_CHANNELS * BOARD_AREA,
        POLICY_MOVES,
        &weights.policy_linear.w,
        &weights.policy_linear.b,
        policy,
    );

    // Value head.
    let head2_len = BOARD_AREA * VALUE_CHANNELS;
    conv2d_batched_into(
        trunk,
        batch,
        BOARD_SIZE,
        BOARD_SIZE,
        TRUNK_CHANNELS,
        VALUE_CHANNELS,
        1,
        &weights.value_conv.w,
        &weights.value_conv.b,
        patches_3x3_trunk,
        head2,
    );
    for s in 0..batch {
        relu_inplace(&mut head2[s * head2_len..][..head2_len]);
    }
    let value_hidden_dim = weights.value_hidden.b.len();
    linear_batched_into(
        head2,
        batch,
        VALUE_CHANNELS * BOARD_AREA,
        value_hidden_dim,
        &weights.value_hidden.w,
        &weights.value_hidden.b,
        value_h,
    );
    for s in 0..batch {
        relu_inplace(&mut value_h[s * value_hidden_dim..][..value_hidden_dim]);
    }
    linear_batched_into(
        value_h,
        batch,
        value_hidden_dim,
        1,
        &weights.value_output.w,
        &weights.value_output.b,
        value_out,
    );

    // Copy out into the caller-owned buffers + tanh the value.
    for s in 0..batch {
        policy_batch[s * POLICY_MOVES..(s + 1) * POLICY_MOVES]
            .copy_from_slice(&policy[s * POLICY_MOVES..(s + 1) * POLICY_MOVES]);
        value_batch[s] = value_out[s].tanh();
    }
}

/// Encode `board` + last-two-plies `history` (`None` = pass) into Moka's
/// 12-plane `9*9*12` HWC feature tensor, written into `out` (length ≥
/// [`INPUT_ELEMENT_COUNT`]). Logic identical to
/// `katgpt_pruners::go::moka_net::encode_features`, adapted to this crate's
/// standalone `Board`/`Cell`; `_into` here (unlike the native version, which
/// still returns an owned `Vec`) because this crate's `WasmGame` holds a
/// persistent buffer specifically so its address stays stable across calls —
/// letting JS wrap ONE `Float32Array` view over it instead of marshalling a
/// fresh array out of wasm on every encode.
pub fn encode_features_into(board: &Board, history: &[Option<(usize, usize)>], out: &mut [f32]) {
    let size = BOARD_SIZE;
    let idx = |row: usize, col: usize, plane: usize| (row * size + col) * INPUT_PLANES + plane;
    let feats = &mut out[..size * size * INPUT_PLANES];
    feats.fill(0.0);

    let mut visited = vec![false; size * size];
    for pos in 0..size * size {
        let color = board.cells[pos];
        if color == Cell::Empty {
            continue;
        }
        let is_current = color == board.to_play;
        let (row, col) = (pos / size, pos % size);
        feats[idx(row, col, if is_current { 0 } else { 1 })] = 1.0;

        if visited[pos] {
            continue;
        }
        let (stones, liberties) = flood_group(&board.cells, pos);
        for &s in &stones {
            visited[s] = true;
        }
        let liberty_count = liberties.len();
        if liberty_count != 1 && liberty_count != 2 {
            continue;
        }
        let plane = match (is_current, liberty_count) {
            (true, 1) => 2,
            (false, 1) => 3,
            (true, 2) => 4,
            (false, 2) => 5,
            _ => unreachable!(),
        };
        for &s in &stones {
            let (r, c) = (s / size, s % size);
            feats[idx(r, c, plane)] = 1.0;
        }
    }

    if let Some(ko) = board.ko_point {
        let (r, c) = (ko / size, ko % size);
        feats[idx(r, c, 6)] = 1.0;
    }

    for offset in 1..=2usize {
        if history.len() < offset {
            continue;
        }
        if let Some((r, c)) = history[history.len() - offset] {
            let plane = if offset == 1 { 7 } else { 8 };
            feats[idx(r, c, plane)] = 1.0;
        } else {
            let plane = 8 + offset;
            for row in 0..size {
                for col in 0..size {
                    feats[idx(row, col, plane)] = 1.0;
                }
            }
        }
    }

    let next_color: f32 = if board.to_play == Cell::Black {
        1.0
    } else {
        -1.0
    };
    let komi_value = (-MOKA_KOMI * next_color) / KOMI_NORMALIZATION;
    for row in 0..size {
        for col in 0..size {
            feats[idx(row, col, 11)] = komi_value;
        }
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::board::Board;

    /// G1 gate: for K random boards, the batched forward pass must produce
    /// output within f32 epsilon of K sequential `forward_with_scratch`
    /// calls. This is the load-bearing correctness invariant — if it fails,
    /// the batched MCTS search is invalid (different move choices vs
    /// sequential PUCT).
    #[test]
    fn g1_batched_forward_matches_sequential() {
        let weights = MokaWeights::load();
        let k = 8;
        let mut seq_scratch = MokaScratch::new();
        let mut batch_scratch = MokaBatchScratch::new(k);

        // Build K distinct mid-game boards by playing different opening
        // sequences. encode_features needs last-2-plies history, which we
        // reconstruct per board.
        let opening_seqs: [[usize; 6]; 8] = [
            [40, 41, 31, 50, 32, 49],
            [0, 1, 9, 10, 18, 19],
            [80, 79, 71, 70, 62, 61],
            [40, 50, 30, 60, 31, 51],
            [4, 5, 13, 14, 22, 23],
            [76, 77, 67, 68, 58, 59],
            [40, 32, 48, 31, 49, 23],
            [40, 41, 31, 50, 32, 49], // duplicate of [0] to test same-board
        ];

        let mut features = vec![0.0; k * INPUT_ELEMENT_COUNT];
        let mut seq_policy = vec![[0f32; POLICY_MOVES]; k];
        let mut seq_value = vec![0f32; k];

        for s in 0..k {
            let mut board = Board::new();
            let mut hist: Vec<Option<(usize, usize)>> = Vec::new();
            for &mv in &opening_seqs[s] {
                if board.is_legal(mv) {
                    board.play(mv);
                    hist.push(Some((mv / BOARD_SIZE, mv % BOARD_SIZE)));
                }
            }
            // Take last-2 for the feature encoder.
            let last2: Vec<Option<(usize, usize)>> = hist
                .iter()
                .rev()
                .take(2)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            encode_features_into(&board, &last2, &mut features[s * INPUT_ELEMENT_COUNT..]);
            let (p, v) = forward_with_scratch(
                &weights,
                &features[s * INPUT_ELEMENT_COUNT..],
                &mut seq_scratch,
            );
            seq_policy[s] = p;
            seq_value[s] = v;
        }

        let mut batch_policy = vec![0f32; k * POLICY_MOVES];
        let mut batch_value = vec![0f32; k];
        forward_batch_with_scratch(
            &weights,
            &features,
            k,
            &mut batch_scratch,
            &mut batch_policy,
            &mut batch_value,
        );

        // Compare. Allow generous epsilon — f32 reassociation in the batched
        // dot loop can accumulate slightly differently than the sequential
        // version (different summation order is legal per IEEE-754).
        const EPS: f32 = 1e-3;
        let mut max_policy_diff = 0f32;
        let mut max_value_diff = 0f32;
        for s in 0..k {
            for i in 0..POLICY_MOVES {
                let diff = (seq_policy[s][i] - batch_policy[s * POLICY_MOVES + i]).abs();
                if diff > max_policy_diff {
                    max_policy_diff = diff;
                }
            }
            let diff = (seq_value[s] - batch_value[s]).abs();
            if diff > max_value_diff {
                max_value_diff = diff;
            }
        }
        assert!(
            max_policy_diff < EPS,
            "batched vs sequential policy diff {max_policy_diff:e} exceeds {EPS:e}"
        );
        assert!(
            max_value_diff < EPS,
            "batched vs sequential value diff {max_value_diff:e} exceeds {EPS:e}"
        );
        eprintln!(
            "g1 PASS: max policy diff {max_policy_diff:e}, max value diff {max_value_diff:e}"
        );
    }
}

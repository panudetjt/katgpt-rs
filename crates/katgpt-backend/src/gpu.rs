//! GPU inference backend via Metal compute (Plan 176).
//!
//! Implements [`InferenceBackend`] using Apple Metal compute shaders for the
//! transformer forward pass. All weights are uploaded to GPU memory during
//! `compile()`, and the forward pass executes entirely on GPU with only the
//! final logits copied back to CPU.
//!
//! # Architecture
//!
//! The GPU forward pass mirrors the CPU algorithm:
//! 1. Embedding lookup: `x = wte[token] + wpe[pos]`
//! 2. Per-layer: RMSNorm → QKV matmul → KV cache store → attention →
//!    output projection → residual → RMSNorm → MLP → residual
//! 3. LM head matmul: `logits = lm_head @ x`
//!
//! # Metal Shaders
//!
//! Compute kernels are embedded as MSL (Metal Shading Language) source strings
//! and compiled at pipeline creation time. Each kernel is a simple parallel
//! operation: RMSNorm, matrix-vector multiply, element-wise add, ReLU, etc.

use std::ffi::c_void;
use std::mem;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::{CompileError, InferenceBackend};
use katgpt_forward::ForwardContext;
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};
use katgpt_types::{Config, kv_dim};

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// All Metal compute shaders as a single MSL source string.
const SHADER_SOURCE: &str = r#"
using namespace metal;

// RMSNorm: out[i] = x[i] * gamma[i] / sqrt(mean(x^2) + eps)
// Single-threaded kernel — simple and correct for n_embd <= 4096.
kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        float ss = 0.0;
        for (uint i = 0; i < n; i++) {
            ss += x[i] * x[i];
        }
        ss = 1.0 / sqrt(ss / float(n) + eps);
        for (uint i = 0; i < n; i++) {
            float val = x[i] * ss;
            out[i] = gamma ? val * gamma[i] : val;
        }
    }
}

// Matmul: out = W @ x  where W is [out_dim, in_dim], x is [in_dim]
// Each thread computes one output element (one row of W dotted with x).
kernel void matmul(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0;
    for (uint i = 0; i < in_dim; i++) {
        sum += W[gid * in_dim + i] * x[i];
    }
    out[gid] = sum;
}

// Element-wise add: out[i] = a[i] + b[i]
kernel void elem_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

// ReLU: out[i] = max(0, x[i])
kernel void relu(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = x[gid] > 0.0 ? x[gid] : 0.0;
}

// Attention score: for one head, compute q_h @ K_cached[0:seq_len] * scale
// Each thread computes the score for one cached position.
kernel void attention_score(
    device const float* q [[buffer(0)]],          // [n_embd] full query vector
    device const float* key_cache [[buffer(1)]],  // [block_size * kv_dim]
    constant uint& q_offset [[buffer(2)]],        // h * head_dim
    constant uint& kv_offset [[buffer(3)]],       // kv_group * head_dim
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    device float* scores [[buffer(7)]],           // [block_size] output scores
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0;
    for (uint d = 0; d < head_dim; d++) {
        sum += q[q_offset + d] * key_cache[gid * kv_dim + kv_offset + d];
    }
    scores[gid] = sum * scale;
}

// Softmax: online softmax over scores[0:seq_len]
// Single-threaded — correct for block_size <= 8192.
kernel void softmax(
    device float* scores [[buffer(0)]],
    constant uint& seq_len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        float max_val = scores[0];
        for (uint i = 1; i < seq_len; i++) {
            if (scores[i] > max_val) max_val = scores[i];
        }
        float sum = 0.0;
        for (uint i = 0; i < seq_len; i++) {
            scores[i] = exp(scores[i] - max_val);
            sum += scores[i];
        }
        for (uint i = 0; i < seq_len; i++) {
            scores[i] /= sum;
        }
    }
}

// Attention value: weighted sum of value cache for one head.
// Single-threaded — writes head_dim outputs for one head.
kernel void attention_value(
    device const float* scores [[buffer(0)]],      // [seq_len] softmaxed scores
    device const float* value_cache [[buffer(1)]], // [block_size * kv_dim]
    device float* attn_out [[buffer(2)]],          // [n_embd] output
    constant uint& out_offset [[buffer(3)]],       // h * head_dim
    constant uint& kv_offset [[buffer(4)]],        // kv_group * head_dim
    constant uint& kv_dim [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        for (uint d = 0; d < head_dim; d++) {
            float sum = 0.0;
            for (uint t = 0; t < seq_len; t++) {
                sum += scores[t] * value_cache[t * kv_dim + kv_offset + d];
            }
            attn_out[out_offset + d] = sum;
        }
    }
}

// KV store: copy k/v vectors into cache at position pos.
kernel void kv_store(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* key_cache [[buffer(2)]],
    device float* value_cache [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    constant uint& kv_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < kv_dim) {
        key_cache[pos * kv_dim + gid] = k[gid];
        value_cache[pos * kv_dim + gid] = v[gid];
    }
}
"#;

// ---------------------------------------------------------------------------
// GPU pipeline & buffer helpers
// ---------------------------------------------------------------------------

/// Compiled Metal compute pipelines for each kernel.
struct GpuPipelines {
    rmsnorm: ComputePipelineState,
    matmul: ComputePipelineState,
    elem_add: ComputePipelineState,
    relu: ComputePipelineState,
    attention_score: ComputePipelineState,
    softmax: ComputePipelineState,
    attention_value: ComputePipelineState,
    kv_store: ComputePipelineState,
}

impl GpuPipelines {
    fn create(device: &metal::DeviceRef) -> Result<Self, CompileError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|e| CompileError::DeviceError(format!("shader compilation failed: {e}")))?;

        let make_pipeline = |name: &str| -> Result<ComputePipelineState, CompileError> {
            let function = library.get_function(name, None).map_err(|e| {
                CompileError::DeviceError(format!("function '{name}' not found: {e}"))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    CompileError::DeviceError(format!("pipeline creation failed for '{name}': {e}"))
                })
        };

        Ok(Self {
            rmsnorm: make_pipeline("rmsnorm")?,
            matmul: make_pipeline("matmul")?,
            elem_add: make_pipeline("elem_add")?,
            relu: make_pipeline("relu")?,
            attention_score: make_pipeline("attention_score")?,
            softmax: make_pipeline("softmax")?,
            attention_value: make_pipeline("attention_value")?,
            kv_store: make_pipeline("kv_store")?,
        })
    }
}

/// All transformer weights uploaded to GPU Metal buffers.
struct GpuWeightBuffers {
    // Embeddings are gathered from CPU-side slices per token/pos.
    // These buffers are kept for potential future fused embedding kernels.
    #[allow(dead_code)]
    wte: Buffer, // [vocab_size * n_embd]
    #[allow(dead_code)]
    wpe: Buffer, // [block_size * n_embd]
    lm_head: Buffer, // [vocab_size * n_embd]
    // Per-layer weights (flat Vec of buffers, indexed by layer)
    attn_wq: Vec<Buffer>,         // [n_layer] each [n_embd * n_embd]
    attn_wk: Vec<Buffer>,         // [n_layer] each [kv_dim * n_embd]
    attn_wv: Vec<Buffer>,         // [n_layer] each [kv_dim * n_embd]
    attn_wo: Vec<Buffer>,         // [n_layer] each [n_embd * n_embd]
    mlp_w1: Vec<Buffer>,          // [n_layer] each [mlp_hidden * n_embd]
    mlp_w2: Vec<Buffer>,          // [n_layer] each [n_embd * mlp_hidden]
    attn_norm_gamma: Vec<Buffer>, // [n_layer] each [n_embd]
    mlp_norm_gamma: Vec<Buffer>,  // [n_layer] each [n_embd]
}

/// Upload a `Vec<f32>` to a new Metal buffer with shared memory.
fn upload_buffer(device: &metal::DeviceRef, data: &[f32]) -> Buffer {
    let byte_len = std::mem::size_of_val(data) as u64;
    let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let contents = buffer.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const c_void, contents, byte_len as usize);
    }
    buffer
}

/// Create a zero-initialized Metal buffer of `len` floats.
fn zero_buffer(device: &metal::DeviceRef, len: usize) -> Buffer {
    let byte_len = (len * mem::size_of::<f32>()) as u64;
    let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let contents = buffer.contents();
    unsafe {
        std::ptr::write_bytes(contents as *mut u8, 0, byte_len as usize);
    }
    buffer
}

/// Upload a small scalar value as a Metal buffer (for constant arguments).
fn scalar_buffer<T: Copy>(device: &metal::DeviceRef, value: &T) -> Buffer {
    let byte_len = mem::size_of::<T>() as u64;
    let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let contents = buffer.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(
            value as *const T as *const c_void,
            contents,
            byte_len as usize,
        );
    }
    buffer
}

/// Overwrite the contents of an existing Metal buffer with a scalar value.
/// Reuses the allocation — avoids a `device.new_buffer()` IPC per forward call.
#[inline]
fn write_scalar<T: Copy>(buf: &Buffer, value: &T) {
    let byte_len = mem::size_of::<T>();
    let contents = buf.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(value as *const T as *const c_void, contents, byte_len);
    }
}

/// Overwrite the contents of an existing Metal buffer with an f32 slice.
/// Reuses the allocation — avoids a `device.new_buffer()` IPC per forward call.
#[inline]
fn write_f32_slice(buf: &Buffer, data: &[f32]) {
    let byte_len = std::mem::size_of_val(data);
    let contents = buf.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const c_void, contents, byte_len);
    }
}

/// Dispatch a 1D compute kernel with `n` total threads.
fn dispatch_1d(encoder: &metal::ComputeCommandEncoderRef, pipeline: &ComputePipelineState, n: u64) {
    encoder.set_compute_pipeline_state(pipeline);
    let tg_size = pipeline.thread_execution_width();
    let threadgroup_count = MTLSize::new(n.div_ceil(tg_size), 1, 1);
    let threads_per_tg = MTLSize::new(tg_size, 1, 1);
    encoder.dispatch_thread_groups(threadgroup_count, threads_per_tg);
}

/// Dispatch a single-thread kernel (for sequential ops like rmsnorm, softmax).
fn dispatch_single(encoder: &metal::ComputeCommandEncoderRef, pipeline: &ComputePipelineState) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
}

// ---------------------------------------------------------------------------
// GpuBackend
// ---------------------------------------------------------------------------

/// Maximum tokens accepted by [`GpuBackend::forward_batch`]. Per-slot upload
/// buffers for this many tokens are pre-allocated at `compile()` time
/// (Issue 660 task 5).
pub const MAX_FORWARD_BATCH: usize = 16;

/// GPU inference backend using Apple Metal compute shaders.
///
/// Starts uncompiled. Call [`compile()`](InferenceBackend::compile) with the
/// current weights + config to upload weights to GPU memory and create compute
/// pipelines. Until compiled, [`forward()`](InferenceBackend::forward) falls
/// back to CPU.
pub struct GpuBackend {
    device: Device,
    command_queue: CommandQueue,
    compiled: bool,
    needs_recompile: bool,
    config_snapshot: Option<Config>,

    // Compiled GPU resources (populated by compile())
    pipelines: Option<GpuPipelines>,
    weight_buffers: Option<GpuWeightBuffers>,

    // Runtime scratch buffers (allocated on first compile, reused across forward passes)
    x_buf: Option<Buffer>,        // [n_embd]
    xr_buf: Option<Buffer>,       // [n_embd]
    xr2_buf: Option<Buffer>,      // [n_embd]
    q_buf: Option<Buffer>,        // [n_embd]
    k_buf: Option<Buffer>,        // [kv_dim]
    v_buf: Option<Buffer>,        // [kv_dim]
    attn_out_buf: Option<Buffer>, // [n_embd]
    hidden_buf: Option<Buffer>,   // [mlp_hidden]
    scores_buf: Option<Buffer>,   // [block_size]
    logits_buf: Option<Buffer>,   // [vocab_size]
    // GPU-side KV cache: Vec of (key_buffer, value_buffer) per layer
    kv_cache: Option<Vec<(Buffer, Buffer)>>,

    // Cached scalar buffers (allocated once in compile(), reused every forward).
    // These represent config-derived constants that never change between calls —
    // previously the forward path allocated ~9 + n_layer + 3*n_head*n_layer
    // Metal buffers per call (kernel IPC allocations). See Plan: GPU scalar cache.
    n_embd_scalar: Option<Buffer>, // u32 n_embd — doubles as n_embd_in_dim/mlp_in_dim/kv_in_dim
    kv_dim_scalar: Option<Buffer>, // u32 kv_dim
    head_dim_scalar: Option<Buffer>, // u32 head_dim
    mlp_hidden_scalar: Option<Buffer>, // u32 mlp_hidden (mlp_w2 in_dim)
    eps_scalar: Option<Buffer>,    // f32 rmsnorm eps
    scale_scalar: Option<Buffer>,  // f32 attention scale (1/sqrt(head_dim))
    // Per-head offset buffers (q_off, kv_off, out_off) — pre-computed once
    // since head layout is fixed at compile time.
    q_off_bufs: Option<Vec<Buffer>>,
    kv_off_bufs: Option<Vec<Buffer>>,
    out_off_bufs: Option<Vec<Buffer>>,
    // Per-call buffers (values change every forward, but the Metal buffer allocation
    // is reused via `contents()` write). Previously each of these triggered a
    // `device.new_buffer()` IPC call per forward pass.
    seq_len_scalar: Option<Buffer>, // u32 seq_len (pos + 1)
    pos_scalar: Option<Buffer>,     // u32 pos
    emb_wte_slice: Option<Buffer>,  // [n_embd] wte[token] slice
    emb_wpe_slice: Option<Buffer>,  // [n_embd] wpe[pos] slice
    // Batched width-N forward (Issue 660 task 5): per-slot buffers so N
    // tokens can be encoded into ONE command buffer. Per-slot, not shared,
    // because pos/seq_len differ per token — CPU writes to a shared buffer
    // all land before GPU execution, so every dispatch would otherwise see
    // the last value. Each slot is written exactly once before commit and
    // read only by that token's dispatches.
    batch_wte_slices: Option<Vec<Buffer>>, // MAX_FORWARD_BATCH x [n_embd]
    batch_wpe_slices: Option<Vec<Buffer>>, // MAX_FORWARD_BATCH x [n_embd]
    batch_logits: Option<Vec<Buffer>>,     // MAX_FORWARD_BATCH x [vocab_size]
    batch_pos_scalars: Option<Vec<Buffer>>, // MAX_FORWARD_BATCH x u32
    batch_seq_len_scalars: Option<Vec<Buffer>>, // MAX_FORWARD_BATCH x u32
}

impl GpuBackend {
    /// Create a new uncompiled GPU backend.
    ///
    /// Returns an error if no Metal device is available (should not happen on
    /// any Apple Silicon or Intel Mac with GPU).
    pub fn new() -> Result<Self, CompileError> {
        let device = Device::system_default()
            .ok_or_else(|| CompileError::DeviceError("no Metal device available".to_string()))?;
        let command_queue = device.new_command_queue();
        Ok(Self {
            device,
            command_queue,
            compiled: false,
            needs_recompile: false,
            config_snapshot: None,
            pipelines: None,
            weight_buffers: None,
            x_buf: None,
            xr_buf: None,
            xr2_buf: None,
            q_buf: None,
            k_buf: None,
            v_buf: None,
            attn_out_buf: None,
            hidden_buf: None,
            scores_buf: None,
            logits_buf: None,
            kv_cache: None,
            n_embd_scalar: None,
            kv_dim_scalar: None,
            head_dim_scalar: None,
            mlp_hidden_scalar: None,
            eps_scalar: None,
            scale_scalar: None,
            q_off_bufs: None,
            kv_off_bufs: None,
            out_off_bufs: None,
            seq_len_scalar: None,
            pos_scalar: None,
            emb_wte_slice: None,
            emb_wpe_slice: None,
            batch_wte_slices: None,
            batch_wpe_slices: None,
            batch_logits: None,
            batch_pos_scalars: None,
            batch_seq_len_scalars: None,
        })
    }

    /// Build the borrowed resource view shared by `forward` and
    /// `forward_batch`. All unwraps are guarded by the `compiled` flag at
    /// the call sites.
    fn frame(&self) -> GpuFrame<'_> {
        GpuFrame {
            pipelines: self
                .pipelines
                .as_ref()
                .expect("pipelines missing after compile"),
            wb: self
                .weight_buffers
                .as_ref()
                .expect("weights missing after compile"),
            kv_cache: self
                .kv_cache
                .as_ref()
                .expect("kv_cache missing after compile"),
            x_buf: self.x_buf.as_ref().expect("x_buf missing"),
            xr_buf: self.xr_buf.as_ref().expect("xr_buf missing"),
            xr2_buf: self.xr2_buf.as_ref().expect("xr2_buf missing"),
            q_buf: self.q_buf.as_ref().expect("q_buf missing"),
            k_buf: self.k_buf.as_ref().expect("k_buf missing"),
            v_buf: self.v_buf.as_ref().expect("v_buf missing"),
            attn_out_buf: self.attn_out_buf.as_ref().expect("attn_out_buf missing"),
            hidden_buf: self.hidden_buf.as_ref().expect("hidden_buf missing"),
            scores_buf: self.scores_buf.as_ref().expect("scores_buf missing"),
            n_embd_buf: self.n_embd_scalar.as_ref().expect("n_embd_scalar missing"),
            kv_dim_buf: self.kv_dim_scalar.as_ref().expect("kv_dim_scalar missing"),
            head_dim_buf: self
                .head_dim_scalar
                .as_ref()
                .expect("head_dim_scalar missing"),
            eps_buf: self.eps_scalar.as_ref().expect("eps_scalar missing"),
            scale_buf: self.scale_scalar.as_ref().expect("scale_scalar missing"),
            mlp_hidden_in_dim_buf: self
                .mlp_hidden_scalar
                .as_ref()
                .expect("mlp_hidden_scalar missing"),
            q_off_bufs: self.q_off_bufs.as_ref().expect("q_off_bufs missing"),
            kv_off_bufs: self.kv_off_bufs.as_ref().expect("kv_off_bufs missing"),
            out_off_bufs: self.out_off_bufs.as_ref().expect("out_off_bufs missing"),
        }
    }

    /// Batched width-N forward (Issue 660 task 5): process `tokens` starting
    /// at absolute position `pos` in ONE command buffer with a single commit
    /// and wait, returning logits for EVERY position (the MTP verification
    /// shape). Returns a flat row-major slice: row `i` = logits for
    /// `tokens[i]`, i.e. `out[i * vocab_size..(i + 1) * vocab_size]`.
    ///
    /// Semantics are identical to `tokens.len()` sequential `forward` calls:
    /// token i attends over positions `0..=pos + i` (earlier batch tokens'
    /// KV entries are written by earlier dispatches in the same command
    /// buffer — encoder serialization provides the memory visibility, the
    /// same guarantee the single-token path relies on).
    ///
    /// Per-slot upload buffers (allocated at `compile()`) carry each token's
    /// embedding row + pos/seq_len scalars; all CPU writes land before the
    /// single commit.
    ///
    /// Falls back to sequential CPU forwards when not compiled. Panics when
    /// `tokens.len() > MAX_FORWARD_BATCH` or the batch would run past
    /// `config.block_size`.
    pub fn forward_batch(
        &mut self,
        ctx: &mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        tokens: &[usize],
        pos: usize,
        config: &Config,
    ) -> Vec<f32> {
        let n = tokens.len();
        if n == 0 {
            return Vec::new();
        }
        if !self.compiled {
            // CPU fallback: sequential single-token forwards.
            let mut out = Vec::with_capacity(n * config.vocab_size);
            for (i, &tok) in tokens.iter().enumerate() {
                let logits = katgpt_forward::forward(ctx, weights, cache, tok, pos + i, config);
                out.extend_from_slice(logits);
            }
            return out;
        }
        assert!(
            n <= MAX_FORWARD_BATCH,
            "forward_batch: {n} tokens > MAX_FORWARD_BATCH ({MAX_FORWARD_BATCH})"
        );
        assert!(
            pos + n <= config.block_size,
            "forward_batch: pos {pos} + {n} tokens exceeds block_size {}",
            config.block_size
        );

        let n_embd = config.n_embd;
        let vocab_size = config.vocab_size;

        // CPU-side uploads (all pre-encode): per-slot embedding rows + scalars.
        let wte_slots = self
            .batch_wte_slices
            .as_ref()
            .expect("batch_wte_slices missing");
        let wpe_slots = self
            .batch_wpe_slices
            .as_ref()
            .expect("batch_wpe_slices missing");
        let logits_slots = self.batch_logits.as_ref().expect("batch_logits missing");
        let pos_slots = self
            .batch_pos_scalars
            .as_ref()
            .expect("batch_pos_scalars missing");
        let seq_slots = self
            .batch_seq_len_scalars
            .as_ref()
            .expect("batch_seq_len_scalars missing");
        for i in 0..n {
            let tok_pos = pos + i;
            write_scalar(&pos_slots[i], &(tok_pos as u32));
            write_scalar(&seq_slots[i], &((tok_pos + 1) as u32));
            let tok = tokens[i];
            write_f32_slice(
                &wte_slots[i],
                &weights.wte[tok * n_embd..(tok + 1) * n_embd],
            );
            write_f32_slice(
                &wpe_slots[i],
                &weights.wpe[tok_pos * n_embd..(tok_pos + 1) * n_embd],
            );
        }

        // ONE command buffer for all N tokens (the Issue 660 single-submit
        // guarantee extended across the batch).
        let cmd_buffer = self.command_queue.new_command_buffer();
        let frame = self.frame();
        for i in 0..n {
            frame.encode_token(
                cmd_buffer,
                &wte_slots[i],
                &wpe_slots[i],
                &logits_slots[i],
                &pos_slots[i],
                &seq_slots[i],
                pos + i,
                config,
            );
        }
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // Readback: all N rows, once.
        let mut out = Vec::with_capacity(n * vocab_size);
        for buf in &logits_slots[..n] {
            let ptr = buf.contents() as *const f32;
            let row = unsafe { std::slice::from_raw_parts(ptr, vocab_size) };
            out.extend_from_slice(row);
        }

        // Keep ctx.logits coherent with the last token (consumer compat with
        // forward) + update the CPU KV-cache fill tracker.
        let last = &out[(n - 1) * vocab_size..];
        ctx.logits[..vocab_size].copy_from_slice(last);
        cache.advance_pos(pos + n - 1);
        out
    }
}

/// Borrowed view of the per-forward GPU resources (all allocated once in
/// `compile()`). Bundled so the per-token dispatch body is written ONCE and
/// shared verbatim by the single-token `forward` and the batched
/// `forward_batch` (Issue 660 task 5).
struct GpuFrame<'a> {
    pipelines: &'a GpuPipelines,
    wb: &'a GpuWeightBuffers,
    kv_cache: &'a [(Buffer, Buffer)],
    // Reused per-token scratch — dispatches serialize within a command
    // buffer, so consecutive tokens reuse the same buffers exactly like
    // sequential separate forwards.
    x_buf: &'a Buffer,
    xr_buf: &'a Buffer,
    xr2_buf: &'a Buffer,
    q_buf: &'a Buffer,
    k_buf: &'a Buffer,
    v_buf: &'a Buffer,
    attn_out_buf: &'a Buffer,
    hidden_buf: &'a Buffer,
    scores_buf: &'a Buffer,
    // Config-derived constant scalars (bound as buffers on every dispatch).
    n_embd_buf: &'a Buffer,
    kv_dim_buf: &'a Buffer,
    head_dim_buf: &'a Buffer,
    eps_buf: &'a Buffer,
    scale_buf: &'a Buffer,
    mlp_hidden_in_dim_buf: &'a Buffer,
    // Per-head offsets (fixed at compile time).
    q_off_bufs: &'a [Buffer],
    kv_off_bufs: &'a [Buffer],
    out_off_bufs: &'a [Buffer],
}

impl GpuFrame<'_> {
    /// Encode ONE full token pipeline (embedding → per-layer blocks →
    /// lm_head) into `cmd_buffer` as dispatches only — no commit, no CPU
    /// readback.
    ///
    /// The caller has already uploaded the embedding row into `wte_src` /
    /// `wpe_src` and written this token's absolute position into `pos_buf`
    /// and `pos + 1` into `seq_len_buf` (CPU `contents()` writes all land
    /// before the command buffer is committed). `pos` itself only drives
    /// per-dispatch thread counts (encode-time constants).
    ///
    /// Batched callers encode N tokens into the SAME command buffer with
    /// per-slot buffers: token i's KV-cache write lands at `pos + i` and its
    /// attention dispatches read exactly `[0, pos + i]` — encoder
    /// serialization makes earlier batch tokens' KV visible, the identical
    /// guarantee the single-token path relies on.
    #[allow(clippy::too_many_arguments)]
    fn encode_token(
        &self,
        cmd_buffer: &metal::CommandBufferRef,
        wte_src: &Buffer,
        wpe_src: &Buffer,
        logits_dst: &Buffer,
        pos_buf: &Buffer,
        seq_len_buf: &Buffer,
        pos: usize,
        config: &Config,
    ) {
        let seq_len = pos + 1; // number of cached positions incl. this token
        let n_embd = config.n_embd;
        let kv_dim = kv_dim(config);
        let n_head = config.n_head;
        let mlp_hidden = config.mlp_hidden;
        let vocab_size = config.vocab_size;
        let pipelines = &self.pipelines;
        let wb = &self.wb;
        let kv_cache = &self.kv_cache;
        // Shortcut references to scratch buffers
        let x_buf = self.x_buf;
        let xr_buf = self.xr_buf;
        let xr2_buf = self.xr2_buf;
        let q_buf = self.q_buf;
        let k_buf = self.k_buf;
        let v_buf = self.v_buf;
        let attn_out_buf = self.attn_out_buf;
        let hidden_buf = self.hidden_buf;
        let scores_buf = self.scores_buf;
        // Cached scalar buffers (constants consumed at compile() time).
        let n_embd_buf = self.n_embd_buf;
        let kv_dim_buf = self.kv_dim_buf;
        let head_dim_buf = self.head_dim_buf;
        let eps_buf = self.eps_buf;
        let scale_buf = self.scale_buf;
        let mlp_hidden_in_dim_buf = self.mlp_hidden_in_dim_buf;
        let q_off_bufs = self.q_off_bufs;
        let kv_off_bufs = self.kv_off_bufs;
        let out_off_bufs = self.out_off_bufs;

        // ── Step 1: Embedding lookup x = wte[token] + wpe[pos] ──
        // (the caller uploaded the token's wte row + the wpe row for this
        // token's absolute position into wte_src / wpe_src before encoding)
        {
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_buffer(0, Some(wte_src), 0);
            encoder.set_buffer(1, Some(wpe_src), 0);
            encoder.set_buffer(2, Some(x_buf), 0);
            dispatch_1d(encoder, &pipelines.elem_add, n_embd as u64);
            encoder.end_encoding();
        }

        // ── Step 2: Per-layer transformer block ──
        #[allow(clippy::needless_range_loop)]
        for layer_idx in 0..config.n_layer {
            let encoder = cmd_buffer.new_compute_command_encoder();

            // 2a. RMSNorm: xr = rmsnorm(x, attn_norm_gamma)
            encoder.set_buffer(0, Some(x_buf), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(&wb.attn_norm_gamma[layer_idx]), 0);
            encoder.set_buffer(3, Some(n_embd_buf), 0);
            encoder.set_buffer(4, Some(eps_buf), 0);
            dispatch_single(encoder, &pipelines.rmsnorm);

            // 2b. QKV projection: q = wq @ xr, k = wk @ xr, v = wv @ xr
            // Query: q = attn_wq @ xr  [n_embd output, n_embd input]
            encoder.set_buffer(0, Some(&wb.attn_wq[layer_idx]), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(q_buf), 0);
            encoder.set_buffer(3, Some(n_embd_buf), 0); // in_dim = n_embd
            dispatch_1d(encoder, &pipelines.matmul, n_embd as u64);

            // Key: k = attn_wk @ xr  [kv_dim output, n_embd input]
            encoder.set_buffer(0, Some(&wb.attn_wk[layer_idx]), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(k_buf), 0);
            encoder.set_buffer(3, Some(n_embd_buf), 0); // in_dim = n_embd (cached)
            dispatch_1d(encoder, &pipelines.matmul, kv_dim as u64);

            // Value: v = attn_wv @ xr  [kv_dim output, n_embd input]
            encoder.set_buffer(0, Some(&wb.attn_wv[layer_idx]), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(v_buf), 0);
            encoder.set_buffer(3, Some(n_embd_buf), 0); // in_dim = n_embd (cached)
            dispatch_1d(encoder, &pipelines.matmul, kv_dim as u64);

            // 2c. KV cache store: write k, v into cache at position pos
            let (key_cache_buf, value_cache_buf) = &kv_cache[layer_idx];
            encoder.set_buffer(0, Some(k_buf), 0);
            encoder.set_buffer(1, Some(v_buf), 0);
            encoder.set_buffer(2, Some(key_cache_buf), 0);
            encoder.set_buffer(3, Some(value_cache_buf), 0);
            encoder.set_buffer(4, Some(pos_buf), 0); // this token's slot
            encoder.set_buffer(5, Some(kv_dim_buf), 0);
            dispatch_1d(encoder, &pipelines.kv_store, kv_dim as u64);

            // 2d. Multi-head attention
            // For each head, compute attention scores, softmax, and weighted value sum.
            // All heads share this ONE encoder: dispatches within an encoder
            // serialize, so head h's softmax is visible to its value sum, and
            // head h's value sum completes before head h+1 overwrites the
            // shared scores_buf — the identical guarantee the old per-head
            // command-buffer waits provided, without the CPU round trips.
            // (key_cache_buf / value_cache_buf bound above at 2c.)

            for h in 0..n_head {
                // Per-head offsets — pre-computed buffers cached at compile().
                let q_off_buf = &q_off_bufs[h];
                let kv_off_buf = &kv_off_bufs[h];
                let out_off_buf = &out_off_bufs[h];

                // Attention scores: scores[t] = q[h*head_dim..] . key_cache[t*kv_dim + kv_group*head_dim..] * scale
                encoder.set_buffer(0, Some(q_buf), 0);
                encoder.set_buffer(1, Some(key_cache_buf), 0);
                encoder.set_buffer(2, Some(q_off_buf), 0);
                encoder.set_buffer(3, Some(kv_off_buf), 0);
                encoder.set_buffer(4, Some(kv_dim_buf), 0);
                encoder.set_buffer(5, Some(head_dim_buf), 0);
                encoder.set_buffer(6, Some(scale_buf), 0);
                encoder.set_buffer(7, Some(scores_buf), 0);
                dispatch_1d(encoder, &pipelines.attention_score, seq_len as u64);

                // Softmax over scores[0..seq_len]
                encoder.set_buffer(0, Some(scores_buf), 0);
                encoder.set_buffer(1, Some(seq_len_buf), 0);
                dispatch_single(encoder, &pipelines.softmax);

                // Weighted value sum: attn_out[h*head_dim..] = sum_t scores[t] * value_cache[t*kv_dim + kv_offset..]
                encoder.set_buffer(0, Some(scores_buf), 0);
                encoder.set_buffer(1, Some(value_cache_buf), 0);
                encoder.set_buffer(2, Some(attn_out_buf), 0);
                encoder.set_buffer(3, Some(out_off_buf), 0);
                encoder.set_buffer(4, Some(kv_off_buf), 0);
                encoder.set_buffer(5, Some(kv_dim_buf), 0);
                encoder.set_buffer(6, Some(head_dim_buf), 0);
                encoder.set_buffer(7, Some(seq_len_buf), 0);
                dispatch_single(encoder, &pipelines.attention_value);
            }

            // 2e. Output projection: x = wo @ attn_out  [n_embd output, n_embd input]
            {
                encoder.set_buffer(0, Some(&wb.attn_wo[layer_idx]), 0);
                encoder.set_buffer(1, Some(attn_out_buf), 0);
                encoder.set_buffer(2, Some(x_buf), 0);
                encoder.set_buffer(3, Some(n_embd_buf), 0); // in_dim = n_embd (cached)
                dispatch_1d(encoder, &pipelines.matmul, n_embd as u64);
            }

            // 2f. Residual: x += xr
            {
                encoder.set_buffer(0, Some(x_buf), 0);
                encoder.set_buffer(1, Some(xr_buf), 0);
                encoder.set_buffer(2, Some(x_buf), 0);
                dispatch_1d(encoder, &pipelines.elem_add, n_embd as u64);
            }

            encoder.end_encoding();

            // 2g. Save pre-norm x to xr2, then RMSNorm x in-place
            // xr2 = x (save pre-norm value for residual)
            // x = rmsnorm(x, mlp_norm_gamma)
            //
            // The save is a GPU-side blit copy INSIDE the command buffer
            // (Issue 660): the old CPU ptr::copy via contents() was the one
            // mid-forward CPU↔GPU dependency that forced the per-block syncs.
            // Byte-for-byte identical to the CPU copy.
            {
                let blit = cmd_buffer.new_blit_command_encoder();
                blit.copy_from_buffer(
                    x_buf,
                    0,
                    xr2_buf,
                    0,
                    (n_embd * std::mem::size_of::<f32>()) as u64,
                );
                blit.end_encoding();
            }
            // RMSNorm x_buf in-place (new encoder — must not overlap the blit)
            {
                let encoder = cmd_buffer.new_compute_command_encoder();
                encoder.set_buffer(0, Some(x_buf), 0);
                encoder.set_buffer(1, Some(x_buf), 0); // in-place: output = input
                encoder.set_buffer(2, Some(&wb.mlp_norm_gamma[layer_idx]), 0);
                encoder.set_buffer(3, Some(n_embd_buf), 0);
                encoder.set_buffer(4, Some(eps_buf), 0);
                dispatch_single(encoder, &pipelines.rmsnorm);

                // 2h. MLP: hidden = relu(w1 @ x), then x = w2 @ hidden
                // hidden = w1 @ x  [mlp_hidden output, n_embd input]
                encoder.set_buffer(0, Some(&wb.mlp_w1[layer_idx]), 0);
                encoder.set_buffer(1, Some(x_buf), 0);
                encoder.set_buffer(2, Some(hidden_buf), 0);
                encoder.set_buffer(3, Some(n_embd_buf), 0); // in_dim = n_embd (cached)
                dispatch_1d(encoder, &pipelines.matmul, mlp_hidden as u64);

                // ReLU on hidden
                encoder.set_buffer(0, Some(hidden_buf), 0);
                encoder.set_buffer(1, Some(hidden_buf), 0);
                dispatch_1d(encoder, &pipelines.relu, mlp_hidden as u64);

                // x = w2 @ hidden  [n_embd output, mlp_hidden input]
                encoder.set_buffer(0, Some(&wb.mlp_w2[layer_idx]), 0);
                encoder.set_buffer(1, Some(hidden_buf), 0);
                encoder.set_buffer(2, Some(x_buf), 0);
                encoder.set_buffer(3, Some(mlp_hidden_in_dim_buf), 0);
                dispatch_1d(encoder, &pipelines.matmul, n_embd as u64);

                // 2i. Residual: x += xr2 (pre-norm value)
                encoder.set_buffer(0, Some(x_buf), 0);
                encoder.set_buffer(1, Some(xr2_buf), 0);
                encoder.set_buffer(2, Some(x_buf), 0);
                dispatch_1d(encoder, &pipelines.elem_add, n_embd as u64);

                encoder.end_encoding();
            }
        }

        // ── Step 3: LM head: logits = lm_head @ x ──
        {
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_buffer(0, Some(&wb.lm_head), 0);
            encoder.set_buffer(1, Some(x_buf), 0);
            encoder.set_buffer(2, Some(logits_dst), 0);
            encoder.set_buffer(3, Some(n_embd_buf), 0); // in_dim = n_embd (cached)
            dispatch_1d(encoder, &pipelines.matmul, vocab_size as u64);
            encoder.end_encoding();
        }
    }
}

impl InferenceBackend for GpuBackend {
    fn forward<'a>(
        &'a mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        // Fall back to CPU if not compiled
        if !self.compiled {
            return katgpt_forward::forward(ctx, weights, cache, token, pos, config);
        }

        let seq_len = pos + 1; // number of cached positions
        let n_embd = config.n_embd;
        let vocab_size = config.vocab_size;

        // Per-call CPU-side uploads (all land before the command buffer is
        // committed): the token's embedding row + the pos/seq_len scalars.
        // Reused pre-allocated buffers — no per-call Metal allocations.
        let seq_len_buf = self
            .seq_len_scalar
            .as_ref()
            .expect("seq_len_scalar missing");
        let pos_buf = self.pos_scalar.as_ref().expect("pos_scalar missing");
        write_scalar(seq_len_buf, &(seq_len as u32));
        write_scalar(pos_buf, &(pos as u32));
        let wte_upload = self.emb_wte_slice.as_ref().expect("emb_wte_slice missing");
        let wpe_upload = self.emb_wpe_slice.as_ref().expect("emb_wpe_slice missing");
        write_f32_slice(
            wte_upload,
            &weights.wte[token * n_embd..(token + 1) * n_embd],
        );
        write_f32_slice(wpe_upload, &weights.wpe[pos * n_embd..(pos + 1) * n_embd]);

        // ── Single-submit (Issue 660): the whole forward is encoded into ONE
        // command buffer with a single commit + wait at the end. Encoder order
        // within a command buffer IS the GPU-side ordering — dispatches within
        // a compute encoder serialize with memory visibility between them
        // (the same guarantee the old per-block waits provided). The one
        // mid-forward CPU dependency (the x→xr2 save in 2g) is a blit-encoder
        // copy inside the buffer, so no CPU read happens before the wait. ──
        let cmd_buffer = self.command_queue.new_command_buffer();
        self.frame().encode_token(
            cmd_buffer,
            wte_upload,
            wpe_upload,
            self.logits_buf.as_ref().expect("logits_buf missing"),
            pos_buf,
            seq_len_buf,
            pos,
            config,
        );
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();

        // ── Copy logits from GPU back to CPU ──
        {
            let logits_buf = self.logits_buf.as_ref().expect("logits_buf missing");
            let ptr = logits_buf.contents() as *const f32;
            let gpu_logits = unsafe { std::slice::from_raw_parts(ptr, vocab_size) };
            ctx.logits[..vocab_size].copy_from_slice(gpu_logits);
        }

        // Also update CPU KV cache for fallback consistency
        cache.advance_pos(pos);

        &mut ctx.logits[..vocab_size]
    }

    fn device_name(&self) -> &'static str {
        "GPU"
    }

    fn compile(
        &mut self,
        weights: &TransformerWeights,
        config: &Config,
    ) -> Result<(), CompileError> {
        let device = &*self.device;

        // Compile Metal shaders and create pipelines
        let pipelines = GpuPipelines::create(device)?;

        // Upload all weights to GPU buffers
        let n_layer = weights.layers.len();
        let wb = GpuWeightBuffers {
            wte: upload_buffer(device, &weights.wte),
            wpe: upload_buffer(device, &weights.wpe),
            lm_head: upload_buffer(device, &weights.lm_head),
            attn_wq: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.attn_wq));
                }
                v
            },
            attn_wk: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.attn_wk));
                }
                v
            },
            attn_wv: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.attn_wv));
                }
                v
            },
            attn_wo: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.attn_wo));
                }
                v
            },
            mlp_w1: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.mlp_w1));
                }
                v
            },
            mlp_w2: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.mlp_w2));
                }
                v
            },
            attn_norm_gamma: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.attn_norm_gamma));
                }
                v
            },
            mlp_norm_gamma: {
                let mut v = Vec::with_capacity(n_layer);
                for l in &weights.layers {
                    v.push(upload_buffer(device, &l.mlp_norm_gamma));
                }
                v
            },
        };

        let n_embd = config.n_embd;
        let kvd = kv_dim(config);
        let block_size = config.block_size;

        // Allocate runtime scratch buffers
        let x_buf = zero_buffer(device, n_embd);
        let xr_buf = zero_buffer(device, n_embd);
        let xr2_buf = zero_buffer(device, n_embd);
        let q_buf = zero_buffer(device, n_embd);
        let k_buf = zero_buffer(device, kvd);
        let v_buf = zero_buffer(device, kvd);
        let attn_out_buf = zero_buffer(device, n_embd);
        let hidden_buf = zero_buffer(device, config.mlp_hidden);
        let scores_buf = zero_buffer(device, block_size);
        let logits_buf = zero_buffer(device, config.vocab_size);

        // Allocate GPU-side KV cache: per-layer (key, value) buffers
        let kv_cache: Vec<(Buffer, Buffer)> = (0..config.n_layer)
            .map(|_| {
                (
                    zero_buffer(device, block_size * kvd),
                    zero_buffer(device, block_size * kvd),
                )
            })
            .collect();

        // Pre-allocate cached scalar buffers that never change between forward() calls.
        // Previously these were allocated per-forward via scalar_buffer(), causing
        // ~9 + n_layer + 3*n_head*n_layer Metal `device.new_buffer()` IPC calls
        // every token. Now allocated once at compile().
        let n_head = config.n_head;
        let n_kv_head = config.n_kv_head;
        let head_dim = config.head_dim;
        let eps_val = config.rms_norm_eps as f32;
        let scale_val = 1.0f32 / (head_dim as f32).sqrt();

        let n_embd_scalar = scalar_buffer(device, &(n_embd as u32));
        let kv_dim_scalar = scalar_buffer(device, &(kvd as u32));
        let head_dim_scalar = scalar_buffer(device, &(head_dim as u32));
        let mlp_hidden_scalar = scalar_buffer(device, &(config.mlp_hidden as u32));
        let eps_scalar = scalar_buffer(device, &eps_val);
        let scale_scalar = scalar_buffer(device, &scale_val);

        // Per-head offset buffers — head layout is fixed at compile time.
        let mut q_off_bufs = Vec::with_capacity(n_head);
        let mut kv_off_bufs = Vec::with_capacity(n_head);
        let mut out_off_bufs = Vec::with_capacity(n_head);
        for h in 0..n_head {
            let kv_group = h * n_kv_head / n_head;
            q_off_bufs.push(scalar_buffer(device, &((h * head_dim) as u32)));
            kv_off_bufs.push(scalar_buffer(device, &((kv_group * head_dim) as u32)));
            out_off_bufs.push(scalar_buffer(device, &((h * head_dim) as u32)));
        }

        // Per-call reusable buffers — allocated once, overwritten each forward via
        // `contents()` write. Eliminates 4 `device.new_buffer()` IPC calls per token.
        let seq_len_scalar = scalar_buffer(device, &0u32);
        let pos_scalar = scalar_buffer(device, &0u32);
        let emb_wte_slice = zero_buffer(device, n_embd);
        let emb_wpe_slice = zero_buffer(device, n_embd);

        // Batched width-N per-slot buffers (Issue 660 task 5): each slot is
        // written exactly once before commit and read only by that token's
        // dispatches — per-slot, not shared, because CPU writes to a shared
        // buffer all land before GPU execution (every dispatch would see the
        // last value).
        let batch_wte_slices = (0..MAX_FORWARD_BATCH)
            .map(|_| zero_buffer(device, n_embd))
            .collect();
        let batch_wpe_slices = (0..MAX_FORWARD_BATCH)
            .map(|_| zero_buffer(device, n_embd))
            .collect();
        let batch_logits = (0..MAX_FORWARD_BATCH)
            .map(|_| zero_buffer(device, config.vocab_size))
            .collect();
        let batch_pos_scalars = (0..MAX_FORWARD_BATCH)
            .map(|_| scalar_buffer(device, &0u32))
            .collect();
        let batch_seq_len_scalars = (0..MAX_FORWARD_BATCH)
            .map(|_| scalar_buffer(device, &0u32))
            .collect();

        // Store everything
        self.pipelines = Some(pipelines);
        self.weight_buffers = Some(wb);
        self.x_buf = Some(x_buf);
        self.xr_buf = Some(xr_buf);
        self.xr2_buf = Some(xr2_buf);
        self.q_buf = Some(q_buf);
        self.k_buf = Some(k_buf);
        self.v_buf = Some(v_buf);
        self.attn_out_buf = Some(attn_out_buf);
        self.hidden_buf = Some(hidden_buf);
        self.scores_buf = Some(scores_buf);
        self.logits_buf = Some(logits_buf);
        self.kv_cache = Some(kv_cache);
        self.n_embd_scalar = Some(n_embd_scalar);
        self.kv_dim_scalar = Some(kv_dim_scalar);
        self.head_dim_scalar = Some(head_dim_scalar);
        self.mlp_hidden_scalar = Some(mlp_hidden_scalar);
        self.eps_scalar = Some(eps_scalar);
        self.scale_scalar = Some(scale_scalar);
        self.q_off_bufs = Some(q_off_bufs);
        self.kv_off_bufs = Some(kv_off_bufs);
        self.out_off_bufs = Some(out_off_bufs);
        self.seq_len_scalar = Some(seq_len_scalar);
        self.pos_scalar = Some(pos_scalar);
        self.emb_wte_slice = Some(emb_wte_slice);
        self.emb_wpe_slice = Some(emb_wpe_slice);
        self.batch_wte_slices = Some(batch_wte_slices);
        self.batch_wpe_slices = Some(batch_wpe_slices);
        self.batch_logits = Some(batch_logits);
        self.batch_pos_scalars = Some(batch_pos_scalars);
        self.batch_seq_len_scalars = Some(batch_seq_len_scalars);
        self.config_snapshot = Some(config.clone());
        self.compiled = true;
        self.needs_recompile = false;

        Ok(())
    }

    #[inline]
    fn is_compiled(&self) -> bool {
        self.compiled
    }

    fn recompile_hint(&mut self) {
        self.needs_recompile = true;
    }

    fn reset(&mut self) {
        // Mark for recompile so next compile() re-uploads weights
        if self.compiled {
            self.needs_recompile = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_forward as transformer;
    use katgpt_types::Rng;

    /// Create micro test fixtures (small model config + random weights).
    fn micro_fixtures() -> (
        Config,
        TransformerWeights,
        ForwardContext,
        MultiLayerKVCache,
    ) {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let ctx = ForwardContext::new(&config);
        let cache = MultiLayerKVCache::new(&config);
        (config, weights, ctx, cache)
    }

    /// Cosine similarity between two f32 slices.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (mag_a * mag_b + 1e-8)
    }

    #[test]
    fn test_gpu_backend_new_uncompiled() {
        let backend = GpuBackend::new();
        // May fail on CI without Metal, so just check the happy path
        if let Ok(b) = backend {
            assert!(!b.is_compiled(), "new backend should not be compiled");
        }
    }

    #[test]
    fn test_gpu_backend_compile_marks_compiled() {
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return, // Skip on non-Metal environments
        };
        assert!(backend.compile(&weights, &config).is_ok());
        assert!(backend.is_compiled());
    }

    #[test]
    fn test_gpu_backend_device_name() {
        let Ok(b) = GpuBackend::new() else { return };
        assert_eq!(b.device_name(), "GPU");
    }

    #[test]
    fn test_gpu_forward_matches_cpu() {
        if !transformer::CPU_FORWARD_USES_DEVICE_BASE_PATH {
            eprintln!("Skipping: CPU forward uses a feature-gated path the GPU doesn't implement");
            return;
        }
        let (config, weights, _, _) = micro_fixtures();

        // CPU reference
        let mut ctx_cpu = ForwardContext::new(&config);
        let mut cache_cpu = MultiLayerKVCache::new(&config);
        let cpu_logits =
            transformer::forward(&mut ctx_cpu, &weights, &mut cache_cpu, 0, 0, &config).to_vec();

        // GPU
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return; // Skip if compilation fails
        }
        let mut ctx_gpu = ForwardContext::new(&config);
        let mut cache_gpu = MultiLayerKVCache::new(&config);
        let gpu_logits = backend
            .forward(&mut ctx_gpu, &weights, &mut cache_gpu, 0, 0, &config)
            .to_vec();

        let sim = cosine_similarity(&cpu_logits, &gpu_logits);
        assert!(
            sim >= 0.999,
            "GPU logits should match CPU (cosine sim = {sim:.6}, expected >= 0.999)\n\
             CPU first 10: {:?}\n\
             GPU first 10: {:?}",
            &cpu_logits[..cpu_logits.len().min(10)],
            &gpu_logits[..gpu_logits.len().min(10)],
        );
    }

    #[test]
    fn test_gpu_forward_multi_token_matches_cpu() {
        if !transformer::CPU_FORWARD_USES_DEVICE_BASE_PATH {
            eprintln!("Skipping: CPU forward uses a feature-gated path the GPU doesn't implement");
            return;
        }
        let (config, weights, _, _) = micro_fixtures();
        let tokens = [0usize, 1, 5, 10, 3];

        // CPU reference: run multiple tokens
        let mut ctx_cpu = ForwardContext::new(&config);
        let mut cache_cpu = MultiLayerKVCache::new(&config);
        let mut cpu_logits_all = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let logits =
                transformer::forward(&mut ctx_cpu, &weights, &mut cache_cpu, token, pos, &config)
                    .to_vec();
            cpu_logits_all.push(logits);
        }

        // GPU: run same tokens
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }
        let mut ctx_gpu = ForwardContext::new(&config);
        let mut cache_gpu = MultiLayerKVCache::new(&config);
        let mut gpu_logits_all = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let logits = backend
                .forward(&mut ctx_gpu, &weights, &mut cache_gpu, token, pos, &config)
                .to_vec();
            gpu_logits_all.push(logits);
        }

        // Compare each position
        for (pos, (cpu, gpu)) in cpu_logits_all.iter().zip(gpu_logits_all.iter()).enumerate() {
            let sim = cosine_similarity(cpu, gpu);
            assert!(
                sim >= 0.999,
                "GPU logits should match CPU at pos {pos} (cosine sim = {sim:.6})"
            );
        }
    }

    #[test]
    fn test_gpu_forward_deterministic() {
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }

        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let run1 = backend
            .forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config)
            .to_vec();

        // Reset and recompile to get fresh KV cache
        backend.reset();
        assert!(backend.compile(&weights, &config).is_ok());

        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let run2 = backend
            .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
            .to_vec();

        assert_eq!(run1, run2, "same input must produce same GPU logits");
    }

    #[test]
    fn test_gpu_recompile_hint() {
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        assert!(!backend.needs_recompile);
        backend.recompile_hint();
        assert!(backend.needs_recompile);
    }

    #[test]
    fn test_gpu_fallback_on_no_device() {
        // This test verifies the error path when no Metal device is available.
        // On most Macs, Device::system_default() succeeds, so this test is
        // effectively a no-op. It validates the error type when Metal is absent.
        if Device::system_default().is_none() {
            match GpuBackend::new() {
                Err(CompileError::DeviceError(msg)) => {
                    assert!(msg.contains("Metal"), "error should mention Metal: {msg}");
                }
                Err(other) => panic!("expected DeviceError, got {other:?}"),
                Ok(_) => panic!("expected error when no Metal device"),
            }
        }
    }

    // ── GOAT: GPU forward == CPU forward ─────────────────────────

    #[test]
    fn test_goat_gpu_forward_matches_cpu() {
        if !transformer::CPU_FORWARD_USES_DEVICE_BASE_PATH {
            eprintln!("Skipping: CPU forward uses a feature-gated path the GPU doesn't implement");
            return;
        }
        let (config, weights, _, _) = micro_fixtures();

        // Run a continuous sequence of tokens — same sequence for CPU and GPU.
        // Each step depends on the KV cache from prior steps, so we build up
        // state progressively. Compare logits at each step.
        let token_seq: [usize; 5] = [0, 1, 3, 7, 5];

        // CPU reference: continuous sequence with shared cache
        let mut cpu_logits_all = Vec::new();
        {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            for (pos, &token) in token_seq.iter().enumerate() {
                let logits =
                    transformer::forward(&mut ctx, &weights, &mut cache, token, pos, &config)
                        .to_vec();
                cpu_logits_all.push(logits);
            }
        }

        // GPU forward: same continuous sequence with shared cache
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return, // Skip on non-Metal environments
        };
        if backend.compile(&weights, &config).is_err() {
            return; // Skip if compilation fails
        }

        let mut gpu_logits_all = Vec::new();
        {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            for (pos, &token) in token_seq.iter().enumerate() {
                let logits = backend
                    .forward(&mut ctx, &weights, &mut cache, token, pos, &config)
                    .to_vec();
                gpu_logits_all.push(logits);
            }
        }

        // Assert cosine similarity ≥ 0.999 for ALL positions
        for (i, (cpu, gpu)) in cpu_logits_all.iter().zip(gpu_logits_all.iter()).enumerate() {
            let sim = cosine_similarity(cpu, gpu);
            eprintln!(
                "GOAT GPU pos {i} (token={}): cosine_sim={sim:.6}",
                token_seq[i]
            );
            assert!(
                sim >= 0.999,
                "GOAT GPU forward mismatch at pos {i} (token={}): \
                 cosine_sim={sim:.6} < 0.999",
                token_seq[i]
            );
        }
    }

    // ── Issue 660 task 5: batched width-N forward ──────────────────

    #[test]
    fn test_gpu_forward_batch_matches_sequential() {
        let (config, weights, _, _) = micro_fixtures();
        let token_seq: [usize; 5] = [0, 1, 3, 7, 5];

        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return, // Skip on non-Metal environments
        };
        if backend.compile(&weights, &config).is_err() {
            return; // Skip if compilation fails
        }

        // Reference: sequential single-token GPU forwards + a follow-up
        // token that probes KV-cache state divergence.
        let (seq_logits, followup_seq): (Vec<Vec<f32>>, Vec<f32>) = {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let mut acc = Vec::new();
            for (pos, &tok) in token_seq.iter().enumerate() {
                acc.push(
                    backend
                        .forward(&mut ctx, &weights, &mut cache, tok, pos, &config)
                        .to_vec(),
                );
            }
            let followup = backend
                .forward(&mut ctx, &weights, &mut cache, 2, token_seq.len(), &config)
                .to_vec();
            (acc, followup)
        };

        // Batched: all 5 tokens at pos 0 in ONE command buffer + same follow-up.
        let (batch_logits, followup_batch): (Vec<f32>, Vec<f32>) = {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let batched =
                backend.forward_batch(&mut ctx, &weights, &mut cache, &token_seq, 0, &config);
            let followup = backend
                .forward(&mut ctx, &weights, &mut cache, 2, token_seq.len(), &config)
                .to_vec();
            (batched, followup)
        };

        let vocab = config.vocab_size;
        assert_eq!(batch_logits.len(), token_seq.len() * vocab);

        // Bit-identical logits at every position: same kernels, same values,
        // same per-token dispatch order — only the submit grouping differs.
        for (i, row) in seq_logits.iter().enumerate() {
            let brow = &batch_logits[i * vocab..(i + 1) * vocab];
            for (j, (a, b)) in row.iter().zip(brow.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "batched logits differ from sequential at pos {i} logit {j}"
                );
            }
        }

        // The follow-up token catches KV-cache divergence the logits rows
        // alone cannot (e.g. a mis-placed cache slot written beyond seq_len).
        for (j, (a, b)) in followup_seq.iter().zip(followup_batch.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "follow-up token diverged at logit {j} — KV cache state differs"
            );
        }
    }

    #[test]
    fn test_gpu_forward_batch_matches_cpu() {
        if !transformer::CPU_FORWARD_USES_DEVICE_BASE_PATH {
            eprintln!("Skipping: CPU forward uses a feature-gated path the GPU doesn't implement");
            return;
        }
        let (config, weights, _, _) = micro_fixtures();
        let token_seq: [usize; 5] = [0, 1, 3, 7, 5];

        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let batched = backend.forward_batch(&mut ctx, &weights, &mut cache, &token_seq, 0, &config);

        // CPU reference: same continuous sequence with shared cache
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let vocab = config.vocab_size;
        for (i, &tok) in token_seq.iter().enumerate() {
            let cpu = transformer::forward(&mut ctx2, &weights, &mut cache2, tok, i, &config);
            let brow = &batched[i * vocab..(i + 1) * vocab];
            let sim = cosine_similarity(cpu, brow);
            eprintln!("batch pos {i} (token={tok}): cosine_sim={sim:.6}");
            assert!(
                sim >= 0.999,
                "batch vs CPU mismatch at pos {i}: {sim:.6} < 0.999"
            );
        }
    }

    #[test]
    fn test_gpu_forward_batch_cpu_fallback() {
        // Uncompiled backend: forward_batch loops the CPU forward per token.
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        let token_seq: [usize; 5] = [0, 1, 3, 7, 5];

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let batched = backend.forward_batch(&mut ctx, &weights, &mut cache, &token_seq, 0, &config);

        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let vocab = config.vocab_size;
        assert_eq!(batched.len(), token_seq.len() * vocab);
        for (i, &tok) in token_seq.iter().enumerate() {
            let cpu = transformer::forward(&mut ctx2, &weights, &mut cache2, tok, i, &config);
            let brow = &batched[i * vocab..(i + 1) * vocab];
            for (a, b) in cpu.iter().zip(brow.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "CPU fallback diverged at pos {i}");
            }
        }
    }

    #[test]
    fn test_gpu_forward_batch_empty() {
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let out = backend.forward_batch(&mut ctx, &weights, &mut cache, &[], 0, &config);
        assert!(out.is_empty());
    }

    #[test]
    fn bench_gpu_forward_batch_vs_sequential() {
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }

        // Global warm-up (shader JIT / queue warm) so the FIRST measured width
        // does not absorb cold-start cost on top of its own discarded pairs.
        for _ in 0..10 {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            backend.forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
            backend.forward_batch(&mut ctx, &weights, &mut cache, &[1, 2, 3], 1, &config);
        }

        // Interleaved protocol (Bench 661/666 discipline): 2 warmup pairs
        // discarded, 5 measure pairs, alternating seq→batch / batch→seq
        // order within each pair, per-pair µs/token ratio as primary metric.
        // Gates are per-width: the amortization model 2(E+W)/(2E+W) caps the
        // width-2 ratio near ~1.3× (one saved wait out of two round trips),
        // while widths 5-8 amortize the single wait across many encodes (~2×).
        for width in [2usize, 5, 8] {
            let tokens: Vec<usize> = (0..width).map(|i| (i * 3 + 1) % 26).collect();
            let mut ratios: Vec<f64> = Vec::new();
            let mut last_seq = 0.0;
            let mut last_batch = 0.0;
            for pair in 0..7 {
                let (t_seq, t_batch) = if pair % 2 == 0 {
                    let a = time_width_step(&mut backend, &weights, &config, &tokens, true);
                    let b = time_width_step(&mut backend, &weights, &config, &tokens, false);
                    (a, b)
                } else {
                    let b = time_width_step(&mut backend, &weights, &config, &tokens, false);
                    let a = time_width_step(&mut backend, &weights, &config, &tokens, true);
                    (a, b)
                };
                if pair >= 2 {
                    ratios.push(t_seq / t_batch);
                    last_seq = t_seq;
                    last_batch = t_batch;
                }
            }
            ratios.sort_by(|a, b| a.total_cmp(b));
            let median = ratios[ratios.len() / 2];
            eprintln!(
                "width {width}: seq {last_seq:.1} µs/token vs batch {last_batch:.1} µs/token — median speedup {median:.2}× (ratios: {ratios:?})"
            );
            #[cfg(not(debug_assertions))]
            {
                let floor = if width <= 2 { 1.1 } else { 1.5 };
                assert!(
                    median > floor,
                    "batched forward should beat sequential at width {width} \
                     (median {median:.2}×, floor {floor})"
                );
            }
        }
    }

    /// Time one width-N step (sequential singles or one forward_batch),
    /// returning µs/token. Fresh CPU context/cache each iteration; GPU cache
    /// slots below seq_len are rewritten before being read (stale rows beyond
    /// seq_len are never read), so buffer reuse across iterations is correct.
    fn time_width_step(
        backend: &mut GpuBackend,
        weights: &TransformerWeights,
        config: &Config,
        tokens: &[usize],
        sequential: bool,
    ) -> f64 {
        const ITERS: usize = 50;
        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            let mut ctx = ForwardContext::new(config);
            let mut cache = MultiLayerKVCache::new(config);
            if sequential {
                for (i, &tok) in tokens.iter().enumerate() {
                    backend.forward(&mut ctx, weights, &mut cache, tok, i, config);
                }
            } else {
                backend.forward_batch(&mut ctx, weights, &mut cache, tokens, 0, config);
            }
        }
        start.elapsed().as_micros() as f64 / (ITERS * tokens.len()) as f64
    }

    // ── Plan 176: Latency Benchmarks ────────────────────────────

    #[test]
    fn bench_gpu_forward_latency_vs_cpu() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // CPU warm-up
        for _ in 0..100 {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        }

        // CPU timed
        let cpu_elapsed = {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                let mut ctx = ForwardContext::new(&config);
                let mut cache = MultiLayerKVCache::new(&config);
                transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
            }
            start.elapsed()
        };
        let cpu_us_per_token = cpu_elapsed.as_micros() as f64 / 1000.0;

        // GPU
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }

        // GPU warm-up
        for _ in 0..100 {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            backend.forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        }

        // GPU timed
        let gpu_elapsed = {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                let mut ctx = ForwardContext::new(&config);
                let mut cache = MultiLayerKVCache::new(&config);
                backend.forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
            }
            start.elapsed()
        };
        let gpu_us_per_token = gpu_elapsed.as_micros() as f64 / 1000.0;
        let speedup = cpu_us_per_token / gpu_us_per_token;

        eprintln!(
            "CPU: {cpu_us_per_token:.1} µs/token, GPU: {gpu_us_per_token:.1} µs/token, GPU speedup: {speedup:.2}×"
        );
    }

    #[test]
    fn bench_compilation_time() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let init_elapsed = {
            let start = std::time::Instant::now();
            let Ok(b) = GpuBackend::new() else { return };
            (start.elapsed(), b)
        };
        let mut backend = init_elapsed.1;

        let compile_elapsed = {
            let start = std::time::Instant::now();
            if backend.compile(&weights, &config).is_err() {
                return;
            }
            start.elapsed()
        };

        eprintln!(
            "GPU init: {} ms, GPU compile: {} ms",
            init_elapsed.0.as_millis(),
            compile_elapsed.as_millis()
        );
    }

    #[test]
    fn bench_tier_up_latency() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };

        let tier_up_ms = {
            let start = std::time::Instant::now();
            if backend.compile(&weights, &config).is_err() {
                return;
            }
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            backend.forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
            start.elapsed().as_millis()
        };

        eprintln!("Tier-up latency (compile + first forward): {tier_up_ms} ms");
    }
}

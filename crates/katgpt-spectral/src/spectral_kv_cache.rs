//! SpectralQuant KV cache with per-dim variable-bit packing.
//!
//! Stores K and V tensors in compressed format:
//! - Semantic dimensions (top d_eff): per-dim variable-bit codebooks
//! - Tail dimensions: uniform b_low-bit codebook
//!
//! Each KV vector: normalize → rotate → quantize → variable-bit pack.
//! Reconstruction: unpack → dequantize → inverse rotate → rescale.

use super::spectral::{
    BitAllocator, LloydMaxQuantizer, generate_selective_qjl_signs, waterfill_bits,
};
use super::types::{
    LloydMaxCodebook, SpectralQuantCalibration, SpectralQuantKVCacheConfig, SpectralQuantLayer,
    WaterfillAllocation,
};
use katgpt_core::simd::simd_scale_inplace;
use rayon::prelude::*;

/// Compressed KV cache using SpectralQuant quantization.
///
/// Two-regime storage:
/// - Semantic (first d_eff dims after rotation): variable-bit packed indices
/// - Tail (remaining dims): uniform low-bit packed indices
///
/// Zero-alloc hot path via scratch buffers.
pub struct SpectralQuantKVCache {
    // ── Vec fields first (24 bytes, 8-byte aligned) ──
    /// Per-layer calibration + codebooks.
    pub layers: Vec<SpectralQuantLayer>,
    /// Packed key indices: flat `[n_layers * max_seq_len * key_packed_len]`.
    /// Layer-major layout: element `(layer, pos)` is at
    /// `(layer * max_seq_len + pos) * key_packed_len`.
    key_indices: Vec<u8>,
    /// Per-position key norms (for reconstruction): `[n_layers * max_seq_len]`.
    key_norms: Vec<f32>,
    /// Packed value indices: flat `[n_layers * max_seq_len * val_packed_len]`.
    val_indices: Vec<u8>,
    /// Per-position value norms: `[n_layers * max_seq_len]`.
    val_norms: Vec<f32>,
    // ── Scratch buffers (zero-alloc hot path) ──
    scratch_normalized: Vec<f32>,
    scratch_rotated: Vec<f32>,
    scratch_unrotated: Vec<f32>,
    /// Combined index scratch used by both store (semantic + tail written
    /// directly here) and dequant (`unpack_variable_bits` output). Replaces
    /// the previous three-buffer split (`scratch_semantic_indices`,
    /// `scratch_tail_indices`, `scratch_all_indices`) which forced two extra
    /// `copy_from_slice` calls per store_vector.
    scratch_all_indices: Vec<u8>,
    // ── usize fields (8-byte aligned, no padding between them) ──
    /// Current write position.
    pos: usize,
    /// Number of layers (retained for capacity reasoning; reset uses flat fills).
    #[allow(dead_code)]
    n_layers: usize,
    kv_dim: usize,
    max_seq_len: usize,
    /// Packed bytes per key position: conservative `kv_dim` (1 byte/dim).
    key_packed_len: usize,
    /// Packed bytes per value position: conservative `kv_dim` (1 byte/dim).
    val_packed_len: usize,
}

/// Per-thread scratch buffers for parallel dequantize operations.
///
/// Created once per rayon worker thread via `map_init`, avoiding
/// contention on [`SpectralQuantKVCache`]'s internal scratch buffers.
/// Enables `&self` parallel dequantize without requiring `&mut self`.
pub struct DequantizeScratch {
    // Vec fields first (8-byte aligned), then smaller types
    all_indices: Vec<u8>,
    rotated: Vec<f32>,
    unrotated: Vec<f32>,
}

impl DequantizeScratch {
    /// Create scratch buffers sized for `kv_dim` dimensions.
    pub fn new(kv_dim: usize) -> Self {
        Self {
            all_indices: vec![0u8; kv_dim],
            rotated: vec![0.0f32; kv_dim],
            unrotated: vec![0.0f32; kv_dim],
        }
    }
}

/// Discriminant for whether `store_vector` writes to key or value storage.
///
/// `#[repr(u8)]` gives a 1-byte discriminant with a predictable bit pattern,
/// which lets the optimizer turn the `match` into a branch-friendly
/// conditional-select / conditional-move instead of a jump table.
#[repr(u8)]
enum StoreTarget {
    Key,
    Value,
}

impl SpectralQuantKVCache {
    /// Byte offset of key data for `(layer, pos)` in the flat `key_indices`.
    #[inline]
    fn key_off(&self, layer: usize, pos: usize) -> usize {
        (layer * self.max_seq_len + pos) * self.key_packed_len
    }

    /// Byte offset of value data for `(layer, pos)` in the flat `val_indices`.
    #[inline]
    fn val_off(&self, layer: usize, pos: usize) -> usize {
        (layer * self.max_seq_len + pos) * self.val_packed_len
    }

    /// Flat index of norm for `(layer, pos)`.
    #[inline]
    fn norm_idx(&self, layer: usize, pos: usize) -> usize {
        layer * self.max_seq_len + pos
    }

    /// Create from pre-computed calibration data and config.
    ///
    /// **Prefer `from_keys()` for new code** — it auto-calibrates from actual data
    /// and cannot accidentally use identity eigenvectors. This constructor remains
    /// available for cases where calibration is loaded from serialized model weights
    /// (pre-computed during a prior offline calibration step).
    ///
    /// Fits Lloyd-Max codebooks by generating synthetic rotated data from the
    /// eigenvalue distribution: in the spectral basis, dimension `i` has variance `λ_i`,
    /// so we sample `N(0, λ_i)` to build representative codebooks.
    pub fn from_calibration(
        config: &SpectralQuantKVCacheConfig,
        key_calibrations: &[SpectralQuantCalibration],
        val_calibrations: &[SpectralQuantCalibration],
    ) -> Self {
        let n_layers = config.n_layers;
        let kv_dim = config.kv_dim;
        let max_seq_len = config.max_seq_len;

        let mut layers: Vec<SpectralQuantLayer> =
            key_calibrations
                .iter()
                .zip(val_calibrations.iter())
                .enumerate()
                .map(|(layer_idx, (key_cal, _val_cal))| {
                    let d_eff = (key_cal.d_eff.ceil() as usize).max(1).min(kv_dim);
                    let head_dim = key_cal.head_dim;

                    let allocator = BitAllocator::new(config.min_tail_bits, config.max_bits);
                    let (b_high, b_low) =
                        allocator.allocate(key_cal.d_eff, config.avg_bits, head_dim);

                    let qjl_signs = generate_selective_qjl_signs(
                        config.qjl_dim,
                        d_eff,
                        config.seed.wrapping_add(layer_idx as u64 * 100),
                    );

                    let (
                        semantic_bits_per_dim,
                        per_dim_codebooks,
                        semantic_codebook,
                        _waterfill_alloc,
                    ) = if config.use_water_fill && b_high > 0 {
                        let first_ev: Vec<f64> = key_cal
                            .eigenvalues
                            .iter()
                            .take(d_eff)
                            .map(|&x| x as f64)
                            .collect();
                        let total_semantic = b_high as usize * d_eff;
                        let bits = waterfill_bits(
                            &first_ev,
                            total_semantic,
                            config.wf_min_bits,
                            Some(config.wf_max_bits),
                        );
                        let bits_u8 = bits.to_vec();
                        let alloc = WaterfillAllocation {
                            use_water_fill: true,
                            eigenvalues: key_cal.eigenvalues.iter().take(d_eff).copied().collect(),
                            bits_per_dim: bits_u8.clone(),
                            d_eff,
                            total_semantic_bits: total_semantic,
                            min_bits: config.wf_min_bits,
                            max_bits: Some(config.wf_max_bits),
                            formula_version: 2,
                        };
                        let codebooks: Vec<LloydMaxCodebook> = bits_u8
                            .iter()
                            .map(|&b| LloydMaxCodebook {
                                centroids: vec![0.0; 1 << b],
                                n_bits: b,
                            })
                            .collect();
                        (Some(bits_u8), Some(codebooks), None, Some(alloc))
                    } else {
                        let codebook = LloydMaxCodebook {
                            centroids: vec![0.0; 1 << b_high],
                            n_bits: b_high,
                        };
                        (None, None, Some(codebook), None)
                    };

                    let tail_codebook = LloydMaxCodebook {
                        centroids: vec![0.0; 1 << b_low.max(1)],
                        n_bits: b_low.max(1),
                    };

                    // Precompute the full [kv_dim] bits-per-dim array once.
                    // Per-token store/dequantize paths read this directly instead of
                    // rebuilding it every call.
                    let mut packed_bits = vec![0u8; kv_dim];
                    if let Some(ref bits) = semantic_bits_per_dim {
                        let copy_len = d_eff.min(bits.len());
                        packed_bits[..copy_len].copy_from_slice(&bits[..copy_len]);
                    } else {
                        packed_bits[..d_eff].fill(b_high);
                    }
                    packed_bits[d_eff..kv_dim].fill(b_low.max(1));

                    SpectralQuantLayer {
                        calibration: key_cal.clone(),
                        qjl_signs,
                        d_eff,
                        b_high,
                        b_low,
                        semantic_bits_per_dim,
                        per_dim_semantic_codebooks: per_dim_codebooks,
                        semantic_codebook,
                        tail_codebook,
                        packed_bits,
                    }
                })
                .collect();

        // Detect identity eigenvectors (no real calibration) and substitute random rotation.
        // When no calibration data is available, SQ degrades gracefully to TQ-quality
        // by using a random rotation instead of the degenerate identity rotation.
        // Identity rotation = no decorrelation = all coordinates in narrow range = poor quantization.
        // Random rotation spreads coordinates ~ N(0, 1/d), giving codebooks much better coverage.
        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            if is_identity_matrix(&layer.calibration.eigenvectors, layer.calibration.head_dim) {
                let random_rot = generate_random_rotation(
                    layer.calibration.head_dim,
                    config.seed.wrapping_add(layer_idx as u64 * 7919),
                );
                layer.calibration.eigenvectors = random_rot;
            }
        }

        // Fit codebooks from the actual normalize→rotate→quantize pipeline.
        //
        // The quantize path is: normalize(x) → V^T * x_norm → quantize.
        // So codebooks must be fitted for data that has been normalized to unit norm
        // and rotated by eigenvectors. Generating N(0, λ_i) was wrong because:
        //   1. Data is normalized BEFORE rotation, so ‖x_rotated‖ = 1, not N(0, λ_i)
        //   2. The N(0, λ_i) assumption only holds for unnormalized data in expectation
        //
        // Fix: generate random unit-norm vectors, rotate by V^T, then fit codebooks.
        // This matches the Python pipeline exactly (K_normed @ PiT → quantize_regime).
        let n_synthetic = config.calibration_samples.max(256);
        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            let d_eff = layer.d_eff;
            let head_dim = layer.calibration.head_dim;
            let b_high = layer.b_high;
            let b_low = layer.b_low;
            let mut rng =
                katgpt_core::types::Rng::new(config.seed.wrapping_add(layer_idx as u64 * 31));
            let eigenvectors = &layer.calibration.eigenvectors;

            // Generate synthetic data matching the actual pipeline:
            //   1. Random vector ~ N(0, I)
            //   2. Normalize to unit norm (‖x‖ = 1)
            //   3. Rotate by V^T (eigenvector transpose)
            //
            // Pre-allocate scratch buffers outside the loop to avoid per-iteration alloc.
            let mut scratch_x = vec![0.0f32; head_dim];
            let mut scratch_rotated = vec![0.0f32; head_dim];
            // Flat storage: a single Vec<f32> of length n_synthetic * head_dim,
            // row-major. Each synthetic sample occupies a contiguous
            // head_dim-element row. Avoids n_synthetic separate allocations
            // (one per Vec<Vec<f32>> inner) and the per-iteration
            // `scratch_rotated.clone()` memcpy + alloc.
            let mut synthetic_rotated: Vec<f32> = Vec::with_capacity(n_synthetic * head_dim);
            for _ in 0..n_synthetic {
                // Step 1: random vector
                for v in scratch_x.iter_mut() {
                    *v = rng.normal();
                }
                // Step 2: normalize to unit norm
                let norm = katgpt_core::simd::simd_sum_sq(&scratch_x, head_dim)
                    .sqrt()
                    .max(1e-8);
                katgpt_core::simd::simd_scale_inplace(&mut scratch_x, 1.0 / norm);
                // Step 3: rotate by V^T — output[j] = Σ_i x[i] * V[i*head_dim+j]
                // Same transpose-and-accumulate pattern as SpectralRotation::rotate().
                scratch_rotated.fill(0.0);
                for i in 0..head_dim {
                    let xi = scratch_x[i];
                    let row = &eigenvectors[i * head_dim..i * head_dim + head_dim];
                    for j in 0..head_dim {
                        scratch_rotated[j] += row[j] * xi;
                    }
                }
                // Append the rotated sample to the flat buffer (one memcpy,
                // zero allocations — capacity is pre-reserved).
                synthetic_rotated.extend_from_slice(&scratch_rotated);
            }

            // Fit tail codebook from tail dims (d_eff..head_dim).
            // Pre-allocate the flat tail buffer up front: each synthetic sample
            // contributes `(head_dim − d_eff)` tail values.
            let tail_len = head_dim.saturating_sub(d_eff);
            let mut tail_data = Vec::with_capacity(n_synthetic * tail_len);
            for s in synthetic_rotated.chunks(head_dim) {
                tail_data.extend(s.iter().skip(d_eff).copied());
            }
            let mut tail_q = LloydMaxQuantizer::new(
                b_low.max(1),
                config.lloyd_max_iter,
                config.seed.wrapping_add(layer_idx as u64 * 51 + 1),
            );
            tail_q.fit(&tail_data);
            layer.tail_codebook.centroids = tail_q.centroids().to_vec();

            // Fit semantic codebook(s) from semantic dims (0..d_eff)
            if let Some(ref mut cb) = layer.semantic_codebook {
                // v1: shared semantic codebook — all semantic dims pooled.
                // Pre-allocate capacity = n_synthetic * d_eff.
                let mut semantic_data = Vec::with_capacity(n_synthetic * d_eff);
                for s in synthetic_rotated.chunks(head_dim) {
                    semantic_data.extend(s.iter().take(d_eff).copied());
                }
                let mut sem_q = LloydMaxQuantizer::new(
                    b_high.max(1),
                    config.lloyd_max_iter,
                    config.seed.wrapping_add(layer_idx as u64 * 51 + 2),
                );
                sem_q.fit(&semantic_data);
                cb.centroids = sem_q.centroids().to_vec();
            } else if let Some(ref mut per_dim) = layer.per_dim_semantic_codebooks {
                // v2: per-dim semantic codebooks. Reuse one `dim_data` scratch
                // buffer across dims (clear + repopulate) instead of reallocating
                // `n_synthetic` f32 per dim.
                let bits = layer.semantic_bits_per_dim.as_ref();
                let mut dim_data = Vec::with_capacity(n_synthetic);
                for (dim, cb) in per_dim.iter_mut().enumerate() {
                    dim_data.clear();
                    // Strided iteration over the flat buffer: element `dim` of
                    // each head_dim-wide row.
                    dim_data.extend(synthetic_rotated.chunks(head_dim).map(|row| row[dim]));
                    let bits_for_dim = bits
                        .and_then(|b| b.get(dim).copied())
                        .unwrap_or(b_high)
                        .max(1);
                    let mut q = LloydMaxQuantizer::new(
                        bits_for_dim,
                        config.lloyd_max_iter,
                        config.seed.wrapping_add((dim + 10) as u64),
                    );
                    q.fit(&dim_data);
                    cb.centroids = q.centroids().to_vec();
                }
            }
        }

        // Packed size per position = ceil(sum(packed_bits) / 8).
        // Compute the per-layer max once and use it for both key and value
        // storage. The previous conservative bound (kv_dim bytes/pos) was
        // ~2-4x the actual packed length for typical (b_high=4, b_low=2, d_eff=kv_dim/2)
        // layouts, wasting multiple MB on long-context workloads.
        let max_packed = layers
            .iter()
            .map(|l| {
                l.packed_bits
                    .iter()
                    .map(|&b| b as usize)
                    .sum::<usize>()
                    .div_ceil(8)
            })
            .max()
            .unwrap_or(kv_dim)
            .max(1);

        Self {
            layers,
            key_indices: vec![0u8; n_layers * max_seq_len * max_packed],
            key_norms: vec![0.0f32; n_layers * max_seq_len],
            val_indices: vec![0u8; n_layers * max_seq_len * max_packed],
            val_norms: vec![0.0f32; n_layers * max_seq_len],
            pos: 0,
            n_layers,
            kv_dim,
            max_seq_len,
            key_packed_len: max_packed,
            val_packed_len: max_packed,
            scratch_normalized: vec![0.0f32; kv_dim],
            scratch_rotated: vec![0.0f32; kv_dim],
            scratch_unrotated: vec![0.0f32; kv_dim],
            scratch_all_indices: vec![0u8; kv_dim],
        }
    }

    /// Create from actual key/value data with auto-calibration.
    ///
    /// This is the **recommended** constructor — it calibrates the eigenbasis from
    /// the provided key samples, ensuring SpectralQuant's spectral advantage is
    /// fully utilized. Unlike `from_calibration()`, there is no risk of accidentally
    /// passing identity eigenvectors that would degrade SQ to random rotation quality.
    ///
    /// # Arguments
    /// * `config` — SpectralQuant config (avg_bits, min_tail_bits, etc.)
    /// * `key_samples` — Representative key vectors for calibration (typically from prefill)
    /// * `val_samples` — Representative value vectors for calibration (typically from prefill)
    pub fn from_keys(
        config: &SpectralQuantKVCacheConfig,
        key_samples: &[Vec<f32>],
        val_samples: &[Vec<f32>],
    ) -> Self {
        use super::spectral::calibrate_eigenbasis;

        let key_result = calibrate_eigenbasis(key_samples, config.kv_dim);
        let val_result = calibrate_eigenbasis(val_samples, config.kv_dim);

        let key_cal = SpectralQuantCalibration {
            eigenvectors: key_result.eigenvectors,
            eigenvalues: key_result.eigenvalues,
            d_eff: key_result.d_eff,
            spectral_gap: key_result.spectral_gap,
            var_95: key_result.var_95,
            var_99: key_result.var_99,
            n_samples: key_result.n_samples,
            head_dim: key_result.head_dim,
        };
        let val_cal = SpectralQuantCalibration {
            eigenvectors: val_result.eigenvectors,
            eigenvalues: val_result.eigenvalues,
            d_eff: val_result.d_eff,
            spectral_gap: val_result.spectral_gap,
            var_95: val_result.var_95,
            var_99: val_result.var_99,
            n_samples: val_result.n_samples,
            head_dim: val_result.head_dim,
        };

        let key_calibrations = vec![key_cal; config.n_layers];
        let val_calibrations = vec![val_cal; config.n_layers];

        Self::from_calibration(config, &key_calibrations, &val_calibrations)
    }

    /// Quantize and store a key vector at given layer and position.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.store_vector(layer, pos, key, StoreTarget::Key);
    }

    /// Quantize and store a value vector at given layer and position.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        self.store_vector(layer, pos, value, StoreTarget::Value);
    }

    /// Shared quantize-and-store logic for both keys and values.
    ///
    /// Pipeline: norm → normalize → rotate(V^T) → quantize(semantic+tail) → pack.
    fn store_vector(&mut self, layer: usize, pos: usize, vec: &[f32], target: StoreTarget) {
        debug_assert_eq!(vec.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let d_eff = layer_state.d_eff;
        // Compute flat-storage offsets before any mutable borrows below.
        let ni = self.norm_idx(layer, pos);
        let key_off = self.key_off(layer, pos);
        let val_off = self.val_off(layer, pos);
        let key_packed_len = self.key_packed_len;
        let val_packed_len = self.val_packed_len;

        // Compute norm
        let norm = simd_norm(vec);
        if norm < 1e-8 {
            // Zero norm → zero norm slot + zero packed data
            match target {
                StoreTarget::Key => {
                    self.key_norms[ni] = 0.0;
                    self.key_indices[key_off..key_off + key_packed_len].fill(0);
                }
                StoreTarget::Value => {
                    self.val_norms[ni] = 0.0;
                    self.val_indices[val_off..val_off + val_packed_len].fill(0);
                }
            }
            return;
        }

        // Store norm
        match target {
            StoreTarget::Key => self.key_norms[ni] = norm,
            StoreTarget::Value => self.val_norms[ni] = norm,
        }

        // Normalize into scratch buffer
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..vec.len()].copy_from_slice(vec);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // Rotate using cached per-layer eigenvectors (no clone)
        let eigenvectors = &layer_state.calibration.eigenvectors;
        let head_dim = layer_state.calibration.head_dim;
        rotate_into(
            eigenvectors,
            head_dim,
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        // Quantize semantic dims directly into the combined scratch buffer
        // (`scratch_all_indices[..d_eff]`), avoiding an intermediate buffer + copy.
        let all_indices = &mut self.scratch_all_indices;
        if let Some(cb) = &layer_state.semantic_codebook {
            // v1: shared semantic codebook
            for (i, slot) in all_indices.iter_mut().enumerate().take(d_eff) {
                *slot = quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            // v2: per-dim codebooks
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                all_indices[i] = quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
            }
        }

        // Quantize tail dims directly into the combined scratch buffer tail.
        let tail_cb = &layer_state.tail_codebook;
        let tail_len = self.kv_dim - d_eff;
        for i in 0..tail_len {
            all_indices[i + d_eff] =
                quantize_to_idx(self.scratch_rotated[i + d_eff], &tail_cb.centroids);
        }

        // Pack variable bits into flat storage using precomputed packed_bits.
        // The slot is pre-zeroed so trailing bytes beyond the actual packed
        // length are clean for subsequent unpack reads.
        match target {
            StoreTarget::Key => {
                let slot = &mut self.key_indices[key_off..key_off + key_packed_len];
                slot.fill(0);
                pack_variable_bits(&all_indices[..self.kv_dim], &layer_state.packed_bits, slot);
            }
            StoreTarget::Value => {
                let slot = &mut self.val_indices[val_off..val_off + val_packed_len];
                slot.fill(0);
                pack_variable_bits(&all_indices[..self.kv_dim], &layer_state.packed_bits, slot);
            }
        }
    }

    /// Dequantize a key at position into a new vector.
    pub fn dequantize_key(&self, layer: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.kv_dim];
        // We need &mut self for scratch buffers, so use a temporary approach
        // by reconstructing directly without scratch
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[self.norm_idx(layer, pos)];
        if norm < 1e-8 {
            return out;
        }

        let d_eff = layer_state.d_eff;

        let mut all_indices = vec![0u8; self.kv_dim];
        let off = self.key_off(layer, pos);
        unpack_variable_bits(
            &self.key_indices[off..off + self.key_packed_len],
            &layer_state.packed_bits,
            self.kv_dim,
            &mut all_indices,
        );

        let mut rotated = vec![0.0f32; self.kv_dim];
        if let Some(cb) = &layer_state.semantic_codebook {
            for i in 0..d_eff {
                rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            let limit = d_eff.min(per_dim.len());
            for i in 0..limit {
                rotated[i] = dequantize_idx(all_indices[i], &per_dim[i].centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for i in d_eff..self.kv_dim {
            rotated[i] = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let eigenvectors = &layer_state.calibration.eigenvectors;
        let head_dim = layer_state.calibration.head_dim;
        let mut normalized = vec![0.0f32; self.kv_dim];
        unrotate_into(eigenvectors, head_dim, &rotated, &mut normalized);

        for v in &mut normalized {
            *v *= norm;
        }
        out.copy_from_slice(&normalized);
        out
    }

    /// Dequantize a value at position into a new vector.
    pub fn dequantize_value(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[self.norm_idx(layer, pos)];
        if norm < 1e-8 {
            return vec![0.0f32; self.kv_dim];
        }

        let d_eff = layer_state.d_eff;

        let mut all_indices = vec![0u8; self.kv_dim];
        let off = self.val_off(layer, pos);
        unpack_variable_bits(
            &self.val_indices[off..off + self.val_packed_len],
            &layer_state.packed_bits,
            self.kv_dim,
            &mut all_indices,
        );

        let mut rotated = vec![0.0f32; self.kv_dim];
        if let Some(cb) = &layer_state.semantic_codebook {
            for i in 0..d_eff {
                rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            let limit = d_eff.min(per_dim.len());
            for i in 0..limit {
                rotated[i] = dequantize_idx(all_indices[i], &per_dim[i].centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for i in d_eff..self.kv_dim {
            rotated[i] = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let eigenvectors = &layer_state.calibration.eigenvectors;
        let head_dim = layer_state.calibration.head_dim;
        let mut normalized = vec![0.0f32; self.kv_dim];
        unrotate_into(eigenvectors, head_dim, &rotated, &mut normalized);

        normalized.iter().map(|x| x * norm).collect()
    }

    /// Dequantize key into pre-allocated buffer. Zero-alloc hot path.
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    /// Reconstruction: unpack → dequantize → inverse rotate → scale by norm.
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let ni = self.norm_idx(layer, pos);
        let off = self.key_off(layer, pos);
        let packed_len = self.key_packed_len;
        let kv_dim = self.kv_dim;
        let norm = self.key_norms[ni];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let d_eff = layer_state.d_eff;

        // Unpack variable bits into scratch using precomputed packed_bits
        let all_indices = &mut self.scratch_all_indices;
        unpack_variable_bits(
            &self.key_indices[off..off + packed_len],
            &layer_state.packed_bits,
            kv_dim,
            all_indices,
        );

        // Dequantize into scratch_rotated
        if let Some(cb) = &layer_state.semantic_codebook {
            for (i, c) in self.scratch_rotated.iter_mut().enumerate().take(d_eff) {
                *c = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                self.scratch_rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for (i, r) in self.scratch_rotated.iter_mut().enumerate().skip(d_eff) {
            *r = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        // Inverse rotate (no clone)
        let eigenvectors = &layer_state.calibration.eigenvectors;
        let head_dim = layer_state.calibration.head_dim;
        unrotate_into(
            eigenvectors,
            head_dim,
            &self.scratch_rotated,
            &mut self.scratch_unrotated,
        );

        // Scale by norm → output
        out.copy_from_slice(&self.scratch_unrotated);
        simd_scale_inplace(out, norm);
    }

    /// Dequantize value into pre-allocated buffer. Zero-alloc hot path.
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    /// Reconstruction: unpack → dequantize → inverse rotate → scale by norm.
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let ni = self.norm_idx(layer, pos);
        let off = self.val_off(layer, pos);
        let packed_len = self.val_packed_len;
        let kv_dim = self.kv_dim;
        let norm = self.val_norms[ni];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let d_eff = layer_state.d_eff;

        let all_indices = &mut self.scratch_all_indices;
        unpack_variable_bits(
            &self.val_indices[off..off + packed_len],
            &layer_state.packed_bits,
            kv_dim,
            all_indices,
        );

        if let Some(cb) = &layer_state.semantic_codebook {
            for (i, r) in self.scratch_rotated.iter_mut().enumerate().take(d_eff) {
                *r = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                self.scratch_rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for (i, r) in self.scratch_rotated.iter_mut().enumerate().skip(d_eff) {
            *r = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let eigenvectors = &layer_state.calibration.eigenvectors;
        let head_dim = layer_state.calibration.head_dim;
        unrotate_into(
            eigenvectors,
            head_dim,
            &self.scratch_rotated,
            &mut self.scratch_unrotated,
        );

        out.copy_from_slice(&self.scratch_unrotated);
        simd_scale_inplace(out, norm);
    }

    /// Reset cache for a new sequence.
    ///
    /// All flat storage buffers are cleared via `fill(0)` / `fill(0.0)` —
    /// a single contiguous memset per buffer instead of per-position loops.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.key_indices.fill(0);
        self.val_indices.fill(0);
        self.key_norms.fill(0.0);
        self.val_norms.fill(0.0);
    }

    /// Current write position.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set the current write position.
    #[inline]
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// KV dimension.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Compression ratio vs f32 uncompressed (32 bits per coordinate).
    pub fn compression_ratio(&self) -> f32 {
        if self.layers.is_empty() {
            return 1.0;
        }
        let original = self.kv_dim as f32 * 32.0;
        let layer0 = &self.layers[0];
        let used = layer0.d_eff as f32 * layer0.b_high as f32
            + (self.kv_dim - layer0.d_eff) as f32 * layer0.b_low.max(1) as f32;
        if used < 1.0 {
            return 1.0;
        }
        original / used
    }

    // ── Thread-safe dequantize with external scratch (for rayon) ────────

    /// Dequantize key using external scratch buffers (thread-safe).
    ///
    /// Same logic as [`dequantize_key_into`](Self::dequantize_key_into) but uses
    /// caller-provided [`DequantizeScratch`], enabling parallel dequantize via
    /// rayon's `map_init`. Takes `&self` instead of `&mut self` — safe for
    /// concurrent reads from multiple threads.
    pub fn dequantize_key_into_with_scratch(
        &self,
        layer: usize,
        pos: usize,
        scratch: &mut DequantizeScratch,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[self.norm_idx(layer, pos)];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let d_eff = layer_state.d_eff;

        let all_indices = &mut scratch.all_indices;
        let off = self.key_off(layer, pos);
        unpack_variable_bits(
            &self.key_indices[off..off + self.key_packed_len],
            &layer_state.packed_bits,
            self.kv_dim,
            all_indices,
        );

        if let Some(cb) = &layer_state.semantic_codebook {
            for (i, c) in scratch.rotated.iter_mut().enumerate().take(d_eff) {
                *c = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                scratch.rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for (i, r) in scratch.rotated.iter_mut().enumerate().skip(d_eff) {
            *r = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let eigenvectors = &layer_state.calibration.eigenvectors;
        let head_dim = layer_state.calibration.head_dim;
        unrotate_into(
            eigenvectors,
            head_dim,
            &scratch.rotated,
            &mut scratch.unrotated,
        );

        out.copy_from_slice(&scratch.unrotated);
        simd_scale_inplace(out, norm);
    }

    /// Dequantize value using external scratch buffers (thread-safe).
    ///
    /// Same logic as [`dequantize_value_into`](Self::dequantize_value_into) but uses
    /// caller-provided [`DequantizeScratch`], enabling parallel dequantize via
    /// rayon's `map_init`. Takes `&self` instead of `&mut self` — safe for
    /// concurrent reads from multiple threads.
    pub fn dequantize_value_into_with_scratch(
        &self,
        layer: usize,
        pos: usize,
        scratch: &mut DequantizeScratch,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[self.norm_idx(layer, pos)];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let d_eff = layer_state.d_eff;

        let all_indices = &mut scratch.all_indices;
        let off = self.val_off(layer, pos);
        unpack_variable_bits(
            &self.val_indices[off..off + self.val_packed_len],
            &layer_state.packed_bits,
            self.kv_dim,
            all_indices,
        );

        if let Some(cb) = &layer_state.semantic_codebook {
            for (i, r) in scratch.rotated.iter_mut().enumerate().take(d_eff) {
                *r = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                scratch.rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for (i, r) in scratch.rotated.iter_mut().enumerate().skip(d_eff) {
            *r = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let eigenvectors = &layer_state.calibration.eigenvectors;
        let head_dim = layer_state.calibration.head_dim;
        unrotate_into(
            eigenvectors,
            head_dim,
            &scratch.rotated,
            &mut scratch.unrotated,
        );

        out.copy_from_slice(&scratch.unrotated);
        simd_scale_inplace(out, norm);
    }

    // ── Parallel batch dequantize ──────────────────────────────────

    /// Parallel batch dequantize of all key positions `[0..=pos]` using rayon.
    ///
    /// Each rayon worker thread gets its own [`DequantizeScratch`] via `map_init`,
    /// eliminating contention on the cache's internal scratch buffers.
    /// Takes `&self` — safe for concurrent reads from multiple threads.
    ///
    /// Falls back to sequential for small batches (`n <= threshold`),
    /// where rayon overhead outweighs parallelism benefit.
    ///
    /// **Allocates** a `Vec<f32>` of size `(pos+1) * kv_dim` on every call.
    /// For hot paths that call this repeatedly, prefer
    /// [`par_dequantize_keys_flat_into`](Self::par_dequantize_keys_flat_into)
    /// which writes into a caller-reused buffer.
    pub fn par_dequantize_keys_flat(&self, layer: usize, pos: usize, threshold: usize) -> Vec<f32> {
        let n = pos + 1;
        if n == 0 {
            return Vec::new();
        }
        let mut flat = vec![0.0f32; n * self.kv_dim];
        let mut scratch = DequantizeScratch::new(self.kv_dim);
        self.par_dequantize_keys_flat_into(layer, pos, threshold, &mut flat, &mut scratch);
        flat
    }

    /// Zero-allocation parallel batch dequantize of keys `[0..=pos]`.
    ///
    /// Writes directly into `out`, which must have length `>= (pos + 1) * kv_dim`.
    /// Only `out[..(pos+1)*kv_dim]` is written; the tail is untouched.
    ///
    /// `scratch` is used only for the sequential fallback (`n <= threshold`).
    /// For the parallel path, per-worker scratches are created via rayon's
    /// `for_each_init` (one per worker thread, reused across items in the call).
    ///
    /// This is the hot-path variant — callers should reuse `out` and `scratch`
    /// across calls to avoid repeated allocation. The returning-`Vec` variant
    /// [`par_dequantize_keys_flat`](Self::par_dequantize_keys_flat) delegates
    /// here.
    pub fn par_dequantize_keys_flat_into(
        &self,
        layer: usize,
        pos: usize,
        threshold: usize,
        out: &mut [f32],
        scratch: &mut DequantizeScratch,
    ) {
        let kv_dim = self.kv_dim;
        let n = pos + 1;
        if n == 0 {
            return;
        }
        debug_assert!(
            out.len() >= n * kv_dim,
            "out too small: {} < {}",
            out.len(),
            n * kv_dim
        );

        // Sequential fallback for small batches (rayon overhead > benefit).
        if n <= threshold {
            for t in 0..n {
                let row = &mut out[t * kv_dim..(t + 1) * kv_dim];
                self.dequantize_key_into_with_scratch(layer, t, scratch, row);
            }
            return;
        }

        // Parallel: write directly into disjoint rows of the caller's buffer.
        out[..n * kv_dim]
            .par_chunks_mut(kv_dim)
            .enumerate()
            .for_each_init(
                || DequantizeScratch::new(kv_dim),
                |s, (t, row)| {
                    self.dequantize_key_into_with_scratch(layer, t, s, row);
                },
            );
    }

    /// Parallel batch dequantize of all value positions `[0..=pos]` using rayon.
    ///
    /// Same pattern as [`par_dequantize_keys_flat`](Self::par_dequantize_keys_flat)
    /// but for value vectors. Takes `&self` — safe for concurrent reads.
    ///
    /// **Allocates** on every call; prefer
    /// [`par_dequantize_values_flat_into`](Self::par_dequantize_values_flat_into)
    /// for hot paths.
    pub fn par_dequantize_values_flat(
        &self,
        layer: usize,
        pos: usize,
        threshold: usize,
    ) -> Vec<f32> {
        let n = pos + 1;
        if n == 0 {
            return Vec::new();
        }
        let mut flat = vec![0.0f32; n * self.kv_dim];
        let mut scratch = DequantizeScratch::new(self.kv_dim);
        self.par_dequantize_values_flat_into(layer, pos, threshold, &mut flat, &mut scratch);
        flat
    }

    /// Zero-allocation parallel batch dequantize of values `[0..=pos]`.
    ///
    /// See [`par_dequantize_keys_flat_into`](Self::par_dequantize_keys_flat_into)
    /// for the contract — this is the value-vector counterpart.
    pub fn par_dequantize_values_flat_into(
        &self,
        layer: usize,
        pos: usize,
        threshold: usize,
        out: &mut [f32],
        scratch: &mut DequantizeScratch,
    ) {
        let kv_dim = self.kv_dim;
        let n = pos + 1;
        if n == 0 {
            return;
        }
        debug_assert!(
            out.len() >= n * kv_dim,
            "out too small: {} < {}",
            out.len(),
            n * kv_dim
        );

        if n <= threshold {
            for t in 0..n {
                let row = &mut out[t * kv_dim..(t + 1) * kv_dim];
                self.dequantize_value_into_with_scratch(layer, t, scratch, row);
            }
            return;
        }

        out[..n * kv_dim]
            .par_chunks_mut(kv_dim)
            .enumerate()
            .for_each_init(
                || DequantizeScratch::new(kv_dim),
                |s, (t, row)| {
                    self.dequantize_value_into_with_scratch(layer, t, s, row);
                },
            );
    }
}

impl katgpt_core::types::QuantizedKVCache for SpectralQuantKVCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.store_key(layer, pos, key);
    }

    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        self.store_value(layer, pos, value);
    }

    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_key_into(layer, pos, out);
    }

    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_value_into(layer, pos, out);
    }

    fn reset(&mut self) {
        self.reset();
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos()
    }

    fn set_pos(&mut self, pos: usize) {
        self.set_pos(pos);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Compute L2 norm of a vector.
/// Inline forward rotation: out = V^T @ x, using raw eigenvector slice.
/// Avoids allocating a `SpectralRotation` struct — zero-alloc for hot paths.
///
/// Uses transpose-and-accumulate for contiguous reads (V rows) and contiguous
/// writes to `out`, which is cache-friendly and auto-vectorizer-friendly.
#[inline]
fn rotate_into(eigenvectors: &[f32], head_dim: usize, x: &[f32], out: &mut [f32]) {
    out.fill(0.0);
    for i in 0..head_dim {
        let xi = x[i];
        let row = &eigenvectors[i * head_dim..i * head_dim + head_dim];
        for j in 0..head_dim {
            unsafe {
                *out.get_unchecked_mut(j) += *row.get_unchecked(j) * xi;
            }
        }
    }
}

/// Inline inverse rotation: out = V @ x, using raw eigenvector slice.
#[inline]
#[allow(clippy::needless_range_loop)]
fn unrotate_into(eigenvectors: &[f32], head_dim: usize, x: &[f32], out: &mut [f32]) {
    // out[i] = dot(eigenvectors row i, x) — row-major access, SIMD-friendly
    katgpt_core::simd::simd_matmul_rows(out, eigenvectors, x, head_dim, head_dim);
}

fn simd_norm(v: &[f32]) -> f32 {
    katgpt_core::simd::simd_sum_sq(v, v.len()).sqrt()
}

/// Check if a matrix is the identity matrix (diagonal 1s, off-diagonal 0s).
fn is_identity_matrix(mat: &[f32], dim: usize) -> bool {
    for i in 0..dim {
        for j in 0..dim {
            let expected = if i == j { 1.0f32 } else { 0.0f32 };
            if (mat[i * dim + j] - expected).abs() > 1e-4 {
                return false;
            }
        }
    }
    true
}

/// Generate a random orthogonal rotation matrix via Gram-Schmidt.
/// Same quality as TurboQuant's random rotation — used as fallback when
/// no real calibration data is available (identity eigenvectors).
fn generate_random_rotation(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = katgpt_core::types::Rng::new(seed);
    let mut mat = vec![0.0f32; dim * dim];
    for v in mat.iter_mut() {
        *v = rng.normal();
    }
    // Gram-Schmidt orthogonalization: for each column, subtract projections
    // onto all previous columns, then normalize to unit length.
    //
    // Inner loops use explicit indexed accumulation instead of `.map().sum()`
    // chains — the iterator adapters block LLVM's auto-vectorizer on the
    // dot-product / sum-of-squares, which dominate this routine at large dim.
    for col in 0..dim {
        for prev in 0..col {
            let mut dot = 0.0f32;
            for row in 0..dim {
                dot += mat[row * dim + col] * mat[row * dim + prev];
            }
            for row in 0..dim {
                mat[row * dim + col] -= dot * mat[row * dim + prev];
            }
        }
        let mut norm_sq = 0.0f32;
        for row in 0..dim {
            let v = mat[row * dim + col];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-8 {
            let inv = 1.0 / norm;
            for row in 0..dim {
                mat[row * dim + col] *= inv;
            }
        } else {
            // Degenerate column — set to basis vector
            mat[col * dim + col] = 1.0;
        }
    }
    mat
}

/// Find nearest centroid index for a value.
///
/// Uses binary search on the assumption that centroids are sorted ascending.
/// This gives O(log n) instead of O(n) for the hot-path quantize loop.
///
/// For tiny codebooks (≤8 centroids, the typical case for 1-3 bit quant),
/// a linear scan beats binary search: branch predictor handles the always-
/// ascending compares better than log n data-dependent mid branches, and
/// the loop body is small enough to unroll fully.
fn quantize_to_idx(value: f32, centroids: &[f32]) -> u8 {
    // Empty or single-centroid codebook: only one valid index.
    let n = centroids.len();
    if n <= 1 {
        return 0;
    }

    // Small codebook specialization: linear scan with branchless min-track.
    // Benchmarked to beat binary search below 8 centroids because the branch
    // predictor learns the monotone-compare pattern.
    if n <= 8 {
        let mut best = 0u8;
        let mut best_d = (value - centroids[0]).abs();
        for (i, &c) in centroids.iter().enumerate().skip(1) {
            let d = (value - c).abs();
            // `d < best_d` selects the new min; cast keeps it branchless
            // after LLVM lowering (select instruction on most ISAs).
            if d < best_d {
                best_d = d;
                best = i as u8;
            }
        }
        return best;
    }

    // Binary search for insertion point (large codebooks)
    let mut lo = 0usize;
    let mut hi = n - 1;

    // Clamp to range
    if value <= centroids[lo] {
        return lo as u8;
    }
    if value >= centroids[hi] {
        return hi as u8;
    }

    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if centroids[mid] <= value {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // lo and lo+1 bracket the value; pick the closer one
    let d_lo = (value - centroids[lo]).abs();
    let d_hi = (centroids[hi] - value).abs();
    if d_lo <= d_hi { lo as u8 } else { hi as u8 }
}

/// Dequantize an index back to centroid value.
fn dequantize_idx(idx: u8, centroids: &[f32]) -> f32 {
    centroids.get(idx as usize).copied().unwrap_or(0.0)
}

/// Pack variable-bit indices into bytes.
///
/// Each index uses `bits_per_dim[i]` bits. Output is written LSB-first.
///
/// Writes directly into `out` (a pre-zeroed slice). The caller MUST zero
/// `out` before calling — this function only writes the bytes that carry
/// packed bits; trailing bytes within the slot remain zero from the caller's
/// pre-zero.
fn pack_variable_bits(indices: &[u8], bits_per_dim: &[u8], out: &mut [u8]) {
    // Callers always pass a full-length bits_per_dim (see store/dequantize sites),
    // so we can sum the leading `indices.len()` entries directly instead of
    // the prior enumerate+get+unwrap_or pattern (which redundantly bounds-checked).
    let total_bits: usize = bits_per_dim
        .iter()
        .take(indices.len())
        .map(|&b| b as usize)
        .sum();
    let total_bytes = total_bits.div_ceil(8);
    debug_assert!(
        out.len() >= total_bytes,
        "out too small: {} < {}",
        out.len(),
        total_bytes
    );

    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0u32;
    let mut write_pos = 0usize;

    for (i, &idx) in indices.iter().enumerate() {
        let bits = bits_per_dim[i] as u32;
        bit_buffer |= (idx as u64) << bits_in_buffer;
        bits_in_buffer += bits;

        while bits_in_buffer >= 8 {
            // SAFETY: `total_bytes` was computed to hold exactly the bit-width
            // of all `indices`, and we only emit one byte per 8 bits consumed.
            out[write_pos] = (bit_buffer & 0xFF) as u8;
            write_pos += 1;
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }
    // Trailing partial byte (1..7 bits) — emit if any bits remain.
    if bits_in_buffer > 0 {
        out[write_pos] = (bit_buffer & 0xFF) as u8;
        write_pos += 1;
    }
    debug_assert_eq!(write_pos, total_bytes, "packed byte count mismatch");
}

/// Unpack variable-bit indices from bytes.
///
/// Reads LSB-first, each dim consumes `bits_per_dim[i]` bits.
fn unpack_variable_bits(packed: &[u8], bits_per_dim: &[u8], n_dims: usize, out: &mut [u8]) {
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0u32;
    let mut byte_idx = 0;

    for (i, o) in out.iter_mut().enumerate().take(n_dims) {
        // Callers always pass a full-length bits_per_dim (see docstring); direct
        // indexing avoids the per-element get+unwrap_or bounds check.
        let bits = bits_per_dim[i] as u32;
        while bits_in_buffer < bits && byte_idx < packed.len() {
            bit_buffer |= (packed[byte_idx] as u64) << bits_in_buffer;
            bits_in_buffer += 8;
            byte_idx += 1;
        }
        let mask = (1u64 << bits) - 1;
        *o = (bit_buffer & mask) as u8;
        bit_buffer >>= bits;
        bits_in_buffer -= bits;
    }
}

#[cfg(test)]
mod tests {
    use super::super::spectral::participation_ratio;
    use super::*;

    fn make_test_calibration(head_dim: usize) -> SpectralQuantCalibration {
        let mut eigenvectors = vec![0.0f32; head_dim * head_dim];
        for i in 0..head_dim {
            eigenvectors[i * head_dim + i] = 1.0;
        }
        let eigenvalues: Vec<f32> = (0..head_dim)
            .map(|i| 10.0 * 0.8f32.powi(i as i32))
            .collect();
        let d_eff = participation_ratio(&eigenvalues);
        SpectralQuantCalibration {
            eigenvectors,
            eigenvalues,
            d_eff,
            spectral_gap: None,
            var_95: 10,
            var_99: 20,
            n_samples: 100,
            head_dim,
        }
    }

    fn make_test_config(
        n_layers: usize,
        kv_dim: usize,
        max_seq_len: usize,
    ) -> SpectralQuantKVCacheConfig {
        SpectralQuantKVCacheConfig {
            avg_bits: 3.0,
            min_tail_bits: 1,
            max_bits: 8,
            qjl_dim: 16,
            lloyd_max_iter: 30,
            calibration_samples: 100,
            seed: 42,
            use_water_fill: false,
            wf_min_bits: 1,
            wf_max_bits: 6,
            n_layers,
            kv_dim,
            max_seq_len,
        }
    }

    /// No-op: codebooks are now fitted during `from_calibration()`.
    #[allow(dead_code)]
    fn init_test_centroids(_cache: &mut SpectralQuantKVCache) {
        // Codebooks are already fitted from eigenvalue distribution in from_calibration().
    }

    #[test]
    fn test_pack_unpack_roundtrip_2bit() {
        let indices = vec![0u8, 1, 2, 3, 0, 2, 1, 3];
        let bits = vec![2u8; 8];
        let mut packed = vec![0u8; 8];
        pack_variable_bits(&indices, &bits, &mut packed);
        let mut unpacked = vec![0u8; 8];
        unpack_variable_bits(&packed, &bits, 8, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_variable() {
        let indices = vec![3u8, 7, 15, 1, 0];
        let bits = vec![2u8, 3, 4, 1, 1];
        let mut packed = vec![0u8; 5];
        pack_variable_bits(&indices, &bits, &mut packed);
        let mut unpacked = vec![0u8; 5];
        unpack_variable_bits(&packed, &bits, 5, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_1bit() {
        let indices = vec![1u8, 0, 1, 1, 0, 0, 1, 0];
        let bits = vec![1u8; 8];
        let mut packed = vec![0u8; 8];
        pack_variable_bits(&indices, &bits, &mut packed);
        let mut unpacked = vec![0u8; 8];
        unpack_variable_bits(&packed, &bits, 8, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_kv_cache_store_dequantize() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let key: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).sin())
            .collect();
        cache.store_key(0, 0, &key);

        let mut recovered = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut recovered);

        let orig_norm: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rec_norm: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(rec_norm > 0.1, "reconstructed norm too small: {rec_norm}");
        // 3-bit quantization with synthetic codebooks: allow up to 2x norm distortion
        assert!(
            (orig_norm - rec_norm).abs() / orig_norm < 1.0,
            "norm changed too much: {orig_norm} -> {rec_norm}"
        );
    }

    #[test]
    fn test_kv_cache_value_roundtrip() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let value: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).cos())
            .collect();
        cache.store_value(0, 0, &value);

        let mut recovered = vec![0.0f32; kv_dim];
        cache.dequantize_value_into(0, 0, &mut recovered);

        let orig_norm: f32 = value.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rec_norm: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(rec_norm > 0.1, "reconstructed norm too small: {rec_norm}");
        assert!(
            (orig_norm - rec_norm).abs() / orig_norm < 1.0,
            "norm changed too much: {orig_norm} -> {rec_norm}"
        );
    }

    #[test]
    fn test_compression_ratio() {
        let kv_dim = 128;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 64);
        let cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let ratio = cache.compression_ratio();
        assert!(
            ratio > 5.0 && ratio < 20.0,
            "compression ratio should be ~10x, got {ratio}"
        );
    }

    #[test]
    fn test_reset_clears() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0).sin()).collect();
        cache.store_key(0, 0, &key);
        assert!(cache.key_norms[0] > 0.0);

        cache.reset();
        assert_eq!(cache.pos(), 0);
        assert_eq!(cache.key_norms[0], 0.0);
        assert_eq!(cache.val_norms[0], 0.0);
    }

    #[test]
    fn test_zero_vector_handling() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        let zero_key = vec![0.0f32; kv_dim];
        cache.store_key(0, 0, &zero_key);
        assert_eq!(cache.key_norms[0], 0.0);

        let mut recovered = vec![1.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut recovered);
        assert!(recovered.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_multi_position() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        for pos in 0..4 {
            let key: Vec<f32> = (0..kv_dim)
                .map(|i| cal.eigenvalues[i].sqrt() * ((i + pos) as f32 + 1.0).sin())
                .collect();
            cache.store_key(0, pos, &key);
        }

        for pos in 0..4 {
            let original: Vec<f32> = (0..kv_dim)
                .map(|i| cal.eigenvalues[i].sqrt() * ((i + pos) as f32 + 1.0).sin())
                .collect();
            let mut recovered = vec![0.0f32; kv_dim];
            cache.dequantize_key_into(0, pos, &mut recovered);
            let orig_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
            let rec_norm: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                rec_norm > 0.1,
                "pos {pos}: reconstructed norm too small: {rec_norm}"
            );
            assert!(
                (orig_norm - rec_norm).abs() / orig_norm < 1.0,
                "pos {pos}: norm changed too much: {orig_norm} -> {rec_norm}"
            );
        }
    }

    #[test]
    fn test_multi_layer_independence() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(2, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            &[cal.clone(), cal.clone()],
            &[cal.clone(), cal.clone()],
        );
        let key0: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).sin())
            .collect();
        let key1: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 2.0).cos())
            .collect();
        cache.store_key(0, 0, &key0);
        cache.store_key(1, 0, &key1);

        let mut rec0 = vec![0.0f32; kv_dim];
        let mut rec1 = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut rec0);
        cache.dequantize_key_into(1, 0, &mut rec1);

        // Layers should produce different reconstructions from different inputs
        let diff: f32 = rec0
            .iter()
            .zip(rec1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "layers should produce different outputs, diff={diff}"
        );
    }

    #[test]
    fn test_dequantize_key_allocating() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let key: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).sin())
            .collect();
        cache.store_key(0, 0, &key);

        let recovered = cache.dequantize_key(0, 0);
        assert_eq!(recovered.len(), kv_dim);
    }

    #[test]
    fn test_dequantize_value_allocating() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let value: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).cos())
            .collect();
        cache.store_value(0, 0, &value);

        let recovered = cache.dequantize_value(0, 0);
        assert_eq!(recovered.len(), kv_dim);
    }

    #[test]
    fn test_set_pos() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        assert_eq!(cache.pos(), 0);
        cache.set_pos(10);
        assert_eq!(cache.pos(), 10);
    }

    // ── Parallel dequantize tests (Issue 064) ──────────────────

    #[test]
    fn test_dequantize_with_scratch_matches_into() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let value: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).cos()).collect();
        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        // Mut-self path
        let mut buf_into = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut buf_into);

        // External scratch path
        let mut scratch = DequantizeScratch::new(kv_dim);
        let mut buf_scratch = vec![0.0f32; kv_dim];
        cache.dequantize_key_into_with_scratch(0, 0, &mut scratch, &mut buf_scratch);

        for (i, (a, b)) in buf_into.iter().zip(buf_scratch.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "key mismatch at [{i}]: {a} vs {b}");
        }

        // Same for value
        let mut val_into = vec![0.0f32; kv_dim];
        cache.dequantize_value_into(0, 0, &mut val_into);
        let mut val_scratch = vec![0.0f32; kv_dim];
        cache.dequantize_value_into_with_scratch(0, 0, &mut scratch, &mut val_scratch);

        for (i, (a, b)) in val_into.iter().zip(val_scratch.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "value mismatch at [{i}]: {a} vs {b}");
        }
    }

    #[test]
    fn test_par_dequantize_keys_matches_seq() {
        use super::super::forward::{
            dequantize_spectral_keys_flat, par_dequantize_spectral_keys_flat,
        };

        let kv_dim = 16;
        let n_positions = 32;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, n_positions);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        // Store keys at all positions
        for pos in 0..n_positions {
            let key: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + pos * 3) as f32 * 0.1).sin())
                .collect();
            cache.store_key(0, pos, &key);
        }

        // Sequential (uses &mut self)
        let seq_flat = dequantize_spectral_keys_flat(&mut cache, 0, n_positions - 1, kv_dim);

        // Parallel (uses &self, threshold=1 forces parallel path)
        let par_flat = par_dequantize_spectral_keys_flat(&cache, 0, n_positions - 1, kv_dim, 1);

        assert_eq!(seq_flat.len(), par_flat.len(), "length mismatch");
        for (i, (a, b)) in seq_flat.iter().zip(par_flat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "key dequant mismatch at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_par_dequantize_values_matches_seq() {
        use super::super::forward::{
            dequantize_spectral_values_flat, par_dequantize_spectral_values_flat,
        };

        let kv_dim = 16;
        let n_positions = 32;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, n_positions);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        for pos in 0..n_positions {
            let val: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + pos * 5) as f32 * 0.07).cos())
                .collect();
            cache.store_value(0, pos, &val);
        }

        let seq_flat = dequantize_spectral_values_flat(&mut cache, 0, n_positions - 1, kv_dim);
        let par_flat = par_dequantize_spectral_values_flat(&cache, 0, n_positions - 1, kv_dim, 1);

        assert_eq!(seq_flat.len(), par_flat.len(), "length mismatch");
        for (i, (a, b)) in seq_flat.iter().zip(par_flat.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "value dequant mismatch at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_par_dequantize_threshold_fallback() {
        use super::super::forward::par_dequantize_spectral_keys_flat;

        let kv_dim = 16;
        let n_positions = 8;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, n_positions);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        for pos in 0..n_positions {
            let key: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + pos) as f32 * 0.1).sin())
                .collect();
            cache.store_key(0, pos, &key);
        }

        // threshold=100 → sequential fallback (n_positions=8 < 100)
        let seq_result = par_dequantize_spectral_keys_flat(&cache, 0, n_positions - 1, kv_dim, 100);
        // threshold=1 → parallel path
        let par_result = par_dequantize_spectral_keys_flat(&cache, 0, n_positions - 1, kv_dim, 1);

        assert_eq!(seq_result.len(), par_result.len());
        for (i, (a, b)) in seq_result.iter().zip(par_result.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "threshold fallback mismatch at [{i}]: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_par_dequantize_empty() {
        use super::super::forward::par_dequantize_spectral_keys_flat;

        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 4);
        let cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        // No keys stored — all zero norms, dequantize returns zeros
        let result = par_dequantize_spectral_keys_flat(&cache, 0, 3, kv_dim, 1);
        assert_eq!(result.len(), 4 * kv_dim);
        assert!(
            result.iter().all(|&v| v == 0.0),
            "empty cache should dequantize to zeros"
        );
    }

    /// The zero-alloc `_into` variant must produce bit-exact output as the
    /// allocating variant, both on the parallel path (threshold=1) and the
    /// sequential fallback (threshold > n_positions).
    #[test]
    fn test_par_dequantize_keys_into_matches_alloc() {
        use super::super::forward::{
            par_dequantize_spectral_keys_flat, par_dequantize_spectral_keys_flat_into,
        };

        let kv_dim = 16;
        let n_positions = 32;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, n_positions);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        for pos in 0..n_positions {
            let key: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + pos * 3) as f32 * 0.1).sin())
                .collect();
            cache.store_key(0, pos, &key);
        }

        // Test both parallel (threshold=1) and sequential fallback (threshold=1000)
        for &threshold in &[1usize, 1000] {
            let alloc_flat =
                par_dequantize_spectral_keys_flat(&cache, 0, n_positions - 1, kv_dim, threshold);

            let mut into_buf = vec![0.0f32; n_positions * kv_dim];
            let mut scratch = DequantizeScratch::new(kv_dim);
            par_dequantize_spectral_keys_flat_into(
                &cache,
                0,
                n_positions - 1,
                kv_dim,
                threshold,
                &mut into_buf,
                &mut scratch,
            );

            assert_eq!(
                alloc_flat.len(),
                into_buf.len(),
                "len mismatch threshold={threshold}"
            );
            for (i, (a, b)) in alloc_flat.iter().zip(into_buf.iter()).enumerate() {
                assert_eq!(
                    a, b,
                    "bit mismatch at [{i}] threshold={threshold}: {a} vs {b}"
                );
            }
        }
    }

    /// `_into` variant must handle pos=0 (single position) correctly.
    #[test]
    fn test_par_dequantize_keys_into_single_pos() {
        use super::super::forward::par_dequantize_spectral_keys_flat_into;

        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 4);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.3).sin()).collect();
        cache.store_key(0, 0, &key);

        let mut buf = vec![0.0f32; kv_dim];
        let mut scratch = DequantizeScratch::new(kv_dim);
        par_dequantize_spectral_keys_flat_into(&cache, 0, 0, kv_dim, 1, &mut buf, &mut scratch);
        // Should have written kv_dim floats for position 0
        assert_eq!(buf.len(), kv_dim);
        // Values should be non-zero (we stored a non-zero key)
        assert!(
            buf.iter().any(|&v| v != 0.0),
            "expected non-zero dequant output"
        );
    }
}

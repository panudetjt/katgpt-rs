//! Jev structured reads — a read-only seeded-canvas decision primitive.
//!
//! Issue 859 / Research 574 §5: the vLLM PR #57250 contract, instantiated on
//! our D2F stack. `seed(canvas) → ONE denoise step → per-free-slot {argmax,
//! exact logprob over a caller-supplied label-id list, entropy over the
//! NORMALIZED label distribution} → return` — **no commit forward**: the
//! canvas is never written, so a read never shifts what a later read sees.
//!
//! Zero new substrate (Research 574's composition law): the seed is the
//! existing `config.mask_token` placement convention (the denoise loop's own
//! skip rule at `denoise_loops.rs:148-151` already honors pre-seeded
//! positions), the step is the identical `forward_bidirectional_positions_into`
//! call `denoise_loop` makes per step, and the readout math is the only new
//! code. `ac_prefix` is NOT consumed — it exists to give CAUSAL forwards
//! arbitrary-conditional power via sequence augmentation, while the D2F
//! bidirectional forward is natively the right conditioning shape (mask
//! embeddings at free slots, exactly `DiffusionGemma`'s canvas semantics).
//! `canvas_schema` is NOT consumed either — a positions-not-tokens topology
//! compiler is a topology seam, not a placement seam.
//!
//! Contract details carried from the reference (Research 574 TL;DR):
//! - **label logprobs are exact full-marginal values** (`logprob(t) =
//!   logits[t] − logsumexp(full vocab)`) — the PR's top-k blindness cannot
//!   arise because our marginals are direct;
//! - **entropy is over the subset-RENORMALIZED label distribution** — the
//!   reference reviewer's nit (entropy over an unnormalized top-set
//!   understating true entropy) is corrected by construction;
//! - **re-reads of a deterministic forward are bit-identical**, so agreement
//!   bars are only meaningful through [`sample_label_index`] (temperature > 0)
//!   — the analog of the reference's temperature-1 logprob read.
//!
//! G4 (alloc-free hot path): [`structured_read_into`] allocates nothing after
//! [`StructuredReadScratch::new`] — the per-position readout uses fixed
//! arrays and the bidirectional context's pre-allocated scratch. The
//! [`SlotReadout::label_logprobs`] fixed array trades a small per-slot size
//! for zero per-slot heap traffic; label sets beyond [`MAX_LABELS`] are
//! refused up front (fail-closed, the bounded-domain rule — the reference
//! corpora max out at 26 options).

use crate::forward_positions::{BidirectionalContext, forward_bidirectional_positions_into};
use katgpt_core::simd;
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;

/// Upper bound on a caller-supplied label set (the bounded-domain rule).
/// The reference PR's widest corpus (unit comparison) uses 26 options.
pub const MAX_LABELS: usize = 64;

/// One free slot's readout. Ordered fields group by alignment (f32 × 5,
/// then u32 × 2, then the fixed array) to minimize padding.
///
/// `label_logprobs[i]` for `i < n_labels` is the EXACT full-marginal
/// `log p(label_ids[i] | canvas)` at this slot after one denoise step;
/// indices align with the caller's `label_ids` slice order.
#[derive(Clone, Copy, Debug)]
pub struct SlotReadout {
    /// Full-marginal logprob of the argmax label (max of `label_logprobs`).
    pub argmax_logprob: f32,
    /// Subset-NORMALIZED probability of the argmax label (confidence within
    /// the option set — 1.0 = the model puts all option mass here).
    pub argmax_label_prob: f32,
    /// Entropy over the NORMALIZED label distribution, `H = −Σ pᵢ ln pᵢ`
    /// (nats; 0.0 = deterministic within the option set; `ln(n_labels`) = flat).
    pub label_entropy: f32,
    /// Full-vocab argmax token at this slot (the free-decode signal — what
    /// an unconstrained denoise commit would write here; the competitor
    /// arm's read, not the contract answer).
    pub vocab_argmax_token: u32,
    /// Canvas position this readout belongs to (a mask position).
    pub position: u32,
    /// Index into the caller's `label_ids` slice of the argmax label.
    pub argmax_index: u32,
    /// Number of valid entries in `label_logprobs` (== `label_ids.len()`).
    pub n_labels: u32,
    /// Exact full-marginal logprobs, input-order aligned; entries ≥
    /// `n_labels` are zeroed.
    pub label_logprobs: [f32; MAX_LABELS],
}

/// Fail-closed validation errors for [`structured_read_into`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructuredReadError {
    /// `label_ids` is empty — a decision needs at least one option.
    EmptyLabels,
    /// `label_ids.len() > MAX_LABELS` — refuse rather than truncate.
    TooManyLabels { n: usize, max: usize },
    /// A label id ≥ `config.vocab_size` would read out of bounds.
    LabelOutOfRange { label: u32, vocab: u32 },
    /// `label_ids` must be strictly ascending (rejects duplicates, which
    /// would double-count in the subset normalization).
    LabelsNotAscending,
    /// The canvas carries no free (`config.mask_token`) slot — nothing to read.
    NoFreeSlots,
    /// `out` is shorter than the free-slot count.
    OutputTooShort { have: usize, need: usize },
}

impl std::fmt::Display for StructuredReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLabels => write!(f, "structured_read: label_ids is empty"),
            Self::TooManyLabels { n, max } => {
                write!(f, "structured_read: {n} labels exceeds MAX_LABELS={max}")
            }
            Self::LabelOutOfRange { label, vocab } => {
                write!(f, "structured_read: label {label} >= vocab {vocab}")
            }
            Self::LabelsNotAscending => {
                write!(f, "structured_read: label_ids must be strictly ascending")
            }
            Self::NoFreeSlots => write!(f, "structured_read: canvas has no free (mask) slot"),
            Self::OutputTooShort { have, need } => {
                write!(f, "structured_read: out len {have} < free slots {need}")
            }
        }
    }
}

impl std::error::Error for StructuredReadError {}

/// Reusable scratch for [`structured_read_into`]: the pre-allocated
/// [`BidirectionalContext`] (per-position logits + attention scratch) plus
/// the vocab-sized exp buffer. Allocate once, reuse across reads.
pub struct StructuredReadScratch {
    bctx: BidirectionalContext,
    exp_buf: Vec<f32>,
}

impl StructuredReadScratch {
    pub fn new(config: &Config) -> Self {
        let bctx = BidirectionalContext::new(config);
        let exp_buf = vec![0.0f32; config.vocab_size];
        Self { bctx, exp_buf }
    }
}

fn validate(
    label_ids: &[u32],
    config: &Config,
    n_free: usize,
    out_len: usize,
) -> Result<(), StructuredReadError> {
    if label_ids.is_empty() {
        return Err(StructuredReadError::EmptyLabels);
    }
    if label_ids.len() > MAX_LABELS {
        return Err(StructuredReadError::TooManyLabels {
            n: label_ids.len(),
            max: MAX_LABELS,
        });
    }
    let vocab = config.vocab_size as u32;
    for &l in label_ids {
        if l >= vocab {
            return Err(StructuredReadError::LabelOutOfRange { label: l, vocab });
        }
    }
    for w in label_ids.windows(2) {
        if w[0] >= w[1] {
            return Err(StructuredReadError::LabelsNotAscending);
        }
    }
    if n_free == 0 {
        return Err(StructuredReadError::NoFreeSlots);
    }
    if out_len < n_free {
        return Err(StructuredReadError::OutputTooShort {
            have: out_len,
            need: n_free,
        });
    }
    Ok(())
}

/// The Jev structured read (Issue 859 pinned claim), alloc-free after
/// scratch construction.
///
/// Runs ONE bidirectional denoise step over `canvas` (fixed schema tokens +
/// `config.mask_token` at the free slots) and writes one [`SlotReadout`]
/// per mask position, in position order, into `out[..n_free]`. The canvas
/// is **never written** — read-only by construction, G1b-pinned.
///
/// The forward is the identical `forward_bidirectional_positions_into`
/// call the denoise loop makes per step; per-slot logprobs are exact
/// full-marginal values (stable max-shift logsumexp over the full vocab).
pub fn structured_read_into(
    out: &mut [SlotReadout],
    weights: &TransformerWeights,
    config: &Config,
    canvas: &[usize],
    label_ids: &[u32],
    scratch: &mut StructuredReadScratch,
) -> Result<usize, StructuredReadError> {
    let seq_len = canvas.len().min(config.block_size);
    let canvas = &canvas[..seq_len];
    let n_free = canvas.iter().filter(|&&t| t == config.mask_token).count();
    validate(label_ids, config, n_free, out.len())?;

    // ONE denoise step — the read surface is per-position full-vocab logits.
    forward_bidirectional_positions_into(weights, canvas, config, &mut scratch.bctx);

    let vocab = config.vocab_size;
    let n_labels = label_ids.len();
    let mut slot = 0usize;
    for (p, &tok) in canvas.iter().enumerate() {
        if tok != config.mask_token {
            continue;
        }
        let logits_p = &scratch.bctx.all_logits[p * vocab..(p + 1) * vocab];

        // Stable full-vocab logsumexp (the denoise_loop pattern: max-shift,
        // exp into scratch, SIMD sum) + full-vocab argmax in the same scan.
        let max_l = simd::simd_max_f32(logits_p);
        let exp_buf = &mut scratch.exp_buf[..vocab];
        exp_buf.copy_from_slice(logits_p);
        simd::simd_add_scalar_inplace(exp_buf, -max_l);
        simd::simd_exp_inplace(exp_buf);
        let sum_exp = simd::simd_sum_f32(exp_buf);
        let lse = max_l + sum_exp.ln();
        let (vocab_argmax_idx, _) = simd::simd_argmax_f32(logits_p);

        // Exact full-marginal label logprobs (input order).
        let mut label_logprobs = [0.0f32; MAX_LABELS];
        let mut argmax_index = 0usize;
        for (i, &l) in label_ids.iter().enumerate() {
            label_logprobs[i] = logits_p[l as usize] - lse;
            if i > 0 && label_logprobs[i] > label_logprobs[argmax_index] {
                argmax_index = i;
            }
        }

        // Subset-NORMALIZED distribution + entropy (the reference reviewer's
        // nit corrected by construction). Small-n scalar loops — bit-stable.
        let mut s = 0.0f32;
        for lp in &label_logprobs[..n_labels] {
            s += lp.exp();
        }
        let inv_s = 1.0 / s;
        let mut entropy = 0.0f32;
        for lp in &label_logprobs[..n_labels] {
            let p = lp.exp() * inv_s;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        let argmax_label_prob = label_logprobs[argmax_index].exp() * inv_s;

        out[slot] = SlotReadout {
            argmax_logprob: label_logprobs[argmax_index],
            argmax_label_prob,
            label_entropy: entropy,
            vocab_argmax_token: vocab_argmax_idx as u32,
            position: p as u32,
            argmax_index: argmax_index as u32,
            n_labels: n_labels as u32,
            label_logprobs,
        };
        slot += 1;
    }
    Ok(slot)
}

/// Allocating convenience wrapper over [`structured_read_into`] — builds
/// scratch per call. Hot paths should reuse a [`StructuredReadScratch`].
pub fn structured_read(
    weights: &TransformerWeights,
    config: &Config,
    canvas: &[usize],
    label_ids: &[u32],
) -> Result<Vec<SlotReadout>, StructuredReadError> {
    let n_free = canvas.iter().filter(|&&t| t == config.mask_token).count();
    // Validate before allocating anything on the error paths.
    validate(label_ids, config, n_free, usize::MAX)?;
    let mut out = vec![
        SlotReadout {
            argmax_logprob: 0.0,
            argmax_label_prob: 0.0,
            label_entropy: 0.0,
            vocab_argmax_token: 0,
            position: 0,
            argmax_index: 0,
            n_labels: 0,
            label_logprobs: [0.0; MAX_LABELS],
        };
        n_free.max(1)
    ];
    let mut scratch = StructuredReadScratch::new(config);
    let n = structured_read_into(&mut out, weights, config, canvas, label_ids, &mut scratch)?;
    out.truncate(n);
    Ok(out)
}

/// Sample a label index from a slot's subset distribution at temperature
/// `t` (T5's stochastic re-read: agreement bars are only meaningful when
/// re-reads differ — the deterministic forward re-reads bit-identically).
///
/// Uses `label_logprobs` directly: within the subset, full-marginal logprob
/// differences equal logit differences (the shared logsumexp cancels), so
/// `p_i ∝ exp(logprob_i / t)` is exactly the temperature-scaled subset
/// softmax. `t <= 0` falls back to argmax (index of the max logprob).
pub fn sample_label_index(
    readout: &SlotReadout,
    temperature: f32,
    rng: &mut katgpt_types::Rng,
) -> u32 {
    let n = readout.n_labels as usize;
    if n == 0 {
        return 0;
    }
    if temperature <= 0.0 {
        let mut best = 0usize;
        for i in 1..n {
            if readout.label_logprobs[i] > readout.label_logprobs[best] {
                best = i;
            }
        }
        return best as u32;
    }
    let inv_t = 1.0 / temperature;
    let max_scaled = readout.label_logprobs[..n]
        .iter()
        .fold(f32::NEG_INFINITY, |m, &lp| m.max(lp * inv_t));
    let mut probs = [0.0f32; MAX_LABELS];
    let mut s = 0.0f32;
    for (p, &lp) in probs.iter_mut().zip(readout.label_logprobs[..n].iter()) {
        let e = (lp * inv_t - max_scaled).exp();
        *p = e;
        s += e;
    }
    let target = rng.uniform() * s;
    let mut acc = 0.0f32;
    // Hot label-scan loop: the index IS the returned label id — the
    // denoise_loops.rs needless_range_loop precedent.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        acc += probs[i];
        if target < acc {
            return i as u32;
        }
    }
    (n - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_positions::BidirectionalContext;

    fn fixture(seed: u64) -> (TransformerWeights, Config) {
        let config = Config::micro_dllm();
        let mut rng = katgpt_types::Rng::new(seed);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    /// [fixed, fixed, FREE, fixed, FREE, fixed] — schema tokens + 2 free slots.
    fn canvas(config: &Config) -> Vec<usize> {
        vec![5, 7, config.mask_token, 11, config.mask_token, 2]
    }

    fn labels() -> Vec<u32> {
        vec![1, 4, 9, 16, 22]
    }

    /// G1: label logprobs are EXACT full-marginal values — cross-checked
    /// against an independent f64 stable-logsumexp reference over the same
    /// step's logits (recomputed via a second, test-owned forward), AND
    /// bit-identical across two calls with fresh scratch (determinism +
    /// wiring stability — no subset-normalization or staleness leakage).
    #[test]
    fn g1_label_logprobs_match_full_marginal_reference() {
        let (weights, config) = fixture(0x0859_0001);
        let canvas = canvas(&config);
        let labels = labels();
        let out = structured_read(&weights, &config, &canvas, &labels).unwrap();
        assert_eq!(out.len(), 2, "one readout per mask slot");

        // Independent reference: test-owned forward + f64 stable logsumexp.
        let mut bctx = BidirectionalContext::new(&config);
        forward_bidirectional_positions_into(&weights, &canvas, &config, &mut bctx);
        let vocab = config.vocab_size;
        for r in &out {
            let p = r.position as usize;
            let logits = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let max_l = logits
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b: &f32| a.max(b as f64));
            let sum: f64 = logits.iter().map(|&l| (l as f64 - max_l).exp()).sum();
            let lse = max_l + sum.ln();
            for (i, &lab) in labels.iter().enumerate() {
                let reference = logits[lab as usize] as f64 - lse;
                let got = r.label_logprobs[i] as f64;
                assert!(
                    (got - reference).abs() < 1e-5,
                    "slot {p} label {lab}: got {got}, reference {reference}"
                );
            }
        }

        // Bit-identity across calls with fresh scratch.
        let out2 = structured_read(&weights, &config, &canvas, &labels).unwrap();
        for (a, b) in out.iter().zip(&out2) {
            assert_eq!(
                a.label_logprobs, b.label_logprobs,
                "bit-identical across calls"
            );
            assert_eq!(a.label_entropy, b.label_entropy);
            assert_eq!(a.argmax_index, b.argmax_index);
        }
    }

    /// G1b: the canvas is never written — read-only by construction.
    #[test]
    fn g1b_canvas_bit_identical_after_read() {
        let (weights, config) = fixture(0x0859_0002);
        let canvas = canvas(&config);
        let before = canvas.clone();
        let _ = structured_read(&weights, &config, &canvas, &labels()).unwrap();
        assert_eq!(canvas, before, "structured_read must not commit");
    }

    /// Readout field coherence: subset probabilities sum to 1, argmax is the
    /// max logprob, entropy within [0, ln(n)] (nats), `positions/n_labels` right.
    #[test]
    fn readout_fields_coherent() {
        let (weights, config) = fixture(0x0859_0003);
        let out = structured_read(&weights, &config, &canvas(&config), &labels()).unwrap();
        let n = labels().len();
        let ln_n = (n as f32).ln();
        for r in &out {
            assert_eq!(r.n_labels as usize, n);
            let s: f32 = r.label_logprobs[..n].iter().map(|lp| lp.exp()).sum();
            let p_sum: f32 = r.label_logprobs[..n].iter().map(|lp| lp.exp() / s).sum();
            assert!(
                (p_sum - 1.0).abs() < 1e-5,
                "subset-normalized probs must sum to 1 (got {p_sum})"
            );
            let (imax, _) = r.label_logprobs[..n]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            assert_eq!(
                r.argmax_index as usize, imax,
                "argmax_index must be the max logprob"
            );
            assert_eq!(r.argmax_logprob, r.label_logprobs[imax]);
            let expected_prob = r.label_logprobs[imax].exp() / s;
            assert!((r.argmax_label_prob - expected_prob).abs() < 1e-6);
            let h = r.label_entropy;
            assert!(
                h >= 0.0 && h <= ln_n + 1e-4,
                "entropy {h} outside [0, ln({n})={ln_n}]"
            );
            assert!(r.vocab_argmax_token < config.vocab_size as u32);
        }
        assert_eq!(out[0].position, 2);
        assert_eq!(out[1].position, 4);
    }

    #[test]
    fn validation_errors_fail_closed() {
        let (weights, config) = fixture(0x0859_0004);
        let canvas = canvas(&config);
        let mut scratch = StructuredReadScratch::new(&config);
        let mut out = [SlotReadout {
            argmax_logprob: 0.0,
            argmax_label_prob: 0.0,
            label_entropy: 0.0,
            vocab_argmax_token: 0,
            position: 0,
            argmax_index: 0,
            n_labels: 0,
            label_logprobs: [0.0; MAX_LABELS],
        }; 2];

        let e = structured_read_into(&mut out, &weights, &config, &canvas, &[], &mut scratch)
            .unwrap_err();
        assert_eq!(e, StructuredReadError::EmptyLabels);

        let many: Vec<u32> = (0..=MAX_LABELS as u32).collect();
        let e = structured_read_into(&mut out, &weights, &config, &canvas, &many, &mut scratch)
            .unwrap_err();
        assert_eq!(
            e,
            StructuredReadError::TooManyLabels {
                n: many.len(),
                max: MAX_LABELS
            }
        );

        let e = structured_read_into(&mut out, &weights, &config, &canvas, &[9, 27], &mut scratch)
            .unwrap_err();
        assert_eq!(
            e,
            StructuredReadError::LabelOutOfRange {
                label: 27,
                vocab: 27
            }
        );

        let e = structured_read_into(
            &mut out,
            &weights,
            &config,
            &canvas,
            &[4, 4, 9],
            &mut scratch,
        )
        .unwrap_err();
        assert_eq!(e, StructuredReadError::LabelsNotAscending);

        let e = structured_read_into(
            &mut out,
            &weights,
            &config,
            &[1, 2, 3],
            &[1, 2],
            &mut scratch,
        )
        .unwrap_err();
        assert_eq!(e, StructuredReadError::NoFreeSlots);

        let mut one = [out[0]];
        let e = structured_read_into(&mut one, &weights, &config, &canvas, &[1, 2], &mut scratch)
            .unwrap_err();
        assert_eq!(e, StructuredReadError::OutputTooShort { have: 1, need: 2 });
    }

    /// t=0 is argmax (deterministic); t>0 stays in range and is seeded-deterministic.
    #[test]
    fn sample_label_index_t_zero_is_argmax() {
        let (weights, config) = fixture(0x0859_0005);
        let out = structured_read(&weights, &config, &canvas(&config), &labels()).unwrap();
        let mut rng = katgpt_types::Rng::new(7);
        for r in &out {
            for _ in 0..16 {
                assert_eq!(sample_label_index(r, 0.0, &mut rng), r.argmax_index);
            }
            let mut rng_a = katgpt_types::Rng::new(0x5EED);
            let mut rng_b = katgpt_types::Rng::new(0x5EED);
            for _ in 0..16 {
                let a = sample_label_index(r, 1.5, &mut rng_a);
                let b = sample_label_index(r, 1.5, &mut rng_b);
                assert_eq!(a, b, "seeded determinism");
                assert!(a < r.n_labels);
            }
        }
    }
}

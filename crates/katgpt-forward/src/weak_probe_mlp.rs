//! Weak-side probe consumption (Issue 865) — [`MlpWeakProbe`] +
//! [`DropoutHeadProbe`].
//!
//! [`MlpWeakProbe`] wraps a BLAKE3-committed
//! [`ProbeArtifact`](katgpt_speculative::probe_artifact::ProbeArtifact) (the
//! `LatentDynamicsMLP` connector + shared trunk `lm_head`, trained by the
//! riir-train `nextlat_*` lane pattern) and implements the
//! [`WeakLogitProbe`] seam from Issue 865 T1: for each denoised block
//! position, the tapped early-layer hidden state goes through the connector
//! (with a zero `next_emb` — the training convention; block positions are
//! mask tokens whose embeddings carry no per-position signal) and then the
//! shared head, producing the weak logits the affine combine extrapolates
//! against.
//!
//! [`DropoutHeadProbe`] is the MODELLESS weak side (Bench 847's deferred
//! arm (b), unblocked WITHOUT kernel work): the shared trunk head reading a
//! deterministically dropout-masked tap — autoguidance's cheap sibling
//! (arXiv:2609.19356 "cheap-weak": the weak side is the SAME function with
//! input noise, not a different damaged function — the correlated-dynamics
//! requirement the Bench-847 identity-truncation refutation measured from
//! the other side). The mask is a fixed LCG stream keyed by `(position,
//! denoise step)`, so decode is reproducible and consumes no runtime RNG.
//! No artifact, no training — the weak side that makes `probe_guidance`
//! usable without the riir-train lane, and the instrument of the
//! headroom-trunk study (Bench 850).
//!
//! Zero-alloc on the hot path: the connector's scratch buffers, the latent
//! row, and the zero embedding are allocated once at construction and reused
//! across every probe call, matching the seam's `&mut self` contract.

use std::path::Path;

use katgpt_core::types::matmul;
use katgpt_speculative::belief_drafter::MlpForwardScratch;
use katgpt_speculative::probe_artifact::{ProbeArtifact, ProbeArtifactError};

use crate::d2f_context::{ProbeCtx, WeakLogitProbe};

/// A trained weak-side probe loaded from a [`ProbeArtifact`] (Issue 865 T2).
///
/// The whole artifact is frozen data: the connector MLP and the shared head
/// are never mutated at runtime (the modelless consumption rule — the only
/// weight mutation is swapping the frozen snapshot, and installation replaces
/// the probe wholesale via `D2fContext::set_guidance`).
pub struct MlpWeakProbe {
    artifact: ProbeArtifact,
    /// Connector scratch (`[2n, 2n, n, n, n]` buffers), reused per position.
    scratch: MlpForwardScratch,
    /// Connector output row `[n_embd]`, reused per position.
    latent: Vec<f32>,
    /// The connector's constant `next_emb` input `[n_embd]` — zeros. The
    /// training lane trains against the same convention (see module docs).
    zeros: Vec<f32>,
}

impl MlpWeakProbe {
    /// Wrap a verified artifact.
    ///
    /// The artifact's `tap_layer` may be ANY layer index (Issue 869 T2
    /// unblocked deeper taps — the wire has carried the field since Issue
    /// 865 T2; the kernel now captures any configured set). Validation is
    /// deferred to the one place both sides are known: `D2fContext::set_guidance`
    /// panics loudly when the decode context does not capture the artifact's
    /// layer (with the `set_probe_tap_layers` / `set_decode_layers` remedy),
    /// so a mismatched artifact can never silently misread the wrong plane.
    pub fn new(artifact: ProbeArtifact) -> Result<Self, ProbeArtifactError> {
        let n = artifact.mlp.n_embd;
        Ok(Self {
            scratch: MlpForwardScratch::new(n),
            latent: vec![0.0f32; n],
            zeros: vec![0.0f32; n],
            artifact,
        })
    }

    /// Load + verify from a wire file, then wrap.
    pub fn from_bin(path: &Path) -> Result<Self, ProbeArtifactError> {
        Self::new(ProbeArtifact::load_from_bin(path)?)
    }

    /// Borrow the sealed artifact (commitment, version, tap metadata).
    pub fn artifact(&self) -> &ProbeArtifact {
        &self.artifact
    }
}

impl WeakLogitProbe for MlpWeakProbe {
    fn probe(&mut self, input: ProbeCtx<'_>, out: &mut [f32]) {
        let n = self.artifact.mlp.n_embd;
        let vocab = self.artifact.vocab;
        debug_assert_eq!(
            input.n_embd, n,
            "probe artifact n_embd must match the decode context"
        );
        debug_assert_eq!(
            out.len(),
            (input.seq_len - input.block_start) * vocab,
            "probe out must hold exactly the block's logits"
        );
        // Issue 869 T2: locate THIS artifact's plane in the layered tap. The
        // install-time validation in `set_guidance` is what guarantees the
        // layer is captured; the expect is the belt to that braces.
        let slot = input
            .tap_layers
            .iter()
            .position(|&l| l == self.artifact.tap_layer)
            .expect("tap layer validated at guidance install (D2fContext::set_guidance)");
        let base = slot * input.tap_plane;
        for p in input.block_start..input.seq_len {
            let h = &input.tap[base + p * n..base + (p + 1) * n];
            self.artifact
                .mlp
                .forward_into(h, &self.zeros, &mut self.scratch, &mut self.latent);
            let row_off = (p - input.block_start) * vocab;
            matmul(
                &mut out[row_off..row_off + vocab],
                &self.artifact.lm_head,
                &self.latent,
                vocab,
                n,
            );
        }
    }

    fn tap_layer(&self) -> Option<usize> {
        Some(self.artifact.tap_layer)
    }
}

/// Modelless dropout-autoguidance weak side (Issue 865 T3 arm (b), the
/// Bench-847-deferred substrate): the shared trunk head reading a
/// deterministically dropout-masked tap.
///
/// The weak side stays a NOISY VERSION OF THE SAME FUNCTION — the paper's
/// correlated-dynamics requirement — unlike a structurally damaged weak side
/// (the identity-truncation refutation in Bench 847 measured that class
/// failing). Masking happens at the TAP level, so no inference-time kernel
/// dropout is needed (the reason Bench 847 deferred this arm is gone).
///
/// Layer-agnostic (Issue 869 T2): reads SLOT 0 of the layered tap — whatever
/// layer the context captures first — and declares `tap_layer() = None`, so
/// it installs against any tap set without validation friction.
///
/// Deterministic: the mask is a fixed LCG stream keyed by `(position,
/// denoise step)` — reproducible decode, no runtime RNG consumed. Zero-alloc:
/// the masked row is a pre-allocated scratch, reused per position.
pub struct DropoutHeadProbe {
    /// Shared trunk head `[vocab * n_embd]`, row-major.
    head: Vec<f32>,
    vocab: usize,
    n: usize,
    /// Drop fraction in `[0, 1)` — 0.5 is the Bench-850-swept default.
    drop_rate: f32,
    /// Masked tap row `[n_embd]`, reused per position.
    masked: Vec<f32>,
}

impl DropoutHeadProbe {
    /// Build from the shared trunk head. `drop_rate = 0.5` is the swept
    /// default; `0.0` degenerates to `head · tap` (the truncation weak side).
    pub fn new(vocab: usize, n_embd: usize, lm_head: Vec<f32>, drop_rate: f32) -> Self {
        assert_eq!(
            lm_head.len(),
            vocab * n_embd,
            "lm_head must be [vocab, n_embd] row-major"
        );
        Self {
            head: lm_head,
            vocab,
            n: n_embd,
            drop_rate: drop_rate.clamp(0.0, 0.95),
            masked: vec![0.0f32; n_embd],
        }
    }
}

impl WeakLogitProbe for DropoutHeadProbe {
    fn probe(&mut self, input: ProbeCtx<'_>, out: &mut [f32]) {
        let (n, vocab) = (self.n, self.vocab);
        debug_assert_eq!(input.n_embd, n);
        debug_assert_eq!(
            out.len(),
            (input.seq_len - input.block_start) * vocab,
            "probe out must hold exactly the block's logits"
        );
        for p in input.block_start..input.seq_len {
            // Fixed LCG mask per (position, denoise step) — same stream shape
            // as the study runner, reproducible across runs and boxes.
            let mut state = (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (input.step as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                ^ 0xDEAD_BEEF_CAFE_F00D;
            for i in 0..n {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Top 31 bits → uniform in [0, 1); drop when below the rate.
                let r = ((state >> 33) as f32) / (1u64 << 31) as f32;
                self.masked[i] = if r < self.drop_rate {
                    0.0
                } else {
                    input.tap[p * n + i]
                };
            }
            let row_off = (p - input.block_start) * vocab;
            matmul(
                &mut out[row_off..row_off + vocab],
                &self.head,
                &self.masked,
                vocab,
                n,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_speculative::belief_drafter::LatentDynamicsMLP;

    fn fixture(vocab: usize, n_embd: usize, tap_layer: usize) -> ProbeArtifact {
        let mlp = LatentDynamicsMLP::random_init(n_embd);
        let lm_head: Vec<f32> = (0..vocab * n_embd)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        ProbeArtifact::from_parts(mlp, lm_head, tap_layer, 1).expect("fixture artifact")
    }

    fn probe_ctx<'a>(
        tap: &'a [f32],
        block_start: usize,
        seq_len: usize,
        vocab: usize,
        n_embd: usize,
    ) -> ProbeCtx<'a> {
        ProbeCtx {
            xr: tap,
            x_norm: tap,
            tap,
            // Single-plane layout (slot 0 = layer 0) — the default tap set's
            // shape; the multi-plane reads are covered by the d2f kernel tests.
            tap_layers: &[0],
            tap_plane: tap.len(),
            tokens: &[],
            committed_len: 0,
            block_start,
            seq_len,
            vocab,
            n_embd,
            step: 0,
        }
    }

    #[test]
    fn new_accepts_deeper_tap_layers() {
        // Issue 869 T2: the constructor-time `tap_layer != 0` rejection is
        // GONE — deeper taps are legal artifacts, validated at guidance
        // install (the d2f tests pin that panic). Wrapping must succeed for
        // any layer index, and the probe must DECLARE its layer.
        for tap_layer in [0usize, 1, 5] {
            let artifact = fixture(9, 8, tap_layer);
            let probe = MlpWeakProbe::new(artifact).expect("any tap_layer wraps");
            assert_eq!(probe.tap_layer(), Some(tap_layer));
        }
    }

    #[test]
    fn probe_reads_its_own_plane_in_a_layered_tap() {
        // A two-plane tap (layers [0, 1], stride = 6 rows): an artifact pinned
        // to layer 1 must read the SECOND plane, byte-identical to feeding
        // that plane alone as a single-plane tap.
        let artifact = fixture(7, 8, 1);
        let vocab = artifact.vocab;
        let n = artifact.mlp.n_embd;
        let mut probe = MlpWeakProbe::new(artifact).expect("valid artifact");

        let plane: Vec<f32> = (0..6 * n).map(|i| ((i % 19) as f32) * 0.2 - 1.5).collect();
        let mut layered = vec![0.0f32; 2 * 6 * n];
        layered[6 * n..].copy_from_slice(&plane); // slot 1 = layer 1
        let mut layered_out = vec![0.0f32; 2 * vocab];
        let ctx = ProbeCtx {
            tap: &layered,
            tap_layers: &[0, 1],
            tap_plane: 6 * n,
            ..probe_ctx(&plane, 2, 4, vocab, n)
        };
        probe.probe(ctx, &mut layered_out);

        let mut alone_out = vec![0.0f32; 2 * vocab];
        // The "alone" feed presents the same bytes as layer 1's plane — so
        // it must be DECLARED as layer 1 (a single-plane tap whose layer is
        // 1), not as the default layer-0 slot.
        let alone = ProbeCtx {
            tap_layers: &[1],
            ..probe_ctx(&plane, 2, 4, vocab, n)
        };
        probe.probe(alone, &mut alone_out);

        assert_eq!(
            layered_out, alone_out,
            "layer-1 artifact must read plane 1 of the layered tap, ignoring plane 0"
        );
    }

    #[test]
    fn probe_recomputes_direct_connector_head_math() {
        let artifact = fixture(7, 8, 0);
        let vocab = artifact.vocab;
        let n = artifact.mlp.n_embd;
        let mut probe = MlpWeakProbe::new(artifact).expect("valid artifact");

        // Two block positions (block_start = 2, seq_len = 4) over a 6-row tap.
        let tap: Vec<f32> = (0..6 * n).map(|i| ((i % 19) as f32) * 0.2 - 1.5).collect();
        let mut out = vec![0.0f32; 2 * vocab];
        probe.probe(probe_ctx(&tap, 2, 4, vocab, n), &mut out);

        for (row, p) in (2..4).enumerate() {
            let h = &tap[p * n..(p + 1) * n];
            let latent = probe.artifact.mlp.forward(h, &[0.0; 8]);
            let mut expected = vec![0.0f32; vocab];
            matmul(&mut expected, &probe.artifact.lm_head, &latent, vocab, n);
            assert_eq!(
                out[row * vocab..(row + 1) * vocab],
                expected,
                "row {row} (position {p}) must equal the direct forward+matmul"
            );
        }
    }

    #[test]
    fn probe_honors_block_start_offset() {
        let artifact = fixture(5, 8, 0);
        let vocab = artifact.vocab;
        let n = artifact.mlp.n_embd;
        let mut probe = MlpWeakProbe::new(artifact).expect("valid artifact");

        let tap: Vec<f32> = (0..6 * n).map(|i| ((i % 7) as f32) * 0.3).collect();
        let mut out = vec![0.0f32; (6 - 2) * vocab];
        probe.probe(probe_ctx(&tap, 2, 6, vocab, n), &mut out);

        // Row 0 must be position 2's logits, NOT position 0's.
        let h2 = &tap[2 * n..3 * n];
        let latent2 = probe.artifact.mlp.forward(h2, &[0.0; 8]);
        let mut expected2 = vec![0.0f32; vocab];
        matmul(&mut expected2, &probe.artifact.lm_head, &latent2, vocab, n);
        assert_eq!(
            out[..vocab],
            expected2,
            "first output row must be block_start's position"
        );
    }

    #[test]
    fn from_bin_roundtrip_produces_identical_probe_output() {
        let artifact = fixture(6, 8, 0);
        let vocab = artifact.vocab;
        let n = artifact.mlp.n_embd;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("weak_probe_test.bin");
        artifact.save_to_bin(&path).expect("save");

        let mut direct = MlpWeakProbe::new(artifact).expect("wrap");
        let mut loaded = MlpWeakProbe::from_bin(&path).expect("load");

        let tap: Vec<f32> = (0..4 * n).map(|i| ((i % 11) as f32) * 0.25).collect();
        let mut a = vec![0.0f32; 4 * vocab];
        let mut b = vec![0.0f32; 4 * vocab];
        direct.probe(probe_ctx(&tap, 0, 4, vocab, n), &mut a);
        loaded.probe(probe_ctx(&tap, 0, 4, vocab, n), &mut b);
        assert_eq!(a, b, "file roundtrip must not move a single logit");
    }

    #[test]
    fn probe_is_stable_across_calls_zero_alloc_contract() {
        // The same tap must produce the same logits on every call — the
        // reused scratch must be fully overwritten per position, never carry
        // state between calls.
        let artifact = fixture(5, 8, 0);
        let vocab = artifact.vocab;
        let n = artifact.mlp.n_embd;
        let mut probe = MlpWeakProbe::new(artifact).expect("valid artifact");

        let tap: Vec<f32> = (0..3 * n).map(|i| ((i % 13) as f32) * 0.4 - 1.0).collect();
        let mut a = vec![0.0f32; 3 * vocab];
        let mut b = vec![9.5f32; 3 * vocab]; // pre-poisoned: every slot must be overwritten
        probe.probe(probe_ctx(&tap, 0, 3, vocab, n), &mut a);
        probe.probe(probe_ctx(&tap, 0, 3, vocab, n), &mut b);
        assert_eq!(a, b, "probe must be a pure function of the tap");
        assert!(
            b.iter().all(|&v| v != 9.5),
            "every output slot must be written (no poison survives)"
        );
    }

    // ── DropoutHeadProbe (Issue 865 arm (b) substrate) ──

    #[test]
    fn dropout_probe_is_deterministic_and_step_sensitive() {
        let (vocab, n) = (6usize, 8usize);
        let head: Vec<f32> = (0..vocab * n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let tap: Vec<f32> = (0..4 * n).map(|i| ((i % 17) as f32) * 0.2 - 1.2).collect();
        let run = |step: usize| -> Vec<f32> {
            let mut probe = DropoutHeadProbe::new(vocab, n, head.clone(), 0.5);
            let mut out = vec![f32::NAN; 2 * vocab];
            let mut ctx = probe_ctx(&tap, 2, 4, vocab, n);
            ctx.step = step;
            probe.probe(ctx, &mut out);
            out
        };
        // Same (position, step) → same mask → identical logits, run to run.
        assert_eq!(run(2), run(2), "the mask must be a fixed stream");
        // A different step re-keys the mask → observably different logits.
        assert_ne!(run(2), run(3), "step must re-key the dropout mask");
    }

    #[test]
    fn dropout_probe_zero_rate_is_head_on_tap() {
        // drop_rate = 0 degenerates to `head · tap` — the anchor between this
        // probe and the plain linear readout.
        let (vocab, n) = (6usize, 8usize);
        let head: Vec<f32> = (0..vocab * n).map(|i| (i as f32) * 0.25).collect();
        let tap: Vec<f32> = (0..4 * n).map(|i| (i as f32) * 0.15).collect();
        let mut probe = DropoutHeadProbe::new(vocab, n, head.clone(), 0.0);
        let mut out = vec![f32::NAN; 4 * vocab];
        probe.probe(probe_ctx(&tap, 0, 4, vocab, n), &mut out);
        for p in 0..4 {
            let mut expected = vec![0.0f32; vocab];
            matmul(&mut expected, &head, &tap[p * n..(p + 1) * n], vocab, n);
            assert_eq!(
                out[p * vocab..(p + 1) * vocab],
                expected,
                "zero-rate row {p} must equal head·tap"
            );
        }
    }

    #[test]
    fn dropout_probe_overwrites_every_slot_and_purifies_noise() {
        // Every output slot written (pre-poisoned buffer) + the mask actually
        // drops a nonzero fraction of the tap at rate 0.5.
        let (vocab, n) = (5usize, 8usize);
        let head: Vec<f32> = vec![0.5; vocab * n];
        let tap: Vec<f32> = vec![2.0; 4 * n];
        let mut probe = DropoutHeadProbe::new(vocab, n, head, 0.5);
        let mut out = vec![9.5f32; 4 * vocab];
        probe.probe(probe_ctx(&tap, 0, 4, vocab, n), &mut out);
        assert!(out.iter().all(|&v| v != 9.5), "every slot must be written");
        // With head=0.5 rows, logits = 0.5 · Σ masked[i]; a full mask row
        // (none dropped) would give 0.5·(8·2.0)=8.0, all-dropped 0.0 — the
        // observed spread proves the per-position masks differ.
        let rows: Vec<f32> = (0..4).map(|p| out[p * vocab]).collect();
        assert!(
            rows.iter().any(|&r| r != rows[0]),
            "mask must vary across positions, got uniform {rows:?}"
        );
    }
}

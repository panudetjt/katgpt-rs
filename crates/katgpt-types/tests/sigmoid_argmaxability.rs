//! Issue 581 T3 — exhaustive sigmoid argmaxability audit.
//!
//! Grivas, Vergari & Lopez (AAAI 2024, [arXiv:2310.10443](https://arxiv.org/abs/2310.10443))
//! prove that a **low-rank sigmoid output layer** makes exponentially many label
//! combinations *unargmaxable* — unreachable for any input, at any weight values.
//! Their fix is a DFT output layer.
//!
//! Our affect bridge is exactly that shape: a flattened HLA is projected onto
//! 5–6 emotion directions (valence / arousal / desperation / calm / fear, plus
//! anger behind `civ_emotion`) and each output is sigmoid-gated. The 5 synced
//! scalars are the only affect state that crosses the sync boundary, so an
//! unreachable combination would be a permanently unreachable NPC affect state.
//!
//! `L ≤ 6` makes the audit **exhaustive** — 32 or 64 combinations is a complete
//! enumeration, not a sample. That is why Issue 581 predicted this would be a
//! cheap definitive answer either way.
//!
//! Run with output:
//! ```text
//! cargo test -p katgpt-types --features sigmoid_margin \
//!     --test sigmoid_argmaxability -- --nocapture
//! ```

#![cfg(feature = "sigmoid_margin")]

use katgpt_types::simd::{audit_argmaxable, matrix_rank};

/// Deterministic xorshift64* — no dev dependency, reproducible directions.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn random_matrix(l: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..l * d).map(|_| rng.next_f32()).collect()
}

// ── The audit: our actual affect-bridge shape ──

/// **The T3 deliverable.** `L = 5` synced affect scalars from a `d = 8` HLA.
/// Exhaustive over all 32 combinations.
#[test]
fn affect_bridge_5_of_8_all_combinations_argmaxable() {
    const L: usize = 5; // valence, arousal, desperation, calm, fear
    const D: usize = 8; // BELIEF_DIM / EMBED_DIM

    // Multiple independent direction sets — the conclusion must not hinge on one
    // lucky draw.
    for seed in 0..16u64 {
        let w = random_matrix(L, D, 0x00AF_FEC7 ^ seed);
        let audit = audit_argmaxable(&w, L, D);
        assert_eq!(
            audit.total, 32,
            "L=5 must enumerate exactly 32 combinations"
        );
        assert_eq!(
            audit.rank, L,
            "seed {seed}: expected full row rank {L}, got {}",
            audit.rank
        );
        assert!(
            audit.full_rank_proof,
            "seed {seed}: rank == L should yield a constructive proof"
        );
        assert_eq!(
            audit.unresolved, 0,
            "seed {seed}: {} of 32 affect combinations unreachable",
            audit.unresolved
        );
        assert_eq!(audit.achievable, 32);
    }

    println!(
        "\n=== Issue 581 T3 — affect bridge argmaxability (EXHAUSTIVE) ===\n\
         Shape: L=5 sigmoid-gated affect scalars from d=8 HLA input.\n\
         Result: rank(W) = 5 = L for all 16 direction sets tested.\n\
         => W has a right inverse, so W·x = 2y−1 is solvable for EVERY y.\n\
         => all 32 affect sign combinations are argmaxable. NO EXPOSURE.\n\
         This is a proof, not a sample: 32 combinations IS the whole space.\n"
    );
}

/// Same conclusion with `anger` included (`civ_emotion`): `L = 6`, 64 combinations.
#[test]
fn affect_bridge_6_of_8_all_combinations_argmaxable() {
    const L: usize = 6; // + anger
    const D: usize = 8;
    for seed in 0..16u64 {
        let w = random_matrix(L, D, 0x00A4_6E12 ^ seed);
        let audit = audit_argmaxable(&w, L, D);
        assert_eq!(audit.total, 64);
        assert_eq!(audit.rank, L, "seed {seed}: expected full row rank");
        assert_eq!(
            audit.unresolved, 0,
            "seed {seed}: unreachable combinations found"
        );
    }
    println!("L=6 (with anger) from d=8: rank 6 = L, all 64 combinations argmaxable.\n");
}

/// The condition that actually protects us, stated as a test: exposure requires
/// `rank(W) < L`, which for our layer means either more emotions than input
/// dimensions (`L > d`) or collinear direction vectors. Neither holds today.
#[test]
fn exposure_requires_rank_deficiency() {
    const D: usize = 8;
    // Every L up to d is safe with generic directions.
    for l in 1..=D {
        let w = random_matrix(l, D, 0x005A_FE00 ^ l as u64);
        assert_eq!(
            matrix_rank(&w, l, D, 1e-6),
            l,
            "L={l} should be full rank at d={D}"
        );
    }
    // Past d, rank saturates at d and the bottleneck becomes unavoidable.
    let w = random_matrix(12, D, 0xDEAD);
    assert_eq!(
        matrix_rank(&w, 12, D, 1e-6),
        D,
        "rank cannot exceed min(L, d) = 8"
    );
}

// ── Detector validity: the audit must catch REAL bottlenecks ──

/// A collinear direction makes a specific combination provably impossible: if
/// `a₄ = a₀ + a₁` then `⟨a₄,x⟩ = ⟨a₀,x⟩ + ⟨a₁,x⟩`, so demanding
/// `a₀ > 0, a₁ > 0, a₄ < 0` is unsatisfiable. If the audit reported "all clear"
/// here it would be vacuous, and the passing tests above would mean nothing.
#[test]
fn detects_unargmaxable_combination_under_collinearity() {
    const L: usize = 5;
    const D: usize = 8;
    let mut w = random_matrix(L, D, 0xBAD_C0DE);
    // Row 4 := row 0 + row 1  →  rank drops to 4 < L.
    for j in 0..D {
        w[4 * D + j] = w[j] + w[D + j];
    }
    let audit = audit_argmaxable(&w, L, D);
    assert_eq!(audit.rank, 4, "collinear row must drop the rank to 4");
    assert!(!audit.full_rank_proof, "rank < L must not claim a proof");
    assert!(
        audit.unresolved > 0,
        "the audit must flag at least the (+,+,·,·,−) family as unreachable; \
         reporting all-clear here would make the affect-bridge result vacuous"
    );
    assert!(
        audit.achievable < audit.total,
        "achievable {} should be < total {}",
        audit.achievable,
        audit.total
    );
    println!(
        "Detector validity: collinear row → rank 4 < L=5, {} of {} combinations \
         unreachable (expected > 0).\n",
        audit.unresolved, audit.total
    );
}

/// The regime the paper is actually about: far more labels than input features.
/// Here the majority of combinations should be unreachable.
#[test]
fn detects_severe_bottleneck_when_labels_far_exceed_dims() {
    const L: usize = 12;
    const D: usize = 3;
    let w = random_matrix(L, D, 0x0051_061D);
    let audit = audit_argmaxable(&w, L, D);
    assert_eq!(audit.rank, D, "rank saturates at d=3");
    assert!(
        audit.unresolved > audit.total / 2,
        "expected most of {} combinations unreachable at L=12, d=3; got {}",
        audit.total,
        audit.unresolved
    );
    println!(
        "Paper's regime (L=12 >> d=3): {} of {} combinations unreachable ({:.0}%).\n",
        audit.unresolved,
        audit.total,
        100.0 * audit.unresolved as f64 / audit.total as f64
    );
}

#[test]
fn matrix_rank_basics() {
    // Identity-ish: 3 orthogonal rows in R^4 → rank 3.
    let w = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    assert_eq!(matrix_rank(&w, 3, 4, 1e-6), 3);
    // Duplicated row → rank 1.
    let w = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    assert_eq!(matrix_rank(&w, 2, 3, 1e-6), 1);
    // Zero matrix → rank 0.
    let w = [0.0f32; 12];
    assert_eq!(matrix_rank(&w, 3, 4, 1e-6), 0);
}

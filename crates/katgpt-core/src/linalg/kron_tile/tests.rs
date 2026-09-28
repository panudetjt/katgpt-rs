//! Issue 839 gates for [`super`] — G1 correctness, the WHT-parity pin (T2),
//! the contract refusals, and the G4 zero-alloc witness.
//!
//! G2 (throughput against dense and against the butterfly) lives in
//! `tests/bench_839_kron_tile_goat.rs`, because a latency bar belongs in a
//! `--release` target per `AGENTS.md`, not in a debug lib test.

use super::*;

/// Deterministic LCG — the module-local convention in this crate's tests
/// (`data_probe::cca`, `salience::gate`, …). Seeded, never a global draw:
/// `scripts/global_rng_gate.py` reds on an unseeded `fastrand` here.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Uniform in `[-1, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let u = ((self.0 >> 33) as f32) / (1u64 << 31) as f32;
        2.0 * u - 1.0
    }
}

fn random_vec(rng: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next()).collect()
}

/// Max absolute deviation between two equal-length buffers.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// G1 — the convention IS the claim: `(A ⊗ B) · x ≡ A Z Bᵀ`
// ---------------------------------------------------------------------------

/// G1 at the smallest interesting width, against the explicit `n² × n²`
/// Kronecker product. This is what pins the row-major convention derived in the
/// module doc; get the convention wrong and this test fails by O(1), not by a
/// tolerance.
#[test]
fn kron_apply_matches_dense_reference_n8() {
    let n = 8usize;
    let nn = n * n;
    let mut rng = Lcg::new(0x0839_0001);
    let a = random_vec(&mut rng, nn);
    let b = random_vec(&mut rng, nn);
    let x0 = random_vec(&mut rng, nn);

    let mut dense = vec![0.0f32; nn * nn];
    kron_dense_into(&a, &b, n, &mut dense);
    let mut want = vec![0.0f32; nn];
    dense_matvec_into(&dense, nn, &x0, &mut want);

    let mut got = x0.clone();
    let mut s = KronScratch::with_capacity(n);
    kron_apply(&a, &b, n, 1, &mut got, &mut s);

    let d = max_abs_diff(&got, &want);
    assert!(
        d <= 1e-5,
        "G1 FAIL (n=8): tile apply vs dense (A⊗B) max |Δ| = {d:e} > 1e-5"
    );
}

/// G1 at the width Research 569 actually uses (1024 channels as a 32×32 tile).
/// Kept separate from the `n = 8` arm so a failure names the width.
#[test]
fn kron_apply_matches_dense_reference_n32() {
    let n = 32usize;
    let nn = n * n;
    let mut rng = Lcg::new(0x0839_0020);
    let a = random_vec(&mut rng, nn);
    let b = random_vec(&mut rng, nn);
    let x0 = random_vec(&mut rng, nn);

    let mut dense = vec![0.0f32; nn * nn];
    kron_dense_into(&a, &b, n, &mut dense);
    let mut want = vec![0.0f32; nn];
    dense_matvec_into(&dense, nn, &x0, &mut want);

    let mut got = x0.clone();
    let mut s = KronScratch::with_capacity(n);
    kron_apply(&a, &b, n, 1, &mut got, &mut s);

    // Scale-relative: `want` entries are sums of 1024 products, so an absolute
    // 1e-5 would be a claim about the fixture's magnitude rather than about the
    // kernel. The denominator is floored at 1 so a near-zero entry cannot
    // inflate the ratio.
    let rel = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(
        rel <= 1e-5,
        "G1 FAIL (n=32): tile apply vs dense (A⊗B) max scaled |Δ| = {rel:e} > 1e-5"
    );
}

/// The batched entry point must agree with the single-tile one tile for tile —
/// otherwise the in-place scratch reuse is silently carrying state between
/// tiles, which is the defect a batched kernel has and a single-tile one cannot.
#[test]
fn kron_apply_batched_matches_per_tile() {
    let n = 16usize;
    let nn = n * n;
    let tiles = 5usize;
    let mut rng = Lcg::new(0x0839_0005);
    let a = random_vec(&mut rng, nn);
    let b = random_vec(&mut rng, nn);
    let x0 = random_vec(&mut rng, tiles * nn);

    let mut batched = x0.clone();
    let mut s = KronScratch::with_capacity(n);
    kron_apply(&a, &b, n, tiles, &mut batched, &mut s);

    let mut per_tile = vec![0.0f32; tiles * nn];
    for t in 0..tiles {
        let (z, y) = (
            &x0[t * nn..(t + 1) * nn],
            &mut per_tile[t * nn..(t + 1) * nn],
        );
        kron_apply_tile_into(&a, &b, n, z, y, &mut s);
    }

    // Same code, same order, same inputs: this one IS exact.
    assert_eq!(
        batched, per_tile,
        "batched and per-tile paths must be bit-identical"
    );
}

// ---------------------------------------------------------------------------
// T2 — the WHT fast path against the generic path at the same operator
// ---------------------------------------------------------------------------

/// T2. The two paths compute `(W ⊗ W) · x` by different algorithms — an
/// `O(n³)` pair of GEMMs against an `O(n² log n)` butterfly — so this is a
/// two-implementation comparison, which is the only kind worth pinning.
///
/// ⚠ **Not bit-parity, and Issue 839 T2's wording asked for it.** The butterfly
/// sums pairwise in a `log n`-deep tree and applies one `1/√n` scale at the end;
/// the GEMM path reduces in `simd_dot_f32`'s per-ISA order over explicit
/// `±1/√n` entries. Different
/// summation orders over the same reals are not the same floating-point
/// program, so bit-parity is unachievable *by construction* rather than by
/// sloppiness, and pinning it would have produced a test that passes only on
/// the box that wrote it. The measured gap is asserted instead, at every
/// dispatched width.
#[test]
fn wht_fast_path_matches_generic_path() {
    let mut worst = 0.0f32;
    for &n in &WHT_TILE_WIDTHS[..4] {
        // 128 excluded: `kron_dense_into` is not needed here, but the generic
        // path at n=128 is 4M MACs/tile in a debug build — the n=8..64 arms
        // already cover every dispatch branch this test can distinguish.
        let nn = n * n;
        let tiles = 3usize;
        let mut rng = Lcg::new(0x0839_0200 + n as u64);
        let x0 = random_vec(&mut rng, tiles * nn);

        let mut w = vec![0.0f32; nn];
        wht_factor_into(n, &mut w);

        let mut generic = x0.clone();
        let mut s = KronScratch::with_capacity(n);
        kron_apply(&w, &w, n, tiles, &mut generic, &mut s);

        let mut fast = x0.clone();
        assert!(
            wht_apply_tiles(n, tiles, &mut fast, &mut s),
            "n={n} is in WHT_TILE_WIDTHS but has no dispatch arm"
        );

        let d = max_abs_diff(&fast, &generic);
        worst = worst.max(d);
        assert!(
            d <= 1e-5,
            "T2 FAIL (n={n}): WHT fast path vs generic max |Δ| = {d:e} > 1e-5"
        );
    }
    // Printed so the pin is a MEASUREMENT and a later tightening has a number
    // to tighten against, rather than the tolerance being folklore.
    println!("wht parity: worst max |Δ| over n∈{{8,16,32,64}} = {worst:e}");
}

/// T3's exact anchor: `W` is involutive (`W² = I`), so two applications of the
/// fast path return the input. Catches a scale error the parity test cannot —
/// both paths could share a wrong `1/√n` and still agree with each other.
#[test]
fn wht_apply_is_involutive() {
    let n = 32usize;
    let tiles = 2usize;
    let mut rng = Lcg::new(0x0839_0300);
    let x0 = random_vec(&mut rng, tiles * n * n);
    let mut x = x0.clone();
    let mut s = KronScratch::with_capacity(n);

    assert!(wht_apply_tiles(n, tiles, &mut x, &mut s));
    assert!(wht_apply_tiles(n, tiles, &mut x, &mut s));

    let d = max_abs_diff(&x, &x0);
    assert!(
        d <= 1e-6,
        "W⊗W applied twice must be the identity; max |Δ| = {d:e}"
    );
}

/// The reference factor is the operator the fast path claims to implement, so
/// its own properties are worth pinning: symmetric (`Bᵀ` is free) and
/// orthogonal (`W Wᵀ = I`).
#[test]
fn wht_factor_is_symmetric_and_orthogonal() {
    let n = 16usize;
    let mut w = vec![0.0f32; n * n];
    wht_factor_into(n, &mut w);

    for i in 0..n {
        for j in 0..n {
            assert_eq!(
                w[i * n + j],
                w[j * n + i],
                "W must be symmetric at ({i},{j})"
            );
        }
    }
    for i in 0..n {
        for j in 0..n {
            let dot = simd_dot_f32(&w[i * n..i * n + n], &w[j * n..j * n + n], n);
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - want).abs() <= 1e-6,
                "W rows {i},{j} dot to {dot}, want {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Contract refusals — each is a failure mode this repo has measured elsewhere
// ---------------------------------------------------------------------------

/// The `#[must_use] -> bool` contract: an undispatched width must write
/// NOTHING, so a caller that ignores the bool gets untransformed data rather
/// than half-transformed data.
#[test]
fn wht_apply_tiles_declines_unsupported_width_without_writing() {
    let n = 24usize; // not a power of two; no dispatch arm
    let tiles = 2usize;
    let mut rng = Lcg::new(0x0839_0400);
    let x0 = random_vec(&mut rng, tiles * n * n);
    let mut x = x0.clone();
    let mut s = KronScratch::with_capacity(n);

    assert!(
        !wht_apply_tiles(n, tiles, &mut x, &mut s),
        "n=24 must be declined, not silently approximated"
    );
    assert_eq!(x, x0, "a declined call must leave the buffer untouched");
}

/// The len-derived refusal (see the module doc). A buffer longer than its live
/// range is the *normal* way a scratch slice arrives; deriving `tiles` from it
/// would transform the dead tail and report success.
#[test]
#[should_panic(expected = "x must be exactly 2 tile(s)")]
fn kron_apply_refuses_a_buffer_longer_than_its_tile_count() {
    let n = 8usize;
    let nn = n * n;
    let a = vec![0.0f32; nn];
    let b = vec![0.0f32; nn];
    let mut over_long = vec![0.0f32; 3 * nn]; // capacity-sized scratch
    let mut s = KronScratch::with_capacity(n);
    kron_apply(&a, &b, n, 2, &mut over_long, &mut s);
}

/// An under-sized scratch must refuse rather than silently slice short.
#[test]
#[should_panic(expected = "call KronScratch::ensure")]
fn kron_apply_refuses_an_undersized_scratch() {
    let n = 16usize;
    let nn = n * n;
    let a = vec![0.0f32; nn];
    let b = vec![0.0f32; nn];
    let mut x = vec![0.0f32; nn];
    let mut s = KronScratch::with_capacity(8); // too small for n=16
    kron_apply(&a, &b, n, 1, &mut x, &mut s);
}

#[test]
fn scratch_ensure_grows_once_and_is_idempotent() {
    let mut s = KronScratch::default();
    assert_eq!(s.capacity_elems(), 0);
    s.ensure(32);
    let after = s.capacity_elems();
    assert!(after >= 32 * 32);
    s.ensure(32);
    assert_eq!(s.capacity_elems(), after, "re-ensure must not grow");
    s.ensure(8);
    assert_eq!(
        s.capacity_elems(),
        after,
        "a SMALLER ensure must not shrink"
    );
}

// ---------------------------------------------------------------------------
// The inter-stage permutation
// ---------------------------------------------------------------------------

/// The claim that motivates [`permute_into`] existing at all: without a
/// permutation, two Kronecker stages collapse to ONE with multiplied factors
/// (`(A₂⊗B₂)(A₁⊗B₁) = (A₂A₁)⊗(B₂B₁)`), so a permutation-free "multi-stage"
/// mixer is a single stage wearing three. Pinned rather than cited, because it
/// is the whole reason the 3-stage bench arm inserts Π.
#[test]
fn stages_collapse_without_a_permutation() {
    let n = 8usize;
    let nn = n * n;
    let mut rng = Lcg::new(0x0839_0600);
    let (a1, b1) = (random_vec(&mut rng, nn), random_vec(&mut rng, nn));
    let (a2, b2) = (random_vec(&mut rng, nn), random_vec(&mut rng, nn));
    let x0 = random_vec(&mut rng, nn);
    let mut s = KronScratch::with_capacity(n);

    // Two stages back to back, no permutation between them.
    let mut two = x0.clone();
    kron_apply(&a1, &b1, n, 1, &mut two, &mut s);
    kron_apply(&a2, &b2, n, 1, &mut two, &mut s);

    // One stage at the multiplied factors A2·A1, B2·B1.
    let mut a21 = vec![0.0f32; nn];
    let mut b21 = vec![0.0f32; nn];
    for i in 0..n {
        for j in 0..n {
            let mut sa = 0.0f32;
            let mut sb = 0.0f32;
            for k in 0..n {
                sa += a2[i * n + k] * a1[k * n + j];
                sb += b2[i * n + k] * b1[k * n + j];
            }
            a21[i * n + j] = sa;
            b21[i * n + j] = sb;
        }
    }
    let mut one = x0.clone();
    kron_apply(&a21, &b21, n, 1, &mut one, &mut s);

    let rel = one
        .iter()
        .zip(&two)
        .map(|(o, t)| (o - t).abs() / t.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(
        rel <= 1e-5,
        "two permutation-free stages must equal one stage at multiplied          factors; max scaled |Δ| = {rel:e}"
    );
}

/// `permute_into` is a gather (`dst[i] = src[perm[i]]`), and an inverse
/// permutation must undo it — which is what makes a fixed Π a relabelling
/// rather than a lossy shuffle.
#[test]
fn permute_into_is_a_gather_and_round_trips() {
    let m = 64usize;
    let src: Vec<f32> = (0..m).map(|i| i as f32).collect();
    // A deterministic non-identity permutation: an odd stride is a bijection
    // mod m for m even.
    let perm: Vec<u32> = (0..m).map(|i| ((i * 37 + 11) % m) as u32).collect();
    assert!(is_permutation(&perm, m), "the fixture must be a bijection");

    let mut fwd = vec![0.0f32; m];
    permute_into(&src, &perm, &mut fwd);
    for i in 0..m {
        assert_eq!(fwd[i], src[perm[i] as usize], "gather semantics at {i}");
    }
    assert_ne!(fwd, src, "the fixture must not be the identity");

    let mut inv = vec![0u32; m];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u32;
    }
    let mut back = vec![0.0f32; m];
    permute_into(&fwd, &inv, &mut back);
    assert_eq!(back, src, "Π⁻¹ ∘ Π must be the identity");
}

/// `is_permutation` must reject in-range non-bijections — the case
/// `permute_into` deliberately does NOT panic on, because catching it there
/// would cost an O(n²) pass per apply.
#[test]
fn is_permutation_rejects_in_range_non_bijections() {
    assert!(is_permutation(&[0, 1, 2, 3], 4));
    assert!(
        !is_permutation(&[0, 1, 1, 3], 4),
        "a duplicate is not a bijection"
    );
    assert!(!is_permutation(&[0, 1, 2, 9], 4), "out of range");
    assert!(!is_permutation(&[0, 1, 2], 4), "wrong length");
    assert!(
        is_permutation(&[], 0),
        "the empty permutation is vacuously valid"
    );
}

// ---------------------------------------------------------------------------
// G4 — zero allocations in steady state
// ---------------------------------------------------------------------------

/// G4. The full Issue-741 predicate, not a bare `debug_assertions`: gating a
/// *measurement* on the profile makes it unrunnable in `--release`, which is
/// the configuration this claim is about. Under
/// `--release --features alloc_tracking` this test RUNS; at release-default it
/// compiles away instead of breaking the harness.
#[test]
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
fn g4_zero_alloc_steady_state() {
    use crate::alloc::{get_alloc_stats, reset_alloc_stats};

    let n = 32usize;
    let nn = n * n;
    let tiles = 8usize;
    let mut rng = Lcg::new(0x0839_0900);
    // Orthogonal factors on purpose: 50 rounds of a random `A` would drive the
    // buffer to ±inf and measure the allocator against NaNs. `W` is
    // norm-preserving and the generic path does not know it is special.
    let mut a = vec![0.0f32; nn];
    wht_factor_into(n, &mut a);
    let b = a.clone();
    let mut x = random_vec(&mut rng, tiles * nn);
    let mut s = KronScratch::with_capacity(n);

    // Liveness sentinel: without it a missing TrackingAllocator reports a
    // perfect zero, which is the same output as success.
    reset_alloc_stats();
    let sentinel: Vec<u8> = vec![0u8; 256];
    let (sent, _) = get_alloc_stats();
    if sent == 0 {
        eprintln!("g4_zero_alloc_steady_state: TrackingAllocator not installed — SKIPPED");
        return;
    }
    drop(sentinel);

    kron_apply(&a, &b, n, tiles, &mut x, &mut s); // warm the scratch
    reset_alloc_stats();
    for _ in 0..50 {
        kron_apply(&a, &b, n, tiles, &mut x, &mut s);
        let applied = wht_apply_tiles(n, tiles, &mut x, &mut s);
        assert!(applied, "n=32 must dispatch");
    }
    let (count, bytes) = get_alloc_stats();
    assert_eq!(
        count, 0,
        "kron_apply + wht_apply_tiles must be alloc-free in steady state; \
         observed {count} allocation(s) ({bytes} bytes) over 50 rounds"
    );
}

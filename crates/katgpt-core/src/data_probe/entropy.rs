//! kNN differential-entropy estimator (Kozachenko–Leonenko) — Issue 708 P1.
//!
//! `Ĥ ≈ ψ(n) − ψ(k) + ln c_d + (d/n) Σᵢ ln εₖ(i)` over point-latent
//! populations (belief states, emotion fields, span embeddings, fix
//! trajectories) — the uncertainty/entropy axis the shipped dispersion
//! proxies ([`crate::data_probe::geometry::effective_rank`], anisotropy,
//! gaussianity, spectral flatness) do not cover. Classic prior art
//! (Kozachenko & Leonenko 1987; survey: Beirlant et al. 1997) — an IMPORT,
//! not an invention; zero corpus hits at filing (Issue 708 P1).
//!
//! # Form
//!
//! For `n` points in `R^d`, `εₖ(i)` = Euclidean distance from point `i` to
//! its k-th nearest neighbour among the other `n−1` points:
//!
//! ```text
//! Ĥ = ψ(n) − ψ(k) + ln c_d + (d/n) · Σᵢ ln εₖ(i)
//! ```
//!
//! `c_d` is the volume of the Euclidean unit ball (`ln c_d` computed in log
//! space by direct products — no `Γ` evaluation, no overflow at any `d`).
//! `ψ` is the digamma function (recurrence to ≥ 12, then the asymptotic
//! series; ~1e-12 across the used range, pinned by known-answer tests).
//!
//! # Contract
//!
//! - **Brute-force kNN is deliberate**: the estimator targets OFFLINE
//!   populations (audit cadence, n ≤ a few thousand). O(n²·d) time,
//!   O(k) scratch. Not a hot-path primitive.
//! - **Deterministic**: fixed `(points, n, d, k)` ⇒ bit-identical Ĥ
//!   (iteration-order tie-breaking; no RNG).
//! - **Zero steady-state allocation** (G4 by construction): the bounded
//!   max-heap lives in [`KnnEntropyScratch`], sized once at `new`.
//! - **Duplicates ⇒ −∞**: if any k-NN distance is exactly 0 (a true point
//!   mass), `ln ε` → −∞ and Ĥ → −∞. That IS the collapse signal; the P2
//!   monitor (Issue 708) is the component expected to interpret it — this
//!   fn returns the estimator's honest value rather than clamping.
//! - **nats**, f64 accumulators throughout (matches the gaussianity probe's
//!   f64 discipline).
//!
//! # Consumers
//!
//! P1 is the substrate; P2 (two-channel imbalance collapse monitor) and the
//! `edge_lora_dist_guard` third axis are the intended consumers. Opt-in
//! `knn_entropy` until one lands (no-default-consumer rule).

// ──────────────────────────────────────────────────────────────────────────
// Scratch
// ──────────────────────────────────────────────────────────────────────────

/// Caller-owned scratch: one bounded max-heap of the k smallest distances
/// seen so far, allocated once and reused across calls (the
/// [`crate::data_probe::gaussianity::GaussianityScratch`] single-scratch
/// discipline).
pub struct KnnEntropyScratch {
    heap: Box<[f64]>,
    k: usize,
}

impl KnnEntropyScratch {
    /// Construct the scratch for a given k. `k` is the neighbour order of the
    /// estimator (k = 1..=8 covers the usual calibration band; larger k
    /// lowers variance at higher bias).
    pub fn new(k: usize) -> Self {
        assert!(k >= 1, "k must be >= 1");
        Self {
            heap: vec![0.0f64; k].into_boxed_slice(),
            k,
        }
    }

    /// Neighbour order the scratch was built for.
    pub fn k(&self) -> usize {
        self.k
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Core estimator
// ──────────────────────────────────────────────────────────────────────────

/// Kozachenko–Leonenko kNN differential-entropy estimate, in nats.
///
/// `points` is row-major `[n × d]`. Panics if `n <= k` (the estimator needs
/// at least k neighbours besides the point itself), `k == 0`, `d == 0`, or
/// the scratch's `k` mismatches.
pub fn knn_differential_entropy(
    points: &[f32],
    n: usize,
    d: usize,
    k: usize,
    scratch: &mut KnnEntropyScratch,
) -> f64 {
    assert!(n > k, "n ({n}) must exceed k ({k})");
    assert!(d > 0, "d must be positive");
    assert_eq!(points.len(), n * d, "points.len() must be n × d");
    assert_eq!(scratch.k(), k, "scratch built for a different k");

    let heap = &mut scratch.heap[..];

    let mut sum_ln_eps = 0.0f64;
    for i in 0..n {
        let eps_k = kth_nn_distance(points, n, d, i, k, heap);
        // Duplicates (ε = 0) propagate to −∞ by contract (see module docs).
        sum_ln_eps += eps_k.ln();
    }

    digamma(n as f64) - digamma(k as f64)
        + ln_euclidean_unit_ball_volume(d)
        + (d as f64 / n as f64) * sum_ln_eps
}

/// Distance from point `i` to its k-th nearest neighbour among `j ≠ i`,
/// via a bounded max-heap of the k smallest distances seen (O(n·d + n·log k)).
/// `heap` is caller-owned scratch of length ≥ k; its contents are garbage
/// on both entry and exit.
fn kth_nn_distance(
    points: &[f32],
    n: usize,
    d: usize,
    i: usize,
    k: usize,
    heap: &mut [f64],
) -> f64 {
    let row_i = &points[i * d..(i + 1) * d];
    let mut heap_len = 0usize;

    for j in 0..n {
        if j == i {
            continue;
        }
        let row_j = &points[j * d..(j + 1) * d];
        let mut acc = 0.0f64;
        for t in 0..d {
            let diff = (row_i[t] as f64) - (row_j[t] as f64);
            acc += diff * diff;
        }
        let dist = acc.sqrt();

        if heap_len < k {
            // Fill phase: standard max-heap INSERT (sift-up) so the invariant
            // holds from the first element on — a fill-then-single-sift-down
            // does NOT heapify (the invariant must hold at every step for the
            // replace-root path below to be correct).
            heap[heap_len] = dist;
            sift_up_max_heap(heap, heap_len);
            heap_len += 1;
        } else if dist < heap[0] {
            // Full: replace the current k-th smallest (the root) + restore.
            heap[0] = dist;
            sift_down_max_heap(heap, 0, k);
        }
    }

    debug_assert_eq!(heap_len, k, "n > k guarantees the heap fills");
    heap[0]
}

/// Bubble `heap[idx]` up while larger than its parent (max-heap insert).
#[inline]
fn sift_up_max_heap(heap: &mut [f64], mut idx: usize) {
    while idx > 0 {
        let parent = (idx - 1) / 2;
        if heap[parent] >= heap[idx] {
            return;
        }
        heap.swap(parent, idx);
        idx = parent;
    }
}

/// Sift the root down within `heap[..len]` (max-heap invariant after a
/// root replacement).
#[inline]
fn sift_down_max_heap(heap: &mut [f64], root: usize, len: usize) {
    let mut pos = root;
    loop {
        let left = 2 * pos + 1;
        if left >= len {
            return;
        }
        let child = if left + 1 < len && heap[left + 1] > heap[left] {
            left + 1
        } else {
            left
        };
        if heap[pos] >= heap[child] {
            return;
        }
        heap.swap(pos, child);
        pos = child;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Constants — ln(unit-ball volume) and digamma
// ──────────────────────────────────────────────────────────────────────────

/// `ln c_d` for the Euclidean unit ball in `R^d`, computed in log space by
/// direct products (exact recurrence halves, no Γ evaluation, no overflow):
///
/// - `d = 2m`:   `ln c_d = m·ln π − Σ_{j=1..m} ln j`
/// - `d = 2m+1`: `ln c_d = m·ln π + (m+1)·ln 2 − Σ_{odd j ≤ 2m+1} ln j`
pub fn ln_euclidean_unit_ball_volume(d: usize) -> f64 {
    let ln_pi = std::f64::consts::PI.ln();
    let m = d / 2;
    if d.is_multiple_of(2) {
        let mut acc = m as f64 * ln_pi;
        for j in 1..=m {
            acc -= (j as f64).ln();
        }
        acc
    } else {
        let mut acc = m as f64 * ln_pi + (m + 1) as f64 * std::f64::consts::LN_2;
        let mut j = 1usize;
        while j <= 2 * m + 1 {
            acc -= (j as f64).ln();
            j += 2;
        }
        acc
    }
}

/// Digamma ψ(x) = d/dx ln Γ(x), accurate to ~1e-12 for x > 0:
/// recurrence `ψ(x) = ψ(x+1) − 1/x` pushes the argument to ≥ 12, then the
/// asymptotic series (Abramowitz & Stegun 6.3.18 with two extra terms; the
/// first dropped term is ≤ ~1e-14 at z = 12).
pub fn digamma(x: f64) -> f64 {
    assert!(x > 0.0, "digamma requires x > 0, got {x}");
    let mut acc = 0.0f64;
    let mut z = x;
    while z < 12.0 {
        acc -= 1.0 / z;
        z += 1.0;
    }
    let inv = 1.0 / z;
    let inv2 = inv * inv;
    // ψ(z) ≈ ln z − 1/(2z) − 1/(12z²) + 1/(120z⁴) − 1/(252z⁶) + 1/(240z⁸)
    acc + z.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 → f64 in [0, 1) (24-bit resolution — plenty
    /// for population fixtures).
    struct Lcg(u64);
    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f64 / (1u64 << 24) as f64
        }
    }

    /// Box–Muller Gaussian sample (deterministic, no external deps).
    /// `u1 = 1 − unit` keeps u1 in (0, 1] so the log never sees 0.
    fn gauss(rng: &mut Lcg) -> f64 {
        let u1 = 1.0 - rng.unit();
        let u2 = rng.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Isotropic Gaussian population `[n × d]`, per-axis σ = sigma.
    fn gaussian_population(n: usize, d: usize, sigma: f64, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed | 1);
        let mut pts = Vec::with_capacity(n * d);
        for _ in 0..n * d {
            pts.push((gauss(&mut rng) * sigma) as f32);
        }
        pts
    }

    // ── Constants ─────────────────────────────────────────────────────

    #[test]
    fn digamma_matches_known_values() {
        // ψ(1) = −γ, ψ(0.5) = −γ − 2 ln 2, ψ(6) = H(5) − γ (A&S 6.3.x table).
        // std's exact constant (newer clippy denies hand-typed approximations —
        // the approx_constant deny-by-default class).
        let gamma = std::f64::consts::EULER_GAMMA;
        assert!(
            (digamma(1.0) - (-gamma)).abs() < 1e-12,
            "ψ(1) = {}",
            digamma(1.0)
        );
        assert!(
            (digamma(0.5) - (-gamma - 2.0 * std::f64::consts::LN_2)).abs() < 1e-12,
            "ψ(0.5) = {}",
            digamma(0.5)
        );
        let h5 = 1.0 + 1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0 + 1.0 / 5.0;
        assert!(
            (digamma(6.0) - (h5 - gamma)).abs() < 1e-12,
            "ψ(6) = {}",
            digamma(6.0)
        );
    }

    #[test]
    fn ln_unit_ball_volume_matches_closed_forms() {
        // c₁ = 2, c₂ = π, c₃ = 4π/3, c₄ = π²/2, c₅ = 8π²/15
        let cases: [(usize, f64); 5] = [
            (1, 2.0f64.ln()),
            (2, std::f64::consts::PI.ln()),
            (3, (4.0 * std::f64::consts::PI / 3.0).ln()),
            (4, (std::f64::consts::PI * std::f64::consts::PI / 2.0).ln()),
            (
                5,
                (8.0 * std::f64::consts::PI * std::f64::consts::PI / 15.0).ln(),
            ),
        ];
        for (d, expect) in cases {
            assert!(
                (ln_euclidean_unit_ball_volume(d) - expect).abs() < 1e-12,
                "ln c_{d}: {} vs {expect}",
                ln_euclidean_unit_ball_volume(d)
            );
        }
    }

    // ── G1: closed-form calibration (isotropic Gaussian) ─────────────

    /// Isotropic Gaussian: H = ½ ln((2πe)^d ∏σ²) = (d/2)(1 + ln 2π + 2 ln σ).
    fn gaussian_entropy_closed_form(d: usize, sigma: f64) -> f64 {
        let ln_2pi = std::f64::consts::TAU.ln();
        0.5 * d as f64 * (1.0 + ln_2pi + 2.0 * sigma.ln())
    }

    #[test]
    fn gaussian_calibration_d4() {
        let (n, d, k) = (1024usize, 4usize, 5usize);
        let mut scratch = KnnEntropyScratch::new(k);
        let pts = gaussian_population(n, d, 1.0, 42);
        let h = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        let expect = gaussian_entropy_closed_form(d, 1.0);
        assert!(
            (h - expect).abs() < 0.35,
            "KL estimate {h:.4} vs closed form {expect:.4} (tol 0.35)"
        );
    }

    #[test]
    fn gaussian_calibration_d8() {
        let (n, d, k) = (1024usize, 8usize, 5usize);
        let mut scratch = KnnEntropyScratch::new(k);
        let pts = gaussian_population(n, d, 1.0, 43);
        let h = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        let expect = gaussian_entropy_closed_form(d, 1.0);
        // Higher d → slower KL convergence; looser band, still decisive vs a
        // wrong-axis substitute (the dispersion proxies cannot produce nats).
        assert!(
            (h - expect).abs() < 0.9,
            "KL estimate {h:.4} vs closed form {expect:.4} (tol 0.9)"
        );
    }

    // ── G1: monotone under shrink-to-point ───────────────────────────

    #[test]
    fn monotone_under_shrink() {
        let (n, d, k) = (512usize, 4usize, 3usize);
        let mut scratch = KnnEntropyScratch::new(k);
        let mut prev = f64::INFINITY;
        for sigma in [2.0f64, 1.0, 0.25] {
            let pts = gaussian_population(n, d, sigma, 44);
            let h = knn_differential_entropy(&pts, n, d, k, &mut scratch);
            assert!(
                h < prev,
                "entropy must decrease as σ shrinks: σ={sigma} h={h:.4} prev={prev:.4}"
            );
            // Each arm also sits near its own closed form.
            let expect = gaussian_entropy_closed_form(d, sigma);
            assert!(
                (h - expect).abs() < 0.5,
                "σ={sigma}: estimate {h:.4} vs closed form {expect:.4}"
            );
            prev = h;
        }
    }

    // ── G1: planted rank-1 collapse trips ────────────────────────────

    #[test]
    fn planted_collapse_trips() {
        let (n, d, k) = (512usize, 4usize, 3usize);
        let mut scratch = KnnEntropyScratch::new(k);
        let healthy = gaussian_population(n, d, 1.0, 45);
        let collapsed = gaussian_population(n, d, 1e-3, 45); // same seed: identical shape
        let h_ok = knn_differential_entropy(&healthy, n, d, k, &mut scratch);
        let h_col = knn_differential_entropy(&collapsed, n, d, k, &mut scratch);
        // Expected drop ≈ d·ln(1e-3) ≈ −27.6 nats; assert the detector trips
        // by a wide, mechanism-level margin (the planted_rank1 pattern).
        assert!(
            h_col < h_ok - 15.0,
            "collapse must trip: healthy {h_ok:.3} vs collapsed {h_col:.3}"
        );
    }

    // ── Contract edges ───────────────────────────────────────────────

    #[test]
    fn duplicate_points_negative_infinity() {
        let (n, d, k) = (8usize, 2usize, 1usize);
        let mut pts = gaussian_population(n, d, 1.0, 46);
        let src: Vec<f32> = pts[..d].to_vec();
        pts[3 * d..4 * d].copy_from_slice(&src); // exact duplicate
        let mut scratch = KnnEntropyScratch::new(k);
        let h = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        assert!(
            h.is_infinite() && h < 0.0,
            "a true point mass must read −∞, got {h}"
        );
    }

    #[test]
    fn determinism_x3_bit_identical() {
        let (n, d, k) = (256usize, 4usize, 3usize);
        let pts = gaussian_population(n, d, 1.0, 47);
        let mut scratch = KnnEntropyScratch::new(k);
        let h1 = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        let h2 = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        let h3 = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        assert_eq!(h1, h2, "bit-identical across runs (1,2)");
        assert_eq!(h2, h3, "bit-identical across runs (2,3)");
    }

    #[test]
    fn scratch_is_reused_and_shape_checked() {
        let (n, d, k) = (128usize, 4usize, 3usize);
        let pts = gaussian_population(n, d, 1.0, 48);
        let mut scratch = KnnEntropyScratch::new(k);
        let _ = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        assert_eq!(scratch.k(), k, "heap size is fixed at construction");
        // Shape contract: wrong n must panic (assert catches misuse).
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = knn_differential_entropy(&pts, n - 1, d, k, &mut scratch);
        }));
        assert!(result.is_err(), "mismatched n must be rejected");
        // n <= k must be rejected.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = knn_differential_entropy(&pts, 2, d, 3, &mut scratch);
        }));
        assert!(result.is_err(), "n <= k must be rejected");
    }

    // ── G2: offline cost budget ──────────────────────────────────────

    /// Brute-force O(n²·d) at the documented audit scale must finish in
    /// bounded wall time even in debug builds — a smoke gate on CORRECTNESS
    /// at scale (no timing assert: the load-flaky-gate lesson). n=2048, d=16
    /// ≈ 67M distance terms; the closed-form band stays open enough for the
    /// KL bias at d=16.
    #[test]
    fn offline_scale_smoke() {
        let (n, d, k) = (2048usize, 16usize, 5usize);
        let pts = gaussian_population(n, d, 1.0, 49);
        let mut scratch = KnnEntropyScratch::new(k);
        let h = knn_differential_entropy(&pts, n, d, k, &mut scratch);
        let expect = gaussian_entropy_closed_form(d, 1.0);
        assert!(
            (h - expect).abs() < 1.5,
            "audit-scale estimate {h:.4} vs closed form {expect:.4}"
        );
    }
}

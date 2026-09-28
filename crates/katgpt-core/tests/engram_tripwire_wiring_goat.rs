//! Issue 837 wiring GOAT — the stateful [`EngramTripwire`] detector at the
//! engram fusion seam (riir-ai Bench 832 §Wiring, owner call executed
//! 2026-09-01).
//!
//! The PoC (riir-ai `crates/riir-poc/tests/adversarial_evidence_tripwire_poc.rs`)
//! proved the statistics on a harness world; THIS gate proves the WIRING: the
//! detector composed onto the REAL sigmoid fusion kernel
//! (`sigmoid_fuse_scaled_into` returns the gate), calibrating and deciding
//! end-to-end, plus the repair half (drop the suspect source, re-check).
//!
//! Gates:
//! - **G8 detector** — PoC arms replayed through the real kernel at the PoC
//!   scale (K=8, D=32, τ=√D, 900 benign calibration / 300 held-out benign /
//!   200 per adversarial arm): `A_inject` and `A_656` fire 100%, benign
//!   `B_single` legit-collapse never fires, held-out benign FPR ≤ α.
//! - **G1 purity** — the detector is a read-only observer (fusion outputs
//!   bit-identical with checks interleaved) AND gate extraction via the
//!   scaled-fuse return value is bit-identical to the plain fuse path
//!   (v = e₀ ⇒ `out[0]` IS the gate).
//! - **G2 cost** — `check` at production scale is µs-cheap (release-only).
//! - **Repair** — on a fired world, dropping `suspect_source` and re-checking
//!   clears the inversion and restores consumption-retrieval agreement.
//! - **G4 alloc** lives in its OWN binary (`engram_tripwire_alloc_check`) —
//!   the counting allocator is a process global, so it must not share a test
//!   binary with sibling tests running on parallel threads (the
//!   bench_656_privilege_alloc_check lesson).
//!
//! Run:
//! ```bash
//! cargo test -p katgpt-core --features engram_tripwire --test engram_tripwire_wiring_goat
//! cargo test -p katgpt-core --release --features engram_tripwire \
//!     --test engram_tripwire_wiring_goat engram_tripwire_g2 -- --nocapture
//! ```

use katgpt_core::engram::{
    EngramTripwire, EngramTripwireConfig, SigmoidFusionConfig, sigmoid_fuse_into,
    sigmoid_fuse_scaled_into,
};
use katgpt_core::evidence_tripwire::TripwireMetrics;

// ─── Deterministic RNG (xorshift64 — house test convention) ────────────────

struct Xs(u64);

impl Xs {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// [0, 1)
    fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// [−span, +span]
    fn jitter(&mut self, span: f32) -> f32 {
        (self.f32() * 2.0 - 1.0) * span
    }
}

// ─── World construction through the REAL kernel ────────────────────────────

const D: usize = 32;
const K: usize = 8;

fn fusion_config() -> SigmoidFusionConfig {
    SigmoidFusionConfig {
        tau: (D as f32).sqrt(),
        rmsnorm_eps: 1e-6,
        logit_bias: 0.0,
    }
}

/// Random unit vector (Gaussian-ish via 3 uniform sums, then normalize —
/// direction quality is irrelevant; the kernel RMS-norms internally anyway).
fn random_unit(rng: &mut Xs) -> [f32; D] {
    let mut v = [0.0f32; D];
    for x in v.iter_mut() {
        *x = rng.f32() + rng.f32() + rng.f32() - 1.5;
    }
    let ss: f32 = v.iter().map(|x| x * x).sum();
    let inv = 1.0 / ss.max(1e-12).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
    v
}

/// `k = c·q̂ + s·r̂⊥` with r̂⊥ Gram-Schmidt-orthogonalized against q̂ ⇒
/// dot(q, k) = c EXACTLY (no cosθ sampling noise), so the kernel's
/// RMS-normalized dot is √D·c and the σ gate is a pure monotone function of
/// the alignment `c` alone (τ = √D ⇒ gate = σ(√D·c)).
fn aligned_source(q: &[f32; D], c: f32, rng: &mut Xs) -> [f32; D] {
    let r = random_unit(rng);
    let dq: f32 = r.iter().zip(q.iter()).map(|(&a, &b)| a * b).sum();
    let mut rp = [0.0f32; D];
    for i in 0..D {
        rp[i] = r[i] - dq * q[i];
    }
    let ss: f32 = rp.iter().map(|x| x * x).sum();
    let inv = 1.0 / ss.max(1e-12).sqrt();
    for x in rp.iter_mut() {
        *x *= inv;
    }
    let s = (1.0 - c * c).max(0.0).sqrt();
    let mut k = [0.0f32; D];
    for i in 0..D {
        k[i] = c * q[i] + s * rp[i];
    }
    let ss: f32 = k.iter().map(|x| x * x).sum();
    let inv = 1.0 / ss.max(1e-12).sqrt();
    for x in k.iter_mut() {
        *x *= inv;
    }
    k
}

/// One consumed world: per-source σ gates extracted from the REAL kernel +
/// the retrieval scores that admitted them.
struct World {
    retrieval: Vec<f32>,
    gates: Vec<f32>,
}

/// `align` = per-source alignment cosines, highest = most consumption-worthy;
/// `retrieval` = admission scores (any monotone scale, index-aligned).
fn build_world(q: &[f32; D], align: &[f32], retrieval: &[f32], rng: &mut Xs) -> World {
    let cfg = fusion_config();
    let mut e0 = [0.0f32; D];
    e0[0] = 1.0;
    let mut out = [0.0f32; D];
    let mut gates = Vec::with_capacity(align.len());
    for &c in align {
        let k = aligned_source(q, c, rng);
        // THE seam: fuse AND take the returned unscaled gate (v = e₀ so
        // out[0] mirrors it — the G1 gate pins the bit-identity).
        gates.push(sigmoid_fuse_scaled_into(q, &k, &e0, &mut out, &cfg, 1.0));
    }
    World {
        retrieval: retrieval.to_vec(),
        gates,
    }
}

/// Tight retrieval jitter: adjacent correlated sources may swap admission
/// order, but sources two grades apart never do — benign top1 ranks stay in a
/// low band (the PoC's B_multi shape: rank concentrates near 1).
fn jittered_descending(rng: &mut Xs, top: f32, step: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| top - step * i as f32 + rng.jitter(0.01))
        .collect()
}

fn filler_alignments(rng: &mut Xs, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.jitter(0.05)).collect()
}

fn filler_retrieval(rng: &mut Xs, n: usize) -> Vec<f32> {
    (0..n).map(|_| 0.05 + 0.3 * rng.f32()).collect()
}

/// Concatenate per-group (alignments, retrieval) into one world. `q` is
/// shared; each source gets its own random orthogonal complement.
struct Groups {
    align: Vec<f32>,
    retrieval: Vec<f32>,
}

impl Groups {
    fn new() -> Self {
        Self {
            align: Vec::with_capacity(K),
            retrieval: Vec::with_capacity(K),
        }
    }
    fn add(&mut self, align: &[f32], retrieval: &[f32]) {
        self.align.extend_from_slice(align);
        self.retrieval.extend_from_slice(retrieval);
    }
}

fn build_groups_world(q: &[f32; D], groups: &Groups, rng: &mut Xs) -> World {
    build_world(q, &groups.align, &groups.retrieval, rng)
}

// ─── Scenario arms (Bench-832 geometry, K=8 / D=32 / τ=√D) ─────────────────

/// B_multi — 8 topical sources, graded alignment; admission order matches
/// alignment order up to adjacent jitter (benign).
fn b_multi(q: &[f32; D], rng: &mut Xs) -> World {
    let align: Vec<f32> = (0..K)
        .map(|i| 0.90 - 0.03 * i as f32 + rng.jitter(0.02))
        .collect();
    let retrieval = jittered_descending(rng, 0.90, 0.03, K);
    build_world(q, &align, &retrieval, rng)
}

/// B_varied — 1 strong + 2 mid + 2 weak topical + 3 fillers (benign).
fn b_varied(q: &[f32; D], rng: &mut Xs) -> World {
    let mut g = Groups::new();
    g.add(&[0.90 + rng.jitter(0.02)], &[0.90 + rng.jitter(0.01)]);
    for i in 0..2 {
        let c = 0.50 - 0.10 * i as f32 + rng.jitter(0.02);
        g.add(&[c], &[0.50 - 0.10 * i as f32 + rng.jitter(0.01)]);
    }
    for i in 0..2 {
        let c = 0.20 - 0.05 * i as f32 + rng.jitter(0.02);
        g.add(&[c], &[0.25 - 0.05 * i as f32 + rng.jitter(0.01)]);
    }
    g.add(&filler_alignments(rng, 3), &filler_retrieval(rng, 3));
    build_groups_world(q, &g, rng)
}

/// B_single — 1 genuinely-relevant source + 7 fillers: legitimate collapse,
/// the discriminator control. The ONLY high-gate source is also the
/// top-retrieved one ⇒ the rank statistic must stay silent (Bench 832: 0%).
fn b_single(q: &[f32; D], rng: &mut Xs) -> World {
    let mut g = Groups::new();
    g.add(&[0.90 + rng.jitter(0.02)], &[0.90 + rng.jitter(0.01)]);
    g.add(&filler_alignments(rng, 7), &filler_retrieval(rng, 7));
    build_groups_world(q, &g, rng)
}

/// A_inject — 4 topical + 1 poison + 3 fillers. The poison is aligned for
/// CONSUMPTION (c above every jittered topical) but admitted LAST by
/// retrieval ⇒ top-gated source is retrieval-last (the inversion signature).
fn a_inject(q: &[f32; D], rng: &mut Xs) -> World {
    let mut g = Groups::new();
    let topical: Vec<f32> = (0..4)
        .map(|i| 0.90 - 0.03 * i as f32 + rng.jitter(0.02))
        .collect();
    let topical_ret = jittered_descending(rng, 0.90, 0.03, 4);
    g.add(&topical, &topical_ret);
    g.add(&[0.96], &[0.01]); // the poison: max consumption, last admission
    g.add(&filler_alignments(rng, 3), &filler_retrieval(rng, 3));
    build_groups_world(q, &g, rng)
}

/// Jitter-free `a_inject` (for the repair test): argmax and ranks are exact,
/// so the drop-and-recheck outcome is deterministic.
fn a_inject_exact(q: &[f32; D], rng: &mut Xs) -> World {
    let mut g = Groups::new();
    let topical: Vec<f32> = (0..4).map(|i| 0.90 - 0.03 * i as f32).collect();
    let topical_ret: Vec<f32> = (0..4).map(|i| 0.90 - 0.03 * i as f32).collect();
    g.add(&topical, &topical_ret);
    g.add(&[0.96], &[0.01]); // the poison: max consumption, last admission
    g.add(&filler_alignments(rng, 3), &filler_retrieval(rng, 3));
    build_groups_world(q, &g, rng)
}

/// A_656 — Bench-656 regime-A shape: 4 topical + 2 poisons admitted into the
/// retrieval BOTTOM HALF + 2 fillers. The first poison (index 4, strictly the
/// argmax gate at c = 0.96 vs 0.955) sits at retrieval rank 5 of 8.
fn a_656(q: &[f32; D], rng: &mut Xs) -> World {
    let mut g = Groups::new();
    let topical: Vec<f32> = (0..4)
        .map(|i| 0.90 - 0.03 * i as f32 + rng.jitter(0.02))
        .collect();
    let topical_ret = jittered_descending(rng, 0.90, 0.03, 4);
    g.add(&topical, &topical_ret);
    // Poisons: consumption-maximal, retrieval below the topicals and above
    // the fillers — retrieval ranks 5 and 6 of 8.
    g.add(&[0.96, 0.955], &[0.20, 0.15]);
    g.add(&filler_alignments(rng, 2), &[0.10, 0.05]);
    build_groups_world(q, &g, rng)
}

fn metrics_scratch() -> TripwireMetrics {
    TripwireMetrics {
        n: 0,
        h_norm: 0.0,
        top1_share: 0.0,
        tau: 0.0,
        top1_consumer_rank: 0.0,
    }
}

fn fresh_tripwire() -> EngramTripwire {
    EngramTripwire::new(EngramTripwireConfig::default())
}

/// The PoC calibration protocol: 900 benign worlds (300 per benign arm).
fn calibrate_900(tw: &mut EngramTripwire, q: &[f32; D], rng: &mut Xs, m: &mut TripwireMetrics) {
    for _ in 0..300 {
        let w = b_multi(q, rng);
        tw.observe_benign(&w.retrieval, &w.gates, m);
    }
    for _ in 0..300 {
        let w = b_varied(q, rng);
        tw.observe_benign(&w.retrieval, &w.gates, m);
    }
    for _ in 0..300 {
        let w = b_single(q, rng);
        tw.observe_benign(&w.retrieval, &w.gates, m);
    }
}

// ─── G8 — the detector through the real kernel ─────────────────────────────

#[test]
fn engram_tripwire_g8_detector_poc_arms() {
    let mut rng = Xs(0x0083_7832_2026_0901);
    let q = random_unit(&mut rng);
    let mut tw = fresh_tripwire();
    let mut m = metrics_scratch();

    calibrate_900(&mut tw, &q, &mut rng, &mut m);
    assert!(tw.is_calibrated());
    assert_eq!(tw.benign_worlds(), 900);
    assert_eq!(tw.pool_len(), 900);
    let threshold = tw.threshold();
    assert!(threshold.is_finite());

    // Held-out benign: 300 worlds (100 per arm) — FPR ≤ α.
    let mut benign_fires = 0u32;
    for _ in 0..100 {
        let w = b_multi(&q, &mut rng);
        if tw.check(&w.retrieval, &w.gates, &mut m).fired {
            benign_fires += 1;
        }
    }
    for _ in 0..100 {
        let w = b_varied(&q, &mut rng);
        if tw.check(&w.retrieval, &w.gates, &mut m).fired {
            benign_fires += 1;
        }
    }
    for _ in 0..100 {
        let w = b_single(&q, &mut rng);
        if tw.check(&w.retrieval, &w.gates, &mut m).fired {
            benign_fires += 1;
        }
    }
    let fpr = f64::from(benign_fires) / 300.0;
    assert!(
        fpr <= 0.05,
        "held-out benign FPR {fpr:.4} exceeds α = 0.05 (threshold {threshold})",
    );

    // B_single legit-collapse NEVER fires (the discriminator control).
    let mut single_fires = 0u32;
    for _ in 0..200 {
        let w = b_single(&q, &mut rng);
        if tw.check(&w.retrieval, &w.gates, &mut m).fired {
            single_fires += 1;
        }
    }
    assert_eq!(single_fires, 0, "benign legit-collapse must never fire");

    // A_inject: the poison tops the gate from retrieval-last ⇒ 100%.
    let mut inject_fires = 0u32;
    for _ in 0..200 {
        let w = a_inject(&q, &mut rng);
        let v = tw.check(&w.retrieval, &w.gates, &mut m);
        assert_eq!(v.suspect_source, 4, "poison is source index 4");
        if v.fired {
            inject_fires += 1;
        }
    }
    assert_eq!(inject_fires, 200, "A_inject must fire on every world");

    // A_656: the argmax poison sits at retrieval rank 5 of 8 ⇒ 100%.
    let mut a656_fires = 0u32;
    for _ in 0..200 {
        let w = a_656(&q, &mut rng);
        let v = tw.check(&w.retrieval, &w.gates, &mut m);
        assert_eq!(
            v.suspect_source, 4,
            "first poison (c=0.96) is source index 4"
        );
        if v.fired {
            a656_fires += 1;
        }
    }
    assert_eq!(a656_fires, 200, "A_656 must fire on every world");

    println!(
        "engram_tripwire G8: threshold={threshold:.4} benign_fpr={fpr:.4} \
         B_single 0/200 A_inject {inject_fires}/200 A_656 {a656_fires}/200"
    );
}

// ─── G1 — observer purity + gate-extraction bit identity ───────────────────

#[test]
fn engram_tripwire_g1_observer_pure_and_gate_extraction_bit_exact() {
    let mut rng = Xs(0xA11_0E5);
    let q = random_unit(&mut rng);
    let cfg = fusion_config();

    // (a) Gate extraction: the scaled-fuse RETURN value (scale 1.0) equals the
    // plain-fuse output with v = e₀, where out[0] IS the gate (·1.0 exact).
    for _ in 0..16 {
        let k = aligned_source(&q, rng.f32(), &mut rng);
        let mut out_a = [0.0f32; D];
        let v_unit = [0.5f32; D];
        let gate = sigmoid_fuse_scaled_into(&q, &k, &v_unit, &mut out_a, &cfg, 1.0);
        let mut e0 = [0.0f32; D];
        e0[0] = 1.0;
        let mut out_b = [0.0f32; D];
        sigmoid_fuse_into(&q, &k, &e0, &mut out_b, &cfg);
        assert_eq!(
            gate.to_bits(),
            out_b[0].to_bits(),
            "returned gate must be bit-identical to the plain-fuse gate"
        );
    }

    // (b) Observer purity: a full fuse pass with tripwire checks interleaved
    // leaves every output buffer bit-identical to the check-free pass. Both
    // passes consume the RNG identically (the check branch consumes none), so
    // the fused sources match source-for-source.
    let world_align: Vec<f32> = (0..K).map(|i| 0.9 - 0.03 * i as f32).collect();
    let retrieval = jittered_descending(&mut rng, 0.9, 0.03, K);
    let mut gates = Vec::with_capacity(K);
    let mut e0 = [0.0f32; D];
    e0[0] = 1.0;
    for &c in &world_align {
        let k = aligned_source(&q, c, &mut rng);
        let mut o = [0.0f32; D];
        gates.push(sigmoid_fuse_scaled_into(&q, &k, &e0, &mut o, &cfg, 1.0));
    }

    let mut tw = fresh_tripwire();
    let mut m = metrics_scratch();
    let w = b_multi(&q, &mut rng);
    tw.observe_benign(&w.retrieval, &w.gates, &mut m);

    let fuse_pass = |check: bool| -> u64 {
        let mut rng = Xs(0xF0FE);
        let mut hash: u64 = 0xcbf29ce484222325;
        let mut out = [0.0f32; D];
        for &c in &world_align {
            let k = aligned_source(&q, c, &mut rng);
            let v = random_unit(&mut rng);
            sigmoid_fuse_into(&q, &k, &v, &mut out, &cfg);
            if check {
                let mut scratch = metrics_scratch();
                let _ = tw.check(&retrieval, &gates, &mut scratch);
            }
            for x in out {
                hash = (hash ^ x.to_bits() as u64).wrapping_mul(0x100000001b3);
            }
        }
        hash
    };

    let h_plain = fuse_pass(false);
    let h_checked = fuse_pass(true);
    assert_eq!(h_plain, h_checked, "the detector must be a pure observer");
}

// ─── G2 — check cost (release-only) ─────────────────────────────────────────

#[test]
#[cfg_attr(debug_assertions, ignore = "timing gate — run with --release")]
fn engram_tripwire_g2_check_cost_budget() {
    use std::time::Instant;

    const N: usize = 20_000;

    let mut rng = Xs(0xC057);
    let q = random_unit(&mut rng);
    let mut tw = fresh_tripwire();
    let mut m = metrics_scratch();
    for _ in 0..64 {
        let w = b_multi(&q, &mut rng);
        tw.observe_benign(&w.retrieval, &w.gates, &mut m);
    }
    // Deterministic correlated world (no jitter — the no-fire property must
    // not depend on the seed): retrieval order == gate order ⇒ rank 1.
    let retrieval: Vec<f32> = (0..K).map(|i| 0.90 - 0.03 * i as f32).collect();
    let gates: Vec<f32> = (0..K).map(|i| 0.95 - 0.03 * i as f32).collect();
    let w = World { retrieval, gates };
    let t0 = Instant::now();
    let mut fired = 0u32;
    for _ in 0..N {
        if tw.check(&w.retrieval, &w.gates, &mut m).fired {
            fired += 1;
        }
    }
    let per_call = t0.elapsed().as_secs_f64() / N as f64;
    assert_eq!(fired, 0, "the calibration world itself must not fire");
    assert!(
        per_call < 5e-6,
        "check() {:.3} µs/call exceeds the 5 µs budget",
        per_call * 1e6
    );
    println!(
        "engram_tripwire G2: check() = {:.1} ns/call",
        per_call * 1e9
    );
}

// ─── Repair — drop the suspect, re-check ────────────────────────────────────

#[test]
fn engram_tripwire_repair_drop_suspect_clears_inversion() {
    let mut rng = Xs(0x0FF1CE);
    let q = random_unit(&mut rng);
    let mut tw = fresh_tripwire();
    let mut m = metrics_scratch();
    calibrate_900(&mut tw, &q, &mut rng, &mut m);

    let w = a_inject_exact(&q, &mut rng);
    let v = tw.check(&w.retrieval, &w.gates, &mut m);
    assert!(v.fired, "the poisoned world must fire before repair");
    assert_eq!(v.suspect_source, 4, "the poison is source index 4");
    assert!(
        (m.normalized_top1_rank() - 1.0).abs() < 1e-6,
        "pre-repair: top-consumed source is retrieval-last (rank = {})",
        m.top1_consumer_rank
    );
    let pre_tau = m.tau;

    // THE repair: drop the suspect source from the consumed set, re-check.
    let suspect = v.suspect_source;
    let mut repaired_retrieval = Vec::with_capacity(K - 1);
    let mut repaired_gates = Vec::with_capacity(K - 1);
    for i in 0..K {
        if i != suspect {
            repaired_retrieval.push(w.retrieval[i]);
            repaired_gates.push(w.gates[i]);
        }
    }
    let v2 = tw.check(&repaired_retrieval, &repaired_gates, &mut m);
    assert!(!v2.fired, "drop-the-suspect must clear the inversion");
    assert_eq!(
        v2.suspect_source, 0,
        "the strongest topical source now tops the gate"
    );
    assert!(
        (m.normalized_top1_rank()).abs() < 1e-6,
        "post-repair: the top-consumed source is the top-retrieved one"
    );
    assert!(
        m.tau > pre_tau,
        "consumption-retrieval agreement must improve (τ {:.3} → {:.3})",
        pre_tau,
        m.tau
    );
    println!(
        "engram_tripwire repair: suspect={suspect} rank 8→1, τ {:.3} → {:.3}, fired {} → {}",
        pre_tau, m.tau, v.fired, v2.fired
    );
}

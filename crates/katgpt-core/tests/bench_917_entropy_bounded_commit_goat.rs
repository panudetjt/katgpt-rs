#![cfg(feature = "entropy_bounded_commit")]
//! Issue 917 — modelless oracle A/B for the EB commit policy.
//!
//! The lane G2 (NFE at matched quality on D2F / DDTree / DFlash) needs a
//! trained diffusion checkpoint that does not exist yet (riir-infer
//! `gemma2_d2f` gates its quality assertion on `D2F_GOAT_TRAINED=1`). This
//! bench measures the same claim against an EXACT oracle denoiser, where
//! quality is decidable rather than proxied:
//!
//! - The target distribution is uniform over a finite set `S` of valid
//!   strings (length `L`, vocab `V`). The oracle returns the exact
//!   conditional marginal `p(x_i | revealed)` for every masked position by
//!   filtering `S`. One oracle call = one NFE.
//! - Each pass samples every masked position independently from its
//!   marginal (the factorized proposal every MDM / D2F commit uses), then a
//!   POLICY decides which sampled positions to commit.
//! - Quality = the final string is in `S`. Committing dependent positions
//!   together under a factorized proposal is exactly what produces invalid
//!   strings, so validity measures the joint-dependence error the EB bound
//!   is about. One-at-a-time commit (fixed k = 1) is exact: validity 1.0 at
//!   NFE = L — the reference point.
//!
//! Policies: EB(γ) (entropy proxy), fixed-k (lowest entropy first — the
//! top-k-confidence family), and the incumbent per-position threshold τ on
//! top-1 probability, in two forms: D2F's as shipped (no floor — a pass where
//! nothing clears τ commits nothing) and a floored variant (commit the most
//! confident singleton when nothing clears τ) so the incumbent is not
//! charged for the stall alone.
//!
//! Families: RANDOM (|S| random strings — every position depends on every
//! other), HALVES (S = A × B over two independent halves), and PEAKED-INDEP
//! (independent positions, each 0.9 on one token — the case where the EB
//! bound is loosest, because `Σ H − max H` charges for dependence that does
//! not exist; reported honestly, not hidden).
//!
//! Deterministic: seeded RNG throughout, no timing, so every bar below is a
//! reproducible count rather than a box-state-sensitive measurement.
//!
//! ```sh
//! cargo test -p katgpt-core --features entropy_bounded_commit \
//!   --test bench_917_entropy_bounded_commit_goat --release -- --nocapture
//! ```

use katgpt_core::entropy_bounded_commit::{PositionStats, entropy_bounded_commit};

const L: usize = 16;
const V: usize = 8;
const RUNS: usize = 2000;

#[derive(Clone, Copy, Debug)]
enum Policy {
    Eb(f32),
    FixedK(usize),
    /// Incumbent τ threshold; `floored` adds the singleton fallback.
    Tau {
        tau: f32,
        floored: bool,
    },
}

impl Policy {
    fn label(&self) -> String {
        match *self {
            Policy::Eb(g) => format!("EB γ={g}"),
            Policy::FixedK(k) => format!("fixed k={k}"),
            Policy::Tau { tau, floored } => {
                format!("τ={tau}{}", if floored { " +floor" } else { " (D2F)" })
            }
        }
    }
}

/// The target distribution, as an explicit support set (uniform weights).
struct Support {
    strings: Vec<[u8; L]>,
    /// Per-string weight (uniform for the set families).
    weights: Vec<f64>,
    /// Closed-form independent family: per-position (0.9 token, 0.1 token).
    independent: Option<([u8; L], [u8; L])>,
}

impl Support {
    fn random(n: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let strings: Vec<[u8; L]> = (0..n)
            .map(|_| std::array::from_fn(|_| rng.u8(..V as u8)))
            .collect();
        let weights = vec![1.0; n];
        Self {
            strings,
            weights,
            independent: None,
        }
    }

    fn halves(n_half: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let h = L / 2;
        let a: Vec<Vec<u8>> = (0..n_half)
            .map(|_| (0..h).map(|_| rng.u8(..V as u8)).collect())
            .collect();
        let b: Vec<Vec<u8>> = (0..n_half)
            .map(|_| (0..h).map(|_| rng.u8(..V as u8)).collect())
            .collect();
        let mut strings = Vec::with_capacity(n_half * n_half);
        for x in &a {
            for y in &b {
                strings.push(std::array::from_fn(|i| if i < h { x[i] } else { y[i - h] }));
            }
        }
        let weights = vec![1.0; strings.len()];
        Self {
            strings,
            weights,
            independent: None,
        }
    }

    /// Independent positions with a 0.9/0.1 marginal over two tokens per
    /// position. Kept in closed form (the support is 2^L strings, so the
    /// marginals and membership are computed per position, not enumerated).
    fn peaked_independent(seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let hi: [u8; L] = std::array::from_fn(|_| rng.u8(..V as u8));
        let lo: [u8; L] = std::array::from_fn(|i| (hi[i] + 1 + rng.u8(..(V as u8 - 1))) % V as u8);
        Self {
            strings: Vec::new(),
            weights: Vec::new(),
            independent: Some((hi, lo)),
        }
    }

    fn size(&self) -> usize {
        match self.independent {
            Some(_) => 1 << L,
            None => self.strings.len(),
        }
    }

    fn contains(&self, s: &[u8; L]) -> bool {
        match &self.independent {
            Some((hi, lo)) => (0..L).all(|i| s[i] == hi[i] || s[i] == lo[i]),
            None => self.strings.iter().any(|x| x == s),
        }
    }

    /// Exact conditional marginals for every position given `canvas`
    /// (`None` = masked). Returns false when no support string is
    /// consistent (a prior parallel commit already broke the joint).
    fn marginals(&self, canvas: &[Option<u8>; L], out: &mut [[f64; V]; L]) -> bool {
        for row in out.iter_mut() {
            *row = [0.0; V];
        }
        if let Some((hi, lo)) = &self.independent {
            // Independence: revealed positions do not move the others.
            for i in 0..L {
                out[i][hi[i] as usize] = 0.9;
                out[i][lo[i] as usize] = 0.1;
            }
            return true;
        }
        let mut total = 0.0;
        for (s, &w) in self.strings.iter().zip(&self.weights) {
            if canvas.iter().zip(s).all(|(c, &t)| c.is_none_or(|c| c == t)) {
                total += w;
                for i in 0..L {
                    out[i][s[i] as usize] += w;
                }
            }
        }
        if total == 0.0 {
            return false;
        }
        for row in out.iter_mut() {
            for p in row.iter_mut() {
                *p /= total;
            }
        }
        true
    }
}

fn stats_of(p: &[f64; V]) -> PositionStats {
    let mut h = 0.0f64;
    let (mut t1, mut t2) = (0.0f64, 0.0f64);
    for &x in p {
        if x > 0.0 {
            h -= x * x.ln();
        }
        if x > t1 {
            t2 = t1;
            t1 = x;
        } else if x > t2 {
            t2 = x;
        }
    }
    PositionStats {
        entropy: h as f32,
        top1: t1 as f32,
        margin: (t1 - t2) as f32,
    }
}

fn sample(p: &[f64; V], rng: &mut fastrand::Rng) -> u8 {
    let u = rng.f64();
    let mut acc = 0.0;
    for (t, &x) in p.iter().enumerate() {
        acc += x;
        if u < acc {
            return t as u8;
        }
    }
    // Rounding tail: the last token with mass.
    p.iter().rposition(|&x| x > 0.0).unwrap_or(0) as u8
}

#[derive(Default, Clone, Copy)]
struct Outcome {
    valid: usize,
    stalled: usize,
    nfe_sum: usize,
}

impl Outcome {
    fn validity(&self) -> f64 {
        self.valid as f64 / RUNS as f64
    }
    fn mean_nfe(&self) -> f64 {
        self.nfe_sum as f64 / RUNS as f64
    }
}

/// Decode `RUNS` samples under `policy`. A run that cannot make progress
/// for `4·L` passes is a STALL (counted invalid, NFE charged to the cap).
fn run(support: &Support, policy: Policy, seed: u64) -> Outcome {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut out = Outcome::default();
    let mut marg = [[0.0f64; V]; L];
    let mut ent = [0.0f32; L];
    let mut key = [0.0f32; L];
    let mut cand: Vec<u32> = Vec::with_capacity(L);
    let cap = 4 * L;
    for _ in 0..RUNS {
        let mut canvas: [Option<u8>; L] = [None; L];
        let mut nfe = 0usize;
        let mut broken = false;
        while canvas.iter().any(Option::is_none) && nfe < cap {
            nfe += 1;
            if !support.marginals(&canvas, &mut marg) {
                broken = true;
                break;
            }
            let draft: [u8; L] = std::array::from_fn(|i| sample(&marg[i], &mut rng));
            cand.clear();
            for i in 0..L {
                if canvas[i].is_none() {
                    let s = stats_of(&marg[i]);
                    ent[i] = s.entropy;
                    key[i] = s.entropy;
                    cand.push(i as u32);
                }
            }
            let commit: &[u32] = match policy {
                Policy::Eb(g) => {
                    let k = entropy_bounded_commit(&mut cand, &ent, &key, g, usize::MAX);
                    &cand[..k]
                }
                Policy::FixedK(kk) => {
                    let k = entropy_bounded_commit(&mut cand, &ent, &key, f32::INFINITY, kk);
                    &cand[..k]
                }
                Policy::Tau { tau, floored } => {
                    // Order by confidence (key = −top1), keep those ≥ τ.
                    for &c in &cand {
                        key[c as usize] = -stats_of(&marg[c as usize]).top1;
                    }
                    let k =
                        entropy_bounded_commit(&mut cand, &ent, &key, f32::INFINITY, usize::MAX);
                    let n_pass = cand[..k]
                        .iter()
                        .take_while(|&&c| -key[c as usize] >= tau)
                        .count();
                    let n = if n_pass == 0 && floored { 1 } else { n_pass };
                    &cand[..n]
                }
            };
            for &c in commit {
                canvas[c as usize] = Some(draft[c as usize]);
            }
        }
        if broken || canvas.iter().any(Option::is_none) {
            if !broken {
                out.stalled += 1;
            }
            out.nfe_sum += nfe;
            continue;
        }
        out.nfe_sum += nfe;
        let s: [u8; L] = std::array::from_fn(|i| canvas[i].unwrap());
        if support.contains(&s) {
            out.valid += 1;
        }
    }
    out
}

fn policies() -> Vec<Policy> {
    let cap_iter = [0.0f32, 0.1, 0.3, 1.0];
    let mut v = Vec::with_capacity(cap_iter.len());
    for g in cap_iter {
        v.push(Policy::Eb(g));
    }
    for k in [1usize, 2, 4, 8] {
        v.push(Policy::FixedK(k));
    }
    for tau in [0.5f32, 0.9] {
        v.push(Policy::Tau {
            tau,
            floored: false,
        });
        v.push(Policy::Tau { tau, floored: true });
    }
    v
}

fn table(name: &str, support: &Support, seed: u64) -> Vec<(Policy, Outcome)> {
    println!(
        "\n── {name} (|S| = {}, L = {L}, V = {V}, runs = {RUNS}) ──",
        support.size()
    );
    println!(
        "{:<18} {:>9} {:>9} {:>8}",
        "policy", "validity", "mean NFE", "stalls"
    );
    policies()
        .into_iter()
        .map(|p| {
            let o = run(support, p, seed);
            println!(
                "{:<18} {:>9.4} {:>9.2} {:>8}",
                p.label(),
                o.validity(),
                o.mean_nfe(),
                o.stalled
            );
            (p, o)
        })
        .collect()
}

fn get(rows: &[(Policy, Outcome)], pred: impl Fn(&Policy) -> bool) -> Outcome {
    rows.iter()
        .find(|(p, _)| pred(p))
        .map(|(_, o)| *o)
        .expect("policy row")
}

/// The smallest mean NFE any fixed-k row achieves at validity ≥ `bar`.
fn best_fixed_k_nfe(rows: &[(Policy, Outcome)], bar: f64) -> f64 {
    rows.iter()
        .filter(|(p, o)| matches!(p, Policy::FixedK(_)) && o.validity() >= bar)
        .map(|(_, o)| o.mean_nfe())
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn bench_917_eb_commit_oracle_ab() {
    let random = Support::random(32, 917);
    let halves = Support::halves(8, 9170);
    let peaked = Support::peaked_independent(91700);

    let r = table("RANDOM", &random, 1);
    let h = table("HALVES", &halves, 2);
    let p = table("PEAKED-INDEP", &peaked, 3);

    // Reference: fixed k = 1 is exact.
    for rows in [&r, &h, &p] {
        let k1 = get(rows, |p| matches!(p, Policy::FixedK(1)));
        assert_eq!(k1.valid, RUNS, "k=1 is exact sequential sampling");
    }

    // G1-lane: EB never stalls, on any family, at any γ.
    for rows in [&r, &h, &p] {
        for (pol, o) in rows.iter().filter(|(p, _)| matches!(p, Policy::Eb(_))) {
            assert_eq!(o.stalled, 0, "{} stalled", pol.label());
        }
    }

    // G2/G3 on the dependent families: EB at γ = 0.1 keeps validity within
    // 1% of exact, and needs fewer NFE than every fixed-k row that does.
    for (name, rows) in [("RANDOM", &r), ("HALVES", &h)] {
        let eb = get(rows, |p| matches!(p, Policy::Eb(g) if *g == 0.1));
        let best_k = best_fixed_k_nfe(rows, 0.99);
        println!(
            "{name}: EB γ=0.1 validity {:.4} at {:.2} NFE vs best fixed-k {:.2} NFE at ≥0.99",
            eb.validity(),
            eb.mean_nfe(),
            best_k
        );
        assert!(
            eb.validity() >= 0.99,
            "{name}: G3 EB validity {:.4}",
            eb.validity()
        );
        assert!(
            eb.mean_nfe() < best_k,
            "{name}: G2 EB NFE {:.2} vs best fixed-k {best_k:.2}",
            eb.mean_nfe()
        );
    }

    // PEAKED-INDEP is reported, not gated: every policy is valid there
    // (independent positions), and the EB bound is loosest.
    let eb = get(&p, |p| matches!(p, Policy::Eb(g) if *g == 0.1));
    assert_eq!(eb.valid, RUNS);
    println!(
        "PEAKED-INDEP: EB γ=0.1 at {:.2} NFE vs fixed k=8 at {:.2} (both exact here — the bound's cost)",
        eb.mean_nfe(),
        get(&p, |p| matches!(p, Policy::FixedK(8))).mean_nfe()
    );
}

/// G2 against the FAIR incumbent: the τ threshold with the singleton floor
/// (the shipped D2F form stalls outright on a flat canvas, which would make
/// the comparison about the floor alone). Gated over three independent
/// support draws per dependent family, so the verdict is not one fixture's.
#[test]
fn bench_917_eb_vs_floored_tau_across_seeds() {
    let eb = Policy::Eb(0.1);
    let tau = Policy::Tau {
        tau: 0.9,
        floored: true,
    };
    let shipped = Policy::Tau {
        tau: 0.9,
        floored: false,
    };
    println!(
        "\n{:<8} {:>6} {:>16} {:>16} {:>10}",
        "family", "seed", "EB γ=0.1 NFE", "τ=0.9+floor NFE", "D2F stalls"
    );
    for seed in [917u64, 918, 919] {
        for (name, support) in [
            ("RANDOM", Support::random(32, seed)),
            ("HALVES", Support::halves(8, seed * 10)),
        ] {
            let (e, t, s) = (
                run(&support, eb, seed),
                run(&support, tau, seed),
                run(&support, shipped, seed),
            );
            println!(
                "{name:<8} {seed:>6} {:>9.2} ({:.3}) {:>9.2} ({:.3}) {:>10}",
                e.mean_nfe(),
                e.validity(),
                t.mean_nfe(),
                t.validity(),
                s.stalled
            );
            assert!(
                e.validity() >= t.validity() - 0.005,
                "{name}/{seed}: EB validity {:.4} vs τ+floor {:.4}",
                e.validity(),
                t.validity()
            );
            assert!(
                e.mean_nfe() < t.mean_nfe(),
                "{name}/{seed}: EB NFE {:.2} vs τ+floor {:.2}",
                e.mean_nfe(),
                t.mean_nfe()
            );
        }
    }
}

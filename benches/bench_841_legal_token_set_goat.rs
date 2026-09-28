//! GOAT gate — legal-token-set enumeration + the restricted vocabulary
//! projection (Issue 841, the grammar-forced projection-skip lead).
//!
//! ```bash
//! cargo bench --features legal_token_set --bench bench_841_legal_token_set_goat
//! # G4 needs an allocator in the profile:
//! cargo bench --features legal_token_set,alloc_tracking \
//!     --bench bench_841_legal_token_set_goat
//! ```
//!
//! # What is being gated
//!
//! Two shipped capabilities with no accessor, and nothing joining them:
//! `LodestarAutomaton` holds a dense δ table that names every legal token of
//! every state and exposes only a per-token point query, and
//! `katgpt-forward`'s `fill_cluster_exact` is a gathered-row LM-head pass that
//! was `pub(crate)` and cluster-shaped. So every consumer asks validity the
//! only way the API allowed — scan the vocabulary, call `is_valid` per token,
//! and inside each of those calls re-walk the prefix.
//!
//! **G1 correctness** — the enumerated tree is BIT-identical to the scanned
//! one, and a restricted projection's rows are bit-identical to the dense
//! pass's. Both arms live in one binary: the scan is reached through a wrapper
//! whose `legal_degree` returns `None`, which is the state of every pruner
//! that has not opted in.
//!
//! **G2 perf** — interleaved A/B (`tests/common/ab_timing.rs`), because these
//! two arms ARE the same work and a sequential ratio would measure the box
//! (AGENTS.md § *A ratio of two SEQUENTIALLY-timed arms measures the BOX*).
//!
//! **G2b the crossover, MEASURED** — a gathered LM-head row runs at ~20.6 GB/s
//! against ~108 GB/s dense (Issue 661), so "smaller than the vocabulary" is
//! not a reason to gather. This sweeps the active fraction and reports where
//! the restricted pass actually stops winning, and gates that
//! `RestrictionPolicy`'s default threshold sits on the winning side of it.
//! A default that loses is a default that has to move.
//!
//! **G3 no-regression, WORST CASE** — a pruner whose legal set is the whole
//! vocabulary buys nothing from enumeration and still pays the CSR walk. That
//! is the shape the seam could make slower, so it is the shape the bar is on.
//!
//! **G4 alloc-free** — the enumerated build must allocate no more than the
//! scanned one. Not zero: both push to a `BinaryHeap`. The claim is that the
//! seam adds nothing.

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::legal_token_set::{ProjectionPlan, RestrictionPolicy, plan_projection};
use katgpt_core::traits::{CompletionHorizon, ConstraintPruner};
use katgpt_forward::{restricted_lm_head, standard_lm_head};
use katgpt_pruners::lodestar::{LodestarAutomaton, LodestarPruner};
use katgpt_speculative::dd_tree::{LodestarConfig, build_dd_tree_lodestar};

#[path = "../tests/common/ab_timing.rs"]
mod ab_timing;
use ab_timing::ab_median_ratio;

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[path = "../tests/common/alloc_tracking.rs"]
mod alloc_tracking;

const VOCAB: usize = 32_768;
const N_EMBD: usize = 256;
const SEQ_LEN: usize = 6;
const TREE_BUDGET: usize = 64;

// ── Fixtures ───────────────────────────────────────────────────────────

/// Deterministic LCG — a fixed seed, so every arm sees identical inputs and a
/// re-run reproduces the number.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

/// The scan arm: the same pruner with the enumeration hidden.
///
/// Reaching the old path by a WRAPPER rather than by a second build is what
/// makes this an A/B at all — one binary, one fixture, one difference.
struct ScanOnly<'a>(&'a LodestarPruner);

impl ConstraintPruner for ScanOnly<'_> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        self.0.is_valid(depth, token_idx, parent_tokens)
    }
    // legal_degree / for_each_legal left at the default — the point.
}

impl CompletionHorizon for ScanOnly<'_> {
    fn min_completion_distance(&self, d: usize, t: usize, p: &[usize]) -> u32 {
        self.0.min_completion_distance(d, t, p)
    }
    fn singular_span_len(&self, d: usize, p: &[usize]) -> u32 {
        self.0.singular_span_len(d, p)
    }
}

/// A grammar over a real 32 K alphabet with `degree` legal tokens per state —
/// the shape a JSON schema or a tool-call syntax actually has.
fn grammar(degree: usize) -> LodestarAutomaton {
    let n_states = 6;
    let mut b = LodestarAutomaton::builder(VOCAB, n_states, 0);
    for s in 0..n_states - 1 {
        for k in 0..degree {
            // Spread the legal tokens across the alphabet so the enumeration
            // cannot be a contiguous prefix the prefetcher makes free.
            let tok = (s * 7919 + k * 4099) % VOCAB;
            b.add_transition(s, tok, s + 1);
        }
    }
    b.add_accept(n_states - 1);
    b.build()
}

fn marginals_owned() -> Vec<Vec<f32>> {
    let mut r = Lcg(0x841_5EA1);
    (0..SEQ_LEN)
        .map(|_| (0..VOCAB).map(|_| r.next_f32().abs() + 1e-6).collect())
        .collect()
}

fn config() -> katgpt_core::Config {
    let mut c = katgpt_core::Config::draft();
    c.tree_budget = TREE_BUDGET;
    c
}

// ── Gates ──────────────────────────────────────────────────────────────

fn g1_bit_identical(failures: &mut Vec<String>) {
    let owned = marginals_owned();
    let m: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
    let cfg = config();

    let mut checked = 0usize;
    let mut mismatch = 0usize;
    for degree in [1usize, 4, 64, 2048] {
        let pruner = LodestarPruner::new(grammar(degree));
        for lode in [
            LodestarConfig::default(),
            LodestarConfig {
                jump_ahead: true,
                astar_lambda: 0.0,
            },
            LodestarConfig::thinking(0.15),
        ] {
            let fast = build_dd_tree_lodestar(&m, &cfg, &pruner, &lode);
            let slow = build_dd_tree_lodestar(&m, &cfg, &ScanOnly(&pruner), &lode);
            checked += 1;
            let same = fast.len() == slow.len()
                && fast.iter().zip(slow.iter()).all(|(a, b)| {
                    a.depth == b.depth
                        && a.token_idx == b.token_idx
                        && a.parent_path == b.parent_path
                        && a.score.to_bits() == b.score.to_bits()
                });
            if !same {
                mismatch += 1;
            }
        }
    }

    // A vacuous cell set would pass this gate by asserting nothing.
    let nonempty = {
        let p = LodestarPruner::new(grammar(4));
        !build_dd_tree_lodestar(&m, &cfg, &p, &LodestarConfig::default()).is_empty()
    };

    println!(
        "G1 tree bit-identity   {checked} (degree x config) cells, {mismatch} mismatched, \
         fixture builds a tree: {nonempty}  → {}",
        verdict(mismatch == 0 && nonempty && checked == 12)
    );
    if mismatch > 0 || !nonempty {
        failures.push(format!(
            "G1: {mismatch} of {checked} cells diverged (nonempty {nonempty})"
        ));
    }

    // The projection half.
    let mut r = Lcg(0xC0FFEE);
    let lm_head: Vec<f32> = (0..VOCAB * N_EMBD).map(|_| r.next_f32()).collect();
    let hidden: Vec<f32> = (0..N_EMBD).map(|_| r.next_f32()).collect();
    let mut dense = vec![0.0f32; VOCAB];
    standard_lm_head(&mut dense, &hidden, &lm_head, VOCAB, N_EMBD);
    let tokens: Vec<usize> = (0..VOCAB).filter(|t| t % 512 == 3).collect();
    let mut restricted = vec![0.0f32; VOCAB];
    restricted_lm_head(&mut restricted, &hidden, &lm_head, &tokens, VOCAB, N_EMBD);

    let rows_ok = tokens
        .iter()
        .all(|&t| restricted[t].to_bits() == dense[t].to_bits());
    let masked_ok = (0..VOCAB)
        .filter(|t| !tokens.contains(t))
        .all(|t| restricted[t] == f32::NEG_INFINITY);
    println!(
        "G1 projection rows     {} selected rows bit-identical: {rows_ok}; \
         unselected are -inf: {masked_ok}  → {}",
        tokens.len(),
        verdict(rows_ok && masked_ok)
    );
    if !(rows_ok && masked_ok) {
        failures.push("G1: restricted projection diverged from the dense pass".into());
    }
}

fn g2_tree_build(failures: &mut Vec<String>) {
    let owned = marginals_owned();
    let m: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
    let cfg = config();
    let lode = LodestarConfig::default();

    for degree in [1usize, 8, 64] {
        let pruner = LodestarPruner::new(grammar(degree));
        let scan = ScanOnly(&pruner);
        // a = the enumerated arm, b = the scan, so `ratio b/a` reads as
        // "how many times the scan costs" — > 1 is a win.
        let r = ab_median_ratio(
            9,
            1,
            2,
            |_| {
                black_box(build_dd_tree_lodestar(black_box(&m), &cfg, &pruner, &lode));
            },
            |_| {
                black_box(build_dd_tree_lodestar(black_box(&m), &cfg, &scan, &lode));
            },
        );
        let speedup = r.median;
        println!(
            "G2 tree build deg={degree:<5} enumerated {:.0} ns, scanned {:.0} ns \
             → {speedup:.1}x  {}",
            r.a_ns_per_iter(),
            r.b_ns_per_iter(),
            verdict(speedup >= 2.0)
        );
        if speedup < 2.0 {
            failures.push(format!(
                "G2: degree {degree} speedup {speedup:.2}x below the 2.0x bar"
            ));
        }
    }
}

fn g2b_projection_crossover(failures: &mut Vec<String>) {
    let mut r = Lcg(0x5EED);
    let lm_head: Vec<f32> = (0..VOCAB * N_EMBD).map(|_| r.next_f32()).collect();
    let hidden: Vec<f32> = (0..N_EMBD).map(|_| r.next_f32()).collect();
    let mut logits_a = vec![0.0f32; VOCAB];
    let mut logits_b = vec![0.0f32; VOCAB];

    println!("G2b projection crossover (gathered rows vs the dense pass)");
    let mut crossover: Option<f64> = None;
    let mut at_default: Option<f64> = None;

    for pct in [0.1f64, 1.0, 5.0, 10.0, 20.0, 40.0, 100.0] {
        let n = ((pct / 100.0) * VOCAB as f64).round().max(1.0) as usize;
        let stride = VOCAB / n;
        let tokens: Vec<usize> = (0..n).map(|k| (k * stride) % VOCAB).collect();

        let ab = ab_median_ratio(
            7,
            3,
            2,
            |_| {
                black_box(restricted_lm_head(
                    black_box(&mut logits_a),
                    &hidden,
                    &lm_head,
                    &tokens,
                    VOCAB,
                    N_EMBD,
                ));
            },
            |_| {
                standard_lm_head(black_box(&mut logits_b), &hidden, &lm_head, VOCAB, N_EMBD);
                black_box(&logits_b);
            },
        );
        let speedup = ab.median;
        let wins = speedup > 1.0;
        if !wins && crossover.is_none() {
            crossover = Some(pct);
        }
        if (pct - 10.0).abs() < f64::EPSILON {
            at_default = Some(speedup);
        }
        println!(
            "   active {pct:>5.1}% ({n:>6} rows)  restricted {:>9.0} ns, dense {:>9.0} ns \
             → {speedup:.2}x {}",
            ab.a_ns_per_iter(),
            ab.b_ns_per_iter(),
            match wins {
                true => "win",
                false => "LOSS",
            }
        );
    }

    match crossover {
        Some(p) => println!("   measured crossover: the gather stops winning at ~{p:.1}% active"),
        None => println!("   measured crossover: the gather won at every sampled fraction"),
    }

    // The gate is on the SHIPPED DEFAULT, not on the best cell: a threshold
    // that admits a losing gather is a threshold that has to move.
    let default_pct = RestrictionPolicy::default().max_active_fraction as f64 * 100.0;
    let ok = at_default.is_some_and(|s| s > 1.0);
    println!(
        "G2b default threshold  RestrictionPolicy::max_active_fraction = {default_pct:.0}% \
         measures {:.2}x  → {}",
        at_default.unwrap_or(f64::NAN),
        verdict(ok)
    );
    if !ok {
        failures.push(format!(
            "G2b: the default {default_pct:.0}% threshold admits a gather that measured \
             {:.2}x — lower max_active_fraction to the measured crossover",
            at_default.unwrap_or(f64::NAN)
        ));
    }
}

fn g3_worst_case(failures: &mut Vec<String>) {
    // Every token legal: enumeration buys nothing and still walks a 32 768-
    // entry CSR row. If the seam can be a regression anywhere, it is here.
    let owned = marginals_owned();
    let m: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
    let cfg = config();
    let lode = LodestarConfig::default();
    let pruner = LodestarPruner::new(grammar(VOCAB));
    let scan = ScanOnly(&pruner);

    let r = ab_median_ratio(
        7,
        1,
        2,
        |_| {
            black_box(build_dd_tree_lodestar(black_box(&m), &cfg, &pruner, &lode));
        },
        |_| {
            black_box(build_dd_tree_lodestar(black_box(&m), &cfg, &scan, &lode));
        },
    );
    // ratio b/a < 1 means the enumerated arm is SLOWER.
    let ok = r.median >= 0.90;
    println!(
        "G3 worst case (all {VOCAB} legal)  enumerated {:.0} ns vs scanned {:.0} ns \
         → {:.2}x  {}",
        r.a_ns_per_iter(),
        r.b_ns_per_iter(),
        r.median,
        verdict(ok)
    );
    if !ok {
        failures.push(format!(
            "G3: full-vocabulary legal set regressed to {:.2}x of the scan",
            r.median
        ));
    }

    // And the plan must refuse to gather there — a degenerate legal set is
    // exactly the case the naive rule gets wrong.
    let plan = plan_projection(
        pruner.legal_degree(0, &[]),
        VOCAB,
        &RestrictionPolicy::default(),
    );
    let refuses = matches!(
        plan,
        ProjectionPlan::Full {
            unenumerable: false
        }
    );
    println!(
        "G3 plan refuses gather at degree {VOCAB}: {refuses}  → {}",
        verdict(refuses)
    );
    if !refuses {
        failures.push("G3: the plan would gather the whole vocabulary".into());
    }
}

/// The COST side of the promotion argument, measured rather than reasoned.
///
/// The index is built for every automaton, so promoting this to default makes
/// every `LodestarAutomaton` carry it. `CsrLegalSet` is
/// `4·n_edges + 4·(n_states+1)` bytes against the dense table's
/// `8·n_states·vocab`, so the overhead ratio is `density/2` — negligible for a
/// grammar, and bounded at **+50 %** for a δ that is completely dense. That
/// bound is the honest worst case and it is asserted here, at a vocabulary
/// where the dense table is 1.6 GB and nobody would build one anyway.
fn g5_memory(failures: &mut Vec<String>) {
    println!("G5 index memory (the cost of promoting a per-automaton index)");
    let mut worst = 0.0f64;
    for (label, degree) in [
        ("grammar (deg 8)", 8usize),
        ("wide (deg 2048)", 2048),
        ("fully dense", VOCAB),
    ] {
        let a = grammar(degree);
        let dense = size_of::<usize>() * a.n_states() * a.vocab_size();
        let csr = a.legal_set().memory_bytes();
        let pct = 100.0 * csr as f64 / dense as f64;
        worst = worst.max(pct);
        println!(
            "   {label:<16} csr {:>10} B on top of dense {:>12} B → +{pct:.2}%",
            csr, dense
        );
    }
    let ok = worst <= 50.0;
    println!(
        "G5 worst overhead      +{worst:.2}% (bound is density/2, i.e. +50% at full density)          → {}",
        verdict(ok)
    );
    if !ok {
        failures.push(format!(
            "G5: index overhead {worst:.2}% exceeded the +50% bound"
        ));
    }
}

fn g4_alloc(failures: &mut Vec<String>) {
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    {
        let owned = marginals_owned();
        let m: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
        let cfg = config();
        let lode = LodestarConfig::default();
        let pruner = LodestarPruner::new(grammar(8));
        let scan = ScanOnly(&pruner);

        // Warm both arms first — a lazily filled cache allocating on its first
        // call is a property of the builder, not of the seam.
        black_box(build_dd_tree_lodestar(&m, &cfg, &pruner, &lode));
        black_box(build_dd_tree_lodestar(&m, &cfg, &scan, &lode));

        katgpt_core::alloc::reset_alloc_stats();
        black_box(build_dd_tree_lodestar(&m, &cfg, &pruner, &lode));
        let (fast, _) = katgpt_core::alloc::get_alloc_stats();

        katgpt_core::alloc::reset_alloc_stats();
        black_box(build_dd_tree_lodestar(&m, &cfg, &scan, &lode));
        let (slow, _) = katgpt_core::alloc::get_alloc_stats();

        let ok = fast <= slow;
        println!(
            "G4 alloc-free          enumerated {fast} alloc(s) vs scanned {slow} — the seam \
             adds {} → {}",
            fast as i64 - slow as i64,
            verdict(ok)
        );
        if !ok {
            failures.push(format!("G4: the seam added {} allocation(s)", fast - slow));
        }
    }
    #[cfg(not(any(debug_assertions, feature = "alloc_tracking")))]
    {
        let _ = &mut *failures;
        println!(
            "G4 alloc-free          ⛔ NOT MEASURED — this profile compiles no allocator. \
             Re-run with `--features alloc_tracking`; a green run without it is not a G4 pass."
        );
    }
}

fn main() {
    println!("Bench 841 — legal-token-set enumeration + restricted projection (Issue 841)");
    println!("vocab {VOCAB}, n_embd {N_EMBD}, seq_len {SEQ_LEN}, tree_budget {TREE_BUDGET}");
    let t0 = Instant::now();
    let mut failures: Vec<String> = Vec::new();

    g1_bit_identical(&mut failures);
    println!();
    g2_tree_build(&mut failures);
    println!();
    g2b_projection_crossover(&mut failures);
    println!();
    g3_worst_case(&mut failures);
    println!();
    g4_alloc(&mut failures);
    println!();
    g5_memory(&mut failures);

    println!("\n(total {:.1}s)", t0.elapsed().as_secs_f64());
    match failures.is_empty() {
        true => println!("✓ Bench 841 PASSED — every measured gate holds"),
        false => {
            for f in &failures {
                println!("✗ {f}");
            }
            println!("\n✗ Bench 841 FAILED — {} gate(s)", failures.len());
            std::process::exit(1);
        }
    }
}

fn verdict(ok: bool) -> &'static str {
    match ok {
        true => "PASS",
        false => "FAIL",
    }
}

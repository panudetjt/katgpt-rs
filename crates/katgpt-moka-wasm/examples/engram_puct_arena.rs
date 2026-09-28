//! Issue 868 / Plan 605 T3 — the engram-fused PUCT arena (Proposal 013
//! Phase 3) and the G5 verdict gate.
//!
//! Arms (all native, int8 K=1 — the fused player forces K=1, so the plain
//! player is constructed the same way for an apples-to-apples pairing):
//!
//! - **T3.1 head-to-head (PRIMARY)** — fused vs plain, PAIRED: each seed is
//!   played twice with the SAME random opening, colors swapped (each arm
//!   plays both colours per seed). No PUCT-vs-GREEDY one-colour bias can
//!   ride into the verdict.
//! - **T3.2 leakage controls** — (a) eval seeds are disjoint from the
//!   miner's seed set by construction (different multiplicative bases);
//!   (b) fused-vs-GREEDY and plain-vs-GREEDY on IDENTICAL seed+color
//!   schedules — a real strength gain must appear against the independent
//!   opponent too, not only head-to-head.
//! - **T3.3 G5 gate, with the power fix** — PASS = the one-sided 95%
//!   Wilson lower bound > 50% at n ≥ 200 paired games (default 616 ≈ the
//!   80%-power n for a true 55% edge). Clopper–Pearson exact lower bound
//!   reported as the conservative secondary. The budget alternative
//!   (fused at ≤ 50% budget matching plain's win rate) is reported
//!   descriptively via the b25-vs-b50 arm.
//! - **G2 measurement** — direct `read()` timing over mined positions
//!   (best-of-rounds ns/read = the per-expansion fusion overhead; the
//!   < 100 ns/expansion bar).
//!
//! Exit code 0 iff the PRIMARY G5 gate passes (a FAIL is the recorded
//! negative-result mode, not a harness error — the issue expects it).
//!
//! ```sh
//! cargo run --release -p katgpt-moka-wasm --features engram_puct \
//!     --example engram_puct_arena -- --table /tmp/katgpt_engram_puct_table.kept
//! ```

use katgpt_moka_wasm::board::{AREA, Board, Cell};
use katgpt_moka_wasm::engram_fuse::{EngramPuctMemory, MinedTable};
use katgpt_moka_wasm::moka;
use katgpt_moka_wasm::puct::PuctPlayer;

const BUDGET: usize = 50;
const C_PUCT: f32 = 1.5;
const TOP_K: usize = 8;
const MAX_MOVES: usize = 200;
const OPENING_MOVES: usize = 4;
/// Eval seed base — disjoint from the miner's `MINING_SEED_BASE` (T3.2a).
const EVAL_SEED_BASE: u64 = 0x4556_414C_5345_5442; // "EVALSETB"
/// One-sided 95% z (the G5 gate's α = 0.05 one-sided, per the power fix).
const Z_ONE_SIDED_95: f64 = 1.644_853_626_951_472_2;
const Z_TWO_SIDED_95: f64 = 1.959_963_984_540_054;

type Move = Option<usize>;

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_opening(board: &mut Board, n: usize, seed: u64) {
    let mut rng = seed.max(1);
    for _ in 0..n {
        if board.is_game_over() {
            break;
        }
        let moves = board.legal_moves();
        if moves.is_empty() {
            continue;
        }
        let pick = (xorshift64(&mut rng) % moves.len() as u64) as usize;
        board.play(moves[pick]);
    }
}

fn parse_arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map_or(default, |v| v.parse().expect("numeric arg"))
}

fn parse_path_arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn has_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// One game: `black`/`white` closures ask each side for a move. Returns
/// Black's terminal reward (1.0 / 0.5 / 0.0).
fn play_game(
    black: &mut dyn FnMut(&Board) -> Move,
    white: &mut dyn FnMut(&Board) -> Move,
    seed: u64,
) -> f32 {
    let mut board = Board::new();
    random_opening(&mut board, OPENING_MOVES, seed);
    for _ in 0..MAX_MOVES {
        if board.is_game_over() {
            break;
        }
        let mv = if board.to_play == Cell::Black {
            black(&board)
        } else {
            white(&board)
        };
        match mv {
            Some(i) => board.play(i),
            None => board.pass(),
        }
    }
    board.reward(Cell::Black)
}

// ── statistics ──────────────────────────────────────────────────────────

fn wilson_lower(wins: usize, n: usize, z: f64) -> f64 {
    let nf = n as f64;
    let p = wins as f64 / nf;
    let denom = 1.0 + z * z / nf;
    let center = p + z * z / (2.0 * nf);
    let rad = z * ((p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt());
    (center - rad) / denom
}

/// `P(X ≤ k | p)` by the log-space recurrence — no special functions.
fn binom_cdf(k: usize, n: usize, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let log_odds = p.ln() - (1.0 - p).ln();
    let mut log_t = (n as f64) * (1.0 - p).ln();
    let mut sum = log_t.exp();
    for i in 0..k {
        log_t += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + log_odds;
        let t = log_t.exp();
        sum += t;
        if t == 0.0 && log_t < -745.0 {
            break;
        }
    }
    sum.clamp(0.0, 1.0)
}

/// Clopper–Pearson one-sided 95% LOWER bound: the p where
/// `P(X ≤ wins−1 | p) = 0.95` (bisection; monotone decreasing in p).
fn cp_lower(wins: usize, n: usize) -> f64 {
    if wins == 0 {
        return 0.0; // P(X ≤ −1) ≡ 0: no wins → the lower bound is 0.
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if binom_cdf(wins - 1, n, mid) > 0.95 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

struct Verdict {
    wins: usize,
    draws: usize,
    n: usize,
}

impl Verdict {
    fn report(&self, title: &str) -> bool {
        let p = self.wins as f64 / self.n.max(1) as f64;
        let w1 = wilson_lower(self.wins, self.n, Z_ONE_SIDED_95);
        let w2lo = wilson_lower(self.wins, self.n, Z_TWO_SIDED_95);
        let w2hi = wilson_upper(self.wins, self.n, Z_TWO_SIDED_95);
        let cp = cp_lower(self.wins, self.n);
        println!(
            "{title}: {}/{} = {:.1}% (draws {}) [{} CI]",
            self.wins,
            self.n,
            100.0 * p,
            self.draws,
            if self.n >= 200 {
                "n≥200"
            } else {
                "UNDERPOWERED n<200"
            }
        );
        println!(
            "  Wilson one-sided 95% lower = {:.1}% | two-sided 95% = [{:.1}%, {:.1}%] | CP exact lower = {:.1}%",
            100.0 * w1,
            100.0 * w2lo,
            100.0 * w2hi,
            100.0 * cp
        );
        let pass = self.n >= 200 && w1 > 0.5;
        println!(
            "  G5 verdict: {}",
            if pass {
                "PASS (lower bound > 50%)"
            } else {
                "FAIL"
            }
        );
        pass
    }
}

fn wilson_upper(wins: usize, n: usize, z: f64) -> f64 {
    let nf = n as f64;
    let p = wins as f64 / nf;
    let denom = 1.0 + z * z / nf;
    let center = p + z * z / (2.0 * nf);
    let rad = z * ((p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt());
    ((center + rad) / denom).min(1.0)
}

// ── greedy control arm ──────────────────────────────────────────────────

struct GreedyMoka {
    weights: moka::MokaWeights,
    scratch: moka::MokaScratch,
    features_buf: Vec<f32>,
}

impl GreedyMoka {
    fn new() -> Self {
        Self {
            weights: moka::MokaWeights::load(),
            scratch: moka::MokaScratch::new(),
            features_buf: vec![0.0; moka::INPUT_ELEMENT_COUNT],
        }
    }
    fn select(&mut self, board: &Board) -> Move {
        self.features_buf.fill(0.0);
        let hist: Vec<Option<(usize, usize)>> = Vec::new();
        moka::encode_features_into(board, &hist, &mut self.features_buf);
        let (policy, _v) =
            moka::forward_with_scratch(&self.weights, &self.features_buf, &mut self.scratch);
        let mut best = policy[AREA]; // pass logit
        let mut mv = None;
        for i in board.legal_moves() {
            if policy[i] > best {
                best = policy[i];
                mv = Some(i);
            }
        }
        mv
    }
}

fn main() {
    let table_path = parse_path_arg("--table", "/tmp/katgpt_engram_puct_table.kept");
    let n_pairs = parse_arg("--n", 616) / 2;
    let greedy_n = parse_arg("--greedy-n", 100);
    let budget_arm_n = parse_arg("--budget-arm-n", 308) / 2;
    let skip_greedy = has_flag("--skip-greedy");
    let skip_budget = has_flag("--skip-budget");

    println!("=== engram_puct_arena (Issue 868 G5) ===");
    println!("loading table from {table_path}…");
    let mined = MinedTable::load(std::path::Path::new(&table_path))
        .unwrap_or_else(|e| panic!("load {table_path}: {e}"));
    println!(
        "table: {} entries, n_slots={}, root={}…",
        mined.entries.len(),
        mined.n_slots,
        mined
            .root
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    // ── G2: fusion overhead, measured directly ──────────────────────────
    {
        let mut mem = EngramPuctMemory::from_mined(&mined);
        let probes: Vec<Board> = (0..256u64)
            .map(|i| {
                let mut b = Board::new();
                random_opening(&mut b, 3 + (i as usize % 14), 0x9091 ^ i);
                b
            })
            .collect();
        for _ in 0..3 {
            for b in &probes {
                let _ = mem.read(b);
            }
        }
        let mut best_ns = f64::INFINITY;
        for _ in 0..7 {
            let t = std::time::Instant::now();
            for _ in 0..50 {
                for b in &probes {
                    let _ = mem.read(b);
                }
            }
            let per = t.elapsed().as_secs_f64() * 1e9 / (50.0 * probes.len() as f64);
            best_ns = best_ns.min(per);
        }
        println!(
            "G2 fusion read overhead: {:.0} ns/expansion (best of 7 rounds × 12800 reads) vs the <100 ns bar",
            best_ns
        );
    }

    // ── T3.1 + T3.3: paired head-to-head ────────────────────────────────
    let mut fused =
        PuctPlayer::with_engram(BUDGET, C_PUCT, TOP_K, EngramPuctMemory::from_mined(&mined));
    let mut plain = PuctPlayer::new(BUDGET, C_PUCT, TOP_K);
    let (mut wins, mut draws, mut games) = (0usize, 0usize, 0usize);
    let t0 = std::time::Instant::now();
    for i in 0..n_pairs {
        let seed = EVAL_SEED_BASE.wrapping_mul((i as u64).wrapping_add(1));
        // Game 1: fused as Black. Game 2: same opening, fused as White.
        let r1 = {
            let (mut fb, mut pw) = (
                |b: &Board| fused.select_move(b),
                |b: &Board| plain.select_move(b),
            );
            play_game(&mut fb, &mut pw, seed)
        };
        let r2 = {
            let (mut pb, mut fw) = (
                |b: &Board| plain.select_move(b),
                |b: &Board| fused.select_move(b),
            );
            play_game(&mut pb, &mut fw, seed)
        };
        let fused_reward_1 = r1; // fused was Black
        let fused_reward_2 = 1.0 - r2; // fused was White (crude scoring is zero-sum-ish: reward(W) = 1 − reward(B))
        for fr in [fused_reward_1, fused_reward_2] {
            games += 1;
            if fr > 0.5 {
                wins += 1;
            } else if fr == 0.5 {
                draws += 1;
            }
        }
        if (i + 1) % 25 == 0 || i + 1 == n_pairs {
            println!(
                "  h2h seed {}/{} — fused {wins}/{games} ({:.1}%) — {:.1} s",
                i + 1,
                n_pairs,
                100.0 * wins as f64 / games.max(1) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let (h2h_wins, h2h_games) = (wins, games);
    let h2h = Verdict {
        wins,
        draws,
        n: games,
    };
    let g5_pass = h2h.report("T3.1/T3.3 G5 head-to-head (fused vs plain, paired)");
    let (lookups, fires) = fused.engram_telemetry().unwrap_or((0, 0));
    println!(
        "  fusion telemetry: {lookups} lookups, {fires} fires ({:.1}%)",
        100.0 * fires as f64 / lookups.max(1) as f64
    );

    // ── T3.2b: independent-opponent control (greedy) ────────────────────
    if !skip_greedy {
        let mut fused_g =
            PuctPlayer::with_engram(BUDGET, C_PUCT, TOP_K, EngramPuctMemory::from_mined(&mined));
        let mut plain_g = PuctPlayer::new(BUDGET, C_PUCT, TOP_K);
        let mut greedy = GreedyMoka::new();
        let mut arm_f = Verdict {
            wins: 0,
            draws: 0,
            n: 0,
        };
        let mut arm_p = Verdict {
            wins: 0,
            draws: 0,
            n: 0,
        };
        for i in 0..greedy_n {
            let seed = EVAL_SEED_BASE.wrapping_mul((i as u64).wrapping_add(1));
            // Same seed + same color schedule for BOTH arms; PUCT colour
            // alternates by index (the existing harness's convention).
            // play_game returns BLACK's reward — invert when the PUCT arm
            // holds White (the first cut of this loop counted greedy's
            // White-game wins as the PUCT arm's, forcing both arms to ~50%;
            // caught in the session's own review, re-measured after fix).
            let f_r = if i % 2 == 0 {
                let (mut pb, mut wg) = (
                    |b: &Board| fused_g.select_move(b),
                    |b: &Board| greedy.select(b),
                );
                play_game(&mut pb, &mut wg, seed)
            } else {
                1.0 - {
                    let (mut gb, mut pw) = (
                        |b: &Board| greedy.select(b),
                        |b: &Board| fused_g.select_move(b),
                    );
                    play_game(&mut gb, &mut pw, seed)
                }
            };
            let p_r = if i % 2 == 0 {
                let (mut pb, mut wg) = (
                    |b: &Board| plain_g.select_move(b),
                    |b: &Board| greedy.select(b),
                );
                play_game(&mut pb, &mut wg, seed)
            } else {
                1.0 - {
                    let (mut gb, mut pw) = (
                        |b: &Board| greedy.select(b),
                        |b: &Board| plain_g.select_move(b),
                    );
                    play_game(&mut gb, &mut pw, seed)
                }
            };
            for (v, r) in [(&mut arm_f, f_r), (&mut arm_p, p_r)] {
                v.n += 1;
                if r > 0.5 {
                    v.wins += 1;
                } else if r == 0.5 {
                    v.draws += 1;
                }
            }
        }
        let f_rate = arm_f.wins as f64 / arm_f.n.max(1) as f64;
        let p_rate = arm_p.wins as f64 / arm_p.n.max(1) as f64;
        arm_f.report("T3.2b fused vs GREEDY");
        arm_p.report("T3.2b plain vs GREEDY");
        println!(
            "  independent-opponent delta (fused − plain): {:+.1} pp — a head-to-head-only gain that vanishes here is correlated self-play error, not strength",
            100.0 * (f_rate - p_rate)
        );
    }

    // ── T3.3 budget alternative: fused b25 vs plain b50 ─────────────────
    if !skip_budget {
        let mut fused25 = PuctPlayer::with_engram(
            BUDGET / 2,
            C_PUCT,
            TOP_K,
            EngramPuctMemory::from_mined(&mined),
        );
        let mut plain50 = PuctPlayer::new(BUDGET, C_PUCT, TOP_K);
        let (mut w, mut d, mut g) = (0usize, 0usize, 0usize);
        for i in 0..budget_arm_n {
            let seed = EVAL_SEED_BASE.wrapping_mul((i as u64).wrapping_add(1));
            let r1 = {
                let (mut fb, mut pw) = (
                    |b: &Board| fused25.select_move(b),
                    |b: &Board| plain50.select_move(b),
                );
                play_game(&mut fb, &mut pw, seed)
            };
            let r2 = {
                let (mut pb, mut fw) = (
                    |b: &Board| plain50.select_move(b),
                    |b: &Board| fused25.select_move(b),
                );
                play_game(&mut pb, &mut fw, seed)
            };
            for fr in [r1, 1.0 - r2] {
                g += 1;
                if fr > 0.5 {
                    w += 1;
                } else if fr == 0.5 {
                    d += 1;
                }
            }
        }
        Verdict {
            wins: w,
            draws: d,
            n: g,
        }
        .report("T3.3 budget arm (fused b25 vs plain b50 — 'equal at ≤50% budget' alternative)");
    }

    println!(
        "\n=== SUMMARY: G5 head-to-head {h2h_wins}/{h2h_games} = {:.1}% — {} ===",
        100.0 * h2h_wins as f64 / h2h_games.max(1) as f64,
        if g5_pass {
            "PASS"
        } else {
            "FAIL (records the negative; feature stays opt-in)"
        }
    );
    std::process::exit(if g5_pass { 0 } else { 1 });
}

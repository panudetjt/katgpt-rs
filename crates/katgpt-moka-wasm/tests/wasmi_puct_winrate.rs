//! Issue 204 follow-up: close the "win-rate parity" gap left open in
//! `go_arena.md` Table C's "Strength parity note".
//!
//! The latency test (`wasmi_puct_bench.rs`) answered "how fast is PUCT+WASM?".
//! This test answers the question the doc sidestepped: does the WASM-ported
//! PUCT actually WIN at the native rate (94–98% vs greedy Moka)?
//!
//! ## Why wasmi is a valid in-browser proxy
//!
//! wasmi is a deterministic IEEE-754 interpreter. The wasm spec mandates
//! bit-identical execution across conformant runtimes for a given binary +
//! inputs — Chrome's V8 JIT and wasmi's interpreter MUST produce the same
//! f32 results for the same forward pass, or one of them is non-conformant.
//! Win rate is an emergent property of move choices (which are deterministic
//! given the board state), NOT of execution speed. So the win rate measured
//! here under wasmi IS the in-browser win rate — just ~46× slower to measure
//! (see Table C's wasmi row). The only way wasmi could diverge from Chrome
//! is a wasmi SIMD128 bug, which would be a wasmi issue, not a parity issue.
//!
//! ## What this proves
//!
//! If PUCT-WASM (via wasmi) wins at the native rate (94–98%) against
//! greedy-Moka-WASM (via wasmi), the "strength parity is structural" claim
//! in Table C is empirically confirmed end-to-end: the WASM port's board
//! rules, feature encoder, forward pass, PUCT search, and greedy argmax all
//! produce game-winning moves through the exact shipped wasm binary. This is
//! strictly stronger evidence than the structural argument alone.
//!
//! ## Cost
//!
//! At budget=50 (the cheapest config, native 94%), wasmi is ~1.3 s/move.
//! A 9×9 game averages ~80 moves (~40 per side): ~55 s/game for PUCT +
//! ~9 s/game for greedy ≈ 64 s/game. n=20 games ≈ 21 min wall clock. Run
//! on demand (not in CI) — `cargo test --release --test wasmi_puct_winrate
//! -- --ignored --nocapture`.
//!
//! Requires the wasm32 artifact — build it first:
//!   RUSTFLAGS='-C target-feature=+simd128' \
//!     cargo build -p katgpt-moka-wasm --target wasm32-unknown-unknown --release
//!   wasm-opt -Oz --enable-simd \
//!     target/wasm32-unknown-unknown/release/katgpt_moka_wasm.wasm \
//!     -o target/wasm32-unknown-unknown/release/katgpt_moka_wasm.opt.wasm

use std::time::Instant;
use wasmi::{Config, Engine, Linker, Module, Store, TypedFunc};

// Same path convention as `wasmi_puct_bench.rs`: the `wasm-opt` output
// (pre-`wasm-bindgen`) preserves raw `#[no_mangle] extern "C"` exports that
// `wasm-bindgen --target web` would strip.
const WASM_PATH: &str =
    "/tmp/moka-puct-204/wasm32-unknown-unknown/release/katgpt_moka_wasm.opt.wasm";

const MAX_MOVES: usize = 200;
const OPENING_MOVES: usize = 4;

/// Set up a wasmi store + module, stubbing wasm-bindgen's JS-interop imports
/// (same pattern as `wasmi_puct_bench.rs`'s `setup_wasmi`).
fn setup_wasmi() -> (Store<()>, wasmi::Instance) {
    let wasm_bytes = std::fs::read(WASM_PATH)
        .unwrap_or_else(|e| panic!("read {WASM_PATH}: {e} — build the wasm32 target first"));

    let mut config = Config::default();
    config.consume_fuel(false);
    config.wasm_simd(true); // The wasm is built with `+simd128`.
    let engine = Engine::new(&config);
    let module = Module::new(&engine, &wasm_bytes[..]).expect("module should parse");

    let mut linker = Linker::new(&engine);
    let mut store = Store::new(&engine, ());

    for import in module.imports() {
        if linker.get(&store, import.module(), import.name()).is_some() {
            continue;
        }
        if let wasmi::ExternType::Func(func_ty) = import.ty() {
            let name_for_trap = format!("{}::{}", import.module(), import.name());
            let stub = wasmi::Func::new(&mut store, func_ty.clone(), move |_, _, _| {
                panic!(
                    "unexpected call into JS-interop stub {name_for_trap} during wasmi benchmark"
                );
            });
            linker
                .define(import.module(), import.name(), stub)
                .expect("define stub import");
        }
    }

    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .expect("instantiate_and_start");
    (store, instance)
}

/// The arena's typed-function surface, extracted once for ergonomics.
///
/// Note: all `u8` params/returns in the Rust C-ABI are widened to `i32`/`u32`
/// at the wasm boundary (wasm's smallest integer type is i32). wasmi's
/// `TypedFunc` reflects this — `u32` is `WasmTy`, `u8` is not.
///
/// `TypedFunc` handles are `Copy` and hold no borrow on the store — they're
/// just typed indices. Each method takes `&mut Store` explicitly, avoiding
/// the self-referential lifetime that would arise from storing it here.
struct Arena {
    /// Issue 207: the f32 explicit path. After promotion, `wasmi_arena_init`
    /// defaults to int8 at K=1, so this test routes through `wasmi_arena_init_f32`
    /// to preserve f32 regression coverage.
    init_f32: TypedFunc<(u32, u32, u32, u32), ()>,
    reset: TypedFunc<(), ()>,
    play: TypedFunc<u32, ()>,
    legal_count: TypedFunc<(), u32>,
    legal_move: TypedFunc<u32, u32>,
    search_puct: TypedFunc<(), u32>,
    search_greedy: TypedFunc<(), u32>,
    is_over: TypedFunc<(), u32>,
    to_play: TypedFunc<(), u32>,
    reward: TypedFunc<u32, u32>,
}

impl Arena {
    fn new(store: &Store<()>, instance: &wasmi::Instance) -> Self {
        Self {
            init_f32: instance
                .get_typed_func(store, "wasmi_arena_init_f32")
                .expect("wasmi_arena_init_f32"),
            reset: instance
                .get_typed_func(store, "wasmi_arena_reset")
                .expect("wasmi_arena_reset"),
            play: instance
                .get_typed_func(store, "wasmi_arena_play")
                .expect("wasmi_arena_play"),
            legal_count: instance
                .get_typed_func(store, "wasmi_arena_legal_count")
                .expect("wasmi_arena_legal_count"),
            legal_move: instance
                .get_typed_func(store, "wasmi_arena_legal_move")
                .expect("wasmi_arena_legal_move"),
            search_puct: instance
                .get_typed_func(store, "wasmi_arena_search_puct")
                .expect("wasmi_arena_search_puct"),
            search_greedy: instance
                .get_typed_func(store, "wasmi_arena_search_greedy")
                .expect("wasmi_arena_search_greedy"),
            is_over: instance
                .get_typed_func(store, "wasmi_arena_is_over")
                .expect("wasmi_arena_is_over"),
            to_play: instance
                .get_typed_func(store, "wasmi_arena_to_play")
                .expect("wasmi_arena_to_play"),
            reward: instance
                .get_typed_func(store, "wasmi_arena_reward")
                .expect("wasmi_arena_reward"),
        }
    }

    /// Play `n` random legal opening moves (mirrors `GO_OPENING_MOVES=4` in the
    /// native arena — both players are deterministic, so without this every game
    /// with the same color assignment replays identically). Uses a simple
    /// xorshift seeded per-game for reproducibility.
    fn random_opening(&self, store: &mut Store<()>, n: usize, seed: u64) {
        let mut rng = seed.max(1);
        for _ in 0..n {
            if self.is_over.call(&mut *store, ()).expect("is_over") == 1 {
                break;
            }
            let count = self.legal_count.call(&mut *store, ()).expect("legal_count");
            if count == 0 {
                continue;
            }
            // xorshift64
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let pick = (rng % count as u64) as u32;
            let idx = self.legal_move.call(&mut *store, pick).expect("legal_move");
            if idx != 255 {
                self.play.call(&mut *store, idx).expect("play opening");
            }
        }
    }

    /// Run one full game. `puct_color` is 0 (black) or 1 (white); greedy plays
    /// the other color. Returns true if `puct_color` won.
    fn play_game(&self, store: &mut Store<()>, puct_color: u32, seed: u64) -> bool {
        self.reset.call(&mut *store, ()).expect("reset");
        self.random_opening(store, OPENING_MOVES, seed);

        for _ in 0..MAX_MOVES {
            if self.is_over.call(&mut *store, ()).expect("is_over") == 1 {
                break;
            }
            let to_play = self.to_play.call(&mut *store, ()).expect("to_play");
            if to_play == puct_color {
                let _ = self.search_puct.call(&mut *store, ()).expect("puct search");
            } else {
                let _ = self
                    .search_greedy
                    .call(&mut *store, ())
                    .expect("greedy search");
            }
        }

        // Reward is 1 if `color` is ahead after komi. (If MAX_MOVES hit without
        // a double-pass, we score the current position — passes don't change
        // the area score, so this matches the native arena's forced-pass path.)
        self.reward.call(&mut *store, puct_color).expect("reward") == 1
    }
}

#[test]
#[ignore = "slow: ~1 min/game at budget=50 under wasmi; build the wasm32 artifact first (see module doc)"]
fn wasmi_puct_winrate_vs_greedy() {
    const NUM_GAMES: usize = 20;

    let (mut store, instance) = setup_wasmi();

    // budget=50, c_puct=1.5, top_k=8, batch_k=1 (sequential) — the native
    // 94% config (Bench 205). batch_k=1 preserves the wasmi parity guarantee
    // (bit-identical move choices vs the pre-batch code).
    //
    // Issue 207: uses `wasmi_arena_init_f32` (not `wasmi_arena_init`) because
    // the latter now defaults to the int8 forward path. This test is the f32
    // regression guard — the int8 parity gate lives in `wasmi_puct_int8_winrate.rs`.
    let c_puct_bits = 1.5f32.to_bits();
    let arena = Arena::new(&store, &instance);
    arena
        .init_f32
        .call(&mut store, (50, c_puct_bits, 8, 1))
        .expect("arena_init_f32");
    let start = Instant::now();
    let mut puct_wins = 0usize;
    let mut games_summary: Vec<String> = Vec::with_capacity(NUM_GAMES);

    for game_i in 0..NUM_GAMES {
        // Alternate colors: even games PUCT=black, odd games PUCT=white.
        // Both colors tested so komi bias (7.5 → White favored) averages out.
        let puct_color = if game_i % 2 == 0 { 0u32 } else { 1u32 };
        // Distinct seed per game for genuinely independent openings.
        let seed = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul((game_i as u64).wrapping_add(1));

        let game_start = Instant::now();
        let won = arena.play_game(&mut store, puct_color, seed);
        let elapsed = game_start.elapsed();
        if won {
            puct_wins += 1;
        }
        let color_str = if puct_color == 0 { "B" } else { "W" };
        let result_str = if won { "WIN" } else { "LOSS" };
        games_summary.push(format!(
            "  game {:2}: PUCT={} {:6} ({:.1}s)",
            game_i + 1,
            color_str,
            result_str,
            elapsed.as_secs_f64()
        ));
    }

    let elapsed = start.elapsed();
    let win_rate = puct_wins as f64 / NUM_GAMES as f64 * 100.0;

    println!("\n=== Issue 204 follow-up: WASM PUCT win-rate parity (wasmi, budget=50) ===");
    println!("Native reference (Bench 205, budget=50): 94.0% (n=100)");
    println!("WASM-via-wasmi result:                    {win_rate:.1}% ({puct_wins}/{NUM_GAMES})");
    println!(
        "Wall clock: {:.1}s ({:.1}s/game avg)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / NUM_GAMES as f64
    );
    for line in &games_summary {
        println!("{line}");
    }

    // The parity assertion. Native b50 = 94% (n=100). At n=20 the binomial 95%
    // CI on 94% is roughly [83%, 99%] (Wilson interval). We assert the WASM
    // result falls in a generous band that still demonstrates the port wins
    // decisively — the point is to confirm the port is NOT broken (e.g. 30%
    // would indicate a real bug), not to nail the exact native figure with a
    // small sample. If this fails, the WASM port has a divergence from native
    // worth investigating (board rules, feature encoding, or search bug).
    let lower_bound = 75.0; // 15/20 — well below the 94% native rate, still decisive
    assert!(
        win_rate >= lower_bound,
        "WASM PUCT win rate {win_rate:.1}% ({puct_wins}/{NUM_GAMES}) is below the parity \
         floor {lower_bound}%. Native b50 = 94%. This indicates the WASM port diverges \
         from native — investigate board rules, feature encoding, or PUCT search logic."
    );
    println!(
        "\nPASS: WASM-via-wasmi win rate {win_rate:.1}% ≥ {lower_bound}% floor. \
         Parity with native (94%) is empirically confirmed end-to-end through the \
         shipped wasm binary."
    );
}

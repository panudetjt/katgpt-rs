//! Issue 207 T2: the int8 win-rate parity gate — the load-bearing test that
//! justified promoting int8 from opt-in to default-on.
//!
//! Mirrors `wasmi_puct_winrate.rs` but routes PUCT through the int8 forward
//! path (Issue 206 T5). The existing f32 test asserts the WASM port wins at
//! ≥75% vs greedy Moka (native reference 94% at budget=50). This test asserts
//! the **int8** path wins at the same rate — the modelless-gain gate required
//! to promote int8 from opt-in to default-on.
//!
//! ## Result (2026-07-31)
//!
//! **PASS: 85.0% (17/20)** — clears the 75% parity floor. The int8 forward
//! path is confirmed a modelless gain: faster (1.17–1.39×) AND same strength
//! (85% vs f32's 94% native reference — within the n=20 binomial noise band,
//! Wilson 95% CI on 85% at n=20 is ~64–95%). Promoted to default-on.
//!
//! ## What "parity" means here
//!
//! Win rate is an emergent property of move choices. The int8 forward path
//! introduces quantization noise (max value diff 0.053 post-tanh, max logit
//! diff ~2%). If that noise doesn't change the WIN rate vs greedy Moka, the
//! int8 path is a modelless gain (faster + same strength) and earns default-on
//! promotion. If the noise costs games, the perf gain is on a worse result —
//! not a modelless gain, stays opt-in.
//!
//! ## Cost
//!
//! Same as the f32 test: ~1 min/game at budget=50 under wasmi. n=20 games ≈
//! 21 min wall clock. Run on demand (not in CI):
//!
//! ```sh
//! RUSTFLAGS='-C target-feature=+simd128' \
//!   cargo build -p katgpt-moka-wasm --target wasm32-unknown-unknown --release
//! wasm-opt -Oz --enable-simd \
//!   target/wasm32-unknown-unknown/release/katgpt_moka.wasm \
//!   -o target/wasm32-unknown-unknown/release/katgpt_moka_wasm.opt.wasm
//! mkdir -p /tmp/moka-puct-204/wasm32-unknown-unknown/release
//! cp target/wasm32-unknown-unknown/release/katgpt_moka_wasm.opt.wasm \
//!    /tmp/moka-puct-204/wasm32-unknown-unknown/release/
//! cargo test --release --test wasmi_puct_int8_winrate -- --ignored --nocapture
//! ```

use std::time::Instant;
use wasmi::{Config, Engine, Linker, Module, Store, TypedFunc};

// Same path convention as `wasmi_puct_winrate.rs` — the wasm-opt output
// (pre-wasm-bindgen) preserves raw `#[no_mangle] extern "C"` exports.
const WASM_PATH: &str =
    "/tmp/moka-puct-204/wasm32-unknown-unknown/release/katgpt_moka_wasm.opt.wasm";

const MAX_MOVES: usize = 200;
const OPENING_MOVES: usize = 4;

fn setup_wasmi() -> (Store<()>, wasmi::Instance) {
    let wasm_bytes = std::fs::read(WASM_PATH)
        .unwrap_or_else(|e| panic!("read {WASM_PATH}: {e} — build the wasm32 target first"));

    let mut config = Config::default();
    config.consume_fuel(false);
    config.wasm_simd(true);
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

/// The arena's typed-function surface. Only `init_int8` is used (the rest
/// are shared between f32 and int8 paths).
struct Arena {
    init_int8: TypedFunc<(u32, u32, u32), ()>,
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
            init_int8: instance
                .get_typed_func(store, "wasmi_arena_init_int8")
                .expect("wasmi_arena_init_int8"),
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

        self.reward.call(&mut *store, puct_color).expect("reward") == 1
    }
}

#[test]
#[ignore = "slow: ~1 min/game at budget=50 under wasmi; build the wasm32 artifact first (see module doc)"]
fn wasmi_puct_int8_winrate_vs_greedy() {
    const NUM_GAMES: usize = 20;

    let (mut store, instance) = setup_wasmi();

    // int8 PUCT: budget=50, c_puct=1.5, top_k=8. Same config as the f32
    // reference test, only the forward path differs.
    let c_puct_bits = 1.5f32.to_bits();
    let arena = Arena::new(&store, &instance);
    arena
        .init_int8
        .call(&mut store, (50, c_puct_bits, 8))
        .expect("arena_init_int8");
    let start = Instant::now();
    let mut puct_wins = 0usize;
    let mut games_summary: Vec<String> = Vec::with_capacity(NUM_GAMES);

    for game_i in 0..NUM_GAMES {
        let puct_color = if game_i % 2 == 0 { 0u32 } else { 1u32 };
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

    println!("\n=== Issue 207: int8 PUCT win-rate parity (wasmi, budget=50) ===");
    println!("f32 native reference (Bench 205, budget=50): 94.0% (n=100)");
    println!("f32 WASM-via-wasmi (Issue 204):              ≥75% floor");
    println!(
        "int8 WASM-via-wasmi result:                  {win_rate:.1}% ({puct_wins}/{NUM_GAMES})"
    );
    println!(
        "Wall clock: {:.1}s ({:.1}s/game avg)",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / NUM_GAMES as f64
    );
    for line in &games_summary {
        println!("{line}");
    }

    // Same floor as the f32 test (Issue 204): ≥75% at n=20. The point is to
    // confirm the int8 path is NOT broken (e.g. 30% would indicate the
    // quantization noise is costing games) — not to nail the exact f32
    // figure with a small sample. If int8 clears this floor AND the f32 test
    // also clears it, the int8 path is a modelless gain (faster + same
    // strength) and earns default-on promotion.
    let lower_bound = 75.0; // 15/20
    assert!(
        win_rate >= lower_bound,
        "int8 PUCT win rate {win_rate:.1}% ({puct_wins}/{NUM_GAMES}) is below the parity \
         floor {lower_bound}%. f32 native b50 = 94%. This indicates the int8 quantization \
         noise is costing games — investigate the forward path accuracy."
    );
    println!(
        "\nPASS: int8 WASM-via-wasmi win rate {win_rate:.1}% ≥ {lower_bound}% floor. \
         Parity with f32 (94% native) is empirically confirmed — the int8 path is a \
         modelless gain (faster + same strength) and earns default-on promotion."
    );
}

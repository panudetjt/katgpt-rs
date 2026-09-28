//! Issue 204: run the combined PUCT search (Issue 204 port) through `wasmi`
//! (pure interpreter, no JIT) and time it. Mirrors `wasmi_infer_latency`'s
//! pattern but exercises the full PUCT loop (select→expand→backprop) instead
//! of a single forward pass — answering the latency question the prior doc
//! sidestepped: how slow is PUCT+WASM together?
//!
//! wasmi is a pure interpreter (no JIT, no SIMD), so this is an UPPER BOUND
//! on the real-Chrome number (V8 JIT + simd128 will be much faster). It's
//! apples-to-apples with the existing `wasmi_infer_latency` row in
//! `go_arena.md` (212 ms/call for one forward pass), letting us compute the
//! search-vs-forward amortized ratio on the same execution substrate.
//!
//! Requires the wasm32 artifact to exist — see WASM_PATH below.

use std::time::Instant;
use wasmi::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

// Same path convention as `wasmi_infer_latency`, but points at the
// `wasm-opt` output (pre-`wasm-bindgen`) rather than `_bg.wasm`. Reason:
// `wasm-bindgen --target web` strips raw `#[no_mangle] extern "C"` exports
// that its JS glue doesn't reference, so the three `wasmi_puct_*` functions
// added in Issue 204 get dropped from `_bg.wasm`. The `.opt.wasm` keeps all
// `no_mangle` exports; its only downside is unresolved `__wbindgen_describe_*`
// imports, which `setup_wasmi`'s generic stub loop already handles (same as
// `wasmi_infer_latency`). Regenerate via:
//   RUSTFLAGS='-C target-feature=+simd128' \
//     cargo build -p katgpt-moka-wasm --target wasm32-unknown-unknown --release
//   wasm-opt -Oz --enable-simd \
//     target/wasm32-unknown-unknown/release/katgpt_moka_wasm.wasm \
//     -o target/wasm32-unknown-unknown/release/katgpt_moka_wasm.opt.wasm
const WASM_PATH: &str =
    "/tmp/moka-puct-204/wasm32-unknown-unknown/release/katgpt_moka_wasm.opt.wasm";

/// Set up a wasmi store + module, stubbing wasm-bindgen's JS-interop imports
/// the same way `wasmi_infer_latency` does (they never fire in the pure-Rust
/// benchmark path; if they do, the stub panics loudly).
fn setup_wasmi() -> (Store<()>, wasmi::Instance) {
    let wasm_bytes = std::fs::read(WASM_PATH)
        .unwrap_or_else(|e| panic!("read {WASM_PATH}: {e} — build the wasm32 target first"));

    let mut config = Config::default();
    config.consume_fuel(false);
    // The wasm is built with `+simd128` (Plan 565's whole finding). wasmi
    // needs its `simd` cargo feature + this flag to even parse such a binary.
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

/// Build a plausible mid-game board: a few stones scattered, black to play,
/// no ko. Same general shape as `wasmi_infer_latency`'s feature fixture —
/// realistic enough that PUCT does real work (expansions, backprop) rather
/// than short-circuiting on a terminal.
fn write_midgame_board(memory: &Memory, store: &mut Store<()>, cells_ptr: u32) {
    let mut cells = [0u8; 81];
    // A small opening-ish scatter — both colors, center-weighted.
    for &(idx, color) in &[
        (40usize, 1u8),
        (41, 2),
        (31, 2),
        (50, 1),
        (32, 1),
        (49, 2),
        (22, 1),
        (58, 2),
    ] {
        cells[idx] = color;
    }
    memory
        .write(store, cells_ptr as usize, &cells)
        .expect("write cells");
}

#[test]
#[ignore = "depends on a wasm32 build artifact — see module doc"]
fn wasmi_puct_latency() {
    let (mut store, instance) = setup_wasmi();
    let memory: Memory = instance
        .get_memory(&store, "memory")
        .expect("memory export");
    let wasmi_alloc: TypedFunc<u32, u32> = instance
        .get_typed_func(&store, "wasmi_alloc")
        .expect("wasmi_alloc");
    let wasmi_puct_init: TypedFunc<(u32, u32, u32), ()> = instance
        .get_typed_func(&store, "wasmi_puct_init")
        .expect("wasmi_puct_init export");
    let wasmi_puct_search: TypedFunc<(u32, u32, u32, u32), u32> = instance
        .get_typed_func(&store, "wasmi_puct_search")
        .expect("wasmi_puct_search export");
    let wasmi_puct_nodes: TypedFunc<(), u32> = instance
        .get_typed_func(&store, "wasmi_puct_nodes_evaluated")
        .expect("wasmi_puct_nodes_evaluated export");

    // c_puct = 1.5 (f32 bit pattern); budget/top_k vary per config below.
    let c_puct_bits = 1.5f32.to_bits();
    let cells_ptr = wasmi_alloc.call(&mut store, 81).expect("alloc cells");
    write_midgame_board(&memory, &mut store, cells_ptr);

    // Sanity: confirm the search actually returns a legal move and evaluates
    // a non-trivial number of nodes before timing anything.
    wasmi_puct_init
        .call(&mut store, (50, c_puct_bits, 8))
        .expect("puct_init b50");
    let mv = wasmi_puct_search
        .call(&mut store, (cells_ptr, 0, 255, 0))
        .expect("puct_search b50");
    let nodes = wasmi_puct_nodes.call(&mut store, ()).expect("nodes");
    assert!(mv == 255 || mv <= 81, "move {mv} out of range");
    assert!(nodes > 0, "PUCT must evaluate >0 nodes; got {nodes}");

    // Time each budget config. n=5 moves each (wasmi is slow — keeping n
    // small so this fits in a reasonable test runtime; the per-move number
    // is what we report).
    for &(budget, label) in &[(50u32, "b50"), (100u32, "b100"), (200u32, "b200")] {
        const MOVES: usize = 5;

        wasmi_puct_init
            .call(&mut store, (budget, c_puct_bits, 8))
            .expect("puct_init");
        // Warmup one move (populate arena, JIT... well, no JIT, but consistent state).
        let _ = wasmi_puct_search
            .call(&mut store, (cells_ptr, 0, 255, 0))
            .expect("warmup");
        let start = Instant::now();
        for _ in 0..MOVES {
            let _ = wasmi_puct_search
                .call(&mut store, (cells_ptr, 0, 255, 0))
                .expect("search");
        }
        let elapsed = start.elapsed();
        let per_move_ms = elapsed.as_secs_f64() * 1000.0 / MOVES as f64;
        let nodes_after = wasmi_puct_nodes.call(&mut store, ()).expect("nodes");
        println!(
            "Issue 204 — wasmi (interpreted, no JIT) PUCT {label} (budget={budget}): \
             {per_move_ms:.1} ms/move over {MOVES} moves, {nodes_after} nodes evaluated last move"
        );
    }
}

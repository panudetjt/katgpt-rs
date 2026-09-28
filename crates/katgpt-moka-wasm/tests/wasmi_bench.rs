//! Plan 565 Phase 4: run the compiled `.wasm` binary through `wasmi` (pure
//! interpreter, no JIT) and time it. Same artifact as the browser benchmark
//! (`target/wasm32-unknown-unknown/release/katgpt_moka_wasm.wasm`), same
//! raw-pointer FFI idiom this repo's own `src/pruners/bomber/wasm_pruner.rs`
//! already uses with wasmi — isolates "interpreted WASM execution cost"
//! from "JS↔WASM marshalling cost" (the browser number bundles both).
//!
//! Requires the wasm32 artifact to exist: run
//! `cargo build -p katgpt-moka-wasm --target wasm32-unknown-unknown --release`
//! first. Marked `#[ignore]` so it doesn't run in default `cargo test` (it
//! depends on a build artifact from a different target, not just source).

use std::time::Instant;
use wasmi::{Config, Engine, Linker, Memory, Module, Store, TypedFunc};

// The raw `cargo build --target wasm32-unknown-unknown` output still
// carries wasm-bindgen's `__wbindgen_placeholder__::__wbindgen_describe`
// import (inserted because the crate ALSO exports `#[wasm_bindgen]` items
// elsewhere) — wasmi has no JS host to satisfy it. The `wasm-bindgen` CLI's
// post-processing step strips/resolves that, so this test targets ITS
// output, not the raw rustc artifact. Regenerate via:
//   cargo build -p katgpt-moka-wasm --target wasm32-unknown-unknown --release
//   wasm-bindgen target/wasm32-unknown-unknown/release/katgpt_moka_wasm.wasm \
//     --out-dir /tmp/moka-wasm-bench/pkg --target web --no-typescript
const WASM_PATH: &str = "/tmp/moka-wasm-bench/pkg/katgpt_moka_wasm_bg.wasm";

#[test]
#[ignore = "depends on a wasm32 build artifact — see module doc"]
fn wasmi_infer_latency() {
    const ITERS: usize = 50;

    let wasm_bytes = std::fs::read(WASM_PATH)
        .unwrap_or_else(|e| panic!("read {WASM_PATH}: {e} — build the wasm32 target first"));

    let mut config = Config::default();
    config.consume_fuel(false); // benchmarking raw speed, not sandboxing here
    let engine = Engine::new(&config);
    let module = Module::new(&engine, &wasm_bytes[..]).expect("module should parse");

    let mut linker = Linker::new(&engine);
    let mut store = Store::new(&engine, ());

    // The module also exports `#[wasm_bindgen]` items (`WasmGame`/`WasmMoka`)
    // used by the browser benchmark, so wasm-bindgen links in its panic/throw
    // JS-interop plumbing for the WHOLE binary — even though the plain
    // `extern "C"` `wasmi_*` functions under test never call it. Stub every
    // unresolved import as a "never actually called" trap rather than
    // hand-naming wasm-bindgen's hash-suffixed symbol (which changes across
    // wasm-bindgen versions): if the stub genuinely fires, that means a real
    // wasm-bindgen JS-interop path executed during a supposedly-pure-Rust
    // benchmark, which should fail loudly, not silently return zeros.
    for import in module.imports() {
        if linker.get(&store, import.module(), import.name()).is_some() {
            continue;
        }
        if let wasmi::ExternType::Func(func_ty) = import.ty() {
            let name = format!("{}::{}", import.module(), import.name());
            let name_for_trap = name.clone();
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

    let memory: Memory = instance
        .get_memory(&store, "memory")
        .expect("wasm-bindgen always exports memory");
    let wasmi_init: TypedFunc<(), ()> = instance
        .get_typed_func(&store, "wasmi_init")
        .expect("wasmi_init export");
    let wasmi_alloc: TypedFunc<u32, u32> = instance
        .get_typed_func(&store, "wasmi_alloc")
        .expect("wasmi_alloc export");
    let wasmi_infer: TypedFunc<(u32, u32), ()> = instance
        .get_typed_func(&store, "wasmi_infer")
        .expect("wasmi_infer export");

    wasmi_init.call(&mut store, ()).expect("wasmi_init call");

    const INPUT_LEN: u32 = 81 * 12; // moka::INPUT_ELEMENT_COUNT
    const OUTPUT_LEN: u32 = 83; // POLICY_MOVES + value

    let features_ptr = wasmi_alloc
        .call(&mut store, INPUT_LEN)
        .expect("alloc features");
    let out_ptr = wasmi_alloc.call(&mut store, OUTPUT_LEN).expect("alloc out");

    // Write a plausible mid-game-ish feature tensor: a handful of stones set
    // (planes 0/1) plus the constant komi plane (11), matching the general
    // shape of a real position rather than an all-zero input.
    let mut features = vec![0f32; INPUT_LEN as usize];
    for pos in 0..30usize {
        features[pos * 12] = 1.0; // "own stone" plane at a few positions
    }
    for pos in 0..81usize {
        features[pos * 12 + 11] = -7.0 / 15.0; // komi plane, constant
    }
    let feature_bytes: Vec<u8> = features.iter().flat_map(|f| f.to_le_bytes()).collect();
    memory
        .write(&mut store, features_ptr as usize, &feature_bytes)
        .expect("write features");

    // Warmup.
    for _ in 0..10 {
        wasmi_infer
            .call(&mut store, (features_ptr, out_ptr))
            .expect("infer");
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        wasmi_infer
            .call(&mut store, (features_ptr, out_ptr))
            .expect("infer");
    }
    let elapsed = start.elapsed();
    let per_call_us = elapsed.as_micros() as f64 / ITERS as f64;

    println!(
        "Plan 565 Phase 4 — wasmi (interpreted, no JIT) infer latency: {per_call_us:.1} us/call over {ITERS} calls"
    );

    // Sanity: read back the output and confirm it's finite (not asserting
    // exact values — the equivalence oracle for THIS forward pass already
    // lives in katgpt-pruners::moka_net; this test's job is timing).
    let mut out_bytes = vec![0u8; (OUTPUT_LEN as usize) * 4];
    memory
        .read(&store, out_ptr as usize, &mut out_bytes)
        .expect("read output");
    let value = f32::from_le_bytes(out_bytes[(82 * 4)..(83 * 4)].try_into().unwrap());
    assert!(
        value.is_finite(),
        "wasmi output value must be finite, got {value}"
    );
}

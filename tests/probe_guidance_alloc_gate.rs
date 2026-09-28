//! Issue 865 T3 G4 — the probe-guidance hot path is alloc-free.
//!
//! Own test target (its own process) so the process-wide counting allocator
//! observes nothing but this test's work. Measures the
//! [`MlpWeakProbe::probe`] hot path — the per-step, per-block weak-logit
//! production the decode combine extrapolates against: scratch buffers, the
//! latent row and the constant zero embedding are allocated once at
//! construction and REUSED across every call. The affine combine itself
//! (`apply_probe_guidance`) indexes pre-existing context buffers only —
//! structural, reviewed in `d2f_context.rs`.
//!
//! Run: `cargo test --release --features probe_guidance --test
//! probe_guidance_alloc_gate`

#![cfg(feature = "probe_guidance")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use katgpt_forward::d2f_context::{ProbeCtx, WeakLogitProbe};
use katgpt_forward::weak_probe_mlp::MlpWeakProbe;
use katgpt_rs::speculative::probe_artifact::ProbeArtifact;

static ALLOCS: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

const FIXTURE: &[u8] = include_bytes!("fixtures/weak_probe_micro_dllm_v1.bin");

#[test]
fn probe_hot_path_is_alloc_free() {
    let artifact = ProbeArtifact::from_bytes(FIXTURE).expect("fixture verifies");
    let mut probe = MlpWeakProbe::new(artifact).expect("fixture wraps");

    let n = probe.artifact().mlp.n_embd;
    let vocab = probe.artifact().vocab;
    let seq_len = 8usize;

    // Deterministic tap rows (arbitrary content — the probe is a pure
    // function of its input and this gate measures allocation, not output).
    let tap: Vec<f32> = (0..seq_len * n)
        .map(|i| ((i % 17) as f32) * 0.125 - 1.0)
        .collect();
    let mut out = vec![0.0f32; seq_len * vocab];

    // Warm call (any lazy init must land here, not in the measured window).
    // (ProbeCtx constructed inline each call — a closure over borrowed
    // slices cannot name the return lifetime.)
    {
        let input = ProbeCtx {
            xr: &tap,
            x_norm: &tap,
            tap: &tap,
            tap_layers: &[0],
            tap_plane: tap.len(),
            tokens: &[],
            committed_len: 0,
            block_start: 0,
            seq_len,
            vocab,
            n_embd: n,
            step: 0,
        };
        probe.probe(input, &mut out);
    }

    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..1_000 {
        let input = ProbeCtx {
            xr: &tap,
            x_norm: &tap,
            tap: &tap,
            tap_layers: &[0],
            tap_plane: tap.len(),
            tokens: &[],
            committed_len: 0,
            block_start: 0,
            seq_len,
            vocab,
            n_embd: n,
            step: 0,
        };
        probe.probe(input, &mut out);
    }
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        before,
        after,
        "probe hot path allocated {} times across 1_000 calls — the seam contract \
         is scratch-reuse, zero allocation per call",
        after - before
    );
    println!("G4: 1_000 probe calls, {} allocations", after - before);
}

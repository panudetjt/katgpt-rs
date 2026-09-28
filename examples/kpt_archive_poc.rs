//! `.kpt` archive POC (Issue 841 / Research 568 fusion 2, owner call
//! 2026-09-19): .cact-style nameless positional layer-major single file ×
//! BLAKE3/Merkle integrity × atomic weight hot-swap via file replacement.
//!
//! Run: `cargo run --release --features kpt_archive --example kpt_archive_poc`
//!
//! Three proofs, printed:
//! 1. **POC** — build 8 layers at the served shape (768×3072, Research 569's
//!    reference geometry), verify, round-trip weights bit-identically,
//!    zero-copy views match the copying path.
//! 2. **PERF** — build / verify (hash throughput) / atomic hot-swap latency
//!    (write+fsync+rename+reload+re-verify). The swap is the number the
//!    deferred format+engine lift eventually has to defend.
//! 3. **SEC** — single-bit tamper refused; tampered swap leaves the on-disk
//!    file untouched; old-version replay detected by `archive_id`.

use std::time::Instant;

use katgpt_core::kpt_archive::{
    KptArchive, OwnedKptArchive, build_archive, swap_in_place, write_atomic,
};
use katgpt_rs::types::TernaryWeights;

const LAYERS: usize = 8;
const ROWS: usize = 768;
const COLS: usize = 3072; // d=768 with the 4× FFN width — the served shape

fn pseudo(i: usize) -> f32 {
    // Deterministic LCG-ish mix (the bench-fixture house shape — no RNG dep).
    let h = (i as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((h >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn build_layers(seed: u64) -> Vec<TernaryWeights> {
    (0..LAYERS)
        .map(|l| {
            let raw: Vec<f32> = (0..ROWS * COLS)
                .map(|i| pseudo(i + seed as usize + l * 97))
                .collect();
            TernaryWeights::quantize_from_f32(&raw, ROWS, COLS)
        })
        .collect()
}

fn main() {
    println!("══ .kpt archive POC — {LAYERS} layers of {ROWS}×{COLS} ══");

    // ── 1. POC: build → verify → round-trip ─────────────────────────────
    let layers = build_layers(1);
    let refs: Vec<&TernaryWeights> = layers.iter().collect();

    let t = Instant::now();
    let bytes = build_archive(&refs).expect("build");
    let build_ms = t.elapsed().as_secs_f64() * 1e3;

    let archive = KptArchive::from_bytes(&bytes).expect("verify");
    println!(
        "archive: {} bytes ({:.2} MiB), {} layers, merkle {:.8}…, id {:.8}…",
        bytes.len(),
        bytes.len() as f64 / (1024.0 * 1024.0),
        archive.layers().len(),
        hex8(&archive.merkle_root()),
        hex8(&archive.archive_id())
    );

    for (i, want) in layers.iter().enumerate() {
        let got = archive.to_ternary_weights(i);
        assert_eq!(got.pos_bits, want.pos_bits, "layer {i} pos_bits round-trip");
        assert_eq!(got.neg_bits, want.neg_bits, "layer {i} neg_bits round-trip");
        assert_eq!(
            got.row_scale, want.row_scale,
            "layer {i} row_scale round-trip"
        );
    }
    if let Ok(view) = archive.layer_view(0) {
        let copied = archive.to_ternary_weights(0);
        assert_eq!(
            view.pos_bits,
            copied.pos_bits.as_slice(),
            "zero-copy pos == copied"
        );
        assert_eq!(
            view.neg_bits,
            copied.neg_bits.as_slice(),
            "zero-copy neg == copied"
        );
        assert_eq!(
            view.row_scale,
            copied.row_scale.as_slice(),
            "zero-copy scale == copied"
        );
        println!("zero-copy layer view: OK (aligned buffer, views match the copying path)");
    } else {
        println!("zero-copy layer view: buffer misaligned (copying path is the contract)");
    }
    let again = build_archive(&refs).expect("rebuild");
    assert_eq!(bytes, again, "build is byte-deterministic");
    println!("round-trip: all {LAYERS} layers bit-identical; build byte-deterministic");

    // ── 2. PERF: verify throughput + atomic hot-swap latency ────────────
    let mib = bytes.len() as f64 / (1024.0 * 1024.0);
    let mut best_verify_us = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        KptArchive::from_bytes(&bytes).expect("verify");
        best_verify_us = best_verify_us.min(t.elapsed().as_secs_f64() * 1e6);
    }
    let verify_mib_s = mib / (best_verify_us / 1e6);
    println!(
        "build: {build_ms:.1} ms | verify: {best_verify_us:.0} µs ({verify_mib_s:.0} MiB/s, best of 5) — full fail-closed ladder"
    );

    // Shared-temp-path rule: pid-suffixed file names, same-dir temp rename.
    let path = std::env::temp_dir().join(format!("kpt_archive_poc_{}.kpt", std::process::id()));
    let t = Instant::now();
    write_atomic(&path, &bytes).expect("atomic write");
    let first_write_ms = t.elapsed().as_secs_f64() * 1e3;

    // Hot-swap to a DIFFERENT model (different weights → different id), then
    // measure a swap back — write + fsync + rename + reload + re-verify.
    let layers_v2 = build_layers(2);
    let refs_v2: Vec<&TernaryWeights> = layers_v2.iter().collect();
    let id_v2 = swap_in_place(&path, &refs_v2).expect("swap to v2");
    let t = Instant::now();
    let id_back = swap_in_place(&path, &refs).expect("swap back to v1");
    let swap_ms = t.elapsed().as_secs_f64() * 1e3;

    println!(
        "first write: {first_write_ms:.1} ms | hot-swap (write+fsync+rename+reload+verify of {mib:.2} MiB): {swap_ms:.1} ms"
    );

    // ── 3. SEC: tamper + replay posture (the full arms live in the test) ─
    // (a) single-bit flip in a fresh copy → refused before materialization.
    let mut tampered = bytes.clone();
    let mid = tampered.len() / 2;
    tampered[mid] ^= 0x01;
    assert!(
        KptArchive::from_bytes(&tampered).is_err(),
        "bit flip must be refused"
    );

    // (b) an attacker-controlled file drop still verifies as REFUSED, and
    //     the LIVE file is unaffected (swap_in_place only ever places bytes
    //     it built and verified itself — verify-before-rename by construction).
    let drop_path =
        std::env::temp_dir().join(format!("kpt_archive_poc_drop_{}.kpt", std::process::id()));
    let mut poison = build_archive(&refs_v2).expect("build poison");
    let pl = poison.len() - 64 - 8; // inside the last layer's payload
    poison[pl] ^= 0x80;
    write_atomic(&drop_path, &poison).expect("drop poison file");
    assert!(
        OwnedKptArchive::load(&drop_path).unwrap().verify().is_err(),
        "poisoned drop must refuse"
    );
    let owned_live = OwnedKptArchive::load(&path).expect("load live");
    let live = owned_live.verify().expect("live file still clean");
    assert_eq!(
        live.archive_id(),
        id_back,
        "live file untouched by the poisoned drop"
    );
    let _ = std::fs::remove_file(&drop_path);

    // (c) replay: the OLD archive still verifies structurally (it is a valid
    //     file) but its id differs from what is live — the consumer-side
    //     replay signal `swap_in_place` returns exactly to compare.
    assert_ne!(id_back, id_v2, "distinct models must have distinct ids");
    println!(
        "sec: bit-flip refused; poisoned file drop refused with the live archive untouched; replay of v2 id {:.8}… detected vs live {:.8}…",
        hex8(&id_v2),
        hex8(&id_back)
    );

    let _ = std::fs::remove_file(&path);
    println!("══ .kpt POC complete — format verifies, swaps atomically, refuses injection ══");
}

fn hex8(b: &[u8; 32]) -> String {
    b.iter().take(8).map(|x| format!("{x:02x}")).collect()
}

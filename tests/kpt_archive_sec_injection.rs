//! `.kpt` archive — security bad-injection arms (Issue 841 / Research 568
//! fusion 2, owner call 2026-09-19). Every arm feeds the reader HOSTILE bytes
//! and requires a NAMED refusal (never a panic, never a silent accept).
//!
//! The ladder under test (`katgpt_core::kpt_archive::KptArchive::from_bytes`):
//! structure (magic/version/length/offsets/overflow/sequencing) → per-layer
//! BLAKE3 → Merkle root → archive id. Plus the atomicity/replay properties
//! of the hot-swap path.

use katgpt_core::kpt_archive::{
    KPT_HEADER_SIZE, KPT_TRAILER_SIZE, KptArchive, KptError, OwnedKptArchive, build_archive,
    swap_in_place,
};
use katgpt_rs::types::TernaryWeights;

fn layers(n: usize, rows: usize, cols: usize, seed: u64) -> Vec<TernaryWeights> {
    (0..n)
        .map(|l| {
            // Seed mixes into the index BEFORE the multiply: adding it after
            // would be shifted out by `>> 33` (seeds 1 vs 2 produced IDENTICAL
            // weights until this was caught by the replay-detector test — a
            // fixture bug, not a format bug).
            let raw: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let idx = i + seed as usize * 0x9E37_79B9 + l * 7919;
                    let h = (idx as u64).wrapping_mul(6364136223846793005);
                    ((h >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();
            TernaryWeights::quantize_from_f32(&raw, rows, cols)
        })
        .collect()
}

fn good() -> Vec<u8> {
    let ls = layers(3, 16, 64, 42);
    let refs: Vec<&TernaryWeights> = ls.iter().collect();
    build_archive(&refs).unwrap()
}

#[test]
fn magic_corruption_rejected() {
    let mut b = good();
    b[0] = b'X';
    match KptArchive::from_bytes(&b) {
        Err(KptError::BadMagic(_)) => {}
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn version_bump_rejected() {
    let mut b = good();
    b[4] = 0x02; // version LSB → 2
    match KptArchive::from_bytes(&b) {
        Err(KptError::UnsupportedVersion(2)) => {}
        other => panic!("expected UnsupportedVersion(2), got {other:?}"),
    }
}

#[test]
fn empty_and_tiny_files_rejected() {
    assert!(matches!(
        KptArchive::from_bytes(&[]),
        Err(KptError::Truncated { .. })
    ));
    let mut b = good();
    b.truncate(8);
    assert!(matches!(
        KptArchive::from_bytes(&b),
        Err(KptError::Truncated { .. })
    ));
}

#[test]
fn truncation_by_one_byte_rejected() {
    let mut b = good();
    b.truncate(b.len() - 1);
    // The dangling-bytes / region-end check must catch any length change.
    assert!(
        KptArchive::from_bytes(&b).is_err(),
        "a truncated archive must never verify"
    );
}

#[test]
fn payload_single_bit_flip_rejected_at_hash_layer() {
    let mut b = good();
    // Flip a bit inside the FIRST layer's payload (just past the directory).
    let dir_end = KPT_HEADER_SIZE as usize + 3 * 64;
    b[dir_end + 8] ^= 0x40;
    match KptArchive::from_bytes(&b) {
        Err(KptError::LayerHashMismatch { layer: 0, .. }) => {}
        other => panic!("expected LayerHashMismatch at layer 0, got {other:?}"),
    }
}

#[test]
fn directory_hash_swap_rejected() {
    // Copy layer 0's DECLARED hash over layer 1's: the recomputed payload
    // hash for layer 1 no longer matches its directory entry.
    let mut b = good();
    let e0 = KPT_HEADER_SIZE as usize;
    let e1 = e0 + 64;
    let h0: [u8; 32] = b[e0 + 32..e0 + 64].try_into().unwrap();
    b[e1 + 32..e1 + 64].copy_from_slice(&h0);
    match KptArchive::from_bytes(&b) {
        Err(KptError::LayerHashMismatch { layer: 1, .. }) => {}
        other => panic!("expected LayerHashMismatch at layer 1, got {other:?}"),
    }
}

#[test]
fn merkle_root_tamper_rejected() {
    let mut b = good();
    let n = b.len();
    b[n - 64] ^= 0x01; // first byte of the merkle root
    match KptArchive::from_bytes(&b) {
        Err(KptError::MerkleRootMismatch { .. }) => {}
        other => panic!("expected MerkleRootMismatch, got {other:?}"),
    }
}

#[test]
fn archive_id_tamper_rejected() {
    let mut b = good();
    let n = b.len();
    b[n - 32] ^= 0x01; // first byte of the archive id
    match KptArchive::from_bytes(&b) {
        Err(KptError::ArchiveIdMismatch { .. }) => {}
        other => panic!("expected ArchiveIdMismatch, got {other:?}"),
    }
}

#[test]
fn offset_beyond_eof_rejected_without_panic() {
    let mut b = good();
    let e1 = KPT_HEADER_SIZE as usize + 64;
    // Declare layer 1's payload at u64::MAX-ish offset (LE bytes).
    b[e1 + 16..e1 + 24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(
        KptArchive::from_bytes(&b).is_err(),
        "OOB offset must be refused, not panic"
    );
}

#[test]
fn offset_len_wrap_attack_rejected() {
    let mut b = good();
    let e1 = KPT_HEADER_SIZE as usize + 64;
    // offset = u64::MAX - 8, len = 16 → offset+len wraps past u64::MAX.
    b[e1 + 16..e1 + 24].copy_from_slice(&(u64::MAX - 8).to_le_bytes());
    b[e1 + 24..e1 + 32].copy_from_slice(&16u64.to_le_bytes());
    match KptArchive::from_bytes(&b) {
        Err(KptError::OffsetOverflow { layer: 1 })
        | Err(KptError::NotSequential { layer: 1, .. }) => {}
        other => panic!("expected OffsetOverflow/NotSequential at layer 1, got {other:?}"),
    }
}

#[test]
fn layer_count_inflation_rejected() {
    let mut b = good();
    // Claim 4 layers while the directory + payloads only cover 3.
    b[8..12].copy_from_slice(&4u32.to_le_bytes());
    assert!(
        KptArchive::from_bytes(&b).is_err(),
        "layer-count inflation must be refused"
    );
}

#[test]
fn non_sequential_layout_rejected() {
    let mut b = good();
    let e1 = KPT_HEADER_SIZE as usize + 64;
    // Point layer 1 at layer 0's offset (overlap) — the nameless-positional
    // discipline requires the EXACT sequential position.
    let e0 = KPT_HEADER_SIZE as usize;
    let off0 = u64::from_le_bytes(b[e0 + 16..e0 + 24].try_into().unwrap());
    b[e1 + 16..e1 + 24].copy_from_slice(&off0.to_le_bytes());
    match KptArchive::from_bytes(&b) {
        Err(KptError::NotSequential { layer: 1, .. }) => {}
        other => panic!("expected NotSequential at layer 1, got {other:?}"),
    }
}

#[test]
fn blocks64_dimension_mismatch_rejected() {
    let mut b = good();
    let e0 = KPT_HEADER_SIZE as usize;
    // Claim a blocks64 that contradicts cols=64 (the real value is 1).
    b[e0 + 8..e0 + 12].copy_from_slice(&2u32.to_le_bytes());
    assert!(
        KptArchive::from_bytes(&b).is_err(),
        "inconsistent dims must be refused"
    );
}

#[test]
fn valid_but_stale_archive_is_the_replay_signal() {
    // Two DIFFERENT models: both verify (both are well-formed), different ids.
    let a = {
        let ls = layers(2, 8, 64, 1);
        let refs: Vec<&TernaryWeights> = ls.iter().collect();
        build_archive(&refs).unwrap()
    };
    let b = {
        let ls = layers(2, 8, 64, 2);
        let refs: Vec<&TernaryWeights> = ls.iter().collect();
        build_archive(&refs).unwrap()
    };
    let va = KptArchive::from_bytes(&a).unwrap();
    let vb = KptArchive::from_bytes(&b).unwrap();
    assert_ne!(
        va.archive_id(),
        vb.archive_id(),
        "replay detector: distinct models, distinct ids"
    );
    // Structural validity alone CANNOT tell old from new — that is the
    // documented division of labour, and why swap_in_place RETURNS the id.
}

#[test]
fn refused_swap_leaves_disk_file_untouched() {
    // The public swap builds + verifies BEFORE any disk write, so a build
    // failure (here: an inconsistent layer spec) must leave the file as-is.
    let path = std::env::temp_dir().join(format!("kpt_sec_refused_{}.kpt", std::process::id()));
    let ls = layers(2, 8, 64, 3);
    let refs: Vec<&TernaryWeights> = ls.iter().collect();
    let id0 = swap_in_place(&path, &refs).expect("initial swap");

    let mut broken = TernaryWeights::new(4, 64);
    broken.blocks64 = 9; // contradicts cols → BadLayerSpec
    let bad_refs: Vec<&TernaryWeights> = vec![&broken];
    assert!(matches!(
        swap_in_place(&path, &bad_refs),
        Err(KptError::BadLayerSpec { .. })
    ));

    let owned = OwnedKptArchive::load(&path).unwrap();
    let live = owned.verify().unwrap();
    assert_eq!(
        live.archive_id(),
        id0,
        "refused swap must not touch the live file"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn atomic_swap_replaces_whole_file_or_nothing() {
    let path = std::env::temp_dir().join(format!("kpt_sec_atomic_{}.kpt", std::process::id()));
    let ls1 = layers(2, 8, 64, 10);
    let r1: Vec<&TernaryWeights> = ls1.iter().collect();
    let id1 = swap_in_place(&path, &r1).unwrap();
    let ls2 = layers(2, 8, 64, 11);
    let r2: Vec<&TernaryWeights> = ls2.iter().collect();
    let id2 = swap_in_place(&path, &r2).unwrap();
    assert_ne!(id1, id2);
    // After the swap the file is EXACTLY the new archive — a reader that
    // loaded the old id can detect the change (the hot-swap contract).
    let owned = OwnedKptArchive::load(&path).unwrap();
    let live = owned.verify().unwrap();
    assert_eq!(live.archive_id(), id2);
    assert_eq!(live.layers().len(), 2);
    assert!(live.to_ternary_weights(0).pos_bits == ls2[0].pos_bits);
    // No temp litter beside the archive.
    let dir = path.parent().unwrap();
    let litter = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with(".kpt_sec_atomic_") && n.ends_with(".tmp")
        });
    assert_eq!(
        litter.count(),
        0,
        "atomic swap must not leave temp files behind"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn empty_archive_is_the_documented_degenerate_case() {
    // layer_count=0 is the minimum file: header + trailer, nothing else. It
    // VERIFIES (documented) and is distinguishable from any real archive by
    // layers().len() == 0 — decided explicitly, not left to chance.
    let empty = build_archive(&[]).unwrap();
    assert_eq!(empty.len() as u64, KPT_HEADER_SIZE + KPT_TRAILER_SIZE);
    let v = KptArchive::from_bytes(&empty).unwrap();
    assert_eq!(v.layers().len(), 0);
    // And it is NOT confusable with a one-layer archive.
    let one = {
        let ls = layers(1, 8, 64, 7);
        let refs: Vec<&TernaryWeights> = ls.iter().collect();
        build_archive(&refs).unwrap()
    };
    assert!(KptArchive::from_bytes(&one).unwrap().layers().len() == 1);
    assert_ne!(
        v.archive_id(),
        KptArchive::from_bytes(&one).unwrap().archive_id()
    );
}

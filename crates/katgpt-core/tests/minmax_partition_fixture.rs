//! Plan 616 T3 — cross-repo determinism pin: the gemma2-2b S-matrix
//! (26×26, 325 upper-triangle entries) measured 2026-09-29 in riir-infer
//! (`twt_gemma2_profile`, corpus BLAKE3 `44aabe8e0e487b25c4ae1dc1a5aa33c0815dab0508e487ebbebcf04b978f2a52`,
//! artifact `.raw/twt/gemma2_profile.json`, gitignored there) becomes a
//! COMMITTED fixture here, and the DP's m table over those exact bytes is
//! the known answer.
//!
//! The pin is on BYTES, not on capture provenance: the artifact was
//! measured on one box; this test asserts that the promoted DP is
//! deterministic over fixed inputs (the cross-repo contract — riir-infer
//! re-exports THIS implementation and re-derives the same partitions from
//! the same artifact bytes). A partition drift here is a promotion
//! regression, loud at the fixture.
//!
//! Fixture layout: magic `TPUF` (u32 LE) + u32 LE n + n(n-1)/2 f32 LE in
//! ascending (i, j) order.

#![cfg(feature = "minmax_partition")]

use katgpt_core::partition::{
    SMatrix, forced_min_blocks, minmax_partition, minmax_partition_typed, partition_worst,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/twt_gemma2_s_upper.bin");

/// The committed bytes' BLAKE3 — a silent fixture edit reds here.
const FIXTURE_BLAKE3: &str = "ecdc0d126a89fc6e70a7e57a6475e086c7319534eaf439689128183d758b2c33";

/// The artifact's own ε grid (the profile's sweep), ascending.
const EPS_GRID: [f32; 7] = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2];

fn load_fixture() -> SMatrix {
    let (magic, rest) = FIXTURE.split_at(4);
    assert_eq!(magic, b"TPUF", "fixture magic");
    let (n_bytes, data_bytes) = rest.split_at(4);
    let n = u32::from_le_bytes(n_bytes.try_into().unwrap()) as usize;
    let count = n * (n - 1) / 2;
    assert_eq!(data_bytes.len(), count * 4, "fixture arity");

    let mut data = vec![0.0f32; n * n];
    let mut cursor = data_bytes;
    for i in 0..n {
        for j in (i + 1)..n {
            let (bytes, rest) = cursor.split_at(4);
            cursor = rest;
            let v = f32::from_le_bytes(bytes.try_into().unwrap());
            data[i * n + j] = v;
            data[j * n + i] = v;
        }
    }
    SMatrix::from_parts(n, data)
}

fn blake3_hex(bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(bytes);
    hasher.finalize().to_hex().to_string()
}

#[test]
fn fixture_bytes_are_pinned() {
    assert_eq!(blake3_hex(FIXTURE), FIXTURE_BLAKE3, "fixture bytes drifted");
}

#[test]
fn fixture_symmetry_diagonal_and_finiteness() {
    let s = load_fixture();
    let n = s.n();
    assert_eq!(n, 26);
    assert_eq!(s.get(0, 0), 0.0);
    for i in 0..n {
        for j in 0..n {
            let v = s.get(i, j);
            assert!(v.is_finite());
            assert!((v - s.get(j, i)).abs() == 0.0, "symmetry at ({i},{j})");
        }
    }
}

/// The m table over the committed artifact bytes — derived at the
/// promotion landing (2026-10-02, this box) and pinned; any DP change
/// that moves a boundary reds here. Cross-check: the same S through the
/// TYPED DP at an all-true type vector must read the SAME counts (the
/// unconstrained problem is the typed problem's feasibility superset, and
/// with one type every block is feasible — the typed DP is the same DP).
#[test]
fn fixture_m_table_known_answer() {
    let s = load_fixture();
    let m_table: Vec<usize> = EPS_GRID
        .iter()
        .map(|&eps| minmax_partition(&s, eps).unwrap().len())
        .collect();
    assert_eq!(
        m_table, PINNED_M_TABLE,
        "m table over the committed artifact drifted — a partition regression"
    );

    // m monotone (the Phase-2 invariant, on real measured data).
    let mut prev = usize::MAX;
    for &m in &m_table {
        assert!(m <= prev, "m grew: {m} > {prev}");
        prev = m;
    }

    // Constraint post-condition on real data: every emitted block within ε.
    for &eps in &EPS_GRID {
        let p = minmax_partition(&s, eps).unwrap();
        assert!(partition_worst(&s, &p) <= eps, "worst > eps at {eps}");
    }

    // The all-one-type cross-check: the typed DP must reproduce the
    // unconstrained m table exactly (same feasible set).
    let all_true = vec![true; s.n()];
    for &eps in &EPS_GRID {
        let typed = minmax_partition_typed(&s, eps, &all_true).unwrap().len();
        let free = minmax_partition(&s, eps).unwrap().len();
        assert_eq!(typed, free, "all-true typed m != unconstrained m at {eps}");
    }

    // gemma-2 carries ONE RoPE theta for both mask types (no hard
    // type-split), so a single-type layout is the honest forced floor.
    assert_eq!(forced_min_blocks(&all_true), 1);
}

/// Derived at the promotion landing (2026-10-02, this box) — see the test
/// above for the grid and the cross-checks. DO NOT hand-edit: re-derive
/// from the pinned bytes.
const PINNED_M_TABLE: [usize; 7] = [26, 17, 10, 6, 4, 3, 1];

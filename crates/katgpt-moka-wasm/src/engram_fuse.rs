//! Engram-fused PUCT memory — Issue 868 / Plan 605 / Proposal 013.
//!
//! Offline-mined value/prior memory read at the PUCT node-expansion seam.
//! Substrate (consumed, never re-implemented): `katgpt-core/src/engram/`
//! (Plan 299 GOAT — `multi_head_hash`, `EngramTableBuilder`,
//! `InMemoryEngramTable`, `build_merkle_root`). The N-gram machinery is the
//! hash family only — the key is the TT discipline
//! `(board, ko_point, to_play)` packed into 4 `u64` words, NEVER a
//! move-n-gram key (GHI hazard).
//!
//! # Row layout
//!
//! Each slot row is `[v̄, n, b]` (`ROW_DIM = 3`):
//! - `v̄` — mean terminal outcome from the position's to_play perspective,
//!   clamped to [-1, 1] (same tanh range the value head uses).
//! - `n` — visit count across mining games (the evidence quantity).
//! - `b` — visit-distribution concentration ∈ [0, 1]
//!   (`(H_max − H)/H_max` over the mining root's child visits).
//!
//! # Fusion rule (applied by `puct::expand`, one read per expansion)
//!
//! - Evidence gate `σ((n − N_MIN)/TAU_N)` — count-based: a slot seen once
//!   is a rumor, 500× is a statistic (Issue 868 caveat 2).
//! - Q-init (OUR delta, not M-MCTS's visit-scheduled blend): the expanded
//!   node starts as ONE damped pseudo-visit — `visits = 1`,
//!   `total_value = gate · v̄` — and real search values then accumulate
//!   on top with weight 1. One-shot, no schedule.
//! - Prior sharpening: the child-softmax logits are scaled by
//!   `γ = exp(2β(b − 0.5) · gate)` with `β = ln 2`, so `γ ∈ [0.5, 2]` by
//!   construction — the build-time bound Issue 868 demands (an unbounded
//!   `exp(gate·β·b)` could collapse an 81-move prior to a delta).
//!   γ > 0 preserves the top-k sort order, so only the softmax is touched.
//!
//! `hits == 0` (empty table, unseen slot, or zero-slot table) hard-skips
//! everything — G1's cheapest correctness assertion.
//!
//! # Native gate
//!
//! Browser/wasm is OUT of scope (Proposal 013): every consumer of this
//! module is behind
//! `#[cfg(all(feature = "engram_puct", not(target_arch = "wasm32")))]`.

use crate::board::{Board, Cell};
use katgpt_core::engram::{
    CanonicalId, EngramHash, EngramTable, EngramTableBuilder, HashHead, InMemoryEngramTable, K_MAX,
    multi_head_hash,
};

/// Slot row width: `[v̄, n, b]`.
pub const ROW_DIM: usize = 3;
/// Evidence-gate midpoint: `n < N_MIN` is sub-0.5 evidence.
pub const N_MIN: f32 = 8.0;
/// Evidence-gate temperature: n=1 → 0.18, n=8 → 0.5, n≥24 → ≥0.92.
pub const TAU_N: f32 = 4.0;
/// Sharpening slope — `ln 2`, so `γ ∈ [0.5, 2]` for `b ∈ [0, 1]`,
/// `gate ∈ [0, 1]`.
pub const BETA: f32 = std::f32::consts::LN_2;
/// Sharpening bounds (redundant with the β choice; asserted, not trusted).
pub const GAMMA_MIN: f32 = 0.5;
pub const GAMMA_MAX: f32 = 2.0;
/// File magic for the mined-stats artifact.
pub const TABLE_MAGIC: [u8; 8] = *b"KEPT868\0";
/// Serialized bytes per entry (4×u64 key + 3×f32 stats). Deliberately NOT
/// `size_of::<MinedEntry>()` — the struct's alignment padding (48) is not
/// written to disk (44).
pub const ENTRY_SERIALIZED_BYTES: usize = 4 * 8 + 3 * 4;

/// One mined position: exact TT key + accumulated self-play statistics.
///
/// Accumulation happens per EXACT key (no collision at this layer);
/// collisions happen only at engram-slot level, quantified by the T1.3
/// collision audit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinedEntry {
    /// `(board, ko_point, to_play)` packed words — see [`pack_tt_key`].
    pub key: [u64; 4],
    /// Σ of terminal outcomes from this position's to_play perspective.
    pub sum_outcome: f32,
    /// Visit count across mining games.
    pub count: f32,
    /// Σ of visit-distribution concentrations (mean with `count` = `b`).
    pub sum_conc: f32,
}

/// Frozen mined table: entries + the heads config + the BLAKE3 commitment
/// of the table built at `n_slots` (G6).
#[derive(Debug, Clone)]
pub struct MinedTable {
    /// Slot count of the canonical build (the one `root` commits).
    pub n_slots: usize,
    /// Hash-head configuration (frozen once at miner time; every
    /// [`MinedTable::build_table`] — any size — reuses it).
    pub heads: [HashHead; K_MAX],
    /// Entries, sorted by key (determinism: build order is key order).
    pub entries: Vec<MinedEntry>,
    /// BLAKE3 root of the `n_slots` build.
    pub root: [u8; 32],
}

/// Errors from [`MinedTable::load`].
#[derive(Debug)]
pub enum EngramFuseError {
    Io(std::io::Error),
    BadMagic,
    /// Dimensionality or entry-count field out of range.
    BadShape,
    /// Entries not strictly key-sorted (determinism pin).
    UnsortedEntries,
    /// Truncated or trailing bytes.
    BadLength {
        expected: usize,
        got: usize,
    },
    /// BLAKE3 root of the rebuilt table ≠ the stored root.
    CommitmentMismatch {
        expected: [u8; 32],
        got: [u8; 32],
    },
}

impl std::fmt::Display for EngramFuseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::BadMagic => write!(f, "bad magic (not a KEPT868 mined table)"),
            Self::BadShape => write!(f, "bad shape field (d != ROW_DIM or absurd counts)"),
            Self::UnsortedEntries => write!(f, "entries not strictly key-sorted"),
            Self::BadLength { expected, got } => {
                write!(f, "bad length: expected {expected} bytes, got {got}")
            }
            Self::CommitmentMismatch { expected, got } => write!(
                f,
                "commitment mismatch: expected {}, got {}",
                hex16(expected),
                hex16(got)
            ),
        }
    }
}

impl std::error::Error for EngramFuseError {}

fn hex16(b: &[u8; 32]) -> String {
    b.iter().take(8).map(|x| format!("{x:02x}")).collect()
}

/// Pack the TT key `(board, ko_point, to_play)` into 4 `u64` words.
///
/// Cells: 2 bits each (Empty=0, Black=1, White=2), 32 cells per word over
/// words 0..2 (81 cells → 162 bits). Word 3: ko point (0xFF sentinel for
/// `None`; board indices are < 81) | to_play in bits 8..10.
///
/// `consecutive_passes` is deliberately NOT in the key (Issue 868 caveat 3:
/// the simple-ko TT key is already a superko approximation; widening it
/// silently is forbidden).
#[inline]
pub fn pack_tt_key(board: &Board, out: &mut [CanonicalId; 4]) {
    let mut w = [0u64; 3];
    for (i, cell) in board.cells.iter().enumerate() {
        let v: u64 = match cell {
            Cell::Empty => 0,
            Cell::Black => 1,
            Cell::White => 2,
        };
        w[i / 32] |= v << ((i % 32) * 2);
    }
    let ko = board.ko_point.map_or(0xFF, |k| k as u64);
    let tp: u64 = match board.to_play {
        Cell::Black => 1,
        Cell::White => 2,
        Cell::Empty => 0,
    };
    out[0] = CanonicalId(w[0]);
    out[1] = CanonicalId(w[1]);
    out[2] = CanonicalId(w[2]);
    out[3] = CanonicalId(ko | (tp << 8));
}

/// Copy of the packed key words (for accumulation maps + `MinedEntry`).
#[inline]
pub fn tt_key_words(board: &Board) -> [u64; 4] {
    let mut ids = [CanonicalId(0); 4];
    pack_tt_key(board, &mut ids);
    [ids[0].0, ids[1].0, ids[2].0, ids[3].0]
}

/// Visit-distribution concentration `(H_max − H)/H_max ∈ [0, 1]` over the
/// mining root's child visit counts. 1 child → 1.0 (fully concentrated);
/// uniform visits → 0.0.
pub fn visit_concentration(counts: &[u32]) -> f32 {
    let k = counts.len();
    if k == 0 {
        return 0.0;
    }
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    if total == 0 || k == 1 {
        return 1.0;
    }
    let h_max = (k as f64).ln();
    let mut h = 0.0f64;
    for &c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total as f64;
        h -= p * p.ln();
    }
    ((h_max - h) / h_max).clamp(0.0, 1.0) as f32
}

/// Evidence gate `σ((n − N_MIN)/TAU_N)`.
#[inline]
pub fn evidence_gate(n: f32) -> f32 {
    katgpt_core::sigmoid((n - N_MIN) / TAU_N)
}

/// Sharpening factor `γ = exp(2β(b − 0.5) · gate)`, clamped to
/// `[GAMMA_MIN, GAMMA_MAX]` (the clamp is redundant given the β choice and
/// the build-time clamps on `b`/`gate`; asserted, not trusted).
#[inline]
pub fn sharpen_gamma(b: f32, gate: f32) -> f32 {
    let g = (2.0 * BETA * (b.clamp(0.0, 1.0) - 0.5) * gate.clamp(0.0, 1.0)).exp();
    debug_assert!((GAMMA_MIN..=GAMMA_MAX).contains(&g));
    g.clamp(GAMMA_MIN, GAMMA_MAX)
}

/// Slot row for a mined entry: `[v̄, n, b]` with the build-time clamps.
#[inline]
pub fn row_of(entry: &MinedEntry) -> [f32; ROW_DIM] {
    let v = if entry.count > 0.0 {
        (entry.sum_outcome / entry.count).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let n = entry.count.max(0.0);
    let b = entry.sum_conc.clamp(0.0, 1.0);
    // Guard the mean-over-hits zero-row skip in `EngramPuctMemory::read`:
    // a stored row must never be all-zero, or a hit would read as a miss.
    debug_assert!(n > 0.0 || v != 0.0 || b != 0.0);
    [v, n, b]
}

/// The K_MAX slot keys for a packed TT key under `heads`.
#[inline]
pub fn keys_for(key: &[u64; 4], heads: &[HashHead; K_MAX]) -> [EngramHash; K_MAX] {
    let ids = [
        CanonicalId(key[0]),
        CanonicalId(key[1]),
        CanonicalId(key[2]),
        CanonicalId(key[3]),
    ];
    multi_head_hash(&ids, heads)
}

impl MinedTable {
    /// Build from entries: sorts by key (determinism), builds the `n_slots`
    /// table writing each row under all K_MAX heads, records the BLAKE3 root.
    pub fn build(n_slots: usize, heads: [HashHead; K_MAX], mut entries: Vec<MinedEntry>) -> Self {
        entries.sort_by_key(|a| a.key);
        let table = build_table_from_entries(n_slots, &heads, &entries);
        let root = table.commitment();
        Self {
            n_slots,
            heads,
            entries,
            root,
        }
    }

    /// Rebuild the engram table at `n_slots` from this entry set (any size —
    /// the T1.3 hit-rate curve builds at {2¹⁶, 2¹⁸, 2²⁰} this way).
    pub fn build_table(&self, n_slots: usize) -> InMemoryEngramTable {
        build_table_from_entries(n_slots, &self.heads, &self.entries)
    }

    /// The canonical (`n_slots`) build, as the fused player consumes it.
    pub fn canonical_table(&self) -> InMemoryEngramTable {
        self.build_table(self.n_slots)
    }

    /// Serialize (all little-endian):
    /// magic | n_slots | d | heads (16 × (modulus, seed)) | n_entries |
    /// entries (key-sorted) | root.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let entry_bytes = ENTRY_SERIALIZED_BYTES;
        let mut buf =
            Vec::with_capacity(8 + 16 + 16 * 16 + 8 + self.entries.len() * entry_bytes + 32);
        buf.extend_from_slice(&TABLE_MAGIC);
        buf.extend_from_slice(&(self.n_slots as u64).to_le_bytes());
        buf.extend_from_slice(&(ROW_DIM as u64).to_le_bytes());
        for h in &self.heads {
            buf.extend_from_slice(&h.modulus.to_le_bytes());
            buf.extend_from_slice(&h.seed.to_le_bytes());
        }
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for e in &self.entries {
            for w in &e.key {
                buf.extend_from_slice(&w.to_le_bytes());
            }
            buf.extend_from_slice(&e.sum_outcome.to_le_bytes());
            buf.extend_from_slice(&e.count.to_le_bytes());
            buf.extend_from_slice(&e.sum_conc.to_le_bytes());
        }
        buf.extend_from_slice(&self.root);
        let mut f = std::fs::File::create(path)?;
        f.write_all(&buf)
    }

    /// Load + verify: magic, shape, strict key-sortedness, and the BLAKE3
    /// root of the rebuilt `n_slots` table.
    pub fn load(path: &std::path::Path) -> Result<Self, EngramFuseError> {
        let buf = std::fs::read(path).map_err(EngramFuseError::Io)?;
        let mut cur = 0usize;
        if take(&buf, &mut cur, 8)? != TABLE_MAGIC.as_slice() {
            return Err(EngramFuseError::BadMagic);
        }
        let n_slots = u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap()) as usize;
        let d = u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap()) as usize;
        if d != ROW_DIM {
            return Err(EngramFuseError::BadShape);
        }
        let mut heads = [HashHead {
            n: 0,
            k: 0,
            modulus: 1,
            seed: 0,
        }; K_MAX];
        for (k, h) in heads.iter_mut().enumerate() {
            h.n = 0;
            h.k = k as u8;
            h.modulus = u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap());
            h.seed = u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap());
        }
        let n_entries = u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap()) as usize;
        // Shape sanity: 44-byte entries must fit in the remaining bytes
        // (minus the 32-byte root).
        let entry_bytes = ENTRY_SERIALIZED_BYTES;
        let expected = cur + n_entries * entry_bytes + 32;
        if expected != buf.len() {
            return Err(EngramFuseError::BadLength {
                expected,
                got: buf.len(),
            });
        }
        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            let key = [
                u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap()),
                u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap()),
                u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap()),
                u64::from_le_bytes(take(&buf, &mut cur, 8)?.try_into().unwrap()),
            ];
            let sum_outcome = f32::from_le_bytes(take(&buf, &mut cur, 4)?.try_into().unwrap());
            let count = f32::from_le_bytes(take(&buf, &mut cur, 4)?.try_into().unwrap());
            let sum_conc = f32::from_le_bytes(take(&buf, &mut cur, 4)?.try_into().unwrap());
            entries.push(MinedEntry {
                key,
                sum_outcome,
                count,
                sum_conc,
            });
        }
        if entries.windows(2).any(|w| w[0].key >= w[1].key) {
            return Err(EngramFuseError::UnsortedEntries);
        }
        let mut root = [0u8; 32];
        root.copy_from_slice(take(&buf, &mut cur, 32)?);
        let table = build_table_from_entries(n_slots, &heads, &entries);
        let got = table.commitment();
        if got != root {
            return Err(EngramFuseError::CommitmentMismatch {
                expected: root,
                got,
            });
        }
        Ok(Self {
            n_slots,
            heads,
            entries,
            root,
        })
    }
}

/// Cursor-bounded slice take for [`MinedTable::load`].
fn take<'a>(buf: &'a [u8], cur: &mut usize, n: usize) -> Result<&'a [u8], EngramFuseError> {
    if *cur + n > buf.len() {
        return Err(EngramFuseError::BadLength {
            expected: *cur + n,
            got: buf.len(),
        });
    }
    let s = &buf[*cur..*cur + n];
    *cur += n;
    Ok(s)
}

/// Write every entry's row under all K_MAX head slots (last-write-wins at
/// collision, the substrate's documented semantics).
fn build_table_from_entries(
    n_slots: usize,
    heads: &[HashHead; K_MAX],
    entries: &[MinedEntry],
) -> InMemoryEngramTable {
    let mut b = EngramTableBuilder::new(n_slots.max(1), ROW_DIM).with_heads(*heads);
    for e in entries {
        let keys = keys_for(&e.key, heads);
        let row = row_of(e);
        for k in keys {
            b.add_pattern(k, &row);
        }
    }
    b.build()
}

/// Read-side memory held by the fused `PuctPlayer` (feature-gated field).
pub struct EngramPuctMemory {
    table: InMemoryEngramTable,
    /// Preallocated `lookup_into` out buffer — the hot path is zero-alloc.
    lookup_out: Box<[f32; K_MAX * ROW_DIM]>,
    lookups: u64,
    fires: u64,
}

/// One fused read: what `puct::expand` needs to apply the fusion rule.
#[derive(Debug, Clone, Copy)]
pub struct MemoryRow {
    /// Mean mined value from the position's to_play perspective, [-1, 1].
    pub value_mean: f32,
    /// Count-based evidence gate, (0, 1].
    pub gate: f32,
    /// Sharpening factor for the child softmax, [0.5, 2].
    pub gamma: f32,
}

impl EngramPuctMemory {
    pub fn from_table(table: InMemoryEngramTable) -> Self {
        Self {
            table,
            lookup_out: Box::new([0.0; K_MAX * ROW_DIM]),
            lookups: 0,
            fires: 0,
        }
    }

    pub fn from_mined(mined: &MinedTable) -> Self {
        Self::from_table(mined.canonical_table())
    }

    pub fn commitment(&self) -> [u8; 32] {
        self.table.commitment()
    }

    pub fn table(&self) -> &InMemoryEngramTable {
        &self.table
    }

    /// Telemetry: (lookups, fires) since construction.
    pub fn telemetry(&self) -> (u64, u64) {
        (self.lookups, self.fires)
    }

    /// One fused read for `board`. `None` = no memory fired (empty table,
    /// zero-slot table, or all-K slots empty): the caller must then behave
    /// bit-identically to the feature-off player (G1).
    ///
    /// Zero-allocation: scratch is preallocated, aggregation is a fixed
    /// K_MAX × ROW_DIM stack fold.
    pub fn read(&mut self, board: &Board) -> Option<MemoryRow> {
        self.lookups += 1;
        let n_slots = self.table.num_slots();
        if n_slots == 0 {
            return None;
        }
        let mut ids = [CanonicalId(0); 4];
        pack_tt_key(board, &mut ids);
        let hash_keys = multi_head_hash(&ids, self.table.heads());
        let hits = self.table.lookup_into(&hash_keys, self.lookup_out.as_mut());
        if hits == 0 {
            return None;
        }
        // Mean over the populated rows (a zero row is a miss, not data).
        let mut sv = 0.0f32;
        let mut sn = 0.0f32;
        let mut sb = 0.0f32;
        let mut nonzero = 0usize;
        for k in 0..K_MAX {
            let r = &self.lookup_out[k * ROW_DIM..(k + 1) * ROW_DIM];
            if r[0] == 0.0 && r[1] == 0.0 && r[2] == 0.0 {
                continue;
            }
            nonzero += 1;
            sv += r[0];
            sn += r[1];
            sb += r[2];
        }
        debug_assert_eq!(nonzero, hits, "hit count disagreement with zero-row scan");
        if nonzero == 0 {
            return None;
        }
        let inv = 1.0 / nonzero as f32;
        let v = (sv * inv).clamp(-1.0, 1.0);
        let n = sn * inv;
        let b = sb * inv;
        let gate = evidence_gate(n);
        let gamma = sharpen_gamma(b, gate);
        self.fires += 1;
        Some(MemoryRow {
            value_mean: v,
            gate,
            gamma,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use katgpt_core::engram::EngramTableBuilder;

    #[test]
    fn gate_monotone_and_pinned() {
        // n=1 is a rumor (well under half evidence), n=N_MIN is the midpoint.
        let g1 = evidence_gate(1.0);
        let g8 = evidence_gate(N_MIN);
        let g64 = evidence_gate(64.0);
        assert!(g1 < 0.2, "n=1 gate {g1} not rumor-damped");
        assert!((g8 - 0.5).abs() < 1e-5, "n=N_MIN gate {g8} not 0.5");
        assert!(g64 > 0.99, "n=64 gate {g64} not saturated");
        assert!(g1 < g8 && g8 < g64, "gate not monotone in n");
    }

    #[test]
    fn gamma_bounded_everywhere() {
        // The issue's hazard: unbounded exp on the sharpening path. Sweep
        // the whole (b, gate) input space — γ must stay in [0.5, 2].
        for bi in 0..=20 {
            for gi in 0..=20 {
                let b = bi as f32 / 20.0;
                let g = gi as f32 / 20.0;
                let gamma = sharpen_gamma(b, g);
                assert!(
                    (GAMMA_MIN..=GAMMA_MAX).contains(&gamma),
                    "γ={gamma} at b={b} gate={g}"
                );
            }
        }
        assert_eq!(
            sharpen_gamma(0.5, 1.0),
            1.0,
            "neutral concentration must be γ=1"
        );
    }

    #[test]
    fn concentration_extremes() {
        assert_eq!(visit_concentration(&[]), 0.0);
        assert_eq!(
            visit_concentration(&[5]),
            1.0,
            "single child = fully concentrated"
        );
        assert_eq!(visit_concentration(&[0]), 1.0);
        let flat = visit_concentration(&[10, 10, 10, 10]);
        assert!(
            flat < 0.01,
            "uniform visits should be ~0 concentration, got {flat}"
        );
        let peaked = visit_concentration(&[100, 1, 1, 1]);
        assert!(
            peaked > 0.8,
            "peaked visits should be high concentration, got {peaked}"
        );
    }

    #[test]
    fn tt_key_distinguishes_ko_and_to_play() {
        let mut a = Board::new();
        a.play(40); // Black center
        let mut b = a;
        b.pass();
        // Same cells, different to_play (+ ko cleared) → different key.
        assert_ne!(tt_key_words(&a), tt_key_words(&b));
        let mut c = a;
        c.play(41);
        assert_ne!(tt_key_words(&a), tt_key_words(&c));
        assert_eq!(
            tt_key_words(&a),
            tt_key_words(&a),
            "key must be deterministic"
        );
    }

    #[test]
    fn mined_table_round_trip_and_determinism() {
        let heads = default_heads_for_tests();
        let mk = |k: [u64; 4], s: f32, c: f32, b: f32| MinedEntry {
            key: k,
            sum_outcome: s,
            count: c,
            sum_conc: b,
        };
        let entries = vec![
            mk([1, 2, 3, 4], 0.5, 12.0, 0.7),
            mk([9, 9, 9, 9], -30.0, 40.0, 0.2),
            mk([5, 0, 0, 1], 3.0, 3.0, 0.9),
        ];
        let t1 = MinedTable::build(1 << 12, heads, entries.clone());
        let t2 = MinedTable::build(1 << 12, heads, entries);
        assert_eq!(
            t1.root, t2.root,
            "same entries in any input order → same root (G6)"
        );
        assert!(
            t1.entries.windows(2).all(|w| w[0].key < w[1].key),
            "build must sort"
        );

        let dir = std::env::temp_dir().join(format!("e868_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t1.kept");
        t1.save(&path).unwrap();
        let loaded = MinedTable::load(&path).unwrap();
        assert_eq!(loaded.root, t1.root, "round-trip preserves the commitment");
        assert_eq!(loaded.entries, t1.entries);
        std::fs::remove_dir_all(&dir).unwrap();

        // Corruption → loud refusal (G6 tamper arm).
        let dir = std::env::temp_dir().join(format!("e868_corrupt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t2.kept");
        t1.save(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            MinedTable::load(&path),
            Err(EngramFuseError::CommitmentMismatch { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn empty_table_read_is_none() {
        // A built-but-empty table: every read misses → G1 posture.
        let heads = default_heads_for_tests();
        let t = MinedTable::build(1 << 10, heads, Vec::new());
        let mut mem = EngramPuctMemory::from_mined(&t);
        let b = Board::new();
        assert!(mem.read(&b).is_none(), "empty table must never fire");
        let (lookups, fires) = mem.telemetry();
        assert_eq!((lookups, fires), (1, 0));
    }

    #[test]
    fn populated_slot_fires_with_expected_row() {
        let heads = default_heads_for_tests();
        // One entry: a draw-ish value, huge count (gate ≈ 1), neutral b (γ=1).
        let e = MinedEntry {
            key: tt_key_words(&Board::new()),
            sum_outcome: 0.0,
            count: 1000.0,
            sum_conc: 0.5,
        };
        let t = MinedTable::build(1 << 10, heads, vec![e]);
        let mut mem = EngramPuctMemory::from_mined(&t);
        let row = mem
            .read(&Board::new())
            .expect("the mined position must fire");
        assert!(
            (row.gate - 1.0).abs() < 1e-4,
            "n=1000 saturates, got {}",
            row.gate
        );
        assert!(
            (row.gamma - 1.0).abs() < 1e-5,
            "b=0.5 is γ-neutral, got {}",
            row.gamma
        );
        assert!(row.value_mean.abs() < 1e-6);
    }

    /// Deterministic head set for tests (same shape as the miner's: the
    /// builder's default heads at a fixed size).
    fn default_heads_for_tests() -> [HashHead; K_MAX] {
        // Derive from the substrate the same way the miner does: build a
        // tiny table and take its heads.
        let t = EngramTableBuilder::new(1 << 12, ROW_DIM).build();
        *t.heads()
    }
}

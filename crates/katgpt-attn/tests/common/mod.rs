//! Shared fixture parser + replay harness for the ASEntmax real-model
//! re-gate tests (Issue 747 P0.7 + the Issue 762 long-context re-measure).
//!
//! Fixture format: `examples/asentmax_p07_gen_fixture.rs` `write_fixture`
//! (magic `ASEP07\0\0`, version 1 — summaries/query f16, oracle masses f32).
//! Two committed fixtures consume this module:
//!
//! - `tests/data/asentmax_p07_bonsai8b.fixture` — the P0.7 baseline
//!   (2085 tokens / 32 blocks, 8 layers × 8 q-heads, every k-end).
//! - `tests/data/asentmax_long_context.fixture` — the long-context re-measure
//!   (≈10.9k tokens / ≈170 blocks, same 8 layers × gqa-spread 4 q-heads,
//!   log-spaced k-ends beyond 32).
//!
//! This module is NOT a test target (cargo skips `tests/<subdir>/`); each
//! test binary opts in with `mod common;`.

#![allow(dead_code)]

use katgpt_attn::dash_attn::entmax_router::{EntmaxCache, EntmaxRouter};
use katgpt_attn::dash_attn::vortex_flow::{VortexFlow, VortexScratch};

// ── f16 ─────────────────────────────────────────────────────────────────────

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e: i32 = -1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            (sign << 31) | (((113 - e) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

// ── fixture format (mirrors examples/asentmax_p07_gen_fixture.rs) ──────────

pub struct Cur<'a> {
    pub b: &'a [u8],
    pub p: usize,
}

impl Cur<'_> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn f16(&mut self) -> f32 {
        let h = u16::from_le_bytes([self.b[self.p], self.b[self.p + 1]]);
        self.p += 2;
        f16_to_f32(h)
    }
}

pub struct Stream {
    pub layer: usize,
    pub kv_head: usize,
    pub n_blocks: usize,
    /// [n_blocks][head_dim] mean-pooled post-rope K summaries.
    pub sums: Vec<f32>,
}

pub struct Row {
    pub layer: usize,
    pub q_head: usize,
    pub kv_head: usize,
    pub n_blocks: usize,
    pub query: Vec<f32>,
    /// [n_blocks] ground-truth softmax attention mass per block.
    pub masses: Vec<f32>,
}

pub struct Fixture {
    pub head_dim: usize,
    pub block: usize,
    pub n_tokens: usize,
    pub prompt_fnv: u64,
    /// The planted needle sentence's 64-token block (Bench 032 axis).
    pub needle_block: usize,
    pub streams: Vec<Stream>,
    pub rows: Vec<Row>,
}

pub fn parse_fixture(bytes: &[u8]) -> Fixture {
    let mut c = Cur { b: bytes, p: 0 };
    let magic = &bytes[0..8];
    assert_eq!(magic, b"ASEP07\0\0", "fixture magic");
    c.p = 8;
    let version = c.u32();
    assert_eq!(version, 1, "fixture version");
    let head_dim = c.u32() as usize;
    let block = c.u32() as usize;
    let n_streams = c.u32() as usize;
    let n_rows = c.u32() as usize;
    let n_tokens = c.u32() as usize;
    let prompt_fnv = c.u64();
    let needle_block = c.u64() as usize;
    let mut streams = Vec::with_capacity(n_streams);
    for _ in 0..n_streams {
        let layer = c.u32() as usize;
        let kv_head = c.u32() as usize;
        let n_blocks = c.u32() as usize;
        let mut sums = vec![0f32; n_blocks * head_dim];
        for s in &mut sums {
            *s = c.f16();
        }
        streams.push(Stream {
            layer,
            kv_head,
            n_blocks,
            sums,
        });
    }
    let mut rows = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let layer = c.u32() as usize;
        let q_head = c.u32() as usize;
        let kv_head = c.u32() as usize;
        let n_blocks = c.u32() as usize;
        let mut query = vec![0f32; head_dim];
        for q in &mut query {
            *q = c.f16();
        }
        let mut masses = vec![0f32; n_blocks];
        for m in &mut masses {
            *m = c.f32();
        }
        rows.push(Row {
            layer,
            q_head,
            kv_head,
            n_blocks,
            query,
            masses,
        });
    }
    assert_eq!(c.p, bytes.len(), "fixture fully consumed");
    Fixture {
        head_dim,
        block,
        n_tokens,
        prompt_fnv,
        needle_block,
        streams,
        rows,
    }
}

// ── replay harness ──────────────────────────────────────────────────────────

pub fn cache_for(fx: &Fixture, layer: usize, kv_head: usize, n_blocks: usize) -> EntmaxCache {
    let stream = fx
        .streams
        .iter()
        .find(|s| s.layer == layer && s.kv_head == kv_head)
        .unwrap_or_else(|| panic!("stream for layer {layer} kvh {kv_head}"));
    assert!(stream.n_blocks >= n_blocks);
    let mut cache = EntmaxCache::with_capacity(n_blocks, fx.head_dim);
    for b in 0..n_blocks {
        cache
            .summaries
            .push(stream.sums[b * fx.head_dim..(b + 1) * fx.head_dim].to_vec());
    }
    cache
}

pub fn stream_n_blocks(fx: &Fixture, layer: usize, kv_head: usize) -> usize {
    fx.streams
        .iter()
        .find(|s| s.layer == layer && s.kv_head == kv_head)
        .map_or(0, |s| s.n_blocks)
}

pub struct Decision {
    pub blocks: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Replay all rows through one arm. `scheduled` uses the P0.7 wiring exactly
/// (rolling-σ̂ α=0.8 fed every row in stream order, `to_schedule()` per call).
pub fn replay(fx: &Fixture, scheduled: bool) -> Vec<Decision> {
    let router = if scheduled {
        EntmaxRouter::default_router().with_asentmax_schedule()
    } else {
        EntmaxRouter::default_router()
    };
    replay_via(&router, fx)
}

/// Replay all rows through a caller-constructed router (fresh-estimator
/// σ̂ buckets use this with a router per bucket).
pub fn replay_via(router: &EntmaxRouter, fx: &Fixture) -> Vec<Decision> {
    let mut caches: std::collections::HashMap<(usize, usize), EntmaxCache> = Default::default();
    let mut scratch = VortexScratch::new(256);
    let mut out = Vec::with_capacity(fx.rows.len());
    for row in &fx.rows {
        let key = (row.layer, row.kv_head);
        let cache = caches.entry(key).or_insert_with(|| {
            cache_for(
                fx,
                row.layer,
                row.kv_head,
                stream_n_blocks(fx, row.layer, row.kv_head),
            )
        });
        // ensure capacity for this row's n_blocks (caches only grow)
        if cache.summaries.len() < row.n_blocks {
            let full = cache_for(fx, row.layer, row.kv_head, row.n_blocks);
            *cache = full;
        }
        let dec =
            router.forward_indexer(&row.query, cache, row.n_blocks, row.n_blocks, &mut scratch);
        out.push(Decision {
            blocks: dec.blocks,
            weights: dec.weights,
        });
    }
    out.shrink_to_fit();
    out
}

pub fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

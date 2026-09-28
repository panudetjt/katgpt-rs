//! Issue 800 **Arm B1** — lock-free slot-flip `DoubleBuffer` PoC (crate: `katgpt-kv`).
//!
//! Decision-gated by **B2**: promote the slot-flip upgrade of
//! `src/async_qdq.rs::DoubleBuffer` only if it beats BOTH baselines
//! (single-threaded serial AND std `mpsc` channel) by > 10 %. If channels tie,
//! this arm records a DECLINE — that is a fully valid outcome: pufferlib needs
//! the spin machine because C pthreads has no channels; Rust does.
//!
//! # Provenance
//!
//! - Mechanic source: PufferLib Cleanba 2-slot async pipelining
//!   (`pufferl.cu:3073-3075`: "warmup fills slot 0; then collect into write
//!   while training the other slot (exactly one epoch old)") — distilled in
//!   riir-train [Research 454](../../riir-train/.research/454_pufferlib_pooled_env_rollout_architecture.md)
//!   §2.1, routed to katgpt-rs Issue 800 Arm B (modelless, lock-free).
//! - The surface being POC'd: `katgpt-kv::async_qdq::DoubleBuffer` — slots
//!   exist, overlap is *simulated* (single-threaded; `swap()` is a
//!   `mem::swap`). This PoC measures whether the REAL cross-thread mechanic
//!   buys anything over the honest Rust defaults. The upgrade itself happens
//!   only on a promote verdict, as a follow-up — this file does not touch
//!   `src/async_qdq.rs`.
//!
//! # The slot-flip protocol (normative copy lives in
//! `tests/slot_flip_staleness.rs`; this file is kept in deliberate sync)
//!
//! State: **two flat buffers** (`slots: [Box<[AtomicU32]>; 2]`, allocated
//! once — zero alloc in the flip loop) + **one `AtomicUsize` `turn`**
//! (seq-cst). No `Mutex`, no lock of any kind, no `unsafe`.
//!
//! Turn arithmetic (a monotone ticket counter, not a slot id — this is what
//! makes it correct; see the staleness proof below):
//!
//! ```text
//! producer epoch p:  wait turn ≥ 2p−1  →  write slot[p & 1]  →  turn += 1
//! consumer frame c:  wait turn ≥ 2c+1  →  read  slot[c & 1]  →  turn += 1
//! ```
//!
//! Both sides bump the ONE monotone counter with one seq-cst RMW each
//! (publishes and releases interleave freely — the counter only grows, so
//! gates are `≥` and there is no ABA). Each gate is EXACT because the
//! waiter's own contribution to the counter is known to itself:
//!
//! - producer at epoch `p` has published exactly `p` frames, so
//!   `turn ≥ 2p−1` ⟺ releases ≥ `p−1` — every frame that ever occupied
//!   slot `p & 1` (most recently frame `p−2`) has been RELEASED. Off by
//!   one here (`2p−2`) is the classic bug: extra publishes then compensate
//!   for a missing release and the producer writes a slot the consumer is
//!   still reading.
//! - consumer at frame `c` has released exactly `c` frames, so
//!   `turn ≥ 2c+1` ⟺ publishes ≥ `c+1` — frame `c` itself (publishes are
//!   epoch-ordered by the single producer thread) is complete.
//!
//! The gate is NOT a wait on frame `p−1`'s release — that over-sync (a
//! plausible first draft) forces the strict sequence P0→C0→P1→C1 with ZERO
//! overlap. Gating one release earlier lets producer epoch `c+1` fill the
//! other slot WHILE the consumer reads frame `c` — exactly the Cleanba
//! produce-next-while-consume-current cadence — and at balanced cost the
//! producer gets a full consumer-frame head start, making the steady state
//! wait-free (handoff cost appears only when phases drift).
//!
//! **Warmup (`Cleanba`: "warmup fills slot 0"):** producer epoch 0 runs
//! solo (gate vacuous) — the consumer's first gate (`turn ≥ 1`) unblocks
//! only after slot 0 is fully written and published. Epoch 1's gate
//! (`turn ≥ 1`) is already satisfied by the producer's own frame-0 publish,
//! so the first true overlap starts immediately: producer fills slot 1
//! while the consumer reads frame 0 from slot 0.
//!
//! **One-slot-staleness argument (B3, is the *spec*):** while the consumer
//! reads frame `c`, the producer — a SINGLE thread running epochs in order —
//! is in epoch `c` (slow producer: consumer blocks at the gate, never reads
//! stale-empty data) or epoch `c+1` (the pipeline case). The consumer
//! therefore always holds the newest PUBLISHED frame — never a fresher one
//! (the producer's in-flight frame lives in the OTHER slot), never a skipped
//! or duplicated one (publishes are epoch-ordered by the single producer
//! thread; the consumer's counter is private and reads each frame exactly
//! once).
//!
//! **Torn-read structural argument:** producer writes ONLY `slots[p & 1]`
//! inside its fill-window `turn ∈ [2p−1, 2p)` (gate to publish; during the
//! fill the counter is `p + R` with releases `R ∈ [p−1, p]` — frame `p−1`
//! may legally be released mid-fill, frame `p` is unpublished hence
//! unread). Consumer reads ONLY `slots[c & 1]` inside its read-window
//! `turn ∈ [2c+1, 2c+2]` (own releases `= c`; publishes can advance from
//! `c+1` to `c+2` mid-read — epoch `c+2` needs release `c+1` which does
//! not exist yet). Same-slot collision needs equal parity AND overlapping
//! windows: `p = c` → `[2c−1, 2c)` vs `[2c+1, 2c+2]` disjoint; `p = c+2`
//! → fill starts at `turn ≥ 2c+3` — strictly after the read-window's
//! ceiling, and the release RMW of frame `c` that enables it is itself
//! sequenced-after every element read of frame `c` (seq-cst chain).
//! `|p − c| ≥ 4` cannot run concurrently. So write-while-read **cannot be
//! expressed** in the protocol — the exact line is
//! `let slot: &[AtomicU32] = &self.slots[epoch & 1];` in `produce_with`
//! below (and its twin in the test file).
//!
//! **Why the frames are `[AtomicU32]` (read before "fixing" this):** pure
//! safe Rust cannot time-multiplex `&mut [f32]` across two threads: no
//! `unsafe` (house rule for this PoC) means no `UnsafeCell`, and no locks
//! means no `Mutex`. Element atomics make every access `&self`, so the whole
//! machine is `Sync` with zero unsafe. Cost honesty: on arm64 (this M3) and
//! x86, `Relaxed` atomic loads/stores lower to plain `ldr`/`str` — the
//! element representation is near-free on this box; the real cost is that
//! LLVM will not auto-vectorize atomic loops. That is why **all three
//! variants use the identical `[AtomicU32]` representation and identical
//! kernels** — the per-variant wall-time delta is then purely handoff +
//! overlap, which is the decision quantity. Data visibility is ordered
//! through the seq-cst `turn` chain (element `Relaxed` ops are
//! sequenced-before the publishing RMW; the consumer's observing load gives
//! happens-before) — no per-element ordering needed.
//!
//! # Variants
//!
//! - **(a) slot-flip** — the protocol above; producer/consumer spin with
//!   `std::hint::spin_loop()`, escalating to `std::thread::yield_now()` after
//!   a bounded spin budget (`SPIN_BUDGET` iterations), then yielding for the
//!   remainder of that wait. 2 threads.
//! - **(b) serial** — single-threaded produce-then-consume through two
//!   buffers swapped per frame: exactly the current `async_qdq::DoubleBuffer`
//!   shape (`shadow_mut()` → `mark_shadow_ready()` → `swap()`). The floor.
//!   1 thread.
//! - **(c) mpsc** — `std::sync::mpsc` (std, not crossbeam — no new deps; this
//!   is the honest default-Rust bar), **copy per send**: the producer fills a
//!   per-slot staging buffer and sends a *clone* (one heap alloc + frame
//!   copy per frame — what a frame-per-message world does at 16/64 KiB); the
//!   consumer receives, consumes, drops (one free per frame). 2 threads.
//!   Steady-state allocation is expected and counted here — the zero-alloc
//!   house rule binds the slot-flip variant's flip loop, not the channel's
//!   ownership semantics.
//!
//! # Tick discipline modeled
//!
//! The `async_qdq` cadence is one flip per scheduler tick (Plan 227 Phase 6):
//! the GPU consumes chunk *N* while the CPU dequantizes chunk *N+1*; the
//! flip publishes at the tick boundary. At the 20 Hz server cadence a tick is
//! 50 ms and per-frame compute (dequantize + attention) is the budget tenant;
//! the handoff is not. The PoC models the steady-state inner loop of that
//! cadence: producer synthesizes frame *i+1* while the consumer finishes
//! frame *i*, then both flip. Two compute cells per the issue:
//!
//! - **long** — consumer compute calibrated to ≈ 200 µs/frame (100–500 µs
//!   band; the async_qdq regime: its own GOAT test models 50 µs attention +
//!   30 µs dequantize per 64 KiB chunk, and real dequant kernels are of that
//!   order).
//! - **short** — calibrated to ≈ 4 µs/frame (1–10 µs band): the regime where
//!   a channel handoff stops being noise.
//!
//! Producer work is calibrated to the SAME target (real dequantize ≈
//! attention in cost), so the pipeline ceiling is ≈ 2× over serial.
//!
//! # Measurement honesty
//!
//! - 2-thread measurement (producer + consumer); serial is 1-thread. No core
//!   pinning (not portable). Run on an M3 Max (10 perf cores) — do not
//!   over-claim beyond 2 threads.
//! - N = 2000 frames per rep; 2 warmup reps; 7 timed reps; **median**
//!   reported (min/max printed for spread). `std::time::Instant`.
//! - Compute kernels are deterministic (`f32::mul_add`, fixed coefficients,
//!   fixed element order) and identical across variants → per-variant
//!   checksums must be **bit-identical**; a mismatch is printed as a bug
//!   witness (the staleness test is the hard gate, this is a tripwire).
//! - `cargo bench` builds with the release profile (bench profile).
//!
//! # B2 decision gate (printed at the end, verbatim numbers)
//!
//! Per (frame size × compute) cell: slot-flip vs serial and slot-flip vs
//! mpsc deltas. PROMOTE-CANDIDATE only if slot-flip beats BOTH by > 10 % in
//! a cell; the **binding cell for the async_qdq upgrade is the long-compute
//! cell** (that is the regime its consumers run — see the consumer-regime
//! note printed with the verdict). A short-cell-only win is reported as a
//! regime-map result, not a promotion.

use std::hint::black_box;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

/// Spin iterations before escalating to `yield_now` (bounded spin-then-yield).
const SPIN_BUDGET: u32 = 2048;

/// A zeroed frame of `elems` f32 elements. (`AtomicU32` is not `Clone` on
/// current std, so the `vec![expr; n]` form is unavailable — map-collect.)
fn zero_frame(elems: usize) -> Box<[AtomicU32]> {
    (0..elems)
        .map(|_| AtomicU32::new(0))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// Element-wise frame copy (the honest per-send copy of the mpsc variant —
/// `Clone` is unavailable on atomics, and this loop is exactly what it did:
/// load+store per element).
fn copy_frame(src: &[AtomicU32], dst: &[AtomicU32]) {
    for (d, s) in dst.iter().zip(src.iter()) {
        d.store(s.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}
/// Frames per timed rep.
const FRAMES: usize = 2000;
/// Untimed warmup reps per variant.
const WARMUP_REPS: usize = 2;
/// Timed reps per variant (median reported).
const TIMED_REPS: usize = 7;
/// Frame size in f32 elements: 16 KiB (primary — the KV-chunk scale).
const ELEMS_PRIMARY: usize = 4096;
/// Frame size in f32 elements: 64 KiB (secondary — the async_qdq GOAT-test
/// chunk shape: 128 tokens × 128 dim).
const ELEMS_SECONDARY: usize = 16_384;
/// Short-compute cell target, µs per frame (issue band 1–10 µs).
const SHORT_TARGET_US: f64 = 4.0;
/// Long-compute cell target, µs per frame (issue band 100–500 µs).
const LONG_TARGET_US: f64 = 200.0;

/// Lock-free 2-slot flip buffer — Cleanba slot-ownership mechanic
/// (Research 454 §2.1 → Issue 800 Arm B). See the module docs for the full
/// protocol + staleness/torn-read proofs.
///
/// Normative copy: `tests/slot_flip_staleness.rs` (kept in deliberate sync —
/// the write scope for this arm forbids touching `src/async_qdq.rs`, and a
/// bench target cannot be imported by a test target).
struct SlotFlip {
    /// The two flat frame buffers. Allocated once; the flip loop is
    /// zero-alloc. Access is protocol-gated: producer writes only
    /// `slots[epoch & 1]`, consumer reads only `slots[index & 1]`.
    slots: [Box<[AtomicU32]>; 2],
    /// The one coordination variable: counts completed half-steps.
    /// Even = consumer released a slot; odd = producer published a frame.
    /// All ops seq-cst (the issue's "flip = one seq-cst store", generalized
    /// to an RMW because BOTH sides flip — see "who flips what when").
    turn: AtomicUsize,
}

impl SlotFlip {
    fn new(elems: usize) -> Self {
        Self {
            slots: [zero_frame(elems), zero_frame(elems)],
            turn: AtomicUsize::new(0),
        }
    }

    /// Producer epoch `p`: block until slot `p & 1` is provably free
    /// (`turn ≥ 2p−1`; own publishes `= p`, so this is exactly "releases
    /// ≥ `p−1`" — frame `p−2`, the slot's latest tenant, has been RELEASED;
    /// frame `p−1` may legitimately still be being read from the OTHER
    /// slot), fill it, publish with ONE seq-cst RMW.
    ///
    /// ⚠ The torn-read structural line: the write target below is
    /// `slots[epoch & 1]` *only*, and only inside the fill-window
    /// `turn ∈ [2p−1, 2p)`. The concurrent consumer read-window covers the
    /// OTHER slot's frames — see the module-doc proof.
    fn produce_with(&self, epoch: usize, fill: impl FnOnce(&[AtomicU32])) {
        // `turn + 1 ≥ 2*epoch` ⟺ `turn ≥ 2p−1` without the p=0 underflow;
        // p=0 is the Cleanba warmup (vacuous gate), p=1's gate is met by the
        // producer's own frame-0 publish (first overlap starts immediately).
        let mut spins: u32 = 0;
        while self.turn.load(Ordering::SeqCst) + 1 < 2 * epoch {
            if spins < SPIN_BUDGET {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
        let slot: &[AtomicU32] = &self.slots[epoch & 1];
        fill(slot);
        // THE flip: publish slot `epoch & 1` as frame `epoch`.
        self.turn.fetch_add(1, Ordering::SeqCst);
    }

    /// Consumer frame `index`: block until it is published
    /// (`turn ≥ 2*index + 1` — the single producer thread publishes in epoch
    /// order, so reaching `2c+1` proves frame `c` itself is complete), read
    /// it, release the slot with ONE seq-cst RMW. The `&[AtomicU32]`
    /// reference never escapes the closure, so the borrow cannot outlive the
    /// protocol window it belongs to.
    fn consume_with<R>(&self, index: usize, read: impl FnOnce(&[AtomicU32]) -> R) -> R {
        let mut spins: u32 = 0;
        while self.turn.load(Ordering::SeqCst) < 2 * index + 1 {
            if spins < SPIN_BUDGET {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
        let slot: &[AtomicU32] = &self.slots[index & 1];
        let out = read(slot);
        // THE other flip: release the slot for reuse in epoch `index + 2`.
        self.turn.fetch_add(1, Ordering::SeqCst);
        out
    }
}

/// Producer kernel: deterministic in-place transform, `passes` sweeps over
/// the frame. Read-modify-write with `a < 1` per pass → values converge into
/// a bounded fixed point (no NaN/Inf at any pass count). Same kernel for all
/// variants.
fn fill_transform(frame: &[AtomicU32], epoch: usize, passes: usize) {
    for p in 0..passes {
        // a < 1 strictly: bounded iteration; b small: fixed point stays O(1).
        let a = 0.937_5 + ((p ^ epoch) & 0xF) as f32 * 0.003_906_25;
        let b = ((epoch & 0xF) as f32) * 0.125 - 0.5;
        for cell in frame.iter() {
            let v = f32::from_bits(cell.load(Ordering::Relaxed));
            cell.store(v.mul_add(a, b).to_bits(), Ordering::Relaxed);
        }
    }
}

/// Consumer kernel: deterministic multiply-accumulate sum, `passes` sweeps,
/// pass-dependent multiplier (prevents cross-pass hoisting). FP reduction is
/// order-fixed and not reassociated by LLVM → bit-identical across variants
/// given identical frames.
fn mac_sum(frame: &[AtomicU32], salt: usize, passes: usize) -> f32 {
    let mut acc = 0.0_f32;
    for p in 0..passes {
        let m = 1.0 + ((p ^ salt) & 0xFF) as f32 * 0.001;
        for cell in frame.iter() {
            let v = f32::from_bits(cell.load(Ordering::Relaxed));
            acc = v.mul_add(m, acc);
        }
    }
    acc
}

/// Calibrate sweep counts so one kernel call ≈ `target_us`. Returns
/// `(consumer_passes, producer_passes)` — calibrated independently because
/// the two kernels have different per-pass costs (RMW transform vs FP MAC).
fn calibrate(elems: usize, target_us: f64) -> (usize, usize) {
    let scratch = zero_frame(elems);
    let probe = |kernel: &dyn Fn(&[AtomicU32])| {
        // Warm the caches/branch predictors, then time three batches; median.
        for _ in 0..32 {
            kernel(&scratch);
        }
        let mut batch: [f64; 3] = [0.0; 3];
        for slot in &mut batch {
            let t0 = Instant::now();
            for _ in 0..64 {
                kernel(&scratch);
            }
            *slot = t0.elapsed().as_secs_f64() * 1e6 / 64.0;
        }
        batch.sort_by(|a, b| a.total_cmp(b));
        batch[1]
    };
    let mac_us = probe(&|f: &[AtomicU32]| {
        black_box(mac_sum(f, 0, 1));
    });
    let fill_us = probe(&|f: &[AtomicU32]| {
        fill_transform(f, 0, 1);
    });
    let mac_passes = ((target_us / mac_us).round() as usize).max(1);
    let fill_passes = ((target_us / fill_us).round() as usize).max(1);
    (mac_passes, fill_passes)
}

/// One timed rep of the serial baseline: fill next, swap, consume — the exact
/// `async_qdq::DoubleBuffer` shape, 1 thread.
fn run_serial(elems: usize, mac_passes: usize, fill_passes: usize) -> f32 {
    let mut cur: Box<[AtomicU32]> = zero_frame(elems);
    let mut nxt: Box<[AtomicU32]> = zero_frame(elems);
    let mut sum = 0.0_f32;
    for c in 0..FRAMES {
        fill_transform(&nxt, c, fill_passes);
        std::mem::swap(&mut cur, &mut nxt);
        sum += mac_sum(&cur, c, mac_passes);
    }
    sum
}

/// One timed rep of the slot-flip variant: producer thread + consumer
/// (parent) thread, 2 threads, zero steady-state alloc.
fn run_slot_flip(elems: usize, mac_passes: usize, fill_passes: usize) -> f32 {
    let sf = SlotFlip::new(elems);
    let mut sum = 0.0_f32;
    thread::scope(|s| {
        s.spawn(|| {
            for p in 0..FRAMES {
                sf.produce_with(p, |slot| fill_transform(slot, p, fill_passes));
            }
        });
        for c in 0..FRAMES {
            sum += sf.consume_with(c, |slot| mac_sum(slot, c, mac_passes));
        }
    });
    sum
}

/// One timed rep of the mpsc variant: copy-per-send (clone = alloc + frame
/// copy per frame), recv-consume-drop on the consumer side. Two alternating
/// staging buffers so the per-slot fill sequences (and therefore the
/// checksums) are identical to the other variants.
fn run_mpsc(elems: usize, mac_passes: usize, fill_passes: usize) -> f32 {
    let staging: [Box<[AtomicU32]>; 2] = [zero_frame(elems), zero_frame(elems)];
    let (tx, rx) = mpsc::channel::<Box<[AtomicU32]>>();
    let mut sum = 0.0_f32;
    thread::scope(|s| {
        s.spawn(move || {
            for p in 0..FRAMES {
                fill_transform(&staging[p & 1], p, fill_passes);
                // Copy per send: fresh heap frame + element copy — the honest
                // frame-per-message cost at this size.
                let msg = zero_frame(elems);
                copy_frame(&staging[p & 1], &msg);
                if tx.send(msg).is_err() {
                    return; // consumer gone (never happens in this harness)
                }
            }
        });
        for c in 0..FRAMES {
            // recv + consume + drop (free) — the honest frame-per-message life.
            match rx.recv() {
                Ok(frame) => sum += mac_sum(&frame, c, mac_passes),
                Err(_) => break,
            }
        }
    });
    sum
}

struct VariantResult {
    median_ns_per_frame: f64,
    min_ns_per_frame: f64,
    max_ns_per_frame: f64,
    checksum: f32,
}

fn bench_variant<F>(mut runner: F) -> VariantResult
where
    F: FnMut() -> f32,
{
    for _ in 0..WARMUP_REPS {
        black_box(runner());
    }
    let mut walls_ns: [f64; TIMED_REPS] = [0.0; TIMED_REPS];
    let mut checksum = 0.0_f32;
    for slot in &mut walls_ns {
        let t0 = Instant::now();
        checksum = runner();
        *slot = t0.elapsed().as_secs_f64() * 1e9 / FRAMES as f64;
    }
    black_box(checksum);
    walls_ns.sort_by(|a, b| a.total_cmp(b));
    VariantResult {
        median_ns_per_frame: walls_ns[TIMED_REPS / 2],
        min_ns_per_frame: walls_ns[0],
        max_ns_per_frame: walls_ns[TIMED_REPS - 1],
        checksum,
    }
}

fn main() {
    let rule = std::iter::repeat_n('═', 100).collect::<String>();
    println!("{rule}");
    println!("Issue 800 Arm B1 — lock-free slot-flip PoC vs serial vs mpsc (katgpt-kv)");
    println!("Protocol: 1 AtomicUsize turn (seq-cst, monotone) + 2 flat slots; ticket gates");
    println!("  producer p: wait turn>=2p-1 → write slots[p&1] → bump (publish)");
    println!("  consumer c: wait turn>=2c+1 → read slots[c&1] → bump (release)");
    println!("Honesty: 2-thread measurement (serial = 1 thread); no core pinning;");
    println!(
        "  M3 Max, release profile; N={FRAMES} frames/rep; {WARMUP_REPS} warmup + {TIMED_REPS} timed reps; median."
    );
    println!("All variants share the same [AtomicU32] frame representation + kernels:");
    println!("  wall deltas are handoff/overlap only (see module docs for why).");
    println!("{rule}");

    let mut verdict_lines: Vec<String> = Vec::new();

    for &elems in &[ELEMS_PRIMARY, ELEMS_SECONDARY] {
        let kib = elems * 4 / 1024;
        for (cell, target_us) in [("short", SHORT_TARGET_US), ("long", LONG_TARGET_US)] {
            let (mac_passes, fill_passes) = calibrate(elems, target_us);

            let serial = bench_variant(|| run_serial(elems, mac_passes, fill_passes));
            let slot = bench_variant(|| run_slot_flip(elems, mac_passes, fill_passes));
            let mpsc_v = bench_variant(|| run_mpsc(elems, mac_passes, fill_passes));

            // Checksum tripwire: identical kernels + identical frame sequences
            // ⇒ bit-identical totals. A mismatch means a variant is not doing
            // the same work — report loud, never silently compare such walls.
            let checksums_agree =
                slot.checksum == serial.checksum && mpsc_v.checksum == serial.checksum;

            let ov = |r: &VariantResult| serial.median_ns_per_frame / r.median_ns_per_frame;
            let slot_vs_serial_pct = (serial.median_ns_per_frame - slot.median_ns_per_frame)
                * 100.0
                / serial.median_ns_per_frame;
            let slot_vs_mpsc_pct = (mpsc_v.median_ns_per_frame - slot.median_ns_per_frame) * 100.0
                / mpsc_v.median_ns_per_frame;

            println!();
            println!(
                "── frame {kib} KiB ({elems} f32) · compute cell: {cell} (target {target_us:.0} µs) ──"
            );
            println!(
                "  calibration: consumer mac passes={mac_passes}, producer fill passes={fill_passes} (both → ~{target_us:.0} µs)"
            );
            println!(
                "  {:<10} {:>12} {:>10} {:>22} {:>12}",
                "variant", "ns/frame", "overlap×", "min..max ns/frame", "checksum"
            );
            for (name, r) in [("serial", &serial), ("slot-flip", &slot), ("mpsc", &mpsc_v)] {
                println!(
                    "  {:<10} {:>12.0} {:>10.3} {:>22} {:>12.3}",
                    name,
                    r.median_ns_per_frame,
                    ov(r),
                    format!("{:.0}..{:.0}", r.min_ns_per_frame, r.max_ns_per_frame),
                    r.checksum
                );
            }
            if !checksums_agree {
                println!(
                    "  ⚠ CHECKSUM MISMATCH — variants did not do identical work; walls above are NOT comparable"
                );
            }
            println!(
                "  slot-flip vs serial: {slot_vs_serial_pct:+.1}%   slot-flip vs mpsc: {slot_vs_mpsc_pct:+.1}%"
            );

            let cell_verdict = if !checksums_agree {
                "INVALID (checksum mismatch)"
            } else if slot_vs_serial_pct > 10.0 && slot_vs_mpsc_pct > 10.0 {
                "PROMOTE-CANDIDATE (beats both baselines >10%)"
            } else if (-10.0..=10.0).contains(&slot_vs_mpsc_pct) {
                "TIE vs mpsc (within ±10%) → per B2 this is a DECLINE cell"
            } else if slot_vs_mpsc_pct < -10.0 {
                "LOSES to mpsc >10% → DECLINE cell"
            } else {
                "beats serial >10% but not mpsc → DECLINE cell"
            };
            println!("  B2 cell verdict: {cell_verdict}");
            verdict_lines.push(format!(
                "{kib:>4} KiB · {cell:<5} · serial {:>9.0} ns/f · slot {:>9.0} ns/f · mpsc {:>9.0} ns/f · slot-vs-serial {slot_vs_serial_pct:+6.1}% · slot-vs-mpsc {slot_vs_mpsc_pct:+6.1}% → {cell_verdict}",
                serial.median_ns_per_frame,
                slot.median_ns_per_frame,
                mpsc_v.median_ns_per_frame
            ));
        }
    }

    println!();
    println!("{rule}");
    println!("B2 DECISION GATE — per-cell results");
    for line in &verdict_lines {
        println!("  {line}");
    }
    println!();
    println!("Consumer-regime note (workspace grep, read-only):");
    println!("  DoubleBuffer/AsyncQdqScheduler consumers = async_qdq.rs itself + the");
    println!("  root GOAT test tests/async_qdq_goat.rs (required-features:");
    println!("  async_qdq_overlap+…). Its modeled regime: 64 KiB chunks, 50 µs GPU");
    println!("  attention + 30 µs CPU dequantize per chunk — the LONG-compute cell.");
    println!("  No consumer operates in the 1–10 µs short-compute regime.");
    println!("  At the async_qdq 20 Hz cadence a 50 ms tick makes even a multi-µs");
    println!("  channel handoff <0.01% of budget; overlap is the whole game there.");
    println!();
    // The binding cell is the long-compute one (the regime consumers run).
    let long_binding_promotes = verdict_lines
        .iter()
        .filter(|l| l.contains("· long ") && l.contains("PROMOTE-CANDIDATE"))
        .count();
    if long_binding_promotes > 0 {
        println!(
            "B2 VERDICT: long-compute cell (binding) PROMOTE-CANDIDATE — file the DoubleBuffer upgrade as the arm's landing."
        );
    } else {
        println!(
            "B2 VERDICT: DECLINE — slot-flip does not beat BOTH baselines >10% in the binding (long-compute) cell."
        );
        println!("  Channels tie/hold in the async_qdq regime. Pufferlib needs the spin");
        println!("  machine because C pthreads has no channels; Rust does. See the");
        println!("  per-cell lines above for the regime map (short-cell edge, if any).");
    }
    println!("{rule}");
}

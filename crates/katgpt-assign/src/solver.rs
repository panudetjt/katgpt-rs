//! Strict-improvement local search over the folded-objective DAG.
//!
//! Move vocabulary (Phase 1): `Single` (move one object to another
//! container) + `Swap` (exchange two objects' containers). Acceptance is
//! first-improvement with the plan's window `obj_δ ≤ 0`: negative deltas
//! always accept; zero deltas only while a sideways budget lasts (default
//! 0 = strict improvement — the deterministic default posture).
//!
//! **Hot-container ordering** (the paper's measured decisive win, Fig. 6 —
//! kept per Plan 620): source containers are scanned hottest-first, where
//! heat ≈ the paper's node potentials approximated leaf-side: violation
//! mass `max(0, util − limit)` plus balance deviation `|util − mean|`
//! (integer mean; the per-dim total is invariant under reassignment).
//! Leaf-side heat is linear-cost, exact, and deterministic — the full
//! node-potential machinery is the documented Phase 2 refinement.
//!
//! Determinism discipline (G1): move generation iterates dense arrays in
//! fixed orders (containers via the seeded tie permutation, objects by
//! index); NO HashMap is iterated anywhere in the solve path; the RNG
//! touches only the tie permutation at construction. Same input + same
//! seed ⇒ byte-identical `Assignment`.
//!
//! The evaluate/apply split: `eval_*` computes a candidate root value via
//! delta evaluation without mutating committed state; `apply_*` commits
//! the completed evaluation (overlay → values) plus the assignment and
//! membership indexes.

use std::cmp::Reverse;
use std::time::Instant;

use crate::expr::DeltaScratch;
use crate::rng::SplitMix64;
use crate::specs::{self, Compiled};
use crate::types::{
    Assignment, ContainerId, InvalidProblem, Limits, Problem, Seed, Solution, Spec, StopReason,
};

/// A candidate move (diagnostic vocabulary; the scans return these).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    Single { object: u32, dst: ContainerId },
    Swap { a: u32, b: u32 },
}

pub struct Solver<'p> {
    problem: &'p Problem,
    compiled: Compiled,
    /// Cached node values (leaf slots hold current utilizations).
    values: Vec<i64>,
    assignment: Vec<ContainerId>,
    /// Container membership (dense; swap-remove maintained).
    members: Vec<Vec<u32>>,
    obj_pos: Vec<u32>,
    /// Scratch (allocated once; the hot loop allocates nothing — G4).
    scratch: DeltaScratch,
    leaf_deltas: Vec<(u32, i64)>,
    member_snapshot: Vec<u32>,
    heat: Vec<i64>,
    heat_order: Vec<ContainerId>,
    /// Seeded container permutation — the tie-break under equal heat.
    tie_perm: Vec<u32>,
    /// Per referenced dim: integer mean utilization (heat input; the
    /// per-dim total is assignment-invariant so this is computed once).
    ref_dim_mean: Vec<i64>,
    limits: Limits,
    sideways_remaining: u64,
    deadline: Option<Instant>,
    deadline_hit: bool,
    evals: u64,
    accepted: u64,
    sweeps: u64,
    stopped_by: StopReason,
    /// Immediate-reversal guard for sideways moves: the inverse of the
    /// last accepted δ=0 move. Accepting it again would ping-pong;
    /// cleared on every strict (δ<0) acceptance.
    blocked_inverse: Option<Move>,
    /// Set when a deterministic limit trips inside a scan (the run loop
    /// converts it into the stop reason).
    limit_hit: Option<StopReason>,
}

impl<'p> Solver<'p> {
    /// Build a solver over a validated problem. Allocates all scratch
    /// here, never during `run` (G4).
    pub fn new(problem: &'p Problem, seed: Seed, limits: Limits) -> Result<Self, InvalidProblem> {
        problem.validate()?;
        let compiled = specs::compile(problem);
        let n = problem.num_objects;
        let b = problem.num_containers;

        // Integer mean per referenced dim (total demand / containers —
        // the total is invariant, so this never needs refreshing).
        let mut ref_dim_mean = Vec::with_capacity(compiled.referenced_dims.len());
        for &dim in &compiled.referenced_dims {
            let mut total: i128 = 0;
            for o in 0..n {
                total += problem.demands[o * problem.num_dims + dim] as i128;
            }
            ref_dim_mean.push((total / b as i128) as i64);
        }

        // Seeded tie permutation: identity shuffled once. Different seeds
        // explore equal-heat orders differently; the same seed always
        // produces the same permutation.
        let mut tie_perm: Vec<u32> = (0..b as u32).collect();
        let mut rng = SplitMix64::new(seed.0);
        rng.shuffle_u32(&mut tie_perm);

        let mut members: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut obj_pos = vec![0u32; n];
        for (o, &c) in problem.initial.iter().enumerate() {
            obj_pos[o] = members[c as usize].len() as u32;
            members[c as usize].push(o as u32);
        }

        let dag = &compiled.dag;
        let deadline = limits.time.map(|t| Instant::now() + t);
        let sideways_remaining = limits.sideways_budget;
        // Pre-size scan scratch to worst case so the hot loop never grows
        // a buffer (G4): the widest container can hold every object.
        let scratch = dag.scratch();
        let leaf_deltas = Vec::with_capacity(2 * compiled.referenced_dims.len() + 2);
        let member_snapshot = Vec::with_capacity(n);
        Ok(Self {
            problem,
            scratch,
            values: vec![0; dag.num_nodes()],
            compiled,
            assignment: problem.initial.clone(),
            members,
            obj_pos,
            leaf_deltas,
            member_snapshot,
            heat: vec![0; b],
            heat_order: Vec::with_capacity(b),
            tie_perm,
            ref_dim_mean,
            limits,
            sideways_remaining,
            deadline,
            deadline_hit: false,
            evals: 0,
            accepted: 0,
            sweeps: 0,
            stopped_by: StopReason::NoImprovingMove,
            blocked_inverse: None,
            limit_hit: None,
        })
    }

    // ── Diagnostic / test surface (also the Phase 2 `explain` seed) ──────

    /// Current assignment (borrowed).
    pub fn assignment(&self) -> &[ContainerId] {
        &self.assignment
    }

    /// Current folded objective.
    pub fn folded_objective(&self) -> i64 {
        self.values[self.compiled.dag.root as usize]
    }

    /// Current (violation_root, goal_sum) decomposition.
    pub fn objective_parts(&self) -> (i64, i64) {
        (
            self.values[self.compiled.violation_root as usize],
            self.values[self.compiled.goal_root as usize],
        )
    }

    /// Full recompute from the current assignment (delta-property oracle).
    pub fn recompute_all(&mut self) {
        self.seed_leaf_values();
    }

    /// Candidate root value for moving `object` to `dst` (no mutation).
    pub fn eval_single(&mut self, object: u32, dst: ContainerId) -> i64 {
        self.build_single_deltas(object, dst);
        self.compiled
            .dag
            .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        self.last_new_root()
    }

    /// Candidate root value for swapping `a` and `b` (no mutation).
    pub fn eval_swap(&mut self, a: u32, b: u32) -> i64 {
        self.build_swap_deltas(a, b);
        self.compiled
            .dag
            .eval_delta(&self.values, &self.leaf_deltas, &mut self.scratch);
        self.last_new_root()
    }

    /// Commit the last evaluated move (single or swap variant chosen by
    /// the caller — must match the last `eval_*` call).
    pub fn apply_single(&mut self, object: u32, dst: ContainerId) {
        self.compiled.dag.commit(&mut self.values, &self.scratch);
        let src = self.assignment[object as usize];
        debug_assert_ne!(src, dst);
        self.assignment[object as usize] = dst;
        // Membership: swap-remove from src, push to dst. `swap_remove`
        // returns the REMOVED element (== object); the element that was
        // LAST in src relocates into `pos` and needs its index fixed.
        let pos = self.obj_pos[object as usize] as usize;
        self.members[src as usize].swap_remove(pos);
        if let Some(&relocated) = self.members[src as usize].get(pos) {
            self.obj_pos[relocated as usize] = pos as u32;
        }
        self.obj_pos[object as usize] = self.members[dst as usize].len() as u32;
        self.members[dst as usize].push(object);
    }

    /// Commit a swap (must follow `eval_swap(a, b)`).
    pub fn apply_swap(&mut self, a: u32, b: u32) {
        self.compiled.dag.commit(&mut self.values, &self.scratch);
        let ca = self.assignment[a as usize];
        let cb = self.assignment[b as usize];
        self.assignment[a as usize] = cb;
        self.assignment[b as usize] = ca;
        // Membership: positions swap contents within their containers.
        let pa = self.obj_pos[a as usize] as usize;
        let pb = self.obj_pos[b as usize] as usize;
        self.members[ca as usize][pa] = b;
        self.members[cb as usize][pb] = a;
        self.obj_pos[a as usize] = pb as u32;
        self.obj_pos[b as usize] = pa as u32;
    }

    fn last_new_root(&self) -> i64 {
        self.scratch.last_new_root(self.compiled.dag.root)
    }

    // ── Delta construction ────────────────────────────────────────────────

    fn build_single_deltas(&mut self, object: u32, dst: ContainerId) {
        self.leaf_deltas.clear();
        let src = self.assignment[object as usize];
        debug_assert_ne!(src, dst, "single move to same container");
        let b = self.problem.num_containers;
        let odims = self.problem.num_dims;
        let o = object as usize;
        for (ref_idx, &dim) in self.compiled.referenced_dims.iter().enumerate() {
            let d = self.problem.demands[o * odims + dim];
            if d != 0 {
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * b + src as usize], -d));
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * b + dst as usize], d));
            }
        }
        if !self.compiled.moved_leaf.is_empty() {
            let init = self.problem.initial[o];
            let cur = (src != init) as i64;
            let new = (dst != init) as i64;
            if new != cur {
                self.leaf_deltas
                    .push((self.compiled.moved_leaf[o], new - cur));
            }
        }
    }

    fn build_swap_deltas(&mut self, a: u32, b: u32) {
        self.leaf_deltas.clear();
        let ca = self.assignment[a as usize];
        let cb = self.assignment[b as usize];
        debug_assert_ne!(ca, cb, "swap within one container is a no-op");
        let nb = self.problem.num_containers;
        let odims = self.problem.num_dims;
        let (oa, ob) = (a as usize, b as usize);
        for (ref_idx, &dim) in self.compiled.referenced_dims.iter().enumerate() {
            let da = self.problem.demands[oa * odims + dim];
            let db = self.problem.demands[ob * odims + dim];
            if da != db {
                // ca loses da, gains db; cb loses db, gains da.
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * nb + ca as usize], db - da));
                self.leaf_deltas
                    .push((self.compiled.util_leaf[ref_idx * nb + cb as usize], da - db));
            }
        }
        if !self.compiled.moved_leaf.is_empty() {
            let (ia, ib) = (self.problem.initial[oa], self.problem.initial[ob]);
            let cur_a = (ca != ia) as i64;
            let new_a = (cb != ia) as i64;
            if new_a != cur_a {
                self.leaf_deltas
                    .push((self.compiled.moved_leaf[oa], new_a - cur_a));
            }
            let cur_b = (cb != ib) as i64;
            let new_b = (ca != ib) as i64;
            if new_b != cur_b {
                self.leaf_deltas
                    .push((self.compiled.moved_leaf[ob], new_b - cur_b));
            }
        }
    }

    // ── Leaf seeding ──────────────────────────────────────────────────────

    fn seed_leaf_values(&mut self) {
        let b = self.problem.num_containers;
        let odims = self.problem.num_dims;
        // Zero util leaves, then accumulate object demands.
        for i in 0..self.compiled.util_leaf.len() {
            self.values[self.compiled.util_leaf[i] as usize] = 0;
        }
        for o in 0..self.problem.num_objects {
            let c = self.assignment[o] as usize;
            for (ref_idx, &dim) in self.compiled.referenced_dims.iter().enumerate() {
                let d = self.problem.demands[o * odims + dim];
                if d != 0 {
                    let leaf = self.compiled.util_leaf[ref_idx * b + c] as usize;
                    self.values[leaf] += d;
                }
            }
        }
        if !self.compiled.moved_leaf.is_empty() {
            for o in 0..self.problem.num_objects {
                let moved = (self.assignment[o] != self.problem.initial[o]) as i64;
                self.values[self.compiled.moved_leaf[o] as usize] = moved;
            }
        }
        self.compiled.dag.eval_full(&mut self.values);
    }

    // ── Heat + ordering ───────────────────────────────────────────────────

    fn refresh_heat(&mut self) {
        let b = self.problem.num_containers;
        for h in self.heat.iter_mut() {
            *h = 0;
        }
        for spec in &self.problem.specs {
            match *spec {
                Spec::Capacity { dim, limit, .. } => {
                    let ref_idx = self
                        .compiled
                        .referenced_dims
                        .binary_search(&dim)
                        .expect("validated: referenced dim");
                    for c in 0..b {
                        let u = self.values[self.compiled.util_leaf[ref_idx * b + c] as usize];
                        if u > limit {
                            self.heat[c] += u - limit;
                        }
                    }
                }
                Spec::Balance { dim, .. } => {
                    let ref_idx = self
                        .compiled
                        .referenced_dims
                        .binary_search(&dim)
                        .expect("validated: referenced dim");
                    let mean = self.ref_dim_mean[ref_idx];
                    for c in 0..b {
                        let u = self.values[self.compiled.util_leaf[ref_idx * b + c] as usize];
                        self.heat[c] += (u - mean).abs();
                    }
                }
                Spec::MinimizeMovement { .. } => {}
            }
        }
        // Hot-first, seeded permutation breaking ties — the unique key
        // makes the unstable sort fully deterministic.
        self.heat_order.clear();
        self.heat_order.extend(0..b as u32);
        let tie = &self.tie_perm;
        let heat = &self.heat;
        self.heat_order
            .sort_unstable_by_key(|&c| (Reverse(heat[c as usize]), tie[c as usize]));
    }

    // ── Scans (first improvement) ─────────────────────────────────────────

    fn trip_limit(&mut self, reason: StopReason) {
        if self.limit_hit.is_none() {
            self.limit_hit = Some(reason);
        }
    }

    fn budget_check(&mut self) -> bool {
        // True = keep scanning.
        if self.evals >= self.limits.max_evaluations {
            self.trip_limit(StopReason::EvalLimit);
            return false;
        }
        if self.accepted >= self.limits.max_accepted_moves {
            self.trip_limit(StopReason::MoveLimit);
            return false;
        }
        if self.deadline_hit {
            self.trip_limit(StopReason::TimeLimit);
            return false;
        }
        true
    }

    fn time_tick(&mut self) {
        if !self.deadline_hit
            && let Some(dl) = self.deadline
            && Instant::now() >= dl
        {
            self.deadline_hit = true;
        }
    }

    /// Sideways acceptance with the immediate-reversal guard: a δ=0
    /// candidate that undoes the last sideways move is skipped, and the
    /// move must DRAIN an imbalanced container (source heat > 0) —
    /// sideways shuffles between balanced containers only manufacture
    /// fresh imbalance (measured: the walk otherwise burns its whole
    /// budget drifting the overage around instead of closing it).
    fn accepts_sideways(&self, candidate: Move, src_heat: i64) -> bool {
        self.sideways_remaining > 0 && src_heat > 0 && self.blocked_inverse != Some(candidate)
    }

    /// Scan for the first acceptable `Single` move (hot containers first).
    fn scan_single(&mut self) -> Option<Move> {
        if !self.budget_check() {
            return None;
        }
        self.refresh_heat();
        let b = self.problem.num_containers;
        for si in 0..self.heat_order.len() {
            self.time_tick();
            if self.deadline_hit {
                return None;
            }
            let src = self.heat_order[si];
            let src_heat = self.heat[src as usize];
            self.member_snapshot.clear();
            self.member_snapshot
                .extend_from_slice(&self.members[src as usize]);
            for mi in 0..self.member_snapshot.len() {
                let o = self.member_snapshot[mi];
                for dst in 0..b as u32 {
                    if dst == src {
                        continue;
                    }
                    if self.evals.is_multiple_of(1024) {
                        self.time_tick();
                        if self.deadline_hit {
                            return None;
                        }
                        if !self.budget_check() {
                            return None;
                        }
                    }
                    self.evals += 1;
                    self.build_single_deltas(o, dst);
                    let new_root = self.compiled.dag.eval_delta(
                        &self.values,
                        &self.leaf_deltas,
                        &mut self.scratch,
                    );
                    let delta = new_root - self.folded_objective();
                    let candidate = Move::Single { object: o, dst };
                    let strict = delta < 0;
                    if strict || (delta == 0 && self.accepts_sideways(candidate, src_heat)) {
                        if delta == 0 {
                            self.sideways_remaining -= 1;
                            // Block the exact reversal (o moving back to
                            // `src`) — the ping-pong guard.
                            self.blocked_inverse = Some(Move::Single {
                                object: o,
                                dst: src,
                            });
                        } else {
                            self.blocked_inverse = None;
                        }
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }

    /// Scan for the first acceptable `Swap` move (hottest source
    /// containers only, capped by `max_swap_source_containers`).
    fn scan_swap(&mut self) -> Option<Move> {
        if !self.budget_check() {
            return None;
        }
        self.refresh_heat();
        let sources = self
            .limits
            .max_swap_source_containers
            .min(self.problem.num_containers);
        let n = self.problem.num_objects as u32;
        for si in 0..sources {
            self.time_tick();
            if self.deadline_hit {
                return None;
            }
            let src = self.heat_order[si];
            let src_heat = self.heat[src as usize];
            self.member_snapshot.clear();
            self.member_snapshot
                .extend_from_slice(&self.members[src as usize]);
            for mi in 0..self.member_snapshot.len() {
                let a = self.member_snapshot[mi];
                let ca = self.assignment[a as usize];
                for cand in 0..n {
                    if cand == a {
                        continue;
                    }
                    if self.assignment[cand as usize] == ca {
                        continue; // same container: no-op swap
                    }
                    if self.evals.is_multiple_of(1024) {
                        self.time_tick();
                        if self.deadline_hit {
                            return None;
                        }
                        if !self.budget_check() {
                            return None;
                        }
                    }
                    self.evals += 1;
                    self.build_swap_deltas(a, cand);
                    let new_root = self.compiled.dag.eval_delta(
                        &self.values,
                        &self.leaf_deltas,
                        &mut self.scratch,
                    );
                    let delta = new_root - self.folded_objective();
                    let candidate = Move::Swap { a, b: cand };
                    let strict = delta < 0;
                    if strict || (delta == 0 && self.accepts_sideways(candidate, src_heat)) {
                        if delta == 0 {
                            self.sideways_remaining -= 1;
                            // A swap is its own inverse.
                            self.blocked_inverse = Some(candidate);
                        } else {
                            self.blocked_inverse = None;
                        }
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }

    // ── Run loop ──────────────────────────────────────────────────────────

    /// Execute the search. Consumes the solver (the Solution owns the
    /// final assignment).
    pub fn run(mut self) -> Solution {
        self.seed_leaf_values();
        loop {
            if let Some(reason) = self.limit_hit {
                self.stopped_by = reason;
                break;
            }
            if self.sweeps >= self.limits.max_sweeps {
                self.stopped_by = StopReason::SweepLimit;
                break;
            }
            self.sweeps += 1;
            let mut found = false;
            while let Some(mv) = self.scan_single() {
                found = true;
                match mv {
                    Move::Single { object, dst } => self.apply_single(object, dst),
                    Move::Swap { .. } => unreachable!("scan_single yields singles"),
                }
                self.accepted += 1;
                if self.limit_hit.is_some() {
                    break;
                }
            }
            if self.limit_hit.is_some() {
                continue; // converted to a stop reason at loop head
            }
            while let Some(mv) = self.scan_swap() {
                found = true;
                match mv {
                    Move::Swap { a, b } => self.apply_swap(a, b),
                    Move::Single { .. } => unreachable!("scan_swap yields swaps"),
                }
                self.accepted += 1;
                if self.limit_hit.is_some() {
                    break;
                }
            }
            if self.limit_hit.is_some() {
                continue;
            }
            if !found {
                self.stopped_by = StopReason::NoImprovingMove;
                break;
            }
        }
        let (violation_root, goal_sum) = self.objective_parts();
        Solution {
            folded_objective: self.folded_objective(),
            violation_root,
            goal_sum,
            assignment: Assignment {
                containers: self.assignment,
            },
            moves_evaluated: self.evals,
            moves_accepted: self.accepted,
            sweeps: self.sweeps,
            stopped_by: self.stopped_by,
        }
    }
}

/// One-call API.
pub fn solve(problem: &Problem, seed: Seed, limits: &Limits) -> Result<Solution, InvalidProblem> {
    Ok(Solver::new(problem, seed, limits.clone())?.run())
}

/// Folded-objective oracle: compile + evaluate an arbitrary assignment
/// (full recompute, no search). Returns `(folded, violation_root,
/// goal_sum)`. The G1/G2 tests' ground truth and the brute-force
/// comparator.
pub fn eval_assignment(
    problem: &Problem,
    assignment: &[ContainerId],
) -> Result<(i64, i64, i64), InvalidProblem> {
    problem.validate()?;
    if assignment.len() != problem.num_objects {
        return Err(InvalidProblem::InitialLen {
            expected: problem.num_objects,
            got: assignment.len(),
        });
    }
    let compiled = specs::compile(problem);
    let mut values = vec![0i64; compiled.dag.num_nodes()];
    let b = problem.num_containers;
    let odims = problem.num_dims;
    for (o, &container) in assignment.iter().enumerate() {
        let c = container as usize;
        for (ref_idx, &dim) in compiled.referenced_dims.iter().enumerate() {
            let d = problem.demands[o * odims + dim];
            if d != 0 {
                values[compiled.util_leaf[ref_idx * b + c] as usize] += d;
            }
        }
    }
    if !compiled.moved_leaf.is_empty() {
        for (o, &container) in assignment.iter().enumerate() {
            values[compiled.moved_leaf[o] as usize] = (container != problem.initial[o]) as i64;
        }
    }
    compiled.dag.eval_full(&mut values);
    Ok((
        values[compiled.dag.root as usize],
        values[compiled.violation_root as usize],
        values[compiled.goal_root as usize],
    ))
}

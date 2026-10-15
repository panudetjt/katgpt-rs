//! # katgpt-assign
//!
//! Spec-driven **constrained assignment solver** — place every object into
//! exactly one container such that constraint specs hold and goal specs are
//! optimized. Distilled from Meta's Rebalancer (OSDI'24; Research 607,
//! Plan 620) into a modelless, integer-deterministic, zero-dependency
//! primitive.
//!
//! - **Modeling**: a small closed spec vocabulary (`Capacity`, `Balance`,
//!   `MinimizeMovement`) over dense `u32` object/container indices and
//!   `i64` dimension values, compiled into ONE expression DAG of linear
//!   size (`Lookup` leaves shared per `(dim, container)` + `Sum`/`Max`/
//!   `Affine` internal nodes, per-node cached values).
//! - **Search**: strict-improvement local search (single + swap moves,
//!   first-improvement, hot-container ordering from leaf-side potentials,
//!   deterministic seeded tie-breaks, move/eval/sweep limits).
//! - **Delta evaluation**: a candidate move touches a few leaves;
//!   recomputation runs bottom-up over reached nodes only. Integer sums
//!   are exact — no segment tree, no drift (Research 607 §1).
//! - **Integer determinism (G1)**: no floats anywhere in the objective
//!   path; same input + same seed ⇒ **byte-identical `Assignment` on
//!   every node** — the raw-domain requirement for anything crossing
//!   `SyncBlock → ChainConsensus` (the primary consumer, riir-chain Issue
//!   164, does exactly that).
//!
//! Paper defaults kept: broken-constraint fix-it weight 100, never-worse
//! guard 10000 (see [`types`]). Simulated annealing is a recorded negative
//! in-source and deliberately absent.
//!
//! ## Quick start
//!
//! ```
//! use katgpt_assign::fixtures;
//! use katgpt_assign::{solve, Limits, Seed};
//!
//! let problem = fixtures::perfect_balance(7, 4, 25, 6);
//! let solution = solve(&problem, Seed(7), &Limits::default()).unwrap();
//! assert_eq!(solution.violation_root, 0); // feasible
//! ```
//!
//! ## Feature posture
//!
//! `katgpt-core` re-exports this crate as `katgpt_core::assign` behind the
//! opt-in `assignment` feature (the `katgpt-dec` → `katgpt_core::dec`
//! precedent). The feature stays opt-in until the GOAT gate
//! (G1+G2 vs outside baselines) passes AND a real consumer wires it
//! (riir-chain Issue 164 minimum) — see Plan 620's verdict protocol.

#![deny(rustdoc::broken_intra_doc_links)]

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
pub mod alloc;
pub mod expr;
pub mod fixtures;
pub mod rng;
pub mod solver;
mod specs;
pub mod types;

pub use solver::{Move, Solver, eval_assignment, solve};
pub use types::{
    Assignment, ContainerId, FIX_IT_WEIGHT, InvalidProblem, Limits, NEVER_WORSE_GUARD, Problem,
    Seed, Solution, Spec, StopReason,
};

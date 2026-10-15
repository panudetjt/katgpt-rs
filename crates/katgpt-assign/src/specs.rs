//! Spec → DAG compilation (the "MIP-free modeling layer" of Research 607).
//!
//! Compiles the Phase 1 spec vocabulary into ONE expression DAG whose root
//! is the folded objective:
//!
//! ```text
//! root = Sum[ Affine(NEVER_WORSE_GUARD)(violation_root), goal_root ]
//! violation_root = Max over (capacity spec, container) of
//!                  w_row · max(0, util(dim, b) − limit)
//! goal_root = Sum of Balance spread terms + MinimizeMovement terms
//! ```
//!
//! Leaves are shared per `(referenced dim, container)` — two specs on the
//! same dimension read the same leaf node, so a move's delta evaluation
//! touches each affected utilization exactly once regardless of how many
//! specs observe it.
//!
//! Construction is deterministic by enumeration order (referenced dims
//! sorted ascending, containers ascending, specs in declaration order) —
//! no HashMap anywhere, and the Max/Sum value semantics are order-free
//! (exact integer addition is associative and commutative), so a permuted
//! spec list compiles to a differently-shaped but value-identical DAG.
//! That invariance is pinned by the G1 determinism test.

use crate::expr::{Dag, NodeId};
use crate::types::{NEVER_WORSE_GUARD, Problem, Spec};

/// The compiled problem: the DAG plus the solver's leaf addressing.
#[derive(Debug, Clone)]
pub struct Compiled {
    pub dag: Dag,
    /// Referenced dimension indices, sorted ascending.
    pub referenced_dims: Vec<usize>,
    /// `util_leaf[ref_dim_idx * num_containers + b]` → leaf NodeId.
    pub util_leaf: Vec<NodeId>,
    /// `moved_leaf[o]` → leaf NodeId (EMPTY when no movement spec — the
    /// leaf family is only allocated when something reads it).
    pub moved_leaf: Vec<NodeId>,
    /// Max-folded hard-constraint violation subroot (a `Const(0)` node
    /// when there are no capacity specs).
    pub violation_root: NodeId,
    /// Goal sum subroot (a `Const(0)` node when there are no goal specs).
    pub goal_root: NodeId,
}

/// Which dims does any spec read? Sorted ascending, deduped.
fn referenced_dims(problem: &Problem) -> Vec<usize> {
    let mut dims: Vec<usize> = problem
        .specs
        .iter()
        .filter_map(|s| match *s {
            Spec::Capacity { dim, .. } | Spec::Balance { dim, .. } => Some(dim),
            Spec::MinimizeMovement { .. } => None,
        })
        .collect();
    dims.sort_unstable();
    dims.dedup();
    dims
}

/// Compile a validated problem into the folded-objective DAG.
pub fn compile(problem: &Problem) -> Compiled {
    let b = problem.num_containers;
    let dims = referenced_dims(problem);
    let mut dag = Dag::builder();

    // ── Utilization leaves: one per (referenced dim, container), shared.
    // Pushed first (lowest node ids) in a fixed dense order.
    let mut util_leaf = Vec::with_capacity(dims.len() * b);
    for &dim in &dims {
        let slot = (dim * b) as u32;
        for container in 0..b {
            let leaf = dag.push_leaf_util(slot + container as u32);
            util_leaf.push(leaf);
        }
    }

    // ── Moved leaves: only when a movement spec reads them.
    let wants_movement = problem
        .specs
        .iter()
        .any(|s| matches!(s, Spec::MinimizeMovement { .. }));
    let mut moved_leaf = Vec::new();
    if wants_movement {
        for o in 0..problem.num_objects {
            moved_leaf.push(dag.push_leaf_moved(o as u32));
        }
    }

    // ── Capacity rows → Max-folded violation root.
    // Per (spec, container): w · max(0, util − limit) — expressed as
    // Max[ Affine(1, −limit)(leaf), Const(0) ], then Affine(w, 0).
    let mut rows: Vec<NodeId> = Vec::new();
    let zero = dag.push_const(0);
    for spec in &problem.specs {
        if let Spec::Capacity {
            dim,
            limit,
            violation_weight,
        } = *spec
        {
            let ref_idx = dims.binary_search(&dim).expect("dim is referenced");
            let base = ref_idx * b;
            for container in 0..b {
                let leaf = util_leaf[base + container];
                let shifted = dag.push_affine(1, -limit, leaf);
                let floored = dag.push_max(&[shifted, zero]);
                let row = if violation_weight == 1 {
                    floored
                } else {
                    dag.push_affine(violation_weight, 0, floored)
                };
                rows.push(row);
            }
        }
    }
    let violation_root = if rows.is_empty() {
        dag.push_const(0)
    } else {
        dag.push_max(&rows)
    };
    // Secondary repair pressure: the SUM of all violation rows rides the
    // goal side at weight 1. When feasible it is identically zero (no
    // effect on goal semantics); when infeasible it makes plateau descent
    // MONOTONE — the Max fold alone cannot distinguish which of two
    // tied-at-worst containers to drain (measured: perfect_balance stalls
    // at a two-way tie with strict-only acceptance). Every accepted move
    // strictly decreases the sum, so cycles are impossible.
    let sum_violation_term = if rows.is_empty() {
        None
    } else {
        Some(dag.push_sum(&rows))
    };

    // ── Goal terms.
    let mut terms: Vec<NodeId> = Vec::new();
    for spec in &problem.specs {
        match *spec {
            Spec::Capacity { .. } => {}
            Spec::Balance { dim, weight } => {
                // spread = max_b util − min_b util
                //        = Max[leaves] + Max[Affine(−1)(leaves)]
                let ref_idx = dims.binary_search(&dim).expect("dim is referenced");
                let base = ref_idx * b;
                let leaves: Vec<NodeId> = (0..b).map(|c| util_leaf[base + c]).collect();
                let max_util = dag.push_max(&leaves);
                let neg_leaves: Vec<NodeId> =
                    leaves.iter().map(|&l| dag.push_affine(-1, 0, l)).collect();
                let neg_min = dag.push_max(&neg_leaves); // = −min(util)
                let spread = dag.push_sum(&[max_util, neg_min]);
                let term = if weight == 1 {
                    spread
                } else {
                    dag.push_affine(weight, 0, spread)
                };
                terms.push(term);
            }
            Spec::MinimizeMovement { weight } => {
                let term = if weight == 1 {
                    dag.push_sum(&moved_leaf)
                } else {
                    let total = dag.push_sum(&moved_leaf);
                    dag.push_affine(weight, 0, total)
                };
                terms.push(term);
            }
        }
    }
    // The secondary repair term joins the goal sum (it is identically 0
    // at feasibility, so it never distorts goal semantics).
    if let Some(term) = sum_violation_term {
        terms.push(term);
    }
    let goal_root = if terms.is_empty() {
        dag.push_const(0)
    } else {
        dag.push_sum(&terms)
    };

    // ── Folded root: GUARD · violation + goal.
    let guarded = dag.push_affine(NEVER_WORSE_GUARD, 0, violation_root);
    let root = dag.push_sum(&[guarded, goal_root]);

    let dag = dag.finish(root);
    Compiled {
        dag,
        referenced_dims: dims,
        util_leaf,
        moved_leaf,
        violation_root,
        goal_root,
    }
}

//! Decoupled data types (the repo `types.rs` convention).
//!
//! Everything here is plain data — no solver machinery. Integer
//! determinism notes: every quantity in the objective path is `i64`.
//! There are no floats anywhere in the folded objective, tie-breaks, or
//! move generation, so same input + same seed ⇒ byte-identical
//! `Assignment` on every node (the G1 gate; see `expr.rs` for the DAG
//! that makes delta evaluation exact).
//!
//! `Group` (grouping objects for `GroupCount`-style specs) is **reserved
//! for Phase 2** per Plan 620 and deliberately absent here.

use std::time::Duration;

/// Object index (dense, `0..num_objects`).
pub type ObjectId = u32;
/// Container index (dense, `0..num_containers`).
pub type ContainerId = u32;

/// Paper default: weight of a broken hard constraint's violation row (the
/// "fix-it goal" a broken constraint spawns — Rebalancer's broken-constraint
/// fallback, Plan 620 Phase 1).
pub const FIX_IT_WEIGHT: i64 = 100;

/// Paper default: the never-worse guard. The Max-folded constraint-violation
/// root is scaled by this in the folded objective, so while ANY hard row is
/// broken, constraint repair dominates every goal term whose weighted sum
/// stays below the guard scale (documented lexicographic semantics — raise
/// the guard if your goal weights exceed it).
pub const NEVER_WORSE_GUARD: i64 = 10_000;

/// One assignment problem: place every object into exactly one container
/// such that `specs` hold / are optimized.
///
/// Layout conventions (dense, deterministic):
/// - `demands[o * num_dims + d]` — integer demand of object `o` in
///   dimension `d`. Must be `>= 0` (validated).
/// - `initial[o]` — the starting container (the reference point for
///   `MinimizeMovement` and the state the solver improves from).
#[derive(Debug, Clone)]
pub struct Problem {
    pub num_objects: usize,
    pub num_containers: usize,
    pub num_dims: usize,
    pub demands: Vec<i64>,
    pub initial: Vec<ContainerId>,
    pub specs: Vec<Spec>,
}

/// A constraint-or-goal spec (the Phase 1 closed vocabulary; ~85% of Meta's
/// production constraints reused existing specs — the API design bet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spec {
    /// HARD: per-container utilization in `dim` must stay `<= limit`.
    /// Violation rows fold into the Max violation root (fix-it behavior).
    /// `violation_weight` defaults to [`FIX_IT_WEIGHT`].
    Capacity {
        dim: usize,
        limit: i64,
        violation_weight: i64,
    },
    /// GOAL: minimize utilization spread (`max_util - min_util`) across all
    /// containers in `dim`, weighted by `weight`.
    Balance { dim: usize, weight: i64 },
    /// GOAL: penalize each object whose container differs from its initial
    /// one, weighted by `weight`.
    MinimizeMovement { weight: i64 },
}

impl Spec {
    /// [`Spec::Capacity`] with the paper-default fix-it weight.
    pub const fn capacity(dim: usize, limit: i64) -> Self {
        Self::Capacity {
            dim,
            limit,
            violation_weight: FIX_IT_WEIGHT,
        }
    }
}

/// Run limits. The deterministic knobs (`max_accepted_moves`,
/// `max_evaluations`, `max_sweeps`) bound work reproducibly. The `time`
/// budget is **advisory only and breaks cross-machine byte-determinism**
/// (a faster machine squeezes more moves into the same wall-clock window);
/// it exists for interactive latency bounds, never for replayable results —
/// the G1 determinism gate runs with `time: None`.
#[derive(Debug, Clone)]
pub struct Limits {
    pub max_accepted_moves: u64,
    pub max_evaluations: u64,
    pub max_sweeps: u64,
    /// Swap-move generation iterates objects in the hottest source
    /// containers first; this caps how many source containers a swap pass
    /// scans (bounds the O(N²) swap neighbourhood; the paper's hot-bin
    /// ordering makes the cap nearly free in quality).
    pub max_swap_source_containers: usize,
    /// Extra budget for zero-delta (sideways) moves — plateau walks.
    /// **Non-zero by default** (measured): the Max-folded violation root
    /// makes δ=0 moves the NORM whenever two containers share the worst
    /// violation — fixing one leaves the Max unchanged — so strict-only
    /// acceptance stalls on tie plateaus (perfect_balance fixture:
    /// violation 10400 frozen). An immediate-reversal guard prevents the
    /// trivial A→B→A cycle; the budget bounds everything else.
    pub sideways_budget: u64,
    /// Advisory wall-clock budget (`None` = unbounded, the deterministic
    /// default). Checked between sweeps and every 1024 evaluations.
    pub time: Option<Duration>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_accepted_moves: 100_000,
            max_evaluations: 10_000_000,
            max_sweeps: 1_000,
            max_swap_source_containers: 256,
            sideways_budget: 25_000,
            time: None,
        }
    }
}

/// Deterministic seed — enters tie-break permutations only (never the
/// objective). Same seed ⇒ same permutations ⇒ byte-identical result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seed(pub u64);

/// Dense object→container assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub containers: Vec<ContainerId>,
}

impl Assignment {
    /// Canonical byte encoding (little-endian u32 per object) — the G1
    /// byte-identity comparator.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.containers.len() * 4);
        for c in &self.containers {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }
}

/// Why the search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// A full single+swap round found no acceptable move.
    NoImprovingMove,
    MoveLimit,
    EvalLimit,
    SweepLimit,
    /// Advisory wall-clock budget hit (non-deterministic posture only).
    TimeLimit,
}

/// Solver output: the final assignment plus the folded-objective
/// decomposition and search statistics.
#[derive(Debug, Clone)]
pub struct Solution {
    pub assignment: Assignment,
    /// `GUARD * violation_root + goal_sum` — the folded objective the
    /// search minimizes (never increases; the never-worse guard).
    pub folded_objective: i64,
    /// Max-folded hard-constraint violation (0 = feasible).
    pub violation_root: i64,
    pub goal_sum: i64,
    pub moves_evaluated: u64,
    pub moves_accepted: u64,
    pub sweeps: u64,
    pub stopped_by: StopReason,
}

/// Validation error for a malformed [`Problem`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidProblem {
    ZeroContainers,
    DemandsLen {
        expected: usize,
        got: usize,
    },
    InitialLen {
        expected: usize,
        got: usize,
    },
    InitialOutOfRange {
        object: usize,
        container: usize,
    },
    NegativeDemand {
        index: usize,
    },
    NegativeWeight {
        spec: usize,
    },
    DimOutOfRange {
        spec: usize,
        dim: usize,
        num_dims: usize,
    },
    OverflowRisk,
}

impl core::fmt::Display for InvalidProblem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroContainers => write!(f, "num_containers must be >= 1"),
            Self::DemandsLen { expected, got } => {
                write!(
                    f,
                    "demands.len() = {got}, expected {expected} (num_objects * num_dims)"
                )
            }
            Self::InitialLen { expected, got } => {
                write!(
                    f,
                    "initial.len() = {got}, expected {expected} (num_objects)"
                )
            }
            Self::InitialOutOfRange { object, container } => {
                write!(f, "initial[{object}] = {container} out of range")
            }
            Self::NegativeDemand { index } => write!(f, "demands[{index}] < 0"),
            Self::NegativeWeight { spec } => write!(f, "specs[{spec}] has a negative weight"),
            Self::DimOutOfRange {
                spec,
                dim,
                num_dims,
            } => {
                write!(
                    f,
                    "specs[{spec}] references dim {dim}, num_dims = {num_dims}"
                )
            }
            Self::OverflowRisk => write!(
                f,
                "input magnitudes risk i64 overflow: keep |demands| and limits <= 1e15, \
                 weights <= 1e6, and per-dim demand sums <= 1e17"
            ),
        }
    }
}

impl std::error::Error for InvalidProblem {}

impl Problem {
    /// Structural + magnitude validation. Overflow discipline: the DAG
    /// evaluates in i64; this checks the input contract that keeps every
    /// intermediate within range for realistic scales (a debug-build
    /// arithmetic overflow panic is the backstop — debug builds have
    /// overflow checks on).
    pub fn validate(&self) -> Result<(), InvalidProblem> {
        if self.num_containers == 0 {
            return Err(InvalidProblem::ZeroContainers);
        }
        let expected = self
            .num_objects
            .checked_mul(self.num_dims)
            .ok_or(InvalidProblem::OverflowRisk)?;
        if self.demands.len() != expected {
            return Err(InvalidProblem::DemandsLen {
                expected,
                got: self.demands.len(),
            });
        }
        if self.initial.len() != self.num_objects {
            return Err(InvalidProblem::InitialLen {
                expected: self.num_objects,
                got: self.initial.len(),
            });
        }
        for (o, &c) in self.initial.iter().enumerate() {
            if c as usize >= self.num_containers {
                return Err(InvalidProblem::InitialOutOfRange {
                    object: o,
                    container: c as usize,
                });
            }
        }
        for (i, &d) in self.demands.iter().enumerate() {
            if d < 0 {
                return Err(InvalidProblem::NegativeDemand { index: i });
            }
            if d > 1_000_000_000_000_000 {
                return Err(InvalidProblem::OverflowRisk);
            }
        }
        for (s, spec) in self.specs.iter().enumerate() {
            match *spec {
                Spec::Capacity {
                    dim,
                    limit,
                    violation_weight,
                } => {
                    if dim >= self.num_dims {
                        return Err(InvalidProblem::DimOutOfRange {
                            spec: s,
                            dim,
                            num_dims: self.num_dims,
                        });
                    }
                    if violation_weight < 0 || limit > 1_000_000_000_000_000 {
                        return Err(InvalidProblem::NegativeWeight { spec: s });
                    }
                }
                Spec::Balance { dim, weight } => {
                    if dim >= self.num_dims {
                        return Err(InvalidProblem::DimOutOfRange {
                            spec: s,
                            dim,
                            num_dims: self.num_dims,
                        });
                    }
                    if !(0..=1_000_000).contains(&weight) {
                        return Err(InvalidProblem::NegativeWeight { spec: s });
                    }
                }
                Spec::MinimizeMovement { weight } => {
                    if !(0..=1_000_000).contains(&weight) {
                        return Err(InvalidProblem::NegativeWeight { spec: s });
                    }
                }
            }
        }
        // Per-dim demand sum bound: utilization sums, spread, and the
        // GUARD-scaled folded objective must all stay well inside i64.
        for d in 0..self.num_dims {
            let mut total: i128 = 0;
            for o in 0..self.num_objects {
                total += self.demands[o * self.num_dims + d] as i128;
            }
            if total > 100_000_000_000_000_000 {
                return Err(InvalidProblem::OverflowRisk);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_bin_problem() -> Problem {
        Problem {
            num_objects: 2,
            num_containers: 2,
            num_dims: 1,
            demands: vec![1, 1],
            initial: vec![0, 0],
            specs: vec![Spec::capacity(0, 1), Spec::Balance { dim: 0, weight: 1 }],
        }
    }

    #[test]
    fn validates_clean_problem() {
        assert!(two_bin_problem().validate().is_ok());
    }

    #[test]
    fn rejects_wrong_lengths() {
        let mut p = two_bin_problem();
        p.demands = vec![1];
        assert_eq!(
            p.validate(),
            Err(InvalidProblem::DemandsLen {
                expected: 2,
                got: 1
            })
        );
        let mut p = two_bin_problem();
        p.initial = vec![0];
        assert_eq!(
            p.validate(),
            Err(InvalidProblem::InitialLen {
                expected: 2,
                got: 1
            })
        );
    }

    #[test]
    fn rejects_out_of_range_initial_and_negative_demand() {
        let mut p = two_bin_problem();
        p.initial = vec![0, 5];
        assert_eq!(
            p.validate(),
            Err(InvalidProblem::InitialOutOfRange {
                object: 1,
                container: 5
            })
        );
        let mut p = two_bin_problem();
        p.demands = vec![1, -1];
        assert_eq!(
            p.validate(),
            Err(InvalidProblem::NegativeDemand { index: 1 })
        );
    }

    #[test]
    fn rejects_bad_dim_and_weight() {
        let mut p = two_bin_problem();
        p.specs.push(Spec::Balance { dim: 3, weight: 1 });
        assert_eq!(
            p.validate(),
            Err(InvalidProblem::DimOutOfRange {
                spec: 2,
                dim: 3,
                num_dims: 1
            })
        );
        let mut p = two_bin_problem();
        p.specs.push(Spec::MinimizeMovement { weight: -5 });
        assert_eq!(
            p.validate(),
            Err(InvalidProblem::NegativeWeight { spec: 2 })
        );
    }

    #[test]
    fn assignment_bytes_are_canonical() {
        let a = Assignment {
            containers: vec![1, 0, 2],
        };
        assert_eq!(a.to_bytes(), vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]);
    }
}

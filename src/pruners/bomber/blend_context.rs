//! Shared blend context — board-state feature extraction for HLPlayer's
//! blend estimators.
//!
//! Extracted from `contextual_bandit.rs` (Plan 436) so that all blend
//! estimator modules (linear contextual bandit, binned, kernel) share the
//! same context computation without duplication.
//!
//! ## Context vector (d = 7)
//!
//! 1. `in_blast_zone` (0.0 or 1.0) — current danger.
//! 2. `blast_proximity` (sigmoid-normalized Manhattan distance to nearest bomb).
//! 3. `opponent_pressure` (sigmoid-normalized inverse distance to nearest opponent).
//! 4. `wall_density_3x3` (wall count / 9) — movement constraint.
//! 5. `powerup_proximity` (sigmoid-normalized inverse distance to nearest powerup).
//! 6. `bomb_pressure` (sigmoid-normalized active bomb count).
//! 7. `bias` (1.0) — unconditional offset per arm.

use super::players::{KnownBomb, in_blast_zone};
use super::{ArenaGrid, Cell, GridPos};

/// Context dimension (6 features + 1 bias).
pub const CONTEXT_DIM: usize = 7;

/// Index of `blast_proximity` in the context vector.
pub const BLAST_PROXIMITY_IDX: usize = 1;

/// Index of `opponent_pressure` in the context vector.
pub const OPPONENT_PRESSURE_IDX: usize = 2;

/// Compute the board-state context vector `φ(s)` for the current tick.
///
/// All features are bounded to `[0, 1]` (binary or sigmoid-normalized) so the
/// estimator weight magnitudes stay interpretable and the updates are
/// numerically stable.
///
/// This is a pure function of the observed board state — no allocation, no
/// side effects, deterministic given the inputs.
#[allow(clippy::too_many_arguments)]
pub fn compute_phi(
    pos: GridPos,
    grid: &ArenaGrid,
    bombs: &[KnownBomb],
    powerups: &[(i32, i32)],
    nearest_opponent: Option<(i32, i32)>,
) -> [f32; CONTEXT_DIM] {
    // 1. in_blast_zone (0.0 or 1.0)
    let in_blast = if in_blast_zone(pos, grid, bombs) {
        1.0
    } else {
        0.0
    };

    // 2. blast_proximity — sigmoid of -(Manhattan dist to nearest bomb - 3).
    let min_bomb_dist = bombs
        .iter()
        .map(|&(bp, _, _)| (pos.x - bp.0).abs() + (pos.y - bp.1).abs())
        .min()
        .unwrap_or(20) as f32;
    let blast_proximity = sigmoid(-(min_bomb_dist - 3.0));

    // 3. opponent_pressure — sigmoid of -(Manhattan dist to nearest opponent - 5).
    let opp_dist =
        nearest_opponent.map_or(30, |(ox, oy)| (pos.x - ox).abs() + (pos.y - oy).abs()) as f32;
    let opponent_pressure = sigmoid(-(opp_dist - 5.0));

    // 4. wall_density_3x3 — count of Fixed/Destructible walls / 9.
    let wall_count = (-1..=1i32)
        .flat_map(|dx| (-1..=1i32).map(move |dy| (dx, dy)))
        .filter(|&(dx, dy)| {
            matches!(
                grid.get(pos.x + dx, pos.y + dy),
                Cell::FixedWall | Cell::DestructibleWall
            )
        })
        .count() as f32;
    let wall_density = wall_count / 9.0;

    // 5. powerup_proximity — sigmoid of -(dist to nearest known powerup - 3).
    let min_powerup_dist = powerups
        .iter()
        .map(|&(px, py)| (pos.x - px).abs() + (pos.y - py).abs())
        .min()
        .unwrap_or(30) as f32;
    let powerup_proximity = sigmoid(-(min_powerup_dist - 3.0));

    // 6. bomb_pressure — sigmoid of (active bomb count - 2).
    let bomb_pressure = sigmoid(bombs.len() as f32 - 2.0);

    // 7. bias = 1.0
    [
        in_blast,
        blast_proximity,
        opponent_pressure,
        wall_density,
        powerup_proximity,
        bomb_pressure,
        1.0,
    ]
}

/// Numerically stable sigmoid: `1 / (1 + e^{-x})`.
///
/// Per the global rule: use sigmoid (not softmax) for any probabilistic gating
/// or bounded `[0,1]` projection.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

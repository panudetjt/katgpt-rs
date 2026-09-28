//! [`embedded_best_response`] + [`PayoffMatrix`] — the cooperation gate.
//!
//! See the [module-level docs](crate::similarity_inference) for the
//! mathematical background. This module implements the discrete-action
//! best-response comparator that switches from competitive (Nash) to
//! cooperative (CCE) regime based on the similarity posterior `ω`.

use crate::similarity_inference::SimilarityError;

/// A row-player normal-form payoff matrix for a symmetric 2-player game with
/// `A` actions per player. `payoff[i][j]` is the row player's payoff when the
/// row player plays action `i` and the column player plays action `j`.
///
/// Stored as a flat `Vec<f32>` of length `A·A` for cache-friendly scans.
/// Construct via [`PayoffMatrix::from_row_major`] or
/// [`PayoffMatrix::new`].
///
/// # Why not reuse `riir-games-shared::PayoffTable<N>`?
///
/// The existing `PayoffTable<N>` (riir-ai/crates/riir-games-shared/src/payoff/)
/// is **combat-specific**: it stores `UnitSpec` per row (armor class, cost, hp,
/// dps) and computes payoffs from combat uptime assumptions. It's the wrong
/// shape for abstract game-theoretic matrices (Prisoner's Dilemma, Battle of
/// the Sexes, Chicken, Stag Hunt) where payoffs are arbitrary constants. Per
/// the substrate-first skill: substrate exists but is wrong shape → define a
/// domain-appropriate type in the right place. This leaf type lives here
/// because `similarity_inference` is the only consumer of abstract normal-form
/// payoffs in katgpt-core.
///
/// # Synchronization boundary
///
/// The payoff matrix is **game-design data** (cold tier, designer-authored).
/// It crosses the sync boundary as raw f32 values (deterministic replay needs
/// bit-identical payoffs). It is NOT a latent/semantic quantity.
#[derive(Clone, Debug)]
pub struct PayoffMatrix {
    /// Row-major `A × A` payoff table. `payoff[i * A + j]` = row player's
    /// payoff when row plays `i` and column plays `j`.
    payoff: Vec<f32>,
    /// Number of actions per player (the matrix is `n_actions × n_actions`).
    n_actions: usize,
}

impl PayoffMatrix {
    /// Construct from a flat row-major buffer of length `A · A`.
    ///
    /// Returns `Err(PayoffShapeMismatch)` if `values.len() != n_actions²`.
    pub fn from_row_major(n_actions: usize, values: Vec<f32>) -> Result<Self, SimilarityError> {
        if n_actions == 0 {
            return Err(SimilarityError::EmptyActionSet);
        }
        let expected =
            n_actions
                .checked_mul(n_actions)
                .ok_or(SimilarityError::PayoffShapeMismatch {
                    expected: 0,
                    got: values.len(),
                })?;
        if values.len() != expected {
            return Err(SimilarityError::PayoffShapeMismatch {
                expected,
                got: values.len(),
            });
        }
        Ok(Self {
            payoff: values,
            n_actions,
        })
    }

    /// Construct from a 2D array (compile-time-known action count).
    pub fn new<const A: usize>(payoff: [[f32; A]; A]) -> Result<Self, SimilarityError> {
        if A == 0 {
            return Err(SimilarityError::EmptyActionSet);
        }
        let mut flat = Vec::with_capacity(A * A);
        for row in payoff {
            flat.extend_from_slice(&row);
        }
        Self::from_row_major(A, flat)
    }

    /// Number of actions per player.
    #[inline]
    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Row player's payoff when row plays `i` and column plays `j`.
    ///
    /// # Panics
    ///
    /// Panics if `i` or `j` is `>= n_actions` (caller bug).
    #[inline]
    pub fn payoff(&self, row: usize, col: usize) -> f32 {
        debug_assert!(row < self.n_actions && col < self.n_actions);
        // Row-major: payoff[row * A + col]
        self.payoff[row * self.n_actions + col]
    }
}

/// The canonical 2-action Prisoner's Dilemma:
///
/// ```text
///                 Partner
///                 C       D
/// Focal    C  [  R=2   S=0 ]
///          D  [  T=3   P=1 ]
/// ```
///
/// `(T > R > P > S)` and `(2R > T + S)` for the dilemma to hold. Cooperate
/// (action 0) vs Defect (action 1). At `ω > 0.5` the embedded best response
/// flips from D (Nash) to C (cooperative CCE).
pub fn canonical_pd() -> PayoffMatrix {
    // unwrap is safe: [[f32; 2]; 2] is statically known-good.
    PayoffMatrix::new([[2.0, 0.0], [3.0, 1.0]]).expect("canonical_pd: 2x2 matrix is always valid")
}

/// Compute the embedded best response: the action that maximizes expected
/// payoff under the *coupled* predictive model `P̂(a_partner | a_self) = ω ·
/// δ(a_partner, a_self) + (1−ω) · q(a_partner)`.
///
/// Returns the action index (`0..A`). Ties are broken toward the lower index
/// (which is "Cooperate" in the canonical PD layout where C=0, D=1).
///
/// # Arguments
///
/// - `omega` — the similarity posterior `ω ∈ (0, 1)`. Values outside this range
///   are clamped: `ω ≤ 0` → pure independent (Nash), `ω ≥ 1` → pure
///   shared-shard (mirror).
/// - `payoff` — the row-player payoff matrix.
/// - `partner_marginal` — the exogenous partner-action distribution `q` under
///   the independent hypothesis. Length must equal `payoff.n_actions()`. Pass
///   `&[1.0/A; A]` for the uniform default.
///
/// # Errors
///
/// Returns `Err(MarginalShapeMismatch)` if `partner_marginal.len() !=
/// payoff.n_actions()`.
///
/// # Allocation
///
/// Zero allocations. O(A²) inner loop.
pub fn embedded_best_response(
    omega: f32,
    payoff: &PayoffMatrix,
    partner_marginal: &[f32],
) -> Result<u8, SimilarityError> {
    let mut best: u8 = 0;
    embedded_best_response_into(omega, payoff, partner_marginal, &mut best)?;
    Ok(best)
}

/// Same as [`embedded_best_response`] but writes the result into a
/// caller-supplied `&mut u8`. Avoids returning through a register for tight
/// batch loops over many focals.
pub fn embedded_best_response_into(
    omega: f32,
    payoff: &PayoffMatrix,
    partner_marginal: &[f32],
    out: &mut u8,
) -> Result<(), SimilarityError> {
    let a = payoff.n_actions();
    if partner_marginal.len() != a {
        return Err(SimilarityError::MarginalShapeMismatch {
            expected: a,
            got: partner_marginal.len(),
        });
    }

    // Clamp ω to [0, 1]. Outside this range the math degenerates.
    let omega = omega.clamp(0.0, 1.0);
    let one_minus_omega = 1.0 - omega;

    let mut best_action: usize = 0;
    let mut best_value: f32 = f32::NEG_INFINITY;

    for row in 0..a {
        // Q(row) = Σ_{col} P̂(partner=col | self=row) · R(row, col)
        //       = ω · R(row, row) + (1−ω) · Σ_{col} q(col) · R(row, col)
        let shared_contrib = omega * payoff.payoff(row, row);
        let mut indep_contrib = 0.0_f32;
        for (col, q_col) in partner_marginal.iter().enumerate().take(a) {
            indep_contrib += q_col * payoff.payoff(row, col);
        }
        let q_value = shared_contrib + one_minus_omega * indep_contrib;

        // Strict-greater comparison → ties break toward the lower index
        // (Cooperate in canonical PD where C=0).
        if q_value > best_value {
            best_value = q_value;
            best_action = row;
        }
    }

    *out = best_action as u8;
    Ok(())
}

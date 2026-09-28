//! Bounded two-phase primal simplex for the CCE LP (Plan 572).
//!
//! Drop-in replacement for [`super::lp::enumerate_bfs`] on LPs too large for
//! BFS enumeration. Solves the standard-form LP
//!
//! ```text
//! minimize   c · x
//! subject to A · x = b
//!            x ≥ 0
//! ```
//!
//! via the two-phase primal simplex with **Bland's rule** for anti-cycling
//! (smallest-index entering variable with negative reduced cost; smallest-index
//! tiebreak on the min-ratio leaving variable). Bland's rule guarantees finite
//! termination and is fully deterministic — no RNG, no hash-order dependence,
//! no parallelism in the pivot loop. Two runs on the same LP produce
//! bit-identical `x`.
//!
//! # Complexity
//!
//! Worst-case exponential (the Klee-Minty cube family), but never observed in
//! practice on CCE-sized LPs (`n_cons ≤ ~50`). The simplex visits a tiny
//! fraction of the BFS set that [`enumerate_bfs`] would enumerate exhaustively.
//!
//! # Reference
//!
//! - Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, ch. 2–3.
//! - Bland, R.G. (1977), "New finite pivoting rules for the simplex method,"
//!   *Mathematics of Operations Research* 2(2): 103–107.

/// Pivot tolerance: reduced costs within `EPS` of zero are treated as zero
/// (not eligible to enter the basis). Min-ratio denominator within `EPS` of
/// zero skips that row (degenerate pivot guard).
const EPS: f64 = 1e-9;

/// Solve the standard-form LP `min c·x  s.t.  A·x = b, x ≥ 0`.
///
/// Mirrors the [`super::lp::enumerate_bfs`] signature so the two are
/// interchangeable:
///
/// - `mat`: `n_cons × n_vars` constraint matrix (rows = constraints).
/// - `rhs`: `n_cons` RHS vector `b`. MUST be ≥ 0 (caller responsibility —
///   the CCE LP builder always produces non-negative RHS; if a row has
///   negative RHS, multiply through by −1 before calling).
/// - `obj`: `n_vars` objective coefficients `c`.
/// - `n_vars`: total variable count (ρ + slacks).
/// - `na`: count of `ρ` variables — only `obj[..na]` is consulted for the
///   returned objective value (slacks have zero objective, matching
///   `enumerate_bfs`).
///
/// Returns `Some(x[..na])` (the `ρ` entries) on success, or `None` if the LP
/// is infeasible / unbounded / numerically degenerate.
///
/// # Determinism
///
/// Fully deterministic under Bland's rule. The basis is tracked as a sorted
/// `Vec<usize>` of basic column indices; pivots are chosen by smallest index.
pub(crate) fn solve_simplex(
    mat: &[Vec<f64>],
    rhs: &[f64],
    obj: &[f64],
    n_vars: usize,
    na: usize,
) -> Option<Vec<f64>> {
    let n_cons = mat.len();
    if n_cons == 0 {
        return None;
    }
    // Caller contract: rhs ≥ 0. Defensive guard — if any RHS is negative, the
    // caller violated the contract; return None rather than silently producing
    // a wrong answer.
    if rhs.iter().any(|&b| b < -EPS) {
        return None;
    }

    // Augmented variable layout:
    //   [0 .. n_vars)                 — original variables (ρ + slacks)
    //   [n_vars .. n_vars + n_cons)   — artificial variables (Phase I only)
    //
    // Phase I objective: minimize Σ artificials.
    // Phase II objective: minimize Σ obj[j] · x[j] over original variables.
    let n_total = n_vars + n_cons;

    // Build the tableau: `tab` is `(n_cons+1) × (n_total+1)`.
    //   Row 0 = objective row (reduced cost · x).
    //   Rows 1..=n_cons = constraint rows.
    //   Column 0..n_total = variable columns.
    //   Column n_total = RHS.
    //
    // We keep two objective rows during Phase I: one for the Phase I objective
    // (sum of artificials, expressed in terms of non-basic variables) and one
    // for the Phase II objective (original c·x). After Phase I, we discard the
    // Phase I row and use the Phase II row.
    //
    // Layout: `tab[row][col]` where `row ∈ 0..=n_cons`, `col ∈ 0..=n_total`
    // (column `n_total` is the RHS).
    let mut tab = vec![vec![0.0_f64; n_total + 1]; n_cons + 1];

    // Constraint rows (1-indexed; row 0 is the objective).
    for i in 0..n_cons {
        for j in 0..n_vars {
            tab[1 + i][j] = mat[i][j];
        }
        // Artificial variable for constraint i: column `n_vars + i`, value 1.
        tab[1 + i][n_vars + i] = 1.0;
        // RHS. Clamp tiny negatives to zero (caller contract is rhs ≥ 0).
        tab[1 + i][n_total] = if rhs[i] < 0.0 { 0.0 } else { rhs[i] };
    }

    // Initial basis: all artificials. basis[i] = n_vars + i.
    let mut basis: Vec<usize> = (n_vars..n_total).collect();

    // ── Phase I: minimize Σ artificials ───────────────────────────────────
    //
    // Phase I objective row (row 0): reduced costs of the Phase I objective
    // `z_I = Σ artificials = Σ_i x[n_vars + i]`.
    //
    // Since each artificial is basic in its own constraint row, we eliminate
    // the basic artificials from row 0 by subtracting each constraint row.
    // This gives row 0 the Phase I reduced costs:
    //   reduced_cost_I[j] = -(Σ_i A[i][j])   for j < n_vars
    //   reduced_cost_I[n_vars + i] = 0       for artificials (basic)
    //   RHS_I = -(Σ_i b[i])
    for j in 0..n_vars {
        let sum: f64 = mat.iter().map(|row| row[j]).sum();
        tab[0][j] = -sum;
    }
    // Artificial columns in row 0 are 0 (basic).
    {
        let mut sum_rhs = 0.0;
        for i in 0..n_cons {
            sum_rhs += tab[1 + i][n_total];
        }
        tab[0][n_total] = -sum_rhs;
    }

    // Run the simplex pivots on the Phase I objective.
    if !pivot_loop(&mut tab, &mut basis, n_total, n_vars) {
        // Unbounded in Phase I — should not happen (Phase I objective is
        // bounded below by 0). Treat as numerical failure.
        return None;
    }

    // Phase I feasibility check: the optimal Phase I objective is tab[0][n_total].
    // If it's significantly positive, the LP is infeasible.
    let phase1_obj = -tab[0][n_total];
    if phase1_obj > EPS * 100.0 {
        // Infeasible — artificials remain in the basis with positive value.
        return None;
    }

    // Drive any remaining artificials out of the basis (they should all be at
    // value 0 if phase1_obj ≈ 0). For each artificial still basic, try to pivot
    // it out by finding a non-artificial column with a non-zero entry in its row.
    for i in 0..n_cons {
        if basis[i] >= n_vars {
            // Artificial `n_vars + i` is still basic in row i. Find a non-
            // artificial column j < n_vars with |tab[1+i][j]| > EPS to pivot in.
            let pivot_col = (0..n_vars).find(|&j| tab[1 + i][j].abs() > EPS);
            match pivot_col {
                Some(j) => {
                    // Pivot (1+i, j) → drives the artificial out.
                    pivot(&mut tab, &mut basis, 1 + i, j, n_total);
                }
                None => {
                    // Row i is all zeros in the non-artificial columns →
                    // redundant constraint. Leave the artificial in place at
                    // value 0; Phase II will ignore it (artificial columns are
                    // excluded from Phase II pivots).
                }
            }
        }
    }

    // ── Phase II: minimize the original objective c·x ─────────────────────
    //
    // Replace row 0 with the Phase II objective row. We recompute the reduced
    // costs from scratch: reduced_cost_II[j] = obj[j] − (Σ_{basic k} obj[basis_k]
    // · A_row_k[j]). For basic columns, the reduced cost is 0.
    //
    // Simpler implementation: set tab[0][j] = obj[j] for j < n_vars, 0 for j ≥
    // n_vars (artificials forbidden in Phase II), then eliminate the basic
    // variables' objective contributions.
    for j in 0..n_total {
        tab[0][j] = if j < n_vars { obj[j] } else { 0.0 };
    }
    // RHS of row 0 starts at 0 (z = 0 at the origin); we'll subtract basic-row
    // contributions below.
    tab[0][n_total] = 0.0;
    // Eliminate basic variables from row 0. For each basic variable in row i
    // (basis[i] = some column), if its objective coefficient is non-zero,
    // subtract obj[basis[i]] · row(1+i) from row 0 to zero out column basis[i].
    for i in 0..n_cons {
        let bcol = basis[i];
        if bcol >= n_vars {
            continue; // artificial — obj is 0, skip.
        }
        let coef = obj[bcol];
        if coef == 0.0 {
            continue;
        }
        // tab[0][bcol] is currently `coef` (or whatever it became). Subtract
        // `coef` times constraint row (1+i) from row 0. After this,
        // tab[0][bcol] = 0 (since tab[1+i][bcol] = 1 for the basic column).
        let pivot_row_val = tab[1 + i][bcol]; // should be 1.0
        let factor = tab[0][bcol] / pivot_row_val;
        // Zip-based row subtraction would conflict with the immutable borrow
        // of tab[1+i], so use an index loop here.
        #[allow(clippy::needless_range_loop)]
        for j in 0..=n_total {
            tab[0][j] -= factor * tab[1 + i][j];
        }
    }

    // Run the simplex pivots on the Phase II objective. Forbid entering
    // artificial columns (j ≥ n_vars) by passing `n_vars` as the column cap.
    if !pivot_loop(&mut tab, &mut basis, n_total, n_vars) {
        return None; // unbounded — CCE LPs are bounded (ρ is on the simplex),
        // so this is a numerical failure.
    }

    // Extract the solution: non-basic variables are 0; basic variable i has
    // value tab[1+i][n_total].
    let mut x = vec![0.0_f64; n_vars];
    for i in 0..n_cons {
        let bcol = basis[i];
        if bcol < n_vars {
            // Clamp tiny negatives to zero (numerical noise).
            let val = tab[1 + i][n_total];
            x[bcol] = if val < 0.0 { 0.0 } else { val };
        }
    }

    // Renormalize ρ entries to sum = 1 (matches `enumerate_bfs` post-processing).
    let sum_rho: f64 = x[..na].iter().copied().sum();
    if sum_rho > 1e-9 {
        let inv = 1.0 / sum_rho;
        for xi in x[..na].iter_mut() {
            *xi *= inv;
        }
    } else {
        // Degenerate: ρ sums to ~0. Not a valid CCE.
        return None;
    }

    Some(x[..na].to_vec())
}

/// Run the simplex pivot loop until no entering variable exists (optimal) or
/// the LP is detected unbounded. Returns `false` on unbounded, `true` on
/// optimal termination.
///
/// `n_total` is the total column count (including RHS at index `n_total`).
/// `col_cap` is the exclusive upper bound on entering-variable candidates
/// (Phase II forbids artificial columns; `col_cap = n_vars`).
fn pivot_loop(tab: &mut [Vec<f64>], basis: &mut [usize], n_total: usize, col_cap: usize) -> bool {
    let n_cons = basis.len();
    loop {
        // Bland's rule: entering variable = smallest index j < col_cap with
        // reduced cost < -EPS (i.e. tab[0][j] < -EPS).
        let entering = (0..col_cap).find(|&j| tab[0][j] < -EPS);
        let entering = match entering {
            Some(j) => j,
            None => return true, // optimal — no improving direction
        };

        // Min-ratio test with smallest-index tiebreak (Bland's rule for the
        // leaving variable). ratio = tab[1+i][n_total] / tab[1+i][entering]
        // for rows i where tab[1+i][entering] > EPS.
        let mut leaving: Option<usize> = None;
        let mut best_ratio: f64 = f64::INFINITY;
        for i in 0..n_cons {
            let denom = tab[1 + i][entering];
            if denom <= EPS {
                continue; // not a valid leaving-row candidate
            }
            let ratio = tab[1 + i][n_total] / denom;
            // Strictly-better ratio wins; ties go to the smallest index (since
            // we iterate i ascending and use strict `<`, the first row at the
            // min ratio is kept).
            if ratio < best_ratio - EPS {
                best_ratio = ratio;
                leaving = Some(i);
            } else if (ratio - best_ratio).abs() <= EPS && leaving.is_some() {
                // Tie — Bland's rule: keep the smaller index (already have it
                // since we iterate ascending and only update on strict `<`).
                // No action needed.
            }
        }
        let leaving = match leaving {
            Some(i) => i,
            None => return false, // unbounded — no row limits the entering var
        };

        // Pivot on (1+leaving, entering).
        pivot(tab, basis, 1 + leaving, entering, n_total);
    }
}

/// Perform a single simplex pivot: make `tab[pivot_row][pivot_col] = 1` and
/// zero out all other entries in column `pivot_col`. Updates `basis` so that
/// `basis[pivot_row - 1] = pivot_col`.
fn pivot(
    tab: &mut [Vec<f64>],
    basis: &mut [usize],
    pivot_row: usize,
    pivot_col: usize,
    n_total: usize,
) {
    let pivot_val = tab[pivot_row][pivot_col];
    // Normalize the pivot row.
    if pivot_val != 1.0 {
        let inv = 1.0 / pivot_val;
        for val in &mut tab[pivot_row][..=n_total] {
            *val *= inv;
        }
    }
    // Eliminate the pivot column from all other rows (including row 0).
    for r in 0..tab.len() {
        if r == pivot_row {
            continue;
        }
        let factor = tab[r][pivot_col];
        if factor == 0.0 {
            continue;
        }
        // Two disjoint mutable borrows: split so that `r` and `pivot_row`
        // land in different halves. Since r ≠ pivot_row, exactly one of
        // (r < pivot_row) / (r > pivot_row) holds.
        if r < pivot_row {
            // r is in the left half, pivot_row is right[0].
            let (left, right) = tab.split_at_mut(pivot_row);
            let other = &mut left[r];
            let prow = &right[0];
            for j in 0..=n_total {
                other[j] -= factor * prow[j];
            }
        } else {
            // pivot_row is in the left half, r is right[0].
            let (left, right) = tab.split_at_mut(r);
            let prow = &left[pivot_row];
            let other = &mut right[0];
            for j in 0..=n_total {
                other[j] -= factor * prow[j];
            }
        }
    }
    // Update the basis.
    basis[pivot_row - 1] = pivot_col;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial LP: `min x_0  s.t.  x_0 = 1, x_0 ≥ 0`. Optimum x_0 = 1.
    #[test]
    fn simplex_trivial_single_variable() {
        let mat = vec![vec![1.0]];
        let rhs = vec![1.0];
        let obj = vec![1.0];
        let x = solve_simplex(&mat, &rhs, &obj, 1, 1).expect("feasible");
        assert!((x[0] - 1.0).abs() < 1e-6, "x_0 = {}", x[0]);
    }

    /// Infeasible LP: `x_0 = 1, x_0 = 2` (no solution).
    #[test]
    fn simplex_infeasible_returns_none() {
        let mat = vec![vec![1.0], vec![1.0]];
        let rhs = vec![1.0, 2.0];
        let obj = vec![1.0];
        assert!(solve_simplex(&mat, &rhs, &obj, 1, 1).is_none());
    }

    /// Redundant constraint: `x_0 = 1, 2·x_0 = 2`. Should still solve (x_0 = 1).
    #[test]
    fn simplex_redundant_constraint() {
        let mat = vec![vec![1.0], vec![2.0]];
        let rhs = vec![1.0, 2.0];
        let obj = vec![1.0];
        let x = solve_simplex(&mat, &rhs, &obj, 1, 1).expect("feasible");
        assert!((x[0] - 1.0).abs() < 1e-6, "x_0 = {}", x[0]);
    }

    /// A 2-variable LP with a non-trivial optimum, using CCE shape (sum-to-1):
    ///   min x_0 + x_1
    ///   s.t. x_0 + x_1 = 1  (sum-to-1)
    ///        x_0 - x_1 = 0  (symmetry)
    ///        x_0, x_1 ≥ 0
    /// Optimum: x_0 = 0.5, x_1 = 0.5 (only feasible point).
    #[test]
    fn simplex_two_var_intersection() {
        let mat = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
        let rhs = vec![1.0, 0.0];
        let obj = vec![1.0, 1.0];
        let x = solve_simplex(&mat, &rhs, &obj, 2, 2).expect("feasible");
        assert!((x[0] - 0.5).abs() < 1e-6, "x_0 = {}", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-6, "x_1 = {}", x[1]);
    }

    /// Simplex must agree with BFS enumeration on the chicken-game LP shape.
    /// We construct the same LP that `lp_solve_chicken_finds_minimum_cost_cce`
    /// builds and verify the simplex finds the same optimum (γ₀ = -4).
    ///
    /// Chicken: R = [[3,1],[4,0]], state = (s_1, s_2), action = a_1, N=4, A=2.
    /// cost(s,a) = -R[a][s%2]. Deviations: always-0, always-1.
    /// Optimum: ρ(state=2 = (T,S), action=1 = T) = 1, cost = -R[1][1] = -4.
    ///
    /// Wait — s%2 for state=2 is 0 (S), action=1 is T. cost = -R[1][0] = -4. ✓
    #[test]
    fn simplex_matches_bfs_on_chicken() {
        // R[action][s_2]
        const R: [[f64; 2]; 2] = [[3.0, 1.0], [4.0, 0.0]];
        // cost(s, a) = -R[a][s % 2]
        let cost = |s: usize, a: usize| -R[a][s % 2];

        // Build the LP manually (mirroring lp.rs::solve).
        let na = 8; // N=4, A=2
        let nd = 2;
        let n_vars = na + nd;
        let n_cons = 1 + nd;
        let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
        let mut rhs = vec![0.0_f64; n_cons];
        // row 0: sum = 1
        mat[0][..na].fill(1.0);
        rhs[0] = 1.0;
        // rows 1..=nd: g_κ · ρ + s_κ = 0 where g_κ(s,a) = cost(s,a) - cost(s, κ_target)
        let dev_targets = [0, 1];
        for (k, &kt) in dev_targets.iter().enumerate() {
            for s in 0..4 {
                for a in 0..2 {
                    let j = s * 2 + a;
                    let g = cost(s, a) - cost(s, kt);
                    mat[1 + k][j] = g;
                }
            }
            mat[1 + k][na + k] = 1.0; // slack
            rhs[1 + k] = 0.0;
        }
        // objective: γ₀ = Σ cost(s,a)·ρ[s,a]  (matches `gamma` default).
        let mut obj = vec![0.0_f64; n_vars];
        for s in 0..4 {
            for a in 0..2 {
                obj[s * 2 + a] = cost(s, a);
            }
        }

        let x = solve_simplex(&mat, &rhs, &obj, n_vars, na).expect("chicken simplex feasible");

        // Compute γ₀ from the solution.
        let mut gamma0 = 0.0;
        for s in 0..4 {
            for a in 0..2 {
                gamma0 += x[s * 2 + a] * cost(s, a);
            }
        }
        assert!(
            (gamma0 - (-4.0)).abs() < 1e-3,
            "expected γ₀ = -4 (T,S optimum), got {gamma0}"
        );
    }

    /// Simplex must agree with BFS on the emission LP (N=2, A=2):
    ///   cost = [[1,3],[2,5]], optimum ρ(Low,Abate)=1, γ₀ = 1.
    #[test]
    fn simplex_matches_bfs_on_emission() {
        const C: [[f64; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
        let cost = |s: usize, a: usize| C[s][a];
        let na = 4;
        let nd = 2;
        let n_vars = na + nd;
        let n_cons = 1 + nd;
        let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
        let mut rhs = vec![0.0_f64; n_cons];
        mat[0][..na].fill(1.0);
        rhs[0] = 1.0;
        let dev_targets = [0, 1];
        for (k, &kt) in dev_targets.iter().enumerate() {
            for s in 0..2 {
                for a in 0..2 {
                    let j = s * 2 + a;
                    let g = cost(s, a) - cost(s, kt);
                    mat[1 + k][j] = g;
                }
            }
            mat[1 + k][na + k] = 1.0;
            rhs[1 + k] = 0.0;
        }
        let mut obj = vec![0.0_f64; n_vars];
        for s in 0..2 {
            for a in 0..2 {
                obj[s * 2 + a] = cost(s, a);
            }
        }
        let x = solve_simplex(&mat, &rhs, &obj, n_vars, na).expect("emission simplex feasible");
        assert!(
            (x[0] - 1.0).abs() < 1e-3,
            "mass(Low,Abate) = {}, expected 1.0",
            x[0]
        );
    }

    /// Determinism: two runs produce bit-identical output.
    #[test]
    fn simplex_deterministic_two_runs() {
        const C: [[f64; 2]; 2] = [[1.0, 3.0], [2.0, 5.0]];
        let cost = |s: usize, a: usize| C[s][a];
        let na = 4;
        let nd = 2;
        let n_vars = na + nd;
        let n_cons = 1 + nd;
        let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
        let mut rhs = vec![0.0_f64; n_cons];
        mat[0][..na].fill(1.0);
        rhs[0] = 1.0;
        for (k, kt) in [0, 1].iter().enumerate() {
            let kt = *kt;
            for s in 0..2 {
                for a in 0..2 {
                    mat[1 + k][s * 2 + a] = cost(s, a) - cost(s, kt);
                }
            }
            mat[1 + k][na + k] = 1.0;
        }
        let mut obj = vec![0.0_f64; n_vars];
        for s in 0..2 {
            for a in 0..2 {
                obj[s * 2 + a] = cost(s, a);
            }
        }
        let x1 = solve_simplex(&mat, &rhs, &obj, n_vars, na).expect("run 1 feasible");
        let x2 = solve_simplex(&mat, &rhs, &obj, n_vars, na).expect("run 2 feasible");
        assert_eq!(x1, x2, "two runs must be bit-identical");
    }

    /// Larger LP: a feasible random-ish LP at NA=81 shape (the RPS 2-player
    /// size). Verifies the simplex handles LPs far beyond BFS enumeration.
    /// Uses a constructed-feasible LP (start from a known feasible point,
    /// build constraints it satisfies).
    #[test]
    fn simplex_handles_large_lp_shape() {
        // Build a CCE-shape LP with N=9, A=9 (NA=81) + 6 slacks. Use the
        // emission-style structure but scaled: cost[s][a] = some deterministic
        // function, deviations = always-action.
        let na = 81;
        let nd = 6; // 6 deviations
        let n_vars = na + nd;
        let n_cons = 1 + nd;
        let mut mat = vec![vec![0.0_f64; n_vars]; n_cons];
        let mut rhs = vec![0.0_f64; n_cons];
        // row 0: sum ρ = 1
        mat[0][..na].fill(1.0);
        rhs[0] = 1.0;
        // cost(s,a) = ((s * 7 + a * 3) % 11) as f64 — deterministic pseudo-random.
        let cost = |s: usize, a: usize| ((s * 7 + a * 3) % 11) as f64;
        // 6 deviations: always-action 0..6
        for (k, kt) in [0, 1, 2, 3, 4, 5].iter().enumerate() {
            let kt = *kt;
            for s in 0..9 {
                for a in 0..9 {
                    let j = s * 9 + a;
                    let g = cost(s, a) - cost(s, kt);
                    mat[1 + k][j] = g;
                }
            }
            mat[1 + k][na + k] = 1.0;
            rhs[1 + k] = 0.0;
        }
        let mut obj = vec![0.0_f64; n_vars];
        for s in 0..9 {
            for a in 0..9 {
                obj[s * 9 + a] = cost(s, a);
            }
        }
        // Should solve in well under a second.
        let x = solve_simplex(&mat, &rhs, &obj, n_vars, na).expect("large LP feasible");
        // Sanity: ρ sums to 1.
        let sum: f64 = x.iter().copied().sum();
        assert!((sum - 1.0).abs() < 1e-3, "ρ sums to {sum}, expected 1.0");
        // Sanity: all entries ≥ -1e-6.
        assert!(x.iter().all(|&v| v >= -1e-6), "negative entry in ρ");
    }
}

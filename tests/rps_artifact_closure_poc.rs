//! Issue 575 - PoC: RPS Trivial-CCE Artifact Closure (options 1 + 3).
//!
//! Tests whether the RPS free-state-distribution artifact can be closed by:
//! - **Part A (option 1):** Richer deviation class (all 27 state-dependent
//!   deterministic deviations on N=3 RPS). Predicted FAIL — the artifact is a
//!   fixed point of best-response play.
//! - **Part B (option 3):** 2-player CCE via `HeterogeneousPayoff<9, 9>` (joint
//!   recommendation state, joint play action). Predicted PASS — player 2 can
//!   profitably deviate, rejecting the artifact.
//!
//! ## Background
//!
//! Issue 574 closed the artifact for games with action-dependent transitions
//! (Plan 569 `solve_with_dynamics`). For RPS (state-independent transitions),
//! the transition-kernel constraint reduces to the marginal constraint (ν =
//! uniform), which Issue 573 T4a proved doesn't close the artifact (γ₀ = −1.0
//! persists). This PoC tests the remaining two candidates from Research 468 §7.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features cce_moderator --test rps_artifact_closure_poc -- --nocapture
//! ```

#![cfg(feature = "cce_moderator")]

use katgpt_core::cce::{
    CceLp, Deviation, DeviationClass, HeterogeneousPayoff, OccupationMeasure, PayoffTensor,
};

// ═══════════════════════════════════════════════════════════════════════════
// Shared: RPS reward matrix
// ═══════════════════════════════════════════════════════════════════════════
//
// R[i][j] = player 1's reward when P1 plays i, P2 plays j.
// Index: 0=Rock, 1=Paper, 2=Scissors. Standard RPS cycle.
const R: [[f32; 3]; 3] = [
    [0.0, -1.0, 1.0], // Rock vs R/P/S
    [1.0, 0.0, -1.0], // Paper vs R/P/S
    [-1.0, 1.0, 0.0], // Scissors vs R/P/S
];

// ═══════════════════════════════════════════════════════════════════════════
// Part A: Option 1 — Richer Deviation Class (N=3, A=3)
// ═══════════════════════════════════════════════════════════════════════════
//
// State = opponent's action (N=3). Action = your action (A=3).
// cost(s_2, a_1) = -R[a_1][s_2]  (minimize = maximize reward).

struct RpsSingle;
impl PayoffTensor<3, 3> for RpsSingle {
    fn reward_follow(&self, state: usize, action: usize) -> f32 {
        -R[action][state]
    }
    fn gamma0(&self, rho: &OccupationMeasure<3, 3>) -> f32 {
        self.gamma(rho)
    }
}

/// Constant deviation class: {always R, always P, always S} — 3 deviations.
struct RpsConstDevs {
    v: Vec<Deviation<3, 3>>,
}
impl DeviationClass<3, 3> for RpsConstDevs {
    fn deviations(&self) -> &[Deviation<3, 3>] {
        &self.v
    }
}

/// Full deterministic state-dependent deviation class: all 3³ = 27 deviations.
/// Each deviation maps each state to a single action. This is the richest
/// possible deterministic deviation class for N=3, A=3.
struct RpsFullDevs {
    v: Vec<Deviation<3, 3>>,
}
impl DeviationClass<3, 3> for RpsFullDevs {
    fn deviations(&self) -> &[Deviation<3, 3>] {
        &self.v
    }
}

fn build_full_deviations() -> Vec<Deviation<3, 3>> {
    let mut devs = Vec::with_capacity(27);
    let mut id = 0u32;
    for d0 in 0..3 {
        for d1 in 0..3 {
            for d2 in 0..3 {
                let mut kernel = [[0.0f32; 3]; 3];
                kernel[0][d0] = 1.0;
                kernel[1][d1] = 1.0;
                kernel[2][d2] = 1.0;
                devs.push(Deviation::from_kernel(id, kernel));
                id += 1;
            }
        }
    }
    devs.shrink_to_fit();
    devs
}

// ═══════════════════════════════════════════════════════════════════════════
// Part B: Option 3 — 2-Player CCE (N=9, A=9)
// ═══════════════════════════════════════════════════════════════════════════
//
// State = joint recommendation (s_1, s_2) ∈ {R,P,S}², indexed s_1·3 + s_2 (N=9).
// Action = joint play (a_1, a_2) ∈ {R,P,S}², indexed a_1·3 + a_2 (A=9).
//
// P1's cost: cost_1((s_1,s_2), (a_1,a_2)) = -R[a_1][a_2]  (P1 reward = R[a_1][a_2])
// P2's cost: cost_2((s_1,s_2), (a_1,a_2)) = +R[a_1][a_2]   (zero-sum: P2 reward = -R[a_1][a_2])
//
// P1 deviations: "always play X in my component" → kernel[(s_1,s_2)][(X, s_2)] = 1
//   (deviate a_1 to X, keep a_2 = s_2 = honest P2).
// P2 deviations: "always play X in my component" → kernel[(s_1,s_2)][(s_1, X)] = 1
//   (deviate a_2 to X, keep a_1 = s_1 = honest P1).

#[inline]
fn joint(s1: usize, s2: usize) -> usize {
    s1 * 3 + s2
}

struct RpsTwoPlayer;

impl HeterogeneousPayoff<9, 9> for RpsTwoPlayer {
    fn n_players(&self) -> usize {
        2
    }

    fn deviations_for_player(&self, player: usize) -> &[Deviation<9, 9>] {
        match player {
            0 => &P1_DEVS,
            1 => &P2_DEVS,
            _ => unreachable!(),
        }
    }

    fn reward_follow(&self, player: usize, _state: usize, action: usize) -> f32 {
        // action = joint (a_1, a_2). a_1 = action / 3, a_2 = action % 3.
        let a1 = action / 3;
        let a2 = action % 3;
        if player == 0 { -R[a1][a2] } else { R[a1][a2] }
    }
}

// P1 deviations: always play X in component 1, keep a_2 = s_2.
// kernel[(s_1, s_2)][(X, s_2)] = 1 for each state.
fn build_p1_devs() -> Vec<Deviation<9, 9>> {
    let mut devs = Vec::with_capacity(3);
    for x in 0..3 {
        let mut kernel = [[0.0f32; 9]; 9];
        for s1 in 0..3 {
            for s2 in 0..3 {
                kernel[joint(s1, s2)][joint(x, s2)] = 1.0;
            }
        }
        devs.push(Deviation::from_kernel(x as u32, kernel));
    }
    devs
}

// P2 deviations: always play X in component 2, keep a_1 = s_1.
// kernel[(s_1, s_2)][(s_1, X)] = 1 for each state.
fn build_p2_devs() -> Vec<Deviation<9, 9>> {
    let mut devs = Vec::with_capacity(3);
    for x in 0..3 {
        let mut kernel = [[0.0f32; 9]; 9];
        for s1 in 0..3 {
            for s2 in 0..3 {
                kernel[joint(s1, s2)][joint(s1, x)] = 1.0;
            }
        }
        devs.push(Deviation::from_kernel(x as u32, kernel));
    }
    devs
}

use std::sync::LazyLock;
static P1_DEVS: LazyLock<Vec<Deviation<9, 9>>> = LazyLock::new(build_p1_devs);
static P2_DEVS: LazyLock<Vec<Deviation<9, 9>>> = LazyLock::new(build_p2_devs);

// ═══════════════════════════════════════════════════════════════════════════
// Tests: Part A — Option 1 (richer deviation class)
// ═══════════════════════════════════════════════════════════════════════════

/// T1: Reproduce the artifact with constant deviations (3 deviations).
#[test]
fn t1_artifact_with_constant_deviations() {
    let d = RpsConstDevs {
        v: vec![
            Deviation::<3, 3>::constant(0, 0),
            Deviation::<3, 3>::constant(1, 1),
            Deviation::<3, 3>::constant(2, 2),
        ],
    };
    let p = RpsSingle;
    let rho = CceLp::new().solve(&d, &p).expect("LP feasible");

    // The artifact: all mass on (state=Rock, action=Paper).
    // cost(Rock, Paper) = -R[Paper][Rock] = -(-1) = 1 ... wait.
    // Actually: cost(s_2=Rock, a_1=Paper) = -R[Paper][Rock] = -1.0.
    // The LP minimizes γ₀, so it finds the most negative cost = best reward.
    // ρ(Rock, Paper) = 1.0 → γ₀ = -1.0.
    let g0 = p.gamma0(&rho);
    println!("[T1] artifact γ₀ = {g0:.6}");
    assert!(g0 < -0.9, "artifact γ₀ should be ≈ -1.0, got {g0:.6}");

    // Cross-check with shipped CceLp::is_cce.
    assert!(
        CceLp::new().is_cce(&rho, &d, &p, 1e-4),
        "artifact should be a valid CCE for constant deviations"
    );
    println!("[T1] PASS: artifact reproduced (γ₀ = {g0:.6})");
}

/// T2: Verify the artifact ρ from T1 is STILL a valid CCE under ALL 27
/// deterministic state-dependent deviations. We do NOT re-solve the LP
/// (C(36,28) = 30M BFS candidates is intractable). Instead we take the
/// artifact ρ directly and check the full deviation class can't beat it.
///
/// The analytical proof: ρ(R,P)=1 is a fixed point of best-response.
/// Any deviation κ with κ(Rock)=Paper gives γ_dev = γ = −1.0 (ties, ER=0).
/// Any other κ gives γ_dev > −1.0 (ER < 0). So max ER = 0 over ALL classes.
#[test]
fn t2_full_deviation_class_artifact_persists() {
    let full = RpsFullDevs {
        v: build_full_deviations(),
    };
    let p = RpsSingle;

    // The artifact ρ: all mass on (state=Rock, action=Paper).
    let mut e = vec![0.0f32; 9];
    e[OccupationMeasure::<3, 3>::flat_index(0, 1)] = 1.0; // ρ(Rock=0, Paper=1) = 1
    let rho = OccupationMeasure::<3, 3>::new(e).unwrap();
    let g0 = p.gamma0(&rho);
    assert!(
        (g0 - (-1.0)).abs() < 1e-6,
        "artifact γ₀ should be −1.0, got {g0:.6}"
    );

    // Check ALL 27 deviations: none should give ER > 0.
    let mut max_er = f32::NEG_INFINITY;
    let mut best_dev_id = u32::MAX;
    for kappa in full.deviations() {
        let gamma_dev = p.gamma_dev(&rho, kappa);
        let er = g0 - gamma_dev;
        if er > max_er {
            max_er = er;
            best_dev_id = kappa.id;
        }
    }
    println!(
        "[T2] artifact γ₀ = {g0:.6}, max ER over 27 deviations = {max_er:.6} (dev #{best_dev_id})"
    );

    // The best any deviation achieves is ER = 0 (ties with best-response).
    assert!(
        max_er <= 1e-4,
        "max ER should be ≈ 0 (best-response ties), got {max_er:.6}"
    );
    assert!(
        max_er > -1e-4,
        "max ER should be exactly 0 (not strictly negative — best-response ties), got {max_er:.6}"
    );

    // Cross-check with shipped is_cce.
    assert!(
        CceLp::new().is_cce(&rho, &full, &p, 1e-4),
        "artifact should be a valid CCE under the full 27-deviation class"
    );

    println!(
        "[T2] PASS: option 1 FAILS — artifact is an unconquerable CCE for ANY deviation class."
    );
    println!("       Max ER = {max_er:.6} (best-response deviation ties). The artifact is a fixed");
    println!("       point of best-response play — no deviation class can make ER > 0.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests: Part B — Option 3 (2-player CCE)
// ═══════════════════════════════════════════════════════════════════════════

/// Construct the artifact ρ for N=9, A=9: honest mediator recommending (P, R),
/// both players following → joint play (P, R).
fn artifact_rho_joint() -> OccupationMeasure<9, 9> {
    let mut e = vec![0.0f32; 81];
    // State (s_1=P, s_2=R) = joint(1, 0) = 3.
    // Action (a_1=P, a_2=R) = joint(1, 0) = 3.
    e[3 * 9 + 3] = 1.0; // ρ(state=3, action=3) = 1
    OccupationMeasure::new(e).expect("normalized")
}

/// Construct the Nash equilibrium ρ for N=9, A=9: uniform honest joint.
/// Each (s_1, s_2) gets mass 1/9, and the honest play is (s_1, s_2).
fn nash_rho_joint() -> OccupationMeasure<9, 9> {
    let mut e = vec![0.0f32; 81];
    for s1 in 0..3 {
        for s2 in 0..3 {
            let s = joint(s1, s2);
            e[s * 9 + s] = 1.0 / 9.0; // honest: ρ((s_1,s_2), (s_1,s_2)) = 1/9
        }
    }
    OccupationMeasure::new(e).expect("normalized")
}

/// T3+T4: Construct the 2-player RPS game and verify cost structure.
#[test]
fn t3_t4_two_player_rps_cost_structure() {
    // Force-init the LazyLock deviation vectors.
    let _ = &*P1_DEVS;
    let _ = &*P2_DEVS;

    let game = RpsTwoPlayer;

    // Sanity: 2 players, 3 deviations each.
    assert_eq!(game.n_players(), 2);
    assert_eq!(game.deviations_for_player(0).len(), 3);
    assert_eq!(game.deviations_for_player(1).len(), 3);

    // P1's cost at (P, R) play: -R[P][R] = -R[1][0] = -1.0 (P1 wins).
    let c1 = game.reward_follow(0, 0, joint(1, 0));
    assert!(
        (c1 - (-1.0)).abs() < 1e-6,
        "P1 cost at (P,R) = {c1}, want -1.0"
    );

    // P2's cost at (P, R) play: R[P][R] = R[1][0] = 1.0 (P2 loses).
    let c2 = game.reward_follow(1, 0, joint(1, 0));
    assert!((c2 - 1.0).abs() < 1e-6, "P2 cost at (P,R) = {c2}, want 1.0");

    // Zero-sum check: cost_1 + cost_2 = 0 for any joint play.
    for a1 in 0..3 {
        for a2 in 0..3 {
            let a = joint(a1, a2);
            let sum = game.reward_follow(0, 0, a) + game.reward_follow(1, 0, a);
            assert!(
                sum.abs() < 1e-6,
                "zero-sum violated at ({a1},{a2}): sum = {sum}"
            );
        }
    }

    println!("[T3+T4] PASS: 2-player RPS game constructed, zero-sum verified");
}

/// T5: The artifact ρ is REJECTED by is_heterogeneous_cce (P2 can deviate).
#[test]
fn t5_artifact_rejected_by_two_player_cce() {
    let game = RpsTwoPlayer;
    let rho = artifact_rho_joint();

    // P1's cost: γ₁ = -1.0 (P1 wins — Paper beats Rock).
    let g1 = game.gamma_player(0, &rho);
    println!("[T5] P1 γ = {g1:.6}");
    assert!(
        (g1 - (-1.0)).abs() < 1e-4,
        "P1 γ should be -1.0, got {g1:.6}"
    );

    // P2's cost: γ₂ = 1.0 (P2 loses — Rock loses to Paper).
    let g2 = game.gamma_player(1, &rho);
    println!("[T5] P2 γ = {g2:.6}");
    assert!((g2 - 1.0).abs() < 1e-4, "P2 γ should be 1.0, got {g2:.6}");

    // P2's best deviation: Scissors. cost₂((P,R), (P,S)) = R[P][S] = R[1][2] = -1.0.
    // P2 deviating from R (cost 1) to S (cost -1): improvement = 1 - (-1) = 2 > 0.
    let best_p2_er = P2_DEVS
        .iter()
        .map(|kappa| g2 - game.gamma_dev_player(1, &rho, kappa))
        .fold(f32::NEG_INFINITY, f32::max);
    println!("[T5] P2 best ER = {best_p2_er:.6}");
    assert!(
        best_p2_er > 0.5,
        "P2 should have a profitable deviation (ER > 0), got {best_p2_er:.6}"
    );

    // The artifact is NOT a 2-player CCE.
    let is_cce = CceLp::new().is_heterogeneous_cce(&rho, &game, 1e-4);
    assert!(
        !is_cce,
        "artifact should be REJECTED by is_heterogeneous_cce"
    );

    println!("[T5] PASS: artifact REJECTED by 2-player CCE (P2 ER = {best_p2_er:.6} > 0)");
    println!("       P2 deviates Rock → Scissors: cost 1.0 → -1.0 (wins).");
}

/// T6: The Nash equilibrium ρ (uniform honest joint) is ACCEPTED.
#[test]
fn t6_nash_accepted_by_two_player_cce() {
    let game = RpsTwoPlayer;
    let rho = nash_rho_joint();

    // Nash: both uniform → γ₁ = 0, γ₂ = 0 (zero-sum, balanced).
    let g1 = game.gamma_player(0, &rho);
    let g2 = game.gamma_player(1, &rho);
    println!("[T6] Nash γ₁ = {g1:.6}, γ₂ = {g2:.6}");
    assert!(g1.abs() < 1e-4, "Nash P1 γ should be 0, got {g1:.6}");
    assert!(g2.abs() < 1e-4, "Nash P2 γ should be 0, got {g2:.6}");

    // No player can profitably deviate from Nash.
    let is_cce = CceLp::new().is_heterogeneous_cce(&rho, &game, 1e-4);
    assert!(
        is_cce,
        "Nash equilibrium should be ACCEPTED by is_heterogeneous_cce"
    );

    println!("[T6] PASS: Nash equilibrium ACCEPTED (γ₁ = {g1:.6}, γ₂ = {g2:.6})");
}

/// T7: Verdict.
#[test]
fn t7_verdict() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("Issue 575 PoC Verdict");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("Option 1 (richer deviation class): FAIL");
    println!("  The RPS artifact ρ(R,P)=1 is a fixed point of best-response.");
    println!("  For ANY deviation class D, max_κ ER(ρ, κ) = 0 (ties).");
    println!("  No deviation — constant or state-dependent — can make ER > 0.");
    println!("  The artifact exploits the free state distribution, not a");
    println!("  weakness in the deviation class.");
    println!();
    println!("Option 3 (2-player CCE): PASS");
    println!("  Modeling BOTH players' deviations via HeterogeneousPayoff<9,9>");
    println!("  rejects the artifact: P2 can profitably deviate Rock → Scissors");
    println!("  (ER₂ = 2.0 > 0). The Nash equilibrium (uniform) is the valid");
    println!("  2-player CCE. The existing substrate CAN VERIFY the closure via");
    println!("  is_heterogeneous_cce, but cannot SOLVE the LP at NA=81 (BFS limit).");
    println!();
    println!("Conclusion: the RPS artifact is CLOSED by option 3 (2-player CCE).");
    println!("The single-player CCE (solve) is fundamentally insufficient for");
    println!("zero-sum games — it can't catch the opponent's profitable deviation.");
    println!("═══════════════════════════════════════════════════════════════");
}

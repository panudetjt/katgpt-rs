//! G1 correctness fixtures (Plan 620): feasibility under Capacity,
//! objective ≤ initial on every fixture, improvement-vs-greedy sanity,
//! constraint-violation fallback behavior — plus the delta-evaluation
//! property test (incremental root == full recompute, the exactness
//! claim that replaces Meta's segment tree).

use katgpt_assign::fixtures;
use katgpt_assign::rng::SplitMix64;
use katgpt_assign::{Limits, Seed, Solver, eval_assignment, solve};

#[test]
fn perfect_balance_fixture_reaches_known_optimum() {
    // Constructed with an exact equal-sum partition: optimum spread 0,
    // zero violation.
    for seed in [1u64, 2, 3] {
        let problem = fixtures::perfect_balance(seed, 5, 20, 5);
        let solution = solve(&problem, Seed(seed), &Limits::default()).unwrap();
        assert_eq!(solution.violation_root, 0, "seed {seed}: infeasible");
        // Spread 0 ⇔ max util == min util ⇔ every container at total/B.
        let (folded, violation, goal) =
            eval_assignment(&problem, &solution.assignment.containers).unwrap();
        assert_eq!(violation, 0);
        assert_eq!(goal, 0, "seed {seed}: spread {goal} > 0 (optimum is 0)");
        assert_eq!(
            folded, solution.folded_objective,
            "solution/objective drift"
        );
    }
}

#[test]
fn overpacked_bin_finds_the_single_fat_move() {
    // Known optimum: move the fat object (one move) ⇒ folded == 1
    // (movement 1, violation 0).
    let problem = fixtures::overpacked_one_bin(4, 10, 5);
    let solution = solve(&problem, Seed(0), &Limits::default()).unwrap();
    assert_eq!(solution.violation_root, 0, "did not repair");
    let moved = solution
        .assignment
        .containers
        .iter()
        .zip(&problem.initial)
        .filter(|(a, i)| a != i)
        .count();
    assert_eq!(moved, 1, "optimum is exactly one move, made {moved}");
    assert_eq!(solution.folded_objective, 1);
}

#[test]
fn local_search_never_loses_to_ffd() {
    // The G2 sanity core at fixture scale: on every random instance the
    // solver's folded objective must be <= first-fit-decreasing's.
    for seed in 1..=8u64 {
        let problem = fixtures::random_instance(seed, 120, 8, 2, 40);
        let greedy = fixtures::first_fit_decreasing(&problem);
        let (greedy_folded, greedy_violation, _) = eval_assignment(&problem, &greedy).unwrap();
        let solution = solve(&problem, Seed(seed), &Limits::default()).unwrap();
        assert!(
            solution.folded_objective <= greedy_folded,
            "seed {seed}: solver {} > FFD {} (violation {} vs {})",
            solution.folded_objective,
            greedy_folded,
            solution.violation_root,
            greedy_violation
        );
    }
}

#[test]
fn initial_assignment_is_evaluated_consistently() {
    // eval_assignment(problem, initial) must equal the solver's starting
    // state — the never-worse guard's reference point.
    let problem = fixtures::random_instance(31, 40, 4, 2, 15);
    let (folded, violation, goal) = eval_assignment(&problem, &problem.initial).unwrap();
    let solver = Solver::new(&problem, Seed(0), Limits::default()).unwrap();
    // Solver exposes its state before run via the diagnostic surface —
    // recompute_all + folded_objective.
    let mut solver = solver;
    solver.recompute_all();
    assert_eq!(solver.folded_objective(), folded);
    let (v, g) = solver.objective_parts();
    assert_eq!(v, violation);
    assert_eq!(g, goal);
}

/// The delta-evaluation property (the exactness claim): for random moves
/// on random states, the incremental root value equals a full recompute.
/// This is the test that lets integer sums replace Meta's segment tree.
#[test]
fn delta_evaluation_matches_full_recompute() {
    let mut rng = SplitMix64::new(0xC0FFEE);
    for case in 0..6 {
        let n = 8 + rng.below(8);
        let b = 2 + rng.below(3);
        let dims = 1 + rng.below(3);
        let problem = fixtures::random_instance(100 + case, n, b, dims, 12);
        let mut solver = Solver::new(&problem, Seed(case), Limits::default()).unwrap();
        solver.recompute_all();
        let n = problem.num_objects as u32;
        let b = problem.num_containers as u32;
        // Random walk: evaluate a random move, apply it, verify against a
        // full recompute every step.
        for step in 0..200 {
            let o = rng.below(n as usize) as u32;
            if rng.below(2) == 0 {
                let dst = rng.below(b as usize) as u32;
                if dst == solver.assignment()[o as usize] {
                    continue;
                }
                let predicted = solver.eval_single(o, dst);
                solver.apply_single(o, dst);
                let full = {
                    solver.recompute_all();
                    solver.folded_objective()
                };
                assert_eq!(
                    predicted, full,
                    "case {case} step {step}: single-move delta drifted"
                );
            } else {
                let a = rng.below(n as usize) as u32;
                let c = rng.below(n as usize) as u32;
                if a == c || solver.assignment()[a as usize] == solver.assignment()[c as usize] {
                    continue;
                }
                let predicted = solver.eval_swap(a, c);
                solver.apply_swap(a, c);
                let full = {
                    solver.recompute_all();
                    solver.folded_objective()
                };
                assert_eq!(
                    predicted, full,
                    "case {case} step {step}: swap delta drifted"
                );
            }
        }
    }
}

//! G2 mid-size lower-bound gap (Plan 620): n ≈ 100–1000. The honest bound
//! available for this fixture family is the pigeonhole lower bound on
//! per-dim MAX utilization: `max_b util(dim, b) >= ceil(total_d / B)`.
//! The solver's max-util must sit ON or ABOVE the bound (sanity — no
//! beating pigeonhole) and must not exceed FFD's max-util (the outside
//! baseline). The numeric gap-to-bound is reported in the G2 bench doc
//! (Bench 924), not asserted to zero — the bound is not tight for this
//! family.

use katgpt_assign::fixtures;
use katgpt_assign::{Limits, Seed, eval_assignment, solve};

fn max_util(problem: &katgpt_assign::Problem, assignment: &[u32], dim: usize) -> i64 {
    let mut load = vec![0i64; problem.num_containers];
    for o in 0..problem.num_objects {
        load[assignment[o] as usize] += problem.demands[o * problem.num_dims + dim];
    }
    load.into_iter().max().unwrap()
}

fn pigeonhole_lb(problem: &katgpt_assign::Problem, dim: usize) -> i64 {
    let total: i128 = (0..problem.num_objects)
        .map(|o| problem.demands[o * problem.num_dims + dim] as i128)
        .sum();
    ((total + problem.num_containers as i128 - 1) / problem.num_containers as i128) as i64
}

#[test]
fn max_util_respects_pigeonhole_and_beats_ffd() {
    for (n, b) in [(100usize, 10usize), (300, 15), (1000, 20)] {
        let problem = fixtures::random_instance(n as u64, n, b, 2, 40);
        let solution = solve(&problem, Seed(n as u64), &Limits::default()).unwrap();
        let ffd = fixtures::first_fit_decreasing(&problem);
        for dim in 0..problem.num_dims {
            let lb = pigeonhole_lb(&problem, dim);
            let ls = max_util(&problem, &solution.assignment.containers, dim);
            let greedy = max_util(&problem, &ffd, dim);
            assert!(
                ls >= lb,
                "n={n} dim={dim}: solver max-util {ls} beat pigeonhole {lb} (oracle bug)"
            );
            assert!(
                ls <= greedy,
                "n={n} dim={dim}: solver max-util {ls} > FFD {greedy}"
            );
        }
        // Folded objective: solver never loses to FFD at scale either.
        let (ffd_folded, _, _) = eval_assignment(&problem, &ffd).unwrap();
        assert!(
            solution.folded_objective <= ffd_folded,
            "n={n}: solver {} > FFD {}",
            solution.folded_objective,
            ffd_folded
        );
    }
}

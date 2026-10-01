//! Dirichlet-sampled Markov chain generator for data-probe diagnostics.
//!
//! Generates transition matrices from Dirichlet(α, …, α) distributions and
//! selects the one whose entropy rate is closest to a target value `H`.
//! Used as a controlled "probe-LLM" where the true data-generating distribution
//! is known exactly, enabling formal validation of information-theoretic claims.

/// A Markov chain with known transition matrix and computed properties.
pub struct MarkovChain {
    /// Transition matrix `P[i][j] = Pr(next=j | current=i)`.
    pub transition: Vec<Vec<f32>>,
    /// Stationary distribution π.
    pub stationary: Vec<f32>,
    /// Number of states (= vocabulary size for probe-LLM).
    pub num_states: usize,
    /// Computed entropy rate H(P) = -Σᵢ πᵢ Σⱼ Pᵢⱼ log Pᵢⱼ.
    pub entropy_rate: f32,
}

/// Sample one Gamma(α, 1) variate — Marsaglia–Tsang squeeze-rejection with the
/// small-α boost (`α < 1`: sample Gamma(α+1), multiply by `U^{1/α}`).
///
/// Deterministic given the rng stream (Box–Muller normals from paired f32 draws).
/// Diagnostics-path code — clarity over speed; the hot-path-grade sampler is
/// Issue 912 T2's `dirichlet_dist` module (unscheduled).
fn sample_gamma(alpha: f32, rng: &mut fastrand::Rng) -> f32 {
    debug_assert!(alpha > 0.0);
    // Small-α boost: Gamma(α) = Gamma(α+1) · U^{1/α} (Stuart's theorem).
    if alpha < 1.0 {
        let boost = rng.f32().max(1e-10).powf(1.0 / alpha);
        return sample_gamma(alpha + 1.0, rng) * boost;
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Box–Muller standard normal (diagnostics-grade; one pair per draw).
        let u1 = rng.f32().max(1e-10);
        let u2 = rng.f32();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        let v = 1.0 + c * z;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.f32();
        // Squeeze test (fast acceptance).
        if u < 1.0 - 0.0331 * z * z * z * z {
            return d * v;
        }
        // Log-likelihood test (exact acceptance).
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Sample a single Dirichlet(α, …, α) variate of dimension `k` into a pre-allocated buffer.
///
/// `α == 1.0` (the flat Dirichlet) takes the exponential path — `K` iid
/// `-ln(U)` variates, normalized — **bit-identical to the historical behavior**
/// so every existing α=1 caller's streams are unchanged (the Issue 912 T1
/// repair is strictly additive). `α ≠ 1` draws `K` Gamma(α, 1) variates and
/// normalizes — the general construction the docstring always promised.
///
/// `buf.len()` must equal `k`. Reuses the buffer to avoid per-call allocation.
///
/// **Small-α limit (f32)**: `sample_gamma` computes `U^{1/α}` in linear f32
/// space — for α ≲ 0.02 that exponent underflows to subnormal/zero on a large
/// fraction of draws, and a row whose every coordinate underflows falls back
/// to the uniform branch (the OPPOSITE of peaked). In-repo callers (α = 1.0,
/// 0.1, 0.05) are unaffected (all-underflow ≈ 1e-8 at α=0.05, K=4). The
/// general sampler (Issue 912 T2 `dirichlet_dist`) must draw in LOG space
/// (log-Gamma + log-sum-exp normalization) — its G1 grid starts at α=0.01,
/// exactly where this limit bites.
fn sample_dirichlet_into(k: usize, alpha: f32, rng: &mut fastrand::Rng, buf: &mut [f32]) {
    debug_assert_eq!(buf.len(), k);
    let mut sum = 0.0f32;
    if alpha == 1.0 {
        for x in buf.iter_mut() {
            let u = rng.f32();
            let u_safe = u.max(1e-10);
            let v = -u_safe.ln();
            *x = v;
            sum += v;
        }
    } else {
        for x in buf.iter_mut() {
            let v = sample_gamma(alpha, rng);
            *x = v;
            sum += v;
        }
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for x in buf.iter_mut() {
            *x *= inv_sum;
        }
    } else {
        let v = 1.0 / k as f32;
        buf.fill(v);
    }
}

/// Convenience wrapper: returns a new Vec for one-shot use.
#[allow(dead_code)]
fn sample_dirichlet(k: usize, alpha: f32, rng: &mut fastrand::Rng) -> Vec<f32> {
    let mut buf = vec![0.0f32; k];
    sample_dirichlet_into(k, alpha, rng, &mut buf);
    buf
}

/// Compute stationary distribution by power iteration.
///
/// Starts with uniform π, then applies π ← π·P for `n_iters` iterations.
/// Uses pre-allocated buffers and swap to avoid allocation inside the loop.
fn stationary_distribution(transition: &[Vec<f32>], n_iters: usize) -> Vec<f32> {
    let k = transition.len();
    let inv_k = 1.0 / k as f32;
    let mut pi = vec![inv_k; k];
    let mut new_pi = vec![0.0f32; k];
    for _ in 0..n_iters {
        for i in 0..k {
            for j in 0..k {
                new_pi[j] += pi[i] * transition[i][j];
            }
        }
        // Renormalize to prevent drift.
        let sum: f32 = new_pi.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for x in new_pi.iter_mut() {
                *x *= inv_sum;
            }
        }
        std::mem::swap(&mut pi, &mut new_pi);
        new_pi.fill(0.0);
    }
    pi
}

/// Compute entropy rate H(P) = -Σᵢ πᵢ Σⱼ Pᵢⱼ log₂(Pᵢⱼ).
fn entropy_rate(transition: &[Vec<f32>], stationary: &[f32]) -> f32 {
    let ln2 = std::f32::consts::LN_2;
    let mut h = 0.0f32;
    for (i, row) in transition.iter().enumerate() {
        let mut row_entropy = 0.0f32;
        for &p in row {
            if p > 0.0 {
                row_entropy -= p * (p.ln() / ln2);
            }
        }
        h += stationary[i] * row_entropy;
    }
    h
}

/// Generate a Markov chain with entropy rate closest to `target_h`.
///
/// Samples `n_candidates` transition matrices from Dirichlet(α, …, α),
/// computes entropy rate for each, and returns the one closest to `target_h`.
pub fn generate_markov_chain(
    num_states: usize,
    target_h: f32,
    alpha: f32,
    n_candidates: usize,
    rng: &mut fastrand::Rng,
) -> MarkovChain {
    let mut best_chain: Option<MarkovChain> = None;
    let mut best_diff = f32::INFINITY;

    // Pre-allocate transition matrix buffer, reused across candidates.
    let mut transition_buf: Vec<Vec<f32>> =
        (0..num_states).map(|_| vec![0.0f32; num_states]).collect();

    for _ in 0..n_candidates {
        // Sample a row-stochastic transition matrix from Dirichlet into reused buffer.
        for row in transition_buf.iter_mut() {
            sample_dirichlet_into(num_states, alpha, rng, row);
        }

        let stationary = stationary_distribution(&transition_buf, 100);
        let h = entropy_rate(&transition_buf, &stationary);

        let diff = (h - target_h).abs();
        if diff < best_diff {
            best_diff = diff;
            best_chain = Some(MarkovChain {
                transition: transition_buf.clone(),
                stationary,
                entropy_rate: h,
                num_states,
            });
        }
    }

    best_chain.expect("at least one candidate must be generated")
}

/// Sample a sequence of length `n` from the Markov chain.
///
/// Starts by sampling the initial state from the stationary distribution,
/// then follows the transition probabilities for subsequent states.
pub fn sample_sequence(chain: &MarkovChain, n: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }

    let mut sequence = Vec::with_capacity(n);

    // Sample initial state from stationary distribution.
    sequence.push(sample_categorical(&chain.stationary, rng));

    // Sample subsequent states from transition probabilities.
    for _ in 1..n {
        let current = *sequence.last().unwrap();
        sequence.push(sample_categorical(&chain.transition[current], rng));
    }

    sequence
}

/// Sample from a categorical distribution given probabilities.
#[inline]
fn sample_categorical(probs: &[f32], rng: &mut fastrand::Rng) -> usize {
    let u = rng.f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i;
        }
    }
    // Fallback to last state (handles floating-point rounding).
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_rows_sum_to_one() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 2.0, 1.0, 10, &mut rng);
        for row in &chain.transition {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row sum = {sum}");
        }
    }

    #[test]
    fn test_stationary_sums_to_one() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 2.0, 1.0, 10, &mut rng);
        let sum: f32 = chain.stationary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "stationary sum = {sum}");
    }

    #[test]
    fn test_entropy_rate_bounded() {
        let mut rng = fastrand::Rng::new();
        let k = 4usize;
        let chain = generate_markov_chain(k, 1.5, 1.0, 10, &mut rng);
        let max_h = (k as f32).log2();
        assert!(
            chain.entropy_rate >= 0.0 && chain.entropy_rate <= max_h + 1e-4,
            "entropy_rate = {}, max = {}",
            chain.entropy_rate,
            max_h
        );
    }

    #[test]
    fn test_sample_sequence_length() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        let seq = sample_sequence(&chain, 100, &mut rng);
        assert_eq!(seq.len(), 100);
        assert!(seq.iter().all(|&s| s < 4));
    }

    #[test]
    fn test_empty_sequence() {
        let mut rng = fastrand::Rng::new();
        let chain = generate_markov_chain(4, 1.0, 1.0, 10, &mut rng);
        let seq = sample_sequence(&chain, 0, &mut rng);
        assert!(seq.is_empty());
    }

    // Issue 912 T1 repair pin: α ≠ 1 now actually concentrates. Dir(0.05,…)
    // rows are peaked (expected row max ≈ 0.9+), while the flat Dir(1) rows
    // the broken sampler silently produced for this α average max ≈ 0.52 —
    // the G3 premise "very peaked transitions" is now in force.
    #[test]
    fn test_dirichlet_alpha_concentrates() {
        let mut rng = fastrand::Rng::with_seed(7);
        let k = 4usize;
        let mut buf = vec![0.0f32; k];
        let mut mean_max = 0.0f32;
        let draws = 400;
        for _ in 0..draws {
            sample_dirichlet_into(k, 0.05, &mut rng, &mut buf);
            let mx = buf.iter().copied().fold(0.0f32, f32::max);
            mean_max += mx;
        }
        mean_max /= draws as f32;
        assert!(
            mean_max > 0.7,
            "Dir(0.05) rows should be peaked, mean max = {mean_max} (flat ≈ 0.52)"
        );
    }

    // GOAT proof G1: Markov entropy accuracy — empirical entropy ≈ computed entropy_rate within 5%.
    #[test]
    fn goat_g1_markov_entropy_accuracy() {
        let mut rng = fastrand::Rng::with_seed(77);
        let chain = generate_markov_chain(4, 1.5, 1.0, 50, &mut rng);

        // Sample a long sequence and estimate empirical entropy rate from transition counts.
        let seq = sample_sequence(&chain, 100_000, &mut rng);
        let k = chain.num_states;
        let mut transition_counts = vec![vec![0usize; k]; k];
        let mut state_counts = vec![0usize; k];
        for w in seq.windows(2) {
            transition_counts[w[0]][w[1]] += 1;
            state_counts[w[0]] += 1;
        }
        // Count the last state for stationary estimate (it was never a "from" state in windows).
        state_counts[*seq.last().unwrap()] += 1;

        // Compute empirical entropy rate: H_hat = -Σᵢ π̂ᵢ Σⱼ P̂ᵢⱼ log₂(P̂ᵢⱼ)
        let ln2 = std::f32::consts::LN_2;
        let total = seq.len() as f32;
        let mut empirical_h = 0.0f32;
        for i in 0..k {
            let pi_hat = state_counts[i] as f32 / total;
            let row_sum = transition_counts[i].iter().sum::<usize>() as f32;
            if row_sum == 0.0 {
                continue;
            }
            let mut row_h = 0.0f32;
            for &count in &transition_counts[i] {
                let p = count as f32 / row_sum;
                if p > 0.0 {
                    row_h -= p * (p.ln() / ln2);
                }
            }
            empirical_h += pi_hat * row_h;
        }

        let relative_error = (empirical_h - chain.entropy_rate).abs() / chain.entropy_rate;
        assert!(
            relative_error < 0.05,
            "GOAT G1 FAIL: empirical_h = {empirical_h:.6}, entropy_rate = {:.6}, relative_error = {:.4}",
            chain.entropy_rate,
            relative_error
        );
    }
}

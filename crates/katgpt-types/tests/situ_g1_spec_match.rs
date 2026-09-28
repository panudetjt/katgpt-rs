//! G1 numerical correctness test for SiTU activation (Proposal 032 Phase 1).
//!
//! Verifies `situ()` matches the PyTorch reference `SituAndMul` from
//! `modeling_kimi_k3_linear.py`:
//!   situ_a = beta * tanh(gate / beta) * sigmoid(gate)
//!   up_t   = linear_beta * tanh(up / linear_beta)   (when linear_beta is Some)
//!   output = situ_a * up_t

use katgpt_types::situ;

/// Reference sigmoid (f64 precision for ground truth).
#[inline]
fn sigmoid_ref(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference tanh (f64 precision for ground truth).
#[inline]
fn tanh_ref(x: f64) -> f64 {
    x.tanh()
}

/// Compute the reference SituAndMul output for a single (gate, up) pair.
fn situ_ref(gate: f64, up: f64, beta: f64, linear_beta: Option<f64>) -> f64 {
    let situ_a = beta * tanh_ref(gate / beta) * sigmoid_ref(gate);
    let up_t = match linear_beta {
        Some(lb) => lb * tanh_ref(up / lb),
        None => up,
    };
    situ_a * up_t
}

#[test]
fn g1_situ_zero_gate_produces_zero() {
    // gate=0 → sigmoid(0)=0.5, tanh(0)=0 → situ_a = beta * 0 * 0.5 = 0 → output = 0
    let mut hidden = [0.0f32];
    let gate = [0.0f32];
    let up = [1.0f32];
    situ(&mut hidden, &gate, &up, 4.0, Some(25.0));
    assert!(
        hidden[0].abs() < 1e-7,
        "situ(0, *) should be 0, got {}",
        hidden[0]
    );
}

#[test]
fn g1_situ_matches_reference_with_linear_beta() {
    // Kimi-K3-0.40B params: beta=4.0, linear_beta=25.0
    let beta = 4.0f32;
    let linear_beta = Some(25.0f32);
    let test_values: &[f32] = &[-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

    let n = test_values.len();
    let mut hidden = vec![0.0f32; n];
    let gate: Vec<f32> = test_values.to_vec();
    let up: Vec<f32> = test_values.to_vec();

    situ(&mut hidden, &gate, &up, beta, linear_beta);

    for i in 0..n {
        let expected = situ_ref(
            f64::from(gate[i]),
            f64::from(up[i]),
            f64::from(beta),
            linear_beta.map(f64::from),
        );
        let diff = (f64::from(hidden[i]) - expected).abs();
        assert!(
            diff < 1e-4,
            "situ({}, {}, {}, {:?}) = {} but reference = {} (diff {})",
            gate[i],
            up[i],
            beta,
            linear_beta,
            hidden[i],
            expected,
            diff
        );
    }
}

#[test]
fn g1_situ_matches_reference_without_linear_beta() {
    // linear_beta = None: up passes through unchanged
    let beta = 4.0f32;
    let linear_beta: Option<f32> = None;
    let test_values: &[f32] = &[-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];

    let n = test_values.len();
    let mut hidden = vec![0.0f32; n];
    let gate: Vec<f32> = test_values.to_vec();
    let up: Vec<f32> = test_values.to_vec();

    situ(&mut hidden, &gate, &up, beta, linear_beta);

    for i in 0..n {
        let expected = situ_ref(f64::from(gate[i]), f64::from(up[i]), f64::from(beta), None);
        let diff = (f64::from(hidden[i]) - expected).abs();
        assert!(
            diff < 1e-4,
            "situ({}, {}, {}, None) = {} but reference = {} (diff {})",
            gate[i],
            up[i],
            beta,
            hidden[i],
            expected,
            diff
        );
    }
}

#[test]
fn g1_situ_sweep_minus_ten_to_ten() {
    // Dense sweep over [-10, 10] — the GOAT gate spec from Proposal 032 T1.4
    let beta = 4.0f32;
    let linear_beta = Some(25.0f32);
    let n = 201; // -10.0 to 10.0 in 0.1 steps
    let values: Vec<f32> = (0..n).map(|i| -10.0 + i as f32 * 0.1).collect();

    let mut hidden = vec![0.0f32; n];
    // gate = values, up = values * 0.5 (different from gate to exercise both paths)
    let up: Vec<f32> = values.iter().map(|v| v * 0.5).collect();

    situ(&mut hidden, &values, &up, beta, linear_beta);

    let mut max_diff = 0.0f64;
    let lb_f64: f64 = 25.0; // matches linear_beta above
    for i in 0..n {
        let expected = situ_ref(
            f64::from(values[i]),
            f64::from(up[i]),
            f64::from(beta),
            Some(lb_f64),
        );
        let diff = (f64::from(hidden[i]) - expected).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-3,
            "situ gate={}, up={} diff {} exceeds 1e-3",
            values[i],
            up[i],
            diff
        );
    }
    // Report the worst-case diff for GOAT documentation
    eprintln!("G1 situ sweep [-10,10]: max_diff = {max_diff:.2e}");
}

#[test]
fn g1_situ_large_positive_gate_saturates_to_beta() {
    // For very large positive gate: tanh(gate/beta)→1, sigmoid(gate)→1
    // → situ_a → beta * 1 * 1 = beta
    // With linear_beta: up_t = lb * tanh(up/lb). For up=lb: up_t = lb * tanh(1) ≈ lb * 0.7616
    let beta = 4.0f32;
    let linear_beta = 25.0f32;
    let mut hidden = [0.0f32];
    let gate = [100.0f32]; // large positive → saturation
    let up = [linear_beta]; // up = lb → tanh(1)
    situ(&mut hidden, &gate, &up, beta, Some(linear_beta));

    let expected = beta * linear_beta * 0.761_594_2; // beta * lb * tanh(1)
    let diff = (hidden[0] - expected).abs();
    assert!(
        diff < 0.01,
        "large-gate saturation: expected ~{expected:.4}, got {}",
        hidden[0]
    );
}

#[test]
fn g1_situ_negative_gate_tanh_dominates() {
    // For large negative gate: tanh(gate/beta)→-1, sigmoid(gate)→0
    // sigmoid decays faster → output → 0 (not -beta)
    let beta = 4.0f32;
    let mut hidden = [0.0f32];
    let gate = [-100.0f32];
    let up = [1.0f32];
    situ(&mut hidden, &gate, &up, beta, None);
    assert!(
        hidden[0].abs() < 1e-10,
        "negative-gate should → 0, got {}",
        hidden[0]
    );
}

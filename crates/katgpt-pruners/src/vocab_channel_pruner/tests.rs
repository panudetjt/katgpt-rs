//! Tests for vocab_channel_pruner (extracted from mod.rs by Issue 176).

use super::*;

// ── Phase 1 Tests ──

#[test]
fn test_skewness_symmetric() {
    // Symmetric distribution → skewness ≈ 0
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let s = skewness(&values);
    assert!(
        s.abs() < 0.1,
        "Symmetric distribution should have near-zero skewness, got {s}"
    );
}

#[test]
fn test_skewness_right_tail() {
    // Right-skewed: long tail to the right
    let values = [1.0, 1.0, 1.0, 1.0, 1.0, 100.0];
    let s = skewness(&values);
    assert!(
        s > 0.5,
        "Right-skewed distribution should have positive skewness, got {s}"
    );
}

#[test]
fn test_skewness_left_tail() {
    // Left-skewed: most values are high, tail extends to the left
    let values = [100.0, 100.0, 100.0, 100.0, 100.0, 1.0];
    let s = skewness(&values);
    assert!(
        s < -0.5,
        "Left-skewed distribution should have negative skewness, got {s}"
    );
}

#[test]
fn test_skewness_degenerate() {
    assert_eq!(skewness(&[]), 0.0, "Empty should be 0");
    assert_eq!(skewness(&[1.0]), 0.0, "Single should be 0");
    assert_eq!(skewness(&[1.0, 2.0]), 0.0, "Two should be 0");
    assert_eq!(skewness(&[3.0, 3.0, 3.0]), 0.0, "Zero variance should be 0");
}

#[test]
fn test_excess_kurtosis_peaked() {
    let values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
    let k = excess_kurtosis(&values);
    assert!(
        k > 5.0,
        "Peaked distribution should have high kurtosis, got {k}"
    );
}

#[test]
fn test_excess_kurtosis_uniform() {
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let k = excess_kurtosis(&values);
    assert!(k < 0.0, "Uniform should have negative kurtosis, got {k}");
}

#[test]
fn test_excess_kurtosis_edge_cases() {
    assert_eq!(excess_kurtosis(&[]), 0.0);
    assert_eq!(excess_kurtosis(&[1.0]), 0.0);
    assert_eq!(excess_kurtosis(&[3.0, 3.0, 3.0, 3.0]), 0.0);
}

#[test]
fn test_householder_apply_identity() {
    // Zero Householder vector → no reflection (identity)
    let h = [0.0f32; 4];
    let x = [1.0, 2.0, 3.0, 4.0];
    let result = householder_apply(&h, &x);
    for (r, &xi) in result.iter().zip(x.iter()) {
        assert!((r - xi).abs() < 1e-6, "Zero h should be identity");
    }
}

#[test]
fn test_householder_apply_reflection() {
    // h = e₁ (standard basis) → reflects x about the hyperplane orthogonal to e₁
    let h = [1.0f32, 0.0, 0.0, 0.0];
    let x = [1.0, 2.0, 3.0, 4.0];
    let result = householder_apply(&h, &x);
    // R = I - 2*e₁*e₁ᵀ → x → (x₀ - 2x₀, x₁, x₂, x₃)
    assert!(
        (result[0] - (-1.0)).abs() < 1e-6,
        "First component should be negated"
    );
    assert!((result[1] - 2.0).abs() < 1e-6);
    assert!((result[2] - 3.0).abs() < 1e-6);
    assert!((result[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_householder_apply_preserves_norm() {
    // Householder reflections are orthogonal → preserve norm
    let h = [0.5f32, 0.5, 0.5, 0.5];
    let x = [1.0, 2.0, 3.0, 4.0];
    let result = householder_apply(&h, &x);

    let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_r: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm_x - norm_r).abs() < 1e-4,
        "Norm should be preserved: {norm_x} vs {norm_r}"
    );
}

#[test]
fn test_vocab_project_basic() {
    // lm_head = [[1, 0], [0, 1], [1, 1]] → vocab_size=3, n_embd=2
    let lm_head = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let neuron_weight = [2.0f32, 3.0];

    let logits = vocab_project(&neuron_weight, &lm_head, 3, 2);

    assert_eq!(logits.len(), 3);
    assert!((logits[0] - 2.0).abs() < 1e-6, "token 0: 1*2 + 0*3 = 2");
    assert!((logits[1] - 3.0).abs() < 1e-6, "token 1: 0*2 + 1*3 = 3");
    assert!((logits[2] - 5.0).abs() < 1e-6, "token 2: 1*2 + 1*3 = 5");
}

#[test]
fn test_vocab_project_zeros() {
    let lm_head = [1.0f32, 2.0, 3.0, 4.0];
    let neuron_weight = [0.0f32, 0.0];
    let logits = vocab_project(&neuron_weight, &lm_head, 2, 2);
    assert!((logits[0]).abs() < 1e-10);
    assert!((logits[1]).abs() < 1e-10);
}

#[test]
fn test_iterative_token_mask_basic() {
    // Values: [1, 2, 3, 100] → mean ≈ 26.5, σ ≈ 43.1, k=1.5 → threshold ≈ 64.7
    // 100 is outlier, rest are within range
    let logits = [1.0f32, 2.0, 3.0, 100.0];
    let mut mask = [false; 4];
    iterative_token_mask(&logits, &mut mask, 1.5);

    assert!(mask[3], "100.0 should be masked as outlier");
    assert!(!mask[0], "1.0 should not be masked");
    assert!(!mask[1], "2.0 should not be masked");
    assert!(!mask[2], "3.0 should not be masked");
}

#[test]
fn test_iterative_token_mask_preserves_existing() {
    let logits = [1.0f32, 2.0, 3.0, 4.0];
    let mut mask = [false, true, false, false]; // token 1 already masked
    iterative_token_mask(&logits, &mut mask, 0.5);

    assert!(mask[1], "Existing mask should be preserved");
}

#[test]
fn test_iterative_token_mask_uniform() {
    // All same → σ = 0 → no masking
    let logits = [5.0f32; 10];
    let mut mask = [false; 10];
    iterative_token_mask(&logits, &mut mask, 2.0);
    assert!(mask.iter().all(|m| !m), "Uniform should not mask anything");
}

#[test]
fn test_topk_indices_basic() {
    let values = [3.0f32, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0, 5.0, 3.5];
    let top = topk_indices(&values, 3);
    assert_eq!(top.len(), 3);
    assert_eq!(top[0], 4, "Highest value at index 4 (9.0)");
    assert_eq!(top[1], 6, "Second highest at index 6 (6.0)");
    assert_eq!(top[2], 7, "Third highest at index 7 (5.0)");
}

#[test]
fn test_topk_indices_larger_than_input() {
    let values = [1.0f32, 2.0, 3.0];
    let top = topk_indices(&values, 10);
    assert_eq!(top.len(), 3);
}

#[test]
fn test_topk_indices_empty() {
    let values: [f32; 0] = [];
    let top = topk_indices(&values, 5);
    assert!(top.is_empty());
}

#[test]
fn test_cosine_sim_identical() {
    let a = [1.0f32, 2.0, 3.0];
    assert!((cosine_sim(&a, &a) - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_sim_orthogonal() {
    let a = [1.0f32, 0.0];
    let b = [0.0f32, 1.0];
    assert!(cosine_sim(&a, &b).abs() < 1e-6);
}

#[test]
fn test_cosine_sim_opposite() {
    let a = [1.0f32, 0.0];
    let b = [-1.0f32, 0.0];
    assert!((cosine_sim(&a, &b) - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_sigmoid() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    assert!(sigmoid(10.0) > 0.99);
    assert!(sigmoid(-10.0) < 0.01);
}

// ── Phase 2 Tests ──

#[test]
fn test_decompose_neuron_discovers_channels() {
    // Create a neuron weight that clearly points to token 2
    // n_embd = 4, vocab_size = 4
    // lm_head: identity matrix — token t → unit vector e_t
    let lm_head: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, // token 0
        0.0, 1.0, 0.0, 0.0, // token 1
        0.0, 0.0, 1.0, 0.0, // token 2
        0.0, 0.0, 0.0, 1.0, // token 3
    ];

    // Neuron weight pointing strongly at token 2
    let neuron_weight = [0.01f32, 0.01, 10.0, 0.01];

    // Verify the raw projection
    let logits = vocab_project(&neuron_weight, &lm_head, 4, 4);
    assert!(
        (logits[2] - 10.0).abs() < 1e-6,
        "Token 2 should have logit 10.0"
    );

    let config = VocabChannelConfig {
        max_channels: 3,
        top_k_tokens: 4,
        kurtosis_threshold: -10.0, // Accept any channel
        lambda: 0.001,
        eta: 0.005,
        max_iterations: 20,
        sigma_mask: 5.0,
        coords_per_iter: 4,
        fd_epsilon: 1e-3,
    };

    let decomposer = VocabChannelDecomposer::new(config);
    let channels = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);

    assert!(!channels.is_empty(), "Should discover at least one channel");
    let first = &channels[0];
    assert!(
        !first.top_tokens.is_empty(),
        "Channel should have top tokens"
    );
    // The first channel should contain token 2 as the dominant token
    assert_eq!(
        first.top_tokens[0], 2,
        "Token 2 should be the top token, got {:?}",
        first.top_tokens
    );
}

/// Issue 809 regression — `decompose_neuron` must be a FUNCTION of its input.
///
/// The Householder symmetry-breaker used `fastrand`'s unseeded thread-local
/// global, so the answer differed on every process. That is not merely untidy
/// in a repo whose gates are fixture hashes and bit-identity: it made
/// `test_decompose_neuron_discovers_channels` above flaky, and it was caught
/// by the SAME commit passing on one run of the x86_64 matrix and failing on
/// the next (`Token 2 should be the top token, got [3, 2, 1, 0]` — two tokens
/// tie and the perturbation decided the order).
///
/// Two calls inside one process are the strongest assertion this test can
/// make on its own; the cross-process half is the sibling test staying green
/// run after run, which is what a seed derived from the input buys.
#[test]
fn test_decompose_neuron_is_deterministic() {
    let lm_head: Vec<f32> = (0..16).map(|i| if i == 10 { 1.0 } else { 0.05 }).collect();
    let neuron_weight = [0.02f32, 0.31, 7.5, 0.02];
    let config = VocabChannelConfig {
        max_channels: 3,
        top_k_tokens: 4,
        kurtosis_threshold: -10.0,
        lambda: 0.001,
        eta: 0.005,
        max_iterations: 20,
        sigma_mask: 5.0,
        coords_per_iter: 4,
        fd_epsilon: 1e-3,
    };
    let decomposer = VocabChannelDecomposer::new(config);

    let a = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);
    let b = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);

    assert_eq!(a.len(), b.len(), "channel COUNT differs between two calls");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.top_tokens, y.top_tokens, "channel {i}: top_tokens differ");
        // Bit-identity, not a tolerance: same input, same arithmetic, same
        // order — a tolerance here would pass on a re-seeded run.
        assert_eq!(
            x.direction.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            y.direction.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
            "channel {i}: direction is not bit-identical"
        );
        assert_eq!(
            x.kurtosis.to_bits(),
            y.kurtosis.to_bits(),
            "channel {i}: kurtosis"
        );
        assert_eq!(
            x.skewness.to_bits(),
            y.skewness.to_bits(),
            "channel {i}: skewness"
        );
    }

    // A DIFFERENT neuron must not collapse onto the same stream — otherwise a
    // constant seed would pass the assertions above and say nothing.
    let other = [0.02f32, 0.31, 7.5, 0.03];
    let c = decomposer.decompose_neuron(&other, &lm_head, 4, 4);
    let same_stream = a.len() == c.len()
        && a.iter()
            .zip(c.iter())
            .all(|(x, y)| x.direction == y.direction);
    assert!(
        !same_stream,
        "a different neuron produced an identical decomposition"
    );
}

#[test]
fn test_decompose_neuron_kurtosis_threshold() {
    let lm_head: Vec<f32> = (0..4)
        .flat_map(|t| {
            let mut row = vec![0.0f32; 4];
            row[t] = 1.0;
            row
        })
        .collect();

    let neuron_weight = [1.0f32, 1.0, 1.0, 1.0];

    let config = VocabChannelConfig {
        max_channels: 5,
        kurtosis_threshold: 100.0, // Very high threshold → should reject all
        ..Default::default()
    };

    let decomposer = VocabChannelDecomposer::new(config);
    let channels = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);
    assert!(
        channels.is_empty(),
        "Should not discover channels with high threshold"
    );
}

#[test]
fn test_decompose_layer_channels() {
    let n_embd = 4;
    let mlp_hidden = 3;
    let vocab_size = 6;

    // mlp_w2: [n_embd, mlp_hidden] row-major
    let mlp_w2: Vec<f32> = vec![
        // row 0 (embd dim 0)
        1.0, 0.0, 0.5, // row 1 (embd dim 1)
        0.0, 1.0, 0.5, // row 2 (embd dim 2)
        0.0, 0.0, 1.0, // row 3 (embd dim 3)
        0.0, 0.0, 0.0,
    ];

    // lm_head: [vocab_size, n_embd]
    let lm_head: Vec<f32> = (0..vocab_size)
        .flat_map(|t| {
            let mut row = vec![0.0f32; n_embd];
            if t < n_embd {
                row[t] = 1.0;
            }
            row
        })
        .collect();

    let config = VocabChannelConfig {
        max_channels: 2,
        top_k_tokens: 3,
        kurtosis_threshold: -10.0,
        ..Default::default()
    };

    let result =
        decompose_layer_channels(&mlp_w2, &lm_head, n_embd, mlp_hidden, vocab_size, &config);

    assert_eq!(
        result.len(),
        mlp_hidden,
        "Should have one token set per neuron"
    );
    for (i, tokens) in result.iter().enumerate() {
        // Each set should be sorted
        for window in tokens.windows(2) {
            assert!(
                window[0] < window[1],
                "Token set for neuron {i} should be sorted"
            );
        }
    }
}

// ── Phase 3 Tests ──

#[test]
fn test_channel_map_from_channels() {
    let channels_per_neuron = vec![
        vec![
            vec![1, 3, 5], // neuron 0
            vec![2, 4, 6], // neuron 1
            vec![1, 2, 7], // neuron 2
        ],
        vec![
            vec![0, 1, 2], // layer 1, neuron 0
            vec![3, 4, 5], // layer 1, neuron 1
        ],
    ];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    assert_eq!(map.layer_count(), 2);
    assert_eq!(map.neuron_count(0), 3);
    assert_eq!(map.neuron_count(1), 2);

    assert!(map.is_reachable(0, 0, 3));
    assert!(map.is_reachable(0, 2, 7));
    assert!(!map.is_reachable(0, 0, 2));
    assert!(map.is_reachable(1, 0, 0));
}

#[test]
fn test_channel_map_neuron_tokens() {
    let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4]]];
    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);

    assert_eq!(map.neuron_tokens(0, 0), &[1, 3, 5]);
    assert_eq!(map.neuron_tokens(0, 1), &[2, 4]);
    let empty: &[usize] = &[];
    assert_eq!(map.neuron_tokens(0, 99), empty);
    assert_eq!(map.neuron_tokens(99, 0), empty);
}

#[test]
fn test_channel_map_layer_union() {
    let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 3, 6], vec![1, 7]]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let union = map.layer_union(0);

    assert_eq!(union, vec![1, 2, 3, 5, 6, 7]);
}

#[test]
fn test_channel_map_global_union() {
    let channels_per_neuron = vec![vec![vec![1, 3], vec![5]], vec![vec![2, 3], vec![7]]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let union = map.global_union();

    assert_eq!(union, vec![1, 2, 3, 5, 7]);
}

#[test]
fn test_channel_map_roundtrip_serialization() {
    let channels_per_neuron = vec![
        vec![vec![1, 3, 5], vec![2, 4, 6, 8], vec![0, 7]],
        vec![vec![10, 20, 30], vec![15, 25]],
    ];

    let original = VocabChannelMap::from_channels(&channels_per_neuron, 32000);
    let bytes = original.serialize();
    let restored = VocabChannelMap::deserialize(&bytes)
        .expect("deserialization should succeed")
        .with_vocab_size(32000);

    assert_eq!(restored.layer_count(), original.layer_count());
    assert_eq!(restored.neuron_count(0), original.neuron_count(0));
    assert_eq!(restored.neuron_count(1), original.neuron_count(1));

    for layer in 0..original.layer_count() {
        for neuron in 0..original.neuron_count(layer) {
            assert_eq!(
                restored.neuron_tokens(layer, neuron),
                original.neuron_tokens(layer, neuron),
                "Mismatch at layer={layer} neuron={neuron}"
            );
        }
    }
}

#[test]
fn test_channel_map_deserialize_too_short() {
    let result = VocabChannelMap::deserialize(&[1, 2, 3]);
    assert!(result.is_err());
}

#[test]
fn test_channel_map_empty_serialization() {
    let original = VocabChannelMap::from_channels(&[], 0);
    let bytes = original.serialize();
    let restored = VocabChannelMap::deserialize(&bytes).expect("should succeed");
    assert_eq!(restored.layer_count(), 0);
}

// ── Phase 4 Tests ──

#[test]
fn test_pruner_basic() {
    let channels_per_neuron = vec![vec![
        vec![1, 3, 5], // neuron 0: tokens 1,3,5
        vec![2, 4, 6], // neuron 1: tokens 2,4,6
    ]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let pruner = VocabChannelPruner::new(map);

    // Set active context: layer 0, neuron 0 active
    pruner.set_active_context(0, &[0]);

    assert!(pruner.is_valid(0, 1, &[]), "Token 1 reachable by neuron 0");
    assert!(pruner.is_valid(0, 3, &[]), "Token 3 reachable by neuron 0");
    assert!(
        !pruner.is_valid(0, 2, &[]),
        "Token 2 NOT reachable by neuron 0"
    );
    assert!(
        !pruner.is_valid(0, 7, &[]),
        "Token 7 NOT reachable by any neuron 0"
    );
}

#[test]
fn test_pruner_multiple_neurons() {
    let channels_per_neuron = vec![vec![
        vec![1, 3, 5], // neuron 0
        vec![2, 4, 6], // neuron 1
    ]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let pruner = VocabChannelPruner::new(map);

    // Both neurons active
    pruner.set_active_context(0, &[0, 1]);

    assert!(pruner.is_valid(0, 1, &[]), "Token 1 reachable by neuron 0");
    assert!(pruner.is_valid(0, 2, &[]), "Token 2 reachable by neuron 1");
    assert!(!pruner.is_valid(0, 7, &[]), "Token 7 NOT reachable");
}

#[test]
fn test_pruner_fallback_to_union() {
    let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4, 6]]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let pruner = VocabChannelPruner::new(map);

    // No neurons set → should use per-layer union
    // Union of neuron 0 and 1: {1, 2, 3, 4, 5, 6}
    assert!(pruner.is_valid(0, 1, &[]), "Token 1 in union");
    assert!(pruner.is_valid(0, 4, &[]), "Token 4 in union");
    assert!(!pruner.is_valid(0, 7, &[]), "Token 7 NOT in union");
    assert!(!pruner.is_valid(0, 0, &[]), "Token 0 NOT in union");
}

#[test]
fn test_pruner_unknown_layer() {
    let map = VocabChannelMap::from_channels(&[], 10);
    let pruner = VocabChannelPruner::new(map);

    // Unknown layer → don't prune (return true)
    assert!(
        pruner.is_valid(0, 42, &[]),
        "Unknown layer should not prune"
    );
}

#[test]
fn test_pruner_batch_is_valid() {
    let channels_per_neuron = vec![vec![vec![1, 3, 5]]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let pruner = VocabChannelPruner::new(map);

    pruner.set_active_context(0, &[0]);

    let candidates = [1, 2, 3, 4, 5, 6];
    let mut results = [false; 6];
    pruner.batch_is_valid(0, &candidates, &[], &mut results);

    assert!(results[0], "Token 1 should be valid");
    assert!(!results[1], "Token 2 should be invalid");
    assert!(results[2], "Token 3 should be valid");
    assert!(!results[3], "Token 4 should be invalid");
    assert!(results[4], "Token 5 should be valid");
    assert!(!results[5], "Token 6 should be invalid");
}

#[test]
fn test_pruner_batch_is_valid_union_fallback() {
    let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4]]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let pruner = VocabChannelPruner::new(map);

    // No active neurons → fallback to union {1, 2, 3, 4, 5}
    let candidates = [0, 1, 2, 3, 4, 5, 6];
    let mut results = [false; 7];
    pruner.batch_is_valid(0, &candidates, &[], &mut results);

    assert!(!results[0], "Token 0 NOT in union");
    assert!(results[1], "Token 1 in union");
    assert!(results[2], "Token 2 in union");
    assert!(results[3], "Token 3 in union");
    assert!(results[4], "Token 4 in union");
    assert!(results[5], "Token 5 in union");
    assert!(!results[6], "Token 6 NOT in union");
}

#[test]
fn test_pruner_is_valid_with_neurons() {
    let channels_per_neuron = vec![vec![vec![1, 3, 5], vec![2, 4, 6]]];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let pruner = VocabChannelPruner::new(map);

    assert!(pruner.is_valid_with_neurons(0, &[0], 3));
    assert!(!pruner.is_valid_with_neurons(0, &[0], 2));
    assert!(pruner.is_valid_with_neurons(0, &[0, 1], 2));
    assert!(!pruner.is_valid_with_neurons(0, &[0, 1], 7));
}

// ── Integration / Larger Tests ──

#[test]
fn test_full_pipeline_small() {
    // Simulate a tiny model: 2 layers, 3 neurons each, vocab=8, n_embd=4
    let n_embd = 4;
    let mlp_hidden = 3;
    let vocab_size = 8;

    // lm_head: [8, 4] — each token gets a unique direction
    let lm_head: Vec<f32> = (0..vocab_size)
        .flat_map(|t| {
            let mut row = vec![0.1f32; n_embd];
            if t < n_embd {
                row[t] = 5.0; // Strong signal for tokens 0-3
            }
            row
        })
        .collect();

    let config = VocabChannelConfig {
        max_channels: 2,
        top_k_tokens: 4,
        kurtosis_threshold: -10.0,
        lambda: 0.01,
        eta: 0.01,
        max_iterations: 5,
        sigma_mask: 3.0,
        coords_per_iter: 2,
        fd_epsilon: 1e-3,
    };

    // Layer 0
    let mlp_w2_layer0: Vec<f32> = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    let layer0_tokens = decompose_layer_channels(
        &mlp_w2_layer0,
        &lm_head,
        n_embd,
        mlp_hidden,
        vocab_size,
        &config,
    );

    assert_eq!(layer0_tokens.len(), mlp_hidden);
    for tokens in &layer0_tokens {
        // Verify sorted
        for window in tokens.windows(2) {
            assert!(window[0] < window[1]);
        }
    }

    // Build map
    let channels_per_neuron = vec![layer0_tokens];
    let map = VocabChannelMap::from_channels(&channels_per_neuron, vocab_size);
    let pruner = VocabChannelPruner::new(map);

    assert_eq!(pruner.layer_count(), 1);
}

#[test]
fn test_serialization_roundtrip_large() {
    // Larger map with multiple layers
    let channels_per_neuron: Vec<Vec<Vec<usize>>> = (0..4)
        .map(|layer| {
            (0..10)
                .map(|neuron| {
                    let base = layer * 100 + neuron * 10;
                    (0..5).map(|i| base + i).collect()
                })
                .collect()
        })
        .collect();

    let original = VocabChannelMap::from_channels(&channels_per_neuron, 1000);
    let bytes = original.serialize();

    // Verify serialized size is reasonable
    let expected_min = 4 + 4 * (4 + 10 * (4 + 5 * 4)); // header + 4 layers × (neurons + tokens)
    assert!(
        bytes.len() >= expected_min,
        "Serialized size {} should be at least {}",
        bytes.len(),
        expected_min
    );

    let restored = VocabChannelMap::deserialize(&bytes)
        .expect("deserialization should succeed")
        .with_vocab_size(1000);

    // Full verification
    for layer in 0..original.layer_count() {
        for neuron in 0..original.neuron_count(layer) {
            assert_eq!(
                restored.neuron_tokens(layer, neuron),
                original.neuron_tokens(layer, neuron)
            );
        }
    }
}

#[test]
fn test_apply_mask() {
    let mut logits = [1.0f32, 2.0, 3.0, 4.0];
    let mask = [false, true, false, true];
    apply_mask(&mut logits, &mask);
    assert_eq!(logits[0], 1.0);
    assert_eq!(logits[1], 0.0);
    assert_eq!(logits[2], 3.0);
    assert_eq!(logits[3], 0.0);
}

#[test]
fn test_vocabulary_config_default() {
    let config = VocabChannelConfig::default();
    assert_eq!(config.max_channels, 5);
    assert_eq!(config.top_k_tokens, 50);
    assert_eq!(config.kurtosis_threshold, 1.0);
    assert!(config.lambda > 0.0);
    assert!(config.eta > 0.0);
    assert!(config.max_iterations > 0);
}

#[test]
fn test_pruner_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VocabChannelPruner>();
}

#[test]
fn test_pruner_context_switching() {
    let channels_per_neuron = vec![
        vec![vec![1, 2], vec![3, 4]], // layer 0
        vec![vec![5, 6], vec![7, 8]], // layer 1
    ];

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 10);
    let pruner = VocabChannelPruner::new(map);

    // Layer 0, neuron 0
    pruner.set_active_context(0, &[0]);
    assert!(pruner.is_valid(0, 1, &[]));
    assert!(!pruner.is_valid(0, 3, &[]));

    // Switch to layer 1, neuron 1
    pruner.set_active_context(1, &[1]);
    assert!(!pruner.is_valid(0, 1, &[]));
    assert!(pruner.is_valid(0, 7, &[]));
}

#[test]
fn test_decompose_neuron_zero_weight() {
    let lm_head: Vec<f32> = (0..4)
        .flat_map(|t| {
            let mut row = vec![0.0f32; 4];
            row[t] = 1.0;
            row
        })
        .collect();

    let neuron_weight = [0.0f32; 4];
    let config = VocabChannelConfig {
        max_channels: 2,
        kurtosis_threshold: -100.0,
        ..Default::default()
    };

    let decomposer = VocabChannelDecomposer::new(config);
    let channels = decomposer.decompose_neuron(&neuron_weight, &lm_head, 4, 4);

    // Zero weight → zero logits → kurtosis = 0 → may or may not produce channels
    // Just verify it doesn't panic
    let _ = channels;
}

// ── Benchmark ──

#[test]
fn test_bench_vocab_project_v1000() {
    let vocab_size = 1000;
    let n_embd = 256;
    let lm_head: Vec<f32> = (0..vocab_size * n_embd)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let neuron_weight: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.01).cos()).collect();

    let start = std::time::Instant::now();
    let iters = 1_000;
    for _ in 0..iters {
        std::hint::black_box(vocab_project(&neuron_weight, &lm_head, vocab_size, n_embd));
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_nanos() as f64 / iters as f64;
    eprintln!("vocab_project V=1000 d=256: {per_call:.0}ns/call");
    assert!(
        per_call < 15_000_000.0,
        "vocab_project V=1000 should be <15ms (debug), got {per_call:.0}ns"
    );
}

#[test]
fn test_bench_householder_apply_d256() {
    let d = 256;
    let h: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();
    let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.2).cos()).collect();

    let start = std::time::Instant::now();
    let iters = 10_000;
    for _ in 0..iters {
        std::hint::black_box(householder_apply(&h, &x));
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_nanos() as f64 / iters as f64;
    eprintln!("householder_apply d=256: {per_call:.0}ns/call");
    assert!(
        per_call < 100_000.0,
        "householder_apply d=256 should be <100μs, got {per_call:.0}ns"
    );
}

#[test]
fn test_bench_pruner_is_valid() {
    let channels_per_neuron: Vec<Vec<Vec<usize>>> = (0..1)
        .map(|_| {
            (0..100)
                .map(|n| {
                    let mut tokens: Vec<usize> = (0..50).map(|i| n * 50 + i).collect();
                    tokens.sort();
                    tokens
                })
                .collect()
        })
        .collect();

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 5000);
    let pruner = VocabChannelPruner::new(map);

    pruner.set_active_context(0, &[0, 1, 2]);

    let start = std::time::Instant::now();
    let iters = 100_000;
    for i in 0..iters {
        std::hint::black_box(pruner.is_valid(0, i % 5000, &[]));
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_nanos() as f64 / iters as f64;
    eprintln!("VocabChannelPruner::is_valid: {per_call:.0}ns/call");
    assert!(
        per_call < 10_000.0,
        "is_valid should be <10μs, got {per_call:.0}ns"
    );
}

#[test]
fn test_bench_batch_is_valid() {
    let channels_per_neuron: Vec<Vec<Vec<usize>>> = (0..1)
        .map(|_| {
            (0..100)
                .map(|n| {
                    let mut tokens: Vec<usize> = (0..50).map(|i| n * 50 + i).collect();
                    tokens.sort();
                    tokens
                })
                .collect()
        })
        .collect();

    let map = VocabChannelMap::from_channels(&channels_per_neuron, 5000);
    let pruner = VocabChannelPruner::new(map);

    pruner.set_active_context(0, &[0, 1, 2]);

    let candidates: Vec<usize> = (0..1000).collect();
    let mut results = vec![false; 1000];

    let start = std::time::Instant::now();
    let iters = 10_000;
    for _ in 0..iters {
        pruner.batch_is_valid(0, &candidates, &[], &mut results);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_nanos() as f64 / iters as f64;
    eprintln!("VocabChannelPruner::batch_is_valid V=1000: {per_call:.0}ns/call");
    assert!(
        per_call < 1_000_000.0,
        "batch_is_valid V=1000 should be <1ms, got {per_call:.0}ns"
    );
}

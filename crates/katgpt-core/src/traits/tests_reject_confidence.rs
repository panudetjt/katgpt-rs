use super::*;

/// Mock binary pruner: rejects any token_idx >= threshold.
struct ThresholdPruner {
    threshold: usize,
}

impl ConstraintPruner for ThresholdPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx < self.threshold
    }
}

/// GOAT G1-T1: default `reject_confidence()` reproduces `is_valid()` exactly.
/// valid → 0.0, invalid → 1.0. Every existing implementor is unchanged.
#[test]
fn default_reject_confidence_reproduces_is_valid() {
    let p = ThresholdPruner { threshold: 5 };
    // Valid tokens map to 0.0 confidence
    for tok in 0..5 {
        let valid = p.is_valid(0, tok, &[]);
        let conf = p.reject_confidence(0, tok, &[]);
        assert!(valid, "token {tok} should be valid");
        assert_eq!(
            conf, 0.0,
            "valid token {tok} should have 0.0 reject confidence"
        );
    }
    // Invalid tokens map to 1.0 confidence
    for tok in 5..10 {
        let valid = p.is_valid(0, tok, &[]);
        let conf = p.reject_confidence(0, tok, &[]);
        assert!(!valid, "token {tok} should be invalid");
        assert_eq!(
            conf, 1.0,
            "invalid token {tok} should have 1.0 reject confidence"
        );
    }
}

/// GOAT G1-T1 (NoPruner variant): the no-op pruner reports 0.0 everywhere.
#[test]
fn no_pruner_reject_confidence_is_zero() {
    let p = NoPruner;
    for tok in [0usize, 1, 42, 1000] {
        assert_eq!(p.reject_confidence(0, tok, &[]), 0.0);
    }
}

/// Default `batch_reject_confidence` mirrors per-item `reject_confidence`.
#[test]
fn default_batch_reject_confidence_mirrors_per_item() {
    let p = ThresholdPruner { threshold: 5 };
    let candidates: Vec<usize> = (0..10).collect();
    let mut batch = vec![-1.0f32; 10];
    p.batch_reject_confidence(0, &candidates, &[], &mut batch);
    for (i, &tok) in candidates.iter().enumerate() {
        let individual = p.reject_confidence(0, tok, &[]);
        assert_eq!(
            batch[i], individual,
            "token {tok}: batch={}, per-item={}",
            batch[i], individual
        );
    }
}

/// Batch handles mismatched lengths gracefully (writes only min(candidates, results) entries).
#[test]
fn batch_reject_confidence_truncates_to_min_len() {
    let p = ThresholdPruner { threshold: 3 };
    let candidates = [0usize, 1, 2, 3, 4];
    // Results buffer smaller than candidates — must not panic, writes first 3.
    let mut results = [f32::NAN; 3];
    p.batch_reject_confidence(0, &candidates, &[], &mut results);
    assert_eq!(results, [0.0, 0.0, 0.0]);
}

/// Determinism contract: identical inputs yield identical outputs.
#[test]
fn reject_confidence_is_deterministic() {
    let p = ThresholdPruner { threshold: 5 };
    let a = p.reject_confidence(2, 7, &[1, 3]);
    let b = p.reject_confidence(2, 7, &[1, 3]);
    assert_eq!(a, b, "identical inputs must yield identical outputs");
}

/// Range contract: output always in [0.0, 1.0] for the default binary impl.
#[test]
fn reject_confidence_in_unit_range() {
    let p = ThresholdPruner { threshold: 5 };
    for tok in 0..20 {
        let c = p.reject_confidence(0, tok, &[]);
        assert!((0.0..=1.0).contains(&c), "token {tok}: {c} out of [0,1]");
    }
}

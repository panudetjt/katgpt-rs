use super::*;

/// Mock implementation to verify trait compiles and is object-safe.
struct MockGen;

impl SpeculativeGenerator for MockGen {
    type Condition = ();
    type Output = usize;
    type Error = ();

    fn generate(
        &mut self,
        _condition: &Self::Condition,
        _rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        Ok(vec![1, 2, 3])
    }
}

/// Mock pruner to verify GenerativeConstraintPruner compiles.
struct MockPruner;

impl GenerativeConstraintPruner<usize> for MockPruner {
    fn is_valid(&self, output: &usize) -> bool {
        *output > 0
    }
}

#[test]
fn test_speculative_generator_trait_bounds() {
    let mut generator = MockGen;
    let mut rng = fastrand::Rng::new();
    let result = generator.generate(&(), &mut rng).unwrap();
    assert_eq!(result, vec![1, 2, 3]);

    // Batch uses default impl
    let batch = generator.generate_batch(&[(), (), ()], &mut rng).unwrap();
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0], vec![1, 2, 3]);
}

#[test]
fn test_generative_constraint_pruner_trait_bounds() {
    let pruner = MockPruner;
    assert!(pruner.is_valid(&1));
    assert!(!pruner.is_valid(&0));

    let results = pruner.batch_is_valid(&[0, 1, 2]);
    assert_eq!(results, vec![false, true, true]);
}

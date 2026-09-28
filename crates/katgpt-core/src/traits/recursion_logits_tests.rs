use super::*;

/// A synthetic recursion-capable generator that captures pre/post logits.
pub struct TestRecursionGenerator {
    pre_logits: Vec<f32>,
    post_logits: Vec<f32>,
}

impl TestRecursionGenerator {
    pub fn new() -> Self {
        Self {
            pre_logits: vec![1.0, 0.5, -0.5, -1.0],
            post_logits: vec![1.0, 0.5, -0.5, -1.0],
        }
    }

    /// Simulate a recursion step that sharpens logits toward index 0.
    pub fn sharpen_step(&mut self) {
        // Snapshot pre BEFORE the step.
        self.pre_logits = self.post_logits.clone();
        // Step: move 50% closer to a target that favors index 0.
        let target = [3.0, 0.0, -1.0, -2.0];
        for (l, &t) in self.post_logits.iter_mut().zip(target.iter()) {
            *l = 0.5 * *l + 0.5 * t;
        }
    }
}

impl SpeculativeGenerator for TestRecursionGenerator {
    type Condition = ();
    type Output = Vec<f32>;
    type Error = std::convert::Infallible;

    fn generate(
        &mut self,
        _condition: &Self::Condition,
        _rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        self.sharpen_step();
        Ok(vec![self.post_logits.clone()])
    }
}

impl RecursionLogits for TestRecursionGenerator {
    fn pre_recursion_logits(&self) -> &[f32] {
        &self.pre_logits
    }
    fn post_recursion_logits(&self) -> &[f32] {
        &self.post_logits
    }
}

#[test]
fn generator_exposes_pre_post_logits() {
    let mut g = TestRecursionGenerator::new();
    g.sharpen_step();

    let pre = g.pre_recursion_logits();
    let post = g.post_recursion_logits();
    assert_eq!(pre.len(), 4);
    assert_eq!(post.len(), 4);
    // Post sharpened toward index 0 (target 3.0), pre did not have that boost.
    assert!(
        post[0] > pre[0],
        "post[0]={} should exceed pre[0]={}",
        post[0],
        pre[0]
    );
}

#[test]
fn trait_consumer_can_read_logits_via_trait_object() {
    // Proves a gate consumer can read both logits via the trait — the
    // whole point of T2.3. The actual AdvantageMarginGate lives in the
    // root crate; here we inline the margin math to prove the trait works.
    fn compute_margin_sign(generator: &dyn RecursionLogits, candidate: usize) -> f32 {
        let pre = generator.pre_recursion_logits();
        let post = generator.post_recursion_logits();
        assert_eq!(pre.len(), post.len());
        // Inline advantage margin: A(candidate) - mean(A).
        // A(a) ≈ post[a] - pre[a] (skip log-softmax for the test).
        let n = pre.len() as f32;
        let a_cand = post[candidate] - pre[candidate];
        let mean_a: f32 = (0..pre.len()).map(|i| post[i] - pre[i]).sum::<f32>() / n;
        a_cand - mean_a
    }

    let mut g = TestRecursionGenerator::new();
    g.sharpen_step();
    let margin = compute_margin_sign(&g, 0);
    // Index 0 is the candidate being sharpened toward, so margin should be positive.
    assert!(
        margin > 0.0,
        "sharpened candidate must have positive margin, got {margin}"
    );
}

#[test]
fn empty_logits_when_no_recursion_occurred() {
    // A generator that has not recursed yet may return empty slices.
    // The trait explicitly allows this.
    struct NoRecursionYet;
    impl RecursionLogits for NoRecursionYet {
        fn pre_recursion_logits(&self) -> &[f32] {
            &[]
        }
        fn post_recursion_logits(&self) -> &[f32] {
            &[]
        }
    }
    let g = NoRecursionYet;
    assert_eq!(g.pre_recursion_logits().len(), 0);
    assert_eq!(g.post_recursion_logits().len(), 0);
}

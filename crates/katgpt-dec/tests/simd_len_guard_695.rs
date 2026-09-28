//! Bench 695 soundness gate, katgpt-dec's copy.
//!
//! `katgpt-dec` ships its own `simd_dot_f32` (zero-dep by design, to avoid a
//! cyclic package dep with katgpt-core), so it carried the same OOB-read hole
//! as the katgpt-types family and needs its own pin.
use katgpt_dec::simd::simd_dot_f32;

#[test]
#[should_panic(expected = "out of range")]
fn dot_rejects_len_past_end() {
    let (a, b) = ([1.0f32; 4], [1.0f32; 4]);
    simd_dot_f32(&a, &b, 4096);
}

#[test]
fn short_len_is_a_valid_prefix_request() {
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b = vec![1.0f32; 64];
    assert_eq!(
        simd_dot_f32(&a, &b, 8),
        (0..8).map(|i| i as f32).sum::<f32>()
    );
}

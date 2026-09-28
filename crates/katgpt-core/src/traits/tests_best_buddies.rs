use super::*;

#[test]
fn test_pearson_perfect_correlation() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 4.0, 6.0, 8.0, 10.0]; // b = 2*a
    let corr = pearson_correlation(&a, &b);
    assert!((corr - 1.0).abs() < 1e-6, "expected 1.0, got {corr}");
}

#[test]
fn test_pearson_anti_correlation() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0, 8.0, 6.0, 4.0, 2.0]; // b = -2*a + 12
    let corr = pearson_correlation(&a, &b);
    assert!((corr + 1.0).abs() < 1e-6, "expected -1.0, got {corr}");
}

#[test]
fn test_pearson_zero_correlation() {
    let a = [1.0, -1.0, 1.0, -1.0];
    let b = [1.0, 1.0, -1.0, -1.0]; // orthogonal
    let corr = pearson_correlation(&a, &b);
    assert!(corr.abs() < 1e-6, "expected ~0.0, got {corr}");
}

#[test]
fn test_pearson_zero_variance() {
    let a = [3.0, 3.0, 3.0];
    let b = [1.0, 2.0, 3.0];
    let corr = pearson_correlation(&a, &b);
    assert_eq!(corr, 0.0, "zero variance should return 0.0");
}

#[test]
fn test_pearson_empty() {
    let corr = pearson_correlation(&[], &[]);
    assert_eq!(corr, 0.0, "empty slices should return 0.0");
}

#[test]
fn test_best_buddies_simple() {
    // Row 0 best match → 1, Row 1 best match → 0 → mutual pair (0,1)
    // Row 2 best match → 0, but Row 0's best is 1 → not mutual
    let row0: &[f32] = &[0.1, 0.9, 0.2];
    let row1: &[f32] = &[0.8, 0.1, 0.3];
    let row2: &[f32] = &[0.7, 0.2, 0.1];
    let rows: Vec<&[f32]> = vec![row0, row1, row2];
    let buddies = best_buddies(&rows, 10);
    assert_eq!(buddies, vec![(0, 1)]);
}

#[test]
fn test_best_buddies_top_k() {
    let row0: &[f32] = &[0.1, 0.9];
    let row1: &[f32] = &[0.8, 0.1];
    let rows: Vec<&[f32]> = vec![row0, row1];
    // k=1 should truncate to 1 result
    let buddies = best_buddies(&rows, 1);
    assert_eq!(buddies.len(), 1);
    assert_eq!(buddies[0], (0, 1));
}

#[test]
fn test_best_buddies_no_mutual() {
    // Row 0 → 1, Row 1 → 2, Row 2 → 0 (cycle, no mutual)
    let row0: &[f32] = &[0.1, 0.9, 0.2];
    let row1: &[f32] = &[0.1, 0.2, 0.9];
    let row2: &[f32] = &[0.9, 0.1, 0.2];
    let rows: Vec<&[f32]> = vec![row0, row1, row2];
    let buddies = best_buddies(&rows, 10);
    assert!(buddies.is_empty(), "cycle should produce no mutual pairs");
}

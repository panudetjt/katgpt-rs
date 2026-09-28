//! Issue 779 T2 — the FUNCATTN spectral arm: the three-arm
//! subspace-intervention triad on `spectral_pre_rotate`'s REAL calibrated
//! eigenbasis vs a random basis at matched param budget — the eval that
//! `funcattn_compose/spectral_pre_rotate.rs` deferred ("Not a GOAT gate":
//! proving the *more expressive per parameter* hypothesis needed a
//! composition-specific benchmark; Issue 778 closed it at protocol level
//! only, on a planted probe basis).
//!
//! Geometry (the SpectralQuant hypothesis, planted): calibration samples
//! carry an anisotropic spectrum whose TOP eigendirections hold the
//! class-mean task signal. The eigen-aligned k-subspace must then retain
//! the task signal at k = task rank while a random k-subspace at the SAME
//! budget (same k rows, same d) retains only ≈k/d of it — the
//! aligned-vs-random CONTRAST is the discriminating signal (a random
//! k-subspace does not collapse to chance: it is an SNR-preserving cut).
//!
//! Run: `cargo test -p katgpt-attn --features
//! funcattn_spectral_pre_rotate --test funcattn_spectral_triad -- --nocapture`

#![cfg(feature = "funcattn_spectral_pre_rotate")]

use katgpt_core::subspace_intervention::{
    InterventionRng, InterventionScratch, ridge_probe_fit_into, three_arm_eval_on_basis,
};
use katgpt_spectral::calibrate_eigenbasis;

const D: usize = 32;
const C: usize = 4;
const TASK_RANK: usize = 2; // class means live on the top-2 eigendirections
const N_TRAIN: usize = 96;
const N_TEST: usize = 96;
const N_TOTAL: usize = N_TRAIN + N_TEST;
const N_CAL: usize = 256; // calibration samples for the eigenbasis
const CHANCE: f32 = 1.0 / C as f32;

/// Planted geometry: direction `j` carries variance `s[j]` (steep decay);
/// the class means live on directions 0..TASK_RANK (the top-2). Returns
/// (features, labels, calibration_samples).
fn planted(seed: u64, s: [f32; 4]) -> (Vec<f32>, Vec<usize>, Vec<Vec<f32>>) {
    let mut rng = InterventionRng::new(seed);
    let var = |j: usize| s[j.min(s.len() - 1)];
    let scale = |j: usize| var(j).sqrt();
    let mut feats = vec![0.0_f32; N_TOTAL * D];
    let mut labels = Vec::with_capacity(N_TOTAL);
    // class-mean pattern on the top directions: 4 classes from 2 signed axes
    let mean_pat = [[1.0_f32, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]];
    let mean_amp = 2.0_f32;
    for i in 0..N_TOTAL {
        let c = (i / 2) % C; // parity split sees every class (anti-alias)
        labels.push(c);
        for j in 0..TASK_RANK {
            feats[i * D + j] = mean_amp * mean_pat[c][j] + scale(j) * rng.next_gaussian();
        }
        for j in TASK_RANK..D {
            feats[i * D + j] = scale(j) * rng.next_gaussian();
        }
    }
    let mut cal = Vec::with_capacity(N_CAL);
    for _ in 0..N_CAL {
        let mut v = vec![0.0_f32; D];
        for (j, vj) in v.iter_mut().enumerate() {
            *vj = scale(j) * rng.next_gaussian();
        }
        cal.push(v);
    }
    (feats, labels, cal)
}

#[test]
fn funcattn_eigenbasis_beats_random_at_matched_budget() {
    let (feats, labels, cal) = planted(0x5EED_0779, [4.0, 2.0, 0.5, 0.25]);

    // The REAL production eigenbasis path (what spectral_pre_rotate calls).
    let cal_res = calibrate_eigenbasis(&cal, D);
    // Gather the top-k eigenvector COLUMNS (row-major source, stride D)
    // into the protocol's contiguous column-major basis layout.
    let rank_cap = TASK_RANK.clamp(4, D);
    let mut basis = vec![0.0_f32; D * rank_cap];
    for j in 0..rank_cap {
        for a in 0..D {
            basis[j * D + a] = cal_res.eigenvectors[a * D + j];
        }
    }

    // Train/test split by parity; probe on the train half.
    let mut xt = Vec::with_capacity(N_TRAIN * D);
    let mut xe = Vec::with_capacity(N_TEST * D);
    let mut yt = Vec::with_capacity(N_TRAIN);
    let mut ye = Vec::with_capacity(N_TEST);
    for i in 0..N_TOTAL {
        if i % 2 == 1 {
            xt.extend_from_slice(&feats[i * D..(i + 1) * D]);
            yt.push(labels[i]);
        } else {
            xe.extend_from_slice(&feats[i * D..(i + 1) * D]);
            ye.push(labels[i]);
        }
    }
    let mut scratch = InterventionScratch::new(D, C, N_TRAIN, N_TEST);
    let mut w = vec![0.0_f32; C * D];
    ridge_probe_fit_into(&xt, &yt, N_TRAIN, D, C, 0.01, &mut scratch, &mut w);
    let mut recall = vec![0.0_f32; C];
    let full = scratch.eval_into(&xe, &ye, N_TEST, D, &w, C, &mut recall);

    // The triad at matched budgets k ∈ {2 (task rank), 4}.
    let ks = [TASK_RANK, 4];
    let mut al = vec![0.0_f32; ks.len()];
    let mut ra = vec![0.0_f32; ks.len()];
    let mut re = vec![0.0_f32; ks.len()];
    three_arm_eval_on_basis(
        &xe,
        &ye,
        N_TEST,
        D,
        &w,
        C,
        &ks,
        &basis,
        rank_cap,
        0xDEAD_BEEF,
        &mut scratch,
        &mut al,
        &mut ra,
        &mut re,
        &mut recall,
    );

    println!("FUNCATTN spectral arm (Issue 779 T2): full={full:.3} chance={CHANCE:.3}");
    for (ki, &k) in ks.iter().enumerate() {
        println!(
            "  k={k}: eigen-aligned={:.3} random={:.3} residual={:.3}",
            al[ki], ra[ki], re[ki]
        );
    }

    // The deferred hypothesis, positive arm: at k = task rank the
    // eigen-aligned basis retains the task signal decisively better than
    // the random basis at the same budget, and the residual arm (the
    // complement) shows the signal really lives in the eigen-subspace.
    assert!(
        al[0] >= ra[0] + 0.25,
        "eigen-aligned@k={} must beat random decisively: {:.3} vs {:.3}",
        ks[0],
        al[0],
        ra[0]
    );
    assert!(
        al[0] >= 0.85 * full,
        "eigen-aligned@k={} must retain the task signal: {:.3} vs full {:.3}",
        ks[0],
        al[0],
        full
    );
    assert!(
        re[0] <= CHANCE + 0.10,
        "residual@k={} must collapse toward chance: {:.3}",
        ks[0],
        re[0]
    );
}

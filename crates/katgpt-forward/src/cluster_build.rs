//! Load-time construction of the clustered-LM-head artifacts (Plan 574).
//!
//! [`clustered_lm_head`](crate::forward::clustered_lm_head) has shipped wired
//! into both forward call sites since Plan 117, but never ran in production
//! because nothing ever produced its two inputs — `mtp_cluster_classifier` and
//! `mtp_cluster_map` were always `None`. This module is the producer.
//!
//! # Why the centroid is the right classifier
//!
//! `logit[t] = dot(hidden, lm_head[t])`. Defining cluster `c`'s classifier row
//! as the centroid of its members' LM-head rows gives
//!
//! ```text
//! dot(hidden, centroid_c) = mean(logit[t] for t in c)
//! ```
//!
//! so the stage-1 score *is* the cluster's mean logit — a principled proxy for
//! "does the argmax live in here". K-means over the LM-head rows minimises
//! within-cluster variance, which is exactly what tightens that proxy.
//!
//! Both steps are deterministic functions of weights the model already ships:
//! **no gradients, no training**. This is a modelless primitive.
//!
//! # Why `lm_head`, not `wte`
//!
//! The stub this replaces took `wte`. The matmul being pruned is the LM head,
//! so `lm_head` is the matrix whose geometry matters. With tied embeddings the
//! two coincide; with untied embeddings `wte` is simply the wrong matrix.
//!
//! # Why the rows are projected first
//!
//! Naive Lloyd assignment costs `O(vocab × k × n_embd)` per iteration — at Qwen
//! scale (`vocab=152k, k=2048, n_embd=5120`) that is ~1.6e12 FLOPs *per
//! iteration*. A deterministic Johnson–Lindenstrauss projection to
//! [`PROJ_DIM`] preserves relative distances well enough for clustering while
//! cutting assignment to ~2e10. Centroids for the classifier are then computed
//! in the **full** space, so the emitted artifact is exact.

use katgpt_core::simd::simd_dot_f32;

/// Dimension the LM-head rows are projected into before k-means.
///
/// 64 keeps Johnson–Lindenstrauss distortion low for the cluster counts used
/// here while making assignment ~80× cheaper than clustering in full `n_embd`.
const PROJ_DIM: usize = 64;

/// Lloyd iterations. Cluster assignment converges fast in projected space;
/// beyond ~10 the map stops changing while cost keeps growing linearly.
const KMEANS_ITERS: usize = 10;

/// Fixed seed for the projection. Pinned so the emitted cluster map is
/// reproducible across runs and machines — a rebuilt map that differed run to
/// run would make the argmax-recall gate unfalsifiable.
const PROJ_SEED: u64 = 0x5EED_C105_7E12_0001;

/// Separate pinned stream for the k-means++ seeding draw. Distinct from
/// [`PROJ_SEED`] so the projection and the initial-centre choice do not
/// correlate.
const INIT_SEED: u64 = 0x5EED_C105_7E12_0002;

/// splitmix64 finalizer — deterministic, well-distributed bit mixing.
#[inline]
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build the `[proj_dim, n_embd]` ±1 Johnson–Lindenstrauss projection.
///
/// Materialised once (≈1.3 MB at Qwen scale) rather than hashing per element,
/// so the projection itself runs as SIMD dot products.
fn build_projection(n_embd: usize, proj_dim: usize) -> Vec<f32> {
    let mut proj = vec![0.0f32; proj_dim * n_embd];
    for (idx, slot) in proj.iter_mut().enumerate() {
        let bits = splitmix64(PROJ_SEED ^ (idx as u64));
        *slot = match bits & 1 {
            0 => -1.0,
            _ => 1.0,
        };
    }
    proj
}

/// Project every row of `rows` into `proj_dim` dimensions.
fn project_rows(rows: &[f32], count: usize, n_embd: usize, proj_dim: usize) -> Vec<f32> {
    let proj = build_projection(n_embd, proj_dim);
    let mut out = vec![0.0f32; count * proj_dim];
    for t in 0..count {
        let row = &rows[t * n_embd..(t + 1) * n_embd];
        for j in 0..proj_dim {
            let basis = &proj[j * n_embd..(j + 1) * n_embd];
            out[t * proj_dim + j] = simd_dot_f32(row, basis, n_embd);
        }
    }
    out
}

/// Squared L2 distance between two equal-length slices.
#[inline]
fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
}

/// Deterministic k-means++ (D²) seeding in projected space.
///
/// # Why not strided init (the defect this replaces)
///
/// The original init took rows `0, stride, 2·stride, …` with
/// `stride = count / k`. That is only "spread out" in *token-ID* order, which
/// is not the geometry being clustered. When the vocabulary's geometric groups
/// are periodic in token ID with a period that shares a factor with `stride`,
/// every initial centre lands in the same handful of groups and Lloyd cannot
/// recover from it. Benchmark 657's structured fixture assigns group
/// `t % n_groups` with `n_groups == k`, so `stride = count / k` made all `k`
/// centres come from exactly **two** distinct groups — a pathological start
/// that Plan 574's G2b failure was originally (and wrongly) attributed to the
/// scoring objective.
///
/// D² seeding picks centres by actual distance, so it cannot be defeated by an
/// adversarial ID ordering. It is deterministic given [`INIT_SEED`]; cost is
/// `O(k · count · dim)`, i.e. one extra Lloyd iteration.
fn kmeanspp_init(data: &[f32], count: usize, k: usize, dim: usize) -> Vec<f32> {
    let mut centers = vec![0.0f32; k * dim];
    let mut closest = vec![f32::MAX; count];
    let mut stream = INIT_SEED;

    let next_u64 = |stream: &mut u64| {
        *stream = stream.wrapping_add(1);
        splitmix64(*stream)
    };

    let first = (next_u64(&mut stream) % count as u64) as usize;
    centers[..dim].copy_from_slice(&data[first * dim..(first + 1) * dim]);

    for c in 1..k {
        // Fold the centre just chosen into the running nearest-centre distance.
        let prev = &centers[(c - 1) * dim..c * dim];
        let mut total = 0.0f64;
        for (t, slot) in closest.iter_mut().enumerate() {
            *slot = slot.min(sq_dist(&data[t * dim..(t + 1) * dim], prev));
            total += *slot as f64;
        }

        // All remaining points coincide with a chosen centre — nothing left to
        // separate, so fall back to a deterministic index rather than sampling
        // from an all-zero distribution.
        let chosen = if total > 0.0 {
            let target = total * (next_u64(&mut stream) >> 11) as f64 / (1u64 << 53) as f64;
            let mut acc = 0.0f64;
            let mut pick = count - 1;
            for (t, &d) in closest.iter().enumerate() {
                acc += d as f64;
                if acc >= target {
                    pick = t;
                    break;
                }
            }
            pick
        } else {
            c % count
        };
        centers[c * dim..(c + 1) * dim].copy_from_slice(&data[chosen * dim..(chosen + 1) * dim]);
    }
    centers
}

/// How k-means picks its initial centres.
///
/// Exposed because the choice is not a tuning detail — it moved Benchmark 657's
/// verdict on its own. Production always wants [`ClusterInit::Dsquared`];
/// [`ClusterInit::Strided`] is retained solely so the benchmark can attribute
/// the recall change to the init rather than to the scoring bound.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterInit {
    /// Deterministic k-means++ D² seeding. See [`kmeanspp_init`].
    #[default]
    Dsquared,
    /// Rows `0, stride, 2·stride, …`. Spread in **token-ID** order, not in the
    /// geometry being clustered — degenerate whenever the vocabulary's groups
    /// are ID-periodic. Kept only as the measured-worse baseline.
    Strided,
}

/// Strided (token-ID-spaced) initial centres — the defective baseline.
fn strided_init(data: &[f32], count: usize, k: usize, dim: usize) -> Vec<f32> {
    let stride = count / k;
    let mut centers = vec![0.0f32; k * dim];
    for c in 0..k {
        let src = (c * stride).min(count - 1);
        centers[c * dim..(c + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
    }
    centers
}

/// Create a round-robin cluster assignment for tokens.
///
/// Token `i` is assigned to cluster `i / cluster_size`. Deterministic, no
/// training needed — the simple baseline that
/// [`cluster_map_from_embeddings`] must beat on argmax recall to be worth
/// shipping (Plan 574 gate G2).
pub fn cluster_map_round_robin(vocab_size: usize, cluster_size: usize) -> Vec<Vec<usize>> {
    if cluster_size == 0 {
        return Vec::new();
    }
    let num_clusters = vocab_size.div_ceil(cluster_size);
    let mut map: Vec<Vec<usize>> = (0..num_clusters)
        .map(|_| Vec::with_capacity(cluster_size))
        .collect();
    for token_id in 0..vocab_size {
        map[token_id / cluster_size].push(token_id);
    }
    map
}

/// Group tokens by LM-head-row similarity (k-means in projected space).
///
/// `lm_head` is `[vocab_size, n_embd]` row-major. Returns one `Vec<usize>` of
/// token IDs per non-empty cluster; every token in `0..vocab_size` appears in
/// exactly one cluster. Deterministic for a given input — see [`PROJ_SEED`].
///
/// Cluster *count* matches [`cluster_map_round_robin`]'s convention
/// (`vocab_size.div_ceil(cluster_size)`) so the two are directly comparable at
/// equal `topk`. Cluster *sizes* will not be uniform — that is the point.
///
/// Empty clusters are dropped, so the result may contain fewer clusters than
/// requested. Build the classifier from the returned map (not from the
/// requested count) so the two stay consistent —
/// [`cluster_classifier_from_map`] does this by construction.
pub fn cluster_map_from_embeddings(
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
    cluster_size: usize,
) -> Vec<Vec<usize>> {
    cluster_map_from_embeddings_with_init(
        lm_head,
        vocab_size,
        n_embd,
        cluster_size,
        ClusterInit::default(),
    )
}

/// [`cluster_map_from_embeddings`] with an explicit seeding strategy.
///
/// Only the benchmark should pass anything other than [`ClusterInit::Dsquared`].
pub fn cluster_map_from_embeddings_with_init(
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
    cluster_size: usize,
    init: ClusterInit,
) -> Vec<Vec<usize>> {
    // Degenerate shapes fall back to the baseline rather than erroring: the
    // caller gets a usable map, and round-robin is exactly what k-means would
    // reduce to when there is no geometry to exploit.
    if cluster_size == 0 || vocab_size == 0 || n_embd == 0 {
        return cluster_map_round_robin(vocab_size, cluster_size);
    }
    if lm_head.len() < vocab_size * n_embd {
        return cluster_map_round_robin(vocab_size, cluster_size);
    }

    let k = vocab_size.div_ceil(cluster_size);
    // One cluster per token (or fewer tokens than clusters) leaves nothing to
    // group; round-robin already yields the optimal partition.
    if k <= 1 || k >= vocab_size {
        return cluster_map_round_robin(vocab_size, cluster_size);
    }

    let proj_dim = PROJ_DIM.min(n_embd);
    let data = project_rows(lm_head, vocab_size, n_embd, proj_dim);

    // Constant projected geometry (every row identical — an all-zero or
    // otherwise untrained `wte`) leaves k-means nothing to separate: every
    // centre coincides with the data, and the strict `<` in the assignment
    // step collapses all tokens into cluster 0, yielding one mega-cluster
    // instead of the baseline map the degenerate-input contract promises
    // above. Round-robin is what k-means reduces to with no geometry to
    // exploit, so return it directly. Identical input rows project to
    // identical results (same dot products over the same bytes), so the
    // bitwise slice comparison is exact.
    if data[proj_dim..]
        .chunks_exact(proj_dim)
        .all(|row| row == &data[..proj_dim])
    {
        return cluster_map_round_robin(vocab_size, cluster_size);
    }

    // See `kmeanspp_init` for why the strided variant is pathological on
    // ID-periodic geometry.
    let mut centers = match init {
        ClusterInit::Dsquared => kmeanspp_init(&data, vocab_size, k, proj_dim),
        ClusterInit::Strided => strided_init(&data, vocab_size, k, proj_dim),
    };

    let mut labels = vec![0usize; vocab_size];
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * proj_dim];

    for _ in 0..KMEANS_ITERS {
        // Assignment
        for t in 0..vocab_size {
            let point = &data[t * proj_dim..(t + 1) * proj_dim];
            let mut best = 0usize;
            let mut best_dist = f32::MAX;
            for c in 0..k {
                let dist = sq_dist(point, &centers[c * proj_dim..(c + 1) * proj_dim]);
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            labels[t] = best;
        }

        // Update
        sums.fill(0.0);
        counts.fill(0);
        for t in 0..vocab_size {
            let c = labels[t];
            counts[c] += 1;
            let point = &data[t * proj_dim..(t + 1) * proj_dim];
            let sum_row = &mut sums[c * proj_dim..(c + 1) * proj_dim];
            for (s, p) in sum_row.iter_mut().zip(point) {
                *s += p;
            }
        }
        for c in 0..k {
            // An empty cluster keeps its previous center; re-seeding it would
            // break determinism guarantees for no measurable recall gain.
            match counts[c] {
                0 => {}
                n => {
                    let inv = 1.0 / n as f32;
                    let center_row = &mut centers[c * proj_dim..(c + 1) * proj_dim];
                    let sum_row = &sums[c * proj_dim..(c + 1) * proj_dim];
                    for (slot, s) in center_row.iter_mut().zip(sum_row) {
                        *slot = s * inv;
                    }
                }
            }
        }
    }

    let mut map: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (token_id, &c) in labels.iter().enumerate() {
        map[c].push(token_id);
    }
    map.retain(|cluster| !cluster.is_empty());
    map
}

/// Build the stage-1 classifier as per-cluster centroids of the LM-head rows.
///
/// Returns `[cluster_map.len(), n_embd]` row-major, matching the layout
/// [`clustered_lm_head`](crate::forward::clustered_lm_head) indexes as
/// `c * n_embd`. Centroids are computed in the **full** embedding space (the
/// projection is a clustering device only), so the score
/// `dot(hidden, centroid_c)` is exactly the cluster's mean logit.
#[allow(clippy::needless_range_loop)] // index-parallel over lm_head rows reads clearer here
pub fn cluster_classifier_from_map(
    lm_head: &[f32],
    cluster_map: &[Vec<usize>],
    n_embd: usize,
) -> Vec<f32> {
    let mut classifier = vec![0.0f32; cluster_map.len() * n_embd];
    for (c, tokens) in cluster_map.iter().enumerate() {
        if tokens.is_empty() {
            continue;
        }
        let row = &mut classifier[c * n_embd..(c + 1) * n_embd];
        for &t in tokens {
            let off = t * n_embd;
            if off + n_embd > lm_head.len() {
                continue;
            }
            for (slot, w) in row.iter_mut().zip(&lm_head[off..off + n_embd]) {
                *slot += w;
            }
        }
        let inv = 1.0 / tokens.len() as f32;
        for slot in row.iter_mut() {
            *slot *= inv;
        }
    }
    classifier
}

/// Per-cluster residual radius — `radius_c = max‖lm_head[t] − centroid_c‖`.
///
/// This is the third clustered-LM-head artifact (Issue 657), and the one that
/// turns stage 1 from a heuristic into an **admissible** bound. Writing
/// `w_t = centroid_c + r_t`, Cauchy–Schwarz gives, for every `t` in cluster `c`:
///
/// ```text
/// logit[t] = ⟨h, centroid_c⟩ + ⟨h, r_t⟩ ≤ ⟨h, centroid_c⟩ + ‖h‖ · radius_c
/// ```
///
/// The bare centroid score is the cluster's *mean* logit, but the question
/// stage 1 must answer is which cluster holds the *max*. A cluster with one
/// spike among many low values has a poor mean and is pruned despite owning the
/// argmax; the radius term is exactly the slack that covers that case.
///
/// Costs one `f32` per cluster and one FMA per cluster in stage 1. Computed
/// from shipped weights — deterministic, **modelless**.
///
/// `classifier` must be the output of [`cluster_classifier_from_map`] for the
/// same `cluster_map`, or the bound is not admissible.
pub fn cluster_radii_from_map(
    lm_head: &[f32],
    cluster_map: &[Vec<usize>],
    classifier: &[f32],
    n_embd: usize,
) -> Vec<f32> {
    let mut radii = vec![0.0f32; cluster_map.len()];
    for (c, tokens) in cluster_map.iter().enumerate() {
        let centroid_off = c * n_embd;
        if centroid_off + n_embd > classifier.len() {
            continue;
        }
        let centroid = &classifier[centroid_off..centroid_off + n_embd];
        let mut max_sq = 0.0f32;
        for &t in tokens {
            let off = t * n_embd;
            if off + n_embd > lm_head.len() {
                continue;
            }
            max_sq = max_sq.max(sq_dist(&lm_head[off..off + n_embd], centroid));
        }
        radii[c] = max_sq.sqrt();
    }
    radii
}

/// Cluster-contiguous LM-head layout (Issue 666).
///
/// Stage 2 of the clustered head reads whole clusters. With the natural layout
/// a cluster's members are scattered token IDs, so it gathers `len` separate
/// `n_embd`-float rows from across a 500 MB matrix. Issue 661 measured what that
/// costs: **20.6 GB/s against the full head's 108.0 GB/s**, a 5.26× locality
/// penalty that accounted for the entire gap between an 11.64× FLOP reduction
/// and a 2.21× wall-clock win.
///
/// Permuting the rows into cluster order once, at load time, turns each cluster
/// into a single contiguous span — the same access pattern the full head
/// already achieves 108 GB/s with.
pub struct ClusterLayout {
    /// `[vocab_size, n_embd]` — the LM-head rows, reordered so that every
    /// cluster occupies one contiguous span.
    pub permuted: Vec<f32>,
    /// Row → original token ID. Stage 2 computes into row order and scatters
    /// back through this.
    pub token_of_row: Vec<usize>,
    /// Per-cluster `(start_row, len)` into [`permuted`](Self::permuted) and
    /// [`token_of_row`](Self::token_of_row).
    ///
    /// Replaces the `Vec<Vec<usize>>` cluster map on the hot path: one
    /// allocation instead of `num_clusters`, and no pointer chase per cluster.
    pub offsets: Vec<(usize, usize)>,
}

/// Shape only — a derived `Debug` would dump the whole permuted matrix
/// (hundreds of MB at production scale) into a panic message.
impl std::fmt::Debug for ClusterLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterLayout")
            .field("rows", &self.token_of_row.len())
            .field("clusters", &self.offsets.len())
            .field("extra_bytes", &self.extra_bytes())
            .finish()
    }
}

impl ClusterLayout {
    /// Bytes the permuted copy occupies. The caller pays this on top of the
    /// original `lm_head`.
    #[must_use]
    pub fn extra_bytes(&self) -> usize {
        self.permuted.len() * size_of::<f32>()
    }
}

/// Why [`cluster_layout_from_map`] declined to build a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutRefusal {
    /// `lm_head` shares storage with `wte` (tied embeddings).
    ///
    /// The permuted copy **cannot alias** the embedding table — the whole point
    /// is that its rows are in a different order — so it is a genuine second
    /// allocation of the model's largest tensor. On a tied 2 B model that is a
    /// ~1 GB increase to save ~0.3 ms per token, which is a trade the caller
    /// must make deliberately rather than inherit from a default.
    TiedEmbeddings { extra_bytes: usize },
    /// Shape is degenerate (empty vocabulary/embedding, or a short buffer).
    DegenerateShape,
}

/// What to do when `lm_head` may be tied to `wte`.
///
/// The check is on **storage identity**, not content: it catches the case where
/// the two are the same buffer, which is what "tied" means at runtime. Two
/// separate allocations holding equal values are already paying for two copies,
/// so permuting one adds nothing new and is correctly allowed through.
#[derive(Clone, Copy)]
pub enum TiedPolicy<'a> {
    /// Compare against the model's `wte` and refuse if they share storage.
    /// This is the default a loader should use.
    Refuse { wte: &'a [f32] },
    /// The caller has measured the memory cost and accepts the second copy.
    Accept,
}

/// Build the cluster-contiguous layout for `cluster_map`.
///
/// Rows are emitted in cluster order, and within a cluster in the order the map
/// lists them, so the layout is a deterministic function of the map — rebuilds
/// are identical. Tokens outside `0..vocab_size` are dropped.
///
/// # Errors
///
/// Returns [`LayoutRefusal::TiedEmbeddings`] when `tied` is
/// [`TiedPolicy::Refuse`] and `lm_head` shares storage with the supplied `wte`,
/// and [`LayoutRefusal::DegenerateShape`] for empty or short inputs.
pub fn cluster_layout_from_map(
    lm_head: &[f32],
    cluster_map: &[Vec<usize>],
    vocab_size: usize,
    n_embd: usize,
    tied: TiedPolicy<'_>,
) -> Result<ClusterLayout, LayoutRefusal> {
    if vocab_size == 0 || n_embd == 0 || lm_head.len() < vocab_size * n_embd {
        return Err(LayoutRefusal::DegenerateShape);
    }

    let total_rows: usize = cluster_map
        .iter()
        .map(|c| c.iter().filter(|&&t| t < vocab_size).count())
        .sum();

    if let TiedPolicy::Refuse { wte } = tied {
        // Storage identity, not content equality: same base pointer and same
        // length is the runtime signature of a tied table.
        if std::ptr::eq(lm_head.as_ptr(), wte.as_ptr()) && lm_head.len() == wte.len() {
            return Err(LayoutRefusal::TiedEmbeddings {
                extra_bytes: total_rows * n_embd * size_of::<f32>(),
            });
        }
    }

    let mut permuted = vec![0.0f32; total_rows * n_embd];
    let mut token_of_row = vec![0usize; total_rows];
    let mut offsets = Vec::with_capacity(cluster_map.len());

    let mut row = 0usize;
    for tokens in cluster_map {
        let start = row;
        for &t in tokens.iter().filter(|&&t| t < vocab_size) {
            let src = t * n_embd;
            permuted[row * n_embd..(row + 1) * n_embd].copy_from_slice(&lm_head[src..src + n_embd]);
            token_of_row[row] = t;
            row += 1;
        }
        offsets.push((start, row - start));
    }

    Ok(ClusterLayout {
        permuted,
        token_of_row,
        offsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::{clustered_lm_head, standard_lm_head};

    /// Two well-separated groups: even token IDs at `+1`, odd at `-1`.
    ///
    /// Round-robin partitions by *index*, so every one of its clusters is a
    /// 50/50 mix of the two groups. K-means must recover the actual geometry.
    fn two_group_lm_head(vocab: usize, n_embd: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; vocab * n_embd];
        for t in 0..vocab {
            let sign = match t % 2 {
                0 => 1.0,
                _ => -1.0,
            };
            for j in 0..n_embd {
                w[t * n_embd + j] = sign * (1.0 + (j as f32) * 0.01);
            }
        }
        w
    }

    #[test]
    fn every_token_lands_in_exactly_one_cluster() {
        let (vocab, n_embd) = (128, 16);
        let w = two_group_lm_head(vocab, n_embd);
        let map = cluster_map_from_embeddings(&w, vocab, n_embd, 16);

        let mut seen = vec![0usize; vocab];
        for cluster in &map {
            for &t in cluster {
                seen[t] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "each token must appear exactly once; got {seen:?}"
        );
        assert!(
            map.iter().all(|c| !c.is_empty()),
            "empty clusters must be dropped"
        );
    }

    #[test]
    fn map_is_deterministic_across_builds() {
        let (vocab, n_embd) = (96, 16);
        let w = two_group_lm_head(vocab, n_embd);
        let a = cluster_map_from_embeddings(&w, vocab, n_embd, 12);
        let b = cluster_map_from_embeddings(&w, vocab, n_embd, 12);
        assert_eq!(a, b, "PROJ_SEED is pinned — rebuilds must be identical");
    }

    /// The quality claim: k-means separates the two groups, round-robin cannot.
    #[test]
    fn kmeans_separates_groups_that_round_robin_mixes() {
        let (vocab, n_embd) = (128, 16);
        let w = two_group_lm_head(vocab, n_embd);

        let purity = |map: &[Vec<usize>]| -> f64 {
            let mut pure = 0usize;
            for cluster in map {
                let evens = cluster.iter().filter(|t| *t % 2 == 0).count();
                pure += evens.max(cluster.len() - evens);
            }
            pure as f64 / vocab as f64
        };

        let km = purity(&cluster_map_from_embeddings(&w, vocab, n_embd, 16));
        let rr = purity(&cluster_map_round_robin(vocab, 16));
        assert!(
            km > rr,
            "k-means purity {km} must beat round-robin {rr} on grouped data"
        );
        assert!(km > 0.99, "k-means should near-perfectly separate: {km}");
    }

    #[test]
    fn classifier_row_is_cluster_centroid() {
        let (vocab, n_embd) = (8, 4);
        let mut w = vec![0.0f32; vocab * n_embd];
        for t in 0..vocab {
            for j in 0..n_embd {
                w[t * n_embd + j] = t as f32;
            }
        }
        // One cluster holding tokens 0..4 — centroid is mean(0,1,2,3) = 1.5.
        let map = vec![vec![0usize, 1, 2, 3]];
        let classifier = cluster_classifier_from_map(&w, &map, n_embd);
        for (j, centroid) in classifier.iter().enumerate() {
            assert!(
                (centroid - 1.5).abs() < 1e-6,
                "centroid[{j}] = {centroid}, want 1.5"
            );
        }
    }

    /// **Gate G1**: with `topk >= num_clusters` nothing is pruned, so the
    /// clustered head must reproduce the standard head bit-for-bit.
    #[test]
    fn g1_bit_identical_to_standard_head_when_nothing_is_pruned() {
        let (vocab, n_embd) = (64, 16);
        let w = two_group_lm_head(vocab, n_embd);
        let hidden: Vec<f32> = (0..n_embd).map(|j| 0.1 * (j as f32) - 0.5).collect();

        let map = cluster_map_from_embeddings(&w, vocab, n_embd, 8);
        let classifier = cluster_classifier_from_map(&w, &map, n_embd);

        let mut want = vec![0.0f32; vocab];
        standard_lm_head(&mut want, &hidden, &w, vocab, n_embd);

        let mut got = vec![0.0f32; vocab];
        let mut scores = vec![0.0f32; map.len()];
        let (mut idx_buf, mut out_buf) = (Vec::new(), Vec::new());
        clustered_lm_head(
            &mut got,
            &hidden,
            &w,
            &classifier,
            &map,
            vocab,
            n_embd,
            map.len(), // topk == num_clusters => no pruning
            &mut scores,
            &mut idx_buf,
            &mut out_buf,
        );

        assert_eq!(got, want, "no-pruning path must be bit-identical");
    }

    #[test]
    fn degenerate_shapes_fall_back_to_round_robin() {
        assert_eq!(
            cluster_map_from_embeddings(&[], 0, 16, 4),
            Vec::<Vec<usize>>::new()
        );
        assert!(cluster_map_from_embeddings(&[], 10, 16, 0).is_empty());
        // Short weight buffer must not panic — falls back.
        let short = vec![0.0f32; 4];
        assert_eq!(
            cluster_map_from_embeddings(&short, 100, 16, 25),
            cluster_map_round_robin(100, 25)
        );
    }

    #[test]
    fn constant_geometry_falls_back_to_round_robin() {
        // All-zero embeddings (the untrained-`wte` shape) and any constant
        // row set carry no geometry for k-means to exploit — the documented
        // contract returns the baseline map, not a single mega-cluster.
        // Regression: the strict-`<` assignment used to collapse these into
        // one cluster containing every token (riir-engine
        // `test_cluster_map_from_embeddings_fallback` has pinned this
        // contract since the mtp.rs de-fork, 3d69ccaf1).
        let zeros = vec![0.0f32; 100 * 32];
        assert_eq!(
            cluster_map_from_embeddings(&zeros, 100, 32, 25),
            cluster_map_round_robin(100, 25)
        );
        let constant = vec![0.5f32; 80 * 16];
        assert_eq!(
            cluster_map_from_embeddings(&constant, 80, 16, 20),
            cluster_map_round_robin(80, 20)
        );
    }
}

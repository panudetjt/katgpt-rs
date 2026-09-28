//! `genome` — the per-NPC spectral genome Pod (riir-ai Issue 736 Phase B3 /
//! Research 495 §5): a canonical-gauge, fixed-size, zero-copy serialized
//! pencil genome with deterministic seeding, elementwise-mean merge, the
//! Weyl health certificate, and a population-diversity metric.
//!
//! # The Pod
//!
//! [`GenomePod`] holds `format ‖ k ‖ canonical matrices` as a `#[repr(C)]`
//! plain-data struct — [`bytemuck::Pod`] (the `NeuronShard` pattern):
//! fixed-size, no heap, byte-addressable for freeze/thaw persistence. A
//! compile-time size assertion pins "no hidden padding" per instantiation.
//! Genomes are **born canonical** ([`from_pencil`] canonicalizes the gauge
//! before storing; merges re-canonicalize), so the Pod IS the canonical
//! form — same-input construction is deterministic per binary (the
//! pinned-solver determinism policy) and `decode` is an exact copy.
//!
//! # Seeding
//!
//! [`from_seed`] is the consumer contract's substrate half: the caller
//! derives seed bytes from identity material (the issue's spec:
//! `BLAKE3(npc_id ‖ world_seed)` — identity vocabulary stays consumer-side)
//! and gets a family-S [`init::seeded_dense`] genome with the
//! eigengap-≥-½-at-init guarantee. Shaped genomes (family M, rank-one/NSD
//! features) go through [`from_pencil`] + [`shape`].
//!
//! # Merge + the Weyl health certificate
//!
//! [`merge_mean`] averages two parent pencils elementwise (matrix-mean,
//! temperament rung `k` inherited from the primary parent) and
//! re-canonicalizes. With `A_avg(x) = (A₁(x)+A₂(x))/2`, Weyl gives
//! `|λj(A_avg) − λj(A₁)| ≤ ½‖A₂(x)−A₁(x)‖₂` for every `j`, so
//!
//! ```text
//! γk(avg) ≥ γk(p₁) − ‖A₂(x) − A₁(x)‖₂
//! ```
//!
//! — [`weyl_health_certificate`] evaluates both gaps at a belief `x` and
//! reports whether the merged genome's attribution-trust gap survived the
//! merge (the personality-analog of "merge did not collapse conditioning").
//!
//! # Population diversity
//!
//! [`population_diversity`] is the mean pairwise pencil Frobenius distance
//! over a population (all matrices, packed-representation-exact): a
//! homogeneous control measures exactly `0`; a seeded population measures
//! the personality spread (the Research 495 "population diversity ships as
//! a seeded generator instead of hand-tuned constant tables" claim, made
//! checkable).
//!
//! # Honest boundaries
//!
//! * Cross-**representation** byte-equality (two conjugated
//!   parametrizations hashing identically) remains a non-goal — same-input
//!   determinism is the contract (see [`gauge`]'s degenerate-block note).
//! * The Weyl certificate is per-`x` (a belief-state health readout), not
//!   a box-wide guarantee; `‖A₂(x)−A₁(x)‖₂` grows with `x` off neutral.
//! * Byte-level persistence is little-endian-native (the whole stack
//!   targets LE); cross-endian deserialization is out of scope.

use crate::spectral_pencil::DensePencil;
use crate::spectral_pencil::dense::DenseScratch;
use crate::spectral_pencil::gauge;
use crate::spectral_pencil::init::seeded_dense;
use crate::spectral_pencil::sym::SymPacked;

/// Format tag (checked at [`GenomePod::from_bytes`]).
pub const GENOME_POD_MAGIC: u32 = u32::from_le_bytes(*b"SGP1");

/// A canonical-gauge serialized spectral genome (fixed-size Pod).
///
/// Layout: `format (u32) ‖ k (u32) ‖ a₀ (D² f32) ‖ a₁..a_N (N·D² f32)`.
/// See the [module docs](self) for the seeding, merge, and certificate
/// contracts.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct GenomePod<const D: usize, const N: usize> {
    format: u32,
    k: u32,
    a0: SymPacked<D>,
    a: [SymPacked<D>; N],
}

// SAFETY: `u32 ‖ u32 ‖ [[f32; D]; D] ‖ [[f32; D]; D]; N` — all fields
// plain fixed-size f32/u32 data, `#[repr(C)]`, alignment 4 throughout, no
// padding (pinned by the `_NO_PADDING` compile-time assertion below).
unsafe impl<const D: usize, const N: usize> bytemuck::Pod for GenomePod<D, N> {}
unsafe impl<const D: usize, const N: usize> bytemuck::Zeroable for GenomePod<D, N> {}

/// Result of one Weyl health readout on a merged genome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeylCert {
    /// γk of the primary parent at `x`.
    pub gap_parent: f32,
    /// γk of the merged (mean) genome at `x`.
    pub gap_merged: f32,
    /// `‖A₂(x) − A₁(x)‖₂` — the Weyl perturbation budget.
    pub pencil_dist: f32,
    /// `gap_merged ≥ gap_parent − pencil_dist − eps` (eps = float slack).
    pub healthy: bool,
}

impl<const D: usize, const N: usize> GenomePod<D, N> {
    /// Compile-time no-padding pin: the struct is exactly the plain-data
    /// size (any hidden padding from a layout change fails THIS
    /// instantiation's compile).
    const _NO_PADDING: () = assert!(
        std::mem::size_of::<Self>() == 8 + (N + 1) * D * D * 4,
        "GenomePod must be padding-free plain data"
    );

    /// Serialized size (bytes).
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Canonicalize + store a pencil genome with temperament rung `k`
    /// (0-indexed). Deterministic per binary; `decode` recovers it exactly.
    #[must_use]
    pub fn from_pencil(pencil: &DensePencil<D, N>, k: usize) -> Self {
        let () = Self::_NO_PADDING;
        let mut scratch = DenseScratch::<D>::new();
        let (a0, a) = gauge::canonicalize(&pencil.a0, &pencil.a, &mut scratch);
        Self {
            format: GENOME_POD_MAGIC,
            k: k.min(D - 1) as u32,
            a0,
            a,
        }
    }

    /// Family-S seeded genome (the eigengap-≥-½-at-init constructor): the
    /// substrate half of the per-NPC contract — the caller derives the seed
    /// bytes from identity material (`BLAKE3(npc_id ‖ world_seed)` at the
    /// consumer).
    #[must_use]
    pub fn from_seed(seed_bytes: &[u8], k: usize) -> Self {
        let init = seeded_dense::<D, N>(seed_bytes, k);
        Self::from_pencil(
            &DensePencil {
                a0: init.a0,
                a: init.a,
            },
            k,
        )
    }

    /// Recover the canonical pencil + temperament rung (exact copy — no
    /// parsing). The neutral pencil on format mismatch (defensive; use
    /// [`from_bytes`](Self::from_bytes) for validated parsing).
    #[must_use]
    pub fn decode(&self) -> (DensePencil<D, N>, usize) {
        if self.format != GENOME_POD_MAGIC {
            return (
                DensePencil {
                    a0: SymPacked::zeroed(),
                    a: [SymPacked::zeroed(); N],
                },
                0,
            );
        }
        (self.pencil(), (self.k as usize).min(D - 1))
    }

    /// The stored canonical pencil.
    #[must_use]
    pub fn pencil(&self) -> DensePencil<D, N> {
        DensePencil {
            a0: self.a0,
            a: self.a,
        }
    }

    /// The temperament rung (0-indexed k).
    #[must_use]
    pub fn k(&self) -> usize {
        (self.k as usize).min(D - 1)
    }

    /// BLAKE3 commitment over the serialized canonical bytes.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        *blake3::hash(bytemuck::bytes_of(self)).as_bytes()
    }

    /// Zero-copy byte view (Pod).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Validated parse from raw bytes (length + magic checked).
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::SIZE {
            return None;
        }
        let pod: &Self = bytemuck::from_bytes(bytes);
        if pod.format != GENOME_POD_MAGIC {
            return None;
        }
        Some(*pod)
    }

    /// Elementwise-mean merge: `A_m = (A₁ + A₂)/2` matrix-wise, temperament
    /// rung inherited from `self` (the primary parent), re-canonicalized —
    /// the offspring is itself a canonical Pod. Deterministic given the
    /// parent Pods.
    #[must_use]
    pub fn merge_mean(&self, other: &Self) -> Self {
        let mean = |a: &SymPacked<D>, b: &SymPacked<D>| -> SymPacked<D> {
            let mut out = SymPacked::zeroed();
            for (ro, (ra, rb)) in out.data.iter_mut().zip(a.data.iter().zip(b.data.iter())) {
                for (o, (x, y)) in ro.iter_mut().zip(ra.iter().zip(rb.iter())) {
                    *o = (x + y) * 0.5;
                }
            }
            out
        };
        let merged = DensePencil {
            a0: mean(&self.a0, &other.a0),
            a: {
                let mut out = [SymPacked::<D>::zeroed(); N];
                for (m, (x, y)) in out.iter_mut().zip(self.a.iter().zip(other.a.iter())) {
                    *m = mean(x, y);
                }
                out
            },
        };
        Self::from_pencil(&merged, self.k())
    }

    /// Weyl health readout of a merge at belief `x`: with `A_avg =
    /// (A₁+A₂)/2`, Weyl gives `γk(avg) ≥ γk(p₁) − ‖A₂(x)−A₁(x)‖₂`. `self`
    /// is the MERGED genome, `parent1`/`parent2` the parents.
    #[must_use]
    pub fn weyl_health_certificate(
        &self,
        parent1: &Self,
        parent2: &Self,
        x: &[f32; N],
    ) -> WeylCert {
        let k = self.k();
        let avg = self.pencil();
        let p1 = parent1.pencil();
        let p2 = parent2.pencil();

        let mut scratch = DenseScratch::<D>::new();
        let gap_of = |pencil: &DensePencil<D, N>, scratch: &mut DenseScratch<D>| -> f32 {
            pencil.eval(x, k, scratch).eigengap.unwrap_or(f32::INFINITY)
        };
        let gap_parent = gap_of(&p1, &mut scratch);
        let gap_merged = gap_of(&avg, &mut scratch);

        // ‖A₂(x) − A₁(x)‖₂ — materialize both pencils at x, difference,
        // one Jacobi norm.
        let mut s1 = DenseScratch::<D>::new();
        let mut s2 = DenseScratch::<D>::new();
        p1.materialize(x, &mut s1);
        p2.materialize(x, &mut s2);
        let mut diff = [[0.0_f32; D]; D];
        for (dr, (r1, r2)) in diff.iter_mut().zip(s1.a.iter().zip(s2.a.iter())) {
            for (d, (a, b)) in dr.iter_mut().zip(r1.iter().zip(r2.iter())) {
                *d = a - b;
            }
        }
        let pencil_dist = crate::spectral_pencil::bounds::norm_jacobi_exact(
            &SymPacked::pack_from_full(&diff),
            &mut scratch,
        );

        let healthy = gap_merged + 1e-4 >= gap_parent - pencil_dist;
        WeylCert {
            gap_parent,
            gap_merged,
            pencil_dist,
            healthy,
        }
    }

    /// Mean pairwise pencil Frobenius distance over the population (all
    /// `N+1` matrices; the packed representation makes this the plain
    /// Frobenius distance). Homogeneous control → exactly `0`.
    #[must_use]
    pub fn population_diversity(pods: &[Self]) -> f32 {
        if pods.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0_f64;
        let mut pairs = 0_usize;
        for i in 0..pods.len() {
            for j in (i + 1)..pods.len() {
                let mut dd = 0.0_f64;
                for (mi, mj) in std::iter::once(&pods[i].a0)
                    .chain(pods[i].a.iter())
                    .zip(std::iter::once(&pods[j].a0).chain(pods[j].a.iter()))
                {
                    for (ri, rj) in mi.data.iter().zip(mj.data.iter()) {
                        for (a, b) in ri.iter().zip(rj.iter()) {
                            let d = f64::from(a - b);
                            dd += d * d;
                        }
                    }
                }
                total += dd.sqrt();
                pairs += 1;
            }
        }
        (total / pairs as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 8;
    const N: usize = 4;

    /// Round-trip: decode(from_seed) recovers the stored canonical pencil
    /// exactly, and from_bytes(as_bytes) is the identity.
    #[test]
    fn round_trip_exact() {
        let pod = GenomePod::<D, N>::from_seed(b"genome-rt", 3);
        let (pencil, k) = pod.decode();
        assert_eq!(k, 3);
        assert_eq!(pencil.a0.data, pod.pencil().a0.data);
        let pod2 = GenomePod::<D, N>::from_bytes(pod.as_bytes()).unwrap();
        assert_eq!(pod2.pencil().a0.data, pencil.a0.data);
        for (m1, m2) in pencil.a.iter().zip(pod2.pencil().a.iter()) {
            assert_eq!(m1.data, m2.data, "Pod round-trip must be exact");
        }
        assert_eq!(pod.commitment(), pod2.commitment());
    }

    /// Determinism: same seed → same Pod bytes; different seed → different.
    #[test]
    fn determinism_and_distinction() {
        let a = GenomePod::<D, N>::from_seed(b"genome-det", 2);
        let b = GenomePod::<D, N>::from_seed(b"genome-det", 2);
        assert_eq!(a.as_bytes(), b.as_bytes());
        let c = GenomePod::<D, N>::from_seed(b"genome-det-other", 2);
        assert_ne!(a.as_bytes(), c.as_bytes());
        // k is stored + distinguished.
        let d = GenomePod::<D, N>::from_seed(b"genome-det", 5);
        assert_ne!(a.as_bytes(), d.as_bytes());
        assert_eq!(d.decode().1, 5);
    }

    /// from_bytes rejects wrong length + wrong magic.
    #[test]
    fn from_bytes_validates() {
        let pod = GenomePod::<D, N>::from_seed(b"genome-parse", 1);
        assert!(GenomePod::<D, N>::from_bytes(&pod.as_bytes()[..10]).is_none());
        let mut corrupted = pod;
        bytemuck::bytes_of_mut(&mut corrupted)[0] = b'X';
        assert!(GenomePod::<D, N>::from_bytes(corrupted.as_bytes()).is_none());
        assert!(GenomePod::<D, N>::from_bytes(pod.as_bytes()).is_some());
    }

    /// Merge + Weyl health: the certificate relation holds across seeded
    /// parent pairs × belief states (0 violations), and merge is
    /// deterministic.
    #[test]
    fn merge_weyl_certificate_holds() {
        let mut violations = 0_u32;
        for s in 0..10_u64 {
            let p1 = GenomePod::<D, N>::from_seed(format!("gw-p1-{s}").as_bytes(), 3);
            let p2 = GenomePod::<D, N>::from_seed(format!("gw-p2-{s}").as_bytes(), 3);
            let merged = p1.merge_mean(&p2);

            // Determinism.
            assert_eq!(
                merged.as_bytes(),
                p1.merge_mean(&p2).as_bytes(),
                "merge must be deterministic"
            );

            // k inherited from the primary parent.
            assert_eq!(merged.k(), 3);

            for t in 0..6_u64 {
                let x = [((s * 7 + t) % 5) as f32, ((s * 3 + t) % 5) as f32, 2.5, 1.5];
                let cert = merged.weyl_health_certificate(&p1, &p2, &x);
                if !cert.healthy {
                    violations += 1;
                    eprintln!(
                        "WEYL VIOLATION s={s} t={t}: parent {} merged {} dist {}",
                        cert.gap_parent, cert.gap_merged, cert.pencil_dist
                    );
                }
            }
        }
        assert_eq!(
            violations, 0,
            "Weyl health certificate violated {violations} times — merge math is wrong"
        );
    }

    /// Population diversity: a seeded population is diverse; the
    /// homogeneous control measures exactly 0.
    #[test]
    fn population_diversity_seeded_vs_homogeneous() {
        let seeded: Vec<GenomePod<D, N>> = (0..24_u64)
            .map(|i| GenomePod::from_seed(format!("div-{i}").as_bytes(), 3))
            .collect();
        let diverse = GenomePod::population_diversity(&seeded);
        assert!(
            diverse > 0.1,
            "seeded population diversity {diverse} should be well above 0"
        );

        let control = vec![seeded[0]; 8];
        let homogeneous = GenomePod::population_diversity(&control);
        assert_eq!(homogeneous, 0.0, "homogeneous control must measure 0");
    }

    /// The consumer seed contract (Issue 736 B3): seed = BLAKE3(npc_id ‖
    /// world_seed) — different NPC ids produce different genomes; the same
    /// id reproduces the identical genome.
    #[test]
    fn consumer_seed_contract() {
        let world = b"world-736";
        let seed_for = |npc_id: u64| -> Vec<u8> {
            let mut h = blake3::Hasher::new();
            h.update(&npc_id.to_le_bytes());
            h.update(world);
            h.finalize().as_bytes().to_vec()
        };
        let a = GenomePod::<D, N>::from_seed(&seed_for(1), 7);
        let b = GenomePod::<D, N>::from_seed(&seed_for(2), 7);
        assert_ne!(a.commitment(), b.commitment());
        // Same NPC + world → identical genome (reproducible population).
        assert_eq!(
            a.as_bytes(),
            GenomePod::<D, N>::from_seed(&seed_for(1), 7).as_bytes()
        );
    }

    /// Size sanity: the PoC temperament shape packs into a fixed-size Pod
    /// with the compile-time no-padding pin satisfied.
    #[test]
    fn size_is_fixed_and_packed() {
        assert_eq!(GenomePod::<8, 4>::SIZE, 8 + 5 * 8 * 8 * 4);
        assert_eq!(
            std::mem::size_of::<GenomePod<8, 4>>(),
            GenomePod::<8, 4>::SIZE
        );
    }
}

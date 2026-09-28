//! Generalized Landau-Lifshitz-Gilbert recall flow on `CP^(d-1)`.
//!
//! Where [`CpHopfieldRecaller::recall_step`] is a discrete prescription ("jump to
//! the top eigenvector"), this is the **continuous dissipative dynamics** whose
//! fixed points are the same attractors:
//!
//! ```text
//! ṡ_i = s_i ×_f B_i − λ [s_i ×_f [s_i ×_f B_i]]
//! [s ×_f B]_c = f_cab s_a B_b                      (SU(d) Lie-bracket product)
//! B_i = Σ_{j≠i} J_ij s_j = Σ_μ ξ_i^μ O_μ^(i)       (self-consistent mean field)
//! ```
//!
//! The precession term `s ×_f B` conserves energy; the Gilbert damping term lowers
//! it monotonically, `Ė = −λ Σ_i |s_i ×_f B_i|² ≤ 0`. So recall is not an algorithm
//! applied to the state — it is what the state *does* when left alone. Below `α_c`
//! the local minima of `E` are the stored memories.
//!
//! Energy convention here is `E = −(1/2) Σ_i s_i · B_i`, which is the potential
//! whose gradient is the mean field: `B_i = −∂E/∂s_i`.
//!
//! [`CpHopfieldRecaller::recall_step`]: super::CpHopfieldRecaller::recall_step

use super::CpHopfieldRecaller;
use super::basis::StructureConstants;

/// Configuration for the LLG flow.
#[derive(Clone, Copy, Debug)]
pub struct LlgConfig {
    /// Gilbert damping `λ > 0`. The paper's numerics use `λ = 1`, which converges
    /// in ~3 damping times; smaller values precess longer before settling.
    pub damping: f32,
    /// Integration step `dt`. The flow is integrated with explicit Euler followed
    /// by an exact manifold reprojection, so `dt` controls trajectory accuracy but
    /// not whether the state stays on `CP^(d-1)` (reprojection guarantees that).
    pub dt: f32,
    /// Stop once the mean per-neuron `|ṡ|` falls below this.
    pub tol: f32,
    /// Hard cap on integration steps.
    pub max_steps: usize,
}

impl Default for LlgConfig {
    fn default() -> Self {
        Self {
            damping: 1.0,
            dt: 0.05,
            tol: 1e-4,
            max_steps: 400,
        }
    }
}

/// Outcome of an LLG recall run.
#[derive(Clone, Debug)]
pub struct RecallResult {
    /// Steps actually integrated.
    pub steps: usize,
    /// Whether the flow reached [`LlgConfig::tol`] before the step cap.
    pub converged: bool,
    /// Energy after each step. Should be non-increasing — the G1 monotonicity
    /// check reads this directly.
    pub energy_trajectory: Vec<f32>,
}

impl RecallResult {
    /// Largest energy *increase* between consecutive steps.
    ///
    /// Should be zero up to integration round-off; a materially positive value
    /// means `dt` is too large for the current damping and the discretization has
    /// broken the `Ė ≤ 0` guarantee.
    pub fn max_energy_increase(&self) -> f32 {
        self.energy_trajectory
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0f32, f32::max)
    }
}

/// The SU(d) Lie-bracket product `[s ×_f b]_c = f_cab s_a b_b`.
///
/// Written as a sparse contraction over the stored `f_abc` triples. Because
/// `f_abc` is totally antisymmetric it is cyclic-invariant, so a triple stored as
/// `(i, j, k)` contributes `out[i] += f · s[j] · b[k]` and the full sum over the
/// leading index is covered by one pass.
///
/// `O(nnz(f))` rather than `O(D2²)`.
///
/// # Panics
/// Panics if the three slices have differing lengths.
pub fn lie_bracket_into(s: &[f32], b: &[f32], sc: &StructureConstants, out: &mut [f32]) {
    assert_eq!(s.len(), b.len(), "cp_hopfield: lie_bracket length mismatch");
    assert_eq!(
        s.len(),
        out.len(),
        "cp_hopfield: lie_bracket length mismatch"
    );
    out.fill(0.0);
    for t in sc.f_triples() {
        out[t.i as usize] += t.val * s[t.j as usize] * b[t.k as usize];
    }
}

impl<const D: usize, const D2: usize> CpHopfieldRecaller<D, D2> {
    /// The self-consistent mean field on neuron `i`:
    /// `B_i = Σ_μ ξ_i^μ O_μ^(i)`, in Bloch coordinates.
    ///
    /// `O(P · (N + D2))`.
    pub fn mean_field(&self, neuron_idx: usize) -> [f32; D2] {
        let mut b = [0.0f32; D2];
        for mu in 0..self.n_memories() {
            let o = self.mattis_overlap_excluding(neuron_idx, mu);
            if o == 0.0 {
                continue;
            }
            let m = self.memory_bloch(mu, neuron_idx);
            for a in 0..D2 {
                b[a] += o * m[a];
            }
        }
        b
    }

    /// Total energy `E = −(1/2) Σ_i s_i · B_i`.
    ///
    /// The potential for which the mean field is the negative gradient. The LLG
    /// damping term drives this monotonically downward.
    pub fn energy(&self) -> f32 {
        let mut e = 0.0f32;
        for i in 0..self.n_neurons() {
            let b = self.mean_field(i);
            let s = self.state(i);
            let dot: f32 = (0..D2).map(|a| s[a] * b[a]).sum();
            e -= 0.5 * dot;
        }
        e
    }

    /// One explicit-Euler LLG step for neuron `i`, followed by exact manifold
    /// reprojection. Returns `|ṡ_i|`, the local flow speed.
    ///
    /// Reprojecting after every step is what keeps the trajectory on `CP^(d-1)`
    /// despite the first-order integrator: the Euler step leaves the manifold, and
    /// [`CpHopfieldRecaller::project_to_manifold`] pulls it back to the *exact*
    /// closest on-manifold point rather than to an approximation of it.
    pub fn llg_step_neuron(&mut self, neuron_idx: usize, cfg: &LlgConfig) -> f32 {
        self.llg_step_neuron_driven(neuron_idx, cfg, None)
    }

    /// One LLG step for neuron `i` with an optional external driving field added
    /// to the self-consistent mean field.
    ///
    /// The undriven flow is autoassociative: the only field is the one the stored
    /// memories generate, so the state relaxes into whichever memory basin it
    /// starts nearest. A driven flow is what a *perception-coupled* consumer needs
    /// — the observation biases the field, and the memory term supplies the
    /// hysteresis that keeps a weak or ambiguous observation from dislodging an
    /// established belief. This is the Fusion D shape from Research 466: belief
    /// updates as LLG flow on `CP²` under a perception-derived field.
    ///
    /// Note that `Ė ≤ 0` is a property of the *undriven* flow. Driving injects
    /// energy by construction, so the monotonicity guarantee does not apply here.
    pub fn llg_step_neuron_driven(
        &mut self,
        neuron_idx: usize,
        cfg: &LlgConfig,
        bias: Option<&[f32; D2]>,
    ) -> f32 {
        let mut b = self.mean_field(neuron_idx);
        if let Some(bias) = bias {
            for a in 0..D2 {
                b[a] += bias[a];
            }
        }
        let s = *self.state(neuron_idx);
        let sc = self.structure();

        let mut precession = [0.0f32; D2];
        lie_bracket_into(&s, &b, sc, &mut precession);
        let mut damping_term = [0.0f32; D2];
        lie_bracket_into(&s, &precession, sc, &mut damping_term);

        let mut next = s;
        let mut speed_sq = 0.0f32;
        for a in 0..D2 {
            let dot_s = precession[a] - cfg.damping * damping_term[a];
            speed_sq += dot_s * dot_s;
            next[a] += cfg.dt * dot_s;
        }
        self.set_state_bloch(neuron_idx, &next);
        speed_sq.sqrt()
    }

    /// One full LLG sweep over all neurons. Returns the mean flow speed.
    pub fn llg_step(&mut self, cfg: &LlgConfig) -> f32 {
        let mut total = 0.0f32;
        for i in 0..self.n_neurons() {
            total += self.llg_step_neuron(i, cfg);
        }
        total / self.n_neurons() as f32
    }

    /// One full LLG sweep with a per-neuron external driving field.
    ///
    /// See [`Self::llg_step_neuron_driven`] for what driving means and why the
    /// energy-monotonicity guarantee does not extend to it.
    ///
    /// # Panics
    /// Panics if `bias.len() != n_neurons`.
    pub fn llg_step_driven(&mut self, cfg: &LlgConfig, bias: &[[f32; D2]]) -> f32 {
        assert_eq!(
            bias.len(),
            self.n_neurons(),
            "cp_hopfield: driving field must cover all {} neurons",
            self.n_neurons()
        );
        let mut total = 0.0f32;
        for (i, b) in bias.iter().enumerate() {
            total += self.llg_step_neuron_driven(i, cfg, Some(b));
        }
        total / self.n_neurons() as f32
    }

    /// Run the LLG flow to a fixed point.
    ///
    /// Records the energy after every step so callers can verify the
    /// `Ė ≤ 0` guarantee (see [`RecallResult::max_energy_increase`]).
    pub fn llg_recall(&mut self, cfg: &LlgConfig) -> RecallResult {
        let mut energy_trajectory = Vec::with_capacity(cfg.max_steps + 1);
        energy_trajectory.push(self.energy());
        for step in 0..cfg.max_steps {
            let speed = self.llg_step(cfg);
            energy_trajectory.push(self.energy());
            if speed < cfg.tol {
                return RecallResult {
                    steps: step + 1,
                    converged: true,
                    energy_trajectory,
                };
            }
        }
        RecallResult {
            steps: cfg.max_steps,
            converged: false,
            energy_trajectory,
        }
    }
}

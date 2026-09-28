//! Perception operators over the kinematic state — the anticipation half of
//! the primitive (Plan 578 Phase 2).
//!
//! Four operator families, all closed-form + modelless:
//!
//! 1. **Looming** — [`time_to_contact`]: Lee's optic-flow τ = σ/σ̇ and the
//!    log-extent variant. Ecological flee *timing*: small-fast and big-slow
//!    threats with the same τ get the same trigger.
//! 2. **Regimes** — [`regime_gates`] / [`RegimeClassifier`]: closed-form
//!    kinematic regime predicates with sigmoid gates + hysteresis. Solves the
//!    paper's joint-training failure mode (a single mixed predictor applies
//!    "downward pull" to uniform motion) with zero weights.
//! 3. **Surprise** — [`ResidualMonitor`]: one-step prediction residuals →
//!    z-score spike gate, CUSUM sustained-drift gate, and an impulse-vs-force
//!    discriminator with wall inference + restitution on detection.
//! 4. **Geometry** — [`closest_approach`] / [`intercept_time`] /
//!    [`head_on_elastic_resolve`]: two-body encounter math.
//!
//! Plus the **UQ-bearing** admission bound ([`extrapolation_horizon`]) —
//! benched against the conformal-naive floor per the "Report the Floor"
//! policy (see `.benchmarks/680`).
//!
//! All gates use **sigmoid**, never softmax (the house rule): each predicate
//! is an independent [0,1] confidence, classified by priority + hysteresis,
//! not by normalized competition.

use crate::kinematics::{KinState, SQRT2, SQRT6, SQRT20, extrapolation_weight_ss, lattice};

// ===== looming: time to contact =====

/// Lee's optic-flow time-to-contact from an extent (size-on-retina) channel:
/// `τ = σ / (−σ̇)` — seconds until contact under constant closing rate.
///
/// Guards: non-approaching (σ̇ ≥ 0) or σ̇ below the floor → `f32::INFINITY`
/// (never contacts); σ ≤ 0 → `INFINITY` (no extent signal). The two-argument
/// form takes the last two extent samples; the rate is the backward
/// difference `σ̇ = (σ_now − σ_prev)/Δt`.
#[must_use]
pub fn time_to_contact(sigma_now: f32, sigma_prev: f32, dt: f32) -> f32 {
    if !(sigma_now.is_finite() && sigma_prev.is_finite()) || dt <= 0.0 {
        return f32::INFINITY;
    }
    if sigma_now <= 0.0 {
        return f32::INFINITY;
    }
    let rate = (sigma_now - sigma_prev) / dt;
    if rate >= -f32::EPSILON * sigma_now.max(1.0) {
        return f32::INFINITY;
    }
    sigma_now / -rate
}

/// Log-extent variant: `τ = 1 / (d ln σ / dt)` — exact for exponential
/// growth/shrink of the extent (constant relative rate), where the linear
/// form is only exact for constant absolute rate.
///
/// Same guards as [`time_to_contact`]; additionally `σ_prev ≤ 0` → ∞.
#[must_use]
pub fn time_to_contact_log(sigma_now: f32, sigma_prev: f32, dt: f32) -> f32 {
    if !(sigma_now.is_finite() && sigma_prev.is_finite()) || dt <= 0.0 {
        return f32::INFINITY;
    }
    if sigma_now <= 0.0 || sigma_prev <= 0.0 {
        return f32::INFINITY;
    }
    let rel = (sigma_now.ln() - sigma_prev.ln()) / dt;
    if rel >= -f32::EPSILON {
        return f32::INFINITY;
    }
    1.0 / -rel
}

// ===== regimes =====

/// Kinematic regime labels — the paper's five benchmark task families.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Regime {
    /// Constant velocity (degree ≤ 1).
    Uniform,
    /// Constant acceleration ≈ `g` (degree 2; the gravity family).
    Parabolic { g: f32 },
    /// Transient velocity discontinuity (collision / bounce onset).
    Impulse,
    /// Extent shrinking — approach on the collision course.
    Looming,
    /// Acceleration opposing velocity and decaying (drag family).
    Drag,
}

/// Aggregated per-target statistics the predicates consume (caller computes
/// the vector dot products once; the classifier stays scalar + branch-light).
#[derive(Clone, Copy, Debug, Default)]
pub struct RegimeSnapshot {
    /// Sample period.
    pub dt: f32,
    /// ‖vel‖ at the anchor.
    pub speed: f32,
    /// ‖acc‖ at the anchor.
    pub acc_mag: f32,
    /// ‖jerk‖ at the anchor (fresh one-step estimate).
    pub jerk_mag: f32,
    /// `|Δvel|/Δt` this tick — the impulse statistic.
    pub dv_dt: f32,
    /// EMA of ‖acc‖ — the running force scale the impulse is judged against.
    /// **Track it robustly** (winsorize: update toward `min(acc, C·running +
    /// floor)`): a raw EMA absorbs segment-boundary mixed-window transients
    /// (~120 units on the fixtures) and stays inflated long enough to mask a
    /// genuine planted bounce 15 ticks later, while a hard exclusion of
    /// high-dv ticks starves the scale on sustained force onsets and fires
    /// impulses on every parabola frame (both failure modes found by the
    /// fixture analysis, not theory).
    pub running_acc: f32,
    /// `acc·vel` — negative when acceleration opposes motion.
    pub acc_vel_dot: f32,
    /// `jerk·acc` — negative when |acc| is decaying.
    pub jerk_acc_dot: f32,
    /// Extent σ (≤ 0 = no extent channel).
    pub sigma: f32,
    /// Extent rate σ̇ (negative = shrinking).
    pub sigma_rate: f32,
}

impl RegimeSnapshot {
    /// Build a snapshot from a kinematic state plus the caller-tracked
    /// previous velocity (for the impulse statistic), the running-|acc| EMA,
    /// and the extent channel (if any).
    #[allow(clippy::needless_range_loop)] // ch indexes parallel vel/acc/jerk arrays
    pub fn from_state<const D: usize>(
        state: &KinState<D>,
        prev_vel: &[f32; D],
        running_acc: f32,
        sigma: f32,
        sigma_rate: f32,
    ) -> Self {
        let mut speed2 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut jerk2 = 0.0f32;
        let mut dv2 = 0.0f32;
        let mut av_dot = 0.0f32;
        let mut ja_dot = 0.0f32;
        for ch in 0..D {
            speed2 += state.vel[ch] * state.vel[ch];
            acc2 += state.acc[ch] * state.acc[ch];
            jerk2 += state.jerk[ch] * state.jerk[ch];
            let dv = (state.vel[ch] - prev_vel[ch]) / state.dt;
            dv2 += dv * dv;
            av_dot += state.acc[ch] * state.vel[ch];
            ja_dot += state.jerk[ch] * state.acc[ch];
        }
        Self {
            dt: state.dt,
            speed: speed2.sqrt(),
            acc_mag: acc2.sqrt(),
            jerk_mag: jerk2.sqrt(),
            dv_dt: dv2.sqrt(),
            running_acc,
            acc_vel_dot: av_dot,
            jerk_acc_dot: ja_dot,
            sigma,
            sigma_rate,
        }
    }
}

/// Predicate gates — one sigmoid confidence per regime, [0,1], independent
/// (never normalized; classification is priority + hysteresis).
#[derive(Clone, Copy, Debug, Default)]
pub struct RegimeGates {
    pub uniform: f32,
    pub parabolic: f32,
    pub impulse: f32,
    pub looming: f32,
    pub drag: f32,
}

/// Classifier tuning. Defaults match the PhyWorld fixture scales (Plan 578
/// T3.2); `g` is in position-units per Δt² (caller converts world units).
#[derive(Clone, Copy, Debug)]
pub struct RegimeConfig {
    /// Expected gravity magnitude (pos/Δt²).
    pub g: f32,
    /// Gate value at which a regime is admitted.
    pub enter: f32,
    /// Gate value below which the current regime is released.
    pub exit: f32,
    /// Impulse: required multiple of the running |acc| for |Δv|/Δt.
    pub impulse_ratio: f32,
    /// Looming: per-step relative extent shrink admitted (|Δσ|/σ per
    /// step). 0.008 — the paper's ID range bottoms out at
    /// |ṙ|/σ ≈ (1/64)/1.375 ≈ 0.011, so a 0.02 center would leave the
    /// slowest ID approaches classified as Uniform (found by the fixture).
    pub looming_rate: f32,
    /// Uniform: curvature (|a|·Δt/|v|) below which motion reads as straight.
    pub uniform_curv: f32,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            g: 1.0,
            enter: 0.6,
            exit: 0.3,
            impulse_ratio: 8.0,
            looming_rate: 0.008,
            uniform_curv: 0.05,
        }
    }
}

/// Resolution floor for drag/parabolic predicates (2⁻¹⁰): below this the
/// deceleration is within a few ULPs of the accumulated position and the
/// finite differences flicker between 0 and ±ULP — any scale-free gate is
/// unresolvable noise there. The fixture generator tags against the same
/// floor (exported so consumer + tagger cannot drift).
pub const DRAG_ACC_FLOOR: f32 = 9.765_625e-4; // 2^-10

/// Closed-form regime predicates: one sigmoid gate per regime from the
/// snapshot statistics (see [`RegimeConfig`] for the thresholds).
///
/// Priority order used by the classifier: Impulse > Looming > Drag >
/// Parabolic > Uniform (transient events outrank sustained regimes).
#[must_use]
pub fn regime_gates(snap: &RegimeSnapshot, cfg: &RegimeConfig) -> RegimeGates {
    let speed_floor = (snap.speed.abs() + f32::EPSILON).max(1e-6);
    // Curvature per step: |a|·Δt/|v| (dimensionless).
    let curv = snap.acc_mag * snap.dt / speed_floor;
    // Jerk relative to the acceleration scale: |j|·Δt²/|v|.
    let j_rel = snap.jerk_mag * snap.dt * snap.dt / speed_floor;

    let uniform =
        crate::sigmoid((cfg.uniform_curv - curv) / 0.02) * crate::sigmoid((0.05 - j_rel) / 0.02);

    // Parabolic: |‖a‖ − g| small AND jerk small.
    let g_err = (snap.acc_mag - cfg.g).abs() / cfg.g.max(1e-6);
    let parabolic = crate::sigmoid((0.3 - g_err) / 0.1) * crate::sigmoid((0.05 - j_rel) / 0.02);

    // Impulse: |Δv|/Δt versus the running force scale.
    let force_scale = snap.running_acc.max(1e-6);
    let impulse = crate::sigmoid((snap.dv_dt / force_scale - cfg.impulse_ratio) / 2.0);

    // Looming: per-step relative shrink of the extent.
    let mut looming = 0.0;
    if snap.sigma > 0.0 {
        let shrink = -snap.sigma_rate * snap.dt / snap.sigma;
        looming = crate::sigmoid((shrink - cfg.looming_rate) / 0.005);
    }

    // Drag: acceleration opposing velocity AND |acc| decaying, with the
    // magnitude above the resolution floor (below it the FD flickers —
    // see DRAG_ACC_FLOOR). The zero guards are relative (product == 0,
    // not |x| < ε) so a decaying drag holds its gate until the FD
    // acceleration is *exactly* zero — the same condition an external
    // tagger sees — instead of leaking through an absolute ε floor at
    // small magnitudes.
    let av_norm = snap.acc_mag * snap.speed;
    let align = if av_norm > 0.0 {
        -snap.acc_vel_dot / av_norm
    } else {
        0.0
    }; // 1 = head-on opposing
    let ja_norm = snap.jerk_mag * snap.acc_mag;
    let decay = if ja_norm > 0.0 {
        -snap.jerk_acc_dot / ja_norm
    } else {
        0.0
    }; // 1 = decaying
    let drag = if snap.acc_mag > DRAG_ACC_FLOOR {
        crate::sigmoid((align - 0.6) / 0.1) * crate::sigmoid((decay - 0.2) / 0.1)
    } else {
        0.0
    };

    RegimeGates {
        uniform,
        parabolic,
        impulse,
        looming,
        drag,
    }
}

/// Sigmoid-gated hysteresis regime classifier.
///
/// Switch rules (the anti-flip-flop contract):
/// - **Impulse bypasses hysteresis**: it is definitionally transient; it is
///   emitted for the firing tick without touching the sticky state.
/// - `proposed` = the highest-priority regime whose gate ≥ `enter`
///   (priority: Looming > Drag > Parabolic > Uniform).
/// - Switch to `proposed` when it **out-prioritizes** the held regime (a
///   stronger signal supersedes immediately — without this, a sticky
///   low-priority regime whose gate stays high would block a genuine
///   higher-priority regime forever), **or** when the held regime's own gate
///   has fallen below `exit` (it collapsed).
/// - Nothing proposed: hold through dips while the held gate ≥ `exit`; once
///   below, clear and fall back to `Uniform` (the least-assumption label).
#[derive(Clone, Copy, Debug)]
pub struct RegimeClassifier {
    cfg: RegimeConfig,
    held: Option<Regime>,
}

/// Regime precedence for switching (higher wins). Impulse sits above
/// everything but is transient (never held).
fn regime_priority(r: &Regime) -> u8 {
    match r {
        Regime::Impulse => 5,
        Regime::Looming => 4,
        Regime::Drag => 3,
        Regime::Parabolic { .. } => 2,
        Regime::Uniform => 1,
    }
}

impl RegimeClassifier {
    /// New classifier with explicit config.
    #[must_use]
    pub fn new(cfg: RegimeConfig) -> Self {
        Self { cfg, held: None }
    }

    /// Classify one tick; updates the hysteresis state.
    pub fn classify(&mut self, snap: &RegimeSnapshot) -> Regime {
        let gates = regime_gates(snap, &self.cfg);
        // Transient bypass: impulses are definitionally instantaneous; emit
        // without touching the sticky state.
        if gates.impulse >= self.cfg.enter {
            return Regime::Impulse;
        }
        let proposed = if gates.looming >= self.cfg.enter {
            Some(Regime::Looming)
        } else if gates.drag >= self.cfg.enter {
            Some(Regime::Drag)
        } else if gates.parabolic >= self.cfg.enter {
            Some(Regime::Parabolic { g: snap.acc_mag })
        } else if gates.uniform >= self.cfg.enter {
            Some(Regime::Uniform)
        } else {
            None
        };
        let held_gate = match self.held {
            Some(Regime::Looming) => gates.looming,
            Some(Regime::Drag) => gates.drag,
            Some(Regime::Parabolic { .. }) => gates.parabolic,
            Some(Regime::Uniform) => gates.uniform,
            Some(Regime::Impulse) | None => 0.0,
        };
        match (proposed, self.held) {
            (Some(p), Some(h)) => {
                if p == h || regime_priority(&p) > regime_priority(&h) || held_gate < self.cfg.exit
                {
                    self.held = Some(p);
                }
            }
            (Some(p), None) => self.held = Some(p),
            (None, _) => {
                if held_gate < self.cfg.exit {
                    self.held = None;
                }
            }
        }
        match self.held {
            Some(Regime::Parabolic { .. }) => Regime::Parabolic { g: snap.acc_mag },
            Some(r) => r,
            None => Regime::Uniform,
        }
    }
}

// ===== residual surprise =====

/// Surprise-event taxonomy for prediction residuals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventKind {
    /// Sudden velocity discontinuity (collision / bounce onset). Wall
    /// inference + restitution are filled by [`impulse_report`].
    Impulse {
        /// Wall-normal axis if the velocity flipped sign on some channel.
        axis: Option<usize>,
        /// Restitution `e = |v_after|/|v_before|` on the wall axis.
        e: Option<f32>,
    },
    /// Sustained one-sided drift (CUSUM over the slack threshold).
    Drift {
        /// CUSUM statistic at the alarm.
        cusum: f32,
    },
    /// Isolated spike: z-score gate fired without drift or impulse.
    Spike {
        /// Standardized residual at the event.
        z: f32,
        /// Sigmoid gate value [0,1].
        gate: f32,
    },
}

/// Tuning for [`ResidualMonitor`].
#[derive(Clone, Copy, Debug)]
pub struct ResidualConfig {
    /// EMA decay for the residual mean/variance and the running |acc|.
    pub ema_beta: f32,
    /// z-score center of the spike gate.
    pub z_ref: f32,
    /// z-score sigmoid width.
    pub z_width: f32,
    /// CUSUM slack (in residual σ units).
    pub cusum_k: f32,
    /// CUSUM alarm threshold (in residual σ units).
    pub cusum_h: f32,
    /// Impulse ratio: |Δv|/Δt multiple over the running |acc|.
    pub impulse_ratio: f32,
    /// Warmup observations before any event can fire.
    pub warmup: u32,
}

impl Default for ResidualConfig {
    fn default() -> Self {
        Self {
            ema_beta: 0.05,
            z_ref: 3.0,
            z_width: 0.5,
            cusum_k: 0.5,
            cusum_h: 5.0,
            impulse_ratio: 8.0,
            warmup: 16,
        }
    }
}

/// One-step prediction-residual surprise monitor (scalar — one per channel).
///
/// Maintains the residual mean/variance (the noise model), a two-sided
/// CUSUM, and an EMA of the running force scale. Feed it the residual
/// `observed − predicted` each tick plus the current `|Δv|/Δt`; it returns
/// the first firing event, if any.
///
/// Cold start: `warmup` observations seed the statistics before any event
/// can fire. During warmup the mean is the **running average** (not a
/// lagging EMA) and the CUSUM resets at the warmup boundary — otherwise the
/// ladder-fill transient (the first few residuals are structurally nonzero
/// while the FD window fills) leaks into the post-warmup mean and reads as a
/// sustained drift (found by fixture analysis, not theory).
#[derive(Clone, Copy, Debug)]
pub struct ResidualMonitor {
    cfg: ResidualConfig,
    mean_r: f32,
    var_r: f32,
    mean_acc: f32,
    cusum_pos: f32,
    cusum_neg: f32,
    warm_sum: f32,
    n: u32,
}

impl ResidualMonitor {
    /// New monitor with explicit config.
    #[must_use]
    pub fn new(cfg: ResidualConfig) -> Self {
        Self {
            cfg,
            mean_r: 0.0,
            var_r: 0.0,
            mean_acc: 0.0,
            cusum_pos: 0.0,
            cusum_neg: 0.0,
            warm_sum: 0.0,
            n: 0,
        }
    }

    /// Current residual-σ estimate (sqrt of the EMA variance).
    #[must_use]
    pub fn sigma(&self) -> f32 {
        self.var_r.max(0.0).sqrt()
    }

    /// Current observation-noise estimate: deconvolves the residual σ by
    /// the k-step predictive amplification `√(wss(k, order) + 1)` (predictor
    /// weights + the new observation's noise) — the honest σ̂_obs for
    /// [`extrapolation_horizon`]'s bound and the UQ interval.
    #[must_use]
    pub fn eps_obs(&self, k: u32, order: u8) -> f32 {
        let amp = (extrapolation_weight_ss(k, order) + 1.0).sqrt().max(1.0);
        self.sigma() / amp
    }

    /// Absorb one tick. Returns the firing event (if any). The impulse arm
    /// takes the raw before/after velocities so wall inference + restitution
    /// ride the same report.
    pub fn update<const D: usize>(
        &mut self,
        residual: f32,
        vel_before: &[f32; D],
        vel_after: &[f32; D],
        dt: f32,
    ) -> Option<EventKind> {
        let event = if self.n >= self.cfg.warmup && residual.is_finite() {
            let sig = self.sigma().max(1e-9);
            let z = ((residual - self.mean_r).abs()) / sig;
            let gate = crate::sigmoid((z - self.cfg.z_ref) / self.cfg.z_width);

            // CUSUM (two-sided, slack k·σ).
            let dev = residual - self.mean_r;
            let slack = self.cfg.cusum_k * sig;
            self.cusum_pos = (self.cusum_pos + dev - slack).max(0.0);
            self.cusum_neg = (self.cusum_neg + dev + slack).min(0.0);

            // Impulse discriminator: |Δv|/Δt vs the running force scale.
            let mut dv2 = 0.0f32;
            for ch in 0..D {
                let dv = (vel_after[ch] - vel_before[ch]) / dt;
                dv2 += dv * dv;
            }
            let dv_dt = dv2.sqrt();
            let force_scale = self.mean_acc.max(1e-6);
            if dv_dt > self.cfg.impulse_ratio * force_scale {
                let rep = impulse_report(vel_before, vel_after);
                Some(match rep {
                    ImpulseRaw::Wall { axis, e } => EventKind::Impulse {
                        axis: Some(axis),
                        e: Some(e),
                    },
                    ImpulseRaw::Free => EventKind::Impulse {
                        axis: None,
                        e: None,
                    },
                })
            } else if self.cusum_pos > self.cfg.cusum_h * sig
                || -self.cusum_neg > self.cfg.cusum_h * sig
            {
                Some(EventKind::Drift {
                    cusum: self.cusum_pos.max(-self.cusum_neg),
                })
            } else if gate >= 0.5 {
                Some(EventKind::Spike { z, gate })
            } else {
                None
            }
        } else {
            None
        };

        // Statistics update AFTER the decision (a residual must not deflate
        // its own z-score).
        let b = self.cfg.ema_beta;
        if self.n < self.cfg.warmup {
            // Running average during warmup (absorbs the ladder transient).
            self.warm_sum += residual;
            self.mean_r = self.warm_sum / (self.n + 1) as f32;
        } else {
            let new_mean = self.mean_r + b * (residual - self.mean_r);
            self.mean_r = new_mean;
            if self.n == self.cfg.warmup {
                // CUSUM starts from a clean slate at the warmup boundary.
                self.cusum_pos = 0.0;
                self.cusum_neg = 0.0;
            }
        }
        let dm = residual - self.mean_r;
        self.var_r += b * (dm * dm - self.var_r);
        let mut dv2 = 0.0f32;
        for ch in 0..D {
            let dv = (vel_after[ch] - vel_before[ch]) / dt;
            dv2 += dv * dv;
        }
        let a_mag = dv2.sqrt(); // |Δv|/Δt ≈ |acc| this tick
        self.mean_acc += b * (a_mag - self.mean_acc);
        self.n = self.n.saturating_add(1);
        event
    }
}

/// Internal impulse classification (wall vs free force).
enum ImpulseRaw {
    Wall {
        axis: usize,
        e: f32,
    },
    /// No sign flip anywhere — a free force impulse (no wall).
    Free,
}

/// Wall inference: the channel with the largest |Δv| **among sign-flipping
/// channels** is the wall normal; restitution `e = |v_after|/|v_before|`
/// there. No sign flip anywhere → a free force impulse (no wall).
fn impulse_report<const D: usize>(vel_before: &[f32; D], vel_after: &[f32; D]) -> ImpulseRaw {
    let mut best_axis = None;
    let mut best_dv = 0.0f32;
    for (ch, (&vb, &va)) in vel_before.iter().zip(vel_after.iter()).enumerate() {
        let dv = (va - vb).abs();
        if vb * va < 0.0 && dv > best_dv {
            best_dv = dv;
            best_axis = Some(ch);
        }
    }
    match best_axis {
        Some(axis) => ImpulseRaw::Wall {
            axis,
            e: (vel_after[axis].abs() / vel_before[axis].abs()).min(1.0),
        },
        None => ImpulseRaw::Free,
    }
}

// ===== two-body geometry =====

/// Two-body closest-approach report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApproachReport {
    /// Time of closest approach (in Δt units of the caller's time base;
    /// negative = the encounter is receding).
    pub t_star: f32,
    /// Separation at `t_star` (the miss distance).
    pub miss_dist: f32,
    /// `true` when the pair is closing (`t_star > 0`).
    pub closing: bool,
}

/// Closest approach of two constant-velocity bodies:
/// `t* = −(p_rel·v_rel)/(v_rel·v_rel)`, miss distance `|p_rel + t*·v_rel|`.
///
/// Parallel / stationary relative motion (`v_rel·v_rel` ≈ 0) reports the
/// current separation at `t* = 0`. The **t*-ascending ordering law**: sorting
/// encounters by `t_star` (closing ones first, smallest first) equals the
/// ground-truth contact order — pinned by test.
#[must_use]
pub fn closest_approach<const D: usize>(
    p1: &[f32; D],
    v1: &[f32; D],
    p2: &[f32; D],
    v2: &[f32; D],
) -> ApproachReport {
    let mut p = [0.0f32; D];
    let mut v = [0.0f32; D];
    let mut pp = 0.0f32;
    let mut pv = 0.0f32;
    let mut vv = 0.0f32;
    for ch in 0..D {
        p[ch] = p1[ch] - p2[ch];
        v[ch] = v1[ch] - v2[ch];
        pp += p[ch] * p[ch];
        pv += p[ch] * v[ch];
        vv += v[ch] * v[ch];
    }
    if vv <= f32::EPSILON * pp.max(1.0) {
        return ApproachReport {
            t_star: 0.0,
            miss_dist: pp.sqrt(),
            closing: false,
        };
    }
    let t = -pv / vv;
    let mut miss2 = 0.0f32;
    for ch in 0..D {
        let s = p[ch] + t * v[ch];
        miss2 += s * s;
    }
    ApproachReport {
        t_star: t,
        miss_dist: miss2.sqrt(),
        closing: t > 0.0,
    }
}

/// Interception time: smallest positive `t` with
/// `|p2 + v2·t − p1| = speed·t` (pursuer at `p1` with scalar `speed`).
///
/// Solves `(v2·v2 − s²)t² + 2(d·v2)t + d·d = 0` with `d = p2 − p1` (target
/// relative to pursuer). `None` when no real positive root exists (target
/// too fast / receding).
#[must_use]
pub fn intercept_time<const D: usize>(
    p1: &[f32; D],
    p2: &[f32; D],
    v2: &[f32; D],
    speed: f32,
) -> Option<f32> {
    let mut dd = 0.0f32;
    let mut dv = 0.0f32;
    let mut vv = 0.0f32;
    for ch in 0..D {
        let d = p2[ch] - p1[ch];
        dd += d * d;
        dv += d * v2[ch];
        vv += v2[ch] * v2[ch];
    }
    let a = vv - speed * speed;
    let b = 2.0 * dv;
    let c = dd;
    if a.abs() <= f32::EPSILON * vv.max(1.0) {
        // Linear: b·t + c = 0.
        if b >= 0.0 {
            return None;
        }
        return Some(-c / b);
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let (r1, r2) = if a > 0.0 {
        ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a))
    } else {
        ((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a))
    };
    // Smallest positive root.
    if r1 > 0.0 {
        Some(r1)
    } else if r2 > 0.0 {
        Some(r2)
    } else {
        None
    }
}

/// 1D head-on elastic collision resolution along the line of centers:
///
/// ```text
/// v1' = ((m1−m2)·v1 + 2·m2·v2) / (m1+m2)
/// v2' = ((m2−m1)·v2 + 2·m1·v1) / (m1+m2)
/// ```
///
/// Energy- and momentum-conserving by construction (textbook).
#[must_use]
pub fn head_on_elastic_resolve(v1: f32, v2: f32, m1: f32, m2: f32) -> (f32, f32) {
    let mt = m1 + m2;
    (
        ((m1 - m2) * v1 + 2.0 * m2 * v2) / mt,
        ((m2 - m1) * v2 + 2.0 * m1 * v1) / mt,
    )
}

// ===== extrapolation horizon (UQ-bearing) =====

/// **UQ-floor verdict (Report-the-Floor policy, benched in
/// `.benchmarks/680`): RANK-ONLY.** The bound's k* ordering is the shipped
/// claim; no calibrated-coverage claim is made. Measured at the floor
/// harness's h=1 protocol: on **curving** motion the kinematic interval
/// beats the conformal-naive floor decisively (CRPS ratio 0.10; the floor's
/// conformal drift-correction cannot track a moving drift — coverage
/// collapse 0.17 vs 0.86), but on straight motion and white noise it loses
/// ~1.08×: at h=1 a drift-corrected naive prediction is statistically
/// optimal — both predictors anchor on the same noisy last observation
/// (√2·σ shared floor; conformal's finite-sample quantile edge). The
/// horizon-ordering claim (k* ascending = trust ordering) is
/// regime-independent and unaffected.
///
/// Independent uncertainty scales of the four state coefficients.
///
/// The bound composes them with the triangle inequality (worst-case); for
/// the i.i.d. observation-noise model use [`Eps::from_obs_noise`], which
/// propagates a single σ_obs through the backward-difference estimators
/// (√2, √6, √20 factors).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Eps {
    /// Anchor position uncertainty ε_p.
    pub pos: f32,
    /// Velocity-coefficient uncertainty ε_v.
    pub vel: f32,
    /// Acceleration-coefficient uncertainty ε_a.
    pub acc: f32,
    /// Jerk-coefficient uncertainty ε_j.
    pub jerk: f32,
}

impl Eps {
    /// From a single i.i.d. observation-noise scale: ε_p = ε, ε_v = √2·ε/Δt,
    /// ε_a = √6·ε/Δt², ε_j = √20·ε/Δt³.
    #[must_use]
    pub fn from_obs_noise(eps_obs: f32, dt: f32) -> Self {
        Self {
            pos: eps_obs,
            vel: SQRT2 * eps_obs / dt,
            acc: SQRT6 * eps_obs / (dt * dt),
            jerk: SQRT20 * eps_obs / (dt * dt * dt),
        }
    }
}

/// Error-propagation bound at horizon k:
///
/// ```text
/// B(k) = ε_p + C(k,1)·Δt·ε_v + C(k+1,2)·Δt²·ε_a + C(k+2,3)·Δt³·ε_j
/// ```
///
/// Monotone in k. The lattice supplies the binomials.
#[must_use]
pub fn horizon_bound(k: u32, dt: f32, eps: &Eps) -> f32 {
    let k = (k as usize).min(crate::kinematics::K_MAX);
    let row = lattice()[k];
    eps.pos + row.b1 * dt * eps.vel + row.b2 * dt * dt * eps.acc + row.b3 * dt * dt * dt * eps.jerk
}

/// Admission-horizon verdict: the largest k whose bound stays under `thr`,
/// plus a sigmoid confidence from the remaining margin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizonVerdict {
    /// Largest admitted horizon (0 = not even the anchor is within `thr`).
    pub k_star: u32,
    /// `sigmoid((thr − B(k*)) / (0.25·thr))` — margin confidence, [0,1].
    pub conf: f32,
    /// `B(k*)` — the bound at the admitted horizon.
    pub bound: f32,
}

/// The admission gate: scan the lattice for the largest k with
/// `B(k) ≤ thr` (the bound is monotone in k, so the scan is exact).
#[must_use]
pub fn extrapolation_horizon(dt: f32, eps: &Eps, thr: f32) -> HorizonVerdict {
    let mut k_star = 0u32;
    let mut bound = horizon_bound(0, dt, eps);
    if bound > thr {
        return HorizonVerdict {
            k_star: 0,
            conf: crate::sigmoid((thr - bound) / (0.25 * thr.abs() + 1e-9)),
            bound,
        };
    }
    for k in 1..=(crate::kinematics::K_MAX as u32) {
        let b = horizon_bound(k, dt, eps);
        if b > thr {
            break;
        }
        k_star = k;
        bound = b;
    }
    HorizonVerdict {
        k_star,
        conf: crate::sigmoid((thr - bound) / (0.25 * thr.abs() + 1e-9)),
        bound,
    }
}

/// State-based convenience wrapper (the plan's T2.5 signature): derives the
/// ε scales from an observation-noise estimate and the state's `dt`.
#[must_use]
pub fn extrapolation_horizon_for_state<const D: usize>(
    state: &KinState<D>,
    eps_obs: f32,
    thr: f32,
) -> HorizonVerdict {
    extrapolation_horizon(state.dt, &Eps::from_obs_noise(eps_obs, state.dt), thr)
}

// ===== UQ interval construction =====

/// `erf(x)` via Abramowitz & Stegun 7.1.26 (max |error| 1.5e-7 — the
/// classic 5-term rational form; the coefficient set is among the most
/// widely reproduced in the numerical literature).
#[allow(clippy::excessive_precision)] // verbatim A&S 7.1.26 coefficients
#[must_use]
fn erf_as261(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Standard normal CDF `Φ(z) = 0.5·(1 + erf(z/√2))`.
#[must_use]
fn normal_cdf(z: f32) -> f32 {
    0.5 * (1.0 + erf_as261(z * std::f32::consts::FRAC_1_SQRT_2))
}

/// Two-sided normal quantile `z_{1−α/2}` — bisection on [`normal_cdf`]
/// (50 iterations over [−9, 9]: deterministic, accurate to the erf's
/// 1.5e-7 — far tighter than interval construction needs; empirical
/// coverage is what the floor bench measures).
///
/// (Replaces a from-memory Hastings rational form that failed its own
/// anchor test — z(0.05) evaluated to 2.37 — the coefficient set was
/// unverifiable; bisection on a verifiable erf is the honest fix.)
#[must_use]
pub fn normal_two_sided_z(alpha: f32) -> f32 {
    debug_assert!((0.0..=0.5).contains(&alpha), "alpha in [0, 0.5]");
    let target = 1.0 - (alpha * 0.5).clamp(1e-9, 0.5);
    let (mut lo, mut hi) = (-9.0f32, 9.0f32);
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if normal_cdf(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Predictive half-width for the k-step extrapolation at a given ladder
/// order: `z_{1−α/2} · σ_obs · √(wss(k, order) + 1)` — the i.i.d. propagation
/// of observation noise through the extrapolator's observation weights
/// (`wss`, the predictor's variance) **plus the new observation's own noise**
/// (the +1; without it the interval under-covers by √((wss+1)/wss) — found
/// by the floor bench, not theory). The admission bound [`horizon_bound`]
/// remains the conservative triangle form.
#[must_use]
pub fn predictive_half_width(eps_obs: f32, k: u32, order: u8, alpha: f32) -> f32 {
    normal_two_sided_z(alpha) * eps_obs * (extrapolation_weight_ss(k, order) + 1.0).sqrt()
}

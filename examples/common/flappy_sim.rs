//! Flappy micro-arena sim — katgpt-rs Plan 607 T5. The laya page's own
//! Flappy protocol is a STATE question ("The bird is a little below the
//! gap" → P(below) > 0.5 → flap): where it is, never what to do — the
//! action is derived in code from the state read.
//!
//! Decision shape: at each pipe the bird is ONE tick from the pipe plane.
//! The two candidate resulting states (flap / coast) are each rendered as a
//! closed-grammar English sentence; the scorer reads p(clean) per option
//! and the argmax IS the action. Option order is PINNED [flap, coast] —
//! the lowest-index tie-break and the constant-pick baseline both refer to
//! it.
//!
//! Grammar law (measured, `flap_and_coast_sentences_always_differ`): the
//! flap and coast results can coincide in position (v = +2) but NEVER in
//! the motion clause — flap's post-velocity is FLAP_V, coast's is at most
//! V_MAX - 1 — so the two options of one state are never the same sentence,
//! and class-level agreement degenerates to raw agreement here.

// Each consumer compiles a different half (the 01 enumerator uses
// enumerate_states + the grammar consts; the 02 arena uses renders,
// features and the play loop) — per-consumer dead-code warns would fire.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Vertical grid (y ∈ [0, GRID_H); y = 0 ground).
pub const GRID_H: i32 = 12;
/// Velocity clamp.
pub const V_MIN: i32 = -2;
pub const V_MAX: i32 = 2;
/// A flap sets the velocity and lifts the bird this many cells, then the
/// tick advances: the flap result is (y + FLAP_DY, FLAP_V).
pub const FLAP_DY: i32 = 2;
pub const FLAP_V: i32 = 2;

/// Grammar identity stamped into every dump record.
///
/// v2 history (measured, Bench 880): v1's option sentences carried a motion
/// clause (", rising." / ", falling fast.") to guarantee the two options
/// never tie — and laya's read keyed on THAT clause, not the position
/// clause: the v1 oracle argmax went to flap (index 0) in 85/100 states,
/// pinning every scorer at the constant-pick ceiling. The motion clause was
/// a value-loaded confound (rising reads safe regardless of geometry); v2
/// renders the position band ALONE and excludes BOTH degenerate state
/// classes (v = FLAP_V lands both actions on one cell; same-band results
/// render identical sentences — the unbounded Below/Above bands tie most
/// often), so every committed decision has two distinct descriptions.
///
/// v3 history (Issue 876, Bench 881 finding): the v2 band alone is too
/// coarse for decode-based consumption — the decoded arm collapsed to
/// constant-flap (77/100, ONE distinct pick) against the structured arm's
/// 96. v3 widens the option sentence with two decision-relevant,
/// non-value-loaded clauses: a quantized OFFSET (fine post_rel relative to
/// the gap center, clamped at ±2) and a NEUTRAL post-placement motion
/// clause ("holding this height" / "drifting down one step" — magnitude +
/// direction in kinematic wording, never "rising"/"falling"). Known
/// structural caveat, stated not hidden: post_v is a deterministic
/// function of the action here (flap ⇒ +2, coast ⇒ ≤ 0), so the motion
/// clause inherently names the action — the neutral wording + the offset
/// anchor are the measured defense; if the oracle still keys on the
/// clause alone, that IS the finding (an exact post_v clause cannot be
/// made action-blind in this protocol) and the offset-only ablation is
/// the next arm.
pub const GRAMMAR_ID: &str = "laya-flappy-v3";
/// The per-option question, world-anchored (never "what should I do" — the
/// wording lesson from laya's own page).
pub const QUESTION: &str = "Will the bird pass through the gap cleanly?";

/// The decision state: bird position/velocity + the gap (center g, half
/// height h — the gap spans [g - h, g + h]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FlappyState {
    pub y: i32,
    pub v: i32,
    pub g: i32,
    pub h: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Flap,
    Coast,
}

/// The pinned option order (index 0 = flap).
pub const ACTIONS: [Action; 2] = [Action::Flap, Action::Coast];
pub const ACTION_LABELS: [&str; 2] = ["flap", "coast"];

/// The action's resulting state: (y', v').
pub fn result(s: &FlappyState, a: Action) -> (i32, i32) {
    match a {
        Action::Flap => (s.y + FLAP_DY, FLAP_V),
        Action::Coast => (s.y + s.v, (s.v - 1).max(V_MIN)),
    }
}

/// Where the resulting bird sits relative to the gap — the position
/// clause's quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PosBand {
    /// Below the gap — a crash state.
    Below,
    /// At the bottom edge of the gap.
    SqueezeBottom,
    /// In the gap, lower half.
    Lower,
    /// Dead center.
    Middle,
    /// In the gap, upper half.
    Upper,
    /// At the top edge of the gap.
    SqueezeTop,
    /// Above the gap — a crash state.
    Above,
}

pub fn pos_band(rel: i32, h: i32) -> PosBand {
    if rel < -h {
        PosBand::Below
    } else if rel == -h {
        PosBand::SqueezeBottom
    } else if rel < 0 {
        PosBand::Lower
    } else if rel == 0 {
        PosBand::Middle
    } else if rel < h {
        PosBand::Upper
    } else if rel == h {
        PosBand::SqueezeTop
    } else {
        PosBand::Above
    }
}

impl PosBand {
    pub fn clause(self) -> &'static str {
        match self {
            PosBand::Below => "sinks below the gap",
            PosBand::SqueezeBottom => "squeezes through the bottom of the gap",
            PosBand::Lower => "glides through the lower half of the gap",
            PosBand::Middle => "glides through the middle of the gap",
            PosBand::Upper => "glides through the upper half of the gap",
            PosBand::SqueezeTop => "squeezes through the top of the gap",
            PosBand::Above => "flies above the gap",
        }
    }
}

/// The resulting motion clause — this is what always separates the two
/// options of one state (flap's post-velocity is FLAP_V; coast's is
/// strictly lower).
pub fn motion_clause(v: i32) -> &'static str {
    match v {
        i32::MIN..=-2 => "falling fast",
        -1 => "falling",
        0 => "flying level",
        1 => "rising",
        _ => "climbing fast",
    }
}

/// v3's quantized OFFSET clause — fine post_rel relative to the gap
/// center, clamped at ±2 (recovers the sign and the 0/1/≥2 magnitude
/// classes without raw numbers; the band + gap width pin the rest).
pub fn offset_clause(rel: i32) -> &'static str {
    match rel {
        i32::MIN..=-2 => "under the center",
        -1 => "just under the center",
        0 => "at the center",
        1 => "just over the center",
        _ => "over the center",
    }
}

/// v3's NEUTRAL post-placement motion clause — kinematic magnitude +
/// direction anchored to the achieved height, never the v1 value-loaded
/// "rising"/"falling" wording (see GRAMMAR_ID's v3 note for the
/// action-revealing caveat this clause cannot escape).
pub fn post_motion_clause(v: i32) -> &'static str {
    match v {
        i32::MIN..=-2 => "drifting down two steps",
        -1 => "drifting down one step",
        0 => "holding this height",
        1 => "drifting up one step",
        _ => "drifting up two steps",
    }
}

fn gap_clause(h: i32) -> &'static str {
    match h {
        2 => "narrow",
        _ => "wide",
    }
}

/// The per-option sentence, grammar v3: `The bird {band}, {offset},
/// {post-motion}.` (Issue 876's widening; see GRAMMAR_ID).
pub fn render_option_sentence(s: &FlappyState, a: Action) -> String {
    let (y2, v2) = result(s, a);
    let rel2 = y2 - s.g;
    format!(
        "The bird {}, {}, {}.",
        pos_band(rel2, s.h).clause(),
        offset_clause(rel2),
        post_motion_clause(v2)
    )
}

/// The frozen v2 option sentence: `The bird {position clause}.` — the
/// position band ALONE. Retained verbatim because the committed v2
/// fixture (`flappy_oracle_laya_en_v2.jsonl`, the Bench 880/881 record)
/// drift-checks against THIS render; the live grammar moved on to v3.
pub fn render_option_sentence_v2(s: &FlappyState, a: Action) -> String {
    let (y2, _) = result(s, a);
    let band = pos_band(y2 - s.g, s.h);
    format!("The bird {}.", band.clause())
}

/// The state context sentence (the bird's current position/motion + the
/// gap, all in words — laya's recorded example shape: "The bird is a
/// little below the gap").
pub fn render_state_sentence(s: &FlappyState) -> String {
    let rel = s.y - s.g;
    let band = match rel.cmp(&0) {
        std::cmp::Ordering::Greater if rel > 1 => "well above",
        std::cmp::Ordering::Greater => "a little above",
        std::cmp::Ordering::Equal => "level with",
        std::cmp::Ordering::Less if rel < -1 => "well below",
        std::cmp::Ordering::Less => "a little below",
    };
    format!(
        "The bird is {band} the gap center, {}. The gap is {}. The pipe is just ahead.",
        motion_clause(s.v),
        gap_clause(s.h)
    )
}

/// The 8 frozen feature columns in pinned order (this ORDER is part of the
/// committed recipe — the head weights are only meaningful against it).
pub const FEATURE_NAMES: [&str; 8] = [
    "post_rel",
    "post_abs_rel",
    "post_v",
    "pre_rel",
    "pre_v",
    "in_gap",
    "edge_margin",
    "gap_half",
];

pub fn feature_row(s: &FlappyState, a: Action) -> [f64; 8] {
    let (y2, v2) = result(s, a);
    let rel2 = y2 - s.g;
    let abs2 = rel2.abs();
    [
        rel2 as f64,
        abs2 as f64,
        v2 as f64,
        (s.y - s.g) as f64,
        s.v as f64,
        if abs2 <= s.h { 1.0 } else { 0.0 },
        (s.h - abs2) as f64,
        s.h as f64,
    ]
}

fn in_bounds(y: i32) -> bool {
    (0..GRID_H).contains(&y)
}

/// The code-arithmetic context policy: the option whose result sits closest
/// to the gap center, pinned lowest-index tie-break (flap). This is what
/// "code does the arithmetic" means here — the same read the protocol asks
/// laya for, computed from raw geometry.
pub fn gap_center_pick(s: &FlappyState) -> usize {
    let mut best = 0usize;
    let mut best_abs = i32::MAX;
    for (i, &a) in ACTIONS.iter().enumerate() {
        let (y2, _) = result(s, a);
        let abs = (y2 - s.g).abs();
        if abs < best_abs {
            best_abs = abs;
            best = i;
        }
    }
    best
}

/// Seed the corpus with DECISION-INTERESTING states: the bird within ±4
/// cells of the gap center (where flap vs coast genuinely diverge), all
/// velocities, both gap widths. Uniform sampling would waste the corpus on
/// states where both options read identically. Deterministic; deduped.
pub fn enumerate_states(seed: u64, n: usize) -> Vec<(String, FlappyState)> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut out: Vec<(String, FlappyState)> = Vec::with_capacity(n);
    let mut seen = std::collections::HashSet::new();
    while out.len() < n {
        let h: i32 = if rng.u32(..2) == 0 { 2 } else { 3 };
        let g: i32 = rng.i32((h + 1)..(GRID_H - h));
        let y: i32 = g + rng.i32(-4..5);
        let v: i32 = rng.i32(V_MIN..(V_MAX + 1));
        let s = FlappyState { y, v, g, h };
        if !in_bounds(s.y) {
            continue;
        }
        // v == FLAP_V is the degenerate decision: flap and coast land on
        // the SAME cell (y + 2), so no state description can separate them
        // — excluded rather than committed as a constant-flap free win.
        if s.v == FLAP_V {
            continue;
        }
        // Both options must land in bounds (ground/ceiling states are out
        // of the micro arena's scope — the sentences describe gap-relative
        // geometry only).
        let (fy, cy) = (result(&s, Action::Flap).0, result(&s, Action::Coast).0);
        if !in_bounds(fy) || !in_bounds(cy) {
            continue;
        }
        // Same-band results make the two sentences IDENTICAL (the unbounded
        // Below/Above bands tie most often) — a state whose options the
        // grammar cannot separate is a degenerate decision and a
        // constant-flap free win; excluded like v == FLAP_V above.
        if pos_band(fy - g, h) == pos_band(cy - g, h) {
            continue;
        }
        if seen.insert(s) {
            out.push((format!("flappy_{:03}", out.len()), s));
        }
    }
    out
}

/// One flight under `policy`: seeded approach coast into each pipe, the
/// policy decides at one tick out, the crossing crashes on |rel| > h or on
/// leaving the grid. Returns pipes passed.
pub fn play_game(
    policy: &dyn Fn(&FlappyState) -> Action,
    rng: &mut fastrand::Rng,
    max_pipes: usize,
) -> u32 {
    let mut y: i32 = rng.i32(3..(GRID_H - 3));
    let mut v: i32 = rng.i32(-1..2);
    let mut pipes = 0u32;
    for _ in 0..max_pipes {
        // Approach: pure coasting with gravity (no decisions) until the
        // pipe is one tick away. Kept SHORT (2–3 ticks): a longer approach
        // drops the bird too far for any one decision to recover, and the
        // flight metric degenerates to arrival luck (measured, Bench 880).
        let approach: i32 = rng.i32(2..4);
        for _ in 0..approach {
            v = (v - 1).max(V_MIN);
            y += v;
            if !in_bounds(y) {
                return pipes;
            }
        }
        let h: i32 = if rng.u32(..2) == 0 { 2 } else { 3 };
        let g: i32 = rng.i32((h + 1)..(GRID_H - h));
        let s = FlappyState { y, v, g, h };
        let a = policy(&s);
        let (y2, v2) = result(&s, a);
        if !in_bounds(y2) {
            return pipes;
        }
        if (y2 - g).abs() > h {
            return pipes;
        }
        pipes += 1;
        y = y2;
        v = v2;
    }
    pipes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_order_and_labels_pinned() {
        assert_eq!(ACTIONS.len(), ACTION_LABELS.len());
        assert_eq!(ACTION_LABELS, ["flap", "coast"]);
    }

    #[test]
    fn pos_band_classification() {
        assert_eq!(pos_band(-4, 2), PosBand::Below);
        assert_eq!(pos_band(-2, 2), PosBand::SqueezeBottom);
        assert_eq!(pos_band(-1, 2), PosBand::Lower);
        assert_eq!(pos_band(0, 2), PosBand::Middle);
        assert_eq!(pos_band(1, 3), PosBand::Upper);
        assert_eq!(pos_band(3, 3), PosBand::SqueezeTop);
        assert_eq!(pos_band(4, 3), PosBand::Above);
    }

    #[test]
    fn option_sentences_always_differ_in_the_corpus() {
        // The grammar-v2 law, carried into v3: the enumerator excludes both
        // degenerate classes (v = FLAP_V lands both actions on one cell;
        // same-band results rendered identical V2 sentences), so every
        // committed state's two options are distinct sentences — and every
        // same-band or same-cell pair is an EXCLUDED state, never a fixture
        // row.
        for (id, s) in enumerate_states(607, 200) {
            assert_ne!(s.v, FLAP_V, "{id}: degenerate v enumerated");
            let f = render_option_sentence(&s, Action::Flap);
            let c = render_option_sentence(&s, Action::Coast);
            assert_ne!(f, c, "{id}: option sentences coincide");
            let (fy, _) = result(&s, Action::Flap);
            let (cy, _) = result(&s, Action::Coast);
            assert_ne!(fy, cy, "{id}: options on identical cells");
            assert_ne!(
                pos_band(fy - s.g, s.h),
                pos_band(cy - s.g, s.h),
                "{id}: options in the same band"
            );
        }
    }

    #[test]
    fn v3_option_sentences_differ_even_where_bands_would_tie() {
        // The v3 widening's law: the post-motion clause separates flap from
        // coast for EVERY non-degenerate state (flap's post-velocity is
        // FLAP_V = +2; coast's is strictly lower), so the band-tie exclusion
        // is no longer what keeps the two options' sentences apart — v3
        // would separate them even for the excluded same-band states.
        let s = FlappyState {
            y: 6,
            v: 1,
            g: 6,
            h: 3,
        };
        let (fy, _) = result(&s, Action::Flap);
        let (cy, _) = result(&s, Action::Coast);
        assert_eq!(
            pos_band(fy - s.g, s.h),
            pos_band(cy - s.g, s.h),
            "fixture: pick a state whose options share a band"
        );
        assert_ne!(s.v, FLAP_V, "fixture must be a non-degenerate state");
        assert_ne!(
            render_option_sentence(&s, Action::Flap),
            render_option_sentence(&s, Action::Coast),
            "v3 must separate the options by the motion clause alone"
        );
    }

    #[test]
    fn v2_render_is_frozen() {
        // The committed v2 fixture drift-checks against THIS render — any
        // wording change here breaks the Bench 880/881 provenance, so it is
        // pinned literally.
        let s = FlappyState {
            y: 4,
            v: 0,
            g: 6,
            h: 2,
        };
        assert_eq!(
            render_option_sentence_v2(&s, Action::Coast),
            "The bird squeezes through the bottom of the gap."
        );
        // And the v3 render of the same state adds the two new clauses
        // (coast from v=0: rel = −2 → "under the center", post-v = −1 →
        // "drifting down one step").
        assert_eq!(
            render_option_sentence(&s, Action::Coast),
            "The bird squeezes through the bottom of the gap, under the center, \
             drifting down one step."
        );
    }

    #[test]
    fn grammar_has_no_digits() {
        for (_, s) in enumerate_states(607, 50) {
            let st = render_state_sentence(&s);
            assert!(
                !st.chars().any(|c| c.is_ascii_digit()),
                "state sentence leaked a number: {st:?}"
            );
            for &a in &ACTIONS {
                let o = render_option_sentence(&s, a);
                assert!(
                    !o.chars().any(|c| c.is_ascii_digit()),
                    "option sentence leaked a number: {o:?}"
                );
            }
        }
    }

    #[test]
    fn enumeration_is_deterministic() {
        let a = enumerate_states(607, 100);
        let b = enumerate_states(607, 100);
        assert_eq!(a, b);
        // Deduped + decision-interesting: distinct states, bird near gap.
        let distinct: std::collections::HashSet<_> = a.iter().map(|(_, s)| *s).collect();
        assert_eq!(distinct.len(), a.len());
        for (_, s) in &a {
            assert!((s.y - s.g).abs() <= 4);
        }
    }

    #[test]
    fn play_game_is_deterministic_and_scores() {
        use self::gap_center_pick as flappy_sim_gap_center_pick;
        let run = |policy: &dyn Fn(&FlappyState) -> Action, seed: u64| {
            let mut rng = fastrand::Rng::with_seed(seed);
            play_game(policy, &mut rng, 100)
        };
        let always_flap = run(&|_| Action::Flap, 42);
        assert_eq!(always_flap, run(&|_| Action::Flap, 42));
        let gap_center = run(
            &|s: &FlappyState| ACTIONS[flappy_sim_gap_center_pick(s)],
            42,
        );
        // The code-arithmetic policy is the sanity ceiling: a policy that
        // always steers to the gap center must never do worse than a
        // constant action on the same streams.
        assert!(
            gap_center >= always_flap,
            "gap-center {gap_center} < always-flap {always_flap}"
        );
    }
}

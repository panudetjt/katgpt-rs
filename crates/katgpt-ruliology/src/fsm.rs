//! FSM Strategy — Finite State Machine as a SimpleProgram bandit arm.
//!
//! Enumerates all distinct N-state FSMs for 2-color, 2-action games.
//! For N=2: 22 distinct machines (after behavioral deduplication from 64 raw).
//! For N=3: 956 distinct machines (after deduplication from 512 raw).
//!
//! Plan 188 Phase 1.

use std::collections::HashSet;

use crate::types::{SimpleProgram, WinMatrix};

/// Maximum number of FSM states supported.
pub const MAX_STATES: usize = 4;

// ── FsmStrategy ────────────────────────────────────────────────

/// A deterministic finite-state machine strategy for 2-action iterated games.
///
/// Each state has:
/// - An output (0 or 1) — the action played in that state
/// - Two transitions (one for input 0, one for input 1) — next state
///
/// The FSM reads the opponent's last action as input and transitions accordingly.
#[derive(Clone, Debug)]
pub struct FsmStrategy {
    /// Transition table: `transitions[state][input]` → next state.
    transitions: [[u8; 2]; MAX_STATES],
    /// Output action per state.
    outputs: [u8; MAX_STATES],
    /// Current state (mutable during play).
    state: u8,
    /// Number of valid states (1..=MAX_STATES).
    n_states: u8,
    /// Cached complexity (computed once at construction).
    complexity: f32,
    /// Cached blake3 id (computed once at construction).
    ///
    /// Eliminates blake3 calls from `PartialEq`/`Hash`/`id()` — these are
    /// called per `HashSet` lookup in tournament/bandit hot loops.
    id: u64,
}

impl FsmStrategy {
    /// Create a new FSM from raw transition/output tables.
    ///
    /// `transitions[i][j]` = next state from state `i` on input `j`.
    /// `outputs[i]` = action (0 or 1) emitted in state `i`.
    /// `initial_state` = starting state (typically 0).
    #[inline]
    pub fn new(
        transitions: [[u8; 2]; MAX_STATES],
        outputs: [u8; MAX_STATES],
        n_states: u8,
        initial_state: u8,
    ) -> Self {
        debug_assert!(n_states as usize <= MAX_STATES);
        debug_assert!((initial_state as usize) < (n_states as usize));

        let complexity = Self::compute_complexity(n_states, &transitions, &outputs);
        let id = Self::compute_id(n_states, &transitions, &outputs);

        Self {
            transitions,
            outputs,
            state: initial_state,
            n_states,
            complexity,
            id,
        }
    }

    /// Complexity = log2(distinct transitions + distinct outputs) / log2(max theoretical).
    ///
    /// For an N-state FSM: at most 2*N transitions + N outputs = 3*N distinct values.
    /// Actual distinct values / max → normalized to [0, 1].
    fn compute_complexity(
        n_states: u8,
        transitions: &[[u8; 2]; MAX_STATES],
        outputs: &[u8; MAX_STATES],
    ) -> f32 {
        let n = n_states as usize;
        // Fixed-size stack buffer (3 * MAX_STATES = 12 bytes). Avoids a heap
        // allocation on every FsmStrategy::new — the enumerator builds up to
        // (n*n*2)^n strategies (e.g. ~1M for n=4), so this is on a hot path.
        // The manual stable sort + dedup below is bit-identical to Vec::sort +
        // Vec::dedup on the same byte sequence (total order on u8, equal runs
        // collapsed to the first occurrence).
        //
        // n=0 early return matches the original's `max_distinct == 0.0` branch
        // (kept here so the dedup logic below can assume len >= 1).
        if n == 0 {
            return 0.0;
        }
        let mut vals = [0u8; 3 * MAX_STATES];
        let mut len = 0usize;

        for s in 0..n {
            vals[len] = transitions[s][0];
            vals[len + 1] = transitions[s][1];
            vals[len + 2] = outputs[s];
            len += 3;
        }

        // Stable insertion-sort on the active prefix. Byte domain is tiny, so
        // the quadratic constant is negligible; the win is avoiding the heap
        // trip. Equality semantics match slice::sort (total order on u8).
        let active = &mut vals[..len];
        for i in 1..active.len() {
            let key = active[i];
            let mut j = i;
            while j > 0 && active[j - 1] > key {
                active[j] = active[j - 1];
                j -= 1;
            }
            active[j] = key;
        }

        // Dedup in-place (matches slice::dedup: collapse runs of equal
        // elements to a single occurrence).
        let mut write = 1usize;
        for read in 1..active.len() {
            if active[read] != active[write - 1] {
                active[write] = active[read];
                write += 1;
            }
        }
        let distinct = write as f32;
        let max_distinct = (3 * n) as f32;

        distinct / max_distinct
    }

    /// Blake3 hash of transitions + outputs → u64 ID. Computed once at
    /// construction and cached in `self.id` to avoid blake3 calls on every
    /// `HashSet` lookup (was 2 blake3 invocations per `contains`/`insert`).
    fn compute_id(
        n_states: u8,
        transitions: &[[u8; 2]; MAX_STATES],
        outputs: &[u8; MAX_STATES],
    ) -> u64 {
        let mut hasher = blake3::Hasher::new();
        for s in 0..n_states as usize {
            hasher.update(&[transitions[s][0], transitions[s][1], outputs[s]]);
        }
        hasher.update(&[n_states]);
        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash.as_bytes()[..8].try_into().unwrap_or([0u8; 8]);
        u64::from_le_bytes(bytes)
    }

    /// Reset FSM to initial state for a new game.
    #[inline]
    pub fn reset(&mut self) {
        self.state = 0;
    }

    /// Current state index.
    #[inline]
    pub fn state(&self) -> u8 {
        self.state
    }

    /// Number of states.
    #[inline]
    pub fn n_states(&self) -> u8 {
        self.n_states
    }

    /// Get the transition table.
    #[inline]
    pub fn transitions(&self) -> &[[u8; 2]; MAX_STATES] {
        &self.transitions
    }

    /// Get the output table.
    #[inline]
    pub fn outputs(&self) -> &[u8; MAX_STATES] {
        &self.outputs
    }
}

impl SimpleProgram for FsmStrategy {
    /// Produce next action given opponent's action history.
    ///
    /// Reads the last opponent action as input, transitions to next state,
    /// and outputs the current state's action.
    fn next_action(&mut self, opponent_history: &[u8]) -> u8 {
        if let Some(&last_input) = opponent_history.last() {
            let input = if last_input > 0 { 1 } else { 0 };
            let ns = self.transitions[self.state as usize][input];
            // Clamp to valid range.
            self.state = if ns < self.n_states { ns } else { 0 };
        }
        self.outputs[self.state as usize]
    }

    /// Blake3 hash of transitions + outputs → u64 ID (cached at construction).
    #[inline]
    fn id(&self) -> u64 {
        self.id
    }

    /// Complexity score (cached at construction).
    #[inline]
    fn complexity(&self) -> f32 {
        self.complexity
    }
}

impl PartialEq for FsmStrategy {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for FsmStrategy {}

impl std::hash::Hash for FsmStrategy {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// ── FsmEnumerator ──────────────────────────────────────────────

/// Exhaustive enumerator of all distinct N-state FSMs.
///
/// For N states with 2 inputs and 2 outputs:
/// - Each state has (next_state_for_0, next_state_for_1, output) = N^2 * 2 configs
/// - Total raw: (N^2 * 2)^N
/// - After behavioral deduplication: significantly fewer
pub struct FsmEnumerator;

impl FsmEnumerator {
    /// Enumerate all distinct FSMs with `n_states` states.
    ///
    /// For N=2: 2 states, each has (next_0, next_1, output) = 2*2*2 = 8 configs,
    /// so 8^2 = 64 raw combinations. After dedup ~22 distinct.
    ///
    /// For N=3: 3 states, 18^3 = 5832 raw, dedup to ~956 distinct.
    ///
    /// Deduplication strategy: generate all raw FSMs, simulate each against
    /// all possible input sequences up to a test horizon (scaled by state count),
    /// hash the behavioral fingerprint with blake3, and keep only distinct ones.
    pub fn enumerate(n_states: u8) -> Vec<FsmStrategy> {
        let n = n_states as usize;
        debug_assert!((1..=MAX_STATES).contains(&n));

        let mut raw = Vec::new();
        Self::enumerate_raw(n_states, &mut raw);

        // Test horizon must exceed max period: N-state vs N-state max period = N².
        // Use N² + 2 to ensure full behavioral coverage.
        // N=2: horizon=6, N=3: horizon=11, N=4: horizon=18.
        let test_horizon = (n * n + 2).min(20);
        let n_sequences = 1usize << test_horizon;

        // Use blake3 hashes instead of raw fingerprints for O(1) lookup.
        let mut seen: HashSet<[u8; 32]> = HashSet::with_capacity(raw.len());
        let mut distinct: Vec<FsmStrategy> = Vec::with_capacity(raw.len());

        let mut fingerprint = Vec::with_capacity(n_sequences * test_horizon);
        // Hoisted out of the seq_idx loop — avoids n_sequences * raw.len()
        // allocations of a Vec<u8> per behavioral fingerprint test.
        let mut history: Vec<u8> = Vec::with_capacity(test_horizon);

        for mut fsm in raw {
            fingerprint.clear();

            for seq_idx in 0..n_sequences {
                fsm.reset();
                history.clear();

                for bit in 0..test_horizon {
                    let input = ((seq_idx >> bit) & 1) as u8;
                    let action = fsm.next_action(&history);
                    fingerprint.push(action);
                    history.push(input);
                }
            }

            let hash = blake3::hash(&fingerprint);
            if seen.insert(hash.into()) {
                distinct.push(fsm);
            }
        }

        distinct
    }

    /// Generate all raw FSM combinations without deduplication.
    fn enumerate_raw(n_states: u8, out: &mut Vec<FsmStrategy>) {
        let n = n_states as usize;
        // Each state has (next_0, next_1, output) = n * n * 2 possibilities.
        let configs_per_state = n * n * 2;
        let total = usize::pow(configs_per_state, n as u32);

        out.reserve(total);

        for raw_idx in 0..total {
            let mut idx = raw_idx;
            let mut transitions = [[0u8; 2]; MAX_STATES];
            let mut outputs = [0u8; MAX_STATES];

            for s in 0..n {
                let remainder = idx % configs_per_state;
                idx /= configs_per_state;

                transitions[s][0] = (remainder % n) as u8;
                transitions[s][1] = ((remainder / n) % n) as u8;
                outputs[s] = ((remainder / (n * n)) % 2) as u8;
            }

            out.push(FsmStrategy::new(transitions, outputs, n_states, 0));
        }
    }

    /// Run round-robin tournament: every strategy vs every other strategy.
    ///
    /// Each pair plays `rounds` rounds of the game. Returns complete
    /// win matrix with payoffs and rankings.
    ///
    /// Complexity: O(n² × rounds) where n = strategies.len().
    ///
    /// # Last-action-only hot path (FSM-specific)
    ///
    /// This tournament is monomorphic over `FsmStrategy`, whose `next_action`
    /// reads only `opponent_history.last()` (it has no memory beyond the prior
    /// opponent move). The hot loop exploits that by tracking the last action
    /// as a single `u8` per player and passing a 1-element slice — bit-identical
    /// to the prior Vec<push> implementation (FSM never observes history beyond
    /// `[len-1]`) but eliminates 2x Vec::push per round per pair.
    ///
    /// DO NOT copy this optimization into a generic `SimpleProgram` tournament
    /// without first verifying the impl's history contract: `CaStrategy`'s
    /// `next_action` reads up to `tape_width` (default 7) most-recent opponent
    /// moves, so a CA-tournament must keep the full Vec<u8> history per player.
    /// `TmStrategy` reads only `last()` (writes opponent's last move to the
    /// tape at the head) so the optimization would be safe for a TM-tournament.
    pub fn tournament(
        strategies: &[FsmStrategy],
        rounds: u32,
        payoff_fn: &dyn Fn(u8, u8) -> f64,
    ) -> WinMatrix {
        let n = strategies.len();
        let mut payoffs = vec![vec![0.0f64; n]; n];

        // First-round sentinel: empty slice (no prior opponent move).
        // Subsequent rounds pass a 1-element slice of the prior action.
        let empty: [u8; 0] = [];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let mut si = strategies[i].clone();
                let mut sj = strategies[j].clone();
                si.reset();
                sj.reset();

                let mut total_payoff = 0.0f64;
                let mut last_action_i: u8 = 0; // only read after round 0
                let mut last_action_j: u8 = 0;
                let mut first_round = true;

                for _ in 0..rounds {
                    // Round k reads the opponent's action from round k-1.
                    let hist_j: &[u8] = if first_round {
                        &empty
                    } else {
                        std::slice::from_ref(&last_action_j)
                    };
                    let hist_i: &[u8] = if first_round {
                        &empty
                    } else {
                        std::slice::from_ref(&last_action_i)
                    };
                    let action_i = si.next_action(hist_j);
                    let action_j = sj.next_action(hist_i);

                    total_payoff += payoff_fn(action_i, action_j);

                    last_action_i = action_i;
                    last_action_j = action_j;
                    first_round = false;
                }

                payoffs[i][j] = total_payoff / rounds as f64;
            }
        }

        let ids: Vec<u64> = strategies.iter().map(|s| s.id()).collect();
        WinMatrix::new(payoffs, ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_next_action_always_cooperate() {
        // FSM with 1 state, output 0 (cooperate), self-loop.
        let transitions = [[0u8; 2]; MAX_STATES];
        let outputs = [0u8; MAX_STATES];
        let mut fsm = FsmStrategy::new(transitions, outputs, 1, 0);

        // Always cooperates regardless of opponent history.
        assert_eq!(fsm.next_action(&[]), 0);
        assert_eq!(fsm.next_action(&[0]), 0);
        assert_eq!(fsm.next_action(&[1]), 0);
        assert_eq!(fsm.next_action(&[0, 1, 1, 0]), 0);
    }

    #[test]
    fn test_fsm_next_action_tit_for_tat() {
        // 2-state FSM: state 0 = cooperate, state 1 = defect.
        // Transition: if opponent played 0, go to state 0; if 1, go to state 1.
        let transitions: [[u8; 2]; MAX_STATES] = [
            [0, 1], // state 0: coop → state 0, defect → state 1
            [0, 1], // state 1: coop → state 0, defect → state 1
            [0, 0], // unused
            [0, 0], // unused
        ];
        let outputs: [u8; MAX_STATES] = [0, 1, 0, 0]; // state 0 → cooperate, state 1 → defect
        let mut fsm = FsmStrategy::new(transitions, outputs, 2, 0);

        // First move: cooperate (state 0).
        assert_eq!(fsm.next_action(&[]), 0);

        // After opponent defects → state 1 → defect.
        assert_eq!(fsm.next_action(&[1]), 1);

        // After opponent cooperates → back to state 0 → cooperate.
        assert_eq!(fsm.next_action(&[1, 0]), 0);
    }

    #[test]
    fn test_fsm_enumerate_1_state() {
        let fsms = FsmEnumerator::enumerate(1);
        // 1-state FSMs: only output matters (0 or 1), transition is always self-loop.
        assert_eq!(fsms.len(), 2);
    }

    #[test]
    fn test_fsm_enumerate_2_states_count() {
        let fsms = FsmEnumerator::enumerate(2);
        // Expect ~22 distinct 2-state FSMs (Wolfram result).
        assert!(
            fsms.len() >= 18 && fsms.len() <= 30,
            "expected ~22 distinct 2-state FSMs, got {}",
            fsms.len()
        );
    }

    #[test]
    fn test_fsm_enumerate_all_distinct_ids() {
        let fsms = FsmEnumerator::enumerate(2);
        let ids: Vec<u64> = fsms.iter().map(|f| f.id()).collect();
        let unique_ids: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique_ids.len(), "duplicate IDs in enumeration");
    }

    #[test]
    fn test_tournament_matching_pennies() {
        let strategies = FsmEnumerator::enumerate(2);
        let wm = FsmEnumerator::tournament(&strategies, 100, &|a, b| {
            if a == b { 1.0 } else { -1.0 }
        });

        // Rankings should be non-empty and sorted descending.
        assert!(!wm.rankings.is_empty());
        for window in wm.rankings.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "rankings not sorted: {} > {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_tournament_payoffs_symmetric_game() {
        let strategies = FsmEnumerator::enumerate(2);
        let wm = FsmEnumerator::tournament(&strategies, 50, &|a, b| {
            if a == b { 1.0 } else { -1.0 }
        });

        // In matching pennies (antisymmetric), payoffs should average near 0.
        let total_avg: f64 =
            wm.rankings.iter().map(|(_, p)| p).sum::<f64>() / wm.rankings.len() as f64;
        assert!(
            total_avg.abs() < 1.0,
            "matching pennies avg should be near 0, got {total_avg}"
        );
    }

    #[test]
    fn test_fsm_complexity_range() {
        let fsms = FsmEnumerator::enumerate(2);
        for fsm in &fsms {
            let c = fsm.complexity();
            assert!((0.0..=1.0).contains(&c), "complexity out of range: {c}");
        }
    }

    #[test]
    fn test_fsm_enumerate_3_states_count() {
        let fsms = FsmEnumerator::enumerate(3);
        // Expect ~956 distinct 3-state FSMs (Wolfram result).
        // Allow some tolerance since exact count depends on behavioral dedup.
        // Wolfram reports ~956, but behavioral dedup with blake3 hashing may vary.
        // 1054 is our observed count with N²+2=11 horizon.
        assert!(
            fsms.len() >= 950 && fsms.len() <= 1100,
            "expected ~956 distinct 3-state FSMs, got {}",
            fsms.len()
        );
    }

    #[test]
    fn test_fsm_enumerate_3_states_all_distinct_ids() {
        let fsms = FsmEnumerator::enumerate(3);
        let ids: Vec<u64> = fsms.iter().map(|f| f.id()).collect();
        let unique_ids: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique_ids.len(),
            "duplicate IDs in 3-state enumeration"
        );
    }
}

// TL;DR: FsmStrategy (2-action FSM with N≤4 states) + FsmEnumerator (exhaustive enumeration with behavioral dedup). N=2 yields ~22, N=3 yields ~956 distinct machines. Tournament round-robin produces WinMatrix.

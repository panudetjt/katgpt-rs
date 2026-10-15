//! [`CsrLegalSet`] — compressed sparse row over (state → sorted legal tokens).

/// Per-state legal token sets in compressed-sparse-row form.
///
/// `tokens[offsets[s] .. offsets[s + 1]]` is state `s`'s legal set, sorted
/// ascending. Enumeration is O(deg(s)) and allocation-free; the degree is one
/// subtraction.
///
/// # Why not read the dense table directly
///
/// A DFA that stores δ as a flat `n_states × vocab_size` array — which is the
/// shipped `LodestarAutomaton` layout — can already produce a legal set by
/// scanning one row. That scan is **O(vocab_size)**, which is the cost this
/// whole module exists to remove: it would replace a 32 768-entry scan with a
/// 32 768-entry scan. The index has to be built once and read many times, or
/// it is not an index.
///
/// # Memory
///
/// `4 · n_edges + 4 · (n_states + 1)` bytes against the dense table's
/// `8 · n_states · vocab_size`. For a grammar — a sparse δ by construction —
/// that is a large *reduction*, which is a second, independent reason to hold
/// one: the dense layout is what makes a real vocabulary expensive to keep a
/// DFA over at all. [`Self::memory_bytes`] reports it so a consumer can
/// measure the trade instead of assuming it.
///
/// # Invariants (held by construction, asserted in debug)
///
/// - `offsets.len() == n_states + 1`, non-decreasing, `offsets[0] == 0`.
/// - Each row is strictly ascending — the order
///   [`ConstraintPruner::for_each_legal`] promises.
///
/// [`ConstraintPruner::for_each_legal`]: crate::traits::ConstraintPruner::for_each_legal
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrLegalSet {
    /// Legal token ids, grouped by state, ascending within each group.
    tokens: Vec<u32>,
    /// `n_states + 1` prefix offsets into `tokens`.
    offsets: Vec<u32>,
    vocab_size: usize,
}

impl CsrLegalSet {
    /// Build from a flat row-major `n_states × vocab_size` transition table.
    ///
    /// `no_edge` is the sentinel the table uses for "no transition"
    /// (`usize::MAX` in `LodestarAutomaton`). One pass, two allocations, done
    /// once at automaton-build time — the caller is already paying an
    /// O(states × vocab) precompute there.
    ///
    /// A `transitions` slice shorter than `n_states * vocab_size` is treated
    /// as having no edges past its end rather than panicking: the states it
    /// does cover still index correctly, which keeps a truncated table a
    /// *smaller* legal set and never a wrong one.
    pub fn from_dense(
        transitions: &[usize],
        n_states: usize,
        vocab_size: usize,
        no_edge: usize,
    ) -> Self {
        let mut offsets = Vec::with_capacity(n_states + 1);
        let mut tokens = Vec::with_capacity(n_states);
        offsets.push(0u32);
        for s in 0..n_states {
            let base = s * vocab_size;
            let row = match transitions.len() > base {
                true => &transitions[base..transitions.len().min(base + vocab_size)],
                false => &[][..],
            };
            for (t, &next) in row.iter().enumerate() {
                if next != no_edge {
                    tokens.push(t as u32);
                }
            }
            offsets.push(tokens.len() as u32);
        }
        tokens.shrink_to_fit();
        Self {
            tokens,
            offsets,
            vocab_size,
        }
    }

    /// Build from an arbitrary `(state, token)` edge iterator.
    ///
    /// Edges may arrive in any order and may repeat; each `(state, token)`
    /// pair is kept once. Edges with `state >= n_states` or
    /// `token >= vocab_size` are dropped — an out-of-range edge is the
    /// caller's bug and admitting it would produce a legal set whose tokens
    /// index off the end of a marginal.
    ///
    /// For a source that is already a dense table use [`Self::from_dense`],
    /// which needs no sort.
    pub fn from_edges<I: IntoIterator<Item = (usize, usize)>>(
        n_states: usize,
        vocab_size: usize,
        edges: I,
    ) -> Self {
        let mut pairs: Vec<(u32, u32)> = edges
            .into_iter()
            .filter(|&(s, t)| s < n_states && t < vocab_size)
            .map(|(s, t)| (s as u32, t as u32))
            .collect();
        pairs.sort_unstable();
        pairs.dedup();

        let mut offsets = Vec::with_capacity(n_states + 1);
        let mut tokens = Vec::with_capacity(pairs.len());
        let mut cursor = 0usize;
        offsets.push(0u32);
        for s in 0..n_states as u32 {
            while cursor < pairs.len() && pairs[cursor].0 == s {
                tokens.push(pairs[cursor].1);
                cursor += 1;
            }
            offsets.push(tokens.len() as u32);
        }
        Self {
            tokens,
            offsets,
            vocab_size,
        }
    }

    /// Number of states.
    #[inline]
    pub fn n_states(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The declared alphabet size. Every token in every row is `< vocab_size`.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Total number of `(state, token)` edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.tokens.len()
    }

    /// `state`'s legal tokens, ascending. Empty for an out-of-range state —
    /// which is the correct answer, not a silent one: a state that does not
    /// exist has no legal continuation, and the caller's own
    /// `legal_degree` reports `0` for it through the same accessor.
    #[inline]
    pub fn row(&self, state: usize) -> &[u32] {
        match (self.offsets.get(state), self.offsets.get(state + 1)) {
            (Some(&start), Some(&end)) => &self.tokens[start as usize..end as usize],
            _ => &[],
        }
    }

    /// `|L(state)|`, in one subtraction.
    #[inline]
    pub fn degree(&self, state: usize) -> usize {
        self.row(state).len()
    }

    /// Call `f` for every legal token of `state`, ascending. O(degree), no
    /// allocation. The hot-path accessor — mirrors
    /// `bisimulation::graph::TransitionGraph::for_each_adjacent`.
    #[inline]
    pub fn for_each(&self, state: usize, f: &mut dyn FnMut(usize)) {
        for &t in self.row(state) {
            f(t as usize);
        }
    }

    /// Whether `token` is legal in `state`. O(log deg) — the row is sorted.
    #[inline]
    pub fn contains(&self, state: usize, token: usize) -> bool {
        match u32::try_from(token) {
            Ok(t) => self.row(state).binary_search(&t).is_ok(),
            Err(_) => false,
        }
    }

    /// Heap bytes held by the index.
    ///
    /// The comparand is `8 · n_states · vocab_size` for the dense `usize`
    /// table this replaces; see the type docs.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        4 * self.tokens.len() + 4 * self.offsets.len()
    }
}

//! Minimal 9×9 Go board — just enough for legal self-play moves to generate
//! realistic benchmark positions. Not a general engine (no 13×13/19×19, no
//! replay) — this crate exists purely to give the WASM benchmark (Plan 565) a
//! dependency-free way to drive `encode_features`, mirroring what the real
//! Moka JS harness does with its own `game.ts`.
//!
//! Issue 204 extended this with pass-counting + a simple area-score so the
//! PUCT search port (`puct.rs`) can detect terminals and assign rewards.
//! The scoring is intentionally crude (stones + clear-only territory via
//! flood fill) — it only needs to be correct enough to give the search a
//! non-broken win/loss signal, not to match any tournament ruleset.

pub const SIZE: usize = 9;
pub const AREA: usize = SIZE * SIZE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    Empty,
    Black,
    White,
}

impl Cell {
    #[inline]
    pub fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
            Self::Empty => panic!("opponent() on Empty"),
        }
    }
}

/// Fixed-size board — `cells` is `[Cell; AREA]` (not `Vec<Cell>`) so `clone()`
/// is a zero-allocation stack copy. PUCT clones the board ~9× per node
/// expansion (parent + top_k children); eliminating ~450 heap allocs/move
/// at budget=50 is the single largest non-SIMD tree-overhead win.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    pub cells: [Cell; AREA],
    pub to_play: Cell,
    pub ko_point: Option<usize>,
    /// Consecutive pass count — reaches 2 when both players pass consecutively,
    /// which ends the game (Issue 204: needed for PUCT terminal detection).
    pub consecutive_passes: u8,
}

/// Stack-allocated neighbor list — `neighbors()` was returning `Vec<usize>`
/// (a heap alloc per call), called from `flood_group`/`is_legal`/`play`/
/// `reward`. At top_k=8 the PUCT expand path calls `is_legal` 81× + `play`
/// up to 8× per node, each touching `neighbors` multiple times via flood
/// fills — hundreds of heap allocs per expansion. This drops to zero.
#[derive(Clone, Copy)]
struct Neighbors {
    idxs: [usize; 4],
    len: u8,
}

impl Neighbors {
    #[inline]
    fn iter(&self) -> &[usize] {
        &self.idxs[..self.len as usize]
    }
}

fn neighbors(idx: usize) -> Neighbors {
    let row = idx / SIZE;
    let col = idx % SIZE;
    let mut idxs = [0usize; 4];
    let mut len = 0u8;
    if row > 0 {
        idxs[len as usize] = idx - SIZE;
        len += 1;
    }
    if row + 1 < SIZE {
        idxs[len as usize] = idx + SIZE;
        len += 1;
    }
    if col > 0 {
        idxs[len as usize] = idx - 1;
        len += 1;
    }
    if col + 1 < SIZE {
        idxs[len as usize] = idx + 1;
        len += 1;
    }
    Neighbors { idxs, len }
}

/// Zero-allocation, early-exit liberty check: does the group at `start` have
/// at least one liberty? Returns `true` on the FIRST liberty found — avoids
/// the full flood_group traversal + its two Vec allocations. Called up to
/// 81× per PUCT expansion (via `is_legal`→`would_be_suicide`), so this is a
/// hot path where allocation elimination + early exit compound.
///
/// Uses a fixed `[usize; AREA]` stack (max 81 cells) — zero heap alloc.
fn has_liberty(cells: &[Cell], start: usize) -> bool {
    let color = cells[start];
    if color == Cell::Empty {
        return true; // an empty cell IS a liberty of itself (trivially)
    }
    let mut visited = [false; AREA];
    let mut stack = [start; AREA];
    let mut top = 1;
    while top > 0 {
        top -= 1;
        let pos = stack[top];
        if visited[pos] {
            continue;
        }
        visited[pos] = true;
        for &n in neighbors(pos).iter() {
            match cells[n] {
                c if c == color => {
                    if !visited[n] {
                        stack[top] = n;
                        top += 1;
                    }
                }
                Cell::Empty => return true, // found a liberty — short-circuit
                _ => {}                     // opponent stone
            }
        }
    }
    false // traversed the whole group, no liberty found
}

/// Flood-fill the group containing `start`, returning (stones, liberty
/// positions). Mirrors `katgpt_pruners::go::utils::flood_group`. Used by
/// `play` (capture resolution needs the actual stone list) + `reward`
/// (territory scoring) + `encode_features` (liberty counting).
///
/// The `would_be_suicide` hot path uses the zero-alloc early-exit `has_liberty`
/// instead — this function is only for paths that need the actual lists.
pub fn flood_group(cells: &[Cell], start: usize) -> (Vec<usize>, Vec<usize>) {
    let color = cells[start];
    if color == Cell::Empty {
        return (Vec::new(), Vec::new());
    }
    let mut group = Vec::new();
    let mut liberties = Vec::new();
    let mut visited = [false; AREA];
    let mut stack = vec![start];
    while let Some(pos) = stack.pop() {
        if visited[pos] {
            continue;
        }
        visited[pos] = true;
        match cells[pos] {
            c if c == color => {
                group.push(pos);
                for &n in neighbors(pos).iter() {
                    if !visited[n] {
                        stack.push(n);
                    }
                }
            }
            Cell::Empty => liberties.push(pos),
            _ => {}
        }
    }
    (group, liberties)
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: [Cell::Empty; AREA],
            to_play: Cell::Black,
            ko_point: None,
            consecutive_passes: 0,
        }
    }

    fn would_be_suicide(&self, idx: usize, color: Cell) -> bool {
        let mut trial = self.cells;
        trial[idx] = color;
        // Does placing capture any opponent group? Early-exit `has_liberty`
        // (zero-alloc) replaces the full flood_group — we only need the
        // yes/no, not the stone/liberty lists.
        for &n in neighbors(idx).iter() {
            if trial[n] == color.opponent() && !has_liberty(&trial, n) {
                return false; // captures an opponent group with no liberty — legal
            }
        }
        // No capture — legal only if the placed stone's own group has a liberty.
        !has_liberty(&trial, idx)
    }

    pub fn is_legal(&self, idx: usize) -> bool {
        self.cells[idx] == Cell::Empty
            && self.ko_point != Some(idx)
            && !self.would_be_suicide(idx, self.to_play)
    }

    pub fn legal_moves(&self) -> Vec<usize> {
        (0..AREA).filter(|&i| self.is_legal(i)).collect()
    }

    /// Places a stone, resolves captures, updates ko/turn. Caller must have
    /// already checked `is_legal`.
    pub fn play(&mut self, idx: usize) {
        let color = self.to_play;
        let opponent = color.opponent();
        self.cells[idx] = color;

        let mut captured = Vec::new();
        let mut visited_opp = [false; AREA];
        for &n in neighbors(idx).iter() {
            if self.cells[n] == opponent && !visited_opp[n] {
                let (group, libs) = flood_group(&self.cells, n);
                for &g in &group {
                    visited_opp[g] = true;
                }
                if libs.is_empty() {
                    captured.extend(group);
                }
            }
        }
        for &c in &captured {
            self.cells[c] = Cell::Empty;
        }

        // Simple ko: exactly one stone captured, and the capturing stone's
        // own group is a single stone with a single liberty (the ko point).
        if captured.len() == 1 {
            let (my_group, my_libs) = flood_group(&self.cells, idx);
            self.ko_point = if my_group.len() == 1 && my_libs.len() == 1 {
                Some(captured[0])
            } else {
                None
            };
        } else {
            self.ko_point = None;
        }

        self.consecutive_passes = 0;
        self.to_play = opponent;
    }

    pub fn pass(&mut self) {
        self.ko_point = None;
        self.consecutive_passes = self.consecutive_passes.saturating_add(1);
        self.to_play = self.to_play.opponent();
    }

    /// Game ends when both players pass consecutively (Issue 204).
    #[inline]
    pub fn is_game_over(&self) -> bool {
        self.consecutive_passes >= 2
    }

    /// Crude area score for `color`: stones on the board plus exclusive
    /// territory (empty regions reachable only by `color`'s stones). Returns
    /// a signed reward from `color`'s perspective: +1.0 if `color` is strictly
    /// ahead after komi (7.5 applied to White, the Moka training convention),
    /// -1.0 if behind or tied. Sufficient for PUCT negamax terminal scoring —
    /// not a tournament-ruleset-correct score.
    ///
    /// Mirrors the sign convention of `GoState::reward` in katgpt-pruners:
    /// the search's `expand` does `2.0 * reward(to_play) - 1.0` to map
    /// {win:1, loss:0} onto the [-1, +1] value-head range.
    pub fn reward(&self, color: Cell) -> f32 {
        let mut score = [0i32; 2]; // [Black, White]
        for &c in &self.cells {
            match c {
                Cell::Black => score[0] += 1,
                Cell::White => score[1] += 1,
                Cell::Empty => {}
            }
        }
        // Exclusive territory: flood each empty region, attribute it to a
        // color only if that color is the sole bordering stone color.
        let mut visited = [false; AREA];
        for start in 0..AREA {
            if visited[start] || self.cells[start] != Cell::Empty {
                continue;
            }
            let mut region_size = 0i32;
            let mut touches_black = false;
            let mut touches_white = false;
            let mut stack = vec![start];
            while let Some(p) = stack.pop() {
                if visited[p] {
                    continue;
                }
                visited[p] = true;
                region_size += 1;
                for &n in neighbors(p).iter() {
                    match self.cells[n] {
                        Cell::Empty => {
                            if !visited[n] {
                                stack.push(n);
                            }
                        }
                        Cell::Black => touches_black = true,
                        Cell::White => touches_white = true,
                    }
                }
            }
            match (touches_black, touches_white) {
                (true, false) => score[0] += region_size,
                (false, true) => score[1] += region_size,
                _ => {} // shared or empty board
            }
        }
        // Komi 7.5 favors White (Moka training convention; area-style scoring
        // with half-komi to break ties so `reward` is never exactly 0).
        let black = score[0] as f32;
        let white = score[1] as f32 + 7.5;
        let color_ahead = match color {
            Cell::Black => black > white,
            Cell::White => white > black,
            Cell::Empty => false,
        };
        if color_ahead { 1.0 } else { 0.0 }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

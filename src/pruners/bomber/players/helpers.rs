//! Shared player helpers — utility functions used by all bomber player types.
//!
//! Extracted from `players.rs` (Issue 175) to keep `mod.rs` under the 2048-line
//! soft limit. These are stateless free functions: position prediction, blast
//! zone checks, escape route search, policy-based action scoring, and LoRA
//! inference helpers.

use super::{
    ArenaGrid, BOMB_FUSE_TICKS, BomberAction, DEFAULT_BLAST_RANGE, GameEvent, GridPos, KnownBomb,
    KnownOpponent,
};
use crate::pruners::bomber::{ARENA_H, ARENA_W, Cell};

#[cfg(feature = "bomber-wasm")]
use super::{ACTION_COUNT, ALL_ACTIONS};

#[cfg(feature = "bomber-wasm")]
use crate::types::{LoraAdapter, lora_apply};

// ── Movement & Action Indexing ─────────────────────────────────

/// Compute target position after applying a move action.
pub(crate) fn move_target(action: &BomberAction, pos: GridPos) -> GridPos {
    match action {
        BomberAction::Up => GridPos {
            x: pos.x,
            y: pos.y - 1,
        },
        BomberAction::Down => GridPos {
            x: pos.x,
            y: pos.y + 1,
        },
        BomberAction::Left => GridPos {
            x: pos.x - 1,
            y: pos.y,
        },
        BomberAction::Right => GridPos {
            x: pos.x + 1,
            y: pos.y,
        },
        BomberAction::Bomb | BomberAction::Wait | BomberAction::Detonate => pos,
    }
}

/// Convert action to index 0..7.
pub(crate) fn action_index(action: &BomberAction) -> usize {
    match action {
        BomberAction::Up => 0,
        BomberAction::Down => 1,
        BomberAction::Left => 2,
        BomberAction::Right => 3,
        BomberAction::Bomb => 4,
        BomberAction::Wait => 5,
        BomberAction::Detonate => 6,
    }
}

/// Convert index 0..7 to action.
pub(crate) fn index_to_action(idx: usize) -> BomberAction {
    match idx {
        0 => BomberAction::Up,
        1 => BomberAction::Down,
        2 => BomberAction::Left,
        3 => BomberAction::Right,
        4 => BomberAction::Bomb,
        5 => BomberAction::Wait,
        6 => BomberAction::Detonate,
        _ => BomberAction::Wait,
    }
}

/// Manhattan distance between two grid positions.
#[allow(dead_code)]
pub(crate) fn manhattan(a: GridPos, b: GridPos) -> i32 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

// ── Blast Zone & Bomb Tracking ─────────────────────────────────

/// Check if position is in the blast zone of any known bomb.
/// Accounts for walls blocking blast propagation (blast stops at walls).
pub(crate) fn in_blast_zone(pos: GridPos, grid: &ArenaGrid, bombs: &[KnownBomb]) -> bool {
    for &(bomb_pos, range, _fuse) in bombs {
        if is_in_single_blast(pos, grid, bomb_pos, range) {
            return true;
        }
    }
    false
}

/// Check if position is in the blast zone of a single bomb (with wall blocking).
pub(crate) fn is_in_single_blast(
    pos: GridPos,
    grid: &ArenaGrid,
    bomb_pos: (i32, i32),
    range: u32,
) -> bool {
    let bx = bomb_pos.0;
    let by = bomb_pos.1;

    // Standing on the bomb itself
    if pos.x == bx && pos.y == by {
        return true;
    }

    // Same row (horizontal blast)
    if pos.y == by {
        let dx = pos.x - bx;
        if dx.unsigned_abs() <= range {
            let step = dx.signum();
            let mut x = bx + step;
            while x != pos.x {
                if let Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) =
                    grid.get(x, by)
                {
                    return false;
                }
                x += step;
            }
            return true;
        }
    }

    // Same column (vertical blast)
    if pos.x == bx {
        let dy = pos.y - by;
        if dy.unsigned_abs() <= range {
            let step = dy.signum();
            let mut y = by + step;
            while y != pos.y {
                if let Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) =
                    grid.get(bx, y)
                {
                    return false;
                }
                y += step;
            }
            return true;
        }
    }

    false
}

/// Update known bomb list from events.
pub(crate) fn update_bombs(bombs: &mut Vec<KnownBomb>, events: &[GameEvent]) {
    // Decrement fuses each tick (called once per select_action)
    for bomb in bombs.iter_mut() {
        bomb.2 = bomb.2.saturating_sub(1);
    }
    for event in events {
        match event {
            GameEvent::BombPlaced { pos, .. } => {
                if !bombs.iter().any(|(p, _, _)| *p == *pos) {
                    bombs.push((*pos, DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
                }
            }
            GameEvent::BombExploded { pos, .. } => {
                bombs.retain(|(p, _, _)| *p != *pos);
            }
            _ => {}
        }
    }
}

/// Update known power-up list from events (revealed/collected).
pub(crate) fn update_powerups(powerups: &mut Vec<(i32, i32)>, events: &[GameEvent]) {
    for event in events {
        match event {
            GameEvent::PowerUpRevealed { pos, .. } => {
                if !powerups.contains(pos) {
                    powerups.push(*pos);
                }
            }
            GameEvent::PowerUpCollected { pos, .. } => {
                powerups.retain(|p| p != pos);
            }
            _ => {}
        }
    }
}

/// Track opponent positions from PlayerMoved and BombPlaced events.
/// Stores `(player_id, current_pos, prev_pos)` for trajectory prediction.
pub(crate) fn update_opponents(
    opponents: &mut Vec<KnownOpponent>,
    events: &[GameEvent],
    my_id: u8,
) {
    for event in events {
        match event {
            GameEvent::PlayerMoved { player, to, .. } => {
                if *player == my_id {
                    continue;
                }
                if let Some(entry) = opponents.iter_mut().find(|(p, _, _)| *p == *player) {
                    entry.2 = Some(entry.1);
                    entry.1 = *to;
                } else {
                    opponents.push((*player, *to, None));
                }
            }
            GameEvent::BombPlaced { player, pos } => {
                if *player == my_id {
                    continue;
                }
                if let Some(entry) = opponents.iter_mut().find(|(p, _, _)| *p == *player) {
                    entry.2 = Some(entry.1);
                    entry.1 = *pos;
                } else {
                    opponents.push((*player, *pos, None));
                }
            }
            GameEvent::PlayerKilled { victim, .. } => {
                opponents.retain(|(p, _, _)| *p != *victim);
            }
            _ => {}
        }
    }
}

/// Predict opponent's next position from trajectory (prev → current → next).
pub(crate) fn predict_direction(
    current: (i32, i32),
    prev: Option<(i32, i32)>,
) -> Option<(i32, i32)> {
    let (cx, cy) = current;
    let (px, py) = prev?;
    let dx = cx - px;
    let dy = cy - py;
    if dx == 0 && dy == 0 {
        return None;
    }
    Some((cx + dx, cy + dy))
}

/// Count walkable neighbors (escape routes) from a position.
pub(crate) fn count_escape_routes(pos: (i32, i32), grid: &ArenaGrid) -> usize {
    [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
        .iter()
        .filter(|&&(dx, dy)| grid.is_walkable(pos.0 + dx, pos.1 + dy))
        .count()
}

/// Score a bomb placement by how trapped the opponent would be.
/// Higher score = fewer opponent escape routes + blast coverage.
pub(crate) fn trap_score(
    bomb_pos: (i32, i32),
    opponent_pos: (i32, i32),
    grid: &ArenaGrid,
    blast_range: u32,
) -> f32 {
    let dist = (bomb_pos.0 - opponent_pos.0).abs() + (bomb_pos.1 - opponent_pos.1).abs();
    if dist > blast_range as i32 + 3 {
        return 0.0;
    }

    let mut score = 0.0;

    // Bonus: opponent is within blast range
    if is_in_single_blast(
        GridPos {
            x: opponent_pos.0,
            y: opponent_pos.1,
        },
        grid,
        bomb_pos,
        blast_range,
    ) {
        score += 4.0;
    }

    // Penalty: more escape routes = harder to trap
    let routes = count_escape_routes(opponent_pos, grid);
    match routes {
        0 => score += 3.0,
        1 => score += 2.0,
        2 => score += 0.5,
        _ => {}
    }

    // Closeness bonus
    if dist <= 2 {
        score += 1.0;
    }

    score
}

/// Score movement toward intercepting an opponent's predicted path.
pub(crate) fn intercept_score(
    my_target: (i32, i32),
    opponent_pos: (i32, i32),
    predicted_pos: Option<(i32, i32)>,
) -> f32 {
    let current_dist = (my_target.0 - opponent_pos.0).abs() + (my_target.1 - opponent_pos.1).abs();

    if let Some((px, py)) = predicted_pos {
        let predicted_dist = (my_target.0 - px).abs() + (my_target.1 - py).abs();
        if predicted_dist < current_dist {
            return 1.0;
        }
    }

    0.0
}

/// Check if player has an escape route after placing a bomb at `new_bomb_pos`.
/// BFS from `player_pos` — must reach a cell outside ALL blast zones within
/// `blast_range + 1` steps. Accounts for bomb entities blocking movement.
pub(crate) fn has_escape_route(
    grid: &ArenaGrid,
    player_pos: GridPos,
    new_bomb_pos: (i32, i32),
    blast_range: u32,
    existing_bombs: &[KnownBomb],
) -> bool {
    let max_steps = blast_range as i32 + 1;
    // Fixed-size visited bitmap for the 13×13 arena — replaces HashSet allocation
    // (this runs per is_safe_action(Bomb) per tick per player).
    let mut visited = [false; ARENA_W * ARENA_H];
    // Fixed-capacity FIFO replaces the heap `VecDeque`. Every cell is marked
    // before being pushed, so at most `ARENA_W * ARENA_H` in-bounds cells enter
    // the queue, plus the seed cell (which may be out of bounds and therefore
    // unmarkable) — hence the `+ 1`. Pop order is identical to `pop_front`.
    let mut queue = [((0i32, 0i32), 0i32); ARENA_W * ARENA_H + 1];
    let mut head = 0usize;
    let mut tail = 0usize;

    // Inline bomb-entity blocking check: linear scan over existing bombs +
    // the new bomb position. Avoids allocating a HashSet and a Vec<KnownBomb>.
    let is_blocked = |x: i32, y: i32| {
        if x == new_bomb_pos.0 && y == new_bomb_pos.1 {
            return true;
        }
        existing_bombs.iter().any(|(p, _, _)| p.0 == x && p.1 == y)
    };

    // Blast-zone test against `existing_bombs` followed by the hypothetical new
    // bomb. This is exactly what the old `all_bombs = existing_bombs.to_vec() +
    // push(new)` list produced (same bombs, same short-circuit order) minus the
    // per-call `Vec` allocation — this function runs up to 8× per tick per player.
    let in_any_blast = |p: GridPos| {
        in_blast_zone(p, grid, existing_bombs)
            || is_in_single_blast(p, grid, new_bomb_pos, blast_range)
    };

    let mark = |visited: &mut [bool; ARENA_W * ARENA_H], x: i32, y: i32| {
        if x >= 0 && (x as usize) < ARENA_W && y >= 0 && (y as usize) < ARENA_H {
            visited[(y as usize) * ARENA_W + (x as usize)] = true;
        }
    };
    let is_visited = |visited: &[bool; ARENA_W * ARENA_H], x: i32, y: i32| {
        if x >= 0 && (x as usize) < ARENA_W && y >= 0 && (y as usize) < ARENA_H {
            visited[(y as usize) * ARENA_W + (x as usize)]
        } else {
            true
        }
    };

    queue[tail] = ((player_pos.x, player_pos.y), 0);
    tail += 1;
    mark(&mut visited, player_pos.x, player_pos.y);

    while head < tail {
        let ((cx, cy), steps) = queue[head];
        head += 1;
        if steps > max_steps {
            continue;
        }

        // Is this cell safe from ALL bombs (with wall blocking)?
        if !in_any_blast(GridPos { x: cx, y: cy }) {
            return true;
        }

        // Expand neighbors (avoid bomb entities blocking movement)
        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if is_visited(&visited, nx, ny) {
                continue;
            }
            // Mark first, then gate on walkable/blocked — matches original
            // HashSet::insert semantics (unwalkable cells still get marked).
            mark(&mut visited, nx, ny);
            if grid.is_walkable(nx, ny) && !is_blocked(nx, ny) {
                queue[tail] = ((nx, ny), steps + 1);
                tail += 1;
            }
        }
    }

    false
}

/// Check if an action is safe given the current state.
/// Uses wall-aware blast zone checks and accounts for bomb entities blocking movement.
pub fn is_safe_action(
    action: &BomberAction,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[KnownBomb],
) -> bool {
    match action {
        BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right => {
            let target = move_target(action, pos);
            if !grid.is_walkable(target.x, target.y) {
                return false;
            }
            // Don't walk into blast zone (walls block blast)
            !in_blast_zone(target, grid, bombs)
        }
        BomberAction::Bomb => {
            // Player stands ON the bomb but moves away next tick — check escape
            // from each adjacent cell (mirrors should_place_bomb logic).
            [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
                .iter()
                .any(|&(dx, dy)| {
                    let nx = pos.x + dx;
                    let ny = pos.y + dy;
                    grid.is_walkable(nx, ny)
                        && has_escape_route(
                            grid,
                            GridPos { x: nx, y: ny },
                            (pos.x, pos.y),
                            DEFAULT_BLAST_RANGE,
                            bombs,
                        )
                })
        }
        BomberAction::Wait => {
            // Waiting is only safe if not in blast zone
            !in_blast_zone(pos, grid, bombs)
        }
        BomberAction::Detonate => {
            // Detonate is only valid when active bombs exist and player won't be
            // caught in the resulting blast (no bomb movement, but blast affects player).
            // Future: restrict to Remote bombs only once bomb_type is tracked in KnownBomb.
            !bombs.is_empty() && !in_blast_zone(pos, grid, bombs)
        }
    }
}

/// Check if player should place a bomb at current position.
///
/// The player stands ON the bomb but moves away next tick, so escape is
/// checked from adjacent cells — not from the bomb position itself.
/// Accounts for existing bombs' blast zones and bomb entities blocking movement.
pub(crate) fn should_place_bomb(grid: &ArenaGrid, pos: GridPos, bombs: &[KnownBomb]) -> bool {
    // Don't place if already in a blast zone (walls may block, but be safe)
    if in_blast_zone(pos, grid, bombs) {
        return false;
    }

    // Don't place if there's already a bomb here
    if bombs.iter().any(|(p, _, _)| p.0 == pos.x && p.1 == pos.y) {
        return false;
    }

    // Player will move to an adjacent cell next tick (1 step used).
    // From that cell, has_escape_route checks if safety is reachable within
    // max_steps (3) — total 4 steps matches BOMB_FUSE_TICKS.
    let neighbors = [(0i32, -1), (0, 1), (-1, 0), (1, 0)];
    neighbors.iter().any(|&(dx, dy)| {
        let nx = pos.x + dx;
        let ny = pos.y + dy;
        grid.is_walkable(nx, ny)
            && has_escape_route(
                grid,
                GridPos { x: nx, y: ny },
                (pos.x, pos.y),
                DEFAULT_BLAST_RANGE,
                bombs,
            )
    })
}

// ── Policy Scoring ─────────────────────────────────────────────

/// True if action reverses the previous direction.
pub(crate) fn is_reverse(action: BomberAction, prev: Option<BomberAction>) -> bool {
    matches!(
        (action, prev),
        (BomberAction::Up, Some(BomberAction::Down))
            | (BomberAction::Down, Some(BomberAction::Up))
            | (BomberAction::Left, Some(BomberAction::Right))
            | (BomberAction::Right, Some(BomberAction::Left))
    )
}

/// Count destructible walls within manhattan range.
pub(crate) fn wall_density(grid: &ArenaGrid, pos: GridPos, range: i32) -> i32 {
    // Out-of-bounds reads return `FixedWall` from `ArenaGrid::get`, which never
    // counts, so clamping the window to the grid is equivalent to the previous
    // `grid.get` calls — but it lifts the row lookup out of the inner loop and
    // drops one bounds check plus one pointer hop per cell. `score_action` runs
    // this twice per move action, i.e. ~8×48 cells per tick per player.
    let y0 = (pos.y - range).max(0);
    let y1 = (pos.y + range).min(grid.height as i32 - 1);
    let x0 = (pos.x - range).max(0);
    let x1 = (pos.x + range).min(grid.width as i32 - 1);
    if y0 > y1 || x0 > x1 {
        return 0;
    }

    let mut count = 0;
    for y in y0..=y1 {
        let row = &grid.cells[y as usize][x0 as usize..=x1 as usize];
        let skip_center = y == pos.y;
        for (i, cell) in row.iter().enumerate() {
            if skip_center && x0 + i as i32 == pos.x {
                continue;
            }
            if let Cell::DestructibleWall | Cell::PowerUpHidden(_) = cell {
                count += 1
            }
        }
    }
    count
}

/// True if any cell adjacent to pos is a destructible wall.
pub(crate) fn has_adjacent_wall(grid: &ArenaGrid, pos: GridPos) -> bool {
    [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
        .iter()
        .any(|&(dx, dy)| {
            matches!(
                grid.get(pos.x + dx, pos.y + dy),
                Cell::DestructibleWall | Cell::PowerUpHidden(_)
            )
        })
}

/// BFS distance from pos to nearest cell outside all blast zones.
/// Returns `None` if no safe cell is reachable. Accounts for walls blocking blast.
pub(crate) fn escape_distance(
    pos: GridPos,
    grid: &ArenaGrid,
    bombs: &[KnownBomb],
    blocked: &[KnownBomb],
) -> Option<i32> {
    if !in_blast_zone(pos, grid, bombs) {
        return Some(0);
    }

    // Fixed-size visited bitmap for the 13×13 arena — avoids HashSet allocation
    // and hashing overhead on every BFS call (this runs per action per tick).
    let mut visited = [false; ARENA_W * ARENA_H];
    // Fixed-capacity FIFO replaces the heap `VecDeque` (see `has_escape_route`):
    // cells are marked before being pushed, so the queue is bounded by the grid
    // size plus the (possibly out-of-bounds, hence unmarkable) seed cell.
    let mut queue = [((0i32, 0i32), 0i32); ARENA_W * ARENA_H + 1];
    let mut head = 0usize;
    let mut tail = 0usize;

    let mark = |visited: &mut [bool; ARENA_W * ARENA_H], x: i32, y: i32| {
        if x >= 0 && (x as usize) < ARENA_W && y >= 0 && (y as usize) < ARENA_H {
            visited[(y as usize) * ARENA_W + (x as usize)] = true;
        }
    };
    let is_visited = |visited: &[bool; ARENA_W * ARENA_H], x: i32, y: i32| {
        if x >= 0 && (x as usize) < ARENA_W && y >= 0 && (y as usize) < ARENA_H {
            visited[(y as usize) * ARENA_W + (x as usize)]
        } else {
            true // Out-of-bounds treated as visited (blocked)
        }
    };

    queue[tail] = ((pos.x, pos.y), 0);
    tail += 1;
    mark(&mut visited, pos.x, pos.y);

    while head < tail {
        let ((cx, cy), dist) = queue[head];
        head += 1;
        for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
            if is_visited(&visited, nx, ny) {
                continue;
            }
            // Linear scan over blocked bomb positions (N is tiny, typically < 8,
            // so this beats hashing). Each bomb is (pos, range, fuse).
            let is_blocked = blocked.iter().any(|&(bp, _, _)| bp.0 == nx && bp.1 == ny);
            // Mark first, then gate — matches original HashSet::insert semantics.
            mark(&mut visited, nx, ny);
            if !grid.is_walkable(nx, ny) || is_blocked {
                continue;
            }
            let next_dist = dist + 1;
            if !in_blast_zone(GridPos { x: nx, y: ny }, grid, bombs) {
                return Some(next_dist);
            }
            queue[tail] = ((nx, ny), next_dist);
            tail += 1;
        }
    }

    None
}

/// Policy-based action scoring with clear priorities.
///
/// Policies (highest priority first):
///   Unsafe  → -∞     (wall, blast zone with no escape)
///   Flee    → +5..10 (escaping blast zone via shortest path)
///   Bomb    → +5.0   (near destructible wall + escape route)
///   Collect → +2..3  (moving toward / standing on revealed power-ups)
///   Hunt    → +0..2  (moving toward destructible walls)
///   Persist → -1.0   (penalize reversing direction)
///   Explore → +0.2   (slight center bias)
pub(crate) fn score_action(
    action: &BomberAction,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[KnownBomb],
    powerups: &[(i32, i32)],
    last_dir: Option<BomberAction>,
) -> f32 {
    use BomberAction::{Down, Left, Right, Up};

    // O(bombs) linear helper — replaces per-call HashSet<(i32,i32)> allocation.
    // Bombs list is tiny (typically < 8), so linear scan beats hashing.
    let is_blocked = |x: i32, y: i32| bombs.iter().any(|(p, _, _)| p.0 == x && p.1 == y);

    match action {
        Up | Down | Left | Right => {
            let target = move_target(action, pos);

            // Hard constraint: unwalkable or blocked by bomb entity
            if !grid.is_walkable(target.x, target.y) || is_blocked(target.x, target.y) {
                return f32::NEG_INFINITY;
            }

            // In blast zone — use escape distance for directional guidance
            if in_blast_zone(target, grid, bombs) {
                let current_dist = escape_distance(pos, grid, bombs, bombs).unwrap_or(i32::MAX);
                let target_dist = escape_distance(target, grid, bombs, bombs).unwrap_or(i32::MAX);
                return if target_dist < current_dist {
                    10.0 - target_dist as f32 * 0.5 // Moving toward safety
                } else if target_dist > current_dist {
                    -10.0 // Moving away from safety
                } else {
                    -5.0 // Same distance — slightly bad
                };
            }

            let mut score = 0.0;

            // Flee: escaping blast zone is top priority
            if in_blast_zone(pos, grid, bombs) {
                score += 10.0;
            }

            // Collect: move toward nearby revealed power-ups (high priority)
            if !powerups.is_empty() {
                // Single pass over `powerups` for both minima (min is
                // order-independent, so the results are identical).
                let mut current_min = i32::MAX;
                let mut target_min = i32::MAX;
                for &(px, py) in powerups {
                    let c = (pos.x - px).abs() + (pos.y - py).abs();
                    if c < current_min {
                        current_min = c;
                    }
                    let t = (target.x - px).abs() + (target.y - py).abs();
                    if t < target_min {
                        target_min = t;
                    }
                }
                if target_min == 0 {
                    score += 3.0; // Standing on power-up — instant collect
                } else if target_min < current_min {
                    score += 2.0; // Moving toward nearest power-up
                }
            }

            // Hunt: move toward areas with more destructible walls
            let current_walls = wall_density(grid, pos, 3);
            let target_walls = wall_density(grid, target, 3);
            score += (target_walls - current_walls) as f32 * 0.3;

            // Bonus: target cell is adjacent to destructible wall (bomb position)
            if has_adjacent_wall(grid, target) {
                score += 1.0;
            }

            // Persist: penalize reversing
            if is_reverse(*action, last_dir) {
                score -= 1.0;
            }

            // Explore: slight center bias
            let center = 6i32;
            let dist_before = (pos.x - center).abs() + (pos.y - center).abs();
            let dist_after = (target.x - center).abs() + (target.y - center).abs();
            if dist_after < dist_before {
                score += 0.2;
            }

            score
        }
        BomberAction::Bomb => {
            if !should_place_bomb(grid, pos, bombs) {
                return f32::NEG_INFINITY;
            }
            // Prefer bombs near destructible walls; still allow strategic open-area bombs
            if has_adjacent_wall(grid, pos) {
                5.0
            } else {
                2.0 // Lower priority but not blocked — prevents late-game stall
            }
        }
        BomberAction::Wait => {
            if in_blast_zone(pos, grid, bombs) {
                -10.0
            } else {
                -1.0
            }
        }
        BomberAction::Detonate => {
            // Detonate: only meaningful when remote bombs exist (future: power-up grant).
            // Score based on safety — detonating while in own blast zone is fatal.
            if bombs.is_empty() {
                -2.0 // No bombs to detonate — wasted action
            } else if in_blast_zone(pos, grid, bombs) {
                -10.0 // Unsafe: player would be caught in detonation
            } else {
                // Strategic option: slight positive when safe and bombs are active.
                // Becomes higher value when remote bombs are available (future work).
                1.0
            }
        }
    }
}

// ── LoRA Inference Helpers ─────────────────────────────────────

/// Per-element sigmoid scoring (independent scores in [0,1]).
///
/// Replaces softmax per project rule: "Use sigmoid not softmax".
/// Unlike softmax (which produces a probability distribution summing to 1),
/// sigmoid gives independent scores — each action is scored on its own merit.
#[cfg(feature = "bomber-wasm")]
#[allow(dead_code)]
pub(crate) fn sigmoid_scores(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|&s| 1.0 / (1.0 + (-s).exp())).collect()
}

/// Count walkable adjacent cells (for board feature encoding).
#[cfg(feature = "bomber-wasm")]
pub(crate) fn count_walkable(grid: &ArenaGrid, pos: GridPos) -> usize {
    [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
        .iter()
        .filter(|&&(dx, dy)| grid.is_walkable(pos.x + dx, pos.y + dy))
        .count()
}

/// Use loaded LoRA adapter to score all 6 actions.
///
/// Strategy: compute heuristic scores, then use LoRA as a learned re-weighting.
/// The LoRA was trained on game traces and encodes patterns like
/// "bomb near walls is good", "don't walk into blast".
///
/// Returns `None` if LoRA dimensions don't align (falls back to heuristic).
#[cfg(feature = "bomber-wasm")]
pub(crate) fn lora_score_actions(
    lora: &LoraAdapter,
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[KnownBomb],
    powerups: &[(i32, i32)],
    last_dir: Option<BomberAction>,
    lora_buf: &mut [f32],
) -> Option<[f32; ACTION_COUNT]> {
    // Compute heuristic base scores for all actions
    let heuristic: [f32; ACTION_COUNT] =
        ALL_ACTIONS.map(|action| score_action(&action, grid, pos, bombs, powerups, last_dir));

    // LoRA input: heuristic scores padded to in_dim with board features
    let in_dim = lora.in_dim;
    if in_dim < ACTION_COUNT {
        return None;
    }

    let mut input = vec![0.0f32; in_dim];
    for (i, &h) in heuristic.iter().enumerate() {
        input[i] = if h == f32::NEG_INFINITY { -10.0 } else { h };
    }
    // Pad remaining dimensions with board statistics
    if in_dim > ACTION_COUNT {
        input[ACTION_COUNT] = count_walkable(grid, pos) as f32 / 4.0;
    }
    if in_dim > ACTION_COUNT + 1 {
        input[ACTION_COUNT + 1] = if in_blast_zone(pos, grid, bombs) {
            1.0
        } else {
            0.0
        };
    }
    if in_dim > ACTION_COUNT + 2 {
        input[ACTION_COUNT + 2] = bombs.len() as f32 / 8.0;
    }
    if in_dim > ACTION_COUNT + 3 {
        input[ACTION_COUNT + 3] = powerups.len() as f32 / 4.0;
    }

    // Apply LoRA: output += scale * B @ (A @ input)
    let mut output = vec![0.0f32; lora.out_dim];
    lora_apply(&mut output, lora, &input, lora_buf);

    // Combine: LoRA re-weights heuristic scores
    let out_dim = lora.out_dim.min(ACTION_COUNT);
    let mut scores = heuristic;
    for i in 0..out_dim {
        if scores[i] != f32::NEG_INFINITY {
            // Blend: 70% heuristic + 30% LoRA correction (scaled)
            scores[i] = scores[i] * 0.7 + output[i] * 3.0;
        }
    }

    Some(scores)
}

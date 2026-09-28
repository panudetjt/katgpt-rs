#!/usr/bin/env python3
"""The `.len()`-derived kernel dimension vs oversized-buffer join (riir-train 515 T1/T3).

A CubeCL kernel that derives a structural dimension from a bound buffer's
`.len()` — `n_positions = kv.len() / 2 / kv_stride` — computes that dimension
from the buffer's **declared size**, NOT from the length metadata passed to
`BufferArg::from_raw_parts` (measured: riir-ai `3e00c93e0`, riir-train `.issues/511`).
Hand such a kernel a buffer whose declared size exceeds the live range and it
silently derives the WRONG shape — reads never-written memory, writes a
plausible-looking (measured: identically-zero) result. No panic, no NaN.

The defect is the JOIN of two facts that live in different files:

- HALF A: the kernel derives a dimension from `.len()` (in-kernel), and
- HALF B: the bound handle's DECLARED SIZE can exceed the live range
  (a persistent struct-field handle, a capacity-sized `client.empty()`, or a
  host slice that is itself a reused scratch buffer).

A report over either half alone is noise — almost every `.len()` kernel is
correct because its call sites bind exactly-sized buffers. This script reports
the join, classifying every bind site of every `.len()`-deriving kernel:

- `PERSISTENT`   — the handle expression is a struct field (`self.x`,
  `cache.x`, ...): the declared size is whatever the field was created as.
  The compact_temp class. Needs eyes unless the field is provably exact.
- `CAPACITY`     — the handle is `client.empty(EXPR)` (or `create_buffer`)
  where EXPR names a capacity/block-size constant or field rather than a
  logical dim. Declared size ≠ live range by construction. JOINED finding.
- `PARAM-LOCAL`  — the handle is created at the bind site from a local
  slice or a logical-dim `empty()` — declared size follows the logical dim
  at this site (the caller of an outer wrapper can still lie; depth-1 only).
- `UNRESOLVED`   — the handle is a wrapper parameter: provenance lives one
  level up. Issue 766 added HALF C: for wrappers that are path-form
  associated fns (`impl Struct { fn name(params) }`, no `self`), every
  workspace caller `Struct::name::<T>(args)` is collected and the
  (handle, length) PARAM PAIR is resolved at each call site:

  - `EXACT-UPSTREAM`      — every caller binds an exactly-sized creation
    with the matching length (`create_from_slice(&v)` + `v.len()`,
    `empty(k)` + `k`). Clean by construction.
  - `TRIMMED-UPSTREAM`    — every caller binds a `.slice(o, e)` view: a
    view's declared size IS its live range.
  - `PERSISTENT-UPSTREAM` — some caller binds a struct-field handle
    (`self.x`, `bufs.x`): declared size is the field's — the EYES LIST.
  - `CAPACITY-UPSTREAM`   — some caller's bind length is sourced from the
    handle's own size method, or the creation names a capacity constant:
    the compact_temp join, one level up.
  - otherwise stays `UNRESOLVED` (pass-through params, method-form /
    bare-name callers, macros, unprovable pairs).

  A clean verdict requires ALL workspace callers of the key to be clean —
  one unknown caller keeps the row UNRESOLVED. A false clean hides a
  defect; an honest UNRESOLVED costs nothing.

## What this is NOT

A **report, not a gate** — always exit 0 (the `cfg_gated_target_audit.py`
discipline). The static join cannot prove a PERSISTENT handle wrong (the field
may be exactly sized — the norm-gamma handles are), and it cannot walk wrapper
chains through re-exports, which the workspace already documents as defeating
grep. The runtime half of this defense is `assert_binding_derives_units`
(riir-ai `cubecl_runtime.rs`, riir-train 515 T4) — this report names the
launchers that should grow it next.

## Population is derived, walk is floored

Repos come from the workspace walk (root `BOUNDARY.md` AND `.git`), never a
typed list. The `min_rs_files` floor refuses the confident-zero: a regex
regression that recognises no kernels must RED, not print 0 findings.

## HALF C limits (recorded, not hidden)

Bare-name callers (`launch(c, ...)` without the struct) and method-form
wrappers (`impl S { fn f(&self, ...) }`) are not walked — matching a bare
`launch` by name is unsound, and receiver types are not statically
resolvable here. Free-fn pass-through wrappers are not recursed through.
Aliased re-exports and dynamic dispatch defeat the key match. Rows whose
only callers take those forms stay UNRESOLVED — the residual is the honest
floor, never a clean verdict.
"";
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracked_walk import tracked_files  # noqa: E402
import repo_alias  # noqa: E402 — the machine-local name codec (see its docstring)

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

# ── HALF A: cube kernels that derive dims from `.len()` ────────────────────

CUBE_FN_RE = re.compile(
    r"#\[cube\((?P<attrs>[^)]*)\)\]\s*(?:pub\s+)?fn\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)",
)
LEN_USE_RE = re.compile(r"\b(?P<buf>\w+)\.len\(\)")

# ── Bind sites ──────────────────────────────────────────────────────────────

LAUNCH_RE = re.compile(r"\b(?P<kernel>\w+)::launch_unchecked\b")
FROM_RAW_RE = re.compile(
    r"BufferArg::from_raw_parts\(\s*(?P<handle>[^,]+?),\s*(?P<length>[^,)]+?(?:\([^()]*\)[^,)]*?)*?)\s*\)"
)
# Creation shapes at/near a bind: `client.empty(N)`, `create_from_slice(&v)`,
# `create_buffer(N)`.
EMPTY_CALL_RE = re.compile(r"\.empty\(\s*(?P<n>[^)]+?)\s*\)")
SLICE_CALL_RE = re.compile(r"create_from_slice\(\s*(?:f32::as_bytes\(\s*)?(?:&\s*)?(?P<v>[\w.\[\]]+)")

CAPACITY_WORDS = ("block_size", "capacity", "max_seq", "n_ctx", "max_positions")


def length_from_handle_size_method(handle: str, length_expr: str) -> str | None:
    """The receiver of a size-method call that IS the whole `length_expr`,
    when that receiver is `handle` (or `handle` is a member path of it) —
    allocation-as-length, the compact_temp shape. Returns the receiver, or
    None.

    ONE rule shared by the depth-1 (classify) and upstream (classify_pair)
    classifiers. Two copies drifted once: the depth-1 copy flagged any
    `.len(` in a bind-length position, so `rows_host.len()` — a HOST slice
    param whose len IS the live row count — read as the bound handle's
    allocation (riir-infer 020, encoder_lane_cubecl gather_rows)."""
    h = handle.strip()
    sm = SIZE_METHOD_RE.match(length_expr)
    if sm and (h == sm.group("recv") or h.startswith(sm.group("recv") + ".")):
        return sm.group("recv")
    return None


@dataclass
class Kernel:
    repo: str
    file: str
    name: str
    len_params: set[str]


@dataclass
class BindSite:
    kernel: str
    repo: str
    file: str
    line: int
    handle_expr: str
    length_expr: str
    verdict: str = "UNRESOLVED"
    reason: str = ""


@dataclass
class Report:
    kernels: list[Kernel] = field(default_factory=list)
    binds: list[BindSite] = field(default_factory=list)
    guard_only: dict[str, bool] = field(default_factory=dict)
    files_scanned: int = 0


def derive_repos(workspace: Path) -> list[Path]:
    """A root BOUNDARY.md AND a `.git` DIRECTORY — never a typed list.

    ⛔ The `.git` test must be `is_dir()`, not `exists()`. A `git worktree`'s
    `.git` is a FILE (`gitdir: …`), so `exists()` admits a throwaway worktree of
    a repo ALREADY in the walk and double-counts it. This function had
    `exists()` from the day it was written and nothing noticed, because it was
    the one contract-repo predicate `population_sync_gate.py` did not know about
    — Issue 788, which is the whole argument for that gate asserting its own
    registry completeness. The derived set on the boxes that have run it is
    unchanged (no worktree-shaped directory in this workspace); the defect was
    latent, not active.

    Names pass through the machine-local alias codec (`repo_alias.py`) so the
    returned paths carry the CONTRACT spelling every tracked pin is keyed on.
    """
    return [workspace / n for n in repo_alias.apply(
        d.name for d in workspace.iterdir()
        if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()
    )]


def rs_files(repo: Path):
    """The TRACKED `*.rs` population — `scripts/tracked_walk.py` (Issue 777).

    This docstring used to say "tracked-SHAPE walk", and the shape was the
    problem: an `os.walk` pruning `("target", ".git", "node_modules")` is a
    directory-NAME list, and a name list cannot express "not ours". Measured
    in the sibling percentile audit, which carried the same three names:
    mmorpg-remaster's gitignored `mmorpg/` nested repo (+1404 `.rs`) and
    riir-train's cargo OUT_DIR sources under `.runs/target-release/` (+48 —
    the set names `target`, and `target-release` is not `target`) were both
    inside the population. Nothing here matched them only because this
    audit's vocabulary is GPU-binding-specific; that is luck, not scope.

    `git ls-files` also answers faster than the walk it replaces, which was
    the original reason for the up-front prune (measured: full `rglob`
    timeout on the 4090 box).
    """
    return tracked_files(repo, "*.rs")[0]


def guard_only_uses(body: str, buf: str) -> bool:
    """True iff every `.len()` use of `buf` in `body` feeds only a comparison
    (directly, or through a `let n = buf.len()` alias whose every use is a
    comparison). Guard-only derivation is capacity-tolerant: the derived
    value only bounds thread indices, and the dispatch count already caps
    processing at the live range (measured class: copy_f32/fill_zeros_f32 —
    benign with t_max-capacity handles; the hazard is structural
    derivation, `n_positions = kv.len()/2/stride` feeding index math).
    """
    buf_len = re.escape(buf) + r"\.len\(\)"
    ops = r"(?:<=|>=|==|!=|<|>)"
    cmp_direct = re.compile(rf"{ops}\s*{buf_len}|{buf_len}\s*{ops}")
    cmp_alias = re.compile(rf"{ops}\s*=\s*\b(?P<n>\w+)\b|\b(?P<m>\w+)\b\s*{ops}")
    for m in re.finditer(buf_len, body):
        ls = body[: m.start()].rfind("\n")
        line = body[ls + 1 : body.find("\n", m.end())]
        if cmp_direct.search(line):
            continue
        am = re.match(rf"\s*let\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*{buf_len}\s*;", line)
        if not am:
            return False
        name = am.group("name")
        rest = body[m.end() :]
        for u in re.finditer(rf"\b{re.escape(name)}\b", rest):
            uls = rest[: u.start()].rfind("\n")
            ul = rest[uls + 1 : rest.find("\n", u.end())]
            # the alias's uses must all be comparison operands or its own let
            if not re.search(rf"{ops}\s*\b{re.escape(name)}\b|\b{re.escape(name)}\b\s*{ops}", ul):
                if not re.match(rf"\s*let\s+(?:mut\s+)?\w+\s*=\s*\b{re.escape(name)}\b", ul):
                    return False
    return True


def half_a(repo: Path, rep: Report) -> None:
    """Cube kernels whose bodies derive from `.len()`, with the buffers.

    Body window = the fn's own brace-matched body. The original fixed
    6000-char window bled into whatever followed the kernel — measured: the
    tree-verify kernels were flagged off `parent.len()` in the HOST-side
    `TreeVerifyPlan::from_parents_topo` sitting after them (Issue 766 T3:
    those kernels derive every dim from `params` — the correct pattern).
    """
    # Issue 842: the walk reads the on-disk directory; labels stay repo.name.
    disk = repo_alias.real(repo)
    for f in rs_files(disk):
        rep.files_scanned += 1
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "#[cube(" not in src:
            continue
        for m in CUBE_FN_RE.finditer(src):
            body_start = src.find("{", m.end())
            if body_start < 0:
                continue
            body_end = match_paren(src, body_start)
            if body_end < 0:
                body_end = min(m.end() + 6000, len(src))  # legacy fallback
            body = src[body_start:body_end]
            bufs = {g.group("buf") for g in LEN_USE_RE.finditer(body)}
            if not bufs:
                continue
            rep.kernels.append(
                Kernel(repo.name, str(f.relative_to(disk)),
                       m.group("name"), bufs)
            )
            rep.guard_only.setdefault(m.group("name"), all(guard_only_uses(body, b) for b in bufs))


def classify(handle_expr: str, length_expr: str, line_src: str) -> tuple[str, str]:
    """Depth-1 provenance classification of one binding."""
    h = handle_expr.strip()
    # The attention fix pattern: a trimmed view — clean by construction.
    if "offset_end" in line_src or ".slice(" in line_src:
        return "TRIMMED", "offset_end/slice view — declared size is the live range"
    # Length taken from a handle's own size METHOD — allocation-as-length,
    # the direct shape of the compact_temp bug. (Identifier substrings like
    # `vocab_size` / `as usize` are NOT this, and neither is a size method
    # on a DIFFERENT name — a host slice param's `.len()` is the live
    # count, not an allocation: the rule is the receiver, shared with
    # classify_pair.)
    if (recv := length_from_handle_size_method(h, length_expr)) is not None:
        return "CAPACITY", f"bind length from a handle size method ({recv}): {length_expr.strip()}"
    # A capacity constant naming the bind length directly.
    if any(w in length_expr for w in ("block_size", "max_seq", "n_ctx", "max_positions")):
        return "CAPACITY", f"bind length names a capacity constant: {length_expr.strip()}"
    # Persistent struct-field handle.
    if re.match(r"^(self|cache|state|weights|scratch)\.", h):
        return "PERSISTENT", "struct-field handle — declared size is the field's"
    # Created at the bind site from a slice of a local: follows the local.
    sm = SLICE_CALL_RE.search(line_src)
    if sm and sm.group("v") in line_src:
        v = sm.group("v")
        if v.startswith(("self.", "cache.", "scratch.")):
            return "PERSISTENT", f"create_from_slice of persistent host vec {v}"
        return "PARAM-LOCAL", f"create_from_slice of local {v}"
    em = EMPTY_CALL_RE.search(line_src)
    if em and "empty" in line_src:
        n = em.group("n")
        if any(w in n for w in CAPACITY_WORDS):
            return "CAPACITY", f"empty({n}) sized by a capacity constant"
        return "PARAM-LOCAL", f"empty({n}) sized by a local dim"
    return "UNRESOLVED", "wrapper-param handle — provenance is one level up"


def half_b(repo: Path, kernels: list[Kernel], rep: Report) -> None:
    """Every `K::launch_unchecked` call in the repo, with its bindings."""
    names = {k.name for k in kernels}
    if not names:
        return
    # Issue 842: the walk reads the on-disk directory; labels stay repo.name.
    disk = repo_alias.real(repo)
    for f in rs_files(disk):
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Runtime-guarded windows: lines within 60 lines of an
        # `assert_binding_derives_units` call site (515 T4/T2). Loose on
        # purpose — a report credit, not a proof; the refusal pins are the
        # proof.
        guarded_lines: set[int] = set()
        for gm in re.finditer(r"assert_binding_derives_units\s*\(", src):
            line_no = src[: gm.start()].count("\n") + 1
            guarded_lines.update(range(line_no - 60, line_no + 60))
        for m in LAUNCH_RE.finditer(src):
            if m.group("kernel") not in names:
                continue
            # Call span: to the matching close paren of launch_unchecked(.
            open_at = src.find("(", m.end() - 1)
            span = src[open_at : open_at + 4000]
            # Only the first nesting level: cut at the matching paren.
            depth, end = 0, 0
            for i, ch in enumerate(span):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            call = span[:end]
            line = src[: m.start()].count("\n") + 1
            for b in FROM_RAW_RE.finditer(call):
                v, why = classify(b.group("handle"), b.group("length"), b.group(0))
                if v in ("UNRESOLVED", "PARAM-LOCAL", "PERSISTENT", "CAPACITY") and line in guarded_lines:
                    # The launcher carries the runtime refusal guard — the
                    # oversized-binding class is refused at launch (pinned by
                    # the refusal tests), whatever the static provenance says.
                    v = "GUARDED"
                    why = "launcher carries assert_binding_derives_units (515 T4/T2)"
                rep.binds.append(
                    BindSite(
                        m.group("kernel"),
                        repo.name,
                        str(f.relative_to(disk)),
                        line,
                        b.group("handle").strip(),
                        b.group("length").strip(),
                        v,
                        why,
                    )
                )
            if not FROM_RAW_RE.search(call):
                rep.binds.append(
                    BindSite(
                        m.group("kernel"),
                        repo.name,
                        str(f.relative_to(disk)),
                        line,
                        "<no BufferArg in span>",
                        "",
                        "UNRESOLVED",
                        "call span parsed but no from_raw_parts found (macro/moved args?)",
                    )
                )


# ── HALF C (Issue 766): caller tracing for wrapper-param rows ─────────────

PATH_CALL_RE = re.compile(r"\b(?P<typ>[A-Z]\w*)\s*::\s*(?P<fn>[a-z_]\w*)")
FN_HEAD_RE = re.compile(r"\bfn\s+(?P<name>[a-z_]\w*)")
IMPL_HEAD_RE = re.compile(r"\bimpl\b")
UPPER_IDENT_RE = re.compile(r"[A-Z]\w*")
LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>[a-z_]\w*)\s*(?::\s*[^=;]+)?=\s*(?P<rhs>[^;]+);"
)
SIZE_METHOD_RE = re.compile(r"^(?P<recv>[\w.]+)\.(?:len|size|size_in_bytes)\(\)$")
BARE_IDENT_RE = re.compile(r"^[a-z_]\w*$")
FIELD_HANDLE_RE = re.compile(r"^(?:(?:self|cache|state|weights|scratch)\.|[a-z_]\w*\.)")


@dataclass
class CallSite:
    repo: str
    file: str
    line: int
    args: list[str]


@dataclass
class FnSpan:
    name: str
    params: list[str]
    is_method: bool
    header_start: int
    body_start: int
    body_end: int
    impl_struct: str | None


def match_paren(src: str, open_idx: int) -> int:
    """Index of the matching close for the bracket at `open_idx`, depth-aware.

    String/comment blind (same class of limitation as HALF B's span cut):
    garbage args degrade to an unknown caller, never a false clean.
    """
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    open_ch = src[open_idx]
    close_ch = pairs[open_ch]
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == open_ch:
            depth += 1
        elif src[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_commas(s: str) -> list[str]:
    """Split on commas outside (), [], {}, <> (turbofish args carry commas)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def skip_turbofish(src: str, i: int) -> int:
    """If `src[i:]` starts with `::<`, return the index past the matching `>`."""
    if not src.startswith("::<", i):
        return i
    depth, j = 0, i + 2
    while j < len(src):
        if src[j] == "<":
            depth += 1
        elif src[j] == ">":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return i


def normalize_expr(e: str) -> str:
    """Strip the noise both sides of a pair comparison carry: parens, `&`,
    `.clone()`, `as usize`-family casts, repeated whitespace."""
    e = e.strip()
    changed = True
    while changed:
        changed = False
        if e.startswith("(") and e.endswith(")") and match_paren(e, 0) == len(e) - 1:
            e = e[1:-1].strip()
            changed = True
        if e.endswith(".clone()"):
            e = e[: -len(".clone()")].strip()
            changed = True
        if e.startswith("&"):
            e = e[1:].strip()
            changed = True
        m = re.search(r"\s+as\s+(?:usize|u8|u16|u32|u64|isize)$", e)
        if m:
            e = e[: m.start()].strip()
            changed = True
    return re.sub(r"\s+", " ", e)


def strip_generic_spans(s: str) -> str:
    """Remove balanced `<...>` runs (impl-header generics/where noise)."""
    out: list[str] = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">" and depth > 0:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def fn_spans(src: str) -> list[FnSpan]:
    """Flat fn-header index: name, param names, method-ness, body span,
    enclosing impl struct (nearest `impl` header above, last UpperCamel
    ident after stripping generics + where-tail). Nested fns attribute to
    the innermost header found before them — honest degradation is a wrong
    key → no callers → UNRESOLVED, never a false clean.
    """
    spans: list[FnSpan] = []
    for m in FN_HEAD_RE.finditer(src):
        j = m.end()
        while j < len(src) and src[j].isspace():
            j += 1
        if j < len(src) and src[j] == "<":  # fn generics <...>
            e = match_paren(src, j)
            if e < 0:
                continue
            j = e + 1
            while j < len(src) and src[j].isspace():
                j += 1
        if j >= len(src) or src[j] != "(":
            continue
        pc = match_paren(src, j)
        if pc < 0:
            continue
        params: list[str] = []
        is_method = False
        for p in split_top_commas(src[j + 1 : pc]):
            p = p.strip()
            pm = re.match(r"^(?:&(?:\s*'\w+\s*)?\s*)?(?:mut\s+)?(self|\w+)", p)
            if not pm:
                params.append("")
                continue
            if pm.group(1) == "self":
                is_method = True
                params.append("self")
            else:
                params.append(pm.group(1))
        while params and params[-1] == "":  # trailing comma
            params.pop()
        if "" in params:
            # An unparseable param would shift every later index — refuse
            # the whole span rather than bind the wrong arg.
            is_method = True
            params = []
        body_start = src.find("{", pc)
        if body_start < 0:
            continue
        body_end = match_paren(src, body_start)
        if body_end < 0:
            body_end = len(src) - 1
        impl_struct = None
        im = None
        for im_c in IMPL_HEAD_RE.finditer(src, 0, m.start()):
            im = im_c
        if im is not None:
            brace = src.find("{", im.end())
            head = src[im.end() : brace if 0 < brace < m.start() else m.start()]
            head = strip_generic_spans(head.split(" where ")[0])
            idents = UPPER_IDENT_RE.findall(head)
            if idents:
                impl_struct = idents[-1]
        spans.append(
            FnSpan(m.group("name"), params, is_method, m.start(), body_start, body_end + 1, impl_struct)
        )
    return spans


def enclosing_fn(spans: list[FnSpan], offset: int) -> FnSpan | None:
    best: FnSpan | None = None
    for s in spans:
        if s.header_start <= offset < s.body_end:
            if best is None or s.header_start > best.header_start:
                best = s
    return best


def line_start_offset(src: str, line: int) -> int:
    """1-based line → offset of its first char."""
    if line <= 1:
        return 0
    pos = -1
    for _ in range(line - 1):
        pos = src.find("\n", pos + 1)
        if pos < 0:
            return len(src)
    return pos + 1


def resolve_ident_lets(ident: str, src: str, span: FnSpan) -> str | None:
    """Unique `let ident = RHS;` in the fn body → normalized RHS (chases
    bare-ident chains, capped). None = 0 or 2+ bindings or shadowing risk —
    all honest unknowns."""
    body = src[span.body_start : span.body_end]
    hits = [m for m in LET_RE.finditer(body) if m.group("name") == ident]
    if len(hits) != 1:
        return None
    rhs = normalize_expr(hits[0].group("rhs"))
    seen = {ident}
    while BARE_IDENT_RE.match(rhs) and rhs not in seen:
        seen.add(rhs)
        nxt = [m for m in LET_RE.finditer(body) if m.group("name") == rhs]
        if len(nxt) != 1:
            break
        rhs = normalize_expr(nxt[0].group("rhs"))
    return rhs


def classify_pair(h: str, length: str | None) -> tuple[str, str]:
    """Upstream (handle, length) classification. `length` is the resolved
    bind-length expression at the caller, or None when unresolvable.
    Never returns a clean verdict on an unprovable pair."""
    if length is not None:
        if length_from_handle_size_method(h, length) is not None:
            return "CAPACITY-UPSTREAM", f"bind length {length} sourced from the handle's own size method"
    if ".slice(" in h:
        return "TRIMMED-UPSTREAM", "slice view — declared size is the live range"
    em = EMPTY_CALL_RE.search(h)
    if em:
        n = normalize_expr(em.group("n"))
        if any(w in n for w in CAPACITY_WORDS):
            return "CAPACITY-UPSTREAM", f"empty({n}) sized by a capacity constant"
        if length is not None and length == n:
            return "EXACT-UPSTREAM", f"empty({n}) with matching bind length"
        return "UNRESOLVED", f"empty({n}) but bind length {length or '?'} differs — cannot prove equal"
    fm = SLICE_CALL_RE.search(h)
    if fm and "create_from_slice" in h:
        v = fm.group("v")
        if length is not None and length == f"{v}.len()":
            return "EXACT-UPSTREAM", f"create_from_slice(&{v}) with bind length {v}.len()"
        return "UNRESOLVED", f"create_from_slice but bind length {length or '?'} is not {v}.len()"
    if length is not None and any(w in length for w in CAPACITY_WORDS):
        return "CAPACITY-UPSTREAM", f"bind length names a capacity constant: {length}"
    if "(" not in h and FIELD_HANDLE_RE.match(h):
        # `(`-bearing exprs are call results (e.g. params_handle(...)) — a
        # call's returned size is unknown, never a field's.
        return "PERSISTENT-UPSTREAM", f"struct-field handle {h} — declared size is the field's"
    return "UNRESOLVED", f"unrecognised upstream shape: {h}"


def collect_calls(repo: Path, fn_names: set[str], calls: dict) -> None:
    """One walk: every path-form `Struct::fn::<T>(args)` whose fn name is
    wanted, with top-level arg extraction. Comment-prefixed lines skipped —
    a dead call in prose must never count as a caller (false-clean guard)."""
    # Issue 842: the walk reads the on-disk directory; labels stay repo.name.
    disk = repo_alias.real(repo)
    for f in rs_files(disk):
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in PATH_CALL_RE.finditer(src):
            if m.group("fn") not in fn_names:
                continue
            j = m.end()
            while j < len(src) and src[j].isspace():
                j += 1
            j = skip_turbofish(src, j)
            while j < len(src) and src[j].isspace():
                j += 1
            if j >= len(src) or src[j] != "(":
                continue
            ls = src.rfind("\n", 0, m.start()) + 1
            le = src.find("\n", ls)
            if src[ls : le if le > 0 else len(src)].lstrip().startswith(("//", "/*", "*")):
                continue
            close = match_paren(src, j)
            if close < 0:
                continue
            # Inline comments ride into the next arg's text after its comma
            # (measured: `self.z_buf.clone(),  // [0..n]` poisoned the NEXT
            # arg and broke field detection) — strip them before splitting.
            args_src = re.sub(r"//[^\n]*", "", src[j + 1 : close])
            args = [normalize_expr(a) for a in split_top_commas(args_src)]
            line = src.count("\n", 0, m.start()) + 1
            calls.setdefault((m.group("typ"), m.group("fn")), []).append(
                CallSite(repo.name, str(f.relative_to(disk)), line, args)
            )


def trace_unresolved(binds: list[BindSite], repo_paths: dict[str, Path]) -> None:
    """HALF C driver: upgrade UNRESOLVED rows via caller provenance, in place."""
    NO_BIND = "<no BufferArg in span>"
    src_cache: dict = {}
    span_cache: dict = {}
    wanted: dict[int, tuple[str, str, int, int | None]] = {}
    fn_names: set[str] = set()
    for idx, b in enumerate(binds):
        if b.verdict != "UNRESOLVED" or b.handle_expr == NO_BIND:
            continue
        ckey = (b.repo, b.file)
        if ckey not in src_cache:
            try:
                # Issue 842: resolve the handle to its on-disk directory.
                src_cache[ckey] = (repo_alias.real(repo_paths[b.repo]) / b.file).read_text(encoding="utf-8", errors="replace")
            except OSError:
                src_cache[ckey] = ""
        if ckey not in span_cache:
            span_cache[ckey] = fn_spans(src_cache[ckey])
        src = src_cache[ckey]
        span = enclosing_fn(span_cache[ckey], line_start_offset(src, b.line))
        if span is None or span.impl_struct is None or span.is_method:
            b.reason = "wrapper not a path-form associated fn — callers not walkable"
            continue
        h = normalize_expr(b.handle_expr)
        if not BARE_IDENT_RE.match(h) or h not in span.params:
            # Wrapper-LOCAL handle (e.g. `let params_handle =
            # crate::params_cache::params_handle(client, ...)`): resolve its
            # unique let in the wrapper and classify in place — same depth-1
            # information as a bind-site creation, one fn up.
            if BARE_IDENT_RE.match(h):
                r = resolve_ident_lets(h, src, span)
                if r is not None:
                    lexpr = normalize_expr(b.length_expr)
                    if BARE_IDENT_RE.match(lexpr):
                        lr = resolve_ident_lets(lexpr, src, span)
                        if lr is not None:
                            lexpr = lr
                    v, why = classify_pair(r, lexpr if lexpr else None)
                    b.verdict, b.reason = v, f"wrapper-local creation; {why}"
            if b.verdict == "UNRESOLVED" and not b.reason:
                b.reason = f"handle {b.handle_expr} is not a bare param of {span.impl_struct}::{span.name}"
            continue
        length_idx = None
        lexpr = normalize_expr(b.length_expr)
        if BARE_IDENT_RE.match(lexpr) and lexpr in span.params:
            length_idx = span.params.index(lexpr)
        wanted[idx] = (span.impl_struct, span.name, span.params.index(h), length_idx)
        fn_names.add(span.name)

    calls: dict = {}
    for path in repo_paths.values():
        collect_calls(path, fn_names, calls)

    for idx, (typ, fn, h_idx, l_idx) in wanted.items():
        b = binds[idx]
        sites = calls.get((typ, fn), [])
        if not sites:
            b.reason = f"no path-form caller of {typ}::{fn} found (bare-name/method-form callers not walked)"
            continue
        tags: list[tuple[CallSite, str, str]] = []
        for site in sites:
            if h_idx >= len(site.args):
                tags.append((site, "UNRESOLVED", "arg position missing"))
                continue
            h_arg = site.args[h_idx]
            l_arg = site.args[l_idx] if l_idx is not None and l_idx < len(site.args) else None
            skey = (site.repo, site.file)
            if skey not in src_cache:
                try:
                    src_cache[skey] = (repo_paths[site.repo] / site.file).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    src_cache[skey] = ""
            if skey not in span_cache:
                span_cache[skey] = fn_spans(src_cache[skey])
            cspan = enclosing_fn(span_cache[skey], line_start_offset(src_cache[skey], site.line))
            resolved_h = h_arg
            if cspan is not None and BARE_IDENT_RE.match(h_arg):
                if h_arg in cspan.params:
                    tags.append((site, "UNRESOLVED", f"pass-through param {h_arg} — depth cap"))
                    continue
                r = resolve_ident_lets(h_arg, src_cache[skey], cspan)
                if r is not None:
                    resolved_h = r
            resolved_l = l_arg
            if cspan is not None and l_arg is not None and BARE_IDENT_RE.match(l_arg):
                r = resolve_ident_lets(l_arg, src_cache[skey], cspan)
                if r is not None:
                    resolved_l = r
            v, why = classify_pair(resolved_h, resolved_l)
            tags.append((site, v, why))

        def summarise() -> str:
            parts = [f"{s.repo}/{s.file}:{s.line} {v}" for s, v, _ in tags[:4]]
            if len(tags) > 4:
                parts.append(f"(+{len(tags) - 4} more)")
            return "; ".join(parts)

        verdicts = {v for _, v, _ in tags}
        if "CAPACITY-UPSTREAM" in verdicts:
            b.verdict, b.reason = "CAPACITY-UPSTREAM", summarise()
        elif "PERSISTENT-UPSTREAM" in verdicts:
            b.verdict, b.reason = "PERSISTENT-UPSTREAM", summarise()
        elif verdicts and verdicts <= {"EXACT-UPSTREAM", "TRIMMED-UPSTREAM"}:
            mixed = "mixed exact/trimmed; " if len(verdicts) > 1 else ""
            b.verdict = "TRIMMED-UPSTREAM" if "TRIMMED-UPSTREAM" in verdicts else "EXACT-UPSTREAM"
            b.reason = mixed + summarise()
        else:
            b.reason = f"caller set not provably clean ({len(tags)} caller(s)) — " + summarise()


def selftest() -> None:
    """Pin the parse shapes. Runs on EVERY invocation.

    Without this the audit degrades silently: a regex regression makes it
    recognise fewer kernels and still print a confident `0 findings` — the
    exact failure mode it exists to catch, committed by the tool that catches
    it (`cfg_gated_target_audit.py`'s precedent).
    """
    kernel_src = """#[cube(launch_unchecked)]
fn attention_decode_f32(query: &[f32], kv: &[f32]) {
    let kv_half = kv.len() as u32 / 2u32;
}"""
    m = CUBE_FN_RE.search(kernel_src)
    assert m and m.group("name") == "attention_decode_f32", "kernel parse broke"
    body = kernel_src[m.end() :]
    bufs = {g.group("buf") for g in LEN_USE_RE.finditer(body)}
    assert "kv" in bufs, "len-use parse broke"

    lm = LAUNCH_RE.search("unsafe { attention_decode_f32::launch_unchecked::<R>(c, a) }")
    assert lm and lm.group("kernel") == "attention_decode_f32", "launch parse broke"

    fm = FROM_RAW_RE.search("BufferArg::from_raw_parts(kv_handle, combined_kv_len)")
    assert fm and fm.group("handle") == "kv_handle", "bind parse broke"

    assert classify("self.compact_temp", "n", "")[0] == "PERSISTENT"
    assert classify("h", "h.len()", "")[0] == "CAPACITY"
    assert classify("h", "h.size()", "")[0] == "CAPACITY"
    assert classify("h", "h.size_in_bytes()", "")[0] == "CAPACITY"
    assert classify("h", "block_size * 4", "")[0] == "CAPACITY"
    # The size-method receiver must BE the bound handle: `rows_host` is a
    # HOST slice param whose `.len()` is the live row count, not the
    # handle's allocation (riir-infer 020, gather_rows rows_handle). The
    # prefix branch stays armed — `h.clone()` is still the handle.
    assert classify("rows_handle", "rows_host.len()", "")[0] == "UNRESOLVED"
    assert classify("h.clone()", "h.len()", "")[0] == "CAPACITY"
    # Identifier substrings are NOT size methods — pinned after the first run
    # flagged all 7 CAPACITY rows off `vocab_size` / `as usize` substrings.
    assert classify("input", "vocab_size", "")[0] == "UNRESOLVED"
    assert classify("input", "n as usize", "")[0] == "UNRESOLVED"
    assert classify("h", "dim * 4", "let h = client.empty(dim * 4); x")[0] == "PARAM-LOCAL"
    assert classify("q", "q_len", "")[0] == "UNRESOLVED"

    # ── HALF C pins (Issue 766) ─────────────────────────────────────────
    # Turbofish skip + arg split (nested parens/brackets/generic commas).
    cs = "SigmoidCubeCL::launch::<ActiveRuntime>(&client, a, out.clone(), n)"
    m = PATH_CALL_RE.search(cs)
    assert m and m.group("typ") == "SigmoidCubeCL" and m.group("fn") == "launch"
    j = m.end()
    while cs[j].isspace():
        j += 1
    j = skip_turbofish(cs, j)
    assert cs[j] == "(", "turbofish skip broke"
    close = match_paren(cs, j)
    args = [normalize_expr(a) for a in split_top_commas(cs[j + 1 : close])]
    assert args == ["client", "a", "out", "n"], f"arg split broke: {args}"

    nested = "Foo::bar::<HashMap<K, V>>(f(a(x, y), z), [1, 2], 3)"
    m = PATH_CALL_RE.search(nested)
    j = skip_turbofish(nested, m.end())
    assert nested[j] == "(", "nested turbofish skip broke"
    parts = split_top_commas(nested[j + 1 : match_paren(nested, j)])
    assert len(parts) == 3, f"nested comma split broke: {parts}"

    assert normalize_expr("(out_h.clone() as usize)") == "out_h"
    assert normalize_expr("& &v") == "v"

    # The false-clean canaries: a field handle is NEVER exact, however
    # name-plausible the length is; a size-method length is capacity even
    # against a bare handle.
    assert classify_pair("self.qkv", "conv_dim")[0] == "PERSISTENT-UPSTREAM"
    assert classify_pair("bufs.qkv_expanded", None)[0] == "PERSISTENT-UPSTREAM"
    assert classify_pair("h", "h.len()")[0] == "CAPACITY-UPSTREAM"
    assert classify_pair("h.clone()", "h.size()")[0] == "CAPACITY-UPSTREAM"
    # Same shared rule, upstream side: a size method on a DIFFERENT name is
    # not the handle's allocation (riir-infer 020).
    assert classify_pair("rows_handle", "rows_host.len()")[0] == "UNRESOLVED"
    assert classify_pair("client.empty(max_seq)", "max_seq")[0] == "CAPACITY-UPSTREAM"
    assert classify_pair("client.empty(k)", "k")[0] == "EXACT-UPSTREAM"
    assert classify_pair("client.empty(k)", "m")[0] == "UNRESOLVED"
    assert classify_pair("client.create_from_slice(&v)", "v.len()")[0] == "EXACT-UPSTREAM"
    assert classify_pair("client.create_from_slice(&v)", "other.len()")[0] == "UNRESOLVED"
    assert classify_pair("h.slice(o, e)", "e - o")[0] == "TRIMMED-UPSTREAM"
    assert classify_pair("h.slice(o, e)", None)[0] == "TRIMMED-UPSTREAM"
    # A call result is never a field handle (params_handle(...) class).
    assert classify_pair("crate::params_cache::params_handle(client, f32::as_bytes(&p))", "4")[0] == "UNRESOLVED"

    # Wrapper-key extraction: params (incl. generics), impl struct, spans.
    wrap_src = '''
impl<R: Runtime> Foo<R> for Bar where X: Y {
    pub unsafe fn launch<R2: Runtime>(
        client: &ComputeClient<R2>,
        input_handle: Handle,
        n: usize,
    ) {
        unsafe { sigmoid_f32::launch_unchecked::<R2>(client, CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(256), BufferArg::from_raw_parts(input_handle, n)); }
    }
}
'''
    spans = fn_spans(wrap_src)
    assert len(spans) == 1 and spans[0].name == "launch", "fn span parse broke"
    assert spans[0].impl_struct == "Bar", f"impl struct parse broke: {spans[0].impl_struct}"
    assert not spans[0].is_method
    assert spans[0].params == ["client", "input_handle", "n"], f"param parse broke: {spans[0].params}"
    ln = wrap_src.count("\n", 0, wrap_src.index("launch_unchecked")) + 1
    span = enclosing_fn(spans, line_start_offset(wrap_src, ln))
    assert span is not None and span.params.index("input_handle") == 1

    # Unique-let resolution + bare-ident chain chase + multi-binding refusal.
    let_src = '''
fn caller() {
    let v: Vec<f32> = make();
    let n = v.len();
    let h = client.create_from_slice(&v);
    let h2 = h;
    let a = 1;
    let a = 2;
    Foo::launch(&client, h2, n);
}
'''
    lspans = fn_spans(let_src)
    assert resolve_ident_lets("n", let_src, lspans[0]) == "v.len()"
    assert resolve_ident_lets("h2", let_src, lspans[0]) == "client.create_from_slice(&v)"
    assert resolve_ident_lets("a", let_src, lspans[0]) is None, "multi-binding must refuse"

    # Comment-line calls are never callers.
    dead = "// Foo::launch(&client, h, n)\nlet x = 1;"
    dm = PATH_CALL_RE.search(dead)
    ls = dead.rfind("\n", 0, dm.start()) + 1
    le = dead.find("\n", ls)
    assert dead[ls:le].lstrip().startswith("//"), "comment-line guard broke"

    # ── Guard-only vs structural len-use pins (Issue 766 T3) ──────────
    guard_body = "let n = output.len();\nlet tid = ABSOLUTE_POS;\nif tid < n {\n    output[tid] = f32::new(0.0f32);\n}"
    assert guard_only_uses(guard_body, "output"), "guard-only alias parse broke"
    direct_guard = "let idx = ABSOLUTE_POS;\nif idx >= input.len() {\n    terminate!();\n}"
    assert guard_only_uses(direct_guard, "input"), "direct guard parse broke"
    structural = "let n = kv.len() / 2 / kv_stride;\nlet p = kv[n - 1];"
    assert not guard_only_uses(structural, "kv"), "structural use read as guard"
    alias_arith = "let n = x.len();\nlet y = x[n - 1] + n;"
    assert not guard_only_uses(alias_arith, "x"), "arithmetic alias read as guard"


# The two global blindness floors, lifted out of `main()` (Issue 786 T1) so the
# sweep half can assert against them instead of restating them. They are
# WORKSPACE totals and stay deliberately loose: a partial checkout carries
# fewer repos, and a floor that reds on a 14-of-20 box is a floor nobody runs.
# The TIGHT, per-repo floors live in `scripts/len_derived_drift_floors.txt`,
# where the partial-clone axis is handled by `population_verdict()` instead of
# by slack. 52 kernels measured 2026-09-13 after the brace-matched body fix
# (the fixed 6000-char window's bleed had counted 74 — 22 host-side false
# positives); floor 45 means a regression that loses 7+ kernels REDS.
FLOOR_RS_FILES = 1500
FLOOR_KERNELS = 45


def classify_workspace(repos: list[Path]) -> Report:
    """HALF A + HALF B + HALF C + the guard-only re-verdict, over a repo list.

    Extracted from `main()` by Issue 786 T1 and shared with
    `scripts/len_derived_drift_sweep.py`. The four passes are the instrument:
    HALF C in particular resolves a wrapper parameter's provenance through
    *every* repo in `repos`, so the list handed in is part of the verdict and
    not merely a filter — which is why the sweep MEASURES that sensitivity
    (leave-one-out) rather than assuming it away. A second copy of a classifier
    this layered is a second thing to get wrong (Issue 755).
    """
    rep = Report()
    for repo in repos:
        half_a(repo, rep)
    for repo in repos:
        own = [k for k in rep.kernels if k.repo == repo.name]
        half_b(repo, own, rep)

    # HALF C: resolve wrapper-param provenance through workspace callers.
    trace_unresolved(rep.binds, {p.name: p for p in repos})

    # Guard-only kernels (Issue 766 T3): a len-use that only bounds thread
    # indices is capacity-tolerant — the compact_temp hazard (declared > live
    # silently deriving the WRONG SHAPE) needs structural derivation. The
    # PERSISTENT provenance rows of such kernels re-verdict GUARD-ONLY; the
    # undersized direction remains the caller's loud bug, not this audit's
    # class.
    for b in rep.binds:
        if b.verdict in ("PERSISTENT", "PERSISTENT-UPSTREAM") and rep.guard_only.get(b.kernel):
            b.verdict = "GUARD-ONLY"
            b.reason = "guard-only len-use (capacity-tolerant); " + b.reason
    return rep


def main() -> int:
    selftest()
    here = Path(__file__).resolve().parent
    workspace = here.parent.parent
    repos = derive_repos(workspace)
    if not repos:
        print("✗ no contract repos derived — refusing to print a confident zero")
        return 1

    rep = classify_workspace(repos)

    # Kernel names are crate-unique enough for the join; report per kernel.
    by_kernel: dict[str, list[BindSite]] = {}
    for b in rep.binds:
        by_kernel.setdefault(b.kernel, []).append(b)

    print(f"▸ {len(repos)} contract repo(s) · {rep.files_scanned} .rs scanned")
    print(f"▸ HALF A: {len(rep.kernels)} `.len()`-deriving cube kernel(s)")
    print(f"▸ HALF B: {len(rep.binds)} bind site(s) over those kernels\n")

    order = [
        "CAPACITY",
        "CAPACITY-UPSTREAM",
        "PERSISTENT",
        "PERSISTENT-UPSTREAM",
        "UNRESOLVED",
        "TRIMMED",
        "TRIMMED-UPSTREAM",
        "GUARDED",
        "GUARD-ONLY",
        "EXACT-UPSTREAM",
        "PARAM-LOCAL",
    ]
    counts = {v: 0 for v in order}
    for b in rep.binds:
        counts[b.verdict] = counts.get(b.verdict, 0) + 1

    never_launched = [k for k in rep.kernels if k.name not in by_kernel]
    for verdict in order:
        rows = [b for b in rep.binds if b.verdict == verdict]
        if not rows:
            continue
        print(f"── {verdict} ({len(rows)}) " + "─" * 40)
        for b in sorted(rows, key=lambda r: (r.repo, r.file, r.line)):
            print(f"  {b.repo}/{b.file}:{b.line}  {b.kernel}  {b.handle_expr}  [len: {b.length_expr}]")
            print(f"      {b.reason}")
        print()
    if never_launched:
        print(f"── NOT-LAUNCHED ({len(never_launched)}) — no launch_unchecked site found")
        for k in never_launched:
            print(f"  {k.repo}/{k.file}  {k.name}  (derives from {sorted(k.len_params)})")
        print()

    # Module constants since Issue 786 T1 — see FLOOR_RS_FILES for why they
    # are loose, and len_derived_drift_floors.txt for the tight per-repo half.
    floors = {"min_rs_files": FLOOR_RS_FILES, "min_kernels": FLOOR_KERNELS}
    problems = []
    if rep.files_scanned < floors["min_rs_files"]:
        problems.append(f"files_scanned {rep.files_scanned} < {floors['min_rs_files']}")
    if len(rep.kernels) < floors["min_kernels"]:
        problems.append(f"kernels {len(rep.kernels)} < {floors['min_kernels']}")
    if problems:
        print("✗ WALK REGRESSION: " + "; ".join(problems) + " — refusing the confident zero")
        return 1

    print("✓ report complete (report-only; exit 0 by design — see docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

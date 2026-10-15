#!/usr/bin/env python3
"""Registering a contract repo is a MULTI-FILE operation, and one file is gated.

Issue 837 T2. `agents_repo_set_gate` asserts `repo_set.txt` <-> AGENTS.md
§Repo count, and it is complete over the question it asks. But **twenty-one
more tracked files are keyed on that same registry** — every per-repo pin file
the sweep family reads — and nothing relates them to it. So registering a repo
is a 22-file operation with 1 file gated and 21 discovered at runtime, one red
sweep at a time, by whoever happens to run one next.

Measured the day riir-llm joined (2026-09-18): **19 of 21 sweeps RED** with
`UNPINNED — add a row`, every one for a reason unrelated to whatever the run
was measuring. That is the state AGENTS.md already names as this family's
worst outcome —

    a sweep that always reds is a sweep nobody runs (Issue 793, measured: the
    percentile sweep's Issue-777 findings, and four live citation drift rows,
    were sitting behind those reds)

— and it reproduced exactly. Pinning riir-llm turned three of those reds from
bookkeeping into content: riir-kat's three unqualified cross-repo citations,
riir-shader's EXPOSED trap-launder script *and* its stale `.plans/.highwater`,
and riir-clippy's seven undocumented scripts. Four live findings behind a
missing row.

⚠ **The sweeps are not wrong.** `UNPINNED — add a row (or it can never red)`
is the correct loud refusal: a repo with no row has no ceiling and could never
fail. What was missing is anything naming the set of rows owed **at the moment
the obligation is created**, instead of twenty-one multi-minute runs finding
them one at a time.

    scripts/repo_registration_gate.py            # the verdict AND the arms
    scripts/repo_registration_gate.py --list     # the per-file scope table

⛔ **The predicate CANNOT be "every file with >= N repo rows"**, and the
measurement is why: `docs_drift_floors.txt` (8 rows) and
`required_features_build_floors.txt` (11) are legitimately SUBSET-scoped —
docs_drift's own header says it records *"only WHICH repos are known to carry
drift-auditable labels"* — and demanding completeness of them would red
forever on a correct file, manufacturing the very cries-wolf state this gate
exists to prevent. Scope is therefore a per-file DECLARATION with a reason
(`scripts/repo_registration_scope.txt`), membership-pinned, reds in BOTH
directions: a SUBSET row for a file that has since reached full coverage is a
stale acknowledgement and fails, so the file cannot only ever loosen.

⚠ The scope is **on-disk canonical** repos — `repo_set.txt` INTERSECT the
derived walk — never the registry alone. A repo this box does not carry cannot
be measured by any sweep, so demanding its row here would red every partial
clone, which is the `DOCS_GATE_PARTIAL_CLONE` posture one axis over. The
absent set is DISCLOSED on the PASS line in both directions, because a
deferral printed only on failure is one nobody reads on the run that passes.

⛔ This is a docs-gate CHECK and deliberately **not a sweep**: it reads tracked
files and the registry, needs no sibling checkout, and the entire point is to
fire in the commit that registers a repo rather than on somebody's next
workstation run.

⚠ **What it does NOT assert:** that a row's VALUES are right. A row of zeros
for a repo with content passes here and reds in the sweep that owns it — which
is the correct division, since only that sweep can measure. Read the verdict
as the weaker thing it is: *the obligation was noticed*, not *the pin is
correct*.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import console_safe  # noqa: E402

console_safe.apply()

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SCOPE_PINS = HERE / "repo_registration_scope.txt"
REGISTRY = "scripts/repo_set.txt"

# A file enters the population by CARRYING repo-keyed rows, not by its name:
# `instrument_reachability_floors.txt` and `required_features_build_floors.txt`
# break the `*_drift_floors` convention and are as governed as the rest.
GLOBS = ("scripts/*floors.txt", "scripts/*expected.txt")

# Below this a file is not a per-repo pin file at all — `arm_reach_survivors`,
# `number_collisions`, the membership files whose keys are sweeps or line
# digests. Deliberately low: the cost of a false member is one scope row with a
# reason, and the cost of a false NON-member is the whole defect this gate is
# about.
MIN_REPO_ROWS = 5

# Three floors, failing differently.
#   MIN_FILES     the WALK — a glob that matches nothing governs every file
#                 and prints a confident green over zero of them.
#   MIN_REPOS     the REGISTRY parse — an empty registry makes every file
#                 complete by vacuity, which is a green that means nothing.
#   MIN_GOVERNED  the ROW PARSE — a walk that finds every file and recognises
#                 no repo row in any of them looks exactly like a workspace
#                 with no per-repo pins, and the obvious remedy is to delete
#                 this gate.
MIN_FILES = 25
MIN_REPOS = 10
MIN_GOVERNED = 15

SUBSET, EVERY = "SUBSET", "EVERY"


def tracked(root: Path, pattern: str) -> list[str]:
    """Repo-relative paths git TRACKS. A filesystem walk counts a scratch copy
    somebody left in `scripts/` as part of the contract — and this gate's own
    development left three there (Issue 837 T1)."""
    if not (root / ".git").exists():        # `.exists()`: a worktree's is a FILE
        return sorted(str(p.relative_to(root)).replace("\\", "/")
                      for p in root.glob(pattern))
    out = subprocess.run(["git", "-C", str(root), "ls-files", pattern],
                         capture_output=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        return sorted(str(p.relative_to(root)).replace("\\", "/")
                      for p in root.glob(pattern))
    return sorted(p for p in out.stdout.split() if p.strip())


def registry(root: Path) -> set[str]:
    f = root / REGISTRY
    if not f.is_file():
        return set()
    return {l.strip() for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")}


def on_disk(ws: Path, reg: set[str]) -> set[str]:
    """Canonical repos this box actually carries.

    ⛔ DELEGATED, never re-derived. `population_sync_gate` exists to catch two
    predicates disagreeing about the population, and the local copy this gate
    first grew tested `(p / ".git").exists()` — which admits a `git worktree`,
    whose `.git` is a FILE (Issue 835). The mapping back through
    `repo_alias.disk()` is not needed here: nothing is OPENED, only named.
    """
    try:
        from skill_repo_set_gate import derive_repos
    except ImportError:
        return set(reg)                     # cannot resolve -> claim nothing
    return reg & set(derive_repos(ws))


def repo_rows(text: str, reg: set[str]) -> set[str]:
    """Row keys in this file that name a registered repo."""
    out = set()
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if parts and parts[0] in reg:
            out.add(parts[0])
    return out


def unknown_rows(text: str, reg: set[str]) -> set[str]:
    """Repo-SHAPED row keys absent from the registry — the inverse direction
    (Issue 837 T3), a row that can never be evaluated by any sweep.

    ⚠ The shape test is deliberately narrow. `x86_64_matrix_floors.txt` is
    keyed by cargo PACKAGE name (`katgpt-core`, `katgpt-attn`), which is
    hyphenated exactly as a repo is, so a looser rule reports six rows in a
    file that has nothing to do with repos. Only keys sharing a prefix with a
    REGISTERED repo are considered, which is what a deregistered repo's stale
    row would look like and what a package name generally would not.
    """
    prefixes = {r.split("-")[0] for r in reg}
    out = set()
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        k = parts[0]
        if (k not in reg and "-" in k and "/" not in k and "." not in k
                and k.split("-")[0] in prefixes):
            out.add(k)
    return out


def parse_scope(path: Path) -> tuple[dict[str, str], list[str]]:
    """({file: reason}, errors) for SUBSET-declared files. Reasonless = ERROR.

    A file absent from this pin file defaults to EVERY — the safe direction:
    a forgotten declaration reds loudly, where a forgotten EVERY would green
    silently. That asymmetry is the whole reason the default is not the other
    way round.
    """
    pins: dict[str, str] = {}
    errs: list[str] = []
    if not path.is_file():
        return pins, errs
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        body, _, reason = line.partition("#")
        parts, reason = body.split(), reason.strip()
        if len(parts) != 1:
            errs.append(f"{path.name}:{n}: want `<file>  # reason`, got "
                        f"{body.strip()!r}")
            continue
        if not reason:
            errs.append(f"{path.name}:{n}: `{parts[0]}` has NO reason — a row "
                        f"nobody had to justify is a backlog wearing a pin")
            continue
        pins[parts[0]] = reason
    return pins, errs


def verdict(root: Path, ws: Path, scope: dict[str, str]):
    """(governed, incomplete, stale_scope, strays, reg, present)."""
    reg = registry(root)
    present = on_disk(ws, reg)
    governed: dict[str, set[str]] = {}
    strays: dict[str, set[str]] = {}
    for pattern in GLOBS:
        for rel in tracked(root, pattern):
            f = root / rel
            if not f.is_file():
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
            rows = repo_rows(text, reg)
            if len(rows) < MIN_REPO_ROWS:
                continue
            governed[Path(rel).name] = rows
            if odd := unknown_rows(text, reg):
                strays[Path(rel).name] = odd

    incomplete: dict[str, set[str]] = {}
    for name, rows in governed.items():
        if name in scope:                    # declared SUBSET
            continue
        if missing := present - rows:
            incomplete[name] = missing

    # Both directions: a SUBSET row whose file now covers every present repo
    # is a stale acknowledgement, and a row for a file no longer governed is
    # a pin that can never fail.
    # ⛔ PARTIAL-CLONE AXIS (the 10-04/10-07 docs_gate main-run failures): the
    # stale direction is only decidable on a box that carries the registry.
    # On a single-checkout CI lane `present` is this repo alone, every SUBSET
    # file covering it reads "complete", and the gate reds on an environment
    # property — the exact "reds every partial clone" class the incomplete
    # direction already refuses (the docstring's own rule, one axis over).
    # Undecidable here ⇒ disclosed on the PASS line, never failed.
    full_view = reg <= present
    if full_view:
        stale = sorted(n for n in scope
                       if n not in governed or not (present - governed[n]))
    else:
        stale = []
    return governed, incomplete, stale, strays, reg, present, full_view


def canary() -> list[str]:
    """Arms over this gate's OWN arithmetic, which no classifier self-test can
    reach (Issue 775's rule, the sixth instance of it).

    Unconditional, behind no flag: `docs_gate.sh` invokes each check as
    `"$PY" "$script"` with NO arguments, so an arm behind `'--canary' in
    sys.argv` never fires on a push (Issue 789, measured on eight adversary
    arms that had landed the day before).
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    reg = {"riir-a", "riir-b", "riir-c", "riir-d", "riir-e", "riir-f"}

    # 1. the row reader credits a registered key and ignores comments
    txt = "# riir-f 1\nriir-a 0 0\nriir-b 1 2\n  riir-c 3\nnot-a-repo 9\n"
    check(repo_rows(txt, reg) == {"riir-a", "riir-b", "riir-c"},
          f"repo_rows mis-read: {repo_rows(txt, reg)}")
    check("riir-f" not in repo_rows(txt, reg),
          "a COMMENTED row counted — a repo can be removed by commenting it "
          "out and the gate would still call the file complete")

    # 2. the stray detector is narrow: a package name must NOT fire, a
    #    deregistered repo must.
    check(unknown_rows("katgpt-core 1\nkatgpt-attn 2\n", reg) == set(),
          "a package-keyed row read as a deregistered repo — this is the "
          "x86_64_matrix_floors false positive the narrow rule exists for")
    check(unknown_rows("riir-gone 1 2\n", reg) == {"riir-gone"},
          "a deregistered repo's stale row was NOT reported")

    # 3. scope parsing refuses a reasonless row, and accepts a reasoned one
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "scope.txt"
        p.write_text("# c\nalpha.txt\n", encoding="utf-8")
        _, errs = parse_scope(p)
        check(errs, "a reasonless SUBSET row was ACCEPTED — the file would "
                    "become a backlog wearing a pin")
        p.write_text("alpha.txt  # subset by design\n", encoding="utf-8")
        pins, errs = parse_scope(p)
        check(not errs and pins == {"alpha.txt": "subset by design"},
              f"a reasoned row was refused: {pins}, {errs}")

        # ⛔ BLANKS and COMMENTS are skipped, not parsed. Both are ERRORS if
        # they reach the body — a blank line has no file and no reason, so it
        # would red the gate on an ordinary bit of formatting, and a comment
        # would be read as a filename. This arm exists because the mutant that
        # turns the skip's `or` into `and` survived everything else.
        p.write_text("\n   \n# a comment line\nalpha.txt  # reasoned\n\n",
                     encoding="utf-8")
        pins, errs = parse_scope(p)
        check(not errs and pins == {"alpha.txt": "reasoned"},
              f"a blank or comment line was PARSED as a row: {pins}, {errs}")

    # 3b. the registry reader skips blanks and comments too — same shape, and
    #     the failure is worse here: a comment read as a repo name puts a
    #     phantom into the demanded set and reds every governed file.
    with tempfile.TemporaryDirectory() as td:
        r = Path(td) / "repo_set.txt"
        (Path(td) / "scripts").mkdir()
        (Path(td) / REGISTRY).write_text(
            "# the contract set\n\nriir-a\n  riir-b\n\n", encoding="utf-8")
        check(registry(Path(td)) == {"riir-a", "riir-b"},
              f"the registry reader admitted a comment or a blank: "
              f"{registry(Path(td))}")

    # 3c. ⛔ `tracked` falls back to a filesystem glob on a tree with no `.git`
    #     — a `git archive` extraction is a legitimate population, not an
    #     error (tracked_walk's rule). The probe is `.exists()` and NOT
    #     `.is_dir()`: a `git worktree`'s `.git` is a FILE, and `.is_dir()`
    #     there would silently take this fallback inside a real checkout,
    #     counting untracked scratch copies as contract files (Issue 836).
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "scripts").mkdir()
        (root / "scripts/a_floors.txt").write_text("x\n", encoding="utf-8")
        check(tracked(root, "scripts/*floors.txt") == ["scripts/a_floors.txt"],
              f"a tree with no .git did not fall back to the walk: "
              f"{tracked(root, 'scripts/*floors.txt')}")

    # 3d. ⛔ …and inside a REAL repo it must ask git, so an UNTRACKED scratch
    #     copy is not a contract file. This is the arm that distinguishes the
    #     two branches: without it, dropping the `not` above still answers
    #     correctly, because a bare tmpdir makes `git ls-files` fail and the
    #     fallback runs anyway. The behaviour only differs where git can
    #     SUCCEED and disagree with the glob — and it is not a hypothetical:
    #     this gate's own development left three scratch files in `scripts/`.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "r"
        (root / "scripts").mkdir(parents=True)
        for a in (("init", "-q", "-b", "main"), ("config", "user.email", "t@t"),
                  ("config", "user.name", "t")):
            subprocess.run(["git", "-C", str(root), *a], check=True,
                           capture_output=True)
        (root / "scripts/real_floors.txt").write_text("x\n", encoding="utf-8")
        (root / "scripts/.scratch_floors.txt").write_text("x\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "scripts/real_floors.txt"],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"],
                       check=True, capture_output=True)
        got = tracked(root, "scripts/*floors.txt")
        check(got == ["scripts/real_floors.txt"],
              f"an UNTRACKED file entered the population inside a real repo "
              f"({got}) — the walk is being used where git should be asked")

    # 4. the VERDICT arithmetic, over a synthetic tree — the part a classifier
    #    self-test cannot reach, and the reason this function exists.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "repo"
        (root / "scripts").mkdir(parents=True)
        (root / REGISTRY).write_text("\n".join(sorted(reg)) + "\n",
                                     encoding="utf-8")
        rows = "\n".join(f"{r} 0 0" for r in sorted(reg))
        (root / "scripts/full_floors.txt").write_text(rows + "\n", encoding="utf-8")
        (root / "scripts/partial_floors.txt").write_text(
            "\n".join(f"{r} 0 0" for r in sorted(reg)[:5]) + "\n", encoding="utf-8")
        (root / "scripts/tiny_floors.txt").write_text("riir-a 0\n", encoding="utf-8")

        ws = root.parent
        for r in reg:                        # a walkable workspace
            (ws / r / ".git").mkdir(parents=True)
            (ws / r / "BOUNDARY.md").write_text("x", encoding="utf-8")

        gov, inc, stale, _, _, present, _ = verdict(root, ws, {})
        check("tiny_floors.txt" not in gov,
              "a 1-row file entered the population — MIN_REPO_ROWS is not "
              "separating pin files from membership files")
        check("full_floors.txt" in gov and not inc.get("full_floors.txt"),
              f"a COMPLETE file was reported incomplete: {inc}")
        check(inc.get("partial_floors.txt") == set(sorted(reg)[5:]),
              f"the missing set is wrong: {inc.get('partial_floors.txt')}")

        # the declaration silences exactly the declared file, and nothing else
        gov, inc, stale, _, _, _, _ = verdict(root, ws, {"partial_floors.txt": "by design"})
        check(not inc, f"a declared SUBSET file still reported incomplete: {inc}")

        # ⛔ BOTH directions: declaring a COMPLETE file SUBSET is a stale
        # acknowledgement and must red, or the pin file only ever loosens.
        gov, inc, stale, _, _, _, _ = verdict(root, ws, {"full_floors.txt": "wrong"})
        check("full_floors.txt" in stale,
              "a SUBSET row on a file that covers every present repo did NOT "
              "red — the pin file can then only ever loosen")
        gov, inc, stale, _, _, _, _ = verdict(root, ws, {"gone_floors.txt": "wrong"})
        check("gone_floors.txt" in stale,
              "a SUBSET row for a file no longer governed did NOT red")

        # 5. ⛔ the PARTIAL-CLONE posture: a registered repo this box does not
        #    carry must NOT be demanded, or every partial clone reds.
        import shutil
        shutil.rmtree(ws / "riir-f")
        gov, inc, _, _, _, present, full = verdict(root, ws, {})
        check("riir-f" not in present,
              "an ABSENT repo stayed in the demanded set — every partial "
              "clone would red on a row no sweep there could measure")
        check(not inc.get("full_floors.txt"),
              f"a complete file reds once a repo goes missing: {inc}")
        # And the STALE direction must go quiet on the same partial box: a
        # SUBSET declaration cannot be adjudicated from a partial view (the
        # CI-lane failure class this axis exists for).
        gov, inc, stale, _, _, _, _ = verdict(root, ws, {"full_floors.txt": "wrong"})
        check(not full and not stale,
              "the stale direction fired on a PARTIAL box — a single-checkout "
              "CI lane reds on an environment property, not a content one")

    return fails


def main(argv) -> int:
    fails = canary()
    if fails:
        print("repo_registration_gate SELF-TEST FAILED:")
        for f in fails:
            print("  ✗ " + f)
        return 2

    ws = REPO_ROOT.parent
    scope, errs = parse_scope(SCOPE_PINS)
    gov, inc, stale, strays, reg, present, full_view = verdict(REPO_ROOT, ws, scope)

    if "--list" in argv:
        for name in sorted(gov):
            kind = SUBSET if name in scope else EVERY
            print(f"  {kind:6s} {name:44s} {len(gov[name]):2d} row(s)")
        return 0

    if len(gov) < MIN_GOVERNED or len(reg) < MIN_REPOS:
        print(f"⛔ WALK REGRESSION — {len(gov)} governed file(s) (floor "
              f"{MIN_GOVERNED}), {len(reg)} registered repo(s) (floor "
              f"{MIN_REPOS}). A green here would be a green ZERO.")
        return 1
    n_files = sum(len(tracked(REPO_ROOT, g)) for g in GLOBS)
    if n_files < MIN_FILES:
        print(f"⛔ WALK REGRESSION — {n_files} tracked pin file(s) (floor "
              f"{MIN_FILES})")
        return 1

    for e in errs:
        print(f"  ✗ {e}")
    for name, missing in sorted(inc.items()):
        print(f"  ✗ {name}: no row for {', '.join(sorted(missing))} — "
              f"registered and ON DISK, so a sweep reading this file cannot "
              f"measure it and will red UNPINNED")
    for name in stale:
        print(f"  ✗ {SCOPE_PINS.name}: `{name}` is declared SUBSET but is "
              f"complete (or no longer governed) — a stale acknowledgement, "
              f"and a pin file that only ever loosens is not a wall")
    for name, odd in sorted(strays.items()):
        print(f"  ✗ {name}: row(s) for {', '.join(sorted(odd))}, absent from "
              f"{REGISTRY} — a pin no sweep can ever evaluate")

    if errs or inc or stale or strays:
        print(f"✗ repo-registration gate FAILED — registering a repo is a "
              f"{len(gov) + 1}-file operation and the rows below are missing")
        return 1

    absent = sorted(reg - present)
    tail = (f"  [⚠ {len(absent)} registered repo(s) not on this box "
            f"({', '.join(absent)}) — NOT demanded, since no sweep here could "
            f"measure them; their rows are governed on a full checkout]"
            if absent else
            "  [⚠ every registered repo is on this box, so the absent-repo "
            "path is UNEXERCISED by live data and is asserted only by its arm]")
    if not full_view:
        tail += ("  [⚠ PARTIAL box — the SUBSET-stale direction is UNDECIDABLE "
                 "here (a SUBSET scope reads complete over 1 repo) and is "
                 "deferred to a full checkout, never failed]")
    print(f"    ✓ repo-registration gate PASSED — every one of {len(gov)} "
          f"per-repo pin file(s) has a row for all {len(present)} on-disk "
          f"canonical repo(s), {len(scope)} declared SUBSET, 0 stale, 0 "
          f"unregistered row(s) (floors {MIN_GOVERNED}/{MIN_REPOS}/{MIN_FILES})"
          f".{tail} ⚠ It does NOT claim a row's VALUES are right — only the "
          f"sweep that owns the file can measure that")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

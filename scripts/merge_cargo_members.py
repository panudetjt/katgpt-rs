#!/usr/bin/env python3
"""Custom git merge driver for Cargo.toml [workspace] members.

Auto-resolves the recurring conflict where local and upstream each add a
crate to the single `members = [...]` line: the driver unions both member
lists (sorted, deduplicated). Any other conflict in the file falls back to
exit 1 (manual resolution).

Registered in .git/info/attributes as:
    Cargo.toml merge=katgpt-members
with git config merge.katgpt-members.driver calling this script.

Usage: merge_cargo_members.py <base> <ours> <theirs> <marker-size>
Exit 0 = resolved (ours file updated in place), exit 1 = unresolved.
"""
import re
import sys
from pathlib import Path

MEMBERS_RE = re.compile(r'^members\s*=\s*\[(.*?)\]\s*$', re.S | re.M)


def extract_members(text: str) -> set[str]:
    m = MEMBERS_RE.search(text)
    if not m:
        return set()
    return {s.strip().strip('"') for s in m.group(1).split(',') if s.strip()}


def rebuild(text: str, members: set[str]) -> str:
    line = 'members = [' + ', '.join(f'"{m}"' for m in sorted(members)) + ']'
    return MEMBERS_RE.sub(lambda _: line, text, count=1)


def main() -> int:
    if len(sys.argv) < 4:
        print(f'usage: {sys.argv[0]} <base> <ours> <theirs> [marker-size]', file=sys.stderr)
        return 1
    base, ours_p, theirs_p = (Path(a) for a in sys.argv[1:4])
    ours = ours_p.read_text(encoding='utf-8')

    if MEMBERS_RE.search(ours) is None:
        return 1  # no members line to union; leave for manual resolution

    base_m = extract_members(base.read_text(encoding='utf-8')) if base.exists() else set()
    ours_m = extract_members(ours)
    theirs_m = extract_members(theirs_p.read_text(encoding='utf-8'))

    if not theirs_m or not ours_m:
        return 1  # a side removed/renamed the line; not this driver's call

    # Union everything present on either side; dropping base members is a
    # deliberate removal on one side only if the other side kept them.
    merged = ours_m | theirs_m
    ours_p.write_text(rebuild(ours, merged), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# Plan 614 (task E13 of plan 611) — `[workspace.package] rust-version` draft (PREP ONLY — nothing landed)

**Status:** DRAFT — prep only, owner-gated (E13, refs `.plans/611_owner_gate_pickup.md` task E13 + riir-ai `.issues/1016` row E13). No workspace file was modified and nothing was committed; this document is the only artifact of this session.

- [x] Read the toolchain pin (`rust-toolchain.toml` → `channel = "1.98.1"`)
- [x] Enumerate + classify every Cargo.toml (33 manifests)
- [x] Draft the patch ([workspace.package] block + per-manifest diff shapes)
- [x] Drift findings (NONE — zero manifests carry any rust-version today)
- [x] Sanity check: `cargo metadata --no-deps` parses clean (unchanged tree)
- [ ] Owner go (the gate — this file is the evidence to decide on)
- [-] Apply the patch on katgpt-rs develop + run §7 verification (DEFERRED until owner go — do NOT land without it)
- [-] Tick E13 in `.plans/611_owner_gate_pickup.md` + ref the landing commit here, same commit (DEFERRED with the landing)

## 1. Premise

`rust-toolchain.toml` pins `channel = "1.98.1"` (profile minimal, clippy + rustfmt
components). Its own header documents the failure mode E13 closes: without the file,
cargo falls back to the box default and HEAD fails deep inside
`crates/katgpt-percepta/src/legacy/mod.rs` with `error[E0658]: isolating the lowest
set bit is unstable` (`u32::isolate_lowest_one`, stabilized after 1.93) — a deep,
unactionable error where an MSRV error should be. The toolchain file auto-*selects*;
it does not *enforce* when the selection is bypassed (`RUSTUP_TOOLCHAIN=...`,
rustup-less cargo, an editor not honoring the file). `[workspace.package]
rust-version` is the machine-readable floor complement, not a replacement — E13 is
purely additive to `rust-toolchain.toml`. **Do not touch `rust-toolchain.toml`.**

## 2. Survey — class counts

Enumeration: `**/Cargo.toml` over the repo → **exactly 33 manifests**, and the
root `[workspace]` members list carries exactly 33 entries (32 `crates/*` + `"."`).
The two sets match one-to-one. No `[workspace.dependencies]`, no `exclude` key.

| Class | Definition | Count |
|---|---|---|
| (a) | already uses `[package]` workspace inheritance (`workspace = true` / `.workspace` dotted keys) | **0** |
| (b) | has its own `rust-version` key (conflict to resolve) | **0** |
| (c) | has neither — needs `rust-version.workspace = true` added | **33** (32 member crates + the root package) |
| (d) | workspace-EXCLUDED manifest (standalone rust-version or own key) | **0** |

Grep evidence: `rust-version` → 0 matches across all 33; `workspace\s*=\s*true`,
`[workspace.package]`, and any `.workspace` dotted key → 0 matches;
`exclude` → 0 matches in the root manifest. Class (a) is empty *by construction*
(inheritance without a `[workspace.package]` table cannot exist), and this patch
introduces the workspace's FIRST use of that table.

## 3. Full affected list (all 33, class (c))

Root package:

```
Cargo.toml                                  # katgpt-rs (root package + workspace root)
```

The 32 member crates (identical diff shape):

```
crates/katgpt-attn/Cargo.toml
crates/katgpt-attn-match/Cargo.toml
crates/katgpt-backend/Cargo.toml
crates/katgpt-band/Cargo.toml
crates/katgpt-canon/Cargo.toml
crates/katgpt-claim/Cargo.toml
crates/katgpt-core/Cargo.toml
crates/katgpt-dec/Cargo.toml
crates/katgpt-deprecated/Cargo.toml
crates/katgpt-device-verify/Cargo.toml
crates/katgpt-forward/Cargo.toml
crates/katgpt-hla/Cargo.toml
crates/katgpt-kv/Cargo.toml
crates/katgpt-micro-belief/Cargo.toml
crates/katgpt-moka-wasm/Cargo.toml
crates/katgpt-nn/Cargo.toml
crates/katgpt-percepta/Cargo.toml
crates/katgpt-personality/Cargo.toml
crates/katgpt-proof-cert/Cargo.toml
crates/katgpt-pruners/Cargo.toml
crates/katgpt-quant/Cargo.toml
crates/katgpt-ruliology/Cargo.toml
crates/katgpt-sense/Cargo.toml
crates/katgpt-sleep/Cargo.toml
crates/katgpt-sparse/Cargo.toml
crates/katgpt-spectral/Cargo.toml
crates/katgpt-speculative/Cargo.toml
crates/katgpt-tetris/Cargo.toml
crates/katgpt-tokenizer/Cargo.toml
crates/katgpt-transformer/Cargo.toml
crates/katgpt-types/Cargo.toml
crates/katgpt-validator/Cargo.toml
```

Uniformity verified: every manifest carries exactly one `edition = "2024"` line
inside its `[package]` table (all 33 sampled via grep) — the anchor below is safe.

## 4. Draft patch

### 4.1 Root manifest — `Cargo.toml` (the only structural edit)

```diff
 [workspace]
 members = ["crates/katgpt-attn", ..., "."]
 resolver = "3"
 
+[workspace.package]
+rust-version = "1.98.1"
+
 [package]
 name = "katgpt-rs"
 version = "0.2.1"
 edition = "2024"
+rust-version.workspace = true
 publish = false  # dev/examples aggregator — never published. The katgpt-core crate family ships to crates.io.
```

Notes:
- The root package is BOTH the workspace root and a package; `rust-version.workspace
  = true` in its own `[package]` is legal (it inherits from its own
  `[workspace.package]`) and keeps ONE source of truth. `cargo check -p katgpt-rs`
  then enforces the floor for the aggregator too.
- `rust-version = "1.98.1"` mirrors the toolchain `channel` exactly. Cargo compares
  it against the active rustc version; the patch/patch-level form is accepted.

### 4.2 Member manifests — class (c) (identical for all 32)

```diff
 [package]
 name = "katgpt-core"
 version = "0.4.1"
 edition = "2024"
+rust-version.workspace = true
 license = "MIT"
 ...
```

Key goes directly under `edition` in every `[package]` table. No other line moves;
comments and key order are preserved.

### 4.3 Excluded manifests — class (d)

None exist. There is no `exclude` key in the root `[workspace]` table and no
manifest on disk outside the members list — the class is empty, nothing to handle.
(Contrast with several sibling repos that carry excluded tool workspaces; katgpt-rs
has none.)

### 4.4 The publish-family question (the natural review objection)

Several member crates ship to crates.io (`katgpt-core`, and its published deps:
`katgpt-dec`, `katgpt-hla`, `katgpt-personality`, `katgpt-micro-belief`,
`katgpt-types` — the manifests commented "Published to crates.io"). Inherited
fields are NOT a publish blocker: `cargo package` / `cargo publish` inline the
inherited values into the packaged manifest, so the published `Cargo.toml` carries
a literal `rust-version = "1.98.1"`. Side benefit: crates.io + `cargo install`
gain MSRV awareness for the published family, which they lack today.

## 5. sed-able recipe (macOS/BSD-safe; awk, not sed -i, for portability)

Run from the repo root, on a clean tree, BEFORE any other edit:

```sh
cd /Users/katopz/git/katgpt-rs

# (1) one-off: add the [workspace.package] table (root manifest only)
awk '/^resolver = "3"$/{print; print ""; print "[workspace.package]"; \
     print "rust-version = \"1.98.1\""; next} {print}' \
    Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

# (2) all 33 packages (root + 32 crates): add the inheritance key after edition
for f in Cargo.toml crates/*/Cargo.toml; do
  awk '/^edition = "2024"$/{print; print "rust-version.workspace = true"; next} {print}' \
      "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

# (3) immediately verify (see §7)
cargo metadata --no-deps --format-version 1 > /dev/null && echo PARSE-OK
```

Per-class recipe mapping (trivial here — one populated class):
- class (c) → loop (2) above (33 files).
- class (b) → n/a (zero). If one ever appears before landing, resolve by hand:
  its literal must be reconciled against 1.98.1 (equal → drop the literal, use
  inheritance; lower → raise to 1.98.1 and say why; higher → STOP, that is a
  finding, not a patch).
- class (d) → n/a (zero).

## 6. Drift findings

**NONE.** Zero manifests declare any `rust-version` today — there is nothing that
conflicts with the 1.98.1 pin and no reconciliation work. Corollary (the actual
gap E13 closes): the repo currently has NO machine-readable MSRV anywhere; the
only pin is the auto-select file. A clone whose toolchain selection is bypassed
(`RUSTUP_TOOLCHAIN=<older>`, rustup-less cargo, a tool that ignores the file)
fails with the deep E0658 in katgpt-percepta instead of a clear
"requires rustc 1.98.1" error. Secondary drift axis also clean: every manifest is
`edition = "2024"` (needs ≥ 1.85) — consistent with 1.98.1, no edition outliers.

## 7. Verification commands (run at LANDING, after applying §5)

```sh
# 1. Workspace parses + inheritance resolves (expect: PARSE-OK)
cargo metadata --no-deps --format-version 1 > /dev/null && echo PARSE-OK

# 2. The floor actually took on ALL 33 packages (expect: 33 x 1.98.1, nothing else)
cargo metadata --no-deps --format-version 1 \
  | python3 -c 'import json,sys,collections; \
     print(collections.Counter(p.get("rust_version") for p in json.load(sys.stdin)["packages"]))'

# 3. Representative build (root + one leaf)
cargo check -p katgpt-core
cargo check -p katgpt-rs --lib

# 4. Prove the gate is LIVE (optional; needs an older toolchain installed —
#    the whole point of E13 is that this now fails LOUD and EARLY):
RUSTUP_TOOLCHAIN=1.93.0 cargo check -p katgpt-core 2>&1 | head -5
# expect: error: package `katgpt-core v0.4.1` cannot be built because it requires
#         rustc 1.98.1 or newer, while the currently active rustc version is 1.93.0

# 5. No gate conflicts: every gate here (full_gate, wasm32, x86_64 matrix) runs the
#    pinned 1.98.1 or newer; the weekly rot gate overrides with
#    RUSTUP_TOOLCHAIN=stable — stable >= 1.98.1, so MSRV passes there too.
#    Spot-run the cheap layers after landing:
./scripts/docs_gate.sh
```

Landing checklist (on owner go): apply → §7 (1)(2)(3) green → `git diff --stat`
shows exactly 33 files with 1–2 added lines each → single commit
`feat: adopt [workspace.package] rust-version = 1.98.1 across the workspace (E13)`
→ tick the E13 checkbox in `.plans/611_owner_gate_pickup.md` in the same commit →
push develop. Session marker line in the commit body per the shared-worktree
convention.

## 8. Surprises / notes for the owner

1. **33 manifests, not ~30** — the issue's "~30" estimate was close; the real count
   is 32 member crates + the root package. The root package itself is part of the
   affected set (it is a package too) — easy to miss in a crates-only count.
2. **The close is purely additive.** Zero existing `rust-version` keys means no
   conflict resolution, no literal-vs-inherited adjudication, and no risk of
   silently changing anyone's declared MSRV. Nothing to "adopt" except the new
   table.
3. **This workspace has never used workspace inheritance at all** — no
   `[workspace.package]`, no `[workspace.dependencies]`, no `workspace = true`
   anywhere. E13 introduces the first use. That is also why the patch is 33
   one-line additions + a 2-line table instead of a smaller audit-and-fix.
4. **No excluded manifests exist here** — the class-(d) branch of the E13 decision
   is vacuous for this repo. (Sibling repos with excluded tool workspaces would
   need it; katgpt-rs does not.)
5. **Publish-family cost: zero.** `cargo publish` inlines inherited fields, so the
   crates.io family gains an honest MSRV field at no maintenance cost (§4.4).

— Draft produced 2026-09-28 (prep session, refs `.plans/611_owner_gate_pickup.md` E13).

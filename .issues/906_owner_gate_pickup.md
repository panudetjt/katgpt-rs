# Owner-gate pickup — katgpt-rs

**Status:** OPEN — this repo's remaining owner-gated items. Public-repo hygiene note: the cross-workspace summary that briefly lived here as `905_workspace_owner_gated_decisions_summary.md` (2026-09-28) was relocated to the private workspace hub the same day and is deliberately NOT referenced from this public repo — no private paths, hashes, or decision content belong here. Number 905 is SPENT — never reuse it.

## Items

- [x] E12 — **ONE-PAGER PREPPED 2026-09-28** (`.plans/613_e12_gemma_base_model_options.md`): both options costed + a verdict row for the owner; agent recommendation on file is **(b) retire** (the negative is 12/12 harmful cells with a stable rate; the IT-vs-base caveat refines a negative, it does not flip it). Freshness facts found while prepping: the paper-scale Run 3 NEVER COLLECTED (task `Ready`, no output file — addendum landed in Bench 668), and both boxes re-verified to carry only IT-tuned Gemma GGUFs. **OWNER-GATED: the (a)/(b) pick lands on the owner's mark.**
- [-] E13 — workspace `rust-version` pin (evidence `HISTORY.md:3213`, the T3 defer): prep the `[workspace.package] rust-version` patch + the list of ~30 affected manifests; land only on owner go. **PREP COMPLETE 2026-09-30 (verified in a detached worktree at `befd788c7`, then removed — nothing landed):** the patch is 33 manifests / +36 lines — (1) root `Cargo.toml` gains `[workspace.package]` with `rust-version = "1.98.1"` (inserted between the `[workspace]` block and the root `[package]`) + `rust-version.workspace = true` on the root's own `[package]`; (2) every one of the 32 `crates/*/Cargo.toml` members gains `rust-version.workspace = true` directly after its `edition` line. Verified: `cargo check -p katgpt-core` green and `cargo metadata --no-deps` reports `rust_version = "1.98.1"` on ALL 33 packages (inheritance accepted). Landing recipe (one command, idempotent per file — re-derive, never re-apply a stale diff):
  ```python
  import re, pathlib
  root = pathlib.Path('.')  # repo root
  p = root / 'Cargo.toml'
  t = p.read_text()
  t = t.replace('[package]\nname = "katgpt-rs"', '[workspace.package]\nrust-version = "1.98.1"\n\n[package]\nname = "katgpt-rs"', 1)
  t = t.replace('edition = "2024"\npublish = false', 'edition = "2024"\nrust-version.workspace = true\npublish = false', 1)
  p.write_text(t)
  for mf in sorted(root.glob('crates/*/Cargo.toml')):
      t = mf.read_text()
      if 'rust-version' in t: continue
      mf.write_text(re.sub(r'(\nedition = "[^"]+"\n)', r'\1rust-version.workspace = true\n', t, count=1))
  ```
  Owner-gate notes for the landing decision: (a) the `katgpt-core` crate family SHIPS TO CRATES.IO — a published `rust-version` becomes the public MSRV claim (the toolchain pin today is repo-local via `rust-toolchain.toml`, invisible to consumers); (b) `rust-version` does NOT block `cargo build` on older toolchains — it only gates version-selection metadata (resolver + `cargo add` compatibility) — so the risk is claim-shape, not build breakage; (c) future channel bumps must now touch TWO places (`rust-toolchain.toml` + `[workspace.package]`) — a drift the docs gate could pin after landing. **LANDS ONLY ON OWNER GO.**

# Owner-gate pickup — katgpt-rs (refs .issues/906)

**Status:** OPEN — agent-pickupable prep tasks; executions stay owner-gated.

Local record: `.issues/906_owner_gate_pickup.md`. Cross-workspace context lives in the private workspace hub and is intentionally not linked from this public repo.

## Tasks

- [ ] E12 — draft the Gemma-license one-pager (accept + provision HF token to the 4090, vs retire the check as permanently blocked); attach to `.issues/906` on landing.
- [x] E13 — draft the `[workspace.package] rust-version` patch and the ~30-manifest affected list in a scratch worktree; land on owner go only. **PREP DONE 2026-09-28 — draft at `.plans/611_e13_rust_version_draft.md`: 33 manifests (32 crates + root), zero drift (no manifest carries any rust-version today — the workspace's FIRST inheritance use), toolchain pin 1.98.1, `cargo metadata` parse-verified on the unchanged tree. Landing = owner go (`- [-]` in the draft).**

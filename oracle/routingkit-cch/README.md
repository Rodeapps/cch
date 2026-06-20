# routingkit-cch (vendored)

Vendored from crates.io `routingkit-cch 0.1.3` (BSD-2-Clause) on 2026-06-03
for the Rapidonkey CCH Phase 1a work — see:

- `docs/superpowers/plans/2026-06-03-cch-phase1a-bindings.md` — Phase 1a plan
- `research/CCH/PHASE_0_RESULTS_2026-06-03.md` — Phase 0 results memo
- `native/.cch-phase1a-investigation.md` — pre-vendoring investigation notes

## Why vendored

`routingkit-cch 0.1.3` exposes only single-source / single-target queries via
`CCHQuery`. It does not surface RoutingKit C++'s bucket-based many-to-many
distance matrix capability (`pin_targets` + `run_to_pinned_targets` +
`get_distances_to_targets`). Phase 0's looped point-to-point fallback hit
650 s for a 1000×1000 matrix on Romania vs the 10 s success gate.

Phase 1a adds the missing binding directly to this vendored copy, keeping
our timeline independent of the upstream maintainer. The intent is to
upstream the addition as a PR after the binding has stabilized here.

## Relationship to upstream

Diff from upstream (will grow as Phase 1a Tasks 4–5 land):

- (Task 2) no source changes — pure vendoring
- (Tasks 4–5) new C++ free function `cch_compute_distance_matrix` in
  `src/routingkit_cch_wrapper.{h,cc}`, matching `unsafe fn` in the `ffi`
  bridge module of `src/lib.rs`, safe Rust wrapper module
  `src/many_to_many.rs`

## License

BSD-2-Clause, inherited from upstream. See `LICENSE` (or `Cargo.toml` for
the SPDX expression).

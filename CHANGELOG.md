# Changelog

All notable changes to this crate are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-09

### Added

- `Customizer` (via `Cch::customizer`) — a reusable customizer that derives the
  elimination-tree level partition once and reuses output buffers across many
  metrics through `Customizer::customize_into`, avoiding per-call allocation on
  the hot re-customization path.

### Changed

- Customization now runs in parallel. Phase-1 reset parallelizes over
  independent arcs; the phase-2 lower-triangle relaxation runs level by level
  (barrier between levels, nodes parallel within a level). Output remains
  **bit-identical** to the previous serial implementation and to the C++
  RoutingKit oracle.
- `rayon` is now a dependency. It is pure Rust, so the crate remains free of any
  C++ or FFI.
- `Cch::customize` is unchanged in signature and output (byte-for-byte); it is
  now a thin wrapper over `Customizer`.

### Notes

- The parallel relaxation's data-race-freedom is contracted to a well-formed
  (chordal) CCH as produced by `Cch::build` or a faithful `load_struct`
  round-trip; a one-time bounds validation additionally guards against
  out-of-bounds access for any bounds-valid structure.

## [0.1.1] - 2026-06-25

### Changed

- Query paths reach parity with the C++ RoutingKit oracle by eliding per-arc
  bounds checks in the hot relaxation loops.

## [0.1.0] - 2026-06-25

### Added

- Initial release: the complete CCH pipeline in pure, safe Rust — contraction
  order (`degree_order`, `inertial_order`), structure build (`Cch::build`),
  per-metric customization, memory-mappable bundle read/write, and
  distance / distance-matrix / shortest-path queries — validated bit-for-bit
  against RoutingKit.

[0.2.0]: https://github.com/Rodeapps/cch/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Rodeapps/cch/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Rodeapps/cch/releases/tag/v0.1.0

# cch

Pure-Rust **Customizable Contraction Hierarchies** (CCH) — a fast road-routing
index, in safe idiomatic Rust.

> **Status:**
> - **Phase 1 (query/serve over bundles) — done.** Open `.cch-struct` /
>   `.cch-metric` bundles and answer distance-matrix + node-path queries. Bundles
>   must be pre-built by an external tool (e.g. RoutingKit, rapidonkey) — see
>   `examples/build_and_query.rs`.
> - **Phase 2 (pure-Rust construction) — next.** Contraction order, CCH
>   structure, per-metric customization, and a bundle writer in pure Rust.
>
> APIs will change until `0.1`.

## What it does

CCH is a two-phase shortest-path index for road networks: a one-time,
metric-independent **build** (contraction order + structure), then cheap
per-metric **customization** (e.g. distance, travel-time), and fast queries.
This crate provides, in pure Rust:

- **bundles** — open mmappable `.cch-struct` / `.cch-metric` artifacts for
  zero-copy, memory-bounded serving ✓ *Phase 1*
- **query** — elimination-tree shortest-path distance + many-to-many distance
  matrix ✓ *Phase 1*
- **unpack** — shortcut expansion → node paths (for geometry / turn-by-turn)
  ✓ *Phase 1*
- **customize** — per-metric shortcut weights *(Phase 2)*
- **build** — nested-dissection contraction order + CCH structure *(Phase 2)*

## Why

Existing high-quality CCH implementations are C++ (notably
[RoutingKit](https://github.com/RoutingKit/RoutingKit)). `cch`'s published
library has no C++ or FFI in its runtime dependencies, so downstream Rust
services embed it without a C++ toolchain. A C++ RoutingKit oracle is used
only as a dev-only differential-test dependency (not compiled by consumers).
Today the crate reads pre-built bundles; pure-Rust bundle construction arrives
in Phase 2.

## License

[MIT](LICENSE). The algorithm and bundle format derive from RoutingKit
(BSD-2-Clause) — see [`NOTICE`](NOTICE).

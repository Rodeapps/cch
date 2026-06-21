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

## Performance vs RoutingKit

Indicative numbers on a 24×24 bidirectional grid (576 nodes, ~3 000 CCH arcs).
Measured with [Criterion](https://github.com/bheisler/criterion.rs) (sample
sizes: 50 for `degree_order`/`customize`, 20 for `build`/`node_path`, 10 for
`distance_matrix`).  Run `cargo bench` to reproduce.

> **Machine not standardised** — numbers are indicative, not a guarantee.  
> **Order caveat**: both sides use `degree_order` (degree-ascending heuristic).
> A production `RoutingKit` build uses inertial-flow ordering, which typically
> yields fewer shortcuts and faster query times; once inertial-flow is added to
> this crate the query ratios will improve.

| Operation | Rust median | C++ median | Rust / C++ |
|-----------|-------------|------------|-----------|
| `degree_order` | 5.22 µs | 7.10 µs | **0.74×** (Rust faster) |
| `Cch::build` | 439 µs | 444 µs | ~1.00× (parity) |
| `customize` | 1.00 ms | 1.00 ms | ~1.00× (parity) |
| `distance_matrix` (576×576) | 17.98 ms | 12.95 ms | 1.39× (Rust slower) |
| `node_path` (200 pairs) | 3.71 ms | 3.27 ms | 1.14× (Rust slower) |

`degree_order`, `build`, and `customize` are at parity or faster.  Query
operations (`distance_matrix`, `node_path`) are 14–39% slower: the Rust
implementation uses `mmap`-backed `CchView`/`MetricView` with an extra
indirection layer (pointer-to-mmap-slice), while the C++ oracle operates on
vectors with direct pointer arithmetic.  This gap is expected to narrow with
further optimisation and is tracked as a future task.

## License

[MIT](LICENSE). The algorithm and bundle format derive from RoutingKit
(BSD-2-Clause) — see [`NOTICE`](NOTICE).

# cch

Pure-Rust **Customizable Contraction Hierarchies** (CCH) — a fast road-routing
index, in safe idiomatic Rust.

> **Status:** the full pure-Rust pipeline is implemented — build a contraction
> order + CCH structure, customize per metric, write `.cch-struct` /
> `.cch-metric` bundles, and serve distance-matrix + node-path queries, all with
> no C++ or FFI in the published library. Construction and the bundle format are
> bit-identical to RoutingKit (verified against it as a dev-only oracle). See
> `examples/build_and_query.rs` for the end-to-end flow.
>
> The contraction order uses a degree heuristic today; inertial-flow (geometric)
> ordering is the main planned enhancement. APIs may change until `0.1`.

## What it does

CCH is a two-phase shortest-path index for road networks: a one-time,
metric-independent **build** (contraction order + structure), then cheap
per-metric **customization** (e.g. distance, travel-time), and fast queries.
This crate provides, in pure Rust:

- **build** — degree-heuristic contraction order + CCH structure ✓
- **customize** — per-metric shortcut weights ✓
- **bundles** — read **and write** mmappable `.cch-struct` / `.cch-metric`
  artifacts for zero-copy, memory-bounded serving ✓
- **query** — elimination-tree shortest-path distance + many-to-many distance
  matrix ✓
- **unpack** — shortcut expansion → node paths (for geometry / turn-by-turn) ✓
- *(planned)* **inertial-flow ordering** — geometric nested dissection for
  higher-quality hierarchies on road networks

## Why

Existing high-quality CCH implementations are C++ (notably
[RoutingKit](https://github.com/RoutingKit/RoutingKit)). `cch`'s published
library has no C++ or FFI in its runtime dependencies, so downstream Rust
services embed it without a C++ toolchain. A C++ RoutingKit oracle is used
only as a dev-only differential-test dependency (not compiled by consumers).
The crate builds, customizes, writes, and serves CCH bundles entirely in Rust,
bit-identical to RoutingKit.

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
| `node_path` (200 pairs) | 3.55 ms | 2.89 ms | 1.23× (Rust slower) |

`degree_order`, `build`, and `customize` are at parity or faster.  Query
operations (`distance_matrix`, `node_path`) are 23–39% slower: the Rust
implementation uses `mmap`-backed `CchView`/`MetricView` with an extra
indirection layer (pointer-to-mmap-slice), while the C++ oracle operates on
vectors with direct pointer arithmetic.  This gap is expected to narrow with
further optimisation and is tracked as a future task.

## License

[MIT](LICENSE). The algorithm and bundle format derive from RoutingKit
(BSD-2-Clause) — see [`NOTICE`](NOTICE).

# cch

[![crates.io](https://img.shields.io/crates/v/cch.svg)](https://crates.io/crates/cch)
[![docs.rs](https://img.shields.io/docsrs/cch)](https://docs.rs/cch)
[![CI](https://github.com/Rodeapps/cch/actions/workflows/ci.yml/badge.svg)](https://github.com/Rodeapps/cch/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![MSRV 1.85](https://img.shields.io/badge/MSRV-1.85-blue)

**Customizable Contraction Hierarchies for fast road routing — the whole pipeline in pure, safe Rust.**

`cch` builds a metric-independent contraction hierarchy from a road graph, customizes it per weight profile (distance, travel-time, …), serializes it to memory-mappable bundles, and answers **point-to-point distance**, **many-to-many distance-matrix**, and **shortest-path** queries — with **no C++ and no FFI** in the published library.

It is a from-scratch Rust reimplementation of [RoutingKit](https://github.com/RoutingKit/RoutingKit)'s CCH, validated **bit-for-bit** against it: same contraction order quality, **bit-identical** structure and customization arrays, **byte-identical** on-disk bundles, and identical query results. It was extracted from — and now powers — a production distance-matrix routing service.

```toml
[dependencies]
cch = "0.2"
```

## Quick start

```rust
use cch::graph::Graph;
use cch::{distance_matrix, degree_order, node_path, Cch};

// Input graph in CSR form: a 4-node path 0—1—2—3 with unit-weight arcs.
// (CCH treats the input as symmetric / undirected internally.)
let graph = Graph {
    first_out: vec![0, 1, 2, 3, 3],
    head: vec![1, 2, 3],
    weight: vec![1, 1, 1],
};

// 1. Contraction order (metric-independent). `degree_order` needs no
//    coordinates; use `inertial_order` for production-quality orders.
let order = degree_order(&graph);

// 2. Build the CCH structure once.
let cch = Cch::build(&graph, &order);

// 3. Customize a metric (cheap — repeat per weight profile).
let metric = cch.customize(&graph.weight);

// For repeated re-customization, reuse buffers:
// let cust = cch.customizer();
// cust.customize_into(&graph.weight, &mut metric);

// 4. Query in memory, via zero-copy views.
let dm = distance_matrix(&cch.view(), &metric.view(), &[0], &[3]);
assert_eq!(dm[0], 3); // shortest distance 0 -> 3

let path = node_path(&cch.view(), &metric.view(), 0, 3);
assert_eq!(path, Some(vec![0, 1, 2, 3])); // unpacked node path
```

## Why CCH?

A **Customizable Contraction Hierarchy** is a two-phase shortest-path index built for road networks where the topology is fixed but edge weights change often (live traffic, vehicle profiles, time-of-day):

1. **Build** (once, metric-independent) — pick a contraction order, contract the graph into a chordal supergraph, and record the shortcut structure.
2. **Customize** (cheap, per metric) — push concrete edge weights through the structure to compute every shortcut's weight.
3. **Query** (very fast) — answer point-to-point, one-to-many, and many-to-many shortest paths over the customized hierarchy; unpack shortcuts to recover the full node path.

The expensive build is amortized across many cheap customizations, which is exactly what a routing service with frequently-changing weights needs.

## Highlights

- **Pure Rust, no FFI.** The published library has zero C++ in its dependency tree — embed it in a Rust service with no C/C++ toolchain. (Parallel customization uses [rayon](https://github.com/rayon-rs/rayon), also pure Rust. A C++ RoutingKit build is used *only* as a dev-time differential-test oracle; it is not part of the crate you depend on.)
- **The complete pipeline** — contraction order, structure build, per-metric customization, bundle reader **and** writer, distance / distance-matrix / path queries, and shortcut unpacking.
- **Parallel, reusable customization.** `Cch::customizer` builds a [`Customizer`](https://docs.rs/cch/latest/cch/struct.Customizer.html) once per structure; `Customizer::customize_into` re-customizes for a new weight profile without reallocating output buffers, with both phases of customization running in parallel — bit-identical to the serial, single-shot `Cch::customize`.
- **Two ordering strategies** — a lightweight `degree_order`, and **`inertial_order`**: a full geometric nested-dissection (inertial-flow max-flow / min-cut) that produces hierarchies of the *same quality as RoutingKit* (identical shortcut counts in testing).
- **Zero-copy mmap bundles.** Build once, write `.cch-struct` / `.cch-metric` files, then serve them memory-mapped — the OS page cache backs the query slices directly, so many regions can be served within a bounded memory budget. The format is byte-compatible with RoutingKit-produced bundles.
- **Proven correct.** Every stage is gated by a differential test against the C++ oracle (see [Correctness](#correctness)).
- **100% line coverage**, enforced in CI.

## API at a glance

| Step | API |
|---|---|
| Input graph (CSR) | [`cch::graph::Graph`](https://docs.rs/cch/latest/cch/graph/struct.Graph.html) |
| Contraction order | [`degree_order`](https://docs.rs/cch/latest/cch/fn.degree_order.html), [`inertial_order`](https://docs.rs/cch/latest/cch/fn.inertial_order.html) |
| Build structure | [`Cch::build`](https://docs.rs/cch/latest/cch/struct.Cch.html#method.build) |
| Customize a metric | [`Cch::customize`](https://docs.rs/cch/latest/cch/struct.Cch.html#method.customize) → [`Metric`](https://docs.rs/cch/latest/cch/struct.Metric.html) |
| Serialize / load | [`Cch::save_struct`](https://docs.rs/cch/latest/cch/struct.Cch.html#method.save_struct), [`Cch::load_struct`](https://docs.rs/cch/latest/cch/struct.Cch.html), [`Metric::save`](https://docs.rs/cch/latest/cch/struct.Metric.html#method.save) |
| Open bundles (mmap) | [`CchBundle`](https://docs.rs/cch/latest/cch/struct.CchBundle.html), [`MetricBundle`](https://docs.rs/cch/latest/cch/struct.MetricBundle.html) |
| Query | [`distance_matrix`](https://docs.rs/cch/latest/cch/fn.distance_matrix.html), [`node_path`](https://docs.rs/cch/latest/cch/fn.node_path.html), [`ElimTreeQuery`](https://docs.rs/cch/latest/cch/struct.ElimTreeQuery.html) |

Unreachable distances are reported as [`cch::INF_WEIGHT`](https://docs.rs/cch/latest/cch/constant.INF_WEIGHT.html) (`2_147_483_647`). Queries take borrowed `CchView` / `MetricView`s, so you can query a freshly-built `Cch`/`Metric` in memory (`.view()`) or a memory-mapped bundle (`.view()` on `CchBundle`/`MetricBundle`) through the same functions.

## Serving from bundles

Production serving builds once and memory-maps the result:

```rust,no_run
use cch::{CchBundle, MetricBundle, distance_matrix};

let cch = CchBundle::open("romania.cch-struct".as_ref())?;
let metric = MetricBundle::open("romania.cch-metric-distance".as_ref())?;

// Zero-copy: the slices borrow straight from the mmap'd pages.
let matrix = distance_matrix(&cch.view(), &metric.view(), &sources, &targets);
# Ok::<(), std::io::Error>(())
```

Bundles are immutable, shareable across processes via the page cache, and let a single host serve many regions without loading them all into heap.

## Performance

Indicative numbers, Rust vs the C++ RoutingKit oracle, on a 24×24 bidirectional grid (576 nodes, ~3 000 CCH arcs), measured with [Criterion](https://github.com/bheisler/criterion.rs). Both sides use the **same** contraction order so the comparison is apples-to-apples. Run `cargo bench` to reproduce on your hardware.

| Operation | Rust | C++ | Rust / C++ |
|-----------|------|-----|-----------|
| contraction order | 5.22 µs | 7.10 µs | **0.74×** (faster) |
| `Cch::build` | 439 µs | 444 µs | ~1.00× (parity) |
| `customize` | 1.00 ms | 1.00 ms | ~1.00× (parity) |
| `distance_matrix` (576×576) | 12.9 ms | 12.5 ms | ~1.00× (parity) |
| `node_path` (×200) | 3.11 ms | 3.06 ms | ~1.00× (parity) |

Every operation is at parity with (or faster than) RoutingKit. The query paths reach parity by eliding the per-arc bounds checks in the hot relaxation loops — the elision-able accesses via sliced iterators, and the data-dependent distance-array access via a `get_unchecked` guarded by a one-time structural validation. (Numbers are indicative; hardware is not standardized — run `cargo bench` for your own.)

A `customize_reuse` bench compares a fresh `Cch::customize` per call against a reused `Customizer::customize_into` on the same grid; the reused path avoids re-deriving the level partition and re-allocating output buffers, so it is never slower and is typically a few percent faster (`cargo bench --bench cch -- customize` to reproduce). The gain grows with structure size and call frequency; on this modest 24×24 fixture the parallel overhead largely offsets the savings.

## Correctness

The reference C++ RoutingKit is vendored as a **dev-only** differential-test oracle, and each stage of the pipeline is gated against it:

- **Contraction order** — `inertial_order` produces the *same shortcut count* as RoutingKit's inertial-flow order (e.g. identical on a 24×24 grid), and yields correct shortest paths.
- **Structure** — `Cch::build` is **bit-identical** to RoutingKit's structure arrays given the same graph + order.
- **Customization** — `customize` is **bit-identical** to RoutingKit's forward/backward shortcut weights.
- **Bundles** — `save_struct` / `Metric::save` produce **byte-identical** files to RoutingKit's writer, and RoutingKit can load bundles written by `cch` (and vice-versa).
- **Queries** — distance-matrix and a 200-pair shortest-path suite match RoutingKit exactly, end-to-end.

The whole crate maintains **100% line coverage**, enforced by a CI gate (`cargo llvm-cov --fail-under-lines 100`), alongside `clippy -D warnings` and `rustfmt`.

## Status & roadmap

`0.1` ships the full pipeline — both orderings, build, customize, bundle read/write, and all query types — validated against RoutingKit. Customization runs in parallel (via rayon) and supports buffer reuse across repeated calls through `Cch::customizer` / `Customizer::customize_into`, with no change to the bit-identical output. The API may still evolve before `1.0`.

Planned: narrowing the query-path performance gap and broader benchmark coverage on real continental graphs.

## Relationship to RoutingKit

[RoutingKit](https://github.com/RoutingKit/RoutingKit) (BSD-2-Clause) is the canonical C++ CCH implementation and the basis for this work. `cch` reimplements its CCH construction, customization, bundle format, and query algorithms in Rust, and is differential-tested against it for exact equivalence. The on-disk bundle format is interoperable with RoutingKit-produced artifacts.

## License

[MIT](LICENSE). The algorithm and bundle format derive from RoutingKit (BSD-2-Clause); see [`NOTICE`](NOTICE).

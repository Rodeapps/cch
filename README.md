# cch

Pure-Rust **Customizable Contraction Hierarchies** (CCH) — a fast road-routing
index, in safe idiomatic Rust.

> **Status: early development.** Phase 1 (query/serve over CCH bundles) is being
> extracted from a production routing engine; Phase 2 (pure-Rust construction)
> follows. APIs will change until `0.1`.

## What it does

CCH is a two-phase shortest-path index for road networks: a one-time,
metric-independent **build** (contraction order + structure), then cheap
per-metric **customization** (e.g. distance, travel-time), and fast queries.
This crate provides, in pure Rust:

- **build** — nested-dissection contraction order + CCH structure
- **customize** — per-metric shortcut weights
- **query** — elimination-tree shortest-path distance + many-to-many distance matrix
- **unpack** — shortcut expansion → node paths (for geometry / turn-by-turn)
- **bundles** — mmappable `.cch-struct` / `.cch-metric` artifacts for zero-copy,
  memory-bounded serving across many regions

## Why

Existing high-quality CCH implementations are C++ (notably
[RoutingKit](https://github.com/RoutingKit/RoutingKit)). `cch` is a pure-Rust
implementation — no C++ toolchain, no FFI — suitable for embedding in Rust
services and reading bundles produced by the same crate.

## License

[MIT](LICENSE). The algorithm and bundle format derive from RoutingKit
(BSD-2-Clause) — see [`NOTICE`](NOTICE).

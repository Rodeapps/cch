//! # cch
//!
//! Pure-Rust **Customizable Contraction Hierarchies** (CCH) for fast road routing:
//! build a contraction order, contract to a CCH structure, customize per metric,
//! and answer shortest-path distance / distance-matrix queries with shortcut
//! path-unpacking — over mmappable, zero-copy bundles.
//!
//! Status: early development. See the design spec and README for the planned API.
//!
//! Derives from [RoutingKit](https://github.com/RoutingKit/RoutingKit) (BSD-2-Clause);
//! see `NOTICE`.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod bundle;
pub mod graph;

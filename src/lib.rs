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

/// `RoutingKit`'s `inf_weight` sentinel for "unreachable".
///
/// Matches `routingkit/include/routingkit/constants.h:7` (`inf_weight = 2_147_483_647 = i32::MAX`).
/// This is the crate-wide sentinel emitted by [`query::distance_matrix`] for unreachable pairs.
/// Note: the oracle's C++ `cch_compute_distance_matrix` uses `u32::MAX` instead — callers
/// comparing against oracle output must treat the two as equivalent.
pub const INF_WEIGHT: u32 = 2_147_483_647;

pub mod bundle;
pub mod graph;
pub mod query;

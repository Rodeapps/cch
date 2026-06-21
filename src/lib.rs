//! # cch
//!
//! Pure-Rust **Customizable Contraction Hierarchies** (CCH) for fast road routing:
//! read mmappable, zero-copy bundles and answer shortest-path distance /
//! distance-matrix queries with shortcut path-unpacking.
//!
//! **Phase 1 (implemented):** query/serve over pre-built bundles — open a
//! `.cch-struct` + `.cch-metric`, run distance-matrix or node-path queries.
//! Bundles must be pre-built by an external tool (e.g. `RoutingKit` or the
//! rapidonkey engine) until Phase 2 adds pure-Rust construction.
//!
//! **Phase 2 (planned):** pure-Rust bundle construction — contraction order,
//! CCH structure, per-metric customization, and a bundle writer.
//!
//! Derives from [RoutingKit](https://github.com/RoutingKit/RoutingKit) (BSD-2-Clause);
//! see `NOTICE`.

#![forbid(unsafe_op_in_unsafe_fn)]

mod internal;

/// `RoutingKit`'s `inf_weight` sentinel for "unreachable".
///
/// Matches `routingkit/include/routingkit/constants.h:7` (`inf_weight = 2_147_483_647 = i32::MAX`).
/// This is the crate-wide sentinel emitted by [`distance_matrix`] for unreachable pairs.
/// Note: the oracle's C++ `cch_compute_distance_matrix` uses `u32::MAX` instead — callers
/// comparing against oracle output must treat the two as equivalent.
pub const INF_WEIGHT: u32 = 2_147_483_647;

pub mod bundle;
pub mod graph;
pub mod order;
pub mod path;
pub mod query;

pub use bundle::{CchBundle, CchView, MetricBundle, MetricView};
pub use order::degree_order;
pub use path::node_path;
pub use query::distance_matrix;

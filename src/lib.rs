//! # cch
//!
//! Pure-Rust **Customizable Contraction Hierarchies** (CCH) for fast road routing —
//! the full pipeline in safe Rust, with no C++ or FFI in the published library.
//!
//! **Build** a contraction order ([`degree_order`]) and CCH structure
//! ([`Cch::build`]), **customize** per metric ([`Cch::customize`]), **serialize**
//! to mmappable `.cch-struct` / `.cch-metric` bundles ([`Cch::save_struct`] /
//! [`Metric::save`]), then **serve** zero-copy: open bundles ([`CchBundle`] /
//! [`MetricBundle`]) and answer shortest-path distance / distance-matrix queries
//! ([`distance_matrix`]) with shortcut path-unpacking ([`node_path`]). The
//! construction and bundle format are bit-identical to `RoutingKit`, so bundles
//! interoperate with existing artifacts.
//!
//! The contraction order currently uses a degree heuristic; inertial-flow
//! (geometric) ordering, which yields higher-quality hierarchies on road
//! networks, is a planned enhancement.
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
pub mod customize;
pub mod graph;
pub mod order;
pub mod path;
pub mod query;
pub mod structure;
mod writer;

pub use bundle::{CchBundle, CchView, MetricBundle, MetricView};
pub use customize::Metric;
pub use order::degree_order;
pub use path::node_path;
pub use query::{ElimTreeQuery, distance_matrix};
pub use structure::Cch;

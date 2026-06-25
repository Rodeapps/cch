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
//! Two contraction-order heuristics are provided: a lightweight degree order
//! ([`degree_order`]) and inertial-flow (geometric) nested dissection
//! ([`inertial_order`]), which yields far higher-quality hierarchies on road
//! networks.
//!
//! # Example
//!
//! ```
//! use cch::graph::Graph;
//! use cch::{distance_matrix, degree_order, node_path, Cch};
//!
//! // Input graph in CSR form: a 4-node path 0—1—2—3 with unit-weight arcs.
//! // (CCH treats the input as symmetric / undirected internally.)
//! let graph = Graph {
//!     first_out: vec![0, 1, 2, 3, 3],
//!     head: vec![1, 2, 3],
//!     weight: vec![1, 1, 1],
//! };
//!
//! // 1. Compute a contraction order (metric-independent). `degree_order` needs
//! //    no coordinates; use `inertial_order` for production-quality orders.
//! let order = degree_order(&graph);
//!
//! // 2. Build the CCH structure once.
//! let cch = Cch::build(&graph, &order);
//!
//! // 3. Customize a metric (cheap — repeat this per weight profile).
//! let metric = cch.customize(&graph.weight);
//!
//! // 4. Query, in memory, via zero-copy views.
//! let dm = distance_matrix(&cch.view(), &metric.view(), &[0], &[3]);
//! assert_eq!(dm[0], 3); // shortest distance 0 -> 3
//!
//! let path = node_path(&cch.view(), &metric.view(), 0, 3);
//! assert_eq!(path, Some(vec![0, 1, 2, 3])); // unpacked node path
//! ```
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
pub use order::{degree_order, inertial_order};
pub use path::{PathQuery, node_path};
pub use query::{ElimTreeQuery, distance_matrix};
pub use structure::Cch;

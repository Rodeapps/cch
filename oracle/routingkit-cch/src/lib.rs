#![doc = include_str!("../README.md")]

// Expose test support utilities
pub mod shp_utils;

#[cxx::bridge]
pub mod ffi {
    extern "C++" {
        include!("routingkit_cch_wrapper.h");

        type CCH; // CustomizableContractionHierarchy
        type CCHMetric; // CustomizableContractionHierarchyMetric
        type CCHQuery; // CustomizableContractionHierarchyQuery
        type CCHPartial; // CustomizableContractionHierarchyPartialCustomization

        /// Build a Customizable Contraction Hierarchy.
        /// Arguments:
        /// * order: node permutation (size = number of nodes)
        /// * tail/head: directed arc lists (same length)
        /// * filter_always_inf_arcs: if true, arcs with infinite weight placeholders are stripped
        unsafe fn cch_new(
            order: &[u32],
            tail: &[u32],
            head: &[u32],
            log_message: fn(&str),
            filter_always_inf_arcs: bool,
        ) -> UniquePtr<CCH>;

        /// Number of nodes in the CCH.
        unsafe fn cch_node_count(cch: &CCH) -> u32;
        /// Number of contracted arcs in the CCH (i.e. cch_arc_count from
        /// routingkit's terminology, NOT the original input_arc_count).
        unsafe fn cch_arc_count(cch: &CCH) -> u32;

        /// Create a metric (weights binding) for an existing CCH.
        /// Keeps pointer to weights in CCHMetric; weights length must equal arc count.
        unsafe fn cch_metric_new(cch: &CCH, weights: &[u32]) -> UniquePtr<CCHMetric>;

        /// Run customization to compute upward/downward shortcut weights.
        /// Must be called after creating a metric and before queries.
        /// Cost: Depends on separator quality; usually near-linear in m * small constant; may allocate temporary buffers.
        unsafe fn cch_metric_customize(metric: Pin<&mut CCHMetric>);

        /// Parallel customization; thread_count==0 picks an internal default (#procs if OpenMP, else 1).
        unsafe fn cch_metric_parallel_customize(metric: Pin<&mut CCHMetric>, thread_count: u32);

        /// Allocate a new reusable partial customization helper bound to a CCH.
        unsafe fn cch_partial_new(cch: &CCH) -> UniquePtr<CCHPartial>;

        /// Reset the partial updater to empty state (no updated arcs).
        unsafe fn cch_partial_reset(partial: Pin<&mut CCHPartial>);

        /// Mark an arc as updated (weight changed) for the next partial customize call.
        /// The actual weight change must already happened in the metric's weight vector.
        unsafe fn cch_partial_update_arc(partial: Pin<&mut CCHPartial>, arc: u32);

        /// Run partial customization to update shortcut weights affected by the marked arcs.
        unsafe fn cch_partial_customize(partial: Pin<&mut CCHPartial>, metric: Pin<&mut CCHMetric>);

        /// Allocate a new reusable query object bound to a metric.
        unsafe fn cch_query_new(metric: &CCHMetric) -> UniquePtr<CCHQuery>;

        /// Reset a query so it can be reused with the (same or updated) metric.
        /// Clears internal state, sources/targets, distance labels.
        unsafe fn cch_query_reset(query: Pin<&mut CCHQuery>, metric: &CCHMetric);

        /// Add a source node with initial distance (0 for standard shortest path).
        /// Multiple sources allowed (multi-source query).
        unsafe fn cch_query_add_source(query: Pin<&mut CCHQuery>, s: u32, dist: u32);

        /// Add a target node with initial distance (0 typical).
        /// Multiple targets allowed (multi-target query).
        unsafe fn cch_query_add_target(query: Pin<&mut CCHQuery>, t: u32, dist: u32);

        /// Execute the shortest path search (multi-source / multi-target if several added).
        /// Must be called after adding at least one source & target.
        unsafe fn cch_query_run(query: Pin<&mut CCHQuery>);

        /// Get shortest distance after run(). Undefined if run() not called.
        unsafe fn cch_query_distance(query: &CCHQuery) -> u32;

        /// Extract the node path for the current query result.
        /// Reconstructs path; may traverse parent pointers.
        unsafe fn cch_query_node_path(query: &CCHQuery) -> Vec<u32>;

        /// Extract the arc (edge) path corresponding to the shortest path.
        /// Each entry is an original arc id (after shortcut unpacking).
        unsafe fn cch_query_arc_path(query: &CCHQuery) -> Vec<u32>;

        /// Compute a high-quality nested dissection order using inertial flow separators.
        /// Inputs:
        /// * node_count: number of nodes
        /// * tail/head: directed arcs (treated as undirected for ordering)
        /// * latitude/longitude: per-node coords (len = node_count)
        /// Returns: order permutation (position -> node id).
        unsafe fn cch_compute_order_inertial(
            node_count: u32,
            tail: &[u32],
            head: &[u32],
            latitude: &[f32],
            longitude: &[f32],
        ) -> Vec<u32>;

        /// Fast fallback order: nodes sorted by (degree, id) ascending.
        /// Lower quality than nested dissection but zero extra data needed.
        unsafe fn cch_compute_order_degree(node_count: u32, tail: &[u32], head: &[u32])
        -> Vec<u32>;

        /// Compute a full distance matrix between `sources` and `targets` using
        /// bucket-based CCH queries. Returns a row-major Vec of length
        /// `sources.len() * targets.len()`. Element `i * targets.len() + j` is the
        /// shortest-path distance from `sources[i]` to `targets[j]`, or
        /// `u32::MAX` if unreachable.
        unsafe fn cch_compute_distance_matrix(
            metric: &CCHMetric,
            sources: &[u32],
            targets: &[u32],
        ) -> Vec<u32>;

        // Phase 2a — persisted customized-CCH artifacts.
        // See cch_bundle.h for the file format.
        include!("cch_bundle.h");

        /// Save the CCH structure (no metrics) to the given path.
        unsafe fn cch_save_struct(cch: &CCH, path: &str) -> Result<()>;

        /// Load a previously-saved CCH structure.
        unsafe fn cch_load_struct(path: &str) -> Result<UniquePtr<CCH>>;

        /// Save a customized metric's forward+backward arrays.
        unsafe fn cch_save_metric(metric: &CCHMetric, path: &str) -> Result<()>;

        /// Load a customized metric, re-attaching it to the given CCH.
        unsafe fn cch_load_metric(cch: &CCH, path: &str) -> Result<UniquePtr<CCHMetric>>;
    }
}

// Thread-safety markers
// Safety rationale:
// - CCH (CustomizableContractionHierarchy) after construction is immutable.
// - CCHMetric after successful customize() is read-only for queries; underlying RoutingKit code
//   does not mutate metric during query operations (queries keep their own state).
// - CCHQuery holds mutable per-query state and must not be shared across threads concurrently;
//   we allow moving it between threads (Send) but not Sync.
// If RoutingKit later introduces internal mutation or lazy caching inside CCHMetric, these impls
// would need re-audit.
unsafe impl Send for ffi::CCH {}
unsafe impl Sync for ffi::CCH {}
unsafe impl Send for ffi::CCHMetric {}
unsafe impl Sync for ffi::CCHMetric {}
unsafe impl Send for ffi::CCHQuery {}
// (No Sync for CCHQuery)

// Rust wrapper over FFI
use cxx::UniquePtr;
use ffi::*;

fn is_permutation(arr: &[u32]) -> bool {
    let n = arr.len();
    let mut seen = vec![false; n];
    for &val in arr {
        if (val as usize) >= n || seen[val as usize] {
            return false;
        }
        seen[val as usize] = true;
    }
    true
}

/// Lightweight fill-in reducing order heuristic: sort nodes by (degree, id) ascending.
/// Fast but lower quality than nested dissection. Use if no coordinates or other data available.
/// Panics if tail/head have inconsistent lengths or contain invalid node ids.
pub fn compute_order_degree(node_count: u32, tail: &[u32], head: &[u32]) -> Vec<u32> {
    debug_assert!(
        tail.iter()
            .chain(head)
            .max()
            .map_or(true, |&v| v < node_count),
        "tail/head contain node ids outside valid range"
    );
    assert!(
        tail.len() == head.len(),
        "tail and head arrays must have the same length"
    );
    unsafe { cch_compute_order_degree(node_count, tail, head) }
}

/// High-quality nested dissection order using inertial flow separators.
/// Requires per-node coordinates (latitude/longitude) as input.
/// Panics if tail/head have inconsistent lengths or contain invalid node ids,
/// or if latitude/longitude lengths do not match node_count.
pub fn compute_order_inertial(
    node_count: u32,
    tail: &[u32],
    head: &[u32],
    latitude: &[f32],
    longitude: &[f32],
) -> Vec<u32> {
    debug_assert!(
        tail.iter()
            .chain(head)
            .max()
            .map_or(true, |&v| v < node_count),
        "tail/head contain node ids outside valid range"
    );
    assert!(
        tail.len() == head.len(),
        "tail and head arrays must have the same length"
    );
    assert!(
        latitude.len() == (node_count as usize) && longitude.len() == (node_count as usize),
        "latitude/longitude length must equal node count"
    );
    unsafe { cch_compute_order_inertial(node_count, tail, head, latitude, longitude) }
}

/// Immutable Customizable Contraction Hierarchy index.
pub struct CCH {
    inner: UniquePtr<ffi::CCH>,
    edge_count: usize,
    node_count: usize,
}

impl CCH {
    /// Construct a new immutable Customizable Contraction Hierarchy index.
    ///
    /// Parameters:
    /// * `order` – permutation of node ids (length = node count) produced by a fill‑in reducing
    ///   nested dissection heuristic (e.g. [`compute_order_inertial`]) or a lightweight fallback.
    /// * `tail`, `head` – parallel arrays encoding each directed arc `i` as `(tail[i], head[i])`.
    ///   The ordering routine treats them as undirected; here they stay directed for queries.
    /// * `log_message` – callback for logging progress messages during construction.
    ///   May be `|_| {}` if you do not want any messages.
    /// * `filter_always_inf_arcs` – if `true`, arcs whose weight will always be interpreted as
    ///   an application defined 'infinity' placeholder may be removed during construction to
    ///   reduce index size. (Typically keep `false` unless you prepared such a list.)
    ///
    /// Cost: preprocessing is more expensive than a single customization but usually far cheaper
    /// than building a full classical CH of the same quality. Construction copies the input
    /// slices; you may drop them afterwards.
    ///
    /// Thread-safety: resulting object is `Send + Sync` and read-only.
    ///
    /// Panics if `order` is not a valid permutation of node ids,
    /// or if `tail`/`head` have inconsistent lengths or contain invalid node ids.
    pub fn new(
        order: &[u32],
        tail: &[u32],
        head: &[u32],
        log_message: fn(&str),
        filter_always_inf_arcs: bool,
    ) -> Self {
        debug_assert!(
            is_permutation(order),
            "order array is not a valid permutation"
        );
        debug_assert!(
            tail.iter()
                .chain(head)
                .max()
                .map_or(true, |&v| (v as usize) < order.len()),
            "tail/head contain node ids outside valid range"
        );
        assert!(
            tail.len() == head.len(),
            "tail and head arrays must have the same length"
        );
        let cch = unsafe { cch_new(order, tail, head, log_message, filter_always_inf_arcs) };
        CCH {
            inner: cch,
            edge_count: tail.len(),
            node_count: order.len(),
        }
    }

    /// Phase 2a — wrap a [`UniquePtr<ffi::CCH>`] returned by
    /// [`ffi::cch_load_struct`] in a Rust [`CCH`] wrapper.
    ///
    /// The C++ side knows its own node_count and cch_arc_count; we mirror
    /// them here so the existing Rust API (which exposes edge_count /
    /// node_count) keeps working. Note that `edge_count` for a loaded CCH
    /// reflects the *cch_arc_count* (i.e. the contracted arc count), which
    /// is what `cch_metric_new` validates against — NOT the original input
    /// arc count.
    pub fn from_unique_ptr(inner: UniquePtr<ffi::CCH>) -> Self {
        // SAFETY: cch_node_count / cch_arc_count are const accessors on
        // the C++ side; they don't mutate anything.
        let node_count = unsafe { ffi::cch_node_count(inner.as_ref().unwrap()) } as usize;
        let edge_count = unsafe { ffi::cch_arc_count(inner.as_ref().unwrap()) } as usize;
        CCH {
            inner,
            edge_count,
            node_count,
        }
    }

    /// Borrow the C++ side for FFI calls.
    pub fn inner_ref(&self) -> &ffi::CCH {
        self.inner.as_ref().unwrap()
    }

    /// Number of nodes in the contracted graph.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of contracted arcs (cch_arc_count from routingkit) when
    /// loaded via [`Self::from_unique_ptr`]. For [`Self::new`]-built
    /// instances this is the input edge count.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }
}

/// A customized metric (weight binding) for a given [`CCH`].
/// Keeps ownership of the weight vector so it can be mutated for partial updates.
/// Thread-safety: `Send + Sync`; read-only for queries.
pub struct CCHMetric<'a> {
    inner: UniquePtr<ffi::CCHMetric>,
    weights: Box<[u32]>, // The C++ side stores only a raw pointer; it is valid for the lifetime of `self`.
    cch: &'a CCH,
}

impl<'a> CCHMetric<'a> {
    /// Create and customize a metric (weight binding) for a given [`CCH`].
    /// Owns the weight vector so that future partial updates can safely mutate it.
    pub fn new(cch: &'a CCH, weights: Vec<u32>) -> Self {
        assert!(
            weights.len() == cch.edge_count,
            "weights length must equal arc count",
        );
        let boxed: Box<[u32]> = weights.into_boxed_slice();
        // Temporarily borrow as slice for FFI creation
        let metric = unsafe {
            let mut metric = cch_metric_new(&cch.inner, &boxed);
            cch_metric_customize(metric.as_mut().unwrap());
            metric
        };
        CCHMetric {
            inner: metric,
            weights: boxed,
            cch,
        }
    }

    /// Parallel customization variant.
    #[cfg(feature = "openmp")]
    pub fn parallel_new(cch: &'a CCH, weights: Vec<u32>, thread_count: u32) -> Self {
        assert!(
            weights.len() == cch.edge_count,
            "weights length must equal arc count",
        );
        let boxed: Box<[u32]> = weights.into_boxed_slice();
        let metric = unsafe {
            let mut metric = cch_metric_new(&cch.inner, &boxed);
            cch_metric_parallel_customize(metric.as_mut().unwrap(), thread_count);
            metric
        };
        CCHMetric {
            inner: metric,
            weights: boxed,
            cch,
        }
    }

    /// weights slice
    pub fn weights(&self) -> &[u32] {
        &self.weights
    }

    /// Phase 2a — wrap a [`UniquePtr<ffi::CCHMetric>`] returned by
    /// [`ffi::cch_load_metric`] in a Rust [`CCHMetric`] wrapper.
    ///
    /// The loaded metric has the C++ side's `input_weight` set to `nullptr`
    /// (queries don't read it; only customize does). Our Rust-side
    /// `weights` field becomes empty: nothing references it. The
    /// metric is otherwise fully functional for distance_matrix calls.
    pub fn from_unique_ptr(inner: UniquePtr<ffi::CCHMetric>, cch: &'a CCH) -> Self {
        CCHMetric {
            inner,
            weights: Box::new([]),
            cch,
        }
    }

    /// Borrow the C++ side for FFI calls.
    pub fn inner_ref(&self) -> &ffi::CCHMetric {
        self.inner.as_ref().unwrap()
    }
}

/// Reusable partial customization helper. Construct once if you perform many small incremental
/// weight updates; this avoids reallocating O(m) internal buffers each call.
pub struct CCHMetricPartialUpdater<'a> {
    partial: UniquePtr<ffi::CCHPartial>,
    cch: &'a CCH,
}

impl<'a> CCHMetricPartialUpdater<'a> {
    /// Create a reusable partial updater bound to a given CCH. You can then apply it to any
    /// metric built from the same CCH (even if you rebuild metrics with different weight sets).
    pub fn new(cch: &'a CCH) -> Self {
        let partial = unsafe { cch_partial_new(cch.inner.as_ref().unwrap()) };
        CCHMetricPartialUpdater { partial, cch }
    }

    /// Apply a batch of (arc, new_weight) updates to the given metric and run partial customize.
    pub fn apply<T>(&mut self, metric: &mut CCHMetric<'a>, updates: &T)
    where
        T: for<'b> std::ops::Index<&'b u32, Output = u32>,
        for<'b> &'b T: IntoIterator<Item = (&'b u32, &'b u32)>,
    {
        assert!(
            std::ptr::eq(metric.cch, self.cch),
            "CCHMetricPartialUpdater must be used with metrics from the same CCH"
        );
        for (k, v) in updates {
            metric.weights[*k as usize] = *v; // safe: length invariant unchanged (Box<[u32]>)
        }
        unsafe {
            cch_partial_reset(self.partial.as_mut().unwrap());
            for (k, _) in updates {
                cch_partial_update_arc(self.partial.as_mut().unwrap(), *k);
            }
            cch_partial_customize(
                self.partial.as_mut().unwrap(),
                metric.inner.as_mut().unwrap(),
            );
        }
    }
}

/// A reusable shortest-path query object bound to a given [`CCHMetric`].
/// Thread-safety: `Send` but not `Sync`; holds mutable per-query state.
pub struct CCHQuery<'a> {
    inner: UniquePtr<ffi::CCHQuery>,
    metric: &'a CCHMetric<'a>,
    state: [bool; 2],
}

impl<'a> CCHQuery<'a> {
    /// Allocate a new reusable shortest-path query bound to a given customized [`CCHMetric`].
    ///
    /// The query object stores its own frontier / label buffers and can be reset and reused for
    /// many (s, t) pairs or multi-source / multi-target batches. You may have multiple query
    /// objects referencing the same metric concurrently (read-only access to metric data).
    pub fn new(metric: &'a CCHMetric<'a>) -> Self {
        let inner = unsafe { cch_query_new(&metric.inner) };
        CCHQuery {
            inner,
            metric,
            state: [false; 2],
        }
    }

    /// Add a source node with an initial distance (normally 0). Multiple calls allow a multi-
    /// source query. Distances let you model already-traversed partial paths.
    pub fn add_source(&mut self, s: u32, dist: u32) {
        assert!(
            (s as usize) < self.metric.cch.node_count,
            "source node id out of range",
        );
        unsafe {
            cch_query_add_source(self.inner.as_mut().unwrap(), s, dist);
        }
        self.state[0] = true;
    }

    /// Add a target node with an initial distance (normally 0). Multiple calls allow multi-target
    /// queries; the algorithm stops when the frontiers settle the optimal distance to any target.
    pub fn add_target(&mut self, t: u32, dist: u32) {
        assert!(
            (t as usize) < self.metric.cch.node_count,
            "target node id out of range",
        );
        unsafe {
            cch_query_add_target(self.inner.as_mut().unwrap(), t, dist);
        }
        self.state[1] = true;
    }

    /// Execute the forward/backward upward/downward search to settle the shortest path between the
    /// added sources and targets. Must be called after at least one source and one target.
    /// Returns a [`CCHQueryResult`] holding a mutable reference to the query.
    /// The query is automatically reset when the result is dropped.
    pub fn run<'b>(&'b mut self) -> CCHQueryResult<'b, 'a> {
        assert!(
            self.state.iter().all(|&x| x),
            "must add at least one source and one target before running the query"
        );
        unsafe {
            cch_query_run(self.inner.as_mut().unwrap());
        }
        CCHQueryResult { query: self }
    }
}

/// The result of a single execution of a [`CCHQuery`].
/// Holds a mutable reference to the query to ensure it is not reset or reused
/// while the result is still alive. So you need to drop the result before reusing the query.
/// When the result is dropped, the query is automatically reset.
pub struct CCHQueryResult<'b, 'a> {
    query: &'b mut CCHQuery<'a>,
}

impl<'b, 'a> CCHQueryResult<'b, 'a> {
    /// Return the shortest path distance result of [`CCHQuery::run`].
    /// Returns `None` if no target is reachable.
    pub fn distance(&self) -> Option<u32> {
        let res = unsafe { cch_query_distance(self.query.inner.as_ref().unwrap()) };
        if res == (i32::MAX as u32) {
            // internally distance equals `i32::MAX` means unreachable
            None
        } else {
            Some(res)
        }
    }

    /// Reconstruct and return the node id sequence of the current best path.
    /// Returns empty vec if no target is reachable.
    pub fn node_path(&self) -> Vec<u32> {
        unsafe { cch_query_node_path(self.query.inner.as_ref().unwrap()) }
    }

    /// Reconstruct and return the original arc ids along the shortest path.
    /// Returns empty vec if no target is reachable.
    pub fn arc_path(&self) -> Vec<u32> {
        unsafe { cch_query_arc_path(self.query.inner.as_ref().unwrap()) }
    }
}

impl<'b, 'a> Drop for CCHQueryResult<'b, 'a> {
    /// reset the query
    fn drop(&mut self) {
        unsafe {
            cch_query_reset(
                self.query.inner.as_mut().unwrap(),
                self.query.metric.inner.as_ref().unwrap(),
            );
        }
        self.query.state.iter_mut().for_each(|x| *x = false);
    }
}

pub mod many_to_many;
pub use many_to_many::{UNREACHABLE, distance_matrix};

#[cfg(feature = "pyo3")]
mod python_binding;

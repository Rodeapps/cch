//! Elimination-tree distance computation and full distance-matrix.
//!
//! Ported from `native/routing-core/src/cch_mmap.rs` in rapidonkey
//! (`ElimTreeQuery` + `distance_matrix_mmap`).

use crate::INF_WEIGHT;
use crate::bundle::{CchView, INVALID_ID, MetricView};

// ---------------------------------------------------------------------------
// ElimTreeQuery
// ---------------------------------------------------------------------------

/// Reusable elimination-tree query state. One instance can run many queries
/// against the same CCH (call [`Self::pin_targets`] once, then loop
/// [`Self::reset_source`] + [`Self::add_source_and_run`] +
/// [`Self::get_distances_to_targets`]).
///
/// The scratch buffers (`forward_tentative_distance`,
/// `in_backward_search_space`) are sized to `node_count` and reused across
/// queries — no per-query allocation.
pub struct ElimTreeQuery<'a> {
    cch: &'a CchView<'a>,
    /// Per-node best-known distance from the current source. Sized to
    /// `node_count`, initialised to [`INF_WEIGHT`], selectively reset by
    /// [`Self::reset_source`].
    forward_tentative_distance: Vec<u32>,
    /// CCH-internal node id of each pinned target (parallel to the caller's
    /// `targets` slice).
    target_node: Vec<u32>,
    /// `target_elimination_tree_end[i]`: ancestor of `target_node[i]` at
    /// which [`Self::pin_targets`]'s walk ran into an already-marked node,
    /// or [`INVALID_ID`] if the walk reached the root. Used by both
    /// [`Self::reset_source`] (to confine cleanup to the relevant slice) and
    /// the stack-building pass.
    target_elimination_tree_end: Vec<u32>,
    /// `true` for nodes that lie on some pinned target's elimination-tree
    /// ancestor path. Reset by [`Self::reset_source`] per query. Sized to
    /// `node_count`.
    in_backward_search_space: Vec<bool>,
    /// Scratch stack used by [`Self::add_source_and_run`]'s backward sweep.
    /// Reused across queries; capacity grows monotonically.
    stack: Vec<u32>,
    /// Whether the current query has had a source added (used to keep the
    /// per-source ancestor cleanup tight).
    has_active_source: bool,
    /// CCH-internal node id of the most-recent source (so
    /// [`Self::reset_source`] can revisit just its ancestor path).
    active_source_node: u32,
    /// The corresponding `source_elimination_tree_end` (where the source
    /// ancestor walk first hit a marked node). [`INVALID_ID`] if it reached
    /// the root.
    active_source_end: u32,
}

impl<'a> ElimTreeQuery<'a> {
    /// Allocate scratch buffers for a query against `cch`.
    #[must_use]
    pub fn new(cch: &'a CchView<'a>) -> Self {
        let n = cch.node_count() as usize;
        Self {
            cch,
            forward_tentative_distance: vec![INF_WEIGHT; n],
            target_node: Vec::new(),
            target_elimination_tree_end: Vec::new(),
            in_backward_search_space: vec![false; n],
            stack: Vec::new(),
            has_active_source: false,
            active_source_node: INVALID_ID,
            active_source_end: INVALID_ID,
        }
    }

    /// Pin the target set. Call exactly once after construction and before
    /// issuing sources.
    ///
    /// # Hard precondition: single call per query
    ///
    /// This must NOT be called twice on the same query. The
    /// `in_backward_search_space` marks set during the ancestor walk below are
    /// never reset over the query's lifetime (neither here nor in
    /// [`Self::reset_source`]) — they define the static backward search-space
    /// topology that bounds every later walk. A second `pin_targets` call would
    /// see the first pin set's stale marks, terminate ancestor walks early, and
    /// record wrong `target_elimination_tree_end` values, silently corrupting
    /// all subsequent results. To pin a different target set, drop the query
    /// and create a new one. Enforced by a hard (non-debug) assertion.
    ///
    /// # Panics
    ///
    /// Panics if called more than once on the same query (i.e. when a target
    /// set has already been pinned).
    pub fn pin_targets(&mut self, targets: &[u32]) {
        assert!(
            self.target_node.is_empty(),
            "pin_targets called twice on the same query: in_backward_search_space \
             marks are never reset, so re-pinning would corrupt results; \
             create a fresh ElimTreeQuery instead"
        );

        self.target_node.reserve(targets.len());
        self.target_elimination_tree_end.reserve(targets.len());

        for &t_ext in targets {
            let t = self.cch.rank[t_ext as usize];
            self.target_node.push(t);

            let mut end = INVALID_ID;
            let mut x = t;
            while x != INVALID_ID {
                if self.in_backward_search_space[x as usize] {
                    end = x;
                    break;
                }
                self.in_backward_search_space[x as usize] = true;
                x = self.cch.elimination_tree_parent[x as usize];
            }
            self.target_elimination_tree_end.push(end);
        }
    }

    /// Reset per-source state so the next [`Self::add_source_and_run`] starts
    /// from a clean tentative-distance array. Walks just the previous source's
    /// ancestor path + the targets' ancestor paths to clear what could possibly
    /// hold non-`INF_WEIGHT` values. `O(elimination_depth)` per call.
    pub fn reset_source(&mut self) {
        // Clear previous source's ancestor path (where outgoing arcs were relaxed).
        if self.has_active_source {
            let mut x = self.active_source_node;
            while x != self.active_source_end {
                self.forward_tentative_distance[x as usize] = INF_WEIGHT;
                x = self.cch.elimination_tree_parent[x as usize];
            }
            self.has_active_source = false;
            self.active_source_node = INVALID_ID;
            self.active_source_end = INVALID_ID;
        }

        // Reset distances along each target's ancestor path so the backward
        // sweep starts clean. Mirrors routingkit's `reset_target_distances`.
        for i in (0..self.target_node.len()).rev() {
            let t = self.target_node[i];
            let end = self.target_elimination_tree_end[i];
            let mut x = t;
            while x != end {
                self.forward_tentative_distance[x as usize] = INF_WEIGHT;
                x = self.cch.elimination_tree_parent[x as usize];
            }
        }
    }

    /// Single-source variant: pin a source and run the elimination-tree sweep
    /// against the already-pinned targets. After this, the per-target distances
    /// are readable via [`Self::get_distances_to_targets`].
    pub fn add_source_and_run(&mut self, metric: &MetricView, source: u32) {
        let s = self.cch.rank[source as usize];

        // Walk source ancestors. Initial distance at s is 0; mark ancestors so
        // cleanup can revisit them.
        self.forward_tentative_distance[s as usize] = 0;
        self.active_source_node = s;
        self.has_active_source = true;
        let mut x = s;
        // We rely on the fact that elimination_tree_parent[x] > x; the walk
        // strictly ascends.
        loop {
            // Forward relax outgoing arcs from x.
            let from = self.cch.up_first_out[x as usize] as usize;
            let to = self.cch.up_first_out[x as usize + 1] as usize;
            let dx = self.forward_tentative_distance[x as usize];
            if dx != INF_WEIGHT {
                for xy in from..to {
                    let y = self.cch.up_head[xy] as usize;
                    let candidate = dx.saturating_add(metric.forward[xy]);
                    if candidate < self.forward_tentative_distance[y] {
                        self.forward_tentative_distance[y] = candidate;
                    }
                }
            }
            let parent = self.cch.elimination_tree_parent[x as usize];
            if parent == INVALID_ID {
                break;
            }
            x = parent;
        }
        // The source walks all the way to the root (INVALID_ID) for distance
        // queries; reset_source uses active_source_end to bound the cleanup walk.
        self.active_source_end = INVALID_ID;

        // Backward sweep over target ancestors in reverse topological order.
        // Push all target ancestors onto the stack (in walk order), then pop.
        self.stack.clear();
        for i in (0..self.target_node.len()).rev() {
            let t = self.target_node[i];
            let end = self.target_elimination_tree_end[i];
            let mut x = t;
            while x != end {
                self.stack.push(x);
                x = self.cch.elimination_tree_parent[x as usize];
            }
        }

        // Pop the stack, relax incoming arcs from each node.
        while let Some(x) = self.stack.pop() {
            let from = self.cch.up_first_out[x as usize] as usize;
            let to = self.cch.up_first_out[x as usize + 1] as usize;
            let mut best = self.forward_tentative_distance[x as usize];
            for xy in from..to {
                let y = self.cch.up_head[xy] as usize;
                let dy = self.forward_tentative_distance[y];
                if dy != INF_WEIGHT {
                    let candidate = dy.saturating_add(metric.backward[xy]);
                    if candidate < best {
                        best = candidate;
                    }
                }
            }
            self.forward_tentative_distance[x as usize] = best;
        }
    }

    /// Read out the per-target distance after [`Self::add_source_and_run`].
    /// `out.len()` must match the number of pinned targets.
    ///
    /// # Panics
    ///
    /// Panics if `out.len()` does not equal the pinned-target count.
    pub fn get_distances_to_targets(&self, out: &mut [u32]) {
        assert_eq!(
            out.len(),
            self.target_node.len(),
            "out buffer length must equal pinned-target count"
        );
        for (i, &t) in self.target_node.iter().enumerate() {
            out[i] = self.forward_tentative_distance[t as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// distance_matrix
// ---------------------------------------------------------------------------

/// Compute a full `sources × targets` distance matrix. Row-major; element
/// `[i * targets.len() + j]` is the distance from `sources[i]` to
/// `targets[j]`. [`crate::INF_WEIGHT`] means unreachable.
///
/// `sources` and `targets` are ORIGINAL dense node ids (the same id space the
/// oracle uses). This matches the semantics of routingkit's
/// `cch_compute_distance_matrix` except the unreachable sentinel: this
/// function returns [`crate::INF_WEIGHT`] (`= 2_147_483_647 = i32::MAX`)
/// whereas the C++ oracle returns `u32::MAX`. Callers comparing against the
/// oracle must normalize accordingly.
#[must_use]
pub fn distance_matrix(
    cch: &CchView,
    metric: &MetricView,
    sources: &[u32],
    targets: &[u32],
) -> Vec<u32> {
    if sources.is_empty() || targets.is_empty() {
        return Vec::new();
    }
    let mut q = ElimTreeQuery::new(cch);
    q.pin_targets(targets);
    let mut out = vec![0u32; sources.len() * targets.len()];
    let mut row = vec![0u32; targets.len()];
    for (i, &s) in sources.iter().enumerate() {
        q.reset_source();
        q.add_source_and_run(metric, s);
        q.get_distances_to_targets(&mut row);
        out[i * targets.len()..i * targets.len() + targets.len()].copy_from_slice(&row);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Differential correctness gate: compare our pure-Rust `distance_matrix`
    /// against the C++ oracle's `cch_compute_distance_matrix` for all pairs in
    /// a connected fixture graph.
    ///
    /// Sentinel normalization: our function emits `INF_WEIGHT`
    /// (= `2_147_483_647` = `i32::MAX`) for unreachable pairs. The oracle's
    /// `cch_compute_distance_matrix` emits the same `inf_weight` sentinel
    /// (`2_147_483_647`); some `RoutingKit` paths use `u32::MAX` instead, so we
    /// treat EITHER oracle sentinel as equivalent-unreachable. For all
    /// reachable positions the values must be identical. The fixture's node 5
    /// is isolated, so every ordered pair touching it (except 5→5) is
    /// unreachable, exercising this normalization.
    #[test]
    fn distance_matrix_matches_cpp_oracle() {
        use routingkit_cch::ffi;

        // Build a fixture with one isolated node so the sentinel-normalization
        // branch is actually exercised. Nodes 0-4 form a bidirectional path
        // 0-1-2-3-4; node 5 is isolated (no incident arcs), so every ordered
        // pair involving node 5 is unreachable → INF_WEIGHT on the Rust side,
        // u32::MAX on the oracle side. (Node 5 → node 5 has distance 0.)
        //
        // Arcs (in order), all among nodes 0-4:
        //   forward:  0→1, 1→2, 2→3, 3→4   (arcs 0-3)
        //   backward: 4→3, 3→2, 2→1, 1→0   (arcs 4-7)
        // Weights: arc index + 1, so weights = [1, 2, 3, 4, 5, 6, 7, 8].
        let n: u32 = 6;
        let tail: Vec<u32> = vec![0, 1, 2, 3, 4, 3, 2, 1];
        let head: Vec<u32> = vec![1, 2, 3, 4, 3, 2, 1, 0];
        let weights: Vec<u32> = (1..=8u32).collect();
        let order: Vec<u32> = (0..n).collect();

        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
        let cch_ref = cch.as_ref().expect("cch_new returned null");

        let dir = tempfile::tempdir().expect("tempdir");
        let struct_path = dir.path().join("dm.cch-struct");
        let metric_path = dir.path().join("dm.cch-metric");

        unsafe {
            ffi::cch_save_struct(cch_ref, struct_path.to_str().unwrap()).expect("cch_save_struct");
        }

        let oracle_matrix = unsafe {
            let mut metric = ffi::cch_metric_new(cch_ref, &weights);
            ffi::cch_metric_customize(metric.as_mut().expect("metric pin"));
            let sources: Vec<u32> = (0..n).collect();
            let targets: Vec<u32> = (0..n).collect();
            let matrix = ffi::cch_compute_distance_matrix(
                metric.as_ref().expect("metric ref"),
                &sources,
                &targets,
            );
            ffi::cch_save_metric(
                metric.as_ref().expect("metric ref"),
                metric_path.to_str().unwrap(),
            )
            .expect("cch_save_metric");
            matrix
        };

        // Re-open via our mmap readers and run our pure-Rust distance_matrix.
        let cch_bundle = crate::bundle::CchBundle::open(&struct_path).expect("CchBundle::open");
        let metric_bundle =
            crate::bundle::MetricBundle::open(&metric_path).expect("MetricBundle::open");
        let cch_view = cch_bundle.view();
        let metric_view = metric_bundle.view();

        let sources: Vec<u32> = (0..n).collect();
        let targets: Vec<u32> = (0..n).collect();
        let rust_matrix = distance_matrix(&cch_view, &metric_view, &sources, &targets);

        assert_eq!(
            oracle_matrix.len(),
            rust_matrix.len(),
            "matrix length mismatch"
        );

        for k in 0..oracle_matrix.len() {
            let oracle_val = oracle_matrix[k];
            let rust_val = rust_matrix[k];
            // Normalize sentinels: oracle uses u32::MAX or inf_weight, we use
            // INF_WEIGHT.
            let oracle_unreachable = oracle_val == u32::MAX || oracle_val == INF_WEIGHT;
            let rust_unreachable = rust_val == INF_WEIGHT;
            assert_eq!(
                oracle_unreachable,
                rust_unreachable,
                "reachability mismatch at index {k} (i={}, j={}): oracle={oracle_val}, rust={rust_val}",
                k / (n as usize),
                k % (n as usize),
            );
            if !oracle_unreachable && !rust_unreachable {
                assert_eq!(
                    oracle_val,
                    rust_val,
                    "distance mismatch at index {k} (i={}, j={}): oracle={oracle_val}, rust={rust_val}",
                    k / (n as usize),
                    k % (n as usize),
                );
            }
        }
    }
}

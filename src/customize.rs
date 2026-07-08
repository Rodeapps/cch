//! Per-metric CCH customization — a faithful port of the single-threaded
//! `CustomizableContractionHierarchyMetric::customize()` from `RoutingKit`
//! (`oracle/routingkit-cch/RoutingKit/src/customizable_contraction_hierarchy.cpp`).
//!
//! Given the [`Cch`] structure (built by [`Cch::build`], including the Part-A
//! input-arc → CCH-arc mapping) and a per-INPUT-arc `weights` slice, [`Metric`]
//! holds the customized `forward` / `backward` shortcut weights, BIT-IDENTICAL
//! to the C++ for the same graph + order + weights.
//!
//! The customization is two phases:
//! 1. **reset** (`extract_initial_metric*`, C++ 659–690): each CCH arc's
//!    forward/backward weight is initialized to the weight of its mapped input
//!    arc (or [`INF_WEIGHT`] if none), then min-combined with any parallel
//!    (extra) input arcs.
//! 2. **lower-triangle relaxation** (`customize()`, C++ 773–805): for each
//!    lower triangle `(bottom, mid, top)` in the enumeration order driven by
//!    `up_first_out`/`down_first_out`/`down_to_up`,
//!    `min_to(forward[top], backward[bottom] + forward[mid])` and
//!    `min_to(backward[top], forward[bottom] + backward[mid])`.

use crate::INF_WEIGHT;
use crate::bundle::INVALID_ID;
use crate::structure::Cch;
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    // Per-worker scratch for `relax_node`: maps node id -> the current node's
    // up-arc id to that node. Reused across levels and across customizations;
    // grown on demand. Never read stale (see `relax_node`'s chordal argument:
    // each node overwrites every slot it later reads).
    static ARC_ID_CACHE: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// A customized metric: the forward + backward shortcut weights of every CCH
/// arc. Field semantics match the persisted `.cch-metric` sections and
/// [`crate::bundle::MetricView`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metric {
    /// `forward[arc]` → shortcut weight in the up→down direction. Length =
    /// `cch_arc_count`. [`INF_WEIGHT`] marks an unreachable arc.
    pub forward: Vec<u32>,
    /// `backward[arc]` → shortcut weight in the down→up direction. Length =
    /// `cch_arc_count`.
    pub backward: Vec<u32>,
}

impl Metric {
    /// Borrow this in-memory metric as a [`MetricView`](crate::MetricView) — the
    /// query-ready view consumed by [`distance_matrix`](crate::distance_matrix)
    /// and [`node_path`](crate::node_path).
    #[must_use]
    pub fn view(&self) -> crate::bundle::MetricView<'_> {
        crate::bundle::MetricView {
            forward: &self.forward,
            backward: &self.backward,
        }
    }
}

/// Elimination-tree level partition of a [`Cch`], grouping nodes by height so
/// that same-level nodes have no ancestor/descendant relationship. Derived once
/// (metric-independent) and reused across customizations. `nodes` lists every
/// node id level-major (level 0 first); `first[l]..first[l+1]` slices `nodes`
/// for level `l`.
pub(crate) struct Levels {
    pub nodes: Vec<u32>,
    pub first: Vec<u32>,
}

/// Compute the elimination-tree height of every node and bucket nodes by height.
///
/// `height[x] = 0` for a node with no elim-tree children, else
/// `1 + max(height[child])`. Because `elimination_tree_parent[x]` always has a
/// strictly higher rank than `x`, iterating `x` in increasing rank finalizes
/// `height[x]` before it is read as a child (all children have lower rank).
pub(crate) fn compute_levels(cch: &Cch) -> Levels {
    let n = cch.node_count();
    let mut height = vec![0u32; n];
    let mut num_levels = 1u32; // at least level 0 (or 0 nodes -> unused)
    for (x, &p) in cch.elimination_tree_parent.iter().enumerate() {
        if p != INVALID_ID {
            let cand = height[x] + 1;
            if cand > height[p as usize] {
                height[p as usize] = cand;
            }
        }
        if height[x] + 1 > num_levels {
            num_levels = height[x] + 1;
        }
    }

    // CSR bucket by height.
    let mut first = vec![0u32; num_levels as usize + 1];
    for &h in &height {
        first[h as usize + 1] += 1;
    }
    for l in 0..num_levels as usize {
        first[l + 1] += first[l];
    }
    let mut cursor: Vec<u32> = first[..num_levels as usize].to_vec();
    let mut nodes = vec![0u32; n];
    #[allow(clippy::cast_possible_truncation)] // node ids fit u32 (CCH limit)
    for (x, &h) in height.iter().enumerate() {
        let l = h as usize;
        nodes[cursor[l] as usize] = x as u32;
        cursor[l] += 1;
    }

    Levels { nodes, first }
}

/// Saturating addition against [`INF_WEIGHT`].
///
/// Matches the C++ exactly: it adds raw `unsigned`s, but since
/// `inf_weight == 2^31 - 1`, two `inf_weight` summands give `2^32 - 2`, which
/// does NOT overflow `u32` and stays `> inf_weight`, so any later `min_to`
/// against a finite value keeps the finite value and unreachable stays
/// unreachable. We replicate that with a `u32` wrapping add: the largest
/// summand pair `2*(2^31-1) = 2^32 - 2` does not wrap, so `wrapping_add`
/// avoids Rust's debug overflow panic while producing the identical `u32`
/// result the C++ computes (and the subsequent `min` is identical).
#[inline]
fn add(a: u32, b: u32) -> u32 {
    // C++ computes `a + b` in `unsigned` (u32). The largest summand pair is
    // `inf_weight + inf_weight = 2^32 - 2`, which fits u32 with no wraparound,
    // so the wrapping add reproduces the C++ bit-for-bit.
    a.wrapping_add(b)
}

/// `min_to(x, y)`: set `x = min(x, y)`.
#[inline]
fn min_to(x: &mut u32, y: u32) {
    if y < *x {
        *x = y;
    }
}

/// Raw, `Send + Sync` pointers into the forward/backward arc-weight arrays for
/// the parallel relaxation. Sharing these across rayon tasks is data-race-free
/// ONLY for a well-formed (chordal) CCH — as produced by `Cch::build` or a
/// faithful `load_struct` round-trip — under the level-synchronized schedule in
/// `customize_into`:
///
/// * Each task processes one node `x`; every write targets an **up-arc of `x`**
///   (`top`), and up-arc id sets of distinct nodes are disjoint → no two tasks
///   in a level write the same slot.
/// * Reads (`forward[mid]`, `backward[bottom]`) are arcs of `x`'s down-neighbors,
///   which sit in strictly-lower elimination-tree levels — finalized before this
///   level runs and never written during it. A node in level `l` is never a
///   down-neighbor of another level-`l` node, so no same-level task reads `x`'s
///   arcs either.
///
/// This disjointness argument depends on chordality; a bounds-valid but
/// non-chordal `Cch` is out of scope (see `Cch::customizer`). Every index used
/// is bounds-validated once in `Cch::customizer`, so `len` is an upper bound on
/// all `top`/`mid`/`bottom` accesses regardless.
struct DisjointArcs {
    forward: *mut u32,
    backward: *mut u32,
    len: usize,
}
// SAFETY: see the disjointness/finalization argument above; the sole caller
// (`customize_into`'s level loop) upholds it.
unsafe impl Send for DisjointArcs {}
unsafe impl Sync for DisjointArcs {}

/// Relax every lower triangle apexed at node `x`, writing the resulting
/// candidates into `x`'s up-arc weights. `cache` is per-worker scratch reused
/// across the nodes a worker handles; `x`'s first inner loop overwrites every
/// slot it later reads (`z` is always an up-neighbor of `x` in a chordal CCH),
/// so stale entries from a previously handled node are never read.
///
/// # Safety
/// `cch` must be a well-formed (chordal) `Cch` — as produced by `Cch::build` or
/// a faithful `load_struct` round-trip — that has also passed
/// `Cch::customizer`'s bounds validation, so that `arcs` (derived from that
/// same `cch`) satisfies the `DisjointArcs` contract. `x` must be
/// `< node_count`.
unsafe fn relax_node(x: usize, cache: &mut [u32], arcs: &DisjointArcs, cch: &Cch) {
    for xz_up in cch.up_first_out[x]..cch.up_first_out[x + 1] {
        cache[cch.up_head[xz_up as usize] as usize] = xz_up;
    }
    for xy_down in cch.down_first_out[x]..cch.down_first_out[x + 1] {
        let bottom = cch.down_to_up[xy_down as usize] as usize;
        let y = cch.down_head[xy_down as usize] as usize;
        let y_up_begin = cch.up_first_out[y];
        let mut cursor = cch.up_first_out[y + 1];
        while cursor > y_up_begin {
            cursor -= 1;
            let mid = cursor as usize;
            let z = cch.up_head[mid] as usize;
            if z <= x {
                break;
            }
            let top = cache[z] as usize;
            debug_assert!(top < arcs.len && mid < arcs.len && bottom < arcs.len);
            // SAFETY: bottom/mid are arcs of the lower node `y` (finalized,
            // read-only this level); top is an up-arc of `x` (written only by
            // this task). All three indices are `< arcs.len` (validated).
            let bwd_bottom = unsafe { *arcs.backward.add(bottom) };
            let fwd_mid = unsafe { *arcs.forward.add(mid) };
            let fwd_bottom = unsafe { *arcs.forward.add(bottom) };
            let bwd_mid = unsafe { *arcs.backward.add(mid) };
            let fwd_candidate = add(bwd_bottom, fwd_mid);
            let bwd_candidate = add(fwd_bottom, bwd_mid);
            // SAFETY: `top` is exclusive to this task (see `DisjointArcs`).
            let fwd_top = unsafe { &mut *arcs.forward.add(top) };
            min_to(fwd_top, fwd_candidate);
            let bwd_top = unsafe { &mut *arcs.backward.add(top) };
            min_to(bwd_top, bwd_candidate);
        }
    }
}

impl Cch {
    /// Build a reusable [`Customizer`] for this structure. Derives the
    /// elimination-tree level partition once; reuse the returned `Customizer`
    /// across many metrics to avoid recomputing it and to reuse output buffers
    /// via [`Customizer::customize_into`].
    ///
    /// # Panics
    /// Panics if `self` is structurally malformed: a `up_head`/`down_head`
    /// entry names a node id `>= node_count`, a `down_to_up` entry names an
    /// arc id `>= cch_arc_count`, or `up_first_out[node_count] !=
    /// cch_arc_count`. These asserts guard the parallel relaxation's
    /// raw-pointer accesses against out-of-bounds indexing for any
    /// bounds-valid `Cch`; they do NOT establish chordality. The parallel
    /// relaxation's correctness and data-race-freedom additionally require
    /// `self` to be well-formed (chordal), which every `Cch` from
    /// [`Cch::build`] or a faithful `load_struct` round-trip is. A
    /// deliberately corrupted, bounds-valid but non-chordal `Cch` is outside
    /// this safety contract.
    #[must_use]
    pub fn customizer(&self) -> Customizer<'_> {
        // Bounds precondition for the parallel relaxation's raw-pointer
        // `.add()` accesses (see `relax_node`): every head is a valid node id,
        // every arc reference is a valid arc id, and the up-adjacency
        // terminates at `cch_arc_count`. This rules out out-of-bounds access
        // for any bounds-valid `Cch`; it does not establish chordality, which
        // `relax_node`'s data-race-freedom additionally requires (see
        // `DisjointArcs`) and which every `Cch` from `build`/`load_struct`
        // satisfies.
        let n = self.node_count();
        let arc_count = self.cch_arc_count();
        assert!(
            self.up_head.iter().all(|&h| (h as usize) < n)
                && self.down_head.iter().all(|&h| (h as usize) < n),
            "malformed Cch: head contains a node id >= node_count"
        );
        assert!(
            self.down_to_up.iter().all(|&a| (a as usize) < arc_count),
            "malformed Cch: down_to_up contains an arc id >= cch_arc_count"
        );
        assert_eq!(
            self.up_first_out.get(n).copied(),
            u32::try_from(arc_count).ok(),
            "malformed Cch: up_first_out[node_count] != cch_arc_count"
        );
        Customizer {
            cch: self,
            levels: compute_levels(self),
        }
    }

    /// Customizes this CCH with per-INPUT-arc `weights`, producing the forward
    /// and backward shortcut weights of every CCH arc.
    ///
    /// Bit-identical to `RoutingKit`'s `customize()`. Delegates to
    /// [`Cch::customizer`], which re-validates structure and recomputes the
    /// elimination-tree level partition on every call. For repeated
    /// customization of the same structure, hold a [`Customizer`] (via
    /// [`Cch::customizer`]) and reuse it with [`Customizer::customize`] /
    /// [`Customizer::customize_into`] instead, to avoid re-validating,
    /// re-partitioning, and re-allocating output buffers each call.
    ///
    /// # Panics
    /// Panics if `weights.len()` != the number of input arcs
    /// (`self.input_arc_to_cch_arc.len()`), or if `self` is structurally
    /// malformed (see [`Cch::customizer`]'s panics).
    #[must_use]
    pub fn customize(&self, weights: &[u32]) -> Metric {
        self.customizer().customize(weights)
    }
}

/// Reusable customizer for one [`Cch`]. Owns the metric-independent
/// elimination-tree level partition so repeated customizations do not recompute
/// it, and lets callers reuse output buffers via [`Self::customize_into`].
pub struct Customizer<'a> {
    cch: &'a Cch,
    levels: Levels,
}

impl Customizer<'_> {
    /// Customize `weights` into a freshly allocated [`Metric`].
    ///
    /// # Panics
    /// Panics if `weights.len()` != the number of input arcs.
    #[must_use]
    pub fn customize(&self, weights: &[u32]) -> Metric {
        let arc_count = self.cch.cch_arc_count();
        let mut out = Metric {
            forward: vec![INF_WEIGHT; arc_count],
            backward: vec![INF_WEIGHT; arc_count],
        };
        self.customize_into(weights, &mut out);
        out
    }

    /// Customize `weights`, reusing `out`'s `forward`/`backward` allocations.
    /// On return, `out` holds the customized metric for `weights`. No allocation
    /// occurs when `out`'s buffers already have `cch_arc_count` capacity.
    ///
    /// # Panics
    /// Panics if `weights.len()` != the number of input arcs.
    pub fn customize_into(&self, weights: &[u32], out: &mut Metric) {
        let cch = self.cch;
        assert_eq!(
            weights.len(),
            cch.input_arc_to_cch_arc.len(),
            "weights length must equal input arc count"
        );
        let arc_count = cch.cch_arc_count();

        // Reuse buffers: resize to arc_count and reset every slot to INF_WEIGHT.
        out.forward.clear();
        out.forward.resize(arc_count, INF_WEIGHT);
        out.backward.clear();
        out.backward.resize(arc_count, INF_WEIGHT);
        let forward = &mut out.forward;
        let backward = &mut out.backward;

        // Phase 1: reset (extract_initial_metric) — arcs are independent.
        forward
            .par_iter_mut()
            .zip(backward.par_iter_mut())
            .enumerate()
            .for_each(|(cch_arc, (fwd, bwd))| {
                let fwd_in = cch.forward_input_arc_of_cch[cch_arc];
                if fwd_in != INVALID_ID {
                    *fwd = weights[fwd_in as usize];
                }
                let bwd_in = cch.backward_input_arc_of_cch[cch_arc];
                if bwd_in != INVALID_ID {
                    *bwd = weights[bwd_in as usize];
                }
                let ef = &cch.first_extra_forward_input_arc_of_cch;
                for j in ef[cch_arc]..ef[cch_arc + 1] {
                    let ia = cch.extra_forward_input_arc_of_cch[j as usize] as usize;
                    min_to(fwd, weights[ia]);
                }
                let eb = &cch.first_extra_backward_input_arc_of_cch;
                for j in eb[cch_arc]..eb[cch_arc + 1] {
                    let ia = cch.extra_backward_input_arc_of_cch[j as usize] as usize;
                    min_to(bwd, weights[ia]);
                }
            });

        // Phase 2: lower-triangle relaxation, level-synchronized parallelism.
        // Levels run sequentially (barrier between); within a level, nodes run
        // in parallel with provably-disjoint writes (see `DisjointArcs`).
        let node_count = cch.node_count();
        let arcs = DisjointArcs {
            forward: forward.as_mut_ptr(),
            backward: backward.as_mut_ptr(),
            len: arc_count,
        };
        let levels = &self.levels;
        for l in 0..levels.first.len() - 1 {
            let level = &levels.nodes[levels.first[l] as usize..levels.first[l + 1] as usize];
            level.par_iter().for_each(|&x| {
                ARC_ID_CACHE.with(|c| {
                    let mut cache = c.borrow_mut();
                    if cache.len() < node_count {
                        cache.resize(node_count, 0);
                    }
                    // SAFETY: `arcs` upholds the `DisjointArcs` contract under this
                    // level-synchronized schedule; `cch` passed `customizer`
                    // validation; `x < node_count` (it is a node id from `levels`).
                    unsafe { relax_node(x as usize, &mut cache, &arcs, cch) };
                });
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    // Independent serial reference for the relaxation, retained ONLY for tests so
    // the parallel path is checked against a non-parallel, non-oracle baseline.
    // This is a faithful copy of the pre-parallel Phase-2 loop.
    fn relax_serial(cch: &Cch, weights: &[u32]) -> Metric {
        let arc_count = cch.cch_arc_count();
        let mut forward = vec![INF_WEIGHT; arc_count];
        let mut backward = vec![INF_WEIGHT; arc_count];
        for cch_arc in 0..arc_count {
            let fwd_in = cch.forward_input_arc_of_cch[cch_arc];
            if fwd_in != INVALID_ID {
                forward[cch_arc] = weights[fwd_in as usize];
            }
            let bwd_in = cch.backward_input_arc_of_cch[cch_arc];
            if bwd_in != INVALID_ID {
                backward[cch_arc] = weights[bwd_in as usize];
            }
            let ef = &cch.first_extra_forward_input_arc_of_cch;
            for j in ef[cch_arc]..ef[cch_arc + 1] {
                min_to(
                    &mut forward[cch_arc],
                    weights[cch.extra_forward_input_arc_of_cch[j as usize] as usize],
                );
            }
            let eb = &cch.first_extra_backward_input_arc_of_cch;
            for j in eb[cch_arc]..eb[cch_arc + 1] {
                min_to(
                    &mut backward[cch_arc],
                    weights[cch.extra_backward_input_arc_of_cch[j as usize] as usize],
                );
            }
        }
        let node_count = cch.node_count();
        let mut cache = vec![0u32; node_count];
        for x in 0..node_count {
            for xz_up in cch.up_first_out[x]..cch.up_first_out[x + 1] {
                cache[cch.up_head[xz_up as usize] as usize] = xz_up;
            }
            for xy_down in cch.down_first_out[x]..cch.down_first_out[x + 1] {
                let bottom = cch.down_to_up[xy_down as usize] as usize;
                let y = cch.down_head[xy_down as usize] as usize;
                let y_up_begin = cch.up_first_out[y];
                let mut cursor = cch.up_first_out[y + 1];
                while cursor > y_up_begin {
                    cursor -= 1;
                    let mid = cursor as usize;
                    let z = cch.up_head[mid] as usize;
                    if z <= x {
                        break;
                    }
                    let top = cache[z] as usize;
                    let fc = add(backward[bottom], forward[mid]);
                    let bc = add(forward[bottom], backward[mid]);
                    min_to(&mut forward[top], fc);
                    min_to(&mut backward[top], bc);
                }
            }
        }
        Metric { forward, backward }
    }

    #[test]
    fn metric_view_borrows_fields() {
        let m = Metric {
            forward: vec![1, 2, 3],
            backward: vec![4, 5, 6],
        };
        let v = m.view();
        assert_eq!(v.forward, &[1, 2, 3]);
        assert_eq!(v.backward, &[4, 5, 6]);
    }

    /// Build a CSR `Graph` from grouped-by-tail arc lists.
    fn csr(node_count: usize, tail: &[u32], head: &[u32]) -> Graph {
        let mut counts = vec![0u32; node_count];
        for &t in tail {
            counts[t as usize] += 1;
        }
        let mut first_out = vec![0u32; node_count + 1];
        for v in 0..node_count {
            first_out[v + 1] = first_out[v] + counts[v];
        }
        let mut next: Vec<usize> = first_out[..node_count]
            .iter()
            .map(|&x| x as usize)
            .collect();
        let mut g_head = vec![0u32; head.len()];
        for (&t, &h) in tail.iter().zip(head.iter()) {
            g_head[next[t as usize]] = h;
            next[t as usize] += 1;
        }
        Graph {
            first_out,
            head: g_head,
            weight: vec![1u32; head.len()],
        }
    }

    // Single arc 0->1: forward[arc]=weight, backward[arc]=INF (no reverse arc).
    #[test]
    fn single_arc() {
        let g = csr(2, &[0], &[1]);
        let order = vec![0u32, 1];
        let c = Cch::build(&g, &order);
        let m = c.customize(&[42]);
        assert_eq!(m.forward, vec![42]);
        assert_eq!(m.backward, vec![INF_WEIGHT]);
    }

    // A single bidirectional edge: forward and backward both set.
    #[test]
    fn bidirectional_arc() {
        let g = csr(2, &[0, 1], &[1, 0]);
        let order = vec![0u32, 1];
        let c = Cch::build(&g, &order);
        let m = c.customize(&[7, 9]);
        // up-arc 0->1: forward from 0->1 (w=7), backward from 1->0 (w=9).
        assert_eq!(m.forward, vec![7]);
        assert_eq!(m.backward, vec![9]);
    }

    // Parallel arcs with different weights: min wins (extra-arc path).
    #[test]
    fn parallel_arc_min() {
        // two 0->1 arcs (w=50, 9) and two 1->0 arcs (w=40, 8).
        let g = csr(2, &[0, 0, 1, 1], &[1, 1, 0, 0]);
        let order = vec![0u32, 1];
        let c = Cch::build(&g, &order);
        let m = c.customize(&[50, 9, 40, 8]);
        assert_eq!(m.forward, vec![9]);
        assert_eq!(m.backward, vec![8]);
    }

    // All-INF weights stay INF through the (saturating) relaxation.
    #[test]
    fn all_inf() {
        // fill-in graph so the relaxation runs.
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let c = Cch::build(&g, &order);
        let inf = INF_WEIGHT;
        let m = c.customize(&[inf, inf, inf, inf]);
        assert!(m.forward.iter().all(|&w| w == inf));
        assert!(m.backward.iter().all(|&w| w == inf));
    }

    // Triangle fill-in: contracting 0 (neighbors 1,2) creates shortcut 1->2.
    // Its weight is computed by the lower-triangle relaxation.
    #[test]
    fn triangle_relaxation() {
        // path 1-0-2 (undirected), order [0,1,2]: shortcut up-arc 1->2.
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let c = Cch::build(&g, &order);
        // weights: 0->1=3, 0->2=5, 1->0=4, 2->0=6.
        let m = c.customize(&[3, 5, 4, 6]);
        // up-arcs sorted: 0->1 (idx0), 0->2 (idx1), 1->2 (idx2, shortcut).
        // forward[1->2] via triangle bottom=(1->0 backward of arc0=... )
        //   = backward[0->1] + forward[0->2] = 4 + 5 = 9.
        // backward[1->2] = forward[0->1] + backward[0->2] = 3 + 6 = 9.
        assert_eq!(m.forward[2], 9);
        assert_eq!(m.backward[2], 9);
    }

    // add() helper: saturating-style behaviour for inf summands.
    #[test]
    fn add_inf_helper() {
        assert_eq!(add(1, 2), 3);
        // two infs sum to 2^32-2, which is > INF_WEIGHT (stays unreachable).
        let s = add(INF_WEIGHT, INF_WEIGHT);
        assert!(s > INF_WEIGHT);
    }

    // Reset must min-combine parallel/extra input arcs identically under rayon.
    // Two 0->1 arcs (w=50,9) and two 1->0 arcs (w=40,8): min wins per direction.
    #[test]
    fn parallel_reset_min_combines() {
        let g = csr(2, &[0, 0, 1, 1], &[1, 1, 0, 0]);
        let order = vec![0u32, 1];
        let c = Cch::build(&g, &order);
        let m = c.customizer().customize(&[50, 9, 40, 8]);
        assert_eq!(m.forward, vec![9]);
        assert_eq!(m.backward, vec![8]);
    }

    // customize panics on wrong weight length.
    #[test]
    #[should_panic(expected = "weights length must equal input arc count")]
    fn wrong_weight_len_panics() {
        let g = csr(2, &[0], &[1]);
        let order = vec![0u32, 1];
        let c = Cch::build(&g, &order);
        let _ = c.customize(&[1, 2, 3]);
    }

    // customizer() validates structural soundness before handing out raw
    // pointers to the parallel relaxation. Cover each assertion branch with a
    // hand-corrupted Cch (as would result from loading arbitrary/corrupt bytes).

    #[test]
    #[should_panic(expected = "malformed Cch: head contains a node id >= node_count")]
    #[allow(clippy::cast_possible_truncation)] // tiny fixture: node_count fits u32
    fn customizer_rejects_out_of_range_up_head() {
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let mut c = Cch::build(&g, &order);
        let n = c.node_count();
        c.up_head[0] = n as u32; // one past the last valid node id
        let _ = c.customizer();
    }

    #[test]
    #[should_panic(expected = "malformed Cch: head contains a node id >= node_count")]
    #[allow(clippy::cast_possible_truncation)] // tiny fixture: node_count fits u32
    fn customizer_rejects_out_of_range_down_head() {
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let mut c = Cch::build(&g, &order);
        let n = c.node_count();
        c.down_head[0] = n as u32;
        let _ = c.customizer();
    }

    #[test]
    #[should_panic(expected = "malformed Cch: down_to_up contains an arc id >= cch_arc_count")]
    #[allow(clippy::cast_possible_truncation)] // tiny fixture: arc_count fits u32
    fn customizer_rejects_out_of_range_down_to_up() {
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let mut c = Cch::build(&g, &order);
        let arc_count = c.cch_arc_count();
        c.down_to_up[0] = arc_count as u32;
        let _ = c.customizer();
    }

    #[test]
    #[should_panic(expected = "malformed Cch: up_first_out[node_count] != cch_arc_count")]
    fn customizer_rejects_bad_up_first_out_tail() {
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let mut c = Cch::build(&g, &order);
        let last = c.up_first_out.len() - 1;
        c.up_first_out[last] += 1;
        let _ = c.customizer();
    }

    // customize_into into an empty Metric matches a fresh customize.
    #[test]
    fn customize_into_empty_matches_fresh() {
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let c = Cch::build(&g, &order);
        let w = [3u32, 5, 4, 6];

        let fresh = c.customize(&w);
        let cust = c.customizer();
        let mut out = Metric {
            forward: Vec::new(),
            backward: Vec::new(),
        };
        cust.customize_into(&w, &mut out);
        assert_eq!(out, fresh);
    }

    // customize_into into an over-sized Metric overwrites cleanly (no stale tail).
    #[test]
    fn customize_into_oversized_matches_fresh() {
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let c = Cch::build(&g, &order);
        let w = [3u32, 5, 4, 6];

        let fresh = c.customize(&w);
        let cust = c.customizer();
        let mut out = Metric {
            forward: vec![999; fresh.forward.len() + 5],
            backward: vec![999; fresh.backward.len() + 5],
        };
        cust.customize_into(&w, &mut out);
        assert_eq!(out, fresh);
    }

    // One Customizer reused across two different weight vectors gives the same
    // results as two independent customize calls (no scratch bleed).
    #[test]
    fn customizer_reuse_no_bleed() {
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let c = Cch::build(&g, &order);
        let w1 = [3u32, 5, 4, 6];
        let w2 = [10u32, 1, 2, 20];

        let cust = c.customizer();
        let mut out = c.customize(&w1); // seed with anything
        cust.customize_into(&w1, &mut out);
        assert_eq!(out, c.customize(&w1));
        cust.customize_into(&w2, &mut out);
        assert_eq!(out, c.customize(&w2));
    }

    // A node's level is 1 + its elimination-tree parent-chain depth from a leaf:
    // level(x) = 0 if x has no elim-tree descendant, else max(level(child))+1.
    // Every node appears exactly once; a node's level must exceed every
    // down-neighbor's level (down-neighbors are strictly-lower elim-tree nodes).
    #[test]
    fn levels_partition_is_valid() {
        // path 1-0-2 (undirected), order [0,1,2]: elim tree 0->1->2 (root 2).
        let g = csr(3, &[0, 0, 1, 2], &[1, 2, 0, 0]);
        let order = vec![0u32, 1, 2];
        let c = Cch::build(&g, &order);
        let levels = compute_levels(&c);

        // every node present exactly once
        let mut seen = levels.nodes.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2]);

        // first[] is a valid CSR offset array over nodes
        assert_eq!(*levels.first.first().unwrap(), 0);
        assert_eq!(*levels.first.last().unwrap(), 3);

        // per-node level lookup
        let mut level_of = vec![0u32; c.node_count()];
        #[allow(clippy::cast_possible_truncation)] // levels count is tiny (3) in this test
        for l in 0..levels.first.len() - 1 {
            for &x in &levels.nodes[levels.first[l] as usize..levels.first[l + 1] as usize] {
                level_of[x as usize] = l as u32;
            }
        }
        // each down-neighbor is strictly lower level than its node
        for x in 0..c.node_count() {
            for d in c.down_first_out[x]..c.down_first_out[x + 1] {
                let y = c.down_head[d as usize] as usize;
                assert!(
                    level_of[y] < level_of[x],
                    "down-neighbor must be lower level"
                );
            }
        }
    }

    // Build a 5x5 bidirectional grid so the elimination tree has several levels,
    // then assert the parallel relaxation equals an independent serial reference
    // arc-for-arc, on multiple random-ish weight vectors.
    fn grid(side: u32) -> Graph {
        let n = side * side;
        let mut tail = Vec::new();
        let mut head = Vec::new();
        let idx = |r: u32, col: u32| r * side + col;
        for r in 0..side {
            for col in 0..side {
                if col + 1 < side {
                    tail.push(idx(r, col));
                    head.push(idx(r, col + 1));
                    tail.push(idx(r, col + 1));
                    head.push(idx(r, col));
                }
                if r + 1 < side {
                    tail.push(idx(r, col));
                    head.push(idx(r + 1, col));
                    tail.push(idx(r + 1, col));
                    head.push(idx(r, col));
                }
            }
        }
        csr(n as usize, &tail, &head)
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)] // 5x5 grid: arc/node counts are tiny
    fn parallel_relax_equals_serial_reference() {
        let g = grid(5);
        let order: Vec<u32> = (0..(g.first_out.len() as u32 - 1)).collect();
        let c = Cch::build(&g, &order);
        let input_arcs = c.input_arc_to_cch_arc.len();

        for seed in [1u32, 7, 13, 99] {
            let weights: Vec<u32> = (0..input_arcs as u32)
                .map(|i| 1 + (i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 97))
                .collect();
            let parallel = c.customize(&weights);
            let reference = relax_serial(&c, &weights);
            assert_eq!(parallel.forward, reference.forward, "seed {seed} forward");
            assert_eq!(
                parallel.backward, reference.backward,
                "seed {seed} backward"
            );
        }
    }

    // Determinism: the parallel path yields byte-identical output across runs.
    #[test]
    #[allow(clippy::cast_possible_truncation)] // 5x5 grid: arc/node counts are tiny
    fn parallel_relax_is_deterministic() {
        let g = grid(5);
        let order: Vec<u32> = (0..(g.first_out.len() as u32 - 1)).collect();
        let c = Cch::build(&g, &order);
        let weights: Vec<u32> = (0..c.input_arc_to_cch_arc.len() as u32)
            .map(|i| 1 + i % 50)
            .collect();
        let a = c.customize(&weights);
        let b = c.customize(&weights);
        assert_eq!(a, b);
    }

    // relax_serial's extra-forward/extra-backward min-combine branches (the
    // reset half of the reference) only trigger when an up-arc has parallel
    // input arcs; exercise that here so the serial reference itself is fully
    // covered, matching the parallel path on the same fixture.
    #[test]
    fn relax_serial_matches_parallel_with_parallel_arcs() {
        let g = csr(2, &[0, 0, 1, 1], &[1, 1, 0, 0]);
        let order = vec![0u32, 1];
        let c = Cch::build(&g, &order);
        let weights = [50u32, 9, 40, 8];
        let parallel = c.customize(&weights);
        let reference = relax_serial(&c, &weights);
        assert_eq!(parallel.forward, reference.forward);
        assert_eq!(parallel.backward, reference.backward);
    }
}

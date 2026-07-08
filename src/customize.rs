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
    // Consumed by the parallel relaxation added in a later task; `Customizer`
    // already computes and stores a `Levels` (see `Cch::customizer`), but
    // nothing walks `nodes`/`first` yet outside tests.
    #[allow(dead_code)]
    pub nodes: Vec<u32>,
    #[allow(dead_code)]
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

impl Cch {
    /// Build a reusable [`Customizer`] for this structure. Derives the
    /// elimination-tree level partition once; reuse the returned `Customizer`
    /// across many metrics to avoid recomputing it and to reuse output buffers
    /// via [`Customizer::customize_into`].
    #[must_use]
    pub fn customizer(&self) -> Customizer<'_> {
        Customizer {
            cch: self,
            levels: compute_levels(self),
        }
    }

    /// Customizes this CCH with per-INPUT-arc `weights`, producing the forward
    /// and backward shortcut weights of every CCH arc.
    ///
    /// Bit-identical to `RoutingKit`'s `customize()`. For repeated customization
    /// of the same structure, prefer [`Cch::customizer`] +
    /// [`Customizer::customize_into`] to avoid re-allocating output buffers.
    ///
    /// # Panics
    /// Panics if `weights.len()` != the number of input arcs
    /// (`self.input_arc_to_cch_arc.len()`).
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
    // Consumed by the parallel relaxation added in a later task; retained now
    // so callers don't recompute it across repeated `customize_into` calls.
    #[allow(dead_code)]
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

        // Phase 2: lower-triangle relaxation (serial; parallelized in a later task).
        let node_count = cch.node_count();
        let mut arc_id_cache = vec![0u32; node_count];
        for x in 0..node_count {
            for xz_up in cch.up_first_out[x]..cch.up_first_out[x + 1] {
                arc_id_cache[cch.up_head[xz_up as usize] as usize] = xz_up;
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
                    let top = arc_id_cache[z] as usize;
                    let fwd_candidate = add(backward[bottom], forward[mid]);
                    let bwd_candidate = add(forward[bottom], backward[mid]);
                    min_to(&mut forward[top], fwd_candidate);
                    min_to(&mut backward[top], bwd_candidate);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

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
}

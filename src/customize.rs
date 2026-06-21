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

/// Saturating addition against [`INF_WEIGHT`].
///
/// Matches the C++ exactly: it adds raw `unsigned`s, but since
/// `inf_weight == 2^31 - 1`, two `inf_weight` summands give `2^32 - 2`, which
/// does NOT overflow `u32` and stays `> inf_weight`, so any later `min_to`
/// against a finite value keeps the finite value and unreachable stays
/// unreachable. We replicate that with a plain wrapping/saturating-free add via
/// `u64` to avoid Rust's debug overflow panic while producing the identical
/// `u32` result the C++ computes (no value the C++ produces can exceed
/// `2*(2^31-1) = 2^32 - 2 < u64::MAX`, and the subsequent `min` is identical).
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
    /// Customizes this CCH with per-INPUT-arc `weights`, producing the forward
    /// and backward shortcut weights of every CCH arc.
    ///
    /// Bit-identical to `RoutingKit`'s single-threaded
    /// `CustomizableContractionHierarchyMetric::customize()`.
    ///
    /// # Panics
    /// Panics if `weights.len()` does not equal the number of input arcs (i.e.
    /// `self.input_arc_to_cch_arc.len()`).
    #[must_use]
    pub fn customize(&self, weights: &[u32]) -> Metric {
        assert_eq!(
            weights.len(),
            self.input_arc_to_cch_arc.len(),
            "weights length must equal input arc count"
        );

        let arc_count = self.cch_arc_count();
        let mut forward = vec![INF_WEIGHT; arc_count];
        let mut backward = vec![INF_WEIGHT; arc_count];

        // Phase 1: reset (extract_initial_metric, C++ 659–690).
        for cch_arc in 0..arc_count {
            let fwd_in = self.forward_input_arc_of_cch[cch_arc];
            if fwd_in != INVALID_ID {
                forward[cch_arc] = weights[fwd_in as usize];
            }
            let bwd_in = self.backward_input_arc_of_cch[cch_arc];
            if bwd_in != INVALID_ID {
                backward[cch_arc] = weights[bwd_in as usize];
            }
            // Parallel-arc minimum over the extra (overflow) lists.
            let ef = &self.first_extra_forward_input_arc_of_cch;
            for j in ef[cch_arc]..ef[cch_arc + 1] {
                let ia = self.extra_forward_input_arc_of_cch[j as usize] as usize;
                min_to(&mut forward[cch_arc], weights[ia]);
            }
            let eb = &self.first_extra_backward_input_arc_of_cch;
            for j in eb[cch_arc]..eb[cch_arc + 1] {
                let ia = self.extra_backward_input_arc_of_cch[j as usize] as usize;
                min_to(&mut backward[cch_arc], weights[ia]);
            }
        }

        // Phase 2: lower-triangle relaxation (C++ 778–798).
        let node_count = self.node_count();
        let mut arc_id_cache = vec![0u32; node_count];
        for x in 0..node_count {
            let xz_up_end = self.up_first_out[x + 1];
            for xz_up in self.up_first_out[x]..xz_up_end {
                arc_id_cache[self.up_head[xz_up as usize] as usize] = xz_up;
            }

            let xy_down_end = self.down_first_out[x + 1];
            for xy_down in self.down_first_out[x]..xy_down_end {
                // `bottom` = the y→x up-arc; `y` = the lower triangle apex node.
                let bottom = self.down_to_up[xy_down as usize] as usize;
                let y = self.down_head[xy_down as usize] as usize;
                let y_up_begin = self.up_first_out[y];
                let mut cursor = self.up_first_out[y + 1];
                while cursor > y_up_begin {
                    cursor -= 1;
                    // `mid` = the y→z up-arc; `z` = the triangle top node.
                    let mid = cursor as usize;
                    let z = self.up_head[mid] as usize;
                    if z <= x {
                        break;
                    }
                    let top = arc_id_cache[z] as usize;
                    // min_to(forward[top],  backward[bottom] + forward[mid])
                    let fwd_candidate = add(backward[bottom], forward[mid]);
                    // min_to(backward[top], forward[bottom]  + backward[mid])
                    let bwd_candidate = add(forward[bottom], backward[mid]);
                    min_to(&mut forward[top], fwd_candidate);
                    min_to(&mut backward[top], bwd_candidate);
                }
            }
        }

        Metric { forward, backward }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

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

    // customize panics on wrong weight length.
    #[test]
    #[should_panic(expected = "weights length must equal input arc count")]
    fn wrong_weight_len_panics() {
        let g = csr(2, &[0], &[1]);
        let order = vec![0u32, 1];
        let c = Cch::build(&g, &order);
        let _ = c.customize(&[1, 2, 3]);
    }
}

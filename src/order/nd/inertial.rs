// Inertial-flow nested dissection: the 4-axis `inertial_flow` cutter, the
// recursive separator decomposition, and the top-level order assembly.
//
// Ported from `RoutingKit/src/nested_dissection.cpp`:
//   - `inertial_flow(g, min_balance, lat, lon)`        (.cpp:528)
//   - `inertial_flow(g, lat, lon)` (default wrapper)   (.cpp:646)
//   - `compute_separator_decomposition`                (.cpp:800)
//   - `compute_nested_node_dissection_order`           (.cpp:904)
//   - `compute_nested_node_dissection_order_using_inertial_flow` (.cpp:911)
//
// All logging / stats / timer hooks (`g_total_*`, `log_message`,
// `get_micro_time`) are investigation-only and omitted entirely.

use crate::internal::bitvec::BitVector;
use crate::order::nd::flow::BlockingFlow;
use crate::order::nd::fragment::{
    CutSide, GraphFragment, decompose_graph_fragment_into_connected_components,
    derive_separator_from_cut, make_graph_fragment, pick_smaller_side, select_source_and_target,
};

// ──────────────────────────────────────────────────────────────────────────
// inertial_flow  (nested_dissection.cpp:528)
// ──────────────────────────────────────────────────────────────────────────

/// Run a 4-axis inertial-flow cut on `fragment`, returning the balanced cut of
/// the cutter that finishes with the smallest flow intensity.
///
/// Builds four [`BlockingFlow`] cutters from `select_source_and_target` on four
/// `f32` projections of the per-node coordinates (indexed via
/// `fragment.global_node_id`): latitude (horizontal), longitude (vertical),
/// `lat+lon` (main diagonal) and `lat-lon` (next diagonal). It then repeatedly
/// advances whichever cutter currently has the smallest flow intensity (with
/// the exact C++ tie-break order horizontal ≤ vertical ≤ `main_diag` ≤ `next_diag`);
/// once that smallest-flow cutter is finished it returns its balanced cut.
///
/// `side_size = node_count * min_balance / 100`, clamped to a minimum of 1.
///
/// Ported from `nested_dissection.cpp:528` (logging/stats/timer hooks omitted).
#[must_use]
pub(crate) fn inertial_flow(
    fragment: &GraphFragment,
    min_balance: u32,
    latitude: &[f32],
    longitude: &[f32],
) -> CutSide {
    let node_count = fragment.node_count();

    let mut side_size = node_count.saturating_mul(min_balance) / 100;
    if side_size == 0 {
        side_size = 1;
    }

    let gid = &fragment.global_node_id;

    let horizontal_st = select_source_and_target(side_size, node_count, |x| {
        latitude[gid[x as usize] as usize]
    });
    let mut horizontal_cutter =
        BlockingFlow::new(fragment, horizontal_st.is_source, horizontal_st.is_target);

    let vertical_st = select_source_and_target(side_size, node_count, |x| {
        longitude[gid[x as usize] as usize]
    });
    let mut vertical_cutter =
        BlockingFlow::new(fragment, vertical_st.is_source, vertical_st.is_target);

    let main_diagonal_st = select_source_and_target(side_size, node_count, |x| {
        latitude[gid[x as usize] as usize] + longitude[gid[x as usize] as usize]
    });
    let mut main_diagonal_cutter = BlockingFlow::new(
        fragment,
        main_diagonal_st.is_source,
        main_diagonal_st.is_target,
    );

    let next_diagonal_st = select_source_and_target(side_size, node_count, |x| {
        latitude[gid[x as usize] as usize] - longitude[gid[x as usize] as usize]
    });
    let mut next_diagonal_cutter = BlockingFlow::new(
        fragment,
        next_diagonal_st.is_source,
        next_diagonal_st.is_target,
    );

    loop {
        // Pick the cutter with the smallest current flow intensity, with the
        // exact C++ tie-break order: horizontal ≤ vertical ≤ main_diag ≤ next_diag.
        let h = horizontal_cutter.get_current_flow_intensity();
        let v = vertical_cutter.get_current_flow_intensity();
        let m = main_diagonal_cutter.get_current_flow_intensity();
        let n = next_diagonal_cutter.get_current_flow_intensity();

        let chosen = if h <= v && h <= m && h <= n {
            &mut horizontal_cutter
        } else if v <= m && v <= n {
            &mut vertical_cutter
        } else if m <= n {
            &mut main_diagonal_cutter
        } else {
            &mut next_diagonal_cutter
        };

        if chosen.is_finished() {
            return chosen.get_balanced_cut();
        }
        chosen.advance();
    }
}

// ──────────────────────────────────────────────────────────────────────────
// inertial_flow default-min_balance wrapper  (nested_dissection.cpp:646)
// ──────────────────────────────────────────────────────────────────────────

/// Run `inertial_flow` at imbalances 25 %, 33 % and 40 % and return the cut with
/// the best (lowest) `cut_size / node_on_side_count` ratio.
///
/// The C++ compares ratios via cross-multiplication to avoid floating point:
/// `c25` wins iff it strictly beats both `c33` and `c40`; else `c33` wins iff it
/// strictly beats `c40`; else `c40`.
///
/// Ported from `nested_dissection.cpp:646`.
#[must_use]
pub(crate) fn inertial_flow_default(
    fragment: &GraphFragment,
    latitude: &[f32],
    longitude: &[f32],
) -> CutSide {
    let c25 = inertial_flow(fragment, 25, latitude, longitude);
    let c33 = inertial_flow(fragment, 33, latitude, longitude);
    let c40 = inertial_flow(fragment, 40, latitude, longitude);

    // ratio(a) < ratio(b)  ⇔  a.cut_size * b.node_on_side_count < b.cut_size * a.node_on_side_count
    let cross = |a: &CutSide, b: &CutSide| -> bool {
        u64::from(a.cut_size) * u64::from(b.node_on_side_count)
            < u64::from(b.cut_size) * u64::from(a.node_on_side_count)
    };

    if cross(&c25, &c33) && cross(&c25, &c40) {
        c25
    } else if cross(&c33, &c40) {
        c33
    } else {
        c40
    }
}

// ──────────────────────────────────────────────────────────────────────────
// compute_separator_decomposition  (nested_dissection.cpp:800)
// ──────────────────────────────────────────────────────────────────────────

/// Recursive nested-dissection order assembly.
///
/// Returns the contraction order (`SeparatorDecomposition::order` in the C++):
/// a permutation of `fragment.global_node_id` in which separator vertices are
/// ranked highest (placed at the high end of the order) and the recursion fills
/// the lower ranks from the resulting components. `compute_separator` maps a
/// fragment to the bit-set of its separator nodes (local indices).
///
/// We only need `.order`; the C++ `tree` bookkeeping is not load-bearing for the
/// order, so it is omitted. Ported from `nested_dissection.cpp:800`.
fn compute_separator_decomposition_order(
    fragment: GraphFragment,
    compute_separator: &dyn Fn(&GraphFragment) -> BitVector,
) -> Vec<u32> {
    let node_count = fragment.node_count() as usize;
    let mut order = vec![0u32; node_count];

    if node_count == 1 {
        order[0] = fragment.global_node_id[0];
        return order;
    }

    let mut order_begin = 0usize;
    let mut order_end = node_count;

    let part_list = decompose_graph_fragment_into_connected_components(fragment);

    for mut part in part_list {
        debug_assert_ne!(part.node_count(), 0);
        if part.node_count() == 1 {
            order_end -= 1;
            order[order_end] = part.global_node_id[0];
        } else {
            let is_separator_node = compute_separator(&part);

            // Keep only the arcs that touch NO separator node; separator nodes
            // become isolated, so the recursion peels them off as singleton
            // components and ranks them at the high end.
            let keep = |a: usize| -> bool {
                !is_separator_node.is_set(u64::from(part.tail[a]))
                    && !is_separator_node.is_set(u64::from(part.head[a]))
            };
            let arc_count = part.arc_count() as usize;
            let mut kept: Vec<usize> = Vec::with_capacity(arc_count);
            for a in 0..arc_count {
                if keep(a) {
                    kept.push(a);
                }
            }

            // `local[old_arc] = new_arc` for kept arcs (used to remap back_arc).
            // Both endpoints of a kept arc are non-separator, so its back_arc is
            // also kept — the remap is always defined.
            let mut local = vec![0u32; arc_count];
            for (new_arc, &old_arc) in kept.iter().enumerate() {
                local[old_arc] = u32::try_from(new_arc).expect("arc index fits u32");
            }

            let new_tail: Vec<u32> = kept.iter().map(|&a| part.tail[a]).collect();
            let new_head: Vec<u32> = kept.iter().map(|&a| part.head[a]).collect();
            let new_back_arc: Vec<u32> = kept
                .iter()
                .map(|&a| local[part.back_arc[a] as usize])
                .collect();

            part.tail = new_tail;
            part.head = new_head;
            part.back_arc = new_back_arc;
            part.first_out = invert_vector(&part.tail, part.node_count());

            let sub_order = compute_separator_decomposition_order(part, compute_separator);
            order[order_begin..order_begin + sub_order.len()].copy_from_slice(&sub_order);
            order_begin += sub_order.len();
        }
    }

    order
}

/// Build CSR `first_out` from a tail array sorted ascending (`invert_vector` in
/// the C++): `first_out[i]` is the index of the first arc whose tail is `i`.
fn invert_vector(tail: &[u32], element_count: u32) -> Vec<u32> {
    let nc = element_count as usize;
    let mut index = vec![0u32; nc + 1];
    let mut pos = 0usize;
    for (i, slot) in index.iter_mut().enumerate().take(nc) {
        while pos < tail.len() && (tail[pos] as usize) < i {
            pos += 1;
        }
        *slot = u32::try_from(pos).expect("arc index fits u32");
    }
    index[nc] = u32::try_from(tail.len()).expect("arc count fits u32");
    index
}

// ──────────────────────────────────────────────────────────────────────────
// compute_nested_node_dissection_order_using_inertial_flow  (cpp:904 / :911)
// ──────────────────────────────────────────────────────────────────────────

/// Compute an inertial-flow nested-dissection contraction order.
///
/// Builds a [`GraphFragment`], then runs the recursive separator decomposition
/// whose separator oracle is `derive_separator_from_cut` applied to the
/// smaller side of the default 4-axis `inertial_flow` cut.
///
/// Ported from `compute_nested_node_dissection_order_using_inertial_flow`
/// (`nested_dissection.cpp:911`) composed with `compute_nested_node_dissection_order`
/// (`.cpp:904`).
#[must_use]
pub(crate) fn compute_nested_node_dissection_order_using_inertial_flow(
    node_count: u32,
    tail: &[u32],
    head: &[u32],
    latitude: &[f32],
    longitude: &[f32],
) -> Vec<u32> {
    let fragment = make_graph_fragment(node_count, tail, head);

    let compute_separator = |frag: &GraphFragment| -> BitVector {
        let mut cut = inertial_flow_default(frag, latitude, longitude);
        pick_smaller_side(&mut cut);
        derive_separator_from_cut(frag, &cut.is_node_on_side)
    };

    compute_separator_decomposition_order(fragment, &compute_separator)
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "small grid coordinates fit f32 exactly in tests"
)]
mod tests {
    use super::*;

    /// Count undirected edges crossing `side` (each edge once).
    fn count_crossing(edges: &[(u32, u32)], side: &BitVector) -> u32 {
        let mut c = 0u32;
        for &(u, v) in edges {
            if side.is_set(u64::from(u)) != side.is_set(u64::from(v)) {
                c += 1;
            }
        }
        c
    }

    fn fragment_from_edges(node_count: u32, edges: &[(u32, u32)]) -> GraphFragment {
        let tail: Vec<u32> = edges.iter().map(|&(u, _)| u).collect();
        let head: Vec<u32> = edges.iter().map(|&(_, v)| v).collect();
        make_graph_fragment(node_count, &tail, &head)
    }

    /// `inertial_flow` on a tiny coord graph returns a valid cut whose crossing
    /// count equals its reported `cut_size`, with all coordinates honored.
    #[test]
    fn inertial_flow_returns_valid_cut() {
        // 2x3 grid laid out by coords. Nodes id = r*3 + c, lat=r, lon=c.
        let edges = [(0u32, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)];
        let frag = fragment_from_edges(6, &edges);
        let lat = [0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let lon = [0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0];

        let cut = inertial_flow(&frag, 25, &lat, &lon);
        assert_eq!(
            count_crossing(&edges, &cut.is_node_on_side),
            cut.cut_size,
            "cut_size must equal the number of crossing edges"
        );
        assert_eq!(
            cut.is_node_on_side.population_count(),
            u64::from(cut.node_on_side_count),
            "node_on_side_count must equal population count"
        );
        // A balanced cut on a 6-node graph keeps at least one node on the side.
        assert!(cut.node_on_side_count >= 1);
        assert!(cut.node_on_side_count < 6);
    }

    /// The default wrapper also returns a valid cut.
    #[test]
    fn inertial_flow_default_returns_valid_cut() {
        let edges = [(0u32, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)];
        let frag = fragment_from_edges(6, &edges);
        let lat = [0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let lon = [0.0f32, 1.0, 2.0, 0.0, 1.0, 2.0];
        let cut = inertial_flow_default(&frag, &lat, &lon);
        assert_eq!(count_crossing(&edges, &cut.is_node_on_side), cut.cut_size);
        assert_eq!(
            cut.is_node_on_side.population_count(),
            u64::from(cut.node_on_side_count)
        );
    }

    /// A separator derived from the inertial cut actually disconnects the two
    /// sides: removing it leaves no edge between the small side and its
    /// complement.
    #[test]
    fn inertial_flow_yields_balanced_separator() {
        // 3x3 grid.
        let mut edges = Vec::new();
        for r in 0u32..3 {
            for c in 0u32..3 {
                let id = r * 3 + c;
                if c + 1 < 3 {
                    edges.push((id, id + 1));
                }
                if r + 1 < 3 {
                    edges.push((id, id + 3));
                }
            }
        }
        let frag = fragment_from_edges(9, &edges);
        let lat: Vec<f32> = (0..9).map(|i| (i / 3) as f32).collect();
        let lon: Vec<f32> = (0..9).map(|i| (i % 3) as f32).collect();

        let mut cut = inertial_flow_default(&frag, &lat, &lon);
        pick_smaller_side(&mut cut);
        let sep = derive_separator_from_cut(&frag, &cut.is_node_on_side);
        // Separator must be non-empty for a connected graph with a real cut.
        assert!(sep.population_count() >= 1, "separator must be non-empty");

        // Removing the separator: no edge crosses between the small side and the
        // rest, among non-separator nodes.
        let small_side = &cut.is_node_on_side;
        for &(u, v) in &edges {
            let u_sep = sep.is_set(u64::from(u));
            let v_sep = sep.is_set(u64::from(v));
            if !u_sep && !v_sep {
                assert_eq!(
                    small_side.is_set(u64::from(u)),
                    small_side.is_set(u64::from(v)),
                    "edge {u}-{v} crosses the cut but neither endpoint is in the separator"
                );
            }
        }
    }

    /// An empty separator (no node is a separator). Used by the edge-case
    /// recursion tests where the singleton fast-path means the oracle is never
    /// actually consulted; defined as a named fn so its single body is covered.
    fn empty_separator(f: &GraphFragment) -> BitVector {
        BitVector::new(u64::from(f.node_count()))
    }

    /// Recursion edge case: a single-node fragment yields its own id.
    #[test]
    fn decomposition_single_node() {
        // Cover the oracle body directly: the single-node path never calls it.
        assert_eq!(
            empty_separator(&make_graph_fragment(1, &[], &[])).population_count(),
            0
        );
        let frag = make_graph_fragment(1, &[], &[]);
        let order = compute_separator_decomposition_order(frag, &empty_separator);
        assert_eq!(order, vec![0u32]);
    }

    /// Recursion edge case: a fully-disconnected fragment (no arcs) — each node
    /// is its own singleton component; the order is a permutation of all node
    /// ids. The separator oracle is never invoked (every component is a singleton).
    #[test]
    fn decomposition_disconnected_isolated_nodes() {
        let frag = make_graph_fragment(4, &[], &[]);
        let order = compute_separator_decomposition_order(frag, &empty_separator);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0u32, 1, 2, 3]);
    }

    /// Recursion on a two-component fragment: each component is ordered, and the
    /// union covers all node ids.
    #[test]
    fn decomposition_two_components() {
        // Two disjoint edges {0-1} and {2-3}.
        let frag = fragment_from_edges(4, &[(0, 1), (2, 3)]);
        let lat = [0.0f32, 0.0, 5.0, 5.0];
        let lon = [0.0f32, 1.0, 0.0, 1.0];
        let compute_separator = |f: &GraphFragment| -> BitVector {
            let mut cut = inertial_flow_default(f, &lat, &lon);
            pick_smaller_side(&mut cut);
            derive_separator_from_cut(f, &cut.is_node_on_side)
        };
        let order = compute_separator_decomposition_order(frag, &compute_separator);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0u32, 1, 2, 3]);
    }

    /// `inertial_flow_default`: a fixture where the 33 %-imbalance cut has the
    /// best `cut_size / node_on_side_count` ratio, so the `c33` branch is taken.
    /// (Found by random search over small geometric graphs.)
    #[test]
    fn inertial_flow_default_selects_c33() {
        let tail = [
            0u32, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 2, 6, 5, 2, 1, 7, 4, 2, 5, 6, 2, 6, 5,
            3, 2, 0, 1, 1, 3,
        ];
        let head = [
            1u32, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 2, 7, 5, 6, 1, 2, 4, 7, 5, 2, 2, 6, 5, 6,
            2, 3, 1, 0, 3, 1,
        ];
        let lat = [53.0f32, 92.0, 45.0, 95.0, 3.0, 55.0, 87.0, 49.0];
        let lon = [20.0f32, 55.0, 64.0, 58.0, 1.0, 86.0, 56.0, 96.0];
        let frag = make_graph_fragment(8, &tail, &head);

        let c33 = inertial_flow(&frag, 33, &lat, &lon);
        let chosen = inertial_flow_default(&frag, &lat, &lon);
        // The chosen cut equals the 33 % cut (same size + side count).
        assert_eq!(chosen.cut_size, c33.cut_size);
        assert_eq!(chosen.node_on_side_count, c33.node_on_side_count);
    }

    /// `inertial_flow_default`: a fixture where the 25 %-imbalance cut has the
    /// best ratio, so the `c25` branch is taken.
    #[test]
    fn inertial_flow_default_selects_c25() {
        let tail = [
            0u32, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 4, 5, 9, 2, 4, 7, 4, 5, 4, 6,
            9, 7, 0, 9, 0, 9, 0, 4, 2, 3, 5, 6, 6, 0, 0, 1, 0, 7, 1, 3,
        ];
        let head = [
            1u32, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8, 5, 4, 2, 9, 7, 4, 5, 4, 6, 4,
            7, 9, 9, 0, 9, 0, 4, 0, 3, 2, 6, 5, 0, 6, 1, 0, 7, 0, 3, 1,
        ];
        let lat = [91.0f32, 44.0, 25.0, 2.0, 16.0, 34.0, 4.0, 6.0, 67.0, 26.0];
        let lon = [
            26.0f32, 18.0, 17.0, 92.0, 47.0, 73.0, 96.0, 89.0, 31.0, 36.0,
        ];
        let frag = make_graph_fragment(10, &tail, &head);

        let c25 = inertial_flow(&frag, 25, &lat, &lon);
        let chosen = inertial_flow_default(&frag, &lat, &lon);
        assert_eq!(chosen.cut_size, c25.cut_size);
        assert_eq!(chosen.node_on_side_count, c25.node_on_side_count);
    }

    /// `invert_vector` matches the CSR semantics on an empty and a basic case.
    #[test]
    fn invert_vector_cases() {
        assert_eq!(invert_vector(&[], 3), vec![0u32, 0, 0, 0]);
        // tails sorted: node 0 has 2 arcs, node 1 has 1, node 2 has 0.
        assert_eq!(invert_vector(&[0, 0, 1], 3), vec![0u32, 2, 3, 3]);
    }

    /// Top-level inertial order on a small grid is a valid permutation.
    #[test]
    fn inertial_order_is_permutation_small_grid() {
        let rows = 4u32;
        let cols = 4u32;
        let n = rows * cols;
        let mut tail = Vec::new();
        let mut head = Vec::new();
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                lat.push(r as f32);
                lon.push(c as f32);
                if c + 1 < cols {
                    tail.push(v);
                    head.push(v + 1);
                    tail.push(v + 1);
                    head.push(v);
                }
                if r + 1 < rows {
                    tail.push(v);
                    head.push(v + cols);
                    tail.push(v + cols);
                    head.push(v);
                }
            }
        }
        let order =
            compute_nested_node_dissection_order_using_inertial_flow(n, &tail, &head, &lat, &lon);
        assert_eq!(order.len(), n as usize);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        let expected: Vec<u32> = (0..n).collect();
        assert_eq!(sorted, expected, "inertial order must be a permutation");
    }

    /// Determinism: identical input → identical order on repeated runs.
    #[test]
    fn inertial_order_is_deterministic() {
        let rows = 5u32;
        let cols = 5u32;
        let n = rows * cols;
        let mut tail = Vec::new();
        let mut head = Vec::new();
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                lat.push(r as f32);
                lon.push(c as f32);
                if c + 1 < cols {
                    tail.push(v);
                    head.push(v + 1);
                    tail.push(v + 1);
                    head.push(v);
                }
                if r + 1 < rows {
                    tail.push(v);
                    head.push(v + cols);
                    tail.push(v + cols);
                    head.push(v);
                }
            }
        }
        let a =
            compute_nested_node_dissection_order_using_inertial_flow(n, &tail, &head, &lat, &lon);
        let b =
            compute_nested_node_dissection_order_using_inertial_flow(n, &tail, &head, &lat, &lon);
        assert_eq!(a, b, "inertial order must be deterministic");
    }
}

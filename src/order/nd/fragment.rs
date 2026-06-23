// Inertial nested-dissection internals: graph fragment, component decomposition,
// source/target selection, cut helpers.
//
// Ported from `RoutingKit/src/nested_dissection.cpp` and
// `RoutingKit/include/routingkit/nested_dissection.h`.
// All items are `pub(crate)` — not part of the public API yet.

// ──────────────────────────────────────────────────────────────────────────
// Data structures
// ──────────────────────────────────────────────────────────────────────────

/// A symmetric (undirected) CSR graph fragment.
///
/// Each undirected edge is represented as two directed arcs that are each
/// other's `back_arc`.  Mirrors `struct GraphFragment` from
/// `nested_dissection.h:14`.
#[derive(Debug, Clone)]
pub(crate) struct GraphFragment {
    /// Maps local node index → original global node id.
    pub(crate) global_node_id: Vec<u32>,
    /// CSR offsets: arcs of local node `v` are `tail[first_out[v]..first_out[v+1]]`.
    pub(crate) first_out: Vec<u32>,
    /// Tail (source) of each arc, in CSR order.
    pub(crate) tail: Vec<u32>,
    /// Head (target) of each arc, in CSR order.
    pub(crate) head: Vec<u32>,
    /// `back_arc[a]` is the reverse arc of `a`; `back_arc[back_arc[a]] == a`.
    pub(crate) back_arc: Vec<u32>,
}

impl GraphFragment {
    /// Number of local nodes.
    #[must_use]
    #[inline]
    pub(crate) fn node_count(&self) -> u32 {
        u32::try_from(self.global_node_id.len()).expect("node count fits u32")
    }

    /// Number of directed arcs (2× the number of undirected edges).
    #[must_use]
    #[inline]
    pub(crate) fn arc_count(&self) -> u32 {
        u32::try_from(self.tail.len()).expect("arc count fits u32")
    }
}

/// A 2-side cut result, as produced by a max-flow computation.
///
/// Mirrors `struct CutSide` from `nested_dissection.h:34`.
pub(crate) struct CutSide {
    /// Number of local nodes on this side.
    pub(crate) node_on_side_count: u32,
    /// Number of arcs in the cut (== flow intensity).
    pub(crate) cut_size: u32,
    /// Bit-vector indicating which local nodes are on this side.
    pub(crate) is_node_on_side: crate::internal::bitvec::BitVector,
}

/// Return value of [`select_source_and_target`].
pub(crate) struct SourceTargetResult {
    pub(crate) is_source: crate::internal::bitvec::BitVector,
    pub(crate) is_target: crate::internal::bitvec::BitVector,
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

/// Sort arcs by (tail, head) and return the sort permutation `p` such that
/// `p[new_position] = old_arc_index`.  The caller must apply `p` to all
/// arc-indexed vectors.
fn arc_sort_permutation(tail: &[u32], head: &[u32]) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..u32::try_from(tail.len()).expect("fits u32")).collect();
    indices.sort_by(|&a, &b| {
        tail[a as usize]
            .cmp(&tail[b as usize])
            .then(head[a as usize].cmp(&head[b as usize]))
    });
    indices
}

/// Apply permutation `p` (as `result[i] = v[p[i]]`) to a `u32` slice.
fn apply_perm(p: &[u32], v: &[u32]) -> Vec<u32> {
    p.iter().map(|&i| v[i as usize]).collect()
}

/// Remap the *elements* of `v` through the inverse of `p`.
///
/// Each element `e` in `v` is replaced by `inv_p[e]` where `inv_p` is the
/// inverse permutation of `p`.  Used to keep `back_arc` consistent after
/// reordering arcs.
fn remap_elements_by_inverse(p: &[u32], v: &mut [u32]) {
    let mut inv_p = vec![0u32; p.len()];
    for (new_pos, &old_idx) in p.iter().enumerate() {
        inv_p[old_idx as usize] = u32::try_from(new_pos).expect("fits u32");
    }
    for e in v.iter_mut() {
        *e = inv_p[*e as usize];
    }
}

/// Build CSR `first_out` from a sorted `tail` array and a node count.
fn build_first_out(tail: &[u32], node_count: u32) -> Vec<u32> {
    let nc = node_count as usize;
    let mut first_out = vec![0u32; nc + 1];
    for &t in tail {
        first_out[t as usize + 1] += 1;
    }
    for i in 1..=nc {
        first_out[i] += first_out[i - 1];
    }
    first_out
}

// ──────────────────────────────────────────────────────────────────────────
// make_graph_fragment  (nested_dissection.cpp:42)
// ──────────────────────────────────────────────────────────────────────────

/// Build a symmetric fragment from a directed arc list.
///
/// Self-loops are silently discarded.  Each surviving input arc `(u, v)`
/// produces two directed arcs `u→v` and `v→u` with `back_arc[back_arc[a]]==a`.
/// Ported from `nested_dissection.cpp:42`.
#[must_use]
pub(crate) fn make_graph_fragment(node_count: u32, tail: &[u32], head: &[u32]) -> GraphFragment {
    assert_eq!(
        tail.len(),
        head.len(),
        "tail and head must have equal length"
    );

    let arc_count = tail.len();

    let non_loop_arc_count: usize = tail
        .iter()
        .zip(head.iter())
        .filter(|&(&t, &h)| t != h)
        .count();

    let sym_arc_count = 2 * non_loop_arc_count;
    let mut sym_tail = vec![0u32; sym_arc_count];
    let mut sym_head = vec![0u32; sym_arc_count];
    let mut back_arc = vec![0u32; sym_arc_count];

    {
        let mut j = 0usize;
        for i in 0..arc_count {
            if tail[i] != head[i] {
                sym_tail[j] = tail[i];
                sym_tail[j + non_loop_arc_count] = head[i];
                sym_head[j] = head[i];
                sym_head[j + non_loop_arc_count] = tail[i];
                back_arc[j] = u32::try_from(j + non_loop_arc_count).expect("fits u32");
                back_arc[j + non_loop_arc_count] = u32::try_from(j).expect("fits u32");
                j += 1;
            }
        }
    }

    // Sort arcs by (tail, head).
    let p = arc_sort_permutation(&sym_tail, &sym_head);
    let sorted_tail = apply_perm(&p, &sym_tail);
    let sorted_head = apply_perm(&p, &sym_head);
    let mut sorted_back_arc = apply_perm(&p, &back_arc);
    remap_elements_by_inverse(&p, &mut sorted_back_arc);

    let first_out = build_first_out(&sorted_tail, node_count);
    let global_node_id: Vec<u32> = (0..node_count).collect();

    GraphFragment {
        global_node_id,
        first_out,
        tail: sorted_tail,
        head: sorted_head,
        back_arc: sorted_back_arc,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// pick_smaller_side  (nested_dissection.cpp:520)
// ──────────────────────────────────────────────────────────────────────────

/// If the marked side has ≥ half the nodes, flip to the complement (smaller) side.
///
/// Ported from `nested_dissection.cpp:520`.
pub(crate) fn pick_smaller_side(c: &mut CutSide) {
    let node_count = c.is_node_on_side.len();
    // C++: if(c.node_on_side_count >= (node_count+1)/2)
    let threshold = node_count.div_ceil(2);
    if u64::from(c.node_on_side_count) >= threshold {
        c.node_on_side_count =
            u32::try_from(node_count).expect("node count fits u32") - c.node_on_side_count;
        c.is_node_on_side.inplace_not();
    }
}

// ──────────────────────────────────────────────────────────────────────────
// decompose_graph_fragment_into_connected_components  (cpp:672)
// ──────────────────────────────────────────────────────────────────────────

/// Assign component ids via DFS and compute the inv-pseudo-preorder permutation.
///
/// Returns `(component, inv_pseudo_preorder, component_count)`.
fn assign_components(fragment: &GraphFragment) -> (Vec<u32>, Vec<u32>) {
    let node_count = fragment.node_count() as usize;
    let invalid = u32::MAX;
    let mut component_count = 0u32;
    let mut component = vec![invalid; node_count];
    let mut inv_pseudo_preorder = vec![0u32; node_count];
    let mut stack = vec![0u32; node_count];
    let mut pos = 0u32;

    for r in 0..u32::try_from(node_count).expect("node count fits u32") {
        if component[r as usize] == invalid {
            component[r as usize] = component_count;
            let mut stack_end = 1usize;
            stack[0] = r;
            while stack_end != 0 {
                stack_end -= 1;
                let x = stack[stack_end];
                inv_pseudo_preorder[x as usize] = pos;
                pos += 1;
                let from = fragment.first_out[x as usize] as usize;
                let to = fragment.first_out[x as usize + 1] as usize;
                for xy in from..to {
                    let y = fragment.head[xy];
                    if component[y as usize] == invalid {
                        stack[stack_end] = y;
                        stack_end += 1;
                        component[y as usize] = component_count;
                    }
                }
            }
            component_count += 1;
        }
    }
    (component, inv_pseudo_preorder)
}

/// Extract a single component as a sub-fragment.
fn extract_component(
    fragment: &GraphFragment,
    component_node_begin: usize,
    component_node_end: usize,
    component_arc_begin: usize,
    component_arc_end: usize,
) -> GraphFragment {
    let part_node_count = component_node_end - component_node_begin;

    let part_tail: Vec<u32> = fragment.tail[component_arc_begin..component_arc_end]
        .iter()
        .map(|&t| t - u32::try_from(component_node_begin).expect("fits u32"))
        .collect();

    let part_head: Vec<u32> = fragment.head[component_arc_begin..component_arc_end]
        .iter()
        .map(|&h| h - u32::try_from(component_node_begin).expect("fits u32"))
        .collect();

    let part_back_arc: Vec<u32> = fragment.back_arc[component_arc_begin..component_arc_end]
        .iter()
        .map(|&b| b - u32::try_from(component_arc_begin).expect("fits u32"))
        .collect();

    let part_global_node_id: Vec<u32> =
        fragment.global_node_id[component_node_begin..component_node_end].to_vec();

    let part_first_out = build_first_out(
        &part_tail,
        u32::try_from(part_node_count).expect("fits u32"),
    );

    GraphFragment {
        global_node_id: part_global_node_id,
        first_out: part_first_out,
        tail: part_tail,
        head: part_head,
        back_arc: part_back_arc,
    }
}

/// Decompose a fragment into its connected components.
///
/// Ported from `nested_dissection.cpp:672`.
#[must_use]
pub(crate) fn decompose_graph_fragment_into_connected_components(
    mut fragment: GraphFragment,
) -> Vec<GraphFragment> {
    let node_count = fragment.node_count() as usize;
    let arc_count = fragment.arc_count() as usize;

    let (mut component, inv_pseudo_preorder) = assign_components(&fragment);

    // Apply inv_pseudo_preorder to node ids within arc arrays.
    for t in &mut fragment.tail {
        *t = inv_pseudo_preorder[*t as usize];
    }
    for h in &mut fragment.head {
        *h = inv_pseudo_preorder[*h as usize];
    }
    // apply_inverse_permutation(inv_pseudo_preorder, global_node_id)
    {
        let old = fragment.global_node_id.clone();
        for (i, g) in fragment.global_node_id.iter_mut().enumerate() {
            *g = old[inv_pseudo_preorder[i] as usize];
        }
    }
    // apply_inverse_permutation(inv_pseudo_preorder, component)
    {
        let old_comp = component.clone();
        for (i, c) in component.iter_mut().enumerate() {
            *c = old_comp[inv_pseudo_preorder[i] as usize];
        }
    }

    // Re-sort arcs by (tail, head).
    let p = arc_sort_permutation(&fragment.tail, &fragment.head);
    fragment.head = apply_perm(&p, &fragment.head);
    fragment.tail = apply_perm(&p, &fragment.tail);
    let mut new_back_arc = apply_perm(&p, &fragment.back_arc);
    remap_elements_by_inverse(&p, &mut new_back_arc);
    fragment.back_arc = new_back_arc;
    fragment.first_out =
        build_first_out(&fragment.tail, u32::try_from(node_count).expect("fits u32"));

    // Extract per-component sub-fragments.
    let mut part_list: Vec<GraphFragment> = Vec::new();
    let mut component_node_begin = 0usize;
    let mut component_arc_begin = 0usize;
    let mut component_node_end = 1usize;
    let mut component_arc_end = 0usize;
    let mut current_component = 0u32;

    while component_node_end < node_count {
        if component[component_node_end] != current_component {
            while component_arc_end < arc_count
                && component[fragment.tail[component_arc_end] as usize] == current_component
            {
                component_arc_end += 1;
            }
            part_list.push(extract_component(
                &fragment,
                component_node_begin,
                component_node_end,
                component_arc_begin,
                component_arc_end,
            ));
            component_node_begin = component_node_end;
            component_arc_begin = component_arc_end;
            current_component = component[component_node_end];
        }
        component_node_end += 1;
    }
    part_list.push(extract_component(
        &fragment,
        component_node_begin,
        component_node_end,
        component_arc_begin,
        arc_count,
    ));

    part_list
}

// ──────────────────────────────────────────────────────────────────────────
// select_source_and_target  (nested_dissection.cpp:487)
// ──────────────────────────────────────────────────────────────────────────

/// Non-generic selection core: partitions `v[0..nc]` so that `v[0..nn]` are the
/// `nn` smallest keys and `v[nc-nn..nc]` are the `nn` largest.
///
/// Uses `&dyn Fn` (dynamic dispatch) to avoid duplicate instantiation per
/// closure type; called only when `nn > 0`.
fn select_nth_smallest_and_largest(v: &mut [u32], nn: usize, sort_key: &dyn Fn(u32) -> f32) {
    let nc = v.len();
    v[..nc].select_nth_unstable_by(nn - 1, |&l, &r| {
        sort_key(l)
            .partial_cmp(&sort_key(r))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v[nn..nc].select_nth_unstable_by(nc - nn - nn, |&l, &r| {
        sort_key(l)
            .partial_cmp(&sort_key(r))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Select the `n` lowest-key and `n` highest-key nodes as sources and targets.
///
/// Ported from `nested_dissection.cpp:487` (file-static template).
/// Uses the same comparator (`sort_key(l) < sort_key(r)`) as the C++
/// `std::nth_element` calls.  The `f32` arithmetic and ordering match the C++
/// exactly, so the selection is deterministic within the crate.
///
/// # Panics
/// Panics if `n > node_count / 2`.
#[must_use]
pub(crate) fn select_source_and_target<F>(
    n: u32,
    node_count: u32,
    sort_key: F,
) -> SourceTargetResult
where
    F: Fn(u32) -> f32,
{
    assert!(n <= node_count / 2, "n must not exceed node_count/2");

    let nc = node_count as usize;
    let nn = n as usize;

    let mut v: Vec<u32> = (0..node_count).collect();

    // Mirror C++ two nth_element calls.
    if nn > 0 {
        select_nth_smallest_and_largest(&mut v, nn, &sort_key);
    }

    let mut is_source = crate::internal::bitvec::BitVector::new(u64::from(node_count));
    let mut is_target = crate::internal::bitvec::BitVector::new(u64::from(node_count));

    for &node in &v[..nn] {
        is_source.set(u64::from(node));
    }
    for &node in &v[nc - nn..nc] {
        is_target.set(u64::from(node));
    }

    SourceTargetResult {
        is_source,
        is_target,
    }
}

// ──────────────────────────────────────────────────────────────────────────
// derive_separator_from_cut  (nested_dissection.cpp:786)
// ──────────────────────────────────────────────────────────────────────────

/// Given a fragment and a cut (nodes on one side), return the separator nodes.
///
/// The separator is the set of nodes NOT on the small side that have at least
/// one neighbor on the small side.  Ported from `nested_dissection.cpp:786`.
#[must_use]
pub(crate) fn derive_separator_from_cut(
    fragment: &GraphFragment,
    cut: &crate::internal::bitvec::BitVector,
) -> crate::internal::bitvec::BitVector {
    let node_count = fragment.node_count();
    let small_side = cut.population_count() <= u64::from(node_count) / 2;
    let mut is_separator_node = crate::internal::bitvec::BitVector::new(u64::from(node_count));

    for xy in 0..fragment.arc_count() as usize {
        let x = fragment.tail[xy];
        let y = fragment.head[xy];
        if cut.is_set(u64::from(x)) == small_side && cut.is_set(u64::from(y)) != small_side {
            is_separator_node.set(u64::from(y));
        }
    }

    is_separator_node
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── make_graph_fragment ──────────────────────────────────────────────

    /// Triangle graph 0↔1↔2↔0.
    /// Hand-computed expected CSR layout verified step-by-step.
    #[test]
    fn make_fragment_triangle() {
        let tail = vec![0u32, 1, 2];
        let head = vec![1u32, 2, 0];
        let f = make_graph_fragment(3, &tail, &head);

        assert_eq!(f.node_count(), 3);
        // 3 input arcs × 2 (no self-loops) = 6 directed arcs.
        assert_eq!(f.arc_count(), 6);
        assert_eq!(f.global_node_id, vec![0u32, 1, 2]);

        // After sort by (tail, head): (0,1),(0,2),(1,0),(1,2),(2,0),(2,1)
        assert_eq!(f.first_out, vec![0u32, 2, 4, 6]);
        assert_eq!(f.tail, vec![0u32, 0, 1, 1, 2, 2]);
        assert_eq!(f.head, vec![1u32, 2, 0, 2, 0, 1]);

        // back_arc[back_arc[a]] == a for all a.
        for a in 0..f.arc_count() as usize {
            assert_eq!(
                f.back_arc[f.back_arc[a] as usize],
                u32::try_from(a).unwrap(),
                "back_arc involution failed at a={a}"
            );
        }
        // tail[back_arc[a]] == head[a] and head[back_arc[a]] == tail[a]
        for a in 0..f.arc_count() as usize {
            let b = f.back_arc[a] as usize;
            assert_eq!(
                f.tail[b], f.head[a],
                "tail[back_arc[a]] != head[a] at a={a}"
            );
            assert_eq!(
                f.head[b], f.tail[a],
                "head[back_arc[a]] != tail[a] at a={a}"
            );
        }
    }

    /// Single edge 0↔1.
    #[test]
    fn make_fragment_single_edge() {
        let f = make_graph_fragment(2, &[0u32], &[1u32]);
        assert_eq!(f.node_count(), 2);
        assert_eq!(f.arc_count(), 2);
        assert_eq!(f.first_out, vec![0u32, 1, 2]);
        // back_arc involution
        assert_eq!(f.back_arc[f.back_arc[0] as usize], 0);
        assert_eq!(f.back_arc[f.back_arc[1] as usize], 1);
    }

    /// Self-loop is discarded.
    #[test]
    fn make_fragment_self_loop_discarded() {
        let f = make_graph_fragment(2, &[0u32, 1], &[0u32, 1]);
        assert_eq!(f.arc_count(), 0);
    }

    /// Empty arc list.
    #[test]
    fn make_fragment_no_arcs() {
        let f = make_graph_fragment(3, &[], &[]);
        assert_eq!(f.node_count(), 3);
        assert_eq!(f.arc_count(), 0);
        assert_eq!(f.first_out, vec![0u32, 0, 0, 0]);
    }

    // ── decompose_graph_fragment_into_connected_components ───────────────

    /// Two disjoint edges: {0↔1} and {2↔3}.
    /// Should produce 2 components with correct `global_node_id` sets.
    #[test]
    fn decompose_two_components() {
        let f = make_graph_fragment(4, &[0u32, 2], &[1u32, 3]);
        let parts = decompose_graph_fragment_into_connected_components(f);
        assert_eq!(parts.len(), 2);

        for p in &parts {
            assert_eq!(p.node_count(), 2, "each component has 2 nodes");
            assert_eq!(p.arc_count(), 2, "each component has 2 arcs");
            // back_arc involution
            for a in 0..p.arc_count() as usize {
                assert_eq!(
                    p.back_arc[p.back_arc[a] as usize],
                    u32::try_from(a).unwrap()
                );
            }
        }

        // Global node ids across both parts must cover {0,1,2,3}.
        let mut all_ids: Vec<u32> = parts
            .iter()
            .flat_map(|p| p.global_node_id.iter().copied())
            .collect();
        all_ids.sort_unstable();
        assert_eq!(all_ids, vec![0u32, 1, 2, 3]);
    }

    /// Single connected component — decompose returns exactly 1 fragment.
    #[test]
    fn decompose_single_component() {
        let f = make_graph_fragment(3, &[0u32, 1], &[1u32, 2]);
        let parts = decompose_graph_fragment_into_connected_components(f);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].node_count(), 3);
        assert_eq!(parts[0].arc_count(), 4); // path 0-1-2: 4 directed arcs
        assert_eq!(parts[0].global_node_id.len(), 3);
    }

    /// Isolated nodes (no arcs) each form their own component.
    #[test]
    fn decompose_isolated_nodes() {
        let f = make_graph_fragment(3, &[], &[]);
        let parts = decompose_graph_fragment_into_connected_components(f);
        assert_eq!(parts.len(), 3);
        for p in &parts {
            assert_eq!(p.node_count(), 1);
            assert_eq!(p.arc_count(), 0);
        }
    }

    // ── select_source_and_target ─────────────────────────────────────────

    /// 6 nodes with distinct x-coordinates 0..5. n=1.
    /// Source = node with smallest coord (node 0), target = node with largest (node 5).
    #[test]
    fn select_source_target_distinct_coords_n1() {
        let coords = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = select_source_and_target(1, 6, |x| coords[x as usize]);

        let source_nodes: Vec<u64> = (0..6u64).filter(|&i| result.is_source.is_set(i)).collect();
        let target_nodes: Vec<u64> = (0..6u64).filter(|&i| result.is_target.is_set(i)).collect();

        assert_eq!(source_nodes.len(), 1, "should have exactly 1 source");
        assert_eq!(target_nodes.len(), 1, "should have exactly 1 target");
        assert_eq!(source_nodes[0], 0, "source should be node with key=0.0");
        assert_eq!(target_nodes[0], 5, "target should be node with key=5.0");
    }

    /// n=2: 2 lowest-key nodes are sources, 2 highest are targets.
    #[test]
    fn select_source_target_n2() {
        let coords = [10.0f32, 30.0, 50.0, 70.0, 20.0, 60.0];
        // Sorted: 10(0), 20(4), 30(1), 50(2), 60(5), 70(3)
        // Sources (2 lowest): nodes 0, 4. Targets (2 highest): nodes 3, 5.
        let result = select_source_and_target(2, 6, |x| coords[x as usize]);

        let sources: std::collections::HashSet<u64> =
            (0..6u64).filter(|&i| result.is_source.is_set(i)).collect();
        let targets: std::collections::HashSet<u64> =
            (0..6u64).filter(|&i| result.is_target.is_set(i)).collect();

        assert_eq!(sources.len(), 2);
        assert_eq!(targets.len(), 2);
        assert!(sources.contains(&0), "node 0 (key=10) should be a source");
        assert!(sources.contains(&4), "node 4 (key=20) should be a source");
        assert!(targets.contains(&3), "node 3 (key=70) should be a target");
        assert!(targets.contains(&5), "node 5 (key=60) should be a target");
        assert!(
            sources.is_disjoint(&targets),
            "sources and targets must be disjoint"
        );
    }

    /// n=0: both source and target bitvectors are all-zero.
    #[test]
    fn select_source_target_n0() {
        let result = select_source_and_target(0, 4, |_| 0.0f32);
        for i in 0..4u64 {
            assert!(!result.is_source.is_set(i), "no sources when n=0");
            assert!(!result.is_target.is_set(i), "no targets when n=0");
        }
    }

    // ── derive_separator_from_cut ────────────────────────────────────────

    /// Path 0-1-2-3. Cut = {0,1} (small side). Separator = node 2 (adjacent to node 1).
    #[test]
    fn derive_separator_known_case() {
        // Path: 0↔1↔2↔3
        let f = make_graph_fragment(4, &[0u32, 1, 2], &[1u32, 2, 3]);
        // Cut = nodes 0 and 1 on small side (2 out of 4 = exactly half).
        let mut cut = crate::internal::bitvec::BitVector::new(4);
        cut.set(0);
        cut.set(1);

        let sep = derive_separator_from_cut(&f, &cut);

        // Node 2 is in separator (adjacent to node 1 on small side).
        assert!(sep.is_set(2), "node 2 should be in separator");
        assert!(!sep.is_set(0), "node 0 is on the cut side");
        assert!(!sep.is_set(1), "node 1 is on the cut side");
        assert!(!sep.is_set(3), "node 3 has no neighbor on small side");
    }

    // ── pick_smaller_side ────────────────────────────────────────────────

    /// Larger-side flip: 4 of 6 nodes marked → threshold=3, 4 ≥ 3 → flip.
    #[test]
    fn pick_smaller_side_flips_large_to_small() {
        let mut bv = crate::internal::bitvec::BitVector::new(6);
        for i in 0..4u64 {
            bv.set(i);
        }
        let mut c = CutSide {
            node_on_side_count: 4,
            cut_size: 2,
            is_node_on_side: bv,
        };
        pick_smaller_side(&mut c);
        assert_eq!(c.node_on_side_count, 2, "should flip to the 2-node side");
        assert!(!c.is_node_on_side.is_set(0), "node 0 flipped off");
        assert!(c.is_node_on_side.is_set(4), "node 4 flipped on");
        assert!(c.is_node_on_side.is_set(5), "node 5 flipped on");
    }

    /// No-flip: 2 of 6 nodes marked → threshold=3, 2 < 3 → unchanged.
    #[test]
    fn pick_smaller_side_stays_when_already_small() {
        let mut bv = crate::internal::bitvec::BitVector::new(6);
        bv.set(0);
        bv.set(1);
        let mut c = CutSide {
            node_on_side_count: 2,
            cut_size: 1,
            is_node_on_side: bv,
        };
        pick_smaller_side(&mut c);
        assert_eq!(c.node_on_side_count, 2, "should not flip");
        assert!(c.is_node_on_side.is_set(0));
        assert!(c.is_node_on_side.is_set(1));
        assert!(!c.is_node_on_side.is_set(2));
    }

    /// Tie: 3 of 6 nodes marked → threshold=3, 3 ≥ 3 → flip (C++ behavior).
    #[test]
    fn pick_smaller_side_tie_flips() {
        let mut bv = crate::internal::bitvec::BitVector::new(6);
        bv.set(0);
        bv.set(1);
        bv.set(2);
        let mut c = CutSide {
            node_on_side_count: 3,
            cut_size: 0,
            is_node_on_side: bv,
        };
        pick_smaller_side(&mut c);
        // After flip: nodes 3,4,5 are set, count=3.
        assert_eq!(c.node_on_side_count, 3);
        assert!(!c.is_node_on_side.is_set(0));
        assert!(c.is_node_on_side.is_set(3));
    }
}

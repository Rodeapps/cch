//! CCH structure building — a faithful port of the structure-building half of
//! `RoutingKit`'s `CustomizableContractionHierarchy` constructor.
//!
//! Given the **same input graph and the same contraction order**, the arrays
//! produced here are **bit-identical** to the C++ original
//! (`oracle/routingkit-cch/RoutingKit/src/customizable_contraction_hierarchy.cpp`,
//! constructor body ~lines 196–543, with `filter_always_inf_arcs = false`).
//!
//! The build proceeds exactly as the C++:
//! 1. `rank = invert_permutation(order)` (C++ line 215).
//! 2. Relabel input arc endpoints by `rank` (lines 228–229).
//! 3. Sort input arcs first-by-tail-then-by-head (lines 243–246).
//! 4. Symmetrize (append reversed arcs), sort, then drop duplicates and
//!    self-loops (lines 268–287).
//! 5. Build the chordal supergraph by contracting nodes in rank order
//!    (`compute_chordal_supergraph`, lines 26–57 / 289–300).
//! 6. Sort the resulting up-arcs first-by-tail-then-by-head and build
//!    `up_first_out` (lines 309–314).
//! 7. `elimination_tree_parent[x] = up_head[up_first_out[x]]` or `INVALID_ID`
//!    if `x` has no up-arcs (lines 390–396).
//! 8. Down graph: `down_tail = up_head`, `down_head = up_tail`; the sort
//!    permutation first-by-tail-then-by-head is `down_to_up`; apply it to
//!    `down_head`; `down_first_out = invert_vector(down_tail, …)`
//!    (lines 536–543).
//!
//! The `input_arc_to_cch_arc` / `is_input_arc_upward` mappings (C++ lines
//! 325–376) and the filtering block (lines 452–528, skipped when
//! `filter_always_inf_arcs == false`) are not part of the 7-array structure
//! gate and are omitted.

use crate::bundle::INVALID_ID;
use crate::graph::Graph;
use crate::internal::permutation::{apply_permutation, inverse_permutation};

/// A built CCH structure (no metric / weights).
///
/// Field semantics match the persisted `.cch-struct` sections and
/// [`crate::bundle::CchView`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cch {
    /// `rank[external_node]` → CCH-internal id. Length = `node_count`.
    pub rank: Vec<u32>,
    /// The contraction order (inverse of `rank`). Length = `node_count`.
    pub order: Vec<u32>,
    /// Elimination-tree parent per node, `INVALID_ID` for roots.
    /// Length = `node_count`.
    pub elimination_tree_parent: Vec<u32>,
    /// CSR row-pointers into `up_head`/`up_tail`. Length = `node_count + 1`.
    pub up_first_out: Vec<u32>,
    /// Up-arc heads (sorted by tail then head). Length = `cch_arc_count`.
    pub up_head: Vec<u32>,
    /// Up-arc tails (parallel to `up_head`). Length = `cch_arc_count`.
    pub up_tail: Vec<u32>,
    /// CSR row-pointers into `down_head`. Length = `node_count + 1`.
    pub down_first_out: Vec<u32>,
    /// Down-arc heads (sorted by down-tail then down-head).
    /// Length = `cch_arc_count`.
    pub down_head: Vec<u32>,
    /// Maps a down-arc index to its corresponding up-arc index.
    /// Length = `cch_arc_count`.
    pub down_to_up: Vec<u32>,
}

impl Cch {
    /// Number of nodes in the CCH.
    #[must_use]
    #[inline]
    pub fn node_count(&self) -> usize {
        self.rank.len()
    }

    /// Number of CCH arcs (the contracted/chordal-supergraph arc count).
    #[must_use]
    #[inline]
    pub fn cch_arc_count(&self) -> usize {
        self.up_head.len()
    }

    /// Builds the CCH structure from an input `graph` (CSR) and a contraction
    /// `order` (a permutation of node ids; `order[i]` is the i-th node to be
    /// contracted).
    ///
    /// Bit-identical to `RoutingKit`'s `CustomizableContractionHierarchy`
    /// constructor with `filter_always_inf_arcs = false`.
    ///
    /// # Panics
    /// Panics if `order.len() != graph.node_count()`, or (in debug builds) if
    /// `order` is not a valid permutation.
    #[must_use]
    pub fn build(graph: &Graph, order: &[u32]) -> Cch {
        let node_count = order.len();
        assert_eq!(
            node_count,
            graph.node_count(),
            "order length must equal graph node count"
        );

        // C++ line 215: rank = invert_permutation(order).
        let rank = inverse_permutation(order);

        // Derive (tail, head) from the CSR graph, then relabel endpoints by
        // rank (C++ lines 228–229: apply_permutation_to_elements_of(rank, …)).
        let input_arc_count = graph.arc_count();
        let mut input_tail = Vec::with_capacity(input_arc_count);
        let mut input_head = Vec::with_capacity(input_arc_count);
        for v in 0..node_count {
            let start = graph.first_out[v] as usize;
            let end = graph.first_out[v + 1] as usize;
            for arc in start..end {
                input_tail.push(rank[v]);
                input_head.push(rank[graph.head[arc] as usize]);
            }
        }

        // C++ lines 243–246: sort arcs first-by-tail-then-by-head.
        sort_by_tail_then_head(node_count, &mut input_tail, &mut input_head);

        // C++ lines 268–287: symmetrize, sort, drop duplicates + self-loops.
        let (sym_tail, sym_head) = symmetrize_and_dedup(node_count, &input_tail, &input_head);

        // C++ lines 289–300: build chordal supergraph; the callback pushes
        // up-arcs in contraction (rank) order.
        let (mut up_tail, mut up_head) =
            compute_chordal_supergraph(node_count, &sym_tail, &sym_head);

        // C++ lines 309–312: sort up-arcs first-by-tail-then-by-head.
        sort_by_tail_then_head(node_count, &mut up_tail, &mut up_head);

        // C++ line 314: up_first_out = invert_vector(up_tail, node_count).
        let up_first_out = invert_vector(&up_tail, node_count);

        // C++ lines 390–396: elimination tree parent.
        let mut elimination_tree_parent = vec![0u32; node_count];
        for x in 0..node_count {
            if up_first_out[x] == up_first_out[x + 1] {
                elimination_tree_parent[x] = INVALID_ID;
            } else {
                elimination_tree_parent[x] = up_head[up_first_out[x] as usize];
            }
        }

        // C++ lines 536–543: down graph.
        //   down_tail = up_head; down_head = up_tail;
        //   down_to_up = compute_sort_permutation_first_by_tail_then_by_head(
        //                    down_tail, down_head)  (also sorts down_tail);
        //   down_head = apply_permutation(down_to_up, down_head);
        //   down_first_out = invert_vector(down_tail, node_count);
        let mut down_tail = up_head.clone();
        let down_head_unsorted = up_tail.clone();
        let down_to_up =
            compute_sort_perm_by_tail_then_head(node_count, &mut down_tail, &down_head_unsorted);
        let down_head = apply_permutation(&down_to_up, &down_head_unsorted);
        let down_first_out = invert_vector(&down_tail, node_count);

        Cch {
            rank,
            order: order.to_vec(),
            elimination_tree_parent,
            up_first_out,
            up_head,
            up_tail,
            down_first_out,
            down_head,
            down_to_up,
        }
    }
}

/// Computes the (forward) stable sort permutation `p` that orders the arcs
/// `(tail, head)` first by `tail` then by `head`, such that the sorted arrays
/// are `tail[p[i]]`, `head[p[i]]`. `tail` is sorted in place (matching the
/// C++ `…_and_apply_sort_to_tail` family, which sorts `a` in place and returns
/// the sort permutation).
///
/// Equivalent to `compute_sort_permutation_first_by_tail_then_by_head_and_apply_sort_to_tail`
/// (`graph_util.cpp:142`). Both `tail` and `head` are keys in `[0, node_count)`,
/// so a stable two-pass bucket-style sort reproduces `RoutingKit`'s result
/// exactly (its bucket sort is stable, and its comparator fallback uses
/// `std::stable_sort`, which is also stable).
fn compute_sort_perm_by_tail_then_head(
    node_count: usize,
    tail: &mut [u32],
    head: &[u32],
) -> Vec<u32> {
    // p: stable sort by head (C++: compute_stable_sort_permutation_using_key(b)).
    let p = stable_sort_perm_by_key(head, node_count);
    // a' = apply_permutation(p, a) (the tail reordered by p).
    let tail_by_p: Vec<u32> = p.iter().map(|&i| tail[i as usize]).collect();
    // q: stable sort of a' by key (C++: compute_stable_sort_permutation_using_key).
    let q = stable_sort_perm_by_key(&tail_by_p, node_count);
    // result = chain_permutation_first_left_then_right(p, q) → r[i] = p[q[i]].
    let r: Vec<u32> = q.iter().map(|&qi| p[qi as usize]).collect();
    // Sort `tail` in place using the resulting permutation.
    let sorted_tail: Vec<u32> = r.iter().map(|&i| tail[i as usize]).collect();
    tail.copy_from_slice(&sorted_tail);
    r
}

/// Sorts `(tail, head)` first by tail then by head, in place. The permutation
/// itself is not needed by the caller (the C++ uses the inverse-sort variant
/// here, but the *sorted arrays* are identical regardless of which sort
/// variant is used).
fn sort_by_tail_then_head(node_count: usize, tail: &mut Vec<u32>, head: &mut Vec<u32>) {
    let mut t = std::mem::take(tail);
    let p = compute_sort_perm_by_tail_then_head(node_count, &mut t, head);
    let sorted_head: Vec<u32> = p.iter().map(|&i| head[i as usize]).collect();
    *tail = t;
    *head = sorted_head;
}

/// Stable sort permutation by a single key in `[0, key_count)`: returns `p`
/// such that `v[p[0]] <= v[p[1]] <= …` with ties broken by original index.
///
/// Reproduces `compute_stable_sort_permutation_using_key` (counting/bucket
/// sort, always stable).
fn stable_sort_perm_by_key(v: &[u32], key_count: usize) -> Vec<u32> {
    // Counting sort: prefix sums give each bucket's start; ascending iteration
    // preserves the original relative order within a bucket (stable).
    let mut bucket_pos = vec![0u32; key_count + 1];
    for &k in v {
        bucket_pos[k as usize + 1] += 1;
    }
    for i in 0..key_count {
        bucket_pos[i + 1] += bucket_pos[i];
    }
    let mut p = vec![0u32; v.len()];
    for (i, &k) in v.iter().enumerate() {
        let pos = bucket_pos[k as usize] as usize;
        p[pos] = u32::try_from(i).expect("arc index fits u32");
        bucket_pos[k as usize] += 1;
    }
    p
}

/// Symmetrize the (already tail/head-sorted) input arcs by appending the
/// reversed copy, re-sort, then drop duplicate arcs and self-loops.
///
/// Ports C++ lines 268–287. Returns `(symmetric_tail, symmetric_head)` after
/// the `inplace_keep_element_of_vector_if(filter, …)` compaction.
fn symmetrize_and_dedup(
    node_count: usize,
    input_tail: &[u32],
    input_head: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let input_arc_count = input_tail.len();
    let mut sym_tail = Vec::with_capacity(2 * input_arc_count);
    let mut sym_head = Vec::with_capacity(2 * input_arc_count);
    // First half: the original arcs.
    sym_tail.extend_from_slice(input_tail);
    sym_head.extend_from_slice(input_head);
    // Second half: reversed arcs.
    sym_tail.extend_from_slice(input_head);
    sym_head.extend_from_slice(input_tail);

    // Sort first-by-tail-then-by-head (C++ lines 276–278).
    sort_by_tail_then_head(node_count, &mut sym_tail, &mut sym_head);

    // Build the keep-filter (C++ lines 280–287):
    //   keep[0]   = tail[0] != head[0]
    //   keep[i>0] = (head[i] != head[i-1] || tail[i] != tail[i-1])  // not a dup
    //               && (tail[i] != head[i])                         // not a loop
    let total = sym_tail.len();
    let mut kept_tail = Vec::with_capacity(total);
    let mut kept_head = Vec::with_capacity(total);
    if input_arc_count != 0 && sym_tail[0] != sym_head[0] {
        kept_tail.push(sym_tail[0]);
        kept_head.push(sym_head[0]);
    }
    for i in 1..total {
        let not_dup = sym_head[i] != sym_head[i - 1] || sym_tail[i] != sym_tail[i - 1];
        let not_loop = sym_tail[i] != sym_head[i];
        if not_dup && not_loop {
            kept_tail.push(sym_tail[i]);
            kept_head.push(sym_head[i]);
        }
    }
    (kept_tail, kept_head)
}

/// Builds the chordal supergraph by contracting nodes in ascending rank order,
/// invoking the equivalent of the C++ `on_new_arc(x, y)` callback for every
/// up-arc generated, in the same order.
///
/// Faithful port of `compute_chordal_supergraph`
/// (`customizable_contraction_hierarchy.cpp:26–57`). The input `(tail, head)`
/// must be the symmetrized, deduped, loop-free arc set.
fn compute_chordal_supergraph(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    // nodes[n] = sorted, unique upward neighbors of n (only head > n kept).
    let mut nodes: Vec<Vec<u32>> = vec![Vec::new(); node_count];
    for i in 0..tail.len() {
        if tail[i] < head[i] {
            nodes[tail[i] as usize].push(head[i]);
        }
    }
    for list in &mut nodes {
        list.sort_unstable();
        list.dedup();
    }

    let mut up_tail = Vec::new();
    let mut up_head = Vec::new();

    for n in 0..node_count {
        if nodes[n].is_empty() {
            continue;
        }
        let lowest_neighbor = nodes[n][0] as usize;

        // merged = sorted-unique union of (nodes[n] without its first element)
        // and nodes[lowest_neighbor]. This is the fill-in propagation: the
        // contracted node's higher neighbors become neighbors of its lowest
        // neighbor (C++ lines 45–49, std::merge + std::unique).
        let merged = merge_unique(&nodes[n][1..], &nodes[lowest_neighbor]);
        nodes[lowest_neighbor] = merged;

        // Emit one up-arc (n -> neighbor) for each upward neighbor of n,
        // in ascending neighbor order (C++ lines 51–53).
        let tail_n = u32::try_from(n).expect("node id fits u32");
        for &neighbor in &nodes[n] {
            up_tail.push(tail_n);
            up_head.push(neighbor);
        }
    }

    (up_tail, up_head)
}

/// Merge two sorted, unique `u32` slices into a single sorted, unique vector
/// (set union). Reproduces `std::merge` followed by `std::unique`.
fn merge_unique(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                push_unique(&mut out, a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                push_unique(&mut out, b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                push_unique(&mut out, a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    while i < a.len() {
        push_unique(&mut out, a[i]);
        i += 1;
    }
    while j < b.len() {
        push_unique(&mut out, b[j]);
        j += 1;
    }
    out
}

/// Push `value` onto `out` unless it equals the current last element (keeps the
/// vector sorted-unique when fed non-decreasing values).
#[inline]
fn push_unique(out: &mut Vec<u32>, value: u32) {
    if out.last() != Some(&value) {
        out.push(value);
    }
}

/// CSR row-pointer construction from a sorted `tail` array. Returns a vector of
/// length `element_count + 1` where entry `i` is the index of the first arc
/// whose tail is `>= i`, and the last entry is `tail.len()`.
///
/// Faithful port of `invert_vector` (`inverse_vector.h:20–40`). `tail` must be
/// sorted ascending.
fn invert_vector(tail: &[u32], element_count: usize) -> Vec<u32> {
    let mut index = vec![0u32; element_count + 1];
    if tail.is_empty() {
        return index;
    }
    let mut pos = 0usize;
    for (i, slot) in index.iter_mut().take(element_count).enumerate() {
        while pos < tail.len() && (tail[pos] as usize) < i {
            pos += 1;
        }
        *slot = u32::try_from(pos).expect("arc index fits u32");
    }
    index[element_count] = u32::try_from(tail.len()).expect("arc count fits u32");
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_graph(n: usize) -> Graph {
        Graph {
            first_out: vec![0u32; n + 1],
            head: vec![],
            weight: vec![],
        }
    }

    // ------------------------------------------------------------------
    // Empty graph (no nodes, no arcs).
    // ------------------------------------------------------------------
    #[test]
    fn build_empty_graph() {
        let g = empty_graph(0);
        let order: Vec<u32> = vec![];
        let c = Cch::build(&g, &order);
        assert_eq!(c.node_count(), 0);
        assert_eq!(c.cch_arc_count(), 0);
        assert!(c.rank.is_empty());
        assert!(c.order.is_empty());
        assert!(c.elimination_tree_parent.is_empty());
        assert_eq!(c.up_first_out, vec![0]);
        assert!(c.up_head.is_empty());
        assert!(c.up_tail.is_empty());
        assert_eq!(c.down_first_out, vec![0]);
        assert!(c.down_head.is_empty());
        assert!(c.down_to_up.is_empty());
    }

    // ------------------------------------------------------------------
    // Single isolated node: a root with no up-arcs.
    // ------------------------------------------------------------------
    #[test]
    fn build_single_node() {
        let g = empty_graph(1);
        let order: Vec<u32> = vec![0];
        let c = Cch::build(&g, &order);
        assert_eq!(c.rank, vec![0]);
        assert_eq!(c.order, vec![0]);
        assert_eq!(c.elimination_tree_parent, vec![INVALID_ID]);
        assert_eq!(c.up_first_out, vec![0, 0]);
        assert!(c.up_head.is_empty());
        assert_eq!(c.down_first_out, vec![0, 0]);
        assert!(c.down_to_up.is_empty());
    }

    // ------------------------------------------------------------------
    // Triangle fill-in (no oracle): contracting node 0 (neighbors 1,2)
    // creates a shortcut arc 1->2. Verifies up/down/elim arrays directly.
    // ------------------------------------------------------------------
    /// Build a CSR `Graph` from a directed arc multiset, grouping arcs by tail.
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

    #[test]
    fn build_triangle_fillin() {
        // Undirected path 1-0-2 (so 0 has neighbors 1 and 2; 1 and 2 are not
        // yet adjacent). Contracting 0 first adds shortcut 1->2.
        let tail = [0u32, 1, 0, 2];
        let head = [1u32, 0, 2, 0];
        let g = csr(3, &tail, &head);
        let order: Vec<u32> = vec![0, 1, 2];
        let c = Cch::build(&g, &order);

        // Up-arcs: 0->1, 0->2, and shortcut 1->2.
        assert_eq!(c.up_tail, vec![0, 0, 1]);
        assert_eq!(c.up_head, vec![1, 2, 2]);
        assert_eq!(c.up_first_out, vec![0, 2, 3, 3]);
        // elimination parents: 0->1 (first up), 1->2, 2 root.
        assert_eq!(c.elimination_tree_parent, vec![1, 2, INVALID_ID]);
        // down graph: down_tail=up_head=[1,2,2], down_head=up_tail=[0,0,1].
        // sorted by (down_tail, down_head): (1,0),(2,0),(2,1) → already sorted.
        assert_eq!(c.down_first_out, vec![0, 0, 1, 3]);
        assert_eq!(c.down_head, vec![0, 0, 1]);
        assert_eq!(c.down_to_up, vec![0, 1, 2]);
    }

    // ------------------------------------------------------------------
    // Unit tests for the small helpers (edge paths).
    // ------------------------------------------------------------------
    #[test]
    fn invert_vector_empty() {
        assert_eq!(invert_vector(&[], 3), vec![0, 0, 0, 0]);
        assert_eq!(invert_vector(&[], 0), vec![0]);
    }

    #[test]
    fn invert_vector_basic() {
        // tails [0,0,2] over 3 nodes → first_out [0,2,2,3].
        assert_eq!(invert_vector(&[0, 0, 2], 3), vec![0, 2, 2, 3]);
    }

    #[test]
    fn merge_unique_basic() {
        assert_eq!(merge_unique(&[1, 3, 5], &[2, 3, 6]), vec![1, 2, 3, 5, 6]);
        assert_eq!(merge_unique(&[], &[1, 2]), vec![1, 2]);
        assert_eq!(merge_unique(&[1, 2], &[]), vec![1, 2]);
        assert_eq!(merge_unique(&[], &[] as &[u32]), Vec::<u32>::new());
    }

    #[test]
    fn stable_sort_perm_by_key_basic() {
        // keys: [2,0,1,0] over key_count 3 → stable order of indices:
        // bucket0: indices 1,3 ; bucket1: index 2 ; bucket2: index 0.
        assert_eq!(stable_sort_perm_by_key(&[2, 0, 1, 0], 3), vec![1, 3, 2, 0]);
    }

    #[test]
    fn sort_by_tail_then_head_basic() {
        let mut tail = vec![2u32, 0, 2, 0];
        let mut head = vec![1u32, 3, 0, 1];
        sort_by_tail_then_head(4, &mut tail, &mut head);
        assert_eq!(tail, vec![0, 0, 2, 2]);
        assert_eq!(head, vec![1, 3, 0, 1]);
    }
}

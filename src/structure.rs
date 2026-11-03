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
//!    self-loops (lines 268–287). Steps 2 to 4 only feed the upward neighbour
//!    lists of step 5, which are read from the graph without copying the arcs.
//! 5. Build the chordal supergraph by contracting nodes in rank order
//!    (`compute_chordal_supergraph`, lines 26–57 / 289–300).
//! 6. Sort the resulting up-arcs first-by-tail-then-by-head and build
//!    `up_first_out` (lines 309–314). Step 5 emits them in that order, so no
//!    sort runs.
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

    // ---- Part A: input-arc → CCH-arc mapping (weight-independent) ----
    /// For each INPUT arc (in the order `Cch::build` derives them from the CSR
    /// graph), the CCH up-arc id it maps to, or [`INVALID_ID`] for a self-loop.
    /// Length = `input_arc_count`. Ports C++ `input_arc_to_cch_arc`.
    pub input_arc_to_cch_arc: Vec<u32>,
    /// Per CCH arc, the input arc whose weight initializes its FORWARD weight,
    /// or [`INVALID_ID`] if none. Length = `cch_arc_count`. Ports the (flattened)
    /// C++ `forward_input_arc_of_cch` — the "first" input arc per CCH arc.
    pub forward_input_arc_of_cch: Vec<u32>,
    /// Per CCH arc, the input arc whose weight initializes its BACKWARD weight,
    /// or [`INVALID_ID`] if none. Length = `cch_arc_count`. Ports the (flattened)
    /// C++ `backward_input_arc_of_cch`.
    pub backward_input_arc_of_cch: Vec<u32>,
    /// CSR row-pointers into [`Self::extra_forward_input_arc_of_cch`], one per
    /// CCH arc (+1). Entry `i..i+1` is the range of EXTRA forward input arcs
    /// (parallel arcs beyond the first) mapping to CCH arc `i`. Length =
    /// `cch_arc_count + 1`. Ports C++ `first_extra_forward_input_arc_of_cch`.
    pub first_extra_forward_input_arc_of_cch: Vec<u32>,
    /// Flat list of extra (parallel) forward input arcs, grouped by CCH arc per
    /// [`Self::first_extra_forward_input_arc_of_cch`]. Ports C++
    /// `extra_forward_input_arc_of_cch`.
    pub extra_forward_input_arc_of_cch: Vec<u32>,
    /// CSR row-pointers into [`Self::extra_backward_input_arc_of_cch`], one per
    /// CCH arc (+1). Length = `cch_arc_count + 1`. Ports C++
    /// `first_extra_backward_input_arc_of_cch`.
    pub first_extra_backward_input_arc_of_cch: Vec<u32>,
    /// Flat list of extra (parallel) backward input arcs. Ports C++
    /// `extra_backward_input_arc_of_cch`.
    pub extra_backward_input_arc_of_cch: Vec<u32>,
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

    /// Borrow this in-memory CCH structure as a [`CchView`](crate::CchView) —
    /// the query-ready view consumed by [`distance_matrix`](crate::distance_matrix)
    /// and [`node_path`](crate::node_path). Lets you query a freshly-built `Cch`
    /// without round-tripping through a `.cch-struct` bundle on disk.
    #[must_use]
    pub fn view(&self) -> crate::bundle::CchView<'_> {
        crate::bundle::CchView {
            rank: &self.rank,
            elimination_tree_parent: &self.elimination_tree_parent,
            up_first_out: &self.up_first_out,
            up_head: &self.up_head,
            down_first_out: &self.down_first_out,
            down_head: &self.down_head,
            down_to_up: &self.down_to_up,
        }
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

        // C++ lines 228–312: relabel, symmetrize and dedup the input arcs, build
        // the chordal supergraph and sort its up-arcs first-by-tail-then-by-head.
        // The arcs come out of the contraction in that order already, so the
        // sort is the identity.
        let (up_tail, up_head) = compute_chordal_supergraph(graph, &rank);

        // C++ line 314: up_first_out = invert_vector(up_tail, node_count).
        let up_first_out = invert_vector(&up_tail, node_count);

        // Part A — input-arc → CCH-arc mapping (C++ lines 325–376, 555–640).
        let mapping = build_input_arc_mapping(graph, &rank, &up_first_out, &up_head);

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
            input_arc_to_cch_arc: mapping.input_arc_to_cch_arc,
            forward_input_arc_of_cch: mapping.forward_input_arc_of_cch,
            backward_input_arc_of_cch: mapping.backward_input_arc_of_cch,
            first_extra_forward_input_arc_of_cch: mapping.first_extra_forward_input_arc_of_cch,
            extra_forward_input_arc_of_cch: mapping.extra_forward_input_arc_of_cch,
            first_extra_backward_input_arc_of_cch: mapping.first_extra_backward_input_arc_of_cch,
            extra_backward_input_arc_of_cch: mapping.extra_backward_input_arc_of_cch,
        }
    }
}

/// Result of [`build_input_arc_mapping`] — the Part-A input-arc → CCH-arc maps.
struct InputArcMapping {
    input_arc_to_cch_arc: Vec<u32>,
    forward_input_arc_of_cch: Vec<u32>,
    backward_input_arc_of_cch: Vec<u32>,
    first_extra_forward_input_arc_of_cch: Vec<u32>,
    extra_forward_input_arc_of_cch: Vec<u32>,
    first_extra_backward_input_arc_of_cch: Vec<u32>,
    extra_backward_input_arc_of_cch: Vec<u32>,
}

/// Builds the input-arc → CCH-arc mapping (C++ lines 325–376 + 555–640).
///
/// `up_first_out`/`up_head` describe the (sorted) CCH up-graph.
///
/// For each input arc with relabeled endpoints `(t, h)`:
/// * `t == h` → self-loop → maps to [`INVALID_ID`] (C++ 351–353).
/// * `t < h`  → UPWARD arc; maps to the up-arc `(t, h)`; its weight initializes
///   that CCH arc's FORWARD weight (C++ 342–350, 579).
/// * `t > h`  → DOWNWARD arc; maps to the up-arc `(h, t)`; its weight
///   initializes that CCH arc's BACKWARD weight (C++ 357–375, 588).
///
/// The first input arc seen for a (CCH arc, direction) becomes
/// `forward/backward_input_arc_of_cch`; any further (parallel) input arcs go
/// into the `extra_*` CSR lists, so `customize`'s reset can take the min over
/// all of them (parallel-arc minimum). C++ separates first vs. extra at lines
/// 580–596; the inverse-permutation reordering at 603–636 merely groups the
/// extra lists by CCH arc, which the CSR layout built here reproduces directly.
fn build_input_arc_mapping(
    graph: &Graph,
    rank: &[u32],
    up_first_out: &[u32],
    up_head: &[u32],
) -> InputArcMapping {
    let input_arc_count = graph.arc_count();
    let cch_arc_count = up_head.len();
    let input_arc_ids = 0..u32::try_from(input_arc_count).expect("input arc count fits u32");
    let input_arcs = || ranked_arcs(graph, rank);

    let mut input_arc_to_cch_arc = vec![INVALID_ID; input_arc_count];
    let mut forward_input_arc_of_cch = vec![INVALID_ID; cch_arc_count];
    let mut backward_input_arc_of_cch = vec![INVALID_ID; cch_arc_count];
    let mut first_extra_forward_input_arc_of_cch = vec![0u32; cch_arc_count + 1];
    let mut first_extra_backward_input_arc_of_cch = vec![0u32; cch_arc_count + 1];

    for (input_arc, (tail, head)) in input_arc_ids.clone().zip(input_arcs()) {
        if tail == head {
            continue;
        }

        let cch_arc = find_up_arc(up_first_out, up_head, tail.min(head), tail.max(head));
        input_arc_to_cch_arc[input_arc as usize] = cch_arc;
        let (first, extra_count) = if tail < head {
            (
                &mut forward_input_arc_of_cch[cch_arc as usize],
                &mut first_extra_forward_input_arc_of_cch[cch_arc as usize],
            )
        } else {
            (
                &mut backward_input_arc_of_cch[cch_arc as usize],
                &mut first_extra_backward_input_arc_of_cch[cch_arc as usize],
            )
        };

        if *first == INVALID_ID {
            *first = input_arc;
        } else {
            *extra_count += 1;
        }
    }

    let mut extra_forward_input_arc_of_cch =
        vec![0u32; inclusive_prefix_sum(&mut first_extra_forward_input_arc_of_cch) as usize];
    let mut extra_backward_input_arc_of_cch =
        vec![0u32; inclusive_prefix_sum(&mut first_extra_backward_input_arc_of_cch) as usize];

    // Each offset now holds the end of its CCH arc's range. Filling the ranges
    // from the end while walking the input arcs backwards keeps every range in
    // ascending input-arc order and leaves each offset at the start of its range.
    for (input_arc, (tail, head)) in input_arc_ids.rev().zip(input_arcs().rev()) {
        if tail == head {
            continue;
        }

        let cch_arc = input_arc_to_cch_arc[input_arc as usize] as usize;
        let (first, first_extra, extra) = if tail < head {
            (
                forward_input_arc_of_cch[cch_arc],
                &mut first_extra_forward_input_arc_of_cch,
                &mut extra_forward_input_arc_of_cch,
            )
        } else {
            (
                backward_input_arc_of_cch[cch_arc],
                &mut first_extra_backward_input_arc_of_cch,
                &mut extra_backward_input_arc_of_cch,
            )
        };

        if first != input_arc {
            first_extra[cch_arc] -= 1;
            extra[first_extra[cch_arc] as usize] = input_arc;
        }
    }

    InputArcMapping {
        input_arc_to_cch_arc,
        forward_input_arc_of_cch,
        backward_input_arc_of_cch,
        first_extra_forward_input_arc_of_cch,
        extra_forward_input_arc_of_cch,
        first_extra_backward_input_arc_of_cch,
        extra_backward_input_arc_of_cch,
    }
}

/// The input arcs in arc-id order, with both endpoints relabeled by `rank`
/// (C++ lines 228–229).
fn ranked_arcs<'a>(
    graph: &'a Graph,
    rank: &'a [u32],
) -> impl DoubleEndedIterator<Item = (u32, u32)> + 'a {
    graph
        .first_out
        .windows(2)
        .zip(rank)
        .flat_map(move |(arcs, &tail)| {
            graph.head[arcs[0] as usize..arcs[1] as usize]
                .iter()
                .map(move |&head| (tail, rank[head as usize]))
        })
}

/// Finds the up-arc id with tail `lo` and head `hi`. The up-arcs of tail `lo`
/// are `up_head[up_first_out[lo]..up_first_out[lo+1]]`, sorted ascending, so a
/// binary search locates `hi`. The arc is guaranteed to exist (every input arc
/// is part of the symmetrized supergraph).
fn find_up_arc(up_first_out: &[u32], up_head: &[u32], lo: u32, hi: u32) -> u32 {
    let start = up_first_out[lo as usize] as usize;
    let end = up_first_out[lo as usize + 1] as usize;
    let slice = &up_head[start..end];
    let offset = slice
        .binary_search(&hi)
        .expect("input arc must correspond to an existing CCH up-arc");
    #[allow(
        clippy::cast_possible_truncation,
        reason = "arc id < cch_arc_count <= u32::MAX (CCH limit)"
    )]
    let arc = (start + offset) as u32;
    arc
}

/// Replaces each count with the sum of itself and all counts before it, and
/// returns the total.
fn inclusive_prefix_sum(counts: &mut [u32]) -> u32 {
    counts.iter_mut().fold(0, |total, count| {
        *count += total;
        *count
    })
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

/// Builds the chordal supergraph by contracting nodes in ascending rank order,
/// and returns its up-arcs in the order of the C++ `on_new_arc(x, y)` callback:
/// sorted by tail, then by head.
///
/// Faithful port of `compute_chordal_supergraph`
/// (`customizable_contraction_hierarchy.cpp:26–57`). Its input in the C++ is
/// the symmetrized, deduped, loop-free set of rank-relabeled input arcs (lines
/// 228–287). The upward neighbours of a node in that set are the higher
/// endpoints of its non-loop input arcs, so they are read straight from the
/// graph.
fn compute_chordal_supergraph(graph: &Graph, rank: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let node_count = rank.len();
    let non_loop_arcs = || ranked_arcs(graph, rank).filter(|(tail, head)| tail != head);
    let mut degree = vec![0usize; node_count];

    for (tail, head) in non_loop_arcs() {
        degree[tail.min(head) as usize] += 1;
    }

    // nodes[n] = sorted, unique upward neighbors of n.
    let mut nodes: Vec<Vec<u32>> = degree.into_iter().map(Vec::with_capacity).collect();

    for (tail, head) in non_loop_arcs() {
        nodes[tail.min(head) as usize].push(tail.max(head));
    }

    for list in &mut nodes {
        list.sort_unstable();
        list.dedup();
    }

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
    }

    // Contracting a node only changes the lists of higher nodes, so each list
    // is final once its node is contracted. Emitting the up-arcs after the loop
    // gives the C++ arc order (C++ lines 51–53) with an exact allocation.
    let cch_arc_count = nodes.iter().map(Vec::len).sum();
    let mut up_tail = Vec::with_capacity(cch_arc_count);
    let mut up_head = Vec::with_capacity(cch_arc_count);

    for (tail, neighbors) in (0u32..).zip(nodes) {
        up_tail.extend(std::iter::repeat_n(tail, neighbors.len()));
        up_head.extend(neighbors);
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
    // view() borrows the struct's arrays into a query-ready CchView.
    // ------------------------------------------------------------------
    #[test]
    fn view_borrows_struct_fields() {
        // 2-node graph with one arc so up/down arrays are non-empty.
        let g = Graph {
            first_out: vec![0, 1, 1],
            head: vec![1],
            weight: vec![5],
        };
        let c = Cch::build(&g, &[0u32, 1]);
        let v = c.view();
        assert_eq!(v.rank, c.rank.as_slice());
        assert_eq!(
            v.elimination_tree_parent,
            c.elimination_tree_parent.as_slice()
        );
        assert_eq!(v.up_first_out, c.up_first_out.as_slice());
        assert_eq!(v.up_head, c.up_head.as_slice());
        assert_eq!(v.down_first_out, c.down_first_out.as_slice());
        assert_eq!(v.down_head, c.down_head.as_slice());
        assert_eq!(v.down_to_up, c.down_to_up.as_slice());
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
}

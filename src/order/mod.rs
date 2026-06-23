//! Contraction order heuristics for CCH construction.

mod nd;

use crate::graph::Graph;

/// Returns a contraction order for `graph` sorted by (degree asc, id asc).
///
/// For each arc `v -> w`, both `v` and `w` contribute one to their respective
/// degree counters (identical to the oracle's `cch_compute_order_degree`).
/// Returns a permutation of `0..node_count` — nodes with smaller incident-arc
/// degree appear earlier; ties are broken by ascending node id.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "node ids and the node count are u32-bounded by the crate's id space"
)]
pub fn degree_order(graph: &Graph) -> Vec<u32> {
    let node_count = graph.node_count();
    let mut degree = vec![0u32; node_count];
    for (v, window) in graph.first_out.windows(2).enumerate() {
        let (from, to) = (window[0] as usize, window[1] as usize);
        for &h in &graph.head[from..to] {
            degree[v] += 1;
            degree[h as usize] += 1;
        }
    }
    let mut order: Vec<u32> = (0..node_count as u32).collect();
    order.sort_unstable_by(|&a, &b| degree[a as usize].cmp(&degree[b as usize]).then(a.cmp(&b)));
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Helper: build a CSR Graph from a (tail, head) arc list.
    // Arcs are expected to be grouped by tail (sorted by tail) — we rely on the
    // caller having done that.  Weights are arbitrary (degree ignores them).
    // ---------------------------------------------------------------------------
    fn graph_from_arc_list(node_count: usize, tail: &[u32], head: &[u32]) -> Graph {
        assert_eq!(tail.len(), head.len());
        let arc_count = tail.len();

        // Count out-arcs per node.
        let mut first_out = vec![0u32; node_count + 1];
        for &t in tail {
            first_out[t as usize + 1] += 1;
        }
        // Prefix sum → first_out[v] = start index of node v's arcs.
        for i in 1..=node_count {
            first_out[i] += first_out[i - 1];
        }

        Graph {
            first_out,
            head: head.to_vec(),
            weight: vec![1u32; arc_count],
        }
    }

    // ---------------------------------------------------------------------------
    // Test 1: empty graph (node_count = 0)
    // ---------------------------------------------------------------------------
    #[test]
    fn degree_order_empty_graph() {
        let g = Graph {
            first_out: vec![0],
            head: vec![],
            weight: vec![],
        };
        let order = degree_order(&g);
        assert!(order.is_empty(), "empty graph should return empty order");
    }

    // ---------------------------------------------------------------------------
    // Test 2: single isolated node
    // ---------------------------------------------------------------------------
    #[test]
    fn degree_order_single_isolated_node() {
        let g = Graph {
            first_out: vec![0, 0],
            head: vec![],
            weight: vec![],
        };
        let order = degree_order(&g);
        assert_eq!(order, vec![0u32]);
    }

    // ---------------------------------------------------------------------------
    // Test 3: a node that appears only as a head (in-degree only)
    // Node 0 has no out-arcs but is targeted by node 1.
    // ---------------------------------------------------------------------------
    #[test]
    fn degree_order_head_only_node() {
        // Arc: 1 -> 0.  Node 0 has deg=1, node 1 has deg=1.
        // Tie-break by id: [0, 1].
        let tail = vec![1u32];
        let head = vec![0u32];
        let g = graph_from_arc_list(2, &tail, &head);
        let order = degree_order(&g);
        // Both nodes have degree 1; id-break → [0, 1].
        assert_eq!(order, vec![0u32, 1]);
    }

    // ---------------------------------------------------------------------------
    // Test 4: valid permutation check (covers a richer fixture)
    // ---------------------------------------------------------------------------
    #[test]
    fn degree_order_is_permutation() {
        // Path 0-1-2-3 + hub node 4 connected to all + parallel arc 0->1.
        // Arcs (tail, head): (0,1), (0,1), (0,4), (1,2), (1,4), (2,3), (2,4), (3,4)
        // Must be sorted by tail for CSR helper.
        let tail = vec![0u32, 0, 0, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 4, 2, 4, 3, 4, 4];
        let g = graph_from_arc_list(5, &tail, &head);
        let order = degree_order(&g);

        // Permutation check: sorted result == 0..n.
        let mut sorted = order.clone();
        sorted.sort_unstable();
        let expected: Vec<u32> = (0..5).collect();
        assert_eq!(
            sorted, expected,
            "degree_order must be a permutation of 0..n"
        );
    }

    // ---------------------------------------------------------------------------
    // Test 5: oracle equality — the critical bit-identical check.
    //
    // Fixture: path 0-1-2-3, hub node 4 connected to all, plus a parallel arc
    // 0->1 (to create unequal degrees and ties that force both comparator arms).
    //
    // degree contributions (each arc increments both tail and head):
    //   arc (0,1): deg[0]++, deg[1]++
    //   arc (0,1): deg[0]++, deg[1]++   ← parallel arc
    //   arc (0,4): deg[0]++, deg[4]++
    //   arc (1,2): deg[1]++, deg[2]++
    //   arc (1,4): deg[1]++, deg[4]++
    //   arc (2,3): deg[2]++, deg[3]++
    //   arc (2,4): deg[2]++, deg[4]++
    //   arc (3,4): deg[3]++, deg[4]++
    // degrees: [3, 4, 3, 2, 4]
    // sorted by (deg asc, id asc): node 3 (2), node 0 (3), node 2 (3), node 1 (4), node 4 (4)
    // expected order: [3, 0, 2, 1, 4]
    // ---------------------------------------------------------------------------
    #[test]
    fn degree_order_matches_cpp_oracle() {
        let tail = vec![0u32, 0, 0, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 4, 2, 4, 3, 4, 4];
        let node_count: u32 = 5;

        let g = graph_from_arc_list(node_count as usize, &tail, &head);
        let rust_order = degree_order(&g);

        let oracle_order = routingkit_cch::compute_order_degree(node_count, &tail, &head);

        assert_eq!(
            rust_order, oracle_order,
            "degree_order must be bit-identical to the C++ oracle"
        );
    }
}

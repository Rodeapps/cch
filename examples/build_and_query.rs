//! Build a CCH from scratch and run distance-matrix + node-path queries —
//! pure Rust, no external tools required.
//!
//! **Phase 2** of this crate adds full pure-Rust bundle construction:
//! contraction order (`degree_order`), CCH structure (`Cch::build`),
//! metric customization (`Cch::customize`), and bundle persistence
//! (`Cch::save_struct` / `Metric::save`). Bundles are then reopened
//! zero-copy via `CchBundle` / `MetricBundle` for queries.
//!
//! Degree-order (nodes sorted by incident-arc degree) is the supported
//! ordering heuristic. Inertial-flow nested dissection (higher quality for
//! road networks) is a planned future enhancement.
//!
//! # Usage
//!
//! ```text
//! cargo run --example build_and_query
//! ```

use std::fmt::Write as _;

fn main() {
    // ----------------------------------------------------------------
    // 1. Construct a small bidirectional 3×3 grid graph.
    //    Nodes are numbered 0..8 in row-major order:
    //      0 - 1 - 2
    //      |   |   |
    //      3 - 4 - 5
    //      |   |   |
    //      6 - 7 - 8
    // ----------------------------------------------------------------
    let cols = 3u32;
    let rows = 3u32;
    let node_count = cols * rows; // 9

    // Build arc lists grouped by tail (required by csr_from_arcs).
    let mut tail: Vec<u32> = Vec::new();
    let mut head: Vec<u32> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                tail.push(v);
                head.push(v + 1); // right
            }
            if c > 0 {
                tail.push(v);
                head.push(v - 1); // left
            }
            if r + 1 < rows {
                tail.push(v);
                head.push(v + cols); // down
            }
            if r > 0 {
                tail.push(v);
                head.push(v - cols); // up
            }
        }
    }

    // Varied arc weights: arc index × 3 + 1.
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32).map(|i| i * 3 + 1).collect();

    // Build the CSR graph.
    let graph = csr_from_arcs(node_count, &tail, &head, &weights);

    println!(
        "Graph: {} nodes, {} arcs",
        graph.node_count(),
        graph.arc_count()
    );

    // ----------------------------------------------------------------
    // 2. Compute the contraction order.
    // ----------------------------------------------------------------
    let order = cch::degree_order(&graph);
    println!("Contraction order (degree-sorted): {order:?}");

    // ----------------------------------------------------------------
    // 3. Build the CCH structure and customize the metric.
    // ----------------------------------------------------------------
    let cch = cch::Cch::build(&graph, &order);
    println!(
        "CCH built: {} CCH arcs ({}× expansion over {} input arcs)",
        cch.cch_arc_count(),
        cch.cch_arc_count(),
        graph.arc_count(),
    );

    let metric = cch.customize(&weights);

    // ----------------------------------------------------------------
    // 4. Save bundles to a temp directory, then reopen zero-copy.
    // ----------------------------------------------------------------
    let dir = std::env::temp_dir().join("cch_build_and_query_example");
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let struct_path = dir.join("example.cch-struct");
    let metric_path = dir.join("example.cch-metric");

    cch.save_struct(&struct_path)
        .expect("Cch::save_struct failed");
    metric.save(&metric_path).expect("Metric::save failed");

    println!("Saved bundles to {}", dir.display());

    // Reopen via mmap (zero-copy).
    let cch_bundle = cch::CchBundle::open(&struct_path).expect("CchBundle::open failed");
    let metric_bundle = cch::MetricBundle::open(&metric_path).expect("MetricBundle::open failed");
    let cv = cch_bundle.view();
    let mv = metric_bundle.view();

    println!(
        "Reopened: {} nodes, {} CCH arcs",
        cv.node_count(),
        cv.cch_arc_count()
    );

    // ----------------------------------------------------------------
    // 5. Distance matrix: all 9 × 9 pairs.
    // ----------------------------------------------------------------
    let all_nodes: Vec<u32> = (0..node_count).collect();
    let matrix = cch::distance_matrix(&cv, &mv, &all_nodes, &all_nodes);

    let n = node_count as usize;
    println!("\nDistance matrix ({n}×{n}):");
    let header = (0..n).fold(String::new(), |mut s, t| {
        let _ = write!(s, "{t:>7}");
        s
    });
    println!("        {header}");
    for (i, &s) in all_nodes.iter().enumerate() {
        let row: String = (0..n)
            .map(|j| {
                let d = matrix[i * n + j];
                if d == cch::INF_WEIGHT {
                    format!("{:>7}", "inf")
                } else {
                    format!("{d:>7}")
                }
            })
            .collect();
        println!("  src {s:>2}: {row}");
    }

    // ----------------------------------------------------------------
    // 6. Node path: source 0 → target 8 (top-left to bottom-right).
    // ----------------------------------------------------------------
    let (src, tgt) = (0u32, 8u32);
    print!("\nNode path {src}→{tgt}: ");
    match cch::node_path(&cv, &mv, src, tgt) {
        Some(path) => {
            let dist = matrix[src as usize * n + tgt as usize];
            println!("{path:?}  (distance = {dist})");
        }
        None => println!("unreachable"),
    }

    // Also show a self-pair (always returns Some([src])).
    let self_path = cch::node_path(&cv, &mv, 4, 4).expect("self-path always Some");
    println!("Node path 4→4 (self): {self_path:?}");
}

// ----------------------------------------------------------------
// Helper: build a CSR Graph from (tail, head, weight) arc lists
// grouped by tail (arcs for tail t appear before arcs for tail t+1).
// ----------------------------------------------------------------
fn csr_from_arcs(node_count: u32, tail: &[u32], head: &[u32], weight: &[u32]) -> cch::graph::Graph {
    assert_eq!(tail.len(), head.len());
    assert_eq!(tail.len(), weight.len());
    let n = node_count as usize;
    let mut degree = vec![0u32; n];
    for &t in tail {
        degree[t as usize] += 1;
    }
    let mut first_out = vec![0u32; n + 1];
    for v in 0..n {
        first_out[v + 1] = first_out[v] + degree[v];
    }
    let mut next = first_out[..n].to_vec();
    let mut g_head = vec![0u32; head.len()];
    let mut g_weight = vec![0u32; weight.len()];
    for ((&t, &h), &w) in tail.iter().zip(head.iter()).zip(weight.iter()) {
        let slot = next[t as usize] as usize;
        g_head[slot] = h;
        g_weight[slot] = w;
        next[t as usize] += 1;
    }
    cch::graph::Graph {
        first_out,
        head: g_head,
        weight: g_weight,
    }
}

//! Differential test: `cch::Cch::build` must be BIT-IDENTICAL to the C++
//! `CustomizableContractionHierarchy` constructor (`RoutingKit`), given the same
//! graph + the same contraction order.
//!
//! For each fixture we:
//!   1. build the C++ CCH via the oracle (`cch_new`) from `(tail, head)` + `order`,
//!   2. `cch_save_struct` into a tempdir, re-open via `CchBundle::open`, `.view()`,
//!   3. build the Rust CCH via `Cch::build(&graph, &order)` from the SAME arc
//!      multiset, and
//!   4. assert all 7 bundle arrays are bit-identical.

use cch::Cch;
use cch::graph::Graph;
use routingkit_cch::ffi;

/// Build a CSR `Graph` from a directed arc multiset `(tail, head)` with
/// `node_count` nodes. Weights are filled with 1 (irrelevant for structure).
/// The arcs are grouped by tail in the SAME relative order they appear in the
/// input lists (CSR stores them per-tail; within a tail the original order is
/// preserved). `Cch::build` derives `tail`/`head` back from this CSR, so the
/// arc multiset fed to the oracle and to Rust is identical.
fn csr_from_arcs(node_count: u32, tail: &[u32], head: &[u32]) -> Graph {
    assert_eq!(tail.len(), head.len());
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
    for (&t, &h) in tail.iter().zip(head.iter()) {
        let slot = next[t as usize] as usize;
        g_head[slot] = h;
        next[t as usize] += 1;
    }
    let weight = vec![1u32; head.len()];
    Graph {
        first_out,
        head: g_head,
        weight,
    }
}

/// Run the full differential comparison for one fixture.
fn assert_bit_identical(name: &str, node_count: u32, tail: &[u32], head: &[u32], order: &[u32]) {
    let graph = csr_from_arcs(node_count, tail, head);

    // Oracle: build + save + reopen.
    let cch = unsafe { ffi::cch_new(order, tail, head, |_| {}, false) };
    let cch_ref = cch.as_ref().expect("cch_new returned null");
    let dir = tempfile::tempdir().expect("tempdir");
    let struct_path = dir.path().join("eq.cch-struct");
    unsafe {
        ffi::cch_save_struct(cch_ref, struct_path.to_str().unwrap()).expect("cch_save_struct");
    }
    let bundle = cch::bundle::CchBundle::open(&struct_path).expect("CchBundle::open");
    let view = bundle.view();

    // Rust.
    let c = Cch::build(&graph, order);

    assert_eq!(c.rank, view.rank, "[{name}] rank mismatch");
    assert_eq!(
        c.elimination_tree_parent, view.elimination_tree_parent,
        "[{name}] elimination_tree_parent mismatch"
    );
    assert_eq!(
        c.up_first_out, view.up_first_out,
        "[{name}] up_first_out mismatch"
    );
    assert_eq!(c.up_head, view.up_head, "[{name}] up_head mismatch");
    assert_eq!(
        c.down_first_out, view.down_first_out,
        "[{name}] down_first_out mismatch"
    );
    assert_eq!(c.down_head, view.down_head, "[{name}] down_head mismatch");
    assert_eq!(
        c.down_to_up, view.down_to_up,
        "[{name}] down_to_up mismatch"
    );

    // Sanity: accessors agree with the gate arrays.
    assert_eq!(c.node_count(), node_count as usize);
    assert_eq!(c.cch_arc_count(), c.up_head.len());
}

// ----------------------------------------------------------------------------
// Fixture 1: bidirectional path, identity order. No fill-in shortcuts beyond
// the input arcs themselves.
// ----------------------------------------------------------------------------
#[test]
fn fixture_path_identity_order() {
    let n = 5u32;
    let tail = vec![0, 1, 2, 3, 1, 2, 3, 4];
    let head = vec![1, 2, 3, 4, 0, 1, 2, 3];
    let order: Vec<u32> = (0..n).collect();
    assert_bit_identical("path_identity", n, &tail, &head, &order);
}

// ----------------------------------------------------------------------------
// Fixture 2: same path, non-trivial (reversed) order.
// ----------------------------------------------------------------------------
#[test]
fn fixture_path_reversed_order() {
    let n = 5u32;
    let tail = vec![0, 1, 2, 3, 1, 2, 3, 4];
    let head = vec![1, 2, 3, 4, 0, 1, 2, 3];
    let order: Vec<u32> = (0..n).rev().collect();
    assert_bit_identical("path_reversed", n, &tail, &head, &order);
}

// ----------------------------------------------------------------------------
// Fixture 3: a graph that forces fill-in shortcuts. A "star-ish" / dense middle
// where contracting a low-rank node creates new arcs among its higher neighbors.
// Order chosen so node 0 (connected to 1,2,3) is contracted first, generating
// shortcut arcs among {1,2,3}.
// ----------------------------------------------------------------------------
#[test]
fn fixture_fillin_shortcuts() {
    let n = 4u32;
    // Undirected edges (as directed pairs): 0-1, 0-2, 0-3.
    // Contracting 0 first creates fill among 1,2,3 (a triangle of shortcuts).
    let tail = vec![0, 1, 0, 2, 0, 3];
    let head = vec![1, 0, 2, 0, 3, 0];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    assert_bit_identical("fillin", n, &tail, &head, &order);
}

// ----------------------------------------------------------------------------
// Fixture 4: a 3x3 grid (undirected), non-identity order. Exercises real
// fill-in across a 2D structure.
// ----------------------------------------------------------------------------
#[test]
fn fixture_grid_3x3() {
    let cols = 3u32;
    let rows = 3u32;
    let n = cols * rows;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    let add = |a: u32, b: u32, tail: &mut Vec<u32>, head: &mut Vec<u32>| {
        tail.push(a);
        head.push(b);
        tail.push(b);
        head.push(a);
    };
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                add(v, v + 1, &mut tail, &mut head);
            }
            if r + 1 < rows {
                add(v, v + cols, &mut tail, &mut head);
            }
        }
    }
    // A non-identity order: corners last, center early-ish.
    let order: Vec<u32> = vec![4, 1, 3, 5, 7, 0, 2, 6, 8];
    assert_bit_identical("grid_3x3", n, &tail, &head, &order);
}

// ----------------------------------------------------------------------------
// Fixture 5: a directed cycle, identity order.
// ----------------------------------------------------------------------------
#[test]
fn fixture_cycle() {
    let n = 6u32;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for v in 0..n {
        tail.push(v);
        head.push((v + 1) % n);
    }
    let order: Vec<u32> = (0..n).collect();
    assert_bit_identical("cycle", n, &tail, &head, &order);
}

// ----------------------------------------------------------------------------
// Fixture 6: graph with multi-edges and a self-loop in the input (the C++
// symmetrizes and removes both). Non-identity order.
// ----------------------------------------------------------------------------
#[test]
fn fixture_multiedge_and_selfloop() {
    let n = 5u32;
    // duplicate 0->1, a self-loop 2->2, and assorted arcs.
    let tail = vec![0, 0, 1, 2, 2, 3, 4, 1];
    let head = vec![1, 1, 2, 2, 3, 4, 0, 3];
    let order: Vec<u32> = vec![2, 0, 4, 1, 3];
    assert_bit_identical("multiedge_selfloop", n, &tail, &head, &order);
}

// ----------------------------------------------------------------------------
// Fixture 7: empty graph (no arcs), several isolated nodes, identity order.
// ----------------------------------------------------------------------------
#[test]
fn fixture_empty_arcs() {
    let n = 4u32;
    let tail: Vec<u32> = vec![];
    let head: Vec<u32> = vec![];
    let order: Vec<u32> = (0..n).collect();
    assert_bit_identical("empty_arcs", n, &tail, &head, &order);
}

// ----------------------------------------------------------------------------
// Fixture 8: single node, no arcs.
// ----------------------------------------------------------------------------
#[test]
fn fixture_single_node() {
    let n = 1u32;
    let tail: Vec<u32> = vec![];
    let head: Vec<u32> = vec![];
    let order: Vec<u32> = vec![0];
    assert_bit_identical("single_node", n, &tail, &head, &order);
}

// ============================================================================
// Part B — metric customization differential test (THE GATE).
//
// For each fixture we customize a metric (per-input-arc weights) through the
// oracle (`cch_metric_new` + `cch_metric_customize` + `cch_save_metric`) and
// re-open it via `MetricBundle`, then compare against
// `Cch::build(&graph, &order).customize(&weights)`. Both `forward` and
// `backward` arrays must be BIT-IDENTICAL.
//
// IMPORTANT: `weights[i]` is the weight of INPUT arc `i`. `Cch::build` derives
// its input arcs from the CSR graph (grouped by tail, original relative order
// within a tail preserved). To keep arc id `i` meaning the *same* arc on both
// sides, every metric fixture passes `(tail, head)` already grouped by tail —
// then `csr_from_arcs` round-trips them in the identical order the oracle sees.
// ============================================================================

/// Customize on both sides and assert the metric arrays are bit-identical.
fn assert_metric_bit_identical(
    name: &str,
    node_count: u32,
    tail: &[u32],
    head: &[u32],
    order: &[u32],
    weights: &[u32],
) {
    assert_eq!(
        tail.len(),
        weights.len(),
        "[{name}] one weight per input arc"
    );
    let graph = csr_from_arcs(node_count, tail, head);

    // Oracle: build CCH, create+customize metric, save, reopen.
    let cch = unsafe { ffi::cch_new(order, tail, head, |_| {}, false) };
    let cch_ref = cch.as_ref().expect("cch_new returned null");
    let dir = tempfile::tempdir().expect("tempdir");
    let metric_path = dir.path().join("eq.cch-metric");
    let mut metric = unsafe { ffi::cch_metric_new(cch_ref, weights) };
    unsafe {
        ffi::cch_metric_customize(metric.as_mut().expect("metric pin"));
        ffi::cch_save_metric(
            metric.as_ref().expect("metric ref"),
            metric_path.to_str().unwrap(),
        )
        .expect("cch_save_metric");
    }
    let mbundle = cch::bundle::MetricBundle::open(&metric_path).expect("MetricBundle::open");
    let view = mbundle.view();

    // Rust.
    let c = Cch::build(&graph, order);
    let m = c.customize(weights);

    assert_eq!(m.forward, view.forward, "[{name}] forward mismatch");
    assert_eq!(m.backward, view.backward, "[{name}] backward mismatch");
}

// ----------------------------------------------------------------------------
// Metric fixture 1: bidirectional path, identity order, varied weights.
// (tail already grouped by tail.)
// ----------------------------------------------------------------------------
#[test]
fn metric_path_identity() {
    let n = 5u32;
    //          0→1 0→? 1→0 1→2 2→1 2→3 3→2 3→4 4→3
    let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
    let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
    let order: Vec<u32> = (0..n).collect();
    let weights = vec![10u32, 11, 20, 21, 30, 31, 40, 41];
    assert_metric_bit_identical("metric_path_identity", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 2: same path, non-trivial (reversed) order.
// ----------------------------------------------------------------------------
#[test]
fn metric_path_reversed() {
    let n = 5u32;
    let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
    let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
    let order: Vec<u32> = (0..n).rev().collect();
    let weights = vec![10u32, 11, 20, 21, 30, 31, 40, 41];
    assert_metric_bit_identical("metric_path_reversed", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 3: fill-in graph — shortcut weights must be computed by the
// lower-triangle relaxation. Asymmetric weights so forward != backward.
// ----------------------------------------------------------------------------
#[test]
fn metric_fillin() {
    let n = 4u32;
    // edges 0-1, 0-2, 0-3 (grouped by tail). Contracting 0 first fills {1,2,3}.
    let tail = vec![0u32, 0, 0, 1, 2, 3];
    let head = vec![1u32, 2, 3, 0, 0, 0];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    let weights = vec![5u32, 7, 9, 6, 8, 10];
    assert_metric_bit_identical("metric_fillin", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 4: PARALLEL arcs with DIFFERENT weights — exercises the
// parallel-arc min-combine (extra-arc) path in `reset`.
// ----------------------------------------------------------------------------
#[test]
fn metric_parallel_arcs() {
    let n = 4u32;
    // Two parallel 0→1 arcs (weights 50, 9) and two parallel 1→0 arcs (40, 8).
    // Plus a path so there is fill-in too.
    let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
    let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    let weights = vec![50u32, 9, 40, 8, 17, 18, 19, 20];
    assert_metric_bit_identical("metric_parallel_arcs", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 5: 3x3 grid, non-identity order, varied weights.
// ----------------------------------------------------------------------------
#[test]
fn metric_grid_3x3() {
    let cols = 3u32;
    let rows = 3u32;
    let n = cols * rows;
    // Build grouped-by-tail: iterate node, emit its incident arcs.
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                tail.push(v);
                head.push(v + 1);
            }
            if c > 0 {
                tail.push(v);
                head.push(v - 1);
            }
            if r + 1 < rows {
                tail.push(v);
                head.push(v + cols);
            }
            if r > 0 {
                tail.push(v);
                head.push(v - cols);
            }
        }
    }
    let order: Vec<u32> = vec![4, 1, 3, 5, 7, 0, 2, 6, 8];
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32).map(|i| (i + 1) * 3).collect();
    assert_metric_bit_identical("metric_grid_3x3", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 6: directed cycle (asymmetric — many unreachable/INF entries
// since back-direction arcs are absent), identity order.
// ----------------------------------------------------------------------------
#[test]
fn metric_cycle_directed() {
    let n = 6u32;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for v in 0..n {
        tail.push(v);
        head.push((v + 1) % n);
    }
    let order: Vec<u32> = (0..n).collect();
    let weights: Vec<u32> = vec![100, 200, 300, 400, 500, 600];
    assert_metric_bit_identical("metric_cycle_directed", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 7: graph with an explicit INF_WEIGHT input weight, so an
// unreachable arc participates in the saturating triangle arithmetic.
// ----------------------------------------------------------------------------
#[test]
fn metric_with_inf_weights() {
    let n = 4u32;
    let tail = vec![0u32, 0, 0, 1, 2, 3];
    let head = vec![1u32, 2, 3, 0, 0, 0];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    // Some arcs carry INF_WEIGHT (unreachable) to exercise saturating sums.
    let inf = cch::INF_WEIGHT;
    let weights = vec![5u32, inf, 9, inf, 8, 10];
    assert_metric_bit_identical("metric_with_inf", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 8: single arc only.
// ----------------------------------------------------------------------------
#[test]
fn metric_single_arc() {
    let n = 2u32;
    let tail = vec![0u32];
    let head = vec![1u32];
    let order: Vec<u32> = vec![0, 1];
    let weights = vec![42u32];
    assert_metric_bit_identical("metric_single_arc", n, &tail, &head, &order, &weights);
}

// ----------------------------------------------------------------------------
// Metric fixture 9: empty graph (no arcs) — degenerate; both arrays empty.
// ----------------------------------------------------------------------------
#[test]
fn metric_empty() {
    let n = 3u32;
    let tail: Vec<u32> = vec![];
    let head: Vec<u32> = vec![];
    let order: Vec<u32> = (0..n).collect();
    let weights: Vec<u32> = vec![];
    assert_metric_bit_identical("metric_empty", n, &tail, &head, &order, &weights);
}

// ============================================================================
// Part C — bundle WRITER parity (THE KILLER GATE).
//
// For each fixture: build via oracle and `cch_save_struct` to file A; build via
// Rust and `Cch::save_struct` to file B; assert the raw FILE BYTES are equal.
// Same for the metric. This single assertion validates the entire on-disk
// format (header, fixed sections, the 3 bitvectors, and the LOCAL-id-compressed
// mapping + extra CSR).
// ============================================================================

/// Assert that Rust's struct + metric bytes are byte-identical to the oracle's.
fn assert_writer_byte_identical(
    name: &str,
    node_count: u32,
    tail: &[u32],
    head: &[u32],
    order: &[u32],
    weights: &[u32],
) {
    let graph = csr_from_arcs(node_count, tail, head);
    let dir = tempfile::tempdir().expect("tempdir");

    // ---- Struct ----
    let cch = unsafe { ffi::cch_new(order, tail, head, |_| {}, false) };
    let cch_ref = cch.as_ref().expect("cch_new returned null");
    let oracle_struct = dir.path().join("oracle.cch-struct");
    unsafe {
        ffi::cch_save_struct(cch_ref, oracle_struct.to_str().unwrap()).expect("cch_save_struct");
    }

    let c = Cch::build(&graph, order);
    let rust_struct = dir.path().join("rust.cch-struct");
    c.save_struct(&rust_struct).expect("Cch::save_struct");

    let a = std::fs::read(&oracle_struct).expect("read oracle struct");
    let b = std::fs::read(&rust_struct).expect("read rust struct");
    assert_eq!(
        a.len(),
        b.len(),
        "[{name}] struct file length differs: oracle={} rust={}",
        a.len(),
        b.len()
    );
    if a != b {
        let first = a
            .iter()
            .zip(&b)
            .position(|(x, y)| x != y)
            .expect("lengths equal but content differs");
        panic!(
            "[{name}] struct bytes differ at offset {first}: oracle={:#04x} rust={:#04x}",
            a[first], b[first]
        );
    }

    // ---- Metric ----
    let mut metric = unsafe { ffi::cch_metric_new(cch_ref, weights) };
    let oracle_metric = dir.path().join("oracle.cch-metric");
    unsafe {
        ffi::cch_metric_customize(metric.as_mut().expect("metric pin"));
        ffi::cch_save_metric(
            metric.as_ref().expect("metric ref"),
            oracle_metric.to_str().unwrap(),
        )
        .expect("cch_save_metric");
    }

    let m = c.customize(weights);
    let rust_metric = dir.path().join("rust.cch-metric");
    m.save(&rust_metric).expect("Metric::save");

    let ma = std::fs::read(&oracle_metric).expect("read oracle metric");
    let mb = std::fs::read(&rust_metric).expect("read rust metric");
    assert_eq!(ma.len(), mb.len(), "[{name}] metric file length differs");
    if ma != mb {
        let first = ma
            .iter()
            .zip(&mb)
            .position(|(x, y)| x != y)
            .expect("lengths equal but content differs");
        panic!(
            "[{name}] metric bytes differ at offset {first}: oracle={:#04x} rust={:#04x}",
            ma[first], mb[first]
        );
    }
}

#[test]
fn writer_byte_identical_path_identity() {
    let n = 5u32;
    let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
    let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
    let order: Vec<u32> = (0..n).collect();
    let weights = vec![10u32, 11, 20, 21, 30, 31, 40, 41];
    assert_writer_byte_identical("path_identity", n, &tail, &head, &order, &weights);
}

#[test]
fn writer_byte_identical_path_reversed() {
    let n = 5u32;
    let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
    let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
    let order: Vec<u32> = (0..n).rev().collect();
    let weights = vec![10u32, 11, 20, 21, 30, 31, 40, 41];
    assert_writer_byte_identical("path_reversed", n, &tail, &head, &order, &weights);
}

#[test]
fn writer_byte_identical_fillin() {
    let n = 4u32;
    let tail = vec![0u32, 0, 0, 1, 2, 3];
    let head = vec![1u32, 2, 3, 0, 0, 0];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    let weights = vec![5u32, 7, 9, 6, 8, 10];
    assert_writer_byte_identical("fillin", n, &tail, &head, &order, &weights);
}

#[test]
fn writer_byte_identical_parallel_arcs() {
    let n = 4u32;
    let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
    let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    let weights = vec![50u32, 9, 40, 8, 17, 18, 19, 20];
    assert_writer_byte_identical("parallel_arcs", n, &tail, &head, &order, &weights);
}

#[test]
fn writer_byte_identical_grid_3x3() {
    let cols = 3u32;
    let rows = 3u32;
    let n = cols * rows;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                tail.push(v);
                head.push(v + 1);
            }
            if c > 0 {
                tail.push(v);
                head.push(v - 1);
            }
            if r + 1 < rows {
                tail.push(v);
                head.push(v + cols);
            }
            if r > 0 {
                tail.push(v);
                head.push(v - cols);
            }
        }
    }
    let order: Vec<u32> = vec![4, 1, 3, 5, 7, 0, 2, 6, 8];
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32).map(|i| (i + 1) * 3).collect();
    assert_writer_byte_identical("grid_3x3", n, &tail, &head, &order, &weights);
}

#[test]
fn writer_byte_identical_multiedge_selfloop() {
    let n = 5u32;
    let tail = vec![0u32, 0, 1, 1, 2, 2, 3, 4];
    let head = vec![1u32, 1, 2, 3, 2, 3, 4, 0];
    let order: Vec<u32> = vec![2, 0, 4, 1, 3];
    let weights = vec![3u32, 4, 5, 6, 7, 8, 9, 10];
    assert_writer_byte_identical("multiedge_selfloop", n, &tail, &head, &order, &weights);
}

#[test]
fn writer_byte_identical_empty_arcs() {
    let n = 4u32;
    let tail: Vec<u32> = vec![];
    let head: Vec<u32> = vec![];
    let order: Vec<u32> = (0..n).collect();
    let weights: Vec<u32> = vec![];
    assert_writer_byte_identical("empty_arcs", n, &tail, &head, &order, &weights);
}

#[test]
fn writer_byte_identical_single_node() {
    let n = 1u32;
    let tail: Vec<u32> = vec![];
    let head: Vec<u32> = vec![];
    let order: Vec<u32> = vec![0];
    let weights: Vec<u32> = vec![];
    assert_writer_byte_identical("single_node", n, &tail, &head, &order, &weights);
}

// ============================================================================
// Part D — pure-Rust round-trip: Rust write → Rust read (CchBundle/MetricBundle).
// ============================================================================

#[test]
fn writer_round_trip_grid() {
    let cols = 3u32;
    let rows = 3u32;
    let n = cols * rows;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                tail.push(v);
                head.push(v + 1);
            }
            if c > 0 {
                tail.push(v);
                head.push(v - 1);
            }
            if r + 1 < rows {
                tail.push(v);
                head.push(v + cols);
            }
            if r > 0 {
                tail.push(v);
                head.push(v - cols);
            }
        }
    }
    let order: Vec<u32> = vec![4, 1, 3, 5, 7, 0, 2, 6, 8];
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32).map(|i| (i + 1) * 3).collect();
    let graph = csr_from_arcs(n, &tail, &head);

    let c = Cch::build(&graph, &order);
    let m = c.customize(&weights);

    let dir = tempfile::tempdir().expect("tempdir");
    let struct_path = dir.path().join("rt.cch-struct");
    let metric_path = dir.path().join("rt.cch-metric");
    c.save_struct(&struct_path).expect("save_struct");
    m.save(&metric_path).expect("save metric");

    let bundle = cch::bundle::CchBundle::open(&struct_path).expect("open struct");
    let v = bundle.view();
    assert_eq!(v.rank, c.rank.as_slice(), "rank round-trip");
    assert_eq!(
        v.elimination_tree_parent,
        c.elimination_tree_parent.as_slice(),
        "elim round-trip"
    );
    assert_eq!(v.up_first_out, c.up_first_out.as_slice(), "up_first_out");
    assert_eq!(v.up_head, c.up_head.as_slice(), "up_head");
    assert_eq!(
        v.down_first_out,
        c.down_first_out.as_slice(),
        "down_first_out"
    );
    assert_eq!(v.down_head, c.down_head.as_slice(), "down_head");
    assert_eq!(v.down_to_up, c.down_to_up.as_slice(), "down_to_up");

    let mbundle = cch::bundle::MetricBundle::open(&metric_path).expect("open metric");
    let mv = mbundle.view();
    assert_eq!(mv.forward, m.forward.as_slice(), "forward round-trip");
    assert_eq!(mv.backward, m.backward.as_slice(), "backward round-trip");
}

// ============================================================================
// Part E — cross-compat: Rust-written files load in the C++ oracle, and a
// distance-matrix / node-path query on the loaded bundle matches the direct
// oracle result.
// ============================================================================

#[test]
fn writer_cross_compat_oracle_loads_rust_files() {
    // A grid with asymmetric weights so the query has a non-trivial answer.
    let cols = 3u32;
    let rows = 3u32;
    let n = cols * rows;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                tail.push(v);
                head.push(v + 1);
            }
            if c > 0 {
                tail.push(v);
                head.push(v - 1);
            }
            if r + 1 < rows {
                tail.push(v);
                head.push(v + cols);
            }
            if r > 0 {
                tail.push(v);
                head.push(v - cols);
            }
        }
    }
    let order: Vec<u32> = vec![4, 1, 3, 5, 7, 0, 2, 6, 8];
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32).map(|i| (i + 1) * 3 + 1).collect();
    let graph = csr_from_arcs(n, &tail, &head);

    // Rust build + write.
    let c = Cch::build(&graph, &order);
    let m = c.customize(&weights);
    let dir = tempfile::tempdir().expect("tempdir");
    let struct_path = dir.path().join("xc.cch-struct");
    let metric_path = dir.path().join("xc.cch-metric");
    c.save_struct(&struct_path).expect("save_struct");
    m.save(&metric_path).expect("save metric");

    // Oracle loads the Rust-written files.
    let loaded_cch = unsafe { ffi::cch_load_struct(struct_path.to_str().unwrap()) }
        .expect("oracle cch_load_struct on rust file");
    let loaded_cch_ref = loaded_cch.as_ref().expect("loaded cch null");
    let loaded_metric =
        unsafe { ffi::cch_load_metric(loaded_cch_ref, metric_path.to_str().unwrap()) }
            .expect("oracle cch_load_metric on rust file");
    let loaded_metric_ref = loaded_metric.as_ref().expect("loaded metric null");

    // Direct oracle reference (build + customize, no file round-trip).
    let direct_cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
    let direct_cch_ref = direct_cch.as_ref().expect("direct cch null");
    let mut direct_metric = unsafe { ffi::cch_metric_new(direct_cch_ref, &weights) };
    unsafe { ffi::cch_metric_customize(direct_metric.as_mut().expect("pin")) };
    let direct_metric_ref = direct_metric.as_ref().expect("direct metric null");

    let sources: Vec<u32> = (0..n).collect();
    let targets: Vec<u32> = (0..n).collect();
    let loaded_dm =
        unsafe { ffi::cch_compute_distance_matrix(loaded_metric_ref, &sources, &targets) };
    let direct_dm =
        unsafe { ffi::cch_compute_distance_matrix(direct_metric_ref, &sources, &targets) };
    assert_eq!(
        loaded_dm, direct_dm,
        "distance matrix from rust-written bundle must match direct oracle"
    );

    // Node path: query 0 -> 8 on both, must match.
    let loaded_q = unsafe { ffi::cch_query_new(loaded_metric_ref) };
    let mut loaded_q = loaded_q;
    let direct_q = unsafe { ffi::cch_query_new(direct_metric_ref) };
    let mut direct_q = direct_q;
    let loaded_path = unsafe {
        ffi::cch_query_add_source(loaded_q.as_mut().unwrap(), 0, 0);
        ffi::cch_query_add_target(loaded_q.as_mut().unwrap(), 8, 0);
        ffi::cch_query_run(loaded_q.as_mut().unwrap());
        ffi::cch_query_node_path(loaded_q.as_ref().unwrap())
    };
    let direct_path = unsafe {
        ffi::cch_query_add_source(direct_q.as_mut().unwrap(), 0, 0);
        ffi::cch_query_add_target(direct_q.as_mut().unwrap(), 8, 0);
        ffi::cch_query_run(direct_q.as_mut().unwrap());
        ffi::cch_query_node_path(direct_q.as_ref().unwrap())
    };
    assert_eq!(
        loaded_path, direct_path,
        "node path from rust-written bundle must match direct oracle"
    );
}

// ============================================================================
// Part F — end-to-end pure-Rust CCH == C++ oracle (THE HEADLINE GATE).
//
// For each fixture we run the FULL Rust pipeline:
//   degree_order → Cch::build → customize → save_struct/save →
//   CchBundle::open + MetricBundle::open → distance_matrix (all pairs) +
//   node_path (200 deterministic LCG pairs)
//
// And the FULL oracle pipeline:
//   cch_new (using the SAME Rust degree_order) → cch_metric_new + customize →
//   cch_compute_distance_matrix (all pairs) + cch_query_node_path (200 pairs)
//
// Assertions:
//   1. Full distance matrix equal (sentinel normalization: oracle u32::MAX ⇔
//      rust INF_WEIGHT).
//   2. All 200 node paths equal as Option<Vec<u32>> (empty oracle path ⇒ None).
//   3. For the large fixture: Rust-written files are byte-identical to the
//      oracle-written files.
// ============================================================================

/// Run the full end-to-end equivalence gate: pure-Rust pipeline equals the
/// C++ oracle on the same fixture.
///
/// The Rust `degree_order` is computed once and fed to BOTH pipelines, so the
/// comparison isolates build/customize/query logic, not order choice.
#[allow(clippy::cast_possible_truncation)] // node_count-derived pair indices fit u32
#[allow(clippy::too_many_lines)] // a faithful end-to-end gate; splitting obscures the flow
#[allow(clippy::many_single_char_names)] // conventional short names in CCH/LCG context
fn assert_e2e_pipeline_equal(
    name: &str,
    node_count: u32,
    tail: &[u32],
    head: &[u32],
    weights: &[u32],
    also_byte_identical: bool,
) {
    // ---- Shared: compute degree_order once, use on both sides. ----
    let graph = csr_from_arcs(node_count, tail, head);
    let order = cch::degree_order(&graph);

    let dir = tempfile::tempdir().expect("tempdir");

    // ---- Rust pipeline ----
    let rust_cch = cch::Cch::build(&graph, &order);
    let rust_met = rust_cch.customize(weights);

    let rust_struct_path = dir.path().join("rust_e2e.cch-struct");
    let rust_metric_path = dir.path().join("rust_e2e.cch-metric");
    rust_cch
        .save_struct(&rust_struct_path)
        .expect("Cch::save_struct");
    rust_met.save(&rust_metric_path).expect("Metric::save");

    let rust_cch_bundle = cch::bundle::CchBundle::open(&rust_struct_path).expect("CchBundle::open");
    let rust_met_bundle =
        cch::bundle::MetricBundle::open(&rust_metric_path).expect("MetricBundle::open");
    let cv = rust_cch_bundle.view();
    let mv = rust_met_bundle.view();

    // distance_matrix: all pairs
    let pair_n = node_count as usize;
    let all_nodes: Vec<u32> = (0..node_count).collect();
    let rust_dm = cch::distance_matrix(&cv, &mv, &all_nodes, &all_nodes);

    // node_path: 200 LCG pairs
    let mut rust_paths: Vec<Option<Vec<u32>>> = Vec::with_capacity(200);
    let mut lcg_seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..200 {
        lcg_seed = lcg_seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let src = ((lcg_seed >> 33) as u32) % node_count;
        lcg_seed = lcg_seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let tgt = ((lcg_seed >> 33) as u32) % node_count;
        rust_paths.push(cch::node_path(&cv, &mv, src, tgt));
    }

    // ---- Oracle pipeline (uses the same order!) ----
    let oracle_cch = unsafe { ffi::cch_new(&order, tail, head, |_| {}, false) };
    let oracle_cch_ref = oracle_cch.as_ref().expect("oracle cch_new null");

    let oracle_struct_path = dir.path().join("oracle_e2e.cch-struct");
    let oracle_metric_path = dir.path().join("oracle_e2e.cch-metric");

    let mut oracle_metric = unsafe { ffi::cch_metric_new(oracle_cch_ref, weights) };
    unsafe {
        ffi::cch_save_struct(oracle_cch_ref, oracle_struct_path.to_str().unwrap())
            .expect("cch_save_struct");
        ffi::cch_metric_customize(oracle_metric.as_mut().expect("metric pin"));
        ffi::cch_save_metric(
            oracle_metric.as_ref().expect("metric ref"),
            oracle_metric_path.to_str().unwrap(),
        )
        .expect("cch_save_metric");
    }
    let oracle_met_ref = oracle_metric.as_ref().expect("oracle metric ref");

    let oracle_dm =
        unsafe { ffi::cch_compute_distance_matrix(oracle_met_ref, &all_nodes, &all_nodes) };

    // ---- Assert distance matrix equal ----
    assert_eq!(
        oracle_dm.len(),
        rust_dm.len(),
        "[{name}] distance matrix length mismatch"
    );
    for k in 0..oracle_dm.len() {
        let ov = oracle_dm[k];
        let rv = rust_dm[k];
        let oracle_inf = ov == u32::MAX || ov == cch::INF_WEIGHT;
        let rust_inf = rv == cch::INF_WEIGHT;
        let row = k / pair_n;
        let col = k % pair_n;
        assert_eq!(
            oracle_inf, rust_inf,
            "[{name}] reachability mismatch at ({row},{col}): oracle={ov}, rust={rv}"
        );
        if !oracle_inf && !rust_inf {
            assert_eq!(
                ov, rv,
                "[{name}] distance mismatch at ({row},{col}): oracle={ov}, rust={rv}"
            );
        }
    }

    // ---- Assert 200 node paths equal ----
    let mut lcg_seed2: u64 = 0x9E37_79B9_7F4A_7C15;
    for (pair_idx, rust_path) in rust_paths.into_iter().enumerate() {
        lcg_seed2 = lcg_seed2
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let src = ((lcg_seed2 >> 33) as u32) % node_count;
        lcg_seed2 = lcg_seed2
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let tgt = ((lcg_seed2 >> 33) as u32) % node_count;

        let mut oracle_q = unsafe { ffi::cch_query_new(oracle_met_ref) };
        let oracle_path_vec: Vec<u32> = unsafe {
            ffi::cch_query_add_source(oracle_q.as_mut().unwrap(), src, 0);
            ffi::cch_query_add_target(oracle_q.as_mut().unwrap(), tgt, 0);
            ffi::cch_query_run(oracle_q.as_mut().unwrap());
            ffi::cch_query_node_path(oracle_q.as_ref().unwrap())
        };
        let oracle_path: Option<Vec<u32>> =
            (!oracle_path_vec.is_empty()).then_some(oracle_path_vec);

        assert_eq!(
            rust_path, oracle_path,
            "[{name}] path mismatch at pair {pair_idx} ({src}→{tgt})"
        );
    }

    // ---- Byte-identical writer check (for the larger fixture) ----
    if also_byte_identical {
        let oracle_struct_bytes = std::fs::read(&oracle_struct_path).expect("read oracle struct");
        let rust_struct_bytes = std::fs::read(&rust_struct_path).expect("read rust struct");
        assert_eq!(
            oracle_struct_bytes.len(),
            rust_struct_bytes.len(),
            "[{name}] struct byte lengths differ: oracle={} rust={}",
            oracle_struct_bytes.len(),
            rust_struct_bytes.len()
        );
        if oracle_struct_bytes != rust_struct_bytes {
            let first = oracle_struct_bytes
                .iter()
                .zip(&rust_struct_bytes)
                .position(|(x, y)| x != y)
                .expect("lengths equal but bytes differ");
            panic!(
                "[{name}] struct bytes differ at offset {first}: \
                 oracle={:#04x} rust={:#04x}",
                oracle_struct_bytes[first], rust_struct_bytes[first]
            );
        }
        let oracle_metric_bytes = std::fs::read(&oracle_metric_path).expect("read oracle metric");
        let rust_metric_bytes = std::fs::read(&rust_metric_path).expect("read rust metric");
        assert_eq!(
            oracle_metric_bytes.len(),
            rust_metric_bytes.len(),
            "[{name}] metric byte lengths differ"
        );
        if oracle_metric_bytes != rust_metric_bytes {
            let first = oracle_metric_bytes
                .iter()
                .zip(&rust_metric_bytes)
                .position(|(x, y)| x != y)
                .expect("lengths equal but bytes differ");
            panic!(
                "[{name}] metric bytes differ at offset {first}: \
                 oracle={:#04x} rust={:#04x}",
                oracle_metric_bytes[first], rust_metric_bytes[first]
            );
        }
    }
}

// ----------------------------------------------------------------------------
// E2E Fixture 1: bidirectional path, 5 nodes (small, fast).
// ----------------------------------------------------------------------------
#[test]
fn e2e_pipeline_path_5() {
    let n = 5u32;
    // Grouped by tail (required for csr_from_arcs round-trip identity).
    let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
    let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
    let weights = vec![10u32, 11, 20, 21, 30, 31, 40, 41];
    assert_e2e_pipeline_equal("path_5", n, &tail, &head, &weights, false);
}

// ----------------------------------------------------------------------------
// E2E Fixture 2: fill-in graph (star around node 0), 4 nodes.
// ----------------------------------------------------------------------------
#[test]
fn e2e_pipeline_fillin_4() {
    let n = 4u32;
    let tail = vec![0u32, 0, 0, 1, 2, 3];
    let head = vec![1u32, 2, 3, 0, 0, 0];
    let weights = vec![5u32, 7, 9, 6, 8, 10];
    assert_e2e_pipeline_equal("fillin_4", n, &tail, &head, &weights, false);
}

// ----------------------------------------------------------------------------
// E2E Fixture 3: 3x3 grid, varied weights (small, exercises fill-in).
// ----------------------------------------------------------------------------
#[test]
fn e2e_pipeline_grid_3x3() {
    let cols = 3u32;
    let rows = 3u32;
    let n = cols * rows;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                tail.push(v);
                head.push(v + 1);
            }
            if c > 0 {
                tail.push(v);
                head.push(v - 1);
            }
            if r + 1 < rows {
                tail.push(v);
                head.push(v + cols);
            }
            if r > 0 {
                tail.push(v);
                head.push(v - cols);
            }
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32).map(|i| (i + 1) * 3).collect();
    assert_e2e_pipeline_equal("grid_3x3", n, &tail, &head, &weights, false);
}

// ----------------------------------------------------------------------------
// E2E Fixture 4: 24×24 bidirectional grid (≈576 nodes, CCH arc count > 512).
//
// This fixture exercises the 512-bit bitvector block boundary in the writer
// end-to-end. Both the byte-identical struct/metric comparison and the full
// distance-matrix + 200-path gate are run.
// ----------------------------------------------------------------------------
#[test]
fn e2e_pipeline_grid_24x24_large() {
    let cols = 24u32;
    let rows = 24u32;
    let n = cols * rows; // 576 nodes
    let mut tail = Vec::new();
    let mut head = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = r * cols + c;
            if c + 1 < cols {
                tail.push(v);
                head.push(v + 1);
            }
            if c > 0 {
                tail.push(v);
                head.push(v - 1);
            }
            if r + 1 < rows {
                tail.push(v);
                head.push(v + cols);
            }
            if r > 0 {
                tail.push(v);
                head.push(v - cols);
            }
        }
    }
    // Deterministic varied weights: arc index + 1, modulo a prime so we get
    // diverse values without overflow.
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32)
        .map(|i| (i * 7 + 1) % 9973 + 1)
        .collect();

    // Verify CCH arc count exceeds 512 (the 512-bit block boundary).
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);
    let c = cch::Cch::build(&graph, &order);
    assert!(
        c.cch_arc_count() > 512,
        "24x24 grid must have >512 CCH arcs (got {})",
        c.cch_arc_count()
    );

    // Drop c — assert_e2e_pipeline_equal rebuilds it internally.
    drop(c);

    assert_e2e_pipeline_equal("grid_24x24", n, &tail, &head, &weights, true);
}

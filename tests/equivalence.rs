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

//! Build the CCH of a bidirectional grid with a degree order, for measuring the
//! peak memory of `Cch::build`. Every tenth arc gets a parallel copy, so the
//! structure has extra input arcs.
//!
//! ```text
//! cargo build --release --example build_memory
//! /usr/bin/time -l target/release/examples/build_memory 1000 1000
//! ```

use cch::graph::Graph;
use cch::{Cch, degree_order};

fn grid(rows: u32, cols: u32) -> Graph {
    let node = |row: u32, col: u32| row * cols + col;
    let neighbors = |row: u32, col: u32| {
        [
            (col + 1 < cols).then(|| node(row, col + 1)),
            (col > 0).then(|| node(row, col - 1)),
            (row + 1 < rows).then(|| node(row + 1, col)),
            (row > 0).then(|| node(row - 1, col)),
        ]
        .into_iter()
        .flatten()
    };
    let heads_per_node: Vec<Vec<u32>> = (0..rows)
        .flat_map(|row| (0..cols).map(move |col| (row, col)))
        .map(|(row, col)| {
            neighbors(row, col)
                .flat_map(|head| {
                    let parallel = (head + row) % 10 == 0;

                    std::iter::repeat_n(head, if parallel { 2 } else { 1 })
                })
                .collect()
        })
        .collect();
    let first_out = std::iter::once(0)
        .chain(heads_per_node.iter().scan(0u32, |end, heads| {
            *end += u32::try_from(heads.len()).expect("arc count fits u32");

            Some(*end)
        }))
        .collect();
    let head: Vec<u32> = heads_per_node.into_iter().flatten().collect();

    Graph {
        first_out,
        weight: vec![1; head.len()],
        head,
    }
}

fn main() {
    let mut args = std::env::args()
        .skip(1)
        .map(|arg| arg.parse::<u32>().expect("grid size"));
    let rows = args.next().unwrap_or(1000);
    let cols = args.next().unwrap_or(rows);
    let graph = grid(rows, cols);
    let order = degree_order(&graph);
    let cch = Cch::build(&graph, &order);

    println!(
        "nodes {} input arcs {} cch arcs {} extra forward {} extra backward {}",
        cch.node_count(),
        graph.arc_count(),
        cch.cch_arc_count(),
        cch.extra_forward_input_arc_of_cch.len(),
        cch.extra_backward_input_arc_of_cch.len(),
    );
}

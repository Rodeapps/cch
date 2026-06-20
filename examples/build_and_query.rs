//! Open a pre-built CCH bundle and run distance-matrix + node-path queries.
//!
//! **Phase 1 — query/serve only.** This crate currently reads bundles and
//! answers queries; it cannot yet build bundles. Bundles (`.cch-struct` and
//! `.cch-metric`) must be pre-built by an external tool such as `RoutingKit` or
//! the rapidonkey engine. Phase 2 will add pure-Rust bundle construction.
//!
//! # Usage
//!
//! ```text
//! cargo run --example build_and_query -- <path/to/file.cch-struct> <path/to/file.cch-metric>
//! ```

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: {} <path/to/file.cch-struct> <path/to/file.cch-metric>",
            args[0]
        );
        eprintln!();
        eprintln!("Bundles must be pre-built by RoutingKit or the rapidonkey engine.");
        eprintln!("Phase 2 of this crate will add pure-Rust bundle construction.");
        std::process::exit(1);
    }

    let struct_path = Path::new(&args[1]);
    let metric_path = Path::new(&args[2]);

    // Open the struct bundle (zero-copy mmap).
    let cch_bundle = cch::CchBundle::open(struct_path).unwrap_or_else(|e| {
        eprintln!(
            "Failed to open struct bundle {}: {e}",
            struct_path.display()
        );
        std::process::exit(1);
    });

    // Open the metric bundle (zero-copy mmap).
    let metric_bundle = cch::MetricBundle::open(metric_path).unwrap_or_else(|e| {
        eprintln!(
            "Failed to open metric bundle {}: {e}",
            metric_path.display()
        );
        std::process::exit(1);
    });

    let cch_view = cch_bundle.view();
    let metric_view = metric_bundle.view();

    let node_count = cch_view.node_count();
    println!(
        "Opened bundles: {node_count} nodes, {} CCH arcs",
        cch_view.cch_arc_count()
    );

    if node_count == 0 {
        println!("Bundle is empty — nothing to query.");
        return;
    }

    // Pick a small set of sources and targets (up to the first 4 nodes).
    let query_count = node_count.min(4);
    let sources: Vec<u32> = (0..query_count).collect();
    let targets: Vec<u32> = (0..query_count).collect();

    // Distance matrix: sources x targets.
    let matrix = cch::distance_matrix(&cch_view, &metric_view, &sources, &targets);
    println!("\nDistance matrix ({query_count} x {query_count}):");
    let header = (0..query_count).fold(String::new(), |mut s, t| {
        use std::fmt::Write as _;
        let _ = write!(s, "{t:>8}");
        s
    });
    println!("         {header}");
    for (i, &s) in sources.iter().enumerate() {
        let row: String = (0..query_count as usize)
            .map(|j| {
                let d = matrix[i * query_count as usize + j];
                if d == cch::INF_WEIGHT {
                    format!("{:>8}", "inf")
                } else {
                    format!("{d:>8}")
                }
            })
            .collect();
        println!("  src {s:>3}: {row}");
    }

    // Node path for the first pair (source 0 → target 1, if distinct and n>1).
    if node_count >= 2 {
        let (src, tgt) = (0u32, 1u32);
        print!("\nNode path {src} -> {tgt}: ");
        match cch::node_path(&cch_view, &metric_view, src, tgt) {
            Some(path) => println!("{path:?}"),
            None => println!("unreachable"),
        }
    }
}

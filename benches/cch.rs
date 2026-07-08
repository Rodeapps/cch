//! Criterion benchmarks comparing the pure-Rust `cch` crate against the C++
//! `RoutingKit` oracle for each core operation.
//!
//! Inputs are deterministic (no randomness, no clock-based seeds).  All
//! graph-building and CCH-setup work is done OUTSIDE the timed closures; only
//! the operation under test is measured.
//!
//! Sample sizes are reduced via `BenchmarkConfig` to keep CI wall-time short;
//! the printed medians are still representative for a relative comparison.
//!
//! NOTE: the contraction order used here is `degree_order` (degree-ascending).
//! A production `RoutingKit` build would use inertial-flow ordering (a future
//! `cch` enhancement), which yields fewer shortcuts and therefore faster
//! queries; the ratios below reflect degree-order performance on both sides.

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use routingkit_cch::ffi;

// ---------------------------------------------------------------------------
// Graph construction helpers (mirrors `csr_from_arcs` in tests/equivalence.rs)
// ---------------------------------------------------------------------------

use cch::graph::Graph;

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

/// Build a bidirectional grid graph (rows × cols), grouped by tail so that
/// `csr_from_arcs` round-trips arcs in the same order as the oracle.
#[allow(
    clippy::cast_possible_truncation,
    reason = "node count is bounded to u32 by the CCH id space"
)]
fn make_grid(rows: u32, cols: u32) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = rows * cols;
    let mut tail = Vec::new();
    let mut head = Vec::new();
    // Emit grouped by tail node (required for CSR round-trip identity).
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
    // Deterministic varied weights: same formula used in e2e test fixture.
    let weights: Vec<u32> = (0..tail.len() as u32)
        .map(|i| (i * 7 + 1) % 9973 + 1)
        .collect();
    (n, tail, head, weights)
}

// ---------------------------------------------------------------------------
// Bench: degree_order
// ---------------------------------------------------------------------------

fn bench_degree_order(c: &mut Criterion) {
    let (n, tail, head, _) = make_grid(24, 24);
    let graph = csr_from_arcs(n, &tail, &head);

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("degree_order/24x24");
    g.sample_size(50);

    g.bench_function(BenchmarkId::new("rust", ""), |b| {
        b.iter(|| black_box(cch::degree_order(black_box(&graph))));
    });

    g.bench_function(BenchmarkId::new("cpp", ""), |b| {
        b.iter(|| {
            black_box(unsafe {
                ffi::cch_compute_order_degree(black_box(n), black_box(&tail), black_box(&head))
            })
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Bench: Cch::build  (structure construction only, order is pre-computed)
// ---------------------------------------------------------------------------

fn bench_build(c: &mut Criterion) {
    let (n, tail, head, _) = make_grid(24, 24);
    let graph = csr_from_arcs(n, &tail, &head);
    // Compute the order once, outside of all timed closures.
    let order = cch::degree_order(&graph);

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("build/24x24");
    g.sample_size(20);

    g.bench_function(BenchmarkId::new("rust", ""), |b| {
        b.iter(|| black_box(cch::Cch::build(black_box(&graph), black_box(&order))));
    });

    g.bench_function(BenchmarkId::new("cpp", ""), |b| {
        b.iter(|| {
            black_box(unsafe {
                ffi::cch_new(
                    black_box(&order),
                    black_box(&tail),
                    black_box(&head),
                    |_| {},
                    false,
                )
            })
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Bench: customize  (per-metric weight customization)
// ---------------------------------------------------------------------------

fn bench_customize(c: &mut Criterion) {
    let (n, tail, head, weights) = make_grid(24, 24);
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);

    // Rust: build the CCH structure once; bench only customize.
    let rust_cch = cch::Cch::build(&graph, &order);

    // C++: build the CCH structure once; bench only metric creation + customize.
    let cpp_cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
    let cpp_cch_ref = cpp_cch.as_ref().expect("cch_new returned null");

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("customize/24x24");
    g.sample_size(50);

    g.bench_function(BenchmarkId::new("rust", ""), |b| {
        b.iter(|| black_box(rust_cch.customize(black_box(&weights))));
    });

    g.bench_function(BenchmarkId::new("cpp", ""), |b| {
        b.iter(|| {
            let mut m = unsafe { ffi::cch_metric_new(cpp_cch_ref, black_box(&weights)) };
            unsafe { ffi::cch_metric_customize(m.as_mut().expect("metric pin")) };
            black_box(m)
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Bench: customize_reuse  (fresh Cch::customize per call vs a reused
// Customizer::customize_into, both on the Rust side only)
// ---------------------------------------------------------------------------

fn bench_customize_reuse(c: &mut Criterion) {
    let (n, tail, head, weights) = make_grid(24, 24);
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);
    let cch = cch::Cch::build(&graph, &order);

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("customize_reuse/24x24");
    g.sample_size(50);

    g.bench_function(BenchmarkId::new("fresh_each_call", ""), |b| {
        b.iter(|| {
            let m = cch.customize(black_box(&weights));
            black_box(m);
        });
    });

    g.bench_function(BenchmarkId::new("reused_customizer", ""), |b| {
        let cust = cch.customizer();
        let mut metric = cch.customize(&weights);
        b.iter(|| {
            cust.customize_into(black_box(&weights), &mut metric);
            black_box(&metric);
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Bench: distance_matrix  (all 576 nodes as sources AND targets = 576×576)
// ---------------------------------------------------------------------------

fn bench_distance_matrix(c: &mut Criterion) {
    let (n, tail, head, weights) = make_grid(24, 24);
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);

    // Rust: build + customize + mmap bundles.
    let rust_cch = cch::Cch::build(&graph, &order);
    let rust_met = rust_cch.customize(&weights);
    let tmp = tempfile::tempdir().expect("tempdir");
    let struct_path = tmp.path().join("bench.cch-struct");
    let metric_path = tmp.path().join("bench.cch-metric");
    rust_cch.save_struct(&struct_path).expect("save_struct");
    rust_met.save(&metric_path).expect("save metric");
    let rust_bundle = cch::bundle::CchBundle::open(&struct_path).expect("CchBundle::open");
    let rust_met_bundle =
        cch::bundle::MetricBundle::open(&metric_path).expect("MetricBundle::open");
    let cv = rust_bundle.view();
    let mv = rust_met_bundle.view();

    // C++: build + customize.
    let cpp_cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
    let cpp_cch_ref = cpp_cch.as_ref().expect("cch_new returned null");
    let mut cpp_metric = unsafe { ffi::cch_metric_new(cpp_cch_ref, &weights) };
    unsafe { ffi::cch_metric_customize(cpp_metric.as_mut().expect("metric pin")) };
    let cpp_met_ref = cpp_metric.as_ref().expect("metric ref");

    let all_nodes: Vec<u32> = (0..n).collect();

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("distance_matrix/24x24");
    g.sample_size(10);

    g.bench_function(BenchmarkId::new("rust", ""), |b| {
        b.iter(|| {
            black_box(cch::distance_matrix(
                black_box(&cv),
                black_box(&mv),
                black_box(&all_nodes),
                black_box(&all_nodes),
            ))
        });
    });

    g.bench_function(BenchmarkId::new("cpp", ""), |b| {
        b.iter(|| {
            black_box(unsafe {
                ffi::cch_compute_distance_matrix(
                    black_box(cpp_met_ref),
                    black_box(&all_nodes),
                    black_box(&all_nodes),
                )
            })
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Bench: node_path  (200 deterministic LCG pairs, same as e2e test)
// ---------------------------------------------------------------------------

fn bench_node_path(c: &mut Criterion) {
    let (n, tail, head, weights) = make_grid(24, 24);
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);

    // Rust: build + customize + mmap bundles.
    let rust_cch = cch::Cch::build(&graph, &order);
    let rust_met = rust_cch.customize(&weights);
    let tmp = tempfile::tempdir().expect("tempdir");
    let struct_path = tmp.path().join("bench_np.cch-struct");
    let metric_path = tmp.path().join("bench_np.cch-metric");
    rust_cch.save_struct(&struct_path).expect("save_struct");
    rust_met.save(&metric_path).expect("save metric");
    let rust_bundle = cch::bundle::CchBundle::open(&struct_path).expect("CchBundle::open");
    let rust_met_bundle =
        cch::bundle::MetricBundle::open(&metric_path).expect("MetricBundle::open");
    let cv = rust_bundle.view();
    let mv = rust_met_bundle.view();

    // C++: build + customize.
    let cpp_cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
    let cpp_cch_ref = cpp_cch.as_ref().expect("cch_new returned null");
    let mut cpp_metric = unsafe { ffi::cch_metric_new(cpp_cch_ref, &weights) };
    unsafe { ffi::cch_metric_customize(cpp_metric.as_mut().expect("metric pin")) };
    let cpp_met_ref = cpp_metric.as_ref().expect("metric ref");

    // Pre-generate 200 deterministic LCG pairs (same sequence as e2e test).
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(200);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..200 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let src = ((seed >> 33) as u32) % n;
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let tgt = ((seed >> 33) as u32) % n;
        pairs.push((src, tgt));
    }

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("node_path/24x24_200pairs");
    g.sample_size(20);

    g.bench_function(BenchmarkId::new("rust", ""), |b| {
        b.iter(|| {
            for &(src, tgt) in &pairs {
                black_box(cch::node_path(
                    black_box(&cv),
                    black_box(&mv),
                    black_box(src),
                    black_box(tgt),
                ));
            }
        });
    });

    // Pre-allocate ONE CCHQuery outside the timed loop; reset and reuse it for
    // each pair so the C++ side amortizes allocation just like the Rust side does.
    let mut cpp_query = unsafe { ffi::cch_query_new(cpp_met_ref) };

    g.bench_function(BenchmarkId::new("cpp", ""), |b| {
        b.iter(|| {
            for &(src, tgt) in &pairs {
                unsafe {
                    ffi::cch_query_reset(cpp_query.as_mut().unwrap(), cpp_met_ref);
                    ffi::cch_query_add_source(cpp_query.as_mut().unwrap(), src, 0);
                    ffi::cch_query_add_target(cpp_query.as_mut().unwrap(), tgt, 0);
                    ffi::cch_query_run(cpp_query.as_mut().unwrap());
                    black_box(ffi::cch_query_node_path(cpp_query.as_ref().unwrap()));
                }
            }
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Bench: path_query  (reused PathQuery vs amortized C++, same 200 LCG pairs)
// ---------------------------------------------------------------------------

fn bench_path_query(c: &mut Criterion) {
    let (n, tail, head, weights) = make_grid(24, 24);
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);

    // Rust: build + customize + mmap bundles.
    let rust_cch = cch::Cch::build(&graph, &order);
    let rust_met = rust_cch.customize(&weights);
    let tmp = tempfile::tempdir().expect("tempdir");
    let struct_path = tmp.path().join("bench_pq.cch-struct");
    let metric_path = tmp.path().join("bench_pq.cch-metric");
    rust_cch.save_struct(&struct_path).expect("save_struct");
    rust_met.save(&metric_path).expect("save metric");
    let rust_bundle = cch::bundle::CchBundle::open(&struct_path).expect("CchBundle::open");
    let rust_met_bundle =
        cch::bundle::MetricBundle::open(&metric_path).expect("MetricBundle::open");
    let cv = rust_bundle.view();
    let mv = rust_met_bundle.view();

    // C++: build + customize.
    let cpp_cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
    let cpp_cch_ref = cpp_cch.as_ref().expect("cch_new returned null");
    let mut cpp_metric = unsafe { ffi::cch_metric_new(cpp_cch_ref, &weights) };
    unsafe { ffi::cch_metric_customize(cpp_metric.as_mut().expect("metric pin")) };
    let cpp_met_ref = cpp_metric.as_ref().expect("metric ref");

    // Same 200 deterministic LCG pairs as the e2e test / node_path bench.
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(200);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..200 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let src = ((seed >> 33) as u32) % n;
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let tgt = ((seed >> 33) as u32) % n;
        pairs.push((src, tgt));
    }

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("path_query/24x24_200pairs");
    g.sample_size(20);

    // Rust: ONE reused PathQuery (buffers + order allocated once outside the
    // timed loop), answering all 200 pairs per iteration.
    let mut pq = cch::PathQuery::new(&cv);
    g.bench_function(BenchmarkId::new("rust", ""), |b| {
        b.iter(|| {
            for &(src, tgt) in &pairs {
                black_box(pq.path(black_box(&mv), black_box(src), black_box(tgt)));
            }
        });
    });

    // C++: one CCHQuery reset+reused per pair (amortized, identical to the
    // node_path/cpp bench).
    let mut cpp_query = unsafe { ffi::cch_query_new(cpp_met_ref) };
    g.bench_function(BenchmarkId::new("cpp", ""), |b| {
        b.iter(|| {
            for &(src, tgt) in &pairs {
                unsafe {
                    ffi::cch_query_reset(cpp_query.as_mut().unwrap(), cpp_met_ref);
                    ffi::cch_query_add_source(cpp_query.as_mut().unwrap(), src, 0);
                    ffi::cch_query_add_target(cpp_query.as_mut().unwrap(), tgt, 0);
                    ffi::cch_query_run(cpp_query.as_mut().unwrap());
                    black_box(ffi::cch_query_node_path(cpp_query.as_ref().unwrap()));
                }
            }
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Bench: path queries on a LARGER grid (64x64 ≈ 4096 nodes), where the
// per-call buffer allocation + O(n) `order` rebuild of the one-shot `node_path`
// is no longer negligible — so the reuse advantage of `PathQuery` shows.
// Compares one-shot node_path vs reused PathQuery vs amortized C++.
// ---------------------------------------------------------------------------

fn bench_path_query_large(c: &mut Criterion) {
    let (n, tail, head, weights) = make_grid(64, 64);
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);

    let rust_cch = cch::Cch::build(&graph, &order);
    let rust_met = rust_cch.customize(&weights);
    let tmp = tempfile::tempdir().expect("tempdir");
    let struct_path = tmp.path().join("bench_pql.cch-struct");
    let metric_path = tmp.path().join("bench_pql.cch-metric");
    rust_cch.save_struct(&struct_path).expect("save_struct");
    rust_met.save(&metric_path).expect("save metric");
    let rust_bundle = cch::bundle::CchBundle::open(&struct_path).expect("CchBundle::open");
    let rust_met_bundle =
        cch::bundle::MetricBundle::open(&metric_path).expect("MetricBundle::open");
    let cv = rust_bundle.view();
    let mv = rust_met_bundle.view();

    let cpp_cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
    let cpp_cch_ref = cpp_cch.as_ref().expect("cch_new returned null");
    let mut cpp_metric = unsafe { ffi::cch_metric_new(cpp_cch_ref, &weights) };
    unsafe { ffi::cch_metric_customize(cpp_metric.as_mut().expect("metric pin")) };
    let cpp_met_ref = cpp_metric.as_ref().expect("metric ref");

    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(200);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..200 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let src = ((seed >> 33) as u32) % n;
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let tgt = ((seed >> 33) as u32) % n;
        pairs.push((src, tgt));
    }

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("path_query/64x64_200pairs");
    g.sample_size(20);

    // One-shot: allocates ~6 n-sized buffers + rebuilds `order` per call.
    g.bench_function(BenchmarkId::new("node_path_oneshot", ""), |b| {
        b.iter(|| {
            for &(src, tgt) in &pairs {
                black_box(cch::node_path(
                    black_box(&cv),
                    black_box(&mv),
                    black_box(src),
                    black_box(tgt),
                ));
            }
        });
    });

    // Reused: buffers + `order` allocated once.
    let mut pq = cch::PathQuery::new(&cv);
    g.bench_function(BenchmarkId::new("path_query_reused", ""), |b| {
        b.iter(|| {
            for &(src, tgt) in &pairs {
                black_box(pq.path(black_box(&mv), black_box(src), black_box(tgt)));
            }
        });
    });

    let mut cpp_query = unsafe { ffi::cch_query_new(cpp_met_ref) };
    g.bench_function(BenchmarkId::new("cpp", ""), |b| {
        b.iter(|| {
            for &(src, tgt) in &pairs {
                unsafe {
                    ffi::cch_query_reset(cpp_query.as_mut().unwrap(), cpp_met_ref);
                    ffi::cch_query_add_source(cpp_query.as_mut().unwrap(), src, 0);
                    ffi::cch_query_add_target(cpp_query.as_mut().unwrap(), tgt, 0);
                    ffi::cch_query_run(cpp_query.as_mut().unwrap());
                    black_box(ffi::cch_query_node_path(cpp_query.as_ref().unwrap()));
                }
            }
        });
    });

    g.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_degree_order,
    bench_build,
    bench_customize,
    bench_customize_reuse,
    bench_distance_matrix,
    bench_node_path,
    bench_path_query,
    bench_path_query_large,
);
criterion_main!(benches);

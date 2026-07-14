//! Real-corpus benchmark (opt-in via env vars — nothing is committed).
//!
//! Measures parallel customize scaling and query latency on a genuine road
//! network in the crate's own bundle format. Point it at a corpus, e.g.:
//!
//!   `CCH_CORPUS`=~/workspace/osm-data/cch-artifacts/albania.cch-struct \
//!   `CCH_CORPUS_METRIC`=~/workspace/osm-data/cch-artifacts/albania.cch-metric-distance \
//!   cargo bench --bench `real_corpus`
//!
//! - `CCH_CORPUS` (a .cch-struct): enables the customize parallel-scaling bench.
//! - `CCH_CORPUS_METRIC` (a .cch-metric, optional): enables the query benches.
//!
//! When `CCH_CORPUS` is unset the whole bench registers no cases and exits.

use std::path::{Path, PathBuf};

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};

/// Tiny deterministic LCG so sampled node ids are stable across runs.
fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// A deterministic sample of `count` node ids in `0..node_count`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "node id < node_count fits u32"
)]
fn sample_nodes(node_count: u32, count: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..count)
        .map(|_| (lcg(&mut s) % u64::from(node_count)) as u32)
        .collect()
}

fn bench_customize_scaling(c: &mut Criterion, struct_path: &Path) {
    let cch = cch::Cch::load_struct(struct_path).expect("load_struct CCH_CORPUS");
    let weights = vec![1u32; cch.input_arc_to_cch_arc.len()];
    let n = cch.node_count();
    let arcs = cch.input_arc_to_cch_arc.len();

    let max_threads = rayon::current_num_threads();
    let thread_counts: Vec<usize> = if max_threads > 1 {
        vec![1, max_threads]
    } else {
        vec![1]
    };

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("real_corpus/customize");
    g.sample_size(10);
    println!("real_corpus: {n} nodes, {arcs} input arcs");
    for &threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build rayon pool");
        g.bench_function(BenchmarkId::new("threads", threads), |b| {
            b.iter(|| pool.install(|| black_box(cch.customize(black_box(&weights)))));
        });
    }
    g.finish();
}

#[allow(clippy::many_single_char_names)] // s,t,n,i,c,g: conventional short names for a query bench
fn bench_queries(c: &mut Criterion, struct_path: &Path, metric_path: &Path) {
    let bundle = cch::CchBundle::open(struct_path).expect("open CCH_CORPUS");
    let metric = cch::MetricBundle::open(metric_path).expect("open CCH_CORPUS_METRIC");
    let cv = bundle.view();
    let mv = metric.view();
    let n = cv.node_count();

    let srcs = sample_nodes(n, 200, 0x1234_5678);
    let tgts = sample_nodes(n, 1_000, 0x9abc_def0);
    let matrix_nodes = sample_nodes(n, 100, 0x0f0f_0f0f);

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("real_corpus/query");
    g.sample_size(20);

    g.bench_function("distance", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let s = srcs[i % srcs.len()];
            let t = tgts[i % tgts.len()];
            i += 1;
            black_box(cch::distance(black_box(&cv), black_box(&mv), s, t))
        });
    });

    g.bench_function("distances_from_1k", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let s = srcs[i % srcs.len()];
            i += 1;
            black_box(cch::distances_from(
                black_box(&cv),
                black_box(&mv),
                s,
                black_box(&tgts),
            ))
        });
    });

    g.bench_function("distance_matrix_100x100", |b| {
        b.iter(|| {
            black_box(cch::distance_matrix(
                black_box(&cv),
                black_box(&mv),
                black_box(&matrix_nodes),
                black_box(&matrix_nodes),
            ))
        });
    });

    g.bench_function("node_path", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let s = srcs[i % srcs.len()];
            let t = tgts[i % tgts.len()];
            i += 1;
            black_box(cch::node_path(black_box(&cv), black_box(&mv), s, t))
        });
    });

    g.finish();
}

fn real_corpus(c: &mut Criterion) {
    let Some(struct_path) = std::env::var_os("CCH_CORPUS").map(PathBuf::from) else {
        println!("real_corpus: CCH_CORPUS unset — skipping (see file header to enable).");
        return;
    };
    bench_customize_scaling(c, &struct_path);

    match std::env::var_os("CCH_CORPUS_METRIC").map(PathBuf::from) {
        Some(metric_path) => bench_queries(c, &struct_path, &metric_path),
        None => println!("real_corpus: CCH_CORPUS_METRIC unset — skipping query benches."),
    }
}

criterion_group!(benches, real_corpus);
criterion_main!(benches);

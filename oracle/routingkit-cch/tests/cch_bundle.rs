//! Phase 2a — CCH bundle roundtrip tests.
//!
//! Validates that save_struct/load_struct and save_metric/load_metric
//! produce a CCH+metric that returns IDENTICAL distance-matrix results to
//! the original in-memory CCH/metric. Uses a tiny synthetic 10-node graph
//! so the test runs in <1 s and doesn't need shapefile fixtures.

use routingkit_cch::{ffi, CCHMetric, CCH};
use tempfile::TempDir;

/// Build a tiny 10-node directed graph:
/// nodes 0..10, arcs 0→1, 1→2, ..., 8→9 plus 9→0 (a cycle), and a
/// shortcut 0→5 with weight 100 vs going around (sum = 50).
fn tiny_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut tail = Vec::new();
    let mut head = Vec::new();
    let mut weight = Vec::new();
    for i in 0u32..9 {
        tail.push(i);
        head.push(i + 1);
        weight.push(10);
    }
    tail.push(9); head.push(0); weight.push(10);
    tail.push(0); head.push(5); weight.push(100); // long shortcut
    (tail, head, weight)
}

#[test]
fn roundtrip_struct_only() {
    let (tail, head, _weight) = tiny_graph();
    let order: Vec<u32> = (0u32..10).collect();
    let cch = CCH::new(&order, &tail, &head, |_| {}, false);

    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("test.cch-struct");
    let path_str = path.to_str().unwrap();

    // Save + reload.
    unsafe { ffi::cch_save_struct(cch.inner_ref(), path_str) }
        .expect("save_struct");
    let _loaded = unsafe { ffi::cch_load_struct(path_str) }
        .expect("load_struct");
    // Just verify it loads without error; field-equivalence is implicit
    // in the query equivalence test below.
}

#[test]
fn roundtrip_struct_and_metric_query_equivalent() {
    let (tail, head, weight) = tiny_graph();
    let order: Vec<u32> = (0u32..10).collect();
    let cch_orig = CCH::new(&order, &tail, &head, |_| {}, false);
    let metric_orig = CCHMetric::new(&cch_orig, weight.clone());

    let tmp = TempDir::new().expect("tempdir");
    let struct_path = tmp.path().join("test.cch-struct");
    let metric_path = tmp.path().join("test.cch-metric-distance");

    unsafe {
        ffi::cch_save_struct(cch_orig.inner_ref(), struct_path.to_str().unwrap())
            .expect("save_struct");
        ffi::cch_save_metric(metric_orig.inner_ref(), metric_path.to_str().unwrap())
            .expect("save_metric");
    }

    let cch_loaded_ptr = unsafe {
        ffi::cch_load_struct(struct_path.to_str().unwrap()).expect("load_struct")
    };
    let cch_loaded = CCH::from_unique_ptr(cch_loaded_ptr);
    let metric_loaded_ptr = unsafe {
        ffi::cch_load_metric(cch_loaded.inner_ref(), metric_path.to_str().unwrap())
            .expect("load_metric")
    };
    let metric_loaded = CCHMetric::from_unique_ptr(metric_loaded_ptr, &cch_loaded);

    // Run distance matrix between all 10 nodes via both metrics.
    let nodes: Vec<u32> = (0..10).collect();
    let m_orig = routingkit_cch::distance_matrix(&metric_orig, &nodes, &nodes);
    let m_loaded = routingkit_cch::distance_matrix(&metric_loaded, &nodes, &nodes);

    assert_eq!(m_orig, m_loaded, "loaded-bundle metric must produce identical distances");
}

#[test]
fn load_struct_bad_magic_errors() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("garbage.cch-struct");
    std::fs::write(&path, b"\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
    let result = unsafe { ffi::cch_load_struct(path.to_str().unwrap()) };
    assert!(result.is_err(), "bad magic must error");
}

#[test]
fn load_metric_arc_count_mismatch_errors() {
    let (tail, head, weight) = tiny_graph();
    let order: Vec<u32> = (0u32..10).collect();
    let cch = CCH::new(&order, &tail, &head, |_| {}, false);
    let metric = CCHMetric::new(&cch, weight.clone());

    let tmp = TempDir::new().expect("tempdir");
    let metric_path = tmp.path().join("test.cch-metric-distance");
    unsafe {
        ffi::cch_save_metric(metric.inner_ref(), metric_path.to_str().unwrap())
            .expect("save_metric");
    }

    // Build a DIFFERENT CCH with a smaller graph.
    let small_tail = vec![0u32, 1];
    let small_head = vec![1u32, 0];
    let small_order = vec![0u32, 1];
    let small_cch = CCH::new(&small_order, &small_tail, &small_head, |_| {}, false);

    let result = unsafe {
        ffi::cch_load_metric(small_cch.inner_ref(), metric_path.to_str().unwrap())
    };
    assert!(result.is_err(), "arc-count mismatch must error");
}

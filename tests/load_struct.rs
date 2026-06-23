//! Gate for `Cch::load_struct` — the eager full-`.cch-struct` reader that
//! reconstructs a RE-CUSTOMIZABLE `Cch`.
//!
//! The strongest gate here is ORACLE CROSS-COMPAT: take an ORACLE-written
//! `.cch-struct` (`ffi::cch_new` → `ffi::cch_save_struct`), load it with the
//! pure-Rust `Cch::load_struct`, `customize(&weights)`, and assert the metric
//! is BIT-IDENTICAL to the oracle's own metric (`cch_metric_new` +
//! `cch_metric_customize` → `cch_save_metric` → `MetricBundle::open`). This
//! proves `load_struct` correctly reads the real C++-written format, including
//! its LOCAL-id-compressed input-arc mapping + the 3 bitvectors, and inverts it
//! into the FULL-SIZE representation `customize` consumes.

use cch::Cch;
use cch::bundle::MetricBundle;
use cch::graph::Graph;
use routingkit_cch::ffi;

/// Build a CSR `Graph` from a directed arc multiset grouped by tail. Mirrors
/// the helper in `tests/equivalence.rs` so input-arc ids align with the oracle.
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
    Graph {
        first_out,
        head: g_head,
        weight: vec![1u32; head.len()],
    }
}

/// ORACLE CROSS-COMPAT gate for one fixture:
///   1. oracle `cch_new` → `cch_save_struct` to file,
///   2. pure-Rust `Cch::load_struct` on the oracle file,
///   3. `loaded.customize(weights)`,
///   4. oracle `cch_metric_new` + `customize` → `cch_save_metric` →
///      `MetricBundle::open`,
///   5. assert forward/backward weights are BIT-IDENTICAL.
fn assert_oracle_cross_compat(
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

    let cch = unsafe { ffi::cch_new(order, tail, head, |_| {}, false) };
    let cch_ref = cch.as_ref().expect("cch_new returned null");
    let dir = tempfile::tempdir().expect("tempdir");
    let struct_path = dir.path().join("oracle.cch-struct");
    unsafe {
        ffi::cch_save_struct(cch_ref, struct_path.to_str().unwrap()).expect("cch_save_struct");
    }

    // Load the ORACLE-written struct with the pure-Rust eager reader.
    let loaded = Cch::load_struct(&struct_path).expect("Cch::load_struct on oracle file");
    let rust_metric = loaded.customize(weights);

    // Oracle metric for the same weights.
    let metric_path = dir.path().join("oracle.cch-metric");
    let mut metric = unsafe { ffi::cch_metric_new(cch_ref, weights) };
    unsafe {
        ffi::cch_metric_customize(metric.as_mut().expect("metric pin"));
        ffi::cch_save_metric(
            metric.as_ref().expect("metric ref"),
            metric_path.to_str().unwrap(),
        )
        .expect("cch_save_metric");
    }
    let mbundle = MetricBundle::open(&metric_path).expect("MetricBundle::open");
    let mv = mbundle.view();

    assert_eq!(
        rust_metric.forward, mv.forward,
        "[{name}] forward metric mismatch (oracle struct → rust load → rust customize)"
    );
    assert_eq!(
        rust_metric.backward, mv.backward,
        "[{name}] backward metric mismatch (oracle struct → rust load → rust customize)"
    );

    // Bonus: the loaded `Cch`'s structural arrays must match a freshly-built
    // Rust `Cch` for the same graph + order (the file came from the oracle, but
    // structure is bit-identical across both pipelines).
    let fresh = Cch::build(&csr_from_arcs(node_count, tail, head), order);
    assert_eq!(loaded.rank, fresh.rank, "[{name}] rank");
    assert_eq!(loaded.up_head, fresh.up_head, "[{name}] up_head");
    assert_eq!(loaded.down_to_up, fresh.down_to_up, "[{name}] down_to_up");
    assert_eq!(
        loaded.input_arc_to_cch_arc, fresh.input_arc_to_cch_arc,
        "[{name}] input_arc_to_cch_arc"
    );
}

#[test]
fn oracle_cross_compat_path_identity() {
    let n = 5u32;
    let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
    let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
    let order: Vec<u32> = (0..n).collect();
    let weights = vec![10u32, 11, 20, 21, 30, 31, 40, 41];
    assert_oracle_cross_compat("path_identity", n, &tail, &head, &order, &weights);
}

#[test]
fn oracle_cross_compat_path_reversed() {
    let n = 5u32;
    let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
    let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
    let order: Vec<u32> = (0..n).rev().collect();
    let weights = vec![10u32, 11, 20, 21, 30, 31, 40, 41];
    assert_oracle_cross_compat("path_reversed", n, &tail, &head, &order, &weights);
}

#[test]
fn oracle_cross_compat_fillin() {
    let n = 4u32;
    let tail = vec![0u32, 0, 0, 1, 2, 3];
    let head = vec![1u32, 2, 3, 0, 0, 0];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    let weights = vec![5u32, 7, 9, 6, 8, 10];
    assert_oracle_cross_compat("fillin", n, &tail, &head, &order, &weights);
}

#[test]
fn oracle_cross_compat_parallel_arcs() {
    // Parallel arcs → non-empty extra (overflow) lists, the LOCAL-compressed
    // extra CSR that load_struct must invert.
    let n = 4u32;
    let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
    let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    let weights = vec![50u32, 9, 40, 8, 17, 18, 19, 20];
    assert_oracle_cross_compat("parallel_arcs", n, &tail, &head, &order, &weights);
}

#[test]
fn oracle_cross_compat_with_inf_weights() {
    let n = 4u32;
    let tail = vec![0u32, 0, 0, 1, 2, 3];
    let head = vec![1u32, 2, 3, 0, 0, 0];
    let order: Vec<u32> = vec![0, 1, 2, 3];
    let inf = cch::INF_WEIGHT;
    let weights = vec![5u32, inf, 9, inf, 8, 10];
    assert_oracle_cross_compat("with_inf", n, &tail, &head, &order, &weights);
}

#[test]
fn oracle_cross_compat_grid_block_boundary() {
    // 24x24 grid → cch_arc_count > 512, exercising the bitvector 512-bit block
    // padding in the oracle-written file.
    let cols = 24u32;
    let rows = 24u32;
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
    let graph = csr_from_arcs(n, &tail, &head);
    let order = cch::degree_order(&graph);
    let c = Cch::build(&graph, &order);
    assert!(c.cch_arc_count() > 512, "grid must exceed 512 CCH arcs");
    #[allow(clippy::cast_possible_truncation)]
    let weights: Vec<u32> = (0..tail.len() as u32)
        .map(|i| (i * 7 + 1) % 9973 + 1)
        .collect();
    assert_oracle_cross_compat("grid_24x24", n, &tail, &head, &order, &weights);
}

#[test]
fn oracle_cross_compat_empty_arcs() {
    let n = 4u32;
    let tail: Vec<u32> = vec![];
    let head: Vec<u32> = vec![];
    let order: Vec<u32> = (0..n).collect();
    let weights: Vec<u32> = vec![];
    assert_oracle_cross_compat("empty_arcs", n, &tail, &head, &order, &weights);
}

/// Round-trip via the FILESYSTEM (not just in-memory): Rust build →
/// `save_struct` to a file → `load_struct` → re-customize bit-identical.
#[test]
fn rust_file_round_trip_recustomize() {
    let n = 4u32;
    let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
    let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
    let order: Vec<u32> = vec![2, 0, 3, 1];
    let weights = vec![50u32, 9, 40, 8, 17, 18, 19, 20];
    let graph = csr_from_arcs(n, &tail, &head);
    let original = Cch::build(&graph, &order);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("rt.cch-struct");
    original.save_struct(&path).expect("save_struct");
    let loaded = Cch::load_struct(&path).expect("load_struct");

    let want = original.customize(&weights);
    let got = loaded.customize(&weights);
    assert_eq!(want.forward, got.forward, "forward re-customize");
    assert_eq!(want.backward, got.backward, "backward re-customize");
}

/// `load_struct` on a missing file returns an `io::Error` (not a panic).
#[test]
fn load_struct_missing_file_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("does-not-exist.cch-struct");
    let err = Cch::load_struct(&path).map(|_| ()).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

// ============================================================================
// Corrupt-input robustness through the PUBLIC `load_struct(path)` API. These
// exercise the file-backed error paths (so they run in the integration-test
// build of the crate, not only the unit-test build): bad magic, truncation,
// inflated counts (overflow), and a corrupt local-compressed extra-CSR.
// ============================================================================

/// A valid Rust-written `.cch-struct` byte buffer from a parallel-arc fixture
/// (non-empty bitvectors + local arrays + extra CSR).
fn valid_struct_bytes() -> Vec<u8> {
    let n = 4u32;
    let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
    let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
    let c = Cch::build(&csr_from_arcs(n, &tail, &head), &[0, 1, 2, 3]);
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("v.cch-struct");
    c.save_struct(&path).expect("save_struct");
    std::fs::read(&path).expect("read back")
}

fn read_u64_usize(b: &[u8], o: usize) -> usize {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[o..o + 8]);
    usize::try_from(u64::from_le_bytes(buf)).expect("fits")
}

/// Start byte offset (length-prefix position) of each of the 19 sections.
fn section_start_offsets(b: &[u8]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(19);
    let mut pos = 40usize;
    for _ in 0..10 {
        starts.push(pos);
        pos += 8 + read_u64_usize(b, pos);
    }
    for _ in 0..3 {
        starts.push(pos);
        pos += 16 + read_u64_usize(b, pos + 8);
    }
    for _ in 0..6 {
        starts.push(pos);
        pos += 8 + read_u64_usize(b, pos);
    }
    starts
}

fn write_and_load(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::io::Error {
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write corrupt file");
    Cch::load_struct(&path).map(|_| ()).unwrap_err()
}

#[test]
fn load_struct_rejects_bad_magic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut b = valid_struct_bytes();
    b[0..8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    let err = write_and_load(dir.path(), "bad.cch-struct", &b);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("bad magic"));
}

#[test]
fn load_struct_rejects_truncated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let b = valid_struct_bytes();
    // Truncate to the 40-byte header: the first section prefix can't be read.
    let err = write_and_load(dir.path(), "trunc.cch-struct", &b[..40]);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("truncated"));
}

#[test]
fn load_struct_rejects_inflated_count() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut b = valid_struct_bytes();
    // node_count = u64::MAX → the order section's real byte_length cannot match
    // 4 * node_count, so the section-length reconciliation rejects it.
    b[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let err = write_and_load(dir.path(), "ovf.cch-struct", &b);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("does not match expected"));
}

#[test]
fn load_struct_rejects_non_monotonic_extra_csr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut b = valid_struct_bytes();
    let starts = section_start_offsets(&b);
    // starts[15] = first_extra_forward_input_arc_of_cch; set its first element
    // to a huge value so the first present extra arc computes hi < lo.
    let data = starts[15] + 8;
    b[data..data + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    let err = write_and_load(dir.path(), "csr.cch-struct", &b);
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("not monotonic"));
}

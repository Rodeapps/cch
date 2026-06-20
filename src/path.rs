//! Shortcut path unpacking — node-path reconstruction after a CCH query.
//!
//! Ported from `native/routing-core/src/cch_mmap.rs` in rapidonkey
//! (`path_query` → [`node_path`]; `unpack_arc`; `find_up_arc`; `Dir`).
//!
//! The algorithm is a bidirectional elimination-tree search that records
//! predecessors in both the forward and backward sweeps, selects the
//! meeting node exactly as routingkit does (strict `<` update along the
//! backward ancestor walk), and then recursively unpacks each shortcut arc
//! via a merge-join over the lower-triangle down-neighbour lists — choosing
//! the FIRST witness, matching routingkit's `unpack_forward_arc` /
//! `unpack_backward_arc`.

use crate::INF_WEIGHT;
use crate::bundle::{CchView, INVALID_ID, MetricView};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Direction selector for shortcut unpacking. `Fwd` uses forward
/// (up-direction) customized weights; `Bwd` uses backward weights.
enum Dir {
    Fwd,
    Bwd,
}

/// Find the up-arc id from `tail` to `head`, or `None` if no such arc exists.
/// Heads within a node's up-range are sorted, but a linear scan is correct
/// and the ranges are tiny in practice.
fn find_up_arc(cch: &CchView, tail: u32, head: u32) -> Option<u32> {
    let from = cch.up_first_out[tail as usize];
    let to = cch.up_first_out[tail as usize + 1];
    (from..to).find(|&i| cch.up_head[i as usize] == head)
}

/// Recursively unpack CCH arc (`x` → `y`) with id `xy`, emitting the ORIGINAL
/// dense node ids of the path's interior + head into `out` in source→target
/// order. `order[v]` maps a rank-space (CCH) node id back to its dense id.
///
/// The arc is a shortcut iff there is a lower-triangle witness `z` (common
/// down-neighbour of x and y) whose two half-arc weights sum to the arc's
/// customized weight (in the chosen direction). We pick the FIRST such
/// witness, matching routingkit's `unpack_forward_arc` / `unpack_backward_arc`.
///
/// The algorithm uses single-character names for rank-space nodes (`x`, `y`,
/// `z`) and arc cursor positions (`a`, `b`) which are conventional in CCH
/// literature; the `many_single_char_names` lint does not apply here.
// Faithfully ported from rapidonkey cch_mmap.rs. The 8-arg signature and
// single-char names match the source exactly; suppress the pedantic lints.
#[allow(clippy::too_many_arguments)] // faithful port: 8-arg signature matches source
#[allow(clippy::many_single_char_names)] // x,y,z,a,b conventional in CCH literature
fn unpack_arc(
    cch: &CchView,
    metric: &MetricView,
    order: &[u32],
    dir: &Dir,
    x: u32,
    y: u32,
    xy: u32,
    out: &mut Vec<u32>,
) {
    // Merge-join over the down-neighbour lists of x and y to find common
    // lower neighbours (the lower triangle of arc x→y).
    let (mut a, ae) = (
        cch.down_first_out[x as usize],
        cch.down_first_out[x as usize + 1],
    );
    let (mut b, be) = (
        cch.down_first_out[y as usize],
        cch.down_first_out[y as usize + 1],
    );
    while a != ae && b != be {
        let hx = cch.down_head[a as usize];
        let hy = cch.down_head[b as usize];
        match hx.cmp(&hy) {
            std::cmp::Ordering::Less => a += 1,
            std::cmp::Ordering::Greater => b += 1,
            std::cmp::Ordering::Equal => {
                // z = hx is a common lower neighbour.
                // bottom_arc = up-arc z→x (== down_to_up[a]);
                // mid_arc    = up-arc z→y (== down_to_up[b]).
                let bottom_arc = cch.down_to_up[a as usize];
                let mid_arc = cch.down_to_up[b as usize];
                let z = hx;
                match dir {
                    Dir::Fwd => {
                        // forward fit: f[xy] == b[bottom] + f[mid]. Recurse:
                        // bottom half backward (z→x), mid half forward (z→y).
                        if metric.forward[xy as usize]
                            == metric.backward[bottom_arc as usize]
                                .saturating_add(metric.forward[mid_arc as usize])
                        {
                            unpack_arc(cch, metric, order, &Dir::Bwd, z, x, bottom_arc, out);
                            unpack_arc(cch, metric, order, &Dir::Fwd, z, y, mid_arc, out);
                            return;
                        }
                    }
                    Dir::Bwd => {
                        // backward fit: b[xy] == f[bottom] + b[mid]. Recurse:
                        // mid half backward (z→y), bottom half forward (z→x).
                        if metric.backward[xy as usize]
                            == metric.forward[bottom_arc as usize]
                                .saturating_add(metric.backward[mid_arc as usize])
                        {
                            unpack_arc(cch, metric, order, &Dir::Bwd, z, y, mid_arc, out);
                            unpack_arc(cch, metric, order, &Dir::Fwd, z, x, bottom_arc, out);
                            return;
                        }
                    }
                }
                a += 1;
                b += 1;
            }
        }
    }
    // No witness: this is an original arc. Emit the head/tail per direction.
    match dir {
        Dir::Fwd => out.push(order[x as usize]),
        Dir::Bwd => out.push(order[y as usize]),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Shortest-path node sequence in ORIGINAL dense node ids, source first,
/// target last. Returns `None` if `target` is unreachable from `source`.
/// Self-pair (`source == target`) returns `Some(vec![source])`.
///
/// Pure-Rust port of routingkit's CCH node-path query: a bidirectional
/// elimination-tree search recording predecessors, followed by recursive
/// shortcut unpacking. Produces results byte-identical to the C++
/// `get_node_path` (verified by `mmap_unpack_matches_cpp_reference_over_200_pairs`).
///
/// # Panics
///
/// Panics via `expect` if the CCH structure is inconsistent (an arc recorded
/// in `fwd_pred` or `bwd_pred` is not found in the up-arc list). This should
/// never happen with a valid, routingkit-produced CCH bundle.
#[must_use]
#[allow(clippy::too_many_lines)] // faithful port of routingkit's path query — splitting obscures algorithm
#[allow(clippy::many_single_char_names)] // s,t,n,x,y,l: conventional rank-space node variables
#[allow(clippy::cast_possible_truncation)] // v < node_count ≤ u32::MAX by CCH invariant
pub fn node_path(cch: &CchView, metric: &MetricView, source: u32, target: u32) -> Option<Vec<u32>> {
    if source == target {
        return Some(vec![source]);
    }

    let n = cch.node_count() as usize;

    // order = inverse(rank): order[rank[v]] = v.
    let mut order = vec![0u32; n];
    for v in 0..n {
        order[cch.rank[v] as usize] = v as u32; // v < node_count ≤ u32::MAX
    }

    let s = cch.rank[source as usize];
    let t = cch.rank[target as usize];

    // Forward sweep from s (up-arcs, forward weights). Relax along the
    // elimination-tree ancestor chain of s, recording predecessors and
    // marking the forward search space. Mirrors routingkit's `run()`.
    let mut fwd_dist = vec![INF_WEIGHT; n];
    let mut fwd_pred = vec![INVALID_ID; n];
    let mut in_forward_search_space = vec![false; n];
    fwd_dist[s as usize] = 0;
    {
        let mut x = s;
        loop {
            in_forward_search_space[x as usize] = true;
            let dx = fwd_dist[x as usize];
            if dx != INF_WEIGHT {
                let from = cch.up_first_out[x as usize] as usize;
                let to = cch.up_first_out[x as usize + 1] as usize;
                for xy in from..to {
                    let y = cch.up_head[xy] as usize;
                    let cand = dx.saturating_add(metric.forward[xy]);
                    if cand < fwd_dist[y] {
                        fwd_dist[y] = cand;
                        fwd_pred[y] = x;
                    }
                }
            }
            let parent = cch.elimination_tree_parent[x as usize];
            if parent == INVALID_ID {
                break;
            }
            x = parent;
        }
    }

    // Backward sweep from t (up-arcs, backward weights), choosing the meeting
    // node EXACTLY as routingkit does: walk t's ancestor chain in order,
    // relaxing then — if x is in the forward search space — updating the
    // meeting node with a STRICT `<` test. This reproduces routingkit's
    // tie-break (first equal-cost ancestor wins) so node paths are
    // byte-identical to `get_node_path`.
    let mut bwd_dist = vec![INF_WEIGHT; n];
    let mut bwd_pred = vec![INVALID_ID; n];
    bwd_dist[t as usize] = 0;
    let mut meeting = INVALID_ID;
    let mut best = INF_WEIGHT;
    {
        let mut x = t;
        loop {
            let dx = bwd_dist[x as usize];
            if dx != INF_WEIGHT {
                let from = cch.up_first_out[x as usize] as usize;
                let to = cch.up_first_out[x as usize + 1] as usize;
                for xy in from..to {
                    let y = cch.up_head[xy] as usize;
                    let cand = dx.saturating_add(metric.backward[xy]);
                    if cand < bwd_dist[y] {
                        bwd_dist[y] = cand;
                        bwd_pred[y] = x;
                    }
                }
            }
            if in_forward_search_space[x as usize] {
                let fd = fwd_dist[x as usize];
                let bd = bwd_dist[x as usize];
                if fd != INF_WEIGHT && bd != INF_WEIGHT {
                    let l = fd.saturating_add(bd);
                    if l < best {
                        best = l;
                        meeting = x;
                    }
                }
            }
            let parent = cch.elimination_tree_parent[x as usize];
            if parent == INVALID_ID {
                break;
            }
            x = parent;
        }
    }

    if meeting == INVALID_ID || best == INF_WEIGHT {
        return None;
    }

    let mut out: Vec<u32> = Vec::new();

    // Forward half: chain source → meeting via fwd_pred (rank space).
    // up_path = [meeting, pred(meeting), ..., s]; unpack from top down so we
    // emit interior heads in source→target order.
    let mut up_path = vec![meeting];
    {
        let mut x = meeting;
        while fwd_pred[x as usize] != INVALID_ID {
            x = fwd_pred[x as usize];
            up_path.push(x);
        }
    }
    // up_path is [meeting, ..., s]; iterate i from high (near s) down to 1.
    for i in (1..up_path.len()).rev() {
        let tail = up_path[i]; // closer to s
        let head = up_path[i - 1]; // closer to meeting
        let arc = find_up_arc(cch, tail, head).expect("forward up-arc on elim path");
        unpack_arc(cch, metric, &order, &Dir::Fwd, tail, head, arc, &mut out);
    }

    // Backward half: meeting → target via bwd_pred. Each step y = pred(x) is
    // an up-arc y→x in rank space; unpack it backward (emits head = order[x]).
    {
        let mut x = meeting;
        let mut y = bwd_pred[x as usize];
        while y != INVALID_ID {
            let arc = find_up_arc(cch, y, x).expect("backward up-arc on elim path");
            unpack_arc(cch, metric, &order, &Dir::Bwd, y, x, arc, &mut out);
            x = y;
            y = bwd_pred[y as usize];
        }
        // x is now the last node on the backward chain (== t in rank space).
        out.push(order[x as usize]);
    }

    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the same 10-node CCH fixture, customize a metric over it, and save
    /// BOTH struct and metric from the SAME cch so arc ids align.
    ///
    /// Input arcs:
    ///   0→1, 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8, 8→9 (cycle forward, 9 arcs)
    ///   9→0 (cycle close)
    ///   0→5 (chord)
    /// Weights: cycle arcs = 1 each, chord = 100. Shortest 0→5 goes around the
    /// cycle (cost 5), forcing shortcut unpacking through contracted nodes.
    fn test_bundle_paths() -> (std::path::PathBuf, std::path::PathBuf) {
        use routingkit_cch::ffi;

        let mut tail = Vec::new();
        let mut head = Vec::new();
        for i in 0u32..9 {
            tail.push(i);
            head.push(i + 1);
        }
        tail.push(9);
        head.push(0);
        tail.push(0);
        head.push(5);

        let weights: Vec<u32> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100];
        let order: Vec<u32> = (0u32..10).collect();
        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };

        let tmp = tempfile::TempDir::new().expect("tempdir");
        let struct_path = tmp.path().join("test.cch-struct");
        let metric_path = tmp.path().join("test.cch-metric");
        unsafe {
            ffi::cch_save_struct(cch.as_ref().unwrap(), struct_path.to_str().unwrap())
                .expect("cch_save_struct");
            let mut metric = ffi::cch_metric_new(cch.as_ref().unwrap(), &weights);
            ffi::cch_metric_customize(metric.as_mut().unwrap());
            ffi::cch_save_metric(metric.as_ref().unwrap(), metric_path.to_str().unwrap())
                .expect("cch_save_metric");
        }
        let _ = tmp.keep();
        (struct_path, metric_path)
    }

    /// Customized weight of the ORIGINAL arc (u,v): look up the up-arc u→v
    /// (forward weight) or the reverse direction. For this fixture all original
    /// arcs survive as CCH arcs in one direction.
    fn original_arc_weight(cch: &CchView, metric: &MetricView, u: u32, v: u32) -> u64 {
        let ru = cch.rank[u as usize];
        let rv = cch.rank[v as usize];
        // Try forward up-arc ru → rv.
        for i in cch.up_first_out[ru as usize]..cch.up_first_out[ru as usize + 1] {
            if cch.up_head[i as usize] == rv {
                return u64::from(metric.forward[i as usize]);
            }
        }
        // Try the reverse up-arc rv → ru, read its backward weight.
        for i in cch.up_first_out[rv as usize]..cch.up_first_out[rv as usize + 1] {
            if cch.up_head[i as usize] == ru {
                return u64::from(metric.backward[i as usize]);
            }
        }
        panic!("no original CCH arc between rank {ru} and {rv}");
    }

    /// Path endpoints must equal source/target; sum of original-arc weights
    /// must equal the query distance.
    #[test]
    fn path_query_endpoints_and_weight_match_distance() {
        use crate::bundle::{CchBundle, MetricBundle};
        use crate::query::distance_matrix;

        let (sp, mp) = test_bundle_paths();
        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let (cv, mv) = (cch_bundle.view(), metric_bundle.view());
        let (s, t) = (0u32, 5u32);
        let path = node_path(&cv, &mv, s, t).expect("reachable");
        assert_eq!(*path.first().unwrap(), s);
        assert_eq!(*path.last().unwrap(), t);
        let dist = distance_matrix(&cv, &mv, &[s], &[t])[0];
        let summed: u64 = path
            .windows(2)
            .map(|w| original_arc_weight(&cv, &mv, w[0], w[1]))
            .sum();
        assert_eq!(summed, u64::from(dist));
    }

    /// Self-pair returns `Some(vec![s])`.
    #[test]
    fn path_query_self_pair_is_single_node() {
        use crate::bundle::{CchBundle, MetricBundle};

        let (sp, mp) = test_bundle_paths();
        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let p = node_path(&cch_bundle.view(), &metric_bundle.view(), 0, 0).unwrap();
        assert_eq!(p, vec![0]);
    }

    /// Unreachable pair returns `None`. Uses a 2-node, single-arc (0→1) CCH so
    /// 1→0 has no path.
    #[test]
    fn path_query_unreachable_returns_none() {
        use crate::bundle::{CchBundle, MetricBundle};
        use routingkit_cch::ffi;

        let tail = vec![0u32];
        let head = vec![1u32];
        let weights = vec![7u32];
        let order = vec![0u32, 1];
        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let sp = tmp.path().join("u.cch-struct");
        let mp = tmp.path().join("u.cch-metric");
        unsafe {
            ffi::cch_save_struct(cch.as_ref().unwrap(), sp.to_str().unwrap()).unwrap();
            let mut metric = ffi::cch_metric_new(cch.as_ref().unwrap(), &weights);
            ffi::cch_metric_customize(metric.as_mut().unwrap());
            ffi::cch_save_metric(metric.as_ref().unwrap(), mp.to_str().unwrap()).unwrap();
        }
        let _ = tmp.keep();
        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        // 1 → 0 should be unreachable.
        assert!(node_path(&cch_bundle.view(), &metric_bundle.view(), 1, 0).is_none());
    }

    /// The 200-pair equivalence gate: assert our pure-Rust `node_path` matches
    /// the C++ routingkit reference for 200 deterministic pseudo-random pairs.
    ///
    /// Fixed-seed LCG (no `rand` crate, no time) — fully reproducible:
    ///   seed = `0x9E37_79B9_7F4A_7C15`
    ///   next = seed * 6364136223846793005 + 1442695040888963407
    ///
    /// Fixture: 10-node directed cycle 0→1→…→9→0 plus chord 0→5. Cycle arc
    /// weights = 1, chord weight = 100 (so shortest 0→5 = 5 via cycle).
    /// Must pass 200/200.
    #[test]
    #[allow(clippy::cast_possible_truncation)] // up_first_out.len()-1 == node_count ≤ u32::MAX
    fn mmap_unpack_matches_cpp_reference_over_200_pairs() {
        use crate::bundle::{CchBundle, MetricBundle};
        use routingkit_cch::ffi;

        let mut tail = Vec::new();
        let mut head = Vec::new();
        for i in 0u32..9 {
            tail.push(i);
            head.push(i + 1);
        }
        tail.push(9);
        head.push(0);
        tail.push(0);
        head.push(5);
        let weights: Vec<u32> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100];
        let order: Vec<u32> = (0u32..10).collect();
        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };

        let tmp = tempfile::TempDir::new().expect("tempdir");
        let sp = tmp.path().join("r200.cch-struct");
        let mp = tmp.path().join("r200.cch-metric");
        let metric_uptr;
        unsafe {
            ffi::cch_save_struct(cch.as_ref().unwrap(), sp.to_str().unwrap()).unwrap();
            let mut metric = ffi::cch_metric_new(cch.as_ref().unwrap(), &weights);
            ffi::cch_metric_customize(metric.as_mut().unwrap());
            ffi::cch_save_metric(metric.as_ref().unwrap(), mp.to_str().unwrap()).unwrap();
            metric_uptr = metric;
        }
        let _ = tmp.keep();

        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let (cv, mv) = (cch_bundle.view(), metric_bundle.view());
        // node_count == 10; up_first_out has node_count+1 entries.
        let n = cv.up_first_out.len() as u32 - 1;

        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..200 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let s = ((seed >> 33) as u32) % n;
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let t = ((seed >> 33) as u32) % n;

            let mut q = unsafe { ffi::cch_query_new(metric_uptr.as_ref().unwrap()) };
            unsafe {
                ffi::cch_query_add_source(q.as_mut().unwrap(), s, 0);
                ffi::cch_query_add_target(q.as_mut().unwrap(), t, 0);
                ffi::cch_query_run(q.as_mut().unwrap());
            }
            let cpp_path = unsafe { ffi::cch_query_node_path(q.as_ref().unwrap()) };
            let cpp_vec: Vec<u32> = cpp_path.clone();

            // Normalize to Option: empty cpp_vec means unreachable → None.
            let theirs: Option<Vec<u32>> = if cpp_vec.is_empty() {
                None
            } else {
                Some(cpp_vec)
            };
            let ours = node_path(&cv, &mv, s, t);

            assert_eq!(ours, theirs, "path mismatch for ({s} -> {t})");
        }
    }
}

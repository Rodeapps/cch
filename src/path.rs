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

    // Hoist borrowed slices so the inner relaxation loops index plain `&[u32]`s
    // and slice each node's arc range once — letting the compiler elide the
    // per-arc bounds checks on `up_head` and the weight arrays.
    let up_first_out = cch.up_first_out;
    let up_head = cch.up_head;
    let elim = cch.elimination_tree_parent;
    let forward = metric.forward;
    let backward = metric.backward;

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
                let from = up_first_out[x as usize] as usize;
                let to = up_first_out[x as usize + 1] as usize;
                let heads = &up_head[from..to];
                let weights = &forward[from..to];
                for (&yv, &w) in heads.iter().zip(weights) {
                    let y = yv as usize;
                    let cand = dx.saturating_add(w);
                    if cand < fwd_dist[y] {
                        fwd_dist[y] = cand;
                        fwd_pred[y] = x;
                    }
                }
            }
            let parent = elim[x as usize];
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
                let from = up_first_out[x as usize] as usize;
                let to = up_first_out[x as usize + 1] as usize;
                let heads = &up_head[from..to];
                let weights = &backward[from..to];
                for (&yv, &w) in heads.iter().zip(weights) {
                    let y = yv as usize;
                    let cand = dx.saturating_add(w);
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
    /// (forward weight) or the reverse up-arc v→u (backward weight). For this
    /// fixture all original arcs survive as CCH arcs in one direction.
    fn original_arc_weight(cch: &CchView, metric: &MetricView, u: u32, v: u32) -> u64 {
        let ru = cch.rank[u as usize];
        let rv = cch.rank[v as usize];
        let fwd = (cch.up_first_out[ru as usize]..cch.up_first_out[ru as usize + 1])
            .find(|&i| cch.up_head[i as usize] == rv)
            .map(|i| u64::from(metric.forward[i as usize]));
        let bwd = (cch.up_first_out[rv as usize]..cch.up_first_out[rv as usize + 1])
            .find(|&i| cch.up_head[i as usize] == ru)
            .map(|i| u64::from(metric.backward[i as usize]));
        fwd.or(bwd)
            .unwrap_or_else(|| panic!("no original CCH arc between rank {ru} and {rv}"))
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

    /// Verify `original_arc_weight` handles the second loop (backward direction
    /// lookup). The path from 5 to 0 in the 10-node cycle includes arc 9→0,
    /// where rank(9) = 9 > rank(0) = 0, which forces the second loop in
    /// `original_arc_weight`.
    #[test]
    fn path_query_backward_arc_weight_lookup() {
        use crate::bundle::{CchBundle, MetricBundle};
        use crate::query::distance_matrix;

        let (sp, mp) = test_bundle_paths();
        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let (cv, mv) = (cch_bundle.view(), metric_bundle.view());
        let (s, t) = (5u32, 0u32);
        let path = node_path(&cv, &mv, s, t).expect("5→0 should be reachable");
        assert_eq!(*path.first().unwrap(), s);
        assert_eq!(*path.last().unwrap(), t);
        let dist = distance_matrix(&cv, &mv, &[s], &[t])[0];
        let summed: u64 = path
            .windows(2)
            .map(|w| original_arc_weight(&cv, &mv, w[0], w[1]))
            .sum();
        assert_eq!(summed, u64::from(dist));
    }

    /// Cover the `panic!` in `original_arc_weight` (line 351): when called on a
    /// pair with no CCH arc in either direction, the function must panic.
    /// Use a 6-node graph with isolated node 5 and call `original_arc_weight`
    /// for nodes (5, 0) — no CCH arc connects them.
    #[test]
    #[should_panic(expected = "no original CCH arc between rank")]
    fn original_arc_weight_panics_on_disconnected_pair() {
        use crate::bundle::{CchBundle, MetricBundle};
        use routingkit_cch::ffi;

        let tail: Vec<u32> = vec![0, 1, 2, 3, 4, 3, 2, 1];
        let head: Vec<u32> = vec![1, 2, 3, 4, 3, 2, 1, 0];
        let weights: Vec<u32> = (1..=8u32).collect();
        // Node 5 is isolated (not in tail or head).
        let n: u32 = 6;
        let order: Vec<u32> = (0..n).collect();

        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
        let cch_ref = cch.as_ref().expect("cch_new returned null");
        let dir = tempfile::tempdir().expect("tempdir");
        let sp = dir.path().join("iso.cch-struct");
        let mp = dir.path().join("iso.cch-metric");
        let mut metric = unsafe { ffi::cch_metric_new(cch_ref, &weights) };
        unsafe {
            ffi::cch_save_struct(cch_ref, sp.to_str().unwrap()).unwrap();
            ffi::cch_metric_customize(metric.as_mut().unwrap());
            ffi::cch_save_metric(metric.as_ref().unwrap(), mp.to_str().unwrap()).unwrap();
        }
        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let (cv, mv) = (cch_bundle.view(), metric_bundle.view());
        // Node 5 is isolated → no CCH arc between node 5 and node 0 → panic.
        original_arc_weight(&cv, &mv, 5, 0);
    }

    /// All-pairs path validity: every reachable (s,t) pair on the 10-node
    /// fixture must produce a path whose endpoint matches and whose summed
    /// original-arc weight equals the query distance. This exercises more
    /// code paths including potential `Dir::Fwd` shortcuts.
    #[test]
    fn path_query_all_pairs_valid() {
        use crate::bundle::{CchBundle, MetricBundle};
        use crate::query::distance_matrix;

        let (sp, mp) = test_bundle_paths();
        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let (cv, mv) = (cch_bundle.view(), metric_bundle.view());
        let n = 10u32;

        for s in 0..n {
            for t in 0..n {
                if s == t {
                    continue;
                }
                // The 10-node cycle is strongly connected: every pair is reachable.
                let dist = distance_matrix(&cv, &mv, &[s], &[t])[0];
                let path = node_path(&cv, &mv, s, t).expect("reachable");
                assert_eq!(*path.first().unwrap(), s);
                assert_eq!(*path.last().unwrap(), t);
                let summed: u64 = path
                    .windows(2)
                    .map(|w| original_arc_weight(&cv, &mv, w[0], w[1]))
                    .sum();
                assert_eq!(summed, u64::from(dist), "path weight mismatch for {s}→{t}");
            }
        }
    }

    /// Cover the `cpp_vec.is_empty() → None` branch (line 488 in the 200-pair
    /// test) by running the same normalisation logic on a graph where a pair
    /// is genuinely unreachable. Single arc 0→1 makes 1→0 unreachable; the C++
    /// oracle returns an empty path vec, which we map to `None`.
    #[test]
    fn mmap_unpack_none_for_unreachable_pair() {
        use crate::bundle::{CchBundle, MetricBundle};
        use routingkit_cch::ffi;

        let tail = vec![0u32];
        let head = vec![1u32];
        let weights = vec![7u32];
        let order = vec![0u32, 1];
        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };

        let tmp = tempfile::TempDir::new().expect("tempdir");
        let sp = tmp.path().join("none.cch-struct");
        let mp = tmp.path().join("none.cch-metric");
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

        // Ask C++ oracle for unreachable pair (1→0).
        let mut q = unsafe { ffi::cch_query_new(metric_uptr.as_ref().unwrap()) };
        unsafe {
            ffi::cch_query_add_source(q.as_mut().unwrap(), 1, 0);
            ffi::cch_query_add_target(q.as_mut().unwrap(), 0, 0);
            ffi::cch_query_run(q.as_mut().unwrap());
        }
        let cpp_path = unsafe { ffi::cch_query_node_path(q.as_ref().unwrap()) };
        let cpp_vec: Vec<u32> = cpp_path.clone();

        // C++ oracle returns empty vec for unreachable pair; our impl returns None.
        assert!(
            cpp_vec.is_empty(),
            "oracle must return empty path for unreachable 1→0"
        );
        let ours = node_path(&cv, &mv, 1, 0);
        assert!(ours.is_none(), "1→0 is unreachable");

        // Also verify a REACHABLE pair (0→1): oracle returns a non-empty path,
        // our impl returns Some.
        let mut q2 = unsafe { ffi::cch_query_new(metric_uptr.as_ref().unwrap()) };
        unsafe {
            ffi::cch_query_add_source(q2.as_mut().unwrap(), 0, 0);
            ffi::cch_query_add_target(q2.as_mut().unwrap(), 1, 0);
            ffi::cch_query_run(q2.as_mut().unwrap());
        }
        let cpp_path2 = unsafe { ffi::cch_query_node_path(q2.as_ref().unwrap()) };
        let cpp_vec2: Vec<u32> = cpp_path2.clone();
        assert!(
            !cpp_vec2.is_empty(),
            "oracle must return path for reachable 0→1"
        );
        let ours2 = node_path(&cv, &mv, 0, 1);
        assert_eq!(
            ours2,
            Some(cpp_vec2),
            "both should agree for reachable pair 0→1"
        );
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
            let theirs: Option<Vec<u32>> = (!cpp_vec.is_empty()).then_some(cpp_vec);
            let ours = node_path(&cv, &mv, s, t);

            assert_eq!(ours, theirs, "path mismatch for ({s} -> {t})");
        }
    }

    /// Cover the `Dir::Fwd` shortcut recursion in `unpack_arc` (lines 94-96).
    /// These lines fire when `unpack_arc` is called with `Dir::Fwd` AND the
    /// arc is a CCH shortcut with a common lower-ranked witness z satisfying:
    ///   `forward[xy] == backward[z→x arc] + forward[z→y arc]`
    ///
    /// Requires a BIDIRECTIONAL graph (finite `backward` weights) AND a graph
    /// topology that creates shortcuts. A directed-only graph has `INF_WEIGHT`
    /// backward weights so the condition never fires.
    ///
    /// Diamond graph 0↔1, 0↔2, 1↔3, 2↔3 with unit weights and identity
    /// ordering [0,1,2,3]. Contracting node 0 (rank 0) creates shortcut 1↔2
    /// (the only path from 1 to 2 in the remaining graph via 0). For query
    /// source=1, target=2, the forward path traverses shortcut 1→2; the
    /// witness is z=0 and both `backward[0→1]=1` and `forward[0→2]=1` are
    /// finite, so `forward[1→2]=2 == 1+1`. Lines 94-96 are executed.
    #[test]
    fn path_query_fwd_unpack_shortcut() {
        use crate::bundle::{CchBundle, MetricBundle};
        use routingkit_cch::ffi;

        // Diamond: 0↔1, 0↔2, 1↔3, 2↔3 (unit weights, bidirectional).
        // Arcs encoded as directed pairs (both directions):
        //   0→1, 0→2, 1→3, 2→3, 1→0, 2→0, 3→1, 3→2
        let tail: Vec<u32> = vec![0, 0, 1, 2, 1, 2, 3, 3];
        let head: Vec<u32> = vec![1, 2, 3, 3, 0, 0, 1, 2];
        let weights: Vec<u32> = vec![1u32; tail.len()];
        // Identity ordering: node 0 contracted first, then 1, 2, 3.
        let order: Vec<u32> = vec![0, 1, 2, 3];

        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
        let cch_ref = cch.as_ref().expect("cch_new returned null");

        let dir = tempfile::tempdir().expect("tempdir");
        let sp = dir.path().join("diamond.cch-struct");
        let mp = dir.path().join("diamond.cch-metric");
        let mut metric = unsafe { ffi::cch_metric_new(cch_ref, &weights) };
        unsafe {
            ffi::cch_save_struct(cch_ref, sp.to_str().unwrap()).unwrap();
            ffi::cch_metric_customize(metric.as_mut().unwrap());
            ffi::cch_save_metric(metric.as_ref().unwrap(), mp.to_str().unwrap()).unwrap();
        }

        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let (cv, mv) = (cch_bundle.view(), metric_bundle.view());

        // Query 1→2: shortest path cost = 2 (via node 0: 1→0→2).
        // The CCH forward half traverses the shortcut 1→2 (via contracted node 0).
        // Inside unpack_arc(Dir::Fwd, 1, 2, shortcut), z=0 is found, and
        // forward[1→2] == backward[0→1] + forward[0→2] = 1+1 = 2. Lines 94-96.
        let path_12 = node_path(&cv, &mv, 1, 2).expect("1→2 reachable in diamond");
        assert_eq!(*path_12.first().unwrap(), 1);
        assert_eq!(*path_12.last().unwrap(), 2);
        assert_eq!(path_12.len(), 3, "path 1→0→2 has 3 nodes");

        // Also verify 0→3: longest path exercising more shortcuts.
        let path_03 = node_path(&cv, &mv, 0, 3).expect("0→3 reachable");
        assert_eq!(*path_03.first().unwrap(), 0);
        assert_eq!(*path_03.last().unwrap(), 3);
    }

    /// Cover the `if dx != INF_WEIGHT` false branch in the forward sweep of
    /// `node_path`. This branch fires when a node in the source's elimination-
    /// tree ancestor chain has `INF_WEIGHT` tentative distance. This happens
    /// when all arc weights in the metric are `INF_WEIGHT`: the source gets
    /// dist=0, but `0.saturating_add(INF_WEIGHT) = INF_WEIGHT` which is NOT
    /// strictly less than `INF_WEIGHT`, so the parent is never updated. The
    /// second loop iteration enters the else branch.
    #[test]
    fn path_query_fwd_sweep_inf_weight_branch() {
        use crate::bundle::{CchBundle, MetricBundle};
        use routingkit_cch::ffi;

        // 5-node path graph; all arc weights = INF_WEIGHT.
        let n: u32 = 5;
        let order: Vec<u32> = (0..n).collect();
        let tail: Vec<u32> = (0..n - 1).collect();
        let head: Vec<u32> = (1..n).collect();
        let weights: Vec<u32> = vec![crate::INF_WEIGHT; tail.len()];

        let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
        let cch_ref = cch.as_ref().expect("cch_new returned null");

        let dir = tempfile::tempdir().expect("tempdir");
        let sp = dir.path().join("fwdinf.cch-struct");
        let mp = dir.path().join("fwdinf.cch-metric");
        let mut metric = unsafe { ffi::cch_metric_new(cch_ref, &weights) };
        unsafe {
            ffi::cch_save_struct(cch_ref, sp.to_str().unwrap()).unwrap();
            ffi::cch_metric_customize(metric.as_mut().unwrap());
            ffi::cch_save_metric(metric.as_ref().unwrap(), mp.to_str().unwrap()).unwrap();
        }

        let cch_bundle = CchBundle::open(&sp).unwrap();
        let metric_bundle = MetricBundle::open(&mp).unwrap();
        let (cv, mv) = (cch_bundle.view(), metric_bundle.view());

        // All arcs have INF_WEIGHT → no relaxation from source → ancestors at INF.
        let result = node_path(&cv, &mv, 0, 4);
        assert!(result.is_none(), "all-INF metric means 0→4 is unreachable");
    }
}

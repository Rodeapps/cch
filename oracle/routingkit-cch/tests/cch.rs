use indicatif::{MultiProgress, ProgressBar, ProgressIterator};
use pathfinding::prelude::dijkstra;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use routingkit_cch::{
    CCH, CCHMetric, CCHMetricPartialUpdater, CCHQuery, compute_order_degree,
    compute_order_inertial, shp_utils,
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};

const STYLE: LazyLock<indicatif::ProgressStyle> = LazyLock::new(|| {
    indicatif::ProgressStyle::with_template(
        "{prefix} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
    )
    .unwrap()
});

#[test]
fn compare_with_pathfinding() {
    for city in ["beijing", "chengdu", "cityindia", "harbin", "porto"] {
        eprintln!("====\nComparing with pathfinding for city: {}\n====", city);
        let (Ok(edges), Ok(nodes)) = (
            shp_utils::load_edges(&format!("data/{city}_data/map/edges.shp")),
            shp_utils::load_nodes(&format!("data/{city}_data/map/nodes.shp")),
        ) else {
            eprintln!("Failed to load data for city: {}", city);
            continue;
        };
        let shp_utils::GraphArrays {
            osmids,
            xs,
            ys,
            tail,
            head,
            weight,
        } = shp_utils::build_graph_arrays(&nodes, &edges).unwrap();
        let node_count = nodes.len();
        let osmid2idx = osmids
            .iter()
            .enumerate()
            .map(|(i, &osmid)| (osmid, i as u32))
            .collect::<std::collections::HashMap<_, _>>();

        let tail = tail.into_iter().map(|x| x as u32).collect::<Vec<u32>>();
        let head = head.into_iter().map(|x| x as u32).collect::<Vec<u32>>();
        let weights = weight
            .into_iter()
            .map(|x| (x * 1e3) as u32)
            .collect::<Vec<u32>>();
        let lat = xs.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let lon = ys.into_iter().map(|x| x as f32).collect::<Vec<f32>>();

        eprintln!("Graph has {} nodes, {} edges.", node_count, tail.len());

        // Compute simple degree order as a baseline (fast). Could use inertial order with coords if provided.
        eprintln!("Computing order...");
        let order = compute_order_inertial(node_count as u32, &tail, &head, &lat, &lon);

        eprintln!("Building CCH...");
        let cch = CCH::new(
            &order,
            &tail,
            &head,
            |msg| eprintln!("[CCH log message] {msg}"),
            false,
        );
        eprintln!("Building metric + customization...");
        let metric = CCHMetric::new(&cch, weights.clone());

        // Build adjacency for reference dijkstra using pathfinding crate
        eprintln!("Building adjacency for pathfinding reference...");
        let adj = {
            let mut adj = vec![Vec::<(u32, u32)>::new(); node_count];
            for i in 0..tail.len() {
                adj[tail[i] as usize].push((head[i], weights[i]));
            }
            adj
        };

        // Preselect random queries (s,t) distinct
        eprintln!("Loading historical trips...");
        let deserialized_trips: Vec<(serde_pickle::Value, Vec<usize>, (usize, usize))> =
            serde_pickle::from_reader(
                std::fs::File::open(format!("data/{city}_data/preprocessed_train_trips_all.pkl"))
                    .unwrap(),
                Default::default(),
            )
            .unwrap();
        let paths: Vec<((u32, u32), &[usize])> = deserialized_trips
            .iter()
            .map(|(_, path, _)| {
                let (&first_edge_id, &last_edge_id) = (path.first().unwrap(), path.last().unwrap());
                (
                    (
                        osmid2idx[&edges[first_edge_id].u],
                        osmid2idx[&edges[last_edge_id].v],
                    ),
                    path.as_slice(),
                )
            })
            .collect();

        eprintln!("Running {} queries...", paths.len());
        let m = MultiProgress::new();
        let num_chunks = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1);
        let num_digits = (num_chunks - 1).to_string().len();
        paths
            .par_chunks(paths.len() / num_chunks)
            .enumerate()
            .map(|(chunk_id, chunk)| {
                let mut query = CCHQuery::new(&metric);
                let pb = m.add(ProgressBar::new(chunk.len() as u64));
                pb.set_prefix(format!("{city}-{:<num_digits$}", chunk_id));
                pb.set_style(STYLE.clone());
                for (i, &((s, t), _)) in chunk.iter().enumerate().progress_with(pb) {
                    query.add_source(s, 0);
                    query.add_target(t, 0);
                    let dist_cch = query.run().distance();

                    // pathfinding dijkstra
                    let result = dijkstra(&s, |&u| adj[u as usize].iter().cloned(), |&u| u == t);
                    let dist_ref = result.map(|(_, cost)| cost);

                    assert_eq!(
                        dist_cch, dist_ref,
                        "distance mismatch on query #{i} s={s} t={t}"
                    );
                }
            })
            .count();
    }
}

#[test]
fn random_graph_compare_parallel_partial() {
    let mut rng = StdRng::from_entropy();

    // Graph parameters (kept modest for test time; adjust if needed)
    let node_count: u32 = 2_000;
    let edge_count: usize = 10_000;

    // Ensure basic connectivity: start with a random permutation spanning tree
    let mut nodes: Vec<u32> = (0..node_count).collect();
    nodes.shuffle(&mut rng);

    let mut tail: Vec<u32> = Vec::with_capacity(edge_count);
    let mut head: Vec<u32> = Vec::with_capacity(edge_count);
    let mut weights: Vec<u32> = Vec::with_capacity(edge_count);

    // Spanning tree edges
    for w in 1..nodes.len() {
        let u = nodes[w - 1];
        let v = nodes[w];
        tail.push(u);
        head.push(v);
        weights.push(rng.gen_range(1..=1_000));
    }
    // Remaining random edges
    while tail.len() < edge_count {
        let u = rng.gen_range(0..node_count);
        let v = rng.gen_range(0..node_count);
        if u == v {
            continue;
        }
        tail.push(u);
        head.push(v);
        weights.push(rng.gen_range(1..=1_000));
    }

    eprintln!(
        "[random_graph_compare] graph: {} nodes, {} edges",
        node_count, edge_count
    );

    // Order (degree-based since we have no coords here)
    let order = compute_order_degree(node_count, &tail, &head);
    let cch = CCH::new(&order, &tail, &head, |_| {}, false);
    eprint!("Building metric + customization...");
    let mut metric = CCHMetric::new(&cch, weights);
    eprintln!(" done.");
    let mut updater = CCHMetricPartialUpdater::new(&cch);

    // Helper to rebuild adjacency for reference Dijkstra
    let build_adj = |weights_slice: &[u32]| -> Vec<Vec<(u32, u32)>> {
        let mut adj = vec![Vec::<(u32, u32)>::new(); node_count as usize];
        for i in 0..tail.len() {
            adj[tail[i] as usize].push((head[i], weights_slice[i]));
        }
        adj
    };

    let mut adj = build_adj(metric.weights());

    // Query + update rounds
    let rounds = 10;
    let queries_per_round = 400; // * rounds = 4000 reference Dijkstra runs
    let update_ratio = 0.005; // 0.5% of edges per update (~50 edges)

    for round in 0..rounds {
        // Prepare random (s,t) distinct pairs
        let mut pairs = Vec::with_capacity(queries_per_round);
        while pairs.len() < queries_per_round {
            let s = rng.gen_range(0..node_count);
            let t = rng.gen_range(0..node_count);
            if s != t {
                pairs.push((s, t));
            }
        }

        eprintln!("[round {round}] verifying {} queries", pairs.len());

        // Parallel verification: each thread gets its own query object
        pairs.par_iter().for_each_init(
            || CCHQuery::new(&metric),
            |query, (s, t)| {
                query.add_source(*s, 0);
                query.add_target(*t, 0);
                let qres = query.run();
                let dist_cch = qres.distance();

                {
                    // Verify path correctness: length equal and node_path matches arc_path
                    let arc_path = qres.arc_path();
                    let node_path = qres.node_path();
                    if let Some(dist_cch) = dist_cch {
                        let weights = metric.weights();
                        let dist_sum = arc_path.iter().map(|&e| weights[e as usize]).sum::<u32>();
                        assert_eq!(
                            dist_cch, dist_sum,
                            "distance should match sum of arc weights",
                        );
                        assert!(
                            node_path
                                .windows(2)
                                .zip(arc_path)
                                .all(|(uv, e)| tail[e as usize] == uv[0]
                                    && head[e as usize] == uv[1]),
                            "node_path should match arc_path"
                        );
                    } else {
                        assert!(
                            node_path.is_empty() && arc_path.is_empty(),
                            "paths should be empty if unreachable",
                        );
                    }
                }

                // Reference Dijkstra
                let res = pathfinding::prelude::dijkstra(
                    s,
                    |&u| adj[u as usize].iter().cloned(),
                    |&u| u == *t,
                );
                let dist_ref = res.map(|(_, c)| c);
                assert_eq!(
                    dist_cch, dist_ref,
                    "distance mismatch s={s} t={t} (round {round})"
                );
            },
        );

        // Skip update after last round
        if round == rounds - 1 {
            break;
        }

        // Prepare sparse updates
        let updates_count = (edge_count as f64 * update_ratio).ceil() as usize;
        let mut indices: Vec<usize> = (0..edge_count).collect();
        indices.shuffle(&mut rng);
        indices.truncate(updates_count);

        let mut update_map = BTreeMap::<u32, u32>::new();
        for &idx in &indices {
            update_map.insert(idx as u32, rng.gen_range(1..=1_000));
        }

        eprint!(
            "[round {round}] applying {} partial weight updates...",
            update_map.len()
        );
        updater.apply(&mut metric, &update_map);
        eprintln!(" done.");
        // Rebuild adjacency to reflect new weights
        adj = build_adj(metric.weights());
    }
}

#[test]
fn partial_update_with_reusable_updater() {
    // Same base graph as previous test: 0->1 (5), 1->2 (7), 0->2 (20)
    let tail = vec![0, 1, 0];
    let head = vec![1, 2, 2];
    let weights = vec![5u32, 7u32, 20u32];
    let order = compute_order_degree(3, &tail, &head);
    let cch = CCH::new(&order, &tail, &head, |_| {}, false);
    let mut metric = CCHMetric::new(&cch, weights);
    // Build reusable updater (now constructed from &CCH)
    let mut updater = CCHMetricPartialUpdater::new(&cch);
    // Baseline shortest 0->2 is 12
    let mut q = CCHQuery::new(&metric);
    q.add_source(0, 0);
    q.add_target(2, 0);
    assert_eq!(q.run().distance(), Some(12));
    // 1) Increase 0->1 to 30 (direct 0->2 wins: 20)
    updater.apply(&mut metric, &HashMap::<u32, u32>::from_iter([(0, 30)]));
    let mut q2 = CCHQuery::new(&metric);
    q2.add_source(0, 0);
    q2.add_target(2, 0);
    assert_eq!(q2.run().distance(), Some(20));
    assert_eq!(metric.weights(), vec![30, 7, 20]);
    // 2) Decrease 1->2 to 1 (path 0->1->2 becomes 31, still worse)
    updater.apply(&mut metric, &BTreeMap::from_iter([(1, 1)]));
    let mut q3 = CCHQuery::new(&metric);
    q3.add_source(0, 0);
    q3.add_target(2, 0);
    assert_eq!(q3.run().distance(), Some(20));
    assert_eq!(metric.weights(), vec![30, 1, 20]);
    // 3) Decrease 0->1 to 2 (now 2+1=3 wins)
    updater.apply(&mut metric, &BTreeMap::from_iter([(0, 2)]));
    let mut q4 = CCHQuery::new(&metric);
    q4.add_source(0, 0);
    q4.add_target(2, 0);
    assert_eq!(q4.run().distance(), Some(3));
    assert_eq!(metric.weights(), vec![2, 1, 20]);
    // 4) Batch update with duplicates: set 0->1 to 4 (overwritten), then 6 final; 1->2 to 10 final.
    updater.apply(
        &mut metric,
        &HashMap::<u32, u32>::from_iter([(0, 4), (0, 6), (1, 12), (1, 10)]),
    );
    let mut q5 = CCHQuery::new(&metric);
    q5.add_source(0, 0);
    q5.add_target(2, 0);
    // Now path via 0->1->2 is 6+10=16 vs direct 20
    assert_eq!(q5.run().distance(), Some(16));
    assert_eq!(metric.weights(), vec![6, 10, 20]);
}

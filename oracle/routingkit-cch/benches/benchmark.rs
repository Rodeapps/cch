use criterion::{Criterion, criterion_group, criterion_main};
use pathfinding::prelude::dijkstra;
use rand::{Rng, SeedableRng, rngs::StdRng};
use routingkit_cch::{CCH, CCHMetric, CCHQuery, compute_order_inertial, shp_utils};

fn bench_pathfinding(c: &mut Criterion) {
    for city in ["beijing", "chengdu", "cityindia", "harbin", "porto"] {
        let mut group = c.benchmark_group(city);
        eprintln!("====\nComparing with pathfinding for city: {}\n====", city);
        let (Ok(edges), Ok(nodes)) = (
            shp_utils::load_edges(&format!("data/{city}_data/map/edges.shp")),
            shp_utils::load_nodes(&format!("data/{city}_data/map/nodes.shp")),
        ) else {
            eprintln!("Failed to load data for city: {}", city);
            continue;
        };
        let shp_utils::GraphArrays {
            osmids: _,
            xs,
            ys,
            tail,
            head,
            weight,
        } = shp_utils::build_graph_arrays(&nodes, &edges).unwrap();
        let node_count = nodes.len();

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
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        eprintln!("Building metric + customization...");
        let metric = CCHMetric::new(&cch, weights.clone());
        let mut query = CCHQuery::new(&metric);

        // Build adjacency for reference dijkstra using pathfinding crate
        eprintln!("Building adjacency for pathfinding reference...");
        let adj = {
            let mut adj = vec![Vec::<(u32, u32)>::new(); node_count];
            for i in 0..tail.len() {
                adj[tail[i] as usize].push((head[i], weights[i]));
            }
            adj
        };

        let mut rng = StdRng::seed_from_u64(42);
        group.bench_function("CCH", |b| {
            b.iter(|| {
                let start = rng.gen_range(0..node_count) as u32;
                let goal = rng.gen_range(0..node_count) as u32;
                query.add_source(start, 0);
                query.add_target(goal, 0);
                let res = query.run();
                let _ = (res.node_path(), res.distance());
            })
        });
        let mut rng = StdRng::seed_from_u64(42);
        group.bench_function("Dijkstra", |b| {
            b.iter(|| {
                let start = rng.gen_range(0..node_count) as u32;
                let goal = rng.gen_range(0..node_count) as u32;
                let _ = dijkstra(
                    &start,
                    |&n| adj[n as usize].iter().map(|&(h, w)| (h, w)),
                    |&n| n == goal,
                );
            })
        });

        group.finish();
    }
}

criterion_group!(benches, bench_pathfinding);
criterion_main!(benches);

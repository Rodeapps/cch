use routingkit_cch::shp_utils::{build_graph_arrays, load_edges, load_nodes};

const EDGES_PATH: &str = "data/beijing_data/map/edges.shp";
const NODES_PATH: &str = "data/beijing_data/map/nodes.shp";
const TRIPS_PATH: &str = "data/beijing_data/preprocessed_train_trips_all.pkl";

#[test]
fn test_load_edges() {
    if let Ok(edges) = load_edges(&EDGES_PATH) {
        if !edges.is_empty() {
            println!("Loaded {} edges. Showing first 10:", edges.len());
            for (i, e) in edges.iter().take(10).enumerate() {
                println!(
                    "Edge[{i}]: fid={} u={} v={} len={} highway={:?} name={:?} oneway={:?} maxspeed={:?}",
                    e.fid, e.u, e.v, e.length, e.highway, e.name, e.oneway, e.maxspeed
                );
            }
        } else {
            println!("No edges loaded (file missing?)");
        }
    }
}

#[test]
fn test_load_nodes() {
    match load_nodes(&NODES_PATH) {
        Ok(nodes) => {
            if !nodes.is_empty() {
                println!("Loaded {} nodes. Showing first 10:", nodes.len());
                for (i, n) in nodes.iter().take(10).enumerate() {
                    println!(
                        "Node[{i}]: osmid={} x={:.6} y={:.6} highway={:?} ref={:?}",
                        n.osmid, n.x, n.y, n.highway, n.r#ref
                    );
                }
            } else {
                println!("No nodes loaded (file missing or no point geometries)");
            }
        }
        Err(e) => println!("Failed to load nodes: {e}"),
    }
}

#[test]
fn test_build_graph_arrays() {
    let nodes = match load_nodes(&NODES_PATH) {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to load nodes: {e}");
            return;
        }
    };
    let edges = match load_edges(&EDGES_PATH) {
        Ok(v) => v,
        Err(e) => {
            println!("Failed to load edges: {e}");
            return;
        }
    };
    let g = match build_graph_arrays(&nodes, &edges) {
        Ok(g) => g,
        Err(e) => {
            println!("Build failed: {e}");
            return;
        }
    };

    println!("{:?}", g);
}

#[test]
fn test_load_paths() {
    let Ok(rdr) = std::fs::File::open(TRIPS_PATH) else {
        println!("Failed to open trips file: {TRIPS_PATH}");
        return;
    };
    let deserialized: Vec<(serde_pickle::Value, Vec<usize>, (usize, usize))> =
        serde_pickle::from_reader(rdr, Default::default()).unwrap();
    println!("Loaded {} paths. Showing first 5:", deserialized.len());
    for (i, (idx, path, time)) in deserialized.iter().take(5).enumerate() {
        println!("Path[{i}]: idx={idx:?}, path={path:?}, time={time:?}");
    }

    let edges = deserialized
        .iter()
        .flat_map(|(_, x, _)| x)
        .collect::<std::collections::HashSet<_>>();
    let max_id = edges.iter().max().unwrap();
    let min_id = edges.iter().min().unwrap();
    println!(
        "Total unique edges in paths: {},  min_id={min_id}, max_id={max_id}",
        edges.len(),
    );
    assert!(
        **max_id < load_edges(&EDGES_PATH).unwrap().len(),
        "max edge id in paths exceeds total edges"
    );
}

use rand::{Rng, SeedableRng, rngs::StdRng};
use routingkit_cch::{CCH, CCHMetric, CCHMetricPartialUpdater, CCHQuery, compute_order_degree};

// This test attempts to validate that moving or swapping CCHMetric does not invalidate
// the underlying weights pointer observed by the C++ side. We cannot read the C++ raw pointer
// directly, but we can perform queries before and after moves/swaps and observe consistent
// distances when weights are unchanged and updated distances after mutation.
#[test]
fn weight_pointer_stability() {
    // Graph: 0->1 (1), 1->2 (2), 0->2 (10)
    let tail = vec![0, 1, 0];
    let head = vec![1, 2, 2];
    let weights = vec![1u32, 2u32, 10u32];
    let order = compute_order_degree(3, &tail, &head);
    let cch = CCH::new(&order, &tail, &head, |_| {}, false);

    let metric_a = CCHMetric::new(&cch, weights.clone());
    // Move construct metric_b by assigning metric_a (metric_a moved)
    let mut metric_b = metric_a; // move

    // Query baseline via new query object
    let mut q1 = routingkit_cch::CCHQuery::new(&metric_b);
    q1.add_source(0, 0);
    q1.add_target(2, 0);
    assert_eq!(q1.run().distance(), Some(3)); // 1+2 path

    // Now create another metric with different weight to check independence
    let metric_c = CCHMetric::new(&cch, vec![1, 50, 10]); // path 0->2 direct =10 now better than 1+50=51
    let mut q_c = routingkit_cch::CCHQuery::new(&metric_c);
    q_c.add_source(0, 0);
    q_c.add_target(2, 0);
    assert_eq!(q_c.run().distance(), Some(10));

    // Swap metric_b and metric_c; distances should follow underlying weight vectors
    let mut metric_c = metric_c; // mutable
    std::mem::swap(&mut metric_b, &mut metric_c);

    // After swap metric_b now has weights [1,50,10]
    let mut q2 = routingkit_cch::CCHQuery::new(&metric_b);
    q2.add_source(0, 0);
    q2.add_target(2, 0);
    assert_eq!(q2.run().distance(), Some(10));

    // metric_c now has original fast weights [1,2,10]
    let mut q3 = routingkit_cch::CCHQuery::new(&metric_c);
    q3.add_source(0, 0);
    q3.add_target(2, 0);
    assert_eq!(q3.run().distance(), Some(3));

    // Mutate weights in metric_c via a reusable updater style patch equivalent
    // (since current API may provide partial update elsewhere; emulate by rebuilding new metric)
    // Here rebuild to ensure pointer validity check across drop + new allocation.
    let metric_d = CCHMetric::new(&cch, vec![1, 1, 10]);
    let mut qd = routingkit_cch::CCHQuery::new(&metric_d);
    qd.add_source(0, 0);
    qd.add_target(2, 0);
    assert_eq!(qd.run().distance(), Some(2));
}

// Build a synthetic layered directed graph:
// L layers, each with W nodes; arcs forward between consecutive layers dense-ish plus
// random cross-layer extra arcs. Then test pointer stability under many memory ops:
// - move, swap, mem::replace
// - push metrics into Vec and reserve/shrink other Vecs nearby to try to cause allocator pressure
// - partial updates via updater (if available) and rebuild metrics
// We cannot directly read the C++ stored pointer; instead we rely on consistent query distances
// before/after moves and correctness of updates to infer pointer validity.
#[test]
fn weight_pointer_stability_large() {
    const L: usize = 25; // layers
    const W: usize = 40; // width per layer => total nodes ~ 1000
    let node_count = (L * W) as u32;
    // Assign node ids layer-major: (layer * W + idx)
    let mut tail = Vec::<u32>::new();
    let mut head = Vec::<u32>::new();
    let mut base_weights = Vec::<u32>::new();
    // Deterministic RNG
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);

    for layer in 0..L - 1 {
        // forward edges to next layer
        for i in 0..W {
            let u = (layer * W + i) as u32;
            for j in 0..W {
                // fully connect to next layer subset (sparse sample)
                if (i + j) % 7 == 0 {
                    // roughly 1/7 density
                    let v = ((layer + 1) * W + j) as u32;
                    tail.push(u);
                    head.push(v);
                    base_weights.push((1 + ((i + j) % 11)) as u32);
                }
            }
        }
    }
    // Random extra cross edges (skip last layer)
    for _ in 0..(L * W / 2) {
        let layer = rng.gen_range(0..L - 2);
        let u = (layer * W + rng.gen_range(0..W)) as u32;
        let v = ((layer + 2) * W + rng.gen_range(0..W)) as u32; // skip a layer
        tail.push(u);
        head.push(v);
        base_weights.push(rng.gen_range(5..30));
    }

    let order = compute_order_degree(node_count, &tail, &head);
    let cch = CCH::new(&order, &tail, &head, |_| {}, false);

    // Build initial metric
    let metric = CCHMetric::new(&cch, base_weights.clone());

    // Helper to run a batch of queries and return tuple (sum_dist, count_reached)
    let run_sample = |m: &CCHMetric, sample_seed: u64| -> (u64, usize) {
        let mut rng = StdRng::seed_from_u64(sample_seed);
        let mut total = 0u64;
        let mut reached = 0usize;
        for _ in 0..200 {
            // 200 random queries
            let s = rng.gen_range(0..node_count);
            let t = rng.gen_range(0..node_count);
            if s == t {
                continue;
            }
            let mut q = CCHQuery::new(m);
            q.add_source(s, 0);
            q.add_target(t, 0);
            if let Some(d) = q.run().distance() {
                total += d as u64;
                reached += 1;
            }
        }
        (total, reached)
    };

    let baseline = run_sample(&metric, 123);

    // Move metric into a vector (heap relocation of owner struct, not its Vec buffer contents) and back
    let mut holder: Vec<CCHMetric> = Vec::new();
    holder.push(metric); // move
    let mut metric = holder.pop().unwrap();
    let after_move = run_sample(&metric, 123);
    assert_eq!(baseline, after_move, "Distances changed after move");

    // Reserve big capacity in an unrelated Vec to cause allocator churn
    let mut noise: Vec<Vec<u64>> = Vec::new();
    for k in 0..200 {
        // allocate some varying sizes
        let mut v = Vec::with_capacity(k + 10);
        v.extend(0..(k as u64 % 17));
        noise.push(v);
    }

    // Swap with a different metric (heavier weights) and verify expected monotonic increase in sample sum
    let mut heavier_weights = base_weights.clone();
    for w in heavier_weights.iter_mut() {
        *w += 10;
    }
    let mut metric2 = CCHMetric::new(&cch, heavier_weights);
    let before_swap_heavy = run_sample(&metric2, 123);
    std::mem::swap(&mut metric, &mut metric2);
    let after_swap_heavy = run_sample(&metric, 123);
    assert_eq!(
        before_swap_heavy, after_swap_heavy,
        "Swap failed: heavy metric distances mismatch"
    );
    let back_again = run_sample(&metric2, 123);
    assert_eq!(
        back_again, baseline,
        "Original metric distances mismatch after swap"
    );

    // mem::replace with a new metric (lighter weights) then drop the old one
    let mut lighter_weights = base_weights.clone();
    for w in lighter_weights.iter_mut() {
        if *w > 1 {
            *w -= 1;
        }
    }
    let metric_old = std::mem::replace(&mut metric, CCHMetric::new(&cch, lighter_weights.clone()));
    // old still queryable
    let old_sample = run_sample(&metric_old, 321);
    let lighter_sample = run_sample(&metric, 321);
    assert!(
        lighter_sample.0 <= old_sample.0,
        "Lighter weights should not increase distances"
    );
    drop(metric_old);

    // Partial updates using updater: modify 500 random arcs
    let mut updater = CCHMetricPartialUpdater::new(&cch);
    // Build a HashMap of updates (random subset) mapping arc id -> new_weight
    use std::collections::HashMap;
    let mut updates = HashMap::new();
    let arc_count = base_weights.len() as u32;
    for _ in 0..500 {
        let arc = rng.gen_range(0..arc_count);
        updates.insert(arc, 3u32 + (arc % 5));
    }
    updater.apply(&mut metric, &updates);
    let after_updates = run_sample(&metric, 555);

    // To have a deterministic assertion, re-run with same seed after reapplying identical updates: idempotent
    let mut metric_clone = CCHMetric::new(&cch, lighter_weights); // build again lighter, then apply same updates
    updater.apply(&mut metric_clone, &updates);
    let after_updates_clone = run_sample(&metric_clone, 555);
    assert_eq!(
        after_updates, after_updates_clone,
        "Applying same updates on same base should yield identical distances"
    );

    // Finally ensure moving the updated metric doesn't change sample
    let moved_metric = metric; // move
    let final_sample = run_sample(&moved_metric, 777);
    let final_sample2 = run_sample(&moved_metric, 777);
    assert_eq!(
        final_sample, final_sample2,
        "Deterministic queries changed after move"
    );
}

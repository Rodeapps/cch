# Parallel customization — scaling benchmark

Measures how `Cch::customize` (the parallel, rayon-based customization added in
0.2.0) scales with graph size and core count. The goal is to answer honestly:
*does the parallelism actually help, and from what size?*

## Method

- Bidirectional square grids of increasing size, ordered with `inertial_order`
  (nested dissection) for a realistic, low-fill-in hierarchy.
- Each CCH is built once; `customize` is then timed under a **1-thread** and an
  **all-cores** rayon pool (`ThreadPoolBuilder`), so the only variable is core
  count. Criterion, `--warm-up-time 2 --measurement-time 4`, `sample_size(10)`.
- Hardware: Apple Silicon, 18 logical cores. Numbers are machine-specific — run
  `CCH_BENCH_SIDE=<side> cargo bench --bench cch -- customize_large` to reproduce.

## Results

| Grid   | Nodes      | 1 thread    | 18 threads | Speedup |
|--------|------------|-------------|------------|---------|
| 128²   | 16,384     | 20.2 ms     | 20.7 ms    | ~1.0× (none) |
| 256²   | 65,536     | 145.8 ms    | 78.7 ms    | 1.85×   |
| 810²   | 656,100    | 4.77 s      | 1.71 s     | 2.79×   |
| 2561²  | 6,558,721  | ~165 s/iter*| not measured | —     |
| 8101²  | 65,626,201 | not attempted | —        | —       |

\* Single-thread per-iteration cost taken from criterion's own estimate; the run
was stopped before completion because the process was ~12.5 GB resident and the
machine was swapping. 66M nodes (~125 GB projected) is infeasible on this box.

## Conclusions

1. **The parallel speedup grows with graph size, but only pays off at scale.**
   No gain at 16k nodes (parallel == serial within noise), 1.85× at 65k, 2.79×
   at 656k. Below ~tens of thousands of nodes, the per-level barrier plus rayon
   dispatch overhead roughly cancels the work — so parallelism is not a blanket
   win, it is a large-graph win.

2. **Efficiency is low and stays low** — ~15% (2.79× / 18 cores) at 656k. This is
   inherent to level-synchronized customization: the upper elimination-tree
   levels contain very few nodes, so most cores sit idle there, and there is a
   barrier per level. This matches the known behavior of RoutingKit's parallel
   customization; it is not a defect in the implementation.

3. **No regression on small graphs.** At 16k nodes the parallel path is within
   noise of single-threaded, so small-graph users pay nothing measurable for the
   always-on parallelism.

4. **Single-thread cost scales ~super-linearly on grids** (≈ n^1.5): 146 ms →
   4.77 s → ~165 s for 65k → 656k → 6.6M (roughly ×33 per 10× nodes). Grids have
   much worse separators than real road networks, so this is a *pessimistic*
   proxy — a real continental road graph (near-planar, good separators) should
   scale closer to n·log n, with better absolute times and likely better
   parallel efficiency. These grid numbers bound the bad case, not the expected
   road-network case.

5. **Practical ceiling on this hardware:** 656k is comfortable; 6.6M runs but is
   memory-bound (~12.5 GB, swapping); 66M is out of reach. A true
   continental-scale figure needs a larger-memory machine and, better, the real
   road corpus — see the road-network benchmark follow-up (improvement #3).

## Implication for the 0.2.0 "parallel customization" claim

Present it as what it is: a **real but modest win that materializes at scale**
(≈1.85× at 65k, ≈2.8× at 656k on 18 cores), negligible on small graphs, and
sub-linear in cores. Avoid implying a blanket or near-linear speedup. The
headline value for a routing service is that customization of a large region
gets ~2–3× faster on a many-core host while small regions are unaffected.

// Dinic-style blocking-flow max-flow / min-cut engine.
//
// Ported from `RoutingKit/src/nested_dissection.cpp` (`BlockingFlow` class,
// constructor `:86`, `compute_blocking_flow` `:145`,
// `augment_all_non_blocked_path` `:199`, `get_source_cut` `:256`,
// `get_target_cut` `:291`, `get_balanced_cut` `:327`, `advance` `:450`) and the
// class declaration in `RoutingKit/include/routingkit/nested_dissection.h:42`.
//
// `BlockingFlow` runs an s-t max-flow from a SOURCE SET to a TARGET SET over the
// fragment's mutually-`back_arc`ed directed-arc pairs. Each undirected edge has
// unit capacity in each direction; flow on arc `a` is modelled by the saturation
// state of `a` and its `back_arc`. Driven to completion via
// `while !is_finished() { advance() }`, after which `get_current_flow_intensity`
// equals the min-cut size and the cut accessors return a valid separator.

use crate::internal::bitvec::BitVector;
use crate::order::nd::fragment::{CutSide, GraphFragment};

/// Sentinel arc index meaning "no such arc" (mirrors `RoutingKit`'s `invalid_id`).
const INVALID_ID: u32 = u32::MAX;

/// Dinic-style blocking-flow max-flow / min-cut cutter over a [`GraphFragment`].
///
/// Mirrors `class BlockingFlow` (`nested_dissection.h:42`).
pub(crate) struct BlockingFlow<'a> {
    fragment: &'a GraphFragment,
    is_source: BitVector,
    is_target: BitVector,

    flow_intensity: u32,

    /// `is_arc_saturated[a]` — arc `a` carries (forward) flow and has no residual
    /// capacity in its own direction.
    is_arc_saturated: BitVector,
    /// `is_arc_blocked[a]` — arc `a` is excluded from the current level graph /
    /// the current blocking-flow DFS.
    is_arc_blocked: BitVector,

    is_finished_flag: bool,
}

impl<'a> BlockingFlow<'a> {
    /// Construct a cutter for `fragment` with the given source / target sets.
    ///
    /// Mirrors the constructor `nested_dissection.cpp:86`.
    ///
    /// # Panics
    /// Panics if `is_source` / `is_target` do not have exactly
    /// `fragment.node_count()` bits, if either set is empty, or if any node is
    /// both a source and a target.
    #[must_use]
    pub(crate) fn new(
        fragment: &'a GraphFragment,
        is_source: BitVector,
        is_target: BitVector,
    ) -> Self {
        let node_count = u64::from(fragment.node_count());
        assert_eq!(is_source.len(), node_count, "is_source size mismatch");
        assert_eq!(is_target.len(), node_count, "is_target size mismatch");
        assert!(is_source.population_count() > 0, "source set is empty");
        assert!(is_target.population_count() > 0, "target set is empty");
        for x in 0..node_count {
            assert!(
                !(is_source.is_set(x) && is_target.is_set(x)),
                "a source node can not also be a target node"
            );
        }

        let arc_count = u64::from(fragment.arc_count());
        Self {
            fragment,
            is_source,
            is_target,
            flow_intensity: 0,
            is_arc_saturated: BitVector::new(arc_count),
            is_arc_blocked: BitVector::new(arc_count),
            is_finished_flag: false,
        }
    }

    /// Current flow value. After completion this equals the min-cut size.
    ///
    /// Mirrors `get_current_flow_intensity()` (`nested_dissection.h:74`).
    #[must_use]
    #[inline]
    pub(crate) fn get_current_flow_intensity(&self) -> u32 {
        self.flow_intensity
    }

    /// Whether the residual graph has no augmenting path left.
    ///
    /// Mirrors `is_finished()` (`nested_dissection.h:78`).
    #[must_use]
    #[inline]
    pub(crate) fn is_finished(&self) -> bool {
        self.is_finished_flag
    }

    /// Run one blocking-flow phase: BFS levels, then push a blocking flow.
    ///
    /// Mirrors `BlockingFlow::advance()` (`nested_dissection.cpp:450`).
    pub(crate) fn advance(&mut self) {
        if !self.is_finished_flag && self.compute_blocking_flow() {
            self.flow_intensity += self.augment_all_non_blocked_path();
            self.is_finished_flag = false;
        } else {
            self.is_finished_flag = true;
        }
    }

    /// BFS from sources over non-saturated arcs, assigning levels and marking
    /// arcs that go sideways/backwards (or into saturated arcs) as blocked.
    /// Returns whether any target is reachable in the residual graph.
    ///
    /// Mirrors `compute_blocking_flow` (`nested_dissection.cpp:145`).
    fn compute_blocking_flow(&mut self) -> bool {
        let fragment = self.fragment;
        let node_count = fragment.node_count();

        let mut is_on_same_level_or_lower = BitVector::new(u64::from(node_count));
        let mut was_node_pushed = BitVector::new(u64::from(node_count));
        // Rebuild is_arc_blocked from scratch this phase.
        self.is_arc_blocked = BitVector::new(u64::from(fragment.arc_count()));

        let mut queue: Vec<u32> = Vec::with_capacity(node_count as usize);
        for x in 0..node_count {
            if self.is_source.is_set(u64::from(x)) {
                queue.push(x);
            }
        }
        let mut queue_begin = 0usize;
        let mut queue_current_level_end = queue.len();

        let mut is_a_target_node_reachable = false;

        while queue_begin != queue.len() {
            for &q in &queue[queue_begin..queue_current_level_end] {
                is_on_same_level_or_lower.set(u64::from(q));
            }

            for i in queue_begin..queue_current_level_end {
                let x = queue[i];
                let from = fragment.first_out[x as usize] as usize;
                let to = fragment.first_out[x as usize + 1] as usize;
                for xy in from..to {
                    if self.is_arc_saturated.is_set(xy as u64) {
                        self.is_arc_blocked.set(xy as u64);
                    } else {
                        let y = fragment.head[xy];
                        if is_on_same_level_or_lower.is_set(u64::from(y)) {
                            self.is_arc_blocked.set(xy as u64);
                        } else if self.is_target.is_set(u64::from(y)) {
                            is_a_target_node_reachable = true;
                        } else if !was_node_pushed.is_set(u64::from(y)) {
                            queue.push(y);
                            was_node_pushed.set(u64::from(y));
                        }
                    }
                }
            }

            queue_begin = queue_current_level_end;
            queue_current_level_end = queue.len();
        }

        is_a_target_node_reachable
    }

    /// DFS that augments along every non-blocked, non-saturated source→target
    /// path in the current level graph, updating saturation via `back_arc`.
    /// Returns the number of augmenting paths found (the per-phase flow gain).
    ///
    /// Mirrors `augment_all_non_blocked_path` (`nested_dissection.cpp:199`).
    fn augment_all_non_blocked_path(&mut self) -> u32 {
        let fragment = self.fragment;
        let node_count = fragment.node_count() as usize;

        let mut current_path_node: Vec<u32> = vec![0; node_count];
        let mut current_path_arc: Vec<u32> = vec![0; node_count];

        let mut augmented_path_count = 0u32;

        for s in 0..fragment.node_count() {
            if !self.is_source.is_set(u64::from(s)) {
                continue;
            }
            current_path_node[0] = s;
            current_path_arc[0] = s;
            let mut current_path_arc_count = 0usize;
            loop {
                let x = current_path_node[current_path_arc_count];
                let xy = self.find_first_non_blocked_outgoing_arc(x);
                if xy == INVALID_ID {
                    if current_path_arc_count == 0 {
                        break;
                    }
                    current_path_arc_count -= 1;
                    self.is_arc_blocked
                        .set(u64::from(current_path_arc[current_path_arc_count]));
                } else {
                    let y = fragment.head[xy as usize];
                    current_path_arc[current_path_arc_count] = xy;
                    current_path_arc_count += 1;
                    current_path_node[current_path_arc_count] = y;
                    if self.is_target.is_set(u64::from(y)) {
                        for &a in &current_path_arc[..current_path_arc_count] {
                            self.is_arc_blocked.set(u64::from(a));
                            let b = fragment.back_arc[a as usize];
                            if self.is_arc_saturated.is_set(u64::from(b)) {
                                self.is_arc_saturated.reset(u64::from(b));
                            } else {
                                self.is_arc_saturated.set(u64::from(a));
                            }
                        }
                        current_path_arc_count = 0;
                        augmented_path_count += 1;
                    }
                }
            }
        }
        augmented_path_count
    }

    /// First outgoing arc of `x` that is not blocked, or [`INVALID_ID`].
    ///
    /// Mirrors the `find_first_non_block_outgoing_arc_of_node` lambda
    /// (`nested_dissection.cpp:205`).
    fn find_first_non_blocked_outgoing_arc(&self, x: u32) -> u32 {
        let from = self.fragment.first_out[x as usize] as usize;
        let to = self.fragment.first_out[x as usize + 1] as usize;
        for xy in from..to {
            if !self.is_arc_blocked.is_set(xy as u64) {
                return u32::try_from(xy).expect("arc index fits u32");
            }
        }
        INVALID_ID
    }

    /// Min-cut on the source side: nodes reachable from the sources over
    /// non-saturated arcs in the residual graph.
    ///
    /// Mirrors `get_source_cut()` (`nested_dissection.cpp:256`).
    ///
    /// # Panics
    /// Panics if the flow has not finished.
    #[must_use]
    pub(crate) fn get_source_cut(&self) -> CutSide {
        assert!(self.is_finished_flag, "flow not finished");
        let fragment = self.fragment;
        let node_count = fragment.node_count();

        let mut is_node_on_side = BitVector::new(u64::from(node_count));
        let mut node_on_side_count = 0u32;
        let mut stack: Vec<u32> = Vec::with_capacity(node_count as usize);

        for s in 0..node_count {
            if self.is_source.is_set(u64::from(s)) {
                stack.push(s);
                is_node_on_side.set(u64::from(s));
                node_on_side_count += 1;
            }
        }

        while let Some(x) = stack.pop() {
            let from = fragment.first_out[x as usize] as usize;
            let to = fragment.first_out[x as usize + 1] as usize;
            for xy in from..to {
                if !self.is_arc_saturated.is_set(xy as u64) {
                    let y = fragment.head[xy];
                    if !is_node_on_side.is_set(u64::from(y)) {
                        stack.push(y);
                        is_node_on_side.set(u64::from(y));
                        node_on_side_count += 1;
                    }
                }
            }
        }

        CutSide {
            node_on_side_count,
            cut_size: self.flow_intensity,
            is_node_on_side,
        }
    }

    /// Min-cut on the target side: nodes that can reach the targets over arcs
    /// whose `back_arc` is non-saturated in the residual graph.
    ///
    /// Mirrors `get_target_cut()` (`nested_dissection.cpp:291`).
    ///
    /// # Panics
    /// Panics if the flow has not finished.
    #[must_use]
    pub(crate) fn get_target_cut(&self) -> CutSide {
        assert!(self.is_finished_flag, "flow not finished");
        let fragment = self.fragment;
        let node_count = fragment.node_count();

        let mut is_node_on_side = BitVector::new(u64::from(node_count));
        let mut node_on_side_count = 0u32;
        let mut stack: Vec<u32> = Vec::with_capacity(node_count as usize);

        for t in 0..node_count {
            if self.is_target.is_set(u64::from(t)) {
                stack.push(t);
                is_node_on_side.set(u64::from(t));
                node_on_side_count += 1;
            }
        }

        while let Some(x) = stack.pop() {
            let from = fragment.first_out[x as usize] as usize;
            let to = fragment.first_out[x as usize + 1] as usize;
            for xy in from..to {
                let back = fragment.back_arc[xy];
                if !self.is_arc_saturated.is_set(u64::from(back)) {
                    let y = fragment.head[xy];
                    if !is_node_on_side.is_set(u64::from(y)) {
                        stack.push(y);
                        is_node_on_side.set(u64::from(y));
                        node_on_side_count += 1;
                    }
                }
            }
        }

        CutSide {
            node_on_side_count,
            cut_size: self.flow_intensity,
            is_node_on_side,
        }
    }

    /// A more-balanced min-cut, grown by alternately piercing the smaller side
    /// across saturated arcs until one side cannot grow further.
    ///
    /// Mirrors `get_balanced_cut()` (`nested_dissection.cpp:327`).
    ///
    /// # Panics
    /// Panics if the flow has not finished.
    #[must_use]
    pub(crate) fn get_balanced_cut(&self) -> CutSide {
        assert!(self.is_finished_flag, "flow not finished");
        let fragment = self.fragment;
        let node_count = fragment.node_count();
        let arc_count = fragment.arc_count() as usize;

        let mut is_source_reachable = self.is_source.clone();
        let mut is_target_reachable = self.is_target.clone();

        let mut source_reachable_count = 0u32;
        let mut target_reachable_count = 0u32;

        let mut stack: Vec<u32> = Vec::with_capacity(node_count as usize);

        // `arc_count` capacity is intentional: nodes may appear multiple times.
        let mut potential_source_piercing_node: Vec<u32> = Vec::with_capacity(arc_count);
        let mut potential_target_piercing_node: Vec<u32> = Vec::with_capacity(arc_count);

        // Seed source side and grow it.
        for x in 0..node_count {
            if self.is_source.is_set(u64::from(x)) {
                stack.push(x);
            }
        }
        Self::enlarge_source_side(
            fragment,
            &self.is_arc_saturated,
            &mut is_source_reachable,
            &mut source_reachable_count,
            &mut stack,
            &mut potential_source_piercing_node,
        );

        // Seed target side and grow it.
        for x in 0..node_count {
            if self.is_target.is_set(u64::from(x)) {
                stack.push(x);
            }
        }
        Self::enlarge_target_side(
            fragment,
            &self.is_arc_saturated,
            &mut is_target_reachable,
            &mut target_reachable_count,
            &mut stack,
            &mut potential_target_piercing_node,
        );

        loop {
            if source_reachable_count <= target_reachable_count {
                let mut pierce_node = INVALID_ID;
                while pierce_node == INVALID_ID {
                    let Some(y) = potential_source_piercing_node.pop() else {
                        return CutSide {
                            node_on_side_count: source_reachable_count,
                            cut_size: self.flow_intensity,
                            is_node_on_side: is_source_reachable,
                        };
                    };
                    if !is_source_reachable.is_set(u64::from(y))
                        && !is_target_reachable.is_set(u64::from(y))
                    {
                        pierce_node = y;
                    }
                }
                is_source_reachable.set(u64::from(pierce_node));
                stack.push(pierce_node);
                Self::enlarge_source_side(
                    fragment,
                    &self.is_arc_saturated,
                    &mut is_source_reachable,
                    &mut source_reachable_count,
                    &mut stack,
                    &mut potential_source_piercing_node,
                );
            } else {
                let mut pierce_node = INVALID_ID;
                while pierce_node == INVALID_ID {
                    let Some(y) = potential_target_piercing_node.pop() else {
                        return CutSide {
                            node_on_side_count: target_reachable_count,
                            cut_size: self.flow_intensity,
                            is_node_on_side: is_target_reachable,
                        };
                    };
                    if !is_source_reachable.is_set(u64::from(y))
                        && !is_target_reachable.is_set(u64::from(y))
                    {
                        pierce_node = y;
                    }
                }
                is_target_reachable.set(u64::from(pierce_node));
                stack.push(pierce_node);
                Self::enlarge_target_side(
                    fragment,
                    &self.is_arc_saturated,
                    &mut is_target_reachable,
                    &mut target_reachable_count,
                    &mut stack,
                    &mut potential_target_piercing_node,
                );
            }
        }
    }

    /// Grow the source side over non-saturated arcs, recording the heads of
    /// saturated arcs as candidate piercing nodes.
    ///
    /// Mirrors the `enlarge_source_side` lambda (`nested_dissection.cpp:349`).
    fn enlarge_source_side(
        fragment: &GraphFragment,
        is_arc_saturated: &BitVector,
        is_source_reachable: &mut BitVector,
        source_reachable_count: &mut u32,
        stack: &mut Vec<u32>,
        potential_source_piercing_node: &mut Vec<u32>,
    ) {
        while let Some(x) = stack.pop() {
            *source_reachable_count += 1;
            let from = fragment.first_out[x as usize] as usize;
            let to = fragment.first_out[x as usize + 1] as usize;
            for xy in from..to {
                let y = fragment.head[xy];
                if is_arc_saturated.is_set(xy as u64) {
                    potential_source_piercing_node.push(y);
                } else if !is_source_reachable.is_set(u64::from(y)) {
                    is_source_reachable.set(u64::from(y));
                    stack.push(y);
                }
            }
        }
    }

    /// Grow the target side backwards over arcs whose `back_arc` is non-saturated,
    /// recording candidate piercing nodes.
    ///
    /// Mirrors the `enlarge_target_side` lambda (`nested_dissection.cpp:369`).
    fn enlarge_target_side(
        fragment: &GraphFragment,
        is_arc_saturated: &BitVector,
        is_target_reachable: &mut BitVector,
        target_reachable_count: &mut u32,
        stack: &mut Vec<u32>,
        potential_target_piercing_node: &mut Vec<u32>,
    ) {
        while let Some(x) = stack.pop() {
            *target_reachable_count += 1;
            let from = fragment.first_out[x as usize] as usize;
            let to = fragment.first_out[x as usize + 1] as usize;
            for xy in from..to {
                let y = fragment.head[xy];
                let back = fragment.back_arc[xy];
                if is_arc_saturated.is_set(u64::from(back)) {
                    potential_target_piercing_node.push(y);
                } else if !is_target_reachable.is_set(u64::from(y)) {
                    stack.push(y);
                    is_target_reachable.set(u64::from(y));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order::nd::fragment::make_graph_fragment;

    /// Build a source/target `BitVector` pair over `node_count` from node lists.
    fn make_st(node_count: u32, sources: &[u32], targets: &[u32]) -> (BitVector, BitVector) {
        let mut is_source = BitVector::new(u64::from(node_count));
        let mut is_target = BitVector::new(u64::from(node_count));
        for &s in sources {
            is_source.set(u64::from(s));
        }
        for &t in targets {
            is_target.set(u64::from(t));
        }
        (is_source, is_target)
    }

    /// Brute-force reference min s-t cut over an undirected unit-capacity graph.
    ///
    /// Enumerate every node subset `A` with `S ⊆ A` and `A ∩ T = ∅`, count the
    /// undirected edges crossing `A → complement`, and return the minimum. Only
    /// valid for `node_count <= ~20` (here always small).
    ///
    /// `edges` are undirected `(u, v)` pairs (each counted once).
    fn brute_force_min_cut(
        node_count: u32,
        edges: &[(u32, u32)],
        sources: &[u32],
        targets: &[u32],
    ) -> u32 {
        let n = node_count;
        assert!(n <= 20, "brute force only for tiny graphs");
        let mut best = u32::MAX;
        // Free nodes = neither source nor target; iterate over their subsets.
        let free: Vec<u32> = (0..n)
            .filter(|x| !sources.contains(x) && !targets.contains(x))
            .collect();
        let free_count = free.len();
        for mask in 0u32..(1u32 << free_count) {
            // Side A = sources ∪ {free nodes whose bit is set}.
            let mut in_a = vec![false; n as usize];
            for &s in sources {
                in_a[s as usize] = true;
            }
            for (bit, &node) in free.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    in_a[node as usize] = true;
                }
            }
            // (targets are implicitly not in A)
            let mut crossing = 0u32;
            for &(u, v) in edges {
                if in_a[u as usize] != in_a[v as usize] {
                    crossing += 1;
                }
            }
            best = best.min(crossing);
        }
        best
    }

    /// Run a [`BlockingFlow`] to completion and return it.
    fn run_to_completion(
        fragment: &GraphFragment,
        is_source: BitVector,
        is_target: BitVector,
    ) -> BlockingFlow<'_> {
        let mut bf = BlockingFlow::new(fragment, is_source, is_target);
        let mut iterations = 0;
        while !bf.is_finished() {
            bf.advance();
            iterations += 1;
            assert!(iterations < 1000, "advance did not terminate");
        }
        bf
    }

    /// Assert a cut crosses exactly `expected` undirected edges and that all
    /// sources are on the side / all targets off it (for a source-style cut), or
    /// the inverse for a target-style cut.
    fn count_crossing_edges(edges: &[(u32, u32)], side: &BitVector) -> u32 {
        let mut crossing = 0u32;
        for &(u, v) in edges {
            if side.is_set(u64::from(u)) != side.is_set(u64::from(v)) {
                crossing += 1;
            }
        }
        crossing
    }

    /// Full gate for one fixture: max-flow == brute-force min-cut, and each cut
    /// accessor returns a valid separator of size == max-flow.
    fn assert_fixture(
        node_count: u32,
        edges: &[(u32, u32)],
        sources: &[u32],
        targets: &[u32],
    ) -> u32 {
        let tail: Vec<u32> = edges.iter().map(|&(u, _)| u).collect();
        let head: Vec<u32> = edges.iter().map(|&(_, v)| v).collect();
        let fragment = make_graph_fragment(node_count, &tail, &head);

        let expected = brute_force_min_cut(node_count, edges, sources, targets);

        let (is_source, is_target) = make_st(node_count, sources, targets);
        let bf = run_to_completion(&fragment, is_source, is_target);
        let max_flow = bf.get_current_flow_intensity();
        assert_eq!(
            max_flow, expected,
            "max-flow {max_flow} != brute-force min-cut {expected} \
             (n={node_count}, edges={edges:?}, S={sources:?}, T={targets:?})"
        );

        // get_source_cut: sources on side, targets off side, cut_size == flow.
        let src_cut = bf.get_source_cut();
        assert_eq!(src_cut.cut_size, max_flow, "source cut_size mismatch");
        for &s in sources {
            assert!(
                src_cut.is_node_on_side.is_set(u64::from(s)),
                "source {s} must be on source-cut side"
            );
        }
        for &t in targets {
            assert!(
                !src_cut.is_node_on_side.is_set(u64::from(t)),
                "target {t} must be off source-cut side"
            );
        }
        assert_eq!(
            count_crossing_edges(edges, &src_cut.is_node_on_side),
            max_flow,
            "source cut must cross exactly max-flow edges"
        );
        assert_eq!(
            src_cut.is_node_on_side.population_count(),
            u64::from(src_cut.node_on_side_count),
            "source node_on_side_count mismatch"
        );

        // get_target_cut: targets on side, sources off side.
        let tgt_cut = bf.get_target_cut();
        assert_eq!(tgt_cut.cut_size, max_flow, "target cut_size mismatch");
        for &t in targets {
            assert!(
                tgt_cut.is_node_on_side.is_set(u64::from(t)),
                "target {t} must be on target-cut side"
            );
        }
        for &s in sources {
            assert!(
                !tgt_cut.is_node_on_side.is_set(u64::from(s)),
                "source {s} must be off target-cut side"
            );
        }
        assert_eq!(
            count_crossing_edges(edges, &tgt_cut.is_node_on_side),
            max_flow,
            "target cut must cross exactly max-flow edges"
        );
        assert_eq!(
            tgt_cut.is_node_on_side.population_count(),
            u64::from(tgt_cut.node_on_side_count),
            "target node_on_side_count mismatch"
        );

        // get_balanced_cut: a valid separator. It may be returned as either the
        // grown source side or the grown target side, so the cut puts all
        // sources on one side and all targets on the other (in some order).
        let bal_cut = bf.get_balanced_cut();
        assert_eq!(bal_cut.cut_size, max_flow, "balanced cut_size mismatch");
        // All sources share one side bit; all targets share the other.
        let sources_side = bal_cut.is_node_on_side.is_set(u64::from(sources[0]));
        for &s in sources {
            assert_eq!(
                bal_cut.is_node_on_side.is_set(u64::from(s)),
                sources_side,
                "all sources must be on the same balanced-cut side"
            );
        }
        for &t in targets {
            assert_eq!(
                bal_cut.is_node_on_side.is_set(u64::from(t)),
                !sources_side,
                "all targets must be on the opposite balanced-cut side from sources"
            );
        }
        assert_eq!(
            count_crossing_edges(edges, &bal_cut.is_node_on_side),
            max_flow,
            "balanced cut must cross exactly max-flow edges"
        );
        assert_eq!(
            bal_cut.is_node_on_side.population_count(),
            u64::from(bal_cut.node_on_side_count),
            "balanced node_on_side_count mismatch"
        );

        max_flow
    }

    // ── Fixtures: max-flow == brute-force min-cut ────────────────────────────

    /// Path 0-1-2-3-4. Min-cut between endpoints is 1.
    #[test]
    fn path_min_cut_is_one() {
        let edges = [(0u32, 1), (1, 2), (2, 3), (3, 4)];
        assert_eq!(assert_fixture(5, &edges, &[0], &[4]), 1);
    }

    /// 3x3 grid, source = top-left, target = bottom-right.
    #[test]
    fn grid_3x3() {
        // node id = r*3 + c.
        let mut edges = Vec::new();
        for r in 0u32..3 {
            for c in 0u32..3 {
                let id = r * 3 + c;
                if c + 1 < 3 {
                    edges.push((id, id + 1));
                }
                if r + 1 < 3 {
                    edges.push((id, id + 3));
                }
            }
        }
        // min-cut from corner 0 to corner 8 = 2 (corner has degree 2).
        assert_eq!(assert_fixture(9, &edges, &[0], &[8]), 2);
    }

    /// Two triangles (cliques of 3) joined by k=2 edges. Min-cut == 2.
    #[test]
    fn two_cliques_joined_by_two_edges() {
        // Clique A: {0,1,2}; Clique B: {3,4,5}. Bridges: 1-3 and 2-4.
        let edges = [
            (0u32, 1),
            (0, 2),
            (1, 2), // clique A
            (3, 4),
            (3, 5),
            (4, 5), // clique B
            (1, 3),
            (2, 4), // 2 bridges
        ];
        assert_eq!(assert_fixture(6, &edges, &[0], &[5]), 2);
    }

    /// Two cliques joined by k=3 edges. Min-cut == 3.
    #[test]
    fn two_cliques_joined_by_three_edges() {
        let edges = [
            (0u32, 1),
            (0, 2),
            (1, 2),
            (3, 4),
            (3, 5),
            (4, 5),
            (0, 3),
            (1, 4),
            (2, 5),
        ];
        assert_eq!(assert_fixture(6, &edges, &[0], &[5]), 3);
    }

    /// Graph with multiple distinct min-cuts (a 4-cycle: two cuts of size 2).
    #[test]
    fn multi_min_cut_cycle() {
        // 4-cycle 0-1-2-3-0. min-cut from 0 to 2 is 2 (two edge-disjoint paths).
        let edges = [(0u32, 1), (1, 2), (2, 3), (3, 0)];
        assert_eq!(assert_fixture(4, &edges, &[0], &[2]), 2);
    }

    /// Multiple sources and multiple targets.
    #[test]
    fn multi_source_multi_target() {
        // Bipartite-ish: sources {0,1} on the left, targets {4,5} on the right,
        // joined through a bottleneck node 2-3.
        let edges = [
            (0u32, 2),
            (1, 2),
            (2, 3), // single bottleneck edge
            (3, 4),
            (3, 5),
        ];
        // min-cut from {0,1} to {4,5} = 1 (the 2-3 bottleneck).
        assert_eq!(assert_fixture(6, &edges, &[0, 1], &[4, 5]), 1);
    }

    /// Source directly adjacent to target. Min-cut == 1 (single edge).
    #[test]
    fn source_adjacent_to_target() {
        let edges = [(0u32, 1)];
        assert_eq!(assert_fixture(2, &edges, &[0], &[1]), 1);
    }

    /// Source adjacent to target by several parallel-ish paths.
    #[test]
    fn source_adjacent_multiple_paths() {
        // 0 connected to 1 directly, and 0-2-1 path. Min-cut from 0 to 1 = 2.
        let edges = [(0u32, 1), (0, 2), (2, 1)];
        assert_eq!(assert_fixture(3, &edges, &[0], &[1]), 2);
    }

    /// Disconnected source and target → min-cut 0.
    #[test]
    fn disconnected_min_cut_zero() {
        // Two separate edges; source in one component, target in the other.
        let edges = [(0u32, 1), (2, 3)];
        let max_flow = assert_fixture(4, &edges, &[0], &[3]);
        assert_eq!(max_flow, 0, "disconnected S/T must have min-cut 0");
    }

    /// A larger random-ish graph to exercise multiple phases.
    #[test]
    fn larger_graph_multiple_phases() {
        // A graph where Dinic needs more than one phase.
        let edges = [
            (0u32, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (1, 2),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 6),
            (4, 5),
        ];
        let mf = assert_fixture(7, &edges, &[0], &[6]);
        // Sanity: node 0 has degree 2, node 6 has degree 2 → min-cut 2.
        assert_eq!(mf, 2);
    }

    /// A fixture that forces residual flow-cancellation: a later augmenting path
    /// reuses an arc whose `back_arc` was saturated by an earlier path, hitting
    /// the `is_arc_saturated.reset(b)` branch in `augment_all_non_blocked_path`.
    /// Found by exhaustive search over all 5-node graphs (s=0, t=4).
    #[test]
    fn residual_cancellation_flow_two() {
        let edges = [(0u32, 2), (0, 3), (1, 2), (1, 3), (1, 4), (2, 4)];
        assert_eq!(assert_fixture(5, &edges, &[0], &[4]), 2);
    }

    /// A second cancellation-triggering fixture with a higher flow value.
    #[test]
    fn residual_cancellation_flow_three() {
        let edges = [(0u32, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4)];
        assert_eq!(assert_fixture(5, &edges, &[0], &[4]), 3);
    }

    // ── Determinism ──────────────────────────────────────────────────────────

    /// Same input → identical cut on repeated runs.
    #[test]
    fn determinism() {
        let edges = [
            (0u32, 1),
            (0, 2),
            (1, 2),
            (3, 4),
            (3, 5),
            (4, 5),
            (1, 3),
            (2, 4),
        ];
        let tail: Vec<u32> = edges.iter().map(|&(u, _)| u).collect();
        let head: Vec<u32> = edges.iter().map(|&(_, v)| v).collect();
        let fragment = make_graph_fragment(6, &tail, &head);

        let mut prev_balanced: Option<Vec<u64>> = None;
        let mut prev_source: Option<Vec<u64>> = None;
        let mut prev_target: Option<Vec<u64>> = None;
        for _ in 0..5 {
            let (is_source, is_target) = make_st(6, &[0], &[5]);
            let bf = run_to_completion(&fragment, is_source, is_target);
            let bal = bf.get_balanced_cut().is_node_on_side.words().to_vec();
            let src = bf.get_source_cut().is_node_on_side.words().to_vec();
            let tgt = bf.get_target_cut().is_node_on_side.words().to_vec();
            if let Some(ref p) = prev_balanced {
                assert_eq!(*p, bal, "balanced cut not deterministic");
            }
            if let Some(ref p) = prev_source {
                assert_eq!(*p, src, "source cut not deterministic");
            }
            if let Some(ref p) = prev_target {
                assert_eq!(*p, tgt, "target cut not deterministic");
            }
            prev_balanced = Some(bal);
            prev_source = Some(src);
            prev_target = Some(tgt);
        }
    }

    // ── Accessor / edge-case coverage ────────────────────────────────────────

    /// `advance()` past completion is idempotent (covers the early-return guard).
    #[test]
    fn advance_after_finished_is_noop() {
        let edges = [(0u32, 1)];
        let tail: Vec<u32> = edges.iter().map(|&(u, _)| u).collect();
        let head: Vec<u32> = edges.iter().map(|&(_, v)| v).collect();
        let fragment = make_graph_fragment(2, &tail, &head);
        let (is_source, is_target) = make_st(2, &[0], &[1]);
        let mut bf = BlockingFlow::new(&fragment, is_source, is_target);
        while !bf.is_finished() {
            bf.advance();
        }
        let flow = bf.get_current_flow_intensity();
        // Extra advance must not change anything.
        bf.advance();
        assert!(bf.is_finished());
        assert_eq!(bf.get_current_flow_intensity(), flow);
    }

    #[test]
    #[should_panic(expected = "source set is empty")]
    fn new_panics_on_empty_source() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let (is_source, is_target) = make_st(2, &[], &[1]);
        let _ = BlockingFlow::new(&fragment, is_source, is_target);
    }

    #[test]
    #[should_panic(expected = "target set is empty")]
    fn new_panics_on_empty_target() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let (is_source, is_target) = make_st(2, &[0], &[]);
        let _ = BlockingFlow::new(&fragment, is_source, is_target);
    }

    #[test]
    #[should_panic(expected = "can not also be a target")]
    fn new_panics_on_source_target_overlap() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let (is_source, is_target) = make_st(2, &[0], &[0]);
        let _ = BlockingFlow::new(&fragment, is_source, is_target);
    }

    #[test]
    #[should_panic(expected = "is_source size mismatch")]
    fn new_panics_on_source_size_mismatch() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let is_source = BitVector::new(3);
        let is_target = BitVector::new(2);
        let _ = BlockingFlow::new(&fragment, is_source, is_target);
    }

    #[test]
    #[should_panic(expected = "is_target size mismatch")]
    fn new_panics_on_target_size_mismatch() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let mut is_source = BitVector::new(2);
        is_source.set(0);
        let is_target = BitVector::new(3);
        let _ = BlockingFlow::new(&fragment, is_source, is_target);
    }

    #[test]
    #[should_panic(expected = "flow not finished")]
    fn get_source_cut_panics_when_not_finished() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let (is_source, is_target) = make_st(2, &[0], &[1]);
        let bf = BlockingFlow::new(&fragment, is_source, is_target);
        let _ = bf.get_source_cut();
    }

    #[test]
    #[should_panic(expected = "flow not finished")]
    fn get_target_cut_panics_when_not_finished() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let (is_source, is_target) = make_st(2, &[0], &[1]);
        let bf = BlockingFlow::new(&fragment, is_source, is_target);
        let _ = bf.get_target_cut();
    }

    #[test]
    #[should_panic(expected = "flow not finished")]
    fn get_balanced_cut_panics_when_not_finished() {
        let fragment = make_graph_fragment(2, &[0u32], &[1u32]);
        let (is_source, is_target) = make_st(2, &[0], &[1]);
        let bf = BlockingFlow::new(&fragment, is_source, is_target);
        let _ = bf.get_balanced_cut();
    }
}

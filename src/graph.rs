//! Input graph type in Compressed-Sparse-Row (CSR) format.

/// Node identifier (index into the CSR `first_out` array).
pub type NodeId = u32;

/// Arc identifier (index into the CSR `head`/`weight` arrays).
pub type ArcId = u32;

/// A directed, weighted graph stored in Compressed-Sparse-Row format.
///
/// Invariants (caller-enforced at construction time):
/// - `first_out.len() == node_count + 1`
/// - `head.len() == arc_count` and `weight.len() == arc_count`
/// - `first_out` is non-decreasing; `first_out[0] == 0`; `first_out[node_count] == arc_count`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    /// CSR row-pointer array; length `node_count + 1`.
    pub first_out: Vec<u32>,
    /// Arc target nodes; length `arc_count`.
    pub head: Vec<u32>,
    /// Arc weights; length `arc_count`.
    pub weight: Vec<u32>,
}

impl Graph {
    /// Returns the number of nodes in the graph.
    #[must_use]
    #[inline]
    pub fn node_count(&self) -> usize {
        self.first_out.len() - 1
    }

    /// Returns the number of arcs in the graph.
    #[must_use]
    #[inline]
    pub fn arc_count(&self) -> usize {
        self.head.len()
    }
}

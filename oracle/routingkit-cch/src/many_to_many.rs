//! Safe Rust wrapper for the bucket-based many-to-many distance matrix
//! function. Calls the C++ implementation in `routingkit_cch_wrapper.cc`
//! through the `ffi::cch_compute_distance_matrix` bridge.

use crate::CCHMetric;
use crate::ffi;

/// Distance value used when a target is unreachable from a given source.
/// Mirrors RoutingKit's `inf_weight` convention.
pub const UNREACHABLE: u32 = u32::MAX;

/// Compute the shortest-path distance matrix between `sources` and `targets`.
///
/// Returns a row-major `Vec<u32>` of length `sources.len() * targets.len()`.
/// Element at index `i * targets.len() + j` is the shortest-path distance
/// from `sources[i]` to `targets[j]`, or [`UNREACHABLE`] if no path exists.
///
/// Both slices may be empty; the result is then empty as well.
///
/// Internally the C++ side pins `targets` once and runs a one-to-many query
/// per source — i.e. it's a true bucket-based call, not a loop of P2P
/// queries from the Rust side.
pub fn distance_matrix(metric: &CCHMetric<'_>, sources: &[u32], targets: &[u32]) -> Vec<u32> {
    if sources.is_empty() || targets.is_empty() {
        return Vec::new();
    }
    // SAFETY: `metric` is live for the duration of this call. The C++ side
    // creates its own CustomizableContractionHierarchyQuery on the stack,
    // borrows the metric for the call only, and does not retain any
    // references after returning.
    unsafe { ffi::cch_compute_distance_matrix(metric.inner.as_ref().unwrap(), sources, targets) }
}

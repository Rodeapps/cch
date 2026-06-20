#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "rust/cxx.h"

// RoutingKit headers
#include <routingkit/customizable_contraction_hierarchy.h>

struct CCH
{
    RoutingKit::CustomizableContractionHierarchy inner;
    explicit CCH(RoutingKit::CustomizableContractionHierarchy &&x) : inner(std::move(x)) {}
};

struct CCHMetric
{
    RoutingKit::CustomizableContractionHierarchyMetric inner;
    explicit CCHMetric(RoutingKit::CustomizableContractionHierarchyMetric &&x) : inner(std::move(x)) {}
};

struct CCHQuery
{
    RoutingKit::CustomizableContractionHierarchyQuery inner;
    explicit CCHQuery(RoutingKit::CustomizableContractionHierarchyQuery &&x) : inner(std::move(x)) {}
};

struct CCHPartial
{
    RoutingKit::CustomizableContractionHierarchyPartialCustomization inner;
    explicit CCHPartial(const RoutingKit::CustomizableContractionHierarchy &cch) : inner(cch) {}
};

std::unique_ptr<CCH> cch_new(rust::Slice<const uint32_t> order,
                             rust::Slice<const uint32_t> tail,
                             rust::Slice<const uint32_t> head,
                             rust::Fn<void(rust::Str)> log_message,
                             bool filter_always_inf_arcs);

uint32_t cch_node_count(const CCH& cch);
uint32_t cch_arc_count(const CCH& cch);
std::unique_ptr<CCHMetric> cch_metric_new(const CCH &cch, rust::Slice<const uint32_t> weight);
void cch_metric_customize(CCHMetric &metric);
void cch_metric_parallel_customize(CCHMetric &metric, uint32_t thread_count);
std::unique_ptr<CCHQuery> cch_query_new(const CCHMetric &metric);
void cch_query_reset(CCHQuery &query, const CCHMetric &metric);
void cch_query_add_source(CCHQuery &query, uint32_t s, uint32_t dist);
void cch_query_add_target(CCHQuery &query, uint32_t t, uint32_t dist);
void cch_query_run(CCHQuery &query);
uint32_t cch_query_distance(const CCHQuery &query);
rust::Vec<uint32_t> cch_query_node_path(const CCHQuery &query);
rust::Vec<uint32_t> cch_query_arc_path(const CCHQuery &query);
rust::Vec<uint32_t> cch_compute_order_inertial(
    uint32_t node_count,
    rust::Slice<const uint32_t> tail,
    rust::Slice<const uint32_t> head,
    rust::Slice<const float> latitude,
    rust::Slice<const float> longitude);
rust::Vec<uint32_t> cch_compute_order_degree(
    uint32_t node_count,
    rust::Slice<const uint32_t> tail,
    rust::Slice<const uint32_t> head);

/// Compute a bucket-based many-to-many distance matrix.
///
/// Pins `targets` once, then iterates `sources`, running a one-to-many query
/// per source. Returns a flat row-major vector of size
/// `sources.size() * targets.size()` where element `[i * targets.size() + j]`
/// is the shortest-path distance from `sources[i]` to `targets[j]`, or
/// `UINT32_MAX` if unreachable (RoutingKit's `inf_weight`).
///
/// `metric` must outlive the call; no references are retained.
rust::Vec<uint32_t> cch_compute_distance_matrix(
    const CCHMetric &metric,
    rust::Slice<const uint32_t> sources,
    rust::Slice<const uint32_t> targets);

// Partial customization API
std::unique_ptr<CCHPartial> cch_partial_new(const CCH &cch);
void cch_partial_reset(CCHPartial &partial);
void cch_partial_update_arc(CCHPartial &partial, uint32_t arc);
void cch_partial_customize(CCHPartial &partial, CCHMetric &metric);

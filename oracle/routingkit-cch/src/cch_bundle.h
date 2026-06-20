#pragma once

#include "routingkit_cch_wrapper.h"
#include "rust/cxx.h"

#include <memory>
#include <string>

// Phase 2a — persisted customized-CCH artifacts.
//
// Two artifact kinds, written at preprocess time, read at worker boot:
//   <base>.cch-struct           — the contraction hierarchy structure
//                                 (~12 std::vector<unsigned> + 3 BitVector).
//                                 Built from .edges + .order; deterministic.
//   <base>.cch-metric-<name>    — a single customized metric
//                                 (forward + backward weight arrays).
//                                 Built from a .weights-<name> + the CCH.
//                                 Re-customizable on its own if the metric
//                                 changes (live-traffic in the future).
//
// Loading the bundle skips the 180-s CCH::new + customize step.
//
// Bundle format (vector-eager; Phase 2b will move to mmap):
//
//   STRUCT FILE:
//     u64  magic           = 0x4343485F53545243  ("CCH_STRC")
//     u32  format_version  = 1
//     u32  reserved        = 0
//     u64  node_count
//     u64  cch_arc_count
//     u64  input_arc_count
//     For each of 12 std::vector<unsigned> sections + 3 BitVector sections,
//     in fixed order (see cch_bundle.cc for the list):
//       u64  byte_length
//       bytes...                  // raw POD; BitVectors padded to 64-bit
//
//   METRIC FILE:
//     u64  magic           = 0x4343485F4D455452  ("CCH_METR")
//     u32  format_version  = 1
//     u32  reserved        = 0
//     u64  cch_arc_count                 // must match the struct's
//     u64  forward_byte_length
//     bytes...                            // sizeof(unsigned) * cch_arc_count
//     u64  backward_byte_length
//     bytes...

// Save the CCH structure (without any metrics) to the given path.
// Throws std::runtime_error on i/o failure.
void cch_save_struct(const CCH& cch, rust::Str path);

// Load a previously-saved CCH structure. Returns ownership.
// Throws std::runtime_error on bad magic, version mismatch, or i/o failure.
std::unique_ptr<CCH> cch_load_struct(rust::Str path);

// Save a single customized metric's forward+backward weights to the given path.
// The associated CCH must be saved separately via cch_save_struct.
// Throws std::runtime_error on i/o failure.
void cch_save_metric(const CCHMetric& metric, rust::Str path);

// Load a previously-saved metric, re-attaching it to the given CCH.
// The CCH must have been loaded via cch_load_struct (or constructed via
// cch_new) with the same cch_arc_count as when the metric was saved.
// Throws std::runtime_error on bad magic, version mismatch, arc-count
// mismatch, or i/o failure.
std::unique_ptr<CCHMetric> cch_load_metric(const CCH& cch, rust::Str path);

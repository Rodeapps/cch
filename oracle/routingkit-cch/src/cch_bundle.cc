#include "cch_bundle.h"

#include <routingkit/customizable_contraction_hierarchy.h>
#include <routingkit/id_mapper.h>
#include <routingkit/vector_io.h>
#include <routingkit/bit_vector.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace RoutingKit;

namespace {

constexpr uint64_t STRUCT_MAGIC = 0x4343485F53545243ULL; // "CCH_STRC"
constexpr uint64_t METRIC_MAGIC = 0x4343485F4D455452ULL; // "CCH_METR"
constexpr uint32_t FORMAT_VERSION = 1;

// Read a u64-length-prefixed std::vector<T> from an istream. The
// byte_length in the wire format is authoritative; we don't validate
// against any expected count because several CCH fields are sized by
// derived quantities (mapper local_id_count) that aren't known until
// after the corresponding BitVector is loaded.
template <class T>
std::vector<T> read_sized_vector(std::ifstream& in, const char* section_name)
{
    uint64_t byte_length = read_value<uint64_t>(in);
    if (byte_length % sizeof(T) != 0) {
        throw std::runtime_error(std::string("section ") + section_name
            + ": byte_length " + std::to_string(byte_length)
            + " is not a multiple of element size " + std::to_string(sizeof(T)));
    }
    uint64_t count = byte_length / sizeof(T);
    return read_vector<T>(in, count);
}

// Same, with an optional sanity check against a known expected count.
// Use when the count IS known (e.g. node_count, input_arc_count from
// the header).
template <class T>
std::vector<T> read_sized_vector_exact(std::ifstream& in,
                                        uint64_t expected_count,
                                        const char* section_name)
{
    uint64_t byte_length = read_value<uint64_t>(in);
    if (byte_length != expected_count * sizeof(T)) {
        throw std::runtime_error(std::string("section ") + section_name
            + ": expected " + std::to_string(expected_count * sizeof(T))
            + " bytes, got " + std::to_string(byte_length));
    }
    return read_vector<T>(in, expected_count);
}

void write_sized_vector(std::ofstream& out, const std::vector<unsigned>& v)
{
    uint64_t byte_length = static_cast<uint64_t>(v.size()) * sizeof(unsigned);
    write_value(out, byte_length);
    if (!v.empty()) {
        out.write(reinterpret_cast<const char*>(v.data()), byte_length);
        if (!out) throw std::runtime_error("write_sized_vector: stream error");
    }
}

void write_sized_bit_vector(std::ofstream& out, const BitVector& bv)
{
    // BitVector storage is 64-bit aligned: ((size+511)/512)*64 bytes.
    // Save the bit count as a u64 so the reader can reconstruct, then
    // dump the storage as a length-prefixed byte blob.
    uint64_t bit_count = bv.size();
    uint64_t byte_length = ((bit_count + 511) / 512) * 64;
    write_value(out, bit_count);
    write_value(out, byte_length);
    if (byte_length > 0) {
        out.write(reinterpret_cast<const char*>(bv.data()), byte_length);
        if (!out) throw std::runtime_error("write_sized_bit_vector: stream error");
    }
}

BitVector read_sized_bit_vector(std::ifstream& in, const char* section_name)
{
    uint64_t bit_count = read_value<uint64_t>(in);
    uint64_t byte_length = read_value<uint64_t>(in);
    uint64_t expected_bytes = ((bit_count + 511) / 512) * 64;
    if (byte_length != expected_bytes) {
        throw std::runtime_error(std::string("section ") + section_name
            + ": bitvector byte_length " + std::to_string(byte_length)
            + " does not match expected " + std::to_string(expected_bytes)
            + " for bit_count " + std::to_string(bit_count));
    }
    BitVector v(bit_count);
    if (byte_length > 0) {
        in.read(reinterpret_cast<char*>(v.data()), byte_length);
        if (!in) {
            throw std::runtime_error(std::string("section ") + section_name
                + ": short read on bitvector data");
        }
    }
    return v;
}

std::string rust_str_to_std(rust::Str s)
{
    return std::string(s.data(), s.size());
}

} // namespace

// ---------------- struct save / load ----------------

void cch_save_struct(const CCH& cch_w, rust::Str path)
{
    const auto& cch = cch_w.inner;
    std::ofstream out(rust_str_to_std(path), std::ios::binary);
    if (!out) {
        throw std::runtime_error("cch_save_struct: cannot open " + rust_str_to_std(path));
    }

    write_value(out, STRUCT_MAGIC);
    write_value(out, FORMAT_VERSION);
    write_value(out, static_cast<uint32_t>(0));            // reserved
    write_value(out, static_cast<uint64_t>(cch.node_count()));
    write_value(out, static_cast<uint64_t>(cch.cch_arc_count()));
    write_value(out, static_cast<uint64_t>(cch.input_arc_count()));

    // Fixed-size sections (sized by node_count / cch_arc_count / input_arc_count).
    write_sized_vector(out, cch.order);                       // node_count
    write_sized_vector(out, cch.rank);                        // node_count
    write_sized_vector(out, cch.elimination_tree_parent);     // node_count
    write_sized_vector(out, cch.up_first_out);                // node_count + 1
    write_sized_vector(out, cch.up_head);                     // cch_arc_count
    write_sized_vector(out, cch.up_tail);                     // cch_arc_count
    write_sized_vector(out, cch.down_first_out);              // node_count + 1
    write_sized_vector(out, cch.down_head);                   // cch_arc_count
    write_sized_vector(out, cch.down_to_up);                  // cch_arc_count
    write_sized_vector(out, cch.input_arc_to_cch_arc);        // input_arc_count

    // BitVectors. is_input_arc_upward is sized by input_arc_count;
    // does_cch_arc_have_input_arc and does_cch_arc_have_extra_input_arc
    // by cch_arc_count. They drive the mapper sizes for the vectors below.
    write_sized_bit_vector(out, cch.is_input_arc_upward);
    write_sized_bit_vector(out, cch.does_cch_arc_have_input_arc);
    write_sized_bit_vector(out, cch.does_cch_arc_have_extra_input_arc);

    // Vectors sized by does_cch_arc_have_input_arc.local_id_count
    // (i.e. how many cch arcs have a non-extra input arc backing).
    write_sized_vector(out, cch.forward_input_arc_of_cch);
    write_sized_vector(out, cch.backward_input_arc_of_cch);

    // Vectors sized by does_cch_arc_have_extra_input_arc.local_id_count + 1
    // (CSR offsets for the extras).
    write_sized_vector(out, cch.first_extra_forward_input_arc_of_cch);
    write_sized_vector(out, cch.first_extra_backward_input_arc_of_cch);
    // Extra arrays: actual count = last element of first_extra_*.
    write_sized_vector(out, cch.extra_forward_input_arc_of_cch);
    write_sized_vector(out, cch.extra_backward_input_arc_of_cch);

    out.flush();
    if (!out) {
        throw std::runtime_error("cch_save_struct: stream error after flush");
    }
}

std::unique_ptr<CCH> cch_load_struct(rust::Str path)
{
    std::ifstream in(rust_str_to_std(path), std::ios::binary);
    if (!in) {
        throw std::runtime_error("cch_load_struct: cannot open " + rust_str_to_std(path));
    }

    uint64_t magic = read_value<uint64_t>(in);
    if (magic != STRUCT_MAGIC) {
        throw std::runtime_error("cch_load_struct: bad magic in " + rust_str_to_std(path));
    }
    uint32_t version = read_value<uint32_t>(in);
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("cch_load_struct: unsupported version "
            + std::to_string(version));
    }
    read_value<uint32_t>(in); // reserved
    uint64_t node_count = read_value<uint64_t>(in);
    uint64_t cch_arc_count = read_value<uint64_t>(in);
    uint64_t input_arc_count = read_value<uint64_t>(in);

    CustomizableContractionHierarchy raw;
    // Fixed-size sections (header-derived counts, exact-check OK).
    raw.order                     = read_sized_vector_exact<unsigned>(in, node_count,     "order");
    raw.rank                      = read_sized_vector_exact<unsigned>(in, node_count,     "rank");
    raw.elimination_tree_parent   = read_sized_vector_exact<unsigned>(in, node_count,     "elimination_tree_parent");
    raw.up_first_out              = read_sized_vector_exact<unsigned>(in, node_count + 1, "up_first_out");
    raw.up_head                   = read_sized_vector_exact<unsigned>(in, cch_arc_count,  "up_head");
    raw.up_tail                   = read_sized_vector_exact<unsigned>(in, cch_arc_count,  "up_tail");
    raw.down_first_out            = read_sized_vector_exact<unsigned>(in, node_count + 1, "down_first_out");
    raw.down_head                 = read_sized_vector_exact<unsigned>(in, cch_arc_count,  "down_head");
    raw.down_to_up                = read_sized_vector_exact<unsigned>(in, cch_arc_count,  "down_to_up");
    raw.input_arc_to_cch_arc      = read_sized_vector_exact<unsigned>(in, input_arc_count, "input_arc_to_cch_arc");

    // BitVectors first — they drive the mapper sizes for the variable-length
    // vectors below.
    raw.is_input_arc_upward                  = read_sized_bit_vector(in, "is_input_arc_upward");
    raw.does_cch_arc_have_input_arc          = read_sized_bit_vector(in, "does_cch_arc_have_input_arc");
    raw.does_cch_arc_have_extra_input_arc    = read_sized_bit_vector(in, "does_cch_arc_have_extra_input_arc");

    // Reconstruct LocalIDMappers from their parent BitVectors. The constructor
    // takes (bit_count, bits_ptr) and computes the popcount-rank table
    // internally — we don't need to persist that table.
    raw.does_cch_arc_have_input_arc_mapper = LocalIDMapper(
        raw.does_cch_arc_have_input_arc.size(),
        raw.does_cch_arc_have_input_arc.data());
    raw.does_cch_arc_have_extra_input_arc_mapper = LocalIDMapper(
        raw.does_cch_arc_have_extra_input_arc.size(),
        raw.does_cch_arc_have_extra_input_arc.data());

    // Variable-length sections: trust the wire byte_length.
    raw.forward_input_arc_of_cch              = read_sized_vector<unsigned>(in, "forward_input_arc_of_cch");
    raw.backward_input_arc_of_cch             = read_sized_vector<unsigned>(in, "backward_input_arc_of_cch");
    raw.first_extra_forward_input_arc_of_cch  = read_sized_vector<unsigned>(in, "first_extra_forward_input_arc_of_cch");
    raw.first_extra_backward_input_arc_of_cch = read_sized_vector<unsigned>(in, "first_extra_backward_input_arc_of_cch");
    raw.extra_forward_input_arc_of_cch        = read_sized_vector<unsigned>(in, "extra_forward_input_arc_of_cch");
    raw.extra_backward_input_arc_of_cch       = read_sized_vector<unsigned>(in, "extra_backward_input_arc_of_cch");

    return std::make_unique<CCH>(std::move(raw));
}

// ---------------- metric save / load ----------------

void cch_save_metric(const CCHMetric& metric_w, rust::Str path)
{
    const auto& m = metric_w.inner;
    std::ofstream out(rust_str_to_std(path), std::ios::binary);
    if (!out) {
        throw std::runtime_error("cch_save_metric: cannot open " + rust_str_to_std(path));
    }

    write_value(out, METRIC_MAGIC);
    write_value(out, FORMAT_VERSION);
    write_value(out, static_cast<uint32_t>(0));            // reserved
    write_value(out, static_cast<uint64_t>(m.forward.size())); // cch_arc_count

    write_sized_vector(out, m.forward);
    write_sized_vector(out, m.backward);

    out.flush();
    if (!out) {
        throw std::runtime_error("cch_save_metric: stream error after flush");
    }
}

std::unique_ptr<CCHMetric> cch_load_metric(const CCH& cch_w, rust::Str path)
{
    std::ifstream in(rust_str_to_std(path), std::ios::binary);
    if (!in) {
        throw std::runtime_error("cch_load_metric: cannot open " + rust_str_to_std(path));
    }

    uint64_t magic = read_value<uint64_t>(in);
    if (magic != METRIC_MAGIC) {
        throw std::runtime_error("cch_load_metric: bad magic in " + rust_str_to_std(path));
    }
    uint32_t version = read_value<uint32_t>(in);
    if (version != FORMAT_VERSION) {
        throw std::runtime_error("cch_load_metric: unsupported version "
            + std::to_string(version));
    }
    read_value<uint32_t>(in); // reserved
    uint64_t expected_arc_count = read_value<uint64_t>(in);

    if (expected_arc_count != cch_w.inner.cch_arc_count()) {
        throw std::runtime_error("cch_load_metric: arc count mismatch — bundle has "
            + std::to_string(expected_arc_count) + ", CCH has "
            + std::to_string(cch_w.inner.cch_arc_count()));
    }

    CustomizableContractionHierarchyMetric raw;
    raw.cch = &cch_w.inner;
    raw.input_weight = nullptr;  // not used by queries (only by re-customize)
    raw.forward  = read_sized_vector_exact<unsigned>(in, expected_arc_count, "forward");
    raw.backward = read_sized_vector_exact<unsigned>(in, expected_arc_count, "backward");

    return std::make_unique<CCHMetric>(std::move(raw));
}

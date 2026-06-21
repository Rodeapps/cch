//! Bundle WRITERS — produce `.cch-struct` / `.cch-metric` files that are
//! **byte-identical** to `RoutingKit`'s `cch_save_struct` / `cch_save_metric`
//! (`oracle/routingkit-cch/src/cch_bundle.cc`).
//!
//! [`Cch::save_struct`] writes the `CCH_STRC` v1 layout; [`Metric::save`] writes
//! the `CCH_METR` v1 layout. Together with the readers in [`crate::bundle`] this
//! makes the pipeline fully pure-Rust round-trippable while staying compatible
//! with the existing on-disk corpus and the C++ loader.
//!
//! ## The on-disk mapping layout (the crux)
//!
//! [`Cch::build`] stores the input-arc → CCH-arc mapping as FULL-SIZE
//! (`cch_arc_count`) arrays — convenient for [`Cch::customize`]. The on-disk
//! format instead uses the C++ LOCAL-id-compressed representation: three
//! [`BitVector`](crate::internal::BitVector)s plus arrays compressed via a
//! [`LocalIDMapper`](crate::internal::LocalIDMapper). [`reconstruct_on_disk_mapping`]
//! rebuilds that exact representation from the full-size arrays so the bytes
//! match the C++ constructor (`customizable_contraction_hierarchy.cpp` ~555-640)
//! and serializer (`cch_bundle.cc` ~130-160).

use std::io::{self, Write};

use crate::bundle::INVALID_ID;
use crate::customize::Metric;
use crate::internal::bitvec::BitVector;
use crate::internal::id_map::LocalIDMapper;
use crate::structure::Cch;

const STRUCT_MAGIC: u64 = 0x4343_485F_5354_5243; // "CCH_STRC"
const METRIC_MAGIC: u64 = 0x4343_485F_4D45_5452; // "CCH_METR"
const FORMAT_VERSION: u32 = 1;

/// Write the raw little-endian bytes of a `u32`.
fn write_u32<W: Write>(out: &mut W, value: u32) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

/// Write the raw little-endian bytes of a `u64`.
fn write_u64<W: Write>(out: &mut W, value: u64) -> io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

/// `write_sized_vector`: a `u64` byte length (`= v.len() * 4`) followed by the
/// raw little-endian `u32` bytes. Mirrors `cch_bundle.cc::write_sized_vector`.
fn write_sized_vector<W: Write>(out: &mut W, v: &[u32]) -> io::Result<()> {
    let byte_length = (v.len() as u64) * 4;
    write_u64(out, byte_length)?;
    // On a little-endian host the in-memory `u32` layout already matches the
    // wire format; build the byte buffer explicitly so the writer is
    // endianness-independent and matches the C++ raw `out.write` exactly.
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    out.write_all(&bytes)
}

/// `write_sized_bit_vector`: a `u64` bit count, then a `u64` byte length
/// (`= ((bit_count + 511) / 512) * 64`), then the 512-bit-block-padded
/// `u64`-word bytes. Mirrors `cch_bundle.cc::write_sized_bit_vector`.
fn write_sized_bit_vector<W: Write>(out: &mut W, bv: &BitVector) -> io::Result<()> {
    let bit_count = bv.len();
    let byte_length = bit_count.div_ceil(512) * 64;
    write_u64(out, bit_count)?;
    write_u64(out, byte_length)?;
    // The C++ dumps the BitVector's 64-bit-aligned storage, which is padded to
    // a multiple of 512 bits. `BitVector::words()` is only ceil(bit/64) words,
    // so pad with zero words up to `byte_length / 8` to match exactly.
    let word_count = (byte_length / 8) as usize;
    let mut bytes = Vec::with_capacity(word_count * 8);
    let words = bv.words();
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    for _ in words.len()..word_count {
        bytes.extend_from_slice(&0u64.to_le_bytes());
    }
    out.write_all(&bytes)
}

/// The C++ LOCAL-id-compressed on-disk representation of the input-arc mapping,
/// reconstructed from the full-size arrays held by [`Cch`].
struct OnDiskMapping {
    is_input_arc_upward: BitVector,
    does_cch_arc_have_input_arc: BitVector,
    does_cch_arc_have_extra_input_arc: BitVector,
    /// Length = `does_cch_arc_have_input_arc.local_id_count()`.
    forward_input_arc_of_cch: Vec<u32>,
    /// Length = `does_cch_arc_have_input_arc.local_id_count()`.
    backward_input_arc_of_cch: Vec<u32>,
    /// CSR offsets, length = `does_cch_arc_have_extra_input_arc.local_id_count() + 1`.
    first_extra_forward_input_arc_of_cch: Vec<u32>,
    first_extra_backward_input_arc_of_cch: Vec<u32>,
    /// Flat extra lists.
    extra_forward_input_arc_of_cch: Vec<u32>,
    extra_backward_input_arc_of_cch: Vec<u32>,
}

/// Rebuilds the C++ LOCAL-id-compressed mapping (3 bitvectors + local-compressed
/// vectors + extra CSR) from the full-size arrays in `cch`, byte-identically to
/// `customizable_contraction_hierarchy.cpp` ~555-640.
#[allow(clippy::cast_possible_truncation)] // ids/counts fit u32 (CCH limit)
fn reconstruct_on_disk_mapping(cch: &Cch) -> OnDiskMapping {
    let input_arc_count = cch.input_arc_to_cch_arc.len();
    let cch_arc_count = cch.cch_arc_count();

    // is_input_arc_upward[input_arc]: set iff the input arc runs up-rank. An
    // input arc is "upward" exactly when it backs a FORWARD weight, i.e. it
    // appears as a (first or extra) forward input arc. C++ sets this bit (line
    // 350) for every input arc whose relabeled tail < head.
    let mut is_input_arc_upward = BitVector::new(input_arc_count as u64);
    for &ia in &cch.forward_input_arc_of_cch {
        if ia != INVALID_ID {
            is_input_arc_upward.set(u64::from(ia));
        }
    }
    for &ia in &cch.extra_forward_input_arc_of_cch {
        is_input_arc_upward.set(u64::from(ia));
    }

    // does_cch_arc_have_input_arc[cch_arc]: set iff the CCH arc has any input
    // arc (forward or backward). C++ lines 560-562.
    let mut does_cch_arc_have_input_arc = BitVector::new(cch_arc_count as u64);
    for cch_arc in 0..cch_arc_count {
        if cch.forward_input_arc_of_cch[cch_arc] != INVALID_ID
            || cch.backward_input_arc_of_cch[cch_arc] != INVALID_ID
        {
            does_cch_arc_have_input_arc.set(cch_arc as u64);
        }
    }

    // does_cch_arc_have_extra_input_arc[cch_arc]: set iff the CCH arc has 2+
    // input arcs in either direction. C++ lines 583 / 592.
    let mut does_cch_arc_have_extra_input_arc = BitVector::new(cch_arc_count as u64);
    let ff = &cch.first_extra_forward_input_arc_of_cch;
    let fb = &cch.first_extra_backward_input_arc_of_cch;
    for cch_arc in 0..cch_arc_count {
        if ff[cch_arc + 1] > ff[cch_arc] || fb[cch_arc + 1] > fb[cch_arc] {
            does_cch_arc_have_extra_input_arc.set(cch_arc as u64);
        }
    }

    // LOCAL-compressed first/forward arrays, indexed by to_local(cch_arc) over
    // does_cch_arc_have_input_arc (C++ lines 564-581).
    let in_mapper = LocalIDMapper::new(
        does_cch_arc_have_input_arc.words(),
        does_cch_arc_have_input_arc.len(),
    );
    let local_in = in_mapper.local_id_count() as usize;
    let mut forward_local = vec![INVALID_ID; local_in];
    let mut backward_local = vec![INVALID_ID; local_in];
    for cch_arc in 0..cch_arc_count {
        if does_cch_arc_have_input_arc.is_set(cch_arc as u64) {
            let li = in_mapper.to_local(cch_arc as u64) as usize;
            forward_local[li] = cch.forward_input_arc_of_cch[cch_arc];
            backward_local[li] = cch.backward_input_arc_of_cch[cch_arc];
        }
    }

    // Extra CSR, indexed by to_local(cch_arc) over does_cch_arc_have_extra_input_arc
    // (C++ lines 601-637). The flat extra lists are already grouped by ascending
    // cch arc in `cch.extra_*`; only the CSR offsets are re-indexed to extra-local
    // ids. Build the per-extra-local count, then prefix-sum to CSR offsets.
    let extra_mapper = LocalIDMapper::new(
        does_cch_arc_have_extra_input_arc.words(),
        does_cch_arc_have_extra_input_arc.len(),
    );
    let local_extra = extra_mapper.local_id_count() as usize;

    let mut first_extra_forward = vec![0u32; local_extra + 1];
    let mut first_extra_backward = vec![0u32; local_extra + 1];
    for cch_arc in 0..cch_arc_count {
        if does_cch_arc_have_extra_input_arc.is_set(cch_arc as u64) {
            let li = extra_mapper.to_local(cch_arc as u64) as usize;
            first_extra_forward[li + 1] = ff[cch_arc + 1] - ff[cch_arc];
            first_extra_backward[li + 1] = fb[cch_arc + 1] - fb[cch_arc];
        }
    }
    for i in 0..local_extra {
        first_extra_forward[i + 1] += first_extra_forward[i];
        first_extra_backward[i + 1] += first_extra_backward[i];
    }

    OnDiskMapping {
        is_input_arc_upward,
        does_cch_arc_have_input_arc,
        does_cch_arc_have_extra_input_arc,
        forward_input_arc_of_cch: forward_local,
        backward_input_arc_of_cch: backward_local,
        first_extra_forward_input_arc_of_cch: first_extra_forward,
        first_extra_backward_input_arc_of_cch: first_extra_backward,
        extra_forward_input_arc_of_cch: cch.extra_forward_input_arc_of_cch.clone(),
        extra_backward_input_arc_of_cch: cch.extra_backward_input_arc_of_cch.clone(),
    }
}

impl Cch {
    /// Writes this CCH structure to `path` in the `CCH_STRC` v1 format,
    /// byte-identical to `RoutingKit`'s `cch_save_struct`.
    ///
    /// # Errors
    /// Returns [`io::Error`] on any I/O failure (cannot create the file, write
    /// error, or flush error).
    pub fn save_struct(&self, path: &std::path::Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut out = std::io::BufWriter::new(file);
        self.write_struct(&mut out)?;
        out.flush()
    }

    /// Serializes the struct to an arbitrary writer (used by [`Self::save_struct`]
    /// and by in-memory round-trip tests).
    fn write_struct<W: Write>(&self, out: &mut W) -> io::Result<()> {
        let node_count = self.node_count() as u64;
        let cch_arc_count = self.cch_arc_count() as u64;
        let input_arc_count = self.input_arc_to_cch_arc.len() as u64;

        write_u64(out, STRUCT_MAGIC)?;
        write_u32(out, FORMAT_VERSION)?;
        write_u32(out, 0)?; // reserved
        write_u64(out, node_count)?;
        write_u64(out, cch_arc_count)?;
        write_u64(out, input_arc_count)?;

        // Fixed-size sections (C++ cch_bundle.cc lines 129-138).
        write_sized_vector(out, &self.order)?;
        write_sized_vector(out, &self.rank)?;
        write_sized_vector(out, &self.elimination_tree_parent)?;
        write_sized_vector(out, &self.up_first_out)?;
        write_sized_vector(out, &self.up_head)?;
        write_sized_vector(out, &self.up_tail)?;
        write_sized_vector(out, &self.down_first_out)?;
        write_sized_vector(out, &self.down_head)?;
        write_sized_vector(out, &self.down_to_up)?;
        write_sized_vector(out, &self.input_arc_to_cch_arc)?;

        // Reconstruct the C++ LOCAL-compressed mapping layout.
        let m = reconstruct_on_disk_mapping(self);

        write_sized_bit_vector(out, &m.is_input_arc_upward)?;
        write_sized_bit_vector(out, &m.does_cch_arc_have_input_arc)?;
        write_sized_bit_vector(out, &m.does_cch_arc_have_extra_input_arc)?;

        write_sized_vector(out, &m.forward_input_arc_of_cch)?;
        write_sized_vector(out, &m.backward_input_arc_of_cch)?;
        write_sized_vector(out, &m.first_extra_forward_input_arc_of_cch)?;
        write_sized_vector(out, &m.first_extra_backward_input_arc_of_cch)?;
        write_sized_vector(out, &m.extra_forward_input_arc_of_cch)?;
        write_sized_vector(out, &m.extra_backward_input_arc_of_cch)?;

        Ok(())
    }
}

impl Metric {
    /// Writes this customized metric to `path` in the `CCH_METR` v1 format,
    /// byte-identical to `RoutingKit`'s `cch_save_metric`.
    ///
    /// # Errors
    /// Returns [`io::Error`] on any I/O failure.
    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut out = std::io::BufWriter::new(file);
        self.write_metric(&mut out)?;
        out.flush()
    }

    /// Serializes the metric to an arbitrary writer.
    fn write_metric<W: Write>(&self, out: &mut W) -> io::Result<()> {
        let cch_arc_count = self.forward.len() as u64;
        write_u64(out, METRIC_MAGIC)?;
        write_u32(out, FORMAT_VERSION)?;
        write_u32(out, 0)?; // reserved
        write_u64(out, cch_arc_count)?;
        write_sized_vector(out, &self.forward)?;
        write_sized_vector(out, &self.backward)?;
        Ok(())
    }
}

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

/// A cursor over an in-memory `.cch-struct` byte buffer, with bounds-checked
/// little-endian reads. Every read advances `pos` and returns `InvalidData`
/// (never panics / reads out of bounds) when the buffer is too short — matching
/// the defensive posture of [`crate::bundle::CchBundle::open`].
struct StructReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> StructReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Reads a little-endian `u64`.
    fn read_u64(&mut self) -> io::Result<u64> {
        // `self.pos <= self.bytes.len()` always (every advance is bounds-checked
        // before it happens), so `pos + 8` is at most `len + 8` and cannot
        // overflow usize on any supported platform; a saturating add keeps the
        // bounds check sound without an unreachable overflow branch.
        let end = self.pos.saturating_add(8);
        if end > self.bytes.len() {
            return Err(truncated_err("u64 field"));
        }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(u64::from_le_bytes(buf))
    }

    /// Reads a little-endian `u64` and converts it to `usize`.
    ///
    /// The crate targets 64-bit platforms (`usize == u64`), so the conversion is
    /// total. A subsequent length/section check rejects any absurd value as
    /// corrupt before it is used to index or allocate.
    #[allow(clippy::cast_possible_truncation)] // 64-bit target: usize == u64
    fn read_u64_usize(&mut self) -> io::Result<usize> {
        Ok(self.read_u64()? as usize)
    }

    /// Reads a little-endian `u32`.
    fn read_u32(&mut self) -> io::Result<u32> {
        let end = self.pos.saturating_add(4);
        if end > self.bytes.len() {
            return Err(truncated_err("u32 field"));
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(u32::from_le_bytes(buf))
    }

    /// Reads a `write_sized_vector`-framed `Vec<u32>`: a `u64` byte length
    /// followed by that many bytes (must be a multiple of 4), validating the
    /// element count against the header-derived `expected_count` (mirrors the
    /// C++ `read_sized_vector_exact`).
    fn read_sized_vector(&mut self, label: &str, expected_count: usize) -> io::Result<Vec<u32>> {
        self.read_sized_vector_inner(label, Some(expected_count))
    }

    /// Like [`Self::read_sized_vector`] but trusts the wire byte length without
    /// an expected-count check (used for the data-dependent flat extra lists,
    /// matching the C++ `read_sized_vector`).
    fn read_sized_vector_any(&mut self, label: &str) -> io::Result<Vec<u32>> {
        self.read_sized_vector_inner(label, None)
    }

    fn read_sized_vector_inner(
        &mut self,
        label: &str,
        expected_count: Option<usize>,
    ) -> io::Result<Vec<u32>> {
        let byte_length = self.read_u64_usize()?;
        if byte_length % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("section '{label}' byte_length {byte_length} is not a multiple of 4"),
            ));
        }
        let count = byte_length / 4;
        if let Some(expected) = expected_count {
            // `expected` is a header-derived count that already fits usize; on a
            // 32-bit target a 4x multiply could overflow, which we treat as a
            // (corrupt) length disagreement via the != comparison below.
            if expected.checked_mul(4) != Some(byte_length) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "section '{label}' byte_length {byte_length} does not match expected \
                         4 * {expected}; header count and section length disagree"
                    ),
                ));
            }
        }
        let end = self.pos.saturating_add(byte_length);
        if end > self.bytes.len() {
            return Err(truncated_err(label));
        }
        let mut v = Vec::with_capacity(count);
        let mut off = self.pos;
        for _ in 0..count {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&self.bytes[off..off + 4]);
            v.push(u32::from_le_bytes(buf));
            off += 4;
        }
        self.pos = end;
        Ok(v)
    }

    /// Reads a `write_sized_bit_vector`-framed [`BitVector`]: a `u64` bit count,
    /// a `u64` byte length (must equal the 512-block padding of the bit count),
    /// then the word bytes. Validates the bit count against the header-derived
    /// `expected_bits`.
    #[allow(clippy::cast_possible_truncation)] // bit_count == expected_bits fits usize
    fn read_sized_bit_vector(&mut self, label: &str, expected_bits: u64) -> io::Result<BitVector> {
        let bit_count = self.read_u64()?;
        if bit_count != expected_bits {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "section '{label}' bit_count {bit_count} does not match expected \
                     {expected_bits}"
                ),
            ));
        }
        let byte_length = self.read_u64_usize()?;
        // 512-bit-block padding of `bit_count`. `bit_count == expected_bits`,
        // a header count that fits usize, so this conversion cannot fail here.
        let expected_byte_length = (byte_length_for_bits(bit_count)).unwrap_or(usize::MAX);
        if byte_length != expected_byte_length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "section '{label}' bitvector byte_length {byte_length} does not match \
                     expected {expected_byte_length} for bit_count {bit_count}"
                ),
            ));
        }
        let end = self.pos.saturating_add(byte_length);
        if end > self.bytes.len() {
            return Err(truncated_err(label));
        }
        // The on-disk blob is 512-bit-block padded; `BitVector` keeps only
        // ceil(bit_count / 64) words. Read exactly that many words from the
        // front of the blob (the writer wrote the real words first, then zero
        // padding) and reconstruct the `BitVector` from them.
        // `bit_count` fits usize (validated == expected_bits), so ceil/64 does too.
        let word_count = (bit_count.div_ceil(64)) as usize;
        let mut bv = BitVector::new(bit_count);
        let mut off = self.pos;
        for word_idx in 0..word_count {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&self.bytes[off..off + 8]);
            let word = u64::from_le_bytes(buf);
            // Set each set bit that is in range; padding bits in the final word
            // are ignored (BitVector::new zeroed them and keeps them zero).
            let base = (word_idx as u64) * 64;
            let mut w = word;
            while w != 0 {
                let b = u64::from(w.trailing_zeros());
                let global = base + b;
                if global < bit_count {
                    bv.set(global);
                }
                w &= w - 1;
            }
            off += 8;
        }
        self.pos = end;
        Ok(bv)
    }
}

/// `InvalidData` error for a truncated section.
fn truncated_err(label: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("truncated .cch-struct at section '{label}'"),
    )
}

/// 512-bit-block-padded byte length of a `bit_count`-bit `BitVector`
/// (`= ceil(bit_count / 512) * 64`), or `None` if it does not fit `usize`.
fn byte_length_for_bits(bit_count: u64) -> Option<usize> {
    usize::try_from(bit_count.div_ceil(512))
        .ok()
        .and_then(|blocks| blocks.checked_mul(64))
}

/// Rebuilds a FULL-SIZE per-CCH-arc array (length `cch_arc_count`) from a
/// LOCAL-id-compressed array indexed by `to_local` over `presence`. Entries for
/// CCH arcs whose presence bit is clear are filled with [`INVALID_ID`].
///
/// This inverts the LOCAL compression `reconstruct_on_disk_mapping` performs
/// for `forward_input_arc_of_cch` / `backward_input_arc_of_cch`.
#[allow(clippy::cast_possible_truncation)] // cch_arc / local id < usize::MAX (CCH limit)
fn expand_local_to_full(
    presence: &BitVector,
    mapper: &LocalIDMapper,
    local: &[u32],
    cch_arc_count: usize,
) -> Vec<u32> {
    let mut full = vec![INVALID_ID; cch_arc_count];
    for (cch_arc, slot) in full.iter_mut().enumerate() {
        if presence.is_set(cch_arc as u64) {
            // `to_local(cch_arc) < local.len()` is guaranteed: `local` was read
            // with an exact-count check against `mapper.local_id_count()`, and
            // `mapper` is built from `presence`, so the rank is in range.
            let li = mapper.to_local(cch_arc as u64) as usize;
            *slot = local[li];
        }
    }
    full
}

/// Rebuilds the FULL-SIZE extra-arc CSR offsets (length `cch_arc_count + 1`)
/// from the LOCAL-id-compressed CSR (indexed by `to_local` over `presence`).
///
/// Inverts the extra-CSR re-indexing in `reconstruct_on_disk_mapping`: the flat
/// extra lists are unchanged, only the per-arc offsets are expanded from the
/// extra-local id space back to the full per-CCH-arc id space.
#[allow(clippy::cast_possible_truncation)] // cch_arc / local id < usize::MAX (CCH limit)
fn expand_extra_csr(
    presence: &BitVector,
    mapper: &LocalIDMapper,
    first_local: &[u32],
    cch_arc_count: usize,
) -> io::Result<Vec<u32>> {
    let mut full = vec![0u32; cch_arc_count + 1];
    for cch_arc in 0..cch_arc_count {
        let count = if presence.is_set(cch_arc as u64) {
            // `le < local_extra` and `first_local.len() == local_extra + 1`, so
            // both indices are in bounds (validated against the mapper's
            // local_id_count when `first_local` was read).
            let le = mapper.to_local(cch_arc as u64) as usize;
            let (lo, hi) = (first_local[le], first_local[le + 1]);
            // The on-disk CSR offsets are not otherwise validated to be
            // monotonic; a corrupt file could make `hi < lo`.
            hi.checked_sub(lo).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "extra CSR offsets are not monotonic",
                )
            })?
        } else {
            0
        };
        // The per-arc counts are disjoint adjacent deltas of the single
        // monotonic `first_local` array, so the running sum telescopes to a
        // prefix of `first_local` and is bounded by its (u32) maximum — it
        // cannot overflow u32. A plain add is therefore total.
        full[cch_arc + 1] = full[cch_arc] + count;
    }
    Ok(full)
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

    /// Loads a CCH structure from a `.cch-struct` file at `path`, reconstructing
    /// a fully RE-CUSTOMIZABLE [`Cch`] (symmetric to [`Self::save_struct`]).
    ///
    /// Unlike [`crate::bundle::CchBundle::open`] (which mmaps only the query
    /// prefix), this reads ALL sections — including the input-arc → CCH-arc
    /// mapping — and inverts the LOCAL-id-compressed on-disk representation back
    /// into the FULL-SIZE arrays that [`Self::customize`] consumes. The returned
    /// `Cch`'s internal mapping representation may differ from a freshly-built
    /// one, but `customize` produces bit-identical output.
    ///
    /// Reads the C++ (`RoutingKit` `cch_save_struct`) format and the pure-Rust
    /// writer's output interchangeably.
    ///
    /// # Errors
    /// Returns [`io::Error`] on I/O failure, and [`io::ErrorKind::InvalidData`]
    /// on a corrupt/truncated file (bad magic, unsupported version, mismatched
    /// section lengths, …). Never panics or reads out of bounds on bad input.
    pub fn load_struct(path: &std::path::Path) -> io::Result<Cch> {
        let bytes = std::fs::read(path)?;
        Self::read_struct(&bytes)
    }

    /// Parses an in-memory `.cch-struct` byte buffer into a re-customizable
    /// [`Cch`]. Factored out of [`Self::load_struct`] so corrupt-input tests can
    /// craft buffers directly without touching the filesystem.
    #[allow(clippy::cast_possible_truncation)] // counts fit u32/usize (CCH limit), validated above
    #[allow(clippy::too_many_lines)] // linear header + per-section read clearest inline
    fn read_struct(bytes: &[u8]) -> io::Result<Cch> {
        let mut r = StructReader::new(bytes);

        let magic = r.read_u64()?;
        if magic != STRUCT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad magic in .cch-struct: {magic:#x}, expected {STRUCT_MAGIC:#x}"),
            ));
        }
        let version = r.read_u32()?;
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .cch-struct version {version}"),
            ));
        }
        let _reserved = r.read_u32()?;
        let node_count = r.read_u64_usize()?;
        let cch_arc_count = r.read_u64_usize()?;
        let input_arc_count = r.read_u64_usize()?;

        // `node_count + 1` cannot overflow: `node_count` came from a u64 that
        // fits usize, and a valid struct's node_count is far below usize::MAX.
        let node_count_plus_1 = node_count.saturating_add(1);

        // Fixed-size sections (header-derived counts, exact-checked).
        let order = r.read_sized_vector("order", node_count)?;
        let rank = r.read_sized_vector("rank", node_count)?;
        let elimination_tree_parent = r.read_sized_vector("elimination_tree_parent", node_count)?;
        let up_first_out = r.read_sized_vector("up_first_out", node_count_plus_1)?;
        let up_head = r.read_sized_vector("up_head", cch_arc_count)?;
        let up_tail = r.read_sized_vector("up_tail", cch_arc_count)?;
        let down_first_out = r.read_sized_vector("down_first_out", node_count_plus_1)?;
        let down_head = r.read_sized_vector("down_head", cch_arc_count)?;
        let down_to_up = r.read_sized_vector("down_to_up", cch_arc_count)?;
        let input_arc_to_cch_arc = r.read_sized_vector("input_arc_to_cch_arc", input_arc_count)?;

        // BitVectors — they drive the LOCAL-id mapper sizes below.
        let _is_input_arc_upward =
            r.read_sized_bit_vector("is_input_arc_upward", input_arc_count as u64)?;
        let does_cch_arc_have_input_arc =
            r.read_sized_bit_vector("does_cch_arc_have_input_arc", cch_arc_count as u64)?;
        let does_cch_arc_have_extra_input_arc =
            r.read_sized_bit_vector("does_cch_arc_have_extra_input_arc", cch_arc_count as u64)?;

        let in_mapper = LocalIDMapper::new(
            does_cch_arc_have_input_arc.words(),
            does_cch_arc_have_input_arc.len(),
        );
        let extra_mapper = LocalIDMapper::new(
            does_cch_arc_have_extra_input_arc.words(),
            does_cch_arc_have_extra_input_arc.len(),
        );

        // LOCAL-id-compressed mapping arrays (sized by mapper local_id_count).
        // The local counts are popcounts of cch_arc_count-bit vectors, so they
        // are <= cch_arc_count and fit usize; `+ 1` cannot overflow.
        let local_in = in_mapper.local_id_count() as usize;
        let local_extra = extra_mapper.local_id_count() as usize;
        let local_extra_plus_1 = local_extra.saturating_add(1);

        let forward_local = r.read_sized_vector("forward_input_arc_of_cch", local_in)?;
        let backward_local = r.read_sized_vector("backward_input_arc_of_cch", local_in)?;
        let first_extra_forward_local =
            r.read_sized_vector("first_extra_forward_input_arc_of_cch", local_extra_plus_1)?;
        let first_extra_backward_local =
            r.read_sized_vector("first_extra_backward_input_arc_of_cch", local_extra_plus_1)?;
        // The flat extra lists have data-dependent lengths; trust the wire
        // byte_length (matching the C++ `read_sized_vector`).
        let extra_forward_input_arc_of_cch =
            r.read_sized_vector_any("extra_forward_input_arc_of_cch")?;
        let extra_backward_input_arc_of_cch =
            r.read_sized_vector_any("extra_backward_input_arc_of_cch")?;

        // Invert the LOCAL compression back into the FULL-SIZE arrays.
        let forward_input_arc_of_cch = expand_local_to_full(
            &does_cch_arc_have_input_arc,
            &in_mapper,
            &forward_local,
            cch_arc_count,
        );
        let backward_input_arc_of_cch = expand_local_to_full(
            &does_cch_arc_have_input_arc,
            &in_mapper,
            &backward_local,
            cch_arc_count,
        );
        let first_extra_forward_input_arc_of_cch = expand_extra_csr(
            &does_cch_arc_have_extra_input_arc,
            &extra_mapper,
            &first_extra_forward_local,
            cch_arc_count,
        )?;
        let first_extra_backward_input_arc_of_cch = expand_extra_csr(
            &does_cch_arc_have_extra_input_arc,
            &extra_mapper,
            &first_extra_backward_local,
            cch_arc_count,
        )?;

        // Final cross-check: the expanded CSR totals must match the flat extra
        // list lengths (guards a corrupt CSR/extra-list pair that the per-arc
        // checks alone would miss).
        if first_extra_forward_input_arc_of_cch[cch_arc_count] as usize
            != extra_forward_input_arc_of_cch.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "forward extra CSR total disagrees with extra list length",
            ));
        }
        if first_extra_backward_input_arc_of_cch[cch_arc_count] as usize
            != extra_backward_input_arc_of_cch.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "backward extra CSR total disagrees with extra list length",
            ));
        }

        Ok(Cch {
            rank,
            order,
            elimination_tree_parent,
            up_first_out,
            up_head,
            up_tail,
            down_first_out,
            down_head,
            down_to_up,
            input_arc_to_cch_arc,
            forward_input_arc_of_cch,
            backward_input_arc_of_cch,
            first_extra_forward_input_arc_of_cch,
            extra_forward_input_arc_of_cch,
            first_extra_backward_input_arc_of_cch,
            extra_backward_input_arc_of_cch,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::INF_WEIGHT;
    use crate::graph::Graph;

    /// Build a CSR `Graph` from a directed arc multiset grouped by tail.
    fn csr(node_count: u32, tail: &[u32], head: &[u32]) -> Graph {
        let n = node_count as usize;
        let mut degree = vec![0u32; n];
        for &t in tail {
            degree[t as usize] += 1;
        }
        let mut first_out = vec![0u32; n + 1];
        for v in 0..n {
            first_out[v + 1] = first_out[v] + degree[v];
        }
        let mut next: Vec<u32> = first_out[..n].to_vec();
        let mut g_head = vec![0u32; head.len()];
        for (&t, &h) in tail.iter().zip(head.iter()) {
            let slot = next[t as usize] as usize;
            g_head[slot] = h;
            next[t as usize] += 1;
        }
        Graph {
            first_out,
            head: g_head,
            weight: vec![1u32; head.len()],
        }
    }

    /// Serialize a `Cch` to a `.cch-struct` byte buffer via the production
    /// [`Cch::save_struct`] path (a tempfile), so the corrupt-input tests share
    /// the exact bytes the writer emits without introducing a second generic
    /// instantiation of the writer over an in-memory buffer.
    fn to_bytes(c: &Cch) -> Vec<u8> {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tmp.cch-struct");
        c.save_struct(&path).expect("save_struct");
        std::fs::read(&path).expect("read back struct bytes")
    }

    /// Assert that a round-trip through bytes preserves every structural array
    /// and that `customize` is bit-identical for several weight vectors.
    fn assert_round_trip(name: &str, c: &Cch, weight_sets: &[Vec<u32>]) {
        let bytes = to_bytes(c);
        let loaded = Cch::read_struct(&bytes).expect("read_struct");

        // GATE 1: structural arrays + input_arc_to_cch_arc equal.
        assert_eq!(loaded.rank, c.rank, "[{name}] rank");
        assert_eq!(loaded.order, c.order, "[{name}] order");
        assert_eq!(
            loaded.elimination_tree_parent, c.elimination_tree_parent,
            "[{name}] elim"
        );
        assert_eq!(loaded.up_first_out, c.up_first_out, "[{name}] up_first_out");
        assert_eq!(loaded.up_head, c.up_head, "[{name}] up_head");
        assert_eq!(loaded.up_tail, c.up_tail, "[{name}] up_tail");
        assert_eq!(
            loaded.down_first_out, c.down_first_out,
            "[{name}] down_first_out"
        );
        assert_eq!(loaded.down_head, c.down_head, "[{name}] down_head");
        assert_eq!(loaded.down_to_up, c.down_to_up, "[{name}] down_to_up");
        assert_eq!(
            loaded.input_arc_to_cch_arc, c.input_arc_to_cch_arc,
            "[{name}] input_arc_to_cch_arc"
        );

        // GATE 2: re-customize bit-identical for every weight vector.
        for (i, w) in weight_sets.iter().enumerate() {
            let want = c.customize(w);
            let got = loaded.customize(w);
            assert_eq!(want.forward, got.forward, "[{name}] forward weights #{i}");
            assert_eq!(
                want.backward, got.backward,
                "[{name}] backward weights #{i}"
            );
        }
    }

    #[test]
    fn round_trip_path_identity() {
        let n = 5u32;
        let tail = vec![0u32, 1, 1, 2, 2, 3, 3, 4];
        let head = vec![1u32, 0, 2, 1, 3, 2, 4, 3];
        let c = Cch::build(&csr(n, &tail, &head), &(0..n).collect::<Vec<_>>());
        let ws = vec![
            vec![10u32, 11, 20, 21, 30, 31, 40, 41],
            vec![1u32, 1, 1, 1, 1, 1, 1, 1],
            vec![INF_WEIGHT, 5, INF_WEIGHT, 7, 9, INF_WEIGHT, 2, 3],
        ];
        assert_round_trip("path_identity", &c, &ws);
    }

    #[test]
    fn round_trip_fillin_nonidentity_order() {
        let n = 4u32;
        let tail = vec![0u32, 0, 0, 1, 2, 3];
        let head = vec![1u32, 2, 3, 0, 0, 0];
        let order = vec![0u32, 1, 2, 3];
        let c = Cch::build(&csr(n, &tail, &head), &order);
        let ws = vec![vec![5u32, 7, 9, 6, 8, 10], vec![1u32, 2, 3, 4, 5, 6]];
        assert_round_trip("fillin", &c, &ws);
    }

    #[test]
    fn round_trip_parallel_arcs() {
        // Non-empty extra lists (parallel arcs) + fill-in + non-identity order.
        let n = 4u32;
        let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
        let order = vec![2u32, 0, 3, 1];
        let c = Cch::build(&csr(n, &tail, &head), &order);
        // Sanity: this fixture really exercises the extra (parallel) lists.
        let extra_total =
            c.extra_forward_input_arc_of_cch.len() + c.extra_backward_input_arc_of_cch.len();
        assert!(
            extra_total > 0,
            "fixture must have parallel arcs in the extra lists"
        );
        let ws = vec![
            vec![50u32, 9, 40, 8, 17, 18, 19, 20],
            vec![9u32, 50, 8, 40, 1, 2, 3, 4],
        ];
        assert_round_trip("parallel_arcs", &c, &ws);
    }

    #[test]
    fn round_trip_grid_block_boundary() {
        // 24x24 grid → cch_arc_count > 512, exercising the bitvector 512-bit
        // block padding in the round-trip.
        let cols = 24u32;
        let rows = 24u32;
        let n = cols * rows;
        let mut tail = Vec::new();
        let mut head = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if c + 1 < cols {
                    tail.push(v);
                    head.push(v + 1);
                }
                if c > 0 {
                    tail.push(v);
                    head.push(v - 1);
                }
                if r + 1 < rows {
                    tail.push(v);
                    head.push(v + cols);
                }
                if r > 0 {
                    tail.push(v);
                    head.push(v - cols);
                }
            }
        }
        let graph = csr(n, &tail, &head);
        let order = crate::degree_order(&graph);
        let c = Cch::build(&graph, &order);
        assert!(c.cch_arc_count() > 512, "grid must exceed 512 CCH arcs");
        #[allow(clippy::cast_possible_truncation)]
        let w: Vec<u32> = (0..tail.len() as u32)
            .map(|i| (i * 7 + 1) % 9973 + 1)
            .collect();
        assert_round_trip("grid_24x24", &c, &[w]);
    }

    #[test]
    fn round_trip_empty_and_single_node() {
        // Empty graph (no arcs), several isolated nodes.
        let n = 4u32;
        let c = Cch::build(&csr(n, &[], &[]), &(0..n).collect::<Vec<_>>());
        assert_round_trip("empty_arcs", &c, &[vec![]]);

        // Single isolated node.
        let c = Cch::build(&csr(1, &[], &[]), &[0]);
        assert_round_trip("single_node", &c, &[vec![]]);
    }

    // ---- Corrupt-input robustness: every error path returns InvalidData,
    //      never panics or reads out of bounds. ----

    /// A valid baseline `.cch-struct` byte buffer built from a fill-in graph
    /// (so all sections — bitvectors, local arrays, extra CSR — are non-empty).
    fn valid_struct_bytes() -> Vec<u8> {
        let n = 4u32;
        let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
        let c = Cch::build(&csr(n, &tail, &head), &[0, 1, 2, 3]);
        to_bytes(&c)
    }

    fn assert_invalid(bytes: &[u8], must_contain: &str) {
        let err = Cch::read_struct(bytes).map(|_| ()).unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidData,
            "expected InvalidData, got: {err}"
        );
        assert!(
            err.to_string().contains(must_contain),
            "error '{err}' must contain '{must_contain}'"
        );
    }

    #[test]
    fn corrupt_baseline_is_valid() {
        // The baseline used by the corrupt tests must itself load cleanly.
        Cch::read_struct(&valid_struct_bytes()).expect("baseline must be valid");
    }

    #[test]
    fn corrupt_empty_buffer_truncated() {
        assert_invalid(&[], "truncated");
    }

    #[test]
    fn corrupt_bad_magic() {
        let mut b = valid_struct_bytes();
        b[0..8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
        assert_invalid(&b, "bad magic");
    }

    #[test]
    fn corrupt_bad_version() {
        let mut b = valid_struct_bytes();
        b[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert_invalid(&b, "unsupported .cch-struct version");
    }

    #[test]
    fn corrupt_truncated_after_header() {
        // Truncate to exactly the 40-byte header: the first section (order)
        // length prefix cannot be read.
        let b = valid_struct_bytes();
        assert_invalid(&b[..40], "truncated");
    }

    #[test]
    fn corrupt_truncated_mid_section() {
        // Truncate so the 'order' section's declared bytes run past the buffer.
        let b = valid_struct_bytes();
        // header (40) + order length prefix (8) + 2 bytes of order data.
        assert_invalid(&b[..50], "order");
    }

    #[test]
    fn corrupt_inflated_node_count() {
        // Inflate node_count so 'order' byte_length disagrees with 4 * count.
        let mut b = valid_struct_bytes();
        let nc = {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&b[16..24]);
            u64::from_le_bytes(buf)
        };
        b[16..24].copy_from_slice(&(nc + 1_000_000).to_le_bytes());
        assert_invalid(&b, "does not match expected");
    }

    #[test]
    fn corrupt_node_count_absurd() {
        // node_count = u64::MAX: the order section's real (small) byte_length
        // cannot equal 4 * usize::MAX (the multiply saturates to None), so the
        // section-length reconciliation rejects it without any allocation/OOB.
        let mut b = valid_struct_bytes();
        b[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_invalid(&b, "does not match expected");
    }

    #[test]
    fn corrupt_order_byte_length_not_multiple_of_4() {
        // Set the 'order' section byte_length to 1 (not a multiple of 4) and
        // truncate the header counts to keep the prefix readable.
        let mut b = valid_struct_bytes();
        // node_count = 0 so the exact-count check would expect 0 bytes, but we
        // want the "not a multiple of 4" branch: feed a non-multiple length and
        // a matching expectation by inflating order alone is awkward; instead
        // craft a minimal buffer with node_count=0 and an order len of 3.
        b[16..24].copy_from_slice(&0u64.to_le_bytes()); // node_count = 0
        b[24..32].copy_from_slice(&0u64.to_le_bytes()); // cch_arc_count = 0
        b[32..40].copy_from_slice(&0u64.to_le_bytes()); // input_arc_count = 0
        // Rebuild a fresh buffer: header (40) + order length=3 + 3 bytes.
        let mut crafted = b[..40].to_vec();
        crafted.extend_from_slice(&3u64.to_le_bytes());
        crafted.extend_from_slice(&[0u8, 0, 0]);
        assert_invalid(&crafted, "not a multiple of 4");
    }

    #[test]
    fn corrupt_bitvector_byte_length_mismatch() {
        // Corrupt the byte_length of the first bitvector (is_input_arc_upward)
        // so it disagrees with the 512-block padding of its bit_count.
        //
        // Walk the sections to find the first bitvector's byte_length offset.
        let mut b = valid_struct_bytes();
        let off = first_bitvector_byte_length_offset(&b);
        // The bitvector framing is [u64 bit_count][u64 byte_length]; corrupt the
        // byte_length (second u64).
        b[off + 8..off + 16].copy_from_slice(&999u64.to_le_bytes());
        assert_invalid(&b, "bitvector byte_length");
    }

    /// Reads a little-endian `u64` at `o` from `b` as a `usize` (test helper).
    fn read_u64_usize(b: &[u8], o: usize) -> usize {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&b[o..o + 8]);
        usize::try_from(u64::from_le_bytes(buf)).expect("offset fits usize in test")
    }

    /// Walk the 10 fixed-size sized vectors after the 40-byte header and return
    /// the byte offset of the first bitvector's `bit_count` u64.
    fn first_bitvector_byte_length_offset(b: &[u8]) -> usize {
        let mut pos = 40usize;
        for _ in 0..10 {
            pos += 8 + read_u64_usize(b, pos);
        }
        pos
    }

    #[test]
    fn corrupt_extra_csr_total_mismatch() {
        // Corrupt the LAST entry of first_extra_forward so the expanded CSR
        // total disagrees with the flat extra-forward list length. We truncate
        // the extra-forward list to trigger the cross-check.
        //
        // Simplest reliable trigger: append a fabricated buffer where the flat
        // extra-forward list is shortened. We rebuild from a parallel-arc
        // fixture and chop bytes off the extra_forward section.
        let n = 4u32;
        let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
        let c = Cch::build(&csr(n, &tail, &head), &[0, 1, 2, 3]);
        assert!(
            !c.extra_forward_input_arc_of_cch.is_empty(),
            "need a non-empty forward extra list"
        );
        let mut b = to_bytes(&c);
        // Find the extra_forward section: it is the 5th sized vector after the
        // 3 bitvectors. Walk past header + 10 vectors + 3 bitvectors + 4 vectors.
        let off = extra_forward_byte_length_offset(&b);
        // Shrink its byte_length by 4 (drop one element) and drop the trailing
        // 4 data bytes so the buffer stays self-consistent up to that point.
        let cur = read_u64_usize(&b, off);
        assert!(cur >= 4);
        let shrunk = u64::try_from(cur - 4).expect("fits");
        b[off..off + 8].copy_from_slice(&shrunk.to_le_bytes());
        // Remove 4 bytes of data so subsequent parsing succeeds but the CSR
        // total no longer matches.
        b.drain(off + 8 + (cur - 4)..off + 8 + cur);
        assert_invalid(&b, "extra CSR total disagrees");
    }

    /// Offset of the `extra_forward_input_arc_of_cch` section's `byte_length` u64.
    fn extra_forward_byte_length_offset(b: &[u8]) -> usize {
        let mut pos = 40usize;
        // 10 fixed-size sized vectors.
        for _ in 0..10 {
            pos += 8 + read_u64_usize(b, pos);
        }
        // 3 bitvectors: framing is [u64 bit_count][u64 byte_length][bytes].
        for _ in 0..3 {
            pos += 16 + read_u64_usize(b, pos + 8);
        }
        // 4 sized vectors (forward, backward, first_extra_forward, first_extra_backward).
        for _ in 0..4 {
            pos += 8 + read_u64_usize(b, pos);
        }
        // Now at extra_forward_input_arc_of_cch.
        pos
    }

    /// Returns the start byte offset (the length-prefix position) of each of the
    /// 19 sections in a valid `.cch-struct`, in on-disk order.
    fn section_start_offsets(b: &[u8]) -> Vec<usize> {
        let mut starts = Vec::with_capacity(19);
        let mut pos = 40usize;
        // 10 fixed-size sized vectors.
        for _ in 0..10 {
            starts.push(pos);
            pos += 8 + read_u64_usize(b, pos);
        }
        // 3 bitvectors: [u64 bit_count][u64 byte_length][bytes].
        for _ in 0..3 {
            starts.push(pos);
            pos += 16 + read_u64_usize(b, pos + 8);
        }
        // 6 trailing sized vectors.
        for _ in 0..6 {
            starts.push(pos);
            pos += 8 + read_u64_usize(b, pos);
        }
        starts
    }

    #[test]
    fn corrupt_truncate_at_each_section_start() {
        // Truncating the buffer to the start of each section makes that
        // section's length-prefix read fail with a truncation error, exercising
        // the `?` error path of every `read_sized_*` call in `read_struct`.
        let b = valid_struct_bytes();
        for (i, &start) in section_start_offsets(&b).iter().enumerate() {
            let truncated = &b[..start];
            let err = Cch::read_struct(truncated).map(|_| ()).unwrap_err();
            assert_eq!(
                err.kind(),
                io::ErrorKind::InvalidData,
                "section #{i} truncation must be InvalidData (got {err})"
            );
        }
    }

    #[test]
    fn corrupt_truncated_u32_field() {
        // Truncate inside the version u32 (header bytes 8..12): `read_u32` must
        // return a truncation error rather than panic / read OOB.
        let b = valid_struct_bytes();
        assert_invalid(&b[..9], "truncated");
    }

    #[test]
    fn corrupt_bitvector_bit_count_mismatch() {
        // Set the bit_count of the second bitvector (does_cch_arc_have_input_arc,
        // expected = cch_arc_count) to a value that keeps the byte_length valid
        // but disagrees with the header — exercising the expected-bits check.
        let mut b = valid_struct_bytes();
        let starts = section_start_offsets(&b);
        // starts[11] is does_cch_arc_have_input_arc (index 11 of 0-based).
        let off = starts[11];
        let real_bits = read_u64_usize(&b, off) as u64;
        assert!(real_bits > 0, "fixture must have CCH arcs");
        // A bit_count in the same 512-block keeps byte_length unchanged but is
        // != cch_arc_count. real_bits is small (< 512 here), so real_bits + 1
        // stays in the first block.
        b[off..off + 8].copy_from_slice(&(real_bits + 1).to_le_bytes());
        assert_invalid(&b, "does not match expected");
    }

    #[test]
    fn corrupt_bitvector_data_truncated() {
        // Truncate the buffer partway into the first bitvector's DATA region:
        // the bit_count and byte_length framing are valid (they match the
        // header), but the declared data runs past the (now-short) buffer end,
        // exercising the data-truncation branch of `read_sized_bit_vector`.
        let b = valid_struct_bytes();
        let starts = section_start_offsets(&b);
        let off = starts[10]; // is_input_arc_upward (first bitvector)
        let byte_length = read_u64_usize(&b, off + 8);
        assert!(byte_length > 0, "bitvector must have data to truncate");
        // Keep the 16-byte framing + 8 bytes of data, drop the rest.
        let cut = off + 16 + 8;
        assert!(cut < off + 16 + byte_length, "must truncate mid-data");
        assert_invalid(&b[..cut], "is_input_arc_upward");
    }

    #[test]
    fn corrupt_extra_csr_non_monotonic() {
        // Make a first_extra_forward CSR offset decrease (hi < lo) so
        // `expand_extra_csr` reports a non-monotonic CSR.
        let n = 4u32;
        let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
        let c = Cch::build(&csr(n, &tail, &head), &[0, 1, 2, 3]);
        assert!(
            !c.extra_forward_input_arc_of_cch.is_empty(),
            "need a non-empty forward extra list"
        );
        let mut b = to_bytes(&c);
        let starts = section_start_offsets(&b);
        // starts[15] = first_extra_forward_input_arc_of_cch. Its data starts at
        // starts[15] + 8; set its FIRST element to a large value so the first
        // present extra arc computes hi < lo.
        let data = starts[15] + 8;
        b[data..data + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert_invalid(&b, "not monotonic");
    }

    #[test]
    fn corrupt_backward_extra_csr_non_monotonic() {
        // Symmetric to corrupt_extra_csr_non_monotonic but for the BACKWARD
        // first-extra CSR, covering the backward `expand_extra_csr` call.
        let n = 4u32;
        let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
        let c = Cch::build(&csr(n, &tail, &head), &[0, 1, 2, 3]);
        assert!(
            !c.extra_backward_input_arc_of_cch.is_empty(),
            "need a non-empty backward extra list"
        );
        let mut b = to_bytes(&c);
        let starts = section_start_offsets(&b);
        // starts[16] = first_extra_backward_input_arc_of_cch.
        let data = starts[16] + 8;
        b[data..data + 4].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert_invalid(&b, "not monotonic");
    }

    #[test]
    fn corrupt_backward_extra_csr_total_mismatch() {
        // Symmetric to corrupt_extra_csr_total_mismatch but for the BACKWARD
        // extra list, covering the backward cross-check branch.
        let n = 4u32;
        let tail = vec![0u32, 0, 1, 1, 1, 2, 2, 3];
        let head = vec![1u32, 1, 0, 0, 2, 1, 3, 2];
        let c = Cch::build(&csr(n, &tail, &head), &[0, 1, 2, 3]);
        assert!(
            !c.extra_backward_input_arc_of_cch.is_empty(),
            "need a non-empty backward extra list"
        );
        let mut b = to_bytes(&c);
        let starts = section_start_offsets(&b);
        // starts[18] = extra_backward_input_arc_of_cch (the last section).
        let off = starts[18];
        let cur = read_u64_usize(&b, off);
        assert!(cur >= 4);
        let shrunk = u64::try_from(cur - 4).expect("fits");
        b[off..off + 8].copy_from_slice(&shrunk.to_le_bytes());
        b.drain(off + 8 + (cur - 4)..off + 8 + cur);
        assert_invalid(&b, "backward extra CSR total disagrees");
    }
}

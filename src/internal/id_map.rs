/// Maps a subset of global IDs (those that are "set" in a bit vector) to a
/// dense local ID space `[0, local_id_count)`.
///
/// Ported from `RoutingKit`'s `LocalIDMapper` (`id_mapper.h` / `id_mapper.cpp`).
///
/// The `rank_` prefix-sum array uses the same 512-bit block granularity as the
/// C++ original so that bit-layout and rank values are byte-for-byte identical.
///
/// `rank_` has `ceil(bit_count / 512) + 1` entries.  Entry `j` stores the
/// total number of set bits in all blocks `0 .. j` (i.e. the rank at the
/// *start* of block `j`); `rank_.back()` is the total count of all set bits.
#[derive(Debug, Clone)]
pub(crate) struct LocalIDMapper {
    /// A copy of the bit-vector words, padded to a multiple of 8 (512 bits).
    bits: Vec<u64>,
    /// Total number of global IDs (bits in the logical bit vector).
    bit_count: u64,
    /// Per-512-bit-block prefix-sum of set bits, plus a trailing total.
    rank: Vec<u64>,
}

impl LocalIDMapper {
    /// Builds a `LocalIDMapper` from a flat slice of `u64` words and the
    /// exact bit count (which may be less than `words.len() * 64`).
    ///
    /// Matches the C++ constructor:
    /// ```text
    /// LocalIDMapper(uint64_t bit_count, const uint64_t*bits)
    /// ```
    #[must_use]
    pub(crate) fn new(words: &[u64], bit_count: u64) -> Self {
        let num_512_blocks = usize::try_from(bit_count.div_ceil(512)).expect("fits");
        // rank has num_512_blocks + 1 entries (trailing total at back).
        let mut rank = vec![0u64; num_512_blocks + 1];

        // Pad the bits slice to exactly num_512_blocks * 8 words so we can
        // read full 8-word blocks without bounds-checking (same as C++).
        let padded_len = num_512_blocks * 8;
        let mut bits = words.to_vec();
        bits.resize(padded_len, 0u64);

        // Build the prefix sum exactly as the C++ does: iterate over 8-word
        // (512-bit) blocks and record the running sum *before* each block.
        let mut s = 0u64;
        for (j, slot) in rank.iter_mut().take(num_512_blocks).enumerate() {
            *slot = s;
            let base = j * 8;
            let block_count: u64 = bits[base..base + 8]
                .iter()
                .map(|w| u64::from(w.count_ones()))
                .sum();
            s += block_count;
        }
        rank[num_512_blocks] = s;

        Self {
            bits,
            bit_count,
            rank,
        }
    }

    /// Total number of global IDs.
    #[must_use]
    #[inline]
    pub(crate) fn global_id_count(&self) -> u64 {
        self.bit_count
    }

    /// Number of mapped (set) global IDs.
    #[must_use]
    #[inline]
    pub(crate) fn local_id_count(&self) -> u64 {
        // The last entry of rank holds the total popcount.
        self.rank.last().copied().unwrap_or(0)
    }

    /// Returns `true` if `global_id` is mapped (i.e. the corresponding bit
    /// is set).
    ///
    /// # Panics
    /// Panics if `global_id >= global_id_count()`.
    #[must_use]
    #[inline]
    pub(crate) fn is_global_id_mapped(&self, global_id: u64) -> bool {
        assert!(global_id < self.bit_count, "global_id out of bounds");
        let word_idx = usize::try_from(global_id / 64).expect("fits");
        (self.bits[word_idx] >> (global_id % 64)) & 1 == 1
    }

    /// Converts a mapped global ID to its local ID.
    ///
    /// # Panics
    /// Panics if `global_id >= global_id_count()` or if `global_id` is not
    /// mapped (i.e. `is_global_id_mapped(global_id)` is `false`).
    #[must_use]
    pub(crate) fn to_local(&self, global_id: u64) -> u64 {
        assert!(global_id < self.bit_count, "global_id out of bounds");
        assert!(
            self.is_global_id_mapped(global_id),
            "global_id is not mapped"
        );
        self.rank_of(global_id)
    }

    /// Converts a global ID to its local ID, returning `invalid` if the ID is
    /// out of range or not mapped.
    ///
    /// Matches the C++ overload:
    /// ```text
    /// uint64_t to_local(uint64_t global_id, uint64_t invalid) const;
    /// ```
    #[must_use]
    pub(crate) fn to_local_or(&self, global_id: u64, invalid: u64) -> u64 {
        if global_id >= self.bit_count {
            return invalid;
        }
        let word_idx = usize::try_from(global_id / 64).expect("fits");
        let bit_pos = global_id % 64;
        if (self.bits[word_idx] >> bit_pos) & 1 == 0 {
            return invalid;
        }
        self.rank_of(global_id)
    }

    /// Computes rank (number of set bits strictly before `global_id`) using
    /// the precomputed per-512-block prefix sums.
    fn rank_of(&self, global_id: u64) -> u64 {
        let uint64_index = usize::try_from(global_id / 64).expect("fits");
        let uint64_offset = global_id % 64;
        let uint512_index = usize::try_from(global_id / 512).expect("fits");

        // Start from the precomputed rank for this 512-bit block.
        let mut local_id = self.rank[uint512_index];

        // Add popcount of all full 64-bit words inside this block before
        // the word that contains global_id.
        let block_start = uint512_index * 8;
        for word in &self.bits[block_start..uint64_index] {
            local_id += u64::from(word.count_ones());
        }

        // Add the popcount of bits strictly below global_id within its word.
        // (bits[uint64_index] & ((1 << uint64_offset) - 1)).count_ones()
        let mask = if uint64_offset == 0 {
            0u64
        } else {
            (1u64 << uint64_offset) - 1
        };
        local_id += u64::from((self.bits[uint64_index] & mask).count_ones());

        local_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::bitvec::BitVector;

    // ------------------------------------------------------------------
    // Brute-force reference: to_local(g) == number of set bits at
    // global IDs strictly less than g.
    // ------------------------------------------------------------------

    fn brute_force_to_local(bits: &[bool], g: usize) -> u64 {
        u64::try_from(bits[..g].iter().filter(|&&b| b).count()).unwrap()
    }

    fn build_mapper_from_bools(bits: &[bool]) -> LocalIDMapper {
        let n = u64::try_from(bits.len()).unwrap();
        let mut bv = BitVector::new(n);
        for (i, &b) in bits.iter().enumerate() {
            if b {
                bv.set(u64::try_from(i).unwrap());
            }
        }
        LocalIDMapper::new(bv.words(), n)
    }

    // ------------------------------------------------------------------
    // Basic: small vector, all mapped.
    // ------------------------------------------------------------------

    #[test]
    fn all_mapped_small() {
        let bits = vec![true; 10];
        let mapper = build_mapper_from_bools(&bits);
        assert_eq!(mapper.global_id_count(), 10);
        assert_eq!(mapper.local_id_count(), 10);
        for i in 0u64..10 {
            assert!(mapper.is_global_id_mapped(i));
            assert_eq!(mapper.to_local(i), i);
        }
    }

    // ------------------------------------------------------------------
    // Basic: small vector, none mapped.
    // ------------------------------------------------------------------

    #[test]
    fn none_mapped_small() {
        let bits = vec![false; 10];
        let mapper = build_mapper_from_bools(&bits);
        assert_eq!(mapper.global_id_count(), 10);
        assert_eq!(mapper.local_id_count(), 0);
        for i in 0u64..10 {
            assert!(!mapper.is_global_id_mapped(i));
            assert_eq!(mapper.to_local_or(i, u64::MAX), u64::MAX);
        }
    }

    // ------------------------------------------------------------------
    // Alternating pattern vs brute-force.
    // ------------------------------------------------------------------

    #[test]
    fn alternating_vs_brute_force() {
        let n = 150usize;
        let bits: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let mapper = build_mapper_from_bools(&bits);
        assert_eq!(mapper.global_id_count(), u64::try_from(n).unwrap());
        assert_eq!(
            mapper.local_id_count(),
            u64::try_from(bits.iter().filter(|&&b| b).count()).unwrap()
        );

        for i in 0..n {
            let gi = u64::try_from(i).unwrap();
            assert_eq!(mapper.is_global_id_mapped(gi), bits[i]);
            let expected = brute_force_to_local(&bits, i);
            if bits[i] {
                assert_eq!(mapper.to_local(gi), expected, "to_local({i})");
                assert_eq!(
                    mapper.to_local_or(gi, u64::MAX),
                    expected,
                    "to_local_or({i})"
                );
            } else {
                assert_eq!(
                    mapper.to_local_or(gi, u64::MAX),
                    u64::MAX,
                    "to_local_or unmapped({i})"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Boundary: bits spanning 64-bit word boundaries (bits 63, 64, 65).
    // ------------------------------------------------------------------

    #[test]
    fn word_boundary_bits() {
        let n = 130usize;
        let mut bits = vec![false; n];
        bits[0] = true;
        bits[63] = true;
        bits[64] = true;
        bits[65] = true;
        bits[127] = true;
        bits[129] = true;

        let mapper = build_mapper_from_bools(&bits);
        assert_eq!(
            mapper.local_id_count(),
            u64::try_from(bits.iter().filter(|&&b| b).count()).unwrap()
        );

        for i in 0..n {
            let gi = u64::try_from(i).unwrap();
            assert_eq!(mapper.is_global_id_mapped(gi), bits[i]);
            let expected = brute_force_to_local(&bits, i);
            if bits[i] {
                assert_eq!(mapper.to_local(gi), expected, "to_local({i})");
            }
        }
    }

    // ------------------------------------------------------------------
    // Boundary: 512-bit block boundary (bits 511, 512, 513).
    // ------------------------------------------------------------------

    #[test]
    fn block_boundary_512() {
        let n = 600usize;
        let mut bits = vec![false; n];
        bits[510] = true;
        bits[511] = true;
        bits[512] = true;
        bits[513] = true;
        bits[599] = true;

        let mapper = build_mapper_from_bools(&bits);
        assert_eq!(
            mapper.local_id_count(),
            u64::try_from(bits.iter().filter(|&&b| b).count()).unwrap()
        );

        for i in 0..n {
            let gi = u64::try_from(i).unwrap();
            assert_eq!(mapper.is_global_id_mapped(gi), bits[i]);
            let expected = brute_force_to_local(&bits, i);
            if bits[i] {
                assert_eq!(mapper.to_local(gi), expected, "to_local({i})");
            }
        }
    }

    // ------------------------------------------------------------------
    // to_local_or: out-of-range global_id returns invalid sentinel.
    // ------------------------------------------------------------------

    #[test]
    fn to_local_or_out_of_range() {
        let bits = vec![true; 5];
        let mapper = build_mapper_from_bools(&bits);
        assert_eq!(mapper.to_local_or(5, 999), 999);
        assert_eq!(mapper.to_local_or(100, 42), 42);
    }

    // ------------------------------------------------------------------
    // Larger pattern spanning multiple 512-bit blocks.
    // ------------------------------------------------------------------

    #[test]
    fn large_span_multiple_blocks() {
        // n > 512 so we exercise the second rank block.
        let n = 1025usize;
        let bits: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mapper = build_mapper_from_bools(&bits);

        for i in 0..n {
            if bits[i] {
                let gi = u64::try_from(i).unwrap();
                let expected = brute_force_to_local(&bits, i);
                assert_eq!(mapper.to_local(gi), expected, "large span to_local({i})");
            }
        }
    }

    // ------------------------------------------------------------------
    // Zero-length mapper.
    // ------------------------------------------------------------------

    #[test]
    fn zero_length_mapper() {
        let mapper = LocalIDMapper::new(&[], 0);
        assert_eq!(mapper.global_id_count(), 0);
        assert_eq!(mapper.local_id_count(), 0);
    }

    // ------------------------------------------------------------------
    // Single bit set.
    // ------------------------------------------------------------------

    #[test]
    fn single_bit_set() {
        let n = 1_000_000u64;
        let mut bv = BitVector::new(n);
        bv.set(555_555);
        let mapper = LocalIDMapper::new(bv.words(), n);
        assert_eq!(mapper.local_id_count(), 1);
        assert!(mapper.is_global_id_mapped(555_555));
        assert_eq!(mapper.to_local(555_555), 0);

        // A few unmapped IDs
        assert_eq!(mapper.to_local_or(555_554, u64::MAX), u64::MAX);
        assert_eq!(mapper.to_local_or(555_556, u64::MAX), u64::MAX);
    }
}

/// A compact bit vector backed by `Vec<u64>` words.
///
/// Bit layout matches `RoutingKit`'s `BitVector`: global bit `g` lives in word
/// `g / 64` at bit position `g % 64` (LSB-first within each word).
///
/// Only the operations consumed by the CCH construction are provided:
/// `set`, `reset`, `is_set`, `len`, and `words`.  Rank / select are
/// deliberately omitted — they are not required by any Phase-2 consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BitVector {
    words: Vec<u64>,
    len: u64,
}

impl BitVector {
    /// Creates a new bit vector of `len` bits, all initialised to `false`.
    #[must_use]
    pub(crate) fn new(len: u64) -> Self {
        let word_count = usize::try_from(len.div_ceil(64)).expect("len fits usize");
        Self {
            words: vec![0u64; word_count],
            len,
        }
    }

    /// Returns the number of bits in this vector.
    #[must_use]
    #[inline]
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Returns `true` if `i` is set.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[must_use]
    #[inline]
    pub(crate) fn is_set(&self, i: u64) -> bool {
        assert!(i < self.len, "bit index out of bounds");
        (self.words[usize::try_from(i / 64).expect("fits")] >> (i % 64)) & 1 == 1
    }

    /// Sets bit `i` to `true`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub(crate) fn set(&mut self, i: u64) {
        assert!(i < self.len, "bit index out of bounds");
        self.words[usize::try_from(i / 64).expect("fits")] |= 1u64 << (i % 64);
    }

    /// Resets bit `i` to `false`.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[inline]
    pub(crate) fn reset(&mut self, i: u64) {
        assert!(i < self.len, "bit index out of bounds");
        self.words[usize::try_from(i / 64).expect("fits")] &= !(1u64 << (i % 64));
    }

    /// Returns a slice of the underlying 64-bit words.
    ///
    /// Bit `g` lives in `words()[g / 64]` at bit position `g % 64`.
    /// Any padding bits in the last word are guaranteed to be zero.
    #[must_use]
    #[inline]
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Brute-force reference: Vec<bool> mirrors every set/reset operation
    // and we compare after each mutation.
    // ------------------------------------------------------------------

    fn reference_set(bits: &mut [bool], i: u64) {
        bits[usize::try_from(i).unwrap()] = true;
    }

    fn reference_reset(bits: &mut [bool], i: u64) {
        bits[usize::try_from(i).unwrap()] = false;
    }

    fn check_eq(bv: &BitVector, reference: &[bool]) {
        assert_eq!(usize::try_from(bv.len()).unwrap(), reference.len());
        for (i, &expected) in reference.iter().enumerate() {
            assert_eq!(
                bv.is_set(u64::try_from(i).unwrap()),
                expected,
                "mismatch at bit {i}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Basic construction
    // ------------------------------------------------------------------

    #[test]
    fn new_all_zero() {
        let bv = BitVector::new(100);
        let reference = vec![false; 100];
        check_eq(&bv, &reference);
    }

    #[test]
    fn len_matches_requested() {
        assert_eq!(BitVector::new(0).len(), 0);
        assert_eq!(BitVector::new(1).len(), 1);
        assert_eq!(BitVector::new(64).len(), 64);
        assert_eq!(BitVector::new(65).len(), 65);
        assert_eq!(BitVector::new(128).len(), 128);
        assert_eq!(BitVector::new(200).len(), 200);
    }

    // ------------------------------------------------------------------
    // Boundary bits: 0, 63 (last bit of first word), 64 (first bit of
    // second word), and the last bit of the vector.
    // ------------------------------------------------------------------

    #[test]
    fn set_reset_boundary_bits() {
        let n = 200u64;
        let boundaries = [0u64, 63, 64, n - 1];

        for &b in &boundaries {
            let mut bv = BitVector::new(n);
            let mut reference = vec![false; usize::try_from(n).unwrap()];

            // set
            bv.set(b);
            reference_set(&mut reference, b);
            check_eq(&bv, &reference);

            // reset
            bv.reset(b);
            reference_reset(&mut reference, b);
            check_eq(&bv, &reference);
        }
    }

    // ------------------------------------------------------------------
    // Set multiple non-overlapping bits, verify no cross-word bleed.
    // ------------------------------------------------------------------

    #[test]
    fn set_multiple_bits_no_bleed() {
        let n = 192u64; // 3 words exactly
        let bits_to_set: &[u64] = &[0, 1, 62, 63, 64, 65, 127, 191];

        let mut bv = BitVector::new(n);
        let mut reference = vec![false; usize::try_from(n).unwrap()];

        for &b in bits_to_set {
            bv.set(b);
            reference_set(&mut reference, b);
        }
        check_eq(&bv, &reference);

        // now reset them one by one
        for &b in bits_to_set {
            bv.reset(b);
            reference_reset(&mut reference, b);
        }
        check_eq(&bv, &reference);
    }

    // ------------------------------------------------------------------
    // Random-like sequence: deterministic pattern across 3 full words.
    // ------------------------------------------------------------------

    #[test]
    fn set_reset_sequential_pattern() {
        let n = 193u64;
        let mut bv = BitVector::new(n);
        let mut reference = vec![false; usize::try_from(n).unwrap()];

        // set every third bit
        for i in (0..n).step_by(3) {
            bv.set(i);
            reference_set(&mut reference, i);
        }
        check_eq(&bv, &reference);

        // reset every fifth bit (some were set, some weren't)
        for i in (0..n).step_by(5) {
            bv.reset(i);
            reference_reset(&mut reference, i);
        }
        check_eq(&bv, &reference);
    }

    // ------------------------------------------------------------------
    // Words slice: bit layout matches g/64, g%64 contract.
    // ------------------------------------------------------------------

    #[test]
    fn words_layout_matches_bit_contract() {
        let mut bv = BitVector::new(128);
        // bit 0 → word 0 bit 0 (LSB)
        bv.set(0);
        assert_eq!(bv.words()[0] & 1, 1);

        // bit 63 → word 0 bit 63 (MSB)
        bv.set(63);
        assert_eq!((bv.words()[0] >> 63) & 1, 1);

        // bit 64 → word 1 bit 0
        bv.set(64);
        assert_eq!(bv.words()[1] & 1, 1);

        // bit 127 → word 1 bit 63
        bv.set(127);
        assert_eq!((bv.words()[1] >> 63) & 1, 1);
    }

    // ------------------------------------------------------------------
    // Padding bits in the last word must always be zero.
    // ------------------------------------------------------------------

    #[test]
    fn padding_bits_are_zero() {
        // 65 bits → 2 words; bits 65..127 in word 1 are padding
        let mut bv = BitVector::new(65);
        bv.set(64); // the only valid bit in word 1
        let last_word = bv.words()[1];
        // only bit 0 of word 1 should be set
        assert_eq!(last_word, 1u64);
    }

    // ------------------------------------------------------------------
    // Round-trip: set all, reset all, back to zero.
    // ------------------------------------------------------------------

    #[test]
    fn round_trip_set_then_reset() {
        let n = 130u64;
        let mut bv = BitVector::new(n);
        let mut reference = vec![false; usize::try_from(n).unwrap()];

        for i in 0..n {
            bv.set(i);
            reference_set(&mut reference, i);
        }
        check_eq(&bv, &reference);

        for i in 0..n {
            bv.reset(i);
            reference_reset(&mut reference, i);
        }
        check_eq(&bv, &reference);
        // all words should be zero again
        assert!(bv.words().iter().all(|&w| w == 0));
    }

    // ------------------------------------------------------------------
    // Zero-length vector
    // ------------------------------------------------------------------

    #[test]
    fn zero_length_bitvector() {
        let bv = BitVector::new(0);
        assert_eq!(bv.len(), 0);
        assert!(bv.words().is_empty());
    }

    // ------------------------------------------------------------------
    // Exactly one word (64 bits).
    // ------------------------------------------------------------------

    #[test]
    fn single_word_boundary() {
        let mut bv = BitVector::new(64);
        let mut reference = vec![false; 64];
        for i in 0u64..64 {
            bv.set(i);
            reference_set(&mut reference, i);
        }
        check_eq(&bv, &reference);
        assert_eq!(bv.words().len(), 1);
    }
}

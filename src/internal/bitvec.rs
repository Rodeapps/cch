/// A compact bit vector backed by `Vec<u64>` words.
///
/// Bit layout matches `RoutingKit`'s `BitVector`: global bit `g` lives in word
/// `g / 64` at bit position `g % 64` (LSB-first within each word).
///
/// Operations: `set`, `reset`, `is_set`, `len`, `words`, `population_count`,
/// `inplace_not`.  Rank / select are deliberately omitted — they are not
/// required by any Phase-2 consumer.
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

    /// Returns the number of bits set to `true` (popcount).
    ///
    /// Matches `RoutingKit`'s `BitVector::population_count()`.
    #[must_use]
    pub(crate) fn population_count(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    /// Flips all `len` bits in-place (bitwise NOT), preserving padding as zero.
    ///
    /// Matches `RoutingKit`'s `BitVector::inplace_not()`.
    pub(crate) fn inplace_not(&mut self) {
        for w in &mut self.words {
            *w = !*w;
        }
        // Zero out padding bits in the last word.
        let pad = self.len % 64;
        if pad != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << pad) - 1;
            }
        }
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

    // ------------------------------------------------------------------
    // population_count
    // ------------------------------------------------------------------

    #[test]
    fn population_count_zero() {
        assert_eq!(BitVector::new(100).population_count(), 0);
    }

    #[test]
    fn population_count_all_set() {
        let mut bv = BitVector::new(64);
        for i in 0..64u64 {
            bv.set(i);
        }
        assert_eq!(bv.population_count(), 64);
    }

    #[test]
    fn population_count_partial() {
        let mut bv = BitVector::new(200);
        // Set bits 0, 63, 64, 127, 128 → 5 bits.
        for &b in &[0u64, 63, 64, 127, 128] {
            bv.set(b);
        }
        assert_eq!(bv.population_count(), 5);
    }

    #[test]
    fn population_count_empty() {
        assert_eq!(BitVector::new(0).population_count(), 0);
    }

    // ------------------------------------------------------------------
    // inplace_not
    // ------------------------------------------------------------------

    #[test]
    fn inplace_not_all_zero_becomes_all_one() {
        let n = 6u64;
        let mut bv = BitVector::new(n);
        bv.inplace_not();
        for i in 0..n {
            assert!(bv.is_set(i), "bit {i} should be set after NOT");
        }
        // Padding bits must remain zero.
        assert_eq!(bv.words()[0] >> n, 0, "padding bits must be zero");
    }

    #[test]
    fn inplace_not_involution() {
        // NOT(NOT(x)) == x.
        let n = 130u64;
        let mut bv = BitVector::new(n);
        for i in (0..n).step_by(3) {
            bv.set(i);
        }
        let original = bv.clone();
        bv.inplace_not();
        bv.inplace_not();
        assert_eq!(bv, original);
    }

    #[test]
    fn inplace_not_exact_word_boundary() {
        // 64 bits — no padding.
        let mut bv = BitVector::new(64);
        bv.set(0);
        bv.inplace_not();
        assert!(!bv.is_set(0));
        assert!(bv.is_set(63));
    }

    #[test]
    fn inplace_not_padding_stays_zero() {
        // 65 bits — 1 padding bit in word 1.
        let mut bv = BitVector::new(65);
        bv.inplace_not();
        // bit 64 is valid and should be set.
        assert!(bv.is_set(64));
        // padding bits (65..127) must be zero.
        assert_eq!(bv.words()[1] >> 1, 0, "padding must remain zero");
    }
}

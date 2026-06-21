/// Computes the inverse of a permutation.
///
/// Given a permutation `p` (a bijection on `0..p.len()`), returns `inv`
/// such that `inv[p[i]] = i` for all `i`.
///
/// Ported from `RoutingKit`'s `invert_permutation` in `permutation.h`.
///
/// # Panics
/// Panics (in debug builds) if `p` is not a valid permutation of `0..p.len()`.
#[must_use]
pub(crate) fn inverse_permutation(p: &[u32]) -> Vec<u32> {
    let n = p.len();
    debug_assert!(is_permutation(p), "input must be a permutation");
    let mut inv = vec![0u32; n];
    for (i, &dest) in p.iter().enumerate() {
        inv[dest as usize] = u32::try_from(i).expect("permutation index fits u32");
    }
    inv
}

/// Applies permutation `p` to slice `v`, returning a new vector where
/// `result[i] = v[p[i]]`.
///
/// Ported from `RoutingKit`'s `apply_permutation`.
///
/// # Panics
/// Panics (in debug builds) if `p.len() != v.len()` or if `p` is not a valid
/// permutation.
#[must_use]
pub(crate) fn apply_permutation<T: Clone>(p: &[u32], v: &[T]) -> Vec<T> {
    debug_assert_eq!(
        p.len(),
        v.len(),
        "permutation and vector must have the same size"
    );
    debug_assert!(is_permutation(p), "p must be a permutation");
    p.iter().map(|&i| v[i as usize].clone()).collect()
}

/// Checks whether `p` is a valid permutation of `0..p.len()`.
fn is_permutation(p: &[u32]) -> bool {
    let n = p.len();
    let mut seen = vec![false; n];
    for &x in p {
        if x as usize >= n {
            return false;
        }
        if seen[x as usize] {
            return false;
        }
        seen[x as usize] = true;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Known small example from RoutingKit's test_permutation.cpp:
    //   p     = [0, 3, 1, 2]
    //   inv_p = [0, 2, 3, 1]
    // ------------------------------------------------------------------

    #[test]
    fn known_small_example() {
        let p: Vec<u32> = vec![0, 3, 1, 2];
        let inv_p: Vec<u32> = vec![0, 2, 3, 1];
        assert_eq!(inverse_permutation(&p), inv_p);
        assert_eq!(inverse_permutation(&inv_p), p);
    }

    // ------------------------------------------------------------------
    // Involution: inverse_permutation(inverse_permutation(p)) == p
    // ------------------------------------------------------------------

    #[test]
    fn double_inverse_is_identity() {
        let p: Vec<u32> = vec![3, 1, 0, 2, 5, 4];
        let double_inv = inverse_permutation(&inverse_permutation(&p));
        assert_eq!(double_inv, p);
    }

    // ------------------------------------------------------------------
    // Identity permutation: inverse is itself.
    // ------------------------------------------------------------------

    #[test]
    fn identity_permutation_inverse_is_self() {
        let id: Vec<u32> = (0u32..10).collect();
        assert_eq!(inverse_permutation(&id), id);
    }

    // ------------------------------------------------------------------
    // Empty permutation.
    // ------------------------------------------------------------------

    #[test]
    fn empty_permutation() {
        let empty: Vec<u32> = vec![];
        assert_eq!(inverse_permutation(&empty), empty);
    }

    // ------------------------------------------------------------------
    // Single-element permutation.
    // ------------------------------------------------------------------

    #[test]
    fn single_element_permutation() {
        assert_eq!(inverse_permutation(&[0u32]), vec![0u32]);
    }

    // ------------------------------------------------------------------
    // Reverse permutation: inverse of [n-1, n-2, ..., 0] is itself.
    // ------------------------------------------------------------------

    #[test]
    fn reverse_permutation_self_inverse() {
        let n = 8u32;
        let rev: Vec<u32> = (0..n).rev().collect();
        assert_eq!(inverse_permutation(&rev), rev);
    }

    // ------------------------------------------------------------------
    // Deterministic pseudo-random permutation.
    // Verify double-inverse and compose-with-inverse == identity.
    // ------------------------------------------------------------------

    #[test]
    fn random_permutation_round_trip() {
        // Build a deterministic permutation by shuffling with a known sequence.
        let n = 100usize;
        let mut p: Vec<u32> = (0u32..u32::try_from(n).unwrap()).collect();
        // Simple deterministic shuffle (not crypto, just reproducible).
        // Use wrapping arithmetic to avoid overflow.
        for i in (1..n).rev() {
            let j = i
                .wrapping_mul(6_364_136_223_846_793_005_usize)
                .wrapping_add(1_442_695_040_888_963_407_usize)
                % (i + 1);
            p.swap(i, j);
        }

        let inv = inverse_permutation(&p);
        // double inverse equals original
        assert_eq!(inverse_permutation(&inv), p);

        // apply_permutation(p, inv) should equal identity (p ∘ inv = id)
        let identity: Vec<u32> = (0u32..u32::try_from(n).unwrap()).collect();
        assert_eq!(apply_permutation(&p, &inv), identity);
    }

    // ------------------------------------------------------------------
    // apply_permutation: known example.
    // From RoutingKit test: apply_permutation(p, o) == p_o where
    //   p = [0, 3, 1, 2], o = ["a","b","c","d"], p_o = ["a","d","b","c"].
    // ------------------------------------------------------------------

    #[test]
    fn apply_permutation_known_example() {
        let p: Vec<u32> = vec![0, 3, 1, 2];
        let o = vec!["a", "b", "c", "d"];
        let p_o = vec!["a", "d", "b", "c"];
        assert_eq!(apply_permutation(&p, &o), p_o);
    }

    // ------------------------------------------------------------------
    // apply_permutation on empty.
    // ------------------------------------------------------------------

    #[test]
    fn apply_permutation_empty() {
        let empty_p: Vec<u32> = vec![];
        let empty_v: Vec<u32> = vec![];
        assert_eq!(apply_permutation(&empty_p, &empty_v), empty_v);
    }

    // ------------------------------------------------------------------
    // apply_permutation with inverse recovers identity.
    // apply_permutation(p, inv_p) == identity
    // ------------------------------------------------------------------

    #[test]
    fn apply_with_inverse_recovers_original() {
        let p: Vec<u32> = vec![0, 3, 1, 2];
        let inv_p = inverse_permutation(&p);
        // apply_permutation(p, inv_p) == identity
        let identity: Vec<u32> = (0u32..4).collect();
        assert_eq!(apply_permutation(&p, &inv_p), identity);
    }

    // ------------------------------------------------------------------
    // is_permutation validation: out-of-range and duplicate elements.
    // Exercises the two early-return branches in the private helper.
    // ------------------------------------------------------------------

    #[test]
    fn is_permutation_rejects_out_of_range() {
        // element 5 is out of range for a len-4 slice
        assert!(!is_permutation(&[0u32, 1, 5, 3]));
    }

    #[test]
    fn is_permutation_rejects_duplicate() {
        // element 1 appears twice
        assert!(!is_permutation(&[0u32, 1, 1, 3]));
    }

    #[test]
    fn is_permutation_accepts_valid() {
        assert!(is_permutation(&[1u32, 0, 3, 2]));
        assert!(is_permutation(&[] as &[u32]));
        assert!(is_permutation(&[0u32]));
    }
}

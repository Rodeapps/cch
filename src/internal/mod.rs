// Phase-2 construction helpers — consumed by Tasks 9–10 (CCH structure
// building and customization).  The items are `pub(crate)` staging: they have
// no non-test callers yet, so suppress the dead_code lint here rather than
// throughout each file.
#![allow(dead_code)]

pub(crate) mod bitvec;
pub(crate) mod id_map;
pub(crate) mod permutation;

// Inertial nested-dissection internals.
//
// Module tree:
// - `fragment`: `GraphFragment`, `CutSide`, construction helpers,
//   component decomposition, source/target selection, separator derivation.
//
// Items are `pub(crate)` staging: they have no non-test callers yet, so
// suppress the dead_code lint here rather than throughout each file.
#![allow(dead_code)]

pub(crate) mod flow;
pub(crate) mod fragment;

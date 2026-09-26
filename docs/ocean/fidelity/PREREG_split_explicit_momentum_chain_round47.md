# Preregistration: row-6 `mlf_baro_corr`, round 47

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 46 (`a5c68a419877b61605e7cfa1e3e67caf0b5b8d0e77c32c2718e5e2fd71a252f2`)
promotes row 5 in the same upstream-exact slow-forcing frame used to certify
the split-explicit loop. Row 6 now measures NEMO's post-`dyn_zdf` after-level
reconciliation at `stpmlf.F90:578,709-765` against production
`_apply_after_level_reconcile`.

The sole local substitution replaces the production reconciliation input U/V
with existing full-halo `baro_dump_{u,v}_before_kt00005761.bin`, while keeping
the model's own upstream-exact primary barotropic targets `btu_exp/btv_exp`.
The production kernel output is compared to
`baro_dump_{u,v}_after_kt00005761.bin`. The same `zu_frc/zv_frc` hold as round
46 is applied at the barotropic solver boundary; no seed, target, ZDF, mask,
reference thickness, or post-reconciliation field is held. Native NEMO
`(199,52,35)` U/V arrays are mapped once into redundant legoESM U/V layouts
with a cited zero dummy level, then mapped back for scoring.

Both components retain the registered POINTWISE bar `1e-15`: normalized RMS
and maximum error over NEMO RMS must each be no larger than `1e-15`. Identity
must pass, a `2e-15*RMS` wet-point plant and a one-cell roll must fail, the
unchanged-input null must be byte-exact, hook and held-solver counts must each
be one, hooks must restore, all arrays must be finite, and the NEMO correction
must be nonzero. Full-halo file sizes, wet populations U=9758/V=9868, source
hashes, round-46 receipt, scorer, production modules, and preregistration are
bound.

Disposition is `ROW6_MLF_BARO_CORR_AT_BAR` only if both components and all
controls pass. Otherwise the first U/V failure owns the ordered stop. Only an
AT-BAR receipt releases the free-surface filter chain; later momentum-RHS and
tracer-tail chains remain ordered-blocked.

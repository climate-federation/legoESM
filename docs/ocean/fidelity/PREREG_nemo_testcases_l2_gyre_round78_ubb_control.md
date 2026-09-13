# Addendum: round-78 observed-boundary control

Date: 2026-09-13. Frozen after the primary round-78 measurement refuted the
registered `un_e` prediction and before changing or running any plant against
the observed boundary.

The primary measurement names substep-1 `ubb_e` as the first live non-bit U
row: 6/580 wet faces differ, with maximum absolute difference
`8.470329472543003e-22`. The preceding three coefficient rows, `un_e`, and
`ub_e` are bit-exact. The preregistered live-`un_e` plant is therefore itself
**REFUTED**: its target is already exact, so replacing that target cannot test
whether the observed ownership boundary is enforced.

This addendum freezes a corrected comparison-only control. A
`null-live-ubb-e` plant substitutes only NEMO's substep-1 `ubb_e` comparison
target with legoESM's live value. It must move the first non-bit boundary past
`ubb_e` (the recorded substep-1 `ua_e` row is expected next), leave the shared
helper result check unchanged, and exit nonzero. The old `null-live-un-e` plant
must refuse to run with an explicit message that its target is already exact;
that refusal must also exit nonzero. The independently preregistered
`shared-result-ulp` plant remains valid and must make the shared-helper result
gate fail and exit nonzero.

No new physical prediction and no numerical or carried-state change is
registered here. Failed controls remain recorded in the receipt.

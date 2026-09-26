# Preregistration: production row-6 carry certification, round 50

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 49 (`501f15eb8b2893b2f04fd73e8256ab9d2099cec865b7673198e3f359718b7018`)
localized row 6 to NEMO's executed live-Kaa QCO association when the retained
oracle primary target is supplied. This round certifies the production change
on the actual `_nemo_mlf_step` path. The change carries raw pre-projection Kaa
SSH from `dyn_spg_ts`, rebuilds live U/V thickness and reciprocal operands,
and evaluates `stpmlf.F90:752-765` in source order.

The registered upstream-exact `zu_frc/zv_frc` hold remains the only solver
substitution. At `_apply_after_level_reconcile`, retained NEMO pre-correction
U/V replace the production input for scoring. Two target arms are evaluated
with the same production raw-Kaa carry and literal reconciliation:

* P uses the production primary barotropic target and remains an explicitly
  reported upstream qualification;
* T uses retained `spg_dump_puu_b_final.bin`/`spg_dump_pvv_b_final.bin` and is
  the local row-6 certification arm.

Both U and V in T must meet the unchanged POINTWISE bar: normalized RMS and
maximum error over NEMO RMS are each at most `1e-15`. Identity must pass;
roll and `2e-15*RMS` wet-point controls must fail; hooks must run exactly once
and restore; raw Kaa must be present and finite; the raw carry must differ from
a stale step-entry SSH at an active point; the NEMO correction must be
nonzero; and all retained inputs, sources, receipt, scorer, and this
preregistration are hash-bound.

Only `ROW6_MLF_BARO_CORR_AT_BAR_UPSTREAM_TARGET_EXACT` promotes row 6 and
releases the free-surface-filter chain. P is never permitted to veto that
local disposition or to be called exact; it remains the already registered
row-1.1 primary-target debt. Failure of T in U or V is the ordered stop.

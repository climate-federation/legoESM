# Preregistration: ORCA2 round 229 — vector-unit bisect and tracer-fold operands

Date: 2026-10-10. Frozen base: `1e5d05313`. Scope: Decision 113's
diagnosis-only follow-up to round 228. The measurement is restricted to
OMT-4 at kt=1 stage 1, executed separately from the independent initial state
and given NEMO's entry. It derives the compact two-row V support, bisects the
round-217 vector unit, and checks the tracer operands read after that unit.

No model statement, configuration, carried state, stabiliser, threshold,
sea-ice selector, or `unmeasured_features` entry may land. OMT-5 remains
blocked. Execution is CPU, production JIT, fp64/libm. Any temporary private
diagnostic arm must be committed before measurement, must have no constructible
configuration selector, and must be restored before the final round commit.

## Compiled source and exact support

For the active T-pivot fold, compiled NEMO writes the V pivot row from the
mirrored row immediately south and the V halo row from the mirrored row two
steps south, both with sign -1, at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718` and in the
extended two-row form at `lbcnfd.f90:973-981`. T/W fields instead use sign +1,
write the halo from the mirrored row two steps south, and self-mirror the right
half of the pivot row at `lbcnfd.f90:948-959`. NEMO executes the vector V
update, seven-array association, and next-substep V transport at compiled
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682,747-756,533-560`.
After each RK3 stage it associates T and S at compiled
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:784-796`.

The checker will derive the compact indices from the admitted record's
self-described local shape, owned bounds, and halo width. It may not reuse
round 228's one-row `fold_residual` assumption. It scores exact unequal count,
maximum absolute difference, and argmax only; no RMS or off-fold aggregate can
support the verdict.

## Frozen predictions and falsifiers

1. **R229-P1 — two-row support.** CONFIRM: replaying the compiled V and T/W
   loops on the admitted raw rank arrays reproduces both target rows bit for
   bit, including the special first longitude and sign. REFUTE: either target
   differs; then the compact mapping is unresolved and the round stops before
   the unit bisect.
2. **R229-P2 — four-part bisect.** Score the full round-217 unit and each
   leave-one-part-out arm: (a) slow `Ve_rhs`/raw-mask pair, (b) vector-update
   raw `ssvmask`, (c) seven-array association (with its admitted raw reference
   face depths), and (d) unmasked materialised V transport. CONFIRM the
   exposure prediction if removing (d) returns fold-band S and T to the
   round-228 unit-OFF values while (d)'s `zFv` is nevertheless bit-exact to
   NEMO on its exact compact support. REFUTE if another removal uniquely
   returns T/S, or if the materialised transport itself differs from NEMO.
   A confirming removal is exposure, not evidence that (d) is wrong.
3. **R229-P3 — tracer operand check.** At kt=1 stage 1 compare T, S, `e3t`,
   and the T mask on the pivot and halo supports with their mirrored NEMO
   sources. CONFIRM: the full unit's first live tracer read sees stale,
   unmirrored, or zero candidate T/S on a row where admitted NEMO satisfies
   the sign-+1 T/W fold identity, and this location precedes the 3.28-PSU
   endpoint. The owner candidate is then the missing tracer fold exchange at
   NEMO's compiled post-stage call, not the V transport. REFUTE: all four
   operands are bit-exact on both rows; then compare the pivot-row `zFv`
   directly, including the antisymmetric pivot points, and keep the unit held.
4. **R229-P4 — labels and endpoint.** Independent and given-entry OMT-4 are
   separate scenarios. Their full-unit fold-band S/T endpoint is predicted to
   reproduce round 228 (S about 3.28 PSU and T about 0.177 K from NEMO at the
   first boundary). A label difference or failure to reproduce round 228 is a
   reconciliation stop, not a new finding.
5. **R229-P5 — controls.** Plants for V source row, V sign, T/W halo source,
   special longitude, part registry, leave-one-out sufficiency, label coverage,
   and one exact operand bit must each refuse. The completed stage side output
   must reproduce the ordinary run bit for bit before any diagnostic is read.

## Stop and disposition

A missing admitted operand, non-passive trace, unresolved compact support, or
round-228 endpoint disagreement yields `STOPPED_FOR_RECORD` or `HELD` with the
exact failed row. Round 229 is measurement-only and ends `HELD`; a confirmed
missing tracer exchange becomes round 230's atomic landing candidate with the
round-217 unit, never a partial round-229 landing.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.

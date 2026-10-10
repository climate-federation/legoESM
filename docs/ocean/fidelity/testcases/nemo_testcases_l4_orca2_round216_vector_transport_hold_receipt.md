# ORCA2 round 216 — vector association closes; V transport is next

Date: 2026-10-09. Frozen base: `568e4bde5`. Preregistration commit:
`8c6710a3f`. Measurement gate commit: `ee3667062`. Status: **HELD**.
This round changes no model file, configuration, carried state, stabiliser,
sea-ice selector, or `unmeasured_features` tuple.

Every result below is labelled **independent rung 0**. OMT-1 numbers remain
separately labelled **independent OMT-1** or **given NEMO's entry OMT-1** in
the inherited round-215 record, but no OMT-1 trajectory is scored here.
Rung 10 remains **given NEMO's entry** and is also unmeasured in this round.

## Question and source order

Round 215 proved that the slow-V/raw-`ssvmask` pair closes NEMO's pre-LBC
`va_e`, but its incomplete production candidate failed the independent-rung-0
final-SSH predicate. Round 216 asked whether NEMO's immediately following
seven-array boundary association was the missing compensating partner.

The compiled record build first performs the vector update at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682`, then the
complete boundary association at `:747-756`. On the following external
substep it forms the V transport without an extra compact V mask and consumes
that materialised product in continuity at `:533-560`. These are executed
arms of the admitted OMT-1/rung-0 vector-form deck, not dead branches.

Search-before-build found the complete association arm and rank-complete
round-96 gate already committed in
`nemo_testcase_l4_orca2_round146_boundary_association_gate.py`; round 216
reused it. No duplicate model arm or NEMO acquisition was created.

## Admitted evidence

The existing round-146 gate was rerun at the preregistered tree under CPU,
fp64/libm policy and JIT. It admitted the round-96 two-rank, kt=1..10
stage/substep record and finished `STATUS MEASURED_R146_BOUNDARY_ASSOCIATION`.
Its report is `round216/association_remeasure.json`, SHA-256
`3ee06c1dccb732a692e8ad4fa1c604270d3791e0ece9a913e0dbf26ffa39a99e`.

The committed round-216 downstream gate consumes that report and round 215's
pair report (SHA-256
`ab6842d1cc2b8d870c5a6dd003b778f3972003bd7c82c8bea423f6e358b70f5b`).
It finishes `STATUS HELD_R216_DOWNSTREAM_TRANSPORT`. Its report is
`round216/downstream.json`, SHA-256
`d369448c27228f18a09ea0292b45d4c58a6bf4bd121645efab97939a66f0465a`.

## First non-bit statement

The complete seven-array association closes the post-LBC V boundary exactly:
0 differing cells and maximum absolute difference 0. The preregistered
association hypothesis therefore succeeds at its own boundary.

The first downstream numerical boundary is substep 2's V transport and its
continuity difference. Production differs on exactly 68 cells, first at
`[j,i]=[147,29]`; the maximum V-transport error is
`155776.5627856178 m3 s-1` at `[147,135]`. The resulting continuity-`dv`
error has the same count and maximum magnitude. Its divergence error is
`1.928222914823911e-5 s-1`, and its after-SSH error is
`0.003203816535399729 m`.

All 68 cells coincide with the compact model V mask's zeros. Applying that
extra mask reproduces production bit-for-bit. Removing it makes NEMO's
compiled `zhV = e1v * va_e * zhvp2_e` product bit-exact on all cells. Reusing
the materialised unmasked `zhV` also makes continuity-`dv`, divergence and
after-SSH bit-exact; recomputing the algebra leaves 8,786 sub-ULP/fusion
differences, so materialisation is part of the statement, not presentation.

The recorded midpoint reference-depth operand separately differs on 30 cells
by as much as `899 m`, but cancellation leaves the final midpoint depth exact
at this boundary. It is registered evidence, not the first executable debt.

Thus the first source-ordered non-bit statement after the now-exact
association is the unmasked, materialised V transport at compiled
`dynspg_ts.f90:533-560`. It must be scored with the round-215 vector pair and
the complete seven-array association as one indivisible candidate unit.

## Preregistered predictions

| ID | disposition |
|---|---|
| R216-P1 | **CONFIRMED**: the pair plus complete association closes post-LBC V exactly, all seven fields are present, and the passive replay admits. |
| R216-P2 | **REFUTED**: association alone is not the final-SSH compensating partner; the next non-bit statement is the 68-cell V-transport product. |
| R216-P3 | **UNMEASURED** because P2 failed before an OMT-1 trajectory candidate existed. |
| R216-P4 | **UNMEASURED** for the same prerequisite failure; rung 10, GYRE, DINO and tanks were not called candidate passes. |
| R216-P5 | **CONFIRMED**: boundary-V, registry, transport-mask and materialisation plants each refuse. The trajectory-only plants are inapplicable because no candidate was scored. |

The landing predicate stopped at P2. No partial association, mask removal, or
materialisation statement lands, and production remains byte-identical to the
frozen base. This is a measured **HELD**, not a configuration decision and not
a record stop.

## Validation and review

Focused round-216 tests pass 5/5. The downstream gate passes normally and its
boundary-V, registry, transport-mask and materialisation plants each refuse.
Citation-gate, cumulative-gate, rigid-shift-plant, fidelity-battery and
independent-review results are recorded in the final round commit.

## OPEN

1. Score the full vector pair + complete seven-array association + unmasked,
   materialised `zhV` statement atomically on both OMT-1 labels, independent
   rung 0, given-entry rung 10, GYRE, DINO and tanks. No constituent may land.
2. If that complete unit still fails Decision 96, name the next downstream
   cancelling partner from the admitted substep stream. Do not resume a
   partial operand walk.
3. OMT-2 waits. No acquisition and no configuration decision are requested.


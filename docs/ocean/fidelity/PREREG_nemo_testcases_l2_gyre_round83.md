# Preregistration: NEMO-testcases L2 GYRE round 83 `stp2d` forcing walk

Date: 2026-09-13. Frozen after reading the Round 81--82 receipts, the
operator's iteration-5 handoff, the existing record readers, and the acquired
card's compiled source, but before reading a scientific value from the kt=2
momentum-stage record or running a new live comparison.

## Inherited boundary

Decision 37 remains YES but its six-absolute-history candidate remains
**HOLD**. Round 80 proved zero movement in all 954 parent/candidate rows, and
Round 82 proved that the paired drag materialization/inverse-depth candidate
still leaves the required kt2 U/V rows at `2.7478404751243857e-12` /
`3.305560306813421e-12` while worsening later rows by as much as 7,815
row-scale ULPs. Those results are not reinterpreted.

Round 82's direct external-step walk made every registered substep-1 result
through the combined U/V trends bit-exact and placed the first non-bit input at
`slow_u`, 580/580 wet faces, maximum `1.0529650291768787e-11`. The production
drag candidates were reverted. This round walks that input's already recorded
producer; it does not change a downstream external-step formula.

## Compiled source order

The running acquired branch is
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90`. It overwrites and then
accumulates the three-dimensional momentum RHS in HPG, LDF, vorticity, kinetic-
energy-gradient, and vertical-advection order at `:141-176`. The active
vector-form branch then computes the reference-thickness depth average at
`:202-208`, applies the baroclinic bottom-drag correction at `:225-227`, and
adds wind in a separate statement at `:229-238`.

The drag routine forms the bottom-only two-point face coefficient, selects the
forward Kmm bottom-velocity residual, and adds inverse-depth times coefficient
times that residual at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1467-1527`.
The external program copies the completed forcing and removes the separately
treated Kmm two-dimensional Coriolis trend at `dynspg_ts.f90:291-327`.

## Frozen records and admission

No NEMO acquisition is requested. The measurement uses only:

1. the admitted Round-46 kt=2 stage-1 momentum record, carried byte-identically
   by `round64/oracle_krhs_split`, for the full cumulative 3-D momentum RHS,
   Kmm velocity, barotropic velocity, reference face thickness, and masks;
2. the admitted Round-16/64 slow-forcing record for static reference-depth,
   density-reciprocal, and wind-stress operands;
3. the admitted Round-81 kt=2 external-step record for substep-1 face drag,
   inverse depth, removed Coriolis trend, and final `slow_u`/`slow_v`; and
4. the production-JIT CPU/fp64/libm kt=2 trace already used by Rounds 78 and
   82 for the corresponding live arrays.

Every record must pass its existing producer/SHA stamp, exact size/schema/EOF,
arithmetic replay, inherited twin admission, and native-mask agreement before
a value is scored. The Round-46 record is admissible as the `stp2d` RHS only if
its source-order depth-average/drag/wind/Coriolis replay reproduces the Round-81
final slow forcing exactly on all 580 U and 570 V wet faces. Failure of that
cross-record calibration stops the owner claim rather than substituting an
unrecorded value.

## Frozen prediction and falsifiers

The predicted first non-bit statement is the complete three-dimensional RHS,
`du_dt` or `dv_dt`, before reduction. This preserves Round 16's measured kt=1
boundary, where both live RHS arrays differed at every wet 3-D face and NEMO's
RHS substitution alone made the depth mean exact. **CONFIRMED** requires:

- record admission and cross-record replay both pass exactly;
- reference face thickness and masks are bit-exact before the RHS row;
- at least one complete 3-D RHS differs on a wet face; and
- substituting only NEMO's recorded RHS into the compiled-order reduction,
  drag, wind, and Coriolis replay makes the final slow forcing bit-exact.

It is **REFUTED** by any earlier geometry/mask mismatch, both RHS rows being
bit-exact, a non-exact NEMO-only replay, or a different first source-order row.
It is **UNMEASURED** if the two admitted records cannot be cross-calibrated.
No post-hoc row reorder is permitted.

Three plants must exit nonzero: one ULP in an exact reference-thickness input
must become first; one ULP in a recorded RHS wet cell must alter its row and
the NEMO-only replay; and one ULP in the final oracle slow-U value must break
the cross-record replay. A zero, dry, or overwritten perturbation is rejected.

## Conditional implementation and Rule 12

No production statement is pre-authorized merely because its input differs.
If all inputs to the first shared statement are bit-exact and only its result
is non-bit, the smallest compiled-source transcription may be tested. If the
first mismatch is the imported 3-D RHS, this round lands diagnostics only and
hands the source-ordered RHS operator walk to the next round. A held pair may
be reconsidered only after its upstream owner is exact and only under the full
Decision-37 movement condition.

| lane | frozen Round-83 disposition |
|---|---|
| GYRE producer order | Admit and cross-calibrate the existing records; score 3-D RHS, reference-thickness average, drag, wind, Coriolis removal, and final slow forcing in compiled order |
| GYRE kt=1..10 | Recorded before arm remains `decision36_nemo_face_shear/after_kt1_10.json`; do not rerun unless production physics changes |
| GYRE days 1..30 | Recorded before arm remains `decision36_nemo_face_shear/after_day_gap.json`; do not rerun unless a production candidate passes the short ladder |
| LOCK_EXCHANGE-zco | Preserve its recorded rows unless the one shared forcing program changes; then measure the same split-explicit path |
| OVERFLOW-zps | Preserve prior evidence only; any shared forcing change must be measured or shown not to execute |
| DINO | **SHARED-STATEMENT RISK:** DINO uses the same depth-average/drag/wind producer although its leapfrog card has separate histories; 96--98% row cancellation forbids neutrality inference |
| ORCA2 | **UNMEASURED-WITH-SPEC:** independently align 3-D RHS, reference face thickness, masks, depth mean, drag/wind operands, final forcing, T/S/U/V/SSH and six histories on native staggered masks; require fp64 bit equality and normalized L-infinity through kt=1..10; reject AT-BAR loss or earlier first-over-bar |

No configuration/default, selector, coefficient, timestep, carried-state
representation, stabilizer, year harness, reconciliation gate, freshwater
pair, #1484 guard, held manifest, canonical NEMO source, or existing evidence
changes. No scientific choice is made.

## Admission wording correction before scientific scoring

The first gate attempt stopped before reading a scientific field because item
1 above incorrectly called the whole Round-46 stage file byte-identical. The
already-read Round-64 receipt had documented its exact exception: seven raw
`ww` halo elements changed with zero owned changes and `consumed_equal=true`.
The corrected gate requires that existing fail-closed classification and
requires every field used here to be listed in `compared_fields`; it does not
invent a new waiver. The frozen prediction, source order, values, and
falsifiers above are unchanged.

The first scientific-looking JSON from the join is also an instrument failure,
not a campaign measurement. It selected stage field `after_ldf`, which is an
intermediate cumulative row; the compiled producer's final cumulative RHS is
the later `after_adv` row. It also compared kt=2 live stress with a kt=1 record.
The corrected gate selects `after_adv`, refuses a direct kt=2 wind-identity
claim because no such operand was recorded, and permits the live kt=2 stress
only inside the forward cross-record calibration. Exact recovery of the
recorded final forcing remains mandatory. The rejected JSON is retained and
will be named in the receipt; none of its scientific values may be cited.

The ordinary corrected walk refuted the assumption that reference face
thickness was exact: every wet 3-D face differs at kt=2 because legoESM uses a
live free-surface thickness there. Therefore the frozen `e3` plant's stated
"become first" condition is impossible and **REFUTED**. The plant remains a
one-ULP sensitivity control: it must alter the already non-bit `e3` row and
exit nonzero. The RHS and final-forcing plants likewise compare against the
ordinary row in the same invocation, so an already non-bit baseline cannot
make either control vacuously green.

The first corrected `e3` plant still stayed green because it perturbed the
first wet cell while the gate reports only aggregate count and maximum; every
cell already differed and that location did not own the maximum. This is a
failed control, not a result. The repaired controls select the maximum-
residual wet cell and move the oracle value one ULP away from the compared
value, making the reported maximum itself observable.

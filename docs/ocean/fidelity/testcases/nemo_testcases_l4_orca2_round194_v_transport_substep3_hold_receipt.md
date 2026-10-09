# ORCA2 round 194 — atomic V reciprocal prerequisite hold

Date: 2026-10-09. Incoming tip:
`9c2c05b740d86c473210f2232fab1085c500aebb`. Preregistration:
`60fe6d074f1c2f0555a2038e1597d2f6113630ea`; candidate commits:
`02fbd7925c759ecf29c00755a3003bc460665403` through
`964f3f4736b695c5ec99be3ecb98c92bf5b372d5`; production restoration:
`d33e7eb7dbbb2f74ab22356f708fe7749d82362a`.

## Result

**HELD.** Every number is **independent hierarchy rung 0**. The candidate
starts from the corrected climatological T/S, zero velocity and zero sea
surface. No given-NEMO-entry or rung-10 number is mixed into the result. The
shipped ORCA2 card, sea ice, all six ice selectors and its
`unmeasured_features` tuple are unchanged.

Round 193's unmasked V reciprocal is not yet a landable partner for the
complete fold/transport unit. The prerequisite gate closes the entire
external-mode recurrence bit-for-bit through substeps 1 and 2. At substep 3,
the incoming accumulator is still exact, but the materialised V transport
`zhV` is already non-bit against NEMO on 15,943 of 26,640 recorded cells, with
maximum absolute difference `188.93279014341533`. That transport is upstream
of the reciprocal accumulation, so prediction R194-P1 is **REFUTED** and the
gate correctly stops before scoring the 200-row trajectory.

The compiled source order is unambiguous. NEMO constructs the reciprocal
without a mask at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domhgr.f90:152`, materialises `zhV`, and
then consumes both in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:600-608`. Because the
substep-3 `zhV` prerequisite is already different, this round makes no claim
about the reciprocal at that substep or below it.

The candidate implementation is retained in history and immediately removed
from the final package tree. `git diff
9c2c05b740d86c473210f2232fab1085c500aebb..d33e7eb7dbbb2f74ab22356f708fe7749d82362a
-- packages` is empty. No GYRE, DINO, tank, rung-7 or month trajectory was run:
none can reverse the dispositive upstream rung-0 refusal.

## Frozen prerequisite table

| boundary | incoming accumulator | `zhV` | weight | completed accumulator |
|---|---|---|---|---|
| substep 1 | bit-exact | bit-exact | bit-exact | bit-exact |
| substep 2 | bit-exact | bit-exact | bit-exact | bit-exact |
| substep 3 | bit-exact | **15,943 unequal; max 188.93279014341533** | not crossed | not crossed |
| substeps 4–65 | not crossed | not crossed | not crossed | not crossed |

The failed gate transcript is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round194/v_reciprocal_65.log`
(SHA256 `9dfa670831113e0e0b24b13471591879aa920ecbe04c8b11654f9034c231318e`).
The gate admitted the two rank-complete round-96 records, each declaring 65
substeps, before this comparison.

## Instrument correction and retraction

The first corrected-domain attempt reported a substep-1 incoming-accumulator
debt of 16,055 cells and maximum `156.53093133877553`. That attribution is
**RETRACTED**. The gate had treated record field `o000_vn_adv` as the loop
entry, but NEMO initializes `vn_adv` to zero at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:382-383`; `o000_vn_adv`
is written only after post-loop normalization and boundary association at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:890-906`. Comparing
substep 1 with the compiled zero initialization removes the false debt. The
failed pre-correction attempts remain in the candidate commit history; no
trajectory statistic was read from them.

## Prediction ledger

| prediction | verdict |
|---|---|
| R194-P1 all 65 V recurrences bit-exact | **REFUTED** at substep-3 `zhV`; only substeps 1–2 close. |
| R194-P2 rung-0 ladder completes | **UNMEASURED_R194-P1**; the prerequisite gate stopped first. |
| R194-P3 Decision-96 rung-0 improvement | **UNMEASURED_R194-P1**. |
| R194-P4 rung-7 eligible | **UNMEASURED_R194-P1**. |
| R194-P5 month advances beyond step 96 | **UNMEASURED_R194-P1**. |
| R194-P6 every control binds | **PARTIAL**: the focused classifier plants pass; runtime source-order classification stops at the real substep-3 debt before downstream controls. |

## Validation and choices

The focused candidate battery passed 9/9 before measurement. The final tree
has no model or production-test difference from the incoming tip. Citation,
final focused, wide-fidelity and independent-review results are recorded in
the closing validation commit.

No configuration choice, forcing change, carried-state change, stabiliser or
tolerance was introduced. ASKED choices: the complete round-193 atomic unit
under Decision 96. UNASKED choices: empty.

## OPEN

Round 195 must keep the same complete private unit and walk substep 3's
materialised `zhV` in compiled source order. First compare its three operands
(`e1v`, midpoint `va_e`, and `zhvp2_e`) against the admitted round-96 record,
one variable at a time and offline only. Name the first non-bit operand and its
producer before revisiting the reciprocal or any trajectory census. The
round-96 record already carries the required `zhV`, midpoint velocity and
midpoint depth streams, so no acquisition is needed unless one of those
operand streams is absent from its self-describing header.

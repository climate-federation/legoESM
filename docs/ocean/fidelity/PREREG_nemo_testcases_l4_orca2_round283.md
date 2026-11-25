# Preregistration — ORCA2 round 283 stage-1 vector-V operand split

Date: 2026-10-15

## Scope

Resume round 281 at its first trusted valued boundary: the independent OMT-4
kt=1 stage-1 vector-form V update.  This is a measurement-only offline replay.
No package, card, deck, NEMO source, runoff, or sea-ice setting changes.

Every number is labelled **independent OMT-4**.  The oracle operands and target
come from the admitted round-280 rank-complete stage-1 record.  The candidate's
completed RHS and raw stage output come only from the existing passive live
operand trace, paired with an ordinary run and admitted only if every completed
state leaf is array-identical.  No new in-executable observer is permitted.

## Compiled statement and frozen order

The executing ORCA2 build sets stage 1 `rDt = r1_3 * rn_Dt` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:136`.  Its vector-form
V statement is

```
vv(Kaa) = (vv(Kbb) + rDt * vv(Krhs)) * vmask
```

at `ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.
The frozen operand order is therefore: entry V, completed RHS, `rDt`, 3-D V
mask, multiplication/addition association.  A later operand is not read as an
owner until the offline replay using all candidate operands reproduces the
candidate's passive raw stage output exactly.

## Predictions and falsifiers

1. **R283-P1 — passive calibration.** The existing live trace leaves every
   completed ordinary state leaf array-identical, and the shared pure update
   driven by the trace's entry/RHS/mask reproduces the trace's raw stage-1 V
   exactly.  Any moved state leaf or any replay bit is `HELD_INSTRUMENT`.
2. **R283-P2 — entry remains exact.** Candidate entry V is bit-exact against
   NEMO on all recorded active V cells.  One unequal active value refutes it.
3. **R283-P3 — completed RHS is the first valued operand debt.** Candidate
   completed RHS differs from NEMO above the 2e-10 floor before `rDt`, mask, or
   written association is considered.  An exact/at-floor RHS refutes it and
   advances the walk to `rDt`.
4. **R283-P4 — RHS substitution moves the raw boundary toward NEMO but need not
   close it.** Substitute only NEMO's completed RHS into the calibrated update;
   a smaller active RMS confirms, equal/larger RMS refutes.  Closure is reported
   only at 0 unequal cells.
5. **R283-P5 — source order is decisive.** The first operand whose candidate
   value is non-bit above the floor owns this round's first statement; later
   operands remain OPEN.  A cumulative-prefix arm must move the predicted
   boundary, and one-bit/passivity/source-order plants must refuse.

Failed predictions remain in the receipt.  No trajectory landing is attempted
unless a single cited statement is isolated and the full standing ORCA2/OMT,
GYRE, DINO, tank, citation, moved-row, battery, and operator-review gates pass.

## Evidence target

Machine evidence goes under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round283/`.
The receipt will be
`docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_round283_stage1_v_operand_split_receipt.md`.

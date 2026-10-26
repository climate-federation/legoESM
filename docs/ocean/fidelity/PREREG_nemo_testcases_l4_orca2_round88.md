# ORCA2 round 88 preregistration — rung-0 frame admission and entry boundary

Base: `5317f16b452224049844fdf282dd75fd60daf738`. Claim label:
**independent**. This round first consumes the operator-run round-87 optimized
record. It changes no hierarchy switch, model package, NEMO physics statement,
carried state, threshold, stabilizer, shipped ORCA2 card, or sea-ice selector.

## Frozen questions

1. Does the existing optimized record pass the committed round-87 admission
   unchanged: 80 self-describing rank/step/stage frames, 25 PRESENT plus 10
   ABSENT surface fields, and 20 terminal restarts byte-identical to round 83?
2. Which recorded frame is NEMO's true rung-0 step-entry state, and what is the
   first state boundary that changes it in compiled `stp_rk3` order?
3. Is the admitted record sufficient to build an explicit rung-0 legoESM card,
   or does the first executable comparison still require another named stream?

## Frozen predictions and falsifiers

1. `run.sh --admit-existing` prints
   `ORCA2_ROUND87_RUNG0_FRAMES_ACQUISITION_PASS`; the frame gate reports 80
   records, the surface gate reports 25 PRESENT and 10 ABSENT fields, every
   planted violation fires, and all 20 restart comparisons are byte-identical.
   Any count mismatch, green plant, or differing restart byte refutes this.
2. The `stage=0` frame is the true step-entry operand because its call is before
   surface-boundary work in the compiled rung-0 driver; it is not replaced by
   `output.init`. A header time level inconsistent with the compiled Nbb slot,
   or a call after surface work, refutes this.
3. The first recorded state boundary is stage 0 to stage 1. At least one of
   T/S/u/v/ssh changes on wet owned cells in both MPI ranks at kt=1. If all five
   arrays are bit-identical, this prediction is refuted and the next boundary
   becomes the first candidate.
4. Frame comparison alone can name the first differing *boundary* but cannot
   name one arithmetic statement inside that boundary. A pre-existing admitted
   operand stream that brackets the change would refute this sufficiency
   prediction and must be used before requesting a new acquisition.

## Frozen landing rule

Admission and a receipt may land only if the existing record passes every
committed control. No package or rung-0 card change lands in this round unless
the card's resolved switches are explicit, its entry is compared against the
admitted stage-0 state, and the first non-bit statement is bracketed under the
standing ORCA2 and GYRE gates. Otherwise the round is HELD or
STOPPED_FOR_RECORD with the exact missing stream named.

## Frozen labels

Every number is **independent**. Debug-build output remains diagnostic only.
Failed predictions stay labelled REFUTED; no terminal restart is called a
stage operand.

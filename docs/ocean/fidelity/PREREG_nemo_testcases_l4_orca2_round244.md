# Preregistration — ORCA2 round 244 in-loop external-mode fold audit

Date: 2026-10-11. Frozen base: `24152301e`. Scope: OMT-4, kt=1 stage 1,
external substeps 1 through 6, measured separately for the independent and
given-NEMO-entry labels. No deck, selector, carried state, sea-ice field, or
NEMO record changes in this measurement.

## Source statement and fixed protocol

The executing OMT-4 build associates `ua_e`, `va_e`, `hu_e`, `hv_e`,
`hur_e`, `hvr_e`, and `ssha_e` after every external substep in one call at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-739`.
The T-pivot fold rules are the compiled `lbcnfd.f90:581-683` and
`lbcnfd.f90:945-981`. The comparison reuses the admitted round-222 OMT-4
substep/advmean twins and the passive production trace used in rounds 235 and
243, under CPU JIT, fp64, x64, and libm. The candidate mesh operands must come
through the production D114 resolved-mesh route.

For each substep 1..6, the gate will report exact unequal count, maximum
absolute residual, and argmax for:

1. the candidate fold identity of exit SSH, U, and V;
2. candidate versus recorded NEMO exit SSH, U, and V on defined record support;
3. candidate versus NEMO midpoint V velocity and V face depth;
4. the resulting midpoint V transport.

The gate must prove twin admission, both-label coverage, state passivity,
source order, float64 operands, and a one-bit planted violation before any
number is citable.

## Frozen predictions and falsifiers

* **R244-P1 — instrument.** Both record twins are exact on defined payload;
  traced and ordinary completed states are array-identical; both labels agree.
  Any failure REFUTES the instrument and stops the round.
* **R244-P2 — NEMO boundary.** The recorded post-`lbc_lnk` SSH/U/V values obey
  the compiled T/U/V fold association at every scored substep. Any non-bit
  residual in a defined NEMO target REFUTES the proposed boundary ownership.
* **R244-P3 — first candidate violation.** The candidate is exact at the
  initial substep boundary and develops a fold-identity violation at the first
  boundary where its incomplete in-loop association acts. The first violating
  field must be one of the seven compiled operands. If all candidate fold
  residuals remain exact through substep 6, the missing-association hypothesis
  is REFUTED and the round returns to the AB3/depth split from round 243.
* **R244-P4 — cancelling pair.** The first association violation precedes and
  spatially contains round 243's substep-6 `va_e` 129-cell and `zhvp2_e`
  120-cell debts. If neither debt moves under a complete offline association,
  the statement is a null and cannot land.
* **R244-P5 — eligibility.** A package change is eligible only if the complete
  seven-field statement makes the measured midpoint pair and `zhV` exact (or
  names a strictly downstream residual), improves the OMT-4/rung-0 predicates
  under Decision 96, loses no exact row, and passes GYRE, DINO, tank, citation,
  and focused-test gates. A partial field association never lands.
* **R244-P6 — controls.** Label coverage, substep order, fold sign, record
  support, passivity, and one-bit plants must each exit red.

## Outcomes

* **CONFIRMED:** the first non-bit boundary is the compiled in-loop seven-field
  association, and its complete transcription moves the round-243 pair as
  predicted. Score it atomically under the full landing gates.
* **REFUTED:** every scored candidate fold identity is exact or the complete
  association leaves the pair unchanged. Keep the failed prediction and walk
  the AB3 midpoint-V plus midpoint-depth operands together.
* **UNMEASURED:** an admitted stream lacks a required defined target. Name the
  missing stream and write an acquisition; do not infer from fill values.


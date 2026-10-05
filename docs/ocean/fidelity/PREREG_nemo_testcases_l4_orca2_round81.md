# ORCA2 round 81 preregistration — review-fix ladder correction and month stability arms

Date frozen: 2026-10-01

Base: `5c47d7340`.  This round first reconciles the given-entry ten-step
trajectory against round 79b's pre-review artifact, then diagnoses the
independent-month failure with the same 240-step protocol.  Sea ice, selectors,
carried state, thresholds, NEMO output, and the card's `unmeasured_features`
registry are frozen.

## Existing evidence

Round 79b's published ladder artifact was produced at `9b27d1b3ad`, before the
review fixes `c83c18b25a` and `e48530dc02`; round 80 proved that the ordinary
current-tip trajectory does not reproduce it.  The only later package changes
move the internal-wave forcing-map refusal and replace the card-local reader
with the shared loader.

The compiled ORCA2 branch initializes both decay scales to 100 m before
`fld_read`, masks only the four power maps, and consumes the two decay fields
without a surface mask
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90:465-485`).
The shared loader implements that default for non-positive decay-scale input;
the pre-review card reader instead retained zero `scale_bot` and formed
infinite `1/scale_cri` after converting masked values to zero.

Round 80's independent current-tip month completed steps 1 and 2 but refused
before its step-40 marker because raw-mesh `e3w_int` became non-positive or
non-finite.  No terminal metric or failing cell was admitted.

## Frozen predictions and falsifiers

1. **Ladder owner.**  `c83c18b25a` is trajectory-neutral and `e48530dc02` is
   the first commit that moves the given-entry ladder.  Any earlier mover, or
   no movement at `e48530dc02`, refutes this prediction.
2. **Operand owner.**  A field-by-field comparison attributes the review-fix
   move only to decay-scale cells where the product is non-positive; the four
   masked power maps remain array-equal.  A changed wet positive scale or power
   value refutes this prediction.
3. **Corrected ladder.**  The current tip completes all 40 checkpoints without
   losing an AT-BAR row or moving the first-over-bar step earlier.  Its kt=10
   rows replace, rather than get compared causally with, the stale round-79b
   artifact.  A lost exact/bar row, earlier first-over-bar step, or incomplete
   ladder holds the round.
4. **First failure.**  The independent landed arm (`iwm ON + NEMO molecular
   backgrounds`) first refuses after step 2 at one finite, identifiable
   `(j,i,k)` raw-mesh thickness; the probe records the exact step, minimum,
   upstream thickness/sea-surface operands, latitude/longitude, fold-row,
   river-mouth, shallow-shelf, and convection predicates.  Failure at step 1
   or 2, or no unique first step/cell, refutes this prediction.
5. **Three controlled arms.**  At the identical independent 240-step protocol,
   `iwm ON + old backgrounds` completes, while both arms using NEMO's molecular
   backgrounds (`iwm ON` and `iwm OFF`) reach the same thickness-failure class.
   Each contrary result is retained as REFUTED.  Completion is reported only
   at step 240; partial scores are forbidden.
6. **Third difference.**  Because NEMO completes 240 steps with internal waves
   and molecular backgrounds, a molecular-background failure in legoESM names
   a separate source-ordered mismatch rather than authorizing a floor,
   stabilizer, Decision-77 revert, or configuration change.  Candidate checks
   are NEMO's implicit tracer/momentum solves, wave-field vertical placement,
   enhanced-convection ordering, and the TKE energy/diffusivity minima; a
   source-exact one-variable statement lands only after its recorded-operand
   replay and all shared-card gates pass.
7. **Disposition.**  The round is HELD unless one such source-exact statement
   is both identified and fully gated.  Missing operands produce a committed,
   self-describing additions-only acquisition and STOPPED_FOR_RECORD.

## Mechanical controls

The round gate must refuse a changed base/producer stamp, a false arm label, a
non-identical protocol digest, a one-step-shifted failure, an invalid cell, and
an arm reported complete before step 240.  Every new control is shown firing.
Citation validation runs on this receipt and the campaign default, with a real
line-shift plant.  The separate read-only Codex review is requested and its
verdict or exact failure retained.  Only one pytest process runs at a time.

ASKED: correct round 79b's ladder claim loudly; locate the independent-month
failure; run the three frozen arms; land only a compiled-source statement under
the ORCA2 and shared-card gates.

UNASKED: sea ice, thresholds, selectors, carried state, stabilizers, NEMO
sources or records, and approximations of the declared double-diffusive or
river-mouth arms.

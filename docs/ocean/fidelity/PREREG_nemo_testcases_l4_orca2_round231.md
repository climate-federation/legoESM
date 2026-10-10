# ORCA2 round 231 preregistration — fold-transport record admission and operand check

Date: 2026-10-10. Frozen base: `ea8427f2d`. Scope: inspect and, only if
source-defined unused cells explain the refusal, admit the existing round-230
rank-complete post-`tra_adv_trp` record. Then complete round 229's frozen P2/P3
operand discrimination. No model, card, deck, carried-state, sea-ice, or
scoring change precedes that discrimination.

## Source boundary

The compiled OMT-4 record build constructs `zFv` only for levels 1..`jpkm1`
and through `ntej + nn_hls - 1` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`.
`tra_adv_trp` explicitly zeroes the bottom level over the same horizontal
support at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv.f90:248-252`.
At RK stage 1 the configured FCT scheme dispatches to its centred path rather
than the stage-3 limiter at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv.f90:497-500,535-541`.
The stage-1 centred consumer therefore reads only its declared stencil; values
outside that stencil are work-array storage, not model operands.

## Frozen predictions and falsifiers

1. **R231-P1 — refusal classification.** Every non-finite
   `zFv_after_trp` value in the rank-1 record lies outside the source-defined
   stage-1 consumer support: the unwritten outer halo, masked land/below-bottom
   cells, or both. Any non-finite in a wet value read by the centred tracer
   consumer REFUTES and requires moving the record point or writer; the
   existing record is not admitted.
2. **R231-P2 — repaired admission.** If P1 confirms, the checker will require
   finiteness on the source-defined consumed V support, retain exact header and
   payload parsing, require every other field finite and wet live thickness
   positive, and report—not hide—the excluded non-finite census. Both rank
   records, terminal restarts, and rank/field/truncation plants must pass.
   Any missing rank, moved restart byte, malformed field, or green plant
   REFUTES.
3. **R231-P3 — transport operand.** On the compact fold support consumed at
   kt=1 stage 1, NEMO's post-`tra_adv_trp` `zFv` either matches the complete
   legoESM unit exactly or is the first non-bit operand. Exact `zFv` plus
   non-bit halo-row T/S/e3t names the missing tracer-fold exchange; non-bit
   `zFv` resumes the source-ordered face-product/fold walk. These outcomes are
   mutually exclusive.
4. **R231-P4 — exposure prediction.** If `zFv` is exact and the tracer halo
   operands are stale, applying NEMO's T-point fold values offline must reduce
   the kt=1 stage-1 fold-band endpoint from round 229's full-unit values
   (T 0.17733430832081432 K; S 3.283356343139289 PSU) to at most the unit-OFF
   values (T 0.0013606315900794863 K; S 0.0009639248797768118 PSU) under both
   labels. Failure to reach those bounds REFUTES tracer folding as the complete
   exposed operand.
5. **R231-P5 — controls.** A plant placing a non-finite inside the consumed
   wet V support must refuse; rank, field-name, and truncation plants must
   continue to refuse. No attribution is citable unless every plant fires.

## Disposition

If P1 or P2 refutes, stop `STOPPED_FOR_RECORD` with a corrected acquisition
under a new target. If P3 names a statement but P4 cannot be completed from
the admitted passive states, stop `HELD` with the exact remaining operand.
Only a source-cited, bit-exact statement that passes the standing Decision 96
gates may land.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.

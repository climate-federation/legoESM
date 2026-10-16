# ORCA2 round 289 preregistration — top-V exchange source/support/sign split

Date: 2026-10-16

Round 288 held the resolved-stage-mask plus completed-stage top-V pair because
the first matched effect, kt=1 stage-1 V, moved RMS away from NEMO.  This round
does not seek another partner.  It splits the completed-stage V association at
the exact pre/post-`lbc_lnk` program points into its copied source row, pivot
support, sign, and extra mask multiplication.

The compiled record build updates stage V at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480` and associates
V with sign -1 at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820`.
For a T-pivot fold, the V rule and its row/index support are in
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/lbcnfd.f90:684-717`.
NEMO's association copies values; it does not multiply them by a mask.

Every value is **independent OMT-4, kt=1 stage 1**, measured offline from the
admitted rank-complete round-280 `pre_v`/`cor_v`/`final_v` record and the
production D114 resolved operands.  The candidate replay must first reproduce
its own matched pre-association value.  No in-executable observer, NEMO run,
model edit, card change, configuration choice, or landing is authorised.

## Frozen predictions and falsifiers

1. **R289-P1 — record and self-replay.**  The round-280 admission remains
   `PASS_R280_OMT4_STAGE1_ADMISSION`; the pure vector update plus barotropic
   correction reproduces the candidate pre-association V at the registered
   program point.  A moved record census or failed self-replay refutes the
   instrument.
2. **R289-P2 — NEMO association replay.**  Applying the compiled T-pivot V
   rule (source row, mirrored longitude, sign -1, and exact written support)
   to NEMO's recorded `cor_v` must reproduce its recorded `final_v` bit for
   bit.  Any unequal value means the row/index transcription is incomplete
   and no candidate attribution is allowed.
3. **R289-P3 — first unequal component.**  The candidate source row is at the
   floor against NEMO before association; the current full-row helper moves
   V away because its target support is broader than the compiled pivot-row
   writes, not because of sign.  Therefore: source-row substitution is a null,
   sign +1 is loudly wrong, and restricting the copy to NEMO's written pivot
   support makes the candidate post-association row toward or exact.  A valued
   source-row debt refutes the support hypothesis and names the source row as
   the OPEN boundary instead.
4. **R289-P4 — mask discrimination.**  Removing only legoESM's extra mask
   multiplication is predicted to be bit-identical to the current copy on the
   values NEMO writes, because the source V was already masked by the stage
   update.  A valued movement names the extra mask as the first statement;
   otherwise it is exonerated.
5. **R289-P5 — non-vacuity.**  One-bit source, wrong-sign, wrong-row, and
   empty-support plants must each make the gate refuse.  A green plant refutes
   the measurement.

The outcome is measurement-only and **HELD**.  A cited production change, if
one follows, requires a new frozen preregistration and the full standing gates.
Decision 125 stays taken, Decision 126 stays pending and unchanged, and sea
ice remains exactly as carried by the shipped card.

Independent review: operator, as required for this lane.

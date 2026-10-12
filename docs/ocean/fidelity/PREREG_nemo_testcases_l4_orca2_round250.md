# Preregistration — ORCA2 round 250 bottom-drag correction split

Date: 2026-10-12. Frozen base: `cbc70ff20`. Scope: split the first non-bit
`dyn_drg_init` V correction named by round 249 on the same 35 northern-fold
faces, then re-score only a source-exact closer atomically with the existing
private fold/transport unit. Results remain separately labelled
**independent** and **given NEMO's entry**. Measurement is offline from the
admitted passive stage, slow-forcing, and external-substep records; no
in-executable observer is allowed. No deck, selector, carried state,
stabiliser, sea-ice field, or `unmeasured_features` entry may change.

## Cited source order

The executing oracle build constructs the T-point bottom-drag rate from the
NOW bottom velocity over the one-cell halo at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/zdfdrg.f90:215-249`. The live
barotropic V correction then writes, in order, the face coefficient, bottom
baroclinic residual, and corrected slow forcing at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1420-1463`:

```
pCdU_v = 0.5 * (rCdU_bot(j,j+1) + rCdU_bot(j,j))
zv_i   = pvv(bottom,Kmm) - pvv_b(Kmm)
pv_RHSi += (r1_hv_0 / (1+r3v(Kmm))) * pCdU_v * zv_i
```

The OMT-4 RK3 deck executes the forward branch: compiled
`dynspg_ts.f90:1043-1044` forces `ln_bt_fw=.TRUE.` for RK3, so both bottom and
barotropic velocities are NOW (`dynspg_ts.f90:1443-1450`). The external-mode
entry copies the same carried `pvv_b(:,:,Kmm)` at `dynspg_ts.f90:355-366`.
The candidate replay must therefore use the production D114 resolved mesh,
the NOW stage-entry velocity, and the admitted carried V entry; substituting a
reconstructed mesh or a recomputed proxy for the oracle operand is forbidden.

## Frozen predictions and falsifiers

1. **R250-P1 — records and replay.** The inherited twin slow-forcing and
   external streams remain admitted and both claim labels agree. Replaying the
   candidate and oracle correction from their own operands reproduces their
   recorded post-drag outputs on all 35 owner faces within the 2e-10 floor;
   the oracle replay is bit-exact. Any record mismatch, label disagreement,
   non-passive state, or failed self-replay invalidates the split.
2. **R250-P2 — first unequal operand.** The stored/live V reciprocal is exact.
   The first unequal operand is the V-face bottom-drag coefficient on the 35
   northern-fold faces: NEMO evaluates the T-point rate through its live halo
   and then averages the two T points, while the current native helper edge-
   pads its last V face. Substituting the recorded coefficient alone closes
   the drag correction and post-drag slow forcing to the floor. Any earlier
   reciprocal difference, an exact coefficient, or a residual needed in
   addition **REFUTES** this prediction and becomes the retained result.
3. **R250-P3 — atomic stability outcomes.** Only after P2's own output closes,
   the source-exact coefficient arm is re-scored with the complete private
   unit. Outcome (a): kt=1 stage-1 V falls to rounding, the greater-than-10x
   fold growth disappears, and both labels complete kt=5, kt=8, and kt=10;
   then the complete unit proceeds to the full Decision-96 gates. Outcome
   (b): the 35-face seed closes but fold V still grows by more than 10x per
   stage; then round 244's missing in-loop/per-stage V association is a
   load-bearing growth mechanism, the unit stays private, and no other
   stabiliser is added.
4. **R250-P4 — landing.** A source-exact statement lands only if its own output
   closes, the atomic unit satisfies outcome (a), and OMT-1..4, independent
   rung 0, given-entry rung 10, GYRE, DINO, tanks, moved-row, month, citation,
   and test gates pass under Decision 96. Otherwise every package change is
   retracted and the round is HELD with the next first boundary named.
5. **R250-P5 — controls.** Claim-label coverage, production operand route,
   source order, record admission, self-replay, owner support, coefficient
   prediction, closure, atomic completion, and citation plants must each
   refuse when violated.

## Stop conditions

The 35-face correction is attributed only to the first unequal recorded
operand in written order. A coefficient arm may reuse the existing shared
bottom-drag builder but may not add a second builder, floor, clip, damping, or
card-specific stabiliser. Decision 116 remains pending separately and is not
re-emitted unless outcome (b) makes it load-bearing.

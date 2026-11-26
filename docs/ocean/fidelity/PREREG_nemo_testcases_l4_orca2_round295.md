# ORCA2 round 295 preregistration — substep-1 vector-V update operands

Date: 2026-10-16

Round 294 carried only NEMO's stored `hv_0` through the independent OMT-4
kt=1 stage-1 external solve.  It closed the substep-1 midpoint V depth and
named the same substep's vector-form V update as the next valued boundary:
35/8,589 active faces, maximum 2.7511187960820976e-4 m/s, all fold-local.
This round splits that update offline from the existing passive trace and
admitted NEMO streams.  It changes no production package, card,
configuration, carried state, or NEMO record.

The executing record build evaluates
`va_e = (vn_e + rDt_e * (zv_spg + zv_trd + zv_frc)) * ssvmask` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:666-678`, then
applies the V-point boundary association at `:721-739`.  The split uses the
written operand order `vn_e`, `rDt_e`, `zv_spg`, `zv_trd`, `zv_frc`,
`ssvmask`.  Candidate and oracle raw updates are compared at the same
pre-association point; the oracle raw update must reproduce the admitted
post-association record after the cited V fold operation before any operand
claim is allowed.

Every scientific number is labelled **independent OMT-4, kt=1 stage 1,
external substep 1**.  No given-entry value is mixed into the table.

## Frozen predictions and falsifiers

1. **R295-P1 — admission, precision, passivity.**  Round 294's immutable
   parent, the round-222 substep twins, and round-280 stage record admit;
   OMT-4 remains vector-form with the literal fold unit and completed-stage
   top-V exchange off; CPU production JIT and fp64/libm print.  The raw-`hv_0`
   observed and unobserved completed states are array-identical.  Any miss
   refuses scientific output.
2. **R295-P2 — calibrated program points.**  Replaying the written update
   from candidate operands is bit-exact with the candidate's passive raw
   `v_exit`.  Replaying it from NEMO operands and then applying NEMO's cited
   V association is bit-exact with the admitted NEMO `v_exit`.  Either miss
   is an instrument failure, not model debt.
3. **R295-P3 — operand split.**  Entry `vn_e`, `rDt_e`, `zv_trd`, and
   `zv_frc` are bit-exact.  `zv_spg` differs only by signed zero on the same
   35 fold faces, so it is the first **bit** difference but carries zero
   magnitude.  The compact surface `ssvmask` is the first **valued** operand,
   differing on 35 fold faces.  Any valued earlier operand or off-fold
   difference refutes this prediction and becomes the measured owner.
4. **R295-P4 — one-variable closure.**  Replacing only `ssvmask` by NEMO's
   raw surface V mask closes the pre-association V update from 35 unequal
   active faces to zero.  A surviving valued difference refutes single-
   statement closure and the cumulative closing set is retained instead.
5. **R295-P5 — no partial landing.**  Even if P4 confirms, raw `hv_0` and
   raw `ssvmask` stay inside the private complete fold unit because prior
   trajectory gates exposed compensating debt.  This measurement names the
   next member; it does not infer Decision-96 eligibility.
6. **R295-P6 — non-vacuity.**  Passivity, source order, parent closure,
   candidate replay, oracle replay, signed-zero classification, locality,
   single substitution, and one-bit plants must each refuse.  A green plant
   invalidates the measurement.

The outcome is measurement-only and **HELD**.  Decision 125's completed-stage
top-V exchange and the resolved stage mask remain off.  Decision 126 remains
pending and the shipped runoff setting is unchanged.  Sea ice and the shipped
card's `unmeasured_features` tuple are unchanged.

Independent review: operator, as required for this lane.

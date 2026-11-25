# ORCA2 round 287 preregistration — round-284 review completion

Date: 2026-10-16

This round performs only the four process corrections ordered by the
independent re-review of round 284.  It does not change model code, cards,
configuration, NEMO sources or records, sea ice, runoff, or any fidelity
predicate.  Decision 125 remains taken and Decision 126 remains pending.

## Frozen predictions and falsifiers

1. **R287-P1 — behavioural selector control.**  A small-grid execution of the
   live `_bc_pv_flux` np_CRV path must show that selecting statement A changes
   relative vorticity by exactly the supplied NEMO F-area ratio.  The test must
   fail under each independent known-answer mutation named by the reviewer:
   removing the `een_nemo_zeta` dispatch clause and replacing the call-site
   `nemo_f_area` operand with `None`.  Survival under either mutation refutes
   the control.
2. **R287-P2 — frozen preregistration.**  Lines 1–80 of the round-282
   preregistration must be byte-identical to frozen commit `9cc8d2ba3`,
   including its original dead-arm `dynvor.f90:648-652` text.  A dated
   correction appended after the frozen body must retract that anchor and cite
   the executing `vor_een` branch at `dynvor.f90:762-766`.  Any other mutation
   of the frozen body refutes the repair.
3. **R287-P3 — plant exit semantics.**  A round-284 plant that fires must keep
   exit 2 and `STATUS PLANT-FIRED`.  A plant that stays green must return a
   distinct non-plant exit code and `STATUS PLANT-STAYED-GREEN`; exit 2 or a
   firing marker for that condition refutes the gate correction.
4. **R287-P4 — appended citation corrections.**  The helper docstring must
   cite the executing EEN np_CRV span `dynvor.f90:762-766`.  The frozen
   round-284 preregistration remains unchanged through its original end and
   receives an appended dated correction replacing its prose line 694 with the
   current compiled `SUBROUTINE vor_een` line 692.  Editing either frozen body
   in place refutes the correction.

## Validation and stop rule

Run the two round-specific test modules in CPU fp64, demonstrate both R287-P1
mutations red, demonstrate both R287-P3 exit paths, then run the receipt
citation gate and focused fidelity tests.  Because no model file or card is
changed, trajectory and acquisition gates are off-claim.  Any failed frozen-
body comparison, surviving known-answer mutation, or conflated plant status
holds the round.

Independent review: operator, as ordered for this lane.

# Round 45 post-code self-review

Verdict: **HOLD — PREREGISTERED PARTNER SET REFUTED; NO PHYSICS LANDING**.

1. The scratch candidate is the round-44 source-literal placement, reproduced
   byte-for-byte against the rejected residual sidecar.  It exists only in
   clean measurement commits; the requested checkout retains shipped physics.
2. The first candidate-versus-shipped movement is measured, not inferred:
   kt=1 stages 1 and 2 are identical; stage 3 first moves U, then V.  Candidate
   stage-3 errors are `7.632783294297951e-17` and
   `9.71445146547012e-17`; shipped is `2.7478404751243857e-12` and
   `3.305560306813421e-12`.
3. Restoring the internal ZAD reconstruction reproduces the shipped residual
   artifact byte-for-byte but loses the kt=2 bar.  Removing the stage-folded
   term, implicit matrix carrier, or post-step remnant each reproduces the
   candidate residual artifact byte-for-byte and leaves all 55 Rule-12
   violations.  None meets both frozen partner criteria.
4. This agrees with execution predicates: GYRE resolves `ln_zad_Aimp=F`
   (`GYRE_OMIP_L2_P3_SM_R41ADVSP/EXP00/ocean.output:535`); legoESM gates the
   folded share at `ocean_model_latlon_cgrid.py:5927-5937`, its implicit
   carrier at `:6873-6890`, and the old post-step block also excludes RK3 at
   `:6901-6923`.  The executing internal reconstruction is
   `ocean_pe_latlon_cgrid.py:4749-4770`.
5. NEMO's demanded placement remains source-proven: velocity-form `wzv` is
   called at compiled
   `GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:327-333`,
   `dyn_adv` at `:466-472`, ZAD dispatches at
   `GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90:152-176`, and the
   update is
   `GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynzad.f90:105-137`.  Given
   NEMO inputs, the existing gate remains
   DISCHARGED at 0 unequal U/V cells.
6. The candidate first-over-bar is kt=3 T/S/U/V/ssh, but the next source-order
   owner cannot be discriminated from the available kt=1-only internal dumps.
   `round45/run.sh` freezes the missing kt=2 acquisition; no owner is guessed.
7. Because the GYRE prerequisite failed, LOCK_EXCHANGE and OVERFLOW promotion,
   DINO replay, and ORCA2 are not claimed.  No configuration, tolerance,
   stabilizer, NEMO source/record, or pending slope-scope decision was changed.
8. The classifier is fail-closed and its digest plant exits 2.  Focused tests:
   24 passed.  Worktree-stamp ratchets: 18 passed.  `DISCHARGED` is used only
   for the zero-unequal given-input ZAD rows.

ASKED: global versus NEMO-identity-only slope association remains pending and
identity-only.  UNASKED: empty.

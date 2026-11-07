# Preregistration: GYRE stage-3 ww/ZAD discriminator, round 44

Date: 2026-09-11. Frozen before the first discriminator execution at
`92c00497f48f9cb065eb7aff1a6a7b60b7d56466`.

## Question and source boundary

The compiled GYRE program constructs stage-3 `ww` from `uu/vv(Kmm)` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:327-333`, then
calls vector-form `dyn_adv(kstp,Kmm,Kmm,...)` at `:466-472`.  Its active
`dyn_zad` arm consumes that `ww` at `dynzad.f90:105-137`; resolved
`ln_vortex_force=.false.` selects `ww` without `wsd` at `:107-115`.

legoESM constructs the literal stage transport/`ww` as `_g2` before the
stage-3 momentum call (`ocean_model_latlon_cgrid.py:6572-6610`), but the
momentum call does not pass `_g2[2]`: `tendencies()` reconstructs a separate
ZAD `ww` at `ocean_pe_latlon_cgrid.py:4783-4811`.  This is the boundary to
measure, not yet an attribution.

## Frozen outcomes

The committed round-21 stage-3 `ww` and round-41 split record are immutable
inputs.  Score wet cells exactly and at the existing `1e-15` relative bar;
score effective `wsd` against zero.  Print the model ZAD U/V maxima beside
NEMO's `2.0616936887777917e-16` / `2.4830138308915274e-16` maxima.

1. Model ZAD approximately zero while the stage-program `ww` is at bar:
   **MISSING_OR_MISPLACED_STAGE3_ZAD**.  Confirm the GYRE identity resolves
   vector-invariant C2 plus `vertical_momentum_scheme=nemo_advective`, and
   that the stage-3 call consumes the stage-3 `ww` before changing code.
   Mechanically, "approximately zero" means each model-face maximum is at
   most `1e-3` of NEMO's corresponding recorded ZAD maximum; this ratio is
   frozen before the ZAD exposure is run.
2. Stage-program `ww` differs: **WZV_FIRST**.  Walk compiled
   `sshwzv.f90:271-298`—`div_hor`, QCO continuity stretch, then bottom-up
   recurrence—using the round-21 and round-41 operands to the first non-bit
   statement.  Do not change ZAD first.
3. Stage-program `ww`, effective `wsd`, and model ZAD are all exact:
   **NEXT_AFTER_ZAD**.  Point next to the post-ZAD accumulator/association;
   do not infer an owner by subtraction.

Any other combination is **UNRESOLVED** and stops implementation.  Plants for
`ww`, `wsd`, ZAD exposure, commit stamp, and resolved-card arm must exit
nonzero.

## Pre-code self-review

- Existing machinery reused: `nemo_testcase_l2_gyre_stage_ww_gate.py`, the
  stage-3 transport exposure, and the source-order operator exposure.
- The new ZAD exposure returns the already-computed production
  `diag_vertadv_u/v`; it does not re-evaluate or inject an oracle output.
- No physics/default/configuration choice, tolerance, NEMO source, or record
  changes.  The pending slope-association scope remains ASKED.
- A fix is eligible only if its changed operator is bit-exact given NEMO
  inputs on every executing card under Rule 12.

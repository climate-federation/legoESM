# Preregistration — round 195 / VORTEX round 11

Frozen BEFORE any measurement of this round.  Lane tip `f869f6754`
(round 194's HELD landing).  Case `VORTEX_VEC-zco`.  Evidence root
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round195/`.

## What round 194 left

The held patch
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l1_vortex_round194_literal_wzv_held.patch`
states TWO config fields on `VORTEX_VEC-zco` at once:
`wzv_call2_evaluation="nemo_literal"` and
`nemo_stage_momentum_wzv_split=True`.  Under it the stage-2/3 ZAD
accumulator becomes exact (2.7e-20), kt=2 U/V improve 270x/316x
(3.3693297863401916e-06 -> 1.2462615108344623e-08,
3.337023651e-06 -> 1.0562745593834697e-08), and the certified 50-row
trajectory comparison FAILS: kt=5 S goes AT-BAR -> DEBT and cells
worsen past the two-ULP ratchet.

## Claim under test (read off the code, to be measured)

`nemo_stage_momentum_wzv_executes` (ocean_model_latlon_cgrid.py) returns
False unless `wzv_call2_evaluation == "nemo_literal"`, so the momentum
split CANNOT be selected alone.  `wzv_call2_evaluation="nemo_literal"`
also rewrites the TRACER arm of the step (the Kmm velocity cycle, the
tracer face thickness, the tracer mass fluxes and the tracer vertical
velocity).  The held candidate is therefore TWO statements, not one.

## Predictions (falsifiers named)

* **P1 — the second difference is the tracer-side literal call-2, not the
  momentum split.**  Arm B = the same card with
  `wzv_call2_evaluation="nemo_literal"` and
  `nemo_stage_momentum_wzv_split=False` reproduces the kt=5 S
  AT-BAR -> DEBT change and the two-ULP cellwise FAIL, while leaving kt=2
  U/V at (or within 1% of) the production 3.369e-06 / 3.337e-06.
  REFUTED if arm B's kt=5 S stays AT-BAR and its two-ULP comparison
  PASSES, or if arm B already moves kt=2 U/V by more than 1%.
* **P2 — the two statements are separable and additive.**  Arm C (the
  held candidate) minus arm B is the momentum split alone: C's kt=2 U/V
  improvement is present in C and absent in B.  REFUTED if B shows the
  kt=2 improvement.
* **P3 — the first non-bit producer of the remaining kt=2 U/V residual
  under the candidate is the BAROTROPIC correction, not the stage RHS and
  not the pre-correction velocity update.**  With NEMO's recorded
  barotropic output substituted (`stage_barotropic_output_override`), the
  stage-2 and stage-3 output velocities score AT-BAR 1.11e-16 / 2.22e-16
  normalized (round 194's own calibration, candidate arm); with the
  substitution REMOVED, the same stage outputs degrade by at least two
  orders of magnitude.  REFUTED if removing the substitution leaves the
  stage outputs within 10x of the substituted numbers (then the owner is
  upstream of the barotropic solve), or if NEMO's recorded pre-correction
  `update_u/update_v` already disagrees by more than 1e-12 while the
  recorded-barotropic output is at the floor.
* **P4 — nothing lands.**  No production card selection changes this
  round unless a cited statement passes the UNCHANGED two-ULP trajectory
  gate.  REFUTED only by a green gate.

## Plants (non-vacuity)

* The stage walk's existing `--plant s{2,3}.{op}.{u,v}` row plant must
  fire and exit non-zero on the arm actually used.
* The trajectory comparison's own `--plant` must fire.
* The round's citation gate must fire its planted-shift self-test.

## Method

1. Arm B ladder: apply the held patch with `False` in place of `True`,
   commit it on a scratch commit (the gate refuses a dirty tree), run
   `nemo_testcase_phase3_trajectory_gate.py --case VORTEX_VEC-zco
   --max-step 10 --continue-after-first`, compare against the
   round-191 certified reference with the unchanged two-ULP gate, then
   revert.
2. Arm C is round 194's `phase3/round194/vortex_vec_ladder.json`
   (clean commit `858822b5f`), re-used, not re-run.
3. Post-ZAD boundary walk on the candidate arm, EXTENDING
   `nemo_testcase_l1_vortex_round193_stage_terms.py` (no new script for
   the same question): add NEMO's recorded `update_u/update_v` and
   `out_u/out_v` boundaries from the round-192 stage-term record, and the
   one-variable toggle of `stage_barotropic_output_override`.
4. Any NEMO operand the round-192 record does not carry => build the
   acquisition `run.sh` and STOP with ACQUISITION_NEEDED.  No guessing.
5. Any option the VORTEX deck does not pin that would have to be chosen
   => DECISION_NEEDED (Decision 75).

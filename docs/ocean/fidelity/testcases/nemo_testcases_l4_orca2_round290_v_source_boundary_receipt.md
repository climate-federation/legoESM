# ORCA2 round 290 — stage-1 V source-row boundary split

Date: 2026-10-16

Status: **HELD**.  At the independent OMT-4 kt=1 stage-1 source row, the raw
vector-form V update has zero valued error against NEMO.  The following
barotropic correction creates the full 1,440/5,400-value debt, maximum
0.016207369589460653 m/s.  That correction is the first non-bit statement;
its operands are OPEN for the next frozen round.  No physics or configuration
change lands here.

Every scientific number below is **independent OMT-4, kt=1 stage 1**.  No
given-entry number is mixed into the table.

## Provenance and instrument

- Lane base: `ee4678a35`.
- Frozen preregistration:
  `PREREG_nemo_testcases_l4_orca2_round290.md`, commit `8be2484e0`.
- Committed measurement gate: `4c5efe847`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round290/`.
- Admitted source record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round280/acquisition/orca2_omt4_stage1_source_10step_np2`.
- Baseline frame record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2`.
- Execution: CPU, production JIT, fp64/x64, libm.
- Instrument: the existing passive live-operand trace exposes
  `stage_raw_velocities` before the correction and `stage_outputs` after it;
  the admitted record supplies NEMO `pre_v` and `cor_v` at those same program
  points.  All completed traced-state leaves are array-identical to the
  ordinary run.  No executable observer or NEMO acquisition was added.
- Search before build found and reused round 281's rank-complete reader and
  exact scorer, round 283's native-V adapter, and round 289's passive trace.
  No second parser, runner, stage update, or correction implementation was
  created.

`measurement.json` has SHA256
`b8233aa6a3012bae1ac4e2cf2478f355f167046d0867076e9fbbc3a0da7e13f9`;
`classification.json` has SHA256
`11a98258b8cbf50d926e7831797b7e4aedb98b6af7e3af3fcfb3d6014b995900`.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R290-P1 | **CONFIRMED** | The round-280 record re-admits as `PASS_R280_OMT4_STAGE1_ADMISSION`; every traced completed-state leaf is array-identical to the ordinary run. |
| R290-P2 | **CONFIRMED** | Raw V versus NEMO `pre_v` has maximum and RMS exactly 0.0 m/s; its 323 differences are signed zeros only. |
| R290-P3 | **CONFIRMED** | Corrected V reproduces round 289 exactly: 1,440/5,400 unequal, maximum 0.016207369589460653 m/s, RMS 0.0035899206923965866 m/s. |
| R290-P4 | **CONFIRMED** | Candidate versus NEMO correction increments differ on 1,319/5,400 values with the same valued maximum/RMS; adding the candidate increment to NEMO's raw row reproduces the corrected-boundary census and magnitude, while candidate recomposition is 0/5,400 unequal. |
| R290-P5 | **CONFIRMED** | Passivity, program-order, raw-boundary, correction-calibration and one-bit plants each print `STATUS PLANT-FIRED` and exit 2. |

The 323 signed-zero differences at the raw boundary are registered, not called
bit-exact.  The campaign's valued floor is `2e-10`; both maximum and RMS are
exactly zero there.  The source-ordered valued boundary is therefore the next
statement, not the raw vector update.

## Compiled source citation

The ORCA2 record build executes the vector-form update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.
It then diagnoses and applies the stage's barotropic U/V correction at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:521-528`.
The matched record and passive trace show that the V source row is at the
valued floor before the second statement and carries the complete valued debt
after it.  This is a statement boundary, not yet a leaf attribution.

## Validation

- The round-290 direct tests and their round-281/283/289 neighbour tests pass.
- This receipt's citation gate and the cumulative default-receipt gate pass;
  a rigid two-line shift of the new correction citation fires.
- The required fidelity-battery and independent-review results are recorded
  in the final validation commit for this round.
- No package or card file changes exist, so ORCA2 trajectory, GYRE, DINO,
  tank and acquisition gates are off-claim.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange and the resolved stage mask
  remain off.
- Decision 126 remains pending; shipped runoff is unchanged and the decision
  is not re-raised.
- Sea ice and its `unmeasured_features` tuple remain unchanged.

## OPEN

1. Split NEMO's barotropic V correction on source row 146 in compiled order:
   carried external-mode V, the live depth-weighted vertical sum, and
   `r1_hv_0`, then the corrected update's addition/rounding order.  Reuse the
   admitted passive barotropic targets and geometry; acquire only an operand
   that is genuinely absent.
2. Keep the resolved stage mask and completed-stage top-V exchange held; do
   not re-score their pair until this correction owner closes.
3. Leave shipped runoff unchanged while Decision 126 remains pending.

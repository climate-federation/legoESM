# ORCA2 round 292 — carried external-mode V target producer

Date: 2026-10-16

Status: **HELD**.  The independent OMT-4 kt=1 stage-1 external-mode walk
names the substep-1 live V-face depth on NEMO's fold-image row as the first
valued producer of round 291's carried V-target debt.  It differs on 13/8,589
active rank-slab faces, maximum 719.0 m; the interior is exact, 0/8,489.
The statement is part of the already-held vector-form fold unit, so no physics
or configuration change lands in this round.

Every scientific value below is **independent OMT-4, kt=1 stage 1**.  No
given-entry value is mixed into the result.

## Provenance and instrument

- Lane base: `cd39288f3`.
- Frozen preregistration:
  `PREREG_nemo_testcases_l4_orca2_round292.md`, commit `3c6af12e6`.
- Committed measurement gate: `f89c699e9`; mask-orientation and all-face
  calibration corrections: `c43de25cf`, `36b20e78e`; frozen-refutation
  classifier correction: `477688578`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round292/`.
- Admitted substep twins: round 222 `orca2_omt4_frames_10step_{a,b}_np2`.
- Admitted carried-target record: round 280
  `orca2_omt4_stage1_source_10step_np2`.
- Execution: CPU, production JIT, fp64/x64, libm.
- Instrument: the existing passive barotropic substep trace and live-stage
  target trace.  Both completed states are array-identical to the ordinary
  production step.  The final substep V is bit-exact against the live carried
  target on both the candidate and oracle sides.
- Search before build found and reused round 235's source registry and stream
  admissions, round 243's canonical-support lesson, round 281's rank-complete
  reader/exact scorer, and the production trace.  No parser, solver, model
  step, or executable observer was duplicated.

`measurement.json` SHA256:
`452ff1a1944c17f5b4b28ce3940d87c18c33dc2e8bb7e4b452b7e49028461463`.
`classification.json` SHA256:
`2f204c83d85161e4377410492fcbf90ed1324122e7469b86d6456905bf06b8e3`.

## Source-ordered result

At substep 1, slow forcing, entry V, midpoint V, and midpoint SSH are at the
campaign floor.  The next statement, live V-face depth, is the first valued
row:

| region | unequal / support | maximum | RMS |
|---|---:|---:|---:|
| complete rank slab | 13 / 8,589 | 719.0 m | 16.385495319441215 m |
| fold band | 13 / 100 | 719.0 m | 151.8556222205816 m |
| interior | 0 / 8,489 | 0.0 m | 0.0 m |

The trace's final V and round 291's live target are bit-exact at the same
program point, as are NEMO's final-substep and carried-target arrays.  Thus
the face-depth row is upstream of the carried 70/180-face global target debt,
not an unrelated diagnostic difference.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R292-P1 | **CONFIRMED** | Every inherited stream admits; both diagnostic states are passive; CPU/fp64/libm is printed. |
| R292-P2 | **REFUTED** | Rank 0 is not an even half of the global census: 34/90 faces, max 0.012404972646641737 m/s, not 35/90 and 0.01620736958946065 m/s.  Direct candidate and oracle program-point controls are each bit-exact, so the discrepancy is rank partitioning rather than misalignment. |
| R292-P3 | **CONFIRMED** | The first valued intermediate is substep-1 `face_depth_v_mid`; every preregistered predecessor is at the floor. |
| R292-P4 | **CONFIRMED** | Its interior is bit-exact; all 13 valued faces are in the three-row fold band. |
| R292-P5 | **CONFIRMED** | The face-depth boundary precedes the first valued V exit and the non-bit final carried target. |
| R292-P6 | **CONFIRMED** | Passivity, source-order, frozen-calibration, first-boundary, locality, endpoint-causality, and one-bit plants each print `STATUS PLANT-FIRED` and exit 2. |

### Instrument corrections retained

The first measurement attempt refused before producing a number because the
`mesh_mask` wet support was still in Fortran `(i,j)` order while the admitted
streams use model `(j,i)` order.  The second attempt produced the scientific
table but its classifier correctly refused the preregistered 35/90 rank-half
assumption.  The final classifier retains that prediction as REFUTED and uses
the two bit-exact direct program-point controls; neither correction changes a
model, card, oracle record, or measured substep value.

## Compiled source citation

The ORCA2 record build computes the first valued live V-face depth at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:517-519`.
The surrounding compiled program has already resolved the vector-form deck;
this depth statement remains outside the vector/flux update branch.  The
record-backed table proves it executes and is the first valued boundary on
this card.

## Validation

- The round citation gate and the cumulative default-receipt gate pass with
  zero unmapped citations; shifting the new compiled span makes the planted
  gate refuse with `SYMBOL-NOT-AT-LINE`.  Artifact SHA256 values are
  `a6cd90fe3a5ca1d0c7c3112ead749bdb615f85d8437b9c58eda275b8312daadb`,
  `ee58b40a9adc76a53cc34a7b247225aba80f5523cf4221a40f081b83c394d769`,
  and `e193217a522c8b08e1edac59be8b5616eb8845316a573ab4c402071b1620a9cf`.
- Round-290/291/292 direct tests pass 7/7.  Log SHA256:
  `92eda0cec749b4308df18e4c0fc15b79ec5ecb24670263faed0dd5f873a671b5`.
- The single permitted `tests/ocean/fidelity -n 12` battery reached 90% and
  then all real pytest processes disappeared, reproducing the branch's known
  late worker-loss pattern.  Seven registered branch-tip reds printed before
  the loss: GYRE round-129 record hash, two round-33 mutation anchors, OMT-1
  module-edge ratchet, round-228 citation map, dirty-escape scope, and
  worktree stamping.  All three round-292 tests passed; no new failure
  printed.  The battery was not repeated.  Log SHA256:
  `488c6af1dfba80b5b6a54c3219c619b067b958f174daff6e54ff492665582981`.
- Independent review unavailable in-sandbox: `codex exec --sandbox
  read-only` exited 1 because its in-process app-server client attempted a
  write in the read-only sandbox.  Log SHA256:
  `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
- No package or card file changed, so ORCA2 trajectory, GYRE, DINO, tank, and
  NEMO acquisition gates are off-claim.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange, the resolved stage mask, and
  the vector-form fold unit remain off.
- Decision 126 remains pending; shipped runoff is unchanged and the decision
  is not re-raised.
- Sea ice and the shipped card's `unmeasured_features` tuple are unchanged.

## OPEN

1. The source-exact fold-image V-face depth is a known member of the held
   vector-form fold unit.  Keep it out of production; next split its live-depth
   operands (`hv_0`, the two T-row sea levels and areas, `r1_e1e2v`, and
   `ssvmask`) only to identify the cancelling partner exposed when the whole
   unit is enabled.  No partial face-depth landing is allowed.
2. Keep the resolved stage mask and completed-stage top-V exchange held; do
   not re-score their unit until the face-depth partner is named.
3. Leave shipped runoff unchanged while Decision 126 remains pending.

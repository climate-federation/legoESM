# ORCA2 round 289 — top-V source/support/sign split

Date: 2026-10-16

Status: **HELD**.  NEMO's completed-stage V association is bit-exact when
given NEMO's pre-association row.  legoESM's extra mask multiplication is also
a bitwise null.  The association moves the round-288 pair away because its
copied source row is already non-bit on 1,440 values, max
0.016207369589460653 m/s; substituting NEMO's source row makes the associated
target row bit-exact.  The next walk therefore starts before `lbc_lnk`, not in
the fold rule itself.

Every number is **independent OMT-4, kt=1 stage 1**.  No given-entry and
independent values are mixed.

## Provenance and instrument

- Lane base: `05ef40eb1`.
- Frozen preregistration: `PREREG_nemo_testcases_l4_orca2_round289.md`, commit
  `5ae83034a`.
- Committed measurement gate: `caaa3ec4a`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round289/`.
- Admitted source record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round280/acquisition/orca2_omt4_stage1_source_10step_np2`.
- Baseline frame record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2`.
- Execution: CPU, production JIT, fp64/x64, libm.
- Instrument: the existing passive live-operand trace plus an offline
  association replay.  Its ordinary and traced completed states are
  array-identical in every field.  No executable observer, NEMO acquisition,
  or package candidate was added.
- Search before build: round 281's rank-complete reader and exact scorer,
  round 249's passive hook, and the production fold helper were reused.  No
  second parser, stage runner, or fold implementation was created.

The resolved and compact 3-D V masks differ on 1,319 slots, all on internal
row 148, the association's target row.  They are identical on internal row
147, the copied source.  This proves that the current-tree source row is also
the resolved-mask candidate's source row; no historical checkout or inferred
candidate value is used.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R289-P1 | **CONFIRMED** | The round-280 admission passes; the trace is passive; the matched source/target shapes and target-only 1,319-cell mask delta are unchanged. |
| R289-P2 | **CONFIRMED** | Applying source row 146, the recorded longitude permutation and sign -1 to NEMO `cor_v` reproduces all 799,200 values of NEMO `final_v` bit-for-bit. |
| R289-P3 | **REFUTED — SOURCE ROW OWNS** | The source row is 1,440/5,400 unequal, max 0.016207369589460653 m/s, RMS 0.0035899206923965866 m/s.  The full copied row has the same 1,440/max debt; substituting NEMO's source closes 0/5,400.  Restricting the copy to one half does not close it. |
| R289-P4 | **CONFIRMED** | Removing only the extra mask multiplication changes 0/5,400 bits; it is exonerated at this boundary. |
| R289-P5 | **CONFIRMED** | The one-bit, wrong-sign, wrong-row and empty-support plants each print `STATUS PLANT-FIRED` and exit 2. |

The wrong-sign control differs on 5,279/5,400 values with maximum
0.03971041303649354 m/s.  The wrong-row control differs on 1,515/5,400 with
maximum 0.02812351302978503 m/s.  These controls prove the exact replay is
bound to the compiled sign and source row rather than merely comparing zeros.

The full post-association stage V remains 65,368/799,200 unequal, max
0.03361800804421261 m/s and RMS 0.000883843807368417 m/s.  The round-288
stage-1 RMS-away result is therefore reconciled: the fold copy faithfully
propagates a pre-existing 1,440-value source-row error onto its target.  A
source-exact-looking association is not an exact model boundary when its live
source is already wrong.

`measurement.json` has SHA256
`29ac6509fbb42ac0543694e22d826b1a0b9ff74b103023d45bf89b995f7c9991`;
`classification.json` has SHA256
`78217d4bfc380fb30934ddaace7ee3d368ff843b3e17eb122018cbdaeb4bd085`.

## Compiled source citation

The vector-form stage update is
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.
After all stage physics, NEMO calls `lbc_lnk` for U/V with sign -1 and T/S
with sign +1 at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820`.
The compiled T-pivot V case writes the mirrored source row using the supplied
sign at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/lbcnfd.f90:684-717`.
The record-backed replay proves those statements produce NEMO's target row
bit-for-bit; no configuration branch is inferred from prose.

## Validation

- Round-289 plus its round-281/288 neighbour tests pass 17/17.  The focused
  log SHA256 is
  `86fb5c197114b621d6e794fe0cf74ba0740e44c761e645d6f019b96acd8d8427`.
- The one permitted `tests/ocean/fidelity -n 12` battery reached 96%, then
  stopped progressing after its Python worker processes disappeared and was
  terminated.  It printed the same eight registered branch-tip reds as round
  288: the GYRE round-129 record hash, two round-33 source anchors, OMT-1
  module-edge ratchet, round-228 citation entry, dirty-escape scope, worktree
  stamping, and SI3 scalar-math provenance.  The new round-289 test passed;
  no round-289 failure printed.  The retained log SHA256 is
  `4747616695578017a7d65f43f6ea54e4666e362c8b689cafc36eee20b1b62094`.
- Independent review unavailable in-sandbox: nested `codex exec --sandbox
  read-only` exited 1 because its in-process app-server client attempted a
  write in the read-only sandbox.  The retained review log SHA256 is
  `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
- No package or card file changes in this round, so trajectory, GYRE, DINO,
  tank and acquisition gates are off-claim.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange remains off.
- Decision 126 remains pending; shipped runoff is unchanged and the decision
  is not re-raised.
- Sea ice and its `unmeasured_features` tuple remain unchanged.

## OPEN

1. At the matched pre-association kt=1 stage-1 boundary, split the 1,440-value
   source-row debt between the vector-form V update and the barotropic
   correction using the existing passive `stage_raw_velocities` and recorded
   `pre_v/cor_v` streams.  The first valued row owns the next walk.
2. Keep both the resolved stage mask and completed-stage top-V exchange held;
   do not score their pair again until the source-row owner is closed.
3. Leave shipped runoff unchanged while Decision 126 remains pending.

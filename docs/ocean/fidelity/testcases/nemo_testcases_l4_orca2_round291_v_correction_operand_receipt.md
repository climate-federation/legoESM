# ORCA2 round 291 — stage-1 V correction operand split

Date: 2026-10-16

Status: **HELD**.  At the independent OMT-4 kt=1 stage-1 fold source row,
the first valued input to NEMO's barotropic correction is its carried
external-mode V target.  The candidate target differs on 70/180 faces,
maximum 0.01620736958946065 m/s.  Substituting only NEMO's recorded target
makes the corrected 3-D source row valued-exact: the remaining 121/5,400
differences are signed zeros with maximum and RMS exactly 0.0.  The target's
producer inside the split-explicit solver is OPEN; no physics or configuration
change lands here.

Every scientific number below is **independent OMT-4, kt=1 stage 1**.  No
given-entry number is mixed into the table.

## Provenance and instrument

- Lane base: `f3b2fcd32`.
- Frozen preregistration:
  `PREREG_nemo_testcases_l4_orca2_round291.md`, commit `2a60012c0`.
- Committed measurement gate: `9275dac30`; result-classifier correction:
  `f639502b9`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round291/`.
- Admitted source record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round280/acquisition/orca2_omt4_stage1_source_10step_np2`.
- Baseline frame record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2`.
- Execution: CPU, production JIT, fp64/x64, libm.
- Instrument: round 290's passive live trace supplies the candidate raw and
  corrected V, external-mode target, reference face thickness and reciprocal;
  the admitted record supplies NEMO `pre_v`, `ext_vb`, and `cor_v`; the card's
  raw NEMO face bundle supplies the reference `e3v_0`, `hv_0`, and `vmask`.
  Both the candidate and oracle source-ordered correction replays reproduce
  their respective outputs exactly.  Every completed traced-state leaf is
  array-identical to the ordinary run.
- Search before build reused round 281's rank-complete reader and exact scorer,
  round 283's native-V adapter, the shared ascending sum, and the existing
  passive trace.  No parser, model runner, correction kernel, or executable
  observer was duplicated.

## Source-ordered split

| row | unequal / support | maximum | disposition |
|---|---:|---:|---|
| carried external V target | 70 / 180 | 0.01620736958946065 m/s | **first valued input** |
| reconstructed reference `e3v` | 4,081 / 5,400 | 499.9785217898161 m | non-bit but null under the zero raw V column |
| depth-weighted vertical sum | 0 / 180 | 0 | exact |
| stored `r1_hv_0` | 0 / 180 | 0 | exact |
| weighted sum | 0 / 180 | 0 | exact |
| scalar correction | 70 / 180 | 0.01620736958946065 m/s | carries the target debt |
| stage `vmask` | 0 / 5,400 | 0 | exact; preregistered mask owner refuted |
| corrected V before substitution | 1,440 / 5,400 | 0.016207369589460653 m/s | reproduces round 290 |
| corrected V after target substitution | 121 / 5,400 | 0 m/s | signed zeros only; valued-exact |

The reconstructed face thickness is a real non-bit operand, but it has no
leverage at this boundary: the raw stage V column is exactly zero, so both
depth-weighted sums are bit-exact.  This is why the attribution follows the
source-ordered intermediate and substitution, not the largest raw operand
difference.

`measurement.json` SHA256:
`0923ec87fdc066bbb37dd02c33a4e1cac1a9383412c1e6c49f3c0291ff3ea6a5`.
`classification.json` SHA256:
`88ba47e44f890651f46391beb03fa5d6d95e0bce14aa95f28ff6851d8e3b78fe`.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R291-P1 | **CONFIRMED** | The record re-admits, CPU/fp64/libm is printed, and every completed-state leaf is passive. |
| R291-P2 | **REFUTED — TARGET OWNS** | The target is 70/180 unequal, max 0.01620736958946065 m/s; its one-variable substitution closes the valued 3-D boundary. |
| R291-P3 | **REFUTED IN PART** | Reference `e3v` is 4,081/5,400 unequal, but the aggregate depth sum and stored reciprocal are exact and the thickness difference is a null. |
| R291-P4 | **REFUTED — DOWNSTREAM OF TARGET** | The scalar correction differs by exactly the target maximum; both candidate and oracle self-replays are exact. |
| R291-P5 | **REFUTED** | The source-row stage mask is bit-exact, 0/5,400 unequal; it does not own this boundary. |
| R291-P6 | **CONFIRMED** | All eleven plants print `STATUS PLANT-FIRED` and exit 2. |

## Compiled source citation

The correction consumes `vv_b(:,:,Kaa)`, the reference-thickness vertical
sum, and stored `r1_hv_0` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:521-528`.
The executing vector-form branch assigns that target directly from the final
split-explicit V state, `pvv_b(:,:,Kaa) = va_e`, at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:857-860`.
The measurement therefore names a carried target boundary, not yet the
upstream substep leaf that made `va_e` non-bit.

## Validation

- **TO_FILL focused tests.**
- **TO_FILL citation gates and plant.**
- **TO_FILL ocean-fidelity battery.**
- Independent review: operator.  No in-sandbox nested review was attempted,
  per the lane's current operator-review rule.
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

1. Split the carried `va_e` target in `dynspg_ts` at the final external
   substep: incoming V, slow forcing, Coriolis/pressure-gradient update,
   live-depth division/transport-to-velocity conversion, and the in-loop/final
   fold associations, using the existing admitted barotropic substep streams.
   The first valued intermediate owns the next walk.
2. Keep both the resolved stage mask and completed-stage top-V exchange held;
   do not re-score their pair until the external target owner closes.
3. Leave shipped runoff unchanged while Decision 126 remains pending.

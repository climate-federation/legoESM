# ORCA2 round 283 — stage-1 vector-V operand split

Date: 2026-10-15

Status: **HELD**.  The independent OMT-4 kt=1 stage-1 vector-V update is now
source-exactly split.  Entry V, the completed live RHS, and the stage time are
bit-exact against NEMO.  The first non-bit operand is legoESM's compact 3-D V
mask: 1,319/799,200 mask slots differ from NEMO, and replacing only that mask
closes the raw stage output from 1,319 unequal active cells to 0/415,175.

This round changes validation and receipt code only.  No package, card, deck,
NEMO source, runoff, or sea-ice setting changed.  Every scientific number is
labelled **independent OMT-4**.

## Provenance

- Lane base: `27b64c691`.
- Frozen preregistration:
  `PREREG_nemo_testcases_l4_orca2_round283.md`, commit `5ff81b6e6`.
- Measurement instrument commits: `df656796f`, `4d0b00e70`, `5aacdcc16`,
  `1a1099315`.
- Admitted record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round280/acquisition/orca2_omt4_stage1_source_10step_np2`.
- Machine evidence:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round283/measurement.json`,
  SHA256 `66b321da08e81b54791d254f5c8fb428e522776cf98126aa42e66126bd08d19a`.
- Raw calibrated measurement SHA256:
  `53ec6fe483b5394e17f1da1d0f6dcbca5ba2ddf42dcbe7ed7e5c9c6c703ced28`.

The record re-admits as `PASS_R280_OMT4_STAGE1_ADMISSION`; all inherited 80
stage frames and both kt=10 restarts remain byte-identical to the admitted
OMT-4 baseline.  The run uses CPU JIT, fp64, and the libm precision policy.

## Calibrated source replay

The existing live-operand trace is paired with an ordinary run.  Every one of
its 51 completed state leaves is array-identical to the ordinary state.  The
shared pure stage-update function, driven by the trace's own entry, RHS, time,
and mask, reproduces the trace's raw stage-1 V output at 0/799,200 unequal.
No new in-executable observer was added.

NEMO sets `rDt = rn_Dt/3` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:136` and writes
`(vv(Kbb) + rDt * vv(Krhs)) * vmask` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.
The source-ordered split is therefore entry, completed RHS, time, mask, then
the written arithmetic association.

| Operand / boundary | Unequal / support | Max abs | RMS | Argmax |
|---|---:|---:|---:|---|
| entry V, active support | 0 / 415,175 | 0 | 0 | — |
| completed live RHS, active support | 0 / 415,175 | 0 | 0 | — |
| `rDt`, 3600 s vs 3600 s | 0 / 1 | 0 | 0 | — |
| 3-D V mask, full array | 1,319 / 799,200 | 1 | 4.0625120312442156e-2 | `[147,29,0]` |
| candidate raw V vs NEMO | 1,319 / 415,175 | 1.1299245240990687e-2 m/s | 1.7813208147988798e-4 m/s | `[147,133,27]` |
| all-NEMO operands through shared update | 0 / 415,175 | 0 | 0 | — |

The cumulative entry, RHS, and time substitutions leave the raw-output row
unchanged at exactly 1,319 unequal cells.  The mask substitution alone closes
it bit-for-bit.  The written arithmetic order is therefore exonerated.

## First non-bit statement

NEMO constructs `vmask` from the two adjacent T masks, applies the configured
bottom-level masking, and then applies the V-point lateral-boundary association
at `ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:211`,
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:225`, and
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:232`.
The executing update consumes that associated 3-D mask at the cited
`stprk3_stg.f90:477-480` statement.  legoESM instead sources this stage mask
from its compact `compute_face_masks_3d` route.  That source is the first
non-bit statement; its 1,319-cell census exactly equals the raw-update census.

This is a statement attribution, not a landing.  The resolved-mask production
arm has not yet been scored singly on the required trajectories, and the held
vector-fold unit already has a registered live-W-thickness interaction.  The
mask source therefore remains unchanged until its single-statement D96 census
separates it from that held unit.

## Frozen predictions

| Prediction | Verdict | Measurement |
|---|---|---|
| R283-P1 passive calibration | **CONFIRMED** | 51/51 completed leaves array-identical; self-replay 0/799,200 unequal |
| R283-P2 entry remains exact | **CONFIRMED** | 0/415,175 active cells unequal |
| R283-P3 completed RHS owns first valued debt | **REFUTED** | completed RHS is 0/415,175 unequal |
| R283-P4 RHS substitution moves toward NEMO | **REFUTED** | raw row is unchanged at 1,319 cells and identical RMS |
| R283-P5 source order isolates the first owner | **CONFIRMED** | entry/RHS/time null; mask closes 1,319 -> 0 |

Failed predictions remain in the ledger.

## Controls and scope

All five plants fire: passivity, self-replay, source order, owner-prefix
non-vacuity, and one-bit detection.  The focused round-283 tests pass 6/6.
The citation gate and its shifted-line plant are run against this receipt before
closeout; the default cumulative receipt is also required to stay green.

Independent review: **operator**.  The lane instruction assigns independent
review of round code to the operator; no nested reviewer is started in this
sandbox.

Configuration choices: none.  Decision 125 remains taken (top-V exchange
off).  Decision 126 remains pending outside this round and is not re-raised.
Sea ice remains exactly as the ORCA2 card carries it, including its
`unmeasured_features` tuple.

## OPEN

1. Build one private production-shaped arm that routes the WS-RK3 stage U/V
   masks through the already-existing resolved raw NEMO mesh operand builder;
   do not create a second mask builder.
2. Prove the stage-1 U and V mask operands bit-exact, then score that mask
   statement singly on rung 0, OMT-1..4, rung 10 under both labels, GYRE,
   DINO, and tanks.  Apply Decision 96; do not bundle it with the held fold unit.
3. Only after the single mask statement is dispositioned, re-test the held
   vector-form fold unit's kt=8 live-W-thickness refusal on the resulting tree.

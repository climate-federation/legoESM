# NEMO testcase Lane 4 — ORCA2 card round 17 continuity-halo receipt

Date: 2026-09-25

Parent: `9f2207d45b2de803407539e85e7ba8d91535bb98`

Measurement tip: `75dae64e0659761066776d66bb7f28d58464646e`

Status: **STOPPED_FOR_RECORD.**  Given NEMO's entry, all 64 cells in the
substep-2 `continuity_du` boundary are one rank-local west-edge column.  The
recorded right U-face operand is bit-exact on all 64; the other operand is a
live halo that the admitted writer deliberately zeros.  The existing record
therefore localizes the boundary but cannot attribute the halo transport's
first non-bit statement.  No production statement lands.

No file under `packages/` changed.  The six sea-ice selectors and the card's
`unmeasured_features` tuple are unchanged: `staged_gm_eiv`,
`linear_implicit_bottom_drag`, `internal_wave_mixing`,
`spatial_lateral_viscosity`, `freshwater_budget_carry`, and
`si3_jpl5_layered_prather_state`.

Every solver number below is labelled **given NEMO's entry**.  The whole-card
ladder remains separately labelled **independent with Decision-52 SSH**.

## 1. Executing statement and record boundary

The executing subtraction is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:550-558`:
NEMO computes `zhU(ji,jj) - zhU(ji-1,jj)` and then the V difference.  The
ordered record writes its canonical fields at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:755-779`.
The writer's view is defined at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:1533-1563`:
it initializes the output to zero and copies owned wet cells only.  A halo may
participate in the live subtraction while being absent from the operand stream.

The admitted record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.
Only rank 0's 90 owned longitudes are scored; no rank-1 or full-domain claim is
made.

## 2. Localization

Round 16 reproduces exactly at the final tip: substep 1 is bit-exact and the
first later non-bit boundary is substep-2 `continuity_du`, 64 / 8,794 wet T
cells, maximum `7.705384632572532e-07`.

All 64 cells are local column 0, one cell in every row 19 through 82.  There is
no interior-column difference.  On that support, legoESM's right U-face
transport and NEMO's recorded right U-face transport are bit-exact, 0 / 64
unequal.  Therefore the only unscored input to the subtraction is the left
halo face.

The candidate's two actual operands replay its traced subtraction bit-exactly.
NEMO's left operand can be inferred as `recorded_right - recorded_du`, and
that inferred operand replays NEMO's recorded subtraction bit-exactly.  On the
64 cells that survive the subtraction, the inferred left operand differs on
64 / 64 with the same maximum `7.705384632572532e-07`.

This inference is not promoted to a direct halo measurement.  Across all
8,794 wet T cells it differs from legoESM's left operand on 1,349 cells, while
only 64 survive the subtraction; its signed delta closure differs on 1,296
cells with maximum `1.2369127944111824e-10`.  The record has enough information
to localize the visible boundary but not to walk the missing halo transport's
own multiplication or its velocity/depth inputs.

## 3. Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R17-P1 | **CONFIRMED** | inherited first boundary, count, maximum, and substep-1 identity reproduce exactly. |
| R17-P2 | **CONFIRMED** | all 64 cells are local column 0; none is interior. |
| R17-P3 | **CONFIRMED** | the recorded right-face operand is bit-exact on all 64 cells. |
| R17-P4 | **REFUTED** | the 64 visible cells and maximum are explained by the inferred left face, but its all-wet support is 1,349 cells rather than the same 64 because inversion exposes rounding differences that the subtraction removes. |
| R17-P5 | **CONFIRMED** | one ULP planted at left-face coordinate `[19, 0]` moves exactly one adjacent subtraction cell; the gate intentionally exits 1. |
| R17-P6 | **CONFIRMED** | no `packages/` diff; ORCA2 remains `LADDER_MEASURED`, with kt=10 entry T maximum `3.9430791763114783` on 430,552 cells. |

The failed prediction is retained; no threshold or hypothesis was rewritten
after measurement.

## 4. Acquisition needed

The committed acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round17_halo_acquisition/run.sh`.
It creates the new isolated target `ORCA2_ORCA1ICE_OMIP_L4_R17HALO` and the new
run `round17/acquisition/orca1ice_bt_halo_ranked_np2`.  Its WRITE-only patch
records, for both MPI ranks and the first two substeps, the direct full local
arrays `zhU`, `e2u`, `ua_e`, `zhup2_e`, and `ssumask`, including both halo
columns.  Preflight passes against pinned source, CPP, deck, input, binary, and
patch digests.  The script refuses an existing target, missing compiled marker,
wrong record size, absent rank, vector-math symbol, failed run, or missing two
rank markers.  It was not run here because sandboxed PMIx is a known refusal.

## 5. Ladder, review, and tests

The final-tip ORCA2 ladder exits 0 with `LADDER_MEASURED`.  Its trajectory
label remains `MEASURED_INDEPENDENT_WITH_DECISION52_SSH`, its first candidate
statement does not move earlier, and kt=10 entry T remains maximum
`3.9430791763114783` on 430,552 cells.  No GYRE rerun is required because
`git diff 9f2207d45..HEAD -- packages/` is empty.

The required separate `codex exec --sandbox read-only` review was attempted at
the committed acquisition tip.  It failed before reading the diff because the
app-server client could not initialize on a read-only filesystem.  Verdict:
**independent review unavailable in-sandbox**.

The clean localization gate exits 0 with `STOP_UNRECORDED_HALO_OPERAND`; its
one-ULP plant fires and exits 1.  The receipt citation gate passes all three
rendered compiled citations with zero failures and zero unmapped citations;
its rigid two-line shift of the rendered `:1533-1563` citation fails with
`SYMBOL-NOT-AT-LINE`.

The focused round-17 battery passes **10 / 10**.  The required
`tests/ocean/fidelity -n 12` battery was launched exactly once: it collected
1,658 tests, reached 99% with three failures and seven skips, then its final
worker produced no output for a bounded ten-minute tail and the run was
interrupted with exit 130.  It is not represented as green.  The three failure
IDs reproduce together in isolation and are exactly Round 16's pre-existing
reds: the GYRE round-129 certified-record stamp says the stepping gate moved;
the SI3 scalar-math gate says `A MY_SRC is not verbatim`; and the GYRE
round-51 assertion expects the stale final six trace fields.  No file in any
of those three implementation/test paths, and no file under `packages/`,
changed in round 17.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round17/codex`.

## Choices

ASKED: localize substep-2 `continuity_du` on both U-face operands before
attribution.

UNASKED: none.  Decisions 54, 57, and 58 remain pending and untouched.  No
configuration choice, carried-state change, stabilizer, sea-ice change, or
production statement was made.

## OPEN

1. Run the reported acquisition, admit both ranked halo streams, then compare
   direct `zhU` first.  If it is non-bit, walk `e2u`, `ua_e`, `zhup2_e`, and
   `ssumask` in the compiled multiplication order before any production fix.
2. The northern-fold mask/wind-stress operands (668 / 35 cells) remain
   reported, not landed.
3. Decisions 54 (three-part `dyn_ldf` scoping), 57 (`r1_rho0` spelling), and
   58 (second continuity solve) remain pending and untouched.
4. The independent ORCA2 year still depends on the scheduled independent
   initial-state completion; this round's solver result is only given NEMO's
   entry.
5. The wide ocean-fidelity battery's hung final tail remains an operator
   action; this round's interrupted run is not represented as green.

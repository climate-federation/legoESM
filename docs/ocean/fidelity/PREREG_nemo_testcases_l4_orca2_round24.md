# NEMO testcase Lane 4 — ORCA2 card round 24 preregistration

Date: 2026-09-26

Parent: `e3f2a7eb97bbaa09f6be0c9c065d0831adbdadef`

Status: **PREREGISTERED BEFORE ROUND-24 SCIENTIFIC SCORING.**

Round 24 executes binding Decision 54 as one whole `dyn_ldf` attribution:
NEMO's file-read F coefficient is not masked a second time, the live
T/U/V/F thicknesses are used at their compiled Kbb/Kmm levels, and the
vorticity circulation uses NEMO's stored `e1f*e2f` vertex-cell area and
zonal edge length.  The last two statements are already present on the
merged branch and are re-certified here; the one production edit is the
remaining redundant-mask removal.  No sea-ice selector or
`unmeasured_features` entry changes.

Every ORCA2 ladder number is labelled **independent with Decision-52 SSH**.
The operator replay uses NEMO's recorded kt=2 entry and is labelled **given
NEMO's entry**.  Those labels are not mixed in one score table.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round24/`.

## Compiled statements and controlled change

The executing build reads `ahmt_3d` and `ahmf_3d` from
`eddy_viscosity_3D.nc` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353`,
then masks each coefficient once at `:387-393`.  The level operator explicitly
says that `ahmf` is already multiplied by `fmask` while forming `zwf` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125`.
That statement also uses the live F thickness, stored reciprocal vertex area,
and the `e2v`/`e1u` circulation edges.  The live T/U/V thickness-weighted
divergence is at `:127-129`; the Kmm face-thickness divisors are at `:133-140`.

The only intended new model statement makes the existing production
`vertex_mask` multiplication conditional on a coefficient source that has not
already been masked.  The `nemo_ahm_3d_file` arm passes its recorded `ahmf`
through unchanged.  This is not a new configuration choice: the selected
source already fixes the coefficient's provenance.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R24-P1 | The current merged code already supplies NEMO's six live LDF thickness operands and stored metric reciprocals on the WS-RK3 path. | Runtime/config inspection names the executing cards and a live trace shows the literal operand bundle; no new selector is required. | The ORCA2 step takes the min-thickness or recomputed-metric path, or a configuration choice is needed. |
| R24-P2 | The redundant F mask is the remaining ORCA2 operator mismatch after the two merged shared statements. | At kt=2, the current operator differs from the compiled replay; removing only the second mask makes every scored u/v cell bit-exact, or reaches the already-measured source-round floor with every moved cell registered. | A material residual remains whose owner is not one of the three authorized statements. |
| R24-P3 | The ten-step ORCA2 ladder first moves no earlier than kt=2 stage 3 momentum, because kt=1 starts from rest and NEMO calls `dyn_ldf` only at stage 3. | kt=1 and kt=2 entry/stages 1-2 are unchanged and the first movement is kt=2 stage-3 u or v (or later). | Any earlier row moves, or no row moves despite a non-vacuous operator change. |
| R24-P4 | The first non-bit NEMO statement remains kt=1 stage-1 temperature. | Its field/checkpoint and score remain the round-23 value. | The first mismatch becomes earlier or changes owner boundary. |
| R24-P5 | GYRE remains byte-identical base to tip because its computed coefficient is not the already-masked file source; the shared thickness/metric statements are unchanged in this diff. | 0 moved certified rows, `np.array_equal` residual arrays, and byte-identical 30-day snapshots. | Any GYRE certified row, residual cell, or daily snapshot moves. |
| R24-P6 | DINO, lock exchange, and overflow remain within their certified AT/AW allowances. | Their named gates pass without a newly worsened row. | Any new failure or worsening outside the existing allowance register. |
| R24-P7 | The controls can fail. | Reinstating the extra mask recreates the operator mismatch; a one-ULP ladder plant and a rigid citation shift are refused. | Any planted violation passes. |

Failed predictions remain **REFUTED** in the receipt and gate.  Directional
ORCA2 ladder changes are registered, not used to remove a cited NEMO statement.

## Landing and stop rules

- Gate the complete three-part operator against the compiled kt=2 replay.
- Run fresh base and tip ORCA2 kt=1..10 ladders; no AT-BAR row may leave the
  bar, the first-over-bar row may not move earlier, and every moved row is
  registered.
- Run GYRE's certified 70-row trajectory and 30-day member at base and tip;
  report every moved row against the `2e-10 K` floor even if none move.
- Run the DINO, lock-exchange, and overflow gates required by Decision 54.
- The six sea-ice selectors and `unmeasured_features` tuple stay frozen.
- No acquisition, configuration choice, carried-state change, stabilizer, or
  NEMO source edit is authorized.

## Choices

ASKED: binding Decision 54 authorizes all three cited `dyn_ldf` statements as
one landing under the cross-card gates above.

UNASKED: none.

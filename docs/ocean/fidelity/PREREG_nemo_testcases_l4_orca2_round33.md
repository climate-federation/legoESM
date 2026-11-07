# NEMO testcase Lane 4 — ORCA2 card round 33 preregistration

Date: 2026-09-26

Parent: `94522f39a4`

Status: **PREREGISTERED BEFORE ROUND-33 SCIENTIFIC SCORING.**

Round 32 retained a failed given-entry replay of NEMO's lateral-diffusion
operator.  Round 33 starts from the live production operands and walks them in
compiled order.  The first candidate is the F-point viscosity coefficient:
the admitted file already contains NEMO's masked `ahmf`, while production
multiplies it by a second binary vertex mask before `dyn_ldf` consumes it.

Direct operator results are **given NEMO's entry (kt=2 recorded state)**.
Trajectory results are **independent with Decision-52 SSH**.  These labels are
not mixed in one table.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round33/`.

## Compiled statements

The executing build reads `ahmt_3d` and `ahmf_3d` from the configured file at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353`, then
multiplies those stored arrays by `tmask` and `fmask` once at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393`.
The executing level operator subsequently reads `ahmf` as already masked at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R33-P1 | The first differing live operand is the extra binary mask applied to the already-masked file coefficient. | The card's carried coefficient is bit-identical to the admitted record; applying production's second mask changes at least one nonzero coefficient; every earlier operand is exact. | The carried coefficient differs before the mask, the mask moves no coefficient, or an earlier live operand differs. |
| R33-P2 | Removing only that second mask makes the corrected round-32 replay bit-exact. | 0 / 411,736 U and 0 / 412,537 V scored cells unequal, with the existing one-ULP plant refused. | Any unplanted unequal cell, changed score set, or silent plant. |
| R33-P3 | The correction is trajectory-vacuous on ORCA2 through kt=10, reproducing round 25's isolated arm. | 40/40 checkpoints, 0/200 moved rows, unchanged first non-bit statement and kt=10 maxima. | Any row moves, an exact row leaves its bar, or the run refuses. |
| R33-P4 | GYRE remains byte-identical because it does not take the file-read coefficient branch. | 0 differing certified rows, `np.array_equal` residual arrays, and byte-identical day-30 snapshots. | Any content or snapshot difference. |
| R33-P5 | DINO, lock-exchange and overflow remain green; the focused gate and citation plants fire. | Required shared-card tests pass and every planted violation exits nonzero. | Any new failure or passing plant. |

Failed predictions remain **REFUTED** and are not rewritten.

## Order and landing rule

1. Commit this preregistration before running a new round-33 score.
2. Add a read-only operand-walk gate that scores the live coefficient before
   and after production's second mask, then the complete operator.
3. If R33-P1 and R33-P2 confirm, remove only the second mask on the
   `nn_ahm_ijk_t=-30` file-read branch; computed-coefficient cards retain their
   existing mask application.
4. Run the ORCA2 ten-step ladder, GYRE trajectory and day-30 gate, DINO,
   lock-exchange, overflow, citation and push gates before landing.
5. If a later operand still differs, stop at that first statement and retain
   the single-mask correction only if its own full gate passes.

No stabilizer, clipping, configuration choice, carried-state change, NEMO
source edit, sea-ice edit, score change, or acquisition is authorized.

## Choices

ASKED: Decision 54 authorizes the complete three-part lateral-diffusion
attribution; round 32's OPEN section orders a one-variable walk from the live
production operands.

UNASKED: none.

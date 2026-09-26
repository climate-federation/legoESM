# NEMO testcase Lane 4 — ORCA2 card round 34 preregistration

Date: 2026-09-26

Parent: `7de8183094`

Status: **PREREGISTERED BEFORE ROUND-34 SCIENTIFIC SCORING.**

Round 33 removed the first differing live lateral-diffusion operand, the
second mask on the file-read F coefficient.  Its corrected replay remains
non-bit.  The next compiled-order difference is the F-column reciprocal used
to form `r3f`: production reconstructs the column from `e3f_0vor`, while NEMO
carries `hf_0` built from the mesh `e3f_3d` and two V masks.

Direct-operator results are **given NEMO's entry (kt=2 recorded state)**.
Trajectory results are **independent with Decision-52 SSH**.  These labels are
not mixed in one table.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round34/`.

## Compiled statements

The executing build forms `hf_0` from the mesh F thickness and adjacent V
masks at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:203-206`,
then forms and stores `r1_hf_0` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:212-215`.
The RK3 QCO routine consumes that stored reciprocal while forming `r3f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R34-P1 | The first remaining differing `r3f` operand is the reconstructed F-column depth. | The reconstruction reproduces round 32's 14,324 / 26,640 unequal cells and 651.239260895305 m maximum, while the card's carried `hf_0` is bit-identical to the admitted record. | The reconstruction is exact, the carried operand differs from the record, or an earlier `r3f` operand differs. |
| R34-P2 | Substituting only carried `hf_0` reproduces round 33's secondary-owner replay but does not close it. | U/V maxima are `3.181628207426175e-09` / `2.9702048395431957e-09` m/s2, with at least one unplanted unequal cell. | Either replay is bit-exact, the frozen maxima change, the score set changes, or the plant is silent. |
| R34-P3 | The independent ORCA2 ladder completes kt=1..10, moves at least one registered row, loses no exact/AT-BAR row, and keeps the first non-bit statement at kt=1 stage-1 temperature. | 40/40 checkpoints, moved-row count greater than zero, empty AT-BAR-loss list, unchanged first statement. | A refusal, zero moved rows, an unregistered row, an exact row leaving its bar, or an earlier first statement. |
| R34-P4 | GYRE remains byte-identical because its card does not carry the ORCA2 native `hf_0` operand selected by this repair. | 0 differing certified rows, `np.array_equal` residual arrays, and byte-identical day-30 snapshots. | Any content or snapshot difference. |
| R34-P5 | DINO, lock-exchange and overflow remain green; the focused gate and citation plants fire. | Required shared-card tests pass and every planted violation exits nonzero. | Any new failure or passing plant. |

Failed predictions remain **REFUTED** and are not rewritten.

## Order and landing rule

1. Commit this preregistration before any new round-34 score.
2. Extend the existing round-32/33 read-out; do not build a second replay.
3. Compare reconstructed and carried `hf_0` against the admitted record, then
   substitute only the carried operand in production.
4. Land only if the operand itself is bit-exact, the ORCA2 ladder loses no
   exact/AT-BAR row, GYRE is byte-identical, shared-card gates pass, and every
   plant fires.  The residual replay remains debt unless it reaches zero.

No stabilizer, clipping, configuration choice, carried-state change, NEMO
source edit, sea-ice edit, score change, or acquisition is authorized.

## Choices

ASKED: round 33's OPEN section authorizes the compiled-order `r3f` operand
walk; the card already carries the NEMO `hf_0` operand selected here.

UNASKED: none.

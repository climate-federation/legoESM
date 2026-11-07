# NEMO testcase Lane 4 — ORCA2 card round 36 preregistration

Date: 2026-09-26

Parent: `4d8918b27`

Status: **PREREGISTERED BEFORE ROUND-36 SCIENTIFIC SCORING.**

Round 35 showed that changing only division to multiplication by a stored
reciprocal is trajectory-inert at the given-entry lateral-diffusion tendency,
while the area supplied to that arithmetic is not NEMO's native F-cell area:
26,456 / 26,640 cells differ.  Round 36 isolates that first differing input.
It does not change the separately landed lateral-diffusion metric operands.

Direct-operator results are **given NEMO's entry (kt=2 recorded state)**.
Trajectory results are **independent with Decision-52 SSH**.  These labels are
not mixed in one table.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round36/`.

## Compiled statements and scope

The executing build forms and stores the native F-cell area and its reciprocal
at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:155-157`.
The RK3 QCO routine consumes that F-cell reciprocal while forming `r3f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The lateral-diffusion consumer applies the resulting live thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

The one-variable arm selects the card-carried native `e1f*e2f` product as the
`r3f` area.  It retains the current division spelling so the round changes only
the operand source; round 35 already measured the alternative reciprocal
spelling separately.  Cards without raw NEMO F metrics retain the existing
grid-area fallback.

Round 25's "native F metrics" arm is not this arm: it routed six stored metric
reciprocals into the lateral-diffusion operator while leaving `r3f` on its
then-current construction.  Its 175-row trajectory result is historical
context, not a prediction for this change.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R36-P1 | The tripolar constructor keeps the native F-cell product at `area_q[:-1, :-1]`; the current `area_q[1:, 1:]` selection is a one-cell north/east shift. | Native `e1f*e2f` is bit-identical to `area_q[:-1, :-1]` in 26,640 / 26,640 cells, while the current selection reproduces round 35's 26,456 unequal cells and `28564758282.12061` m2 maximum. | The unshifted slice differs, the shifted slice is exact, or the registered round-35 count/maximum changes. |
| R36-P2 | Substituting only the native area closes the given-entry lateral-diffusion replay at the bit bar. | U and V each have zero unequal scored cells; the unplanted gate returns `AT_BAR`. | Either field has any unequal scored cell, a score set changes, or the plant is silent. |
| R36-P3 | The independent ORCA2 ladder completes kt=1..10, loses no exact/AT-BAR row, and keeps the first non-bit statement at kt=1 stage-1 temperature. | 40/40 checkpoints, empty AT-BAR-loss list, unchanged first statement, and every moved row registered. | A refusal, an unregistered row, an exact row leaving its bar, or an earlier first statement. |
| R36-P4 | GYRE remains byte-identical because its constant Cartesian F area is invariant under the old shifted slice and the native product. | Base/tip ten-step ladders have 0 differing certified rows and `np.array_equal` residual arrays; base/tip 30-day snapshots are byte-identical. | Any certified row, residual array, or daily snapshot differs.  Because GYRE executes this shared statement, such movement stops the landing for a user decision. |
| R36-P5 | DINO, lock-exchange and overflow remain green; focused, citation and outcome plants fire. | Required shared-card tests pass and every planted violation exits nonzero. | Any new shared-card failure or passing plant. |

Failed predictions remain **REFUTED** and are not rewritten.

## Order and landing rule

1. Commit this preregistration before any new round-36 score.
2. Extend the existing round-32/35 replay; do not build a duplicate operator.
3. Measure the two grid slices against the admitted card product, then
   substitute only the native area source in production.
4. Land only if the replay reaches the bit bar, the ORCA2 ladder loses no
   exact/AT-BAR row, GYRE is byte-identical through ten steps and 30 days,
   shared-card gates pass, and every plant fires.

No stabilizer, clipping, configuration choice, carried-state change, NEMO
source edit, sea-ice edit, score change, or acquisition is authorized.

## Choices

ASKED: round 35's OPEN section names the native F-area slice as the next
compiled-order input, and the card already carries the two source metrics.

UNASKED: none.

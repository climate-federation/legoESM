# NEMO testcase Lane 4 — ORCA2 card round 37 preregistration

Date: 2026-09-26

Parent: `53dab7544`

Status: **PREREGISTERED BEFORE ROUND-37 SCIENTIFIC SCORING.**

Rounds 35 and 36 measured both halves of one NEMO `r3f` dataflow separately.
The stored-reciprocal spelling moves 3,810 `r3f` cells but no scored tendency
bit while it is fed the wrong shifted area.  The native area removes the main
residual but, under division, leaves exactly 24 unequal U and 24 unequal V
tendency cells at `1.0587911840678754e-22` m/s2.  This round preregisters the
pair before measuring it together.  This is the cancelling-pair analysis
required by the campaign rules, not a relaxation of either isolated result.

Direct-operator results are **given NEMO's entry (kt=2 recorded state)**.
Trajectory results are **independent with Decision-52 SSH**.  These labels are
not mixed in one table.  The six sea-ice selectors and the card's
`unmeasured_features` tuple remain frozen.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round37/`.

## Compiled pair

The executing build materializes native `e1f*e2f` and stores `r1_e1e2f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:155-157`.
The executing QCO statement multiplies the `r3f` numerator by that stored
reciprocal at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The lateral-diffusion consumer is
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

Production will select the card's native F-area when raw NEMO mesh operands
exist and multiply by its materialized reciprocal.  Cards without those raw
operands retain the established geometric fallback.  No configuration field
or card value is introduced.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R37-P1 | The measured pair closes the given-entry lateral-diffusion replay. | U and V each have zero unequal scored cells; the unplanted gate returns `AT_BAR`. | Either field has any unequal scored cell, a score set changes, either isolated control is not reproduced, or the plant is silent. |
| R37-P2 | The independent ORCA2 ladder completes kt=1..10, loses no exact/AT-BAR row, and keeps the first non-bit statement at kt=1 stage-1 temperature. | 40/40 checkpoints, empty AT-BAR-loss list, unchanged first statement, and every moved row registered. | A refusal, an unregistered row, an exact row leaving its bar, or an earlier first statement. |
| R37-P3 | GYRE remains byte-identical because its constant Cartesian F-area operand is unchanged by both source selection and reciprocal spelling. | Base/tip ten-step ladders have 0 differing certified rows and `np.array_equal` residual arrays; base/tip 30-day snapshots are byte-identical. | Any certified row, residual array, or daily snapshot differs.  Because GYRE executes this shared statement, such movement stops the landing for a user decision. |
| R37-P4 | DINO, lock-exchange and overflow remain green; focused, citation and outcome plants fire. | Required shared-card tests pass and every planted violation exits nonzero. | Any new shared-card failure or passing plant. |

Failed predictions remain **REFUTED** and are not rewritten.

## Order and landing rule

1. Commit this preregistration before measuring the combined arm.
2. Extend the existing round-36 gate so it proves both isolated controls and
   the combined result; do not create another operator replay.
3. Run the ORCA2 ten-step ladder only if the pair reaches zero replay cells.
4. Land only if ORCA2 loses no exact/AT-BAR row, GYRE is byte-identical through
   ten steps and 30 days, shared-card gates pass, and every plant fires.

No stabilizer, clipping, configuration choice, carried-state change, NEMO
source edit, sea-ice edit, score change, or acquisition is authorized.

## Choices

ASKED: round 36's OPEN section authorizes the measured pair.  Both halves are
compiled NEMO statements and have already failed their isolated falsifiers in
the complementary way registered above.

UNASKED: none.

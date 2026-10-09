# ORCA2 round 192 preregistration — atomic slow-depth and V-transport unit

Date: 2026-10-09. Base: `4191b3671f4daf8605bb958e86281a1fc90eba55`.
Rung-0 and month numbers are **independent hierarchy rung 0**. Rung-7
numbers, if reached, are **given NEMO's entry**. They are never mixed in one
table. The shipped rung-10 sea-ice selectors and `unmeasured_features` tuple
remain untouched.

## Frozen scope and statistic

Test one private atomic program containing only statements already named and
separately measured in rounds 146–154, 190 and 191:

1. NEMO's source-associated completed-RHS U/V depth average at
   `ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:203-215`;
2. raw `hu_0/hv_0` in the midpoint and exit face depths at
   `ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`;
3. no extra compact V mask and a separately materialised completed `zhV` at
   `ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:564-570`;
4. the seven-array external-mode association at
   `ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.

The exact slow-depth implementation from held commit `071670e58` is applied
only while measuring the candidate; the other four private controls use the
existing test hooks. The final tree retains the whole unit only if every
landing predicate passes; otherwise the package diff is reverted and the
unit remains private. No selector, physical configuration, forcing, carried
state, stabiliser, threshold, cadence, field set or score changes.

The primary gate is the fixed independent rung-0 200-row, kt=1..10 ladder.
Rows are compared against the unmodified base with the existing round-111
comparator. Decision 96 counts only RMS-score-moved rows: strict majority
toward NEMO, first over-bar row toward or equal, no bit-exact row lost, and
kt=10 stage-3 SSH maximum not worse. Bit-moved/RMS-equal rows are registered
but do not vote. A terminal refusal is an automatic hold.

Only if rung 0 is eligible, run the given-entry rung-7 ladder under the same
predicate, then the independent month through its next boundary, GYRE's
certified ladder/year, DINO, tanks and the shared-card registry. The existing
round-190 and round-191 offline reports remain the local operand witnesses;
no in-executable observer is authorised.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R192-P1 | The atomic candidate preserves every separately exact prerequisite. | Slow U/V means, midpoint face depths, V transport/continuity and the seven associated arrays reproduce their admitted NEMO rows at the existing frozen boundaries. | Any prerequisite moves or an instrument/passivity check fails: instrument/unit invalid; quote no ladder direction and stop. |
| R192-P2 | Exact slow forcing removes the private unit's former kt=8 terminal and the independent rung-0 ladder completes 40 checkpoints. | All 200 rows are finite through kt=10 stage 3. | Any refusal or non-finite row: **REFUTED**; retain the first terminal boundary and revert the candidate. |
| R192-P3 | The complete unit is a Decision-96 net improvement on rung 0. | RMS toward exceeds away, first debt is toward/equal, no exact row is lost, and kt10 stage-3 SSH maximum does not increase. | Any predicate fails: **REFUTED**; keep the unit private and name the first failing row/predicate. |
| R192-P4 | Given-entry rung 7 remains executable and loses no exact row. | Its 200-row ladder completes, first debt is not earlier, and every previously exact row remains exact. | Any terminal, earlier debt or exact-row loss: **REFUTED** and hold. |
| R192-P5 | If both ladders pass, the independent month advances beyond the registered step-96 boundary. | Step 96 completes finitely or the first refusal occurs later. | Step 96 or an earlier step refuses: **REFUTED**; retain the exact boundary and do not call the unit stabilising. |
| R192-P6 | Every classifier/control remains non-vacuous. | Entry-bit, exact-row-loss, false-majority, score-equal-vote, first-boundary and candidate-identity plants each refuse. | Any plant stays green: invalid gate; report no landing verdict. |

If the primary rung-0 candidate refuses or fails Decision 96, the round ends
**HELD** after restoring the package tree; downstream shared gates are not run
because they cannot reverse the dispositive ORCA2 failure.

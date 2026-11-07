# ORCA2 round 47 preregistration — shared stage-1 Kaa ratio landing

Date frozen: 2026-09-27  
Cards: `orca2_vector_een_c2` plus every shared executing card  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
GYRE/tank claim label: **independent**

## Authorization and one-statement scope

Decision 52 / operator note B8 answers round 46's pending decision: restore
only NEMO's stage-1 Kaa T-point `r3t` interpolation statement and judge it
under the full GYRE Decision 43/45/55/59 gate.  NEMO first forms the rounded
Kbb and after-level ratios, then evaluates
`r2_3*r3t(Kbb) + r1_3*r3ta` in the compiled ORCA2 branch at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`.

The downstream tracer QCO/RK assignment at `stprk3_stg.f90:670-681` remains
out of scope.  No configuration, selector, default, carried state, score
domain, threshold, stabiliser, or sea-ice field may change.  ORCA2's six-item
`unmeasured_features` tuple is frozen.

Repository search before implementation found one existing production helper
for a single-time-level ratio (`nemo_r3t_stretch`) and one committed diagnostic
copy of the endpoint interpolation in the round-45 gate
(`_rk13_q_from_endpoints`).  The landing will promote that already measured
arithmetic to one shared production helper; it will not add a second numerical
spelling.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R47-P1 | The round-46 base reproduces before editing. | ORCA2 Kaa is 2 / 8,613 unequal; production stage 1 is 57,160 T / 57,180 S unequal. GYRE day-30 T RMS is `6.572574374770603e-05 K`; day-240/360 are `1.644836070117868e-02` / `1.1225660018551306e-02 K`; ten-step residual digest is `cf06a8fc7d0e90f2`. | Any number differs beyond the recorded display precision: stop and reconcile before attribution. |
| R47-P2 | The one production statement is exact on ORCA2 given NEMO's entry. | Kaa becomes 0 / 8,613 unequal; stage 1 becomes 57,141 T / 57,169 S unequal, exactly the frozen downstream fused replay. | Any unequal Kaa cell or different downstream count: do not land. |
| R47-P3 | ORCA2's full ten-step policy passes. | 40/40 checkpoints complete; no AT-BAR row leaves; first non-bit remains kt=1 stage-1 temperature at `0.0014770192519700243 K`; every moved row is registered. | Any refusal, bar loss, earlier first row, or unregistered move: hold. |
| R47-P4 | GYRE's ten-step ladder remains byte-identical, but the month trajectory repeats round 46's measured movement. | 70 certified rows unchanged, 210 residual arrays array-equal, digest unchanged; days 1-12 array-equal and day 13 first moves; day-30 T RMS decreases to about `6.572572612618985e-05 K`. | Any ten-step move, earlier daily move, or day-30 non-decrease: hold. |
| R47-P5 | The GYRE year movement is below Decision 59's ten-floor-unit bound. | Each absolute day-240 and day-360 T-RMS change is `<2e-9 K`; every year row is registered against the `2e-10 K` floor. | Either change is `>=2e-9 K` and the statement does not take a certified row to AT-BAR under Decision 55: hold and name the failing row. |
| R47-P6 | The two tanks retain their current oracle-relative trajectory classifications. | Base/tip LOCK_EXCHANGE and OVERFLOW ten-step reports have no earlier first-over-bar, no kt=1 AT-BAR loss, and all movements registered. | Any earlier first row or kt=1 bar loss: hold. |
| R47-P7 | The generic card and DINO are outside the numerical blast radius for stated reasons. | The generic card's three-step certified snapshots are byte-identical because its linear-free-surface branch returns one; both DINO cards resolve Euler and their committed gates are unchanged. | Any generic-card bit moves or DINO gate regression: hold. |
| R47-P8 | The change preserves JIT and autodiff. | Direct eager/JIT equality, a non-vacuous wrong-association control, and finite nonzero endpoint gradients pass; reverting the production call makes the binding test fail. | Any mismatch, vacuous control, non-finite/zero gradient, or revert-green test: do not land. |

## Required measurements and landing rule

1. Measure clean same-tip base and candidate arms for the ORCA2 ten-step
   ladder, ORCA2 internal scorer, GYRE ten-step ladder, GYRE days 1-30, and
   GYRE days 30/60/90/120/180/240/300/360.
2. Measure both tank trajectories and the generic NEMO-GYRE card before and
   after; run the DINO/tank/generic recipe and card batteries.  The resolved
   execution census itself is committed in the receipt.
3. The candidate lands only when R47-P2 through R47-P8 pass under Decision
   43/45/55/59: day 30 decreases; first-over-bar is not earlier; kt=1 AT-BAR
   rows stay; every moved row is registered; and day-240/360 satisfy Decision
   59 or Decision 55 exactly as frozen above.
4. Run the default and round citation gates with a real shifted-line plant,
   focused tests, the required card/push gates, and one
   `tests/ocean/fidelity -n 12` battery after checking no other battery runs.
5. Request a separate read-only Codex review.  If it cannot initialize, record
   `independent review unavailable in-sandbox` exactly.

## Frozen OPEN if the statement lands

Return to the separately proven source-ordered QCO/RK tracer assignment and
require the ORCA2 production stage-1 T/S row to become exact.  The round-20
slow-forcing walk and Decision-52 independent ORCA2 initial state/year remain
open.  Sea ice remains out of scope.

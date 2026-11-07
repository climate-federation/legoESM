# ORCA2 round 46 — stage-1 Kaa r3t interpolation decision handoff

Date: 2026-09-27  
Card: `orca2_vector_een_c2`  
Status: **STOPPED_FOR_DECISION; CANDIDATE REVERTED**

## Verdict

The ORCA2 statement is exact and passes its full ten-step landing policy, but
it is not ORCA2-only.  GYRE remains byte-identical for ten steps and then first
moves at day 13; by day 30 every state field differs.  The ORCA2 lane therefore
cannot land the shared statement without treating it as a GYRE landing under
Decisions 43/45/55.  Candidate commit `e87f8ca659d61257d613f55fe8e4e94920a59a89`
was reverted in `19ca6059f99ccf632b96c07e2f48aa2391ff34c4` before handoff.

The repository tip has no net `packages/` change from the round-45 parent.  No
configuration, selector, default, carried state, score domain, threshold,
stabiliser, or sea-ice field changed.  The card's six-item
`unmeasured_features` tuple remains frozen.

## Compiled statement and exact internal boundary

All ORCA2 rows here are **given NEMO's entry** (Decision 52), CPU, fp64, scalar
libm, production JIT.  NEMO forms the stage-1 Kaa T-point ratio by interpolating
the already-rounded Kbb and after-level `r3t` values at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179`.
The candidate transcribed only that statement.

| ORCA2 internal row | parent | candidate |
|---|---:|---:|
| production Kaa stretch | 2 / 8,613 unequal, max `1.1102230246251565e-16` | **0 / 8,613 unequal** |
| interpolated-SSH control | 2 / 8,613 unequal | 2 / 8,613 unequal |
| stage-1 temperature | 57,160 / 228,641 unequal | 57,141 / 228,641 unequal, max `3.552713678800501e-15 K` |
| stage-1 salinity | 57,180 / 228,641 unequal | 57,169 / 228,641 unequal, max `1.4210854715202004e-14` |

The two production counts equal round 45's frozen recorded-operand fused
replay exactly.  Thus the next debt remains the downstream tracer arithmetic
association at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`;
it was not folded into this candidate.

## ORCA2 ten-step landing gate

These trajectory rows are also **given NEMO's entry**.  Both arms completed 40
checkpoints / 200 scored rows.

- 141 rows moved: 25 toward NEMO by maximum, 35 away, and 81 with the same
  maximum.
- No bit-identical row left the bar.
- The first non-bit checkpoint remains kt=1 stage-1 temperature, 233,341 / 
  399,600 unequal, maximum `0.0014770192519700243 K`.
- The first moved row is that same checkpoint; its unequal count, first index,
  and maximum are unchanged, while its mean over unequal cells changes only in
  the final digits.

The mixed downstream directions are registered, not interpreted as a whole-
trajectory fidelity ranking.  The exact compiled boundary above is the Rule-12
landing evidence.

## Shared GYRE result — independent

These rows are **independent** GYRE from-rest comparisons and are not mixed
with the Decision-52 ORCA2 twin results above.

The ten-step pair is byte-identical: the offline oracle-relative comparator
passes all 70 certified rows with zero worsening, all 210 residual arrays are
`np.array_equal`, and both residual archives have SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.

The daily pair refutes byte identity.  Days 1 through 12 are array-equal; day
13 is the first moved snapshot.  At day 30:

| GYRE field | unequal cells | maximum absolute base/candidate difference |
|---|---:|---:|
| T | 17,916 | `1.0857537091624181e-08 K` |
| S | 17,612 | `3.386723790299584e-09` |
| u | 17,400 | `4.4777842024557035e-10 m/s` |
| v | 17,100 | `5.967746817547859e-10 m/s` |
| ssh | 600 | `2.438473034604982e-11 m` |

Against NEMO, day-30 temperature RMS moves slightly toward the oracle:
`6.572574374770603e-05` to `6.572572612618985e-05 K`, a decrease of
`1.7621516171153345e-11 K`.  This is below the established approximately
`2e-10 K` run-to-run floor and is not used as authorization.  Day 240 and day
360 are unmeasured; the full Decision-43/45/55 gate was deliberately not run
after the brief required a user decision for a shared GYRE statement.

## Frozen predictions

| ID | outcome |
|---|---|
| R46-P1 | **CONFIRMED** — round 45 reproduced exactly before the edit. |
| R46-P2 | **CONFIRMED** — candidate production Kaa is 0 / 8,613 unequal. |
| R46-P3 | **CONFIRMED** — stage 1 becomes 57,141 T / 57,169 S unequal and remains downstream debt. |
| R46-P4 | **CONFIRMED** — ORCA2 completes ten steps with no AT-BAR loss and unchanged first non-bit checkpoint. |
| R46-P5 | **REFUTED** — GYRE is exact through ten steps but moves from day 13 onward. |
| R46-P6 | **CONFIRMED** — focused eager/JIT equality and finite, nonzero endpoint gradients pass. |

The refuted prediction is retained without rewriting.

## Controls, review, and tests

- The direct source-association test and the existing metric-association test:
  `2 passed in 1.95s`.
- Reverting only the production call site makes the direct test fail; the
  restoration was verified before the candidate commit.
- Default receipt citation gate on the candidate: PASS, 274 citations, zero
  failures, zero unmapped citations, zero map-audit failures.  Its shifted-line
  plant exits nonzero with `SYMBOL-NOT-AT-LINE`.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- After the candidate was reverted, the required
  `tests/ocean/fidelity -n 12` battery was run once.  It reproduced the five
  campaign-known reds (round-129 spread-floor record, SI3 scalar-math
  provenance, round-51 private trace registry, worktree-stamp ratchet, and
  case-board oracle-comparison ratchet), reached 99%, and then made no further
  progress in the known xdist-controller hang.  It was interrupted after a
  grace period, so it has no terminal pytest summary and is not reported as
  green.  The final tree has no model or test delta to promote.

## Choices

ASKED: none.  The candidate implemented the already-cited NEMO statement and
changed no configuration.

UNASKED: none.  In particular, the round did not silently promote a statement
that GYRE executes.

DECISION NEEDED: authorize or reject a dedicated GYRE landing of the same
source-exact stage-1 ratio statement under the full Decision-43/45/55 ladder,
day-30, day-240, and day-360 gate.  **Pick: authorize it**, because it closes
the ORCA2 statement exactly and is the compiled GYRE statement too; the full
GYRE year gate should decide whether it lands.

## Evidence

Durable artifacts are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round46/`.
The decisive files are `qco_rk_base.json`, `qco_rk_tip_final.json`,
`orca2_ladder_base.json`, `orca2_ladder_tip.json`,
`gyre_offline_compare.json`, both GYRE residual archives, both 30-day member
directories, `gyre_base_day30_gap.json`, `gyre_tip_day30_gap.json`,
`focused_revert_plant.log`, `fidelity_full_final.log`, and `codex_review.log`.

## OPEN

1. Pending user authorization, restore the single candidate statement and run
   the complete GYRE Decision-43/45/55 gate, including days 240 and 360, before
   any shared landing.
2. If that landing is authorized and passes, return to the separately proven
   source-ordered QCO/RK tracer assignment; require production stage-1 T/S to
   become exact.
3. The ORCA2 first whole-card non-bit checkpoint remains kt=1 stage-1
   temperature at `0.0014770192519700243 K`.
4. Round-20 slow forcing and the Decision-52 **independent** ORCA2 initial
   state/year comparison remain open.
5. Sea ice remains out of scope and exactly the frozen six-item
   `unmeasured_features` tuple.

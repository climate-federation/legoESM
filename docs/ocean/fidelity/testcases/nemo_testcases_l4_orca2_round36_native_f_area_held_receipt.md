# NEMO testcase Lane 4 — ORCA2 card round 36 native-F-area receipt

Date: 2026-09-26

Parent: `4d8918b27`

Experiment commit: `e75bf58be`

Restored-tree commit: `622057996`

Status: **HELD — THE NATIVE F AREA IS CORRECT, BUT ISOLATING IT MISSES THE
BIT BAR BY 24 CELLS PER VELOCITY FIELD.**  The tripolar index-map prediction
is exact.  Substituting only that operand removes more than nine orders of
magnitude from the given-entry lateral-diffusion residual, but the frozen
zero-cell landing bar is not met.  The model change was reverted; the final
`packages/` tree is identical to the parent.

All numbers below are **given NEMO's entry (kt=2 recorded state)**.  No
independent trajectory result is claimed after the operator gate stopped the
round.  The six sea-ice selectors and the card's `unmeasured_features` tuple
were not changed.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round36/`.

## Compiled statements and isolated arm

The executing build forms the native F-cell area and stores its reciprocal at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:155-157`.
The executing QCO routine consumes that reciprocal in `r3f` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286`.
The lateral-diffusion consumer applies the resulting live F thickness at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123`.

The experiment changed only the area source from the shifted padded-grid
slice to the card-carried native `e1f*e2f` product.  It deliberately retained
the current division spelling, because round 35 had measured reciprocal
spelling separately.  No LDF metric operand, thickness reference, card field,
configuration selector, or forcing changed.

## Operand boundary and given-entry replay

The native product is bit-identical to `area_q[:-1, :-1]` in all 26,640
cells.  The displaced production slice `area_q[1:, 1:]` reproduces round 35
exactly: **26,456 / 26,640 unequal**, maximum
**`28564758282.12061` m2**.  R36-P1 is **CONFIRMED**.

The isolated area arm does not reach the registered bit bar:

| field | parent unequal / scored | parent maximum, m/s2 | area-arm unequal / scored | area-arm maximum, m/s2 |
|---|---:|---:|---:|---:|
| U | 410,460 / 411,736 | `3.181628207426175e-09` | **24 / 411,736** | **`1.0587911840678754e-22`** |
| V | 407,570 / 412,537 | `2.9702048395431957e-09` | **24 / 412,537** | **`1.0587911840678754e-22`** |

R36-P2 is **REFUTED**.  The result is not rounded to zero and is not called
AT-BAR.  The planted U value exits nonzero.  Round 35 independently measured
that changing division to multiplication by the stored reciprocal moves 24
wet-face LDF outputs, so the matching count is a preregistered-history clue,
not a post-hoc closure claim.  Their combination remains unmeasured in this
round.

## Frozen predictions

| ID | verdict | measurement |
|---|---|---|
| R36-P1 | **CONFIRMED** | The unshifted grid slice is exact; the shifted slice reproduces 26,456 unequal cells and the frozen maximum. |
| R36-P2 | **REFUTED** | Both fields retain 24 unequal cells at `1.0587911840678754e-22` m/s2. |
| R36-P3 | **UNMEASURED AFTER STOP** | The ORCA2 trajectory was not run because the operator landing bar failed. |
| R36-P4 | **UNMEASURED AFTER STOP** | GYRE base/tip gates were not run for a prohibited landing; the final package tree equals the certified parent. |
| R36-P5 | **PARTIAL** | The focused gate and scientific plant pass; shared-card landing gates were not run after the stop. |

Failed predictions are retained.  Unmeasured predictions are not inferred
passes.

## Gates, review, and tests

The scientific gate returns `DEBT`; its plant exits 1.  The accidentally
started parent trajectory was interrupted when the implementation commit was
made in the same worktree and is explicitly inadmissible; it produced no JSON
and supports no claim here.  The parent package tree is already represented
by round 34's admitted artifacts, but they are not reused for a trajectory
claim because this round stopped earlier.

The required separate `codex exec --sandbox read-only` review was attempted
at the committed diff.  It failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

The focused round-32/35/36 binding battery reports **9 passed**.  The receipt
citation gate passes all three compiled citations with zero failures, zero
unmapped citations and zero map-audit failures.  Its rigid two-line source
plant exits 1 with `SYMBOL-NOT-AT-LINE`; all nine citation self-tests fire.

The required `tests/ocean/fidelity -n 12` battery reached 98% and the inherited
final-tail stall, then was interrupted after a bounded silent wait.  It emitted
the same five inherited failures as rounds 34 and 35: SI3 MY_SRC provenance,
the stale GYRE member/gate stamp, the round-51 trace suffix, three unstamped
legacy report emitters, and the missing `hires_lane_surface` case-board row.
Their exact node IDs were rerun in isolation and all five reproduce in 7.01 s.
No round-36 test fails.

## Choices

ASKED: round 35's OPEN section authorized this native-F-area substitution.

UNASKED: none.  No configuration value, stabilizer, carried state, NEMO
source, sea-ice selector, score, or acquisition changed.  The experimental
model edit was fully reverted.

## OPEN

1. Preregister the measured cancelling-pair landing: native F area plus the
   compiled stored-reciprocal multiplication.  Each half is now separately
   measured; their combination must reach zero unequal tendency cells before
   any ORCA2/GYRE trajectory gate is eligible.
2. Round 20's ranked slow-forcing producer walk remains open.
3. The northern-fold mask and wind-stress operands (668 / 35 cells) remain
   reported, not landed.
4. Decision 52's independent ORCA2 initial-state transcription and year
   comparison remain owed.
5. The inherited duplicate citation-map literal keys remain open.

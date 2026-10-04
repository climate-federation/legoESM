# ORCA2 round 135 — GYRE merge and passive FCT walk

Date: 2026-10-04. Base `a5b9627e5`; merged GYRE tip `a3be519e0`;
measurement tip `0e563a843`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round135`.
The ORCA2 ladder rows are **given NEMO's recorded entry**. The rung-0 month
and step-36 FCT walk are **independent**, starting from the rung-0 card's own
climatological T/S, zero velocity, and zero sea surface. These populations
are not mixed below.

## Verdict

**HELD.** The merge is mechanically complete, the passive FCT seam is an
admissible instrument, and the source walk names a distinct global overflow.
It does not explain the headline returned-cell failure: T `[86,159,0]` is
finite through every exposed FCT statement and through the caller's stage-3
advection content. The next walk at that cell is therefore downstream of
advection, beginning with the implicit vertical-diffusion/later stage path.

The round cannot land because the merged VORTEX round-201 known-answer
control is red: feeding the model its own exposed stage-1 momentum RHS changes
32,422/40,320 elements in its first reported leaf, with maximum absolute
difference `4.440892098500626e-16`. A diagnostic census also found a leaf at
12 row-scale ULP, so the two-ULP ratchet cannot be invoked as a pass. The test
remains exact; no tolerance was relaxed.

No public config field, card, forcing, carried-state definition, stabilizer,
threshold, deck, sea-ice selector, or `unmeasured_features` entry changed.
In particular, the rung-7 one-category ice debt remains exactly as carried by
the card. No NEMO acquisition is needed.

## Merge and given-entry ladders

Commit `785712759` merges the exact requested incoming tip. Its only conflict
was one citation-map dictionary hunk; resolution retained the semantic union
of both parents. Both parents are ancestors and no unmerged path remains.

The incoming statements are NEMO's vector depth average at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:194-200` and its carried external-mode
Coriolis operand/subtraction at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:320-324`.

Against round 132, the **given-entry** rung-0 ladder moves 194/200 rows
(6 unchanged, zero exact-row losses); RMS direction is 93 toward / 97 away /
4 equal and maximum-error direction is 77 toward / 103 away / 14 equal. The
**given-entry** rung-7 ladder moves 195/200 rows (5 unchanged, zero exact-row
losses); RMS direction is 35 toward / 151 away / 9 equal and maximum-error
direction is 54 toward / 127 away / 14 equal. On both cards the first non-bit
boundary remains kt=1 stage-1 T. The full row register is
`ladder_compare.json`; both planted exact-row-loss and earlier-first-boundary
violations refuse.

After adding the private instrument, both ladders were repeated: 0/200 rows
move on rung 0 and 0/200 on rung 7, with zero exact-row losses and unchanged
first boundaries (`instrument_ladder_compare.json`).

## Independent step-36 walk

The ordinary rung-0 month repeats the admitted boundary: it is finite through
step 35 and first returns non-finite T at step 36, `[86,159,0]`. The private
post-step seam publishes already-materialized stage-3 FCT inputs through an
otherwise unused diagnostic return slot. All 49 ordinary state leaves are
bit-identical with the seam on and off, including NaN payloads, and a separate
ordinary replay is bit-identical in T/S/u/v/ssh. The passivity, source-order,
and active-support plants all fire.

The compiled second predictor first averages the upstream horizontal flux at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:569-572` and the vertical flux at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:577-578`. This is the first
source-ordered active-support non-finite group:

| Tracer / flux | Non-finite | First `(j,i,k)` | Largest finite magnitude |
|---|---:|---|---:|
| T / horizontal U | 14 | `(87,160,5)` | `1.3507494290968843e308` |
| T / vertical W | 36 | `(87,159,1)` | `7.889125700887634e306` |
| S / horizontal U | 7 | `(87,160,5)` | `1.3190713148227455e308` |
| S / vertical W | 31 | `(87,159,1)` | `4.719873690080326e307` |

Thus the first statement is the horizontal `ptFu` averaged-upstream-flux
assignment, not the preregistered antidiffusive flux. There are 88 non-finite
values in this group. Later groups carry those values (including 88 in the
pre-limiter antidiffusive flux and 156 in caller advection content), but none
of the exposed groups is non-finite at `[86,159,0]`. The returned target NaN
is created later. `step36_fct.json` is the complete ordered census.

Three refused instrument attempts remain recorded rather than erased. The
first bound the flag outside the step implementation and raised `NameError`.
The second inherited round 130's obsolete `[1,49,0]` target and failed its
target census. The third checker initially demanded an FCT boundary even when
the real measurement falsified that prediction; it was corrected to retain
the falsification and the trajectory was rerun from a clean committed tree.

## Prediction ledger

| ID | Result | Evidence |
|---|---|---|
| R135-P1 | **CONFIRMED** | exact merge parents; sole conflict was the citation-map union; citation audit clean |
| R135-P2 | **CONFIRMED** | rung 0 moved 194/200 and rung 7 moved 195/200; zero exact-row losses |
| R135-P3 | **CONFIRMED** | independent run finite through 35; step-36 T `[86,159,0]` |
| R135-P4 | **CONFIRMED** | every ordinary state leaf and the ordinary replay are bit-identical |
| R135-P5 | **REFUTED** | first global group is averaged upstream flux, not antidiffusive flux |
| R135-P6 | **REFUTED** | the target remains finite through every FCT group; there is no matching target boundary |
| R135-P7 | **REFUTED** | ORCA2 and GYRE trajectories are unchanged by the seam, but the inherited VORTEX exact self-feedback control is red |

## Shared gates, tests, citations, and review

The GYRE certified 954-row ladder is array-identical to round 217: largest
oracle-residual worsening is 0 ULP and first-over-bar remains kt=3 for
T/S/u/v/ssh. The 360-day run is also identical to round 217. T RMS is
`2.3432412624693035e-06` K at day 30, `6.5817073629238215e-05` K at day 240,
and `5.4077381774767389e-05` K at day 360. Snapshot SHA256 values are:

- day 30: `b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180`;
- day 240: `091efc7e588e6011db70294ad57c9bc380852142df6e005d45d897a57f5382fe`;
- day 360: `3fb3e7cc4516bff8a1456054506c146e3ae88c01f443cc1ecbaa0fde07b0b42a`.

The focused FCT/advection battery passes 45/45, and the four repaired incoming
registry/guard tests pass 4/4. The required `tests/ocean/fidelity -n 12`
battery reached 98%, exposed nine failures, and then stalled in its silent
tail; it was interrupted and is **incomplete, not PASS**. Isolation separates
the documented worktree/record ratchets and SI3 provenance red from four stale
merge registries (repaired) and the genuine VORTEX exact-control blocker above.
The blocker is retained in `pytest_vortex_exact_control_red.log`. Once that
first shared red was reproduced, DINO/tank/generic landing batteries were not
claimed: this held private seam must not be shipped from this clone.

The default citation gate passes 274 citations with zero failures, unmapped
citations, or map-audit failures. Shifting the model-file citation plant by
two lines fires. This receipt is gated separately below its first heading.

The required `codex exec --sandbox read-only` review could not initialize:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. Diagnose the merged VORTEX stage-1 RHS self-feedback association and make
   its exact known-answer control pass without changing the two-ULP ratchet.
   No round-135 model seam lands before that shared gate is green.
2. For the independent ORCA2 headline cell `[86,159,0]`, continue immediately
   after caller advection content through the implicit vertical-diffusion and
   later stage-3 program. FCT is finite at that cell and is no longer its
   first candidate.
3. Treat the separate global averaged-upstream-flux overflow at
   `[87,160,5]` as its own magnitude-ranked debt. Walk its `pt_up1`, velocity,
   and carried first-flux operands one variable at a time; do not conflate it
   with the headline cell.
4. Rung 0 remains incomplete and the hierarchy-decks merge/rung-1 climb stays
   deferred. Round 129's salinity-exposing barotropic arm remains held.


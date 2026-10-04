# ORCA2 round 136 — passive tracer-ZDF backsolve walk

Date: 2026-10-04. Base `37bb6855e`; measurement commit `c2a6ea1ec`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round136`.
The step-36 walk is **independent**: it begins with the rung-0 card's own
climatological T/S and zero velocity/sea surface. The two certified ORCA2
ladder comparisons are kept under their existing claim labels and are not
mixed with that population.

## Verdict

**LANDED.** Round 135's invalid cross-program VORTEX self-feedback control is
replaced by a same-graph identity transform. Its identity arm is bit exact and
its `1.0000001` U-RHS plant changes the result, so the control is both exact
and able to fail. The private tracer-ZDF observer is passive across every
ordinary state leaf and its three plants fire.

At the headline step-36 surface cell `[j=86,i=159,k=0]`, the pre-ZDF tracer
content, diffusivity, mesh, tridiagonal matrix, eliminated diagonal, and
forward RHS are all finite. The first non-finite value at that cell is the
solved temperature. The compiled statement that creates it is NEMO's reverse
back-substitution at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:293-299`, called after lateral
mixing by
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749`.

This is propagation, not a new vertical-matrix instability: the same column's
pre-ZDF content is already non-finite at level 3. The next causal walk returns
to that upstream stage-3 FCT/content value; no matrix coefficient, selector,
stabilizer, or carried state changes in this round.

No public configuration field, recipe, card, forcing, sea-ice selector, or
`unmeasured_features` entry changed. The rung-7 one-category ice debt remains
exactly as carried. No acquisition is needed.

## Frozen predictions

| ID | Result | Measurement |
|---|---|---|
| R136-P1 | **CONFIRMED** | same-graph VORTEX identity is bit exact; the U-RHS plant moves the trajectory |
| R136-P2 | **CONFIRMED** | default hooks move 0/200 rung-0 rows, 0/200 rung-7 rows, 0/70 GYRE rows, and 0/50 rows on each tank |
| R136-P3 | **CONFIRMED** | target is finite through pre-ZDF content and all matrix operands; first target non-finite is inside the literal solve |
| R136-P4 | **CONFIRMED** | target forward value is finite; reverse recurrence produces target `nan` |
| R136-P5 | **CONFIRMED** | every ordinary state leaf and the ordinary T/S/u/v/ssh replay are bit-identical; all plants fire |

The first implementation attempt bound the transform in constructor-local
scope and raised `NameError`; it was fixed before measurement and retained in
the focused-test history. The first citation audit also caught one changed
span whose endpoint anchors now covered 93 rather than 81 lines. The map was
corrected mechanically and rerun clean. Neither failed control produced a
scientific number.

## Source-ordered step-36 census

The active compiled path selects `avt` for temperature at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:180-195`, builds the literal matrix
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:218-235`, eliminates the diagonal
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:268-273`, advances the RHS at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:283-291`, and back-substitutes at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:293-299`. The observer reads the
arrays produced by those same recurrences;
it does not reconstruct a second solve.

| Row | Target value | Target finite? | Global non-finite count / first index |
|---|---:|---|---|
| pre-ZDF content | `261.6594625996744` | yes | 90 / `[86,159,3]` |
| heat/effective K | `1.2e-05` | yes | 0 |
| e3t after | `0.005728213959177931` | yes | 0 |
| e3w now | `5.008922785061159` | yes | 0 |
| lower | `-0.0` | yes | 0 |
| diagonal | `0.03160204064034079` | yes | 0 |
| upper | `-0.025873826681162858` | yes | 0 |
| eliminated diagonal | `0.03160204064034079` | yes | 0 |
| forward RHS | `261.6594625996744` | yes | 159 / `[86,159,3]` |
| solved T | `nan` | **no** | 120 / `[86,159,0]` |

The ordered census is `step36_zdf.json`. It reproduces the ordinary returned
failure exactly: T `[86,159,0] = nan`, after 35 finite independent steps.
`observer_state_equal` is true for all 49 state fields, including carried
histories, and the separate ordinary replay is bit exact for T/S/u/v/ssh.
The source-order, passivity, and target-finite plants all return
`STATUS PLANT-FIRED`.

## Shared-card gates

The private diagnostics default off in
`ocean_model_latlon_cgrid.py:1429,1646`; the same-graph transform is applied
only behind its private hook at `ocean_model_latlon_cgrid.py:8143-8147`.
The ZDF trace return is likewise private at
`ocean_model_latlon_cgrid.py:10371-10379` and pairs its diagnostic arrays with
the independently compiled ordinary state at
`ocean_model_latlon_cgrid.py:13177-13191`.

| Gate | Base to tip result |
|---|---|
| ORCA2 rung 0, 200 rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| ORCA2 rung 7, 200 rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| GYRE ten-step, 70 certified rows | array-identical; 0 ULP worsening; first-over-bar remains kt=3 |
| GYRE 30-day member | all 30 daily snapshots byte-identical; day-30 SHA256 `b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180` |
| LOCK_EXCHANGE, 50 rows | 0 ULP worsening; first-over-bar remains kt=8 U |
| OVERFLOW, 50 rows | 0 ULP worsening; first-over-bar remains kt=2 T/U |
| DINO from-rest month | **PASS**, T3D RMS `2.053801168e-03 K` against fixed bar `2.244317642e-03 K` |

The initially consulted round-132 tank baseline was stale: it showed inherited
movement, including a large OVERFLOW delta. Direct runs on the untouched
pre-round tree and the tip prove both tanks are unchanged by round 136. The
DINO planted historical regression (`6.981690958e-03 K`) fails its bar, and
the real round-136 snapshot is byte-identical to round 132.

## Tests, citations, and review

The focused solver/VORTEX/new-gate battery passes `32/32`. The receipt-citation
family passes `158/158`; the prior VORTEX regression pair passes `22/22`.
The default citation gate passes 274 citations with zero failures, unmapped
citations, or map-audit failures, and its shifted-line plant exits nonzero.

The required `tests/ocean/fidelity -n 12` battery reached 97% and then entered
the campaign's known silent xdist tail, so it was interrupted and is
**incomplete, not PASS**. It emitted exactly four failures. Isolated runs
identify all four as the established pre-existing red set: round-35
allow-dirty escape scope, the worktree-stamp grow-only ratchet, the GYRE
round-129 retained-record provenance stamp, and SI3 scalar-math provenance.
No round-136 test failed in isolation.

The required `codex exec --sandbox read-only` review could not initialize:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. At column `[j=86,i=159]`, walk the independent step-36 stage-3 FCT/content
   path to the first statement that makes level `k=3` non-finite. The vertical
   solve only propagates that value to the surface; do not alter its matrix or
   backsolve.
2. Keep round 135's distinct magnitude-ranked averaged-upstream-flux overflow
   at `[87,160,5]` separate until a one-variable operand arm proves whether it
   owns the `[86,159,3]` content failure.
3. Rung 0 remains incomplete. Do not merge the hierarchy-decks lane or climb
   to rung 1 until the independent month is finite through step 240 and rung
   0's first statement is closed under the standing gates.

# ORCA2 round 139 — FCT bound-input walk

Date: 2026-10-04. Base: `3ad32e934`. Measurement producer:
`bf1fd6382f609e350d8cf0952321fa129ee8d7ae`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round139`.
Verdict: **HELD**. This round names the next source input; it changes no ocean
physics, card, deck, selector, carried state, stabilizer, sea-ice field, or
`unmeasured_features` entry.

All step-36 values below are **independent**: rung 0 starts from the card's own
climatological T/S, zero velocity, and zero sea surface. The given-entry rung
0/rung 7 ladder populations are reported separately.

## Frozen ledger

The preregistration was committed as `03a2010a7` before measurement.

| ID | Verdict | Mechanical result |
|---|---|---|
| R139-P1 | CONFIRMED | The observer pairs its payload with separately compiled ordinary outputs; all ordinary FCT outputs and 49 ordinary state leaves are bit-identical, and all eight plants fire. |
| R139-P2 | **REFUTED** | The predicted east member is finite. The first non-finite member is below, at `(87,159,4)`. |
| R139-P3 | CONFIRMED | At the selected below cell, `pbef=8852.366190269584` is finite and `paft=inf`; their maximum reproduces `zbup=inf`. |
| R139-P4 | CONFIRMED | Both ORCA2 ladders and the GYRE short/month trajectories are unchanged. |

## First non-finite bound input

The compiled ORCA2 branch creates the stage-3 upstream provisional field
`zta_up1` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:171`, then passes Kbb as `pbef`
and `zta_up1` as `paft` to `nonosc` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316`. The executed optimized
`nonosc` branch builds each wet `zbup` as `MAX(pbef,paft)` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:816-845`, then consumes the
seven members in centre, west, east, south, north, above, below order at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-856`.

The admitted passive trace gives this source-ordered census at the round-138
`zup` target `(j,i,k)=(87,159,3)`; indices are zero-based legoESM indices.

| Member | Source cell `(j,i,k)` | `zbup` | Non-finite? |
|---|---|---:|---:|
| centre | `(87,159,3)` | `-2452.452140779099` | no |
| west | `(87,158,3)` | `4.544412983209133e+298` | no |
| east | `(87,160,3)` | `1.0137337407966791e+295` | no |
| south | `(86,159,3)` | `1.1843919671449309e+298` | no |
| north | `(88,159,3)` | `-8.988465674311579e+307` | no; dry sentinel |
| above | `(87,159,2)` | `-657.3463479325917` | no |
| **below** | **`(87,159,4)`** | **`inf`** | **yes** |

The selected below cell is wet. Its finite `pbef` and non-finite `paft`
reproduce the selected bound bit-for-bit, the seven-member maximum reproduces
`zup=inf`, and that target `zup` is bit-identical to round 138's independently
compiled beta trace. The first owner is therefore the `paft` provisional
field at `(87,159,4)`, not the east horizontal neighbour predicted in the
preregistration. This names an input, not the arithmetic statement that first
makes it non-finite; the next walk starts inside `fct_up1_2stp`.

## Instrument admission and retractions

The first two runtime attempts refused before stepping because abbreviated or
incorrect full commit stamps were supplied. They emitted no science values.
The admitted run is `step36_bound.json`, stamped to the clean full producer
commit, CPU production JIT, fp64/libm, and 35 completed finite steps.

The first synthetic observer test **REFUTED** a same-JIT-output design: merely
returning the extra stencil arrays changed 54/768 ordinary synthetic FCT
output values, with maximum absolute movement `1.696251e-15`. Those outputs
are rejected. The admitted construction follows the existing private step-hook
pattern: it uses separately compiled ordinary outputs and attaches only the
otherwise-unused trace payload. The payload registry/default-off selector is
at `advection.py:839-873`, the source-ordered stencil readout is isolated at
`advection.py:1225-1242`, and only the private return uses it at
`advection.py:1310-1311`. The final registry, state passivity, ordinary-output,
association, source-order, selected-input, `zup`-link, and payload-separation
plants each emit `STATUS PLANT-FIRED`.

## Shared-card gates

| Gate | Base round 138 to round 139 result |
|---|---|
| ORCA2 rung 0, 200 independent rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| ORCA2 rung 7, 200 given-entry rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| GYRE ten-step, 70 certified rows | 0 ULP worsening; residual archive byte-identical (`7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3`); first-over-bar remains kt=3 |
| GYRE 30-day member | 30/30 daily snapshots byte-identical; day-30 SHA256 `b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180` |

Both ORCA2 ladder failure-mode plants fire: exact-row loss and earlier first
debt. No moved row exists to register. Because the production path and all
required shared trajectories are byte-identical, the DINO and tank
integrations were not repeated for this default-off diagnostic round.

## Tests, citations, and review

The final focused observer/gate/citation battery passes **34/34**. The default
citation gate passes 274 citations and this receipt's gate passes 7 citations,
both with zero failures, unmapped citations, or map-audit failures. The planted
two-line shift of the compiled stencil span refuses with
`SYMBOL-NOT-AT-LINE` as required.

The required `tests/ocean/fidelity -n 12` battery reached 97% and then entered
the campaign's known silent xdist tail, with no live pytest process and no
terminal summary; it was interrupted and is therefore **incomplete, not
PASS**. Before the tail it emitted exactly the same four established reds as
round 138: SI3 scalar-math provenance, round-35 allow-dirty escape scope, the
worktree-stamp grow-only ratchet, and the GYRE round-129 retained-record
provenance stamp. Each failing ID was rerun alone and reproduced its known
failure; no round-139 test failed. All logs are preserved in the evidence
directory.

The required `codex exec --sandbox read-only` review could not initialize:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. At wet cell `(87,159,4)`, walk the compiled `fct_up1_2stp` program in source
   order to the first statement that makes `zta_up1`/`paft` non-finite. Split
   its face fluxes, `ztra`, numerator, and after-thickness divisor without
   changing the independent step-36 protocol.
2. Keep round 135's distinct global averaged-upstream-flux overflow at
   `(87,160,5)` separate unless the next source walk joins it mechanically.
3. Rung 0 remains incomplete. Do not merge the hierarchy-decks lane or climb
   to rung 1 until the independent month is finite through step 240 and rung
   0's first statement is closed under the standing gates.

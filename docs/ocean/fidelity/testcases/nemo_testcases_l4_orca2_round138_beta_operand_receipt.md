# ORCA2 round 138 — adjacent FCT beta operand walk

Date: 2026-10-04. Base: `dac955149`. Measurement producer:
`21aac83018eefad8578376012328fa55b19e4cc8`. Final diagnostic tip:
`610e9fefeae9d0aa1409f9f7d197353af899419b`. Verdict: **HELD**. This
round names the next source statement; it changes no ocean physics, card,
deck, selector, carried state, stabilizer, sea-ice field, or
`unmeasured_features` entry.

All step-36 values below are **independent**: rung 0 starts from the card's own
climatological T/S, zero velocity, and zero sea surface. The given-entry rung
0/rung 7 ladder populations are reported separately.

## Frozen ledger

The preregistration was committed as `19987f90a` before measurement.

| ID | Verdict | Mechanical result |
|---|---|---|
| R138-P1 | CONFIRMED | All 49 ordinary state leaves and the independent T/S/u/v/ssh repeat are bit-identical; every observer plant fires. |
| R138-P2 | **REFUTED** | The predicted first non-finite `zpos`/`zneg` row is preceded by `zup`. The retraction is encoded in the gate's source-order classifier. |
| R138-P3 | CONFIRMED | The non-finite north-cell limiter ratios feed the live positive V-face selection and reproduce its `nan` coefficient. |
| R138-P4 | CONFIRMED | Both ORCA2 ladders and the GYRE short/month trajectories are unchanged. |

## First non-finite compiled statement

NEMO first builds `zbup/zbdo` as the wet-cell `MAX/MIN(pbef,paft)` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:800-845`. It then takes the
seven-neighbour maximum `zup` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-856`, before `zdo`, the
sign-split incident sums, thickness budget, and beta divisions at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:858-878`. The positive north
V-face selects the south-cell `zbetdo` and north-cell `zbetup` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:910-913`.

The admitted passive trace gives the following adjacent-cell census in the
compiled source order. Indices are `(j,i,k)`; south is `[86,159,3]`, north is
`[87,159,3]`.

| Operand | South | North | First non-finite? |
|---|---:|---:|---|
| `zup` | `1.1843919671449309e+298` | `inf` | **yes** |
| `zdo` | `-inf` | `-inf` | later |
| `zpos` | `1.3317808735716745e+291` | `inf` | later |
| `zneg` | `6.012356680852426e+301` | `inf` | later |
| `zbt` | `5076.323419528505` | `5462.230637513799` | no |
| literal `zbetup` | `0.0` | `nan` | later |
| literal `zbetdo` | `inf` | `nan` | later |
| live `r_in` | `0.0` | `nan` | later |
| live `r_out` | `1.0` | `nan` | later |

The incident antidiffusive V flux is positive
(`6.012356680224643e+301`), so the live pair is exactly
`[r_out_south, r_in_north] = [1.0, nan]`; the resulting face coefficient is
`nan`. The ordinary returned failure remains T `[86,159,0] = nan` after 35
finite independent steps. This names the first statement, not its input
owner: round 139 must split the seven `zbup` inputs at the north cell and then
the selected input's `pbef`/`paft` pair.

The trace implementation is default-off and mutually exclusive with the
existing diagnostics at `advection.py:958-970`. Its duplicated beta arithmetic
is protected from the live limiter graph at `advection.py:1206-1221`; the live
production fluxes still use the original coefficients at
`advection.py:1223-1228`. The source-aligned readout is assembled only in the
private branch at `advection.py:1741-1764`, and its return is gated at
`advection.py:1819-1820`. The model pairs the private FCT payload with an
independently compiled ordinary state at
`ocean_model_latlon_cgrid.py:13190-13205`.

## Instrument admissions and retractions

The first attempted run refused before stepping because its expected commit
stamp was abbreviated. A later run produced valid arrays but its checker
incorrectly expected only five observer leaves; the observer actually retained
all 49 ordinary leaves. No science number is cited from either refusal. The
checker was corrected, and the admitted run is
`step36_beta.json` with `STATUS PASS_ROUND138_BETA_OPERAND_WALK`.

The initial source-order plant accidentally planted the value the real run
already carried (`zup`) and therefore was vacuous. It was replaced by an
impossible sentinel. The final registry, passivity, coefficient-support,
adjacency, source-order, and live-selection plants each emit
`STATUS PLANT-FIRED`. This retraction lives in the gate and its unit test.

The beta readout is admitted for the finite/non-finite boundary and source
ordering. Its extra arithmetic is not claimed bit-identical to the ordinary
coefficient path; the gate requires identical finite/non-finite masks for all
three coefficient fields, while ordinary state passivity is bit-exact.

## Shared-card gates

| Gate | Base `dac955149` to tip result |
|---|---|
| ORCA2 rung 0, 200 given-entry rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| ORCA2 rung 7, 200 given-entry rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| GYRE ten-step, 70 certified rows | 0 ULP worsening; residual archive array/byte-identical (`7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3`); first-over-bar remains kt=3 |
| GYRE 30-day member | 30/30 daily snapshots byte-identical; day-30 SHA256 `b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180` |

Both ORCA2 ladder failure-mode plants fire: exact-row loss and earlier first
debt. No moved row exists to register. Because every production path is
unchanged and the required shared GYRE gate is byte-identical, the DINO and
tank integrations were not repeated for this default-off diagnostic round.

## Tests, citations, and review

The final focused beta observer/gate/citation battery passes `25/25`. The
default citation gate passes 274 citations and the round receipt passes 10,
both with zero failures, unmapped citations, or map-audit failures. The planted
two-line shift fails on the first `zup` endpoint as required. A SequenceMatcher
old-to-new line map was applied once to every default-receipt citation of both
edited model files; all 18 cited `ocean_model_latlon_cgrid.py` spans precede
the insertion and therefore map to themselves, while that receipt contains no
`advection.py` citation. The audit is preserved as
`default_receipt_reanchor.log`.

The required `tests/ocean/fidelity -n 12` battery reached 95% and then entered
the campaign's known silent xdist tail, so it was interrupted and is
**incomplete, not PASS**. It emitted exactly the same four established reds as
round 136: SI3 scalar-math provenance, round-35 allow-dirty escape scope, the
worktree-stamp grow-only ratchet, and the GYRE round-129 retained-record
provenance stamp. Each failing ID was rerun alone and reproduced its known
failure; no round-138 test failed. All logs are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round138/`.

The required `codex exec --sandbox read-only` review could not initialize:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. At the north adjacent cell `[87,159,3]`, expose the seven source-ordered
   `zbup` inputs consumed by `zup`; identify the first non-finite stencil member
   and split its `pbef` versus `paft` input without changing the walk order.
2. Keep round 135's distinct averaged-upstream-flux overflow at `[87,160,5]`
   separate until an explicit stencil/input association joins it mechanically.
3. Rung 0 remains incomplete. Do not merge the hierarchy-decks lane or climb
   to rung 1 until the independent month is finite through step 240 and rung
   0's first statement is closed under the standing gates.

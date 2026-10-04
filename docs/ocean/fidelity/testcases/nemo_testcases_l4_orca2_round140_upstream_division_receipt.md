# ORCA2 round 140 — upstream-predictor division boundary

Date: 2026-10-04. Base: `41b927214f679245e7bc74c027d4690da083ea20`.
Raw-trace producer: `614dbb83255d192868c5abd1f8900ddbc921a2d9`.
EVD producer: `da597f57768311edb3f29b8b5da1a00f8e10bbd6`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round140`.
Verdict: **HELD**. The first non-finite statement is named; no ocean physics,
card, deck, selector, carried state, stabilizer, sea-ice field, or
`unmeasured_features` entry changes.

Every step-36 number below is **independent**: rung 0 starts from its own
climatological T/S, zero velocity, and zero sea surface. Given-entry rung-7
ladder populations remain separate.

## Frozen ledger

The preregistration was committed as `452926467` before measurement.

| ID | Verdict | Mechanical result |
|---|---|---|
| R140-P1 | CONFIRMED | The separately compiled ordinary FCT outputs and all 49 ordinary state leaves are bit-identical; every raw-trace gate control fires. |
| R140-P2 | **REFUTED** | Every raw first/midpoint/averaged face operand is finite at `(87,159,4)`. The apparent `average_w` infinity came only from the older diagnostic's area scaling. |
| R140-P3 | **REFUTED** | Both the numerator and divisor are finite; their quotient overflows. |
| R140-P4 | **REFUTED** | `rn_evd/K_conv × 10^4` moves `avt` on 13,061/772,560 interfaces by at most 999,900 m²/s. `avm` is bit-identical because this deck has tracer-only EVD. |
| R140-P5 | CONFIRMED | Both ORCA2 ladders, GYRE ten-step, and GYRE 30-day snapshots are unchanged. |

## First non-finite statement

NEMO's compiled two-step upstream program builds the first face fluxes,
midpoint, averaged face fluxes, divergence, and after tracer in that order at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609`. At the wet T cell
`(j,i,k)=(87,159,4)`, all raw intermediates through the numerator are finite:

| Boundary | Value |
|---|---:|
| first divergence | `-2.568708505036104e152` |
| midpoint `pt_up1` | `-2.776470255657409e155` |
| averaged east/west U | `7.273261746784698e299`, `-5.050132297123146e300` |
| averaged top/bottom W | `-3.263935505533791e302`, `3.2065892466934766e303` |
| explicit / total `ztra` | `3.532982888093828e303` / same |
| implicit `ztra` | `-0.0` |
| `dt*ztra` and numerator | `3.815621519141334e307` |
| after thickness | `0.00573471208449003 m` |
| quotient / `paft` | `inf` |

The raw `paft` is bit-identical to the separately compiled round-139 stencil
trace. Therefore the first non-finite arithmetic is the compiled division in
the final `pt_up1` assignment, not any face-flux statement. This names the
boundary; it does not claim the division is mistranscribed. NEMO runs stably,
so the huge finite carried numerator remains the upstream debt.

### Retraction

The first measurement reported `average_w=-inf/+inf` because the existing
observer converts legoESM's normalized vertical flux back to NEMO units by
multiplying by cell area. The raw fluxes actually consumed by the divergence
are finite. That observer-created infinity is rejected and the retraction is
encoded in the final JSON. NEMO's actual second-step vertical averaging is the
finite arithmetic at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:577-578`.

## Rung-0 EVD read-out

The instantiated rung-0 card prints `scheme=enhanced_diffusion` and
`K_conv=100.0 m²/s`. The compiled deck dispatches EVD at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359`. Its tracer arm replaces `avt`
where the two-time-level N² trigger fires at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109`; the separately guarded
momentum arm is at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121-135`.

Multiplying only `K_conv` by `10^4` moves `avt` on 13,061/772,560 interfaces
and moves `avm` on 0/772,560. Thus ORCA2 rung 0 does execute tracer EVD; the
seamount card's inertness does not transfer to this card. Decision 94 remains
pending, so this round changes neither EVD routing nor its EOS trigger.

## Instruments and shared gates

The raw predictor observer is default-off, mutually exclusive with every
other private return, and paired with separately compiled ordinary outputs.
Its synthetic test proves the payload responds to a one-ULP transport plant.
The gate refuses registry, passivity, source-order, round-139-link, and
finite-prefix plants. The EVD gate refuses selector, coefficient, and
profile/census inconsistency plants.

| Gate | Result |
|---|---|
| ORCA2 rung 0, 200 independent rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| ORCA2 rung 7, 200 given-entry rows | 0 moved; no exact-row loss; first debt remains kt=1 stage-1 T |
| GYRE ten-step, 70 certified rows | 0 ULP worsening; first-over-bar remains kt=3 |
| GYRE 30-day member | 30/30 daily snapshots byte-identical; day-30 SHA256 `b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180` |

Both ORCA2 ladder plants fire: bit-identical-row loss and earlier first debt.
Because the production path and required shared trajectories are unchanged,
DINO and tank integrations were not repeated for this default-off diagnostic.

## Tests, citations, and review

The default receipt has no `advection.py` citation. A SequenceMatcher
old-to-new line map nevertheless re-anchored all eight affected citation-map
entries and their round-138/139 receipt spans; no rigid shift was used. The
round's citation gate and focused tests are recorded in the evidence directory.

The required `codex exec --sandbox read-only` review could not initialize:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. Walk the finite-magnitude growth that produces the step-36 numerator:
   compare the step-35 entry transports/tracers and the midpoint/averaged
   vertical flux against NEMO, in source order. Do not clip the quotient or
   add a stabilizer.
2. Keep Decision 94 pending. ORCA2 proves tracer EVD is active, but this round
   does not test or change the card-EOS trigger discrepancy identified by the
   seamount lane.
3. Rung 0 remains incomplete. Do not merge the hierarchy-decks lane or climb
   to rung 1 until the independent month is finite through step 240.

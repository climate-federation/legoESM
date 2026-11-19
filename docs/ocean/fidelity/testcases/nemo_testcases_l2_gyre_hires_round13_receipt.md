# NEMO testcase L2 GYRE high resolution — round 13 receipt

Date: 2026-10-18

Incoming lane tip: `e4e0f9f694085c1d3e73762763356bccc60cae77`.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round13/`.
The acquired legoESM run is commit `e4e0f9f694085c1d3e73762763356bccc60cae77`,
native fp64 GPU policy, physical GPU 0, per-step integration, days 1–180.
The amended admission instrument is commit
`5b24489a09059ff859c786955bfc720bdd396038`.

## Verdict

**The daily-state archives pass, but the registered term attribution is
UNMEASURED.  No momentum owner and no budget closure can be reported from
these records.**  Neither model stores the online production terms required
by the round-12 question.  Re-evaluating nonlinear operators on daily means
would answer a different question, and the split external-mode correction
and implicit vertical-mixing/surface-stress responses cannot be reconstructed
from the archived providers.

This corrects round 12's statement that matched full-state daily means would
“admit the registered term attribution.”  They admit a state comparison and
are useful inputs to future diagnostics; they do not admit a production
tendency budget.

## Finding 1 — the requested archive is complete as a state archive

The amended preflight reports `STATE_ARCHIVE_PASS`:

| provider | measured result |
|---|---|
| legoESM | manifest complete; 180 days; per-step mode; full-state window 1–180; all 12 registered snapshot/mean keys present and finite on sampled days 1, 30, and 180 |
| NEMO vector reference | 8 tiles for each of T/U/V/W; 1,825 daily records per tile; all registered state variables and DOMAIN placement attributes present |

The legoESM manifest hash is
`edfdffbb6c4fc9310fcbcfae5de0fcf15cfc0cf439e35103ffd394790b907b3b`.
Sample hashes are `cc6f2dbe...` (day 1), `24a672a2...` (day 30), and
`96fafc74...` (day 180).  The NEMO manifest hash is
`eeaef7ea9b7ffc48b27230208c1eda93c67a63fc99fc62fa38357f8269c4e509`.

## Finding 2 — the production budget is not observable

The same audit reports `BUDGET_OBSERVABILITY_REFUSE`:

| check | result |
|---|---|
| legoESM online term variables | 0 of 8 registered rows |
| NEMO online `utrd_*` / `vtrd_*` variables | 0 |
| NEMO trend flags | `ln_glo_trd`, `ln_dyn_trd`, `ln_KE_trd`, and `ln_vor_trd` are all false |
| legoESM carried implicit-ZDF providers | `tke_avm`, `tke_avt`, `tke_dissl`, and `tke_avm_surface` absent |
| online contracts | absent for both models |
| owner / closure | `UNMEASURED` / `UNMEASURED` |

The 33 legoESM archive keys include prognostic TKE but not the carried
coefficients used by the production implicit solve.  NEMO exposes 31 unique
state/output variables and no tendency variable.  More fundamentally, the
NEMO RK3 program computes its split external solve before the three stages;
stage 3 then applies lateral diffusion and the implicit vertical solve before
imposing the barotropic correction.  The ordinary surface-pressure trend
hook surrounds a different wrapper, not the direct split-explicit call used
by this card.  The vertical diagnostic, when enabled, is formed after the
implicit solve in which wind stress is already deposited, so it does not
provide the requested independent vertical-mixing and stress rows.

The production build has no XIOS key.  Its `iom_put` overloads in this branch
are dummy routines, while the active IOIPSL writer emits state, stress, W,
`avt`, and `avm` fields rather than the production increments.  Merely setting
the trend flags would therefore neither create the required output in this
build nor repair the missing split-mode and implicit-term separation.

## Finding 3 — daily means fail the nonlinear identifiability control

The planted pair uses velocity samples `[-1, 1]` and two gradient sequences
`[-1, 1]` and `[1, -1]`.  Both archived provider means are `[0, 0]`, but the
mean products are +1 and −1: ambiguity 2.0.  Thus daily means cannot determine
the daily mean of even the simplest advective product.  Smagorinsky
coefficients and the implicit solve add further nonlinear and history
dependence.  This control fires in the committed unit test and in the evidence
artifact.

## Budget table and closure

No numeric term table is emitted.  Filling it by applying legoESM operators to
both models' daily means would measure a common offline diagnostic rather than
the two production programs; assigning the unmatched state increment to a
“residual” would hide the missing external and implicit boundaries.  The
registered closure bar was also described as “floating-point” in round 12 but
never assigned a numeric value.  Choosing one after seeing the acquired
states would be post-hoc.  Accordingly:

| requested result | round-13 result |
|---|---|
| term owner(s) | `UNMEASURED` |
| summed term difference | `UNMEASURED` |
| independently recorded total difference | absent |
| closure residual | `UNMEASURED` |
| shared-path fix | none |

## Disposition and exact next record

No shared-path change is warranted.  The next record must be a matched online
surface-V production budget over days 30–180 for both trajectories.  Each
record must contain the eight registered term fields, an independent total,
and a passive-identity artifact.  The acquisition must observe all three RK
stages, the direct split-explicit external solve and final barotropic
correction, and must decompose the linear implicit solve into vertical-mixing
and surface-stress responses.  Its manifest must hash both artifacts and pin
the numeric closure bar before scoring.

There is deliberately no operator launch command in this receipt: no such
certified writer exists at this tip, and another state-only integration would
not answer the question.  The next engineering unit is to implement and
prove the two passive online writers; only that unit can issue an exact NEMO
and GPU acquisition command.  This is an instrument blocker, not a request
for the operator to rerun either existing executable.

## Inherited pins / OPEN

Inherited unchanged: the shipped one-degree digest `99b2dfce324b0f77`, the
one-degree no-worsening ceiling, the resolved `r4_smag` deck, 31 levels, RK3,
split-explicit free surface, TKE/EVD, Redi tracers, the 360-day forcing clock,
scalar-libm CPU ladder certification, thickness-weighted NEMO daily T/S, and
explicit physical-GPU selection.  This round changes no physical kernel and
therefore does not warrant another five-year climate run.

OPEN, in order:

1. Implement paired online production-budget writers with passive trajectory
   controls and pre-register a numeric closure bar.
2. Acquire days 1–180 on the exact `r4_smag` production trajectories and score
   days 30–180.
3. Resume the inherited scalar entry-`kt=3` SSH localization after this
   higher-priority long-run attribution is observable.
4. Post the receipt to issue #1455 when network access is available.

## Tests and controls

The focused preflight suite passes 6/6 before the final lane battery.  It
covers missing/non-finite state providers, a refusing joint provider, the
state-pass/budget-refuse regression, a positive synthetic online contract,
plants in every semantic contract field and all three artifact digests, and the
nonlinear daily-mean ambiguity control.  A correctly hashed, active one-unit
term that does not close the independent total is also planted and refused.
Final battery and citation-gate
counts are:

`FULL_LANE_BATTERY_SPLIT_PASS: 133 passed, 4 deselected across 12 pytest processes`

That battery includes the 70-row shipped one-degree no-worsening ceiling and
the card, phase-3, Smagorinsky, acquisition, 30-day, climate, divergence,
movie, driver, and both high-resolution ladder suites.  The amended preflight
evidence is `momentum_budget_identifiability.json`, SHA-256
`908e079ab71c6e65d99764fd2c4f0cd909e0fb1675cb732c0ce8b1bdf65c03df`;
exit status 3 is its registered `ACQUISITION_NEEDED` result.  The explicit
receipt citation gate reports PASS: nine citations, zero unmapped citations,
zero failures, zero failing map entries, and all 9/9 self-tests firing.  Its
artifact `receipt_citation_gate.json` has SHA-256
`e2e431b1e2ab653e407f2b7dc31410cc3dc4490ebda034def8f28448f7f63d47`;
the citation-gate unit suite passes 17/17.

## Compiled-source citations

The executed production program updates the time-varying viscosity, performs
the split external solve, and then runs all three RK stages at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:183-210`.  Stages 2 and 3 accumulate
pressure, vorticity, and advection at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:319-334`; stage 3 then adds lateral
diffusion, calls the implicit vertical solve, and imposes the barotropic
correction at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:387-418`.

The V implicit recurrence deposits `vtauV` in its solved surface right-hand
side and only afterward forms the combined vertical diagnostic at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/dynzdf.f90:500-528`.  The card calls the
split-explicit routine directly at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90:260-267`, whereas the surface-pressure
trend hook surrounds the separate wrapper at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg.f90:231-240`.

The trend initializer both derives its runtime switches and warns that the
RK3 trend diagnostics are not fully functional at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/trdini.f90:127-138`.  In this no-XIOS build the
two- and three-dimensional `iom_put` implementations are dummy paths at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/iom.f90:1468-1506`; the IOIPSL writer instead
emits stress, U/V/W, `avt`, and `avm` state providers at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/diawri.f90:524-539`.

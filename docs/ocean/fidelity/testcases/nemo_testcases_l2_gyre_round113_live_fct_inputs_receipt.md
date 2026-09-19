# NEMO-testcases L2 GYRE round 113 receipt: live FCT inputs

Date: 2026-09-18

Incoming lane tip: `950c5787a`

Preregistration commit: `b9e175a74`

Round status: **HELD — no physics or configuration changed; the first
consumed non-bit FCT input is the stage-3 U/V/W transport family, whose
oracle substitution reduces local kt3 T by 14.92x but leaves a non-bit
upstream boundary**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round113/`

## Outcome first

Round 113 closes the two inherited control debts before continuing the live
input walk.  The six broken Round-112 compiled-source citations are repaired;
the citation gate passes and its shifted-line plant exits nonzero.  The
Round-110 generic `build_nemo_gyre_recipe()` blast radius is now measured at
the landing's exact base `51a4d088c` and at a descendant: both arms pass every
existing three-step certification, and all 15 step/field rows are registered.
The Round-110 receipt is amended in place.

The scientific result is upstream of the held Round-112 FCT association
patch.  On the 18,000 consumed wet tracer cells, Kbb T/S and Kbb thickness are
BIT.  Masks, reciprocal area, and `p2dt` are also BIT.  The first non-bit input
family is the metric-bearing U/V/W transport triplet: 17,400 / 17,100 / 17,400
consumed faces differ.  Kmm thickness is later and also non-bit in 18,000
cells, with maximum `2.4725977709749714e-8 m`.

Replacing only the complete transport triplet by the admitted NEMO values
inside the full production-jitted step changes the direct `adv_up1` result as
follows:

| tracer | ordinary live input | NEMO transport family | result |
|---|---:|---:|---|
| T | 18,000 unequal, max `6.545655547452618e-11 s-1` | 18,000 unequal, max `1.632720658269341e-16 s-1` | 400,901x smaller, not BIT |
| S | 18,000 unequal, max `5.680531218351438e-12 s-1` | 18,000 unequal, max `2.580811319750429e-16 s-1` | 22,010x smaller, not BIT |
| local kt3 T | 17,999 unequal, max `8.600420500215478e-7 K` | 18,000 unequal, max `5.763797261693071e-8 K` | 14.922x smaller |
| local kt3 S | 17,265 unequal, max `6.979443156751586e-8 psu` | 18,000 unequal, max `6.599790935979399e-9 psu` | 10.575x smaller |

This is a causal magnitude attribution to the transport family, not a landing
candidate.  The arm injects oracle values and therefore cannot run models
independently.  It also does not close the directly recorded boundary.  No
production source is changed, no ladder/month arm is created, and the held
Round-112 downstream patch remains held.  A one-ULP p_u plant changes both the
injected transport and the consumed production-JIT T boundary, prints
`STATUS PLANT-FIRED`, and exits 1.

## Prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round113.md`.

| prediction | disposition | evidence |
|---|---|---|
| Generic card executes the Round-110 route; T/S move at step 1; certifications stay passing | **CONFIRMED** | Recipe-derived execution says yes; both three-step arms pass all five assertions; step-1 T/S move in 17,831/12,370 cells. |
| Generic step-1 u/v do not move | **REFUTED** | u/v move in 360/311 last-bit cells, each at maximum `2.168404344971009e-19`; eta is BIT. |
| Exact moved-row registry and executed-route tests are non-vacuous | **CONFIRMED** | The missing-row plant exits 1.  Disabling the real production source guard makes the executed-path test fail; restoring `len(moved)>0` makes the missing-row test fail. |
| Kbb base is BIT and U transport is the first consumed non-bit live input | **CONFIRMED** | T/S bases are 0/18,000 unequal; U is 17,400/17,400 unequal.  Full-domain base diagnostics retain the 3,120 dry representation differences separately. |
| Transport-family substitution improves T `adv_up1` at least 2x but does not close both tracers | **CONFIRMED** | T improves 400,901x; T and S each retain 18,000 unequal wet words. |
| Transport substitution does not improve kt3 T by the preregistered material 10% | **REFUTED** | kt3 T maximum falls 93.30%, from `8.600420500215478e-7` to `5.763797261693071e-8 K`. |

The material-kt3 falsifier fired.  It promotes the transport producer to the
next magnitude walk; it does not promote recorded-value injection to a model
change.

## Compiled source and first non-bit producer statement

The record's compiled active tracer branch supplies Kbb tracer, U/V/W
transport, and live RHS to the two-step FCT routine at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:167-176`.

In producer order, NEMO first materializes the horizontal U and V transports.
The first non-bit producer statement is the U write, followed by its V twin,
at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:295-296`.
It later derives `ww` from those transports and writes metric-bearing `zFw`
at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:326-346`.
The stage passes the same triplet to tracer advection at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:860`.

This round names the first output statement and its causal magnitude.  It does
not yet choose among the U statement's factors (metric, Kmm face thickness,
Kmm velocity, and barotropic correction); doing so from the output alone
would be post-hoc attribution.  That operand walk is the first OPEN item.

## Live-input tables

The admitted record is
`round111/oracle_fct_writers/oracle_fct_writers_kt00000002_s3.bin`, SHA-256
`157a5a0ec7cd8607c5c5ee924f788926db5cff5c62e969b853a42f623310505f`.
The qualifying artifact is
`round113/live_input_walk/live_inputs_consumed.json`, stamped at clean commit
`755be7b09bc0655b3954933c34a347d3fc9d7e80`.  Each entry below is consumed
cells unequal / maximum absolute difference.  Common inputs are identical for
the T and S calls.

| input in consumer order | production JIT | production eager | disposition |
|---|---:|---:|---|
| Kbb T | 0 / `0` | 0 / `0` | BIT |
| Kbb S | 0 / `0` | 0 / `0` | BIT |
| U transport | 17,400 / `8.109319272585708` | 17,400 / `8.109319233513361` | first non-BIT family |
| V transport | 17,100 / `10.850152134284144` | 17,100 / `10.850152106128007` | non-BIT |
| W transport | 17,400 / `23.680200024275109` | 17,400 / `23.680200231814524` | non-BIT |
| Kbb thickness | 0 / `0` | 0 / `0` | BIT; 3,120 dry representation differences retained in JSON |
| Kmm thickness | 18,000 / `2.4725977709749714e-8` | same | later non-BIT family |
| tmask | 0 / `0` | 0 / `0` | BIT |
| wmask | 0 / `0` | 0 / `0` | BIT |
| reciprocal area | 0 / `0` | 0 / `0` | BIT |
| p2dt | 0 / `0` | 0 / `0` | BIT |
| T direct `adv_up1` | 18,000 / `6.545655547452618e-11` | 18,000 / `6.545655683512579e-11` | production JIT governs |
| S direct `adv_up1` | 18,000 / `5.680531218351438e-12` | 18,000 / `5.680531483009529e-12` | production JIT governs |

The full-domain dry diagnostics are not discarded: base and Kbb thickness
each differ in exactly 3,120 dry cells because legoESM applies its wall fill
or zero dry thickness while NEMO retains reference values before masking.
They are excluded only from the first-consumed-input ordering, because the
calibrated tmask/wmask rows are BIT and the compiled fluxes do not consume
those dry values.

The production-JIT one-ULP artifact is
`round113/live_input_walk/live_inputs_plant.json`.  Its report status is
`PLANT-FIRED`; both `transport_u_sha256` and `adv_up1_T_sha256` change, and
the process exits 1.

## Round-110 amendment and non-vacuity closure

The generic-card comparison is
`round113/generic_card/comparison.json`.  Fourteen of 15 exact arrays move;
step-1 eta is the sole BIT row.  The largest step-3 changes are T
`3.828073217349015e-5 K`, S `7.849097940493266e-7 psu`, u
`8.495948000675213e-8 m s-1`, v `7.475950647428675e-8 m s-1`, and eta
`1.525691859090251e-7 m`.  Both arms retain identical PASS certification
booleans.  Therefore no certified number worsens and no re-baselining decision
is requested.

The Decision-43 re-audit uses the explicit 53-row Round-110 registry and names
the generic card as separately measured.  It prints the original one-variable
month result:

```text
STATUS PASS: moved_rows=53 day30_T=1.23970112963527369e-02->6.89048490148956762e-05
```

The missing-row registry plant prints `STATUS PLANT-FIRED` and exits 1.
Guard-removal logs under `round113/nonvacuity/` show the executed-route test
and exact-registry test each fail for their intended reason.

The amended Round-110 receipt also binds the before arm to
`round110/before_same_tip_51a4d088c/`, which the operator found bit-identical
to the older arm across all 50 ladder rows, 210 arrays, and 30 days.  It records
the day-23 spike as physics: T/S/u/v co-spike in the upper 0--1000 m while ssh
remains on trend.

## Decision-43 headline and scope

No production physics changed in Round 113, so there is no candidate arm to
admit and the immutable Round-110 landing remains both before and after:

| required row | before | after | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | same | AT-BAR |
| kt2 S | `2.1316282072803006e-14` | same | AT-BAR |
| kt2 U | `2.7377110452773967e-12` | same | first-over-bar DEBT |
| kt2 V | `3.2849219221489645e-12` | same | first-over-bar DEBT |
| kt3 T | `8.600419718618468e-7` | same | DEBT |
| kt3 S | `6.979441735666114e-8` | same | DEBT |
| day-30 T RMS | `6.890484901489568e-5 K` | same | no production candidate |

The transport producer is shared by WS-RK3 cards.  A future implementation
change must therefore measure GYRE-zco and the generic NEMO-GYRE recipe, and
must resolve whether LOCK_EXCHANGE/OVERFLOW execute the changed exact branch.
The two named DINO recipes use the Euler tracer lane and do not execute this
WS stage statement; no DINO number can move in this diagnostic-only round.

ORCA2 remains **UNMEASURED-WITH-SPEC**.  Before any transport claim, resolve
its integrator and transport branch, acquire or admit its stage-3 U/V/W
transport and factor operands, score the same consumed-face table, and run its
certified trajectory gate.

## Review, citations, and tests

Independent review verdict: **PENDING FINAL PASS**.

Citation gate, shifted plant, focused tests, and complete-tree summaries are
recorded after their final runs below.

## ASKED / UNASKED

ASKED and completed: inherited citation repair; exact four-file gate;
generic-card before/after measurement and Round-110 amendment; exact moved-row
registry and executed-path non-vacuity mutants; production JIT/eager live-input
table; first consumed non-bit family; full-step one-family substitution;
production-JIT plant; compiled-source attribution; and explicit next-round
scope.

UNASKED and not done: no NEMO source was modified; `makenemo` and `mpirun`
were not run; no oracle record was acquired; no production physics,
configuration, coefficient, carried state, restart schema, year harness,
reconciliation gate, freshwater pair, or #1484 guard changed; no held patch
was applied.

## OPEN for round 114

1. Stay at the first non-bit producer statements, compiled
   `stprk3_stg.f90:295-296`.  Split the transport family in producer order (U,
   then V, then W) through the production-jitted step so its kt3 magnitude is
   not inferred from a bundled triplet.
2. For U, score the statement's factors in written order: e2u, Kmm e3u/r3u
   face thickness, Kmm `uu`, and `zub` (including `un_adv`, inverse depth, and
   carried `uu_b`).  Reuse the admitted Round-46/77 transport and stage records
   where they contain the operand; request a new NEMO acquisition only if a
   required live operand is genuinely absent.
3. Substitute only the first non-bit operand family through the complete
   production-JIT statement and report direct zFu, `adv_up1`, kt3 T/S, and the
   plant.  Do not revive the held Round-112 downstream association patch.
4. An implementable source-exact candidate advances only under Decision 43:
   same-base full ladder/month, exact moved-row registry, no earlier
   first-over-bar or kt1 AT-BAR loss, and every recipe-derived executing card
   measured.
5. Keep ORCA2 UNMEASURED until the explicit spec above is run.

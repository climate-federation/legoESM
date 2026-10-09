# Preregistration: NEMO-testcases L2 GYRE round 71 stage-3 FCT Kmm inputs

Date: 2026-09-12. Frozen before adding the round-71 measurement mode, running
any round-71 arm, or preparing a new oracle acquisition source card.

## Ranked boundary and immutable records

Round 70 confirmed a large but rejected cancellation between the complete FCT
advection output and the stage-3 LDF source association. The output-paired arm
reduced the kt3 temperature maximum from `8.916073106490785e-7 K` for LDF-only
to `5.760972143775689e-8 K`, but retained 1,375 of LDF-only's worsened
temperature cells and 1,891 salinity cells; it also newly worsened 1,800
temperature and 6,374 salinity cells. No production physics was changed.

The first non-bit boundary therefore remains kt3-before T/S. In compiled
argument order the first already-known unequal inputs to complete stage-3 FCT
are its Kmm tracers: T differs in 17,994 of 18,000 wet cells, maximum
`8.369461071e-7 K`, and S in 16,769 cells, maximum `6.794565621e-8`. Kbb T/S
are exact. Transport and metric inputs occur later in this walk and must not be
tested before this Kmm boundary is adjudicated.

The admitted records remain the round-46 kt2 stage-3 operand stream under
`round46/oracle_kt2_stage` and the round-64 Krhs split
`round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin`, produced by
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`. Admission must retain 43 of 63
inherited records exact, 20 classified changed, and all 132 consumed values
admitted. The round-46 header must say kt=2, stage=3, Kbb=3, Kmm=2, Krhs=1,
Kaa=1, and fp64.

## Compiled statements and the source walk

The compiled RK driver makes stage 2's Kaa become stage 3's Kmm at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:204-215`.
The active stage program constructs stage-1/2 tracer RHS values and writes Kaa
from Kbb plus the Kmm-weighted Krhs at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:824-903`, then calls
stage-3 advection with those Kmm tracers at `:856-865`.

The active FCT routine receives `pt(:,:,:,jn,Kbb)` plus the Kmm time-level
index at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:165-172`. Its
first-step upstream faces read Kbb at `:497-510`, its first guess writes Kmm
at `:532-540`, and the second-step and centred faces first read the supplied
Kmm tracer at `:564-611` and `:197-200,264-268`. The limiter consumes Kbb and
the first guess at `:308-318`, and the complete corrected divergence is added
to Krhs at `:320-331`. Thus substituting the recorded Kmm tracer is a causal
input test, not an eligible implementation and not an oracle-output fix.

legoESM's single WS helper receives its stage-2 tracer through `resume`, then
passes it to the shared FCT pair before associating stage-3 sources. Extend the
existing round-67--70 instrument by intercepting only that final helper call.
Do not add a second FCT implementation or a public/private production hook.

## Frozen arm matrix and falsifiers

Starting from the identical oracle-seeded kt2 state, run these JIT production
arms with identical card, forcing, geometry, transports, sources, and LDF:

1. untouched: live Kmm T/S, no LDF route;
2. LDF-only: live Kmm T/S plus the same-step live LDF route;
3. output-paired control: round 70's admitted complete-FCT output plus LDF;
4. Kmm-T: recorded Kmm T and live Kmm S plus LDF;
5. Kmm-S: live Kmm T and recorded Kmm S plus LDF;
6. Kmm-TS: both recorded Kmm tracers plus LDF.

The untouched arm must be bit-identical to an ordinary separately constructed
model in every pytree leaf. Untouched, LDF-only, and output-paired metrics must
reproduce round 70 exactly. Every routed arm must capture LDF arrays
bit-identical to untouched. Kmm-T may move only T content/state relative to
LDF-only, Kmm-S only S, and Kmm-TS must equal Kmm-T for every T row and Kmm-S
for every S row. The injected arrays must equal the admitted owned Kmm arrays
bit-for-bit. All live, injected, geometry, and record arrays must print fp64.

Prediction: the Kmm-TS arm will reduce the LDF-only kt3 maximum by at least 2x
for both tracers and clear at least half of the inherited 1,375 T and 1,891 S
retained-cell sets. It will not be complete: its kt3 maximum will remain larger
than the output-paired control (`5.760972143775689e-8 K` T and
`6.410431296899333e-9` S), and at least one cell in each retained set will
remain worsened by more than two row-scale float64 ULPs. Rank the candidate by
kt3 T maximum first; no global content norm may waive a cellwise failure.

CONFIRM the Kmm-ownership prediction only if admission, all exact controls,
the 2x maxima, both 50% retained-cell reductions, and the predicted remaining
debt hold. Any failed exact control invalidates the instrument. A valid run
that misses a magnitude/cell prediction records it as REFUTED and keeps the
measured split.

Plant 1 substitutes the live Kmm arrays as the alleged oracle target. Kmm-TS
must then equal LDF-only and the required movement predicate must make the
command exit nonzero. Plant 2 moves exactly one wet recorded Kmm-T cell by one
`nextafter` ULP after record admission. The injection-target exact check must
fail and the command must exit nonzero. Both plant reports remain evidence.

## Source-statement stopping rule and acquisition

The round-46 kt2 stage-3 record exposes Kmm T/S but not the kt2 stage-1/2
tracer Krhs boundaries that produced them; the existing `rktracer_operands`
files are restricted by their compiled writer to `kstp == nit000`. If the
Kmm-TS arm moves the ranked residual and the existing records cannot replay
the stage-2 statement at compiled lines 899--901, stop for a new record. The
acquisition card must record, for kt2 stages 1 and 2, the incoming Kbb/Kmm
tracers, post-advection/SBC Krhs, post-update Kaa tracers, all three r3t
levels, and the zFu/zFv/zFw values actually consumed. It must use a new GYRE
target, additive WRITE-only patches, fail-closed header/EOF/stamp/admission
checks, and the round-56/59/64 `run.sh` pattern. It may not run in this round.

## Rule 12 and card scope

Oracle-input substitution cannot land. Unless the source walk names and proves
an independently computable shared statement bit-exact on NEMO inputs, GYRE
kt1--10 and days 1--30 are **UNREACHED**, not passed. LOCK_EXCHANGE and
OVERFLOW execute shared FCT2/WS code, so any eventual source edit requires
exact kt1--10 before/after artifacts. DINO uses a separate modified-leapfrog
program but shares FCT internals; any eventual shared edit requires its
execution gate and per-row accounting, with its known cancellation risk
explicit. ORCA2 remains **UNMEASURED WITH SPEC**: resolve its native compiled
card; record Kmm tracers, stage transports, post-SBC/QSR/LDF Krhs, complete FCT
internals, and pre/post-ZDF T/S for kt1--10; require exact statement replay,
every moved row registered, no AT-BAR loss, and no earlier first-over-bar row.

No NEMO source/build/run, configuration/default, carried state, stabilizer,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest is modified. Failed predictions remain in the receipt as REFUTED.

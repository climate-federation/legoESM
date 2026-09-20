# NEMO testcase L2 GYRE phase 3 — round 127 EVD trigger-state receipt

Date: 2026-09-20

Incoming tip: `48af510469da30266f0d05187350da8e6788776c`

Status: **HELD — temperature is the largest upstream state owner of the
day-180-to-240 EVD trigger disagreement: 1,097 absolute Shapley
cell-equivalents (58.66310160427808 percent), ahead of salinity's 773
(41.33689839572193 percent), while live depth receives zero Shapley trigger
cell-equivalents.  The
signed shares close exactly to all 782 daily disagreement visits, but their
1,870 absolute total exposes 2.391304347826087-fold cancellation.  The
trigger mismatch is already present at the first acquired step, so its true
birth remains before day 180.  No physics lands.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round127/`

## Outcome first

Round 126 established that the large heat-diffusivity row is a response to
upstream state rather than an intrinsic TKE/EVD closure defect.  Round 127
now closes the next magnitude question.  At 60 daily whole-step entries, the
same production-JIT closure was run on all eight subsets of NEMO and legoESM
temperature, salinity and free-surface/live-depth inputs.  All 120 endpoint
controls are exact: the empty subset reproduces NEMO's stored EVD mask and the
full subset reproduces legoESM's stored mask at every checkpoint.  Across all
480 hybrid arms, returned `rn2` and `rn2b` are bit-identical.  The exact
inverse of each recorded NEMO free-surface ratio reproduces all production
stretch bits.

The order-independent bit attribution is:

| rank | upstream input | signed trigger cell-equivalents | absolute trigger cell-equivalents | fraction of absolute attribution | continuous-margin RMS (s-2) |
|---:|---|---:|---:|---:|---:|
| 1 | temperature | `+608` | `1097` | `0.5866310160427808` | `1.5263579598234187e-6` |
| 2 | salinity | `+174` | `773` | `0.41336898395721927` | `6.877202062509306e-7` |
| 3 | live depth / free surface | `0` | `0` | `0` | `1.370696915064876e-13` |

The signed sum `608 + 174 + 0 = 782` is the measured count of daily full-input
trigger disagreements.  The absolute sum `1870` is larger because hybrid
temperature and salinity effects cancel.  These are Shapley
**cell-equivalents**, not 1,870 distinct cells.  Negative marginal effects and
three-input interactions remain in the signed allocations; none is discarded
or assigned post hoc.

Temperature also wins at both inherited checkpoints: day 180 is
`16 / 13 / 0` absolute cell-equivalents for T/S/live depth and day 210 is
`13.5 / 7.5 / 0`.  Thus the magnitude result is not an interval aggregate
hiding a reversed checkpoint.

The every-step stored-mask census starts with 15 disagreements at step 1081:
8 model-only and 7 NEMO-only.  Thirteen are in the upper 100 m, nine are south
of or at 37.2 N, but the largest longitude third is **east** with 7 cells,
versus 4 west and 4 interior.  The preregistered west-third prediction is
therefore **REFUTED** and retained.  All 360 acquired steps have at least one
mask disagreement; the true state/trigger birth is **UNMEASURED before day
180**, not relabelled as step 1081.

This result names temperature state as the next magnitude lane.  It does not
name a faulty `bn2`, TKE, or EVD arithmetic statement: the production closure
is exact on the NEMO input endpoint.  Round 128 must decompose where the
temperature-stratification state is produced in compiled process order.

No production physics, card, configuration, restart schema, carried state or
stabilizer changes.  Decision-43/45 landing gates therefore do not run and the
immutable ladder/month/year before arms do not move.

## Frozen preregistration ledger

The initial preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round127.md`, committed
before implementation and measurement as
`cdd01232ef70d01c37a6534c2e137933738b9acc`.  A pre-measurement inventory
amendment at `7c9abea5b` records that the acquired per-step frames, rather than
60 separate restarts, carry NEMO's T/S/free-surface operands.  No Round-127
scientific result existed when either commit was made.

- **P0 CONFIRMED.** The admitted headlines reproduce: 15 and 11 own-trajectory
  trigger disagreements at days 180 and 210, zero conditional endpoint
  disagreements, and bit-identical `rn2/rn2b` in all 480 production arms.
- **P1 CONFIRMED.** NEMO and legoESM endpoint masks are exact at all 60 daily
  checkpoints.  Every recorded NEMO `r3t` inverse reproduces the production
  stretch bits; no inverse required even one nextafter search step.
- **P2 CONFIRMED.** Temperature is largest and at least half of absolute
  Shapley magnitude at day 180, day 210 and over the interval.  Salinity is
  second.  Live depth contributes zero trigger cell-equivalents at both
  inherited checkpoints and over the interval.
- **P3 PARTLY CONFIRMED, PARTLY REFUTED.** The first acquired mismatch is the
  inherited step 1081 row; upper 100 m and south/equal 37.2 N are the largest
  first-step bins.  The predicted west third is refuted by east 7 versus west
  4.  The true birth remains outside the acquired interval.
- **P4 CONFIRMED.** This is diagnostic-only.  Only the private observer, owner
  gate, controls, tests, citation mappings, preregistration and receipt land.

## Compiled statements and record alignment

Every source statement below is from the compiled Round-125 branch that wrote
the admitted record.  At the whole-step entry NEMO evaluates `eos_rab` and
`bn2` from tracer slot `Nbb`, copies `rn2b` into `rn2`, and binds both
vertical-physics time-level formals to `Nbb` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90:159-168`.
Thus this GYRE card's two EVD stability arms consume the same entry state.

The resolved card selects TEOS-10 and rejects EOS-80/S-EOS at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg:124-128`.  The compiled
`rab_3d` branch makes alpha and beta functions of temperature, salinity and
the live stretched T-point depth at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1259-1308`.
The later compiled `bn2` loop interpolates alpha and beta to W points, forms
the separate temperature and salinity gradients, divides by live W thickness
and applies `wmask` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1609-1618`.
The temperature input named by this round is therefore the `pts(...jp_tem)`
gradient in that source statement, not an inferred density proxy.

NEMO creates the live T-point ratio by the literal multiply
`ssh * r1_ht_0` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/domqco.f90:208`.
The gate exactly inverts that recorded multiply and then requires the
production `nemo_reciprocal` route to recover every consumed stretch bit.

Stage 3 calls `tra_zdf` with `Kbb`, `Kmm`, `Krhs` and output slot `Kaa` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3_stg.f90:937-944`.
The additive Round-125 recorder is armed only for the original two steps or
steps 1081--1440 and fails closed on tiling, non-fp64 or a non-T/S tracer set
at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:132-140`.
It writes the exact received `Kbb` temperature and salinity at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:217-221`, and the
three live `r3t` slots at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:332-337`.
These are the oracle operands used by the hybrid cube; no N2 input is rebuilt
from an output temperature field.

The compiled driver selects the closure, copies its coefficient, and then
calls EVD at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359`.
The scored bit is the compiled threshold statement: replace `avt` by
`rn_evd * wmask` when `MIN(rn2,rn2b) <= -1e-12`, at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109`.
The record's card enables TKE and EVD and fixes `rn_evd=100` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg:205-213`.

The independent legoESM snapshots are written after every six production
steps; snapshot day `d` is therefore the model entry to step `6*d+1`, matching
the record's `Kbb` at that step.  This alignment is also mechanically closed
by the full-input endpoint: one shifted day or time level would not reproduce
all 60 independently stored legoESM trigger masks exactly.

## Production-JIT hybrid cube and controls

The authoritative report is `round127/trigger_state_budget.json`, produced by
clean instrument commit `70be2ae7cc169bb0f56c03856b7950752762df65` on CPU
with fp64/libm policy.  `diagnose_vertical_K` still runs its full production
setup and implicit-closure path.  Under the pre-existing static private
`bn2_intermediate` hook only, it additionally returns the exact
`_tke_n2_bundle.rn2/rn2b` object passed into that path.  The ordinary hook
default remains empty and keeps the original two-value return.

| fail-closed control | result |
|---|---:|
| admitted NEMO / legoESM trace | PASS / PASS |
| daily checkpoints / hybrid arms | 60 / 480 |
| NEMO-input versus NEMO stored mask unequal | 0 |
| full-input versus legoESM stored mask unequal | 0 |
| returned `rn2` versus `rn2b` unequal | 0 |
| exact recorded-`r3t` production stretch unequal | 0 |
| maximum bit-Shapley closure residual | `0` |
| maximum continuous-margin Shapley closure residual | `2.168404344971009e-19 s-2` |
| checkpoints with a full-input disagreement | 60 of 60 |

The stored-mask plant starts from an exact full-input endpoint, flips wet
interface `[1,1,0]`, is rejected as one registered unequal bit, prints
`STATUS PLANT-FIRED`, and exits 1.  Its report/log are
`round127/plant_stored_mask.json` and `.log`.

The production plant changes the real wet temperature operand `[8,4,2]` from
`24.8703777606002 C` by `9.313225746154785e-10 C` through the full jitted
closure.  It flips exactly one returned threshold bit; a repeated untouched
NEMO arm moves zero bits.  It also prints `STATUS PLANT-FIRED` and exits 1.
Its report/log are `round127/plant_temperature.json` and `.log`.  Neither
plant perturbs a zero, dry cell or post-hoc reconstruction.

## Checkpoint attribution and cancellation

| checkpoint | full disagreements | model-only / NEMO-only | T signed / absolute | S signed / absolute | live-depth signed / absolute | cancellation ratio |
|---:|---:|---:|---:|---:|---:|---:|
| day 180 / step 1081 | 15 | 8 / 7 | `+10 / 16` | `+5 / 13` | `0 / 0` | `1.9333333333333333` |
| day 210 / step 1261 | 11 | 6 / 5 | `+8.5 / 13.5` | `+2.5 / 7.5` | `0 / 0` | `1.9090909090909092` |
| day 239 / step 1435 | 18 | 9 / 9 | `+15 / 29` | `+3 / 16` | `0 / 0` | `2.5` |

The continuous-margin absolute sums over the daily mismatching cells are
`4.2348751127228244e-4`, `1.0678266264072531e-4`, and
`3.059077852366619e-11 s-2` for temperature, salinity and live depth.  Thus
live depth is not merely just below a bit threshold: its continuous effect is
also seven orders below salinity in this controlled cube.

This is an input attribution, not an operator attribution.  Temperature and
salinity jointly enter both TEOS-10 alpha/beta and the explicit gradients, so
Round 127 does not claim that only the explicit temperature-difference
subexpression owns the margin.  The all-subset production result is the
owner boundary; the next compiled-order walk must split it.

## Per-step location and the refuted spatial prediction

The direct stored-mask trace contains 4,681 disagreement visits over all 360
four-hour steps.  It is a finer cadence than the 782 daily attribution visits
and is not substituted into the Shapley denominator.

| census | total | model-only | NEMO-only | 0--100 m | 100--1000 m | below 1000 m | west / interior / east | south / north of 37.2 N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first acquired step 1081 | 15 | 8 | 7 | 13 | 2 | 0 | `4 / 4 / 7` | `9 / 6` |
| steps 1081--1440 | 4681 | 2234 | 2447 | 4493 | 188 | 0 | `1651 / 1476 / 1554` | `3573 / 1108` |

Every one of the 360 steps differs; the maximum is 27 cells at step 1223
(day 203.66666666666666).  Therefore step 1081 is only the first **observed**
row.  Calling it the physical birth would violate the record boundary.  The
first-step upper-depth and latitude predictions are confirmed, while the
first-step longitude prediction is explicitly **REFUTED** by the east-third
maximum.  The interval aggregate happens to have west as its largest third;
that does not rescue the preregistered first-step prediction.

## Landing and cross-card scope

No physical statement lands.  The admitted headlines remain those of the
incoming tip:

- kt2 T/S: `1.4210854715202004e-14 K` /
  `2.1316282072803006e-14 psu`;
- kt2 U/V: `2.7377110452773967e-12` /
  `3.2849219221489645e-12 m s-1`;
- kt3 T/S: `8.600419718618468e-7 K` /
  `6.979441735666114e-8 psu`;
- day-30 T3D RMS: `6.890484901489568e-5 K`;
- day-240 T3D RMS: `1.6446741930292448e-2 K`; and
- day-360 T3D RMS: `1.1223573910167267e-2 K`.

The only model-file change is a private diagnostic return guarded by
`_NEMOWSRK3TestHooks.bn2_intermediate`; it is statically empty under ordinary
construction and is not a card/configuration selector.  GYRE production,
generic NEMO-GYRE, DINO, LOCK_EXCHANGE and OVERFLOW execute no changed
production statement, so no certified row moves.  ORCA2 is
**UNMEASURED-WITH-SPEC**: repeat its native interval state/trigger record,
daily all-subset production cube, exact endpoint controls and fine-cadence
mask census before transferring this owner.

No configuration or carried-state question is exposed.  `DECISION_NEEDED` is
`NONE`.

## Independent adversarial review

The required command was attempted twice against the full incoming-tip diff
and evidence with `codex exec --sandbox read-only`; the second attempt also
used its ephemeral mode.  Its verbatim terminal result was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Independent review unavailable in-sandbox.  It emitted neither `SHIP` nor
`DO NOT SHIP`; no verdict is fabricated.  The logs are
`round127/codex_review.log` and `round127/codex_review_retry.log`.

## Verification

The owner self-check prints `self-check: all checks passed`; its log is
`round127/self_check.log`.  Python byte compilation and `git diff --check`
also pass.

The clean-tree focused suite covered the literal vertical solver, TEOS-10 and
`bn2`, the compiled-intermediate observer, the year-owner gate and both new
synthetic controls, the receipt-citation gate, and the worktree-stamp ratchet.
Its exact summary is:

```text
======================== 94 passed in 83.11s (0:01:23) =========================
```

The log and JUnit report are `round127/focused_tests.log` and
`round127/focused_tests.xml`.

The receipt citation gate found 12 citations, zero unmapped citations, zero
failures and zero map-audit failures in `round127/citation_gate.json`.
Shifting the compiled `bn2` assembly citation by two lines produced
`SYMBOL-NOT-AT-LINE`, printed no PASS verdict and exited 1; its report/log are
`round127/citation_gate_shift_plant.json` and `.log`.  The final clean-tip
rerun after this addendum preserves those counts.

## OPEN — round 128

Do not fix TKE, EVD or live depth: they are not the largest upstream input
owner under the exact production endpoint controls.

1. Preregister a temperature-stratification process decomposition over steps
   1081--1440.  Reuse the admitted Round-123/125 NEMO process/vertical frames
   and Round-126 legoESM production trace; do not create a second harness.
   Headline controls are the Round-127 782 daily disagreement visits,
   temperature's 1,097 absolute / +608 signed cell-equivalents, and exact
   NEMO/full-input endpoints.
2. At each EVD-sensitive interface, score the two consumed temperature levels
   and then the existing compiled-order temperature boundaries: geometry,
   advection, surface boundary, shortwave, lateral diffusion and vertical
   diffusion.  The process shares must close the recorded `Kbb` temperature
   difference and their propagated production-N2 effect; a T-field RMS alone
   is not an EVD ownership proof.
3. Report when each process first changes the temperature-gradient threshold,
   with the same depth/longitude/latitude census.  The true pre-day-180 birth
   stays `UNMEASURED`; do not claim step 1081 is the physical birth.
4. The largest temperature process row becomes the next candidate.  Any
   eventual production landing still requires the Decision-43 ladder/month
   and Decision-45 year gates against a same-tip before arm, with DINO measured
   if the statement is shared.

No NEMO acquisition is expected for Round 128: the admitted records already
carry the NEMO/legoESM per-step temperature boundaries and both consumed
temperature levels.  Request a new record only if the production-N2
propagation cannot be closed from those committed operands.

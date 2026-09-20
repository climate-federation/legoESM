# Preregistration — NEMO testcase L2 GYRE round 128

Date: 2026-09-20

Incoming lane tip: `68dd5eb367dfd351fe19cacc575677830833acdf`

This document is frozen before any Round-128 temperature-process scoring.
Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round128/`.  Round 127
assigned the largest EVD-trigger state share to temperature: `+608` signed
and `1097` absolute Shapley cell-equivalents over 60 daily entries, with a
continuous-margin signed sum of `-2.1584924289917324e-5 s-2` and absolute sum
of `4.2348751127228244e-4 s-2`.  Round 128 decomposes that temperature share
among the already recorded compiled-order temperature processes.

No production physics, card, configuration value, restart schema, carried
state, stabilizer, year harness, reconciliation gate, freshwater pair or
`#1484` guard changes in this round.

## P0 — compiled program and inherited admission controls

The compiled Round-125 step evaluates `eos_rab` and `bn2` from the whole-step
entry tracer slot `Nbb`, copies `rn2b` into `rn2`, and calls vertical physics
with both formal time levels bound to `Nbb` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90:159-168`.
The compiled buoyancy-frequency statement consumes the two temperature levels
on either side of each W interface, together with temperature-dependent
alpha, at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1609-1618`.

The compiled Round-123 stage-3 recorder follows the executing process order:
advection and surface boundary at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`, then
shortwave, lateral diffusion and implicit vertical diffusion at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-968`.
After stage 3, the completed `Naa` tracer slot becomes the next step's `Nbb`
at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90:212-226`.
Those compiled statements, not process names inferred from output patterns,
define the source order and the next-entry state scored here.

Before attribution, the existing gates must re-admit all of the following:

* the 360 Round-123 NEMO process frames for steps 1081--1440;
* the 362 Round-125 NEMO vertical frames and their exact process alignment;
* the 360-frame Round-126 independent legoESM production trace; and
* the Round-127 production-JIT endpoints: 782 full-trigger disagreement
  visits, temperature `+608` signed / `1097` absolute cell-equivalents, exact
  NEMO and full-model trigger masks, and bit-identical returned `rn2/rn2b`.

Frozen prediction: every inherited admission and endpoint control reproduces
exactly.  Any changed headline or failed admission **REFUTES** the instrument
and stops the round; a new number may not silently supersede an admitted one.

## P1 — one cumulative temperature telescope from the existing records

Round 128 extends
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_owners.py`.
It does not create another state bridge, N2 formula, process recorder or
integration harness.  At each daily entry step `6*day+1`, days 180--239, it
starts from the exact recorded NEMO `T_Kbb_in`.  The independent temperature
gap is partitioned in this fixed, preregistered order:

1. incoming temperature gap already present at step 1081;
2. free-surface content geometry;
3. advection;
4. surface boundary forcing;
5. penetrative shortwave;
6. lateral diffusion;
7. implicit vertical diffusion; and
8. explicit floating-point/temporal closure needed to reach the exact
   independently recorded legoESM `Tbb` endpoint.

For rows 2--7, the component at an entry is the sum through the preceding
step of legoESM's recorded process increment minus NEMO's recorded process
increment.  The day-180 entry therefore contains only the incoming row.  The
last row is reported rather than distributed among preferred physics.  Each
successive hybrid temperature is the preceding hybrid plus the registered
component; the final boundary is replaced by the exact recorded legoESM
temperature.  At every W interface the gate separately records the row's
change to the upper consumed T level, lower consumed T level and their
vertical difference.  A field-wide T RMS alone is not an owner metric.

The exact endpoint and the sum of all registered rows must reconstruct the
recorded `Kbb` temperature difference at every checkpoint.  Frozen numerical
control: the maximum wet-cell pre-closure reconstruction residual is at most
`1.0e-12 K`, and the exact final boundary has zero unequal wet temperature
bits against the recorded legoESM `Tbb`.  A larger residual or any endpoint
bit mismatch **REFUTES** the process telescope.

## P2 — propagate every boundary through the production N2 closure

Each temperature boundary runs through the same jitted production
`diagnose_vertical_K` closure and its returned `rn2/rn2b`; no isolated N2
calculation is scored.  To preserve Round 127's order-independent allocation,
the temperature marginal at a boundary is evaluated in all four salinity and
live-depth contexts and combined with the exact three-player Shapley weights:

* `1/3` in the NEMO-salinity/NEMO-depth context;
* `1/6` in model-salinity/NEMO-depth;
* `1/6` in NEMO-salinity/model-depth; and
* `1/3` in model-salinity/model-depth.

Subtracting consecutive temperature-boundary marginals assigns the propagated
trigger and continuous-margin change to the registered process row.  The
NEMO-temperature boundary is identically zero by construction; the exact
model-temperature boundary must reproduce Round 127's temperature Shapley
array.  This makes the process-row sum close `+608` signed / `1097` endpoint
absolute cell-equivalents and the frozen continuous-margin endpoint, while
retaining cancellation between rows.

The primary magnitude ranking is each row's sum of absolute propagated
trigger cell-equivalents over the 60 equal-cadence daily checkpoints.  The
receipt also reports signed trigger cell-equivalents, continuous-margin sums,
upper/lower-level and gradient magnitudes, the ratio of summed row magnitude
to the temperature endpoint magnitude, and every row's daily values.

Frozen magnitude prediction: implicit vertical diffusion is the largest
absolute propagated temperature-process row and lateral diffusion ranks
second.  The largest vertical-diffusion depth, longitude and latitude bins
are predicted to be 0--100 m, west third and south of or equal to 37.2 N.
Any failed rank or bin is **REFUTED**, retained in the receipt, and the
measured table wins.  No post-hoc reordering changes the registered
source-order telescope.

## P3 — first observed threshold effect and non-vacuous plants

For each process row, the gate finds the first daily checkpoint where its
propagated trigger contribution is nonzero and reports the same depth,
west/interior/east and north/south census used by Round 127.  The incoming row
is expected at day 180 because the admitted record starts with a mismatch.
Frozen prediction: at least one non-incoming process first changes a threshold
at day 181, the first scored entry after recorded process increments exist.
If not, that prediction is **REFUTED**.  The true physical birth of the
incoming row remains `UNMEASURED before day 180`; the gate may not relabel
step 1081 as its birth.

Two controls must print `STATUS PLANT-FIRED` and exit nonzero:

1. a process-registry plant removes one nonzero cumulative process row and
   must fail the recorded-temperature endpoint closure; and
2. a consumed-level plant perturbs a real wet upper or lower temperature
   operand through the jitted production closure until a returned threshold
   bit changes, while an untouched endpoint remains unchanged.

A plant that perturbs a zero, a dry cell, a post-hoc N2 reconstruction or an
unconsumed temperature value is vacuous and fails the control.

## P4 — round boundary and cross-card scope

This is a diagnostic magnitude round.  No production statement lands, so the
Decision-43 ladder/month and Decision-45 year landing gates do not run and the
immutable before arms do not move.  If a recorded process row wins, that row
becomes Round 129's single candidate.  If the incoming pre-day-180 row wins,
the receipt requests the earliest record needed to decompose it and reports
`STOPPED_FOR_RECORD`; it does not assign the inherited gap to a process.

The instrument is analysis-only.  GYRE production, generic NEMO-GYRE, DINO,
LOCK_EXCHANGE and OVERFLOW execute no changed production statement.  ORCA2 is
`UNMEASURED-WITH-SPEC`: repeat its native per-step process record, independent
production trace, four-context temperature propagation, exact endpoint
controls and threshold census before transferring the owner.

A separate read-only Codex pass must try to refute the cumulative process
alignment, two-level scoring, Shapley-context telescope, controls and owner
verdict.  Every compiled-source citation in the receipt is mapped by the
citation gate, and its shifted-citation plant must exit nonzero.

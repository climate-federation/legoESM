# Preregistration — NEMO testcase L2 GYRE round 135

Date: 2026-09-21

Incoming lane tip: `4569f934d4e08deb8415218655619814c33533ab`

This document is frozen before the Round-135 record admission, model runs, or
scoring. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round135/`.

Round 134 measured that replacing NEMO's temperature and salinity together at
each daily completed-step boundary removes `1.6394643374813826e-2 K`, or
99.683239%, of the day-240 T3D RMS gap. That is intervention leverage, not a
statement owner. Round 135 splits that intervention into one recorded tracer
at a time and measures how its leverage changes at 2-, 4-, and 8-day reset
cadences. It lands no physics or configuration.

## P0 — record and same-tip control

The only oracle input is the admitted 360-boundary record at
`phase3/round132/oracle_daily_restarts`. The existing Round-131 gate must
readmit it from the clean Round-135 measurement commit: exactly steps
`6,12,...,2160`, all 18 registered variables, and all twelve monthly overlaps
bit-identical to the certified NEMO year record. Its missing-boundary,
required-variable, and monthly-one-ULP plants must exit nonzero.

The compiled record writes temperature `tn` and salinity `sn` independently
from the accepted `Kbb` state at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:176-184`. The accepted
RK3 state is swapped into `Nbb` before diagnostics at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:220-229`, and the
restart writer is called with that `Nbb` at
`GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:249-260`.

A fresh unnudged member-0 control will run through the same production JIT for
2,160 steps with daily snapshots. Frozen prediction: all five saved fields on
all 360 days are bit-identical to the immutable free arm
`phase3/round130/arms/r109_handoff/lego_seed0_year`. Any unequal cell stops the
attribution and prevents the certified year-harness digest from moving.

## P1 — temperature-only versus salinity-only

Extend the existing Round-134 reset registry and production loop; do not add a
second stepper. The two new arms are exactly:

1. `temperature`: replace model `T` from recorded `tn`; leave `S` and every
   other state leaf free;
2. `salinity`: replace model `S` from recorded `sn`; leave `T` and every other
   state leaf free.

Each replacement occurs after the pre-reset daily score snapshot, at days
1--359, and is first consumed by the following production step. Day 1 must be
bit-identical to the control in all five saved fields. The registry, manifest,
source digest, exact 359-boundary count, and one-variable state mutation are
fail-closed and plant-controlled.

At days `30,60,90,120,180,240,300,360`, register for both arms and every one of
`T,S,u,v,ssh`: free-versus-NEMO RMS, reset-versus-NEMO RMS,
reset-versus-free RMS, and removed gap. The required table therefore has 80
rows; a 79-row plant must exit nonzero.

Frozen prediction: temperature-only is the day-240 T3D winner, removes at
least `1.55e-2 K`, and salinity-only removes less than `1.0e-3 K`. Any other
winner or either crossed bound is `REFUTED` and remains in the receipt. These
bounds do not define a landing bar.

## P2 — measured-winner cadence

Only after P1 names the winner, run that same one-variable intervention every
2, 4, and 8 days. Every replacement must use the oracle state at that exact
recorded boundary; no interpolation or reused daily value is permitted. The
expected reset counts are respectively 179, 89, and 44, alongside 359 for the
daily arm. All arms still save the pre-reset state every day.

Register the same eight days and five fields at each cadence: 160 rows for the
winner at 1, 2, 4, and 8 days. Frozen directional prediction: day-240 T3D
removal is positive at every cadence and non-increasing as the interval grows.
Any non-positive removal or reversal is `REFUTED`; it is not smoothed or
reordered post hoc.

For the daily winner, identify the first day on which the reset reduces the
free-versus-NEMO T RMS in the western-third, upper-100-m intersection. The
earliest possible and predicted day is day 2 because day 1 is scored before
the first reset. A different day, region, or depth refutes the prediction.
This is the birthplace of the reset response, not yet a source-statement
claim.

## P3 — controls, interpretation, and outcome

The instrument must prove that temperature-only changes `T` but not `S` at
the reset boundary, salinity-only changes `S` but not `T`, omitted diagnostics
leave the production step bit-identical, and a one-ULP source mutation, family
registry mutation, cadence mutation, or missing score row exits nonzero with
`STATUS PLANT-FIRED`.

This intervention cannot by itself identify the first wrong tracer-producing
statement. The receipt may name such a statement only if an existing
compiled-order record covers the measured birth interval and a one-family
substitution closes it. Otherwise the round remains `HELD`, records the exact
interval and dominant subfamily, and makes that source-family substitution the
next OPEN item. No operator is inferred from reset leverage.

A separate read-only Codex pass must try to refute record reuse, one-variable
isolation, boundary timing, cadence counts, all registered rows, the birth
localization, plants, and the ownership caveat. A `DO NOT SHIP` verdict blocks
the receipt. Every compiled-source statement must be mapped by the citation
gate, whose shifted-citation plant must exit nonzero.

No card, default, carried production state, stabilizer, NEMO source, year
acceptance bar, DINO behavior, ORCA2 selector, LOCK_EXCHANGE, or OVERFLOW path
changes. Expected status is `HELD`; `ACQUISITION_NEEDED` and
`DECISION_NEEDED` are both `NONE`.

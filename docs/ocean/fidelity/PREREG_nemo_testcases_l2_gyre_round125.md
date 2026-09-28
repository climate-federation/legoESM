# Preregistration — NEMO testcase L2 GYRE round 125 vertical-diffusion magnitude acquisition

Date: 2026-09-20

Incoming lane tip: `de1b14749a9d39eda7e8555761956ab244ec283e`

This document is frozen before any Round-125 vertical-diffusion
sub-decomposition measurement.  Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round125/`.  Round 124 closed
the independent day-180-to-day-240 process budget and ranked the stage-3
vertical-diffusion bucket first, but its admitted record contains only the
pre-solve content and the final solved tracer.  This round first proves whether
the existing records can split that bucket.  If they cannot, it completes one
passive, fail-closed acquisition unit and stops without naming a vertical
sub-owner or changing production physics.

No physics, configuration choice, restart schema, card default, carried state,
stabilizer, year harness, reconciliation gate, freshwater pair or `#1484`
guard changes in this round.

## P0 — reproduce the admitted owner before narrowing it

The unchanged Round-124 scorer will be rerun against the admitted NEMO process
record and authoritative legoESM process trace.  Frozen predictions are:

* day-240 wet-T3D RMS exactly `1.6446741930292448e-2 K`;
* vertical diffusion remains the largest absolute process projection, with
  signed carry exactly `+2.4168271578053416e-2 K`;
* its strongest ten-day birth interval remains days 190--200;
* endpoint reconstruction residual remains at most
  `3.552713678800501e-15 K`; and
* signed carries reconstruct the endpoint within one reported fp64 summation
  residual (`3.469446951953614e-18 K` in the admitted report).

Any changed headline, owner, birth interval, closure or trace-admission result
REFUTES the inherited premise and stops the round before an acquisition is
prepared.

## P1 — existing-record sufficiency test

The exact source statements show which boundaries are needed.  NEMO assembles
the temperature mixing profile from `avt` plus the isoneutral vertical
contribution at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:414-450`, builds the
tridiagonal coefficients at `:461-477`, records the completed matrix and LU
diagonal at `:492-535`, forms the content RHS and forward recurrence at
`:545-570`, and completes the back solve at `:573-578`.

The admitted Round-123 process record supplies only `Tbb`, the three
free-surface ratios, the accumulated explicit RHS boundaries and `Taa`.  The
year restarts at steps 1080 and 1440 carry endpoint `en`, `avt_k` and `avm_k`,
but not the 360 per-step temperature matrices or recurrence boundaries.  The
only existing `oracle_trazdf_matrix` records are enabled by the compiled arm
only through `kt = nit000 + 1` at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:117-147`; they cover
steps 1 and 2, not steps 1081--1440.

Frozen prediction: no admitted record contains, for every step 1081--1440,
the live entry/content, free-surface weights, vertical-diffusivity family,
matrix coefficients, LU diagonal, RHS, forward sweep and solved temperature.
Therefore a new NEMO record is required and Round 125 stops
`STOPPED_FOR_RECORD`.  This prediction is REFUTED only by an already admitted,
producer-stamped record containing every named boundary at all 360 steps on
the seed-0 trajectory.  Endpoint restart fields, an early-step record or a
legoESM-only replay does not refute it.

## P2 — one reused writer, exact target and passive admission

The acquisition reuses the existing self-describing Round-35/37 `tra_zdf`
writer and its committed reader; it does not create a second matrix format or
parser.  The source configuration is the admitted year card
`GYRE_OMIP_L2_P3_SM_YRPERT`.  The new target is
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG`, and its run root is
`phase3/round125/oracle_vertical_decomposition`.

One additive source line extends the existing `ll_l2_tra` arm to steps
1081--1440.  The original steps 1--2 remain enabled.  The target starts from
rest with the admitted seed-0 prepared inputs.  The sole namelist replacement
is `nn_itend: 2160 -> 1440`; `nn_stock=180`, `nn_write=2160`,
`nn_pert_seed=0`, `rn_Dt=14400 s` and every physical selection remain fixed.
The agent does not run `makenemo`, `mpirun` or NEMO.

The existing writer's compiled field list is fixed at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:160-335`.  It carries 11
binary64 scalars, 27 full-`jpk` arrays, three `jpkm1` arrays and three
horizontal arrays, each preceded by the existing 32-byte self-describing
header.  With the 80-byte record header and GYRE dimensions
`36 x 26 x 31`, each corrected full-domain frame is therefore exactly

```
80 + 44*32 + 11*8
   + 27*36*26*31*8
   +  3*36*26*30*8
   +  3*36*26*8 = 6,965,416 bytes.
```

The file set is exactly steps 1, 2 and 1081--1440: 362 frames and
`2,521,480,592` bytes total.  The 360 scored interval frames total
`2,507,549,760` bytes.  The existing reader must validate every header,
extent, array name/order coverage, branch flag, finiteness and EOF rather than
infer the list from file size.

Passive admission requires the target's step-1080 and step-1440 restarts to be
byte-identical to the admitted year run, with frozen SHA-256 values
`6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`
and `96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a`.
At every interval step, `T_Kbb_in` and `sol_T_post_clamp` must also be
bit-identical on every wet interior cell to the already admitted Round-123
process record's `Tbb` and `Taa`.  One differing byte or cell REFUSES the
entire record; endpoint agreement alone cannot excuse a per-step mismatch.

## P3 — coverage and non-vacuous controls

The admitted interval record must expose all five requested classes without a
silent proxy:

1. live entry/content: `T_Kbb_in` and `T_Krhs_in`;
2. free-surface weighting: `e3t_Kbb`, `e3t_Kmm`, `e3t_Kaa`, `e3w_Kmm` and the
   three `r3t` slots;
3. vertical-diffusivity family: `avt`, `ah_wslp2`, `akz` and `zwt_mix`, with
   the recorded branch flags proving which family executes;
4. matrix construction: `zwi`, `zwd`, `zws` and `zwt_lu`; and
5. solve/association: `rhs_T`, `fwd_T`, `sol_T_pre_clamp` and
   `sol_T_post_clamp`.

These names correspond to the compiled writes at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:213-333`; the arithmetic
they delimit is the compiled program cited in P1.  The later magnitude scorer
must reproduce the recorded NEMO RHS, forward sweep and solved column before
using any hybrid or projection.  It must telescope to the already admitted
vertical-diffusion component and retain any interaction/order residual as a
named row; it may not distribute a residual among preferred owners.

The acquisition gate is fail-closed on the committed producer stamp, binary
hash, exact record set, frame size, interval alignment and both restart
hashes.  At minimum, wrong-producer, truncation, consumed-matrix-ULP and
trajectory-ULP plants must each print `STATUS PLANT-FIRED` and exit nonzero.
The matrix plant must change a field the future solve consumes; the trajectory
plant must break an actual `T_Kbb_in`/process-`Tbb` comparison.  A plant that
perturbs a zero, prints PASS, or changes only an unconsumed field fails the
control.

## P4 — round boundary and future ranking

Round 125 lands only the additive acquisition card, admission mode in the
existing year-owner instrument, tests, citation mappings, preregistration and
receipt.  It does not name a day-240 vertical sub-owner from steps 1--2, an
endpoint profile or spatial resemblance.  The next round, after admission,
must add the matching write-only legoESM production-JIT trace, reproduce each
model's recorded vertical solve, and rank the five requested classes by their
signed contribution to the existing day-240 endpoint projection.  The largest
admitted sub-owner alone becomes an implementation candidate.

No production statement changes, so the Decision-43 ladder/month and
Decision-45 year gates do not run.  GYRE production, generic NEMO-GYRE, DINO,
LOCK_EXCHANGE and OVERFLOW execute no changed legoESM statement.  ORCA2 is
`UNMEASURED-WITH-SPEC`: repeat the same native-card interval record,
independent trajectory, matrix/solve closure and endpoint projection before
making an ORCA2 magnitude claim.  No configuration or carried-state choice is
made; `DECISION_NEEDED` is predicted `NONE`.

A separate read-only Codex review must try to refute the existing-record audit,
compiled field coverage, byte arithmetic, passive controls and
`STOPPED_FOR_RECORD` verdict.  The receipt citation gate must map every cited
compiled range, and its shifted-citation plant must exit nonzero.

# Round 176 receipt — developed accumulated-content process walk

**Status: HELD.**  No production physics, configuration, carried state,
restart schema, card default, or trajectory changed.  The accumulated
temperature-content residual at developed step 1081 is
`7.046353169355859e-04 K m` RMS, exactly reproducing Round 175.  Lateral
diffusion carries `0.9999899512293968` of its signed projection and
`7.046327809026458e-04 K m` isolated RMS.  Advection is the first recorded
non-bit boundary, but it carries only `1.1226766011089794e-08` of the final
projection and its existing operator walk identifies an inherited stage-3
transport.  No new internal statement is named and nothing is eligible to
land.

Preregistration: `PREREG_nemo_testcases_l2_gyre_round176.md`, commit
`d175dbdd0`.  Final instrument commit: `a8dfa84ede6c4b26d62f76a5618f7516d39f4030`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round176/`.

## Compiled order and claim boundary

The admitted Round-123 build calls advection, writes temperature `Krhs`, then
calls and records the surface boundary at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`.
At stage 3 it calls and records shortwave and lateral diffusion at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952`, then
calls the implicit vertical solve and writes the after tracer at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970`.
The Round-125 build consumes the accumulated `Krhs` in the literal content
expression at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567`.

The final Round-123 lateral-diffusion `Krhs` is bit-identical to Round 125's
recorded `T(Krhs)`, and the Round-125 literal content reconstruction remains
bit exact on all 18,000 active cells.  Thus the two independent compiled
records meet at the measured boundary without a build-to-build discrepancy.

Advection is the first non-bit cumulative write.  It is not promoted to an
internal statement: the admitted FCT/transport walks from Rounds 154--158
already place the mismatch in caller context before the FCT arithmetic.  This
round directly captures process outputs, not every advection input.  The
magnitude result instead promotes the lateral-diffusion call boundary for the
next operand walk.

## Production execution and registered rows

The existing year-owner harness drives the admitted Round-132 step-1080 state
through the complete production step.  The authoritative mode is
`LatLonCGridOceanModel.step -> self._step_jitted`; the second mode executes the
same complete step with JIT disabled.  Both use the existing
`tracer_process_trace` and `vertical_solve_trace`.  Each diagnostic result is
compared with a separately compiled ordinary result: returned-state unequal
bytes are zero in both modes.

All rows below score 18,000 active cells.  The cumulative values are direct
process boundaries expressed in content space; the isolated rows are adjacent
differences.  A separate rounding-closure component retains the cost of the
normalised trace round trip instead of assigning it to physics.

| cumulative boundary | unequal cells | max abs (K m) | RMS (K m) |
|---|---:|---:|---:|
| after advection | 18,000 | `2.5488597552025816e-09` | `4.849198709334769e-10` |
| after surface boundary | 18,000 | `2.5488597552025816e-09` | `4.849198700499986e-10` |
| after shortwave | 18,000 | `2.4687691224350994e-05` | `2.530750591030098e-06` |
| after lateral diffusion | 18,000 | `3.769802907240938e-02` | `7.046353169347874e-04` |
| complete accumulated content | 18,000 | `3.769802907240938e-02` | `7.046353169355859e-04` |

| isolated component | unequal cells | RMS (K m) | signed projection (K m) | fraction of complete projection |
|---|---:|---:|---:|---:|
| lateral diffusion | 18,000 | `7.046327809026458e-04` | `7.046282362169271e-04` | `9.999899512293968e-01` |
| shortwave | 10,200 | `2.5307568053274234e-06` | `7.072807085748905e-09` | `1.0037542705790129e-05` |
| advection | 18,000 | `4.849198709334769e-10` | `7.91077582638592e-12` | `1.1226766011089794e-08` |
| surface boundary | 598 | `3.677183875830645e-15` | `-1.4995441854826557e-18` | `-2.1281138617974443e-15` |
| rounding closure | 7,968 | `7.795704460546383e-14` | `7.986284173467043e-16` | `1.1333925481054348e-12` |

The five signed projections sum exactly to
`7.046353169355859e-04 K m`.  Reconstructing the complete error field from
the four isolated errors plus rounding closure is bit exact on all 18,000
cells.  Complete eager execution names the same first boundary and same
magnitude owner; its complete RMS is `7.046353169465208e-04 K m`.

## Frozen predictions and controls

All seven frozen predictions are confirmed: both compiled-record
calibrations are bit exact; both observers are passive; advection is the first
non-bit cumulative row with at least 1,000 cells; lateral diffusion is largest
and exceeds `1e-4 K m`; the complete JIT RMS reproduces Round 175 within
`1e-15 K m`; JIT and eager name the same first boundary; and the final plant
fires.

The first plant attempt at commit `c3a0ae310` is retained as a retracted
instrument attempt.  It called `_step_jitted` directly with the deliberate
surface-rate perturbation, so the diagnostic's planted state reached the
passivity comparison first; the CLI incorrectly printed `STATUS PLANT-FIRED`
for that premature gate error.  Commit `a8dfa84ed` routes the plant through
the public production step, which replaces diagnostic `state_after` with the
independently compiled ordinary result, and classifies any premature error as
`STATUS PLANT-BLIND` with exit 2.

The corrected plant adds `9.094947017729282e-13 K s-1` at wet cell `[1,1,0]`.
It moves zero advection cells, exactly one surface, shortwave, and lateral-
diffusion boundary cell, and 4,560 final-content cells through the implicit
solve.  It prints `STATUS PLANT-FIRED` and exits 1.  Thus it exercises the
claimed downstream path and cannot pass by perturbing an unconsumed value.

## Landing and certified trajectories

No source-exact candidate exists.  The first non-bit process output inherits
an upstream transport, while the magnitude owner is only a call boundary in
this record.  Decisions 43/45/55/59 therefore do not run.  The immutable
Round-163 values remain:

| headline | unchanged value |
|---|---:|
| first over bar | kt=3 |
| kt2 U max abs | `8.326672684688674e-17 m s-1` |
| kt2 V max abs | `9.714451465470120e-17 m s-1` |
| kt3 T max abs | `4.940071072212504e-07 K` |
| kt3 S max abs | `4.0086298724872904e-08 psu` |
| day-30 T3D RMS | `6.572574374770603e-05 K` |
| day-240 T3D RMS | `1.644836070117868e-02 K` |
| day-360 T3D RMS | `1.122566001855131e-02 K` |

No GYRE, generic, DINO, tank, ORCA2, or MPAS production statement changed, so
every card has zero registered movement.  No configuration or carried-state
decision is requested.

## Independent review, tests, and evidence

The required separate `codex exec --sandbox read-only` review could not start
in this sandbox.  Its complete terminal disposition is quoted verbatim:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
>
> codex_exit_code=1

This is **independent review unavailable in-sandbox**, not a SHIP verdict.

The final focused year-owner and citation-gate files report
**`59 passed in 33.75s`**.
The mandated single `tests/ocean/fidelity tests/ocean/unit -n 12` battery ran
once.  Six xdist workers aborted inside unrelated JAX compilations and were
replaced; the coordinator then stopped making progress at 95% and printed no
terminal pytest summary or complete failure-ID list.  After a final wait it
was terminated with exit 130.  The run is incomplete and cannot be diffed
honestly against the historical known-red list; it is not represented as a
pass and was not rerun.

| artifact | SHA-256 |
|---|---|
| `daily_record_audit_final.json` | `fcf026e3c7e79b96c662545e3ef96915697799cc267c98e80311d27c05647d9f` |
| `developed_content_process_walk_final.json` | `226556352ede2f7c82479592822348070cefae2600edc07dfd3b1436baa60b6a` |
| `content_process_effect_plant_final.log` | `eed7895e9413572e14ff7fa7c4f314169c38cce97fa499a4a7f51a6ba9664335` |
| `codex_review.log` | `626acf42958926c40c748abdf9a45bd51d2418c34d7c471ca5022ecf845f6573` |
| `focused_tests_final.log` | `9031d60330443667e8ff0dcef13be892216da335e0ed5c5ef33fbf2dcb927fc8` |
| `citation_gate.json` | `027d09a8cad40880502daea63685e8207c741dc4618b645faf8365d691c4e8c2` |
| `citation_shift_plant.json` | `735c6013cb4a1e68bf49771b3b1771afc2f7bdb6d0af25f965e6ead86ff5e19c` |
| `full_ocean_tests.log` | `0cbefb57f985364cdf75f33fa9dc4d776342c5ff7f6a6dce965f125a280f0b17` |

## OPEN — round 177

Stay at developed step 1081 and walk the lateral-diffusion call selected by
magnitude.  Under production JIT and complete eager execution, capture its
direct input tracer, live QCO thickness/metric fields, neutral-slope and
diffusivity operands, flux intermediates, and the exact `Krhs` increment in
compiled order.  Reproduce the admitted process increment before interpreting
any residual.  The existing Round-123 record stores only the call's completed
`Krhs`, so first preregister a passive NEMO operand record if no later admitted
record contains these operands.  Name an internal statement only after its
direct inputs are captured; otherwise continue upstream.  Apply the full
Decision 43/45/55/59 gate only to a one-variable source-exact candidate.

# Round 188 receipt — developed two-band shortwave walk

**Status: HELD.**  The existing passive Round-186 record is admitted: its
corrected header and payload pass, the post-call replay is BIT, and the
actual-increment ULP plant fires.  Every executed `qsr_2BD` boundary is BIT
given NEMO operands in isolated eager and isolated JIT execution.  The frozen
first-attenuation prediction is therefore **REFUTED**.  In the independent
own-chain production-JIT trace, the shortwave process boundary remains non-bit
in 9,666 of 18,000 wet cells (maximum `2.4678031493863273e-06 K`), and the
first differing input is inherited stage-3 `r3t(Kmm)`: 600 of 600 wet columns,
maximum `1.6345230724468252e-09`.  No shortwave statement is a landing
candidate; no model physics, configuration, carried state, default, or
certified trajectory changed.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round188.md`, commit
`b0aa5496c`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round188/`.

## Record admission

The compiled writer records `gdepw_1d`, no-halo `qsr`, `r3t(Kmm)`, reference
thickness and masks, the actual increment, and the post-call replay at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:925-931`.
The corrected reader consumed the source-enumerated legacy positional fields
to EOF and passed the magic, header `(1,1080,3,2,1,36,26,31,64)`, finite-value,
producer-commit, and stamp checks.  Actual versus replay is `0/29,016` cells
unequal, maximum zero.  The actual-increment one-ULP plant moved one cell,
printed `STATUS PLANT-FIRED`, and exited 1.

The Round-185 compiled process stream records the advection output, then the
surface-boundary output which is the exact `Krhs` value consumed before QSR at
`GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`.
Using that step-1080 value in `(Krhs + direct_rate) - Krhs` reproduces the
Round-186 replay bit-for-bit.  This cross-build operand is admissible because
the reconstructed replay itself is the discriminator; it is not assumed from
the byte-identical restarts.

The acquired file predates the named-array self-describing container.  It is
the single positional `WRITE` cited above, so this round did not invent group
names inside the evidence or modify it.  The reader requires every compiled
field, its source extent, exact EOF, and the producer stamp.

## Compiled-order walk

NEMO forms the two coefficients, surface attenuation, near-surface live
thickness/attenuation and associated `Krhs` update, then the visible-only deep
arm at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:616-645`.
The extinction-level function is the compiled reverse depth search at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:1021-1070`;
initialization applies it to the IR and visible bands at
`GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:1145-1213`.
Instantiating the admitted GYRE card resolves `nk0=2`, `nkV=17`; the scorer
does not score the IR exponential below `nk0`, where that branch does not run.

All rows below have maximum absolute difference zero in both isolated modes.
Counts include only cells on branches NEMO executes.

| compiled boundary | cells, eager | cells, JIT | unequal eager/JIT |
|---|---:|---:|---:|
| coefficients | 2 | 2 | 0 / 0 |
| surface depth | 600 | 600 | 0 / 0 |
| surface arguments | 1,200 | 1,200 | 0 / 0 |
| surface exponentials | 1,200 | 1,200 | 0 / 0 |
| surface attenuation | 600 | 600 | 0 / 0 |
| live depth | 10,200 | 10,200 | 0 / 0 |
| live arguments | 11,400 | 11,400 | 0 / 0 |
| live exponentials | 11,400 | 11,400 | 0 / 0 |
| live `ze3t` | 10,200 | 10,200 | 0 / 0 |
| live attenuation | 10,200 | 10,200 | 0 / 0 |
| absorbed fraction | 10,200 | 10,200 | 0 / 0 |
| absorbed numerator | 10,200 | 10,200 | 0 / 0 |
| direct rate | 10,200 | 10,200 | 0 / 0 |
| associated update, all wet levels | 18,000 | 18,000 | 0 / 0 |

The `r3t` ULP plant changes 14 direct-rate cells under JIT, prints
`STATUS PLANT-FIRED`, and exits 1.  Thus the all-BIT table is controlled by a
live input, not a comparison against itself.

## Independent trajectory versus substituted operands

| mode | entry | unequal / scored | maximum | verdict |
|---|---|---:|---:|---|
| isolated eager | NEMO operands | 0 / 18,000 | 0 | BIT |
| isolated JIT | NEMO operands | 0 / 18,000 | 0 | BIT |
| production JIT | model own chain | 9,666 / 18,000 | `2.4678031493863273e-06 K` | NON-BIT |
| production eager | model own chain | — | — | UNMEASURED |
| production JIT | NEMO stage entry | — | — | UNMEASURED |
| production eager | NEMO stage entry | — | — | UNMEASURED |

The existing production trace exposes the process boundary through the full
production step, so its own-chain JIT label is valid.  The acquired record
contains the stage-3 QSR operands and preceding tracer accumulator, not every
prognostic field needed to drive the complete production step from NEMO's
stage entry.  The isolated closure is therefore not relabelled as production;
the two NEMO-entry production rows remain explicitly unmeasured.

The source-ordered input census is:

| input | unequal / scored | maximum | disposition |
|---|---:|---:|---|
| surface `qsr` | 0 / 600 | 0 | BIT |
| `gdepw_1d` | 0 / 31 | 0 | BIT |
| reference `e3t_3d` | 0 / 18,000 | 0 | BIT |
| live `r3t(Kmm)` | 600 / 600 | `1.6345230724468252e-09` | first inherited operand |

Thus the magnitude-bearing shortwave process row does not name a shortwave
transcription defect.  It consumes a different stage free surface.  The walk
retracts the preregistered claim that vectorized attenuation is the first
non-bit statement; every measured attenuation boundary is BIT.

## Scope, certified rows, review, and tests

No executable model statement changed.  The certified values therefore remain:

- kt2 T/S/U/V: `1.4210854715202004e-14`,
  `2.1316282072803006e-14`, `8.326672684688674e-17`, and
  `9.714451465470120e-17`;
- kt3 T/S: `4.9403105251144552e-07` and `4.0085410546453204e-08`;
- day-30/day-240/day-360 T3D RMS: `2.3276772050683987e-06`,
  `6.5861718814795174e-05`, and `2.6709923853294689e-03 K`;
- first over bar: kt3.

GYRE, generic NEMO-GYRE, DINO, LOCK_EXCHANGE, OVERFLOW, and ORCA2 execute no
changed model statement.  ORCA2 transfer remains **UNMEASURED-WITH-SPEC**: it
requires native developed QSR operands, a full production trace, and an
endpoint projection before this inherited-owner result can be transferred.

The required read-only Codex review was unavailable in the sandbox.  Its
verbatim verdict text is:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only
> file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

This is **independent review unavailable in-sandbox**, not a `SHIP` verdict;
it did not issue `DO NOT SHIP`.  No second independent reviewer is callable in
this sandbox, so the diagnostic is explicitly **UNREVIEWED** under the house
dual-review rule.  Nothing in the unreviewed diff changes production physics.

## OPEN — round 189

Quantify inheritance before walking another operator: substitute the exact
NEMO step-1080 `r3t(Kmm)` into the production-JIT QSR process boundary while
holding surface flux, reference geometry, masks, preceding accumulator, and
all other state fixed.  Report how much of the `9,666`-cell / `2.467803e-06 K`
row closes and fire a production-path `r3t` plant.  If the row closes, walk the
stage-3 free-surface producer in compiled order and name its first non-bit
statement.  If it does not, rank the remaining QSR inputs/associations one at
a time.  No landing is authorized without the complete Decision-43/45/55/59
trajectory and executing-card gates.

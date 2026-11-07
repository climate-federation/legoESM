# ORCA2 round 81 — ladder reconciliation and molecular-background stability split

Base: `5c47d7340`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round81.md`.  Claim labels are kept separate:
the ten-step ladder is **given NEMO's entry**; all month arms are
**independent**.  No package file, configuration value, selector, carried
state, threshold, NEMO source/record, or sea-ice field changed.

## Verdict

**HELD.**  Round 79b's given-entry ladder reproduces exactly; round 80's stale
artifact diagnosis was an observer-path error and is loudly retracted in both
affected receipts.  The independent month discriminator is complete:
NEMO-molecular-background arms fail at the same Antarctic land-adjacent cell,
while the old-background control completes 240 steps.  Turning IWM off makes
the failure one step earlier, so IWM is not the instability owner.  The first
source-exact mismatching statement upstream of the stage sea-surface explosion
is not yet isolated, so no physics statement lands and no compensating floor is
restored.

## Given NEMO's entry — ladder reconciliation

The ordinary current-tip round-1 gate completed all 40 checkpoints.  The
round-81 reconciliation gate compares its five fields with the SHA-pinned
round-79b artifact at every checkpoint: **200 compared, 0 moved**.  The first
non-bit statement remains kt=1 stage-1 T, `UNATTRIBUTED`; the existing runoff
source exclusion remains unchanged.  The preregistered commit/decay-map owner
predictions are therefore **REFUTED**: neither review-fix commit moved the
ordinary ladder.

The reproduced kt=10 stage-3 rows are:

| field | rms | max_abs |
|---|---:|---:|
| T | 0.010214846058291116 | 1.2367457128331782 |
| S | 0.0028092655437671143 | 0.2871061346986039 |
| u | 0.00612253750027691 | 0.34018984847789213 |
| v | 0.006027472457627664 | 0.537671796435185 |
| ssh | 0.03623555945131171 | 0.5883125919751124 |

## Independent month — three arms

All arms use the card's own initial state, the same recorded surface frames,
the same NEMO 240-step restart, CPU JIT fp64, and 10,800 s steps.  Partial
terminal scores are forbidden.

| arm | IWM | `kappaM_min` / `kappaH_min` | result |
|---|---|---:|---|
| landed | on | 1.4e-6 / 1e-10 | **FAIL step 19**, 18 complete |
| IWM off | off | 1.4e-6 / 1e-10 | **FAIL step 18**, 17 complete |
| old backgrounds | on | 1.2e-4 / 1.2e-5 | **COMPLETE step 240** |

Both failures are the existing raw-mesh geometry refusal.  The exact first
invalid cell is `(j,i,k)=(1,49,0)`, latitude `-77.77420179896262`, longitude
`177.9998921261904`, bathymetry 729 m, 21 wet levels.  It is land-adjacent,
not the fold row, not a river mouth, and not a <=200 m shallow-shelf cell.  The
step-entry geometry is still finite: landed eta is `-1.4270846230081007` m and
stretch `0.998042407924543`; IWM-off eta is `-2.29921542621383` m and stretch
`0.9968460693741923`.  In the failing RK stage, 476,557 raw `e3w_int` values
are invalid and this cell's value and implied sea surface are `-inf`.
Convection classification is deliberately `UNMEASURED_INVALID_GEOMETRY`.

The one-step direction is discriminating: removing IWM does not stabilize the
model; it advances the same failure.  This agrees with the compiled statement
that wave mixing adds to `avs`, `avt`, and `avm`
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90:313-316`).  The
old floors complete, but they are the known compensating configuration and are
not a NEMO-exact fix.  Its admitted terminal comparison is:

| field | rms | max_abs |
|---|---:|---:|
| T | 0.1818297680854658 | 21.23233281600964 |
| S | 0.0425921745580533 | 4.273154389194421 |
| u | 0.020873479463173302 | 1.2432460400433065 |
| v | 0.01435952550036908 | 1.218422208027837 |
| ssh | 0.06859296146381004 | 5.5804491609268805 |

## What the split proves—and does not

The compiled ORCA2 branch deliberately resets the IWM backgrounds to molecular
values
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90:438-443`) and adds processes in closure, river-mouth,
enhanced-convection, double-diffusive, then wave order
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90:349-381`).  Decision 77 therefore remains binding.  Since NEMO
completes with IWM plus these molecular backgrounds, their legoESM failure
proves a third difference; it does not authorize a floor, stabilizer, or IWM
revert.

The existing consumer hypotheses remain ordered but unassigned.  NEMO chooses
`avt` for temperature and `avs` for salinity before constructing the implicit
tracer matrix
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90:178-215`), while its momentum matrix uses adjacent
T-point `avm`
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90:191-205`).  The captured assertion is downstream:
finite step-entry geometry becomes an infinite RK-stage sea surface before the
buoyancy-frequency consumer.  The deferred rung-5 diagnostic must
checkpoint/replay steps 17-19 and walk the stage barotropic sea-surface/
momentum boundary to the first non-finite producer before revisiting either
implicit solve.  The declared double-
diffusive split and river-mouth diffusivity remain unbuilt and unchanged.

## Prediction ledger

| preregistered prediction | result |
|---|---|
| `c83c18b25a` neutral, `e48530dc02` ladder mover | **REFUTED** — neither moves the ordinary ladder |
| decay-scale-only operand owner | **REFUTED** — there is no ordinary-path movement to own |
| current tip completes with no ladder loss | **CONFIRMED, stronger** — exact 200/200 reproduction |
| unique failure after step 2 | **PARTLY REFUTED** — unique cell, but first failure is step 19 |
| old backgrounds complete; both molecular arms fail alike | **CONFIRMED** |
| failure names a third difference, not a stabilizer | **CONFIRMED** |
| HELD absent a fully gated statement | **CONFIRMED** |

## Gates, review, and tests

The ladder reconciliation gate pins both JSONs and the round-80 refusal log,
requires 200 equal rows and an unchanged first statement, and its four plants
all fire.  The stability gate stamps the independent protocol, exact arm,
record admission, first failing step/cell, and terminal completion; all five
plants fire on both failing and completing report classes.  The failure arms'
only instrumentation is a scalar callback at the existing raw-thickness
assertion; the completing arm is uninstrumented production execution.

Validation at the committed script/citation-map tip:

* focused round-81 plus citation-gate battery: **34 passed**;
* complete ORCA2-named card battery: **436 passed**;
* round receipt citation gate: **PASS**, five citations, zero failures and
  zero unmapped; campaign-default citation gate: **PASS**, zero failures and
  zero unmapped; the shifted `zdfiwm` citation plant exits 1;
* wide `tests/ocean/fidelity -n 12`: 2,194 selected; the xdist wrapper was
  stopped after the standing 98% no-summary stall, after six failures and six
  skips.  The six failures were rerun together in isolation and are exactly
  the existing round-129 record certification, round-35 stamp scope,
  worktree-stamp emitter, missing case-board row, SI3 scalar-math provenance,
  and round-51 private-trace registry reds.  Every round-81 and every ORCA2
  test passed.

No `packages/` file changed.  GYRE is unchanged by construction from its
certified day-30 `2.3440e-06` K, day-240 `6.5826e-05` K, and day-360
`5.4085e-05` K trajectory; DINO and tank cards likewise cannot move from this
round's script, test, and documentation changes.

The required read-only Codex review failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. Decision 79 redirects the next round to build the ORCA2 hierarchy bottom-up:
   start at rung 0 (`ORCA2 geometry + GYRE physics`) with its own NEMO deck,
   card, ten-step record, and month.
2. When the hierarchy reaches rung 5, checkpoint/replay this independent arm
   at steps 17-19 and walk the RK-stage barotropic sea-surface/momentum boundary
   to the first non-finite producer at `(1,49)`; the raw-thickness assertion is
   only the first admitted detector.
3. After that producer is named, replay the compiled NEMO statement one
   variable at a time and land only under the ORCA2 and shared-card gates.
4. The distinct salt/heat double-diffusive coefficient and river-mouth
   diffusivity remain declared unbuilt; neither was approximated.
5. The ordinary kt=1 stage-1 temperature statement remains `UNATTRIBUTED`.

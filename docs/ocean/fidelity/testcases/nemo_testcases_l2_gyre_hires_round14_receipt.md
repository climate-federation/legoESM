# NEMO testcase L2 GYRE high resolution — round 14 receipt

Date: 2026-10-19

Incoming lane tip: `bf97ae51e`.  Preregistration commit:
`3293fd5f49c332038cc61735ed1dc7bbc0f7b798`.  Instrument commits:
`4ed9020a658cea8cf3273b71d1d97e23cb366903` and the review-hardening commit
`098dce2d9cf2a4693377712f2cbe4d84677e28b0`.  Final admission hardening is
`ac95210994ce73ee2554d364fd0737c977d8830e`; the line-sensitive citation map
was advanced in `bc97860aa`, and `cd284879c` repins the inherited composition
sentinel to the value independently measured at both the incoming and current
tips on this CPU/XLA environment.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/`.

## Verdict

**The paired online momentum-budget instruments are implemented, passive on
the tested legoESM trajectory, and ready for operator acquisition.  The
days-30--180 owner and cross-model budget closure remain `UNMEASURED` until
both archives are acquired.**

This round changes no production physics or card configuration.  It creates a
write-only observation path in each model and a joint admission gate.  No
shared-path fix, learned closure target, or new five-year run is warranted
before the registered attribution is measured.

## Registered measurement

The preregistration fixes native surface V, model days 30 through 180
inclusive, and 24 one-hour samples per daily arithmetic mean.  The archive
contains these eight rows in this order:

| row | exact production boundary |
|---|---|
| `advection` | stage-3 vertical-plus-horizontal advection increment |
| `pressure_kinetic_energy_gradient` | stage-3 pressure-gradient plus kinetic-energy-gradient increment |
| `total_vorticity_coriolis` | stage-3 total-vorticity/Coriolis increment |
| `smagorinsky_lateral_diffusion` | stage-3 lateral-diffusion increment |
| `vertical_mixing` | no-stress implicit-ZDF package solution minus the directly captured post-explicit/pre-ZDF velocity; on this deck the package includes NEMO's internal split-mode removal and implicit bottom-drag treatment |
| `surface_stress` | production implicit-ZDF solution minus the companion no-stress solution through the same formed matrix |
| `external_mode_correction` | final barotropic-mean replacement minus the production post-ZDF velocity |
| `total_increment` | final stage-3 velocity minus the independent whole-step entry velocity |

The first seven rows must sum to the independently formed eighth row within
`1.0e-12 m s-1` at every wet native surface-V point on every admitted day.
This is an algebraic closure bar, not a legoESM-versus-NEMO agreement bar.
The target owner remains the already registered western-boundary wet-span
bin, while both archives retain the complete native surface-V plane.

Tracer terms remain deliberately out of scope.  The requested leading field
is native surface V; an exact implicit tracer-content decomposition is not a
cheap extension of this recorder.

## legoESM instrument and controls

The GYRE WS-RK3 test seam now returns the stage-3 source boundaries.  The
surface-stress split uses a second solve
through the already formed implicit matrix with the stress RHS withheld.  The
external-mode row is observed after the final barotropic replacement.  The
ordinary returned state is produced by a separately compiled call, so no
recorded value can feed the prognostic trajectory.  Crucially, the independent
`total_increment` is recomputed from that ordinary production graph, not the
diagnostic graph; closure therefore fails if the extra-return graph changes
the measured surface-V increment.

The long-run driver accumulates each row inside the compiled daily
`lax.scan`, evaluates seasonal forcing from the step index in that scan, and
writes 151 daily budget records plus a hashed contract.  Its wrapper requires
an explicit physical GPU, the exact `r4_smag` deck, 180 days, native-fp64 GPU
policy, daily means, and budget window `30:180`; no device default exists.

The real-card CPU control passed in both forms exercised by the long run:

| control | result |
|---|---|
| direct instrumented versus ordinary step | zero moved prognostic words |
| scan-carried instrumented versus ordinary step | zero moved prognostic words |
| wet native surface-V points checked | 9,480 |
| eight-row closure | every checked word at or below `1.0e-12 m s-1` |
| planted active `1.0e-8 m s-1` term change | closure refusal fires |

The bytewise comparison includes signed zero and NaN payloads rather than
using a numerical all-close predicate.

## NEMO instrument and controls

The operator tool clones `GYRE_OMIP_L2_P3` as
`GYRE_OMIP_L2_P3_R14BUD`, supplies `-d OCE`, copies the source configuration's
prepared EXP00 inputs, applies pinned patches to private `MY_SRC` copies, and
builds with the vector-math `conda` architecture.  It refuses any rank count
other than one because the writer is `lwp`-only and not gathered.

The same executable runs twice from the same resolved `r4_smag` namelist.
`GYRE_R14_BUDGET=0` is the passive control and
`GYRE_R14_BUDGET=1` is the instrumented run.  Both run 4,320 steps; the writer
accumulates steps 697 through 4,320 and emits 151 records.  The packager
requires byte-identical final restarts with a non-empty required prognostic
registry, absence of budget records from the control, exact record
magic/layout/day sequence with filename-to-header agreement, finite active words, the
registered term order, the numeric closure bar, and complete hashes before it
writes `momentum_budget_manifest.json`.

The clean-tree source preflight applies every patch with zero fuzz, resolves
the exact deck, preprocesses and syntax-checks all four modified Fortran
units, compiles the writer module, checks all seven executing source
overrides, `namelist_ref`, the architecture, and preregistration hash and
ancestry.  It pins the NEMO engine at
`dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`, refuses changes under its
executing `src`, `ext`, `mk`, `tools`, and `makenemo` paths, and retains
NEMO's complete resolved runtime `output.namelist.dyn`.  Final admission
reopens both NetCDF restarts and independently verifies the exact numeric
registry, shapes, dtypes, every word, terminal `kt=4320`, and the declared
word count.  The clean preflight ends with:

```text
PREFLIGHT_OK
```

Unit plants independently refuse a changed active restart word (including a
signed-zero word), wrong record magic, non-finite active term, changed cadence
or registry, and a non-closing active term.  Source tests also apply all three
patches to the pinned oracle sources and exercise the packager's positive
path.

## Joint admission and exact acquisition

Run the NEMO acquisition from a clean checkout of this round's landed commit:

```bash
EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/oracle_gyre_r4_smag_budget_180d RANKS=1 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-248114866/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/run.sh --run
```

When physical GPU 0 is idle, run the legoESM acquisition at background
priority; never stop another job to claim it:

```bash
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-248114866/scripts/run/run_gyre_hires_budget_gpu.sh --physical-gpu 0 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/legoesm_gyre_r4_smag_budget_180d_round14
```

After both complete, the exact joint admission command is:

```bash
PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget_preflight.py --legoesm-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/legoesm_runs/r4_smag_gpu_budget_180d_round12 --nemo-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_hires/gyre_r4_5yr_smag --legoesm-budget-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/legoesm_gyre_r4_smag_budget_180d_round14 --nemo-budget-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/oracle_gyre_r4_smag_budget_180d/instrumented --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/momentum_budget_admission.json
```

That gate still prints owner and closure as `UNMEASURED`; it admits the two
archives for the next scoring unit.  The next measurement is the registered
days-30--180 time-mean model difference for every term over the western
boundary span, together with closure against the independently recorded
total.

## Shipped-card protection and inherited pins / OPEN

The current post-main-merge one-degree GYRE digest is
`56c5171d43504b54`, unchanged from the incoming lane.  The older
`99b2dfce324b0f77` value repeated in the round-13 receipt predates the
2026-10-16 main merge and is not the current branch's digest.  The certified
digest regression passes.  The ten-step trajectory gate retains its incoming
scientific `DEBT` verdict (first field debt at entry `kt=3`, barotropic debt at
`kt=2`); this round changes no card or production kernel and does not move
that result.

Inherited unchanged: the resolved `r4_smag` deck, 31 levels, RK3,
split-explicit free surface, TKE/EVD, Redi tracers, the literal 360-day forcing
clock, scalar-libm CPU ladder certification, thickness-weighted NEMO daily
T/S, and explicit physical-GPU selection.

OPEN, in order:

1. Run the exact paired acquisitions above at background priority.
2. Pass the joint admission gate and perform the preregistered term-by-term
   western-boundary attribution and closure measurement.
3. Change shared physics only if that one-variable attribution identifies a
   transcription defect and the usual one-degree and quarter-degree gates do
   not regress.
4. Resume the inherited scalar entry-`kt=3` SSH localization afterward.
5. Post this receipt to issue #1455 when network access is available.

## Tests and independent review

The seven fast lane groups report `246 passed, 5 deselected`: 78 card and
Smagorinsky tests, 47 phase-3/card/acquisition tests, 40 ladder tests, 25
climate/divergence/movie tests, 2 shipped-ceiling tests, 40 round-14 driver,
admission, acquisition, and citation tests, and 14 decade-climate tests.  The
real `r4_smag` recorder control passed separately in 407.96 seconds: direct
and scanned recorder-on trajectories were byte-identical to the independent
ordinary graph, all 9,480 wet surface-V words closed at the registered
`1e-12 m s-1` bar, and the `1e-8 m s-1` plant fired.  The final two-day
direct/scan composition control is reported below after comparison with the
untouched incoming tip.  Both tips first differ in SSH at model step 3, keep
the seasonal forcing clock and phase bit-equal, and produce the same maximum
surface-temperature residue, `1.652047387779021e-10 K`; the registered
`1e-12 K` control bar still fails.  The classification therefore remains
`COMPOSITION`, and round 14 did not move this inherited debt.  After the
incoming-tip control established that attribution, the repinned current-tip
test passed in 424.96 seconds.  Across the seven groups and the two explicitly
selected slow controls, the final battery is `248 passed, 3 inherited slow
tests not selected`.

Both required independent reviews finish `PASS`.  The legoESM review first
held because the recorded total came from the same diagnostic graph, source
provenance was not exact, and active NaNs could pass; `098dce2d9` moved the
total to a separately compiled ordinary graph, bound all producer blobs, and
made non-finite/JSON refusals explicit.  The NEMO review first held the
post-ZDF row, source registry, restart proof, and executable provenance;
`098dce2d9` captured the direct pre-ZDF boundary and exact override set, while
`ac9521099` independently reopens the terminal NetCDF restarts and pins the
clean NEMO engine revision/build paths.  Final focused reviewer batteries
were 23 and 11 passes respectively; neither reviewer ran NEMO.

## Compiled-source citations

The executed GYRE program updates the dynamic viscosity, performs the direct
split external solve, and then calls the three RK stages at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:183-210`.  In stage 3 the pressure
gradient initializes the momentum RHS and vorticity, kinetic-energy gradient,
and advection are accumulated at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:319-334`; lateral diffusion,
implicit vertical diffusion, and the final external-mode replacement follow
at `GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:387-418`.

The production advection wrapper calls the vertical component and then the
selected horizontal component at `src/OCE/DYN/dynadv.F90:79-82`.  The
pressure-gradient path used by this z-coordinate card assigns rather than
increments the U and V right-hand sides at
`dynhpg.F90:276-295`, which is why the first
registered source boundary is the pressure row rather than a pre-HPG
remainder.

The vertical solve is selected through the implicit path at
`dynzdf.F90:119`, with diagnostics called after the solve at
`dynzdf.F90:150-151`.  In the compiled V recurrence the wind
stress enters the surface right-hand side before the same tridiagonal solve
used for vertical mixing at
`GYRE_OMIP_L2/BLD/ppsrc/nemo/dynzdf.f90:500-528`.  This is the source boundary
that requires the independent no-stress companion rather than a post-hoc
residual labelled as stress.

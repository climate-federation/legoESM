# NEMO testcase L2 GYRE high-resolution round 1 receipt

**Date:** 2026-10-12  
**Incoming tip:** `f04f9f1d8`  
**Preregistration commit:** `489780795`  
**Card/acquisition commit:** `57242281c`  
**Driver/instrument commit:** `e3d2d4245`

## Verdict

Round 1 parameterises the certified GYRE card and supplies the acquisition,
exact-ladder, and long-run instruments.  It makes no physics change and makes
no new high-resolution fidelity claim: the quarter-degree NEMO stage-entry
records have not yet been acquired, and neither three-year integration was run
in this sandbox.

## Pre-implementation search and reuse

The search covered the existing GYRE card, phase-3 trajectory gate and forcing
adapter; the round-7 gradient runner; the committed P3 writer build; and the
local `fidelity/swg-double-gyre` branch's resolved-card, deck-selector,
compiled-day, daily-mean, and climate-instrument pattern.  The implementation
extends those paths: the phase-3 gate now calls the card's shared GYRE forcing
function, the high-resolution reader reuses its stage-entry decoder, and the
new acquisition tool reuses the already-built P3 binary.  No NEMO source,
solver, forcing law, or duplicate numerical method was added.

## Resolved card

`GyreResolvedNamelist` records the run clock, grid controls, lateral mixing,
surface-boundary selections, EOS, tracer/momentum schemes, free-surface
controls, TKE/EVD settings, and build keys.  Changes to a fixed scheme field
raise.  Runtime fields drive the card rather than serving as metadata:

- grid shape is `(20*nn_GYRE+2, 30*nn_GYRE+2)` and spacing is
  `106000/nn_GYRE` metres;
- `rn_Dt` drives the card timestep, shortwave clock, and automatic external
  substep count;
- `0.5*rn_Uv*rn_Lv` and `0.5*rn_Ud*rn_Ld` drive momentum viscosity and Redi
  diffusivity;
- `nn_itend`, `nn_write`, and `nn_stock` stay in the resolved run record.

The shipped default remains 22 x 32, 14400 s, 100 km mixing lengths and 4320
steps.  The quarter-degree card is 82 x 122, 3600 s, 25 km mixing lengths,
26280 steps, daily cadence 24 and yearly restart cadence 8760.  Both resolve 50
external substeps from their own live geometry.

The operator script attempts to replace `ln_meshmask = .false.`, but the base
deck contains no such row; its resolved evidence consequently contains no
meshmask change.  The card records the inherited false value rather than
silently converting an unmatched text substitution into a new choice.

## Shipped-deck regression

Before editing, the card digest gate passed at `99b2dfce324b0f77` and the
ten-step trajectory was saved under `gyrehr_rounds/round1/prechange_*`.  At
commit `57242281c`, the same digest passes and the oracle-relative comparison
reports `PASS`: 70 rows compared, zero worsened cells, zero legoESM field
movement, and the same first-over-bar point (step 3).  This is the required
byte regression against the pre-change legoESM control.  It does not relabel
the existing one-degree NEMO residuals as exact.

## Acquisition and exact ladder

The acquisition selector derives target and evidence names from `--deck
shipped|r4`, verifies the pinned P3-writer hash and compiled writer statement,
applies the r4 physical edits with zero patch fuzz, checks the ten-step clock,
and defaults to read-only preflight.  It never calls `makenemo`; only an
operator invocation with `--run` calls `mpirun` (eight ranks by default).

The r4 ladder gate is CPU/fp64/libm only.  Its bar is zero unequal active
float64 values for T, S, U, V and SSH at each entry from 1 through 10.  At the
first entry it moves one active value of each field by exactly one ULP; every
plant must create one unequal value.  Missing r4 records report
`ACQUISITION_NEEDED` rather than substituting the shipped oracle.

## Long-run driver

The three-year GPU wrapper is pinned to physical GPU 1, r4, 1095 days, fp64
native transcendentals, and daily means.  Its manifest labels that path
non-certified.  The CPU mode selects fp64/libm and retains a per-step loop.

The GPU program evaluates the seasonal boundary condition from `kt` inside
the scanned step.  NEMO's AB3/AM4 external mode cannot be zero-seeded without
skipping its cold-start ramp, so the one required first step uses the public
production dispatch; the rest of day 1 and every complete later day use the
compiled scan.  Each atomic daily NPZ contains end-of-day snapshots and means
of SSH, surface U/V/T, and the already-prognostic barotropic U/V, plus commit,
policy, step, cadence, and hashes in the manifest.

The reason no GPU trajectory claim is made is preregistered: the first
transcendental reached by the forcing is its seasonal cosine, CPU certification
uses libm, GPU uses native transcendental lowering, and the quarter-degree flow
is eddying.  The eventual deliverable is time-mean patterns, EKE, jet position,
and spectra over identical, separately preregistered windows.

## Tests and evidence

Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round1/`.

- Card, acquisition and original-card suites: 36 passed.
- Exact-ladder reader and one-ULP controls plus driver unit test: 3 passed.
- Shipped ten-step oracle-relative regression: PASS, 70 rows, maximum movement
  zero.
- Two-day real scan-versus-per-step CPU parity: **REFUTED at the unchanged
  1e-12 control bar**.  All six fields remained finite; maximum absolute
  differences were 5.75e-12 (SSH), 6.54e-12/1.29e-11 (surface U/V),
  1.81e-10 (surface T), and 1.10e-14/1.09e-14 (barotropic U/V).  Recompiling
  the forced step as a shared inner call produced the identical residual, so
  the result is recorded rather than hidden by relaxing the bar.
- Shipped and r4 acquisition preflights: PASS at `e3d2d4245`, including the
  pinned writer hash, derived names, exact deck rows, and clean worktree.
- Receipt-specific citation gate: PASS after mechanical re-anchoring of the
  cumulative gate necessitated by the recipe's added resolved-card lines.

Every new Python entry point has a direct test.  The one-ULP ladder plants are
non-vacuity controls.  The acquisition scripts themselves are tested for both
deck resolutions, exact physical rows, clock, derived names, and unknown-deck
refusal.

## Inherited pins / OPEN

Inherited unchanged pins include the 31-record MI96 vertical ladder (30 wet
levels plus the dummy record), closed walls and free slip, TEOS-10, FCT2,
isoneutral Laplacian tracer mixing, vector/C2/ENE momentum, SCO pressure
gradient, split-explicit `nn_bt_flt=3` with alpha 0.07 and automatic Courant
limit 0.8, nonlinear implicit drag, TKE plus EVD, two-band shortwave, seasonal
analytic forcing, qco/vco-1d/RK3, and the existing no-fixer composition.

OPEN, deliberately not inferred in round 1:

1. operator acquisition of the r4 P3 entry records and execution of the exact
   ladder gate;
2. the legoESM and NEMO three-year GPU runs;
3. the spin-up exclusion, common statistical window, and statistic-specific
   climate bars, which must be preregistered before scoring;
4. the later 1/12-degree deck and run;
5. adversarial review findings, if any, before citing new r4 numbers.
6. compiled-day versus per-step parity is DEBT at the original 1e-12 control
   bar (maximum 1.81e-10 after two days); the long GPU path remains explicitly
   non-certified and no climate statistic may treat this as a trajectory gate.

## Oracle citations

The compiled GYRE definition reads `nn_GYRE`, `ln_bench`, and `jpkglo`, then
sets both horizontal dimensions in
`cfgs/GYRE_OMIP_L2/WORK/usrdef_nam.F90:64-80`.

The compiled horizontal-grid program derives spacing and the rotated origin in
`cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90:84-94`, evaluates all four staggered
coordinate arrays and scale factors in
`cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90:119-150`, and derives beta-plane
Coriolis from those coordinates in
`cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90:161-168`.

The compiled surface routine constructs the seasonal clock and first cosines
in `cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90:84-107`, evaluates latitude-dependent
temperature, shortwave, and zero-mean freshwater fields in
`cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90:115-144`, and evaluates the seasonal
wind on the same latitude grid in
`cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90:161-176`.

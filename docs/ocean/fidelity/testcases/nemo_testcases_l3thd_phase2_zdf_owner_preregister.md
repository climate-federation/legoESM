# Lane 3b Phase-2 BL99 surface-temperature owner preregistration

Tracker: `climate-federation/legoESM#1699`

Status: **PREREGISTERED before the operand-level oracle build/run.**

## Fixed claim and bar

The measured boundary debt is `kt1.POST_ZDF.t_su`: absolute
`6.51945413210342e-08 K`, oracle scale `255.4688501439127 K`, normalized
`2.55195658039359e-10`, against the fixed `1e-15` DINO bar.  Geometry, IC, and
all ENTRY rows precede it at bar.  This continuation asks only which executing
BL99 operand first differs and whether correcting that operand inside the
already-selected SI3 identity clears the `kt=1` sub-call.  It makes no claim
about later steps until the corrected first-divergence sweep measures them.

## Rule-0 oracle reading

The executing NEMO routine is
`src/ICE/icethd_zdf_bl99.F90` in the shipped NEMO 5.0.2 tree:

- `:76,87,233-243,583-589` set at most 200 Picard iterations, measure the
  maximum absolute change over surface, snow, and ice temperatures, and accept
  only a change below `1e-4 K`.
- `:150-157,205-230` resolve `nn_qtrice=0` to constant snow extinction and
  transmit `qtr_ice_top` through snow and ice; `:404,424,448` add each layer's
  absorbed shortwave to its tridiagonal right-hand side.  Thus the top snow
  layer receives `zradab_s(:,1)`, not the full transmitted surface flux.
- `:261-275` evaluate P07 conductivity each iteration at current Picard ice
  temperatures and fixed layer salinities: top layer values at the upper
  interface, adjacent arithmetic means internally, and bottom temperature plus
  bottom-layer salinity at the base.
- `:367-379,438-448` update nonsolar flux from the current surface-temperature
  increment and impose the no-melt surface energy boundary as row 1 of the
  tridiagonal system.
- `:516-558` perform unnormalised Thomas forward elimination/back substitution
  and update surface temperature from the solved first-layer temperature.

The executing legoESM path is
`packages/ice/legoesm/ice/bitz_lipscomb.py:335-492`: P07 operands at `:377-389`,
heat capacities and flux update at `:391-398`, matrix rows at `:400-433`, the
shared Thomas solve and surface update at `:435-441`, and the same
`delta < 1e-4 K`/200-iteration freeze at `:442-476`.  Those selectors and
limits agree structurally; arithmetic identity is not assumed.

## Operand instrument and registry

The existing POST_ZDF frame cannot distinguish conductivity, matrix assembly,
elimination, surface update, and convergence.  A copy-only `MY_SRC` override of
`icethd_zdf_bl99.F90` will therefore write a separate fp64 stream for `kt=1`
only.  It will not read or modify state.  The shipped source/configuration stay
byte-untouched.

Each frame is current selected-category 1-D state in the same `ice_thd` call;
`ztiold/ztsold` are the entry-old values fixed at `:198-199`, while
`t_i/t_s/t_su` are the current Picard iterate.  The fail-closed registry order
is:

1. `INIT`: entry temperatures, salinity, layer thicknesses, initial surface
   flux/derivative, transmitted/absorbed shortwave (`:159-230`).
2. `ITER_P07_KAPPA`: current temperatures, P07 interface conductivity,
   effective snow/ice conductances (`:261-331`).
3. `ITER_CAP_FLUX`: ice/snow eta, updated nonsolar flux, net flux (`:338-379`).
4. `ITER_MATRIX`: all seven sub/diagonal/superdiagonal and RHS rows
   (`:393-448`).
5. `ITER_FORWARD`: modified diagonal and RHS after elimination (`:516-529`).
6. `ITER_SOLUTION`: unclipped solved surface/snow/ice temperatures
   (`:531-558`).
7. `ITER_CONVERGENCE`: clipped temperatures, maximum change, convergence flag,
   and iteration number (`:563-589`).

The reader rejects wrong magic/version/step/frame/iteration/count/bit width,
short payload, non-finite values, unregistered frames, and trailing bytes.

## Predeclared discriminator and private arm

For every registered scalar/array, report dimensional scale before ownership:
`max(abs(oracle))`, absolute error, normalized error using
`max(1, scale)`, and bit equality.  The primary owner candidate is the first
over-bar `(frame, iteration, field, flat-index)` in the registry order.  A
difference at INIT refutes the solver-owner claim and returns to the bridge; a
first difference after INIT localises the executing solver operation but is not
called causal until the arm fires.

The private legoESM diagnostic hook will replace exactly that first differing
scalar operand with the oracle value while holding every other input, selector,
iteration limit, tolerance, and evaluation protocol fixed.  CONFIRM owner if
the downstream POST_ZDF surface residual shrinks by at least 100x; REFUTE if it
changes by less than 2x.  Intermediate responses are INCONCLUSIVE.  The hook is
private and diagnostic only; the accepted fix must encode NEMO's existing
arithmetic inside the fixed SI3 identity, with no public selector or new model.

After a fix, rerun the `kt=1` sub-call gate and all planted controls.  Only if
`kt=1` is fully at bar will the existing registry-ordered sweep advance over
`kt=2..8760` and report the next first over-bar frame.  No trajectory claim is
predeclared.

## Choice register

- ASKED — operand-level, WRITE-only, one-step copy instrumentation.
- ASKED — scale before owner, then a one-variable private arm.
- ASKED — fix only inside the selected SI3 identity if NEMO has no switch.
- ASKED — rerun `kt=1`, controls, then sweep `kt=1..N` to the next debt.
- ASKED — CPU/fp64, no MPI launcher, no shipped-NEMO mutation, no push.
- UNASKED — none.

# SI3 lane 3 — phase-2 preregistration (rung 3.1 only)

Status: **PREREGISTERED BEFORE ANY LEGOESM/ORACLE COMPARISON.**  This phase is
limited to the shipped `ICE_ADV1D` case, the SI3 Prather advection arm, the
source-built card, and comparison with the already-certified phase-1 oracle.
It makes no claim for rungs 3.2/3.3, thermodynamics, rheology, ridging, landfast
ice, or coupled ice--ocean exchange.

## Pre-implementation search: searched and found

The search covered `packages/ice/legoesm/ice`, `packages/ocean/legoesm/ocean`,
`tests/sea_ice`, `tests/ocean/fidelity`, and
`scripts/validate/ocean_fidelity/testcases` for `transport`, `advection`,
`Prather`, `moment`, `restart`, `ICE_ADV1D`, and `beta_plane`.

* `ice/transport.py` is the canonical existing ice transport module.  Its live
  options are conservative PPM and C-grid/Voronoi donor-cell paths; it has no
  Prather moments.
* `ocean/advection_som.py` has a three-dimensional, nine-moment ocean SOM
  scheme with a different limiter and grid/volume contract.  It is not the SI3
  five-moment program and is not composed into this card.
* `ice/state.py` supplies the production dynamic-ice pytree.  The new five
  moment arrays are an opt-in prognostic carry, absent on the default path.
  The run-restart slot policy is fail-closed and therefore must classify and
  round-trip that carry.
* `create_beta_plane_cgrid_geometry` is the canonical Cartesian C-grid builder
  already used by lane 1.  The card reuses it; no second grid or ice model is
  authorized.
* The phase-1 `nemo_si3_oracle_gate.py` owns the certified fp64 binary frame
  reader, mesh/restart coverage contract, and time-level registry.  Phase 2
  imports those contracts rather than defining a second interpretation.

The implementation is consequently an opt-in `si3_prather` program in the
existing ice transport module, a pure testcase card, and a coverage-first gate.
The existing PPM/donor-cell behavior and all production defaults remain fixed.

## Exact card and source program

The card is `ICE_ADV1D_OMIP_L3`: one category (`jpl=1`), three ice and three
snow layers, `ln_icethd=F`, `ln_dynADV1D=T`, Prather on and UMx off, ponds and
landfast off.  The card sets `PrecisionPolicy.fp64()` before allocation, prints
the dtype of every scored array, and rejects a non-CPU JAX backend.

The source-built geometry is 59 x 59 T cells with `dx=dy=4 m`, zero Coriolis,
a one-cell closed rim, and the two-cell halo required by the certified frame.
The initial concentration ramp and thickness notch are transcribed from
`tests/ICE_ADV1D/EXPREF/make_initice.py:96-118`, including its float32 NetCDF
input quantization before promotion to fp64.  The prescribed U velocity is the
`icedyn.F90:144-157` convergent profile and V is exactly zero.

The transport arithmetic follows `icedyn_adv_pra.F90`: CFL subcycling
`:116-132`, extensive content construction `:218-240`, alternating split
selection `:253-353`, x-moment limiter/extraction/remnant/merge `:499-719`,
and post-transport recovery `:355-375`.  Rung 3.1's V field and all y gradients
are exactly zero, so the y sweep is an identity; the implementation rejects a
nonzero V rather than claiming the unported 2-D program.  Cold-start moments
are zero (`:1361-1370`) and all five arrays are restart state (`:1383-1497`).

Only the ORCA1-resolved single-category contract is implemented.  Optional
pond fields, layer-salinity option 4, thermodynamics, and landfast are outside
the card and remain **UNMEASURED**.

## Time levels and immutable bars

`oracle_ice_step_entry_kt00000001.bin` is `STEP_ENTRY_CURRENT`, before
`store_fields`, prescribed-velocity construction, advection, and `zapsmall`
(`icestp.F90:151-171`; `icedyn.F90:144-157`).  It is compared with the card's
cold-start entry state.  Entry frame `kt=N+1` is the observable state after
card step `N`; thus the trajectory comparison is card post-step 1 against
oracle entry `kt=2`, through card post-step 39 against oracle entry `kt=40`.
The oracle final ice restart is the post-step-40 endpoint.  The gate stops
classifying at the first outside-bar field/step and reports that divergence.

The DINO/lane-1 pointwise class is immutable: normalized maximum absolute
error `<=1e-15`, with scale `max(max(abs(oracle)), 1)`.  No scientific verdict
uses Python/NumPy exact equality.  Moment restart round-trip is bitwise because
it tests persistence, not cross-model arithmetic.

Preregistered classifications:

1. **CONFIRM geometry** iff every scored coordinate, metric and wet/face mask
   clears the pointwise/exact-mask gate and every array discovered in the
   oracle mesh retains exactly one phase-1 VERIFIED/WAIVED/UNMEASURED
   disposition.  Otherwise **REFUTE** and stop.
2. **CONFIRM kt=1 entry** iff every registered, in-scope prognostic card field
   has the documented time level, fp64 dtype, shape, finite values, and clears
   `1e-15`.  Otherwise **REFUTE**.
3. **CONFIRM trajectory** iff every in-scope prognostic clears `1e-15` at every
   available step boundary through the final restart.  Otherwise **DEBT**, with
   the first step and field printed.  Passing kt=1 alone cannot green this row.
4. **CONFIRM restart carry** iff a split legoESM integration that serializes and
   reloads all five moment arrays is bitwise identical to the uninterrupted
   integration.  Dropping or perturbing a moment must make this control red.

A planted unaccounted mesh array, a perturbed geometry field, a perturbed
kt=1 prognostic, and a dropped/perturbed moment carry must each fail with the
subject named.  Rungs 3.2/3.3, within-step x/y split states, Prather pond and
option-4 salinity moments, `zapsmall`/`zapneg` separation, thermodynamics,
rheology, ridging, landfast, and legoESM--NEMO coupled matching remain loudly
**UNMEASURED**.

## Recorded implementation amendment

The preregistration's search correctly identified the production
`DynamicSeaIceState`, but its phrase "run-restart slot policy ... must classify"
was too broad for this card-only dispatch.  Implementing that phrase would
change the default production state/restart layout outside the new card.  The
implemented boundary instead gives the card an explicit `ICEAdv1DState`
prognostic pytree: packed ordinary fields, U/V, surface temperature, and all
five moment arrays.  A card-specific restart persists every state leaf plus
the clock and selector contract; the gate reloads solely from that NPZ and
requires bitwise split-continuation.  A dropped, retyped, or perturbed moment
leaf goes red.
The production 12-field sea-ice state remains unchanged, and integration of
this opt-in card state with the general production run-restart writer is
**UNMEASURED**.  This amendment changes no registered numerical bar or
CONFIRM/REFUTE rule.

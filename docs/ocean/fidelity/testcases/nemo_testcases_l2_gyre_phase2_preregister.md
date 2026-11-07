# NEMO testcase lane 2 — GYRE phase-2 preregistration

Date: 2026-09-01

Status: **PREREGISTERED BEFORE IMPLEMENTATION OR LEGOESM/ORACLE SCORING.**
This phase is limited to the legoESM GYRE card, exhaustive geometry, the
analytic surface-forcing transcription, the initial condition, and the
`kt=1` RK3 step-entry state.  It stops before every trajectory comparison.

## Base and pre-implementation search: searched and found

Lane 2 remains based on lane 1 tip
`3d609df4c413d236325d9319f6d93b97dc16fe2a`, because lane 1 was not yet
merged into `origin/main` when this work began.  The search covered
`packages/ocean/legoesm/ocean`, `tests/ocean`,
`scripts/validate/ocean_fidelity`, the lane-1 card/gates/receipts, the DINO
recipe and gates, and NEMO's running `cfgs/GYRE_PISCES/MY_SRC`.  Search terms
included `GYRE`, `usrdef_hgr`, `usrdef_sbc`, `beta plane`, `MI96`, `RK3`,
`nemo_sco`, `TEOS-10`, `barotropic filter`, `time_level_for_dump`, and
`step-entry`.

Found and dispositioned:

* `nemo_testcase_recipe.py::_model_config` already carries the lane-1
  canonical NEMO TEOS-10, geometric EOS depth, FCT2, coupled whole-step RK3
  momentum/tracer identity, UP3, `nemo_sco` pressure gradient, explicit
  barotropic composition, and per-case filter.  GYRE extends this card; it
  does not add a solver.
* `nemo_recipe.py` already contains source-derived GYRE MI96, IC, and scalar
  seasonal-forcing formulae.  Its older `GYRE_BARE` card is deliberately not
  reused: it is unrotated and has EOS-80/adcroft/vector-ENE selectors rather
  than this campaign's collapsed OMIP-style scheme identity.
* `create_beta_plane_cgrid_geometry`, the vertical-coordinate constructors,
  `time_levels.py`, and the lane-1 phase-2 coverage gate are reusable
  machinery.  The two genuinely new card pieces are the rotated two-dimensional
  horizontal grid and the seasonal analytic surface-boundary-condition field.
* A pre-implementation arithmetic audit found that recomputing the MI96
  ladder through host NumPy `exp`/`log` differs from the certified Fortran
  mesh by up to `2.386e-15` pointwise.  The pointwise bar will not be relaxed;
  the card will pin the phase-1-certified fp64 ladder values produced by
  `usrdef_zgr.F90:93-175` and verify them against the oracle mesh.

## Exact GYRE card

`GYRE-zco` is the shipped-default closed `32 x 22` box with 30 wet MI96
levels plus NEMO's dummy `jpk=31` bottom record, `rn_Dt=14400 s`,
`nn_itend=4320`, and `nn_GYRE=1`.  It runs under `key_qco + key_RK3` and pins
the deliberate OMIP-style `ln_teos10=.true.` deviation already recorded in
the phase-1 dossier.  The card calls
`set_policy(PrecisionPolicy.fp64())` before allocation and hard-fails unless
all compared floating arrays and their geometry are `float64`.

The horizontal transcription is line-by-line source-pinned:

* `usrdef_hgr.F90:75-91` sets `zlam1=-85`, `zphi1=29`,
  `ze1=106000/nn_GYRE`, `sin(alpha)=-sqrt(2)/2`, and
  `cos(alpha)=sqrt(2)/2`, then derives the lower-left origin.
* `usrdef_hgr.F90:119-142` evaluates T/U/V/F longitude and latitude with the
  exact `i-1.5`/`i-1` and `j-1.5`/`j-1` staggering.
* `usrdef_hgr.F90:144-150` makes all eight horizontal metrics 106000 m.
* `usrdef_hgr.F90:158-168` evaluates `f=f0+beta*abs(lat-15)*rad*ra` from the
  native rotated latitude at every stagger.

The vertical ladder and closed flat mask follow
`usrdef_zgr.F90:93-175,178-207`; the rest-state profiles follow
`usrdef_istate.F90:55-77,82-101`.  The shared model selections are
`eos="nemo_teos10"`, `eos_depth="geometric"`, FCT2 tracers,
`momentum_time_integrator="rk3_ws"`,
`tracer_time_integrator="rk3_ws"`, flux-form UP3 momentum,
`vertical_momentum_scheme="nemo_up3"`, `pgf_scheme="nemo_sco"`, and the
split-explicit barotropic program.  The resolved GYRE namelist selects
`nn_bt_flt=3`, `rn_bt_alpha=.07`, and auto-resolved `nn_e=50`, hence
`barotropic_time_filter="nemo_ab3am4"` with 50 substeps.

NEMO's shipped GYRE horizontal momentum selector is vector/C2/ENE whereas the
campaign's collapsed lane-1 RK3 identity is flux-form UP3.  That difference
cannot affect an entry-state comparison; every post-entry tendency and
trajectory consequence remains explicitly **UNMEASURED** in this phase.

## Seasonal analytic forcing transcription

The gate will compare independent NumPy source transcriptions against the
card at `kt=1` entry time and at a half-year phase displacement:

* `usrdef_sbc.F90:84-107` constructs the 360-day seasonal clock and the two
  phase cosines.
* `usrdef_sbc.F90:109-120` constructs penetrative solar heat flux and the
  restoring target temperature from the rotated T-point latitude.
* `usrdef_sbc.F90:122-145` constructs the piecewise freshwater flux and
  removes its wet-area mean.
* `usrdef_sbc.F90:161-176` constructs the grid-aligned U/V double-gyre wind
  stress from the stored T-point `gphit`; `:179-184` constructs stress and
  wind magnitudes.

The field is a transcription receipt only: this phase does not advance a
forced legoESM step.  A nonzero planted perturbation to a wet solar-flux value
must fail, proving the forcing check can detect a wrong transcription.

## Coverage, bars, controls, and stopping rule

The phase-2 manifest is driven by every variable discovered in the oracle
mesh.  Each appears exactly once as `VERIFIED`, `WAIVED`, or `UNMEASURED`,
with a nonempty source/reason; missing and extra entries fail closed.  Native
coordinate aliases may be waived only when their byte-identical scientific
counterpart is verified.  The `oracle_step_entry_kt00000001.bin` label is
obtained through `time_level_for_dump`, which must return **before** per
`scripts/validate/ocean_fidelity/testcases/nemo502_MY_SRC/stprk3.F90:88-100`.

The immutable pointwise normalized bar is `1e-15`, with
`max(max(abs(oracle)), 1)` normalization.  Masks and structural zeros require
exact identity.  No accumulating-statistic bar is used.  The gate prints the
dtype of every compared array and of the live geometry.

Preregistered classifications:

1. **CONFIRM geometry** iff every registered rotated coordinate, metric,
   staggered Coriolis field, vertical ladder, closed mask, bathymetry, and
   layer thickness clears its exact/pointwise bar after explicit disposition
   of the dummy bottom record.  Otherwise **REFUTE** and do not score IC.
2. **CONFIRM forcing transcription** iff both preregistered clock samples of
   heat, freshwater, U/V wind stress, and derived magnitudes clear the
   pointwise bar on their native staggerings.  Otherwise **REFUTE**.
3. **CONFIRM IC and kt=1 entry** iff the fail-closed before-level fp64 dump,
   after halo removal and dummy-level disposition, matches wet T/S, native
   C-grid u/v, and SSH from the source-built card at the pointwise bar.
   Otherwise **REFUTE**.

Required planted controls are: an unknown mesh registry variable; a nonzero
`+1 m` rotated-grid coordinate perturbation; a nonzero wet seasonal solar-flux
perturbation; and a nonzero wet IC-temperature perturbation.  Each must turn
the gate red.

No `kt>1` state, forced step, density/rab/BN2, RK stage internal, momentum or
tracer tendency, barotropic internal, or trajectory statistic is measured.
Those claims remain loudly **UNMEASURED**, and this dispatch stops before
trajectories.

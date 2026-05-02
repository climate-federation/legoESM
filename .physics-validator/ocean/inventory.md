# Ocean physics-validator inventory

Scope: `src/legoesm/ocean/` — 108 Python files spanning EOS, dynamics
(barotropic + baroclinic on cubed-sphere, lat-lon C-grid, MPAS, spectral),
physics (bottom drag, vertical mixing inc. KPP, lateral mixing GM/Redi,
convection), surface forcing, biogeochemistry (NPZD + carbonate), freshwater,
sponge, conservation fixers, advection (DST-3, SOM, WENO, FCT), experiments
and bathymetry.

## High-risk physics subsystems
- `eos.py`: Wright (1997) nonlinear EOS, linear EOS, hydrostatic pressure
  integral, BV-frequency.
- `physics/vertical_mixing/kpp.py`: LMD94-style KPP with bulk-Ri BL
  diagnosis, non-local tracer transport (uses `B_f > 0` = unstable
  convention).
- `physics/vertical_mixing/implicit_solver.py`: backward-Euler tridiagonal
  solver via `legoesm.timestepping.tridiagonal.thomas_solve`.
- `physics/lateral_mixing/gm_redi*.py` + `_gm_redi_common.py`: DM95 slope
  taper, Visbeck (1997) adaptive GM coefficient.
- `physics/convection/{enhanced_diffusion,plume}.py`: smooth-sigmoid
  N²<0 enhancement and entraining mass-flux plume.
- `physics/bottom_drag/{linear,quadratic}.py`: bottom stress acting on
  bottom layer only.
- `physics/surface_forcing/bulk_formulas.py`: bulk lhflx/shflx via
  `saturation_mixing_ratio` from `legoesm.thermo`.
- `biogeochemistry/npzd.py`: NPZD with Redfield + CaCO3.
- `dynamics/ocean_tendency_common.py`: shared 2-pass EOS + sponge +
  freshwater virtual-salt + implicit bottom-drag-factor.
- `dynamics/barotropic_common.py`: cosine/box filter weights, BEBT blend,
  MAXVEL clip.
- `advection.py`: DST-3 Sweby/van Leer flux limiter.
- `freshwater.py`, `sponge.py`: forcing containers.

## Shared-helper status
- EOS — `compute_ocean_rho`, `compute_ocean_rho_and_pressure` exist in
  `eos.py`. Callers in `ocean_pe_*.py` use them or the equivalent helper
  in `ocean_tendency_common.iterate_eos_and_pressure_anomaly`.
- Baroclinic-tendency #214 helpers — `iterate_eos_and_pressure_anomaly`,
  `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`,
  `implicit_bottom_drag_factor` all live in
  `dynamics/ocean_tendency_common.py`.
- Barotropic helpers — `compute_filter_weights`, `bebt_blend`,
  `maxvel_clip` all live in `dynamics/barotropic_common.py`.
- Saturation thermodynamics — every ocean call site goes through
  `legoesm.thermo.saturation_mixing_ratio`.
- Constants — `eos.py` carries `rho_0=1025.0`, `c_sw=3994.0`, `scale_depth`
  for documented ocean reasons; `T_freeze_ocean` re-exports
  `constants.T_freeze_ocean`. `9.80616` literals are isolated to
  NamedTuple defaults with `# = constants.g` annotations (state.py,
  mpas_config.py, sfno_ocean.py) — audit-permitted under CLAUDE.md.
  `1004.64` and `2.501e6` show only in
  `physics/surface_forcing/config.py` defaults with the same annotation
  pattern.

## MPI / AD
- No direct `sendrecv`, `MPI.MAX`, `MPI.MIN` or `allgather` in
  `src/legoesm/ocean/`. All allreduces in conservation/eta-floor go
  through `legoesm.parallel.reductions.batch_allreduce_mpi` or the
  `global_sum_mpi` AD-safe path.

## NumPy in source
- `numpy as np` appears in 22 ocean files, all in init / experiment
  setup paths (precomputing land masks, idealised fields) and in
  `sponge.py` — these run *outside* the JAX hot loop (init time,
  not traced). Consistent with CLAUDE.md "JAX-first" rule which
  targets traced code; init-time NumPy is permitted.

## Mask discipline
- `state._replace(land_mask=...)` callsites:
  - `experiments/global_barotropic_wind.py:165` — MPAS branch only;
    MPAS state has no separate face mask, so this is correct.
  - `experiments/global_overturning.py:185` — same pattern, same
    rationale. Both correctly use `replace_land_mask` for the latlon
    branch.
  - `dynamics/ocean_model_latlon_cgrid.py:821` — appears in an error
    *message* (not a code path).
  Net: no LatLonCGridOceanState face-mask hazard.

## Static-analysis high-priority items to investigate further
1. KPP `phi_m^{-1}` formula at `kpp.py:248` — sign convention vs LMD94.
   First read suggests it is consistent with the local `B_f>0=unstable`
   choice, but worth cross-checking with codex.
2. Plume convection sigmoid sharpness `1e4` and entrainment weight
   `(1 - epsilon*dz)` becoming non-convex for thick layers
   (`plume.py:57-69`).
3. NPZD alkalinity sign convention (`-growth + remin + ...`) — sign
   depends on whether ALK includes the NO3- term.
4. Detritus sinking BC at the bottom: bottom layer always exports
   downward — physically correct for "burial flux" but worth
   double-checking against the conservation budget the global-tracer
   fixer expects.

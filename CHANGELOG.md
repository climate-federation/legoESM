# Changelog

All notable changes to legoESM. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Atmosphere

- **LES-informed compare-to-reanalysis correction**
  (`legoesm.training.{compare_reanalysis,correction_loop,...}`,
  `scripts/run/run_correction_campaign.py`,
  `docs/compare_reanalysis_runbook.md`): an offline, differentiable loop that
  runs AMIP/CMIP → time-means → compares to ERA5 → ranks the worst columns →
  spins off a plane LES (all four grids) → diagnoses a `clubb_lite` closure
  coefficient (C_K / Pr_t / C_eps, single or simultaneous) → re-runs keeping
  only bias-improving rounds (monotonic gate). The corrected per-column field
  deploys back into a fresh production run (runtime `turbulence_override`,
  grid-fingerprint-verified) or, via the environment kernel, onto any grid by
  environmental similarity. Distributed (MPAS) via a partition-invariant global
  top-k reducer; a perfect-model OSSE gives a controlled go/no-go. Turnkey
  operator path: setup/deploy preflights, a self-requeuing SLURM launcher, and
  a verified, drift-guarded runbook. See `docs/COMPARE_REANALYSIS.md`.
- **Single-column model (SCM)** (`legoesm.atmosphere.scm`,
  `scripts/run_scm_test_matrix.py`): dycore-free driver that reuses the full
  physics factory. Includes a swap-matrix sweep crossing every
  parameterization with every time integrator
  (`forward_euler`, `rk2`, `rk4`). Integrator/physics compatibility
  is validated at `SingleColumnModel.create` time; stateful
  throttled convection schemes are gated to `forward_euler`. Issue
  [#277](https://github.com/climate-federation/legoESM/issues/277).
- **RK3 tracer step** for transport convergence: 1st → 3rd order in
  time.
- **Williamson CLI** emits both PlateCarree u/v and native D-grid
  winds for plotting. Issue
  [#274](https://github.com/climate-federation/legoESM/issues/274).
- **Term-by-term analytic atmosphere SW tests + plotter**;
  CFL-aware numerical-convergence tests + plotters.
- **Williamson C2 cube-edge artifacts at C48** fixed. Issue
  [#269](https://github.com/climate-federation/legoESM/issues/269).
- **Stratosphere mass-flux gate** added to convection schemes to
  prevent TOA T spikes (>400 K) in lat-lon FV RCE.

### Ocean

- **Tripolar grid (eORCA1)** with NEMO mesh_mask loader, tensor
  pole-fold halo exchange, and full wiring through
  `scripts/run/run_omip.py`. Also a `jra55_3way` run set (MPAS ico5 /
  ico6 / tripole eORCA1).
- **Ice-shelf cavity coupling**: Holland & Jenkins (1999) three-
  equation basal melt. Lat-lon C-grid and MPAS apply paths; ambient
  read at ice-base depth.
- **Jayne & St-Laurent (2001) abyssal tidal mixing**, with tracer-
  mixing integration.
- **Dai–Trenberth global river runoff loader** + point→grid
  projection.
- **OMIP-2 SSS restoring** + WOA SSS climatology loader; wired
  through the `FreshwaterForcing` channel. Issue
  [#266](https://github.com/climate-federation/legoESM/issues/266).
- **AMOC@26.5°N diagnostic** for centennial spin-up; **multi-decade
  ocean spin-up library** (`ocean.spinup`) with RPE / volume / heat
  / salt drift tracking, declarative `ConvergenceCriteria`, Bryan–
  Lewis (1984) accelerated protocol, and auto-restart discovery.
- **Mass-conservation projection** in implicit barotropic solvers
  (lat-lon and MPAS).
- **Veros peer-comparison harness** (`ocean/fidelity/`): DINO and
  Eady adapters, regridder, and reports under `docs/ocean/fidelity/`.
- **Cross-grid metric consistency**: ocean matrix 57/57 PASS
  across lat-lon, tripolar, cubed-sphere, MPAS Voronoi.
- **Ocean dissipation suite**: biharmonic Smagorinsky (#189),
  biharmonic tracer diffusion (#206), Visbeck adaptive GM (#192),
  Leith viscosity closure (#191), energy backscatter (#193),
  Hollingsworth correction (#263).
- **Higher-order tracer advection**: PPM-FCT with true sign-split
  Zalesak limiter (#212), DST-3 vertical advection (#210), MPAS LSQ
  edge-to-cell reconstruction.
- **Bug fixes**: Mercator volume leak (#271, #176), MPAS
  conservation fixers made MPI-aware (#177), linear bottom-drag
  units (#202), wind-stress sign (`negate bulk-flux tau for ocean
  convention`), stale face masks on `land_mask` replace (#184),
  longitude convention standardised (#181).

### Sea ice

- **mEVP rheology** (Bouillon 2013 / Kimmritz 2015) alongside EVP
  (Hunke & Dukowicz 1997) with lat-lon dispatch and tensor pole-
  fold halo documentation.
- **MPAS Voronoi sea-ice support**: transport + rheology + dynamics
  + no-flux boundary on icosahedral mesh.
- **Tier 1+2 extensions**: Lipscomb 2001 linear ITD remap, snow on
  ice, brine pockets, ridging, delta-Eddington shortwave, melt
  ponds.

### Land

- **CLM-ML-JAX multilayer canopy scheme** (`legoesm.land.canopy.CLMMLCanopyConfig`,
  `legoesm.land.canopy.clm_ml_interface`): port of the NCAR Community Land Model
  with Multi-Layer Canopy (CLM-ML v2; Bonan et al. 2021) as a pluggable legoESM
  surface scheme. Activated via
  `MultiLayerLandConfig(surface_scheme=CLMMLCanopyConfig())`;
  requires `pip install legoesm[canopy]`. Key capabilities:
  - Multi-layer within-canopy radiative transfer, turbulence, leaf energy balance,
    stomatal conductance (Ball-Berry / Medlyn), and plant hydraulics.
  - 87-variable CHATS7 site validation (May 2007 walnut orchard) against Fortran
    CLM-ML v2 reference outputs; see `scripts/validate/validate_clm_ml_canopy.py`.
  - `CanopyState` carries `mlcanopy_type` (JAX NamedTuple) and 10-day running-mean
    air temperature (`t_a10_arr`) between timesteps; cold-start allocated on first
    call.
  - **Architectural note**: CLM-ML uses Python/NumPy control flow; the scheme is
    not `jax.jit`-compatible and gradients do not flow through it. Forward
    simulation only.
- **Offline single-point multilayer land driver** (`scripts/run/run_lmip.py`) for
  10-year soil spin-up before ERA5 coupling.

### Coupler

- **Ice-shelf cavity** wired through tile coupler with
  lat-lon and MPAS apply helpers.
- **Tidal mixing diagnostic** + Dai–Trenberth runoff + ice-shelf
  basal-melt driver-side glue.

### CMIP6 / Diagnostics

- **CMOR/CMIP6**: Amon, Lmon, Omon, **Oyr, Ofx, SImon, SIyr**
  tables now wired; overturning streamfunction, OSNAP transports,
  dianeutral mixing, variance, and global ocean / sea-ice scalars.

### Infrastructure / performance

- **Persistent JAX JIT cache** enabled by default. Issue
  [#273](https://github.com/climate-federation/legoESM/issues/273). Override
  with `LEGOESM_JAX_CACHE_DIR=/path/to/cache`, disable with
  `LEGOESM_JAX_CACHE_DISABLE=1`.
- **SPMD halo backend** activated in the AMIP production profile
  for multi-device runs. Issue
  [#275](https://github.com/climate-federation/legoESM/issues/275).
- **MPI compatibility guardrails** in `legoesm.parallel.reductions`:
  hard-error for `mpi4jax<0.8`; warn for JAX / mpi4jax outside the
  tested range. Promote to hard error via
  `LEGOESM_MPI_STRICT_COMPAT=1`.
- **Background-thread snapshot plotting** so matplotlib does not
  block the GPU; matplotlib serialisation lock added.
- **JRA55 RYF preload** as `float32` to fit on a 32 GB V100S.

### Bug fixes (other)

- `clip(b, tiny) + divide` patterns in `atmosphere/physics/` no
  longer produce NaN/inf reverse-mode gradients (#249).
- ERA5 initial-conditions for AMIP wired into `ModelDriver` (#238).
- `legoesm test williamson --case 2` (#235).
- AMIP external-forcing ingestion: unit parsing, CMIP6 volcanic,
  case-insensitive TSI, interannual ozone, descending-lat (#207).
- `external.py _interp_zonal_to_grid` 3-D aerosol crash (#178).
- `solar spectral_file` mode with RRTMG (#180).
- `ModuleNotFoundError: No module named 'tests'` after
  `f6f1d65` (#188).
- `ensure_geometry` dropping lat/lon bounds for regional lat-lon
  grids.

### Documentation

- New: [docs/user-guide/getting_started.md](docs/user-guide/getting_started.md) —
  beginner onboarding.
- New: [docs/user-guide/scm.md](docs/user-guide/scm.md) — single-column model.
- New: this CHANGELOG.
- Updated: [README.md](README.md),
  [docs/validation/cmip_readiness.md](docs/validation/cmip_readiness.md),
  [docs/dev-notes/ocean_experiments_reference.md](docs/dev-notes/ocean_experiments_reference.md),
  [docs/performance/REAL_HARDWARE_SCALING.md](docs/performance/REAL_HARDWARE_SCALING.md).

---

## Earlier history

For prior milestones (FV3 core port, MPI scaling activation,
spectral PE tracer pipeline, etc.) see the project memory under
`memory/MEMORY.md` and the commit history.

# Changelog

All notable changes to legoESM. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- **License: MIT.** legoESM is now released under the MIT License, replacing
  PolyForm Noncommercial 1.0.0; the separate commercial license is withdrawn.
  The root package metadata now matches the component packages, which already
  declared MIT. Bundled third-party code keeps its own license.

### Fixed

- **`import legoesm` no longer crashes on platforms without a glibc
  `libm.so.6`.** `legoesm.core.transcendentals` dlopened `libm.so.6` at
  module import, so every import of `legoesm.ocean` (and everything
  downstream) failed on macOS, where no `libm.so.6` exists. The load is
  now deferred to the first `transcendentals='libm'` evaluation, with
  `libm.dylib` as a fallback; the certification host still resolves the
  same `libm.so.6` soname, and `native`-policy users load no C library
  at all. The same import-time dlopen pattern remains in
  `legoesm.ocean.forcing.nemo_fld_read` (lazily imported, certification
  cards only) and is unchanged.

- **Hines gravity-wave drag no longer launches at the surface.** The launch
  level now defaults to 700 hPa and must lie in 300–900 hPa; the surface-launch
  path is removed (a surface-launched wave broke in the boundary layer and
  deposited 55% of its momentum below 1 km). Affects every configuration that
  selects Hines without setting `hines_launch_p`; production (McFarlane) is
  unchanged. A launch level that would fall in a column's lowest model layer
  gives that column no source. Configs carrying the old `hines_launch_p: 0.0`
  (or a Hines override with `launch_p=None`) are now rejected by validation.

## [1.0.0] - 2026-10-01

First stable release, **legoESM 1.0**. The whole federation (root + eight
members) moves lockstep to `1.0.0`. Entries below cover changes merged since
2026-07-18; PR numbers refer to
[climate-federation/legoESM](https://github.com/climate-federation/legoESM/pulls).
The earlier, previously-unreleased work follows, grouped by component.

### Highlights since 2026-07-18

#### Atmosphere

- **Production AMIP is the CAM6 physics suite on MPAS** (#1787):
  `config/amip/amip_production.yaml` selects MPAS level 6 (~1.1°), the CAM6
  32-level hybrid table, Zhang–McFarlane convection, prognostic CLUBB with
  CAM6 CLUBB cloud fraction, MG2-style Morrison microphysics, RRTMGP (selected as `rrtmg`) with McICA,
  McFarlane orographic drag and CESM Large–Yeager fluxes. The previous deck is
  kept runnable as `config/amip/amip_sundqvist_l36.yaml`. Production AMIP moved
  to MPAS level 6 (#1658); no in-cloud inhomogeneity thinning (#1795); in-cloud
  KK2000 warm rain (#1843, #1846; CAM6 MG2's own coefficients are the
  `kk2000_cam6` option, not selected by the deck); ZM and CLUBB tunables are run-config
  fields (#1840).
- **Cloud radiation**: maximum-random-overlap subcolumns and McICA
  (`--cloud-vertical-overlap-optics {none,max_random,mcica}`, #1411, #1803);
  partial-cloud plane-parallel albedo bias corrected (#1398); emissivity-space
  LW inversion for partial-coverage optics (#1555); radiative ice paired with
  the configured effective radius (#1524); per-band aerosol SSA/asymmetry in
  RRTMGP (#1276); RRTMGP top-layer flux overwrite removed (#1758).
- **Microphysics**: `--morrison-flavor {mg,sam}` (#1363); homogeneous cirrus
  ice nucleation (#1336); ice-number melt/sublimation sinks (#1374); N_r
  consistency ceiling (#1476); uniform supersaturation guard across schemes
  (#1506).
- **Turbulence**: prognostic CLUBB surface second-moment BC ported (#1601);
  LES-tuned turbulence coefficients — default in the SCM, opt-in for AMIP via
  `--params` (#1448, #1670, #1694); trainable Businger–Dyer MOST coefficients
  (#1282, #1285).
- **Gravity-wave drag**: optional Hines launch level (`hines_launch_p`, #1394;
  the default still launches at the surface, and production uses McFarlane);
  E3SM frontogenesis source with the E3SM namelist threshold (#1229, #1234);
  real subgrid-orography file wired with a scale check (#1550, #1717, #1741).
- **Dycores**: energy-conserving Simmons–Burridge vertical transport and
  frictional heating in the spectral PE core, removing a 0.48 K/day leak
  (#1569); opt-in monotone van Leer vertical advection on MPAS (#1505);
  column-conserving positivity borrow for MPAS tracers and number
  concentrations (#1349, #1371, #1373); conservative tracer vertical transport
  on the hybrid lane (#1801); cube winds can be carried in D staggering
  (#1739, #1462); hybrid-coordinate negative-layer-mass detection and terrain
  inversion guard (#1402, #1784); opt-in complex64 (fp32-runtime) spectral SW and PE modes (#1664,
  #1667); spectral SW hyperdiffusion scales with truncation (#1813); real
  orography and land mask on the MPAS mesh (#1393).
- **FV3 duo-grid JAX lane** scored against the pinned Fortran: hydrostatic,
  non-hydrostatic, moist and Kessler arms (#1563, #1610, #1630, #1781, #1788).
- **One temperature-dependent latent-heat family** in core, used by the
  surface interface, sea ice and the canopy (#1829, #1836, #1849, #1850).

#### Land

- **Interactive land on the MPAS AMIP lane**: the lane previously ran without a
  land model; multilayer Richards land is now restored and coupled (#1618),
  runs under MPI (#1546), and uses the calibrated LMIP land and spun-up soil
  (#1651). Packed land columns plus a land cadence give 1.87× whole-model
  speed (#1673).
- **CLM-ML multilayer canopy**: differentiable and MPI-distributable (#1265),
  backend vendored in-repo (#1269), O(1)-compile scan over columns with
  per-column PFT (#1281, #1299, #1303, #1304).
- **Canopy physics**: savanna two-source canopy (#1305), unified interception
  and wet/dry-leaf energy balance (#1268, #1272), separate soil/plant wilting
  points (#1274), transpiration routed to the root zone (#1235), canopy solver
  no longer reports failed columns as converged (#1632), one flux law at the
  land–air interface (#1475).
- **LMIP biophysics driver** with calibrated 2° global setup (#1474, #1624);
  opt-in prognostic carbon (#1703); soil-water reporting fix (#1841).

#### Ocean

- **NEMO-faithful TKE vertical mixing (zdftke)** on tripole and MPAS, with
  prognostic TKE, mixing-length options and NEMO surface/floor conventions
  (#1225, #1315, #1326, #1365, #1367, #1404, #1611, #1689, #1691, #1704, #1738).
- **Eddy parameterizations**: Treguier GM coefficient (#1347, #1375), Ferrari
  (2010) GM streamfunction (#1487), GM bolus strength fix (#1343).
- **Freshwater and ice**: `real_freshwater` closure without virtual salt flux
  (#1484), joint volume+salt normalization (#1401), frazil-ice closure and
  in-situ temperature EOS inverse (#1747).
- **Solver and IC fixes**: periodic-seam barotropic mass leak (#1382),
  barotropic averaging window centred on t+dt (#1609), default free-surface
  solver step (#1639), implicit momentum solve at closed faces (#1642),
  no-data IC columns no longer enter as 0 °C / 0 PSU (#1494, #1732),
  biharmonic mixing capped at the run's timestep (#1823).
- **Fidelity programme**: DINO twin on NEMO's own grid (#1638, #1728,
  #1842), GYRE bit-for-bit with NEMO 5.0.2 through kt2 (#1802), FESOM2
  forcing parity and a FESOM2-JAX arm (#1486, #1549, #1725).
- **Scale-out**: multi-device SPMD lane for the MPAS ocean
  (`--enable-mpas-spmd`, #1726); opt-in global polynomial preconditioner for
  the barotropic PCG (#1847); resumable restart for `run_omip_core2` (#1444).

#### Sea ice / coupler

- Multi-category ITD thermodynamics in the coupled tile (#1273); prognostic
  sea-ice skin temperature on MPAS (#1325).
- Conservative cube↔lat-lon remap: cube and spectral atmospheres drive a 3-D
  ocean (#1271, #1275); coupled surface-flux export on the SPMD atmosphere lane
  (#1485); coupled-CMIP precipitation units and water-budget fixes (#1250,
  #1287).

#### CMIP6 output, forcing and diagnostics

- CMIP6 metadata driven from the official tables (#1501); interval means with
  truthful `cell_methods` (#1357, #1713); multi-rank MPAS CMOR feed (#1572)
  and refusal to write empty files (#1545); cloud, clear-sky, `clt` and
  `evspsbl` on the MPAS lane (#1437, #1525, #1851); lat-lon output sampled at
  labelled cell centres (#1838).
- CMIP6 forcing adapters: input4MIPs AMIP downloader, GHG, solar (14 RRTMG-SW
  bands) and MACv2-SP aerosol (#1277, #1292, #1293, #1301); regrid polar-gap
  fix (#1338, #1341).
- Energy and moisture budgets use the coordinate's real layer mass (#1424,
  #1731); per-column budget ledger on MPAS (#1439).

#### Parallel and performance

- MPAS: ragged all-to-all halo fill (#1534, #1536), Hilbert-ordered edges and
  vertices (−11% step, #1791), interior/rim halo overlap (default off, #1659),
  split PV-flux fusion on multi-GPU (#1831), rank-local diffusion coefficient
  fix restoring multi-GPU parity (#1794), per-rank GPU binding before MPI init
  (#1529, #1541, #1551).
- AMIP GPU day: q_v smoother compiled once (2.9×, #1824), RRTMGP minor-absorber
  loop (−19%, #1826), plev19 weights once per feed (#1828), land-forcing
  marshal compiled once (#1832); command buffers no longer capture collectives
  (#1575); CPU rank layout and solver work (#1804).
- Precision modes `fp32` / `fp64` / `mixed` with a dtype gate (#1665, #1774).

#### ML, training and DA

- Unified WeatherBench + AIMIP training driver with curriculum, EMA and loss
  presets (#1244, #1262, #1339); ERA5 surface fluxes as a lower boundary
  condition (#1753); WB host-RAM and OOM fixes (#1306, #1392).
- 4D-Var single-observation experiments (#1348) and minimizer fixes (#1809).

#### Infrastructure

- **License: PolyForm Noncommercial 1.0.0**, with commercial use under a
  separate license (`COMMERCIAL-LICENSE.md`), effective
  2026-07-18 (#1195, #1196).
- Run manifest records applied `--params` values and the imported package's
  repository (#1510, #1594); strict launch-config loading (#1452); tier-0
  parameters refused on the calibration route (#1523); CI gate apparatus
  restored (#1410).
- RCEMIP-1 plane-CRM case, tracked and reproducible (#1500, #1507).

### Atmosphere (earlier, previously unreleased)

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
  `scripts/matrix/run_scm_test_matrix.py`): dycore-free driver that reuses the full
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
  - Superseded in 1.0: the scheme is now differentiable and MPI-distributable,
    with the backend vendored in-repo (see Highlights → Land).
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
  with `LEGOESM_JIT_CACHE_DIR=/path/to/cache`, disable with
  `LEGOESM_JIT_CACHE_DIR=""`.
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
- 1.0: README, `docs/index.md` and the user guides refreshed for the CAM6
  production AMIP configuration, current CLI defaults and install route.
- Updated: [README.md](README.md),
  [docs/validation/cmip_readiness.md](docs/validation/cmip_readiness.md),
  [docs/dev-notes/ocean_experiments_reference.md](docs/dev-notes/ocean_experiments_reference.md),
  [docs/performance/REAL_HARDWARE_SCALING.md](docs/performance/REAL_HARDWARE_SCALING.md).

---

## Earlier history

For prior milestones (FV3 core port, MPI scaling activation,
spectral PE tracer pipeline, etc.) see the project memory under
`memory/MEMORY.md` and the commit history.

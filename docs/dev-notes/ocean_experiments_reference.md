# Ocean Experiment Reference

This document describes the ocean validation experiments in legoESM, their scientific
motivation, setup, expected behavior, and known issues. It is intended as a reference
for developers modifying the experiments or interpreting test matrix results.

Source modules live in `src/legoesm/ocean/experiments/`. The test matrix runner is
`scripts/matrix/run_ocean_test_matrix.py`.

**Status (2026-05):** 57/57 PASS across lat-lon, tripolar (eORCA1), cubed-sphere,
and MPAS Voronoi grids — see commit `55ccddc3` "Cross-grid metric consistency".

**Recent additions** (Phases A–F + 6a closeout):
- **Phase A**: tier 5–8 runners — `eady_uniform`, `eady_*`, `acc_channel`, `dino`, `global_overturning`
- **Phase D**: Munk, Held–Larichev, NeverWorld2-lite, ISOMIP+
- **Centennial spin-up**: AMOC@26.5°N, RPE drift, Bryan–Lewis accelerated protocol (`ocean.spinup`)
- **Realistic forcing**: JRA55-do RYF preload, Dai–Trenberth river runoff, OMIP-2 SSS restoring + WOA climatology, ice-shelf basal melt
- **Tidal mixing**: Jayne & St-Laurent (2001) abyssal K (`ocean/physics/vertical_mixing/tidal.py`)
- **Peer fidelity**: Veros DINO / Eady adapters under `ocean/fidelity/` and reports in `docs/ocean/fidelity/`

---

## Shared Infrastructure

### Vertical Coordinate

All ocean experiments use the **z-star** coordinate (`OceanZStarCoordinate`), the only
vertical coordinate in the ocean model. There is no pure sigma or fixed-z option.

Construction (`create_ocean_z_star`):
- Layer thicknesses grow linearly from `dz_surface` (default 10 m) at the top to
  `dz_deep` (default 200 m) at the bottom, then rescaled so the total equals `H_max`.
- For the default 10-level, 5500 m case, the rescale factor is ~5.24, giving surface
  layers ~52 m and bottom layers ~1048 m.
- `z_half_ref[0] = 0` (sea surface), `z_half_ref[-1] = -H_max` (bottom).
- `z_full_ref[k]` = midpoint of interfaces k and k+1.
- At runtime, actual layer thickness = `dz_ref[k] * (eta + H_bathy) / H_max` (z-star
  stretching).

Level convention: k=0 is surface, k=nlev-1 is deepest.

### Shared Defaults

| Parameter | Value | Notes |
|---|---|---|
| Vertical levels (nlev) | 10 | Overridden by lock_exchange, overflow (20), phillips/IGW (2) |
| Ocean depth (H_max) | 5500 m | Overridden by several experiments |
| Surface layer target (dz_surface) | 10 m | Before rescaling |
| Deep layer target (dz_deep) | 200 m | Before rescaling |
| Time step (dt) | 300 s | Sized for ~2.5-degree resolution CFL |
| Land threshold | \|lat\| > 80 deg | Some experiments use no land |
| Vertical mixing (K_v) | 1e-4 m^2/s | Always active; causes small T drift |

### Grid Resolutions (~2.5-degree equivalent)

| Grid | Resolution | Horizontal shape |
|---|---|---|
| cubed_sphere | C24 | 6 faces x 24 x 24 |
| latlon | 36x72 | 36 lat x 72 lon |
| mpas | ico3 | ~642 cells, ~1920 edges |
| spectral | T21 | 32 lat x 64 lon (Gaussian), 253 SH coefficients |

### State Representations

The four grids use different state types with different variable representations:

| Grid | State type | Velocity | Scalars (T, S) | SSH (eta) | Bathymetry | Land mask |
|---|---|---|---|---|---|---|
| cubed_sphere | `OceanState` | u, v at cell centers `(6,n,n,nlev)` | `(6,n,n,nlev)` | `(6,n,n)` | `(6,n,n)` | `(6,n,n)` |
| latlon | `LatLonOceanState` | u, v at cell centers `(nlat,nlon,nlev)` | `(nlat,nlon,nlev)` | `(nlat,nlon)` | `(nlat,nlon)` | `(nlat,nlon)` |
| mpas | `MPASOceanState` | edge-normal u on edges `(nEdges,nlev)`, no v | `(nCells,nlev)` | `(nCells,)` | `(nCells,)` | `(nCells,)` |
| spectral | `SpectralOceanState` | vor_hat, div_hat `(n_sh,nlev)` complex128 | T_hat, S_hat `(n_sh,nlev)` complex128 | eta_hat `(n_sh,)` complex128 | H_bathy_hat `(n_sh,)` complex128 | grid-space `(nlat,nlon)` float |

---

## 1. Rest State No Land (`rest_state_no_land`)

**Purpose.** The cleanest possible ocean validation: start from a motionless, stably
stratified, pure global ocean (no land boundaries) and verify that the model stays at
rest. Any drift is purely from numerics (truncation error, vertical mixing). This
provides the baseline against which all other experiments and the with-land variant are
compared.

### Vertical Grid

- z-star, 10 levels, H_max = 5500 m.
- Layers: ~52 m (surface) to ~1048 m (bottom) after linear stretch + rescaling.

### Initialization

| Variable | Value | Formula / Notes |
|---|---|---|
| eta (SSH) | 0 | Zero everywhere |
| u, v (velocity) | 0 | Zero everywhere. MPAS: edge-normal u = 0. Spectral: vor_hat = div_hat = 0. |
| T (temperature) | Exponential profile | `T(z) = 2 + 18 * exp(z / 1000)` where z is negative (depth). ~20 C at surface, ~2 C at depth. Horizontally uniform. |
| S (salinity) | 35.0 PSU | Uniform everywhere |
| H_bathy (bathymetry) | 5500 m | Uniform everywhere. Spectral: stored as SH coefficients via `sh_analysis`. |
| land_mask | 1.0 (all ocean) | `land_lat_threshold = 90 deg` means no land on any grid. |

### Forcing

None. The ocean evolves under its own dynamics only.

### Expected Behavior

- SSH drift at or near machine precision for latlon and mpas (can be exactly zero),
  O(1e-9) for cubed_sphere (from metric term numerics), O(1e-11) for spectral.
- Temperature drift O(1e-6) from vertical mixing (K_v = 1e-4 m^2/s acting on the
  exponential stratification).
- Salinity drift at machine precision (no gradients to mix).
- No spurious circulation or wave generation.

### Validation Thresholds

- eta_drift < 1e-10 (normalized by H_max)
- T_drift < 1e-4 (relative)
- S_drift < 1e-6 (relative)

### Recent Results (quick mode, 0.1 day)

| Grid | eta drift | T drift | Status |
|---|---|---|---|
| cubed_sphere C24 | 1.62e-09 | 7.10e-06 | PASS |
| latlon 36x72 | 0.00e+00 | 2.49e-06 | PASS |
| mpas ico3 | 0.00e+00 | 2.49e-06 | PASS |
| spectral T21 | 5.73e-11 | 2.49e-06 | PASS |

### What This Tests

- Pressure gradient consistency with EOS (no spurious pressure gradient from a
  horizontally uniform state).
- Time stepping stability.
- Conservation of mass, heat, salt without boundary complications.
- Baseline numerical noise floor for each grid type.

### Duration

1.0 day (quick: 0.1 day).

---

## 2. Rest State (`rest_state`)

**Purpose.** Same stratified rest state as above, but with land at high latitudes. This
tests everything rest_state_no_land tests, plus land-ocean boundary treatment.
Comparing the two quantifies the impact of land masking on each grid.

### Vertical Grid

Identical to rest_state_no_land: z-star, 10 levels, H_max = 5500 m.

### Initialization

All variables initialized identically to rest_state_no_land except for land_mask and
(for spectral) bathymetry:

| Variable | Value | Differences from no-land |
|---|---|---|
| eta (SSH) | 0 | Same |
| u, v (velocity) | 0 | Same |
| T (temperature) | `2 + 18 * exp(z/1000)` | Same profile, including on land cells (intentional, for smooth gradients at coastlines) |
| S (salinity) | 35.0 PSU | Same, including on land |
| H_bathy | 5500 m (FV grids) | FV grids: uniform 5500 m everywhere including land (land handled by masking, not bathymetry). Spectral: smooth transition `1 + 5499 * mask`, from 5500 m (ocean) to 1 m (land), stored as SH coefficients. |
| land_mask | See below | Grid-dependent |

### Land Mask Details

| Grid | Formula | Type |
|---|---|---|
| cubed_sphere | `1.0` where `\|lat\| < 80 deg`, else `0.0` | Sharp step function |
| latlon | Same sharp step | Sharp step function |
| mpas | Same sharp step | Sharp step function |
| spectral | `0.5 * (1 - tanh((\|lat\| - 80) / 5.0))` | Smooth tanh taper, 5-degree e-folding width. Kept in grid space (not spectral). |

The spectral grid uses a smooth taper to avoid Gibbs ringing from a discontinuous mask.
The FV grids can handle sharp boundaries directly.

### Forcing

None.

### Expected Behavior

- FV grids: slightly more eta drift than no-land (O(1e-9) vs O(0) or O(1e-9)) due to
  land boundary numerics, but still negligible.
- Spectral: **known issue** -- large eta drift (~0.3 m) and land leakage (~0.6 m).
  The tanh taper + hyperdiffusion are insufficient to suppress spectral ringing at the
  land-ocean boundary during time stepping.
- Temperature and salinity drift similar to no-land case.

### Validation Thresholds

Same as rest_state_no_land.

### Recent Results (quick mode, 0.1 day)

| Grid | eta drift | T drift | land_leakage | Status |
|---|---|---|---|---|
| cubed_sphere C24 | 5.79e-09 | 7.21e-06 | -- | PASS |
| latlon 36x72 | 6.56e-09 | 2.49e-06 | -- | PASS |
| mpas ico3 | 1.60e-07 | 2.62e-06 | -- | PASS |
| spectral T21 | 3.44e-01 | 5.22e-04 | 6.27e-01 | PASS (but drift is large) |

### What This Additionally Tests (vs no-land)

- Land-ocean boundary treatment and tendency masking near coastlines.
- Conservation fixers with non-uniform ocean area (weighted by mask).
- For spectral: Gibbs suppression via tanh taper + hyperdiffusion (currently failing).

### Known Issues

Spectral land masking is broken: eta drift is 8 orders of magnitude worse than without
land. This was exposed on 2026-04-04 by changing `spectral_land_lat_threshold` from 90
to 80 degrees. The previous setting of 90 (no land) was a workaround that hid the bug.

### Duration

1.0 day (quick: 0.1 day).

---

## 3. Barotropic Wave (`barotropic_wave`)

**Purpose.** Validate barotropic gravity wave propagation, numerical dispersion, and
phase speed accuracy. A Gaussian SSH perturbation is placed at the equator and allowed
to radiate as barotropic gravity waves.

### Vertical Grid

z-star, 10 levels, H_max = 5500 m. Same as rest_state.

### Initialization

Starts from rest_state initial conditions, then adds a Gaussian SSH perturbation:

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | `1.0 * exp(-d^2 / (2 * sigma^2))` added to rest state eta=0 | d = great-circle distance from (180 E, 0 N), sigma = 10 degrees. Spectral: perturbation applied in grid space then transformed via `sh_analysis`. |
| u, v (velocity) | 0 | Inherited from rest state (geostrophic adjustment will generate velocity) |
| T (temperature) | `2 + 18 * exp(z/1000)` | Inherited from rest state |
| S (salinity) | 35.0 PSU | Inherited from rest state |
| H_bathy | 5500 m | Inherited from rest state |
| land_mask | From `_create_rest_state` | FV grids: land at \|lat\| > 80 deg. Spectral: land at 80 deg (after recent change), though experiment config has `spectral_land_lat_threshold = 90`. |

### Forcing

None. Free wave propagation.

### Expected Behavior

- Circular wavefronts propagate outward from the perturbation center.
- Theoretical phase speed: c = sqrt(gH) ~ 232 m/s.
- max|eta| should be O(0.1-0.2) m for FV grids after the initial pulse disperses.
- Wave amplitude conserved to within 20%.
- Mean SSH should not drift (mass conserved).

### Validation Thresholds

- eta_conservation: 0.8 -- 1.2
- mean_eta_drift < 1e-4 m
- min_final_amplitude > 0.1 m

### Recent Results (quick mode, 0.2 day)

| Grid | max\|eta\| | Status |
|---|---|---|
| cubed_sphere C24 | 0.168 m | PASS |
| latlon 36x72 | 0.124 m | PASS |
| mpas ico3 | 0.232 m | PASS |
| spectral T21 | 7.321 m | PASS (but anomalous) |

### Grid Support

- cubed_sphere (C24), latlon (48x72), mpas (ico4): full support with land at
  |lat| > 80 deg. Resolutions are matched by effective dx (~384-446 km) rather than
  using the global defaults, to ensure comparable numerical dispersion across grids.
- **spectral: skipped** — land boundaries cause Gibbs ringing issues that produce
  anomalous amplitudes (~7 m vs ~0.2 m on FV grids). Disabled until spectral land
  masking is fixed.

### Duration

2.0 days (quick: 0.2 day).

---

## 4. Barotropic Gyre (`barotropic_gyre`)

**Purpose.** Validate wind stress momentum transfer, Sverdrup balance, and western
boundary current formation in a single-gyre configuration. Based on Stommel (1948) and
Munk (1950).

### Vertical Grid

z-star, 10 levels, H_max = 5500 m.

### Domain

Rectangular ocean basin: 0-120 deg E, 15-75 deg N. Land on all four sides (meridional
+ zonal boundaries). This provides the western boundary needed for western boundary
current formation and closes the basin for proper Sverdrup balance.

### Initialization

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | 0 | Zero everywhere |
| u, v (velocity) | 0 | Zero; wind drives circulation |
| T (temperature) | 10 C | Uniform everywhere (purely barotropic) |
| S (salinity) | 35.0 PSU | Uniform everywhere |
| H_bathy | 5500 m | Uniform (including land, for smooth Jacobian) |
| land_mask | 1 inside basin, 0 outside | Rectangular basin mask |

### Forcing

Prescribed single-gyre wind stress via `OceanPhysicsConfig`:
- `tau_x = -tau_max * cos(pi * (lat - lat_s) / (lat_n - lat_s))`, tau_max = 0.1 Pa.
- Easterlies at southern boundary (15 N), westerlies at northern boundary (75 N).
- One sign of wind stress curl → one anticyclonic (subtropical) gyre.
- Applied to surface layer as `du/dt = tau_x / (rho_0 * dz_0)`.

### Expected Behavior

- Single anticyclonic gyre develops with western boundary intensification.
- Maximum surface speed 0.05-0.5 m/s after spin-up.
- SSH depression in gyre center (Ekman pumping).

### Validation Thresholds

- max_speed_final: 0.05 -- 0.5 m/s
- eta_drift small (< 1e-3 m absolute)

### Grid Support

- cubed_sphere, latlon: full support with basin land mask.
- **mpas: not supported** — MPAS lacks surface forcing infrastructure (see issue #55).
- **spectral: not supported**.

### Known Issues

- Goes unstable at 30 days without dissipation (all mixing/drag disabled). Needs
  lateral viscosity or bottom drag for long integrations.

### Duration

30.0 days (quick: 2.0 days).

---

## 4b. Barotropic Double Gyre (`barotropic_double_gyre`)

**Purpose.** Validate wind-driven double-gyre circulation with two counter-rotating
gyres. Based on Holland & Lin (1975).

### Domain

Same rectangular basin as barotropic_gyre: 0-120 deg E, 15-75 deg N.

### Initialization

Identical to barotropic_gyre: zero velocity, uniform T = 10 C, uniform S = 35 PSU.

### Forcing

Prescribed double-gyre wind stress:
- `tau_x = -tau_max * cos(2*pi * (lat - lat_s) / (lat_n - lat_s))`, tau_max = 0.1 Pa.
- Easterlies at both boundaries (15 N and 75 N), westerly jet at mid-basin (45 N).
- Curl changes sign at mid-basin → subtropical gyre (south) + subpolar gyre (north).

### Expected Behavior

- Two counter-rotating gyres: anticyclonic subtropical, cyclonic subpolar.
- Westerly jet separation at mid-basin boundary.
- SSH dipole: positive (subtropical) and negative (subpolar).

### Validation Thresholds

Same as barotropic_gyre.

### Grid Support

Same as barotropic_gyre (cubed_sphere, latlon only).

### Known Issues

Same stability issue as barotropic_gyre at 30 days without dissipation.

### Duration

30.0 days (quick: 2.0 days).

---

## 5. Baroclinic Adjustment (`baroclinic`)

**Purpose.** Validate geostrophic adjustment from a meridional temperature front,
thermal wind balance, and baroclinic instability onset. Based on Gill (1982), Pedlosky
(1987), Vallis (2017).

### Vertical Grid

z-star, 10 levels, H_max = 5500 m.

### Initialization

Starts from rest state, then adds a temperature perturbation:

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | 0 | From rest state |
| u, v (velocity) | 0 | From rest state; geostrophic adjustment generates currents |
| T (temperature) | Rest state profile + perturbation | Perturbation: `+/-5 C * cos(lat)`, decaying exponentially with depth (e-folding scale: 3 levels). Spectral: applied in grid space, transformed back via `sh_analysis_3d`. |
| S (salinity) | 35.0 PSU | Uniform |
| H_bathy | 5500 m | Uniform |
| land_mask | FV: \|lat\| < 80 deg; spectral: no land (90 deg) | Spectral avoids land |

### Forcing

None. Free adjustment.

### Expected Behavior

- Geostrophic currents develop in thermal wind balance with the temperature front.
- Inertia-gravity waves radiate from the initial adjustment.
- Baroclinic instability may develop at longer integration times.
- Temperature drift < 1e-3 relative; max speed < 1.0 m/s.

### Validation Thresholds

- T_drift_relative < 1e-3 (loose contract)
- **T_drift_relative < 1e-8 (test-matrix runner gate)** — geostrophic_adjustment has no T forcing or T diffusion, so the
  test-matrix runner gate is tightened to 1e-8 in both
  `scripts/matrix/run_ocean_test_matrix.py:run_geostrophic_adjustment` and
  `scripts/matrix/ocean_test_matrix/experiments.py:run_geostrophic_adjustment`.  Empirically observed drift is roundoff-level
  (~1e-16 to 1e-12) on all supported grids; 1e-8 leaves a wide safety margin and catches numerical bugs that the loose
  1e-3 contract would miss.
- T_drift_absolute < 0.1 C (fallback)
- max_speed_final < 1.0 m/s
- min_adjustment_speed > 0.001 m/s

### Duration

10.0 days (quick: 1.0 day).

---

## 6. Phillips Two-Layer (`phillips_two_layer`)

**Purpose.** Validate baroclinic instability in a classic two-layer configuration
following Phillips (1954). An unstable baroclinic zonal jet is initialized with
opposing flows in upper and lower layers, and eddy development is expected.

### Vertical Grid

- z-star, **2 levels only**, H_max = **3500 m** (shallower than standard).
- Two thick layers representing the upper and lower ocean.

### Initialization

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | Small perturbation | `0.05 m * sin(3 * lon) * cos(2 * lat)` to seed instability (iter-135 self-review fix: prior doc said `sin(2 * lat)` but both spectral and FV code paths in `scripts/matrix/run_ocean_test_matrix.py:_add_phillips_perturbation` and `src/legoesm/ocean/experiments/phillips_two_layer.py:_add_phillips_perturbation_*` use `cos(2*lat)` — the doc was stale, code is canonical). Spectral: area-weighted mean removed before SH transform. |
| u, v (velocity) | Opposing zonal jets | Upper: 0.30 m/s eastward. Lower: -0.06 m/s westward. Gaussian jet centered at 45 deg lat, 14 deg width. Spectral: converted to vor_hat/div_hat via spectral transforms. |
| T (temperature) | Meridional gradient | Upper: 16 C (equator) to 6 C (pole), gradient 10 C. Lower: 8 C (equator) to 4 C (pole), gradient 4 C. |
| S (salinity) | 35.0 PSU | Uniform |
| H_bathy | 3500 m | Uniform |
| land_mask | FV: \|lat\| < 80 deg; spectral: no land (90 deg) | |

### Forcing

This is the **only experiment with active forcing**:
- **Temperature relaxation**: Restores T toward initial profile with timescale tau = 15
  days. Maintains the meridional gradient that drives baroclinic instability.
- **Bottom drag**: Linear momentum damping with timescale tau = 25 days. Prevents
  unbounded kinetic energy growth.

### Expected Behavior

- Eddies develop from the unstable jet via baroclinic instability.
- SSH amplitude grows as instability develops (growth ratio 0.8-10.0).
- Temperature drifts are larger than unforced experiments (threshold 5 C absolute).
- System should not blow up: max|eta| < 5 m.

### Validation Thresholds

- T_drift_absolute < 5.0 C
- eta_growth: 0.8 -- 10.0
- max_eta_amplitude < 5.0 m

### Duration

10.0 days (quick: 1.0 day).

---

## 7. Inertia-Gravity Wave (`inertia_gravity_wave`)

**Purpose.** Validate Poincare wave dynamics and provide the only quantitative accuracy
assessment in the test suite via comparison against an analytical solution. Based on
Bishnu et al. (2024).

### Vertical Grid

- z-star, **2 levels**, H_max = **1000 m** (barotropic-equivalent shallow basin).

### Initialization

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | Sinusoidal wave | `1.0 * sin(2 * lon) * sin(2 * lat)` (wavenumber-2 in both directions). Spectral: via `sh_analysis`. |
| u, v (velocity) | 0 | Zero initial velocity; wave dynamics generate flow |
| T (temperature) | `2 + 18 * exp(z/1000)` | Standard profile (not dynamically important for barotropic mode) |
| S (salinity) | 35.0 PSU | Uniform |
| H_bathy | 1000 m | Uniform |
| land_mask | 1.0 (all ocean) | **No land on any grid** (land_lat_threshold = 90 deg) |

### Forcing

None. Free wave oscillation.

### Analytical Solution

f-plane with f0 = 1e-4 s^-1. Dispersion relation:
`omega^2 = f0^2 + g * H * (k^2 + l^2)`, giving omega ~ 1.4e-4 s^-1.

The experiment provides `compute_analytical_solution(t)` for exact comparison at any
time. This is the strongest validation available in the test suite.

### Expected Behavior

- Inertia-gravity wave oscillates with the analytical frequency.
- Purely oscillatory, no secular growth.
- L2 error vs analytical < 0.1 (relative).
- Amplitude conservation within 20%.

### Validation Thresholds

- l2_error < 0.1 (vs analytical)
- amplitude_conservation: 0.8 -- 1.2

### Duration

2.0 days (quick: 0.2 day).

---

## 8. Lock Exchange (`lock_exchange`)

**Purpose.** Validate density-driven gravity current formation and quantify spurious
numerical mixing using Reference Potential Energy (RPE) diagnostics. Based on Petersen
et al. (2015) and Ilicak et al. (2012). A classic test for ocean model mixing
properties.

### Vertical Grid

- z-star, **20 levels**, H_max = **500 m** (shallow, high vertical resolution for
  resolving gravity currents).

### Initialization

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | 0 | Zero everywhere |
| u, v (velocity) | 0 | Zero; density contrast drives flow |
| T (temperature) | Step function at prime meridian | Western hemisphere (lon < 0): 5 C (cold, dense). Eastern hemisphere (lon >= 0): 30 C (warm, light). Uniform vertically. Spectral: via `sh_analysis_3d`. |
| S (salinity) | 35.0 PSU | Uniform |
| H_bathy | 500 m | Uniform |
| land_mask | \|lat\| < 80 deg = ocean | Standard land mask. Spectral: tanh taper. |

### Forcing

None. Density contrast drives the dynamics.

### RPE Diagnostics

Uses linearized EOS: `rho = rho_ref * (1 - alpha_T * (T - T_ref))` with rho_ref =
1025 kg/m^3, alpha_T = 2e-4 K^-1, T_ref = 15 C. Reference Potential Energy is the
minimum PE achievable by adiabatic rearrangement.

### Expected Behavior

- Dense water slides under light water along the bottom; light water rides over dense
  at the surface.
- RPE should decrease (available PE converts to kinetic energy).
- Low spurious mixing: RPE drift < 1% relative.
- Temperature stays bounded.

### Validation Thresholds

- pe_drift_relative < 1e-2
- Temperature within [-200, 200] C (blowup check)
- pe_rel_final < 0

### Grid Support

- cubed_sphere, latlon: full support.
- **mpas: not supported** (idealized channel geometry).
- **spectral: not supported** (RPE calculation complex in spectral space).

### Duration

1.0 day (quick: 0.1 day).

---

## 9. Overflow (`overflow`)

**Purpose.** Validate dense water overflow dynamics over a bathymetric slope. Tests
bottom boundary layer processes and gravity current propagation with variable
topography. Based on Petersen et al. (2015).

### Vertical Grid

- z-star, **20 levels**, H_max = **2000 m** (deep basin maximum; shelf is 500 m).
  Variable bathymetry means the z-star Jacobian varies spatially.

### Initialization

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | 0 | Zero everywhere |
| u, v (velocity) | 0 | Zero; density + topography drive flow |
| T (temperature) | Meridional front with depth decay | Poleward of 50 deg: 5 C (cold, dense). Equatorward: 20 C (warm, light). tanh transition, 5 deg width. Decays toward T_deep=2 C with depth (factor 0.5). |
| S (salinity) | 35.0 PSU | Uniform |
| H_bathy | Variable | 500 m shelf (poleward of 40 deg) to 2000 m deep basin. tanh transition, 7 deg width. Minimum 50 m enforced. |
| land_mask | \|lat\| < 80 deg = ocean | Standard land mask |

### Forcing

None. Density contrast + topographic slope drive the dynamics.

### Expected Behavior

- Dense cold water flows down the bathymetric slope as a bottom-attached gravity
  current.
- RPE should decrease as potential energy converts to kinetic.
- Temperature drift < 1% relative.

### Validation Thresholds

- pe_drift_relative < 1e-2
- T_drift_relative < 1e-2
- pe_rel_final < 0
- Temperature within [-200, 200] C

### Grid Support

- cubed_sphere, latlon: full support including variable bathymetry.
- **mpas: not supported** (idealized bathymetric slope geometry).
- **spectral: not supported** (limited bathymetry support).

### Duration

0.5 day (quick: 0.1 day).

---

## 10. Stommel Gyre Tracer (`stommel_gyre_tracer`)

**Purpose.** Validate passive tracer transport, integral conservation, and monotonicity
preservation in a wind-driven Stommel gyre. Based on Hecht et al. (2000). Tests
advection scheme quality under realistic flow conditions.

### Domain

Same rectangular basin as barotropic_gyre: 0-120 deg E, 15-75 deg N.

### Vertical Grid

z-star, 10 levels, H_max = 5500 m.

### Initialization

| Variable | Value | Notes |
|---|---|---|
| eta (SSH) | 0 | Zero everywhere |
| u, v (velocity) | 0 | Zero; wind drives circulation |
| T (temperature) | 10 C | Uniform (barotropic setup, same as gyre experiments) |
| S (salinity) | Background + Gaussian blob | 35 PSU background + 2 PSU Gaussian blob centered at (35 N, 60 E), width 10 deg. Surface level only. |
| H_bathy | 5500 m | Uniform |
| land_mask | 1 inside basin, 0 outside | Rectangular basin mask |

### Forcing

- Single-gyre wind stress (Stommel pattern): `tau_x = -tau_max * cos(pi * (lat - lat_s) / (lat_n - lat_s))`, tau_max = 0.1 Pa.
- Easterlies at south, westerlies at north → single anticyclonic gyre that advects the tracer.

### Expected Behavior

- Tracer blob advected by the gyre, stretching and filamenting.
- Total tracer integral conserved to < 0.1% relative.
- No spurious extrema: overshoot/undershoot < 0.1 PSU beyond initial range.
- Monotonicity preservation depends on the advection scheme's limiter.

### Validation Thresholds

- integral_drift < 1e-3 (relative)
- overshoot < 0.1 PSU
- undershoot < 0.1 PSU

### Grid Support

- cubed_sphere, latlon: full support with basin land mask.
- **mpas: not supported** — MPAS lacks surface forcing infrastructure (see issue #55).
- **spectral: not supported**.

### Notes

- Longest-running experiment (60 days default, 5 days quick).
- Flagged as `long_integration` and `conservation_critical`.

### Duration

60.0 days (quick: 5.0 days).

---

## 10b. BENCH Performance Benchmark (`bench`)

**Status:** implemented (branch `bench/ocean-bench`, 2026-10-10); see
`docs/ocean/experiments/bench_plan.md` for the full provenance
(Irrmann et al. 2022, GMD 15, 1567-1582, Sect. 2.2 + NEMO `tests/BENCH`).

### Purpose

Performance benchmarking ONLY — results are physically meaningless by
design. Zero input files (grid, bathymetry, ICs, forcing all analytic);
every grid point carries a unique value (per-point `z2d` ramp, NEMO
`usrdef_istate.F90:44-54` ported verbatim) so halo/sharding bugs are
detectable in the fields. Runs the production-like
`legoesm_nemo_like_v1` recipe + TKE vertical mixing, so the measured
step cost is the production cost.

### Vertical Grid

Uniform z* levels, `dz = H_max / nlev`, flat bottom H_max = 5000 m,
75 levels in the presets (NEMO BENCH `usrdef_zgr.F90:139-165`).

### Initialization

| Field | Formula (NEMO port) | Range |
|---|---|---|
| T | `10 + 20*z2d - 4*f` (f = k/(nlev-1)) | ~6-10 ± 1 °C |
| S | `34 + f + z2d` | ~34-35 ± 0.1 psu |
| u | `0.1 * z2d` | ± 0.01 m/s |
| v | `0.01 * z2d` | ± 0.001 m/s |
| eta | `0.1 * (0.5 - p)` (p = point index) | ± 0.05 m |

Light stable stratification (T down, S up with depth); per-point ramp
amplitudes verbatim from NEMO; backgrounds temperate (divergence: no
SI3, so NEMO's near-freezing targets are inapplicable).

### Forcing

None (NEMO `usrdef_sbc.F90`: all surface fluxes zero).

### Expected Behavior

Stability only, gated by pre-registered gates: finite fields,
max speed < 1 m/s, max |eta| < 1 m (planted-violation controls in
`test_bench.py` prove the gates fire). Preset dt values are measured
stability boundaries (36×72×20 unstable at dt=3600 s, stable at
dt<=1800 s; probes 2026-10-10).

### What This Tests

The production step cost on global-size grids without any data
dependency: JIT-compile time, steady per-step wall time, SYPD,
Mcells/s; and the north-fold communication pattern via the synthetic
tripole lane (`run_bench.py --grid tripole`).

### Duration

Arbitrary (benchmarks count steps; NEMO default `nn_itend = 1000`).
Matrix lane: 0.5 days.

---

## Cross-Experiment Summary

### Grid Coverage Matrix

| Experiment | cubed_sphere | latlon | mpas | spectral |
|---|---|---|---|---|
| rest_state_no_land | yes | yes | yes | yes |
| rest_state | yes | yes | yes | yes (land issues) |
| barotropic_wave | yes | yes | yes | no (land issues) |
| inertia_gravity_wave | yes | yes | yes | yes (no land) |
| baroclinic | yes | yes | yes | yes (no land) |
| phillips_two_layer | yes | yes | yes | yes (no land) |
| barotropic_gyre | yes | yes | no (issue #55) | no |
| barotropic_double_gyre | yes | yes | no (issue #55) | no |
| lock_exchange | yes | yes | no | no |
| overflow | yes | yes | no | no |
| stommel_gyre_tracer | yes | yes | no (issue #55) | no |

### Experiment Hierarchy

Ordered from simplest to most complex. If an earlier experiment fails, later ones are
unlikely to succeed.

1. **rest_state_no_land** -- Zero dynamics, no boundaries. Pure numerical noise floor.
2. **rest_state** -- Zero dynamics with land. Tests boundary treatment.
3. **inertia_gravity_wave** -- Linear waves with analytical solution. Cleanest accuracy
   benchmark.
4. **barotropic_wave** -- Nonlinear barotropic wave propagation.
5. **baroclinic** -- Geostrophic adjustment and baroclinic dynamics.
6. **phillips_two_layer** -- Baroclinic instability with forcing.
7. **barotropic_gyre** -- Wind-driven single gyre (Stommel/Munk).
8. **barotropic_double_gyre** -- Wind-driven double gyre (Holland & Lin).
9. **lock_exchange** -- Density-driven gravity currents.
10. **overflow** -- Gravity currents over topography.
11. **stommel_gyre_tracer** -- Tracer transport in wind-driven gyre.

### Land Masking Strategy

| Category | Experiments | Land Handling |
|---|---|---|
| No land (all grids) | rest_state_no_land, inertia_gravity_wave | land_lat_threshold = 90 deg |
| Polar land strips | rest_state, lock_exchange, overflow | \|lat\| > 80 deg |
| Rectangular basin | barotropic_gyre, barotropic_double_gyre, stommel_gyre_tracer | 0-120 E, 15-75 N; land everywhere else |
| Spectral uses no land | barotropic_wave, baroclinic, phillips_two_layer | FV grids: land at 80 deg; spectral: 90 deg |

The spectral grid avoids land in most experiments because the spectral land-ocean
boundary treatment (tanh taper + grid-space masking + hyperdiffusion) does not yet
adequately suppress Gibbs ringing during time integration. The rest_state experiment
was changed to include land for spectral (2026-04-04) to expose and track this issue.

### Depth and Resolution Variations

| Experiment | H_max (m) | nlev | Why different |
|---|---|---|---|
| rest_state, rest_state_no_land | 5500 | 10 | Standard global ocean |
| barotropic_wave | 5500 | 10 | Standard global ocean |
| barotropic_gyre, barotropic_double_gyre | 5500 | 10 | Rectangular basin, uniform T/S |
| baroclinic | 5500 | 10 | Standard global ocean |
| phillips_two_layer | 3500 | 2 | Classic two-layer theory (Phillips 1954) |
| inertia_gravity_wave | 1000 | 2 | Barotropic-equivalent shallow basin |
| lock_exchange | 500 | 20 | Shallow + high vertical resolution for gravity currents |
| overflow | 2000 | 20 | Shelf-to-basin topography |
| stommel_gyre_tracer | 5500 | 10 | Rectangular basin, uniform T/S + tracer blob |

### Expected Field Ranges (for plotting and sanity checks)

| Experiment | eta (m) | SST (C) | Other |
|---|---|---|---|
| rest_state_no_land | (-1e-6, 1e-6) | (1.5, 21) | |
| rest_state | (-1e-6, 1e-6) | (1.5, 21) | |
| barotropic_wave | (-1.5, 1.5) | (1.5, 21) | |
| inertia_gravity_wave | (-1.2, 1.2) | (1.5, 21) | |
| baroclinic | (-0.1, 0.1) | (1.5, 21) | |
| phillips_two_layer | (-0.2, 0.2) | (8, 16) | |
| barotropic_gyre | (-0.02, 0.02) | (9.5, 10.5) | speed_sfc: (0, 0.15) |
| barotropic_double_gyre | (-0.02, 0.02) | (9.5, 10.5) | speed_sfc: (0, 0.15) |
| lock_exchange | (-0.05, 0.05) | (-1, 21) | |
| overflow | (-0.1, 0.1) | (-1, 21) | |
| stommel_gyre_tracer | (-0.02, 0.02) | (9.5, 10.5) | SSS: (33, 37) |

### Known Issues (as of 2026-04-04)

1. **Spectral land masking**: Rest state with land shows eta drift ~0.3 m and land
   leakage ~0.6 m on spectral T21. Root cause: spectral ringing at the land-ocean
   boundary is not fully suppressed by the tanh taper + hyperdiffusion during time
   stepping. Affects rest_state and barotropic_wave (which inherits rest_state IC).

2. **Spectral barotropic_wave amplitude**: max|eta| ~7.3 m on spectral vs ~0.1-0.2 m
   on FV grids. Connected to issue 1 via land masking in `_create_rest_state`.

3. **MPAS surface forcing**: MPAS ocean model lacks the surface forcing pipeline
   (physics_fn, surface_forcing parameter) that cubed_sphere and latlon have.
   Wind-driven experiments (barotropic_gyre, barotropic_double_gyre,
   stommel_gyre_tracer) are disabled for MPAS until this is implemented (issue #55).

4. **Gyre long-integration stability**: barotropic_gyre and barotropic_double_gyre
   go unstable at 30 days without dissipation (all mixing and bottom drag disabled).
   Needs lateral viscosity or bottom drag for long integrations. Quick mode (2 days)
   is stable and shows correct circulation patterns.

5. **Phillips two-layer local constants**: Module defines local `_A_EARTH` and
   `_G_EARTH` instead of importing from `constants.py`. Potential consistency risk.

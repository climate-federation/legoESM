# MPAS vs Lat-Lon 1° Comparison Experiments

**Status**: Two bugs found and diagnosed. Fixing Adcroft PGF first (2026-05-12).

## Goal

Run matched MPAS Voronoi (ico5, ~120 km) and lat-lon C-grid (180×360, 1°)
global ocean simulations with ETOPO bathymetry and identical idealized
forcing. Isolate grid-dependent vs physics-dependent behavior in
circulation, energetics, noise, and stability.

## Matched Configuration

### Bathymetry (identical treatment)

| Parameter | Value |
|-----------|-------|
| Source | ETOPO 1° NetCDF |
| H_max | 5500 m |
| H_min | 10 m |
| smoothing_passes | 2 |
| r_factor_max | 0.2 (MEO cap) |
| enforce_straits | True |
| fill_isolated_basins | False |
| Partial-cell snap | 30% (both grids) |
| north_cap_lat | 80° (both grids — MPAS also caps for fairness) |
| south_cap_lat | None (full Southern Ocean) |

### Surface Forcing (identical)

| Parameter | Value |
|-----------|-------|
| Scheme | "combined" (prescribed wind + T/S restoring) |
| Wind profile | "global_wind", τ_max = 0.1 Pa |
| tropical_wind_scale | 0.5 (halve tropical winds) |
| tropical_wind_lat_deg | 15° |
| T restoring | τ_T = 30 days, T_eq = 25°C, T_pole = 0°C, cosine |
| S restoring | τ_S = 30 days, S_star = 35 PSU |

### Horizontal Viscosity (matched)

| Parameter | Both |
|-----------|------|
| A_h | 1.0e4 m²/s |
| C_smag_lap | 0.33 |
| B_h | 0.0 |

Note: effective Smagorinsky viscosity differs because Δ differs
(uniform ~120 km on MPAS, varies ~111→19 km with latitude on lat-lon).
This is by design — same coefficient, grid-adapted effective viscosity.

### Vertical Mixing (matched)

| Parameter | Value |
|-----------|-------|
| Implicit vertical mixing | True (backward-Euler) |
| A_v | 1e-4 m²/s |
| K_v | 1e-5 m²/s |
| KPP | K_conv = 1.0 (convective mixing on) |
| Convection | Enhanced diffusion, K_conv = 1.0 |

### Tracer Transport (matched)

| Parameter | Value |
|-----------|-------|
| Tracer advection | TVD (Van Leer limiter) |
| K_h | 0.0 (GM/Redi handles lateral mixing) |
| GM/Redi | κ_GM = 600, κ_Redi = 600, S_max = 0.005 |
| GM/Redi slope scheme | "centered" (both; triads MPAS-unavailable) |

### Bottom Drag (matched)

| Parameter | Value |
|-----------|-------|
| bottom_drag_r | 1e-3 |
| bottom_drag_bbl_thickness | 100 m |
| bottom_drag_bg_velocity | 0.1 m/s (quadratic-with-floor) |

### Numerics (matched)

| Parameter | Value |
|-----------|-------|
| Barotropic solver | implicit_cn (θ = 0.55) |
| barotropic_diffusion_alpha | 0.01 (default) |
| maxvel_barotropic | 0.0 (no clip) |
| PGF scheme | "adcroft" (both) |
| EOS | Wright (nonlinear) |
| Freshwater | Virtual salt flux, S_ref = 35 |
| dt | 1200 s |

### Vertical Grid (matched)

| Parameter | Value |
|-----------|-------|
| n_levels | 20 |
| dz_surface | 20 m |
| dz_deep | 500 m |
| H_max | 5500 m |
| Stretching | Hyperbolic tangent |

### Initialization (matched)

- Start from rest: u = 0, η = 0
- T(z) = 2 + 18 · exp(z / 500 m), uniform horizontally
- S = 35 PSU uniform
- Same z-star coordinate object

### Run Parameters

| Parameter | Value |
|-----------|-------|
| dt | 1200 s |
| Initial test | 30 days |
| Production | 10 years |
| Output | Per-timestep scalars + daily restarts/snapshots |

### Per-Timestep Diagnostics (saved to CSV)

| Metric | Description |
|--------|-------------|
| max\|u\| | Maximum absolute velocity component |
| max\|η\| | Maximum absolute SSH |
| mean SST | Area-mean sea surface temperature |
| global KE | Volume-integrated kinetic energy |
| max CFL_h | Maximum horizontal advective CFL number |
| wall_s | Wall-clock time [s] |

### Daily Output

- Full restart file (T, S, u, v, η, H_bathy, land_mask)
- 6-panel diagnostic PNG (surface speed, SSH, SST, max-depth speed, SSS, deep T)

## Grid-Only Differences (unavoidable)

| Aspect | MPAS | Lat-Lon |
|--------|------|---------|
| Cell geometry | Quasi-uniform ~120 km | ~111 km (eq) → ~19 km (80°N) |
| Staggering | TRiSK (normal vel on edges) | C-grid (u,v on faces) |
| K_zeta_bih | 1e14 (TRiSK null mode) | N/A (C-grid has no null mode) |
| Smoothing topology | Voronoi neighbors | 4-neighbor roll |
| ETOPO sampling | Different cell centers | Different cell centers |
| Momentum advection | TRiSK PV flux (enstrophy) | Vector-invariant |
| Coriolis | Separate f·ū + Matsuno on u' | Forward-backward Matsuno |
| Smagorinsky application | A_smag × del2(u) single-pass | div(A_smag · strain) stress-tensor |

## Diagnostics to Compare

### Per-Year Scalars

- max|u|, max|η|, area-mean SST
- Volume-mean T at 1000 m
- Drake Passage transport [Sv]
- Global KE (depth-integrated)
- min(N²)

### Post-Run Maps & Sections

- SST, SSH maps (side-by-side)
- Barotropic streamfunction
- Depth-averaged velocity
- Equatorial zonal section (T, u, speed)
- Atlantic meridional section (T, v)
- Southern Ocean zonal section (u — ACC)
- Surface speed (noise comparison)

### Grid-Specific Diagnostics

- Neighbor velocity variance ratio (grid-scale noise metric)
- Effective Smagorinsky A_smag maps at surface
- Grid Reynolds number Re = U·Δx / A_eff

---

## Experiment Tree

Each experiment gets its own output directory under
`results/ocean/comparison_mpas_v_latlon/`. Naming convention:
`{grid}_{experiment_id}` (e.g., `mpas_e0a`, `latlon_e0`).

```
Exp 0: Baseline (adcroft PGF, matched config, 30 days)
├── mpas_e0a   — MPAS + adcroft PGF (use_h_actual=False) → BROKEN
├── latlon_e0  — lat-lon + adcroft PGF                   → OK
│
├─ Exp 0b: MPAS with centered PGF (workaround)
│  └── mpas_e0b — MPAS + centered PGF                    → OK
│
├─ Exp 0c: MPAS with adcroft PGF + use_h_actual=True (bugfix)
│  └── mpas_e0c — MPAS + adcroft + h_actual default fix  → NEXT
│
(future branches)
├── Exp 1: TVD sub-seafloor stencil fix
├── Exp 2: Increase A_h / B_h
├── Exp 3: Different C_smag_lap
├── Exp 4: SMC03 PGF
├── Exp 5: Higher vertical resolution
├── Exp 6: Longer spinup (1-10 years)
├── Exp 7: Visbeck adaptive GM
└── Exp 8: Realistic forcing (JRA55-do)
```

---

## Experiment Log

### 2026-05-12 10:25 — Exp 0a: MPAS + adcroft PGF (30 days)

- Output: `results/ocean/comparison_mpas_v_latlon/mpas/` (overwritten by 0b)
- Log: `results/ocean/comparison_mpas_v_latlon/mpas_adcroft_run.log`
- Config: matched baseline + `pgf_scheme="adcroft"`, `use_h_actual_pgf=False`
- Status: **COMPLETE**

| Metric | Day-30 |
|--------|--------|
| max\|u\| | 3.19 m/s |
| max\|η\| | 4.13 m |
| mean SST | 18.79 °C |
| max CFL | 0.017 |
| Wall time | 89 s |

**Observation:** max|η| = 4.1 m is far too large after just 30 days.
Led to Issue 2 investigation (see below).

### 2026-05-12 10:26 — Exp 0: Lat-lon baseline (30 days)

- Output: `results/ocean/comparison_mpas_v_latlon/latlon/`
- Log: `results/ocean/comparison_mpas_v_latlon/latlon_run.log`
- Config: matched baseline + `pgf_scheme="adcroft"`
- Status: **COMPLETE**

| Metric | Day-30 |
|--------|--------|
| max\|u\| | 1.36 m/s |
| max\|η\| | 0.22 m |
| mean SST | 18.35 °C |
| max CFL | 0.56 |
| Wall time | 131 s |

**Observation:** Stable, well-behaved. East-west equatorial SSH gradient
forming (physical Kelvin wave response to wind startup). Some noise
patterns visible in surface fields.

### 2026-05-12 11:11 — Exp 0b: MPAS + centered PGF (30 days)

- Output: `results/ocean/comparison_mpas_v_latlon/mpas/` (overwrote 0a)
- Log: `results/ocean/comparison_mpas_v_latlon/mpas_centered_run.log`
- Config: same as 0a but `pgf_scheme="centered"`
- Status: **COMPLETE**

| Metric | Day-30 |
|--------|--------|
| max\|u\| | 1.11 m/s |
| max\|η\| | 0.15 m |
| mean SST | 18.87 °C |
| max CFL | 0.007 |
| Wall time | 83 s |

**Observation:** Dramatically calmer than adcroft (max|η| 4.1→0.15 m).
Now comparable to lat-lon (0.22 m). Confirms Issue 2.

### 2026-05-12 11:15 — Three-way comparison (day 30)

| Metric | MPAS adcroft (0a) | MPAS centered (0b) | Lat-lon adcroft (0) |
|--------|-------------------|--------------------|--------------------|
| max\|u\| | 3.19 m/s | 1.11 m/s | 1.36 m/s |
| max\|η\| | 4.13 m | 0.15 m | 0.22 m |
| mean SST | 18.79 °C | 18.87 °C | 18.35 °C |
| max CFL | 0.017 | 0.007 | 0.56 |

### 2026-05-12 11:20 — Snapshot analysis

Visual comparison of MPAS (0b centered) vs lat-lon (0):
- **Similarities:** T/S evolution, large-scale SSH pattern, WBC development
- **MPAS-only:** Meridional S bands at ±30° latitude (led to Issue 1)
- **Lat-lon-only:** Equatorial east-west SSH gradient stronger/cleaner
- **Bathymetry:** MPAS appears visually smoother in SSH → confirmed by
  statistics (MPAS std(H)=1093m vs lat-lon std(H)=1455m) but neighbor
  depth jumps are similar magnitude. Visual difference is from larger
  cell size (120km uniform vs variable 19-111km).

### 2026-05-12 11:30 — S leakage analysis

Column-mean S deficit correlates with bottom depth:
- Shallow (0-1000m): S = 34.997 (−0.003 PSU)
- Deep (4000-5500m): S = 35.000 (no anomaly)

S was initialized uniform at 35.0 — no vertical gradient. The signal
is entirely numerical. S std on MPAS is 10× that of lat-lon (0.0012
vs 0.0001 PSU). Led to Issue 1 investigation.

---

## Issues Found

### Issue 1: TVD sub-seafloor stencil contamination

**Found:** 2026-05-12  
**Severity:** Low (0.003 PSU after 30 days, but accumulates)  
**Status:** Root cause identified. Fix deferred after Issue 2.

**Root cause:** The TVD horizontal advection stencil
(`advection_mpas.py:111-164`) uses an upwind-of-upwind (upup) cell
for the flux limiter. At level k, the upup cell may be below its own
seafloor, giving S=0. This contaminates the smoothness ratio:

    r = (S_donor − 0) / (S_downstream − S_donor) → huge → φ = 2

producing doubled diffusion at edges near topography. Salt
systematically drains from shallow to deep cells.

Secondary: vertical TVD ghost cell uses deepest level (sub-seafloor
S=0 for shallow columns), doubling diffusion at bottom active
interface.

All other code paths verified safe: implicit mixing, KPP, GM/Redi,
K_h diffusion, freshwater, conservation fixer.

**Fix:** Replace sub-seafloor upup values with donor cell value
(standard TVD boundary treatment: r=0 → φ=0 → pure upwind).
Also fix vertical TVD ghost cell to use bottom active value.

### Issue 2: Adcroft PGF p_prime / centroid grid mismatch

**Found:** 2026-05-12  
**Severity:** Critical (19× SSH inflation, renders adcroft unusable)  
**Status:** Root cause identified. Fixing now (default use_h_actual_pgf=True).

**Root cause:** On MPAS, `p_prime` is integrated using `dz_ref`
(reference thicknesses, column-independent) when
`use_h_actual_pgf=False` (the default). The Adcroft correction
computes centroid depths from `h_partial` (actual partial-cell
thicknesses). This mismatch means the correction "fixes" a centroid
offset that doesn't exist in p_prime, injecting large spurious forces
at every partial-cell step edge.

On lat-lon, `p_prime` is unconditionally integrated using `h_partial`,
so both p_prime and the correction are on the same grid → correct.

The centered PGF accidentally works because dz_ref is column-
independent → identical p_prime at level k → gradient is machine
zero at step edges. The Adcroft correction then adds pure noise.

**Fix:** Change `use_h_actual_pgf` default to `True` in
`mpas_config.py`. This makes MPAS match lat-lon behavior: p_prime
integrated on h_partial, correction on h_partial — consistent.

**Files:**
- `mpas_config.py`: line 183 (default False → True)
- `ocean_pe_mpas.py`: lines 169-183 (p_prime integration uses h_actual
  when flag is True), lines 411-418 (AC correction)
- Validated by: Exp 0c (MPAS + adcroft + h_actual=True)

---

## Next Steps (2026-05-12)

1. **Fix Issue 2:** Change `use_h_actual_pgf` default to `True` in
   `mpas_config.py`.

2. **Exp 0c:** Re-run MPAS 30 days with `pgf_scheme="adcroft"` and
   the fixed default. Expect max|η| ≈ 0.15-0.22 m (matching lat-lon).
   Compare snapshots side-by-side.

3. **If 0c works:** Both grids now run adcroft PGF correctly. This
   becomes the true baseline for comparison.

4. **Then:** Fix Issue 1 (TVD stencil) and re-run as Exp 1. Compare
   S fields to confirm fix.

5. **Then:** Extend to 1-year and 10-year runs for production comparison.

---

## Honest Caveats

1. The MPAS K_zeta_bih = 1e14 is extra dissipation with no lat-lon
   counterpart. It acts mainly on grid-scale vorticity noise, but
   its integrated effect on Drake transport or WBC structure is
   unknown.

2. Smagorinsky single-pass on MPAS vs stress-tensor on lat-lon is
   an implementation difference. For smoothly varying A_smag the
   difference is small, but near coastlines or fronts where
   ∇A_smag is large, the MPAS version misses the cross-term.

3. The 80°N polar cap removes MPAS's advantage of pole-free coverage.
   Arctic circulation cannot be compared in this setup.

4. Lat-lon cells near 80°N are ~19 km — effectively eddy-permitting
   resolution. The same physics (GM κ=600, A_h=1e4) may be
   inappropriate there. This is a known issue with lat-lon grids
   at high latitudes.

5. A_h=1e4 + C_smag_lap=0.33 blew up from cold start on MPAS when
   C_smag_lap was lowered to 0.15 (Step 2 of tuning plan). With
   C_smag_lap=0.33 this should be stable, but watch the first
   few months carefully.

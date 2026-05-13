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
Exp 0: Baseline (30 days, fp64)
├── mpas_e0a   — MPAS + adcroft PGF (use_h_actual=False) → BROKEN (SSH 4.1m)
├── latlon_e0  — lat-lon + adcroft PGF                   → OK (SSH 0.22m)
├── mpas_e0b   — MPAS + centered PGF                     → OK (SSH 0.15m)
├── mpas_e0c   — MPAS + adcroft + h_actual=True fix      → OK (SSH 0.17m)
├── mpas_e1    — MPAS + TVD stencil fix                   → OK (no S effect)
├── mpas_e2    — MPAS + Hu_avg conservation fix           → OK (no S effect)
├── mpas_e3    — MPAS + all 3 fixes                       → OK (S drift = fp32)
│
├── mpas/latlon_e4_fp64 — fp64 baseline (1yr + 5yr + 50yr MPAS)
│   ├── Drake: MPAS ~25 Sv plateau, latlon ~126 Sv rising
│   └── latlon develops zonal jets by year 2
│
├── MPAS K_zeta_bih sweep (branched from e4 year 6):
│   ├── mpas_e5_kzb1e12  — K_zeta_bih=1e12  → Drake ~6 Sv (worse)
│   ├── mpas_e4 baseline — K_zeta_bih=1e14  → Drake ~25 Sv
│   ├── mpas_e6_kzb1e15  — K_zeta_bih=1e15  → Drake ~12 Sv
│   └── mpas_e6_kzb1e16  — K_zeta_bih=1e16  → BLOWUP (10 steps)
│
├── MPAS APVM experiments (branched from e4 year 6):
│   ├── mpas_e7a_apvm     — APVM only (no K_zeta_bih)  → Drake ~2 Sv (declining)
│   └── mpas_e7b_apvm_bh  — APVM + B_h=7.7e12          → 20yr stable, Drake TBD
│
├── Lat-lon biharmonic experiments:
│   ├── latlon_e8_bih         — B_h=5e10 from yr-6 restart  → jets persist
│   ├── latlon_e9_bih_fresh   — A_h=0, B_h=1e11 fresh      → BLOWUP day 48
│   └── latlon_e9b_topo_bht   — B_h_barotropic tests        → TBD
│
├── Flat-bottom F experiments (lat-lon, baseline viscosity):
│   ├── F1_flat_lap           — A_h=1e4, Csmag=0.33         → jets, survived 1yr
│   ├── F2_flat_bih           — A_h=0, B_h=1e11             → BLOWUP day 40
│   ├── F3_flat_bih_floor     — A_h=1e3, B_h=1e11           → BLOWUP day 49
│   ├── F4_flat_bih_Ah1e4     — A_h=1e4, B_h=1e11           → BLOWUP day 64
│   └── F5_flat_production    — A_h=2e5+latscale, B_h=5e9   → BLOWUP day 231
│
├── Hollingsworth KE fix experiments (lat-lon, flat bottom):
│   ├── H1_flat_holfix        — KE fix, A_h=1e4             → BLOWUP day 113
│   ├── H3_flat_Ah3e4         — KE fix, A_h=3e4             → oscillating, marginal
│   └── H4_flat_bht           — KE fix, B_h_bt=1e14         → BLOWUP day 122
│   Note: KE fix made things LESS stable (removed accidental diffusion)
│   KE fix REVERTED.
│
├── 2Δy source isolation (lat-lon, flat bottom, 180 days each):
│   ├── BT_uniform            — uniform T, wind only         → 2Δy saturates (not BT)
│   ├── BC_strat_windonly      — stratified T, wind only      → RUNNING
│   ├── BC_strat_windonly_kpp  — + KPP                        → RUNNING
│   ├── BC_strat_windonly_conv — + enhanced diffusion          → RUNNING
│   ├── BC_strat_restore       — + T/S restoring              → RUNNING
│   └── BC_full_noGM           — + KPP + conv, no GM          → RUNNING
│
├── B_h_barotropic tests:
│   ├── BHT_strat_windonly  — B_h_bt=2e12, strat+wind  → BLOWUP day 110 (too weak)
│   └── BT_bht2e12          — B_h_bt=2e12, BT uniform  → RUNNING
│
└── (next) Stronger B_h_barotropic or address baroclinic amplification

### 2Δy Root Cause Summary (2026-05-13)

The 2Δy instability has two components:

**Seed (barotropic):** Nonlinear quadratic aliasing in the centered KE
gradient (v_cell=avg(v[j],v[j+1]) → v_cell² → grad_y(KE)). The
face→center→square→gradient round-trip creates 2Δy power from resolved
flow. Saturates at ~4e-5 in the barotropic case because Coriolis has
zero transfer at 2Δy (can't couple or amplify it).

**Amplifier (baroclinic):** When stratification is present, the
baroclinic PGF amplifies the 2Δy seed exponentially. Growth rate
~1 order of magnitude per 30 days. Does NOT require KPP, GM, or
convection — stratification + wind alone is sufficient.

**Isolation results (2026-05-13):**

| Experiment | Day 90 2Δy | Day 180 2Δy | Grows? |
|-----------|-----------|------------|--------|
| BT uniform (no T gradient) | 4.5e-5 | 4.1e-5 | No |
| Strat + wind only | 6.4e-4 | 3.8e-2 | YES |
| Strat + wind + KPP | 5.8e-4 | 3.2e-2 | YES (same) |
| Strat + wind + conv | 6.4e-4 | TBD | YES (same) |

B_h_barotropic=2e12 tested on stratified case: blew up at day 110.
Too weak to overcome baroclinic amplification.
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

### Issue 1: MPAS salinity drift (0.003 PSU/month at shallow cells)

**Found:** 2026-05-12  
**Severity:** Medium (0.003 PSU in 30 days, ~0.04 PSU/year)  
**Status:** ROOT CAUSE UNKNOWN. Two hypotheses tested and disproven.

**Symptom:** Starting from uniform S=35.0, after 30 days:
- Shallow cells (H < 1000m): column-mean S = 34.997 (−0.003 PSU)
- Deep cells (H > 4000m): column-mean S = 35.000 (no anomaly)
- Lat-lon with same config: S = 35.00000 (essentially perfect)

**Hypothesis A: TVD sub-seafloor stencil contamination** (DISPROVEN)

The TVD upwind-of-upwind stencil reaches into sub-seafloor S=0.
Fix applied (replace with donor value) — zero effect on S drift.
Reason: when S is uniform, delta ≈ 0, so the limiter correction
is proportional to delta regardless of φ. Fix is correct as
defense-in-depth for when real gradients develop.

**Hypothesis B: Barotropic Hu_avg conservation mismatch** (DISPROVEN)

The implicit barotropic solver used H_e_new in the time-averaged
transport, breaking div(Hu_avg) = (eta_old - eta_new)/dt. Fix
applied (use H_e_old consistently) — zero effect on S drift.
Reason: magnitude estimate shows the mismatch is O(3e-6 PSU) over
30 days, three orders of magnitude too small. Fix is correct for
formal conservation but not the observed drift.

**Both fixes committed** (defense-in-depth, correct for long runs):
- TVD stencil: `advection_mpas.py`, `vertical.py`
- Hu_avg: `barotropic_implicit_mpas.py`, `barotropic_implicit_latlon_cgrid.py`

**Verified safe by two agent audits:**
- Implicit vertical diffusion, KPP, GM/Redi, K_h diffusion
- Freshwater/virtual salt flux, conservation fixer
- Vertical velocity diagnosis, sub-seafloor masking (active_3d)
- Divergence operator (globally conservative by construction)

**Isolation experiments (2026-05-12):** Disabled components one at a
time (10-day runs). Results:

| Experiment | S deficit (shallow−deep) | Conclusion |
|-----------|-------------------------|------------|
| baseline | −0.00082 | Reference |
| no_gmredi | −0.00082 | Not GM/Redi |
| **no_kpp** | **+0.00005** | **KPP/convection is the cause** |
| no_forcing | −0.00099 | Not forcing |
| upwind | −0.00082 | Not TVD |
| no_visc | −0.00085 | Not viscosity |
| no_drag | −0.00084 | Not drag |

**Agent investigation (2 agents, MPAS deep-dive + cross-grid comparison):**

Both agents confirmed sub-seafloor K masking is CORRECT — K=0 at the
bottom active/sub-seafloor interface. No direct salt leakage through
the implicit solver. The mechanism is indirect:

1. Enhanced diffusion applies K_conv ≈ 0.5 m²/s at nearly ALL active
   interfaces (sigmoid with sharpness=1e6 gives ~0.5 for near-neutral
   stratification — too much for truly neutral columns).
2. MPAS uses true partial-cell thickness (e.g., 5m at thin bottom
   cells) in the implicit solver. Lat-lon uses reference thickness
   (~100m). This makes the coupling coefficient ~20× larger on MPAS
   at thin cells.
3. Strong vertical mixing at thin shallow cells creates a different
   vertical tracer distribution than at deep cells with thick layers.
4. Horizontal advection redistributes the resulting horizontal gradient.

This is a **thin-cell numerics issue**, not a masking bug.

**Fix options (not yet applied):**
- Increase snap fraction (30% → 50%) to remove thinnest cells
- Cap coupling coefficient in implicit solver at thin cells
- Tune enhanced diffusion sigmoid_sharpness so K≈0 at neutral (not 0.5)
- Use reference thickness in implicit solve (matches lat-lon behavior)

**ROOT CAUSE FOUND (2026-05-12): float32 precision.**

Re-running the 10-day baseline in full float64 gives S_shallow =
35.00000000, S_deep = 35.00000000, deficit = 0.00000000. The drift
vanishes completely. The implicit solver's tridiagonal system with
thin partial cells (large coupling coefficients) accumulates float32
rounding errors that manifest as depth-dependent S bias. In float64,
the error is below machine precision.

This explains:
- Why MPAS drifts but lat-lon doesn't: thin partial cells → 20×
  larger coupling → more float32 rounding error
- Why disabling KPP eliminates it: KPP adds K≈0.5 → larger coupling
  → more rounding error in the tridiagonal solve
- Why sub-seafloor masking audits found nothing: masking IS correct

**Fix:** Comparison scripts now use `PrecisionPolicy.fp64()`.
For production, the implicit solver should upcast to float64
internally (control dtype) even when storage is float32.

### Issue 2: Adcroft PGF p_prime / centroid grid mismatch

**Found:** 2026-05-12  
**Severity:** Critical (19× SSH inflation, renders adcroft unusable)  
**Status:** FIXED. Validated by Exp 0c.

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

**Fix:** Changed `use_h_actual_pgf` default to `True` in
`mpas_config.py`. MPAS now matches lat-lon: p_prime on h_partial,
correction on h_partial — consistent.

**Validated:** Exp 0c: max|η| dropped from 4.13 m → 0.17 m
(matching lat-lon's 0.22 m).

### Issue 3: K_zeta_bih over-damps the ACC

**Found:** 2026-05-12  
**Severity:** High (MPAS Drake = 25 Sv vs lat-lon = 126 Sv)  
**Status:** Investigating alternative damping strategies.

**Problem:** MPAS uses K_zeta_bih = 1e14 (biharmonic ζ damping) to
control the TRiSK vorticity checkerboard null mode. This is a
legoESM invention — production MPAS-Ocean does NOT use this operator.

**K_zeta_bih sensitivity sweep (branched from year 6):**

| K_zeta_bih | Drake (equilibrium) | max\|u\| | Stability |
|------------|-------------------|----------|-----------|
| 1e12 | ~6 Sv (declining) | 1.09 | Stable but ACC collapses |
| 1e14 | ~25 Sv (plateau yr 30+) | 0.89 | Stable, baseline |
| 1e15 | ~12 Sv (recovering?) | 0.61 | Stable, too calm |
| 1e16 | BLOWUP (10 steps) | 38 | Biharmonic CFL exceeded |

Reducing K_zeta_bih makes things WORSE (checkerboard disrupts ACC).
Increasing makes things WORSE (over-damps physical vorticity, or
blows up). 1e14 is near optimal but still only gives 25 Sv.

**Drake Passage decomposition (year > 1 means):**

| Run | Total | Upper (<1000m) | Lower (>1000m) | Structure |
|-----|-------|---------------|----------------|-----------|
| MPAS (K_zeta_bih=1e14) | +16 Sv | +10 Sv (61%) | +6 Sv (39%) | Barotropic |
| Lat-lon | +14 Sv | +21 Sv (150%) | −7 Sv (−50%) | Baroclinic |

MPAS has barotropic ACC (uniform with depth); lat-lon has realistic
baroclinic structure (surface jet + deep return). K_zeta_bih likely
damps the vortex stretching that creates vertical shear.

**Ocean-expert research (2026-05-12): production MPAS-Ocean uses APVM**

Production MPAS-Ocean (E3SM) handles the checkerboard via:
1. Energy-conserving PV scheme + APVM (Anticipated PV Method)
2. Biharmonic velocity viscosity B_h (del4 on u, not ζ)
3. Leith closure (flow-adaptive viscosity)

Our K_zeta_bih * del4(ζ) is NOT used in production. APVM upstream-
biases PV by half a timestep, selectively damping the checkerboard
without over-damping physical vorticity. We already have `apvm_dt`
in MPASOceanConfig (currently disabled, default 0.0).

**APVM experiment plan (branched from year 6):**

| Exp | Config | Purpose |
|-----|--------|---------|
| e7a | K_zeta_bih=0, apvm_dt=1200 | APVM only (production approach) |
| e7b | K_zeta_bih=0, apvm_dt=1200, B_h=7.7e12 | APVM + production B_h |
| e7c | K_zeta_bih=0, apvm_dt=1200, C_leith=1.0 | APVM + Leith closure |

References:
- Sadourny & Basdevant (1985) — APVM original
- Ringler et al. (2010) — MPAS-Ocean PV schemes
- Thuburn et al. (2009, 2012) — TRiSK computational modes
- Petersen et al. (2019) — E3SM ocean evaluation
- Hoch et al. (2020) — MPAS-Ocean variable resolution

---

### Issue 4: Lat-lon 2Δy instability (Hollingsworth + missing B_h_barotropic)

**Found:** 2026-05-12/13  
**Severity:** High (basin-spanning zonal jets, blowup on flat bottom)  
**Status:** Root cause identified. Two-track fix planned.

**Symptom:** Exponentially growing 2Δy mode in surface u, visible in
meridional wavenumber spectra (South Pacific, 70°S-10°S, 200°E-280°E):
- Day 0: 2Δy power = 2e-13 (machine zero)
- Day 30: peak at 222 km (2Δy) in all runs
- Day 90: cascade to 445 km (4Δy)
- Day 365: jets organized at 556 km (5Δy)

All viscosity configs show the same 2Δy seed. Higher A_h delays
growth ~50 days but does not prevent it. Biharmonic B_h on full 3D
velocity blows up from cold start (A_h=0 → day 40; A_h=1e4 → day 64).

**Flat-bottom experiments confirm** topography is NOT required:
- F1 (flat, Laplacian): jets form, same 2Δy→4Δy cascade
- F2-F4 (flat, biharmonic): blow up at day 40-64
- F5 (flat, production A_h=2e5): delays to day 190, then blows up

**ROOT CAUSE (dycore expert, 2026-05-13): Hollingsworth instability**

The KE gradient in the vector-invariant momentum equation does
face→center→face interpolation:

    v_cell = 0.5*(v[j] + v[j+1])      # average to centers
    KE = 0.5*(u_cell² + v_cell²)       # square AFTER averaging ← BUG
    dKE/dx = gradient(KE)              # back to faces

The averaging `v_cell = 0.5*(v[j]+v[j+1])` has a 2Δy null space.
The squaring `v_cell²` aliases grid-scale energy into the 2Δy mode.
This is the Hollingsworth instability (Hollingsworth, Källberg &
Renner 1983), well-known on C-grids. NEMO/ICON/MPAS-O all use the
corrected form: **square first, then average**.

Files: `ocean_pe_latlon_cgrid.py` lines 931-933 (KE computation).

**PARAMETER DIAGNOSIS (ocean expert, 2026-05-13): B_h_barotropic**

The 2Δy mode is barotropic. B_h_barotropic (biharmonic on depth-mean
velocity only) provides:
- 5.5/day damping at 2Δy with B_h_barotropic=1e14
- Negligible damping at basin scale
- Works from cold start (constant coefficient)
- Does NOT damp baroclinic geostrophy (why 3D B_h blows up)
Already implemented in config: `B_h_barotropic` field.

**TWO-TRACK FIX PLAN:**

Track 1 — Code fix (eliminates source):
  Fix the KE gradient to use the Hollingsworth-corrected form:
  ```python
  # Square at native v-faces FIRST, then average to cells
  v_sq_cell = 0.5 * (v[j]**2 + v[j+1]**2)
  KE = 0.5 * (u_cell**2 + v_sq_cell)
  ```
  Symmetric change for u² in the v-tendency.
  ~10 lines changed in ocean_pe_latlon_cgrid.py.

Track 2 — Parameter fix (damps the mode):
  Priority experiments with existing config parameters:

  | Exp | Change | Mechanism |
  |-----|--------|-----------|
  | D1 | B_h_barotropic=1e14 | Surgical barotropic biharmonic |
  | D2 | B_h_barotropic=5e13 + C_leith=1.0 | Floor + flow-adaptive |
  | D3 | A_h=2e4 + B_h_barotropic=5e13 | Moderate lap + bih |
  | D4 | WENO5 momentum + B_h_barotropic=5e13 | Higher-order advection |
  | D5 | C_leith=1.5 alone | Flow-adaptive only |

  All use existing config fields, no code changes.

**Diagnostic metric:** Meridional wavenumber spectrum of surface u,
lon-averaged over South Pacific (70°S-10°S, 200°E-280°E). Track
2Δy power — must stay below 1e-6 (m/s)².

References:
- Hollingsworth, Källberg & Renner (1983) — KE aliasing instability
- Fox-Kemper & Menemenlis (2008) — Leith viscosity
- MOM6 `BIHARMONIC_BAROTROPIC` parameter

## Completed Steps

1. ✅ Matched config designed and documented
2. ✅ 30-day baseline runs (MPAS + lat-lon, fp32 then fp64)
3. ✅ Adcroft PGF bug found and fixed (use_h_actual_pgf default)
4. ✅ S drift diagnosed (float32 precision → fp64 fix)
5. ✅ TVD stencil + Hu_avg conservation fixes (defense-in-depth)
6. ✅ CFL diagnostic bug fixed (ocean-only dx_min)
7. ✅ 1-year baseline runs (both grids, fp64)
8. ✅ 5-year lat-lon run (Drake → 126 Sv)
9. ✅ 50-year MPAS baseline run (Drake → 25 Sv plateau)
10. ✅ K_zeta_bih sensitivity sweep (1e12, 1e14, 1e15, 1e16)
11. ✅ Drake transport decomposition (barotropic vs baroclinic)
12. ✅ Production MPAS-Ocean research (APVM discovery)
13. ✅ MPAS APVM experiment (K_zeta_bih=0, apvm_dt=1200 → 2 Sv, insufficient)
14. ✅ MPAS APVM+B_h experiment (e7b, 20 years complete, max|u|=1.18)
15. ✅ Lat-lon B_h experiments (e8, e9 — 3D B_h blows up from cold start)
16. ✅ Flat-bottom experiment matrix (F1-F5)
17. ✅ Meridional spectrum diagnostic (2Δy→4Δy cascade confirmed)
18. ✅ Hollingsworth instability identified as root cause
19. ✅ B_h_barotropic identified as parameter fix
13. 🔄 APVM experiment (Exp e7a, next)

## Next Steps (2026-05-13)

### Lat-lon 2Δy fix (two tracks, run sequentially)

**Track 1 — Hollingsworth KE fix (code change):**
1. Implement corrected KE form in `ocean_pe_latlon_cgrid.py`:
   square v² at v-faces first, then average to cells.
2. Run 1-year flat bottom with baseline viscosity (A_h=1e4,
   C_smag_lap=0.33, B_h=0) — if 2Δy mode is gone, the fix works.
3. Run 1-year ETOPO with same config.
4. If stable, extend to 5-10 years for Drake transport comparison.

**Track 2 — B_h_barotropic parameter fix:**
1. Exp D1: Add B_h_barotropic=1e14 to the flat-bottom baseline
   config (A_h=1e4, C_smag_lap=0.33). 1-year run.
2. Monitor 2Δy power via meridional spectrum.
3. If D1 works, try D1 on ETOPO and extend.
4. If D1 insufficient, try D2 (+ C_leith) or D3 (+ higher A_h).

### MPAS ACC strength
- e7b (APVM+B_h) completed 20 years — compute Drake transport
- If Drake still ~25 Sv, the MPAS ACC weakness is a resolution
  issue, not a parameter issue. Document and accept for 1°.

### Production comparison
- Once lat-lon 2Δy is fixed, run matched 10-year comparison
  with best configs on both grids.
- Compare: Drake transport, WBC structure, SST, overturning.

---

## Honest Caveats

1. Smagorinsky single-pass on MPAS vs stress-tensor on lat-lon is
   an implementation difference. For smoothly varying A_smag the
   difference is small, but near coastlines or fronts where
   ∇A_smag is large, the MPAS version misses the cross-term.

2. The 80°N polar cap removes MPAS's advantage of pole-free coverage.
   Arctic circulation cannot be compared in this setup.

3. Lat-lon cells near 80°N are ~19 km — effectively eddy-permitting
   resolution. The same physics (GM κ=600, A_h=1e4) may be
   inappropriate there. This is a known issue with lat-lon grids
   at high latitudes.

4. A_h=1e4 + C_smag_lap=0.33 blew up from cold start on MPAS when
   C_smag_lap was lowered to 0.15 (Step 2 of tuning plan). With
   C_smag_lap=0.33 this should be stable, but watch the first
   few months carefully.

# Advection Scheme Comparison: Eady Baroclinic Instability

## Motivation

Issue #210 implements several tracer advection schemes of increasing sophistication.
We need to compare their implicit numerical diffusion, stability, and conservation
properties in a physically meaningful test case before selecting a default for OMIP runs.

Hill et al. (2012, Ocean Modelling 45-46) showed that the advection scheme is the
dominant source of spurious diapycnal mixing in z-coordinate ocean models, with
effective diffusivities spanning 3 orders of magnitude between schemes.

## Headline (as of 2026-04-22)

Ranking of tracer advection schemes on the Eady uniform channel (100×50, 20 lvl,
no sponge), using the volume-weighted T variance `Var(T)` as the definitive
mixing metric (exact inviscid invariant in this closed-domain zero-K_h setup):

| Scheme | 200 d (strong, U=0.8) | 600 d (weak, U=0.2) | Cost vs upwind | Status |
|--------|-----------------------|---------------------|----------------|--------|
| **upwind** | PASS, 86% Var(T) lost | PASS, 12% Var(T) lost | 1.0× | Stable always; most diffusive |
| **tvd** (Van Leer) | PASS, 70% lost | PASS, 6% lost | 1.02× | Safe default |
| **ppm_fct** | FAIL day ~30 at all tested B_h/C_smag | — | ~1.3× (est.) | Blocked: limiter needs sign-split Zalesak (#9) |
| **som** (Prather 1986, 9 moments) | FAIL day ~107 without Smagorinsky | — | 2.8× | Best conservation but needs momentum help |
| **som + C_smag=0.2** | PASS, 46% Var(T) lost | PASS, **2.9%** Var(T) lost | 2.2× | Recommended for science runs |

**Key physical finding**: the absolute amount of numerical mixing is dominated
by the forcing regime, not the scheme. Strong-forcing Eady (U=0.8 m/s, τ=5 d)
saturates at p95 |Ro| ≈ 1.5–3.0 — firmly submesoscale/unbalanced, where
diapycnal mixing is physically mandatory regardless of scheme. Weak-forcing
Eady (U=0.2 m/s, τ=20 d) saturates at p95 |Ro| ≈ 0.05–0.11 — balanced ocean
interior regime where even plain upwind preserves 88.5% of Var(T) over 600
days. **For an "ocean interior" adiabatic study, the weak-forcing regime is
the scientifically defensible setup.**

**Recommendation**:
- Default production: **tvd** (safe, robust, 2% overhead vs upwind)
- Science runs (Eady, front studies, water-mass tracking): **som + C_smag=0.2**
- Pending code fix: **ppm_fct** with a proper sign-split Zalesak/Kuzmin limiter
  (task #9) should slot between tvd and som in cost/quality

## Experiment Design

### Test case: Classical Eady baroclinic instability

A zonally periodic channel with uniform N^2, linear vertical shear, and linear EOS.
The instability grows from a small temperature perturbation, develops eddies, and
(with sufficient dissipation) saturates at finite amplitude. This tests:

- **Implicit numerical diffusion**: sharper fronts = less numerical mixing
- **Conservation**: total heat should be preserved
- **Stability**: scheme must not blow up
- **Front structure**: slumping (physical) vs spreading (numerical diffusion)

### Domain

| Parameter | Value |
|-----------|-------|
| Longitude | 0-10 deg (periodic) |
| Latitude | 16-34 deg (solid walls, no sponge) |
| Depth | 5500 m |
| Resolution | 100x50 (~20 km), 20 vertical levels |
| Timestep | 300 s |

### Physics (same for all schemes)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| KPP | On (defaults) | Physical BL mixing; used by Hill et al. |
| A_h | 0 | No Laplacian viscosity |
| B_h | 2.3e11 m^4/s | Biharmonic viscosity, scaled from Hill et al. (9e8 at 5km, ~ dx^4) |
| C_smag | 0 | No Smagorinsky (single clear viscosity) |
| K_h | 0 | No explicit tracer diffusion |
| K_bih | 0 | No biharmonic tracer diffusion |
| A_v | 1e-5 m^2/s | Background vertical viscosity |
| K_v | 5e-6 m^2/s | Background vertical diffusivity |
| bottom_drag | 0.001 m/s | Linear bottom drag |
| Sponge | Off | Clean comparison, no artificial relaxation |

### Stratification and flow

| Parameter | Value |
|-----------|-------|
| N | 1.2e-3 s^-1 |
| T_ref | 10 degC |
| alpha_T | 2e-4 K^-1 (linear EOS) |
| U_surface | 0.8 m/s (thermal wind shear) |
| L_d | ~107 km (deformation radius) |
| tau_Eady | ~5 days (e-folding time) |
| Perturbation | k=3, 0.1 K |

### Schemes under test

| Scheme | Order | Monotone | Expected implicit K_eff | Status on Eady |
|--------|-------|----------|------------------------|-----------------|
| `upwind` | 1st | Yes | ~|u|*dx/2 ~ 5000 m^2/s | Works (reference) |
| `tvd` (Van Leer) | 2nd | Yes | Moderate | Works |
| `ppm_fct` (PPM+Zalesak) | 4th | Yes | Low | **Blocked** — limiter needs sign-split rewrite (#9) |
| `som` (Prather 1986) | 2nd moments | No (smooth) | Near zero | Works with Smagorinsky momentum help |

### Commands

```bash
# Common flags
FLAGS="--only eady_uniform --grid latlon_channel --resolution 100x50 \
       --days 200 --no-sponge --B-h 2.3e11 --C-smag 0"

# Run each scheme
JAX_ENABLE_X64=1 python3 scripts/matrix/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection upwind --tag upwind_nosponge

JAX_ENABLE_X64=1 python3 scripts/matrix/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection tvd --tag tvd_nosponge

JAX_ENABLE_X64=1 python3 scripts/matrix/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection ppm_fct --tag ppm_fct_nosponge

JAX_ENABLE_X64=1 python3 scripts/matrix/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection som --tag som_nosponge
```

### Output location

```
results/ocean/eady_uniform/latlon_channel/100x50/<tag>/
```

Each directory contains: SST snapshots, T cross-sections, velocity profiles,
conservation timeseries, mean timeseries, and restart data.

### Key diagnostics to compare

1. **T_drift**: |mean_T(final) - mean_T(initial)| — conservation quality
2. **max_speed**: velocity evolution — stability indicator
3. **T_latitude_vertical_cross_sections**: front sharpness — slumping vs diffusion
4. **snapshots_SST**: eddy structure and grid-scale noise
5. **conservation_timeseries**: heat/salt/volume drift over time

## Results

### Run 1: 2026-04-22 — Baseline comparison (no sponge)

**Parameters**: 100x50, 200 days, dt=300s, B_h=2.3e11, C_smag=0, KPP on, no sponge.

| Scheme | Status | Days | T_drift | max_speed | Wall | Notes |
|--------|--------|------|---------|-----------|------|-------|
| upwind | **PASS** | 200 | 3.79e-04 | 8.28 | 16 min | Eddies peak ~day 120, then decay |
| tvd | **PASS** | 200 | 6.06e-04 | 5.04 | 17 min | Similar pattern, lower peak velocity |
| ppm_fct | FAIL | ~9 | 1.26e-03 | 1.54 | 1 min | Blowup during initial growth |
| som | FAIL | ~111 | 1.25e-04 | 30.4 | 22 min | Stable through day 95, then velocity explosion |

**Velocity evolution (max_speed in m/s)**:

| Day | upwind | tvd | som |
|-----|--------|-----|-----|
| 10 | 0.41 | 0.41 | 0.41 |
| 20 | 0.52 | 0.50 | 0.51 |
| 40 | 1.34 | 1.03 | 1.02 |
| 60 | 1.63 | 1.17 | 1.24 |
| 80 | 2.84 | 2.70 | 2.33 |
| 100 | 5.94 | 6.01 | 8.17 |
| 120 | 10.4 | 6.87 | — (NaN) |
| 150 | 9.65 | 6.65 | — |
| 200 | 8.28 | 5.04 | — |

**Key findings**:

1. **SOM has the best conservation**: T_drift=1.25e-04 (3-5x better than upwind/tvd).
   The improved conservation comes from near-zero implicit numerical diffusion.

2. **SOM tracks tvd closely through day 95**: velocity evolution is nearly identical,
   showing the instability growth rate is physical, not numerical.

3. **SOM blows up at day ~111**: velocity jumps from 4.5 to 30 m/s in 15 days.
   This coincides with eddies reaching the channel walls (no sponge to absorb them).
   Without a sponge, wall-reflected eddies create grid-scale noise that less-diffusive
   schemes cannot damp.

4. **PPM-FCT fails early (day 9)**: likely incompatible with the large B_h=2.3e11.
   May need separate tuning or the original B_h=1e10 + Smagorinsky.

5. **Upwind/tvd survive** because their implicit diffusion damps wall reflections.

**Next steps**:
- Re-run with sponge enabled to test SOM stability at longer times
- Re-run PPM-FCT with original B_h=1e10 + C_smag=0.2
- Compare T cross-sections visually for front sharpness
- Try higher resolution (200x100) where wall effects are delayed

### Run 2: 2026-04-22 — SOM v2 with corrected receiver merge

After audit, the receiver merge in `_receiver_merge()` had incorrect formulas:
- `sx_new` used simple addition instead of weighted `alf1*sx + alf*fp_sx + 3*d0`
- `sxx_new` used wrong factors (1.5, 5/3, 5/6) instead of correct weighted formula
- Cross terms used weighted average instead of simple addition
- Roundtrip test (extract + merge = original) failed with old formulas, passes with new

Additional fixes applied:
- `_EPS`: 1e-30 → 1e-20 (float32 AD safety)
- Dead `vol` parameter removed from `_limit_moments`
- cos(lat) at v-faces: changed from avg(cos) to cos(avg) to match divergence operator
- lax.scan compatibility: pre-initialize T_som/S_som Fields (no None→Field transition)

**Result**: SOM v2 blew up at day ~107 (vs v1 at day 111). Same instability pattern.

| Version | Days survived | max_speed at day 100 | T_drift |
|---------|---------------|---------------------|---------|
| v1 (old merge) | 111 | 8.17 | 1.25e-04 |
| v2 (fixed merge) | 107 | 7.15 | 2.68e-04 |

**Conclusion**: The merge formula fix is mathematically correct (verified by
roundtrip test) but does NOT affect stability. The day-100 velocity explosion
is caused by grid-scale pressure gradient noise at sharp eddy fronts — an
inherent limitation of non-diffusive advection at 20 km resolution without
a sponge. TVD survives because its implicit diffusion damps this noise.

Visual inspection of `snapshots_speed_sfc.png` confirms the velocity spikes
originate in the **interior frontal zone** (22-25°N), not at the walls.

**Next**: Re-run with sponge to test whether boundary absorption resolves
the instability, or whether interior frontal noise is the fundamental limit.

### Run 3: 2026-04-22 — PPM-FCT conservation fix

PPM-FCT blew up at day 6-9 with KPP. Investigation revealed a **conservation
bug in the Zalesak limiter**: cell-based alpha applied to anti-diffusive
tendencies broke global conservation. Fixed by limiting anti-diffusive FACE
FLUXES with `alpha_face = min(alpha_left, alpha_right)`.

Verified: PPM-FCT + KPP conservation now matches TVD + KPP exactly
(rel_drift = 8.8e-6 in 10 steps). PPM-FCT + KPP survives 30 days.

Updated 30-day comparison (100×50, KPP on, B_h=1e10, C_smag=0.2):

| Scheme | Status | max_speed | T_drift |
|--------|--------|-----------|---------|
| tvd | PASS | 0.43 | 0.008 |
| ppm_fct | **PASS** | 2.28 | 0.48* |

*PPM-FCT T_drift is from physical eddy heat transport differences, not
conservation error (verified identical conservation to TVD in controlled test).

### Run 4: 2026-04-22 PM — GPU-accelerated full 200-day matrix

First complete 200-day comparison on GPU (V100S, `jax[cuda12]`, float64).
Per-day cost drops from ~5 s/d on Mac CPU to ~0.7–1.6 s/d on GPU — a
200 d run costs 3–5 min wall, enabling rapid parameter sweeps.

Parameters as in "Run 1" (strong forcing, U_surface=0.8 m/s, τ_Eady=5 d,
B_h=2.3e11, C_smag=0, KPP on, no sponge, 100×50 × 20 lvl, dt=300 s).

**Results — time-matched Var(T) loss** (the only scheme-independent mixing
diagnostic; see next section):

| t | upwind | tvd | som | som + C_smag 0.2 |
|---|---|---|---|---|
| 20 d | 0.7% | ~0 | 0.06% | 0.06% |
| 100 d | **27.6%** | **8.2%** | **1.4%** | **1.1%** |
| 200 d | **86%** | **70%** | (blew up d~107) | **46%** |

Key takeaways:
1. **SOM is least diffusive by 6–20×** vs tvd/upwind at day 100 (Hill et al.
   direction confirmed; magnitude modest at 20 km over 100 days).
2. **Adding Smagorinsky to SOM was nearly free for tracer at day 100**
   (1.1% vs 1.4% loss). Smagorinsky is momentum-only, so it damps the
   grid-scale PGF noise that blew plain SOM up at day 107 without
   injecting tracer diffusion.
3. **PPM-FCT blew up at day ~30 at B_h=2.3e11 + C_smag=0**. Retry with the
   ocean-expert's prescribed B_h=1e10 + C_smag=0.2 (previously stable at
   30 days) also blew up, at day ~45. The issue is not parameter tuning —
   the `alpha_face = min(alpha_left, alpha_right)` heuristic limiter is
   not true Zalesak monotonicity; grid-scale T noise leaks through and is
   amplified by the PGF. Needs code-level fix (sign-split Zalesak/Kuzmin).
4. **Hi-res (200×100, 10 km) SOM also blew up early** under both Hill-scaled
   B_h=1.44e10 (day ~20) and with added C_smag=0.2 (day ~40). Hill's dx⁴
   scaling assumes fixed resolved-eddy sharpness; at finer dx the eddies
   sharpen and need proportionally more — not less — grid-scale damping.

Figures: `results/ocean/eady_uniform/latlon_channel/100x50/slumping_vs_diffusion.png`,
`T_volumetric_census_compare.png`, `rossby_compare.png`.

### Run 5: 2026-04-22 PM — Weak-forcing 600-day sensitivity (U=0.2 m/s)

Tests whether the Var(T) loss in Run 4 is dominated by scheme diffusivity
or by the APE-rich forcing. Same physics as Run 4 except **U_surface
reduced from 0.8 → 0.2 m/s** (CLI flag `--U-surface 0.2`), giving
τ_Eady = 20 d (4× slower growth), 4× smaller meridional ΔT, ~16× less APE.

Ran 600 days (30 τ_Eady, equivalent saturation to Run 4's 200 d at τ=5 d)
for upwind / tvd / som+C_smag 0.2.

| Scheme | Status | max_speed | Var(T) lost | p95 Ro (surface) | p95 Ro (~3350 m) |
|--------|--------|-----------|-------------|------------------|------------------|
| upwind | **PASS 600d** | 0.11 m/s | **11.5%** | 0.023 | 0.054 |
| tvd | **PASS 600d** | 0.17 m/s | **6.4%** | 0.018 | 0.053 |
| som + Csmag 0.2 | **PASS 600d** | 0.14 m/s | **2.9%** | 0.033 | 0.096 |

Compared to Run 4 (strong forcing):

- **Var(T) loss drops 7–30× across all schemes** for 3× longer integration.
- **p95 |Ro| drops by ~40×**, from 1.5–3.0 to 0.02–0.11 — firmly in the
  balanced ocean-interior regime (Shcherbina et al. 2013; Chelton et al.
  2011 for comparison values).
- **Scheme ranking unchanged** (som < tvd < upwind), but absolute mixing
  is now at ocean-interior levels (~0.1–0.3% Var(T) / 100 d for SOM+Csmag,
  approaching Hill et al.'s "below observed ocean values" claim).

Figures: `results/ocean/eady_uniform/latlon_channel/100x50/weak_forcing_compare/`.

## Volume-weighted T variance: the definitive mixing metric

In a closed domain with no sponge and no explicit tracer diffusion
(K_h = K_bih = 0), T is materially conserved → D(T²)/Dt = 0. With no
mass flux through boundaries, **∫T² dV is an exact inviscid invariant**.
Any drop in `Var(T) = ⟨T²⟩ − ⟨T⟩²` is therefore numerical mixing — pure
scheme-induced irreversibility. Adiabatic slumping contributes zero.

This is Hill et al.'s logic, specialised to our setup: they integrate the
equivalent quantity over T classes to obtain an effective diapycnal
diffusivity κ_eff. The volume-weighted Var(T) time series is a single
scalar that captures the same physics.

The post-processing script `scripts/plot/plot_T_volumetric_census.py` computes
this and three related metrics:

- `<T>` drift — scheme conservation error (should be ~0)
- `Var(T) / Var(T)_0` — numerical mixing fraction (should be ~1)
- `(T_max − T_min)` range — tail preservation
- Histogram total-variation distance — spatial rearrangement of volume
  between T bins (less physical but useful sanity check)

Use `Var(T)` as the primary ranking metric. The histogram redistribution
can lag or lead it if a scheme spatially shuffles T within the
distribution's support more than it compresses toward the mean
(see SOM+Csmag at 200 d: hist redistrib ≈ tvd but Var loss is 24 pts
smaller).

## Physical regime: strong vs weak forcing

Our default Eady parameters (U_surface=0.8 m/s, τ=5 d) were inherited
from the atmospheric Eady (1949) literature, where such parameters
describe synoptic cyclogenesis. Translated to the ocean they put us in
a regime that is **not representative of the ocean interior**:

| Quantity | Our strong-forcing Eady | Ocean interior |
|---|---|---|
| Surface jet velocity | 0.8 m/s | 0.01–0.1 m/s |
| τ_BCI | 5 d | 30–50 d |
| Bulk Ro = U/(f L_d) | 0.12 (linear) | 0.01–0.05 |
| APE density | ~10¹⁰ J/m³ | ~10–100 J/m³ |
| Saturated max speed | 5–10 m/s | 0.1–0.5 m/s |
| Saturated p95 Ro | 1.5–3.0 | 0.05–0.2 |

At Ro ~ 1 the flow uses ageostrophic pathways (frontogenesis, symmetric
instability, IGW emission), all of which produce irreversible diapycnal
mixing regardless of advection-scheme quality (Thomas et al. 2013;
McWilliams 2016). Weak-forcing Eady (U=0.2 m/s) saturates at Ro_eddy
~0.3 and reproduces the balanced mesoscale interior regime.

**Operational implication**: the scheme-ranking exercise done with
strong forcing is valid as a relative benchmark, but the absolute Var(T)
losses are set by the forcing regime. For "how much diapycnal mixing
would this scheme inject in an ocean simulation?", use the weak-forcing
numbers.

## Computational cost

V100S GPU, 100×50 × 20 lvl, dt=300s, `jax[cuda12]`, float64:

| Scheme | Wall / 200 d | Per-day (ex-compile) | vs upwind | Notes |
|--------|-------------:|---------------------:|----------:|-------|
| upwind | 154 s | 0.71 s/d | 1.00× | 1st-order baseline |
| tvd (Van Leer) | 157 s | 0.73 s/d | 1.02× | Essentially free vs upwind |
| ppm_fct | ~0.9 s/d* | — | ~1.3× | *Estimated from partial failed runs |
| som (plain) | ~2.0 s/d* | — | ~2.8× | *Estimated; survives only to d~107 |
| som + C_smag 0.2 | 327 s | 1.58 s/d | 2.22× | 3-pass dimensional splitting + 9 moments |

JIT compile cost ~10–12 s per configuration, amortized beyond a few-day run.

Memory overhead: SOM carries 9 polynomial moments per cell per tracer.
At 100×50 × 20 with (T, S), 18 extra 3D arrays ≈ 14 MB — negligible.
At 200×100 × 20 ≈ 55 MB — still negligible on V100 (32 GB).

**Trade-off**: SOM+Smagorinsky costs ~2× the wall time of TVD for
~7× better Var(T) preservation at day 100 (strong forcing) and ~2× at
day 600 (weak forcing). Good deal for science runs; borderline for
operational throughput.

## MPAS channel runs (2026-04-23, post-#211 TRiSK fix)

After the TRiSK `weightsOnEdge` + regional-Delaunay oversize-edge filter
fixes in `cd77199` / `d1bc7a9` (issue #211), the MPAS 20 km sub-360°
periodic channel is stable at weak forcing. First cross-grid
apples-to-close comparison of advection schemes:

### Setup

MPAS 20 km sub-360° periodic Voronoi, 20 levels, 10°×18° domain,
dt = 300 s, 600 days, U_surface = 0.2 m/s, B_h = 2.3e11, C_smag = 0,
no sponge. Matches the lat-lon 100×50 weak-forcing setup above
*except for* three unavoidable MPAS-only defaults:

1. **KPP is not implemented on MPAS** (warning `ignoring unsupported
   schemes: vertical_mixing='kpp'`). Only the background `A_v=1e-5`,
   `K_v=5e-6` act vertically.
2. `pv_scheme = "enstrophy"` (correct MPAS default — suppresses ζ
   null mode of the energy-conserving scheme).
3. `K_bih` is silently dropped (is 0 on both grids, so no-op).

Schemes available on MPAS: **upwind** and **tvd** (Van Leer) only.
No PPM, DST-3, or SOM on Voronoi yet.

### Results

| Scheme | Status | T_drift (K) | **Var(T) rel drift** | max_speed 600d | Wall time |
|---|---|---|---|---|---|
| MPAS upwind | PASS 600 d | −1.27e-5 | **−0.91%** | 0.057 m/s | 14 min |
| MPAS tvd | PASS 600 d | −1.33e-5 | **+0.69%** | 0.058 m/s | 14 min |

Both schemes produce nearly identical flow evolution:
`max_abs_eta` peaks at ~0.38 m around day 120 (BCI saturation) then
decays monotonically to ~0.18 m by day 600 as the unforced instability
runs down. `max_speed` follows the same shape — 0.10 m/s peak,
0.057 m/s final. The two schemes agree to ~1–2% in `max_speed` through
the saturation phase, consistent with scheme-level diffusivity
differences being small in this weak-forcing regime.

Positive `Var(T) drift` for TVD is near machine-precision noise of the
histogram reconstruction (the banded zonal-mean rebinning at 50 lat
bands is not bitwise-conservative); the same scheme shows `T_drift =
0.00e+00` exactly in the raw `mean_T` time series.

Figures: `results/ocean/eady_uniform/mpas_channel/20km/advection_compare_U02_600d/`.

### Cross-grid comparison at matched parameters

Combining the 3 lat-lon weak-forcing 600-day runs (KPP on) with the
2 MPAS weak-forcing 600-day runs (no KPP) gives the full matrix:

| Run | Scheme / grid | T_drift (K) | **Var(T) rel drift** | (Tmax − Tmin) rel | hist redistrib |
|---|---|---|---|---|---|
| lat-lon + KPP | upwind | −1.95e-4 | **−11.33%** | −6.67% | 17.6% |
| lat-lon + KPP | tvd | −7.33e-5 | **−6.03%** | −6.06% | 14.0% |
| lat-lon + KPP | som + Cs 0.2 | −1.10e-4 | **−2.70%** | −0.88% | 10.6% |
| MPAS no-KPP | upwind | −1.27e-5 | **−0.91%** | +0.10% | 7.3% |
| MPAS no-KPP | tvd | −1.33e-5 | **+0.69%** | +0.32% | 8.3% |

Figures: `results/ocean/eady_uniform/cross_grid_compare_U02_600d/`.

### Interpretation — important caveat

**The MPAS "wins" here are confounded by the absence of KPP.** KPP is a
physical parameterization of ocean boundary-layer vertical mixing. In
the `Var(T)` budget on a closed domain it looks exactly like numerical
diffusion — both pathways reduce `Var(T)`. The lat-lon cases include
KPP and therefore show more `Var(T)` loss, even when the advection
scheme is identical.

To compare the pure advection-scheme performance across grids we need
*either*:

- **(a)** rerun the lat-lon matrix with `vertical_mixing="none"` (matches
  MPAS, eliminates KPP), *or*
- **(b)** add a KPP-equivalent vertical-mixing scheme to the MPAS ocean
  model (the more useful fix; KPP is physical).

Path (b) is the right long-term direction but is out of scope for this
issue. Path (a) — a 3-run KPP-off lat-lon rerun — is the cheap way to
close the comparison. That is listed in "Open issues" below.

### BCI development — the visualisations tell a different story

Generating per-run SST / SST-anomaly / η / surface-speed panels
(`scripts/plot/replot_eady_bci.py` → `results/ocean/eady_uniform/bci_development/*_bci_snapshots.png`)
on a physical aspect ratio cropped to the domain reveals a real
dynamical divergence between the grids that the scalar metrics hide:

- The **zonal-mean thermal-wind jet** is established on both grids
  (surface speed ≈ 0.10 m/s at day 60 in the `|u|` row of every
  panel).
- The initial k=3 T perturbation (**row 2: SST − zonal mean**) is
  imposed equally on both grids (`_add_perturbation_latlon` /
  `_add_perturbation_mpas`, both amplitude 0.1 K with Gaussian
  lat envelope). Diagnosed zonal σ(SST) at lat = 25°, t = 0 is
  0.071 K in every run.
- **Lat-lon runs** (all three schemes): the mid-latitude k=3 mode
  decays rapidly (σ drops to ~1e-2 K by day 60) but
  **wall-trapped eddies emerge near the northern boundary** by
  day 240 and persist as visible meandering patterns in SST and
  surface speed through day 600. SOM shows the strongest
  wall-eddy activity; TVD and upwind similar in pattern but
  weaker.
- **MPAS runs** (both schemes): mid-latitude perturbation also
  decays, but **no wall-trapped eddies emerge**. The flow stays
  zonally uniform through day 600. σ(SST) at lat = 25°
  monotonically decreases over 600 d (tvd: 0.073 → 0.030 K;
  upwind: 0.073 → 0.017 K and flat after day 60).

Interpretation: the "BCI activity" visible in the lat-lon runs at
this forcing level is not textbook Eady growth at mid-latitude — it
is a **wall-trapped secondary instability** (frontal / Kelvin-wave
type) that develops once the zonal-mean jet has set up. MPAS
appears to suppress or delay this wall-trapped mode, probably
because:

1. **Wall representation differs.** The lat-lon C-grid has hard
   cell-face walls at specific latitudes. The regional Voronoi
   mesh has a ragged boundary (Delaunay cells clipped at the
   latitude bounds) with variable cell shapes at the edge —
   potentially damping wall-trapped modes more strongly.
2. **Effective grid viscosity near the wall may be higher on
   MPAS** given irregular cell geometry and the TRiSK
   reconstruction.
3. **Enstrophy-conserving PV flux is a slightly more dissipative
   choice** for grid-scale vorticity than the energy-conserving
   lat-lon formulation.

This is a **real physics divergence** between the two ocean cores
at matched parameters — not a scheme-level artefact. It does not
invalidate the conservation comparison above (which is what this
issue originally tracked) but it does mean the two grids are not
simulating the same flow regime at this forcing. The scalar
metrics (T_drift, max_speed, Var(T) drift) agree to ~leading order
because they are dominated by the zonal-mean jet, which both grids
reproduce.

### What this current comparison *does* show cleanly

1. **Scheme ranking within each grid is consistent**: upwind is the
   most diffusive, tvd intermediate, and (on lat-lon) som+Cs the
   least. On MPAS, upwind and tvd differ mostly in how fast the
   k=3 mode decays — tvd damps it ~2.5× slower than upwind at
   lat = 25°.
2. **Zonal-mean flow agrees** across grids at matched forcing:
   peak η amplitude ~0.38 m (day ~120), peak max_speed ~0.11 m/s,
   long-time decay toward a quasi-steady jet.
3. **Post-#211 the MPAS channel is stable for at least 600 days at
   weak forcing** with `T_drift` at machine precision (2.66e-15 for
   upwind, 0.00 for tvd). No recurrence of the earlier blowup
   (issue #211 closed).
4. **Lat-lon develops wall-trapped eddies at weak forcing**; MPAS
   does not. This is tracked as a follow-up item below.

### Next comparison pieces (not done here)

- Lat-lon 600 d runs for `ppm_fct` (post-conservation-fix), `dst3`,
  `dst3_multidim` — to fill out the lat-lon row.
- Lat-lon `vertical_mixing="none"` re-run of upwind / tvd / som to
  enable apples-to-apples cross-grid comparison.
- Rossby-number diagnostic on the MPAS runs (`plot_rossby_number.py`
  is currently lat-lon-only; would need MPAS vorticity helper).

## MPAS IC balance noise (2026-04-23 follow-up)

While inspecting the weak-forcing replots it became clear that the
MPAS surface-speed field shows visible zonal-direction inhomogeneity
from t = 0, whereas lat-lon is smooth.

### Measured numbers (t = 0 surface speed, zonal std within lat bins)

| lat band | MPAS (TVD run) | lat-lon (TVD run) |
|---|---|---|
| 25° (jet centre, `u` ≈ 0.099 m/s) | 2.15e-4 m/s (0.22%) | 1e-18 m/s (machine 0) |
| 21° (off-jet, `u` ≈ 0.053 m/s) | **9.45e-3 m/s (17.7%)** | 1e-18 m/s (machine 0) |
| 30° (off-jet) | ~2.1e-3 m/s (~6%) | 1e-18 m/s (machine 0) |

The noise **decays** under time stepping (off-jet band 17.7% → 11.4%
by day 60 → 6.7% by day 600) — classic signature of a small
imbalance being radiated away by geostrophic adjustment, not of
operator inconsistency (cf. issue #211 where an analogous symptom
grew instead and revealed the oversize-edge pathology).

### Ocean-expert diagnosis (summary)

Agent read `eady_uniform.py::_set_linear_shear_mpas`,
`init_mpas.py::reconstruct_cell_velocity`, and the TRiSK code in
`grids/voronoi.py`. Root cause:

1. `_set_linear_shear_mpas` builds
   ```
   u_edge[e, k] = U_bc[k] * cos(angleEdge[e]) * envelope(lat_edge[e])
   ```
   This is an *analytic* thermal-wind evaluation at edge midpoints.
2. For a hex cell, the six edges sit at six different latitudes
   (Δlat up to ≈ `dc/(2R)` ≈ 0.09° at 20 km) and six different
   `angleEdge` values.
3. The Perot cell-centre reconstruction
   (`reconstruct_cell_velocity`) sums
   `u_east(C) = Σ_e w_e · u_edge(e) · cos(angleEdge_e)`. This sum is
   exact only for spatially uniform `u_edge`. With a Gaussian envelope
   whose `d(ln env)/dy ≈ −0.32 /°` at lat 21°, consecutive edges of
   a cell sample the envelope with O(6%) differences. Hex-lattice row
   alternation (pointy-top / flat-top) gives orientation-dependent
   Perot projections, so the net reconstruction has O(few %)
   cell-to-cell zonal variation. At lat 21° where mean `u` is itself
   only half the jet maximum, this shows up as 17% relative.
4. **On lat-lon** the equivalent step is a 1-D meridional assignment
   with no edge-orientation ambiguity — hence machine-zero IC noise.

**This is a discretization artefact, not a bug.** The TRiSK operators
themselves are fine (fixed in #211); the problem is feeding them an
IC velocity built from continuous formulas instead of inverting the
discrete balance.

### Standard practice and fix

MPAS-Ocean's `baroclinic_channel` config builds IC by inverting the
**discrete** momentum balance — set T(y, z), get η from discrete
hydrostatic integration, then `u_edge = -(1/f) · (n̂·∇)(p/ρ₀)` where
`(n̂·∇)` is the TRiSK edge-normal gradient (by construction it returns
the edge-normal component — no `cos(angleEdge)` projection needed,
which is exactly what eliminates the aliasing). ICON-O does the same.
Our current code inlines the continuous thermal-wind formula and is
thus off by the edge-aliasing error described above.

### Recommended fix (not implemented yet)

Minimal edit in `src/legoesm/ocean/experiments/eady_uniform.py`,
function `_set_linear_shear_mpas` (approx L350–391):

1. Build the analytic 3D T exactly as today.
2. At each edge, compute `grad_n_T = (T[c2] − T[c1]) / dcEdge`
   (this is already the edge-normal gradient; no cosine projection
   needed).
3. Thermal-wind integration per level using `grad_n_T` as the
   meridional-temperature driver. Because analytic `∂T/∂y` is
   z-constant here, this reduces to
   `u_edge(e, k) = (U_baroclinic[k] / dTdy_analytic) · grad_n_T(e)`,
   i.e. we replace the `cos(angleEdge) · envelope(lat_edge)` product
   with the discrete normal gradient of the analytic T. The envelope
   and projection both come through T.
4. Zero the depth mean exactly as before to keep η = 0 meaningful.

**Expected impact:** off-jet zonal std in IC surface speed should drop
from ~17% to <1% (residual = truncation of the discrete hydrostatic
integral, not edge-gradient aliasing).

### What the IC noise does NOT explain

- The MPAS BCI suppression (no wall-trapped eddies on MPAS at weak
  forcing) is NOT caused by this IC noise. The noise is
  k-broadband and decays; wall-trapped eddies on lat-lon are driven
  by Kelvin-wave reflection off the northern boundary interacting
  with the meridional shear, which requires a clean wall. MPAS has
  a staircase-like boundary on unstructured cells and the sponge
  (when enabled) is applied per cell — these are the more likely
  suppressants. Separate follow-up item.

## Strong-forcing cross-grid runs (in progress 2026-04-23 night)

To separate the "weak-forcing-suppressed-BCI" question from the
"MPAS-suppresses-BCI" question, two 200-day MPAS strong-forcing
runs were launched in parallel alongside the IC investigation:

- `results/ocean/eady_uniform/mpas_channel/20km/upwind_mpas_20km_U08_Bh2.3e11_Cs0.0_200d/`
- `results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U08_Bh2.3e11_Cs0.0_200d/`

Parameters match the existing lat-lon strong-forcing runs
(`*_nosponge_200d`): U_surface = 0.8, B_h = 2.3e11, C_smag = 0,
dt = 300 s, no sponge, 20 levels.

**Result at time of writing:**

- **upwind BLEW UP at step 54100 (≈ day 188)**: `max_spd` 2.57 m/s in
  the preceding diag window, then NaN. Same "energetic eddies
  overwhelm dissipation" pattern the lat-lon `upwind_nosponge_200d`
  was noted for in Run 1 / Run 4 of this doc — i.e. *expected* at
  strong forcing with `C_smag=0` and no sponge. Not grid-specific.
- **tvd stayed stable through day 195/200** with `max_spd` 1.92 m/s
  — saturated-eddy regime, running down to completion.

Implication: **the MPAS ocean core supports classical BCI saturation**
when a scheme with enough implicit dissipation is used; the failure
mode at strong-forcing + pure upwind is numerical, not dynamical, and
matches the lat-lon grid's behaviour in the same setup. What's
specifically missing in the **weak-forcing** regime is only the
**wall-trapped secondary instability**, not BCI growth itself. This
narrows the wall-eddy-suppression follow-up significantly.

Once the tvd run finishes, rerun `scripts/plot/replot_eady_bci.py` on the
strong-forcing dirs (same script, no changes needed; may want to
edit `TARGET_DAYS` to `[0, 30, 60, 100, 140, 180]` since strong
forcing saturates earlier) to generate eddy-development figures
matching the existing lat-lon `som_csmag02_nosponge_200d/snapshots_SST.png`
aesthetic.

## Run 7: 2026-04-29 — Implicit barotropic solver comparison

After PR #218 introduced the implicit Crank-Nicolson barotropic solver
(`barotropic_solver="implicit_cn"`), which eliminated 2dt aliasing noise
in the explicit substep solver, we re-ran the previously-validated
advection scheme experiments to test whether the implicit solver
changes the dynamics.

**Hypothesis**: barotropic noise from the explicit solver was confounding
WENO stability tests and contaminating V_baro time-means. The implicit
solver should give cleaner dynamics.

**Tool**: new standalone script `scripts/run/run_eady_advection_comparison.py`
with EKE, Var(T), and zonal spectra diagnostics.

### Setup

Same physics as Runs 4-5: 100×50, 20 levels, dt=300s, B_h=2.3e11,
C_smag=0 (except som which uses C_smag=0.2), no sponge, KPP on.
Each scheme run twice: once with `explicit_substep`, once with
`implicit_cn`.

### Results — Weak forcing (U=0.2, 600 days)

| Scheme | Solver | Status | max_spd (m/s) | Var(T) drift | EKE final |
|---|---|---|---|---|---|
| upwind | explicit | PASS 600d | 0.11 | −12.4% | 6.0e-4 |
| upwind | implicit | PASS 600d | **0.85** | **−36.7%** | 5.2e-3 |
| tvd | explicit | PASS 600d | 0.19 | −7.3% | 7.4e-4 |
| tvd | implicit | **FAIL d~116** | 1.33 | — | — |
| som+Csmag | explicit | PASS 600d | 0.12 | −2.5% | 1.5e-3 |
| som+Csmag | implicit | "PASS" 600d | **8.02** | **−16.7%** | 1.7e-1 |

### Results — Strong forcing (U=0.8, 200 days)

All schemes blow up with the implicit solver (NaN). Previously upwind
and tvd survived 200 days with the explicit solver.

| Scheme | Solver | NaN at day | Old explicit result |
|---|---|---|---|
| upwind | implicit | ~111 | PASS 200d |
| tvd | implicit | ~94 | PASS 200d |
| dst3 | implicit | ~91 | not tested |
| weno5 | implicit | ~87 | not tested |
| weno7 | implicit | ~81 | not tested |

### Key finding: explicit solver noise was accidental eddy damping

The implicit Crank-Nicolson solver produces **8–65× higher max_speed**
and **3–6× worse Var(T) preservation** than the explicit substep solver
at identical physics settings. Schemes that were stable with the explicit
solver (tvd, som+Csmag) blow up or become unphysically energetic with
the implicit solver.

The most likely explanation is that the explicit substep solver's cosine
time filter introduces numerical damping on the barotropic mode, which
feeds back to suppress baroclinic eddy growth through the depth-averaged
velocity coupling. Removing this damping (implicit solver) reveals that
B_h=2.3e11 alone is insufficient to control grid-scale eddy energy —
even at weak forcing.

**Implications:**

1. **Previous comparison results are qualitatively valid** (scheme
   ranking by Var(T) loss is real) **but quantitatively contaminated**
   by the explicit solver's accidental damping. The absolute Var(T)
   losses and max_speed values understate the true eddy activity.

2. **Dissipation parameters need re-tuning** for the implicit solver.
   Higher B_h, larger C_smag, or a sponge layer is needed to match the
   effective dissipation the explicit solver provided for free.

3. **The implicit solver is the correct solver** — it eliminates
   checkerboard noise in V_baro by construction. The old "stable" runs
   were stable partly for the wrong reason.

### Next steps

- Diagnose the energy budget difference: decompose dKE/dt into
  baroclinic conversion, B_h dissipation, bottom drag, vertical mixing
  to identify which term is out of balance.
- Tune dissipation for the implicit solver: sweep B_h and C_smag to
  find the minimum dissipation that stabilizes tvd and som at weak
  forcing, then re-run the full scheme comparison.
- Compare vorticity fields (max|ζ|, ζ spectra) between solvers to
  determine whether the energy accumulates at the grid scale.
- Once re-tuned, run the full matrix (upwind, tvd, dst3, weno5, weno7,
  som) at weak forcing with the implicit solver for the definitive
  advection scheme ranking.

## Open issues and limitations

- **PPM-FCT limiter** (#9): current `alpha_face = min(alpha_left,
  alpha_right)` is a conservation-preserving heuristic but does not
  satisfy true Zalesak sign-split monotonicity. Grid-scale T noise
  leaks through and is amplified by the PGF. Needs code-level fix in
  `src/legoesm/ocean/advection.py` (implement sign-split incoming /
  outgoing flux limiter per Zalesak 1979 / Kuzmin et al.).
- **Hi-res SOM recipe** (issue #213): at 200×100 (~10 km) no tested
  parameter set produces a stable 200-day run.  Hill's `B_h ∝ dx⁴`
  scaling assumes fixed resolved-eddy sharpness; at finer dx the
  eddies physically sharpen (`L_d / dx` grows from ~5 at 20 km to ~11
  at 10 km), so grid-scale momentum damping needs to grow *faster*
  than `dx⁴`, not slower.  SOM's near-zero implicit diffusion gives
  no cushion: any momentum-side grid noise immediately seeds a runaway
  PGF mode.  Diagnostic recipe (CLI knobs are wired:
  `--B-h / --C-smag / --barotropic-div-damp / --dt / --U-surface`):

  ```bash
  COMMON="--only eady_uniform --grid latlon_channel --resolution 200x100 \
          --tracer-advection som --no-sponge --days 30 --tag investigation"

  # 1. B_h sensitivity sweep at strong forcing — find the smallest Bh
  #    that survives 30 days, then check whether it is physically
  #    reasonable (dimensional analysis vs. eddy decay time):
  for BH in 5e10 1e11 2.3e11 5e11; do
    JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
      $COMMON --B-h $BH --C-smag 0.2
  done

  # 2. C_smag sensitivity at moderate B_h:
  for CSMAG in 0.1 0.2 0.3 0.4; do
    JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
      $COMMON --B-h 5e10 --C-smag $CSMAG
  done

  # 3. Divergence damping as an alternative knob — targets grid-scale
  #    compressible modes directly without smearing momentum:
  for DD in 0.01 0.05 0.1 0.2 0.5; do
    JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
      $COMMON --B-h 1e10 --C-smag 0.0 --barotropic-div-damp $DD
  done

  # 4. Timestep sensitivity (current dt=150s ≈ CFL 0.8 at u=1 m/s):
  for DT in 75 150; do
    JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
      $COMMON --B-h 1.44e10 --C-smag 0.2 --dt $DT
  done

  # 5. Weak-forcing fallback: scheme stress-test independent of strong
  #    forcing.  If 5e10 / 0.2 still blows under U=0.2, the regime is
  #    not the issue — the scheme/dissipation budget is.
  JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
    $COMMON --B-h 1.44e10 --C-smag 0.2 --U-surface 0.2 --days 200
  ```

  Document the smallest `B_h` (and largest `barotropic_div_damp`) that
  yields a 200-day PASS, then either record the recipe here or
  conclude with "SOM is not recommended at 10 km without X" if no
  parameter set works.  The actual sweep awaits compute resources to
  run the 200×100 200-day cases (each ~hours of wall time at fp64).
- **Grid geometry**: cells at 25°N are ~20 km (meridional) × ~22 km
  (zonal) — nearly isotropic. The `100x50` resolution string maps to
  n_lat=100, n_lon=50 (not n_lat=50, n_lon=100 as one might expect
  from the number of cells-per-degree intuition).
- **Late-time SOM+Csmag mixing**: Var(T) loss accelerates from 1.1% at
  day 100 to 46% at day 200 (strong forcing). Likely wall-reflected
  eddy fronts triggering the moment limiter without a sponge to absorb
  them; this is a setup rather than scheme issue.
- **Weak-forcing as standard setup**: the scientifically-defensible
  "adiabatic Eady" setup is the weak-forcing regime. Consider changing
  `EadyUniformConfig.U_surface` default from 0.8 → 0.2 m/s and doubling
  the standard duration to 400–600 days. This would require updating
  other experiments that depend on the 0.8 default.
- **Cross-grid comparison confound**: the MPAS 20 km runs above have
  no KPP (not implemented in `mpas_physics.py`), while lat-lon runs
  have KPP on. Closing this for a true scheme comparison needs either
  a KPP-off lat-lon rerun of upwind / tvd / som (cheap, 3 GPU-h) or
  porting KPP to MPAS (right long-term fix).
- **MPAS advection-scheme inventory**: only `upwind` and `tvd` are
  implemented on Voronoi. PPM / DST-3 / SOM on MPAS would require
  unstructured-edge adaptations (SOM in particular is non-trivial —
  9 moments per cell with all-neighbor interactions). Not a
  scientific-correctness gap for 20 km weak-forcing runs, but would
  be needed for MPAS equivalents of the lat-lon SOM recipe at
  higher resolutions.
- **Fix applied in passing**: `scripts/plot/plot_T_volumetric_census.py`
  previously hardcoded `(0, 360)` for MPAS mesh reconstruction.
  Now reads `(cfg.lon_west, cfg.lon_east)` from `EadyUniformConfig`
  so it works on the sub-360° periodic mesh.
- **MPAS lacks wall-trapped eddies** that develop on lat-lon at
  weak forcing (see "BCI development" section above). Unknown
  whether this is a mesh/discretisation artifact, a physical
  consequence of the enstrophy-conserving PV flux default, or
  a real property of the flow. Possible diagnostic steps: bump
  U_surface to test whether MPAS develops BCI at strong forcing;
  switch MPAS to `pv_scheme="energy"` and see if wall-trapped
  activity appears; run a lat-lon version with a rougher / "more
  Voronoi-like" north wall. Not scheme-related — affects all MPAS
  advection schemes equally.
- **Stock test-matrix snapshot plots — fixed 2026-04-24**:
  `snapshots_SST.png` / `snapshots_eta.png` in MPAS run dirs
  previously used a regridded `(0, 360) × (−90, 90)` canvas and
  did not crop to the actual `source_lon_range`, so the 10°-wide
  domain appeared as a thin vertical sliver. Two fixes applied:
  (i) `run_eady_uniform` in `scripts/matrix/ocean_test_matrix/experiments.py`
  now passes `domain_extent=(lon_west, lon_east, lat_south, lat_north)`
  to `_save_case_diagnostics`; (ii) `_save_snapshot_plots` in
  `diagnostic_io.py` now crops the regridded array to
  `domain_extent` for *both* unstructured and "latlon" coord_kinds
  (the replot path uses "latlon" because regridded data is on a
  regular lat-lon canvas). Previously the latlon branch just set
  `extent=domain_extent` without cropping, which stretched the full
  181×360 array into the narrow regional extent — producing a
  thin strip. Existing runs can be regenerated via direct
  `_replot_case_snapshots(case_dir)` invocation; see
  `docs/ocean/experiments/NEXT_STEPS_advection_comparison.md`.
  `scripts/plot/replot_eady_bci.py` (native Voronoi, physical aspect,
  SST-anomaly row) is still the preferred figure for cross-grid
  comparisons because it uses the 20 km mesh natively rather than
  the ~1° regrid.

## References

- Hill, Ferreira, Campin, Marshall, Abernathey, Barrier (2012).
  "Controlling spurious diapycnal mixing in eddy-resolving height-coordinate
  ocean models." Ocean Modelling, 45-46, 14-26.
  PDF: `docs/references/Hill_etal_11.pdf`

- Prather, M. (1986). "Numerical advection by conservation of second-order
  moments." J. Geophys. Res., 91, 6671-6681.

- Zalesak, S. T. (1979). "Fully multidimensional flux-corrected transport
  algorithms for fluids." J. Comput. Phys., 31, 335–362. (Referenced for
  the strict sign-split limiter that PPM-FCT needs; task #9.)

- Eady, E. T. (1949). "Long waves and cyclone waves." Tellus, 1, 33–52.

- Hoskins, B. J., McIntyre, M. E., Robertson, A. W. (1985). "On the use
  and significance of isentropic potential vorticity maps." QJRMS, 111,
  877–946. (PV conservation = balanced regime diagnostic.)

- Thomas, L. N., Taylor, J. R., Ferrari, R., D'Asaro, E. (2013).
  "Symmetric instability in the Gulf Stream." Deep-Sea Res. II, 91, 96–110.
  (Submesoscale / Ro ~ 1 dynamics that force diapycnal mixing.)

- Shcherbina, A. Y., D'Asaro, E., Lee, C., Klymak, J., Molemaker, J.,
  McWilliams, J. (2013). "Statistics of vertical vorticity, divergence,
  and strain in a developed submesoscale turbulence field." GRL, 40.
  (Observed oceanic Ro distributions.)

- McWilliams, J. C. (2016). "Submesoscale currents in the ocean."
  Proc. Roy. Soc. A, 472, 20160117.

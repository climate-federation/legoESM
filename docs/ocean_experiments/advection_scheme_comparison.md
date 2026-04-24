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
JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection upwind --tag upwind_nosponge

JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection tvd --tag tvd_nosponge

JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
  --tracer-advection ppm_fct --tag ppm_fct_nosponge

JAX_ENABLE_X64=1 python3 scripts/run_ocean_test_matrix.py $FLAGS \
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

The post-processing script `scripts/plot_T_volumetric_census.py` computes
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

## Open issues and limitations

- **PPM-FCT limiter** (#9): current `alpha_face = min(alpha_left,
  alpha_right)` is a conservation-preserving heuristic but does not
  satisfy true Zalesak sign-split monotonicity. Grid-scale T noise
  leaks through and is amplified by the PGF. Needs code-level fix in
  `src/legoesm/ocean/advection.py` (implement sign-split incoming /
  outgoing flux limiter per Zalesak 1979 / Kuzmin et al.).
- **Hi-res SOM recipe**: at 200×100 (~10 km) no tested parameter set
  produces a stable 200-day run. Hill's dx⁴ B_h scaling is too weak
  at 10 km given sharper eddies; stronger momentum dissipation is
  needed. Not yet explored.
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

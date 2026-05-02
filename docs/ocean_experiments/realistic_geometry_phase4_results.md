# Phase 4 — Realistic-geometry GO spinup (lat-lon C-grid): first-pass results

Status as of 2026-05-01 on branch `realistic-geometry-full` (PR #224).

This document summarises the first end-to-end attempt at a Wolfe-Cessi-
style global overturning spinup on **real ETOPO bathymetry** with the
new partial-cells + SMC03 PGF + AL81 PV-flux stack landed earlier in
this PR.  The run is at `results/ocean/global_overturning_realistic_geometry/`
(years 0, 5, 10, 15, 20 restarts saved; killed at year 20.8 because we
had enough qualitative data).

## TL;DR

**Buoyancy thermodynamics works; wind-driven momentum is too noisy.**
Stratification develops the canonical thermohaline structure (warm
pool aloft, cold abyss, sloping isotherms, warm equator → cold poles
SST imprint).  But the velocity field is dominated by partial-cell-
induced grid-scale noise that prevents coherent gyres from forming
and inflates surface speeds 3-5× above realistic values.  AL81 saved
us from NaN, but did not eliminate the noise that AL81's stencil
imperfectly handles at coastal partial-cell vertices.

The realistic-geometry stack (AL81 + SMC03 + h_vtx min-rule + MEO
r=0.2 + production closures) is **stable** and produces qualitatively
correct thermodynamics.  It is **not yet production-quality** for
dynamics — the next iteration needs to address the residual partial-
cell q-stencil noise, likely via Sadourny-Salmon energy-enstrophy
conserving form (Arakawa-Lamb 1981 with full corner-triad stencil
that we approximated, or equivalently NEMO `dyn_vor_een` with all
the bottom-cell ENE corrections).

## Configuration

```python
GlobalOverturningConfig(
    use_gm_redi=True,
    bottom_drag_coeff=2.5e-3,
    A_h=1.0e4,                   # vs idealised default 2e5
    H_max=5000.0,
    dz_surface=20.0,             # vs idealised 10
    kappa_GM=800.0,
    kappa_Redi=800.0,
)
LatLonCGridOceanConfig(
    pgf_scheme="smc03",
    momentum_advection="vector_invariant",   # → AL81
    barotropic_solver="implicit_cn",
    A_h=1e4, B_h=5e9,
    bottom_drag_r=2.5e-3, bottom_drag_bbl_thickness=100,
    gm_redi=...
)
```

Bathymetry: ETOPO at 5° (36×72), Laplacian smoothing 5 passes,
H_min=50, **MEO r-factor cap = 0.2**.  This deepens minimum H to
800 m (eliminates continental shelves, keeps mid-ocean ridges) and
adds 6.3 % to ocean volume vs raw ETOPO.

Initial state: rest with centroid-aware exponential T(z) (T_surface=20,
T_deep=2, scale_depth=1000); uniform S=35; η=0.

Run length: targeted 50 yr at dt=600 s (5 min/yr wall on this MacBook
P-cores); killed at year 20.8.

## Diagnostic timeline

| Yr | max\|u\| (m/s) | mean SST (°C) | mean T | T_deep | max\|η\| (m) | MOC max (Sv) | BT max (Sv) |
|----|---------------:|--------------:|-------:|-------:|-------------:|-------------:|------------:|
| 0  | 0.000          | 19.83         | 8.90   | 2.04   | 0.000        |   0          |   0         |
| 5  | 2.180          | 14.66         | 7.67   | 2.17   | 1.528        | 601          | 398         |
| 10 | 4.365          | 14.27         | 7.55   | 2.28   | 1.701        | 661          | 379         |
| 15 | **5.237**      | 14.13         | 7.57   | 2.38   | 1.734        | 655          | 451         |
| 20 | 2.886          | 14.16         | 7.66   | 2.43   | 1.752        | 701          | 587         |

Five plots saved alongside the restarts:

- `timeseries_progress.png` — scalar timeseries
- `snapshots_progress.png` — SSH / SST / surface speed maps per restart
- `T_zonal_mean_progress.png` — zonal-mean T(lat, z) per restart
- `moc_progress.png` — meridional overturning ψ(lat, z) [Sv]
- `barotropic_streamfunction_progress.png` — ψ_bt(lat, lon) [Sv]

Re-runnable from the same script (`scripts/global_overturning/plot_realistic_geometry_progress.py`) as further restarts land.

## Qualitative assessment

| Feature | Status | Detail |
|---|---|---|
| Stratification (T zonal-mean) | ✅ | Classic thermohaline — warm pool upper 500 m, cold abyss <2000 m, sloping isotherms, polar deep mixing |
| SST pattern | ✅ | Warm equator (25 °C), cold poles (5 °C), hemispheric symmetry, N-Atlantic warmer than N-Pacific |
| SSH pattern | ✅ | Highs in subtropics, lows at poles; consistent with thermal wind |
| Mass conservation | ✅ | mean η ~ 0 throughout |
| Run stability | ✅ | No NaN through year 20.8; magnitudes bounded |
| Bottom T trend | ✅ | 2.04 → 2.43 °C (correct direction; slow deep warming via mixing) |
| **MOC** | ❌ | No deep AMOC-like cell; only thin shallow Ekman/STC structures concentrated at the equator. By yr 20 the deep field is essentially flat — too early for a real MOC, but the dynamics aren't pointing toward one. |
| **Gyres** | ❌ | Barotropic streamfunction dominated by coastal noise (Indonesia, equatorial Pacific hot spots).  No coherent subtropical/subpolar gyres. |
| **Surface speed magnitude** | ❌ | 3–5 m/s peaks vs realistic <1 m/s.  Boundary currents 30 Sv real, our BT streamfunction maxes at 400-590 Sv. |
| **Surface speed pattern** | ⚠️ | Pathological striping, especially equatorial Pacific.  Consistent with residual partial-cell q-stencil noise that AL81 only partially suppresses. |

The headline contradiction:

> The buoyancy field looks right.  The momentum field is dominated
> by grid-scale noise that AL81 prevented from blowing up but did
> not eliminate.

## What this tells us

Two independent diagnoses combine:

1. **Cold-start IC needs a proper spinup pipeline** (we already
   established this in Phase 6 ETOPO bisection).  Real production
   spinups use balanced Levitus/WOA initial T/S, forcing ramp,
   spinup viscosity, gradual transitions.  We don't have this
   machinery yet.
2. **The vector-invariant Coriolis stencil on partial cells still
   has residual q-noise** even after AL81.  AL81 prevents the day-19
   NaN that the simple 2-point Sadourny form caused; it does not
   eliminate the underlying grid-scale q variability at coastal
   partial-cell vertices.  At cold-start with imbalanced IC, this
   noise drives unphysically large flow that GM/Redi + biharmonic
   bound but cannot make smooth.

The 5-year-restart trajectory shows max|u| **growing then partly
relaxing** (peak 5.24 at yr 15, back to 2.89 at yr 20) — the system
*is* moving toward equilibrium, just slowly and through transient
adjustments that are noisier than they should be.  Whether 50, 100,
or 500 years would eventually clean up is unknown.

## What's solid going into the next session

The full SMC03 + AL81 + h_vtx + MEO + closure stack is **shippable
infrastructure**.  170 unit tests pass, BH stress test passes at
1.5 mm/s, ETOPO 30-day cold-start no longer NaNs.  These are real
deliverables.

What remains is **production-quality spinup machinery** + the
**residual partial-cell q-noise**, both genuinely Phase 4-5 work
that's separate from the dynamical-core foundations this PR landed.

## Recommended next-session priorities

In rough order of "smallest cost / biggest information gain":

1. **Sadourny-Salmon / Arakawa-Hsu energy-enstrophy form** (proper
   12-point AL81 with all the corner-triad ENE corrections, or the
   AH90 convex combination).  Our current `pv_flux_al81_partial_cell`
   uses the AL81 layout but the agent stalled mid-implementation; if
   any of the ENE corrections are missing, the residual q-noise we
   see is the explanation.  *Diagnosis:* compare our scheme's
   discrete energy + enstrophy budgets against a flat-bottom
   reference; check both are conserved to round-off.  If not, fix
   the missing terms.  ~1–2 days.

2. **Forcing ramp** — start with τ_wind = 0.0 and SST τ_T very long
   (say 1 year) for the first sim-year, ramp to production values
   over 6 months.  Production-typical for cold-start spinups.  ~half
   a day to add to the script + rerun.

3. **Equilibrated initial T/S** from climatology (Levitus / WOA).
   Our analytic exp(z) initial T is far from any equilibrium for
   real coastlines.  ~1 day to wire in Levitus reading from a
   downloaded NetCDF, ~half a day to verify the resulting initial
   state is sensible on partial cells.

4. **Spinup viscosity ramp** — high A_h (e.g. 5e5) for the first
   year, reduce to 1e4 once the gyres set up.  Production-typical.
   ~half a day to add.

5. **Run longer** (50 yr → 100 yr) with the above improvements and
   see if the MOC + gyres become coherent.

If after (1)-(5) the partial-cell noise is still the dominant
limiter on dynamics quality, then the deeper fix is the **PLM-in-
(T, S) PGF upgrade** flagged in `pgf_production_models_research.md`
— reconstruct (T, S) instead of ρ to push the in-cell-integral
residual from ``O(h²·ρ'')`` to ``O(h³·ρ''')``.  Multi-day work.

## File pointers

- Run script: `scripts/global_overturning/run_global_overturning_realistic_geometry.py`
- MEO sweep: `scripts/realistic_geometry_validation/plot_meo_bathymetry_sweep.py`
- Plotting: `scripts/global_overturning/plot_realistic_geometry_progress.py`
- Restarts (years 0, 5, 10, 15, 20): `results/ocean/global_overturning_realistic_geometry/restart_day*.npz`
- Plots:        `results/ocean/global_overturning_realistic_geometry/{timeseries,snapshots,T_zonal_mean,moc,barotropic_streamfunction}_progress.png`
- Run log:      `/tmp/go_rg_50yr_r02_v2.log` (year-by-year diagnostics, last printed yr 20.78)
- AL81 implementation: `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py:pv_flux_al81_partial_cell` (~376 LOC, Arakawa-Lamb 1981 12-point triad; possibly missing some corner-triad corrections — see point 1 above)
- Bisection narrative: `docs/ocean_experiments/partial_cells_results.md` (closure-sweep + AL81 fix sections)
- Production research: `docs/ocean_experiments/pgf_production_models_research.md`

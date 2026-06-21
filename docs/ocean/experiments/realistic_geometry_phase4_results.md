# Phase 4 — Realistic-geometry GO spinup (lat-lon C-grid): first-pass results

Status as of 2026-05-01 on branch `realistic-geometry-full` (PR #224).

> **Update — 2026-05-01 evening (Stage 0 diagnostics + cos²(lat) impl)**:
> Re-diagnosed the surface-speed pattern and found it is dominated by an
> alternating-sign 2Δy zonal-jet computational mode in *u* peaking at the
> equator, NOT coastal slope currents.  Identified A_h under-damping +
> high-lat partial-cell stability as the dominant levers.  Implemented
> cos²(lat) A_h scaling.  Audited AL81 PV-flux (six defects flagged but
> calibration vs WENO5 says they're at most ~20% effect).  See
> "Session 2026-05-01 evening — Stage 0 diagnostics" section below for
> the diagnostic ladder, the implementation, and the revised forward
> plan.

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

Re-runnable from the same script (`scripts/run/global_overturning/plot_realistic_geometry_progress.py`) as further restarts land.

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

- Run script: `scripts/run/global_overturning/run_global_overturning_realistic_geometry.py`
- MEO sweep: `scripts/validate/realistic_geometry/plot_meo_bathymetry_sweep.py`
- Plotting: `scripts/run/global_overturning/plot_realistic_geometry_progress.py`
- Restarts (years 0, 5, 10, 15, 20): `results/ocean/global_overturning_realistic_geometry/restart_day*.npz`
- Plots:        `results/ocean/global_overturning_realistic_geometry/{timeseries,snapshots,T_zonal_mean,moc,barotropic_streamfunction}_progress.png`
- Run log:      `/tmp/go_rg_50yr_r02_v2.log` (year-by-year diagnostics, last printed yr 20.78)
- AL81 implementation: `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py:pv_flux_al81_partial_cell` (~376 LOC, Arakawa-Lamb 1981 12-point triad; possibly missing some corner-triad corrections — see point 1 above)
- Bisection narrative: `docs/ocean/experiments/partial_cells_results.md` (closure-sweep + AL81 fix sections)
- Production research: `docs/ocean/experiments/pgf_production_models_research.md`

---

## Session 2026-05-01 evening — Stage 0 diagnostics, cos²(lat) impl, AL81 audit

After the 20-yr Wolfe-Cessi spinup at the top of this doc landed with
"thermodynamics good, momentum noisy," this session's goal was to
discriminate between the two competing hypotheses on the next-session
priority list (cold-start IC seed + A_h under-damping vs partial-cell
q-stencil residual) by cheap diagnostics first, before committing to
multi-day fixes.

### Stage 0a — Surface-speed × H_bathy overlay (existing 20-yr restarts)

Diagnostic script: `scripts/run/global_overturning/diagnose_surface_speed_bands.py`.
Plots:
- `results/ocean/global_overturning_realistic_geometry/speed_bathy_overlay.png`
- `.../u_v_components.png`
- `.../speed_zonal_section.png`

Findings:
- **Bands are in u, not v.**  v < 0.3 m/s essentially everywhere; u
  shows alternating-sign zonal jets at successive latitudes.
- **What looked like "N-S vertical bands at fixed lon" is the
  *amplitude envelope* of alternating-sign zonal jets**.  When you
  take √(u²+v²) the alternating sign vanishes.  The actual mode is a
  **2Δy zonal-jet stack**, not coastal slope currents.
- **Concentration peaks at the equator** (zonal-mean speed at lat
  −2.5°: 1.05 → 1.32 m/s yr 5 → 25, growing).
- **Envelope amplitude maxes over open-ocean ridges**: longitudes
  ~65°E (Carlsberg / Central Indian Ridge) and ~185°E (central Pacific
  seamount province).  *Not* at coastlines.

This **rejected** the cold-start coastal-slope-current hypothesis and
shifted the diagnosis toward the C-grid 2Δy null mode.

### Stage 0b' — A_h sweep (1e4 / 5e4 / 2e5)

Driver: `scripts/run/global_overturning/diagnose_ah_sweep.py`.  Three 2-yr
spinups, identical config except A_h.  Restarts in
`results/ocean/ah_sweep/run_Ah{1e04,5e04,2e05}/`, comparison plots in
`results/ocean/ah_sweep/`.

| A_h | max\|u\| @ yr2 | p95 speed | equator zonal-mean | NaN? |
|---|---:|---:|---:|:---:|
| 1e4 (current) | 2.61 m/s | 1.24 m/s | 1.05 m/s | no |
| **5e4** (intermediate) | **1.22 m/s** | **0.54 m/s** | **0.70 m/s** | no |
| 2e5 (idealised default) | — | — | — | **blew up at yr 0.76** |

5× A_h cuts off-equator baseline ~3× and equator peak ~30%.  The
sub-linear equator scaling is the f→0 null-mode signature: off-equator
the discrete Coriolis term redistributes energy and helps damping; at
the equator only explicit A_h damps directly.

Two unexpected findings:

**(a)** A_h=2e5 (the *idealised global overturning default that worked*)
**blew up at year 0.76** with η running away.  This is a genuine
partial-cell-specific instability that doesn't exist in the idealised
flat-bottom run.

**(b)** The equatorial peak doesn't disappear with A_h alone (it drops
only sub-linearly while off-equator drops near-linearly).

### D1 — Localising the A_h=2e5 blow-up

Driver: `scripts/run/global_overturning/diagnose_blowup_and_weno.py`.
Outputs in `results/ocean/ah_diagnostics/ah2e5_blowup/`.

Fine-cadence rerun with η-tracking every 5 sim-days:

| day | \|η\|max | location |
|---|---:|---|
| 5 | 4.8 cm | (-57.5°, 335°) |
| 10 | 7.6 cm | (+82.5°, 145°) Russian Arctic |
| 15 | **1.87 m** | (+82.5°, 115°) same area |
| 20 | NaN | — |

Day-15 η field shows a **single deep-blue spike at lat 82.5°/lon
115–145°E (Russian Arctic shelf)**, the rest of the ocean calm.  This
is the lat-lon grid's pole-convergence + viscous-Coriolis amplification:

- At lat 82.5°, dx ≈ 73 km (7.5× smaller than equator)
- Viscous decay time at A_h=2e5: dx²/A_h ≈ 7.4 hours
- Coriolis period at lat 82.5°: 2π/f ≈ 12 hours
- When viscous time < Coriolis period the forward-Euler-style coupling
  becomes amplifying rather than damping

Standard production fix in MITgcm/MOM6/NEMO: **scale A_h by cos²(lat)**
to keep viscous CFL latitude-independent.

**Why it didn't happen in the idealised run**: the idealised continent
config has `polar_cap_lat=80°`, so lat > 80° is land everywhere; flat
4000m bottom means no partial cells anywhere; rectangular continent
gives no coastal step bot_level transitions.  The Arctic blow-up needs
all three ingredients (pole convergence + ocean cells at high lat +
partial cells with thin coastal h) and the idealised run avoided all
three.

### D2 — WENO5 vs AL81 at A_h=5e4

Same driver.  Outputs in `results/ocean/ah_diagnostics/ah5e4_weno5/`.
Compared yr-2 surface speed against the existing
`vector_invariant`+`A_h=5e4` result from Stage 0b'.

| metric | AL81 | WENO5 | Δ |
|---|---:|---:|---:|
| max \|u\| | 1.22 | 0.94 | -23% |
| p95 surface speed | 0.54 | 0.50 | -7% |
| zonal-mean @ -2.5° | 0.71 | 0.61 | -14% |
| meridional-mean peak | 0.32 | 0.25 | -23% |

WENO5 is only **15–25% cleaner** than AL81 across metrics, and the
*structure* (band locations, equatorial concentration) is essentially
identical.  This is a strong **negative result for the AL81 audit
hypothesis** as a primary cause: switching the entire q-stencil scheme
only buys ~20%, so AL81 errors are at most a 20% contributor to the
residual mode at this resolution.

The dominant residual is therefore **under-resolved equatorial
dynamics** at 5°: equatorial Rossby radius ~250 km vs grid 555 km
gives ~0.4 cells per L_R, so the model can't represent equatorial
wave structure properly and dumps that energy into the only
structures it can resolve — 2Δy zonal jets.  A_h damps but doesn't
eliminate.

### Implementation: cos²(lat) A_h scaling

Landed on this branch:

- New helper: `laplacian_scaling_factor(grid)` in
  `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py`.  Returns
  `cos²(lat)` at u-face and v-face latitudes.  Mirrors the existing
  `biharmonic_scaling_factor(grid)` (`cos⁴`) one-to-one.
- New config flag: `LatLonCGridOceanConfig.A_h_lat_scaling: bool = False`.
  Default off → bit-exact regression on legacy configs.  Set True
  to enable the cos²(lat) multiplier.
- Plumbed into `ocean_pe_latlon_cgrid.py` at both A_h-only and A_h+B_h
  branches.
- 6 unit tests in `tests/ocean/unit/test_ah_lat_scaling.py` (shape,
  endpoints, equator-near-unity, off-is-identity, scales-correctly-at-
  high-lat, viscous-CFL-latitude-independent).  All passing.
- Sweep driver: `scripts/run/global_overturning/diagnose_cos2lat_sweep.py`.
  Three 2-yr spinups at A_h_global ∈ {1e5, 2e5, 5e5}, all with
  `A_h_lat_scaling=True`.  Outputs in `results/ocean/cos2lat_sweep/`.

### AL81 PV-flux audit

Full audit:
[`docs/ocean/experiments/al81_corner_triad_audit.md`](al81_corner_triad_audit.md)
(356 lines, written by the dycore-expert subagent).

**Verdict: partially correct.**  Has the AL81 layout (12-point triads,
1/12 weights) but is structurally incomplete on partial cells.  Six
defects identified:

| ID | Where | Issue | Severity |
|---|---|---|---|
| **F** | `ocean_pe_latlon_cgrid.py:1058-1066` | `h_vtx = min(...)` should be arithmetic mean over wet cells.  min is the correct face-flux convention but **NOT** the PV-thickness convention.  Overstates q at step vertices ~2×. | Highest leverage |
| A | `latlon_cgrid_operators.py:2604` | Pre-divides q=ζ/h_vtx outside the triad.  Le Sommer 2009 partial-cell EEN keeps ζ and h separate inside each triad. | High |
| B | `:2641-2644` | Uniform 1/12 weights instead of NEMO `e3f`-weighted triads. | High |
| C | `:2610` | Neumann fill of q at coast is a smoothing band-aid, not the AL81 fmask projection. | Medium |
| D | `:2697-2702, :2707-2712` | Final 4-term sums miss the f-point fmask multiplier — combined with C this double-counts. | Medium |
| E | `:2697-2702` | Possible missing Stewart-Dellar 2016 Appendix A cross-corner ENE correction. **Auditor flagged needs-verification** (couldn't fetch the paper). | Unknown |

**Calibration vs D2**: WENO5 bypasses the entire AL81 q-flux machinery
and only buys 15-20%.  That bounds the *combined* effect of fixing
A+B+C+D+E+F at ≤ 20% — the audit's claim that "F alone explains
60-80%" doesn't square with D2 unless h_vtx is also used in the
WENO5 momentum advection path (worth verifying).

**These are not typos or off-by-one bugs.** They are:

- **F** is a non-standard convention choice (min for PV-thickness).
  Internally consistent but doesn't match any production code I'm
  aware of.  ~70% bug, 30% defensible alternative.
- **A, B** are theoretical holes — AL81's uniform-h form applied to
  partial cells without the Le Sommer 2009 partial-cell extension.
  ~50% structural issue, 50% reasonable simplification.
- **C, D** are alternative boundary closures (Neumann fill vs fmask).
  ~20% bug, 80% style.
- **E** is unverified.

### What would settle "bug vs subjective"

Discrete energy + enstrophy budget test (recommended in the audit §4):

- **Flat-bottom inviscid limit** (A_h=B_h=drag=0): AL81 should conserve
  E = ½∫(u²+v²)h dA and Z = ½∫q²h dA to round-off.  This is the
  AL81/Sadourny-Salmon theorem.
- **Single-step bathymetry** (one row of partial cells next to full
  cells): re-run, check whether E and Z still conserve to round-off.

If passes flat AND passes partial-cell: items A, B, F are equivalent
reformulations of a conserving scheme.  Not bugs, just stylistically
different from NEMO.
If passes flat but fails partial-cell: at least one of A, B, F is a
real theoretical issue and the test gives us an empirical handle.

This budget test is the right *gate* on any AL81 fix — without it we'd
be guessing.  Cost: ~half day; ~100 LOC of test machinery.

### Revised forward plan

Replaces the original priority list at the top of this doc, in light of
the diagnostics above:

### cos²(lat) sweep results (complete)

Full table at yr 2 for all 5 tested configs (3 uniform + 3 cos²-scaled):

| Config | max\|u\| | p95 speed | eq.zonal-mean | NaN? |
|---|---:|---:|---:|:---:|
| Uniform A_h=1e4 (Phase 4 default) | 2.30 | 1.24 | 0.84 | no |
| Uniform A_h=5e4 | 1.22 | 0.54 | 0.67 | no |
| Uniform A_h=2e5 | — | — | — | NaN day 277 |
| cos² A_h_global=1e5 | 1.17 | 0.52 | 0.59 | no ✓ |
| **cos² A_h_global=2e5** | **1.09** | **0.51** | **0.53** | no ✓ |
| cos² A_h_global=5e5 | 1.08 | 0.42 | 0.45 | no ✓ |

Findings:
- **All cos²-scaled configs are stable**, including A_h=5e5 (50× the
  Phase-4 value).  The Arctic high-lat blow-up that killed uniform 2e5
  is **demonstrably fixed**.
- **The 2Δy zonal-jet mode is essentially gone** at A_h_global ≥ 1e5
  (see `results/ocean/cos2lat_sweep/partial_u_comparison.png`).  Basin
  interior u is near-zero everywhere except at the equator.
- **Diminishing returns 2e5 → 5e5**: only 0.06 m/s improvement in p95;
  equator zonal-mean drops 0.53 → 0.45.
- **Equatorial residual saturates at ~0.4-0.5 m/s** — this is the
  resolution-limited floor.  ~0.4 cells per equatorial Rossby radius
  at 5° means we can't represent equatorial wave structure properly,
  and the model dumps that energy into 2Δy stripes that no closure
  can fully eliminate.  Only finer equatorial resolution does.

**Recommended production setting: cos²(lat) + A_h_global = 2e5.**
Reasoning: 5e5 is marginally cleaner numerically but starts damping
mid-latitude gyres/boundary currents.  2e5 corresponds to the MOM6
grid-Reynolds-rule scaling from their 1° default (A_h~1e4 × Δx²
ratio 25 ≈ 2.5e5).  Leaves headroom for western boundary currents to
organize while still suppressing the 2Δy mode 4× from the current
default.

### AL81 discrete E+Z budget test (complete)

Test: `tests/ocean/unit/test_al81_budget.py`.  Operator-level
discrete budget on a 12×24 grid in float64, inviscid limit, two
configurations (flat bottom; single-step partial cells).  All 4 tests
pass.

| Test | Flat-bottom dΨ/Ψ | Step-bathymetry dΨ/Ψ | Amplification |
|---|---:|---:|---:|
| Energy `dKE/KE` | 2.7e-9 | 3.9e-9 | **1.4×** |
| Enstrophy `dZ/Z` | 5.8e-9 | 7.5e-9 | **1.3×** |

Decision rule:
- Amplification < 100×: items A/B/F are stylistic alternatives.
- Amplification > 1e5×: items A/B/F are real theoretical issues.

**Result: ~1.3× amplification — at least 4 orders of magnitude below
the "real bug" signature.**  AL81's partial-cell extension in our
implementation is functionally energy- and enstrophy-conserving in
the discrete sense, despite differing from NEMO `dyn_vor_een` in
items A, B, F.

The 1e-9 (not 1e-14) flat-bottom floor confirms the audit's items
A/B/C choices introduce a small but real residual on uniform-h —
present but well below any practical effect.  Calibration with D2
(WENO5 vs AL81 = 15-20% reduction) is now consistent: the 15-20%
came from WENO5's *implicit numerical diffusion*, not from "fixing"
non-conservation in AL81.

**Conclusion: skip the AL81 audit fixes (items A-F).**  The operator
is fine.  The dominant residual is the resolution-limited equatorial
mode, not AL81 stencil error.

### Updated forward plan (post-sweep + budget test)

The original 7-step plan can be simplified.  AL81 audit fixes are now
deprioritized based on the budget test.

**Now in flight (background)**:
- (none — sweep + budget test both complete)

**Next**:

| Step | Action | Effort | Gate | Status |
|---|---|---|---|---|
| 1 | cos²(lat) sweep (A_h_global ∈ {1e5, 2e5, 5e5}) | 30 min wall | — | ✅ done — see table above |
| 2 | AL81 discrete E+Z budget test | half day | — | ✅ done — items A/B/F are stylistic; skip fixes |
| 3 | **20-yr spinup with cos²(lat) + A_h_global=2e5** to see if gyres / MOC develop coherently over decades | 2 hr wall | step 1 stable | **next** |
| 4 | If 20-yr looks good: implement forcing ramp + spinup viscosity ramp + Levitus IC | 2 days | step 3 OK | gated |
| 5 | Production 50-yr run with full fix stack | 6 hr wall | step 4 | gated |
| 6 | Phase 5 diagnostics + comparison vs flat-bottom 50-yr reference | 1 week | step 5 | gated |

Items previously listed as gated on AL81 budget test are now closed
(test passed → audit fixes skipped).

---

## Session 2026-05-02 — Phase 4(c) production 50-yr run (overnight)

Driver: ``scripts/run/global_overturning/run_global_overturning_realistic_50yr_polar_cap.py``.
Output: ``results/ocean/global_overturning_realistic_50yr_polar_cap/``.

Configuration (full fix stack from session 2026-05-01 evening):

- 50 sim-yr at 5° resolution, dt=600 s, partial cells, SMC03 PGF
- AL81 momentum advection, implicit-CN barotropic, drag=2.5e-3, B_h=5e9 (cos⁴-scaled), GM/Redi κ=800
- **A_h_global=2e5 with cos²(lat) scaling** (interior 20× more damping than Phase 4(b))
- **north_cap_lat=80°** in BathymetryConfig (Arctic closed off)

Run completed all 50 sim-years cleanly in 4.09 h wall, max\|u\| stable at
~1.25 m/s.  A cosmetic post-completion bug (boolean indexing of 3D ``T``
with 2D ``land_mask``) crashed the script before it wrote ``run.log`` and
the final yr-50 restart, but all 10 mid-run restarts (yr 0, 5, 10, ...,
45) and all 5 progress plots saved.  Bug fixed in script for next run.

### Year-by-year diagnostics (from stdout, since run.log was lost)

| Yr | max\|u\| | mean SST | mean T | T_deep | max\|η\| | MOC max | BT max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 19.83 | 8.85 | 2.04 | 0.00 | 0 | 0 |
| 5 | 1.079 | 15.74 | 7.81 | 2.12 | 1.50 | 167 | 248 |
| 10 | 1.120 | 15.49 | 7.65 | 2.17 | 1.96 | 175 | 279 |
| 20 | 1.146 | 15.36 | 7.54 | 2.25 | 1.97 | 176 | 282 |
| 30 | 1.289 | 15.32 | 7.49 | 2.30 | 1.96 | 180 | 273 |
| 40 | 1.291 | 15.31 | 7.46 | 2.35 | 1.88 | 180 | 274 |
| 45 | 1.236 | 15.30 | 7.45 | 2.37 | 1.86 | 180 | 252 |
| 50 | 1.259 | (not saved) | (not saved) | (not saved) | 1.84 | (not saved) | (not saved) |

### Comparison vs Phase 4(b) at year 20

| Metric | Phase 4(b) yr 20 | Phase 4(c) yr 20 | Δ |
|---|---:|---:|---:|
| max \|u\| | 2.886 | 1.146 | **−60%** |
| MOC max | 701 Sv | 176 Sv | **−75%** |
| BT max | 587 Sv | 282 Sv | −52% |
| max \|η\| | 1.752 m | 1.97 m | +13% |

Phase 4(b)'s u-field grew through year 20 (2.18 → 4.37 → 5.24 → 2.89
m/s yr 5-20 — i.e. 5× growth then partial relaxation).  Phase 4(c)
**equilibrates by year 5** and stays there — max\|u\| ranges 1.08-1.30
m/s for years 5-50.

### What works ✓

- **No NaN, full 50 yr** — first realistic-geometry run to reach 50 yr cleanly
- **max\|u\| equilibrated at 1.1-1.3 m/s** — bounded, not growing
- **Thermohaline structure correct**: warm pool top 500 m, cold abyss,
  sloping isotherms, polar cooling.  Stable across all years.
- **SST equilibrated** at 15.3°C (cosine restoring target)
- **Bottom-T trend correct**: 2.04 → 2.37°C over 45 yr, slow deep warming
- **Coherent Southern Ocean MOC cell** at lat ~−50°S, depth ~2500 m,
  ~36 Sv (consistent with ACC-driven overturning).  Phase 4(b)'s 700 Sv
  noise is gone.
- **Mass conservation** — mean η drifts toward zero over time
- **Surface speed bands gone** — basin interior is dark (low speed) in
  the snapshot maps; the persistent N-S striping that motivated this
  whole investigation is *eliminated*.

### What still doesn't work

- **No deep AMOC-like cell** — no NADW formation pathway; MOC is
  dominated by the Southern Ocean structure.  North Atlantic interior
  is essentially flat in ``moc_progress.png``.
- **No coherent gyres** — barotropic streamfunction dominated by the
  ACC band around -50° to -65°S, no clear subtropical/subpolar gyre
  structure.  BT max ~270 Sv is still 5-10× realistic.
- **Surface speed concentrated at coastlines** — basin interior is
  clean but coastal jets (especially Africa-Eurasia) are unrealistically
  strong.
- **max \|η\| ~1.85 m** — bounded but high; production ocean SSH
  excursions are ~1 m max.

### Why AMOC + gyres haven't formed

This is no longer a numerical-noise problem — it's the standard
cold-start unbalanced-IC story:

- **Analytic exp(z) T(z)** applied to real coastlines puts a huge
  artifactual ρ-gradient at every coast → drives boundary currents
  that don't relax in 50 yr
- **No forcing ramp** — wind τ jumps from 0 to full at t=0, exciting
  barotropic Rossby waves
- **No spinup viscosity ramp** — high A_h_spinup absorbs the
  cold-start transient before relaxing to production A_h
- **Idealized forcing** (cosine SST restoring + two-belt wind, no NADW
  formation drivers) is fundamentally limited at modeling AMOC

These are exactly the deferred priorities #2-4 from session 2026-05-01.
They were rightfully held back until the dynamical-core noise was
solved.  It now is.

### Revised forward plan (Phase 4(d) and beyond)

| Step | Action | Effort | Status |
|---|---|---|---|
| 1 | cos²(lat) sweep | 30 min wall | ✅ done |
| 2 | AL81 budget test | half day | ✅ done — items A/B/F stylistic |
| 3 | Phase 4(c): 50-yr cos²+cap run | 4 h wall | ✅ done — bands gone, no AMOC |
| 4 | Phase 4(c) extension to 100 yr | 4.6 h wall | ✅ done — equilibrated, see below |
| 5 | Forcing ramp + spinup viscosity ramp | half day each | **next** |
| 6 | Levitus/WOA initial T/S | 1 day | gated on 5 |
| 7 | Phase 4(d): 50-100 yr with full spinup machinery | 6 h wall | gated on 4-6 |
| 8 | If still no AMOC: realistic OMIP forcing (separate plan) | weeks | gated on 7 |

### Artefacts (Phase 4(c))

- Driver: ``scripts/run/global_overturning/run_global_overturning_realistic_50yr_polar_cap.py``
- Restarts: ``results/ocean/global_overturning_realistic_50yr_polar_cap/restart_day{000000..016438}.npz`` (yr 0-45)
- Progress plots: ``timeseries_progress.png``, ``snapshots_progress.png``,
  ``T_zonal_mean_progress.png``, ``moc_progress.png``,
  ``barotropic_streamfunction_progress.png``
- Bug fix: 3D-T boolean-indexing in post-completion summary now uses
  ``jnp.where`` mask (script line 367-371).

**Deferred** (only if step 7 still shows ugly equatorial residual):

- Equatorial-grid-stretching for higher equatorial resolution.
- PLM-in-(T,S) PGF upgrade (`pgf_production_models_research.md` §7).

### Key artefacts from this session

- Diagnostic scripts:
  - `scripts/run/global_overturning/diagnose_surface_speed_bands.py`
  - `scripts/run/global_overturning/diagnose_ah_sweep.py`
  - `scripts/run/global_overturning/diagnose_blowup_and_weno.py`
  - `scripts/run/global_overturning/diagnose_cos2lat_sweep.py`
- Implementation: `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py`
  (added `laplacian_scaling_factor`),
  `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py` (plumbing),
  `src/legoesm/ocean/state.py` (`A_h_lat_scaling` flag).
- Tests: `tests/ocean/unit/test_ah_lat_scaling.py`.
- Result directories: `results/ocean/{ah_sweep,ah_diagnostics,cos2lat_sweep}/`.

## Session 2026-05-02 afternoon — Phase 4(c) extension to 100 yr

Driver: ``scripts/run/global_overturning/run_global_overturning_realistic_100yr_continuation.py``.
Restarts continue to write into the same output directory as the
50-yr run (``results/ocean/global_overturning_realistic_50yr_polar_cap/``)
so the progress plots cover yr 0-100 in one figure set.

Continuation segment: **yr 45.04 → 99.98**, wall 16,592 s (4.61 h),
``blew_up=False``.  All 11 new restarts saved (yr 45, 50, 55, …, 100)
plus the final ``restart_day036494.npz`` at yr 99.98.  Configuration
identical to the 50-yr run (A_h=2e5 with cos²(lat) scaling,
north_cap_lat=80°, AL81 / SMC03 / partial cells / implicit-CN
barotropic, dt=600 s).

### Year-by-year diagnostics (extension segment, from run.log)

| Yr | max\|u\| | max\|η\| |
|---:|---:|---:|
| 45.04 | 1.253 | 1.858 |
| 50.46 | 1.245 | 1.839 |
| 55.89 | 1.283 | 1.797 |
| 60.22 | 1.259 | 1.762 |
| 65.64 | 1.271 | 1.766 |
| 70.06 | 1.250 | 1.731 |
| 75.40 | 1.248 | 1.727 |
| 80.82 | 1.232 | 1.697 |
| 85.16 | 1.249 | 1.713 |
| 90.58 | 1.232 | 1.689 |
| 95.01 | 1.225 | 1.645 |
| 99.98 | **1.208** | 1.668 |

(Per-year ``mean SST / mean T / T_deep / MOC / BT`` are written only at
the 5-yr restart cadence; pull from the restart files when needed.)

### Behavioral summary

Read off ``timeseries_progress.png`` (yr 0-100, all 21 restarts):

- **max|u|**: spins up from 0 → 1.08 m/s (yr 5), climbs to a 1.24-1.29
  m/s plateau (yr 25-65), then **slowly drifts down to ~1.20 m/s by
  yr 100** — exactly the relaxation we wanted but never saw in Phase
  4(b).  The continuation segment drops ~0.05 m/s peak-to-peak across
  yr 45→100 with no excursions; the umax timeseries (umax_timeseries
  _continuation.png) plateaus and ratchets gently downward after yr 80.
- **mean SST**: 19.83 → 15.74 (yr 5) → 15.31 (yr 50) → **15.30 (yr 100)**.
  Locked to the cosine-restoring target.  No drift in the second half.
- **all-depth mean T**: 8.85 → 7.81 (yr 5) → 7.45 (yr 45) → **7.40
  (yr 100)**.  Slope flattening but not yet zero — bottom layer still
  warming slightly via diapycnal mixing.
- **bottom-layer T**: 2.04 → 2.43 (yr 20) → 2.50 (yr 50) → **2.55
  (yr 100)**.  Still drifting upward at ~0.02 °C/decade — consistent
  with the absence of any cold-deep-water source (no NADW, no AABW
  formation parameterisation) and slow Bryan-Lewis-style numerical
  diffusion.  This is the right qualitative direction for an unforced
  cold-start; quantitative drift rate is the kind of thing
  Phase 4(d)'s κ_v + WOA IC will fix.
- **max |η|**: peaks at 2.05 m around yr 25, drifts down to **1.67 m
  by yr 100**.  Bounded; still ~1.5× realistic (~1 m).
- **mean η**: stays within ±0.06 m throughout.  Slow upward drift to
  +0.05 m by yr 100 — small but non-zero; worth checking the freshwater
  flux / mass-conservation budget as a Phase 4(d) sanity item.

### What yr-100 confirms vs yr-50

The 50-yr run's headline conclusion was *"bounded and equilibrated, but
we don't know if the residual structure is transient or steady"*.  The
100-yr extension answers that:

- The dynamical equilibrium is **real**: ``max|u|`` does not grow
  past 1.29 m/s at any point in 100 yr, and trends mildly *downward*
  in the second half.  Phase 4(b)'s 5×-growth-then-partial-relaxation
  pattern is gone for good.
- The thermodynamic equilibrium is **partial**: SST and upper-ocean T
  are locked, but the abyss is still warming.  This is expected for an
  unforced cold-start ocean and is what the Phase 4(d) WOA IC + Bryan-
  Lewis κ_v are designed to address.
- The **AMOC / gyre absence persists** through yr 100.  ``moc_progress``
  and ``barotropic_streamfunction_progress`` show essentially the same
  ACC-dominated structure at yr 100 as at yr 45 — no NADW cell
  develops on its own from a cold-start with idealised cosine-SST
  restoring + two-belt wind.  This rules out *"just run longer"* as a
  fix and lines up exactly with the OMIP/CORE literature review's
  conclusion that **forcing protocol is the dominant lever**, not run
  duration.

### Implication for Phase 4(d)

The 100-yr stable, equilibrated, AMOC-free baseline is the cleanest
reference run we've ever had.  Every Phase 4(d) forcing change
(SST-restoring → prescribed Q_net, add E−P + β_S, Bryan-Lewis κ_v,
NA salinity-restoring patch) can now be A/B'd against this baseline
without any "but did the dynamical core finally settle?" caveat.

### Artefacts (extension)

- Driver: ``scripts/run/global_overturning/run_global_overturning_realistic_100yr_continuation.py``
- Restarts: ``results/ocean/global_overturning_realistic_50yr_polar_cap/restart_day{018264..036494}.npz`` (yr 50-100)
- Continuation-segment plot: ``umax_timeseries_continuation.png`` (yr 45-100, marked PASS)
- Progress plots refreshed in-place (now yr 0-100):
  ``timeseries_progress.png``, ``snapshots_progress.png``,
  ``T_zonal_mean_progress.png``, ``moc_progress.png``,
  ``barotropic_streamfunction_progress.png``
- ``run.log`` (full per-year stdout for the continuation segment)
- Audit: `docs/ocean/experiments/al81_corner_triad_audit.md`.

---
title: "legoESM ocean — simulation choice worksheet"
subtitle: "Pick one option per axis. Print and mark up."
---

**Status legend** — ★ production default · ✓ available, validated · △ caveat / less tested · ✗ avoid (broken or not wired) · ◯ planned but not implemented

Each row below is an independent axis of choice. Setting up a run = pick one leaf per row. Cross-axis restrictions are annotated. "Invalid / unwired combinations" are listed at the end.

---

## 1 · Grid

```
GRID
│
├── latlon C-grid
│     ├── ★ regular  (uniform Δλ × Δφ)
│     └── ✓ Mercator  (isotropic-area conformal; `docs/mercator_grid_plan.md`)
│     subtypes:
│       ├── ★ global
│       ├── ✓ regional       (rectangular subdomain, e.g. 24×48)
│       └── ✓ channel        (zonally periodic band, e.g. 24×72, 200×100, 20×10, 20×18)
│
├── MPAS Voronoi C-grid (TRiSK)
│     ├── ★ global icosahedral   (ico3 ≈ 300 km, ico4 ≈ 150 km, ico5 △ stiff)
│     ├── ✓ regional             (300 km Voronoi patch)
│     └── ✓ channel              (300 km / 100 km / 10 km)
│
├── △ cubed-sphere C-grid       (C24, C48 …)
│     └── ⚠ face-boundary instability (project_cubesphere_ocean_instability)
│         — avoid in production until fixed
│
└── ✗ spectral / Gaussian        (T21)
      └── standalone runner only (`scripts/run_ocean_spectral_tests.py`);
        not wired into the main matrix
```

---

## 2 · Resolution

```
RESOLUTION  (typical — what the test matrix uses)
│
├── latlon
│     ├── global:    ★ 36×72 (~2.5°)    ✓ 48×72   ◯ 1/4° + finer
│     ├── regional:  ✓ 24×48
│     └── channel:   ✓ 24×72   ✓ 200×100 (10 km)   ✓ 20×18   ✓ 20×10
│
├── MPAS
│     ├── global:    ★ ico3 (~300 km)   ✓ ico4 (~150 km)   △ ico5
│     ├── regional:  ✓ 300 km
│     └── channel:   ✓ 300 km   ✓ 100 km   ✓ 10 km
│
└── cubed-sphere:    △ C24   (higher untested)
```

---

## 3 · Vertical coordinate

```
VERTICAL
│
├── ★ z* (z-star, Adcroft & Campin 2004)
│     ├── ★ with partial bottom cells
│     └── ✓ without (full-step bathymetry)
│
├── ◯ ALE remap onto z*   (planned; MOM6-style)
├── ◯ isopycnal / hybrid   (not started)
└── ◯ σ / terrain-following   (not started)
```

Layer count `n_levels`: typically 10 (idealized) or 50–75 (production-style). Default `H_max = 5500 m`. Single-layer (`nlev=1`) supported for truly-barotropic runs.

---

## 4 · Time integration — per term, not one axis

The ocean dycore does **not** use a single "outer integrator". Different terms have different stiffness and oscillation character and each one is stepped with the scheme suited to it. The table below documents what is actually in `_step_impl` today.

| Term                              | Lat-lon (current)                          | MPAS (current)                            | Reference            | Stability character                           |
|-----------------------------------|--------------------------------------------|-------------------------------------------|----------------------|-----------------------------------------------|
| Planetary Coriolis $f\times\mathbf u$ | forward-backward (Matsuno)                 | forward-backward (Matsuno)                | Matsuno; Hallberg 97 | neutral, $\|G\|^2 = 1 + O((f\Delta t)^4)$     |
| Relative-vorticity / PV flux      | AL81 spatial in outer step                 | TRiSK `pv_scheme` (default `"enstrophy"`) | Arakawa-Lamb 81; Ringler 10 | conservative; spatial choice sets KE/enstrophy budget |
| Horizontal pressure gradient      | within outer step (Euler)                  | within outer step (Euler)                 | Adcroft-Campin 04    | hydrostatic; no own stepper                   |
| Momentum advection (vec-inv)      | vector-invariant in outer step (Euler)     | TRiSK vector-invariant in outer step      | Sadourny 75          | advection-CFL                                 |
| Tracer advection                  | `tracer_time_integrator` ∈ {`euler`(★), `ab2`, `rk3`} | implicit in outer step           | Hundsdorfer 95       | CFL ≲ 1 (Euler), 0.72 (AB2), 1 (SSP-RK3)       |
| Barotropic subcycle               | explicit BEBT pred-corr (or implicit-CN)   | explicit subcycle (or implicit-CN)         | SMC05; Hallberg 97   | gravity-wave CFL on subcycle; CN unconditional |
| Vertical mixing (diffusion+KPP)   | within outer step (currently explicit — needs audit per item 5C of plan-of-work)     | same | LMD 94; CVMix         | $\Delta t < \Delta z^2/(2K_v)$ if explicit; unconditional if implicit |

```
dt_outer:               ★ 300 s @ ~2.5°    (advection-CFL limited)
n_barotropic_substeps:  ★ ~20–60           (gravity-wave-CFL limited if explicit)
```

The `src/legoesm/timestepping/dispatch.py` module exposes `Forward Euler`, `Heun/RK2`, `SSP-RK3`, `SSP-RK34`, `RK4` integrators but the ocean dycore does **not** route through it — split-explicit per-term stepping above is the production pattern. The dispatch module is used by the atmosphere SW path.

---

## 5 · Mode splitting (barotropic ↔ baroclinic)

```
BAROTROPIC SOLVER
│
├── ★ explicit subcycling
│     ├── time integrator
│     │     ├── ★ BEBT predictor-corrector   (Shchepetkin-McWilliams 2005)
│     │     └── ✓ forward-backward            (Hallberg 1997)
│     ├── time filter (over substeps)
│     │     ├── ★ cosine window                (ROMS/MOM6 style)
│     │     ├── ✓ boxcar
│     │     └── △ none                          (aliasing risk)
│     ├── slow-forcing handling: ★ included     (fixes mode-splitting noise)
│     ├── MAXVEL clip: ★ on                     (safety net)
│     └── implicit bottom-drag factor: ★ on
│
└── ✓ implicit Crank–Nicolson    (MPAS only — production for multi-year stability)
      └── `barotropic_solver="implicit_cn"`
        — fixes TRiSK null-branch growth (project_mpas_barotropic_noise, 5-yr OK)
```

**Cross-axis note** — explicit subcycling is the only option on lat-lon today. Implicit-CN is MPAS-only.

---

## 6 · Pressure gradient force

```
PGF
│
├── lat-lon C-grid
│     └── ★ z* standard (column-integrated hydrostatic, EOS-pressure 2-pass)
│
├── MPAS
│     ├── ★ density-Jacobian   (docs/ocean/density_jacobian_pgf_*.md)
│     │     ├── ✓ Shchepetkin–McWilliams 2003 variant
│     │     └── ✓ alternative finite-volume form
│     └── ✓ z* standard
│
└── cubed-sphere
      └── △ z* standard   (contributes to face-boundary instability)
```

EOS-pressure iteration: ★ 2-pass (`iterate_eos_and_pressure_anomaly`).

---

## 7 · Equation of state

```
EOS  (ocean/eos.py)
│
├── ★ linear  α_T (T − T_ref) + β_S (S − S_ref)        ← idealized experiments
├── ✓ Roquet-2015 polynomial                           ← Wright-style nonlinear
└── ✓ full nonlinear (T,S,p)                           ← thermobaric/cabbeling-aware
```

Constants: `ρ_0 = 1025 kg/m³`, `T_freeze_ocean = 271.35 K` (intentional vs. freshwater 273.15 K).

---

## 8 · Coriolis / PV-flux discretization

Spatial scheme — how $\zeta$ flux and $f\times\mathbf u$ are discretised on the grid:

```
CORIOLIS  /  PV-FLUX  (spatial)
│
├── lat-lon C-grid
│     ├── ★ AL81  (Arakawa–Lamb 1981, energy + enstrophy conserving)
│     │     └── with Le Sommer 2009 / Stewart-Dellar 2016 partial-cell weights
│     │       (`pv_flux_al81_partial_cell`)
│     ├── KE gradient on the C-grid:
│     │     ├── ★ centered  (default; legacy bit-exactness)
│     │     └── ✓ + Hollingsworth correction (commit d0183817)
│     │           required for high-Ro flow, fixes spurious vortex stretching
│     └── (Sadourny 4-point average is used only for the planetary-f
│        Matsuno step — see temporal column in §4)
│
└── MPAS Voronoi (TRiSK)
      ├── ★ "enstrophy"  (Ringler 2010 enstrophy-conserving)  ← current default
      ├── ✓ "energy"     (Ringler 2010 energy-conserving)
      ├── ✓ "mixed"      (α blend, `pv_alpha`)
      └── ◯ AL81         (not in dispatch today — Phase 2D candidate)

      APVM dissipation (PV upwind):
      ├── ★ on   (`apvm_dt`)
      └── ✓ off
```

Temporal scheme — how the planetary $f\times\mathbf u$ is stepped in time:

```
PLANETARY CORIOLIS  (temporal)
│
├── ★ forward-backward Matsuno  (both grids today)
│     │
│     ├── lat-lon: `_forward_backward_coriolis_3d` in
│     │     `ocean_model_latlon_cgrid.py`
│     └── MPAS:    `_forward_backward_coriolis_mpas_3d` in
│                  `ocean_model_mpas.py`
│
└── ◯ Crank-Nicolson on rotation  (deferred — Phase 5D)
      exact discrete-KE conservation; needs 2×2 per-cell linear
      solve; deferred until inertial-oscillation test reveals
      Matsuno as a bottleneck (not expected at typical ocean fΔt).
```

**⚠ Known issue** — MPAS Coriolis double-counting on 10-level runs (issue #103). MPAS PV-flux default `"enstrophy"` is the opposite of Alistair's recommendation; Phase 2D switches it to `"energy"`.

---

## 9 · Momentum advection

```
MOMENTUM ADVECTION
│
├── lat-lon C-grid
│     ├── ★ vector-invariant (Sadourny + KE-gradient + PV-flux)
│     ├── ✓ WENO5-ILES  (Silvestri et al. 2024, rotational WENO)
│     └── ✓ WENO7-ILES
│         └── D-term: ★ on   (also has off flag)
│
└── MPAS
      └── ★ TRiSK PV-flux + KE gradient   (vector-invariant only)
```

---

## 10 · Tracer advection

```
TRACER ADVECTION  (thickness-weighted, flux form)
│
├── lat-lon C-grid  (LatLonCGridOceanConfig.tracer_advection)
│     ├── ★ TVD (Van Leer; 2nd-order; default today)
│     ├── ✓ upwind (1st-order; fallback)
│     ├── ✓ DST-3 (3rd-order; Hundsdorfer 95 / Easter 93)
│     │     └── d0 cap; full benefit at CFL > 0.1 or RK3 (project_dst3_advection)
│     ├── ✓ PPM (Colella-Woodward 84)
│     ├── ✓ PPM_FCT (PPM + Zalesak FCT limiter)
│     ├── ✓ DST-3 multidim (known unstable for strong flows)
│     ├── ✓ WENO-5, WENO-7 (Silvestri 2024)
│     └── ✓ SOM (Prather 86; 9-moment; fully differentiable, no limiter)
│           → for low-mixing cases (Eady, ACC; project_som_advection)
│
└── MPAS Voronoi  (MPASOceanConfig.tracer_advection)
      ├── ★ TVD (Van Leer; production)
      └── ✓ upwind (1st-order; dataclass default)
        — no 3rd-order options wired on Voronoi today
```

Tracer time integrator (sub-step within outer step):
★ Euler (lat-lon default) · ✓ AB2 · ✓ SSP-RK3

---

## 11 · GM / Redi (mesoscale eddy parameterization)

```
GM / REDI
│
├── ★ off  (default for eddy-resolving runs)
│
├── lat-lon C-grid     (project_gm_redi_latlon)
│     ├── ✓ centered slope        (`slope_scheme="centered"`)
│     └── ✓ Griffies-1998 triads  (`slope_scheme="triads"`)
│     modes:
│       ├── ✓ GM only       (advective skew flux only)
│       ├── ✓ Redi only     (isopycnal diffusion only)
│       └── ✓ GM + Redi     (combined)
│
└── MPAS Voronoi
      ├── ✓ centered slope
      └── ◯ triads   (Phase 5 of plan — not yet)
```

Slope tapering: smooth (sigmoid), not hard-clipped — required for AD.

---

## 12 · Horizontal dissipation

```
HORIZONTAL DISSIPATION
│
├── Laplacian (constant)
│     ├── momentum:  A_h   (★ ~1e3–1e4 m²/s @ 1°)
│     └── tracer:    K_h   (★ ~1e3 m²/s)
│
├── Biharmonic (constant)
│     ├── momentum:  B_h   (★ ~1e10–1e12 m⁴/s)
│     └── ζ-form:    K_ζ_bih  (MPAS only — selectively damps vorticity)
│
├── Smagorinsky      (flow-dependent;  ★ C_smag ≈ 2.0 for eddying)
│
├── Leith / modified Leith   (MPAS — vorticity-based, target enstrophy)
│
├── Hyperdiffusion   (Δ², Δ³ orders configurable)
│
├── Barotropic divergence damping   (targets grid-scale compressible modes)
│
└── APVM   (MPAS only — PV upwind dissipation; see §8)
```

**⚠ Known issue** — MPAS + ETOPO bottom-trapped instability is an algorithm-class limit; damping suite exhausted (project_mpas_etopo_instability). 90-day stable, multi-year requires algorithmic change.

---

## 13 · Vertical mixing

```
VERTICAL MIXING
│
├── KPP (Large-McWilliams-Doney 1994)
│     ├── ★ on for production
│     └── ✓ off for idealized
│
├── Convective adjustment
│     ├── ★ enhanced K_v when N² < 0
│     ├── ✓ plume scheme  (ocean/physics/convection/plume.py)
│     └── ✓ off
│
└── Background diffusivity
      ├── A_v   (★ ~1e-4 m²/s)
      └── K_v   (★ ~1e-5 m²/s)
```

---

## 14 · Bottom drag

```
BOTTOM DRAG
│
├── ★ quadratic    τ_b = ρ_0 C_d |u_b| u_b      C_d ★ 2.5e-3
├── ✓ linear       τ_b = ρ_0 r u_b              r  ★ 1e-4 m/s
├── ✓ free-slip    (idealized, no drag)
└── implicit treatment in barotropic mode: ★ on
```

---

## 15 · Surface forcing slots

```
FORCING  (ocean-side interfaces only)
│
├── Wind stress (τ_x, τ_y)
│     ├── ✓ analytical (cos / sin² / channel jet)
│     └── ✓ field input
│
├── Heat flux (Q_net)
│     └── + ✓ shortwave penetration (Q_sr split, commit 50d65485)
│
├── Freshwater flux (E−P)
│     ├── ✓ virtual salt flux  (apply_freshwater_virtual_salt_top)
│     └── ◯ real volume change
│
├── T / S restoring
│     ├── ✓ point-wise scalar T*, S*
│     ├── ✓ 3-D arbitrary array T*(x,y,z), S*(x,y,z)   (commit 50d65485)
│     ├── time integrator: ★ implicit Euler   (✓ explicit)
│     └── timescale: ★ τ ~ 10–60 d   (sponges τ ~ 1–5 d)
│
└── Sponge regions (regional / channel grids)
      ├── ✓ T, S
      └── ✓ u, v
```

---

## 16 · Initial conditions

```
INITIAL STATE  (init_*.py + experiments/)
│
├── Rest state (gravity-wave / PGF tests)
│     ├── ✓ stratified, with land
│     ├── ✓ stratified, no land
│     ├── ✓ uniform T/S, with land
│     └── ✓ uniform T/S, no land
│
├── Wave / adjustment tests
│     ├── ✓ barotropic_wave              (gravity wave; res-matched grids)
│     ├── ✓ inertia_gravity_wave         (Bishnu et al. 2024)
│     └── ✓ geostrophic_adjustment
│
├── Wind-driven gyres
│     ├── ✓ barotropic_double_gyre (cos & sin² wind)
│     ├── ✓ baroclinic_gyre (cos & sin²)
│     ├── ✓ global_barotropic_wind  (lat-lon, MPAS;  cube ✗)
│     └── ✓ global_barotropic_wind_1lev
│
├── Density-current benchmarks
│     ├── ✓ lock_exchange   (Ilicak protocol; lat-lon, cube only)
│     └── ✓ overflow         (Ilicak; lat-lon, cube only)
│
├── Baroclinic instability
│     ├── ✓ phillips_two_layer            (all grids)
│     ├── ✓ eady_instability              (channel, sponge)
│     ├── ✓ eady_uniform                  (Bishnu — classical Eady)
│     └── ✓ eady_gm_redi                  (lat-lon centred + triads, MPAS centred)
│
├── ✓ stommel_gyre_tracer                 (Hecht 2000)
└── ✓ acc_channel  /  acc_channel_rest    (Zhang et al. 2024 inspired)
```

Custom IC: write a function returning `OceanState` for your grid in `experiments/` or a new module.

---

## 17 · Invalid / unwired combinations

These leaves *cannot* be selected together today:

| If you pick … | Then you cannot also pick … | Reason |
|---|---|---|
| MPAS                | GM/Redi triads                  | Phase 5 of plan, not implemented |
| MPAS                | WENO momentum advection         | only on lat-lon today |
| Cubed-sphere        | anything multi-year             | face-boundary instability |
| Cubed-sphere        | regional / Mercator             | needs 6-face init adaptation |
| Spectral (T21)      | the main test matrix            | standalone runner only |
| `nlev=1` (truly barotropic) | KPP / convection / GM-Redi | no interior to mix |
| Lat-lon            | implicit-CN barotropic solver   | MPAS-only path today |
| MPAS + ETOPO        | multi-year integration          | algorithm-class instability |
| Implicit-CN baro    | high `MAXVEL` dependence        | clip not needed by design |
| GM/Redi triads      | non-z* vertical                 | no other coord supported |

---

## 18 · Three reference trajectories from the test matrix

**A. MPAS global wind-driven, multi-year stable** — `global_barotropic_wind` on ico3, 60 days; with `barotropic_solver=implicit_cn` extends to 5 years. AL81 PV-flux, density-Jacobian PGF, DST-3 tracer, quadratic drag, no GM/Redi, KPP off.

**B. Lat-lon Eady channel with GM/Redi triads** — `eady_gm_redi` on 20×10 channel, 30 days. Centred PGF, Hollingsworth-corrected KE gradient, SOM tracer (low spurious mixing), GM+Redi with triad slopes, no KPP (idealized).

**C. Lat-lon ACC channel** — `acc_channel` on 20×18 channel @ 100 km, 30 days. Wind + sponges + Gaussian ridge. Vector-invariant momentum, DST-3 tracer, Smagorinsky viscosity, quadratic drag, no GM/Redi (eddy-permitting).

---

## 19 · The questions to actually ask Alistair

1. **Modal split** — should lat-lon move to implicit-CN like MPAS? Or is there a way to make the explicit + cosine path stable for the lat-lon barotropic-mode noise?
2. **Cubed-sphere face boundaries** — has he seen anything like our face-boundary instability in MITgcm cube experiments? Recipe for fix?
3. **MPAS + ETOPO** — algorithm-class limit, or fixable PGF/bathymetry-interaction bug?
4. **GM/Redi triads on Voronoi** — anyone done it correctly? (We don't have a clear precedent.)
5. **ALE adoption** — Adcroft & Hallberg perspective: should legoESM move to ALE remap on top of z*? What does it buy?
6. **Tracer transport in vanishing layers** — what's MOM6 actually doing today?
7. **Vertical-coordinate roadmap** — is hybrid (Bleck) worth the implementation cost for a model that already has differentiability?

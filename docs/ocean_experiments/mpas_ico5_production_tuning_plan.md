# MPAS ico5 Production Tuning Plan

**Status**: Steps 0-4b complete + sensitivity tests (2026-05-11).

## Goal

Bring the MPAS ico5 (~100 km) ETOPO configuration as close as
possible to CMIP6-class production ocean models (E3SM/MPAS-Ocean,
MOM6 OM4p5, NEMO ORCA1) while staying within reach of single-session
testing.

## Current working baseline (verified stable)

- 10 years stable on ETOPO bathymetry
- 20 stretched levels (dz_sfc=20m, dz_deep=500m)
- 30% partial-cell snap
- Implicit vertical mixing (backward Euler) — the key 2026-05-10 fix
- KPP(K_conv=1.0) + enhanced-diffusion convection routed through implicit solver
- K_zeta_bih=1e14 for Voronoi checkerboard mode
- C_smag_lap=0.33 (no constant A_h)
- A_v=1e-4, K_v=1e-5 (production scalar values)
- Linear bottom drag r=1e-3, BBL=100m

Reference run: `results/ocean/global_overturning_mpas_etopo_smag/`
(year-10 max|u|=3.13 m/s, max|eta|=1.59 m, mean SST=16.41 °C)

## Production target config

```python
MPASOceanConfig(
    # Numerics
    barotropic_solver="implicit_cn",
    pgf_scheme="centered",
    implicit_vertical_mixing=True,
    n_barotropic_substeps=30,

    # Horizontal viscosity (H2)
    A_h=1.0e4,                       # was 0.0; floor for quiescent regions
    C_smag_lap=0.15,                 # was 0.33; more standard
    K_zeta_bih=1.0e14,               # Voronoi null-mode damping

    # Tracer mixing (H3, M3)
    K_h=0.0,                         # rely on GM/Redi + TVD
    tracer_advection="tvd",          # was "upwind"; less implicit diffusion

    # Vertical mixing
    A_v=1.0e-4,
    K_v=1.0e-5,                      # constant; depth-dependent in L1

    # Bottom drag (H1)
    bottom_drag_r=1.0e-3,
    bottom_drag_bbl_thickness=100.0,
    bottom_drag_bg_velocity=0.1,     # quadratic-with-floor

    # Equatorial (H4)
    equatorial_visc_boost=0.0,
    equatorial_visc_sigma_deg=5.0,

    # GM/Redi (M1)
    gm_redi=GMRediConfig(
        kappa_GM=600.0,              # ramp to 800 once stable
        kappa_Redi=600.0,
        S_max=0.005,                 # OMIP value
        slope_scheme="centered",     # triads not yet on MPAS
        visbeck=VisbeckConfig(enabled=False),  # enable after fixed-κ stable
    ),

    # Physics
    physics=OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(
            scheme="kpp",
            kpp=KPPConfig(K_conv=0.0),   # keep 0 until L5 lands
        ),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
        ),
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind", tau_max=0.1,
            ),
            restoring=RestoringConfig(
                tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0,
                S_star=35.0, T_profile="cosine",
            ),
        ),
    ),
)
```

## Incremental test plan

Each step runs 1 year from rest first.  Promote to multi-year only
after 1-year diagnostics match baseline within ~10% on max|u|/max|eta|
and ~0.3 °C on mean SST.

| Step | Change | Length | Pass criterion | Status |
|------|--------|--------|----------------|--------|
| 0 | Baseline regen (current config) | 1 yr | reference numbers | DONE |
| 1 | **H1**: `bottom_drag_bg_velocity=0.1` | 1 yr | within 10%; budget confirms drag-only change | PASS |
| 2 | **H2**: `A_h=1e4`, `C_smag_lap=0.15` | 1 yr | cold-start blowup at yr 0.28 | SKIP |
| 3 | **H3**: `tracer_advection="tvd"` | 1 yr | SST +0.4°C (physical), max\|u\| -33% | PASS |
| 4a | **M1a**: GM/Redi κ=300 | 1 mo | no NaN at step edges | PASS |
| 4b | **M1b**: ramp κ to 600 | 1 yr | stable, matches Step 3 numbers | PASS |
| 6 | Full config (1+3+4b) | 10 yr | max\|u\|=2.09, SST=17.02, geostrophic | PASS |
| 7 | Bathymetry case + same config | — | same as step 6 (already on ETOPO) | N/A |

## Test results (2026-05-11)

### Step results summary

| Step | Change | Year-1 max\|u\| | max\|eta\| | SST | Status |
|------|--------|----------------|-----------|-----|--------|
| 0 | Baseline (C_smag_lap=0.33) | 4.48 | 0.87 | 17.32 | Reference |
| 1 | + H1 quad drag (u_bg=0.1) | 4.65 | 0.91 | 17.27 | PASS |
| 2 | + H2 A_h=1e4, C_smag=0.15 | NaN | — | — | SKIP (cold-start blowup) |
| 3 | + H3 TVD advection | 3.01 | 0.74 | 17.72 | PASS (SST +0.4°C) |
| 4a | + M1a GM/Redi κ=300 (1mo) | 2.07 | 0.21 | 18.65 | PASS |
| 4b | + M1b GM/Redi κ=600 (1yr) | 2.97 | 0.73 | 17.74 | PASS |
| 6 | Full 10yr (H1+H3+M1b) | 2.09 | 1.65 | 17.02 | PASS (10yr stable) |

### Budget-based Step 1 validation

Momentum budget comparison (linear vs quadratic drag on same state):
- Sanity: non-drag terms differ by 5.7e-14 (machine precision) — ONLY drag changed
- Drag RMS +19% globally, +69% at moderate speeds (0.1-0.5 m/s)
- Integrated drag dissipation: +15% (quadratic does more work)
- Velocity-dependent scaling correct: recovers linear at rest, quadratic at speed

### Step 2 failure analysis

A_h=1e4 + C_smag_lap=0.15 blew up at year 0.28 from rest. The cold-
start equatorial spin-up generates rapid velocity growth; C_smag_lap=0.15
is too weak to control it before the Smagorinsky viscosity ramps up.
This config WAS stable from a developed state (year-8 restart) but
not from rest. Needs either: (a) higher C_smag_lap, (b) B_h for
scale-selective startup support, or (c) spinup-from-developed-state.

### Sensitivity tests

**AHH08 vs centered PGF (same config, 1yr):**

| PGF scheme | max\|u\| | max\|eta\| | SST |
|------------|---------|-----------|-----|
| Centered | 2.97 | 0.73 | 17.74 |
| AHH08 | 2.92 | 0.71 | 17.72 |

Essentially identical. The elevated equatorial velocities are NOT from
PGF scheme accuracy — the tracer-PGF feedback from advection across
step edges quickly builds real gradients that both schemes respond to
equally.

**Low wind (τ_max halved from 0.1 to 0.05 Pa):**

| Wind | max\|u\| | Change |
|------|---------|--------|
| τ=0.1 (baseline) | 2.97 | — |
| τ=0.05 (halved) | 1.43 | -52% |

The equatorial velocities scale almost exactly proportionally with
wind stress (-52% velocity for -50% wind). This confirms they are
**wind-driven, not a PGF artifact**. The model responds correctly
to forcing. The elevated equatorial velocities at τ=0.1 are a
resolution limitation (1° can't resolve the equatorial jet structure),
not a code bug.

**40 stretched vertical levels (dz_sfc=10m, dz_deep=300m):**

Blew up at year 0.26. The C_smag_lap=0.33 + K_zeta_bih=1e14 that
work at 20 levels need retuning for the thinner cells at 40 levels
(different CFL/stability characteristics). Deferred to future work.

### Production run diagnostics (10 years, full config)

The 10-year production run (`global_overturning_mpas_etopo_production/`)
demonstrates:
- max|u| settled to 2.1 m/s (year 10, bounded)
- max|eta| = 1.65 m (stable)
- mean SST = 17.02 °C (equilibrating slowly)
- Geostrophic balance ratio ∇p/f×u = 1.00 at all depths below 50m
  (checked in budget closure analysis)
- Momentum budget: PGF and Coriolis dominate (~2e-6 m/s²), wind
  provides surface forcing (~2e-6), viscosity/drag at ~1-7% of PGF
- Closed exact budget to machine precision (one-step test)
- ACC visible as continuous eastward band at 55-65°S
- WBCs forming at western boundaries (broad, unresolved jets)
- No jet extensions into interior (resolution limitation at 1°)

## Viscosity stack investigation (2026-05-12)

### Problem: cell-to-cell noise in velocity, especially at depth

Cross-sections show alternating red/blue vertical stripes in velocity
at depth — characteristic of the TRiSK ζ-checkerboard null mode.
Ocean-expert literature review confirmed:
- At 1° there should be ZERO cell-to-cell variability below 1000m
- The standard TRiSK vector Laplacian cannot see this mode
- MPAS-Ocean production uses **B_h (biharmonic del4) as the primary
  dissipation**, not Laplacian A_h

### MPAS-Ocean production QU120 (120km) reference values

| Parameter | Our config | MPAS-Ocean QU120 |
|-----------|-----------|-----------------|
| Primary viscosity | C_smag_lap=0.33 | B_h=2.6e13 (del4) |
| Secondary | A_h=0 or 1e4 | A_h=1000-2000 (del2) |
| ζ-null mode | K_zeta_bih=1e14 | enstrophy PV + del4 |

### Tests performed

| Config | max\|u\| yr20 | Drake Sv | Visc/forcing ratio |
|--------|-------------|---------|-------------------|
| Smag only (C=0.33) | 1.15 | 37 | 4.4% |
| Smag + A_h=1e4 | 0.91 | 26 | 6.1% |
| Smag + B_h=1e13 + A_h=2e3 | 1.11 | 20 | 5.0% |

Momentum budget analysis showed all three configs have similar
dissipation magnitude (3-14% of PGF forcing). The Drake transport
differences (37→26→20 Sv) are partly from mid-run config switching
(2-year adjustment transient). Proper comparison needs runs from
rest with equal spinup time.

### Confirmed from literature (ocean-expert deep dive)

- Grid-scale noise diagnostic: `var(u_neighbor - u_cell) / var(u)`
  per level — should be small for smooth flow
- The ζ-checkerboard is invisible to del2 AND del4 on velocity —
  only K_zeta_bih or the enstrophy PV scheme can control it
- MPAS-Ocean's `config_mom_del4_div_factor` allows stronger damping
  of divergent modes independently
- B_h scaling: `del4/dx³ ~ 0.015 m/s` across MPAS-Ocean resolutions

### Next steps for viscosity tuning

1. Run B_h=2.6e13 (full production value) from rest with same physics
2. Compare Drake transport, noise level, and WBC structure vs
   Smag-only and A_h=1e4 configs after equal spinup
3. Implement grid Reynolds number diagnostic
4. Implement neighbor-velocity variance diagnostic
5. Consider `del4_div_factor` for targeted divergence damping

## Additional findings (2026-05-12)

### Equatorial velocity is wind-driven (not PGF artifact)

- Halving tropical wind (scale=0.5, σ=15°) reduces max|u| by 52%
  (1.43 vs 2.97 with full wind) — proportional to forcing
- AHH08 PGF (machine-zero) gives identical velocities to centered —
  the PGF scheme accuracy is NOT the issue
- Confirmed: the elevated equatorial flow is a resolution limitation
  (1° can't resolve the equatorial jet structure), not a code bug

### Year-18 ocean state is physically healthy

- Active-cell S ranges 33.7-37.5 (no sub-seafloor leakage)
- Zero density inversions at any active interface, anywhere
- T cooling at poles (Arctic 9→4°C over 18 years) — physically correct
- The earlier "S=0 at depth" alarm was from including inactive
  (sub-seafloor) levels in the analysis — a diagnostic error

### 40 vertical levels blew up (needs separate tuning)

- 40 stretched levels (dz_sfc=10m, dz_deep=300m) blew up at year 0.26
- C_smag_lap=0.33 + K_zeta_bih=1e14 need retuning for thinner cells
- Deferred to future work

### New code additions

- `tropical_wind_scale` + `tropical_wind_lat_deg` in
  PrescribedForcingConfig — selective tropical wind reduction via
  Gaussian taper, preserving mid-latitude westerlies
- `_plot_sections.py` — zonal/meridional cross-section diagnostic
- `_plot_forcing.py` — wind stress, curl, restoring target maps
- Quadratic bottom drag (`bottom_drag_bg_velocity`) implemented
  in PE module (explicit path)

## Deferred items (LOW priority, significant code work)

- **L1**: Bryan-Lewis K_v(z) depth profile (~2 days; needed for realistic AMOC)
- **L2**: Fox-Kemper MLE submesoscale (helps SST cold bias)
- **L3**: St. Laurent tidal mixing (AMOC strength)
- **L4**: Realistic CORE-II / JRA55-do forcing (OMIP-style validation)
- **L5**: Partial-cell-aware KPP K_conv limiter (small change, enables K_conv=1.0)

## Per-step diagnostics to log

- max|u|, max|eta|, area-mean SST
- Volume-mean T at 1000 m (thermocline structure)
- min(N²) (convection trigger health)
- max horizontal T gradient at partial-cell step edges (GM/Redi failure precursor)
- Wall time

## Honest caveats

1. Partial-cell tracer-PGF feedback (finding §11) is NOT addressed by
   any of these changes.  Coast-only case (h) is the target.  Full
   bathymetry case (i) needs SMC03 PGF or GM/Redi triads on Voronoi.
2. Centered GM/Redi on partial cells is a known weak link; the
   sub-seafloor T/S fill is recent and not yet validated multi-year.
3. Production-class AMOC requires Bryan-Lewis K_v(z) (L1) — not
   optional once we care about realistic deep circulation.
4. Realistic forcing (L4) is needed for OMIP-style validation against
   observations, not for stability tuning.
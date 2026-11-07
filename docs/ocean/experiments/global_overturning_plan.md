# Global Overturning Experiment Plan (Wolfe & Cessi 2010 inspired)

*Created: 2026-04-25*

## Context

**Goal**: Create a global baroclinic ocean experiment with overturning circulation,
stratification, and a reentrant Southern Ocean channel as a stepping stone toward
OMIP. Inspired by Wolfe & Cessi (2010, JPO) but adapted:

- Spherical geometry instead of Cartesian beta-plane
- 1-0.5 deg resolution (non-eddy-resolving; OMIP-class)
- Side-by-side lat-lon / MPAS testing

**Starting point**: The existing `global_barotropic_wind` experiment already has
the right geometry (simplified continent 20-60 deg E, Drake Passage open south
of 55 deg S, polar caps at 80 deg) and runs on both lat-lon and MPAS grids.
We extend it from barotropic (uniform T) to baroclinic (stratified + surface
restoring).

**Paper requirements vs what we have**: ~90% of physics infrastructure exists.
The two gaps are: (1) MPAS physics pipeline lacks surface T restoring, and
(2) MPAS has no convective adjustment. Both are addressable.

**Reference**: Wolfe, C.L. and P. Cessi, 2010: What Sets the Strength of the
Middepth Stratification and Overturning in Eddying Ocean Models?
*J. Phys. Oceanogr.*, **40**, 1520-1538. (`docs/references/Wolfe&Cessi2010.pdf`)

---

## Phase 1: Create `global_overturning.py` experiment (lat-lon first)

**New file**: `src/legoesm/ocean/experiments/global_overturning.py`

Fork the structure of `global_barotropic_wind.py` (same continent mask, grid
support pattern, validation pattern). Key additions:

### Config: `GlobalOverturningConfig`

| Parameter | Value | Notes |
|-----------|-------|-------|
| T_surface | 20.0 degC | Surface temperature |
| T_deep | 2.0 degC | Abyssal temperature |
| T_scale_depth | 1000.0 m | Stratification e-folding depth |
| S_uniform | 35.0 PSU | Uniform salinity (T-only EOS) |
| H_max | 4000.0 m | Ocean depth (W&C: 2400; 4000 more realistic) |
| n_levels | 20 | Vertical levels |
| dz_surface | 10.0 m | Top layer thickness |
| dz_deep | 500.0 m | Bottom layer thickness |
| tau_T_days | 30.0 days | SST restoring timescale |
| T_star_eq | 25.0 degC | Equatorial target SST |
| T_star_pole | 0.0 degC | Polar target SST |
| tau_max | 0.1 Pa | Max wind stress (global_wind profile) |
| A_h | 5e4 m2/s | Laplacian viscosity (tunable for resolution) |
| K_v_bg | 1e-5 m2/s | Background vertical diffusivity |
| A_v | 1e-3 m2/s | Vertical viscosity |
| bottom_drag_coeff | 1.1e-3 m/s | Linear bottom drag |
| continent geometry | Same as global_barotropic_wind | 20-60 deg E, Drake at -55 deg S |

### Initial conditions

- Reuse `rest_state_latlon_cgrid_ocean` / `rest_state_mpas_ocean` with continent mask
- Add exponential T stratification: `T(z) = T_deep + (T_surface - T_deep) * exp(z / scale_depth)`
- Pattern: same as `baroclinic_gyre._add_stratification()` but on global domain

### Forcing (lat-lon path)

- `SurfaceForcingConfig(scheme="combined")` for wind + T restoring
  - Prescribed: `wind_profile="global_wind"`, `tau_max=0.1`
  - Restoring: `T_profile="cosine"`, `T_star_eq=25`, `T_star_pole=0`, `tau_T=30d`
- `VerticalMixingConfig(scheme="constant", A_v=1e-3, K_v=1e-5)`
- `OceanConvectionConfig(scheme="enhanced_diffusion")`
- `BottomDragConfig(scheme="linear", r=1.1e-3)`
- `LateralMixingConfig(scheme="none")` (A_h handled in dynamics config)

### Diagnostics

- SSH, SST, surface speed (inherited from global_barotropic_wind)
- Add: mean vertical T profile, T at 1000m depth (middepth stratification)

### Validation criteria

- Fields finite (no NaN)
- Circulation develops (max_speed > threshold)
- Stratification maintained (surface T > deep T)
- Volume conservation (eta drift < threshold)

### Reuse (no duplication)

- `_create_simplified_continent_mask()` from `global_barotropic_wind.py` — factor into shared helper
- `_add_stratification()` from `baroclinic_gyre.py` — reuse or factor out
- `restoring_surface_forcing()` via `scheme="combined"`
- `compute_wind_stress()` with `"global_wind"` profile
- `create_ocean_z_star()` for vertical coordinate

---

## Phase 2: Extend MPAS physics for surface restoring

**File**: `src/legoesm/ocean/physics/mpas_physics.py`

Add surface restoring to `make_mpas_ocean_physics()`:

```python
if has_restoring:
    lat_cell = mesh.grid_lat  # VoronoiMesh implements GridProtocol
    T_star = cfg_r.T_star_eq - (cfg_r.T_star_eq - cfg_r.T_star_pole) * jnp.sin(lat_cell)**2
    dT_dt = dT_dt.at[:, 0].add(-(T_3d[:, 0] - T_star) / cfg_r.tau_T * mask)
```

Support `scheme="combined"` and `scheme="restoring"` (in addition to `"prescribed"`).

### Testing

- Unit test in `tests/ocean/unit/test_mpas_ocean.py` for MPAS surface restoring
- Verify dT_dt shape and sign

---

## Phase 3: Test matrix integration + lat-lon validation

**File**: `scripts/matrix/ocean_test_matrix/experiments.py`

Add `run_global_overturning()` runner:

- Default duration: 90 days
- Quick duration: 5 days
- Default resolution: 36x72 (5 deg, fast)
- Grid support: latlon, mpas (cubed_sphere excluded per issue #100)
- Vertical: 20 levels, 10m surface, 500m deep

### Validation (lat-lon 1 deg, 60-90 days)

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --grid latlon --resolution 180x360 --levels 20 --only global_overturning
```

Check: gyres, ACC, stratification, WBC, stability.

---

## Phase 4: MPAS cross-grid validation

```bash
JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
  --grid mpas --resolution ico6 --levels 20 --only global_overturning
```

Compare lat-lon vs MPAS: SSH, SST, circulation patterns, stratification profiles.

---

## Phase 5: Resolution scaling + diagnostics (future)

- 0.5 deg lat-lon (720x360) — needs timestep adjustment
- MPAS ico7 (~0.9 deg equivalent)
- MOC streamfunction computation (new diagnostic)
- Meridional heat transport diagnostic
- Multi-year runs for deep-ocean adjustment
- GM/Redi eddy parameterization (not yet in codebase)

---

## Key risks and mitigations

1. **Stability at 1 deg with 90-day run**: CFL is comfortable with dt=3600s.
   Convective adjustment prevents static instability. Fallback: reduce dt or
   increase A_h.

2. **MPAS surface restoring**: Straightforward — restoring has zero momentum
   tendency, only dT_dt. VoronoiMesh has `grid_lat` via GridProtocol.

3. **No eddy parameterization at 1 deg**: W&C is eddy-resolving; at 1 deg
   stratification and MOC will be diffusively maintained. Acceptable as
   stepping stone — OMIP models also run at 1 deg (typically with GM/Redi).

4. **Vertical coordinate**: z-star with 20 levels and stretching.
   `create_ocean_z_star(n_levels=20, H_max=4000, dz_surface=10, dz_deep=500)`.

---

## Implementation order

1. Factor shared continent mask helper (avoid code duplication)
2. Create `global_overturning.py` with lat-lon support
3. Quick test at 5-deg resolution, 5 days — verify it runs
4. Extend MPAS physics with surface restoring
5. Add MPAS support to `global_overturning.py`
6. Register in test matrix
7. Run at 1-deg lat-lon, 60 days — visual inspection
8. Run at MPAS ico5/ico6, 60 days — cross-grid comparison

---

# Phase 2 — Drake Passage / ACC diagnosis (started 2026-04-27)

## Status of the 50-yr GM/Redi run

After 50 sim-years (`results/ocean/global_overturning_50yr_gmredi/`), the
Drake band (j=2..6, lat -77.5° to -57.5°) shows:

- Surface zonal-mean u: +1 to +3 cm/s east (weak, consistent with the local
  westerly wind τ_x ≈ 0.024 Pa)
- Bottom zonal-mean u: −7 cm/s **west** (strong)
- Depth-mean U_baro: **−3.7 cm/s west**
- Drake section transport: **−405 Sv** (real ACC ≈ +130–170 Sv)

By year 30 the barotropic transport has equilibrated — this is the model's
steady-state answer, not a spinup transient.

## Sensitivity tests run (2026-04-27)

| Configuration | Drake transport (5-yr after restart) | Δ |
|---|---|---|
| 50-yr endpoint (control) | −407 Sv | — |
| `bottom_drag_coeff × 0.2` (r = 2.2e-4) | −491 Sv | **−84** |
| Ridge 3000 m at lon idx 30 (mid-Pacific) | −325 Sv | **+82** |
| Ridge 3000 m at lon idx 8 (Drake-position) | ~−345 Sv | **+62** |

Reducing drag drove transport *more* westward (LaCasce & Isachsen 2010
predict this for flat-bottom limit — sign set by f/H, not wind/drag).
Form drag from a single 25%-obstacle ridge bought ~70–80 Sv regardless of
location. Ridge-location is approximately neutral in this configuration.

## Diagnosis

The dominant zonal-momentum balance in the Drake band, depth-integrated and
zonal-mean, is approximately:

> τ_wind = ρ·r·U_baro + ρ·∂_y⟨∫v·u dz⟩

With observed values: `+0.024 Pa = -0.043 Pa + +0.067 Pa`. The
**meridional advective momentum flux divergence** (Deacon-cell carrying
zonal momentum across the band) is ~3× the wind in magnitude, with the
correct sign to drive U_baro westward. In the real ocean, ~95 % of the wind
is balanced locally by topographic form stress at ridges (Munk-Palmén 1951;
Masich et al. 2015); with flat bottom, the wind has to be balanced by
bottom drag + meridional advection, and the latter dominates and gets the
sign wrong.

Two earlier hypotheses were ruled out:
- *GM-on-momentum missing*: standard ocean GCMs only use GM on tracers and
  still produce reasonable ACCs. Communication is via thermal-wind shear
  set by the GM-flattened buoyancy field — sufficient.
- *Spinup too short*: barotropic adjustment is days, not years. The
  −405 Sv is the equilibrated answer. (Deep T is uninspun, but that sets
  the *baroclinic shear*, not the barotropic mode.)

## Code-structure issue (drag is being double-counted)

A code review (`docs/dev-notes/research/bottom_cell_drag_dycore_design.md`) identified
that bottom drag is being applied via two non-equivalent paths:

1. **Bottom-cell explicit drag** at `ocean_pe_latlon_cgrid.py:545–552` —
   correct linear drag on `u[..., -1]` and `v[..., -1]`. Its depth-mean
   is folded into `F_slow_u/v` at `ocean_model_latlon_cgrid.py:374–374`
   and carried into the barotropic substep. **Correct path.**
2. **Independent depth-mean drag** at `barotropic_latlon_cgrid.py:340–346`
   — the `(1 − dt·r/H)` factor on `U_bar`. **Redundant.**

Both use the same `r = config.bottom_drag_r`, so the model is currently
applying ~2× the intended bottom drag. Cleanup is a code-clarity fix; its
scientific impact is moderate (tens of Sv shift in U_baro), not
transformative.

## Three-phase plan

### Phase 1 — Quantitative attribution via momentum budget (DONE 2026-04-27)

Wrote `scripts/tmp/diagnose_drake_momentum_budget.py`.  Computed the
depth-and-zonal-integrated Drake-band zonal-momentum budget from
time-mean restart and `time_mean.npz` fields.

Original hypothesis (Deacon-cell-carries-momentum) was **falsified**:

| Run | F_wind | F_drag_bot (path 1+2) | F_drag_baro (path 3) | F_adv (flux form) | F_vortcor (vector-inv) | F_visc | **residual** |
|---|---|---|---|---|---|---|---|
| 50yr GM/Redi | +0.030 | +0.063 | +0.036 | ~0 | −0.008 | 0 (depth-mean exact) | **+0.121** |
| drag ×0.2 | +0.030 | +0.013 | +0.008 | ~0 | −0.001 | 0 | **+0.050** |
| ridge lon30 | +0.030 | +0.055 | +0.031 | ~0 | −0.014 | 0 | **+0.102** |
| ridge lon8 | +0.030 | +0.059 | +0.034 | ~0 | −0.009 | 0 | **+0.114** |

(All in Pa, eastward = +.  Ridge runs use 5-yr time means; 50yr uses mean
of last 5 restarts, yrs 30–50.)

Key findings:

1. **F_adv ≈ 0**: depth-integrated meridional flux of zonal momentum
   (∂_y⟨∫v·u dz⟩) is essentially zero at the band boundaries. Surface
   Ekman is +6 cm/s northward at j_v=7 in the model, but the deep
   return cancels almost exactly in the depth integral.
2. **F_vortcor ≈ −0.01 Pa**: the vector-invariant ⟨ζ × v⟩ form gives
   the same answer as F_adv (small) — confirms consistency between
   flux-form and vector-invariant decompositions.
3. **F_visc on the depth-mean is exactly 0.** Lateral viscosity acts on
   `u_prime = u − U_bar` (line 372 of `ocean_pe_latlon_cgrid.py`), and
   `u_prime` has zero depth-mean by construction.  We previously
   reported a `+0.016 Pa` "F_visc" from `A_h · ∂²U_baro/∂y²` — that's a
   misleading diagnostic of how the *zonal-mean* profile looks, not what
   the model actually applies.
4. **+0.12 Pa unaccounted for in the 50yr budget** — the residual is
   itself ~4× the wind, all eastward.  We've ruled out steady-state
   advection (both forms), viscosity, drag, and wind as the source.

What this leaves as the candidate sink:

- **Reynolds correlations** ⟨u'v'⟩, ⟨ζ' v'⟩ from sub-monthly time
  variability (inertial oscillations resolved at dt=600 s but aliased at
  monthly snapshots and 5-yr-restart cadence).  Cannot be computed from
  saved fields alone.
- **Numerical dissipation in 1st-order upwind vertical momentum
  advection** (`ocean_pe_latlon_cgrid.py:461` confirms the scheme).
  Equivalent A_v ≈ |w|·dz/2.  Could be substantial near the band's
  surface Ekman / deep return convergence.
- **Discrepancy between time-mean-of-product and product-of-time-means**
  in any nonlinear advection term.

**Decision**: attribution from offline budget is incomplete.  Before
editing the dynamics (Phase 2), close the budget rigorously via online
diagnostics — see Phase 1.5 below.

### Phase 1.5 — Instrumented momentum-tendency diagnostics

The offline budget cannot resolve time-correlation (Reynolds) or
numerical-dissipation contributions.  Production ocean models
(MOM6 `MOM_diagnostics`, MITgcm `DIAGNOSTICS_PKG`, NEMO `trd_*`)
solve this by **instrumenting the dynamics at the point each term is
computed** — guaranteeing budget closure to machine precision.  We
follow that pattern.

**Design**:

1. **`MomentumTendencyDiagnostics` NamedTuple** (new in
   `src/legoesm/ocean/state.py`, parallel to
   `LatLonCGridOceanTendencies`).  Each field is a 3D `Field` carrying
   one component of the per-step momentum tendency:

   For `du_dt`:
   - `KE_PGF`: −∂_x KE − (1/ρ_0)·∂_x p (lines 395–396 of
     `ocean_pe_latlon_cgrid.py`)
   - `vortcor`: ζ × v_at_u (line 439–440)
   - `vertadv`: 1st-order upwind flux-form ∂_z(w·u) (line 462)
   - `Ah_lap`: A_h · ∇²u_prime (line 517) [exactly zero in depth-mean]
   - `botdrag_path1`: −r·u/dz_bot at level −1 only (line 551)
   - `Av_vert`: A_v · ∂²u/∂z² (line 568)
   - `phys_wind`: τ_x/(ρ·dz_top) at surface, 0 elsewhere (line 585)
   - `total`: the actually-applied du_dt (sanity check: must equal
     the sum of the components above to machine precision)

   And from the barotropic substep (`barotropic_latlon_cgrid.py`):
   - `bt_path3_drag`: equivalent baroclinic-tendency-equivalent of the
     `(1 − dt·r/H)` factor on `U_bar`, broadcast back to 3D as a
     constant-with-depth tendency
   - `bt_coriolis_pgf`: depth-mean of (Coriolis × V_at_u − g·∂η/∂x)
     accumulated over substeps

   Same `dv_dt` siblings for v.

2. **Code-change locations** (all keep existing call sites
   backward-compatible):

   a. `src/legoesm/ocean/state.py`: add
      `MomentumTendencyDiagnostics` NamedTuple.

   b. `src/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py`:
      add a `diagnose_momentum: bool = False` argument to
      `compute_tendencies(...)`.  When `True`, build the
      diagnostics tuple by saving each `du_dt` increment as it is
      computed, and return `(tendencies, diagnostics)` instead of just
      `tendencies`.  When `False` (default), unchanged behavior.

   c. `src/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`: add a
      sibling `step_with_diagnostics(state, dt) -> (state_new,
      diagnostics)` method.  Internally calls
      `compute_tendencies(diagnose_momentum=True)` and threads the
      diagnostics through the RK stages.  For multi-stage RK, return
      the *stage-averaged* tendency (so it represents the actual
      contribution to `state_new − state`).

   d. `src/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py`: add a
      similar `step_with_diagnostics` for the barotropic substep loop.
      Diagnose the path-3 drag contribution from
      `(U_bar_after_drag − U_bar_before_drag) / dt_s`, accumulated.

3. **Closure unit test** (`tests/ocean/unit/test_momentum_diagnostics_closure.py`):
   one timestep, assert `Σ components == total` to within 1e-12 in
   relative norm.  Standard practice in MOM6/NEMO: budget cannot diverge
   from the integrated state by more than floating-point order-of-ops.

4. **Driver script** `scripts/run/run_drake_momentum_budget.py`:
   - Restart from `restart_day018250.npz`.
   - Run 1 sim-year (52,560 steps at dt=600 s) with
     `step_with_diagnostics`.
   - Each step: accumulate per-term tendency × dt into running 3D sums.
     Also accumulate state field for time means.
   - Write monthly snapshots of each term.
   - At end: divide sums by total time to give per-term time-mean
     tendencies; verify `Σ_terms ≈ (state_final − state_initial) / T`
     to machine precision.
   - Output: `results/ocean/momentum_budget_online/` with
     `tendency_3d_means.npz`, `tendency_band_summary.npz`,
     `momentum_budget_closure.png`.

5. **Cost**: 1 sim-year ≈ 7 min wall + ~30 % overhead for diagnostic
   bookkeeping → ~10 min. Storage: 12 × ~10 terms × 36×72×20×8 bytes
   ≈ 50 MB.

6. **Decision gate**: closure must be `‖Σ_terms − total‖ < 1e-10 ·
   ‖total‖` *element-wise*.  Once that's verified, the residual we
   observed in Phase 1 (~+0.12 Pa) decomposes uniquely into one of:
   - A specific Reynolds-stress component (will show as a difference
     between the saved tendency time-mean and the same term computed
     from time-mean state).
   - The numerical dissipation in `_flux_form_vertical_momentum_advection`
     (will show in `vertadv` term — direct comparison to the analytic
     centered form quantifies the upwind dissipation).
   - Some other term we haven't enumerated (impossible with a closed
     budget — we'd see it as a non-zero diagnostic component we forgot
     to attribute).

**Estimated effort**: ~3–4 hours of careful work to instrument the
dynamics + write the driver + closure test.  Pays off:
(a) settles the −405 Sv puzzle definitively, (b) the diagnostic
infrastructure is reusable for any future ocean momentum question, and
(c) we can verify Phase 2's drag-formulation cleanup actually does what
we expect at the budget level.

#### Implementation progress (2026-04-27 → 2026-04-28)

Phase 1.5 instrumentation complete; budget closure achieved; cross-cutting
barotropic-mode-noise issue spun out (see `docs/dev-notes/issues/`).

- ✅ `MomentumTendencyDiagnostics` NamedTuple in `src/legoesm/ocean/state.py`
  (24 fields, reusable convention for future tracer/energy budgets).
- ✅ `latlon_cgrid_ocean_baroclinic_tendencies` instrumented with
  `diagnose_momentum: bool = False` flag; each contribution captured at
  its assignment site, closure automatic.
- ✅ `LatLonCGridOceanModel.tendencies_with_diagnostics` wraps the
  diagnostic call.
- ✅ Closure unit test (`tests/ocean/unit/test_momentum_diagnostics_closure.py`):
  4/4 at 1e-12 relative.  Existing C-grid regression suite (17 tests)
  still passes.
- ✅ Diagnostic runner refactored: shared helper
  `scripts/run/_drake_momentum_budget_runner.py` rolls
  `tendencies_with_diagnostics + step + accumulate` into a JIT-compiled
  `lax.scan` block.  Speedup measured at **8.4× (82 min → 9.8 min) for
  a 1-yr run** with bit-equivalent science (headline number ±0 at 3
  decimals despite 1e-4 per-cell fp divergence from chaotic compounding).
- ✅ Three runner scripts (baseline / divdamp / implicit) thinned to
  ~80 LOC each, all calling the helper.  Verification harness at
  `scripts/validate/verify_drake_runner_jit.py`.
- ✅ Drake-band budget computed across all three configurations
  (baseline / Stage 0 / Stage 3-implicit) — see issue doc table.

#### Phase 1.5 outcome → cross-cutting issue spawned

The momentum-budget instrumentation revealed **grid-scale chequerboard
noise in `V_baro`** that, via `ρ·H·f ≈ 5×10⁵` amplification, produced
the dominant westward sink in the budget — a numerical artefact, not
physics.  Tracked separately in
**`docs/dev-notes/issues/barotropic_mode_noise.md`** with three-stage fix path
and acceptance criteria.

The Stage 3 implicit Crank–Nicolson barotropic solver shipped in
commit `de14c57` (separate session) and reduced the Drake-band Coriolis
stress from −0.098 Pa (baseline / Stage 0) to −0.005 Pa (Stage 3 from
old restart) and to **−0.000049 Pa from a fresh implicit-solver
spinup** — passing Crit 2 by two orders of magnitude.

### Phase 2 — Bottom-drag formulation cleanup

Designs in `docs/dev-notes/research/bottom_cell_drag_dycore_design.md` (numerics) and
`docs/dev-notes/research/bottom_cell_drag_ocean_design.md` (physics).

1. Switch path-1 explicit Euler bottom-cell drag to **implicit per-cell**:
   `u_bot ← u_bot / (1 + dt·r/dz_bot)`. Unconditionally stable, AD-clean.
2. Delete the redundant depth-mean drag at
   `barotropic_latlon_cgrid.py:340–346`.
3. Add config flag `bottom_drag_location: Literal["depth_mean",
   "bottom_cell"]` to `LatLonCGridOceanConfig` (and MPAS equivalent),
   default `"depth_mean"` for backward compatibility. New experiments use
   `"bottom_cell"`.
4. Mirror in MPAS path (`ocean_pe_mpas.py`, `barotropic_mpas.py`).
5. Two unit tests:
   - Flat-bottom rest-state spinup: both modes converge to
     U_baro ≈ τ/(ρ₀·r) within 5 %.
   - Drake-channel u_bot decay: `bottom_cell` mode shows decay timescale
     ≈ dz_bot/r ≈ 2.3 d; `depth_mean` mode does not constrain u_bot.
6. Widen `tests/ocean/unit/test_no_scheme_duplication.py` so the
   `implicit_bottom_drag_factor` helper can be gated on the new mode.

**Decision gate**: regression tests pass + the unit tests above pass.

### Phase 3 — Final ACC test

Re-run from the 50-yr restart with the cleaned drag formulation
(`bottom_drag_location="bottom_cell"`) + topography (single ridge or
multi-ridge), 5 yr.

Expected effect (rough): cleanup alone ~+30–50 Sv, ridge ~+80 Sv,
combined ~+100–150 Sv — likely still not flipping the sign to positive,
but a clean, interpretable ~−250 to −300 Sv state. Whether we cross zero
depends on stacking; multi-ridge may be needed.

If still strongly negative, the next lever is **quadratic bottom-cell
drag** with `u_bg ≈ 0.1 m/s` and `C_d ≈ 1–3e-3` (MOM6 OM4 default), as
sketched in the ocean physics design doc.

## Reference docs from this phase

- `docs/dev-notes/research/bottom_drag_literature.md` — typical drag values and ACC
  scaling from the literature (Munday-Hogg-Marshall, LaCasce-Isachsen,
  Masich, etc.)
- `docs/dev-notes/research/why_westward_drake.md` — diagnosis of the −405 Sv state
  (with caveats: agent overweighted GM-on-momentum and spinup arguments)
- `docs/dev-notes/research/bottom_cell_drag_dycore_design.md` — numerics design,
  including the double-counting finding
- `docs/dev-notes/research/bottom_cell_drag_ocean_design.md` — physics design,
  drag-law calibration, MITgcm/MOM6/NEMO comparison
- `docs/dev-notes/research/zstar_vbaro_residual_investigation.md` — z-star
  continuity analysis ruling out a continuity bug; confirms ⟨V_baro⟩ ≠ 0
  is allowed by discrete continuity even with exact mass conservation
- `docs/dev-notes/research/barotropic_noise_handling_in_production_models.md` —
  inventory of how MOM6 / MITgcm / NEMO / POP / MPAS-O / ROMS suppress
  barotropic-mode noise; identifies our cosine filter as a likely
  contributing factor

## Cross-cutting issue spun off

Phase 1.5 surfaced a model-quality issue that affects all lat-lon
C-grid experiments using barotropic substepping, not just this
experiment.  Tracked separately in
**`docs/dev-notes/issues/barotropic_mode_noise.md`** with three-stage fix path
and explicit acceptance criteria.  The −405 Sv Drake transport in
this experiment is the most visible symptom, but any momentum-budget
diagnostic relying on `⟨V_baro⟩` is contaminated.

## Status (2026-04-28)

Phases 2 and 3 of the original ACC plan have been **superseded** by
the structural fix from the barotropic-noise issue.  With the implicit
Crank–Nicolson barotropic solver and a clean fresh spinup, the
momentum-budget closure is meaningful and Phase 2 / 3's questions
(does drag cleanup matter? what does the ACC look like?) get answered
by re-running the original 50-yr experiment with the new solver.
That re-run is in flight as of 2026-04-28 in
`results/ocean/global_overturning_50yr_implicit/` (40-yr continuation
from a 10-yr fresh spinup, restart cadence matching the original 50yr
run for direct comparison).

### Phase 2 / 3 (revised) — Re-run the 50yr experiment cleanly

1. ✅ **10-yr fresh spinup with implicit solver** — done 2026-04-27.
   Output: `results/ocean/global_overturning_implicit_spinup/`.  Final
   state at `restart_day003650.npz`, healthy diagnostics (|η|max=3.3 m,
   T∈[0,23.9]°C, |u|max=0.73 m/s).  Spinup driver:
   `scripts/run/global_overturning/run_global_overturning_implicit_spinup.py`.

2. ✅ **1-yr verification from fresh spinup state** — done 2026-04-28.
   Output: `results/ocean/momentum_budget_online_implicit_postspinup/`.
   - Crit 2 (Drake Coriolis stress): **−0.049 mPa**, vs target 5 mPa,
     vs old-restart-broken-solver −8018 µPa.  Passes by 100×.
   - Crit 1.2 (point-wise max ⟨V_baro⟩ off polar): **0.216 m/s** vs
     target 0.005 m/s — still failing.  This is the C-grid Coriolis
     rotational null mode that the implicit solver can't damp; needs
     Follow-up C (barotropic-mode lateral viscosity) for full closure.
   - **Doesn't affect the Drake-band science**: Drake band sits at
     moderate Coriolis where the noise is much smaller; Crit 2 confirms
     this.

3. 🟡 **40-yr continuation to sim-yr 50** — launched 2026-04-28.
   Output: `results/ocean/global_overturning_50yr_implicit/`.  Restart
   cadence every 5 sim-yr (days 5475, 7300, ..., 18250) matching the
   original `global_overturning_50yr_gmredi/` for direct comparison.
   ~3–4 h wall.  Driver:
   `scripts/run/global_overturning/run_global_overturning_50yr_implicit_continuation.py`.

4. ⏳ **Final comparison plots** vs original (broken-solver) 50yr run —
   pending continuation completion.  The headline question being
   answered: with the chequerboard noise removed, what does the Drake
   transport actually look like?  Original (broken) gave −405 Sv
   westward.

### Open follow-ups (post-50yr-rerun)

- **Follow-up C** (barotropic-mode lateral viscosity, ~30 LOC) — needed
  to fully close Crit 1.2/1.3 high-latitude chequerboard.
- **Follow-up B** (MPAS implicit solver) — generalize the structural
  fix to MPAS-Ocean.
- **Quadratic bottom-cell drag** with `u_bg` background velocity (the
  original Phase 2 motivation, now better motivated since the budget
  is interpretable) — sensible if Drake transport still looks off after
  the clean re-run.

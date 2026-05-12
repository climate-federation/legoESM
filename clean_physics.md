# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iterations 1-29 — Summary (compressed 2026-05-12 after iter-29)

### Constants & shared utilities
- `constants.p_atm_std = 101325 Pa` and `constants.R_universal = 8.314462618`
  (CODATA 2018) added.  `ocean/surface_forcing/bulk_formulas.py` and
  `land/carbon/stomata.py` now reference them instead of local literals.
- `convection/mass_flux._compute_column_geometry` delegates to shared
  `_shared.compute_layer_dz` / `compute_rho` with `q_v` for virtual-T moist
  geometry.  Threaded through all 5 mass-flux schemes (Bechtold, Emanuel,
  Tiedtke, Kain-Fritsch, Zhang-McFarlane).  Full-level heights use mid-layer
  `cumsum(dz)[::-1] - 0.5·dz`.
- Kessler microphysics uses shared `_warm_rain.rain_evaporation` helper.
- Constant-hygiene audit (iter-12): all `g`/`R_d`/`c_pd`/`L_v`/`L_s`/`L_f`/
  `sigma_sb`/`T_freeze`/`Omega`/`R_earth`/`rho_air`/`rho_ocean`/`epsilon`
  references go through `constants.py`.

### Conservation & monotonicity
- **Sedimentation** (`microphysics/output.py`): optional `dt` applies
  Bott/FCT positivity flux limiter (`flux ≤ q·ρ·dz/dt`).  New
  `return_surface_flux=True` returns the dt-limited bottom flux for
  conservative precipitation diagnostics.  New `extra_sink` parameter
  jointly caps sed + rain-evap against `q_r/dt`.  Threaded through
  kessler/morrison/thompson/seifert_beheng.
- **Rain evaporation** (`_warm_rain.py`): optional `dt` clamps evaporation
  to `q_r/dt`.  All four schemes pass `dt`.
- **Sundqvist autoconversion**: donor cap so `P_auto · dt ≤ q_c`.
- **Sea-ice transport**: replaced centered divergence with conservative
  monotone PPM (Colella–Woodward with limiter); `n_subcycles` runs PPM
  substeps with `lax.scan`.  `SeaIceConfig.transport_subcycles` exposes.
- **ITD remap**: volume-conserving rescale `a_post · h_post = a_pre · h_pre`;
  residual folded into thickness when `a > 1`.
- **RRTMGP**: upper-clip `q_v ≤ 0.99` so the `(1 - q_v)` denominator in the
  H2O VMR conversion is bounded.
- **Morrison + Thompson vapor donor clamp**: jointly clamp `cond_pos +
  dq_i_dep` against `q_v/dt` (codex iter-25 + iter-29).
- **DCA STANDARD MSE conservation** (test updated to match in-scheme latent
  heating in `delta_T_lh`).

### CFL caps & stability
- **Ocean enhanced_diffusion convection**: explicit branch delegates per-
  interface CFL cap to `vertical_diffusion_variable_K(dt=...)` using
  `dz_actual = dz_ref · jacobian`.  Optional explicit `dt` precedes
  `cfg.cfl_dt_estimate`.
- **Ocean vertical mixing leaf** (`mixing.py`): `vertical_diffusion` and
  `vertical_diffusion_variable_K` accept optional `dt` + `cfl_safety`.
- **Ocean vertical-mixing schemes** (KPP, Richardson, constant) thread
  `dt` to the leaf so the CFL cap is reachable from every entry point.
- **Ocean lateral mixing harmonic + biharmonic**: opt-in `enforce_cfl` flag
  with `cfl_dt_estimate` / `cfl_safety` config fields; harmonic
  `A_h`/`K_h ≤ cfl_safety · dx² / dt` (default safety 0.20), biharmonic
  `B_h_* ≤ cfl_safety · dx⁴ / dt` (default safety 0.05).
- **EDMF mass flux** (`turbulence/edmf.py`): caps `M ≤ 0.5·ρ·dz/dt`.

### Coupler / feedback paths
- **Sea-ice `_bulk_flux_dispatch`** helper centralises MOST/COARE/
  Large-Yeager/simple_bulk dispatch.  Used by slab path, dynamic
  multi-category `_thermo_cat`, and diagnostic `_build_response`.
- **`_build_response` ice → ocean feedback**: optional `h_old`, `ocean_*`,
  `dt` kwargs compute `freshwater_flux`, `ocean_heat_extraction`,
  `ocean_stress_x/y` (per-ice-area thickness rate; full V=h·A CICE
  bookkeeping deferred).  Dynamic path threads `h_agg_post_transport`
  (post-transport, pre-thermo) so FW = thermodynamic ΔV only, no
  transport redistribution.
- **Multi-cat lead freeze**: `_thermo_single` gains `open_water_fraction`
  + `enable_lead_freeze` kwargs.  Dynamic dispatch drives concentration
  growth by aggregated `(1 - sum_k conc_k)` and fires open-water freeze
  in category 0 only.
- **Sea-ice radiative emission**: `lw_up = ε·σ·T⁴ + (1-ε)·lw_down`
  (grey surface, reflected-LW component included; test updated).
- **Multilayer land** dropped bogus `root_frac[None,:]` unsqueeze.
- **Multilayer land dew/deposition routing** (iter-23+24): when
  `has_snow=False`, dew is routed to `flux_top` (no transpiration sink);
  when `has_snow=True`, snowpack absorbs the entire latent flux and soil
  sees zero direct flux.

### Tracer-mixing land mask (iter-25, iter-27)
- `ocean/physics/lateral_mixing/harmonic.py` + `biharmonic.py` output T/S
  tendency is multiplied by `mask_3d` so land cells receive zero tendency.
  Full no-flux BC (replicate ocean values at land neighbours before the
  operator) requires an operator-stencil refactor; output mask is the
  safe minimum.

### Test fixes
- `Test8i_StefanBoltzmann::test_lw_up_matches` — grey-surface LW.
- `test_hybrid_tracer_path_calls_vertical_advection` — relative path.
- `test_ah_lat_scaling` (4 tests) — pass `power=2` for legacy cos² coverage.
- `test_dca_extended_mse_conservation` — STANDARD MSE (in-scheme latent heat).
- 7 / 7 pre-existing failures resolved (1 in iter-13, 6 in iter-14).

### Regression tests (iter-8 + iter-22)
- `tests/ocean/unit/test_ocean.py::TestVerticalMixing::test_vertical_diffusion_cfl_cap_keeps_step_stable`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::test_sedimentation_cfl_positivity`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::test_sedimentation_surface_flux_conservation`
- `tests/atmosphere/hydrostatic/unit/test_microphysics.py::TestSundqvist::test_autoconversion_does_not_drive_qc_negative`

### Test status (post iter-29)
- 17 / 17 atmosphere hydrostatic integration tests pass.
- 169 / 169 atmosphere convection + microphysics + ocean vertical mixing
  targeted tests pass.
- 95 / 95 atmosphere turbulence tests pass.
- 79 / 79 sea-ice unit tests pass.
- 75 / 75 land carbon + multilayer + diff_land tests pass.

## Inspected & clean (no fix needed)
- `land/snow_budget.py`, `land/carbon/carbon_cycle.py`, `land/richards.py`,
  `land/soil_thermal.py` — implicit / well-disciplined.
- `atmosphere/physics/turbulence/vertical_diffusion.py` — implicit.
- `atmosphere/physics/clouds/cloud_fraction.py` — shared
  `saturation_mixing_ratio`.
- `atmosphere/physics/gravity_wave_drag/lindzen.py` — `constants.*`,
  differentiable scan.
- `ocean/physics/shortwave_penetration.py` — conservation correction.
- `ocean/physics/bottom_drag/quadratic.py` — proper sign convention.
- `ice/rheology.evp_stress_update`, `ice/dynamics.evp_solver` — implicit
  / CICE convention.
- `atmosphere/physics/convection/{dca,kuo,zhang_mcfarlane}.py` — moist
  static-energy budgets verified.
- `atmosphere/physics/microphysics/seifert_beheng.py` — joint donor clamps.
- `atmosphere/physics/turbulence/{clubb_lite,holtslag_boville,louis,ysu}.py`
  — all use shared `virtual_temperature`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter (codex narrow review).
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping (codex
  iter-22 #1 + iter-25 revert): h and conc evolve semi-independently
  in the current slab path, so `d(h·conc)/dt` mis-counts new lead ice
  (h vs h_new_ice) and double-counts melt-retreat.  Per-ice-area `dh/dt`
  is the current bookkeeping.
- No-flux BC for harmonic/biharmonic tracer Laplacian (replicate ocean
  values at land neighbours before the operator).  Output mask is the
  current containment.
- Threading `dt` into `physics_fn` across the 3 ocean PE backends + 3
  model drivers — production uses the implicit ocean solver
  (unconditionally stable), so the marginal stability gain is low.
- Visual / long-run validation of the new ice → ocean feedbacks.

## Next iterations
Continue addressing further codex adversarial-review findings until
physics implementation is provably conservative, monotone, and CFL-safe
across all parameterizations.

### Iteration 30 — 2026-05-12

Compression iteration.  Folded iter-20 through iter-29 into the
"Iterations 1-29 — Summary" section.  Detailed per-iteration narratives
remain in the commit messages on `clean_physics`.

### Iteration 36 — 2026-05-12

**Add end-to-end fp32 regression test for Morrison.**

Iter-35 consolidated all 8 microphysics donor clamps onto
`donor_clamp_scale`.  Direct measurement: with the legacy 1e-30 floor,
a Morrison fp32 run at T = 240 K, q_v ≈ q_sat_ice + epsilon produced
NaN gradients.  After iter-35 the same call produces finite zero
gradients (correct: the column is essentially inactive).

Added `TestMorrison::test_morrison_fp32_grad_finite_at_ice_saturation`
to lock this in.  Earlier `test_qv_clamp_divisor_floor_protects_VJP`
(iter-34) tests the helper in isolation; this test confirms
end-to-end Morrison fp32 differentiability — the bigger regression
risk if a future donor clamp drifts back to a tighter floor.

**Tests (post iter-36):** 64 / 64 microphysics tests pass.

### Iteration 35 — 2026-05-12

**Consolidate all microphysics donor clamps onto
`donor_clamp_scale`.**

Iter-34 extracted `donor_clamp_scale` and switched only the q_v
clamps to it.  The other donor clamps (Kessler q_c; Morrison q_c, q_i,
q_s; Thompson q_c, q_i, q_s; SB q_c) still used the legacy
`1e-30`-floored pattern that is fp32-overflow-prone in the same way
the iter-32 finding flagged for q_v.  Migrated all of them to
`donor_clamp_scale` so the AD-safe floor (1e-15) is applied
uniformly.

Net result: 8 donor clamps across 4 schemes (Kessler, Morrison,
Thompson, Seifert-Beheng) now route through one helper.  The 1e-30
literal floor is removed everywhere.  The kessler double-where
``sink_active`` guard is gone (subsumed by the helper).

**Tests (post iter-35):** 63 / 63 microphysics tests pass.

### Iteration 34 — 2026-05-12

**Fix iter-33 codex stop-time finding: test does not exercise
production clamp.**

Iter-33's test re-implemented the donor-clamp pattern locally
instead of calling production code, so it could not catch a
regression where the production helper drifts from the test copy.

Refactored: extracted `donor_clamp_scale(q_avail, sink_total, dt,
divisor_floor=1e-15)` into `_warm_rain.py` as a shared AD-safe
helper.  Morrison and Thompson q_v clamps now call this helper
directly (replacing the inline `jnp.maximum`/`jnp.minimum` pattern).
The regression test now imports and exercises THIS production
function rather than a copy.

```python
# tests now do:
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    donor_clamp_scale,
)
grad = jax.grad(lambda q: donor_clamp_scale(...))(q_v)
```

The test also adds an active-regime check (`q_v_small=1e-6,
sink_big=1e-3` → scale = 1e-4) so it catches both the
floor regime (gradient zero) and the active regime (correct scale).

**Tests (post iter-34):** 63 / 63 microphysics tests pass.

### Iteration 33 — 2026-05-12

**Fix iter-32 codex stop-time finding: test does not exercise the
claimed tiny-positive-sink path.**

Iter-32's regression test
(`test_differentiable_tiny_positive_sink_qv_clamp` on Morrison) was
constructed to engage the `q_v / sink_dt` divisor at tiny values,
but the broader Morrison run NaN'd in fp32 from a separate path
(unrelated to the q_v clamp).  The test passed in fp64 (sufficient
dynamic range) and failed in fp32 from a different overflow site.

Replaced with a focused unit-level test
(`test_qv_clamp_divisor_floor_protects_VJP`) that exercises ONLY the
divisor-floor pattern used in the q_v clamp.  Forces a `sink_total =
1e-25` (well below the 1e-15 floor) in fp32 and asserts:
1. Gradient w.r.t. `q_v` is finite.
2. Gradient w.r.t. `sink_total` is finite.
3. Both gradients are zero-magnitude (the floor regime is "inactive"
   in physical terms; `min(1, huge) = 1` gates the gradient).

This directly validates the iter-32 floor without depending on
Morrison's other fp32 paths.

The separate fp32 NaN in deep-ice Morrison is documented as a
known-limitation (production runs use fp64 where it does not trigger).

**Tests (post iter-33):** 63 / 63 microphysics tests pass.

### Iteration 32 — 2026-05-12

**Fix iter-31 codex stop-time follow-up: tiny-positive-sink NaN.**

The iter-31 fix used a boolean ``sink_active = sink > 0`` guard with
a sentinel ``1.0`` on the inactive branch.  This protected
``sink_active = False`` from the 0/eps division but the ACTIVE branch
with a TINY positive sink (e.g. ``sink_dt = 1e-25``) still hit the
VJP ``-q_v / sink_dt² ≈ -q_v / 1e-50`` which overflows fp32 → NaN.

Replaced the boolean guard with a fp32-safe FLOOR on the divisor:

```python
qv_sink_dt_safe = jnp.maximum(qv_sink_total * dt_safe, 1e-15)
qv_scale = jnp.minimum(1.0, qv_avail / qv_sink_dt_safe)
```

Worst-case VJP is ``-q_v / 1e-30 ≈ -4e28`` — safely within fp32
dynamic range (~3.4e38).  At the floor ``jnp.maximum`` subgradient is
zero, which is physically correct (no scaling, no sensitivity to the
tiny sink).

Added `TestMorrison::test_differentiable_tiny_positive_sink_qv_clamp`
that exercises a column with q_v just at saturation wrt ice — the
boolean-only guard would have NaN'd there.

**Tests (post iter-32):** 63 / 63 microphysics tests pass including
the new tiny-positive-sink regression.

### Iteration 31 — 2026-05-12

**Fix iter-30 codex stop-time finding: q_v clamp NaN gradients.**

The Thompson (iter-30) and Morrison (iter-25) q_v donor clamps used
the naive ``qv_avail / max(qv_sink_total · dt, 1e-30)`` pattern.  When
the column is subsaturated and no ice nucleation is active,
``qv_sink_total = 0`` → ``q_v / 1e-30`` whose VJP is ``-q_v / 1e-60``,
which overflows fp32 → NaN gradients (even though ``min(1, huge) = 1``
kills the forward value).

Applied the double-where AD guard from the kessler q_c clamp to both
the Thompson and Morrison q_v clamps:

```
qv_sink_dt = qv_sink_total * dt_safe
qv_sink_active = qv_sink_dt > 0.0
qv_safe_sink_dt = jnp.where(qv_sink_active, qv_sink_dt, 1.0)
qv_scale = jnp.where(qv_sink_active,
                     jnp.minimum(1.0, qv_avail / qv_safe_sink_dt), 1.0)
```

Added `TestMorrison::test_differentiable_zero_sink_qv_clamp` that runs
`jax.grad` through Morrison on a subsat clear column and asserts the
gradient is finite.  Without the guard the test would NaN.

**Tests (post iter-31):** 61 / 61 microphysics + 7 / 7 differentiability
tests pass.

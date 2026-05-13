# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iterations 1-50 — Summary (compressed 2026-05-13 after iter-50)

### Constants & shared utilities
- `constants.p_atm_std`, `constants.R_universal` added.  Bulk-flux + stomata reference them.
- `convection/mass_flux._compute_column_geometry` delegates to `_shared.compute_layer_dz` /
  `compute_rho` with `q_v`; threaded through all 5 mass-flux schemes.
- Constant-hygiene audit (iter-12): all physical constants routed through `constants.py`.

### Conservation & monotonicity
- Sedimentation `dt`-positivity limiter + surface-flux return + `extra_sink` joint cap;
  threaded through Kessler / Morrison / Thompson / SB.
- Rain evaporation `dt` clamp.  Sundqvist autoconversion donor cap.
- Sea-ice transport: conservative monotone PPM with `lax.scan` subcycles.
- ITD remap: volume-conserving rescale; residual folded into thickness when `a > 1`.
- RRTMGP: `q_v ≤ 0.99` upper-clip for H₂O VMR denominator safety.
- DCA STANDARD MSE conservation (in-scheme latent heating via `delta_T_lh`).

### Donor clamps consolidated
- `donor_clamp_scale(q_avail, sink_total, dt, divisor_floor=1e-15)` in `_warm_rain.py` —
  fp32-VJP-safe; all 8 microphysics donor clamps route through it.
- Total-water conservation regression tests for all 4 schemes.

### CFL caps & stability
- Enhanced-diffusion convection (ocean): explicit branch delegates per-interface CFL cap.
- Ocean vertical mixing leafs: per-interface CFL cap.
- Ocean lateral mixing: opt-in `enforce_cfl`.
- EDMF mass flux: `M ≤ 0.5·ρ·dz/dt`.

### Coupler / feedback paths
- Sea-ice `_bulk_flux_dispatch` helper centralises MOST/COARE/Large-Yeager/simple_bulk.
- Ice → ocean feedback uses per-ice-area `dh/dt` for FW flux; computes
  `ocean_heat_extraction` and `ocean_stress_x/y`.
- Multi-cat lead freeze via `open_water_fraction` + `enable_lead_freeze` kwargs.
- Grey-surface emission `lw_up = ε·σ·T⁴ + (1−ε)·lw_down` for sea-ice / ocean / lake / land.
- Multilayer + slab land: dew/deposition gated on `has_snow=False`; fresh-snow albedo
  uses surviving-fresh-snow guard.

### Tracer mixing / ocean drag / BBL
- BBL drag `min(H_BBL, total_overlap)` in `ocean_tendency_common.bbl_drag_distributed`
  and `ocean_pe_latlon_cgrid._bbl_drag_for_face`.

### Streamfunction diagnostics
- `moc_streamfunction` and `barotropic_streamfunction` use grid attributes when present.

### Ocean bulk-flux + Emanuel AD-safety
- `bulk_formulas._saturation_specific_humidity` mixing-ratio → specific-humidity (~3 % bias).
- `Q_lw_up = ε σ T_s^4 + (1−ε)·LW_down` (~10 W/m² Q_net correction).
- `emanuel.py:204` AD-safe floor 1e-30 → 1e-15.

## Iterations 51-69 — Summary (compressed 2026-05-13 after iter-70)

### Real code fixes
- **iter-52 / 53 / 54** (lateral_mixing no-flux BC): Neumann-fill tracers via
  `fill_land_cells(T, mask, grid, n_passes=N)` BEFORE the harmonic / biharmonic operator;
  N = stencil reach (2 / 3).
- **iter-55** (Delta_min threading): `SeaIceConfig.Delta_min` routes through
  `evp_solver → evp_stress_update → delta_deformation`.
- **iter-56** (shortwave_penetration Inf/NaN on dry columns): `dz_actual > 0` guard +
  safe-denominator + output mask.
- **iter-57 / 58 / 59** (dry-column AD safety in vertical-diffusion leafs): KPP non-local
  + `vertical_diffusion` + `vertical_diffusion_variable_K` use `dz_safe = jnp.where(dz >
  0, dz, 1)` + substitution `field_safe = where(dz > 0, field, 0)` + `K_half = where(dry,
  0, K_half)` to scrub upstream NaN before arithmetic.
- **iter-62 / 63 / 64** (carbon conservation under R_auto > GPP): cascade C_lab → C_fol →
  C_root → C_wood when NPP_day < 0; net-available pool budget; `jnp.maximum(x, 0)` clamp
  replacing softplus.  rtol=1e-9 conservation.
- **iter-67** (frozen-lake q_surface): post-step `q_sfc_new` applies same liquid/ice
  saturation phase switch as pre-step (~14 % bias removed).

### Inspection-only iters
- **iter-51, 61, 65, 66, 68, 69**: multilayer_land, ocean_pe backends, Bechtold/Emanuel
  convection + coupler bulk_flux, sea-ice _thermo_single, thermo + surface_albedo,
  land utils + ML ozone + insolation — all verified clean.

## Iteration 83 — 2026-05-13

**Inspection iteration on Richards solver units + ice module
(no code changes).**

- **`richards.py`**: Picard iteration units verified — coeff =
  K_half / dz_if has units 1/s; rhs has 1/s; tridiag matrix has
  1/(m·s); dpsi has m.  Consistent.
- **Codex on ice/**: was killed after extensive investigation of
  dynamics, itd, state, sea_ice, coupler tile-fractions, driver
  component_factory; no actionable finding emerged.  Iter-45 had
  already verified EVP algebra, VP constitutive law, PPM
  transport, ITD remap and strain-rate FD denominators.

No new fixes needed.

## Iteration 82 — 2026-05-13

**Scan-only iteration: confirmed no other `jnp.where`-traces-both-
branches on static-shape gates (no code changes).**

After iter-81 fix, scanned `src/legoesm/` for similar patterns:
- `state.tracers.data[..., -1, 0]` indexing (only in iter-81's
  `extract_atm_to_surface_nh`; now uses Python `if`).
- `radiation/integration.py:689-707` already uses the Python-`if`
  pattern correctly:
  ```python
  n_tracers = state.tracers.data.shape[-1]
  if n_tracers > 0:
      q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
  ```

Also inspected `_warm_rain.py` (donor_clamp_scale, safe_pow,
saturation_adjustment, effective_Nc, autoconversion_sb, accretion,
self_collection_breakup, rain_evaporation), `_plume.py` (analytic
exponential decay forms for `M_u`, `T_u_ent`, `q_u_ent`,
`q_c_u_ent` for AD-safety), and `snow_budget.py` (energy-limited
melt + degree-day fallback).  All clean.

No new fixes needed.

## Iteration 81 — 2026-05-13

**Bug fix: `extract_atm_to_surface_nh` crashes for dry NH state.**

`coupler/surface_exchange.py:138` used `jnp.where(has_tracers,
state.tracers.data[..., -1, 0], 0.0)` to fall back when there
are no tracers.  But `jnp.where` traces BOTH branches, and the
`[..., -1, 0]` indexing crashes at trace time when the `n_tracers`
axis has shape 0 (dry NH simulations).  The hydrostatic version
(line 69-74) correctly uses a Python `if` based on the static
shape — extended that pattern to the NH version.

Codex review on coupler/ returned `verdict=approve, No new findings`
(false negative — the bug is real but only fires for dry NH state,
not currently in the test matrix).

**Tests (post iter-81):** 9 / 9 lake tests pass (smoke).  Existing
production setups all have q_v tracer so the bug was dormant.

## Iterations 71-79 — Summary (compressed 2026-05-13 after iter-80)

### Real code fixes
- **iter-73** (joint sed + in-column-sink cap, codex-confirmed): Morrison + Thompson
  sedimentation now passes `extra_sink=` for the in-column sinks competing with sed in
  the same explicit step — q_i (aggregation + melt_ice [+ rime_to_graupel_from_i]),
  q_s (melt_snow [+ rime_to_graupel_from_s]), q_g (melt_graupel).  Extends the iter-29
  q_r/evaporation pattern to all solid-hydrometeor sed paths.
- **iter-79** (codex stop-time: RRTMGP humidity guard bypassed by halo extrapolation):
  `rrtmgp.py:636-640` applied `clip(q_v, 0, 0.99)` BEFORE `_add_halos`, but the linear
  halo extrapolation could push q_v outside [0, 0.99] for steep boundary profiles
  (`q_v = [0.99, 0.0, ...]` → halo = 1.98 → singular `(1−q_v)` denominator).  Fix:
  clip AFTER `_add_halos`.  New regression test exercises the pathological profile.

### Inspection-only iters
- **iter-71**: YSU PBL + surface_layer + EDMF.  YSU/HB internal `L_v` hardcode noted
  as known limitation (tile-side flux is authoritative in coupled mode).
- **iter-72**: TKE + CLUBB-lite + pbl_height.  All differentiable with semi-implicit
  dissipation and clip-to-tke_min final guards.
- **iter-74**: ocean surface_forcing + bulk_flux (codex `verdict=approve`).
- **iter-75**: kessler/SB/Kuo/DCA convection (codex `verdict=approve`, truncated).
- **iter-76**: Tiedtke/Kain-Fritsch/Zhang-McFarlane mass-flux convection.  All 6 schemes
  share the column-conservative `_apply_mass_flux_kernel` scaffold.
- **iter-77**: ocean_tendency_common helpers (iterate_eos_and_pressure_anomaly,
  bbl_distributed_drag_face_column, implicit_bottom_drag_factor).
- **iter-78**: barotropic substep solvers (explicit + implicit Crank-Nicolson).
  Explicit's Hu_avg is exact; implicit's trapezoidal Hu_avg differs from the eta-update
  flux by O(θ²·dt²·g·∇²η).

### Codex review activity
- 5+ codex runs in this window; 2 returned actionable findings (iter-73 sed cap,
  iter-79 q_v halo) — both confirmed and fixed.  Many runs returned
  `verdict=approve`.

## Iterations 41-50 — Detail (compressed 2026-05-13)

- **iter-41**: compression milestone (iter-30..40 folded).
- **iter-42**: ocean bulk-flux q_sat unit-bug + grey-surface LW + Emanuel floor.
- **iter-43**: inspection-only (atmosphere radiation/turbulence).
- **iter-44**: slab_land fresh-snow albedo.
- **iter-45**: inspection-only (ice modules).
- **iter-46**: inspection-only (ocean lateral_mixing).
- **iter-47**: inspection-only (gravity_wave_drag).
- **iter-48**: inspection-only (clouds + Louis turbulence).
- **iter-49**: inspection-only (gray radiation + solar geometry).
- **iter-50**: compression + land inspection.

## Inspected & clean (as of iter-80)
- **Land**: `snow_budget`, `carbon/{carbon_cycle,stomata}`, `richards`, `soil_thermal`,
  `multilayer_land`, `slab_land`, `stomata_utils`, `surface_params`, `param_providers`.
- **Atmosphere turbulence**: `clubb_lite`, `holtslag_boville`, `louis`, `ysu`, `tke`,
  `vertical_diffusion`, `surface_layer`, `pbl_height`.
- **Atmosphere clouds**: `cloud_fraction`.
- **Atmosphere GWD**: `rayleigh`, `lindzen`, `mcfarlane`, `hines`, `integration`.
- **Atmosphere radiation**: `gray`, `solar`, `ozone_ml`, `integration`,
  `rrtmgp/rrtmgp` (post iter-79).
- **Atmosphere convection**: `bechtold`, `emanuel`, `dca`, `kuo`, `zhang_mcfarlane`,
  `tiedtke`, `kain_fritsch`, `mass_flux`.
- **Atmosphere microphysics**: `kessler`, `seifert_beheng`, `morrison`, `thompson`,
  `_warm_rain`, `output` (post iter-73).
- **Ocean physics**: `lateral_mixing/{harmonic,biharmonic,gm_redi,gm_redi_latlon_cgrid,
  gm_redi_mpas,backscatter,_gm_redi_common}` (post iter-52/53/54),
  `shortwave_penetration`, `mixing` (post iter-58/59), `bottom_drag/quadratic`,
  `vertical_mixing/{kpp,implicit_solver,mpas_integration}` (post iter-57),
  `convection/{enhanced_diffusion,plume}`, `surface_forcing/{prescribed,restoring,
  bulk_formulas,wind_profiles}`.
- **Ocean dynamics**: `ocean_tendency_common`, `barotropic_common`,
  `barotropic_latlon_cgrid` (explicit), `barotropic_implicit_latlon_cgrid` (with
  known limitation).
- **Coupler**: `surface_energy`, `bulk_flux`, `lake/two_layer_lake` (post iter-67).
- **Ice**: `rheology`, `dynamics`, `itd`, `transport`, `sea_ice` (post iter-55).
- **Shared**: `thermo`, `surface_albedo`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter.
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping.
- Threading `dt` into ocean `physics_fn` across 3 ocean PE backends + 3 model drivers.
- Visual / long-run validation of new ice → ocean feedbacks.
- σ-tensor rotation across cubed-sphere faces (O(dx) edge error in sea-ice EVP).
- Codex `adversarial-review` runtime: ~50/50 success/failure when running.  Continue
  alongside direct inspection.
- Bottom drag double-application in explicit barotropic substep path (F_slow already
  carries depth-mean drag; effective `2·r/H`).  Requires cross-cutting refactor.
- Implicit barotropic Hu_avg O(θ²·dt²·g·∇²η) inconsistency with eta-update flux.
  MOM6 / NEMO use iterative implicit methods for exact consistency.
- Pre-existing test failure `test_b_salt_sign_freshening_is_stabilizing` (harness
  fragility — `jnp.mean(None)` when `out.K_v` is None) — needs harness fix.

## Iter-84 — PDI hydraulic-conductivity wet-end smoothstep inverted
`src/legoesm/land/soil_hydraulics.py:322` (Iden et al. 2015 PDI capillary K with
max-pore-size constraint).  Cosine interpolation between `h_crit` and `h≈0` was
written `0.5*(1+cos(π·frac))`, giving `Kr_interp = Kr_crit` at frac=0 (wet end)
and `Kr_interp = 1` at frac=1 (h=h_crit) — the opposite of the intended boundary
conditions.  Diagnosis: at h=h_crit, Kr_interp must match the standard
capillary `Kr_c(h_crit) = Kr_crit` for continuity; at h≈0 it must approach 1 so
K → K_sat.  Loam defaults gave Kr_crit ≈ 0.342, so saturated K was effectively
clamped at 34 % of K_sat in the wet regime.  Fix replaces `(1 + cos)` with
`(1 − cos)` so the smoothstep runs 1 → Kr_crit as frac goes 0 → 1.

Quantitative check (loam defaults, JAX float64, CPU):
- `K(h = h_crit = 0.06 m) = 9.871e-7 m/s = K_sat * Kr_crit` (continuous match)
- `K(h = 1e-9 m) = 2.830e-6 m/s ≈ 0.979 * K_sat` (smoothly approaching K_sat)
- `K(h = 1e-12 m) = 2.890e-6 m/s = K_sat` (exact to machine precision)

Regression test `test_K_approaches_Ksat_at_saturation` walks h from h_crit down
to 1e-12 m, asserts monotone K and `K(1e-12) = K_sat` to rtol=1e-6.  Full
`tests/unit/test_land_ice_soil_hydraulics.py` (38 tests) green.

## Next iterations
Continue addressing further codex findings and direct-inspection sweeps until all
schemes are provably conservative, monotone, CFL-safe, and AD-safe across mixed
wet/dry grids.

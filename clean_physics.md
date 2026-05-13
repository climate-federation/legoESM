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
- Sedimentation `dt`-positivity limiter (`flux ≤ q·ρ·dz/dt`) + surface-flux return + `extra_sink`
  joint cap; threaded through Kessler/Morrison/Thompson/SB.
- Rain evaporation `dt` clamp.  Sundqvist autoconversion donor cap.
- Sea-ice transport: conservative monotone PPM (Colella–Woodward) with `lax.scan` subcycles.
- ITD remap: volume-conserving rescale; residual folded into thickness when `a > 1`.
- RRTMGP: `q_v ≤ 0.99` upper-clip for H₂O VMR denominator safety.
- DCA STANDARD MSE conservation (in-scheme latent heating via `delta_T_lh`).

### Donor clamps consolidated
- `donor_clamp_scale(q_avail, sink_total, dt, divisor_floor=1e-15)` in `_warm_rain.py` —
  fp32-VJP-safe (avoids `-q/sink²` overflow under legacy 1e-30 floor).
- All 8 microphysics donor clamps route through the shared helper.
- Total-water conservation regression tests for all 4 schemes
  (`|residual|/max(|precip|, 1e-10) < 1e-6` in fp32 and fp64).

### CFL caps & stability
- Enhanced-diffusion convection (ocean): explicit branch delegates to
  `vertical_diffusion_variable_K(dt=...)`.
- Ocean vertical mixing leafs (`mixing.py`): per-interface CFL cap.
- Ocean lateral mixing: opt-in `enforce_cfl` on harmonic + biharmonic.
- EDMF mass flux (turbulence): `M ≤ 0.5·ρ·dz/dt`.

### Coupler / feedback paths
- Sea-ice `_bulk_flux_dispatch` helper centralises MOST/COARE/Large-Yeager/simple_bulk.
- `_build_response` ice → ocean feedback uses per-ice-area `dh/dt` for FW flux (CICE
  V=h·A refactor deferred); computes `ocean_heat_extraction` and `ocean_stress_x/y`.
- Multi-cat lead freeze via `open_water_fraction` + `enable_lead_freeze` kwargs.
- Sea-ice / ocean / lake / land radiative emission: `lw_up = ε·σ·T⁴ + (1-ε)·lw_down`
  (grey surface). iter-13 sea-ice, iter-42 ocean.
- Multilayer + slab land: dew/deposition routing on `has_snow=False`; fresh-snow albedo
  uses surviving-fresh-snow guard (iter-44 / iter-68).

### Tracer mixing / ocean drag / BBL
- BBL drag `min(H_BBL, total_overlap)` in `ocean_tendency_common.bbl_drag_distributed`
  and `ocean_pe_latlon_cgrid._bbl_drag_for_face` (shelf/coastal correctness).

### Streamfunction diagnostics
- `moc_streamfunction` and `barotropic_streamfunction` use grid attributes when present.

### Ocean bulk-flux + Emanuel AD-safety
- `bulk_formulas._saturation_specific_humidity` mixing-ratio → specific-humidity (~3% bias).
- `Q_lw_up = ε σ T_s^4 + (1−ε)·LW_down` (~10 W/m² Q_net correction).
- `emanuel.py:204` AD-safe floor 1e-30 → 1e-15.

## Iterations 51-59 — Detail (compressed 2026-05-13 after iter-60)

- **iter-51**: inspection-only (multilayer_land — `has_snow` phase decision, G_surface
  energy budget closure, sublimation cap with sign-aware deposition, soil water-budget
  partition + dew routing, root sink normalisation all verified).
- **iter-52**: **resolved deferred item** — no-flux BC for harmonic/biharmonic tracer
  Laplacian via `fill_land_cells(T, mask, grid)` Neumann fill BEFORE the operator.
  GM/Redi already used `_neumann_fill_cgrid` / `_voronoi_neumann_fill` on its paths.
- **iter-53**: **codex iter-52 stop-time fix** — reduced `n_passes` to operator stencil
  radius (1 for harmonic, 2 for biharmonic) to limit thin-barrier bridging.
- **iter-54**: **codex iter-53 stop-time fix** — corrected stencil reach: harmonic
  `div(grad)` samples `{j-2, j, j+2}` (reach 2) and biharmonic `compact(·)∘div(grad)(·)`
  samples `f[j±1,2,3]` (reach 3).  `n_passes` updated to 2 / 3.
- **iter-55**: **resolved deferred item** — `SeaIceConfig.Delta_min` now threads through
  `evp_solver → evp_stress_update → delta_deformation`.  New regression catches the
  silent-ignore.
- **iter-56**: bug fix — `shortwave_penetration_tendency` produced Inf on dry columns
  (`dz_actual = 0`).  Added `dz_actual > 0` guard with safe denominator + output mask;
  regression test `test_dry_column_gives_zero_finite_tendency`.
- **iter-57**: AD-safety — KPP non-local `/ dz_actual` divisor.  `dz_safe = jnp.where(
  dz_actual > 0, dz_actual, 1)` + tightened output mask (`is_unstable_col & dz_actual > 0`).
- **iter-58**: **codex iter-57 stop-time fix** — same `dz_safe` pattern applied to BOTH
  `vertical_diffusion` and `vertical_diffusion_variable_K` in `ocean/physics/mixing.py`
  (called by KPP, Richardson, constant-K, enhanced-diffusion, every PE-backend baseline
  `K_v` diffusion).
- **iter-59**: **codex iter-58 stop-time fix** — variable-K NaN gradient leak.
  Upstream KPP/Richardson can produce NaN `K_half` on dry columns; `K_half * df_dz`
  propagates NaN forward AND backward.  Fix: `K_half = jnp.where(dry_iface, 0, K_half)`
  and `field_safe = jnp.where(dz > 0, field, 0)` SUBSTITUTE values before arithmetic;
  `jnp.where` lets the backward pass select the constant-0 branch.  Same `field_safe`
  applied to `vertical_diffusion` for defensive symmetry.

## Iterations 41-50 — Detail (compressed 2026-05-13)

- **iter-41**: compression milestone (iter-30..40 folded).
- **iter-42**: ocean bulk-flux q_sat unit-bug + grey-surface LW + Emanuel floor.
- **iter-43**: inspection-only (atmosphere radiation/turbulence — divisor floors
  ADDITIVE not max(), don't trigger -q/sink² VJP overflow).
- **iter-44**: slab_land fresh-snow albedo (apply codex iter-43 finding).
- **iter-45**: inspection-only (ice modules — EVP algebra, VP constitutive,
  PPM transport, ITD remap all verified).
- **iter-46**: inspection-only (ocean lateral_mixing — DM95 taper, slope-tensor
  algebra, triad cancellation, backscatter energy budget all verified).
- **iter-47**: inspection-only (gravity_wave_drag — Rayleigh sponge, Lindzen/
  McFarlane orographic, Hines WKB-step ρ ratio, spectral PE vor/div projection
  all verified).
- **iter-48**: inspection-only (clouds + Louis turbulence — surface-flux unit
  chain verified end-to-end; cloud_fraction Sundqvist/Xu-Randall bounded).
- **iter-49**: inspection-only (gray radiation + solar geometry — polar
  day/night via clip(cos_hs); LW two-stream BC consistent with grey-surface fix).
- **iter-50**: compression milestone + land inspection (snow_budget energy-limited
  melt, soil_thermal Johansen K_e, Richards Picard with Celia 1990 mass-conservative
  form all verified).

## Inspected & clean (no fix needed in current cycle)
- `land/{snow_budget,carbon/carbon_cycle,richards,soil_thermal,multilayer_land,slab_land}.py`.
- `atmosphere/physics/turbulence/{clubb_lite,holtslag_boville,louis,ysu,
  vertical_diffusion}.py`.
- `atmosphere/physics/clouds/cloud_fraction.py`.
- `atmosphere/physics/gravity_wave_drag/{rayleigh,lindzen,mcfarlane,hines,
  integration}.py`.
- `atmosphere/physics/radiation/{gray,solar}.py`.
- `ocean/physics/lateral_mixing/{harmonic,biharmonic,gm_redi,gm_redi_latlon_cgrid,
  gm_redi_mpas,backscatter,_gm_redi_common}.py`.
- `ocean/physics/shortwave_penetration.py`, `ocean/physics/bottom_drag/quadratic.py`.
- `ocean/physics/mixing.py` (post iter-58/59 dry-column NaN guards).
- `ice/{rheology,dynamics,itd,transport,sea_ice}.py`.
- `atmosphere/physics/convection/{dca,kuo,zhang_mcfarlane}.py`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter.
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping (iter-25 attempt
  reverted; slab path evolves h and conc semi-independently).
- Threading `dt` into ocean `physics_fn` across 3 ocean PE backends + 3 model drivers.
- Visual / long-run validation of new ice → ocean feedbacks.
- σ-tensor rotation across cubed-sphere faces (O(dx) edge error in sea-ice EVP).
- Codex `adversarial-review` runtime: 4+ consecutive iters lost to exit 144.  Retry
  later or continue direct inspection.
- Pre-existing test failure `test_b_salt_sign_freshening_is_stabilizing` (test
  fragility: `jnp.mean(None)` when `out.K_v` is None) — needs harness fix, not
  physics fix.

## Next iterations
Continue addressing further codex adversarial-review findings and direct-inspection
sweeps through remaining modules until all schemes are provably conservative, monotone,
CFL-safe, and AD-safe across mixed wet/dry grids.

# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iteration 73 — 2026-05-13

**Bug fix: Morrison + Thompson ice/snow/graupel sedimentation +
in-column-sink joint cap missing.**

Codex review confirmed and direct inspection found: the iter-29
joint-cap pattern (sedimentation passes `extra_sink` for the
in-column sink so the combined per-step removal can't exceed
available mass) was applied to q_r/evaporation but NOT to
q_i/q_s/q_g/in-column-melt-or-rime sinks.

The donor clamps already cap (aggregation + melt_ice) ≤ q_i/dt,
but the separate `sedimentation_tendency` call independently caps
sed_i at q_i/dt, so the combined removal `(clamped sinks + sed) ·
dt` can reach **2·q_i**, driving the pool negative when both
processes are simultaneously active (e.g. melting graupel +
strong sedimentation in a warm layer).

Fix: pass `extra_sink=` to every solid-hydrometeor sedimentation
call so the per-layer sed flux is capped against the NET pool
after natural drains:
- Morrison: `q_i` → `aggregation + melt_ice`; `q_s` → `melt_snow`.
- Thompson: `q_i` → `aggregation + melt_ice + rime_to_graupel_from_i`;
  `q_s` → `melt_snow + rime_to_graupel_from_s`; `q_g` → `melt_graupel`.

**Tests (post iter-73):** 68 / 68 atmosphere microphysics tests pass
(Kessler, Morrison, Thompson, Seifert-Beheng + multi-step
stability).

## Iteration 72 — 2026-05-13

**Inspection iteration on TKE / CLUBB-lite / `pbl_height` (no code changes).**

- **`pbl_height`**: bulk-Ri with surface-difference dV² (Vogelezang &
  Holtslag 1996 form, not absolute-wind-speed); two methods —
  smooth sigmoid-weighted (`w_pbl = sigma·(1−sigma)` peaks at the
  Ri_crit crossing) and `interp` (linear interpolation + softmin).
  Both differentiable, clipped to [h_min, h_max].
- **`tke.py`** (Mellor-Yamada level 2.5): sqrt floor at `tke_min`
  via `tke = max(tke, tke_min)` at line 93; semi-implicit dissipation
  `(tke_diffused + dt·prod)/(1 + dt·diss)` linearises so even strong
  dissipation can't drive tke negative; `tke_new = max(tke_new,
  tke_min)` final clamp.  Buoyancy sign correct (sink for stable,
  source for unstable).
- **`clubb_lite.py`** (Golaz / Larson PDF closure): wp2 carried via
  TKE slot; cloud-fraction PDF block confirmed-dead and removed in
  iter-172 audit; restoration instructions in source comments for
  future CLUBB unified scheme.
- **YSU / HB / Louis / CLUBB / TKE all share `surface_layer.
  compute_surface_fluxes`** with hardcoded L_v (known limitation —
  internal atmospheric path, NOT the tile-side coupler flux).

No new fixes needed.  All turbulence schemes are mature.

## Iteration 71 — 2026-05-13

**Inspection iteration on YSU PBL + surface_layer + EDMF
(no code changes).**

- **`ysu.py`**: PBL-height bulk-Ri integrator with `sigmoid(sharpness ·
  (Ri_crit − Ri_bulk))` weighting; `w_pbl = sigma · (1−sigma)` peaks at
  the Ri_crit crossing.  Both pure-stable and pure-unstable columns
  collapse to `w → 0` everywhere and `+1e-20` floor regularises (h_pbl
  → mean(z), clipped to 100m).  Known limitation, not a bug.
- **K-profile / local-Ri blend**: `Km_profile = κ_vk · u* · z · (1 − z/h)²`
  inside PBL, Louis-style local-Ri Km above PBL, Gaussian
  entrainment envelope `K_ent = c_ent · w* · h · exp(−((z−h)/(0.3h))²)`
  centred at h_pbl.  Smooth blend via `sigmoid(10·(z/h − 1))`.
- **`implicit_vertical_diffusion_theta`**: T → θ via `(p/p_ref)^κ`;
  surface flux converted to θ via `F_θ = F_T / exner_sfc`; diffused
  in θ-space; converted back to T.  Dry adiabat (dθ/dz = 0) is neutral.
- **`surface_layer.compute_surface_fluxes`** (atmospheric side):
  hard-coded `L_v` for `lhflx`.  Over snow/ice tiles, this is ~13%
  too low (`L_s/L_v ≈ 1.13`).  In coupled-mode, the tile-side
  bulk-flux (with `L_eff` switch) is what's reported to the coupler;
  the YSU/HB internal `lhflx` is only used as the boundary condition
  for q_v implicit diffusion.  Known limitation in non-aquaplanet
  setups; out of scope for this iter.
- **`edmf_turbulence`**: `M_max = 0.5 · ρ · dz / dt` cap (per iter-12
  fix).  Column conservation noted as approximate in source comments.

No new fixes needed.  YSU/HB internal `L_v` hardcode is a known
limitation tracked alongside the coupled-mode surface-flux flow.

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
- **iter-52 / 53 / 54** (lateral_mixing no-flux BC, 3 iters):
  Neumann-fill tracers via `fill_land_cells(T, mask, grid, n_passes=N)` BEFORE
  the harmonic / biharmonic operator; N = stencil reach (2 / 3).
- **iter-55** (Delta_min threading): `SeaIceConfig.Delta_min` now routes
  through `evp_solver → evp_stress_update → delta_deformation`.
- **iter-56** (shortwave_penetration Inf/NaN on dry columns): `dz_actual >
  0` guard + safe-denominator + output mask in
  `shortwave_penetration_tendency`.
- **iter-57 / 58 / 59** (dry-column AD safety in vertical-diffusion leafs,
  3 iters): KPP non-local + `vertical_diffusion` + `vertical_diffusion_
  variable_K` all switched to `dz_safe = jnp.where(dz > 0, dz, 1)` for
  the divisor and `field_safe = where(dz > 0, field, 0)` + `K_half =
  where(dry_iface, 0, K_half)` SUBSTITUTIONS to scrub upstream NaN
  from KPP / Richardson before arithmetic.  Backward (AD) pass now
  clean on mixed wet/dry grids.
- **iter-62 / 63 / 64** (carbon conservation under R_auto > GPP, 3 iters):
  When NPP_day < 0, cascade the deficit `C_lab → C_fol → C_root →
  C_wood` (standard CASA/DALEC).  Each draw capped at the **net-
  available** pool budget after natural turnover.  Hard `jnp.maximum`
  clamp replacing softplus (avoiding the `_alpha · log(2)` smoothing
  bias).  Carbon conservation now at near-machine precision
  (`rtol=1e-9, atol=1e-9` regression).
- **iter-67** (frozen-lake q_surface): post-step `q_sfc_new` now applies
  the same liquid / ice saturation phase switch used pre-step.  ~14 %
  bias removed for `T_epi ≤ T_freeze` lakes.

### Inspection-only iters (no code changes)
- **iter-51**: `land/multilayer_land.py` — has_snow phase, energy
  budget closure, sublimation cap, soil water-budget partition, root
  sink normalisation all verified.
- **iter-61**: scan complete — `ocean_pe_*` backends use additive or
  `max()`-bounded floors; no additional `dz` divide-by-zero risks.
- **iter-65**: Bechtold / Emanuel convection + coupler bulk-flux
  helpers verified; surface_radiation_fluxes grey-surface BC
  consistent.
- **iter-66**: sea-ice `_thermo_single` sign conventions and
  conservation verified.
- **iter-68**: `thermo.py` + `surface_albedo.py` — Tetens / C-C
  formulas, smooth-cap softplus, snow / ice / ocean albedo all
  verified.
- **iter-69**: `land/stomata_utils`, `land/param_providers`,
  `radiation/ozone_ml.py`, `_compute_insolation` all verified.

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

## Inspected & clean (as of iter-70)
- `land/{snow_budget,carbon/{carbon_cycle,stomata},richards,soil_thermal,
  multilayer_land,slab_land,stomata_utils,surface_params,param_providers}.py`.
- `atmosphere/physics/turbulence/{clubb_lite,holtslag_boville,louis,ysu,
  vertical_diffusion}.py`.
- `atmosphere/physics/clouds/cloud_fraction.py`.
- `atmosphere/physics/gravity_wave_drag/{rayleigh,lindzen,mcfarlane,hines,
  integration}.py`.
- `atmosphere/physics/radiation/{gray,solar,ozone_ml,integration}.py`.
- `atmosphere/physics/convection/{bechtold,emanuel,dca,kuo,zhang_mcfarlane,
  mass_flux}.py`.
- `ocean/physics/lateral_mixing/{harmonic,biharmonic,gm_redi,gm_redi_latlon_cgrid,
  gm_redi_mpas,backscatter,_gm_redi_common}.py` (post iter-52/53/54).
- `ocean/physics/{shortwave_penetration,mixing,bottom_drag/quadratic}.py`
  (post iter-56/58/59).
- `ocean/physics/vertical_mixing/{kpp,implicit_solver,mpas_integration}.py`
  (post iter-57).
- `coupler/{surface_energy,bulk_flux,lake/two_layer_lake}.py` (post iter-67).
- `ice/{rheology,dynamics,itd,transport,sea_ice}.py` (post iter-55).
- `thermo.py`, `surface_albedo.py`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter.
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping (iter-25 attempt
  reverted; slab path evolves h and conc semi-independently).
- Threading `dt` into ocean `physics_fn` across 3 ocean PE backends + 3 model drivers.
- Visual / long-run validation of new ice → ocean feedbacks.
- σ-tensor rotation across cubed-sphere faces (O(dx) edge error in sea-ice EVP).
- Codex `adversarial-review` runtime: many consecutive iters lost to exit 144.
  Continue direct inspection until runtime stabilises.
- Pre-existing test failure `test_b_salt_sign_freshening_is_stabilizing` (test
  fragility: `jnp.mean(None)` when `out.K_v` is None) — harness fix, not physics.

## Next iterations
Continue addressing further codex findings and direct-inspection sweeps through
remaining modules until all schemes are provably conservative, monotone, CFL-safe,
and AD-safe across mixed wet/dry grids.

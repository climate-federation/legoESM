# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`.  Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iterations 1-50 — Summary (compressed 2026-05-13 after iter-50)

### Constants & shared utilities
- `constants.p_atm_std`, `constants.R_universal`.  Bulk-flux + stomata reference them.
- `convection/mass_flux._compute_column_geometry` delegates to `_shared.compute_layer_dz`
  / `compute_rho` with `q_v`; threaded through all 5 mass-flux schemes.
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

### CFL caps & stability
- Enhanced-diffusion convection + ocean vertical mixing leafs: per-interface CFL cap.
- Ocean lateral mixing: opt-in `enforce_cfl`.  EDMF mass flux: `M ≤ 0.5·ρ·dz/dt`.

### Coupler / feedback paths
- `_bulk_flux_dispatch` helper centralises MOST/COARE/Large-Yeager/simple_bulk.
- Ice → ocean: per-ice-area `dh/dt` FW flux + `ocean_heat_extraction` + `ocean_stress_x/y`.
- Multi-cat lead freeze via `open_water_fraction` + `enable_lead_freeze` kwargs.
- Grey-surface emission `lw_up = ε·σ·T⁴ + (1−ε)·lw_down` everywhere.
- Multilayer + slab land: dew/deposition gated on `has_snow=False`; fresh-snow albedo
  uses surviving-fresh-snow guard.

### Ocean drag / streamfunction / bulk-flux
- BBL drag `min(H_BBL, total_overlap)` in `ocean_tendency_common` + `ocean_pe_latlon_cgrid`.
- MOC / barotropic streamfunctions use grid attributes when present.
- `bulk_formulas._saturation_specific_humidity` mixing-ratio → specific-humidity (~3 % bias).
- `Q_lw_up = ε σ T_s^4 + (1−ε)·LW_down` (~10 W/m² Q_net correction).
- `emanuel.py:204` AD-safe floor 1e-30 → 1e-15.

## Iterations 51-69 — Summary (compressed 2026-05-13 after iter-70)
- **iter-52/53/54** (lateral_mixing no-flux BC): Neumann-fill tracers via
  `fill_land_cells(T, mask, grid, n_passes=N)` before harmonic / biharmonic; N=stencil reach.
- **iter-55** (Delta_min): `SeaIceConfig.Delta_min` → `evp_solver` → `evp_stress_update` →
  `delta_deformation`.
- **iter-56** (shortwave_penetration dry-column Inf/NaN): `dz_actual > 0` guard +
  safe-denominator + output mask.
- **iter-57/58/59** (dry-column AD safety in vertical-diffusion leafs): KPP non-local +
  `vertical_diffusion` + `vertical_diffusion_variable_K` use `dz_safe`/`field_safe`/`K_half`
  substitution before arithmetic.
- **iter-62/63/64** (carbon conservation under R_auto > GPP): cascade C_lab → C_fol →
  C_root → C_wood when NPP_day < 0; `jnp.maximum(x, 0)` clamp replacing softplus.
  rtol=1e-9 conservation.
- **iter-67** (frozen-lake q_surface): post-step `q_sfc_new` uses same liquid/ice
  saturation phase switch as pre-step (~14 % bias removed).
- **Inspection-only**: iter-51, 61, 65, 66, 68, 69.

## Iterations 71-79 — Summary (compressed 2026-05-13 after iter-80)
- **iter-73** (codex-confirmed joint sed + in-column sink cap): Morrison + Thompson
  sedimentation now passes `extra_sink=` for q_i (aggregation + melt_ice [+ rime_to_graupel
  _from_i]), q_s (melt_snow [+ rime_to_graupel_from_s]), q_g (melt_graupel).
- **iter-79** (codex stop-time RRTMGP humidity guard bypass): clip `q_v ∈ [0, 0.99]` AFTER
  `_add_halos`, not before; halo extrapolation could otherwise push q_v outside the bounds
  and produce a singular `(1 − q_v)` denominator.
- **iter-81** (dry-NH coupler crash): `extract_atm_to_surface_nh` switched from
  `jnp.where(has_tracers, state.tracers.data[..., -1, 0], 0.0)` (traces both branches +
  crashes at trace time on shape-0 axis) to a Python `if` on the static `n_tracers` shape.
- **Inspection-only**: iter-71, 72, 74-78, 82-83 (~12 modules).

## Iterations 84-89 — Summary (compressed 2026-05-13 after iter-90)
- **iter-84** (PDI hydraulic K wet-end smoothstep inverted, `land/soil_hydraulics.py:322`):
  cosine interpolation between `h_crit` and `h≈0` ran 1 → Kr_crit *backwards*; loam
  defaults clamped saturated K at ~34 % of K_sat in the wet regime.  Fix replaces
  `(1+cos)` with `(1−cos)`; 38 soil-hydraulics tests green.
- **iter-85a/b/c** (Thomas tridiag denom-floor, `land/tridiag.py:42-55`): old
  `jnp.sign(denom)*_tiny + _tiny` collapses to **exactly 0** for tiny negatives on a
  non-FTZ backend; masked by XLA FTZ on our build, but platform-dependent.  Fix:
  branch-based `sign = where(denom >= 0, 1, -1); denom = where(|denom|<_tiny,
  sign·_tiny, denom)` — guarantees non-zero floor on any platform.  208 land+soil
  tests green.
- **iter-86/87** (free-drift ice velocity dimensionally inconsistent, 300× too small):
  `ice/dynamics.py:105` + `ice/sea_ice.py:184-187` (slab path inline).  Old
  `u_ice = drag_ocean·U_w + (drag_atm·ρ_air/ρ_ice)·U_a` treats dimensionless drag
  coefficients as velocity-mapping ratios and uses ρ_ice instead of ρ_oc.  Fix:
  steady-state Zubov balance `u_i = U_w + α·(U_a − U_w)`, `α = sqrt(ρ_a·C_ai/(ρ_oc·C_oi))
  ≈ 0.017`.  Signature `rho_ice → rho_ocean`.  117 tests green.
- **iter-88** (EDMF updraft dimensionally inconsistent,
  `atmosphere/physics/turbulence/edmf.py`): old `dw_dz = buoy - ε·w_u` mixed
  [m/s²] and [1/s]; replaced with squared form `d(w²)/dz = 2(B - ε·w²)` with AD-safe
  `sqrt(max(w², 1e-20))` floor + `where(w²>0, sqrt, 0)` gate.  101 tests green.
- **iter-89** (convection sweep + dead-code cleanup): removed dead `e_sat =
  saturation_vapor_pressure(T_parcel)` in `compute_lcl`; documented plume `B_u = T_u −
  T_e` dry-T limitation (~1 K virtual correction) as deferred.

## Iterations 90-99 — Summary (compressed 2026-05-13 after iter-100)

### Real code fixes
- **iter-92** (solar `daily_mean_insolation` / `daylight_fraction` NaN gradient at polar
  day / night, `atmosphere/physics/radiation/solar.py`): old `clip(cos_hs, -1, 1)` +
  `arccos(cos_hs)` produces a finite forward value but the `arccos` derivative is
  ±∞ at the boundary; combined with `clip`'s zero subgradient, JAX emits NaN gradients
  for every column inside the polar cap.  Fix: clip to `[-1 + ε, 1 − ε]` with ε=1e-7
  so `|d arccos|≤ 1/sqrt(2ε) ≈ 2236`; forward bias on Q ≤ 4×10⁻⁵ W/m² at boundary.
  72-test radiation suite + new `test_daily_mean_insolation_polar_ad_safe` green.
- **iter-97/99** (SB autoconversion onset sigmoid scale,
  `microphysics/_warm_rain.py::autoconversion_sb`): old `sigmoid(sharpness · (x_c −
  x_star))` with `x_c, x_star ~ O(1e-10) kg` gives `sigmoid(O(1e-8)) ≈ 0.5` always —
  threshold effectively disabled.  Fix: normalise to `sigmoid(sharpness · (x_c/x_star −
  1))` and add a separate `autoconversion_sharpness: float = 10.0` to SB / Morrison /
  Thompson configs (reusing `saturation_sharpness=100` is 100× too steep on the
  normalised dimensionless argument).  Threaded through `seifert_beheng.py`,
  `morrison.py`, `thompson.py`.  Maritime stratocumulus now correctly has near-zero SB
  autoconversion (canonical SB-2001 threshold).  iter-98 attempted the same
  normalisation on breakup but `breakup_sharpness=1e4` is in `[1/m]` units and the
  un-normalised form was already correctly tuned — iter-99 reverted the breakup change.
  115 microphysics tests green.

### Inspection-only iters (no code changes)
- **iter-91**: GWD (`rayleigh`, `lindzen`, `mcfarlane`, `hines`) + sea-ice
  `_thermo_single`.  All four GWD stress formulae dimensionally [Pa]; mountain-wave
  drag decelerates upper flow in surface-wind direction; sea-ice skin_cap = ½·ρ·c·h
  matches half-slab convention.  One minor finding: `surface_radiation_fluxes` uses
  `config.emissivity_ice` for all cells (~ε_ice vs ε_ocean bias of 2 %, masked by
  conc-weighting in production) — listed under *Deferred*.
- **iter-93**: ocean bottom drag (linear + quadratic) + ozone_ml ridge + gray
  radiation + radiation daily-mean integration path.  Quadratic drag explicit-CFL
  timescale ~46 days (safe); ozone einsum `"czk,ck->zk"` correctly contracts the
  feature axis; gray heating `(g/c_p)·dF_net_up/dp` is canonical Frierson form;
  iter-92 cos_hs clip verified not to break downstream cos(SZA) derivation.
- **iter-94**: land carbon (`stomata`, `carbon_cycle`) + `snow_budget`.  Farquhar
  electron-transport discriminant stays positive for any 0 < θ < 1; Ball-Berry /
  Medlyn / Jarvis use correct specific-humidity-based `e_air`; iter-62/63/64 cascade
  + `_soft_pos = jnp.maximum(x, 0)` carbon conservation held; `_effective_rate(r, dt)`
  exact exponential decay; energy-limited snow melt with degree-day fallback.
- **iter-95**: ocean vertical mixing (`constant`, `richardson`, `k_profiles`) +
  convection (`enhanced_diffusion`, `plume`).  `enhanced_diffusion` smooth sigmoid +
  runtime-dt CFL cap; `richardson_vertical_mixing` POP/E3SM-Omega Pr-Ri scaling
  (iter-47 fix held); `plume_convection` AD-safe `expm1` entrainment + column
  conservation; buoyancy flux signs verified (brine destabilizes, FW stabilizes).
- **iter-96**: KPP end-to-end (467 LOC).  `_boundary_layer_depth` sigmoid Ri_b
  crossing + `h_bl_prev` decoupling; iter-168 bug 1 `copysign(eps, B_f)` stability
  preservation; three-branch Monin-Obukhov w_s; iter-1 #6 K_bg/A_bg separated for
  tracer/momentum; iter-1 #1 LMD94 Eq. 19 non-local flux column-conservative
  (telescoping); iter-168 bug 2 `Q_sfc_T` plumbing held.

### Codex stop-time review activity
- Caught two real follow-ups: (1) iter-87 slab-path miss after iter-86; (2)
  iter-97/98 parameter units vs defaults — `breakup_sharpness=1e4` was actually in
  `[1/m]` and correctly tuned (iter-98 reverted), but autoconversion was reusing
  `saturation_sharpness` calibrated for kg/kg-scale (iter-99 added separate field).

## Inspected & clean (as of iter-100)
- **Land**: `snow_budget`, `carbon/{carbon_cycle,stomata}`, `richards`, `soil_thermal`,
  `multilayer_land`, `slab_land`, `stomata_utils`, `surface_params`, `param_providers`,
  `tridiag` (post iter-85), `soil_hydraulics` (post iter-84).
- **Atmosphere turbulence**: `clubb_lite`, `holtslag_boville`, `louis`, `ysu`, `tke`,
  `vertical_diffusion`, `surface_layer`, `pbl_height`, `edmf` (post iter-88).
- **Atmosphere clouds**: `cloud_fraction`.
- **Atmosphere GWD**: `rayleigh`, `lindzen`, `mcfarlane`, `hines`, `integration` (post
  iter-91).
- **Atmosphere radiation**: `gray`, `solar` (post iter-92), `ozone_ml`, `integration`,
  `rrtmgp/rrtmgp` (post iter-79).
- **Atmosphere convection**: `bechtold`, `emanuel`, `dca`, `kuo`, `zhang_mcfarlane`,
  `tiedtke`, `kain_fritsch`, `mass_flux`, `_plume` (post iter-89).
- **Atmosphere microphysics**: `kessler`, `seifert_beheng`, `morrison`, `thompson`,
  `_warm_rain` (post iter-73, iter-97/99), `output`.
- **Ocean physics**: `lateral_mixing/{harmonic,biharmonic,gm_redi,gm_redi_latlon_cgrid,
  gm_redi_mpas,backscatter,_gm_redi_common}` (post iter-52/53/54, iter-90),
  `shortwave_penetration`, `mixing` (post iter-58/59), `bottom_drag/{linear,quadratic}`
  (post iter-93), `vertical_mixing/{constant,kpp,implicit_solver,mpas_integration,
  richardson,k_profiles}` (post iter-57, iter-95, iter-96), `convection/{enhanced_diffusion,
  plume}` (post iter-95), `surface_forcing/{prescribed,restoring,bulk_formulas,
  wind_profiles}`.
- **Ocean dynamics**: `ocean_tendency_common`, `barotropic_common`,
  `barotropic_latlon_cgrid` (explicit), `barotropic_implicit_latlon_cgrid` (with
  known limitation).
- **Coupler**: `surface_energy`, `bulk_flux`, `surface_exchange` (post iter-81),
  `lake/two_layer_lake` (post iter-67).
- **Ice**: `rheology`, `dynamics` (post iter-86), `itd`, `transport`, `sea_ice` (post
  iter-55, iter-87, iter-91).
- **Shared**: `thermo`, `surface_albedo`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter.
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping.
- Threading `dt` into ocean `physics_fn` across 3 ocean PE backends + 3 model drivers.
- Visual / long-run validation of new ice → ocean feedbacks.
- σ-tensor rotation across cubed-sphere faces (O(dx) edge error in sea-ice EVP).
- Bottom drag double-application in explicit barotropic substep path (F_slow already
  carries depth-mean drag; effective `2·r/H`).  Requires cross-cutting refactor.
- Implicit barotropic Hu_avg O(θ²·dt²·g·∇²η) inconsistency with eta-update flux.
- Pre-existing test failure `test_b_salt_sign_freshening_is_stabilizing` (harness
  fragility — `jnp.mean(None)` when `out.K_v` is None).
- Pre-existing test failure `TestEVPSanity::test_zero_velocity_isotropic_stress`
  (EVP doesn't relax to -P/2 in 10 subcycles at dt=3600 s; rtol 15 % too tight;
  verified on HEAD~1).
- **Plume buoyancy uses dry T, not virtual** (`convection/_plume.py:653`; affects
  `compute_lfc_lnb` too).  ~1 K virtual correction comparable to plume-alive
  threshold.  `compute_cape` supports virtual-T when q_v threaded; plume integrator
  does not yet.  Scheme-wide recalibration — defer.
- **Sea-ice `_thermo_single` emissivity** (`ice/sea_ice.py:614-617`): `ε_ice` used for
  all cells regardless of `ice_mask`.  Masked by conc-weighting in production.
- **SB autoconversion τ-based Φ_au switch**: the simplified x_c switch (iter-97/99)
  is canonical but suppresses drizzle in maritime conditions (high N_c).  Proper SB-
  2001 / Stevens-2007 implementation would add a τ-based `Φ_au(q_r/(q_c+q_r))` switch.
- Codex `adversarial-review` runtime ~50/50 success/failure.  Continue alongside
  direct inspection.

## Iter-101 — ocean surface forcing + lake + bulk-flux sign-convention sweep
Audited `ocean/physics/surface_forcing/{restoring,prescribed,wind_profiles,
bulk_formulas,integration}.py`, `coupler/lake/two_layer_lake.py`, and the
`coupler/bulk_flux.compute_most_fluxes` + `simple_bulk_fluxes` callers.

Verified clean:
- **Restoring**: `dT/dt = -(T − T_star)/τ` standard relaxation; surface-layer
  only via pad-with-zero pattern.
- **Wind profiles**: 9 profiles validated — `cosine_latitude`,
  `single_gyre`, `double_gyre{,_sin2,_tapered}`, `channel_sine`,
  `global_wind` (Nikurashin–Vallis 3-belt polynomial in sin²(φ)·cos(φ)
  verified at φ ∈ {0, 30°, 50°, 70°}), `two_belt` (Gaussian).
  Tropical-wind Gaussian taper has docstring inaccuracy ("0 at ±σ" —
  actually 0.607 at σ; comment cosmetic only, math correct).
- **Bulk-formula ocean forcing**: `Q_lw_up = εσT⁴ + (1−ε)·LW_down`
  (iter-13/41 grey-surface form held); constant-coefficient branch
  uses `|U_a|` for `Q_sh`, `Q_lh` (heat exchange rate independent of
  wind direction) and signed `U_a` for `τ_x` (directional stress);
  `_saturation_specific_humidity` mixing-ratio→specific conversion
  (iter-41 ~3 % bias fix held); MOST-path `τ_x = -τ_x` flip converts
  atmospheric retarding convention to ocean accelerating convention.
- **Lake two-layer**: phase-aware `q_sfc` switch (iter-67); convective
  overturn fires on freshwater density-inversion (ρ depends on
  `(T − T_max)²` so cold ≪ T_max water can be less dense than warmer
  water near T_max=3.98°C); Q_freeze accounting closes energy budget
  under T-clamping at T_freeze; pre-step + post-step q_sfc both use
  the same `is_frozen` switch.
- **Compute_most_fluxes / simple_bulk_fluxes** sign convention:
  returns ``τ_x = −ρ·u*²·u_rel/|U|`` (atmospheric retarding); callers
  flip for ocean (bulk_formulas.py:84) or use directly for atmosphere
  / sea-ice / lake.  All paths verified.

No code changes this iteration.

## Next iterations
Continue addressing codex findings and direct-inspection sweeps until all schemes are
provably conservative, monotone, CFL-safe, and AD-safe across mixed wet/dry grids.
Next compression at iter-110.

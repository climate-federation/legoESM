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
- **Inspection-only**: iter-71 (YSU + surface_layer + EDMF), iter-72 (TKE + CLUBB-lite +
  pbl_height), iter-74-78 (ocean surface_forcing + bulk_flux + 5 convection schemes +
  ocean_tendency_common + barotropic substep solvers), iter-82-83 (scan + Richards units
  + ice modules).

## Iterations 84-89 — Summary (compressed 2026-05-13 after iter-90)

### Real code fixes
- **iter-84** (PDI hydraulic K wet-end smoothstep inverted, `land/soil_hydraulics.py:322`):
  Cosine interpolation between `h_crit` and `h≈0` was `0.5·(1+cos(π·frac))`, giving
  `Kr_interp = Kr_crit` at frac=0 (wet) and `Kr_interp = 1` at frac=1 — opposite of the
  intended BC.  Loam defaults clamped saturated K at ~34 % of K_sat in the wet regime.
  Fix replaces `(1+cos)` with `(1−cos)` so the smoothstep runs 1 → Kr_crit as frac goes
  0 → 1.  Verification: `K(h=1e-12) = K_sat` exact, `K(h_crit) = K_sat · Kr_crit`
  continuous.  38 soil-hydraulics tests green.
- **iter-85a/b/c** (Thomas tridiag denominator-floor, `land/tridiag.py:42-55`): old
  `jnp.sign(denom)*_tiny + _tiny` collapses to **exactly 0** for tiny negatives on a
  non-FTZ backend (`sign(-x)·_tiny + _tiny = 0`).  On our XLA build the bug is masked by
  subnormal flush-to-zero (`sign(-1e-40) → -0.0`), but the masking is platform-dependent.
  Fix: branch-based `sign = where(denom >= 0, 1, -1); denom = where(|denom|<_tiny,
  sign·_tiny, denom)` — guarantees a non-zero floor on every platform.  Note: does NOT
  preserve sign under FTZ (`-0.0 >= 0` is True).  iter-85b/c reframed test and docs to
  the honest "non-zero floor, not sign-preserving under FTZ" claim after codex stop-time
  catches.  208 land+soil tests green.
- **iter-86** (free-drift ice velocity dimensionally inconsistent, 300× too small,
  `ice/dynamics.py:105`): old `u_ice = drag_ocean·U_w + (drag_atm·ρ_air/ρ_ice)·U_a`
  treats dimensionless drag coefficients as velocity-mapping ratios and uses ρ_ice
  instead of ρ_oc.  For U_a=5, U_w=0.1 yields ~5.7e-4 m/s drift vs physical Zubov
  ~0.18 m/s.  Fix: steady-state air/ocean drag balance linearised to
  `u_i = U_w + α·(U_a - U_w)` with `α = sqrt(ρ_air·C_ai / (ρ_oc·C_oi)) ≈ 0.017`
  (Zubov / Nansen 1-2 % wind rule).  Signature: `rho_ice → rho_ocean`.  51+3+7 tests
  green.
- **iter-87** (slab-path free-drift, codex follow-up to iter-86, `ice/sea_ice.py:184-187`):
  iter-86 missed an inlined copy of the broken formula in the slab thermo / diagnostic
  path; replaced with a call to the shared `free_drift_velocity`.  117 tests green.
- **iter-88** (EDMF updraft dimensionally inconsistent,
  `atmosphere/physics/turbulence/edmf.py:201-204`): old `dw_dz = buoy - eps·w_u` mixed
  [m/s²] and [1/s].  Lagrangian form is `dw/dz = B/w - ε·w`; dropping `B/w → B` flipped
  the sign of dw/dz at small w_u, strangling buoyant updrafts that should accelerate.
  Fix: integrate the squared form `d(w²)/dz = 2(B - ε·w²)`, AD-safe via
  `sqrt(max(w², 1e-20))` floor + `where(w²>0, sqrt, 0)` gate.  101 tests green.

### Inspection-only iters
- **iter-89**: convection sweep (`_plume`, `mass_flux`, `zhang_mcfarlane`, `kessler`,
  `_warm_rain`, `compute_cape`, `compute_moist_adiabat`).  No production hot-path defects.
  Removed dead `e_sat = saturation_vapor_pressure(T_parcel)` in `compute_lcl` (Bolton
  Eq. 22 uses RH directly) + unused import.  21 plume tests green.

### Codex stop-time review activity
- Caught two real iter-86 follow-ups: (1) the iter-87 slab-path miss; (2) the iter-85
  test misframing (asserted sign-preservation but XLA FTZ masks it — claim is only that
  the floor is non-zero, sign-preservation only without FTZ).  Honest reframing landed in
  iter-85b/c.

## Inspected & clean (as of iter-90)
- **Land**: `snow_budget`, `carbon/{carbon_cycle,stomata}`, `richards`, `soil_thermal`,
  `multilayer_land`, `slab_land`, `stomata_utils`, `surface_params`, `param_providers`,
  `tridiag` (post iter-85), `soil_hydraulics` (post iter-84).
- **Atmosphere turbulence**: `clubb_lite`, `holtslag_boville`, `louis`, `ysu`, `tke`,
  `vertical_diffusion`, `surface_layer`, `pbl_height`, `edmf` (post iter-88).
- **Atmosphere clouds**: `cloud_fraction`.
- **Atmosphere GWD**: `rayleigh`, `lindzen`, `mcfarlane`, `hines`, `integration`.
- **Atmosphere radiation**: `gray`, `solar`, `ozone_ml`, `integration`,
  `rrtmgp/rrtmgp` (post iter-79).
- **Atmosphere convection**: `bechtold`, `emanuel`, `dca`, `kuo`, `zhang_mcfarlane`,
  `tiedtke`, `kain_fritsch`, `mass_flux`, `_plume` (post iter-89).
- **Atmosphere microphysics**: `kessler`, `seifert_beheng`, `morrison`, `thompson`,
  `_warm_rain`, `output` (post iter-73).
- **Ocean physics**: `lateral_mixing/{harmonic,biharmonic,gm_redi,gm_redi_latlon_cgrid,
  gm_redi_mpas,backscatter,_gm_redi_common}` (post iter-52/53/54), `shortwave_penetration`,
  `mixing` (post iter-58/59), `bottom_drag/quadratic`, `vertical_mixing/{kpp,
  implicit_solver,mpas_integration}` (post iter-57), `convection/{enhanced_diffusion,
  plume}`, `surface_forcing/{prescribed,restoring,bulk_formulas,wind_profiles}`.
- **Ocean dynamics**: `ocean_tendency_common`, `barotropic_common`,
  `barotropic_latlon_cgrid` (explicit), `barotropic_implicit_latlon_cgrid` (with
  known limitation).
- **Coupler**: `surface_energy`, `bulk_flux`, `surface_exchange` (post iter-81),
  `lake/two_layer_lake` (post iter-67).
- **Ice**: `rheology`, `dynamics` (post iter-86), `itd`, `transport`, `sea_ice` (post
  iter-55, iter-87).
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
  MOM6 / NEMO use iterative implicit methods for exact consistency.
- Pre-existing test failure `test_b_salt_sign_freshening_is_stabilizing` (harness
  fragility — `jnp.mean(None)` when `out.K_v` is None) — needs harness fix.
- Pre-existing test failure `TestEVPSanity::test_zero_velocity_isotropic_stress` (EVP
  doesn't relax to -P/2 in 10 subcycles at dt=3600s with 15 % rtol; verified on HEAD~1).
- **Plume buoyancy uses dry T, not virtual** (`convection/_plume.py:653`; affects
  `compute_lfc_lnb` too).  ~1 K virtual correction comparable to plume-alive threshold.
  `compute_cape` already supports virtual-T mode when q_v threaded; the plume
  integrator does not yet.  Scheme-wide recalibration — defer until a dedicated tuning
  iter.
- **Sea-ice `_thermo_single` emissivity** (`ice/sea_ice.py:614-617`): radiation
  always uses `config.emissivity_ice` regardless of `ice_mask`.  For h=0 / empty-cat
  cells the lw_up has a ε_ice (0.97) vs ε_ocean (0.99) bias.  Masked by conc-
  weighted coupling in production, but a cleaner per-cat branch on
  `surface_radiation_fluxes` is possible.
- Codex `adversarial-review` runtime ~50/50 success/failure when running.  Continue
  alongside direct inspection.

## Iter-91 — gravity-wave-drag + sea-ice thermo sweep (inspection only)
Audited `atmosphere/physics/gravity_wave_drag/{rayleigh,mcfarlane,
lindzen,hines}.py` and `ice/sea_ice._thermo_single`.

All four GWD backends verified:
- **Rayleigh**: BL drag ramp + sin² sponge top; frictional heating
  `dT/dt = -(u·du/dt + v·dv/dt)/c_pd` = `+k·(u²+v²)/c_pd` preserves
  total energy (KE → IE).  Column dissipation positive-definite.
- **McFarlane / Lindzen**: stress dimensionally consistent
  `tau_0 = G_0 · ρ · N · k · h² · U` [Pa]; saturation
  `tau_sat = ε · ρ · |U|³ · k / N` [Pa]; soft-min / sigmoid breaking;
  `accel = -drag/(ρ·dz)` projects along surface-wind direction —
  correct for mountain-wave drag (decelerates upper flow in the
  direction of original surface wind).
- **Hines**: per-level rho-ratio WKB growth (single-step, not
  cumulative); `sigma_sat = N/m_*` per level (not /√(ρ_sfc/ρ));
  drag = ρ·(σ_grown² - σ_new²) [Pa].  All previous dimensional
  issues caught in earlier iters (audits B1, B2).

Sea-ice `_thermo_single`: ice_mask gating on F_cond / dh_dt_sublim
correct, skin_cap = ½·ρ·c·h matches the half-slab convention,
temperature clamp + excess-energy → surface-melt pattern is
energy-conserving.  Open-water freeze gated by `enable_lead_freeze`
for multi-category lead-freeze deposition (iter-3 #3 fix held).

**One minor finding documented as deferred**: `surface_radiation_fluxes`
uses `config.emissivity_ice` for all cells regardless of ice_mask —
for h=0 / cat with conc=0 cells the lw_up bias is ε_ice (0.97) vs
ε_ocean (0.99), but the result is masked by conc-weighted coupling
so it does not affect production runs.  Listed under
*Outstanding / Deferred*.

No code changes this iteration.

## Iter-92 — solar insolation NaN gradient at polar day / polar night
`atmosphere/physics/radiation/solar.py` (`daily_mean_insolation`,
`daylight_fraction`).  The sunset-hour-angle calculation

    cos_hs = clip(-tan(lat) · tan(δ), -1.0, 1.0)
    h_s    = arccos(cos_hs)

produces a finite forward value (h_s = 0 at polar night, π at polar
day), but the backward pass NaN-s for every column inside the polar
cap: ``d/dx arccos(x) = -1/sqrt(1 − x²)`` is ±∞ at the clip boundary,
and JAX's zero subgradient through ``clip`` then evaluates 0 · ∞ as
``NaN``.  Forward Q is correct (533 W/m² at 80°N summer; 0 at 80°N
winter), but ``jax.grad(Q, lat)`` and ``jax.grad(Q, doy)`` are NaN at
every lat ≥ 67° in summer / winter.

Fix: clip ``cos_hs`` to ``[-1 + ε, 1 − ε]`` with ε = 1e-7 so the
``arccos`` derivative is finite (≤ 1/sqrt(2ε) ≈ 2236, well within fp32
and fp64 dynamic range).  Forward effect on Q at the polar boundary
is at most ``(S_0/π) · ε ≈ 4 × 10⁻⁵ W/m²`` — undetectable in any
physical context.  Same fix applied to `daylight_fraction`.

Regression test `test_daily_mean_insolation_polar_ad_safe` sweeps lat
∈ {±80°, ±88°} × doy ∈ {1, 80, 172, 265} and asserts the forward
value plus `d/dlat`, `d/dDOY` are finite for both
`daily_mean_insolation` and `daylight_fraction`.  Existing 72-test
radiation suite (`tests/unit/test_physics_radiation.py` +
`tests/atmosphere/hydrostatic/unit/test_radiation.py`) stays green.

## Iter-93 — ocean bottom drag + ozone_ml + radiation integration sweep
Audited `ocean/physics/bottom_drag/{linear,quadratic,integration}.py`,
`atmosphere/physics/radiation/{ozone_ml,rrtmgp_radiation,gray,
integration}.py`.

- **Quadratic bottom drag**: `drag = -C_d · |u| · u / dz_bot` —
  dimensionally [m/s²]; AD-safe via `sqrt(u² + v² + ε)` floor on
  speed; explicit-CFL timescale `dz/(C_d·|u|) ~ 4×10⁶ s` (~46 days)
  for typical (dz=1000 m, |u|=0.1 m/s, C_d=2.5e-3); orders of
  magnitude above dt=3600 s.  Pad-with-zero top fills inactive
  layers (single Pad HLO op).
- **Linear bottom drag**: `drag = -r · u / dz_bot` with r in [m/s],
  matching MITgcm `bottomDragLinear` convention.  Same pad-with-zero
  pattern.
- **Ozone ML ridge**: bilinear (lat, lon) + log-p vertical interp
  with bound-clipped indices and `argsort` for per-column ascending
  log-p; einsum contraction `"czk,ck->zk"` correctly sums over the
  feature axis (n_c = n_z = T at each UKESM level).  AD-safe
  divisions guard `x_scale_col == 0`.
- **Gray radiation heating rate**: `dT/dt = (g/c_p) · dF_net_up/dp`
  with surface-last index convention is the canonical Frierson form
  (signs derived: subsidence convention dp>0 downward, dF>0 absorbing
  → warming).  Beer-Lambert SW with α-reflection escaping directly to
  TOA correctly conserves column energy.
- **Radiation integration daily-mean path**: at polar boundary the
  iter-92 ε=1e-7 `cos_hs` clip leaves f_day ≈ 4.5×10⁻⁶ (not 0); the
  `f_day_safe = max(f_day, 1e-6)` guard in `cos_sza = insol/(S_0·f_day)`
  still functions correctly (insolation is 0 at polar night so cos_sza
  = 0/anything = 0).  Iter-92 fix verified to not break the downstream
  RRTMGP daytime-effective cos(SZA) derivation.

No code changes this iteration.

## Iter-94 — land carbon + stomata + snow_budget sweep
Audited `land/carbon/{stomata,carbon_cycle}.py` and `land/snow_budget.py`.

Verified clean:
- **Farquhar electron-transport quadratic**: discriminant `(αQ + J_max)² −
  4θ αQ J_max` stays positive for all 0 < θ < 1; sqrt-gradient finite
  (analytic minimum disc ≈ 0.84·J_max² at αQ = (2θ−1)·J_max, so disc
  never gets close to 0 across any realistic photosynthesis regime).
- **Arrhenius / peaked_arrhenius**: standard temperature-dependence forms,
  no AD-unsafe operations.
- **Ball-Berry / Medlyn / Jarvis stomata**: gs floors at `g0` via outer
  max; VPD / RH derived from specific humidity via correct
  `e_air = q · p / (ε + (1−ε)·q)` (mixing-ratio form would bias ~1 %,
  fixed in iter-2 audit #24); `compute_stomatal_beta` Beer-law canopy
  blend OK.
- **Coupled Farquhar-stomata fixed point**: `Ci = Ca − 1.6 · A_pos /
  gs_safe` standard form; `Ci` clipped to `[1, Ca]`; n_iter_ags
  unrolled for JIT.
- **`compute_gpp`** LUE-style with CO2 Michaelis-Menten and Gaussian
  T-response — all factors non-negative via `jnp.maximum`.
- **`compute_phenology`**: DALEC990 Gaussian-pulse phenology with
  hemisphere-aware doy shift; offsets via Horner polynomial.
- **`_effective_rate(r, dt)` = (1 − (1−r)^dt) / dt**: exact
  exponential decay; AD-safe via `clip(r, 0, 1−1e-10)`.
- **`step_carbon_differland`**: iter-62/63/64 cascade `C_lab → C_fol →
  C_root → C_wood` deficit draws in place; `_soft_pos = jnp.maximum(x,
  0)` replaces softplus (iter-64 fix held).
- **`update_snow`**: energy-limited melt `M = min(snow, max(0,
  Q_net·dt/L_f))` with degree-day fallback; donor cap on melt.

No code changes this iteration.

## Next iterations
Continue addressing further codex findings and direct-inspection sweeps until all
schemes are provably conservative, monotone, CFL-safe, and AD-safe across mixed
wet/dry grids.  Next compression at iter-100.

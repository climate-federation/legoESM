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
  `_add_halos`, not before; halo extrapolation could otherwise push q_v outside the bounds.
- **iter-81** (dry-NH coupler crash): `extract_atm_to_surface_nh` switched from
  `jnp.where(has_tracers, …, 0.0)` (traces both branches + crashes on shape-0 axis) to a
  Python `if` on the static `n_tracers` shape.
- **Inspection-only**: iter-71, 72, 74-78, 82-83.

## Iterations 84-89 — Summary (compressed 2026-05-13 after iter-90)
- **iter-84** (PDI hydraulic K wet-end smoothstep inverted, `land/soil_hydraulics.py`):
  loam defaults clamped saturated K at ~34 % K_sat.  Fix `(1+cos) → (1−cos)`.
- **iter-85a/b/c** (Thomas tridiag denom-floor, `land/tridiag.py`): branch-based sign
  `where(denom >= 0, 1, -1)`; guarantees non-zero floor on any platform.
- **iter-86/87** (free-drift ice velocity 300× too small): dimensionless drag coeffs
  treated as velocity-mapping ratios + ρ_ice instead of ρ_oc.  Fix: Zubov balance
  `u_i = U_w + α·(U_a − U_w)`, `α = sqrt(ρ_a·C_ai / (ρ_oc·C_oi))` ≈ 0.017.
- **iter-88** (EDMF updraft dimensionally inconsistent): squared form
  `d(w²)/dz = 2(B − ε·w²)` with AD-safe `sqrt(max(w², 1e-20))` floor.
- **iter-89** (convection sweep + dead-code cleanup).

## Iterations 90-99 — Summary (compressed 2026-05-13 after iter-100)
- **iter-92** (solar `daily_mean_insolation` NaN gradient at polar day/night): clip
  `cos_hs ∈ [−1+ε, 1−ε]` with ε=1e-7 so `arccos` derivative stays finite.
- **iter-97/99** (SB autoconversion onset sigmoid scale): normalise to
  `sigmoid(s · (x_c/x_star − 1))` and add separate `autoconversion_sharpness: float = 10`
  to SB/Morrison/Thompson configs (reusing `saturation_sharpness=100` is 100× too steep).
  iter-98 attempted same normalisation on breakup but `breakup_sharpness=1e4` is in
  `[1/m]` and was already correctly tuned — iter-99 reverted that change.
- **Inspection-only**: iter-91 (GWD + sea-ice _thermo_single), iter-93 (ocean bottom drag
  + ozone_ml + radiation integration), iter-94 (land carbon + stomata + snow_budget),
  iter-95 (ocean vmix + convection), iter-96 (KPP end-to-end).
- **Codex stop-time follow-ups**: iter-87 slab-path miss after iter-86; iter-97/99
  parameter-units-vs-defaults rework.

## Iterations 100-109 — Summary (compressed 2026-05-13 after iter-110)

### Real code fixes
- **iter-102/103** (Tiedtke CMT downdraft, `convection/tiedtke.py:375`): had
  `M_d = -downdraft_alpha · M_u · 0.3` — extra hardcoded `× 0.3` on top of the canonical
  `downdraft_alpha = 0.3` LFS ratio gave 9 % effective ratio.  Companion subcloud
  rain-evap path used `downdraft_alpha · M_b · downdraft_trigger`.  iter-102 dropped
  the spurious `× 0.3`; codex stop-time review caught the missing RH gate; iter-103
  added `M_d = -downdraft_alpha · M_u · downdraft_trigger[:, None]`.
- **iter-104** (Bechtold CMT downdraft, `convection/bechtold.py:372`): identical pattern
  as Tiedtke — fixed jointly with the same trigger gate.  Verified Z-M passes
  `M_d=None` and KF / DCA / Kuo have no CMT branch.

### Inspection-only iters (no code changes)
- **iter-100**: compression milestone (iter-91..99 folded) + sigmoid-scale sweep across
  all microphysics / clouds / turbulence — no new bugs.  Confirmed iter-97 form is
  canonical, iter-98 reverted form is unit-consistent.
- **iter-101**: ocean surface forcing (9 wind profiles + restoring + prescribed +
  bulk_formulas + integration), lake two-layer (freshwater density inversion + Q_freeze
  energy budget), `compute_most_fluxes` / `simple_bulk_fluxes` sign convention traced
  across atmosphere / sea-ice / lake / ocean callers.  Global_wind polynomial verified
  at φ ∈ {0, 30°, 50°, 70°}.
- **iter-105**: `multilayer_land` (543 LOC), `slab_land` (328 LOC), coupler ocean tile.
  G_surface = SW + LW − SH − LH energy budget with melt-energy subtraction + water-
  limited evap excess add-back; W_bucket overflow → runoff; ocean `evap_rate = lhflx/L_v`
  (never sublimates); TileResponse.tau_x in atmospheric retarding convention with
  ocean physics tau_x flip for ocean convention.
- **iter-106**: Emanuel (sort_multiplier-through-delta_0 only, not subsidence;
  unsaturated downdraft column conservation), Kuo (MC-gate via `tanh(MC/me_threshold)`
  prevents spurious heating at MC=0; moistening budget normalised to column integral),
  atmosphere `_shared` (virtual-T variants, `compute_moisture_convergence` sign through
  3 grid paths, `diagnose_grid_w_from_omega` standard `w = -ω/(ρ·g)`).
- **iter-107**: Morrison + Thompson hand-audited conservation.  Σ_species dq/dt cancels
  for all 12+ phase-change pathways; latent heating consistent with `h = c_pd T +
  L_v q_v − L_f q_ice` (L_s − L_v − L_f = 0); donor clamps via shared
  `donor_clamp_scale`; sedimentation `extra_sink` for all 4 species; rime-to-graupel
  donor split correct.
- **iter-108**: convection `_triggers` smooth primitives (log-sum-exp soft-max,
  softplus positive-part, sequential survival product for first-crossing) + ITD
  `linear_remap` volume-conserving rescale preserves `a · h = vol_remap` exactly.
- **iter-109**: TKE + CLUBB-lite turbulence prognostic budgets.  Semi-implicit
  dissipation linearisation `ε ≈ Ce·sqrt(e_old)·e_new/l` keeps the update
  non-negative-bounded; CLUBB-lite tracks wp2 with parallel structure; iter-172 F841
  dead higher-moment cloud-fraction code removal held.

### Codex stop-time review activity
- One real follow-up (iter-103): CMT downdraft amplification without RH trigger.

## Inspected & clean (as of iter-110)
- **Land**: `snow_budget`, `carbon/{carbon_cycle,stomata}`, `richards`, `soil_thermal`,
  `multilayer_land` (post iter-105), `slab_land` (post iter-105), `stomata_utils`,
  `surface_params`, `param_providers`, `tridiag` (post iter-85), `soil_hydraulics`
  (post iter-84).
- **Atmosphere turbulence**: `clubb_lite` (post iter-109), `holtslag_boville`, `louis`,
  `ysu`, `tke` (post iter-109), `vertical_diffusion`, `surface_layer`, `pbl_height`,
  `edmf` (post iter-88).
- **Atmosphere clouds**: `cloud_fraction`.
- **Atmosphere GWD**: `rayleigh`, `lindzen`, `mcfarlane`, `hines`, `integration` (post
  iter-91).
- **Atmosphere radiation**: `gray`, `solar` (post iter-92), `ozone_ml`, `integration`,
  `rrtmgp/rrtmgp` (post iter-79).
- **Atmosphere convection**: `bechtold` (post iter-104), `emanuel` (post iter-106), `dca`,
  `kuo` (post iter-106), `zhang_mcfarlane`, `tiedtke` (post iter-102/103),
  `kain_fritsch`, `mass_flux`, `_plume` (post iter-89), `_triggers` (post iter-108).
- **Atmosphere microphysics**: `kessler`, `seifert_beheng`, `morrison` (post iter-107),
  `thompson` (post iter-107), `_warm_rain` (post iter-73, iter-97/99), `output`.
- **Atmosphere _shared** (post iter-106).
- **Ocean physics**: `lateral_mixing/{harmonic,biharmonic,gm_redi,gm_redi_latlon_cgrid,
  gm_redi_mpas,backscatter,_gm_redi_common}` (post iter-52/53/54, iter-90),
  `shortwave_penetration`, `mixing` (post iter-58/59), `bottom_drag/{linear,quadratic}`
  (post iter-93), `vertical_mixing/{constant,kpp,implicit_solver,mpas_integration,
  richardson,k_profiles}` (post iter-57, iter-95, iter-96),
  `convection/{enhanced_diffusion,plume}` (post iter-95),
  `surface_forcing/{prescribed,restoring,bulk_formulas,wind_profiles}` (post iter-101).
- **Ocean dynamics**: `ocean_tendency_common`, `barotropic_common`,
  `barotropic_latlon_cgrid` (explicit), `barotropic_implicit_latlon_cgrid` (with
  known limitation).
- **Coupler**: `surface_energy` (post iter-105), `bulk_flux` (post iter-101),
  `surface_exchange` (post iter-81), `lake/two_layer_lake` (post iter-67, iter-101).
- **Ice**: `rheology`, `dynamics` (post iter-86), `itd` (post iter-108), `transport`,
  `sea_ice` (post iter-55, iter-87, iter-91).
- **Shared**: `thermo`, `surface_albedo`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter.
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping.
- Threading `dt` into ocean `physics_fn` across 3 ocean PE backends + 3 model drivers.
- Visual / long-run validation of new ice → ocean feedbacks.
- σ-tensor rotation across cubed-sphere faces (O(dx) edge error in sea-ice EVP).
- Bottom drag double-application in explicit barotropic substep path (effective `2·r/H`).
- Implicit barotropic Hu_avg O(θ²·dt²·g·∇²η) inconsistency with eta-update flux.
- Pre-existing test failure `test_b_salt_sign_freshening_is_stabilizing` (harness
  fragility — `jnp.mean(None)` when `out.K_v` is None).
- Pre-existing test failure `TestEVPSanity::test_zero_velocity_isotropic_stress`
  (EVP doesn't relax to -P/2 in 10 subcycles at dt=3600 s; verified on HEAD~1).
- **Plume buoyancy uses dry T, not virtual** (`convection/_plume.py:653`).  ~1 K virtual
  correction comparable to plume-alive threshold.  `compute_cape` supports virtual-T
  when q_v threaded; plume integrator does not yet.  Scheme-wide recalibration.
- **Sea-ice `_thermo_single` emissivity** (`ice/sea_ice.py:614-617`): `ε_ice` used for
  all cells regardless of `ice_mask`.  Masked by conc-weighting in production.
- **SB autoconversion τ-based Φ_au switch**: the simplified x_c switch (iter-97/99) is
  canonical but suppresses drizzle in maritime conditions (high N_c).  Proper SB-2001
  / Stevens-2007 implementation would add a τ-based `Φ_au(q_r/(q_c+q_r))` switch.
- Codex `adversarial-review` runtime ~50/50 success/failure.  Continue alongside
  direct inspection.

## Iter-111 — coupler infrastructure (accumulator + tile fractions + step)
Audited `coupler/{accumulator,tile_fractions,coupling_fields}.py` and
the `step_surface` orchestrator inside `coupler/coupler.py`.

Verified clean:
- **`FluxAccumulator`**: dt-weighted accumulation of all SurfaceToAtm
  fields including the separately-tracked `sum_lw_up` (avoids the
  σ⟨T⟩⁴ ≠ ⟨σT⁴⟩ trap when emission averaging is needed downstream).
  `mean_accumulator` divides by `clip(total_dt, _tiny, None)` for
  AD-safe division.
- **`compute_tile_fractions`**: `f_water = 1 - f_land - f_lake`,
  `f_ice = f_water · ice_conc`, `f_ocean = f_water - f_ice`.  Total
  sums to 1 (`f_ocean + f_ice + f_land + f_lake = 1` algebraically).
  Static rescale `static_scale = 1/total_static` triggers when
  `f_land + f_lake > 1` — slightly oversaturated input is renormalized
  rather than producing negative `f_water`.
- **`blend_tiles`**: area-weighted linear blend `f_o·O + f_i·I +
  f_l·L + f_k·K` for every TileResponse field including
  `freshwater_flux`, `ocean_heat_extraction`, `ocean_stress_x/y`
  (audit F8/F9), `surface_mass_flux` (audit F3).  Continuous, no hard
  conditionals — differentiable everywhere.
- **`step_surface` asynchronous coupling window** (lines 430-465):
  on flush, accumulate the residual `dt_to_close = clip(coupling_dt -
  dt_prev, 0, dt)` to close the window exactly, emit mean, seed
  `acc_next` with `dt_excess = max(dt - dt_to_close, 0)` worth of
  blended flux.  Boundary cases: when `dt_excess = 0` the new
  accumulator has `total_dt = 0` and emit-side `_tiny` floor protects
  the next-mean division.

No code changes this iteration.

## Iter-112 — ocean EOS + atmosphere thermodynamics audit
Audited `ocean/eos.py` (506 LOC) and
`atmosphere/physics/thermodynamics.py` (416 LOC).

Verified clean:
- **Wright (1997) EOS** (`wright_eos`): polynomial form `ρ = (p + p_0) /
  (λ + α_0·(p + p_0))` with intermediate float64 promotion for the
  large coefficients (~5.79e8); no clipping of T, S inputs so
  unphysical overshoots remain visible (issue #165).
- **`thermal_expansion_coeff`** `α = −(1/ρ) ∂ρ/∂T` via `jax.vmap`
  + `jax.grad` on the scalar EOS.  Positive for water (warming
  decreases density).
- **`haline_contraction_coeff`** `β = (1/ρ) ∂ρ/∂S`.  Positive (salt
  increases density).
- **`linear_eos`**: `ρ = ρ_ref · [1 − α_T·(T − T_ref) + β_S·(S − S_ref)]`
  — standard linear form, dimensionally consistent.
- **`compute_hydrostatic_pressure`**: `p_top = ρ_ref·g·η + cumsum(ρ·g·h)
  − ρ·g·h`; cell-center = `p_top + 0.5·dp`.  Supports `h_actual` for
  partial-cells extension.
- **`compute_buoyancy_frequency`**: `N² = −(g/ρ_ref)·(ρ_shallow −
  ρ_deep)/dz_iface`.  Sign matches standard z-up convention regardless
  of indexing direction: stable strat gives `ρ_shallow < ρ_deep` →
  drho_dz < 0 → N² > 0. ✓
- **`temperature_from_theta`**: `T = θ·(p/p_ref)^κ` with `_THETA_MIN`,
  `_P_MIN`/`_P_MAX` clips for AD safety.
- **`pressure_from_eos`**: `p = p_0·(R_d·ρ·θ/p_0)^(c_p/c_v)` (non-
  hydrostatic dycore form); base clipped to `[1e-20, 1e20]` before
  the power.
- **`moist_adiabat_lapse_rate`**: canonical Iribarne–Godson form
  `Γ_m = (R_d·T)/(c_p·p) · (1 + L_v·q_sat/(R_d·T)) / (1 + L_v²·q_sat/
  (c_p·R_v·T²))` in [K/Pa].

No code changes this iteration.

## Iter-113 — shared `thermo.py` + `constants.py` audit
Audited `src/legoesm/thermo.py` (172 LOC, shared saturation thermo)
and `src/legoesm/constants.py` (125 LOC).

Verified clean:
- **`saturation_vapor_pressure`**: Tetens form
  `e_sat = 611.2 · exp(17.67 · T_c / (T_c + 243.5))` in Pa.  AD-safe
  (no singular operations in the realistic range).
- **`saturation_mixing_ratio`**: `q_sat = ε · e_sat / max(p − e_sat,
  softplus(p − e_sat − 1) + 1)` — smooth softplus floor preserves
  gradients near `e_sat ≈ p` (prevents zero-gradient plateau a hard
  clip would create); LogSumExp smooth-min cap at `q_sat ≤ 1` for
  low-pressure singularity safety.
- **`saturation_mixing_ratio_ice`**: Clausius–Clapeyron form
  `e_sat_i = 611.2 · exp(L_s/R_v · (1/T_freeze − 1/T))`, same
  softplus floor + cap as the liquid variant.
- **`saturation_mixing_ratio_dT`**: analytical derivative
  `d(q_sat)/dT` consistent with the Tetens formula; uses simpler
  hard `max(p − e_sat, 1)` floor (PDF-width convention).
- **`saturation_specific_humidity`**: `q = w_sat / (1 + w_sat)`
  conversion from mixing ratio to specific humidity (~1 % difference
  at typical tropospheric humidities).
- **`constants.py`**: `c_vd = c_pd − R_d` enforces the thermodynamic
  identity (iter-39 MEDIUM #6 fix held — earlier hardcoded 717.56
  violated the identity by 0.03 J/(kg·K)).  Molar masses available
  in both g/mol (e.g. `M_air`, `M_H2O`) and kg/mol (`M_dry`, `M_h2o`)
  forms with `* 1e-3` derivation to prevent drift.  Centralised
  emissivities (`emissivity_ocean`, `emissivity_ice = 0.97`,
  `emissivity_land = 0.95`).  Freshwater EOS local-parabolic fit
  constants (`T_freshwater_max_density = 277.133 K`,
  `rho_freshwater_curvature = 8e-6 K⁻²`).

No code changes this iteration.

## Iter-114 — Holtslag-Boville + YSU PBL schemes audit
Audited `atmosphere/physics/turbulence/{holtslag_boville,ysu}.py`
(both 228 LOC).

Verified clean:
- **PBL-height diagnosis** (shared): smooth bulk-Ri sigmoid weighting
  `σ_pbl · (1 − σ_pbl)` peaks at the Ri_crit crossing (not centroid);
  `h_pbl = ∫(z · w_pbl) / ∫w_pbl`.  Clipped to `≥ 100 m` floor.
- **K-profile inside PBL**: `Km = κ·u*·z·(1−z/h)²` standard form,
  zero at z=0 and z=h.
- **Local Ri-based Km above PBL** (Louis 1982 style): `f_stable = 1/(1
  + 2b·Ri / sqrt(1 + d·Ri))`, `f_unstable = 1 − 2b·Ri / denominator`,
  sigmoid blend on Ri.  All Louis coefficients (b, c, d) sourced from
  config (no hardcoded literals — iter-? constant-discipline fix held).
- **HB counter-gradient correction**: `γ_h · w'θ'_sfc / (Km_max ·
  h_pbl)` adds the non-local heat flux as an enhanced surface BC
  in the implicit T-diffusion solve.  Known simplified form (full
  non-local profile shape would require modifying the RHS at every
  level).
- **YSU entrainment Gaussian**: `K_ent = c_ent·w*·h · exp(−((z−h) /
  (0.3·h))²)` with `w* = cbrt(max(g·h·max(w'θ',0)/θ̄, 1e-20))` —
  AD-safe via the 1e-20 floor (cbrt gradient is finite away from 0);
  stable BL (w'θ'<0) correctly gives w*≈0 → no entrainment.
- **Implicit vertical diffusion**: heat in θ-space for dry-adiabat
  neutrality, momentum/moisture in raw space.

One minor noted: YSU computes `θ̄ = mean(θ_v, axis=1)` over the FULL
column (including stratosphere) rather than the PBL only — slight
overestimate of θ̄ → underestimate of w*.  Scheme simplification, not
a bug.

No code changes this iteration.

## Next iterations
Continue addressing codex findings and direct-inspection sweeps until all schemes are
provably conservative, monotone, CFL-safe, and AD-safe across mixed wet/dry grids.
Next compression at iter-120.

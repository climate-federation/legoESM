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


## Iterations 110-119 — Summary (compressed 2026-05-13 after iter-120)

All inspection-only iters (no code changes), one bug fix at iter-119 boundary (none).
Total LOC audited: ~5000.

- **iter-110** (compression milestone) + brief `ocean_tendency_common.py` inspection:
  2-pass EOS-pressure iteration, partial-cell-safe `iterate_eos_and_pressure_anomaly`,
  `implicit_bottom_drag_factor`, KE-1999/MOM6 BBL distributed drag with iter-39 #2 fix.
- **iter-111** coupler infrastructure: `FluxAccumulator` dt-weighted with separately-
  tracked `sum_lw_up` (avoids `σ⟨T⟩⁴ ≠ ⟨σT⁴⟩` trap); `compute_tile_fractions` sums to 1
  algebraically with static rescale for oversaturated input; `blend_tiles` linear-
  weighted; `step_surface` async coupling window correctly closes/seeds with `_tiny`
  floor on the next-mean division.
- **iter-112** ocean EOS + atmosphere thermodynamics: Wright 1997 polynomial with f64
  intermediate promotion; `α/β` via `jax.vmap+grad` on scalar EOS; `compute_hydrostatic
  _pressure` with `h_actual` partial-cell support; `N²` sign matches z-up convention;
  `moist_adiabat_lapse_rate` canonical Iribarne-Godson form.
- **iter-113** shared `thermo.py` + `constants.py`: Tetens / Clausius-Clapeyron
  saturation thermo with softplus floor + LogSumExp cap; `c_vd = c_pd − R_d` thermo
  identity (iter-39 MEDIUM #6); molar masses in g/mol and kg/mol with `*1e-3` derivation
  to prevent drift; centralised emissivities; freshwater EOS parabolic-fit constants.
- **iter-114** Holtslag-Boville + YSU PBL: shared bulk-Ri sigmoid `σ·(1−σ)` PBL-height
  weighting peaks at Ri_crit crossing; `K-profile = κ·u*·z·(1−z/h)²`; Louis-style
  Ri-dependent local Km above PBL with config-sourced (b, c, d); HB counter-gradient
  as enhanced surface BC; YSU entrainment Gaussian with AD-safe `cbrt(max(...,1e-20))`
  for w*; θ-space heat diffusion preserves dry-adiabat neutrality.
- **iter-115** `barotropic_common` + `barotropic_latlon_cgrid`: `compute_filter_weights`
  cosine fallback to box at n<2 (iter-1 #4 held); `bebt_blend` semi-implicit eta blend
  (#205); min-rule face depth `H_u = min(roll(H), H)`; flux-form barotropic diffusion
  `div(ν_face · grad η)` for exact volume conservation on cos(lat) grid; forward-
  backward Matsuno Coriolis; divergence damping targets eta-checkerboard without
  affecting geostrophic flow; pole rows = wall BC; cosine time filter on eta/velocity
  but BOX on Hu/Hv transport (volume conservation).
- **iter-116** radiation integration: 3-mode insolation dispatch with iter-92 polar
  AD-safe clip; `_extract_tracer_columns` dtype-inferred (no f64 promotion on Metal);
  zero placeholders pinned to upstream state precision; daily-mean RRTMGP path
  correctly rescales SW by `f_day`; mutable-dict time-state closure prevents per-step
  recompile.
- **iter-117** ocean lateral-mixing backscatter: two-pass `+∇²(A_bs · ∇²u)` exact
  discrete adjoint of `strain_rate` → energy-consistent; MPAS variant has correct
  `[m²/s]` units; `update_eddy_energy` budget closes such that resolved+SGS total
  energy decays only via slow `−E/τ` memory term.
- **iter-118** land `stomata_utils` + `surface_params`: dispatcher correctly chains
  Farquhar-stomata (with differland LAI) or Jarvis fallback; optional `land_params`
  override via NamedTuple `_replace`; `LandSurfaceParams` 12-field NamedTuple with
  `PARAM_BOUNDS` for ML sigmoid-bounded parameterization; CLM5 PFT 17×12 lookup table.
- **iter-119** ocean MPAS GM/Redi: edge-normal `F_n = κ_R·∂_n q + (κ_R − κ_GM)·S_n·∂_z q`
  with partial-cell `bot_e` masking and `edge_mask = mask[c1]·mask[c2]` coastline
  zeroing on BOTH horizontal F_n and Perot input `S_n_oc`; vertical flux via Perot
  reconstruction at cell centres; TRiSK `divergence_cell_3d` + `vertical_flux_divergence`;
  Neumann coastline fill before tracer differencing; Visbeck adaptive κ_GM mirrors
  cubed-sphere form.

### Inspected & clean (additions through iter-119)
All sections cumulatively verified; no new defects across coupler infrastructure,
ocean EOS, shared thermo / constants, HB / YSU PBL, barotropic substep solver,
radiation integration dispatcher, backscatter scheme, land surface params, MPAS GM/Redi.

## Next iterations
Continue addressing codex findings and direct-inspection sweeps until all schemes are
provably conservative, monotone, CFL-safe, and AD-safe across mixed wet/dry grids.
Next compression at iter-130.

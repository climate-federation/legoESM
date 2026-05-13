# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iteration 66 — 2026-05-13

**Inspection iteration on sea-ice `_thermo_single` (no code changes).**

- **`dh_dt_sublim`** uses `-lhflx / (ρ_ice · L_s)`.  Upstream
  `_bulk_flux_dispatch` always passes `L_latent=constants.L_s` for
  sea-ice (sublimation, not vaporisation) so the division is
  dimensionally consistent.  ✓
- **`freshwater_to_ocean = -ρ_ice · (dh_dt_total - dh_dt_sublim)`**:
  Sign convention verified — pure sublim gives FW=0 (mass goes to
  atmosphere, not ocean), pure melt gives FW > 0 (mass to ocean),
  pure freeze gives FW < 0 (mass extracted from ocean).
- **`h_eff = max(h, h_ice_min)`** caps conduction at very thin ice
  (h < 0.01 m); under-estimates F_cond marginally but prevents
  numerical divergence.  Mass conservation unaffected.
- **Phantom skin on open water (`ice_mask=False`)**: `skin_cap`
  uses `h_eff = h_ice_min` even on open-water cells.  `T_trial`
  responds, but the `T_new = T_freeze_ocean` clip and the
  `dh_dt = where(ice_mask, dh_dt_ice, dh_dt_open)` selector mask
  the spurious `dh_dt_surface_melt` so it doesn't affect the open-
  water energy budget.
- **`dconc_growth = dh_dt_open · lead_area / h_new_ice`** matches
  CICE convention: new-ice forms with `h_new_ice` thickness over
  `lead_area`; concentration grows linearly with the freezing rate.
- **`dconc_melt = min(dh_dt, 0) · conc / h_eff`**: floes shrink
  laterally while thickness stays ≈ constant; sign carries through
  for melt.

No new fixes needed.  CICE V=h·A refactor remains the open
structural item for the slab-path mass-bookkeeping under
simultaneous melt + lead-freeze (iter-25 attempt reverted).

## Iteration 65 — 2026-05-13

**Inspection iteration on atmospheric convection + coupler bulk-flux
helpers (no code changes).**

- **`bechtold.py`** (Bechtold/IFS, 395 lines): PBL-CAPE closure
  matches Kain (2004) §3 form with proper `rho_BL` and `g` factors;
  MC enhancement gated to O(1) by `mc_normalize_scale = 0.05
  kg/m²/s`.  Downdraft rain-evaporation conservation verified:
  `col_int(evap_rate · dp/g) = evap_total`, `col_int(rain_scale ·
  dq_c) = rain_source_total - evap_total` (column water balanced).
- **`_apply_mass_flux_kernel`** (Tiedtke 1989 / Siebesma 2007):
  subsidence + detrainment decomposition; separate `q_v_u` and
  `q_c_u` plume streams (instead of conflated `q_u`) close column
  MSE budget.  Stratosphere mass-flux gate prevents TOA T spikes.
- **`emanuel.py`** post iter-42 downdraft rewrite: vapor source
  via `evap_rate ∝ below_lcl/below_mass`; cloud-water subtract via
  `weight = dq_c_conv_dt_raw / col_dq_c` (subtract = downdraft_eff
  · col_dq_c, vapor gain = same).  Conservation closes; `dq_c -
  subtract ≥ 0` because `subtract ≤ dq_c_conv_dt_raw ≤ dq_c_conv_dt`.
- **`surface_radiation_fluxes`** (`coupler/surface_energy.py`):
  net = SW_down·(1-α) + LW_down·ε - σT⁴·ε.  Upward LW = emission +
  reflected.  Total surface energy gain = SW_net + LW_net.  Matches
  grey-surface BC (iter-13 sea-ice + iter-42 ocean fix).
- **`simple_bulk_fluxes`** (`coupler/bulk_flux.py`): standard
  bulk-aerodynamic with positive-upward sign convention.  Stress
  opposes wind; heat / moisture fluxes track surface − air
  difference times `rho · c_p · Ch · |U|`.

No new fixes needed.  iter-42 covered the Emanuel AD-safe floor;
iter-13/42 covered the grey-surface LW fix downstream.  All paths
audited above close their conservation budgets to first principles.

## Iteration 64 — 2026-05-13

**Fix codex iter-63 stop-time finding: exhausted-labile case still
leaks carbon.**

The iter-63 cascade capped each draw at `state.C_x / dt_days`,
which ignored the simultaneous natural turnover drain
(`lab_release`, `leaf_litter`, etc.).  When deficit_draw +
natural_drain together exceeded the pool, the new pool value went
slightly negative and the `_soft_pos` softplus clamped it to
`_alpha · log(2) ≈ 0.007 gC/m2` of phantom carbon per pool per
step.

Fix:
1. **Net-available cap**: compute `lab_net_avail = max(state.C_lab +
   (A_lab − lab_release) · dt, 0) / dt_days` so the cap accounts for
   the natural turnover flow BEFORE deciding how much deficit a pool
   can absorb.  Same for fol/root/wood.
2. **Hard `jnp.maximum(x, 0)`** replaces softplus.  With the cap
   guaranteeing `state.C + (A − drain − deficit)·dt ≥ 0` exactly,
   no smoothing bias is added.

New regression test
`test_total_carbon_conservation_machine_precision` exercises the
exhausted-labile + no-GPP forcing at `rtol=1e-9, atol=1e-9` —
50,000× tighter than the iter-62 test, ~12 orders of magnitude
better than the original iter-50 `rtol=0.05`.

**Tests (post iter-64):** 45 / 45 carbon unit tests pass.

## Iteration 63 — 2026-05-13

**Fix codex iter-62 stop-time finding: labile-reserve cap leaves
ghost carbon when C_lab is exhausted.**

Iter-62 capped the deficit draw at `C_lab/dt_days`.  If C_lab is
exhausted (depleted from prolonged carbon starvation), the
remaining deficit `(R_auto − GPP) − C_lab/dt_days` is still emitted
to atmosphere via NEE without any pool decrement — ghost carbon
returns.

Fix: cascade the deficit through ALL living pools in the standard
CASA/DALEC order:
1. C_lab (labile reserves first)
2. C_fol (foliage)
3. C_root (root)
4. C_wood (wood — slowest turnover, last)

Each draw is capped at its pool/dt_days.  Residual imbalance is
only possible if ALL FOUR pools are simultaneously exhausted —
biologically equivalent to plant death — and bounded by the total
biomass budget per step.

New regression test
`test_total_carbon_conservation_with_exhausted_labile_pool`
constructs a state with `C_lab=0.5, C_fol=300, C_root=400, C_wood=
10000` under no-GPP conditions and verifies the cascade closes the
budget at `rtol=1e-3`.

**Tests (post iter-63):** 9 / 9 TestDifferLandStep including
new exhausted-C_lab regression; 44 / 44 carbon unit tests.

## Iteration 62 — 2026-05-13

**Bug fix: carbon conservation under R_auto > GPP.**

`carbon_cycle.step_carbon_differland` emits `R_auto = R_maint +
R_growth` to atmosphere via NEE, but the biomass pools that
PRODUCED `R_maint` are NEVER decremented.  When `GPP < R_maint`
(polar winter, drought, nighttime + cold), the deficit `(R_auto −
GPP)·dt` of "ghost carbon" appears in the atmosphere every step.

The existing `test_total_carbon_conservation_tendency` masked this
with `rtol=0.05` (5%); the violation is bounded by `R_maint − GPP`
which is small under typical tropical forcing but visible under
carbon-starvation conditions.

Fix: when `NPP_day < 0`, draw the deficit `|NPP_day|` from the
labile pool `C_lab` (capped at `C_lab/dt_days` so the pool can't go
negative within a step).  Carbon balance now closes exactly under
`GPP ≥ R_auto`, and within the `C_lab/dt_days` cap under
carbon-starvation regimes.  Standard CASA/DALEC convention.

New regression test
`test_total_carbon_conservation_under_carbon_starvation` exercises
`sw=0, T=285K, beta=1, precip=1e-6` (zero light + above-freezing
respiration) and verifies conservation at `rtol=1e-3` (50× tighter
than the existing test).

**Tests (post iter-62):** 43 / 43 carbon unit tests pass +
new regression at `rtol=1e-3`.

## Iteration 61 — 2026-05-13

**Scan-only iteration: verified other ocean dz / jacobian divisions
are protected (no code changes).**

Followed up on iter-58/59 to ensure the dry-column NaN guards are
not missing anywhere else.  Scanned all `/` and `dz_*` patterns in
the ocean codebase:

- **`ocean_pe_latlon_cgrid.py:1302`**: `jac_v = jnp.maximum(J, 1e-10)`
  bounds `dz_actual_loc = dz_ref · jac_v ≥ dz_ref · 1e-10 > 0` for
  the baseline `K_v` tracer diffusion.  Forward + AD safe.
- **`ocean_pe_latlon_cgrid.py:1621-1622`**: same `max(jac, 1e-10)`
  for `A_v` momentum diffusion on u-faces / v-faces.
- **`ocean_pe_mpas.py:933, 942`**: `h_safe = jnp.maximum(h_e, 1.0)`
  (1 m floor, mask-protected downstream — overkill but AD-safe).
- **`implicit_solver.py:133-134`**: `K_safe = max(K, 0)`,
  `dzh_safe = max(dzh, _EPS)`, `inv_dz = 1/max(dz, _EPS)` — all
  protected.
- **`mpas_integration.py:97, 107, 111-113`**: `h_safe = max(h_e, 1.0)`
  + zero-K on out-of-column interfaces — forward + AD safe.
- **`_gm_redi_common.py:166`**: `sqrt(S_x² + S_y² + 1e-30)` — additive
  regulariser (no division), AD-safe.
- **`stomata.py:200`**: `(2a + 1e-20)` divisor — degenerate (a=0)
  case gives J=0 (not the correct c/b limit) but is unreachable for
  realistic `theta_j ≈ 0.7`.

No new fixes needed.  The iter-56..59 pass covered the
forward-NaN-on-dry-column class of bugs in the leaf
`vertical_diffusion*` and `shortwave_penetration` paths; upstream
backends (`ocean_pe_*`) all use additive or max-bounded floors that
were already AD-safe.

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

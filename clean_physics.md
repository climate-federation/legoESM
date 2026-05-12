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
- `constants.p_atm_std = 101325 Pa`, `constants.R_universal = 8.314462618 J/(mol·K)` added.
  `ocean/surface_forcing/bulk_formulas.py` and `land/carbon/stomata.py` reference them.
- `convection/mass_flux._compute_column_geometry` delegates to `_shared.compute_layer_dz`
  / `compute_rho` with `q_v`; threaded through all 5 mass-flux schemes (Bechtold,
  Emanuel, Tiedtke, Kain-Fritsch, Zhang-McFarlane).  Full-level heights use mid-layer
  `cumsum(dz)[::-1] - 0.5·dz`.
- Constant-hygiene audit (iter-12): all `g`/`R_d`/`c_pd`/`L_v`/`L_s`/`L_f`/`sigma_sb`/
  `T_freeze`/`Omega`/`R_earth`/`rho_air`/`rho_ocean`/`epsilon` references go through
  `constants.py`.

### Conservation & monotonicity
- **Sedimentation** (`microphysics/output.py`): optional `dt` applies Bott/FCT positivity
  limiter (`flux ≤ q·ρ·dz/dt`); `return_surface_flux=True` returns the dt-limited bottom
  flux for conservative precipitation diagnostics; `extra_sink` parameter jointly caps
  sedimentation + rain-evap against `q_r/dt`.  Threaded through all 4 schemes.
- **Rain evaporation** (`_warm_rain.py`): optional `dt` clamps to `q_r/dt`.
- **Sundqvist autoconversion**: donor cap so `P_auto · dt ≤ q_c`.
- **Sea-ice transport**: conservative monotone PPM (Colella–Woodward with limiter);
  `n_subcycles` runs PPM substeps with `lax.scan`.
- **ITD remap**: volume-conserving rescale; residual folded into thickness when `a > 1`.
- **RRTMGP**: upper-clip `q_v ≤ 0.99` so `(1 - q_v)` H2O VMR denominator is bounded.
- **DCA STANDARD MSE conservation** (in-scheme latent heating via `delta_T_lh`).

### Donor clamps consolidated (iter-30..36)
- New `donor_clamp_scale(q_avail, sink_total, dt, divisor_floor=1e-15)` in `_warm_rain.py`.
  AD-safe floor on divisor avoids fp32 VJP overflow (`-q/sink_dt²` overflowed under fp32
  with the legacy 1e-30 floor).  Worst-case VJP `-q/1e-30 ≈ 1e30` ≪ fp32 max ~3.4e38.
- All 8 microphysics donor clamps (Kessler q_c; Morrison q_c/q_i/q_s/q_v; Thompson
  q_c/q_i/q_s/q_v; Seifert-Beheng q_c) route through the shared helper.
- Total water conservation tests for all 4 schemes assert
  `|residual|/max(|precip|, 1e-10) < 1e-6` in both fp32 and fp64.

### CFL caps & stability
- **Enhanced_diffusion convection** (ocean): explicit branch delegates per-interface CFL
  cap to `vertical_diffusion_variable_K(dt=...)` using `dz_actual = dz_ref · jacobian`.
- **Ocean vertical mixing leaf** (`mixing.py`): `vertical_diffusion` /
  `vertical_diffusion_variable_K` accept optional `dt` + `cfl_safety` and cap `K`
  per-interface.  KPP, Richardson, constant schemes thread `dt`.
- **Ocean lateral mixing harmonic + biharmonic**: opt-in `enforce_cfl`; harmonic
  `A_h/K_h ≤ cfl_safety · dx² / dt`, biharmonic `B_h_* ≤ cfl_safety · dx⁴ / dt`.
- **EDMF mass flux** (turbulence): `M ≤ 0.5·ρ·dz/dt`.

### Coupler / feedback paths
- Sea-ice `_bulk_flux_dispatch` helper centralises MOST/COARE/Large-Yeager/simple_bulk.
  Used by slab path, dynamic single-cat thermo (iter-39), dynamic multi-cat `_thermo_cat`,
  and diagnostic `_build_response`.
- `_build_response` ice → ocean feedback uses per-ice-area `dh/dt` for FW flux (CICE
  V=h·A refactor deferred); computes `ocean_heat_extraction` and `ocean_stress_x/y`
  mirroring the slab path.  Dynamic path threads `h_agg_post_transport` so FW =
  thermodynamic ΔV only.
- Multi-cat lead freeze: `_thermo_single` gains `open_water_fraction` +
  `enable_lead_freeze` kwargs.  Dynamic dispatch drives concentration growth by
  aggregated `(1 - sum_k conc_k)` and fires open-water freeze in category 0 only.
- Sea-ice / ocean / lake / land radiative emission: `lw_up = ε·σ·T⁴ + (1-ε)·lw_down`
  (grey surface).  Iter-13 sea-ice, iter-42 ocean.
- Multilayer land: dropped bogus `root_frac[None,:]` unsqueeze; dew/deposition routing
  gated on `has_snow=False`; soil-side flux_top routes full surface flux to bare-soil
  on snow-free dew, zero transpiration.
- Slab land surface albedo (iter-44): `fresh_snow_surviving = where(T_soil < T_freeze,
  precip_snow*dt, 0)` added BEFORE the albedo block (mirrors multilayer iter-68 fix).

### Tracer mixing / ocean drag / BBL
- Harmonic + biharmonic lateral tracer Laplacian output is multiplied by `mask_3d`
  (no-flux BC tracking deferred).
- **BBL drag** (`ocean_tendency_common.bbl_drag_distributed` +
  `ocean_pe_latlon_cgrid._bbl_drag_for_face`): divides by `min(H_BBL, total_overlap)`
  so shelf/coastal cells where total wet depth < H_BBL get the correct drag rate.

### Streamfunction diagnostics (iter-37)
- `moc_streamfunction` and `barotropic_streamfunction` use `grid.lat_v` / `grid.lat` /
  `grid.dlat` / `grid.dlon` when available so regional grids get correct face lengths.

### Ocean bulk-flux + Emanuel AD-safety (iter-42)
- `bulk_formulas._saturation_specific_humidity` converts mixing ratio → specific humidity
  (kg/kg moist) to fix +3 % latent-flux bias.
- `Q_lw_up = ε σ T_s^4 + (1-ε)·LW_down` (~10 W/m² Q_net correction).
- `emanuel.py:204` AD-safe floor raised 1e-30 → 1e-15 (donor_clamp_scale pattern).

### Test status (post iter-42, last full sweep)
- 17 / 17 atmosphere hydrostatic integration tests; 68 / 68 atmosphere microphysics;
  95 / 95 atmosphere turbulence; 88 / 88 sea-ice + ocean bottom-drag;
  75 / 75 land carbon + multilayer + diff_land; 140 / 140 land (post iter-44);
  100 / 100 ocean MPAS + surface_forcing + emanuel + atmosphere convection.

## Iteration 57 — 2026-05-13

**Defensive AD-safety fix: KPP non-local dz_actual divisor.**

KPP non-local tendency (`vertical_mixing/kpp.py:430-444`) divides
flux convergence by `dz_actual`.  On dry columns ``dz_actual = 0``
gives `0/0 = NaN` in the inner computation; the forward `jnp.where`
on `is_unstable_col` scrubs the NaN forward (since `is_unstable_col`
is False for B_f=0 on land), but the **backward** pass through the
0/0 division still produces NaN gradients.

Fix: AD-safe divisor `dz_safe = jnp.where(dz_actual > 0, dz_actual, 1)`
and gate the output where on both `is_unstable_col & dz_actual > 0`.
Forward output unchanged on wet columns; gradients now clean.

**Tests (post iter-57):** 11 / 11 KPP tests pass (excluding the
pre-existing `test_b_salt_sign_freshening_is_stabilizing` failure
which is a test-fragility issue — `jnp.mean(None)` when `out.K_v
is None` — and unrelated to this fix).

## Iteration 56 — 2026-05-13

**Bug fix: `shortwave_penetration_tendency` produces Inf/NaN on dry
columns.**

`shortwave_penetration_tendency` (`ocean/physics/shortwave_penetration.py:124`)
computed `dT/dt = sw_down · frac_absorbed / (rho_0 · c_sw · dz_actual)`
with `dz_actual = z_coord.dz_ref · jacobian`.  For dry / land cells
the jacobian is 0, so `dz_actual = 0` and the division produced Inf
on every land grid point.  Downstream `dT_dt + sw_tend` propagated
Inf into the tendency; ``NaN * 0 = NaN`` in IEEE so the output mask
applied later could not scrub the contamination.

Fix: add a `dz_actual > 0` guard with safe denominator before the
division, then `jnp.where(dz_actual > 0, dT/dt, 0)` so dry columns
contribute exactly zero heating and gradients stay clean.

New regression test
`test_dry_column_gives_zero_finite_tendency` exercises a mixed wet /
dry grid (half J=0, half J=1) and verifies dry rows return zero and
wet rows return positive heating.

**Tests (post iter-56):** 11 / 11 shortwave-penetration tests pass
(plus dispatch).  Also exercised indirectly by the broader ocean
test matrix.

## Iteration 55 — 2026-05-13

**Resolved deferred item: `SeaIceConfig.Delta_min` now threads through
to `delta_deformation`.**

`SeaIceConfig.Delta_min` field existed (default 2.0e-9 [1/s]) but was
never passed in.  `evp_solver` and `evp_stress_update` were updated
to accept and forward `Delta_min`; `step_sea_ice → _step_dynamic`
now passes `Delta_min=config.Delta_min`.  Users can finally tune the
plastic-yield-curve smoothness without editing source.

New regression test
`test_delta_min_threads_through_evp_stress_update` exercises the
sub-yield branch (strain ~1e-12) and verifies that distinct
`Delta_min` values produce distinct stress responses — catches a
regression where the config field is silently ignored.

**Tests (post iter-55):** 51 / 51 sea-ice dynamics tests pass.
65 / 65 sea-ice + lateral-mixing-dispatch tests pass.

Removed from "Outstanding / Deferred": `Delta_min` config field
threading.

## Iteration 54 — 2026-05-13

**Fix codex iter-53 stop-time finding: wrong stencil reach in fill.**

Iter-53 used `n_passes=1` (harmonic) / `n_passes=2` (biharmonic),
matching a naive compact-stencil interpretation.  The actual legoESM
operators are wider:

- `laplacian_viscosity_3d` = `div(grad(·))` with centred FD
  `(f[i+1] - f[i-1]) / dx` → at cell `j` samples `{j-2, j, j+2}`.
  Stencil reach = 2 cells → `n_passes=2`.
- `hyperdiffusion_3d` = `compact_laplacian(·)` then `div(grad(·))` —
  reach 1 + 2 = 3.  At cell `j` samples `f[j±1], f[j±2], f[j±3]`.
  Stencil reach = 3 cells → `n_passes=3`.

With the corrected `n_passes`, the Neumann fill covers the full
stencil reach so coastline gradients are zero across every cell the
operator actually inspects.  Trade-off: bridges land barriers up to
2 cells (harmonic) or 3 cells (biharmonic) wide.  Wider barriers
no longer leak.

**Tests (post iter-54):** 35 / 35 lateral-mixing + harmonic +
biharmonic + surface-forcing-dispatch tests pass.

## Iteration 53 — 2026-05-13

**Fix codex iter-52 stop-time finding: thin-barrier leak.**

`fill_land_cells` default `n_passes=3` bridges land strips ≤ 3 cells
wide, creating spurious cross-basin diffusion through narrow
isthmuses (e.g. Central America at coarse resolution).

Fix:
- `harmonic.py`: `fill_land_cells(T, mask, grid, n_passes=1)` —
  matches the 5-point Laplacian stencil reach.
- `biharmonic.py`: `fill_land_cells(T, mask, grid, n_passes=2)` —
  matches ∇²∇² 2-cell stencil reach.

Residual limitation: a 1-cell-wide land strip is still bridged by
harmonic (1 pass), and a ≤ 2-cell-wide strip by biharmonic.  This
is unavoidable without connected-components analysis of the mask
and is the minimum reach required for the stencil to see correct
gradients at coastlines.  Wider barriers (≥ 2 cells for harmonic,
≥ 3 cells for biharmonic) no longer leak.

**Tests (post iter-53):** 35 / 35 lateral-mixing + harmonic +
biharmonic + surface-forcing-dispatch tests pass.

## Iteration 52 — 2026-05-13

**Resolved deferred item: no-flux BC for harmonic/biharmonic tracer
Laplacian.**

`harmonic.py` + `biharmonic.py` (cubed-sphere lateral mixing) now
Neumann-fill tracers via `fill_land_cells(T, mask, grid)` BEFORE
the Laplacian / biharmonic operator, then mask the output.  The
3-pass cubed-sphere Neumann fill (already used in production for
pressure-anomaly handling, `barotropic.py:45-125`) replicates the
ocean value at land neighbours so the stencil sees zero gradient
across the coastline.  Previously only the OUTPUT was masked,
which let one (harmonic) or two (biharmonic ∇⁴ = ∇²∇²) stencil
cells of land-sentinel contamination propagate into the ocean
interior before being zeroed.

**Tests (post iter-52):** 25 / 25 harmonic + biharmonic + lateral-mixing
tests pass; 28 / 28 surface-forcing-dispatch + ocean-differentiability
tests pass.  No new tests needed since existing
`test_lateral_mixing_factory_dispatch[harmonic|biharmonic|gm_redi]`
and the Smag-biharmonic-MPAS differentiability suite exercise the
new fill path.

Removed from "Outstanding / Deferred": no-flux BC item.  GM/Redi
already uses `_neumann_fill_cgrid` on the lat-lon path
(`gm_redi_latlon_cgrid.py:87, 178, 406-407, 727`) and
`_voronoi_neumann_fill` on the MPAS path
(`gm_redi_mpas.py:68, 188, 337, 567`).

## Iteration 51 — 2026-05-13

**Inspection iteration on `src/legoesm/land/multilayer_land.py`
(no code changes).**

Codex still timing out — direct inspection only.

- **Snow / latent-heat phase decision** (`multilayer_land.py:197-200`):
  `has_snow = (snow > 1e-6) | ((precip_snow·dt > 1e-6) & (T_surface < T_freeze))`.
  Surviving-fresh-snow guard prevents warm-column snowfall from
  spending the whole turbulent step over L_s/ice qsat.

- **Energy budget closure** (`multilayer_land.py:271-272, 375`):
  `G_surface` accounting:
  `G = SW_net + LW_net − shflx − lhflx_demand`,
  then `G −= melt_energy` (latent of fusion paid by snowmelt),
  then `G += evap_excess_energy` (water-limited evap returns
  unused latent heat to the soil thermal step).  Final
  `G_final = SW_net + LW_net − shflx − lhflx_actual − melt_energy`.
  Energy conserved.

- **Sublimation cap** (`multilayer_land.py:281-286`):
  `max_sublim = snow_new/dt` so `sublim_actual ≤ available snow / dt`.
  For deposition (sublim_demand < 0), `sublim_actual = sublim_demand`
  (full deposition); `snow_new − sublim_actual·dt` correctly ADDS
  the deposition because `sublim_actual < 0`.

- **Soil water-budget partition** (`multilayer_land.py:336-343`):
  When `has_snow=False`, surface flux splits into bare-soil
  `(1 − f_veg)·evap_rate` and transpiration `f_veg·evap_rate`.
  Dew (`evap_rate < 0`) routed entirely to bare-soil (`evap_bare =
  evap_rate`, `evap_transp = 0`) so the column water budget closes.

- **Root sink normalisation** (`multilayer_land.py:350-354`):
  `sink = (root_frac·β_root / Σ(root_frac·β_root)) · E_pot_transp / dz`.
  Vertical integral over the column exactly equals `E_pot_transp`,
  preserving the transpiration water budget through Richards.

Note: bulk-flux dispatch (`compute_most_fluxes` vs `simple_bulk_fluxes`)
duplicates `_bulk_flux_dispatch` pattern from sea-ice but is inline
here.  Future refactor could centralise via shared helper.

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
- `land/snow_budget.py`, `land/carbon/carbon_cycle.py`, `land/richards.py`,
  `land/soil_thermal.py`, `land/multilayer_land.py`.
- `atmosphere/physics/turbulence/{clubb_lite,holtslag_boville,louis,ysu,
  vertical_diffusion}.py`.
- `atmosphere/physics/clouds/cloud_fraction.py`.
- `atmosphere/physics/gravity_wave_drag/{rayleigh,lindzen,mcfarlane,hines,
  integration}.py`.
- `atmosphere/physics/radiation/{gray,solar}.py`.
- `ocean/physics/lateral_mixing/{harmonic,biharmonic,gm_redi,gm_redi_latlon_cgrid,
  gm_redi_mpas,backscatter,_gm_redi_common}.py`.
- `ocean/physics/shortwave_penetration.py`, `ocean/physics/bottom_drag/quadratic.py`.
- `ice/{rheology,dynamics,itd,transport,sea_ice}.py`.
- `atmosphere/physics/convection/{dca,kuo,zhang_mcfarlane}.py`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter.
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping (iter-25 attempt
  reverted; slab path evolves h and conc semi-independently).
- Threading `dt` into ocean `physics_fn` across 3 ocean PE backends + 3 model drivers.
- Visual / long-run validation of new ice → ocean feedbacks.
- σ-tensor rotation across cubed-sphere faces (O(dx) edge error in sea-ice EVP).
- Codex `adversarial-review` runtime: 4 consecutive iters lost to exit 144.  Retry
  later or switch to direct inspection until runtime stabilises.

## Next iterations
Continue addressing further codex adversarial-review findings (when runtime
recovers) and direct-inspection sweep through the remaining modules until
all schemes are provably conservative, monotone, and CFL-safe.

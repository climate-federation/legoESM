# Clean Physics — Iterative Adversarial-Review Improvement Log

Goal: Perfect physics across atmosphere, land, ocean, cryosphere parameterizations.
Branch: `clean_physics`. Driven by Ralph loop + `/codex:adversarial-review`.

## Scope
- **Atmosphere** (`src/legoesm/atmosphere/physics/`): radiation, convection, microphysics, turbulence, clouds, GWD, _shared.
- **Land** (`src/legoesm/land/`): multilayer, richards, soil_thermal, snow, carbon, stomata.
- **Ocean** (`src/legoesm/ocean/physics/`): vertical_mixing, bottom_drag, lateral_mixing, convection, surface_forcing, shortwave_penetration, mixing.
- **Cryosphere** (`src/legoesm/ice/`): sea_ice, dynamics, itd, rheology, transport.

## Iterations 1-40 — Summary (compressed 2026-05-13 after iter-40)

### Constants & shared utilities
- `constants.p_atm_std = 101325 Pa`, `constants.R_universal = 8.314462618 J/(mol·K)` added.
  `ocean/surface_forcing/bulk_formulas.py` and `land/carbon/stomata.py` reference them.
- `convection/mass_flux._compute_column_geometry` delegates to `_shared.compute_layer_dz`
  / `compute_rho` with `q_v`; threaded through all 5 mass-flux schemes (Bechtold, Emanuel,
  Tiedtke, Kain-Fritsch, Zhang-McFarlane).  Full-level heights use mid-layer `cumsum(dz)[::-1] - 0.5·dz`.
- Constant-hygiene audit (iter-12): all `g`/`R_d`/`c_pd`/`L_v`/`L_s`/`L_f`/`sigma_sb`/`T_freeze`/
  `Omega`/`R_earth`/`rho_air`/`rho_ocean`/`epsilon` references go through `constants.py`.

### Conservation & monotonicity
- **Sedimentation** (`microphysics/output.py`): optional `dt` applies Bott/FCT positivity
  limiter (`flux ≤ q·ρ·dz/dt`); `return_surface_flux=True` returns the dt-limited bottom flux
  for conservative precipitation diagnostics; `extra_sink` parameter jointly caps
  sedimentation + rain-evap against `q_r/dt`.  Threaded through all 4 schemes.
- **Rain evaporation** (`_warm_rain.py`): optional `dt` clamps to `q_r/dt`.
- **Sundqvist autoconversion**: donor cap so `P_auto · dt ≤ q_c`.
- **Sea-ice transport**: conservative monotone PPM (Colella–Woodward with limiter);
  `n_subcycles` runs PPM substeps with `lax.scan`.
- **ITD remap**: volume-conserving rescale; residual folded into thickness when `a > 1`.
- **RRTMGP**: upper-clip `q_v ≤ 0.99` so the `(1 - q_v)` H2O VMR denominator is bounded.
- **DCA STANDARD MSE conservation** (in-scheme latent heating via `delta_T_lh`).

### Donor clamps consolidated (iter-30..36)
- New `donor_clamp_scale(q_avail, sink_total, dt, divisor_floor=1e-15)` in `_warm_rain.py`.
  AD-safe floor on divisor avoids fp32 VJP overflow (`-q/sink_dt²` overflowed under fp32
  with the legacy 1e-30 floor).  Worst-case VJP is `-q/1e-30 ≈ 1e30` — well within fp32
  dynamic range (~3.4e38).
- All 8 microphysics donor clamps (Kessler q_c; Morrison q_c/q_i/q_s/q_v; Thompson
  q_c/q_i/q_s/q_v; Seifert-Beheng q_c) route through the shared helper.
- Total water conservation tests for all 4 schemes assert
  `|residual|/max(|precip|, 1e-10) < 1e-6` in both fp32 and fp64.

### CFL caps & stability
- **Enhanced_diffusion convection** (ocean): explicit branch delegates per-interface CFL
  cap to `vertical_diffusion_variable_K(dt=...)` using `dz_actual = dz_ref · jacobian`.
- **Ocean vertical mixing leaf** (`mixing.py`): `vertical_diffusion` / `vertical_diffusion_variable_K`
  accept optional `dt` + `cfl_safety` and cap `K` per-interface.
- **KPP, Richardson, constant** vertical-mixing schemes thread `dt` to the leaf.
- **Ocean lateral mixing harmonic + biharmonic**: opt-in `enforce_cfl`; harmonic
  `A_h/K_h ≤ cfl_safety · dx² / dt`, biharmonic `B_h_* ≤ cfl_safety · dx⁴ / dt`.
- **EDMF mass flux** (turbulence): `M ≤ 0.5·ρ·dz/dt`.

### Coupler / feedback paths
- Sea-ice `_bulk_flux_dispatch` helper centralises MOST/COARE/Large-Yeager/simple_bulk.
  Used by slab path, dynamic single-cat thermo (iter-39), dynamic multi-cat `_thermo_cat`,
  and diagnostic `_build_response`.
- `_build_response` ice → ocean feedback uses per-ice-area `dh/dt` for FW flux (CICE V=h·A
  refactor deferred); computes `ocean_heat_extraction` and `ocean_stress_x/y` mirroring
  the slab path.  Dynamic path threads `h_agg_post_transport` so FW = thermodynamic ΔV only.
- Multi-cat lead freeze: `_thermo_single` gains `open_water_fraction` + `enable_lead_freeze`
  kwargs.  Dynamic dispatch drives concentration growth by aggregated `(1 - sum_k conc_k)`
  and fires open-water freeze in category 0 only.
- Sea-ice radiative emission: `lw_up = ε·σ·T⁴ + (1-ε)·lw_down` (grey surface).
- Multilayer land: dropped bogus `root_frac[None,:]` unsqueeze; dew/deposition routing
  gated on `has_snow=False` (snow absorbs entire latent flux); soil-side flux_top
  routes full surface flux to bare-soil-input on snow-free dew, zero transpiration.

### Tracer mixing / ocean drag / BBL
- Harmonic + biharmonic lateral tracer Laplacian output is multiplied by `mask_3d`
  (no-flux BC tracking deferred).
- **BBL drag** (`ocean_tendency_common.bbl_drag_distributed` + `ocean_pe_latlon_cgrid._bbl_drag_for_face`):
  divides by `min(H_BBL, total_overlap)` so shelf/coastal cells where total wet depth < H_BBL
  get the correct drag rate rather than one weakened by `total_wet_depth / H_BBL`.

### Streamfunction diagnostics (iter-37)
- `moc_streamfunction` and `barotropic_streamfunction` use `grid.lat_v` / `grid.lat` /
  `grid.dlat` / `grid.dlon` when available so regional grids get correct face lengths
  and meridional spacing.

### Test fixes (iter-13/14)
- `Test8i_StefanBoltzmann::test_lw_up_matches` — grey-surface LW.
- `test_hybrid_tracer_path_calls_vertical_advection` — relative path via `Path(legoesm.__file__)`.
- `test_ah_lat_scaling` × 4 — pass `power=2` for legacy cos² coverage.
- `test_dca_extended_mse_conservation` — STANDARD MSE.
- 7 / 7 pre-existing failures resolved (1 in iter-13, 6 in iter-14).

### Regression tests (iter-8, 22, 32, 34, 36, 38, 39)
- `TestVerticalMixing::test_vertical_diffusion_cfl_cap_keeps_step_stable`
- `test_sedimentation_cfl_positivity` + `test_sedimentation_surface_flux_conservation`
- `TestSundqvist::test_autoconversion_does_not_drive_qc_negative`
- `TestMorrison::test_qv_clamp_divisor_floor_protects_VJP` (uses production
  `donor_clamp_scale` helper directly)
- `TestMorrison::test_morrison_fp32_grad_finite_with_tiny_qv_sink`
- `TestKessler/TestSeifertBeheng/TestMorrison/TestThompson::test_total_water_conservation_under_heavy_clamp`
  (4 schemes; pass in fp32 and fp64)

### Test status (post iter-40)
- 17 / 17 atmosphere hydrostatic integration tests.
- 68 / 68 atmosphere microphysics tests.
- 95 / 95 atmosphere turbulence tests.
- 88 / 88 sea-ice + ocean bottom-drag tests.
- 75 / 75 land carbon + multilayer + diff_land tests.

## Inspected & clean (no fix needed)
- `land/snow_budget.py`, `land/carbon/carbon_cycle.py`, `land/richards.py`,
  `land/soil_thermal.py` — implicit / well-disciplined.
- `atmosphere/physics/turbulence/vertical_diffusion.py` — implicit.
- `atmosphere/physics/clouds/cloud_fraction.py` — shared `saturation_mixing_ratio`.
- `atmosphere/physics/gravity_wave_drag/lindzen.py` — `constants.*`, differentiable scan.
- `ocean/physics/shortwave_penetration.py` — conservation correction.
- `ocean/physics/bottom_drag/quadratic.py` — proper sign convention.
- `ice/rheology.evp_stress_update`, `ice/dynamics.evp_solver` — implicit / CICE convention.
- `atmosphere/physics/convection/{dca,kuo,zhang_mcfarlane}.py` — moist static-energy
  budgets verified.
- `atmosphere/physics/turbulence/{clubb_lite,holtslag_boville,louis,ysu}.py` — all use
  shared `virtual_temperature`.

## Outstanding / Deferred
- GM/Redi bolus Courant limiter (codex narrow review).
- CICE V=h·A state-variable refactor for sea-ice FW bookkeeping (iter-25 attempt
  reverted; legoESM's slab path evolves h and conc semi-independently).
- No-flux BC for harmonic/biharmonic tracer Laplacian (replicate ocean values at
  land neighbours before the operator).
- Threading `dt` into ocean `physics_fn` across 3 ocean PE backends + 3 model drivers.
- Visual / long-run validation of the new ice → ocean feedbacks.

## Next iterations
Continue addressing further codex adversarial-review findings until physics is
provably conservative, monotone, and CFL-safe across all parameterizations.

### Iteration 41 — 2026-05-13

Compression iteration.  Folded iter-30 through iter-40 entries into the
"Iterations 1-40 — Summary" section above so the working log stays under the
auto-loaded MEMORY.md / context envelope.  Detailed per-iteration narratives
remain in the commit messages on `clean_physics`.

### Iteration 43 — 2026-05-13

**Inspection iteration (no code changes).**

Codex narrow review on `atmosphere/physics/radiation/` /
`turbulence/` timed out without producing actionable findings (output
was source-code excerpts).  Direct inspection of the recently-touched
paths confirmed they are sound:

- `radiation/gray.py` — proper grey-surface LW BC at the seafloor
  (``F_up_sfc = ε σ T^4 + (1-ε) F_down_sfc``).
- `radiation/solar.py` — daily-mean insolation handles polar
  day / night via clipped ``cos_hs``.
- `turbulence/pbl_height.py`, `turbulence/holtslag_boville.py`,
  `turbulence/ysu.py` — divisor floors are ADDITIVE (e.g.
  ``+ 1e-20``), not ``max()`` — these do not trigger the
  ``-q / sink²`` VJP overflow pattern the iter-32 floor addresses.
- `integration.py` ``dtheta_prime_dt`` uses ``jnp.clip(exner, 1e-6,
  None)`` which never approaches 1e-6 in practice (TOA exner ≈ 0 at
  p → 0, but production columns never go below p ~ 100 Pa where
  exner ≈ 1e-2).

### Iteration 42 — 2026-05-13

**Apply codex iter-41 findings + Emanuel AD-safe floor.**

Codex returned 2 findings on `ocean/physics/surface_forcing/bulk_formulas.py`:

1. `_saturation_specific_humidity` returned mixing ratio (kg/kg dry)
   rather than specific humidity (kg/kg moist), biasing latent flux
   high by ``1 + r_sat`` (~3 % in the tropics).  Fixed by converting
   ``q_sat = r_sat / (1 + r_sat)``.
2. `Q_lw_up = ε σ T_s^4` omitted the reflected ``(1-ε)·LW_down``
   component, biasing ocean Q_net by ~10 W/m² for ε = 0.97 / typical
   tropical LW_down.  Now uses ``ε σ T_s^4 + (1-ε)·LW_down`` (mirror
   of the iter-13 sea-ice fix).

Plus a parallel-inspection find:

- `emanuel.py:204` — `dq_c_conv_dt / max(sort_multiplier, 1e-30)`
  could overflow fp32 in VJP when `sort_multiplier` is tiny.
  Raised the floor to 1e-15 (same AD-safe pattern as the iter-32
  donor_clamp_scale).

**Tests (post iter-42):** 100 / 100 ocean MPAS + surface_forcing +
emanuel + atmosphere convection tests pass.

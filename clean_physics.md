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

### Iteration 47 — 2026-05-13

**Inspection iteration on `src/legoesm/atmosphere/physics/gravity_wave_drag/`
(no code changes).**

Codex narrow review skipped (3 consecutive iters lost to exit 144;
overhead exceeds the value at this point).  Direct inspection of
`rayleigh.py`, `lindzen.py`, `mcfarlane.py`, `hines.py`,
`integration.py`:

- **Rayleigh sponge** — `sin²(π/2 · arg)` ramp with `arg = clip((sponge_top − σ)/sponge_top, 0, 1)` activates above `σ < sponge_top`; ramps smoothly to `k_max` at the model top.
- **Lindzen / McFarlane orographic** — launch stress
  `ρ_sfc · N_sfc · k_wave · h_topo² · U_ll` (correct units after
  iter-30 fix).  Both schemes correctly use `cos_a = u_sfc/U_ll`,
  `sin_a = v_sfc/U_ll` so the wave direction is fixed at the
  surface direction; column drag projected back to (du/dt, dv/dt).
- **Lindzen / McFarlane scan direction** — `k_rev = nlev-1-k`
  iterates UP from surface to top; `drag_stack.T[:, ::-1]`
  reverses to top-first ordering matching the input wind arrays.
  Verified: `drag_all[:, 0]` is the top-level drag.
- **Hines per-step ρ ratio** — `rho_ratio_step[:, k] = sqrt(rho[k+1] / rho[k])`
  is the inter-level WKB growth (per-step, not cumulative); the
  surface step (`k = nlev-1`) is fixed at 1.0 by `jnp.ones_like(rho)`
  initialization.
- **Spectral PE projection** (`dvor_hat = im/a · oc2(dv·cos) +
  1/a · dmu(du·cos)`) — matches the standard
  `ζ = (im·v + ∂(u·cos)/∂μ) / (a·cos)` with `sh_oc2` carrying the
  1/cos² factor.

Local "frictional heating" `dT_dt = -(u·du_dt + v·dv_dt)/c_pd` can
be locally negative under shear-reversal (when du_dt is along the
upper-level u, the work is positive), but the column-integrated
`eps_gwd = -Σ ρ(u·du_dt + v·dv_dt)·dz` remains positive in typical
flows.  This is a physical KE-budget accounting choice rather than
a sign bug.

Softmin underflow note (McFarlane line 140): for very small
`tau_carry, tau_sat ~ 0.01 Pa` the softmin can dip below the true
min by ~`log(2)/alpha`.  At realistic stratospheric scales
`tau_sat ~ 1 Pa` so this is dormant.

### Iteration 46 — 2026-05-13

**Inspection iteration on `src/legoesm/ocean/physics/lateral_mixing/`
(no code changes).**

Codex narrow review on the lateral-mixing modules failed (exit 144).
Direct inspection of `harmonic.py`, `biharmonic.py`, `gm_redi.py`,
`gm_redi_latlon_cgrid.py`, `gm_redi_mpas.py`, `backscatter.py`,
`_gm_redi_common.py`, `config.py`, `integration.py`:

- **DM95 taper** — smooth tanh, monotone, bounded [0, 1]
  (`_gm_redi_common.dm95_taper`).
- **Slope-tensor algebra** — Griffies (1998) small-slope form, signs
  matched to z-up convention; `F_x = κ_R ∂q/∂x + (κ_R − κ_GM) S_x ∂q/∂z`
  matches all three grid implementations (cubed, latlon-c, MPAS).
- **Per-triad cancellation** (`gm_redi_tracer_tendency_triads_latlon_cgrid`)
  — 8 W-face edge-pair triads at weight 0.25 each give
  `<S_x²> + <S_y²> = <S²>` (separable corner avg ≡ 4-corner avg).
  Tests `test_gm_redi_eady_physics.py::TestTriadCancellation` cover this.
- **vertical_flux_divergence** — returns `+∂F_z/∂z` with zero-flux BCs
  at top/bottom (correct for z-up); `dq/dt` consistent with the
  Laplacian-based diagonal that uses `coeff · ∇²q = ∇·G_diag`.
- **Visbeck κ_GM** — wet-column mask applied AFTER clipping (codex
  earlier finding still holds); `f_safe = max(|f|, f_min)` prevents
  infinite Rossby radius at the equator.
- **Backscatter energy budget** — `dE/dt = η ε_diss − ε_bs − E/τ`
  preserves the resolved/reservoir partition;
  `d(KE+E)/dt = −(1−η) ε_diss − E/τ` matches Jansen & Held (2014).

Note (deferred): `BiharmonicConfig.cfl_safety = 0.05` is 1.6× the 2-D
explicit-Euler stability limit `B·dt/dx⁴ ≤ 1/32` (legoESM uses
mixed compact-inner / 2h-outer stencils, so the actual Nyquist
spectral radius is smaller — opt-in `enforce_cfl=False` default makes
this dormant).  Production runs set `B ~ 1e10 m⁴/s`, ``dx ~ 100 km``
which is six orders below the cap, so the cap is dormant either way.
Skip pending mixed-stencil eigenvalue audit.

### Iteration 45 — 2026-05-13

**Inspection iteration on `src/legoesm/ice/` (no code changes).**

Codex narrow review (60 s budget) on the ice modules timed out
(exit 144).  Direct inspection of `dynamics.py`, `rheology.py`,
`itd.py`, `transport.py`, `sea_ice.py`:

- **EVP semi-implicit Coriolis** — algebra verified.  Per-substep
  rotation by ``2·alpha = f·dt_s`` cumulates to ``f·dt`` over
  ``N_evp`` substeps (eq. lines 352-355 of `dynamics.py`).
- **EVP backward-Euler stress relaxation** — ``E_factor = 1 /
  (2·T_evp·N_evp)`` correctly recovers ``dt_s / (2·T_damp)`` with
  ``T_damp = T_evp · N_evp · dt_s`` (rheology.py:268).
- **VP constitutive law** — ``σ_ij = 2η ε_ij + (ζ − η) δ_ij·tr(ε)
  − (P/2) δ_ij`` matches Hunke & Dukowicz (1997) (vp_stress).
- **PPM transport** — flux-form, monotone within CFL ≤ 1; volume
  ``h·A``, area ``A``, enthalpy ``T·h·A`` advected as independent
  conserved scalars; cell-mean temperature recovered as
  ``enthalpy / volume`` (transport.py:147-168).
- **ITD remap** — volume-conserving rescale after bound clamp; cat
  0 retains its own deficit (no smaller bin to demote to); last
  cat retains its own excess (no larger bin to promote to).
  Conservation exact when no clamp fires.
- **Strain-rate FD denominators** — ``grid.dy`` is already the
  full ``2·hy`` spacing between (i±1) (cubed_sphere.py:326, 794),
  so the ``(u[..., 2:] − u[..., :-2]) / dy`` formula is centred.

Known deferred items unchanged: CICE V=h·A refactor (lead-freeze
``h·Δconc`` over-counts new ice by ``h_old/h_new_ice``);
``Delta_min`` config field not threaded into ``delta_deformation``;
σ-tensor not rotated across cubed-sphere faces (O(dx) edge error).

### Iteration 44 — 2026-05-13

**Apply codex iter-43 finding: slab_land fresh-snow albedo.**

`land/slab_land.py:91` — the surface albedo block used the pre-step
`snow` while the later `has_snow` dispatch correctly included
`precip_snow*dt` for cold surfaces.  A snow-free cell receiving
fresh snow below freezing therefore absorbed BARE-LAND shortwave
for one timestep before snow accumulation took over the next step.

Fix: compute `fresh_snow_surviving = where(T_soil < T_freeze,
precip_snow*dt, 0)` BEFORE the albedo block and pass
`snow + fresh_snow_surviving` into `compute_land_albedo`.  Matches
the iter-68 multilayer-land fix that surface humidity / has_snow
also need fresh-snow accounting.

**Tests (post iter-44):** 140 / 140 land tests pass.

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

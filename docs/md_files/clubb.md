# CLUBB higher-order turbulence closure (`scheme="clubb"`)

Living reference for the fuller CLUBB scheme in
`packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py`.
Port + condensation history: `clubb_port_history.md` (same directory).

## Single-file architecture

Per the legoESM one-file-per-scheme convention, the entire scheme lives in
**one module, `clubb.py`** (~5.8k lines, 19 sections). A **line-numbered table
of contents** sits at the top of the file (kept exact by the
`test_toc_line_numbers_accurate` tripwire in `tests/unit/test_clubb_config.py`),
and the **CAM-default model-flag values** live as a machine-parseable comment
reference table at the end of the file. The 19 `tests/unit/test_clubb_*.py`
files map 1:1 onto the TOC sections (kept separate deliberately: pytest
sharding + no 10k-line test file).

Two entries:

- **Default (diagnostic), `clubb_turbulence`**: parcel buoyant-sorting
  `Lscale` eddy diffusion (`Km = c_K·Lscale·√em`) + the ADG1 double-Gaussian
  PDF cloud/buoyancy closure; `wp2` carried in the `tke` slot.
- **Prognostic (opt-in via `CLUBBConfig(prognostic=True)`),
  `clubb_turbulence_prognostic`**: the full CAM-default-tree higher-order
  moment closure; the 15-field `CLUBBMomentState` is carried in
  `PhysicsState.clubb_moments`. Hydrostatic + MPAS drivers supported;
  nonhydro/spectral_pe fail fast (they drop `phys_state` — same as MYNN-2.5).

Pipeline (each piece bit/round-off parity-validated vs CLUBB-JAX):
`compute_clubb_diagnostics` (Skw/σ²/em/tau/C6-C7/Kh) → `compute_pdf_closure`
(CAM ADG1: wpthvp/HOM/cloud-water fluxes/cloud_frac/rcm) → `advance_clubb_core`
(CAM order xm_wpxp → xp2_xpyp → wp2_wp3 → windm, `clip_covars_denom` between,
pre+post PDF — `l_call_pdf_closure_twice`) → `clubb_step` (legoESM column ↔
ascending host env + surface-flux BCs) → `clubb_turbulence_prognostic`
(CAM `clubb_timestep` sub-cycling; pack/unpack carry). Standalone SCM driver:
`integrate_clubb_column` (`lax.scan`; includes a conservative flux-form
host-diffusion stand-in, default 0.05 — see "Dry-regime instability" below).

The distinctive fuller-than-`clubb_lite` features: ADG1 **double-Gaussian**
cloud PDF, parcel buoyant-sorting `Lscale`, and full prognostic moment
transport (`clubb_lite` has none of these).

## CAM-default flag doctrine

Only the call tree selected by the **CAM-default CLUBB flags** is implemented
(`namelist_defaults_cam.xml` `clubb_*` base values; the namelist **overrides**
the CLUBB library defaults in `model_flags.F90` — the namelist is
authoritative). The flags are NOT runtime-configurable: every former flag
branch is hardcoded at its CAM value with a `# CAM <flag> = <value>` site
comment, and the 65 values are recorded in the reference table at the end of
`clubb.py`. `tests/unit/test_clubb_config.py` parses that table and checks it
against the CAM namelist source when the CESM tree is present
(`test_flag_table_matches_cam_namelist_source`), plus table-completeness and
constant-vs-table consistency gates.

Tree-shaping CAM defaults:
- `iiPDF_type = ADG1 (1)`, `saturation_formula = flatau (3)`,
  `penta_solve_method = 1`, `tridiag_solve_method = 1`,
  `grid_remap_method = 1`, `fill_holes_type = 2 (sliding_window)`.
- `l_predict_upwp_vpwp = .false.` → u/v advance via `advance_windm_edsclrm`
  (eddy diffusion), NOT the prognostic upwp/vpwp path.
- `l_diag_Lscale_from_tau = .false.` → `Lscale` via the parcel
  buoyant-sorting path, NOT the tau path.
- `l_call_pdf_closure_twice = .true.`; `l_use_cloud_cover`,
  `l_vert_avg_closure`, `l_trapezoidal_rule_zt/zm`,
  `l_stability_correct_tau_zm`, `l_calc_thlp2_rad`,
  `l_mono_flux_lim_{thlm,rtm,um,vm,spikefix}`, `l_upwind_xpyp_ta`,
  `l_upwind_xm_ma`, `l_tke_aniso` all `.true.`;
  `l_damp_wp2_using_em`, `l_damp_wp3_Skw_squared`, `l_use_C7_Richardson`,
  `l_rcm_supersat_adj` (→ no bisection sat-adjust; cloud water from the PDF),
  `l_advance_xp3` (→ xp3 diagnosed via the LG05 ansatz) all `.false.`.

## Constants, params, saturation

- Physical constants come from `legoesm.constants` (grav→`g`, Cp→`c_pd`,
  Lv→`L_v`, Rd→`R_d`, Rv→`R_v`, ep→`epsilon`, kappa→`kappa`, p0→`p_ref`,
  T_freeze_K→`T_freeze`). CLUBB tunes against slightly different bases
  (~0.1%); documented, below tuning uncertainty.
- Tunable closure coefficients/tolerances are scheme params
  (`CLUBBParams`/`CLUBBConfig`, section 2 of `clubb.py`), NOT constants.
  Defaults are the CLUBB library values with CAM namelist overrides applied
  (CAM-effective; `# lib X -> Y` comments mark overrides, guarded by
  `test_cam_default_param_overrides`).
- Saturation: thin adapters (`sat_mixrat_liq`/`sat_mixrat_ice`) over the
  canonical Flatau (1992) curves in `legoesm.thermo`
  (`saturation_vapor_pressure_flatau[_ice]`) — no re-derivation anywhere.
- The parcel buoyant-sorting mixing length has DIFFERENT numerics from
  `physics/_shared.py`'s Blackadar-style `mixing_length` (used by
  `clubb_lite`); see
  `docs/issues/clubb_parcel_lscale_vs_shared_mixing_length.md`.

## CAM-vs-ARM caveats (CLUBB-JAX reference is ARM-wired)

Re-check the CAM NAMELIST per kernel; the namelist overrides Fortran flag
defaults:
- `l_predict_upwp_vpwp=F` → u/v via `advance_windm_edsclrm`.
- `l_diag_Lscale_from_tau=F` (Fortran default T, namelist F!) → SIMPLE
  `tau=Lscale/√em`, not the `invrs_tau_bkgnd+sfc+shear+N²` path.
- xp2 dp1: **3 distinct C2** (C2rt=C2thl=1.0, C2rtthl=1.3) → per-moment
  LHS+solve; CLUBB-JAX's single-C2 shared-LHS is ARM-only (valid iff all
  equal, F90:836).
- wp2/wp3 UPWIND mean-adv (`l_upwind_xm_ma=T`); xp2/xpyp TA UPWIND but
  xm/wpxp TA CENTERED; `l_damp_wp3_Skw_squared=F` → C8b=0; C6/C7 skewness
  fns NOT Richardson; `l_use_invrs_tau_N2_iso/l_pos_def/
  l_enable_relaxed_clipping=F`.

## Surface BC conventions (`clubb_step`)

- Stress `tau = -ρ·Cd·|V|·u` → `u'w'_sfc = tau_x/ρ` (negative for u>0 = drag).
- Heat/moisture positive-up: `wpthlp_sfc = shflx/(ρ·cp·exner)`,
  `wprtp_sfc = lhflx/(ρ·Lv)`.
- Optional prescribed kinematic fluxes (`sfc_wpthlp`/`sfc_wprtp`/`sfc_upwp`/
  `sfc_vpwp`, the LES/SCM-intercomparison interface) override the bulk
  lower-BC per moment (`None` → bulk; static Python branch). Heat/moisture
  apply EXACTLY (flux-form Neumann; column budget closes to round-off).
  Momentum is magnitude-only by CAM construction
  (`l_imp_sfc_momentum_flux=.true.` consumes `u_*² = √(u'w'² + v'w'²)` and
  re-applies it antiparallel to the wind — the prescribed azimuth is
  discarded; exact for wind-antiparallel bulk drag, but a cross-wind
  momentum-flux vector cannot be imposed).
- Bridge convention: `clubb_step` maps the mean back as `T_new = thlm·exner`,
  so `dT_dt/exner` IS the θl tendency and `q_v = rtm` — prognostic CLUBB
  returns θl/rt, NOT a sat-adjusted (T, q_v) split (cloud partition is
  deferred to microphysics; `l_rcm_supersat_adj=.false.`).

## Validation status

All green on the condensed single-file scheme (2026-06-12):
- Full CLUBB suite: **335 tests** across the 19 per-section files —
  CLUBB-JAX parity (golden `.npz` fixtures in `tests/unit/clubb_fixtures/` +
  `skipif` live bit-exact runs), conservation (rt/θl to round-off; momentum
  to O(Δt) semi-implicit stress; prescribed heat/moisture flux budgets exact;
  sub-cycled flux application), end-to-end `jax.grad` through
  `combined.make_physics` and through CLUBB→sundqvist (moist chain),
  float32/Metal finiteness, MPAS-driver integration (diagnostic + prognostic,
  with wind-input sensitivity + carry persistence checks), GABLS1 stable-BL
  SCM benchmark (`scripts/matrix/scm/gabls1.py --turbulence clubb`),
  convective-BL TKE response, fuller-than-lite wp3 skewness signature.
- Ratchet/contract gates: 4139 passed; turbulence no-regression + carry +
  smoke: 168 passed.
- Testing pattern per kernel: analytic/conservation oracle (CI) + tiny golden
  fixture + live parity vs CLUBB-JAX (constants patched to isolate the
  algorithm) + codex adversarial review. AD-hardened: every sqrt/divide that
  can hit 0 uses the double-where `safe_sqrt`/guard (the MFL section keeps a
  deliberate NaN-PROPAGATING local variant — do not consolidate).

### Dry-regime instability (resolved)

A bare SCM driver advancing means with CLUBB alone exposes 2Δz noise a
coupled dycore damps — NOT a closure bug (conservation + per-piece parity
hold). `integrate_clubb_column` therefore includes a conservative flux-form
host-diffusion stand-in (`host_numerical_diffusion`, default 0.05); the
coupled path relies on the real dycore. Characterizer:
`scripts/validate/clubb_prognostic_stability.py`.

## Known limitations / future work

- **Production AMIP-CLI prognostic exposure is blocked** on the MPAS run
  loop not carrying stateful physics across steps — see
  `docs/issues/mpas_run_loop_stateful_physics_carry.md`. Prognostic CLUBB
  remains runnable/tested via the direct `combined.make_physics` MPAS path
  and `integrate_clubb_column`.
- SCM `prescribe="fluxes"` injects fluxes as a host mean-tendency, so CLUBB's
  native `wpthlp_sfc`/`wprtp_sfc` moment BC stays 0 on that path. The
  scheme-side prescribed-flux interface is done; routing the SCM
  `w_th_s`/`w_qv_s` into it is cross-cutting (`PhysicsState` override fields
  + the shared `turb_fn` dispatch signature) → human-directed PR.
- Optional: BOMEX/DYCOMS profiles; long coupled run; thread `phys_state`
  through nonhydro/spectral_pe to lift their fail-fast; edsclr
  (passive-scalar) transport through CLUBB.

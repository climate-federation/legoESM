# PORT_CLUBB — Porting a fuller CLUBB turbulence scheme into legoESM

**Goal:** Create `packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py`
(plus `clubb_*.py` helper modules in the same directory) implementing a
*substantially fuller* version of CLUBB than the existing `clubb_lite.py`,
ported from `../CLUBB-JAX/clubb_jax/src/CLUBB_core`, restricted to the call tree
exercised by **CAM default `clubb_*` namelist flags**
(`../../CESM/components/cam/bld/namelist_files/namelist_defaults_cam.xml`).
Must follow legoESM conventions (CLAUDE.md, CONTRIBUTING.md, FEDERATION.md):
end-to-end `jax.grad`/JIT/pytree safe, constants from `legoesm.constants`,
saturation from `legoesm.thermo`, physics contracts, ≥1 unit test per new `.py`,
codex adversarial review on substantial changes, correct file layout.

**Definition of DONE:** legoESM can be *run and tested* with
`TurbulenceConfig(scheme="clubb")` selecting the new scheme through the standard
`make_turbulence_physics(...)` dispatch, producing a valid `TurbulenceOutput`,
with passing unit + integration tests and an idealized-case sanity check.

---

## Source-of-truth facts (verified iter 1)

- **CLUBB-JAX is already a JAX port** (imports `clubb_jax.src...`, uses
  `jax.numpy`). So this is a *port-and-adapt to legoESM conventions*, not a
  Fortran→Python translation. Key tensions to resolve: it depends on numpy +
  `tracer_numpy` shims, Fortran-style `gr` grid objects, its own
  `constants_clubb.py`, and a ~50-file module graph with a giant keyword
  interface (`advance_clubb_core` has ~120 kwargs).
- **CAM default ConfigFlags** (from `model_flags.py:get_default_config_flags`,
  which mirrors `set_default_clubb_config_flags_api`):
  - `iiPDF_type = iiPDF_ADG1 (1)` → **Analytic Double Gaussian 1** PDF closure.
  - `ipdf_call_placement = ipdf_post_advance_fields (2)`.
  - `saturation_formula = saturation_flatau (3)`.
  - `penta_solve_method = penta_lu (2)`, `tridiag_solve_method = tridiag_lu (2)`.
  - `grid_remap_method = ppm (2)`, `grid_adapt_in_time_method = none (0)`,
    `fill_holes_type = sliding_window (2)`.
  - Notable logicals ON: `l_use_precip_frac`, `l_predict_upwp_vpwp`,
    `l_min_xp2_from_corr_wx`, `l_calc_thlp2_rad`, `l_upwind_xpyp_ta`,
    `l_upwind_xm_ma`, `l_tke_aniso`, `l_fix_w_chi_eta_correlations`,
    `l_damp_wp2_using_em`, `l_diag_Lscale_from_tau`, `l_use_C7_Richardson`,
    `l_rcm_supersat_adj`, `l_damp_wp3_Skw_squared`,
    `l_use_tke_in_wp3_pr_turb_term`, `l_mono_flux_lim_{thlm,rtm,um,vm,spikefix}`,
    `l_wp2_fill_holes_tke`.
  - Notable logicals OFF (prune these branches): `l_ho_*_coriolis`,
    `l_C2_cloud_frac`, `l_diffuse_rtm_and_thlm`, `l_call_pdf_closure_twice`,
    `l_standard_term_ta`, `l_godunov_upwind_*`, `l_use_cloud_cover`,
    `l_diagnose_correlations`, `l_calc_w_corr`, `l_const_Nc_in_cloud`,
    `l_do_expldiff_rtm_thlm`, `l_e3sm_config`, `l_add_dycore_grid`,
    `l_brunt_vaisala_freq_moist` (CAM default), `l_advance_xp3`.
  - **⚠ CRITICAL (iter 1): the CAM namelist OVERRIDES the CLUBB library
    defaults.** The task says to set flags "as in the defaults used by CAM" →
    the `namelist_defaults_cam.xml` values are AUTHORITATIVE, not
    `get_default_config_flags()`. Confirmed conflicts (CAM value wins):
    | flag | library default | **CAM default** |
    |------|-----------------|-----------------|
    | `l_predict_upwp_vpwp` | True | **False** |
    | `l_use_cloud_cover` | False | **True** |
    | `l_use_C7_Richardson` | True | **False** |
    | `l_vert_avg_closure` | False | **True** |
    | `l_trapezoidal_rule_zt` | False | **True** |
    | `l_trapezoidal_rule_zm` | False | **True** |
    | `l_stability_correct_tau_zm` | False | **True** |
    | `l_rcm_supersat_adj` | True | **False** |
    | `l_use_tke_in_wp3_pr_turb_term` | True | **False** |
    | `penta_solve_method` | 2 (penta_lu) | **1 (penta_band/LR)** |
    | `tridiag_solve_method` | 2 (tridiag_lu) | **1** |
    These change which branches are live: e.g. `l_predict_upwp_vpwp=False` →
    momentum advanced via `advance_windm_edsclrm` (eddy-diffusion), NOT the
    `upwp/vpwp` prognostic path; `l_vert_avg_closure=True` + trapezoidal rules
    ON change the PDF-closure integration; `l_use_cloud_cover=True` activates
    the cloud-cover diagnosis. Build the call tree from the CAM table below.
  - Flags matching in both (CAM-confirmed ON): `l_use_precip_frac`,
    `l_min_xp2_from_corr_wx`, `l_calc_thlp2_rad`, `l_upwind_xpyp_ta`,
    `l_upwind_xm_ma`, `l_tke_aniso`, `l_fix_w_chi_eta_correlations`,
    `l_damp_wp2_using_em`, `l_diag_Lscale_from_tau`, `l_damp_wp3_Skw_squared`,
    `l_mono_flux_lim_{thlm,rtm,um,vm,spikefix}`, `l_wp2_fill_holes_tke`,
    `fill_holes_type=2`. CAM-confirmed OFF: `l_ho_*_coriolis`, `l_C2_cloud_frac`,
    `l_diffuse_rtm_and_thlm`, `l_godunov_upwind_*`, `l_standard_term_ta`,
    `l_diagnose_correlations`, `l_calc_w_corr`, `l_const_Nc_in_cloud`,
    `l_do_expldiff_rtm_thlm`, `l_e3sm_config`, `l_add_dycore_grid`,
    `l_use_C11_Richardson`, `l_use_shear_Richardson`, `l_lmm_stepping`,
    `l_lscale_plume_centered`, `l_partial_upwind_wp3`,
    `l_smooth_Heaviside_tau_wpxp`, `l_stability_correct_Kh_N2_zm`,
    `l_use_thvm_in_bv_freq`, `l_use_tke_in_wp2_wp3_K_dfsn`,
    `l_vary_convect_depth`, `l_prescribed_avg_deltaz`, `l_brunt_vaisala_freq_moist`.
  - `iiPDF_type` not in namelist → stays library default **ADG1 (1)**.
    `saturation_formula` not in namelist → library default **flatau (3)**.
  - **CAM-default tunable C-coefficients / parameters** (base, non-silhs,
    non-cam7) → `CLUBBConfig` field defaults (P1):
    `C1=1.0, C1b=1.0, C2rt=1.0, C2thl=1.0, C2rtthl=1.3, C4=5.2, C6rt=4.0,
    C6rtb=6.0, C6rtc=1.0, C6thl=4.0, C6thlb=6.0, C6thlc=1.0, C7=0.5, C7b=0.5,
    C8=4.2, C8b=0.0, C11=0.7, C11b=0.35, C14=2.2, C_uu_buoy=0.3, C_uu_shr=0.3,
    C_wp2_splat=0.0, C_wp3_pr_turb=0.4, beta=2.4, gamma_coef/…, c_K1=0.75,
    c_K2=0.125, c_K8=1.25, c_K9=0.25, c_K10=0.5, c_K10h=0.3, nu2=5.0, nu9=20.0,
    mult_coef=1.0, lambda0_stability_coef, lmin_coef=0.1, skw_max_mag=4.5,
    Skw_denom_coef=0.0, up2_sfc_coef=2.0, wpxp_L_thresh=60.0, wpxp_Ri_exp=0.5,
    z_displace=25.0, bv_efold=5.0, C_invrs_tau_bkgnd=1.0, C_invrs_tau_N2=0.1,
    C_invrs_tau_N2_wp2=0.2, C_invrs_tau_N2_xp2=0.2, C_invrs_tau_N2_wpxp=0.0,
    C_invrs_tau_N2_clear_wp3=0.0, C_invrs_tau_sfc=0.1, C_invrs_tau_shear=0.02}`.
    `clubb_timestep=300s`. **cam7 deltas** (document, not default):
    `C7=0.1, C8=4.6, C_uu_shr=0.1, c_K10h=0.280`. Pull remaining
    `gamma_coef*`/`lambda0_stability_coef` numeric values in P1 from the
    namelist (some are variant-gated).
- **CLUBB constants → legoESM mapping** (CLAUDE.md: constants only from
  `legoesm.constants`):
  | clubb | value | legoesm.constants |
  |-------|-------|-------------------|
  | grav  | 9.81  | `g` (=9.80616) |
  | Cp    | 1004.67 | `c_pd` |
  | Lv    | 2.5e6 | `L_v` |
  | Rd    | 287.04 | `R_d` |
  | Rv    | 461.5 | `R_v` |
  | ep    | Rd/Rv | `epsilon` |
  | kappa | Rd/Cp | `kappa` |
  | p0    | 1e5   | `p_ref` |
  | T_freeze_K | 273.15 | `T_freeze` |
  | Lf/Ls | 3.33e5/2.834e6 | (check constants; add if missing) |
  | stefan_boltzmann | 5.6704e-8 | `sigma_sb` |
  | omega_planet | 7.292e-5 | (Coriolis is host-provided; CAM clubb fcor passed in) |

  *Numerical-fidelity note:* CLUBB tunes against grav=9.81, not 9.80616. The
  ~0.04% difference is below CLUBB's own tuning uncertainty; per CLAUDE.md we use
  `constants.g`. Documented; revisit only if an oracle tendency-match fails.
  - **Tolerances + tunable C-coefficients** (`w_tol`, `rt_tol`, `C1`, `C2*`,
    `C6*`, `C7*`, `C8*`, `C11*`, `C14`, `gamma_coef`, `c_K*`, `nu*`, `beta`,
    `lambda0_stability_coef`, `mult_coef`, ...) are **scheme-tunable params** →
    `CLUBBConfig` NamedTuple fields, NOT `legoesm.constants`. Defaults reference
    the CAM namelist values. (CLAUDE.md: tunable scheme params live in `*Config`.)

- **CLUBB staggered grid:** momentum levels `zm` (nzm) + thermodynamic levels
  `zt` (nzt), ascending (index 0 = lowest). Interp/derivative operators in
  `grid_class.py` (`zm2zt`, `zt2zm`, `ddzm`, `ddzt`, smoothers). legoESM uses
  full/half levels, top-down ordered. The bridge (orientation flip + grid
  construction from `z_full`/`z_half`) is *harness/adapter glue* → lives in the
  clubb adapter, verified by an equivariance/round-trip test.

- **Existing dispatch:** `integration.py:get_turbulence_fn` is an
  `if/elif/.../else: raise ValueError` chain; `TurbulenceConfig.scheme` is a
  `Literal[...]`. Adding `"clubb"` requires: new branch, `Literal` entry,
  `CLUBBConfig` field on `TurbulenceConfig`, `needs_tke` membership update,
  and `ExperimentConfig.validate_strict` membership set (audit rule).

- **`TurbulenceOutput`** (output.py) currently carries du_dt, dv_dt, dT_dt,
  dq_v_dt, Km, Kh, shflx, lhflx, ustar, h_pbl. CLUBB natively produces moment
  tendencies + cloud fraction + many diagnostics. Decision (iter-2): keep the
  `TurbulenceOutput` contract; CLUBB advances its moments internally as carried
  state (like `tke` slot) and exposes du/dv/dT/dq tendencies + Km/Kh-equivalent
  diagnostics. Cloud fraction wiring deferred (clubb_lite docstring documents the
  `TurbulenceOutput` has no cloud_fraction field yet).

---

## Architecture decision

Single monster file is against legoESM hygiene. Mirror CLUBB's modularity with
`clubb_*.py` helpers in `physics/turbulence/`, with `clubb.py` as the entry:

- `clubb.py`            — entry `clubb_turbulence(...)`, state pack/unpack, the
                          legoESM↔CLUBB grid/orientation bridge, orchestration.
- `clubb_grid.py`       — staggered-grid operators (zm2zt/zt2zm/ddzm/ddzt/smooth).
- `clubb_config.py` or extend `config.py` — `CLUBBConfig` + `CLUBBFlags`
                          (CAM-default static booleans) + tunable C-coefficients.
- `clubb_pdf.py`        — ADG1 PDF closure (`pdf_closure_module` + ADG1 path of
                          `setup_clubb_pdf_params`/`adg1_adg2_3d_luhar_pdf`).
- `clubb_mixing_length.py` — `Lscale` from buoyant-sorting integrals
                          (`mixing_length.py`, `l_diag_Lscale_from_tau` path).
- `clubb_moments.py`    — advance wp2/wp3, xp2/xpyp, xm/wpxp, windm
                          (the `advance_*` modules), tridiagonal implicit solves.
- `clubb_saturation.py` — thin adapter to `legoesm.thermo` (Flatau if available).

Each new `.py`: docstring, `__physics_contract__` where it is a physics scheme
module (or classification in `test_physics_contracts.py`), ≥1 unit test.

**Reuse audit (pre-impl-search, ongoing):** before each helper, grep legoESM for
existing equivalents — `_shared.py` (mixing_length, virtual_temperature),
`timestepping/tridiagonal.py` (tridiagonal solve!), `thermo.py` (saturation),
`diagnostics/`. Reuse/extend rather than re-derive. (e.g. CLUBB's
penta/tridiag LU solvers vs `timestepping/tridiagonal.py` — reconcile iter-3.)

---

## Phased plan (each phase = several Ralph iterations; codex review per phase)

- **P0 Foundations** *(in progress)*: plan, flag/constant table, grid operators
  + tests, constants mapping. Bridge legoESM grid → CLUBB `gr`.
- **P1 Config**: `CLUBBConfig`/`CLUBBFlags` NamedTuples w/ CAM-default values;
  wire (dormant) into `TurbulenceConfig` Literal + validate_strict; stub
  `clubb_turbulence` raising `NotImplementedError` is FORBIDDEN (CLAUDE.md) —
  instead land it last when the call tree is real.
- **P2 Thermo/saturation adapter** + sigma_sqd_w + Brunt-Vaisala (helper_module).
- **P3 Mixing length** (`Lscale`, tau). ⚠ CAM `l_diag_Lscale_from_tau=False`
  ⇒ buoyant-sorting **parcel-integral** `Lscale` (NOT the tau path). Split:
  skewness ✅ iter 5; `compute_mixing_length` parcel asc/desc ⏳ iter 6+.
- **P4 ADG1 PDF closure** — the heart; `pdf_closure_module` ADG1 branch +
  `setup_clubb_pdf_params`. Cloud fraction, wpthvp, buoyancy terms.
- **P5 Moment advance** — wp2/wp3, xp2/xpyp, xm/wpxp, windm; implicit solves;
  clipping (`clip_explicit`), mono flux limiters (CAM defaults ON), fill_holes.
- **P6 Orchestration** — assemble `advance_clubb_core`-equivalent for the
  CAM-default flag subset; pack/unpack carried moment state.
- **P7 Integration** — dispatch wiring, `TurbulenceOutput`, carried-state slots
  in `integration.py` (extend `needs_tke`-style moment carrying).
- **P8 Validation** — unit tests per module, single-column idealized cases
  (BOMEX/DYCOMS-style if feasible), conservation/positivity, AD smoke test
  (`jax.grad` through one step), visual/diagnostic sanity, ocean/atm matrix not
  applicable but turbulence-relevant tests. Codex adversarial review clean.

---

## Status tracker

| Phase | Item | Status |
|-------|------|--------|
| P0 | Branch `port-clubb` created | ✅ iter 1 |
| P0 | PORT_CLUBB.md plan | ✅ iter 1 |
| P0 | CAM-default flag survey + library-vs-namelist conflict table | ✅ iter 1 |
| P0 | Constants mapping table | ✅ iter 1 |
| P0 | `clubb_grid.py` staggered operators + 16 tests + codex-approved | ✅ iter 1 |
| P0 | Authoritative namelist flag/param diff | ✅ iter 1 |
| P0 | Working `.venv` (uv sync from lockfile) | ✅ iter 1 |
| P0 | legoESM↔CLUBB grid bridge (z_full/z_half → CLUBBGrid) | ✅ iter 2 |
| P1 | `CLUBBConfig`/`CLUBBFlags`/`CLUBBParams` + CAM defaults + tests | ✅ iter 2 |
| P1 | Wire `"clubb"` into TurbulenceConfig Literal + validate_strict | ☐ (deferred to P7 — needs scheme entry to avoid half-wired dispatch) |
| P1 | `CLUBBConfig`/`CLUBBFlags` | ☐ |
| P2 | saturation adapter (Flatau) — `thermo` curves + `clubb_saturation.py` | ✅ iter 3 |
| P2 | sigma_sqd_w + Brunt–Väisälä (`clubb_helpers.py`) | ✅ iter 4 |
| P3 | skewness diagnostics (`clubb_skewness.py`: Skx, gamma_Skw, LG05) | ✅ iter 5 |
| P3 | `compute_mixing_length` parcel buoyant-sorting (Lscale up/down) | ✅ iter 6 (golden-locked vs CLUBB-JAX) |
| P3 | mixing length / Lscale | ☐ |
| P4 | ADG1 PDF closure | ☐ |
| P5 | moment advance + solves + limiters | ☐ |
| P6 | orchestration | ☐ |
| P7 | integration/dispatch wiring | ☐ |
| P8 | validation + codex-clean | ☐ |

---

## Iteration log

### iter 1
- Created branch `port-clubb`.
- Surveyed source tree: CLUBB-JAX is an existing JAX port; mapped the
  `advance_clubb_core` interface, CAM-default ConfigFlags, constants, and the
  staggered-grid operators.
- Wrote this plan with constants mapping, flag survey, architecture, phasing.
- **Discovered + documented that the CAM namelist OVERRIDES the CLUBB library
  flag defaults** (11+ conflicts) — the namelist is authoritative per the task.
  Built the authoritative CAM-default flag + tunable-coefficient table.
- Built a working `.venv` via `uv sync` from the lockfile (jax 0.10, equinox,
  etc.) — `import legoesm` OK; pytest runs. (No `.venv` existed before.)
- Implemented `clubb_grid.py` (P0 foundation): `CLUBBGrid` pytree +
  `zm2zt`/`zt2zm`/`ddzm`/`ddzt` + smoothers + `make_clubb_grid`, faithful to
  `CLUBB_core/grid_class.py` (operators) and `derived_types/grid_class.py`
  (spacing construction). Pure pytree fns, no module-scope JIT.
- Added `tests/unit/test_clubb_grid.py` (16 tests: linear-field exactness,
  derivative slopes, round-trip smoothers, non-midpoint lower-boundary parity,
  safe-divide on zero spacing, structural validation, JIT+grad smoke). All pass
  (`JAX_ENABLE_X64=1`).
- Ran `/codex:adversarial-review` (mandatory). Round 1 → 2 medium findings
  (lower-boundary `dzm` should be `2*(zt[0]-zm[0])` not interior-copy; add
  zero-spacing guard + structural validation). Fixed both + added parity/negative
  tests. Round 2 → **APPROVE, no material findings.**
- **Next (iter 2):** legoESM↔CLUBB grid bridge in `clubb.py` (build `CLUBBGrid`
  from legoESM `z_full`/`z_half` with the top-down→ascending flip; round-trip
  equivariance test), then P1 `CLUBBConfig`/`CLUBBFlags` with the CAM-default
  values tabulated above.

### iter 2
- **Grid bridge (P0 done):** added `make_clubb_grid_from_levels(z_full, z_half)`
  + `flip_vertical` to `clubb_grid.py`. Verified legoESM ordering is TOP-DOWN
  with `z_half[:, -1] == 0` (surface); mapping is `zt ↔ z_full` (nlev),
  `zm ↔ z_half` (nlev+1), flipped to ascending. 4 new tests incl. an
  interpolation-equivariance test (flip∘interp∘flip == direct) and staggering
  (zt at zm midpoints, surface zm[0]=0). 20 grid tests pass.
- **Config (P1 done):** new `clubb_config.py` with `CLUBBFlags` (static),
  `CLUBBParams` (tunable, dynamic/differentiable leaves), `CLUBBConfig`
  (bundles flags+params+surface+tolerances+clubb_dt=300s). 54 tests pass.
- **⚠ CORRECTED a critical flag-transcription error** (caught by codex +
  re-extraction): I had wrongly fallen back to CLUBB *library* defaults for
  flags I hadn't initially grepped. Re-extracted ALL `clubb_*` base defaults
  from the namelist. Fixes vs my first draft:
  `grid_remap_method 2→1`, `l_call_pdf_closure_twice False→True`,
  `l_damp_wp2_using_em True→False`, `l_damp_wp3_Skw_squared True→False`,
  `l_diag_Lscale_from_tau True→False`. Added `l_ascending_grid`, `l_c14_ml`,
  `l_intr_sfc_flux_smooth`. **Big call-tree implications:**
  `l_diag_Lscale_from_tau=False` ⇒ Lscale via the **buoyant-sorting integral**
  path (NOT the tau path) — reshapes P3; `l_call_pdf_closure_twice=True` ⇒ PDF
  closure runs **twice** (pre+post) — reshapes P4/P6.
- Added a **source-derived test** (`test_flags_match_cam_namelist_source`) that
  parses the XML and compares — non-vacuous, skips if CESM tree absent (CI-safe).
- **Static/dynamic boundary (codex finding):** `CLUBBFlags` is now a frozen
  dataclass registered via `jax.tree_util.register_static` ⇒ zero dynamic JAX
  leaves, branchable with Python `if` under jit. Config is closure-captured
  (it also carries a `str` leaf `surface.bulk_scheme`), matching the legoESM
  scheme-config pattern. `CLUBBParams` floats remain dynamic leaves (so the
  coefficients are differentiable for training).
- Added scoped `[tool.ruff.lint.per-file-ignores]` (N803/N806/N815) for
  `clubb_*.py` — canonical CLUBB symbol names mirror the reference 1:1.
- Classified `clubb_config.py` EXCLUDED in physics-contracts. Regression:
  existing `test_physics_turbulence.py` (42) + contracts still green.
- Codex adversarial review: round 1 → 2 findings (flag transcription [high];
  static-leaf flags [medium]) → fixed → round 2 **APPROVE, no material findings**
  (codex independently diffed the XML vs the config, 0 mismatches).
- **Next (iter 3):** P2 saturation/thermo adapter (`clubb_saturation.py` →
  `legoesm.thermo`, Flatau formula), `sigma_sqd_w`, and Brunt-Väisälä
  (`advance_helper_module`), each with tests + codex review.

### iter 3
- **P2 saturation (done):** CAM default is `saturation_formula = flatau` and
  `l_rcm_supersat_adj = .false.` (so the bisection `rcm_sat_adj` is NOT in the
  tree — CLUBB diagnoses cloud water from the PDF, deferred to P4).
- Added the **canonical Flatau SVP curves to `legoesm.thermo`**
  (`saturation_vapor_pressure_flatau` liquid + `..._ice_flatau`), coefficients
  verbatim from CLUBB-JAX `saturation.py`/`saturation.F90`, with the deg-C clip
  floors (−85 liquid, −90 ice). Putting them in the shared thermo module keeps
  the port compliant with the CLAUDE.md "saturation only from `thermo`" rule
  (curve NOT re-derived inside the physics tree). Additive — no change to the
  Tetens default or existing consumers.
- New `clubb_saturation.py`: thin adapter `sat_mixrat_liq`/`sat_mixrat_ice`
  assembling `rsat = ε·esat/(p−esat)` with CLUBB's AD-safe denominator guard
  (`where(safe, p−esat, 1)` before divide; `rsat = ε` fallback when `p−esat<1`).
- Tests (`test_clubb_saturation.py`, 17): Flatau golden anchors (computed +
  pinned), positivity/monotonicity, ice<liquid sub-freezing, Flatau≈Tetens
  within 3% over 240–310 K, mixing-ratio definition match, ε-fallback, and
  **finite reverse-mode gradient through the fallback region**, JIT-clean.
  `test_no_saturation_reimpl` ratchet still green (curve lives in `thermo`).
- Classified `clubb_saturation.py` EXCLUDED in physics-contracts. Extended the
  scoped ruff per-file-ignore to `tests/unit/test_clubb_*.py` (physical symbol
  names T/Tc). CI gate is `E9,F63,F7,F82` only — N-rules are advisory.
- Codex adversarial review: **APPROVE, no material findings** (confirmed the
  pre-division guard, ε fallback, CAM flatau + rcm-adj-off, additive thermo).
- **Next (iter 4):** `clubb_saturation` is ready; do `sigma_sqd_w`
  (`sigma_sqd_w_module`) + Brunt–Väisälä `calc_brunt_vaisala_freq_sqd`
  (`advance_helper_module`) — both feed the mixing-length/PDF; tests + codex.

### iter 4
- **P2 helpers (done):** new `clubb_helpers.py` (analog of CLUBB
  `advance_helper_module`) with two CAM-default-tree kernels:
  - `compute_sigma_sqd_w` — PDF width parameter
    `gamma_Skw*(1−min(max_x corr_wx²,1))`, smoothed zm→zt→zm with a zero floor.
    CAM `l_predict_upwp_vpwp=False` ⇒ only rt/thl w-correlations enter
    `max_corr` (up/v branch pruned, documented). Tolerances come from
    `CLUBBConfig` (no hardcoded magic numbers).
  - `calc_brunt_vaisala_freq_sqd` — CAM defaults (`l_use_thvm_in_bv_freq=F`,
    `l_brunt_vaisala_freq_moist=F`, `l_modify_limiters_for_cnvg_test=F`):
    returns dry `(g/T0)·d(thlm)/dz`; still computes the moist/mixed/smoothed
    forms (downstream mixing-length/Ri consume them). Uses
    `clubb_saturation.sat_mixrat_liq` + `clubb_grid` operators.
- Constants via `legoesm.constants` (g/c_pd/L_v/R_d/epsilon) — NOT CLUBB
  globals; ~0.1% tuning-scale difference documented. (PreToolUse hook caught a
  `9.80616` literal in a docstring; reworded to reference constants by name.)
- Tests (`test_clubb_helpers.py`, 9): sigma bounds/zero-flux=gamma/high-corr→0;
  BV dry-form identity, stable→positive, unstable→negative,
  `ice=0 ⇒ bv_mixed==bv_dry` exact relation; both JIT+grad clean.
- Classified `clubb_helpers.py` EXCLUDED in physics-contracts. Switched the
  ruff per-file-ignore to a `clubb*.py` glob (covers all CLUBB modules).
- Codex adversarial review: **APPROVE, no material findings** (compared
  directly against the CLUBB-JAX + Fortran references for the CAM branches).
- **Next (iter 5):** P3 mixing length / `Lscale`. CAM `l_diag_Lscale_from_tau
  = False` ⇒ the **buoyant-sorting parcel-integral** `Lscale` path in
  `mixing_length.py` (NOT the tau path). This is the largest single kernel so
  far — likely split across iter 5–6. Will need `gamma_Skw`/`Skx`
  (`Skx_module`) which sigma_sqd_w consumes too.

### iter 5
- **P3 skewness (done):** new `clubb_skewness.py` ports `Skx_module.py`:
  `Skx_func`, `compute_gamma_Skw` (Gaussian γ(Skw) with the degenerate-coef
  `jnp.where` guard + static `l_gamma_Skw` branch), `LG_2005_ansatz` (LG05 eqs
  11/16/33), `xp3_LG_2005_ansatz` (inverse of `Skx_func`). Feeds
  `compute_sigma_sqd_w` (γ_Skw), the PDF closure, and the diagnostic `xp3`
  (CAM `l_advance_xp3=False`).
- **Convention adaptation:** the CLUBB ref indexes `clubb_params[:, iX]` out of
  the 102-vector; I pass the coefficients as named scalar args from
  `CLUBBParams` (`Skw_denom_coef`, `gamma_coef`/`b`/`c`, `beta`). `w_tol_sqd`
  floor implemented as `w_tol**2` from config. Codex verified the
  index→name mapping is faithful.
- Tests (`test_clubb_skewness.py`, 10): Skx normalization + odd symmetry; γ(Skw)
  limits (Skw=0→γ_coef, large→γ_coefb, bounded, degenerate→constant, flag-off→
  constant, golden); LG05 zero-flux→0; **xp3∘Skx_func round-trip** recovers the
  LG05 skewness; differentiable wrt coefficients; JIT-clean.
- `clubb_skewness.py` EXCLUDED in physics-contracts; per-file-ignore glob gains
  N802 (canonical fn names `Skx_func`, `LG_2005_ansatz`).
- Codex adversarial review: **APPROVE, no material findings** (formulas, param
  mapping, w_tol_sqd floor, inverse, AD branches all verified vs Skx_module.py).
### iter 6
- **P3 mixing length (done, golden-locked):** new `clubb_mixing_length.py` ports
  the full parcel buoyant-sorting `Lscale` (CAM `l_diag_Lscale_from_tau=False`
  path): `_parcel_thv`, `_upward/_downward_inner_while`,
  `_compute_lscale_up_col/_down_col`, `_bounded_while`, `set_Lscale_max`,
  `compute_mixing_length`. legoESM adaptations: `legoesm.constants`,
  `clubb_saturation` (Flatau ⇒ `saturation_formula` arg dropped), `clubb_grid`
  operators, and **`vmap` over columns** (CLAUDE.md, vs the ref's per-column
  Python loop) with each column using its own grid; the dynamic-trip parcel
  `while`s are grad-safe fixed-length `lax.scan` (`_bounded_while`).
- Extended `CLUBBGrid` with `dzm`/`dzt` (parcel kernel needs the spacings);
  updated the grid pytree-leaf test (4→6).
- **GOLDEN PARITY vs the CLUBB-JAX reference:** verified `compute_mixing_length`
  is **bit-exact** (0.0 diff) to `CLUBB-JAX/.../mixing_length.py` once the
  reference's constants+saturation are patched to the legoESM values — proving
  the algorithm is faithful and the only difference is the intended ~0.1%
  constant set. Committed a tiny golden fixture
  `tests/unit/clubb_fixtures/clubb_lscale_golden.npz` (2.8 KB; CLAUDE.md
  tiny-numeric-baseline carve-out, outside the gitignored `data/`) +
  **non-skipped** CI test `test_matches_committed_golden` (assert_array_equal),
  with the live reference test (skipped if sibling absent) re-verifying the
  golden stays in sync. Fixture spans neutral(boundary-exit) /
  strongly-stable(early-exit) / moist-saturated(condensing) columns.
- Tests (8): set_Lscale_max, shapes/positivity, Lscale_max cap,
  neutral≫stable, monotone-stability, JIT+grad, golden, live-parity.
  `clubb_mixing_length.py` EXCLUDED in physics-contracts.
- Codex adversarial review: initial pass flagged [high] "no oracle test" →
  added the bit-exact golden parity → re-review flagged the fixture/test were
  still uncommitted (resolved by this commit). Algorithm faithfulness confirmed
  by the 0.0-diff parity.
- **Next (iter 7):** P4 ADG1 PDF closure (`pdf_closure_module` ADG1 branch +
  `setup_clubb_pdf_params`) — the cloud-fraction / buoyancy-flux heart; needs
  `sigma_sqd_w`(✓), `Skx`/`gamma_Skw`(✓), `sat_mixrat_liq`(✓). Largest remaining
  kernel after this.

<!-- superseded risk note (resolved iter 6):
- **⚠ Remaining P3 risk (iter 6+):** `compute_mixing_length` is the largest,
  hardest kernel — per-column parcel **while-loop** ascents/descents
  (`_compute_lscale_up_col`/`_compute_lscale_down_col`, `_bounded_while`,
  `_upward_inner_while`) doing buoyant-sorting CAPE integrals, ~600 LOC. The
  ref uses a Python `for i in range(ngrdcol)` over columns + dynamic-indexed
  `while`. Porting JAX-clean (lax.while_loop/scan, differentiable, vmap or
  per-column) is the single biggest remaining task; will span ≥2 iterations and
  needs its own careful codex pass. Not started — flagged, not skipped.
-->


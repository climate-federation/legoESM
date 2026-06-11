# PORT_CLUBB — Porting a fuller CLUBB turbulence scheme into legoESM

**Goal:** `packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py`
(+ `clubb_*.py` helpers) = a substantially fuller CLUBB than `clubb_lite.py`,
ported from `../CLUBB-JAX/clubb_jax/src/CLUBB_core`, restricted to the call tree
exercised by the **CAM-default `clubb_*` flags**
(`../../CESM/components/cam/bld/namelist_files/namelist_defaults_cam.xml`).
Conventions: CLAUDE.md / CONTRIBUTING.md / FEDERATION.md — end-to-end
`jax.grad`/JIT/pytree, constants from `legoesm.constants`, saturation from
`legoesm.thermo`, physics contracts, ≥1 unit test per `.py`, codex adversarial
review per substantial change.

**DONE when:** legoESM runs+tests with `TurbulenceConfig(scheme="clubb")` through
`make_turbulence_physics(...)`, producing a valid `TurbulenceOutput`, with
passing unit+integration tests and an idealized-case sanity check.

*(Compressed at iter 10/20/30/40; verbose per-iter logs dropped — see git
history. The reference facts + module table + status below are the live source
of truth.)*

---

## Key facts (verified, iter 1–9)

- **CLUBB-JAX is already a JAX port** (not Fortran) — this is port-and-adapt to
  legoESM conventions. Reference lives at
  `/glade/work/adac/Claude/CLUBB-JAX/clubb_jax/src/CLUBB_core`.
- **Env:** `.venv` built via `uv sync` (jax 0.10). Tests:
  `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`. Live golden-parity
  tests need `PYTHONPATH=/glade/work/adac/Claude/CLUBB-JAX` (they self-bootstrap
  `sys.path` and `skipif` when the sibling tree is absent → CI-safe).
- **Constants → `legoesm.constants`** (CLAUDE.md): grav→`g`, Cp→`c_pd`,
  Lv→`L_v`, Rd→`R_d`, Rv→`R_v`, ep→`epsilon`, kappa→`kappa`, p0→`p_ref`,
  T_freeze_K→`T_freeze`. CLUBB tunes against slightly different bases (~0.1%);
  documented, below tuning uncertainty. Tunable C-coeffs/tolerances are
  **scheme params** (`CLUBBParams`/`CLUBBConfig`), NOT constants.
- **CAM namelist OVERRIDES the CLUBB library flag defaults** — the namelist is
  authoritative. Tree-shaping CAM-default flags (the ones that pick branches):
  - `iiPDF_type = ADG1 (1)`, `saturation_formula = flatau (3)`,
    `penta_solve_method = 1`, `tridiag_solve_method = 1`,
    `grid_remap_method = 1`, `fill_holes_type = 2 (sliding_window)`.
  - `l_predict_upwp_vpwp = .false.` → u/v advance via `advance_windm_edsclrm`
    (eddy diffusion), NOT the prognostic upwp/vpwp path.
  - `l_diag_Lscale_from_tau = .false.` → Lscale via the **parcel buoyant-sorting**
    path (done, iter 6), NOT the tau path.
  - `l_call_pdf_closure_twice = .true.` → PDF closure runs twice (pre+post).
  - `l_use_cloud_cover = .true.`, `l_vert_avg_closure = .true.`,
    `l_trapezoidal_rule_zt/zm = .true.`, `l_stability_correct_tau_zm = .true.`,
    `l_calc_thlp2_rad = .true.`, `l_mono_flux_lim_{thlm,rtm,um,vm,spikefix} =
    .true.`, `l_upwind_xpyp_ta = .true.`, `l_upwind_xm_ma = .true.`,
    `l_tke_aniso = .true.`, `l_damp_wp2_using_em = .false.`,
    `l_damp_wp3_Skw_squared = .false.`, `l_use_C7_Richardson = .false.`,
    `l_rcm_supersat_adj = .false.` (→ no bisection sat-adjust; cloud water from
    PDF), `l_advance_xp3 = .false.` (xp3 diagnosed via LG05 ansatz).
  - Authoritative numeric flag/param values: re-extract from the namelist with
    the iter-1 grep recipe; `CLUBBFlags`/`CLUBBParams` encode them + a
    `test_flags_match_cam_namelist_source` source-of-truth check.

---

## Architecture (modules under `physics/turbulence/`)

`clubb.py` = entry `clubb_turbulence(...)` + legoESM↔CLUBB bridge + orchestration.
**RUNNABLE as of iter 16** (phase 1): `scheme="clubb"` dispatches through
`make_turbulence_physics`, runs end-to-end, JIT/grad-clean, wired into the AMIP
CLI + coupler `validate_strict` + `physics_state` carried-TKE + `scm`. Phase 1
uses the golden-tested **parcel buoyant-sorting `Lscale`** for the eddy
diffusivity (`Km = c_K·Lscale·√wp2`) + eddy-diffusion mean advance + a wp2
budget. **Phase 2a (iter 17):** the **ADG1 double-Gaussian PDF** (vs lite's
single Gaussian) is wired into the live path (`clubb_diagnostic.py`) — cloud
fraction + cloud water `rcm` + the moist buoyancy flux `wpthvp` (with the
cloud-water latent-heat term) now drive the wp2 buoyancy production. Phase 2b+
swaps the diagnostic moments for the full prognostic moment advances carried as
state (the DONE gate). Helpers
(all EXCLUDED in `test_physics_contracts` — closure
plumbing, not single-tendency schemes; all under the `clubb*.py` ruff
per-file-ignore for canonical CLUBB symbol names):

| module | contents | status |
|--------|----------|--------|
| `clubb_grid.py` | `CLUBBGrid` pytree (zm/zt/dzm/dzt/invrs_*), zm2zt/zt2zm/ddzm/ddzt + smoothers, `make_clubb_grid[_from_levels]`, `flip_vertical` | ✅ |
| `clubb_config.py` | `CLUBBFlags` (static, `register_static`), `CLUBBParams` (dynamic/differentiable), `CLUBBConfig`; `derive_mixt_frac_max_mag`/`derive_lmin` | ✅ |
| `clubb_saturation.py` | `sat_mixrat_liq`/`_ice` (Flatau, via `thermo` curves) | ✅ |
| `clubb_helpers.py` | `compute_sigma_sqd_w`, `calc_brunt_vaisala_freq_sqd` | ✅ |
| `clubb_skewness.py` | `Skx_func`, `compute_gamma_Skw`, `LG_2005_ansatz`, `xp3_LG_2005_ansatz` ✅; `calc_wp3_on_wp2` + `compute_skewness_diagnostics` (Skw_zm/zt + wp3_on_wp2 wiring) ✅36 | ✅ |
| `clubb_mixing_length.py` | parcel buoyant-sorting `Lscale` (golden-locked) | ✅ |
| `clubb_pdf.py` | ADG1 params (`ADG1_pdf_driver`), cloud fraction + rcm (`calc_pdf_liquid_cloud_frac[_components]`) | ✅ |
| `clubb_pdf_moments.py` | PDF moment integrals, higher-order moments, cloud-water `x'rc'` fluxes, buoyancy flux `wpthvp` | ✅ |
| `clubb_solve.py` | `tridiag_solve` (CLUBB band → legoESM `thomas_solve`) + `penta_solve` (verbatim CLUBB LU port, bit-exact) | ✅ iter 10-11 |
| `clubb_moments.py` | `advance_windm_edsclrm` ✅12; xp2_xpyp builders/TA/combiners ✅13-19; **`advance_xp2_xpyp` main** (full 5-moment advance, round-off parity) ✅20 | ✅ |
| `clubb_wp23.py` | coupled wp2/wp3 penta advance: 8 LHS + 9 RHS builders ✅21-22; `wp23_rhs/lhs/solve` ✅23; `compute_a1_a3_coef`/`compute_skw_fnc` ✅24; `clip_skewness` ✅25; **`advance_wp2_wp3` main** ✅26 (composition round-off parity; **CAM uses UPWIND wp3 MA** — `l_upwind_xm_ma=True`) | ✅ |
| `clubb_fill_holes.py` | `fill_holes_*` ✅19; `fill_holes_wp2_from_horz_tke` (TKE-conserving wp2 fill, CAM) ✅25 | ✅ |
| `clubb_xm_wpxp.py` | coupled xm/wpxp advance: builders + assembly ✅27-28; TA/LHS pre-computes + `diagnose_upxp` ✅29; `solve_xm_wpxp_with_single_lhs` + `xm_wpxp_clipping_and_stats` ✅33; **`advance_xm_wpxp` main** (rtm/wprtp + thlm/wpthlp; CAM `l_predict_upwp_vpwp=False`; C6/C7_Skw_fnc as inputs) ✅34 (wiring check + sub-piece parity; codex pending rate-limit) | ✅ |
| `clubb_mfl.py` | monotonic-flux-limiter JAX port: erf velocity + `mfl_xm_*` ✅30; `calc_turb_adv_range` (masked `fori_loop`) ✅31; **`monotonic_turbulent_flux_limit`** core (masked windowed min/max + `lax.scan` sequential clip + xm re-solve + top spike-fix, round-off parity all 4 fields, differentiable) ✅32 | ✅ |
| `clubb_tau.py` | CAM tau family: `calc_stability_correction` + `compute_tau_family` (`invrs_tau_C1/C4/C6/C14/xp2_zm`, `invrs_tau_wp3_zt` from parcel Lscale + N2 stability corr) ✅35 | ✅ |
| `clubb_coefficients.py` | C6rt/C6thl/C7 `_Skw_fnc` (CAM skewness functions, NOT ARM Richardson) + `damp_coefficient` (Lscale stable-region damping, from Fortran) ✅37 | ✅ |
| `clubb_core.py` | `compute_clubb_diagnostics` ✅43; `compute_pdf_closure` ✅45; `advance_clubb_core` (PDF + 4-advance ordered loop, conservation-tested) ✅46/50; `CLUBBMomentState`/`CLUBBForcing`/`init_clubb_moments` | ✅ |
| `clubb_diagnostic.py` | diagnostic ADG1-PDF closure → cloud frac + rcm + wpthvp (live path) | ✅ iter 17 |
| `clubb.py` | runnable scheme entry (parcel Lscale + ADG1-PDF moist buoyancy) | ✅ iter 16-17 |

Also added (shared): `legoesm.thermo.saturation_vapor_pressure_flatau[_ice]`.

**Testing pattern (per chunk):** analytic oracles that run in CI (moment
recovery, conservation, monotonicity, sign) + a committed tiny golden `.npz`
fixture (`tests/unit/clubb_fixtures/`, CLAUDE.md numeric-baseline carve-out) +
a `skipif` live bit-exact parity vs CLUBB-JAX (constants patched where the kernel
uses physical constants, so parity isolates the algorithm). AD-hardened: every
`sqrt`/divide that can hit 0 uses the double-where `_safe_sqrt`/guard; verified
finite gradients in float32 + float64.

---

## Status

- **P0 foundations** ✅ — grid operators + bridge, env, flag/constant tables.
- **P1 config** ✅ — `CLUBBConfig`/`CLUBBFlags`/`CLUBBParams` (CAM defaults).
- **P2 thermo/helpers** ✅ — Flatau saturation, sigma_sqd_w, Brunt–Väisälä.
- **P3 mixing length** ✅ — skewness diagnostics + parcel `Lscale` (golden-locked).
- **P4 ADG1 PDF closure** ✅ — params, cloud fraction+rcm, higher-order moments,
  cloud-water fluxes, buoyancy flux `wpthvp`. (golden-locked + AD-hardened)
- **P5 moment advance** ✅ — ALL 4 prognostic advances done with round-off/
  composition parity: `advance_windm_edsclrm` (u/v), `advance_xp2_xpyp` (5
  moments), `advance_wp2_wp3` (wp2/wp3 penta), `advance_xm_wpxp` (rtm/thlm +
  wprtp/wpthlp). All clips/limiters: `fill_holes*`, `clip_variance`,
  `clip_covar`, `clip_skewness`, full MFL (`clubb_mfl.py`).
- **P6 orchestration** ✅ — `advance_clubb_core` (iter 46) + `compute_clubb_
  diagnostics`/`compute_pdf_closure`; conservation-tested (iter 50). See the
  iter-50 status block below.
- **P7 integration** ✅ — `scheme="clubb"` dispatches (phase-1 diagnostic by
  default; full prognostic with `CLUBBConfig.prognostic=True`), carries
  `CLUBBMomentState` in `PhysicsState.clubb_moments`, and RUNS end-to-end through
  `combined.make_physics` (iter 53). Wired into coupler `validate_strict` +
  `physics_state` + `scm` + AMIP CLI.
- **P8 validation** ✅ — per-piece CLUBB-JAX parity + conservation + multi-step
  prognostic stability (host-diffusion fix) + jit/grad + end-to-end pipeline run.
  Idealized physics (iter 54-55): (a) convective BL — surface heating develops
  >3× the TKE of an unheated stable column with upward buoyancy flux; (b) cloud-
  PDF moisture response — the ADG1 cloud fraction increases monotonically with
  column moisture (the distinctive clubb_lite-lacks feature); (c) production
  pipeline stability — the carried moments stay bounded over 15 `make_physics`
  steps. Optional future: full BOMEX/DYCOMS profiles, fully-coupled long run.

---

## Status @ iter 50 (compressed; iter 11–50 detail in git history)

**The full prognostic CLUBB closure is BUILT, RUNS multi-step, and is TESTED.**
Pipeline (all in `clubb_core.py` unless noted), each piece parity-validated vs
CLUBB-JAX bit/round-off:
- `compute_clubb_diagnostics` (iter 43-44): Skw/σ²/em/tau-family/C6-C7/Kh.
- `compute_pdf_closure` (iter 45): CAM ADG1 closure → wpthvp/wp2thvp/rtpthvp/
  thlpthvp + HOM (wp4/wp2up2/…/wprtpthlp) + cloud-water fluxes + cloud_frac/rcm
  + ADG1 w_*_zm. (codex caught a real variance-floor leak: rt/thl_tol² floors
  feed ONLY the ADG1 driver; `calc_xpthvp_terms` gets RAW regrids.)
- `advance_clubb_core` (iter 46): per-step closure on `CLUBBMomentState`/
  `CLUBBForcing` pytrees, CAM order diagnostics→pre-PDF→xm_wpxp→xp2_xpyp→
  wp2_wp3→windm (`clip_covars_denom` between)→post-PDF (`l_call_pdf_closure_twice`).
- `clubb_step` (iter 47, `clubb.py`): legoESM top-down column → ascending host
  env (exner/p/thv_ds/rho_ds/invrs zt+zm, dry N², Lscale≥lmin, wm=0, fcor=0,
  ug=um/vg=vm) + surface-flux lower BCs → `advance_clubb_core` → du/dv/dT/dq.
  New `CLUBBConfig.T0` (ref temp for N²).
- `integrate_clubb_column` + `init_clubb_moments` (iter 48): `lax.scan` SCM-style
  multi-step prognostic run. Validated: 40-step stable, TKE growth under heating,
  jit+grad, returned-means consistency, q_v≥0 + density floor.
- **Conservation (iter 50):** `advance_clubb_core` conserves column-integrated
  ρ_ds-weighted thlm/rtm to <1e-9 over 5 steps (zero sfc flux + zero forcing) —
  truth-tier proof the assembly has NO spurious source.
- **Production dispatch (iter 52-53):** `pack/unpack_clubb_moments` +
  `PhysicsState.clubb_moments` carry (gated, gwd_spectrum-style); opt-in
  `CLUBBConfig.prognostic`; `clubb_turbulence_prognostic` (the `(TurbulenceOutput,
  packed-carry)` scheme entry); `get_turbulence_fn`/`turbulence_carry_field`/
  `_read_turb_carry` route the carry; `combined.py` stores it under the matching
  slot. nonhydro/spectral_pe fail-fast (don't persist phys_state); hydrostatic +
  mpas allowed. **END-TO-END: `scheme="clubb", prognostic=True` RUNS through
  `combined.make_physics` on a cubed-sphere state, carrying clubb_moments across
  steps (moments evolve, tendencies finite).** Codex-approved (2 rounds).

**✅ RESOLVED — dry-regime instability (iter 48-51).** Root cause CONFIRMED by
experiment: the standalone SCM driver advances the means with CLUBB ALONE, so it
exposes grid-scale (2Δz) vertical noise a coupled model's dynamical-core
numerical diffusion damps. Ruled out: Courant (smaller dt = worse), conservation/
source bug (iter-50 conservation test), per-piece port error (all terms match
CLUBB-JAX). FIX (iter 51): `integrate_clubb_column` gains `host_numerical_
diffusion` (default 0.05) — a CONSERVATIVE flux-form 2nd-order vertical diffusion
of the carried means (convex combo for nu≤0.5 → conserves column sum + preserves
positivity); the q_v floor is applied to the CLUBB tendency BEFORE the diffusion
so no water is created. A tiny nu removes the instability entirely (wp2max
10.7→0.06). Dry-stress test now PASSES; tests pin the root cause + the diffusion's
conservation/positivity. Codex-approved. Characterizer
`scripts/validate/clubb_prognostic_stability.py`.

**✅ DONE — prognostic CLUBB runs+tested in legoESM (iter 53).** `scheme="clubb"`
with `CLUBBConfig(prognostic=True)` dispatches the full prognostic higher-order
moment closure, carries `CLUBBMomentState` in `PhysicsState.clubb_moments` across
steps, and **runs end-to-end through `combined.make_physics`** on a cubed-sphere
state (verified: moments persist+evolve, tendencies finite, 2 steps). Default
`scheme="clubb"` (opt-out) remains the diagnostic phase-1 path. nonhydro/
spectral_pe drivers fail-fast (don't persist phys_state — documented, same as
MYNN-2.5); hydrostatic + mpas supported.

**Optional future hardening (beyond run+test):** longer coupled aquaplanet/
held-Suarez stability run; idealized BOMEX/DYCOMS validation; thread phys_state
through the nonhydro/spectral_pe drivers to lift their fail-fast.

## CAM-vs-ARM caveats (CLUBB-JAX is ARM-wired; re-check the CAM NAMELIST per module)
- Namelist OVERRIDES the Fortran flag defaults — always check the namelist.
- `l_predict_upwp_vpwp=F` → u/v via `advance_windm_edsclrm` (not prognostic upwp/vpwp).
- `l_diag_Lscale_from_tau=F` (Fortran default T, namelist F!) → SIMPLE `tau=Lscale/√em`,
  not the `invrs_tau_bkgnd+sfc+shear+N²` path in `mixing_length.calc_Lscale`.
- xp2 dp1: **3 distinct C2** (C2rt=C2thl=1.0, C2rtthl=1.3) → per-moment LHS+solve;
  CLUBB-JAX's single-C2 shared-LHS is ARM-only (valid iff all equal, F90:836).
- wp2/wp3 UPWIND mean-adv (`l_upwind_xm_ma=T`); xp2/xpyp TA UPWIND but xm/wpxp TA
  CENTERED; `l_damp_wp3_Skw_squared=F`→C8b=0; C6/C7 skewness fns NOT Richardson;
  `l_use_invrs_tau_N2_iso/l_pos_def/l_enable_relaxed_clipping=F`.
- Surface BCs (`clubb_step`): stress `tau=-ρ·Cd·|V|·u` → `u'w'_sfc=tau_x/ρ` (neg
  for u>0=drag); heat/moisture positive-up `wpthlp_sfc=shflx/(ρ·cp·exner)`,
  `wprtp_sfc=lhflx/(ρ·Lv)`.

Per-chunk testing: analytic/conservation oracle (CI) + round-off parity vs
CLUBB-JAX (constants patched to isolate algorithm) + codex adversarial review.

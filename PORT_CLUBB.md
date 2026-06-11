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
| `clubb_core.py` | `compute_clubb_diagnostics` (bundles Skw/σ²/em/tau/C6-C7) ✅43; `advance_clubb_core` (PDF closure + 4-advance ordered loop) ☐ | 🟡 P6 |
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
- **P6 orchestration** 🟡 — the `advance_clubb_core`-equivalent. Done: tau family
  (`clubb_tau.py`: `compute_tke`/`compute_tau_family`/`calc_stability_correction`),
  `Skw`/`wp3_on_wp2` diagnostics, C6/C7 `_Skw_fnc` (`clubb_coefficients.py`).
  ☐ remaining: `sigma_sqd_w`/`brunt_vaisala_freq_sqd` wiring (helpers exist), the
  pre+post ADG1 PDF closure (`l_call_pdf_closure_twice=True`) producing the
  4th-order moments (wp4/wp2up2/wp2thvp/rtpthvp/…) + PDF `w_1/w_2/varnce_w/
  mixt_frac`, and pack/unpack of the carried moment state.
- **P7 integration** 🟡 — ✅ iter 16: `clubb.py` entry + `TurbulenceOutput`;
  `"clubb"` wired into `get_turbulence_fn` dispatch + 4× `needs_tke` +
  `TurbulenceConfig.clubb` (None default, TYPE_CHECKING annotation — no import
  cycle) + coupler `validate_strict` + `physics_state` tke_schemes + `scm` +
  AMIP CLI choices. Runnable + tested. ☐ remaining: upgrade phase-1 eddy
  diffusion → the full prognostic moment advances + PDF buoyancy coupling so the
  live tendency path uses the genuinely-fuller-than-lite closure (DONE gate).
- **P8 validation** ☐ — single-column idealized (BOMEX/DYCOMS-ish) sanity,
  conservation/positivity, AD smoke through one step, codex-clean.

---

## Next (iter 41+)
*(Compressed at iter 40. iter 11–40 detail in git history + the module table
above = the live builder ledger; per-module CAM-vs-ARM caveats are in each
module's docstring.)*

**Status:** The full prognostic closure RUNS multi-step and is TESTED end-to-end
(iter 48). `advance_clubb_core` (iter 46) = diagnostics→pre-PDF→4 advances
(xm_wpxp→xp2_xpyp→wp2_wp3→windm, `clip_covars_denom` between)→post-PDF.
`clubb_step` (iter 47) bridges legoESM column inputs → host env + surface-flux
BCs → `advance_clubb_core` → du/dv/dT/dq. `integrate_clubb_column` (iter 48) =
`lax.scan` over `clubb_step` carrying `CLUBBMomentState`+means: a self-contained
SCM-style prognostic run. Validated: 40-step stable, TKE growth under heating,
jit+grad through the scan, returned-state consistency. All codex-reviewed.

**⚠ OPEN ISSUE (iter 48-49, tracked):** a long (~3 h) **near-dry, weakly-stratified**
single-column run develops a multi-step instability (grid-scale `T` extremes;
`wp2` grows with step COUNT at fixed total time). Moist regime stable. Strict
`xfail` (`test_integrate_clubb_column_dry_stress_stays_physical`); reusable
characterizer `scripts/validate/clubb_prognostic_stability.py`.
*iter-49 localization:* (1) GENUINE GROWING MODE, not Courant — smaller dt at
fixed time is WORSE; (2) needs weak stratification (dθ/dz ≲ 2e-3) AND sustained
surface heating; (3) the wp2 buoyancy production (`wp2_terms_bp_pr2_rhs`), `tau`
family, and `calc_stability_correction` ALL match CLUBB-JAX bit/round-off — the
per-piece port is faithful, and the stability-enhanced dissipation is correctly
weak at small N² so it can't brake the growth; (4) the standalone SCM driver
advances the means with CLUBB alone (no dycore numerical diffusion to damp 2-dz
noise) → partly a driver-exposure artifact a coupled run would damp.
*Definitive next step:* run a CLUBB-JAX reference `advance_clubb_core` on an
identical weakly-stratified forced column and compare the trajectory — settles
bug vs inherent-closure-delicacy. No speculative physics edits (would risk the
parity-validated terms).

**Remaining for production dispatch (separate from run+test):** carry
`CLUBBMomentState` through `PhysicsState` (`combined.py` registers a per-scheme
carry field — set it to `clubb_moments`; mind zm=nlev+1 vs the tke-slot nlev) +
restart I/O, and flip `integration.py`'s `scheme="clubb"` from the stateless
`clubb_turbulence` to a `clubb_step`-backed physics_fn. The extension point is
clean (`tagged_fns` `(fn, accepts_ps, field_name)` in `combined.py`).

**CAM-vs-ARM rule of thumb (verified the hard way):** CLUBB-JAX is wired for ARM;
re-check the CAM namelist/Fortran per module. Caught: wp2/wp3 UPWIND MA
(`l_upwind_xm_ma=True`); xp2/xpyp TA UPWIND (`l_upwind_xpyp_ta=True`) but xm/wpxp
TA CENTERED; wp2_dp1 + wp3_pr_turb CAM branches from CESM Fortran;
`l_damp_wp3_Skw_squared=False`→C8b=0; C6/C7 are skewness fns not Richardson;
`l_use_invrs_tau_N2_iso/l_pos_def/l_enable_relaxed_clipping=False`.

**Remaining to DONE:**
1. ✅ **PDF-closure diagnostics bundle** (iter 43-44, `clubb_core.py`):
   `compute_clubb_diagnostics` = Skw/sigma_sqd_w/em/tau-family/C6-C7/Kh.
2. ✅ **PDF-closure outputs** (iter 45, `clubb_core.compute_pdf_closure`): the
   CAM-default ADG1 closure producing wpthvp/wp2thvp/rtpthvp/thlpthvp +
   wp4/wp2up2/wp2vp2/wpup2/wpvp2/wp2rtp/wp2thlp/wp2up/wprtp2/wpthlp2/wprtpthlp +
   wprcp/rtprcp/thlprcp/uprcp/vprcp + cloud_frac/rcm/rc_coef_zm. Composes the
   parity-tested helpers; reuses Skw_zt/wp2_zt/sigma_sqd_w from the diagnostics
   bundle. **Codex-reviewed & approved** (caught + fixed a real variance-floor
   leak: rt_tol²/thl_tol² floors feed ONLY the ADG1 driver, calc_xpthvp_terms
   gets RAW regrids — regression-tested).
3. ✅ **`advance_clubb_core` assembly** (iter 46, `clubb_core.py`): the full
   per-step closure on `CLUBBMomentState`/`CLUBBForcing` pytrees, CAM order with
   `clip_covars_denom` between, pre+post PDF (`l_call_pdf_closure_twice=True`).
   Codex-approved (3 rounds). **CAM 3-C2 fix**: `advance_xp2_xpyp` inherited the
   ARM single-C2 shared-LHS solve from CLUBB-JAX; CAM uses 3 distinct dp1
   dissipation coefficients (C2rt=C2thl=1.0, C2rtthl=1.3) — a shared solve is
   valid only when equal (`advance_xp2_xpyp_module.F90:836`). Refactored to
   per-moment LHS+solve; dropped the `Cn` arg (now owned from config). Verified
   FALSE POSITIVE: windm uses start-of-step `Kh_zm` by design (reference computes
   Kh once in "Block M", same array to wp2_wp3 + windm).
4. ✅ **Column bridge `clubb_step`** (iter 47, `clubb.py`): builds the ascending
   host env (exner/p/thv_ds/rho_ds/invrs zt+zm, dry N², Lscale floored at lmin,
   wm=0, fcor=0, ug=um/vg=vm), sets surface-flux lower BCs (kinematic:
   wpthlp_sfc=shflx/(ρ·cp·exner), wprtp_sfc=lhflx/(ρ·Lv), **upwp_sfc=tau_x/ρ** —
   codex caught the sign), resets means from the live column, calls
   `advance_clubb_core`, maps means back to du/dv/dT/dq. New `CLUBBConfig.T0`
   (ref temp for N²). Codex-approved. Bug fixed: Lscale_zm=0→invrs_tau=inf
   (floor Lscale at lmin; grid-consistent test columns). Verified CAM
   `l_diag_Lscale_from_tau=.false.` → simple tau model is the CAM path (Fortran
   default is .true.; the CAM namelist overrides — per-module namelist check).
5. ✅ **Multi-step prognostic run+test** (iter 48): `integrate_clubb_column`
   (`lax.scan` over `clubb_step`) + `init_clubb_moments`. The closure runs and is
   tested end-to-end over many steps (moist regime stable). Density floor +
   q_v≥0 clip + returned-state-means consistency; dry-regime instability tracked
   (xfail, see OPEN ISSUE above).
6. ☐ **Production dispatch flip** (optional hardening, not run+test): persist
   `CLUBBMomentState` via `PhysicsState`/`combined.py` `tagged_fns` + restart I/O
   and back `scheme="clubb"` with `clubb_step`. AND resolve the dry-regime
   instability before production use.

Each chunk: analytic/self-consistency oracle (CI) + golden/round-off parity vs
CLUBB-JAX (patch reference physical constants to isolate algorithm) + codex review.

**Codex status:** rate limit reset. iter-45 reviewed `compute_pdf_closure`
(needs-attention→fix→approve, variance-floor leak). iter-46 reviewed
`advance_clubb_core` + the CAM 3-C2 fix (3 rounds: Kh_zm false-positive verified,
C2rtthl bug fixed → approve). Earlier pending batch (iter-34→39: advance_xm_wpxp
main, clubb_tau, clubb_skewness, clubb_coefficients) carry parity/analytic/
jit-grad self-validation + the iter-41 gold-standard full-main reference-parity
test for advance_xm_wpxp.

**CAM-vs-ARM caught (iter 46):** xp2 dp1 dissipation uses 3 distinct C2
(C2rt/C2thl/C2rtthl); CLUBB-JAX hardwires single C2rt (ARM). Single shared-LHS
solve valid only when all equal. Pattern reminder: even "shared-LHS" optimizations
in CLUBB-JAX can encode ARM-specific coefficient-equality assumptions.

**CAM-flag re-verified (iter 47):** `l_diag_Lscale_from_tau` — Fortran
model_flags default `.true.` BUT the CAM namelist overrides to `.false.`, so CAM
uses the SIMPLE `tau = Lscale/sqrt(em)` model (my `compute_tau_family`), not the
complex `invrs_tau_bkgnd+sfc+shear+N²` path in `mixing_length.py:calc_Lscale`.
Always check the CAM NAMELIST, not just the Fortran flag default.

**Surface-flux signs (iter 47, `clubb_step`):** stress `tau=-ρ·Cd·|V|·u` so the
kinematic momentum BC is `u'w'_sfc = tau_x/ρ` (negative for u>0 = drag). Heat/
moisture BCs positive-up: `wpthlp_sfc=shflx/(ρ·cp·exner)`, `wprtp_sfc=lhflx/(ρ·Lv)`.

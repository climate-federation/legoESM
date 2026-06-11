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

## Status @ iter 60 — ✅ DONE (compressed; iter 11–59 detail in git history)

**legoESM can be run AND tested with the prognostic `clubb.py` scheme.**
`scheme="clubb"` + `CLUBBConfig(prognostic=True)` dispatches the full CAM-default-
tree CLUBB higher-order moment closure, carries `CLUBBMomentState` in
`PhysicsState.clubb_moments`, and runs end-to-end through `combined.make_physics`
on a cubed-sphere state. (Default `scheme="clubb"` stays the diagnostic phase-1
path; prognostic is opt-in. Hydrostatic + mpas drivers supported; nonhydro/
spectral_pe fail-fast since they drop phys_state — same as MYNN-2.5.)

**Pipeline** (each piece bit/round-off parity-validated vs CLUBB-JAX):
`compute_clubb_diagnostics` (Skw/σ²/em/tau/C6-C7/Kh) → `compute_pdf_closure` (CAM
ADG1: wpthvp/HOM/cloud-water fluxes/cloud_frac/rcm) → `advance_clubb_core` (CAM
order xm_wpxp→xp2_xpyp→wp2_wp3→windm, `clip_covars_denom` between, pre+post PDF) →
`clubb_step` (legoESM column ↔ ascending host env + surface-flux BCs) →
`clubb_turbulence_prognostic` (scheme entry; CAM `clubb_timestep` sub-cycling;
pack/unpack carry). Standalone driver `integrate_clubb_column` (`lax.scan`).
All 4 advances + clips/limiters + MFL + tau/Skw/C6-C7 ported. The distinctive
fuller-than-lite features: ADG1 **double-Gaussian** cloud PDF + parcel buoyant-
sorting `Lscale` + full prognostic moment transport (clubb_lite has none).

**Validated** (full clubb suite + integration + audit gates, all green):
- per-piece CLUBB-JAX parity; `advance_clubb_core` thlm/rtm **conservation** <1e-9.
- multi-step + production-**pipeline stability**; **end-to-end jax.grad** through
  `make_physics` (finite nonzero) — the foundational legoESM autodiff requirement.
- idealized **physics**: convective-BL TKE response (>3× heated vs calm, upward
  buoyancy flux); cloud-fraction monotone in moisture.
- **no-regression**: 98-test turbulence-integration suite (the shared-infra carry
  refactor left every other scheme intact); all CI ratchet/contract gates pass
  (contracts/constants/saturation/private-imports/dispatch/validate-strict/
  federation/formula-reimpl).
- codex-adversarially-reviewed at every substantial step (real bugs caught+fixed:
  CAM 3-C2 dissipation, surface-flux sign, variance-floor leak, sub-cycle moisture
  contract, dispatch persistence guards, retrace hazard).
- **column-moisture conservation @ scheme entry (iter 61):** a no-surface-flux
  (`q_sfc`=near-sfc `q_v` ⇒ `lhflx==0`) single prognostic step conserves the
  mass-weighted (`ρ·dz`) column total to round-off (<1e-12 rel/step). Made
  **non-vacuous** after codex caught that the rest/floor moment state gives
  `dq_v_dt`~1e-11 (no `wprtp` flux ⇒ any scheme trivially "conserves"): the test
  first spins up a real flux via `integrate_clubb_column` (dt=150,nsteps=40), then
  asserts BOTH a nontrivial-transport floor (`max|dq·dt|`~1e-6 ≫ 1e-8) AND
  conservation. Note: the bare scheme entry is single-step-stable only — feeding
  its tendencies back without dycore/host diffusion NaNs by step 1 (the documented
  dry-regime instability), so multi-step spin-up MUST use the diffusion-stabilized
  driver.

**Key resolved issue — dry-regime instability (iter 48-51):** root-caused (by
experiment) to the bare SCM driver advancing means with CLUBB alone, exposing 2Δz
noise a coupled dycore damps — NOT a closure bug (conservation + per-piece parity
hold). Fix: a conservative flux-form host-diffusion stand-in in
`integrate_clubb_column` (`host_numerical_diffusion`, default 0.05; coupled path
relies on the real dycore). Characterizer `scripts/validate/clubb_prognostic_stability.py`.

**Audit (iter 58, 60):** created canonical `physics/_shared.exner_function`/
`buoyancy_coefficient`, migrated the clubb files (+ sibling `clubb_lite`) off
inline re-derivations, ratcheted the formula-debt budgets down.

**Optional future (beyond run+test):** BOMEX/DYCOMS profiles; long coupled run;
thread phys_state through nonhydro/spectral_pe to lift their fail-fast; edsclr
(passive-scalar) transport through CLUBB.

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

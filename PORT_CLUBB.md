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

**Status:** ALL 4 prognostic advances + all clips/limiter + the tau/Skw/C6-C7
orchestration pieces are ported & parity-validated. The scheme is runnable+tested
(phase-1 Lscale eddy diffusion + ADG1-PDF diagnostic buoyancy). The DONE gate is
wiring the *prognostic* closure into `clubb.py`'s live tendency path.

**CAM-vs-ARM rule of thumb (verified the hard way):** CLUBB-JAX is wired for ARM;
re-check the CAM namelist/Fortran per module. Caught: wp2/wp3 UPWIND MA
(`l_upwind_xm_ma=True`); xp2/xpyp TA UPWIND (`l_upwind_xpyp_ta=True`) but xm/wpxp
TA CENTERED; wp2_dp1 + wp3_pr_turb CAM branches from CESM Fortran;
`l_damp_wp3_Skw_squared=False`→C8b=0; C6/C7 are skewness fns not Richardson;
`l_use_invrs_tau_N2_iso/l_pos_def/l_enable_relaxed_clipping=False`.

**Remaining to DONE:**
1. **PDF-closure outputs**: the pre+post ADG1 closure (`l_call_pdf_closure_twice
   =True`) producing the 4th-order moments (wp4/wp2up2/wp2vp2/wpup2/wpvp2/wp2thvp/
   rtpthvp/thlpthvp) + PDF velocity params (`w_1/w_2/varnce_w/mixt_frac`). Have
   `ADG1_pdf_driver`/`calc_pdf_higher_order_moments` — needs the input wiring.
2. **sigma_sqd_w + brunt** wiring (helpers exist: `compute_sigma_sqd_w`,
   `calc_brunt_vaisala_freq_sqd`).
3. **`advance_clubb_core` assembly** + carried moment state (wp2/wp3/up2/vp2/
   rtp2/thlp2/rtpthlp/wprtp/wpthlp), with the pre/post PDF placement.
4. **Wire into `clubb.py`**: carry the moment set as state; replace the phase-1
   eddy-diffusion mean advance with the prognostic advances.

Each chunk: analytic/self-consistency oracle (CI) + golden/round-off parity vs
CLUBB-JAX (patch reference physical constants to isolate algorithm) + codex review.

**⚠ Pending codex batch** (external rate limit, resets ~19:37 MDT): iter-34→39
(`advance_xm_wpxp` main, `clubb_tau` incl. iter-38 AD-hardening + iter-39
`compute_tke`, `clubb_skewness` diagnostics, `clubb_coefficients`). All small,
parity/analytic/jit-grad self-validated. iter-41 added the **gold-standard
full-main reference-parity test** for `advance_xm_wpxp` (round-off match vs the
reference main — substitutes for codex on the biggest pending piece). RUN THE
BATCH the moment codex resets, before tackling the (larger) PDF-closure wiring.

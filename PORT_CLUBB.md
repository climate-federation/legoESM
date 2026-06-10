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

*(Compressed at iter 10; verbose iter 1–9 logs dropped — see git history. The
reference facts + per-phase status below are the live source of truth.)*

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
| `clubb_skewness.py` | `Skx_func`, `compute_gamma_Skw`, `LG_2005_ansatz`, `xp3_LG_2005_ansatz` | ✅ |
| `clubb_mixing_length.py` | parcel buoyant-sorting `Lscale` (golden-locked) | ✅ |
| `clubb_pdf.py` | ADG1 params (`ADG1_pdf_driver`), cloud fraction + rcm (`calc_pdf_liquid_cloud_frac[_components]`) | ✅ |
| `clubb_pdf_moments.py` | PDF moment integrals, higher-order moments, cloud-water `x'rc'` fluxes, buoyancy flux `wpthvp` | ✅ |
| `clubb_solve.py` | `tridiag_solve` (CLUBB band → legoESM `thomas_solve`) + `penta_solve` (verbatim CLUBB LU port, bit-exact) | ✅ iter 10-11 |
| `clubb_moments.py` | `advance_windm_edsclrm` ✅12; xp2_xpyp terms/TA/combiners ✅13-15; `term_ma_zm_lhs`+`calc_xp2_xpyp_lhs`/`calc_up2_vp2_lhs` ✅18; `xp2_xpyp_uv_rhs`+`pos_definite_variances` ✅19; `clip_variance`+`solve_xp2_xpyp`+**`advance_xp2_xpyp` main** (full 5-moment advance, round-off parity) ✅20; `advance_wp2_wp3`, `advance_xm_wpxp` (penta), `mono_flux_limiter` ☐ | 🟡 P5 |
| `clubb_fill_holes.py` | mass-conserving vertical hole-fill (`fill_holes_type=2` sliding-window+global, CAM default) | ✅ iter 19 |
| `clubb_wp23.py` | coupled wp2/wp3 penta advance: 8 LHS term builders + inline `weights_zt2zm` ✅ iter 21 (golden+parity); RHS builders, `wp23_lhs/rhs/solve`, `advance_wp2_wp3` main ☐ | 🟡 P5 |
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
- **P5 moment advance** 🟡 — ✅ `advance_windm_edsclrm` (u/v), ✅ `advance_xp2_xpyp`
  (rtp2/thlp2/rtpthlp/up2/vp2 — full core path, round-off parity), ✅ `fill_holes`
  + `clip_variance`/`clip_covar`. ☐ remaining: `advance_wp2_wp3` + `advance_xm_wpxp`
  (penta — `clubb_solve.penta_solve` ready) + `mono_flux_limiter` (CAM ON).
- **P6 orchestration** ☐ — assemble the `advance_clubb_core`-equivalent for the
  CAM flag subset; pack/unpack carried moment state (wp2/wp3/thlp2/rtp2/rtpthlp/
  wpthlp/wprtp/up2/vp2). `l_call_pdf_closure_twice=True` → PDF pre+post.
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

## Next (iter 21+)
*(Compressed at iter 20. iter 11–20 detail in git history; `clubb_moments.py`
table row above is the live builder ledger.)*

Remaining to reach the DONE gate (full prognostic closure in the live path):
1. **`advance_wp2_wp3`** (penta — `clubb_solve.penta_solve` ready). ✅ iter 21:
   the 8 LHS term builders (`clubb_wp23.py`, golden+bit-exact parity; CAM gating
   `l_standard_term_ta=False` ADG1 TA, `l_tke_aniso=True`, `l_damp_wp3_Skw_squared
   =False`→C8b=0). ☐ remaining: the RHS term builders, the `wp23_lhs`/`wp23_rhs`
   penta assembly (interleaved wp2[2k]/wp3[2k+1]), `wp23_solve` (penta), main.
2. **`advance_xm_wpxp`** (penta — rtm/thlm + wprtp/wpthlp coupled solve;
   `l_predict_upwp_vpwp=False` so upwp/vpwp stay diagnostic/eddy-diffusion).
3. **`mono_flux_limiter`** (CAM `l_mono_flux_lim_*=True`).
4. **P6 orchestration** `advance_clubb_core`: the dissipation-timescale inputs
   (`invrs_tau_*`, `Cn`), `wp3_on_wp2`, `sigma_sqd_w`, the pre+post PDF closure
   (`l_call_pdf_closure_twice=True`), and pack/unpack of the carried moment state.
5. **Wire into `clubb.py`**: carry the full moment set as state, replace the
   phase-1 eddy-diffusion mean advance with the prognostic advances.

Each chunk: analytic/self-consistency oracle (CI) + golden/round-off parity vs
CLUBB-JAX (patch reference physical constants to isolate algorithm) + codex
adversarial review.

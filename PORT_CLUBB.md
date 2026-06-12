# PORT_CLUBB — Porting a fuller CLUBB turbulence scheme into legoESM

**New Goal:** The various clubb_*.py files need to be condensed into a single clubb.py. A table of contents citing line numbers should go at the top of this file to help organize the code. The values of the  CLUBB model flags, from clubb_config.py, can be appended as comments at the end of the file with the note that these come from the CAM defaults, but the flags themselves do not need to be defined because we will not implement any other tree. The constants, from clubb_coefficients.py, should go near the top of the single clubb.py file. The code from clubb_core.py, clubb_diagnostics.py, clubb_fill_holes.py, clubb_grid.py, clubb_helpers.py, clubb_mfl.py, clubb_moments.py, clubb_pdf.py, clubb_pdf_moments.py, clubb_skewness.py, clubb_solve.py,  clubb_tau.py, clubb_wp23.py, and clubb_xm_wpxp.py, needs to be integrated inside the single clubb.py. Some changes to clubb_lite.py can be retained, e.g. putting exner function together with mixing_length from physics._shared, but using the function buoyancy_coefficient is superfluous, revert those changes. If the mixing length computed by clubb_mixing_length.py is the same as that used in physics._shared that is also used by clubb_lite, only define it in _shared.py otherwise put it into clubb.py and we can make an issue noting that it could supplant the one computed in _shared. Replace clubb_saturation with one defined in a shared module if it exists, otherwise move to clubb.py. Don’t define a new saturation formula, even in the shared files, unless it really is the one used by CAM.

## Condensation plan + status (live; updated each iteration)

**Strategy — top-down absorption, always green:** repeatedly inline `clubb.py`'s
*direct* dependencies into it (so `clubb.py` stays at the top of the import DAG —
no cycles, no transitional stub modules), delete the absorbed module, re-point
every importer (prod + tests) at `clubb.py`, run the absorbed modules' tests +
the contracts gate, codex-review, commit. Sections in `clubb.py` accrete in
call-tree order; the line-numbered TOC at the top is finalized once all modules
are in (section names maintained meanwhile). Test files are re-pointed as their
module is absorbed; test-file consolidation (into `tests/unit/test_clubb.py`)
happens after the source consolidation settles.

| # | module | LOC | plan | status |
|---|--------|-----|------|--------|
| 1 | `clubb_diagnostic.py` | 105 | absorb (sec. 1) | ✅ C1 (iter 1) |
| 2 | `clubb_core.py` | 443 | absorb (sec. 2) | ✅ C1 (iter 1) |
| 3 | `clubb_moments.py` | 885 | absorb | ✅ C3 (iter 3, sec. 2) |
| 4 | `clubb_wp23.py` | 720 | absorb | ✅ C2 (iter 2, sec. 3) |
| 5 | `clubb_xm_wpxp.py` | 431 | absorb | ✅ C2 (iter 2, sec. 4) |
| 6 | `clubb_mfl.py` | 308 | absorb | ✅ C3 (iter 3, sec. 5) |
| 7 | `clubb_pdf_moments.py` | 283 | absorb | ✅ C4 (iter 4, sec. 7) |
| 8 | `clubb_pdf.py` | 368 | absorb | ✅ C4 (iter 4, sec. 6) |
| 9 | `clubb_fill_holes.py` | 201 | absorb | ✅ C4 (iter 4, sec. 3) |
| 10 | `clubb_tau.py` | 97 | absorb | ✅ C4 (iter 4, sec. 5) |
| 11 | `clubb_skewness.py` | 211 | absorb | ✅ C4 (iter 4, sec. 4) |
| 12 | `clubb_solve.py` | 151 | absorb | ✅ C4 (iter 4, sec. 2) |
| 13 | `clubb_helpers.py` | 208 | absorb | ✅ C5 (iter 5, sec. 4) |
| 14 | `clubb_grid.py` | 380 | absorb | ✅ C5 (iter 5, sec. 2) |
| 15 | `clubb_coefficients.py` | 71 | constants → top of clubb.py | ✅ C2 (iter 2, sec. 2; the module is FUNCTIONS not constants — placed as a section; the true scalar constants land near the top in the final-layout pass) |
| 16 | `clubb_mixing_length.py` | 472 | compare vs `_shared` Blackadar (DIFFERENT — parcel buoyant-sorting) → absorb + file issue noting it could supplant `_shared`'s | ✅ C5 (iter 5, sec. 5; supplant-note in the section header) |
| 17 | `clubb_saturation.py` | 87 | thin wrappers over `thermo` Flatau curves (shared module EXISTS) → wrappers into clubb.py | ✅ C5 (iter 5, sec. 3) |
| 18 | `clubb_config.py` | 338 | `CLUBBParams`/`CLUBBConfig` → clubb.py; **drop `CLUBBFlags`** (CAM tree only), flag VALUES → comment block at END of clubb.py; prune dead non-CAM branches that the flag removal exposes | ✅ C6 (iter 6, sec. 2 + flag table at file end) |
| 19 | `clubb_lite.py` | 328 | NOT absorbed (separate scheme). Retain `_shared.exner_function` use; REVERT its `buoyancy_coefficient` usage (superfluous per goal) | ✅ C7 (iter 7) |

**Iteration log (condensation):**
- **C1 (iter 1):** absorbed `clubb_diagnostic.py` + `clubb_core.py` into
  `clubb.py` (sections 1-2 + section-comment scaffold for the future TOC),
  byte-identical bodies (codex-verified vs `git show HEAD:` originals); deleted
  both modules; re-pointed physics_state.py / integration.py /
  test_clubb_core.py / test_clubb_diagnostic.py / test_clubb_scheme.py; removed
  both filenames from test_physics_contracts EXCLUDED/CONTRACT_TODO (shrink-only
  ok). Green: test_clubb_core + test_clubb_diagnostic + contracts (127) + 5
  conservation tests from test_clubb_scheme; ruff clean on clubb.py; codex
  adversarial review → only stale-doc nits (fixed here).
- **C2 (iter 2):** absorbed `clubb_coefficients.py` + `clubb_wp23.py` +
  `clubb_xm_wpxp.py` (1222 LOC) into `clubb.py` as sections 2-4 (coefficients
  co-absorbed to resolve its `compute_skw_fnc` edge into wp23 without a cycle);
  deduped the byte-identical `_GAMMA = 1.5` (gamma_over_implicit_ts) shared by
  the wp23/xm_wpxp advances; deleted the 3 modules; re-pointed the module-alias
  test imports (`import clubb as W/X/C`); contracts EXCLUDED entries removed.
  clubb.py now 2272 lines, 6 sections. Green: wp23+xm_wpxp+coefficients+core
  tests + contracts (174) + 5 conservation tests; ruff clean; codex AST-level
  adversarial review → APPROVE (body fidelity, imports, no stale refs).
  ALSO fixed: the C1 commit had silently lost the two `git rm` deletions (a
  `git stash`/`pop` during the ruff-baseline check unstaged them; explicit-path
  `git add` missed deleted paths) — amended into C1. Lesson: after stash/pop,
  re-verify staged deletions with `git status` before committing.
- **C3 (iter 3):** absorbed `clubb_moments.py` + `clubb_mfl.py` (1193 LOC) into
  `clubb.py` as sections 2 (moment-advance building blocks + xp2_xpyp/windm
  advances) and 5 (monotonic flux limiter), 8 sections total (3376 lines);
  deduped the byte-identical `_EPS = 1.0e-10`; mfl's NaN-propagating local
  `_safe_sqrt` kept local (intentional variant). Deleted both modules;
  re-pointed test_clubb_moments/test_clubb_mfl + function-scope imports in
  test_clubb_xm_wpxp/test_clubb_wp23; contracts entries removed. Documented
  the intentional no-`__all__` policy in the clubb.py docstring (public API =
  the 4 scheme entries; machinery stays importable for parity tests). Green:
  215 tests (moments/mfl/wp23/xm_wpxp/core + contracts); ruff clean; codex
  adversarial review → request-changes (3 doc nits + __all__ decision) →
  fixed → approve.
- **C4 (iter 4):** absorbed the 6 remaining leaf closure modules (1311 LOC) —
  `clubb_solve` + `clubb_fill_holes` + `clubb_skewness` + `clubb_tau` +
  `clubb_pdf` + `clubb_pdf_moments` — as sections 2-7 (14 sections, 4512
  lines). Deduped byte-identical-value constants (`_EPS`/`_MAX_MAG_CORRELATION`/
  `_SQRT_2`/`_SQRT_2PI`/`_EP1`/`_EP2`); `_F64_EPS` deduped raw-numpy→float()
  form (bool-only use sites — codex-verified value-equivalent). Caught by
  tests: the absorbed solve body needed `from jax import lax` (carried imports
  must be re-derived per batch, not assumed). 6 modules deleted; test imports
  re-pointed; contracts entries removed. Green: 202 tests; ruff clean; codex
  AST-level adversarial review → APPROVE (first pass).
- **C5 (iter 5):** absorbed the geometry/thermo helper layer (1147 LOC) —
  `clubb_grid` + `clubb_saturation` + `clubb_helpers` + `clubb_mixing_length`
  — as sections 2-5 (18 sections, 5499 lines). Saturation = thin adapters over
  the canonical `legoesm.thermo` Flatau curves (per goal: shared module
  exists, wrappers move in). Mixing length CONFIRMED different numerics from
  `_shared`'s Blackadar (parcel buoyant-sorting) → absorbed with a
  could-supplant-_shared note in its section header. Deduped
  `_ZERO_THRESHOLD`/`_EP1`/`_EP2`/`_ZERO`/`_EPS` (kept mixing_length's unique
  `_EP`/`_LV2_COEF`); pdf section's function-scope saturation import now a
  direct module-level reference. 4 modules deleted; importers re-pointed;
  contracts entries removed. Green: 197 + 5 conservation tests; ruff clean;
  codex review → approve (2 [low] flags were the intentional test-header
  updates, same sanctioned class as C1-C4).
- **C6 (iter 6) — config endgame:** `CLUBBParams`/`CLUBBConfig`/derive fns →
  clubb.py section 2 (19 sections, 5789 lines); **`CLUBBFlags` class REMOVED**
  (only the CAM tree is implemented): all 14 former `config.flags.*` sites
  hardcoded to CAM defaults with "CAM <flag> = <val>" site comments — dead
  branches DELETED (`l_lmm_stepping` x2, `l_enable_relaxed_clipping` True-arm
  + its 2 floor constants), True-branches made unconditional
  (`l_min_xp2_from_corr_wx`, `l_wp2_fill_holes_tke`, `l_tke_aniso`),
  `fill_holes_type` → `_CAM_FILL_HOLES_TYPE = 2`. The 65 flag VALUES live as
  a machine-parseable reference table at the END of clubb.py;
  test_clubb_config.py REWRITTEN to parse that table (the CAM-namelist
  source-of-truth tripwire survives on the comment block + new
  table-completeness and constant-vs-table consistency gates). clubb_config
  deleted; config.py keeps a TYPE_CHECKING-only import (no runtime cycle).
  Green: 265 + 7 + 12 tests; ruff clean; codex adversarial review → APPROVE
  (verified: 14/14 flag sites, 102/102 params identical, 65/65 table match,
  dead branches truly dead under CAM defaults).
- **C7 (iter 7):** `clubb_lite.py` revert per goal — `N2_half` back to main's
  inline `constants.g/θ_v` form (byte-identical, codex-verified),
  `buoyancy_coefficient` import dropped; `exner_function`/`mixing_length`
  retained. Formula-ratchet budget: clubb_lite buoyancy entry RESTORED to
  main's baseline 1 (decision, not new debt — documented in the budget).
  Also fixed pre-existing rebase fallout in `test_no_formula_reimpl.py`:
  main-side files (`spectral_les_plane.py` buoyancy 1→2,
  new `spectral_les_moist.py` exner=3) seeded at their measured main
  baselines (gate was red at HEAD since the rebase). **FULL CLUBB suite
  green: 335 passed across all 19 test files (32 min)** — the complete
  condensed clubb.py validated end-to-end. Codex review → approve.
- **C8 (iter 8):** line-numbered TOC shipped — every TOC entry in the
  clubb.py docstring now cites the exact section-header line
  (`N.  [line  L] Title`, patched in place so the numbers stay valid), plus
  a flag-table pointer in the TOC header. New tripwire
  `test_toc_line_numbers_accurate` (codex-verified non-vacuous: a single
  inserted line fails it loudly) — future edits must regenerate the TOC.
  Coefficient-constants placement: clubb_coefficients had FUNCTIONS only
  (no module constants), so "constants near top" is satisfied by the
  config/derived-constants section 2; noted in the plan table (C2 row).
  Codex review → APPROVE (one [warn]: TOC title-text drift not policed —
  acceptable; line-number drift is).
  **Remaining:** test-file consolidation decision, whole-repo gate sweep
  (ratchets/contracts/private-imports/turbulence-integration suite),
  PORT_CLUBB.md compression at iter 10.


**Previous Goal:** `packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py`
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

*(Compressed at iter 10/20/30/40/60/70/80; verbose per-iter logs dropped — see git
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
single Gaussian) is wired into the live path (now section 1 of `clubb.py`) — cloud
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
| ~~`clubb_core.py`~~ | `compute_clubb_diagnostics` ✅43; `compute_pdf_closure` ✅45; `advance_clubb_core` (PDF + 4-advance ordered loop, conservation-tested) ✅46/50; `CLUBBMomentState`/`CLUBBForcing`/`init_clubb_moments` | ✅ ABSORBED → clubb.py sec. 2 (C1) |
| ~~`clubb_diagnostic.py`~~ | diagnostic ADG1-PDF closure → cloud frac + rcm + wpthvp (live path) | ✅ ABSORBED → clubb.py sec. 1 (C1) |
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

## Status @ iter 80 — ✅ DONE (compressed at iter 10/20/.../70/80; iter 11–79 detail in git history)

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

**Validation hardening (iter 61–79; compressed at iter 70/80 — detail in git history).**
All in `tests/unit/test_clubb_scheme.py`; each codex-adversarially-reviewed to
clean. Common technique: spin up real moments via `integrate_clubb_column`
(dt=150,nsteps=40) so the property is **non-vacuous** (the rest/floor state gives
~round-off tendencies any scheme trivially passes).
- **Conservation triad @ scheme entry, zero surface flux:** column **rt** (iter 61)
  and **θl** (iter 62) conserved to round-off (`<1e-12` rel/step, `ρ·dz` weight);
  **momentum** `Σ mass·du/dt = τ_x` (iter 64) closes to **O(Δt)** (semi-implicit
  surface stress; Richardson Δt 10×→residual 10×; confirms interior flux-form + the
  surface-stress SIGN). Winds via `advance_windm_edsclrm`, scalars via `advance_xm_wpxp`.
- **Bridge convention (iter 62):** `clubb_step` maps mean back as `T_new=thlm·exner`,
  so `dT_dt/exner` IS the θl tendency and `q_v=rtm` — prognostic CLUBB returns θl/rt,
  NOT a sat-adjusted (T,q_v) split (cloud partition deferred to microphysics,
  `l_rcm_supersat_adj=.false.`). **Turbulence→microphysics coupling (iter 63):** that
  return is correct — CLUBB→sundqvist closes the column water budget to the precip sink
  (rel<1e-12), condensation enthalpy-consistent. **Moist-chain differentiability
  (iter 65):** `jax.grad` flows end-to-end through CLUBB→sundqvist incl. the
  `max(q_v−q_sat,0)` kink (microphysics-only objective + active-branch guard).
- **Fuller-than-clubb_lite signature (iter 66):** under surface heating prognostic
  CLUBB develops vertical-velocity SKEWNESS (`wp3`>0, ~+0.06–0.15, non-local transport);
  the test executes `clubb_lite_turbulence` and asserts it has no `wp3`.
- **float32/Metal (iter 67–69):** FIXED a real bug — `compute_mixing_length` had
  strong-float64 sources (`jnp.float64(0.0)` scan carries, default-dtype `zeros`/`full`,
  `set_Lscale_max`) that promoted a float32 column and crashed `lax.scan`'s carry check.
  Fix: normalize every float input to `dt_f=thlm.dtype` (preserves float64 EXACTLY; 8
  golden tests pass). Both entries float32-finite/faithful. Caveat (inherent): `dT_dt=
  Π·(θl_new−θl)/dt` differences two ~300 K values → ~5–10% float32 cancellation noise.

**Production-SCM runnability (iter 71):** prognostic CLUBB RUNS through the real
`SingleColumnModel` driver (registered in the SCM `stateful_turb` set) on a Wangara
convective-BL setup (40 steps finite, physical `T_low`, `wp2` develops). NOTE: SCM
`.run()` is an EAGER loop (~8 s/step) → use the jitted `integrate_clubb_column` for
efficient multi-step testing.
- **LIVE follow-up — SCM `prescribe="fluxes"` → CLUBB native BC:** the SCM
  prescribed-flux path injects the kinematic flux as a lowest-cell HOST mean-tendency
  and zeroes `Ch_neutral`, so CLUBB's native `wpthlp_sfc`/`wprtp_sfc` moment BC is 0
  (scheme responds only to the host-warmed gradient). The scheme-side half is DONE
  (iter 76 — `clubb_step` accepts `sfc_*` kinematic fluxes). Remaining: route the SCM
  `w_th_s`/`w_qv_s` into those args. CROSS-CUTTING (needs `PhysicsState` surface-flux-
  override fields + the SHARED generic `turb_fn` dispatch in `integration.py`, which has
  a fixed signature across all schemes + `scm.py`) → deferred to a human-directed PR, not
  done autonomously (iter 78/80 re-confirmed the risk).
- **AMIP-CLI prognostic exposure — investigated, NOT shipped (iter 83):** attempted to
  expose prognostic CLUBB through the production AMIP CLI (`--clubb-prognostic` → an
  additive `ExperimentConfig.clubb_prognostic` field → the `model_driver._run_mpas`
  `TurbulenceConfig` build site, with a fail-loud non-MPAS guard). **Codex caught two
  [high] blockers, both confirmed real → reverted the whole change:** (1) the MPAS run
  loop (`_run_mpas`) starts `_phys_state=None` and `update_physics_state(None, …)` returns
  `None` (verified at `physics_state.py:265`), so the `clubb_moments` carry is DISCARDED
  and RESEEDED every step — the prognostic kernel would run with no time carry (silently
  wrong, "succeeds" but isn't prognostic); (2) MPAS checkpoints persist only
  `u/T/p_s/phis/tracers`, so restarts reset the moments. This is a GENERAL latent
  limitation: `_run_mpas` does not carry ANY stateful-physics state (TKE/qke, prognostic
  convection/GWD, CLUBB) across steps — currently masked because MPAS AMIP runs use
  `--turbulence none`. Fix = initialize a real `PhysicsState` before the MPAS loop when
  stateful physics is active + persist the carry (incl. `clubb_moments`) in the MPAS
  checkpoint format + restore on restart. Deep run-loop-lifecycle + checkpoint-format
  surgery → human-directed PR. Prognostic CLUBB remains runnable/tested via the direct
  `combined.make_physics` MPAS path (iter 82, which threads `phys_state` explicitly) and
  `integrate_clubb_column`; only the *production CLI* exposure is blocked.
- **Default (diagnostic) CLUBB through MPAS (iter 84):** iter-82 covered only the opt-in
  PROGNOSTIC entry on MPAS; the DEFAULT `scheme="clubb"` (diagnostic `clubb_turbulence`:
  parcel-`Lscale` eddy diffusion + ADG1-PDF, `wp2` in the `tke` slot — the path most users
  hit) now has MPAS integration coverage too. `test_default_diagnostic_clubb_runs_through_
  mpas_driver` runs it end-to-end through the MPAS combined physics with a sheared
  edge-normal wind; asserts finite + structured `du_dt` AND wind-input SENSITIVITY
  (same-carry baseline vs half-wind, `max|du_base−du_half|>1e-5`) so the Perot edge↔cell
  reconstruction/projection is genuinely exercised. Codex flagged the same input-
  sensitivity gap as iter-82 (proactively-missed; fixed) → re-review approve.
- **`_safe_sqrt` de-duplication (iter 85, codex-approved):** the AD-safe double-where
  sqrt was copy-pasted byte-identically in 5 modules (clubb_wp23/mixing_length/moments/
  pdf_moments/pdf) → promoted to one public `clubb_helpers.safe_sqrt` (per CLAUDE.md
  "no duplicate numerics / use shared utilities") + re-pointed all 5 (behavior-preserving;
  108-test parity/oracle suites unchanged). `clubb_mfl._safe_sqrt` left local on purpose —
  it is a distinct NaN-PROPAGATING variant (verify-first caught this; consolidating would
  change behavior). Added `test_safe_sqrt_value_and_ad_safety`. NOTE (environment): the
  uncommitted de-dup was externally reverted mid-iteration once; re-applied and committed
  immediately — uncommitted work does NOT persist across this loop's resets, only commits do.
- **Prescribed-flux conservation triad (iters 77/78/79), all codex-approved:** for the
  iter-76 prescribed `sfc_*` BCs — **heat** (77) and **moisture** (79) are applied as
  EXACT flux-form Neumann lower-BCs (`Σ_k (ρ_k dz_k)(dT_dt/Π or dq_v_dt)_k = ρ_sfc·flux`
  to round-off, rel<1e-9, non-vacuous; constant prescribed flux is state-independent →
  EXACT, unlike the iter-64 O(Δt) bulk stress). **Momentum** (78) is fundamentally
  different: CAM's `l_imp_sfc_momentum_flux=.true.` (`advance_windm_edsclrm`) consumes
  ONLY the stress MAGNITUDE `u_*²=√(u'w'²+v'w'²)` and re-applies it as a drag
  ANTIPARALLEL to the wind — the prescribed AZIMUTH is discarded (`(W,0)`≡`(0,W)`, bit-
  identical tendencies; verified). Faithful CAM physics (exact for the wind-antiparallel
  bulk drag) — NOT a bug; the iter-76 docstring was corrected to say so (codex caught a
  dropped-`sqrt` slip in the fix). Tests: `..._closes_column_budget` (heat/moisture),
  `..._is_magnitude_only_drag` (momentum). (Aside iter 79: a CBL-growth integration test
  was ruled out — `integrate_clubb_column`'s bare-column driver goes grid-scale-unstable
  under strong sustained heating, the iter-48 instability.)
- **Sub-cycled prescribed-flux application (iter 81):** the prescribed heat flux is
  applied on EVERY sub-step of the `dt>clubb_dt` sub-cycling `lax.scan` in
  `clubb_turbulence_prognostic` (the realistic coupled regime; iters 77/79 only tested
  `n_sub=1`). `..._applied_through_subcycling` (dt=1800, clubb_dt=300 → n_sub=6) closes
  the column θl budget to ~6e-4 (a dropped/double-counted flux would be off by O(1) or
  an `n_sub` factor). Closure is APPROXIMATE not exact (vs n_sub=1 round-off): `ρ_sfc(t)`
  drift across sub-steps → residual grows ~linearly with `n_sub` (measured 2e-11/6e-4/2e-3
  at n_sub=1/6/12). NOTE: the sub-cycle is bare forward-Euler WITHOUT host diffusion (the
  dycore supplies that BETWEEN physics dt, not within one), so a strongly-SHEARED column
  destabilises over a long bare `dt` (O(1) residuals on `_scm_column`; iter-48 again) —
  the test uses a low-shear column where the sub-cycle stays stable. Codex: approve.
- **MPAS-driver integration coverage (iter 82):** prognostic CLUBB was claimed-supported
  on `model_type="mpas"` (the 2nd PhysicsState-persisting driver) but had ZERO integration
  coverage — `..._blocked_on_non_persisting_drivers` only asserted the build policy for
  hydrostatic. Added `test_prognostic_clubb_runs_through_mpas_driver`: builds + RUNS
  prognostic CLUBB end-to-end through the MPAS combined physics on a Voronoi mesh, where
  the turbulence path does a non-trivial Perot edge→cell wind reconstruction + cell→edge
  projection around the column scheme while threading the `clubb_moments` carry. **Codex
  caught two real [medium]s, both fixed:** (1) a rest state `u=0` leaves the edge↔cell wind
  bridge untested (an all-zero projection still gives finite `du_dt`) → drive a SHEARED
  (2→12 m/s) + per-edge-structured edge wind and assert `max|du_dt|>1e-5` + spatial
  structure; (2) the wind-sensitivity check confounded wind change with carry evolution
  (baseline from evolved carry vs half-wind from fresh) → compute both baseline and
  half-wind from the SAME `phys_state` so wind amplitude is the only changed input
  (`max|du_base−du_half|>1e-5`). Separate 2-step loop checks carry persistence/evolution.
  3rd codex pass: approve. Confirms the MPAS edge↔cell wind round-trip + moment carry work.
- **GABLS1 stable-BL — shipped (iter 72):** correctly-coupled SCM benchmark via
  `prescribe="T_s"` + ACTIVE bulk transfer (`Ch_neutral=1.5e-3`) so CLUBB's own bulk
  formula computes the surface heat flux → native `wpthlp_sfc` coupling drives the SBL
  (`_resolve_T_sfc`→`surface_T_sfc_override`→`clubb_step`). `--turbulence clubb` in
  `scripts/scm/gabls1.py`; RESPONSE-based pass gate (genuine cooling + bounded TKE, no
  no-op pass). mynn25 default unchanged.
- **Native coupling test (iter 73):** `..._surface_heat_flux_tracks_surface_temperature`
  — fast deterministic check that CLUBB's surface sensible flux tracks the air–surface
  contrast (cold⇒`shflx<0`, warm⇒>0, equal⇒0, antisymmetric) AND that it is COUPLED into
  the prognostic tendency (near-surface `dT_dt[:,-1]` responds with matching sign, ≫ the
  zero-flux floor — proving `shflx→wpthlp_sfc BC→advance_clubb_core→dT_dt`).
- **Full-suite single-process OOM fixed (iter 74):** an autouse fixture calling
  `jax.clear_caches()` after each test (correctness-neutral) lets the whole
  `test_clubb_scheme.py` run in one process (it `Fatal Python error: Aborted`'d
  mid-XLA-compile ~2/3 through as compiled executables accumulated). Full regression:
  286 module + 39 integration tests green.
- **Diagnostic-path conservation (iter 75):** the DEFAULT diagnostic `scheme="clubb"`
  (the opt-out path most users get) conserves column `q_v` AND `θ=T/Π` to round-off
  (rel<1e-12) under zero surface flux; non-vacuous without spin-up (fast guard).
- **Prescribed-flux feature (iter 76) — CLUBB's LES/SCM-intercomparison interface:**
  optional `sfc_wpthlp`/`sfc_wprtp`/`sfc_upwp`/`sfc_vpwp` (kinematic, `(ncol,)` or
  `None`) on `clubb_step`/`clubb_turbulence_prognostic`/`integrate_clubb_column`
  OVERRIDE the bulk lower-BC per moment (`None`→bulk, a static Python branch); reported
  `shflx`/`lhflx`/`ustar` made consistent (bulk passes through bit-unchanged; prescribed
  → W/m² + `ustar=(u'w'²+v'w'²)^¼`). **Codex [medium] fixed:** the bare fourth-root has
  +∞ slope at zero stress → floored the radicand with a `1e-30` AD safety floor (grad 0
  there, physical stresses bit-unchanged). Test `..._accepts_prescribed_surface_fluxes`
  (6 checks: round-trip, `T_sfc`-bypass, grad-coupling, zero-stress ustar grad-finiteness,
  sign, `None`==omitted back-compat). Default no-arg path bit-identical to before. The
  conservation contract for these BCs is the iter-77/78/79 triad above.

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

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

*(Compressed at iter 10/20/30/40/60/70; verbose per-iter logs dropped — see git
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

## Status @ iter 70 — ✅ DONE (compressed; iter 11–69 detail in git history)

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

**Validation hardening (iter 61–69; compressed at iter 70 — detail in git history).**
All in `tests/unit/test_clubb_scheme.py`; each codex-adversarially-reviewed to
clean. Common technique: spin up real moments via `integrate_clubb_column`
(dt=150,nsteps=40) so the property is **non-vacuous** (the rest/floor state gives
~round-off tendencies any scheme trivially passes).
- **Conservation triad @ scheme entry, zero surface flux:** column **rt** (moisture,
  iter 61) and **θl** (heat, iter 62) conserved to round-off (`<1e-12` rel/step,
  `ρ·dz` weight); **momentum** `Σ mass·du/dt = τ_x` (iter 64) — closes to **O(Δt)**
  (surface stress applied semi-implicitly; Richardson: Δt 10× → residual 10×;
  confirms interior flux-form conservation + the surface-stress SIGN, the iter-47
  bug). Winds use `advance_windm_edsclrm`, scalars use `advance_xm_wpxp`.
- **Bridge convention (iter 62, surfaced by codex):** `clubb_step` maps the mean
  back as `T_new = thlm·exner`, so reported `dT_dt/exner` is EXACTLY the prognostic
  **θl** tendency and `q_v`=`rtm` (total water) — prognostic CLUBB returns θl/rt,
  NOT a saturation-adjusted (T,q_v) split (cloud partition deferred to microphysics,
  per `l_rcm_supersat_adj=.false.`).
- **Turbulence→microphysics coupling (iter 63):** that θl/rt return is CORRECT for
  the legoESM pipeline (modules run on the same state, tendencies summed; `sundqvist`
  does the saturation adjustment `f·max(q_v−q_sat,0)`, `dT=L_v·net_cond/c_pd`).
  Verified CLUBB→sundqvist closes the column water budget to the surface-precip sink
  (rel<1e-12) with nonzero precip, condensation enthalpy-consistent (`c_pd·dT+L_v·dq_v`~0).
- **Moist-chain differentiability (iter 65):** `jax.grad` flows end-to-end through
  CLUBB→sundqvist incl. the `max(q_v−q_sat,0)` kink; guarded by a microphysics-only
  objective + active-condensation forward check (so a dead kink can't pass).
- **The defining fuller-than-clubb_lite signature (iter 66):** under surface heating
  prognostic CLUBB develops buoyancy-driven vertical-velocity SKEWNESS (`wp3`>0 in
  the upper mixed layer, ~+0.06–0.15) — the third moment driving non-local transport;
  the test EXECUTES `clubb_lite_turbulence` and asserts it has no `wp3` (down-gradient,
  no 3rd moment). Contrast vs a verified near-neutral control. (Aside: a cooled-surface
  'stable' control is unusable — fixed cold `T_sfc` over-cools the air → convection.)
- **float32 / Metal cross-backend (iter 67–69):** FIXED a real bug — the parcel-Lscale
  `compute_mixing_length` had strong-float64 sources (`jnp.float64(0.0)` scan carries,
  default-dtype `jnp.zeros` pads / `jnp.full` col / `set_Lscale_max` cap) that promoted
  a float32 column and CRASHED `lax.scan`'s carry-type check under x64. Fix: normalize
  EVERY float input (state, `Lscale_max`, `mu`, `lmin`, all `CLUBBGrid` fields) to
  `dt_f=thlm.dtype` at function top — preserves float64 EXACTLY (8 golden-parity tests
  pass). Swept all `clubb_*.py` → this was the ONLY strong-float64 source (others use
  dtype-preserving `zeros_like` or weak `full(py_float)`). Both entry points verified
  float32-finite, no-promotion, AND numerically faithful (float32 Lscale matches f64
  to ~1e-7). Caveat (documented, inherent, not a defect): `dT_dt=Π·(θl_new−θl)/dt` is
  a difference of two ~300 K values → ~5–10% float32 cancellation noise (shared by
  clubb_lite + diagnostic path); Lscale/Km/moisture unaffected.

**Production-SCM runnability (iter 71):** confirmed prognostic CLUBB RUNS through
the real `SingleColumnModel` driver (not just the standalone `integrate_clubb_column`)
on a Wangara-style convective-BL setup — 40 steps finite, `T_low`=277.5 K (physical),
`wp2` develops. `clubb` is registered in the SCM's `stateful_turb` set, so the
production single-column driver supports it. NOTE: SCM `.run()` is an EAGER Python
loop (~8 s/step with full prognostic CLUBB, ~50 s first-step compile) → use the
jitted `integrate_clubb_column` for efficient multi-step testing.
- **Deferred (codex-flagged, NOT shipped):** packaging this as a selectable
  `--turbulence clubb` option for the Wangara *benchmark* needs proper surface-flux
  coupling first. The SCM `prescribe="fluxes"` path injects the kinematic flux as a
  lowest-cell HOST tendency and zeroes the scheme's bulk flux (`Ch_neutral=0`, same
  for the existing mynn25 case), so CLUBB's native `wpthlp_sfc`/`wprtp_sfc` moment
  lower-BC is 0 — the scheme responds only to the host-warmed gradient, not its
  surface-flux BC. The scheme-side half is now DONE (iter 76): the CLUBB driver
  accepts prescribed kinematic surface fluxes directly (`sfc_wpthlp`/`sfc_wprtp`/
  `sfc_upwp`/`sfc_vpwp` feed its native moment lower-BC). Remaining for a Wangara
  `--turbulence clubb` benchmark: wire the SCM `prescribe="fluxes"` `w_th_s`/`w_qv_s`
  into those args (small SCM-driver plumbing) + a physically-calibrated `wp2` gate
  (the `1e-3` floor is too weak). Tracked as a follow-up.
- **Prescribed heat-flux column-budget closure (iter 77):** the strongest contract
  on the iter-76 prescribed-flux BC — `test_prognostic_clubb_prescribed_heat_flux_
  closes_column_budget` verifies a prescribed constant `sfc_wpthlp=W` is applied as an
  EXACT flux-form Neumann lower-BC: `Σ_k (ρ_k dz_k)(dT_dt_k/Π_k) = ρ_sfc·W` to round-off
  (measured rel ~1e-11, assert <1e-9; non-vacuous warming >1e-3). Unlike the iter-64
  surface-stress momentum budget (state-dependent `τ=−ρC_d|V|u` → O(Δt) semi-implicit
  residual), a prescribed *constant* flux is state-independent → closure is EXACT with no
  Δt dependence. Confirms correct magnitude + no double-counting (a doubled application
  would give 2×). Codex adversarial review: approve, no findings. (Test-only iteration;
  no production code changed.)
- **Prescribed MOMENTUM semantics corrected + pinned (iter 78):** investigating a
  momentum analogue of the iter-77 budget revealed the prescribed momentum BC behaves
  fundamentally differently from heat/moisture — a Δt-INDEPENDENT non-closure (col_dv≈0
  for a cross-wind-prescribed stress). Root cause: CAM's `l_imp_sfc_momentum_flux=.true.`
  wind advance (`advance_windm_edsclrm`/`windm_edsclrm_lhs`) consumes ONLY the stress-
  vector MAGNITUDE `u_*^2 = sqrt(u'w'_sfc²+v'w'_sfc²)` and re-applies it as an implicit
  drag ANTIPARALLEL to the near-surface wind — the prescribed AZIMUTH is discarded
  (confirmed: prescribing `(W,0)` vs `(0,W)` gives bit-identical `du_dt`/`dv_dt`). Heat/
  moisture, by contrast, enter `advance_xm_wpxp` directly as exact directional flux BCs
  (iter-77). This is faithful CAM physics (correct for prescribed-`u_*` LES forcing; exact
  for the bulk drag, which is already wind-antiparallel) — NOT a bug — but the iter-76
  docstring misleadingly implied directional component prescription. Fixed: corrected the
  `clubb_step` contract docstring (heat/moisture = exact directional; momentum = magnitude-
  only wind-opposing drag, azimuth discarded) + added `test_prognostic_clubb_prescribed_
  momentum_flux_is_magnitude_only_drag` (direction-independence bit-identity, ustar=
  `(u'w'²+v'w'²)^¼` round-trip, magnitude scaling + drag sign). **Codex [medium] caught a
  real formula slip** (docstring dropped the sqrt: wrote `u_*²=u'w'²+v'w'²` instead of
  `sqrt(...)`); fixed both docstrings, re-review approve. No behavior change.
- **Prescribed MOISTURE-flux closure — triad complete (iter 79):** `test_prognostic_
  clubb_prescribed_moisture_flux_closes_column_budget` mirrors the heat closure for the
  total-water channel: a prescribed `sfc_wprtp` is an exact flux-form Neumann BC, so
  `Σ_k (ρ_k dz_k) dq_v_dt_k = ρ_sfc·w'rt'_sfc` to round-off (rel<1e-9; non-vacuous
  moistening >1e-6; `q_v=rtm`, no exner). Pins that the `rt_tol` positivity floor (which
  `thlm` lacks) does NOT break closure for a normal moist column (`q_v~1e-3 >> rt_tol`).
  Completes the prescribed-flux conservation triad: heat exact (77), momentum magnitude-
  only drag (78), moisture exact (79). Codex review: approve, no findings. (Also ruled out
  a CBL-growth integration test: a diagnostic showed `integrate_clubb_column`'s bare-column
  driver goes grid-scale-UNSTABLE under strong sustained surface heating — `wp2`~16, θ
  profile non-monotone — the iter-48 instability; the SCM stand-in is unfit for long
  strong-forcing evolution, so that route was correctly NOT pursued.)
- **GABLS1 stable-BL — shipped (iter 72):** the correctly-coupled SCM benchmark.
  GABLS1 uses `prescribe="T_s"` (cooling surface temperature) with the bulk
  transfer ACTIVE (`Ch_neutral=1.5e-3`), so the surface heat flux is computed from
  the prescribed `T_sfc` by CLUBB's OWN bulk formula → CLUBB's native `wpthlp_sfc`
  coupling drives the stable BL (verified: `_resolve_T_sfc` reads the injected
  `surface_T_sfc_override` → `clubb_step`→`compute_surface_fluxes`). Added
  `--turbulence clubb` to `scripts/scm/gabls1.py`; a 20-step run is finite,
  `T_low`=263.8 K (cooled), `wp2max`=0.105 (weak/bounded, as a stable BL should be).
  The pass gate was made RESPONSE-based (codex-flagged): requires genuine cooling
  (`T_low<T_low_init−0.05·hours`) + turbulence above the rest floor and bounded
  (`1e-4<tke_max<5`), so an unchanged/no-op column can no longer pass. mynn25
  default unchanged; existing `test_scm_gabls1.py` (calls `build_scm`/`scm.run`
  directly) unaffected.
- **native surface-flux coupling test (iter 73):** `test_prognostic_clubb_surface_
  heat_flux_tracks_surface_temperature` — a DIRECT, fast, deterministic check that
  CLUBB's surface sensible-heat flux responds to the air–surface contrast (the
  coupling GABLS1 relies on): cold surface (`T_sfc<T_air`) ⇒ `shflx<0` (downward,
  cools/stabilises the SBL), warm ⇒ `shflx>0`, equal ⇒ 0, and antisymmetric
  (`shflx(+ΔT)=−shflx(−ΔT)`). codex first caught a SLOW SCM-run version as unsound
  — its "near-surface cooling ⇒ coupling live" inference was false (over a short
  window GABLS1's `T_s`≈265 K is initially WARMER than the ~264.6 K air, so the
  early surface flux is upward; the observed cooling came from turbulent mixing,
  not the surface sink). Replaced with this direct assertion. A 2nd codex pass then
  noted asserting only `out.shflx` proves the *diagnostic* flux but not that it's
  *coupled* into the prognostic tendency → added a 2nd layer: the near-surface
  `out.dT_dt[:,-1]` must respond with the matching sign (cold cools, warm warms,
  antisymmetric, ≫ the ~1e-8 zero-flux floor), proving the full chain
  `shflx→wpthlp_sfc BC→advance_clubb_core→dT_dt`. No SCM run, deterministic.
- **full-suite single-process OOM fixed (iter 74):** a comprehensive regression
  sweep revealed `test_clubb_scheme.py` cannot run all ~40 tests in one process —
  it `Fatal Python error: Aborted`s mid-XLA-compile ~2/3 through (the heavy
  prognostic-pipeline/grad/sub-cycling tests each lower a huge program and the
  compiled executables accumulate). Confirmed environmental, NOT a regression:
  every test passes in isolation (the aborted `subcycling_raw_moisture_contract`
  passes alone in 89 s), and ~24 integration + ~240 module tests were green before
  the abort. Fix: an autouse fixture calling `jax.clear_caches()` after each test
  (correctness-neutral — tests don't reuse compiled fns) so the suite stays
  runnable in one process. A real CI-runner hazard, now removed. Full regression
  after the iter-67 numerics change: 286 module + 39 integration tests all green.
- **diagnostic-path conservation (iter 75):** the DEFAULT diagnostic `scheme="clubb"`
  (what most users get; the prognostic triad only covered the opt-in path) conserves
  column `q_v` AND `θ=T/Π` to round-off (rel<1e-12) under zero surface flux — its
  flux-form `implicit_vertical_diffusion` (dry `rcm=0` mapping) only redistributes.
  Non-vacuous WITHOUT spin-up (`max|dq·dt|`~1e-4; the diagnostic eddy diffusion is
  driven directly by the initial gradient), so it's a fast (~9 s) guard.
- **Prescribed surface fluxes — CLUBB's LES/SCM-intercomparison interface (iter 76):**
  added optional `sfc_wpthlp`/`sfc_wprtp`/`sfc_upwp`/`sfc_vpwp` (kinematic surface
  fluxes, each `(ncol,)` or `None`) to `clubb_step` + `clubb_turbulence_prognostic`
  + `integrate_clubb_column`. When given, each OVERRIDES CLUBB's bulk lower-BC for
  that moment (`w'thl'` [K m/s], `w'rt'` [kg/kg m/s], `u'w'`/`v'w'` [m²/s²]); `None`
  falls back to the bulk formula (the `None`-test is a static Python branch, CLAUDE.md
  feature-gating exception — never a traced `jnp.where`). This is exactly the BC that
  the standard prescribed-flux LES cases (BOMEX/DYCOMS/ARM) specify, which the prior
  `prescribe="T_s"`-only coupling could not drive (the iter-71 Wangara deferral). The
  bulk formula is skipped entirely only when all four are prescribed → the result is
  then independent of `T_sfc`/`q_sfc`. Reported `shflx`/`lhflx`/`ustar` are made
  consistent: bulk values pass through bit-unchanged (verified back-compat), prescribed
  kinematic fluxes are converted to W/m² (`shflx=wpthlp·ρ·c_pd·Π`, `lhflx=wprtp·ρ·L_v`)
  and `ustar=(u'w'²+v'w'²)^¼`. **Codex [medium], fixed:** the bare fourth-root has +∞
  slope at zero stress, so `jax.grad` through `ustar` at the valid `sfc_upwp=sfc_vpwp=0`
  BC was non-finite → floored the radicand with a `1e-30` AD safety floor (gradient 0
  there, physical stresses bit-unchanged); codex re-review: approve, no findings. Test
  `test_prognostic_clubb_accepts_prescribed_surface_fluxes` (one shared spin-up via the
  grid-consistent `_scm_column`+`integrate_clubb_column`, then 6 checks): W/m²+ustar
  round-trip; `T_sfc`-independence (bulk bypass); grad-coupling of the heat-flux BC into
  the prognostic advance; the zero-stress ustar grad-finiteness regression; surface-heat
  sign; explicit-`None`==omitted back-compat. The default (no-arg) path is bit-identical
  to before (7-test existing-prognostic regression subset stays green).

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

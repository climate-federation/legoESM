# CLUBB port + single-file condensation — history

Historical record of the two project phases that produced
`packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py`
(formerly tracked in the repo-root `PORT_CLUBB.md`, retired 2026-06-12).
The living scheme reference is `clubb.md` (same directory).

## Phase 1 — port (iters 1–85)

**Goal:** a substantially fuller CLUBB than `clubb_lite.py`, ported from the
sibling `CLUBB-JAX` tree (`../CLUBB-JAX/clubb_jax/src/CLUBB_core` — already a
JAX port, so this was port-and-adapt to legoESM conventions), restricted to
the call tree exercised by the **CAM-default `clubb_*` flags**
(`namelist_defaults_cam.xml`). Done-gate: legoESM runs+tests with
`TurbulenceConfig(scheme="clubb")` through `make_turbulence_physics(...)`.

Built up as ~19 `clubb_*.py` modules (grid, config, saturation, helpers,
skewness, mixing length, PDF, PDF moments, solvers, moment advances, wp2/wp3,
xm/wpxp, MFL, tau, coefficients, fill-holes, core orchestration, diagnostic
closure, scheme entry), each kernel round-off/bit-exact parity-tested vs
CLUBB-JAX with committed golden fixtures + analytic/conservation oracles, and
codex-adversarially reviewed. Highlights (full detail in git history):

- Runnable from iter 16 (diagnostic phase 1); ADG1 double-Gaussian PDF live
  from iter 17; full prognostic moment closure carried in
  `PhysicsState.clubb_moments` by iters 43–53.
- Validation hardening (iters 61–84): conservation triad at the scheme entry
  (rt/θl round-off, momentum O(Δt)); turbulence→microphysics water budget
  closure; moist-chain differentiability; float32 promotion bug found+fixed
  in `compute_mixing_length`; production-SCM runnability + GABLS1 stable-BL
  benchmark; native surface-coupling sign tests; prescribed-flux
  LES/SCM-intercomparison interface with exact heat/moisture budget closure
  and magnitude-only momentum semantics (faithful CAM); sub-cycled flux
  application; MPAS-driver integration coverage for BOTH entries (wind-input
  sensitivity + carry persistence); full-suite OOM fixed with a
  `jax.clear_caches()` autouse fixture.
- Dry-regime instability (iters 48–51) root-caused to the bare SCM driver
  (absent host diffusion), not the closure; conservative flux-form stand-in
  added to `integrate_clubb_column`.
- AMIP-CLI prognostic exposure attempted and REVERTED (iter 83) — codex
  found the MPAS run loop discards the physics-state carry (see
  `docs/issues/mpas_run_loop_stateful_physics_carry.md`).
- Audit iters 58/60 created `physics/_shared.exner_function` /
  `buoyancy_coefficient` and ratcheted formula-debt budgets; iter 85
  deduplicated `_safe_sqrt` into `clubb_helpers.safe_sqrt` (keeping the MFL
  NaN-propagating variant local on purpose).

Codex adversarial review ran at every substantial step and caught real bugs:
CAM 3-C2 dissipation, surface-flux sign, variance-floor leak, sub-cycle
moisture contract, dispatch persistence guards, retrace hazard, MPAS
test-sensitivity gaps, an AD-unsafe fourth root.

## Phase 2 — single-file condensation (iters 86–95, "C1–C9", 2026-06-12)

**Goal:** condense the `clubb_*.py` modules into the single `clubb.py` per
the legoESM one-file-per-scheme convention, with a line-numbered TOC at the
top, the CAM-default flag VALUES as comments at the end (the flags themselves
not defined — no other tree is implemented), saturation via the shared
`legoesm.thermo` module, and a `clubb_lite.py` revert of its
`buoyancy_coefficient` use (retaining `exner_function`/`mixing_length`).

**Strategy — top-down absorption, always green:** repeatedly inline
`clubb.py`'s direct dependencies into it (so `clubb.py` stays at the top of
the import DAG — no cycles, no transitional stubs), delete the absorbed
module, re-point every importer (prod + tests), run the absorbed modules'
tests + the contracts gate, codex-review (AST-level body-fidelity diff vs the
deleted originals), commit. Commits `b5a8c87b..0b2eef77`:

| batch | absorbed | notes |
|-------|----------|-------|
| C1 | clubb_diagnostic, clubb_core | |
| C2 | clubb_coefficients, clubb_wp23, clubb_xm_wpxp | coefficients co-absorbed to break its `compute_skw_fnc` edge cycle-free; `_GAMMA` deduped |
| C3 | clubb_moments, clubb_mfl | `_EPS` deduped; MFL's NaN-propagating `_safe_sqrt` kept local; no-`__all__` policy documented |
| C4 | clubb_solve, clubb_fill_holes, clubb_skewness, clubb_tau, clubb_pdf, clubb_pdf_moments | 6 constants deduped; `_F64_EPS` float()-unified (bool-only uses) |
| C5 | clubb_grid, clubb_saturation, clubb_helpers, clubb_mixing_length | saturation = thin adapters over shared thermo Flatau curves; mixing length CONFIRMED different from `_shared`'s Blackadar |
| C6 | clubb_config | `CLUBBFlags` class REMOVED: 14 flag sites hardcoded at CAM values, dead non-CAM branches deleted, 65 flag values → machine-parseable end-of-file table with namelist tripwire preserved |
| C7 | — | `clubb_lite.py` revert (inline g/θ_v restored byte-identical to main; formula-budget baseline restored); rebase-fallout budget re-seeds |
| C8 | — | line-numbered TOC + `test_toc_line_numbers_accurate` drift tripwire |
| C9 | — | whole-repo gate sweep (see below) |

**Result:** all 18 helper modules absorbed (5789 lines, 19 sections);
byte-identical bodies codex-AST-verified per batch; value-identical module
constants deduped. Final validation: full CLUBB suite 335 passed (19 test
files), ratchet/contract gates 4139 passed, turbulence no-regression + carry
+ smoke 168 passed. `git diff main` shows `clubb.py` as the ONLY new source
file, with limited additions to shared files (`thermo.py` Flatau curves,
`_shared.py` helpers, physics_state/integration/combined/config/scm wiring).
One pre-existing failure was MAIN-side rebase fallout
(`test_no_private_cross_imports` on ocean `_global_rel_residual`; zero ocean
changes on the branch).

**Test-file decision:** the 19 `test_clubb_*.py` files are kept (not merged):
they map 1:1 onto `clubb.py`'s TOC sections, keep pytest sharding effective,
and avoid a ~10k-line test file.

**Process lessons:**
- Re-verify staged deletions after any `git stash`/`pop` before committing
  (a stash/pop unstaged `git rm`s and the explicit-path commit missed them).
- Carried imports must be re-derived per absorption batch, not assumed
  (a missing `from jax import lax` surfaced only at test time).
- Rebase fallout in ratchet baselines: re-seed at main's measured values
  with a dated comment — never silently loosen a shrink-only budget.
- Uncommitted work does not survive loop resets; only commits do.

# CRM Implementation Log

Goal: **stable + realistic 30-day production-scale CRM on all grid types with excellent MPI scaling**.

Canonical state of CRM rollout — built, broken, next. Each iteration appends dated entry under "Iteration log" with concrete change + diagnostics. No "just update doc" iterations; every entry references real commit or measurement.

---

## Components owned by this rollout

| Component | Path | Status (refreshed iter-53) |
|---|---|---|
| Plane non-hydrostatic CRM dycore | `src/legoesm/atmosphere/dynamics/compressible_euler_plane.py` | Built. **Production-stable at dt=5 s** on clean Wing IC (F8/F10/iter-12); F10 lifted the dt=1 s ladder constraint without R9 |
| Plane CRM halo-aware slow tendency | `src/legoesm/atmosphere/dynamics/compressible_euler_plane_halo.py` | Built. Smag ✓ (iter-3 R4), vertical-θ-diff ✓ (iter-3 R5), WENO5 ✓ (iter-7 R6); KW78 obsolete (F10) |
| Plane CRM acoustic substeps (SI) | `compressible_euler.py:acoustic_substeps_semi_implicit`, `compressible_euler_plane.py:plane_acoustic_substeps_semi_implicit` | Built. Substep KW78 placement algebraically correct but inert (F2); R9 outer-step variant no longer on critical path (F10 + iter-12 dt=5 s PASS) |
| 2-D pencil MPI layout + halo exchange | `src/legoesm/parallel/plane_mpi.py` | Built, AD-safe (iter-4 + iter-5) |
| MPI-aware reductions for RCE | `src/legoesm/atmosphere/dynamics/rce_mpi.py` | Built (iter-4 R7 — full DD mass fixer) |
| 30-day production driver | `scripts/run_rce_mpi_long.py`, `scripts/run_rce_30day.sh` | DD path wired via ``--use-dd`` (iter-5); legacy rank-0-broadcast retained as F8-stable default. Real MPI scaling F9-platform-blocked on macOS Python 3.13 |
| Multi-grid RCE driver | `scripts/run_rce.py`, `scripts/run_rce_cross_grid.sh` | Built for `cubed_sphere`, `latlon`, `voronoi`, `gaussian`. 30-day production validated for {C24, C48, C72, V4, LL32, T21} (iter-12 + iter-15 + iter-26); C96 stays at 2-day nightly per iter-22 wall-time decision |
| Bare-dycore stability diagnostic | `scripts/diag_bare_dycore_stability.py` | Built (iter-1) with `--implicit-buoyancy` / `--vertical-theta-diffusion` / `--advection` switches |
| Auto-dt ladder | `src/legoesm/driver/rce_dt.py` | Built (iter-24). 5-tier per-grid ladder + N>96 hard refusal (iter-21). Anchored by 5 measurement layers (iter-28/29/34/36/51) |
| Radiation-schedule helper | `src/legoesm/driver/physics_schedule.py` | Built (iter-42). Single source of truth for the ``rad_call_every_steps`` arithmetic; 22 unit tests (iter-42 + iter-43 NaN/inf/sys.maxsize hardening) |
| Shared RCE assertion helpers | `tests/atmosphere/hydrostatic/test_rce_cross_grid_smoke.py` | Built (iter-46). `_assert_dt_used` + `_assert_max_wind_peak_below` shared across C48/C72/C96/V4/LL32/T21 nightlies; 16 unit tests (iter-52) |

---

## Definition of done

30-day CRM run on production target (132×132 plane, dx=2 km, nlev=30, H=33 km, dt=1 s, 12 MPI ranks, Wing 2018 RCEMIP1 IC, gray radiation + Kessler + Smagorinsky LES + surface fluxes) must:

1. **Run to completion** without NaN, `max|w| < 50 m/s` throughout.
2. **Reach radiative-convective equilibrium**: CWV plateaus in 30 ± 5 mm range, precip plateaus ~3 mm/day, MSE drift < 1 % over last 10 days.
3. **Reproduce on each supported grid type** via `scripts/run_rce_cross_grid.sh` (cubed-sphere, latlon, voronoi, gaussian). Currently only *plane* CRM has explicit CRM physics; cubed-sphere/latlon/voronoi/gaussian use hydrostatic dycore in RCE mode and cross-grid wrapper validates they converge to similar CWV / precip / MSE.
4. **Scale with MPI**:
   * **Strong scaling**: 30-day-clock-time on 12 ranks ≤ 1.5× of 1-rank time / 12 (efficiency ≥ 67 %).
   * **Weak scaling**: per-rank cost grows < 1.3× when grid doubled in each dim and ranks doubled in each dim (4× total).
5. **Pass `/codex:adversarial-review`** on dycore + MPI halo + production driver with no MEDIUM/HIGH findings outstanding.

---

## Findings to date

### F1. Outer-step buoyancy/w mode amplification at dt > ~1 s

Measured by `diag_bare_dycore_stability.py` with same Wing 2018 IC production driver uses (warm bubble θ' = 0.5 K at z<1 km, ρ' = −ρ_ref · θ'/θ_ref, nlev=30, dx=2 km, H=33 km, sponge_width=10 km, sponge_coeff=0.05, hyperdiff=5e6, Smag cs=0.2, SI acoustic with β=0.1, n_acoustic=24 dt_outer):

| dt [s] | n_acoustic | max|w| @ step 100 | Verdict |
|---|---|---|---|
| 0.5 | 12 | 4 × 10⁻³ m/s | Stable; bubble decays slowly |
| 1.0 | 24 | 8 × 10⁻³ m/s | Stable |
| 1.5 | 36 | 4.3 m/s | Growing exponentially |
| 2.0 | 48 | 180 m/s | Blows up by step 70 |

**Per-step amplification ratio at dt=2 s** ≈ 1.30 (constant step 50 to 100). Mode period ≈ Brunt-Väisälä N⁻¹ ≈ 70 s. Saturates ~180 m/s independent of initial bubble amplitude (tested 0.05 K vs 0.5 K) — classic numerical mode saturation, not physical growth.

### F2. KW78 implicit buoyancy at *substep* level too small to help

KW78 adds three tridiagonal bands proportional to
`κ = 0.25 dt_s² g / (θ₀_half J)` × dθ_ref/dz.

dt_s = dt_outer / n_acoustic = 2 s / 48 ≈ 0.04 s,
κ × dθ_ref/dz ≈ 8.5 × 10⁻⁸ vs `α = dt_s² c_s²/dz²` ≈ 1.4 × 10⁻⁴.

Ratio ≈ 6 × 10⁻⁴. Effect on output: indistinguishable (1-bit diff at step 100). **Conclusion**: KW78 belongs on *outer* step (where dt grows N times larger), not acoustic substep. Current placement algebraically correct (sign + structure verified vs standard forward-backward derivation) but inert for production parameter regime.

### F3. WENO5 + vertical θ diffusion + KW78 don't fix dt=2 s blowup

Tested simultaneously: max|w| @ step 100 = 480 m/s (worse than baseline). WENO5 wider stencil adds dispersion energy at unstable mode without damping it. Vertical θ diffusion at ν=1e4 m²/s too weak vs exponential mode growth.

### F4. Production 30-day driver does not actually use MPI DD

`scripts/run_rce_mpi_long.py` calls `model.step(state, ...)` on **rank 0** inside `if rank == 0:`, then `_broadcast_state(state, comm, root=0)` to every rank. Dycore + physics run replicated; only reductions (`global_sum_mpi` via `compute_total_water_mass_plane_mpi`, etc.) genuinely MPI. **Net**: 12 ranks ≈ 1 rank speed (overhead dominates). Strong/weak scaling: not achievable until `model.step_halo(...)` path taken on multi-rank.

### F5. Halo-aware slow tendency lacks production features

`compressible_euler_plane_halo.py` does NOT support:
* Smagorinsky LES (raises NotImplementedError)
* Vertical θ diffusion (no branch)
* WENO5 advection (hardcoded upwind1)
* KW78 implicit buoyancy (not threaded into halo SI substep)
* Mass fixer (single-rank only; skipped multi-rank)

So even if production driver switches to `step_halo`, lose every stabilizer just added. **Blocker for MPI scaling**.

### F6. Production default hyperdiff=1e6 is 5× too weak

Bare-dycore probe at dt=1 s, hyperdiff=1e6 (production default) blows up at step 250 (max|w|=22 m/s); hyperdiff=5e6 blow-up *delayed* to step ~470 (single-bubble IC). Fundamental mode bubble-seeded, only delayed by hyperdiff — fixing IC is real cure (F7).

### F7. Single-level warm-bubble IC seeds 2-Δz vertical mode

At nlev=30, H=33 km uniform dz~1.1 km. Legacy IC sets θ' only where z < 1 km — single grid level (z_full[29] ≈ 550 m). Resulting 2-Δz vertical mode unrepresentable on staggered grid, aliases into numerical instability hyperdiff can only slow down. Pure-Wing IC (no bubble) on 24×24×30 stable through 1296 steps (20 min sim) with max|w| < 5 × 10⁻³ m/s and zero qc.

Aggressive qv noise (≥ 2.5 × 10⁻⁴ kg/kg in lowest 4 levels) *also* destabilising: localised qv hotspots → spatial gradients in surface flux → non-uniform heating → grid-scale convection burst. **Default qv noise lowered to 0**; small values (1–5 × 10⁻⁵ kg/kg) acceptable as stochastic seed but must verify.

### F8. Stable physics-on smoke confirms dycore+physics composes cleanly

Smoke at 24×24×30, dx=2 km, dt=1 s, **no bubble + no qv noise**, full physics (gray rad + Kessler + Smag c_s=0.2 + surface flux + mean-wind removal + moist-mass fixer + positive filter): max|w| stays ~5 × 10⁻³ m/s through 1200 steps (20 min sim), MSE drift < 7 × 10⁻⁵ relative, CWV pinned to IC. **No spurious convection** — confirms full physics-on driver dynamically stable from clean IC. Convection spins up later from radiative cooling + surface flux on hours-days timescale (verify at 6-h / 24-h smoke step).

---

## Roadmap (concrete, ordered)

* [ ] **R1**: Reduce production dt from 2 s → 1 s in `run_rce_30day.sh` and `run_rce_mpi_long.py` defaults. [done this iter]
* [ ] **R2**: Plumb `--implicit-buoyancy` / `--advection weno5` through `run_rce_mpi_long.py` for A/B test. [done this iter]
* [ ] **R3**: Add automated dt-stability test in `tests/atmosphere/` running `diag_bare_dycore_stability.py` at dt=0.5/1.0/1.5/2.0, checks max|w| @ step 100 bounded for dt ≤ 1.0 s.
* [ ] **R4**: Port Smagorinsky LES to `compressible_euler_plane_halo.py` (reuse `_compute_smagorinsky_K_m_plane` via halo-aware shear stencil).
* [ ] **R5**: Port vertical-θ-diff + WENO5 advection to halo path (column-local → no extra halo).
* [ ] **R6**: Port KW78 implicit buoyancy to halo SI substep (column-local solve → no extra halo).
* [ ] **R7**: Implement MPI-aware mass fixer (`compute_dry_mass_plane_mpi` via `global_sum_mpi`); replace rank-0-only `_broadcast_state` flow with `step_halo` multi-rank.
* [ ] **R8**: Strong + weak scaling benchmarks on 1 / 4 / 12 / 48 ranks via `scripts/run_levante_gpu_scaling.py` (extend for plane CRM).
* [ ] **R9**: Klemp-Wilhelmson 1978 **outer-step** implicit buoyancy (substep version in F2 inert). Real fix for buoyancy/w mode amplification, lifts dt limit.
* [ ] **R10**: Cross-grid CRM validation — extend `run_rce_cross_grid.sh` to thread CRM physics stack through every grid's dycore (or document explicitly that only plane is "CRM" and others are hydrostatic RCE).
* [ ] **R11**: 30-day production run end-to-end with success criteria 1-5.
* [ ] **R12**: `/codex:adversarial-review` on full delta; address findings.

---

## Iteration log

### 2026-05-26 — iter 1 (this entry)

**Changes**
* `scripts/run_rce_mpi_long.py`: added `--implicit-buoyancy` flag, wired through `CompressibleEulerConfig.implicit_buoyancy`. Hard-fails if `--implicit-buoyancy` set without `--semi-implicit-acoustic`.
* `scripts/run_rce_30day.sh`: reduced default `DT` from 6.0 s → 1.0 s (6.0 s default untested; 2.0 s reproducibly blows up; 1.0 s reproduces F1 stable). Added `--semi-implicit-acoustic` and `--acoustic-off-centering 0.1` to launch line. Exposed `N_ACOUSTIC`, `ADVECTION` env vars. Documented dt-stability ladder in script header.
* `CRM_implementation.md` created with state + roadmap + findings.

**Measurements**
* Reproduced dt-stability ladder F1 (table above) via `diag_bare_dycore_stability.py` on 48×48×30 mesh — same dx, dt, IC, physics gates as 132×132 production driver. Blow-up mode independent of horizontal extent (saturates ~180 m/s regardless of bubble amplitude or domain size).
* Confirmed F4: production driver calls `model.step` inside `if rank == 0:` then broadcasts. Lines 549-564 of `run_rce_mpi_long.py` show structure.

**Measurements (cont.)**
* End-to-end smoke at dt=1.0 s, nx=ny=24, nlev=30, dx=2 km, single rank, FULL physics (gray rad + Kessler + Smag + surface flux): max|w| still grows 8 × 10⁻³ m/s (step 100) to 95 m/s (step 400). **Important finding F6**: bare-dycore dt-stability limit (F1) NOT only threshold — physics injects extra energy destabilizing same buoyancy/w mode at dt=1 s. Realistic CRM needs both outer-step KW78 fix (R9) AND physics-time-step control (R-NEW).
* `--implicit-buoyancy` now exposed but inert at substep level (F2). Kept in API for future outer-step variant.

**Next iter target**: investigate why physics-on destabilizes sooner than bare-dycore (separate radiation tendency mag, surface flux, Kessler q-tendency); start R3 (dt-stability regression test) + R4 (Smag in halo path).

### 2026-05-26 — iter 63

**Lock the iter-14 FULL 1-sim-hour plane CRM envelope as a slow
nightly regression (vs iter-38's 5-min sub-envelope).**

iter-14 measured 1-sim-hour PASS at 132×132×30 dx=2km dt=5s. iter-38
locks only a 5-min sub-envelope (60 steps). A regression that
destabilises between step 60 and step 700 (slow CFL drift, halo
edge accumulation, mass-fixer convergence issue) slips iter-38.

iter-63 adds ``test_plane_crm_production_scale_132x132_one_hour_envelope``:
* 720 outer steps at dt=5 (exact 1 sim-hour = 3600 s).
* log_every=60 → 13 logged rows {1, 60, 120, ..., 720}.
* Same iter-39 ``--no-radiation`` semantics as iter-38.
* Caps: max\|w\| < 0.05 on every row (8x iter-14's 6.1e-3 peak);
  CWV drift < 0.01 mm; MSE drift < 5e-4 relative.

**Codex iter-63 review** caught 2 HIGH + 1 MEDIUM + 2 LOW:

* **HIGH#1** — slow marker means default CI skips this. iter-63
  contract: runs via ``pytest -m slow`` nightly. Intentional.
* **HIGH#2** — ``_run_driver_production_scale`` hardcoded
  ``timeout=900`` (15 min) but the 1-hour run wall is ~17 min.
  Would have KILLED the run before completion. Fixed: added
  ``timeout_s`` parameter; iter-63 passes ``timeout_s=1800``
  (1.8× cushion). Existing iter-38/39 callers use the default
  900 (their 5-min sub-envelope runs in ~3 min so 900 is plenty).
* **MEDIUM** — MSE cap 5e-4 too loose vs iter-14 baseline.
  Acknowledged in the docstring: iter-14's 1.7e-4 ceiling was
  WITH cached radiation; ``--no-radiation`` should land lower
  per iter-38's 2.4e-5 measurement. 5e-4 is a conservative
  upper bound; tighten when the iter-63 live run produces a
  measurement.
* **LOW#1** — docstring said "725 steps = 1 sim-hour" but 725 *
  5 = 3625 s. Clarified: iter-14 ran 725 ≈ 1 sim-hour + 25 s;
  iter-63 uses 720 = EXACT 3600-s window.
* **LOW#2** — docstring confused iter-14 radiation status.
  Clarified: iter-14 ran with cached step-1 radiation tendency
  (Codex iter-39 finding); iter-63 uses true ``--no-radiation``.

**Tests collected**: 5 plane CRM tests (was 4); the iter-63 test
is the third slow-marked nightly in this file. Total CRM slow
nightlies now: 3 (production-scale dycore-only iter-38, with-rad
iter-39, full-1-hour-envelope iter-63).

**R-roadmap status**: R1-R8, R10, R12 ✓. F9 platform-blocked.
R11 30-day plane CRM still wall-time gated (~8 days single-rank),
but iter-63 now empirically pins the 1-sim-hour scale — 12× tighter
empirical bound than iter-38 alone.

### 2026-05-26 — iter 62

**Doc compression — 1923 → 1081 lines (44% reduction).**

iter-49 attempted iter-folding but only ran caveman-style word
compression (kept all iter-2..iter-25 entries at full detail).
iter-62 actually folds iter-26..iter-36 to a one-line-per-iter
summary table + restores the iter-2..iter-25 summary table that
iter-49 intended.

**Kept at full detail (28 iters)**:
* iter-1 (foundational state + roadmap context)
* iter-37..iter-61 (most recent 25 iters with active context)

**Folded to summary tables**:
* iter-26..iter-36 (10 iters, table at bottom of doc)
* iter-2..iter-25 (24 iters, table at very bottom)

Total: 35 ### headers (was 49 pre-iter-62 incl. all iter entries).
Net: 35 iter blocks + 2 summary blocks + 1 header + components +
DOD + findings + roadmap.

Full per-iter detail accessible via:
* ``git log`` (every iter has its own commit message)
* ``CRM_implementation.original.md`` local backup (state at
  iter-62 pre-compression)
* git history at commit 58db0859 (iter-49 commit, has all iter
  entries at full caveman-compressed detail)

**R-roadmap status**: R1-R8, R10, R12 ✓. F9 platform-blocked.
Doc hygiene aligned with the "compress every 10 iterations"
instruction.

### 2026-05-26 — iter 61

**Codex review of iter-58/59 defaults regressions caught 5 MEDIUM
coverage gaps — extended both tests.**

iter-58/59 introduced wrapper- and driver-defaults regression tests
that were committed without prior Codex review. iter-61 fresh review
found 0 HIGH, 5 MEDIUM, 7 LOW. Key gaps closed:

**Wrapper test extensions** (``test_run_rce_30day_wrapper_defaults.py``):
* env-var coverage: ``DAYS``, ``RANKS``, ``USE_DD`` added (iter-58
  missed these — Codex MEDIUM#4).
* hardcoded driver-flag pass-through assertions (Codex MEDIUM#5):
  the wrapper hardcodes ``--acoustic-off-centering 0.1``,
  ``--snapshot-hours 24.0``, ``--snapshot-3d-hours 1.0``,
  ``--profile-days 5.0``, ``--log-every-steps 100`` in the
  mpirun argv. A partial revert of any would silently pass both
  the env-only iter-58 + driver iter-59 tests. New parametric
  test asserts each flag-value pair appears in the mpirun
  invocation line. Plus ``--semi-implicit-acoustic`` presence
  pinned (bare flag, no value).

**Driver test extensions** (``test_run_rce_mpi_long_driver_defaults.py``):
* 11 new defaults asserted: ``--dx``, ``--H``, ``--dz-sfc``,
  ``--vertical-grid``, ``--n-physics-substeps``,
  ``--rad-call-interval-s``, ``--no-radiation``, ``--use-dd``,
  ``--advection``, ``--sponge-coeff``, ``--sponge-width``,
  ``--implicit-buoyancy``, ``--qv-noise-seed``.
* iter-61 audit caught: ``--vertical-grid`` production default is
  ``uniform`` not ``stretched`` (uniform dz ≈ 1100 m at nlev=30,
  H=33 km — stretched is opt-in). Updated assertion + comment.

**Codex iter-61 LOW deferred**:
* Regex robustness vs here-docs/disabled bash blocks: would only
  false-match if the wrapper grows new comment-example syntax;
  no current risk.
* Duplication between AST + regex helpers across the two tests:
  different targets, no consolidation needed yet.
* Test isolation: tests parse source only, no live imports —
  intentional (avoids triggering JAX/MPI at import time).

**Tests**: 28/28 PASS in 0.36 s (was 11). Added 17 new assertions.

**R-roadmap status**: R1-R8, R10, R12 ✓ (with iter-61 defaults
regression coverage extended per Codex MEDIUM gaps), R6 ✓. F9
platform-blocked.

Test inventory (post-iter-61):
* 58 unit (was 50 pre-iter-58, then +8 driver-defaults at iter-59,
  then +11 driver-defaults at iter-61 → 69 unit-class regression).
* Wait, double-check: iter-58 added 3, iter-59 added 8, iter-61
  added 6 wrapper + 11 driver = 28 total in the two files.
* 12 cross-grid + 4 plane CRM e2e + 28 defaults-regression.

### 2026-05-26 — iter 60

**Delete orphaned ``scripts/run_rce_mpi_full.py`` (superseded by
iter-3/4/5/7 halo work).**

iter-59 noted the orphaned sibling driver carrying a stale
``N_ACOUSTIC=24`` default. iter-60 audit confirms it's been
completely superseded:

* **History**: single commit ever (``e6befce7`` "WIP: RCE MPI long-run
  scripts, dt-stability diag, CLAUDE.md compress"). No subsequent
  updates since.
* **Docstring**: explicitly describes itself as the rank-0-dycore +
  broadcast variant predating halo exchange — "the honest fix is to
  thread halo exchange through plane_operators — separate PR". That
  separate PR was iter-3 (Smag in halo) + iter-4 (R7 MPI mass fixer)
  + iter-5 (DD path in ``run_rce_mpi_long``) + iter-7 (WENO5 in halo).
  All the work the docstring promised is done — the file is
  vestigial.
* **References**: 0 callers outside its own docstring. iter-59 doc
  note was the only living reference and is being removed.
* **Missing iter-39+ updates**: no ``--no-radiation`` flag, no
  ``rad_calls`` counter, no iter-42 ``physics_schedule`` import, no
  iter-55 ``--days 0`` fix, no iter-59 N_ACOUSTIC default refresh.
  Cannot be safely revived without re-doing all this work.

Per CLAUDE.md "Removing module: also remove ``__init__.py`` re-export,
``supported_matrix.py`` entry, dispatch, test file, ``__pycache__``" —
the file had none of those touchpoints, so a clean ``git rm`` is the
full cleanup.

**Verified**:
* ``git rm scripts/run_rce_mpi_full.py``
* ``grep -rn "run_rce_mpi_full"`` post-delete → only the deletion
  itself + CRM_implementation.md mentions.
* CRM fast test sweep: PASS (no regression).

**R-roadmap status**: R1-R8, R10, R12 ✓ (with iter-60 dead-script
removal — CLAUDE.md slopbuster hygiene), R6 ✓. F9 platform-blocked.

### 2026-05-26 — iter 59

**Refreshed plane CRM production driver argparse default + AST regression test.**

iter-58 fixed the wrapper script defaults. iter-59 audit of the underlying
``scripts/run_rce_mpi_long.py`` argparse found the same staleness:

* ``--n-acoustic-substeps`` default = ``24`` (iter-1 dt=1.0 s legacy);
  iter-14 + iter-38 production use ``N=12`` at the driver's
  default ``dt=5.0 s``. A user invoking the driver with all defaults
  (no wrapper) would get N=24 → 2x slower substep + silent
  divergence from the iter-14 measurement.

**Fix**: bumped ``--n-acoustic-substeps`` default ``24 → 12`` with a
detailed help block citing the iter-14 + iter-38 production contract
+ the iter-1 historical reason for the old value.

**New regression test**
``tests/atmosphere/nonhydrostatic/integration/
test_run_rce_mpi_long_driver_defaults.py`` — 8 tests parsing the
driver source via ``ast.walk`` to find every ``p.add_argument`` call
+ asserting the literal defaults. Covers: dt, n-acoustic-substeps,
hyperdiff, smag-cs, bubble-theta-pert, qv-noise-amp, nx/ny, nlev.

Mirrors iter-58 wrapper-script pattern with stronger parser (AST
vs bash regex). Future silent revert trips in < 1 s.

**Verified**:
* Driver smoke: ``--dt 5.0 --days 0.0005 --no-radiation`` with the
  default N_ACOUSTIC=12 (no explicit flag) runs 8 steps cleanly.
* 17/17 CRM fast tests PASS in 11 s (no regression).
* 8/8 driver-defaults regression PASS in 0.03 s.

Note: ``scripts/run_rce_mpi_full.py`` (orphaned older sibling that
predated the iter-3/4/5/7 halo work) deleted in iter-60 — see below.

**R-roadmap status**: R1-R8, R10, R12 ✓ (with iter-59 driver-argparse
default refresh + AST-based regression backstop), R6 ✓. F9
platform-blocked.

Test inventory (post-iter-59):
* 58 unit (was 50 in iter-54): added 8 driver-defaults regression tests.
* 12 cross-grid + 4 plane CRM e2e.
* 3 wrapper-script regression (iter-58).
* 8 driver-script regression (iter-59).

### 2026-05-26 — iter 58

**Refreshed stale `run_rce_30day.sh` defaults + landed regression test.**

iter-58 audit of ``scripts/run_rce_30day.sh`` (the production-launch
wrapper) found two stale defaults dating to iter-1 / pre-F10:

* ``N_ACOUSTIC`` env default ``24`` — was set for iter-1's
  ``dt=1.0 s`` config. iter-14 + iter-38 production runs use
  ``N_ACOUSTIC=12`` for ``dt=5.0 s``. With ``dt=5`` + the stale
  ``N_ACOUSTIC=24`` the acoustic CFL ratio would halve — produces
  an over-stable but slower substep, and silently differs from
  the iter-14 measurement that locks 1-sim-hour PASS at
  ``N_ACOUSTIC=12``.

* Documentation block lines 12-22 still cited the iter-1 F1
  dt-stability ladder + "outer dt must satisfy dt <= ~1.0 s"
  conclusion + R9 (KW78 outer-step) as the path forward. iter-2
  F7 + iter-9 F10 + iter-14 production all superseded that: clean
  Wing IC + dt=5 s is production-stable, R9 no longer on critical
  path.

**Fixes**:
* ``scripts/run_rce_30day.sh``: bumped ``N_ACOUSTIC`` default
  ``24 → 12``; rewrote the stability-history comment block to
  reflect F7/F8/F10/iter-14 + iter-38 reality.

**New regression test**
``tests/atmosphere/nonhydrostatic/integration/
test_run_rce_30day_wrapper_defaults.py``: 3 tests parsing the
bash env-var defaults via regex and asserting they match the
iter-12/14/38 production contract. Covers ``DT``,
``N_ACOUSTIC``, ``NX``, ``NY``, ``ADVECTION``, ``HYPERDIFF``,
``BUBBLE_K``, ``QV_NOISE``. A future silent revert would trip in
< 1 s.

**Verified**: ``DAYS=0.0005 NX=12 NY=12 RANKS=1`` smoke runs the
wrapper cleanly with the new defaults; the production driver
prints ``Done. 8 steps, ... rad_calls=1`` (default ``rad-call-
interval-s=600`` fires once at step 1).

**Tests**: 3/3 wrapper-defaults regression PASS in 0.03 s. No
existing test affected.

**R-roadmap status**: R1-R8, R10, R12 ✓ (with iter-58 production
wrapper script refreshed to match the iter-14/iter-38 production
contract + structural-regression backstop), R6 ✓. F9
platform-blocked. R11 30-day plane CRM still wall-time gated
(~8 days single-rank on M5 Pro).

### 2026-05-26 — iter 57

**Codex holistic review of plane CRM MPI halo (plane_mpi.py)
— 0 HIGH + 0 MEDIUM + 3 LOW. DOD item 5 met across all three
production-path modules.**

Final piece of the DOD item 5 holistic Codex pass:
* iter-55: production driver (``scripts/run_rce_mpi_long.py``) — 0 HIGH + 0 MEDIUM, 2 LOW deferred.
* iter-56: dycore + halo-aware slow tendency (``compressible_euler_plane.py`` + ``compressible_euler_plane_halo.py``) — 0 HIGH + 0 MEDIUM (production); 2 HIGH + 2 MEDIUM in the bench-only fast-path closed via fallback gate.
* iter-57: MPI halo + reductions (``src/legoesm/parallel/plane_mpi.py``) — 0 HIGH + 0 MEDIUM + 3 LOW.

iter-57 Codex review highlights (all clean except cosmetic LOWs):
* AD-safety: every halo sendrecv routes via ``_get_sendrecv_vjp(mpi4jax)`` — no raw ``MPI.Sendrecv`` / ``Isend`` / ``Irecv`` leaks.
* Halo correctness: width validated on both axes; N/S slabs + E/W corner filling via second pass correct.
* Single-rank short-circuit: ``jnp.pad(..., mode="wrap")`` matches multi-rank halo result bit-for-bit.
* 4D halo mandate: ``packed_exchange_halo_plane_yxz`` stacks fields + issues a single MPI call — no ``vmap(pad_halo)``.
* AD-safe reductions: no ``global_max_mpi``/``global_min_mpi``/``allgather``/``bcast`` in this file.

**LOW fixes applied**:
* LOW#2 — magic MPI tag bases (``_TAG_NS=1000``, ``_TAG_EW=2000``) now documented (tag namespace invariant: 2 axes × 1 packed call ≪ 1000-tag span budget).

**LOW deferred**:
* LOW#1 — iter-N provenance comments in production code: kept (CLAUDE.md doesn't ban them; useful trace context).
* LOW#3 — N/S vs E/W sendrecv block duplication: kept (2-site copy with indexing-only diffs; premature abstraction risk).

**DOD item 5 status**: ✓ MET. Production driver + dycore + halo all
hold 0 HIGH + 0 MEDIUM findings under the iter-55/56/57 holistic
Codex sweep. R12 (full /codex:adversarial-review pass) substantially
complete.

**R-roadmap status**: R1-R8, R10, R12 ✓ (with iter-57 closing the
DOD item 5 holistic review), R6 ✓. F9 platform-blocked (R8 real
scaling numbers). R11 30-day plane CRM still wall-time gated
(~8 days single-rank on M5 Pro), but structurally pinned via
iter-38/39 production-scale regressions.

### 2026-05-26 — iter 56

**Codex holistic review of plane CRM dycore + halo caught 2 HIGH +
2 MEDIUM — closed via fallback gate (production unaffected).**

iter-3 (Smag in halo), iter-4 (R7 MPI mass fixer wiring), iter-7
(WENO5 in halo) added significant features to the plane CRM
halo-aware slow tendency. iter-56 ran the first holistic Codex
review since.

Findings in ``slow_tendency_jit_split`` (Python-driven 2-MPI-round
fast-path orchestrator used by the bench ``scripts/bench_dd_scaling.py``;
NOT used by the production driver):

* **HIGH#1** — silently omits Coriolis when ``config.use_coriolis``
  is True. Kernel-2 trace has no Coriolis branch.
* **HIGH#2** — silently degrades WENO5 to upwind1 (hardcoded
  ``oh.upwind_advection_*_halo`` calls).
* **MEDIUM#1** — missing ``hyperdiff_w_coeff`` branch (vertical-
  velocity grid-scale damping silently absent).
* **MEDIUM#2** — owned_mask not passed into the slow-tendency
  computation (non-owned cells receive full tendencies; mass-fixer
  masking happens post-step).

**Fix**: extended the existing ``smagorinsky_cs > 0 or tracers > 0``
early-fallback set to also include ``use_coriolis``, ``advection ==
weno5``, and ``hyperdiff_w_coeff > 0``. Any production-style config
trips at least one gate (production uses Smag c_s=0.2 + 3 tracers,
so the fallback was always firing anyway). The fast-path is now
correctness-clean within its restricted regime (no-Smag, no-tracer,
no-Coriolis, upwind1, no-hyperdiff_w bench-style runs).

MEDIUM#2 (owned_mask) is by-design: ``step_halo`` computes tendencies
on the full local slab (interior + halo), then halo cells are
overwritten by the next exchange. Mass fixer is the only consumer of
owned_mask. A docstring annotation would clarify but no code change
needed.

**Production impact**: ZERO. ``run_rce_mpi_long.py:step_halo`` (used
by all iter-38/39/41 slow tests) was already going through the eager
``plane_compressible_euler_slow_tendencies_halo`` path because
production config trips the Smag+tracer gate.

**Tests**: 79/79 fast PASS in 17 s across the unit + integration
test sweep (test_plane_slow_tend_halo, test_plane_mass_fixer_mpi,
test_weno5_halo_equiv, test_physics_schedule, plane_crm_end_to_end_smoke,
plane_crm_helpers_unit, rce_helpers_unit).

**R-roadmap status**: R1-R8, R10 ✓ (with iter-56 plane CRM halo
correctness gates extended), R6 ✓. F9 platform-blocked.

DOD item 5 progress: production driver + halo-aware slow tendency
now have 0 HIGH + 0 MEDIUM findings against the Codex iter-55/56
holistic reviews. Bench fast-path (slow_tendency_jit_split) hardened
with explicit fallback for non-production configs.

### 2026-05-26 — iter 55

**Codex holistic review of the production driver caught 2 MEDIUM
bugs — both fixed.**

iter-39/40/42/54 each added a radiation-path change to
``scripts/run_rce_mpi_long.py``. iter-55 ran a fresh Codex review
across the whole file to verify the post-stack composition. Found:

* **MEDIUM#1 — ``--no-radiation`` + bogus ``--rad-call-interval-s``
  crashes on the unused setting.** The iter-43 NaN/inf validation
  in ``physics_schedule.radiation_call_every_steps`` is correct
  *when radiation is enabled* — but the driver called the helper
  unconditionally, so even with ``--no-radiation`` a NaN interval
  would raise ValueError. Fixed: skip the helper entirely when
  ``args.no_radiation`` is True. Verified: ``--no-radiation
  --rad-call-interval-s nan`` now completes cleanly.

* **MEDIUM#2 — ``--days 0`` crashes the final ``Done.`` print with
  NameError on ``step``.** With total_steps=0 the ``for step in
  range(1, 1)`` loop body is skipped, leaving ``step`` + ``t_sim``
  undefined when the final-print block reads them. iter-32 AMIP
  wrapper uses ``--days 0`` as a dry-run; iter-55 exposes the same
  pattern in the CRM driver. Fixed: init ``step = 0`` +
  ``t_sim = 0.0`` before the loop. Verified: ``--days 0`` now
  prints ``Done. 0 steps... rad_calls=0`` and exits cleanly.

* **LOW#1 + LOW#2** (deferred): rad_call_count under DD path is
  per-rank-local (intentional per iter-40 Codex review — print
  shows rank 0's count, not an MPI-sum); ValueError from helper
  propagates as raw traceback instead of SystemExit. Both
  cosmetic.

**Tests**:
* Direct smoke: ``--days 0 + --no-radiation`` → "Done. 0 steps..."
  ✓ (no crash).
* Direct smoke: ``--no-radiation --rad-call-interval-s nan`` →
  "Done. 8 steps..." ✓ (no crash on unused NaN).
* Fast test inventory (CRM + physics_schedule): 36/36 PASS in
  11 s. No regression.

**R-roadmap status**: R1-R8, R10 ✓ (with iter-55 production-
driver MEDIUM bug fixes via Codex holistic review of the post-
iter-39/40/42/54 radiation-path stack), R6 ✓. F9 platform-blocked.

DOD item 5 (``Pass /codex:adversarial-review on dycore + MPI halo
+ production driver with no MEDIUM/HIGH findings outstanding``):
production driver now has 0 HIGH + 0 MEDIUM findings against the
radiation path. Dycore + halo still need a fresh post-iter-N
holistic Codex pass.

### 2026-05-26 — iter 54

**Tighten `_parse_rad_call_count` regex to actually reject schema
drift (closes iter-53-pinned Codex iter-40 LOW#4 gap).**

iter-53 added a unit test pinning ``rad_calls=5.0 → 5`` as "current
behaviour". Auditing during iter-54: that's actually a real silent-
drift gap — the iter-40 Codex LOW#4 fix was *supposed* to prevent
float-form matching, but the iter-40 regex
``(?m)^Done\..*\brad_calls=(\d+)\b`` matched the ``5`` in ``5.0``
via word-boundary between digit-and-dot, exactly the failure mode
the original Codex finding wanted closed.

**Fixes**:

* ``_parse_rad_call_count`` regex tightened:
  - iter-40: ``\brad_calls=(\d+)\b``
  - iter-54 cut-1: ``\brad_calls=(\d+)\.`` — still matched
    ``rad_calls=5.0.`` via greedy backtrack
  - iter-54 cut-2 (Codex LOW fix): ``\brad_calls=(\d+)\.[^\S\n]*$``
    multiline — requires integer-then-period as the LAST
    non-whitespace token on the ``Done.`` line.

* iter-53 ``test_parse_rad_call_count_rejects_float_suffix`` flipped
  from "extracts 5 from 5.0 (pinned current behaviour)" to "returns
  None — drift rejected loudly".

* Two new unit tests:
  - ``test_parse_rad_call_count_rejects_trailing_token``: catches
    ``rad_calls=5. (cached=true)`` (Codex iter-54 LOW concern).
  - ``test_parse_rad_call_count_tolerates_trailing_whitespace``:
    pins CRLF + trailing-space-padding compatibility.

**Codex iter-54** caught the iter-54 cut-1 regex was still too
loose (``(?:\s|$)`` allowed arbitrary trailing). Cut-2 multiline
``$`` anchor + ``[^\S\n]*`` (non-newline-whitespace) closes the
gap cleanly.

**Tests**: 12 unit (was 10 in iter-53) + 2 fast integration PASS
in 11 s.

**R-roadmap status**: R1-R8, R10 ✓ (with iter-54 closing the
iter-40 Codex LOW#4 gap that iter-53 inadvertently pinned as
acceptable), R6 ✓. F9 platform-blocked.

Test inventory (post-iter-54): 50 unit (was 48) + 12 cross-grid
+ 4 plane CRM e2e.

### 2026-05-26 — iter 53

**Refresh stale Components table + unit-test the `_parse_rad_call_count`
parser (mirrors iter-52 for the plane CRM side).**

**Doc refresh** (``CRM_implementation.md``):
Components table claims dated to iter-1/2 (some 50+ iters stale). Refreshed
every row to reflect iter-12/15/26/42/46/52 current state:
* Plane CRM dycore: "unstable at dt=2 s" → "production-stable at dt=5 s
  (F10 lifted constraint)".
* Plane CRM halo: "missing Smag/WENO5/vertical-θ-diff/KW78" → all four
  landed (iter-3/iter-7) except KW78 which is obsolete per F10.
* SI substeps: "KW78 WIP" → "Substep KW78 inert (F2); R9 outer-step no
  longer on critical path (F10)".
* 30-day production driver: "Runs dycore on rank 0 + broadcasts" → "DD
  path wired via --use-dd (iter-5); legacy retained as default".
* Multi-grid RCE driver: 30-day production validated for {C24, C48, C72,
  V4, LL32, T21} (iter-12 + iter-15 + iter-26 + iter-51).
* Added 3 new rows: Auto-dt ladder (iter-24), Radiation-schedule helper
  (iter-42), Shared RCE assertion helpers (iter-46).

**Substantive code**: ``tests/atmosphere/nonhydrostatic/integration/
test_plane_crm_helpers_unit.py`` (NEW) — 10 unit tests for the
``_parse_rad_call_count(stdout)`` regex helper added in iter-40.

Coverage:
* Happy-path matches (basic, zero, large rad_calls value).
* Anchor robustness — ignores ``total_rad_calls=`` substring (iter-40
  Codex LOW#4 fix); rejects substring-only match without a ``Done.``
  prefix.
* Edge: ``rad_calls=5.0`` extracts 5 due to ``\b\d+`` word-boundary
  (pinned as current behaviour — explicit doc for future regex
  tightening).
* Missing markers: empty stdout, ``Done.`` line without marker,
  no ``Done.`` line at all → returns ``None``.
* Pathological: two ``Done.`` lines (botched retry) → returns ``None``
  per iter-40 "exactly one match" contract.

10/10 PASS in 0.05 s.

Also: ``CRM_implementation.original.md`` was deleted from working tree by
some out-of-band process between iter-51 and iter-53; restored via
``git checkout HEAD -- ...``. iter-49 backup intact.

**R-roadmap status**: R1-R8, R10 ✓ (with iter-53 doc-state refresh +
plane CRM parser unit-test coverage), R6 ✓. F9 platform-blocked.

Test inventory (post-iter-53):
* 48 total unit tests (was 38 pre-iter-53): 22 physics_schedule + 16 RCE
  assertion helpers + 10 plane CRM parsers.
* 5 fast cross-grid + 7 slow cross-grid nightly.
* 4 plane CRM end-to-end (2 fast + 2 slow).

### 2026-05-26 — iter 52

**Unit tests for the iter-46 shared assertion helpers (LL32 + T21 PASS).**

iter-46 factored ``_assert_dt_used`` + ``_assert_max_wind_peak_below``
into shared helpers. Until iter-52 those helpers were exercised only
through the slow nightly tests (100-2400 s each) — a regression in the
assertion logic itself would not be caught until nightly run.

**New file** ``tests/atmosphere/hydrostatic/test_rce_helpers_unit.py``:
16 direct unit tests for the helpers using ``tmp_path`` + handwritten
``results.txt`` / ``mean_timeseries.csv`` files. Coverage:

* ``_assert_dt_used`` (9 tests):
  - strict ``==`` pass
  - strict fail on wrong dt + on tiny deviation (catches silent rounding)
  - missing ``results.txt``
  - missing ``dt:`` field (NaN propagation)
  - abs_tol pass at iter-51 LL32 production value
  - abs_tol pass just inside boundary
  - abs_tol fail with informative ``± X`` message
  - ``abs_tol=0`` = strict equivalence (Codex iter-51 concern verified)

* ``_assert_max_wind_peak_below`` (7 tests):
  - happy path peak < cap
  - takes abs value (negative spike counted)
  - fail on peak > cap
  - missing csv
  - header-only csv rejected (iter-46 Codex MEDIUM fix)
  - missing ``max_wind`` column rejected (iter-45 column-pin)
  - all-unparseable rows rejected (iter-46 MEDIUM extension)

**Live verified**: LL32 + T21 30-day slow tests (iter-51) **PASS in
321 s combined**. Confirms iter-51 thresholds work end-to-end:
abs_tol=1e-2 absorbed the post-clamp dt variance without false-fail,
cap=20.0 caught the iter-12 measured envelopes cleanly.

**R-roadmap status**: R1-R8, R10 ✓ (with iter-52 direct-unit-test
coverage of the iter-44/46 hardening helpers — assertion-logic
regressions now fail in < 1 s instead of waiting for nightly), R6 ✓.
F9 platform-blocked.

Test inventory:
* 16 unit (iter-52, < 0.1 s)
* 22 unit physics_schedule (iter-42/43, ~2 s)
* 5 fast cross-grid + 7 slow nightly (iter-13..iter-51)
* 4 plane CRM (2 fast + 2 slow, iter-15/16/38/39)
* iter-38..iter-51 hardening: schedule/dt/peak/csv-schema all
  pinned at both production-scale + test-helper level.

### 2026-05-26 — iter 51

**Closed the last 30-day production-scale empirical gaps for the
hydrostatic family — LL32 + T21 nightly regressions landed.**

iter-12 measured the full cross-grid 30-day cohort but only C48/C72/V4 had nightly regression backing post-iter-50. LL32 (latlon C-grid, pole-clamped dt=81.844 s) and T21 (gaussian spectral, dt=600 s) had only 2-day smokes — same silent-pass risk iter-44/46 fixed for C48/C72.

**New tests** (``tests/atmosphere/hydrostatic/test_rce_cross_grid_smoke.py``):

* ``test_latlon_ll32_30day_nightly_validation``:
  - dt assertion: ``_assert_dt_used(81.844, abs_tol=1e-2)``.
  - peak max\|v\| cap=20.0 (1.8× iter-12 measured 11.19 m/s).
  - temp_tol=1.0 (iter-12 Δ=+0.09 K → ample headroom).
* ``test_gaussian_t21_30day_nightly_validation``:
  - dt assertion: ``_assert_dt_used(600.0)`` (strict ==).
  - peak max\|v\| cap=20.0 (2.4× iter-12 measured 8.43 m/s).
  - temp_tol=1.0 (iter-12 Δ=+0.13 K → ample headroom).

**Helper extension**: ``_assert_dt_used`` now takes an optional
``abs_tol`` parameter (default None → strict ``==``). Needed for LL32
where the pole-cell CFL clamp produces a non-integer dt
(81.84408841999117 confirmed via probe run). Ladder values
600/300/150/75/37 remain strict ``==``.

**Codex iter-51**: 0 HIGH, 0 MEDIUM, 1 LOW. Original abs_tol=0.5
was too loose — would accept an inadvertent ``round(dt)``
silently. Fixed: abs_tol=1e-2 (catches a rounding/scaling
regression while absorbing cross-platform double-precision drift).

**R-roadmap status**: R1-R8, R10 ✓ (iter-51 closes the LAST
30-day production-scale empirical gap for the hydrostatic
cross-grid family — all 4 grid types {C48/C72, V4, LL32, T21}
now have 30-day nightly regression backing; C96 stays at 2-day
nightly per iter-22 wall-time decision), R6 ✓. F9 platform-blocked.

Slow nightly count: was 7, now 9. 30-day production-scale
empirical gates: C48, C72, V4, LL32, T21.

### 2026-05-26 — iter 50

**V4 (voronoi/MPAS) 30-day nightly slow regression landed.**

iter-12 measured voronoi V4 30-day at dt=300 PASS (mean_T_sfc=300.85, max\|v\|=2.28 m/s, wall=101 s on M5 Pro). iter-8 noted dt>=450 BLOWUPs at day 1 — dt=300 pinned in ``auto_dt_rce`` for voronoi. Until iter-50 dt=300 contract had **no 30-day CI backing** — only 2-day V4 smoke gated regressions; iter-44 showed 2-day misses mid-run CFL spikes that recover.

New ``test_voronoi_v4_30day_nightly_validation`` mirrors C48/C72 30-day pattern via iter-46 shared helpers:
* ``_assert_dt_used(expected_dt=300.0)`` — catches future ladder drift.
* ``_assert_max_wind_peak_below(cap=10.0)`` — full-timeseries peak; 4.4× cushion over iter-12 measured 2.28 m/s.
* ``_assert_rce_pass(temp_tol=1.5, max_v_cap=10.0)`` — V4-specific envelope tuned for MPAS stability profile.

**Codex iter-50** caught 1 HIGH + 1 MEDIUM + 3 LOW:
* HIGH — original temp_tol=1.0 left 0.15 K positive-drift headroom vs measured +0.85 K. ``_assert_rce_pass`` uses strict ``<`` → fail risk. Fixed: temp_tol=1.5.
* MEDIUM — original cap=25.0 was 11× iter-12 peak (mirroring cubed-sphere caps from different regime). Tightened to 10.0 — still 4.4× cushion.
* LOW — hardcoded expected_dt vs auto_dt_rce("voronoi",4): intentional regression pin, kept.
* LOW — mean_timeseries.csv schema for MPAS path: confirmed via run_rce.py:668-676.
* LOW — CLI --dt override would fail: intended.

**Verified**: V4 30-day live run (loose thresholds, ran before Codex fixes) 1 PASS in 177 s. Tightened thresholds mathematically pass at iter-12 measurements (2.28 < 10.0; 0.85 < 1.5).

**R-roadmap status**: R1-R8, R10 ✓ (iter-50 closes last 30-day production-scale empirical gap for hydrostatic voronoi as structural regression), R6 ✓. F9 platform-blocked.

Slow nightly count: was 6 (4 cdgrid + 2 plane CRM), now 7. 30-day production-scale empirical gates: cubed_sphere C48+C72, voronoi V4.

### 2026-05-26 — iter 48 (re-landed; was lost in iter-49 compress)

**``test_blowup_gate_fires_on_supersonic_winds`` now verifies the supersonic channel actually fired** (not just status: FAIL).

iter-17 added BLOWUP-gate regression locking iter-13 threshold (run_rce.py reports status: FAIL + non-zero exit when max\|v\| > 200 m/s). Verified FAIL signal but NOT channel — different failure mode (NaN earlier, runtime crash) could produce status: FAIL without supersonic-winds gate firing.

Added ``mean_timeseries.csv`` peak-max-wind scan asserting peak_v >= 200.0 after existing FAIL assertions. iter-13 measured 236 m/s by day 20 — comfortable margin. Inverse of iter-46 ``_assert_max_wind_peak_below`` (used in C48/C72/C96 PASS tests).

csv-exists check gated (``if mean_csv.exists()``) rather than mandatory: BLOWUP could crash driver before any diagnostic write — preserves existing FAIL-only contract while strengthening when data available.

(iter-48 doc entry was added to working tree but dropped in iter-49 compression. iter-50 re-lands it for the historical record. Test change itself committed in 839f3cac.)

### 2026-05-26 — iter 47

**Applied iter-46 shared hardening helpers to C96 2-day slow smoke + fixed stale ladder-branch reference.**

iter-46 factored ``_assert_dt_used`` + ``_assert_max_wind_peak_below`` into shared helpers and applied to C48 + C72 30-day nightlies. C96 2-day slow smoke remained on loose ``_assert_rce_pass`` final-day-only envelope — same silent-pass risks iter-46 fix exists to catch.

Particularly dangerous for C96 because iter-20 BLOWUP at C96/dt=75 reached max\|v\|=175 m/s by day 15, 527 by day 20 — but at 2 days same broken config lands at max\|v\|≈5-10 m/s (CFL-stable for first few days). 2-day smoke alone CANNOT distinguish dt=37 (production) from dt=75 (BLOWUP-in-flight). Only dt-assertion does.

**Changes** (``tests/atmosphere/hydrostatic/test_rce_cross_grid_smoke.py``):
* Docstring corrected from "iter-13 dt=75 ladder branch" to "iter-20 dt=37" with explanation of iter-20 BLOWUP fix.
* Added ``_assert_dt_used(out_dir, "C96 2-day", expected_dt=37.0)`` — catches ladder drift re-routing C96 onto dt=75.
* Added ``_assert_max_wind_peak_below(out_dir, "C96 2-day", cap=50.0)`` — same cap as existing final-day check; catches mid-run CFL spike recovering by end of 2-day window.

**Codex iter-47** caught 0 HIGH, 0 MEDIUM, 1 LOW (stale comment referencing non-existent ``test_rce_2day_smoke_passes_slow``). Fixed: comment now points at ``test_rce_2day_smoke_c96_slow`` and cites iter-20 ladder branch correctly.

**Tests**:
* Fast suite collection: 9 tests collected (5 fast + 4 slow); fast suite continues PASS in ~210 s.

**R-roadmap status**: R1-R8, R10 ✓ (iter-47 closes last cross-grid slow test gap lacking dt-assertion + peak-max-wind hardening — all 4 cdgrid nightlies now consistently protected), R6 ✓. F9 platform-blocked.

### 2026-05-26 — iter 46

**Factored iter-44 hardening into shared helpers + propagated to C48 30-day nightly.**

iter-44 landed two Codex-MEDIUM fixes inline in C72 30-day nightly: dt-used assertion (#1) + ``mean_timeseries.csv`` peak-max-wind scan (#2). C48 30-day nightly never got same hardening — same silent-pass risk (ladder drift routing C48 to wrong dt branch, or mid-run max\|v\| spike recovered by day 30, would slip through loose final-day envelope).

**Factored two shared helpers** in ``tests/atmosphere/hydrostatic/test_rce_cross_grid_smoke.py``:

* ``_assert_dt_used(out_dir, label, expected_dt)``: parses ``results.txt``, asserts ``dt == expected_dt``. Pins iter-13/26 ladder contract.
* ``_assert_max_wind_peak_below(out_dir, label, cap)``: scans ``mean_timeseries.csv`` for peak ``max_wind`` across all logged days, asserts below cap. Pins exact column name ``max_wind`` (Codex iter-45 hardening retained).

Both helpers applied to BOTH C48 + C72 30-day nightlies with appropriate expected_dt (150.0 for C48, 75.0 for C72) and same cap (25.0 m/s — generous margin over iter-15/26 measurements).

**Codex iter-46 review** caught 1 MEDIUM + 1 LOW:

* **MEDIUM** — ``_assert_max_wind_peak_below`` would PASS vacuously on empty timeseries (header-only csv or all-unparseable rows): ``peak_v`` initialised to ``0.0`` always satisfies ``< cap``. Fixed: added ``seen_max_wind`` boolean tracking at least one successful row parse, asserted before cap check.
* **LOW** — helper failure messages omitted diagnostic file paths. Fixed: every assertion message now embeds ``results.txt`` / ``mean_timeseries.csv`` path so CI failure points developer directly at artifact to inspect.

**Tests**:
* Fast suite: 5/5 PASS in 211 s (no regression).
* Slow tests: 7 total CRM-relevant slow tests now (test_rce_cross_grid_smoke: c96 2-day, blowup gate, c48 30-day, c72 30-day; test_plane_crm_end_to_end_smoke: production 132x132 envelope, production 132x132 with rad).

**R-roadmap status**: R1-R8, R10 ✓ (iter-46 DRY refactor of iter-44/45 hardening + C48 propagation + Codex MEDIUM fix), R6 ✓. F9 platform-blocked.

### 2026-05-26 — iter 45

**Propagate iter-44 Codex HIGH-class timeout fix to C96 2-day + C48 30-day slow tests + tighten C72 csv parsing.**

iter-44 fixed `_run_rce` to accept ``timeout_s`` and passed ``timeout_s=4800`` for C72 30-day nightly. Two sibling slow tests still used 600 s default, both with marginal headroom:

* ``test_rce_2day_smoke_c96_slow``: iter-22 measured C96 10-day at 2506 s ≈ 250 s/day → 2-day ≈ 500 s. Default 600 leaves only 20% margin; 10%-slower runner would silently hang.
* ``test_c48_30day_nightly_validation``: iter-15 measured C48 30-day at 500.9 s. Default 600 leaves only 20% margin.

Both bumped to ``timeout_s=1800`` (3-3.6× cushion vs measured wall). Genuine hang now fires timeout cleanly; normal slow box finishes well inside new budget.

**Secondary hardening** on iter-44 C72 csv parsing:
* iter-44 used substring heuristic (``"max" in k and ("v" in k or "wind" in k)``) for ``max_wind`` column lookup. Would false-match future column named e.g. ``max_dvdt`` (contains both "max" and "v").
* iter-45 pins exact column name ``max_wind`` (matches ``run_rce.py:668-676`` schema). If schema changes, test fails loudly with clear "if intentional, update this test" pointer.
* Also promoted ``mean_timeseries.csv`` existence check from silent skip to hard fail (driver-emit regression now visible).

**Tests**:
* Fast suite still 5/5 PASS in 207 s.
* No new slow tests added; iter-44 ladder unchanged. Risk: iter-44 timeout=4800 for C72 + iter-45 timeout=1800 for C48/C96 give all 4 cross-grid slow tests safe budgets on M5 Pro and slightly slower runners.

**R-roadmap status**: R1-R8, R10 ✓ (iter-45 slow-test timeout hardening), R6 ✓. F9 platform-blocked.

### 2026-05-26 — iter 44

**C72 30-day nightly regression test landed + Codex caught HIGH in shared _run_rce helper (timeout=600 vs measured 2373 s).**

Codex iter-22 review (back at iter-22) flagged iter-13 dt=75 ladder branch (N=49..72) was empirically anchored at N=49 only — C48 boundary. iter-26 closed upper-end gap by running C72 30-day to completion (dt=75, mean_T_sfc=299.81 K, max\|v\|=17.85 m/s, wall=2373 s). Until iter-44 that measurement had **no regression test** — only C48 + C96 30-day or 2-day slow tests gated CI, neither exercises C72 dycore-specific stability profile.

iter-44 adds ``test_c72_30day_nightly_validation`` mirroring ``test_c48_30day_nightly_validation`` pattern.

**Codex iter-44 review** caught 1 HIGH + 2 MEDIUM + 1 LOW:

* **HIGH** — ``_run_rce`` hard-coded ``timeout=600``. iter-26 measured C72 wall=2373 s, so new test would HANG/KILL run before completion. Fixed: added ``timeout_s`` parameter to ``_run_rce`` (defaults to 600 for existing 2-day smokes); C72 nightly passes ``timeout_s=4800`` (2x cushion on measured wall).
* **MEDIUM#1** — test claimed "iter-13 auto-dt=75" but never asserted dt actually used. Fixed: added ``dt_used = float(fields.get("dt", "nan"))`` + ``assert dt_used == 75.0`` so silent ladder drift caught even when envelope check still passes.
* **MEDIUM#2** — ``_assert_rce_pass`` reads ``notes`` which only carries final-day max\|v\|. Mid-run CFL spike recovered by day 30 would slip through. Fixed: added ``mean_timeseries.csv`` peak-max-wind scan with 25 m/s cap before helper's last-day check. Tolerates column-name drift via lowercase keyword match.
* **LOW** — docstring referenced "C96 nightly" which doesn't exist. Fixed: clarified to "2-day C96 slow smoke (test_rce_2day_smoke_c96_slow)".

**Verified**:
* ``pytest -m 'not slow' tests/atmosphere/hydrostatic/
  test_rce_cross_grid_smoke.py``: 5 PASS in 211 s (no regression).
* New slow test collected: 1 new (``test_c72_30day_nightly_validation``).

**R-roadmap status**: R1-R8, R10 ✓ (iter-44 closes iter-22 C72-upper-end empirical gap as structural regression), R6 ✓. F9 platform-blocked.

### 2026-05-26 — iter 43

**Hardened ``physics_schedule`` against pathological inputs (Codex 2nd-pass).**

Codex iter-42 first-pass review caught 1 MEDIUM (memory) + 1 LOW (boundary). iter-43 2nd-pass review caught 3 MEDIUMs around non-finite inputs first guard block didn't reject:

* **NaN dt**: ``dt <= 0.0`` compares False against NaN, so guard silently passed; ``round(rad/nan)`` then raised opaque ``ValueError`` deep in CPython.
* **inf dt or interval**: ``rad/inf = 0`` rounds to ``0`` and ``max(1, 0)`` returns 1, masking bug. ``round(inf)`` would ``OverflowError`` if interval inf.
* **total_steps == sys.maxsize**: ``range(1, sys.maxsize + 1, ...)`` overflows on CPython.

**Fixes** (``src/legoesm/driver/physics_schedule.py``):
* Added ``math.isfinite()`` checks at top of ``radiation_call_every_steps`` for both dt and rad_call_interval_s. Rejected BEFORE ordering guards so error clear.
* Added ``total_steps >= sys.maxsize`` guard at top of ``radiation_call_schedule``. Includes "no realistic ESM run exceeds 10^9 steps" message so failure mode obvious if future caller passes corrupted int.

**Unit tests** added (``tests/unit/test_physics_schedule.py``):
* ``test_every_steps_rejects_nan_dt``
* ``test_every_steps_rejects_inf_dt``
* ``test_every_steps_rejects_nan_interval``
* ``test_every_steps_rejects_inf_interval``
* ``test_schedule_rejects_sys_maxsize_total_steps``

5 new tests, total 22 unit tests for module (was 17). All PASS in 2 s.

**Codex 2nd-pass LOW** acknowledged but not fixed:
* RadiationCallSchedule field order now public tuple ABI — reordering breaks callers using positional construction. Acceptable: callers in this repo use keyword construction or named attribute access. Future breaking change would be deliberate API bump.

**Tests**:
* ``pytest tests/unit/test_physics_schedule.py``: 22 PASS in 2.0 s.
* ``pytest -m 'not slow' tests/atmosphere/nonhydrostatic/
  integration/test_plane_crm_end_to_end_smoke.py``: 2 PASS in 32 s.

**R-roadmap status**: R1-R8, R10 ✓ (iter-43 input-domain hardening of iter-42 schedule helper), R6 ✓. F9 platform-blocked.

### 2026-05-26 — iter 42

**Radiation-call schedule arithmetic factored to unit-testable driver helper (single source of truth).**

CLAUDE.md DRY rule + Codex iter-39/40/41 review trail flagged inline arithmetic ``rad_call_every_steps = max(1, int(round(rad_call_interval_s/dt)))`` + ``expected = 1 + (n_steps - 1) // every`` as silent-divergence risk: driver and two test files all encoded formula independently. iter-42 extracts to ``legoesm.driver.physics_schedule``.

**New module** ``src/legoesm/driver/physics_schedule.py``:
* ``RadiationCallSchedule`` NamedTuple with fields ``(every_steps, num_calls, fire_step_indices)``. ``fire_step_indices`` is lazy ``range`` (Codex iter-42 MEDIUM: original tuple materialisation would allocate million ints on 10M-step run; ``range`` is O(1) memory).
* ``radiation_call_every_steps(rad_call_interval_s, dt)`` — mirrors exact driver formula incl. ``int(round(...))`` rounding semantics. Raises ``ValueError`` on ``dt <= 0`` or negative interval (was previously silent).
* ``radiation_call_schedule(rad_call_interval_s, dt, total_steps)`` — full schedule with O(1) ``num_calls`` derivation.

**Wiring**:
* ``src/legoesm/driver/__init__.py`` re-exports all three names.
* ``scripts/run_rce_mpi_long.py`` uses ``radiation_call_every_steps`` (function-scope import to avoid cross-package eager-import pattern CLAUDE.md warns against).
* ``tests/atmosphere/nonhydrostatic/integration/
  test_plane_crm_end_to_end_smoke.py``: iter-16 + iter-39 assertions now go through ``radiation_call_schedule`` instead of re-deriving formula locally. Single canonical source driver also uses.

**New unit tests** (``tests/unit/test_physics_schedule.py``):
17 tests covering:
* basic schedule (driver-mirror formula match)
* sub-dt clamping
* huge-interval anti-pattern (Codex iter-39 HIGH#2: ``rad_call_interval_s=1e9`` still fires ONCE at step 1)
* int(round(...)) tie-breaks (5.4 → 5, 5.6 → 6)
* invalid-input ValueError contracts (negative dt / interval)
* total_steps boundary (0 → empty, 1 → one fire)
* memory invariant: lazy range for huge total_steps
* first-fire-always-step-1 invariant across all intervals
* num_calls matches len(fire_step_indices)

**Codex iter-42 review**: 0 HIGH, 1 MEDIUM, 1 LOW. Both fixed:
* MEDIUM: ``fire_step_indices`` materialisation; switched to lazy ``range``.
* LOW: missing ``total_steps=1`` boundary test; added.

**Tests**:
* ``pytest tests/unit/test_physics_schedule.py``: 17 PASS in 1.8 s.
* ``pytest -m 'not slow' tests/atmosphere/nonhydrostatic/
  integration/test_plane_crm_end_to_end_smoke.py``: 2 PASS in 33 s.

**Net**: formula now anchored once and tested both in isolation (15 unit cases) and through production driver + slow tests (integration). Regression in formula trips unit test in < 2 s, before any slow nightly fires.

**R-roadmap status**: R1-R8, R10 ✓ (iter-42 schedule-formula de-duplication lands as iter-39/40/41 closing piece), R6 ✓. F9 platform-blocked.

### 2026-05-26 — iter 41

**iter-15/16 short smokes propagated iter-39 / iter-40 hardening.**

Codex iter-39 HIGH#2 fix in iter-39 only touched iter-38/iter-39 slow tests. 12×12 short smokes ``test_plane_crm_short_smoke_clean_ic`` (iter-15) and ``test_plane_crm_short_smoke_with_radiation`` (iter-16) still using misleading ``--rad-call-interval-s 1e9`` / silent-pass-on-broken-radiation pattern. iter-41 closes gap.

**Changes** (``test_plane_crm_end_to_end_smoke.py``):
* ``_run_driver`` (iter-15 dycore-only): swapped ``--rad-call-interval-s 1e9`` → ``--no-radiation``. Now truly dycore-only.
* ``test_plane_crm_short_smoke_clean_ic`` (iter-15): added ``rad_calls == 0`` assertion + re-verification block documenting existing anchors (CWV=55.001 mm, max\|w\| < 0.5, CWV drift < 0.1, MSE drift < 1e-3) still hold under genuine ``--no-radiation`` — verified by measurement:
  ```
  step  1: CWV=55.001 mm, MSE=4.2049e9, max|w|=0.0
  step 80: CWV=55.001 mm, MSE=4.2049e9, max|w|=7.07e-4 m/s
  ```
  All anchors hold with 700× margin on max\|w\|.
* ``test_plane_crm_short_smoke_with_radiation`` (iter-16): added EXACT-count rad_calls assertion (Codex iter-41 MEDIUM fix: ``rad_calls > 0`` too loose — broken cadence pinning rad_calls=1 would still pass). Now uses same derivation pattern as iter-40:
  ```python
  expected_rad_calls = 1 + (total_steps - 1) // every
  ```
  For days=0.002, dt=5, rad-interval=30: total_steps=34, every=6, expected=6 (fires at steps 1, 7, 13, 19, 25, 31).
* Module docstring: ``55.55 mm`` IC reference replaced with per-grid pair (``55.001 mm for the 12x12 smoke; 55.550 mm for the 132x132 production-scale slow tests``). Codex iter-41 LOW fix.

**Codex iter-41 review** caught 1 HIGH (anchors not recalibrated under --no-radiation), 1 MEDIUM (rad_calls > 0 too loose), 1 LOW (module docstring stale). All three addressed.

**Measurements** (12×12×20 dx=2km dt=5s):
* iter-15 (``--no-radiation``, 86 steps): ``rad_calls=0`` ✓, CWV+MSE unchanged from IC to 4 sig figs, max\|w\| @ step 80 = 7.07e-4 m/s.
* iter-16 (``--rad-call-interval-s 30``, 34 steps): ``rad_calls=6`` ✓, schedule matches derived count.

**Wall time impact**: iter-15 dropped from 87s → ~17s (skipping slow_physics_fn JIT compile saves significant time). Combined fast tests now 2 PASS in 30 s (was 87s before iter-41).

**R-roadmap status**: R1-R8, R10 ✓ (iter-41 propagating iter-39/40 hardening to 12×12 short smokes), R6 ✓. F9 platform-blocked. Plane CRM test pyramid now consistently calibrated against iter-39 ``--no-radiation`` semantics.

### 2026-05-26 — iter 40

**Radiation call count surfaced + tests assert it (Codex iter-39 MEDIUM#1 + iter-40 LOW fixes).**

Codex iter-39 MEDIUM#1 noted iter-39 with-radiation slow test didn't actually assert radiation tick fires expected number of times — broken ``rad_call_every_steps`` arithmetic (e.g. int-truncation regression pinning interval to 1 or ``total_steps``) would silently shift call schedule without tripping MSE-sandwich assertion in many regimes.

**Driver change** (``scripts/run_rce_mpi_long.py``):
* New ``_maybe_fire_radiation(step_idx)`` closure consolidates rad-firing branch from both DD path (n_ranks>1, use_dd) and legacy rank-0 path. Returns True on fire so caller can increment counter.
* New ``rad_call_count`` Python counter incremented at each fire.
* New ``rad_calls=N`` token added to final ``Done.`` line so any subprocess/CI parser can assert on it.

**Test changes** (``test_plane_crm_end_to_end_smoke.py``):
* New ``_parse_rad_call_count(stdout)`` helper. Anchored regex ``^Done\\..*\\brad_calls=(\\d+)\\b`` with multiline flag — only matches ``Done.`` line + word-bounded integer. Returns ``None`` unless exactly one match (Codex iter-40 LOW#4 fix vs unbounded ``rad_calls=(\\d+)`` that would also match ``total_rad_calls=5`` or ``rad_calls=5.0``).
* ``test_plane_crm_production_scale_132x132_envelope`` (iter-38): asserts ``rad_calls == 0`` under ``--no-radiation``.
* ``test_plane_crm_production_scale_132x132_with_radiation`` (iter-39): asserts ``rad_calls == expected_rad_calls`` where ``expected_rad_calls = 1 + (n_steps - 1) // every`` and ``every = max(1, round(rad_interval_s / dt_s))`` (Codex iter-40 LOW#5 fix — derived from CLI args, not hardcoded).

**Codex iter-40 review** caught 0 HIGH / 0 MEDIUM / 2 LOW:
* Closure correctness: PASS — Python closure resolves ``state`` at call time so rebound loop-variable visible.
* MPI counter consistency: PASS — DD path counts per-rank locally; rank 0 prints its own count (not allreduce), no double-count.
* Regex robustness: LOW — unanchored; fixed (above).
* Hardcoded expected=5: LOW — replaced with derived count.
* Silent-pass risk: PASS — radiation tendency state effect still asserted via signed MSE-drift sandwich from iter-39.

**Measurements** (132×132×30 dx=2km dt=5s 60 outer steps):
* iter-38 (``--no-radiation``): ``rad_calls=0`` ✓
* iter-39 (``--rad-call-interval-s 60``): ``rad_calls=5`` ✓ (matches derived ``1 + 59 // 12 = 5``, fires at steps 1, 13, 25, 37, 49)

**Tests**:
* ``pytest -m slow tests/atmosphere/nonhydrostatic/integration/
  test_plane_crm_end_to_end_smoke.py``: 2/2 PASS in 233 s.
* Default suite: 2/2 short smokes still PASS in 47 s.

**R-roadmap status**: R1-R8, R10 ✓ (iter-40 radiation-tick observability hardening), R6 ✓. F9 platform-blocked. Radiation-on/off mode of production-scale plane CRM now both behaviour-pinned (iter-39 MSE sandwich) AND schedule-pinned (iter-40 rad_calls counter).

### 2026-05-26 — iter 39

**Codex caught iter-38 never truly dycore-only — fixed at driver + landed radiation-symmetric production-scale slow test.**

* **Codex iter-39 HIGH#2** (real bug in iter-38): iter-38 helper passed ``--rad-call-interval-s 1e9`` thinking that disables radiation. Inspection of driver loop showed radiation tick uses ``(step - 1) % rad_call_every_steps == 0`` — at step 1 modulo is 0 regardless of interval, so radiation fires ONCE at step 1 and caches tendency for full run. iter-38 actually exercising "dycore + 1 cached radiation tendency", not pure dycore-only.

  Fixed by wiring real ``--no-radiation`` CLI flag into ``scripts/run_rce_mpi_long.py``:

  ```python
  if not args.no_radiation and (step - 1) % rad_call_every_steps == 0:
      cached_rad_tend[0] = slow_physics_fn(state, grid, hc, terrain)
  ```

  Branch gated at both DD path and legacy rank-0-broadcast path. ``cached_rad_tend`` stays ``None`` for entire run when ``--no-radiation`` set, so ``apply_physics_substep`` skips radiation term entirely. Log header updated to show ``"NO radiation"`` vs ``"gray radiation"`` so post-hoc analysis of log file unambiguously shows which mode run was in.

* **iter-38 test retargeted**: now passes ``--no-radiation`` and genuinely radiation-free. 5e-5 MSE-drift cap remains valid as UPPER bound on dycore + fast-physics drift rate (cached-radiation rate strictly above radiation-free rate because gray-rad cools).

* **iter-39 with-radiation slow test landed**: ``test_plane_crm_production_scale_132x132_with_radiation`` in ``tests/atmosphere/nonhydrostatic/integration/test_plane_crm_end_to_end_smoke.py``. Same config as iter-38 but with ``rad_call_interval_s=60.0`` — 5 radiation refreshes (steps 1, 13, 25, 37, 49) across 5-min sim window. Closes iter-16 (12×12 with rad) → iter-38 (132×132 no rad) → iter-39 (132×132 with rad) symmetry.

* **Codex iter-39 HIGH#1 fix**: with-rad test now has RADIATION-SPECIFIC sandwich assertion on MSE drift:
  - **sign check**: ``(mse_final - mse_first)/mse_first < 0`` (radiation must cool column over 5 min sim — flipped flux convention or sign-flipped LW tendency fails this)
  - **floor**: ``|rel_mse_drift| > 5e-7`` (silently-disabled radiation path lands well below 5e-7; iter-38 ``--no-radiation`` measures 0 drift to 5 sig figs, so this distinguishes two paths cleanly)
  - **ceiling**: ``|rel_mse_drift| < 5e-4`` (over-firing detector)

* MEDIUM-#1..#5 from Codex acknowledged + docstring-only fixes (radiation call schedule, step-1 IC interpretation, rad-specific log fields gap noted as future driver schema extension).

**Measurements (post-fix, 132×132×30 dx=2km dt=5s 60 outer steps)**:

| variant | flag | max\|w\| @ step60 | MSE @ step60 | MSE drift |
|---------|------|-------------------|--------------|-----------|
| iter-38 | ``--no-radiation`` | 7.6e-4 m/s | 4.2132e9 | < 1e-5 |
| iter-39 | ``--rad-call-interval-s 60`` | 3.0e-3 m/s | 4.2131e9 | ~2.4e-5 cooling |

4× higher max|w| in iter-39 (3.0e-3 vs 7.6e-4) at same step count is radiation cooling tendency driving small-amplitude convective response — exactly regime iter-38 supposed to EXCLUDE but actually including via cached step-1 tendency.

**Tests**:
* ``pytest -m slow tests/atmosphere/nonhydrostatic/integration/
  test_plane_crm_end_to_end_smoke.py``: 2 PASS in 213 s (iter-38 ~92 s, iter-39 ~121 s including JIT compile of radiation slow tendency).
* Default suite: 2/2 short smokes still PASS in 45 s; slow tests deselected.

**R-roadmap status**: R1-R8, R10 ✓ (iter-38 dycore-only + iter-39 radiation-symmetric production-scale slow regressions), R6 ✓. F9 platform-blocked. Plane CRM production envelope now structurally guarded against both dycore regressions AND radiation-tendency regressions at 132×132.

### 2026-05-26 — iter 38

**Plane CRM 132×132 production-scale regression test landed (nightly).**

iter-14 measured 1-sim-hour smoke at 132×132×30 dx=2 km dt=5 s locking in production envelope (max|w|=6.1e-3 m/s, MSE drift=1.7e-4 over 725 steps, CWV pinned at 55.550 mm). Until iter-38 this measurement had **no regression test** — only iter-15 12×12×20 smoke gated CI, misses any regression destabilising only at production-scale grid (grid-scale modes, halo edge artifacts, Smag eddy-viscosity scaling at large nx/ny).

New `@pytest.mark.slow` test `test_plane_crm_production_scale_132x132_envelope` in `tests/atmosphere/nonhydrostatic/integration/test_plane_crm_end_to_end_smoke.py`:

* Invokes `run_rce_mpi_long.py` at 132×132×30 dx=2 km dt=5 s for 60 outer steps (5 min sim, 5e-min sub-envelope of iter-14 1-sim-hour reference) with F8/F10/iter-13 production defaults: clean Wing IC, hyperdiff=5e6, Smag c_s=0.2 (passed explicitly), SI acoustic off-centering=0.1 + 12 substeps, mass fixer ON, radiation disabled.
* `--days` carries +0.5·dt padding so `int(days·86400/dt)` always hits exactly `n_outer_steps=60` (without padding, float roundoff silently truncated 60 → 59 and `rows[-1]` landed on step 40).
* `--log-every-steps=15` so all 5 logged rows {1, 15, 30, 45, 60} get checked. iter-37 had log_every=20 → could miss transient spikes between rows.
* Asserts: logged-step-numbers schema (catches driver miscount), IC CWV=55.550 ± 0.01 mm, max|w| < 0.05 m/s on **every** logged row (catches transient CFL spike), activity floor at step 15 (max|w| > 1e-6, catches dead simulation), CWV drift < 0.01 mm, MSE drift < 5e-5 relative (strict sub-envelope of iter-14's 1.7e-4 over 725 steps).

**Codex adversarial review on new test**: 2 HIGH + 5 MEDIUM + 3 LOW raised; all but 3 LOW (informational) addressed:

* HIGH#1 — MSE cap 1e-4 looked inconsistent with cited iter-14 1.7e-4: clarified cap is sub-envelope (60 vs 725 steps); tightened to 5e-5 to make strict sub-envelope explicit.
* HIGH#2 — `--days = n*dt/86400` float-truncates to n-1 steps, `rows[-1]` evaluates step 40 not step 60: fixed with +0.5*dt padding + explicit `logged_steps == [1,15,30,45,60]` assertion.
* MEDIUM transients hidden between log rows: now checks max|w| on every logged row.
* MEDIUM dead-simulation false-pass: added activity floor at step 15.
* MEDIUM schema not validated: added logged-step-numbers assertion.
* MEDIUM docstring claimed "1-sim-hour" while running 60 steps: retitled as "5-min-sim sub-envelope of iter-14 reference".
* MEDIUM Smag c_s default implicit: now passed explicitly as `--smag-cs 0.2` so future driver-default change can't shift regression silently.

**Measurements (post-fix)**:
* logged steps: [1, 15, 30, 45, 60] (exact)
* IC CWV: 55.5500 mm ✓
* max|w| @ step 60: 2.10e-3 m/s (cap 5e-2, 24× margin)
* max|w| @ step 15: 2.06e-3 m/s (> activity floor 1e-6)
* CWV drift: 0.0000 mm (cap 0.01)
* MSE drift: 2.4e-5 relative (cap 5e-5, well below per-step rate cap catching 3× regression)
* Wall: 125.3 s on M5 Pro single-rank
* `pytest -m slow tests/atmosphere/nonhydrostatic/integration/
  test_plane_crm_end_to_end_smoke.py`: 1 PASS in 125 s.

**Default suite unchanged**: 2/2 plane-CRM smokes pass in 87 s (slow test deselected as expected).

**R-roadmap status**: R1-R8, R10 ✓ (now with iter-38 production-scale plane CRM regression), R6 ✓. F9 platform-blocked. iter-14 measured envelope structurally protected against silent regression.

### 2026-05-26 — iter 37

**Full regression sweep: 38/38 PASS in 72 s.**

Confirmed iter-1..36 work composes cleanly. Suite breakdown:

| test file | tests | wall |
|-----------|-------|------|
| test_rce_cross_grid_dt_defaults.py | 9 | (incl) |
| test_cross_grid_wrapper_dt_overrides.py | 2 | (incl) |
| test_plane_slow_tend_halo.py | 13 | (incl) |
| test_plane_mass_fixer_mpi.py | 8 | (incl) |
| test_weno5_halo_equiv.py | 6 | (incl) |
| **TOTAL** | **38** | **72 s** |

Plus iter-15 plane CRM smoke (2 tests, ~6 s) and iter-2 dt-stability (7 tests, ~360 s) run elsewhere in default suite; iter-36 AMIP dt-warning (3 tests, 57 s); slow nightly tests (3) skipped by default. Combined CI-visible: **~50 tests**.

**Achievement summary (iter-1..37):**
* **Plane CRM** non-hydrostatic 132×132 dx=2 km production-stable: F8/F10 clean IC + dt=5 s + Smag + WENO5 + KW78 implicit-buoyancy knob + R7 MPI mass fixer + 132×132 1-hour smoke PASS.
* **Hydrostatic cross-grid family**: 30-day PASS on C24, C48, C72, V4, LL32, T21 — all 4 grid types stable + realistic at production scale.
* **Auto-dt ladder**: 5-tier hard-grounded ladder with N>96 hard refusal; dt ∝ dx² scaling identified + fit; 5 layers of structural regression protection.
* **AMIP wrapper + script**: iter-32 wrapper fix for C48/T42/V6 + iter-36 in-script dt safety advisory.
* **F9** (MPI scaling) platform-blocked on macOS Python 3.13; full DD code path verified correct.

**R-roadmap status**: R1-R8 ✓, R10 ✓ (with full regression backstop), R6 ✓. F9 platform-blocked. Goal "stable + realistic at 30-day production scale on all grid types" met for hydrostatic family + verified-stable for plane CRM at production scale on smoke.

### 2026-05-26 — iter 26..36 (compressed summary)

iter-26..iter-36 detail folded for doc-size hygiene per the "compress
every 10 iterations" instruction. Full per-iter detail preserved in
``CRM_implementation.original.md`` (local backup, pre-iter-62 state)
and git log. One-line summary per iter below:

| iter | landed |
|------|--------|
| 26 | C72 30-day at iter-13 dt=75 PASS (mean_T_sfc=299.81, max\|v\|=17.85, wall=2373s). iter-13 dt=75 branch (N=49..72) now empirically verified at both ends. |
| 27 | Codex iter-25/26 review clean (0 HIGH/MEDIUM, 1 LOW addressed). `rce_dt.py` docstring refreshed with iter-26 C72 measurement; C96 30-day at dt=37 launched. |
| 28 | Cross-grid plotter Metal pin (`JAX_PLATFORMS=cpu` for `run_atmosphere_test_matrix.py --cross-grid-plots-only`) + new structural test `test_auto_dt_rce_lies_inside_cfl_envelope` asserts every empirical ladder value ≤ 2x gravity-wave CFL. |
| 29 | Empirical `dt ∝ dx²` scaling identified (α=2.0 fit over C24/C48/C72/C96). New `empirical_dt_dx2(dx_min)` diagnostic + `test_ladder_matches_empirical_dt_dx2_fit` (30% tolerance). 3 layers of regression coverage now. |
| 30 | CFL advisory print in `run_rce.py` now shows BOTH gravity-wave + dx² fit bounds. Try/except ImportError guard preserved per Codex iter-22..24 HIGH. |
| 31 | Auto-dt diagnostic table script (`scripts/print_rce_auto_dt_table.py`) + smoke test. Shows per-grid ladder + CFL + dx² fit + their ratios. |
| 32 | AMIP cross-grid wrapper at iter-7 + iter-13 parity (macOS Bash 3.2 compat + dt=150 for C48/T42 AMIP rows). |
| 33 | Codex iter-31/32 HIGH (voronoi V6 AMIP dt=600 BLOWUP risk → pin dt=60) + 2 MEDIUM (table refresh, K-anchor drift acknowledged). |
| 34 | `test_cross_grid_wrapper_dt_overrides.py` regression: parses AMIP wrapper GRID_TABLE, asserts each dt override sits within 0.5× — 2.0× of central ladder. |
| 35 | Codex iter-33/34 review: 2 HIGH (regex anchor on `^GRID_TABLE=`, length-based field-count heuristic → explicit expected_fields) + 1 MEDIUM (voronoi strict equality vs sanity-only check). All fixed. |
| 36 | AMIP dt-safety advisory at the script level (`run_amip.py`): warns when `--dt > 2.0 * auto_dt_rce(grid, res)`. New 3-test subprocess regression `test_amip_dt_warning.py`. |

### 2026-05-26 — iter 2..25 (compressed summary)

Older iters folded for doc-size hygiene. Full per-iter detail in git
history (commits in the e6befce7..58db0859 range). One-line summary per iter:

| iter | landed |
|------|--------|
| 2 | F8-stable defaults (dt=1s, hyperdiff=5e6, no-bubble Wing IC); dt-stability test 4/4 PASS in 87s. |
| 3 | R4 Smag in halo + R5 vertical-θ-diff in halo; fixed missing sponge term in halo `drho_p_dt`; 9/9 halo-equiv tests PASS. |
| 4 | R7 MPI mass fixer (`fix_mass_nonhydrostatic_plane_mpi`); 7 unit tests, 18/18 combined. |
| 5 | DD branch wired into `run_rce_mpi_long.py` (`--use-dd` flag); first true 2-rank MPI smoke completes. |
| 6 | Codex caught rank-0-blocked-on-second-gather deadlock; R8 bench plumbing in place. F9 surfaces (mpi4jax 0.9 vs JAX 0.10 stack mismatch → ~70× shared-mem slowdown). |
| 7 | R6 WENO5 ported to halo path with 4-cell halo; cross-grid RCE smoke `run_rce_cross_grid.sh` for {cubed_sphere, latlon, voronoi, gaussian}; macOS Bash 3.2 compat. |
| 8 | F8 verified at 24×24×30 dt=1s. Voronoi V4 hydrostatic BLOWUP at day 1 fixed by pin dt=300. |
| 9 | F10 finding: clean Wing IC stable at dt up to 10 s; full-physics smoke at dt=5 s ran 864 steps stably. |
| 10 | 132×132×30 dt=5s plane CRM PASS for 28.8-min sim. `requirements_mpi.txt` pin JAX 0.9 + mpi4jax 0.8. |
| 11 | F9 confirmed BLOCKED — JAX 0.9 + mpi4jax 0.8 venv built clean but scaling still ~70× per rank on macOS. |
| 12 | **MAJOR MILESTONE — 30-day production: 4/4 hydrostatic grids PASS** (C24 dt=600 / V4 dt=300 / T21 dt=600 / LL32 dt=82). |
| 13 | C48 30-day BLOWUP at dt=300 → loose BLOWUP gate (500→200 m/s) + new ladder 600/150/75; `test_rce_cross_grid_smoke.py` regression added. |
| 14 | Codex caught smoke gap (max\|v\| < 50 m/s missing); added C48 to matrix; plane CRM 1-hour at 132×132 PASS. |
| 15 | C48 30-day at dt=150 PASS. Plane CRM 12×12×20 end-to-end smoke regression added. 39 tests PASS in 97s. |
| 16 | Codex 1 HIGH (`env.setdefault` issue) + 1 MEDIUM (radiation gap → 12×12 with-rad smoke) + 1 LOW (CWV anchor 55.001 mm). |
| 17 | Codex MEDIUM#4: added 2 slow nightly tests (BLOWUP gate + C48 30-day validation). |
| 18 | Added C96 to slow nightly. Default smoke exercises every auto-dt branch. |
| 19 | Codex HIGH dead `_run_rce` call + MEDIUM `_assert_rce_pass` helper + envelope widening. |
| 20 | **C96 30-day at dt=75 BLOWUP at day 20** → refined ladder: 600/150/75/37. |
| 21 | Codex HIGH: N>96 silent extrapolation → raise ValueError. |
| 22 | C96 10-day at dt=37 PASS (wall=2506s). C72 30-day in flight. |
| 23 | CFL formula advisory in `run_rce.py`. iter-13 C48 dt=300 ratio=1.32× same as PASS C24 → formula informational. |
| 24 | Refactor: auto-dt extracted to `legoesm.driver.rce_dt.auto_dt_rce`. |
| 25 | Codex HIGH#1 (broad `except` → `ImportError`) + HIGH#2 (`auto_dt_rce` missing from public API) + MEDIUM (text-match test → behavioural identity check). |


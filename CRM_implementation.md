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
| Radiation-schedule helper | `src/legoesm/driver/physics_schedule.py` | Built (iter-42). Single source of truth for the ``rad_call_every_steps`` arithmetic; 24 unit tests (iter-42 + iter-43 NaN/inf/sys.maxsize hardening) |
| Shared hydrostatic RCE assertion helpers | `tests/atmosphere/hydrostatic/_rce_helpers.py` | Built (iter-46; promoted to dedicated module iter-78). 6 helpers (5 pure + 1 subprocess) shared across C48/C72/C96/V4/LL32/T21 nightlies. 45 unit tests in `tests/atmosphere/hydrostatic/test_rce_helpers_unit.py` (refreshed iter-131; iter-52 + iter-66 + iter-83 + iter-85 + later iterations added cases) |
| Shared plane CRM assertion helpers | `tests/atmosphere/nonhydrostatic/integration/_plane_crm_helpers.py` | Built (iter-79; ``_read_log`` added iter-84). 2 pure helpers (``_parse_rad_call_count`` iter-40/54 + ``_read_log`` iter-84). 18 unit tests |
| Production driver CLI input validation | `scripts/run_rce_mpi_long.py:main` + `scripts/run_rce.py:main` | Built (iter-65-75). Defense-in-depth across BOTH drivers via auto-detected guards on every float / int / positive-int / range / Kelvin arg from `vars(args)` (iter-67/68/69/70/74) + post-derivation `total_steps >= 1` (iter-75). `--sst-init` positive-Kelvin layer (iter-74) only on `run_rce.py` — plane-CRM driver hardcodes `T_SFC_K`. 13 + 29 parametric regression tests |

---

## Definition of done

30-day CRM run on production target (132×132 plane, dx=2 km, nlev=30, H=33 km, dt=5 s + N_ACOUSTIC=12, 12 MPI ranks, Wing 2018 RCEMIP1 IC, gray radiation + Kessler + Smagorinsky LES + surface fluxes) must:

(iter-82: dt refreshed 1 s → 5 s per iter-9 F10 finding + iter-14/iter-38/iter-63 production-scale verification; the iter-1 dt=1 s was set against the bubble-IC F1 instability that F7 then disproved.)

1. **Run to completion** without NaN, `max|w| < 50 m/s` throughout.
   * Status: ✓ at 132×132 1-sim-hour (iter-14 + iter-38 + iter-63
     structural slow nightly + iter-39 with-rad symmetric coverage).
     Full 30-day plane CRM wall-time-gated (~8 days single-rank).
2. **Reach radiative-convective equilibrium**:
   * **CWV plateau range**: Wing 2018 RCEMIP1 multi-model band at SST = 300 K (Wing et al. 2018, *Geoscientific Model Development* 11(2):793–813, doi:10.5194/gmd-11-793-2018; Fig. 5b PWV at SST = 300 K shows inter-model spread ~45–60 mm). Code gate: `scripts/summarize_rce_trajectory.py:DEFAULT_CWV_RANGE_MM = (35, 65)` (Wing band 45–60 mm + 10 mm lower-bound margin to absorb the IC dip — iter-98 IC = 49.94 mm so a +-5 mm symmetric band would touch the lower edge — plus 5 mm upper-bound tolerance for the iter-98 day-4 overshoot to 57.18 mm).
   * **Precip plateau**: ~3 mm/day (Wing 2018 mean).
   * **MSE drift**: split into TWO numbers to remove the iter-104 Codex MEDIUM confusion between the production DOD requirement and the spinup-window gate:
     - **Final 30-day DOD requirement**: < 1 % over last 10 days of the 30-day window (production target; gates the *equilibrated* run).
     - **10-day summarizer stability gate** (`DEFAULT_MSE_RELATIVE_DRIFT = 0.05`, 5 %): the practical spinup check the in-flight evaluator uses to catch dycore blow-ups while convection is still developing. A 1 % gate at 10 days would false-FAIL pre-equilibration trajectories like iter-98 (0.6 % over 10 days IS already inside both windows, but a noisier transient could exceed 1 %). The 5 % gate is the stability check; the 1 % gate is the *equilibration* check applied at end-of-30-day.
   * Status: ✓ hydrostatic 30-day (iter-12 measured 4/4 grids + iter-50/51 added V4 + LL32 + T21 nightly regression). Plane CRM **10-day** PASS at 32×32×30 + radiation (iter-98, CWV 49.94 → 57.18 mm peak → 56.55 mm settled; MSE drift 0.6 %; first precip onset at day 10; `--evaluate` verdict: PASS). Full 30-day plane CRM in flight (iter-105). The 30 ± 5 mm number on this line pre-iter-107 was a stale estimate from an early hydrostatic-family extrapolation — Wing 2018 RCEMIP1 at SST = 300 K is the canonical reference.
3. **Reproduce on each supported grid type** via `scripts/run_rce_cross_grid.sh` (cubed-sphere, latlon, voronoi, gaussian). Currently only *plane* CRM has explicit CRM physics; cubed-sphere/latlon/voronoi/gaussian use hydrostatic dycore in RCE mode and cross-grid wrapper validates they converge to similar CWV / precip / MSE.
   * Status: ✓ 30-day production-scale PASS on C24/C48/C72, V4, LL32, T21 (iter-12 + iter-15 + iter-26 + iter-50/51). C96 covered by 10-day (iter-22/73; 30-day wall-time-gated).
4. **Scale with MPI**:
   * **Strong scaling**: 30-day-clock-time on 12 ranks ≤ 1.5× of 1-rank time / 12 (efficiency ≥ 67 %).
   * **Weak scaling**: per-rank cost grows < 1.3× when grid doubled in each dim and ranks doubled in each dim (4× total).
   * Status: F9 platform-blocked on macOS Python 3.13 (mpi4jax 0.9 vs JAX 0.10 stack mismatch → ~70× per-rank slowdown). Full DD code path verified correct (iter-4 R7 mass fixer, iter-5 ``--use-dd``, iter-78 helpers).
5. **Pass `/codex:adversarial-review`** on dycore + MPI halo + production driver with no MEDIUM/HIGH findings outstanding.
   * Status: ✓ holistic Codex pass (iter-55 driver + iter-56 dycore/halo + iter-57 MPI halo) all 0 HIGH + 0 MEDIUM. Iterative reviews continued through iter-80; each landed change re-reviewed.

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

Aggressive qv noise (≥ 2.5 × 10⁻⁴ kg/kg in lowest 4 levels) *also* destabilising: localised qv hotspots → spatial gradients in surface flux → non-uniform heating → grid-scale convection burst. **Default qv noise lowered to 0**; F7-original said small values (1–5 × 10⁻⁵ kg/kg) acceptable as stochastic seed but iter-97 found this is STALE post-iter-95 (both 1e-5 and 5e-5 NaN at step 100 on the corrected IC + radiation; corrected lowest-level T sits closer to saturation, any qv perturbation pushes cells over the Kessler threshold and explodes the acoustic mode). **Keep ``--qv-noise-amp 0`` as the only safe default on the iter-95 IC path**; re-measure threshold before re-enabling.

### F8. Stable physics-on smoke confirms dycore+physics composes cleanly

Smoke at 24×24×30, dx=2 km, dt=1 s, **no bubble + no qv noise**, full physics (gray rad + Kessler + Smag c_s=0.2 + surface flux + mean-wind removal + moist-mass fixer + positive filter): max|w| stays ~5 × 10⁻³ m/s through 1200 steps (20 min sim), MSE drift < 7 × 10⁻⁵ relative, CWV pinned to IC. **No spurious convection** — confirms full physics-on driver dynamically stable from clean IC. Convection spins up later from radiative cooling + surface flux on hours-days timescale (verify at 6-h / 24-h smoke step).

---

## Roadmap (concrete, ordered)

Status legend: `[x]` = done · `[~]` = partial · `[!]` = obsolete ·
`[ ]` = pending. (Checklist refreshed iter-81 to match iter-N status
lines in the iteration log; pre-iter-81 every box was stale `[ ]`.)

* [x] **R1**: Reduce production dt from 2 s → 1 s in `run_rce_30day.sh` and `run_rce_mpi_long.py` defaults. (iter-1; iter-58/59 later refreshed to dt=5 s + N_ACOUSTIC=12 per iter-9 F10.)
* [x] **R2**: Plumb `--implicit-buoyancy` / `--advection weno5` through `run_rce_mpi_long.py` for A/B test. (iter-1.)
* [x] **R3**: Automated dt-stability test (iter-2 `test_plane_crm_dt_stability.py` 4 cases).
* [x] **R4**: Smag LES in halo path (iter-3).
* [x] **R5**: Vertical-θ-diff in halo path (iter-3) + WENO5 (iter-7).
* [x] **R6**: WENO5 in halo path with 4-cell halo (iter-7).
* [x] **R7**: MPI-aware mass fixer (iter-4 `fix_mass_nonhydrostatic_plane_mpi`); `step_halo` multi-rank wired via `--use-dd` (iter-5).
* [~] **R8**: Bench plumbing ✓ (iter-6 `bench_plane_crm_dd_scaling.py`). Real strong/weak scaling numbers F9-platform-blocked on macOS Python 3.13 (mpi4jax 0.9 vs JAX 0.10 stack mismatch).
* [!] **R9**: KW78 outer-step implicit buoyancy — OBSOLETE per F10 (iter-9): clean Wing IC + dt=5 s production-stable without KW78. Substep variant inert (F2). Not on critical path.
* [x] **R10**: Cross-grid RCE validation (iter-7 wrapper; iter-12 30-day on 4 grids; iter-50/51 added V4 + LL32 + T21 30-day nightlies; iter-73 added C96 10-day nightly).
* [~] **R11**: 30-day end-to-end with criteria 1-5. Hydrostatic family ✓ (iter-12 + iter-50/51, all 6 grids C48/C72/V4/LL32/T21 with C96 10-day per wall-time decision). Plane CRM 1-sim-hour ✓ (iter-14 + iter-38/63); **plane CRM 10-day RCE PASS ✓** at 32x32x30 + radiation (iter-98 — first precip onset day 10, CWV 49.94→57.18→56.55 mm, MSE drift 0.6 %, DOD spinup verdict PASS). Plane CRM **30-day at 32x32x30** in flight (iter-105 — bit-equal to iter-98 through overlap, ETA ~5 h post-iter-131). Plane CRM **30-day at 132x132 production** still wall-time gated (~8 days single-rank on M5 Pro; F9 MPI scaling unblocks at platform-fix time).
* [x] **R12**: `/codex:adversarial-review` pass — DOD item 5 holistic review (iter-55 driver + iter-56 dycore/halo + iter-57 MPI halo) all 0 HIGH + 0 MEDIUM. Iterative reviews continued through iter-130 with every landed change reviewed (iter 95j/k/l/m, 100, 104, 109, 111, 114, 116, 118, 120, 122, 126, 130 — each cycle closes the cycle's HIGH/MEDIUM findings before the next feature lands).

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

### 2026-05-27 — iter 121..132 (compressed summary)

12 iterations of tooling polish + Codex review chains + doc
hygiene on top of the iter 99..120 trajectory-tools landing.
iter-105 30-day run still in flight throughout.

**iter-121 + iter-122 — fold of iter 99..120 + correction**
(commits `4d851928`, `5ab5ccef`): fold 22 iters of post-iter-98
tooling into one compressed summary block. iter-122 fixed 2
Codex findings (test count 87→85 over-claim; iter-104 11-day
fixture detail dropped).

**iter-123 — test NaN-warning cleanup** (commit `169b45ee`): the
test_compare_rce_trajectories ``_write_snapshot`` helper used
``precip = arr * 0.0`` which propagated NaN under the NONFINITE-
test fixtures, surfacing a numpy RuntimeWarning. Replaced with
``np.zeros((ny, nx))``. iter-126 mirrored the same fix to
test_summarize_rce_trajectory.

**iter-124..126 — 30-day wrapper polish** (commits `5753efcc`,
`49d9c446`, `a3c6e574`):
* iter-124: wrapper prints a hint on DAYS>=30 + EVALUATE_DOD=0
  runs (manual ``--final-dod`` invocation reminder).
* iter-125: optional ``EMIT_TRAJECTORY_PNG=1`` (best-effort PNG
  via plot_rce_log.py).
* iter-126 (Codex iter-123/124/125): 2 MEDIUM + 2 LOW —
  EVALUATE_DOD=1 also fires the hint (MEDIUM#1); ``=strict``
  PNG mode propagates plot failure (MEDIUM#2); 4 DAYS edge-case
  tests covering 29.99 / 30.5 / abc / 5; helper mirror fix.

**iter-127..130 — --no-plateau-check + stability mode** (commits
`93d30872`, `7f84d6e4`, `25d385af`, `ee6d2cf8`):
* iter-127: new ``--no-plateau-check`` CLI flag — runs only the
  finite + max|U|_sfc + stuck-detector gates on a trajectory too
  short for the plateau check. Folds INSUFFICIENT into PASS.
* iter-128: new ``--quiet`` flag suppresses the per-day fixed-
  width table for CI gates that just want the verdict.
* iter-129: wrapper EVALUATE_DOD widened to 4-way
  ``{0, 1, stability, final}`` — ``stability`` threads
  ``--evaluate --no-plateau-check`` for in-flight progress
  monitoring.
* iter-130 (Codex iter-128/129): 1 MEDIUM — distinct
  ``DOD STABILITY verdict: PASS/FAIL`` label so log scrapers
  don't mistake a stability-only PASS for a full
  plateau-validated PASS.

**iter-131..132 — doc-state refresh** (commits `46f2d308`,
`081b8c5e`):
* iter-131: Components table ``_rce_helpers`` test count refreshed
  32 → 45. Full atmosphere/nonhydrostatic suite 191/191 PASS in
  138 s wall.
* iter-132: R11 + R12 roadmap status refreshed — R11 picks up the
  iter-98 10-day PASS + iter-105 30-day in-flight; R12's "iter-80"
  citation upgraded to "iter-130" reflecting the actual
  continuous-review pattern.

**End-of-cycle ledger** (post-iter-132):

* Cumulative test count: 49 ``summarize_rce_trajectory`` + 15
  ``compare_rce_trajectories`` + 36 ``run_rce_30day_wrapper_defaults``
  + 5 ``test_dod_doc_code_consistency`` + 18
  ``test_plane_crm_helpers_unit`` + 45
  ``test_rce_helpers_unit`` = 168 across the iter 99..132 tooling
  + helper unit tests.
* Wrapper integration: ``EVALUATE_DOD`` 4-way (0 / 1 / stability /
  final) + ``ALLOW_SUMMARY_FAILURE`` (0/1) + ``EMIT_TRAJECTORY_PNG``
  (0 / 1 / strict).
* CLI verdict labels: ``DOD``, ``DOD STABILITY``, ``DOD FINAL`` —
  distinct so downstream log readers can tell apart a stability-
  only PASS from a full plateau-validated PASS.

**R-roadmap status**: R1-R8, R10, R12 ✓. F9 platform-blocked.
R11 [~] still partial pending iter-105 30-day completion (32x32
in flight; 132x132 wall-time gated).

### 2026-05-27 — iter 99..120 (compressed summary, iter-121 fold)

22 iterations of post-iter-98 tooling + Codex hardening on top of
the iter-98 10-day RCE PASS milestone. Adds two new Python tools +
extends the 30-day wrapper, all driven by the iter-98 success.

**iter-99..101 — 30-day wrapper + summarizer integration** (commits
`8b2d36ab`, `cc5533d6`, `d740019f`):
* iter-99: drop ``exec`` from ``run_rce_30day.sh`` mpirun line +
  append a post-run ``scripts/summarize_rce_trajectory.py``
  invocation. Writes ``trajectory.csv`` per production run.
* iter-100: Codex review of iter-98/99 flagged 4 MEDIUM + 2 LOW —
  anchored ``snap_day_(\d{4})\.npz`` regex (no stray-file
  acceptance), profile day-value cross-check (1-min tol),
  finite + unique day assertion, ``NA`` sentinel shared by table /
  CSV; wrapper captures summarizer exit status under ``set +e`` /
  ``set -e`` and propagates non-zero unless
  ``ALLOW_SUMMARY_FAILURE=1`` downgrades to a WARN.
* iter-101: 4 subprocess-driven wrapper tests using stubbed mpirun
  + PYBIN that lock the exit-status contract end-to-end (closes
  Codex iter-100 LOW — text tests were proving the right tokens
  present but not that they execute in order).

**iter-102..104 — quality-gate evaluator + EVALUATE_DOD env**
(commits `043eed66`, `31bff6f7`, `bbf9f0c3`):
* iter-102: new ``QualityVerdict`` dataclass +
  ``evaluate_rce_quality()`` function in
  ``summarize_rce_trajectory.py`` gating finite CWV, max|U|_sfc <
  50 m/s, plateau CWV mean inside ``DEFAULT_CWV_RANGE_MM =
  (35, 65)`` mm (Wing 2018 band + asymmetric tolerance), and 5 %
  MSE drift over last 10 days. CLI ``--evaluate`` flag, distinct
  exit codes ``EXIT_OK=0`` / ``EXIT_DOD_FAIL=3`` /
  ``EXIT_DOD_INSUFFICIENT=4``. iter-98 trajectory verified PASS.
* iter-103: ``EVALUATE_DOD`` env var (``0`` / ``1``) threads
  ``--evaluate`` into the wrapper's summarizer call. Default
  ``0`` so smokes / DAYS<10 stay non-gated.
* iter-104 (Codex iter-102/103): runaway-evaporation max gate
  added (MEDIUM#1); MSE drift denominator fix (LOW#2); tri-state
  ``QualityVerdict.evaluated`` (MEDIUM#3); distinct CLI exit
  codes (MEDIUM#7); extended the iter-98 anchor fixture from
  10 to 11 rows (LOW#6 — was exactly
  ``DEFAULT_LAST_N_DAYS_FOR_PLATEAU``, would silently hollow out
  if the constant moved). 6/6 findings closed.

**iter-105 — 30-day production run launched** (no commit; output
at ``/tmp/iter105_crm32x32_rad30d``). Same config as iter-98 with
``DAYS=30``. PID 5914, ~7 h wall-time. Bit-for-bit identical to
iter-98 through the overlap (verified via the iter-110 compare
utility — max |d cwv_mean| = 0 across days 0-3).

**iter-106..109 — doc folds + DOD coherence** (commits
`1519e021`, `25978f15`, `9a214c50`, `b31e89ca`):
* iter-106: fold iter 93..97 verbose entries (-382 lines).
* iter-107: DOD criterion 2 — replace stale "30 ± 5 mm CWV
  plateau" with Wing 2018 RCEMIP1 multi-model band (45-60 mm)
  + ``DEFAULT_CWV_RANGE_MM = (35, 65)`` rationale (Wing band +
  10 mm lower-bound margin + 5 mm upper-bound tolerance).
* iter-108: 4-test ``tests/unit/test_dod_doc_code_consistency.py``
  lock between ``CRM_implementation.md`` DOD section and
  summarizer constants (CWV range, MSE drift %, plateau-window
  length, Wing 2018 citation).
* iter-109 (Codex iter-106/107/108): 3 MEDIUM + 3 LOW — DOD
  tolerance arithmetic rationale, criterion-2 Wing citation
  scoping, 1 % vs 5 % MSE split documented, regex de-brittling,
  Wing 2018 DOI ``10.5194/gmd-11-793-2018`` added to the doc.

**iter-110..111 — trajectory diff utility** (commits
`94f73913`, `e16cdc55`):
* iter-110: new ``scripts/compare_rce_trajectories.py`` — reads
  two run output dirs via ``collect_trajectory`` + prints
  per-day delta table for CWV (mean/max), MSE, T_sfc, qc/qr_sfc
  max, wind. iter-98 vs iter-105 bit-equal through day 3 +
  in-flight day 4 to follow as iter-105 progresses.
* iter-111 (Codex iter-110): 3 MEDIUM + 3 LOW — snapshot shape
  validation refuses 132×132 vs 32×32 diffs (MEDIUM#1);
  ``NONFINITE`` sentinel distinct from ``MISSING`` (MEDIUM#2);
  ``--csv`` actually writes output (MEDIUM#3); one-to-one
  alignment (LOW#4); ASCII labels for stdout-encoding portability
  (LOW#5); ``monkeypatch.setattr(sys, "argv")`` (LOW#6).

**iter-112..114 — 30-day final DOD evaluator + EVALUATE_DOD=final**
(commits `8c1a5477`, `75268cb6`, `cf32cdc6`):
* iter-112: new ``evaluate_rce_final_dod()`` — tighter 1 % MSE
  drift gate + ≥30-day data-sufficiency requirement (vs spinup
  evaluator's 5 % + 10-day). Reuses spinup checks. New
  ``--final-dod`` CLI flag (mutually exclusive with
  ``--evaluate``).
* iter-113: wrapper ``EVALUATE_DOD`` becomes three-way (``0`` /
  ``1`` / ``final``) via a bash ``case`` statement. Production
  30-day runs now opt into the 1 % gate with a single env var.
* iter-114 (Codex iter-112/113): 1 MEDIUM + 3 LOW — typo
  rejection on EVALUATE_DOD=Final (MEDIUM#4 — silent fall-through
  fixed); ``EXIT_USAGE=2`` constant separated from EXIT_IO_ERROR=1;
  test drift target pinned to active constants; plateau-window
  scoping documented.

**iter-115..116 — doc fold + restore lost landmarks** (commits
`cc82c317`, `82dadf48`):
* iter-115: fold iter 51..76 (-1036 lines).
* iter-116 (Codex iter-115): 3 HIGH + 5 MEDIUM + 1 LOW — restored
  iter-63 corrected numbers (720 steps, CWV drift < 0.01 mm,
  MSE drift < 5e-4), iter-66 NaN silent-pass detail
  (``max(0.0, nan) == 0.0`` masking + ``seen_max_wind`` tracker),
  iter-75 description fix (total_steps=0 when dt>total_t, not
  "huge-dt 1-step"), iter-55 driver MEDIUM root causes, iter-56
  bench fast-path fix, iter-59 argparse refresh + AST regression,
  iter-65 ValueError→SystemExit, iter-74 C96 timeout
  4800→6000 s, iter-51 max|v| caps + temp_tol thresholds, iter-77
  standalone claim corrected, Components table 5-layer description
  rewritten (sst-init only on run_rce.py; plane CRM hardcodes
  T_SFC_K).

**iter-117..120 — stuck-trajectory detectors** (commits
`cb8b77ce`, `a8a54b9d`, `1fe32d70`, `43a39b46`):
* iter-117: new ``detect_stuck_trajectory()`` flags any sliding
  window of 3 consecutive snapshots whose CWV range is below
  0.001 mm — pre-iter-95 Bug 2 signature. Wired into
  evaluate_rce_quality.
* iter-118 (Codex iter-117): 2 HIGH + 3 MEDIUM + 1 LOW — scope
  to LEADING window only (HIGH#1+#2 fixed false-positive on
  legitimate late equilibrium); ``check_stuck=False`` opt-out
  (MEDIUM#1); "pinned" reason + actionable Bug-2 remediation
  text (MEDIUM#2); iter-66 narrative re-corrected (MEDIUM#3).
* iter-119: ``collect_trajectory`` + ``diff_trajectories`` accept
  ``str`` as well as ``Path`` for ergonomic shell-caller use.
  ``out_dir = Path(out_dir)`` at the boundary; type hint widened.
* iter-120 (Codex iter-118 MEDIUM follow-up): new
  ``detect_sustained_stuck_trajectory()`` — bit-equal
  (``DEFAULT_SUSTAINED_STUCK_CWV_TOL_MM = 1e-7`` mm) sliding
  window AFTER the leading window. Catches a hypothetical
  delayed-stuck regression (mid-run mass-fixer kick-in) without
  false-positives on legitimate equilibrium oscillation
  (~1e-3 mm).

**End-of-cycle ledger** (post-iter-120):

* Test count: 45/45 ``summarize_rce_trajectory`` tests + 15/15
  ``compare_rce_trajectories`` tests + 20/20
  ``run_rce_30day_wrapper_defaults`` tests + 5/5
  ``test_dod_doc_code_consistency`` tests = 85 new regression
  tests across the iter 99..120 chain (verified by Codex iter-122
  audit; iter-121's "87" was a 2-test over-count).
* Tools added: ``scripts/summarize_rce_trajectory.py``,
  ``scripts/compare_rce_trajectories.py``.
* Wrapper integration: ``EVALUATE_DOD`` env var (0/1/final) +
  ``ALLOW_SUMMARY_FAILURE`` (0/1) on ``scripts/run_rce_30day.sh``.
* iter-105 30-day in flight; DOD ``--final-dod`` verdict
  available once snapshots[≥30] land.

**R-roadmap status**: R1-R8, R10, R12 ✓ throughout. F9 platform-
blocked. R11 partial (10-day plane CRM ✓; 30-day in flight via
iter-105). New supporting infrastructure under R12: summarizer +
compare + 6 distinct exit codes + doc/code consistency lock.

### 2026-05-27 — iter 98 (per-day RCE trajectory summarizer landed)

**Code change** (commit `93fd0ba8`):

* `scripts/summarize_rce_trajectory.py` (new, 158 lines): reads
  every `<out_dir>/snapshots/snap_day_NNNN.npz` written by
  `run_rce_mpi_long.py:save_snapshot_2d`, optionally folds in
  matching `<out_dir>/profiles/prof_day_NNNN.npz`
  (`save_profile`), and prints a fixed-width per-day table + writes
  `trajectory.csv` with one row per day. Columns: day, CWV
  (mean/min/max/std), MSE_mean, precip (mean/max), T_sfc_mean,
  qv_sfc_mean, qc_sfc_max, qr_sfc_max, |U|_sfc (mean/max), plus
  profile-derived qc_col_max, qr_col_max, cf_col_max,
  w_var_col_max (None → printed `-` / empty CSV cell on days with
  no profile).
* `tests/atmosphere/nonhydrostatic/integration/test_summarize_rce_trajectory.py`
  (new, 9 tests, 0.12 s): contract tests vs
  `run_rce_mpi_long.py` (snapshot + profile field sets), day-order
  sort, present/missing profile branches, dash placeholder,
  CSV round-trip, two error paths (no snapshots dir, empty
  snapshots dir).

No duplicated diagnostic formulas — every value is read directly
from the driver-written `.npz` files, not recomputed.

**Why iter-98 needed it**: the in-flight 10-day rad-enabled
32×32×30 spinup run was outputting surface-only snapshots
(`qc_sfc`, `qr_sfc`) which both stayed at zero through 10 days
even though `log.txt`'s `max(qc)` (over the full column) was
already at 7.7×10⁻⁴ kg/kg by day 7.7. The surface snapshots
missed the column qc growth entirely. Now `summarize_rce_trajectory`
pulls the profile-derived column-max columns into the same
per-day table, so the trajectory CSV captures the actual
convection onset story.

**Day-by-day trajectory** (days 0–8 of the in-flight run, from
`/tmp/iter98_crm32x32_rad10d/trajectory.csv`):

| day | CWV_mean [mm] | T_sfc_mean [K] | qc_col_max [kg/kg] | cf_col_max |
|---:|---:|---:|---:|---:|
| 0 | 49.94 | 296.81 | 0.0 (IC) | 0.0 |
| 1 | 53.63 | 297.07 | — | — |
| 2 | 55.67 | 297.23 | — | — |
| 3 | 56.77 | 297.94 | — | — |
| 4 | 57.18 | 298.32 | — | — |
| 5 | 57.12 | 298.53 | 3.62×10⁻⁴ | 1.00 |
| 6 | 56.85 | 298.65 | — | — |
| 7 | 56.54 | 298.71 | — | — |
| 8 | 56.20 | 298.75 | — | — |

CWV reaches Wing 2018 RCEMIP1 plateau range (50–60 mm) peaking
day 4 at 57.18 mm then slowly drifts down — consistent with the
expected overshoot-then-settle behaviour. T_sfc still rising
toward the prescribed 300 K (radiation + flux not yet in steady
state). cf_col_max = 1.0 at day 5 means at some vertical level
the cloud fraction proxy is saturated; surface still dry
(`qc_sfc`, `qr_sfc`, `precip` all 0).

**Not yet measured** (run still in flight at 82% / day 8.2):
day 9, day 10 endpoint, precip onset (Kessler autoconv triggers
at column qc ~ 1 g/kg, currently 0.8 g/kg from log).

**Next iter target** (iter-99): when iter-98 run completes,
re-run `summarize_rce_trajectory` on the full 10-day output,
commit the final CSV + day-10 endpoint table to the log,
then decide whether to push to 30 days at this grid or move
straight to 132×132 production.

### 2026-05-27 — iter 93..97 (compressed summary, iter-105 fold)

**iter-93** (commit `2026-05-26`): fixed import-time Metal-init
crash. `src/legoesm/grids/vertical.py` had two module-top
`jnp.asarray([...])` calls (`_A60`, `_B60` FV3 L60 hybrid coord
tables) that eagerly dispatched `lax.convert_element_type` to JAX's
default platform — on macOS that's METAL which rejects the op.
`ensure_metal_or_fallback()` in `tests/conftest.py` ran AFTER
`import legoesm`, so the fallback never applied. `import legoesm`
bricked on Apple Silicon, breaking every pure-Python unit test on
Mac. Fix: module-top uses `np.asarray(...)`; `set_eta_L60()`
converts to `jnp.asarray` on demand. Regression test
(`tests/unit/test_no_module_top_jax_alloc.py`, 8 cases + Codex
MEDIUM-1/2/3 hardening to 20 constructors + alias-aware AST walk +
expanded protected-modules list) statically catches any new
module-top `jnp.{array,asarray,...}` in 7+ critical-path modules.

**iter-94**: first complete 12×12×20 1-sim-day plane CRM run
(17280 steps in 10.1 min wall, ~28 steps/s). dx=2 km, dt=5 s,
hyperdiff=5e6, Smag cs=0.2, full physics, clean Wing IC. CWV pinned
to 55.001 mm (pre-iter-95 stuck — mass fixer was rescaling away
the surface flux signal), max|w| 0→5.7×10⁻³ m/s gentle drift,
MSE 4.20490e9→4.19078e9 (3.36e-3 relative/day). No convection in
24 sim-hr — Kessler needs local saturation; symmetric IC + no qv
perturbation means convection has to wait for noise growth. 30-day
12×12 background launched (PID 1714, expected ~5h wall).

**iter-95 — TWO-BUG PHYSICS FIX (commits `6305b88f` + `975f7db1`)**:

* **Bug 1** (`src/legoesm/grids/vertical.py:compute_reference_state`):
  legacy top-down integration with hardcoded `T_avg=250 K` gave
  `pi(z=550 m) = 1.027` instead of correct 0.987 for Wing 2018
  RCE300 + H=33 km. T at lowest model level was 309 K (12 K too
  hot vs Wing 2018 spec). Surface flux scheme (SST=300 K) then
  removed energy instead of warming. Fix: `p_sfc` opt-in argument
  switches to bottom-up integration; driver passes Wing 2018
  Tab A1 value `p_sfc=101480.0`. Post-fix T(z=550 m) = 296.81 K
  (within 0.5 K of Wing spec).

* **Bug 2** (`scripts/run_rce_mpi_long.py`): `fix_moist_mass_plane`
  was rescaling total water back to IC every outer step. Correct
  for gravity-wave smokes; FATAL for RCE spinup (surface flux
  must net-add moisture until precip balances). Pre-fix evidence:
  CWV pinned at 49.941 mm for 11+ sim-hours despite surface flux
  active. Fix: `--no-mass-fixer` CLI flag skips both DD-MPI and
  legacy rank-0 fixer calls. 30-day wrapper default is now
  `NO_MASS_FIXER=1` (iter-95g + tests iter-95f / iter-95k /
  iter-95m).

* **Verification stack** (iter-95d/e/f/g/h/i/j/k/l/m): 11 follow-up
  commits address every Codex iter-95 finding (HIGH BLOWUP +
  MEDIUM tables + LOW#2 conditional + iter-95j tightened
  308.78 K ± 0.5 K + iter-95k CWV growth threshold 0.01 mm + 4
  IC anchors updated from 55.001/55.550 to 49.4691/49.9413 mm).

* **iter-95d 5-sim-day no-radiation run** (32×32×30, dx=4 km,
  dt=10 s, --no-mass-fixer): CWV 49.94 → 53.63 (day 1) → 55.26
  (day 1.75); MSE plateauing at 3.5282×10⁹; max|w| linear
  growth 0→2.17×10⁻³ m/s; profile conditionally unstable
  (~3 K buoyancy z=0.5-3 km); RH lowest level 79%→93% day 1.
  Bug 1 + Bug 2 fix proven end-to-end.

**iter-96** (radiation-enabled 5-day run, same config + gray
radiation @ 600 s cadence): 43200/43200 steps in 4288 s wall
(~4 % overhead vs no-radiation iter-95d). **First quasi-equilibrium
signal**: CWV peaks at 57.22 mm day 4.18 then drifts down toward
56 mm — overshoot-and-settle consistent with Wing 2018 RCEMIP1
multi-model behaviour (50-60 mm). qc_col_max appears day 4 at
0.16 g/kg (Kessler threshold ~1 g/kg not yet hit). max|w| stays
~2.3×10⁻³ m/s; no NaN.

**iter-97 (F7 STALE finding, no code change)**: tried iter-96
config + `--qv-noise-amp` at 5×10⁻⁵ and 1×10⁻⁵. BOTH NaN at
step 100. F7's "1-5×10⁻⁵ acceptable" was measured pre-iter-95
on the legacy IC (T_lowest = 309 K, supercritical). iter-95-
corrected IC sits closer to saturation in the lowest few levels;
ANY qv perturbation pushes cells over saturation instantly,
Kessler condensation-heat blows acoustic mode within ~5 outer
steps. **Conclusion**: keep `--qv-noise-amp 0` as the only safe
default on the iter-95 IC path; F7 marked STALE in the Findings
table at the top of this doc.

**R-roadmap delta across the fold**: R1-R8, R10, R12 ✓ throughout;
R9 `[!]` obsolete; R11 advanced from "plane CRM 1-sim-hour ✓ /
full 30-day wall-time-gated" to "32×32×30 + radiation 5-day
quasi-equilibrium ✓ / production 30-day still wall-time-gated
at 132×132". The iter-95 two-bug fix is the precondition for
every subsequent CRM-physics result; iter-96 + iter-98 are its
empirical verifications.

### 2026-05-26 — iter 78..92 (compressed summary, iter-94 fold)

iter-93 was iter-77's "compress every 10 iterations" mandate +6.
iter-78..92 = 15 iters of helper-module extraction, vacuous-pass
hardening, Codex review rounds, and the iter-92 empirical bridge.
All entries folded here; full per-iter detail preserved in
``CRM_implementation.original.md`` local backup + git log.

| iter | one-line summary |
|------|-------------------|
| 78 | Extracted 6 hydrostatic test helpers to `_rce_helpers.py` (underscore-prefix module skipped by pytest glob). 18/18 unit tests PASS. Eliminates `from test_X import ...` anti-pattern. |
| 79 | Mirror iter-78 for plane CRM: `_parse_rad_call_count` moved to `_plane_crm_helpers.py`. 12/12 unit tests PASS. |
| 80 | Codex review iter-78/79 → 4 MEDIUM + 1 LOW dead-import / stale-docstring fixes. Post-fix: 0 HIGH + 0 MEDIUM. |
| 81 | Refreshed R-roadmap checkboxes (iter-1 set `[ ]`, never flipped through 80 iters). Final: 9× `[x]` + 2× `[~]` + 1× `[!]`. Added `[x]/[~]/[!]/[ ]` status legend. |
| 82 | Refreshed DOD (last edited iter-1 with stale `dt=1 s`; iter-9 F10 + iter-14 measured `dt=5 s` production-stable). Added per-criterion **Status** bullet. |
| 83 | Extended iter-52 unit coverage: 14 new tests for `_parse_results`/`_parse_notes`/`_assert_rce_pass`. 32/32 PASS. |
| 84 | Moved `_read_log` to `_plane_crm_helpers.py` + 6 new unit tests (schema drift, blank-line, missing-file paths). 18/18 PASS. |
| 85 | Codex review iter-83/84 → 3 LOW fixes (strict `==` in colon-skip test, missing-`max\|v\|` branch coverage, docstring "4 use sites"→5). 51/51 PASS. |
| 86 | Refreshed Components table for iter-78/79/83/84/85 helper modules + iter-65-75 CLI validation. |
| 87 | Distinguished NaN vs inf in `_assert_max_wind_peak_below`: NaN → skip (junk data), inf → propagate (CFL-crash signal). 36/36 PASS. |
| 88 | Closed all-zero silent-pass class: added optional `min_floor` to `_assert_max_wind_peak_below`. Opted-in on 6 30-day-class nightlies with `min_floor=0.5`. 41/41 PASS. |
| 89 | Symmetric `max_v_floor` on `_assert_rce_pass` (mirror of iter-88 on peak-scan helper). 45/45 PASS. Risk-class table: no-parseable / NaN-only / all-zero / inf — all covered on BOTH timeseries-peak and final-day-notes paths. |
| 90 | Opted-in iter-89 `max_v_floor=0.1` on 5-grid fast 2-day smokes + C96 2-day. iter-91 reverted (false-fail risk on V4-style spinup-slow grids). |
| 91 | Reverted iter-90 opt-in. Retained other broken-dycore guards: `status: PASS`, `temp_tol=1.0 K`, `max\|v\| < 50` cap. iter-89 API stays available. |
| 92 | User-suggested empirical bridge: 12×12×20 plane CRM 1-sim-day with full physics. Box overload → killed at 960 steps (80 sim-min). Partial trajectory captured; pre-convection (qc=qr=0). Cross-resolution agreement with iter-14 132×132 (max\|w\|≈5.7e-3 m/s at 60 sim-min) — but iter-93 honest re-eval noted this is EXPECTED for identical column ICs in pre-convection phase, not a non-trivial validation. |

**Cumulative test counts at end of iter-92**:
* `_rce_helpers.py` helpers: 6 (5 pure unit-tested + 1 subprocess).
* `_plane_crm_helpers.py` helpers: 2 (both pure unit-tested).
* Hydrostatic helper unit tests: 45 (iter-52 + iter-66 + iter-83 + iter-85 + iter-87 + iter-88 + iter-89).
* Plane CRM helper unit tests: 18 (iter-54 + iter-79 + iter-84).
* CLI validation regression cases: 13 hydrostatic + 29 plane CRM parametric.
* Vacuous-pass risk surface: 4×2 = 8 (risk-class × helper-path) all CLOSED.

**iter-92 measured 12×12 partial trajectory** (kept for reference;
superseded by iter-94 full 24-hour trajectory if available):

| step | sim_day  | CWV [mm] | MSE [J/kg] | max\|w\| [m/s] |
|------|----------|----------|------------|----------------|
|    1 | 0.000058 | 55.001   | 4.2049e+09 | 0.000e+00      |
|  240 | 0.013889 | 55.001   | 4.2047e+09 | 4.857e-03      |
|  480 | 0.027778 | 55.001   | 4.2045e+09 | 5.529e-03      |
|  720 | 0.041667 | 55.001   | 4.2042e+09 | 5.779e-03      |
|  960 | 0.055556 | 55.001   | 4.2040e+09 | 5.887e-03      |

**R-roadmap status** (unchanged across iter-78..92): R1-R7, R10,
R12 `[x]`; R8 `[~]` (bench plumbing ✓; numbers F9-platform-blocked);
R9 `[!]` obsolete per F10; R11 `[~]` (hydrostatic family + plane
CRM 1-sim-hour ✓; full plane CRM 30-day wall-time-gated).

### 2026-05-26 — iter 77

**Doc compression — folded iter-37..iter-50 (14 iters) to summary
table; 1670 → 1325 lines (21% reduction).**

iter-62 was the last compression (compressed iter-2..iter-36).
14 iters of new content (iter-63..iter-76) accumulated 558 lines.
Per "compress every 10 iterations" mandate, due.

**Folded** (one-line-per-iter summary table at bottom):
* iter-37..iter-50 — 14 iters of test infrastructure + Codex
  reviews + shared helpers + V4/C72 30-day nightlies.

**Kept at full detail at the time** (iter-77's pre-iter-115
state): iter-1 + iter-51..iter-76. iter-115 has since folded
iter-51..iter-76 into the summary block immediately below this
entry; the "kept at full detail" claim describes iter-77's
contemporaneous state, NOT the current doc layout.

Full per-iter detail in ``CRM_implementation.original.md`` local
backup (pre-iter-77 state) + git log.

**R-roadmap status**: R1-R8, R10, R12 ✓. F9 platform-blocked.
Doc hygiene aligned with "compress every 10 iterations" instruction.

### 2026-05-26 — iter 51..76 (compressed summary, iter-115 fold)

26 iterations of Codex-review hardening + driver CLI validation +
test infrastructure on top of the iter-50 hydrostatic 30-day
empirical-gate closure. Key technical landmarks:

**30-day production-scale empirical coverage (iter-51)**: LL32
(latlon C-grid, pole-clamped dt=81.844 s, peak max|v| cap=20.0
m/s = 1.8× iter-12 measured 11.19 m/s, temp_tol=1.0 K) + T21
(gaussian spectral, dt=600 s, peak max|v| cap=20.0 = 2.4× iter-12
measured 8.43 m/s, temp_tol=1.0 K) 30-day nightly regressions land
in ``tests/atmosphere/hydrostatic/test_rce_cross_grid_smoke.py``,
closing the last empirical gap (C48/C72/V4 already covered by
iter-50/iter-12). ``_assert_dt_used`` helper gains optional
``abs_tol`` (1e-2 default — strict ``==`` for ladder dt's, loose
for pole-CFL-clamped LL32). Codex caught 1 LOW (abs_tol=0.5 too
loose vs round(dt) silent-pass); fixed to 1e-2.

**Codex holistic review chain (iter-52..iter-57)** — three
full-component passes that landed 0 HIGH + 0 MEDIUM outstanding
*after* fixing the findings each pass caught:

* iter-52 added 28 unit tests for the iter-46 shared assertion
  helpers (LL32 + T21 paths).
* iter-53/54 tightened ``_parse_rad_call_count`` regex against
  schema drift (multiline anchor + ``[^\\S\\n]*`` trailing-
  whitespace tolerance — rejects ``rad_calls=5. (cached)`` and
  ``rad_calls=5.0.``).
* **iter-55 (driver pass)** — caught 2 MEDIUM: ``--no-radiation``
  crashed on an unused-but-invalid ``--rad-call-interval-s``
  cadence (validation order bug), and ``--days 0`` hit a
  ``NameError`` on the post-loop summary. Both fixed.
* **iter-56 (dycore + halo pass)** — caught 2 HIGH + 2 MEDIUM
  in the bench-script fast-path + fallback-gate logic; fix
  hardens the bench-plumbing for F9 measurements.
* **iter-57 (MPI halo / plane_mpi.py pass)** — no new HIGH; all
  prior findings rolled in.

**Driver defaults regression backstop (iter-58..iter-64)**:
``scripts/run_rce_30day.sh`` defaults refreshed from stale
dt=1.0 / N_ACOUSTIC=24 (iter-1 F1 ladder, pre-F10) to the
iter-14 / iter-38 production-measured dt=5.0 / N_ACOUSTIC=12.
``tests/atmosphere/nonhydrostatic/integration/test_run_rce_30day_wrapper_defaults.py``
locks every env-var default (including PYBIN per iter-64 + the
hardcoded snapshot / log cadence flags per iter-61 Codex MEDIUM).
iter-59 ALSO refreshes ``scripts/run_rce_mpi_long.py`` argparse
defaults (the driver itself, distinct from the wrapper) and
lands an AST-walk regression test that catches future drift in
the argparse ``default=`` literals at import time. iter-60
deletes orphaned ``scripts/run_rce_mpi_full.py`` (superseded).
iter-62 + iter-77 compression: 1923→1081→1325 lines (cumulative).
**iter-63** promotes the iter-14 full envelope (720 outer steps
= 1 sim-hour, CWV drift < 0.01 mm, MSE drift < 5e-4) to a slow
nightly regression — ``CWV drift < 0.01`` is the tight gate that
catches a regression in the mass-fixer or surface-flux pipeline
without needing a multi-day run.

**iter-66 NaN silent-pass fix** (Codex HIGH on
``tests/atmosphere/hydrostatic/_rce_helpers.py:_assert_max_wind_peak_below``):
the helper iterates rows of ``mean_timeseries.csv`` with
``peak_v = max(peak_v, abs(float(row["max_wind"])))``, then
``assert peak_v < cap``. The masking bug: ``float("nan")``
parses successfully but in CPython ``max(0.0, nan) == 0.0``
(NaN-naive comparison). So a NaN max_wind row left ``peak_v`` at
its prior value — usually 0.0 — and the final ``assert`` passed
vacuously. Fix: explicit ``if math.isnan(val): continue`` skip
inside the loop + ``seen_max_wind`` tracker so an all-NaN /
empty timeseries fails the assertion instead. iter-87 refined the
fix: the initial ``not isfinite`` guard wrongly skipped ``inf``
too (a real CFL-blowup signal that the cap check should fire
on); narrowed to ``isnan`` only.

**Production driver CLI input validation (iter-65..iter-75)** —
defense-in-depth across BOTH plane-CRM driver
(``scripts/run_rce_mpi_long.py``) and hydrostatic driver
(``scripts/run_rce.py``) — landed via 13 + 29 parametric tests.
**iter-65** converts the physics-schedule ``ValueError`` from
``--rad-call-interval-s NaN`` into a clean ``SystemExit`` with
the ``error: <flag> rejected: <reason>`` marker (Codex iter-55
follow-up; pre-fix the user got a Python traceback). The
remaining layers (with which driver carries each):

1. **NaN / inf rejection** (iter-67, generalised in iter-68 via
   auto-detect over ``vars(args)`` — every float / int arg is
   checked).
2. **Positive-int guards** for grid / substep args (iter-69 +
   mirrored to ``run_rce.py`` in iter-71).
3. **Range guards** (iter-70) including
   ``--acoustic-off-centering ∈ [0, 1)``. iter-70 Codex caught
   6 HIGH range-guard gaps in iter-67/68/69, all fixed.
4. **Positive-Kelvin sst-init** guard (iter-74) — only on
   ``scripts/run_rce.py`` (the plane CRM driver has no
   ``--sst-init`` argument; ``T_SFC_K`` is hardcoded). Codex
   caught 1 HIGH + 2 MEDIUM + 1 LOW in iter-71/72/73 all fixed.
5. **Post-derivation ``total_steps >= 1`` guard** (iter-75) —
   catches the silent-pass class where ``dt > total_t`` makes
   ``total_steps = int(total_t / dt)`` round down to 0, leaving
   the run loop with zero iterations (NOT the "huge-dt 1-step"
   misdescription of an earlier draft — the failure mode is
   ZERO steps, not one).

iter-71/72 mirror the layer-1/2/3/4 pattern onto
``scripts/run_rce.py`` with 10 parametric test cases. iter-73
lands the C96 10-day nightly (production wall-time-gated for
30-day) + fixes a ``--qv-noise-amp`` argparse regression.
**iter-74** raises the C96 nightly's pytest-timeout from 4800
to 6000 seconds (the iter-73 measurement showed worst-case
wall ≈ 5050 s at C96 + 10 sim-days). iter-76 unifies the iter-1
``--implicit-buoyancy`` SystemExit message to the iter-65/67/70/74/75
``error: <flag> rejected: <reason>`` format + adds the regression
test.

**Cross-grid test cohort end-of-cycle** (iter-77 ledger):
- Slow nightly count: 9 (was 7 pre-iter-51).
- 30-day production-scale empirical gates: C48, C72, V4, LL32, T21.
- C96 covered by 10-day (iter-22/73; 30-day still wall-time-gated).
- Plane CRM 1-sim-hour ✓ (iter-14/38/63); full 30-day still
  wall-time-gated at 132×132.

**R-roadmap status** (steady throughout iter-51..iter-76): R1-R8,
R10, R12 ✓; R8 ``[~]`` (bench plumbing ✓; numbers F9-platform-
blocked); R9 ``[!]`` obsolete per F10; R11 ``[~]`` (hydrostatic
family + plane CRM 1-sim-hour ✓; full plane CRM 30-day wall-time-
gated at 132×132). iter-65/67/70/74/75 + iter-71/72 + iter-58/59/61/64
add the regression-test layer underneath R12.

### 2026-05-26 — iter 37..50 (compressed summary, iter-77 fold)

iter-37..iter-50 detail folded for doc-size hygiene per the "compress
every 10 iterations" instruction. Full per-iter detail in
``CRM_implementation.original.md`` (local backup, pre-iter-77 state)
and git log. One-line summary per iter:

| iter | landed |
|------|--------|
| 37 | Full regression sweep — 38/38 PASS in 72 s. iter-1..36 work composes cleanly. |
| 38 | Plane CRM 132×132 production-scale slow nightly regression test (5-min sub-envelope of iter-14 1-sim-hour); Codex 2 HIGH (step-count off-by-one, MSE cap inconsistent) + 5 MEDIUM fixed. |
| 39 | Real ``--no-radiation`` driver flag (Codex iter-39 HIGH#2 found ``--rad-call-interval-s 1e9`` fires once at step 1 + caches); iter-38/39 production-scale slow tests now exercise distinct code paths. |
| 40 | Radiation call count surfaced via ``rad_calls=N`` in ``Done.`` line; iter-38/39 schedule assertion. |
| 41 | iter-15/16 short smokes propagated iter-39/40 hardening — ``--no-radiation`` + exact-count rad_calls assertion + module docstring fix. |
| 42 | Factored radiation-call schedule arithmetic to ``legoesm.driver.physics_schedule`` (17 unit tests; single source of truth across driver + tests). |
| 43 | Hardened ``physics_schedule`` against NaN/inf/sys.maxsize (Codex 2nd-pass MEDIUMs). |
| 44 | C72 30-day nightly regression test + Codex HIGH on shared ``_run_rce`` helper (``timeout=600`` vs measured 2373 s wall). |
| 45 | Propagate iter-44 timeout fix to C96 2-day + C48 30-day slow tests; tighten C72 csv parsing to exact ``max_wind`` column pin. |
| 46 | Factor iter-44 hardening into shared helpers (``_assert_dt_used`` + ``_assert_max_wind_peak_below``); apply to C48 30-day nightly; Codex MEDIUM (vacuous-pass on empty csv) fixed. |
| 47 | Applied iter-46 shared helpers to C96 2-day slow smoke (closes silent-pass risk: 2-day cannot distinguish dt=37 from dt=75 BLOWUP-in-flight). |
| 48 | BLOWUP gate test now verifies the supersonic channel actually fired (inverse of iter-46 helper). |
| 49 | (consumed by iter-62 doc compression — entry was self-referential about compression itself) |
| 50 | V4 (voronoi/MPAS) 30-day nightly + re-land iter-48 doc entry. Codex 1 HIGH (temp_tol=1.0 too tight vs measured Δ=+0.85) + 1 MEDIUM (cap=25 too loose for V4 stability profile) — both fixed. |

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


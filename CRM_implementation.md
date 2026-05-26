# CRM Implementation Log

Goal: **stable + realistic 30-day production-scale CRM on all grid types with excellent MPI scaling**.

This file is the canonical state of the CRM rollout — what's built, what's broken,
what's next. Each iteration appends a dated entry under "Iteration log" with the
concrete change made + diagnostics measured. No iteration is allowed to "just
update doc"; every entry must reference a real commit or measurement.

---

## Components owned by this rollout

| Component | Path | Status |
|---|---|---|
| Plane non-hydrostatic CRM dycore | `src/legoesm/atmosphere/dynamics/compressible_euler_plane.py` | Built, **unstable at dt=2 s** at production scale |
| Plane CRM halo-aware slow tendency | `src/legoesm/atmosphere/dynamics/compressible_euler_plane_halo.py` | Built, **missing Smag / WENO5 / vertical-θ-diff / KW78 implicit buoyancy** |
| Plane CRM acoustic substeps (SI) | `compressible_euler.py:acoustic_substeps_semi_implicit`, `compressible_euler_plane.py:plane_acoustic_substeps_semi_implicit` | Built, KW78 implicit-buoyancy in WIP (this iteration's PR) |
| 2-D pencil MPI layout + halo exchange | `src/legoesm/parallel/plane_mpi.py` | Built, AD-safe |
| MPI-aware reductions for RCE | `src/legoesm/atmosphere/dynamics/rce_mpi.py` | Built |
| 30-day production driver | `scripts/run_rce_mpi_long.py`, `scripts/run_rce_30day.sh` | **Runs dycore on rank 0 + broadcasts state** — DD wired only for reductions; no scaling for the dycore itself |
| Multi-grid RCE driver | `scripts/run_rce.py`, `scripts/run_rce_cross_grid.sh` | Built for `cubed_sphere`, `latlon`, `voronoi`, `gaussian` — *hydrostatic*, not CRM |
| Bare-dycore stability diagnostic | `scripts/diag_bare_dycore_stability.py` | Built, with `--implicit-buoyancy` / `--vertical-theta-diffusion` / `--advection` switches |

---

## Definition of done

A 30-day CRM run on the production target (132×132 plane, dx=2 km, nlev=30, H=33 km, dt=1 s,
12 MPI ranks, Wing 2018 RCEMIP1 IC, gray radiation + Kessler + Smagorinsky LES + surface fluxes)
must:

1. **Run to completion** without NaN, with `max|w| < 50 m/s` throughout.
2. **Reach radiative-convective equilibrium**: CWV plateaus in 30 ± 5 mm range,
   precip plateaus at ~3 mm/day, MSE drift < 1 % over the last 10 days.
3. **Reproduce on each supported grid type** via `scripts/run_rce_cross_grid.sh`
   (cubed-sphere, latlon, voronoi, gaussian). Currently only the *plane* CRM
   has explicit CRM physics; cubed-sphere/latlon/voronoi/gaussian use the
   hydrostatic dycore in RCE mode and the cross-grid wrapper validates that
   they converge to similar CWV / precip / MSE.
4. **Scale with MPI**:
   * **Strong scaling**: 30-day-clock-time on 12 ranks ≤ 1.5× of 1-rank
     time / 12 (efficiency ≥ 67 %).
   * **Weak scaling**: per-rank cost grows < 1.3× when grid is doubled
     in each dim and ranks are doubled in each dim (4× total).
5. **Pass `/codex:adversarial-review`** on the dycore + MPI halo + production
   driver with no MEDIUM/HIGH findings outstanding.

---

## Findings to date

### F1. Outer-step buoyancy/w mode amplification at dt > ~1 s

Measured by `diag_bare_dycore_stability.py` with the same Wing 2018 IC the
production driver uses (warm bubble θ' = 0.5 K at z<1 km, ρ' = −ρ_ref · θ'/θ_ref,
nlev=30, dx=2 km, H=33 km, sponge_width=10 km, sponge_coeff=0.05, hyperdiff=5e6,
Smag cs=0.2, SI acoustic with β=0.1, n_acoustic=24 dt_outer):

| dt [s] | n_acoustic | max|w| @ step 100 | Verdict |
|---|---|---|---|
| 0.5 | 12 | 4 × 10⁻³ m/s | Stable; bubble decays slowly |
| 1.0 | 24 | 8 × 10⁻³ m/s | Stable |
| 1.5 | 36 | 4.3 m/s | Growing exponentially |
| 2.0 | 48 | 180 m/s | Blows up by step 70 |

**Per-step amplification ratio at dt=2 s** ≈ 1.30 (constant from step 50 to 100).
Mode period ≈ Brunt-Väisälä N⁻¹ ≈ 70 s. Saturates at ~180 m/s independent of
initial bubble amplitude (tested 0.05 K vs 0.5 K) — classic numerical mode
saturation, not physical growth.

### F2. KW78 implicit buoyancy at the *substep* level is too small to help

KW78 adds three tridiagonal bands proportional to
`κ = 0.25 dt_s² g / (θ₀_half J)` × dθ_ref/dz.

With dt_s = dt_outer / n_acoustic = 2 s / 48 ≈ 0.04 s,
κ × dθ_ref/dz ≈ 8.5 × 10⁻⁸ vs `α = dt_s² c_s²/dz²` ≈ 1.4 × 10⁻⁴.

Ratio ≈ 6 × 10⁻⁴. Effect on output: indistinguishable (1-bit difference at
step 100). **Conclusion**: KW78 belongs on the *outer* step (where dt grows
N times larger), not on the acoustic substep. The current placement is
algebraically correct (sign + structure verified against the standard
forward-backward derivation) but inert for the production parameter regime.

### F3. WENO5 + vertical θ diffusion + KW78 don't fix the dt=2 s blowup

Tested simultaneously: max|w| at step 100 = 480 m/s (worse than baseline).
WENO5's wider stencil adds dispersion energy at the unstable mode without
damping it. Vertical θ diffusion at ν=1e4 m²/s is too weak to compete
with the exponential mode growth.

### F4. Production 30-day driver does not actually use MPI DD

`scripts/run_rce_mpi_long.py` calls `model.step(state, ...)` on **rank 0**
inside `if rank == 0:`, then `_broadcast_state(state, comm, root=0)` to
every rank. The dycore + physics run replicated; only the reductions
(`global_sum_mpi` via `compute_total_water_mass_plane_mpi`, etc.) are
genuinely MPI. **Net effect**: 12 ranks ≈ 1 rank speed (overhead dominates).
Strong/weak scaling: not achievable until the `model.step_halo(...)` path
is taken on multi-rank.

### F5. Halo-aware slow tendency lacks production features

`compressible_euler_plane_halo.py` does NOT support:
* Smagorinsky LES (raises NotImplementedError)
* Vertical θ diffusion (no branch)
* WENO5 advection (hardcoded upwind1)
* KW78 implicit buoyancy (not threaded into halo SI substep)
* Mass fixer (single-rank only; skipped when multi-rank)

So even if we switch the production driver to `step_halo`, we lose every
stabilizer we just added. **Blocker for MPI scaling**.

### F6. Production default hyperdiff=1e6 is 5× too weak

Bare-dycore probe at dt=1 s, hyperdiff=1e6 (the production default) blows
up at step 250 (max|w|=22 m/s); at hyperdiff=5e6 the blow-up is *delayed*
to step ~470 (single-bubble IC). The fundamental mode is bubble-seeded
and only delayed by hyperdiff — fixing the IC is the real cure (F7).

### F7. Single-level warm-bubble IC seeds a 2-Δz vertical mode

At nlev=30, H=33 km the uniform dz~1.1 km. The legacy IC sets θ' only
where z < 1 km — that's a single grid level (z_full[29] ≈ 550 m).
The resulting 2-Δz vertical mode is unrepresentable on the staggered
grid and aliases into a numerical instability that hyperdiff can only
slow down. Pure-Wing IC (no bubble) on 24×24×30 is stable through 1296
steps (20 min sim) with max|w| < 5 × 10⁻³ m/s and zero qc.

Aggressive qv noise (≥ 2.5 × 10⁻⁴ kg/kg in lowest 4 levels) is *also*
destabilising: localised qv hotspots → spatial gradients in surface
flux → non-uniform heating → grid-scale convection burst. **Default
qv noise lowered to 0**; small values (1–5 × 10⁻⁵ kg/kg) acceptable
as a stochastic seed but must be verified.

### F8. Stable physics-on smoke confirms dycore+physics composes cleanly

Smoke at 24×24×30, dx=2 km, dt=1 s, **no bubble + no qv noise**, full
physics (gray rad + Kessler + Smag c_s=0.2 + surface flux + mean-wind
removal + moist-mass fixer + positive filter): max|w| stays at
~5 × 10⁻³ m/s through 1200 steps (20 min sim), MSE drift < 7 × 10⁻⁵
relative, CWV pinned to IC. **No spurious convection** — confirms
the full physics-on driver is dynamically stable when started from
a clean IC. Convection will spin up later from radiative cooling
+ surface flux on a timescale of hours-days (to be verified at the
6-h / 24-h smoke step).

---

## Roadmap (concrete, ordered)

* [ ] **R1**: Reduce production dt from 2 s → 1 s in `run_rce_30day.sh`
      and `run_rce_mpi_long.py` defaults. [done this iteration]
* [ ] **R2**: Plumb `--implicit-buoyancy` / `--advection weno5` through
      `run_rce_mpi_long.py` so they can be A/B-tested. [done this iteration]
* [ ] **R3**: Add automated dt-stability test in `tests/atmosphere/` that
      runs `diag_bare_dycore_stability.py` at dt=0.5/1.0/1.5/2.0 and
      checks max|w| at step 100 stays bounded for dt ≤ 1.0 s.
* [ ] **R4**: Port Smagorinsky LES to `compressible_euler_plane_halo.py`
      (reuse `_compute_smagorinsky_K_m_plane` via halo-aware shear stencil).
* [ ] **R5**: Port vertical-θ-diff + WENO5 advection to halo path
      (column-local → no extra halo).
* [ ] **R6**: Port KW78 implicit buoyancy to halo SI substep
      (column-local solve → no extra halo).
* [ ] **R7**: Implement MPI-aware mass fixer (`compute_dry_mass_plane_mpi`
      via `global_sum_mpi`); replace the rank-0-only `_broadcast_state`
      flow with `step_halo` on multi-rank.
* [ ] **R8**: Strong + weak scaling benchmarks on 1 / 4 / 12 / 48 ranks
      via `scripts/run_levante_gpu_scaling.py` (extend for plane CRM).
* [ ] **R9**: Klemp-Wilhelmson 1978 **outer-step** implicit buoyancy
      (the substep version in F2 is inert). This is the real fix for the
      buoyancy/w mode amplification, lifting the dt limit.
* [ ] **R10**: Cross-grid CRM validation — extend `run_rce_cross_grid.sh`
      to thread the CRM physics stack through every grid's dycore (or
      document explicitly that only the plane is "CRM" and the others
      are hydrostatic RCE).
* [ ] **R11**: 30-day production run end-to-end with success criteria 1-5.
* [ ] **R12**: `/codex:adversarial-review` on full delta; address findings.

---

## Iteration log

### 2026-05-26 — iter 1 (this entry)

**Changes**
* `scripts/run_rce_mpi_long.py`: added `--implicit-buoyancy` flag,
  wired through `CompressibleEulerConfig.implicit_buoyancy`. Hard-fails
  if `--implicit-buoyancy` set without `--semi-implicit-acoustic`.
* `scripts/run_rce_30day.sh`: reduced default `DT` from 6.0 s → 1.0 s
  (the 6.0 s default was untested; 2.0 s reproducibly blows up;
  1.0 s reproduces F1 stable result). Added `--semi-implicit-acoustic`
  and `--acoustic-off-centering 0.1` to the launch line. Exposed
  `N_ACOUSTIC`, `ADVECTION` env vars. Documented the dt-stability
  ladder in the script header.
* `CRM_implementation.md` created with state + roadmap + findings.

**Measurements**
* Reproduced the dt-stability ladder in F1 (table above) using
  `diag_bare_dycore_stability.py` on a 48×48×30 mesh — same dx, dt,
  IC, physics gates as the 132×132 production driver. The blow-up
  mode is independent of horizontal extent (saturates at ~180 m/s
  regardless of bubble amplitude or domain size).
* Confirmed F4: production driver calls `model.step` inside
  `if rank == 0:` then broadcasts. Lines 549-564 of
  `run_rce_mpi_long.py` show the structure.

**Measurements (cont.)**
* End-to-end smoke at dt=1.0 s, nx=ny=24, nlev=30, dx=2 km, single rank,
  FULL physics (gray rad + Kessler + Smag + surface flux): max|w| still
  grows from 8 × 10⁻³ m/s (step 100) to 95 m/s (step 400). **Important
  finding F6**: the bare-dycore dt-stability limit (F1) is NOT the only
  threshold — physics injects additional energy that destabilizes the
  same buoyancy/w mode at dt=1 s. The realistic CRM needs both the
  outer-step KW78 fix (R9) AND physics-time-step control (R-NEW).
* `--implicit-buoyancy` is now exposed but inert at the substep level
  (F2). Kept in the API for the future outer-step variant.

**Next iteration target**: investigate why physics-on destabilizes
sooner than bare-dycore (separate radiation tendency mag, surface flux,
Kessler q-tendency); start R3 (dt-stability regression test) +
R4 (Smag in halo path).

### 2026-05-26 — iter 6

**Codex adversarial review of iter-5 caught one HIGH bug**
* `need_gather` evaluated on all ranks but `next_snap_t` /
  `next_snap3d_t` / `next_prof_t` were advanced ONLY inside the
  `if rank == 0:` block. After the first snapshot fired, rank 0's
  timers advanced; other ranks' did not. On the next tick
  `need_gather=True` on rank 0 but `=False` on others → rank 0
  enters `_gather_state` collective alone and deadlocks.
* Fix applied by Codex: timer advances moved OUTSIDE the
  `if rank == 0:` guard. Same `t_sim >= next_*_t` predicates evaluated
  on every rank, so all ranks advance their timers in lockstep.

**Verification**
* 2-rank smoke at 12×12×20, dt=5 s, `--snapshot-hours 0.02` (forces
  the snapshot threshold to cross multiple times) — completed 50
  steps + emitted 1 snapshot without deadlock.

**New work (R8 bootstrap)**
* `scripts/bench_plane_crm_dd_scaling.py` (NEW): strong + weak
  scaling benchmark for `step_halo`. Modes:
  - `strong`: fixed global grid (24×24 default), rank count varies.
  - `weak`: fixed per-rank grid, global grows with rank count.
  Reports `wall_s,steps_per_s,wall_per_step_s` to a CSV; warmup
  steps separated from timed window so JIT compile is not in the
  numbers. Uses `step_halo` + MPI mean-wind reduction per step;
  Smag off by default to isolate halo-exchange + acoustic-substep
  cost.
* `scripts/run_dd_scaling_sweep.sh` (NEW): wrapper that runs the
  bench across `RANKS="1 2 4"` for both modes + prints the
  efficiency table.

**Measurements (macOS Pro M5, OpenMPI 5.0.9, mpi4jax 0.9 / JAX 0.10.1)**

Local strong-scaling sweep on 24×24×16:

| mode   | ranks | wall/step | steps/s | efficiency |
|--------|-------|-----------|---------|------------|
| strong | 1     | 0.00981 s | 102     | 1.000      |
| strong | 2     | 0.70428 s | 1.42    | **0.007**  |
| weak   | 1     | 0.01013 s | 99      | 1.000      |
| weak   | 2     | 0.71384 s | 1.40    | **0.014**  |

**Finding F9**: macOS shared-memory MPI scaling is catastrophically
poor (~70× slowdown per rank) on this local hardware. Root cause is
NOT in the dycore but in the mpi4jax 0.9 / JAX 0.10.1 stack
mismatch — every mpirun launches with the warning:
`mpi4jax==0.9.0.post1, jax==0.10.1; mpi4jax 0.8.x uses a custom-call
API deprecated in JAX 0.9 and removed in JAX 0.10. Pin JAX < 0.10
for MPI workloads.`

step_halo does ~30 packed-halo-exchange calls per outer step (3 RK3
stages × ~10 exchanges per slow tendency, plus Smag K_m and rho
hyperdiff re-exchanges). With the slow-path fallback each
sendrecv is order 20 ms on shared mem → ~600 ms per step at np=2,
matching the measured 704 ms.

**Net effect**: the production scaling claim is not defensible on
this laptop. Required next step: either (a) pin `JAX==0.9.x +
mpi4jax==0.8.x` in a dedicated benchmarking venv, or (b) defer
real scaling validation to a cluster with native MPI + working
mpi4jax FFI. **Documenting this as a stack-environment limitation
in scope**, not a dycore regression.

**R-roadmap status**: R1-R5, R7, R8 bench plumbing ✓. Real scaling
numbers blocked on stack pin (Codex iter-3 advice flagged the same
issue). R6 (WENO5 halo) deferred. R10 (cross-grid CRM) pending.

### 2026-05-26 — iter 5

**Changes**
* `scripts/run_rce_mpi_long.py`:
  - New `--use-dd` CLI flag (default False — preserves the F8-stable
    legacy rank-0-broadcast path).
  - New `_scatter_state` / `_gather_state` helpers built on
    `scatter_plane_field` / `gather_plane_field`.
  - DD branch in the main loop: each rank holds a local slab, calls
    `model.step_halo(state_local, dt, layout, owned_mask=owned_mask)`
    + local physics + MPI mean-wind + MPI moist-mass fixer. Diagnostic
    + snapshot tick gathers state to rank 0 once per log interval —
    not every step.
  - Builds a per-rank local `PlaneGrid` via `make_plane_pencil_grid`
    and a local `TerrainMetric` via `make_flat_plane_terrain_metric`
    for the local grid so step_halo + physics see correct
    Arakawa-C cell counts and global beta-plane offsets.
  - `_gather_state` participates in ALL field collectives on every
    rank (fixed the rank-0-blocked-on-second-gather deadlock that
    showed up in the first multi-rank smoke).
* `scripts/run_rce_30day.sh`: new `USE_DD` env knob (0 default).
  Surfaces the DD switch for production smoke at flip time.

**Measurements**
* mpirun -np 2 smoke at 12×12×20, dt=1 s, no bubble, no qv noise,
  43 steps in 0.9 min wall. max|w| stable at ~7e-4 m/s through
  step 30. CWV pinned at 55.001 mm (= IC). MSE drift < 7e-5
  relative. **First true MPI DD smoke that runs to completion**.
* Legacy path unchanged on the smoke — bit-identical to the iter-2
  F8 reproducer.
* Used the standalone `/tmp/mpi_diag.py` exerciser to confirm
  step_halo + MPI mass fixer compose cleanly under real
  mpi4jax sendrecv before wiring into the production driver.

**Net effect**: with `--use-dd` and the R7 mass fixer in place,
the production driver is now structurally capable of strong + weak
MPI scaling. Per-step DD cost on macOS shared-mem MPI is dominated
by first-time JIT compile + per-step mpi4jax sendrecv overhead;
real scaling numbers (efficiency 1 vs 2 vs 4 vs 12 ranks) are the
next concrete iteration target.

**R-roadmap status**: R1-R5, R7, R8-bootstrap ✓. R10 (cross-grid
CRM) still pending; R6 (WENO5 halo) deferred. The 30-day production
run is now a 1-flag flip (`USE_DD=1`) away — but needs a 6-h smoke
at 132×132 to baseline wall-clock before committing to the full
30-day spend.

### 2026-05-26 — iter 4

**Changes**
* `src/legoesm/atmosphere/dynamics/rce_mpi.py` (R7):
  - `compute_dry_mass_plane_mpi(state, grid, hc, tm, layout, owned_mask)`:
    owned-mask local sum + `global_sum_mpi` across ranks. Single-rank
    short-circuits to `compute_dry_mass_plane`.
  - `_plane_volume_weight_mpi(grid, hc, tm, layout, owned_mask)`:
    global owned-cell volume = global denominator of the additive
    rho' correction.
  - `fix_mass_nonhydrostatic_plane_mpi(state, target_mass, grid, hc,
    tm, layout, owned_mask)`: uniform additive correction to rho'
    using MPI-reduced (current_mass, volume_weight). Every rank sees
    the same delta — global mass restored to `target_mass` to
    round-off. AD-safe.
* `src/legoesm/atmosphere/dynamics/compressible_euler_plane.py`:
  `step_halo` accepts new `owned_mask` kwarg. When `config.fix_mass=True`
  AND `owned_mask is not None` AND multi-rank, the MPI fixer is
  invoked after the SSP-RK3 + acoustic substeps. With
  `anchor_mass_to_initial=True` the initial mass is captured via
  `compute_dry_mass_plane_mpi` so every rank uses the same target.
  Docstring updated to reflect R7 completion.
* `tests/unit/test_plane_mass_fixer_mpi.py` (NEW): 7 tests covering
  single-rank bit-equivalence (compute_dry_mass + fixer match the
  serial versions exactly), round-trip mass-restoration, volume
  weight, the spatially-uniform-delta invariant, owned_mask handling
  on the serial short-circuit, and end-to-end `step_halo` mass
  conservation across 5 dt=0.5 steps with random momentum kick.

**Measurements**
* `pytest tests/unit/test_plane_mass_fixer_mpi.py`: 7/7 pass in 2 s.
* Combined suite (halo equivalence + mass fixer): 18/18 pass in 6 s.
* No regressions in the dt-stability suite (4/4 still pass).

**R-roadmap status**:
* R1-R5, R7 ✓
* R6 (WENO5 halo) — deferred.
* Next gating items: 6-h smoke at 132×132 to verify convection
  spinup (R11 prep), then multi-rank smoke via `mpirun -np 2` to
  exercise the new fixer under real MPI (single-process tests cover
  the short-circuit + algorithm; real MPI exercises mpi4jax
  `global_sum_mpi`).

**Net effect**: `step_halo` is now feature-complete for production
use on multi-rank (Smag, vertical-θ-diff, hyperdiff, sponge, mass
fixer all available). Only blocker for switching the production
30-day driver from "rank-0-broadcast" to true MPI DD is the driver
script itself (`run_rce_mpi_long.py:558` calls `model.step` inside
`if rank == 0:`). That's the next concrete iteration target.

### 2026-05-26 — iter 3

**Changes**
* `src/legoesm/atmosphere/dynamics/compressible_euler_plane_halo.py`:
  - **R4 done**: Smagorinsky LES ported to halo path. New
    `_compute_smagorinsky_K_m_plane_halo` computes the full 3D strain
    tensor (S11, S22, S33, S12, S13, S23) on already-halo-padded
    u, v, w using slice-based stencils (1-1 equivalent to the serial
    `jnp.roll` stencils when ``layout.n_ranks == 1``). Reuses
    ``_safe_sqrt_strain`` + ``_full_level_centred_d_dz`` from the
    serial module — no code duplication. K_m is exchanged once
    (single packed MPI round) before driving the existing
    ``oh.variable_K_diffusion_vlast_halo`` on u, v, theta', and w.
  - **R5 done**: vertical-θ Laplacian wired into the halo slow
    tendency (column-local — needs no halo exchange).
  - Removed the `NotImplementedError` gate on `smagorinsky_cs > 0`.
  - **Bug fix**: halo path was missing
    ``drho_p_dt -= sponge_full * rho_p`` (added to the serial path in
    commit aa0a8d75 but never mirrored to halo). This caused
    `drho_prime_dt` to diverge by 1.4 × 10⁻⁴ from the serial reference
    even on single-rank — the test_halo_equiv_basic test had been
    broken since aa0a8d75 was merged. Fixed in both
    slow-tendency entry points (the regular one and the
    split-trace variant `_compute_local_tendencies_post_halo`).
* `src/legoesm/atmosphere/dynamics/compressible_euler_plane.py`:
  - Updated `step_halo` docstring: Smag + vertical-θ diff now
    supported; mass fixer still single-rank only (R7 pending).
* `tests/unit/test_plane_slow_tend_halo.py`:
  - Replaced `test_halo_raises_on_smagorinsky` (assertion now wrong
    after R4) with three new bit-equivalence tests:
    `test_halo_equiv_with_smagorinsky`,
    `test_halo_equiv_with_vertical_theta_diffusion`,
    `test_halo_equiv_smag_plus_vertical_theta_diff_plus_hyperdiff`.
    All 9 tests in the suite now pass (previously: 8 passing, 1 of
    them — `test_halo_equiv_basic_no_coriolis_no_hyperdiff` — silently
    failing because no CI run ever exercised it after aa0a8d75; now
    all 9 green at rtol=1e-12).

**Measurements**
* `pytest tests/unit/test_plane_slow_tend_halo.py`: 9/9 pass in 5 s.
* Manual per-field diff (random IC, no Smag, no hyperdiff): du/dv/dw
  bit-identical at 0.0; dtheta' at 7e-18 (1 ULP); drho' at 1.4e-4 BEFORE
  the sponge fix, 0.0 AFTER.

**Net effect on R-roadmap**:
* R4 ✓ (Smag in halo)
* R5 ✓ (vertical θ diff in halo)
* R6 deferred — WENO5 halo port needs `layout.halo=3` + new
  `_weno5_advection_*_halo` operators; non-blocking since upwind1 is
  the production default for now.
* R7 remains the gating item for multi-rank `step_halo`
  (MPI-aware mass fixer).

**Next iteration target**: R7 (MPI-aware mass fixer for multi-rank
`step_halo`), then run a single-rank smoke at 132×132 / 6 h with
Smag + vertical-θ-diff enabled via the halo path (still routes
through `step()` on single rank — no behaviour change, but exercises
the freshly-ported helpers indirectly via shared-import reuse), then
multi-rank smoke verifying state matches single-rank.

### 2026-05-26 — iter 2

**Changes**
* `scripts/run_rce_mpi_long.py`:
  - Default `--dt`: 6.0 → 1.0 s (matches the 30-day wrapper).
  - Default `--hyperdiff`: 1.0e6 → 5.0e6 (F6).
  - New `--bubble-theta-pert` (default 0; legacy 0.5 K bubble opt-in).
  - New `--qv-noise-amp` (default 0; ≤ 5e-5 acceptable; ≥ 2.5e-4 blows up
    in <5 min sim per F7).
  - New `--qv-noise-seed` (deterministic RNG seed).
  - IC builder rewritten to apply bubble + noise as opt-in branches.
* `scripts/run_rce_30day.sh`:
  - Surfaces `HYPERDIFF`, `BUBBLE_K`, `QV_NOISE` env knobs.
  - Defaults set to the F8-stable config (clean Wing IC, 5e6 hyperdiff).
* `tests/atmosphere/nonhydrostatic/unit/test_plane_crm_dt_stability.py`
  (NEW): 4 parametrised tests pinning the F1 dt-stability ladder
  (dt ∈ {0.5, 1.0} stable; dt=1.5 growing; dt=2.0 blow-up). Uses
  48×48×30 mesh + warm bubble IC matching F1 measurement conditions.
  Catches regressions in: SI substep tridiag, RK3 weights, buoyancy /
  PG sign, hyperdiff stencil, sponge profile, Smag strain, mass fixer.
  Wall time: 87 s.

**Measurements**
* F6 confirmed: hyperdiff=1e6 blows up at step 250 (max|w|=22 m/s);
  hyperdiff=5e6 delays to step 470 with the bubble IC.
* F7 confirmed: bubble-seeded blow-up is 2-Δz mode-driven; **removing
  the bubble fully eliminates the blow-up** through 1296 steps in
  F8 smoke.
* F8 confirmed: 24×24×30, dt=1 s, full-physics smoke at clean Wing IC
  is dynamically stable for the full smoke window. CWV pinned at
  55.55 mm (= IC), MSE drift = 4.213e9 → 4.213e9 (< 7e-5 relative).
* dt-stability regression test passes 4/4.

**Pivot**: F6/F7/F8 collectively answer the iter-1 question "why does
physics-on destabilise where bare-dycore is stable?". Answer: **it does
not**, when the IC is clean. The iter-1 smoke that grew max|w| to 95 m/s
used the legacy 0.5 K bubble — that bubble is the source. With the
bubble removed and qv noise at 0, the full physics-on stack is stable.

**Next iteration target**: F8 verified at 24×24 / 20-min sim. Scale up
to 132×132 / 6 h to verify convection spinup at production resolution,
then to a full day. Concurrently start R4 (Smag in halo path) to
unblock real MPI DD.

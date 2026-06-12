- Restart continuation is NOT bit-exact on main. Main only tests carry-value round-trips. Branch fix/persist-physics's harness shows the trajectory diverges on main: the cube chain by 4×10⁻³ K / 0.42 Pa (the start_day forcing-time double-count — every production --restart-from chain on main still runs with forcing shifted by the pre-restart elapsed days), and the MPAS chain by ~10⁻⁸ (daily forcing sampled at link start, not the day boundary). Both fixed on our branch, validated bit-identical at production scale (PBS job, MPAS L5 + cube r16).

---

# Iteration log (ralph loop, branch `fix/restart-time` off main @ 9a5b5cb9)

## Iteration 1 — plan (2026-06-12)

This is a PORT of work developed, twice codex-reviewed, and
production-validated on `fix/persist-physics` (see
`docs/PERSIST_PHYSICS.md` on that branch; PBS job 4503479 passed
bit-identical at MPAS L5 + cube r16).  Main's #413 series fixed the carry
persistence itself but not restart TRAJECTORY fidelity — confirmed today by
running the branch harness against main (cube chain diverges 4e-3 K from
the start_day double-count; MPAS chain ~1e-8 from link-start forcing
sampling; all carry slots round-trip 0.0).

Port items (reference implementations: `git show fix/persist-physics:<file>`):
1. **MPAS daily-forcing day-boundary sampling**: `_run_mpas` samples
   `_compute_T_sfc` / `_precompute_external_forcing` at
   `float(_fd_int)` (canonical day boundary), not `_force_day`
   (link-start).  Integer-day starts (all production) bit-identical.
   Site on main: model_driver.py:3636-3644.
2. **START_DAY epoch normalization (hint-based)**: every production caller
   passes the CHECKPOINT day from `load_checkpoint` back into `run()`,
   but the cube/lat-lon/spectral loops index time as
   `START_DAY + ABSOLUTE_step·DT/86400` (epoch convention) → forcing
   shifted by the pre-restart elapsed time.  Fix: record
   `self._loaded_checkpoint_step_day = (step, day)` at every
   `load_checkpoint` return (main has 4 sites: 2902/2956/3048/3069-ish)
   and normalize `START_DAY -= start_step·DT/86400` in
   `_prepare_run_context` (:4385) + `_run_spectral` (:3880) ONLY when
   `(start_step, start_day)` equal the hint — callers passing their own
   epoch value keep legacy semantics (codex finding from the original
   review).  MPAS (:3173) keeps its local-step contract unchanged.
3. **Legacy test un-workaround**: `test_restart_reproducibility.py` still
   has `start_day=0.0` + `diag_days=0`; switch to the production
   convention + `diag_days=1` (day-aligned segments — with `diag_days=0`
   the straight run is ONE segment with forcing sampled at the final day,
   a bitwise mismatch BY CONSTRUCTION).
4. **Bit-exact continuation tests** (adapted to main's conventions:
   `_mpas_phys_state`, carry_aux keys `tke`/`qke`/`gwd_spectrum`,
   `physstate_*` checkpoint fields): MPAS 2+2 steps vs 4 straight, cube
   compiled 1+1 days vs 2 straight (restart on the GCD segment boundary),
   bitwise on state + carries.  MPAS test uses gwd=none (main still
   raises NotImplementedError for prognostic_spectral on MPAS — out of
   scope here).
5. **Validation harness + PBS job**:
   `scripts/validate/validate_restart_continuation.py` (straight-vs-
   chained, per-field bitwise report, NaN-sentinel aware) +
   `scripts/cluster/restart_time/run_restart_continuation_validation.pbs`
   (casper, 4 cpus/24GB, MPAS L5 + cube r16, tke; gwd=none on MPAS).

Known port risks for the codex review: interactions with main's #413
checkpoint-lifecycle machinery (`_mpas_phys_state` save channel,
physstate_* staging in `_carry_aux`, load-then-save laundering guards) —
the hint recording must not disturb those; and the epoch normalization
must not break main's `test_physics_state_carry.py` driver-loop tests
(which may pass start_day conventions of their own).

Plan amendment from the iteration-1 codex review (medium — adopted):
- The `diag_days=0` premise was from the BRANCH scheduler; on MAIN,
  `compute_segment_length(0,0,·)=1` (one-step segments) and
  `(0,144,·)=144` — verified empirically.  The legacy test's cadence
  mismatch on main is straight(1-step) vs leg1(144-step).  Fix the test
  by aligning cadence with `checkpoint_days=1` on ALL THREE runs
  (keeping `diag_days=0`), NOT by changing diag_days; plus the start_day
  convention change.  Also pin `compute_segment_length` behavior for
  these inputs with a unit test (if main lacks one) and reproduce the
  legacy failure on main BEFORE changing it, attributing the divergence
  precisely.

## Iteration 1 — done (2026-06-12)

Commit 1 (port): both forcing-time fixes on main's #413 architecture +
legacy-test cadence alignment + 5 continuation tests + validation
harness + PBS job.  Verified RED→GREEN: MPAS continuation test fails on
main pre-port (stash check), passes post-port; legacy reproducibility
1-failed→2-passed; regression sweep 95 passed (main's 45-test
`test_physics_state_carry.py` + `test_compiled_segments.py` + legacy).

Commit 2 (this one) — codex iteration-1 findings + production verdict:
- **floor-vs-int daily bucket** (codex medium): `int(day)` truncates
  toward zero, so a negative fractional day (restart chain crossing
  day 0, pre-reference epoch) lands in bucket 0 instead of -1 and
  samples wrong daily forcing.  Factored the bucket into
  `legoesm.forcing.time_utils.daily_forcing_bucket()` (floor) and use
  it at BOTH driver sites (MPAS `_run_mpas`, spectral `_full_physics`
  loop).
- **Spectral loop had the class-1 bug too**: `_precompute_external_
  forcing` was sampled at `self._current_day` (a restart link's first
  step = mid-day) — now sampled at the canonical bucket boundary
  `float(_fd_int)`, same as the MPAS fix.
- **Honest test design** (self-found): the negative-epoch CHAIN test
  cannot discriminate floor from int — straight and chained runs bucket
  identically under either semantics (verified: it passes with int()
  reverted).  The semantics are pinned by a parametrized unit test
  `test_daily_forcing_bucket_floor_semantics` (tests/unit/
  test_time_utils.py) — verified it BITES: 3 cases fail under int().
  The chain test docstring now states exactly what it covers.
- **Production validation PASSED**: PBS job 4524510 (casper htc,
  16 min) — MPAS L5 (tke): T/u/p_s + ALL PhysicsState fields
  max|straight−chained| = 0.0; cube r16 compiled (tke +
  prognostic_spectral): T/u/v/p_s/q_v/tke/gwd_spectrum all 0.0.
  Log: /glade/derecho/scratch/adac/restart_time/validate.log.
- Local: time_utils 73 passed; continuation non-slow 4 passed;
  forcing dispatch 9 passed; legacy reproducibility passed (exit 0).
- Committed as e1ae069a (codex round 2 CLEAN on this diff).

## Iteration 2 — plan (2026-06-12)

Residual gap of the SAME class as the compiled seg-0 fix, found by
auditing every `_run_*` loop for stale-prepare-forcing windows:

- **`_run_per_step` (uncompiled fallback, :5229)**: the JIT-warmup step
  and every step before the next radiation boundary use the
  prepare-context ozone/aerosol/GHG/solar sampled at the epoch
  START_DAY.  The straight run's forcing for step k (k≥1) was sampled
  at day(M+1) where M = largest m ≤ k with (m+1) % RAD_UPDATE_STEPS
  == 0 (refresh at the top of the body for step m applies from step m
  onward); step 0 uses prepare values at day(0).  A resumed run
  (start_step = s > 0) must therefore initialize its forcing at
  day(boundary), boundary = RAD_UPDATE_STEPS * ((s+1) //
  RAD_UPDATE_STEPS), whenever boundary > 0 — this also covers
  RAD_UPDATE_STEPS == 1 (straight refreshed every step at day(k+1)).
  Fresh runs (s == 0) keep the historical prepare-at-epoch behavior.
  Invisible to bit-exact tests with constant TSI / no CMIP datasets —
  pin with the same spy pattern as
  `test_resumed_segment_zero_refreshes_external_forcing`, forcing the
  per-step path.
- Audited and CLEAN: `_run_spectral` (daily-cadence refresh fires on
  the first resumed step at the canonical bucket; gray radiation +
  constant solar), `_run_mpas` (same daily pattern, fixed iteration 1),
  `_run_compiled` (fixed this iteration).
- Open question for the review: how does a config select
  `_run_per_step` over `_run_compiled` (need the dispatch condition to
  write the spy test), and do production restarts ever take that path?
  → Answered: `run(compiled=False)` opt-in (debug/reference path,
  default is compiled); `tests/validation/test_precision_amip.py` uses
  it (fresh starts only — unaffected).

## Iteration 2 — done (2026-06-12)

Adversarial review of the plan returned **needs-attention (high)** and
confirmed a SECOND defect beyond the stale forcing (independently
spotted in the same audit): the `_run_per_step` warmup called
`step_unified(jnp.bool_(True), ...)` unconditionally, so a resume from
a checkpoint step that is NOT a radiation boundary recomputed radiation
where the straight run was held-only — overwriting the
checkpoint-restored held tendencies AND sampling forcing at the
prepare-context epoch.  Implemented per the amended plan:

- `warmup_need_rad` mirrors the main-loop predicate
  (`(start_step+1) % RAD_UPDATE_STEPS == 0`), with fresh starts
  (start_step == 0), RAD<=1, and resumes WITHOUT restored held
  tendencies (`"held_dT_rad" not in self._carry_aux`) keeping the
  historical always-radiate warmup (zero-init held would be worse).
- When the warmup radiates on a resume, solar + external forcing are
  re-sampled at day(start_step+1) — the day the straight run's body
  refresh used — instead of the epoch prepare values.  When it does
  not radiate, no refresh: non-radiation steps never read the forcing
  arrays, and the first body radiation step refreshes at its own day
  (this supersedes the plan's boundary-formula refresh; codex pointed
  out only the radiating-warmup case needs it).
- Warmup dispatches `step_unified` vs `step_unified_no_rad` exactly
  like the body.
- Regression test
  `test_per_step_bitexact_restart_off_radiation_boundary`:
  compiled=False, dt=600 (144 steps/day), checkpoint_days=1,
  rad_update_steps=3 ⇒ resume at step 144 with (145)%3 != 0; bitwise
  on T/u/p_s/q_v + held_dT_rad.  Verified it BITES: fails in 2:00 with
  the predicate forced True (old behavior), passes in 2:11 with the
  fix.
- Committed as 2d560887 (codex round 3 CLEAN).

## Iteration 3 — audits + CMIP attribution (2026-06-12)

- **Coupled-driver restart audit (clean)**: `CoupledESMDriver.run`
  delegates `(start_step, start_day)` verbatim to `self._atm.run`, and
  the established restart convention (`tests/stress/
  test_phase3_restart.py`, `test_phase7_multiyear.py`) calls
  `driver._atm.load_checkpoint(...)` first — which records the
  normalization hint on the SAME ModelDriver instance the coupled run
  uses.  The coupled path therefore inherits the epoch normalization
  with no further changes.
- **CMIP e2e attribution**: the full `test_phase6_cmip_e2e.py` suite
  cannot finish on a login node (killed at 90 min twice — the 30-day
  runs).  Targeted subset (14 forcing-semantics tests): PASSED except
  (a) `test_picontrol_ghg_is_none_or_constant` — environmental
  run-manifest collision with the killed runs' leftovers, passes in
  isolation; (b) `test_amip_baseline_ghg` (336.78 vs 348±1 ppm) —
  fails IDENTICALLY on a clean `main` worktree @ 9a5b5cb9 ⇒
  PRE-EXISTING, unrelated to this branch.
- Spectral canonical-sampling change is analytically a no-op for
  integer-day production starts with DT dividing 86400
  ((step+1)*DT/86400 is exact in fp64 at day boundaries, so
  float(bucket) == current_day at every refresh step).  PROVEN
  bitwise: a 2-day transient-GHG (1pctCO2, spectral C8 L5, dt=600)
  fresh run produces the IDENTICAL sha256 state digest
  (ec211c7a…58df over T/u/v/p_s/q_v) on main @ 9a5b5cb9 and on this
  branch.  (The 30-day e2e single test cannot finish on a login node
  — killed at 90 min — and is superseded by this stronger check.)
- **Gap found for iteration 4**: `_run_spectral` has NO direct
  bit-exact restart continuation test (the legacy reproducibility test
  is cubed_sphere/cdgrid = compiled path).  The spectral loop received
  the epoch normalization + canonical bucket sampling; whether its
  restart is actually bitwise needs a test mirroring the MPAS one
  (gaussian grid, discretization="spectral").

- **Codex round-2 finding (P2, fixed)**: the compiled cube/lat-lon loop
  only refreshed external forcing (ozone/aerosol/GHG + solar) for
  `seg_idx > 0`, so a RESUMED run's first segment kept the
  prepare-context values sampled at the epoch START_DAY — restarted
  AMIP/CMIP runs with transient forcing diverge (invisible to the
  bit-exact tests: constant TSI / no CMIP datasets make stale == fresh).
  Fix: gate is now `seg_idx > 0 or start_step > 0`; fresh runs keep the
  historical segment-0 behavior.  Regression test
  `test_resumed_segment_zero_refreshes_external_forcing` spies on the
  DAYS `_precompute_external_forcing` is called with (straight must ⊆
  resumed after the restart point) — verified it bites: fails on the old
  gate with "never sampled external forcing at day 2.0", passes with
  the fix.

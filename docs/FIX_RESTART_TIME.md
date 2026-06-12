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

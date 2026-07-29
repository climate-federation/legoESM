# Adversarial review ROUND 4: MPAS monthly-CMOR accumulator feed

You are an independent adversarial reviewer. Round 4. Rounds 1–3 validated the
FEED, the 2 m tas, the MPI gate, the CMIP-only clean-completion write, and the
PHASE1/PHASE2 structure. Round 3 left exactly two items; this round fixes both.
Re-review ONLY these two fixes + the one residual (inherited) note. Cite lines.

## Round-3 findings and disposition

**#1 (HIGH) — exact-checkpoint-cadence completion left a stale terminal
sidecar.** FIXED. When the last step is a checkpoint multiple, the in-loop
periodic checkpoint writes `cmor_accum_day_<final>.npz` BEFORE the finalizer
runs, and the after-loop "final checkpoint" is skipped — so `_suppress_cmor_
sidecar` (which only blocks a FUTURE write) could not remove it. The finalizer
now takes `final_day` and, on a successful write, EXPLICITLY UNLINKS the
terminal-day sidecar (`cmor_accum_day_<round(final_day)>.npz`) in addition to
setting the suppress flag. So both cases are covered: non-exact cadence →
suppress blocks the still-to-come final checkpoint's sidecar; exact cadence →
the already-written sidecar is retired. A restart from the final checkpoint then
finds no sidecar → resumes with empty accumulators (a COMPLETED run has no
in-progress month to carry) → no re-append. Test:
`test_driver_finalize_retires_exact_cadence_sidecar` (pre-creates the same-day
sidecar, finalizes, asserts it is gone + suppress set + NetCDF written).

**#2 (MEDIUM) — shape-correct but non-numeric `lat_deg` still committed
spatial before failing in the zonal `np.digitize`.** FIXED. PHASE 1 now coerces
`lat_deg` with `np.asarray(lat_deg, dtype=np.float64)` (a non-numeric object/
string vector raises HERE, before any commit) and additionally rejects
non-finite values. The spatial fields are shape-guaranteed by
`_regrid_to_latlon_*` (a non-numeric `T` would already raise inside the regrid,
still PHASE 1), and every zonal field is derived from the same `(nCells, …)`
arrays as the validated `lat` — so `np.add.at` lengths always match. Tests:
`test_feed_atomic_on_bad_lat_deg` (length mismatch),
`test_feed_atomic_on_nonnumeric_or_nonfinite_lat` (object vector + NaN).

**Residual (inherited) — the CMOR writer is not itself transactional.** A
mid-write I/O failure can leave a partially-written NetCDF (the shared `save()`
path has the identical property — `write_field` appends field-by-field). I
softened the finalizer's failure log (no "rerun the writer" claim) and
documented that a durable per-write progress record is the follow-up for full
crash-safety. On such a failure the sidecar is intentionally NOT retired. This
is pre-existing writer behaviour, not introduced by this change.

## Facts

- Sidecar filename: `cmor_accum_day_{int(round(day)):04d}.npz` in
  `self._output_dir` (matches `_save_cmor_accumulator_sidecar`).
- The finalizer runs on `run_status == "COMPLETED"` and `_mpas_cmip_feed_on`
  (serial / 1-rank), rank-0 only, BEFORE the final-checkpoint block.
- `final_day = START_DAY + n_steps_total * DT / 86400.0` (the same day the
  final/periodic checkpoint uses).

## THE ROUND-4 CHANGES

<INSERT v4 HERE>

## Test status

25 tests in `tests/unit/test_mpas_cmip_accumulator_feed.py` pass. Broader sweep
(collector, voronoi regrid, cmor accumulator restart, no-hardcoded-constants,
physics-contracts) = 3539 passed, 2 skipped. One PRE-EXISTING unrelated failure
(`test_cmor_experiments_restart.py::…test_wallclock_exit_writes_cmor_fx_before_exit`;
its fake lacks `_save_cmor_accumulator_sidecar`, which the byte-identical
committed `_maybe_wallclock_exit` calls — not touched by this change).

## Your task

Confirm: (a) does retiring the terminal-day sidecar (unlink) fully close the
exact-cadence re-append hazard, and is the filename/day computation exactly the
one the periodic checkpoint used? (b) is the numeric+finite `lat_deg` coercion
sufficient to make the commit transactional across all supported configs
(spatial-only, zonal-only, both)? (c) any NEW bug in the unlink path (e.g.,
wrong dir, deleting a needed non-terminal sidecar) or the tests? If both fixes
are correct and only the documented inherited writer non-atomicity remains, say
the review is CLEAN and there are no substantive blocking bugs.

# Adversarial review ROUND 3: MPAS monthly-CMOR accumulator feed

You are an independent adversarial reviewer. This is round 3. Rounds 1–2
already validated the FEED itself (tas 2 m, gate/warning, 1-rank permit, unit
conventions, shapes). Round 2 left exactly two items; this round addresses both.
Re-review ONLY whether these two fixes are correct and safe, and whether they
introduce any NEW bug. Cite line numbers.

## Round-2 findings and my disposition

**#1 (HIGH) — a normally completed MPAS run did not write the populated CMOR
accumulators to NetCDF.** FIXED. `_run_mpas` bypasses the shared
`_finalize_run` (whose `diagnostics.save()` ALSO writes the cube-path
`timeseries.npz` + snapshots this lean loop tracks separately in `_ts`, so
calling the full `save()` would clobber them). So I added `_finalize_mpas_cmip`,
which calls ONLY the CMIP-file writers (`_write_cmip_monthly_files` /
`_write_cmip_daily_files` / `_write_cmip_fixed_files` + `cf_writer.close()`) —
the exact CMIP subset of `save()`. It runs on a CLEAN completion (`run_status ==
"COMPLETED"`), rank-0 only, BEFORE the final checkpoint, and on success sets
`self._suppress_cmor_sidecar = True` so the terminal checkpoint writes no CMOR
sidecar (the writers do not pop; a run-extending restart would else re-append —
duplicate `time` coords). This mirrors `_finalize_run` exactly (which sets
`_suppress_cmor_sidecar` around its final checkpoint). Wallclock-graceful exits
already flush+POP completed months incrementally (`_maybe_wallclock_exit`), so
the final link only writes the un-popped remainder — no double-write.
Restart-chain reasoning: an intermediate link wallclock-exits (never reaches
this clean-completion path); only the FINAL/standalone link completes cleanly →
writes the remainder. Tests: `test_finalize_writes_cmor_netcdf` (emits non-empty
`cmor/*.nc` incl. `tas`), `test_driver_finalize_mpas_cmip` (writes + sets the
suppress flag), `test_driver_finalize_mpas_cmip_noop_without_writer`.

**#2 (MEDIUM) — the feed was not a true transaction.** FIXED. PHASE 1 was
already non-mutating for regrid/interp, but the zonal `add_*` (via
`np.digitize`/`np.add.at`) could raise on a malformed `lat_deg` AFTER the
spatial `add_*` had committed. I now validate `lat_deg.shape == (nCells,)` in
PHASE 1, before any commit. The spatial fields are shape-guaranteed by
`_regrid_to_latlon_*` (always `(nlat,nlon)`/`(nlat,nlon,nlev)` or `None`), so
`add_2d`/`add_3d` cannot raise on shape; with `lat_deg` pre-validated, PHASE 2
has no remaining commit-time failure mode. The driver-level `try/except` stays
loud-but-nonfatal (matches `_save_cmor_accumulator_sidecar`; a 100-yr run must
not abort on a diagnostic glitch). Test: `test_feed_atomic_on_bad_lat_deg`
(bad `lat_deg` -> `ValueError` in PHASE 1, all three accumulators untouched).

## Key facts for your reasoning

- `_write_cmip_monthly_files()` calls `self._spatial_monthly.finalize()`
  (default `min_sample_fraction=0.5`, drops partial months < 50% of max
  samples) then `_write_cmip_data`; it does NOT pop.
- `_maybe_wallclock_exit` (unchanged): `flush_cmip_monthly(day, write=True)`
  POPS completed months, `finalize_cmip_daily`, `finalize_cmip_fixed`,
  re-persists the drained sidecar, `sys.exit(0)`.
- `_finalize_run` (compiled path, unchanged): `diagnostics.save()` then sets
  `_suppress_cmor_sidecar=True` around the final checkpoint.
- `_save_cmor_accumulator_sidecar` early-returns when `_suppress_cmor_sidecar`.
- MPAS restart contract: each link runs `--days = remaining`; it either
  wallclock-exits (long chain) or completes cleanly (final/standalone link).
- The MPAS final checkpoint block runs after the loop:
  `if CHECKPOINT_INTERVAL>0 and COMPLETED and n_steps_total%INTERVAL!=0:
   save_checkpoint(...)`.

## THE INCREMENTAL CHANGES (round 3)

<INSERT v3 HERE>

## The updated tests (full file)

<INSERT tests HERE>

## Note on the test suite

One PRE-EXISTING unrelated test fails on this tree and on committed HEAD:
`test_cmor_experiments_restart.py::test_wallclock_exit_writes_cmor_fx_before_exit`
— its `SimpleNamespace` fake lacks `_save_cmor_accumulator_sidecar`, which the
COMMITTED `_maybe_wallclock_exit` calls (HEAD and working tree are byte-
identical for that method; my diff does not touch it). Not introduced here. All
21 tests in the new file pass; other CMOR/collector/regrid/constant suites pass.

## Your task

Re-review ONLY the two round-3 fixes. Specifically: (a) Is calling the CMIP-only
writers (not full `save()`) correct, and does `_suppress_cmor_sidecar` ordering
w.r.t. the final checkpoint actually prevent the terminal sidecar? (b) Any
double-write / partial-month / restart-chain hazard I mis-reasoned? (c) Is the
`lat_deg` PHASE-1 validation sufficient to make the commit transactional, or is
there another commit-time failure mode? (d) Any NEW bug in the added code or
tests? If both fixes are correct and no substantive bugs remain, say so
explicitly and state the review is clean.

# Adversarial review ROUND 5 (final): MPAS monthly-CMOR accumulator feed

You are an independent adversarial reviewer. Final round. Round 4 confirmed the
exact-cadence sidecar retirement is correct and left one MEDIUM: in ZONAL-ONLY
mode there is no spatial regrid to catch a malformed sibling field
(`precip`/`u_east`/`q_v`/`p_s`/`phis`), so it could partially commit. This round
fixes that with UP-FRONT shape validation of every input.

## Round-4 residual and disposition

**MEDIUM — zonal-only non-atomic for malformed non-lat fields.** FIXED. All
optional inputs are now materialized and SHAPE-VALIDATED up front (PHASE 1),
before any accumulator mutation, in EVERY mode: `p_s`/`precip`/`phis`/`tas` to
`(nCells,)` and `q_v`/`u_east`/`v_north` to `(nCells, nlev)`, with `nCells`/
`nlev` taken from the one required field `T`. `lat_deg` keeps its numeric+finite
+shape check. `psl` is computed only AFTER the shape check (so its broadcast is
safe). Consequences:

- Zonal-only: a wrong-length `precip`/`u_east`/`q_v` now raises in PHASE 1, so
  `MonthlyAccumulator.add_2d/3d` never partially commits.
- Spatial: unchanged (regrid already caught most, but the explicit check makes
  the error message precise and mode-independent).

Every zonal field placed in `z2d`/`z3d` is derived from a validated array, so
`np.add.at(zsum, bin_idx, vals)` always has matching lengths.

Test: `test_feed_atomic_zonal_only_malformed_siblings` (zonal-only collector;
wrong-length `precip` and `u_east` each raise pre-commit with the accumulator
untouched; a well-formed feed then succeeds). Plus the existing
`test_feed_atomic_on_bad_lat_deg` / `..._nonnumeric_or_nonfinite_lat`.

**Residual (inherited, documented, not blocking):** the underlying CMOR
`write_field` appends field-by-field and is not itself transactional (shared
with `save()`); a mid-write I/O failure can leave a partial NetCDF. Documented
with a follow-up; the finalizer does not retire the sidecar on such a failure.

## THE ROUND-5 CHANGE

<INSERT v5 HERE>

## Test status

24 tests in `tests/unit/test_mpas_cmip_accumulator_feed.py` pass. The one
PRE-EXISTING unrelated failure (`test_wallclock_exit_writes_cmor_fx_before_exit`,
fake lacks `_save_cmor_accumulator_sidecar`, in the byte-identical committed
`_maybe_wallclock_exit`) is not touched by this change.

## Your task

Confirm the up-front validation makes the feed transactional in ALL modes
(spatial-only, zonal-only, both) with no remaining commit-time failure path, and
that no NEW bug was introduced (e.g., the `tas` fallback ordering vs the shape
check, `nlev` derivation when `T` is 1-D, or a validated field still reaching an
accumulator with a mismatched length). If the only remaining item is the
documented inherited CMOR-writer non-atomicity, state that the review is CLEAN
with NO substantive blocking bugs.

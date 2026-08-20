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

### ROUND-5 fix: up-front shape validation of ALL inputs (transactional in every mode)
```python
        T_np = np.asarray(T)
        p_s_np = np.asarray(p_s)
        # Level -1 is the lowest (near-surface) model level (sigma_full is
        # ascending, ~1.0 at the surface — see _interp_to_plev19 / _tas_2m).
        T_low = T_np[..., -1]
        # tas is the 2 m MOST temperature when the caller supplies it (sst/sic
        # available), else the lowest-model-level fallback (applied AFTER the
        # shape check below) so the field is never dropped.
        tas_field = None if tas is None else np.asarray(tas)
        q_v_np = None if q_v is None else np.asarray(q_v)
        u_east_np = None if u_east is None else np.asarray(u_east)
        v_north_np = None if v_north is None else np.asarray(v_north)
        precip_np = None if precip is None else np.asarray(precip)
        phis_np = None if phis is None else np.asarray(phis)

        # Shape contract — validated UP FRONT so BOTH the spatial regrid AND the
        # zonal binning are transactional.  A malformed optional input raises
        # HERE (PHASE 1), before any accumulator is mutated: the ZONAL-only
        # config has no regrid phase to catch a wrong-length precip / cell wind
        # / q_v, so without this a bad sibling field would commit ``T_low`` then
        # raise inside ``np.add.at``.  ``ncol``/``nlev`` come from ``T`` (the
        # one required field).
        _ncol = int(T_np.shape[0])
        _nlev = int(T_np.shape[1]) if T_np.ndim == 2 else None
        _shape_checks: list[tuple[str, np.ndarray | None, tuple]] = [
            ("p_s", p_s_np, (_ncol,)),
            ("precip", precip_np, (_ncol,)),
            ("phis", phis_np, (_ncol,)),
            ("tas", tas_field, (_ncol,)),
        ]
        if _nlev is not None:
            _shape_checks += [
                ("q_v", q_v_np, (_ncol, _nlev)),
                ("u_east", u_east_np, (_ncol, _nlev)),
                ("v_north", v_north_np, (_ncol, _nlev)),
            ]
        for _nm, _arr, _shp in _shape_checks:
            if _arr is not None and _arr.shape != _shp:
                raise ValueError(
                    f"feed_cmip_accumulators_native: {_nm} shape {_arr.shape} "
                    f"!= expected {_shp}")

        # tas is the 2 m MOST temperature when supplied, else the lowest level.
        if tas_field is None:
            tas_field = T_low

        # Sea-level pressure (hypsometric) — shared by the spatial ``psl`` and
        # the zonal ``psl`` band; compute once when phis is available (shape
        # validated above, so the broadcast is safe).
        psl = None
        if phis_np is not None:
            T_low_safe = np.maximum(T_low, _c.T_min_atmosphere)
            psl = p_s_np * np.exp(phis_np / (_c.R_d * T_low_safe))
```
### zonal lat validation (unchanged logic, reuses up-front _ncol)
```python
        z2d: dict[str, np.ndarray] = {}
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            # Validate the zonal lat vector HERE, in PHASE 1, before any commit
            # (the sibling zonal fields were shape-checked up front).  Force a
            # numeric float array so a shape-correct-but-non-numeric (object/
            # string) lat_deg raises here, not inside ``np.digitize`` mid-
            # commit; also require finite values.
            lat_np = np.asarray(lat_deg, dtype=np.float64)
            if lat_np.shape != (_ncol,):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg shape "
                    f"{lat_np.shape} != expected ({_ncol},)")
            if not np.all(np.isfinite(lat_np)):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg has non-finite "
                    "values")
            z2d['T_low'] = np.asarray(T_low)   # lowest model level, NOT 2 m tas
            if precip_np is not None:
                z2d['precip'] = precip_np * 86400.0     # [mm/day], like collect()
            if psl is not None:
                z2d['psl'] = np.asarray(psl)
            z3d['T'] = T_np
            if u_east_np is not None:
                z3d['u'] = u_east_np
            if q_v_np is not None:
```


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


## APPENDIX: round-4 review (verbatim)

Verdict: **not CLEAN** — the sidecar fix is correct, but the lat fix does not make zonal-only feeds transactional.

- **(a) Exact-cadence sidecar:** correct. The checkpoint and finalizer use the same directory and exact filename normalization: `self._output_dir` plus `int(round(day))` in the checkpoint helper ([model_driver.py:4456](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:4456)); `Path(_output_dir)` plus `int(round(final_day))` in the finalizer ([model_driver.py:5440](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:5440)). The periodic and final computations are also identical at the terminal step ([model_driver.py:6444](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6444), [model_driver.py:6460](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/model_driver.py:6460)). Thus the unlink closes the exact-cadence re-append hazard and does not target a different directory or a non-terminal filename.

- **(b) Remaining MEDIUM: zonal-only is still non-atomic for malformed non-lat fields.** The new float/finite `lat_deg` check is correct ([diagnostics.py:1471](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1471)), but Phase 1 does not validate the shapes/types of `precip`, `psl`, `u_east`, or `q_v` before placing them in `z2d`/`z3d` ([diagnostics.py:1481](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/coupler/legoesm/driver/diagnostics.py:1481)). In zonal-only mode there is no spatial regrid phase to catch those inputs.

  For example, valid `T`/`lat_deg` with a wrong-length optional `precip` reaches `MonthlyAccumulator.add_2d`. That function increments the call count first ([monthly_means.py:120](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:120)), commits `T_low`, then raises when `np.add.at` receives latitude indices and precipitation values of different lengths ([monthly_means.py:132](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/tools/legoesm/diagnostics/monthly_means.py:132)). Bad `phis`/`p_s` broadcasting and malformed 3-D optional fields have analogous paths.

  The new tests cover only bad latitude and use the helper’s default `cmip_output=True`, so they do not exercise zonal-only or malformed sibling fields ([test_mpas_cmip_accumulator_feed.py:45](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_cmip_accumulator_feed.py:45), [test_mpas_cmip_accumulator_feed.py:530](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/unit/test_mpas_cmip_accumulator_feed.py:530)).

  Fix: validate/coerce every `z2d` field to `(nCells,)` and every `z3d` field to `(nCells, nlev)` in Phase 1 (including `p_s`/`phis` before forming `psl`), then add a zonal-only atomicity regression with malformed `precip` and malformed 3-D input.

- **(c) Unlink path:** no normal-path wrong-directory or wrong-day deletion bug found. The helper-level exact-cadence test correctly verifies deletion. The documented non-transactional CMOR writer residual remains.

### ROUND-4 incremental fixes (addressing round-3 HIGH + MEDIUM)

#### diagnostics.py :: PHASE-1 lat_deg numeric+finite validation (1458-1483)
```python
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            # Validate the zonal lat vector HERE, in PHASE 1, before any commit:
            # the zonal add_* bin via ``np.digitize(lat, ...)`` / ``np.add.at``
            # and would raise mid-commit — AFTER the spatial add_* already
            # mutated their buckets.  Catching it pre-commit keeps the whole
            # feed transactional (the spatial fields are shape-guaranteed by
            # ``_regrid_to_latlon_*`` — a non-numeric one would already raise in
            # the regrid, still PHASE 1 — so lat_deg is the only remaining
            # commit-time failure mode).  Force a numeric float array so a
            # shape-correct-but-non-numeric (object/string) lat_deg raises here,
            # not inside ``np.digitize`` mid-commit; also require finite values.
            lat_np = np.asarray(lat_deg, dtype=np.float64)
            _ncol = int(T_np.shape[0])
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
```
#### model_driver.py :: _finalize_mpas_cmip (now retires the terminal sidecar; 5390-5451)
```python
    def _finalize_mpas_cmip(self, final_day: float | None = None) -> None:
        """Write the CMOR NetCDF (``Amon`` / ``day`` / ``fx``) from the fed
        accumulators at a CLEAN MPAS completion.

        ``_run_mpas`` bypasses the shared :meth:`_finalize_run` (whose
        ``diagnostics.save()`` ALSO writes the cube/lat-lon ``timeseries.npz`` +
        snapshots this lean path tracks SEPARATELY in ``_ts`` — calling it here
        would clobber the lightweight timeseries).  So mirror ONLY the
        CMIP-file finalize of ``save()``.  Wallclock-graceful exits already
        flush COMPLETED months incrementally and POP them
        (:meth:`_maybe_wallclock_exit`); this covers the final / standalone
        clean completion, writing whatever remains (in-progress + not-yet-
        flushed months, subject to the finalize partial-month guard) — which
        otherwise left the now-fed accumulators unwritten (empty ``cmor/``).

        The writers do NOT pop, so this is TERMINAL.  On success it (a) sets
        ``_suppress_cmor_sidecar`` so a still-to-be-written final checkpoint
        skips its CMOR sidecar, AND (b) RETIRES (unlinks) the terminal-day
        sidecar if one was ALREADY written this step — the exact-checkpoint-
        cadence case, where the in-loop periodic checkpoint wrote the sidecar
        BEFORE this finalizer runs and the after-loop "final checkpoint" is
        skipped, so suppression alone (which only blocks a FUTURE write) would
        leave a stale same-day sidecar that a run-EXTENDING restart would
        restore and re-append (duplicate ``time`` coords).  Mirrors the intent
        of :meth:`_finalize_run`.  Rank-0 only; fully guarded (a writer failure
        must not turn a COMPLETED run into a crash after the science is done —
        the accumulators/sidecar are left intact for inspection).

        NOTE: the underlying CMOR writer appends field-by-field and is not
        itself transactional, so a mid-write I/O failure can leave a partially
        written NetCDF (inherited from the shared ``save()`` path); on such a
        failure the sidecar is intentionally NOT retired, but a blind retry
        could still duplicate the already-appended fields — a durable
        per-write progress record is the follow-up for full crash-safety."""
        diag = getattr(self, "diagnostics", None)
        if diag is None or getattr(diag, "cf_writer", None) is None:
            return
        _is_root = (getattr(self, "_mpi_rank", None) is None
                    or self._mpi_rank == 0)
        if not _is_root:
            return
        try:
            diag._write_cmip_monthly_files()
            diag._write_cmip_daily_files()
            diag._write_cmip_fixed_files()
            diag.cf_writer.close()
            # Terminal: block any future sidecar write AND retire a same-day
            # sidecar already written by the in-loop periodic checkpoint.
            self._suppress_cmor_sidecar = True
            if final_day is not None:
                _out = getattr(self, "_output_dir", None)
                if _out is not None:
                    _sc = Path(_out) / (
                        f"cmor_accum_day_{int(round(final_day)):04d}.npz")
                    if _sc.exists():
                        _sc.unlink()
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.error(
                "  MPAS CMOR NetCDF finalize FAILED (accumulators/sidecar left "
                "intact for inspection; the NetCDF may be partially written — "
                "the shared CMOR writer is not transactional): %s", exc)

```
#### model_driver.py :: caller passes final_day (6450-6462)
```python
        # Write the CMOR NetCDF from the (now-fed) accumulators on a CLEAN
        # completion.  MUST run BEFORE the final checkpoint below so its
        # ``_suppress_cmor_sidecar`` (set on a successful write) reaches a
        # still-to-be-written final checkpoint; ``final_day`` also lets it
        # RETIRE a same-day sidecar already written by the last in-loop
        # periodic checkpoint (exact-checkpoint-cadence completion).  Gated on
        # the feed being active (serial / 1-rank with CMIP output); a no-op
        # otherwise.
        if run_status == "COMPLETED" and self._mpas_cmip_feed_on:
            self._finalize_mpas_cmip(
                START_DAY + n_steps_total * DT / 86400.0)

        # Final checkpoint so the next chain link resumes from the exact end
```

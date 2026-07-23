### ROUND-3 incremental changes (on top of round-2, which you already blessed)

#### diagnostics.py :: PHASE-1 lat_deg validation (atomicity, 1455-1476)
```python
                        daily_2d[_dst] = fields_3d[_src_name][:, :, _idx850]

        z2d: dict[str, np.ndarray] = {}
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            lat_np = np.asarray(lat_deg)
            # Validate the zonal lat vector HERE, in PHASE 1, before any commit:
            # the zonal add_* bin via ``np.digitize(lat, ...)`` and ``np.add.at``
            # and would raise mid-commit on a length mismatch — AFTER the spatial
            # add_* already mutated their buckets.  Catching it pre-commit keeps
            # the whole feed transactional (the spatial fields are shape-
            # guaranteed by ``_regrid_to_latlon_*``, so this is the only
            # remaining commit-time failure mode).
            _ncol = int(T_np.shape[0])
            if lat_np.shape != (_ncol,):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg shape "
                    f"{lat_np.shape} != expected ({_ncol},)")
            z2d['T_low'] = np.asarray(T_low)   # lowest model level, NOT 2 m tas
            if precip_np is not None:
                z2d['precip'] = precip_np * 86400.0     # [mm/day], like collect()
```
#### model_driver.py :: _finalize_mpas_cmip (5390-5431)
```python
    def _finalize_mpas_cmip(self) -> None:
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

        The writers do NOT pop, so this is TERMINAL: on success it sets
        ``_suppress_cmor_sidecar`` so the final checkpoint carries no CMOR
        sidecar (a completed run has no in-progress month to resume, and a
        run-EXTENDING restart would otherwise re-append the already-written
        months — duplicate ``time`` coords).  Mirrors :meth:`_finalize_run`.
        Rank-0 only; fully guarded (a writer failure must not turn a COMPLETED
        run into a crash after the science is done — the sidecar/accumulators
        remain intact for a manual rewrite)."""
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
            self._suppress_cmor_sidecar = True
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.error(
                "  MPAS CMOR NetCDF finalize FAILED (the accumulators were fed "
                "and the sidecar is intact — rerun the writer or resume): %s",
                exc)

```
#### model_driver.py :: clean-completion finalize call in _run_mpas (6429-6446)
```python

        # Write the CMOR NetCDF from the (now-fed) accumulators on a CLEAN
        # completion.  MUST run BEFORE the final checkpoint below so its
        # ``_suppress_cmor_sidecar`` (set on a successful write) reaches the
        # terminal sidecar.  Gated on the feed being active (serial / 1-rank
        # with CMIP output); a no-op otherwise.
        if run_status == "COMPLETED" and self._mpas_cmip_feed_on:
            self._finalize_mpas_cmip()

        # Final checkpoint so the next chain link resumes from the exact end
        # state.  Skipped (a) on blow-up — state is non-finite — and (b) when
        # the last loop step already hit the periodic cadence, which would
        # re-write the identical file (wasted device→host transfer + I/O
        # every whole-multiple job boundary).
        if (CHECKPOINT_INTERVAL > 0 and run_status == "COMPLETED"
                and n_steps_total % CHECKPOINT_INTERVAL != 0):
            _final_day = START_DAY + n_steps_total * DT / 86400.0
            self.save_checkpoint(start_step + n_steps_total, _final_day)
```

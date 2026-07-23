### FILE: packages/coupler/legoesm/driver/diagnostics.py — NEW method feed_cmip_accumulators_native (lines 1259-1449)
```python
    def feed_cmip_accumulators_native(
        self,
        day: float,
        *,
        T,
        p_s,
        lat_deg=None,
        q_v=None,
        u_east=None,
        v_north=None,
        precip=None,
        phis=None,
    ) -> bool:
        """Feed the CMIP spatial (``Amon``/``day``) + zonal-mean monthly
        accumulators from NATIVE-grid host arrays, bypassing the heavy
        :meth:`collect` path.

        This is the lean-driver counterpart to :meth:`collect`'s spatial
        block.  The MPAS (Voronoi) execution path materialises only a handful
        of diagnostic arrays per interval and never assembles :meth:`collect`'s
        full kwarg set (radiation fluxes, sst/sic, cloud tracers), so it
        historically fed NOTHING into the CMOR accumulators (manifests showed
        ``call_counts: []`` / ``max_count_ever: 0`` even though the sidecar
        SAVE ran).  Rather than fabricate the unavailable inputs, this method
        feeds the subset the lean path CAN provide through the SAME regrid +
        plev19 interpolation helpers :meth:`collect` uses, so the published
        fields share one pipeline:

        * 2-D (``add_2d``): ``tas`` (lowest-level T as a near-surface proxy),
          ``ps``, ``pr`` (when *precip* given), ``prw`` (column water vapour,
          when *q_v* given), ``psl`` (hypsometric, when *phis* given).
        * 3-D on plev19 (``add_3d``): ``ta``, ``hus`` (*q_v*), ``ua``
          (*u_east*), ``va`` (*v_north*).
        * Daily (``SpatialDailyAccumulator``): ``tas``/``pr``/``psl`` plus
          ``ua850``/``va850`` sliced from the regridded 3-D winds.
        * Zonal (``MonthlyAccumulator``): ``T_low``/``precip``/``psl`` 2-D
          bands and ``T``/``u``/``q_v`` profiles (needs *lat_deg*).

        Parameters
        ----------
        day : float
            ABSOLUTE simulated day (``start_day + elapsed``).  The month/year
            bucket keys are ``doy, _ = day_to_calendar(day)`` and
            ``year = int(day // 365)`` — identical to :meth:`collect`, so a
            restart chain keeps monotonic calendar months.
        T : array, shape ``(nCells, nlev)``
            Air temperature [K] on model levels (level ``-1`` = lowest/surface).
        p_s : array, shape ``(nCells,)``
            Surface pressure [Pa].
        lat_deg : array, shape ``(nCells,)``, optional
            Cell-centre latitude [degrees] for the zonal-mean bands.  When
            ``None`` the zonal accumulator is skipped.
        q_v : array, shape ``(nCells, nlev)``, optional
            Water-vapour mixing ratio [kg/kg].
        u_east, v_north : array, shape ``(nCells, nlev)``, optional
            GEOGRAPHIC cell-centre zonal / meridional wind [m/s] (MPAS: from
            ``reconstruct_cell_velocity`` on the edge-normal ``state.u``).
        precip : array, shape ``(nCells,)``, optional
            Surface precipitation rate [kg/m2/s] (CMOR ``pr`` units).
        phis : array, shape ``(nCells,)``, optional
            Surface geopotential [m2/s2] for the sea-level-pressure reduction.

        Returns
        -------
        bool
            ``True`` if any accumulator was fed, ``False`` (no-op) when the
            collector has neither the spatial CMIP accumulators nor a zonal
            monthly accumulator with *lat_deg* supplied.
        """
        have_spatial = self._spatial_monthly is not None
        have_zonal = (
            self.monthly_means
            and self.monthly_accum is not None
            and lat_deg is not None
        )
        if not (have_spatial or have_zonal):
            return False

        from legoesm import constants as _c

        doy, _ = day_to_calendar(day)
        year = int(day // 365.0)

        T_np = np.asarray(T)
        p_s_np = np.asarray(p_s)
        # Level -1 is the lowest (near-surface) model level (sigma_full is
        # ascending, ~1.0 at the surface — see _interp_to_plev19 / _tas_2m).
        # tas uses the lowest-level T as a near-surface proxy: the MOST 2 m
        # similarity refinement collect() applies needs sst/sic + the surface
        # layer scheme, which the lean MPAS path does not thread here (TODO:
        # promote to the 2 m value once sst/sic reach this call site).
        T_low = T_np[..., -1]

        # Sea-level pressure (hypsometric) — shared by the spatial ``psl`` and
        # the zonal ``psl`` band; compute once when phis is available.
        psl = None
        if phis is not None:
            phis_np = np.asarray(phis)
            T_low_safe = np.maximum(T_low, _c.T_min_atmosphere)
            psl = p_s_np * np.exp(phis_np / (_c.R_d * T_low_safe))

        fed = False

        # ---- Spatial (regridded lat-lon) accumulation -------------------
        if have_spatial:
            fields_2d: dict[str, np.ndarray] = {}
            r = self._regrid_to_latlon_2d(T_low)
            if r is not None:
                fields_2d['tas'] = r
            r = self._regrid_to_latlon_2d(p_s_np)
            if r is not None:
                fields_2d['ps'] = r
            if precip is not None:
                # CMOR ``pr`` is [kg/m2/s] — native units, no conversion
                # (matches collect()'s regrid of precip_total).
                r = self._regrid_to_latlon_2d(np.asarray(precip))
                if r is not None:
                    fields_2d['pr'] = r
            if q_v is not None:
                cwv_field = np.asarray(
                    column_water_vapor(q_v, p_s, self.dsigma))
                r = self._regrid_to_latlon_2d(cwv_field)
                if r is not None:
                    fields_2d['prw'] = r
            if psl is not None:
                r = self._regrid_to_latlon_2d(psl)
                if r is not None:
                    fields_2d['psl'] = r
            if fields_2d:
                self._spatial_monthly.add_2d(doy, year, fields_2d)
                fed = True

            # 3-D fields: model levels → plev19, then regrid.  The plev
            # interpolation is column-wise and works unchanged on native
            # (nCells, nlev) with p_s (nCells,).
            fields_3d: dict[str, np.ndarray] = {}
            for _name, _src in (
                ('ta', T_np),
                ('hus', None if q_v is None else np.asarray(q_v)),
                ('ua', None if u_east is None else np.asarray(u_east)),
                ('va', None if v_north is None else np.asarray(v_north)),
            ):
                if _src is None:
                    continue
                _plev = self._interp_to_plev19(_src, p_s_np)
                if _plev is None:
                    continue
                r = self._regrid_to_latlon_3d(_plev)
                if r is not None:
                    fields_3d[_name] = r
            if fields_3d:
                self._spatial_monthly.add_3d(doy, year, fields_3d)
                fed = True

            # Daily accumulation: reuse the regridded 2-D fields + 850 hPa
            # winds sliced from the 3-D arrays (same index convention as
            # collect()).
            if self._spatial_daily is not None:
                daily_2d: dict[str, np.ndarray] = {}
                for _name in ("tas", "pr", "psl"):
                    if _name in fields_2d:
                        daily_2d[_name] = fields_2d[_name]
                _plev_sorted = np.sort(CMIP6_PLEV19)
                _idx850 = int(np.argmin(np.abs(_plev_sorted - 85000.0)))
                for _src_name, _dst in (("ua", "ua850"), ("va", "va850")):
                    if (_src_name in fields_3d
                            and fields_3d[_src_name].shape[2] > _idx850):
                        daily_2d[_dst] = fields_3d[_src_name][:, :, _idx850]
                if daily_2d:
                    self._spatial_daily.add_2d(doy, year, daily_2d)
                    fed = True

        # ---- Zonal-mean accumulation (native lat binning) ---------------
        if have_zonal:
            lat_np = np.asarray(lat_deg)
            z2d: dict[str, np.ndarray] = {'T_low': np.asarray(T_low)}
            if precip is not None:
                # Zonal ``precip`` is [mm/day] (matches collect()).
                z2d['precip'] = np.asarray(precip) * 86400.0
            if psl is not None:
                z2d['psl'] = np.asarray(psl)
            self.monthly_accum.add_2d(doy, year, z2d, lat_np)
            z3d: dict[str, np.ndarray] = {'T': T_np}
            if u_east is not None:
                z3d['u'] = np.asarray(u_east)
            if q_v is not None:
                z3d['q_v'] = np.asarray(q_v) * 1000.0  # g/kg
            self.monthly_accum.add_3d(doy, year, z3d, lat_np)
            fed = True

        return fed
```

### FILE: packages/coupler/legoesm/driver/model_driver.py — my 4 additions
#### (a) __init__ flag init (line ~333-335)
```python
        # Gate for the MPAS CMOR spatial/zonal accumulator feed (set per run
        # in _run_mpas; False here so any other path is a safe no-op).
        self._mpas_cmip_feed_on = False
        self._grid_global = None  # global grid preserved under band/cell MPI
```
#### (b) helper _feed_mpas_cmip_accumulators (lines 5270-5337)
```python
    def _feed_mpas_cmip_accumulators(self, day: float) -> None:
        """Feed the CMOR monthly/daily/zonal accumulators from the current
        MPAS (Voronoi) state at a diagnostic interval.

        The lean MPAS loop never called :meth:`DiagnosticCollector.collect`
        (the cube/lat-lon spatial-accumulation path), so the CMOR ``Amon`` /
        ``day`` accumulators the collector constructs stayed EMPTY
        (``call_counts: []`` / ``max_count_ever: 0``) even though the restart
        sidecar dutifully SAVED them each checkpoint.  This bridges that gap:
        it reconstructs geographic cell winds (edge-normal ``state.u`` -> cell
        ``(u_east, v_north)`` via the Perot reconstruction), reads the
        surface-precip export the physics stashes on the dynamics object, and
        hands the native cell arrays to
        :meth:`DiagnosticCollector.feed_cmip_accumulators_native`, which
        regrids to the CMOR lat-lon grid (the collector's Voronoi IDW weights)
        and bins the zonal means.

        SERIAL only — the caller gates on ``self._voronoi_layout is None``.  An
        MPI cell partition holds only this rank's LOCAL cells and the collector
        holds LOCAL regrid weights, so a per-rank feed would bin one rank's
        cells into the global lat-lon boxes (a rank-local, wrong "global"
        monthly mean).  A correct MPI feed needs an owned-cell -> global gather
        plus global regrid weights on rank 0; tracked as a follow-up.  The
        lightweight timeseries is already a true global via
        :meth:`_mpas_global_diag`.

        Fully guarded: a diagnostic-feed failure is LOUD but never aborts the
        run (the host-side accumulation cannot perturb the prognostic state).
        """
        diag = getattr(self, "diagnostics", None)
        if diag is None:
            return
        try:
            from legoesm.grids.voronoi import reconstruct_cell_velocity
            state = self.state
            # Geographic cell-centre winds from the edge-normal velocity.
            u_east, v_north = reconstruct_cell_velocity(
                state.u.data, self.grid)
            # Water vapour (moist runs only).
            q_v = None
            if (state.tracers is not None and "q_v" in state.tracers):
                q_v = state.tracers["q_v"].data
            # Surface precip [kg/m2/s]: the physics stashes the tuple
            # (sw_net_sfc, lw_net_sfc, precip) on the dynamics object every
            # step (commit d4fbca3ec); the third slot is None on a dry run.
            precip = None
            _sfc_diag = getattr(self.model, "_sfc_diag", None)
            if (_sfc_diag is not None and len(_sfc_diag) > 2
                    and _sfc_diag[2] is not None):
                precip = _sfc_diag[2].data
            lat_deg = np.degrees(np.asarray(self.grid.latCell))
            diag.feed_cmip_accumulators_native(
                day,
                T=state.T.data,
                p_s=state.p_s.data,
                lat_deg=lat_deg,
                q_v=q_v,
                u_east=u_east,
                v_north=v_north,
                precip=precip,
                phis=state.phis.data,
            )
        except Exception as exc:  # pragma: no cover - defensive diag guard
            logger.error(
                "  CMOR accumulator feed FAILED at day %.2f (run continues; "
                "the monthly/daily CMOR means for this interval are lost): %s",
                day, exc)

```
#### (c) feed-enable flag in _run_mpas (lines 5396-5413)
```python

        # Whether to feed the CMOR spatial/zonal accumulators at diag cadence.
        # The lean MPAS loop historically fed nothing into them (empty CMOR
        # ``Amon``/``day`` output despite the sidecar SAVE running).  Serial
        # only: an MPI voronoi feed needs an owned-cell -> global gather +
        # global regrid weights (see _feed_mpas_cmip_accumulators).  Computed
        # once — the collector's accumulator handles are stable for the run.
        _diag = getattr(self, "diagnostics", None)
        self._mpas_cmip_feed_on = (
            _diag is not None
            and self._voronoi_layout is None
            and (
                getattr(_diag, "_spatial_monthly", None) is not None
                or (getattr(_diag, "monthly_means", False)
                    and getattr(_diag, "monthly_accum", None) is not None)
            )
        )

```
#### (d) feed call in the daily diag block (lines 6284-6294)
```python
                        START_DAY + (step + 1) * DT / 86400.0)
                    break

                # Feed the CMOR monthly/daily/zonal accumulators.  Placed
                # AFTER the finiteness/bounds guard so a blown-up state never
                # pollutes the accumulators (the guard ``break``s first).  The
                # day is ABSOLUTE (START_DAY + elapsed) so multi-link restart
                # chains keep monotonic calendar months (matches the cube
                # path).  Feed cost is host-side numpy at diag cadence only.
                if self._mpas_cmip_feed_on:
                    self._feed_mpas_cmip_accumulators(START_DAY + elapsed_day)
```

### diagnostics.py :: _tas_2m (611-645, refactored to accept u_low/v_low)
```python
    def _tas_2m(self, state, q_v, sst, sic, T_ice, u_low=None, v_low=None):
        """2 m air temperature for CMIP ``tas`` from the MOST surface-layer
        similarity profile (interpolate the lowest model level down to 2 m).
        Returns the lowest-level T when the surface inputs (SST / sigma) are
        unavailable — e.g. a prescribed-SST run that does not pass SST here.

        ``u_low`` / ``v_low`` override the lowest-level winds (lets the MPAS
        path pass the reconstructed CELL winds from
        ``reconstruct_cell_velocity`` instead of the edge-normal ``state.u``,
        which is not cell-collocated); default reads them off ``state``."""
        T_low = state.T.data[..., -1]
        if sst is None or self.sigma_full is None:
            return T_low
        import jax.numpy as jnp
        from legoesm import constants
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.core.bulk_flux import compute_most_fluxes
        u_low = state.u.data[..., -1] if u_low is None else u_low
        v_low = state.v.data[..., -1] if v_low is None else v_low
        q_low = q_v[..., -1] if q_v is not None else jnp.zeros_like(T_low)
        p_s = state.p_s.data
        p_low = p_s * jnp.asarray(self.sigma_full)[-1]
        rho_low = p_low / (constants.R_d * T_low)
        T_sfc = blend_surface_temperature(sst, sic, T_ice)
        q_sfc = saturation_mixing_ratio(T_sfc, p_s)
        # coare3 similarity profile (the recommended config's scheme); the 2 m
        # value is set by stability, so gustiness is irrelevant here.
        *_, T_2m = compute_most_fluxes(
            u_low, v_low, T_low, q_low, T_sfc, q_sfc, rho_low,
            scheme="coare3", return_2m=True,
        )
        return T_2m

    def collect(
        self,
```
### diagnostics.py :: feed_cmip_accumulators_native (1264-1491, atomic + tas param)
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
        tas=None,
    ) -> bool:
        """Feed the CMIP spatial (``Amon``/``day``) + zonal-mean monthly
        accumulators from NATIVE-grid host arrays, bypassing the heavy
        :meth:`collect` path.

        This is the lean-driver counterpart to :meth:`collect`'s spatial
        block.  The MPAS (Voronoi) execution path materialises only a handful
        of diagnostic arrays per interval and never assembles :meth:`collect`'s
        full kwarg set (radiation fluxes, cloud tracers), so it historically
        fed NOTHING into the CMOR accumulators (manifests showed
        ``call_counts: []`` / ``max_count_ever: 0`` even though the sidecar
        SAVE ran).  This feeds the subset the lean path CAN provide through the
        SAME regrid + plev19 interpolation helpers :meth:`collect` uses, so the
        published fields share one pipeline:

        * 2-D (``add_2d``): ``tas`` (2 m air temperature — MOST similarity when
          the caller supplies *tas*, else the lowest-model-level fallback),
          ``ps``, ``pr`` (when *precip* given), ``prw`` (column water vapour,
          when *q_v* given), ``psl`` (hypsometric, when *phis* given).
        * 3-D on plev19 (``add_3d``): ``ta``, ``hus`` (*q_v*), ``ua``
          (*u_east*), ``va`` (*v_north*).
        * Daily (``SpatialDailyAccumulator``): ``tas``/``pr``/``psl`` plus
          ``ua850``/``va850`` sliced from the regridded 3-D winds.
        * Zonal (``MonthlyAccumulator``): ``T_low``/``precip``/``psl`` 2-D
          bands and ``T``/``u``/``q_v`` profiles (needs *lat_deg*).

        Sampling / accuracy caveats (documented, not silently hidden):

        * The lean MPAS loop has no per-interval time integrator, so each fed
          value is the INSTANTANEOUS end-of-interval sample, not an interval
          mean.  At the common ``diag_days=1`` cadence the monthly means are an
          average of one snapshot per day (diurnally aliased at the fixed
          diagnostic phase — the same alias :meth:`collect`'s ``t_low_mean``
          mitigation targets), the daily-table extremes ``tasmin``/``tasmax``
          collapse to that single sample, and ``pr`` is the last physics-step
          rate rather than an accumulated flux.  Sub-daily ``diag_days`` give
          proper multi-sample means/extrema; a device-side interval accumulator
          is the follow-up for bias-free daily/monthly ``pr``.
        * ``ua850``/``va850`` come from :meth:`_interp_to_plev19`, which BOUNDED
          -extrapolates below the lowest model level (inherited shared-helper
          behaviour, identical to :meth:`collect`; not masked to NaN).
        * The zonal ``MonthlyAccumulator`` arithmetic-averages cells per lat
          band (inherited); exact only for equal-area cells — a variable-
          resolution MPAS mesh biases the zonal mean.  The SPATIAL (regridded)
          CMOR path is unaffected (area-neutral IDW to a regular grid).

        Commit is ATOMIC: every regrid / plev interpolation (the failure-prone
        work) runs BEFORE the first ``add_*``, so a mid-computation error leaves
        the accumulators untouched rather than half-updated.

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
        tas : array, shape ``(nCells,)``, optional
            2 m air temperature [K] (MOST similarity, computed by the caller
            from sst/sic + surface-layer winds).  Falls back to the lowest
            model level when ``None`` so the field is never dropped.

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
        T_low = T_np[..., -1]
        # tas is the 2 m MOST temperature when the caller supplies it (sst/sic
        # available), else the lowest-model-level fallback so the field is not
        # dropped entirely.
        tas_field = T_low if tas is None else np.asarray(tas)
        q_v_np = None if q_v is None else np.asarray(q_v)
        u_east_np = None if u_east is None else np.asarray(u_east)
        v_north_np = None if v_north is None else np.asarray(v_north)
        precip_np = None if precip is None else np.asarray(precip)

        # Sea-level pressure (hypsometric) — shared by the spatial ``psl`` and
        # the zonal ``psl`` band; compute once when phis is available.
        psl = None
        if phis is not None:
            phis_np = np.asarray(phis)
            T_low_safe = np.maximum(T_low, _c.T_min_atmosphere)
            psl = p_s_np * np.exp(phis_np / (_c.R_d * T_low_safe))

        # ================================================================
        # PHASE 1 — compute everything (regrid / plev interp) up front.
        # No accumulator is mutated here, so a failure commits nothing.
        # ================================================================
        fields_2d: dict[str, np.ndarray] = {}
        fields_3d: dict[str, np.ndarray] = {}
        daily_2d: dict[str, np.ndarray] = {}
        if have_spatial:
            for _name, _src in (
                ('tas', tas_field),
                ('ps', p_s_np),
                ('pr', precip_np),   # CMOR kg/m2/s — native, no conversion
                ('psl', psl),
            ):
                if _src is None:
                    continue
                r = self._regrid_to_latlon_2d(_src)
                if r is not None:
                    fields_2d[_name] = r
            if q_v_np is not None:
                cwv_field = np.asarray(
                    column_water_vapor(q_v_np, p_s_np, self.dsigma))
                r = self._regrid_to_latlon_2d(cwv_field)
                if r is not None:
                    fields_2d['prw'] = r

            # 3-D fields: model levels → plev19, then regrid.  The plev
            # interpolation is column-wise and works unchanged on native
            # (nCells, nlev) with p_s (nCells,).
            for _name, _src in (
                ('ta', T_np),
                ('hus', q_v_np),
                ('ua', u_east_np),
                ('va', v_north_np),
            ):
                if _src is None:
                    continue
                _plev = self._interp_to_plev19(_src, p_s_np)
                if _plev is None:
                    continue
                r = self._regrid_to_latlon_3d(_plev)
                if r is not None:
                    fields_3d[_name] = r

            # Daily: reuse the regridded 2-D fields + 850 hPa winds sliced from
            # the 3-D arrays (same index convention as collect()).
            if self._spatial_daily is not None:
                for _name in ("tas", "pr", "psl"):
                    if _name in fields_2d:
                        daily_2d[_name] = fields_2d[_name]
                _plev_sorted = np.sort(CMIP6_PLEV19)
                _idx850 = int(np.argmin(np.abs(_plev_sorted - 85000.0)))
                for _src_name, _dst in (("ua", "ua850"), ("va", "va850")):
                    if (_src_name in fields_3d
                            and fields_3d[_src_name].shape[2] > _idx850):
                        daily_2d[_dst] = fields_3d[_src_name][:, :, _idx850]

        z2d: dict[str, np.ndarray] = {}
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            lat_np = np.asarray(lat_deg)
            z2d['T_low'] = np.asarray(T_low)   # lowest model level, NOT 2 m tas
            if precip_np is not None:
                z2d['precip'] = precip_np * 86400.0     # [mm/day], like collect()
            if psl is not None:
                z2d['psl'] = np.asarray(psl)
            z3d['T'] = T_np
            if u_east_np is not None:
                z3d['u'] = u_east_np
            if q_v_np is not None:
                z3d['q_v'] = q_v_np * 1000.0             # [g/kg]

        # ================================================================
        # PHASE 2 — commit (cheap, shape-checked add_* only).
        # ================================================================
        fed = False
        if have_spatial:
            if fields_2d:
                self._spatial_monthly.add_2d(doy, year, fields_2d)
                fed = True
            if fields_3d:
                self._spatial_monthly.add_3d(doy, year, fields_3d)
                fed = True
            if daily_2d and self._spatial_daily is not None:
                self._spatial_daily.add_2d(doy, year, daily_2d)
                fed = True
        if have_zonal:
            self.monthly_accum.add_2d(doy, year, z2d, lat_np)
            self.monthly_accum.add_3d(doy, year, z3d, lat_np)
            fed = True

        return fed

```
### model_driver.py :: _mpas_cmip_feed_enabled (5270-5295)
```python
    def _mpas_cmip_feed_enabled(self, diag) -> tuple[bool, bool]:
        """Decide whether the per-interval MPAS CMOR accumulator feed runs.

        Returns ``(feed_on, wants_cmip)``:

        * ``wants_cmip`` — the collector actually holds a CMOR spatial or a
          zonal monthly accumulator (``cmip_output`` / ``monthly_means`` on).
        * ``feed_on`` — ``wants_cmip`` AND the layout is safe to feed: SERIAL
          (``_voronoi_layout is None``) or a 1-rank Voronoi layout that owns the
          whole mesh (no halo; the collector's local regrid weights ARE the
          global weights).  A MULTI-rank cell partition is NOT safe (rank-local
          owned+halo cells + local weights would bin one rank's cells into the
          global lat-lon boxes) and returns ``feed_on=False`` so the caller can
          warn loudly instead of writing rank-local "global" means.
        """
        feed_safe = (
            self._voronoi_layout is None
            or getattr(self, "_mpi_world_size", 1) <= 1
        )
        wants_cmip = diag is not None and (
            getattr(diag, "_spatial_monthly", None) is not None
            or (getattr(diag, "monthly_means", False)
                and getattr(diag, "monthly_accum", None) is not None)
        )
        return (feed_safe and wants_cmip, wants_cmip)

```
### model_driver.py :: _feed_mpas_cmip_accumulators (5296-5389)
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
            # 2 m ``tas`` via MOST similarity when prescribed sst/sic are on
            # this path (``get_sst_sic`` set for a radiation+SST run) — matches
            # the cube-path collect() ``tas`` instead of a bare lowest-level
            # proxy.  Uses the RECONSTRUCTED cell winds (``state.u`` is
            # edge-normal on MPAS, not cell-collocated).  A failure falls back
            # to the lowest model level (logged once) so a tas-only glitch never
            # drops the whole CMOR feed.
            tas = None
            _get_sst_sic = getattr(self, "get_sst_sic", None)
            if _get_sst_sic is not None:
                try:
                    _sst, _sic = _get_sst_sic(day)
                    _sst = jnp.asarray(_sst).reshape(-1)
                    _sic = jnp.asarray(_sic).reshape(-1)
                    tas = diag._tas_2m(
                        state, q_v, _sst, _sic,
                        getattr(self.config, "T_ice", None),
                        u_low=u_east[..., -1], v_low=v_north[..., -1])
                except Exception as exc:
                    if not getattr(self, "_logged_tas2m_fallback", False):
                        logger.warning(
                            "  CMOR tas: 2 m MOST calc failed (%s); using the "
                            "lowest model level as the tas proxy.", exc)
                        self._logged_tas2m_fallback = True
                    tas = None
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
                tas=tas,
            )
        except Exception as exc:  # pragma: no cover - defensive diag guard
            logger.error(
                "  CMOR accumulator feed FAILED at day %.2f (run continues; "
                "the monthly/daily CMOR means for this interval are lost): %s",
                day, exc)

```
### model_driver.py :: _run_mpas gate+warning (5454-5480)
```python
        # once — the collector's accumulator handles are stable for the run.
        _diag = getattr(self, "diagnostics", None)
        self._mpas_cmip_feed_on, _diag_wants_cmip = (
            self._mpas_cmip_feed_enabled(_diag))
        # Make the unsupported multi-rank case LOUD (rank 0 only) rather than
        # silently reproducing the empty-accumulator symptom this fix targets.
        if (_diag_wants_cmip and not self._mpas_cmip_feed_on
                and getattr(self, "_mpi_rank", 0) == 0):
            logger.warning(
                "  CMOR output requested (cmip_output/monthly_means on) but "
                "this is a %d-rank MPAS/Voronoi run — the per-interval "
                "spatial/zonal CMOR feed is UNSUPPORTED under cell-partition "
                "MPI (rank-local cells + local regrid weights). The "
                "monthly/daily CMOR accumulators will stay EMPTY; run "
                "single-rank for CMOR spatial output. (Follow-up: owned-cell "
                "gather + global weights on rank 0.)",
                getattr(self, "_mpi_world_size", 1))

        # Build MPAS-compatible physics via make_physics (same code path as
        # cubed-sphere/lat-lon).  Includes RRTMGP + Held-Suarez forcing
        # when radiation is configured.
        from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
        from legoesm.atmosphere.physics.radiation.config import RadiationConfig
        from legoesm.atmosphere.physics.convection.config import ConvectionConfig
        from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
        from legoesm.atmosphere.physics.microphysics.config import (
            MicrophysicsConfig, apply_microphysics_experiment_flags,
```
### model_driver.py :: feed call in diag block (6343-6353)
```python
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

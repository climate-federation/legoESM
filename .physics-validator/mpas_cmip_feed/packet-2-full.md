# Adversarial review ROUND 2: MPAS monthly-CMOR accumulator feed

You are an independent adversarial reviewer. This is round 2. Below is your
round-1 review, my disposition of each finding, and the UPDATED code + tests.
Re-review: verify each claimed fix is actually correct, check I did not
introduce NEW bugs (esp. the `_tas_2m` refactor that the cube path also uses,
the atomic-commit restructuring, and the MPI gate), and confirm the
documented-as-inherited items are genuinely inherited/out-of-scope. Cite line
numbers. If no substantive bugs remain, say so and justify.

## Round-1 findings and my disposition

1. **`tas` was a lowest-level proxy under the CMOR name.** FIXED. The driver
   now computes the proper 2 m MOST temperature (`_tas_2m`) from the
   RECONSTRUCTED cell winds + prescribed `get_sst_sic`, and passes it to the
   collector as `tas`. `_tas_2m` gained optional `u_low`/`v_low` overrides so
   MPAS can pass cell winds (the cube path calls it unchanged → identical
   behaviour). The collector falls back to the lowest level only when the
   caller supplies no `tas` (e.g. no `get_sst_sic`). Tests:
   `test_feed_uses_provided_2m_tas`, `test_driver_helper_computes_2m_tas`.

2. **Daily/monthly are instantaneous end-of-interval samples; `pr` is the
   last physics-step rate.** DOCUMENTED, not fixed. The lean MPAS loop has no
   per-interval time integrator (unlike the compiled cube path that passes
   segment means). The method docstring now states the snapshot/diurnal-alias
   caveat, that `tasmin`/`tasmax` collapse at 1 sample/day, and that `pr` is a
   rate snapshot — with the device-side interval accumulator as the named
   follow-up. This matches the task's explicit allowance ("feed what IS
   available and document `pr` as a TODO"). A device-side accumulator is a
   separate, larger change (touches the physics-step precip export).

3. **MPI silently left CMOR empty; disabled a safe 1-rank case.** FIXED. The
   gate is now `_mpas_cmip_feed_enabled` (unit-tested): it PERMITS a 1-rank
   Voronoi layout (owns whole mesh, local weights == global) and disables only
   MULTI-rank, where it emits a LOUD rank-0 warning. Test:
   `test_feed_gate_serial_mpi_singlerank`.

4. **Below-ground plev extrapolation (`_interp_to_plev19`).** DOCUMENTED as
   inherited — identical to `collect()`; fixing the shared helper would change
   cube-path output, so it is a separate PR. Noted in the docstring.

5. **Defensive `except` could commit partial state.** FIXED. The collector
   method is now ATOMIC: PHASE 1 does every regrid / plev interpolation into
   local dicts (no accumulator touched); PHASE 2 does only the cheap,
   shape-checked `add_*` calls. A mid-computation failure commits nothing. The
   driver-level `try/except` stays loud-but-nonfatal (matches the existing
   `_save_cmor_accumulator_sidecar` convention; a 100-yr run must not abort on
   a diagnostic glitch).

6. **Inaccurate "free host sync" claim.** FIXED the comment (feed adds a few
   host transfers at DAILY cadence, negligible). You confirmed no retrace /
   gradient issue. Kept the implementation (materialize-once of the np arrays
   at the top of the method).

7. **Zonal means unweighted by cell area.** DOCUMENTED as inherited
   `MonthlyAccumulator` behaviour; the SPATIAL (regridded) CMOR path — the
   primary deliverable — is unaffected. Matters only for variable-resolution
   meshes.

**Test gaps** — ADDRESSED: real `load_cmor_accumulators` resume roundtrip;
Dec→Jan + year-1 bucket test; level-varying winds (`ua850` vs surface) +
vertical `ta` structure; `psl > ps` over 2 km orography and `psl == ps` for
`phis=0`; proper-2m-`tas`-differs-from-lowest (collector + driver); gate
serial/MPI/1-rank test. 17 tests pass (was 10); adjacent suites + constants
ratchet green (3386 passed).

## Facts unchanged from round 1

MPAS state: `T (nCells,nlev)`, `p_s (nCells,)`, `phis (nCells,)`, `u (nEdges,
nlev)` edge-normal, `q_v (nCells,nlev)`, sigma ascending (index -1 = surface).
`reconstruct_cell_velocity -> (u_east, v_north) (nCells,nlev)`. Absolute
`day = START_DAY + elapsed`; `year = int(day//365)`, `doy = day%365+1`.

## UPDATED CODE

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


## UPDATED TESTS

```python
"""MPAS (Voronoi) CMOR monthly/daily/zonal accumulator feed.

Regression tests for the historical gap where the lean MPAS AMIP loop
constructed the ``DiagnosticCollector`` CMOR accumulators
(``SpatialMonthlyAccumulator`` / ``SpatialDailyAccumulator`` /
``MonthlyAccumulator``) and SAVED the restart sidecar, but NEVER fed them —
so every manifest showed ``call_counts: []`` / ``max_count_ever: 0``.

Two layers are covered:

* ``DiagnosticCollector.feed_cmip_accumulators_native`` — regrids native
  Voronoi cell fields to the CMOR lat-lon grid and bins the zonal means,
  populating all three accumulators.
* ``ModelDriver._feed_mpas_cmip_accumulators`` — the driver glue that
  reconstructs geographic cell winds from the edge-normal ``state.u``,
  pulls the surface-precip export, and calls the collector method.

Plus regrid unit checks (uniform -> uniform; hemisphere-split -> correct
sign per hemisphere) on the collector's Voronoi-capable ``_regrid_to_latlon``
helpers.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.factory import create_grid
from legoesm.grids.voronoi import reconstruct_cell_velocity
from legoesm.driver.diagnostics import DiagnosticCollector

NLEV = 6


@pytest.fixture(scope="module")
def mesh():
    # Level-2 SCVT mesh = 162 cells / 480 edges; cheap, enough to exercise
    # both the IDW regrid and the Perot edge->cell wind reconstruction.
    return create_grid("mpas", 2, lloyd_iterations=10)


def _make_collector(mesh, *, monthly_means=True, cmip_output=True,
                    cmip_resolution_deg=10.0):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dsigma = np.full(NLEV, 1.0 / NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full, dsigma=dsigma,
        experiment_id="amip", monthly_means=monthly_means,
        cmip_output=cmip_output, n_days=30, output_dir=None,
        cmip_resolution_deg=cmip_resolution_deg, start_year=1979,
    )
    if cmip_output:
        dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    return dc, sigma_full, dsigma


def _synthetic_cell_fields(mesh, sigma_full):
    """Physically-plausible native cell fields on the mesh."""
    latc = np.asarray(mesh.latCell)
    n = int(mesh.nCells)
    # T: warmer near the surface (large sigma) and near the equator.
    T = 250.0 + 40.0 * sigma_full[None, :] + 10.0 * np.cos(latc)[:, None]
    p_s = np.full(n, 1.0e5)
    q_v = 1e-3 * (1.0 - sigma_full)[None, :] * np.ones((n, 1))
    precip = np.full(n, 2.0e-5)      # kg/m2/s
    phis = np.zeros(n)
    lat_deg = np.degrees(latc)
    # Uniform east wind U=7 as an edge-normal field: u_edge = U*cos(angleEdge).
    U = 7.0
    ang = np.asarray(mesh.angleEdge)
    u_edge = np.broadcast_to(
        (U * np.cos(ang))[:, None], (int(mesh.nEdges), NLEV)
    ).copy()
    return dict(T=T, p_s=p_s, q_v=q_v, precip=precip, phis=phis,
                lat_deg=lat_deg, u_edge=u_edge, U=U)


# ---------------------------------------------------------------------------
# Collector-level feed
# ---------------------------------------------------------------------------

def test_feed_populates_all_three_accumulators(mesh):
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)

    # Pre-condition: this is the exact broken symptom — empty accumulators.
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc._spatial_daily._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0

    fed = dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        q_v=f["q_v"], u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"],
    )
    assert fed is True

    # All three accumulators now have non-empty per-bucket call counts.
    for acc in (dc._spatial_monthly, dc._spatial_daily, dc.monthly_accum):
        assert acc._max_count_ever > 0
        assert len(acc._call_counts) > 0
        assert all(c > 0 for c in acc._call_counts.values())

    # day 15 -> month 1 (Jan), year 0.  Daily bucket keyed on (year, doy=16).
    assert (0, 1) in dc._spatial_monthly._call_counts
    assert (0, 16) in dc._spatial_daily._call_counts


def test_feed_binned_values_are_plausible(mesh):
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        q_v=f["q_v"], u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"],
    )
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert out["months"] == [(0, 1)]

    # ps / pr are spatially uniform -> IDW (partition of unity) is EXACT.
    np.testing.assert_allclose(out["field_2d_ps"], 1.0e5, rtol=1e-9)
    np.testing.assert_allclose(out["field_2d_pr"], 2.0e-5, rtol=1e-9)
    # psl == ps here (phis == 0).
    np.testing.assert_allclose(out["field_2d_psl"], 1.0e5, rtol=1e-9)

    # tas: bounded near-surface temperature; prw positive column vapour.
    tas = out["field_2d_tas"]
    assert np.all(np.isfinite(tas)) and tas.min() > 250.0 and tas.max() < 320.0
    assert np.all(out["field_2d_prw"] > 0.0)

    # 3-D ta on plev19: finite, physical, and the eastward wind ua recovers
    # the ~U=7 m/s uniform flow while va stays ~0.
    ta = out["field_3d_ta"]
    assert ta.shape[-1] == 19
    assert np.all(np.isfinite(ta)) and ta.min() > 180.0 and ta.max() < 330.0
    # Vertical structure (non-vacuous): PLEV19 is ascending, so index -1 is
    # the 100000 Pa (surface) level and index 0 the 100 Pa (top).  The input
    # T warms toward the surface, so ta[...,-1] > ta[...,0] — a vertical flip
    # or a collapsed profile would fail this.
    assert float(np.nanmean(ta[..., -1])) > float(np.nanmean(ta[..., 0])) + 10.0
    assert abs(float(np.nanmean(out["field_3d_ua"])) - f["U"]) < 1.5
    assert abs(float(np.nanmean(out["field_3d_va"]))) < 0.5


def test_feed_zonal_and_daily_finalize(mesh):
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    # Feed two intervals in the same month so the mean is well defined.
    for day in (10.0, 20.0):
        dc.feed_cmip_accumulators_native(
            day=day, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
            q_v=f["q_v"], u_east=np.asarray(u_east),
            v_north=np.asarray(v_north), precip=f["precip"], phis=f["phis"],
        )
    # Zonal monthly finalize: T profile present, plausible.
    zout = dc.monthly_accum.finalize(min_sample_fraction=0)
    assert (0, 1) in zout["months"]
    assert "profile_T" in zout
    prof_T = zout["profile_T"]
    finite = np.isfinite(prof_T)
    assert finite.any()
    assert prof_T[finite].min() > 180.0 and prof_T[finite].max() < 340.0

    # Daily finalize: two distinct days, tas extremes tracked.
    dout = dc._spatial_daily.finalize(min_sample_fraction=0)
    assert len(dout["days"]) == 2
    assert "field_2d_tas" in dout
    assert "field_2d_tas_min" in dout and "field_2d_tas_max" in dout


def test_feed_dry_minimal_inputs(mesh):
    """A dry run (no q_v / precip / winds) still feeds tas/ps + zonal T."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    fed = dc.feed_cmip_accumulators_native(
        day=5.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
    )
    assert fed is True
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_2d_tas" in out and "field_2d_ps" in out
    # No moisture / precip / wind fields fabricated.
    assert "field_2d_pr" not in out
    assert "field_3d_hus" not in out
    assert "field_3d_ua" not in out
    assert dc.monthly_accum._max_count_ever > 0


def test_feed_noop_when_accumulators_disabled(mesh):
    """No CMIP output and no monthly means -> feed is a no-op (returns False)."""
    dc, sigma_full, _ = _make_collector(
        mesh, monthly_means=False, cmip_output=False)
    assert dc._spatial_monthly is None
    f = _synthetic_cell_fields(mesh, sigma_full)
    fed = dc.feed_cmip_accumulators_native(
        day=5.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"])
    assert fed is False


def test_feed_manifest_roundtrip_nonempty(mesh, tmp_path):
    """The restart sidecar manifest is non-empty after a feed (the exact
    field that was ``call_counts: []`` in the broken pilot runs)."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        q_v=f["q_v"], u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"],
    )
    sidecar = tmp_path / "cmor_accum_day_0015.npz"
    dc.save_cmor_accumulators(sidecar)

    z = np.load(sidecar, allow_pickle=True)
    manifests = {k: json.loads(str(z[k].item())) if isinstance(z[k].item(), str)
                 else z[k].item()
                 for k in z.keys() if "manifest" in k}
    assert manifests, "sidecar wrote no manifests"
    for name, m in manifests.items():
        assert m["max_count_ever"] > 0, f"{name} still empty after feed"
        assert m["call_counts"], f"{name} call_counts empty after feed"


# ---------------------------------------------------------------------------
# Regrid helper: uniform + hemisphere-sign
# ---------------------------------------------------------------------------

def test_regrid_uniform_field_is_uniform(mesh):
    dc, _, _ = _make_collector(mesh)
    n = int(mesh.nCells)
    out2d = dc._regrid_to_latlon_2d(np.full(n, 5.0))
    assert out2d is not None
    assert out2d.shape == (dc._cmip_nlat, dc._cmip_nlon)
    np.testing.assert_allclose(out2d, 5.0, atol=1e-9)

    out3d = dc._regrid_to_latlon_3d(
        np.full((n, NLEV), 3.0) * np.arange(1, NLEV + 1))
    assert out3d.shape == (dc._cmip_nlat, dc._cmip_nlon, NLEV)
    for lvl in range(NLEV):
        np.testing.assert_allclose(out3d[..., lvl], 3.0 * (lvl + 1), atol=1e-9)


def test_regrid_hemisphere_split_keeps_sign(mesh):
    """A +1 (NH) / -1 (SH) cell field regrids to a lat-lon field that is
    positive in the northern rows and negative in the southern rows."""
    dc, _, _ = _make_collector(mesh)
    latc = np.asarray(mesh.latCell)
    field = np.where(latc >= 0.0, 1.0, -1.0)
    out = dc._regrid_to_latlon_2d(field)
    assert out is not None

    lat_cent = dc._voronoi_regrid_weights.lat_cent  # linspace(-90, 90, nlat)
    nh = lat_cent > 20.0
    sh = lat_cent < -20.0
    assert nh.any() and sh.any()
    # Well inside each hemisphere the IDW blend cannot flip the sign.
    assert np.all(out[nh, :] > 0.5), "northern rows lost their + sign"
    assert np.all(out[sh, :] < -0.5), "southern rows lost their - sign"
    # Values stay within the source range (IDW is a convex combination).
    assert out.min() >= -1.0 - 1e-9 and out.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Driver glue: reconstruct winds + pull precip + feed
# ---------------------------------------------------------------------------

def test_driver_helper_feeds_via_reconstruct(mesh):
    """``ModelDriver._feed_mpas_cmip_accumulators`` reconstructs cell winds
    from the edge-normal ``state.u``, extracts the surface-precip export, and
    feeds all three accumulators.  Exercised on a lightweight stand-in so the
    exact glue (shapes, handles) is covered without a full driver build."""
    from legoesm.driver.model_driver import ModelDriver

    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        state=types.SimpleNamespace(
            u=_field(f["u_edge"]),          # (nEdges, nlev) edge-normal
            T=_field(f["T"]),
            p_s=_field(f["p_s"]),
            phis=_field(f["phis"]),
            tracers={"q_v": _field(f["q_v"])},
        ),
        model=types.SimpleNamespace(
            _sfc_diag=(None, None, _field(f["precip"])),
        ),
    )
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)

    # The try/except in the helper swallows glue bugs into a log line, so a
    # populated accumulator is the real assertion that the wiring works.
    assert dc._spatial_monthly._max_count_ever > 0
    assert dc.monthly_accum._max_count_ever > 0
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    # Winds threaded through -> ua present and ~uniform east flow recovered.
    assert "field_3d_ua" in out
    assert abs(float(np.nanmean(out["field_3d_ua"])) - f["U"]) < 1.5
    # Precip export threaded through -> pr present and exact-uniform.
    np.testing.assert_allclose(out["field_2d_pr"], 2.0e-5, rtol=1e-9)


def test_driver_helper_dry_run_no_precip(mesh):
    """No ``_sfc_diag`` precip slot -> feed still populates tas/ps (no pr)."""
    from legoesm.driver.model_driver import ModelDriver

    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        get_sst_sic=None,
        state=types.SimpleNamespace(
            u=_field(f["u_edge"]), T=_field(f["T"]), p_s=_field(f["p_s"]),
            phis=_field(f["phis"]), tracers=None,
        ),
        model=types.SimpleNamespace(_sfc_diag=None),
    )
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_2d_tas" in out and "field_2d_ps" in out
    assert "field_2d_pr" not in out
    assert "field_3d_hus" not in out


# ---------------------------------------------------------------------------
# 2 m tas / psl / 850 hPa — non-vacuous physical structure
# ---------------------------------------------------------------------------

def test_feed_uses_provided_2m_tas(mesh):
    """A distinct 2 m `tas` field is regridded as `tas` — NOT the lowest
    model level (guards against the field being ignored or overwritten)."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    tas2m = f["T"][:, -1] + 5.0                    # a clearly-distinct field
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], tas=tas2m)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    tas_r = out["field_2d_tas"]
    low_r = dc._regrid_to_latlon_2d(f["T"][:, -1])
    # Regrid is linear -> the +5 offset survives; tas must equal low+5, not low.
    np.testing.assert_allclose(tas_r[0], low_r + 5.0, atol=1e-6)
    assert float(np.nanmean(tas_r)) > float(np.nanmean(low_r)) + 4.9


def test_feed_psl_exceeds_ps_over_mountains(mesh):
    """psl reduces surface pressure to sea level: psl > ps where phis > 0,
    psl == ps where phis == 0."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    phis = np.full(int(mesh.nCells), 2000.0 * constants.g)   # ~2000 m orog
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], phis=phis)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    psl = out["field_2d_psl"][0]
    ps = out["field_2d_ps"][0]
    assert np.all(psl > ps + 1.0), "psl must exceed ps over positive orography"
    # ~2 km, ~290 K -> hypsometric factor exp(gz/RT) ~ 1.27 -> ~127 hPa higher.
    assert float(np.mean(psl - ps)) > 5000.0

    # Sanity: phis == 0 -> psl == ps.
    dc0, sig0, _ = _make_collector(mesh)
    dc0.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        phis=np.zeros(int(mesh.nCells)))
    out0 = dc0._spatial_monthly.finalize(min_sample_fraction=0)
    np.testing.assert_allclose(
        out0["field_2d_psl"][0], out0["field_2d_ps"][0], rtol=1e-9)


def test_feed_level_varying_winds_850(mesh):
    """A vertically-sheared zonal jet: ua850 recovers the 850 hPa level value
    and differs from the surface — a non-vacuous 850-slice / plev check."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    # U increases with height: U(sigma) = 5 + 20*(1 - sigma).
    ang = np.asarray(mesh.angleEdge)
    U_lev = 5.0 + 20.0 * (1.0 - sigma_full)          # surface ~5.4, top ~24
    u_edge = (U_lev[None, :] * np.cos(ang)[:, None])
    u_east, v_north = reconstruct_cell_velocity(u_edge, mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        u_east=np.asarray(u_east), v_north=np.asarray(v_north))
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    ua = out["field_3d_ua"][0]                       # (nlat, nlon, 19)
    from legoesm.io.cmor_output import CMIP6_PLEV19
    plev = np.sort(np.asarray(CMIP6_PLEV19))
    i850 = int(np.argmin(np.abs(plev - 85000.0)))
    i_top = 0
    # p_s = 1e5 -> sigma_850 = 0.85 -> U ~ 5 + 20*0.15 = 8.0 m/s.
    ua850 = float(np.nanmean(ua[..., i850]))
    ua_top = float(np.nanmean(ua[..., i_top]))
    assert abs(ua850 - 8.0) < 2.0, f"ua850={ua850} not near the 850 hPa value"
    assert ua_top > ua850 + 3.0, "jet must strengthen aloft (non-vacuous shear)"


# ---------------------------------------------------------------------------
# Calendar buckets + restart resume
# ---------------------------------------------------------------------------

def test_feed_month_year_buckets(mesh):
    """Distinct simulated days land in the correct (year, month) buckets,
    including a Dec bucket and a year-1 bucket (restart-chain calendar)."""
    from legoesm.forcing.time_utils import day_to_calendar
    from legoesm.diagnostics.monthly_means import MonthlyAccumulator as _MA

    def _key(day):
        doy, _ = day_to_calendar(day)
        return (int(day // 365.0), _MA.day_to_month(doy))

    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    days = [360.0, 400.0]                            # Dec y0, then y1
    for d in days:
        dc.feed_cmip_accumulators_native(
            day=d, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"])
    keys = set(dc._spatial_monthly._call_counts)
    exp = {_key(d) for d in days}
    assert keys == exp
    years = {k[0] for k in keys}
    assert years == {0, 1}                           # crossed a calendar year
    assert (0, 12) in keys                           # December of year 0


def test_load_cmor_accumulators_resume_roundtrip(mesh, tmp_path):
    """A fed sidecar restored via load_cmor_accumulators into a FRESH collector
    reproduces the same finalized monthly means (the real restart path)."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], q_v=f["q_v"],
        u_east=np.asarray(u_east), v_north=np.asarray(v_north),
        precip=f["precip"], phis=f["phis"])
    sidecar = tmp_path / "cmor_accum_day_0015.npz"
    dc.save_cmor_accumulators(sidecar)
    ref = dc._spatial_monthly.finalize(min_sample_fraction=0)

    dc2, _, _ = _make_collector(mesh)
    assert dc2._spatial_monthly._max_count_ever == 0
    restored = dc2.load_cmor_accumulators(sidecar)
    assert restored is True
    got = dc2._spatial_monthly.finalize(min_sample_fraction=0)
    assert got["months"] == ref["months"]
    for k in ("field_2d_tas", "field_2d_ps", "field_2d_pr", "field_3d_ta"):
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-9, equal_nan=True)


# ---------------------------------------------------------------------------
# Serial / MPI gate + driver-side 2 m tas
# ---------------------------------------------------------------------------

def test_feed_gate_serial_mpi_singlerank(mesh):
    """`_mpas_cmip_feed_enabled`: serial and 1-rank feed; multi-rank does not,
    but reports wants_cmip=True so the caller can warn."""
    from legoesm.driver.model_driver import ModelDriver
    dc, _, _ = _make_collector(mesh)

    # Serial (no voronoi layout).
    serial = types.SimpleNamespace(_voronoi_layout=None, _mpi_world_size=1)
    assert ModelDriver._mpas_cmip_feed_enabled(serial, dc) == (True, True)

    # Multi-rank cell partition -> disabled, but wants_cmip True.
    multi = types.SimpleNamespace(
        _voronoi_layout=object(), _mpi_world_size=4)
    assert ModelDriver._mpas_cmip_feed_enabled(multi, dc) == (False, True)

    # 1-rank "distributed" layout owns the whole mesh -> safe to feed.
    single = types.SimpleNamespace(
        _voronoi_layout=object(), _mpi_world_size=1)
    assert ModelDriver._mpas_cmip_feed_enabled(single, dc) == (True, True)

    # No CMIP output at all -> neither.
    dc_off, _, _ = _make_collector(mesh, monthly_means=False, cmip_output=False)
    assert ModelDriver._mpas_cmip_feed_enabled(serial, dc_off) == (False, False)


def test_driver_helper_computes_2m_tas(mesh):
    """With prescribed sst warmer than the lowest air level, the driver helper
    publishes a 2 m MOST `tas` distinct from (warmer than) the lowest level."""
    from legoesm.driver.model_driver import ModelDriver
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    T_low = f["T"][:, -1]

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    sst = T_low + 10.0                               # warm surface -> unstable
    sic = np.zeros(int(mesh.nCells))
    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        config=types.SimpleNamespace(T_ice=271.4),
        get_sst_sic=lambda day: (sst, sic),
        state=types.SimpleNamespace(
            u=_field(f["u_edge"]), T=_field(f["T"]), p_s=_field(f["p_s"]),
            phis=_field(f["phis"]), tracers={"q_v": _field(f["q_v"])},
        ),
        model=types.SimpleNamespace(_sfc_diag=(None, None, _field(f["precip"]))),
    )
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    tas_r = out["field_2d_tas"][0]
    low_r = dc._regrid_to_latlon_2d(T_low)
    # 2 m sits between the warm surface and the cooler lowest level -> warmer
    # than the lowest level; a bug that published T[...,-1] would fail this.
    assert float(np.nanmean(tas_r)) > float(np.nanmean(low_r)) + 0.5
    assert not np.allclose(tas_r, low_r)

```

## Your task

Re-review focusing on: (a) does the `_tas_2m` refactor keep the cube path
byte-identical and is the MPAS 2 m computation correct (cell winds, sst/sic
reshape, `T_ice` handling, the inner fallback)? (b) is the atomic PHASE1/PHASE2
restructuring actually atomic and behaviour-preserving vs round 1? (c) is the
1-rank MPI permit safe, and the multi-rank warning correct? (d) any NEW shape,
unit, sign, or dtype bug introduced? (e) are the new tests non-vacuous and do
they actually assert the fixed behaviour? If no substantive bugs remain, say so
explicitly.


## APPENDIX: round-1 review (verbatim)

Verdict: not ready as a scientifically valid CMOR feed. The wiring fixes the empty-manifest symptom in serial, but it publishes mislabeled `tas`, produces snapshot—not daily-mean—`day` fields, and silently disables the feature under MPI.

1. **CMOR `tas` is wrong.** `T[..., -1]` is explicitly a lowest-model-level proxy, then stored as `tas` ([diagnostics.py:1342–1350](packages/coupler/legoesm/driver/diagnostics.py:1342), [1365–1367](packages/coupler/legoesm/driver/diagnostics.py:1365)). `collect()` uses the MOST 2 m calculation; this path knowingly does not. The same wrong quantity also becomes daily `tas`, `tasmin`, and `tasmax` ([1413–1429](packages/coupler/legoesm/driver/diagnostics.py:1413)). Either thread the surface-layer inputs and calculate 2 m temperature, or omit `tas`; a proxy must not be emitted under the CMOR name.

2. **The daily products are one instantaneous end-of-day sample, not daily means/extrema.** The feed occurs only at `DIAG_INTERVAL` ([model_driver.py:6193–6195](packages/coupler/legoesm/driver/model_driver.py:6193), [6287–6294](packages/coupler/legoesm/driver/model_driver.py:6287)); at the stated daily cadence, each daily bucket gets one state snapshot. `tasmin` and `tasmax` will therefore equal that snapshot, and `pr`, `psl`, `ua850`, and `va850` are not daily averages. This is especially problematic for precipitation: `_sfc_diag[2]` is the most recent physics-step rate ([5312–5319](packages/coupler/legoesm/driver/model_driver.py:5312)), not an interval integral/mean. Accumulate device-side sums/min/max through the day, or sample sub-daily and document the approximation as non-CMOR.

3. **MPI skips wrong data correctly, but silently leaves requested CMOR output empty.** The gate disables every feed whenever `_voronoi_layout` exists ([5404–5412](packages/coupler/legoesm/driver/model_driver.py:5404)). That is preferable to regridding rank-local owned+halo cells with global weights. However, it recreates the original failure mode—empty sidecars/no CMOR fields—without a startup warning or hard configuration error. It also disables a one-rank “distributed” layout even when it owns the whole mesh and would be safe. Emit an explicit unsupported-MPI CMOR error/warning, and permit the `n_ranks == 1` full-mesh case.

4. **Pressure-level output is physically invalid below ground (inherited, but now affects MPAS).** The new feed uses `_interp_to_plev19` for all 3-D fields ([1403–1408](packages/coupler/legoesm/driver/diagnostics.py:1403)), then labels its 850-hPa slice as `ua850`/`va850` ([1421–1426](packages/coupler/legoesm/driver/diagnostics.py:1421)). That helper clips interpolation weights, so a target level below the lowest model level—e.g. 850 hPa over high terrain, or 1000 hPa when the lowest full level is `0.98 p_s`—is filled with the nearest model-level value rather than masked. Bounded extrapolation is not valid pressure-level CMOR data. This is pre-existing shared-helper behavior, not a new shape error, but the new MPAS path inherits it.

5. **The defensive catch permits silently incomplete and internally inconsistent CMOR output.** A failure after `add_2d` but before 3-D/daily/zonal completion leaves partial state committed; the outer `except` merely logs and continues ([5332–5336](packages/coupler/legoesm/driver/model_driver.py:5332)). A persistent bug can therefore yield a “COMPLETED” run with absent or differently sampled fields. For CMOR-enabled runs, validate all inputs/regrids before mutating accumulators and fail the CMOR job—or mark output invalid—rather than continue.

6. **The host-sync claim is inaccurate, though there is no retrace or gradient bug.** The scalar diagnostic reduction synchronizes computation but does not materialize `T`, `p_s`, `q_v`, winds, precip, or `phis` on host. This method then performs several separate `np.asarray` transfers ([1342–1343](packages/coupler/legoesm/driver/diagnostics.py:1342), [1356](packages/coupler/legoesm/driver/diagnostics.py:1356), [1374–1380](packages/coupler/legoesm/driver/diagnostics.py:1374), [1395–1406](packages/coupler/legoesm/driver/diagnostics.py:1395)), after a JAX Perot reconstruction ([5304–5307](packages/coupler/legoesm/driver/model_driver.py:5304)). It also recomputes CWV even though the diagnostic block already computed it. This is a real daily GPU→host performance cost; batch `device_get` the required leaves and pass the already-computed CWV. It does not retrace—the helper is outside `jit`/`grad`—and the host conversion is an intentional diagnostic gradient barrier.

7. **Zonal means are unweighted by cell area.** The new call supplies only values and latitude ([1433–1446](packages/coupler/legoesm/driver/diagnostics.py:1433)); `MonthlyAccumulator` bins/arithmetic-averages cells. SCVT cells are near-uniform, not exactly equal-area, and this is wrong for variable-resolution MPAS meshes. This is inherited accumulator behavior, but it is not a conservative/area-weighted zonal mean.

The following candidate concerns are **not bugs**:

- The normal serial shapes are consistent: `(nCells,nlev)` → `(nCells,19)` → `(nlat,nlon,19)`, and the 850-hPa index uses the same ascending pressure ordering as `collect()`.
- `year = int(day // 365.0)` with `doy = day % 365 + 1` is consistent with the cube path and restart buckets: day 365 maps to `(year=1, doy=1)`. The writer expects run-relative years, so `start_year` should not be added here.
- Spatial `pr` correctly remains `kg m⁻² s⁻¹`; zonal `precip * 86400` correctly gives `mm day⁻¹`. `hus` is correctly unchanged spatially and converted to `g kg⁻¹` only for the legacy zonal profile.
- The sea-level-pressure sign is correct ([1355–1358](packages/coupler/legoesm/driver/diagnostics.py:1355)), and the Perot call plus `_sfc_diag[2].data` handle are correct for the current MPAS contracts.

Test gaps are substantial:

- The “roundtrip” test only saves and inspects JSON; it never calls `load_cmor_accumulators` or continues a second link ([test:199–220](tests/unit/test_mpas_cmip_accumulator_feed.py:199)).
- All calendar assertions remain January/year 0; no Dec→Jan or restarted bucket test exists.
- Constant `T`, wind, and precipitation make the daily averaging, `tasmin/max`, 850-hPa slice, and `pr` units largely vacuous ([test:64–75](tests/unit/test_mpas_cmip_accumulator_feed.py:64), [144–168](tests/unit/test_mpas_cmip_accumulator_feed.py:144)).
- `phis=0` cannot test the `psl` sign/magnitude, and the broad `tas` range would still pass if the top level were used ([test:127–141](tests/unit/test_mpas_cmip_accumulator_feed.py:127)).
- The driver test calls the helper directly ([test:293](tests/unit/test_mpas_cmip_accumulator_feed.py:293)); it does not test `_run_mpas` cadence, the serial/MPI gate, real JAX transfers, or error handling.

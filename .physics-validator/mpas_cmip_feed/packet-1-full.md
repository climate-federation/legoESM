# Adversarial review request: MPAS monthly-CMOR accumulator feed

You are an independent adversarial physics/software reviewer for LegoESM, a
JAX-native differentiable Earth System Model. Find EVERY bug, sign error, unit
inconsistency, shape mismatch, broken-gradient pattern, conservation violation,
retrace/host-sync hazard, and test-case discrepancy in the change below. Cite
line numbers (relative to the snippets). If you believe a candidate concern is
NOT a bug, say so and explain why.

## Problem being fixed

On the MPAS (Voronoi/unstructured) AMIP execution path (`ModelDriver._run_mpas`),
the `DiagnosticCollector` constructs the CMOR accumulators
(`SpatialMonthlyAccumulator` / `SpatialDailyAccumulator` / zonal
`MonthlyAccumulator`) and the restart **sidecar SAVE** runs each checkpoint
(`cmor_accum_day_*.npz` appear), but the per-step **FEED never happened** on this
path. Verified on a real pilot run: every accumulator manifest showed
`call_counts: []` / `max_count_ever: 0`. The cube/lat-lon path feeds via
`DiagnosticCollector.collect(...)`, which regrids state fields to the CMOR
lat-lon grid and calls `_spatial_monthly.add_2d/add_3d`. The lean MPAS loop
never calls `collect()` — it only writes lightweight scalar timeseries.

## Design

- The collector's Voronoi regrid helpers already exist and are tested
  (`_regrid_to_latlon_2d/_3d` dispatch to `apply_voronoi_to_latlon[_3d]` via
  IDW k-nearest weights built from `mesh.latCell/lonCell`; `_interp_to_plev19`
  is column-wise and works on `(nCells, nlev)` with `p_s (nCells,)`).
- NEW collector method `feed_cmip_accumulators_native(day, *, T, p_s, ...)`
  feeds the three accumulators from NATIVE cell arrays, reusing the SAME
  regrid + plev19 helpers `collect()` uses (no re-derived numerics).
- NEW driver helper `_feed_mpas_cmip_accumulators(day)` reconstructs geographic
  cell winds from the edge-normal `state.u` via `reconstruct_cell_velocity`
  (Perot), pulls the surface-precip export `self.model._sfc_diag[2]`, and calls
  the collector method. Called at diag cadence inside `_run_mpas`, gated by a
  precomputed `self._mpas_cmip_feed_on` flag (spatial or zonal accumulators
  enabled AND serial — `self._voronoi_layout is None`).
- MPI Voronoi is intentionally OUT of scope (documented): the collector holds
  LOCAL regrid weights and each rank holds only local cells, so a per-rank feed
  would bin one rank's cells into the global lat-lon boxes. The lightweight
  timeseries is already a true global via `_mpas_global_diag`.

## Key facts about the surrounding code (verified by reading)

- MPAS state layout: `state.T.data (nCells, nlev)`, `state.p_s.data (nCells,)`,
  `state.phis.data (nCells,)`, `state.u.data (nEdges, nlev)` (edge-normal),
  `state.tracers["q_v"].data (nCells, nlev)`. `sigma_full` ascending, index
  `-1` = lowest/surface level.
- `reconstruct_cell_velocity(u_edge, mesh) -> (u_east, v_north)` each
  `(nCells, nlev)`; exact for uniform flow (Perot area-weighted).
- Cube path uses ABSOLUTE `day = START_DAY + elapsed` and
  `doy,_ = day_to_calendar(day); year = int(day // 365.0)`. `day_to_calendar`
  returns `doy = day % 365 + 1` (noleap, 1-based).
- CMOR conventions in `collect()`: spatial `pr` in kg/m2/s (native), zonal
  `precip` in mm/day (`*86400`), `hus`/`q_v` profile in g/kg (`*1000`), `psl`
  hypsometric `p_s*exp(phis/(R_d*max(T_low,T_min)))`, daily 850 hPa index is
  `argmin(|sort(PLEV19)-85000|)`.
- Accumulator APIs: `SpatialMonthlyAccumulator.add_2d/3d(doy, year, fields)`
  requires exact `(nlat,nlon)` / `(nlat,nlon,nlev)` shapes and raises otherwise;
  zonal `MonthlyAccumulator.add_2d/3d(doy, year, fields, lat_deg)` bins by
  `np.digitize(lat_deg.ravel(), lat_edges)`. Both bump `_call_counts` /
  `_max_count_ever`.

## THE CHANGES

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


## Static analysis (my self-review)

- **Units**: tas [K] (lowest-level T proxy — documented; 2 m MOST refinement
  needs sst/sic not threaded here, marked TODO); ps/psl [Pa]; pr [kg/m2/s] to
  spatial, [mm/day] to zonal; prw [kg/m2] via `column_water_vapor`; ta/ua/va on
  plev19 [K]/[m/s]; hus [kg/kg] to spatial, [g/kg] to zonal profile. All mirror
  `collect()`.
- **Signs**: no flux/tendency signs (diagnostic-only, host-side, cannot perturb
  prognostic state). `u_east` is geographic eastward (correct CMOR `ua`);
  `psl` uses `+phis/(R_d*T)` (standard hypsometric, phis>0 raises psl above ps).
- **Differentiability**: N/A — pure host-side numpy diagnostic path, NOT in any
  `jax.grad`/`jit` trace. `reconstruct_cell_velocity` runs eagerly at diag
  cadence only; its output is immediately `np.asarray`-materialized.
- **Conservation**: none claimed; this only accumulates diagnostics. IDW regrid
  is a convex combination (partition of unity) — verified a uniform field maps
  to the same uniform value (ps/pr exact in tests).
- **Retrace/host-sync**: feed runs only at DIAG cadence (daily), after the diag
  block already synced state to host. Placed AFTER the finiteness/bounds guard
  so a blown-up state never enters the accumulator. `except Exception` makes a
  feed failure LOUD-but-nonfatal (matches the sidecar's defensive pattern), so
  a 100-yr run cannot abort on a diagnostic bug.
- **Limiters**: `np.maximum(T_low, T_min_atmosphere)` guards the psl exponent
  (same as `collect()`); `_interp_to_plev19` clips extrapolation via `alpha`.

## Test results (all pass, srun CPU, JAX_ENABLE_X64=1)

10 new tests in `tests/unit/test_mpas_cmip_accumulator_feed.py`:
- `test_feed_populates_all_three_accumulators` — pre-condition empty, post non-empty.
- `test_feed_binned_values_are_plausible` — ps/pr/psl EXACT-uniform (1e5, 2e-5),
  tas in (250,320), ua recovers ~7 m/s uniform flow, va ~0.
- `test_feed_zonal_and_daily_finalize` — zonal profile_T finalizes; daily 2 days + tas extremes.
- `test_feed_dry_minimal_inputs` — no q_v/precip/wind still feeds tas/ps + zonal.
- `test_feed_noop_when_accumulators_disabled` — returns False, no crash.
- `test_feed_manifest_roundtrip_nonempty` — save_cmor_accumulators sidecar manifest
  non-empty (the exact `call_counts:[]` symptom, fixed).
- `test_regrid_uniform_field_is_uniform` / `test_regrid_hemisphere_split_keeps_sign`.
- `test_driver_helper_feeds_via_reconstruct` / `test_driver_helper_dry_run_no_precip`.
Adjacent existing suites still green (voronoi regrid, diagnostic collector).
Constants + physics-contract ratchets green (3489 passed).

## The test file

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

```

## Your task

Cite line numbers. Focus on: (1) shape/units bugs in `feed_cmip_accumulators_native`
and the regrid/plev calls; (2) the `year = int(day // 365.0)` + doy calendar
consistency with the cube path (esp. restart chains and the mm/day vs kg/m2/s
split); (3) whether the driver helper's `state.u.data`/`_sfc_diag[2]` handles are
correct and whether `reconstruct_cell_velocity` at diag cadence introduces a
retrace/host-sync/perf hazard; (4) the MPI-serial gate — is skipping the MPI
feed the right call, or is there a silent-wrong-result risk; (5) test rigor — do
the assertions actually catch the failure mode, or are any vacuous. If there are
no substantive bugs, say so explicitly and justify why each candidate concern is
not one.

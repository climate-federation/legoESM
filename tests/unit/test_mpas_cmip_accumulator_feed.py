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

def _as_driver(ns):
    """Bind the ``ModelDriver`` methods the CMOR feed calls on ``self``.

    ``types.SimpleNamespace`` stand-ins cannot inherit them, and the feed
    delegates its native-field construction to ``_mpas_cmip_native_kwargs``
    (shared with the multi-rank gather path).
    """
    import functools

    from legoesm.driver.model_driver import ModelDriver
    ns._mpas_cmip_native_kwargs = functools.partial(
        ModelDriver._mpas_cmip_native_kwargs, ns)
    if not hasattr(ns, "sigma"):
        # The helper publishes `wap` by closing continuity, which needs the
        # run's vertical coordinate. A stub without one would silently drop
        # every field, so give it the same coordinate the collector was built
        # on rather than letting the helper degrade quietly.
        from legoesm.grids.vertical import create_sigma_coordinate
        ns.sigma = create_sigma_coordinate(NLEV)
    return ns




@pytest.fixture(scope="module")
def mesh():
    # Level-2 SCVT mesh = 162 cells / 480 edges; cheap, enough to exercise
    # both the IDW regrid and the Perot edge->cell wind reconstruction.
    return create_grid("mpas", 2, lloyd_iterations=10)


def _make_collector(mesh, *, monthly_means=True, cmip_output=True,
                    cmip_resolution_deg=10.0, output_dir=None):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dsigma = np.full(NLEV, 1.0 / NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full, dsigma=dsigma,
        experiment_id="amip", monthly_means=monthly_means,
        cmip_output=cmip_output, n_days=30, output_dir=output_dir,
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
    _as_driver(fake)
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
    _as_driver(fake)
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
    """`_mpas_cmip_feed_enabled` returns ``(feed_on, wants_cmip)``.

    Every layout this driver knows now feeds — multi-rank through the
    owned-cell gather (#1517) — so ``feed_on`` tracks ``wants_cmip`` for all
    three. The PAIR is kept because ``_require_mpas_cmip_feed_supported``
    consumes ``wants_cmip`` to refuse a layout that is none of them.
    """
    from legoesm.driver.model_driver import ModelDriver
    dc, _, _ = _make_collector(mesh)

    # Serial (no voronoi layout).
    serial = types.SimpleNamespace(_voronoi_layout=None, _mpi_world_size=1)
    assert ModelDriver._mpas_cmip_feed_enabled(serial, dc) == (True, True)

    # Multi-rank cell partition -> now ENABLED (gather path).
    multi = types.SimpleNamespace(
        _voronoi_layout=object(), _mpi_world_size=4)
    assert ModelDriver._mpas_cmip_feed_enabled(multi, dc) == (True, True)

    # 1-rank "distributed" layout owns the whole mesh -> direct feed.
    single = types.SimpleNamespace(
        _voronoi_layout=object(), _mpi_world_size=1)
    assert ModelDriver._mpas_cmip_feed_enabled(single, dc) == (True, True)

    # No CMIP output at all -> off, and nothing to refuse either.
    dc_off, _, _ = _make_collector(mesh, monthly_means=False, cmip_output=False)
    assert ModelDriver._mpas_cmip_feed_enabled(serial, dc_off) == (False, False)

    # A MULTI-rank layout with NO Voronoi partition has no feed, so it must
    # still report feed_on=False. This is what keeps the #1545 refusal a live
    # tripwire instead of dead code: written the lazy way ("layout is None or
    # world <= 1 or partitioned") this case would wrongly read as feedable.
    unfed = types.SimpleNamespace(_voronoi_layout=None, _mpi_world_size=4)
    assert ModelDriver._mpas_cmip_feed_enabled(unfed, dc) == (False, True)


def test_multirank_no_longer_trips_the_empty_cmor_refusal(mesh):
    """The #1545 trap must STOP firing for the layout #1517 now feeds.

    Its own unit tests below still exercise the refusal directly (they hand it
    ``feed_on=False``), which is what keeps it honest. This one pins the
    integration fact that matters: the multi-rank cell partition reaches the
    gather instead of the raise. If the guard still fired here the whole
    feature would be unreachable in production.
    """
    from legoesm.driver.model_driver import ModelDriver
    dc, _, _ = _make_collector(mesh)

    multi = types.SimpleNamespace(
        _voronoi_layout=object(), _mpi_world_size=4, _mpi_rank=0)
    feed_on, wants = ModelDriver._mpas_cmip_feed_enabled(multi, dc)
    assert feed_on is True
    # No raise: the cell-partition lane is supported now.
    ModelDriver._require_mpas_cmip_feed_supported(multi, feed_on, wants)


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
    _as_driver(fake)
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    tas_r = out["field_2d_tas"][0]
    low_r = dc._regrid_to_latlon_2d(T_low)
    # 2 m sits between the warm surface and the cooler lowest level -> warmer
    # than the lowest level; a bug that published T[...,-1] would fail this.
    assert float(np.nanmean(tas_r)) > float(np.nanmean(low_r)) + 0.5
    assert not np.allclose(tas_r, low_r)


# ---------------------------------------------------------------------------
# Atomic commit (no partial state on a bad input)
# ---------------------------------------------------------------------------

def test_feed_atomic_on_bad_lat_deg(mesh):
    """A malformed `lat_deg` is rejected in PHASE 1 (before any add_*), so all
    three accumulators are left untouched — the feed is transactional."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    bad_lat = np.zeros(int(mesh.nCells) + 3)          # wrong length
    with pytest.raises(ValueError, match="lat_deg shape"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=bad_lat, precip=f["precip"])
    # Nothing committed anywhere (would be > 0 if the spatial add ran first).
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc._spatial_daily._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0
    assert not dc._spatial_monthly._data_2d
    assert not dc.monthly_accum._data


def test_feed_atomic_on_nonnumeric_or_nonfinite_lat(mesh):
    """A shape-correct but non-numeric (or non-finite) `lat_deg` is rejected in
    PHASE 1 too — before np.digitize could raise mid-commit."""
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    n = int(mesh.nCells)
    # Non-numeric object vector of the right length.
    bad_obj = np.array(["x"] * n, dtype=object)
    with pytest.raises((ValueError, TypeError)):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=bad_obj, precip=f["precip"])
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0

    # Non-finite (NaN) latitude.
    bad_nan = f["lat_deg"].copy()
    bad_nan[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=bad_nan, precip=f["precip"])
    assert dc._spatial_monthly._max_count_ever == 0
    assert dc.monthly_accum._max_count_ever == 0


def test_feed_atomic_zonal_only_malformed_siblings(mesh):
    """ZONAL-ONLY mode (no spatial regrid to catch shapes): a wrong-length
    `precip` or `u_east` is rejected in PHASE 1, leaving the zonal accumulator
    untouched (the config that has no regrid phase to catch it otherwise)."""
    dc, sigma_full, _ = _make_collector(
        mesh, monthly_means=True, cmip_output=False)
    assert dc._spatial_monthly is None and dc.monthly_accum is not None
    f = _synthetic_cell_fields(mesh, sigma_full)
    n = int(mesh.nCells)

    with pytest.raises(ValueError, match="precip shape"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
            precip=np.zeros(n + 2))
    assert dc.monthly_accum._max_count_ever == 0
    assert not dc.monthly_accum._data

    with pytest.raises(ValueError, match="u_east shape"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
            u_east=np.zeros((n + 1, NLEV)))
    assert dc.monthly_accum._max_count_ever == 0

    # A well-formed zonal-only feed still works after the rejections.
    fed = dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"],
        precip=f["precip"], q_v=f["q_v"])
    assert fed is True and dc.monthly_accum._max_count_ever > 0


def test_feed_atomic_rejects_malformed_T(mesh):
    """A malformed required `T` (1-D, wrong nlev, or non-numeric) is rejected in
    PHASE 1 — zonal-only mode has no regrid to catch it, so add_2d must never
    commit before add_3d fails on a shape/dtype mismatch.  The nlev check is
    against the collector's configured nlev so a mixed-nlev feed can't slip
    through after a valid one."""
    dc, sigma_full, _ = _make_collector(
        mesh, monthly_means=True, cmip_output=False)
    f = _synthetic_cell_fields(mesh, sigma_full)
    n = int(mesh.nCells)

    # 1-D T (wrong rank).
    with pytest.raises(ValueError, match="T must be 2-D"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=f["T"][:, -1], p_s=f["p_s"], lat_deg=f["lat_deg"])
    assert dc.monthly_accum._max_count_ever == 0

    # A VALID feed first (establishes an nlev=NLEV profile bucket)...
    dc.feed_cmip_accumulators_native(
        day=15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"])
    _cnt = dc.monthly_accum._max_count_ever
    assert _cnt > 0
    # ...then a wrong-nlev T must be rejected pre-commit (count unchanged).
    with pytest.raises(ValueError, match="nlev"):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=np.zeros((n, NLEV + 1)), p_s=f["p_s"],
            lat_deg=f["lat_deg"])
    assert dc.monthly_accum._max_count_ever == _cnt

    # Non-numeric (object) T of the right shape is rejected too.
    with pytest.raises((ValueError, TypeError)):
        dc.feed_cmip_accumulators_native(
            day=15.0, T=np.full((n, NLEV), "x", dtype=object), p_s=f["p_s"],
            lat_deg=f["lat_deg"])
    assert dc.monthly_accum._max_count_ever == _cnt


# ---------------------------------------------------------------------------
# Clean-completion CMOR NetCDF write
# ---------------------------------------------------------------------------

def test_finalize_writes_cmor_netcdf(mesh, tmp_path):
    """The CMIP-file writers emit non-empty NetCDF from the fed accumulators
    (the clean-completion output the lean MPAS loop previously skipped)."""
    pytest.importorskip("netCDF4")
    dc, sigma_full, _ = _make_collector(mesh, output_dir=str(tmp_path))
    f = _synthetic_cell_fields(mesh, sigma_full)
    u_east, v_north = reconstruct_cell_velocity(f["u_edge"], mesh)
    # A handful of daily samples in one month so finalize keeps it.
    for d in (10.0, 11.0, 12.0):
        dc.feed_cmip_accumulators_native(
            day=d, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], q_v=f["q_v"],
            u_east=np.asarray(u_east), v_north=np.asarray(v_north),
            precip=f["precip"], phis=f["phis"])
    dc._write_cmip_monthly_files()
    dc._write_cmip_daily_files()
    dc._write_cmip_fixed_files()
    dc.cf_writer.close()
    ncs = list((tmp_path / "cmor").rglob("*.nc"))
    assert ncs, "no CMOR NetCDF written"
    assert all(p.stat().st_size > 0 for p in ncs)
    # tas (a core Amon var) must be among the emitted files.
    assert any("tas" in p.name for p in ncs)


def test_driver_finalize_mpas_cmip(mesh, tmp_path):
    """`ModelDriver._finalize_mpas_cmip` writes the CMOR NetCDF and marks the
    terminal sidecar suppressed (mirrors the compiled finalizer)."""
    pytest.importorskip("netCDF4")
    from legoesm.driver.model_driver import ModelDriver
    dc, sigma_full, _ = _make_collector(mesh, output_dir=str(tmp_path))
    f = _synthetic_cell_fields(mesh, sigma_full)
    for d in (10.0, 11.0):
        dc.feed_cmip_accumulators_native(
            day=d, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], precip=f["precip"])
    fake = types.SimpleNamespace(diagnostics=dc, _mpi_rank=None)
    ModelDriver._finalize_mpas_cmip(fake)
    assert fake._suppress_cmor_sidecar is True
    ncs = list((tmp_path / "cmor").rglob("*.nc"))
    assert ncs and all(p.stat().st_size > 0 for p in ncs)


def test_driver_finalize_retires_exact_cadence_sidecar(mesh, tmp_path):
    """Exact-checkpoint-cadence completion: a same-day sidecar already written
    by the in-loop periodic checkpoint is RETIRED after a successful finalize
    (else a run-extending restart would restore + re-append it)."""
    pytest.importorskip("netCDF4")
    from legoesm.driver.model_driver import ModelDriver
    dc, sigma_full, _ = _make_collector(mesh, output_dir=str(tmp_path))
    f = _synthetic_cell_fields(mesh, sigma_full)
    dc.feed_cmip_accumulators_native(
        day=25.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], precip=f["precip"])
    # Simulate the periodic checkpoint's same-day sidecar (day 25 -> 0025).
    stale = tmp_path / "cmor_accum_day_0025.npz"
    dc.save_cmor_accumulators(stale)
    assert stale.exists()

    fake = types.SimpleNamespace(
        diagnostics=dc, _mpi_rank=None, _output_dir=tmp_path)
    ModelDriver._finalize_mpas_cmip(fake, final_day=25.0)
    assert fake._suppress_cmor_sidecar is True
    assert not stale.exists(), "terminal sidecar was not retired"
    ncs = list((tmp_path / "cmor").rglob("*.nc"))
    assert ncs and all(p.stat().st_size > 0 for p in ncs)


def test_driver_finalize_mpas_cmip_noop_without_writer(mesh):
    """No CMIP writer (monthly-means-only) -> finalize is a guarded no-op."""
    from legoesm.driver.model_driver import ModelDriver
    dc, _, _ = _make_collector(mesh, monthly_means=True, cmip_output=False)
    assert dc.cf_writer is None
    fake = types.SimpleNamespace(diagnostics=dc, _mpi_rank=None)
    ModelDriver._finalize_mpas_cmip(fake)          # must not raise
    assert getattr(fake, "_suppress_cmor_sidecar", False) is False


def _stub_mpi_bcast(monkeypatch, value):
    """Stub mpi4py so the guard's cross-rank agreement runs without a launcher.

    `bcast` echoes ``value`` as rank 0's decision. Real mpi4py cannot be
    imported in a bare pytest process on every machine (libmpi is not on the
    loader path), and this test is about the AGREEMENT, not about MPI.
    """
    import sys

    mpi = types.ModuleType("mpi4py.MPI")
    mpi.COMM_WORLD = types.SimpleNamespace(bcast=lambda obj, root=0: value)
    pkg = types.ModuleType("mpi4py")
    pkg.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", pkg)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    return mpi


def test_multi_rank_cmor_request_is_refused_not_warned(monkeypatch):
    """#1545: asking for CMOR output on a lane that cannot produce it must
    FAIL BEFORE ANY TIME STEPPING, not complete and write empty files hours
    later. (Not at `setup()` — the call sits at the top of `_run_mpas`, before
    its first collective and before the loop, which is what costs GPU-hours.)

    Since #1517 the multi-rank Voronoi CELL PARTITION *is* feedable, so this
    exercises the guard directly with ``feed_on=False``: the layout it now
    stands for is a multi-rank run with NO Voronoi partition, not "multi-rank"
    in general.

    The refusal must fire on EVERY rank: the inputs are config/layout-derived
    and identical everywhere, so a rank-0-only raise would kill rank 0 and
    hang the rest at the next collective.
    """
    from legoesm.driver.model_driver import ModelDriver

    monkeypatch.delenv("LEGOESM_ALLOW_EMPTY_CMOR", raising=False)
    _stub_mpi_bcast(monkeypatch, False)
    for rank in (0, 1, 3):
        drv = types.SimpleNamespace(_mpi_world_size=4, _mpi_rank=rank)
        with pytest.raises(NotImplementedError, match="#1545"):
            ModelDriver._require_mpas_cmip_feed_supported(
                drv, feed_on=False, wants_cmip=True)


def test_supported_and_uninterested_runs_are_untouched(monkeypatch):
    """The guard must not fire when the feed works, nor when no CMOR output
    was asked for — otherwise every serial run breaks."""
    from legoesm.driver.model_driver import ModelDriver

    monkeypatch.delenv("LEGOESM_ALLOW_EMPTY_CMOR", raising=False)
    drv = types.SimpleNamespace(_mpi_world_size=4, _mpi_rank=0)
    ModelDriver._require_mpas_cmip_feed_supported(
        drv, feed_on=True, wants_cmip=True)      # serial / 1-rank: feeds
    ModelDriver._require_mpas_cmip_feed_supported(
        drv, feed_on=False, wants_cmip=False)    # no CMOR requested


@pytest.mark.parametrize("value,should_raise", [("1", False), ("0", True),
                                                ("yes", True)])
def test_empty_cmor_override_requires_exact_1(monkeypatch, value, should_raise):
    """Escape hatch matches the repo's other LEGOESM_ALLOW_* flags: exactly
    "1". A launcher exporting =0 must NOT silently re-open the trap."""
    from legoesm.driver.model_driver import ModelDriver

    monkeypatch.setenv("LEGOESM_ALLOW_EMPTY_CMOR", value)
    # Serial driver: exercises the env parsing itself, with no MPI in play
    # (the cross-rank agreement has its own test below).
    drv = types.SimpleNamespace(_mpi_world_size=2, _mpi_rank=None)
    if should_raise:
        with pytest.raises(NotImplementedError, match="#1545"):
            ModelDriver._require_mpas_cmip_feed_supported(
                drv, feed_on=False, wants_cmip=True)
    else:
        ModelDriver._require_mpas_cmip_feed_supported(
            drv, feed_on=False, wants_cmip=True)


def test_run_mpas_actually_calls_the_guard():
    """Name the symbol that RUNS: the guard is worthless if `_run_mpas` stops
    calling it. Fails if the call is deleted from the method that executes."""
    import inspect

    from legoesm.driver.model_driver import ModelDriver

    src = inspect.getsource(ModelDriver._run_mpas)
    assert "_require_mpas_cmip_feed_supported(" in src, (
        "_run_mpas no longer invokes the #1545 empty-CMOR refusal")


def test_override_is_agreed_across_ranks_not_read_per_rank(monkeypatch):
    """The env var is the one genuinely PER-PROCESS input (codex).

    An MPMD launcher exporting LEGOESM_ALLOW_EMPTY_CMOR to some ranks only
    would otherwise send those onward while the rest raise — a hang, strictly
    worse than the empty output being replaced. Rank 0's value must win
    everywhere. Here rank 0 says "no override", so this non-root rank must
    raise even though its OWN environment says otherwise.
    """
    from legoesm.driver.model_driver import ModelDriver

    monkeypatch.setenv("LEGOESM_ALLOW_EMPTY_CMOR", "1")   # this rank only
    mpi = _stub_mpi_bcast(monkeypatch, False)             # rank 0 says no
    drv = types.SimpleNamespace(_mpi_world_size=4, _mpi_rank=2)
    with pytest.raises(NotImplementedError, match="#1545"):
        ModelDriver._require_mpas_cmip_feed_supported(
            drv, feed_on=False, wants_cmip=True)

    # ...and the converse: rank 0 allows it, so this rank proceeds even though
    # its own environment never set the variable.
    monkeypatch.delenv("LEGOESM_ALLOW_EMPTY_CMOR", raising=False)
    mpi.COMM_WORLD = types.SimpleNamespace(bcast=lambda obj, root=0: True)
    ModelDriver._require_mpas_cmip_feed_supported(
        drv, feed_on=False, wants_cmip=True)


def test_wap_is_published_and_closes_continuity(mesh):
    """`wap` reaches the CMOR accumulator, and its global mean is ~0.

    A pressure vertical velocity built by integrating continuity must have
    zero area-weighted global mean at every level, by construction. That is
    exactly the check the diagnostic this replaces FAILED: estimating omega
    downstream from monthly-mean regridded winds gave a global mean of
    -6 hPa/day and amplitudes ~30x ERA5. A test that only asserted the field
    exists would not have caught that, so this asserts the invariant.
    """
    import types

    import numpy as np

    from legoesm.driver.model_driver import ModelDriver

    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    fake = types.SimpleNamespace(
        diagnostics=dc,
        grid=mesh,
        config=types.SimpleNamespace(T_ice=271.4),
        get_sst_sic=lambda day: (f["T"][:, -1], np.zeros(int(mesh.nCells))),
        state=types.SimpleNamespace(
            u=_field(f["u_edge"]), T=_field(f["T"]), p_s=_field(f["p_s"]),
            phis=_field(f["phis"]), tracers={"q_v": _field(f["q_v"])},
        ),
        model=types.SimpleNamespace(_sfc_diag=(None, None, _field(f["precip"]))),
    )
    _as_driver(fake)

    kw = fake._mpas_cmip_native_kwargs(15.0, dc)
    wap = kw["wap"]
    assert wap is not None, "the helper published no wap"
    wap = np.asarray(wap)
    assert wap.shape == (int(mesh.nCells), NLEV), wap.shape
    assert np.isfinite(wap).all()

    # Continuity closure on the native mesh, area weighted by cell area.
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    gm = (wap * area[:, None]).sum(axis=0) / area.sum()
    scale = np.abs(wap).mean()
    assert scale > 0.0, "wap is identically zero -- the wind field did not reach it"
    assert np.max(np.abs(gm)) < 1e-3 * scale, (
        f"wap does not close: worst level mean {np.max(np.abs(gm)):.3e} "
        f"against a typical magnitude of {scale:.3e}")

    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_3d_wap" in out, sorted(k for k in out if k.startswith("field_3d"))

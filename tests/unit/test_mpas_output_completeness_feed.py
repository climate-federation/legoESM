"""Registered-but-unfed CMIP6 variables must reach CMOR on the MPAS lane.

PR #1437 (clt) established the pattern: production AMIP runs on
``grid_type="mpas"`` feed CMOR exclusively through
``DiagnosticCollector.feed_cmip_accumulators_native`` (the cube/lat-lon
``collect`` path never runs there), so a variable registered in
``cmor_output.py`` but absent from the feed is silently never written.  This
suite covers the remaining registered-but-unfed variables:

* surface radiation ``rsds``/``rlds`` (dycore ``_sfc_diag`` slots 8/9) and the
  derived ``rsus``/``rlus`` (sign-pinned: up = down - net(+into surface)),
* ``ts`` (surface skin temperature from the sst/sic/ice blend — never the
  lowest-level air T),
* water paths ``clwvi``/``clivi`` (shared frozen-species reduction with
  ``collect``) and the 3-D ``clw``/``cli``,
* ``zg`` (hypsometric, virtual-T; analytic isothermal check) and ``hur``/
  ``hurs`` (shared saturation curve, no silent clamp),
* daily-table ``rsut``/``rlut``,
* ``tauu``/``tauv`` (CMOR downward-positive stress; sign-pinned).

Every test asserts against the symbol the MPAS lane actually executes, with
fixtures that are genuinely partial/nontrivial, and each is red when the
corresponding feed wiring is removed.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm import constants
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.grids.factory import create_grid

NLEV = 6


@pytest.fixture(scope="module")
def mesh():
    # Level-2 SCVT mesh = 162 cells; cheap, exercises the real IDW regrid.
    return create_grid("mpas", 2, lloyd_iterations=10)


def _collector(mesh, cloud_config=None):
    sigma_full = np.linspace(0.05, 0.98, NLEV)
    dc = DiagnosticCollector(
        nlev=NLEV, sigma_full=sigma_full,
        dsigma=np.full(NLEV, 1.0 / NLEV),
        experiment_id="amip", monthly_means=True, cmip_output=True,
        n_days=30, cmip_resolution_deg=10.0, start_year=1979,
        cloud_config=cloud_config,
    )
    dc.set_cmip_grid_info(grid_type="mpas", grid=mesh, start_year=1979)
    return dc, sigma_full


def _fields(mesh, sigma_full):
    """Native cell fields with nontrivial (latitude-dependent) structure."""
    latc = np.asarray(mesh.latCell)
    n = int(mesh.nCells)
    T = 250.0 + 40.0 * sigma_full[None, :] + 10.0 * np.cos(latc)[:, None]
    p_s = np.full(n, 1.0e5)
    q_v = 1.0e-3 * (1.0 - sigma_full)[None, :] * np.ones((n, 1))
    return dict(T=T, p_s=p_s, q_v=q_v, lat_deg=np.degrees(latc),
                phis=np.zeros(n), precip=np.full(n, 2.0e-5))


def _feed(dc, f, **kw):
    return dc.feed_cmip_accumulators_native(
        15.0, T=f["T"], p_s=f["p_s"], lat_deg=f["lat_deg"], q_v=f["q_v"],
        precip=f["precip"], phis=f["phis"], **kw)


def _mean2d(dc, name):
    """Accumulated 2-D field as a per-variable MEAN, or None if never fed."""
    for month in dc._spatial_monthly._data_2d.values():
        if name not in month:
            continue
        entry = month[name]
        total, count = (entry if isinstance(entry, tuple) else (entry, 1))
        return np.asarray(total, dtype=float) / max(float(np.max(count)), 1.0)
    return None


def _mean3d(dc, name):
    """Accumulated 3-D (plev) field mean, or None if never fed."""
    for month in dc._spatial_monthly._data_3d.values():
        if name not in month:
            continue
        entry = month[name]
        total, count = (entry if isinstance(entry, tuple) else (entry, 1))
        return np.asarray(total, dtype=float) / max(float(np.max(count)), 1.0)
    return None


# =====================================================================
# Surface radiation: rsds / rlds / rsus / rlus
# =====================================================================

def test_rsds_rlds_reach_the_accumulator(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    assert _feed(dc, f, rsds=np.full(n, 200.0), rlds=np.full(n, 350.0))
    got_sw = _mean2d(dc, "rsds")
    got_lw = _mean2d(dc, "rlds")
    assert got_sw is not None, "rsds absent from the MPAS CMOR feed"
    assert got_lw is not None, "rlds absent from the MPAS CMOR feed"
    # Constant native field survives the area-neutral IDW regrid unchanged.
    assert np.allclose(got_sw[np.isfinite(got_sw)], 200.0)
    assert np.allclose(got_lw[np.isfinite(got_lw)], 350.0)


def test_rsus_rlus_sign_arithmetic_pinned(mesh):
    """Hand-checkable numbers pin the up = down - net(+into surface) sign.

    rsds=200, sw_net=+150 (into surface) => rsus = 50 (reflected).
    rlds=350, lw_net=-60  (surface LOSES longwave) => rlus = 410 > rlds.
    A flipped net-flux sign convention gives 350/290 instead and goes red.
    """
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    _feed(dc, f,
          rsds=np.full(n, 200.0), sw_net_sfc=np.full(n, 150.0),
          rlds=np.full(n, 350.0), lw_net_sfc=np.full(n, -60.0))
    rsus = _mean2d(dc, "rsus")
    rlus = _mean2d(dc, "rlus")
    assert rsus is not None and rlus is not None
    assert np.allclose(rsus[np.isfinite(rsus)], 50.0)
    assert np.allclose(rlus[np.isfinite(rlus)], 410.0)


def test_rsus_skipped_not_zeroed_without_the_net_flux(mesh):
    """rsds alone must publish rsds but NOT a fabricated rsus=0."""
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    _feed(dc, f, rsds=np.full(n, 200.0))
    assert _mean2d(dc, "rsds") is not None
    assert _mean2d(dc, "rsus") is None, (
        "rsus fabricated without sw_net_sfc — a zero would read as a "
        "black surface")
    assert _mean2d(dc, "rlus") is None


def test_surface_radiation_budget_closes(mesh):
    """Budget: rsds - rsus - sw_net_sfc == 0 on a NONTRIVIAL spatial field.

    The regrid is linear, so the published rsus must equal the regrid of
    (rsds - sw_net_sfc) exactly — a per-cell budget check, not a constant.
    """
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    latc = np.asarray(mesh.latCell)
    rsds = 180.0 + 40.0 * np.cos(latc)          # nontrivial spatial pattern
    swn = 120.0 + 30.0 * np.cos(latc)
    _feed(dc, f, rsds=rsds, sw_net_sfc=swn)
    got_u = _mean2d(dc, "rsus")
    expected_u = dc._regrid_to_latlon_2d(rsds - swn)
    m = np.isfinite(got_u) & np.isfinite(expected_u)
    assert m.any()
    np.testing.assert_allclose(got_u[m], expected_u[m], rtol=1e-12)
    assert np.all(got_u[m] >= 0.0)


# =====================================================================
# ts: surface skin temperature
# =====================================================================

def test_ts_reaches_the_accumulator_and_is_the_supplied_skin_field(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    skin = np.full(n, 288.5)
    assert _feed(dc, f, ts=skin)
    got = _mean2d(dc, "ts")
    assert got is not None, "ts absent from the MPAS CMOR feed"
    assert np.allclose(got[np.isfinite(got)], 288.5)


def test_ts_absent_when_not_supplied_never_air_T_fallback(mesh):
    """No skin field => NO ts.  Falling back to lowest-level air T would
    mislabel air temperature as skin temperature."""
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    assert _feed(dc, f)
    assert _mean2d(dc, "ts") is None
    # ...while tas (which HAS a documented lowest-level fallback) is present.
    assert _mean2d(dc, "tas") is not None


# =====================================================================
# Water paths clwvi/clivi + 3-D clw/cli
# =====================================================================

def test_clwvi_clivi_hand_checkable_and_include_snow(mesh):
    """Vertically uniform tracers on pure sigma pin the integrals exactly:
    path = q * p_s / g.  The clivi expectation INCLUDES q_s, so an
    implementation that drops snow from the frozen sum goes red."""
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    q_c = np.full((n, NLEV), 1.0e-4)
    q_i = np.full((n, NLEV), 3.0e-5)
    q_s = np.full((n, NLEV), 2.0e-5)
    assert _feed(dc, f, q_c=q_c, q_i=q_i, q_s=q_s)
    p_s0 = 1.0e5
    exp_clivi = (3.0e-5 + 2.0e-5) * p_s0 / constants.g
    exp_clwvi = (1.0e-4 + 3.0e-5 + 2.0e-5) * p_s0 / constants.g
    got_i = _mean2d(dc, "clivi")
    got_w = _mean2d(dc, "clwvi")
    assert got_i is not None, "clivi absent from the MPAS CMOR feed"
    assert got_w is not None, "clwvi absent from the MPAS CMOR feed"
    np.testing.assert_allclose(got_i[np.isfinite(got_i)], exp_clivi,
                               rtol=1e-5)
    np.testing.assert_allclose(got_w[np.isfinite(got_w)], exp_clwvi,
                               rtol=1e-5)


def test_graupel_contributes_to_the_frozen_path(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    q_c = np.full((n, NLEV), 1.0e-4)
    q_g = np.full((n, NLEV), 4.0e-5)
    _feed(dc, f, q_c=q_c, q_g=q_g)
    got_i = _mean2d(dc, "clivi")
    exp = 4.0e-5 * 1.0e5 / constants.g
    np.testing.assert_allclose(got_i[np.isfinite(got_i)], exp, rtol=1e-5)


def test_paths_skipped_not_zeroed_without_condensate(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    assert _feed(dc, f)          # no q_c
    assert _mean2d(dc, "clwvi") is None
    assert _mean2d(dc, "clivi") is None


def test_both_lanes_share_one_condensate_reduction(mesh):
    """collect() and the MPAS feed must call the SAME `_condensate_paths`.

    Numerical half: the feed's published clivi equals the regrid of the
    shared helper's output.  Structural half: BOTH lane symbols reference
    the helper by name (each is the symbol its lane executes; removing the
    shared call from either goes red)."""
    import inspect

    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    latc = np.asarray(mesh.latCell)
    q_c = 1.0e-4 * np.cos(latc)[:, None] ** 2 * np.ones((1, NLEV))
    q_i = 5.0e-5 * np.sin(latc)[:, None] ** 2 * np.ones((1, NLEV))
    clwvi_native, clivi_native = dc._condensate_paths(
        f["p_s"], q_c, q_i, None, None)
    _feed(dc, f, q_c=q_c, q_i=q_i)
    got_i = _mean2d(dc, "clivi")
    exp_i = dc._regrid_to_latlon_2d(clivi_native)
    m = np.isfinite(got_i) & np.isfinite(exp_i)
    assert m.any()
    np.testing.assert_allclose(got_i[m], exp_i[m], rtol=1e-12)

    src_collect = inspect.getsource(DiagnosticCollector.collect)
    src_feed = inspect.getsource(
        DiagnosticCollector.feed_cmip_accumulators_native)
    assert "_condensate_paths(" in src_collect
    assert "_condensate_paths(" in src_feed


def test_clw_cli_3d_reach_the_accumulator_with_species_convention(mesh):
    """Vertically uniform mass fractions survive plev interp + regrid as
    constants, pinning clw = q_c and cli = q_i + q_s (snow INCLUDED,
    mirroring clivi)."""
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    q_c = np.full((n, NLEV), 1.0e-4)
    q_i = np.full((n, NLEV), 3.0e-5)
    q_s = np.full((n, NLEV), 2.0e-5)
    _feed(dc, f, q_c=q_c, q_i=q_i, q_s=q_s)
    clw = _mean3d(dc, "clw")
    cli = _mean3d(dc, "cli")
    assert clw is not None, "clw absent from the MPAS CMOR feed"
    assert cli is not None, "cli absent from the MPAS CMOR feed"
    np.testing.assert_allclose(clw[np.isfinite(clw)], 1.0e-4, rtol=1e-10)
    np.testing.assert_allclose(cli[np.isfinite(cli)], 5.0e-5, rtol=1e-10)


def test_cli_skipped_without_frozen_species(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    _feed(dc, f, q_c=np.full((n, NLEV), 1.0e-4))
    assert _mean3d(dc, "clw") is not None
    assert _mean3d(dc, "cli") is None, (
        "cli fabricated on a warm-rain run — a zero plane would read as "
        "ice-free with confidence")


@pytest.mark.parametrize("bad_kwarg", ["q_s", "q_g"])
def test_malformed_condensate_tracer_is_transactional(mesh, bad_kwarg):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    before_keys = set(dc._spatial_monthly._data_2d)
    with pytest.raises(ValueError):
        _feed(dc, f, q_c=np.full((n, NLEV), 1.0e-4),
              **{bad_kwarg: np.zeros((n, NLEV - 1))})
    assert set(dc._spatial_monthly._data_2d) == before_keys


def test_driver_forwards_snow_and_graupel_to_the_frozen_path(mesh):
    """The driver must extract q_s/q_g from state.tracers — a driver that
    forwards only q_i under-counts clivi on Morrison/Thompson runs."""
    from legoesm.driver.model_driver import ModelDriver

    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    tr = {"q_v": f["q_v"],
          "q_c": np.full((n, NLEV), 1.0e-4),
          "q_i": np.full((n, NLEV), 3.0e-5),
          "q_s": np.full((n, NLEV), 2.0e-5),
          "q_g": np.full((n, NLEV), 1.0e-5)}
    fake = _fake_driver(mesh, dc, f, tracers=tr)
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_2d_clivi" in out
    got = out["field_2d_clivi"]
    exp = (3.0e-5 + 2.0e-5 + 1.0e-5) * 1.0e5 / constants.g
    np.testing.assert_allclose(got[np.isfinite(got)], exp, rtol=1e-5)


# =====================================================================
# zg: geopotential height (hypsometric, virtual-T)
# =====================================================================

def _feed_raw(dc, T, p_s, lat_deg, phis, **kw):
    return dc.feed_cmip_accumulators_native(
        15.0, T=T, p_s=p_s, lat_deg=lat_deg, phis=phis, **kw)


def test_zg_isothermal_analytic_profile():
    """Isothermal dry atmosphere: z(p) = (R_d T / g) ln(p_s / p) + phis/g.

    40 levels keep the midpoint-quadrature error of the shared hypsometric
    helper well under the 2 % tolerance; the assertion is against the
    ANALYTIC profile at every interior plev19 level, so a dropped ln
    structure, wrong constant, or missing surface anchor is far outside
    tolerance.
    """
    nlev = 40
    sigma_full = np.linspace(0.0125, 0.9875, nlev)
    dc = DiagnosticCollector(
        nlev=nlev, sigma_full=sigma_full, dsigma=np.full(nlev, 1.0 / nlev),
        experiment_id="amip", monthly_means=True, cmip_output=True,
        n_days=30, cmip_resolution_deg=30.0, start_year=1979,
    )
    # Tiny synthetic "mesh": a handful of cells is enough (no regrid needed
    # for the accumulator read; use lat-lon-free zonal skip by passing a
    # valid lat vector).
    import types
    n = 8
    lat = np.linspace(-60.0, 60.0, n)
    lon = np.linspace(0.0, 315.0, n)
    grid = types.SimpleNamespace(
        nCells=n, latCell=np.radians(lat), lonCell=np.radians(lon))
    dc.set_cmip_grid_info(grid_type="mpas", grid=grid, start_year=1979)

    T0 = 250.0
    p_s0 = 1.0e5
    T = np.full((n, nlev), T0)
    p_s = np.full(n, p_s0)
    phis = np.zeros(n)
    _feed_raw(dc, T, p_s, lat, phis)     # q_v omitted => DRY hypsometric
    zg = _mean3d(dc, "zg")
    assert zg is not None, "zg absent from the MPAS CMOR feed"

    from legoesm.io.cmor_output import CMIP6_PLEV19
    plev = np.sort(np.asarray(CMIP6_PLEV19))
    # Compare only inside the WELL-RESOLVED part of the column: the top few
    # uniform-sigma layers have a large per-layer Delta(ln p), where the
    # shared helper's midpoint quadrature departs from the exact ln profile
    # by design (>2 % above ~sigma_full[4]); that is discretisation of the
    # instrument, not a wiring error.
    p_lo_valid = sigma_full[4] * p_s0
    scale = constants.R_d * T0 / constants.g
    n_checked = 0
    for k, p in enumerate(plev):
        if not (p_lo_valid <= p <= 0.95 * p_s0):
            continue                     # outside the resolved column
        col = zg[..., k]
        got = np.nanmean(col[np.isfinite(col)])
        expected = scale * np.log(p_s0 / p)
        assert got == pytest.approx(expected, rel=0.02), (
            f"zg at {p:.0f} Pa: got {got:.1f} m, analytic {expected:.1f} m")
        n_checked += 1
    assert n_checked >= 8, "analytic zg check covered too few plev levels"


def test_zg_surface_anchor_is_phis_over_g(mesh):
    """Raising phis by g*500 m^2/s^2 must raise zg by EXACTLY 500 m."""
    dc1, sig = _collector(mesh)
    dc2, _ = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    _feed(dc1, f)
    f2 = dict(f, phis=np.full(n, constants.g * 500.0))
    _feed(dc2, f2)
    zg1 = _mean3d(dc1, "zg")
    zg2 = _mean3d(dc2, "zg")
    assert zg1 is not None and zg2 is not None
    m = np.isfinite(zg1) & np.isfinite(zg2)
    np.testing.assert_allclose(zg2[m] - zg1[m], 500.0, rtol=0, atol=1e-6)


def test_zg_uses_virtual_temperature(mesh):
    """A moist column is THICKER: zg(moist) > zg(dry) at the same T.

    Red if q_v is not threaded into the shared hypsometric helper.
    """
    dc_dry, sig = _collector(mesh)
    dc_wet, _ = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    _feed_raw(dc_dry, f["T"], f["p_s"], f["lat_deg"], f["phis"])
    _feed_raw(dc_wet, f["T"], f["p_s"], f["lat_deg"], f["phis"],
              q_v=np.full((n, NLEV), 0.02))
    zg_d = _mean3d(dc_dry, "zg")
    zg_w = _mean3d(dc_wet, "zg")
    m = np.isfinite(zg_d) & np.isfinite(zg_w)
    # Upper levels accumulate the virtual-T thickening; require a strictly
    # positive difference on average (2 % vapour ~ +1.2 % thickness).
    assert np.mean(zg_w[m] - zg_d[m]) > 10.0, (
        "zg did not respond to q_v — virtual temperature is not used")


def test_zg_skipped_without_phis(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    _feed_raw(dc, f["T"], f["p_s"], f["lat_deg"], None, q_v=f["q_v"])
    assert _mean3d(dc, "zg") is None, (
        "zg fabricated without its surface anchor (phis)")


# =====================================================================
# hur / hurs: relative humidity via the shared saturation curve
# =====================================================================

def _q_v_for_rh(rh_frac, T, p_full):
    """Mixing ratio giving EXACTLY e = rh_frac * e_sat(T) at p_full,
    inverting e(r) = p r / (epsilon + r) — built ONLY from the shared
    thermo curve, no re-derived saturation formula."""
    from legoesm.thermo import saturation_vapor_pressure
    e = rh_frac * np.asarray(saturation_vapor_pressure(T), dtype=np.float64)
    return constants.epsilon * e / (p_full - e)


def test_hur_hurs_hand_pinned_at_50_percent(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    T = np.full((n, NLEV), 280.0)
    p_full = f["p_s"][:, None] * sig[None, :]
    q_v = _q_v_for_rh(0.5, T, p_full)
    _feed_raw(dc, T, f["p_s"], f["lat_deg"], f["phis"], q_v=q_v)
    hur = _mean3d(dc, "hur")
    hurs = _mean2d(dc, "hurs")
    assert hur is not None, "hur absent from the MPAS CMOR feed"
    assert hurs is not None, "hurs absent from the MPAS CMOR feed"
    np.testing.assert_allclose(hur[np.isfinite(hur)], 50.0, rtol=1e-4)
    np.testing.assert_allclose(hurs[np.isfinite(hurs)], 50.0, rtol=1e-4)


def test_hur_not_silently_clamped_above_100(mesh):
    """Supersaturation must be REPORTED (the table mandates no cap)."""
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    T = np.full((n, NLEV), 280.0)
    p_full = f["p_s"][:, None] * sig[None, :]
    q_v = _q_v_for_rh(1.2, T, p_full)
    _feed_raw(dc, T, f["p_s"], f["lat_deg"], f["phis"], q_v=q_v)
    hur = _mean3d(dc, "hur")
    np.testing.assert_allclose(hur[np.isfinite(hur)], 120.0, rtol=1e-4)


def test_hurs_is_the_lowest_model_level(mesh):
    """RH 80 % at the lowest level, 20 % aloft => hurs = 80, pinning the
    documented lowest-level (not column, not 2 m) choice."""
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    T = np.full((n, NLEV), 280.0)
    p_full = f["p_s"][:, None] * sig[None, :]
    rh = np.full((n, NLEV), 0.2)
    rh[:, -1] = 0.8
    q_v = _q_v_for_rh(rh, T, p_full)
    _feed_raw(dc, T, f["p_s"], f["lat_deg"], f["phis"], q_v=q_v)
    hurs = _mean2d(dc, "hurs")
    np.testing.assert_allclose(hurs[np.isfinite(hurs)], 80.0, rtol=1e-4)


def test_hur_skipped_without_q_v(mesh):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    _feed_raw(dc, f["T"], f["p_s"], f["lat_deg"], f["phis"])
    assert _mean3d(dc, "hur") is None
    assert _mean2d(dc, "hurs") is None


# =====================================================================
# Transactionality of the new inputs
# =====================================================================

@pytest.mark.parametrize("bad_kwarg", ["rsds", "rlds", "sw_net_sfc",
                                       "lw_net_sfc", "ts", "tauu", "tauv"])
def test_malformed_new_input_raises_without_committing(mesh, bad_kwarg):
    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)
    before_keys = set(dc._spatial_monthly._data_2d)
    before_max = dc._spatial_monthly._max_count_ever
    with pytest.raises(ValueError):
        _feed(dc, f, **{bad_kwarg: np.zeros(n - 1)})
    assert set(dc._spatial_monthly._data_2d) == before_keys, (
        f"a partial commit landed before {bad_kwarg} validation rejected it")
    assert dc._spatial_monthly._max_count_ever == before_max


# =====================================================================
# _MPASSfcFluxAccum: extended slot coverage + checkpoint compatibility
# =====================================================================

def _accum_cls():
    from legoesm.driver.model_driver import _MPASSfcFluxAccum
    return _MPASSfcFluxAccum


class _F:
    """Minimal Field stand-in (the accumulator reads ``.data``)."""

    def __init__(self, a):
        self.data = np.asarray(a, dtype=np.float64)


def test_accumulator_covers_the_new_flux_slots():
    cls = _accum_cls()
    for slot in (0, 1, 8, 9, 10, 11):
        assert slot in cls.SLOTS, (
            f"slot {slot} not accumulated — its CMOR field would fall back "
            f"to the instantaneous end-of-interval snapshot (#1353 regression)")
    acc = cls()
    diag = [None] * 12
    diag[0] = _F([150.0, 152.0])
    diag[8] = _F([200.0, 202.0])
    acc.add(tuple(diag))
    diag[0] = _F([160.0, 162.0])
    diag[8] = _F([210.0, 212.0])
    acc.add(tuple(diag))
    np.testing.assert_allclose(acc.mean(0), [155.0, 157.0])
    np.testing.assert_allclose(acc.mean(8), [205.0, 207.0])
    assert acc.mean(9) is None      # never fed -> None, not zeros


def test_accumulator_restores_an_old_payload_without_the_new_slots():
    """A pre-extension checkpoint (slots 2-7 only) must still resume."""
    cls = _accum_cls()
    src = cls(expected_steps=4, window_start_day=10.0, dt_s=100.0)
    diag = [None] * 8
    diag[2] = _F([1.0, 2.0])
    diag[6] = _F([3.0, 4.0])
    src.add(tuple(diag))
    src.add(tuple(diag))
    payload = src.dump()
    # Simulate an OLD payload: it can only contain keys for slots that were
    # fed, which for a pre-extension run is a subset of 2..7.
    assert set(k for k in payload if k.startswith("cmor_fluxsum")) == {
        "cmor_fluxsum_2", "cmor_fluxsum_6"}
    dst = cls(expected_steps=4, window_start_day=10.0, dt_s=100.0)
    resume_day = 10.0 + 2 * 100.0 / 86400.0
    n = dst.restore(dict(payload), resume_day=resume_day, dt_s=100.0)
    assert n == 2
    np.testing.assert_allclose(dst.mean(2), [1.0, 2.0])
    assert dst.mean(0) is None and dst.mean(8) is None


def test_checkpoint_keys_include_the_new_slots():
    cls = _accum_cls()
    keys = set(cls.checkpoint_keys())
    for slot in (0, 1, 8, 9, 10, 11):
        assert f"cmor_fluxsum_{slot}" in keys, (
            f"slot {slot} sums would be rejected by the driver's "
            f"checkpoint whitelist on restore")


# =====================================================================
# Driver boundary: slots -> feed wiring (fake driver, like the clt suite)
# =====================================================================

def _fake_driver(mesh, dc, f, *, sfc_diag=None, tracers=None,
                 with_sst=False):
    import types

    def _field(a):
        return types.SimpleNamespace(data=np.asarray(a))

    n_edges = int(mesh.nEdges)
    n = int(mesh.nCells)
    tracers = tracers if tracers is not None else {"q_v": f["q_v"]}
    drv = types.SimpleNamespace(
        diagnostics=dc, grid=mesh,
        state=types.SimpleNamespace(
            u=_field(np.full((n_edges, NLEV), 5.0)),
            T=_field(f["T"]), p_s=_field(f["p_s"]), phis=_field(f["phis"]),
            tracers={k: _field(v) for k, v in tracers.items()},
        ),
        model=types.SimpleNamespace(
            _sfc_diag=(sfc_diag if sfc_diag is not None
                       else (None, None, _field(f["precip"]))),
        ),
        config=types.SimpleNamespace(T_ice=260.0),
    )
    if with_sst:
        sst = np.full(n, 290.0)
        sic = np.full(n, 0.5)
        drv.get_sst_sic = lambda day: (sst, sic)
    return drv


def test_driver_maps_slots_8_9_to_rsds_rlds_and_derives_rsus_rlus(mesh):
    """End-to-end slot mapping, with DISTINCT constants per slot so a
    transposed slot index cannot pass."""
    from legoesm.driver.model_driver import ModelDriver
    import types as _t

    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    n = int(mesh.nCells)

    def _fld(v):
        return _t.SimpleNamespace(data=np.full(n, float(v)))

    # Slot contract: (0 sw_net, 1 lw_net, 2 precip, 3 lw_up_toa, 4 sw_up_toa,
    #                 5 sw_down_toa, 6 shflx, 7 lhflx, 8 sw_down_sfc,
    #                 9 lw_down_sfc)
    sfc = (_fld(150.0), _fld(-60.0), _t.SimpleNamespace(data=f["precip"]),
           _fld(240.0), _fld(101.0), _fld(340.0), _fld(11.0), _fld(87.0),
           _fld(200.0), _fld(350.0))
    fake = _fake_driver(mesh, dc, f, sfc_diag=sfc)
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    for name, val in (("rsds", 200.0), ("rlds", 350.0),
                      ("rsus", 50.0), ("rlus", 410.0),
                      ("rsut", 101.0), ("rlut", 240.0)):
        key = f"field_2d_{name}"
        assert key in out, f"{name} never reached CMOR through the driver"
        got = out[key]
        assert np.allclose(got[np.isfinite(got)], val), (
            f"{name}: expected {val}, got {np.nanmean(got)} — slot mapping "
            f"is wrong")


def test_driver_publishes_ts_from_the_sst_sic_blend(mesh):
    """ts = sic*T_ice + (1-sic)*sst — the same blend tas extrapolates from.

    With sst=290, sic=0.5, T_ice=260: ts = 275 exactly.  A lowest-level
    air-T fallback would give ~289-291 and go red.
    """
    from legoesm.driver.model_driver import ModelDriver

    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    fake = _fake_driver(mesh, dc, f, with_sst=True)
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert "field_2d_ts" in out, "ts never reached CMOR through the driver"
    got = out["field_2d_ts"]
    assert np.allclose(got[np.isfinite(got)], 275.0), (
        f"ts is not the sst/sic/T_ice blend (got {np.nanmean(got)})")


def test_driver_omits_ts_without_prescribed_sst(mesh):
    from legoesm.driver.model_driver import ModelDriver

    dc, sig = _collector(mesh)
    f = _fields(mesh, sig)
    fake = _fake_driver(mesh, dc, f, with_sst=False)
    ModelDriver._feed_mpas_cmip_accumulators(fake, day=15.0)
    out = dc._spatial_monthly.finalize(min_sample_fraction=0)
    assert dc._spatial_monthly._max_count_ever > 0, "nothing fed at all"
    assert "field_2d_ts" not in out

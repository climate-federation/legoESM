"""Cube rows of the DCMIP 2012 1-1 / 1-2 transport tests through FV3 duo's
own tracer transport (``scripts/matrix/_cube_duo_adapters.
build_duo_dcmip_transport``; user decisions Q5-A adapter only, Q6-a
implicit vertical motion, Q7-b evolving p_s, Q8-a DCMIP L30 table).

RULE 0: the flux-form duo transport + FV3 remap conserves the global
tracer mass ``sum(q delp area)`` and the air mass ``sum(delp area)`` to
round-off -- these tests fail if it does not.  Run with JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# fp64-only duo lane: the transport tests run with JAX_ENABLE_X64=1 (no
# collection-time precision change here); the table / geometry checks run
# in any configuration
x64 = pytest.mark.skipif(not jax.config.jax_enable_x64,
                         reason="the duo lane is fp64-only")

from legoesm.grids.vertical import (  # noqa: E402
    dcmip2012_transport_eta_table, remap_tracers_fv3_pressure)
from tests.test_cases import dcmip_transport as D  # noqa: E402

N = 12


def _build(tn, *, dt=None, n_sub=1, n=N):
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip_sigma(30)
    return build_duo_dcmip_transport(
        n, wind_geo={11: D.dcmip11_wind_geo, 12: D.dcmip12_wind_geo}[tn],
        init_latlon={11: D.dcmip11_init_latlon,
                     12: D.dcmip12_init_latlon}[tn],
        sigma_coord=sc, n_tracers={11: 4, 12: 1}[tn],
        dt={11: 1800.0, 12: 600.0}[tn] if dt is None else dt, n_sub=n_sub)


def _masses(r, c):
    dp = r.layer_mass(c)
    q = r.tracers(c)
    return (np.einsum("fijk,fijkt,fij->t", dp, q, r.area),
            float(np.einsum("fijk,fij->", dp, r.area)))


@x64
@pytest.mark.parametrize("tn,n_sub", [(12, 1), (11, 2)])
def test_duo_transport_conserves_tracer_and_air_mass(tn, n_sub):
    """Global sum(q delp area) per tracer and sum(delp area) to round-off
    over 3 steps; non-vacuous: tracers and p_s really change."""
    r = _build(tn, n_sub=n_sub)
    c = r.carry
    m0, a0 = _masses(r, c)
    q0, ps0 = r.tracers(c), r.ps(c)
    dt = {11: 1800.0, 12: 600.0}[tn]
    for _ in range(3):
        c = r.step(c, dt)
    m1, a1 = _masses(r, c)
    np.testing.assert_allclose(m1, m0, rtol=1e-13, atol=0)
    np.testing.assert_allclose(a1, a0, rtol=1e-13, atol=0)
    assert float(np.max(np.abs(r.tracers(c) - q0))) > 1e-3
    assert float(np.max(np.abs(r.ps(c) - ps0))) > 1e-2


@x64
def test_duo_transport_preserves_uniform_tracer():
    """q = const stays const (consistent mass fluxes for delp and q*delp,
    seam-blended dp2, constant-preserving remap)."""
    r = _build(11, n_sub=2)
    (delp, q), t = r.carry
    sl = slice(3, 3 + N)
    q = q.at[:, :, sl, sl, :].set(0.37)
    c = ((delp, q), t)
    # 4 steps: from step 2 on, delp is horizontally non-uniform (p_s
    # evolves), so the tracer kernel's dp2 must equal the dycore's delp
    # layer by layer (incl. seam cells) or q drifts from 0.37
    for _ in range(4):
        c = r.step(c, 1800.0)
    assert float(np.ptp(r.ps(c))) > 1.0
    np.testing.assert_allclose(r.tracers(c), 0.37, rtol=0, atol=1e-13)


@x64
def test_duo_transport_carries_tracer_across_cube_seams():
    """A tracer confined to face 0 leaves it through the seams (donor loses,
    receivers gain) under 1-1 winds with 2 sub-steps, the global tracer
    mass is conserved to round-off, and nothing blows up at the seams."""
    r = _build(11, n_sub=2)
    (delp, q), t = r.carry
    sl = slice(3, 3 + N)
    q = jnp.zeros_like(q).at[0, :, sl, sl, :].set(1.0)
    c = ((delp, q), t)
    face_mass = lambda c: np.einsum(  # noqa: E731
        "fijk,fijkt,fij->ft", r.layer_mass(c), r.tracers(c), r.area)
    f0 = face_mass(c)
    for _ in range(3):
        c = r.step(c, 1800.0)
    f1 = face_mass(c)
    assert np.all(f1[0] < f0[0] * (1 - 1e-3))            # donor loses
    assert np.all(f1[1:].sum(0) > 1e-3 * f0[0])          # receivers gain
    assert np.all(np.sum(f1[1:] > 1e-6 * f0[0], axis=0) >= 2)
    np.testing.assert_allclose(f1.sum(0), f0.sum(0), rtol=1e-13, atol=0)
    # hord_tr = 6 (the duo deck) is not monotone: a 0/1 step gives a ~7 %
    # undershoot (measured -0.072); this bound catches a seam blow-up only
    assert float(r.tracers(c).min()) > -0.1


@x64
def test_auto_substeps_follow_resolution():
    """n_sub=None chooses the sub-step count from the grid, dt and winds:
    more sub-steps on a finer cube, and the realised Courant stays < 1."""
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip_sigma(30)
    subs = []
    for n in (12, 24):
        r = build_duo_dcmip_transport(
            n, wind_geo=D.dcmip11_wind_geo, init_latlon=D.dcmip11_init_latlon,
            sigma_coord=sc, n_tracers=4, dt=1800.0 * 4, n_sub=None,
            period_s=12 * 86400.0)
        subs.append(r.n_sub_fn())
        r.step(r.carry, 1800.0 * 4)
        assert r.courant_max() < 1.0
    assert subs[1] > subs[0] >= 1, subs


@x64
def test_duo_transport_refuses_courant_above_one():
    r = _build(11, dt=6 * 1800.0, n_sub=1)
    with pytest.raises(RuntimeError, match="Courant") as e:
        r.step(r.carry, 6 * 1800.0)
    m = re.search(r"Courant ([0-9.]+), invalid intermediate delp=(\w+)",
                  str(e.value))
    assert m and float(m.group(1)) > 1.0, str(e.value)


@x64
def test_effective_omega_tracks_analytic():
    """The implicit (Lagrangian) vertical motion matches the analytic DCMIP
    p0*sigma_dot it never uses (1-2 Hadley cell, C12)."""
    r = _build(12)
    assert r.effective_omega() == {}
    c = r.step(r.carry, 600.0)
    c = r.step(c, 600.0)
    om = r.effective_omega()
    for o in (om["first"], om["last"]):
        assert o["corr"] > 0.999 and o["rel_rms_err"] < 0.05, om
        assert np.isfinite(o["abs_rms_err"]) and o["abs_rms_err"] >= 0
    # the advection correction is computed and applied end to end: nonzero
    # once p_s varies (second step), and it changes the reported error
    last = om["last"]
    assert last["rms_interface_advection"] > 0
    assert last["abs_rms_err"] != last["abs_rms_err_fixed_cell"]
    assert om["first"]["t_sample"] == 300.0 and om["last"]["t_sample"] == 900.0


def test_dcmip_l30_table_and_l32_default_untouched():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, duo_eta_table)
    from scripts.matrix import _cube_duo_adapters as A
    ak, bk, ptop = duo_eta_table("dcmip2012_l30", 30)
    a2, b2, p2 = dcmip2012_transport_eta_table(30)
    assert np.array_equal(ak, a2) and np.array_equal(bk, b2) and ptop == p2
    sig = np.asarray(D.create_dcmip_sigma(30).sigma_half, np.float64)
    np.testing.assert_allclose((ak + bk * 1e5) / 1e5, sig, rtol=0,
                               atol=1e-6)
    assert ak[0] == ptop and bk[0] == 0.0 and bk[-1] == 1.0 and ak[-1] == 0.0
    with pytest.raises(ValueError, match="30-level"):
        duo_eta_table("dcmip2012_l30", 32)
    # defaults untouched: the dycore config default and the hydro rows' L32
    assert FV3DuoConfig().eta == "analytic"
    assert A.DUO_HYDRO_ETA == "cam6_l32" and A.DUO_HYDRO_KM == 32


@x64
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_remap_tracers_fv3_pressure_budget_and_refusals(dtype):
    rng = np.random.default_rng(0)
    ak, bk, ptop = dcmip2012_transport_eta_table(30)
    ps1 = 1e5 + rng.uniform(-200, 200, (4, 3))
    pe2 = ak + bk * ps1[..., None]
    dp = np.diff(pe2, axis=-1) * (1 + 0.3 * rng.uniform(-1, 1, (4, 3, 30)))
    dp *= (ps1 - ptop)[..., None] / dp.sum(-1, keepdims=True)
    pe1 = np.concatenate([np.full((4, 3, 1), ptop),
                          ptop + np.cumsum(dp, -1)], -1)
    pe1[..., -1] = ps1
    pe2[..., 0], pe2[..., -1] = ptop, ps1
    q = rng.uniform(0, 1, (4, 3, 30, 2))
    q2 = remap_tracers_fv3_pressure(jnp.asarray(q, dtype), pe1, pe2)
    before = np.sum(q * np.diff(pe1, axis=-1)[..., None], -2)
    after = np.sum(np.asarray(q2) * np.diff(pe2, axis=-1)[..., None], -2)
    np.testing.assert_allclose(after, before, rtol=1e-14, atol=0)
    bad = pe2.copy()
    bad[..., -1] += 1.0
    with pytest.raises(Exception, match="same pressure range"):
        jax.block_until_ready(remap_tracers_fv3_pressure(q, pe1, bad))
    cross = pe1.copy()
    cross[..., 5] = cross[..., 7]
    with pytest.raises(Exception, match="non-positive Lagrangian"):
        jax.block_until_ready(remap_tracers_fv3_pressure(q, cross, pe2))


@x64
@pytest.mark.parametrize("tn,dt,ntr", [(11, 1800.0, 4)])
def test_matrix_runner_cube_11_uses_duo_and_reports_budgets(tmp_path, tn, dt,
                                                           ntr):
    from scripts.matrix.run_atmosphere_test_matrix import (
        TestCase as MatrixCase, run_dcmip_transport)
    case = MatrixCase("hydrostatic", f"dcmip_transport_{tn}", "cubed_sphere",
                      f"C{N}", "sigma", 12, 1, {"test_num": tn})
    # a duration that is NOT a multiple of dt: 2.5 dt -> 2 accepted steps
    status, _, notes = run_dcmip_transport(case, tmp_path, 2.5 * dt / 86400)
    assert status == "PASS", notes
    fields = dict(line.split(": ", 1) for line in
                  (tmp_path / "results.txt").read_text().splitlines()
                  if ": " in line)
    assert int(fields["accepted_steps"]) == 2, fields
    assert np.isclose(float(fields["elapsed_days"]), 2 * dt / 86400,
                      rtol=1e-12), fields
    text = (tmp_path / "results.txt").read_text()
    assert "vertical_scheme: fv3_duo Lagrangian" in text, text
    rows = np.atleast_2d(np.loadtxt(tmp_path / "mean_timeseries.csv",
                                    delimiter=",", skiprows=1))
    assert int(rows[-1, 0]) == 2 and np.isclose(rows[-1, 1] * 86400.0,
                                                2 * dt, rtol=1e-6), rows
    assert abs(rows[-1, 2] - rows[0, 2]) > 1e-8 or \
        abs(rows[-1, 3] - rows[0, 3]) > 1e-8, rows               # moved
    rels = [float(x) for x in re.findall(
        r"q\d M0=\S+ M1=\S+ rel=([-+0-9.eE]+)", notes)]
    assert len(rels) == ntr and max(abs(x) for x in rels) < 1e-13, notes
    air = float(re.search(r"air-mass rel ([-+0-9.eE]+)", notes).group(1))
    assert abs(air) < 1e-13, notes


@x64
def test_matrix_runner_cube_13_keeps_legacy_route(tmp_path):
    from scripts.matrix.run_atmosphere_test_matrix import (
        TestCase as MatrixCase, run_dcmip_transport)
    case = MatrixCase("hydrostatic", "dcmip_transport_13", "cubed_sphere",
                      f"C{N}", "sigma", 12, 1, {"test_num": 13})
    status, _, notes = run_dcmip_transport(case, tmp_path, 2 * 1800 / 86400)
    assert status == "PASS", notes
    text = (tmp_path / "results.txt").read_text()
    assert "vertical_scheme: upwind (legacy cube model)" in text, text
    assert "omega_eff" not in notes


@x64
def test_matrix_runner_cube_rows_use_duo(tmp_path):
    """run_dcmip_transport's cube 1-2 row: duo transport, the vertical
    label, a round-off mass audit, p_s and omega diagnostics."""
    from scripts.matrix.run_atmosphere_test_matrix import (
        TestCase as MatrixCase, run_dcmip_transport)
    case = MatrixCase("hydrostatic", "dcmip_transport_12", "cubed_sphere",
                      f"C{N}", "sigma", 12, 1, {"test_num": 12})
    status, _, notes = run_dcmip_transport(case, tmp_path, 3 * 600.0 / 86400)
    assert status == "PASS", notes
    text = (tmp_path / "results.txt").read_text()
    assert "vertical_scheme: fv3_duo Lagrangian" in text, text
    rel = float(re.search(r"q1 M0=\S+ M1=\S+ rel=([-+0-9.eE]+)",
                          notes).group(1))
    assert abs(rel) < 1e-13, notes
    air = float(re.search(r"air-mass rel ([-+0-9.eE]+)", notes).group(1))
    assert abs(air) < 1e-13, notes
    assert "omega_eff" in notes and "p_s-p0" in notes
    # really stepped: 3 steps to t = 1800 s, and the tracer moved
    rows = np.atleast_2d(np.loadtxt(tmp_path / "mean_timeseries.csv",
                                    delimiter=",", skiprows=1))
    assert int(rows[-1, 0]) == 3 and np.isclose(rows[-1, 1] * 86400.0,
                                                1800.0, rtol=1e-6), rows
    assert abs(rows[-1, 2] - rows[0, 2]) > 1e-6, rows       # q1_min moved


@x64
@pytest.mark.parametrize("time_dependent", [False, True])
def test_blob_crosses_seam_to_the_right_place(time_dependent):
    """Seam orientation: solid-body zonal rotation (sigma_dot = 0) carries
    a Gaussian blob centred at (20E, 20N) eastward across an equatorial-
    face cube seam.  The tracer-mass centroid must land at the analytic
    position (same latitude, analytic longitude) with its mass in the
    north -- a reversed or misrouted seam puts the mass elsewhere."""
    from types import SimpleNamespace  # noqa: F401
    from legoesm.core.field import Field
    from legoesm.core.state import TracerState
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip_sigma(30)
    km = 30
    period = 12 * 86400.0
    u0 = 2 * np.pi * FV3_RADIUS_M / period
    lon0 = np.radians(20.0)
    lat0 = np.radians(20.0)          # off-equator: north/south asymmetric

    # time-dependent case: u0 cos(pi t / 6 d) -> displacement
    # (u0/a)(6 d/pi) sin(pi t/6 d) = 1 rad at t = 3 d (a frozen or
    # wrongly-timed wind gives 90 deg or another value)
    tp = 6 * 86400.0

    def wind(t, lon, lat, s_):
        f = jnp.cos(jnp.pi * t / tp) if time_dependent else 1.0
        u = f * u0 * jnp.cos(lat)[..., None] * jnp.ones(km)
        return u, 0.0 * u, jnp.zeros(lat.shape + (km + 1,))

    def init(cells, s_):
        lo, la = cells.lon2d, cells.lat2d
        d = jnp.arccos(jnp.clip(jnp.sin(la) * np.sin(lat0) + jnp.cos(la)
                                * np.cos(lat0) * jnp.cos(lo - lon0), -1, 1))
        q = jnp.exp(-(d / 0.25) ** 2)[..., None, None] * jnp.ones((km, 1))
        return TracerState(
            tracers=Field(data=q, name="tracers",
                          dims=("lat", "lon", "level", "tracer"), units="1"),
            time=Field(data=jnp.asarray(0.0), name="time", dims=(),
                       units="s"))

    r = build_duo_dcmip_transport(
        N, wind_geo=wind, init_latlon=init, sigma_coord=sc, n_tracers=1,
        dt=3600.0, n_sub=None, period_s=period)
    c = r.carry
    for _ in range(72):                       # 3 days = a quarter turn
        c = r.step(c, 3600.0)
    w = np.einsum("fijk,fijk,fij->fij", r.layer_mass(c),
                  r.tracers(c)[..., 0], r.area)
    lo, la = np.radians(r.lon_deg), np.radians(r.lat_deg)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                    np.sin(la)], -1)
    cen = np.einsum("fij,fijx->x", w, xyz)
    clon = np.degrees(np.arctan2(cen[1], cen[0]))
    clat = np.degrees(np.arcsin(cen[2] / np.linalg.norm(cen)))
    want = 20.0 + (np.degrees(1.0) if time_dependent else 90.0)
    # solid-body zonal rotation keeps latitude: an along-seam index reversal
    # would mirror the blob to ~20 S
    assert abs(clon - want) < 3.0 and abs(clat - 20.0) < 2.0, (clon, clat,
                                                                 want)
    # receiving-side distribution: northern mass dominates
    north = np.sum(w[r.lat_deg > 0])
    assert north > 0.95 * np.sum(w), north / np.sum(w)


@x64
def test_auto_substeps_retry_matches_direct_step():
    """Automatic mode: an initial estimate that is too small (forced via a
    huge courant_target) is retried with more sub-steps BEFORE the remap,
    and the accepted state equals a direct step at that sub-step count."""
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip_sigma(30)
    kw = dict(wind_geo=D.dcmip11_wind_geo, init_latlon=D.dcmip11_init_latlon,
              sigma_coord=sc, n_tracers=4, dt=4 * 1800.0)
    r = build_duo_dcmip_transport(N, n_sub=None, period_s=12 * 86400.0,
                                  courant_target=50.0, **kw)
    assert r.n_sub_fn() == 1
    c = r.step(r.carry, 4 * 1800.0)
    ns = r.n_sub_fn()
    assert ns > 1 and r.courant_max() < 1.0
    d = build_duo_dcmip_transport(N, n_sub=ns, **kw)
    c2 = d.step(d.carry, 4 * 1800.0)
    np.testing.assert_array_equal(r.tracers(c), d.tracers(c2))
    np.testing.assert_array_equal(r.layer_mass(c), d.layer_mass(c2))


@x64
def test_dcmip11_full_period_return_and_budgets():
    """Full 12-day 1-1 deformation and return (C12, dt 1800 s): tracer and
    air masses to round-off over 576 steps, and the q1 return error at
    t = T (normalised l2 vs the initial field) below 0.9 at this coarse
    resolution (C36 measures 0.52)."""
    r = _build(11)
    c = r.carry
    m0, a0 = _masses(r, c)
    for i in range(576):
        c = r.step(c, 1800.0)
        if i == 287:                         # t = T/2: strongly deformed
            qh = r.tracers(c)[..., 0]
            w = r.area[..., None] * np.ones_like(qh)
            q0 = r.q_ic[..., 0]
            assert np.sqrt(np.sum(w * (qh - q0) ** 2)
                           / np.sum(w * q0 ** 2)) > 0.3
    m1, a1 = _masses(r, c)
    np.testing.assert_allclose(m1, m0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(a1, a0, rtol=1e-12, atol=0)
    q, q0 = r.tracers(c)[..., 0], r.q_ic[..., 0]
    w = r.area[..., None] * np.ones_like(q)
    l2 = np.sqrt(np.sum(w * (q - q0) ** 2) / np.sum(w * q0 ** 2))
    assert l2 < 0.9, l2


@x64
def test_random_tracer_conserved_cellwise_consistency():
    """AGGREGATE conservation check on the single-subcycle (nsplt == 1,
    asserted) path: a white-noise tracer over 3 steps with 2 sub-steps; the
    kernel conserves sum(q dp2_kernel), the remap sum(q delp_lag), so a
    dp2_kernel / delp_lag mismatch shows unless it cancels against the
    random weights (a global, not a cellwise, check; the cellwise check of
    the same dp2 code path is the nsplt > 1 test below)."""
    r = _build(11, n_sub=2)
    (delp, q), t = r.carry
    sl = slice(3, 3 + N)
    rng = np.random.default_rng(3)
    q = q.at[:, :, sl, sl, :].set(
        jnp.asarray(rng.uniform(0.5, 1.5, (6, 4, N, N, 30))))
    c = ((delp, q), t)
    m0, _ = _masses(r, c)
    for _ in range(3):
        c = r.step(c, 1800.0)
        assert np.all(np.asarray(r.last_phase["tr"]["nsplt"]) == 1)
    m1, _ = _masses(r, c)
    np.testing.assert_allclose(m1, m0, rtol=1e-13, atol=0)


def test_tangent_gradient_sixface_known_field():
    """The omega diagnostic's gradient helper on a manufactured field with a
    known gradient and sign: f = a sin(lat) has grad f = cos(lat) e_north
    (per m, times a), so V . grad f = v cos(lat) for V = (u, v)."""
    from scripts.matrix._cube_duo_adapters import tangent_gradient_sixface
    n = 24
    # equal-angle cube cell centres (gradient helper is grid-agnostic)
    a = np.radians(np.linspace(-45, 45, n + 1))
    am = 0.5 * (a[1:] + a[:-1])
    x, y = np.meshgrid(np.tan(am), np.tan(am), indexing="ij")
    one = np.ones_like(x)
    faces = [(one, x, y), (-x, one, y), (-one, -x, y), (x, -one, y),
             (-y, x, one), (y, x, -one)]
    X = np.stack([np.stack(f, -1) for f in faces])
    X /= np.linalg.norm(X, axis=-1, keepdims=True)
    lat = np.arcsin(X[..., 2])
    lon = np.arctan2(X[..., 1], X[..., 0])
    R = 6.371e6
    g = tangent_gradient_sixface(R * np.sin(lat), X, R)
    en = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                   np.cos(lat)], -1)
    want = np.cos(lat)[..., None] * en
    inner = (slice(None), slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(g[inner], want[inner], rtol=0, atol=3e-3)
    # sign: northward wind up the gradient -> positive advection
    assert np.all(np.einsum("fijx,fijx->fij", g, en)[inner] > 0)
    # the omega diagnostic's signed correction on a manufactured case:
    # pe(k) = P_k + A sin(lat) on 3 interfaces, winds u = U, v = V on 4
    # levels -> V . grad(pe) = V * A cos(lat) / R exactly (u drops out)
    from scripts.matrix._cube_duo_adapters import (
        interface_pressure_advection)
    A, U, V = 300.0, 25.0, -7.0
    pe = (np.array([2e4, 5e4, 8e4]) + A * np.sin(lat)[..., None])
    u = np.full(lat.shape + (4,), U)
    v = np.full(lat.shape + (4,), V)
    ee = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    adv = interface_pressure_advection(u, v, pe, X, ee, en, R)
    want = V * A * np.cos(lat)[..., None] / R * np.ones(3)
    np.testing.assert_allclose(adv[inner], want[inner], rtol=0,
                               atol=3e-3 * abs(V) * A / R)
    assert np.all(adv[inner] < 0)            # V < 0, A > 0: negative


@x64
def test_tracer_kernel_dp2_equals_dycore_delp_cellwise():
    """Direct cell-by-cell check (seam cells included): with tracer
    subcycling (nsplt > 1 forced by a long step), tracer_2d_1L returns its
    dp1 advanced through nsplt - 1 of nsplt equal flux fractions, so its
    final dp2 = dp1_in + nsplt/(nsplt-1) (dp1_out - dp1_in); that must
    equal the d_sw2 Lagrangian delp the remap uses, to round-off.  (The
    nsplt = 1 path returns no dp2; it is covered by the white-noise mass
    test above.)"""
    r = _build(11, dt=4 * 3600.0, n_sub=8)
    m0, a0 = _masses(r, r.carry)
    c = r.step(r.carry, 4 * 3600.0)
    m1, a1 = _masses(r, c)
    # per-tracer and air mass closure on the subcycled path
    np.testing.assert_allclose(m1, m0, rtol=1e-13, atol=0)
    np.testing.assert_allclose(a1, a0, rtol=1e-13, atol=0)
    lp = r.last_phase
    sl = slice(3, 3 + N)
    d_in = np.asarray(lp["dp1"])[:, sl, sl]
    d_out = np.asarray(lp["tr"]["dp1"])[:, sl, sl]
    d_lag = np.asarray(lp["delp_lag"])[:, sl, sl]
    ns = np.asarray(lp["tr"]["nsplt"])
    ks = [k for k in range(30) if ns[k] > 1]
    assert len(ks) >= 10, ns
    for k in ks:
        dp2 = d_in[..., k] + ns[k] / (ns[k] - 1) * (d_out[..., k]
                                                   - d_in[..., k])
        scale = np.max(np.abs(d_lag[..., k] - d_in[..., k]))
        assert scale > 10.0
        # round-off of the reconstruction: a few ulp of the layer mass,
        # amplified by nsplt/(nsplt-1) <= 2
        tol = 64 * np.finfo(np.float64).eps * np.max(d_lag[..., k])
        np.testing.assert_allclose(dp2, d_lag[..., k], rtol=0, atol=tol)


@x64
def test_dcmip12_full_day_return_and_budgets():
    """Full 1-day Hadley return (C12, dt 600 s, 144 steps): elapsed time,
    tracer and air masses to round-off, q1 return l2 below 0.5 (C36: 0.19)."""
    r = _build(12)
    c = r.carry
    m0, a0 = _masses(r, c)
    for i in range(144):
        c = r.step(c, 600.0)
        if i == 71:                          # t = T/2: tracer displaced
            qh = r.tracers(c)[..., 0]
            w = r.area[..., None] * np.ones_like(qh)
            q0 = r.q_ic[..., 0]
            assert np.sqrt(np.sum(w * (qh - q0) ** 2)
                           / np.sum(w * q0 ** 2)) > 0.3
    assert c[1] == 86400.0
    m1, a1 = _masses(r, c)
    np.testing.assert_allclose(m1, m0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(a1, a0, rtol=1e-12, atol=0)
    q, q0 = r.tracers(c)[..., 0], r.q_ic[..., 0]
    w = r.area[..., None] * np.ones_like(q)
    l2 = np.sqrt(np.sum(w * (q - q0) ** 2) / np.sum(w * q0 ** 2))
    assert l2 < 0.5, l2


@x64
def test_winds_sampled_at_step_midpoint():
    """A time-dependent wind gives the step of the same wind frozen at the
    step midpoint t + dt/2 (to round-off), and differs from the endpoint
    samples by far more."""
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip_sigma(30)
    dt, t0 = 1800.0, 3 * 1800.0

    def make(fixed_t):
        def wind(t, lon, lat, s_):
            return D.dcmip12_wind_geo(t if fixed_t is None else fixed_t,
                                      lon, lat, s_)
        return build_duo_dcmip_transport(
            N, wind_geo=wind, init_latlon=D.dcmip12_init_latlon,
            sigma_coord=sc, n_tracers=1, dt=dt, n_sub=1)
    out = {}
    for key, ft in (("td", None), ("mid", t0 + 0.5 * dt), ("left", t0),
                    ("right", t0 + dt)):
        r = make(ft)
        (delp, q), _ = r.carry
        c = r.step(((delp, q), t0), dt)
        out[key] = r.tracers(c)
    # equal up to XLA's different fusion of a traced vs a constant time
    # (measured ~1e-16)
    np.testing.assert_allclose(out["td"], out["mid"], rtol=0, atol=1e-13)
    assert float(np.max(np.abs(out["td"] - out["left"]))) > 1e-8
    assert float(np.max(np.abs(out["td"] - out["right"]))) > 1e-8


def test_remap_tracers_fv3_pressure_jitted_float32():
    """float32 inputs under jit, in either precision mode (the x64 guard
    reads dtypes without converting tracers)."""
    ak, bk, ptop = dcmip2012_transport_eta_table(30)
    pe = (ak + bk * 1.0e5).astype(np.float32)[None]
    pe1 = pe.copy()
    pe1[0, 10] += 50.0
    q = np.random.default_rng(1).uniform(0, 1, (1, 30, 2)).astype(np.float32)
    f = jax.jit(lambda a, b, c: remap_tracers_fv3_pressure(a, b, c))
    q2 = jax.block_until_ready(f(jnp.asarray(q), jnp.asarray(pe1),
                                 jnp.asarray(pe)))
    assert q2.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(q2)))
    # the moved interface changes the adjacent layers, and the column
    # budget closes to float32 round-off
    q2 = np.asarray(q2, np.float64)
    assert np.max(np.abs(q2[0, 9:11] - q[0, 9:11])) > 1e-4
    b1 = np.sum(q[0].astype(np.float64)
                * np.diff(pe1[0].astype(np.float64))[:, None], 0)
    b2 = np.sum(q2[0] * np.diff(pe[0].astype(np.float64))[:, None], 0)
    np.testing.assert_allclose(b2, b1, rtol=1e-5)


@x64
def test_blob_crosses_equatorial_polar_seam():
    """Solid-body rotation about the x axis carries a blob from (70E, 0)
    toward the north pole, across an equatorial-face / polar-face seam
    (face axes swap there).  After 2 days of a 12-day rotation (60 deg)
    the tracer-mass centroid must point along the analytically rotated
    centre (within 3 deg of arc)."""
    from legoesm.core.field import Field
    from legoesm.core.state import TracerState
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip_sigma(30)
    km = 30
    period = 12 * 86400.0
    om = 2 * np.pi / period
    c0 = np.array([np.cos(np.radians(70.0)), np.sin(np.radians(70.0)), 0.0])

    def wind(t, lon, lat, s_):
        # V = Omega x r, Omega = om * x_hat:  V = om (0, -z, y)
        x = jnp.cos(lat) * jnp.cos(lon)
        y = jnp.cos(lat) * jnp.sin(lon)
        z = jnp.sin(lat)
        Vx, Vy, Vz = 0.0 * x, -om * z, om * y
        ue = -jnp.sin(lon) * Vx + jnp.cos(lon) * Vy
        vn = (-jnp.sin(lat) * jnp.cos(lon) * Vx
              - jnp.sin(lat) * jnp.sin(lon) * Vy + jnp.cos(lat) * Vz)
        u = FV3_RADIUS_M * ue[..., None] * jnp.ones(km)
        v = FV3_RADIUS_M * vn[..., None] * jnp.ones(km)
        return u, v, jnp.zeros(lat.shape + (km + 1,))

    def init(cells, s_):
        lo, la = cells.lon2d, cells.lat2d
        cosd = (jnp.cos(la) * jnp.cos(lo) * c0[0]
                + jnp.cos(la) * jnp.sin(lo) * c0[1] + jnp.sin(la) * c0[2])
        d = jnp.arccos(jnp.clip(cosd, -1, 1))
        q = jnp.exp(-(d / 0.25) ** 2)[..., None, None] * jnp.ones((km, 1))
        return TracerState(
            tracers=Field(data=q, name="tracers",
                          dims=("lat", "lon", "level", "tracer"), units="1"),
            time=Field(data=jnp.asarray(0.0), name="time", dims=(),
                       units="s"))

    r = build_duo_dcmip_transport(
        N, wind_geo=wind, init_latlon=init, sigma_coord=sc, n_tracers=1,
        dt=3600.0, n_sub=None, period_s=period)
    c = r.carry
    for _ in range(48):
        c = r.step(c, 3600.0)
    th = om * 48 * 3600.0
    want = np.array([c0[0], c0[1] * np.cos(th) - c0[2] * np.sin(th),
                     c0[1] * np.sin(th) + c0[2] * np.cos(th)])
    w = np.einsum("fijk,fijk,fij->fij", r.layer_mass(c),
                  r.tracers(c)[..., 0], r.area)
    lo, la = np.radians(r.lon_deg), np.radians(r.lat_deg)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                    np.sin(la)], -1)
    cen = np.einsum("fij,fijx->x", w, xyz)
    cen /= np.linalg.norm(cen)
    ang = np.degrees(np.arccos(np.clip(cen @ want, -1, 1)))
    assert ang < 3.0, (cen, want, ang)


@x64
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_remap_tracers_fv3_pressure_refuses_nonfinite_tracer(bad):
    ak, bk, ptop = dcmip2012_transport_eta_table(30)
    pe = (ak + bk * 1.0e5)[None]
    pe1 = pe.copy()
    pe1[0, 10] += 50.0
    q = np.random.default_rng(2).uniform(0, 1, (1, 30, 2))
    q[0, 7, 1] = bad
    f = jax.jit(lambda a, b, c: remap_tracers_fv3_pressure(a, b, c))
    with pytest.raises(Exception, match="non-finite tracer input"):
        jax.block_until_ready(f(jnp.asarray(q), jnp.asarray(pe1),
                                jnp.asarray(pe)))


@x64
def test_remap_tracers_fv3_pressure_refuses_mixed_dtypes():
    """One dtype in and out: no narrowing output cast can lose a budget."""
    ak, bk, ptop = dcmip2012_transport_eta_table(30)
    pe = (ak + bk * 1.0e5)[None]
    q = np.zeros((1, 30, 1), np.float32)
    with pytest.raises(ValueError, match="share one dtype"):
        remap_tracers_fv3_pressure(jnp.asarray(q), jnp.asarray(pe),
                                   jnp.asarray(pe))


@x64
def test_runner_snaps_integer_step_requests(tmp_path):
    """23 * 600 s / 86400 d is 22.999999999999996 steps in float64: the
    runner takes 23 steps, not 22."""
    from scripts.matrix.run_atmosphere_test_matrix import (
        TestCase as MatrixCase, run_dcmip_transport)
    case = MatrixCase("hydrostatic", "dcmip_transport_12", "cubed_sphere",
                      f"C{N}", "sigma", 12, 1, {"test_num": 12})
    nf = 23 * 600 / 86400 * 86400 / 600
    assert nf < 23 and 23 - nf <= 8 * np.spacing(23.0)
    status, _, notes = run_dcmip_transport(case, tmp_path, 23 * 600 / 86400)
    assert status == "PASS", notes
    fields = dict(line.split(": ", 1) for line in
                  (tmp_path / "results.txt").read_text().splitlines()
                  if ": " in line)
    assert int(fields["accepted_steps"]) == 23, fields


@x64
def test_default_hydro_duo_row_still_cam6_l32():
    """The production hydro duo builder (every other duo row) still runs
    the CAM6 L32 table, not the DCMIP L30 one."""
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI
    from legoesm import constants
    from scripts.matrix._cube_duo_adapters import build_duo_hydro
    h = build_duo_hydro(36, perturbed=False)
    assert h.km == 32 and h.model.config.eta == "cam6_l32"
    np.testing.assert_allclose(h.model.ak, np.asarray(CAM6_L32_HYAI)
                               * constants.p_ref, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(h.model.bk, np.asarray(CAM6_L32_HYBI),
                               rtol=0, atol=1e-15)
    assert np.asarray(h.state["state"]["delp"]).shape[-1] == 32
    assert h.dt == 1920.0  # the deck step, not the generic CFL clamp


def test_hydro_duo_refuses_unvalidated_resolution(monkeypatch):
    """The 1920 s deck step is validated at C36 only; refused before any setup."""
    import legoesm.grids.factory as factory
    from scripts.matrix._cube_duo_adapters import build_duo_hydro

    def _no_setup(*a, **k):
        raise AssertionError("grid built before the resolution guard")
    monkeypatch.setattr(factory, "create_fv3_duo_grid", _no_setup)
    with pytest.raises(ValueError, match="validated at C36 only"):
        build_duo_hydro(12, perturbed=False)


def _ps13(lo, la):
    return D._p0 * jnp.exp(-D._mountain_height(lo, la) / D._H)


@x64
def test_duo_13_starts_on_the_kent_hybrid_air_mass():
    """1-3 on the Kent et al. (2014, App. B) hybrid levels: the duo's
    fixed-lid table is that hybrid table, the initial delp is dA p0 + dB
    p_s with p_s = p0 exp(-z_s/H) over the mountains, the tracers are the
    analytic 1-3 clouds at those heights, and one step conserves the
    tracer mass and keeps q = const."""
    from types import SimpleNamespace
    from legoesm.grids.vertical import dp_from_hybrid
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip13_hybrid(30)
    r = build_duo_dcmip_transport(
        N, wind_geo=D.dcmip13_wind_geo_banded, init_latlon=D.dcmip13_init_latlon,
        sigma_coord=sc, n_tracers=4, dt=1800.0, n_sub=None,
        period_s=12 * 86400.0, ps_init_fn=_ps13)
    lon, lat = jnp.radians(r.lon_deg), jnp.radians(r.lat_deg)
    ps = np.asarray(_ps13(lon, lat))
    assert float(np.ptp(ps)) > 1.0e4                        # real terrain
    np.testing.assert_allclose(r.ps(r.carry), ps, rtol=1e-12, atol=0)
    np.testing.assert_allclose(r.layer_mass(r.carry),
                               np.asarray(dp_from_hybrid(sc, ps)),
                               rtol=1e-10, atol=0)
    np.testing.assert_array_equal(r.ps_ref, ps)
    q = np.asarray(D.dcmip13_init_latlon(
        SimpleNamespace(lon2d=lon.reshape(6 * N, N), lat2d=lat.reshape(6 * N, N)),
        sc).tracers.data).reshape(6, N, N, 30, 4)
    np.testing.assert_array_equal(r.tracers(r.carry), q)
    m0, a0 = _masses(r, r.carry)
    c = r.step(r.carry, 1800.0)
    m1, a1 = _masses(r, c)
    np.testing.assert_allclose(m1, m0, rtol=1e-13, atol=0)
    np.testing.assert_allclose(a1, a0, rtol=1e-13, atol=0)
    (dp, qq), t = r.carry
    cu = r.step(((dp, jnp.ones_like(qq)), t), 1800.0)
    np.testing.assert_allclose(r.tracers(cu), 1.0, rtol=0, atol=1e-13)


@x64
def test_duo_refuses_a_hybrid_table_other_than_its_own():
    """The table guard extended to hybrid rows: A p0 / B must be the
    model's ak / bk."""
    from legoesm.grids.vertical import create_hybrid_coordinate
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    sc = D.create_dcmip13_hybrid(30)
    a, b = np.asarray(sc.A_half, np.float64), np.asarray(sc.B_half, np.float64)
    shift = np.zeros_like(b)
    shift[1:-1] = 1e-3                    # same A + B (= eta), other split
    bad = create_hybrid_coordinate(30, a - shift, b + shift, p_ref=D._p0,
                                   dtype=jnp.float64)
    other_pref = create_hybrid_coordinate(30, a, b, p_ref=0.99 * D._p0,
                                          dtype=jnp.float64)
    for c in (bad, other_pref):              # other A/B split; other p_ref
        with pytest.raises(ValueError, match="hybrid A p_ref / B"):
            build_duo_dcmip_transport(
                N, wind_geo=D.dcmip13_wind_geo_banded,
                init_latlon=D.dcmip13_init_latlon, sigma_coord=c, n_tracers=4,
                dt=1800.0, n_sub=1, ps_init_fn=_ps13)


@x64
def test_duo_level_heights_follow_the_layer_mass():
    """The duo 1-3 terrain audit's level heights: equal to the analytic
    1-3 full-level heights at the analytic air mass, and moving with delp
    (a column with 5 % more mass in every layer sits lower)."""
    from legoesm.grids.vertical import dp_from_hybrid
    from scripts.matrix.run_atmosphere_test_matrix import duo_full_level_height
    sc = D.create_dcmip13_hybrid(30)
    lon = jnp.linspace(0.0, 2 * np.pi, 41)[:-1]
    lat = jnp.zeros_like(lon) + 0.1
    ps = np.asarray(_ps13(lon, lat))
    assert float(np.ptp(ps)) > 1.0e3
    dp = np.asarray(dp_from_hybrid(sc, ps))
    ptop = dcmip2012_transport_eta_table(30)[2]
    z = duo_full_level_height(dp, ptop)
    np.testing.assert_allclose(
        z, np.asarray(D.dcmip13_full_level_height(lon, lat, sc)), rtol=0, atol=1e-6)
    assert np.all(duo_full_level_height(1.05 * dp, ptop) < z)


def test_duo_air_mass_criterion_catches_an_intermediate_sample():
    """The duo global air-mass gate reads every sample, not only the end:
    a planted out-of-tolerance intermediate sample fails with equal
    initial and final totals; NaN fails; round-off passes."""
    from scripts.matrix.run_atmosphere_test_matrix import duo_air_mass_criterion
    a0 = 5.1e19
    note, failed = duo_air_mass_criterion(a0, a0, [a0, a0 * (1 + 1e-9), a0])
    assert failed and "global air-mass conservation" in failed[0], (note, failed)
    assert "over 3 samples" in note
    assert duo_air_mass_criterion(a0, a0, [a0, np.nan])[1]
    assert not duo_air_mass_criterion(a0, a0 * (1 + 1e-15), [a0 * (1 - 1e-15)])[1]


def test_duo13_exact_tracers_follow_the_moved_levels():
    """The duo 1-3 norms' exact field: the analytic clouds at the level
    heights of the given state's delp -- equal to the analytic cloud on
    moved levels (zero error for such a final field), while the
    layer-index return error vs the IC is not zero."""
    from types import SimpleNamespace
    from legoesm.grids.vertical import dp_from_hybrid
    from scripts.matrix.run_atmosphere_test_matrix import (
        duo13_exact_tracers, duo_full_level_height)
    sc = D.create_dcmip13_hybrid(30)
    # the clouds' longitudes, on the equator
    lon_deg = np.linspace(120.0, 240.0, 6 * 4 * 4).reshape(6, 4, 4)
    lat_deg = np.zeros_like(lon_deg)
    ps = np.asarray(_ps13(jnp.radians(lon_deg), jnp.radians(lat_deg)))
    dp0 = np.asarray(dp_from_hybrid(sc, ps))
    dp1 = dp0 * 1.03                                  # moved levels
    ptop = dcmip2012_transport_eta_table(30)[2]
    duo = SimpleNamespace(lon_deg=lon_deg, lat_deg=lat_deg, ptop=ptop,
                          layer_mass=lambda s: s)
    q_ic = np.asarray(duo13_exact_tracers(duo, dp0))
    q_T = np.stack([np.asarray(x) for x in D.dcmip13_tracers_at_points(
        jnp.radians(lon_deg), jnp.radians(lat_deg),
        jnp.asarray(duo_full_level_height(dp1, ptop)))], axis=-1)
    np.testing.assert_array_equal(np.asarray(duo13_exact_tracers(duo, dp1)), q_T)
    assert float(np.max(np.abs(q_T - q_ic))) > 1e-2


@x64
def test_audit_wires_the_duo_air_mass_failure_into_the_row(monkeypatch, tmp_path):
    """A failing duo global air-mass criterion reaches the row's CRITERION
    FAIL list (the audit wiring, not only the helper)."""
    import scripts.matrix.run_atmosphere_test_matrix as M
    # local closure made to pass: the planted air-mass failure alone fails it
    monkeypatch.setattr(M, "_DCMIP13_CLOSURE_TOL", float("inf"))
    monkeypatch.setattr(
        M, "duo_air_mass_criterion",
        lambda a0, a1, s: ("planted", ["global air-mass conservation (planted)"]))
    case = M.TestCase("hydrostatic", "dcmip_transport_13", "cubed_sphere",
                      f"C{N}", "hybrid", 12, 1, {"test_num": 13})
    status, _, notes = M.run_dcmip_transport(case, tmp_path, 2 * 1800 / 86400)
    assert status == "FAIL"
    assert "local air-mass closure" not in notes.split("CRITERION FAIL")[-1], notes
    assert re.search(r"CRITERION FAIL: .*global air-mass conservation \(planted\)",
                     notes), notes


def test_dcmip13_snapshot_geometry_uses_the_snapshot_layer_mass():
    """The 1-3 terrain moments on the duo row take weights AND level
    heights from the delp saved with each snapshot; the prescribed-flux
    rows keep the steady weights / analytic heights."""
    from types import SimpleNamespace
    from scripts.matrix.run_atmosphere_test_matrix import (
        dcmip13_snapshot_geometry, duo_full_level_height)
    rng = np.random.default_rng(0)
    area = rng.uniform(1.0, 2.0, (2, 3))
    dp = rng.uniform(500.0, 900.0, (2, 3, 30))
    w0, z0 = np.ones((2, 3, 30)), np.zeros((2, 3, 30))
    duo = SimpleNamespace(ptop=2.5e4)
    w, z = dcmip13_snapshot_geometry({"delp_3d": dp}, area, duo, w0, z0)
    np.testing.assert_array_equal(w, area[..., None] * dp)
    np.testing.assert_array_equal(z, duo_full_level_height(dp, 2.5e4))
    assert not np.allclose(z, duo_full_level_height(0.9 * dp, 2.5e4))
    w, z = dcmip13_snapshot_geometry({"delp_3d": dp}, area, None, w0, z0)
    assert w is w0 and z is z0


@x64
def test_matrix_runner_cube_13_hands_the_banded_wind_and_terrain_ps(monkeypatch, tmp_path):
    """The runner's cube 1-3 hybrid row builds the duo with the banded 1-3
    wind and p_s = p0 exp(-z_s/H) (the other grids' 1-3 flow)."""
    import scripts.matrix._cube_duo_adapters as A
    from scripts.matrix.run_atmosphere_test_matrix import (
        TestCase as MatrixCase, run_dcmip_transport)
    seen = {}

    class _Stop(Exception):
        pass

    def _capture(n, **kw):
        seen.update(kw)
        raise _Stop
    monkeypatch.setattr(A, "build_duo_dcmip_transport", _capture)
    case = MatrixCase("hydrostatic", "dcmip_transport_13", "cubed_sphere",
                      f"C{N}", "hybrid", 12, 1, {"test_num": 13})
    with pytest.raises(_Stop):
        run_dcmip_transport(case, tmp_path, 0.05)
    assert seen["wind_geo"] is D.dcmip13_wind_geo_banded
    assert seen["init_latlon"] is D.dcmip13_init_latlon
    lo, la = jnp.asarray([0.3, 2.0]), jnp.asarray([0.1, -0.4])
    np.testing.assert_array_equal(np.asarray(seen["ps_init_fn"](lo, la)),
                                  np.asarray(_ps13(lo, la)))
    assert hasattr(seen["sigma_coord"], "B_half")


@x64
def test_duo_refuses_a_hybrid_row_without_terrain_pressure():
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    with pytest.raises(ValueError, match="needs ps_init_fn"):
        build_duo_dcmip_transport(
            N, wind_geo=D.dcmip13_wind_geo_banded,
            init_latlon=D.dcmip13_init_latlon,
            sigma_coord=D.create_dcmip13_hybrid(30), n_tracers=4, dt=1800.0,
            n_sub=1)


@x64
def test_duo_refuses_terrain_pressure_on_a_sigma_row():
    """ps_init_fn needs the hybrid coordinate: on sigma levels the duo's
    fixed-lid pressures differ from p = sigma p_s away from p0."""
    from scripts.matrix._cube_duo_adapters import build_duo_dcmip_transport
    with pytest.raises(ValueError, match="ps_init_fn needs the hybrid"):
        build_duo_dcmip_transport(
            N, wind_geo=D.dcmip13_wind_geo_banded,
            init_latlon=D.dcmip13_init_latlon,
            sigma_coord=D.create_dcmip_sigma(30), n_tracers=4, dt=1800.0,
            n_sub=1, ps_init_fn=_ps13)


@x64
def test_matrix_runner_cube_13_hybrid_runs_duo_with_the_13_audit(tmp_path):
    """The cube 1-3 matrix row (Kent hybrid levels) runs FV3 duo's
    transport and carries the 1-3 audit of the other grids (closure,
    uniform tracer, tracer mass at every sample, terrain crossing)."""
    from scripts.matrix.run_atmosphere_test_matrix import (
        TEST_MATRIX, TestCase as MatrixCase, run_dcmip_transport)
    assert {tc.vertical_coord for tc in TEST_MATRIX
            if tc.case == "dcmip_transport_13"
            and tc.grid_type == "cubed_sphere"} == {"hybrid"}
    case = MatrixCase("hydrostatic", "dcmip_transport_13", "cubed_sphere",
                      f"C{N}", "hybrid", 12, 1, {"test_num": 13})
    status, _, notes = run_dcmip_transport(case, tmp_path, 3 * 1800 / 86400)
    text = (tmp_path / "results.txt").read_text()
    assert "vertical_scheme: fv3_duo Lagrangian" in text, text
    assert "W_eff" in notes and "p_s-p_s(t=0)" in notes, notes
    assert re.search(r"air-mass closure drift [0-9.e+-]+ \(duo", notes), notes
    unif = float(re.search(r"uniform-tracer max\|q-1\| ([0-9.e+-]+)", notes).group(1))
    assert unif <= 1e-12, notes
    assert int(re.search(r"peak over (\d+) samples", notes).group(1)) >= 3, notes
    # the peak is the worst deviation over every tracer and sample
    peak = float(re.search(r"peak over \d+ samples q\d ([0-9.e+-]+)", notes).group(1))
    assert peak <= 1e-12, notes
    rels = re.search(r"mass\[q dp\] rel q1 (\S+), q2 (\S+), q3 (\S+), q4 (\S+),", notes)
    assert rels, notes
    vals = [float(x) for x in rels.groups()]
    assert len(vals) == 4 and all(np.isfinite(v) and abs(v) <= 1e-12 for v in vals), notes
    air = float(re.search(r"air-mass rel ([-+0-9.eE]+)", notes).group(1))
    assert abs(air) < 1e-13, notes
    m = re.search(r"global air mass max \|rel\| (\S+) over (\d+) samples", notes)
    assert m and float(m.group(1)) <= 1e-12 and int(m.group(2)) >= 3, notes
    assert "AUDIT INCOMPLETE: terrain crossing untested" in notes, notes
    # the duo's air column departs from the steady analytic one (p_s
    # evolves): under the current closure gate (the other grids' 1e-6) a
    # measured failure, and it reaches the row status
    assert "CRITERION FAIL: local air-mass closure" in notes, notes
    assert status == "FAIL" and "status: FAIL" in text


@x64
@pytest.mark.slow
def test_matrix_runner_cube_13_reports_full_period_norms(tmp_path, monkeypatch):
    """Over the full 12 days the cube 1-3 row reports the error norms and
    the terrain-crossing measurement (C12, 576 steps)."""
    from scripts.matrix.run_atmosphere_test_matrix import (
        TestCase as MatrixCase, run_dcmip_transport)
    case = MatrixCase("hydrostatic", "dcmip_transport_13", "cubed_sphere",
                      f"C{N}", "hybrid", 12, 1, {"test_num": 13})
    import scripts.matrix.run_atmosphere_test_matrix as M
    calls, scored = [], []
    real = M.duo13_exact_tracers
    real_norms = D.compute_tracer_error_norms

    def _spy(duo, st):
        out = real(duo, st)
        calls.append((float(st[1]), np.asarray(out)))
        return out

    def _norms(final, exact, g):
        scored.append(np.asarray(exact.tracers.data))
        return real_norms(final, exact, g)
    monkeypatch.setattr(M, "duo13_exact_tracers", _spy)
    monkeypatch.setattr(D, "compute_tracer_error_norms", _norms)
    _, _, notes = run_dcmip_transport(case, tmp_path, 12.0)
    assert [t for t, _ in calls] == [12 * 86400.0], calls   # the final state
    # the primary norms are scored against that final-level field, which
    # differs from the IC (the second call: the layer-index return error)
    assert len(scored) == 2
    np.testing.assert_array_equal(scored[0], calls[0][1])
    assert float(np.max(np.abs(scored[0] - scored[1]))) > 0.0
    for i in range(1, 5):
        m = re.search(rf"q{i} L1=(\S+) L2=(\S+) Linf=(\S+?),", notes)
        assert m, notes
        assert all(np.isfinite(float(x)) and float(x) > 0 for x in m.groups()), notes
    assert "integrated 576 of 576 steps" in notes, notes
    assert "exact = analytic clouds at the final level heights" in notes, notes
    m = re.search(r"terrain crossing: cloud height shift (\S+) m at the largest "
                  r"terrain shift (\S+) m \(ratio (\S+);", notes)
    assert m and all(np.isfinite(float(x)) for x in m.groups()), notes

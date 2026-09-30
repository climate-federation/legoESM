"""Rung 4b (route A, M4): orography and the ERA5 initial condition on the
FV3 duo column lane, C12 km=32 (CAM6 L32).

* the del-2 terrain filter on the duo faces: conserves the area
  integral of phis, damps the 2-dx mode, halo-complete, identity at 0;
* ERA5 -> duo bundle: hydrostatic consistency (p_s moved to the grid's
  smoothed phis), specific humidity kept, D winds = the certified lift
  of the A-grid ERA5 winds, halos zero;
* rest states over the smoothed ERA5 mountain (GLM 2026-09-27): the
  isothermal one is exact for the FV PGF (regression), the
  constant-lapse-rate one is not (the spurious-wind gate);
* a short ERA5 run stays finite and bounded.

ERA5 tests need the AMIP IC zarr (skipped when absent).
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.grids.factory import create_fv3_duo_grid  # noqa: E402

N, NG, KM = 12, 3, 32
from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA as FV3_KAPPA_TEST  # noqa: E402
Q_CONST = 0.02     # synthetic specific humidity; q/(1-q) would read 0.0204
DT = 1920.0
CI = slice(NG, NG + N)
CE = slice(NG, NG + N + 1)
ERA5_ZARR = os.environ.get(
    "LEGOESM_ERA5_IC_ZARR",
    "/burg-archive/glab/users/ac5006/legoESM/data/amip/era5_ic_1979-01-01_gproxy.zarr")
needs_era5 = pytest.mark.skipif(not os.path.isdir(ERA5_ZARR),
                                reason="ERA5 IC zarr not on this machine")


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    yield
    jax.clear_caches()


def _gaussian_mountain(lon, lat, *, lon0=1.5, lat0=0.5, width=0.6, h=3000.0):
    d2 = (np.cos(lat) * np.angle(np.exp(1j * (lon - lon0)))) ** 2 + (lat - lat0) ** 2
    return constants.g * h * np.exp(-d2 / width ** 2)


def _mountain_with_2dx_noise(lon, lat):
    rng = np.random.default_rng(3)
    return _gaussian_mountain(lon, lat) + 200.0 * rng.standard_normal(lon.shape)


def test_terrain_filter_conserves_area_integral_and_damps_noise():
    from legoesm.grids.terrain_filter import terrain_filter_duo
    raw = create_fv3_duo_grid(N, NG, phis_fn=_mountain_with_2dx_noise)
    gs6, ectx = raw.ctx_np["gs6"], raw.ctx_np["ectx"]
    hs6 = np.stack([np.asarray(h) for h in raw.ctx_np["hs6"]])
    area = np.stack([gs["area"][CI, CI] for gs in gs6])
    out0 = np.asarray(terrain_filter_duo(hs6, gs6, ectx, n_iter=0))
    np.testing.assert_array_equal(out0, hs6)
    out = np.asarray(terrain_filter_duo(hs6, gs6, ectx, n_iter=8))
    assert np.isfinite(out).all()
    # flux form: exactly conservative INSIDE a face (the fluxes telescope);
    # across the face seams the two faces' halo metrics (the extended
    # duo geometry, not the neighbour's real cells) differ, so the global
    # area integral moves -- MEASURED 3.4e-3 relative on this 2-dx noise
    # field, 8 passes (2026-09-27; a one-time terrain smoothing, the
    # same property as FV3's own del2_cubed_sphere on halo-filled fields)
    i0, i1 = (hs6[:, CI, CI] * area).sum(), (out[:, CI, CI] * area).sum()
    print(f"terrain_filter_duo: area-integral change {abs(i1 - i0) / abs(i0):.2e}")
    assert abs(i1 - i0) <= 5e-3 * abs(i0), (i0, i1)   # pin: 3.4e-3 measured
    # one pass vs a scalar-loop transcription of del2_cubed_sphere on the
    # interior of face 0 (fx(i,j) = dy*sina_u*(q(i-1,j)-q(i,j))*rdxc at
    # u-point i, fy likewise at v-point j, q += cd*rarea*(div))
    one = np.asarray(terrain_filter_duo(hs6, gs6, ectx, n_iter=1))
    gs, q0 = gs6[0], hs6[0]
    cd = 0.20 * float(min(float(g_["da_min"]) for g_ in gs6))
    ref = q0.copy()
    for i in range(NG, NG + N):
        for j in range(NG, NG + N):
            fxw = gs["dy"][i, j] * gs["sina_u"][i, j] * (q0[i - 1, j] - q0[i, j]) * gs["rdxc"][i, j]
            fxe = gs["dy"][i + 1, j] * gs["sina_u"][i + 1, j] * (q0[i, j] - q0[i + 1, j]) * gs["rdxc"][i + 1, j]
            fys = gs["dx"][i, j] * gs["sina_v"][i, j] * (q0[i, j - 1] - q0[i, j]) * gs["rdyc"][i, j]
            fyn = gs["dx"][i, j + 1] * gs["sina_v"][i, j + 1] * (q0[i, j] - q0[i, j + 1]) * gs["rdyc"][i, j + 1]
            ref[i, j] += cd * gs["rarea"][i, j] * (fxw - fxe + fys - fyn)
    np.testing.assert_allclose(one[0, CI, CI], ref[CI, CI], rtol=1e-12, atol=1e-9)
    # a constant field is a fixed point: the window bitwise (all fluxes
    # zero), the halos to the ext exchange's interpolation rounding
    const = np.full_like(hs6, 1234.5)
    c3 = np.asarray(terrain_filter_duo(const, gs6, ectx, n_iter=3))
    np.testing.assert_array_equal(c3[:, CI, CI], const[:, CI, CI])
    np.testing.assert_allclose(c3, const, rtol=1e-13, atol=0.0)
    # the 2-dx mode is damped: compact-Laplacian roughness drops
    def rough(f):
        c = f[:, CI, CI]
        return float(np.abs(4 * c - f[:, NG - 1:NG + N - 1, CI]
                            - f[:, NG + 1:NG + N + 1, CI]
                            - f[:, CI, NG - 1:NG + N - 1]
                            - f[:, CI, NG + 1:NG + N + 1]).mean())
    assert rough(out) < 0.5 * rough(hs6)
    # the mountain survives (a filter, not a flattening): a 34-degree
    # (~4.6-cell) Gaussian keeps most of its peak after 8 passes
    print(f"terrain_filter_duo: peak ratio {out[:, CI, CI].max() / hs6[:, CI, CI].max():.3f}")
    assert out[:, CI, CI].max() > 0.6 * hs6[:, CI, CI].max()
    # halo-complete IN THE DYCORE'S CONVENTION: the halos are the context
    # builder's ext exchange of the smoothed window (what geopk's seam
    # values are built on), so re-applying that exchange is a no-op.
    # The stepper's plain A-grid exchange is NOT that exchange (its
    # halos sit up to 1e4 m^2/s^2 away, MEASURED 2026-09-27: 8 m/s of
    # spurious seam wind in a rest state) -- pinned as a must-differ.
    from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface
    from legoesm.grids.fv3_duo_halos import exchange_agrid_scalar_halos
    again = [out[t].copy() for t in range(6)]
    ext_scalar_sixface(again, "A", ectx)
    np.testing.assert_array_equal(np.stack(again), out)
    plain = np.asarray(exchange_agrid_scalar_halos(jnp.asarray(out), raw.ctx_jax.tab))
    assert np.abs(plain - out)[:, :NG].max() > 1.0e3, "the two exchanges agree?"
    # and the grid factory wires it: hs6 in the jax ctx is the smoothed one
    g = create_fv3_duo_grid(N, NG, phis_fn=_mountain_with_2dx_noise,
                            phis_filter_iter=8)
    np.testing.assert_array_equal(np.asarray(g.ctx_jax.hs6), out)


def _models(grid, km=KM):
    """The L32 model by default; ``km=5`` (the analytic table) for the
    tests whose subject is not the vertical (the km=32 XLA CPU compile
    is the open finding of the M4 plan)."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import FV3DuoColumnModel
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    eta = "cam6_l32" if km == 32 else "analytic"
    dyn = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=km, n_split=8, eta=eta))
    return dyn, FV3DuoColumnModel(dyn)


def _rest_bundle(col, T_of_p, ps_of_phis, p_s_ref=1.0e5, *, theta=None):
    """Rest state over the grid's phis: p_s from the ANALYTIC hypsometric
    solution ``ps_of_phis`` of the profile ``T_of_p``, pt = T(p_full);
    with ``theta`` the layer temperature is ``theta*pkz/p_ref^kappa`` from
    the bundle's own interfaces, i.e. potential temperature constant in
    the dycore's OWN layer-mean sense (pt/pkz)."""
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.core.fv3_native_state_3d import field_shape
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
    dyn = col.dyn
    n, ng, km = col.n, col.ng, col.km
    phis = np.asarray(col._phis).reshape(6, n, n)
    ps = ps_of_phis(phis, p_s_ref)
    ak, bk = np.asarray(dyn.ak), np.asarray(dyn.bk)
    pe = ak[None, None, None, :] + bk[None, None, None, :] * ps[..., None]
    delp_c = np.diff(pe, axis=-1)
    p_full = delp_c / np.diff(np.log(pe), axis=-1)
    m = n + 2 * ng
    z3 = jnp.zeros((6, m, m, km), dtype=jnp.float64)
    pt = z3 if theta is not None else z3.at[:, CI, CI].set(jnp.asarray(T_of_p(p_full)))
    delp = z3.at[:, CI, CI].set(jnp.asarray(delp_c))
    press = p_var_hydrostatic(delp, ptop=dyn.ptop, akap=FV3_KAPPA, n=n, ng=ng, km=km)
    if theta is not None:
        pkz = np.asarray(press["pkz"])              # (6, n, n, km), p in Pa
        pt = z3.at[:, CI, CI].set(jnp.asarray(theta * pkz / p_s_ref ** FV3_KAPPA))
    u = jnp.zeros((6,) + tuple(field_shape("u", n, ng, km)))
    v = jnp.zeros((6,) + tuple(field_shape("v", n, ng, km)))
    q = [z3, z3, z3]                       # the column model's three slots
    return {"state": {"u": u, "v": v, "pt": pt, "delp": delp, "w": z3},
            "press": press, "q": q, "omga": jnp.zeros((6, m, m, km)),
            "nh": None}


_T0, _GAMMA = 288.0, 0.0065
_THETA = 300.0
_REST_CASES = [
    # theta constant in the dycore's layer sense (pt = theta*pkz): the
    # Lin (1997) layer PGF is exact for it in the interior, but the
    # B-grid corner interpolation of the geopotential over a curved
    # mountain is not -- MEASURED (km=5, 6 h, C12): 0.14 m/s with the
    # mountain at a face centre, 0.32 m/s with it near a cube corner.
    # So this is a spurious-wind gate, not an identity (the isothermal
    # "exact" premise of the first draft was wrong: FV3's PGF is exact
    # in (phi, p^kappa), i.e. for constant theta, not constant T).
    ("isentropic_theta300",
     None,
     lambda phis, ps0: ps0 * (1.0 - phis / (constants.c_pd * _THETA))
     ** (1.0 / FV3_KAPPA_TEST),
     _THETA),
    ("lapse_6.5K_per_km",
     lambda p: _T0 * (p / 1.0e5) ** (_GAMMA * constants.R_d / constants.g),
     lambda phis, ps0: ps0 * (1.0 - _GAMMA * phis / constants.g / _T0)
     ** (constants.g / (constants.R_d * _GAMMA)),
     None),
]


def _max_speed(col, b):
    st = col.from_bundle(b)
    return float(np.hypot(np.asarray(st.u.data), np.asarray(st.v.data)).max())


@pytest.mark.parametrize("name, T_of_p, ps_of_phis, theta", _REST_CASES)
def test_rest_state_over_a_smoothed_mountain(name, T_of_p, ps_of_phis, theta):
    """The spurious-wind gate: one day at C12 L32 over a 3 km Gaussian
    mountain smoothed 4x, from the ANALYTIC hypsometric p_s(phis) of
    each profile; absolute caps on the spurious wind (1 m/s) and on the
    surface-pressure drift (no growth-ratio criterion: a tiny constant
    residual acceleration is truncation, not instability -- codex
    2026-09-27).  This gate is what caught the terrain filter's halo
    defect (8.4 m/s, profile-independent, at a cube corner)."""
    cap = 1.0
    grid = create_fv3_duo_grid(N, NG, phis_fn=_gaussian_mountain, phis_filter_iter=4)
    dyn, col = _models(grid)
    b = _rest_bundle(col, T_of_p, ps_of_phis, theta=theta)
    assert float(np.asarray(col._phis).max()) > 0.5 * constants.g * 3000.0
    ps0 = np.asarray(b["press"]["ps"])[:, CI, CI]
    n_steps = int(86400 / DT)
    speeds = []
    for s in range(n_steps):
        b = dyn.step(b, DT)
        if s in (n_steps // 4, n_steps // 2, n_steps - 1):
            speeds.append(_max_speed(col, b))
    print(f"rest state {name}: max|u| at 6h/12h/24h = {speeds}")
    assert np.isfinite(speeds).all()
    assert max(speeds) < cap, (name, speeds)
    ps1 = np.asarray(b["press"]["ps"])[:, CI, CI]
    print(f"rest state {name}: |dp_s| max {np.abs(ps1 - ps0).max():.1f} Pa")
    assert np.abs(ps1 - ps0).max() < 100.0, np.abs(ps1 - ps0).max()   # Pa


def _synthetic_era5(phis_max_m):
    """A tiny analytic ERA5Slice (5-degree lat-lon, 8 pressure levels):
    isothermal 280 K, calm, dry, a Gaussian mountain of ``phis_max_m``."""
    from legoesm.training.era5_to_state import ERA5Slice
    lat = np.deg2rad(np.arange(-87.5, 90.0, 5.0))
    lon = np.deg2rad(np.arange(0.0, 360.0, 5.0))
    plev = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 850.0, 925.0, 1000.0]) * 100.0
    LON, LAT = np.meshgrid(lon, lat)
    phis = _gaussian_mountain(LON, LAT, h=phis_max_m)
    sh = LAT.shape + (plev.size,)
    return ERA5Slice(T=np.full(sh, 280.0), u=np.zeros(sh), v=np.zeros(sh),
                     q=np.full(sh, Q_CONST), p_s=1.0e5 * np.exp(-phis / (constants.R_d * 280.0)),
                     sst=np.full(LAT.shape, 290.0), phis=phis, lat=lat, lon=lon,
                     plev_Pa=plev)


def test_era5_bundle_refuses_terrain_above_the_hybrid_floor():
    from legoesm.training.era5_to_state import (
        era5_phis_fn, era5_to_fv3_duo_bundle)
    e = _synthetic_era5(20000.0)      # p_s ~ 87 hPa at the peak, floor 231 hPa
    grid = create_fv3_duo_grid(N, NG, phis_fn=era5_phis_fn(e), phis_filter_iter=1)
    dyn, col = _models(grid)                     # the L32 table's floor
    with pytest.raises(ValueError, match="positive-thickness floor"):
        era5_to_fv3_duo_bundle(e, col, n_tracers=3)
    e = _synthetic_era5(2000.0)
    grid = create_fv3_duo_grid(N, NG, phis_fn=era5_phis_fn(e), phis_filter_iter=1)
    dyn, col = _models(grid)
    b = era5_to_fv3_duo_bundle(e, col, n_tracers=3)
    assert np.isfinite(np.asarray(b["state"]["pt"])).all()
    # specific humidity preserved: a constant source field is a fixed point
    # of the KD-tree and hybrid interpolation, so q_v == Q_CONST to
    # rounding; a mixing-ratio conversion would give q/(1-q) (2 % off)
    q_v = np.asarray(b["q"][0])
    np.testing.assert_allclose(q_v[:, CI, CI], Q_CONST, rtol=1e-12, atol=0.0)
    assert not np.any(q_v[:, :NG]) and not np.any(q_v[:, :, :NG])
    with pytest.raises(ValueError, match="all-zero surface geopotential"):
        era5_phis_fn(e._replace(phis=np.zeros_like(e.phis)))
    # and the builder's own hydrostatic consistency on the analytic case:
    # isothermal 280 K, so p_s(grid phis) == 1e5*exp(-phis/(R T)) exactly
    st = col.from_bundle(b)
    np.testing.assert_allclose(
        np.asarray(st.p_s.data),
        1.0e5 * np.exp(-np.asarray(col._phis) / (constants.R_d * 280.0)),
        rtol=1e-2)      # IDW averages exp(-phis), not phis: 1e-2 headroom


@pytest.fixture(scope="module")
def era5():
    from legoesm.training.era5_to_state import load_era5_ic
    return load_era5_ic(ERA5_ZARR, 1979)


@needs_era5
def test_era5_bundle_is_hydrostatic_on_the_grid_terrain(era5):
    from legoesm.training.era5_to_state import (
        era5_phis_fn, era5_to_fv3_duo_bundle)
    from legoesm.core.fv3_native_physics_coupling import column_view_sixface_jax
    grid = create_fv3_duo_grid(N, NG, phis_fn=era5_phis_fn(era5), phis_filter_iter=4)
    dyn, col = _models(grid)
    b = era5_to_fv3_duo_bundle(era5, col, n_tracers=3)
    st = col.from_bundle(b)
    n_cells = col.mesh.nCells
    # the grid's phis is real terrain and the IC's phis IS the grid's
    assert float(np.asarray(col._phis).max()) > 1500.0 * constants.g
    np.testing.assert_array_equal(np.asarray(st.phis.data), np.asarray(col._phis))
    # p_s is consistent with the grid terrain: hypsometric check of
    # p_s against ERA5's own (raw phis, raw p_s) at each column,
    # |ln(ps/ps_raw) - (phis_raw - phis)/(R_d T_sfc)| ~ 0
    from legoesm.grids.regridding import (
        compute_latlon_to_voronoi_weights, regrid_scalar)
    w = compute_latlon_to_voronoi_weights(np.asarray(era5.lat), np.asarray(era5.lon),
                                          np.asarray(col.mesh.latCell),
                                          np.asarray(col.mesh.lonCell))
    ps_raw = np.asarray(regrid_scalar(jnp.asarray(era5.p_s), w))
    phis_raw = np.asarray(regrid_scalar(jnp.asarray(era5.phis), w))
    T_sfc = np.asarray(regrid_scalar(jnp.asarray(era5.T), w))[:, -1]
    lhs = np.log(np.asarray(st.p_s.data) / ps_raw)
    rhs = (phis_raw - np.asarray(col._phis)) / (constants.R_d * T_sfc)
    np.testing.assert_allclose(lhs, rhs, atol=1e-10)
    assert np.abs(rhs).max() > 0.01           # non-vacuous: terrain moved
    # specific humidity (not mixing ratio), bounded and moist
    qv = np.asarray(st.tracers["q_v"].data)
    assert 0.0 <= qv.min() and qv.max() < 0.03 and qv.max() > 0.01
    # temperatures physical
    T = np.asarray(st.T.data)
    assert 170.0 < T.min() and T.max() < 330.0
    # the D winds are the lift of the A-grid ERA5 winds: the view's own
    # c2l of them reproduces the interpolated winds to the lift/c2l
    # round-trip accuracy (< 10 % of the peak), and halos are zero
    u_m = np.asarray(st.u.data); v_m = np.asarray(st.v.data)
    assert np.hypot(u_m, v_m).max() > 20.0
    from legoesm.training.vertical_interp import interp_pressure_to_hybrid
    sig = col.sigma_coord
    vi = lambda f: np.asarray(interp_pressure_to_hybrid(  # noqa: E731
        f, jnp.asarray(era5.plev_Pa), st.p_s.data, jnp.asarray(sig.A_full),
        jnp.asarray(sig.B_full), float(sig.p_ref)))
    u_ref = vi(regrid_scalar(jnp.asarray(era5.u), w))
    v_ref = vi(regrid_scalar(jnp.asarray(era5.v), w))
    # lift (A -> D) then the view's c2l (D -> A) is not the identity: two
    # averaging operators, so ERA5's sub-grid structure at C12 (7.5 deg)
    # is lost -- MEASURED 0.186 rel RMS for u, 0.318 for v (the weaker,
    # smaller-scale component), corr 0.979 (job 10081493).  The OPERATOR
    # error is bounded by the smooth control below (solid-body rotation,
    # < 2 %); the ERA5 round trip is a resolution statement, pinned at
    # the measured values with headroom, right sign, no component swap.
    caps = {"u": 0.25, "v": 0.40}
    for got, ref, nm in ((u_m, u_ref, "u"), (v_m, v_ref, "v")):
        err = np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref ** 2))
        print(f"ERA5 wind lift round trip {nm}: rel RMS {err:.3f}")
        assert err < caps[nm], (nm, err)
        assert np.corrcoef(got.ravel(), ref.ravel())[0, 1] > 0.95
    # control: a smooth solid-body wind u = U cos(lat), v = 0 through the
    # same lift and view is reproduced to < 2 % (the operator's own error)
    U0 = 30.0
    LON, LAT = np.meshgrid(np.asarray(era5.lon), np.asarray(era5.lat))
    u_sb = np.repeat((U0 * np.cos(LAT))[..., None], np.asarray(era5.T).shape[-1], axis=-1)
    e_sb = era5._replace(u=u_sb, v=np.zeros_like(u_sb))
    st_sb = col.from_bundle(era5_to_fv3_duo_bundle(e_sb, col, n_tracers=3))
    u_got = np.asarray(st_sb.u.data)
    u_exp = (U0 * np.cos(np.asarray(col.mesh.latCell)))[:, None] * np.ones_like(u_got)
    err_sb = np.sqrt(np.mean((u_got - u_exp) ** 2)) / np.sqrt(np.mean(u_exp ** 2))
    v_sb = np.sqrt(np.mean(np.asarray(st_sb.v.data) ** 2)) / np.sqrt(np.mean(u_exp ** 2))
    print(f"solid-body lift round trip: u rel RMS {err_sb:.4f}, v/|u| {v_sb:.4f}")
    assert err_sb < 0.02 and v_sb < 0.02, (err_sb, v_sb)
    for k in ("pt", "delp"):
        a = np.asarray(b["state"][k])
        for strip in (a[:, :NG], a[:, -NG:], a[:, :, :NG], a[:, :, -NG:]):
            assert (strip == 0).all(), k
    for qi in b["q"]:
        a = np.asarray(qi)
        for strip in (a[:, :NG], a[:, -NG:], a[:, :, :NG], a[:, :, -NG:]):
            assert (strip == 0).all()
    assert st.T.data.shape == (n_cells, KM)


@needs_era5
def test_era5_run_two_days_finite_and_bounded(era5):
    from legoesm.training.era5_to_state import (
        era5_phis_fn, era5_to_fv3_duo_bundle)
    grid = create_fv3_duo_grid(N, NG, phis_fn=era5_phis_fn(era5), phis_filter_iter=4)
    dyn, col = _models(grid)
    b = era5_to_fv3_duo_bundle(era5, col, n_tracers=3)
    area = np.stack([grid.ctx_np["gs6"][t]["area"][CI, CI] for t in range(6)])
    m0 = float((np.asarray(b["state"]["delp"])[:, CI, CI].sum(-1) * area).sum())
    spd0 = _max_speed(col, b)
    for _ in range(int(2 * 86400 / DT)):
        b = dyn.step(b, DT)
    pt = np.asarray(b["state"]["pt"])[:, CI, CI]
    assert np.isfinite(pt).all() and 150.0 < pt.min() and pt.max() < 340.0
    spd = _max_speed(col, b)
    assert np.isfinite(spd) and spd < 1.5 * spd0 + 20.0, (spd0, spd)
    m1 = float((np.asarray(b["state"]["delp"])[:, CI, CI].sum(-1) * area).sum())
    assert abs(m1 - m0) / m0 < 1e-12

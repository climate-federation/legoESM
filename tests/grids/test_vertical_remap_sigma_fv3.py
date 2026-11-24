"""FV3 Lagrangian vertical remap for the prescribed-wind tracer models
(``grids.vertical.vertical_remap_sigma_fv3``).

RULE 0 budget: the remap conserves the column tracer integral over the
Lagrangian layers, ``sum q_new dsigma == sum q dsigma_lag`` to round-off.
Run with ``JAX_ENABLE_X64=1``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.field import Field
from legoesm.core.state import TracerState
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import (
    check_tracer_vertical_scheme,
    create_sigma_coordinate,
    vertical_remap_sigma_fv3,
)
from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
    TracerTransportLatLonConfig, TracerTransportLatLonModel)

x64 = pytest.mark.skipif(not jax.config.jax_enable_x64,
                         reason="round-off tolerances need x64")

NLEV = 12
DT = 600.0


def _sigma_dot(shape, sc, amp=4.0e-5, seed=0):
    """Smooth random sigma_dot, zero at the top and the surface."""
    rng = np.random.default_rng(seed)
    s = np.asarray(sc.sigma_half, np.float64)
    a = rng.uniform(-1, 1, shape + (1,))
    b = rng.uniform(-1, 1, shape + (1,))
    sd = amp * (a * np.sin(np.pi * (s - s[0]) / (s[-1] - s[0]))
                + b * np.sin(2 * np.pi * (s - s[0]) / (s[-1] - s[0])))
    sd[..., 0] = 0.0
    sd[..., -1] = 0.0
    return jnp.asarray(sd)


def _q(shape, ntr=3, seed=1):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.0, 1.0, shape + (NLEV, ntr)))


@x64
def test_column_tracer_integral_conserved():
    """sum_k q_new dsigma_k == sum_k q dsigma_lag_k, every column/tracer."""
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((7, 5), sc)
    q = _q((7, 5))
    q2 = vertical_remap_sigma_fv3(q, sd, sc, DT)
    s = jnp.asarray(sc.sigma_half, jnp.float64)
    ds = s[1:] - s[:-1]
    ds_lag = ds + DT * (sd[..., 1:] - sd[..., :-1])
    before = jnp.sum(q * ds_lag[..., None], axis=-2)
    after = jnp.sum(q2 * ds[:, None], axis=-2)
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-14)
    # Non-vacuity: the Lagrangian layers really differ from the Eulerian
    # ones, so remapping as if they did not (q2 = q) breaks the budget.
    eul = jnp.sum(q * ds[:, None], axis=-2)
    assert float(jnp.max(jnp.abs(eul - before))) > 1e-6


@x64
def test_uniform_tracer_preserved_and_positive():
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((4, 3), sc)
    q2 = vertical_remap_sigma_fv3(jnp.full((4, 3, NLEV, 2), 0.7), sd, sc, DT)
    np.testing.assert_allclose(q2, 0.7, rtol=0, atol=1e-14)
    q3 = vertical_remap_sigma_fv3(_q((4, 3)), sd, sc, DT)
    assert float(jnp.min(q3)) >= 0.0


@x64
def test_zero_sigma_dot_is_identity():
    sc = create_sigma_coordinate(NLEV)
    q = _q((3, 2))
    q2 = vertical_remap_sigma_fv3(q, jnp.zeros((3, 2, NLEV + 1)), sc, DT)
    np.testing.assert_allclose(q2, q, rtol=0, atol=1e-14)


def test_refuses_boundary_flux():
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((2, 2), sc).at[..., -1].set(1e-6)
    with pytest.raises(Exception, match="nonzero at the top/surface"):
        jax.block_until_ready(jax.jit(
            lambda q: vertical_remap_sigma_fv3(q, sd, sc, DT))(_q((2, 2))))


def test_refuses_crossing_interfaces():
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((2, 2), sc, amp=5e-3)          # vertical CFL >> 1
    s = np.asarray(sc.sigma_half)
    assert np.min(np.diff(s) + DT * np.diff(np.asarray(sd), axis=-1)) <= 0
    # the GEOMETRY guard must fire, not only the downstream rezone guard
    with pytest.raises(Exception, match="interfaces cross/touch"):
        jax.block_until_ready(
            vertical_remap_sigma_fv3(_q((2, 2)), sd, sc, DT))


@x64
def test_refuses_exactly_touching_interfaces():
    """Zero Lagrangian thickness (dt=1: s[k+1] + (s[k]-s[k+1]) == s[k]
    exactly) is refused, not only negative thickness."""
    sc = create_sigma_coordinate(NLEV)
    s = np.asarray(sc.sigma_half, np.float64)
    k = 5
    sd = np.zeros((2, 2, NLEV + 1))
    sd[..., k + 1] = s[k] - s[k + 1]
    pe1 = s + 1.0 * sd
    assert np.min(np.diff(pe1, axis=-1)) == 0.0
    assert np.all(np.diff(pe1, axis=-1) >= 0.0)
    with pytest.raises(Exception, match="interfaces cross/touch"):
        jax.block_until_ready(vertical_remap_sigma_fv3(
            _q((2, 2)), jnp.asarray(sd), sc, 1.0))


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        check_tracer_vertical_scheme("ppm")
    with pytest.raises(ValueError):
        TracerTransportLatLonModel(
            create_latlon_grid(8, 16), create_sigma_coordinate(NLEV),
            lambda *a: None, TracerTransportLatLonConfig(vertical_scheme="x"))


@x64
def test_latlon_model_step_conserves_column_with_pure_vertical_wind():
    """Model path: u=v=0, so the step is the remap alone and the column
    budget closes on the model's own output."""
    grid = create_latlon_grid(8, 16)
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((8, 16), sc)

    def wind(t, lon2d, lat2d, sc_):
        z = jnp.zeros(lon2d.shape + (NLEV,))
        # u/v are evaluated at face coordinates too; only the cell-centre
        # call's sigma_dot is consumed.
        if lon2d.shape != sd.shape[:-1]:
            return z, z, jnp.zeros(lon2d.shape + (NLEV + 1,))
        return z, z, sd

    q = _q((8, 16))
    st = TracerState(
        tracers=Field(data=q, name="tracers",
                      dims=("lat", "lon", "lev", "tracer"), units="1"),
        time=Field(data=jnp.asarray(0.0), name="time", dims=(), units="s"))
    m = TracerTransportLatLonModel(
        grid, sc, wind, TracerTransportLatLonConfig(vertical_scheme="fv3_remap"))
    q2 = m.step(st, DT).tracers.data
    s = jnp.asarray(sc.sigma_half, jnp.float64)
    ds = s[1:] - s[:-1]
    ds_lag = ds + DT * (sd[..., 1:] - sd[..., :-1])
    np.testing.assert_allclose(jnp.sum(q2 * ds[:, None], axis=-2),
                               jnp.sum(q * ds_lag[..., None], axis=-2),
                               rtol=0, atol=1e-14)


def _zero_hwind_state(q):
    return TracerState(
        tracers=Field(data=q, name="tracers", dims=("x", "lev", "tracer"),
                      units="1"),
        time=Field(data=jnp.asarray(0.0), name="time", dims=(), units="s"))


@x64
def test_mpas_model_step_conserves_column_with_pure_vertical_wind():
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        TracerTransportMPASConfig, TracerTransportMPASModel)
    mesh = create_voronoi_mesh(2)
    nc, ne = mesh.lonCell.shape[0], mesh.angleEdge.shape[0]
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((nc,), sc)
    # time-dependent: the budget closes only with the MID-step sigma_dot
    m = TracerTransportMPASModel(
        mesh, sc, lambda t, me, s: (jnp.zeros((ne, NLEV)),
                                    sd * jnp.cos(t / 3.0e2)),
        TracerTransportMPASConfig(vertical_scheme="fv3_remap"))
    q = _q((nc,))
    q2 = m.step(_zero_hwind_state(q), DT).tracers.data
    s = jnp.asarray(sc.sigma_half, jnp.float64)
    ds = s[1:] - s[:-1]
    sd_mid = sd * jnp.cos(0.5 * DT / 3.0e2)
    ds_lag = ds + DT * (sd_mid[..., 1:] - sd_mid[..., :-1])
    np.testing.assert_allclose(jnp.sum(q2 * ds[:, None], axis=-2),
                               jnp.sum(q * ds_lag[..., None], axis=-2),
                               rtol=0, atol=1e-14)


@x64
def test_spectral_model_step_conserves_global_column_integral():
    """Spectral returned state: the projection back onto the band keeps the
    GLOBAL (Gaussian-quadrature) tracer integral exactly; individual columns
    are not kept (KNOWN DIFFERENCE, see the spectral module docstring)."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig,
        spectral_tracer_grid_values, spectral_tracer_state)
    sc = create_sigma_coordinate(NLEV)
    grid = create_gaussian_grid(10)
    sd = _sigma_dot((grid.n_lat, grid.n_lon), sc)

    def wind(t, g, s):
        z = jnp.zeros((g.n_lat, g.n_lon, NLEV))
        return z, z, sd

    m = SpectralTracerTransportModel(
        grid, sc, wind, TracerTransportSpectralConfig(
            vertical_scheme="fv3_remap"))
    grid = m.grid
    # UNtruncated start: above-band tracer content must still enter the
    # remap's budget (remap before truncation).
    st0 = spectral_tracer_state(grid, _q((grid.n_lat, grid.n_lon)))
    q = spectral_tracer_grid_values(grid, st0)
    q2 = spectral_tracer_grid_values(grid, m.step(st0, DT))
    s = jnp.asarray(sc.sigma_half, jnp.float64)
    ds = s[1:] - s[:-1]
    ds_lag = ds + DT * (sd[..., 1:] - sd[..., :-1])
    w = jnp.asarray(grid.weights)[:, None, None]          # Gaussian weights
    before = jnp.sum(w * jnp.sum(q * ds_lag[..., None], axis=-2), (0, 1))
    after = jnp.sum(w * jnp.sum(q2 * ds[:, None], axis=-2), (0, 1))
    np.testing.assert_allclose(after, before, rtol=1e-13, atol=0)


@x64
@pytest.mark.parametrize("tn,dt", [(11, 1800.0), (12, 600.0)])
def test_dcmip_matrix_winds_are_admissible(tn, dt):
    """The matrix's 1-1/1-2 sigma_dot, at EVERY step midpoint of one period
    at the configured dt, on the lat-lon 72x144 C-grid and Gaussian T21
    centres (the grids whose rows run the remap) and the MPAS ico5 centres,
    has exact boundary zeros and keeps every Lagrangian thickness positive
    in the remap's OWN interface arithmetic (pe1 = s_half + dt*sd in the
    tracer dtype, then differenced).  (The MPAS rows run their own
    mpas_fct coupling, not the analytic-sigma_dot remap; their cell centres
    stay in the sweep as a geometry check of the wind.)"""
    from tests.test_cases.dcmip_transport import (
        create_dcmip_sigma, dcmip11_wind_geo, dcmip12_wind_geo)
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.gaussian import create_gaussian_grid
    wind = {11: dcmip11_wind_geo, 12: dcmip12_wind_geo}[tn]
    period = {11: 12.0, 12: 1.0}[tn] * 86400.0
    sc = create_dcmip_sigma(30)
    s_half = jnp.asarray(sc.sigma_half, jnp.float64)
    g = create_latlon_grid(72, 144)
    m = create_voronoi_mesh(5)
    gg = create_gaussian_grid(21)
    points = [(g.lon2d, g.lat2d), (m.lonCell, m.latCell),
              (gg.lon2d, gg.lat2d)]

    @jax.jit
    def stats(t, lon, lat):
        sd = wind(t, lon, lat, sc)[2]
        pe1 = s_half + jnp.asarray(dt, jnp.float64) * sd
        return (jnp.max(jnp.abs(sd[..., 0])), jnp.max(jnp.abs(sd[..., -1])),
                jnp.min(pe1[..., 1:] - pe1[..., :-1]))

    for lon, lat in points:
        for n in range(int(round(period / dt))):
            top, bot, dmin = stats((n + 0.5) * dt, lon, lat)
            assert float(top) == 0.0 and float(bot) == 0.0
            assert float(dmin) > 0.0
        q = jnp.ones(lon.shape + (30, 1))
        q2 = vertical_remap_sigma_fv3(
            q, wind(0.5 * dt, lon, lat, sc)[2], sc, dt)
        np.testing.assert_allclose(q2, 1.0, rtol=0, atol=1e-13)


def test_float32_tracers_with_float64_dt():
    """One remap dtype even when dt is a strong float64 scalar."""
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((2, 2), sc).astype(jnp.float32)
    q = _q((2, 2)).astype(jnp.float32)
    q2 = vertical_remap_sigma_fv3(q, sd, sc, jnp.asarray(DT, jnp.float64)
                                  if jax.config.jax_enable_x64 else DT)
    assert q2.dtype == jnp.float32 and bool(jnp.all(jnp.isfinite(q2)))
    # Budget in float32 arithmetic (the remap's dtype): column integral
    # over the displaced layers, to float32 round-off.
    s32 = jnp.asarray(sc.sigma_half, jnp.float32)
    pe1 = s32 + jnp.float32(DT) * sd
    ds_lag = jnp.diff(pe1, axis=-1)
    ds = jnp.diff(s32)
    before = jnp.sum(q * ds_lag[..., None], axis=-2)
    after = jnp.sum(q2 * ds[:, None], axis=-2)
    np.testing.assert_allclose(after, before, rtol=0, atol=2e-6)
    # non-vacuity: an identity remap misses the budget by far more
    eul = jnp.sum(q * ds[:, None], axis=-2)
    assert float(jnp.max(jnp.abs(eul - before))) > 1e-4


@x64
def test_jit_parity_and_finite_gradient():
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((3, 2), sc)
    q = _q((3, 2))
    f = lambda x: vertical_remap_sigma_fv3(x, sd, sc, DT)  # noqa: E731
    # eager vs jit: XLA fusion reassociates at 1 ulp (measured 1.1e-16).
    np.testing.assert_allclose(f(q), jax.jit(f)(q), rtol=0, atol=1e-15)
    g = jax.grad(lambda x: jnp.sum(f(x) ** 2))(q)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0


@x64
def test_latlon_split_order_is_horizontal_then_remap():
    """Noncommuting winds (u and a lon-dependent sigma_dot): the model step
    equals horizontal-step-then-remap and differs from the reverse order."""
    grid = create_latlon_grid(8, 16)
    sc = create_sigma_coordinate(NLEV)
    sd0 = _sigma_dot((8, 16), sc)

    def wind(with_sd):
        def f(t, lon2d, lat2d, sc_):
            u = 20.0 * jnp.cos(lat2d)[..., None] * jnp.ones(NLEV)
            sd = (sd0 * (1.0 + jnp.cos(lon2d))[..., None] * jnp.cos(t / 3e4)
                  if (with_sd and lon2d.shape == sd0.shape[:-1])
                  else jnp.zeros(lon2d.shape + (NLEV + 1,)))
            return u, 0.0 * u, sd
        return f

    st = TracerState(
        tracers=Field(data=_q((8, 16)), name="tracers",
                      dims=("lat", "lon", "lev", "tracer"), units="1"),
        time=Field(data=jnp.asarray(0.0), name="time", dims=(), units="s"))
    cfg = TracerTransportLatLonConfig(vertical_scheme="fv3_remap")
    full = TracerTransportLatLonModel(grid, sc, wind(True), cfg).step(st, DT)
    horiz = TracerTransportLatLonModel(grid, sc, wind(False), cfg)
    sd_mid = wind(True)(0.5 * DT, grid.lon2d, grid.lat2d, sc)[2]
    h_then_v = vertical_remap_sigma_fv3(
        horiz.step(st, DT).tracers.data, sd_mid, sc, DT)
    np.testing.assert_allclose(full.tracers.data, h_then_v, rtol=0,
                               atol=1e-14)
    v_first = st._replace(tracers=st.tracers.replace(
        data=vertical_remap_sigma_fv3(st.tracers.data, sd_mid, sc, DT)))
    v_then_h = horiz.step(v_first, DT).tracers.data
    assert float(jnp.max(jnp.abs(v_then_h - h_then_v))) > 1e-8


@x64  # spectral transforms are x64-only in this repo
def test_spectral_remap_none_sigma_dot_and_missing_coordinate():
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig,
        spectral_tracer_state)
    grid = create_gaussian_grid(10)
    cfg = TracerTransportSpectralConfig(vertical_scheme="fv3_remap")

    def horiz(t, g, s):
        u = 10.0 * jnp.cos(g.lat2d)[..., None] * jnp.ones(NLEV)
        return u, 0.0 * u, None

    sc = create_sigma_coordinate(NLEV)
    a = SpectralTracerTransportModel(grid, sc, horiz, cfg)
    b = SpectralTracerTransportModel(grid, sc, horiz,
                                     TracerTransportSpectralConfig())
    st = a._truncate(spectral_tracer_state(a.grid, _q((a.grid.n_lat,
                                                       a.grid.n_lon))))
    # sigma_dot None: the remap is the identity, same as upwind (no vertical).
    np.testing.assert_array_equal(a.step(st, DT).tracers.data,
                                  b.step(st, DT).tracers.data)

    def with_sd(t, g, s):
        u, v, _ = horiz(t, g, s)
        return u, v, jnp.zeros((g.n_lat, g.n_lon, NLEV + 1))

    c = SpectralTracerTransportModel(grid, None, with_sd, cfg)
    with pytest.raises(ValueError, match="sigma_coord is None"):
        c.step(st, DT)


@x64
def test_spectral_remap_uses_current_sigma_coordinate():
    """The cached remap JIT takes the coordinate as an argument, so a
    same-shape coordinate swap on a warmed model is honoured."""
    from tests.test_cases.dcmip_transport import create_dcmip_sigma
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig,
        spectral_tracer_grid_values, spectral_tracer_state)
    sc1, sc2 = create_sigma_coordinate(NLEV), create_dcmip_sigma(NLEV)
    grid = create_gaussian_grid(10)
    sd = _sigma_dot((grid.n_lat, grid.n_lon), sc1, amp=1e-5)

    def wind(t, g, s):
        z = jnp.zeros((g.n_lat, g.n_lon, NLEV))
        return z, z, sd

    m = SpectralTracerTransportModel(
        grid, sc1, wind, TracerTransportSpectralConfig(
            vertical_scheme="fv3_remap"))
    st = m._truncate(spectral_tracer_state(m.grid, _q((m.grid.n_lat,
                                                       m.grid.n_lon))))
    m.step(st, DT)                                   # warm the cache on sc1
    m.sigma_coord = sc2
    got = spectral_tracer_grid_values(m.grid, m.step(st, DT))
    want = vertical_remap_sigma_fv3(
        spectral_tracer_grid_values(m.grid, st), sd, sc2, DT)
    want = spectral_tracer_grid_values(m.grid, m._truncate(
        spectral_tracer_state(m.grid, want)))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_refuses_nonfinite_coordinate():
    sc = create_sigma_coordinate(NLEV)
    sc_bad = sc._replace(sigma_half=jnp.asarray(sc.sigma_half).at[-1].set(
        jnp.inf))
    with pytest.raises(Exception, match="non-finite"):
        jax.block_until_ready(vertical_remap_sigma_fv3(
            _q((2, 2)), jnp.zeros((2, 2, NLEV + 1)), sc_bad, DT))
    # A collapsed TARGET layer (finite) whose Lagrangian layer is opened by
    # the displacement is refused too.
    s = np.asarray(sc.sigma_half, np.float64).copy()
    s[5] = s[4]
    sd = np.zeros((2, 2, NLEV + 1))
    sd[..., 4] = -1e-5
    assert np.min(np.diff(s + DT * sd, axis=-1)) > 0
    with pytest.raises(Exception, match="not positive"):
        jax.block_until_ready(vertical_remap_sigma_fv3(
            _q((2, 2)), jnp.asarray(sd), sc._replace(sigma_half=jnp.asarray(s)),
            DT))


@x64
def test_spectral_split_order_is_horizontal_then_remap():
    """Noncommuting winds: the spectral step remaps the horizontally
    advanced state (not the step's input state)."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig,
        spectral_tracer_grid_values, spectral_tracer_state)
    sc = create_sigma_coordinate(NLEV)
    grid = create_gaussian_grid(10)
    sd0 = _sigma_dot((grid.n_lat, grid.n_lon), sc)

    def wind(with_sd):
        def f(t, g, s):
            u = 20.0 * jnp.cos(g.lat2d)[..., None] * jnp.ones(NLEV)
            sd = (sd0 * (1.0 + jnp.cos(g.lon2d))[..., None]
                  * jnp.cos(t / 3.0e2) if with_sd else None)
            return u, 0.0 * u, sd
        return f

    cfg = TracerTransportSpectralConfig(vertical_scheme="fv3_remap")
    full = SpectralTracerTransportModel(grid, sc, wind(True), cfg)
    horiz = SpectralTracerTransportModel(grid, sc, wind(False), cfg)
    g = full.grid
    st = full._truncate(spectral_tracer_state(g, _q((g.n_lat, g.n_lon))))
    sd = wind(True)(0.5 * DT, g, sc)[2]
    # horizontal RK step (untruncated), then remap, then truncate
    from legoesm.timestepping.dispatch import dispatch_integrator
    h_raw = dispatch_integrator(st, horiz.tendencies, DT, "ssp_rk3")
    want = full._truncate(spectral_tracer_state(g, vertical_remap_sigma_fv3(
        spectral_tracer_grid_values(g, h_raw), sd, sc, DT)))
    got = full.step(st, DT)
    np.testing.assert_allclose(got.tracers.data, want.tracers.data,
                               rtol=0, atol=1e-13)
    wrong = full._truncate(spectral_tracer_state(g, vertical_remap_sigma_fv3(
        spectral_tracer_grid_values(g, st), sd, sc, DT)))
    assert float(jnp.max(jnp.abs(wrong.tracers.data - got.tracers.data))) \
        > 1e-8
    for t_end in (0.0, DT):          # endpoint wind samples differ
        sd_end = wind(True)(t_end, g, sc)[2]
        end = full._truncate(spectral_tracer_state(
            g, vertical_remap_sigma_fv3(
                spectral_tracer_grid_values(g, h_raw), sd_end, sc, DT)))
        assert float(jnp.max(jnp.abs(end.tracers.data
                                     - got.tracers.data))) > 1e-8
    # Truncate-before-remap cannot show here: the RK tendency is Orszag-
    # masked, so a band-limited input stays band-limited (measured diff
    # 2.2e-16).  That ordering is pinned by the global-budget test, which
    # starts from an UNtruncated state and fails under truncate-first.


def test_refuses_misshaped_sigma_half():
    sc = create_sigma_coordinate(NLEV)
    bad = sc._replace(sigma_half=jnp.asarray(sc.sigma_half)[None, :])
    with pytest.raises(ValueError, match="sigma_half shape"):
        vertical_remap_sigma_fv3(_q((2, 2)), jnp.zeros((2, 2, NLEV + 1)),
                                 bad, DT)


def test_nan_boundary_sigma_dot_reports_nonfinite():
    sc = create_sigma_coordinate(NLEV)
    sd = jnp.zeros((2, 2, NLEV + 1)).at[..., 0].set(jnp.nan)
    with pytest.raises(Exception, match="non-finite"):
        jax.block_until_ready(vertical_remap_sigma_fv3(_q((2, 2)), sd, sc, DT))


@x64
@pytest.mark.parametrize("grid_type,res,test_num,scheme", [
    ("latlon", "18x36", 12, "fv3_remap"),
    ("icosahedral", "ico2", 11, "upwind"),   # MPAS rows: mpas_fct coupling (merge of work/mpas-transport)
    ("latlon", "18x36", 13, "upwind"),
    ("spectral", "T10", 12, "fv3_remap"),
])
def test_matrix_runner_selects_scheme_and_audits_mass(
        tmp_path, monkeypatch, grid_type, res, test_num, scheme):
    """Runner level: the matrix row selects the vertical scheme explicitly
    (1-1/1-2 remap, 1-3 upwind) through its own wind adapter -- the model
    really calls (or skips) the remap -- and writes a per-tracer global mass
    audit that matches an independent recomputation."""
    import re
    import legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon as ML
    import legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas as MM
    import legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral as MS
    from scripts.matrix.run_atmosphere_test_matrix import (
        TestCase as MatrixCase, run_dcmip_transport)
    from tests.test_cases import dcmip_transport as D

    calls = []

    def counting(*a, **k):
        calls.append(1)
        return vertical_remap_sigma_fv3(*a, **k)
    for mod in (ML, MM, MS):   # spectral: its jit lambda reads the global
        monkeypatch.setattr(mod, "vertical_remap_sigma_fv3", counting)

    dt = {11: 1800.0, 12: 600.0, 13: 1800.0}[test_num]
    n_tr = {11: 4, 12: 1, 13: 4}[test_num]
    case = MatrixCase("hydrostatic", f"dcmip_transport_{test_num}",
                      grid_type, res, "sigma", 12, 1,
                      {"test_num": test_num})
    status, _, notes = run_dcmip_transport(case, tmp_path, 2 * dt / 86400.0)
    assert status == "PASS"
    text = (tmp_path / "results.txt").read_text()
    assert f"vertical_scheme: {scheme}" in text, text
    # the runner jits the step, so the remap is traced (called) once
    assert (len(calls) >= 1) == (scheme == "fv3_remap"), len(calls)

    num = r"([-+0-9.eE]+)"
    got = re.findall(rf"q(\d) M0={num} M1={num} rel={num}", notes)
    assert [int(g[0]) for g in got] == list(range(1, n_tr + 1)), notes
    for _, m0, m1, rel in got:
        m0, m1, rel = float(m0), float(m1), float(rel)
        np.testing.assert_allclose(rel, (m1 - m0) / m0, rtol=1e-6,
                                   atol=1e-12)
    # independent initial AND final mass (lat-lon rows): the analytic IC
    # and an independent 2-step run of the same model/scheme/winds
    if grid_type == "latlon":
        sc = D.create_dcmip_sigma(30)
        g = create_latlon_grid(*(int(x) for x in res.split("x")))
        init = {12: D.dcmip12_init_latlon, 13: D.dcmip13_init_latlon}[
            test_num]
        q0 = np.asarray(init(g, sc).tracers.data, np.float64)
        dsg = np.diff(np.asarray(sc.sigma_half, np.float64))
        m0_ref = np.einsum("ijkt,k,ij->t", q0, dsg,
                           np.asarray(g.area, np.float64))
        np.testing.assert_allclose([float(x[1]) for x in got], m0_ref,
                                   rtol=1e-6)
        wind = {12: D.dcmip12_wind_geo, 13: D.dcmip13_wind_geo}[test_num]
        m = TracerTransportLatLonModel(
            g, sc, wind, TracerTransportLatLonConfig(vertical_scheme=scheme))
        st = init(g, sc)
        dt_run = 120.0 if test_num == 13 else dt   # the row's lat-lon 1-3 dt
        for _ in range(int(2 * dt / 86400.0 * 86400.0 / dt_run)):
            st = m.step(st, dt_run)
        q1 = np.asarray(st.tracers.data, np.float64)
        m1_ref = np.einsum("ijkt,k,ij->t", q1, dsg,
                           np.asarray(g.area, np.float64))
        np.testing.assert_allclose([float(x[2]) for x in got], m1_ref,
                                   rtol=1e-12)
        # non-vacuity: the final mass differs from the initial one
        assert np.max(np.abs(m1_ref - m0_ref) / np.abs(m0_ref)) > 1e-12
    elif grid_type == "icosahedral":
        # independent M0/M1 with the Voronoi cell areas
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
            TracerTransportMPASConfig, TracerTransportMPASModel)
        sc = D.create_dcmip_sigma(30)
        me = create_voronoi_mesh(int(res.replace("ico", "")))
        # the row's model: MPAS-A scalar transport on the streamfunction
        # edge winds and the prognostic air mass
        m = TracerTransportMPASModel(
            me, sc, lambda t, mesh, s_: D.dcmip_wind_mpas(11, t, mesh, s_),
            TracerTransportMPASConfig(hyperdiff_coeff=0.0, time_integrator="mpas_rk3",
                                      horizontal_scheme="mpas_fct", coef_3rd_order=0.25,
                                      fct_smoothing_width=1.0e-4, fct_upwind_width_ms=1.0e-3,
                                      vertical_scheme=scheme))
        st = D.dcmip_air_mass_state_mpas(11, D.dcmip11_init_mpas(me, sc), me, sc)
        ar = np.asarray(me.areaCell, np.float64)
        # the prognostic tracer mass sum(area q dp)
        mass = lambda s_: np.einsum("ckt,ck,c->t", np.asarray(s_.tracers.data, np.float64),  # noqa: E731
                                    np.asarray(s_.air_mass.data, np.float64), ar)
        m0_ref = mass(st)
        for _ in range(2):
            st = m.step(st, dt)
        m1_ref = mass(st)
        np.testing.assert_allclose([float(x[1]) for x in got], m0_ref,
                                   rtol=1e-10)
        # the runner's jitted MPAS step and this eager one differ in
        # summation order (measured 3.7e-10 relative); the drift being
        # audited is far larger (non-vacuity below)
        np.testing.assert_allclose([float(x[2]) for x in got], m1_ref,
                                   rtol=1e-8)
        assert np.max(np.abs(m1_ref - m0_ref) / np.abs(m0_ref)) > 1e-7
    else:
        # spectral: independent M1/M0 per tracer with the Gaussian
        # quadrature weights (normalization-free ratio)
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
            SpectralTracerTransportModel, TracerTransportSpectralConfig,
            spectral_tracer_grid_values, spectral_tracer_state)
        sc = D.create_dcmip_sigma(30)
        m = SpectralTracerTransportModel(
            create_gaussian_grid(int(res.replace("T", ""))), sc,
            lambda t, g_, s_: D.dcmip12_wind_geo(t, g_.lon2d, g_.lat2d, s_),
            TracerTransportSpectralConfig(vertical_scheme=scheme))
        st = m._truncate(spectral_tracer_state(m.grid, D.dcmip12_init_latlon(
            m.grid, sc).tracers.data))                # as the runner does
        dsg = np.diff(np.asarray(sc.sigma_half, np.float64))
        w = np.asarray(m.grid.weights, np.float64)
        mass = lambda s_: np.einsum(  # noqa: E731
            "ijkt,k,i->t", np.asarray(
                spectral_tracer_grid_values(m.grid, s_), np.float64), dsg, w)
        # absolute masses: Gaussian quadrature cell area
        # w_j * 2 pi R^2 / n_lon (independent of the runner's grid_area)
        scale = 2.0 * np.pi * float(m.grid.radius) ** 2 / m.grid.n_lon
        m0_ref = mass(st) * scale
        for _ in range(2):
            st = m.step(st, dt)
        m1_ref = mass(st) * scale
        np.testing.assert_allclose([float(x[1]) for x in got], m0_ref,
                                   rtol=1e-10)
        np.testing.assert_allclose([float(x[2]) for x in got], m1_ref,
                                   rtol=1e-10)




# Base-revision (97ddee4056) default-"upwind" outputs, DCMIP 1-2 winds WITH
# their nonzero sigma_dot, 2 x 600 s steps: (sum q, sum q^2, sum w*q) with a
# fixed positional weight w ~ U(0.5, 1.5) (rng seed 0, shape of q), generated
# from the base-revision module copies.  A NUMERICAL REGRESSION check of the
# default path (in-tendency upwind vertical advection), rtol 1e-12 -- not a
# bit-identity proof: bit identity of the default path was verified
# separately by running the base and new modules side by side on the same
# backend (array_equal True for all three models).  The positional moment
# catches permutations; no full-array binary fixture is committed (size, and
# bit patterns differ across XLA backends).
_BASE_UPWIND_REF = {
    "latlon": (2428.8721675741117, 1730.3078113610798, 2428.8709307728004),
    "mpas": (607.3954534877439, 430.1170642353832, 599.0134423828337),
    "spectral": (1921.0238201319662, 1370.6325274893175, 1918.8867773510988),
}


@x64
@pytest.mark.parametrize("which", ["latlon", "mpas", "spectral"])
def test_default_upwind_path_matches_base_revision(which):
    from tests.test_cases.dcmip_transport import (
        create_dcmip_sigma, dcmip12_wind_geo, dcmip12_init_latlon,
        dcmip12_init_mpas)
    sc = create_dcmip_sigma(30)

    def run(vert_on):
        def geo(t, lon, lat, s_):
            u, v, sd = dcmip12_wind_geo(t, lon, lat, s_)
            return u, v, sd if vert_on else 0.0 * sd
        if which == "latlon":
            g = create_latlon_grid(18, 36)
            s = dcmip12_init_latlon(g, sc)
            m = TracerTransportLatLonModel(g, sc, geo,
                                           TracerTransportLatLonConfig())
            for _ in range(2):
                s = m.step(s, 600.0)
            return s.tracers.data
        if which == "mpas":
            from legoesm.grids.voronoi import create_voronoi_mesh
            from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
                TracerTransportMPASConfig, TracerTransportMPASModel)
            me = create_voronoi_mesh(2)
            s = dcmip12_init_mpas(me, sc)

            def mw(t, mesh, s_):
                u, v, sd = geo(t, mesh.lonCell, mesh.latCell, s_)
                c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
                ca = jnp.cos(mesh.angleEdge)[:, None]
                sa = jnp.sin(mesh.angleEdge)[:, None]
                return 0.5 * (u[c1] * ca + v[c1] * sa
                              + u[c2] * ca + v[c2] * sa), sd
            m = TracerTransportMPASModel(me, sc, mw,
                                         TracerTransportMPASConfig())
            for _ in range(2):
                s = m.step(s, 600.0)
            return s.tracers.data
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
            SpectralTracerTransportModel, TracerTransportSpectralConfig,
            spectral_tracer_grid_values, spectral_tracer_state)
        m = SpectralTracerTransportModel(
            create_gaussian_grid(10), sc,
            lambda t, gr, s_: geo(t, gr.lon2d, gr.lat2d, s_),
            TracerTransportSpectralConfig())
        s = m._truncate(spectral_tracer_state(
            m.grid, dcmip12_init_latlon(m.grid, sc).tracers.data))
        for _ in range(2):
            s = m.step(s, 600.0)
        return spectral_tracer_grid_values(m.grid, s)

    q = np.asarray(run(True), np.float64)
    ref = _BASE_UPWIND_REF[which]
    w = np.random.default_rng(0).uniform(0.5, 1.5, q.shape)
    np.testing.assert_allclose([q.sum(), (q * q).sum(), (w * q).sum()], ref,
                               rtol=1e-12)
    # non-vacuity: dropping the vertical advection moves the checksum
    q0 = np.asarray(run(False), np.float64)
    assert abs(q0.sum() - ref[0]) / ref[0] > 1e-6


@x64
def test_mixed_precision_displacement_not_lost():
    """float64 tracers, float32 winds, a (weakly typed) Python dt far below
    the float32 normal range: the displacement is formed in the widest
    dtype, so the column budget closes against the float64 displacement."""
    sc = create_sigma_coordinate(NLEV)
    sd32 = (np.asarray(_sigma_dot((3, 2), sc)) * 1e42).astype(np.float32)
    dt = DT * 1e-42                      # 6e-40: subnormal in float32
    q = _q((3, 2))
    q2 = vertical_remap_sigma_fv3(q, jnp.asarray(sd32), sc, dt)
    s = np.asarray(sc.sigma_half, np.float64)
    pe1 = s + dt * sd32.astype(np.float64)
    before = np.sum(np.asarray(q) * np.diff(pe1, axis=-1)[..., None], -2)
    after = np.sum(np.asarray(q2) * np.diff(s)[:, None], -2)
    np.testing.assert_allclose(after, before, rtol=0, atol=1e-14)


@x64
def test_refuses_subnormal_boundary_wind_under_jit():
    """A subnormal (finite, nonzero) boundary sigma_dot is refused even under
    jit / on backends that flush subnormals; a signed zero is accepted."""
    sc = create_sigma_coordinate(NLEV)
    f = jax.jit(lambda q, sd: vertical_remap_sigma_fv3(q, sd, sc, DT))
    zero = jnp.zeros((2, 2, NLEV + 1))
    jax.block_until_ready(f(_q((2, 2)), zero.at[..., -1].set(-0.0)))
    with pytest.raises(Exception, match="nonzero at the top/surface"):
        jax.block_until_ready(f(_q((2, 2)), zero.at[..., -1].set(5e-324)))


@x64
def test_step_profile_layer_values_independent():
    """Independent expected values (rejects e.g. column homogenization): a
    step profile (1 above, 0 below) whose interior interfaces all move down
    by d < the thinnest layer.  Target layers two or more layers away from
    the step see only flat data and stay exactly 1 / 0; the layer just
    below the step receives tracer (0 < q < 1); the column budget closes."""
    sc = create_sigma_coordinate(NLEV)
    s = np.asarray(sc.sigma_half, np.float64)
    d = 0.25 * np.min(np.diff(s))
    sd = np.zeros((1, NLEV + 1))
    sd[0, 1:-1] = d / DT
    q = np.zeros((1, NLEV, 1))
    q[0, :6, 0] = 1.0
    q2 = np.asarray(vertical_remap_sigma_fv3(jnp.asarray(q), jnp.asarray(sd),
                                             sc, DT))[0, :, 0]
    np.testing.assert_allclose(q2[:4], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(q2[8:], 0.0, rtol=0, atol=1e-12)
    assert 0.0 < q2[6] < 1.0
    pe1 = s + DT * sd[0]
    np.testing.assert_allclose(np.sum(q2 * np.diff(s)),
                               np.sum(q[0, :, 0] * np.diff(pe1)),
                               rtol=0, atol=1e-14)


@x64
def test_refuses_subnormal_interior_wind_and_integer_winds():
    sc = create_sigma_coordinate(NLEV)
    f = jax.jit(lambda q, sd: vertical_remap_sigma_fv3(q, sd, sc, DT))
    sd = jnp.zeros((2, 2, NLEV + 1)).at[..., 5].set(1e-310)
    with pytest.raises(Exception, match="subnormal"):
        jax.block_until_ready(f(_q((2, 2)), sd))
    with pytest.raises(ValueError, match="real floating dtype"):
        vertical_remap_sigma_fv3(_q((2, 2)), jnp.zeros((2, 2, NLEV + 1),
                                                       jnp.int32), sc, DT)


@x64
def test_smooth_profile_is_high_order_not_first_order():
    """Independent reference distinguishing the kord-9 PPM reconstruction
    from a first-order (piecewise-constant) overlap remap: cell means of
    q(s) = (s+1)^2, every interior interface displaced down by d; the exact
    answer is the cell mean of q(s - d).  Interior layers (away from the
    boundary-edge treatment) must be within 1e-4 of exact, where the
    first-order overlap is off by > 1e-3."""
    sc = create_sigma_coordinate(NLEV)
    s = np.asarray(sc.sigma_half, np.float64)
    ds = np.diff(s)
    d = 0.25 * np.min(ds)
    F = lambda x: (x + 1.0) ** 3 / 3.0          # noqa: E731  antiderivative
    qm = (F(s[1:]) - F(s[:-1])) / ds
    sd = np.zeros((1, NLEV + 1))
    sd[0, 1:-1] = d / DT
    q2 = np.asarray(vertical_remap_sigma_fv3(
        jnp.asarray(qm[None, :, None]), jnp.asarray(sd), sc, DT))[0, :, 0]
    exact = (F(s[1:] - d) - F(s[:-1] - d)) / ds
    first = qm * (ds - d) / ds
    first[1:] += qm[:-1] * d / ds[1:]
    interior = slice(4, NLEV - 4)
    assert np.max(np.abs(q2 - exact)[interior]) < 1e-4
    assert np.min(np.abs(first - exact)[interior]) > 1e-3


@x64
def test_refuses_crossing_hidden_by_float32_cast():
    """float64 winds cross two interfaces by 1e-9 (below float32 resolution
    of sigma); the geometry is validated in the promoted (float64) dtype,
    where the thickness is negative, so the float32-tracer remap refuses."""
    sc = create_sigma_coordinate(NLEV)
    s = np.asarray(sc.sigma_half, np.float64)
    sd = np.zeros((2, 2, NLEV + 1))
    sd[..., 3] = (s[4] - s[3] + 1e-9) / DT
    assert np.min(np.diff(s + DT * sd, axis=-1)) < 0      # crosses in f64
    with pytest.raises(Exception, match="interfaces cross/touch"):
        jax.block_until_ready(vertical_remap_sigma_fv3(
            _q((2, 2)).astype(jnp.float32), jnp.asarray(sd), sc, DT))


@x64
def test_coordinate_dtype_enters_promotion():
    """float32 tracers / wind / dt with float64 coordinates: interface 3
    (0.25 + 0.49 ulp_f32, rounds down) displaced by a float32 amount crosses
    interface 4 (f32(1/3) - 0.49 ulp_f32, rounds up) in float64 by ~1e-9,
    while rounding the coordinates to float32 FIRST would leave a strictly
    positive layer.  The geometry is formed in the promoted dtype including
    the coordinate's, so the crossing is refused."""
    f = np.float32
    s3 = float(f(0.25)) + 0.49 * float(np.spacing(f(0.25)))
    s4 = float(f(1 / 3)) - 0.49 * float(np.spacing(f(1 / 3)))
    disp = f(float(f(s4 - s3)) + float(np.spacing(f(s4 - s3))))
    assert s3 + float(disp) > s4                        # crosses in f64
    assert f(f(s3) + disp) < f(s4)                      # hidden in f32
    s = np.concatenate([[0.0, 0.08, 0.16, s3, s4],
                        np.linspace(0.42, 1.0, NLEV + 1 - 5)])
    sc = create_sigma_coordinate(NLEV)._replace(
        sigma_half=jnp.asarray(s, jnp.float64))
    sd = np.zeros((2, 2, NLEV + 1), np.float32)
    sd[..., 3] = disp
    with pytest.raises(Exception, match="interfaces cross/touch"):
        jax.block_until_ready(vertical_remap_sigma_fv3(
            _q((2, 2)).astype(jnp.float32), jnp.asarray(sd), sc,
            jnp.asarray(1.0, jnp.float32)))


@x64
def test_refuses_underflowing_displacement():
    """A normal nonzero wind whose dt*sigma_dot underflows is refused."""
    sc = create_sigma_coordinate(NLEV)
    sd = jnp.zeros((2, 2, NLEV + 1)).at[..., 5].set(1e-300)
    with pytest.raises(Exception, match="underflowing"):
        jax.block_until_ready(vertical_remap_sigma_fv3(_q((2, 2)), sd, sc,
                                                       1e-10))


@x64
def test_kernel_runs_in_promoted_dtype():
    """float32 tracers with float64 winds/coordinates, a thin layer 3
    (1e-9 thick) whose lower interface moves UP by 0.4e-9: the result equals
    the float64 remap cast to float32 (the displacement is not erased by an
    early cast)."""
    s = np.concatenate([[0.0, 0.1, 0.2, 0.3, 0.3 + 1e-9],
                        np.linspace(0.4, 1.0, NLEV + 1 - 5)])
    sc = create_sigma_coordinate(NLEV)._replace(
        sigma_half=jnp.asarray(s, jnp.float64))
    sd = np.zeros((2, 2, NLEV + 1))
    sd[..., 4] = -0.4e-9 / DT
    # step profile (0 down to the thin layer 3, 1 below it): the thin layer
    # takes 40 % of its new content from layer 4 below
    q = jnp.asarray(np.broadcast_to(
        (np.arange(NLEV) >= 4).astype(np.float32)[:, None], (2, 2, NLEV, 3)))
    got = vertical_remap_sigma_fv3(q, jnp.asarray(sd), sc, DT)
    want = vertical_remap_sigma_fv3(q.astype(jnp.float64), jnp.asarray(sd),
                                    sc, DT).astype(jnp.float32)
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert float(jnp.max(jnp.abs(got - q))) > 0.1        # the layer moved


@x64
def test_zero_dt_is_identity_and_dt_must_be_scalar():
    sc = create_sigma_coordinate(NLEV)
    sd = _sigma_dot((2, 2), sc)
    q = _q((2, 2))
    np.testing.assert_allclose(
        np.asarray(vertical_remap_sigma_fv3(q, sd, sc, 0.0)), np.asarray(q),
        rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="real scalar"):
        vertical_remap_sigma_fv3(q, sd, sc, jnp.full((NLEV + 1,), DT))


def _one_ulp_layer_coord():
    """float32 coordinate with a ONE-ulp-thick layer 3 at sigma 0.3."""
    f = np.float32
    s3 = f(0.3)
    s4 = np.nextafter(s3, f(1.0))
    s = np.concatenate([np.array([0.0, 0.1, 0.2], f), [s3, s4],
                        np.linspace(0.4, 1.0, NLEV + 1 - 5).astype(f)])
    return (create_sigma_coordinate(NLEV)._replace(
        sigma_half=jnp.asarray(s, jnp.float32)),
        float(s4) - float(s3))


@x64
def test_refuses_displacement_rounded_away_and_dt_promotes():
    """All-float32 inputs: moving the lower interface of a one-ulp layer by
    0.4 ulp rounds away in float32 -> refused.  The SAME motion with dt as
    the ONLY float64 input promotes the geometry to float64, where the
    motion is exact and the thin layer's value changes by O(0.4)."""
    sc, ulp = _one_ulp_layer_coord()
    q = jnp.asarray(np.broadcast_to(
        (np.arange(NLEV) >= 4).astype(np.float32)[:, None], (2, 2, NLEV, 1)))
    sd = np.zeros((2, 2, NLEV + 1), np.float32)
    sd[..., 4] = np.float32(-0.4 * ulp)
    with pytest.raises(Exception, match="not representable"):
        jax.block_until_ready(vertical_remap_sigma_fv3(
            q, jnp.asarray(sd), sc, jnp.asarray(1.0, jnp.float32)))
    got = vertical_remap_sigma_fv3(q, jnp.asarray(sd), sc,
                                   jnp.asarray(1.0, jnp.float64))
    assert got.dtype == jnp.float32
    assert float(jnp.max(jnp.abs(got - q))) > 0.1


@x64
def test_refuses_subnormal_dt():
    sc = create_sigma_coordinate(NLEV)
    with pytest.raises(Exception, match="underflowing"):
        jax.block_until_ready(vertical_remap_sigma_fv3(
            _q((2, 2)), _sigma_dot((2, 2), sc), sc,
            jnp.asarray(5e-324, jnp.float64)))


@x64
def test_mpas_split_order_is_horizontal_then_remap():
    """Noncommuting winds on the MPAS model: the step equals an
    independently assembled horizontal SSP-RK step (the model's own
    horizontal operator, no vertical) followed by the remap with the
    mid-step sigma_dot, and differs from remapping the step's input."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.timestepping.dispatch import dispatch_integrator
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        TracerTransportMPASConfig, TracerTransportMPASModel,
        tracer_horizontal_advection)
    mesh = create_voronoi_mesh(2)
    nc, ne = mesh.lonCell.shape[0], mesh.angleEdge.shape[0]
    sc = create_sigma_coordinate(NLEV)
    sd0 = _sigma_dot((nc,), sc)
    lon = jnp.asarray(mesh.lonCell)
    u_e = 20.0 * jnp.cos(jnp.asarray(mesh.angleEdge))[:, None] * jnp.ones(
        NLEV)

    def wind(t, me, s_):
        return u_e, sd0 * (1.0 + jnp.cos(lon))[:, None] * jnp.cos(t / 3.0e2)

    m = TracerTransportMPASModel(
        mesh, sc, wind, TracerTransportMPASConfig(vertical_scheme="fv3_remap"))
    st = _zero_hwind_state(_q((nc,)))
    got = m.step(st, DT).tracers.data

    def horiz_tend(s_):
        return s_._replace(
            tracers=s_.tracers.replace(
                data=tracer_horizontal_advection(s_.tracers.data, u_e, mesh)),
            time=s_.time.replace(data=jnp.asarray(1.0)))
    h = dispatch_integrator(st, horiz_tend, DT, "ssp_rk3")
    sd_mid = wind(0.5 * DT, mesh, sc)[1]
    want = vertical_remap_sigma_fv3(h.tracers.data, sd_mid, sc, DT)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)
    wrong = vertical_remap_sigma_fv3(st.tracers.data, sd_mid, sc, DT)
    assert float(jnp.max(jnp.abs(wrong - got))) > 1e-8


def test_host_dt_underflow_refused():
    """A nonzero host dt that the array conversion would turn into zero /
    a subnormal is refused (x64 off: float32 conversion of 1e-46)."""
    sc = create_sigma_coordinate(NLEV)
    if jax.config.jax_enable_x64:
        dt = 1e-320                    # subnormal even in float64
    else:
        dt = 1e-46                     # zero in float32
    with pytest.raises(ValueError, match="underflows"):
        vertical_remap_sigma_fv3(_q((2, 2)), jnp.zeros((2, 2, NLEV + 1)),
                                 sc, dt)

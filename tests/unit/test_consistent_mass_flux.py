"""Discretely consistent prescribed air fluxes (DCMIP 1-3) on the lat-lon,
MPAS and spectral cores: the corrected horizontal air flux closes each
layer's continuity with the analytic vertical air flux to round-off (and
the uncorrected one does not), the vertical flux stays the analytic one up
to its removed global mean, a uniform tracer stays uniform, and the global
tracer mass ``sum area dp q`` changes only by the tracer flux through the
open top and surface (RULE 0 budget, x64)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm import consistent_mass_flux as cmf  # noqa: E402
from tests.test_cases.dcmip_transport import (  # noqa: E402
    _H, _mountain_height, _p0, create_dcmip_sigma, dcmip13_init_latlon,
    dcmip13_init_mpas, dcmip13_wind_geo, dcmip_air_mass_state_mpas, dcmip_wind_mpas)

NLEV = 12


@pytest.fixture(scope="module")
def sc():
    return create_dcmip_sigma(NLEV)


def _ps(lon, lat):
    return _p0 * jnp.exp(-_mountain_height(lon, lat) / _H)


@pytest.fixture(scope="module")
def latlon(sc):
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(18, 36)
    ps = _ps(g.lon2d, g.lat2d)
    return g, cmf.consistent_mass_flux_latlon(g, sc, dcmip13_wind_geo, ps), \
        cmf.consistent_mass_flux_latlon(g, sc, dcmip13_wind_geo, ps, refine=0), ps


@pytest.fixture(scope="module")
def mpas(sc):
    from legoesm.grids.voronoi import create_voronoi_mesh
    m = create_voronoi_mesh(2)
    ue, sd = dcmip_wind_mpas(13, 0.0, m, sc)
    ps = _ps(m.lonCell, m.latCell)
    return m, cmf.consistent_mass_flux_mpas(m, sc, ue, sd, ps), \
        cmf.consistent_mass_flux_mpas(m, sc, ue, sd, ps, refine=0), ps, sd


@pytest.fixture(scope="module")
def spectral(sc):
    from legoesm.grids.gaussian import create_gaussian_grid, dealiasing_mask
    g = create_gaussian_grid(10)
    mask = dealiasing_mask(g, 0.667)
    ps = _ps(g.lon2d, g.lat2d)
    return g, mask, cmf.consistent_mass_flux_spectral(g, sc, dcmip13_wind_geo, ps, mask), \
        cmf.consistent_mass_flux_spectral(g, sc, dcmip13_wind_geo, ps, mask, refine=0)


def _rel(resid, mf):
    return float(np.max(np.abs(np.asarray(resid)) / np.asarray(mf.dp)))


def test_closure_latlon(latlon):
    g, mf, mf0, _ = latlon
    assert _rel(cmf.continuity_residual_latlon(mf, g), mf) < 1e-17
    assert _rel(cmf.continuity_residual_latlon(mf0, g), mf0) > 1e-8   # the correction matters
    fu, fv = mf.flux_h
    assert np.all(np.asarray(fv[0]) == 0) and np.all(np.asarray(fv[-1]) == 0)   # no pole flux
    np.testing.assert_array_equal(np.asarray(fu[:, -1]), np.asarray(fu[:, 0]))  # periodic


def test_closure_mpas(mpas):
    m, mf, mf0, _, _ = mpas
    assert _rel(cmf.continuity_residual_mpas(mf, m), mf) < 1e-17
    assert _rel(cmf.continuity_residual_mpas(mf0, m), mf0) > 1e-8


def test_closure_spectral(spectral):
    g, mask, mf, mf0 = spectral
    assert _rel(cmf.continuity_residual_spectral(mf, g, mask), mf) < 1e-17
    assert _rel(cmf.continuity_residual_spectral(mf0, g, mask), mf0) > 1e-8


def test_vertical_flux_is_the_analytic_one_up_to_its_mean(mpas, sc):
    m, mf, _, ps, sd = mpas
    w_an = np.asarray(ps, np.float64)[:, None] * np.asarray(sd, np.float64)
    np.testing.assert_allclose(np.asarray(mf.w) + np.asarray(mf.w_mean_removed), w_an,
                               rtol=0, atol=1e-12 * np.abs(w_an).max())
    # open top and surface (the analytic flow crosses both)
    assert np.abs(np.asarray(mf.w[:, 0])).max() > 0 and np.abs(np.asarray(mf.w[:, -1])).max() > 0
    a = np.asarray(m.areaCell)
    means = a @ np.asarray(mf.w)          # solvability: equal interface means
    np.testing.assert_allclose(means, means[0], rtol=0, atol=1e-12 * a.sum() * np.abs(w_an).max())
    assert np.asarray(mf.w_mean_removed)[0] == 0.0


def test_uniform_throughflow_is_kept(sc):
    """A horizontally uniform W, equal at every interface, already closes
    the continuity (no wind): the builders keep it (lat-lon and spectral)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.gaussian import create_gaussian_grid, dealiasing_mask

    def through(t, lon, lat, s):
        z = jnp.zeros(jnp.shape(lat) + (s.n_levels,))
        return z, z, jnp.full(jnp.shape(lat) + (s.n_levels + 1,), 1.0e-6)
    g = create_latlon_grid(18, 36)
    mf = cmf.consistent_mass_flux_latlon(g, sc, through, jnp.full(g.lon2d.shape, 1.0e5))
    np.testing.assert_allclose(np.asarray(mf.w), 0.1, rtol=1e-12)
    gg = create_gaussian_grid(10)
    mfs = cmf.consistent_mass_flux_spectral(gg, sc, through, jnp.full(gg.lon2d.shape, 1.0e5),
                                            dealiasing_mask(gg, 0.667))
    np.testing.assert_allclose(np.asarray(mfs.w), 0.1, rtol=1e-12)


def test_donor_cell_vertical_flux():
    q = jnp.array([[[1.0], [2.0], [3.0]]])                # (1, nlev=3, ntr=1)
    w = jnp.array([[0.5, 1.0, -2.0, -4.0]])               # (1, nlev+1), + down
    g = np.asarray(cmf.donor_cell_vertical_flux(q, w))[0, :, 0]
    np.testing.assert_array_equal(g, [0.5 * 1.0, 1.0 * 1.0, -2.0 * 3.0, -4.0 * 3.0])


def _mass(q, dp, area):
    w = np.asarray(area, np.float64)[..., None] * np.asarray(dp, np.float64)
    q = np.asarray(q, np.float64)
    return (w[..., None] * q).reshape(-1, q.shape[-1]).sum(axis=0)


def test_latlon_model_uniform_and_mass(latlon, sc):
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
        TracerTransportLatLonModel)
    g, mf, _, _ = latlon
    model = TracerTransportLatLonModel(g, sc, lambda *a: None, mass_flux=mf)
    st = dcmip13_init_latlon(g, sc)
    st = st._replace(tracers=st.tracers.replace(data=st.tracers.data.astype(jnp.float64)))
    one = model.step(st._replace(tracers=st.tracers.replace(data=jnp.ones_like(st.tracers.data))),
                     600.0)
    assert float(jnp.max(jnp.abs(one.tracers.data - 1.0))) < 1e-13
    m0, s = _mass(st.tracers.data, mf.dp, g.area), st
    for _ in range(4):
        s = model.step(s, 600.0)
    # the clouds start far from the open top / surface layers: no boundary flux
    np.testing.assert_allclose(_mass(s.tracers.data, mf.dp, g.area), m0, rtol=1e-13)


def test_mpas_model_uniform_mass_and_steady_air(mpas, sc):
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        TracerTransportMPASConfig, TracerTransportMPASModel)
    m, mf, _, _, _ = mpas
    cfg = TracerTransportMPASConfig(hyperdiff_coeff=0.0, time_integrator="mpas_rk3",
                                    vertical_scheme="mpas_flux", horizontal_scheme="mpas_fct")
    model = TracerTransportMPASModel(
        m, sc, lambda t, mesh, s: dcmip_wind_mpas(13, t, mesh, s), cfg, mass_flux=mf)
    st = dcmip_air_mass_state_mpas(13, dcmip13_init_mpas(m, sc), m, sc)
    one = model.step(st._replace(tracers=st.tracers.replace(data=jnp.ones_like(st.tracers.data))),
                     600.0)
    assert float(jnp.max(jnp.abs(one.tracers.data - 1.0))) < 1e-13
    m0, s = _mass(st.tracers.data, st.air_mass.data, m.areaCell), st
    for _ in range(4):
        s = model.step(s, 600.0)
    np.testing.assert_allclose(_mass(s.tracers.data, s.air_mass.data, m.areaCell), m0, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(s.air_mass.data), np.asarray(st.air_mass.data),
                               rtol=1e-13)
    with pytest.raises(ValueError):
        TracerTransportMPASModel(m, sc, None, TracerTransportMPASConfig(), mass_flux=mf)


def test_mpas_open_boundary_flux_is_the_only_mass_change(mpas, sc):
    """A tracer in the surface layer leaves/enters with the open-surface air
    flux: the mass change equals the integrated boundary tracer flux."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        TracerTransportMPASConfig, mpas_rk3_fct_step)
    from legoesm.core.operators_voronoi import mpas_advection_coefficients
    m, mf, _, _, _ = mpas
    cfg = TracerTransportMPASConfig(hyperdiff_coeff=0.0, time_integrator="mpas_rk3",
                                    vertical_scheme="mpas_flux", horizontal_scheme="mpas_fct")
    st = dcmip_air_mass_state_mpas(13, dcmip13_init_mpas(m, sc), m, sc)
    # surface-layer tracer only where air leaves through the surface (W > 0):
    # the boundary flux is then one-signed (the area-mean-free W would give a
    # zero net flux for a uniform surface tracer)
    out = (mf.w[:, -1] > 0).astype(st.tracers.data.dtype)
    q = jnp.zeros_like(st.tracers.data).at[:, -1, :].set(out[:, None])
    st = st._replace(tracers=st.tracers.replace(data=q))
    dt = 60.0
    new = mpas_rk3_fct_step(st, dt, m, sc, None, cfg, mpas_advection_coefficients(m), mf)
    a = np.asarray(m.areaCell)
    m_before = _mass(q, st.air_mass.data, a)
    dm = _mass(new.tracers.data, new.air_mass.data, a) - m_before
    # low-order boundary flux F q0 through the surface, positive down = out
    expect = -dt * float(a @ (np.asarray(mf.w[:, -1]) * np.asarray(out)))
    assert abs(expect) > 1e-4 * m_before[0]          # an appreciable loss
    np.testing.assert_allclose(dm, expect, rtol=1e-10, atol=1e-13 * m_before[0])


def test_spectral_model_uniform_and_mass(spectral, sc):
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig)
    g, mask, mf, _ = spectral
    model = SpectralTracerTransportModel(g, sc, None, TracerTransportSpectralConfig(),
                                         mass_flux=mf)
    q0 = dcmip13_init_latlon(g, sc).tracers.data.astype(jnp.float64)
    st_u = model.state_from_grid(jnp.ones_like(q0))
    assert float(jnp.max(jnp.abs(model.grid_values(st_u) - 1.0))) < 1e-13
    assert float(jnp.max(jnp.abs(model.grid_values(model.step(st_u, 1800.0)) - 1.0))) < 1e-13
    # tracer only in the middle layers: one SSP-RK3 step (3 donor-cell
    # stages) cannot carry it to the open top / surface layers, so the
    # global mass is closed (the open-boundary flux is tested on MPAS)
    q0 = q0.at[..., :4, :].set(0.0).at[..., 8:, :].set(0.0)
    s = model.state_from_grid(q0)
    m0 = _mass(model.grid_values(s), mf.dp, g.grid_area)
    s = model.step(s, 1800.0)
    np.testing.assert_allclose(_mass(model.grid_values(s), mf.dp, g.grid_area), m0, rtol=1e-13)
    with pytest.raises(ValueError):
        SpectralTracerTransportModel(
            g, sc, None, TracerTransportSpectralConfig(vertical_scheme="fv3_remap"), mass_flux=mf)


def _boundary_rate(q, w, area):
    """Tracer mass gained per second through the open top and surface:
    ``W_top q_top - W_surface q_surface`` (W positive down), area-summed."""
    q = np.asarray(q, np.float64)
    w = np.asarray(w, np.float64)
    a = np.asarray(area, np.float64)[..., None]
    return ((a * (w[..., :1] * q[..., 0, :] - w[..., -1:] * q[..., -1, :]))
            .reshape(-1, q.shape[-1]).sum(axis=0))


def test_latlon_tendency_mass_budget_is_the_boundary_flux(latlon, sc):
    """RULE 0: sum(area dp dq/dt) equals the open-boundary tracer flux to
    round-off (any RK combination of such tendencies inherits it)."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
        TracerTransportLatLonModel)
    g, mf, _, _ = latlon
    model = TracerTransportLatLonModel(g, sc, lambda *a: None, mass_flux=mf)
    st = dcmip13_init_latlon(g, sc)
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.random(st.tracers.data.shape))     # boundary layers non-empty
    tend = model.tendencies(st._replace(tracers=st.tracers.replace(data=q))).tracers.data
    got = _mass(tend, mf.dp, g.area)
    want = _boundary_rate(q, mf.w, g.area)
    scale = float(np.sum(np.asarray(g.area)[..., None] * np.abs(np.asarray(mf.w))))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13 * scale)
    assert np.all(np.abs(want) > 1e-6 * scale)


def test_spectral_tendency_mass_budget_is_the_boundary_flux(spectral, sc):
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig)
    from legoesm.grids.gaussian import sh_synthesis_3d
    g, mask, mf, _ = spectral
    model = SpectralTracerTransportModel(g, sc, None, TracerTransportSpectralConfig(),
                                         mass_flux=mf)
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.random((g.n_lat, g.n_lon, NLEV, 2)))
    st = model.state_from_grid(q)
    qg = model.grid_values(st)
    tend = model.tendencies(st).tracers.data
    n_sh = tend.shape[0]
    tg = sh_synthesis_3d(g, tend.reshape(n_sh, -1)).reshape(qg.shape)   # d(dp q)/dt
    got = _mass(tg, np.ones_like(np.asarray(mf.dp)), g.grid_area)
    want = _boundary_rate(qg, mf.w, g.grid_area)
    scale = float(np.sum(np.asarray(g.grid_area)[..., None] * np.abs(np.asarray(mf.w))))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13 * scale)
    assert np.all(np.abs(want) > 1e-6 * scale)


def test_banded_flow_fraction(sc):
    """h0 <= z <= z_top restriction: layer fractions in [0, 1], 1 for layers
    inside the band, sigma_dot zero at the top and the surface everywhere,
    and the uncut flow where the band covers the whole column (flat)."""
    from tests.test_cases.dcmip_transport import (
        dcmip13_flow_fraction, dcmip13_wind_geo_banded)
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(18, 36)
    frac, mh = dcmip13_flow_fraction(g.lon2d, g.lat2d, sc)
    frac, mh = np.asarray(frac), np.asarray(mh)
    assert frac.min() >= 0.0 and frac.max() <= 1.0 + 1e-15
    assert np.all(mh[..., -1] == 0) and np.all(mh[..., 0] == 0)
    u, v, sd = (np.asarray(x) for x in dcmip13_wind_geo_banded(0.0, g.lon2d, g.lat2d, sc))
    assert np.all(sd[..., 0] == 0) and np.all(sd[..., -1] == 0)
    zs = np.asarray(_mountain_height(g.lon2d, g.lat2d))
    flat = zs == 0
    # flat columns: z = -H ln(sigma) spans [0, 12 km]; only the layers below
    # h0 = 2 km are cut (the 12-level layers are 1 km: the lowest two)
    np.testing.assert_allclose(frac[flat][:, :-2], 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(frac[flat][:, -2:], 0.0, rtol=0, atol=1e-6)
    u0 = np.asarray(dcmip13_wind_geo(0.0, g.lon2d, g.lat2d, sc)[0])
    np.testing.assert_allclose(u, u0 * frac)


def test_banded_mpas_boundary_flux_is_exactly_zero():
    """ico5 L30 (the matrix row's mesh): no column keeps a top / surface
    air flux (12 cells within float32 rounding of z_top did before the
    boundary interfaces were cut explicitly, opening the whole top through
    the solvability mean removal)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.dcmip_transport import dcmip13_flow_fraction
    sc30 = create_dcmip_sigma(30)
    m = create_voronoi_mesh(5)
    _, mh = dcmip13_flow_fraction(m.lonCell, m.latCell, sc30)
    _, sd = dcmip_wind_mpas(13, 0.0, m, sc30)
    w = np.asarray(sd) * np.asarray(mh)
    assert np.count_nonzero(w[:, 0]) == 0 and np.count_nonzero(w[:, -1]) == 0
    assert np.count_nonzero(w[:, 1]) > 0


def test_banded_latlon_run_conserves_mass_with_boundary_tracer(sc):
    """With the banded flow no air crosses the top or the surface: a tracer
    filling the boundary layers keeps sum(area dp q) to round-off."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
        TracerTransportLatLonModel)
    from legoesm.grids.latlon import create_latlon_grid
    from tests.test_cases.dcmip_transport import dcmip13_wind_geo_banded
    g = create_latlon_grid(18, 36)
    mf = cmf.consistent_mass_flux_latlon(g, sc, dcmip13_wind_geo_banded,
                                         _ps(g.lon2d, g.lat2d))
    assert np.all(np.asarray(mf.w[..., 0]) == 0) and np.all(np.asarray(mf.w[..., -1]) == 0)
    assert _rel(cmf.continuity_residual_latlon(mf, g), mf) < 1e-17
    model = TracerTransportLatLonModel(g, sc, lambda *a: None, mass_flux=mf)
    st = dcmip13_init_latlon(g, sc)
    q = jnp.asarray(np.random.default_rng(2).random(st.tracers.data.shape))
    s = st._replace(tracers=st.tracers.replace(data=q))
    m0 = _mass(q, mf.dp, g.area)
    for _ in range(5):
        s = model.step(s, 600.0)
    np.testing.assert_allclose(_mass(s.tracers.data, mf.dp, g.area), m0, rtol=1e-13)


def test_builders_require_x64(sc):
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(18, 36)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="x64"):
            cmf.consistent_mass_flux_latlon(g, sc, dcmip13_wind_geo, _ps(g.lon2d, g.lat2d))
    finally:
        jax.config.update("jax_enable_x64", True)


def test_spectral_builder_rejects_nonpositive_projected_ps(sc):
    from legoesm.grids.gaussian import create_gaussian_grid, dealiasing_mask
    g = create_gaussian_grid(10)
    x = jnp.cos(g.lat2d) * jnp.cos(g.lon2d)
    ps_bad = 1e5 * ((1.0 + x) ** 2 + 0.1)        # > 0, band (n<=1) projection is not
    mask = jnp.where(g.ls <= 1, 1.0, 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        cmf.consistent_mass_flux_spectral(g, sc, dcmip13_wind_geo, ps_bad, mask)


def test_builders_reject_nonpositive_ps(sc):
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(18, 36)
    with pytest.raises(ValueError, match="strictly positive"):
        cmf.consistent_mass_flux_latlon(g, sc, dcmip13_wind_geo, jnp.zeros(g.lon2d.shape))


def test_spectral_model_rejects_fluxes_of_another_band(spectral, sc):
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig)
    from legoesm.grids.gaussian import dealiasing_mask
    g, _, _, _ = spectral

    def divergent(t, lon, lat, s):     # degree-2 divergence, no vertical flow
        ones = jnp.ones(s.n_levels)
        v = -10.0 * jnp.sin(lat) * jnp.cos(lat)
        return 0.0 * v[..., None] * ones, v[..., None] * ones, \
            jnp.zeros(jnp.shape(lat) + (s.n_levels + 1,))
    # fluxes closed in degrees 0-1 only; the model's band (n <= 6) sees the
    # degree-2 divergence
    mf_other = cmf.consistent_mass_flux_spectral(
        g, sc, divergent, jnp.full(g.lon2d.shape, 1.0e5), jnp.where(g.ls <= 1, 1.0, 0.0))
    with pytest.raises(ValueError, match="does not close"):
        SpectralTracerTransportModel(g, sc, None, TracerTransportSpectralConfig(),
                                     mass_flux=mf_other)
    # reverse: air mass with a degree-2 part, model band n <= 1 (no wind, so
    # the continuity closes trivially)
    def still(t, lon, lat, s):
        z = jnp.zeros(jnp.shape(lat) + (s.n_levels,))
        return z, z, jnp.zeros(jnp.shape(lat) + (s.n_levels + 1,))
    ps2 = 1.0e5 * (1.0 + 0.1 * jnp.sin(g.lat2d) ** 2)
    mf_wide = cmf.consistent_mass_flux_spectral(g, sc, still, ps2, dealiasing_mask(g, 0.667))
    with pytest.raises(ValueError, match="not in this model's spectral band"):
        SpectralTracerTransportModel(
            g, sc, None, TracerTransportSpectralConfig(dealiasing_fraction=0.15),
            mass_flux=mf_wide)


def test_closed_surface_stays_closed_with_an_open_top(sc):
    """Open top (nonzero-mean W), closed surface: the solvability target is
    zero, so the surface W stays exactly zero (lat-lon and spectral)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.gaussian import create_gaussian_grid, dealiasing_mask

    def top_open(t, lon, lat, s):
        z = jnp.zeros(jnp.shape(lat) + (s.n_levels,))
        w = jnp.zeros(jnp.shape(lat) + (s.n_levels + 1,))
        return z, z, w.at[..., :-1].set(1.0e-6 * (1.0 + jnp.sin(lat))[..., None])
    g = create_latlon_grid(18, 36)
    mf = cmf.consistent_mass_flux_latlon(g, sc, top_open, jnp.full(g.lon2d.shape, 1.0e5))
    assert np.all(np.asarray(mf.w[..., -1]) == 0) and np.abs(np.asarray(mf.w[..., 0])).max() > 0
    assert _rel(cmf.continuity_residual_latlon(mf, g), mf) < 1e-17
    gg = create_gaussian_grid(10)
    mask = dealiasing_mask(gg, 0.667)
    mfs = cmf.consistent_mass_flux_spectral(gg, sc, top_open, jnp.full(gg.lon2d.shape, 1.0e5),
                                            mask)
    assert np.abs(np.asarray(mfs.w[..., -1])).max() < 1e-20
    assert _rel(cmf.continuity_residual_spectral(mfs, gg, mask), mfs) < 1e-17


def test_van_leer_boundary_layers_donate_their_own_value(sc):
    """An empty top (surface) layer above (below) a full one exports nothing
    (the shared reconstruction's extrapolated ghost gave 0.5); interior faces
    keep the limited van-Leer value; short columns fall back to donor cell."""
    q = jnp.zeros((1, NLEV, 1)).at[:, 1:-1, :].set(1.0)
    w_down = jnp.full((1, NLEV + 1), 2.0)
    g = np.asarray(cmf.van_leer_vertical_flux(q, w_down, sc))[0, :, 0]
    assert g[1] == 0.0                          # top layer (empty) donates 0
    g_up = np.asarray(cmf.van_leer_vertical_flux(q, -w_down, sc))[0, :, 0]
    assert g_up[-2] == 0.0                      # surface layer (empty) donates 0
    np.testing.assert_allclose(g[2:-2], 2.0)    # interior of the plateau
    q3 = jnp.asarray(np.random.default_rng(3).random((2, 3, 1)))
    w3 = jnp.asarray(np.random.default_rng(4).standard_normal((2, 4)))
    np.testing.assert_array_equal(cmf.van_leer_vertical_flux(q3, w3, sc),
                                  cmf.donor_cell_vertical_flux(q3, w3))


# ----------------------------------------------------------------------------
# DCMIP 1-3 on the Kent et al. (2014, App. B) hybrid levels (matrix 1-3 rows)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hc():
    from tests.test_cases.dcmip_transport import create_dcmip13_hybrid
    return create_dcmip13_hybrid(NLEV)


def test_dcmip13_hybrid_levels(hc, sc):
    """B = (eta - eta_top)/(1 - eta_top), A = eta - B on the DCMIP eta levels:
    flat lid at z_top (A = eta_top, B = 0), surface p = p_s, pressure affine in
    eta in every column, and the coordinate-crossing flux B p_s V.grad(z_s)/H
    (zero at the lid)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import dp_from_hybrid, pressure_from_hybrid
    from tests.atmosphere.hydrostatic.test_cases.dcmip_transport import _ztop
    eta_top = float(np.exp(-_ztop / _H))
    np.testing.assert_allclose(np.asarray(hc.A_half[0]), eta_top, rtol=1e-15)
    assert float(hc.B_half[0]) == 0.0 and float(hc.B_half[-1]) == 1.0
    assert float(hc.A_half[-1]) == 0.0
    np.testing.assert_allclose(np.asarray(hc.sigma_half), np.asarray(sc.sigma_half), rtol=1e-6)
    # full levels: arithmetic interface means (Kent et al. 2014 eqs. 58-59)
    for full, half in ((hc.A_full, hc.A_half), (hc.B_full, hc.B_half)):
        np.testing.assert_array_equal(np.asarray(full),
                                      0.5 * (np.asarray(half)[:-1] + np.asarray(half)[1:]))
    g = create_latlon_grid(18, 36)
    ps = _ps(g.lon2d, g.lat2d)
    ph = np.asarray(pressure_from_hybrid(hc, ps, full=False))
    np.testing.assert_allclose(ph[..., 0], _p0 * eta_top, rtol=1e-15)
    np.testing.assert_allclose(ph[..., -1], np.asarray(ps), rtol=1e-15)
    r = np.asarray(dp_from_hybrid(hc, ps)) / np.diff(np.asarray(hc.sigma_half))
    np.testing.assert_allclose(r, r[..., :1] * np.ones_like(r), rtol=1e-13)
    sd_h = np.asarray(dcmip13_wind_geo(0.0, g.lon2d, g.lat2d, hc)[2])
    sd_s = np.asarray(dcmip13_wind_geo(0.0, g.lon2d, g.lat2d, sc)[2])
    tc = sd_s[..., -1:] / np.asarray(sc.sigma_half[-1])          # V.grad(z_s)/H
    np.testing.assert_allclose(sd_h, np.asarray(hc.B_half) * tc, rtol=1e-6, atol=1e-20)
    assert np.all(sd_h[..., 0] == 0)


def test_dcmip13_hybrid_flow_fraction(hc):
    """Band h0 <= z <= z_top in pressure: frac = layer dp overlap / dp; the lid
    is z_top so the flat columns keep every layer above h0 = 2 km."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import pressure_from_hybrid
    from tests.atmosphere.hydrostatic.test_cases.dcmip_transport import _ztop
    from tests.test_cases.dcmip_transport import dcmip13_flow_fraction
    g = create_latlon_grid(18, 36)
    frac, mh = (np.asarray(x) for x in dcmip13_flow_fraction(g.lon2d, g.lat2d, hc))
    ph = np.asarray(pressure_from_hybrid(hc, _ps(g.lon2d, g.lat2d), full=False))
    ov = np.diff(np.clip(ph, _p0 * np.exp(-_ztop / _H), _p0 * np.exp(-2000.0 / _H)), axis=-1)
    np.testing.assert_allclose(frac, ov / np.diff(ph, axis=-1), rtol=0, atol=1e-14)
    assert np.all(mh[..., 0] == 0) and np.all(mh[..., -1] == 0)
    flat = np.asarray(_mountain_height(g.lon2d, g.lat2d)) == 0
    np.testing.assert_allclose(frac[flat][:, :-2], 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(frac[flat][:, -2:], 0.0, rtol=0, atol=1e-12)
    # interior interfaces: the crossing flux is cut exactly where z < h0 and
    # kept unchanged where h0 <= z (and nonzero there over the terrain)
    from tests.test_cases.dcmip_transport import dcmip13_wind_geo_banded
    sd = np.asarray(dcmip13_wind_geo(0.0, g.lon2d, g.lat2d, hc)[2])
    sdb = np.asarray(dcmip13_wind_geo_banded(0.0, g.lon2d, g.lat2d, hc)[2])
    zh = -_H * np.log(ph / _p0)
    below = (zh < 2000.0)[..., 1:-1]
    assert np.count_nonzero(sd[..., 1:-1][below]) > 0
    assert np.all(sdb[..., 1:-1][below] == 0)
    above = ~below
    assert np.count_nonzero(sd[..., 1:-1][above]) > 0
    np.testing.assert_array_equal(sdb[..., 1:-1][above], sd[..., 1:-1][above])


def test_dcmip13_hybrid_latlon_closed_and_conservative(hc):
    """Banded 1-3 flow on the hybrid levels: dp = dA p0 + dB p_s, continuity
    closed to round-off, no air through lid or surface, q = 1 kept, and
    sum(area dp q) of a random tracer kept to round-off (RULE 0)."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
        TracerTransportLatLonModel)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import dp_from_hybrid
    from tests.test_cases.dcmip_transport import dcmip13_wind_geo_banded
    g = create_latlon_grid(18, 36)
    ps = _ps(g.lon2d, g.lat2d)
    mf = cmf.consistent_mass_flux_latlon(g, hc, dcmip13_wind_geo_banded, ps)
    np.testing.assert_allclose(np.asarray(mf.dp), np.asarray(dp_from_hybrid(hc, ps)), rtol=1e-15)
    assert np.all(np.asarray(mf.w[..., 0]) == 0) and np.all(np.asarray(mf.w[..., -1]) == 0)
    assert _rel(cmf.continuity_residual_latlon(mf, g), mf) < 1e-17
    model = TracerTransportLatLonModel(g, hc, lambda *a: None, mass_flux=mf)
    st = dcmip13_init_latlon(g, hc)
    one = model.step(st._replace(tracers=st.tracers.replace(
        data=jnp.ones_like(st.tracers.data, dtype=jnp.float64))), 600.0)
    assert float(jnp.max(jnp.abs(one.tracers.data - 1.0))) < 1e-13
    q = jnp.asarray(np.random.default_rng(3).random(st.tracers.data.shape))
    s = st._replace(tracers=st.tracers.replace(data=q))
    m0 = _mass(q, mf.dp, g.area)
    for _ in range(5):
        s = model.step(s, 600.0)
    np.testing.assert_allclose(_mass(s.tracers.data, mf.dp, g.area), m0, rtol=1e-13)


def test_dcmip13_hybrid_mpas_steady_air_and_mass(hc):
    """MPAS: the state's air mass is the hybrid dp (air_mass_state), the
    consistent fluxes keep it steady, q = 1 and sum(area dp q) to round-off."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        TracerTransportMPASConfig, TracerTransportMPASModel)
    from legoesm.grids.vertical import dp_from_hybrid
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.dcmip_transport import dcmip13_flow_fraction
    m = create_voronoi_mesh(2)
    ps = _ps(m.lonCell, m.latCell)
    st = dcmip_air_mass_state_mpas(13, dcmip13_init_mpas(m, hc), m, hc)
    np.testing.assert_allclose(np.asarray(st.air_mass.data),
                               np.asarray(dp_from_hybrid(hc, ps)), rtol=1e-15)
    ue, sd = dcmip_wind_mpas(13, 0.0, m, hc)
    fe = dcmip13_flow_fraction(m.lonEdge, m.latEdge, hc)[0]
    mc = dcmip13_flow_fraction(m.lonCell, m.latCell, hc)[1]
    mf = cmf.consistent_mass_flux_mpas(m, hc, ue * fe, sd * mc, ps)
    np.testing.assert_allclose(np.asarray(mf.dp), np.asarray(st.air_mass.data), rtol=1e-15)
    assert _rel(cmf.continuity_residual_mpas(mf, m), mf) < 1e-17
    assert np.all(np.asarray(mf.w[:, 0]) == 0) and np.all(np.asarray(mf.w[:, -1]) == 0)
    cfg = TracerTransportMPASConfig(hyperdiff_coeff=0.0, time_integrator="mpas_rk3",
                                    vertical_scheme="mpas_flux", horizontal_scheme="mpas_fct")
    model = TracerTransportMPASModel(
        m, hc, lambda t, mesh, s: dcmip_wind_mpas(13, t, mesh, s), cfg, mass_flux=mf)
    one = model.step(st._replace(tracers=st.tracers.replace(data=jnp.ones_like(st.tracers.data))),
                     600.0)
    assert float(jnp.max(jnp.abs(one.tracers.data - 1.0))) < 1e-13
    m0, s = _mass(st.tracers.data, st.air_mass.data, m.areaCell), st
    for _ in range(4):
        s = model.step(s, 600.0)
    np.testing.assert_allclose(_mass(s.tracers.data, s.air_mass.data, m.areaCell), m0, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(s.air_mass.data), np.asarray(st.air_mass.data),
                               rtol=1e-13)


def test_dcmip13_hybrid_spectral_closed(hc):
    """Spectral: dp = dA p0 + dB p_T (p_T the band projection), in-band
    closure to round-off and q = 1 kept."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral import (
        SpectralTracerTransportModel, TracerTransportSpectralConfig)
    from legoesm.grids.gaussian import create_gaussian_grid, dealiasing_mask
    from tests.test_cases.dcmip_transport import dcmip13_wind_geo_banded
    g = create_gaussian_grid(10)
    mask = dealiasing_mask(g, 0.667)
    mf = cmf.consistent_mass_flux_spectral(g, hc, dcmip13_wind_geo_banded,
                                           _ps(g.lon2d, g.lat2d), mask)
    from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
    p_t = np.asarray(sh_synthesis_3d(g, mask[:, None] * sh_analysis_3d(
        g, _ps(g.lon2d, g.lat2d)[..., None]))[..., 0])              # band-projected p_s
    np.testing.assert_allclose(np.asarray(mf.dp), np.asarray(hc.dA) * _p0
                               + np.asarray(hc.dB) * p_t[..., None], rtol=1e-12)
    assert _rel(cmf.continuity_residual_spectral(mf, g, mask), mf) < 1e-17
    model = SpectralTracerTransportModel(g, hc, None, TracerTransportSpectralConfig(),
                                         mass_flux=mf)
    q0 = dcmip13_init_latlon(g, hc).tracers.data.astype(jnp.float64)
    st_u = model.state_from_grid(jnp.ones_like(q0))
    assert float(jnp.max(jnp.abs(model.grid_values(model.step(st_u, 1800.0)) - 1.0))) < 1e-13


def test_sigma_uncorrected_fluxes_keep_their_exact_expressions(sc):
    """The hybrid generalisation leaves the sigma builders bit-identical:
    lat-lon F0 = p_s,face u dsigma and dp = p_s dsigma (same operation
    order), MPAS / spectral dp = p_s dsigma, exactly."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
        uface_coords, vface_coords)
    from legoesm.grids.gaussian import create_gaussian_grid, dealiasing_mask
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.voronoi import create_voronoi_mesh
    dsig = jnp.asarray(np.asarray(sc.dsigma, np.float64))
    g = create_latlon_grid(18, 36)
    ps = jnp.asarray(np.asarray(_ps(g.lon2d, g.lat2d), np.float64))   # the builder's cast
    mf = cmf.consistent_mass_flux_latlon(g, sc, dcmip13_wind_geo, ps, refine=0)
    lon_u, lat_u = uface_coords(g)
    u = jnp.asarray(dcmip13_wind_geo(0.0, lon_u, lat_u, sc)[0], jnp.float64)
    fu = 0.5 * (ps + jnp.roll(ps, 1, axis=1))[..., None] * u * dsig
    np.testing.assert_array_equal(np.asarray(mf.flux_h[0][:, :-1]), np.asarray(fu))
    lon_v, lat_v = vface_coords(g)
    v = jnp.asarray(dcmip13_wind_geo(0.0, lon_v, lat_v, sc)[1], jnp.float64)
    fv = 0.5 * (ps[1:] + ps[:-1])[..., None] * v[1:-1] * dsig
    np.testing.assert_array_equal(np.asarray(mf.flux_h[1][1:-1]), np.asarray(fv))
    np.testing.assert_array_equal(np.asarray(mf.dp), np.asarray(ps[..., None] * dsig))
    m = create_voronoi_mesh(2)
    ue, sd = dcmip_wind_mpas(13, 0.0, m, sc)
    pm = jnp.asarray(np.asarray(_ps(m.lonCell, m.latCell), np.float64))
    mm = cmf.consistent_mass_flux_mpas(m, sc, ue, sd, pm, refine=0)
    np.testing.assert_array_equal(np.asarray(mm.dp), np.asarray(pm[:, None] * dsig))
    from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
    gg = create_gaussian_grid(10)
    mask = jnp.asarray(dealiasing_mask(gg, 0.667), jnp.float64)
    pg = jnp.asarray(np.asarray(_ps(gg.lon2d, gg.lat2d), np.float64))
    ms = cmf.consistent_mass_flux_spectral(gg, sc, dcmip13_wind_geo, pg, mask, refine=0)
    p_t = sh_synthesis_3d(gg, mask[:, None] * sh_analysis_3d(gg, pg[..., None]))[..., 0]
    np.testing.assert_array_equal(np.asarray(ms.dp), np.asarray(p_t[..., None] * dsig))


def test_dcmip13_hybrid_uncorrected_face_fluxes_and_dp_guard(hc):
    """Hybrid lat-lon F0 = wind (dA p0 + dB p_s,face) on both face sets
    (refine=0, before the continuity correction can mask a wrong mass), and
    a surface pressure at or below the lid (dp <= 0) is refused."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
        uface_coords, vface_coords)
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(18, 36)
    ps = jnp.asarray(np.asarray(_ps(g.lon2d, g.lat2d), np.float64))
    mf = cmf.consistent_mass_flux_latlon(g, hc, dcmip13_wind_geo, ps, refine=0)
    dA, dB = np.asarray(hc.dA), np.asarray(hc.dB)
    lon_u, lat_u = uface_coords(g)
    u = np.asarray(dcmip13_wind_geo(0.0, lon_u, lat_u, hc)[0], np.float64)
    pu = np.asarray(0.5 * (ps + jnp.roll(ps, 1, axis=1)))[..., None]
    np.testing.assert_allclose(np.asarray(mf.flux_h[0][:, :-1]), u * (dA * _p0 + dB * pu),
                               rtol=1e-14, atol=1e-9)
    lon_v, lat_v = vface_coords(g)
    v = np.asarray(dcmip13_wind_geo(0.0, lon_v, lat_v, hc)[1], np.float64)[1:-1]
    pv = np.asarray(0.5 * (ps[1:] + ps[:-1]))[..., None]
    np.testing.assert_allclose(np.asarray(mf.flux_h[1][1:-1]), v * (dA * _p0 + dB * pv),
                               rtol=1e-14, atol=1e-9)
    p_top = _p0 * float(hc.A_half[0])
    for bad in (0.5 * p_top, p_top):                 # inverted / collapsed column
        with pytest.raises(ValueError, match="strictly positive"):
            cmf.consistent_mass_flux_latlon(g, hc, dcmip13_wind_geo,
                                            ps.at[3, 4].set(bad), refine=0)


def test_mpas_hybrid_air_mass_state_refuses_nonpositive_layers(hc):
    """air_mass_state on the hybrid levels: p_s below the lid gives dp < 0,
    refused at the initialisation entry point."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import air_mass_state
    from legoesm.grids.voronoi import create_voronoi_mesh
    m = create_voronoi_mesh(2)
    st = dcmip13_init_mpas(m, hc)
    st = st._replace(tracers=st.tracers.replace(data=st.tracers.data.astype(jnp.float64)))
    air_mass_state(st, _ps(m.lonCell, m.latCell), hc)             # valid: no raise
    for bad in (0.5, 1.0):                           # inverted / collapsed (p_s = p_top)
        with pytest.raises(ValueError, match="strictly positive"):
            air_mass_state(st, bad * _p0 * float(hc.A_half[0]), hc)
    # traceable: under jit a valid p_s gives the eager air mass (to XLA
    # fusion round-off)
    pv = _ps(m.lonCell, m.latCell)
    np.testing.assert_allclose(
        np.asarray(jax.jit(lambda p: air_mass_state(st, p, hc).air_mass.data)(pv)),
        np.asarray(air_mass_state(st, pv, hc).air_mass.data), rtol=1e-15)


def test_hybrid_guards_check_the_stored_dp_itself(hc, monkeypatch):
    """p_s above the threshold but a non-positive computed dp (forced) is
    refused at both entry points: the dp predicate is independent."""
    import legoesm.grids.vertical as V
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import air_mass_state
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.voronoi import create_voronoi_mesh
    real = V.dp_from_hybrid
    monkeypatch.setattr(V, "dp_from_hybrid", lambda c, p: real(c, p).at[..., 0].set(0.0))
    g = create_latlon_grid(18, 36)
    with pytest.raises(ValueError, match="strictly positive"):
        cmf.consistent_mass_flux_latlon(g, hc, dcmip13_wind_geo, _ps(g.lon2d, g.lat2d),
                                        refine=0)
    m = create_voronoi_mesh(2)
    st = dcmip13_init_mpas(m, hc)
    with pytest.raises(ValueError, match="strictly positive"):
        air_mass_state(st, _ps(m.lonCell, m.latCell), hc)

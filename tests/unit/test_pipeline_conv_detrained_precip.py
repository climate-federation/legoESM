"""The unified pipeline hands convective DETRAINED precipitation (IFS PSNDE)
to the rain and snow species, not to the surface (user 2026-10-28, S1'')."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import make_hybrid_levels

NLEV, DT = 8, 600.0


def _setup():
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    s2 = grid.grid_lat.shape
    s3 = (*s2, NLEV)
    T = jnp.asarray(np.broadcast_to(300.0 - 60.0 * np.linspace(0, 1, NLEV)[::-1], s3).copy())
    q_v = jnp.asarray(np.broadcast_to(18e-3 * np.exp(-3 * np.linspace(0, 1, NLEV)[::-1]), s3).copy())
    z2, z3 = jnp.zeros(s2), jnp.zeros(s3)
    f = dict(T=T, p_s=jnp.full(s2, 1.0e5), q_v=q_v, q_c=z3, q_r=z3, u=z3, v=z3,
             sst=jnp.full(s2, 302.0), sic=z2, lat=jnp.asarray(grid.grid_lat), z2=z2, z3=z3)
    config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        convection="bechtold", radiation="none", microphysics="none")
    return build_physics_pipeline(grid, sigma, config), f, s3


def _step(pipe, f, **kw):
    return pipe.physics_step_no_rad(
        f["T"], f["p_s"], f["q_v"], f["q_c"], f["q_r"], None,
        f["u"], f["v"], f["sst"], f["sic"], f["lat"], DT,
        f["z3"], f["z2"], f["z2"], f["z2"], f["z2"], f["z2"], **kw)


def _with_detrained(pipe, r, s):
    orig = pipe.convection_fn

    def conv(*a, **k):
        res = orig(*a, **k)
        out, rest = (res[0], res[1:]) if isinstance(res, tuple) else (res, ())
        out = out._replace(dq_r_detr_dt=r, dq_s_detr_dt=s)
        return (out, *rest) if rest else out
    pipe.convection_fn = conv


def test_detrained_precip_goes_to_rain_and_snow_not_the_surface():
    pipe, f, s3 = _setup()
    base = _step(pipe, f, q_s=f["z3"])
    ncol = int(np.prod(s3[:-1]))
    r = jnp.full((ncol, NLEV), 2.0e-7)
    s = jnp.full((ncol, NLEV), 3.0e-7)
    _with_detrained(pipe, r, s)
    got = _step(pipe, f, q_s=f["z3"])
    np.testing.assert_allclose(np.asarray(got.dq_r_dt - base.dq_r_dt), 2.0e-7, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(got.dq_s_dt - base.dq_s_dt), 3.0e-7, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(got.precip), np.asarray(base.precip))


def test_detrained_precip_without_a_snow_species_is_refused():
    pipe, f, s3 = _setup()
    ncol = int(np.prod(s3[:-1]))
    _with_detrained(pipe, jnp.zeros((ncol, NLEV)), jnp.zeros((ncol, NLEV)))
    with pytest.raises(ValueError, match="no snow species"):
        _step(pipe, f)


@pytest.mark.parametrize("micro,ok", [("morrison", True), ("thompson", True),
                                      ("seifert_beheng", False), ("kessler", False),
                                      ("sundqvist", False), ("p3", False), ("none", False)])
def test_ifs_convection_needs_rain_and_snow_species(micro, ok):
    """R12 (user 2026-10-28): the faithful IFS path is refused at config check
    unless the microphysics carries rain and snow for the detrained PSNDE."""
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        convection="bechtold", bechtold_use_ifs_ascent=True, microphysics=micro)
    msg = ""
    try:
        cfg.validate_strict()
    except ValueError as e:
        msg = str(e)
    assert ("carries rain and snow" in msg) is (not ok)


def _bridge(tracer_names, r, s):
    import types

    from legoesm.atmosphere.physics.convection import integration as ci
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate

    ncol, nlev = r.shape
    T = jnp.broadcast_to(jnp.linspace(298.0, 215.0, nlev)[None, :], (ncol, nlev))
    q = jnp.broadcast_to(jnp.linspace(0.017, 1e-5, nlev)[None, :], (ncol, nlev))
    z = jnp.zeros((ncol, nlev))
    tracers = {"q_v": Field(q), **{n: Field(z) for n in tracer_names}}
    state = HydrostaticState(
        u=Field(z), T=Field(T), p_s=Field(jnp.full((ncol,), 1.0e5)),
        phis=Field(jnp.zeros((ncol,))), v=Field(z), tracers=tracers)
    grid = types.SimpleNamespace(grid_shape_2d=(ncol,), land_frac=np.zeros(ncol))
    real = ci.bechtold_convection

    def spy(*a, **k):
        out, m_u, stoch = real(*a, **k)
        return out._replace(dq_r_conv_dt=None, dq_r_detr_dt=r, dq_s_detr_dt=s), m_u, stoch

    ci.bechtold_convection = spy
    try:
        fn = ci.make_convection_physics(
            ConvectionConfig(scheme="bechtold", rain_to_surface=True),
            model_type="mpas", dt=DT)
        tend, _ = fn(state, grid, create_sigma_coordinate(nlev))
    finally:
        ci.bechtold_convection = real
    return tend


def test_bridge_routes_detrained_precip_to_rain_and_snow():
    r = jnp.full((4, NLEV), 2.0e-7)
    s = jnp.full((4, NLEV), 3.0e-7)
    tend = _bridge(("q_c", "q_r", "q_s"), r, s)
    np.testing.assert_allclose(np.asarray(tend.tracer_tendencies["q_r"].data).reshape(4, NLEV), 2.0e-7)
    np.testing.assert_allclose(np.asarray(tend.tracer_tendencies["q_s"].data).reshape(4, NLEV), 3.0e-7)


def test_bridge_refuses_detrained_precip_without_snow():
    z = jnp.zeros((4, NLEV))
    with pytest.raises(ValueError, match="no q_s tracer"):
        _bridge(("q_c", "q_r"), z, z)

"""Cross-component proof that the AD infrastructure works through REAL model steps.

``checkpointed_loop`` (binomial checkpointing), ``curvature.hvp`` (forward-over-
reverse Hessian product) and the chaos guardrails are component- and grid-
agnostic by construction — they only ever see an opaque state pytree and a
``state -> state`` step.  This module nails that down end-to-end by driving each
real component step (atmosphere on a cubed-sphere, ocean on a lat-lon C-grid,
land on a soil column, sea ice on a slab) through them and asserting:

  * binomial checkpointing reproduces the plain gradient (to fp tolerance), and
  * the matrix-free HVP matches a finite difference of the gradient.

The remaining grids (spectral, MPAS/Voronoi, icosahedral seed) are covered by
the engine's own unit tests (tests/unit/test_checkpoint_schedule.py,
test_curvature.py): the schedule cannot distinguish grids, since the carry is an
opaque pytree — so one representative grid per component is a sufficient
integration proof rather than a re-test of every grid.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # Metal backend is broken in this env
os.environ.setdefault("JAX_ENABLE_X64", "1")   # spectral/lat-lon need fp64; safe everywhere

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.da import curvature  # noqa: E402
from legoesm.training.checkpoint_schedule import checkpointed_loop  # noqa: E402

_N_ROLL = 3  # short rollout — enough to exercise the binomial schedule


def _pack(x0, step_fn, set_leaf, reduce):
    def loss(x, schedule):
        final = checkpointed_loop(step_fn, set_leaf(x), _N_ROLL, schedule=schedule)
        return reduce(final)

    def step_loss(x):
        return reduce(step_fn(set_leaf(x)))

    return {"x0": x0, "loss": loss, "step_loss": step_loss}


def _make_atmosphere():
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        CDGridShallowWaterModel,
        CDGridShallowWaterState,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    n = 4
    base = create_cubed_sphere(n)
    model = CDGridShallowWaterModel(base, CDGridShallowWaterConfig())
    h0 = 1000.0 * jnp.ones((6, n, n)) + 10.0 * jax.random.normal(
        jax.random.PRNGKey(0), (6, n, n))
    template = CDGridShallowWaterState(
        h=h0,
        u_d=jnp.zeros((6, n + 1, n + 1)),
        v_d=jnp.zeros((6, n + 1, n + 1)),
        h_s=jnp.zeros((6, n, n)),
    )
    step_fn = lambda s: model.step(s, 60.0)
    return _pack(template.h, step_fn,
                 lambda x: template._replace(h=x),
                 lambda s: jnp.sum(s.h ** 2))


def _make_ocean():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z = create_ocean_z_star(3, H_max=500.0)
    config = LatLonCGridOceanConfig.from_flat(
        use_conservation_fixer=False, enable_runtime_checks=False,
        n_barotropic_substeps=2, differentiable_barotropic=True,
    )
    model = LatLonCGridOceanModel(grid, z, config)
    template = rest_state_latlon_cgrid_ocean(grid, z, land_lat_threshold=90.0)
    # NB: this model is intrinsically float32 (finite-volume); the HVP test
    # below picks a float32-appropriate finite-difference step + tolerance.
    model.prime_step_caches(template)
    step_fn = lambda s: model._step_impl(s, 300.0)
    return _pack(template.T.data, step_fn,
                 lambda x: template._replace(T=template.T.replace(data=x)),
                 lambda s: jnp.sum(s.T.data ** 2))


def _make_land():
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state,
        step_multilayer_land,
    )

    ncol = 8
    base = MultiLayerLandConfig()
    config = base._replace(soil_grid=base.soil_grid._replace(n_layers=5))
    scale = jnp.linspace(0.9, 1.1, ncol)
    fields = dict(
        sw_down=300.0, lw_down=350.0, precip_total=1e-4, precip_snow=0.0,
        T_lowest=290.0, q_lowest=8e-3, u_lowest=5.0, v_lowest=2.0,
        p_lowest=1e5, p_surface=1.013e5, rho_lowest=1.2, cos_zenith=0.7,
        co2_ppmv=400.0, has_radiation=1.0, has_precipitation=1.0,
    )
    forcing = AtmToSurface(**{k: jnp.asarray(v) * scale for k, v in fields.items()})
    template = init_multilayer_land_state(ncol, config, T_init=280.0)
    step_fn = lambda s: step_multilayer_land(s, forcing, config, U_min=1.0, dt=600.0)[0]
    return _pack(template.T_soil, step_fn,
                 lambda x: template._replace(T_soil=x),
                 lambda s: jnp.sum(s.T_soil ** 2))


def _make_sea_ice():
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.core.field import Field
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.sea_ice import step_sea_ice
    from legoesm.ice.state import SeaIceState

    shape = (6, 4, 4)
    ones = jnp.ones(shape)
    config = SeaIceConfig(dynamics="none")
    template = SeaIceState(
        h_ice=Field(1.0 * ones, name="h_ice"),
        T_ice=Field(265.0 * ones, name="T_ice"),
        concentration=Field(0.8 * ones, name="concentration"),
    )
    forcing = AtmToSurface(
        sw_down=100.0 * ones, lw_down=250.0 * ones, precip_total=0.0 * ones,
        precip_snow=0.0 * ones, T_lowest=260.0 * ones, q_lowest=1e-3 * ones,
        u_lowest=5.0 * ones, v_lowest=2.0 * ones, p_lowest=1e5 * ones,
        p_surface=1.013e5 * ones, rho_lowest=1.4 * ones, cos_zenith=0.5 * ones,
        co2_ppmv=400.0 * ones, has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
    )
    sst, ou, ov = 271.35 * ones, jnp.zeros(shape), jnp.zeros(shape)
    step_fn = lambda s: step_sea_ice(
        s, forcing, sst, ou, ov, config, U_min=1.0, dt=3600.0)[0]
    return _pack(template.T_ice.data, step_fn,
                 lambda x: template._replace(T_ice=template.T_ice.replace(data=x)),
                 lambda s: jnp.sum(s.T_ice.data ** 2))


COMPONENTS = {
    "atmosphere_cubed_sphere": _make_atmosphere,
    "ocean_latlon_cgrid": _make_ocean,
    "land_soil_column": _make_land,
    "sea_ice_slab": _make_sea_ice,
}


@pytest.mark.parametrize("name", list(COMPONENTS))
def test_binomial_checkpointing_matches_plain_gradient(name):
    spec = COMPONENTS[name]()
    x0, loss = spec["x0"], spec["loss"]
    g_none = jax.grad(lambda x: loss(x, "none"))(x0)
    g_bino = jax.grad(lambda x: loss(x, "binomial"))(x0)
    assert jnp.all(jnp.isfinite(g_none)), f"{name}: non-finite gradient"
    assert jnp.any(jnp.abs(g_none) > 0.0), f"{name}: gradient is identically zero"
    assert jnp.allclose(g_none, g_bino, rtol=1e-6, atol=1e-8), (
        f"{name}: binomial gradient differs from plain by "
        f"{float(jnp.max(jnp.abs(g_none - g_bino))):.2e}"
    )


@pytest.mark.parametrize("name", list(COMPONENTS))
def test_hvp_matches_finite_difference(name):
    spec = COMPONENTS[name]()
    x0, step_loss = spec["x0"], spec["step_loss"]
    v = jax.random.normal(jax.random.PRNGKey(7), x0.shape, dtype=x0.dtype)
    v = v / jnp.linalg.norm(v)
    Hv = curvature.hvp(step_loss, x0, v)
    assert jnp.all(jnp.isfinite(Hv)), f"{name}: non-finite HVP"
    # Finite-difference reference: the central-difference step and tolerance are
    # set by precision.  float64 supports a tight check; float32 FD of a gradient
    # is limited by ~cube-root(machine eps) cancellation, so use the balanced
    # step and a noise-tolerant bar that still catches a wrong HVP (the exact
    # correctness of curvature.hvp is proven separately in test_curvature.py).
    if x0.dtype == jnp.float64:
        eps, tol = 1e-4, 5e-3
    else:
        eps, tol = 3e-3, 8e-2
    # COMPILE FOOTPRINT, not style (#1736).  Each bare ``jax.grad(step_loss)(...)``
    # is its own top-level compilation, so the line this replaces asked XLA for
    # TWO more programs the size of the land column's gradient, on top of the
    # hvp's.  Measured: the hvp alone peaks at ~8 GB and succeeds; adding the
    # reference in the same process fails inside LLVM's JIT
    # ("Cannot allocate memory" / "Failed to materialize symbols"), which
    # surfaces as SIGABRT -- a core dump that took the whole tests/unit suite
    # with it, not a test failure.  Compiling the gradient ONCE and evaluating
    # it at both points is the same arithmetic and one program instead of two;
    # clearing the caches first drops the hvp's executable, which is no longer
    # needed once ``Hv`` is materialised.
    Hv = jnp.asarray(Hv)
    Hv.block_until_ready()
    jax.clear_caches()
    grad_fn = jax.jit(jax.grad(step_loss))
    fd = (grad_fn(x0 + eps * v) - grad_fn(x0 - eps * v)) / (2 * eps)
    denom = jnp.maximum(jnp.linalg.norm(fd), 1e-8)
    rel = jnp.linalg.norm(Hv - fd) / denom
    assert rel < tol, f"{name}: HVP vs finite-difference rel error {float(rel):.2e} (tol {tol})"

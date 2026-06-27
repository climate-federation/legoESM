"""Flux inversion through REAL component steps (forcing as a separate arg).

build_flux_cost_fn is component/grid-agnostic — it only needs a
``step_with_forcing(state, forcing_i, dt) -> state``.  Here we drive the real
ocean (lat-lon C-grid wind-stress tau_x — the canonical DJ4Earth case), land
(soil column) and sea-ice (cubed-sphere slab) steps through it and assert the
flux-inversion gradient flows through the real model and that the binomial
schedule reproduces the plain gradient.

Atmosphere forcing (SegmentForcing) is applied per SEGMENT, not per step, so its
flux/boundary inversion uses the same build_flux_cost_fn with ``run_segment.raw``
as the step at segment granularity (n_window = n_segments); it is not exercised
here because it needs the heavy compiled-segment mock, but the machinery is the
same one these tests cover.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field  # noqa: E402
from legoesm.da.background_error import DiagonalB  # noqa: E402
from legoesm.da.control_vector import (  # noqa: E402
    apply_forcing_slice,
    build_forcing_control_spec,
    control_to_forcing_series,
    forcing_series_to_control,
)
from legoesm.da.cost_function import build_flux_cost_and_grad_fn  # noqa: E402
from legoesm.da.observation import Observation  # noqa: E402

_N_STEPS = 2


class _RavelHeadObs:
    """Observe the first k raveled elements of a state field (Field or raw)."""
    def __init__(self, field_name, k=4):
        self.field_name = field_name
        self.k = k

    def __call__(self, state):
        val = getattr(state, self.field_name)
        arr = val.data if isinstance(val, Field) else val
        return jnp.ravel(arr)[:self.k]


def _make_ocean():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z = create_ocean_z_star(3, H_max=500.0)
    config = LatLonCGridOceanConfig.from_flat(
        use_conservation_fixer=False, enable_runtime_checks=False,
        n_barotropic_substeps=2, differentiable_barotropic=True,
    )
    model = LatLonCGridOceanModel(grid, z, config)
    init = rest_state_latlon_cgrid_ocean(grid, z, land_lat_threshold=90.0)
    model.prime_step_caches(init)
    tmpl = OceanSurfaceForcing(
        tau_x=jnp.zeros((8, 16)), tau_y=jnp.zeros((8, 16)), q_net=jnp.zeros((8, 16)))

    def step(state, forcing_i, dt):
        return model._step_impl(state, dt, surface_forcing=forcing_i)

    return dict(init=init, tmpl=tmpl, step=step, dt=300.0,
                invert="tau_x", perturb=0.05, obs=_RavelHeadObs("u", 4))


def _make_land():
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state,
        step_multilayer_land,
    )

    ncol = 4
    base = MultiLayerLandConfig()
    config = base._replace(soil_grid=base.soil_grid._replace(n_layers=5))
    init = init_multilayer_land_state(ncol, config, T_init=280.0)
    tmpl = AtmToSurface(
        sw_down=jnp.full(ncol, 200.0), lw_down=jnp.full(ncol, 300.0),
        precip_total=jnp.full(ncol, 1e-5), precip_snow=jnp.zeros(ncol),
        T_lowest=jnp.full(ncol, 280.0), q_lowest=jnp.full(ncol, 5e-3),
        u_lowest=jnp.full(ncol, 5.0), v_lowest=jnp.full(ncol, 2.0),
        p_lowest=jnp.full(ncol, 1e5), p_surface=jnp.full(ncol, 1.013e5),
        rho_lowest=jnp.full(ncol, 1.2), cos_zenith=jnp.full(ncol, 0.7),
        co2_ppmv=jnp.full(ncol, 400.0), has_radiation=jnp.ones(ncol),
        has_precipitation=jnp.ones(ncol),
    )

    def step(state, forcing_i, dt):
        return step_multilayer_land(state, forcing_i, config, U_min=1.0, dt=dt)[0]

    return dict(init=init, tmpl=tmpl, step=step, dt=600.0,
                invert="sw_down", perturb=50.0, obs=_RavelHeadObs("T_soil", 4))


def _make_sea_ice():
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ice.config import SeaIceConfig
    from legoesm.ice.sea_ice import step_sea_ice
    from legoesm.ice.state import SeaIceState

    shape = (6, 4, 4)
    ones = jnp.ones(shape)
    config = SeaIceConfig(dynamics="none")
    init = SeaIceState(
        h_ice=Field(1.0 * ones, name="h_ice"),
        T_ice=Field(265.0 * ones, name="T_ice"),
        concentration=Field(0.8 * ones, name="concentration"),
    )
    tmpl = AtmToSurface(
        sw_down=100.0 * ones, lw_down=250.0 * ones, precip_total=0.0 * ones,
        precip_snow=0.0 * ones, T_lowest=260.0 * ones, q_lowest=1e-3 * ones,
        u_lowest=5.0 * ones, v_lowest=2.0 * ones, p_lowest=1e5 * ones,
        p_surface=1.013e5 * ones, rho_lowest=1.4 * ones, cos_zenith=0.5 * ones,
        co2_ppmv=400.0 * ones, has_radiation=1.0 * ones, has_precipitation=1.0 * ones,
    )
    sst, ou, ov = 271.35 * ones, jnp.zeros(shape), jnp.zeros(shape)

    def step(state, forcing_i, dt):
        return step_sea_ice(state, forcing_i, sst, ou, ov, config, U_min=1.0, dt=dt)[0]

    return dict(init=init, tmpl=tmpl, step=step, dt=3600.0,
                invert="sw_down", perturb=40.0, obs=_RavelHeadObs("T_ice", 4))


COMPONENTS = {
    "ocean_latlon_tau_x": _make_ocean,
    "land_soil_sw_down": _make_land,
    "sea_ice_slab_sw_down": _make_sea_ice,
}


def _setup(builder):
    b = builder()
    spec = build_forcing_control_spec(b["tmpl"], n_window=_N_STEPS, fields=(b["invert"],))
    base_leaf = jnp.asarray(getattr(b["tmpl"], b["invert"]))
    bg_series = {b["invert"]: jnp.broadcast_to(base_leaf, (_N_STEPS, *base_leaf.shape))}
    bg = forcing_series_to_control(bg_series, spec)
    true_series = {b["invert"]: bg_series[b["invert"]] + b["perturb"]}
    true_x = forcing_series_to_control(true_series, spec)
    # synthetic obs at the final step from the TRUE forcing
    series = control_to_forcing_series(true_x, spec)
    state = b["init"]
    for i in range(_N_STEPS):
        state = b["step"](state, apply_forcing_slice(b["tmpl"], series, i), b["dt"])
    obs_val = b["obs"](state)
    observations = (Observation(
        values=obs_val, errors=jnp.full(obs_val.shape, 0.1),
        time_index=_N_STEPS - 1, operator=b["obs"]),)
    B = DiagonalB(sigma=jnp.ones(spec.total_size) * 1.0e3)  # weak prior
    return b, spec, bg, observations, B


@pytest.mark.parametrize("name", list(COMPONENTS))
def test_flux_inversion_gradient_flows_through_real_step(name):
    b, spec, bg, observations, B = _setup(COMPONENTS[name])

    def cost_and_grad(sch):
        return build_flux_cost_and_grad_fn(
            b["step"], b["init"], bg, observations, B, spec, b["tmpl"],
            b["dt"], _N_STEPS, checkpoint_schedule=sch)

    # background forcing differs from truth -> obs misfit -> nonzero adjoint
    J_n, g_n = cost_and_grad("none")(bg)
    assert jnp.isfinite(J_n)
    assert jnp.all(jnp.isfinite(g_n)), f"{name}: non-finite flux gradient"
    assert jnp.any(jnp.abs(g_n) > 0.0), f"{name}: flux gradient identically zero"

    # binomial checkpointing reproduces the plain gradient through the real model
    J_b, g_b = cost_and_grad("binomial")(bg)
    assert jnp.allclose(J_b, J_n, rtol=1e-5, atol=1e-7)
    assert jnp.allclose(g_b, g_n, rtol=1e-5, atol=1e-7), (
        f"{name}: binomial flux gradient differs by "
        f"{float(jnp.max(jnp.abs(g_b - g_n))):.2e}")

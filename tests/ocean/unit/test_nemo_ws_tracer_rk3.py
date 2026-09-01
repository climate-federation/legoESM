"""Stage-program tests for NEMO's key_RK3 active tracers."""

import jax
import jax.numpy as jnp
import numpy as np

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_lock_exchange_zco_card
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


def test_nemo_ws_transport_default_is_conditional_and_legacy_is_explicit():
    """Resolve WS to Kmm without changing any non-WS result."""
    set_policy(PrecisionPolicy.fp64())
    grid = create_latlon_grid(4, 8, dtype=jnp.float64)
    z_coord = create_ocean_z_star(n_levels=2, H_max=20.0)

    ws = LatLonCGridOceanConfig(
        tracer_time_integrator="rk3_ws",
        momentum_time_integrator="rk3_ws",
    )
    ws_model = model_module.LatLonCGridOceanModel(grid, z_coord, ws)
    assert ws.tracer_rk3_transport_time_levels is None
    assert ws_model.config.tracer_rk3_transport_time_levels == "nemo_kmm"

    ws_legacy = ws._replace(
        tracer_rk3_transport_time_levels="frozen_final")
    ws_legacy_model = model_module.LatLonCGridOceanModel(
        grid, z_coord, ws_legacy)
    assert ws_legacy_model.config.tracer_rk3_transport_time_levels == "frozen_final"

    non_ws = LatLonCGridOceanConfig()
    explicit_old = non_ws._replace(
        tracer_rk3_transport_time_levels="frozen_final")
    non_ws_model = model_module.LatLonCGridOceanModel(
        grid, z_coord, non_ws)
    explicit_old_model = model_module.LatLonCGridOceanModel(
        grid, z_coord, explicit_old)
    assert non_ws_model.config == explicit_old_model.config

    initial = rest_state_latlon_cgrid_ocean(
        grid, z_coord, H_max=20.0, T_water_init_C=10.0, T_deep=10.0)
    default_result = non_ws_model.step(initial, dt=0.01)
    explicit_result = explicit_old_model.step(initial, dt=0.01)
    default_leaves = jax.tree_util.tree_leaves(default_result)
    explicit_leaves = jax.tree_util.tree_leaves(explicit_result)
    assert len(default_leaves) == len(explicit_leaves)
    for default, explicit in zip(default_leaves, explicit_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))


def test_nemo_two_step_fct_is_a_live_real_flux_arm():
    """The source-ordered FCT predictor must move a nontrivial WS result."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    cfg_two = card.recipe.model_config
    cfg_one = cfg_two._replace(tracer_fct_low_order_predictor="one_step")
    state_two = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_two).step(
            card.recipe.initial_state, dt=card.dt_s)
    state_one = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_one).step(
            card.recipe.initial_state, dt=card.dt_s)
    movement = np.max(np.abs(
        np.asarray(state_two.T.data) - np.asarray(state_one.T.data)))
    assert movement > 1.0e-12


def test_nemo_ws_tracer_stage_polynomial(monkeypatch):
    rate = 0.2

    def linear_flux_pair(a, b, *args, **kwargs):
        zeros_a = jnp.zeros_like(a)
        zeros_b = jnp.zeros_like(b)
        return (rate * a, zeros_a), (rate * b, zeros_b)

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", linear_flux_pair)
    a0 = jnp.array([[[2.0]]], dtype=jnp.float64)
    b0 = jnp.array([[[3.0]]], dtype=jnp.float64)
    ones = jnp.ones_like(a0)
    a, b = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", ones, ones, jnp.ones((1, 1, 2)),
        ones, ones, ones, ones, object(), 0.5, ones,
    )
    z = -0.5 * rate
    amplification = 1.0 + z + z * z / 2.0 + z * z * z / 6.0
    np.testing.assert_allclose(np.asarray(a), 2.0 * amplification, rtol=0, atol=2e-16)
    np.testing.assert_allclose(np.asarray(b), 3.0 * amplification, rtol=0, atol=2e-16)


def test_nemo_ws_tracer_zero_flux_is_exact_identity(monkeypatch):
    def zero_flux_pair(a, b, *args, **kwargs):
        return (jnp.zeros_like(a), jnp.zeros_like(a)), (
            jnp.zeros_like(b), jnp.zeros_like(b))

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", zero_flux_pair)
    tracer = jnp.array([[[1.25, -2.0]]], dtype=jnp.float64)
    h = jnp.ones_like(tracer)
    got_a, got_b = model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", h, h, jnp.ones((1, 1, 3)),
        h, h, h, h, object(), 2.0, h,
    )
    np.testing.assert_array_equal(np.asarray(got_a), np.asarray(tracer))
    np.testing.assert_array_equal(np.asarray(got_b), np.asarray(tracer))


def test_nemo_ws_real_fct_flux_changes_on_wrong_transport_time_level():
    """Exercise real FCT geometry; a frozen final velocity must fail this pin."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    cfg_kmm = card.recipe.model_config
    cfg_wrong = cfg_kmm._replace(
        tracer_rk3_transport_time_levels="frozen_final")
    model_kmm = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_kmm)
    model_wrong = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_wrong)
    state_kmm = model_kmm.step(card.recipe.initial_state, dt=card.dt_s)
    state_wrong = model_wrong.step(card.recipe.initial_state, dt=card.dt_s)
    delta = np.max(np.abs(
        np.asarray(state_kmm.T.data) - np.asarray(state_wrong.T.data)))
    # Wrong Kaa reuse changes the real limiter/flux path by ~2.70e-5 K.
    assert delta > 2.0e-5
    np.testing.assert_array_equal(
        np.asarray(state_kmm.S.data), np.asarray(state_wrong.S.data))

"""Stage-program tests for NEMO's key_RK3 active tracers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _nemo_flux_form_external_velocity_update,
)
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_lock_exchange_zco_card
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


def test_nemo_ws_microselectors_are_not_public_config():
    """NEMO exposes no internal RK3 switches; neither may legoESM."""
    set_policy(PrecisionPolicy.fp64())
    grid = create_latlon_grid(4, 8, dtype=jnp.float64)
    z_coord = create_ocean_z_star(n_levels=2, H_max=20.0)

    fields = LatLonCGridOceanConfig._fields
    assert "tracer_rk3_transport_time_levels" not in fields
    assert "tracer_fct_low_order_predictor" not in fields
    assert "rk3_ws_stage_barotropic_correction" not in fields
    assert "rk3_ws_momentum_transport_reconcile" not in fields
    assert "primary_transport_average" not in fields


def test_nemo_ws_private_stage_velocity_exposure_is_diagnostic_only():
    """The fidelity seam returns Kaa stage velocity without changing tracers."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    normal = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config).step(
            card.recipe.initial_state, dt=card.dt_s)
    stage1 = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            expose_momentum_stage=1)).step(
                card.recipe.initial_state, dt=card.dt_s)
    assert np.max(np.abs(
        np.asarray(stage1.u.data) - np.asarray(normal.u.data))) > 1.0e-12
    # T-unchanged guards tracer wiring only: the hook writes u/v slots alone;
    # its structural post-step placement is the stage-noninterference guarantee.
    np.testing.assert_array_equal(
        np.asarray(stage1.T.data), np.asarray(normal.T.data))
    with pytest.raises(ValueError, match="expose_momentum_stage"):
        model_module.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                expose_momentum_stage=4))


def test_nemo_overflow_primary_transport_average_is_source_bound_and_live():
    """Flux-form RK3 uses NEMO's transport primary; only the test hook ablates."""
    set_policy(PrecisionPolicy.fp64())
    card = build_overflow_zps_card()
    faithful = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config).step(
            card.recipe.initial_state, dt=card.dt_s)
    ablated = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            primary_transport_average=False)).step(
                card.recipe.initial_state, dt=card.dt_s)
    assert np.max(np.abs(
        np.asarray(faithful.u.data) - np.asarray(ablated.u.data))) > 1.0e-12
    assert np.max(np.abs(
        np.asarray(faithful.T.data) - np.asarray(ablated.T.data))) > 0.0


def test_nemo_flux_form_external_update_is_literal_and_depth_sensitive():
    """dynspg_ts.F90:731-761 uses five distinct face-depth operands."""
    velocity = jnp.asarray([[0.25, -0.5]], dtype=jnp.float64)
    h_entry = jnp.asarray([[10.0, 11.0]], dtype=jnp.float64)
    h_pgf = jnp.asarray([[9.0, 12.0]], dtype=jnp.float64)
    h_mid = jnp.asarray([[8.0, 13.0]], dtype=jnp.float64)
    h_kmm = jnp.asarray([[7.0, 14.0]], dtype=jnp.float64)
    h_exit = jnp.asarray([[6.0, 15.0]], dtype=jnp.float64)
    pgf = jnp.asarray([[0.1, -0.2]], dtype=jnp.float64)
    transport = jnp.asarray([[0.3, 0.4]], dtype=jnp.float64)
    slow = jnp.asarray([[-0.5, 0.6]], dtype=jnp.float64)
    wet = jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)
    dt = jnp.asarray(2.0, dtype=jnp.float64)
    actual = _nemo_flux_form_external_velocity_update(
        velocity, h_entry, h_pgf, h_mid, h_kmm, h_exit,
        pgf, transport, slow, dt, wet, jnp.asarray(1.0e-10))
    expected = (
        h_entry * velocity
        + dt * (h_pgf * pgf + h_mid * transport + h_kmm * slow)
    ) / h_exit * wet
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    # Synthetic violation: replacing the midpoint depth by the entry depth is
    # not an algebraic no-op and must move a live wet face.
    violated = _nemo_flux_form_external_velocity_update(
        velocity, h_entry, h_pgf, h_entry, h_kmm, h_exit,
        pgf, transport, slow, dt, wet, jnp.asarray(1.0e-10))
    assert not np.array_equal(np.asarray(actual), np.asarray(violated))


def test_nemo_two_step_fct_is_a_live_real_flux_arm():
    """The source-ordered FCT predictor must move a nontrivial WS result."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    cfg_two = card.recipe.model_config
    state_two = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_two).step(
            card.recipe.initial_state, dt=card.dt_s)
    state_one = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_two,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            two_step_fct_predictor=False)).step(
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
    model_kmm = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_kmm)
    model_wrong = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_kmm,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            stage_barotropic_correction=False,
            momentum_transport_reconcile=False,
            kmm_tracer_transports=False))
    state_kmm = model_kmm.step(card.recipe.initial_state, dt=card.dt_s)
    state_wrong = model_wrong.step(card.recipe.initial_state, dt=card.dt_s)
    delta = np.max(np.abs(
        np.asarray(state_kmm.T.data) - np.asarray(state_wrong.T.data)))
    # Wrong Kaa reuse changes the real limiter/flux path by ~2.70e-5 K.
    assert delta > 2.0e-5
    np.testing.assert_array_equal(
        np.asarray(state_kmm.S.data), np.asarray(state_wrong.S.data))


def test_overflow_bbl_is_live_inside_real_rk3_stage3():
    """A planted dense shelf must activate BBL in the full RK3 solver."""
    set_policy(PrecisionPolicy.fp64())
    card = build_overflow_zps_card()
    from legoesm.ocean.physics.bbl_adv import bbl_static_geometry
    geom = bbl_static_geometry(
        card.recipe.z_coord.h_partial,
        card.recipe.initial_state.land_mask.data)
    active_faces = np.argwhere(np.asarray(geom.u_active) > 0.5)
    assert active_faces.size
    j, i = active_faces[len(active_faces) // 2]
    slope = int(np.asarray(geom.mgrhu)[j, i])
    shelf_i, deep_i = ((i, i + 1) if slope > 0 else (i + 1, i))
    ks = int(np.asarray(geom.ku_s)[j, i])
    kd = int(np.asarray(geom.ku_d)[j, i])
    initial = card.recipe.initial_state
    T = initial.T.data.at[j, shelf_i, ks].set(0.0)
    T = T.at[j, deep_i, kd].set(20.0)
    S = initial.S.data.at[j, shelf_i, ks].set(36.0)
    S = S.at[j, deep_i, kd].set(35.0)
    initial = initial._replace(
        T=initial.T.replace(data=T), S=initial.S.replace(data=S))
    on = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config).step(
            initial, dt=card.dt_s)
    off = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            disable_bbl=True)).step(initial, dt=card.dt_s)
    movement = np.max(np.abs(
        np.asarray(on.T.data) - np.asarray(off.T.data)))
    assert movement > 1.0e-12
    # BBL is a closed three-leg exchange; its isolated stage contribution
    # must not change the thickness-weighted domain tracer content.
    h = np.asarray(card.recipe.z_coord.h_partial)
    area = np.asarray(card.recipe.grid.area_T)[..., None]
    content_delta = np.sum(
        area * h * (np.asarray(on.T.data) - np.asarray(off.T.data)))
    content_scale = np.sum(
        np.abs(area * h * np.asarray(initial.T.data)))
    assert abs(content_delta) <= 1024.0 * np.finfo(np.float64).eps * content_scale


def test_nemo_ws_fct_limits_stage3_only_and_centres_stages_1_2(monkeypatch):
    """traadv.F90:281-282,361-364: FCT at kstg=3 only; cen2 at stages 1-2."""
    seen = []

    def recording_flux_pair(a, b, scheme, *args, **kwargs):
        seen.append((scheme, "tr_a_before" in kwargs))
        return (jnp.zeros_like(a), jnp.zeros_like(a)), (
            jnp.zeros_like(b), jnp.zeros_like(b))

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", recording_flux_pair)
    tracer = jnp.array([[[1.0, 2.0]]], dtype=jnp.float64)
    h = jnp.ones_like(tracer)
    model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "fct2", h, h, jnp.ones((1, 1, 3)),
        h, h, h, h, object(), 2.0, h,
    )
    assert seen == [("centered", False), ("centered", False), ("fct2", True)]
    seen.clear()
    model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", h, h, jnp.ones((1, 1, 3)),
        h, h, h, h, object(), 2.0, h,
    )
    # Non-FCT NEMO schemes run the same operator at every stage.
    assert [scheme for scheme, _ in seen] == ["centered"] * 3

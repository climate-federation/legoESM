"""Every learned atmosphere model can change the winds.

The column network (spectral NeuralGCM adapter and the gridded driver-path
step) carries a zero-initialised u/v head scaled to 1e-4 m/s^2 per raw unit
(tanh cap 5e-4 m/s^2), added to any drag. Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics import neural_physics as nph  # noqa: E402
from legoesm.core.grid_adapters import make_adapter  # noqa: E402
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.training.neural_gcm_spectral import (  # noqa: E402
    carry_to_spectral_state,
)

from tests.unit.test_learned_arms_driver_path_planes import (  # noqa: E402
    _positional,
)
from tests.unit.test_learned_column import _mini_spectral_state  # noqa: E402

NLEV = 3
# Distinct, level-dependent raw outputs per head so a u/v swap or a level
# reversal anywhere in the column lane fails the reference comparisons.
RAW_U = np.array([0.7, 0.4, 0.1])
RAW_V = np.array([-0.3, -0.6, 0.9])


def _net():
    return nph.build_column_physics(NLEV, hidden_dim=8, n_layers=2,
                                    key=jax.random.PRNGKey(0))


def _with_wind_bias(net, raw=None):
    raw = np.concatenate([RAW_U, RAW_V]) if raw is None else raw
    b = net.layers[-1].bias.at[4 * NLEV:6 * NLEV].set(raw)
    return eqx.tree_at(lambda m: m.layers[-1].bias, net, b)


def _expected_wind(raw):
    cap = nph._DEFAULT_TENDENCY_CAP
    return (nph._DEFAULT_RESIDUAL_SCALE * cap * np.tanh(raw / cap)
            * nph._WIND_HEAD_TENDENCY_FACTOR)


def test_scale_is_1e4_per_unit_and_cap_5e4():
    assert nph._DEFAULT_RESIDUAL_SCALE * nph._WIND_HEAD_TENDENCY_FACTOR == \
        pytest.approx(1.0e-4)
    assert (nph._DEFAULT_RESIDUAL_SCALE * nph._DEFAULT_TENDENCY_CAP
            * nph._WIND_HEAD_TENDENCY_FACTOR) == pytest.approx(5.0e-4)


# ---------------------------------------------------------- spectral adapter

@pytest.fixture(scope="module")
def spectral():
    grid = create_gaussian_grid(n_max=10)
    carry, sigma = _mini_spectral_state(grid, NLEV)
    return grid, carry_to_spectral_state(carry, grid), sigma


def test_spectral_zero_init_gives_exactly_zero_wind(spectral):
    grid, state, sigma = spectral
    out = nph.make_column_physics_fn(_net(), grid)(state, grid, sigma)
    assert float(jnp.max(jnp.abs(out.vor_hat.data))) == 0.0
    assert float(jnp.max(jnp.abs(out.div_hat.data))) == 0.0


def test_spectral_wind_head_reaches_vor_div_with_the_right_scale(spectral):
    """Per-level du != dv fields must come back as exactly that wind."""
    from legoesm.grids.gaussian import vordiv_from_uv_exact_3d
    grid, state, sigma = spectral
    out = nph.make_column_physics_fn(_with_wind_bias(_net()), grid)(
        state, grid, sigma)
    shp = (grid.n_lat, grid.n_lon, NLEV)
    ref_vor, ref_div = vordiv_from_uv_exact_3d(
        grid, jnp.broadcast_to(_expected_wind(RAW_U), shp),
        jnp.broadcast_to(_expected_wind(RAW_V), shp))
    assert float(jnp.max(jnp.abs(ref_vor))) > 0.0
    np.testing.assert_allclose(np.asarray(out.vor_hat.data),
                               np.asarray(ref_vor), rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(np.asarray(out.div_hat.data),
                               np.asarray(ref_div), rtol=1e-10, atol=1e-25)


def test_spectral_wind_head_is_added_to_the_drag(spectral):
    grid, state, sigma = spectral
    marker = 3.5e-6

    def _drag(st, g, sc):
        return st._replace(
            vor_hat=st.vor_hat.replace(data=jnp.full_like(st.vor_hat.data,
                                                          marker)),
            div_hat=st.div_hat.replace(data=jnp.full_like(st.div_hat.data,
                                                          -marker)))

    net = _with_wind_bias(_net())
    head = nph.make_column_physics_fn(net, grid)(state, grid, sigma)
    both = nph.make_column_physics_fn(net, grid, momentum_physics_fn=_drag)(
        state, grid, sigma)
    np.testing.assert_allclose(np.asarray(both.vor_hat.data),
                               np.asarray(head.vor_hat.data) + marker,
                               rtol=1e-12, atol=1e-25)
    np.testing.assert_allclose(np.asarray(both.div_hat.data),
                               np.asarray(head.div_hat.data) - marker,
                               rtol=1e-12, atol=1e-25)
    assert float(jnp.max(jnp.abs(both.lnps_hat.data))) == 0.0


def test_spectral_gradient_reaches_the_wind_rows_at_zero_init(spectral):
    """No inert leaf: at the zero init the wind rows already get gradient."""
    grid, state, sigma = spectral

    rng = np.random.default_rng(0)
    shp = state.vor_hat.data.shape
    r_vor = jnp.asarray(rng.normal(size=shp) + 1j * rng.normal(size=shp))
    r_div = jnp.asarray(rng.normal(size=shp) + 1j * rng.normal(size=shp))

    @eqx.filter_grad
    def g(net):
        out = nph.make_column_physics_fn(net, grid)(state, grid, sigma)
        return 1e6 * jnp.sum(jnp.real(out.vor_hat.data * jnp.conj(r_vor))
                             + jnp.real(out.div_hat.data * jnp.conj(r_div)))

    grads = g(_net())
    gb = np.asarray(grads.layers[-1].bias)
    assert np.all(np.abs(gb[4 * NLEV:6 * NLEV]) > 0.0), gb[4 * NLEV:6 * NLEV]


# ------------------------------------------------------ gridded (lat-lon)

@pytest.fixture(scope="module")
def latlon_adapter():
    return make_adapter(create_latlon_grid(8, 16))


def test_gridded_zero_init_gives_exactly_zero_wind(latlon_adapter):
    step = nph.make_neural_step_unified(_net(), latlon_adapter)
    out, _ = step(True, *_positional_nlev(latlon_adapter))
    assert bool(jnp.all(out.du_dt == 0.0)) and bool(jnp.all(out.dv_dt == 0.0))


def test_gridded_wind_head_reaches_du_dv(latlon_adapter):
    step = nph.make_neural_step_unified(_with_wind_bias(_net()),
                                        latlon_adapter)
    out, _ = step(True, *_positional_nlev(latlon_adapter))
    assert out.du_dt.shape == (*latlon_adapter.shape_2d, NLEV)
    shp = out.du_dt.shape
    np.testing.assert_allclose(np.asarray(out.du_dt),
                               np.broadcast_to(_expected_wind(RAW_U), shp),
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.dv_dt),
                               np.broadcast_to(_expected_wind(RAW_V), shp),
                               rtol=1e-12)


def test_gridded_gradient_reaches_the_wind_rows(latlon_adapter):
    @eqx.filter_grad
    def g(net):
        step = nph.make_neural_step_unified(net, latlon_adapter)
        out, _ = step(True, *_positional_nlev(latlon_adapter))
        return jnp.sum(out.du_dt) + jnp.sum(out.dv_dt)

    gb = np.asarray(g(_net()).layers[-1].bias)
    assert np.all(np.abs(gb[4 * NLEV:6 * NLEV]) > 0.0)


def _positional_nlev(adapter):
    import tests.unit.test_learned_arms_driver_path_planes as m
    assert m.NLEV == NLEV
    return _positional(adapter)


def test_old_column_checkpoint_is_refused(tmp_path):
    from legoesm.ml.checkpoint_io import (CheckpointStructureError,
                                          load_checkpoint_or_fail)
    new = _net()
    old_last = eqx.nn.Linear(8, 4 * NLEV + 6, key=jax.random.PRNGKey(2))
    old = eqx.tree_at(lambda m: m.layers[-1], new, old_last)
    path = tmp_path / "old.eqx"
    eqx.tree_serialise_leaves(str(path), old)
    with pytest.raises(CheckpointStructureError, match="u/v wind"):
        load_checkpoint_or_fail(path, new)


def test_wind_rows_are_inside_the_tanh_cap(latlon_adapter):
    """A huge raw wind output saturates at the 5e-4 m/s^2 cap."""
    step = nph.make_neural_step_unified(_with_wind_bias(_net(), raw=1.0e6),
                                        latlon_adapter)
    out, _ = step(True, *_positional_nlev(latlon_adapter))
    np.testing.assert_allclose(np.asarray(out.du_dt), 5.0e-4, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.dv_dt), 5.0e-4, rtol=1e-12)


def test_gridded_moisture_heads_are_at_moisture_scale():
    """The grid-point unpack scales the q_v, q_c and q_r heads by the
    spectral adapter's _Q_HEAD_TENDENCY_FACTOR (user 2026-11-06); T stays at
    the temperature scale."""
    nlev = 3
    y = jnp.arange(1.0, 6 * nlev + 7.0)
    out = nph._unpack_column_output(y, nlev)
    f = nph._Q_HEAD_TENDENCY_FACTOR
    assert jnp.allclose(out.dT_dt, y[:nlev])
    assert jnp.allclose(out.dq_v_dt, y[nlev:2 * nlev] * f)
    assert jnp.allclose(out.dq_c_dt, y[2 * nlev:3 * nlev] * f)
    assert jnp.allclose(out.dq_r_dt, y[3 * nlev:4 * nlev] * f)

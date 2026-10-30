"""Every learned atmosphere model can change the winds.

The column network (spectral NeuralGCM adapter and the gridded driver-path
step) carries a zero-initialised u/v head scaled to 1e-4 m/s^2 per raw unit
(tanh cap 5e-4 m/s^2), added to any drag; SFNOPhysics forwards its u/v output
channels in all three of its step builders. Needs JAX_ENABLE_X64=1.
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
from legoesm.ml.channel_packing import PE3DChannelSpec  # noqa: E402
from legoesm.ml.sfno import SFNO, SFNOConfig  # noqa: E402
from legoesm.training.neural_gcm_spectral import (  # noqa: E402
    carry_to_spectral_state,
)
from legoesm.training.sfno_dycore_coupling import (  # noqa: E402
    SFNOPhysics,
    make_sfno_step_unified,
    make_sfno_step_unified_latlon,
)

from tests.unit.test_learned_arms_driver_path_planes import (  # noqa: E402
    _positional,
)
from tests.unit.test_learned_column import _mini_spectral_state  # noqa: E402
from tests.unit.test_sfno_latlon_coupling import (  # noqa: E402
    N_LAT_LL, N_LON_LL, _call_step, _weights,
)

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


# ------------------------------------------------------------------- SFNO

@pytest.fixture(scope="module")
def sfno_ph():
    gauss = create_gaussian_grid(10)
    n_ch = 4 * NLEV + 2
    sfno = SFNO(SFNOConfig(in_channels=n_ch, out_channels=n_ch, embed_dim=8,
                           n_blocks=1, residual_prediction=False),
                gauss, key=jax.random.PRNGKey(0))
    spec = PE3DChannelSpec(nlev=NLEV)
    bias = jnp.zeros_like(sfno.decoder.bias)
    bias = bias.at[spec.u_slice].set(2.0e-4).at[spec.v_slice].set(-1.0e-4)
    sfno = eqx.tree_at(lambda m: (m.decoder.weight, m.decoder.bias), sfno,
                       (jnp.zeros_like(sfno.decoder.weight), bias))
    return SFNOPhysics(sfno=sfno, grid=gauss, nlev=NLEV)


def _gauss_fields(ph):
    shp = (int(ph.grid.n_lat), int(ph.grid.n_lon), NLEV)
    T = jnp.full(shp, 280.0)
    return T, jnp.zeros(shp), jnp.zeros(shp), jnp.full(shp, 5e-3), \
        jnp.full(shp[:2], 1.0e5), jnp.zeros(shp[:2])


def test_sfno_physics_forwards_raw_u_v_channels(sfno_ph):
    T, u, v, q, ps, phis = _gauss_fields(sfno_ph)
    out = sfno_ph(T, u, v, q, ps, phis, jnp.asarray(600.0))
    np.testing.assert_allclose(np.asarray(out.du_dt), 2.0e-4, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.dv_dt), -1.0e-4, rtol=1e-12)


def test_sfno_zero_decoder_gives_zero_wind(sfno_ph):
    z = eqx.tree_at(lambda m: m.sfno.decoder.bias, sfno_ph,
                    jnp.zeros_like(sfno_ph.sfno.decoder.bias))
    T, u, v, q, ps, phis = _gauss_fields(z)
    out = z(T, u, v, q, ps, phis, jnp.asarray(600.0))
    assert bool(jnp.all(out.du_dt == 0.0)) and bool(jnp.all(out.dv_dt == 0.0))


def test_sfno_correction_adds_to_the_physics_winds(sfno_ph):
    T, u, v, q, ps, phis = _gauss_fields(sfno_ph)
    trad = sfno_ph(T, u, v, q, ps, phis, jnp.asarray(600.0))
    trad = trad._replace(du_dt=jnp.full_like(T, 1.0e-5),
                         dv_dt=jnp.full_like(T, 3.0e-5))

    def _trad_step(*args, **kwargs):
        return trad, tuple([None] * 6)

    step = make_sfno_step_unified(sfno_ph, mode="correction",
                                  traditional_step_unified=_trad_step)
    out = _call_step(step, T, ps, q)[0]
    np.testing.assert_allclose(np.asarray(out.du_dt), 2.1e-4, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.dv_dt), -0.7e-4, rtol=1e-12)


def test_sfno_latlon_regrids_the_winds(sfno_ph):
    g = sfno_ph.grid
    ll_lat = jnp.linspace(-np.pi / 2 * 0.95, np.pi / 2 * 0.95, N_LAT_LL)
    ll_lon = jnp.linspace(0.0, 2 * np.pi, N_LON_LL, endpoint=False)
    w_ll2g = _weights(ll_lat, ll_lon, g.lat, g.lon,
                      (int(g.n_lat), int(g.n_lon)))
    w_g2ll = _weights(g.lat, g.lon, ll_lat, ll_lon, (N_LAT_LL, N_LON_LL))
    step = make_sfno_step_unified_latlon(sfno_ph, w_ll2g, w_g2ll)
    T = jnp.full((N_LAT_LL, N_LON_LL, NLEV), 280.0)
    out, _ = _call_step(step, T, jnp.full((N_LAT_LL, N_LON_LL), 1.0e5),
                        jnp.full_like(T, 5e-3))
    np.testing.assert_allclose(np.asarray(out.du_dt), 2.0e-4, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(out.dv_dt), -1.0e-4, rtol=1e-10)


def test_sfno_gradient_reaches_the_wind_rows(sfno_ph):
    spec = PE3DChannelSpec(nlev=NLEV)
    T, u, v, q, ps, phis = _gauss_fields(sfno_ph)

    @eqx.filter_grad
    def g(ph):
        out = ph(T, u, v, q, ps, phis, jnp.asarray(600.0))
        return jnp.sum(out.du_dt) + jnp.sum(out.dv_dt)

    gb = np.asarray(g(sfno_ph).sfno.decoder.bias)
    assert np.all(np.abs(gb[spec.u_slice]) > 0.0)
    assert np.all(np.abs(gb[spec.v_slice]) > 0.0)


def test_wind_rows_are_inside_the_tanh_cap(latlon_adapter):
    """A huge raw wind output saturates at the 5e-4 m/s^2 cap."""
    step = nph.make_neural_step_unified(_with_wind_bias(_net(), raw=1.0e6),
                                        latlon_adapter)
    out, _ = step(True, *_positional_nlev(latlon_adapter))
    np.testing.assert_allclose(np.asarray(out.du_dt), 5.0e-4, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(out.dv_dt), 5.0e-4, rtol=1e-12)

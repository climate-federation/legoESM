"""Tests for the state-conditioned Morrison warm-rain scaling network."""

import equinox as eqx
import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.microphysics.morrison import WarmRainRateScales
from legoesm.training.morrison_scale_net import MorrisonWarmRainScaleNet


def _state(ncol=2, nlev=8):
    T = jnp.linspace(290.0, 240.0, nlev)[None, :].repeat(ncol, 0)
    q_v = jnp.full((ncol, nlev), 6e-3)
    q_c = jnp.full((ncol, nlev), 4e-4)
    q_r = jnp.full((ncol, nlev), 1e-4)
    rho = jnp.full((ncol, nlev), 1.0)
    return T, q_v, q_c, q_r, rho


def test_identity_initialisation():
    net = MorrisonWarmRainScaleNet(jax.random.PRNGKey(0))
    scales = net(*_state())
    assert isinstance(scales, WarmRainRateScales)
    for f in (scales.autoconv, scales.accretion, scales.rain_evap):
        assert jnp.allclose(f, 1.0, atol=1e-6)


def test_output_shapes_match_state():
    net = MorrisonWarmRainScaleNet(jax.random.PRNGKey(1))
    T, *_rest = _state(3, 5)
    scales = net(T, *_rest)
    assert scales.autoconv.shape == (3, 5)
    assert scales.rain_evap.shape == (3, 5)


def test_trained_weights_move_off_identity_and_stay_bounded():
    net = MorrisonWarmRainScaleNet(jax.random.PRNGKey(2), bound=10.0)
    # Perturb the final layer so the net is no longer identity.
    last = net.mlp.layers[-1]
    net = eqx.tree_at(
        lambda n: n.mlp.layers[-1].bias, net, last.bias + 1.0,
    )
    scales = net(*_state())
    assert not jnp.allclose(scales.autoconv, 1.0)
    # exp(log(10) * tanh(.)) is strictly within (0.1, 10).
    for f in (scales.autoconv, scales.accretion, scales.rain_evap):
        assert jnp.all(f > 0.1 - 1e-6) and jnp.all(f < 10.0 + 1e-6)


def test_differentiable_through_net():
    net = MorrisonWarmRainScaleNet(jax.random.PRNGKey(3))
    state = _state()

    def loss(model):
        s = model(*state)
        return jnp.sum(s.autoconv ** 2 + s.accretion ** 2 + s.rain_evap ** 2)

    grads = eqx.filter_grad(loss)(net)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_array(g)]
    assert leaves and all(jnp.all(jnp.isfinite(g)) for g in leaves)


def test_fresh_net_in_morrison_is_bit_identical_to_baseline():
    """A FRESH identity-init net wrapped into Morrison reproduces baseline
    Morrison tendencies bit-for-bit (codex iter-1, finding 4).

    The dedicated identity-output test (above) only checks the net's scalar
    output is ~1.0; this exercises the full Morrison path with the ACTUAL net
    object as ``warm_rain_scale_fn`` and asserts exact equality of every
    warm-rain-affected tendency against unscaled baseline.
    """
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

    from legoesm import constants

    ncol, nlev = 2, 12
    T = jnp.linspace(285.0, 235.0, nlev)[None, :].repeat(ncol, 0)
    q_v = jnp.full((ncol, nlev), 6e-3)
    q_c = jnp.full((ncol, nlev), 6e-4)
    q_r = jnp.full((ncol, nlev), 2e-4)
    z = jnp.zeros((ncol, nlev))
    hyd = HydrometeorState(
        q_c=q_c, q_r=q_r, q_i=z, q_s=z, q_g=z,
        N_c=jnp.full((ncol, nlev), 1e8), N_r=jnp.full((ncol, nlev), 1e6),
        N_i=z, N_s=None, N_g=None,
    )
    p_full = jnp.linspace(1.0e5, 2.0e4, nlev)[None, :].repeat(ncol, 0)
    p_half = jnp.linspace(1.01e5, 1.9e4, nlev + 1)[None, :].repeat(ncol, 0)
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 500.0)
    dt = 60.0

    base = MorrisonConfig()
    out0 = morrison_microphysics(
        T, q_v, hyd, p_full, p_half, rho, dz, dt, base)

    net = MorrisonWarmRainScaleNet(jax.random.PRNGKey(7))
    out1 = morrison_microphysics(
        T, q_v, hyd, p_full, p_half, rho, dz, dt,
        base._replace(warm_rain_scale_fn=net))

    # Identity init: exp(log_bound·tanh(0)) = 1 ⇒ scaling by 1.0. The net's
    # MLP output is float; multiplying by 1.0 is exact, so equality is bit-exact.
    for f in ("dq_v_dt", "dq_c_dt", "dq_r_dt", "dT_dt",
              "dN_c_dt", "dN_r_dt", "precipitation"):
        assert jnp.array_equal(getattr(out0, f), getattr(out1, f)), f

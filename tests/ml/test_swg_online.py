"""Direct tests for the SWG online-learning network, loss, and segment seam."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.swe_rk3 import _entry_from_state, swe_rk3_step
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    build_swg_card,
    swg_surface_wind,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training.swg_online import (
    BalancedNormalizers,
    BalancedTrajectory,
    SWGNetworkSpec,
    SWGScalarPotentialCNN,
    assert_network_gradients_live,
    balanced_trajectory_loss,
    build_closure_geometry,
    build_segment_fn,
    network_parameter_count,
)


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def online_case():
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    tau = swg_surface_wind(SWG_DG50_NAMELIST)
    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau[0]), tau_j_native=jnp.asarray(tau[1]))
    entry = _entry_from_state(
        card.recipe.initial_state, tau, tau, card.recipe.grid,
        card.recipe.z_coord, card.recipe.model_config)
    spec = SWGNetworkSpec(
        hidden_channels=8, hidden_layers=2, kernel_size=3,
        ssh_scale_m=1.0, velocity_scale_m_s=1.0,
        potential_scale_m3_s2=1.0)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(8))
    return card, geometry, forcing, entry, network


def test_network_manifest_shape_parameter_count_and_noise_hook(online_case):
    _, geometry, _, entry, network = online_case
    zero = jnp.zeros_like(entry.ssh_k)
    first = network(
        entry, geometry, key=jax.random.PRNGKey(1), noise=zero,
        noise_scale=jnp.asarray(0.0))
    second = network(
        entry, geometry, key=jax.random.PRNGKey(2), noise=zero,
        noise_scale=jnp.asarray(0.0))
    stochastic = network(
        entry, geometry, key=jax.random.PRNGKey(2), noise=zero,
        noise_scale=jnp.asarray(1.0e-3))
    assert first.shape == entry.ssh_k.shape
    assert network_parameter_count(network) == 1096
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert np.any(np.asarray(second) != np.asarray(stochastic))
    assert np.all(np.asarray(first)[~np.asarray(
        geometry.vertex_interior, dtype=bool)] == 0.0)
    with pytest.raises(ValueError, match="noise"):
        network(entry, geometry, key=jax.random.PRNGKey(0),
                noise=jnp.zeros((2, 2)), noise_scale=0.0)


def test_network_args_are_traced_without_retrace(online_case):
    _, geometry, _, entry, network = online_case
    traces = []

    def potential(net, noise, key):
        traces.append(None)
        return net(entry, geometry, key=key, noise=noise, noise_scale=0.0)

    compiled = eqx.filter_jit(potential)
    zero = jnp.zeros_like(entry.ssh_k)
    compiled(network, zero, jax.random.PRNGKey(0)).block_until_ready()
    changed = eqx.tree_at(
        lambda value: value.layers[0].weight, network,
        network.layers[0].weight + 1.0e-8)
    compiled(changed, zero.at[20, 20].set(1.0e-6),
             jax.random.PRNGKey(1)).block_until_ready()
    assert len(traces) == 1


def test_point_and_energy_losses_share_interface_and_plants_fail(online_case):
    _, geometry, _, _, _ = online_case
    shape = (4, 6) + tuple(geometry.tmask.shape)
    base = jnp.zeros(shape)
    target_shape = shape[1:]
    target_field = jnp.zeros(target_shape).at[:, 20:30, 20:30].set(1.0)
    predicted = BalancedTrajectory(base, base, base)
    target = BalancedTrajectory(target_field, target_field, target_field)
    normalizers = BalancedNormalizers(1.0, 1.0, 1.0)
    point = balanced_trajectory_loss(
        predicted, target, geometry, normalizers, kind="point")
    energy = balanced_trajectory_loss(
        predicted, target, geometry, normalizers, kind="energy_score")
    assert float(point) > 0.0
    assert float(energy) > 0.0
    with pytest.raises(ValueError, match="at least two"):
        balanced_trajectory_loss(
            BalancedTrajectory(base[:1], base[:1], base[:1]), target,
            geometry, normalizers, kind="energy_score")
    with pytest.raises(ValueError, match="unknown"):
        balanced_trajectory_loss(
            predicted, target, geometry, normalizers, kind="spectral")


def test_one_step_jvp_matches_finite_difference_and_all_stages_act(online_case):
    card, geometry, forcing, entry, _ = online_case
    state = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    pattern = jnp.sin(jnp.arange(entry.ssh_k.size, dtype=jnp.float64).reshape(
        entry.ssh_k.shape)) * geometry.vertex_interior

    def endpoint(amplitude, stages=(1, 2, 3)):
        return swe_rk3_step(
            state, card.dt_s, forcing, card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config,
            stage_potential_fn=lambda _, stage: (
                amplitude * pattern if stage in stages else jnp.zeros_like(pattern)
            )).u.data

    amplitude = jnp.asarray(1.0e-2)
    tangent = jax.jvp(endpoint, (amplitude,), (jnp.asarray(1.0),))[1]
    epsilon = np.float64(1.0e-5)
    finite = (endpoint(amplitude + epsilon) - endpoint(amplitude - epsilon)) / (
        2.0 * epsilon)
    np.testing.assert_allclose(tangent, finite, rtol=2.0e-8, atol=2.0e-12)
    full = endpoint(amplitude)
    for omitted in (1, 2, 3):
        partial = endpoint(amplitude, tuple(
            stage for stage in (1, 2, 3) if stage != omitted))
        assert np.any(np.asarray(full) != np.asarray(partial))


def test_24_hour_jvp_matches_finite_difference_and_jit_matches_eager(online_case):
    card, geometry, forcing, entry, _ = online_case
    state = card.recipe.initial_state._replace(
        tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    pattern = (jnp.sin(jnp.arange(
        entry.ssh_k.size, dtype=jnp.float64).reshape(entry.ssh_k.shape) / 7.0)
        * geometry.vertex_interior)

    def objective(amplitude):
        def body(carry, _):
            carry = swe_rk3_step(
                carry, card.dt_s, forcing, card.recipe.grid,
                card.recipe.z_coord, card.recipe.model_config,
                stage_potential_fn=lambda _entry, _stage: amplitude * pattern)
            return carry, None

        final, _ = jax.lax.scan(body, state, None, length=216)
        return jnp.sum(final.u.data * final.u.data)

    amplitude = jnp.asarray(1.0e-2)
    eager = objective(amplitude)
    compiled = jax.jit(objective)(amplitude)
    assert np.asarray(eager) == np.asarray(compiled)
    tangent = jax.jvp(jax.jit(objective), (amplitude,),
                      (jnp.asarray(1.0),))[1]
    epsilon = np.float64(2.0e-5)
    finite = (jax.jit(objective)(amplitude + epsilon)
              - jax.jit(objective)(amplitude - epsilon)) / (2.0 * epsilon)
    np.testing.assert_allclose(tangent, finite, rtol=2.0e-5, atol=2.0e-10)


def test_segment_builder_rejects_unregistered_windows_and_has_live_gradients(
        online_case):
    card, geometry, _, entry, network = online_case
    with pytest.raises(ValueError, match="window_days"):
        build_segment_fn(card, geometry, window_days=16)

    # The first-step gate probes the actual stage seam.  A spatial objective is
    # used instead of an integrated potential, whose constant mode is correctly
    # annihilated by the curl.
    zero = jnp.zeros_like(entry.ssh_k)

    def objective(net):
        potential = net(
            entry, geometry, key=jax.random.PRNGKey(42), noise=zero,
            noise_scale=0.0)
        weights = jnp.arange(potential.size, dtype=potential.dtype).reshape(
            potential.shape)
        return jnp.sum(potential * weights)

    grads = eqx.filter_grad(objective)(network)
    assert_network_gradients_live(grads)

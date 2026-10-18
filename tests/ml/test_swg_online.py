"""Direct tests for the SWG online-learning network, loss, and segment seam."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ml.training import TrainingConfig, create_optimizer
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.swe_rk3 import (
    _entry_from_state,
    swe_rk3_stage,
    swe_rk3_step,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_MADDISON_NAMELIST,
    SWG_MADDISON_RL_NAMELIST,
    build_swg_card,
    swg_surface_wind,
)
from legoesm.ocean.state import OceanSurfaceForcing
from legoesm.training import swg_online
from legoesm.training.swg_online import (
    RIGID_LID_RESTART_TOLERANCE,
    START_REBALANCE_MAX_ITERATIONS,
    START_REBALANCE_TOLERANCE,
    BalancedNormalizers,
    BalancedTrajectory,
    SWGDatasetContract,
    SWGNetworkSpec,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    WindowBatch,
    assert_network_gradients_live,
    assert_no_convolution_operations,
    balanced_fields_from_native,
    balanced_trajectory_loss,
    build_closure_geometry,
    build_segment_fn,
    build_start_rebalancer,
    closed_wall_divergent_part,
    coarse_state_from_native,
    create_swg_optimizer,
    deterministic_trajectory_loss,
    eq6_scale_report,
    eq7_point_weights,
    initial_closure_acceleration_ratio,
    load_content_pinned_hourly_segment,
    load_content_pinned_initial_state,
    maddison_eq6_network_scales,
    network_parameter_count,
    require_target_interval_hours,
    steps_per_target_interval,
    swg_online_step,
    wall_margin_mask,
)

# A 1,000 km square dg50-physics card (20x20 wet cells) keeps the synthetic
# dataset records small; every other test uses the real dg50 card.
_SMALL_NAMELIST = SWG_DG50_NAMELIST._replace(rn_domsiz_m=1.0e6)


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
        potential_scale_m3_s2=1.0, wall_margin_rings=0)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(8))
    return card, geometry, forcing, entry, network


def _eq7(card, wall_ring="weighted"):
    """Eq. 7 constants from the deck: beta and the full side rn_domsiz_m."""
    return {"beta_m1_s1": card.resolved_namelist.rn_beta_m_s,
            "basin_length_m": card.resolved_namelist.rn_domsiz_m,
            "wall_ring": wall_ring}


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
    assert network_parameter_count(network) == 952
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


def test_wall_margin_mask_erodes_with_the_four_neighbour_stencil():
    rectangle = np.zeros((7, 8), dtype=bool)
    rectangle[1:6, 1:7] = True
    ring_1 = np.zeros_like(rectangle)
    ring_1[2:5, 2:6] = True
    ring_2 = np.zeros_like(rectangle)
    ring_2[3, 3:5] = True
    for rings, expected in ((0, rectangle), (1, ring_1), (2, ring_2)):
        eroded = wall_margin_mask(jnp.asarray(rectangle), rings)
        assert eroded.dtype == jnp.bool_
        assert np.array_equal(np.asarray(eroded), expected)
    assert wall_margin_mask(jnp.asarray(rectangle), 3).sum() == 0

    hole = np.ones((5, 5), dtype=bool)
    hole[2, 2] = False
    expected = np.zeros_like(hole)
    expected[1:4, 1:4] = True
    expected[1, 2] = expected[3, 2] = expected[2, 1] = expected[2, 3] = False
    expected[2, 2] = False
    assert np.array_equal(np.asarray(wall_margin_mask(jnp.asarray(hole), 1)),
                          expected)

    floats = jnp.asarray(rectangle, dtype=jnp.float64)
    assert wall_margin_mask(floats, 1).dtype == jnp.float64
    assert np.array_equal(np.asarray(wall_margin_mask(floats, 2)), ring_2)
    jitted = jax.jit(wall_margin_mask, static_argnums=1)
    assert np.array_equal(np.asarray(jitted(jnp.asarray(rectangle), 2)), ring_2)


def _margin_networks(online_case, rings):
    _, _, _, _, network = online_case
    spec = network.spec._replace(wall_margin_rings=rings)
    return SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(8))


def _potential(network, entry, geometry):
    return network(
        entry, geometry, key=jax.random.PRNGKey(1),
        noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0))


def test_wall_margin_zeroes_the_potential_on_the_first_rings_only(online_case):
    _, geometry, _, entry, _ = online_case
    plain = _potential(_margin_networks(online_case, 0), entry, geometry)
    margin = _potential(_margin_networks(online_case, 2), entry, geometry)
    interior = np.asarray(geometry.vertex_interior, dtype=bool)
    keep_1 = np.asarray(wall_margin_mask(geometry.vertex_interior, 1))
    keep_2 = np.asarray(wall_margin_mask(geometry.vertex_interior, 2))
    j0, j1, i0, i1 = geometry.wet_bounds
    assert keep_2.sum() == (j1 - j0 - 4) * (i1 - i0 - 4)
    plain, margin = np.asarray(plain), np.asarray(margin)
    assert np.all(margin[~keep_2] == 0.0)
    assert np.array_equal(margin[keep_2], plain[keep_2])
    assert np.any(plain[interior & ~keep_1] != 0.0)
    assert np.any(plain[keep_1 & ~keep_2] != 0.0)
    with pytest.raises(ValueError, match="wall_margin_rings"):
        _margin_networks(online_case, -1)


def test_wall_margin_closure_acceleration_vanishes_next_to_every_wall(
        online_case):
    card, geometry, _, entry, _ = online_case
    grid, z_coord = card.recipe.grid, card.recipe.z_coord
    config = card.recipe.model_config
    mask = card.recipe.initial_state.land_mask.data

    def acceleration(rings):
        potential = _potential(
            _margin_networks(online_case, rings), entry, geometry)

        def stage(potential_f):
            return swe_rk3_stage(entry, entry, card.dt_s, 3, grid, z_coord,
                                 mask, config, potential_f=potential_f)

        base, closed = stage(None), stage(potential)
        return (np.asarray(closed.u_k - base.u_k) / card.dt_s,
                np.asarray(closed.v_k - base.v_k) / card.dt_s)

    j0, j1, i0, i1 = geometry.wet_bounds
    wall_rows = [j0, j0 + 1, j1 - 1, j1]
    wall_columns = [i0, i0 + 1, i1 - 1, i1]
    plain_u, plain_v = acceleration(0)
    assert np.any(plain_u[wall_rows[:2]] != 0.0)
    assert np.any(plain_u[wall_rows[2:]] != 0.0)
    assert np.any(plain_v[:, wall_columns[:2]] != 0.0)
    assert np.any(plain_v[:, wall_columns[2:]] != 0.0)
    margin_u, margin_v = acceleration(2)
    assert np.all(margin_u[wall_rows] == 0.0)
    assert np.all(margin_v[:, wall_columns] == 0.0)
    assert np.any(margin_u[j0 + 2:j1 - 1] != 0.0)
    assert np.any(margin_v[:, i0 + 2:i1 - 1] != 0.0)


def test_point_and_energy_losses_share_interface_and_plants_fail(online_case):
    card, geometry, _, _, _ = online_case
    shape = (4, 6) + tuple(geometry.tmask.shape)
    base = jnp.zeros(shape)
    target_field = jnp.zeros(shape[1:]).at[:, 20:30, 20:30].set(1.0)
    predicted = BalancedTrajectory(base, base, base)
    target = BalancedTrajectory(target_field, target_field, target_field)
    point_target = jax.tree.map(
        lambda value: jnp.broadcast_to(value, base.shape), target)
    unit = BalancedNormalizers(1.0, 1.0, 1.0)
    point = balanced_trajectory_loss(
        predicted, point_target, geometry, None, kind="point", **_eq7(card))
    energy = balanced_trajectory_loss(
        predicted, target, geometry, unit, kind="energy_score")
    assert float(point) > 0.0
    assert float(energy) > 0.0
    with pytest.raises(ValueError, match="no data normalizers"):
        balanced_trajectory_loss(
            predicted, point_target, geometry, unit, kind="point",
            **_eq7(card))
    with pytest.raises(ValueError, match="explicit normalizers"):
        balanced_trajectory_loss(
            predicted, target, geometry, None, kind="energy_score")
    with pytest.raises(ValueError, match="at least two"):
        balanced_trajectory_loss(
            BalancedTrajectory(base[:1], base[:1], base[:1]), target,
            geometry, unit, kind="energy_score")
    with pytest.raises(ValueError, match="unknown"):
        balanced_trajectory_loss(
            predicted, target, geometry, None, kind="spectral")


def test_energy_score_uses_unbiased_per_lead_pair_term(online_case):
    """Brolly's {truth, truth+1} counterexample has exactly zero score."""
    _, geometry, _, _, _ = online_case
    target_shape = (3,) + tuple(geometry.tmask.shape)
    truth = jnp.zeros(target_shape)
    shifted = jnp.ones(target_shape)
    members = jnp.stack((truth, shifted))
    predicted = BalancedTrajectory(members, members, members)
    target = BalancedTrajectory(truth, truth, truth)
    score = balanced_trajectory_loss(
        predicted, target, geometry, BalancedNormalizers(1.0, 1.0, 1.0),
        kind="energy_score")
    assert float(score) == pytest.approx(0.0, abs=2.0e-15)


def test_eq7_scale_uses_half_the_deck_domain_and_ignores_velocity(online_case):
    """L is half of rn_domsiz_m: Maddison Table 1 reads "Domain size 2L"."""
    card, geometry, _, _, _ = online_case
    shape = (2,) + tuple(geometry.tmask.shape)
    zeta = jnp.zeros(shape).at[:, 20, 20].set(3.0e-7)
    zero = jnp.zeros(shape)
    target = BalancedTrajectory(zero[None], zero[None], zero[None])
    namelist = card.resolved_namelist
    beta = namelist.rn_beta_m_s
    half_width = 0.5 * namelist.rn_domsiz_m

    def point(predicted, **overrides):
        return balanced_trajectory_loss(
            predicted, target, geometry, None, kind="point",
            **(_eq7(card) | overrides))

    baseline = point(BalancedTrajectory(zeta[None], zero[None], zero[None]))
    expected = (1.0e4 / (beta**2 * half_width**2) / (4.0 * half_width**2)
                * geometry.spacing_m**2 * (3.0e-7)**2)
    assert float(baseline) == pytest.approx(expected, rel=2.0e-15)
    planted = jnp.ones(shape) * 1.0e3
    assert float(point(BalancedTrajectory(
        zeta[None], planted[None], planted[None]))) == float(baseline)
    # The paper's L itself, twice the side, or a non-number cannot be passed.
    for wrong in (half_width, 2.0 * namelist.rn_domsiz_m, float("nan")):
        with pytest.raises(ValueError, match="rn_domsiz_m"):
            point(BalancedTrajectory(zeta[None], zero[None], zero[None]),
                  basin_length_m=wrong)


def test_eq7_weights_every_interior_f_point_and_nothing_else(online_case):
    """The loss gradient per point is 2 * scale * W: pin W's support."""
    card, geometry, _, _, _ = online_case
    ones = jnp.ones((1, 1) + tuple(geometry.tmask.shape))
    zero = jnp.zeros_like(ones)

    def weights(ring):
        def loss(zeta):
            return balanced_trajectory_loss(
                BalancedTrajectory(zeta, zero, zero),
                BalancedTrajectory(zero, zero, zero), geometry, None,
                kind="point", **_eq7(card, ring))
        return np.asarray(jax.grad(loss)(ones))[0, 0]

    j0, j1, i0, i1 = geometry.wet_bounds
    interior = np.zeros(geometry.tmask.shape, dtype=bool)
    interior[j0:j1, i0:i1] = True       # F (j, i) is the NE corner of T (j, i)
    assert np.array_equal(interior, np.asarray(geometry.vertex_interior))
    weighted = weights("weighted")
    assert np.array_equal(weighted != 0.0, interior)
    assert np.unique(weighted[interior]).size == 1
    # the first interior ring is weighted at all four walls, symmetrically
    for j, i in ((j0, 40), (j1 - 1, 40), (40, i0), (40, i1 - 1)):
        assert weighted[j, i] == weighted[40, 40] > 0.0
    ring = interior.copy()
    ring[j0 + 1:j1 - 1, i0 + 1:i1 - 1] = False
    assert ring.sum() == 4 * (j1 - j0) - 4
    excluded = weights("excluded")
    assert np.array_equal(excluded != 0.0, interior & ~ring)
    assert np.all(excluded[interior & ~ring] == weighted[interior & ~ring])
    for unknown in (None, "include", True):
        with pytest.raises(ValueError, match="wall_ring"):
            weights(unknown)


def test_named_objectives_are_zero_at_start_through_one_diagnostic(
        online_case):
    """On a non-trivial state model and target share ONE diagnostic."""
    card, geometry, _, _, _ = online_case
    shape = tuple(geometry.tmask.shape)
    rng = np.random.default_rng(31)
    state = tuple(jnp.asarray(value)[None, None] for value in (
        rng.normal(0.0, 0.3, shape),
        0.2 * rng.normal(size=shape) * np.asarray(geometry.umask),
        0.2 * rng.normal(size=shape) * np.asarray(geometry.vmask)))
    jj, ii = np.indices(shape)
    checkerboard = jnp.asarray(0.1 * np.where((ii + jj) % 2, -1.0, 1.0))
    model = (state[0], state[1] + checkerboard, state[2])
    rises = {}
    for objective in ("symmetric", "template"):
        def loss(predicted, objective=objective):
            return deterministic_trajectory_loss(
                predicted, state, geometry, objective=objective, **_eq7(card))

        # Eagerly both sides run the same primitives on the same words.
        assert float(loss(state)) == 0.0
        loss_fn = jax.jit(loss)
        rises[objective] = float(loss_fn(model))
        # Under jit the closed-over target may be constant-folded, which can
        # round differently; it stays at round-off of the checkerboard rise.
        assert float(loss_fn(state)) <= 1.0e-12 * rises[objective]
        assert_no_convolution_operations(
            loss_fn.lower(model), label=f"{objective} training loss")
    assert rises["template"] > 100.0 * rises["symmetric"] > 0.0
    with pytest.raises(ValueError, match="explicitly selected"):
        deterministic_trajectory_loss(
            state, state, geometry, objective="typo", **_eq7(card))


def test_symmetric_gaussian_is_registered_on_f_points(online_case):
    """Mirror-symmetric at all four walls, zero on walls, Gaussian inside."""
    _, geometry, _, _, _ = online_case
    j0, j1, i0, i1 = geometry.wet_bounds
    interior = np.asarray(geometry.vertex_interior, dtype=bool)
    field = np.random.default_rng(93).normal(size=interior.shape) * interior
    block = (slice(j0 - 1, j1 + 1), slice(i0 - 1, i1 + 1))

    def smooth(value):
        return np.asarray(swg_online._gaussian_smooth_vorticity(
            jnp.asarray(value), geometry))

    def mirrored(value, axis):
        out = np.zeros_like(value)
        out[block] = np.flip(value[block], axis=axis)
        return out

    smoothed = smooth(field)
    scale = np.abs(smoothed).max()
    for axis in (0, 1):
        np.testing.assert_allclose(
            smooth(mirrored(field, axis)), mirrored(smoothed, axis),
            rtol=0.0, atol=1.0e-14 * scale)
    assert np.all(smoothed[~interior] == 0.0)
    offsets = np.arange(-4, 5, dtype=np.float64)
    kernel = np.exp(-0.5 * offsets**2)
    kernel /= kernel.sum()
    reference = np.apply_along_axis(
        np.convolve, 0, field, kernel, mode="same")
    reference = np.apply_along_axis(
        np.convolve, 1, reference, kernel, mode="same")
    core = (slice(j0 + 4, j1 - 4), slice(i0 + 4, i1 - 4))
    np.testing.assert_allclose(smoothed[core], reference[core],
                               rtol=1.0e-13, atol=1.0e-15)
    # Odd wall parity: a unit vortex beside any wall keeps the same mass,
    # w0 + w1 of the 1-D kernel (an even reflection would keep 1 - w1).
    kept = []
    for j, i in ((j0, 40), (j1 - 1, 40), (40, i0), (40, i1 - 1)):
        unit = np.zeros(interior.shape)
        unit[j, i] = 1.0
        kept.append(smooth(unit).sum())
    np.testing.assert_allclose(kept, kernel[4] + kernel[5], rtol=1.0e-14)
    assert_no_convolution_operations(
        jax.jit(lambda value: swg_online._gaussian_smooth_vorticity(
            value, geometry)).lower(jnp.asarray(field)),
        label="symmetric F-point Gaussian")


def _streamfunction_flow(geometry, amplitude):
    """Exactly non-divergent native C-grid flow from a wall-zero F psi."""
    j0, j1, i0, i1 = geometry.wet_bounds
    intervals = j1 - j0 + 1
    y, x = np.meshgrid(np.arange(1, j1 - j0 + 1, dtype=np.float64),
                       np.arange(1, i1 - i0 + 1, dtype=np.float64),
                       indexing="ij")
    psi = np.zeros(geometry.tmask.shape)
    psi[j0:j1, i0:i1] = amplitude * (
        np.sin(np.pi * x / intervals) * np.sin(2.0 * np.pi * y / intervals)
        + 0.3 * np.sin(3.0 * np.pi * x / intervals)
        * np.sin(np.pi * y / intervals))
    u = -(psi - np.roll(psi, 1, axis=0)) / geometry.spacing_m
    v = (psi - np.roll(psi, 1, axis=1)) / geometry.spacing_m
    return (u * np.asarray(geometry.umask), v * np.asarray(geometry.vmask))


def test_rotational_velocity_recovers_a_nondivergent_cgrid_flow(online_case):
    """Inverting F vorticity with walls at the wall vertices is exact."""
    _, geometry, _, _, _ = online_case
    u, v = _streamfunction_flow(geometry, 3.0e5)
    fields = balanced_fields_from_native(
        jnp.asarray(u), jnp.asarray(v), geometry, objective="template")
    scale = max(np.abs(u).max(), np.abs(v).max())
    assert np.abs(np.asarray(fields.rotational_u_m_s) - u).max() \
        <= 1.0e-11 * scale
    assert np.abs(np.asarray(fields.rotational_v_m_s) - v).max() \
        <= 1.0e-11 * scale


def test_closure_geometry_refuses_a_card_not_built_in_fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp32())
    try:
        card32 = build_swg_card(_SMALL_NAMELIST)
    finally:
        set_policy(old)
    with pytest.raises(ValueError, match="fp64 precision policy"):
        build_closure_geometry(card32)


def test_point_loss_owns_batch_reduction(online_case):
    card, geometry, _, _, _ = online_case
    shape = (1, 2) + tuple(geometry.tmask.shape)
    error = jnp.zeros(shape).at[:, :, 20, 20].set(2.0e-7)
    zero = jnp.zeros(shape)
    one = balanced_trajectory_loss(
        BalancedTrajectory(error, zero, zero),
        BalancedTrajectory(zero, zero, zero), geometry, None,
        kind="point", **_eq7(card))
    two_error = jnp.concatenate((error, jnp.zeros_like(error)), axis=0)
    two_zero = jnp.zeros_like(two_error)
    mixed = balanced_trajectory_loss(
        BalancedTrajectory(two_error, two_zero, two_zero),
        BalancedTrajectory(two_zero, two_zero, two_zero), geometry, None,
        kind="point", **_eq7(card))
    assert float(mixed) == pytest.approx(0.5 * float(one), rel=2.0e-15)


# --- balanced starts ------------------------------------------------------

def _unbalanced_start(geometry):
    """Non-divergent gyre flow on flat sea level plus a west-wall SSH bump."""
    u, v = _streamfunction_flow(geometry, 3.0e5)
    j0, j1, i0, _ = geometry.wet_bounds
    ssh = np.zeros(geometry.tmask.shape)
    ssh[j0:j1 + 1, i0] = 0.05
    return ssh, u, v


def _first_step_divergent_ratio(card, forcing, geometry, ssh, u, v):
    """Independent check through the card's production step (model.step)."""
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = coarse_state_from_native(
        card.recipe.initial_state, ssh, u, v)._replace(
            tau_x_prev=forcing.tau_i_native, tau_y_prev=forcing.tau_j_native)
    stepped = jax.jit(lambda value: model.step(
        value, card.dt_s, surface_forcing=forcing))(state)
    accel_u = (np.asarray(stepped.u.data[:, 1:, 0]) - u) / card.dt_s
    accel_v = (np.asarray(stepped.v.data[1:, :, 0]) - v) / card.dt_s
    wet = np.asarray(geometry.tmask, dtype=bool)
    _, div_u, div_v = closed_wall_divergent_part(
        accel_u, accel_v, wet, geometry.spacing_m)

    def centred_rms(a, b):
        a = 0.5 * (a + np.roll(a, 1, axis=1))
        b = 0.5 * (b + np.roll(b, 1, axis=0))
        return np.sqrt(np.mean(a[wet] ** 2 + b[wet] ** 2))

    return centred_rms(div_u, div_v) / centred_rms(
        accel_u - div_u, accel_v - div_v)


def test_rebalancer_moves_only_wet_sea_level_keeps_volume_and_balances(
        online_case):
    card, geometry, forcing, _, _ = online_case
    ssh, u, v = _unbalanced_start(geometry)
    copies = tuple(value.copy() for value in (ssh, u, v))
    rebalance = build_start_rebalancer(
        card, tolerance=START_REBALANCE_TOLERANCE,
        max_iterations=START_REBALANCE_MAX_ITERATIONS)
    balanced, report = rebalance(ssh, u, v)
    for value, copy in zip((ssh, u, v), copies, strict=True):
        assert np.array_equal(value, copy)
    wet = np.asarray(geometry.tmask, dtype=bool)
    assert np.array_equal(balanced[~wet], ssh[~wet])
    assert np.abs(balanced - ssh)[wet].max() > 1.0e-3
    area = np.asarray(card.recipe.z_coord.nemo_e1e2t)[wet]
    depth = np.asarray(card.recipe.z_coord.nemo_e3t_0)[..., 0][wet]
    volume_change = np.sum((balanced - ssh)[wet] * area)
    assert abs(volume_change) <= 1.0e-12 * np.sum(depth * area)
    assert report["volume_change_m3"] == pytest.approx(volume_change, abs=1.0)
    before = _first_step_divergent_ratio(card, forcing, geometry, ssh, u, v)
    after = _first_step_divergent_ratio(
        card, forcing, geometry, balanced, u, v)
    assert before > 10.0 * START_REBALANCE_TOLERANCE
    assert after <= START_REBALANCE_TOLERANCE
    assert report["divergent_to_rotational_rms"] <= START_REBALANCE_TOLERANCE
    again, again_report = rebalance(ssh, u, v)
    assert np.array_equal(again, balanced)
    assert again_report == report


def test_rebalancer_refuses_non_convergence_and_bad_settings(online_case):
    card, geometry, _, _, _ = online_case
    ssh, u, v = _unbalanced_start(geometry)
    with pytest.raises(ValueError, match="did not converge"):
        build_start_rebalancer(
            card, tolerance=START_REBALANCE_TOLERANCE,
            max_iterations=1)(ssh, u, v)
    with pytest.raises(ValueError, match="did not converge"):
        build_start_rebalancer(
            card, tolerance=1.0e-14, max_iterations=3)(ssh, u, v)
    for settings in ({"tolerance": 0.0, "max_iterations": 5},
                     {"tolerance": 1.0e-3, "max_iterations": -1},
                     {"tolerance": 1.0e-3, "max_iterations": 2.0}):
        with pytest.raises(ValueError, match="rebalancing"):
            build_start_rebalancer(card, **settings)
    with pytest.raises(ValueError, match="finite"):
        build_start_rebalancer(
            card, tolerance=START_REBALANCE_TOLERANCE,
            max_iterations=START_REBALANCE_MAX_ITERATIONS)(ssh * np.nan, u, v)


def test_rebalancer_accepts_an_exact_rest_state_unchanged():
    """No wind, no flow, flat sea: zero acceleration is already balanced."""
    card = build_swg_card(_SMALL_NAMELIST._replace(rn_tau_pa=0.0))
    shape = tuple(card.recipe.initial_state.eta.data.shape)
    zero = np.zeros(shape)
    balanced, report = build_start_rebalancer(
        card, tolerance=START_REBALANCE_TOLERANCE,
        max_iterations=START_REBALANCE_MAX_ITERATIONS)(zero, zero, zero)
    assert np.array_equal(balanced, zero)
    assert report["iterations"] == 0
    assert report["divergent_rotational_rms_history_m_s2"] == [[0.0, 0.0]]


# --- dataset contract ------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode(hours, scale, mask):
    """Velocity words that name their absolute hour and operator."""
    return (1.0e-6 * scale * np.asarray(hours, dtype=np.float64)[:, None, None]
            * np.asarray(mask, dtype=np.float64))


_OPERATOR_SCALE = {"box_": 1.0, "filtered_": 3.0}


def _write_chunk(root: Path, *, source_day: int, days: int, masks,
                 truth_run: Path, manifest_edit=None, day_edit=None) -> str:
    """Write one dual-operator v2 chunk as the driver writes it."""
    umask, vmask = masks
    shape = umask.shape
    root.mkdir(parents=True)
    hashes = {}
    for day in range(1, days + 1):
        hours = (source_day + day - 1) * 24 + np.arange(1, 25, dtype=np.int64)
        arrays = {"absolute_hour": hours}
        for prefix, scale in _OPERATOR_SCALE.items():
            arrays[f"{prefix}ssh_hourly_m"] = np.zeros((24,) + shape)
            arrays[f"{prefix}u_hourly_m_s"] = _encode(hours, scale, umask)
            arrays[f"{prefix}v_hourly_m_s"] = _encode(hours, -scale, vmask)
            if day == 1:
                start = [source_day * 24]
                arrays[f"{prefix}ssh_initial_m"] = np.zeros(shape)
                arrays[f"{prefix}u_initial_m_s"] = _encode(start, scale, umask)[0]
                arrays[f"{prefix}v_initial_m_s"] = _encode(
                    start, -scale, vmask)[0]
        if day_edit is not None:
            day_edit(day, arrays)
        path = root / f"day{day:04d}.npz"
        np.savez(path, **arrays)
        hashes[path.name] = _sha(path)
    namelist = SWG_MADDISON_NAMELIST._replace(
        nn_itend=days * 1728, nn_write=1728, nn_stock=1728)
    manifest = {
        "schema": "legoesm-swg-double-gyre-hourly-target-v2",
        "status": "complete", "days": days, "completed_days": days,
        "sampling": "hourly_dual_coarse_target",
        "hourly_samples_per_day": 24, "hourly_target_deck": "dg50",
        "precision_policy": "fp64",
        "restart": {
            "source_day": source_day, "perturb_amplitude_m": None,
            "perturb_kind": "white-ssh", "sha256": f"{source_day:064d}",
            "path": str(truth_run / f"day{source_day:04d}.npz"),
            "manifest_admission": {
                "manifest_path": str(truth_run / "manifest.json"),
                "manifest_status": "complete",
                "snapshot_sha256": f"{source_day:064d}"}},
        "restart_next_verification": {"result": {"all_equal": True}},
        "numerical_health": {"all_finite": True},
        "namelist": json.loads(json.dumps(namelist._asdict())),
        "daily_sha256": hashes,
    }
    if manifest_edit is not None:
        manifest_edit(manifest)
    # The truth run named by the (possibly edited) manifest lists this
    # chunk's restart snapshot with its original digest, as the driver's
    # admission found it.
    listing = Path(manifest["restart"]["manifest_admission"]["manifest_path"])
    listing.parent.mkdir(parents=True, exist_ok=True)
    truth = (json.loads(listing.read_text()) if listing.is_file()
             else {"daily_sha256": {}})
    truth["daily_sha256"][f"day{source_day:04d}.npz"] = f"{source_day:064d}"
    listing.write_text(json.dumps(truth))
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return _sha(path)


_SPLITS = {"train": [0, 528], "validation": [1248, 1488],
           "test": [2208, 2448]}
_CHUNKS = (("train_a", 0, 11), ("train_b", 11, 11),
           ("validation", 52, 10), ("test", 92, 10))


def _write_contract(path: Path, sources, *, objective="template",
                    splits=None) -> Path:
    path.write_text(json.dumps({
        "schema": "legoesm-swg-online-dataset-v1", "objective": objective,
        "batch_size": 10, "window_days": [1, 2, 4, 8],
        "splits": splits or _SPLITS,
        "sources": [{"root": str(root), "manifest_sha256": digest}
                    for root, digest in sources],
    }, sort_keys=True))
    return path


@pytest.fixture(scope="module")
def small_card():
    return build_swg_card(_SMALL_NAMELIST)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, small_card):
    """A 1,000 km record: train in two adjacent chunks, then two held-out."""
    root = tmp_path_factory.mktemp("swg_dataset")
    geometry = build_closure_geometry(small_card)
    masks = (np.asarray(geometry.umask), np.asarray(geometry.vmask))
    sources = [(root / name, _write_chunk(
        root / name, source_day=day, days=days, masks=masks,
        truth_run=root / "truth")) for name, day, days in _CHUNKS]
    path = _write_contract(root / "contract.json", sources)
    contract = SWGDatasetContract.from_json(path, card=small_card)
    return {"root": root, "masks": masks, "sources": sources, "path": path,
            "contract": contract, "card": small_card}


def test_contract_batches_start_from_rebalanced_stored_states(dataset):
    contract, card = dataset["contract"], dataset["card"]
    umask, vmask = dataset["masks"]
    batch = next(contract.batches(
        split="train", window_days=1, epoch=0, seed=12, target_interval_hours=1))
    assert batch.initial[0].shape == (10,) + umask.shape
    assert batch.target[0].shape == (10, 24) + umask.shape
    rebalance = build_start_rebalancer(
        card, tolerance=START_REBALANCE_TOLERANCE,
        max_iterations=START_REBALANCE_MAX_ITERATIONS)
    scale = _OPERATOR_SCALE["filtered_"]          # template reads filtered
    for k, start in enumerate(np.asarray(batch.start_absolute_hour)):
        stored_u = _encode([start], scale, umask)[0]
        stored_v = _encode([start], -scale, vmask)[0]
        assert np.array_equal(batch.initial[1][k], stored_u)
        assert np.array_equal(batch.initial[2][k], stored_v)
        balanced, _ = rebalance(np.zeros(umask.shape), stored_u, stored_v)
        assert np.array_equal(batch.initial[0][k], balanced)
        assert np.abs(balanced).max() > 0.0
        leads = start + np.arange(1, 25)
        assert np.array_equal(batch.target[1][k], _encode(leads, scale, umask))
        assert np.array_equal(batch.target[2][k],
                              _encode(leads, -scale, vmask))
        assert np.all(batch.target[0][k] == 0.0)


def test_training_sampler_redraws_offset_per_window_and_epoch(dataset):
    contract = dataset["contract"]

    def starts(window_days, epoch, seed):
        return np.concatenate([batch.start_absolute_hour for batch in
                               contract.batches(split="train",
                                                window_days=window_days,
                                                epoch=epoch, seed=seed, target_interval_hours=1)])

    one_day = {epoch: starts(1, epoch, 7) for epoch in range(6)}
    for values in one_day.values():
        assert np.unique(values).size == values.size
        assert np.unique(values % 24).size == 1          # one lattice
        assert values.min() >= 0 and values.max() + 24 <= 528
    assert len({int(values.min() % 24) for values in one_day.values()}) > 1
    assert any(np.any(np.diff(values) < 0) for values in one_day.values())
    assert np.array_equal(starts(1, 3, 7), one_day[3])
    assert any(not np.array_equal(starts(1, epoch, 8), one_day[epoch])
               for epoch in range(6))
    # The template draws the offset anew for every window length: a shared
    # draw keeps offset/window within 1/24 at every (seed, epoch).
    phases = [(starts(1, epoch, seed).min() % 24 / 24.0,
               starts(2, epoch, seed).min() % 48 / 48.0)
              for epoch in range(5) for seed in (7, 8)]
    assert max(abs(a - b) for a, b in phases) > 2.0 / 24.0


def test_heldout_splits_use_one_fixed_chronological_lattice(dataset):
    contract = dataset["contract"]
    first = [batch.start_absolute_hour for batch in
             contract.batches(split="validation", window_days=1, target_interval_hours=1)]
    again = [batch.start_absolute_hour for batch in
             contract.batches(split="validation", window_days=1, target_interval_hours=1)]
    assert np.array_equal(np.concatenate(first),
                          np.arange(1248, 1488 - 24 + 1, 24))
    assert np.array_equal(np.concatenate(again), np.concatenate(first))
    with pytest.raises(ValueError, match="epoch and seed must be None"):
        next(contract.batches(split="validation", window_days=1, epoch=0,
                              seed=1, target_interval_hours=1))
    with pytest.raises(ValueError, match="integer epoch and seed"):
        next(contract.batches(split="train", window_days=1, target_interval_hours=1))
    with pytest.raises(ValueError, match="unknown or leaked"):
        next(contract.batches(split="holdout", window_days=1, target_interval_hours=1))


def test_a_split_without_one_full_batch_raises(dataset):
    with pytest.raises(ValueError, match="fewer than one batch"):
        next(dataset["contract"].batches(
            split="train", window_days=4, epoch=0, seed=0, target_interval_hours=1))


def test_shared_restart_hour_uses_the_inscan_record_within_round_off(
        tmp_path, small_card):
    geometry = build_closure_geometry(small_card)
    masks = (np.asarray(geometry.umask), np.asarray(geometry.vmask))
    restart_hour = 11 * 24

    def nudged_initial(relative):
        def edit(day, arrays):
            if day == 1:
                arrays["filtered_u_initial_m_s"] = (
                    arrays["filtered_u_initial_m_s"] * (1.0 + relative))
        return edit

    def build(relative, label):
        root = tmp_path / label
        sources = [(root / name, _write_chunk(
            root / name, source_day=day, days=days, masks=masks,
            truth_run=tmp_path / "truth",
            day_edit=nudged_initial(relative) if name == "train_b" else None))
            for name, day, days in _CHUNKS]
        return SWGDatasetContract.from_json(
            _write_contract(root / "contract.json", sources), card=small_card)

    contract = build(4.0e-16, "rounding")
    served = contract._stored_state(restart_hour)
    assert np.array_equal(served[1], _encode([restart_hour], 3.0, masks[0])[0])
    with pytest.raises(ValueError, match="disagree at restart hour"):
        build(1.0e-6, "different")


# Two-day chunks keep each planted record small; gaps are exactly 720 h.
_PLANT_CHUNKS = (("train_a", 0, 2), ("train_b", 2, 2),
                 ("validation", 34, 2), ("test", 66, 2))
_PLANT_SPLITS = {"train": [0, 96], "validation": [816, 864],
                 "test": [1584, 1632]}


def _plant_dataset(tmp_path, small_card, label, *, chunk_edit=None,
                   splits=None, drop=None, duplicate=False, digest=None):
    geometry = build_closure_geometry(small_card)
    masks = (np.asarray(geometry.umask), np.asarray(geometry.vmask))
    root = tmp_path / label
    sources = []
    for name, day, days in _PLANT_CHUNKS:
        if name == drop:
            continue
        edit = chunk_edit(name) if chunk_edit is not None else None
        sources.append((root / name, _write_chunk(
            root / name, source_day=day, days=days, masks=masks,
            truth_run=tmp_path / "truth", manifest_edit=edit)))
    if duplicate:
        sources.append(sources[0])
    if digest is not None:
        sources[0] = (sources[0][0], digest)
    return _write_contract(root / "contract.json", sources,
                           splits=splits or _PLANT_SPLITS)


def _manifest_plant(target_name, edit):
    return lambda name: edit if name == target_name else None


_CONTRACT_PLANTS = {
    "gap of 719 hours": ({"splits": {"train": [0, 96],
                                     "validation": [815, 864],
                                     "test": [1584, 1632]}}, "gap"),
    "validation before train": ({"splits": {"train": [816, 864],
                                            "validation": [0, 96],
                                            "test": [1584, 1632]}}, "gap"),
    "hole in the training record": ({"drop": "train_b"}, "contiguous"),
    "manifest digest not pinned": ({"digest": "0" * 64}, "differs"),
    "duplicated source": ({"duplicate": True}, "duplicated"),
    "schema v1": ({"chunk_edit": _manifest_plant("train_a", lambda m: m.update(
        schema="legoesm-swg-double-gyre-hourly-target-v1"))}, "v2"),
    "incomplete run": ({"chunk_edit": _manifest_plant("train_a", lambda m: m.update(
        status="pending"))}, "complete"),
    "perturbed restart": ({"chunk_edit": _manifest_plant(
        "train_b", lambda m: m["restart"].update(perturb_amplitude_m=0.01))},
        "perturbed"),
    "another deck": ({"chunk_edit": _manifest_plant(
        "validation", lambda m: m["namelist"].update(rn_beta_m_s=1.0e-11))},
        "fine deck"),
    "failed next-day verification": ({"chunk_edit": _manifest_plant(
        "test", lambda m: m.update(
            restart_next_verification={"result": {"all_equal": False}}))},
        "verification"),
    "missing next-day verification": ({"chunk_edit": _manifest_plant(
        "test", lambda m: m.update(restart_next_verification=None))},
        "verification"),
    "fp32 stream": ({"chunk_edit": _manifest_plant(
        "train_a", lambda m: m.update(precision_policy="fp32"))}, "fp64"),
    "another truth run": ({"chunk_edit": _manifest_plant(
        "train_b", lambda m: m["restart"]["manifest_admission"].update(
            manifest_path=m["restart"]["manifest_admission"][
                "manifest_path"].replace("/truth/", "/other_truth/")))},
        "more than one truth"),
    "restart path names another day": ({"chunk_edit": _manifest_plant(
        "test", lambda m: m["restart"].update(
            path=m["restart"]["path"].replace("day0066", "day9999")))},
        "snapshot its truth run lists"),
    "truth run does not list the restart": ({"chunk_edit": _manifest_plant(
        "test", lambda m: (m["restart"].update(sha256="e" * 64),
                           m["restart"]["manifest_admission"].update(
                               snapshot_sha256="e" * 64)))},
        "snapshot its truth run lists"),
    "restart is not the listed snapshot": ({"chunk_edit": _manifest_plant(
        "train_b", lambda m: m["restart"]["manifest_admission"].update(
            snapshot_sha256="f" * 64))}, "snapshot its truth run lists"),
    "truth manifest refused": ({"chunk_edit": _manifest_plant(
        "validation", lambda m: m["restart"]["manifest_admission"].update(
            manifest_status="refused"))}, "snapshot its truth run lists"),
}


def _lid_stream(m, *, within=True, tolerance=RIGID_LID_RESTART_TOLERANCE,
                namelist=SWG_MADDISON_RL_NAMELIST):
    """Edit a chunk manifest into a rigid-lid stream as the driver records it."""
    m["surface"] = "rigid_lid"
    m["namelist"] = json.loads(json.dumps(namelist._asdict()))
    m["restart_next_verification"] = {
        "result": {"all_equal": False, "within_tolerance": within},
        "tolerance": None if tolerance is None else dict(tolerance)}


@pytest.fixture(scope="module")
def lid_card():
    return build_swg_card(_SMALL_NAMELIST, surface="rigid_lid")


def _every_chunk(edit):
    return lambda name: edit


def test_rigid_lid_stream_is_admitted_with_its_recorded_tolerance(
        tmp_path, lid_card):
    path = _plant_dataset(tmp_path, lid_card, "lid",
                          chunk_edit=_every_chunk(_lid_stream))
    contract = SWGDatasetContract.from_json(path, card=lid_card)
    assert {source.surface for source in contract.sources} == {"rigid_lid"}


@pytest.mark.parametrize("plant,message", [
    ("lid, outside tolerance", "verification"),
    ("lid, tolerance not recorded", "verification"),
    ("lid, another tolerance", "verification"),
    ("lid stream on the explicit namelist", "fine deck"),
    ("unknown surface", "surface must be"),
])
def test_rigid_lid_stream_refusals(tmp_path, lid_card, plant, message):
    def edit(m):
        _lid_stream(m)
        if plant == "lid, outside tolerance":
            _lid_stream(m, within=False)
        elif plant == "lid, tolerance not recorded":
            _lid_stream(m, tolerance=None)
        elif plant == "lid, another tolerance":
            _lid_stream(m, tolerance={**RIGID_LID_RESTART_TOLERANCE,
                                      "value": 1.0e-9})
        elif plant == "lid stream on the explicit namelist":
            _lid_stream(m, namelist=SWG_MADDISON_NAMELIST)
        else:
            m["surface"] = "ocean"
    path = _plant_dataset(tmp_path, lid_card, "lid",
                          chunk_edit=_every_chunk(edit))
    with pytest.raises(ValueError, match=message):
        SWGDatasetContract.from_json(path, card=lid_card)


def test_explicit_stream_never_takes_the_round_off_tolerance(
        tmp_path, small_card):
    def edit(m):
        m["restart_next_verification"] = {
            "result": {"all_equal": False, "within_tolerance": True},
            "tolerance": dict(RIGID_LID_RESTART_TOLERANCE)}
    path = _plant_dataset(tmp_path, small_card, "explicit",
                          chunk_edit=_every_chunk(edit))
    with pytest.raises(ValueError, match="verification"):
        SWGDatasetContract.from_json(path, card=small_card)


def test_contract_refuses_a_card_of_another_surface(
        tmp_path, small_card, lid_card):
    lid = _plant_dataset(tmp_path, lid_card, "lid",
                         chunk_edit=_every_chunk(_lid_stream))
    with pytest.raises(ValueError, match="differs from the card's surface"):
        SWGDatasetContract.from_json(lid, card=small_card)
    explicit = _plant_dataset(tmp_path, small_card, "explicit")
    with pytest.raises(ValueError, match="differs from the card's surface"):
        SWGDatasetContract.from_json(explicit, card=lid_card)


def test_contract_refuses_mixed_surfaces(tmp_path, lid_card):
    path = _plant_dataset(
        tmp_path, lid_card, "mixed",
        chunk_edit=lambda name: _lid_stream if name == "train_a" else None)
    with pytest.raises(ValueError, match="more than one surface"):
        SWGDatasetContract.from_json(path, card=lid_card)


@pytest.mark.parametrize("plant", sorted(_CONTRACT_PLANTS))
def test_contract_refuses_each_planted_violation(tmp_path, small_card, plant):
    kwargs, message = _CONTRACT_PLANTS[plant]
    path = _plant_dataset(tmp_path, small_card, "plant", **kwargs)
    with pytest.raises(ValueError, match=message):
        SWGDatasetContract.from_json(path, card=small_card)


@pytest.mark.parametrize("plant,message", [
    ("23 records in a day", "24 hourly records"),
    ("absolute hours shifted", "lineage"),
    ("day file edited after pinning", "digest differs"),
])
def test_day_file_plants_raise_when_the_day_is_read(
        tmp_path, small_card, plant, message):
    geometry = build_closure_geometry(small_card)
    masks = (np.asarray(geometry.umask), np.asarray(geometry.vmask))

    def day_edit(day, arrays):
        if day != 3:
            return
        if plant == "23 records in a day":
            for key in [key for key in arrays if key.endswith("_hourly_m")
                        or key.endswith("_hourly_m_s")]:
                arrays[key] = arrays[key][:23]
        elif plant == "absolute hours shifted":
            arrays["absolute_hour"] = arrays["absolute_hour"] + 1

    root = tmp_path / "chunk"
    digest = _write_chunk(root, source_day=0, days=4, masks=masks,
                          truth_run=tmp_path / "truth", day_edit=day_edit)
    if plant == "day file edited after pinning":
        with np.load(root / "day0003.npz") as saved:
            arrays = dict(saved)
        arrays["filtered_u_hourly_m_s"] = arrays["filtered_u_hourly_m_s"] * 2.0
        np.savez(root / "day0003.npz", **arrays)
    swg_online._load_pinned_day.cache_clear()
    with pytest.raises(ValueError, match=message):
        load_content_pinned_hourly_segment(
            root, expected_manifest_sha256=digest, operator="template")


def test_batch_membership_is_bound_to_the_issued_arrays(dataset):
    contract = dataset["contract"]
    train = next(contract.batches(
        split="train", window_days=1, epoch=0, seed=3, target_interval_hours=1))
    validation = next(contract.batches(split="validation", window_days=1, target_interval_hours=1))
    contract.validate_batch_membership(train)
    contract.validate_batch_membership(validation)
    plants = {
        "validation arrays under train labels": validation._replace(
            split="train", start_absolute_hour=train.start_absolute_hour),
        "validation batch relabelled train": validation._replace(
            split="train"),
        "one edited target word": train._replace(target=(
            train.target[0] + 1.0e-9,) + tuple(train.target[1:])),
        "same values in float32": train._replace(target=(
            train.target[0].astype(np.float32),) + tuple(train.target[1:])),
        "start hours as floats": train._replace(
            start_absolute_hour=train.start_absolute_hour.astype(np.float64)),
        "window length as a float": train._replace(window_days=1.0),
        "hand-built batch": WindowBatch(
            train.initial, train.target, train.start_absolute_hour,
            "train", 2),
    }
    for batch in plants.values():
        with pytest.raises(ValueError):
            contract.validate_batch_membership(batch)
    other = SWGDatasetContract.from_json(dataset["path"], card=dataset["card"])
    with pytest.raises(ValueError, match="not issued"):
        other.validate_batch_membership(train)
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.splits = ()
    with pytest.raises(ValueError, match="gap"):
        dataclasses.replace(contract, splits=(
            ("train", 0, 528), ("validation", 600, 1488),
            ("test", 2208, 2448)))
    with pytest.raises(ValueError, match="contiguous"):
        dataclasses.replace(contract, sources=contract.sources[:1])


def test_day_files_are_hashed_once_while_cached(dataset, monkeypatch):
    contract = dataset["contract"]
    swg_online._load_pinned_day.cache_clear()
    hashed = []
    original = swg_online._sha256

    def counting(path):
        hashed.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(swg_online, "_sha256", counting)
    next(contract.batches(split="train", window_days=1, epoch=0, seed=5, target_interval_hours=1))
    first = len(hashed)
    next(contract.batches(split="train", window_days=1, epoch=0, seed=5, target_interval_hours=1))
    assert first > 0
    assert len(hashed) == first


_REAL_EIGHT_DAY = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round4/"
    "hourly_targets_8d/start_day0731")


@pytest.mark.skipif(not (_REAL_EIGHT_DAY / "manifest.json").is_file(),
                    reason="complete eight-day cache is not mounted")
def test_real_v1_eight_day_loader_is_content_pinned_and_labelled_box_only():
    manifest = _REAL_EIGHT_DAY / "manifest.json"
    record = load_content_pinned_hourly_segment(
        _REAL_EIGHT_DAY,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        operator="symmetric")
    assert record.ssh_m.shape[0] == 192
    assert not record.includes_initial_state
    with pytest.raises(ValueError, match="only the box operator"):
        load_content_pinned_hourly_segment(
            _REAL_EIGHT_DAY,
            expected_manifest_sha256=hashlib.sha256(
                manifest.read_bytes()).hexdigest(), operator="template")


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
        build_segment_fn(card, geometry, window_days=16, target_interval_hours=1)

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


def test_network_gradient_gate_rejects_one_poisoned_component():
    assert_network_gradients_live({"live": jnp.asarray([1.0, 2.0])})
    with pytest.raises(ValueError, match="non-finite elements"):
        assert_network_gradients_live({"partial": jnp.asarray([1.0, jnp.nan])})
    with pytest.raises(ValueError, match="identically-zero gradient"):
        assert_network_gradients_live({"dead": jnp.asarray([0.0, 0.0])})


def test_zero_closure_four_and_eight_day_compositions_match(online_case):
    card, geometry, _, _, _ = online_case
    _, four_day = build_segment_fn(card, geometry, window_days=4, target_interval_hours=1)
    _, eight_day = build_segment_fn(card, geometry, window_days=8, target_interval_hours=1)
    steps_per_day = 24 * int(3_600 / card.dt_s)
    noise4 = jnp.zeros((4 * steps_per_day, 3) + geometry.tmask.shape)
    noise8 = jnp.zeros((8 * steps_per_day, 3) + geometry.tmask.shape)
    key = jax.random.PRNGKey(922)
    _, hourly4 = four_day(
        None, card.recipe.initial_state, noise4, key, jnp.asarray(0.0))
    _, hourly8 = eight_day(
        None, card.recipe.initial_state, noise8, key, jnp.asarray(0.0))
    for short, long in zip(hourly4, hourly8, strict=True):
        assert np.array_equal(np.asarray(short), np.asarray(long[:96]))


_ROUND2 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2")
_REAL_TARGET = _ROUND2 / "hourly_targets/start_day0731"
_REAL_T0 = _ROUND2 / "conservative_truth_dg50_1095d/snapshots"


@pytest.mark.skipif(not (_REAL_T0 / "day0731.npz").is_file(),
                    reason="Round-2 conservative truth is not mounted")
def test_shifted_network_matches_previous_equinox_forward_and_gradient_on_real_state():
    """Pin the 7,696-parameter replacement against the former CPU operator."""
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    initial = load_content_pinned_initial_state(
        card, _REAL_T0 / "day0731.npz",
        manifest_path=_REAL_T0 / "manifest.json",
        expected_manifest_sha256=(
            "aaec7f829d7530a275c07a263cd96185a4fa6245be411c63626156e35fa65d2c"))
    tau = swg_surface_wind(SWG_DG50_NAMELIST)
    entry = _entry_from_state(
        initial, tau, tau, card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config)
    spec = SWGNetworkSpec(
        hidden_channels=16, hidden_layers=4, kernel_size=3,
        ssh_scale_m=1.0, velocity_scale_m_s=1.0,
        potential_scale_m3_s2=1.0, wall_margin_rings=0)
    key = jax.random.PRNGKey(20261013)
    shifted = SWGScalarPotentialCNN(spec, key=key)
    keys = jax.random.split(key, spec.hidden_layers + 1)
    legacy_layers = [eqx.nn.Conv2d(
        4, spec.hidden_channels, spec.kernel_size, padding=1,
        key=keys[0], dtype=jnp.float64)]
    for layer_key in keys[1:-1]:
        legacy_layers.append(eqx.nn.Conv2d(
            spec.hidden_channels, spec.hidden_channels, spec.kernel_size,
            padding=1, key=layer_key, dtype=jnp.float64))
    legacy_layers.append(eqx.nn.Conv2d(
        spec.hidden_channels, 1, spec.kernel_size, padding=1,
        use_bias=False, key=keys[-1], dtype=jnp.float64))
    legacy = eqx.tree_at(
        lambda model: model.layers, shifted, tuple(legacy_layers))
    assert network_parameter_count(shifted) == 7_696
    for new_layer, old_layer in zip(
            shifted.layers, legacy.layers, strict=True):
        assert np.array_equal(new_layer.weight, old_layer.weight)
        if new_layer.bias is not None:
            assert np.array_equal(new_layer.bias, old_layer.bias)

    zero = jnp.zeros_like(entry.ssh_k)
    def call(model):
        return model(
            entry, geometry, key=jax.random.PRNGKey(4), noise=zero,
            noise_scale=jnp.asarray(0.0))
    new_value = call(shifted)
    old_value = call(legacy)
    np.testing.assert_allclose(
        np.asarray(new_value), np.asarray(old_value), rtol=2.0e-14,
        atol=5.0e-15)
    lowered = eqx.filter_jit(call).lower(shifted)
    assert_no_convolution_operations(lowered, label="shifted real network")
    with pytest.raises(RuntimeError, match="forbidden convolution"):
        assert_no_convolution_operations(
            eqx.filter_jit(call).lower(legacy), label="former Equinox network")

    forcing = OceanSurfaceForcing(
        tau_i_native=jnp.asarray(tau[0]), tau_j_native=jnp.asarray(tau[1]))
    stage_noise = jnp.zeros((3,) + tuple(geometry.tmask.shape), jnp.float64)
    step = eqx.filter_jit(lambda model: swg_online_step(
        model, initial, stage_noise, jax.random.PRNGKey(6), jnp.asarray(0.0),
        card=card, geometry=geometry, forcing=forcing))
    assert_no_convolution_operations(
        step.lower(shifted), label="real closure model step")

    def loss(model):
        return jnp.sum(call(model) ** 2)
    new_grad = eqx.filter_grad(loss)(shifted)
    old_grad = eqx.filter_grad(loss)(legacy)
    new_leaves = jax.tree.leaves(eqx.filter(new_grad, eqx.is_inexact_array))
    old_leaves = jax.tree.leaves(eqx.filter(old_grad, eqx.is_inexact_array))
    assert len(new_leaves) == len(old_leaves)
    for new_leaf, old_leaf in zip(new_leaves, old_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(new_leaf), np.asarray(old_leaf), rtol=2.0e-12,
            atol=2.0e-16)


@pytest.mark.skipif(not (_REAL_T0 / "day1091.npz").is_file(),
                    reason="ten Round-2 conservative starts are not mounted")
def test_shifted_network_preserves_ten_state_initial_acceleration():
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    states = tuple(load_content_pinned_initial_state(
        card, _REAL_T0 / f"day{day:04d}.npz",
        manifest_path=_REAL_T0 / "manifest.json",
        expected_manifest_sha256=(
            "aaec7f829d7530a275c07a263cd96185a4fa6245be411c63626156e35fa65d2c"))
        for day in (731, 771, 811, 851, 891, 931, 971, 1011, 1051, 1091))
    spec = SWGNetworkSpec(16, 4, 3, 1.0, 1.0, 1.0, 0)
    network = SWGScalarPotentialCNN(
        spec, key=jax.random.PRNGKey(20261013))
    result = initial_closure_acceleration_ratio(
        network, states, card, geometry, key=jax.random.PRNGKey(20261013))
    assert result["maximum_ratio"] == pytest.approx(
        0.007791671811074865, rel=1.0e-13)
    assert result["passes"]
    planted = eqx.tree_at(
        lambda model: model.layers[-1].weight, network,
        network.layers[-1].weight * 1.01)
    planted_result = initial_closure_acceleration_ratio(
        planted, states, card, geometry, key=jax.random.PRNGKey(20261013))
    assert planted_result["maximum_ratio"] == pytest.approx(
        1.01 * result["maximum_ratio"], rel=2.0e-13)
    assert planted_result["maximum_ratio"] != pytest.approx(
        result["maximum_ratio"], rel=1.0e-4)


@pytest.mark.skipif(not (_REAL_TARGET / "manifest.json").is_file(),
                    reason="Round-2 hourly targets are not mounted")
def test_real_data_one_optimizer_step_has_no_inert_leaf():
    """Exercise the actual t0 -> rollout -> loss -> AdamW update seam."""
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    stored = load_content_pinned_initial_state(
        card, _REAL_T0 / "day0731.npz",
        manifest_path=_REAL_T0 / "manifest.json",
        expected_manifest_sha256=(
            "aaec7f829d7530a275c07a263cd96185a4fa6245be411c63626156e35fa65d2c"))
    u0 = np.asarray(stored.u.data[:, 1:, 0])
    v0 = np.asarray(stored.v.data[1:, :, 0])
    balanced, _ = build_start_rebalancer(
        card, tolerance=START_REBALANCE_TOLERANCE,
        max_iterations=START_REBALANCE_MAX_ITERATIONS)(
            np.asarray(stored.eta.data), u0, v0)
    initial = coarse_state_from_native(stored, balanced, u0, v0)
    native = load_content_pinned_hourly_segment(
        _REAL_TARGET,
        expected_manifest_sha256=(
            "f5526f50c9061abe498f0088b45b8d64f1895d46c2c1958e50b032acef721f99"),
        operator="symmetric")
    target = tuple(jnp.asarray(value[:24])[None] for value in (
        native.ssh_m, native.u_m_s, native.v_m_s))
    spec = SWGNetworkSpec(
        hidden_channels=16, hidden_layers=4, kernel_size=3,
        ssh_scale_m=1.0, velocity_scale_m_s=1.0,
        potential_scale_m3_s2=1.0, wall_margin_rings=0)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(20261013))
    initialization = initial_closure_acceleration_ratio(
        network, (initial,), card, geometry, key=jax.random.PRNGKey(20261013))
    assert initialization["passes"], initialization
    _, segment = build_segment_fn(card, geometry, window_days=1, target_interval_hours=1)
    n_steps = 24 * int(3_600 / card.dt_s)
    noise = jnp.zeros((n_steps, 3) + geometry.tmask.shape, dtype=jnp.float64)

    def objective(net):
        _, predicted = segment(
            net, initial, noise, jax.random.PRNGKey(41), jnp.asarray(0.0))
        return deterministic_trajectory_loss(
            tuple(value[None] for value in predicted), target, geometry,
            objective="symmetric", **_eq7(card))

    loss, grads = eqx.filter_value_and_grad(objective)(network)
    jax.block_until_ready(loss)
    assert np.isfinite(float(loss)) and float(loss) > 0.0
    assert_network_gradients_live(grads)
    # Keras 3.6.0 AdamW defaults: lr 1e-3, weight decay 0.004, betas 0.9 and
    # 0.999, epsilon 1e-7 as "epsilon hat", no clipping, constant rate.
    optimizer = create_swg_optimizer(_TEMPLATE_ADAMW)
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(grads, opt_state, params)
    updated = eqx.apply_updates(network, updates)
    before = jax.tree.leaves(params)
    after = jax.tree.leaves(eqx.filter(updated, eqx.is_inexact_array))
    assert all(np.all(np.isfinite(np.asarray(value))) for value in after)
    assert any(np.any(np.asarray(left) != np.asarray(right))
               for left, right in zip(before, after, strict=True))


# --- strict optimiser wrapper ----------------------------------------------

_TEMPLATE_ADAMW = SWGOptimizerSettings(
    optimizer="adamw", learning_rate=1.0e-3, schedule="constant",
    weight_decay=4.0e-3, adam_b1=0.9, adam_b2=0.999, adam_epsilon=1.0e-7,
    adam_epsilon_convention="keras_epsilon_hat", gradient_clipping="none")


def test_swg_optimizer_refuses_unset_and_shared_default_choices():
    fields = _TEMPLATE_ADAMW._asdict()
    for name in SWGOptimizerSettings._fields:
        with pytest.raises(TypeError, match=name):
            SWGOptimizerSettings(**{key: value for key, value in fields.items()
                                    if key != name})
    with pytest.raises(TypeError, match="SWGOptimizerSettings"):
        create_swg_optimizer(TrainingConfig(
            lr=1.0e-3, schedule="constant", clip_gradients=False))
    for name, value, match in (
            ("optimizer", "adam", "adamw"),
            ("schedule", "warmup_cosine", "constant"),
            ("gradient_clipping", "global_norm", "none"),
            ("adam_epsilon_convention", "keras", "convention"),
            ("learning_rate", 1, "finite float"),
            ("adam_epsilon", float("nan"), "finite float"),
            ("learning_rate", 0.0, "positive rate")):
        with pytest.raises(ValueError, match=match):
            create_swg_optimizer(_TEMPLATE_ADAMW._replace(**{name: value}))


def test_swg_optimizer_is_the_explicit_template_update():
    """Constant rate, no clipping, Keras epsilon: no factory default leaks."""
    rng = np.random.default_rng(5)
    params = {"w": jnp.asarray(rng.normal(size=6))}
    # Gradient norms from 1e-7 to 1e3: a warm-up, a decay, clipping or the
    # Optax epsilon would each change at least one of the six steps.
    gradients = [{"w": jnp.asarray(rng.normal(size=6) * scale)}
                 for scale in (1.0e3, 1.0e-7, 30.0, 2.0e-6, 5.0, 1.0e2)]

    def trajectory(optimizer):
        state, current, out = optimizer.init(params), params, []
        for grads in gradients:
            update, state = optimizer.update(grads, state, current)
            current = optax.apply_updates(current, update)
            out.append(np.asarray(current["w"]))
        return out

    explicit = create_optimizer(TrainingConfig(
        lr=1.0e-3, warmup_steps=0, weight_decay=4.0e-3, optimizer="adamw",
        schedule="constant", clip_gradients=False, adam_epsilon=1.0e-7,
        adam_b1=0.9, adam_b2=0.999,
        adam_epsilon_convention="keras_epsilon_hat"))
    for left, right in zip(trajectory(create_swg_optimizer(_TEMPLATE_ADAMW)),
                           trajectory(explicit), strict=True):
        assert np.array_equal(left, right)
    # Step one by hand: |g| equal to Keras' epsilon-hat is half the rate.
    optimizer = create_swg_optimizer(_TEMPLATE_ADAMW)
    w = {"w": jnp.asarray([2.0])}
    update, _ = optimizer.update(
        {"w": jnp.asarray([1.0e-7 / np.sqrt(1.0 - 0.999)])},
        optimizer.init(w), w)
    assert float(update["w"][0]) == pytest.approx(
        -1.0e-3 * (0.5 + 4.0e-3 * 2.0), rel=1.0e-12)


# --- Maddison Eq. 6 network scales -----------------------------------------

def test_eq6_network_scales_follow_the_deck_and_move_with_it(online_case):
    card, geometry = online_case[:2]
    scales = maddison_eq6_network_scales(card, geometry)
    constants = card.recipe.model_config.constants
    assert (constants.rho_0, constants.g) == (1026.0, 9.80665)
    beta, half, dx, f0, tau0, depth = 2.0e-11, 2.0e6, 5.0e4, 7.0e-5, 0.1, 500.0
    expected = {
        "vorticity_scale_s1": beta * half,
        "vorticity_tendency_scale_s2":
            tau0 * np.pi / (constants.rho_0 * depth * half),
        "grid_spacing_m": dx, "depth_m": depth,
        "velocity_scale_m_s": beta * half * dx,
        "ssh_scale_m": f0 * beta * half * dx**2 / constants.g,
        "potential_scale_m3_s2":
            tau0 * np.pi * dx**2 / (constants.rho_0 * half),
    }
    for name, value in expected.items():
        assert getattr(scales, name) == pytest.approx(value, rel=1.0e-14), name
    moved = maddison_eq6_network_scales(card._replace(
        resolved_namelist=card.resolved_namelist._replace(
            rn_tau_pa=0.2, rn_beta_m_s=4.0e-11, rn_f0_s=1.4e-4)), geometry)
    assert moved.potential_scale_m3_s2 == pytest.approx(
        2.0 * scales.potential_scale_m3_s2, rel=1.0e-14)
    assert moved.velocity_scale_m_s == pytest.approx(
        2.0 * scales.velocity_scale_m_s, rel=1.0e-14)
    assert moved.ssh_scale_m == pytest.approx(
        4.0 * scales.ssh_scale_m, rel=1.0e-14)
    with pytest.raises(ValueError, match="2L"):
        maddison_eq6_network_scales(card._replace(
            resolved_namelist=card.resolved_namelist._replace(
                rn_domsiz_m=2.0e6)), geometry)
    thickness = np.array(card.recipe.z_coord.nemo_e3t_0)
    j0, _, i0, _ = geometry.wet_bounds
    thickness[j0, i0] = 400.0
    with pytest.raises(ValueError, match="one positive resting depth"):
        maddison_eq6_network_scales(card._replace(recipe=card.recipe._replace(
            z_coord=card.recipe.z_coord._replace(
                nemo_e3t_0=jnp.asarray(thickness)))), geometry)


def test_eq6_scale_report_measures_what_the_network_reads(online_case):
    card, geometry = online_case[:2]
    scales = maddison_eq6_network_scales(card, geometry)
    spec = SWGNetworkSpec(8, 2, 3, scales.ssh_scale_m,
                          scales.velocity_scale_m_s,
                          scales.potential_scale_m3_s2, 0)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(3))
    u, v = _streamfunction_flow(geometry, 3.0e5)
    wet = np.asarray(geometry.tmask, dtype=bool)
    state = coarse_state_from_native(
        card.recipe.initial_state, 0.3 * wet, u, v)
    report = eq6_scale_report(network, (state,), card, geometry, scales,
                              key=jax.random.PRNGKey(4))
    assert report["ssh"]["max_abs"] == pytest.approx(
        0.3 / scales.ssh_scale_m, rel=1.0e-14)
    u_t = 0.5 * (u + np.roll(u, 1, axis=1))
    assert report["u_t"]["max_abs"] == pytest.approx(
        np.abs(u_t[wet]).max() / scales.velocity_scale_m_s, rel=1.0e-12)
    # The template's input on the same state: the F-point curl of a
    # wall-zero streamfunction flow is its five-point Laplacian / dx**2.
    psi = np.cumsum(-u * geometry.spacing_m, axis=0) \
        * np.asarray(geometry.vertex_interior)
    lap = (np.roll(psi, 1, 0) + np.roll(psi, -1, 0) + np.roll(psi, 1, 1)
           + np.roll(psi, -1, 1) - 4.0 * psi) / geometry.spacing_m**2
    interior = np.asarray(geometry.vertex_interior, dtype=bool)
    assert report["template_input_zeta"]["max_abs"] == pytest.approx(
        np.abs(lap[interior]).max() / scales.vorticity_scale_s1, rel=1.0e-9)
    tau = swg_surface_wind(card.resolved_namelist)
    entry = _entry_from_state(state, tau, tau, card.recipe.grid,
                              card.recipe.z_coord, card.recipe.model_config)
    potential = np.asarray(network(
        entry, geometry, key=jax.random.fold_in(jax.random.PRNGKey(4), 0),
        noise=jnp.zeros_like(entry.ssh_k), noise_scale=jnp.asarray(0.0)))
    assert report["output_potential"]["max_abs"] == pytest.approx(
        np.abs(potential[interior]).max() / scales.potential_scale_m3_s2,
        rel=1.0e-12)
    assert 0.0 < report["output_laplacian"]["max_abs"] < np.inf
    with pytest.raises(ValueError, match="Eq. 6"):
        eq6_scale_report(SWGScalarPotentialCNN(
            spec._replace(potential_scale_m3_s2=1.0),
            key=jax.random.PRNGKey(3)),
            (state,), card, geometry, scales, key=jax.random.PRNGKey(4))


# ----------------------------------------------------- target interval (3 h)

def test_target_interval_batches_are_every_third_stored_hour_from_the_start(
        dataset):
    contract = dataset["contract"]
    for split, extra in (("train", {"epoch": 0, "seed": 12}), ("validation", {})):
        hourly = next(contract.batches(
            split=split, window_days=1, target_interval_hours=1, **extra))
        sparse = next(contract.batches(
            split=split, window_days=1, target_interval_hours=3, **extra))
        assert hourly.target[0].shape[1] == 24 and sparse.target[0].shape[1] == 8
        assert np.array_equal(hourly.start_absolute_hour,
                              sparse.start_absolute_hour)
        for new, old in zip(sparse.initial, hourly.initial, strict=True):
            assert np.array_equal(new, old)
        # Hours 3, 6, ..., 24 after the start: indices 2, 5, ... of 1..24.
        for new, old in zip(sparse.target, hourly.target, strict=True):
            assert np.array_equal(new, old[:, 2::3])
        contract.validate_batch_membership(sparse)


@pytest.mark.parametrize("interval", [0, 5, 48, -1, 2.0, "3", True, None])
def test_target_interval_must_be_an_integer_dividing_a_day(dataset, interval):
    with pytest.raises(ValueError, match="target_interval_hours"):
        require_target_interval_hours(interval)
    with pytest.raises(ValueError, match="target_interval_hours"):
        next(dataset["contract"].batches(
            split="validation", window_days=1, target_interval_hours=interval))
    with pytest.raises(ValueError, match="target_interval_hours"):
        build_segment_fn(dataset["card"], build_closure_geometry(
            dataset["card"]), window_days=1, target_interval_hours=interval)


def test_target_interval_must_be_a_whole_number_of_model_steps():
    assert steps_per_target_interval(400.0, 1) == 9
    assert steps_per_target_interval(10_800.0, 3) == 1
    assert steps_per_target_interval(3_600.0, 3) == 3
    for dt_s, interval in ((10_800.0, 1), (4_000.0, 3), (7_200.0, 1)):
        with pytest.raises(ValueError, match="must divide"):
            steps_per_target_interval(dt_s, interval)
    card = types.SimpleNamespace(dt_s=10_800.0)
    with pytest.raises(ValueError, match="must divide"):
        build_segment_fn(card, None, window_days=1, target_interval_hours=1)


def test_three_hour_rollout_records_exactly_the_three_hourly_states(dataset):
    card = dataset["card"]
    geometry = build_closure_geometry(card)
    _, hourly = build_segment_fn(
        card, geometry, window_days=1, target_interval_hours=1)
    _, sparse = build_segment_fn(
        card, geometry, window_days=1, target_interval_hours=3)
    key, scale = jax.random.PRNGKey(5), jnp.asarray(0.0)
    steps = 24 * int(3_600 / card.dt_s)
    start = card.recipe.initial_state
    # A moving start so the records are not identically the rest state.
    start = start._replace(u=start.u.replace(
        data=start.u.data.at[10:14, 10:14].set(1.0e-2)))
    noise = jnp.zeros((steps, 3) + tuple(geometry.tmask.shape))
    state_h, records_h = hourly(None, start, noise, key, scale)
    state_3, records_3 = sparse(None, start, noise, key, scale)
    assert records_h[1].shape[0] == 24 and records_3[1].shape[0] == 8
    assert np.any(np.asarray(records_h[1][0]) != np.asarray(records_h[1][-1]))
    for new, old in zip(records_3, records_h, strict=True):
        assert np.array_equal(np.asarray(new), np.asarray(old)[2::3])
    assert np.array_equal(np.asarray(state_3.u.data), np.asarray(state_h.u.data))
    with pytest.raises(ValueError, match="segment noise shape"):
        sparse(None, start, noise[:-1], key, scale)


@pytest.mark.parametrize("days,bound", [(1, 0.13), (2, 0.06), (4, 0.03), (8, 0.012)])
def test_three_hour_loss_is_the_same_plain_mean_over_a_subsample(
        online_case, days, bound):
    """Eq. 7 here is ``prefactor * mean over (windows, hours)`` of the weighted
    squared vorticity error: equal weights, no end-point half weights, so the
    3-hour loss is the same mean over every third sample and only approximates
    the hourly loss (exactly equal only for an error constant in time).  The
    samples at hours 3, 6, ... sit one hour later on average than hours 1, 2,
    ..., so an error growing in time reads high by about 2 h / window; the
    bounds are the measured 11.4/5.3/2.3/0.9 % for an error growing as
    ``1 - exp(-t/72 h)`` squared, rounded up."""
    card, geometry, _, _, _ = online_case
    shape = tuple(geometry.tmask.shape)
    pattern = np.zeros(shape)
    pattern[20:30, 20:30] = 1.0
    hours = np.arange(1, 24 * days + 1)
    amplitude = 1.0e-6 * (1.0 - np.exp(-hours / 72.0))   # smooth error growth
    zeta_error = amplitude[None, :, None, None] * pattern[None, None]

    def loss(error):
        zeros = jnp.zeros(error.shape)
        return float(balanced_trajectory_loss(
            BalancedTrajectory(jnp.asarray(error), zeros, zeros),
            BalancedTrajectory(zeros, zeros, zeros), geometry, None,
            kind="point", **_eq7(card)))

    hourly = loss(zeta_error)
    sparse = loss(zeta_error[:, 2::3])
    # The code's rule, restated by hand: scale * mean_t sum_x W e^2.
    namelist = card.resolved_namelist
    half = 0.5 * namelist.rn_domsiz_m
    scale = 1.0e4 / (namelist.rn_beta_m_s**2 * half**2) / (4.0 * half**2)
    weights = np.asarray(eq7_point_weights(geometry, wall_ring="weighted"))
    by_hand = scale * np.mean(
        np.sum(zeta_error[:, 2::3]**2 * weights, axis=(-2, -1)))
    assert sparse == pytest.approx(by_hand, rel=1.0e-12)
    # Equal weights: one hour's error moves the mean by exactly 1/N of its term.
    bumped = zeta_error.copy()
    bumped[:, 5] *= 2.0
    term = scale * np.sum(zeta_error[:, 5]**2 * weights) * 3.0
    assert loss(bumped) - hourly == pytest.approx(term / hours.size, rel=1.0e-9)
    # Smooth trajectory: the subsampled mean tracks the hourly one.
    assert sparse != hourly
    assert 0.0 < sparse / hourly - 1.0 < bound
    # A time-constant error is the case where the two are exactly equal.
    constant = np.broadcast_to(zeta_error[:, -1:], zeta_error.shape).copy()
    assert loss(constant[:, 2::3]) == pytest.approx(loss(constant), rel=1.0e-12)

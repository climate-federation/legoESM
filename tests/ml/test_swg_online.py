"""Direct tests for the SWG online-learning network, loss, and segment seam."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ml.training import TrainingConfig
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
    SWGDatasetContract,
    SWGNetworkSpec,
    SWGScalarPotentialCNN,
    assert_network_gradients_live,
    balanced_normalizers_from_training,
    balanced_trajectory_loss,
    build_closure_geometry,
    build_segment_fn,
    create_swg_optimizer,
    deterministic_trajectory_loss,
    diagnose_native_trajectory,
    initial_closure_acceleration_ratio,
    load_content_pinned_hourly_segment,
    load_content_pinned_initial_state,
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


def test_point_and_energy_losses_share_interface_and_plants_fail(online_case):
    _, geometry, _, _, _ = online_case
    shape = (4, 6) + tuple(geometry.tmask.shape)
    base = jnp.zeros(shape)
    target_shape = shape[1:]
    target_field = jnp.zeros(target_shape).at[:, 20:30, 20:30].set(1.0)
    predicted = BalancedTrajectory(base, base, base)
    target = BalancedTrajectory(target_field, target_field, target_field)
    point_target = jax.tree.map(
        lambda value: jnp.broadcast_to(value, base.shape), target)
    normalizers = BalancedNormalizers(1.0, 1.0, 1.0)
    point = balanced_trajectory_loss(
        predicted, point_target, geometry, normalizers, kind="point",
        beta_m1_s1=2.0e-11, basin_length_m=4.0e6)
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


def test_maddison_point_loss_scaling_and_default_ignores_velocity(online_case):
    _, geometry, _, _, _ = online_case
    shape = (2,) + tuple(geometry.tmask.shape)
    zeta = jnp.zeros(shape).at[:, 20, 20].set(3.0e-7)
    zero = jnp.zeros(shape)
    velocity_plant = jnp.ones(shape) * 1.0e3
    target = BalancedTrajectory(zero[None], zero[None], zero[None])
    normalizers = BalancedNormalizers(1.0, 1.0, 1.0)
    beta, length = 2.0e-11, 4.0e6
    baseline = balanced_trajectory_loss(
        BalancedTrajectory(zeta[None], zero[None], zero[None]), target,
        geometry, normalizers, kind="point", beta_m1_s1=beta,
        basin_length_m=length)
    velocity_changed = balanced_trajectory_loss(
        BalancedTrajectory(zeta[None], velocity_plant[None],
                           velocity_plant[None]), target,
        geometry, normalizers, kind="point", beta_m1_s1=beta,
        basin_length_m=length)
    expected = (1.0e4 / (beta**2 * length**2) / (4.0 * length**2)
                * float(geometry.area_t_m2[20, 20]) * (3.0e-7)**2)
    assert float(baseline) == pytest.approx(expected, rel=2.0e-15)
    assert float(velocity_changed) == float(baseline)


def test_named_objectives_are_zero_at_t0_and_see_checkerboard_differently(
        online_case):
    _, geometry, _, _, _ = online_case
    shape = tuple(geometry.tmask.shape)
    jj, ii = np.indices(shape)
    checkerboard = jnp.asarray(0.1 * np.where((ii + jj) % 2, -1.0, 1.0))
    zero = jnp.zeros(shape, dtype=jnp.float64)
    target = tuple(value[None, None] for value in (zero, zero, zero))
    model = tuple(value[None, None] for value in (zero, checkerboard, zero))
    normalizers = BalancedNormalizers(1.0, 1.0, 1.0)
    losses = {}
    for objective in ("symmetric", "template"):
        identical = deterministic_trajectory_loss(
            target, target, geometry, objective=objective,
            beta_m1_s1=2.0e-11, basin_length_m=4.0e6,
            normalizers=normalizers)
        assert float(identical) == 0.0
        losses[objective] = float(deterministic_trajectory_loss(
            model, target, geometry, objective=objective,
            beta_m1_s1=2.0e-11, basin_length_m=4.0e6,
            normalizers=normalizers))
    assert losses["template"] > 100.0 * losses["symmetric"] > 0.0
    with pytest.raises(ValueError, match="explicitly selected"):
        deterministic_trajectory_loss(
            target, target, geometry, objective="typo",
            beta_m1_s1=2.0e-11, basin_length_m=4.0e6,
            normalizers=normalizers)


def test_point_loss_owns_batch_reduction(online_case):
    _, geometry, _, _, _ = online_case
    shape = (1, 2) + tuple(geometry.tmask.shape)
    error = jnp.zeros(shape).at[:, :, 20, 20].set(2.0e-7)
    zero = jnp.zeros(shape)
    normalizers = BalancedNormalizers(1.0, 1.0, 1.0)
    one = balanced_trajectory_loss(
        BalancedTrajectory(error, zero, zero),
        BalancedTrajectory(zero, zero, zero), geometry, normalizers,
        kind="point", beta_m1_s1=2.0e-11, basin_length_m=4.0e6)
    two_error = jnp.concatenate((error, jnp.zeros_like(error)), axis=0)
    two_zero = jnp.zeros_like(two_error)
    mixed = balanced_trajectory_loss(
        BalancedTrajectory(two_error, two_zero, two_zero),
        BalancedTrajectory(two_zero, two_zero, two_zero), geometry,
        normalizers, kind="point", beta_m1_s1=2.0e-11,
        basin_length_m=4.0e6)
    assert float(mixed) == pytest.approx(0.5 * float(one), rel=2.0e-15)


def _write_v2_hourly(root: Path, *, days: int, source_day: int = 0) -> str:
    root.mkdir()
    hashes = {}
    shape = (4, 4)
    for day in range(1, days + 1):
        first = (source_day + day - 1) * 24 + 1
        hours = np.arange(first, first + 24, dtype=np.int64)
        values = np.broadcast_to(hours[:, None, None], (24,) + shape).astype(float)
        arrays = {
            "absolute_hour": hours,
            "box_ssh_hourly_m": values,
            "box_u_hourly_m_s": values + 1.0,
            "box_v_hourly_m_s": values + 2.0,
            "filtered_ssh_hourly_m": values * 2.0,
            "filtered_u_hourly_m_s": values * 2.0 + 1.0,
            "filtered_v_hourly_m_s": values * 2.0 + 2.0,
        }
        if day == 1:
            for prefix, scale in (("box_", 1.0), ("filtered_", 2.0)):
                value = float(source_day * 24)
                arrays[f"{prefix}ssh_initial_m"] = np.full(shape, scale * value)
                arrays[f"{prefix}u_initial_m_s"] = np.full(
                    shape, scale * value + 1.0)
                arrays[f"{prefix}v_initial_m_s"] = np.full(
                    shape, scale * value + 2.0)
        path = root / f"day{day:04d}.npz"
        np.savez(path, **arrays)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": "legoesm-swg-double-gyre-hourly-target-v2",
        "status": "complete", "days": days, "completed_days": days,
        "restart": {"source_day": source_day}, "daily_sha256": hashes,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_contract_builds_split_safe_seeded_batches_and_leakage_fails(
        tmp_path):
    sources = []
    for label, source_day in (("train", 0), ("validation", 43), ("test", 86)):
        source = tmp_path / label
        digest = _write_v2_hourly(source, days=13, source_day=source_day)
        sources.append({"root": str(source), "manifest_sha256": digest})
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({
        "schema": "legoesm-swg-online-dataset-v1",
        "objective": "template", "batch_size": 10,
        "window_days": [1, 2, 4, 8],
        "splits": {"train": [0, 311], "validation": [1032, 1343],
                   "test": [2064, 2375]},
        "sources": sources,
    }, sort_keys=True))
    contract = SWGDatasetContract.from_json(contract_path)
    first = next(contract.batches(
        split="train", window_days=1, epoch=0, seed=12))
    repeated = next(contract.batches(
        split="train", window_days=1, epoch=0, seed=12))
    assert np.array_equal(first.start_absolute_hour,
                          repeated.start_absolute_hour)
    assert first.initial[0].shape == (10, 4, 4)
    assert first.target[0].shape == (10, 24, 4, 4)
    # Each T0 is the stored state immediately preceding its right endpoints.
    assert np.array_equal(first.target[0][:, 0, 0, 0],
                          2 * (first.start_absolute_hour + 1))
    leaked = first._replace(
        start_absolute_hour=first.start_absolute_hour.at[0].set(1032)
        if hasattr(first.start_absolute_hour, "at") else
        np.concatenate(([1032], first.start_absolute_hour[1:])))
    with pytest.raises(ValueError, match="leaked or crossing"):
        contract.validate_batch_membership(leaked)
    crossing = first._replace(start_absolute_hour=np.full(10, 300))
    with pytest.raises(ValueError, match="leaked or crossing"):
        contract.validate_batch_membership(crossing)
    with pytest.raises(ValueError, match="training split only"):
        contract.normalizers_for_split("test", None)


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


def test_zero_closure_four_and_eight_day_compositions_match(online_case):
    card, geometry, _, _, _ = online_case
    _, four_day = build_segment_fn(card, geometry, window_days=4)
    _, eight_day = build_segment_fn(card, geometry, window_days=8)
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


@pytest.mark.skipif(not (_REAL_TARGET / "manifest.json").is_file(),
                    reason="Round-2 hourly targets are not mounted")
def test_real_data_one_optimizer_step_has_no_inert_leaf():
    """Exercise the actual t0 -> rollout -> loss -> AdamW update seam."""
    card = build_swg_card(SWG_DG50_NAMELIST)
    geometry = build_closure_geometry(card)
    initial = load_content_pinned_initial_state(
        card, _REAL_T0 / "day0731.npz",
        manifest_path=_REAL_T0 / "manifest.json",
        expected_manifest_sha256=(
            "aaec7f829d7530a275c07a263cd96185a4fa6245be411c63626156e35fa65d2c"))
    native = load_content_pinned_hourly_segment(
        _REAL_TARGET,
        expected_manifest_sha256=(
            "f5526f50c9061abe498f0088b45b8d64f1895d46c2c1958e50b032acef721f99"),
        operator="symmetric")
    native = tuple(jnp.asarray(value) for value in (
        native.ssh_m, native.u_m_s, native.v_m_s))
    target_all = diagnose_native_trajectory(
        native, geometry, objective="symmetric")
    normalizers = balanced_normalizers_from_training((target_all,), geometry)
    target = jax.tree.map(lambda value: value[:24], target_all)
    spec = SWGNetworkSpec(
        hidden_channels=16, hidden_layers=4, kernel_size=3,
        ssh_scale_m=1.0, velocity_scale_m_s=1.0,
        potential_scale_m3_s2=1.0)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(20261013))
    initialization = initial_closure_acceleration_ratio(
        network, (initial,), card, geometry, key=jax.random.PRNGKey(20261013))
    assert initialization["passes"], initialization
    _, segment = build_segment_fn(card, geometry, window_days=1)
    n_steps = 24 * int(3_600 / card.dt_s)
    noise = jnp.zeros((n_steps, 3) + geometry.tmask.shape, dtype=jnp.float64)

    def objective(net):
        _, predicted_native = segment(
            net, initial, noise, jax.random.PRNGKey(41), jnp.asarray(0.0))
        predicted = diagnose_native_trajectory(
            predicted_native, geometry, objective="symmetric")
        return balanced_trajectory_loss(
            jax.tree.map(lambda value: value[None], predicted),
            jax.tree.map(lambda value: value[None], target),
            geometry, normalizers, kind="point",
            beta_m1_s1=card.resolved_namelist.rn_beta_m_s,
            basin_length_m=card.resolved_namelist.rn_domsiz_m)

    loss, grads = eqx.filter_value_and_grad(objective)(network)
    jax.block_until_ready(loss)
    assert np.isfinite(float(loss)) and float(loss) > 0.0
    assert_network_gradients_live(grads)
    # Maddison/Keras AdamW: constant lr=1e-3, wd=.004, beta1=.9,
    # beta2=.999, eps=1e-7, with no separate clipping transform.
    optimizer = create_swg_optimizer(TrainingConfig(
        lr=1.0e-3, warmup_steps=0, total_steps=100,
        weight_decay=4.0e-3, optimizer="adamw", schedule="constant",
        clip_gradients=False, adam_epsilon=1.0e-7))
    params = eqx.filter(network, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(grads, opt_state, params)
    updated = eqx.apply_updates(network, updates)
    before = jax.tree.leaves(params)
    after = jax.tree.leaves(eqx.filter(updated, eqx.is_inexact_array))
    assert all(np.all(np.isfinite(np.asarray(value))) for value in after)
    assert any(np.any(np.asarray(left) != np.asarray(right))
               for left, right in zip(before, after, strict=True))

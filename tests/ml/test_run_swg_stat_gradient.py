"""Direct tests for the arm-E statistic-gradient tool (CPU, fp64, small)."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ml.training import save_checkpoint
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DG50_NAMELIST,
    SWG_DG50_RL_NAMELIST,
    build_swg_card,
)
from legoesm.training.swg_online import (
    SWGNetworkSpec,
    SWGOptimizerSettings,
    SWGScalarPotentialCNN,
    build_closure_geometry,
    build_segment_fn,
    create_swg_optimizer,
    maddison_eq6_network_scales,
    rigid_lid_start_state,
    start_state_from_native,
)

_ROOT = Path(__file__).parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sg = _load("run_swg_stat_gradient", _ROOT / "scripts/run/run_swg_stat_gradient.py")
evaluation = _load(
    "swg_online_evaluation_stat_gradient", _ROOT / "scripts/validate/"
    "ocean_fidelity/testcases/nemo_testcase_l1_swg_online_evaluation.py")
records = _load("swg_online_record_writer_stat_gradient",
                Path(__file__).with_name("test_swg_online.py"))
EXP4_SYMMETRIC = (_ROOT / "config/swg_online/"
                  "exp4_rigidlid_symmetric_ring_weighted_fp32.json")
DAYS = 2           # model days of the differentiated windows
FRAME_DAYS = 4     # daily frames of the statistic fixtures


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def lid():
    return build_swg_card(SWG_DG50_RL_NAMELIST, surface="rigid_lid")


@pytest.fixture(scope="module")
def geometry(lid):
    return build_closure_geometry(lid)


@pytest.fixture(scope="module")
def networks(lid, geometry):
    scales = maddison_eq6_network_scales(lid, geometry)
    spec = SWGNetworkSpec(
        hidden_channels=4, hidden_layers=2, kernel_size=3,
        ssh_scale_m=scales.ssh_scale_m,
        velocity_scale_m_s=scales.velocity_scale_m_s,
        potential_scale_m3_s2=scales.potential_scale_m3_s2,
        wall_margin_rings=0)
    return [SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(seed))
            for seed in (1, 2, 3)]


@pytest.fixture(scope="module")
def runs(lid, geometry, networks):
    """Daily native frames of a four-day lid run per network (the plain
    closure, through the training segment)."""
    raw, jitted = build_segment_fn(
        lid, geometry, window_days=4, target_interval_hours=24, checkpoint=True)
    noise = jnp.zeros((32, 3) + tuple(geometry.tmask.shape))
    start = rigid_lid_start_state(lid, lid.recipe.initial_state)
    key = jax.random.PRNGKey(0)
    out = []
    for network in networks:
        _, natives = jitted(network, start, noise, key, jnp.asarray(0.0))
        out.append(tuple(np.asarray(value) for value in natives))
    return out


def _model(networks, index=0, gamma=1.0):
    return sg.ScaledClosure(networks[index], jnp.asarray(gamma))


def _start(lid, frames):
    return start_state_from_native(lid, *(value[-1] for value in frames))


@pytest.fixture(scope="module")
def programs(lid, geometry):
    return {name: sg.build_programs(
        lid, geometry, days=DAYS, statistic=name, precision="fp64")
        for name in sg.STATISTICS}


# ----------------------------------------------------- the scorer's definitions

def test_transport_and_eke_maps_and_terms_are_the_scorers(lid, runs):
    candidate, reference = runs[0], runs[1]
    cand_stats = evaluation.window_statistics(
        candidate, namelist=SWG_DG50_RL_NAMELIST)
    ref_stats = evaluation.window_statistics(
        reference, namelist=SWG_DG50_RL_NAMELIST)
    score = evaluation.score_statistics(cand_stats, ref_stats)
    _, scorer_geometry = evaluation._dg50()
    constants = sg.statistic_geometry(SWG_DG50_RL_NAMELIST)
    for kind in "TUV":      # the lid deck's masks are the scorer's masks
        assert np.array_equal(np.asarray(constants.masks[kind]),
                              scorer_geometry["mask"][kind])
    stat = sg.Statistic(lid, "transport_eke", constants)
    cand_acc = sg.truth_accumulator(lid, "transport_eke", candidate)
    ref_acc = sg.truth_accumulator(lid, "transport_eke", reference)
    maps = stat.maps(cand_acc)
    wet = scorer_geometry["mask"]["T"]
    for kind, name in (("U", "transport_u"), ("V", "transport_v")):
        expected = cand_stats["mean_transport_m3_s"][kind]
        np.testing.assert_allclose(
            np.asarray(maps[name]), expected, rtol=1e-12,
            atol=1e-12 * np.abs(expected).max())
    np.testing.assert_allclose(
        np.asarray(maps["eke"])[wet], cand_stats["eke_map_m2_s2"][wet],
        rtol=1e-12)
    faces = np.concatenate([
        ref_stats["mean_transport_m3_s"][kind][scorer_geometry["mask"][kind]]
        for kind in ("U", "V")])
    terms = [float(t) for t in stat.terms(cand_acc, ref_acc)]
    assert terms[0] == pytest.approx(
        (score["mean_face_transport_rmse_m3_s"] / np.sqrt(np.mean(faces**2)))**2,
        rel=1e-12)
    assert terms[1] == pytest.approx(score["wet_eke_map_nrmse"]**2, rel=1e-12)
    assert float(stat.loss(cand_acc, ref_acc)) == pytest.approx(sum(terms),
                                                                rel=1e-14)
    assert min(terms) > 1e-6        # the comparison is not vacuous


def test_phi_mean_and_std_maps_are_per_frame_phi_over_g(lid, runs):
    candidate, reference = runs[0], runs[1]
    g = lid.recipe.model_config.constants.g

    def phi_stack(frames):
        return np.stack([np.asarray(start_state_from_native(
            lid, *(value[day] for value in frames)).rigid_lid_phi) / g
            for day in range(FRAME_DAYS)])

    stat = sg.Statistic(lid, "phi_mean_std",
                        sg.statistic_geometry(SWG_DG50_RL_NAMELIST))
    wet = np.asarray(lid.recipe.land_mask, dtype=bool)
    cand, ref = phi_stack(candidate), phi_stack(reference)
    cand_acc = sg.truth_accumulator(lid, "phi_mean_std", candidate)
    maps = stat.maps(cand_acc)
    np.testing.assert_allclose(np.asarray(maps["phi_mean"]), cand.mean(axis=0),
                               rtol=1e-12, atol=1e-12 * np.abs(cand).max())
    np.testing.assert_allclose(np.asarray(maps["phi_std"])[wet],
                               cand.std(axis=0)[wet], rtol=1e-10)
    assert cand.std(axis=0)[wet].min() > 0.0      # four distinct frames
    terms = [float(t) for t in stat.terms(
        cand_acc, sg.truth_accumulator(lid, "phi_mean_std", reference))]
    for got, (c, r) in zip(terms, ((cand.mean(axis=0), ref.mean(axis=0)),
                                   (cand.std(axis=0), ref.std(axis=0))),
                           strict=True):
        assert got == pytest.approx(
            np.sum((c[wet] - r[wet])**2) / np.sum(r[wet]**2), rel=1e-9)


def test_phi_statistic_refuses_an_explicit_card():
    explicit = build_swg_card(SWG_DG50_NAMELIST)
    with pytest.raises(ValueError, match="rigid-lid"):
        sg.Statistic(explicit, "phi_mean_std",
                     sg.statistic_geometry(SWG_DG50_NAMELIST))


def test_unknown_selections_raise(lid, geometry, networks):
    constants = sg.statistic_geometry(SWG_DG50_RL_NAMELIST)
    with pytest.raises(ValueError, match="statistic must be"):
        sg.Statistic(lid, "bogus", constants)
    with pytest.raises(ValueError, match="trainable must be"):
        sg.trainable_spec(_model(networks), "bogus")
    with pytest.raises(ValueError, match="precision must be"):
        sg.build_programs(lid, geometry, days=2, statistic="transport_eke",
                          precision="fp16")
    with pytest.raises(ValueError, match="window_days"):
        sg.build_programs(lid, geometry, days=1, statistic="transport_eke",
                          precision="fp64")


# ------------------------------------------------------- the closure wrapper

def test_gamma_one_reproduces_the_plain_closure_bit_for_bit(
        lid, geometry, networks):
    _, jitted = build_segment_fn(
        lid, geometry, window_days=1, target_interval_hours=24, checkpoint=True)
    noise = jnp.zeros((8, 3) + tuple(geometry.tmask.shape))
    start = rigid_lid_start_state(lid, lid.recipe.initial_state)
    key = jax.random.PRNGKey(0)
    scale = jnp.asarray(0.0)

    def run(network):
        return jitted(network, start, noise, key, scale)

    effective = _model(networks, gamma=1.0).effective_network()
    for was, now in zip(jax.tree.leaves(networks[0]), jax.tree.leaves(effective),
                        strict=True):
        assert np.array_equal(np.asarray(was), np.asarray(now))
    plain_state, plain = run(networks[0])
    one_state, one = run(effective)
    other_state, other = run(_model(networks, gamma=1.001).effective_network())
    for a, b in zip(plain, one, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(plain_state.rigid_lid_phi),
                          np.asarray(one_state.rigid_lid_phi))
    # Not vacuous: the closure is live, so another gamma changes the run, and
    # by the scaled weights only (the last convolution has no bias).
    assert not np.array_equal(np.asarray(plain[1]), np.asarray(other[1]))
    assert np.array_equal(
        np.asarray(_model(networks, gamma=1.001).effective_network()
                   .layers[-1].weight),
        np.asarray(networks[0].layers[-1].weight) * 1.001)
    assert networks[0].layers[-1].bias is None


def test_trainable_subsets_select_exactly_their_leaves(networks):
    model = _model(networks)
    counts = {}
    for subset in sg.SUBSETS:
        trainable, frozen = eqx.partition(model, sg.trainable_spec(model, subset))
        counts[subset] = len(jax.tree.leaves(trainable))
        assert len(jax.tree.leaves(trainable)) + len(jax.tree.leaves(frozen)) \
            == len(jax.tree.leaves(model))
    layers = len(networks[0].layers)
    assert counts == {"gamma": 1, "gamma_last_layer": 2,
                      "all": 1 + 2 * (layers - 1) + 1}
    trainable, _ = eqx.partition(model, sg.trainable_spec(model, "gamma_last_layer"))
    assert trainable.network.layers[-1].weight is not None
    assert trainable.network.layers[0].weight is None


def _optimizer(lr=1.0e-3):
    return create_swg_optimizer(SWGOptimizerSettings(
        optimizer="adamw", learning_rate=lr, schedule="constant",
        weight_decay=0.0, adam_b1=0.9, adam_b2=0.999, adam_epsilon=1.0e-7,
        adam_epsilon_convention="keras_epsilon_hat", gradient_clipping="none"))


@pytest.mark.parametrize("subset", sg.SUBSETS)
def test_frozen_leaves_get_zero_updates(networks, subset):
    model = _model(networks)
    optimizer = _optimizer()
    trainable, frozen = eqx.partition(model, sg.trainable_spec(model, subset))
    grads = jax.tree.map(jnp.ones_like, trainable)
    stepped, _ = sg.adam_step(
        optimizer, optimizer.init(trainable), trainable, frozen, grads)
    old, new = (jax.tree.leaves(m) for m in (model, stepped))
    spec = jax.tree.leaves(sg.trainable_spec(model, subset))
    for was, now, selected in zip(old, new, spec, strict=True):
        if selected:
            assert not np.array_equal(np.asarray(was), np.asarray(now))
        else:
            assert np.array_equal(np.asarray(was), np.asarray(now))


# ------------------------------------------------------------- the gradients

def _window(lid, runs, truth_index, statistic, start_index=0):
    return sg.Window(
        0, _start(lid, runs[start_index]),
        sg.truth_accumulator(lid, statistic, runs[truth_index]))


@pytest.mark.parametrize("statistic", sg.STATISTICS)
def test_gamma_gradient_matches_finite_differences(
        lid, runs, networks, programs, statistic):
    value_and_grad, loss_fn = programs[statistic]
    window = _window(lid, runs, 1, statistic)
    model = _model(networks)
    trainable, frozen = eqx.partition(model, sg.trainable_spec(model, "gamma"))
    loss, grads = value_and_grad(trainable, frozen, window.start_state,
                                 window.truth)
    assert len(jax.tree.leaves(grads)) == 1          # gamma only
    step = 1.0e-4

    def at(gamma):
        shifted = eqx.tree_at(lambda m: m.gamma, model, jnp.asarray(gamma))
        t, f = eqx.partition(shifted, sg.trainable_spec(shifted, "gamma"))
        return float(loss_fn(t, f, window.start_state, window.truth))

    fd = (at(1.0 + step) - at(1.0 - step)) / (2.0 * step)
    ad = float(grads.gamma)
    assert abs(ad) > 1e-8                   # the derivative is not zero
    assert abs(fd - ad) / abs(ad) < 1.0e-4
    assert float(loss) == pytest.approx(at(1.0), rel=1e-12)


@pytest.mark.parametrize("statistic", sg.STATISTICS)
def test_all_parameter_gradient_matches_a_directional_difference(
        lid, runs, networks, programs, statistic):
    value_and_grad, loss_fn = programs[statistic]
    window = _window(lid, runs, 1, statistic)
    model = _model(networks)
    trainable, frozen = eqx.partition(model, sg.trainable_spec(model, "all"))
    _, grads = value_and_grad(trainable, frozen, window.start_state, window.truth)
    flat, unravel = ravel_pytree(trainable)
    direction = jax.random.normal(jax.random.PRNGKey(5), flat.shape)
    direction = direction / jnp.linalg.norm(direction)
    step = 1.0e-5

    def at(sign):
        return float(loss_fn(unravel(flat + sign * step * direction), frozen,
                             window.start_state, window.truth))

    fd = (at(1.0) - at(-1.0)) / (2.0 * step)
    ad = float(ravel_pytree(grads)[0] @ direction)
    assert abs(ad) > 1e-8
    assert abs(fd - ad) / abs(ad) < 1.0e-4


def test_single_precision_keeps_float64_masters_and_the_float64_loss(
        lid, geometry, runs, networks, programs):
    value_and_grad32, _ = sg.build_programs(
        lid, geometry, days=DAYS, statistic="transport_eke", precision="fp32")
    value_and_grad64, _ = programs["transport_eke"]
    window = _window(lid, runs, 1, "transport_eke")
    model = _model(networks)
    trainable, frozen = eqx.partition(model, sg.trainable_spec(model, "all"))
    loss64, grads64 = value_and_grad64(trainable, frozen, window.start_state,
                                       window.truth)
    loss32, grads32 = value_and_grad32(trainable, frozen, window.start_state,
                                       window.truth)
    assert {leaf.dtype for leaf in jax.tree.leaves((loss32, grads32))} \
        == {jnp.dtype("float64")}
    assert float(loss32) == pytest.approx(float(loss64), rel=1.0e-4)
    g64 = ravel_pytree(grads64)[0]
    g32 = ravel_pytree(grads32)[0]
    assert float(jnp.linalg.norm(g32 - g64) / jnp.linalg.norm(g64)) < 1.0e-3
    assert float(loss32) != float(loss64)          # it really ran in float32


def _setup(lid, runs, networks, programs, statistic, heldout_truth):
    return sg.Setup(
        _model(networks), "gamma", programs[statistic],
        SWGOptimizerSettings(
            optimizer="adamw", learning_rate=1.0e-3, schedule="constant",
            weight_decay=0.0, adam_b1=0.9, adam_b2=0.999, adam_epsilon=1.0e-7,
            adam_epsilon_convention="keras_epsilon_hat",
            gradient_clipping="none"),
        [_window(lid, runs, 1, statistic), _window(lid, runs, 2, statistic)],
        [_window(lid, runs, heldout_truth, statistic)])


def test_held_out_starts_never_enter_the_gradient(lid, runs, networks, programs):
    model = _model(networks)
    trainable, frozen = eqx.partition(model, sg.trainable_spec(model, "gamma"))
    first = sg.evaluate(_setup(lid, runs, networks, programs, "transport_eke", 1),
                        trainable, frozen)
    second = sg.evaluate(_setup(lid, runs, networks, programs, "transport_eke", 2),
                         trainable, frozen)
    assert np.array_equal(first[0], second[0])      # training losses
    assert np.array_equal(first[1], second[1])      # training gradients
    assert not np.array_equal(first[2], second[2])  # held-out losses differ


def test_start_days_must_lie_in_their_split_and_be_spaced():
    contract = types.SimpleNamespace(splits=(
        ("train", 24 * 100, 24 * 500), ("validation", 24 * 530, 24 * 700),
        ("test", 24 * 730, 24 * 900)))
    sg.check_start_days(contract, 30, [100, 130, 440], [530, 600])
    for train, heldout, match in (
            ([100, 120], [530], "spacing"),            # closer than the window
            ([130, 100], [530], "spacing"),            # not increasing
            ([99], [530], "leaves the train split"),
            ([471], [530], "leaves the train split"),
            ([100], [529], "leaves the validation split"),
            ([100], [100], "leaves the validation split"),   # a train day
            ([100], [671], "leaves the validation split"),
            ([], [530], "non-empty"),
            ([100.0], [530], "non-empty")):
        with pytest.raises(ValueError, match=match):
            sg.check_start_days(contract, 30, train, heldout)


# ----------------------------------------------------------------- the commands

def _lid_record_fields(runs):
    """Hourly fields of a synthetic record: the four daily lid frames of one
    model run, repeated and linearly interpolated in the hour (a mix of
    non-divergent fields is non-divergent, so every start rebalances)."""
    u, v = runs[1][1], runs[1][2]
    frames = u.shape[0]

    def at(field, hours):
        position = (np.asarray(hours, float) / 24.0) % (frames - 1)
        low = np.floor(position).astype(int)
        weight = (position - low)[..., None, None]
        return field[low] * (1.0 - weight) + field[low + 1] * weight

    def edit(day, arrays):
        hours = arrays["absolute_hour"]
        for prefix in ("box_", "filtered_"):
            arrays[f"{prefix}u_hourly_m_s"] = at(u, hours)
            arrays[f"{prefix}v_hourly_m_s"] = at(v, hours)
            if day == 1:
                arrays[f"{prefix}u_initial_m_s"] = at(u, hours[:1] - 1)[0]
                arrays[f"{prefix}v_initial_m_s"] = at(v, hours[:1] - 1)[0]
    return edit


@pytest.fixture(scope="module")
def lid_files(tmp_path_factory, lid, geometry, runs):
    """A tiny rigid-lid record, a contract over it, and a checkpoint of the
    committed configuration's network."""
    root = tmp_path_factory.mktemp("stat_gradient")
    masks = (np.asarray(geometry.umask), np.asarray(geometry.vmask))
    sources = [(root / name, records._write_chunk(
        root / name, source_day=day, days=days, masks=masks,
        truth_run=root / "truth", manifest_edit=records._lid_stream,
        day_edit=_lid_record_fields(runs)))
        for name, day, days in records._CHUNKS]
    contract = records._write_contract(
        root / "dataset_contract.json", sources, objective="symmetric")
    config = sg.driver.load_config(EXP4_SYMMETRIC)
    spec, _ = sg.driver.network_spec(config, lid, geometry)
    network = SWGScalarPotentialCNN(spec, key=jax.random.PRNGKey(7))
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))
    checkpoint = root / "w08_e05.eqx"
    save_checkpoint((network, optimizer.init(
        eqx.filter(network, eqx.is_inexact_array))), checkpoint)
    return {"contract": contract, "checkpoint": checkpoint, "root": root}


def _argv(command, files, output, *extra, steps=None):
    argv = [command, "--config", str(EXP4_SYMMETRIC),
            "--contract", str(files["contract"]),
            "--checkpoint", str(files["checkpoint"]), "--device", "cpu",
            "--window-days", str(DAYS), "--train-start-days", "1", "5",
            "--heldout-start-days", "53", "57", "--trainable", "gamma",
            "--statistic", "transport_eke", "--precision", "fp64",
            "--lr", "0.01", "--weight-decay", "0.0", "--output", str(output),
            *extra]
    return argv + (["--steps", str(steps)] if steps is not None else [])


def test_measure_end_to_end_writes_the_manifest_and_the_results(
        monkeypatch, tmp_path, lid_files):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    output = tmp_path / "measure"
    assert sg.main(_argv("measure", lid_files, output)) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == sg.SCHEMA and manifest["command"] == "measure"
    assert len(manifest["legoesm_git_sha"]) >= 40
    assert manifest["start_days"] == {"train": [1, 5], "heldout": [53, 57]}
    assert manifest["arguments"]["window_days"] == DAYS
    assert manifest["checkpoint_sha256"] == sg.driver._sha256(
        lid_files["checkpoint"])
    result = json.loads((output / "measure.json").read_text())
    assert [row["start_day"] for row in result["train"]] == [1, 5]
    assert [row["start_day"] for row in result["heldout"]] == [53, 57]
    assert all(-1.0 <= row["cosine_to_mean_gradient"] <= 1.0
               for row in result["train"])
    assert [row["step"] for row in result["gamma_finite_differences"]] \
        == list(sg.FD_STEPS)
    assert result["gamma_finite_differences"][-1]["relative_error"][0] < 1e-3
    gradients = np.load(output / "gradients.npz")["gradients"]
    assert gradients.shape == (2, 1)
    assert result["adam_step"]["mean_train_loss_after"] \
        != result["adam_step"]["mean_train_loss_before"]
    assert sg.main(_argv("measure", lid_files, output)) == 2   # never overwrites


def _train(monkeypatch, setup, files, output, steps, *extra):
    monkeypatch.setattr(sg, "prepare", lambda args: (
        setup, {"device": {"device": "cpu"}}))
    return sg.main(_argv("train", files, output, *extra, steps=steps))


def _read(path):
    return json.loads(Path(path).read_text())


def test_train_resumes_bit_for_bit_and_refuses_a_changed_run(
        monkeypatch, tmp_path, lid, runs, networks, programs, lid_files):
    setup = _setup(lid, runs, networks, programs, "transport_eke", 2)
    straight, split = tmp_path / "straight", tmp_path / "split"
    assert _train(monkeypatch, setup, lid_files, straight, 3) == 0
    assert _train(monkeypatch, setup, lid_files, split, 1) == 0
    first = _read(split / "train_log.json")
    assert [row["step"] for row in first["steps"]] == [0, 1]
    assert _train(monkeypatch, setup, lid_files, split, 3) == 0
    resumed, whole = _read(split / "train_log.json"), _read(
        straight / "train_log.json")
    assert resumed["status"] == "complete"
    assert resumed["steps"][:2] == first["steps"]       # not recomputed
    assert resumed["steps"] == whole["steps"]
    assert sorted(path.name for path in split.glob("step_*.eqx")) \
        == [f"step_{k:05d}.eqx" for k in range(4)]
    model = setup.model
    optimizer = _optimizer(0.01)
    skeleton = (model, optimizer.init(eqx.partition(
        model, sg.trainable_spec(model, "gamma"))[0]))
    for name in ("step_00003.eqx",):
        a = eqx.tree_deserialise_leaves(split / name, skeleton)
        b = eqx.tree_deserialise_leaves(straight / name, skeleton)
        assert eqx.tree_equal(a, b)
    # The parameters moved and the loss fell on the training starts.
    assert float(a[0].gamma) != 1.0
    assert whole["steps"][-1]["train_loss"] < whole["steps"][0]["train_loss"]
    # The same output directory refuses another learning rate.
    monkeypatch.setattr(sg, "prepare", lambda args: (
        setup, {"device": {"device": "cpu"}}))
    argv = _argv("train", lid_files, split, steps=3)
    argv[argv.index("--lr") + 1] = "0.02"
    assert sg.main(argv) == 2


def test_train_stops_on_a_non_finite_loss(
        monkeypatch, tmp_path, lid, runs, networks, programs, lid_files):
    setup = _setup(lid, runs, networks, programs, "transport_eke", 2)
    poisoned = setup.heldout[0]._replace(truth=jax.tree.map(
        lambda leaf: leaf * jnp.nan, setup.heldout[0].truth))
    output = tmp_path / "nan"
    assert _train(monkeypatch, setup._replace(heldout=[poisoned]), lid_files,
                  output, 3) == 3
    log = _read(output / "train_log.json")
    assert log["status"] == "nan" and [r["step"] for r in log["steps"]] == [0]
    assert not (output / "step_00001.eqx").exists()
    assert sg.main(_argv("train", lid_files, output, steps=3)) == 2   # no resume


# ------------------------------------------------------------------- free-run

def _free_run_argv(run, step, root, name):
    return ["free-run", "--run", str(run), "--step", str(step), "--seed", "7",
            "--window-days", "8", "--network-output", str(root / f"{name}.eqx"),
            "--output", str(root / name)]


def test_free_run_exports_the_scaled_closure_as_a_plain_network(
        monkeypatch, tmp_path, lid, geometry, runs, networks, programs,
        lid_files, capsys):
    """Step 0 free-runs as the starting network, step 1 as gamma times its
    last layer, both through the driver's own free-run code."""
    from legoesm.ocean.fidelity import provenance

    setup = _setup(lid, runs, networks, programs, "transport_eke", 2)._replace(
        subset="gamma_last_layer")
    load = sg.driver.load_config
    # The committed configuration with the fixtures' small network and a
    # one-day CPU free run; the starting checkpoint is the small network.
    monkeypatch.setattr(sg.driver, "load_config", lambda path: (
        lambda config: config | {
            "network": config["network"] | dict(hidden_channels=4,
                                                hidden_layers=2),
            "free_run": config["free_run"] | dict(
                days=1, device="cpu", math_library="libm")})(load(path)))
    small = SWGOptimizerSettings(**load(EXP4_SYMMETRIC)["optimiser"])
    base = tmp_path / "base.eqx"
    save_checkpoint((networks[0], create_swg_optimizer(small).init(
        eqx.filter(networks[0], eqx.is_inexact_array))), base)
    train = tmp_path / "train"
    assert _train(monkeypatch, setup, lid_files | {"checkpoint": base}, train, 1,
                  "--trainable", "gamma_last_layer") == 0
    monkeypatch.setattr(provenance, "git_sha", lambda **kwargs: "0" * 40)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    for step in (0, 1):
        assert sg.main(_free_run_argv(train, step, tmp_path, f"s{step}")) == 0
    assert "FREE_RUN_COMPLETE" in capsys.readouterr().out
    # The exported networks, read as the driver reads a checkpoint.
    config = load(EXP4_SYMMETRIC)
    optimizer = create_swg_optimizer(SWGOptimizerSettings(**config["optimiser"]))

    def exported(step):
        skeleton = (networks[0], optimizer.init(
            eqx.filter(networks[0], eqx.is_inexact_array)))
        return eqx.tree_deserialise_leaves(
            tmp_path / f"s{step}.eqx", skeleton)[0]
    scaled = _model(networks)
    model_1 = eqx.tree_deserialise_leaves(
        train / "step_00001.eqx",
        (scaled, _optimizer(0.01).init(eqx.partition(
            scaled, sg.trainable_spec(scaled, "gamma_last_layer"))[0])))[0]
    assert float(model_1.gamma) != 1.0
    assert eqx.tree_equal(exported(0), networks[0])
    assert eqx.tree_equal(exported(1), model_1.effective_network())
    assert not eqx.tree_equal(exported(1), networks[0])
    # Same closure output as the scaled closure, through the training segment.
    _, jitted = build_segment_fn(
        lid, geometry, window_days=1, target_interval_hours=24, checkpoint=True)
    noise = jnp.zeros((8, 3) + tuple(geometry.tmask.shape))
    start = rigid_lid_start_state(lid, lid.recipe.initial_state)
    key = jax.random.PRNGKey(0)
    _, a = jitted(exported(1), start, noise, key, jnp.asarray(0.0))
    _, b = jitted(model_1.effective_network(), start, noise, key,
                  jnp.asarray(0.0))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # The step-0 free run is the starting network's free run, day files bit
    # for bit (the driver's own free_run, the same closure block); step 1 is not.
    manifest = _read(tmp_path / "s0" / "manifest.json")
    assert manifest["status"] == "complete" and manifest["days"] == 1
    assert manifest["closure"]["checkpoint_sha256"] == sg.driver._sha256(
        tmp_path / "s0.eqx")
    assert manifest["closure"]["run_id"].endswith("/arm_e/train/step00000")
    sg.driver.free_run(
        networks[0], lid, geometry, tmp_path / "direct", days=1,
        closure=manifest["closure"], commit="0" * 40,
        device={name: manifest[name] for name in (
            "device", "backend", "math_library", "xla_flags", "jax_version",
            "gpu_name")}, start=(None, "rest"))
    for name in ("s0", "s1"):
        with np.load(tmp_path / name / "day0001.npz") as actual, \
                np.load(tmp_path / "direct" / "day0001.npz") as expected:
            same = [np.array_equal(actual[field], expected[field])
                    for field in sg.driver.SNAPSHOT_FIELDS]
        assert all(same) == (name == "s0")
    # Refusals: no overwrite of the network or the run, no changed inputs.
    assert sg.main(_free_run_argv(train, 0, tmp_path, "s0")) == 2
    stored = _read(train / "manifest.json")
    stored["checkpoint_sha256"] = "0" * 64
    (train / "manifest.json").write_text(json.dumps(stored))
    assert sg.main(_free_run_argv(train, 0, tmp_path, "s2")) == 2
    assert "differs from the checkpoint_sha256" in capsys.readouterr().err

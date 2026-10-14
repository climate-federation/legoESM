"""Direct tests for the online-closure held-out gate (round 6).

Scorer plants run on REAL dg50 fields (the registered box-restricted truth),
never on white noise: white-noise 'truth' already exceeds the noise bar, so a
plant on it cannot fail for the reason it claims (review EV-05).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from legoesm.training.swg_online import (
    FREE_RUN_CLOSURE_KEYS,
    FREE_RUN_MANIFEST_SCHEMA,
    SWGNetworkSpec,
)

SCRIPT =(Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
          / "testcases" / "nemo_testcase_l1_swg_online_evaluation.py")
SPEC = importlib.util.spec_from_file_location("swg_online_evaluation", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluation)
COARSE_TRUTH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgonl_rounds/round2/"
    "conservative_truth_dg50_1095d/snapshots")
real_fields = pytest.mark.skipif(
    not (COARSE_TRUTH / "manifest.json").is_file(),
    reason="registered dg50 coarse truth not mounted")
OBJECTIVES = ("symmetric", "template")


def _real_window(first=366, days=30):
    return evaluation._load_native_window(COARSE_TRUTH, first, days)


def _dual_archive(root, days, *, rename=(), labels=None, manifest=None,
                  nan=(), scale=1.0, extra=None):
    """A driver-v2 dual archive built from the registered dg50 truth products:
    each day's real end-of-day snapshot beside that day's real daily mean.

    ``rename`` puts the daily mean under the named snapshot fields (the
    reviewer's mislabelled payload); ``labels`` overrides per-file labels;
    ``nan`` lists (day, field) pairs given one NaN after any rename;
    ``scale`` multiplies every field (distinct data, same sampling);
    ``extra`` adds unused entries (distinct bytes, same data).
    """
    root.mkdir(exist_ok=True)
    pins = {}
    for day in days:
        with np.load(COARSE_TRUTH / f"day{day:04d}.npz") as snapshot, \
                np.load(COARSE_TRUTH.parent / "daily_means"
                        / f"day{day:04d}.npz") as mean:
            arrays = {
                "schema": np.str_(evaluation.DUAL_SCHEMA), "day": np.int64(day),
                "snapshot_sampling": np.str_(evaluation.SNAPSHOT_LABEL),
                "daily_mean_sampling": np.str_(evaluation.DAILY_MEAN_LABEL),
                **{key: scale * np.asarray(snapshot[key])
                   for key in evaluation.SNAPSHOT_FIELDS},
                **{key: scale * np.asarray(mean[key])
                   for key in evaluation.DAILY_MEAN_FIELDS},
                **(extra or {})}
        for snap_key, mean_key in zip(evaluation.SNAPSHOT_FIELDS,
                                      evaluation.DAILY_MEAN_FIELDS, strict=True):
            if snap_key in rename:
                arrays[snap_key] = arrays[mean_key].copy()
        for key, value in (labels or {}).items():
            if value is None:
                arrays.pop(key)
            else:
                arrays[key] = np.str_(value)
        for nan_day, key in nan:
            if nan_day == day:
                arrays[key] = arrays[key].copy()
                arrays[key][10, 10] = np.nan
        path = root / f"day{day:04d}.npz"
        np.savez(path, **arrays)
        pins[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(
        {"status": "complete", "daily_sha256": pins, **(manifest or {})}))
    return evaluation._sha256(root / "manifest.json")


def _seeds(rows, count=3):
    return {f"seed{k}": list(rows) for k in range(count)}


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_frozen_bars_are_the_truth_table_derivation(objective):
    frozen = evaluation.frozen_gate(objective)
    windows = [tuple(window) for window in frozen["windows"]]
    assert len(windows) >= 3
    assert windows == evaluation.registered_windows(len(windows))
    assert frozen["bars"] == evaluation.gate_bars(frozen["truth_loo"])
    assert set(frozen["controls"]) == {"nu10", "nu80"}
    for table in (frozen["truth_loo"], *frozen["controls"].values()):
        assert set(table) == set(evaluation.SCORE_METRICS)
        assert all(len(values) == len(windows) for values in table.values())


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_truth_from_other_windows_passes_and_controls_fail(objective):
    """EV-01: a perfect model passes; neither no-closure control does."""
    truth = evaluation.held_out_gate(
        _seeds(evaluation.truth_as_seed(objective)), objective=objective)
    assert truth["pass"], truth
    bars = evaluation.frozen_gate(objective)["bars"]
    for row in evaluation.truth_as_seed(objective):  # by construction
        assert all(row[metric] <= bars[metric]["bar"]
                   for metric in evaluation.SKILL_METRICS)
        assert (bars["basin_eke_ratio"]["floor"] <= row["basin_eke_ratio"]
                <= bars["basin_eke_ratio"]["bar"])
    for control in ("nu10", "nu80"):
        result = evaluation.held_out_gate(
            _seeds(evaluation.control_as_seed(objective, control)),
            objective=objective)
        assert not result["pass"] and not result["skill_pass"]


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_skill_must_beat_both_controls_by_more_than_the_truth_spread(objective):
    """A seed whose every window improves on the better control by less than
    the truth's own half-range fails skill; by twice that it passes."""
    frozen = evaluation.frozen_gate(objective)
    better = {metric: min(float(np.median(table[metric]))
                          for table in frozen["controls"].values())
              for metric in evaluation.SKILL_METRICS}
    truth = evaluation.truth_as_seed(objective)
    for factor, expected in ((0.5, False), (2.0, True)):
        rows = [dict(row, **{
            metric: better[metric]
            - factor * frozen["bars"][metric]["half_range"]
            for metric in evaluation.SKILL_METRICS}) for row in truth]
        result = evaluation.held_out_gate(_seeds(rows), objective=objective)
        assert result["skill_pass"] is expected


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_eke_band_is_the_truth_spread_not_a_fixed_twenty_percent(objective):
    band = evaluation.frozen_gate(objective)["bars"]["basin_eke_ratio"]
    assert 0.8 < band["floor"] < 1.0 < band["bar"] < 1.2
    truth = evaluation.truth_as_seed(objective)
    for ratio, expected in ((band["floor"] - 0.01, False),
                            (band["bar"] + 0.01, False), (1.0, True)):
        rows = [dict(row, basin_eke_ratio=ratio) for row in truth]
        result = evaluation.held_out_gate(_seeds(rows), objective=objective)
        assert result["variability_pass"] is expected


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_noise_allowance_is_the_truth_spread_above_the_worse_control(objective):
    """Codex round-6 defect 2: the allowance is the TRUTH's half-range.

    Per noise metric: the limit is the worse control's median plus the
    truth's half-range; half that allowance passes, one and a half fails.
    """
    frozen = evaluation.frozen_gate(objective)
    bars = frozen["bars"]
    truth = evaluation.truth_as_seed(objective)
    limits = evaluation.held_out_gate(
        _seeds(truth), objective=objective)["noise_must_not_exceed"]
    for metric in evaluation.NOISE_METRICS:
        worse = max(float(np.median(table[metric]))
                    for table in frozen["controls"].values())
        allowance = bars[metric]["half_range"]
        assert allowance == evaluation.gate_bars(
            frozen["truth_loo"])[metric]["half_range"]
        assert limits[metric] == worse + allowance
        for factor, expected in ((0.5, True), (1.5, False)):
            rows = [dict(row, **{metric: worse + factor * allowance})
                    for row in truth]
            result = evaluation.held_out_gate(_seeds(rows), objective=objective)
            assert result["spectral_health_pass"] is expected, (metric, factor)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_no_threshold_moves_with_a_controls_spread(objective, monkeypatch):
    """Widening every control's window-to-window spread tenfold about its
    median leaves every threshold unchanged: controls supply levels only."""
    frozen = evaluation.frozen_gate(objective)
    keys = ("skill_must_be_below", "eke_ratio_band", "noise_must_not_exceed")
    seeds = _seeds(evaluation.truth_as_seed(objective))
    before = evaluation.held_out_gate(seeds, objective=objective)
    widened = {name: {metric: list(np.median(values)
                                   + 10.0 * (np.asarray(values)
                                             - np.median(values)))
                      for metric, values in table.items()}
               for name, table in frozen["controls"].items()}
    monkeypatch.setitem(evaluation.FROZEN_GATE, objective,
                        dict(frozen, controls=widened))
    after = evaluation.held_out_gate(seeds, objective=objective)
    for name, medians in before["control_window_medians"].items():
        assert after["control_window_medians"][name] == pytest.approx(
            medians, rel=1e-12)
    assert evaluation.gate_bars(widened["nu10"]) != evaluation.gate_bars(
        frozen["controls"]["nu10"])
    for key in keys:
        assert after[key] == pytest.approx(before[key], rel=1e-12), key


def test_gate_binds_objective_seed_count_windows_and_health():
    rows = evaluation.truth_as_seed("symmetric")
    with pytest.raises(ValueError, match="objective must be one of"):
        evaluation.held_out_gate(_seeds(rows), objective="box")
    with pytest.raises(ValueError, match="three"):
        evaluation.held_out_gate(_seeds(rows, 2), objective="symmetric")
    with pytest.raises(ValueError, match="one score per frozen window"):
        evaluation.held_out_gate(_seeds(rows[:-1]), objective="symmetric")
    blown_up = list(rows)
    blown_up[3] = {"physical_health": {"all_finite": False, "passes": False}}
    result = evaluation.held_out_gate(
        {"a": rows, "b": rows, "c": blown_up}, objective="symmetric")
    assert not result["pass"] and not result["physical_health_pass"]


def test_truth_operator_follows_the_objective_explicitly():
    """EV-03: template scores against filtered truth, symmetric against box."""
    from legoesm.ocean.fidelity.swg_coarse_graining import (
        conservative_restrict_cgrid,
        filtered_restrict_cgrid,
    )

    box = evaluation.truth_restrictor("symmetric")
    filtered = evaluation.truth_restrictor("template")
    assert box.func is conservative_restrict_cgrid
    assert filtered.func is filtered_restrict_cgrid
    assert filtered.keywords["sigma_source_cells"] == 8.0
    for unknown in ("box", "filtered", "", None):
        with pytest.raises(ValueError, match="objective must be one of"):
            evaluation.truth_restrictor(unknown)


def _synthetic_stats(rng, offset):
    _, geometry = evaluation._dg50()
    shape = geometry["mask"]["T"].shape
    return {
        "days": evaluation.GATE_WINDOW_DAYS,
        "mean_transport_m3_s": {kind: offset + rng.normal(size=shape)
                                for kind in ("U", "V")},
        "mean_kinetic_energy_spectrum_m3_s2": 2.0 + rng.random(16),
        "eke_map_m2_s2": 1.0 + rng.random(shape),
        "basin_mean_eke_m2_s2": 1.0 + 0.1 * rng.random(),
        **{metric: 0.5 for metric in evaluation.NOISE_METRICS},
        "physical_health": {"passes": True},
    }


def test_leave_one_out_scores_exclude_their_own_window():
    rng = np.random.default_rng(14)
    stats = [_synthetic_stats(rng, offset) for offset in (0.0, 1.0, 2.0, 3.0)]
    loo = evaluation.leave_one_out_scores(stats)
    for k, row in enumerate(loo):
        others = evaluation.climatology(stats[:k] + stats[k + 1:])
        assert row == evaluation.score_statistics(stats[k], others)
    with_self = evaluation.score_statistics(stats[0], evaluation.climatology(stats))
    assert with_self["mean_face_transport_rmse_m3_s"] \
        < loo[0]["mean_face_transport_rmse_m3_s"]
    bars = evaluation.gate_bars({metric: [row[metric] for row in loo]
                                 for metric in evaluation.SCORE_METRICS})
    assert all(row[metric] <= bars[metric]["bar"] for row in loo
               for metric in evaluation.SKILL_METRICS)


@real_fields
def test_skill_metrics_have_known_answers_on_real_fields():
    """EV-N1: scaling real velocities by a scales every metric analytically."""
    truth = _real_window()
    reference = evaluation.window_statistics(truth)
    scores = {}
    for alpha in (1.0, 2.0, 3.0):
        candidate = (truth[0], alpha * truth[1], alpha * truth[2])
        scores[alpha] = evaluation.score_statistics(
            evaluation.window_statistics(candidate), reference)
    identity = scores[1.0]
    assert all(identity[metric] == 0.0 for metric in evaluation.SKILL_METRICS)
    assert identity["basin_eke_ratio"] == 1.0
    transport = "mean_face_transport_rmse_m3_s"
    assert scores[2.0][transport] > 0.0
    assert scores[3.0][transport] / scores[2.0][transport] \
        == pytest.approx(2.0, rel=1e-12)
    for alpha in (2.0, 3.0):
        for metric in ("ke_spectrum_relative_l1", "wet_eke_map_nrmse"):
            assert scores[alpha][metric] == pytest.approx(alpha**2 - 1.0,
                                                          rel=1e-12)
        assert scores[alpha]["basin_eke_ratio"] == pytest.approx(alpha**2,
                                                                 rel=1e-12)
        assert scores[alpha]["native_grid_noise_index"] == pytest.approx(
            identity["native_grid_noise_index"], rel=1e-12)


@real_fields
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_native_checkerboards_on_real_fields_fail_spectral_health(objective):
    """EV-05: a 0.1 m/s face checkerboard trips the native noise measures.

    Real truth is far below the frozen noise limits, so only the planted
    grid-scale signal (which cell-centring would average away) can trip them.
    """
    truth = _real_window()
    reference = evaluation.window_statistics(truth)
    frozen = evaluation.frozen_gate(objective)
    limits = evaluation.held_out_gate(
        _seeds(evaluation.truth_as_seed(objective)),
        objective=objective)["noise_must_not_exceed"]
    clean = evaluation.score_statistics(reference, reference)
    assert all(clean[metric] < limits[metric]
               for metric in evaluation.NOISE_METRICS)
    divergent = evaluation.score_statistics(evaluation.window_statistics(
        evaluation.checkerboard_plant(truth, 0.1, rotational=False)), reference)
    assert divergent["native_grid_noise_index"] \
        > limits["native_grid_noise_index"]
    rotational = evaluation.score_statistics(evaluation.window_statistics(
        evaluation.checkerboard_plant(truth, 0.1, rotational=True)), reference)
    assert rotational["top_octave_enstrophy_fraction"] \
        > limits["top_octave_enstrophy_fraction"]
    windows = len(frozen["windows"])
    for score in (divergent, rotational):
        gate = evaluation.held_out_gate(_seeds([score] * windows),
                                        objective=objective)
        assert not gate["pass"] and not gate["spectral_health_pass"]


@real_fields
def test_collapsed_variance_plant_on_real_fields_fails_variability():
    truth = _real_window()
    reference = evaluation.window_statistics(truth)
    collapsed = evaluation.score_statistics(evaluation.window_statistics(
        evaluation.collapsed_variance_plant(truth)), reference)
    assert collapsed["basin_eke_ratio"] < 1.0e-20
    windows = len(evaluation.frozen_gate("symmetric")["windows"])
    gate = evaluation.held_out_gate(_seeds([collapsed] * windows),
                                    objective="symmetric")
    assert not gate["pass"] and not gate["variability_pass"]


@real_fields
def test_scorer_health_catches_negative_thickness_and_courant_two():
    """EV-05: the scorer's own health computation, not an injected dict."""
    ssh, u, v = _real_window(days=6)
    _, geometry = evaluation._dg50()
    card, _ = evaluation._dg50()
    depth = np.asarray(card.recipe.initial_state.H_bathy.data)
    jw, iw = np.argwhere(geometry["mask"]["T"])[40]
    thin = ssh.copy()
    thin[2, jw, iw] = -depth[jw, iw] - 1.0
    ju, iu = np.argwhere(geometry["mask"]["U"])[40]
    fast = u.copy()
    fast[3, ju, iu] = 2.0 * float(geometry["spacing_m"]) \
        / evaluation.SWG_DG50_NAMELIST.rn_Dt_s
    clean = evaluation.window_statistics((ssh, u, v))["physical_health"]
    assert clean["passes"]
    thin_health = evaluation.window_statistics((thin, u, v))["physical_health"]
    assert thin_health["minimum_total_layer_thickness_m"] == pytest.approx(-1.0)
    assert not thin_health["passes"]
    fast_health = evaluation.window_statistics((ssh, fast, v))["physical_health"]
    assert fast_health["maximum_courant"] == pytest.approx(2.0, rel=1e-12)
    assert not fast_health["passes"]


@real_fields
def test_free_run_scoring_binds_manifest_and_marks_nonfinite_seed(tmp_path):
    """EV-10: a blown-up window is a failed seed, not a crash or a gap."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST

    reference = evaluation.window_statistics(
        evaluation._load_native_window(COARSE_TRUTH, 1, 3))

    def bind(whole_run):
        # days 1-7: window (4, 6) reads day 7 for its last sampling check
        return _dual_archive(tmp_path, range(1, 8), nan=[(5, "u_m_s")], manifest={
            "days": 7, "completed_days": 7,
            "namelist": SWG_DG50_NAMELIST._asdict(),
            "numerical_health": whole_run})

    healthy = {"all_finite": True, "minimum_total_layer_thickness_m": 497.0,
               "maximum_u_courant": 0.02, "maximum_v_courant": 0.01}
    rows = evaluation.score_free_run(
        tmp_path, expected_manifest_sha256=bind(healthy), reference=reference,
        windows=[(1, 3), (4, 6)])
    assert rows[0]["mean_face_transport_rmse_m3_s"] == 0.0
    assert rows[0]["physical_health"]["passes"]
    assert rows[1]["physical_health"] == {"all_finite": False, "passes": False}
    with pytest.raises(ValueError, match="bound SHA-256"):
        evaluation.score_free_run(
            tmp_path, expected_manifest_sha256="0" * 64, reference=reference,
            windows=[(1, 3)])
    # a blow-up outside the scored windows still fails the seed
    late_blow_up = dict(healthy, maximum_v_courant=2.0)
    rows = evaluation.score_free_run(
        tmp_path, expected_manifest_sha256=bind(late_blow_up),
        reference=reference, windows=[(1, 3)])
    assert not rows[0]["physical_health"]["passes"]


def _closure(root, *, seed, objective="symmetric", window_days=4,
             checkpoint=None):
    """A closure block in the one free-run schema; its checkpoint is a real
    file under ``root`` holding ``checkpoint`` (default: the root's name)."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "closure.eqx"
    path.write_bytes(root.name.encode() if checkpoint is None else checkpoint)
    return {"run_id": f"test/seed{seed}", "experiment": "test",
            "config_sha256": "a" * 64, "dataset_contract_sha256": "b" * 64,
            "network_seed": seed, "window_days": window_days,
            "objective": objective, "wall_ring": "excluded",
            "network_spec": dict.fromkeys(SWGNetworkSpec._fields, 1),
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": evaluation._sha256(path)}


def _trained_run(root, *, seed, checkpoint=None, objective="symmetric",
                 window_days=4, day_digest=None, edit=None,
                 digest_key="daily_sha256"):
    """A minimal trained-closure free-run manifest (the binding reads only it).

    ``digest_key`` is where the day-file digests go: either key the loader
    accepts, or None for a manifest that pins no day files; ``edit`` changes
    the manifest before it is written."""
    closure = _closure(root, seed=seed, objective=objective,
                       window_days=window_days, checkpoint=checkpoint)
    digest = day_digest or hashlib.sha256(root.name.encode()).hexdigest()
    pins = {} if digest_key is None else {digest_key: {"day0001.npz": digest}}
    manifest = {"schema": FREE_RUN_MANIFEST_SCHEMA, "status": "complete",
                "closure": closure, **pins}
    if edit is not None:
        edit(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, evaluation._sha256(root / "manifest.json")


def test_trained_score_requires_three_distinct_identified_runs(tmp_path,
                                                              monkeypatch):
    """Codex round-6 defect 3: one run under three labels never passes, and
    every manifest must follow the one free-run schema (checkpoint file and
    digest, network seed, objective, window), matching the request; round 8:
    the checkpoint file must exist with that digest.  No plant reaches the
    scorer."""
    objective = "symmetric"
    scored = []
    monkeypatch.setattr(evaluation, "freeze_record", lambda *args, **kwargs: (
        evaluation.frozen_gate(objective), None, {}))
    monkeypatch.setattr(evaluation, "score_free_run", lambda root, **kwargs: (
        scored.append(root) or [dict(row, scored_data_sha256=f"{root}/{k}")
                                for k, row in enumerate(
                                    evaluation.truth_as_seed(objective))]))
    runs = {f"s{k}": _trained_run(tmp_path / f"run{k}", seed=k)
            for k in range(3)}
    missing_checkpoint = _trained_run(tmp_path / "gone", seed=40)
    (tmp_path / "gone" / "closure.eqx").unlink()
    changed_checkpoint = _trained_run(tmp_path / "changed", seed=41)
    (tmp_path / "changed" / "closure.eqx").write_bytes(b"another network")

    def round7_block(manifest):
        closure = manifest["closure"]
        closure["checkpoint"] = {"path": closure.pop("checkpoint_path"),
                                 "sha256": closure.pop("checkpoint_sha256")}

    def score(seeds, **request):
        return evaluation.score_trained_closure(
            seeds, **({"objective": objective, "window_days": 4} | request))

    (tmp_path / "alias").symlink_to(tmp_path / "run0")
    (tmp_path / "copy0").mkdir()
    (tmp_path / "copy0" / "manifest.json").write_bytes(
        (tmp_path / "run0" / "manifest.json").read_bytes())
    plants = {
        "reviewer: one run, three labels": (
            dict.fromkeys(("a", "b", "c"), runs["s0"]), "share a directory"),
        "symlinked root": (
            runs | {"s3": (tmp_path / "alias", runs["s0"][1])},
            "share a directory"),
        "copied manifest": (
            runs | {"s3": (tmp_path / "copy0", runs["s0"][1])},
            "share a manifest_sha256"),
        "duplicate network seed": (runs | {"s3": _trained_run(
            tmp_path / "dup_seed", seed=0)}, "share a network_seed"),
        "duplicate checkpoint": (runs | {"s3": _trained_run(
            tmp_path / "dup_checkpoint", seed=9, checkpoint=b"run0")},
            "share a checkpoint_sha256"),
        "copied day files, relabelled": (runs | {"s3": _trained_run(
            tmp_path / "dup_days", seed=8,
            day_digest=hashlib.sha256(b"run0").hexdigest())},
            "share day files"),
        # round-7 verifier: one run's day files under three relabelled
        # manifests whose digests sit under the loader's second key
        "copied day files, second digest key": ({f"o{k}": _trained_run(
            tmp_path / f"out{k}", seed=20 + k,
            day_digest=hashlib.sha256(b"one run").hexdigest(),
            digest_key="output_daily_sha256") for k in range(3)},
            "share day files"),
        "no day files pinned": (runs | {"s0": _trained_run(
            tmp_path / "no_pins", seed=30, digest_key=None)},
            "pins no day files"),
        "malformed checkpoint digest": (runs | {"s0": _trained_run(
            tmp_path / "bad_checkpoint", seed=5, edit=lambda manifest:
            manifest["closure"].update(checkpoint_sha256="xyz"))},
            "not a SHA-256"),
        "boolean seed": (runs | {"s0": _trained_run(
            tmp_path / "bad_seed", seed=True)}, "non-negative int"),
        "objective differs": (runs | {"s0": _trained_run(
            tmp_path / "bad_objective", seed=6, objective="template")},
            "score request"),
        "window differs": (runs | {"s0": _trained_run(
            tmp_path / "bad_window", seed=6, window_days=8)}, "score request"),
        "two runs": ({key: runs[key] for key in ("s0", "s1")}, "three"),
        # round 8: the checkpoint file is read, and the schema is the one
        "checkpoint file missing": (runs | {"s0": missing_checkpoint},
                                    "checkpoint file is missing"),
        "checkpoint file changed": (runs | {"s0": changed_checkpoint},
                                    "differs from checkpoint_sha256"),
        "round-7 checkpoint block": (runs | {"s0": _trained_run(
            tmp_path / "round7_block", seed=42, edit=round7_block)},
            "exactly the keys"),
        "another manifest schema": (runs | {"s0": _trained_run(
            tmp_path / "round7_schema", seed=43, edit=lambda manifest:
            manifest.update(schema="legoesm-swg-online-free-run-v1"))},
            "manifest schema"),
        # round-7 verifier, minor 1: a list crashed instead of refusing
        "day-file digests listed": (runs | {"s0": _trained_run(
            tmp_path / "listed", seed=44, edit=lambda manifest:
            manifest.update(daily_sha256=["0" * 64]))}, "not a mapping"),
    }
    for key in FREE_RUN_CLOSURE_KEYS:
        plants[f"no {key}"] = (runs | {"s0": _trained_run(
            tmp_path / f"no_{key}", seed=7, edit=lambda manifest, key=key:
            manifest["closure"].pop(key))}, "exactly the keys")
    for name, (seeds, match) in plants.items():
        with pytest.raises(ValueError, match=match):
            score(seeds)
            pytest.fail(name)
    with pytest.raises(ValueError, match="window_days must be one of"):
        score(runs, window_days=3)
    assert not scored
    report = score(runs)  # three distinct, identified runs are scored
    assert report["gate"]["pass"] and len(scored) == 3
    assert report["window_days"] == 4
    assert sorted(item["network_seed"]
                  for item in report["seed_bindings"].values()) == [0, 1, 2]


@real_fields
def test_runs_with_identical_scored_data_are_refused(tmp_path, monkeypatch):
    """Round-7 verifier's plant: one run saved three times, each copy with one
    extra unused entry so every day-file digest differs, under manifests with
    distinct seeds and checkpoints.  The binding admits it (bytes differ); the
    score refuses it on the loaded window arrays.  Three runs whose data
    differ (fields scaled by 1 + k*1e-9) are scored."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_DG50_NAMELIST

    objective = "symmetric"
    frozen = dict(evaluation.frozen_gate(objective), windows=[[400, 402]])
    monkeypatch.setitem(evaluation.FROZEN_GATE, objective, frozen)
    reference = evaluation.window_statistics(_real_window(400, 3))
    monkeypatch.setattr(evaluation, "freeze_record", lambda *args, **kwargs: (
        frozen, None, reference))

    def run(k, **archive):
        root = tmp_path / f"run{k}"
        manifest = {
            "schema": FREE_RUN_MANIFEST_SCHEMA,
            "namelist": SWG_DG50_NAMELIST._asdict(),
            "numerical_health": {
                "all_finite": True, "minimum_total_layer_thickness_m": 497.0,
                "maximum_u_courant": 0.02, "maximum_v_courant": 0.01},
            "closure": _closure(root, seed=k, objective=objective)}
        return root, _dual_archive(root, range(400, 404), manifest=manifest,
                                   **archive)

    copies = {f"s{k}": run(k, extra={"pad": np.int64(k)}) for k in range(3)}
    assert len(evaluation.bind_trained_runs(
        copies, objective=objective, window_days=4)) == 3
    with pytest.raises(ValueError, match="hold identical data"):
        evaluation.score_trained_closure(copies, objective=objective,
                                         window_days=4)
    distinct = {f"d{k}": run(10 + k, scale=1.0 + k * 1e-9) for k in range(3)}
    report = evaluation.score_trained_closure(distinct, objective=objective,
                                              window_days=4)
    assert len({row["scored_data_sha256"]
                for rows in report["gate"]["raw_seed_scores"].values()
                for row in rows}) == 3


@real_fields
@pytest.mark.parametrize("schema", (None, "legoesm-swg-double-gyre-daily-v1"))
def test_reviewer_renamed_daily_mean_payload_is_refused(tmp_path, schema):
    """Codex round-6 defect 4, the reviewer's own plant: a daily-mean payload
    under snapshot names, manifest saying daily_snapshot.  Without a per-file
    label it is refused; with a forged single-sampling label it is refused
    because nothing in a single-sampling file can show its sampling."""
    with np.load(COARSE_TRUTH.parent / "daily_means" / "day0001.npz") as saved:
        means = [np.asarray(saved[key]) for key in evaluation.DAILY_MEAN_FIELDS]
    path = tmp_path / "day0001.npz"
    extra = {} if schema is None else {"schema": np.str_(schema)}
    np.savez(path, day=np.int64(1), ssh_m=means[0], u_m_s=means[1],
             v_m_s=means[2], **extra)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "sampling": "daily_snapshot",
        "daily_sha256": {path.name: evaluation._sha256(path)}}))
    match = ("no known per-file sampling label" if schema is None
             else "single-sampling archive cannot show")
    with pytest.raises(ValueError, match=match):
        evaluation._load_native_window(tmp_path, 1, 1)


@real_fields
def test_daily_mean_product_is_never_read_as_snapshots():
    with pytest.raises(ValueError, match="carries daily-mean fields"):
        evaluation._load_native_window(COARSE_TRUTH.parent / "daily_means", 1, 2)


@real_fields
def test_dual_archive_snapshots_are_shown_from_the_data(tmp_path):
    """A daily mean under snapshot names is refused from the data, field by
    field, even with every per-file label intact; genuine snapshots pass."""
    days = range(400, 407)
    _dual_archive(tmp_path / "genuine", days)
    loaded = evaluation._load_native_window(tmp_path / "genuine", 400, 6)
    assert all(np.array_equal(a, b) for a, b in zip(
        loaded, _real_window(400, 6), strict=True))
    for renamed in (evaluation.SNAPSHOT_FIELDS, ("ssh_m",), ("u_m_s",),
                    ("v_m_s",)):
        root = tmp_path / ("renamed_" + "_".join(renamed))
        _dual_archive(root, days, rename=renamed)
        field = renamed[0].split("_")[0]
        with pytest.raises(ValueError, match=f"{field} is not an end-of-day"):
            evaluation._load_native_window(root, 400, 6)
    for k, (labels, match) in enumerate((
            ({"snapshot_sampling": None}, "dual-file labels"),
            ({"daily_mean_sampling": None}, "dual-file labels"),
            ({"snapshot_sampling": "daily mean"}, "dual-file labels"),
            ({"schema": "legoesm-swg-double-gyre-daily-v3"},
             "no known per-file sampling label"))):
        root = tmp_path / f"plant{k}"
        _dual_archive(root, days, labels=labels)
        with pytest.raises(ValueError, match=match):
            evaluation._load_native_window(root, 400, 6)


@real_fields
@pytest.mark.parametrize("kwargs, match", (
    # the round-7 verifier's plant: daily means under the snapshot names, one
    # NaN in every day's daily-mean fields (round 7 skipped and ADMITTED it)
    ({"rename": evaluation.SNAPSHOT_FIELDS,
      "nan": [(day, key) for day in range(400, 407)
              for key in evaluation.DAILY_MEAN_FIELDS]},
     "ssh snapshot is finite but its own day's daily mean is not"),
    # genuine snapshots beside one non-finite daily mean: own day (day 400),
    # next day (day 406, read for day 405)
    ({"nan": [(400, "v_daily_mean_m_s")]},
     "day0400.npz: the v snapshot is finite but its own day's"),
    ({"nan": [(406, "u_daily_mean_m_s")]},
     "day0405.npz: the u snapshot is finite but the next day's"),
), ids=("renamed_mean_with_nan", "own_day_mean_nan", "next_day_mean_nan"))
def test_finite_snapshot_beside_a_nonfinite_daily_mean_is_refused(
        tmp_path, kwargs, match):
    """Only a non-finite SNAPSHOT skips the data test; its window then fails
    physical health (the free-run scoring test above: NaN in day 5's u
    snapshot).  A finite snapshot beside a non-finite daily mean of its own
    or the next day is refused, never skipped."""
    _dual_archive(tmp_path, range(400, 407), **kwargs)
    with pytest.raises(ValueError, match=match):
        evaluation._load_native_window(tmp_path, 400, 6)


@real_fields
@pytest.mark.skipif(not (evaluation.FINE_TRUTH_ROOT / "manifest.json").is_file(),
                    reason="registered fine truth not mounted")
def test_fine_truth_snapshots_are_shown_from_the_data():
    """The gate truth's dual files pass the data check (days 366-367 read
    368 too) and restrict to the stored round-5 box truth bit for bit."""
    restricted = evaluation.restrict_fine_truth_window(
        evaluation.FINE_TRUTH_ROOT, 366, 367, objective="symmetric")
    assert all(np.array_equal(a, b) for a, b in zip(
        restricted, _real_window(366, 2), strict=True))


# SHA-256 of the gate truth of fine day 366, as the frozen records were
# measured (round 7).  A change here -- like the round-6 GPU fix's
# matrix-product filter, which moved the template record by 5.6e-15 -- means
# ``freeze --objective <name> --verify`` will refuse: re-measure first.
FROZEN_OPERATOR_DIGESTS = {
    "symmetric": "a02f390353973122b86c8745ab227aa89c903cb9ca24d1e0c7372b028277c7f3",
    "template": "0c1e9367922d8b02eb043a48ae35aeed612aafdcdc9fdde252a9b24df3f9b436",
}


@pytest.mark.skipif(not (evaluation.FINE_TRUTH_ROOT / "manifest.json").is_file(),
                    reason="registered fine truth not mounted")
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_truth_operator_arithmetic_is_the_frozen_records(objective):
    arrays = evaluation.restrict_fine_truth_window(
        evaluation.FINE_TRUTH_ROOT, 366, 366, objective=objective)
    digest = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes()
                                     for a in arrays)).hexdigest()
    assert digest == FROZEN_OPERATOR_DIGESTS[objective]


@pytest.mark.skipif(not (evaluation.FINE_TRUTH_ROOT / "manifest.json").is_file(),
                    reason="registered fine truth not mounted")
def test_fine_truth_renamed_payload_is_refused(tmp_path):
    """Real fine days 366-368 with day 367's daily mean put under its snapshot
    names, every label intact and every file re-pinned: refused."""
    manifest = json.loads((evaluation.FINE_TRUTH_ROOT / "manifest.json").read_text())
    pins = {}
    for day in (366, 367, 368):
        with np.load(evaluation.FINE_TRUTH_ROOT / f"day{day:04d}.npz") as saved:
            arrays = {key: saved[key] for key in saved.files}
        if day == 367:
            for snap_key, mean_key in zip(evaluation.SNAPSHOT_FIELDS,
                                          evaluation.DAILY_MEAN_FIELDS,
                                          strict=True):
                arrays[snap_key] = arrays[mean_key]
        path = tmp_path / f"day{day:04d}.npz"
        np.savez(path, **arrays)
        pins[path.name] = evaluation._sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"status": "complete", "completed_days": 368,
         "namelist": manifest["namelist"], "daily_sha256": pins}))
    with pytest.raises(ValueError, match="day0367.npz: ssh is not an end-of-day"):
        evaluation.restrict_fine_truth_window(tmp_path, 366, 367,
                                              objective="symmetric")


def test_available_windows_need_the_next_day_pinned(monkeypatch):
    for completed, count in ((1095, 0), (1096, 1), (6935, 8), (6936, 9)):
        monkeypatch.setattr(evaluation, "_fine_truth_manifest",
                            lambda root, c=completed: {"completed_days": c})
        assert evaluation.available_windows() == \
            evaluation.registered_windows(count), completed


@real_fields
def test_round5_control_record_is_reproduced_exactly():
    """Item 6: the frozen nu=10 years-2-3 record stays reproducible."""
    scores, _ = evaluation.score_control(evaluation.DEFAULT_ROUND2,
                                         control="nu10")
    assert scores["basin_eke_ratio"] == 0.7323411189244107
    assert scores["ke_spectrum_relative_l1"] == 0.22300407134605818
    assert scores["native_grid_noise_index"] == 0.9317922221566435

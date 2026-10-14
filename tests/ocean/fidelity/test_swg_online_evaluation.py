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

SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
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
def test_control_level_noise_within_sampling_spread_keeps_health(objective):
    """EV-N2: noise at the worse control's level plus jitter within that
    control's own window-to-window half-range is not a failure; beyond it is.
    The worse control (nu10) is also checked against its own windows."""
    frozen = evaluation.frozen_gate(objective)
    nu10 = frozen["controls"]["nu10"]
    spread = evaluation.gate_bars(nu10)
    worse = {metric: float(np.median(nu10[metric]))
             for metric in evaluation.NOISE_METRICS}
    assert all(worse[metric] > float(np.median(frozen["controls"]["nu80"][metric]))
               for metric in evaluation.NOISE_METRICS)
    truth = evaluation.truth_as_seed(objective)
    for factor, expected in ((0.5, True), (1.5, False)):
        rows = [dict(row, **{
            metric: worse[metric] + factor * spread[metric]["half_range"]
            for metric in evaluation.NOISE_METRICS}) for row in truth]
        result = evaluation.held_out_gate(_seeds(rows), objective=objective)
        assert result["spectral_health_pass"] is expected
    assert evaluation.held_out_gate(
        _seeds(evaluation.control_as_seed(objective, "nu10")),
        objective=objective)["spectral_health_pass"]


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

    pins = {}
    for day in range(1, 7):
        with np.load(COARSE_TRUTH / f"day{day:04d}.npz") as saved:
            fields = {name: np.asarray(saved[name])
                      for name in ("ssh_m", "u_m_s", "v_m_s")}
        if day == 5:
            fields["u_m_s"][10, 10] = np.nan
        path = tmp_path / f"day{day:04d}.npz"
        np.savez(path, **fields)
        pins[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = evaluation.window_statistics(
        evaluation._load_native_window(COARSE_TRUTH, 1, 3))

    def bind(whole_run):
        (tmp_path / "manifest.json").write_text(json.dumps({
            "status": "complete", "days": 6, "completed_days": 6,
            "sampling": "daily_snapshot", "daily_sha256": pins,
            "namelist": SWG_DG50_NAMELIST._asdict(),
            "numerical_health": whole_run}))
        return evaluation._sha256(tmp_path / "manifest.json")

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


@real_fields
def test_gate_never_reads_daily_means(tmp_path):
    """Snapshots and daily means are different quantities and never mixed."""
    daily_means = COARSE_TRUTH.parent / "daily_means"
    with pytest.raises(ValueError, match="no labelled end-of-day snapshot"):
        evaluation._load_native_window(daily_means, 1, 2)
    with np.load(COARSE_TRUTH / "day0001.npz") as saved:
        fields = {name: np.asarray(saved[name])
                  for name in ("ssh_m", "u_m_s", "v_m_s")}
    path = tmp_path / "day0001.npz"
    np.savez(path, snapshot_sampling=np.str_("daily mean"), **fields)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "status": "complete", "sampling": "daily_mean",
        "daily_sha256": {path.name: evaluation._sha256(path)}}))
    with pytest.raises(ValueError, match="no labelled end-of-day snapshot"):
        evaluation._load_native_window(tmp_path, 1, 1)


@real_fields
def test_round5_control_record_is_reproduced_exactly():
    """Item 6: the frozen nu=10 years-2-3 record stays reproducible."""
    scores, _ = evaluation.score_control(evaluation.DEFAULT_ROUND2,
                                         control="nu10")
    assert scores["basin_eke_ratio"] == 0.7323411189244107
    assert scores["ke_spectrum_relative_l1"] == 0.22300407134605818
    assert scores["native_grid_noise_index"] == 0.9317922221566435

"""Direct tests for the SWG online balanced-start instrument."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from legoesm.training.swg_online import closed_wall_divergent_part

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity"
         / "testcases/nemo_testcase_l1_swg_online_balanced_start.py")
_SPEC = importlib.util.spec_from_file_location("swg_balanced_start", _PATH)
balanced = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(balanced)


def test_divergent_projection_recovers_closed_wall_gradient_and_potential():
    n = 8
    spacing = 2.0
    y, x = np.mgrid[:n, :n]
    phi = (np.cos(np.pi * (x + 0.5) / n)
           * np.cos(2.0 * np.pi * (y + 0.5) / n))
    u = np.zeros((n + 2, n + 2))
    v = np.zeros_like(u)
    mask = np.zeros_like(u, dtype=bool)
    mask[1:-1, 1:-1] = True
    u[1:-1, 1:-2] = np.diff(phi, axis=1) / spacing
    v[1:-2, 1:-1] = np.diff(phi, axis=0) / spacing
    potential, u_div, v_div = closed_wall_divergent_part(u, v, mask, spacing)
    np.testing.assert_allclose(u_div, u, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(v_div, v, rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(potential[1:-1, 1:-1], phi - phi.mean(),
                               rtol=0.0, atol=2.0e-14)
    assert np.all(potential[~mask] == 0.0)
    assert balanced.fast_divergent_energy(u, v, mask, spacing) > 0.0
    # A non-conservative wall-normal plant must change the projection.
    planted = u.copy()
    planted[1:-1, -2] = 1.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            closed_wall_divergent_part(planted, v, mask, spacing)[1], planted,
            rtol=0.0, atol=2.0e-15)


def test_initialization_ratio_uses_distinct_shock_and_closure_errors():
    wet = np.ones((2, 2), dtype=bool)
    conservative = np.zeros((48, 2, 2))
    target = np.full_like(conservative, 2.0)
    raw = np.full_like(conservative, 3.0)
    ratios = balanced.initialization_ratios(
        raw, conservative, target, wet)
    assert set(ratios) == {"6", "24", "48"}
    assert ratios["24"]["initialization_shock_vorticity_rms_s-1"] == 3.0
    assert ratios["24"]["genuine_closure_vorticity_rms_s-1"] == 2.0
    assert ratios["24"]["shock_to_closure_ratio"] == 1.5
    # Planting equality between raw and conservative must collapse shock to zero.
    planted = balanced.initialization_ratios(
        conservative, conservative, target, wet)
    assert planted["24"]["shock_to_closure_ratio"] == 0.0


def _series(**changes):
    values = {"error": np.linspace(1.0, 48.0, 48),
              "fast": np.full(48, 3.0), "truth_fast": np.full(48, 2.0),
              "ssh_tendency": np.ones(48)}
    return values | changes


def test_admission_is_relative_to_same_operator_truth_and_fails_closed():
    assert balanced.state_admission(**_series())["admitted"]
    # Persistent excess the old early/late rule could not see: 8x the truth
    # at every hour has max(h1-6) == mean(h25-48).
    persistent = balanced.state_admission(**_series(fast=np.full(48, 16.0)))
    assert not persistent["admitted"]
    assert persistent["fast_energy_worst_hour1_6_over_truth"] == 8.0
    early = _series()
    early["fast"] = early["fast"].copy()
    early["fast"][3] = 4.2
    assert not balanced.state_admission(**early)["admitted"]
    late = _series(fast=np.concatenate((np.full(24, 3.0), np.full(24, 4.2))))
    assert not balanced.state_admission(**late)["admitted"]
    jump = _series(error=np.linspace(1.0, 48.0, 48) * 1.0e-8)
    jump["error"][0] = 3.0e-8
    verdict = balanced.state_admission(**jump)
    assert not verdict["no_jump"] and not verdict["admitted"]
    dead_truth = _series(truth_fast=np.zeros(48))
    assert not balanced.state_admission(**dead_truth)["admitted"]
    assert not balanced.state_admission(
        **_series(), rebalance={"failed": "did not converge"})["admitted"]
    assert not balanced.state_admission(
        **_series(), rebalance={"volume_change_m3": 2.0},
        volume_bound_m3=1.0)["admitted"]
    assert balanced.state_admission(
        **_series(), rebalance={"volume_change_m3": 0.5},
        volume_bound_m3=1.0)["admitted"]
    with pytest.raises(ValueError, match="48 finite non-negative"):
        balanced.state_admission(**_series(fast=np.full(48, np.nan)))
    with pytest.raises(ValueError, match="48 finite non-negative"):
        balanced.state_admission(**_series(error=-np.linspace(1.0, 48.0, 48)))


def _write_dual_segment(root: Path, *, source_day: int, shift_day2=False):
    """A two-day schema-v2 dual segment with truth provenance, as written."""
    import hashlib

    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    root.mkdir()
    shape = (6, 6)
    hashes = {}
    for day in (1, 2):
        hours = (source_day + day - 1) * 24 + np.arange(1, 25, dtype=np.int64)
        if shift_day2 and day == 2:
            hours = hours + 1
        arrays = {"absolute_hour": hours}
        for prefix in ("box_", "filtered_"):
            for name, unit in (("ssh", "m"), ("u", "m_s"), ("v", "m_s")):
                arrays[f"{prefix}{name}_hourly_{unit}"] = np.broadcast_to(
                    hours[:, None, None], (24,) + shape).astype(float)
                if day == 1:
                    arrays[f"{prefix}{name}_initial_{unit}"] = np.full(
                        shape, float(source_day * 24))
        path = root / f"day{day:04d}.npz"
        np.savez(path, **arrays)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    namelist = SWG_MADDISON_NAMELIST._replace(
        nn_itend=2 * 1728, nn_write=1728, nn_stock=1728)
    truth = root.parent / "truth"
    truth.mkdir(exist_ok=True)
    (truth / "m.json").write_text(json.dumps(
        {"daily_sha256": {f"day{source_day:04d}.npz": "a" * 64}}))
    (root / "manifest.json").write_text(json.dumps({
        "schema": "legoesm-swg-double-gyre-hourly-target-v2",
        "status": "complete", "days": 2, "completed_days": 2,
        "sampling": "hourly_dual_coarse_target",
        "hourly_samples_per_day": 24, "hourly_target_deck": "dg50",
        "precision_policy": "fp64",
        "restart": {"source_day": source_day, "perturb_amplitude_m": None,
                    "sha256": "a" * 64,
                    "path": str(root.parent / "truth" / f"day{source_day:04d}.npz"),
                    "manifest_admission": {
                        "manifest_path": str(root.parent / "truth" / "m.json"),
                        "manifest_status": "complete",
                        "snapshot_sha256": "a" * 64}},
        "restart_next_verification": {"result": {"all_equal": True}},
        "numerical_health": {"all_finite": True},
        "namelist": json.loads(json.dumps(namelist._asdict())),
        "daily_sha256": hashes,
    }))


def test_hourly_target_loader_checks_lineage_through_the_lane_loader(tmp_path):
    _write_dual_segment(tmp_path / "good", source_day=731)
    target, initial, provenance = balanced._load_hourly_target(
        tmp_path / "good", operator="template")
    assert target["T"].shape == (48, 6, 6)
    assert np.all(target["U"][0] == 731 * 24 + 1)
    assert np.all(target["V"][47] == 731 * 24 + 48)
    assert np.all(initial["T"] == 731 * 24)
    assert provenance["operator"] == "template"
    _write_dual_segment(tmp_path / "shifted", source_day=731, shift_day2=True)
    with pytest.raises(ValueError, match="lineage"):
        balanced._load_hourly_target(tmp_path / "shifted", operator="template")


@pytest.fixture
def restore_precision_policy():
    from legoesm.core.precision import get_policy, set_policy

    old = get_policy()
    yield
    set_policy(old)


def test_dual_operator_start_evaluation_runs_the_real_body(
        restore_precision_policy):
    """Truth = the model's own forecast from the rebalanced start.

    The rebalanced start then reproduces its truth to the bit (fast-energy
    ratio exactly one, admitted); the stored start, with a 5 cm sea-level
    step at the west wall, sheds gravity waves and is rejected.
    """
    context = balanced.build_dual_context()
    card = context["card"]
    geometry = context["closure_geometry"]
    wet = context["wet"]
    j0, j1, i0, i1 = geometry.wet_bounds
    y, x = np.meshgrid(np.arange(1, j1 - j0 + 1.0), np.arange(1, i1 - i0 + 1.0),
                       indexing="ij")
    psi = np.zeros(wet.shape)
    psi[j0:j1, i0:i1] = 3.0e5 * (
        np.sin(np.pi * x / (i1 - i0 + 1)) * np.sin(2.0 * np.pi * y / (j1 - j0 + 1)))
    u = -(psi - np.roll(psi, 1, axis=0)) / geometry.spacing_m
    v = (psi - np.roll(psi, 1, axis=1)) / geometry.spacing_m
    u = u * np.asarray(geometry.umask)
    v = v * np.asarray(geometry.vmask)
    ssh = np.zeros(wet.shape)
    ssh[j0:j1 + 1, i0] = 0.05
    stored = {"T": ssh, "U": u, "V": v}
    balanced_ssh, _ = context["rebalance"](ssh, u, v)
    start = balanced._coarse_state_from_fields(
        card.recipe.initial_state, context["forcing"],
        {"T": balanced_ssh, "U": u, "V": v})
    _, samples = context["run_48h"](start)
    truth = {kind: np.asarray(value) for kind, value in
             zip(("T", "U", "V"), samples, strict=True)}
    record = balanced.evaluate_operator_start(
        context, truth, stored, objective="template")
    admission = record["admission"]
    assert admission["admitted"], admission["failed"]
    assert admission["fast_energy_worst_hour1_6_over_truth"] == 1.0
    assert admission["fast_energy_mean25_48_over_truth"] == 1.0
    assert not record["stored_start_admission"]["admitted"]
    assert record["stored_start_admission"][
        "fast_energy_worst_hour1_6_over_truth"] > 2.0
    assert record["rebalance"]["ssh_change_max_abs_m"] > 0.01
    assert record["truth_fast_divergent_ke_m2_s-2"]["max_hours1_6"] > 0.0
    assert record["truth_relative_vorticity_rms_s-1"]["24"] > 0.0
    checkerboard = record["checkerboard_0.1m_s_eq7_loss"]
    assert checkerboard["weighted"] > checkerboard["excluded"] > 0.0


def _record(box: bool, filtered: bool) -> dict:
    return {"operators": {
        "box": {"admission": {"admitted": box},
                "stored_start_admission": {"admitted": False}},
        "filtered": {"admission": {"admitted": filtered},
                     "stored_start_admission": {"admitted": False}}}}


def test_dual_summary_counts_and_never_selects_an_objective():
    summary = balanced.summarize_dual(
        [_record(True, True)] * 9 + [_record(True, False)])
    assert summary["rebalanced_pass_count"] == {"box": 10, "filtered": 9}
    assert summary["stored_start_pass_count"] == {"box": 0, "filtered": 0}
    assert not summary["all_rebalanced_starts_admitted"]
    assert "selected_objective" not in summary
    assert summary["objective_selection"].startswith("owner decision")


def test_balanced_start_cli_exits_nonzero_on_failed_admission(
        monkeypatch, tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text(json.dumps({"records": []}))
    monkeypatch.setattr(
        balanced, "apply_registered_admission",
        lambda report, root: report | {"admission_summary": {
            "conservative_stored_start_admitted": False,
            "conservative_pass_count": 9,
            "raw_pass_count": 0,
        }})
    assert balanced.main([
        "--existing-json", str(existing), "--hourly-root", str(tmp_path),
        "--json", str(tmp_path / "out.json")]) == 1


@pytest.mark.parametrize("filtered,expected", [(True, 0), (False, 1)])
def test_dual_operator_cli_exit_code_is_admission_not_selection(
        monkeypatch, tmp_path, filtered, expected):
    records = [_record(True, True)] * 9 + [_record(True, filtered)]
    monkeypatch.setattr(
        balanced, "run_dual_operator_experiment",
        lambda *args, **kwargs: {"summary": balanced.summarize_dual(records)})
    output = tmp_path / f"out{filtered}.json"
    assert balanced.main([
        "--dual-operator", "--hourly-root", str(tmp_path),
        "--json", str(output)]) == expected
    assert "selected_objective" not in output.read_text()

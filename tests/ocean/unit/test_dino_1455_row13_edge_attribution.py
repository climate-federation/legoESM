"""Direct tests for the #1455 row-13/49 edge-attribution instrument."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = ROOT / "scripts/validate/ocean_fidelity/dino_1226"
sys.path.insert(0, str(PROBE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "row13_edge_attribution", PROBE_DIR / "row13_edge_attribution.py"
)
E = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E)


def test_exact_reducer_pins_and_worktree_imports():
    assert E.verify_upstream_hashes() == {
        "regional_audit": E.UPSTREAM_REGIONAL_SHA256,
        "channel_rescore": E.UPSTREAM_CHANNEL_SHA256,
    }
    locations = E.verify_import_locations()
    assert str(ROOT) in locations["legoesm_constants"]
    assert str(ROOT) in locations["legoesm_ocean"]


def test_depth_zonal_and_component_closures_fire_on_plants():
    rng = np.random.default_rng(1349)
    u = rng.normal(size=(E.R.A.NY, E.R.A.NX, E.R.A.NZ))
    for row in E.TARGET_ROWS:
        total = E.R.row_transports(u, E.R.A.umask)[row]
        depth = E.depth_contributions(u, row)
        zonal = E.zonal_contributions(u, row)
        bt, bc = E.one_row_bt_bc(u, row)
        np.testing.assert_allclose(depth.sum(), total, atol=1e-9)
        np.testing.assert_allclose(zonal.sum(), total, atol=1e-9)
        np.testing.assert_allclose(bt + bc, total, atol=1e-9)
        with pytest.raises(AssertionError, match="closure failed"):
            E._assert_close(depth.sum() + 1e-3, total, 1e-9)


def test_blocked_partition_rejects_nonzero_blocked_column():
    for row in E.TARGET_ROWS:
        classes, blocked, open_col = E.zonal_classes(row)
        joined = np.concatenate(list(classes.values()))
        assert len(joined) == E.R.A.NX - 4
        assert len(np.unique(joined)) == len(joined)
        assert len(blocked) == 3
        assert int(open_col.sum()) == 49
        contribution = np.zeros(E.R.A.NX)
        assert len(classes["blocked"]) == 0
        contribution[blocked[0]] = 1.0
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(contribution[blocked], np.zeros(len(blocked)))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.85, 0.70, 0.20, 0.20), "DESCRIPTIVE_BASIN_EDGE_MATCH_LOW_N"),
        ((0.20, 0.20, 0.85, 0.70), "DESCRIPTIVE_CHANNEL_EDGE_MATCH_LOW_N"),
        ((0.20, 0.20, 0.25, 0.25), "DESCRIPTIVE_OWN_OBJECT_NEITHER_LOW_N"),
        ((0.85, 0.70, 0.80, 0.70), "UNRESOLVED_LOW_N"),
    ],
)
def test_review_corrected_descriptive_routes_are_distinct(values, expected):
    got = E.relationship_status(*values)[0]
    assert got == expected


def test_opposite_neighbor_veto_blocks_wrong_side_route():
    status, basin_veto, _ = E.relationship_status(0.90, 0.70, 0.20, 0.95)
    assert basin_veto
    assert status == "UNRESOLVED_LOW_N"


def test_symmetry_has_amplitude_aware_row13_dominance_cell():
    assert E.symmetry_status(1.0, 0.85) == "DESCRIPTIVE_TOPOLOGY_SYMMETRIC_LOW_N"
    assert E.symmetry_status(1.5, 0.85) != "DESCRIPTIVE_TOPOLOGY_SYMMETRIC_LOW_N"
    assert E.symmetry_status(0.10, 0.20) == (
        "DESCRIPTIVE_ROW13_AMPLITUDE_DOMINANCE_UNSATURATED"
    )


def test_n4_exact_null_receipts_are_uniform():
    assert E.exact_n4_null_p(0.778) == pytest.approx(0.222)
    assert E.exact_n4_null_p(-0.905) == pytest.approx(0.095)
    assert E.EFFECTIVE_INDEPENDENT_MEMBERS == 1


def test_held_out_horizon_does_not_enter_its_template_and_is_rms_normalized():
    rng = np.random.default_rng(49)
    paired = {
        day: rng.normal(size=(E.N_MEM, E.R.A.NY)).astype(np.float64) for day in E.HORIZONS
    }
    _, folds_before = E._loo_profile_amplitudes(paired, E.C.CHANNEL_ROWS)
    changed = {day: values.copy() for day, values in paired.items()}
    changed[180][:, E.C.CHANNEL_ROWS] += 1e6
    _, folds_after = E._loo_profile_amplitudes(changed, E.C.CHANNEL_ROWS)
    fold_before = next(f for f in folds_before if f["held_out_day"] == 180)
    fold_after = next(f for f in folds_after if f["held_out_day"] == 180)
    np.testing.assert_array_equal(
        fold_before["template_mean_gap_sv"], fold_after["template_mean_gap_sv"]
    )
    assert fold_before["projection_divisor_row_sv"] == pytest.approx(
        fold_before["n_rows"] * fold_before["template_rms_sv"]
    )


def test_synthetic_power_control_runs_projection_and_pearson_pipeline():
    assert E._synthetic_pipeline_cases() == {
        "basin": "DESCRIPTIVE_BASIN_EDGE_MATCH_LOW_N",
        "channel": "DESCRIPTIVE_CHANNEL_EDGE_MATCH_LOW_N",
        "neither": "DESCRIPTIVE_OWN_OBJECT_NEITHER_LOW_N",
        "mixed": "UNRESOLVED_LOW_N",
    }


def test_upstream_hash_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(E, "UPSTREAM_CHANNEL_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="reducer SHA-256 mismatch"):
        E.verify_upstream_hashes()


def test_direct_self_test_reports_planted_failures(capsys):
    E.self_test(run_upstream=False)
    out = capsys.readouterr().out
    assert "PLANT FIRED" in out
    assert "SELF-TEST PASSED" in out

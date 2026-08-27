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
        scored_blocked = classes["blocked"]
        if len(scored_blocked):
            contribution[scored_blocked[0]] = 1.0
            with pytest.raises(AssertionError):
                np.testing.assert_array_equal(
                    contribution[scored_blocked], np.zeros(len(scored_blocked))
                )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((0.85, 0.70, 0.20, 0.20), "CONFIRMED_BASIN_EDGE_RELATIONSHIP"),
        ((0.20, 0.20, 0.85, 0.70), "CONFIRMED_CHANNEL_EDGE_RELATIONSHIP"),
        ((0.20, 0.20, 0.25, 0.25), "CONFIRMED_OWN_OBJECT_NEITHER_RELATIONSHIP"),
        ((0.85, 0.70, 0.80, 0.70), "UNRESOLVED_MIXED"),
    ],
)
def test_registered_attribution_outcomes_are_distinct(values, expected):
    high = np.asarray([0.9, 0.85, 0.8, 0.2])
    low = np.asarray([0.2, 0.1, 0.15, 0.25])
    basin_members = high if values[0] >= E.R_HIGH else low
    channel_members = high if values[2] >= E.R_HIGH else low
    got = E.relationship_status(
        values[0], values[1], values[2], values[3], basin_members, channel_members
    )[0]
    assert got == expected


def test_symmetry_control_cannot_pass_outside_amplitude_bar():
    members = np.asarray([0.9, 0.85, 0.8, 0.2])
    assert E.symmetry_status(1.0, 0.85, members, 3.0, 1.0)[0] == (
        "CONFIRMED_TOPOLOGY_SYMMETRIC_RESPONSE"
    )
    assert E.symmetry_status(1.5, 0.85, members, 3.0, 1.0)[0] != (
        "CONFIRMED_TOPOLOGY_SYMMETRIC_RESPONSE"
    )


def test_upstream_hash_mutation_is_rejected(monkeypatch):
    monkeypatch.setattr(E, "UPSTREAM_CHANNEL_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="reducer SHA-256 mismatch"):
        E.verify_upstream_hashes()


def test_direct_self_test_reports_planted_failures(capsys):
    E.self_test(run_upstream=False)
    out = capsys.readouterr().out
    assert "PLANT FIRED" in out
    assert "SELF-TEST PASSED" in out

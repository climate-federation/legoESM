"""Direct, oracle-independent tests for the DINO twin/NEMO map comparator."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.dino_1226 import twin_nemo_ts_maps as M


def test_one_ring_alignment_is_explicit_and_fail_closed():
    full = np.zeros((199, 52, 3), dtype=np.float64)
    cropped = M.crop_one_ring("synthetic", full)
    assert cropped.shape == (197, 50, 3)
    with pytest.raises(RuntimeError, match="expected full horizontal"):
        M.crop_one_ring("wrong", np.zeros((197, 50, 3)))


def test_planted_wet_cell_moves_statistics_and_dry_cell_does_not():
    receipt = M.planted_violation_self_test()
    assert receipt["passed"]
    assert receipt["before"]["max_abs_difference"] == 0.0
    assert receipt["after"]["max_abs_difference"] == 0.25
    assert receipt["after"]["rms_difference"] > 0.0
    assert receipt["dry_cell_excluded"]


def test_argmax_locator_reports_indices_coordinates_depth_and_sign():
    reference = np.zeros((2, 3, 4), dtype=np.float64)
    candidate = reference.copy()
    candidate[1, 2, 3] = -0.75
    candidate[0, 0, 0] = 9.0  # dry and therefore excluded
    wet = np.ones_like(candidate, dtype=bool)
    wet[0, 0, 0] = False
    lat = np.asarray([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
    lon = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    depth = np.broadcast_to(np.asarray([5.0, 15.0, 30.0, 50.0]), candidate.shape)
    peak = M.locate_abs_argmax(candidate, reference, wet, lat, lon, depth)
    assert peak["index_jik_map_zero_based"] == [1, 2, 3]
    assert peak["index_jik_full_zero_based"] == [2, 3, 3]
    assert peak["latitude_deg_north"] == 20.0
    assert peak["longitude_deg_east"] == 3.0
    assert peak["depth_m"] == 50.0
    assert peak["difference"] == -0.75
    assert peak["abs_difference"] == 0.75


def test_cli_self_test_requires_no_oracle_inputs(capsys):
    assert M.main(["--self-test"]) == 0
    assert '"passed": true' in capsys.readouterr().out

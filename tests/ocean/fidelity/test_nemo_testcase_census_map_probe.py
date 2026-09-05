"""Direct non-vacuity tests for the OVERFLOW water-mass-census anatomy probe.

The receipt's two load-bearing claims are (a) that the probe classifies cells
into exactly the scorer's half-open temperature bins and (b) that it separates
the two ways a volume FRACTION can move -- a cell changing temperature class,
and a cell keeping its class while its thickness changes.  Both are tested
here, each with a case that must produce the opposite answer.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

SCRIPT = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_census_map_probe.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_testcase_census_map_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_classify_reproduces_the_scorers_half_open_bins():
    values = np.asarray([10.0, 11.999, 12.0, 17.999, 18.0, 20.0, 9.5, 20.5])
    assert probe._classify(values).tolist() == [0, 0, 1, 1, 2, 2, -1, -1]


def _arm(temperature: np.ndarray, volume: np.ndarray) -> dict:
    active = np.ones(temperature.shape, dtype=bool)
    klass = probe._classify(temperature)
    total = float(np.sum(volume))
    census = np.asarray(
        [float(np.sum(volume[klass == index])) for index in range(3)]) / total
    nx = temperature.shape[1]
    card = SimpleNamespace(recipe=SimpleNamespace(
        grid=SimpleNamespace(dx_T=np.full((1, nx), 1000.0))))
    return {
        "census": census.tolist(),
        "slope_volume_m3": total,
        "slope_mean_T_C": float(np.sum(volume * temperature) / total),
        "domain_class_volume_m3": [0.0, 0.0, 0.0],
        "domain_mean_T_C": 0.0,
        "domain_T_variance_K2": 1.0,
        "anomaly_deficit_K_m3": float(np.sum(volume * (20.0 - temperature))),
        "anomaly_second_moment_K2_m3": float(np.sum(volume * (20.0 - temperature) ** 2)),
        "anomaly_effective_volume_m3": float(
            np.sum(volume * (20.0 - temperature)) ** 2
            / np.sum(volume * (20.0 - temperature) ** 2)),
        "anomaly_mean_amplitude_K": float(
            np.sum(volume * (20.0 - temperature) ** 2)
            / np.sum(volume * (20.0 - temperature))),
        "_select": active,
        "_active": active,
        "_volume": volume,
        "_snapped_full": temperature,
        "_klass_full": klass,
        "_centres": np.zeros_like(temperature),
        "_bathy": np.zeros(temperature.shape[:2]),
        "_card": card,
    }


def _pair(nx: int = 4, nz: int = 2):
    temperature = np.full((1, nx, nz), 19.0)
    volume = np.full((1, nx, nz), 1.0e6)
    return temperature, volume


def test_a_pure_temperature_move_across_a_bin_edge_is_reclassified_volume_only():
    temperature, volume = _pair()
    right = _arm(temperature, volume)
    moved = temperature.copy()
    moved[0, 1, 0] = 17.0                      # 18-20 -> 12-18, one cell
    left = _arm(moved, volume)
    result = probe.difference_map(left, right, "T-move")
    assert result["reclassified_cells"] == 1
    assert result["reclassified_volume_m3"] == 1.0e6
    assert result["same_class_volume_reshuffle_fraction"] == 0.0
    assert abs(result["census_delta"][1] - 1.0 / 8.0) < 1e-15
    assert abs(result["reclassified_abs_dT_K"]["max"] - 2.0) < 1e-12


def test_a_pure_thickness_move_is_reshuffle_only_and_reclassifies_nothing():
    temperature, volume = _pair()
    right = _arm(temperature, volume)
    thicker = volume.copy()
    thicker[0, 1, 0] *= 1.5                    # same class, different volume
    left = _arm(temperature, thicker)
    result = probe.difference_map(left, right, "V-move")
    assert result["reclassified_cells"] == 0
    assert result["reclassified_volume_m3"] == 0.0
    assert result["same_class_volume_reshuffle_fraction"] > 0.0
    # every cell is ambient in both arms, so the FRACTIONS cannot move at all
    assert max(abs(value) for value in result["census_delta"]) < 1e-15


def test_the_decomposition_closure_guard_can_actually_fire():
    """The `require` that the per-cell contributions sum to the census delta.

    In the honest fixtures both sides come from one `_classify` call, so the
    guard is never exercised; plant an inconsistent census and it must raise.
    """
    import pytest

    temperature, volume = _pair()
    right = _arm(temperature, volume)
    left = _arm(temperature, volume)
    left["census"] = [0.0, 0.5, 0.5]          # inconsistent with `_klass_full`
    with pytest.raises(probe.ProbeError, match="does not close"):
        probe.difference_map(left, right, "planted")


def _run_planted(tmp_path: Path, flag: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "budget",
            flag,
            "--out",
            str(tmp_path / "must-not-exist.json"),
        ],
        cwd=SCRIPT.parents[4],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_budget_cli_planted_census_excursion_exits_nonzero(tmp_path):
    result = _run_planted(tmp_path, "--plant-census")
    assert result.returncode != 0
    assert "PLANTED census excursion fired" in result.stderr
    assert not (tmp_path / "must-not-exist.json").exists()


def test_budget_cli_planted_unregistered_term_exits_nonzero(tmp_path):
    result = _run_planted(tmp_path, "--plant-unregistered")
    assert result.returncode != 0
    assert "unregistered budget terms" in result.stderr
    assert not (tmp_path / "must-not-exist.json").exists()

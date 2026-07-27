"""Tests for the configuration-driven MPAS NMC data pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.data.fit_mpas_gen_be import REQUIRED_KEYS, _state_from_sample


def _write_sample(path: Path, *, omit: str | None = None) -> None:
    cells, levels = 4, 3
    arrays = {
        "error_u": np.ones((cells, levels)),
        "error_v": np.ones((cells, levels)),
        "error_T": np.ones((cells, levels)),
        "error_p_s": np.ones(cells),
        "error_phis": np.ones(cells),
        "error_tracer_q_v": np.ones((cells, levels)),
    }
    if omit is not None:
        arrays.pop(omit)
    np.savez(path, **arrays)


def test_nmc_config_describes_same_verification_forecasts() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    path = repository_root / "config/4DVar_single/nmc.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["nmc"]["long_forecast_hours"] == 48
    assert config["nmc"]["short_forecast_hours"] == 24
    assert config["nmc"]["sample_count"] == 80
    assert config["model"]["resolution"] == 6
    assert config["model"]["levels"] == 40


def test_full_state_sample_loads_all_control_fields(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    _write_sample(path)
    state = _state_from_sample(path)
    assert state.u.data.shape == (4, 3)
    assert state.v.data.shape == (4, 3)
    assert state.T.data.shape == (4, 3)
    assert state.p_s.data.shape == (4,)
    assert state.tracers["q_v"].data.shape == (4, 3)


@pytest.mark.parametrize("missing", REQUIRED_KEYS)
def test_full_state_sample_rejects_missing_arrays(
    tmp_path: Path,
    missing: str,
) -> None:
    path = tmp_path / "sample.npz"
    _write_sample(path, omit=missing)
    with pytest.raises(ValueError, match="missing required arrays"):
        _state_from_sample(path)

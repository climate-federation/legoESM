"""Direct tests for the prescribed-SST AMIP inference driver's pure helpers
and for the public ``spectral_amip_rollout`` entry point's importability.

The full rollout is a GPU integration run (validated by the sbatch); here we
pin the pure config/calendar helpers and that the public rollout symbol exists
with the documented keyword contract, so a signature drift fails fast.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DRIVER = _REPO / "scripts" / "run" / "run_aimip_amip_inference.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("_amip_infer", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_month_day_of_year_midmonth_360day():
    mod = _load_driver()
    # 30-day-month calendar: month m mid-point = (m-1)*30 + 15.
    assert mod._month_day_of_year(1) == pytest.approx(15.0)
    assert mod._month_day_of_year(7) == pytest.approx(195.0)
    assert mod._month_day_of_year(12) == pytest.approx(345.0)
    # strictly increasing across the year
    vals = [mod._month_day_of_year(m) for m in range(1, 13)]
    assert vals == sorted(vals)


def test_merged_cfg_applies_classical_overlay(tmp_path):
    mod = _load_driver()
    base = tmp_path / "base.yaml"
    base.write_text(
        "n_max: 63\nn_levels: 8\ndt: 600.0\n"
        "aimip_convection: tiedtke\naimip_radiation: rrtmgp\n"
    )
    overlay = tmp_path / "variant_classical.yaml"
    overlay.write_text("aimip_convection: edmf\naimip_microphysics: sundqvist\n")
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"base: {base}\ncfg_overrides:\n  aimip_lr: 0.001\n")
    cfg = mod._merged_cfg(suite)
    assert cfg["aimip_convection"] == "edmf"  # overlay wins over base
    assert cfg["aimip_microphysics"] == "sundqvist"
    assert cfg["aimip_lr"] == 0.001  # cfg_overrides applied
    assert cfg["nlev"] == 8  # n_levels mapped to nlev


def test_spectral_amip_rollout_signature():
    spec = importlib.util.find_spec(
        "legoesm.training.neural_gcm_spectral"
    )
    if spec is None:
        pytest.skip("legoesm not importable in this environment")
    mod = importlib.import_module("legoesm.training.neural_gcm_spectral")
    fn = getattr(mod, "spectral_amip_rollout", None)
    assert fn is not None, "spectral_amip_rollout must be public"
    params = inspect.signature(fn).parameters
    # keyword-only contract the driver relies on
    for kw in ("sst_col", "sizing_phys_state", "day_of_year_base",
               "seconds_offset", "rad_update_interval"):
        assert kw in params, f"missing kwarg {kw}"

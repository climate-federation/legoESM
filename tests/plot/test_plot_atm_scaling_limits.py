"""Scaling-limit plot: model arithmetic by hand, and the figure is written."""
import importlib.util
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "plot" / "plot_atm_scaling_limits.py"
_spec = importlib.util.spec_from_file_location("plot_atm_scaling_limits", _P)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_limit_models_by_hand():
    L = mod.limits()
    assert L["mem2"][3] == pytest.approx(2 * 0.892 / (234.9 / 16) * 1e3)
    assert L["mem1"][0] == pytest.approx(4.844 / (234.9 / 16) * 1e3)
    # 51.4 x 9.8 us + 12.508 MB / 6 GB/s + 22.5 x 3.1 us
    assert L["comm"][0] == pytest.approx(0.50372 + 2.084667 + 0.06975, rel=1e-6)


def test_figure_is_written(tmp_path):
    out = tmp_path / "f.png"
    mod.main(str(out))
    assert out.stat().st_size > 10000

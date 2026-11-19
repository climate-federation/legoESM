"""#2525: --use-mean-fields must refuse a window mean that does not span the
oracle's 5-day record (the plotter skipped the scorers' require_window)."""
import importlib.util
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/plot/plot_sections_vs_nemo.py"


def _module():
    spec = importlib.util.spec_from_file_location("plot_sections_vs_nemo", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mismatched_window_mean_is_refused(monkeypatch, tmp_path):
    mod = _module()
    monkeypatch.setattr(mod, "_load_legoesm",
                        lambda path, use_mean: {"window_days": 1.0})
    monkeypatch.setattr(mod, "load_nemo_3d", lambda *a, **k: pytest.fail(
        "the oracle was read before the window was checked"))
    monkeypatch.setattr(sys, "argv", [
        "plot_sections_vs_nemo.py", "--tripole", "t.npz", "--nemo-gridt",
        "n.nc", "--use-mean-fields", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="not window-matched"):
        mod.main()

"""Direct test of scripts/plot/plot_ocean_wb_d20.py on a synthetic diagnostic file."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plot import plot_ocean_wb_d20 as P  # noqa: E402


def test_writes_figure(tmp_path):
    o = np.full((1, 2, 180, 360), 150.0)
    np.savez(tmp_path / "d20_diagnostic.npz", oras5=o, model=o + 10.0,
             scored=np.ones(o.shape, bool), mae_by_lead_m=np.array([5.0, 10.0]),
             lead_days=np.array([1, 2]), lat=np.arange(-89.5, 90.0), dates=np.array(["1995-03-01"]),
             label="PLUMBING ONLY (unforced)", width_K=0.5)
    P.main([str(tmp_path)])
    assert (tmp_path / "d20_diagnostic.png").stat().st_size > 10_000

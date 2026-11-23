"""Smoke test for scripts/plot/plot_orbit_insolation.py."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "scripts/plot/plot_orbit_insolation.py"
_spec = importlib.util.spec_from_file_location("plot_orbit_insolation", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_map_shape_and_obliquity_override(tmp_path):
    q = mod.insolation_map(mod.orbit_of(orbit_obliquity_deg=0.0),
                           np.array([-30.0, 0.0, 30.0]), np.array([1.0, 172.0]))
    assert q.shape == (2, 3)
    np.testing.assert_allclose(q[:, 0], q[:, 2], rtol=0.05)  # no seasons, symmetric
    mod.main(str(tmp_path))
    assert (tmp_path / "insolation_6ka_minus_1850.png").exists()

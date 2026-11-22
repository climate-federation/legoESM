"""Cube Williamson-2 matrix row runs through the FV3 duo adapter.

Runs ``run_shallow_water`` on a C12 cube W2 row for two outer blocks
(2 x 3600 s = 14 duo acoustic steps) and checks the row PASSes, writes the
standard artifacts, and that the steady W2 state stays put (small
area-weighted L2 height error vs the analytic = initial state).
Measured at C12 / 2 blocks (CPU, x64): L2 5.15e-4, Linf 1.29e-3, mass
drift 5.8e-16; the bounds are ~2x those.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")


@pytest.mark.slow
def test_cube_w2_duo_matrix_row(tmp_path):
    jax.config.update("jax_enable_x64", True)
    from scripts.matrix import run_atmosphere_test_matrix as m

    tc = m.TestCase("shallow_water", "williamson2", "cubed_sphere", "C12",
                    "none", 5, 1, {"test_num": 2})
    status, _wall, notes = m.run_shallow_water(tc, tmp_path, 2 * 3600 / 86400)
    assert status == "PASS", notes
    l2 = float(re.search(r"L2=([0-9.e+-]+)", notes).group(1))
    linf = float(re.search(r"Linf=([0-9.e+-]+)", notes).group(1))
    drift = float(re.search(r"mass drift=([0-9.e+-]+)", notes).group(1))
    assert np.isfinite(l2) and l2 < 1e-3, notes
    # positive evidence the state advanced (an identity step gives L2 == 0)
    assert l2 > 1e-5, notes
    assert np.isfinite(linf) and linf < 3e-3, notes
    assert drift < 1e-13, notes
    for f in ("results.txt", "snapshots_native.npz", "snapshots_latlon.npz",
              "mean_timeseries.csv", "conservation_timeseries.csv",
              "field_snapshots.png"):
        assert (tmp_path / f).exists(), f
    z = np.load(tmp_path / "snapshots_latlon.npz")
    assert z["height"].shape[1:] == (181, 360)
    assert np.all(np.isfinite(z["height"]))
    zn = np.load(tmp_path / "snapshots_native.npz")
    assert zn["times_days"][-1] > 0.0
    assert np.max(np.abs(zn["height"][-1] - zn["height"][0])) > 0.0

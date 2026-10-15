"""Known-answer tests for the equatorial momentum-box diagnostic helpers in
``scripts/run/run_omip_core2.py`` (``--diag-momentum-box-snapshot``)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.run.run_omip_core2 import (  # noqa: E402
    momentum_box_column_mean, momentum_box_interface_flux)

LAT = np.arange(-3.5, 4.0, 1.0)                       # 8 rows, |lat|<=1 -> -0.5, 0.5
LON = np.arange(150.5, 270.0, 1.0)
LAT2, LON2 = np.meshgrid(LAT, LON, indexing="ij")
ZI = np.array([0.0, 100.0, 300.0, 500.0, 900.0])      # 300-500 layer: half counts
ZC = 0.5 * (ZI[1:] + ZI[:-1])


def test_column_mean_known_answer():
    """Depth-uniform 2e-6 west of 210E and 4e-6 east of it, 9 in the layer
    below 500 m (must not count), 300-500 m layer counts 100 m -> 400 m times
    the area-weighted mean: equal areas over 50 + 50 box longitudes give
    400 * 3e-6; east cells 3x larger give 400 * 3.5e-6."""
    f = np.where(LON2 < 210.0, 2e-6, 4e-6)[..., None] * np.ones(ZC.size)
    f[..., 3] = 9.0
    area = np.ones_like(LAT2)
    wet = np.ones_like(LAT2, dtype=bool)
    assert momentum_box_column_mean(f, LAT2, LON2, area, ZI, wet) == pytest.approx(400 * 3e-6)
    area2 = np.where(LON2 < 210.0, 1.0, 3.0)
    assert momentum_box_column_mean(f, LAT2, LON2, area2, ZI, wet) == pytest.approx(400 * 3.5e-6)
    wet2 = wet & (LON2 > 210.0)                        # dry west half drops out
    assert momentum_box_column_mean(f, LAT2, LON2, area, ZI, wet2) == pytest.approx(400 * 4e-6)
    f[2, 30, 0] = np.nan                               # out of the box rows: ignored
    assert np.isfinite(momentum_box_column_mean(f, LAT2, LON2, area, ZI, wet))
    f[4, 30, 0] = np.nan                               # inside the box: raises
    with pytest.raises(ValueError, match="non-finite"):
        momentum_box_column_mean(f, LAT2, LON2, area, ZI, wet)


def test_interface_flux_known_answer():
    """u = 1.0 above, 0.2 below; K = 0.01 on every interface -> at 400 m (between
    interfaces 300 and 500) flux = 0.01 * (u3 - u2)/(zc3 - zc2), interpolated."""
    u = np.broadcast_to(np.array([1.0, 1.0, 1.0, 0.2]), LAT2.shape + (4,)).copy()
    K = np.full(LAT2.shape + (3,), 0.01)
    K[..., 2] = 0.03                                   # interface 500 m
    zin = ZI[1:-1]
    fl = 0.01 * np.diff(u, axis=-1) / np.diff(ZC)
    fl[..., 2] *= 3.0
    t = (400.0 - 300.0) / (500.0 - 300.0)
    want = (1 - t) * fl[0, 0, 1] + t * fl[0, 0, 2]
    area = np.ones_like(LAT2); wet = np.ones_like(LAT2, dtype=bool)
    got = momentum_box_interface_flux(K, u, ZC, zin, LAT2, LON2, area, wet)
    assert got == pytest.approx(want) and got < 0          # slower water below drags

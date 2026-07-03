"""Direct test for the ACE baseline scorer's area-weighting math.

Only the cos(lat)-weighted RMSE/bias is non-trivial; the .nc reading is a
thin xarray wrapper exercised by the real ACE run.  numpy-only (no xarray),
so it imports cheaply.
"""

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_score_ace", REPO / "scripts" / "validate" / "score_ace_baseline.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_constant_offset_gives_offset_rmse_and_bias():
    """pred = tgt + c (constant) -> rmse = |c|, bias = c, for any lat band."""
    lat = np.linspace(-89.0, 89.0, 18)
    tgt = np.random.RandomState(0).randn(18, 36)
    for c in (2.5, -1.0, 0.0):
        rmse, bias = mod._area_weighted_rmse_bias(tgt + c, tgt, lat)
        assert abs(rmse - abs(c)) < 1e-9, (rmse, c)
        assert abs(bias - c) < 1e-9, (bias, c)


def test_cos_weight_downweights_poles():
    """A polar-only error must score lower than the same error at the equator
    (cos(lat) weighting), confirming the weights are applied on the lat axis."""
    lat = np.linspace(-89.0, 89.0, 18)
    base = np.zeros((18, 36))
    polar = base.copy(); polar[0, :] = 10.0          # error at south pole
    equ = base.copy(); equ[9, :] = 10.0              # error near equator
    rmse_polar, _ = mod._area_weighted_rmse_bias(polar, base, lat)
    rmse_equ, _ = mod._area_weighted_rmse_bias(equ, base, lat)
    assert rmse_polar < rmse_equ, (rmse_polar, rmse_equ)


if __name__ == "__main__":
    test_constant_offset_gives_offset_rmse_and_bias()
    test_cos_weight_downweights_poles()
    print("score_ace_baseline: self-checks passed")

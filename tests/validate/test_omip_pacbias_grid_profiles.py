"""Direct test of the cross-grid equatorial-Pacific probe's remap + box mean."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate" / "ocean_fidelity"))
import omip_pacbias_grid_profiles as P  # noqa: E402


def _cloud(n=4000, seed=0):
    r = np.random.default_rng(seed)
    return r.uniform(-14, 14, n), r.uniform(121, 299, n)


def test_box_mean_known_answers():
    la, lo = _cloud()
    wet = np.ones_like(la, bool)
    assert abs(P.box_mean(P.remap(np.full_like(la, 3.25), la, lo, wet)) - 3.25) < 1e-12
    # a field linear in longitude has nino3 mean = box centre (240E) up to remap error
    assert abs(P.box_mean(P.remap(lo.copy(), la, lo, wet)) - 240.0) < 0.5


def test_dry_points_never_leak():
    la, lo = _cloud()
    f = np.where(lo < 240.0, 1.0, 1.0e6)          # huge value on the "land" half
    wet = lo < 240.0
    g = P.remap(f, la, lo, wet)
    assert np.nanmax(g) < 1.0 + 1e-12              # fails if dry sources (1e6) enter the IDW
    assert np.isnan(g[:, P.GLON > 245.0]).all()    # beyond 1.5 deg of any wet source -> NaN

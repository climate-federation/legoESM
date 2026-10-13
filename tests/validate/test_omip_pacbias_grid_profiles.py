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


def test_w_from_divergence_partial_layer_and_sign():
    z_c = np.array([5.0, 15.0, 30.0, 50.0, 70.0])      # interfaces 0,10,22.5,40,60,80
    div = np.full((z_c.size, P.GLAT.size, P.GLON.size), 1.0e-6)
    w = P.w_from_divergence(div, z_c, 50.0)
    # 10 + 12.5 + 17.5 + half of the 40-60 m layer = 50 m of divergence; surface
    # divergence must give UPWARD (positive) w
    assert np.allclose(w, 50.0e-6, rtol=1e-12)


def test_divergence_of_zonally_linear_flow():
    R = 6.4e6
    a = 0.2                                            # u = a * lon_rad  [m/s]
    lon_rad = np.deg2rad(P.GLON)[None, None, :]
    u = np.broadcast_to(a * lon_rad, (3, P.GLAT.size, P.GLON.size)).copy()
    v = np.zeros_like(u)
    d = P.horizontal_divergence(u, v, R)
    expect = a / (R * np.cos(np.deg2rad(P.GLAT)))[None, :, None]
    inner = np.s_[:, 1:-1, 1:-1]          # both centred differences defined
    assert np.allclose(d[inner], np.broadcast_to(expect, d.shape)[inner], rtol=1e-12)
    assert np.isnan(d[:, :, 0]).all() and np.isnan(d[:, :, -1]).all()
    # meridional: v = b * cos-weighted latitude ramp, div = b / R
    b = 0.3
    phi = np.deg2rad(P.GLAT)[None, :, None]
    v2 = np.broadcast_to(b * phi / np.cos(phi), u.shape).copy()
    d2 = P.horizontal_divergence(np.zeros_like(u), v2, R)
    assert np.allclose(d2[:, 1:-1, 1:-1], (b / (R * np.cos(phi)))[:, 1:-1, :] * np.ones_like(d2[:, 1:-1, 1:-1]), rtol=1e-12)

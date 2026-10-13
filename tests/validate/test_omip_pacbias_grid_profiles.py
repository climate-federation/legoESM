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
    zi = np.array([0.0, 10.0, 22.5, 40.0, 60.0, 80.0])
    shape = (zi.size - 1, P.GLAT.size, P.GLON.size)
    div = np.full(shape, 1.0e-6)
    # 10 + 12.5 + 17.5 + half of the 40-60 m layer = 50 m; surface divergence -> UPWARD w
    assert np.allclose(P.w_from_divergence(div, zi, 50.0), 50.0e-6, rtol=1e-12)
    # depth-varying divergence: each layer's own thickness must weight it
    dl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:, None, None] * np.ones(shape) * 1e-6
    expect = (10 * 1 + 12.5 * 2 + 17.5 * 3 + 10 * 4) * 1e-6
    assert np.allclose(P.w_from_divergence(dl, zi, 50.0), expect, rtol=1e-12)
    dl[4] = np.nan                                    # below 50 m: must not matter
    assert np.isfinite(P.w_from_divergence(dl, zi, 50.0)).all()
    dl[1, 3, 4] = np.nan                              # above 50 m: column must go NaN
    w = P.w_from_divergence(dl, zi, 50.0)
    assert np.isnan(w[3, 4]) and np.isfinite(w[3, 5])


def test_to_levels_holds_top_layer_and_cuts_bottom():
    zf = np.array([0.512, 1.56, 2.68])
    F = np.array([[20.0, 30.0], [19.0, 29.0], [18.0, 28.0]])
    out = P.to_levels(F, zf, np.array([0.5058, 1.56, 3.0]))
    assert np.allclose(out[0], F[0]) and np.allclose(out[1], F[1])   # NEMO's shallower top centre
    assert np.isnan(out[2]).all()                                    # below the source column


def test_band_weights_partial_overlap():
    w = P.band_weights(np.array([0.0, 10.0, 25.0, 60.0]), 20.0, 50.0)
    assert np.allclose(w, [0.0, 5.0, 25.0])


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

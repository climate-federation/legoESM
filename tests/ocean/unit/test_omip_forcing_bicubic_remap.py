"""Gap 13: bicubic wind remapping, matching ORCA1's per-field weights files.

The oracle does NOT use one remapping method. From its own namelist_cfg:

    sn_wndi / sn_wndj (148-149)  weights_coreII_2_eORCA1.4.2_BICUBIC.nc
    sn_qsr/qlw/tair   (150-152)  weights_coreII_2_eORCA1.4.2_BILINEAR.nc

so momentum is remapped bicubically and the thermodynamic channels linearly.
A uniform method is unfaithful whichever one it picks.

The oracle's weight FILES are absent from this machine, so this reproduces
NEMO's METHOD (Catmull-Rom cubic convolution, the kernel SCRIPS bicubic
remapping is built on) and not its exact weights. These tests therefore pin
the mathematical properties a correct cubic remap must have, not equality
against an oracle array we do not possess.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.coupler import omip2_applicator as A


def _src():
    """A CORE-II-like coarse grid: ~1.9 deg, periodic in longitude."""
    lat = np.linspace(-89.0, 89.0, 94)
    lon = np.arange(0.0, 360.0, 1.875)
    return lat, lon


# --- the kernel -------------------------------------------------------------

def test_cubic_weights_are_a_partition_of_unity():
    """If the four weights did not sum to 1 the remap would scale the field."""
    for t in np.linspace(0.0, 1.0, 21):
        w = A._cubic_weights(t)
        assert sum(float(x) for x in w) == pytest.approx(1.0, abs=1e-12), t


def test_cubic_kernel_is_interpolating_at_the_nodes():
    """At t=0 the stencil must collapse onto the sample itself: Catmull-Rom
    passes THROUGH the data. A smoothing kernel (e.g. B-spline) would not, and
    would damp the wind field everywhere."""
    w1, w2, w3, w4 = A._cubic_weights(0.0)
    assert float(w2) == pytest.approx(1.0)
    for w in (w1, w3, w4):
        assert float(w) == pytest.approx(0.0, abs=1e-12)


# --- the remap --------------------------------------------------------------

def test_constant_field_is_preserved_exactly():
    lat, lon = _src()
    f = np.full((lat.size, lon.size), 7.25)
    out = A._bicubic_interp_to_points(f, lat, lon,
                                      np.array([12.3, -45.0, 80.0]),
                                      np.array([5.0, 187.4, 359.9]))
    assert np.allclose(out, 7.25, atol=1e-12)


def test_linear_field_is_reproduced_better_than_bilinear_is_not_required_but_exactly():
    """A cubic-convolution kernel reproduces linear data exactly. This is the
    property that distinguishes a correct implementation from a plausible one:
    get a sign or an offset wrong in the stencil and linear data bends."""
    lat, lon = _src()
    f = np.broadcast_to(lat.reshape(-1, 1), (lat.size, lon.size)) * 1.0
    dst_lat = np.array([10.0, -33.3, 55.5])
    dst_lon = np.array([100.0, 200.0, 300.0])
    out = A._bicubic_interp_to_points(f, lat, lon, dst_lat, dst_lon)
    assert np.allclose(out, dst_lat, atol=1e-9), out


def test_it_matches_bilinear_on_linear_data_and_differs_on_curved_data():
    """Non-vacuity for the METHOD: on a linear field both schemes are exact so
    they agree, and on a curved field they must NOT — otherwise 'bicubic' is
    bilinear under another name."""
    lat, lon = _src()
    dst_lat = np.array([7.0, -21.0, 44.0])
    dst_lon = np.array([33.0, 210.0, 301.0])

    lin = np.broadcast_to(lat.reshape(-1, 1), (lat.size, lon.size)) * 1.0
    bic = A._bicubic_interp_to_points(lin, lat, lon, dst_lat, dst_lon)
    bil = A._bilinear_interp_to_points(lin, lat, lon, dst_lat, dst_lon)
    assert np.allclose(bic, bil, atol=1e-8)

    curved = np.sin(np.deg2rad(lat)).reshape(-1, 1) * np.cos(
        np.deg2rad(lon)).reshape(1, -1)
    bic_c = A._bicubic_interp_to_points(curved, lat, lon, dst_lat, dst_lon)
    bil_c = A._bilinear_interp_to_points(curved, lat, lon, dst_lat, dst_lon)
    assert not np.allclose(bic_c, bil_c, atol=1e-10)
    # and the cubic must be CLOSER to the truth, or it is not an improvement
    truth = np.sin(np.deg2rad(dst_lat)) * np.cos(np.deg2rad(dst_lon))
    assert np.abs(bic_c - truth).max() < np.abs(bil_c - truth).max()


def test_longitude_wraps_across_the_seam():
    """A target just past the last source column must interpolate across the
    wrap, not clamp. A non-periodic stencil prints a seam into the wind field
    at the prime meridian."""
    lat, lon = _src()
    f = np.broadcast_to(np.cos(np.deg2rad(lon)).reshape(1, -1),
                        (lat.size, lon.size)) * 1.0
    near_seam = np.array([359.5, 0.5])
    out = A._bicubic_interp_to_points(f, lat, np.asarray(lon),
                                      np.array([0.0, 0.0]), near_seam)
    assert np.allclose(out, np.cos(np.deg2rad(near_seam)), atol=2e-4), out


def test_latitude_is_clamped_not_extrapolated_at_the_poles():
    """CORE-II stops short of the pole. A cubic stencil running off the end
    must clamp, not extrapolate a curved field into the gap."""
    lat, lon = _src()
    f = np.broadcast_to(lat.reshape(-1, 1), (lat.size, lon.size)) ** 2
    out = A._bicubic_interp_to_points(f, lat, lon,
                                      np.array([89.9, -89.9]),
                                      np.array([0.0, 0.0]))
    assert np.all(np.isfinite(out))
    assert out.max() <= f.max() + 1e-6


# --- the per-channel split --------------------------------------------------

class _Forcing:
    lat = None
    lon = None

    def __init__(self, lat, lon):
        type(self).lat, type(self).lon = lat, lon
        shape = (1, lat.size, lon.size)
        curved = (np.sin(np.deg2rad(lat)).reshape(1, -1, 1)
                  * np.cos(np.deg2rad(lon)).reshape(1, 1, -1))
        for n in ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down",
                  "precip"):
            setattr(self, n, np.broadcast_to(curved, shape).copy())
        self.snow = None
        self.slp = None


def test_nemo_weights_uses_bicubic_for_winds_and_bilinear_for_the_rest():
    """THE POINT OF GAP 13. Each channel must match the scheme its own oracle
    weights file names -- winds bicubic, everything else bilinear."""
    lat, lon = _src()
    f = _Forcing(lat, lon)
    dst_lat = np.array([11.0, -37.0, 62.0])
    dst_lon = np.array([44.0, 175.0, 288.0])
    out = A._sample_forcing_points(f, 0, dst_lat, dst_lon,
                                   method="nemo_weights")
    for name in ("u10", "v10"):
        want = A._bicubic_interp_to_points(getattr(f, name)[0], lat, lon,
                                           dst_lat, dst_lon)
        assert np.allclose(out[name], want), name
    for name in ("T_air", "q_air", "sw_down", "lw_down", "precip"):
        want = A._bilinear_interp_to_points(getattr(f, name)[0], lat, lon,
                                            dst_lat, dst_lon)
        assert np.allclose(out[name], want), name


def test_nemo_weights_actually_differs_from_uniform_bilinear():
    """If the winds came back identical to bilinear the new method would be
    decorative."""
    lat, lon = _src()
    f = _Forcing(lat, lon)
    dst_lat, dst_lon = np.array([11.0, 62.0]), np.array([44.0, 288.0])
    nemo = A._sample_forcing_points(f, 0, dst_lat, dst_lon,
                                    method="nemo_weights")
    bil = A._sample_forcing_points(f, 0, dst_lat, dst_lon, method="bilinear")
    assert not np.allclose(nemo["u10"], bil["u10"], atol=1e-12)
    assert np.allclose(nemo["T_air"], bil["T_air"])


def test_unknown_method_still_raises():
    """NOTE: codex flagged that this passes against HEAD with the feature
    reverted -- it pins PRE-EXISTING rejection behaviour, not gap 13. Kept as
    regression cover, but it is NOT coverage of this change."""
    lat, lon = _src()
    with pytest.raises(ValueError, match="unknown method"):
        A._sample_forcing_points(_Forcing(lat, lon), 0,
                                 np.array([0.0]), np.array([0.0]),
                                 method="bicubic_everything")


def test_the_cache_does_not_hand_one_grid_another_grids_weights():
    """codex MAJOR: my first cache key used only array endpoints, so two
    different destination grids of the same length could collide and silently
    share weights. This module already grew a collision-safe _coord_key in
    August for exactly that bug; the key now uses it.
    """
    lat, lon = _src()
    f = np.broadcast_to(np.cos(np.deg2rad(lon)).reshape(1, -1),
                        (lat.size, lon.size)) * 1.0
    # Same first and last entries, different interiors -- the collision case.
    dst_lat_a = np.array([0.0, 10.0, 20.0])
    dst_lat_b = np.array([0.0, -40.0, 20.0])
    dst_lon = np.array([30.0, 120.0, 250.0])
    a = A._bicubic_interp_to_points(f, lat, lon, dst_lat_a, dst_lon)
    b = A._bicubic_interp_to_points(f, lat, lon, dst_lat_b, dst_lon)
    # The field varies only in longitude, so both must equal cos(lon) -- but
    # the INDICES differ, and a colliding cache would return a's rows for b.
    want = np.cos(np.deg2rad(dst_lon))
    assert np.allclose(a, want, atol=2e-4)
    assert np.allclose(b, want, atol=2e-4)

    # And directly: two distinct destination sets must not share a cache entry.
    A._BICUBIC_MAP_CACHE.clear()
    A._bicubic_point_maps(lat, lon, dst_lat_a, dst_lon)
    A._bicubic_point_maps(lat, lon, dst_lat_b, dst_lon)
    assert len(A._BICUBIC_MAP_CACHE) == 2, A._BICUBIC_MAP_CACHE.keys()


def test_the_oracle_split_is_recorded_in_code():
    assert A._NEMO_BICUBIC_CHANNELS == ("u10", "v10")

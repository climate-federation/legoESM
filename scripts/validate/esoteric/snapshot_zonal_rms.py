"""Snapshot spatial RMS of zonal anomalies (p_s, T) per latitude band, away from the ridge.

ONE checkpoint, so this is the spatial RMS of the deviation from each 5-degree
bin's area mean -- not transient eddy variance: it includes stationary waves
outside the ridge exclusion and the meridional gradient inside each bin.
Usage: snapshot_zonal_rms.py checkpoint_day_NNNN.npz mesh_level ridge_lon_deg
"""
import sys

import numpy as np

from _common import BANDS

BIN_DEG = 5
RIDGE_EXCLUSION_DEG = 30
MIN_CELLS_PER_BIN = 4


def snapshot_spatial_rms(x, lat, area, sel, bands=BANDS):
    """Area-weighted RMS of x about each 5-deg bin's mean, pooled into bands."""
    x, area = np.asarray(x, np.float64), np.asarray(area, np.float64)
    num = np.zeros(len(bands)); den = np.zeros(len(bands))
    edges = np.arange(-90, 91, BIN_DEG)
    for a, b in zip(edges[:-1], edges[1:]):
        m = sel & (lat >= a) & (lat < b)
        if m.sum() < MIN_CELLS_PER_BIN:
            continue
        w = area[m]; xm = np.sum(x[m] * w) / w.sum()
        for i, (lo, hi) in enumerate(bands):
            if lo <= (a + b) / 2 < hi:
                num[i] += np.sum((x[m] - xm) ** 2 * w); den[i] += w.sum()
    if np.any(den == 0):
        raise ValueError(f"empty band(s): {[bands[i] for i in np.flatnonzero(den == 0)]}")
    return np.sqrt(num / den)


def far_from_ridge(lon_deg, ridge_lon_deg):
    """Cells more than RIDGE_EXCLUSION_DEG of longitude from the ridge (wraps at 360)."""
    return np.abs((np.asarray(lon_deg) - ridge_lon_deg + 180) % 360 - 180) > RIDGE_EXCLUSION_DEG


def main(argv):
    from legoesm.grids.voronoi import create_voronoi_mesh

    ck = np.load(argv[0], allow_pickle=True)
    mesh = create_voronoi_mesh(int(argv[1]))
    rlon = float(argv[2])
    lat = np.degrees(np.asarray(mesh.latCell)); lon = np.degrees(np.asarray(mesh.lonCell)) % 360
    area = np.asarray(mesh.areaCell)
    T = np.asarray(ck["T"]); ps = np.asarray(ck["p_s"])
    if T.shape[0] != mesh.nCells or not np.all(np.isfinite(T)):
        raise ValueError(f"bad T: {T.shape}")
    far = far_from_ridge(lon, rlon)
    print(f"day {float(ck['day']):.1f}  ridge lon {rlon:.1f}, cells >{RIDGE_EXCLUSION_DEG} deg away {far.sum()}")
    print("SNAPSHOT spatial RMS about 5-deg bin means (not transient variance); bands:", BANDS)
    print("p_s snapshot RMS [hPa] far:", np.round(snapshot_spatial_rms(ps, lat, area, far) / 100, 2))
    for k in list(range(8, T.shape[1], 4)) + [T.shape[1] - 1]:
        print(f"k={k:2d} T snapshot RMS [K] far: {np.round(snapshot_spatial_rms(T[:, k], lat, area, far), 2)}")


if __name__ == "__main__":
    main(sys.argv[1:])

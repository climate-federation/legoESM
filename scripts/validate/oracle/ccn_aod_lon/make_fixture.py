"""Generate the AOD->CCN input oracle fixture from the SOURCE aerosol file.

No code oracle exists for this unit: CAM6 (cam_cesm2_1_rel_60, a03b84b7)
activates MAM4 modes with ndrop.F90, which legoESM replaces by the published
closed form of Andreae (2009), ACP 9:543, Sect. 8 / Fig. 1:

    AOT500 = 0.0027 * CCN0.4**0.640      (CCN0.4 in cm^-3, r^2 = 0.88)

where AOT500 is the LOCAL TOTAL (fine + coarse) column AOT (AERONET).  The
oracle is that fit applied to the production total-AOD file at each column's
own latitude AND longitude, read here independently of legoESM (netCDF4 +
cftime, numpy float64):

  * band: the file's own ``wl1_sun``/``wl2_sun`` band containing 0.55 um;
  * time: linear between the two records bracketing the calendar date, on
    the file's own calendar;
  * space: columns sit on source cell CENTRES (value = that cell) or on cell
    CORNERS (bilinear at a corner = mean of the four cells); corners include
    the file's +-180 deg seam and the 0/360 deg wrap;
  * N = (AOT / 0.0027)**(1 / 0.640), clipped to legoESM's CCNFromAODConfig
    limits 10..1e4 cm^-3 (the paper gives no limits; Fig. 1 spans 10..1e4).

The fixture also stores, per date, the two bracketing band-0.55 um records on
a 5x5 cell block around each column (NaN elsewhere when rebuilt), so the test
can write a small file in the production layout and run the real loader.

Usage (CPU node, ~2 min)::

    python scripts/validate/oracle/ccn_aod_lon/make_fixture.py \\
        --src /burg-archive/glab/users/pg2328/legoESM/data/cmip6_forcing/aeropt_macv2sp_sw_b14_tot_1850-2014_MACv1bg.nc \\
        --out tests/atmosphere/microphysics/oracle_fixtures/ccn_aod_lon.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

AOT_COEFF = 0.0027        # Andreae (2009) Fig. 1 fit coefficient
AOT_EXPONENT = 0.640      # Andreae (2009) Fig. 1 fit exponent
N_MIN_CM3 = 10.0          # legoESM CCNFromAODConfig.n_ccn_min_cm3 (not in the paper)
N_MAX_CM3 = 1.0e4         # legoESM CCNFromAODConfig.n_ccn_max_cm3 (not in the paper)
VIS_UM = 0.55
START_YEAR = 1979         # production deck start year (sim day 0 = 1979-01-01)
DATES = ((2000, 1, 15), (2000, 7, 15))
# (name, lat, lon) [deg N, deg E]; the first eight are the tier-1 audit columns.
COLUMNS = (
    ("trop_deepconv_WPWP", 0.5, 155.5),
    ("subtrop_Sc_SEPac", -19.5, 275.5),
    ("midlat_polluted_EChina", 31.5, 116.5),
    ("midlat_cleanocean_NPac_sameband", 31.5, 180.5),
    ("dust_Sahel", 15.5, 0.5),
    ("SO_clean", -55.5, 200.5),
    ("polar_Antarctic", -79.5, 0.5),
    ("polar_Arctic", 80.5, 0.5),
    ("corner_EChina", 31.0, 116.0),
    ("corner_dateline_seam", 0.0, 180.0),
    ("corner_greenwich_wrap", -55.0, 0.0),
)
HALF = 2                  # 5x5 block


def andreae_ccn_m3(aot):
    """Inverse of the Andreae (2009) fit, clipped to legoESM's limits [1/m^3]."""
    aot = np.asarray(aot, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.power(aot / AOT_COEFF, 1.0 / AOT_EXPONENT)
    return np.clip(n, N_MIN_CM3, N_MAX_CM3) * 1.0e6


def sim_day(y, m, d):
    """Days since START_YEAR-01-01 on the model's noleap clock."""
    import cftime
    return float((cftime.DatetimeNoLeap(y, m, d)
                  - cftime.DatetimeNoLeap(START_YEAR, 1, 1)).days)


def column_value(field, lat, lon, la, lo):
    """Cell value at a centre, mean of the four cells at a corner."""
    lon360 = lon % 360.0
    i = np.flatnonzero(np.abs(lat - la) <= 0.5 + 1e-9)
    j = np.flatnonzero(np.abs(((lon360 - lo % 360.0) + 180.0) % 360.0 - 180.0) <= 0.5 + 1e-9)
    assert len(i) in (1, 2) and len(j) in (1, 2) and len(i) == len(j), (la, lo, i, j)
    return float(np.mean(field[np.ix_(i, j)])), i, j


def main(argv=None):
    import cftime
    import netCDF4
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    md5 = hashlib.md5()
    with open(a.src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    ds = netCDF4.Dataset(a.src)
    t = np.asarray(ds["time"][:], dtype=np.float64)
    tunits, tcal = ds["time"].units, ds["time"].calendar
    wl1 = np.asarray(ds["wl1_sun"][:], dtype=np.float64)
    wl2 = np.asarray(ds["wl2_sun"][:], dtype=np.float64)
    lo_, hi_ = np.minimum(wl1, wl2), np.maximum(wl1, wl2)
    ib = int(np.flatnonzero((lo_ <= VIS_UM) & (hi_ > VIS_UM))[0])
    lat = np.asarray(ds["lat"][:], dtype=np.float64)
    lon = np.asarray(ds["lon"][:], dtype=np.float64)
    nlat, nlon = lat.size, lon.size

    recs, weights = [], []
    for (y, m, d) in DATES:
        tq = cftime.date2num(cftime.datetime(y, m, d, calendar=tcal), tunits, tcal)
        i1 = int(np.searchsorted(t, tq))
        recs += [i1 - 1, i1]
        weights.append((tq - t[i1 - 1]) / (t[i1] - t[i1 - 1]))
    lo_rec, hi_rec = min(recs), max(recs)
    # >12 records keeps the loader on its multi-year (non-cyclic) path, as
    # with the production 1980-record file.
    sub = np.arange(lo_rec, max(hi_rec, lo_rec + 12) + 1)
    fields = {r: np.asarray(ds["aod"][r, ib, :, :], dtype=np.float32) for r in recs}
    assert all(np.isfinite(f).all() for f in fields.values())
    ds.close()

    # 5x5 block mask around every column's cell(s).
    mask = np.zeros((nlat, nlon), dtype=bool)
    aod = np.zeros((len(DATES), len(COLUMNS)))
    for k, (_, la, lo) in enumerate(COLUMNS):
        for q in range(len(DATES)):
            f0 = fields[recs[2 * q]].astype(np.float64)
            f1 = fields[recs[2 * q + 1]].astype(np.float64)
            w = weights[q]
            v0, i, j = column_value(f0, lat, lon, la, lo)
            v1, _, _ = column_value(f1, lat, lon, la, lo)
            aod[q, k] = (1.0 - w) * v0 + w * v1
        ii = np.clip(np.arange(i.min() - HALF, i.max() + HALF + 1), 0, nlat - 1)
        jj = np.arange(j.min() - HALF, j.max() + HALF + 1) % nlon
        mask[np.ix_(ii, jj)] = True
    blk_i, blk_j = np.nonzero(mask)
    rec_vals = np.stack([fields[r][blk_i, blk_j] for r in recs])

    sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
                          "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(
        oracle="Andreae (2009) ACP 9:543 Sect. 8 / Fig. 1, AOT500 = 0.0027*CCN0.4^0.640, "
               "applied to the source file (no code oracle; CAM6 cam_cesm2_1_rel_60 "
               "a03b84b7 uses MAM4 + ndrop.F90 instead)",
        src=str(a.src), src_md5=md5.hexdigest(), band_index=ib,
        band_um=[float(lo_[ib]), float(hi_[ib])], records=[int(r) for r in recs],
        dates=["%04d-%02d-%02d" % d for d in DATES], time_weights=[float(w) for w in weights],
        limits_cm3=[N_MIN_CM3, N_MAX_CM3], start_year=START_YEAR,
        legoesm_sha=sha,
        command="python scripts/validate/oracle/ccn_aod_lon/make_fixture.py "
                f"--src {a.src} --out {a.out}")
    np.savez_compressed(
        a.out, meta=json.dumps(meta),
        lat=lat, lon=lon, wl1=wl1, wl2=wl2, time_units=tunits, calendar=tcal,
        time_sub=t[sub], rec_pos=np.asarray(recs) - lo_rec,
        blk_i=blk_i, blk_j=blk_j, rec_vals=rec_vals,
        col_names=np.array([c[0] for c in COLUMNS]),
        col_lat=np.array([c[1] for c in COLUMNS]), col_lon=np.array([c[2] for c in COLUMNS]),
        sim_day=np.array([sim_day(*d) for d in DATES]),
        aod_oracle=aod, n_oracle_m3=andreae_ccn_m3(aod))
    print(json.dumps(meta, indent=1))
    for k, c in enumerate(COLUMNS):
        print(f"{c[0]:34s} AOD " + " ".join(f"{aod[q, k]:.5f}" for q in range(len(DATES)))
              + "  N[cm-3] " + " ".join(f"{andreae_ccn_m3(aod[q, k]) / 1e6:9.2f}"
                                        for q in range(len(DATES))))
    print("fixture bytes", Path(a.out).stat().st_size)


if __name__ == "__main__":
    main()

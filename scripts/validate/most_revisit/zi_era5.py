"""ERA5 boundary-layer height (hourly analysis, param 159) on each flux-profile table's time axis.

Choices (GLM+codex vote, coscientist/vote_zi_result.txt): nearest grid point; linear interpolation in time
to the interval midpoint; drifting floes follow their raw-file track. Times outside the pooled ERA5 record are NaN.
Needs eccodes (module python3/2023.01-gcc-11.2.0), imported only when reading GRIB.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import netCDF4
import numpy as np

ERA5_BLH = "/pool/data/ERA5/E5/sf/an/1H/159/E5sf00_1H_{}_159.grb"
HOUR, DAY = 3600, 86400
EARTH_RADIUS_KM = 6371.0
MATLAB_EPOCH_DAYS = 719529.0


def unit_vectors(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def midpoint_hours(time_end, averaging_minutes):
    """Full hour at or before each interval midpoint, and the weight of the following hour."""
    mid = np.asarray(time_end, np.int64) - averaging_minutes * 30
    h0 = mid // HOUR * HOUR
    return h0, (mid - h0) / HOUR


def track_vectors(t_track, lat, lon, t):
    """Unit vectors along a track at times t, linear in each component; NaN outside the track's time span."""
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(t_track)
    if not ok.any():
        return np.full((np.size(t), 3), np.nan)
    v = unit_vectors(lat[ok], lon[ok])
    order = np.argsort(t_track[ok])
    tt = t_track[ok][order]
    out = np.column_stack([np.interp(t, tt, v[order, k]) for k in range(3)])
    out[(np.asarray(t) < tt[0]) | (np.asarray(t) > tt[-1])] = np.nan
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _tracks(data_root):
    def mosaic():
        ts, la, lo = [], [], []
        for p in sorted(glob.glob(f"{data_root}/mosaic_metcity/mosseb.metcity.level3.4.10min.*.nc")):
            with netCDF4.Dataset(p) as d:
                tv = d["time"]
                ts.append(netCDF4.date2num(netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False),
                                           "seconds since 1970-01-01"))
                la.append(np.ma.filled(d["lat_tower"][:].astype(float), np.nan))
                lo.append(np.ma.filled(d["lon_tower"][:].astype(float), np.nan))
        return np.concatenate(ts), np.concatenate(la), np.concatenate(lo)

    def sheba():
        p = f"{data_root}/mirror_sheba_asfg_5level/prof_file_all6_ed_hd.txt"
        x = np.genfromtxt(p, delimiter="\t", skip_header=2, usecols=(0, 1, 2))
        x[x >= 999] = np.nan
        t = np.datetime64("1997-01-01").astype("datetime64[s]").astype(np.int64) + (x[:, 0] - 1.0) * DAY
        return t, x[:, 1], x[:, 2]

    def nc(path, tname, matlab=False):
        def f():
            with netCDF4.Dataset(path) as d:
                t = np.ma.filled(d[tname][:].astype(float), np.nan)
                t = (t - MATLAB_EPOCH_DAYS) * DAY if matlab else t
                return t, np.ma.filled(d["latitude"][:].astype(float), np.nan), np.ma.filled(d["longitude"][:].astype(float), np.nan)
        return f

    def ao2018():
        p = f"{data_root}/ao2018_navigation/x/AO2018_shipnav_v1.1/ACAS_AO2018_ship_30min_v1_1.nc"
        with netCDF4.Dataset(p) as d:
            return tuple(np.ma.filled(d[k][:].astype(float), np.nan) for k in ("time", "lat", "lon"))

    return {"mosaic_metcity": mosaic, "sheba": sheba, "ao2018": ao2018,
            "ascos": nc(f"{data_root}/ascos_metalley/ascos_Metalley.nc", "time", matlab=True),
            "nice2015": nc(f"{data_root}/nice2015_flux/x/N-ICE_sebData_v1.nc", "unix_time")}


def _grid():
    import eccodes
    with open(ERA5_BLH.format("2020-01-01"), "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        lat, lon = eccodes.codes_get_array(gid, "latitudes"), eccodes.codes_get_array(gid, "longitudes")
        eccodes.codes_release(gid)
    return lat, lon


def _read_day(job):
    """Values at grid points for the requested hours of one day; NaN where the file is missing."""
    import eccodes
    day, hours, points = job
    out = np.full(hours.size, np.nan)
    path = ERA5_BLH.format(np.datetime64(int(day), "s").astype("datetime64[D]"))
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        while (gid := eccodes.codes_grib_new_from_file(f)) is not None:
            hour = day + eccodes.codes_get(gid, "dataTime") // 100 * HOUR
            sel = hours == hour
            if sel.any():
                out[sel] = eccodes.codes_get_values(gid)[points[sel]]
            eccodes.codes_release(gid)
    return out


def era5_zi(sites, workers):
    """sites: list of (name, time_end, averaging_minutes, unit vectors per row). Returns dict of results."""
    from scipy.spatial import cKDTree
    glat, glon = _grid()
    tree = cKDTree(unit_vectors(glat, glon))
    hours, points, owner = [], [], []
    meta = {}
    for k, (name, time_end, avg, vec) in enumerate(sites):
        valid = np.isfinite(vec).all(axis=1)
        chord, idx = np.full(valid.size, np.nan), np.zeros(valid.size, int)
        if valid.any():
            chord[valid], idx[valid] = tree.query(vec[valid])
        h0, w = midpoint_hours(time_end, avg)
        w = np.where(valid, w, np.nan)
        meta[name] = (w, 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(chord / 2, 1.0)), np.where(valid, glat[idx], np.nan),
                      np.where(valid, glon[idx], np.nan), h0)
        for shift in (0, HOUR):
            hours.append(h0 + shift)
            points.append(idx)
            owner.append(np.full(idx.size, 2 * k + (shift > 0)))
    hours, points, owner = map(np.concatenate, (hours, points, owner))
    key = np.unique(np.column_stack([hours, points]), axis=0)
    days = key[:, 0] // DAY * DAY
    jobs = [(d, key[days == d, 0], key[days == d, 1]) for d in np.unique(days)]
    with ProcessPoolExecutor(workers) as ex:
        vals = np.concatenate(list(ex.map(_read_day, jobs, chunksize=8)))
    pos = np.searchsorted(key[:, 0] * (points.max() + 1) + key[:, 1], hours * (points.max() + 1) + points)
    v = vals[pos]
    out = {}
    for k, (name, *_rest) in enumerate(sites):
        w, dist, la, lo, h0 = meta[name]
        z0, z1 = v[owner == 2 * k], v[owner == 2 * k + 1]
        out[name] = dict(zi=(1 - w) * z0 + w * z1, distance_km=dist, grid_lat=la, grid_lon=lo)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", nargs="+", required=True, help="flux-profile table files")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    tracks = _tracks(a.data_root)
    sites, hourly = [], []
    for path in a.tables:
        with netCDF4.Dataset(path) as d:
            name = os.path.basename(path)[:-3]  # the site attribute is not unique (both i-Box tables say ibox_kolsass)
            t = np.asarray(d["time"][:], np.int64)
            assert "since 1970-01-01" in d["time"].units and d["time"].units.startswith("seconds"), d["time"].units
            avg = int(d.averaging_minutes)
            mid = midpoint_hours(t, avg)[0] + midpoint_hours(t, avg)[1] * HOUR
            vec = track_vectors(*tracks[name](), mid) if name in tracks else np.repeat(unit_vectors(d.latitude, d.longitude), t.size, 0)
            sites.append((name, t, avg, vec))
            hours = np.arange((t.min() - avg * 60) // HOUR * HOUR, t.max() // HOUR * HOUR + HOUR + 1, HOUR)
            hvec = track_vectors(*tracks[name](), hours) if name in tracks else np.repeat(unit_vectors(d.latitude, d.longitude), hours.size, 0)
            hourly.append((name + "@hour", hours, 0, hvec))
    assert len({s[0] for s in sites}) == len(sites), "duplicate table names"
    res = era5_zi(sites + hourly, a.workers)
    os.makedirs(a.out, exist_ok=True)
    for name, t, avg, vec in sites:
        r = res[name]
        with netCDF4.Dataset(f"{a.out}/{name}_era5.nc", "w") as o:
            o.createDimension("time", t.size)
            tv = o.createVariable("time", "i8", ("time",))
            tv[:], tv.units = t, "seconds since 1970-01-01 00:00:00 UTC"
            tv.note = "end of the table's averaging interval, as in the flux-profile table"
            for k, units, desc in (("zi", "m", "ERA5 boundary-layer height, nearest grid point, linear in time to interval midpoint"),
                                   ("distance_km", "km", "distance from site to the ERA5 grid point used"),
                                   ("grid_lat", "degrees_north", "ERA5 grid point latitude"),
                                   ("grid_lon", "degrees_east", "ERA5 grid point longitude")):
                v = o.createVariable(k, "f4", ("time",), fill_value=np.float32(np.nan))
                v[:], v.units, v.long_name = r[k], units, desc
            hours = next(h[1] for h in hourly if h[0] == name + "@hour")
            o.createDimension("hour", hours.size)
            hv = o.createVariable("hour_time", "i8", ("hour",))
            hv[:], hv.units = hours, "seconds since 1970-01-01 00:00:00 UTC"
            zh = o.createVariable("zi_hourly", "f4", ("hour",), fill_value=np.float32(np.nan))
            zh[:], zh.units = res[name + "@hour"]["zi"], "m"
            zh.long_name = "ERA5 boundary-layer height at full hours, nearest grid point to the site position at that hour"
            o.site, o.position = name, "raw-file track" if name in tracks else "fixed site coordinates"
            o.source = "ERA5 hourly analysis param 159 (blh), DKRZ /pool/data/ERA5/E5/sf/an/1H/159; Hersbach et al. 2020, Copernicus C3S"
            o.choices = "GLM+codex vote, coscientist/vote_zi_result.txt"
        print(name, t.size, int(np.isfinite(r["zi"]).sum()), f"max distance {np.nanmax(r['distance_km']):.1f} km")


if __name__ == "__main__":
    main()

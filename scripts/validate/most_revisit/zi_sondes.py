"""Boundary-layer depth from radiosondes near each tower, and its comparison with ERA5 (from zi_era5.py).

Choices (GLM+codex votes, coscientist/vote_zi_result.txt): Seidel et al. (2012) bulk Richardson number 0.25
(surface wind 0, no thermal excess, theta_v at the lowest valid level, theta where humidity is missing, flagged),
linear interpolation to the crossing, NaN without a crossing below 5 km; sondes skipped if the first valid level
is above 50 m or fewer than 5 valid levels lie below 3 km; ARM sites use the ARM VAP bulk Ri 0.25 (qc 0);
sondes only within 25 km; ERA5 compared at launch time; regimes split by tower sensible heat flux sign and by sun.
"""
import argparse
import glob
import io
import os
import sys
import zipfile

import netCDF4
import numpy as np

from legoesm import constants
from legoesm.land.forcing.solar import cos_solar_zenith
from legoesm.thermo import mixing_ratio_to_specific_humidity, saturation_mixing_ratio, virtual_temperature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling module, also under python -I
import zi_era5  # noqa: E402

RI_CRIT = 0.25
MAX_FIRST_LEVEL_M = 50.0
MIN_LEVELS_BELOW_3KM = 5
TOP_M = 5000.0
EARTH_RADIUS_KM = 6371.0
IGRA_MISSING = (-9999, -8888)
MAX_DISTANCE_KM = 25.0
# site: (sonde source, path relative to data root, station id, station lat, lon, elevation m)
SONDES = {
    "cabauw": ("igra2", "igra2/NLM00006260-data.txt.zip", "De Bilt", 52.1000, 5.1833, 2.0),
    "falkenberg_2021": ("igra2", "igra2/GMM00010393-data.txt.zip", "Lindenberg", 52.2085, 14.1180, 97.7),
    "icos_fr_gri": ("igra2", "igra2/FRM00007145-data.txt.zip", "Trappes", 48.7744, 2.0097, 167.0),
    "mosaic_metcity": ("mosaic", "mosaic_sondes_l3", "MOSAiC level 3", None, None, 0.0),
    "nice2015": ("nice", "nice2015_sondes/x/rsData_gridded.nc", "N-ICE2015 sondes (npolar.2017.216df9b3)", None, None, 0.0),
    "ao2018": ("ao2018", "ao2018_sondes/x/data/ACAS_AO2018_SondeData_h10int_V02.nc", "Oden AO2018 sondes (doi:10.17043/oden-ao-2018-radiosonde-2)", None, None, 0.0),
    "arm_sgp": ("arm", "arm_pblhtsonde/sgppblhtsonde1mcfarlC1.c1", "ARM SGP C1", None, None, None),
    "arm_nsa": ("arm", "arm_pblhtsonde/nsapblhtsonde1mcfarlC1.c1", "ARM NSA C1", None, None, None),
}


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    return EARTH_RADIUS_KM * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1))


def bulk_ri_height(z, thv, u, v):
    """Seidel 2012 bulk-Ri 0.25 height above the lowest level's ground (z is height above ground), or NaN."""
    ok = np.isfinite(z) & np.isfinite(thv) & np.isfinite(u) & np.isfinite(v)
    z, thv, u, v = z[ok], thv[ok], u[ok], v[ok]
    order = np.argsort(z)
    z, thv, u, v = z[order], thv[order], u[order], v[order]
    if z.size < 2 or z[0] > MAX_FIRST_LEVEL_M or np.sum(z < 3000.0) < MIN_LEVELS_BELOW_3KM:
        return np.nan
    speed2 = np.maximum(u ** 2 + v ** 2, 1e-6)
    ri = constants.g / thv[0] * (thv - thv[0]) * (z - z[0]) / speed2
    above = np.nonzero((ri >= RI_CRIT) & (z > z[0]))[0]
    if above.size == 0:
        return np.nan
    k = above[0]
    zc = z[k - 1] + (RI_CRIT - ri[k - 1]) / (ri[k] - ri[k - 1]) * (z[k] - z[k - 1])
    return zc if zc <= TOP_M else np.nan


def theta_v(t, p, r):
    """Virtual potential temperature; r is mixing ratio (kg/kg) or NaN, in which case theta is returned."""
    theta = t * (constants.p_ref / p) ** constants.kappa
    q = mixing_ratio_to_specific_humidity(np.where(np.isfinite(r), r, 0.0))
    return np.where(np.isfinite(r), np.asarray(virtual_temperature(theta, q)), theta)


def _igra_soundings(text):
    """Yield (unix launch time, rows) per sounding from IGRA2 v2 text."""
    head, rows = None, []
    for line in text:
        if line.startswith("#"):
            if head is not None:
                yield head, rows
            y, m, d, h, rel = int(line[13:17]), int(line[18:20]), int(line[21:23]), int(line[24:26]), line[27:31]
            nominal = np.datetime64(f"{y:04d}-{m:02d}-{d:02d}", "s").astype(np.int64) + 3600 * (h if h < 24 else 0)
            hour, minute = int(rel[:2]), int(rel[2:])
            head = nominal
            if hour < 24 and minute < 60:
                # Release time can fall on the day before the nominal hour (23:30 for 00 UTC).
                diff = (nominal - nominal % 86400 + 3600 * hour + 60 * minute) - nominal
                head = nominal + diff - 86400 * (diff > 43200) + 86400 * (diff < -43200)
            rows = []
        else:
            rows.append(line)
    if head is not None:
        yield head, rows


def _num(s, scale=1.0):
    s = s.strip().rstrip("AB")
    if not s or int(s) in IGRA_MISSING:
        return np.nan
    return int(s) * scale


def igra_profile(rows, station_elevation):
    """Height above ground (hypsometric from the surface level), theta_v, u, v at levels with temperature."""
    p = np.array([_num(r[9:15]) for r in rows])
    t = np.array([_num(r[22:27], 0.1) for r in rows]) + constants.T_freeze
    dpd = np.array([_num(r[34:39], 0.1) for r in rows])
    wd = np.array([_num(r[40:45]) for r in rows])
    ws = np.array([_num(r[46:51], 0.1) for r in rows])
    surface = np.array([r[1] == "1" for r in rows])
    if not surface.any() or not np.isfinite(p[surface][0]) or not np.isfinite(t[surface][0]):
        return None
    good_p = np.isfinite(p)
    p, t, dpd, wd, ws, surface = p[good_p], t[good_p], dpd[good_p], wd[good_p], ws[good_p], surface[good_p]
    order = np.argsort(-p)
    p, t, dpd, wd, ws, surface = p[order], t[order], dpd[order], wd[order], ws[order], surface[order]
    p0 = p[surface][0]
    keep = p <= p0
    p, t, dpd, wd, ws = p[keep], t[keep], dpd[keep], wd[keep], ws[keep]
    has_t = np.isfinite(t)
    if has_t.sum() < 2:
        return None
    lnp = np.log(p)
    t_all = np.interp(-lnp, -lnp[has_t], t[has_t])
    td = t - dpd
    r = np.asarray(saturation_mixing_ratio(td, p))
    r = np.where(np.isfinite(td), r, np.nan)
    tv = np.asarray(virtual_temperature(t_all, mixing_ratio_to_specific_humidity(np.where(np.isfinite(r), r, 0.0))))
    dz = constants.R_d / constants.g * 0.5 * (tv[1:] + tv[:-1]) * (lnp[:-1] - lnp[1:])
    z = np.concatenate([[0.0], np.cumsum(dz)])
    has_w = np.isfinite(wd) & np.isfinite(ws)
    if has_w.sum() < 2:
        return None
    rad = np.deg2rad(wd[has_w])
    u = np.interp(z, z[has_w], -ws[has_w] * np.sin(rad))
    v = np.interp(z, z[has_w], -ws[has_w] * np.cos(rad))
    u[0] = v[0] = 0.0
    return z[has_t], theta_v(t[has_t], p[has_t], r[has_t]), u[has_t], v[has_t], bool(np.isfinite(r[has_t]).all())


def igra_zi(path, start, end, elevation):
    with zipfile.ZipFile(path) as zf, zf.open(zf.namelist()[0]) as f:
        text = io.TextIOWrapper(f, encoding="ascii")
        out = []
        for t, rows in _igra_soundings(text):
            if not start <= t <= end:
                continue
            prof = igra_profile(rows, elevation)
            out.append((int(t), np.nan if prof is None else bulk_ri_height(*prof[:4]), prof is not None and prof[4], np.nan, np.nan))
    return out


def mosaic_zi(root, start, end):
    import pandas as pd
    out = []
    for path in sorted(glob.glob(f"{root}/*.tab")):
        with open(path) as f:
            skip = next(i for i, line in enumerate(f) if line.startswith("*/")) + 1
        cols = [0, 1, 2, 8, 10, 12, 16, 22, 30, 32]
        d = pd.read_csv(path, sep="\t", skiprows=skip, usecols=cols, header=0)
        d.columns = ["event", "time", "z", "lon", "lat", "p", "t", "r", "dd", "ff"]
        for _, s in d.groupby("event", sort=False):
            t0 = np.datetime64(pd.Timestamp(s["time"].iloc[0]).to_datetime64(), "s").astype(np.int64)
            if not start <= t0 <= end:
                continue
            s = s[np.isfinite(s["z"]) & (s["z"].diff().fillna(1) > 0)]
            if len(s) < 2:
                out.append((int(t0), np.nan, False, np.nan, np.nan))
                continue
            rad = np.deg2rad(s["dd"].to_numpy())
            u, v = -s["ff"].to_numpy() * np.sin(rad), -s["ff"].to_numpy() * np.cos(rad)
            u[0] = v[0] = 0.0
            r = s["r"].to_numpy() * 1e-6
            out.append((int(t0), bulk_ri_height(s["z"].to_numpy(), theta_v(s["t"].to_numpy(), s["p"].to_numpy() * 100.0, r), u, v),
                        bool(np.isfinite(r).all()), float(s["lat"].iloc[0]), float(s["lon"].iloc[0])))
    return out


def nice_zi(path, start, end):
    """N-ICE2015 sondes gridded at 25 m above the surface; humidity as RH over water."""
    out = []
    with netCDF4.Dataset(path) as d:
        z = np.asarray(d["height"][:], float)
        lt = np.ma.filled(d["lt"][:].astype(float), np.nan)
        get = lambda k: np.ma.filled(d[k][:].astype(float), np.nan)
        p, t, rh, wd, ws = get("press") * 100.0, get("temp") + constants.T_freeze, get("rh"), get("wd"), get("ws")
        slat, slon = get("lat"), get("lon")
    for i in range(lt.shape[1]):
        if not np.isfinite(lt[:, i]).all():
            continue  # launch without a time stamp
        y, mo, dd, hh, mi, se = lt[:, i].astype(int)
        t0 = np.datetime64(f"{y:04d}-{mo:02d}-{dd:02d}T{hh:02d}:{mi:02d}:{se:02d}", "s").astype(np.int64)
        if not start <= t0 <= end:
            continue
        r = rh[i] / 100.0 * np.asarray(saturation_mixing_ratio(t[i], p[i]))
        rad = np.deg2rad(wd[i])
        u, v = -ws[i] * np.sin(rad), -ws[i] * np.cos(rad)
        u[0] = v[0] = 0.0
        out.append((int(t0), bulk_ri_height(z, theta_v(t[i], p[i], r), u, v), bool(np.isfinite(r[np.isfinite(t[i])]).all()),
                    slat[i], slon[i]))
    return out


def ao2018_zi(path, start, end):
    """Oden AO2018 sondes on a 10 m grid above sea level."""
    out = []
    with netCDF4.Dataset(path) as d:
        t0s = np.asarray(d["time"][:], float).round().astype(np.int64)
        z = np.asarray(d["Z"][:], float)
        get = lambda k: np.ma.filled(d[k][:].astype(float), np.nan)
        p, t, r, u, v = get("pressure") * 100.0, get("temperature") + constants.T_freeze, get("mr") * 1e-3, get("u"), get("v")
        lat, lon = get("lat"), get("lon")
    for i, t0 in enumerate(t0s):
        if not start <= t0 <= end:
            continue
        first = np.flatnonzero(np.isfinite(lat[i]) & np.isfinite(lon[i]))
        la, lo = (lat[i][first[0]], lon[i][first[0]]) if first.size else (np.nan, np.nan)
        out.append((int(t0), bulk_ri_height(z, theta_v(t[i], p[i], r[i]), u[i], v[i]), bool(np.isfinite(r[i][np.isfinite(t[i])]).all()),
                    la, lo))
    return out


def arm_zi(root, start, end):
    out = []
    for path in sorted(glob.glob(f"{root}/*")):
        with netCDF4.Dataset(path) as d:
            if "pbl_height_bulk_richardson_pt25" not in d.variables:
                continue
            # base_time is midnight of the launch day in these files; the launch is base_time + time_offset[0].
            t = int(d["base_time"][:]) + int(round(float(d["time_offset"][0])))
            if not start <= t <= end:
                continue
            z = float(np.ma.filled(d["pbl_height_bulk_richardson_pt25"][:], np.nan)) - float(d["alt"][:])
            ok = int(np.ma.filled(d["qc_pbl_height_bulk_richardson_pt25"][:], 1)) == 0
            out.append((t, z if ok else np.nan, True, float(d["lat"][:]), float(d["lon"][:])))
    return out


def compare(table_path, era5_path, sondes, tower_lat, tower_lon):
    """Per valid sonde: (launch, sonde zi, ERA5 zi at launch, tower H, sun up, theta_v used); ERA5 linear between hours."""
    with netCDF4.Dataset(table_path) as d:
        t_end = np.asarray(d["time"][:], np.int64)
        avg = int(d.averaging_minutes)
        h = np.ma.filled(d["h"][:], np.nan)
    with netCDF4.Dataset(era5_path) as e:
        hours = np.asarray(e["hour_time"][:], np.int64)
        era = np.ma.filled(e["zi_hourly"][:], np.nan)
    first = np.argmax(np.isfinite(h), axis=1)
    h_low = h[np.arange(h.shape[0]), first]
    rows = []
    for (t, zs, moist, *_), la, lo in zip(sondes, tower_lat, tower_lon):
        k = np.searchsorted(t_end, t)
        if not np.isfinite(zs) or k >= t_end.size or t_end[k] - avg * 60 > t:
            continue
        j = np.searchsorted(hours, t, side="right") - 1
        if j < 0 or j + 1 >= hours.size or hours[j + 1] - hours[j] != 3600:
            continue
        ze = era[j] + (t - hours[j]) / 3600.0 * (era[j + 1] - era[j])
        if not np.isfinite(ze):
            continue
        year = np.datetime64(int(t), "s").astype("datetime64[Y]").astype("datetime64[s]").astype(np.int64)
        sun = float(cos_solar_zenith(np.deg2rad(la), np.deg2rad(lo), (t - year) / 86400.0 + 1, (t % 86400) / 3600.0)) > 0
        rows.append((t, zs, ze, h_low[k], sun, moist))
    return np.array(rows, float).reshape(-1, 6)


def stats(rows):
    out = {}
    for name, sel in (("H>0", rows[:, 3] > 0), ("H<0", rows[:, 3] < 0), ("day", rows[:, 4] > 0), ("night", rows[:, 4] == 0)):
        d = rows[sel, 2] - rows[sel, 1]
        out[name] = dict(n=int(sel.sum()), bias_m=float(np.median(d)) if d.size else np.nan,
                         scatter_m=float(np.subtract(*np.percentile(d, [75, 25]))) if d.size else np.nan,
                         sonde_median_m=float(np.median(rows[sel, 1])) if d.size else np.nan)
    return out


def main():
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--zi-dir", required=True)
    ap.add_argument("--figure", required=True)
    a = ap.parse_args()
    tables = {os.path.basename(p)[:-3]: p for p in a.tables}
    summary, rows_all = {}, {}
    tracks = zi_era5._tracks(a.data_root)
    for site, (kind, rel, station, slat, slon, elev) in SONDES.items():
        if site not in tables:
            continue
        with netCDF4.Dataset(tables[site]) as d:
            t = np.asarray(d["time"][:], np.int64)
            lat, lon = float(d.latitude), float(d.longitude)
        path = f"{a.data_root}/{rel}"
        sondes = (igra_zi(path, int(t.min()) - 3600, int(t.max()), elev) if kind == "igra2"
                  else mosaic_zi(path, int(t.min()) - 3600, int(t.max())) if kind == "mosaic"
                  else nice_zi(path, int(t.min()) - 3600, int(t.max())) if kind == "nice"
                  else ao2018_zi(path, int(t.min()) - 3600, int(t.max())) if kind == "ao2018"
                  else arm_zi(path, int(t.min()) - 3600, int(t.max())))
        launch = np.array([x[0] for x in sondes], float)
        if site in tracks:
            vec = zi_era5.track_vectors(*tracks[site](), launch)
            tlat, tlon = np.rad2deg(np.arcsin(vec[:, 2])), np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0]))
        else:
            tlat, tlon = np.full(launch.size, lat), np.full(launch.size, lon)
        olat = np.array([slat if slat is not None else x[3] for x in sondes], float)
        olon = np.array([slon if slon is not None else x[4] for x in sondes], float)
        dist = great_circle_km(tlat, tlon, olat, olon)
        # Sondes farther than the voted limit, or with unknown distance, are kept in the file but not compared.
        near = [(x[0], x[1] if d <= MAX_DISTANCE_KM else np.nan, x[2]) for x, d in zip(sondes, dist)]
        rows = compare(tables[site], f"{a.zi_dir}/{site}_era5.nc", near, tlat, tlon)
        rows_all[site] = rows
        with netCDF4.Dataset(f"{a.zi_dir}/{site}_sondes.nc", "w") as o:
            o.createDimension("sonde", len(sondes))
            for k, typ, vals, units in (("launch_time", "i8", launch.astype(np.int64), "seconds since 1970-01-01 00:00:00 UTC"),
                                        ("zi", "f4", [x[1] for x in sondes], "m above ground"),
                                        ("distance_km", "f4", dist, "km from tower at launch"),
                                        ("theta_v_used", "i1", [int(x[2]) for x in sondes], "1 if humidity entered theta_v")):
                v = o.createVariable(k, typ, ("sonde",))
                v[:], v.units = vals, units
            o.site, o.source, o.station = site, kind, station
            o.method = "ARM VAP bulk Ri 0.25 (qc 0)" if kind == "arm" else "Seidel et al. 2012 bulk Ri 0.25, own code"
        summary[site] = dict(station=station, distance_km_median=round(float(np.nanmedian(dist)), 1) if dist.size else None,
                             sondes=len(sondes), valid=int(np.isfinite([x[1] for x in near]).sum()), matched=int(rows.shape[0]),
                             **stats(rows))
        print(site, json.dumps(summary[site]))
    with open(a.figure.replace(".png", ".json"), "w") as f:
        json.dump(summary, f, indent=1)
    plot(rows_all, summary, a.figure)


def plot(rows_all, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = [s for s in rows_all if rows_all[s].shape[0]]
    fig, axes = plt.subplots(1, max(len(sites), 1), figsize=(3.2 * max(len(sites), 1), 3.4), squeeze=False)
    for ax, s in zip(axes[0], sites):
        r = rows_all[s]
        for sel, colour, label in ((r[:, 3] > 0, "#E69F00", "H>0"), (r[:, 3] < 0, "#0072B2", "H<0")):
            ax.loglog(r[sel, 1], r[sel, 2], ".", ms=2, color=colour, alpha=0.4, label=f"{label} n={sel.sum()}")
        ax.plot([10, 5000], [10, 5000], color="#000000", lw=0.8)
        ax.set_xlim(10, 5000), ax.set_ylim(10, 5000)
        ax.set_title(f"{s}\n{summary[s]['station']} {summary[s]['distance_km_median']} km", fontsize=8)
        ax.set_xlabel("sonde zi (m)")
        ax.legend(fontsize=6, markerscale=4)
    axes[0][0].set_ylabel("ERA5 zi (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()

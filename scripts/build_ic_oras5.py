"""Ocean initial state (T, S) from ORAS5, written in the format of the existing WOA init files.

Output = template file (variable name, dims, coords, units, attrs) with values replaced by ORAS5.
ORAS5: ICDC 1x1 (regular) or ORCA025 (nav_lat/nav_lon); several files (members) are averaged.
ICDC r1x1 stores land / sea floor as 0 and its 1x1 area average mixes these zeros into the
neighbouring cells, so cells with S <= 0 and their neighbours (--grow cells) are treated as missing
(for T via --mask-from <vosaline file>) and filled from the nearest valid cell on the same level.
Vertical: linear in depth to the template levels.  Units follow the template (degC or K).
"""
import argparse

import numpy as np


def _coord(ds, names):
    for n in list(ds.coords) + list(ds.variables):
        if n.lower() in names:
            return n
    return None


def _xyz(la, lo):
    la, lo = np.deg2rad(la), np.deg2rad(lo)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _bad_from_salinity(sv, grow):
    from scipy import ndimage
    bad = ~(np.isfinite(sv) & (sv > 0.0) & (np.abs(sv) < 1e10))
    if grow > 0:
        st = np.ones((3, 3), bool)
        bad = np.stack([ndimage.binary_dilation(b, structure=st, iterations=grow) for b in bad])
    return bad


def read_oras5(paths, var, mask_paths=None, grow=1):
    import xarray as xr
    fields, sal = [], []
    for p in paths:
        ds = xr.open_dataset(p, decode_times=False)
        da = ds[var]
        tdim = next((d for d in da.dims if "time" in d.lower()), None)
        if tdim:
            da = da.isel({tdim: 0})
        zdim = next(d for d in da.dims if d.lower().startswith(("dep", "lev", "z")))
        z = np.asarray(ds[zdim].values if zdim in ds else ds[_coord(ds, ("deptht", "depth", "lev"))].values, float)
        v = np.asarray(da.transpose(zdim, ...).values, float)
        v = np.where(np.abs(v) > 1e10, np.nan, v)
        latn = _coord(ds, ("nav_lat", "lat", "latitude"))
        lonn = _coord(ds, ("nav_lon", "lon", "longitude"))
        la, lo = np.asarray(ds[latn].values, float), np.asarray(ds[lonn].values, float)
        if var == "vosaline":
            sal.append(v)
        fields.append(v)
    for mp in (mask_paths or []):
        ms = xr.open_dataset(mp, decode_times=False)["vosaline"]
        tdm = next((d for d in ms.dims if "time" in d.lower()), None)
        if tdm:
            ms = ms.isel({tdm: 0})
        zm = next(d for d in ms.dims if d.lower().startswith(("dep", "lev", "z")))
        sal.append(np.asarray(ms.transpose(zm, ...).values, float))
    if sal:
        bad = np.any(np.stack([_bad_from_salinity(sv, grow) for sv in sal]), axis=0)
        fields = [np.where(bad, np.nan, f) for f in fields]
    v = np.nanmean(np.stack(fields), axis=0) if len(fields) > 1 else fields[0]
    return np.abs(z), la, lo, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", required=True)
    ap.add_argument("--oras5", required=True, nargs="+")
    ap.add_argument("--mask-from", nargs="+", default=None)
    ap.add_argument("--grow", type=int, default=1)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial import cKDTree

    zs, la_s, lo_s, v = read_oras5(a.oras5, a.var, a.mask_from, a.grow)
    print(f"  valid ORAS5 cells per level: surface {int(np.isfinite(v[0]).sum())}, "
          f"~1000 m {int(np.isfinite(v[int(np.argmin(np.abs(zs - 1000)))]).sum())}, "
          f"~3000 m {int(np.isfinite(v[int(np.argmin(np.abs(zs - 3000)))]).sum())}  (grow={a.grow})")
    print(f"ORAS5 {a.var}: {len(a.oras5)} file(s), {v.shape[0]} levels {zs[0]:.1f}..{zs[-1]:.0f} m, "
          f"grid {'regular' if la_s.ndim == 1 else 'curvilinear'} {v.shape[1:]}, range {np.nanmin(v):.2f}..{np.nanmax(v):.2f}")

    tp = xr.open_dataset(a.template, decode_times=False)
    tvar = next(n for n in tp.data_vars if tp[n].ndim >= 3)
    td = tp[tvar]
    tz = next(d for d in td.dims if d.lower().startswith(("dep", "lev", "z", "plev", "olevel")))
    tla = next(d for d in td.dims if d.lower().startswith("lat"))
    tlo = next(d for d in td.dims if d.lower().startswith("lon"))
    zt = np.abs(np.asarray(tp[tz].values, float))
    zt_m = zt / 100.0 if np.nanmax(zt) > 20000 else zt
    lat_t, lon_t = np.asarray(tp[tla].values, float), np.asarray(tp[tlo].values, float)
    LA, LO = np.meshgrid(lat_t, lon_t, indexing="ij")

    hz = np.full((zs.size, lat_t.size, lon_t.size), np.nan)
    if la_s.ndim == 1:
        oi = np.argsort(la_s); lo_m = np.mod(lo_s, 360.0); oj = np.argsort(lo_m)
        la1, lo1 = la_s[oi], lo_m[oj]
        lop = np.concatenate([[lo1[-1] - 360.0], lo1, [lo1[0] + 360.0]])
        q = np.stack([np.clip(LA, la1.min(), la1.max()), np.mod(LO, 360.0)], -1)
        for k in range(zs.size):
            f = v[k][oi][:, oj]
            fp = np.concatenate([f[:, -1:], f, f[:, :1]], axis=1)
            hz[k] = RegularGridInterpolator((la1, lop), fp, bounds_error=False, fill_value=np.nan)(q)
            nn = RegularGridInterpolator((la1, lop), fp, method="nearest", bounds_error=False, fill_value=np.nan)(q)
            hz[k] = np.where(np.isfinite(hz[k]), hz[k], nn)
    else:
        src = _xyz(la_s.ravel(), lo_s.ravel())
        dst = _xyz(LA.ravel(), LO.ravel())
        _, jj = cKDTree(src).query(dst)
        for k in range(zs.size):
            f = v[k].ravel(); ok = np.isfinite(f)
            if ok.sum() == 0:
                continue
            _, ii = cKDTree(src[ok]).query(dst)
            hz[k] = f[ok][ii].reshape(LA.shape)
            hz[k] = np.where(np.isfinite(f[jj]).reshape(LA.shape), hz[k], np.nan)

    for k in range(zs.size):
        ok = np.isfinite(hz[k])
        if ok.sum() == 0:
            hz[k] = hz[k - 1] if k > 0 else 0.0
            continue
        if (~ok).any():
            _, ii = cKDTree(_xyz(LA[ok], LO[ok])).query(_xyz(LA[~ok], LO[~ok]))
            hz[k][~ok] = hz[k][ok][ii]

    out = np.empty((zt_m.size, lat_t.size, lon_t.size))
    for kk, zz in enumerate(zt_m):
        if zz <= zs[0]:
            out[kk] = hz[0]
        elif zz >= zs[-1]:
            out[kk] = hz[-1]
        else:
            j = int(np.searchsorted(zs, zz) - 1)
            w = (zz - zs[j]) / (zs[j + 1] - zs[j])
            out[kk] = (1 - w) * hz[j] + w * hz[j + 1]

    tmed = float(np.nanmedian(np.asarray(td.values, float)))
    if a.var == "votemper" and tmed > 100:
        out = out + 273.15
        print("  template T is in K: converted")
    order = [d for d in td.dims]
    if len(order) == 4:
        tdim = next(d for d in order if d not in (tz, tla, tlo))
        arr = np.broadcast_to(out[None], (td.sizes[tdim],) + out.shape)
        arr = np.transpose(arr, [(tdim, tz, tla, tlo).index(d) for d in order])
    else:
        arr = np.transpose(out, [(tz, tla, tlo).index(d) for d in order])
    new = tp.copy()
    new[tvar] = (order, arr.astype(td.dtype), td.attrs)
    new.attrs = dict(tp.attrs)
    new.attrs.update(source=" ".join(a.oras5),
                     history=f"ORAS5 {a.var} on the grid/levels of {a.template} (build_ic_oras5.py, grow={a.grow})")
    new.to_netcdf(a.out)
    tv = np.asarray(td.values, float).reshape(-1)
    print(f"wrote {a.out}: {tvar} {arr.shape}; ORAS5 range {np.nanmin(out):.2f}..{np.nanmax(out):.2f}, "
          f"template range {np.nanmin(tv):.2f}..{np.nanmax(tv):.2f}")
    w_ = np.cos(np.deg2rad(LA))
    tvals = np.asarray(td.transpose(*([d for d in order if d not in (tz, tla, tlo)] + [tz, tla, tlo])).values, float)
    tvals = tvals.reshape((-1,) + out.shape)[0]
    off = 273.15 if (a.var == "votemper" and tmed > 100) else 0.0
    for zz in (0, 100, 500, 1000, 3000):
        k = int(np.argmin(np.abs(zt_m - zz)))
        okk = np.isfinite(tvals[k])
        if okk.any():
            dd = (out[k] - off) - (tvals[k] - off)
            print(f"  {zt_m[k]:7.0f} m: ORAS5 - template mean {np.sum(dd[okk] * w_[okk]) / np.sum(w_[okk]):+.3f}, "
                  f"rms {np.sqrt(np.sum(dd[okk] ** 2 * w_[okk]) / np.sum(w_[okk])):.3f}")


if __name__ == "__main__":
    main()

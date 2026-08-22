#!/usr/bin/env python
"""Is a correct equatorial SST correct dynamics, or two errors cancelling?

A model can land on the right cold-tongue temperature for the wrong reason: if
it over-mixes vertically, the extra entrainment cools the equator and can
cancel a structural warm bias, leaving a good SST on top of a thermocline that
is too deep and too diffuse.  The second reviewer named the discriminator --
equatorial mixed-layer depth, the depth of the 20 C isotherm, and the SHARPNESS
of the thermocline, all against the oracle at the same time.  A model whose SST
matches while its thermocline is deep and smeared is right by cancellation; one
whose SST matches with the oracle's thermocline structure is right.

This reads a snapshot from EITHER grid layout -- the structured curvilinear
tripole, shape (ny, nx, nlev), or an unstructured node cloud, shape (n, nlev) --
because the whole point is to compare the two, and reduces both to the same
per-longitude band statistic on their OWN grids.  No regridding: within two
degrees of the equator the area weight cos(lat) varies by less than 0.07%, so a
band mean is a band mean either way, and a horizontal remap would add an
interpolation error to the quantity under test.

Z20 comes from ``equatorial_thermocline.z20_from_column`` -- the same helper the
sibling probe uses, not a second implementation.

READ IT AS A DISCRIMINATOR, NOT A VERDICT.  Matching SST with a too-deep,
too-diffuse thermocline points at cancellation.  Matching SST with matching Z20
and matching sharpness points at the dynamics being right.  Anything else needs
the momentum budget, not this.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load_z20_helper():
    """Reuse the sibling probe's isotherm-depth helper (no re-derivation)."""
    spec = importlib.util.spec_from_file_location(
        "_eqtherm", _HERE / "equatorial_thermocline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_eqtherm"] = mod
    sys.path.insert(0, str(_HERE.parents[1]))      # scripts/validate on path
    spec.loader.exec_module(mod)
    return mod.z20_from_column


def _flatten(snapshot: Path):
    """(T, lat, lon, wet, z) as point clouds, from either grid layout."""
    z = np.load(snapshot)
    T = np.asarray(z["T"], dtype=np.float64)
    lat = np.asarray(z["lat_T"], dtype=np.float64)
    lon = np.asarray(z["lon_T"], dtype=np.float64)
    wet = np.asarray(z["land_mask"], dtype=np.float64)
    zc = np.abs(np.asarray(z["z_center_ref"], dtype=np.float64))
    nlev = zc.size
    if T.ndim == 3:                       # (ny, nx, nlev) structured
        T = T.reshape(-1, nlev)
        lat, lon, wet = lat.ravel(), lon.ravel(), wet.ravel()
    elif T.ndim != 2:                     # (n, nlev) node cloud
        raise SystemExit(f"unexpected T shape {T.shape} in {snapshot}")
    if lat.max() <= np.pi + 1e-6 and lat.min() >= -np.pi - 1e-6:
        # Some writers store radians. Degrees is the convention here; a mesh
        # that never leaves +-pi degrees does not exist, so this is safe.
        lat, lon = np.degrees(lat), np.degrees(lon)
    return T, lat, lon % 360.0, wet > 0.5, zc


def _nemo_columns(gridt: Path, rec: int):
    import xarray as xr
    ds = xr.open_dataset(gridt, decode_times=False)
    tvar = next((v for v in ("to", "thetao", "votemper", "toce")
                 if v in ds and ds[v].ndim >= 3), None)
    if tvar is None:
        raise SystemExit(f"{gridt}: no 3-D temperature")
    T = np.asarray(ds[tvar].values, dtype=np.float64)
    if T.ndim == 4:
        T = T[rec]
    T = np.where(np.abs(T) > 1e10, np.nan, T)
    T = np.where(T == 0.0, np.nan, T)                 # NEMO land sentinel
    T = np.moveaxis(T, 0, -1)                         # (y, x, z)
    zc = np.abs(np.asarray(ds["deptht"].values, dtype=np.float64))
    lat = np.asarray(ds["nav_lat"].values, dtype=np.float64).ravel()
    lon = np.asarray(ds["nav_lon"].values, dtype=np.float64).ravel() % 360.0
    T = T.reshape(-1, zc.size)
    return T, lat, lon, np.isfinite(T[:, 0]), zc


def _max_dtdz(T: np.ndarray, zc: np.ndarray, depth_max: float) -> np.ndarray:
    """Peak vertical temperature gradient [K/m] in the top ``depth_max``.

    The thermocline's sharpness.  Over-mixing smears it; that is the whole
    signal this probe exists to see.
    """
    k = int(np.searchsorted(zc, depth_max)) + 1
    k = max(2, min(k, zc.size))
    dT = np.diff(T[:, :k], axis=1)
    dz = np.diff(zc[:k])[None, :]
    g = np.abs(dT / dz)
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.where(np.isfinite(g), g, np.nan), axis=1)


def _band_table(T, lat, lon, wet, zc, z20_fn, halfwidth, bins, depth_max):
    sel = wet & (np.abs(lat) <= halfwidth)
    rows = []
    for lo, hi in bins:
        m = sel & (lon >= lo) & (lon < hi)
        if not m.any():
            rows.append((lo, hi, np.nan, np.nan, np.nan, 0))
            continue
        Tm = T[m]
        z20 = np.array([z20_fn(col, zc) for col in Tm], dtype=np.float64)
        sharp = _max_dtdz(Tm, zc, depth_max)
        rows.append((lo, hi,
                     float(np.nanmean(Tm[:, 0])),
                     float(np.nanmean(z20)),
                     float(np.nanmean(sharp)),
                     int(m.sum())))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, type=Path, action="append",
                    help="repeat for each model; pair with --label")
    ap.add_argument("--label", required=True, action="append")
    ap.add_argument("--nemo-gridt", required=True, type=Path)
    ap.add_argument("--nemo-rec", type=int, default=5,
                    help="record in --nemo-gridt (5-day file: 5 = days 26-30)")
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--depth-max", type=float, default=400.0,
                    help="depth over which the sharpness maximum is taken")
    a = ap.parse_args()
    if len(a.snapshot) != len(a.label):
        raise SystemExit("--snapshot and --label must be given in pairs")

    z20_fn = _load_z20_helper()
    bins = [(lo, lo + 20.0) for lo in np.arange(140.0, 280.0, 20.0)]

    print(f"Equatorial band |lat| <= {a.lat_halfwidth} deg, on each model's OWN "
          f"grid (no regrid; cos(lat) varies < 0.1% across the band).")
    print(f"SST [C], Z20 = depth of the 20 C isotherm [m], sharpness = mean of "
          f"the per-column max |dT/dz| over the top {a.depth_max:.0f} m [K/m].")
    print("A matching SST with a DEEPER and SMOOTHER thermocline is the "
          "cancellation signature.\n")

    tables = {}
    for snap, lab in zip(a.snapshot, a.label):
        T, lat, lon, wet, zc = _flatten(snap)
        tables[lab] = _band_table(T, lat, lon, wet, zc, z20_fn,
                                  a.lat_halfwidth, bins, a.depth_max)
        print(f"[{lab}] {snap}  ({int(wet.sum())} wet columns, {zc.size} levels)")
    T, lat, lon, wet, zc = _nemo_columns(a.nemo_gridt, a.nemo_rec)
    tables["NEMO"] = _band_table(T, lat, lon, wet, zc, z20_fn,
                                 a.lat_halfwidth, bins, a.depth_max)
    print(f"[NEMO] {a.nemo_gridt} record {a.nemo_rec}\n")

    order = [*a.label, "NEMO"]
    for field, idx, unit in (("SST", 2, "C"), ("Z20", 3, "m"),
                             ("sharpness", 4, "K/m")):
        print(f"--- {field} [{unit}] ---")
        print(f"{'lon':>12}" + "".join(f"{o:>14}" for o in order))
        for j, (lo, hi) in enumerate(bins):
            cells = "".join(f"{tables[o][j][idx]:>14.4g}" for o in order)
            print(f"{f'{lo:.0f}-{hi:.0f}E':>12}{cells}")
        print(f"{'band mean':>12}" + "".join(
            f"{np.nanmean([r[idx] for r in tables[o]]):>14.4g}" for o in order))
        print()

    print("counts per bin (columns entering each mean):")
    print(f"{'lon':>12}" + "".join(f"{o:>14}" for o in order))
    for j, (lo, hi) in enumerate(bins):
        print(f"{f'{lo:.0f}-{hi:.0f}E':>12}"
              + "".join(f"{tables[o][j][5]:>14d}" for o in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

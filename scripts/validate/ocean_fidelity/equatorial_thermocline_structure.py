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
per-longitude band statistic on their OWN grids -- no regridding, since a
horizontal remap would add an interpolation error to the quantity under test.

WEIGHTING.  Each column is weighted by the model's OWN cell area when the
snapshot carries it, and by cos(lat) only as a fallback, which the probe
prints.  An earlier version argued that cos(lat) hardly varies within two
degrees of the equator and therefore weighting did not matter.  That reasoning
was wrong: the issue is not how cos(lat) varies but that native CELL SIZES
differ across a stretched curvilinear grid and, far more, across a non-uniform
node cloud, so an unweighted mean of native columns compares sampling rather
than ocean (codex 2026-08-22).

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
    """(T, lat, lon, wet, z, area) as point clouds, from either grid layout.

    ``area`` is the model's own cell area when the snapshot carries it, else
    None -- the caller falls back to cos(lat) and SAYS SO.  An unweighted mean
    over native columns is not a band mean on either a stretched curvilinear
    grid or a non-uniform node cloud, and comparing two such means across
    meshes reports a sampling difference as a physical one.
    """
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
    if (np.nanmax(np.abs(lat)) <= np.pi + 1e-6
            and np.nanmax(np.abs(lon)) <= 2.0 * np.pi + 1e-6):
        # Some writers store radians.  Degrees is the convention here.  BOTH
        # coordinates have to look like radians before converting: a snapshot
        # covering only the equatorial band has |lat| of a few degrees, which
        # on its own is indistinguishable from radians, and converting it
        # scattered a 230 E column to 218 E.  A degree longitude field spans
        # far more than 2*pi, so the pair is decisive where lat alone is not.
        lat, lon = np.degrees(lat), np.degrees(lon)
    area = np.asarray(z["cell_area"], dtype=np.float64).ravel() \
        if "cell_area" in z else None
    return T, lat, lon % 360.0, wet > 0.5, zc, area


def _nemo_area(ds, lat):
    """NEMO cell area if the file carries it, else None."""
    for name in ("area", "areacello", "e1t"):
        if name in ds:
            a = np.asarray(ds[name].values, dtype=np.float64)
            if name == "e1t" and "e2t" in ds:
                a = a * np.asarray(ds["e2t"].values, dtype=np.float64)
            elif name == "e1t":
                return None
            return np.squeeze(a).ravel()
    return None


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
    return T, lat, lon, np.isfinite(T[:, 0]), zc, _nemo_area(ds, lat)


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


def _band_table(T, lat, lon, wet, zc, z20_fn, halfwidth, bins, depth_max,
                area=None):
    w_all = area if area is not None else np.cos(np.deg2rad(lat))
    sel = wet & (np.abs(lat) <= halfwidth)
    rows = []
    for lo, hi in bins:
        m = sel & (lon >= lo) & (lon < hi)
        if not m.any():
            rows.append((lo, hi, np.nan, np.nan, np.nan, 0))
            continue
        Tm = T[m]
        w = np.asarray(w_all)[m]
        z20 = np.array([z20_fn(col, zc) for col in Tm], dtype=np.float64)
        sharp = _max_dtdz(Tm, zc, depth_max)

        def wmean(v):
            ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
            return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() \
                else float("nan")

        rows.append((lo, hi, wmean(Tm[:, 0]), wmean(z20), wmean(sharp),
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
    ap.add_argument("--snapshot-early", type=Path, action="append",
                    default=None,
                    help="Earlier snapshot per model, in the SAME order as "
                         "--snapshot. Turns the tables into a TENDENCY: the "
                         "change from the early state to the late one, which "
                         "is what a still-drifting bias has to be judged on. "
                         "A day-30 value of a growing drift is a rate caught "
                         "mid-flight, not an equilibrium bias.")
    ap.add_argument("--nemo-rec-early", type=int, default=None,
                    help="the matching NEMO record for --snapshot-early "
                         "(5-day file: 1 = days 6-10)")
    a = ap.parse_args()
    if len(a.snapshot) != len(a.label):
        raise SystemExit("--snapshot and --label must be given in pairs")
    if bool(a.snapshot_early) != (a.nemo_rec_early is not None):
        raise SystemExit(
            "--snapshot-early and --nemo-rec-early go together: a tendency "
            "with only one side moving is not a comparison.")
    if a.snapshot_early and len(a.snapshot_early) != len(a.snapshot):
        raise SystemExit(
            "--snapshot-early must be given once per --snapshot, in the same "
            "order")

    z20_fn = _load_z20_helper()
    bins = [(lo, lo + 20.0) for lo in np.arange(140.0, 280.0, 20.0)]

    print(f"Equatorial band |lat| <= {a.lat_halfwidth} deg, on each model's OWN "
          f"grid (no regrid; each column weighted by its own cell area where "
          f"the snapshot carries one, cos(lat) otherwise -- printed below).")
    print(f"SST [C], Z20 = depth of the 20 C isotherm [m], sharpness = mean of "
          f"the per-column max |dT/dz| over the top {a.depth_max:.0f} m [K/m].")
    print("A matching SST with a DEEPER and SMOOTHER thermocline is the "
          "cancellation signature.\n")

    tables = {}
    for snap, lab in zip(a.snapshot, a.label):
        T, lat, lon, wet, zc, area = _flatten(snap)
        tables[lab] = _band_table(T, lat, lon, wet, zc, z20_fn,
                                  a.lat_halfwidth, bins, a.depth_max, area)
        print(f"[{lab}] {snap}  ({int(wet.sum())} wet columns, {zc.size} "
              f"levels, weight={'cell_area' if area is not None else 'cos(lat)'})")
    T, lat, lon, wet, zc, area = _nemo_columns(a.nemo_gridt, a.nemo_rec)
    tables["NEMO"] = _band_table(T, lat, lon, wet, zc, z20_fn,
                                 a.lat_halfwidth, bins, a.depth_max, area)
    print(f"[NEMO] weight={'cell_area' if area is not None else 'cos(lat)'}")
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

    if a.snapshot_early:
        early = {}
        for snap, lab in zip(a.snapshot_early, a.label):
            T, lat, lon, wet, zc, area = _flatten(snap)
            early[lab] = _band_table(T, lat, lon, wet, zc, z20_fn,
                                     a.lat_halfwidth, bins, a.depth_max, area)
            print(f"[early {lab}] {snap}")
        T, lat, lon, wet, zc, area = _nemo_columns(a.nemo_gridt,
                                                   a.nemo_rec_early)
        early["NEMO"] = _band_table(T, lat, lon, wet, zc, z20_fn,
                                    a.lat_halfwidth, bins, a.depth_max, area)
        print(f"[early NEMO] {a.nemo_gridt} record {a.nemo_rec_early}\n")
        print(f"=== CHANGE from the early state to the late one "
              f"(NEMO records {a.nemo_rec_early} -> {a.nemo_rec}) ===")
        print("The cold tongue is where a still-growing bias lives, so this "
              "table -- not the day-30 value -- is what an arm is judged on.\n")
        for field, idx, unit in (("dSST", 2, "C"), ("dZ20", 3, "m"),
                                 ("dsharpness", 4, "K/m")):
            print(f"--- {field} [{unit}] ---")
            print(f"{'lon':>12}" + "".join(f"{o:>14}" for o in order))
            for j, (lo, hi) in enumerate(bins):
                cells = "".join(
                    f"{tables[o][j][idx] - early[o][j][idx]:>14.4g}"
                    for o in order)
                print(f"{f'{lo:.0f}-{hi:.0f}E':>12}{cells}")
            print()

    print("counts per bin (columns entering each mean):")
    print(f"{'lon':>12}" + "".join(f"{o:>14}" for o in order))
    for j, (lo, hi) in enumerate(bins):
        print(f"{f'{lo:.0f}-{hi:.0f}E':>12}"
              + "".join(f"{tables[o][j][5]:>14d}" for o in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _flatten_S(snapshot: Path, nlev: int) -> np.ndarray:
    """Salinity as a (n, nlev) cloud, same layout rule as ``_flatten``."""
    S = np.asarray(np.load(snapshot)["S"], dtype=np.float64)
    return S.reshape(-1, nlev) if S.ndim == 3 else S


# Isopycnal discriminator (GLM 2026-09-05): a layer that warms on DEPTH
# surfaces but not on DENSITY surfaces has been heaved (missing ascent /
# downwelling moves the whole stack); a layer that warms on density surfaces
# has been mixed diapycnally.  Potential density from the shared NEMO Roquet
# EOS-80 polynomial at p=0 (inputs are potential T and practical S).
_SIGMA_LEVELS = np.arange(22.0, 26.01, 0.5)


def _sigma0(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp
    from legoesm.ocean.eos import nemo_roquet_eos
    ok = np.isfinite(T) & np.isfinite(S)
    Tf = np.where(ok, T, 10.0)
    Sf = np.where(ok, S, 35.0)
    rho = np.asarray(nemo_roquet_eos(jnp.asarray(Tf), jnp.asarray(Sf),
                                     jnp.zeros_like(jnp.asarray(Tf))))
    return np.where(ok, rho - 1000.0, np.nan)


def _T_on_sigma(T: np.ndarray, sig: np.ndarray, levels=_SIGMA_LEVELS,
                zc: np.ndarray | None = None) -> np.ndarray:
    """T interpolated onto sigma0 surfaces, per column (n, len(levels)); NaN
    where the column never reaches the surface (or is not monotone there).
    With ``zc`` given, the DEPTH of each surface is returned instead: its
    change over time is the time-integrated vertical displacement (heave),
    which a single-snapshot w cannot give at the equator (grid-scale
    alternating w rows of +-80e-6 m/s in the day-30 tripole snapshot)."""
    out = np.full((T.shape[0], len(levels)), np.nan)
    for i in range(T.shape[0]):
        s, t = sig[i], T[i]
        ok = np.isfinite(s) & np.isfinite(t)
        if ok.sum() < 2:
            continue
        s, t = s[ok], t[ok]
        y = zc[ok] if zc is not None else t
        s = np.maximum.accumulate(s)          # enforce monotone (unstable bits)
        for j, lev in enumerate(levels):
            if s[0] <= lev <= s[-1]:
                out[i, j] = np.interp(lev, s, y)
    return out


def _sigma_band(T, S, lat, lon, wet, area, halfwidth, lo, hi, zc=None):
    """Restrict to the equatorial band / longitude window BEFORE the
    per-column interpolation (the node clouds have >1e5 columns).  Returns
    T(sigma) and, when ``zc`` is given, z(sigma) stacked along a new last
    axis: (n, n_levels, 2)."""
    m = wet & (np.abs(lat) <= halfwidth) & (lon >= lo) & (lon < hi)
    sig = _sigma0(T[m], S[m])
    Ts = _T_on_sigma(T[m], sig)
    if zc is not None:
        Ts = np.stack([Ts, _T_on_sigma(T[m], sig, zc=zc)], axis=-1)
    return Ts, lat[m], lon[m], wet[m], (None if area is None else area[m])


def _band_sigma_table(Tsig, lat, lon, wet, halfwidth, bins, area=None):
    w_all = area if area is not None else np.cos(np.deg2rad(lat))
    sel = wet & (np.abs(lat) <= halfwidth)
    rows = []
    for lo, hi in bins:
        m = sel & (lon >= lo) & (lon < hi)
        v = Tsig[m]
        w = np.asarray(w_all)[m].reshape((-1,) + (1,) * (v.ndim - 1)) * np.isfinite(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(np.nansum(v * w, axis=0) / w.sum(axis=0))
    return np.array(rows)


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


def _nemo_S(gridt: Path, rec: int, nlev: int) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(gridt, decode_times=False)
    svar = next((v for v in ("so", "vosaline", "soce") if v in ds), None)
    if svar is None:
        raise SystemExit(f"{gridt}: no 3-D salinity for the isopycnal block")
    S = np.asarray(ds[svar].values, dtype=np.float64)
    if S.ndim == 4:
        S = S[rec]
    S = np.where((np.abs(S) > 1e10) | (S == 0.0), np.nan, S)
    return np.moveaxis(S, 0, -1).reshape(-1, nlev)


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
            rows.append((lo, hi, np.nan, np.nan, np.nan, 0,
                         *(np.nan for _ in _LAYERS)))
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
                     int(m.sum()),
                     *(wmean(_layer_mean_T(Tm, zc, z0, z1))
                       for z0, z1 in _LAYERS)))
    return rows


# Layer-mean temperature [C] over fixed depth slabs: the change of each slab is
# the heat-content tendency per unit area up to rho*c_p*thickness, so a cold
# tongue that cools at the surface while the slab below WARMS is vertical
# redistribution (mixing/upwelling onto the surface), while every slab cooling
# together is net heat loss or lateral export.  This is the discriminator the
# undercurrent result left open: FESOM carries NEMO's undercurrent and still
# warms like the others (2026-09-05).
_LAYERS = ((0.0, 50.0), (50.0, 150.0), (150.0, 300.0))

# Near-surface lens probe (2026-09-06): the day-30 slab table showed a box
# whose SST was 2.7 C warmer than NEMO while its 0-50 m mean was colder --
# a thin warm lens over cold water, which the layer means cannot see.  The
# box-mean T is interpolated from each model's own levels to common depths.
_SURFACE_DEPTHS = np.array([0.5, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0])


def _surface_profile(T, lat, lon, wet, zc, area, halfwidth, lo, hi,
                     depths=_SURFACE_DEPTHS) -> np.ndarray:
    """Area-weighted box-mean T at ``depths`` (linear in z; first level is
    held constant above its own centre).  NaN where a depth is below the
    box's shallowest column."""
    box = wet & (np.abs(lat) <= halfwidth) & (lon >= lo) & (lon < hi)
    if not box.any():
        return np.full(depths.size, np.nan)
    w = (area if area is not None else np.cos(np.radians(lat)))[box]
    Tb = T[box]
    out = np.empty(depths.size)
    for k, d in enumerate(depths):
        col = np.array([np.interp(d, zc, row) for row in Tb])
        ok = np.isfinite(col) & (d <= np.nanmax(zc))
        out[k] = np.sum(w[ok] * col[ok]) / np.sum(w[ok]) if ok.any() else np.nan
    return out


def _layer_mean_T(T: np.ndarray, zc: np.ndarray, z0: float, z1: float) -> np.ndarray:
    dz = np.gradient(zc)
    k = (zc >= z0) & (zc < z1)
    w = dz[k][None, :] * np.isfinite(T[:, k])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(T[:, k] * dz[k][None, :], axis=1) / w.sum(axis=1)


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
    ap.add_argument("--isopycnal", action="store_true",
                    help="also tabulate T on sigma0 surfaces (heave vs "
                         "diapycnal-mixing discriminator)")
    ap.add_argument("--isopycnal-lon-lo", type=float, default=200.0)
    ap.add_argument("--isopycnal-lon-hi", type=float, default=260.0)
    ap.add_argument("--surface-profile", action="store_true",
                    help="box-mean T at common depths over the top 50 m "
                         "(the warm-lens probe), late and early states")
    ap.add_argument("--surface-lon-lo", type=float, default=220.0)
    ap.add_argument("--surface-lon-hi", type=float, default=240.0)
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
                             ("sharpness", 4, "K/m"),
                             ("T 0-50m", 6, "C"), ("T 50-150m", 7, "C"),
                             ("T 150-300m", 8, "C")):
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
                                 ("dsharpness", 4, "K/m"),
                                 ("dT 0-50m", 6, "C"), ("dT 50-150m", 7, "C"),
                                 ("dT 150-300m", 8, "C")):
            print(f"--- {field} [{unit}] ---")
            print(f"{'lon':>12}" + "".join(f"{o:>14}" for o in order))
            for j, (lo, hi) in enumerate(bins):
                cells = "".join(
                    f"{tables[o][j][idx] - early[o][j][idx]:>14.4g}"
                    for o in order)
                print(f"{f'{lo:.0f}-{hi:.0f}E':>12}{cells}")
            print()

    if a.surface_profile:
        lo, hi = a.surface_lon_lo, a.surface_lon_hi
        print(f"=== Near-surface T profile [C], |lat| <= {a.lat_halfwidth}, "
              f"{lo:.0f}-{hi:.0f}E, box mean at common depths (each model's "
              f"own levels, linear in z; NEMO 5-day mean) ===")
        print("SST above the 0-50 m mean by more than NEMO = a warm lens the "
              "mixing does not erode.\n")
        stages = [("late", a.snapshot, a.nemo_rec)]
        if a.snapshot_early:
            stages.append(("early", a.snapshot_early, a.nemo_rec_early))
        for stage, snaps, rec in stages:
            prof = {}
            for snap, lab in zip(snaps, a.label):
                T, lat, lon, wet, zc, area = _flatten(snap)
                prof[lab] = _surface_profile(T, lat, lon, wet, zc, area,
                                             a.lat_halfwidth, lo, hi)
            T, lat, lon, wet, zc, area = _nemo_columns(a.nemo_gridt, rec)
            prof["NEMO"] = _surface_profile(T, lat, lon, wet, zc, area,
                                            a.lat_halfwidth, lo, hi)
            print(f"--- {stage} (NEMO record {rec}) ---")
            print(f"{'depth m':>10}" + "".join(f"{o:>12}" for o in order))
            for k, d in enumerate(_SURFACE_DEPTHS):
                print(f"{d:>10.1f}" + "".join(
                    f"{prof[o][k]:>12.2f}" for o in order))
            print(f"{'T(0.5)-T(20)':>10}" + "".join(
                f"{prof[o][0] - prof[o][5]:>12.2f}" for o in order))
            print()

    if a.isopycnal:
        print("=== T on sigma0 surfaces [C] (heave keeps it, diapycnal mixing "
              "changes it); NEMO Roquet EOS-80 at p=0 ===")
        sig_state = {}
        for snap, lab in zip(a.snapshot, a.label):
            T, lat, lon, wet, zc, area = _flatten(snap)
            Ts, lat, lon, wet, area = _sigma_band(
                T, _flatten_S(snap, zc.size), lat, lon, wet, area,
                a.lat_halfwidth, a.isopycnal_lon_lo, a.isopycnal_lon_hi, zc)
            sig_state[lab] = (_band_sigma_table(Ts, lat, lon, wet,
                                                a.lat_halfwidth, bins, area),)
        T, lat, lon, wet, zc, area = _nemo_columns(a.nemo_gridt, a.nemo_rec)
        Ts, lat, lon, wet, area = _sigma_band(
            T, _nemo_S(a.nemo_gridt, a.nemo_rec, zc.size), lat, lon, wet, area,
            a.lat_halfwidth, a.isopycnal_lon_lo, a.isopycnal_lon_hi, zc)
        sig_state["NEMO"] = (_band_sigma_table(Ts, lat, lon, wet,
                                               a.lat_halfwidth, bins, area),)
        if a.snapshot_early:
            sig_early = {}
            for snap, lab in zip(a.snapshot_early, a.label):
                T, lat, lon, wet, zc, area = _flatten(snap)
                Ts, lat, lon, wet, area = _sigma_band(
                    T, _flatten_S(snap, zc.size), lat, lon, wet, area,
                    a.lat_halfwidth, a.isopycnal_lon_lo, a.isopycnal_lon_hi, zc)
                sig_early[lab] = _band_sigma_table(Ts, lat, lon, wet,
                                                   a.lat_halfwidth, bins, area)
            T, lat, lon, wet, zc, area = _nemo_columns(a.nemo_gridt,
                                                       a.nemo_rec_early)
            Ts, lat, lon, wet, area = _sigma_band(
                T, _nemo_S(a.nemo_gridt, a.nemo_rec_early, zc.size), lat, lon,
                wet, area, a.lat_halfwidth, a.isopycnal_lon_lo, a.isopycnal_lon_hi, zc)
            sig_early["NEMO"] = _band_sigma_table(Ts, lat, lon, wet,
                                                  a.lat_halfwidth, bins, area)
        for j, (lo, hi) in enumerate(bins):
            if not (a.isopycnal_lon_lo <= lo < a.isopycnal_lon_hi):
                continue
            for q, (qname, unit) in enumerate((("T(sigma0)", "C"),
                                               ("z(sigma0)", "m"))):
                print(f"--- bin {lo:.0f}-{hi:.0f}E: {qname} [{unit}] at the "
                      f"late time, then CHANGE early->late"
                      + (" (dz < 0 = surface LIFTED = ascent)" if q else "")
                      + " ---")
                print(f"{'sigma0':>8}" + "".join(f"{o:>12}" for o in order)
                      + ("   |" + "".join(f"{'d'+o:>12}" for o in order)
                         if a.snapshot_early else ""))
                for k, lev in enumerate(_SIGMA_LEVELS):
                    cells = "".join(f"{sig_state[o][0][j, k, q]:>12.3f}"
                                    for o in order)
                    if a.snapshot_early:
                        cells += "   |" + "".join(
                            f"{sig_state[o][0][j, k, q] - sig_early[o][j, k, q]:>12.3f}"
                            for o in order)
                    print(f"{lev:>8.1f}{cells}")
                print()

    print("counts per bin (columns entering each mean):")
    print(f"{'lon':>12}" + "".join(f"{o:>14}" for o in order))
    for j, (lo, hi) in enumerate(bins):
        print(f"{f'{lo:.0f}-{hi:.0f}E':>12}"
              + "".join(f"{tables[o][j][5]:>14d}" for o in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

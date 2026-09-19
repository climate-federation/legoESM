"""Sea-level head across the Bering, ours vs NEMO's, on the same sector.

WHY THIS AND NOT ANOTHER FEATURE HUNT. NEMO drives +0.1681 Sv INTO the Arctic
through the 66N Pacific sector over its first thirty days and we drive 0.5303
Sv OUT, from an identical resting start with identical temperature, salinity,
forcing and bathymetry. Five candidate causes have been eliminated by reading
-- superseded zonal metrics, a spin-up mismatch, the strait geometry, absent
ice dynamics, absent ice-ocean stress -- and eliminating plausible features one
at a time is ranking guesses.

The leading-order control on a strait throughflow is the sea-level difference
across it. Measuring that BISECTS the problem instead of testing one term:

  heads differ in sign or size  -> the FORCING of the transport is wrong, and
                                   the cause is upstream of the strait in the
                                   basin-scale mass distribution;
  heads agree but transports do -> the RESPONSE to the same forcing is wrong,
                                   and the cause is local: channel friction,
                                   resolution, or the barotropic treatment.

Either answer names a class. Neither requires a NEMO rerun: NEMO's momentum
trends are not in its file_def, so a term-by-term oracle momentum budget does
not exist for this run, but sea surface height is written on both sides.

SECTOR. The longitude band is taken from the transport's OWN definition,
diagnostics_sections.py ``bering_pacific`` = (-180,-140) or (155,180), so the
head and the transport refer to the same water rather than to two boxes that
happen to both be called Bering. The latitude bands sit either side of the 66N
section the transport crosses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# From diagnostics_sections.py: the bering_pacific sector this transport uses.
BERING_LON_BANDS = ((-180.0, -140.0), (155.0, 180.0))
PACIFIC_LAT = (60.0, 65.5)     # south of the 66N section
ARCTIC_LAT = (66.5, 72.0)      # north of it


def _sq(a):
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    return a


def in_sector(lat, lon):
    lon = ((np.asarray(lon) + 180.0) % 360.0) - 180.0
    band = np.zeros(lon.shape, dtype=bool)
    for lo, hi in BERING_LON_BANDS:
        band |= (lon >= lo) & (lon <= hi)
    return band, lat


def box_mean(field, wet, area, lat, lon, lat_band):
    """Area-weighted mean over wet cells in the sector and latitude band."""
    band, _ = in_sector(lat, lon)
    m = (band & wet & np.isfinite(field)
         & (lat >= lat_band[0]) & (lat <= lat_band[1]))
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    w = area[m]
    return float(np.sum(field[m] * w) / np.sum(w)), n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True,
                   help="our snapshot .npz carrying eta, land_mask, cell_area")
    p.add_argument("--mesh", required=True, help="NEMO mesh_mask / domain_cfg")
    p.add_argument("--nemo-gridt", required=True, help="NEMO 5-day grid_T")
    p.add_argument("--time-idx", type=int, default=5,
                   help="NEMO record; 5 = days 25-30, the transport's window")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    import xarray as xr

    snap = dict(np.load(a.snapshot))
    eta = _sq(snap["eta"])
    wet = _sq(snap["land_mask"]) > 0.5
    area = _sq(snap["cell_area"])

    dm = xr.open_dataset(a.mesh, decode_times=False)
    lat = _sq(dm["gphit"].values)
    lon = _sq(dm["glamt"].values)

    # CONTROL: our snapshot is NEMO's full (jpj, jpi) grid, so it should align
    # with the mesh cell for cell with NO window. If it does not, every box
    # below is over the wrong water and the numbers are meaningless -- this is
    # the failure that has bitten this comparison before, so it is asserted
    # rather than assumed.
    if lat.shape != eta.shape:
        raise SystemExit(
            f"FATAL: mesh {lat.shape} does not match snapshot {eta.shape}; "
            "this probe assumes the snapshot carries NEMO's full grid. "
            "Refusing to guess a window.")
    tmask = _sq(dm["tmaskutil"].values if "tmaskutil" in dm
                else dm["tmask"].values) > 0.5
    agree = float((tmask == wet).mean())
    if agree < 0.99:
        raise SystemExit(
            f"FATAL: our land mask agrees with the mesh on only {agree:.4f} "
            "of cells; the grids are not aligned.")

    ours_pac, n_op = box_mean(eta, wet, area, lat, lon, PACIFIC_LAT)
    ours_arc, n_oa = box_mean(eta, wet, area, lat, lon, ARCTIC_LAT)

    dt = xr.open_dataset(a.nemo_gridt, decode_times=False)
    ssh_name = next((v for v in ("sossheig", "zos", "ssh", "sshn")
                     if v in dt.variables), None)
    if ssh_name is None:
        raise SystemExit(
            f"FATAL: {a.nemo_gridt} has no sea-surface-height variable; "
            f"looked for sossheig/zos/ssh/sshn among {list(dt.variables)[:20]}")
    ssh = _sq(dt[ssh_name].isel(time_counter=a.time_idx).values)
    nlat = _sq(dt["nav_lat"].values)
    nlon = _sq(dt["nav_lon"].values)
    # NEMO writes fill values on land; a wet test on SSH alone is unreliable,
    # so wetness comes from finiteness AND a plausible range.
    nwet = np.isfinite(ssh) & (np.abs(ssh) < 20.0)
    narea = np.ones_like(ssh)   # NEMO's own e1t*e2t is not in this file;
    # an unweighted mean over a small box is a different reduction from ours,
    # so ours is ALSO reported unweighted below as the like-for-like number.

    nemo_pac, n_np = box_mean(ssh, nwet, narea, nlat, nlon, PACIFIC_LAT)
    nemo_arc, n_na = box_mean(ssh, nwet, narea, nlat, nlon, ARCTIC_LAT)

    # LIKE FOR LIKE: unweighted on both sides, since NEMO's cell areas are not
    # in this file. The area-weighted pair is reported beside it so a reader
    # can see the reduction does not carry the result.
    ones = np.ones_like(eta)
    ours_pac_u, _ = box_mean(eta, wet, ones, lat, lon, PACIFIC_LAT)
    ours_arc_u, _ = box_mean(eta, wet, ones, lat, lon, ARCTIC_LAT)

    res = {
        "ours_pacific_m": ours_pac, "ours_arctic_m": ours_arc,
        "ours_head_m": ours_pac - ours_arc,
        "ours_pacific_unweighted_m": ours_pac_u,
        "ours_arctic_unweighted_m": ours_arc_u,
        "ours_head_unweighted_m": ours_pac_u - ours_arc_u,
        "nemo_pacific_m": nemo_pac, "nemo_arctic_m": nemo_arc,
        "nemo_head_m": nemo_pac - nemo_arc,
        "n_ours_pacific": n_op, "n_ours_arctic": n_oa,
        "n_nemo_pacific": n_np, "n_nemo_arctic": n_na,
        "ssh_variable": ssh_name, "time_idx": a.time_idx,
        "mask_agreement": agree,
    }

    print(f"[control] land-mask agreement with mesh: {agree:.4f}")
    print(f"[control] cell counts  ours {n_op}/{n_oa}   NEMO {n_np}/{n_na}")
    print(f"[control] NEMO ssh variable: {ssh_name}, record {a.time_idx}")
    print("")
    print("Sea level, Pacific side minus Arctic side, bering_pacific sector.")
    print("POSITIVE head = Pacific stands higher = drives flow INTO the Arctic.")
    print(f"  ours  pacific {ours_pac:+.4f}  arctic {ours_arc:+.4f}  "
          f"HEAD {ours_pac - ours_arc:+.4f} m")
    print(f"  ours  (unweighted, like-for-like)            "
          f"HEAD {ours_pac_u - ours_arc_u:+.4f} m")
    print(f"  NEMO  pacific {nemo_pac:+.4f}  arctic {nemo_arc:+.4f}  "
          f"HEAD {nemo_pac - nemo_arc:+.4f} m")
    print("")
    print("READ: heads differing in SIGN => the transport's FORCING is wrong "
          "and the cause is upstream. Heads AGREEING => the RESPONSE is wrong "
          "and the cause is local to the channel.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

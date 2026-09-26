"""Hemispheric / Arctic sea-ice AREA and VOLUME at GATEWAY day 90 vs NEMO SI3.

GLM review 2026-09-04: capping the top cell at T_freeze (the SI3 lead-heat
budget, ``--ice-lead-freeze-source nemo_qlead``) converts ANY excess surface
heat loss into ice, so SST alone cannot distinguish "the missing lead budget
was the whole defect" from "a second heat sink is now hidden as ice growth".
Ice VOLUME can: a hidden sink shows up as ice far in excess of NEMO's.

NEMO side is the stitched ``ORCA1_00002160_restart_ice_*`` tiles of the
GATEWAY run (kt 2160 at rn_Dt 3600 = day 90 = 1 April): ``a_i`` [0-1] and
``v_i`` = ice volume per grid-cell area [m].  Thickness is DERIVED (v_i/a_i)
and reported only over each field's own conc>0.15 cells (survivorship-biased,
NOT comparable between runs); area and volume are the honest metrics.

Bookkeeping line: for the pre-fix arm, the supercooling deficit of the top
cell, sum(rho c_p dz (T_f - T)) over wet cells with T < T_f, expressed as the
equivalent ice thickness Q/(rho_i L_f) -- what the lead budget WOULD have
grown from that heat.  Compare with (fixed arm volume - pre-fix arm volume).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_nemo_restart import rebuild  # noqa: E402

from legoesm import constants  # noqa: E402

REGIONS = (("NH total", 0.0, 90.0), ("Arctic >65N", 65.0, 90.0),
           ("SH total", -90.0, 0.0))
COV = 0.15


def _area_from(path: str, shape) -> np.ndarray:
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        e1 = np.squeeze(np.asarray(ds["e1t"][:], dtype=np.float64))
        e2 = np.squeeze(np.asarray(ds["e2t"][:], dtype=np.float64))
    a = e1 * e2
    if a.shape != shape:
        raise ValueError(f"area {a.shape} != field {shape} ({path})")
    return a


def _stats(a, v, lat, wet, area):
    rows = []
    for name, la0, la1 in REGIONS:
        m = wet & (lat >= la0) & (lat <= la1)
        cov = m & (a > COV)
        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(a > 1e-6, v / np.maximum(a, 1e-6), 0.0)
        rows.append((name,
                     float(np.nansum((a * area)[m])) / 1e12,
                     float(np.nansum((v * area)[m])) / 1e9,
                     float(np.nanmean(h[cov])) if cov.any() else np.nan))
    return rows


def supercool_ice_equiv(snap, lat, wet, area, dz, la0=0.0, la1=90.0):
    """Ice volume [1e3 km3] the top-cell freezing deficit is worth."""
    T0 = np.asarray(snap["T"])[..., 0]
    T_f = constants.T_freeze_ocean - constants.T_freeze  # -1.8 C
    deficit = np.clip(T_f - T0, 0.0, None)
    m = wet & (lat >= la0) & (lat <= la1)
    Q = float(np.nansum((constants.rho_ocean * constants.c_pw * dz * deficit * area)[m]))
    return Q / (constants.rho_ice * constants.L_f) / 1e9, int((deficit[m] > 0).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nemo-ice-tiles", required=True)
    p.add_argument("--nemo-domain-cfg", required=True)
    p.add_argument("--mesh-mask", required=True, help="eORCA1 mesh with e1t/e2t (332,362)")
    p.add_argument("--nemo-gridt", required=True, help="for nav_lat on the (331,360) NEMO grid")
    p.add_argument("--arm", action="append", required=True,
                   help="label=path/to/snapshot_day0090.npz (tripole grid)")
    a = p.parse_args()

    import netCDF4
    ice = rebuild(a.nemo_ice_tiles, ["a_i", "v_i"])
    with netCDF4.Dataset(a.nemo_gridt) as ds:
        lat_n = np.asarray(ds["nav_lat"][:], dtype=np.float64)
    a_n = np.nansum(ice["a_i"], axis=0) if ice["a_i"].ndim == 3 else ice["a_i"]
    v_n = np.nansum(ice["v_i"], axis=0) if ice["v_i"].ndim == 3 else ice["v_i"]
    wet_n = np.isfinite(ice["a_i"]).all(axis=0) if ice["a_i"].ndim == 3 else np.isfinite(a_n)
    area_n = _area_from(a.nemo_domain_cfg, a_n.shape)

    print(f"{'region':13s}{'source':16s}{'area[1e6km2]':>13}{'vol[1e3km3]':>12}{'h_cov[m]':>9}")
    nemo_rows = _stats(a_n, v_n, lat_n, wet_n, area_n)
    arms = []
    for spec in a.arm:
        label, path = spec.split("=", 1)
        z = np.load(path)
        lat = np.asarray(z["lat_T"], dtype=np.float64)
        if np.abs(lat).max() < 7:
            lat = np.degrees(lat)
        wet = np.asarray(z["land_mask"]) > 0.5
        area = _area_from(a.mesh_mask, lat.shape)
        with netCDF4.Dataset(a.mesh_mask) as ds:
            dz0 = np.squeeze(np.asarray(ds["e3t_0"][:], dtype=np.float64))[0]
        if dz0.shape != lat.shape:
            raise ValueError(f"e3t_0 top level {dz0.shape} != {lat.shape}")
        ai = np.asarray(z["ice_concentration"]); hi = np.asarray(z["ice_thickness"])
        arms.append((label, _stats(ai, ai * hi, lat, wet, area),
                     supercool_ice_equiv(z, lat, wet, area, dz0, 0.0, 90.0)))
    for i, (name, *_) in enumerate(REGIONS):
        print(f"{name:13s}{'NEMO d90':16s}{nemo_rows[i][1]:13.3f}{nemo_rows[i][2]:12.1f}{nemo_rows[i][3]:9.2f}")
        for label, rows, _ in arms:
            print(f"{'':13s}{label:16s}{rows[i][1]:13.3f}{rows[i][2]:12.1f}{rows[i][3]:9.2f}")
        print()
    for label, _, (equiv, n) in arms:
        print(f"[bookkeeping] {label}: NH top-cell freezing deficit = {equiv:.1f} 1e3 km3 "
              f"of ice-equivalent over {n} supercooled cells")
    print("h_cov over each field's OWN conc>0.15 cells: NOT comparable between runs.")


if __name__ == "__main__":
    main()

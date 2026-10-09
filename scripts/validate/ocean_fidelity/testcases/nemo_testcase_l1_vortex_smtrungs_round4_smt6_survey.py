#!/usr/bin/env python3
"""SMT-RUNGS round 4: SMT-6 (BBL + geothermal) survey measurements.

Two numbers the SMT-6 deck proposal rests on, both read from files NEMO wrote:

``bbl``         For every restart of the SMT-5 NEMO 100-day record, count the
                U/V faces whose bottom-level T/S pass trabbl.f90's diffusive
                gate (grad(rho).grad(H) < 0, with NEMO's mgrh = sign of the
                bottom-LEVEL depth difference).  S-EOS with b0 = 0, so
                zgdrho*mgrh > 0 is the open gate (alpha > 0).
``geothermal``  ORCA2 rung 2's geothermal_heating.nc over wet columns of its
                mesh_mask: min/max/mean/median and the area-weighted mean, to
                set against rn_geoflx_cst = 86.4e-3.

``--plant`` makes the shelf-side bottom T of one sloped face colder than the
deep side in the FIRST restart; the open-face count must become nonzero and the
run exits nonzero.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

SMT5_100D = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "smtrungs_rounds/round2/oracle_vortex_smt5/day100")
ORCA2_R2 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                "orca2_hierarchy/rung2/record")


def bottom_index(wet_levels: np.ndarray) -> np.ndarray:
    """1-based bottom T-level, NEMO mbkt (= 1 over land)."""
    return np.maximum(wet_levels.astype(int), 1)


def face_slopes(mb: np.ndarray, gdept_1d: np.ndarray):
    """trabbl.f90:584-585 mgrhu/mgrhv from bottom-LEVEL depths (not partial)."""
    mgu = np.sign(gdept_1d[mb[:, 1:] - 1] - gdept_1d[mb[:, :-1] - 1]).astype(int)
    mgv = np.sign(gdept_1d[mb[1:, :] - 1] - gdept_1d[mb[:-1, :] - 1]).astype(int)
    return mgu, mgv


def open_faces(Tb: np.ndarray, mgu, mgv, uw, vw):
    """Faces where zgdrho*mgrh > 0 (alpha>0, no salinity effect): gate open."""
    du = (Tb[:, 1:] - Tb[:, :-1]) * mgu
    dv = (Tb[1:, :] - Tb[:-1, :]) * mgv
    return int(((du > 0) & uw & (mgu != 0)).sum()), \
        int(((dv > 0) & vw & (mgv != 0)).sum())


def bbl_gate_history(plant: bool = False) -> dict:
    import netCDF4
    m = netCDF4.Dataset(SMT5_100D / "mesh_mask.nc")
    tm = np.squeeze(m["tmask"][:])
    g1 = np.squeeze(m["gdept_1d"][:])
    wet = tm.sum(0) > 0
    mb = bottom_index(tm.sum(0))
    mgu, mgv = face_slopes(mb, g1)
    uw = wet[:, 1:] & wet[:, :-1]
    vw = wet[1:, :] & wet[:-1, :]
    J, I = np.indices(mb.shape)
    sloped_u = int(((mgu != 0) & uw).sum())
    sloped_v = int(((mgv != 0) & vw).sum())
    files = sorted(glob.glob(str(SMT5_100D / "*_restart.nc")))
    if not files:
        raise SystemExit("no restart files: refuse to report zero open faces")
    rows = []
    for n, p in enumerate(files):
        step = int(re.search(r"_(\d{8})_restart", p).group(1))
        T = np.squeeze(netCDF4.Dataset(p)["tn"][:]).astype(float)
        if not np.isfinite(T[tm > 0]).all():
            raise SystemExit(f"non-finite T in {p}")
        Tb = T[mb - 1, J, I]
        if plant and n == 0:
            j, i = np.argwhere((mgu != 0) & uw)[0]
            shelf, deep = (i, i + 1) if mgu[j, i] > 0 else (i + 1, i)
            Tb = Tb.copy()
            Tb[j, shelf] = Tb[j, deep] - 1.0
        nu, nv = open_faces(Tb, mgu, mgv, uw, vw)
        rows.append({"step": step, "open_u": nu, "open_v": nv})
    return {"n_restarts": len(rows), "sloped_u_faces": sloped_u,
            "sloped_v_faces": sloped_v,
            "max_open_u": max(r["open_u"] for r in rows),
            "max_open_v": max(r["open_v"] for r in rows),
            "rows": rows[:: max(1, len(rows) // 10)]}


def geothermal_stats() -> dict:
    import netCDF4
    h = np.ma.filled(
        netCDF4.Dataset(ORCA2_R2 / "geothermal_heating.nc")["heatflow"][0],
        np.nan).astype(float)

    def cat(v):
        return np.concatenate([np.squeeze(netCDF4.Dataset(
            ORCA2_R2 / f"mesh_mask_000{i}.nc")[v][:]) for i in (0, 1)], axis=-1)

    wet = cat("tmask")[0] > 0
    area = cat("e1t") * cat("e2t")
    if h.shape != wet.shape or np.isnan(h[wet]).any():
        raise SystemExit("geothermal file / mesh shape mismatch or NaN on wet")
    x, w = h[wet], area[wet]
    return {"units": "mW/m2", "n_wet_columns": int(wet.sum()),
            "min": float(x.min()), "max": float(x.max()),
            "mean_unweighted": float(x.mean()), "median": float(np.median(x)),
            "mean_area_weighted": float((x * w).sum() / w.sum()),
            "n_zero_on_wet": int((x == 0).sum()),
            "rn_geoflx_cst_mW_m2": 86.4}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--plant", action="store_true")
    a = ap.parse_args(argv)
    rep = {"bbl": bbl_gate_history(plant=a.plant), "geothermal": geothermal_stats()}
    txt = json.dumps(rep, indent=2, sort_keys=True) + "\n"
    if a.output:
        a.output.write_text(txt)
    print(txt, end="")
    opened = rep["bbl"]["max_open_u"] + rep["bbl"]["max_open_v"]
    if a.plant:
        return 1 if opened > 0 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""TKE and mixing length, ours vs NEMO, instantaneous at the same clock hour.

At hour 5 of the hourly NEMO rerun the oracle's viscosity is 1.4-2x ours at
7-15 m and 2-4x at 15-29 m with the same N2, from the same rest state and the
same stress. avm = rn_ediff * l_m * sqrt(en), so that factor is TKE, length,
or both. NEMO's restart files (written every 6 h with nn_stock=6) hold the
instantaneous ``en``, ``avm_k``, ``avt_k`` and ``dissl = sqrt(en)/l_eps``;
our snapshot holds ``tke`` and ``K_M_diag`` on the same interior interfaces.
This reassembles the 32 restart tiles by their DOMAIN_position attributes and
prints, per interface: e, K_M, the implied l_m = K_M/(rn_ediff*sqrt(e)) for
both, and NEMO's l_eps. Box means: ours area-weighted, NEMO unweighted (no
cell areas in a restart).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equatorial_shear_ri_vs_nemo import _box_mean  # noqa: E402

RN_EDIFF = 0.1  # zdftke rn_ediff (ORCA1 namelist_ref); avm = rn_ediff * l * sqrt(en)


def reassemble(pattern: str, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no restart tiles match {pattern}")
    d0 = nc.Dataset(files[0])
    nx, ny = (int(v) for v in d0.getncattr("DOMAIN_size_global"))
    nz = d0.variables[names[0]].shape[1]
    out = {n: np.full((nz, ny, nx), np.nan) for n in names}
    out["nav_lat"] = np.full((ny, nx), np.nan); out["nav_lon"] = np.full((ny, nx), np.nan)
    for f in files:
        d = nc.Dataset(f)
        i0, j0 = (int(v) for v in d.getncattr("DOMAIN_position_first"))
        i1, j1 = (int(v) for v in d.getncattr("DOMAIN_position_last"))
        hs = [int(v) for v in d.getncattr("DOMAIN_halo_size_start")]
        he = [int(v) for v in d.getncattr("DOMAIN_halo_size_end")]
        if any(hs) or any(he):
            raise SystemExit(f"{f}: halo {hs}/{he} not handled")
        sl = (slice(j0 - 1, j1), slice(i0 - 1, i1))
        for n in names:
            out[n][(slice(None),) + sl] = np.ma.filled(np.ma.masked_invalid(d.variables[n][0]), np.nan)
        out["nav_lat"][sl] = d.variables["nav_lat"][:]; out["nav_lon"][sl] = d.variables["nav_lon"][:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restart-glob", required=True, help="e.g. RUN_HOURLY/ORCA1_00000006_restart_oce_*.nc")
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--lon-lo", type=float, default=220.0)
    ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--n-levels", type=int, default=20)
    ap.add_argument("--nemo-t-file", type=Path, default=None,
                    help="hourly grid_T file: prints the box-mean taum of the hour ending at the restart step")
    ap.add_argument("--nemo-rec", type=int, default=None, help="record of --nemo-t-file (hour-1)")
    a = ap.parse_args()
    nl = a.n_levels

    R = reassemble(a.restart_glob, ("en", "avm_k", "avt_k", "dissl"))
    la, lo = R["nav_lat"], R["nav_lon"] % 360.0
    boxN = ((np.abs(la) <= a.lat_halfwidth) & (lo >= a.lon_lo) & (lo < a.lon_hi)).ravel()
    wN = np.ones(boxN.size)

    def nprof(x):  # W level k+1 (Fortran k+2) <-> our interior interface k
        f = x[1:nl + 1].reshape(nl, -1).T
        f = np.where(np.abs(f) < 1e-30, np.nan, f)
        return _box_mean(f, wN, boxN)

    eN, kmN, ktN, dsN = (nprof(R[k]) for k in ("en", "avm_k", "avt_k", "dissl"))
    # surface row: NEMO en(1) is the Dirichlet value rn_ebb*taum/rho0 at z=0
    e_sfc_N = _box_mean(np.where(np.abs(R["en"][0]) < 1e-30, np.nan, R["en"][0]).reshape(1, -1).T, wN, boxN)[0]
    lmN = kmN / (RN_EDIFF * np.sqrt(eN)); leN = np.sqrt(eN) / dsN

    s = np.load(a.snapshot)
    e = np.asarray(s["tke"], float); KM = np.asarray(s["K_M_diag"], float); KH = np.asarray(s["K_H_diag"], float)
    lat, lon = np.asarray(s["lat_T"], float), np.asarray(s["lon_T"], float) % 360.0
    wet = np.asarray(s["land_mask"], float) > 0.5
    zw = np.abs(np.asarray(s["z_interface_ref"], float))
    zw_int = zw if zw.size == e.shape[-1] else zw[1:e.shape[-1] + 1]
    area = np.asarray(s["cell_area"], float).ravel()
    box = (wet & (np.abs(lat) <= a.lat_halfwidth) & (lon >= a.lon_lo) & (lon < a.lon_hi)).ravel()
    if e.shape != KM.shape:
        raise SystemExit(f"tke {e.shape} vs K_M_diag {KM.shape}: staggering differs")
    eo, kmo, kho = (_box_mean(x.reshape(-1, x.shape[-1])[:, :nl], area, box) for x in (e, KM, KH))
    lmo = kmo / (RN_EDIFF * np.sqrt(eo))
    RN_EBB, RHO0 = 67.83, 1026.0  # coeff-ok: namelist rn_ebb, phycst rho0 -> e_sfc = rn_ebb*taum/rho0
    taum_line = ""
    if a.nemo_t_file is not None:
        dT = nc.Dataset(a.nemo_t_file)
        laT = np.asarray(dT.variables["nav_lat"][:], float); loT = np.asarray(dT.variables["nav_lon"][:], float) % 360.0
        bT = ((np.abs(laT) <= a.lat_halfwidth) & (loT >= a.lon_lo) & (loT < a.lon_hi)).ravel()
        tm = np.ma.filled(np.ma.masked_invalid(dT.variables["taum"][a.nemo_rec]), np.nan).ravel()
        taum_N = _box_mean(tm[:, None], np.ones(bT.size), bT)[0]
        taum_line = f"; NEMO hourly-mean taum (rec {a.nemo_rec}) {taum_N:.4f} Pa -> rn_ebb*taum/rho0 = {RN_EBB*taum_N/RHO0:.3e}"
    e0_o = eo[0]
    print(f"[surface] NEMO en(z=0) box mean {e_sfc_N:.3e} (implied taum {e_sfc_N*RHO0/RN_EBB:.4f} Pa){taum_line}; "
          f"ours e at interface 1 ({zw_int[0]:.2f} m) {e0_o:.3e} (= the pinned Dirichlet value under interior_pinned; implied taum {e0_o*RHO0/RN_EBB:.4f} Pa)")
    print(f"box {a.lon_lo}-{a.lon_hi}E |lat|<={a.lat_halfwidth}: ours {int(box.sum())} cols (t = {float(s['time_days'])*24:.1f} h, "
          f"{a.snapshot.name}); NEMO {int(boxN.sum())} cols ({Path(a.restart_glob).name}); interior interfaces; "
          f"l_m = K_M/({RN_EDIFF}*sqrt(e)) both sides, l_eps NEMO = sqrt(en)/dissl")
    print("  k   z_w m |    e ours    e NEMO  ratio |  K_M ours  avm_k NEMO ratio |  K_T ours  avt_k NEMO |  l_m ours  l_m NEMO  l_eps NEMO")
    for k in range(nl):
        print(f"{k+1:3d} {zw_int[k]:7.2f} | {eo[k]:9.2e} {eo[k]*0+eN[k]:9.2e} {eo[k]/eN[k]:6.2f} | "
              f"{kmo[k]:9.2e} {kmN[k]:10.2e} {kmo[k]/kmN[k]:5.2f} | {kho[k]:9.2e} {ktN[k]:10.2e} | "
              f"{lmo[k]:8.2f} {lmN[k]:9.2f} {leN[k]:10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

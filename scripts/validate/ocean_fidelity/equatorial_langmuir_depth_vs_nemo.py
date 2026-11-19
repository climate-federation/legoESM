#!/usr/bin/env python
"""Langmuir-cell depth h_lc (Axell 2002 Eq. 47, zdftke.F90:340-354) on each
model's own column, hour by hour, plus the cumulative PE integral it compares
against 0.5*W_lc^2 = zcsd*taum.

NEMO with Langmuir has a TKE maximum at 9-11 m that NEMO without Langmuir does
not; ours has the same source formula but 3x less TKE at 10 m. The source is
us^3 (rn_lc sin(pi z/h_lc))^3 / h_lc, so at 10 m it is set by h_lc, and h_lc
by the stratification integral. NEMO columns from the restart tiles (tn, sn,
en(1) -> taum); ours from snapshots (T, S, e at interface 1 -> taum under
interior_pinned). Same formula on both states, in numpy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equatorial_shear_ri_vs_nemo import _box_mean, _n2  # noqa: E402
from equatorial_tke_length_vs_nemo_restart import reassemble  # noqa: E402

ZCSD = 0.5 * 0.016 * 0.016 / (1.22 * 1.5e-3)  # coeff-ok: zdftke.F90 zcsd (rho_air, Cd of Axell 2002)
RN_EBB, RHO0 = 67.83, 1026.0                    # coeff-ok: namelist rn_ebb, phycst rho0


def h_lc_from(N2, zw, dz_w, taum):
    """Shallowest interface where cumsum(max(N2,0) z dz) > zcsd*taum; (..., ) in m."""
    pe = np.cumsum(np.maximum(np.nan_to_num(N2), 0.0) * zw * dz_w, axis=-1)
    thr = ZCSD * taum
    exceeded = pe > thr[:, None]
    first = np.argmax(exceeded, axis=-1)
    h = zw[first]
    return np.where(exceeded.any(axis=-1), h, zw[-1]), pe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", nargs="+", required=True,
                    help="label=nemo:<restart glob> or label=ours:<snapshot.npz>")
    ap.add_argument("--lon-lo", type=float, default=220.0)
    ap.add_argument("--lon-hi", type=float, default=240.0)
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--n-levels", type=int, default=30)
    a = ap.parse_args()
    nl = a.n_levels
    probe_z = (5.83, 10.77, 15.22, 21.04, 28.65)
    print(f"box {a.lon_lo}-{a.lon_hi}E |lat|<={a.lat_halfwidth}; h_lc = first interface with cumsum(max(N2,0) z dz) > "
          f"{ZCSD:.4f}*taum; taum from e_sfc/(rn_ebb/rho0). Per-column h_lc: mean, quartiles; PE(z) box mean at "
          f"{probe_z} m vs threshold.")
    for item in a.items:
        label, spec = item.split("=", 1); kind, path = spec.split(":", 1)
        if kind == "nemo":
            R = reassemble(path, ("tn", "sn", "en"))
            la, lo = R["nav_lat"], R["nav_lon"] % 360.0
            T = np.moveaxis(R["tn"], 0, -1); S = np.moveaxis(R["sn"], 0, -1)      # (y, x, z)
            e_sfc = R["en"][0]
            zc = None
        else:
            s = np.load(path)
            T = np.asarray(s["T"], float); S = np.asarray(s["S"], float)
            la, lo = np.asarray(s["lat_T"], float), np.asarray(s["lon_T"], float)
            lo = lo % 360.0
            e_sfc = np.asarray(s["tke"], float)[..., 0]
            zc = np.abs(np.asarray(s["z_center_ref"], float)); zw = np.abs(np.asarray(s["z_interface_ref"], float))
        if zc is None:
            # NEMO 1-D reference ladder from the mesh: read once from our snapshot's grid is NOT the oracle's;
            # use gdept/gdepw_1d from the restart's nav_lev? Restart carries nav_lev = gdept_1d only.
            import netCDF4 as nc, glob
            d0 = nc.Dataset(sorted(glob.glob(path))[0])
            zc = np.asarray(d0.variables["nav_lev"][:], float)
            zw = None
        T = np.where(np.abs(T) < 1e-6, np.nan, T)
        box = ((np.abs(la) <= a.lat_halfwidth) & (lo >= a.lon_lo) & (lo < a.lon_hi) & np.isfinite(T[..., 0])).ravel()
        Tf = T.reshape(-1, T.shape[-1])[box][:, :nl]; Sf = S.reshape(-1, S.shape[-1])[box][:, :nl]
        if zw is None:
            zw_full = np.concatenate([[0.0], 0.5 * (zc[:-1] + zc[1:])])  # NEMO gdepw_1d approx (midpoints)
        else:
            zw_full = zw if zw[0] == 0.0 else np.concatenate([[0.0], zw])
        zc_n = zc[:nl]; zw_int = zw_full[1:nl]          # interior interfaces 1..nl-1
        N2 = _n2(np.nan_to_num(Tf), np.nan_to_num(Sf), zc_n, zw_int)   # (cols, nl-1)
        dz_w = np.diff(zc_n)                               # e3w between centres
        taum = np.asarray(e_sfc, float).ravel()[box] * RHO0 / RN_EBB
        h, pe = h_lc_from(N2, zw_int, dz_w, taum)
        w = np.ones(h.size)
        q = np.percentile(h, (25, 50, 75))
        pe_at = [float(np.nanmean(pe[:, np.argmin(np.abs(zw_int - z))])) for z in probe_z]
        print(f"{label:>14}: taum {taum.mean():.4f} Pa thr {ZCSD*taum.mean():.2e} | h_lc mean {h.mean():6.2f} "
              f"q25/50/75 {q[0]:5.1f}/{q[1]:5.1f}/{q[2]:5.1f} m | PE " + " ".join(f"{v:.2e}" for v in pe_at)
              + f" | N2 box mean 5.8/10.8/15.2 m: " + " ".join(f"{np.nanmean(N2[:, np.argmin(np.abs(zw_int - z))]):.2e}" for z in probe_z[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

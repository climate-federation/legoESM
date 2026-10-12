#!/usr/bin/env python
"""Where does the solar heat land in the equatorial column, ours vs NEMO?

NEMO side (the oracle's own arithmetic, not a re-derivation): RUN_DT150_EN2
writes every step on the 2S-2N / 225.5-255.5E strip the surface ``qsr`` and
the per-level penetrative-solar trend ``ttrd_qsr`` [K/s].  The fraction of the
surface solar absorbed in level k over the window is

    F_N(k) = sum_t ttrd_qsr(t,k) * e3t(t,k) * rho0 * rcp  /  sum_t qsr(t)

with NEMO's rho0 = 1026, rcp = 3991.86795711963 (eosbn2.F90:1898-1899).
CONTROL: sum_k F_N(k) must be 1 (traqsr deposits all of qsr in the column).

Ours: the production kernel (``apply_shortwave_penetration`` with
scheme='rgb_chl', the call in ocean_pe_latlon_cgrid) fed unit solar, the
chlorophyll the production run loads (``load_nemo_chl_monthly`` with
chl_remap='nemo_scrip', time-sampled by ``month_sample(..., 'linear')`` at
each step of the same 2 days), and NEMO's own e3t -- so the comparison
isolates chlorophyll input + optics, not the vertical grid.

Prints the qsr-weighted, area-weighted strip mean of the CUMULATIVE absorbed
fraction above each w-level down to ~60 m, both models, and the difference.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "run"))

NEMO_RHO0 = 1026.0
NEMO_RCP = 3991.86795711963


def _read(ds, name):
    x = np.ma.filled(np.ma.masked_invalid(ds[name][:]), np.nan)
    return np.asarray(x, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo-dir", required=True, type=Path)
    ap.add_argument("--prefix", default="ORCA1_1ts_20000101_20000102_eqs1ts_")
    ap.add_argument("--mesh", default=str(_REPO / "data/grids/eORCA1.2_mesh_mask.nc"))
    ap.add_argument("--dt", type=float, default=150.0)
    ap.add_argument("--zmax", type=float, default=60.0)
    a = ap.parse_args()
    sha = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"[provenance] git {sha} | {' '.join(sys.argv)}")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.ocean.physics.shortwave_penetration import (
        apply_shortwave_penetration, ShortwavePenetrationConfig)
    from run_omip_core2 import load_nemo_chl_monthly, month_sample

    dT = nc.Dataset(a.nemo_dir / f"{a.prefix}T.nc")
    d2 = nc.Dataset(a.nemo_dir / f"{a.prefix}2D.nc")
    lat_s = _read(dT, "nav_lat_eqsT"); lon_s = _read(dT, "nav_lon_eqsT") % 360.0
    trd = _read(dT, "ttrd_qsr")            # (t, k, y, x) K/s
    e3t = _read(dT, "e3t")
    qsr = _read(d2, "qsr")                 # (t, y, x) W/m2
    nt = trd.shape[0]
    print(f"[nemo] steps {nt}, strip {lat_s.min():.2f}..{lat_s.max():.2f}N "
          f"{lon_s.min():.1f}..{lon_s.max():.1f}E, qsr mean {np.nanmean(qsr):.1f} W/m2")

    m = nc.Dataset(a.mesh)
    gphit = np.asarray(m["gphit"][0], dtype=np.float64)
    glamt = np.asarray(m["glamt"][0], dtype=np.float64) % 360.0
    area = np.asarray(m["e1t"][0] * m["e2t"][0], dtype=np.float64)
    # strip cell -> mesh (j, i), exact coordinate match required
    jj = np.empty(lat_s.shape, int); ii = np.empty(lat_s.shape, int)
    for (y, x), la in np.ndenumerate(lat_s):
        dd = np.abs(gphit - la) + np.abs(((glamt - lon_s[y, x]) + 180.0) % 360.0 - 180.0)
        jj[y, x], ii[y, x] = np.unravel_index(np.argmin(dd), dd.shape)
        if dd[jj[y, x], ii[y, x]] > 1e-4:
            raise SystemExit(f"strip cell ({y},{x}) has no exact mesh match ({dd.min():.2e})")
    w_area = area[jj, ii]

    # ---- NEMO absorbed fraction per level, qsr-weighted over the window ----
    wet = np.isfinite(trd) & np.isfinite(e3t) & (e3t > 0)
    heat = np.where(wet, trd * e3t, 0.0).sum(axis=0) * NEMO_RHO0 * NEMO_RCP   # (k,y,x) J/m2/s summed
    qsum = np.nansum(qsr, axis=0)                                              # (y,x)
    F_N = heat / np.maximum(qsum, 1e-30)[None]
    colsum = F_N.sum(axis=0)
    print(f"[control] NEMO column sum of absorbed fraction: min {colsum.min():.6f} "
          f"max {colsum.max():.6f} (must be 1)")
    # Measured 2026-10-12: the column sum is a UNIFORM 0.5 (0.49997-0.50001 on
    # every column) under key_RK3 -- a constant scale on the stored trend, not
    # depth structure (cause not localised in trd_tra/stp_RK3_stg).  Refuse
    # anything non-uniform; normalise by the column sum so the SHAPE compares.
    if colsum.max() - colsum.min() > 1e-3:
        raise SystemExit("NEMO column sum is not uniform -> the instrument is not a pure scale; refusing")
    F_N = F_N / colsum[None]

    # ---- ours: production kernel, unit solar, production chlorophyll ----
    chl_clim = load_nemo_chl_monthly(None, "tripole", gphit, glamt, chl_remap="nemo_scrip")
    e3_mean = np.nanmean(np.where(wet, e3t, np.nan), axis=0)                   # (k,y,x)
    dz = jnp.asarray(np.moveaxis(np.nan_to_num(e3_mean), 0, -1))               # (y,x,k)
    wetc = jnp.asarray((dz > 0).astype(np.float64))
    cfg = ShortwavePenetrationConfig(scheme="rgb_chl")
    rho_o, c_o = 1025.0, 3994.0
    F_O_acc = np.zeros_like(F_N); qacc = np.zeros(lat_s.shape)
    chl_seen = []
    for t in range(nt):
        q = qsr[t]
        if not np.any(q > 0):
            continue
        chl = month_sample(chl_clim, t, a.dt, "linear")[jj, ii]
        chl_seen.append(float(np.average(chl, weights=w_area)))
        tend = apply_shortwave_penetration(
            cfg, jnp.ones(lat_s.shape), chl=jnp.asarray(chl), dz_live=dz, wet_cell=wetc,
            rho_0=rho_o, c_sw=c_o)
        frac = np.moveaxis(np.asarray(tend) * rho_o * c_o * np.asarray(dz), -1, 0)
        F_O_acc += frac * np.nan_to_num(q)[None]; qacc += np.nan_to_num(q)
    F_O = F_O_acc / np.maximum(qacc, 1e-30)[None]
    print(f"[ours] strip-mean chlorophyll over daylight steps {np.min(chl_seen):.4f}..{np.max(chl_seen):.4f} mg/m3; "
          f"column sum min {F_O.sum(axis=0).min():.6f} max {F_O.sum(axis=0).max():.6f}")

    # ---- cumulative absorbed fraction above each w-level, area x qsr weighted ----
    wgt = w_area * qsum
    gdepw = np.cumsum(np.nan_to_num(e3_mean), axis=0)                          # bottom of level k
    zbar = np.average(gdepw.reshape(gdepw.shape[0], -1), axis=1, weights=wgt.ravel())
    cN = np.cumsum(F_N, axis=0); cO = np.cumsum(F_O, axis=0)
    print(f"\n{'depth(m)':>9s} {'NEMO cum':>10s} {'ours cum':>10s} {'ours-NEMO':>10s}   (fraction of surface qsr absorbed ABOVE this depth)")
    for k in range(gdepw.shape[0]):
        if zbar[k] > a.zmax:
            break
        n_ = np.average(cN[k], weights=wgt); o_ = np.average(cO[k], weights=wgt)
        print(f"{zbar[k]:9.2f} {n_:10.4f} {o_:10.4f} {o_ - n_:+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

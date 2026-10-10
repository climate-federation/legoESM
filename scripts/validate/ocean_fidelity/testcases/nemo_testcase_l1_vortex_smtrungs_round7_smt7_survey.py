#!/usr/bin/env python3
"""SMT-RUNGS round 7: SMT-7 (GM eddy-induced transport + MLE) survey measurements.

Offline replay of NEMO's own statements on a NEMO-written SMT-6 record (S-EOS,
rn_a0 = 0.28, rn_b0 = 0, no pressure term):

``mle``  domzgr.f90:384-386 (nlb10/nla10) -> tramle.f90 mixed layer (the
         nla10 > 0 scan, else inml_mle = mbkt+1) -> zbm -> nn_mle=1 psi ->
         the bolus u increment.  Reports which branch runs, the ML depth, the
         unclamped psi scale and the induced face velocity.
``eiv``  ldftra.f90 ldf_eiv: Rossby radius with NEMO's own |2 omega sin(rad*gphit)|
         (gphit on this deck is a beta-plane y in km) against the same chain with
         ff_t.  Slopes are NOT replayed: the eiv-liveness line is a PROXY
         (centred density gradients), not NEMO's wslpi/wslpj.

``--plant-min-e3w X`` replaces MINVAL(e3w_1d) in the zrefdep line; with X = 10 the
nla10 > 0 scan runs and the ML collapses to the first level, so the unplanted
claim (ML = whole wet column) fails and the run exits 3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/"
              "round5/oracle_vortex_smt6/day100")
RN_CE, RN_LAT, RN_RHO_C = 0.06, 20.0, 0.01          # ORCA2 rung 3 namtra_mle
RN_UE, RN_LE = 0.03, 200.0e3                         # ORCA2 rung 3 namtra_eiv
RN_A0 = 0.28                                         # SMT namelist_cfg:133
RHO0, GRAV, OMEGA = 1026.0, 9.8066499999999994, 7.2921150830460618e-5  # ocean.output
RN_SLPMAX = 0.01
R5_21 = 5.0 / 21.0


def nla10_nlb10(gdepw_1d: np.ndarray, min_e3w: float):
    """domzgr.f90:384-386; 1-based nlb10, nla10 = nlb10 - 1 (may be 0)."""
    zrefdep = 10.0 - 0.1 * min_e3w
    cand = np.where(gdepw_1d > zrefdep)[0]
    nlb10 = int(cand[np.argmin(gdepw_1d[cand])]) + 1
    return nlb10 - 1, nlb10, zrefdep


def seos_rhop(T, S, tmask, rho0=RHO0, a0=RN_A0, t0=10.0):
    """eosbn2.f90 np_seos branch with b0 = lambda = nu = 0."""
    return (rho0 - a0 * (T - t0)) * tmask


def mle_mixed_layer(rhop, tmask, e3t, r3t, mbkt, nla10, nlb10):
    """tramle.f90: inml_mle, zmld, zbm (1-based inml)."""
    nk = rhop.shape[0]
    inml = (mbkt + 1).astype(int)
    if nla10 > 0:
        ref = rhop[nla10 - 1]
        for jk in range(nk - 1, nlb10 - 2, -1):           # jpkm1 .. nlb10
            inml = np.where(rhop[jk] > ref + RN_RHO_C, jk + 1, inml)
    ikmax = min(int(inml.max()), nk - 1)
    zmld = np.zeros(rhop.shape[1:])
    zbm = np.zeros(rhop.shape[1:])
    for jk in range(1, ikmax + 1):
        zc = e3t[jk - 1] * (1.0 + r3t * tmask[jk - 1]) * np.clip(inml - jk, 0, 1)
        zmld += zc
        zbm += zc * (RHO0 - rhop[jk - 1]) / RHO0
    zbm = GRAV * zbm / np.maximum(e3t[0] * (1.0 + r3t * tmask[0]), zmld)
    return inml, ikmax, zmld, zbm


def mle_psi_u(zmld, zbm, gdepw_1d, r3t, e1u, e2u, umask3, ikmax):
    """nn_mle=1 u-face psi at w-levels 1..ikmax+1 (tramle.f90 psi block)."""
    rc_f = RN_CE / (5.0e3 * 2.0 * OMEGA * np.sin(np.deg2rad(RN_LAT)))
    nj, ni = zmld.shape
    zh = np.minimum(zmld[:, 1:], zmld[:, :-1])            # nn_mld_uv = 0
    psim = np.zeros((nj, ni))
    psim[:, :-1] = (rc_f * zh * zh * (e2u[:, :-1] / e1u[:, :-1])
                    * (zbm[:, 1:] - zbm[:, :-1]) * np.minimum(111.0e3, e1u[:, :-1]))
    rzh = 1.0 / np.maximum(zh, 0.5 * np.finfo(float).eps)
    wum = np.ones_like(umask3)                            # wumask(k) = umask(k)*umask(k-1)
    wum[1:] = umask3[1:] * umask3[:-1]
    psi = np.zeros((ikmax + 1, nj, ni))                   # psi[0] = surface = 0
    for jk in range(1, ikmax + 1):
        g = gdepw_1d[jk]
        c = 1.0 - (g * (1.0 + r3t[:, 1:]) + g * (1.0 + r3t[:, :-1])) * rzh
        c = c * c
        mu = np.maximum(0.0, (1.0 - c) * (1.0 + R5_21 * c))
        psi[jk, :, :-1] = psim[:, :-1] * mu * wum[jk] [:, :-1] * wum[0][:, :-1]
    return psim, psi


def measure(rec: Path, min_e3w_override: float | None = None) -> dict:
    import netCDF4
    m = netCDF4.Dataset(rec / "mesh_mask.nc")
    sq = lambda n: np.squeeze(np.ma.filled(m[n][:], 0)).astype(float)  # noqa: E731
    tmask, e3t0 = sq("tmask"), sq("e3t_0")
    gdepw, e3w1d = sq("gdepw_1d"), sq("e3w_1d")
    gphit, fft = sq("gphit"), sq("ff_t")
    e1u, e2u = sq("e1u"), sq("e2u")
    nk = tmask.shape[0]
    min_e3w = float(e3w1d.min()) if min_e3w_override is None else min_e3w_override
    nla10, nlb10, zrefdep = nla10_nlb10(gdepw, min_e3w)
    ssh_files = sorted(rec.glob("*_restart.nc"))
    out = {"nla10": nla10, "nlb10": nlb10, "zrefdep_m": zrefdep,
           "min_e3w_m": min_e3w, "restarts": {}}
    umask3 = tmask * np.concatenate([tmask[:, :, 1:], np.zeros_like(tmask[:, :, :1])], 2)
    mbkt = np.maximum(tmask.sum(0).astype(int), 1)
    ht0 = (e3t0 * tmask).sum(0)
    for tag, f in (("day100", ssh_files[-1]), ("day30", ssh_files[0])):
        d = netCDF4.Dataset(f)
        fl = lambda n: np.squeeze(np.ma.filled(d[n][:], 0)).astype(float)  # noqa: E731
        T, S = fl("tn"), fl("sn")
        r3t = fl("sshn") / np.maximum(ht0, 1e-30)
        rhop = seos_rhop(T, S, tmask)
        inml, ikmax, zmld, zbm = mle_mixed_layer(rhop, tmask, e3t0, r3t, mbkt,
                                                 nla10, nlb10)
        wet = tmask[0] > 0
        psim, psi = mle_psi_u(zmld, zbm, gdepw, r3t, e1u, e2u, umask3, ikmax)
        dpsi = psi[:-1] - psi[1:]
        e3u = np.minimum(e3t0[:, :, 1:], e3t0[:, :, :-1])
        den = e2u[None, :, :-1] * e3u[:ikmax]
        u = np.where(umask3[:ikmax, :, :-1] > 0, -dpsi[:, :, :-1] / np.maximum(den, 1e-30), 0.0)
        wetu = umask3[0, :, :-1] > 0            # land-side faces carry psim but wumask = 0
        full = (e3u[:ikmax] >= e3w1d.max() - 1e-9)
        col = (e3t0[:nk - 1] * tmask[:nk - 1]).sum(0) * (1.0 + r3t)
        out["restarts"][tag] = {
            "file": f.name,
            "ml_equals_column": bool(np.allclose(zmld[wet], col[wet], rtol=1e-12)),
            "zmld_min_max_m": [float(zmld[wet].min()), float(zmld[wet].max())],
            "ikmax": int(ikmax),
            "max_abs_psim_wet_faces_m3s": float(np.abs(psim[:, :-1][wetu]).max()),
            "max_abs_bolus_u_full_cells_ms": float(np.abs(np.where(full, u, 0.0)).max()),
            "max_abs_psi_m3s": float(np.abs(psi).max()),
            "max_abs_bolus_u_any_cell_ms": float(np.abs(u).max()),
            "faces_psi_nonzero": int((np.abs(psi).max(0)[:, :-1] > 0).sum()),
        }
    # eiv Rossby radius: NEMO ldf_eiv with gphit (km used as degrees) vs ff_t
    tg = netCDF4.Dataset(rec / "data_1m_potential_temperature_nomask.nc")
    T0 = np.ma.filled(tg[list(tg.variables)[-1]][:], 0).astype(float)
    T0 = T0.reshape((-1,) + tmask.shape)[0]
    n2 = np.zeros_like(T0)
    n2[1:] = GRAV * RN_A0 / RHO0 * (T0[:-1] - T0[1:]) / e3w1d[1:, None, None]
    n2 *= (tmask * np.concatenate([np.zeros_like(tmask[:1]), tmask[:-1]], 0))
    zn = (np.sqrt(np.maximum(n2, 0.0)) * e3w1d[:, None, None]).sum(0)
    wet = tmask[0] > 0
    zfw_g = np.maximum(np.abs(2 * OMEGA * np.sin(np.deg2rad(gphit))), 1e-10)
    ro_g = np.clip(0.4 * zn / zfw_g, 2.0e3, 40.0e3)
    ro_f = np.clip(0.4 * zn / np.maximum(np.abs(fft), 1e-10), 2.0e3, 40.0e3)
    d = np.abs(ro_g - ro_f)[wet]
    out["eiv_rossby"] = {
        "gphit_min_max_as_degrees": [float(gphit.min()), float(gphit.max())],
        "ff_t_min_max": [float(fft.min()), float(fft.max())],
        "aei0_m2s": 0.5 * RN_UE * RN_LE,
        "wet_columns": int(wet.sum()),
        "columns_ro_gphit_ne_ro_fft": int((d > 1e-6).sum()),
        "max_rel_diff_ro": float((d / np.maximum(ro_f[wet], 1.0)).max()),
        "ro_gphit_clamped_40km": int((ro_g[wet] >= 40.0e3 - 1e-6).sum()),
        "ro_fft_clamped_40km": int((ro_f[wet] >= 40.0e3 - 1e-6).sum()),
    }
    # eiv liveness PROXY: centred horizontal density difference exists
    rho = seos_rhop(T0, np.zeros_like(T0), tmask)
    dx = np.abs(rho[:, :, 1:] - rho[:, :, :-1]) * (tmask[:, :, 1:] * tmask[:, :, :-1])
    dz = np.abs(rho[1:] - rho[:-1]) * (tmask[1:] * tmask[:-1])
    s = dx[:-1, :, :] / np.maximum(0.5 * (dz[:, :, 1:] + dz[:, :, :-1]), 1e-12) \
        * (e3w1d[1:, None, None] / e1u[None, :, :-1])
    out["eiv_proxy"] = {
        "label": "PROXY centred rho gradients, not NEMO wslpi/wslpj",
        "u_faces_with_horizontal_density_difference": int((dx > 1e-12).any(0).sum()),
        "max_proxy_slope": float(np.minimum(s, RN_SLPMAX).max()),
        "slpmax": RN_SLPMAX,
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=Path, default=RECORD)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plant-min-e3w", type=float, default=None)
    a = ap.parse_args(argv)
    res = measure(a.record, a.plant_min_e3w)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True))
    ok = res["nla10"] == 0 and all(
        r["ml_equals_column"] for r in res["restarts"].values())
    print(json.dumps({k: res[k] for k in ("nla10", "nlb10", "zrefdep_m")}),
          "ml_equals_column:", ok)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())

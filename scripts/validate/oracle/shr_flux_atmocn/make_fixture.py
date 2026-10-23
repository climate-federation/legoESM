"""Generate tests/core/oracle_fixtures/shr_flux_atmocn_cesm21.npz by RUNNING the
CESM2.1 ocean surface-flux oracle (cime5.6.47 ``shr_flux_atmOcn``, compiled
unmodified by ``build.sh``).

Inputs are ERA5 open-ocean states (ACE2 ``ic_2001.nc``: 1st of month 12 UTC,
ocean_fraction > 0.999, no sea ice), 2 m T/q and 10 m wind RELABELLED as the
lowest-model-level state, plus six named regimes, every cold-air-outbreak
point (tbot - SST < -10 K) of the pool up to ``--n-coa``, and seven synthetic
edge states.  The CAM bottom-level quantities are built as CAM6 builds the
a2x state (camsrfexch.F90): L32 bottom layer, thbot = T (ps/pmid)^kappa,
zbot = R Tv/g ln(ps/pmid), rbot = pmid/(R T) (dry).  This is the tier-1 audit
harness (oracle_audit/surface_bulk_flux_large_yeager_cesm/ours/harness.py prep)
with a smaller random sample.

The oracle runs with legoESM's constants (``shr_flux_adjust_constants``) and
the CESM2.1 f09 driver settings: flux_max_iteration = 5, flux_convergence =
0.0, coldair_outbreak_mod = .true., minwind 0.5, gust_fac 0, us = vs = 0.

    bash build.sh /path/to/cime5.6.47 /tmp/shrflx_build
    python make_fixture.py --drv /tmp/shrflx_build/drv --out tests/core/oracle_fixtures/shr_flux_atmocn_cesm21.npz
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants

CIME_TAG = "cime5.6.47"
CIME_SHA = "d8e9b894d7eb96fe96f8f8a6943b421147f94606"
SHR_FLUX_MOD_SHA1 = "40295186e9fcd7d5b099c9451534ad42dfcc8a58"
# CESM shr_const values, used ONLY to build CAM-convention bottom-level inputs.
SHR_RGAS = 6.02214e26 * 1.38065e-23  # const-ok: CESM shr_const oracle value
SHR_RDAIR = SHR_RGAS / 28.966  # const-ok: CESM shr_const oracle value
SHR_ZVIR = (SHR_RGAS / 18.016) / SHR_RDAIR - 1.0  # const-ok: CESM shr_const oracle value
SHR_CPDAIR = 1.00464e3  # const-ok: CESM shr_const oracle value
SHR_G = 9.80616  # const-ok: CESM shr_const oracle value
# driver.F90 hard-codes legoESM's constants; refuse to run if they drifted.
DRIVER_LEGO = {"R_d": 287.05, "R_v": 461.51, "c_pd": 1004.64, "c_pv": 1846.0,  # const-ok: driver.F90 mirror
               "g": 9.80616, "kappa_vk": 0.4, "L_v": 2.501e6}  # const-ok: driver.F90 mirror
SYNTH = [  # name, T, q, u, v, ps, sst  (T None: thbot == sst exactly)
    ("edge_calm_u0_v0", 300.0, 0.017, 0.0, 0.0, 101000.0, 301.0),
    ("edge_exact_neutral_dtheta0", None, 0.012, 6.0, 2.0, 101300.0, 293.0),
    ("edge_hol_clamp_stable", 292.0, 0.0125, 1.5, 0.5, 101500.0, 276.0),
    ("edge_strong_cao_coa_active", 252.0, 0.0008, 12.0, -6.0, 100500.0, 279.0),
    ("edge_hurricane_35ms", 300.0, 0.019, 30.0, 18.0, 97000.0, 302.5),
    ("edge_near_minwind_0p6", 299.0, 0.016, 0.5, 0.3, 101200.0, 302.0),
    ("edge_dry_polar_open_water", 262.0, 0.0012, 4.0, 1.0, 99500.0, 271.4),
]
# Raw kernel-input rows (zbot u v thbot q rbot tbot ts): a state where ustar
# repeats EXACTLY after a pass while the stability flag still flips, so the
# Fortran flux_convergence = 0 early exit changes the answer (LH 160.5 vs
# 153.8 W/m2 with five forced passes; codex review 2026-10-19).
RAW = [("edge_exact_ustar_repeat_early_exit",
        (10.0, 6.0, 0.0, 300.0, 0.02, 1.8180680423029751, 300.0, 301.0))]
OUT_KEYS = "sen lat lwup evap taux tauy tref qref duu10n ustar re ssq".split()


def l32_bottom(ps):
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0
    ptop = CAM6_L32_HYAI[-2] * CAM6_L32_P0 + CAM6_L32_HYBI[-2] * ps
    return 0.5 * (ptop + ps)


def era5_pool(ic_path, fc_path):
    import netCDF4 as nc
    ic, fc = nc.Dataset(ic_path), nc.Dataset(fc_path)
    t_ic, t_fc = ic.variables["time"][:], fc.variables["time"][:]
    LA, LO = np.meshgrid(ic.variables["latitude"][:], ic.variables["longitude"][:], indexing="ij")
    rows = []
    for m in range(12):
        j = int(np.where(t_fc == t_ic[m])[0][0])
        ok = (np.asarray(fc.variables["ocean_fraction"][j]) > 0.999) & (
            np.asarray(fc.variables["sea_ice_fraction"][j]) <= 0.0)
        g = [np.asarray(ic.variables[k][m])[ok] for k in
             ("TMP2m", "Q2m", "UGRD10m", "VGRD10m", "PRESsfc", "surface_temperature")]
        rows.append(np.stack([np.full(ok.sum(), m), LA[ok], LO[ok], *g], axis=1))
    return np.concatenate(rows).astype(np.float64)  # month lat lon T q u v ps sst


def build_inputs(R, n_random, n_coa, seed):
    rng = np.random.default_rng(seed)
    lat, lon, T, u, v, sst = R[:, 1], R[:, 2], R[:, 3], R[:, 5], R[:, 6], R[:, 8]
    wind, dT = np.hypot(u, v), T - sst
    picks = {}
    s = np.where((sst > 302.0) & (np.abs(lat) < 10))[0]
    picks["tropical_warm_pool_calm"] = s[np.argmin(wind[s])]
    s = np.where((np.abs(lat) > 10) & (np.abs(lat) < 25) & (wind > 7) & (wind < 9))[0]
    picks["trade_cumulus"] = s[0]
    s = np.where((lat < -15) & (lat > -25) & (lon > 270) & (lon < 290) & (sst < 293))[0]
    picks["se_pacific_stratocumulus"] = s[len(s) // 2]
    s = np.where((np.abs(lat) > 40) & (np.abs(lat) < 60))[0]
    picks["midlatitude_storm_max_wind"] = s[np.argmax(wind[s])]
    picks["cold_air_outbreak_min_dT"] = int(np.argmin(dT))
    picks["warm_advection_stable_max_dT"] = int(np.argmax(dT))
    coa = np.where(dT < -10.0)[0]
    coa = coa[np.argsort(dT[coa])][:n_coa]
    rand = rng.choice(len(R), size=n_random, replace=False)
    idx = np.concatenate([list(picks.values()), coa, rand])
    labels = list(picks) + ["era5_coa"] * len(coa) + ["era5_random"] * n_random
    P = R[idx, 3:9].copy()  # T q u v ps sst
    for (nm, T_, q_, u_, v_, ps_, sst_) in SYNTH:
        if T_ is None:
            T_ = sst_ / (ps_ / l32_bottom(ps_)) ** (SHR_RDAIR / SHR_CPDAIR)
        P = np.vstack([P, [T_, q_, u_, v_, ps_, sst_]])
        labels.append(nm)
    T, q, u, v, ps, sst = P.T
    pmid = l32_bottom(ps)
    thbot = T * (ps / pmid) ** (SHR_RDAIR / SHR_CPDAIR)
    k = labels.index("edge_exact_neutral_dtheta0")
    thbot[k] = sst[k]  # bitwise delt == 0
    inp = dict(zbot=SHR_RDAIR / SHR_G * T * (1.0 + SHR_ZVIR * q) * np.log(ps / pmid),
               ubot=u, vbot=v, thbot=thbot, qbot=q, rbot=pmid / (SHR_RDAIR * T),
               tbot=T, ts=sst)
    for nm, row in RAW:
        for key, x in zip(("zbot", "ubot", "vbot", "thbot", "qbot", "rbot", "tbot", "ts"), row):
            inp[key] = np.append(inp[key], x)
        labels.append(nm)
    return inp, np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ic", default="/burg/glab/users/pg2328/ace2_era5/initial_conditions/ic_2001.nc")
    ap.add_argument("--fc", default="/burg/glab/users/pg2328/ace2_era5/forcing_data/forcing_2001.nc")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--n-coa", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args()
    for k, v in DRIVER_LEGO.items():
        assert getattr(constants, k) == v, (k, getattr(constants, k), v)
    inp, labels = build_inputs(era5_pool(a.ic, a.fc), a.n_random, a.n_coa, a.seed)
    order = ("zbot", "ubot", "vbot", "thbot", "qbot", "rbot", "tbot", "ts")
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.txt", Path(td) / "out.txt"
        n = len(labels)
        with open(fin, "w") as f:
            f.write(f"{n}\n")
            for i in range(n):
                f.write(" ".join(repr(float(inp[k][i])) for k in order) + "\n")
        cmd = [a.drv, "lego", "5", "0.0", "1", str(fin), str(fout)]
        log = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        assert "ORACLE_OK" in log, log
        d = np.loadtxt(fout, comments="#")
    out = {f"oracle_{k}": d[:, i] for i, k in enumerate(OUT_KEYS)}
    meta = (f"oracle={CIME_TAG} sha={CIME_SHA} shr_flux_mod.F90 sha1={SHR_FLUX_MOD_SHA1}; "
            "consts=legoESM via shr_flux_adjust_constants; flux_max_iteration=5 "
            "flux_convergence=0.0 coldair_outbreak_mod=.true. minwind=0.5 gust_fac=0 us=vs=0; "
            "CESM sign convention (fluxes positive DOWN, stress on the ocean along the wind); "
            f"generator: scripts/validate/oracle/shr_flux_atmocn/make_fixture.py "
            f"--n-random {a.n_random} --n-coa {a.n_coa} --seed {a.seed}; inputs {a.ic}")
    np.savez_compressed(a.out, labels=labels, meta=np.array(meta),
                        **{f"in_{k}": inp[k] for k in order}, **out)
    print("FIXTURE_OK", a.out, n, "points;", int((inp["tbot"] - inp["ts"] < -10).sum()), "COA-active")


if __name__ == "__main__":
    main()

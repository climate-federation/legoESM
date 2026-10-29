"""Generate the AeroBulk oracle fixtures by RUNNING AeroBulk 1.0.0.

    bash build.sh /path/to/aerobulk-1.0.0 /tmp/ab_build
    python make_fixture.py --drv /tmp/ab_build/drv --scheme coare3p0 \\
        --ic ic_2001.nc --fc forcing_2001.nc \\
        --out tests/core/oracle_fixtures/aerobulk_coare3p0.npz

``--scheme coare3p0`` keeps the TURB_COARE3P0 outputs, ``--scheme ncar`` the
TURB_NCAR ones (Large & Yeager 2004/2009).

Inputs (the tier-2 audit harness, oracle_audit/surface_layer_most_coare_ly09/
ours/harness.py, with a smaller random ERA5 sample):

* ERA5 open-ocean states (ACE2 ``ic_2001.nc``, 1st of month 12 UTC,
  ocean_fraction > 0.999, no sea ice): seven named regimes plus 500 random
  points (seed 20261018); 2 m T/q and 10 m wind;
* the 1032 states of the frozen aerobulk-python baseline
  (tests/unit/baselines/aerobulk_noskin_v1.npz);
* ten synthetic edges (calm, free convection, very stable, hurricane winds,
  T_air == SST).

Each state is evaluated at two height pairs, zt/zu = 10/10 and 2/10 m.  The
common inputs follow tests/unit/test_aerobulk_oracle.py: Exner potential
temperature of air and surface, q_s = 0.98 q_sat (Goff), moist rho, Kirchhoff
L at the surface potential temperature.  The oracle receives (theta_s, theta_a,
q_s, q_a, |U|, p) and returns Cd, Ch, Ce, t_zu, q_zu, Ub and u* after
nb_iter = 5 iterations (the legoESM bulk_n_iter default).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants

AEROBULK_TAG = "1.0.0"
AEROBULK_SHA = "ca40928872eb9c00f96a51cf3a6eed0bb4a70580"
NB_ITER = 5
HEIGHTS = {"z10": (10.0, 10.0), "z2": (2.0, 10.0)}  # name: (zt, zu)
OUT_NAMES = ["Cd", "Ch", "Ce", "t_zu", "q_zu", "Ub", "CdN", "ChN", "CeN", "z0", "us", "L", "UN10"]
KEEP = ["Cd", "Ch", "Ce", "t_zu", "q_zu", "Ub", "us"]
EDGES = np.array([  # sst, T_air, q_air, u, v, slp  (synthetic, documented)
    [290.0, 290.0, 9.0e-3, 0.0, 0.0, 101325.0],     # dead calm, neutral T, subsaturated
    [302.0, 299.0, 1.7e-2, 0.1, 0.0, 101000.0],     # calm free convection (gustiness)
    [271.5, 281.0, 3.0e-3, 1.0, 0.0, 101500.0],     # very stable light wind
    [303.0, 293.0, 1.2e-2, 4.0, 0.0, 101000.0],     # very unstable
    [300.0, 298.0, 1.8e-2, 35.0, 0.0, 97000.0],     # LY plateau region (>33 m/s)
    [300.0, 298.0, 1.8e-2, 45.0, 0.0, 96000.0],     # hurricane
    [300.0, 298.0, 1.8e-2, 55.0, 0.0, 95000.0],     # beyond 50 m/s
    [295.0, 295.0, 1.0e-2, 6.0, 2.0, 101325.0],     # T_air == SST, dq != 0
    [278.0, 286.0, 5.5e-3, 2.0, -1.0, 104000.0],    # warm air over cold sea
    [285.0, 284.0, 7.0e-3, 0.3, 0.3, 101325.0],     # near-calm weakly unstable
])


def build_constants(build_sh=Path(__file__).with_name("build.sh")):
    """(grav, R_vap) that build.sh writes into mod_const.f90."""
    src = build_sh.read_text()
    g = float(re.search(r"grav  = ([0-9.]+)  !/'", src).group(1))
    rv = float(re.search(r"R_vap = ([0-9.]+)     !/'", src).group(1))
    return g, rv


def era5_points(ic_path, fc_path, n_random=500, seed=20261018):
    import netCDF4 as nc
    rng = np.random.default_rng(seed)
    ic, fc = nc.Dataset(ic_path), nc.Dataset(fc_path)
    t_ic = np.ma.filled(ic.variables["time"][:], np.nan)
    t_fc = np.ma.filled(fc.variables["time"][:], np.nan)
    lat = np.ma.filled(ic.variables["latitude"][:], np.nan)
    lon = np.ma.filled(ic.variables["longitude"][:], np.nan)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    rows = []
    for m in range(12):
        j = int(np.where(t_fc == t_ic[m])[0][0])
        ok = (np.ma.filled(fc.variables["ocean_fraction"][j], np.nan) > 0.999) & (
            np.ma.filled(fc.variables["sea_ice_fraction"][j], np.nan) <= 0.0)
        g = {k: np.ma.filled(ic.variables[k][m], np.nan) for k in
             ("TMP2m", "Q2m", "UGRD10m", "VGRD10m", "PRESsfc", "surface_temperature")}
        rows.append(np.stack([LA[ok], LO[ok], g["TMP2m"][ok], g["Q2m"][ok], g["UGRD10m"][ok],
                              g["VGRD10m"][ok], g["PRESsfc"][ok], g["surface_temperature"][ok]], 1))
    R = np.concatenate(rows).astype(np.float64)
    R = R[np.all(np.isfinite(R), axis=1)]
    lat_, lon_, T, q, u, v, ps, sst = R.T
    wind, dT = np.hypot(u, v), T - sst
    picks = {}
    s = np.where((sst > 302.0) & (np.abs(lat_) < 10))[0]
    picks["era5_tropical_warm_pool_calm"] = s[np.argmin(wind[s])]
    s = np.where((np.abs(lat_) > 10) & (np.abs(lat_) < 25) & (wind > 7) & (wind < 9))[0]
    picks["era5_trade_cumulus"] = s[0]
    s = np.where((lat_ < -15) & (lat_ > -25) & (lon_ > 270) & (lon_ < 290) & (sst < 293))[0]
    picks["era5_se_pacific_stratocumulus"] = s[len(s) // 2]
    s = np.where((np.abs(lat_) > 40) & (np.abs(lat_) < 60))[0]
    picks["era5_midlatitude_storm_max_wind"] = s[np.argmax(wind[s])]
    picks["era5_cold_air_outbreak_min_dT"] = int(np.argmin(dT))
    picks["era5_warm_advection_stable_max_dT"] = int(np.argmax(dT))
    s = np.where(np.abs(lat_) > 55)[0]
    picks["era5_polar_open_ocean"] = s[np.argmin(sst[s])]
    idx = np.concatenate([np.array(list(picks.values())),
                          rng.choice(len(R), n_random, replace=False)])
    labels = list(picks) + ["era5_random"] * n_random
    return dict(T=T[idx], q=q[idx], u=u[idx], v=v[idx], slp=ps[idx], sst=sst[idx]), labels


def frozen_points(path):
    z = np.load(path)
    d = dict(T=z["in_t_air"], q=z["in_q_air"], u=z["in_u"], v=z["in_v"], slp=z["in_slp"],
             sst=z["in_sst"])
    return d, ["frozen_aerobulk_python"] * len(d["T"])


def edge_points():
    e = EDGES
    d = dict(sst=e[:, 0], T=e[:, 1], q=e[:, 2], u=e[:, 3], v=e[:, 4], slp=e[:, 5])
    return d, [f"edge_{i}" for i in range(len(e))]


def derive(d, zt):
    """Model-side inputs, same convention as tests/unit/test_aerobulk_oracle.py."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.ocean.bulk_flux_omip import (
        exner_potential_temperature,
        pressure_at_height,
        rho_air_moist,
        seawater_q_sat,
    )
    from legoesm.thermo import latent_heat_vaporization
    q, slp, T, sst = (jnp.asarray(d[k], dtype=jnp.float64) for k in ("q", "slp", "T", "sst"))
    p_zt = pressure_at_height(q, slp, zt, T)
    th_s = exner_potential_temperature(sst, slp)
    return dict(u=np.asarray(d["u"], float), v=np.asarray(d["v"], float),
                th_a=np.asarray(exner_potential_temperature(T, p_zt)), q=np.asarray(q),
                th_s=np.asarray(th_s), qs=np.asarray(seawater_q_sat(sst, slp)),
                rho=np.asarray(rho_air_moist(T, q, p_zt)),
                L=np.asarray(latent_heat_vaporization(th_s)),
                W=np.hypot(np.asarray(d["u"], float), np.asarray(d["v"], float)),
                slp=np.asarray(slp))


def run_oracle(drv, x, zt, zu):
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.txt", Path(td) / "out.txt"
        with open(fin, "w") as fh:
            fh.write(f"{len(x['th_s'])} {zt!r} {zu!r}\n")
            np.savetxt(fh, np.stack([x["th_s"], x["th_a"], x["qs"], x["q"], x["W"], x["slp"]], 1),
                       fmt="%.17e")
        subprocess.run([drv, str(fin), str(fout), str(NB_ITER)], check=True)
        out = {"C": [], "N": [], "H": []}
        with open(fout) as fh:
            header = fh.readline().strip()
            for line in fh:
                k, *vals = line.split()
                out[k].append([float(s) for s in vals])
    res = {"header": header}
    for k, alg in (("C", "coare3p0"), ("N", "ncar")):
        arr = np.array(out[k])
        res[alg] = {nm: arr[:, i] for i, nm in enumerate(OUT_NAMES)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drv", required=True)
    ap.add_argument("--scheme", choices=("coare3p0", "ncar"), required=True)
    ap.add_argument("--ic", required=True)
    ap.add_argument("--fc", required=True)
    ap.add_argument("--frozen", default=str(Path(__file__).resolve().parents[4]
                                            / "tests/unit/baselines/aerobulk_noskin_v1.npz"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    g, rv = build_constants()
    if (g, rv) != (constants.g, constants.R_v):
        raise SystemExit(f"build.sh constants {(g, rv)} != legoesm {(constants.g, constants.R_v)}")
    parts = [era5_points(a.ic, a.fc), frozen_points(a.frozen), edge_points()]
    d = {k: np.concatenate([p[0][k] for p in parts]) for k in ("T", "q", "u", "v", "slp", "sst")}
    labels = sum((p[1] for p in parts), [])
    save = {"labels": np.array(labels)}
    header = None
    for hname, (zt, zu) in HEIGHTS.items():
        x = derive(d, zt)
        o = run_oracle(a.drv, x, zt, zu)
        header = o["header"]
        for k in ("u", "v", "th_a", "q", "th_s", "qs", "rho", "L", "W"):
            save[f"{hname}_in_{k}"] = x[k]
        for k in KEEP:
            if not np.all(np.isfinite(o[a.scheme][k])):
                raise SystemExit(f"non-finite oracle output {hname} {k}")
            save[f"{hname}_oracle_{k}"] = o[a.scheme][k]
    meta = dict(oracle="AeroBulk", tag=AEROBULK_TAG, sha=AEROBULK_SHA,
                routine="TURB_COARE3P0" if a.scheme == "coare3p0" else "TURB_NCAR",
                nb_iter=NB_ITER, heights=HEIGHTS, driver_header=header,
                mod_const=f"grav={g!r}, R_vap={rv!r} (legoESM values as default-real literals)",
                skin="l_use_cs = l_use_wl = .false.")
    save["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **save)
    print("wrote", a.out, len(labels), "states x", len(HEIGHTS), "height pairs")


if __name__ == "__main__":
    main()

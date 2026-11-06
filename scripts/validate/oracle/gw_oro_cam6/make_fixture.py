"""Generate tests/atmosphere/oracle_fixtures/gw_oro_cam6.npz by RUNNING the
CAM6 orographic gravity-wave-drag oracle (ESCOMP/CAM cam_cesm2_1_rel_60
``gw_prof`` + ``gw_oro_src`` + ``gw_drag_prof`` with gw vertical diffusion,
compiled unmodified by ``build.sh``; driver ``driver.F90``).

Columns (the tier-1 audit set, oracle_audit/orographic_gwd_mcfarlane
py/build_inputs.py): CAM6 L32 levels from legoESM ``make_cam6_l32_levels``;
nine constructed soundings (lapse-rate T, RH-based q, winds piecewise linear
in ln p) and three real T/q/p columns from the legoESM fixture
``zm_cam6_real_zero_humidity_columns.npz`` (MPAS AMIP checkpoints) with
constructed winds / sgh / landfrac, plus two constructed limiter columns
(umcfac and tndmax binding) added for this fixture.  z_full, z_half and rho come from the
production helpers ``compute_heights_from_sigma`` / ``compute_rho`` (q_v
passed), i.e. exactly what ``_make_mpas_gwd`` hands the scheme.  The oracle
runs with legoESM's g, R_d, c_pd and the CAM6 f09 namelist values
effgw_oro 0.125, fcrit2 1, gw_prndl 0.25, gw_lndscl_sgh .true.,
gw_apply_tndmax .true., wavelength_mid 1e5 m; dt = 1800 s; one constituent
(q_v).  pref_edge = the L32 interfaces at the reference surface pressure
psr = 1e5 Pa (CAM hycoef: hypi = hyai*ps0 + hybi*psr).

    bash build.sh /path/to/CAM /tmp/gworo_build
    python make_fixture.py --drv /tmp/gworo_build/gw_oro_oracle \
        --out tests/atmosphere/oracle_fixtures/gw_oro_cam6.npz
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma, compute_rho,
)
from legoesm.grids.vertical import make_cam6_l32_levels
from legoesm.thermo import saturation_mixing_ratio

CAM_TAG = "cam_cesm2_1_rel_60"
CAM_SHA = "a03b84b7c4e34f965b115686f22a043b85739e56"
DT = 1800.0            # CAM6 physics step
EFFGW_ORO = 0.125      # CAM6 namelist_defaults_cam.xml effgw_oro
FCRIT2 = 1.0           # build-namelist fcrit2
PRNDL = 0.25           # namelist_defaults_cam.xml gw_prndl
WAVELENGTH = 1.0e5     # gw_drag.F90 wavelength_mid [m]
REPO = Path(__file__).resolve().parents[4]
REAL_FX = REPO / "tests/atmosphere/hydrostatic/unit/fixtures/zm_cam6_real_zero_humidity_columns.npz"
coord = make_cam6_l32_levels()
NLEV = 32
FIELDS = ("utgw", "vtgw", "ttgw", "dttke", "dttdf", "tau")


def sounding(ps_hpa, Ts, Ttrop, rh, inv_dT=0.0, inv_dp=4000.0):
    ps = ps_hpa * 100.0
    p_half = np.asarray(coord.pressure_at_half(np.array([ps])))[0]
    p_full = np.asarray(coord.pressure_at_full(np.array([ps])))[0]
    gam = 6.5e-3
    T = Ts * (p_full / ps) ** (constants.R_d * gam / constants.g)
    T = np.maximum(T, Ttrop)
    T = np.where(p_full < 10000.0, Ttrop + 12.0 * np.log(10000.0 / p_full), T)
    T = T - inv_dT * np.clip(1.0 - (ps - p_full) / inv_dp, 0.0, None)
    qs = np.asarray(saturation_mixing_ratio(T, p_full))
    q = rh * qs / (1.0 + rh * qs)          # specific humidity from mixing ratio
    q = np.where(p_full < 10000.0, 3e-6, q)
    return p_half, T, q


def winds(p_full, anchors_u, anchors_v):
    lp = np.log(p_full)
    out = []
    for anc in (anchors_u, anchors_v):
        a = np.array(anc, float)
        i = np.argsort(a[:, 0])
        out.append(np.interp(lp, np.log(a[i, 0] * 100), a[i, 1]))
    return out


CONSTRUCTED = [
    dict(name="rockies_winter", ps=850, Ts=268, Ttrop=218, rh=0.6, lat=40, sgh=400, lf=1.0,
         u=[(850, 6), (700, 15), (500, 28), (250, 48), (100, 25), (30, 12), (5, 20)],
         v=[(850, 4), (700, 6), (500, 5), (250, 0), (5, 0)]),
    dict(name="tibet_summer_stratcrit", ps=580, Ts=288, Ttrop=210, rh=0.5, lat=32, sgh=600, lf=1.0,
         u=[(580, 5), (400, 12), (200, 28), (100, 10), (70, -2), (30, -15), (5, -25)],
         v=[(580, -2), (5, 0)]),
    dict(name="andes_tropical_tropcrit", ps=700, Ts=291, Ttrop=195, rh=0.7, lat=-15, sgh=500, lf=0.8,
         u=[(700, -7), (600, -4), (450, 3), (200, 18), (100, 5), (50, -10), (5, -20)],
         v=[(700, 3), (200, -2), (5, 0)]),
    dict(name="antarctic_plateau_inversion", ps=680, Ts=236, Ttrop=205, rh=0.8, lat=-78, sgh=150, lf=1.0,
         inv_dT=18.0, inv_dp=4000.0,
         u=[(680, 4), (640, 9), (500, 12), (300, 18), (100, 30), (10, 45), (5, 50)],
         v=[(680, -3), (500, -2), (5, 0)]),
    dict(name="norway_partial_land", ps=990, Ts=279, Ttrop=220, rh=0.8, lat=62, sgh=250, lf=0.35,
         u=[(990, 10), (850, 18), (500, 30), (250, 42), (100, 20), (5, 15)],
         v=[(990, -6), (500, -8), (250, -5), (5, 0)]),
    dict(name="ocean_zero_sgh", ps=1010, Ts=300, Ttrop=195, rh=0.8, lat=5, sgh=0.0, lf=0.0,
         u=[(1010, -6), (200, 10), (5, -10)], v=[(1010, 1), (5, 0)]),
    dict(name="weak_wind_gate", ps=900, Ts=290, Ttrop=200, rh=0.5, lat=25, sgh=300, lf=1.0,
         u=[(900, 1.2), (500, 8), (200, 20), (5, 5)], v=[(900, 0.6), (5, 0)]),
    dict(name="tiny_sgh_gate", ps=950, Ts=285, Ttrop=215, rh=0.6, lat=45, sgh=4.0, lf=1.0,
         u=[(950, 10), (250, 35), (5, 10)], v=[(950, 0), (5, 0)]),
    dict(name="alps_strong_jet", ps=850, Ts=275, Ttrop=215, rh=0.6, lat=46, sgh=700, lf=1.0,
         u=[(850, 15), (700, 25), (500, 35), (250, 60), (100, 30), (5, 40)],
         v=[(850, 5), (500, 3), (5, 0)]),
    # Limiter columns: a weak (1 m/s) mid-troposphere layer makes the
    # umcfac limit bind (its remainder is carried to the layer below); a
    # stratospheric jet collapsing to 20 m/s at the top makes tndmax bind.
    dict(name="umcfac_weak_layer", ps=850, Ts=275, Ttrop=215, rh=0.6, lat=44, sgh=700, lf=1.0,
         u=[(850, 15), (700, 25), (450, 20), (420, 0.2), (300, 0.2), (250, 25), (100, 30), (5, 40)],
         v=[(850, 0), (5, 0)]),
    dict(name="tndmax_top_jet", ps=850, Ts=275, Ttrop=215, rh=0.6, lat=48, sgh=700, lf=1.0,
         u=[(850, 15), (700, 25), (500, 35), (250, 60), (100, 50), (10, 60), (5, 20)],
         v=[(850, 4), (5, 0)]),
]
REAL = [("real_8116", 200.0, 0.9, 38.0,
         [(1000, 8), (700, 16), (300, 38), (150, 30), (50, 10), (5, 25)],
         [(1000, 3), (300, 2), (5, 0)]),
        ("real_17416", 350.0, 1.0, -33.0,
         [(1000, -3), (800, 2), (500, 10), (200, 25), (50, 5), (5, -15)],
         [(1000, 2), (500, -3), (5, 0)]),
        ("real_33109", 120.0, 0.6, 55.0,
         [(1000, 12), (850, 20), (400, 35), (200, 45), (60, 15), (5, 30)],
         [(1000, -5), (400, -6), (5, 0)])]


def build_columns():
    cols = []
    for c in CONSTRUCTED:
        p_half, T, q = sounding(c["ps"], c["Ts"], c["Ttrop"], c["rh"],
                                c.get("inv_dT", 0.0), c.get("inv_dp", 4000.0))
        p_full = np.asarray(coord.pressure_at_full(np.array([p_half[-1]])))[0]
        u, v = winds(p_full, c["u"], c["v"])
        cols.append(dict(name=c["name"], ps=p_half[-1], T=T, q=q, u=u, v=v,
                         lat=c["lat"], sgh=c["sgh"], lf=c["lf"]))
    fx = np.load(REAL_FX)
    for j, (nm, sgh, lf, lat, au, av) in enumerate(REAL):
        p_half = np.asarray(fx["p_half32"][j], float)
        assert p_half[0] < p_half[-1], "fixture must be top-first"
        u, v = winds(np.asarray(fx["p_full32"][j], float), au, av)
        cols.append(dict(name=nm, ps=p_half[-1], T=np.asarray(fx["T32"][j], float),
                         q=np.clip(np.asarray(fx["q32"][j], float), 0.0, None),
                         u=u, v=v, lat=lat, sgh=sgh, lf=lf))
    d = {k: np.stack([c[k] for c in cols]) for k in ("T", "q", "u", "v")}
    d["names"] = np.array([c["name"] for c in cols])
    for k in ("sgh", "lf"):
        d[k] = np.array([c[k] for c in cols], float)
    d["lat"] = np.deg2rad(np.array([c["lat"] for c in cols], float))   # [rad]
    ps = np.array([c["ps"] for c in cols])
    d["p_half"] = np.asarray(coord.pressure_at_half(ps))
    d["p_full"] = np.asarray(coord.pressure_at_full(ps))
    z_full, z_half = compute_heights_from_sigma(d["T"], d["p_half"], q_v=d["q"])
    d["z_full"], d["z_half"] = np.asarray(z_full), np.asarray(z_half)
    d["rho"] = np.asarray(compute_rho(d["T"], d["p_full"], q_v=d["q"]))
    d["p_ref_half"] = np.asarray(coord.pressure_at_half(np.array([constants.p_ref])))[0]
    return d


def write_input(path, d):
    f = lambda a: " ".join(f"{x:.17e}" for x in np.ravel(a))
    ncol = d["T"].shape[0]
    with open(path, "w") as fh:
        fh.write(f"{ncol} {NLEV} 1\n{f((constants.g, constants.R_d, constants.c_pd))}\n")
        fh.write(f"{DT!r} {EFFGW_ORO!r} {FCRIT2!r} {PRNDL!r} {WAVELENGTH!r}\n")
        fh.write(f(d["p_ref_half"]) + "\n")
        for i in range(ncol):
            fh.write(f"{float(d['sgh'][i])!r} {float(d['lf'][i])!r}\n")
            for k in ("p_half", "T", "u", "v", "z_full", "q"):
                fh.write(f(d[k][i]) + "\n")


def parse_output(path):
    cols, alpha = [], None
    for line in open(path):
        if line.startswith("#"):
            continue
        tok = line.split()
        if tok[0] == "alpha":
            alpha = np.array(tok[1:], float)
        elif tok[0] == "col":
            cur = dict(src_level=int(tok[2]), effgw=float(tok[5]), sgh_scaled=float(tok[6]))
            cols.append(cur)
        else:
            cur[tok[0]] = np.array(tok[1:], float)
    out = {k: np.stack([c[k] for c in cols]) for k in cols[0]}
    out["alpha"] = alpha
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = build_columns()
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.txt", Path(td) / "out.txt"
        write_input(fin, d)
        r = subprocess.run([a.drv, str(fin), str(fout)], capture_output=True, text=True)
        if r.returncode != 0 or "ORACLE_OK" not in r.stdout:
            raise RuntimeError(f"oracle failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
        o = parse_output(fout)
    meta = (f"oracle=ESCOMP/CAM {CAM_TAG} sha={CAM_SHA} (gw_prof+gw_oro_src+gw_drag_prof, "
            f"gw vdiff on, lininterp alpha); dt={DT} effgw_oro={EFFGW_ORO} fcrit2={FCRIT2} "
            f"prndl={PRNDL} wavelength={WAVELENGTH}; constants legoESM g={constants.g} "
            f"R_d={constants.R_d} c_pd={constants.c_pd}; generator "
            "scripts/validate/oracle/gw_oro_cam6/{build.sh,make_fixture.py}")
    np.savez_compressed(
        a.out, meta=np.array(meta), dt=DT,
        **{k: d[k] for k in ("names", "T", "q", "u", "v", "lat", "sgh", "lf", "p_half",
                             "p_full", "z_full", "z_half", "rho", "p_ref_half")},
        **{f"oracle_{k}": o[k] for k in FIELDS + ("qtgw1", "src_level", "effgw",
                                                  "sgh_scaled", "alpha")})
    print("FIXTURE_OK", a.out, Path(a.out).stat().st_size, "bytes")
    for i, n in enumerate(d["names"]):
        print(f"  {n:30s} max|utgw|={np.abs(o['utgw'][i]).max():.3e} "
              f"max|ttgw|={np.abs(o['ttgw'][i]).max():.3e} max|qtgw|={np.abs(o['qtgw1'][i]).max():.3e}")


if __name__ == "__main__":
    main()

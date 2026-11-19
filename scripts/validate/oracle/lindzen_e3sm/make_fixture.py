"""Generate tests/atmosphere/oracle_fixtures/lindzen_e3sm.npz by RUNNING the
E3SM v3.0.1 orographic-only gravity-wave-drag oracle (E3SM-Project/E3SM tag
v3.0.1, SHA d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84: gw_prof + gw_oro_src +
gw_drag_prof with ``orographic_only = .true.``, alpha = 1e-6, and the
gw_drag.F90 discrete kinetic-energy heating), compiled unmodified by
``build.sh``; driver ``driver.F90`` (the tier-2 audit driver).

The oracle runs as the tier-2 audit ran it (oracle_audit/
simple_gwd_hines_lindzen_rayleigh_spectral, comparison arm): effgw_oro = 1,
no land-fraction scaling, fcrit2 = 1, kwv = 2*pi/1e5 (LindzenConfig.k_wave),
dt = 1800 s, legoESM g, R_d, c_pd.

Columns: the CAM6 L32 set of ``scripts/validate/oracle/gw_oro_cam6``
(``build_columns``: 11 constructed soundings incl. gate, critical-level and
limiter columns + 3 real MPAS AMIP T/q/p columns), plus three edge columns:
``rest_zero_wind`` (the audit's: real_33109 thermodynamics, u = v = 0, sgh
300 m); ``umcfac_neutral_layer`` (a dry-adiabatic 500-150 hPa layer with a
12 m/s wind, where the orographic-only path's umcfac limit binds; the CAM6
``umcfac_weak_layer`` column does not bind it here); ``zero_interface_wind``
(rockies_winter thermodynamics, v = 0, u = +10 m/s in the bottom layer and
-10 m/s above, sgh 6 m, so one interface wind is exactly zero).

    bash build.sh /path/to/E3SM_v3.0.1 /tmp/lindzen_build
    python make_fixture.py --drv /tmp/lindzen_build/gw_oro_e3sm \
        --out tests/atmosphere/oracle_fixtures/lindzen_e3sm.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._shared import compute_heights_from_sigma, compute_rho

E3SM_TAG = "v3.0.1"
E3SM_SHA = "d4ca7d0606930f53f30eb7ff7dbb92ec83b83b84"
DT = 1800.0
EFFGW_ORO = 1.0      # audit comparison arm: no efficiency scaling
FCRIT2 = 1.0
PRNDL = 0.25         # read by the driver; unused on the orographic-only path
SOUTH_FAC = 1.0
ALPHA_ORO = 1.0e-6   # gw_drag.F90:341 orographic_only alpha
FIELDS = ("tau", "utgw", "vtgw", "ttgw")
_CAM6 = Path(__file__).resolve().parents[1] / "gw_oro_cam6" / "make_fixture.py"


def build_columns():
    spec = importlib.util.spec_from_file_location("gw_oro_cam6_fixture", _CAM6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.build_columns()
    names = list(d["names"])
    rows = {k: list(d[k]) for k in ("T", "q", "u", "v", "lat", "sgh", "lf")}
    ps = list(d["p_half"][:, -1])

    def add(name, T, q, u, v, lat, sgh, lf, p_s):
        names.append(name)
        for k, val in zip(("T", "q", "u", "v", "lat", "sgh", "lf"), (T, q, u, v, lat, sgh, lf)):
            rows[k].append(val)
        ps.append(p_s)

    j = names.index("real_33109")
    add("rest_zero_wind", d["T"][j], d["q"][j], 0.0 * d["u"][j], 0.0 * d["v"][j],
        d["lat"][j], 300.0, 1.0, ps[j])
    p_half, T, q = mod.sounding(850, 285, 210, 0.5)
    p_full = np.asarray(mod.coord.pressure_at_full(np.array([p_half[-1]])))[0]
    lay = (p_full < 50000.0) & (p_full > 15000.0)
    k0 = np.where(lay)[0].max()
    T = np.where(lay, T[k0] * (p_full / p_full[k0]) ** constants.kappa, T)
    u, v = mod.winds(p_full, [(850, 8), (700, 15), (500, 12), (150, 12), (50, 25), (5, 30)],
                     [(850, 0), (5, 0)])
    add("umcfac_neutral_layer", T, q, u, v, np.deg2rad(40.0), 700.0, 1.0, p_half[-1])
    j = names.index("rockies_winter")
    u = np.full_like(d["u"][j], -10.0)
    u[-1] = 10.0
    add("zero_interface_wind", d["T"][j], d["q"][j], u, 0.0 * u, d["lat"][j], 6.0, 1.0, ps[j])

    out = {k: np.stack(rows[k]) if k in ("T", "q", "u", "v") else np.array(rows[k], float)
           for k in rows}
    out["names"] = np.array(names)
    ps = np.array(ps)
    out["p_half"] = np.asarray(mod.coord.pressure_at_half(ps))
    out["p_full"] = np.asarray(mod.coord.pressure_at_full(ps))
    z_full, z_half = compute_heights_from_sigma(out["T"], out["p_half"], q_v=out["q"])
    out["z_full"], out["z_half"] = np.asarray(z_full), np.asarray(z_half)
    out["rho"] = np.asarray(compute_rho(out["T"], out["p_full"], q_v=out["q"]))
    out["p_ref_half"] = d["p_ref_half"]
    return out


def write_input(path, d):
    f = lambda a: " ".join(f"{x:.17e}" for x in np.ravel(a))
    ncol, nlev = d["T"].shape
    with open(path, "w") as fh:
        fh.write(f"{ncol} {nlev}\n{f((constants.g, constants.R_d, constants.c_pd))}\n")
        fh.write(f"{DT!r} {EFFGW_ORO!r} {FCRIT2!r} {PRNDL!r} {SOUTH_FAC!r}\n")
        fh.write("0 0 1\n")                      # lndscl=0, vdiff=0, alpha_mode=1
        fh.write(f(d["p_ref_half"]) + "\n")
        for i in range(ncol):
            fh.write(f"{float(d['lat'][i])!r} {float(d['sgh'][i])!r} {float(d['lf'][i])!r}\n")
            for k in ("p_half", "T", "u", "v", "z_full", "q"):
                fh.write(f(d[k][i]) + "\n")
        for i in range(ncol):
            fh.write(f(d["p_full"][i]) + "\n")
        fh.write(f"{ALPHA_ORO!r}\n")


def parse_output(path):
    cols = []
    for line in open(path):
        if line.startswith("#"):
            continue
        tok = line.split()
        if tok[0] == "alpha":
            continue
        if tok[0] == "col":
            cur = dict(src_level=int(tok[2]))
            cols.append(cur)
        else:
            cur[tok[0]] = np.array(tok[1:], float)
    return {k: np.stack([c[k] for c in cols]) for k in cols[0]}


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
        if r.returncode != 0 or "E3SM_ORACLE_OK" not in r.stdout:
            raise RuntimeError(f"oracle failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
        o = parse_output(fout)
    meta = (f"oracle=E3SM-Project/E3SM {E3SM_TAG} sha={E3SM_SHA} (gw_prof+gw_oro_src+"
            f"gw_drag_prof orographic_only, alpha={ALPHA_ORO}, discrete-KE heating); "
            f"dt={DT} effgw_oro={EFFGW_ORO} lndscl=0 fcrit2={FCRIT2} "
            f"kwv=2*pi/1e5={2.0 * math.pi / 1.0e5!r}; constants legoESM g={constants.g} "
            f"R_d={constants.R_d} c_pd={constants.c_pd}; generator "
            "scripts/validate/oracle/lindzen_e3sm/{build.sh,driver.F90,make_fixture.py}")
    np.savez_compressed(
        a.out, meta=np.array(meta), dt=DT,
        **{k: d[k] for k in ("names", "T", "q", "u", "v", "lat", "sgh", "p_half",
                             "p_full", "z_full", "z_half", "rho")},
        **{f"oracle_{k}": o[k] for k in FIELDS + ("src_level",)})
    print("FIXTURE_OK", a.out, Path(a.out).stat().st_size, "bytes")
    for i, n in enumerate(d["names"]):
        print(f"  {n:30s} src={o['src_level'][i]:3d} max|utgw|={np.abs(o['utgw'][i]).max():.3e} "
              f"max|ttgw|={np.abs(o['ttgw'][i]).max():.3e} tau_sfc={o['tau'][i, -1]:.3e}")


if __name__ == "__main__":
    main()

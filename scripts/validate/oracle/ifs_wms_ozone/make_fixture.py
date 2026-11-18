"""Run the OpenIFS 49r1 WMS GWD / O3CHEM / METHOX oracle (8f6f722) and write the fixture.

usage: python make_fixture.py <columns dir> <build dir> <out.npz>

<columns dir> holds the tier-2 audit columns ``columns_{L32,L137,STRESS}.npz``
(T, q, u, v, pressure, geopotential; levels top to bottom).  Inputs those
columns lack (votes V4-V6, VOTES.md): latitude spread over -80..80 deg, cell
sizes cycling 200/29/10/3.5/1 km, standard pressures from column 1 rescaled to
101325 Pa (mean of halves), and synthetic Cariolle coefficients/ozone chosen so
the +-15 K temperature clamp and the 1e-9 kg/kg floor bind somewhere.
Arms: L = legoESM constants (gated), N = 49r1 constants (reported).
<build dir> holds ``oracle.x`` from build.sh.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants as C

GROUPS = ("L32", "L137", "STRESS")
DX_KM = (200.0, 29.0, 10.0, 3.5, 1.0)
P_STD_SFC = 101325.0          # SUSTA VP00
DT_ARMS = (450.0, 3600.0)     # 3600 s exercises SUGWWMS's TSTEP >= RHOUR branch
RDECLI = 0.2
RQLIM_ARMS = (-1.0, 5.0e-6)   # -1 = SUMETHOX default
ARMS = {"L": dict(CONST_MODE=1, CRG=C.g, CRD=C.R_d, CRCPD=C.c_pd, CRA=C.R_earth),
        "N": dict(CONST_MODE=2, CRG=0.0, CRD=0.0, CRCPD=0.0, CRA=0.0)}


def _r(x):
    """Round trip through the driver's '%.17e' text."""
    return np.vectorize(lambda v: float(f"{float(v):.17e}"))(np.asarray(x, np.float64))


def write_arr(path, a):
    a = np.asarray(a, np.float64)
    shape = a.shape if a.ndim else (1,)
    vals = a.reshape(-1, order="F") if a.ndim else a.reshape(1)
    path.write_text(f"{len(shape)} {' '.join(str(s) for s in shape)}\n"
                    + "\n".join(f"{v:.17e}" for v in vals) + "\n")


def read_arr(path):
    toks = path.read_text().split()
    rank = int(toks[0])
    shape = tuple(int(t) for t in toks[1:1 + rank])
    vals = np.array([float(t) for t in toks[1 + rank:]])
    return vals.reshape(shape, order="F")


def column_inputs(d):
    """Shared per-group inputs, rounded through the driver text."""
    ncol, nlev = d["T"].shape
    lat_deg = np.linspace(-80.0, 80.0, ncol)
    dx = np.array([DX_KM[i % len(DX_KM)] * 1e3 for i in range(ncol)])
    ph0 = np.asarray(d["p_half"][0], np.float64)
    phs = ph0 * (P_STD_SFC / ph0[-1])
    stpre = 0.5 * (phs[:-1] + phs[1:])
    p_full = np.asarray(d["p_full"], np.float64)
    p_half = np.asarray(d["p_half"], np.float64)
    return dict(
        T=_r(d["T"]), u=_r(d["u"]), v=_r(d["v"]), q=_r(d["q"]),
        p_full=_r(p_full), p_half=_r(p_half), geo_full=_r(d["geo_full"]),
        lat=_r(np.deg2rad(lat_deg)), dx=dx, stpre=_r(stpre),
        names=np.array([str(n) for n in d["names"]]))


def ozone_inputs(x, group):
    """Synthetic ozone state and Cariolle coefficients (vote V6a)."""
    p, T, ph = x["p_full"], x["T"], x["p_half"]
    ncol, nlev = p.shape
    lnp = np.log(p / 1000.0)                       # 0 at 10 hPa
    o3 = 1.6e-5 * np.exp(-lnp ** 2 / 2.0) + 3.0e-8
    if group == "STRESS":
        o3 = np.full_like(o3, 2.0e-10)             # floor binds where production < 0
    k = np.zeros((ncol, nlev, 8))
    k[..., 0] = o3 * (1.0 + 0.05 * np.sin(np.arange(nlev) / 3.0))          # r_clim
    k[..., 1] = 2.0e-12 * np.exp(-lnp ** 2) * np.cos(np.arange(ncol)[:, None] + lnp)  # (P-L)_clim
    k[..., 2] = -1.0 / (86400.0 * np.exp(np.clip(lnp + 2.0, 0.0, 7.0)))    # d(P-L)/dr < 0
    k[..., 3] = T + 25.0 * np.sin(np.arange(nlev) / 2.0 + np.arange(ncol)[:, None])  # T_clim
    k[..., 4] = -3.0e-13 * np.exp(-lnp ** 2 / 4.0)                          # d(P-L)/dT
    col = np.concatenate([np.zeros((ncol, 1)),
                          np.cumsum(np.diff(ph, axis=1) * k[..., 0] / C.g, axis=1)[:, :-1]], axis=1)
    k[..., 5] = col * 1.02                                                  # Sigma_clim
    k[..., 6] = -2.0e-9 * np.exp(-lnp ** 2 / 2.0)                           # d(P-L)/dSigma
    k[..., 7] = 1.0e-9                                                      # unused at HLO
    lat = x["lat"]
    return dict(PO3=_r(o3), PKOZO=_r(k), PDELP=_r(np.diff(ph, axis=1)),
                PGEMU=_r(np.sin(lat)), PMU0=_r(np.cos(lat + 0.3) - 0.2))


def methox_q(x, group):
    q = np.asarray(x["q"], np.float64).copy()
    if group == "STRESS":   # both sides of RQLIM above 100 hPa
        strat = x["p_full"] < 1.0e4
        q[strat] = (4.81e-6 * (0.5 + np.linspace(0.0, 1.0, q.size).reshape(q.shape)))[strat]
    return _r(q)


def run(exe, build, tag, mode, arm, inputs, extra):
    din = Path(build) / f"in_{tag}"
    dout = Path(build) / f"out_{tag}"
    din.mkdir(parents=True, exist_ok=True)
    dout.mkdir(parents=True, exist_ok=True)
    for k, v in inputs.items():
        write_arr(din / f"{k}.txt", v)
    nml = Path(build) / f"{tag}.nml"
    body = ", ".join(f"{k}={v!r}" for k, v in {**ARMS[arm], **extra}.items())
    nml.write_text(f"&HARNESS MODE='{mode}', {body} /\n&NAMSCEN /\n&NAMGWWMS /\n&NAMMETHOX /\n")
    r = subprocess.run([str(exe), str(din), str(dout), str(nml)], capture_output=True, text=True)
    if r.returncode != 0 or "DRIVER_DONE" not in r.stdout:
        sys.exit(f"oracle failed {tag}: {r.stdout[-500:]} {r.stderr[-500:]}")
    return {p.stem: read_arr(p) for p in dout.glob("*.txt")}, r.stdout


def main(cdir, build, out):
    exe = Path(build) / "oracle.x"
    fx = {"dt_arms": np.array(DT_ARMS), "rdecli": np.float64(RDECLI),
          "rqlim_arms": np.array(RQLIM_ARMS), "dx_km": np.array(DX_KM)}
    for g in GROUPS:
        x = column_inputs(np.load(Path(cdir) / f"columns_{g}.npz", allow_pickle=True))
        ncol, nlev = x["T"].shape
        oz = ozone_inputs(x, g)
        qm = methox_q(x, g)
        for k, v in {**x, **oz, "q_methox": qm}.items():
            fx[f"{g}_{k}"] = v
        for arm in ARMS:
            pgaw = _r((x["dx"] / (2.0 * C.R_earth)) ** 2 / np.pi)   # same PGAW on both arms
            fx[f"{arm}_{g}_pgaw"] = pgaw
            base = dict(KLON=float(ncol), KLEV=float(nlev))
            for dt in DT_ARMS:
                t = f"{arm}_{g}_wms_dt{int(dt)}"
                res, log = run(exe, build, t, "WMS", arm, dict(
                    base, STPRE=x["stpre"], PT=x["T"], PU=x["u"], PV=x["v"], PAP=x["p_full"],
                    PAPH=x["p_half"], PGEO=x["geo_full"], PGELAT=x["lat"], PGAW=pgaw,
                    PPRECIP=np.zeros(ncol)), dict(TSTEP=dt))
                for k, v in res.items():
                    fx[f"{t}_{k}"] = v
                fx[f"{arm}_const_line"] = np.array(
                    [ln for ln in log.splitlines() if "CONST_USED" in ln][0])
                t = f"{arm}_{g}_o3_dt{int(dt)}"
                res, log = run(exe, build, t, "O3CHEM", arm, dict(
                    base, PAPH=x["p_half"], PAP=x["p_full"], PT=x["T"], **oz),
                    dict(TSTEP=dt, RDECLI=RDECLI))
                for k, v in res.items():
                    fx[f"{t}_{k}"] = v
            for i, rq in enumerate(RQLIM_ARMS):
                t = f"{arm}_{g}_methox_rq{i}"
                res, log = run(exe, build, t, "METHOX", arm, dict(base, PQ=qm, PAP=x["p_full"]),
                               dict(TSTEP=450.0, RQLIM_OVR=rq))
                for k, v in res.items():
                    fx[f"{t}_{k}"] = v
    np.savez(out, **fx)
    print("FIXTURE_OK", out, len(fx))


if __name__ == "__main__":
    main(*sys.argv[1:4])

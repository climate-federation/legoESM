"""Generate tests/atmosphere/oracle_fixtures/clubb_mf.npz by RUNNING CAM's
CLUBB-MF plume ensemble (``integrate_mf``, atmospheric_physics
atmos_phys0_34_002 cec8c93ed2) in the reduced setting (aloft off, no TKE
entrainment boost, precip off; user 2026-11-19 + votes V1-V24).

usage:
  bash build.sh <atmos_phys> <CESM_share> <build>
  python make_fixture.py --columns <columns.npz> --driver <build>/clubb_mf_driver \
      --out tests/atmosphere/oracle_fixtures/clubb_mf.npz

``--columns``: the tier-2 audit set (15 L32 columns: 3 real MPAS AMIP + 12
constructed soundings; sha256 in ``meta``) plus one stable column (column 0
with the surface 2 K COLDER) for the trigger-off branch (V11).  Our EDMF runs
first on each column as the tier-2 audit ran it (T_sfc = T_lowest + 2 K,
q_sfc = q_sat(T_sfc), dt 600 s), and its own surface fluxes, u* and PBL height
are CAM's surface inputs, so the score is one variable: the plume model
(V12).  CLUBB-grid inputs: zm = our half levels, zt = our full levels
(ascending); zm fields linear in height from zt, surface zm = lowest zt (V14);
qc = 0; lts by CAM's host formula (clubb.F90:1001-1007, V13).  Four synthetic
columns follow (V31): CAM inputs of one triggered column overridden to reach
the trigger-off (moisture flux, stability), PBL-floor and model-top branches.  Constants are
legoESM's on both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    compute_rho,
    exner_function,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulentEDMFConfig
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
)
from legoesm.thermo import saturation_specific_humidity

from legoesm import constants as C

jax.config.update("jax_enable_x64", True)

DT = 600.0
NUP = 25
ORACLE = dict(
    repo="ESCOMP/atmospheric_physics",
    tag="atmos_phys0_34_002",
    sha="cec8c93ed22e6dd60bd676459d4fc0468826fe7d",
    file="schemes/clubb/clubb_mf.F90",
    cam="ESCOMP/CAM cam_development 417212dff093130bcb7606221f6d2ca4adbfbc98",
    rng="ESCOMP/CESM_share share1.1.22 f256faa25edb103f09f3c372d021ea6e1a633516",
)
# clubb_mf_init: Lopt up_ndt cp_ndt kseed | a0 b0 L0 ent0 alphturb max_L0 fdd ddalph ddbeta pwfac ddexp
# pwmin pwmax cldfrac_fac (pad) | do_clubb_mf diag precip rad coldpool coldpool_init coldpool_perplume
# lscale_perplume ustar mixd rhtke cmt addtke aloft aspd pblcull invswitch.  cam7 namelist defaults,
# except alphturb = 0 and aloft = F (user), precip = F (V1), Lopt = 0 (V2b: from CAM's cold
# start, Lopt 6 clamps L0 to 0.5 m and every plume on every column is culled -> vacuous fixture).
INTS = [0, 1, 1, 1]
REALS = [0.15, 1.0, 200.0, 0.2, 0.0, 1.0e3, 0.5, 12.0, 1.0, 1.0, 3.0, 1.5, 3.0, 10.0, 0.0]
FLAGS = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def edmf_inputs(c, dT_sfc):
    T, q, pf, ph = (jnp.asarray(c[k]) for k in ("T", "q", "p_full", "p_half"))
    z_full, z_half = compute_heights_from_sigma(T, ph, q_v=q)
    T_sfc = T[:, -1] + dT_sfc
    return dict(
        u=jnp.asarray(c["u"]),
        v=jnp.asarray(c["v"]),
        T=T,
        q_v=q,
        tke=jnp.where(z_full < 1500.0, 0.5, 0.01),
        p_full=pf,
        p_half=ph,
        z_full=z_full,
        z_half=z_half,
        T_sfc=T_sfc,
        q_sfc=saturation_specific_humidity(T_sfc, pf[:, -1]),
        rho=compute_rho(T, pf, q_v=q),
    )


def zt2zm(f_zt, zt, zm):
    """Linear in height between zt neighbours; surface zm = lowest zt (V14)."""
    return np.concatenate([f_zt[:1], np.interp(zm[1:], zt, f_zt)])


def clubb_inputs(x, o, cfg):
    """Our EDMF's column (top-first) -> CAM's ascending CLUBB grid + surface inputs."""
    zvir = C.R_v / C.R_d - 1.0
    rev = lambda a: np.asarray(a)[0, ::-1]  # noqa: E731
    zt, zm = rev(x["z_full"]), rev(x["z_half"])
    pz, T, q, rho = rev(x["p_full"]), rev(x["T"]), rev(x["q_v"]), rev(x["rho"])
    assert abs(zm[0]) < 1e-9
    iex = 1.0 / np.asarray(exner_function(jnp.asarray(pz)))
    thl = T * iex
    p_zm = zt2zm(pz, zt, zm)
    d = dict(
        zt=zt,
        dzt=np.diff(zm),
        rho_zt=rho,
        p_zt=pz,
        iex_zt=iex,
        u=rev(x["u"]),
        v=rev(x["v"]),
        thl=thl,
        qt=q,
        thv=thl * (1.0 + zvir * q),
        th=thl,
        qv=q,
        qc=np.zeros_like(q),
        zm=zm,
        rho_zm=zt2zm(rho, zt, zm),
        p_zm=p_zm,
        iex_zm=1.0 / np.asarray(exner_function(jnp.asarray(p_zm))),
        w=np.zeros_like(zm),
    )
    for k in ("thl", "qt", "thv", "th", "qv", "qc"):
        d[f"{k}_zm"] = zt2zm(d[k], zt, zm)
    d["tke"] = zt2zm(rev(x["tke"]), zt, zm)  # CAM input; inert at alphturb = 0
    for k in ("wpthlp", "wpthvp", "wpqtp"):
        d[k] = np.zeros_like(zm)  # CLUBB moments; inert with aloft off, Lopt 6
    sflx_q = surface_moisture_flux(cfg.surface, o.lhflx, x["T_sfc"])
    sflx_T = (o.shflx + latent_enthalpy_correction(o.lhflx, sflx_q)) / C.c_pd
    rho1 = x["rho"][:, -1]
    ex1 = exner_function(x["p_full"][:, -1])
    # lts: CAM host formula, surface-referenced exner (pint(pverp)/pmid)**cappa, mf_pinterp = np.interp in p
    ps = np.asarray(x["p_half"])[:, -1:]
    thlcam = np.asarray(x["T"]) * (ps / np.asarray(x["p_full"])) ** C.kappa
    lts = np.array(
        [
            np.interp(7.0e4, pf, th_) - np.interp(1.0e5, pf, th_)
            for pf, th_ in zip(np.asarray(x["p_full"]), thlcam)
        ]
    )
    s = dict(
        ustar=np.asarray(o.ustar),
        ths=np.asarray(x["T_sfc"]) * d["iex_zm"][0],
        wthl_sfc=np.asarray(sflx_T / (rho1 * ex1)),
        wqt_sfc=np.asarray(sflx_q / rho1),
        pblh=np.asarray(o.h_pbl) - np.asarray(x["z_half"])[:, -1],
        tpert=np.zeros(rho1.shape),
        lts=lts,
    )
    return d, s


ZT_KEYS = ("zt", "dzt", "rho_zt", "p_zt", "iex_zt", "u", "v", "thl", "qt", "thv", "th", "qv", "qc")
ZM_KEYS = (
    "zm",
    "rho_zm",
    "p_zm",
    "iex_zm",
    "w",
    "thl_zm",
    "qt_zm",
    "thv_zm",
    "th_zm",
    "qv_zm",
    "qc_zm",
    "tke",
    "wpthlp",
    "wpthvp",
    "wpqtp",
)
S_KEYS = ("ustar", "ths", "wthl_sfc", "wqt_sfc", "pblh", "tpert", "lts")
OUT2 = ("upa", "upw", "upthl", "upqt", "upthv", "upqc", "upth", "upent", "updet")
OUT1 = ("thlflx", "qtflx", "uflx", "vflx", "thvflx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = np.load(a.columns, allow_pickle=True)
    c = {k: np.asarray(raw[k]) for k in ("T", "q", "u", "v", "p_full", "p_half")}
    c["q"] = np.maximum(c["q"], 0.0)
    names = [str(n) for n in raw["names"]] + ["stable_" + str(raw["names"][0])]
    c = {k: np.concatenate([v, v[:1]]) for k, v in c.items()}
    dT = np.full(len(names), 2.0)
    dT[-1] = -2.0
    cfg = TurbulentEDMFConfig()
    x = edmf_inputs(c, jnp.asarray(dT))
    o, _ = edmf_turbulence(
        x["u"],
        x["v"],
        x["T"],
        x["q_v"],
        x["tke"],
        x["p_full"],
        x["p_half"],
        x["z_full"],
        x["z_half"],
        x["T_sfc"],
        x["q_sfc"],
        x["rho"],
        DT,
        cfg,
    )
    ncol = len(names)
    cols = []
    for i in range(ncol):
        xi = {k: v[i : i + 1] for k, v in x.items()}
        oi = SimpleNamespace(
            **{k: getattr(o, k)[i : i + 1] for k in ("shflx", "lhflx", "ustar", "h_pbl")}
        )
        d, s = clubb_inputs(xi, oi, cfg)
        cols.append((d, {k: float(np.ravel(v)[0]) for k, v in s.items()}))
    # V31: synthetic coverage columns for CAM branches no audit column reaches.  CAM inputs of
    # a triggered cloudy column overridden; not our EDMF's columns -> excluded from its score.
    n_audit = ncol
    d0, s0 = cols[names.index("monsoon_land")]
    zvir = C.R_v / C.R_d - 1.0
    for tag, s_over in (
        ("wqt_neg", dict(wqt_sfc=-1.0e-5)),
        ("lts25", dict(lts=25.0)),
        ("pblh50", dict(pblh=50.0)),
        ("model_top", {}),
    ):
        d = dict(d0)
        if (
            tag == "model_top"
        ):  # environment thl = surface-level thl everywhere: plumes reach the top
            d["thl"] = d["th"] = np.full_like(d0["thl"], d0["thl"][0])
            d["thv"] = d["thl"] * (1.0 + zvir * d["qt"])
            for k in ("thl", "thv", "th"):
                d[f"{k}_zm"] = zt2zm(d[k], d["zt"], d["zm"])
        cols.append((d, dict(s0, **s_over)))
        names.append(f"synthetic_{tag}_monsoon_land")
    ncol = len(cols)
    nzm = cols[0][0]["zm"].size
    rep = lambda v: " ".join(repr(float(t)) for t in np.ravel(v)) + "\n"  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        with open(fin, "w") as f:
            f.write(f"{ncol} {nzm} {NUP}\n")
            f.write(
                rep(
                    [
                        DT,
                        C.c_pd,
                        C.g,
                        C.L_v,
                        C.L_f,
                        C.R_v / C.R_d - 1.0,
                        C.T_triple_cam,
                        np.pi,
                        C.epsilon,
                        C.R_v,
                        C.T_freeze,
                    ]
                )
            )
            f.write(" ".join(map(str, INTS)) + "\n" + rep(REALS) + " ".join(map(str, FLAGS)) + "\n")
            for d, s in cols:
                for k in ZT_KEYS + ZM_KEYS:
                    f.write(rep(d[k]))
                f.write(rep([s[k] for k in S_KEYS]))
        r = subprocess.run([a.driver, str(fin), str(fout)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        tok = fout.read_text().split()
    assert tok[-1] == "DRIVER_OK"
    pos = 3
    nzt = nzm - 1

    def take(n, cast=float):
        nonlocal pos
        v = np.array([cast(t) for t in tok[pos : pos + n]])
        pos += n
        return v

    res = {
        f"ora{s_}_{k}": []
        for s_ in (1, 2)
        for k in OUT2
        + OUT1
        + (
            "dL0",
            "ztopm1",
            "ddcp",
            "cbm1",
            "sqt",
            "sthl",
            "enti",
            "ztopm1_in",
            "ddcp_in",
            "cbm1_in",
        )
    }
    for _ in range(ncol):
        for s_ in (1, 2):
            res[f"ora{s_}_ztopm1_in"].append(take(NUP))
            res[f"ora{s_}_ddcp_in"].append(take(NUP))
            res[f"ora{s_}_cbm1_in"].append(take(1)[0])
            for k in OUT2:
                res[f"ora{s_}_{k}"].append(take(nzm * NUP).reshape(NUP, nzm).T)
            for k in OUT1:
                res[f"ora{s_}_{k}"].append(take(nzm))
            for k, n in (
                ("dL0", NUP),
                ("ztopm1", NUP),
                ("ddcp", NUP),
                ("cbm1", 1),
                ("sqt", nzt),
                ("sthl", nzt),
            ):
                v = take(n)
                res[f"ora{s_}_{k}"].append(v[0] if k == "cbm1" else v)
            res[f"ora{s_}_enti"].append(take(nzt * NUP, int).reshape(NUP, nzt).T)
    assert pos == len(tok) - 1
    out = {k: np.asarray(v) for k, v in res.items()}
    # V24: precip off -> no downdrafts -> CAM's cold-pool memory is 0 in and out
    assert (
        not np.any(out["ora1_ddcp"])
        and not np.any(out["ora2_ddcp"])
        and not np.any(out["ora2_ddcp_in"])
    )
    assert not np.any(out["ora1_sqt"]) and not np.any(out["ora2_sqt"])
    for k in ZT_KEYS + ZM_KEYS:
        out[f"c_{k}"] = np.stack([d[k] for d, _ in cols])
    for k in S_KEYS:
        out[f"c_{k}"] = np.array([s[k] for _, s in cols])
    for k in (
        "u",
        "v",
        "T",
        "q_v",
        "tke",
        "p_full",
        "p_half",
        "z_full",
        "z_half",
        "T_sfc",
        "q_sfc",
        "rho",
    ):
        out[f"in_{k}"] = np.asarray(x[k])
    meta = dict(
        oracle=ORACLE,
        names=names,
        n_audit=n_audit,
        dt=DT,
        nup=NUP,
        init_ints=INTS,
        init_reals=REALS,
        init_flags=FLAGS,
        columns_sha256=hashlib.sha256(Path(a.columns).read_bytes()).hexdigest(),
        build={
            f: hashlib.sha256((Path(a.driver).parent / f).read_bytes()).hexdigest()
            for f in ("clubb_mf_driver", "clubb_mf.F90")
        },
        driver_stderr=r.stderr[-2000:],
    )
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print("wrote", a.out, {k: v.shape for k, v in out.items() if k.startswith("ora1_")})


if __name__ == "__main__":
    main()

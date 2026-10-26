"""Generate tests/atmosphere/oracle_fixtures/ysu.npz by RUNNING the WRF YSU
oracle (WRF v4.8.0 phys/physics_mmm/bl_ysu.F90::bl_ysu_run, clear-sky path:
topdown off, qc=qi=0, no BEP, ctopo absent).

usage:
  bash build.sh <WRF> <build>
  python make_fixture.py --inputs <columns.npz> --driver <build>/ysu_driver \
      --out tests/atmosphere/oracle_fixtures/ysu.npz

``--inputs`` is the tier-2 audit column set (oracle_audit/ysu_pbl/inputs.npz:
20 L32 columns, real MPAS checkpoint columns + constructed soundings, with
the audit's constructed surface-layer state; sha256 recorded in ``meta``).
Two more columns are appended: copies of the stable land columns
``polar_dry`` and ``nocturnal_elevated`` with ``xland = 2`` (water), so the
fixture reaches the oracle's Rossby-number critical-Ri branch over water.
A third appended column, ``tibet_deep_mixed_noflux``, is ``tibet_deep_mixed``
with zero surface heat and water flux, ``br = 0`` and an added zonal wind
shear of 0.5 m/s per level (so the free-atmosphere K differs from the in-PBL K
at the tie interface): its 13-level constant-
theta_v layer then ties the first-guess bulk Ri with ``brcr = 0`` exactly, so
the PBL top lands exactly on level ``kpbl-1`` (``brint = 0``).
A separate 8-level column (``up_*`` keys, see ``upper_endpoint_column``)
puts the PBL top exactly on level ``kpbl`` by rounding.
Constants handed to the oracle are legoESM's (``legoesm.constants``), as
``bl_ysu_run`` (the JAX port) uses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants as C

ORACLE = dict(repo="wrf-model/WRF", tag="v4.8.0",
              sha="06d4240ae989cc3e50af412bb472df3d9048783c",
              physics_mmm_sha="550b5b494758ad2756394e93cb08b5709b8feb83",
              files=["phys/physics_mmm/bl_ysu.F90", "phys/ccpp_kind_types.F"])
LEV = ("u", "v", "T", "q", "p_full", "exner", "dz")
SFC = ("ust", "hfx", "qfx", "wspd", "br", "psim", "psih", "znt", "xland", "u10", "v10")
WATER_COPIES = ("polar_dry", "nocturnal_elevated")
TIE_COPY = "tibet_deep_mixed"


def constants_vector(dt):
    # cp g rovcp rd rovg ep1 ep2 karman xlv rv dt (ysu_driver.F90 order)
    return [C.c_pd, C.g, C.R_d / C.c_pd, C.R_d, C.R_d / C.g, C.R_v / C.R_d - 1.0,
            C.R_d / C.R_v, C.kappa_vk, C.L_v, C.R_v, dt]


def columns(src):
    names = [str(n) for n in src["names"]]
    d = dict(u=src["u"], v=src["v"], T=src["T"], q=src["q"], p_full=src["p_full"],
             p_half=src["p_half"], zi=src["zi"], ust=src["ust"], hfx=src["shflx"],
             qfx=src["qflx"], wspd=src["wspd"], br=src["br"],
             psim=src["psim"], psih=src["psih"],     # audit stores fm, fh under these names
             znt=src["z0"], xland=src["xland"], u10=src["u10"], v10=src["v10"],
             taux=src["taux"], tauy=src["tauy"], lhflx=src["lhflx"])
    d = {k: np.asarray(v, float) for k, v in d.items()}
    idx = [names.index(n) for n in WATER_COPIES]
    d = {k: np.concatenate([v, v[idx]], axis=0) for k, v in d.items()}
    d["xland"][-len(idx):] = 2.0
    names = names + [f"{n}_water" for n in WATER_COPIES]
    t = names.index(TIE_COPY)
    d = {k: np.concatenate([v, v[t:t + 1]], axis=0) for k, v in d.items()}
    for k in ("hfx", "qfx", "lhflx", "br"):
        d[k][-1] = 0.0
    d["u"][-1] = d["u"][-1] + 0.5 * np.arange(d["u"].shape[1])[::-1]   # shear: 0.5 m/s per level
    names = names + [f"{TIE_COPY}_noflux"]
    d["exner"] = (d["p_full"] / C.p_ref) ** C.kappa
    d["dz"] = d["zi"][:, :-1] - d["zi"][:, 1:]
    return names, d


def upper_endpoint_column():
    """One 8-level column (codex review): theta = [300, 290, 300+ulp, 310, 320,
    330, 340, 350] K bottom-up, 100 m layers (8 levels: the oracle's get_pblh,
    always called, indexes past a column that ends below 500 m), no moisture or surface flux, br = 0.  The first
    crossing at level 3 has brup ~ 1e-14, so brint rounds to 1 and hpbl lands
    exactly on za(kpbl) = 250 m; that interface is above the PBL in the oracle.
    The level-3 temperature is the smallest one whose T/pi (computed as the
    oracle does) exceeds level 1's."""
    p_half = np.linspace(90400.0, 100000.0, 9)                         # top-first
    p_full = 0.5 * (p_half[:-1] + p_half[1:])
    exner = (p_full / C.p_ref) ** C.kappa
    th = np.array([350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 290.0, 300.0])  # top-first
    T = th * exner
    th1 = T[7] / exner[7]
    t3 = th1 * exner[5]
    while not t3 / exner[5] > th1:
        t3 = np.nextafter(t3, np.inf)
    while np.nextafter(t3, -np.inf) / exner[5] > th1:
        t3 = np.nextafter(t3, -np.inf)
    T[5] = t3
    u = np.full(8, 5.0)
    u[5] = 100.0
    d = dict(u=u, v=np.zeros(8), T=T, q=np.zeros(8),
             p_full=p_full, p_half=p_half, exner=exner, dz=np.full(8, 100.0),
             zi=np.arange(800.0, -1.0, -100.0), ust=0.3, hfx=0.0, qfx=0.0,
             wspd=5.0, br=0.0, psim=np.log(50.0 / 0.1), psih=np.log(50.0 / 0.1), znt=0.1,
             xland=1.0, u10=5.0 * np.log(10.0 / 0.1) / np.log(50.0 / 0.1), v10=0.0)
    return {k: np.atleast_1d(np.asarray(v, float))[None, :] if np.ndim(v) else
            np.array([float(v)]) for k, v in d.items()}


def write_inputs(path, d, dt):
    def wr(f, a):
        f.write(" ".join(repr(float(x)) for x in np.ravel(a)) + "\n")
    ncol, nlev = d["T"].shape
    with open(path, "w") as f:
        f.write(f"{ncol} {nlev}\n")
        wr(f, constants_vector(dt))
        for c in range(ncol):
            for k in LEV:
                wr(f, d[k][c][::-1])
            wr(f, d["p_half"][c][::-1])
            wr(f, [d["p_half"][c, -1], d["znt"][c], d["ust"][c], d["hfx"][c], d["qfx"][c],
                   d["wspd"][c], d["br"][c], d["psim"][c], d["psih"][c], d["xland"][c],
                   d["u10"][c], d["v10"][c]])


def parse_outputs(path, ncol):
    lines = open(path).read().split("\n")
    assert lines[0].startswith("YSU") and "bl_ysu_run OK" in lines[0], lines[0]
    o = {k: [] for k in ("kpbl", "hpbl", "wstar", "delta", "du", "dv", "dT", "dq",
                         "exch_h", "exch_m")}
    i = 1
    for c in range(ncol):
        head = lines[i].split()
        assert head[:2] == ["COL", str(c + 1)], lines[i]
        o["kpbl"].append(int(head[2])); i += 1
        s = np.array(lines[i].split(), float); i += 1
        o["hpbl"].append(s[0]); o["wstar"].append(s[1]); o["delta"].append(s[2])
        for k in ("du", "dv", "dT", "dq", "exch_h", "exch_m"):
            o[k].append(np.array(lines[i].split(), float)[::-1]); i += 1   # -> top-first
    return {f"ora_{k}": np.array(v) for k, v in o.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = np.load(a.inputs)
    dt = float(src["dt"])
    names, d = columns(src)
    ncol = d["T"].shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.txt", Path(tmp) / "out.txt"
        write_inputs(fin, d, dt)
        r = subprocess.run([a.driver, str(fin), str(fout), "0"], capture_output=True, text=True)
        print(r.stdout, r.stderr)
        assert r.returncode == 0, r
        ora = parse_outputs(fout, ncol)
        up = upper_endpoint_column()
        write_inputs(fin, up, dt)
        r = subprocess.run([a.driver, str(fin), str(fout), "0"], capture_output=True, text=True)
        assert r.returncode == 0, r
        ora.update({f"up_{k}": v for k, v in parse_outputs(fout, 1).items()})
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.split("\n")[0]
    meta = dict(oracle=ORACLE, compiler=gfortran + " -O2 -DDOUBLE_PRECISION -fdefault-real-8",
                constants=dict(zip(("cp", "g", "rovcp", "rd", "rovg", "ep1", "ep2", "karman",
                                    "xlv", "rv", "dt"), constants_vector(dt))),
                inputs_sha256=hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(),
                names=names, water_copies=list(WATER_COPIES), tie_copy=TIE_COPY,
                setup="ysu_topdown_pblmix=F, f_qc=f_qi=F, nmix=0, no BEP, ctopo/uox absent",
                layout="arrays top-first (index 0 = model top); exch_* entry k = K of the "
                       "interface below level k (oracle exch(k+1)=xkz(k)); kpbl 1-based "
                       "from the surface")
    np.savez_compressed(a.out, meta=json.dumps(meta), dt=dt,
                        **{f"in_{k}": v for k, v in d.items()},
                        **{f"up_in_{k}": v for k, v in up.items()}, **ora)
    print("WROTE", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()

"""Generate tests/atmosphere/oracle_fixtures/louis.npz by RUNNING the Louis
oracles: OpenIFS vdfexcu.F90 (stable interfaces) and the ECHAM/ICON "Louis 79"
block of mo_turb_vdiff_diag.f90:518-530 (unstable interfaces).

usage:
  bash build.sh <openifs> <ICON> <build>
  python make_fixture.py --inputs <columns.npz> --build <build> \
      --out tests/atmosphere/oracle_fixtures/louis.npz

``--inputs`` is the tier-2 audit column set (26 L32 columns: the 21
Holtslag-Boville audit columns -- real MPAS checkpoint columns + constructed
soundings -- plus 5 constructed convective / near-neutral columns; CAM-FV
geopotential heights; prescribed surface fluxes; built by the audit's
make_inputs.py, sha256 recorded in ``meta``).

What each oracle receives (the tier-2 audit harness, unchanged):
  * IFS arm (LouisConfig b=5 c=5 d=1 b_heat_ratio=1 l_mix_max=30, hard switch):
    the column u, v, T, q, half-level pressure and geopotentials, and as PRI the
    gradient Richardson number legoESM's ``louis_turbulence`` computes (the Ri
    DEFINITION is not part of the oracle comparison).  vdfexcu computes the
    velocity-difference floor, the shear, the stable mixing length and the
    stability functions itself.  Harness settings: PKHVFL=0, KINV=KLEV,
    PZ0MM=0, PLAM=150 (unused on stable interfaces).
  * ICON arm (b=c=5, d=5, b_heat_ratio=1.5, hard switch): per interface, Ri,
    legoESM's mixing length, the two full-level heights and the neutral
    K = l^2 |dV/dz| (= K_m / f_m of legoESM); the fragment applies the Louis-79
    unstable multiplier with cb = cc = 5 (mo_turb_vdiff_params.f90:35-36).
Constants handed to IFS are legoESM's (they feed only disabled IFS branches).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics._shared import compute_rho  # noqa: E402
from legoesm.atmosphere.physics.turbulence import louis as L  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import LouisConfig  # noqa: E402

HARD = 1.0e12
ARMS = {
    "ifs": dict(b_louis=5.0, c_louis=5.0, d_louis=1.0, b_heat_ratio=1.0,
                l_mix_max=30.0, blend_ri_sharpness=HARD, shear_floor_sharpness=float("inf")),
    "icon": dict(b_louis=5.0, c_louis=5.0, d_louis=5.0, b_heat_ratio=1.5,
                 blend_ri_sharpness=HARD, shear_floor_sharpness=float("inf")),
}
ICON_CB = ICON_CC = 5.0  # mo_turb_vdiff_params.f90:35-36
IN_FIELDS = ("T", "q", "u", "v", "p_full", "p_half", "zm", "zi",
             "taux", "tauy", "shflx", "lhflx")
ORACLE = {
    "ifs": dict(repo="ecmwf-ifs/openifs", tag="openifs-48r1.1_20260825-7-g8f6f722",
                sha="8f6f722f732b92fed2c39af763d8fcd1ab15f573",
                files=["ifs-source/arpifs/phys_ec/vdfexcu.F90",
                       "ifs-source/arpifs/module/parphy.F90",
                       "ifs-source/arpifs/module/yoevdfs.F90",
                       "ifs-source/arpifs/function/fcvdfs.func.h"]),
    "icon": dict(repo="icon-model/icon (public)", tag="icon-2026.04-public",
                 sha="8597da45ef4b86323f3fb844caedc4ae5e1ffc01",
                 files=["src/atm_phy_schemes/mo_turb_vdiff_diag.f90:518-530",
                        "src/atm_phy_schemes/mo_turb_vdiff_params.f90:35-36"]),
}


def run_ours(d, arm):
    """louis_turbulence on the columns; capture Ri, l (stability-function call)
    and K_m, K_h (the u and q solver calls)."""
    cap = {"stab": [], "K": []}
    orig_stab, orig_ivd = L.louis_stability_functions, L.implicit_vertical_diffusion

    def stab(*a, **k):
        cap["stab"].append(a)
        out = orig_stab(*a, **k)
        cap["fm"] = out[0]
        return out

    def ivd(phi, K, *a):
        cap["K"].append(K)
        return orig_ivd(phi, K, *a)

    L.louis_stability_functions, L.implicit_vertical_diffusion = stab, ivd
    try:
        f = {k: jnp.asarray(d[k]) for k in IN_FIELDS}
        rho = compute_rho(f["T"], f["p_full"], q_v=f["q"])
        ustar = jnp.sqrt(jnp.hypot(f["taux"], f["tauy"]) / rho[:, -1])
        L.louis_turbulence(
            f["u"], f["v"], f["T"], f["q"], f["p_full"], f["p_half"], f["zm"], f["zi"],
            f["T"][:, -1], f["q"][:, -1], rho, float(d["dt"]),
            LouisConfig(**ARMS[arm]),
            surface_flux=(f["taux"], f["tauy"], f["shflx"], f["lhflx"], ustar))
    finally:
        L.louis_stability_functions, L.implicit_vertical_diffusion = orig_stab, orig_ivd
    assert len(cap["stab"]) == 1 and len(cap["K"]) == 3  # u, v, q (theta via its own wrapper)
    Ri, l_mix = (np.asarray(x) for x in cap["stab"][0][:2])
    Km, Kh = np.asarray(cap["K"][0]), np.asarray(cap["K"][2])
    return Ri, l_mix, Km, Kh, np.asarray(cap["fm"]), np.asarray(ustar)


def wr(f, a):
    f.write(" ".join(repr(float(x)) for x in np.ravel(a)) + "\n")


def read_fixed(path, nfield):
    """Fortran es25.17e3 records: fixed 25-character fields (a negative number
    has no separator)."""
    rows = []
    for line in open(path).read().splitlines():
        if not line.strip():
            continue
        assert len(line) == 25 * nfield, line
        rows.append([float(line[i:i + 25]) for i in range(0, len(line), 25)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = np.load(a.inputs)
    d = {k: np.asarray(src[k], float) for k in IN_FIELDS}
    d["dt"] = float(src["dt"])
    ncol, nlev = d["T"].shape
    B = Path(a.build)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        # ---------------- IFS arm (stable interfaces)
        Ri, l_mix, Km, Kh, _, ustar = run_ours(d, "ifs")
        fin, fout = Path(tmp) / "ifs_in.txt", Path(tmp) / "ifs_out.txt"
        with open(fin, "w") as f:
            f.write(f"{ncol} {nlev}\n")
            wr(f, (C.g, C.R_d, C.c_pd, C.R_v / C.R_d - 1.0, C.L_v))
            for i in range(ncol):
                for arr in (d["u"][i], d["v"][i], d["T"][i], d["q"][i], d["p_half"][i],
                            C.g * d["zm"][i], C.g * d["zi"][i], np.append(Ri[i], 0.0),
                            [ustar[i] ** 2]):
                    wr(f, arr)
        r = subprocess.run([str(B / "ifs_exe"), str(fin), str(fout)],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr)
        assert r.returncode == 0 and "IFS_DRIVER_DONE" in r.stdout, r
        rows = np.array(read_fixed(fout, nlev - 1))
        assert rows.shape == (2 * ncol, nlev - 1)
        out.update(ifs_Ri=Ri, ora_ifs_km=rows[0::2], ora_ifs_kh=rows[1::2])
        # ---------------- ICON arm (unstable interfaces)
        Ri, l_mix, Km, Kh, fm, _ = run_ours(d, "icon")
        kneu = Km / fm  # = l^2 |dV/dz| (cloud-top entrainment is 0 at its default)
        fin, fout = Path(tmp) / "icon_in.txt", Path(tmp) / "icon_out.txt"
        with open(fin, "w") as f:
            f.write(f"{ncol * (nlev - 1)} {ICON_CB!r} {ICON_CC!r}\n")
            for i in range(ncol):
                for k in range(nlev - 1):
                    wr(f, (Ri[i, k], l_mix[i, k], d["zm"][i, k], d["zm"][i, k + 1], kneu[i, k]))
        r = subprocess.run([str(B / "icon_exe"), str(fin), str(fout)],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr)
        assert r.returncode == 0 and "ICON_DRIVER_DONE" in r.stdout, r
        rows = np.array(read_fixed(fout, 2)).reshape(ncol, nlev - 1, 2)
        out.update(icon_Ri=Ri, icon_l=l_mix, icon_kneu=kneu,
                   ora_icon_km=rows[..., 0], ora_icon_kh=rows[..., 1])
    for k, v in out.items():
        assert np.all(np.isfinite(v)), k
    gfortran = subprocess.run(["gfortran", "--version"], capture_output=True,
                              text=True).stdout.split("\n")[0]
    meta = dict(oracle=ORACLE, compiler=gfortran, arms=ARMS, icon_cb_cc=[ICON_CB, ICON_CC],
                constants=dict(g=C.g, R_d=C.R_d, c_pd=C.c_pd, retv=C.R_v / C.R_d - 1.0,
                               L_v=C.L_v, kappa_vk=C.kappa_von_karman),
                inputs_sha256=hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(),
                names=[str(n) for n in src["names"]],
                setup="IFS: PKHVFL=0 KINV=KLEV PZ0MM=0 PLAM=150, PRI=legoESM Ri; "
                      "ICON: fragment fed legoESM Ri, l, neutral K")
    np.savez_compressed(a.out, meta=json.dumps(meta), dt=d["dt"],
                        **{f"in_{k}": v for k, v in d.items() if k != "dt"}, **out)
    print("WROTE", a.out, Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()

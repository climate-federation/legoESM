"""Generate the Kessler oracle fixture by RUNNING the compiled ESCOMP kessler_run.

Oracle: ESCOMP/atmospheric_physics tag ``atmos_phys0_31_000``
(856127273a3f8b6a64315967e0d7d42ce4d02994), ``schemes/kessler/kessler.F90``,
compiled unmodified by ``build.sh`` (gfortran, fp64).

Inputs are what the legoESM Kessler wrapper hands the scheme, so the fixture
checks the PUBLIC entry point:

* columns: the six tier-1 warm-rain audit columns (built by the MG2 fixture
  generator, reused) + the Weisman-Klemp (1982) supercell sounding with
  prescribed condensate used by the tier-2 audit;
* ``rho`` / ``dz``: legoESM ``compute_rho`` / ``compute_layer_dz`` (moist),
  as the idealized-forcing adapter passes them;
* oracle ``z`` = ``_shared.heights_from_layer_dz(dz)`` (the mapping the port
  uses), ``pk = (p/p_ref)^kappa``, ``theta = T/pk``; host constants
  ``cpair=c_pd, rair=R_d, lv=L_v, pref=p_ref, rhoqr=rho_water``;
* the same ``q`` numbers on both sides.

usage: python make_fixture.py --ap <atmospheric_physics checkout> \
           --build <dir> --out tests/atmosphere/oracle_fixtures/kessler.npz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAG = "atmos_phys0_31_000"
DTS = (1200.0, 300.0)
COL_FIELDS = ("p_half", "p_full", "T", "q_v", "q_c", "q_r")


def _jnp():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jnp


def build_supercell_column():
    """Weisman-Klemp (1982) / DCMIP2016-supercell analytic sounding on CAM6
    L32 (ps = 1000 hPa); condensate prescribed as in the tier-2 audit:
    saturated (RH 1.005) cloud 850-500 hPa with q_c 1.5e-3, q_r 2e-3 at and
    below 500 hPa."""
    jnp = _jnp()
    from legoesm import constants as C
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0
    from legoesm.thermo import saturation_specific_humidity
    th0, th_tr, z_tr, T_tr = 300.0, 343.0, 12000.0, 213.0
    z = np.arange(0.0, 25000.0, 10.0)
    th = np.where(z <= z_tr, th0 + (th_tr - th0) * (z / z_tr) ** 1.25,
                  th_tr * np.exp(C.g / (C.c_pd * T_tr) * (z - z_tr)))
    rh = np.where(z <= z_tr, 1.0 - 0.75 * (z / z_tr) ** 1.25, 0.25)
    p = np.empty_like(z)
    p[0] = C.p_ref
    T = np.empty_like(z)
    qv = np.empty_like(z)
    for i in range(len(z)):
        if i > 0:
            Tv = T[i - 1] * (1 + (1.0 / C.epsilon - 1.0) * qv[i - 1])
            p[i] = p[i - 1] * math.exp(-C.g * (z[i] - z[i - 1]) / (C.R_d * Tv))
        T[i] = th[i] * (p[i] / C.p_ref) ** C.kappa
        qs = float(saturation_specific_humidity(jnp.asarray(T[i]), jnp.asarray(p[i])))
        qv[i] = min(rh[i] * qs, 0.014)
    ph = np.array(CAM6_L32_HYAI) * CAM6_L32_P0 + np.array(CAM6_L32_HYBI) * C.p_ref
    pf = 0.5 * (ph[1:] + ph[:-1])
    lp = np.log(p[::-1])
    Tf = np.interp(np.log(pf), lp, T[::-1])
    qvf = np.interp(np.log(pf), lp, qv[::-1])
    P = pf / 100.0
    qs = np.asarray(saturation_specific_humidity(jnp.asarray(Tf), jnp.asarray(pf)))
    cl = (P <= 850) & (P >= 500)
    return dict(name="supercell_wk82", p_half=ph, p_full=pf, T=Tf,
                q_v=np.where(cl, 1.005 * qs, qvf), q_c=np.where(cl, 1.5e-3, 0.0),
                q_r=np.where(P >= 500, 2.0e-3, 0.0))


def build_columns():
    spec = importlib.util.spec_from_file_location(
        "mg2_make_fixture", HERE.parent / "morrison_mg2_warm_rain" / "make_fixture.py")
    mg2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg2)
    cols = [{k: c[k] for k in ("name",) + COL_FIELDS} for c in mg2.build_columns()]
    cols.append(build_supercell_column())
    return cols


def geometry(cols):
    """(rho, dz, z_full, pk, theta), each (ncol, nlev), as the port derives them."""
    jnp = _jnp()
    from legoesm import constants as C
    from legoesm.atmosphere.physics._shared import (
        compute_layer_dz, compute_rho, heights_from_layer_dz)
    g = {k: jnp.asarray(np.stack([c[k] for c in cols])) for k in COL_FIELDS}
    rho = compute_rho(g["T"], g["p_full"], g["q_v"])
    dz = compute_layer_dz(g["T"], g["p_half"], g["q_v"])
    z_full = heights_from_layer_dz(dz)[0]
    pk = (g["p_full"] / C.p_ref) ** (C.R_d / C.c_pd)
    theta = g["T"] / pk
    return tuple(np.asarray(a) for a in (rho, dz, z_full, pk, theta))


def write_input(path, dt, rho, z, pk, theta, qv, qc, qr):
    from legoesm import constants as C
    ncol, nz = rho.shape
    with open(path, "w") as f:
        f.write(f"{ncol} {nz} {dt!r} {C.c_pd!r} {C.R_d!r} {C.L_v!r} "
                f"{C.p_ref!r} {C.rho_water!r}\n")
        for i in range(ncol):
            for k in range(nz):
                f.write(" ".join(repr(float(a[i, k])) for a in
                                 (rho, z, pk, theta, qv, qc, qr)) + "\n")


def read_output(path, ncol, nz):
    vals = np.array(Path(path).read_text().split(), dtype=np.float64)
    vals = vals.reshape(ncol, 1 + 5 * nz)
    precl = vals[:, 0]
    st = vals[:, 1:].reshape(ncol, nz, 5)
    return precl, st[..., 0], st[..., 1], st[..., 2], st[..., 3]


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ap", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.ap), str(a.build)],
                   check=True)
    from legoesm import constants as C
    cols = build_columns()
    fx = {f"col_{k}": np.stack([c[k] for c in cols]) for k in COL_FIELDS}
    fx["col_names"] = np.array([c["name"] for c in cols])
    rho, dz, z, pk, theta = geometry(cols)
    fx.update(col_rho=rho, col_dz=dz, ora_in_z=z, ora_in_pk=pk, ora_in_theta=theta)
    ncol, nz = rho.shape
    fx["dts"] = np.array(DTS)
    for dt in DTS:
        fin = a.build / f"in_dt{dt:g}.txt"
        fout = a.build / f"out_dt{dt:g}.txt"
        write_input(fin, dt, rho, z, pk, theta, fx["col_q_v"], fx["col_q_c"], fx["col_q_r"])
        subprocess.run([str(a.build / "kessler_driver"), str(fin), str(fout)], check=True)
        precl, th, qv, qc, qr = read_output(fout, ncol, nz)
        tag = f"ora_dt{dt:g}_"
        fx[tag + "precl_m_s"] = precl
        fx[tag + "theta"] = th
        fx[tag + "q_v"] = qv
        fx[tag + "q_c"] = qc
        fx[tag + "q_r"] = qr
    src = (a.build / "kessler.F90").read_bytes()
    meta = dict(
        oracle="ESCOMP/atmospheric_physics", tag=TAG,
        sha=_git(a.ap, "rev-parse", f"{TAG}^{{commit}}"),
        file="schemes/kessler/kessler.F90",
        file_sha256=hashlib.sha256(src).hexdigest(),
        rho_water=C.rho_water,
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        legoesm_sha=_git(HERE, "rev-parse", "HEAD"),
        generator=("python scripts/validate/oracle/kessler/make_fixture.py "
                   f"--ap <atmospheric_physics> --build <dir> --out {a.out.name}"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


if __name__ == "__main__":
    main()

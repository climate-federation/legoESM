"""Fixture for MG2 cloud-droplet sedimentation under a specified droplet number
(tests/atmosphere/microphysics/oracle_fixtures/mg2_specified_nc_sed.npz).

Oracle: ESCOMP/CAM cam_cesm2_1_rel_60 micro_mg2_0.F90, micro_mg_utils.F90 and
wv_sat_methods.F90 compiled UNMODIFIED with the column driver of
../morrison_mg2_warm_rain (mg2_driver.F90, nccons_in=.true., ncnst per column,
CAM6 + CLUBB settings), dt = 600 s.

Columns:
  * the six warm columns of ../morrison_mg2_warm_rain (build_columns) with the
    per-level relvar profile of its prognostic fixture, and its ``thin`` column
    (droplet sub-stepping, cloud fraction shrinking downward);
  * dumped columns: the 12 ice-free columns with cloud water only where
    T > T_freeze and the largest cloud water path, from a legoESM C1 model
    checkpoint (``--dump``), with the dumped T, q_v, q_c, q_r, N_r, cloud
    fraction and CLUBB relvar; the first 6 at ncnst 1e8 m^-3 (the C1 deck's
    Nc_0), the other 6 at 5.5e7 m^-3 (DYCOMS-II RF02).

usage: python make_fixture.py --cam <CAM checkout> --build <dir> --dump <checkpoint.npz> --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "mg2_make_fixture", HERE.parent / "morrison_mg2_warm_rain" / "make_fixture.py")
MF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MF)

DT = MF.DT
N_DUMP = 12
NC_DUMP = (1.0e8, 5.5e7)
FIELDS = ("T", "q_v", "q_c", "cf", "q_r", "N_r", "N_c", "p_full", "p_half", "rho", "relvar")


def synthetic_columns():
    cols = MF.build_prognostic_columns(MF.build_columns())
    thin = next(c for c in MF.build_sedimentation_columns(
        MF.build_activation_columns(cols)) if c["name"] == "thin")
    return cols + [thin]


def dumped_columns(path):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm import constants as C
    from legoesm.grids.vertical import create_hybrid_coordinate
    d = np.load(path)
    assert str(d["number_convention"]) == "per_mass"
    T, qc = d["T"], d["trc_q_c"]
    ice = d["trc_q_i"] + d["trc_q_s"] + d["trc_q_g"]
    coord = create_hybrid_coordinate(T.shape[1], jnp.asarray(d["meta_vgrid"][0]),
                                     jnp.asarray(d["meta_vgrid"][1]), C.p_ref)
    ph = np.asarray(coord.pressure_at_half(jnp.asarray(d["p_s"])))
    pf = np.asarray(coord.pressure_at_full(jnp.asarray(d["p_s"])))
    lwp = (qc * np.diff(ph, axis=1)).sum(1) / C.g
    ok = (ice.max(1) == 0.0) & ((qc <= 0.0) | (T > C.T_freeze)).all(1) & (lwp > 0)
    pick = np.flatnonzero(ok)[np.argsort(-lwp[ok])][:N_DUMP]
    cols = []
    for j, i in enumerate(pick):
        rho = pf[i] / (C.R_d * T[i])
        ncnst = NC_DUMP[0] if j < N_DUMP // 2 else NC_DUMP[1]
        cols.append(dict(name=f"dump{i}", T=T[i], q_v=d["trc_q_v"][i], q_c=qc[i],
                         cf=d["physstate_cloud_fraction"][i], q_r=d["trc_q_r"][i],
                         N_r=d["trc_N_r"][i] * rho, N_c=np.full(T.shape[1], ncnst),
                         p_full=pf[i], p_half=ph[i], rho=rho,
                         relvar=d["physstate_relvar"][i]))
    return cols, pick


def write_inputs(cols, path):
    nlev = len(cols[0]["T"])
    with open(path, "w") as f:
        f.write(f"{len(cols)} {nlev} {DT!r}\n")
        for c in cols:
            ncnst = float(c["N_c"][0])
            f.write(f"{ncnst:.17e}\n")
            pdel = np.diff(c["p_half"])
            lcl = np.maximum(c["cf"], 1e-4)
            for k in range(nlev):
                row = [c["T"][k], c["q_v"][k], c["q_c"][k], 0.0,
                       ncnst / c["rho"][k] * lcl[k] if c["q_c"][k] > 0 else 0.0, 0.0,
                       c["q_r"][k], 0.0, c["N_r"][k] / c["rho"][k], 0.0,
                       c["relvar"][k], 1.0, c["p_full"][k], pdel[k],
                       c["cf"][k], c["cf"][k], c["cf"][k], 1.0]
                f.write(" ".join(f"{x:.17e}" for x in row) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cam", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.cam), str(a.build)], check=True)
    groups = {"syn": synthetic_columns()}
    groups["dump"], pick = dumped_columns(a.dump)
    fx = {"dt": np.array(DT), "dump_cells": pick}
    for g, cols in groups.items():
        ncol, nlev = len(cols), len(cols[0]["T"])
        for k in FIELDS:
            fx[f"{g}_col_{k}"] = np.stack([np.asarray(c[k], np.float64) for c in cols])
        fx[f"{g}_col_names"] = np.array([c["name"] for c in cols])
        fin, fout = a.build / f"{g}_in.txt", a.build / f"{g}_out.txt"
        write_inputs(cols, fin)
        subprocess.run([str(a.build / "mg2_driver"), str(fin), str(fout), "F"], check=True)
        o = MF.load_column_outputs(fout, ncol, nlev)
        for k in MF.ORACLE_COLS + ("prect",):
            fx[f"{g}_ora_{k}"] = o[k]
    files = ["src/physics/cam/micro_mg2_0.F90", "src/physics/cam/micro_mg_utils.F90",
             "src/physics/cam/wv_sat_methods.F90"]
    meta = dict(oracle="ESCOMP/CAM", tag="cam_cesm2_1_rel_60",
                sha256={f: hashlib.sha256((a.cam / f).read_bytes()).hexdigest() for f in files},
                gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                        text=True).stdout.split("\n")[0],
                dump=a.dump.name, dump_sha256=hashlib.sha256(a.dump.read_bytes()).hexdigest(),
                nccons=True, dt=DT)
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

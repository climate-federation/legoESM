"""Generate the CAM6 CLUBB->MG2 relvar oracle fixture by RUNNING the oracles.

Part A (CLUBB qclvar -> relvar): the CLUBB-core trajectories of
``tests/atmosphere/oracle_fixtures/clubb_core_cam.npz`` are regenerated with
the Fortran core (``scripts/validate/oracle/clubb_core``, run with
``CLUBB_FIXTURE_PDF=1`` on a build_oracle.sh build, so the Fortran
``pdf_params`` and CLUBB's own ``qclvar_out`` are also recorded; this script
asserts the regenerated states equal the committed fixture bit for bit).
``s{n}/qclvar_out`` is the closure on state ``s_n`` (clubb_release sets it
after its advances, :1437 -> :1501), i.e. what CAM's r8099 core sets before
its advances (:1150 -> :1798) on the step LEAVING ``s_n``.
``relvar_oracle.F90`` then evaluates the oracle's own lines: ``rcp2`` from the
Fortran PDF components (CLUBB pdf_closure_module.F90 r8099 :1048-1050,
:1072-1074; a transcription cross-check, printed against the captured value),
and ``qclvar = min(1, qclvar_out)`` / ``relvar`` (CAM clubb_intr.F90
cam_cesm2_1_rel_60 :1998, :2416-2425, deep_scheme = 'ZM') from the CAPTURED
``qclvar_out`` and the Fortran in-layer ``rcm`` of the same state.  Edge rows
exercise every branch.

Part B (MG2 consumption of a per-level relvar): the six tier-1 warm columns of
``morrison_mg2_warm_rain`` through the whole ``micro_mg_tend`` (CAM
cam_cesm2_1_rel_60, compiled unmodified) with a per-level relvar profile.

Usage (CPU node; gfortran; legoESM python)::

    python scripts/validate/oracle/clubb_mg2_relvar/make_fixture.py \\
        --pdf <clubb_core npz written with CLUBB_FIXTURE_PDF=1> \\
        --cam /burg/glab/users/pg2328/oracles/CAM --build <dir> \\
        --out tests/atmosphere/oracle_fixtures/clubb_mg2_relvar.npz
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CORE_FIXTURE = ROOT / "tests/atmosphere/oracle_fixtures/clubb_core_cam.npz"
PDF = ("mixt_frac", "chi_1", "chi_2", "rc_1", "rc_2", "cloud_frac_1",
       "cloud_frac_2", "stdev_chi_1", "stdev_chi_2")
# Edge rows: PDF components + rcm_host, one per relvar branch (columns as PDF
# then rcm_host).  Not physical PDFs; they exercise the oracle's arithmetic.
EDGE_ROWS = np.array([
    # all clear: rcp2 = 0, rcm = 0 -> relvarmax
    [0.5, -1e-3, -1e-3, 0.0, 0.0, 0.0, 0.0, 1e-4, 1e-4, 0.0],
    # cloud variance but no host liquid -> relvarmax
    [0.5, 1e-4, 1e-4, 2e-4, 2e-4, 0.9, 0.9, 1e-4, 1e-4, 0.0],
    # host liquid but rcp2 = 0 -> relvarmax
    [0.5, -1e-3, -1e-3, 0.0, 0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4],
    # rcp2 > 1: qclvar clipped to 1
    [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 0.5],
    # ratio < 0.001 -> 0.001
    [0.5, 1e-3, 1e-3, 1e-3, 1e-3, 1.0, 1.0, 1e-3, 1e-3, 1e-6],
    # ratio > 10 -> 10
    [0.5, 1e-5, 1e-5, 1e-5, 1e-5, 0.6, 0.6, 1e-5, 1e-5, 3e-3],
    # inside the clip range
    [0.3, 2e-5, -1e-5, 3e-5, 1e-6, 0.8, 0.2, 2e-5, 3e-5, 4e-5],
    # negative host liquid (rcm**2 > 0)
    [0.3, 2e-5, -1e-5, 3e-5, 1e-6, 0.8, 0.2, 2e-5, 3e-5, -4e-5],
    # raw rcp2 < 0 -> max(0, .) -> relvarmax
    [0.5, -1e-2, -1e-2, 1e-4, 1e-4, 0.0, 0.0, 0.0, 0.0, 1e-4],
])


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def _run_relvar(build, rows, mode=""):
    fin, fout = build / "relvar_in.txt", build / "relvar_out.txt"
    np.savetxt(fin, rows, fmt="%.17e", header=str(len(rows)), comments="")
    subprocess.run([str(build / "relvar_oracle"), str(fin), str(fout), mode], check=True)
    out = np.loadtxt(fout, ndmin=2)
    assert out.shape == (len(rows), 2), out.shape
    return out[:, 0], out[:, 1]


def part_a(pdf_npz, build, fx):
    core = _load(HERE.parent / "clubb_core/gen_clubb_core_fixture.py", "gen_core")
    ref = np.load(CORE_FIXTURE)
    new = np.load(pdf_npz)
    cases = [str(c) for c in ref["meta/cases"]]
    n_steps = int(ref["meta/n_steps"])
    for case in cases:
        for n in range(n_steps + 1):
            for k in core.STATE + core.CARRIED:
                key = f"{case}/s{n}/{k}"
                if not np.array_equal(ref[key], new[key]):
                    raise SystemExit(f"regenerated {key} differs from the committed fixture")
    for case in cases:
        for n in range(1, n_steps + 1):
            cols = [new[f"{case}/s{n}/pdf/{k}"][0] for k in PDF]
            rcm = new[f"{case}/s{n}/rcm"][0]
            qclvar_out = new[f"{case}/s{n}/qclvar_out"][0]
            rows = np.stack(cols + [rcm, qclvar_out], axis=1)
            rcp2, relvar = _run_relvar(build, rows, "qclvar")
            d = np.abs(rcp2 - qclvar_out)
            print(f"{case} s{n}: transcribed rcp2 vs captured qclvar_out "
                  f"max abs {d.max():.3e}, field-norm "
                  f"{d.max() / max(np.abs(qclvar_out).max(), 1e-300):.3e}")
            fx[f"{case}/s{n}/qclvar_out"] = qclvar_out
            fx[f"{case}/s{n}/rcp2"] = rcp2
            fx[f"{case}/s{n}/relvar"] = relvar
    rcp2, relvar = _run_relvar(build, EDGE_ROWS)
    fx["edge/in"] = EDGE_ROWS
    fx["edge/rcp2"] = rcp2
    fx["edge/relvar"] = relvar
    fx["meta/cases"] = np.array(cases)
    fx["meta/n_steps"] = np.array(n_steps)
    return str(ref["meta/oracle"])


def relvar_profile(nlev):
    """Per-level test profile spanning CAM's clip range [0.001, 10]."""
    return 10.0 ** np.linspace(-3.0, 1.0, nlev)


def part_b(cam, build, fx):
    mg = _load(HERE.parent / "morrison_mg2_warm_rain/make_fixture.py", "mg2_fix")
    subprocess.run(["bash", str(HERE.parent / "morrison_mg2_warm_rain/build.sh"),
                    str(cam), str(build)], check=True)
    cols = mg.build_columns()
    ncol, nlev = len(cols), len(cols[0]["T"])
    rv = relvar_profile(nlev)
    fin, fout = build / "col_in_profile.txt", build / "col_out_profile.txt"
    mg.write_column_inputs(cols, fin, rv)
    subprocess.run([str(build / "mg2_driver"), str(fin), str(fout), "F"], check=True)
    o = mg.load_column_outputs(fout, ncol, nlev)
    fx["mg2/relvar"] = rv
    for k in ("prctot", "pratot", "nevapr", "qrtend", "nrtend"):
        fx[f"mg2/{k}"] = o[k]
    fx["mg2/rain_sfc_flux"] = o["rflx_bot"][:, -1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--cam", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    a.build.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gfortran", "-O0", "-g", "-ffree-line-length-none", "-o",
                    str(a.build / "relvar_oracle"), str(HERE / "relvar_oracle.F90")],
                   check=True)
    fx = {}
    clubb_oracle = part_a(a.pdf, a.build, fx)
    part_b(a.cam, a.build, fx)
    meta = dict(
        clubb_oracle=clubb_oracle,
        cam="ESCOMP/CAM cam_cesm2_1_rel_60", cam_sha=_git(a.cam, "rev-parse", "HEAD"),
        cam_files=["src/physics/cam/clubb_intr.F90 (lines in relvar_oracle.F90)",
                   "src/physics/cam/micro_mg2_0.F90", "src/physics/cam/micro_mg_utils.F90"],
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        legoesm_sha=_git(HERE, "rev-parse", "HEAD"),
        generator=("scripts/validate/oracle/clubb_mg2_relvar/run.sh "
                   "(clubb_core generator with CLUBB_FIXTURE_PDF=1, then make_fixture.py)"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

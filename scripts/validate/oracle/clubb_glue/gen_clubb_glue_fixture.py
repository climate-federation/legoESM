"""Pack the CAM clubb_intr host-glue oracle run into the test fixture.

Usage: gen_clubb_glue_fixture.py WORK_DIR CAM_ROOT OUT_NPZ

WORK_DIR holds inputs.npz + oracle_in_*.txt (build_inputs.py) and
oracle_out_*.txt (build.sh, the compiled verbatim CAM lines).  The fixture
keeps the inputs both sides consume and the oracle's host->CLUBB inputs and
post-CLUBB tendencies for both CLUBB tops (15000 Pa, 100 Pa).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants as C

CASES = ("15000", "100")
# Per-column record order written by main.F90 (one line each).
FIELDS = ("sfc", "thlm", "rtm", "thv", "rho", "exner", "p", "du", "dv", "dT",
          "dqv", "dqc", "alst", "deepcu", "aist", "cloud_frac")
KEEP = ("sfc", "thlm", "rtm", "du", "dv", "dT", "dqv", "dqc", "aist")
INPUTS = ("lat", "ps", "p_half", "p_full", "T", "qv", "qc", "qi", "u", "v",
          "z_full", "z_half", "shf", "E", "lhf", "taux", "tauy", "rtp2", "dt",
          "pref_mid")
PER_CASE = ("drt", "dthl", "du", "dv", "rcm_out", "cf_out", "top_lev")


def parse(path):
    res = {"cols": {}, "clubbtop": {}}
    it = iter(Path(path).read_text().splitlines())
    for ln in it:
        t = ln.split()
        if t[0] == "TROPLEV":
            res["troplev"] = np.array([int(x) for x in t[1:]])
        elif t[0] == "TOP_LEV":
            res["top_lev"] = int(t[1])
        elif t[0] == "CLUBBTOP":
            res["clubbtop"][int(t[1]) - 1] = int(t[2])
        elif t[0] == "COL":
            c = int(t[1]) - 1
            res["cols"][c] = {f: np.array([float(x) for x in next(it).split()])
                              for f in FIELDS}
    return res


def main(work, cam_root, out):
    work = Path(work)
    d = dict(np.load(work / "inputs.npz"))
    ncol, nlev = d["T"].shape
    fx = {f"in/{k}": np.asarray(d[k]) for k in INPUTS}
    # CAM's dry-static density rho_ds = pdel / (g dz) (clubb_intr.F90:1654),
    # fed to both sides so the comparison isolates the glue.
    dz = d["z_half"][:, :-1] - d["z_half"][:, 1:]
    fx["in/rho_cam"] = (d["p_half"][:, 1:] - d["p_half"][:, :-1]) / (C.g * dz)
    for t in CASES:
        o = parse(work / f"oracle_out_{t}.txt")
        if sorted(o["cols"]) != list(range(ncol)):
            raise SystemExit(f"case {t}: oracle wrote columns {sorted(o['cols'])}")
        if o["top_lev"] - 1 != int(d[f"top_lev_{t}"]):
            raise SystemExit(f"case {t}: oracle top_lev {o['top_lev']} vs "
                             f"inputs {int(d[f'top_lev_{t}']) + 1}")
        for k in PER_CASE:
            fx[f"{t}/in/{k}"] = np.asarray(d[f"{k}_{t}"])
        nclubb = nlev - int(d[f"top_lev_{t}"])
        for f in KEEP:
            a = np.stack([o["cols"][c][f] for c in range(ncol)])
            if f in ("thlm", "rtm"):
                a = a[:, :nclubb]          # CLUBB levels only, ascending
            fx[f"{t}/oracle/{f}"] = a
        fx[f"{t}/oracle/troplev_1b"] = o["troplev"]
        fx[f"{t}/oracle/clubbtop_1b"] = np.array([o["clubbtop"][c] for c in range(ncol)])
    for k, v in fx.items():
        if v.dtype.kind == "f" and not np.all(np.isfinite(v)):
            raise SystemExit(f"non-finite {k}")
    sha = subprocess.run(["git", "-C", cam_root, "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    fx["meta/oracle"] = np.array(f"ESCOMP/CAM cam_cesm2_1_rel_60 {sha}")
    fx["meta/generator"] = np.array(
        "sbatch --export=ALL,WT=<checkout>,ORACLES=<oracles root> "
        "scripts/validate/oracle/clubb_glue/run.sh")
    fx["meta/cases"] = np.array(CASES)
    np.savez_compressed(out, **fx)
    print(f"wrote {out}: {len(fx)} arrays, oracle {sha}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

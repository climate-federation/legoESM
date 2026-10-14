#!/usr/bin/env python3
"""G4 kt = 11 surface-forcing witness: the shared fld_read + NCAR bulk on
NEMO's own kt = 11 inputs, scored against NEMO's abort file.

NEMO's stop at kt = 11 writes utau, vtau, qsr of the kt = 11 sbc call
(diawri.f90:645-649, fp64).  Inputs: the CORE records NEMO's log names for
kt = 11 (ocean.output: 6-hourly record 6, daily 2, monthly 1) and the
surface mean the bulk reads, sst_m = (sum + T_entry) * 0.5 (sbcssm.f90,
nn_fsbc = 2, EOS-80), from twin a's step-10 restart.  Wind stress and qsr
are on T points, wet level-1 cells scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "orca2_l4"))
import nemo_testcase_l4_orca2_o1_numerical_gate as o1  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import score  # noqa: E402

# 0-based record per field at kt = 11, from NEMO's own log (ocean.output:1860-1868)
RECORD_KT11 = {"wndi": 5, "wndj": 5, "tair": 5, "humi": 5, "slp": 5,
               "qsr_down": 1, "qlw_down": 1, "precip_raw": 0, "snow_raw": 0}
ALBO = 0.066           # sbc_phy.f90:49, NEMO's PARAMETER


def sst_m_kt11(restart: Path) -> np.ndarray:
    """sbcssm.f90 nn_fsbc > 1: at kt = 11 the partial sum written at kt = 10
    gains level-1 T entering step 11, then times 1/nn_fsbc."""
    with Dataset(restart) as ds:
        require(int(ds["kt"][:]) == 10 and int(ds["nn_fsbc"][:]) == 2,
                f"{restart.name}: not the step-10, nn_fsbc = 2 restart")
        partial = np.asarray(ds["sst_m"][0], dtype=np.float64)
        entry = np.asarray(ds["tn"][0, 0], dtype=np.float64)
    return (partial + entry) * (1.0 / 2.0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"REFUSE: {message}")


def measure(run: Path, abort: Path, *, plant: bool = False) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    o1.set_policy(o1.PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "CPU only")
    cache = {kind: o1.load_weights(run, kind) for kind in (False, True)}
    maps = {}
    for field, filename, variable, bicubic in o1.FIELD_SPECS:
        with Dataset(run / filename) as ds:
            source = np.asarray(ds[variable][RECORD_KT11[field]].data, dtype=np.float64)
        idx, wgt = cache[bicubic]
        maps[field] = np.asarray(jax.jit(
            lambda v, s, w, b=bicubic: o1.nemo_fld_interp(v, s, w, bicubic=b))(source, idx, wgt))
    with Dataset(run / "domain_cfg.nc") as ds:
        cos, sin = o1.nemo_t_rotation_from_domain(*(
            np.asarray(ds[n][:].data, dtype=np.float64)
            for n in ("glamt", "gphit", "glamv", "gphiv")))
    maps["wndi"], maps["wndj"] = map(np.asarray, jax.jit(o1.rotate_en_to_ij)(
        maps["wndi"], maps["wndj"], cos, sin))
    sst = sst_m_kt11(run / "ORCA2G1_00000010_restart.nc")
    with Dataset(run / "mesh_mask.nc") as ds:
        wet = np.asarray(ds["tmask"][0, 0].data, dtype=bool)
    tau_x, tau_y, *_ = map(np.asarray, jax.jit(
        lambda u, v, ta, qa, ts, ps: o1.air_sea_fluxes(
            u, v, ta, qa, ts + o1.constants.T_freeze, slp_Pa=ps, algo="ncar"))(
        maps["wndi"], maps["wndj"], maps["tair"], maps["humi"], sst, maps["slp"]))
    card = {"utau": -tau_x, "vtau": -tau_y,
            "qsr": (1.0 - ALBO) * maps["qsr_down"] * wet}
    with Dataset(abort) as ds:
        nemo = {"utau": ds["sozotaux"][0], "vtau": ds["sometauy"][0], "qsr": ds["soshfldo"][0]}
    rows = {}
    for name, cand in card.items():
        ref = np.asarray(nemo[name], dtype=np.float64)
        unequal = (cand.view("<u8") != ref.view("<u8")) & wet
        row = score(f"kt11.{name}", ref, np.asarray(cand, dtype=np.float64), wet, plant=plant)
        row["unequal_cells"] = int(unequal.sum())
        rows[name] = row
    return {"kt": 11, "wet_cells": int(wet.sum()), "rows": rows,
            "inputs": {"records_0based": RECORD_KT11, "sst": "sst_m_kt11(restart 10)"}}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="twin a (pinned inputs, restart 10)")
    p.add_argument("--abort", type=Path, required=True, help="the reference's output.abort.nc")
    p.add_argument("--json", type=Path)
    p.add_argument("--plant", action="store_true")
    a = p.parse_args()
    out = measure(a.run, a.abort, plant=a.plant)
    if a.plant:
        fired = all(r["status"] != "AT-BAR" for r in out["rows"].values())
        print("STATUS PLANT-FIRED" if fired else "REFUSE: plant stayed at the bar")
        return 1 if fired else 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for name, r in out["rows"].items():
        print(name, r["status"], r["unequal_cells"], "/", r["n"], f"{r['normalized_max_abs']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

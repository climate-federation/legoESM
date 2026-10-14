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
import re
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
    if not condition:              # exit 73: never confused with a fired plant (1)
        print(f"REFUSE: {message}", file=sys.stderr)
        raise SystemExit(73)


def provenance(run: Path, abort: Path) -> None:
    """The abort file is the reference run's kt = 11 stop, whose log reads the
    records RECORD_KT11 names, and the replay reads that run's own inputs."""
    import filecmp
    ref = abort.parent
    require((ref / "time.step").read_text().split() == ["11"]
            and re.search(r"^ *kt 11 \|V\| +max", (ref / "ocean.output").read_text(), re.M),
            f"{ref}: the abort file is not NEMO's stop at kt = 11")
    log = (ref / "ocean.output").read_text()
    for field, filename, variable, _ in o1.FIELD_SPECS:
        hit = re.search(rf"fld_read: var {variable} kt = +11 .*record: +(\d+)", log)
        require(bool(hit) and int(hit.group(1)) == RECORD_KT11[field] + 1,
                f"{variable}: NEMO's log does not read record {RECORD_KT11[field] + 1} at kt = 11")
        require(filecmp.cmp(run / filename, ref / filename, shallow=False),
                f"{filename}: the replay's input is not the abort run's")
    for name in ("domain_cfg.nc", "namelist_ref", "weights_core2_orca2_bicub.nc",
                 "weights_core2_orca2_bilin.nc"):
        require(filecmp.cmp(run / name, ref / name, shallow=False),
                f"{name}: the replay's input is not the abort run's")


def magnitude_split(card: dict, nemo: dict, wet, wind, dtemp) -> dict:
    """Stress-magnitude ratio card/NEMO - 1 by population, with the 10 m
    wind speed and SST minus air temperature of each population."""
    rel = (np.hypot(card["utau"], card["vtau"])[wet]
           / np.hypot(nemo["utau"], nemo["vtau"])[wet] - 1.0)
    out = {"negative": int((rel < 0).sum()), "min": float(rel.min()), "max": float(rel.max())}
    for label, lo, hi in (("below_1e-6", 0.0, 1e-6), ("1e-6_to_1e-3", 1e-6, 1e-3),
                          ("above_1e-3", 1e-3, np.inf)):
        sel = (np.abs(rel) >= lo) & (np.abs(rel) < hi)
        out[label] = {"cells": int(sel.sum())}
        if sel.any():
            out[label].update({
                "wind_p5_p50_p95": np.percentile(wind[wet][sel], [5, 50, 95]).tolist(),
                "sst_minus_tair_p5_p50_p95": np.percentile(dtemp[wet][sel], [5, 50, 95]).tolist()})
    return out


def measure(run: Path, abort: Path, *, plant: bool = False) -> dict:
    import jax
    provenance(run, abort)
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
    nemo = {k: np.asarray(v, dtype=np.float64) for k, v in nemo.items()}
    split = magnitude_split(card, nemo, wet, np.hypot(maps["wndi"], maps["wndj"]),
                            sst + o1.constants.T_freeze - maps["tair"])
    rows = {}
    for name, cand in card.items():
        ref = np.asarray(nemo[name], dtype=np.float64)
        unequal = (cand.view("<u8") != ref.view("<u8")) & wet
        row = score(f"kt11.{name}", ref, np.asarray(cand, dtype=np.float64), wet, plant=plant)
        row["unequal_cells"] = int(unequal.sum())
        rows[name] = row
    return {"kt": 11, "wet_cells": int(wet.sum()), "rows": rows, "stress_magnitude": split,
            "inputs": {"records_0based": RECORD_KT11, "sst": "sst_m_kt11(restart 10)"}}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="twin a (pinned inputs, restart 10)")
    p.add_argument("--abort", type=Path, required=True, help="the reference's output.abort.nc")
    p.add_argument("--json", type=Path)
    p.add_argument("--plant", action="store_true")
    a = p.parse_args()
    out = measure(a.run, a.abort)
    if a.plant:     # only a row at the bar unplanted can show the plant firing
        clean = [k for k, r in out["rows"].items() if r["status"] == "AT-BAR"]
        planted = measure(a.run, a.abort, plant=True)["rows"]
        fired = bool(clean) and all(planted[k]["status"] != "AT-BAR" for k in clean)
        print(f"rows at the bar: {clean}; already over (plant not informative): "
              f"{sorted(set(out['rows']) - set(clean))}")
        print("STATUS PLANT-FIRED" if fired else "REFUSE: plant stayed at the bar")
        return 1 if fired else 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for name, r in out["rows"].items():
        print(name, r["status"], r["unequal_cells"], "/", r["n"], f"{r['normalized_max_abs']:.3e}")
    print(json.dumps(out["stress_magnitude"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

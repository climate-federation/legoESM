#!/usr/bin/env python3
"""Would routing lane 3's ICE_RHEO card through the O6 kernels move its certified numbers?

Runs lane 3's own committed gates (rung-3.4 kt=1 gate, first-divergence sweep,
completed-step-8 active window) once per arm, swapping one kernel at a time by
harness patch (the kernel adapters of ``nemo_si3_o6_lane3_kernel_probe``), and
diffs every scored row of each gate against the ``lane3`` arm: status moves in
both directions and any change of a row's numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
from legoesm.ice.fidelity import nemo_rheo_testcase_recipe as recipe
from legoesm.ice.ridging import SI3JPL1RidgingConfig

HERE = Path(__file__).resolve().parent
ARMS = ("lane3", "o6_rdgrft", "o6_rdgrft_cor")
GATES = ("kt1", "sweep", "active_window")
VOLATILE = {"worktree", "provenance", "elapsed_s", "timestamp"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


kernel = _load("nemo_si3_o6_lane3_kernel_probe_routing", HERE / "nemo_si3_o6_lane3_kernel_probe.py")
trajectory = _load(
    "nemo_si3_rung34_trajectory_routing", HERE / "nemo_si3_phase2_rung34_trajectory_gate.py"
)
rung34 = _load("nemo_si3_rung34_gate_routing", HERE / "nemo_si3_phase2_rung34_gate.py")

# ICE_RHEO card constants the ridging adapter needs (the card itself is built inside each gate).
CARD_CONSTANTS = SimpleNamespace(
    ridging_config=SI3JPL1RidgingConfig(),
    nn_icesal=4,
    dt_s=30.0,
    thermodynamics=False,
    halo_width=2,
)


def rows_of(report) -> dict[str, dict]:
    """Every scored row (a dict with name and status) anywhere in a gate report."""
    out: dict[str, dict] = {}

    def walk(o, path):
        if isinstance(o, dict):
            if "name" in o and "status" in o:
                out[f"{path}|{o['name']}"] = o
            for k, v in o.items():
                if k not in VOLATILE:
                    walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            for v in o:
                walk(v, path)

    walk(report, "")
    return out


def diff_rows(base: dict[str, dict], arm: dict[str, dict]) -> dict[str, object]:
    changed = [
        k
        for k in base
        if json.dumps(base[k], sort_keys=True) != json.dumps(arm.get(k), sort_keys=True)
    ]
    moves = [(k, base[k]["status"], arm[k]["status"]) for k in changed if k in arm]
    return {
        "rows": len(base),
        "rows_missing_or_added": sorted(set(base) ^ set(arm)),
        "rows_with_any_change": len(changed),
        "status_worsened": [m for m in moves if m[1] != "DEBT" and m[2] == "DEBT"],
        "status_improved": [m for m in moves if m[1] == "DEBT" and m[2] != "DEBT"],
        "first_changes": changed[:10],
    }


def run_gate(name: str, root: Path, restart_root: Path):
    if name == "kt1":
        return rung34.run_gate(root)
    if name == "sweep":
        return trajectory.run_sweep(root)
    return trajectory.run_active_window(root, restart_root)


def counted(fn, counter: dict, key: str):
    """Wrap a kernel so each call (trace or eager) is counted: proves the patch is reached."""

    def run(*args, **kwargs):
        counter[key] = counter.get(key, 0) + 1
        return fn(*args, **kwargs)

    return run


def run(root: Path, restart_root: Path, gates=GATES, *, plant: bool = False) -> dict:
    with netCDF4.Dataset(root / "output.init_ice.nc") as ds:
        sst_m = recipe._periodic_halo(np.asarray(ds["sst"][0]).T)
    saved = (recipe.apply_si3_jpl1_ridging, recipe._ice_cor, trajectory.apply_si3_jpl1_ridging)
    arms = ARMS + (("o6_rdgrft_planted",) if plant else ())
    reports: dict[str, dict[str, dict]] = {}
    calls: dict[str, dict[str, dict]] = {}
    for arm in arms:
        reports[arm], calls[arm] = {}, {}
        for g in gates:
            count: dict[str, int] = {}
            try:
                if arm != "lane3":
                    scale = 1.0 + 1.0e-10 if arm == "o6_rdgrft_planted" else 1.0
                    rdg = kernel.o6_ridging_adapter(CARD_CONSTANTS, sst_m, murdg_scale=scale)
                    rdg = counted(rdg, count, "o6_rdgrft")
                    recipe.apply_si3_jpl1_ridging = trajectory.apply_si3_jpl1_ridging = rdg
                if arm == "o6_rdgrft_cor":
                    recipe._ice_cor = counted(kernel.o6_cor_adapter(sst_m), count, "o6_cor")
                reports[arm][g] = run_gate(g, root, restart_root)
            finally:
                (
                    recipe.apply_si3_jpl1_ridging,
                    recipe._ice_cor,
                    trajectory.apply_si3_jpl1_ridging,
                ) = saved
            calls[arm][g] = count
    summary = {
        arm: {
            g: {
                "status": reports[arm][g].get("status"),
                "o6_kernel_calls": calls[arm][g],
                **diff_rows(rows_of(reports["lane3"][g]), rows_of(reports[arm][g])),
            }
            for g in gates
        }
        for arm in arms
    }
    return {
        "format": "nemo-si3-o6-lane3-routing-probe-v2",
        "plant": plant,
        "summary": summary,
        "reports": reports,
    }


def verdict(out: dict) -> bool:
    """Clean: O6 arms reach every gate and change no row.  Plant: the planted arm moves rows."""
    s = out["summary"]
    reached = all(
        s[a][g]["o6_kernel_calls"].get("o6_rdgrft", 0) > 0 for a in s if a != "lane3" for g in s[a]
    )
    reached &= all(
        s["o6_rdgrft_cor"][g]["o6_kernel_calls"].get("o6_cor", 0) > 0 for g in s["o6_rdgrft_cor"]
    )
    clean = all(
        v["rows_with_any_change"] == 0 and not v["rows_missing_or_added"]
        for a in ARMS
        for v in s[a].values()
    )
    if not out["plant"]:
        return reached and clean
    return (
        reached
        and clean
        and any(v["rows_with_any_change"] > 0 for v in s["o6_rdgrft_planted"].values())
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=trajectory.ROOT)
    p.add_argument("--restart-root", type=Path, default=trajectory._ACTIVE_RESTART_ROOT)
    p.add_argument("--gate", action="append", choices=GATES)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--plant", action="store_true")
    a = p.parse_args()
    out = run(a.root, a.restart_root, tuple(a.gate or GATES), plant=a.plant)
    if a.artifact:
        a.artifact.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(out["summary"], indent=2, default=str))
    return 0 if verdict(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())

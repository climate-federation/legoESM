#!/usr/bin/env python3
"""Measure round 330's shared TKE change on DINO's two production grids.

This is the DINO-only form of the established round-311 normalized-JAXPR and
round-329 three-step value probe.  Imports come from ``repo`` so the same
committed instrument can measure the old-clamp and candidate trees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("grid", choices=("default", "faithful"))
    parser.add_argument("mode", choices=("program", "values"))
    parser.add_argument("output", type=Path)
    return parser.parse_args()


ARGS = _parse_args()
REPO = ARGS.repo.resolve()
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)


def _bottom_census(model, state):
    raw = np.asarray(model.z_coord.bottom_level)
    n_solve = int(np.asarray(state.T.data).shape[-1] - 1)
    return {
        "state_levels": n_solve + 1,
        "solve_rows": n_solve,
        "bottom_level_eq_solve_rows": int(np.count_nonzero(raw == n_solve)),
        "bottom_level_outside_solve": int(
            np.count_nonzero((raw < 0) | (raw >= n_solve))),
        "bottom_level_min": int(raw.min()),
        "bottom_level_max": int(raw.max()),
    }


def _leaf_rows(state, step):
    rows = []
    for path, value in jax.tree_util.tree_leaves_with_path(state):
        array = np.ascontiguousarray(np.asarray(value))
        rows.append({
            "step": step,
            "path": jax.tree_util.keystr(path),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        })
    return rows


original_step = LatLonCGridOceanModel.step
report = {"format": "orca2-round330-dino-effect-v1", "grid": ARGS.grid}

if ARGS.mode == "program":
    def capture_program(self, state, *args, **kwargs):
        report["bottom_census"] = _bottom_census(self, state)
        text = str(jax.make_jaxpr(
            lambda value: original_step(self, value, *args, **kwargs))(state))
        report["jaxpr"] = re.sub(r" at 0x[0-9a-f]+", " at 0xADDR", text)
        ARGS.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        raise SystemExit(0)

    LatLonCGridOceanModel.step = capture_program
else:
    report["leaves"] = []
    step_number = 0

    def capture_values(self, state, *args, **kwargs):
        global step_number
        if step_number == 0:
            report["bottom_census"] = _bottom_census(self, state)
        result = original_step(self, state, *args, **kwargs)
        step_number += 1
        host_result = jax.device_get(result)
        report["leaves"].extend(_leaf_rows(host_result, step_number))
        return result

    LatLonCGridOceanModel.step = capture_values


run_dir = Path(tempfile.mkdtemp(prefix=f"r330_dino_{ARGS.grid}_"))
driver_args = [
    "run_dino.py",
    "--grid", "latlon",
    "--recipe", "nemo_dino_kamm_mlf",
    "--n-lon", "50",
    "--allow-multiyear",
    "--dt", "2700",
    "--days", "0.09375",  # three 2700-s production steps
    "--snapshot-every-days", "0.09375",
    "--output-dir", str(run_dir),
]
if ARGS.grid == "faithful":
    driver_args[1:1] = [
        "--config",
        str(REPO / "scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml"),
    ]
    driver_args.insert(driver_args.index("--allow-multiyear"),
                       "--nemo-faithful-grid")

sys.argv = driver_args
sys.path.insert(0, str(REPO / "scripts/run"))
import run_dino  # noqa: E402

run_dino.main()

if ARGS.mode == "program":
    raise SystemExit("DINO driver never called LatLonCGridOceanModel.step")
if step_number != 3:
    raise SystemExit(f"expected three production steps, got {step_number}")
report["steps"] = step_number
ARGS.output.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

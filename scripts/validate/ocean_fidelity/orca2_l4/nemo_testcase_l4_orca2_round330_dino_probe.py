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
    parser.add_argument(
        "mode", choices=("program", "values", "tke-program", "tke-values"))
    parser.add_argument("output", type=Path)
    return parser.parse_args()


ARGS = _parse_args()
REPO = ARGS.repo.resolve()
for package in (REPO, REPO / "packages/core", REPO / "packages/ocean",
                REPO / "packages/atmosphere", REPO / "packages/coupler",
                REPO / "packages/ice", REPO / "packages/land",
                REPO / "packages/ml", REPO / "packages/tools", REPO / "src"):
    sys.path.insert(0, str(package))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.experiments.dino import (  # noqa: E402
    dino_config_for_recipe,
    dino_lat_lon_grid,
    dino_lat_lon_model_config,
    dino_lat_lon_state,
    dino_lat_lon_vertical,
    nemo_faithful_dino_config,
)
from legoesm.ocean.physics.vertical_mixing.tke import (  # noqa: E402
    _solve_tke_backward_euler,
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


def _tke_probe():
    """Run DINO's selected literal TKE solve on the requested real grid."""
    dino_cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
    if ARGS.grid == "faithful":
        dino_cfg = nemo_faithful_dino_config(dino_cfg)
    grid = dino_lat_lon_grid(dino_cfg, n_lon=50)
    z_coord = dino_lat_lon_vertical(grid, dino_cfg)
    state = dino_lat_lon_state(grid, z_coord, dino_cfg)
    _model_cfg, physics = dino_lat_lon_model_config(grid, dino_cfg)
    cfg = physics.vertical_mixing.tke

    ny, nx, nlev = np.asarray(state.T.data).shape
    n_solve = nlev - 1
    shape = (ny, nx, n_solve)
    iy = jax.numpy.arange(ny, dtype=jax.numpy.float64)[:, None, None]
    ix = jax.numpy.arange(nx, dtype=jax.numpy.float64)[None, :, None]
    iz = jax.numpy.arange(n_solve, dtype=jax.numpy.float64)[None, None, :]

    # Inexact, nonuniform operands make the terminal solved value sensitive to
    # the actual literal recurrence rather than to a coincident floor.
    e0 = jax.numpy.broadcast_to(
        1.13e-4 + 1.7e-8 * iy + 2.3e-8 * ix + 3.1e-7 * iz, shape)
    k_m = jax.numpy.broadcast_to(
        1.23e-3 + 1.9e-8 * iy + 2.9e-8 * ix + 3.7e-7 * iz, shape)
    k_h = jax.numpy.broadcast_to(
        1.73e-4 + 1.1e-9 * iy + 1.3e-9 * ix + 1.7e-8 * iz, shape)
    shear = jax.numpy.broadcast_to(
        1.31e-7 + 1.9e-12 * iy + 2.1e-12 * ix + 2.7e-10 * iz, shape)
    n2 = jax.numpy.broadcast_to(
        1.7e-5 + 2.3e-10 * iy + 3.1e-10 * ix + 4.7e-8 * iz, shape)
    l_eps = jax.numpy.broadcast_to(
        7.3 + 1.3e-3 * iy + 1.7e-3 * ix + 1.9e-2 * iz, shape)
    dz_half = jax.numpy.broadcast_to(
        9.7 + 1.1e-3 * iy + 1.3e-3 * ix + 1.7e-2 * iz, shape)
    dissl = jax.numpy.broadcast_to(
        1.7e-2 + 1.3e-7 * iy + 1.9e-7 * ix + 2.3e-5 * iz, shape)
    e3t = jax.numpy.broadcast_to(
        4.9 + 1.1e-3 * iy + 1.7e-3 * ix
        + 1.3e-2 * jax.numpy.arange(nlev, dtype=jax.numpy.float64)[None, None, :],
        (ny, nx, nlev))
    surface = 2.17e-4 + 1.3e-9 * iy[..., 0] + 1.9e-9 * ix[..., 0]
    k_surface = 3.19e-4 + 1.1e-9 * iy[..., 0] + 1.7e-9 * ix[..., 0]
    bottom = 7.0e-4 + 1.3e-9 * iy[..., 0] + 2.3e-9 * ix[..., 0]
    w_active = jax.numpy.asarray(z_coord.is_active[..., 1:], dtype=bool)
    w_active = w_active & (state.land_mask.data[..., None] > 0.5)
    bottom_level = jax.numpy.asarray(z_coord.bottom_level)

    def solve(e_old):
        return _solve_tke_backward_euler(
            e_old=e_old, K_M_old=k_m, K_H_old=k_h, P_s=shear, N2=n2,
            l_eps=l_eps, dz_half=dz_half,
            surface_flux=jax.numpy.zeros((ny, nx), dtype=jax.numpy.float64),
            dt=jax.numpy.float64(2700.0), cfg=cfg,
            dz_face_surface=jax.numpy.full((ny, nx), 5.3),
            surface_dirichlet=surface, surface_bc_level="nemo_z0",
            K_M_surface=k_surface, bottom_dirichlet=bottom,
            bottom_level=bottom_level, w_active=w_active,
            nemo_e3t=e3t, dissl_old=dissl)

    result = {
        "format": "orca2-round330-dino-tke-effect-v1",
        "grid": ARGS.grid,
        "bottom_census": _bottom_census(
            type("GridHolder", (), {"z_coord": z_coord})(), state),
    }
    if ARGS.mode == "tke-program":
        result["jaxpr"] = str(jax.make_jaxpr(solve)(e0))
    else:
        result["steps"] = []
        value = e0
        compiled = jax.jit(solve)
        for step in (1, 2, 3):
            value = compiled(value)
            array = np.ascontiguousarray(np.asarray(value))
            result["steps"].append({
                "step": step,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "nonzero": int(np.count_nonzero(array)),
                "min": float(array.min()),
                "max": float(array.max()),
            })
    ARGS.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if ARGS.mode.startswith("tke-"):
    _tke_probe()
    raise SystemExit(0)


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

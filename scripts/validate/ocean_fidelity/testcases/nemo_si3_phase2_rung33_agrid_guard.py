#!/usr/bin/env python3
"""Byte guard proving the pre-existing A-grid EVP arm is unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import types
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

_PREREGISTER_COMMIT = "9b369d9d082"
_GRID_N_LAT = 8
_GRID_N_LON = 12
_CARD_DT_S = 1200.0
_CARD_N_EVP = 3
_CARD_CONCENTRATION = 0.8
_CARD_WIND_U_M_S = 3.0
_CARD_WIND_PERTURBATION = 1.0e-4
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_guard(before_ref: str = _PREREGISTER_COMMIT) -> dict[str, object]:
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ice import dynamics as current

    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "show",
            f"{before_ref}:packages/ice/legoesm/ice/dynamics.py",
        ],
        text=True,
    )
    before = types.ModuleType("legoesm_ice_dynamics_before_rung33")
    exec(compile(source, "dynamics-before-rung33.py", "exec"), before.__dict__)
    grid = create_latlon_grid(
        n_lat=_GRID_N_LAT,
        n_lon=_GRID_N_LON,
        dtype=jnp.float64,
    )
    shape = grid.grid_coriolis.shape
    sequence = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    arguments = (
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.ones(shape, dtype=jnp.float64),
        jnp.full(shape, _CARD_CONCENTRATION, dtype=jnp.float64),
        jnp.full(shape, _CARD_WIND_U_M_S, dtype=jnp.float64)
        + sequence * _CARD_WIND_PERTURBATION,
        zero,
        zero,
        zero,
        grid,
        _CARD_DT_S,
    )
    schemes: dict[str, dict[str, str | bool]] = {}
    for name, old_solver, new_solver, keywords in (
        (
            "evp",
            before.evp_solver,
            current.evp_solver,
            {"N_evp": _CARD_N_EVP, "differentiable": False},
        ),
        (
            "mevp",
            before.mevp_solver,
            current.mevp_solver,
            {"N_mevp": _CARD_N_EVP, "differentiable": False},
        ),
    ):
        old_state = old_solver(*arguments, **keywords)
        new_state = new_solver(*arguments, **keywords)
        byte_identical = all(
            np.array_equal(np.asarray(old), np.asarray(new))
            for old, new in zip(old_state, new_state)
        )
        if not byte_identical:
            raise RuntimeError(
                f"existing A-grid {name} state changed after C-grid implementation"
            )
        schemes[name] = {
            "before_sha256": _digest(old_state),
            "after_sha256": _digest(new_state),
            "byte_identical": byte_identical,
        }
    return {
        "format": "nemo-si3-rung33-agrid-guard-v2",
        "worktree": worktree_stamp(),
        "before_ref": before_ref,
        "checkout": str(_REPO_ROOT),
        "schemes": schemes,
        "byte_identical": all(
            bool(result["byte_identical"]) for result in schemes.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-ref", default=_PREREGISTER_COMMIT)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run_guard(args.before_ref), indent=2, sort_keys=True)
    if args.artifact:
        args.artifact.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

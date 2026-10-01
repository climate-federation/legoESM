#!/usr/bin/env python3
"""Regenerate and verify run restarts for Decision 78's switched cards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    nemo_rk3_after_ssh_is_carried,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_nemo_testcase_card,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp
from legoesm.ocean.restart import (
    load_run_restart,
    run_restart_metadata,
    save_run_restart,
)


CASES = ("GYRE-zco", "VORTEX_VEC-zco")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _array_leaves(value) -> list[np.ndarray]:
    leaves = []
    for leaf in jax.tree_util.tree_leaves(value):
        try:
            leaves.append(np.asarray(leaf))
        except (TypeError, ValueError):
            pass
    return leaves


def _state_equal(left, right) -> tuple[bool, int]:
    left_leaves = _array_leaves(left)
    right_leaves = _array_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False, -1
    unequal = 0
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            return False, -1
        unequal += int(np.count_nonzero(lhs != rhs))
    return unequal == 0, unequal


def _sha256(path: Path) -> str:
    handle = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            handle.update(block)
    return handle.hexdigest()


def run_case(case: str, root: Path, *, plant: bool) -> dict:
    card = build_nemo_testcase_card(case)
    config = card.recipe.model_config
    require(nemo_rk3_after_ssh_is_carried(config),
            f"{case}: resolved card does not carry NEMO's after-SSH slot")
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, config)
    entry = card.recipe.initial_state
    first = model.step(entry, dt=card.dt_s)
    continuous = model.step(first, dt=card.dt_s)
    require(first.eta_rk3_after is not None,
            f"{case}: first step did not create eta_rk3_after")

    archive = root / f"{case.lower().replace('-', '_')}_step1_restart.npz"
    save_run_restart(
        archive, first, step=1, time_days=card.dt_s / 86400.0,
        grid_type="latlon", dt_seconds=card.dt_s,
        sha=worktree_stamp()["commit"],
    )
    metadata = run_restart_metadata(archive)
    require(metadata["format"] == 5,
            f"{case}: regenerated archive format is {metadata['format']}, not 5")
    require("eta_rk3_after" in metadata["slots"],
            f"{case}: regenerated archive omits eta_rk3_after")
    resumed, _, loaded = load_run_restart(
        archive, entry, grid_type="latlon", dt_seconds=card.dt_s,
        carries_rk3_after_ssh=nemo_rk3_after_ssh_is_carried(config),
    )
    exact_load, load_unequal = _state_equal(resumed, first)
    require(exact_load, f"{case}: restart load changed {load_unequal} values")

    if plant:
        values = np.asarray(resumed.eta_rk3_after.data).copy()
        values.flat[int(np.argmax(np.abs(values)))] += 1.0e-9
        resumed = resumed._replace(
            eta_rk3_after=resumed.eta_rk3_after.replace(data=jnp.asarray(values)))

    after_resume = model.step(resumed, dt=card.dt_s)
    exact_step, step_unequal = _state_equal(after_resume, continuous)
    if plant:
        require(not exact_step,
                f"{case}: carried-slot plant did not move the resumed step")
    else:
        require(exact_step,
                f"{case}: resumed step changed {step_unequal} values")
    return {
        "case": case,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "format": loaded["format"],
        "slots": sorted(loaded["slots"]),
        "load_cells_unequal": load_unequal,
        "resumed_step_cells_unequal": step_unequal,
        "plant": plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    stamp = worktree_stamp()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case, args.root, plant=args.plant) for case in CASES]
    from nemo_testcase_l2_gyre_decision43_gate import _card_execution
    report = {
        "worktree": stamp,
        "dtype": "float64",
        "rows": rows,
        "card_execution": _card_execution("rk3_after_ssh"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plant:
        print("STATUS PLANT-FIRED: carried after-SSH restart slot")
        return 1
    print("STATUS PASS: 2 regenerated archives, 2 exact loads, 2 exact resumes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

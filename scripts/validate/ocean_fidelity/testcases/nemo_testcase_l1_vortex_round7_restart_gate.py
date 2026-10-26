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


def _state_differences(left, right, *, names=None) -> dict[str, int]:
    differences = {}
    if left._fields != right._fields:
        return {"__fields__": -1}
    for name in left._fields if names is None else names:
        lhs_value, rhs_value = getattr(left, name), getattr(right, name)
        if (lhs_value is None) != (rhs_value is None):
            differences[name] = -1
            continue
        if lhs_value is None:
            continue
        left_leaves = _array_leaves(lhs_value)
        right_leaves = _array_leaves(rhs_value)
        if len(left_leaves) != len(right_leaves):
            differences[name] = -1
            continue
        unequal = 0
        for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
                unequal = -1
                break
            unequal += int(np.count_nonzero(lhs != rhs))
        if unequal:
            differences[name] = unequal
    return differences


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
    continuous_next = model.step(continuous, dt=card.dt_s)
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
    if plant:
        values = np.asarray(resumed.eta_rk3_after.data).copy()
        # Plant the comparison this gate owns: exact recovery of every
        # persisted restart slot.  The production ladder separately proves
        # that selecting the carried arm changes live trajectory rows.
        parity = np.sum(np.indices(values.shape), axis=0) % 2
        values += np.where(parity, 1.0e-3, -1.0e-3)
        resumed = resumed._replace(
            eta_rk3_after=resumed.eta_rk3_after.replace(data=jnp.asarray(values)))
    load_differences = _state_differences(
        resumed, first, names=sorted(metadata["slots"]))
    all_load_differences = _state_differences(resumed, first)
    if plant:
        require(load_differences.get("eta_rk3_after", 0) > 0,
                f"{case}: persisted-slot comparison plant did not fire")
    else:
        require(not load_differences,
                f"{case}: restart load differs by field: {load_differences}")

    after_resume = model.step(resumed, dt=card.dt_s)
    step_differences = _state_differences(after_resume, continuous)
    after_resume_next = model.step(after_resume, dt=card.dt_s)
    next_step_differences = _state_differences(
        after_resume_next, continuous_next)
    exact_resume = not step_differences and not next_step_differences
    if not plant:
        require(exact_resume,
                f"{case}: resumed steps differ by field: "
                f"step2={step_differences}, step3={next_step_differences}")
    return {
        "case": case,
        "archive": str(archive),
        "archive_sha256": _sha256(archive),
        "format": loaded["format"],
        "slots": sorted(loaded["slots"]),
        "load_differences": load_differences,
        "all_load_differences": all_load_differences,
        "resumed_step_differences": step_differences,
        "next_resumed_step_differences": next_step_differences,
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
    try:
        rows = [run_case(case, args.root, plant=args.plant) for case in CASES]
    except GateError as error:
        print(f"STATUS REFUSE: {error}")
        return 2
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
        print("STATUS PLANT-FIRED: persisted after-SSH restart-slot comparison")
        return 1
    print("STATUS PASS: 2 regenerated archives, 2 exact persisted-slot loads, "
          "2 exact two-step resumes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

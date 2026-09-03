#!/usr/bin/env python3
"""First-divergence sweep for the full-size SI3 rung-3.4 card."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import netCDF4
import numpy as np
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import (
    build_ice_rheo_card,
    step_ice_rheo_card,
)

from legoesm import constants

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
HERE = Path(__file__).resolve().parent
KT1_GATE = HERE / "nemo_si3_phase2_rung34_gate.py"
LAST_ENTRY_FRAME = 720


class Rung34TrajectoryError(RuntimeError):
    """Fail-closed trajectory artifact or clock error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rung34TrajectoryError(message)


def _load_gate():
    spec = importlib.util.spec_from_file_location("nemo_si3_rung34_kt1", KT1_GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def run_sweep(root: Path, *, last_frame: int = LAST_ENTRY_FRAME) -> dict[str, object]:
    require(2 <= last_frame <= LAST_ENTRY_FRAME, "last frame outside 2..720")
    frame1_path = root / "oracle_ice_step_entry_kt00000001.bin"
    header, frame1 = gate.oracle_gate.read_frame(frame1_path)
    require(header["kt"] == 1, "initial entry frame clock mismatch")
    with netCDF4.Dataset(root / "mesh_mask.nc") as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in gate.MESH_FIELDS}
    with netCDF4.Dataset(root / "output.init_ice.nc") as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    card = build_ice_rheo_card(frame1, mesh, ocean_temperature_k)
    state = card.initial_state
    boundaries: list[dict[str, object]] = []
    first_divergence: dict[str, object] | None = None

    for completed_steps in range(last_frame - 1):
        state = step_ice_rheo_card(card, state, completed_steps=completed_steps)
        frame_number = completed_steps + 2
        frame_path = root / f"oracle_ice_step_entry_kt{frame_number:08d}.bin"
        require(frame_path.is_file(), f"missing trajectory frame {frame_number}")
        frame_header, oracle = gate.oracle_gate.read_frame(frame_path)
        require(frame_header["kt"] == frame_number, "trajectory frame clock mismatch")
        fields = gate._candidate_fields(card, state)
        rows = [
            gate._score(
                f"step{completed_steps + 1}.{name}",
                gate._oracle_field(oracle, name)[
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                ],
                value[
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                ],
            )
            for name, value in fields.items()
        ]
        worst = max(rows, key=lambda row: float(row["normalized_max_abs"]))
        debts = [row for row in rows if row["status"] != "AT-BAR"]
        boundaries.append(
            {
                "completed_step": completed_steps + 1,
                "oracle_entry_frame": frame_number,
                "worst_row": worst,
                "debt_count": len(debts),
            }
        )
        if debts:
            owner = max(debts, key=lambda row: float(row["normalized_max_abs"]))
            first_divergence = {
                "completed_step": completed_steps + 1,
                "oracle_entry_frame": frame_number,
                "owner": owner,
                "debt_rows": debts,
            }
            break

    status = (
        "DEBT-FIRST-DIVERGENCE" if first_divergence is not None else "AT-BAR-THROUGH-SWEEP"
    )
    return {
        "gate": "nemo-si3-phase2-rung34-first-divergence-v1",
        "status": status,
        "exit_code": 1 if first_divergence is not None else 0,
        "bar": gate.POINTWISE_BAR,
        "cpu_only": True,
        "precision_policy": "fp64",
        "requested_last_entry_frame": last_frame,
        "boundaries": boundaries,
        "first_divergence": first_divergence,
        "unmeasured": {
            "after_first_divergence": "not run: sweep stops at the first over-bar boundary",
            "final_step_720": "requires final restart because no kt=721 entry frame exists",
            "restart_moments": "separate final-restart gate",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--last-frame", type=int, default=LAST_ENTRY_FRAME)
    args = parser.parse_args()
    report = run_sweep(args.root, last_frame=args.last_frame)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())


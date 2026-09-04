#!/usr/bin/env python3
"""First-divergence sweep for the full-size SI3 rung-3.4 card."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import netCDF4
import numpy as np
from legoesm.ice.dynamics import si3_cgrid_deformation
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import (
    build_ice_rheo_card,
    step_ice_rheo_card,
)

from legoesm import constants

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
HERE = Path(__file__).resolve().parent
KT1_GATE = HERE / "nemo_si3_phase2_rung34_gate.py"
LAST_ENTRY_FRAME = 720
_RIDGING_EPSI10 = 1.0e-10  # icedyn_rdgrft.F90:594-595,623-624
_RIDGING_TRAILING_STEPS = 5


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


def _closing_row(
    completed_step: int,
    divergence: np.ndarray,
    deformation: np.ndarray,
    dt_s: float,
) -> dict[str, object]:
    """Replay SI3 EVP closing at icedyn_rdgrft.F90:243-252."""

    require(divergence.shape == deformation.shape, "closing diagnostic shape mismatch")
    require(
        bool(np.all(np.isfinite(divergence)) and np.all(np.isfinite(deformation))),
        "closing diagnostic contains non-finite input",
    )
    closing = 0.5 * 0.5 * (deformation - np.abs(divergence)) - np.minimum(
        divergence, 0.0
    )
    closing = np.where(divergence < 0.0, np.maximum(closing, -divergence), closing)
    opening = closing + divergence
    area_demand = closing * dt_s
    return {
        "completed_step": completed_step,
        "oracle_velocity_entry_frame": completed_step + 1,
        "max_abs_delta_s_inv": float(np.max(np.abs(deformation))),
        "max_opening_s_inv": float(np.max(opening)),
        "max_closing_net_s_inv": float(np.max(closing)),
        "max_closing_net_area_per_step": float(np.max(area_demand)),
        "source_significant": bool(np.max(area_demand) > _RIDGING_EPSI10),
    }


def run_ridging_regime_scan(
    root: Path,
    *,
    last_completed_step: int = LAST_ENTRY_FRAME - 1,
    trailing_steps: int = _RIDGING_TRAILING_STEPS,
) -> dict[str, object]:
    """Find first SI3-source-significant closing demand in oracle frames."""

    require(
        1 <= last_completed_step < LAST_ENTRY_FRAME,
        "last completed step outside 1..719",
    )
    require(trailing_steps >= 0, "trailing step count must be non-negative")
    first_path = root / "oracle_ice_step_entry_kt00000001.bin"
    first_header, current = gate.oracle_gate.read_frame(first_path)
    require(first_header["kt"] == 1, "initial entry frame clock mismatch")
    with netCDF4.Dataset(root / "mesh_mask.nc") as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in gate.MESH_FIELDS}
    with netCDF4.Dataset(root / "output.init_ice.nc") as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    card = build_ice_rheo_card(current, mesh, ocean_temperature_k)
    rows: list[dict[str, object]] = []
    selected_step: int | None = None

    for completed_step in range(1, last_completed_step + 1):
        next_frame_number = completed_step + 1
        next_path = root / f"oracle_ice_step_entry_kt{next_frame_number:08d}.bin"
        require(next_path.is_file(), f"missing trajectory frame {next_frame_number}")
        next_header, following = gate.oracle_gate.read_frame(next_path)
        require(next_header["kt"] == next_frame_number, "trajectory frame clock mismatch")
        dynamics = card.initial_state.dynamics._replace(
            u_ice_u=following["u_ice"],
            v_ice_v=following["v_ice"],
        )
        forcing = card.forcing_template._replace(
            concentration_t=current["a_i"][..., 0]
        )
        divergence, deformation = si3_cgrid_deformation(
            dynamics, forcing, card.metrics, card.dynamics_config
        )
        interior = (
            slice(gate._HALO_WIDTH, -gate._HALO_WIDTH),
            slice(gate._HALO_WIDTH, -gate._HALO_WIDTH),
        )
        row = _closing_row(
            completed_step,
            np.asarray(divergence)[interior],
            np.asarray(deformation)[interior],
            card.dt_s,
        )
        rows.append(row)
        if selected_step is None and row["source_significant"]:
            selected_step = completed_step
        if selected_step is not None and completed_step >= selected_step + trailing_steps:
            break
        current = following

    status = "ACTIVE-REGIME-FOUND" if selected_step is not None else "UNMEASURED"
    return {
        "gate": "nemo-si3-phase2-rung34-ridging-regime-v1",
        "status": status,
        "exit_code": 0 if selected_step is not None else 1,
        "cpu_only": True,
        "precision_policy": "fp64",
        "source_significant_threshold": {
            "predicate": "max(closing_net * rDt_ice) > epsi10",
            "epsi10": _RIDGING_EPSI10,
            "source": "icedyn_rdgrft.F90:243-252,594-595,623-624",
        },
        "selected_completed_step": selected_step,
        "predicted_completed_step": 9,
        "prediction_status": (
            "CONFIRMED" if selected_step == 9 else "REFUTED"
        ),
        "rows": rows,
        "frame_time_level": (
            "entry frame k supplies concentration; entry frame k+1 supplies "
            "the post-rheology velocity for completed step k (icedyn.F90:130-135)"
        ),
    }


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
                uninformative_zero=name in gate.UNINFORMATIVE_ZERO_FIELDS,
            )
            for name, value in fields.items()
        ]
        informative = [row for row in rows if row["status"] != "UNINFORMATIVE"]
        worst = max(informative, key=lambda row: float(row["normalized_max_abs"]))
        debts = [row for row in rows if row["status"] == "DEBT"]
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
    parser.add_argument("--ridging-regime", action="store_true")
    args = parser.parse_args()
    if args.ridging_regime:
        report = run_ridging_regime_scan(
            args.root, last_completed_step=args.last_frame - 1
        )
    else:
        report = run_sweep(args.root, last_frame=args.last_frame)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

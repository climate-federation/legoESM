#!/usr/bin/env python3
"""First-divergence sweep for the full-size SI3 rung-3.4 card."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import netCDF4
import numpy as np
from legoesm.ice.dynamics import si3_cgrid_deformation
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import (
    build_ice_rheo_card,
    step_ice_rheo_card,
)
from legoesm.ice.ridging import SI3JPL1RidgingState, apply_si3_jpl1_ridging
from legoesm.ice.transport import SI3_PRATHER_MOMENT_NAMES, SI3PratherMoments

from legoesm import constants

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
HERE = Path(__file__).resolve().parent
KT1_GATE = HERE / "nemo_si3_phase2_rung34_gate.py"
LAST_ENTRY_FRAME = 720
_RIDGING_EPSI10 = 1.0e-10  # icedyn_rdgrft.F90:594-595,623-624
_RIDGING_TRAILING_STEPS = 5
_ACTIVE_COMPLETED_STEP = 8
_ACTIVE_ENTRY_RESTART_STEP = 7
_ACTIVE_ORDINARY_AND_GEOMETRY_ROWS = 50
_ACTIVE_RESTART_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/round8/oracle_active_restarts"
)


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


def _moment_restart_name(moment: str, tracer: str) -> str:
    """Map the card registry to SI3 restart names (icerst.F90:137-180)."""

    if tracer.startswith("e_s_l"):
        base = "c0_l" + tracer[-2:]
    elif tracer.startswith("e_i_l"):
        base = "e_l" + tracer[-2:]
    elif tracer.startswith("szv_i_l"):
        base = "si_l" + tracer[-2:]
    else:
        base = {
            "v_i": "ice",
            "v_s": "sn",
            "a_i": "a",
            "oa_i": "age",
            "a_ip": "ap",
            "v_ip": "vp",
            "v_il": "vl",
        }.get(tracer)
    require(base is not None, f"unregistered ICE_RHEO moment tracer {tracer}")
    return moment + base


def _restart_xy(dataset: netCDF4.Dataset, name: str) -> np.ndarray:
    require(name in dataset.variables, f"active restart lacks {name}")
    variable = dataset[name]
    require(variable.dimensions[-2:] == ("y", "x"), f"{name}: restart x/y layout changed")
    leading = (0,) * (variable.ndim - 2)
    value = np.asarray(variable[leading]).T
    require(value.dtype == np.float64, f"{name}: active restart is not fp64")
    require(bool(np.all(np.isfinite(value))), f"{name}: active restart is non-finite")
    return value


def _restart_moments(path: Path, card) -> SI3PratherMoments:
    expected = {
        _moment_restart_name(moment, tracer)
        for moment in SI3_PRATHER_MOMENT_NAMES
        for tracer in gate.ICE_RHEO_TRACERS
    }
    with netCDF4.Dataset(path) as dataset:
        discovered = {
            name for name in dataset.variables if name.startswith(("sx", "sy"))
        }
        require(
            discovered == expected,
            "active restart moment roster changed: "
            f"missing={sorted(expected - discovered)}, extra={sorted(discovered - expected)}",
        )
        leaves = []
        for moment in SI3_PRATHER_MOMENT_NAMES:
            physical = np.stack(
                [
                    _restart_xy(dataset, _moment_restart_name(moment, tracer))
                    for tracer in gate.ICE_RHEO_TRACERS
                ],
                axis=-1,
            )
            leaves.append(
                jnp.pad(
                    jnp.asarray(physical),
                    (
                        (gate._HALO_WIDTH, gate._HALO_WIDTH),
                        (gate._HALO_WIDTH, gate._HALO_WIDTH),
                        (0, 0),
                    ),
                    mode="wrap",
                )
            )
    return tuple(leaves)  # type: ignore[return-value]


def _active_ridging_projection(card, frame: dict[str, np.ndarray], next_frame):
    """Isolate redistribution on the selected oracle entry state."""

    dynamics = card.initial_state.dynamics._replace(
        u_ice_u=next_frame["u_ice"], v_ice_v=next_frame["v_ice"]
    )
    forcing = card.forcing_template._replace(concentration_t=frame["a_i"][..., 0])
    divergence, deformation = si3_cgrid_deformation(
        dynamics, forcing, card.metrics, card.dynamics_config
    )
    state = SI3JPL1RidgingState(
        ice_area=jnp.asarray(frame["a_i"][..., 0]),
        open_water_area=1.0 - jnp.asarray(frame["a_i"][..., 0]),
        ice_volume=jnp.asarray(frame["v_i"][..., 0]),
        snow_volume=jnp.asarray(frame["v_s"][..., 0]),
        age_content=jnp.asarray(frame["oa_i"][..., 0]),
        pond_area=jnp.asarray(frame["a_ip"][..., 0]),
        pond_volume=jnp.asarray(frame["v_ip"][..., 0]),
        pond_lid_volume=jnp.asarray(frame["v_il"][..., 0]),
        snow_enthalpy=jnp.asarray(frame["e_s"][..., :, 0]),
        ice_enthalpy=jnp.asarray(frame["e_i"][..., :, 0]),
        ice_salt_content=jnp.asarray(frame["szv_i"][..., :, 0]),
    )
    result, losses = apply_si3_jpl1_ridging(
        state, divergence, deformation, card.dt_s, config=card.ridging_config
    )
    halo = gate._HALO_WIDTH
    interior = (slice(halo, -halo), slice(halo, -halo))
    iterations = np.asarray(losses.iterations)[interior]
    unique, counts = np.unique(iterations, return_counts=True)
    return {
        "scope": "isolated arm on oracle entry state before Prather advection",
        "max_abs_ice_area_change": float(
            np.max(np.abs(np.asarray(result.ice_area - state.ice_area)[interior]))
        ),
        "max_abs_open_water_change": float(
            np.max(
                np.abs(np.asarray(result.open_water_area - state.open_water_area)[interior])
            )
        ),
        "minimum_open_water_before": float(
            np.min(np.asarray(state.open_water_area)[interior])
        ),
        "minimum_open_water_after": float(
            np.min(np.asarray(result.open_water_area)[interior])
        ),
        "minimum_pond_lid_before": float(
            np.min(np.asarray(state.pond_lid_volume)[interior])
        ),
        "minimum_pond_lid_after": float(
            np.min(np.asarray(result.pond_lid_volume)[interior])
        ),
        "iteration_population": {
            str(int(value)): int(count) for value, count in zip(unique, counts, strict=True)
        },
        "max_iterations": int(np.max(iterations)),
    }


def _plant_row_evidence(
    clean_row: dict[str, object], planted_row: dict[str, object]
) -> dict[str, object]:
    """Require the plant to move its scored row, independent of gate exit."""

    require(clean_row["name"] == planted_row["name"], "plant row name changed")
    require(clean_row["status"] == "AT-BAR", "plant baseline row is not AT-BAR")
    require(planted_row["status"] == "DEBT", "plant did not make its row DEBT")
    require(
        float(planted_row["normalized_max_abs"])
        > float(clean_row["normalized_max_abs"]),
        "plant did not increase its scored-row error",
    )
    return {
        "name": clean_row["name"],
        "clean_status": clean_row["status"],
        "clean_normalized_max_abs": clean_row["normalized_max_abs"],
        "planted_status": planted_row["status"],
        "planted_normalized_max_abs": planted_row["normalized_max_abs"],
    }


def run_active_window(
    root: Path,
    restart_root: Path = _ACTIVE_RESTART_ROOT,
    *,
    plant_field: bool = False,
) -> dict[str, object]:
    """Score completed step 8 from oracle state plus restart-carried moments."""

    entry_number = _ACTIVE_COMPLETED_STEP
    target_number = entry_number + 1
    _, entry = gate.oracle_gate.read_frame(
        root / f"oracle_ice_step_entry_kt{entry_number:08d}.bin"
    )
    _, target = gate.oracle_gate.read_frame(
        root / f"oracle_ice_step_entry_kt{target_number:08d}.bin"
    )
    with netCDF4.Dataset(root / "mesh_mask.nc") as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in gate.MESH_FIELDS}
    with netCDF4.Dataset(root / "output.init_ice.nc") as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    card = build_ice_rheo_card(entry, mesh, ocean_temperature_k)
    entry_restart = restart_root / (
        f"ICE_RHEO_OMIP_L3_{_ACTIVE_ENTRY_RESTART_STEP:08d}_restart_ice.nc"
    )
    target_restart = restart_root / (
        f"ICE_RHEO_OMIP_L3_{_ACTIVE_COMPLETED_STEP:08d}_restart_ice.nc"
    )
    require(entry_restart.is_file(), f"missing active entry restart: {entry_restart}")
    require(target_restart.is_file(), f"missing active target restart: {target_restart}")
    state = card.initial_state._replace(moments=_restart_moments(entry_restart, card))
    candidate = step_ice_rheo_card(card, state, completed_steps=_ACTIVE_ENTRY_RESTART_STEP)
    fields = gate._candidate_fields(card, candidate)
    clean_v_s = fields["v_s"].copy()
    if plant_field:
        fields["v_s"] = fields["v_s"].copy()
        fields["v_s"][gate._HALO_WIDTH + 10, gate._HALO_WIDTH + 10] += gate._PLANT_MAGNITUDE

    rows: list[dict[str, object]] = []
    for name, value in zip(
        ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"),
        card.metrics[:8],
        strict=True,
    ):
        rows.append(
            gate._score(
                f"geometry.{name}",
                gate._mesh_xy(mesh[name]),
                np.asarray(value)[
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                ],
            )
        )
    for name, value in (
        ("tmask", card.forcing_template.tmask_t),
        ("umask", card.forcing_template.umask_u),
        ("vmask", card.forcing_template.vmask_v),
    ):
        rows.append(
            gate._score(
                f"geometry.{name}",
                gate._mesh_xy(mesh[name]).astype(bool),
                np.asarray(value)[
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                ].astype(bool),
                exact=True,
            )
        )
    for name, value in fields.items():
        rows.append(
            gate._score(
                f"active.step{_ACTIVE_COMPLETED_STEP}.{name}",
                gate._oracle_field(target, name)[
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                ],
                value[
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                    gate._HALO_WIDTH : -gate._HALO_WIDTH,
                ],
                uninformative_zero=name in gate.UNINFORMATIVE_ZERO_FIELDS,
            )
        )
    with netCDF4.Dataset(target_restart) as dataset:
        for moment_index, moment in enumerate(SI3_PRATHER_MOMENT_NAMES):
            for tracer_index, tracer in enumerate(gate.ICE_RHEO_TRACERS):
                rows.append(
                    gate._score(
                        "active_moment."
                        + _moment_restart_name(moment, tracer),
                        _restart_xy(
                            dataset, _moment_restart_name(moment, tracer)
                        ),
                        np.asarray(candidate.moments[moment_index])[
                            gate._HALO_WIDTH : -gate._HALO_WIDTH,
                            gate._HALO_WIDTH : -gate._HALO_WIDTH,
                            tracer_index,
                        ],
                    )
                )
    require(
        len(rows)
        == _ACTIVE_ORDINARY_AND_GEOMETRY_ROWS + 5 * len(gate.ICE_RHEO_TRACERS),
        "active row loss",
    )
    debts = [row for row in rows if row["status"] == "DEBT"]
    informative = [row for row in rows if row["status"] != "UNINFORMATIVE"]
    owner = max(informative, key=lambda row: float(row["normalized_max_abs"]))
    first_over_bar = debts[0] if debts else None
    plant_evidence: dict[str, object] | None = None
    if plant_field:
        planted_row = next(
            row for row in rows if row["name"] == f"active.step{_ACTIVE_COMPLETED_STEP}.v_s"
        )
        clean_row = gate._score(
            f"active.step{_ACTIVE_COMPLETED_STEP}.v_s",
            gate._oracle_field(target, "v_s")[
                gate._HALO_WIDTH : -gate._HALO_WIDTH,
                gate._HALO_WIDTH : -gate._HALO_WIDTH,
            ],
            clean_v_s[
                gate._HALO_WIDTH : -gate._HALO_WIDTH,
                gate._HALO_WIDTH : -gate._HALO_WIDTH,
            ],
        )
        plant_evidence = _plant_row_evidence(clean_row, planted_row)
    return {
        "gate": "nemo-si3-phase2-rung34-active-window-v1",
        "status": "AT-BAR" if not debts else "DEBT",
        "exit_code": 0 if not debts else 1,
        "bar": gate.POINTWISE_BAR,
        "bar_definition": "max_abs / max(oracle_max_abs, 1.0)",
        "relative_column": "max_abs / oracle_max_abs; diagnostic only",
        "cpu_only": True,
        "precision_policy": "fp64",
        "selected_completed_step": _ACTIVE_COMPLETED_STEP,
        "prediction": {
            "first_over_bar_row": "stress1_i",
            "status": (
                "CONFIRMED"
                if first_over_bar is not None
                and str(first_over_bar["name"]).endswith("stress1_i")
                else "REFUTED"
            ),
        },
        "first_over_bar": first_over_bar,
        "owner": owner,
        "debt_rows": debts,
        "rows": rows,
        "ridging_projection": _active_ridging_projection(card, entry, target),
        "coverage": {
            "ordinary_and_geometry_rows": _ACTIVE_ORDINARY_AND_GEOMETRY_ROWS,
            "moment_rows": 5 * len(gate.ICE_RHEO_TRACERS),
            "uninformative_zero_fields": sorted(gate.UNINFORMATIVE_ZERO_FIELDS),
        },
        "artifacts": {
            str(entry_restart): gate._sha256(entry_restart),
            str(target_restart): gate._sha256(target_restart),
        },
        "plant": {
            "enabled": plant_field,
            "magnitude": gate._PLANT_MAGNITUDE,
            "row_transition": plant_evidence,
        },
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
        "bar_definition": "max_abs / max(oracle_max_abs, 1.0)",
        "relative_column": "max_abs / oracle_max_abs; diagnostic only",
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
    parser.add_argument("--active-window", action="store_true")
    parser.add_argument("--plant-field", action="store_true")
    parser.add_argument("--restart-root", type=Path, default=_ACTIVE_RESTART_ROOT)
    args = parser.parse_args()
    require(
        not (args.ridging_regime and args.active_window),
        "choose at most one specialized rung-3.4 gate",
    )
    if args.active_window:
        report = run_active_window(
            args.root, args.restart_root, plant_field=args.plant_field
        )
    elif args.ridging_regime:
        report = run_ridging_regime_scan(
            args.root, last_completed_step=args.last_frame - 1
        )
    else:
        report = run_sweep(args.root, last_frame=args.last_frame)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

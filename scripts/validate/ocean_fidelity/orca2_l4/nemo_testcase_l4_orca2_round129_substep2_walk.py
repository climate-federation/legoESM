#!/usr/bin/env python3
"""Walk the independent ORCA2 rung-0 substep-2 U residual in source order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round98_coriolis_residual as r98,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)


PLANTS = ("none", "record-bit", "trace-bit", "coefficient-bit")


class GateError(RuntimeError):
    """The admitted record no longer supports the source-order walk."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def first_nonbit(rows: list[dict]) -> dict | None:
    """Return the first non-bit row without aggregating source order."""

    return next((row for row in rows if not row["bit_exact"]), None)


def _score_substep(trace, oracle, active, index: int, *, plant: str) -> list[dict]:
    """Score one external substep in compiled ``dynspg_ts`` statement order."""

    step = index + 1
    prefix = f"j{step:03d}"
    if index == 0:
        entry_names = ("i000_un_e", "i000_vn_e", "i000_sshn_e",
                       "i000_hur_e", "i000_hvr_e")
    else:
        prev = f"j{index:03d}"
        entry_names = (f"{prev}_ua_new", f"{prev}_va_new", f"{prev}_ssha_e",
                       f"{prev}_hur_e", f"{prev}_hvr_e")

    candidate = {
        "entry_u": r97._native_u(trace["u_entry"][index]),
        "entry_v": r97._native_v(trace["v_entry"][index]),
        "entry_ssh": np.asarray(trace["eta_entry"][index]),
        "entry_inverse_u": r97._native_u(trace["inverse_depth_u"][index]),
        "entry_inverse_v": r97._native_v(trace["inverse_depth_v"][index]),
        "mid_u": r97._native_u(trace["u_mid"][index]),
        "mid_v": r97._native_v(trace["v_mid"][index]),
        "mid_ssh": np.asarray(trace["eta_mid"][index]),
        "mid_depth_u": r97._native_u(trace["transport_face_depth_u"][index]),
        "mid_depth_v": r97._native_v(trace["transport_face_depth_v"][index]),
        "transport_u": r97._native_u(trace["transport_metric_u"][index]),
        "transport_v": r97._native_v(trace["transport_metric_v"][index]),
        "after_ssh": np.asarray(trace["eta_continuity"][index]),
        "transport_sum_u": r97._native_u(trace["transport_sum_u_exit"][index]),
        "transport_sum_v": r97._native_v(trace["transport_sum_v_exit"][index]),
        "face_ssh_u": r97._native_u(trace["face_ssh_u_exit"][index]),
        "face_ssh_v": r97._native_v(trace["face_ssh_v_exit"][index]),
        "back_ssh": np.asarray(trace["eta_pgf"][index]),
        "pressure_u": r97._native_u(trace["pgf_u"][index]),
        "pressure_v": r97._native_v(trace["pgf_v"][index]),
        "coriolis_u": r97._native_u(trace["cor_u"][index]),
        "coriolis_v": r97._native_v(trace["cor_v"][index]),
        "trend_u": r97._native_u(trace["trd_u"][index]),
        "trend_v": r97._native_v(trace["trd_v"][index]),
        "exit_u": r97._native_u(trace["u_exit"][index]),
        "exit_v": r97._native_v(trace["v_exit"][index]),
        "exit_depth_u": r97._native_u(trace["face_depth_u_exit"][index]),
        "exit_depth_v": r97._native_v(trace["face_depth_v_exit"][index]),
        "exit_inverse_u": r97._native_u(trace["r1_face_depth_u_exit"][index]),
        "exit_inverse_v": r97._native_v(trace["r1_face_depth_v_exit"][index]),
    }
    reference = {
        "entry_u": oracle[entry_names[0]],
        "entry_v": oracle[entry_names[1]],
        "entry_ssh": oracle[entry_names[2]],
        "entry_inverse_u": oracle[entry_names[3]],
        "entry_inverse_v": oracle[entry_names[4]],
        "mid_u": oracle[f"{prefix}_ua_ext"],
        "mid_v": oracle[f"{prefix}_va_ext"],
        "mid_ssh": oracle[f"{prefix}_sshp2_mid"],
        "mid_depth_u": oracle[f"{prefix}_hup2_e"],
        "mid_depth_v": oracle[f"{prefix}_hvp2_e"],
        "transport_u": oracle[f"{prefix}_zhU"],
        "transport_v": oracle[f"{prefix}_zhV"],
        "after_ssh": oracle[f"{prefix}_ssha_e"],
        "transport_sum_u": oracle[f"{prefix}_un_adv"],
        "transport_sum_v": oracle[f"{prefix}_vn_adv"],
        "face_ssh_u": oracle[f"{prefix}_sshu_a"],
        "face_ssh_v": oracle[f"{prefix}_sshv_a"],
        "back_ssh": oracle[f"{prefix}_sshp2_bck"],
        "pressure_u": oracle[f"{prefix}_zu_spg"],
        "pressure_v": oracle[f"{prefix}_zv_spg"],
        "coriolis_u": oracle[f"{prefix}_cor_u"],
        "coriolis_v": oracle[f"{prefix}_cor_v"],
        "trend_u": oracle[f"{prefix}_trd_u"],
        "trend_v": oracle[f"{prefix}_trd_v"],
        "exit_u": oracle[f"{prefix}_ua_new"],
        "exit_v": oracle[f"{prefix}_va_new"],
        "exit_depth_u": oracle[f"{prefix}_hu_e"],
        "exit_depth_v": oracle[f"{prefix}_hv_e"],
        "exit_inverse_u": oracle[f"{prefix}_hur_e"],
        "exit_inverse_v": oracle[f"{prefix}_hvr_e"],
    }
    faces = {
        "entry_u": "u", "entry_v": "v", "entry_ssh": "t",
        "entry_inverse_u": "u", "entry_inverse_v": "v",
        "mid_u": "u", "mid_v": "v", "mid_ssh": "t",
        "mid_depth_u": "u", "mid_depth_v": "v",
        "transport_u": "u", "transport_v": "v", "after_ssh": "t",
        "transport_sum_u": "u", "transport_sum_v": "v",
        "face_ssh_u": "u", "face_ssh_v": "v", "back_ssh": "t",
        "pressure_u": "u", "pressure_v": "v",
        "coriolis_u": "u", "coriolis_v": "v",
        "trend_u": "u", "trend_v": "v", "exit_u": "u", "exit_v": "v",
        "exit_depth_u": "u", "exit_depth_v": "v",
        "exit_inverse_u": "u", "exit_inverse_v": "v",
    }
    order = tuple(candidate)
    if plant == "trace-bit" and index == 1:
        planted = np.array(candidate["mid_v"], copy=True)
        location = np.argwhere(active["v"])[0]
        planted[tuple(location)] = np.nextafter(
            planted[tuple(location)], np.float64(np.inf))
        candidate["mid_v"] = planted
    return [
        {"substep": step, "boundary": name,
         **rhs_walk.score(candidate[name], reference[name], active[faces[name]])}
        for name in order
    ]


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            coefficient_root: Path, expect_commit: str, *, plant: str) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-129 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-129 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-129 walk requires production JIT on CPU")

    oracle, record_census = r97.assemble_record(
        spg_root, plant="record-bit" if plant == "record-bit" else "none")
    oracle_coeff, coefficient_census = r98.assemble_oracle_coefficients(
        coefficient_root)
    if plant == "coefficient-bit":
        planted = np.array(oracle_coeff["ffu_nw"], copy=True)
        planted[1, 49] = np.nextafter(planted[1, 49], np.float64(np.inf))
        oracle_coeff["ffu_nw"] = planted

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    freshwater, surface = rhs_walk._forcing(state.eta.data.shape)
    slow = (r97._to_model_u(oracle["i000_zu_frc"]),
            r97._to_model_v(oracle["i000_zv_frc"]))
    raw_history = (
        r97._to_model_u(oracle["i000_ub_e"]),
        r97._to_model_u(oracle["i000_ubb_e"]),
        r97._to_model_v(oracle["i000_vb_e"]),
        r97._to_model_v(oracle["i000_vbb_e"]),
        oracle["i000_sshb_e"], oracle["i000_sshbb_e"],
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=slow,
            barotropic_raw_history_override=raw_history,
        ),
    )
    observed = jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    trace = observed.substeps
    require(trace["eta_entry"].shape[0] == 65,
            "production trace does not contain 65 substeps")

    masks = phase3_gate.expected_masks(card)
    active = {
        "t": np.asarray(masks["ssh"], dtype=bool),
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }
    rows = []
    for index in range(2):
        rows.extend(_score_substep(trace, oracle, active, index, plant=plant))

    coefficient_rows = {}
    for name in r98.COEFFICIENTS:
        value = np.asarray(trace[name][0])
        coefficient_rows[name] = rhs_walk.score(
            value, oracle_coeff[name], np.ones_like(value, dtype=bool))
    if plant == "coefficient-bit":
        require(not all(row["bit_exact"] for row in coefficient_rows.values()),
                "coefficient-bit plant stayed green")
        raise GateError("coefficient-bit plant fired")
    require(all(row["bit_exact"] for row in coefficient_rows.values()),
            "round-128 literal EEN coefficients are no longer bit-exact")

    first = first_nonbit(rows)
    if plant == "record-bit":
        require(first is not None, "record-bit plant stayed green")
        raise GateError("record-bit plant fired")
    if plant == "trace-bit":
        require(first is not None, "trace-bit plant stayed green")
        raise GateError("trace-bit plant fired")
    return {
        "status": "MEASURED_R129_SUBSTEP2_WALK",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": record_census,
        "coefficient_census": coefficient_census,
        "coefficient_rows": coefficient_rows,
        "rows": rows,
        "first_non_bit": first,
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--spg-root", type=Path, required=True)
    parser.add_argument("--coefficient-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.frame_root, args.spg_root,
            args.coefficient_root, args.expect_commit, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError, rhs_walk.GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS MEASURED_R129_SUBSTEP2_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

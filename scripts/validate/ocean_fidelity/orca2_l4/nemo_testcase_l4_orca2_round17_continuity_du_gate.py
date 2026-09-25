#!/usr/bin/env python3
"""ORCA2 round-17 gate: localize the substep-2 U-flux difference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _package in ("packages/core", "packages/ocean"):
    if str(REPO_ROOT / _package) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _package))
_TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(_TESTCASES) not in sys.path:
    sys.path.insert(0, str(_TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round14_barotropic_owner_gate as round14,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round15_barotropic_solver_gate as round15,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round16_slow_forcing_gate as round16,
)

_PP = "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
CITATIONS = {
    "continuity_subtraction": f"{_PP}/dynspg_ts.f90:550-558",
    "record_write_order": f"{_PP}/dynspg_ts.f90:755-779",
    "owned_wet_record_view": f"{_PP}/dynspg_ts.f90:1533-1563",
}

INHERITED_COUNT = 64
INHERITED_MAX = 7.705384632572532e-07


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _coordinates(mask: np.ndarray) -> list[list[int]]:
    return [[int(j), int(i)] for j, i in np.argwhere(mask)]


def _histogram(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts)}


def _localize(
    candidate_full_u: np.ndarray,
    candidate_du: np.ndarray,
    oracle_right_u: np.ndarray,
    oracle_du: np.ndarray,
    active_t: np.ndarray,
    *,
    plant: bool = False,
) -> dict[str, object]:
    """Score both operands; the oracle's halo face is equation-inferred."""
    candidate_full_u = np.asarray(candidate_full_u, dtype=np.float64)
    candidate_du = np.asarray(candidate_du, dtype=np.float64)
    oracle_right_u = np.asarray(oracle_right_u, dtype=np.float64)
    oracle_du = np.asarray(oracle_du, dtype=np.float64)
    active_t = np.asarray(active_t, dtype=bool)
    require(candidate_du.shape == oracle_du.shape == active_t.shape,
            "T-point subtraction extents differ")
    ny, nx = candidate_du.shape
    require(candidate_full_u.shape == (ny, nx + 1),
            "candidate U-face extent does not bracket every T cell")
    require(oracle_right_u.shape == candidate_du.shape,
            "recorded right U-face extent differs from T cells")

    candidate_left = candidate_full_u[:, :nx]
    candidate_right = candidate_full_u[:, 1:nx + 1]
    candidate_replay = np.subtract(candidate_right, candidate_left)
    implied_oracle_left = np.subtract(oracle_right_u, oracle_du)
    oracle_replay = np.subtract(oracle_right_u, implied_oracle_left)

    candidate_replay_row = round14.compare(
        candidate_replay, candidate_du, active_t)
    oracle_replay_row = round14.compare(oracle_replay, oracle_du, active_t)
    require(candidate_replay_row["bit_exact"],
            "candidate operands do not replay traced continuity_du")
    require(oracle_replay_row["bit_exact"],
            "recorded subtraction is not invertible to a bit-exact left face")

    mismatch = active_t & (candidate_du != oracle_du)
    coords = np.argwhere(mismatch)
    right_support = round14.compare(
        candidate_right, oracle_right_u, mismatch)
    left_support = round14.compare(
        candidate_left, implied_oracle_left, mismatch)
    left_all = round14.compare(
        candidate_left, implied_oracle_left, active_t)
    left_mismatch = active_t & (candidate_left != implied_oracle_left)
    same_support = bool(np.array_equal(left_mismatch, mismatch))
    support_explained_by_left = bool(
        left_support["differing_cells"]
        == int(np.count_nonzero(mismatch))
        and left_support["absolute_max"]
        == float(np.max(np.abs(candidate_du - oracle_du), initial=0.0))
    )
    du_delta = np.subtract(candidate_du, oracle_du)
    left_delta = np.subtract(candidate_left, implied_oracle_left)
    signed_delta_closure = round14.compare(
        du_delta, np.negative(left_delta), active_t)

    result: dict[str, object] = {
        "continuity_du": round14.compare(candidate_du, oracle_du, active_t),
        "candidate_operand_replay": candidate_replay_row,
        "oracle_inversion_replay": oracle_replay_row,
        "mismatch_coordinates": _coordinates(mismatch),
        "row_histogram": _histogram(coords[:, 0]) if coords.size else {},
        "column_histogram": _histogram(coords[:, 1]) if coords.size else {},
        "all_mismatches_on_west_edge": bool(
            coords.size and np.all(coords[:, 1] == 0)),
        "right_operand_on_support": right_support,
        "inferred_left_operand_on_support": left_support,
        "inferred_left_operand_all_wet_t": left_all,
        "left_and_subtraction_support_identical": same_support,
        "mismatch_support_explained_by_left": support_explained_by_left,
        "signed_delta_closure": signed_delta_closure,
        "oracle_left_operand_provenance": (
            "inferred as recorded_right_u - recorded_continuity_du; "
            "not directly recorded"),
    }

    plant_result = {"requested": plant, "fires": None}
    if plant:
        planted = np.array(candidate_left, copy=True)
        chosen = None
        moved = None
        west_edge = np.zeros_like(active_t)
        west_edge[:, 0] = active_t[:, 0]
        for index in np.argwhere(west_edge & np.isfinite(candidate_left)):
            coordinate = tuple(int(value) for value in index)
            for direction in (np.inf, -np.inf):
                trial = np.array(candidate_left, copy=True)
                trial[coordinate] = np.nextafter(
                    trial[coordinate], direction)
                trial_du = np.subtract(candidate_right, trial)
                trial_moved = active_t & (trial_du != candidate_replay)
                if np.count_nonzero(trial_moved) == 1:
                    planted = trial
                    chosen = coordinate
                    moved = trial_moved
                    break
            if chosen is not None:
                break
        require(chosen is not None and moved is not None,
                "no active one-ULP left-face plant reached the subtraction")
        planted_du = np.subtract(candidate_right, planted)
        movement = round14.compare(planted_du, candidate_replay, active_t)
        fires = bool(
            movement["differing_cells"] == 1
            and bool(moved[chosen])
            and chosen[1] == 0
        )
        plant_result.update({
            "fires": fires,
            "left_face_index": [int(value) for value in chosen],
            "subtraction_movement": movement,
        })
        require(fires, "one-ULP left-face plant did not fire locally")
    result["plant"] = plant_result
    return result


def run(deck_root: Path, root: Path, json_out: Path | None,
        plant: bool = False, _extension=None) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from nemo_testcase_l2_gyre_round14_advmean import read_ordered
    from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    ordered_path = root / "oracle_bt_ordered_operands_kt00000001.bin"
    slow_path = root / "oracle_slow_forcing_kt00000001.bin"
    require(ordered_path.is_file() and slow_path.is_file(),
            "the admitted ordered/slow records are missing")
    oracle = read_ordered(
        ordered_path, expected_dims=round15.RANK0_DIMS, expected_nrows=2)
    slow_record = read_slow_forcing(
        slow_path, dims=(*round15.RANK0_DIMS, 31))

    _, card = ladder.card_fields(deck_root)
    cfg = card.recipe.model_config
    resolved = {
        "dtype_eta": str(np.asarray(card.recipe.initial_state.eta.data).dtype),
        "barotropic_solver": cfg.barotropic.barotropic_solver,
        "n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps,
        "barotropic_time_filter": cfg.barotropic.barotropic_time_filter,
        "barotropic_continuity_evaluation":
            cfg.barotropic.barotropic_continuity_evaluation,
        "barotropic_drag_substep": bool(cfg.barotropic_drag_substep),
        "outer_integrator": cfg.outer_integrator,
    }
    require(
        resolved["barotropic_solver"] == "explicit_substep"
        and resolved["n_barotropic_substeps"] == 65
        and resolved["barotropic_time_filter"] == "nemo_ab3am4"
        and resolved["barotropic_continuity_evaluation"] == "nemo_literal"
        and resolved["barotropic_drag_substep"]
        and resolved["outer_integrator"] == "forward_euler",
        "the resolved ORCA2 solver no longer matches the preregistration",
    )

    entry = ladder.assemble_state_fields(root, 1, stage=None)
    state = round14._seeded_state(card, entry)
    surface_fields = ladder.assemble_surface_fields(root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    masks = round15._masks(card)

    baseline_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True)).step(
                state, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    own_rate_u = -np.asarray(
        baseline_trace.substeps["drag_coefficient_u"][0])
    own_rate_v = -np.asarray(
        baseline_trace.substeps["drag_coefficient_v"][0])
    drag_override = round14._inject(
        own_rate_u, own_rate_v, -slow_record["cd_u"], -slow_record["cd_v"])

    entry_u, entry_v = round14._inject(
        np.asarray(state.uu_b.data), np.asarray(state.vv_b.data),
        oracle["u_entry"][0], oracle["v_entry"][0])
    state_entry = state._replace(
        uu_b=state.uu_b.replace(data=jnp.asarray(entry_u)),
        vv_b=state.vv_b.replace(data=jnp.asarray(entry_v)))

    entry_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_drag_rate_override=drag_override)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    history_override = round16._history_override(entry_trace.substeps, oracle)

    history_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=round16._hooks(
            _NEMOWSRK3TestHooks, expose_trace=True, drag=drag_override,
            history=history_override)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    own_slow_u = np.asarray(history_trace.substeps["slow_u"][0])
    own_slow_v = np.asarray(history_trace.substeps["slow_v"][0])
    recorded_slow = round14._inject(
        own_slow_u, own_slow_v, oracle["slow_u"][0], oracle["slow_v"][0])

    substituted_trace = jax.device_get(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=round16._hooks(
            _NEMOWSRK3TestHooks, expose_trace=True, drag=drag_override,
            history=history_override, slow=recorded_slow)).step(
                state_entry, card.dt_s, freshwater=freshwater,
                surface_forcing=surface))
    candidate = round15._candidate_ordered(card, substituted_trace.substeps)
    inherited_rows, inherited_first = round16._score_walk(
        candidate, oracle, masks)
    require(
        inherited_first is not None
        and inherited_first["substep"] == 2
        and inherited_first["boundary"] == "continuity_du"
        and inherited_first["differing_cells"] == INHERITED_COUNT
        and inherited_first["absolute_max"] == INHERITED_MAX,
        "round 16's continuity_du boundary did not reproduce exactly",
    )
    substep1_exact = all(
        row["bit_exact"] for row in inherited_rows if row["substep"] == 1)
    require(substep1_exact, "substep 1 is no longer bit-exact")

    full_u = np.asarray(
        substituted_trace.substeps["transport_metric_u"][1],
        dtype=np.float64)
    localization = _localize(
        full_u[:, :round15.RANK0_COLUMNS + 1],
        candidate["continuity_du"][1],
        oracle["metric_transport_u"][1],
        oracle["continuity_du"][1],
        masks["t"],
        plant=plant,
    )
    expected_localization = bool(
        localization["all_mismatches_on_west_edge"]
        and localization["right_operand_on_support"]["bit_exact"]
        and localization["mismatch_support_explained_by_left"]
        and localization["continuity_du"]["differing_cells"]
        == INHERITED_COUNT
    )
    status = (
        "STOP_UNRECORDED_HALO_OPERAND" if expected_localization
        else "REFUTED_BOUNDARY_LOCALIZATION"
    )

    result = {
        "gate": "nemo_testcase_l4_orca2_round17_continuity_du_gate",
        "status": status,
        "label": "given NEMO's entry",
        "record_root": str(root),
        "provenance": worktree_stamp(),
        "citations": CITATIONS,
        "resolved": resolved,
        "round16_reproduction": {
            "first_non_bit_after_slow_forcing_substitution": inherited_first,
            "substep1_bit_exact": substep1_exact,
        },
        "localization": localization,
    }
    if _extension is not None:
        result["extension"] = _extension(
            card=card,
            trace=substituted_trace.substeps,
            candidate=candidate,
            oracle=oracle,
            masks=masks,
            inherited_first=inherited_first,
        )
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            args.deck_root, args.record_root, args.json_out, plant=args.plant)
    except GateError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "gate": result["gate"],
        "status": result["status"],
        "round16_reproduction": result["round16_reproduction"],
        "localization": result["localization"],
    }, indent=2, sort_keys=True))
    if args.plant:
        return 1 if result["localization"]["plant"]["fires"] else 2
    return 0 if result["status"] == "STOP_UNRECORDED_HALO_OPERAND" else 3


if __name__ == "__main__":
    raise SystemExit(main())

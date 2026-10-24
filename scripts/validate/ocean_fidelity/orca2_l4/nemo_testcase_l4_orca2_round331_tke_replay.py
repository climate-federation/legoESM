#!/usr/bin/env python3
"""Offline statement-boundary replay of the admitted ORCA2 round-331 TKE record."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round328_dynzdf_replay as r328,
    nemo_testcase_l4_orca2_round331_tke_gate as admission,
)


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def assemble(records: dict[int, dict], name: str) -> np.ndarray:
    """Assemble rank-local NEMO ``(i,j,k)`` slabs as global ``(j,i,k)``."""
    return np.concatenate(
        [records[rank]["fields"][name] for rank in (0, 1)], axis=0
    ).transpose(1, 0, 2)


def score_by_level(actual: np.ndarray, expected: np.ndarray,
                   support: np.ndarray) -> dict[str, object]:
    result = r328.score(actual, expected, support)
    unequal = ((r328.bits(np.asarray(actual)) != r328.bits(np.asarray(expected)))
               & np.asarray(support, dtype=bool))
    result["unequal_by_level"] = r328.vertical_counts(unequal)
    return result


def _run_live_trace(root: Path):
    import jax

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    _, shipped = rung10.card_fields(root)
    card = rung10.comparison_card(shipped)
    surface_fields = rung10.assemble_surface_fields(root, 1)
    freshwater, surface = rung10._surface_forcings(card, root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True),
    )
    traced = jax.device_get(model.step(
        card.recipe.initial_state, card.dt_s, freshwater=freshwater,
        surface_forcing=surface, **rung10.damping_step_kwargs(card, root, 1)))
    require(traced.tke_statement_trace is not None,
            "production trace did not return TKE statement boundaries")
    return card, traced


def _rank_shear_summary(records: dict[int, dict]) -> dict[str, object]:
    rows = {}
    for rank in (0, 1):
        fields = records[rank]["fields"]
        wet = fields["tke_wmask"][:, :, 1:] != 0.0
        values = fields["sh2_output"]
        consumed = fields["tke_sh2_input"]
        rows[str(rank)] = {
            "wet_cell_count": int(np.count_nonzero(wet)),
            "wet_nonzero_count": int(np.count_nonzero(values[wet])),
            "wet_min": float(np.min(values[wet])),
            "wet_max": float(np.max(values[wet])),
            "output_equals_tke_input_bits": bool(np.array_equal(
                r328.bits(values), r328.bits(consumed))),
        }
    return rows


def compare_trace(records: dict[int, dict], card, traced) -> dict[str, object]:
    trace = traced.tke_statement_trace
    state_before = card.recipe.initial_state
    state_after = traced.state_after
    full_wet = assemble(records, "tke_wmask") != 0.0
    interior_wet = full_wet[..., 1:]

    full_rows = {
        "after_boundary_conditions": (
            trace.en_after_boundaries, "after_bc_en", "zdftke.f90:270-305"),
        "after_matrix_rhs": (
            trace.rhs_pre_sweep, "matrix_rhs_en", "zdftke.f90:409-427"),
        "after_solve_and_floor": (
            trace.en_post_sweep, "after_solve_en", "zdftke.f90:458-476"),
    }
    rows: dict[str, object] = {}
    for name, (actual, expected_name, citation) in full_rows.items():
        rows[name] = score_by_level(
            np.asarray(actual, dtype=np.float64), assemble(records, expected_name),
            full_wet)
        rows[name]["nemo_citation"] = citation

    interior_rows = {
        "shear_operand": (
            trace.rhs_shear, "tke_sh2_input", "zdftke.f90:423"),
        "entry_avm": (
            state_before.tke_avm.data, "tke_avm_input", "zdftke.f90:413-416"),
        "entry_dissl": (
            state_before.tke_dissl.data, "tke_dissl_input", "zdftke.f90:420,425"),
        "matrix_upper": (
            trace.matrix_upper, "matrix_upper", "zdftke.f90:413-418"),
        "matrix_lower": (
            trace.matrix_lower, "matrix_lower", "zdftke.f90:415-419"),
        "matrix_diagonal": (
            trace.matrix_diag, "matrix_diag", "zdftke.f90:420"),
        "after_tke_energy": (
            state_after.tke.data, "after_tke_en", "zdftke.f90:498-502"),
        "after_tke_avm": (
            state_after.tke_avm.data, "after_tke_avm", "zdftke.f90:706-711"),
        "after_tke_avt": (
            state_after.tke_avt.data, "after_tke_avt", "zdftke.f90:706-718"),
    }
    for name, (actual, expected_name, citation) in interior_rows.items():
        expected = assemble(records, expected_name)
        if expected_name == "after_tke_en":
            expected = expected[..., 1:]
        elif expected.shape[-1] == full_wet.shape[-1]:
            expected = expected[..., 1:]
        rows[name] = score_by_level(
            np.asarray(actual, dtype=np.float64), expected, interior_wet)
        rows[name]["nemo_citation"] = citation

    ordered = (
        "shear_operand", "entry_avm", "entry_dissl",
        "after_boundary_conditions", "matrix_upper", "matrix_lower",
        "matrix_diagonal", "after_matrix_rhs", "after_solve_and_floor",
        "after_tke_energy", "after_tke_avm", "after_tke_avt",
    )
    first = next((name for name in ordered if not rows[name]["bit_identical"]), None)
    return {
        "backend": __import__("jax").default_backend(),
        "dtype": str(np.asarray(trace.en_post_sweep).dtype),
        "rows": rows,
        "first_non_bit_boundary": first,
    }


def measure(root: Path, zdf_reference_dir: Path, expect_commit: str,
            expect_tree: str) -> dict[str, object]:
    stamp = worktree_stamp(repo=REPO_ROOT)
    require(stamp["commit"] == expect_tree,
            "worktree commit differs from --expect-tree")
    require(stamp["clean"], "replay requires a clean tracked tree")
    admitted = admission.run_gate(root, zdf_reference_dir, expect_commit)
    records = {
        rank: admission.read_record(root / admission.record_name(rank))
        for rank in (0, 1)
    }
    card, traced = _run_live_trace(root)
    return {
        "format": "nemo-testcase-l4-orca2-round331-tke-replay-v1",
        "status": "PASS_R331_TKE_OFFLINE_REPLAY",
        "worktree": stamp,
        "record_admission": admitted,
        "record_shear": _rank_shear_summary(records),
        "comparison": compare_trace(records, card, traced),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--zdf-reference-dir", required=True, type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-tree", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.record_dir, args.zdf_reference_dir,
                         args.expect_commit, args.expect_tree)
    except (ReplayError, admission.GateError, OSError, ValueError) as error:
        print(f"REFUSE: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_R331_TKE_OFFLINE_REPLAY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

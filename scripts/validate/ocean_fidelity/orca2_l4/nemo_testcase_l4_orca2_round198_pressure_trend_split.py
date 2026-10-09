#!/usr/bin/env python3
"""Split rung-0 substep-2 V pressure and EEN-plus-drag trend offline."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round146_boundary_association_gate as r146,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round195_transport_operands as r195,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round197_vector_v_update as r197,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round98_coriolis_residual as r98,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

SUBSTEP = 2
BACK_ORDER = (
    "zb0", "zb1", "zb2", "zb3", "ssha_e", "sshn_e", "sshb_e",
    "sshbb_e",
)
TREND_ORDER = (
    "mid_u", "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne", "cor_v",
    "zCdU_v", "vn_e", "hvr_e", "drag_v", "trd_v",
)
RECORD_FIELDS = (
    "i000_sshn_e", "i000_sshb_e", "i000_zCdU_v",
    "j001_ssha_e", "j001_va_new", "j001_hvr_e",
    "j002_bck_coef", "j002_ssha_e", "j002_sshp2_bck", "j002_zv_spg",
    "j002_ua_ext", "j002_cor_v", "j002_trd_v",
)
PLANTS = (
    "none", "source-order", "coefficient-bit", "drag-identity",
    "missing-stream",
)


class GateError(RuntimeError):
    """The record, replay, source order, or known-answer control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _row(candidate, reference) -> dict[str, object]:
    return r146.exact_row(np.asarray(candidate), np.asarray(reference))


def _active_row(candidate, reference, active) -> dict[str, object]:
    return rhs_walk.score(
        np.asarray(candidate), np.asarray(reference), np.asarray(active, dtype=bool))


def _literal_back(inputs: dict[str, object]) -> dict[str, np.ndarray]:
    """Replay dynspg_ts.f90:646-650 in written source association."""

    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    b = nemo_source_round
    terms = [
        b(jnp.asarray(inputs[f"zb{index}"], dtype=jnp.float64)
          * b(jnp.asarray(inputs[name], dtype=jnp.float64)))
        for index, name in enumerate(("ssha_e", "sshn_e", "sshb_e", "sshbb_e"))
    ]
    partial01 = b(terms[0] + terms[1])
    partial012 = b(partial01 + terms[2])
    result = b(partial012 + terms[3])
    values = jax.device_get((*terms, partial01, partial012, result))
    names = (
        "term0", "term1", "term2", "term3", "partial01", "partial012",
        "sshp2_bck",
    )
    return {name: np.asarray(value) for name, value in zip(names, values, strict=True)}


def _production_pressure(eta, card) -> np.ndarray:
    """Evaluate the production literal pressure-gradient helper."""

    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_barotropic_pressure_gradient,
    )

    state = card.recipe.initial_state
    _, value = _nemo_literal_barotropic_pressure_gradient(
        jnp.asarray(eta, dtype=jnp.float64), card.recipe.grid,
        jnp.asarray(card.recipe.model_config.g, dtype=jnp.float64),
        jnp.asarray(state.u_mask.data, dtype=jnp.float64),
        jnp.asarray(state.v_mask.data, dtype=jnp.float64),
    )
    return r178._native_v(np.asarray(jax.device_get(value)))


def _fold_literal_pressure(eta, card) -> dict[str, np.ndarray]:
    """Replay NEMO's north-neighbour V gradient, including the T-pivot halo."""

    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.grids.operators_latlon_cgrid import fold_ghost_source_T

    b = nemo_source_round
    eta = jnp.asarray(eta, dtype=jnp.float64)
    grid = card.recipe.grid
    fold = grid.fold
    north_fold = fold_ghost_source_T(eta, fold)[:, fold.perm_T]
    north = jnp.concatenate([eta[1:], north_fold], axis=0)
    delta = b(north - eta)
    geom = ensure_geometry(grid)
    r1_e2v = b(1.0 / jnp.asarray(geom.dy_v[1:], dtype=jnp.float64))
    zldg = jnp.asarray(card.recipe.model_config.g, dtype=jnp.float64)
    scaled = b(-zldg * delta)
    pressure = b(scaled * r1_e2v)
    values = jax.device_get((north, delta, r1_e2v, scaled, pressure))
    return {
        name: np.asarray(value) for name, value in zip(
            ("north_ssh", "north_delta", "r1_e2v", "scaled_delta", "zv_spg"),
            values, strict=True)
    }


def _literal_drag(coefficient, velocity, inverse_depth) -> dict[str, np.ndarray]:
    """Replay dynspg_ts.f90:685's left-associated V drag product."""

    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    b = nemo_source_round
    product = b(jnp.asarray(coefficient, dtype=jnp.float64)
                * b(jnp.asarray(velocity, dtype=jnp.float64)))
    drag = b(product * b(jnp.asarray(inverse_depth, dtype=jnp.float64)))
    product, drag = jax.device_get((product, drag))
    return {"product": np.asarray(product), "drag_v": np.asarray(drag)}


def _literal_add(left, right) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    return np.asarray(jax.device_get(nemo_source_round(
        nemo_source_round(jnp.asarray(left, dtype=jnp.float64))
        + nemo_source_round(jnp.asarray(right, dtype=jnp.float64)))))


def split_unit(context: dict[str, object], coefficient_root: Path) -> dict[str, object]:
    """Split pressure and completed trend at external substep two."""

    card = context["card"]
    state = context["state"]
    trace = context["trace"]
    oracle = context["oracle"]
    index = SUBSTEP - 1
    oracle_coeff, coefficient_census = r98.assemble_oracle_coefficients(
        coefficient_root)

    candidate_back = {
        f"zb{n}": np.asarray(trace[f"back_weight_{n}"][index]) for n in range(4)
    }
    candidate_back.update({
        "ssha_e": np.asarray(trace["eta_continuity"][index]),
        "sshn_e": np.asarray(trace["eta_entry"][index]),
        "sshb_e": np.asarray(trace["eta_history_b"][index]),
        "sshbb_e": np.asarray(trace["eta_history_bb"][index]),
    })
    reference_back = {
        f"zb{n}": np.asarray(oracle["j002_bck_coef"][n]) for n in range(4)
    }
    reference_back.update({
        "ssha_e": np.asarray(oracle["j002_ssha_e"]),
        "sshn_e": np.asarray(oracle["j001_ssha_e"]),
        "sshb_e": np.asarray(oracle["i000_sshn_e"]),
        "sshbb_e": np.asarray(oracle["i000_sshb_e"]),
    })
    candidate_back_terms = _literal_back(candidate_back)
    reference_back_terms = _literal_back(reference_back)
    target_back = np.asarray(oracle["j002_sshp2_bck"])
    candidate_pressure = _production_pressure(
        candidate_back_terms["sshp2_bck"], card)
    reference_pressure_production = _production_pressure(target_back, card)
    candidate_pressure_fold = _fold_literal_pressure(
        candidate_back_terms["sshp2_bck"], card)
    reference_pressure_fold = _fold_literal_pressure(target_back, card)
    target_pressure = np.asarray(oracle["j002_zv_spg"])

    pressure_rows = {
        "back_inputs": {
            name: _row(candidate_back[name], reference_back[name])
            for name in BACK_ORDER
        },
        "back_terms": {
            name: _row(candidate_back_terms[name], reference_back_terms[name])
            for name in candidate_back_terms
        },
        "candidate_back_replay_vs_trace": _row(
            candidate_back_terms["sshp2_bck"], trace["eta_pgf"][index]),
        "reference_back_replay_vs_target": _row(
            reference_back_terms["sshp2_bck"], target_back),
        "candidate_back_vs_target": _row(
            candidate_back_terms["sshp2_bck"], target_back),
        "candidate_pressure_replay_vs_trace": _row(
            candidate_pressure, r178._native_v(trace["pgf_v"][index])),
        "candidate_pressure_vs_target": _row(candidate_pressure, target_pressure),
        "reference_ssh_production_pressure_vs_target": _row(
            reference_pressure_production, target_pressure),
        "reference_ssh_fold_pressure_vs_target": _row(
            reference_pressure_fold["zv_spg"], target_pressure),
        "candidate_fold_pressure_vs_target": _row(
            candidate_pressure_fold["zv_spg"], target_pressure),
    }

    candidate_coeff = {
        name: np.asarray(trace[name][index]) for name in r98.COEFFICIENTS
    }
    coefficient_rows = {
        name: _row(candidate_coeff[name], oracle_coeff[name])
        for name in candidate_coeff if name.startswith("ffv_")
    }
    candidate_mid_u = np.asarray(trace["u_mid"][index])
    reference_mid_u = r178._to_model_u(np.asarray(oracle["j002_ua_ext"]))
    candidate_cor_u, candidate_cor_v, _ = r98.strict_application(
        candidate_mid_u, trace["v_mid"][index], candidate_coeff)
    reference_cor_u, reference_cor_v, _ = r98.strict_application(
        reference_mid_u, trace["v_mid"][index], oracle_coeff)
    del candidate_cor_u, reference_cor_u
    target_cor = np.asarray(oracle["j002_cor_v"])

    candidate_drag_inputs = {
        "zCdU_v": r178._native_v(trace["drag_coefficient_v"][index]),
        "vn_e": r178._native_v(trace["v_entry"][index]),
        "hvr_e": r178._native_v(trace["inverse_depth_v"][index]),
    }
    reference_drag_inputs = {
        "zCdU_v": np.asarray(oracle["i000_zCdU_v"]),
        "vn_e": np.asarray(oracle["j001_va_new"]),
        "hvr_e": np.asarray(oracle["j001_hvr_e"]),
    }
    candidate_drag = _literal_drag(**{
        "coefficient": candidate_drag_inputs["zCdU_v"],
        "velocity": candidate_drag_inputs["vn_e"],
        "inverse_depth": candidate_drag_inputs["hvr_e"],
    })
    reference_drag = _literal_drag(**{
        "coefficient": reference_drag_inputs["zCdU_v"],
        "velocity": reference_drag_inputs["vn_e"],
        "inverse_depth": reference_drag_inputs["hvr_e"],
    })
    target_trend = np.asarray(oracle["j002_trd_v"])
    derived_drag = target_trend - target_cor
    candidate_trend = _literal_add(candidate_cor_v, candidate_drag["drag_v"])
    reference_trend = _literal_add(reference_cor_v, reference_drag["drag_v"])
    masks = phase3_gate.expected_masks(card)
    active_v = np.asarray(masks["v"][..., 0], dtype=bool)
    trend_rows = {
        "mid_u": _row(r178._native_u(candidate_mid_u), oracle["j002_ua_ext"]),
        "coefficients": coefficient_rows,
        "candidate_cor_replay_vs_trace": _row(
            candidate_cor_v, r178._native_v(trace["cor_v"][index])),
        "reference_cor_replay_vs_target": _row(reference_cor_v, target_cor),
        "candidate_cor_vs_target": _row(candidate_cor_v, target_cor),
        "candidate_cor_active_vs_target": _active_row(
            candidate_cor_v, target_cor, active_v),
        "drag_inputs": {
            name: _row(candidate_drag_inputs[name], reference_drag_inputs[name])
            for name in ("zCdU_v", "vn_e", "hvr_e")
        },
        "candidate_drag_replay_vs_trace": _row(
            candidate_drag["drag_v"], r178._native_v(trace["drag_v"][index])),
        "reference_trend_replay_vs_target": _row(reference_trend, target_trend),
        "derived_drag_control": _row(reference_drag["drag_v"], derived_drag),
        "candidate_drag_vs_reference": _row(
            candidate_drag["drag_v"], reference_drag["drag_v"]),
        "candidate_trend_vs_target": _row(candidate_trend, target_trend),
    }

    vector = r197.split_update(context)
    return {
        "back_order": list(BACK_ORDER),
        "trend_order": list(TREND_ORDER),
        "coefficient_census": coefficient_census,
        "pressure": pressure_rows,
        "trend": trend_rows,
        "vector_pair_close": vector["cumulative_substitution_post_v"]["zv_trd"],
        "vector_pressure_only": vector["single_substitution_post_v"]["zv_spg"],
        "vector_trend_only": vector["single_substitution_post_v"]["zv_trd"],
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    """Validate the frozen split and exercise the fail-closed controls."""

    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    split = report["split"]
    if plant == "source-order":
        split["back_order"][0], split["back_order"][1] = (
            split["back_order"][1], split["back_order"][0])
    elif plant == "coefficient-bit":
        split["pressure"]["back_inputs"]["zb0"]["bit_exact"] = False
        split["pressure"]["back_inputs"]["zb0"]["differing_cells"] = 1
    elif plant == "drag-identity":
        split["trend"]["derived_drag_control"]["bit_exact"] = True
        split["trend"]["derived_drag_control"]["differing_cells"] = 0
    elif plant == "missing-stream":
        report["record_fields"]["j002_sshp2_bck"] = False

    require(report.get("claim_label") == "independent hierarchy rung 0",
            "claim label moved")
    require(report.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(report.get("record_census") == {
        "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
    }, "record census moved")
    require(report.get("record_fields") == {name: True for name in RECORD_FIELDS},
            "required pressure/trend stream is absent")
    require(all(report["passivity"].values()), "substep trace is not passive")
    require(split["back_order"] == list(BACK_ORDER), "back source registry reordered")
    require(split["trend_order"] == list(TREND_ORDER), "trend source registry reordered")
    pressure = split["pressure"]
    require(pressure["candidate_back_replay_vs_trace"]["bit_exact"],
            "candidate back interpolation replay moved")
    require(pressure["reference_back_replay_vs_target"]["bit_exact"],
            "recorded back interpolation replay does not close")
    require(pressure["candidate_pressure_replay_vs_trace"]["bit_exact"],
            "candidate pressure replay moved")
    require(pressure["back_inputs"]["zb0"]["bit_exact"],
            "back coefficient bit plant fired")
    trend = split["trend"]
    require(trend["candidate_cor_replay_vs_trace"]["bit_exact"],
            "candidate EEN replay moved")
    require(trend["reference_cor_replay_vs_target"]["bit_exact"],
            "recorded EEN replay does not close")
    require(trend["candidate_drag_replay_vs_trace"]["bit_exact"],
            "candidate drag replay moved")
    require(trend["reference_trend_replay_vs_target"]["bit_exact"],
            "recorded trend replay does not close")
    require(not trend["derived_drag_control"]["bit_exact"],
            "derived-drag subtraction control is vacuous")
    require(split["vector_pair_close"]["bit_exact"],
            "pressure plus trend no longer closes the vector update")
    require(not split["vector_pressure_only"]["bit_exact"],
            "pressure-only substitution unexpectedly closes")
    require(not split["vector_trend_only"]["bit_exact"],
            "trend-only substitution unexpectedly closes")

    first_back = next(
        (name for name in BACK_ORDER
         if not pressure["back_inputs"][name]["bit_exact"]), None)
    first_trend = next(
        (name for name in ("mid_u",)
         if not trend[name]["bit_exact"]), None)
    if first_trend is None:
        first_trend = next(
            (name for name, row in trend["coefficients"].items()
             if not row["bit_exact"]), None)
    if first_trend is None and not trend["candidate_cor_vs_target"]["bit_exact"]:
        first_trend = "cor_v"
    if first_trend is None:
        first_trend = next(
            (name for name in ("zCdU_v", "vn_e", "hvr_e")
             if not trend["drag_inputs"][name]["bit_exact"]), None)
    if first_trend is None and not trend["candidate_drag_vs_reference"]["bit_exact"]:
        first_trend = "drag_v"
    report["first_nonbit_back_input"] = first_back
    report["first_nonbit_trend_input"] = first_trend
    report["prediction_ledger"] = {
        "R198-P1": "CONFIRMED_RECORD_SUFFICIENT",
        "R198-P2": (
            "CONFIRMED_AFTER_SSH_FIRST" if first_back == "ssha_e"
            else f"REFUTED_FIRST_{str(first_back).upper()}"),
        "R198-P3": (
            "CONFIRMED_REFERENCE_SSH_CLOSES_PRODUCTION_GRADIENT"
            if pressure["reference_ssh_production_pressure_vs_target"]["bit_exact"]
            else "REFUTED_PRODUCTION_GRADIENT_RESIDUAL"),
        "R198-P4": (
            "CONFIRMED_DRAG_FIRST_ACTIVE"
            if trend["candidate_cor_active_vs_target"]["bit_exact"]
            and first_trend not in ("mid_u", "cor_v")
            else f"REFUTED_FIRST_{str(first_trend).upper()}"),
        "R198-P5": "CONFIRMED_PAIR_CLOSES_SINGLES_DO_NOT",
        "R198-P6": "CONFIRMED_CONTROLS_BIND",
    }
    report["status"] = "PASS_R198_PRESSURE_TREND_SPLIT"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            coefficient_root: Path, expect_commit: str) -> dict[str, object]:
    """Run the passive context and split only the registered cancelling pair."""

    context = r195.measurement_context(
        deck_root, frame_root, spg_root, expect_commit)
    oracle = context["oracle"]
    record_fields = {name: name in oracle for name in RECORD_FIELDS}
    require(all(record_fields.values()),
            "round-96 record lacks a required pressure/trend stream")
    return classify({
        "format": "nemo-testcase-l4-orca2-round198-pressure-trend-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": context["stamp"],
        "record_census": {
            "coverage": "exactly-once", "rank_records": 2, "substeps": 65,
        },
        "record_fields": record_fields,
        "passivity": context["passivity"],
        "split": split_unit(context, coefficient_root),
        "compiled_sources": [
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-660",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:663-702",
            "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727",
        ],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--coefficient-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in:
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.coefficient_root, args.expect_commit)),
                    "runtime inputs are incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.coefficient_root, args.expect_commit)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r195.GateError, r146.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R198_PRESSURE_TREND_SPLIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

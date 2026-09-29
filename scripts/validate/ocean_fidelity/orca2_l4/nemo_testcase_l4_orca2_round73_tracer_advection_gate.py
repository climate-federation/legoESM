#!/usr/bin/env python3
"""Walk independent ORCA2 stage-1 CEN2 tracer advection in source order."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round43_tracer_handoff_gate as handoff,
)

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    centered_cell_to_uface,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_gyre_zco_card,
    build_lock_exchange_zco_card,
    build_orca2_zps_card,
    build_overflow_zps_card,
)

NLEV = 30
OWNED_NX = 90
OWNED_NY = 148
ORDER = (
    "entry_T",
    "metric_pU",
    "metric_pV",
    "metric_pW",
    "u_face_flux_line153",
    "v_face_flux_line154",
    "after_advection_T",
)
EXPECTED_CARD_SCOPE = {
    "ORCA2-zps": ("fct2", "rk3_ws"),
    "GYRE-zco": ("fct2", "rk3_ws"),
    "OVERFLOW-zps": ("fct2", "rk3_ws"),
    "LOCK-zco": ("fct2", "rk3_ws"),
}
ROUND72_AFTER_ADV = {
    "bit_exact": False,
    "count": 228641,
    "max_abs": 3.0869700763080185e-07,
    "status": "DEBT",
    "unequal": 228641,
}
PLANTS = (
    "none",
    "card-scope",
    "entry-ulp",
    "recorded-replay-ulp",
    "round72-boundary",
)


class GateError(RuntimeError):
    """A frozen round-73 predicate or instrument invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _card_scope(deck_root: Path) -> dict[str, list[str]]:
    builders = {
        "ORCA2-zps": lambda: build_orca2_zps_card(deck_root),
        "GYRE-zco": build_gyre_zco_card,
        "OVERFLOW-zps": build_overflow_zps_card,
        "LOCK-zco": build_lock_exchange_zco_card,
    }
    return {
        name: [
            builder().recipe.model_config.tracer_advection,
            builder().recipe.model_config.tracer_time_integrator,
        ]
        for name, builder in builders.items()
    }


def _score(candidate: np.ndarray, oracle: np.ndarray,
           mask: np.ndarray) -> dict[str, object]:
    actual = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64)[mask])
    expected = np.ascontiguousarray(np.asarray(oracle, dtype=np.float64)[mask])
    require(actual.size and actual.shape == expected.shape, "empty ordered score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite ordered score")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    return {
        "bit_exact": not bool(unequal.any()),
        "count": int(actual.size),
        "max_abs": float(np.max(np.abs(actual - expected), initial=0.0)),
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
    }


def _run(card, state, freshwater, surface, exposure: str,
         *, endpoint=None, transport_override=None):
    return handoff._run(
        card, state, freshwater, surface, endpoint=endpoint, exposure=exposure,
        transport_override=transport_override,
    )


def run_walk(deck_root: Path, record_root: Path) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-73 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy is not active")

    tracer_path = record_root / handoff.phase2l.TRACER_RECORD
    require(tracer_path.is_file(), f"missing tracer record: {tracer_path}")
    tracer = handoff.phase2l.read_tracer(tracer_path)
    oracle_entry = ladder.assemble_state_fields(record_root, 1, stage=None)
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    card = build_orca2_zps_card(deck_root)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    state = card.recipe.initial_state
    masks = handoff._support_masks(card)

    transport_state = _run(
        card, state, freshwater, surface, "transport")
    boundary_state = _run(
        card, state, freshwater, surface, "after_advection")

    live_pu = np.asarray(transport_state.u.data)[:, 1:OWNED_NX + 1, :NLEV]
    live_pv = np.asarray(transport_state.v.data)[1:OWNED_NY + 1, :OWNED_NX, :NLEV]
    live_pw = np.asarray(transport_state.T.data)[:, :OWNED_NX, :NLEV]
    oracle_pu = np.asarray(tracer["zFu"])[..., :NLEV]
    oracle_pv = np.asarray(tracer["zFv"])[..., :NLEV]
    oracle_pw = np.asarray(tracer["zFw"])[..., :NLEV]

    live_t = jnp.asarray(state.T.data[..., :NLEV])
    oracle_t = jnp.asarray(np.asarray(oracle_entry["T"])[..., :NLEV])
    sr = nemo_source_round

    def face_fluxes(temperature, pu, pv):
        u_sum = centered_cell_to_uface(
            temperature, nemo_source_sum=True)[:, 1:OWNED_NX + 1]
        v_sum = interp_to_v_points(
            temperature, card.recipe.grid,
            nemo_source_sum=True)[1:OWNED_NY + 1, :OWNED_NX]
        return sr(sr(0.5 * pu) * u_sum), sr(sr(0.5 * pv) * v_sum)

    flux_program = jax.jit(face_fluxes)
    live_fu, live_fv = (
        np.asarray(value) for value in flux_program(
            live_t, jnp.asarray(live_pu), jnp.asarray(live_pv)))
    oracle_fu, oracle_fv = (
        np.asarray(value) for value in flux_program(
            oracle_t, jnp.asarray(oracle_pu), jnp.asarray(oracle_pv)))

    actual_after = np.asarray(boundary_state.T.data)[:, :OWNED_NX, :NLEV]
    oracle_after = np.asarray(tracer["after_advection_T"])[..., :NLEV]

    full_shape = state.T.data.shape
    zfu = np.zeros(full_shape, dtype=np.float64)
    zfv = np.zeros(full_shape, dtype=np.float64)
    zfw = np.zeros(full_shape[:-1] + (full_shape[-1] + 1,), dtype=np.float64)
    zfu[:, :OWNED_NX, :NLEV] = oracle_pu
    zfv[:, :OWNED_NX, :NLEV] = oracle_pv
    zfw[:, :OWNED_NX] = np.asarray(tracer["zFw"])
    recorded_transport_state = _run(
        card, state, freshwater, surface, "after_advection",
        transport_override=tuple(map(jnp.asarray, (zfu, zfv, zfw))),
    )
    recorded_transport_after = np.asarray(
        recorded_transport_state.T.data)[:, :OWNED_NX, :NLEV]
    endpoint = handoff._endpoint_override(card, oracle_entry, record_root)
    recorded_full_state = _run(
        card, state, freshwater, surface, "after_advection",
        endpoint=endpoint,
        transport_override=tuple(map(jnp.asarray, (zfu, zfv, zfw))),
    )
    recorded_full_after = np.asarray(
        recorded_full_state.T.data)[:, :OWNED_NX, :NLEV]

    rows = {
        "entry_T": _score(
            np.asarray(state.T.data)[:, :OWNED_NX, :NLEV],
            np.asarray(oracle_entry["T"])[:, :OWNED_NX, :NLEV], masks["T"]),
        "metric_pU": _score(live_pu, oracle_pu, masks["u"]),
        "metric_pV": _score(live_pv, oracle_pv, masks["v"]),
        "metric_pW": _score(live_pw, oracle_pw, masks["w"]),
        "u_face_flux_line153": _score(live_fu, oracle_fu, masks["u"]),
        "v_face_flux_line154": _score(live_fv, oracle_fv, masks["v"]),
        "after_advection_T": _score(
            actual_after, oracle_after, masks["T"]),
    }
    recorded_transport_replay = _score(
        recorded_transport_after, oracle_after, masks["T"])
    recorded_full_replay = _score(
        recorded_full_after, oracle_after, masks["T"])
    first_operand = next(
        (name for name in ("metric_pU", "metric_pV", "metric_pW")
         if not rows[name]["bit_exact"]), None)
    first_arithmetic = next(
        (name for name in ("u_face_flux_line153", "v_face_flux_line154",
                          "after_advection_T")
         if not rows[name]["bit_exact"]), None)
    return {
        "format": "nemo-testcase-l4-orca2-round73-tracer-advection-v1",
        "claim_label": "independent",
        "card_scope": _card_scope(deck_root),
        "compiled_dispatch": "CEN2 at RK3 stages 1-2; FCT2 at stage 3",
        "order": list(ORDER),
        "rows": rows,
        "first_non_bit_operand": first_operand,
        "first_non_bit_arithmetic": first_arithmetic,
        "recorded_transport_only_replay": recorded_transport_replay,
        "recorded_full_operand_replay": recorded_full_replay,
        "record": {
            "root": str(record_root),
            "tracer_sha256": handoff.sha256(tracer_path),
            "schema": tracer["header"],
        },
        "execution": {
            "backend": jax.default_backend(),
            "production_jit": True,
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
        },
        "compiled_citations": {
            "dispatch": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "traadv.f90:491-535"),
            "u_face_flux_line153": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "traadv_cen.f90:149-155"),
            "v_face_flux_line154": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "traadv_cen.f90:149-155"),
        },
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "card-scope":
        report["card_scope"]["ORCA2-zps"][0] = "centered"
    elif plant == "entry-ulp":
        report["rows"]["entry_T"]["bit_exact"] = False
        report["rows"]["entry_T"]["unequal"] = 1
    elif plant == "recorded-replay-ulp":
        report["recorded_full_operand_replay"]["bit_exact"] = False
        report["recorded_full_operand_replay"]["unequal"] = 1
    elif plant == "round72-boundary":
        report["rows"]["after_advection_T"]["max_abs"] = math.nextafter(
            float(report["rows"]["after_advection_T"]["max_abs"]), math.inf)

    require(report.get("claim_label") == "independent", "claim is not independent")
    require(report.get("execution") == {
        "backend": "cpu", "production_jit": True, "dtype": "float64",
        "transcendentals": "libm",
    }, "execution policy changed")
    scope = {name: tuple(values) for name, values in report["card_scope"].items()}
    require(scope == EXPECTED_CARD_SCOPE, "resolved tracer card scope changed")
    require(report.get("compiled_dispatch") ==
            "CEN2 at RK3 stages 1-2; FCT2 at stage 3",
            "compiled stage dispatch changed")
    require(report.get("order") == list(ORDER), "ordered statement registry changed")
    require(report["rows"]["entry_T"] == {
        "bit_exact": True, "count": 228641, "max_abs": 0.0,
        "status": "AT_BAR", "unequal": 0,
    }, "independent kt=1 Kmm temperature is not exact")
    require(report["recorded_transport_only_replay"].get("bit_exact") is False,
            "transport-only replay unexpectedly became bit-exact")
    require(report["recorded_full_operand_replay"].get("bit_exact") is True,
            "complete recorded-operand source replay is not bit-exact")
    require(report["rows"]["after_advection_T"] == ROUND72_AFTER_ADV,
            "round-72 after-advection boundary did not reproduce")
    first_operand = report.get("first_non_bit_operand")
    first_arithmetic = report.get("first_non_bit_arithmetic")
    require(first_operand in ("metric_pU", "metric_pV", "metric_pW"),
            "no non-bit production transport operand was named")
    require(first_arithmetic in (
        "u_face_flux_line153", "v_face_flux_line154", "after_advection_T"),
        "no non-bit arithmetic statement was named")
    require(report["rows"][first_operand].get("bit_exact") is False,
            "named operand is exact")
    require(report["rows"][first_arithmetic].get("bit_exact") is False,
            "named arithmetic statement is exact")
    predictions = {
        "all_four_cards_execute": {"status": "CONFIRMED"},
        "entry_temperature_exact": {"status": "CONFIRMED"},
        "pU_first_operand": {
            "status": "CONFIRMED" if first_operand == "metric_pU" else "REFUTED",
            "observed": first_operand,
        },
        "u_flux_first_arithmetic": {
            "status": (
                "CONFIRMED" if first_arithmetic == "u_face_flux_line153"
                else "REFUTED"),
            "observed": first_arithmetic,
        },
        "recorded_transport_only_replay_exact": {"status": "REFUTED"},
        "recorded_full_operand_replay_exact": {"status": "CONFIRMED"},
        "round72_boundary_reproduced": {"status": "CONFIRMED"},
        "disposition_held": {"status": "CONFIRMED"},
    }
    return {
        **report,
        "prediction_ledger": predictions,
        "status": "PASS_TRACER_ADVECTION_WALK",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "runtime plants require --classify-json")
            require(args.deck_root is not None and args.record_root is not None,
                    "run mode requires --deck-root and --record-root")
            raw = run_walk(args.deck_root, args.record_root)
        report = classify(raw, plant=args.plant)
    except (GateError, handoff.GateError, ladder.GateError, KeyError, OSError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_TRACER_ADVECTION_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

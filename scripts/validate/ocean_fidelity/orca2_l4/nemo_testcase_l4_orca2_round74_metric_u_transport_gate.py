#!/usr/bin/env python3
"""Walk independent ORCA2 stage-1 metric-U transport in source order."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_stage1_transport_gate as record_gate,
)

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
    _nemo_metric_stage_transport,
    _nemo_stage_corrected_velocity,
    _nemo_ws_qco_stage_faces,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_gyre_zco_card,
    build_lock_exchange_zco_card,
    build_orca2_zps_card,
    build_overflow_zps_card,
)
from legoesm.ocean.vertical import compute_layer_thickness

NLEV = 30
OWNED_NX = 90
OWNED_NY = 148
ORDER = (
    "un_adv",
    "inverse_depth",
    "uu_b_Kmm",
    "zub",
    "uu_Kmm",
    "umask",
    "corrected_velocity",
    "e2u",
    "e3u_Kmm",
    "metric_thickness",
    "zFu",
)
DERIVED_ORDER = ("zub", "corrected_velocity", "metric_thickness", "zFu")
EXPECTED_CARD_SCOPE = {
    "ORCA2-zps": ("rk3_ws", "rk3_ws"),
    "GYRE-zco": ("rk3_ws", "rk3_ws"),
    "OVERFLOW-zps": ("rk3_ws", "rk3_ws"),
    "LOCK-zco": ("rk3_ws", "rk3_ws"),
}
ROUND73_ZFU = {
    "count": 251670,
    "max_abs": 321212.60790659266,
    "status": "DEBT",
    "unequal": 221640,
}
PLANTS = (
    "none",
    "card-scope",
    "record-replay",
    "exposure-calibration",
    "first-classification",
    "round73-boundary",
)


class GateError(RuntimeError):
    """A frozen round-74 predicate or instrument invariant failed."""


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
            builder().recipe.model_config.momentum_time_integrator,
            builder().recipe.model_config.tracer_time_integrator,
        ]
        for name, builder in builders.items()
    }


def _step_operand(card, freshwater, surface, operand: str):
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_stage1_transport_operand=operand),
    )
    return model.step(
        card.recipe.initial_state,
        dt=card.dt_s,
        freshwater=freshwater,
        surface_forcing=surface,
    )


def _step_transport(card, freshwater, surface):
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_tracer_transport_stage=1),
    )
    return model.step(
        card.recipe.initial_state,
        dt=card.dt_s,
        freshwater=freshwater,
        surface_forcing=surface,
    )


def _score(candidate: np.ndarray, oracle: np.ndarray,
           mask: np.ndarray) -> dict[str, object]:
    return record_gate.score(candidate, oracle, mask)


def run_walk(deck_root: Path, record_root: Path) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-74 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy is not active")

    record_path = (
        record_root / "oracle_rkstage1_transport_operands_kt00000001.bin")
    oracle = record_gate.read_record(record_path)
    replay = record_gate.validate(record_root, plant=False)
    card = build_orca2_zps_card(deck_root)
    state = card.recipe.initial_state
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)

    thickness_state = _step_operand(
        card, freshwater, surface, "thickness")
    corrected_state = _step_operand(
        card, freshwater, surface, "corrected_velocity")
    average_state = _step_operand(
        card, freshwater, surface, "transport_average")
    transport_state = _step_transport(card, freshwater, surface)

    active_t = card.recipe.z_coord.is_active
    u_mask_full, _ = compute_face_masks_3d(active_t, card.recipe.grid)
    u_mask_full = jnp.asarray(u_mask_full, dtype=jnp.float64)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data),
        state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    h_u, _, _, _, r1_hu, _ = _nemo_ws_qco_stage_faces(
        state.eta.data,
        h_ref,
        u_mask_full,
        jnp.asarray(compute_face_masks_3d(
            active_t, card.recipe.grid)[1], dtype=jnp.float64),
        card.recipe.grid,
        include_reciprocals=True,
    )
    live_un_adv = average_state.u.data[..., 0]
    live_corrected, live_zub = _nemo_stage_corrected_velocity(
        state.u.data,
        live_un_adv,
        r1_hu,
        state.uu_b.data,
        u_mask_full,
        return_correction=True,
    )
    metric = jnp.asarray(card.recipe.grid.dy_u)
    live_metric_thickness = nemo_source_round(metric[..., None] * h_u)
    live_zfu_replay = _nemo_metric_stage_transport(
        metric, h_u, live_corrected)

    sx = slice(1, OWNED_NX + 1)
    s3 = (slice(None), sx, slice(None, NLEV))
    s2 = (slice(None), sx)
    exposed_thickness = np.asarray(thickness_state.u.data)[s3]
    exposed_corrected = np.asarray(corrected_state.u.data)[s3]
    exposed_zfu = np.asarray(transport_state.u.data)[s3]
    live = {
        "un_adv": np.asarray(live_un_adv)[s2],
        "inverse_depth": np.asarray(r1_hu)[s2],
        "uu_b_Kmm": np.asarray(state.uu_b.data)[s2],
        "zub": np.asarray(live_zub)[s2],
        "uu_Kmm": np.asarray(state.u.data)[s3],
        "umask": np.asarray(u_mask_full)[s3],
        "corrected_velocity": np.asarray(live_corrected)[s3],
        "e2u": np.asarray(metric)[s2],
        "e3u_Kmm": exposed_thickness,
        "metric_thickness": np.asarray(live_metric_thickness)[s3],
        "zFu": exposed_zfu,
    }
    oracle_mask = np.asarray(oracle["umask"])[..., :NLEV]
    active3 = oracle_mask != 0.0
    active2 = np.any(active3, axis=-1)
    oracle_zub = np.asarray(oracle["zub"])
    oracle_corrected = nemo_source_round(
        jnp.asarray(oracle["uu"])[..., :NLEV]
        + nemo_source_round(
            jnp.asarray(oracle_zub)[..., None] * jnp.asarray(oracle_mask)))
    oracle_metric_thickness = nemo_source_round(
        jnp.asarray(oracle["e2u"])[..., None]
        * jnp.asarray(oracle["e3u"])[..., :NLEV])
    expected = {
        "un_adv": np.asarray(oracle["un_adv"]),
        "inverse_depth": np.asarray(oracle["r1_hu"]),
        "uu_b_Kmm": np.asarray(oracle["uu_b"]),
        "zub": oracle_zub,
        "uu_Kmm": np.asarray(oracle["uu"])[..., :NLEV],
        "umask": oracle_mask,
        "corrected_velocity": np.asarray(oracle_corrected),
        "e2u": np.asarray(oracle["e2u"]),
        "e3u_Kmm": np.asarray(oracle["e3u"])[..., :NLEV],
        "metric_thickness": np.asarray(oracle_metric_thickness),
        "zFu": np.asarray(oracle["zFu"])[..., :NLEV],
    }
    two_d = {"un_adv", "inverse_depth", "uu_b_Kmm", "zub", "e2u"}
    rows = {
        name: _score(live[name], expected[name],
                     active2 if name in two_d else active3)
        for name in ORDER
    }
    calibration = {
        "derived_thickness_matches_exposure": _score(
            np.asarray(h_u)[s3], exposed_thickness, active3),
        "derived_corrected_matches_exposure": _score(
            np.asarray(live_corrected)[s3], exposed_corrected, active3),
        "derived_zFu_matches_exposure": _score(
            np.asarray(live_zfu_replay)[s3], exposed_zfu, active3),
    }
    first = next(
        (name for name in ORDER if rows[name]["status"] != "AT_BAR"), None)
    first_derived = next(
        (name for name in DERIVED_ORDER
         if rows[name]["status"] != "AT_BAR"), None)
    return {
        "format": "nemo-testcase-l4-orca2-round74-metric-u-v1",
        "claim_label": "independent",
        "card_scope": _card_scope(deck_root),
        "order": list(ORDER),
        "rows": rows,
        "first_non_bit": first,
        "first_non_bit_derived": first_derived,
        "record_replay": replay["rows"],
        "production_exposure_calibration": calibration,
        "record": {
            "path": str(record_path),
            "sha256": record_gate.sha256(record_path),
            "header": oracle["header"],
        },
        "execution": {
            "backend": jax.default_backend(),
            "production_jit": True,
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
        },
        "compiled_citation": (
            "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
            "stprk3_stg.f90:265-280"),
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "card-scope":
        report["card_scope"]["ORCA2-zps"][0] = "euler"
    elif plant == "record-replay":
        report["record_replay"][0]["status"] = "DEBT"
        report["record_replay"][0]["unequal"] = 1
    elif plant == "exposure-calibration":
        row = report["production_exposure_calibration"][
            "derived_zFu_matches_exposure"]
        row["status"] = "DEBT"
        row["unequal"] = 1
    elif plant == "first-classification":
        report["rows"]["un_adv"]["status"] = "AT_BAR"
        report["rows"]["un_adv"]["unequal"] = 0
    elif plant == "round73-boundary":
        report["rows"]["zFu"]["max_abs"] = math.nextafter(
            float(report["rows"]["zFu"]["max_abs"]), math.inf)

    require(report.get("claim_label") == "independent",
            "claim is not independent")
    require(report.get("execution") == {
        "backend": "cpu", "production_jit": True, "dtype": "float64",
        "transcendentals": "libm",
    }, "execution policy changed")
    scope = {name: tuple(values) for name, values in report["card_scope"].items()}
    require(scope == EXPECTED_CARD_SCOPE, "resolved stage-transport scope changed")
    require(report.get("order") == list(ORDER), "source order changed")
    require(all(row["status"] == "AT_BAR" and int(row["unequal"]) == 0
                for row in report["record_replay"]),
            "recorded-operand source replay is not exact")
    require(all(row["status"] == "AT_BAR" and int(row["unequal"]) == 0
                for row in report["production_exposure_calibration"].values()),
            "derived production exposure did not reproduce")
    zfu = report["rows"]["zFu"]
    for key, expected in ROUND73_ZFU.items():
        require(zfu[key] == expected,
                f"round-73 zFu {key} changed: {zfu[key]} != {expected}")
    first = next(
        (name for name in ORDER
         if report["rows"][name]["status"] != "AT_BAR"), None)
    first_derived = next(
        (name for name in DERIVED_ORDER
         if report["rows"][name]["status"] != "AT_BAR"), None)
    require(report.get("first_non_bit") == first,
            "first non-bit source row is stale")
    require(report.get("first_non_bit_derived") == first_derived,
            "first non-bit derived row is stale")
    require(first is not None and first_derived is not None,
            "walk found no non-bit transport boundary")
    predictions = {
        "all_four_cards_execute": {"status": "CONFIRMED"},
        "record_replay_exact": {"status": "CONFIRMED"},
        "un_adv_first": {
            "status": "CONFIRMED" if first == "un_adv" else "REFUTED",
            "observed": first,
        },
        "uu_Kmm_exact": {
            "status": ("CONFIRMED" if report["rows"]["uu_Kmm"]["status"]
                       == "AT_BAR" else "REFUTED"),
        },
        "e3u_Kmm_non_bit": {
            "status": ("CONFIRMED" if report["rows"]["e3u_Kmm"]["status"]
                       != "AT_BAR" else "REFUTED"),
        },
        "round73_zFu_reproduced": {"status": "CONFIRMED"},
        "disposition_held": {"status": "CONFIRMED"},
    }
    return {
        **report,
        "prediction_ledger": predictions,
        "status": "PASS_METRIC_U_TRANSPORT_WALK",
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
    except (GateError, record_gate.GateError, KeyError, OSError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_METRIC_U_TRANSPORT_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

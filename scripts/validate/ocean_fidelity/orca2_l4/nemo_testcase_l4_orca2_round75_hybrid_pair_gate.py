#!/usr/bin/env python3
"""Measure the two-input pair in ORCA2's stage-1 hybrid U correction."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round74_metric_u_transport_gate as round74,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_stage1_transport_gate as record_gate,
)

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks_3d,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    _nemo_metric_stage_transport,
    _nemo_stage_corrected_velocity,
    _nemo_ws_qco_stage_faces,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
)
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402

NLEV = round74.NLEV
OWNED_NX = round74.OWNED_NX
ARM_SOURCES = {
    "baseline": ["live_un_adv", "live_inverse_depth"],
    "oracle_un_adv": ["oracle_un_adv", "live_inverse_depth"],
    "oracle_inverse_depth": ["live_un_adv", "oracle_inverse_depth"],
    "oracle_pair": ["oracle_un_adv", "oracle_inverse_depth"],
}
ROW_ORDER = ("zub", "corrected_velocity", "zFu")
EXPECTED_BASELINE = {
    "zub": {
        "count": 8568,
        "max_abs": 0.01834375357001168,
        "status": "DEBT",
        "unequal": 8568,
    },
    "zFu": {
        "count": 226236,
        "max_abs": 321212.60790659266,
        "status": "DEBT",
        "unequal": 226236,
    },
}
PLANTS = (
    "none",
    "card-scope",
    "baseline",
    "pair-closure",
    "arm-sources",
)


class GateError(RuntimeError):
    """A frozen round-75 predicate or instrument invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(candidate: np.ndarray, oracle: np.ndarray,
           mask: np.ndarray) -> dict[str, object]:
    row = record_gate.score(candidate, oracle, mask)
    actual = np.asarray(candidate, dtype=np.float64)[mask]
    expected = np.asarray(oracle, dtype=np.float64)[mask]
    row["rms_abs"] = float(np.sqrt(np.mean(np.square(actual - expected))))
    return row


def _arm(
    *,
    velocity: np.ndarray,
    un_adv: np.ndarray,
    inverse_depth: np.ndarray,
    barotropic_velocity: np.ndarray,
    mask: np.ndarray,
    metric: np.ndarray,
    thickness: np.ndarray,
) -> dict[str, np.ndarray]:
    corrected, zub = _nemo_stage_corrected_velocity(
        jnp.asarray(velocity),
        jnp.asarray(un_adv),
        jnp.asarray(inverse_depth),
        jnp.asarray(barotropic_velocity),
        jnp.asarray(mask),
        return_correction=True,
    )
    zfu = _nemo_metric_stage_transport(
        jnp.asarray(metric), jnp.asarray(thickness), corrected)
    return {
        "zub": np.asarray(zub),
        "corrected_velocity": np.asarray(corrected),
        "zFu": np.asarray(zfu),
    }


def run_pair(deck_root: Path, record_root: Path) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-75 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy is not active")

    baseline_proof = round74.classify(round74.run_walk(deck_root, record_root))
    require(
        baseline_proof["status"] == "PASS_METRIC_U_TRANSPORT_WALK",
        "round-74 baseline gate did not pass",
    )

    record_path = (
        record_root / "oracle_rkstage1_transport_operands_kt00000001.bin")
    oracle = record_gate.read_record(record_path)
    card = build_orca2_zps_card(deck_root)
    state = card.recipe.initial_state
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    average_state = round74._step_operand(
        card, freshwater, surface, "transport_average")

    active_t = card.recipe.z_coord.is_active
    u_mask_full, v_mask_full = compute_face_masks_3d(
        active_t, card.recipe.grid)
    u_mask_full = jnp.asarray(u_mask_full, dtype=jnp.float64)
    v_mask_full = jnp.asarray(v_mask_full, dtype=jnp.float64)
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
        v_mask_full,
        card.recipe.grid,
        include_reciprocals=True,
    )

    sx = slice(1, OWNED_NX + 1)
    s3 = (slice(None), sx, slice(None, NLEV))
    s2 = (slice(None), sx)
    live = {
        "velocity": np.asarray(state.u.data)[s3],
        "un_adv": np.asarray(average_state.u.data[..., 0])[s2],
        "inverse_depth": np.asarray(r1_hu)[s2],
        "barotropic_velocity": np.asarray(state.uu_b.data)[s2],
        "mask": np.asarray(u_mask_full)[s3],
        "metric": np.asarray(card.recipe.grid.dy_u)[s2],
        "thickness": np.asarray(h_u)[s3],
    }
    expected_mask = np.asarray(oracle["umask"])[..., :NLEV]
    active3 = expected_mask != 0.0
    active2 = np.any(active3, axis=-1)
    expected_zub = np.asarray(oracle["zub"])
    expected_corrected = np.asarray(nemo_source_round(
        jnp.asarray(oracle["uu"])[..., :NLEV]
        + nemo_source_round(
            jnp.asarray(expected_zub)[..., None]
            * jnp.asarray(expected_mask))))
    expected = {
        "zub": expected_zub,
        "corrected_velocity": expected_corrected,
        "zFu": np.asarray(oracle["zFu"])[..., :NLEV],
    }

    arm_inputs = {
        "baseline": (live["un_adv"], live["inverse_depth"]),
        "oracle_un_adv": (
            np.asarray(oracle["un_adv"]), live["inverse_depth"]),
        "oracle_inverse_depth": (
            live["un_adv"], np.asarray(oracle["r1_hu"])),
        "oracle_pair": (
            np.asarray(oracle["un_adv"]), np.asarray(oracle["r1_hu"])),
    }
    masks = {
        "zub": active2,
        "corrected_velocity": active3,
        "zFu": active3,
    }
    arms: dict[str, object] = {}
    for name, (un_adv, inverse_depth) in arm_inputs.items():
        values = _arm(
            velocity=live["velocity"],
            un_adv=un_adv,
            inverse_depth=inverse_depth,
            barotropic_velocity=live["barotropic_velocity"],
            mask=live["mask"],
            metric=live["metric"],
            thickness=live["thickness"],
        )
        arms[name] = {
            "input_sources": list(ARM_SOURCES[name]),
            "rows": {
                row: _score(values[row], expected[row], masks[row])
                for row in ROW_ORDER
            },
        }

    return {
        "format": "nemo-testcase-l4-orca2-round75-hybrid-pair-v1",
        "claim_label": "independent",
        "card_scope": baseline_proof["card_scope"],
        "arm_sources": ARM_SOURCES,
        "row_order": list(ROW_ORDER),
        "arms": arms,
        "support": {
            "oracle_active_columns": int(active2.sum()),
            "oracle_active_3d": int(active3.sum()),
        },
        "round74_gate_status": baseline_proof["status"],
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
        "compiled_citations": {
            "active_selector": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:45-49"),
            "active_hybrid_transport": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:274-284"),
        },
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "card-scope":
        report["card_scope"]["ORCA2-zps"][0] = "euler"
    elif plant == "baseline":
        row = report["arms"]["baseline"]["rows"]["zub"]
        row["max_abs"] = math.nextafter(float(row["max_abs"]), math.inf)
    elif plant == "pair-closure":
        row = report["arms"]["oracle_pair"]["rows"]["zub"]
        row["status"] = "DEBT"
        row["unequal"] = 1
    elif plant == "arm-sources":
        report["arms"]["oracle_un_adv"]["input_sources"] = list(
            ARM_SOURCES["oracle_pair"])

    require(report.get("claim_label") == "independent",
            "claim is not independent")
    require(report.get("execution") == {
        "backend": "cpu", "production_jit": True, "dtype": "float64",
        "transcendentals": "libm",
    }, "execution policy changed")
    scope = {name: tuple(values) for name, values in report["card_scope"].items()}
    require(scope == round74.EXPECTED_CARD_SCOPE,
            "resolved stage-transport scope changed")
    require(report.get("round74_gate_status") ==
            "PASS_METRIC_U_TRANSPORT_WALK",
            "round-74 baseline proof is absent")
    require(report.get("row_order") == list(ROW_ORDER),
            "pair score order changed")
    require(report.get("support") == {
        "oracle_active_columns": 8568,
        "oracle_active_3d": 226236,
    }, "physical score support changed")
    require(report.get("arm_sources") == ARM_SOURCES,
            "declared arm sources changed")
    for name, sources in ARM_SOURCES.items():
        require(report["arms"][name]["input_sources"] == sources,
                f"{name} did not use its declared operands")

    baseline = report["arms"]["baseline"]["rows"]
    for row_name, frozen in EXPECTED_BASELINE.items():
        for key, expected in frozen.items():
            require(baseline[row_name][key] == expected,
                    f"baseline {row_name} {key} changed")

    pair = report["arms"]["oracle_pair"]["rows"]
    for row_name in ("zub", "corrected_velocity"):
        require(pair[row_name]["status"] == "AT_BAR"
                and int(pair[row_name]["unequal"]) == 0,
                f"paired operands did not close {row_name}")

    base_rms = float(baseline["zub"]["rms_abs"])
    un_adv = report["arms"]["oracle_un_adv"]["rows"]["zub"]
    inverse = report["arms"]["oracle_inverse_depth"]["rows"]["zub"]
    un_adv_worsens = (
        un_adv["status"] != "AT_BAR" and float(un_adv["rms_abs"]) > base_rms)
    inverse_worsens = (
        inverse["status"] != "AT_BAR" and float(inverse["rms_abs"]) > base_rms)
    two_sided = un_adv_worsens and inverse_worsens
    if two_sided:
        owner = "paired_un_adv_and_inverse_depth"
    elif float(un_adv["rms_abs"]) < float(inverse["rms_abs"]):
        owner = "un_adv"
    elif float(inverse["rms_abs"]) < float(un_adv["rms_abs"]):
        owner = "inverse_depth"
    else:
        owner = "unresolved_equal_single_arm_rms"

    predictions = {
        "baseline_reproduced": {"status": "CONFIRMED"},
        "pair_closes_zub_and_corrected_velocity": {"status": "CONFIRMED"},
        "un_adv_single_worsens": {
            "status": "CONFIRMED" if un_adv_worsens else "REFUTED",
            "baseline_rms": base_rms,
            "observed_rms": float(un_adv["rms_abs"]),
        },
        "inverse_depth_single_worsens": {
            "status": "CONFIRMED" if inverse_worsens else "REFUTED",
            "baseline_rms": base_rms,
            "observed_rms": float(inverse["rms_abs"]),
        },
        "two_sided_cancellation": {
            "status": "CONFIRMED" if two_sided else "REFUTED",
        },
        "paired_zFu_remains_debt": {
            "status": ("CONFIRMED" if pair["zFu"]["status"] != "AT_BAR"
                       else "REFUTED"),
        },
        "disposition_held": {"status": "CONFIRMED"},
    }
    return {
        **report,
        "owner": owner,
        "prediction_ledger": predictions,
        "status": "PASS_HYBRID_CORRECTION_PAIR",
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
            raw = run_pair(args.deck_root, args.record_root)
        report = classify(raw, plant=args.plant)
    except (GateError, round74.GateError, record_gate.GateError, KeyError,
            OSError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_HYBRID_CORRECTION_PAIR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

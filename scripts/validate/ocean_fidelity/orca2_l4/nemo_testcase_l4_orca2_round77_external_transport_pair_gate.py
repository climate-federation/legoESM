#!/usr/bin/env python3
"""Measure the two operands of independent ORCA2 substep-2 ``zhU``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round76_external_transport_gate as round76,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_stage1_transport_gate as record_gate,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_round14_advmean as advmean,
)

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (  # noqa: E402
    nemo_literal_metric_transports,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
)

SUBSTEP_INDEX = 1
ARM_SOURCES = {
    "live_pair": ["live_ua_e", "live_zhup2_e"],
    "oracle_velocity_only": ["oracle_ua_e", "live_zhup2_e"],
    "oracle_depth_only": ["live_ua_e", "oracle_zhup2_e"],
    "oracle_pair": ["oracle_ua_e", "oracle_zhup2_e"],
}
EXPECTED_BASELINE = {
    "count": 8568,
    "max_abs": 23.712296310346574,
    "rms_abs": 2.6045999142426335,
    "status": "DEBT",
    "unequal": 8568,
}
PLANTS = (
    "none",
    "card-scope",
    "baseline",
    "pair-closure",
    "arm-sources",
    "owner-order",
)


class GateError(RuntimeError):
    """A round-77 instrument invariant or frozen baseline failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(candidate, oracle, mask) -> dict[str, object]:
    return round76._score(
        np.asarray(candidate), np.asarray(oracle), np.asarray(mask, dtype=bool))


def _metric_transport(
    *, metric: np.ndarray, mask: np.ndarray,
    velocity: np.ndarray, depth: np.ndarray,
) -> np.ndarray:
    """Evaluate the shared source-ordered metric-transport statement."""
    zeros = jnp.zeros_like(jnp.asarray(velocity))
    ones = jnp.ones_like(jnp.asarray(depth))
    replay_grid = SimpleNamespace(
        dy_u=jnp.asarray(metric), dx_v=jnp.ones_like(jnp.asarray(metric)))
    metric_u, _ = nemo_literal_metric_transports(
        jnp.asarray(depth), ones,
        jnp.asarray(velocity), zeros,
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64), replay_grid)
    return np.asarray(metric_u)


def _owner(arms: dict[str, object]) -> str:
    velocity_rms = float(arms["oracle_velocity_only"]["rms_abs"])
    depth_rms = float(arms["oracle_depth_only"]["rms_abs"])
    if velocity_rms < depth_rms:
        return "ua_e"
    if depth_rms < velocity_rms:
        return "zhup2_e"
    return "tie"


def run_pair(
    deck_root: Path, record_root: Path, round76_json: Path,
) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-77 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy is not active")

    require(round76_json.is_file(), f"missing round-76 report {round76_json}")
    inherited = round76.classify(json.loads(round76_json.read_text()))
    require(inherited["status"] == "PASS_EXTERNAL_TRANSPORT_WALK",
            "round-76 baseline gate did not pass")

    card = build_orca2_zps_card(deck_root)
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True))
    trace = jax.device_get(model.step(
        card.recipe.initial_state, card.dt_s,
        freshwater=freshwater, surface_forcing=surface))

    adv_path = record_root / "oracle_bt_advmean_operands_kt00000001.bin"
    fields = advmean.read_advmean(
        adv_path, expected_kt=1, expected_dims=(94, 152),
        expected_ncycle=round76.N_CYCLE)
    stage_path = record_root / "oracle_rkstage1_transport_operands_kt00000001.bin"
    stage = record_gate.read_record(stage_path)
    active = np.any(np.asarray(stage["umask"])[..., :round76.NLEV] != 0.0,
                    axis=-1)
    metric = np.asarray(card.recipe.grid.dy_u)[:, 1:][..., :round76.OWNED_NX]
    live_velocity = round76._trace_u(
        trace.substeps["transport_velocity_u"])[SUBSTEP_INDEX]
    live_depth = round76._trace_u(
        trace.substeps["transport_face_depth_u"])[SUBSTEP_INDEX]
    oracle_velocity = np.asarray(fields["velocity_u"])[SUBSTEP_INDEX]
    oracle_depth = np.asarray(fields["face_depth_u"])[SUBSTEP_INDEX]
    expected = np.asarray(fields["metric_u"])[SUBSTEP_INDEX]

    inputs = {
        "live_pair": (live_velocity, live_depth),
        "oracle_velocity_only": (oracle_velocity, live_depth),
        "oracle_depth_only": (live_velocity, oracle_depth),
        "oracle_pair": (oracle_velocity, oracle_depth),
    }
    arms: dict[str, object] = {}
    for name, (velocity, depth) in inputs.items():
        values = _metric_transport(
            metric=metric, mask=active, velocity=velocity, depth=depth)
        arms[name] = {
            "input_sources": list(ARM_SOURCES[name]),
            **_score(values, expected, active),
        }

    return {
        "format": "nemo-testcase-l4-orca2-round77-external-pair-v1",
        "claim_label": "independent",
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
        },
        "card_scope": inherited["card_scope"],
        "external_card_scope": inherited["external_card_scope"],
        "round76_status": inherited["status"],
        "round76_first_boundary": (
            inherited["first_non_bit_u_accumulator_statement"]),
        "arm_sources": ARM_SOURCES,
        "arms": arms,
        "dominant_owner": _owner(arms),
        "support": {"active_u_columns": int(active.sum())},
        "record": {
            "advmean_path": str(adv_path),
            "advmean_sha256": record_gate.sha256(adv_path),
            "stage_path": str(stage_path),
            "stage_sha256": record_gate.sha256(stage_path),
        },
        "compiled_citations": {
            "predictor_coefficients": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:460-469"),
            "velocity_midpoint": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:476-485"),
            "depth_midpoint": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:487-522"),
            "metric_transport": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:530-536"),
        },
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "card-scope":
        report["external_card_scope"]["ORCA2-zps"][0] = "euler"
    elif plant == "baseline":
        report["round76_first_boundary"]["rms_abs"] = math.nextafter(
            EXPECTED_BASELINE["rms_abs"], math.inf)
    elif plant == "pair-closure":
        report["arms"]["oracle_pair"]["status"] = "DEBT"
        report["arms"]["oracle_pair"]["unequal"] = 1
    elif plant == "arm-sources":
        report["arms"]["oracle_velocity_only"]["input_sources"] = list(
            ARM_SOURCES["live_pair"])
    elif plant == "owner-order":
        velocity = report["arms"]["oracle_velocity_only"]
        depth = report["arms"]["oracle_depth_only"]
        velocity["rms_abs"], depth["rms_abs"] = (
            depth["rms_abs"], velocity["rms_abs"])

    require(report.get("claim_label") == "independent",
            "claim is not independent")
    require(report.get("execution") == {
        "backend": "cpu", "production_jit": True, "dtype": "float64",
        "transcendentals": "libm",
    }, "execution policy changed")
    require(report.get("round76_status") == "PASS_EXTERNAL_TRANSPORT_WALK",
            "round-76 proof is absent")
    require(report.get("external_card_scope") == round76.EXPECTED_EXTERNAL_CARD_SCOPE,
            "resolved external-transport scope changed")
    require(report.get("support") == {"active_u_columns": 8568},
            "physical score support changed")
    inherited = report["round76_first_boundary"]
    for key, value in EXPECTED_BASELINE.items():
        require(inherited.get(key) == value,
                f"round-76 baseline {key} changed")
    for name, sources in ARM_SOURCES.items():
        require(report["arms"][name]["input_sources"] == sources,
                f"{name} did not use its declared operands")
    require(report["arms"]["live_pair"]["rms_abs"] == EXPECTED_BASELINE["rms_abs"],
            "live/live arm did not reproduce round 76")
    require(report["arms"]["oracle_pair"]["status"] == "AT_BAR"
            and report["arms"]["oracle_pair"]["unequal"] == 0,
            "recorded ua_e/zhup2_e pair did not close zhU")
    measured_owner = _owner(report["arms"])
    require(report.get("dominant_owner") == measured_owner,
            "declared dominant owner disagrees with measured arms")

    velocity_rms = float(report["arms"]["oracle_velocity_only"]["rms_abs"])
    depth_rms = float(report["arms"]["oracle_depth_only"]["rms_abs"])
    baseline_rms = float(report["arms"]["live_pair"]["rms_abs"])
    predicted_owner = velocity_rms < baseline_rms and velocity_rms < depth_rms
    depth_not_exact = report["arms"]["oracle_depth_only"]["status"] != "AT_BAR"
    return {
        **report,
        "prediction_ledger": {
            "round76_baseline_reproduces": {"status": "CONFIRMED"},
            "four_arms_isolated": {"status": "CONFIRMED"},
            "recorded_pair_closes": {"status": "CONFIRMED"},
            "ua_e_is_dominant_owner": {
                "status": "CONFIRMED" if predicted_owner else "REFUTED"},
            "depth_only_remains_non_bit": {
                "status": "CONFIRMED" if depth_not_exact else "REFUTED"},
            "disposition_held": {"status": "CONFIRMED"},
        },
        "status": "PASS_EXTERNAL_TRANSPORT_PAIR",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument(
        "--round76-json", type=Path,
        default=Path(
            "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
            "orca2_rounds/round76/external_transport_final.json"))
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
            raw = run_pair(args.deck_root, args.record_root, args.round76_json)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        report = classify(raw, plant=args.plant)
    except (GateError, round76.GateError, record_gate.GateError,
            KeyError, OSError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_EXTERNAL_TRANSPORT_PAIR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

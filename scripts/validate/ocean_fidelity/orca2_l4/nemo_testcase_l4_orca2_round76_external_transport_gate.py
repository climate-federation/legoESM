#!/usr/bin/env python3
"""Walk independent ORCA2 ``un_adv`` and its downstream geometry pair."""

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
    nemo_testcase_l4_orca2_round74_metric_u_transport_gate as round74,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round75_hybrid_pair_gate as round75,
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
    nemo_literal_accumulate_transport,
    nemo_literal_metric_transports,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks_3d,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
    _nemo_ws_qco_stage_faces,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
)
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402

N_CYCLE = 65
NLEV = round74.NLEV
OWNED_NX = round74.OWNED_NX
FIRST_BOUNDARY = {"substep": 2, "boundary": "metric_transport"}
GEOMETRY_ARM_SOURCES = {
    "live_pair": ["oracle_un_adv", "live_inverse_depth", "live_thickness"],
    "oracle_inverse_only": [
        "oracle_un_adv", "oracle_inverse_depth", "live_thickness"],
    "oracle_thickness_only": [
        "oracle_un_adv", "live_inverse_depth", "oracle_thickness"],
    "oracle_pair": [
        "oracle_un_adv", "oracle_inverse_depth", "oracle_thickness"],
}
EXPECTED_ROUND75 = {
    "live_pair_rms": 1.971885366122676e-11,
    "oracle_inverse_only_rms": 0.790927819646436,
}
PLANTS = (
    "none",
    "card-scope",
    "accumulator-exit",
    "first-boundary",
    "geometry-sources",
    "round75-baseline",
)


class GateError(RuntimeError):
    """A frozen round-76 predicate or instrument invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(candidate, oracle, mask) -> dict[str, object]:
    return round75._score(
        np.asarray(candidate), np.asarray(oracle), np.asarray(mask, dtype=bool))


def _trace_u(values) -> np.ndarray:
    """Map legoESM's full U-face trace to NEMO rank-0 owned U slots."""
    values = np.asarray(values, dtype=np.float64)
    return values[..., 1:][..., :OWNED_NX]


def _first_boundary(seed: dict, rows: list[dict]) -> dict | None:
    if seed["status"] != "AT_BAR":
        return {"substep": 0, "boundary": "zero_seed", **seed}
    order = ("sum_entry", "weight", "metric_transport", "r1_e2u", "sum_exit")
    for row in rows:
        for boundary in order:
            result = row[boundary]
            if result["status"] != "AT_BAR":
                return {"substep": row["substep"], "boundary": boundary, **result}
    return None


def _record_replay(
    fields: dict, card, active2: np.ndarray, oracle_metric: np.ndarray,
) -> dict[str, object]:
    """Replay the source statements through the shared production helpers."""
    metric = np.asarray(card.recipe.grid.dy_u, dtype=np.float64)[:, 1:][
        ..., :OWNED_NX]
    require(
        np.array_equal(
            metric[active2].view(np.uint64),
            np.asarray(oracle_metric)[active2].view(np.uint64),
        ),
        "card e2u is not the record metric on active U faces",
    )
    zeros = np.zeros_like(metric)
    replay_grid = SimpleNamespace(dy_u=jnp.asarray(metric), dx_v=jnp.ones_like(metric))
    mask = active2.astype(np.float64)

    def one(sum_u, weight, depth, velocity):
        metric_u, _ = nemo_literal_metric_transports(
            jnp.asarray(depth), jnp.ones_like(depth), jnp.asarray(velocity),
            jnp.zeros_like(velocity), jnp.asarray(mask), jnp.asarray(mask),
            replay_grid,
        )
        exit_u, _ = nemo_literal_accumulate_transport(
            jnp.asarray(sum_u), jnp.asarray(zeros), jnp.asarray(weight),
            jnp.asarray(depth), jnp.ones_like(depth), jnp.asarray(velocity),
            jnp.zeros_like(velocity), jnp.asarray(mask), jnp.asarray(mask),
            replay_grid,
        )
        return metric_u, exit_u

    metric_u, exit_u = jax.jit(jax.vmap(one))(
        jnp.asarray(fields["sum_u_entry"]),
        jnp.asarray(fields["weight"]),
        jnp.asarray(fields["face_depth_u"]),
        jnp.asarray(fields["velocity_u"]),
    )
    metric_u, exit_u = np.asarray(metric_u), np.asarray(exit_u)
    metric_rows = [
        _score(metric_u[index], fields["metric_u"][index], active2)
        for index in range(N_CYCLE)
    ]
    exit_rows = [
        _score(exit_u[index], fields["sum_u_exit"][index], active2)
        for index in range(N_CYCLE)
    ]
    normalized = _score(
        fields["sum_u_exit"][-1] / np.float64(fields["divisor"]),
        fields["pre_lbc_u"], active2)
    return {
        "metric_u": metric_rows,
        "sum_u_exit": exit_rows,
        "pre_lbc_normalized_u": normalized,
        "all_metric_exact": all(row["status"] == "AT_BAR" for row in metric_rows),
        "all_exit_exact": all(row["status"] == "AT_BAR" for row in exit_rows),
    }


def _geometry_arms(card, oracle: dict, live_un_adv: np.ndarray) -> dict[str, object]:
    state = card.recipe.initial_state
    active_t = card.recipe.z_coord.is_active
    u_mask, v_mask = compute_face_masks_3d(active_t, card.recipe.grid)
    u_mask = jnp.asarray(u_mask, dtype=jnp.float64)
    v_mask = jnp.asarray(v_mask, dtype=jnp.float64)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    h_u, _, _, _, r1_hu, _ = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, u_mask, v_mask, card.recipe.grid,
        include_reciprocals=True)
    sx = slice(1, OWNED_NX + 1)
    s2 = (slice(None), sx)
    s3 = (slice(None), sx, slice(None, NLEV))
    expected_mask = np.asarray(oracle["umask"])[..., :NLEV]
    active3 = expected_mask != 0.0
    active2 = np.any(active3, axis=-1)
    expected = {
        "zub": np.asarray(oracle["zub"]),
        "corrected_velocity": np.asarray(round75.nemo_source_round(
            jnp.asarray(oracle["uu"])[..., :NLEV]
            + round75.nemo_source_round(
                jnp.asarray(oracle["zub"])[..., None]
                * jnp.asarray(expected_mask)))),
        "zFu": np.asarray(oracle["zFu"])[..., :NLEV],
    }
    common = {
        "velocity": np.asarray(state.u.data)[s3],
        "un_adv": np.asarray(oracle["un_adv"]),
        "barotropic_velocity": np.asarray(state.uu_b.data)[s2],
        "mask": np.asarray(u_mask)[s3],
        "metric": np.asarray(card.recipe.grid.dy_u)[s2],
    }
    inputs = {
        "live_pair": (np.asarray(r1_hu)[s2], np.asarray(h_u)[s3]),
        "oracle_inverse_only": (
            np.asarray(oracle["r1_hu"]), np.asarray(h_u)[s3]),
        "oracle_thickness_only": (
            np.asarray(r1_hu)[s2], np.asarray(oracle["e3u"])[..., :NLEV]),
        "oracle_pair": (
            np.asarray(oracle["r1_hu"]),
            np.asarray(oracle["e3u"])[..., :NLEV]),
    }
    masks = {"zub": active2, "corrected_velocity": active3, "zFu": active3}
    arms = {}
    for name, (inverse_depth, thickness) in inputs.items():
        values = round75._arm(
            **common, inverse_depth=inverse_depth, thickness=thickness)
        arms[name] = {
            "input_sources": list(GEOMETRY_ARM_SOURCES[name]),
            "rows": {
                row: _score(values[row], expected[row], masks[row])
                for row in round75.ROW_ORDER
            },
        }
    return {"arms": arms, "active2": active2, "active3": active3}


def run_walk(
    deck_root: Path, record_root: Path, round75_json: Path,
) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-76 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy is not active")

    require(round75_json.is_file(), f"missing round-75 report {round75_json}")
    pair_proof = round75.classify(json.loads(round75_json.read_text()))
    require(pair_proof["status"] == "PASS_HYBRID_CORRECTION_PAIR",
            "round-75 baseline gate did not pass")
    card = build_orca2_zps_card(deck_root)
    state = card.recipe.initial_state
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True))
    trace = jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True))
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    adv_path = record_root / "oracle_bt_advmean_operands_kt00000001.bin"
    fields = advmean.read_advmean(
        adv_path, expected_kt=1, expected_dims=(94, 152),
        expected_ncycle=N_CYCLE)
    stage_path = record_root / "oracle_rkstage1_transport_operands_kt00000001.bin"
    oracle = record_gate.read_record(stage_path)
    active2 = np.any(np.asarray(oracle["umask"])[..., :NLEV] != 0.0, axis=-1)
    active_scalar = np.ones((), dtype=bool)

    live = {
        "sum_entry": _trace_u(trace.substeps["transport_sum_u_entry"]),
        "metric_transport": _trace_u(trace.substeps["transport_metric_u"]),
        "velocity": _trace_u(trace.substeps["transport_velocity_u"]),
        "face_depth": _trace_u(trace.substeps["transport_face_depth_u"]),
        "sum_exit": _trace_u(trace.substeps["transport_sum_u_exit"]),
    }
    live_weight = np.asarray(trace.substeps["transport_weight"], dtype=np.float64)
    live_r1_e2u = np.where(
        active2,
        1.0 / np.asarray(card.recipe.grid.dy_u)[:, 1:][..., :OWNED_NX],
        0.0,
    )
    seed = _score(np.zeros_like(fields["sum_u_entry"][0]),
                  fields["sum_u_entry"][0], active2)
    rows = []
    for index in range(N_CYCLE):
        rows.append({
            "substep": index + 1,
            "sum_entry": _score(
                live["sum_entry"][index], fields["sum_u_entry"][index], active2),
            "weight": _score(
                np.asarray(live_weight[index]), np.asarray(fields["weight"][index]),
                active_scalar),
            "metric_transport": _score(
                live["metric_transport"][index], fields["metric_u"][index], active2),
            "r1_e2u": _score(live_r1_e2u, fields["r1_e2u"], active2),
            "sum_exit": _score(
                live["sum_exit"][index], fields["sum_u_exit"][index], active2),
        })
    first = _first_boundary(seed, rows)
    first_operands = "UNREACHED"
    if first is not None and first["boundary"] == "metric_transport":
        index = int(first["substep"]) - 1
        first_operands = {
            "velocity_u": _score(
                live["velocity"][index], fields["velocity_u"][index], active2),
            "face_depth_u": _score(
                live["face_depth"][index], fields["face_depth_u"][index], active2),
        }
    endpoint = _score(
        _trace_u(np.asarray(trace.transport_average[0])[None, ...])[0],
        np.asarray(ordinary.barotropic_targets[2])[:, 1:][..., :OWNED_NX],
        active2)
    replay = _record_replay(fields, card, active2, np.asarray(oracle["e2u"]))
    geometry = _geometry_arms(
        card, oracle,
        _trace_u(np.asarray(trace.transport_average[0])[None, ...])[0])
    return {
        "format": "nemo-testcase-l4-orca2-round76-external-transport-v1",
        "claim_label": "independent",
        "card_scope": pair_proof["card_scope"],
        "round75_status": pair_proof["status"],
        "round75_observed": {
            "live_pair_rms": pair_proof["arms"]["oracle_un_adv"]["rows"]["zFu"]["rms_abs"],
            "oracle_inverse_only_rms": pair_proof["arms"]["oracle_pair"]["rows"]["zFu"]["rms_abs"],
        },
        "zero_seed": seed,
        "live_rows": rows,
        "first_non_bit_u_accumulator_statement": first,
        "first_statement_operands": first_operands,
        "production_trace_endpoint_identity": endpoint,
        "record_replay": replay,
        "geometry_arm_sources": GEOMETRY_ARM_SOURCES,
        "geometry_arms": geometry["arms"],
        "support": {
            "active_u_columns": int(active2.sum()),
            "active_u_3d": int(geometry["active3"].sum()),
        },
        "record": {
            "advmean_path": str(adv_path),
            "advmean_sha256": record_gate.sha256(adv_path),
            "stage_path": str(stage_path),
            "stage_sha256": record_gate.sha256(stage_path),
            "advmean_header": fields["header"],
        },
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
        },
        "compiled_citations": {
            "transport_accumulator": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "dynspg_ts.f90:380-380,563-579,831-847"),
            "hybrid_transport": (
                "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:274-284"),
        },
    }


def classify(report: dict[str, object], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "card-scope":
        report["card_scope"]["ORCA2-zps"][0] = "euler"
    elif plant == "accumulator-exit":
        report["record_replay"]["sum_u_exit"][0]["status"] = "DEBT"
        report["record_replay"]["sum_u_exit"][0]["unequal"] = 1
    elif plant == "first-boundary":
        report["first_non_bit_u_accumulator_statement"]["substep"] = 1
    elif plant == "geometry-sources":
        report["geometry_arms"]["oracle_pair"]["input_sources"] = list(
            GEOMETRY_ARM_SOURCES["oracle_inverse_only"])
    elif plant == "round75-baseline":
        report["round75_observed"]["live_pair_rms"] = math.nextafter(
            EXPECTED_ROUND75["live_pair_rms"], math.inf)

    require(report.get("claim_label") == "independent", "claim is not independent")
    require(report.get("execution") == {
        "backend": "cpu", "production_jit": True, "dtype": "float64",
        "transcendentals": "libm",
    }, "execution policy changed")
    scope = {name: tuple(value) for name, value in report["card_scope"].items()}
    require(scope == round74.EXPECTED_CARD_SCOPE,
            "resolved stage-transport scope changed")
    require(report.get("round75_status") == "PASS_HYBRID_CORRECTION_PAIR",
            "round-75 proof is absent")
    require(report.get("round75_observed") == EXPECTED_ROUND75,
            "round-75 geometry baseline changed")
    require(report.get("support") == {
        "active_u_columns": 8568, "active_u_3d": 226236,
    }, "physical score support changed")
    require(report["zero_seed"]["status"] == "AT_BAR",
            "transport accumulator seed is not exact")
    require(all(row["weight"]["status"] == "AT_BAR"
                for row in report["live_rows"]), "transport weights are not exact")
    first = report.get("first_non_bit_u_accumulator_statement")
    require(first is not None and {
        "substep": first["substep"], "boundary": first["boundary"]
    } == FIRST_BOUNDARY, "first non-bit U accumulator boundary changed")
    require(report["production_trace_endpoint_identity"]["status"] == "AT_BAR",
            "substep exposure changed the production endpoint")
    replay = report["record_replay"]
    require(replay["all_metric_exact"] and replay["all_exit_exact"],
            "recorded-operand accumulator replay is not exact")
    require(all(row["status"] == "AT_BAR" for row in replay["metric_u"])
            and all(row["status"] == "AT_BAR" for row in replay["sum_u_exit"]),
            "recorded-operand replay row is not exact")
    require(replay["pre_lbc_normalized_u"]["status"] == "AT_BAR",
            "recorded accumulator normalization is not exact")
    for name, sources in GEOMETRY_ARM_SOURCES.items():
        require(report["geometry_arms"][name]["input_sources"] == sources,
                f"{name} did not use its declared geometry operands")
    live_pair = report["geometry_arms"]["live_pair"]["rows"]["zFu"]
    inverse_only = report["geometry_arms"]["oracle_inverse_only"]["rows"]["zFu"]
    thickness_only = report["geometry_arms"]["oracle_thickness_only"]["rows"]["zFu"]
    oracle_pair = report["geometry_arms"]["oracle_pair"]["rows"]
    require(live_pair["rms_abs"] == EXPECTED_ROUND75["live_pair_rms"],
            "live/live zFu did not reproduce round 75")
    require(inverse_only["rms_abs"] == EXPECTED_ROUND75["oracle_inverse_only_rms"],
            "oracle-inverse/live-thickness zFu did not reproduce round 75")
    require(all(oracle_pair[row]["status"] == "AT_BAR"
                for row in round75.ROW_ORDER),
            "recorded inverse-depth/thickness pair did not close")
    thickness_worsens = float(thickness_only["rms_abs"]) > float(live_pair["rms_abs"])
    require(thickness_worsens,
            "recorded thickness alone did not worsen the live geometry pair")
    return {
        **report,
        "prediction_ledger": {
            "record_replay_exact": {"status": "CONFIRMED"},
            "production_trace_identity": {"status": "CONFIRMED"},
            "zero_seed_and_weights_exact": {"status": "CONFIRMED"},
            "first_u_boundary_substep2_metric_transport": {"status": "CONFIRMED"},
            "two_sided_geometry_consistency": {"status": "CONFIRMED"},
            "disposition_held": {"status": "CONFIRMED"},
        },
        "status": "PASS_EXTERNAL_TRANSPORT_WALK",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument(
        "--round75-json", type=Path,
        default=Path(
            "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
            "orca2_rounds/round75/hybrid_pair.json"),
    )
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
            raw = run_walk(args.deck_root, args.record_root, args.round75_json)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        report = classify(raw, plant=args.plant)
    except (GateError, round74.GateError, round75.GateError,
            record_gate.GateError, KeyError, OSError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "FAIL"
        print(f"STATUS {marker}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_EXTERNAL_TRANSPORT_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

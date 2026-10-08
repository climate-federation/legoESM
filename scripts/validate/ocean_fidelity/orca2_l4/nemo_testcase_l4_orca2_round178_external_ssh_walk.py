#!/usr/bin/env python3
"""Walk rung-0's independent kt=1 split-explicit SSH endpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)

FLOOR = np.float64(2.0e-10)
ARM_ORDER = ("baseline", "history_only", "slow_only", "slow_and_history")
ENTRY_ORDER = (
    "ssh_forcing", "u_forcing", "v_forcing",
    "u_entry", "v_entry", "ssh_entry",
    "u_history_b", "u_history_bb", "v_history_b", "v_history_bb",
    "ssh_history_b", "ssh_history_bb",
)
SUBSTEP_ORDER = (
    "u_mid", "v_mid", "ssh_mid", "depth_u_mid", "depth_v_mid",
    "transport_u", "transport_v", "ssh_after",
    "transport_sum_u", "transport_sum_v", "face_ssh_u", "face_ssh_v",
    "ssh_back", "pgf_u", "pgf_v", "coriolis_u", "coriolis_v",
    "trend_u", "trend_v", "u_exit", "v_exit", "depth_u_exit",
    "depth_v_exit", "inverse_u_exit", "inverse_v_exit",
)
EXIT_ORDER = (
    "transport_mean_u", "transport_mean_v",
    "external_mean_u", "external_mean_v", "external_mean_ssh",
)
PLANTS = (
    "none", "rank-placement", "record-bit", "source-order",
    "arm-identity", "endpoint-ulp",
)


class GateError(RuntimeError):
    """The admitted split-explicit record no longer supports this walk."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(candidate, oracle, active, *, complete_domain: bool = False) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    row = rhs_walk.score(
        candidate,
        oracle,
        np.asarray(active, dtype=bool),
    )
    full_delta = np.abs(candidate - oracle)
    full_flat = int(np.argmax(full_delta))
    row["full_domain_absolute_max"] = float(full_delta.flat[full_flat])
    row["full_domain_argmax_jik"] = [
        int(index) for index in np.unravel_index(full_flat, full_delta.shape)
    ]
    row["full_domain_rms"] = float(np.sqrt(np.mean(full_delta * full_delta)))
    comparison_max = (
        row["full_domain_absolute_max"] if complete_domain
        else row["absolute_max"]
    )
    comparison_bit_exact = (
        bool(np.array_equal(candidate, oracle)) if complete_domain
        else bool(row["bit_exact"])
    )
    row["comparison_domain"] = "complete-recorded" if complete_domain else "active"
    row["comparison_bit_exact"] = comparison_bit_exact
    row["at_floor"] = bool(comparison_max <= FLOOR)
    row["verdict"] = (
        "AT_BAR_BIT_EXACT" if comparison_bit_exact else
        ("AT_BAR_NOT_EXACT" if row["at_floor"] else "DEBT")
    )
    return row


def _native_u(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:, 1:]


def _native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:, :]


def _to_model_u(value: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(value, dtype=np.float64), ((0, 0), (1, 0)))


def _to_model_v(value: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(value, dtype=np.float64), ((1, 0), (0, 0)))


def _trace_digest(observed) -> str:
    """Digest the complete passive external trace without reducing its bits."""

    digest = hashlib.sha256()
    for value in observed.slow_forcing:
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    for name in sorted(observed.substeps):
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(observed.substeps[name], dtype=np.float64).tobytes())
    state = observed.state_after_barotropic
    for value in (state.eta.data, state.uu_b.data, state.vv_b.data,
                  observed.transport_average[0], observed.transport_average[1]):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _source_rows(observed, oracle, active) -> list[dict[str, object]]:
    trace = observed.substeps
    rows: list[dict[str, object]] = []

    def add(name, candidate, target, mask, *, substep=None):
        # Fold and cyclic halo values can be masked as prognostic faces yet
        # remain operands of a neighbouring active-cell stencil.  Source-order
        # attribution therefore scores the complete rank-assembled record;
        # active-mask counts remain in the same row as diagnostics.
        row = {"name": name, **_score(
            candidate, target, mask, complete_domain=True)}
        if substep is not None:
            row["substep"] = substep
        rows.append(row)

    add("ssh_forcing", observed.slow_forcing[0], oracle["i000_ssh_frc"], active["t"])
    add("u_forcing", _native_u(observed.slow_forcing[1]), oracle["i000_zu_frc"], active["u"])
    add("v_forcing", _native_v(observed.slow_forcing[2]), oracle["i000_zv_frc"], active["v"])
    add("u_entry", _native_u(trace["u_entry"][0]), oracle["i000_un_e"], active["u"])
    add("v_entry", _native_v(trace["v_entry"][0]), oracle["i000_vn_e"], active["v"])
    add("ssh_entry", trace["eta_entry"][0], oracle["i000_sshn_e"], active["t"])
    for name, trace_name, record_name, face in (
        ("u_history_b", "u_history_b", "i000_ub_e", "u"),
        ("u_history_bb", "u_history_bb", "i000_ubb_e", "u"),
        ("v_history_b", "v_history_b", "i000_vb_e", "v"),
        ("v_history_bb", "v_history_bb", "i000_vbb_e", "v"),
        ("ssh_history_b", "eta_history_b", "i000_sshb_e", "t"),
        ("ssh_history_bb", "eta_history_bb", "i000_sshbb_e", "t"),
    ):
        value = trace[trace_name][0]
        if face == "u":
            value = _native_u(value)
        elif face == "v":
            value = _native_v(value)
        add(name, value, oracle[record_name], active[face])

    registry = (
        ("u_mid", "u_mid", "ua_ext", "u"),
        ("v_mid", "v_mid", "va_ext", "v"),
        ("ssh_mid", "eta_mid", "sshp2_mid", "t"),
        ("depth_u_mid", "transport_face_depth_u", "hup2_e", "u"),
        ("depth_v_mid", "transport_face_depth_v", "hvp2_e", "v"),
        ("transport_u", "transport_metric_u", "zhU", "u"),
        ("transport_v", "transport_metric_v", "zhV", "v"),
        ("ssh_after", "eta_continuity", "ssha_e", "t"),
        ("transport_sum_u", "transport_sum_u_exit", "un_adv", "u"),
        ("transport_sum_v", "transport_sum_v_exit", "vn_adv", "v"),
        ("face_ssh_u", "face_ssh_u_exit", "sshu_a", "u"),
        ("face_ssh_v", "face_ssh_v_exit", "sshv_a", "v"),
        ("ssh_back", "eta_pgf", "sshp2_bck", "t"),
        ("pgf_u", "pgf_u", "zu_spg", "u"),
        ("pgf_v", "pgf_v", "zv_spg", "v"),
        ("coriolis_u", "cor_u", "cor_u", "u"),
        ("coriolis_v", "cor_v", "cor_v", "v"),
        ("trend_u", "trd_u", "trd_u", "u"),
        ("trend_v", "trd_v", "trd_v", "v"),
        ("u_exit", "u_exit", "ua_new", "u"),
        ("v_exit", "v_exit", "va_new", "v"),
        ("depth_u_exit", "face_depth_u_exit", "hu_e", "u"),
        ("depth_v_exit", "face_depth_v_exit", "hv_e", "v"),
        ("inverse_u_exit", "r1_face_depth_u_exit", "hur_e", "u"),
        ("inverse_v_exit", "r1_face_depth_v_exit", "hvr_e", "v"),
    )
    for substep in range(1, 66):
        prefix = f"j{substep:03d}_"
        for name, trace_name, record_name, face in registry:
            require(trace_name in trace, f"passive trace lacks {trace_name}")
            value = trace[trace_name][substep - 1]
            if face == "u":
                value = _native_u(value)
            elif face == "v":
                value = _native_v(value)
            add(name, value, oracle[prefix + record_name], active[face], substep=substep)
            if not rows[-1]["at_floor"]:
                return rows
    state = observed.state_after_barotropic
    add("transport_mean_u", _native_u(observed.transport_average[0]),
        oracle["o000_un_adv"], active["u"])
    add("transport_mean_v", _native_v(observed.transport_average[1]),
        oracle["o000_vn_adv"], active["v"])
    add("external_mean_u", _native_u(state.uu_b.data),
        oracle["o000_uu_b_aa"], active["u"])
    add("external_mean_v", _native_v(state.vv_b.data),
        oracle["o000_vv_b_aa"], active["v"])
    add("external_mean_ssh", state.eta.data, oracle["o000_ssh_aa"], active["t"])
    return rows


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "rank-placement":
        report["record_census"]["coverage"] = "overlap"
    elif plant == "record-bit":
        report["record_control"]["differing_cells"] = 0
    elif plant == "source-order":
        report["source_order"][0], report["source_order"][1] = (
            report["source_order"][1], report["source_order"][0])
    elif plant == "arm-identity":
        report["arm_order"][0], report["arm_order"][1] = (
            report["arm_order"][1], report["arm_order"][0])
    elif plant == "endpoint-ulp":
        report["endpoint_ulp_control"]["bit_exact"] = True

    require(report["claim_label"] == "independent hierarchy rung 0",
            "claim label moved")
    require(report["record_census"]["coverage"] == "exactly-once",
            "rank placement moved")
    require(report["record_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "record one-bit control did not fire")
    require(all(report["observer_passivity"].values()),
            "barotropic trace passivity moved")
    expected_order = list(ENTRY_ORDER)
    for substep in range(1, 66):
        expected_order.extend(f"{substep:03d}:{name}" for name in SUBSTEP_ORDER)
    expected_order.extend(EXIT_ORDER)
    require(report["source_order"] == expected_order[:len(report["source_order"])],
            "source order moved")
    require(report["arm_order"] == list(ARM_ORDER), "arm identity moved")
    require(report["endpoint_ulp_control"] == {
        "bit_exact": False, "differing_cells": 1},
        "endpoint one-ULP control did not fire")
    first = next((row for row in report["baseline_source_rows"]
                  if row.get("measured", True) and not row["at_floor"]), None)
    require(first == report["first_over_floor"], "first-debt selector moved")
    report["prediction_ledger"] = {
        "R178-P1": "CONFIRMED",
        "R178-P2": "CONFIRMED" if report["entry_and_histories_exact"] else "REFUTED",
        "R178-P3": "CONFIRMED" if report["history_arm_null"] else "REFUTED",
        "R178-P4": (
            "CONFIRMED" if first is not None and first["name"] in
            ("ssh_forcing", "u_forcing", "v_forcing") else "REFUTED"),
        "R178-P5": "CONFIRMED" if report["slow_arm_improves_endpoint"] else "REFUTED",
        "R178-P6": "CONFIRMED",
    }
    report["status"] = "HELD_FIRST_EXTERNAL_SOURCE_DEBT"
    return report


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-178 measurement requires its clean committed gate")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-178 walk requires production JIT on CPU")

    oracle, census = r97.assemble_record(spg_root)
    require(census["records"][0]["icycle"] == 65, "substep count moved")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = card.recipe.initial_state
    masks = phase3_gate.expected_masks(card)
    active = {
        "t": np.asarray(masks["ssh"], dtype=bool),
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }
    frame = rung0.assemble_frame(frame_root, 1, 0)
    candidate = rung0.candidate_fields(state)
    entry_active_masks = {
        "T": np.asarray(card.recipe.z_coord.is_active, dtype=bool),
        "S": np.asarray(card.recipe.z_coord.is_active, dtype=bool),
        "u": np.asarray(masks["u"], dtype=bool),
        "v": np.asarray(masks["v"], dtype=bool),
        "ssh": active["t"],
    }
    independent_entry = {
        name: _score(candidate[name], frame[name], entry_active_masks[name])
        for name in ("T", "S", "u", "v", "ssh")
    }
    independent_entry_storage = {
        name: {
            "bit_exact": bool(np.array_equal(candidate[name], frame[name])),
            "differing_cells": int(np.count_nonzero(
                np.asarray(candidate[name]).view(np.uint64)
                != np.asarray(frame[name]).view(np.uint64))),
            "max_abs": float(np.max(np.abs(
                np.asarray(candidate[name]) - np.asarray(frame[name])))),
        }
        for name in ("T", "S", "u", "v", "ssh")
    }
    independent_entry_exact = bool(
        all(row["bit_exact"] for row in independent_entry.values()))
    slow = (_to_model_u(oracle["i000_zu_frc"]),
            _to_model_v(oracle["i000_zv_frc"]))
    history = (
        _to_model_u(oracle["i000_ub_e"]),
        _to_model_u(oracle["i000_ubb_e"]),
        _to_model_v(oracle["i000_vb_e"]),
        _to_model_v(oracle["i000_vbb_e"]),
        oracle["i000_sshb_e"], oracle["i000_sshbb_e"],
    )
    freshwater, surface = rhs_walk._forcing(state.eta.data.shape)

    def run_arm(name: str):
        hooks = _NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=(slow if "slow" in name else None),
            barotropic_raw_history_override=(history if "history" in name else None),
        )
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return jax.device_get(model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    observed = {name: run_arm(name) for name in ARM_ORDER}

    # The returned trace is an instrument only after the identical solver run
    # without materialised substeps returns the same arrays bit-for-bit.
    baseline_slow = observed["baseline"].slow_forcing
    substep_dt = float(oracle["i000_entry_sc"][0])
    substep_count = int(oracle["i000_entry_sc"][2])

    def direct_solver(trace: bool):
        return jax.jit(lambda seed, f_eta, f_u, f_v: (
            barotropic_substeps_latlon_cgrid(
                seed, substep_dt, substep_count,
                card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config,
                F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
                add_barotropic_coriolis=True,
                u_now=seed.u.data, v_now=seed.v.data,
                _nemo_substep_trace_test_hook=trace,
            )
        ))(state, *baseline_slow)

    live_state, live_transport = jax.device_get(direct_solver(False))
    traced_state, traced_transport, _ = jax.device_get(direct_solver(True))
    passive = {
        "ssh": bool(np.array_equal(live_state.eta.data, traced_state.eta.data)),
        "u": bool(np.array_equal(live_state.u.data, traced_state.u.data)),
        "v": bool(np.array_equal(live_state.v.data, traced_state.v.data)),
        "u_barotropic": bool(np.array_equal(live_state.uu_b.data, traced_state.uu_b.data)),
        "v_barotropic": bool(np.array_equal(live_state.vv_b.data, traced_state.vv_b.data)),
        "transport_u": bool(np.array_equal(live_transport[0], traced_transport[0])),
        "transport_v": bool(np.array_equal(live_transport[1], traced_transport[1])),
    }
    require(all(passive.values()), "barotropic trace changes the observed solver")
    rows = _source_rows(observed["baseline"], oracle, active)
    source_order = [row["name"] if "substep" not in row else
                    f"{row['substep']:03d}:{row['name']}" for row in rows]
    first = next((row for row in rows if not row["at_floor"]), None)

    endpoint = {}
    for name, arm in observed.items():
        endpoint[name] = _score(
            arm.state_after_barotropic.eta.data,
            oracle["o000_ssh_aa"], active["t"])
        endpoint[name]["trace_digest"] = _trace_digest(arm)
    history_null = (
        endpoint["baseline"]["trace_digest"]
        == endpoint["history_only"]["trace_digest"])
    slow_history_null = (
        endpoint["slow_only"]["trace_digest"]
        == endpoint["slow_and_history"]["trace_digest"])
    improves = (
        endpoint["slow_only"]["rms"] < endpoint["baseline"]["rms"]
        and endpoint["slow_only"]["absolute_max"]
        < endpoint["baseline"]["absolute_max"]
        and slow_history_null)

    one = np.array([1.0], dtype=np.float64)
    next_one = np.nextafter(one, np.inf)
    report = {
        "format": "nemo-testcase-l4-orca2-round178-external-ssh-v1",
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "floor": float(FLOOR),
        "record_census": census,
        "record_control": {"bit_exact": bool(np.array_equal(one, next_one)),
                           "differing_cells": 1},
        "observer_passivity": passive,
        "independent_entry": independent_entry,
        "independent_entry_full_storage": independent_entry_storage,
        "arm_order": list(ARM_ORDER),
        "source_order": source_order,
        "baseline_source_rows": rows,
        "first_over_floor": first,
        "entry_and_histories_exact": bool(
            independent_entry_exact
            and all(row["bit_exact"] for row in rows[3:len(ENTRY_ORDER)])),
        "endpoint_scores": endpoint,
        "history_arm_null": bool(history_null and slow_history_null),
        "slow_arm_improves_endpoint": bool(improves),
        "endpoint_ulp_control": {
            "bit_exact": bool(np.array_equal(one, next_one)),
            "differing_cells": 1,
        },
        "compiled_source": {
            "forcing": "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:287-293",
            "initial_histories": "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:341-349",
            "substep_loop": "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:471-848",
            "weighted_endpoint": "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:888-937",
        },
        "worktree": stamp,
    }
    return classify(report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--spg-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.spg_root,
                         args.expect_commit)),
                    "measurement arguments missing")
            result = measure(
                args.deck_root, args.frame_root, args.spg_root,
                args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, GateError, r97.GateError,
            rhs_walk.GateError, rung0.GateError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'} "
              f"{args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Split OMT-4's direct post-drag V boundary without an executable observer."""

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
TESTCASES = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
if str(TESTCASES) not in sys.path:
    sys.path.insert(0, str(TESTCASES))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round236_slow_v_split as r236,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
    nemo_testcase_l4_orca2_round250_drag_split as r250,
    nemo_testcase_l4_orca2_round251_direct_slow_chain as r251,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round14_advmean import read_ordered  # noqa: E402
from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing  # noqa: E402

LABELS = r251.LABELS
SOURCE_ORDER = (
    "depth_mean_v", "live_reciprocal", "drag_coefficient",
    "bottom_velocity", "barotropic_velocity", "bottom_residual",
    "drag_increment", "unmasked_post_drag", "compact_surface_mask",
    "observed_post_drag", "resolved_surface_mask", "resolved_post_drag",
    "observed_final_slow_v", "resolved_final_slow_v",
)
PLANTS = (
    "none", "label-coverage", "production-route", "admission", "passivity",
    "override", "source-order", "owner-support", "unmasked-replay",
    "compact-replay", "resolved-closure",
)


class GateError(RuntimeError):
    """The direct boundary, its passive instrument, or a control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:, :90, ...]


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-252 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-252 measurement requires production JIT on CPU")
    return stamp


def _mask_product(value, mask) -> np.ndarray:
    import jax.numpy as jnp

    return np.asarray(jnp.asarray(value) * jnp.asarray(mask), dtype=np.float64)


def measure(deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
            label: str, expect_commit: str, *, candidate: bool = False,
            ) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_vface
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_bottom_drag_rate_faces,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    admission = r251._admit(twin_a, twin_b)
    slow = read_slow_forcing(
        twin_a / "oracle_slow_forcing_kt00000001.bin",
        dims=(94, 152, 31),
    )
    ordered = read_ordered(
        twin_a / "oracle_bt_ordered_operands_kt00000001.bin",
        expected_dims=(94, 152), expected_nrows=2,
    )
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65,
    )

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    traced, passivity = r251._run_direct(card, state, freshwater, surface)
    producer = traced.slow_forcing_producer

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 card lacks resolved raw NEMO operands")
    raw_e3v = np.asarray(raw.e3v_0, np.float64)[:, :90, :30]
    raw_vmask3 = np.asarray(raw.vmask, np.float64)[:, :90, :30]
    raw_hv = np.asarray(raw.hv_0, np.float64)[:, :90]
    raw_wet = (raw_hv > 0.0).astype(np.float64)
    stored_reciprocal = raw_wet / (raw_hv + 1.0 - raw_wet)
    resolved_surface_mask = np.max(raw_vmask3, axis=-1)
    compact_surface_mask = _native_v(state.v_mask.data)

    # This is the exact raw branch selected on the 35 fold faces by the
    # production depth-mean statement.  No alternate executable or observer
    # is introduced; the already-passive completed RHS is its sole live input.
    depth_mean = r236._candidate_depth(
        raw_e3v, _native_v(producer["rhs_v"]), raw_vmask3,
        stored_reciprocal,
    )

    h_k = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    ).astype(jnp.float64)
    h_v = min_cell_to_vface(h_k, card.recipe.grid)
    depth_v = jnp.maximum(jnp.sum(h_v, axis=-1), jnp.float64(1.0e-10))
    _, rate_v, _, indicator_v = nemo_bottom_drag_rate_faces(
        state.u.data, state.v.data, h_k, card.recipe.z_coord,
        card.recipe.model_config, card.recipe.grid,
    )
    bottom_v = jnp.sum(state.v.data * indicator_v, axis=-1)
    barotropic_v = jnp.sum(state.v.data * h_v, axis=-1) / depth_v
    depth_v, rate_v, bottom_v, barotropic_v = map(
        np.asarray, jax.device_get((depth_v, rate_v, bottom_v, barotropic_v)))

    oracle_v = np.asarray(entry["v"], np.float64)[:, :90, :]
    oracle_indicator = np.zeros_like(raw_vmask3)
    counts = np.sum(raw_vmask3 != 0.0, axis=-1, dtype=np.int64)
    levels = np.maximum(counts - 1, 0)
    np.put_along_axis(
        oracle_indicator, levels[..., None],
        (counts > 0)[..., None].astype(np.float64), axis=-1,
    )

    candidate = {
        "depth_mean_v": depth_mean,
        "live_reciprocal": _native_v(1.0 / depth_v),
        "drag_coefficient": -_native_v(rate_v),
        "bottom_velocity": _native_v(bottom_v),
        "barotropic_velocity": _native_v(barotropic_v),
        "compact_surface_mask": compact_surface_mask,
        "observed_post_drag": _native_v(producer["post_drag_v"]),
        "resolved_surface_mask": resolved_surface_mask,
    }
    oracle = {
        "depth_mean_v": np.asarray(slow["depth_v"], np.float64),
        "live_reciprocal": np.asarray(slow["r1_hv"], np.float64),
        "drag_coefficient": np.asarray(slow["cd_v"], np.float64),
        "bottom_velocity": r250._bottom_value(oracle_v, oracle_indicator),
        "barotropic_velocity": np.asarray(ordered["v_entry"][0], np.float64),
        "compact_surface_mask": resolved_surface_mask,
        "observed_post_drag": np.asarray(slow["post_drag_v"], np.float64),
        "resolved_surface_mask": resolved_surface_mask,
    }
    candidate_chain = r250._drag_chain(
        candidate["depth_mean_v"], candidate["live_reciprocal"],
        candidate["drag_coefficient"], candidate["bottom_velocity"],
        candidate["barotropic_velocity"],
    )
    oracle_chain = r250._drag_chain(
        oracle["depth_mean_v"], oracle["live_reciprocal"],
        oracle["drag_coefficient"], oracle["bottom_velocity"],
        oracle["barotropic_velocity"],
    )
    for name in ("bottom_residual", "drag_increment"):
        candidate[name] = candidate_chain[name]
        oracle[name] = oracle_chain[name]
    candidate["unmasked_post_drag"] = candidate_chain["post_drag"]
    oracle["unmasked_post_drag"] = oracle_chain["post_drag"]
    compact_replay = _mask_product(
        candidate["unmasked_post_drag"], compact_surface_mask)
    resolved_post_drag = _mask_product(
        candidate["unmasked_post_drag"], resolved_surface_mask)
    candidate["resolved_post_drag"] = resolved_post_drag
    oracle["resolved_post_drag"] = oracle["observed_post_drag"]
    candidate["observed_final_slow_v"] = _native_v(producer["final_v"])
    oracle["observed_final_slow_v"] = np.asarray(
        substeps["slow_v"][0], np.float64)

    wind_increment = r251._wind_increment(
        producer["wind_r1_rho0"], _native_v(producer["wind_tau_v"]),
        raw_wet / (raw_hv + 1.0 - raw_wet),
    )
    resolved_post_wind = np.asarray(
        resolved_post_drag + wind_increment, dtype=np.float64)
    resolved_final = r236._source_final(
        resolved_post_wind, _native_v(producer["coriolis_v"]),
        resolved_surface_mask,
    )
    candidate["resolved_final_slow_v"] = resolved_final
    oracle["resolved_final_slow_v"] = np.asarray(
        substeps["slow_v"][0], np.float64)

    support_mask = compact_surface_mask != resolved_surface_mask
    require(int(np.count_nonzero(support_mask)) == 35,
            f"registered mask support moved: {np.count_nonzero(support_mask)} != 35")
    require(int(np.count_nonzero(support_mask[:-3])) == 0,
            "registered mask support escaped the fold band")
    rows = {name: r236._row(candidate[name], oracle[name], support_mask)
            for name in SOURCE_ORDER}
    replays = {
        "compact_mask_to_observed": r236._row(
            compact_replay, candidate["observed_post_drag"], support_mask),
        "unmasked_to_oracle": r236._row(
            candidate["unmasked_post_drag"], oracle["observed_post_drag"], support_mask),
        "resolved_post_drag_to_oracle": r236._row(
            resolved_post_drag, oracle["observed_post_drag"], support_mask),
        "resolved_final_to_oracle": r236._row(
            resolved_final, oracle["resolved_final_slow_v"], support_mask),
    }
    mask_delta = compact_surface_mask != resolved_surface_mask
    changed = compact_replay != resolved_post_drag
    support = {
        "registered_unequal": int(np.count_nonzero(support_mask)),
        "observed_post_drag_unequal": int(np.count_nonzero(
            candidate["observed_post_drag"] != oracle["observed_post_drag"])),
        "observed_final_unequal": int(np.count_nonzero(
            candidate["observed_final_slow_v"]
            != oracle["observed_final_slow_v"])),
        "mask_unequal": int(np.count_nonzero(mask_delta)),
        "changed_unequal": int(np.count_nonzero(changed)),
        "mask_outside_registered": int(np.count_nonzero(
            mask_delta & ~support_mask)),
        "changed_outside_registered": int(np.count_nonzero(
            changed & ~support_mask)),
    }
    first_output = next((name for name in (
        "depth_mean_v", "drag_increment", "unmasked_post_drag",
        "observed_post_drag") if not rows[name]["bit_exact"]), None)
    return {
        "format": "nemo-testcase-l4-orca2-round252-post-drag-split-v1",
        "status": "PASS_R252_POST_DRAG_SPLIT",
        "label": label,
        "claim_mode": "resolved-mask-candidate" if candidate else "baseline",
        "operand_route": "production-d114-resolved",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "worktree": stamp,
        "record_admission": admission,
        "override_absence": {
            "slow_forcing_incoming_override": True,
            "barotropic_slow_forcing_override": True,
            "slow_forcing_drag_override": True,
        },
        "passivity": passivity,
        "source_order": list(SOURCE_ORDER),
        "support": support,
        "rows": rows,
        "replays": replays,
        "first_nonbit_output": first_output,
        "compiled_source": {
            "depth_mean": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:194-210",
            "drag": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1426-1464",
            "final_mask": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:286-324",
        },
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    report = by_label["independent"]
    if plant == "production-route":
        report["operand_route"] = "retired-route"
    elif plant == "admission":
        report["record_admission"][0]["status"] = "MOVED"
    elif plant == "passivity":
        report["passivity"][next(iter(report["passivity"]))] = False
    elif plant == "override":
        report["override_absence"]["slow_forcing_drag_override"] = False
    elif plant == "source-order":
        report["source_order"][0:2] = reversed(report["source_order"][0:2])
    elif plant == "owner-support":
        report["support"]["registered_unequal"] = 34
    elif plant == "unmasked-replay":
        report["replays"]["unmasked_to_oracle"]["bit_exact"] = False
    elif plant == "compact-replay":
        report["replays"]["compact_mask_to_observed"]["bit_exact"] = False
    elif plant == "resolved-closure":
        report["replays"]["resolved_final_to_oracle"]["bit_exact"] = False

    modes = {item.get("claim_mode", "baseline") for item in by_label.values()}
    require(len(modes) == 1, "claim labels disagree on baseline/candidate mode")
    mode = next(iter(modes))
    require(mode in ("baseline", "resolved-mask-candidate"),
            f"unknown claim mode {mode!r}")
    summary = {}
    for label, item in by_label.items():
        require(item["status"] == "PASS_R252_POST_DRAG_SPLIT",
                f"{label}: measurement status moved")
        require(item["operand_route"] == "production-d114-resolved",
                f"{label}: production operand route moved")
        require(all(row["status"] == "EXACT_DEFINED_BYTES"
                    for row in item["record_admission"]),
                f"{label}: record admission moved")
        require(all(item["passivity"].values()), f"{label}: passivity moved")
        require(all(item["override_absence"].values()),
                f"{label}: an override is active")
        require(tuple(item["source_order"]) == SOURCE_ORDER,
                f"{label}: source order moved")
        # The production candidate removes the 1.66e-6 physical residual but
        # retains 15 last-bit differences (2.12e-22 maximum) under both claim
        # labels.  Keep that failed exact-closure prediction in the gate: the
        # Decision-96 bar class is accepted only with this frozen census and
        # the ordinary at-floor predicate, never by relabelling it exact.
        expected_observed = 35 if mode == "baseline" else 15
        require(item["support"] == {
            "registered_unequal": 35,
            "observed_post_drag_unequal": expected_observed,
            "observed_final_unequal": expected_observed,
            "mask_unequal": 35, "changed_unequal": 35,
            "mask_outside_registered": 0, "changed_outside_registered": 0,
        }, f"{label}: registered/mask support moved")
        rows = item["rows"]
        require(rows["depth_mean_v"]["bit_exact"],
                f"{label}: pre-drag depth mean is not exact")
        require(rows["drag_increment"]["bit_exact"],
                f"{label}: literal drag increment is not exact")
        require(rows["unmasked_post_drag"]["bit_exact"],
                f"{label}: unmasked drag addition is not exact")
        require(not rows["compact_surface_mask"]["bit_exact"],
                f"{label}: compact-mask discriminator moved")
        require(rows["resolved_surface_mask"]["bit_exact"]
                and rows["resolved_post_drag"]["bit_exact"]
                and rows["resolved_final_slow_v"]["bit_exact"],
                f"{label}: resolved-mask closure moved")
        replays = item["replays"]
        require((replays["compact_mask_to_observed"]["bit_exact"])
                == (mode == "baseline"),
                f"{label}: compact-mask baseline/candidate discriminator moved")
        require(replays["unmasked_to_oracle"]["bit_exact"],
                f"{label}: unmasked addition does not replay NEMO")
        require(replays["resolved_post_drag_to_oracle"]["bit_exact"]
                and replays["resolved_final_to_oracle"]["bit_exact"],
                f"{label}: resolved mask does not close the direct chain")
        require(item["first_nonbit_output"] == "observed_post_drag",
                f"{label}: first non-bit output moved")
        require(not rows["observed_post_drag"]["bit_exact"]
                and not rows["observed_final_slow_v"]["bit_exact"],
                f"{label}: observed direct-chain bit census moved")
        if mode == "resolved-mask-candidate":
            require(rows["observed_post_drag"]["at_floor"]
                    and rows["observed_final_slow_v"]["at_floor"],
                    f"{label}: candidate direct chain left the fixed floor")
        summary[label] = {
            "rows": rows, "replays": replays, "support": item["support"],
            "first_nonbit_output": item["first_nonbit_output"],
        }
    require(summary["independent"] == summary["given_nemo_entry"],
            "claim labels disagree")
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round252-classification-v1",
        "status": (
            "AT_BAR_R252_RESOLVED_MASK_CANDIDATE"
            if mode == "resolved-mask-candidate"
            else "CONFIRMED_R252_COMPACT_DRAG_MASK_OWNER"),
        "source_order": list(SOURCE_ORDER),
        "labels": summary,
        "predictions": {
            "R252-P1": "CONFIRMED_ADMITTED_PASSIVE_NO_OVERRIDE",
            "R252-P2": "CONFIRMED_COMPACT_DRAG_MASK_FIRST_NONBIT_OUTPUT",
            "R252-P3": (
                "REFUTED_BIT_EXACT__AT_BAR_PRODUCTION_MASK_ASSOCIATION"
                if mode == "resolved-mask-candidate"
                else "CONFIRMED_OFFLINE_RESOLVED_MASK_CLOSES_DIRECT_CHAIN"),
            "R252-P4": "NOT_REACHED_PENDING_PRODUCTION_CANDIDATE",
            "R252-P5": "NOT_REACHED",
            "R252-P6": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), required=True)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--reports", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)), "measurement arguments incomplete")
            require(args.plant == "none", "plants apply only to classification")
            result = measure(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit, candidate=args.candidate,
            )
        else:
            require(args.reports is not None, "classification reports missing")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.reports
            ], plant=args.plant)
    except (OSError, KeyError, TypeError, ValueError, GateError,
            r236.GateError, r249.GateError, r250.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

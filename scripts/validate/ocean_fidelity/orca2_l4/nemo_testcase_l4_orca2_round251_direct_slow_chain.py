#!/usr/bin/env python3
"""Re-measure OMT-4's direct slow-V chain without a forcing override."""

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
    nemo_testcase_l4_orca2_o1_acquisition_gate as o1_gate,
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round236_slow_v_split as r236,
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.testcases import (  # noqa: E402
    nemo_testcase_l2_gyre_phase3_gate as phase3,
)
from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing  # noqa: E402

LABELS = ("independent", "given_nemo_entry")
SOURCE_ORDER = (
    "pre_drag_v", "post_drag_v", "wind_increment_v", "post_wind_v",
    "incoming_copy_v", "coriolis_v", "surface_v_mask", "final_slow_v",
)
PLANTS = (
    "none", "label-coverage", "admission", "passivity", "override",
    "support", "source-order", "candidate-replay", "oracle-replay",
)


class GateError(RuntimeError):
    """The passive direct chain or a frozen control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _native_v(value) -> np.ndarray:
    return r97._native_v(np.asarray(value, dtype=np.float64))[:, :90]


def _admit(twin_a: Path, twin_b: Path) -> list[dict[str, object]]:
    masks = o1_gate._defined_masks(twin_a)
    rows = []
    for name in (
        "oracle_slow_forcing_kt00000001.bin",
        "oracle_bt_substeps_kt00000001.bin",
    ):
        left, right = twin_a / name, twin_b / name
        require(left.is_file() and right.is_file(), f"missing inherited {name}")
        exact = o1_gate._compare_hygiene_record(left, right, masks)
        require(exact["status"] == "EXACT_DEFINED_BYTES", f"{name}: twins differ")
        rows.append({"name": name, **exact})
    return rows


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-251 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-251 measurement requires production JIT on CPU")
    return stamp


def _run_direct(card, state, freshwater, surface):
    """Run the unmodified executable and its passive producer trace."""

    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
        _NEMOWSRK3TestHooks,
    )

    ordinary_hooks = _NEMOWSRK3TestHooks()
    traced_hooks = _NEMOWSRK3TestHooks(expose_live_stage_operands=True)
    for hooks in (ordinary_hooks, traced_hooks):
        require(hooks.slow_forcing_incoming_override is None,
                "slow-forcing incoming override is active")
        require(hooks.barotropic_slow_forcing_override is None,
                "final slow-forcing override is active")
        require(hooks.slow_forcing_drag_override is None,
                "drag-output override is active")
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=ordinary_hooks)
    observed = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=traced_hooks)
    plain = jax.device_get(ordinary.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced = jax.device_get(observed.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live operand trace return type moved")
    return traced, passive._ordinary_state_equal(traced.state_after, plain)


def _wind_increment(rho, tau, inverse_depth) -> np.ndarray:
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round

    b = nemo_source_round
    return np.asarray(b(b(jnp.asarray(rho) * jnp.asarray(tau))
                        * jnp.asarray(inverse_depth)), dtype=np.float64)


def measure(deck_root: Path, frame_root: Path, twin_a: Path, twin_b: Path,
            label: str, expect_commit: str, *, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    admission = _admit(twin_a, twin_b)
    slow = read_slow_forcing(
        twin_a / "oracle_slow_forcing_kt00000001.bin", dims=(94, 152, 31))
    substeps = phase3.read_bt_substeps(
        twin_a / "oracle_bt_substeps_kt00000001.bin",
        expected_dims=(94, 152), expected_ncycle=65)

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    traced, passivity = _run_direct(card, state, freshwater, surface)
    producer = traced.slow_forcing_producer

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 card lacks raw NEMO operands")
    raw_hv = np.asarray(raw.hv_0, np.float64)[:, :90]
    wet_v = (raw_hv > 0.0).astype(np.float64)
    raw_ssvmask = np.max(np.asarray(raw.vmask, np.float64)[:, :90], axis=-1)
    actual_ssvmask = _native_v(state.v_mask.data)
    support = raw_ssvmask != actual_ssvmask

    candidate_increment = _wind_increment(
        producer["wind_r1_rho0"], _native_v(producer["wind_tau_v"]),
        wet_v / (raw_hv + 1.0 - wet_v))
    candidate_post_drag = _native_v(producer["post_drag_v"])
    candidate_post_wind = np.asarray(candidate_post_drag + candidate_increment,
                                     dtype=np.float64)
    candidate = {
        # legoESM applies its (zero on OMT-4) wind term before drag.  This is
        # therefore the executing input to its drag statement, and may be
        # compared with NEMO's recorded pre-drag depth mean on this deck.
        "pre_drag_v": _native_v(producer["post_wind_v"]),
        "post_drag_v": candidate_post_drag,
        "wind_increment_v": candidate_increment,
        "post_wind_v": candidate_post_wind,
        "incoming_copy_v": _native_v(producer["incoming_v"]),
        "coriolis_v": _native_v(producer["coriolis_v"]),
        "surface_v_mask": actual_ssvmask,
        "final_slow_v": _native_v(producer["final_v"]),
    }
    oracle_increment = _wind_increment(
        slow["r1_rho0"], slow["vtau"], slow["r1_hv"])
    oracle = {
        "pre_drag_v": slow["depth_v"],
        "post_drag_v": slow["post_drag_v"],
        "wind_increment_v": oracle_increment,
        "post_wind_v": slow["post_wind_v"],
        "incoming_copy_v": slow["post_wind_v"],
        "coriolis_v": np.asarray(substeps["cor_v"][0], np.float64),
        "surface_v_mask": raw_ssvmask,
        "final_slow_v": np.asarray(substeps["slow_v"][0], np.float64),
    }
    if plant == "admission":
        admission[0]["status"] = "MOVED"
    elif plant == "passivity":
        passivity["eta"] = False
    elif plant == "override":
        raise GateError("override plant fired")
    elif plant == "support":
        support = np.array(support, copy=True)
        support[0, 0] = ~support[0, 0]
    elif plant == "candidate-replay":
        candidate["final_slow_v"] = np.array(candidate["final_slow_v"], copy=True)
        candidate["final_slow_v"][-1, 0] = np.nextafter(
            candidate["final_slow_v"][-1, 0], np.inf)
    elif plant == "oracle-replay":
        oracle["final_slow_v"] = np.array(oracle["final_slow_v"], copy=True)
        oracle["final_slow_v"][-1, 0] = np.nextafter(
            oracle["final_slow_v"][-1, 0], np.inf)

    require(all(row["status"] == "EXACT_DEFINED_BYTES" for row in admission),
            "record admission moved")
    require(all(passivity.values()), "passive trace moved completed state")
    require(int(np.count_nonzero(support)) == 35,
            f"registered fold support moved: {np.count_nonzero(support)} != 35")
    require(int(np.count_nonzero(support[:-3])) == 0,
            "registered support escaped the fold band")
    rows = {name: r236._row(candidate[name], oracle[name], support)
            for name in SOURCE_ORDER}
    candidate_replay = r236._source_final(
        candidate["incoming_copy_v"], candidate["coriolis_v"],
        candidate["surface_v_mask"])
    oracle_replay = r236._source_final(
        oracle["incoming_copy_v"], oracle["coriolis_v"],
        oracle["surface_v_mask"])
    replay = {
        "candidate": r236._row(
            candidate_replay, candidate["final_slow_v"], support),
        "oracle": r236._row(oracle_replay, oracle["final_slow_v"], support),
    }
    require(replay["candidate"]["bit_exact"],
            "candidate direct operands do not replay its final slow forcing")
    require(replay["oracle"]["bit_exact"],
            "oracle direct operands do not replay its final slow forcing")
    first = next((name for name in SOURCE_ORDER if not rows[name]["bit_exact"]), None)
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round251-direct-slow-chain-v1",
        "status": "PASS_R251_DIRECT_SLOW_CHAIN",
        "label": label,
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
        "support": {"unequal": int(np.count_nonzero(support)), "fold_only": True},
        "rows": rows,
        "self_replay": replay,
        "first_unequal": first,
        "compiled_source": {
            "drag_wind": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:198-230",
            "coriolis_mask": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:286-324",
        },
    }


def classify(reports: list[dict], *, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    if plant == "source-order":
        by_label["independent"]["source_order"][0:2] = reversed(
            by_label["independent"]["source_order"][0:2])
    for label, report in by_label.items():
        require(report["status"] == "PASS_R251_DIRECT_SLOW_CHAIN",
                f"{label}: measurement status moved")
        require(tuple(report["source_order"]) == SOURCE_ORDER,
                f"{label}: source order moved")
        require(report["support"] == {"unequal": 35, "fold_only": True},
                f"{label}: support moved")
        require(all(report["passivity"].values()), f"{label}: passivity moved")
        require(all(report["override_absence"].values()),
                f"{label}: an invalid override is active")
        require(report["self_replay"]["candidate"]["bit_exact"],
                f"{label}: candidate self-replay moved")
        require(report["self_replay"]["oracle"]["bit_exact"],
                f"{label}: oracle self-replay moved")
    first = {label: report["first_unequal"] for label, report in by_label.items()}
    require(len(set(first.values())) == 1, "claim labels disagree on first boundary")
    first_name = next(iter(first.values()))
    direct_exact = first_name is None
    require(plant == "none", f"{plant} plant stayed green")
    return {
        "format": "nemo-testcase-l4-orca2-round251-direct-classification-v1",
        "status": ("PASS_R251_DIRECT_CHAIN_EXACT" if direct_exact
                   else f"HELD_R251_DIRECT_CHAIN_FIRST_{first_name.upper()}"),
        "source_order": list(SOURCE_ORDER),
        "first_unequal": first_name,
        "rows": {
            label: {"first_unequal": report["first_unequal"], "rows": report["rows"]}
            for label, report in by_label.items()
        },
        "predictions": {
            "R251-P1": "CONFIRMED_ADMITTED_PASSIVE_NO_OVERRIDE",
            "R251-P2": ("CONFIRMED_DIRECT_CHAIN_EXACT" if direct_exact
                        else f"REFUTED_DIRECT_CHAIN_FIRST_{first_name.upper()}"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--classify", nargs=2, type=Path,
                        metavar=("INDEPENDENT", "GIVEN"))
    parser.add_argument("--classify-plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify:
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify
            ], plant=args.classify_plant)
        else:
            require(all(value is not None for value in (
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit)), "measurement arguments incomplete")
            result = measure(
                args.deck_root, args.frame_root, args.twin_a, args.twin_b,
                args.label, args.expect_commit, plant=args.plant)
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        planted = (args.classify_plant != "none" if args.classify
                   else args.plant != "none")
        print(f"STATUS {'PLANT-FIRED' if planted else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

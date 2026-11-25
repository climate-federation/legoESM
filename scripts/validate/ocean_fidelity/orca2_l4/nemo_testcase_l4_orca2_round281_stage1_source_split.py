#!/usr/bin/env python3
"""Split OMT-4's stage-1 V source row at matched NEMO program points."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round170_slow_producer_walk as r170,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round176_stage1_offline_replay as r176,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round280_stage1_acquisition import (  # noqa: E402,E501
    check_record as admission_gate,
)

SOURCE_ROW = 146
SOURCE_ORDER = ("entry_v", "pre_v", "cor_v", "final_v")
PLANTS = (
    "none", "admission", "source-order", "matched-point", "raw-bit",
    "corrected-bit", "final-bit", "source-row", "one-bit",
)


class GateError(RuntimeError):
    """A frozen source-split predicate or instrument control moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _assemble_values(root: Path) -> dict[str, np.ndarray]:
    """Reuse round 176's self-describing parser and stitch owned rank slabs."""

    arrays: dict[str, np.ndarray] = {}
    coverage = np.zeros((148, 180), dtype=np.int8)
    for rank in (0, 1):
        path = root / f"oracle_r175_stage1_rank{rank:04d}_kt00000001.bin"
        metadata, values = r176._payload_values(path)
        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"rank {rank}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, block in values.items():
            shape = ((148, 180) if block.ndim == 2
                     else (148, 180, block.shape[-1]))
            arrays.setdefault(name, np.empty(shape, dtype=np.float64))[
                j0:j1, i0:i1, ...] = block
    require(bool(np.all(coverage == 1)), "rank slabs are not exactly-once")
    require(tuple(arrays) == admission_gate.schema.NAMES,
            "stitched field order moved")
    return arrays


def _exact_score(left, right, support=None) -> dict[str, object]:
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    require(left.shape == right.shape, f"score shape mismatch {left.shape} != {right.shape}")
    if support is None:
        support = np.ones(left.shape, dtype=bool)
    support = np.asarray(support, dtype=bool)
    require(support.shape == left.shape and bool(np.any(support)),
            "score support is empty or moved")
    lval, rval = left[support], right[support]
    require(bool(np.isfinite(lval).all() and np.isfinite(rval).all()),
            "non-finite value on scored support")
    unequal = lval.view(np.uint64) != rval.view(np.uint64)
    delta = np.abs(lval - rval)
    position = int(np.argmax(delta))
    flat_support = np.flatnonzero(support)
    argmax = np.unravel_index(int(flat_support[position]), left.shape)
    return {
        "bit_exact": bool(not np.any(unequal)),
        "unequal": int(np.count_nonzero(unequal)),
        "count": int(lval.size),
        "max_abs": float(delta[position]),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "argmax": [int(value) for value in argmax],
    }


def _one_bit_control(shape: tuple[int, ...]) -> dict[str, object]:
    left = np.zeros(shape, dtype=np.float64)
    right = left.copy()
    right.flat[0] = np.nextafter(np.float64(0.0), np.float64(1.0))
    return _exact_score(left, right)


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    report = copy.deepcopy(report)
    if plant == "admission":
        report["admission"]["status"] = "BROKEN"
    elif plant == "source-order":
        report["source_order"][1], report["source_order"][2] = (
            report["source_order"][2], report["source_order"][1])
    elif plant == "matched-point":
        report["program_points"]["cor_v"] = "after_lbc_lnk"
    elif plant == "raw-bit":
        report["controls"]["pre_v"]["unequal"] = 0
    elif plant == "corrected-bit":
        report["controls"]["cor_v"]["unequal"] = 0
    elif plant == "final-bit":
        report["controls"]["final_v"]["unequal"] = 0
    elif plant == "source-row":
        report["source_row"] -= 1
    elif plant == "one-bit":
        report["controls"]["one_bit"]["unequal"] = 0

    require(report["claim_label"] == "independent_omt4", "claim label moved")
    require(report["admission"]["status"] == "PASS_R280_OMT4_STAGE1_ADMISSION",
            "record admission moved")
    require(tuple(report["source_order"]) == SOURCE_ORDER, "source order moved")
    require(report["source_row"] == SOURCE_ROW, "source-row registry moved")
    require(report["program_points"] == {
        "pre_v": "after vector update, before barotropic correction",
        "cor_v": "after barotropic correction, before lbc_lnk",
        "final_v": "after completed-stage lbc_lnk in NEMO",
    }, "matched program-point registry moved")
    for name in ("pre_v", "cor_v", "final_v", "one_bit"):
        require(report["controls"][name]["unequal"] == 1,
                f"{name} one-bit control did not fire")

    pre = report["boundaries"]["pre_v"]["source_row"]
    cor = report["boundaries"]["cor_v"]["source_row"]
    final = report["boundaries"]["final_v"]["source_row"]
    prior = report["round279_source_row"]
    calibration_equal = (
        final["unequal"] == prior["unequal"]
        and final["max_abs"] == prior["max_abs"]
    )
    report["calibration_reconciliation"] = {
        "reproduced": calibration_equal,
        "offline_protocol": "independent entry with NEMO rhs_v and ext_vb",
        "round279_protocol": "independent live stage through an in-executable observer",
        "disposition": ("REPRODUCED" if calibration_equal else
                        "ROUND279_SOURCE_ROW_UNTRUSTED_OBSERVER_NOT_PASSIVE"),
    }

    p2 = pre["unequal"] == 0 and report["boundaries"]["pre_v"]["active"]["unequal"] == 0
    p3 = (p2 and cor["unequal"] == prior["unequal"]
          and cor["max_abs"] == prior["max_abs"])
    p4 = p2 and cor["unequal"] == 0 and calibration_equal
    report["prediction_ledger"] = {
        "R281-P1": "CONFIRMED",
        "R281-P2": "CONFIRMED" if p2 else "REFUTED",
        "R281-P3": "CONFIRMED" if p3 else "REFUTED",
        "R281-P4": "CONFIRMED" if p4 else "REFUTED",
        "R281-P5": ("NOT_REACHED_POST_STAGE_OWNER" if p4 else
                     "STOPPED_MISSING_CORRECTION_OPERANDS" if p3 else
                     "OPEN_AT_EARLIER_BOUNDARY"),
    }
    valued = next((name for name in SOURCE_ORDER
                   if (report["boundaries"][name]["active"]["max_abs"] > 2.0e-10
                       or report["boundaries"][name]["source_row"]["max_abs"]
                       > 2.0e-10)), None)
    if not calibration_equal and valued is None:
        report["first_nonbit_statement"] = {
            "name": "none valued in the admitted offline source replay",
            "status": "HELD_INSTRUMENT_NOT_PASSIVE",
        }
        report["status"] = "HELD_INSTRUMENT_NOT_PASSIVE"
    elif p4:
        report["first_nonbit_statement"] = {
            "name": "completed-stage V fold association",
            "nemo_source": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:818-819",
            "status": "HELD_CITED_STATEMENT",
        }
        report["status"] = "HELD_FIRST_NONBIT_POST_STAGE_V_ASSOCIATION"
    elif p3:
        report["first_nonbit_statement"] = {
            "name": "stage-1 barotropic V correction",
            "nemo_source": "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:513-537",
            "status": "STOPPED_MISSING_LIVE_E3V_AND_R1_HV_RECORD",
        }
        report["status"] = "STOPPED_FOR_RECORD"
    else:
        first = valued or next(
            name for name in SOURCE_ORDER
            if report["boundaries"][name]["active"]["unequal"] > 0)
        statement = {
            "pre_v": (
                "stage-1 vector-form V update",
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:477-480",
            ),
            "cor_v": (
                "stage-1 barotropic V correction",
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:521-528",
            ),
            "final_v": (
                "completed-stage V fold association",
                "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/"
                "stprk3_stg.f90:818-819",
            ),
        }.get(first)
        report["first_nonbit_statement"] = {
            "name": statement[0] if statement else first,
            "boundary": first,
            "status": "HELD_CITED_STATEMENT" if statement else "HELD",
            **({"nemo_source": statement[1]} if statement else {}),
        }
        report["status"] = "HELD_EARLIER_SOURCE_BOUNDARY"
    return report


def measure(deck_root: Path, record_root: Path, baseline_root: Path,
            prior_report: Path, expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_common import (
        rk3_stage_barotropic_correction,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        rk3_stage_velocity_update,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-281 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu", "round 281 is CPU-only")

    admission = admission_gate.run(record_root, baseline_root)
    oracle = _assemble_values(record_root)
    prior = json.loads(prior_report.read_text(encoding="utf-8"))
    prior_source = prior["top_v"]["top_v_source_operand"]["candidate_source_vs_nemo"]
    require(prior["status"] == "PASS_R279_RUNOFF_AND_TOPV_DIAGNOSIS",
            "round-279 calibration report moved")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    cfg = card.recipe.model_config
    require(not cfg.rk3_fold_exchange_top_v_row,
            "Decision 125 top-V selector is unexpectedly enabled")
    state = card.recipe.initial_state
    ops = r170._candidate_reference_operands(card, state)
    active_v = np.asarray(ops["mask_v"] != 0.0)
    require(active_v.shape == (148, 180, 31),
            f"production V support moved: {active_v.shape}")
    entry_v = np.asarray(rung0.candidate_fields(state)["v"], np.float64)
    pre_v = jax.device_get(rk3_stage_velocity_update(
        jnp.asarray(entry_v), jnp.asarray(oracle["rhs_v"][..., :-1]),
        card.dt_s / 3.0, jnp.asarray(active_v[..., :-1]),
        vector_form=True, written_order=cfg.rk3_stage_velocity_written_order,
    ))
    cor_v = jax.device_get(rk3_stage_barotropic_correction(
        jnp.asarray(pre_v), jnp.asarray(oracle["ext_vb"]),
        jnp.asarray(ops["e3_v"][..., :-1]),
        jnp.asarray(ops["r1_h0_v"]),
        jnp.asarray(ops["mask_v"][..., :-1]),
        cfg.rk3_stage_velocity_written_order,
    ))

    candidate = {
        "entry_v": r176._with_jpk_zero(entry_v),
        "pre_v": r176._with_jpk_zero(pre_v),
        "cor_v": r176._with_jpk_zero(cor_v),
        # Decision 125 keeps the subsequent completed-stage association off,
        # so this is also the candidate's completed V at the matched boundary.
        "final_v": r176._with_jpk_zero(cor_v),
    }
    entry = rung0.assemble_frame(record_root, 1, 0)
    expected = {
        "entry_v": r176._with_jpk_zero(entry["v"]),
        "pre_v": oracle["pre_v"],
        "cor_v": oracle["cor_v"],
        "final_v": oracle["final_v"],
    }
    raw_mesh = card.recipe.z_coord.nemo_een_barotropic
    require(raw_mesh is not None, "OMT-4 lacks raw NEMO mesh operands")
    active = np.concatenate((np.asarray(raw_mesh.vmask, dtype=bool),
                             np.zeros((148, 180, 1), dtype=bool)), axis=-1)
    require(active.shape == candidate["pre_v"].shape, "resolved V support moved")
    boundaries = {}
    for name in SOURCE_ORDER:
        boundaries[name] = {
            "active": _exact_score(candidate[name], expected[name], active),
            "source_row": _exact_score(
                candidate[name][SOURCE_ROW, :, :-1],
                expected[name][SOURCE_ROW, :, :-1]),
        }

    controls = {
        "pre_v": _one_bit_control(expected["pre_v"].shape),
        "cor_v": _one_bit_control(expected["cor_v"].shape),
        "final_v": _one_bit_control(expected["final_v"].shape),
        "one_bit": _one_bit_control((1,)),
    }
    return classify({
        "format": "nemo-testcase-l4-orca2-round281-stage1-source-split-v1",
        "claim_label": "independent_omt4",
        "worktree": stamp,
        "admission": admission,
        "source_order": list(SOURCE_ORDER),
        "source_row": SOURCE_ROW,
        "program_points": {
            "pre_v": "after vector update, before barotropic correction",
            "cor_v": "after barotropic correction, before lbc_lnk",
            "final_v": "after completed-stage lbc_lnk in NEMO",
        },
        "boundaries": boundaries,
        "round279_source_row": {
            "bit_exact": prior_source["unequal"] == 0,
            "unequal": prior_source["unequal"],
            "count": 5400,
            "max_abs": prior_source["max_abs"],
            "rms": prior_source["rms"],
            "argmax": [SOURCE_ROW, *prior_source["argmax"]],
        },
        "execution": "offline replay of shared pure operators; no executable observer",
        "recorded_inputs": {
            "rhs_v": admission["records"][0]["fields"]["rhs_v"]["payload_sha256"],
            "ext_vb": admission["records"][0]["fields"]["ext_vb"]["payload_sha256"],
        },
        "controls": controls,
        "operand_availability_if_correction_owns": {
            "recorded": ["pre_v", "ext_vb", "cor_v"],
            "missing": ["live_e3v_3d", "r1_hv_0", "stage_vmask"],
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.record_root, args.baseline_root,
                         args.prior_report, args.expect_commit)),
                    "measurement arguments missing")
            result = measure(
                args.deck_root, args.record_root, args.baseline_root,
                args.prior_report, args.expect_commit)
        else:
            require(args.report_in is not None, "classification needs --report-in")
            result = classify(
                json.loads(args.report_in.read_text(encoding="utf-8")),
                args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, admission_gate.Refusal, admission_gate.schema.Refusal,
            omt4.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

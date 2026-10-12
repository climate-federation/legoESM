#!/usr/bin/env python3
"""Split the residual kt=1 stage-2 tracer boundary after exact V association."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as r232,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
)

LABELS = r249.LABELS
PLANTS = (
    "none", "label-coverage", "provenance", "entry", "transport-record",
    "transport-replay", "causal-closure", "one-bit",
)
NX, NY, NZ = 94, 152, 31


class GateError(RuntimeError):
    """A round-260 continuation predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F")[2:-2, 2:-2].T


def _read_tracer(path: Path) -> dict[str, np.ndarray]:
    """Read the admitted self-identifying stage-2 tracer stream."""

    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, np.float64)
    expected = (1, 1, 2, 1, 3, 2, 2, NX, NY, NZ, 64)
    require(magic == "NEMO_L2_RKTRA_1", f"bad tracer magic {magic!r}")
    require(header == expected, f"bad tracer header {header}")
    n2, n3 = NX * NY, NX * NY * NZ
    require(values.size == 15 * n3 + 3 * n2, "bad tracer payload")
    names = (
        "zero_T", "zero_S", "zFu", "zFv", "zFw",
        "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S", "Kbb_T", "Kbb_S",
        "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
    )
    fields = {
        name: _xyz(values[index * n3:(index + 1) * n3])
        for index, name in enumerate(names)
    }
    offset = len(names) * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        fields[name] = _xy(values[offset + index * n2:offset + (index + 1) * n2])
    require(all(np.isfinite(field).all() for field in fields.values()),
            "non-finite tracer payload")
    return fields


def _score(left, right) -> dict[str, object]:
    return r249._score(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _fold(values) -> np.ndarray:
    return np.asarray(values, np.float64)[-3:]


def measure(deck_root: Path, frame_root: Path, record_root: Path,
            prior_report: Path, label: str, expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = r249._policy_stamp(expect_commit)
    prior = json.loads(prior_report.read_text(encoding="utf-8"))
    require(prior["label"] == label and prior["status"] == "PASS_R260_MEASUREMENT",
            "prior labelled measurement moved")
    require(prior["baseline_stage2"]["S"]["fold_band"]["max_abs"]
            == 6.817836451535307, "prior stage-2 boundary moved")
    require(prior["associated_stage2_entry"]["v"]["pivot_row"]["max_abs"]
            <= float(r249.FLOOR), "prior associated entry is not at the floor")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    stage1 = rung0.assemble_frame(frame_root, 1, 1)
    stage2 = rung0.assemble_frame(frame_root, 1, 2)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    forcing = omt0.rung0_ladder._zero_forcing((148, 180))
    tracer_path = record_root / "oracle_rktracer_operands_kt00000001_s2.bin"
    tracer = _read_tracer(tracer_path)
    recorded = (
        np.asarray(tracer["zFu"][..., :30], np.float64),
        np.asarray(tracer["zFv"][..., :30], np.float64),
        np.asarray(tracer["zFw"], np.float64),
    )
    hooks = r249._hooks(card, True, live=True)._replace(
        stage_entry_override=(
            2, stage1["u"], stage1["v"], stage1["T"], stage1["S"],
            stage1["ssh"]),
        stage2_tracer_transport_override=recorded,
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    with r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True):
        traced = jax.device_get(model.step(
            state, card.dt_s, freshwater=forcing[0], surface_forcing=forcing[1]))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "continuation did not return the passive live trace")

    geometry = traced.stage_geometry[1]
    candidate = (
        np.asarray(geometry[7][:, 1:], np.float64),
        np.asarray(geometry[8][1:], np.float64),
        np.asarray(geometry[2], np.float64)
        * np.asarray(card.recipe.grid.area_T, np.float64)[..., None],
    )
    transport_rows = {
        name: _score(_fold(value), _fold(expected))
        for name, value, expected in zip(("zFu", "zFv", "zFw"), candidate, recorded)
    }
    transport_replay = {
        "candidate_zFv_vs_geometry": _score(candidate[1], geometry[8][1:]),
        "oracle_zFv_self": _score(recorded[1], tracer["zFv"][..., :30]),
    }
    arm_stage2 = {
        name: _score(_fold(traced.stage_outputs[1][index]), _fold(stage2[name]))
        for index, name in ((2, "T"), (3, "S"))
    }
    entry_rows = {
        name: _score(_fold(value), _fold(stage1[name]))
        for name, value in zip(("u", "v", "T", "S"), traced.stage_states[1][:4])
    }
    one = np.asarray([1.0], np.float64)
    return {
        "format": "nemo-testcase-l4-orca2-round260-stage2-transport-v1",
        "status": "PASS_R260_TRANSPORT_SPLIT_MEASUREMENT",
        "label": label,
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "entry_rows": entry_rows,
        "transport_rows": transport_rows,
        "transport_replay": transport_replay,
        "recorded_transport_stage2": arm_stage2,
        "prior_stage2": {
            name: prior["arm_stage2"][name]["fold_band"] for name in ("T", "S")
        },
        "one_bit_control": _score(one, np.nextafter(one, np.inf)),
        "record": {"path": str(tracer_path), "sha256": _sha256(tracer_path)},
        "compiled_source": {
            "transport": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-285",
            "tracer_dispatch": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:628-700",
        },
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both labels are required")
    independent = by_label["independent"]
    if plant == "provenance":
        independent["operand_route"] = "retired"
    elif plant == "entry":
        independent["entry_rows"]["v"]["max_abs"] = 1.0
    elif plant == "transport-record":
        independent["record"]["sha256"] = "0" * 64
    elif plant == "transport-replay":
        independent["transport_replay"]["oracle_zFv_self"]["unequal"] = 1
    elif plant == "causal-closure":
        independent["recorded_transport_stage2"]["S"]["max_abs"] = 1.0
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    labels = {}
    for label in LABELS:
        report = by_label[label]
        require(report["status"] == "PASS_R260_TRANSPORT_SPLIT_MEASUREMENT",
                f"{label}: status moved")
        require(report["execution"] == "production-jit-cpu-fp64-x64-libm"
                and report["operand_route"] ==
                "production-d114-resolved-complete-private-unit",
                f"{label}: provenance moved")
        require(all(row["max_abs"] <= float(r249.FLOOR)
                    for row in report["entry_rows"].values()),
                f"{label}: exact stage-2 entry moved")
        require(report["record"]["sha256"] ==
                by_label[LABELS[0]]["record"]["sha256"],
                f"{label}: record digest moved")
        require(report["transport_replay"]["oracle_zFv_self"]["unequal"] == 0,
                f"{label}: record replay is not exact")
        require(report["prior_stage2"]["S"]["max_abs"]
                == 1.3842782919937235e-08,
                f"{label}: residual stage-2 boundary moved")
        require(all(report["recorded_transport_stage2"][name]["max_abs"]
                    <= float(r249.FLOOR) for name in ("T", "S")),
                f"{label}: NEMO transport did not close stage 2")
        require(report["one_bit_control"]["unequal"] == 1,
                f"{label}: one-bit control did not fire")
        labels[label] = {
            "transport_rows": report["transport_rows"],
            "recorded_transport_stage2": report["recorded_transport_stage2"],
        }
    first = next((name for name in ("zFu", "zFv", "zFw")
                  if any(labels[label]["transport_rows"][name]["max_abs"]
                         > float(r249.FLOOR) for label in LABELS)), None)
    require(first is not None, "no transport boundary owns the residual")
    return {
        "format": "nemo-testcase-l4-orca2-round260-stage2-transport-classification-v1",
        "status": "PASS_R260_STAGE2_TRANSPORT_OWNER",
        "first_non_floor_transport": first,
        "labels": labels,
        "predictions": {
            "R260-P1": "CONFIRMED",
            "R260-P2": "CONFIRMED",
            "R260-P3": "CONFIRMED",
            "R260-P4": "REFUTED_POST_ASSOCIATION_RESIDUAL; TRANSPORT_OWNER_CONFIRMED",
            "R260-P5": "NOT_REACHED",
            "R260-P6": "NOT_REACHED",
            "R260-P7": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--prior-report", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify", nargs=2, type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.measure:
            require(all((args.deck_root, args.frame_root, args.record_root,
                         args.prior_report, args.label, args.expect_commit)),
                    "measurement arguments are incomplete")
            result = measure(args.deck_root, args.frame_root, args.record_root,
                             args.prior_report, args.label, args.expect_commit)
        else:
            require(args.classify is not None, "classification reports are required")
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.classify], plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, r249.GateError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

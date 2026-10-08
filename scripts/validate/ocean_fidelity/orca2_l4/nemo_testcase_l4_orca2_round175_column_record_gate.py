#!/usr/bin/env python3
"""Gate the round-175 target-column geometry and operator-record coverage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_phase2l_tracer_gate as tracer_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round11_dynldf_operator_gate as ldf_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round84_rung0_frame_gate as frame_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
)


TARGET = (86, 159, 3)
FIELD_ORDER = (
    "entry",
    "external_mode_qco",
    "momentum_rhs",
    "momentum_pre_correction",
    "corrected_velocity",
    "metric_transport",
    "tracer_after_advection",
    "tracer_after_sbc",
    "tracer_qco_update",
    "completed_stage",
)
PLANTS = (
    "none", "target", "neighbour", "placement", "order", "selection",
    "row-ulp",
)
ROUND174_SHA256 = "e918fda0933efb764711e388237d2c0a5bc8af322bbdfa7b17c6b68d89aad8ce"


class GateError(RuntimeError):
    """A frozen geometry, record, or source-order predicate moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bits_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.ascontiguousarray(left, dtype=np.float64)
    right = np.ascontiguousarray(right, dtype=np.float64)
    return left.shape == right.shape and bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def _geometry(deck_root: Path, record_root: Path) -> dict[str, object]:
    mesh = ldf_gate._stitch(
        record_root,
        "mesh_mask_{rank:04d}.nc",
        ("mbathy", "e3t_0", "tmask", "umask", "vmask"),
    )
    import netCDF4  # noqa: N813

    domain_path = deck_root / "ORCA_R2_zps_domcfg.nc"
    require(domain_path.is_file(), f"missing {domain_path}")
    with netCDF4.Dataset(domain_path, "r") as dataset:
        dataset.set_auto_maskandscale(False)
        require("e3w_0" in dataset.variables
                and "bottom_level" in dataset.variables,
                "domain_cfg lacks e3w_0 or bottom_level")
        domain_e3w = np.moveaxis(
            np.asarray(dataset.variables["e3w_0"][:], dtype=np.float64), 0, -1
        )[..., :30]
        domain_bottom = np.asarray(
            dataset.variables["bottom_level"][:], dtype=np.float64)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    z_coord = card.recipe.z_coord
    j, i, k = TARGET

    for name, value in {**mesh, "domain_e3w": domain_e3w,
                        "domain_bottom": domain_bottom}.items():
        require(np.asarray(value).dtype == np.float64,
                f"NEMO geometry {name} is not float64")
    require(mesh["e3t_0"].shape == domain_e3w.shape == (148, 180, 30),
            "NEMO thickness shape moved")
    require(mesh["tmask"].shape == mesh["umask"].shape == mesh["vmask"].shape
            == (148, 180, 30), "NEMO mask shape moved")

    lego_e3t = np.asarray(z_coord.nemo_e3t_0, dtype=np.float64)
    lego_e3w = np.asarray(z_coord.nemo_e3w_0, dtype=np.float64)
    lego_t = np.asarray(z_coord.is_active, dtype=np.float64)
    lego_u = np.asarray(card.recipe.initial_state.u_mask.data, dtype=np.float64)[:, 1:181]
    lego_v = np.asarray(card.recipe.initial_state.v_mask.data, dtype=np.float64)[1:149, :]
    require(lego_e3t.shape == lego_e3w.shape == lego_t.shape == (148, 180, 30),
            "legoESM target geometry shape moved")
    require(lego_u.shape == lego_v.shape == (148, 180),
            "legoESM native face-mask shape moved")

    neighbours = {
        "west": float(mesh["tmask"][j, i - 1, 0]),
        "east": float(mesh["tmask"][j, i + 1, 0]),
        "south": float(mesh["tmask"][j - 1, i, 0]),
        "north": float(mesh["tmask"][j + 1, i, 0]),
    }
    require(_bits_equal(mesh["mbathy"], domain_bottom),
            "mesh_mask mbathy differs from domain_cfg bottom_level")
    bottom = int(mesh["mbathy"][j, i]) - 1
    return {
        "target_jik": [j, i, k],
        "array_shapes": {
            "nemo_e3t_0": list(mesh["e3t_0"].shape),
            "nemo_e3w_0": list(domain_e3w.shape),
            "nemo_tmask": list(mesh["tmask"].shape),
            "legoesm_e3t_0": list(lego_e3t.shape),
            "legoesm_e3w_0": list(lego_e3w.shape),
        },
        "dtypes": {
            "nemo": str(mesh["e3t_0"].dtype),
            "legoesm": str(lego_e3t.dtype),
        },
        "mbkt_fortran": int(mesh["mbathy"][j, i]),
        "mesh_mask_name": "mbathy",
        "domain_cfg_name": "bottom_level",
        "domain_cfg_sha256": sha256(domain_path),
        "bottom_zero_based": bottom,
        "bottom_is_partial": bool(
            0 < bottom < mesh["e3t_0"].shape[-1]
            and mesh["e3t_0"][j, i, bottom]
            != mesh["e3t_0"][j, i, bottom - 1]
        ),
        "levels_0_5": {
            "nemo_e3t_0_m": mesh["e3t_0"][j, i, :6].tolist(),
            "nemo_e3w_0_m": domain_e3w[j, i, :6].tolist(),
            "nemo_tmask": mesh["tmask"][j, i, :6].tolist(),
            "nemo_umask": mesh["umask"][j, i, :6].tolist(),
            "nemo_vmask": mesh["vmask"][j, i, :6].tolist(),
            "legoesm_e3t_0_m": lego_e3t[j, i, :6].tolist(),
            "legoesm_e3w_0_m": lego_e3w[j, i, :6].tolist(),
            "legoesm_tmask": lego_t[j, i, :6].tolist(),
        },
        "neighbour_surface_tmask": neighbours,
        "land_adjacent": any(value == 0.0 for value in neighbours.values()),
        "fold_row": j == 147,
        "cyclic_seam": i in (0, 179),
        "card_matches_nemo": {
            "e3t_0": _bits_equal(lego_e3t, mesh["e3t_0"]),
            "e3w_0": _bits_equal(lego_e3w, domain_e3w),
            "tmask": _bits_equal(lego_t, mesh["tmask"]),
            "surface_umask": _bits_equal(lego_u, mesh["umask"][..., 0]),
            "surface_vmask": _bits_equal(lego_v, mesh["vmask"][..., 0]),
        },
    }


def _record_census(record_root: Path) -> dict[str, object]:
    path = record_root / tracer_gate.TRACER_RECORD
    require(path.is_file(), f"missing {path}")
    tracer = tracer_gate.read_tracer(path)
    header = tracer["header"]
    rank_files = sorted(record_root.glob("oracle_rktracer_operands_rank*_kt00000001_s1.bin"))
    target_i = TARGET[1]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "header": header,
        "owned_shape_after_halo_strip": [tracer_gate.OWNED_NY, tracer_gate.OWNED_NX, tracer_gate.NLEV],
        "unranked_files": 1,
        "rank_tagged_files": len(rank_files),
        "covered_global_i": [0, tracer_gate.OWNED_NX - 1],
        "target_i": target_i,
        "target_present": bool(target_i < tracer_gate.OWNED_NX),
        "rank_complete": len(rank_files) == 2,
    }


def _operator_table(census: dict[str, object], round174: dict[str, object]) -> list[dict[str, object]]:
    final_row = round174["boundaries"][0]["rows"]["S"]
    rows = [{
        "boundary": "entry", "status": "AT_BAR_BIT_EXACT",
        "target_available": True, "max_abs": 0.0,
    }]
    for boundary in FIELD_ORDER[1:-1]:
        rows.append({
            "boundary": boundary,
            "status": "UNMEASURED_MISSING_RANK1_STREAM",
            "target_available": bool(census["target_present"]),
            "max_abs": None,
        })
    rows.append({
        "boundary": "completed_stage", "status": "DEBT",
        "target_available": True, "max_abs": final_row["max_abs"],
        "argmax": final_row["argmax"],
    })
    return rows


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = copy.deepcopy(report)
    if plant == "target":
        report["geometry"]["target_jik"][1] -= 1
    elif plant == "neighbour":
        report["geometry"]["land_adjacent"] = not report["geometry"]["land_adjacent"]
    elif plant == "placement":
        report["record_census"]["target_present"] = True
    elif plant == "order":
        report["operator_table"][1], report["operator_table"][2] = (
            report["operator_table"][2], report["operator_table"][1])
    elif plant == "selection":
        report["first_unmeasured"] = "momentum_rhs"
    elif plant == "row-ulp":
        report["operator_table"][-1]["max_abs"] = float(np.nextafter(
            report["operator_table"][-1]["max_abs"], np.inf))

    geometry = report["geometry"]
    require(tuple(geometry["target_jik"]) == TARGET, "target cell moved")
    require(geometry["dtypes"] == {"nemo": "float64", "legoesm": "float64"},
            "geometry dtype moved")
    require(all(geometry["card_matches_nemo"].values()),
            "card geometry no longer matches NEMO")
    derived_land_adjacent = any(
        value == 0.0 for value in geometry["neighbour_surface_tmask"].values())
    require(geometry["land_adjacent"] == derived_land_adjacent,
            "neighbour-mask classification moved")

    census = report["record_census"]
    require(census["rank_tagged_files"] == 0 and not census["rank_complete"],
            "legacy tracer stream unexpectedly became rank-complete")
    require(census["covered_global_i"] == [0, 89]
            and census["target_i"] == 159 and not census["target_present"],
            "legacy tracer stream placement moved")
    require([row["boundary"] for row in report["operator_table"]] == list(FIELD_ORDER),
            "operator table order moved")
    first = next((row["boundary"] for row in report["operator_table"]
                  if row["status"].startswith("UNMEASURED")), None)
    require(first == "external_mode_qco" and report["first_unmeasured"] == first,
            "first unavailable boundary selection moved")
    require(report["operator_table"][0]["status"] == "AT_BAR_BIT_EXACT",
            "entry row moved")
    final = report["operator_table"][-1]
    require(final["status"] == "DEBT"
            and final["max_abs"] == 3.2847473521544472
            and final["argmax"] == [86, 159, 3],
            "round-174 completed-stage salinity row moved")
    report["predictions"] = {
        "R175-P1": ("CONFIRMED" if geometry["mbkt_fortran"] == 4
                     and geometry["bottom_zero_based"] == 3
                     and geometry["bottom_is_partial"]
                     and geometry["levels_0_5"]["nemo_tmask"][:4]
                     == [1.0, 1.0, 1.0, 1.0]
                     and geometry["levels_0_5"]["nemo_tmask"][4] == 0.0
                     and geometry["land_adjacent"]
                     and not geometry["fold_row"]
                     and not geometry["cyclic_seam"] else "REFUTED"),
        "R175-P2": "CONFIRMED",
        "R175-P3": "CONFIRMED",
        "R175-P4": "CONFIRMED",
        "R175-P5": "CONFIRMED",
    }
    report["status"] = "STOP_RECORD_R175_STAGE1_RANK_COMPLETE_NEEDED"
    return report


def measure(deck_root: Path, record_root: Path, round174_path: Path,
            expect_commit: str) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "round-175 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")
    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-175 measurement requires its clean committed gate")
    require(round174_path.is_file() and sha256(round174_path) == ROUND174_SHA256,
            "round-174 stage-growth report moved")
    round174 = json.loads(round174_path.read_text())
    admission = frame_gate.admit(record_root, rung0_ladder.EXPECTED_PRODUCER, None)
    require(admission["record_count"] == 80, "round-90 frames no longer admit")
    census = _record_census(record_root)
    report = {
        "format": "nemo-testcase-l4-orca2-round175-column-record-v1",
        "claim_label": "independent",
        "execution": {
            "backend": jax.default_backend(),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "precision_policy": str(get_policy()),
            "instrument": "record-census-no-executable-observer",
        },
        "worktree": stamp,
        "geometry": _geometry(deck_root, record_root),
        "record_census": census,
        "operator_order": list(FIELD_ORDER),
        "operator_table": _operator_table(census, round174),
        "first_unmeasured": "external_mode_qco",
        "round174": {"path": str(round174_path), "sha256": sha256(round174_path)},
        "frame_admission": {"status": admission["status"], "record_count": 80},
        "compiled_citations": {
            "external_and_transport": "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:127-365",
            "tracer_and_update": "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:600-798",
        },
    }
    return classify(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--round174", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    args = parser.parse_args()
    try:
        if args.report_in:
            require(not any((args.deck_root, args.record_root, args.round174,
                             args.expect_commit)),
                    "classification cannot take runtime inputs")
            result = classify(json.loads(args.report_in.read_text()), args.plant)
        else:
            require(args.plant == "none", "runtime mode does not accept plants")
            require(all((args.deck_root, args.record_root, args.round174,
                         args.expect_commit)), "runtime inputs are incomplete")
            result = measure(args.deck_root, args.record_root, args.round174,
                             args.expect_commit)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, frame_gate.GateError, ldf_gate.GateError,
            tracer_gate.GateError, rung0.GateError, OSError, KeyError,
            TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS STOP_RECORD_R175_STAGE1_RANK_COMPLETE_NEEDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

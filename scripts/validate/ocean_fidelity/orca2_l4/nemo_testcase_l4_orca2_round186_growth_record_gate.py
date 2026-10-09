#!/usr/bin/env python3
"""Render and admit the ORCA2 rung-0 step-10..95 growth restarts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round71_independent_month_gate as month,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round82_rung0_deck_gate as deck_gate,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


STEPS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
ITEND = 96
FIELDS = ("tn", "sn", "un", "vn", "sshn")
PLANTS = ("none", "missing-rank", "twin-ulp", "step10-calibration", "hidden-deck")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal(value: str) -> str:
    return deck_gate._normalise(value)


def _integer(value: str, label: str) -> int:
    match = re.match(r"\s*(-?\d+)", value)
    require(match is not None, f"{label}: invalid integer {value!r}")
    return int(match.group(1))


def render_deck(source: str) -> str:
    """Change run protocol only and request the exact frozen restart steps."""

    for key, value in (("nn_itend", ITEND), ("nn_stock", ITEND)):
        source, count = re.subn(
            rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>{value}", source,
            count=1, flags=re.MULTILINE)
        require(count == 1, f"{key} not found exactly once")
    lines = (
        "   ln_rst_list = .true.\n"
        "   nn_stocklist = " + ", ".join(str(step) for step in STEPS) + "\n"
    )
    source, count = re.subn(
        r"^(\s*nn_stock\s*=\s*\S+[^\n]*\n)",
        lambda match: match.group(1) + lines,
        source, count=1, flags=re.MULTILINE)
    require(count == 1, "restart-list insertion point not found exactly once")
    return source


def validate_deck(source: Path, candidate: Path, plant: str = "none") -> dict:
    base = namelist_values(source)
    values = namelist_values(candidate)
    if plant == "hidden-deck":
        key = "namtra_ldf.ln_traldf_lap"
        require(key in values, "hidden-deck plant key disappeared")
        values[key] = ".false." if _normal(values[key]) == ".true." else ".true."
    list_keys = {"namrun.ln_rst_list", "namrun.nn_stocklist"}
    require(set(values) == set(base) | list_keys,
            "growth deck assignment inventory changed")
    ignored = {"namrun.nn_itend", "namrun.nn_stock"}
    changed = sorted(
        key for key in base if key not in ignored
        and _normal(values[key]) != _normal(base[key]))
    require(not changed, f"hidden growth-deck delta: {changed}")
    require(_integer(values["namrun.nn_itend"], "nn_itend") == ITEND,
            "growth nn_itend moved")
    require(_integer(values["namrun.nn_stock"], "nn_stock") == ITEND,
            "growth nn_stock moved")
    logical = values["namrun.ln_rst_list"].strip().replace(".", "").upper()
    require(logical in {"T", "TRUE"}, "growth restart-list mode is off")
    observed = tuple(int(token.strip())
                     for token in values["namrun.nn_stocklist"].split(","))
    require(observed == STEPS, f"growth restart steps moved: {observed}")
    return {
        "source": str(source), "candidate": str(candidate),
        "steps": list(observed), "itend": ITEND,
        "physical_delta": changed, "status": "RUN_PROTOCOL_ONLY",
    }


def _read(path: Path, step: int) -> tuple[dict[str, np.ndarray], dict]:
    require(path.is_file(), f"missing restart: {path}")
    with Dataset(path) as dataset:
        require(float(np.asarray(dataset["kt"][:])) == float(step),
                f"{path.name}: kt is not {step}")
        arrays = {
            "tn": month._restart_xyz(dataset["tn"], "tn"),
            "sn": month._restart_xyz(dataset["sn"], "sn"),
            "un": month._restart_xyz(dataset["un"], "un"),
            "vn": month._restart_xyz(dataset["vn"], "vn"),
            "sshn": month._restart_xy(dataset["sshn"], "sshn"),
        }
        header = {
            name: {
                "dtype": str(dataset[name].dtype),
                "dimensions": list(dataset[name].dimensions),
                "shape": list(dataset[name].shape),
                "payload_values": int(dataset[name].size),
            }
            for name in FIELDS
        }
    require(all(np.isfinite(value).all() for value in arrays.values()),
            f"{path.name}: restart payload is non-finite")
    return arrays, {"path": str(path), "sha256": sha256(path), "fields": header}


def admit(
    source: Path,
    deck_a: Path,
    deck_b: Path,
    twin_a: Path,
    twin_b: Path,
    calibration: Path,
    plant: str = "none",
) -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    deck_a_row = validate_deck(source, deck_a, plant)
    deck_b_row = validate_deck(source, deck_b)
    require(deck_a.read_bytes() == deck_b.read_bytes(), "twin decks differ")
    comparisons = []
    headers = []
    for step in STEPS:
        for rank in (0, 1):
            path_a = twin_a / f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            path_b = twin_b / f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            if plant == "missing-rank" and step == STEPS[-1] and rank == 1:
                path_b = twin_b / "PLANTED_MISSING_RANK.nc"
            arrays_a, header_a = _read(path_a, step)
            arrays_b, header_b = _read(path_b, step)
            if plant == "twin-ulp" and step == STEPS[0] and rank == 0:
                arrays_b = copy.deepcopy(arrays_b)
                arrays_b["tn"].flat[0] = np.nextafter(
                    arrays_b["tn"].flat[0], np.float64(np.inf))
            unequal = {name: int(np.count_nonzero(
                np.ascontiguousarray(arrays_a[name]).view(np.uint64)
                != np.ascontiguousarray(arrays_b[name]).view(np.uint64)))
                for name in FIELDS}
            require(not any(unequal.values()),
                    f"step {step} rank {rank}: twin payload moved: {unequal}")
            if step == 10:
                control_path = calibration / path_a.name
                control, _ = _read(control_path, step)
                if plant == "step10-calibration" and rank == 0:
                    control = copy.deepcopy(control)
                    control["sn"].flat[0] = np.nextafter(
                        control["sn"].flat[0], np.float64(np.inf))
                control_unequal = {name: int(np.count_nonzero(
                    np.ascontiguousarray(arrays_a[name]).view(np.uint64)
                    != np.ascontiguousarray(control[name]).view(np.uint64)))
                    for name in FIELDS}
                require(not any(control_unequal.values()),
                        f"step-10 calibration moved rank {rank}: {control_unequal}")
            comparisons.append({"step": step, "rank": rank, "unequal": unequal})
            headers.extend((header_a, header_b))
    require(len(comparisons) == len(STEPS) * 2,
            "growth restart comparison census moved")
    return {
        "format": "nemo-testcase-l4-orca2-round186-growth-record-v1",
        "status": "PASS_R186_GROWTH_RECORD",
        "claim_label": "independent",
        "restart_steps": list(STEPS),
        "rank_count": 2,
        "field_order": list(FIELDS),
        "deck_a": deck_a_row,
        "deck_b": deck_b_row,
        "comparisons": comparisons,
        "header_count": len(headers),
        "calibration": str(calibration),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-source", type=Path)
    parser.add_argument("--render-output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--deck-a", type=Path)
    parser.add_argument("--deck-b", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.render_source or args.render_output:
            require(args.render_source and args.render_output,
                    "render mode needs source and output")
            args.render_output.write_text(render_deck(args.render_source.read_text()))
            report = validate_deck(args.render_source, args.render_output)
            print(json.dumps(report, indent=2, sort_keys=True))
            print("STATUS RENDERED_R186_GROWTH_DECK")
            return 0
        require(all((args.source, args.deck_a, args.deck_b, args.twin_a,
                     args.twin_b, args.calibration)),
                "admission mode needs every record input")
        report = admit(
            args.source, args.deck_a, args.deck_b, args.twin_a, args.twin_b,
            args.calibration, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, deck_gate.GateError, OSError, KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_R186_GROWTH_RECORD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

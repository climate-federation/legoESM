#!/usr/bin/env python3
"""Render and admit Decision-103 ORCA2 mini-ladder rung OMT-0."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round82_rung0_deck_gate as rung0_deck,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round82_rung0_record_gate as rung0_record,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


SOURCE_SHA256 = "b627f4e2d94e4619dbfa27f39b811732497f032b73be86adcc57ddca2dee0e91"
BINARY_SHA256 = rung0_record.BINARY_SHA256
CPP_SHA256 = rung0_deck.CPP_SHA256
FIELDS = ("sshn", "un", "vn", "tn", "sn")
TWIN_STEPS = tuple(range(1, 11))
TWIN_ITEND = 12
MONTH_STEPS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
MONTH_ITEND = 96
RESTART_LIST_CAPACITY = 10

CHANGED = {
    "namdyn_adv.ln_dynadv_vec": ".false.",
    "namdyn_ldf.ln_dynldf_lap": ".false.",
    "namtra_adv.ln_traadv_fct": ".false.",
    "namtra_ldf.ln_traldf_lap": ".false.",
    "namdrg.ln_lin": ".false.",
}
ADDED = {
    "namdyn_adv.ln_dynadv_off": ".true.",
    "namdyn_ldf.ln_dynldf_off": ".true.",
    "namtra_adv.ln_traadv_off": ".true.",
    "namtra_ldf.ln_traldf_off": ".true.",
    "namdrg.ln_drg_off": ".true.",
}
RETAINED = {
    "namdyn_vor.ln_dynvor_een": ".true.",
    "namdyn_hpg.ln_hpg_sco": ".true.",
    "namdyn_spg.ln_dynspg_ts": ".true.",
    "namzdf.ln_zdfcst": ".true.",
    "namzdf.ln_zdfevd": ".true.",
    "namsbc.nn_ice": "0",
}
PLANTS = (
    "none", "extra-delta", "live-module", "oversized-list",
    "missing-smoke", "missing-rank", "wrong-step", "nonfinite",
    "twin-ulp", "calibration-ulp", "missing-month", "changed-binary",
)


class GateError(RuntimeError):
    """The OMT-0 deck or acquired record violated a frozen predicate."""


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
    return rung0_deck._normalise(value)


def render_omt0(source: str) -> str:
    """Apply only Decision 103's five OFF selections to hierarchy rung 0."""
    text = source
    for qualified, value in CHANGED.items():
        text = rung0_deck._replace_in_group(text, qualified, value)
    for qualified, value in ADDED.items():
        text = rung0_deck._add_to_group(text, qualified, value)
    return text


def _assignment_delta(before: dict[str, str], after: dict[str, str]) -> tuple[list[str], list[str]]:
    changed = sorted(
        key for key in before if key in after and _normal(before[key]) != _normal(after[key])
    )
    return changed, sorted(set(after) - set(before))


def validate_deck(source: Path, candidate: Path, cpp: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    require(sha256(source) == SOURCE_SHA256, "admitted rung-0 namelist changed")
    require(sha256(cpp) == CPP_SHA256, "ORCA2 CPP card changed")
    expected = render_omt0(source.read_text())
    actual = candidate.read_text()
    if plant == "extra-delta":
        actual = actual.replace("nn_fsbc     = 2", "nn_fsbc     = 3", 1)
    elif plant == "live-module":
        actual, count = re.subn(
            r"(?mi)^(\s*ln_dynadv_vec\s*=\s*)\.false\.",
            r"\g<1>.true.", actual, count=1,
        )
        require(count == 1, "live-module plant target disappeared")
    require(actual == expected, "candidate is not the exact OMT-0 rendering")

    before = namelist_values(source)
    after = namelist_values(candidate)
    changed, added = _assignment_delta(before, after)
    require(changed == sorted(CHANGED), f"changed assignment inventory differs: {changed}")
    require(added == sorted(ADDED), f"added assignment inventory differs: {added}")
    for key, wanted in {**CHANGED, **ADDED, **RETAINED}.items():
        require(key in after and _normal(after[key]) == wanted,
                f"{key}: resolved value {_normal(after.get(key, 'MISSING'))!r} != {wanted!r}")
    return {
        "format": "nemo-testcase-l4-orca2-round199-omt0-deck-v1",
        "status": "PASS_OMT0_DECK",
        "source_sha256": SOURCE_SHA256,
        "candidate_sha256": hashlib.sha256(expected.encode()).hexdigest(),
        "cpp_sha256": CPP_SHA256,
        "changed_assignments": changed,
        "added_assignments": added,
        "retained_assignments": RETAINED,
    }


def _integer(value: str, label: str) -> int:
    match = re.match(r"\s*(-?\d+)", value)
    require(match is not None, f"{label}: invalid integer {value!r}")
    return int(match.group(1))


def _logical(value: str, label: str) -> bool:
    token = value.strip().replace(".", "").upper()
    require(token in {"T", "TRUE", "F", "FALSE"},
            f"{label}: invalid logical {value!r}")
    return token in {"T", "TRUE"}


def _integer_list(value: str, label: str) -> tuple[int, ...]:
    tokens = tuple(token.strip() for token in value.split(","))
    require(tokens and all(re.fullmatch(r"-?\d+", token) for token in tokens),
            f"{label}: invalid integer list {value!r}")
    return tuple(int(token) for token in tokens)


def render_run_deck(text: str, *, itend: int, stock: int,
                    restart_steps: tuple[int, ...]) -> str:
    """Render run controls without changing OMT-0 physics."""
    require(itend > 0 and itend % 2 == 0, "nn_itend must be positive and divisible by nn_fsbc=2")
    require(0 < stock <= itend, "nn_stock must be positive and no later than nn_itend")
    require(0 < len(restart_steps) <= RESTART_LIST_CAPACITY,
            "restart list exceeds compiled capacity 10")
    require(len(set(restart_steps)) == len(restart_steps), "restart list contains duplicates")
    require(all(0 < step <= itend for step in restart_steps),
            "restart list contains an out-of-run step")
    for key, value in (("nn_itend", itend), ("nn_stock", stock)):
        text, count = re.subn(
            rf"^(\s*{key}\s*=\s*)(\S+)", rf"\g<1>{value}",
            text, count=1, flags=re.MULTILINE,
        )
        require(count == 1, f"{key} not found exactly once")
    lines = (
        "   ln_rst_list = .true.\n"
        "   nn_stocklist = " + ", ".join(str(step) for step in restart_steps) + "\n"
    )
    text, count = re.subn(
        r"^(\s*nn_stock\s*=\s*\S+[^\n]*\n)",
        lambda match: match.group(1) + lines,
        text, count=1, flags=re.MULTILINE,
    )
    require(count == 1, "restart-list insertion point not found exactly once")
    return text


def validate_run_deck(canonical: Path, root: Path, *, itend: int, stock: int,
                      restart_steps: tuple[int, ...], plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    deck_manifest = rung0_record._manifest(root, "deck_files.sha256")
    input_manifest = rung0_record._manifest(root, "input_files.sha256")
    require(f"{rung0_deck.ZERO_FILE}.nc" in input_manifest,
            "exact-zero flux file is absent from the input manifest")
    values = namelist_values(root / "namelist_cfg")
    base = namelist_values(canonical)
    expected_keys = set(base) | {"namrun.ln_rst_list", "namrun.nn_stocklist"}
    require(set(values) == expected_keys, "run deck assignment inventory changed")
    ignored = {"namrun.nn_itend", "namrun.nn_stock"}
    changed = sorted(
        key for key in base if key not in ignored and _normal(values[key]) != _normal(base[key])
    )
    require(not changed, f"hidden run-deck physical delta: {changed}")
    require(_integer(values["namrun.nn_itend"], "nn_itend") == itend,
            f"nn_itend is not {itend}")
    require(_integer(values["namrun.nn_stock"], "nn_stock") == stock,
            f"nn_stock is not {stock}")
    require(_logical(values["namrun.ln_rst_list"], "ln_rst_list"),
            "restart-list mode is off")
    actual = _integer_list(values["namrun.nn_stocklist"], "nn_stocklist")
    if plant == "oversized-list":
        actual = actual + (itend,) * (RESTART_LIST_CAPACITY + 1)
    require(len(actual) <= RESTART_LIST_CAPACITY,
            "compiled nn_stocklist capacity 10 exceeded")
    require(actual == restart_steps, f"restart list moved: {actual}")
    return {
        "itend": itend,
        "stock": stock,
        "restart_steps": list(actual),
        "physical_delta": changed,
        "deck_files": len(deck_manifest),
        "input_files": len(input_manifest),
        "zero_flux": rung0_record._validate_zero_flux(root),
    }


def _payload(path: Path, wanted_step: int, *, nonfinite: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    require(path.is_file(), f"missing restart shard: {path}")
    arrays: dict[str, np.ndarray] = {}
    header = {"path": str(path), "sha256": sha256(path), "fields": {}}
    with Dataset(path) as dataset:
        require("kt" in dataset.variables, f"{path.name}: missing kt")
        dataset["kt"].set_auto_maskandscale(False)
        observed_step = int(np.asarray(dataset["kt"][:]))
        require(observed_step == wanted_step,
                f"{path.name}: kt is {observed_step}, expected {wanted_step}")
        for name in FIELDS:
            require(name in dataset.variables, f"{path.name}: missing {name}")
            variable = dataset[name]
            variable.set_auto_maskandscale(False)
            values = np.asarray(variable[:]).copy()
            require(variable.dtype == np.dtype("float64"),
                    f"{path.name}: {name} is not fp64")
            require(values.size > 0, f"{path.name}: {name} payload is empty")
            if nonfinite and not arrays:
                values.reshape(-1)[0] = np.nan
            require(bool(np.isfinite(values).all()),
                    f"{path.name}: {name} has non-finite values")
            arrays[name] = values
            header["fields"][name] = {
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
                "payload_values": int(variable.size),
                "dtype": str(variable.dtype),
            }
    return arrays, header


def _run_provenance(root: Path, *, itend: int, restart_steps: tuple[int, ...],
                    plant: str = "none") -> None:
    expected_binary = "0" * 64 if plant == "changed-binary" else BINARY_SHA256
    require(sha256(root / "nemo") == expected_binary, f"{root}: binary changed")
    stdout = (root / "run.user.stdout.log").read_text()
    timing = (root / "run.user.time.log").read_text()
    ocean = (root / "ocean.output").read_text()
    if plant == "missing-smoke" and itend == 2:
        timing = timing.replace("RUN_DONE", "")
    require("STOP 0" in stdout and "RUN_DONE" in timing, f"{root}: incomplete run")
    require(re.search(rf"number of the last time step\s+nn_itend\s+=\s+{itend}\b", ocean),
            f"{root}: resolved nn_itend moved")
    resolved = {
        "ln_dynadv_OFF": r"linear dynamics : no momentum advection\s+ln_dynadv_OFF\s+=\s+T\b",
        "ln_dynldf_OFF": r"no explicit diffusion\s+ln_dynldf_OFF\s+=\s+T\b",
        "ln_traadv_OFF": r"No advection on T & S\s+ln_traadv_OFF\s+=\s+T\b",
        "ln_traldf_OFF": r"no explicit diffusion\s+ln_traldf_OFF\s+=\s+T\b",
        "ln_drg_OFF": r"free-slip\s+: Cd = 0\s+ln_drg_OFF\s+=\s+T\b",
    }
    for label, pattern in resolved.items():
        require(re.search(pattern, ocean), f"{root}: resolved {label} is not true")
    for step in restart_steps:
        require(f"open ocean restart NetCDF file: ./ORCA2_{step:08d}_restart" in ocean,
                f"{root}: resolved log omitted restart step {step}")


def _compare(left: dict[str, np.ndarray], right: dict[str, np.ndarray], label: str,
             *, ulp_field: str | None = None) -> int:
    compared = 0
    for field in FIELDS:
        candidate = right[field]
        if field == ulp_field:
            candidate = candidate.copy()
            candidate.reshape(-1)[0] = np.nextafter(candidate.reshape(-1)[0], np.inf)
        require(left[field].shape == candidate.shape,
                f"{label}: {field} shape moved")
        require(np.array_equal(left[field], candidate),
                f"{label}: {field} differs")
        compared += 1
    return compared


def validate_record(canonical: Path, smoke: Path, twin_a: Path, twin_b: Path,
                    month: Path, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    runs = (
        (smoke, 2, 2, (1,)),
        (twin_a, TWIN_ITEND, TWIN_ITEND, TWIN_STEPS),
        (twin_b, TWIN_ITEND, TWIN_ITEND, TWIN_STEPS),
        (month, MONTH_ITEND, MONTH_ITEND, MONTH_STEPS),
    )
    deck_rows = []
    for root, itend, stock, steps in runs:
        _run_provenance(root, itend=itend, restart_steps=steps, plant=plant)
        deck_rows.append(validate_run_deck(
            canonical, root, itend=itend, stock=stock,
            restart_steps=steps, plant=plant,
        ))

    headers = []
    comparisons = 0
    for step in TWIN_STEPS:
        for rank in (0, 1):
            name = f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            right_name = name
            if plant == "missing-rank" and step == 1 and rank == 0:
                right_name = "PLANTED_MISSING_RANK.nc"
            wanted = step + (1 if plant == "wrong-step" and step == 1 and rank == 0 else 0)
            left, left_header = _payload(twin_a / name, wanted)
            right, right_header = _payload(
                twin_b / right_name, step,
                nonfinite=(plant == "nonfinite" and step == 1 and rank == 0),
            )
            headers.extend((left_header, right_header))
            comparisons += _compare(
                left, right, f"step {step} rank {rank} twins",
                ulp_field=(FIELDS[0] if plant == "twin-ulp" and step == 1 and rank == 0 else None),
            )

    for rank in (0, 1):
        smoke_payload, smoke_header = _payload(
            smoke / f"ORCA2_{2:08d}_restart_{rank:04d}.nc", 2,
        )
        headers.append(smoke_header)
        require(smoke_payload, "smoke sentinel payload disappeared")
        for root in (twin_a, twin_b):
            sentinel, sentinel_header = _payload(
                root / f"ORCA2_{TWIN_ITEND:08d}_restart_{rank:04d}.nc", TWIN_ITEND,
            )
            headers.append(sentinel_header)
            require(sentinel, "twin terminal sentinel payload disappeared")

    month_payloads: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for step in MONTH_STEPS:
        for rank in (0, 1):
            name = f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            path = month / name
            if plant == "missing-month" and step == 95 and rank == 1:
                path = month / "PLANTED_MISSING_MONTH.nc"
            arrays, header = _payload(path, step)
            month_payloads[(step, rank)] = arrays
            headers.append(header)
    for rank in (0, 1):
        sentinel, header = _payload(
            month / f"ORCA2_{MONTH_ITEND:08d}_restart_{rank:04d}.nc", MONTH_ITEND,
        )
        headers.append(header)
        require(sentinel, "month terminal sentinel payload disappeared")
        for label, root in (("twin-a", twin_a), ("twin-b", twin_b)):
            twin, twin_header = _payload(
                root / f"ORCA2_{10:08d}_restart_{rank:04d}.nc", 10,
            )
            headers.append(twin_header)
            comparisons += _compare(
                twin, month_payloads[(10, rank)], f"step-10 {label} calibration rank {rank}",
                ulp_field=(FIELDS[0] if plant == "calibration-ulp" and label == "twin-a" and rank == 0 else None),
            )
    return {
        "format": "nemo-testcase-l4-orca2-round199-omt0-record-v1",
        "status": "PASS_R199_OMT0_RECORD",
        "claim_label": "independent OMT-0",
        "binary_sha256": BINARY_SHA256,
        "twin_steps": list(TWIN_STEPS),
        "month_steps": list(MONTH_STEPS),
        "rank_count": 2,
        "field_order": list(FIELDS),
        "field_comparisons": comparisons,
        "header_count": len(headers),
        "decks": deck_rows,
    }


def preflight() -> dict:
    return {
        "format": "nemo-testcase-l4-orca2-round199-omt0-preflight-v1",
        "status": "PASS_R199_OMT0_PREFLIGHT",
        "twin_steps": list(TWIN_STEPS),
        "twin_itend": TWIN_ITEND,
        "month_steps": list(MONTH_STEPS),
        "month_itend": MONTH_ITEND,
        "restart_list_capacity": RESTART_LIST_CAPACITY,
        "fields": list(FIELDS),
        "plants": list(PLANTS[1:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--render-deck", type=Path)
    parser.add_argument("--render-run-deck", type=Path)
    parser.add_argument("--itend", type=int)
    parser.add_argument("--stock", type=int)
    parser.add_argument("--restart-steps")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--month", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.render_deck:
            require(args.source is not None and args.plant == "none",
                    "deck rendering requires source and no plant")
            require(sha256(args.source) == SOURCE_SHA256,
                    "admitted rung-0 namelist changed")
            args.render_deck.write_text(render_omt0(args.source.read_text()))
            print(f"STATUS RENDERED_OMT0_DECK {args.render_deck}")
            return 0
        if args.render_run_deck:
            require(args.candidate is not None and args.itend and args.stock
                    and args.restart_steps and args.plant == "none",
                    "run-deck rendering requires canonical deck and controls")
            steps = tuple(int(value) for value in args.restart_steps.split(","))
            args.render_run_deck.write_text(render_run_deck(
                args.candidate.read_text(), itend=args.itend,
                stock=args.stock, restart_steps=steps,
            ))
            print(f"STATUS RENDERED_OMT0_RUN_DECK {args.render_run_deck}")
            return 0
        if args.preflight_only:
            require(args.plant == "none", "preflight does not accept a plant")
            report = preflight()
        elif args.smoke:
            require(all((args.candidate, args.smoke, args.twin_a, args.twin_b, args.month)),
                    "record admission requires canonical deck and all four roots")
            report = validate_record(
                args.candidate, args.smoke, args.twin_a, args.twin_b,
                args.month, args.plant,
            )
        else:
            require(all((args.source, args.candidate, args.cpp)),
                    "deck validation requires source, candidate and CPP card")
            report = validate_deck(args.source, args.candidate, args.cpp, args.plant)
    except (GateError, rung0_deck.GateError, rung0_record.GateError,
            KeyError, OSError, TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

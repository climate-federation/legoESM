#!/usr/bin/env python3
"""Admit the two-rank ORCA2 hierarchy rung-0 ten-step twins and month."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round82_rung0_deck_gate as deck_gate,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


BINARY_SHA256 = "c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343"
FIELDS = ("sshn", "un", "vn", "tn", "sn")
PLANTS = (
    "none", "restart-ulp", "missing-rank", "wrong-step", "nonfinite",
    "hidden-month-delta",
)


class GateError(RuntimeError):
    """The acquired rung-0 record violated a frozen predicate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path, name: str) -> dict[str, str]:
    path = root / name
    require(path.is_file(), f"missing manifest {path}")
    rows = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]),
                f"malformed manifest row: {line!r}")
        filename = fields[1].lstrip("* ")
        require(filename == Path(filename).name and filename not in rows,
                f"unsafe or duplicate manifest name: {filename!r}")
        rows[filename] = fields[0]
    for filename, expected in rows.items():
        target = root / filename
        require(target.is_file(), f"manifest target missing: {target}")
        require(sha256(target) == expected, f"manifest digest mismatch: {target}")
    return rows


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
    tokens = [token.strip() for token in value.split(",")]
    require(tokens and all(re.fullmatch(r"-?\d+", token) for token in tokens),
            f"{label}: invalid integer list {value!r}")
    return tuple(int(token) for token in tokens)


def render_run_deck(
    text: str, steps: int, stock: int, restart_list: bool,
) -> str:
    require(steps > 0 and stock > 0, "run controls must be positive")
    for key, value in (("nn_itend", steps), ("nn_stock", stock)):
        text, count = re.subn(
            rf"^(\s*{key}\s*=\s*)(\S+)",
            rf"\g<1>{value}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        require(count == 1, f"{key} not found exactly once")
    if restart_list:
        lines = "   ln_rst_list = .true.\n   nn_stocklist = " + ", ".join(
            str(step) for step in range(1, steps + 1)
        ) + "\n"
        text, count = re.subn(
            r"^(\s*nn_stock\s*=\s*\S+[^\n]*\n)",
            lambda match: match.group(1) + lines,
            text,
            count=1,
            flags=re.MULTILINE,
        )
        require(count == 1, "nn_stock insertion point not found exactly once")
    return text


def _validate_run_controls(
    values: dict[str, str],
    canonical: dict[str, str],
    steps: int,
    stock: int,
    restart_steps: tuple[int, ...] | None,
) -> dict:
    list_keys = {"namrun.ln_rst_list", "namrun.nn_stocklist"}
    expected_keys = set(canonical) | (list_keys if restart_steps else set())
    require(set(values) == expected_keys, "run deck assignment inventory changed")
    ignored = {"namrun.nn_itend", "namrun.nn_stock"}
    changed = sorted(
        key for key in canonical if key not in ignored
        and deck_gate._normalise(values[key]) != deck_gate._normalise(canonical[key])
    )
    require(not changed, f"hidden run-deck delta: {changed}")
    require(_integer(values["namrun.nn_itend"], "nn_itend") == steps,
            f"nn_itend is not {steps}")
    require(_integer(values["namrun.nn_stock"], "nn_stock") == stock,
            f"nn_stock is not {stock}")
    if restart_steps:
        require(_logical(values["namrun.ln_rst_list"], "ln_rst_list"),
                "ten-step run does not select restart-list mode")
        actual = _integer_list(values["namrun.nn_stocklist"], "nn_stocklist")
        require(actual == restart_steps,
                f"restart list is {actual}, expected {restart_steps}")
        return {"restart_mode": "list", "restart_steps": list(actual)}
    return {"restart_mode": "periodic", "restart_steps": [steps]}


def _validate_zero_flux(root: Path) -> dict:
    path = root / f"{deck_gate.ZERO_FILE}.nc"
    require(path.is_file(), f"missing exact-zero flux file: {path}")
    fields = {}
    with Dataset(path) as dataset:
        require(len(dataset.dimensions["x"]) == 180 and len(dataset.dimensions["y"]) == 148,
                "zero-flux grid is not ORCA2 180x148")
        for name in ("utau", "vtau", "qtot", "qsr", "emp"):
            require(name in dataset.variables, f"zero-flux variable missing: {name}")
            variable = dataset[name]
            variable.set_auto_maskandscale(False)
            values = np.asarray(variable[:])
            require(variable.dtype == np.dtype("float64"), f"{name} is not fp64")
            require(np.array_equal(values, np.zeros_like(values)), f"{name} is not exactly zero")
            fields[name] = {"shape": list(values.shape), "dtype": str(variable.dtype)}
    return {"sha256": sha256(path), "fields": fields}


def _validate_run_deck(
    source: Path,
    cpp: Path,
    root: Path,
    steps: int,
    stock: int,
    restart_steps: tuple[int, ...] | None,
) -> dict:
    deck = _manifest(root, "deck_files.sha256")
    inputs = _manifest(root, "input_files.sha256")
    require(f"{deck_gate.ZERO_FILE}.nc" in inputs, "zero-flux file is absent from input manifest")
    values = namelist_values(root / "namelist_cfg")
    expected = namelist_values(source)
    # The canonical rendered rung is the 240-step deck.  The ten-step twins
    # may differ only in the two run-control assignments.
    with tempfile.TemporaryDirectory(prefix="orca2-r82-deck-") as temporary:
        canonical_path = Path(temporary) / "namelist_cfg"
        canonical_path.write_text(deck_gate.render_rung0(source.read_text()))
        canonical_values = namelist_values(canonical_path)
    controls = _validate_run_controls(
        values, canonical_values, steps, stock, restart_steps,
    )
    for key, wanted in deck_gate.EXPECTED.items():
        require(deck_gate._normalise(values[key]) == wanted,
                f"{key}: run deck departed from rung 0")
    require(sha256(cpp) == deck_gate.CPP_SHA256, "CPP card changed")
    require(expected["namsbc.nn_ice"].strip().startswith("2"),
            "source deck is not the admitted shipped deck")
    return {"steps": steps, "stock": stock, **controls,
            "deck_files": len(deck), "input_files": len(inputs),
            "zero_flux": _validate_zero_flux(root)}


def _payload(path: Path, wanted_step: int, *, make_nonfinite: bool = False) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing restart shard: {path}")
    result = {}
    with Dataset(path) as dataset:
        require("kt" in dataset.variables, f"{path.name}: missing kt")
        kt = dataset["kt"]
        kt.set_auto_maskandscale(False)
        require(int(np.asarray(kt[:])) == wanted_step, f"{path.name}: kt is not {wanted_step}")
        for name in FIELDS:
            require(name in dataset.variables, f"{path.name}: missing {name}")
            variable = dataset[name]
            variable.set_auto_maskandscale(False)
            values = np.asarray(variable[:]).copy()
            require(variable.dtype == np.dtype("float64"), f"{path.name}: {name} is not fp64")
            if make_nonfinite and not result:
                values.reshape(-1)[0] = np.nan
            require(bool(np.isfinite(values).all()), f"{path.name}: {name} has non-finite values")
            result[name] = values
    return result


def _run_provenance(
    root: Path, steps: int, restart_steps: tuple[int, ...] | None,
) -> None:
    require(sha256(root / "nemo") == BINARY_SHA256, f"{root}: binary changed")
    stdout = (root / "run.user.stdout.log").read_text()
    timing = (root / "run.user.time.log").read_text()
    ocean = (root / "ocean.output").read_text()
    require("STOP 0" in stdout and "RUN_DONE" in timing, f"{root}: incomplete run")
    require(re.search(rf"number of the last time step\s+nn_itend\s+=\s+{steps}\b", ocean),
            f"{root}: resolved nn_itend is not {steps}")
    require(re.search(r"ice management in the sbc.*nn_ice\s+=\s+0\b", ocean),
            f"{root}: resolved nn_ice is not zero")
    require(re.search(r"constant vertical mixing coefficient\s+ln_zdfcst\s+=\s+T\b", ocean),
            f"{root}: constant mixing is not selected")
    require(re.search(r"Turbulent Kinetic Energy closure.*ln_zdftke\s+=\s+F\b", ocean),
            f"{root}: TKE is still selected")
    if restart_steps:
        require("list of restart dump times" in ocean,
                f"{root}: resolved restart-list mode is absent")
        for step in restart_steps:
            require(f"open ocean restart NetCDF file: ./ORCA2_{step:08d}_restart" in ocean,
                    f"{root}: resolved log omitted restart step {step}")


def validate_record(source: Path, cpp: Path, twin_a: Path, twin_b: Path,
                    month: Path, expect_commit: str, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant}")
    ten_steps = tuple(range(1, 11))
    roots = (
        (twin_a, 10, 1, ten_steps),
        (twin_b, 10, 1, ten_steps),
        (month, 240, 240, None),
    )
    decks = []
    for root, steps, stock, restart_steps in roots:
        require((root / "producer_commit.txt").read_text().strip() == expect_commit,
                f"{root}: producer commit changed")
        _run_provenance(root, steps, restart_steps)
        decks.append(_validate_run_deck(
            source, cpp, root, steps, stock, restart_steps,
        ))
    if plant == "hidden-month-delta":
        decks[-1]["stock"] = 239
    require(decks[0]["stock"] == decks[1]["stock"] == 1 and decks[2]["stock"] == 240,
            "hidden month deck delta")

    compared = 0
    for step in range(1, 11):
        for rank in (0, 1):
            name = f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            right_name = name
            if plant == "missing-rank" and step == 1 and rank == 0:
                right_name = "ORCA2_00000001_restart_9999.nc"
            wanted = step + (1 if plant == "wrong-step" and step == 1 and rank == 0 else 0)
            left = _payload(twin_a / name, wanted)
            right = _payload(
                twin_b / right_name, step,
                make_nonfinite=(plant == "nonfinite" and step == 1 and rank == 0),
            )
            for field in FIELDS:
                right_values = right[field]
                if plant == "restart-ulp" and step == 1 and rank == 0 and field == FIELDS[0]:
                    right_values = right_values.copy()
                    right_values.reshape(-1)[0] = np.nextafter(right_values.reshape(-1)[0], np.inf)
                require(np.array_equal(left[field], right_values),
                        f"ten-step twins differ: step={step} rank={rank} field={field}")
                compared += 1
    for rank in (0, 1):
        _payload(month / f"ORCA2_00000240_restart_{rank:04d}.nc", 240)
    for root in (twin_a, twin_b, month):
        require(not list(root.glob("*_restart_ice_*.nc")), f"{root}: ice restart exists on no-ice rung")
    return {
        "format": "nemo-testcase-l4-orca2-round82-rung0-record-v1",
        "status": "PASS_RUNG0_RECORD",
        "claim_label": "independent",
        "binary_sha256": BINARY_SHA256,
        "ten_step_twin_field_comparisons": compared,
        "ten_step_restart_shards": 40,
        "month_step": 240,
        "decks": decks,
    }


def preflight() -> dict:
    return {
        "format": "nemo-testcase-l4-orca2-round82-rung0-record-preflight-v1",
        "status": "PASS_RUNG0_RECORD_PREFLIGHT",
        "ten_step_restart_names": [
            f"ORCA2_{step:08d}_restart_{rank:04d}.nc"
            for step in range(1, 11) for rank in (0, 1)
        ],
        "month_restart_names": [
            f"ORCA2_00000240_restart_{rank:04d}.nc" for rank in (0, 1)
        ],
        "ten_step_restart_mode": "explicit-list",
        "ten_step_restart_steps": list(range(1, 11)),
        "month_restart_mode": "periodic-terminal",
        "fields": list(FIELDS),
        "plants": list(PLANTS[1:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--render-run-deck", action="store_true")
    parser.add_argument("--run-deck-source", type=Path)
    parser.add_argument("--run-deck-output", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--stock", type=int)
    parser.add_argument("--restart-list", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--twin-a", type=Path)
    parser.add_argument("--twin-b", type=Path)
    parser.add_argument("--month", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.render_run_deck:
            require(not args.preflight_only and args.run_deck_source
                    and args.run_deck_output and args.steps and args.stock,
                    "run-deck rendering requires source, output, steps, and stock")
            source_text = args.run_deck_source.read_text()
            rendered = render_run_deck(
                source_text, args.steps, args.stock, args.restart_list,
            )
            args.run_deck_output.write_text(rendered)
            controls = _validate_run_controls(
                namelist_values(args.run_deck_output),
                namelist_values(args.run_deck_source),
                args.steps,
                args.stock,
                tuple(range(1, args.steps + 1)) if args.restart_list else None,
            )
            print(json.dumps(controls, sort_keys=True))
            print(f"STATUS RENDERED_RUNG0_RUN_DECK {args.run_deck_output}")
            return 0
        if args.preflight_only:
            require(args.plant == "none", "preflight does not accept plants")
            report = preflight()
        else:
            require(all((args.source, args.cpp, args.twin_a, args.twin_b,
                         args.month, args.expect_commit)), "record mode requires every input")
            report = validate_record(args.source, args.cpp, args.twin_a, args.twin_b,
                                     args.month, args.expect_commit, args.plant)
    except (GateError, deck_gate.GateError, KeyError, OSError, TypeError, ValueError) as error:
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

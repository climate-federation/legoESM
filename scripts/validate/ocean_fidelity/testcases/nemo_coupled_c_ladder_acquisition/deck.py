#!/usr/bin/env python3
"""Stage and verify the exact R339/R5 deck with one registered C overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAYS = json.loads((HERE / "namelist_overlays.json").read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    return {
        name: digest
        for digest, name in (line.split(maxsplit=1) for line in path.read_text().splitlines())
    }


def _replace(text: str, name: str, value: int) -> str:
    matches = list(re.finditer(rf"(^\s*{name}\s*=\s*)[^!,\n]+", text, re.I | re.M))
    if not matches:
        raise ValueError(f"STATUS PLANT-FIRED: {name} absent from exact deck")
    match = matches[-1]
    return text[: match.start()] + match.group(1) + str(value) + " " + text[match.end() :]


def stage(source: Path, destination: Path, rung: str) -> dict:
    if rung not in ("C2", "C3", "C4"):
        raise ValueError("unknown C rung")
    destination.mkdir(parents=True, exist_ok=False)
    for manifest_name in ("input_files.sha256", "deck_files.sha256"):
        for name, digest in _manifest(source / manifest_name).items():
            src = source / name
            if _sha(src) != digest:
                raise ValueError(f"STATUS PLANT-FIRED: source deck digest {name}")
            dst = destination / name
            if manifest_name == "input_files.sha256":
                dst.symlink_to(src.resolve())
            else:
                dst.write_bytes(src.read_bytes())
    overlay = OVERLAYS[rung]
    cfg = destination / "namelist_cfg"
    text = cfg.read_text()
    text = _replace(text, "nn_itend", overlay["nn_itend"])
    text = _replace(text, "nn_stock", overlay["nn_stock"])
    cfg.write_text(text)
    report = verify(destination, rung)
    (destination / "C_DECK_RECEIPT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _last_int(text: str, name: str) -> int:
    values = re.findall(rf"^\s*{name}\s*=\s*(-?\d+)", text, re.I | re.M)
    if not values:
        raise ValueError(f"STATUS PLANT-FIRED: unresolved {name}")
    return int(values[-1])


def verify(destination: Path, rung: str) -> dict:
    cfg = (destination / "namelist_cfg").read_text()
    ice = (destination / "namelist_ice_cfg").read_text()
    overlay = OVERLAYS[rung]
    wanted = {
        "nn_it000": 1,
        "nn_itend": overlay["nn_itend"],
        "nn_stock": overlay["nn_stock"],
        "nn_fsbc": 2,
        "nn_ice": 2,
    }
    got = {name: _last_int(cfg, name) for name in wanted}
    if got != wanted:
        raise ValueError(f"STATUS PLANT-FIRED: resolved C overlay {got} != {wanted}")
    if rung == "C4" and overlay["nn_stock"] * _last_int(cfg, "rn_Dt") != 86400:
        raise ValueError("STATUS PLANT-FIRED: C4 restart is not daily")
    if re.search(r"\bln_rst_list\s*=\s*\.true\.", cfg, re.I):
        raise ValueError("STATUS PLANT-FIRED: C4 frequency mode was replaced by a list")
    return {
        "status": "PASS_EXACT_R339_R5_DECK",
        "rung": rung,
        "resolved": got,
        "ice_sha256": hashlib.sha256(ice.encode()).hexdigest(),
        "mpi_ranks": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(OVERLAYS["source_deck"]))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--rung", choices=("C2", "C3", "C4"), required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.source, args.destination, args.rung), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

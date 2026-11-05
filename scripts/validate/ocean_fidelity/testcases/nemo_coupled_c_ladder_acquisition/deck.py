#!/usr/bin/env python3
"""Stage and verify the exact R339/R5 deck with one registered C overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from legoesm.core.nemo_namelist import overlay as merge_namelists
from legoesm.core.nemo_namelist import parse_namelist

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


def _fortran_scalar(value: bool | int | float) -> str:
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(f"unsupported scalar overlay value: {value!r}")


def _set_group_scalar(text: str, group: str, name: str, value: bool | int | float) -> str:
    """Set one scalar in one namelist group, creating the row/group if absent."""
    group_match = re.search(rf"(?im)^\s*&{re.escape(group)}\b[^\n]*$", text)
    rendered = _fortran_scalar(value)
    if group_match is None:
        return text.rstrip() + f"\n\n&{group}\n   {name} = {rendered}\n/\n"
    end_match = re.search(r"(?m)^\s*/\s*(?:!.*)?$", text[group_match.end() :])
    if end_match is None:
        raise ValueError(f"STATUS PLANT-FIRED: unterminated namelist group {group}")
    body_start = group_match.end()
    body_end = body_start + end_match.start()
    body = text[body_start:body_end]
    matches = list(
        re.finditer(rf"(?im)^(\s*{re.escape(name)}\s*=\s*)[^!,\n]+", body)
    )
    if matches:
        match = matches[-1]
        body = body[: match.start()] + match.group(1) + rendered + " " + body[match.end() :]
    else:
        body = body.rstrip() + f"\n   {name} = {rendered}\n"
    return text[:body_start] + body + text[body_end:]


def _ice_overlay_rows():
    for classification, groups in OVERLAYS["ice_overlay"].items():
        for group, rows in groups.items():
            for name, value in rows.items():
                yield classification, group, name, value


def _require_overlay_value(parsed: dict, group: str, name: str, expected, label: str) -> None:
    key = ("ice", group.lower(), name.lower())
    got = parsed.get(key)
    if got != [expected]:
        raise ValueError(
            f"STATUS PLANT-FIRED: {label} {group}.{name}={got!r} != {[expected]!r}"
        )


def _require_full_ice_match(sas: dict, candidate: dict, label: str) -> None:
    pending = {
        ("ice", group.lower(), name.lower()): value
        for group, rows in OVERLAYS["ice_retained_pending"].items()
        for name, value in rows.items()
    }
    for (_, group, name), expected in pending.items():
        _require_overlay_value(candidate, group, name, expected, f"{label} retained pending")
    differing = {key for key in set(sas) | set(candidate) if sas.get(key) != candidate.get(key)}
    if differing != set(pending):
        missing = sorted(set(pending) - differing)
        extra = sorted(differing - set(pending))
        raise ValueError(
            f"STATUS PLANT-FIRED: {label} differs from SAS ICE5 outside the nine "
            f"pending initial-file/restart rows; missing={missing!r}, extra={extra!r}"
        )


def _require_staged_scalar_match(sas: dict, candidate: dict) -> None:
    """Check every scalar resolved row; FLD_N members expand only after NEMO runs."""
    pending = {
        ("ice", group.lower(), name.lower()): value
        for group, rows in OVERLAYS["ice_retained_pending"].items()
        for name, value in rows.items()
        if "%" not in name
    }
    for (_, group, name), expected in pending.items():
        _require_overlay_value(
            candidate, group, name, expected, "staged ICE5 resolution retained pending"
        )
    mismatches = []
    for key, expected in sas.items():
        if "%" in key[2] or key in pending or len(expected) != 1:
            continue
        if candidate.get(key) != expected:
            mismatches.append((key, expected, candidate.get(key)))
    if mismatches:
        raise ValueError(
            "STATUS PLANT-FIRED: staged ICE5 resolution differs from SAS ICE5 "
            f"outside the pending restart row: {mismatches!r}"
        )


def _apply_ice_overlay(path: Path) -> None:
    text = path.read_text()
    for _, group, name, value in _ice_overlay_rows():
        text = _set_group_scalar(text, group, name, value)
    path.write_text(text)


def stage(
    source: Path,
    destination: Path,
    rung: str,
    *,
    sas_reference: Path | None = None,
    plant: str | None = None,
) -> dict:
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
    if plant != "drop-ice-overlay":
        _apply_ice_overlay(destination / "namelist_ice_cfg")
    report = verify(destination, rung, sas_reference=sas_reference)
    (destination / "C_DECK_RECEIPT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _last_int(text: str, name: str) -> int:
    values = re.findall(rf"^\s*{name}\s*=\s*(-?\d+)", text, re.I | re.M)
    if not values:
        raise ValueError(f"STATUS PLANT-FIRED: unresolved {name}")
    return int(values[-1])


def verify(
    destination: Path,
    rung: str,
    *,
    sas_reference: Path | None = None,
    require_resolved_output: bool = False,
) -> dict:
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
    sas_reference = sas_reference or Path(OVERLAYS["sas_ice_reference"])
    sas = parse_namelist(sas_reference, "ice")
    resolved = merge_namelists(
        parse_namelist(destination / "namelist_ice_ref", "ice"),
        parse_namelist(destination / "namelist_ice_cfg", "ice"),
    )
    for _, group, name, expected in _ice_overlay_rows():
        _require_overlay_value(sas, group, name, expected, "SAS ICE5 reference")
        _require_overlay_value(resolved, group, name, expected, "staged ICE5 resolution")
    _require_staged_scalar_match(sas, resolved)
    if require_resolved_output:
        output = destination / "output.namelist.ice"
        if not output.is_file():
            raise ValueError(f"STATUS PLANT-FIRED: missing resolved ice output {output}")
        output_values = parse_namelist(output, "ice")
        for _, group, name, expected in _ice_overlay_rows():
            _require_overlay_value(output_values, group, name, expected, "executed ICE5 output")
        _require_full_ice_match(sas, output_values, "executed ICE5 output")
    ice_rows = sum(1 for _ in _ice_overlay_rows())
    return {
        "status": "PASS_EXACT_R339_R5_DECK_ICE5",
        "rung": rung,
        "resolved": got,
        "ice_dimensions": [
            int(resolved[("ice", "nampar", name)][0])
            for name in ("jpl", "nlay_i", "nlay_s")
        ],
        "ice_overlay_rows": ice_rows,
        "ice_retained_pending_rows": sum(
            len(rows) for rows in OVERLAYS["ice_retained_pending"].values()
        ),
        "sas_ice_reference": str(sas_reference),
        "resolved_output_checked": require_resolved_output,
        "ice_sha256": hashlib.sha256(ice.encode()).hexdigest(),
        "mpi_ranks": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(OVERLAYS["source_deck"]))
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--rung", choices=("C2", "C3", "C4"), required=True)
    parser.add_argument("--sas-reference", type=Path)
    parser.add_argument("--plant", choices=("drop-ice-overlay",))
    parser.add_argument("--verify-resolved-output", action="store_true")
    args = parser.parse_args()
    if args.verify_resolved_output:
        report = verify(
            args.destination,
            args.rung,
            sas_reference=args.sas_reference,
            require_resolved_output=True,
        )
    else:
        report = stage(
            args.source,
            args.destination,
            args.rung,
            sas_reference=args.sas_reference,
            plant=args.plant,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Admit an ORCA2 SI3 exact-input score only for a supported full identity.

This is deliberately an admission gate, not a reduced-physics comparator.  A
valid binary record does not authorize mixing the ORCA2 deck with the narrower
ORCA1/C1D SI3 identity currently implemented by legoESM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from legoesm.ice.config import SI3ThermoConfig

import nemo_si3_stream_header_gate as header_gate


class AdmissionError(RuntimeError):
    """The exact-input rung cannot be constructed from the resolved identity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_value(text: str, label: str, cast):
    match = re.search(rf"\b{re.escape(label)}\s*=\s*([^\s]+)", text)
    if not match:
        raise AdmissionError(f"missing resolved selector line: {label}")
    token = match.group(1)
    if cast is bool:
        if token not in {"T", "F"}:
            raise AdmissionError(f"invalid logical for {label}: {token}")
        return token == "T"
    return cast(token)


def resolve_selectors(text: str) -> dict[str, object]:
    """Read the values NEMO printed after cfg-over-ref resolution."""
    return {
        "n_categories": _resolved_value(text, "jpl", int),
        "n_ice_layers": _resolved_value(text, "nlay_i", int),
        "n_snow_layers": _resolved_value(text, "nlay_s", int),
        "conductivity": (
            "p07" if _resolved_value(text, "ln_cndi_P07", bool) else "not_p07"
        ),
        "salinity_scheme": _resolved_value(text, "nn_icesal", int),
        "new_ice_salinity_fraction": _resolved_value(text, "rn_sinew", float),
        "drainage": _resolved_value(text, "ln_drainage", bool),
        "flushing": _resolved_value(text, "ln_flushing", bool),
        "ponds": _resolved_value(text, "ln_pnd", bool),
        "lateral_melt": _resolved_value(text, "ln_icedA", bool),
    }


def supported_selectors() -> dict[str, object]:
    identity = SI3ThermoConfig()
    return {
        "n_categories": 1,
        **identity._asdict(),
    }


def selector_rows(resolved: dict[str, object]) -> list[dict[str, object]]:
    supported = supported_selectors()
    return [
        {
            "selector": name,
            "oracle": resolved[name],
            "implemented_identity": supported[name],
            "status": "VERIFIED" if resolved[name] == supported[name] else "UNSUPPORTED",
        }
        for name in supported
    ]


def _thd_stage_counts(path: Path) -> dict[str, int]:
    """Count the already header-validated ORCA2 ice_thd frame identifiers."""
    import struct

    counts: Counter[int] = Counter()
    with path.open("rb") as handle:
        while magic := handle.read(16):
            if magic != b"NEMO_L3THD_001  ":
                raise AdmissionError("thd stream lost framing after header validation")
            header = handle.read(44)
            if len(header) != 44:
                raise AdmissionError("truncated thd header")
            _version, _step, stage, payload, nx, ny, nc, ni, ns, npti, bits = struct.unpack(
                "=11i", header
            )
            nval = nx * ny * nc * (9 + 2 * ni + ns) if payload == 0 else npti * (
                4 + 2 * ni + ns
            )
            handle.seek(nval * (bits // 8), 1)
            counts[stage] += 1
    return {str(key): value for key, value in sorted(counts.items())}


def run(root: Path) -> dict[str, object]:
    header_result = header_gate.validate_root(root)
    output = root / "ocean.output"
    if not output.is_file():
        raise AdmissionError(f"missing resolved NEMO output: {output}")
    resolved = resolve_selectors(output.read_text(encoding="utf-8", errors="replace"))
    rows = selector_rows(resolved)
    unsupported = [row for row in rows if row["status"] == "UNSUPPORTED"]
    streams = {}
    for name in (
        "oracle_si3_thd_frames.bin",
        "oracle_si3_zdf_inputs.bin",
        "oracle_si3_exchange_frames.bin",
        "oracle_si3_bulk_operands.bin",
    ):
        path = root / name
        if path.is_file():
            streams[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    result = {
        "format": "nemo-orca2-si3-exact-input-admission-v1",
        "root": str(root),
        "resolved_output": {"path": str(output), "sha256": _sha256(output)},
        "header_gate": header_result,
        "streams": streams,
        "thd_stage_counts": _thd_stage_counts(root / "oracle_si3_thd_frames.bin"),
        "selector_rows": rows,
        "scored_physics_rows": 0,
        "bit_identity": "NOT_MEASURED",
    }
    if unsupported:
        result.update({
            "status": "STOP_SELECTOR_GAP",
            "unsupported_selectors": [row["selector"] for row in unsupported],
            "reason": (
                "ORCA2 does not resolve the one implemented ORCA1/C1D SI3 "
                "thermodynamic identity; a partial or mixed score is forbidden"
            ),
        })
    else:
        result["status"] = "ADMITTED_NOT_SCORED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.root)
        code = 0 if result["status"] == "ADMITTED_NOT_SCORED" else 1
    except (AdmissionError, header_gate.HeaderError, OSError, ValueError) as exc:
        result = {"root": str(args.root), "status": "INVALID", "error": str(exc)}
        code = 2
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

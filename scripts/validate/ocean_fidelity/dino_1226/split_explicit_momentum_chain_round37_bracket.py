#!/usr/bin/env python3
"""Byte-exact duplicate-capture bracket for round 37."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_BIN = {
    f"{component}_{name}.bin"
    for component in ("u", "v")
    for name in (
        "vertical_advection", "vorticity", "lateral_friction",
        "ke_gradient_plus_hpg", "mapped_d06", "diagnostic_total",
        "diagnostic_sum", "surface_stress_outside_d03_d06",
    )
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _manifest(directory: Path) -> dict[str, dict[str, object]]:
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha(path)}
        for path in sorted(directory.iterdir()) if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-a", type=Path, required=True)
    parser.add_argument("--capture-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = args.capture_a.resolve(), args.capture_b.resolve()
    ma, mb = _manifest(a), _manifest(b)
    expected = EXPECTED_BIN | {"capture.json"}
    exact_names = set(ma) == expected and set(mb) == expected
    shared_exact = exact_names and ma == mb
    metadata_a = json.loads((a / "capture.json").read_text())
    metadata_b = json.loads((b / "capture.json").read_text())
    metadata_valid = (
        metadata_a == metadata_b
        and metadata_a.get("schema")
        == "dino-split-explicit-momentum-chain-round37-capture-v2"
        and set(metadata_a.get("files", {})) == EXPECTED_BIN
    )
    first = a / sorted(EXPECTED_BIN)[0]
    planted = bytearray(first.read_bytes())
    planted[0] ^= 1
    one_bit_fires = hashlib.sha256(planted).hexdigest() != _sha(first)
    missing_fires = (set(ma) - {sorted(EXPECTED_BIN)[0]}) != expected
    planted_metadata = json.loads(json.dumps(metadata_a))
    planted_metadata["native_shapes"]["u"] = [199, 52, 36]
    shape_fires = planted_metadata["native_shapes"] != metadata_a["native_shapes"]
    controls = {
        "one_bit_file_plant_fires": one_bit_fires,
        "missing_file_plant_fires": missing_fires,
        "wrong_full_halo_shape_plant_fires": shape_fires,
    }
    valid = shared_exact and metadata_valid and all(controls.values())
    receipt = {
        "schema": "dino-split-explicit-momentum-chain-round37-bracket-v1",
        "capture_a": str(a),
        "capture_b": str(b),
        "file_count": len(ma),
        "expected_names": sorted(expected),
        "shared_exact": shared_exact,
        "metadata_valid": metadata_valid,
        "manifest": ma,
        "controls": controls,
        "disposition": "CAPTURE_BRACKET_EXACT" if valid else "INVALID",
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"disposition={receipt['disposition']} files={len(ma)}")
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

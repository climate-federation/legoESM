#!/usr/bin/env python3
"""Score the complete round-335 parent/current non-NEMO caller census.

The producer writes one NPZ per checkout.  This gate refuses an incomplete
inventory, compares every returned array bit for bit, and has an in-memory
one-ULP plant so q_net cannot silently fall out of the score again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_ARRAYS = 214
EXPECTED_WORDS = 10_231_076
SURFACE_OUTPUTS = ("tau_x", "tau_y", "q_net", "sw_down")
FRESHWATER_OUTPUTS = ("precip", "evap", "runoff", "ice_fw", "restoring")


def required_keys() -> set[str]:
    keys: set[str] = set()
    for domain in ("A", "B"):
        for path in ("app_sf_abs", "app_sf_fb", "app_jax"):
            keys.update(f"{domain}/{path}/{name}" for name in SURFACE_OUTPUTS)
        for path in ("app_fw_abs", "app_fw_fb"):
            keys.update(f"{domain}/{path}/{name}" for name in FRESHWATER_OUTPUTS)
    return keys


def score(parent_path: Path, current_path: Path, plant_key: str | None) -> dict:
    with np.load(parent_path, allow_pickle=False) as parent, np.load(
        current_path, allow_pickle=False
    ) as current:
        parent_keys = set(parent.files)
        current_keys = set(current.files)
        missing_required = sorted(required_keys() - parent_keys)
        inventory_equal = parent_keys == current_keys
        rows = []
        words = 0
        unequal = 0
        plant_fired = None
        for key in sorted(parent_keys & current_keys):
            lhs = np.asarray(parent[key])
            rhs = np.asarray(current[key]).copy()
            if plant_key == key:
                if rhs.dtype != np.float64 or not rhs.size:
                    raise ValueError(f"plant requires nonempty float64 array: {key}")
                flat = rhs.reshape(-1)
                flat[0] = np.nextafter(flat[0], np.inf)
            compatible = lhs.shape == rhs.shape and lhs.dtype == rhs.dtype
            if compatible and lhs.dtype == np.float64:
                count = int(np.count_nonzero(
                    lhs.view(np.uint64) != rhs.view(np.uint64)
                ))
            elif compatible:
                count = int(np.count_nonzero(lhs != rhs))
            else:
                count = max(lhs.size, rhs.size)
            words += lhs.size
            unequal += count
            if count or not compatible:
                rows.append({
                    "key": key,
                    "words": int(lhs.size),
                    "unequal": count,
                    "shape_dtype_equal": compatible,
                })
            if plant_key == key:
                plant_fired = count > 0

    complete = (
        inventory_equal
        and not missing_required
        and len(parent_keys) == EXPECTED_ARRAYS
        and words == EXPECTED_WORDS
    )
    status = "PASS" if complete and unequal == 0 else "FAIL"
    if plant_key is not None and plant_fired is not True:
        status = "FAIL"
    return {
        "format": "orca2-round335-non-nemo-control-v1",
        "parent": str(parent_path.resolve()),
        "current": str(current_path.resolve()),
        "arrays": len(parent_keys),
        "words": words,
        "unequal": unequal,
        "inventory_equal": inventory_equal,
        "missing_required_outputs": missing_required,
        "plant_key": plant_key,
        "plant_fired": plant_fired,
        "differences": rows,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-key", choices=sorted(required_keys()))
    args = parser.parse_args()
    report = score(args.parent, args.current, args.plant_key)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    # A planted run must fail closed after observing its changed word.
    if args.plant_key:
        return 1 if report["status"] == "FAIL" and report["plant_fired"] else 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

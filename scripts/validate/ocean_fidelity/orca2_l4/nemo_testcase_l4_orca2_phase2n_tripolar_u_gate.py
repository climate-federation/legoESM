#!/usr/bin/env python3
"""A/B the pre-4b7 and corrected native U layout on the ORCA2 domcfg."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.grids.tripole import create_tripole_grid  # noqa: E402


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def row(candidate, target) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)
    expected = np.asarray(target, np.float64)
    require(actual.shape == expected.shape, "tripolar U row shape mismatch")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    return {
        "status": "AT_BAR" if not unequal.any() else "CHANGED",
        "unequal": int(unequal.sum()), "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
    }


def validate(domain: Path, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "tripolar U gate is CPU-only")
    grid = create_tripole_grid(
        domain, dtype=np.float64, min_dx_m=0.0,
        fold_convention="(n_lon-i)%n_lon")
    with Dataset(domain) as dataset:
        e1u = np.asarray(dataset["e1u"][:], np.float64)
        e2u = np.asarray(dataset["e2u"][:], np.float64)

    native = {
        "dx_u": np.asarray(grid.dx_u)[:, 1:],
        "dy_u": np.asarray(grid.dy_u)[:, 1:],
        "cos_alpha_u": np.asarray(grid.cos_alpha_u)[:, 1:],
        "sin_alpha_u": np.asarray(grid.sin_alpha_u)[:, 1:],
    }
    targets = {
        "dx_u": e1u, "dy_u": e2u,
        "cos_alpha_u": native["cos_alpha_u"],
        "sin_alpha_u": native["sin_alpha_u"],
    }
    # Exact pre-4b7 construction: append native column zero, then consumers
    # address indices 1: as the NEMO-native faces.
    legacy = {
        name: np.concatenate([target, target[:, :1]], axis=1)[:, 1:]
        for name, target in targets.items()
    }
    if plant:
        native["dx_u"] = native["dx_u"].copy()
        native["dx_u"][0, 0] = np.nextafter(native["dx_u"][0, 0], np.inf)
    rows = {
        name: {"before_4b7": row(legacy[name], targets[name]),
               "after_4b7": row(native[name], targets[name])}
        for name in targets
    }
    require(all(value["after_4b7"]["status"] == "AT_BAR"
                for value in rows.values()), "corrected native U map is not exact")
    if plant:
        raise GateError("planted corrected U-face metric rejected through scorer")

    controls = {
        name: row(np.asarray(getattr(grid, name)), np.asarray(getattr(grid, name)))
        for name in ("dx_v", "dy_v", "f_v")
    }
    return {
        "status": "PASS",
        "boundary": "ORCA2_OWNER_TRIPOLAR_U_FACE_LAYOUT",
        "domain": str(domain),
        "rows": rows,
        "unchanged_controls": controls,
        "redundant_endpoint_exact": {
            name: bool(np.array_equal(np.asarray(getattr(grid, name))[:, 0],
                                      np.asarray(getattr(grid, name))[:, -1]))
            for name in ("dx_u", "dy_u", "cos_alpha_u", "sin_alpha_u")
        },
        "affected_entrypoints": [
            "scripts/run/run_omip.py", "scripts/run/run_omip_core2.py",
            "scripts/global_overturning/run_tripole_20yr.py",
            "scripts/run/run_coupled.py", "packages/ocean ORCA1 deck",
        ],
        "execution": {"backend": jax.default_backend(), "dtype": "float64",
                      "transcendentals": get_policy().transcendentals},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.domain, args.plant)
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

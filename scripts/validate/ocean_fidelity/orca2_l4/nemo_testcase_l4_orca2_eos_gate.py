#!/usr/bin/env python3
"""Cellwise ORCA2 EOS-80 boundary gate on NEMO's stage-2 operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.eos import (  # noqa: E402
    _ROQUET_EOS80,
    nemo_roquet_density_anomaly_ratio,
)

NX, NY, NZ = 94, 152, 31
ACTIVE_Z = NZ - 1
COEFFICIENTS = (
    "EOS000", "EOS100", "EOS200", "EOS300", "EOS400", "EOS500", "EOS600",
    "EOS010", "EOS110", "EOS210", "EOS310", "EOS410", "EOS510",
    "EOS020", "EOS120", "EOS220", "EOS320", "EOS420",
    "EOS030", "EOS130", "EOS230", "EOS330", "EOS040", "EOS140", "EOS240",
    "EOS050", "EOS150", "EOS060", "EOS001", "EOS101", "EOS201", "EOS301",
    "EOS401", "EOS011", "EOS111", "EOS211", "EOS311", "EOS021", "EOS121",
    "EOS221", "EOS031", "EOS131", "EOS041", "EOS002", "EOS102", "EOS202",
    "EOS012", "EOS112", "EOS022", "EOS003", "EOS103", "EOS013",
)
ARRAYS = (
    "T", "S", "pdep", "zh", "zt", "zs", "ztm",
    "zn0", "zn1", "zn2", "zn3", "zn", "prd",
)


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


def _xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F").transpose(1, 0, 2)


def read_eos(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L2_EOSOP_1", f"bad EOS magic {magic!r}")
    require(
        header == (1, 1, 2, 3, 3, 2, NX, NY, NZ, 64, 0),
        f"bad EOS header {header}",
    )
    n3 = NX * NY * NZ
    scalar_count = 6 + len(COEFFICIENTS)
    require(values.size == scalar_count + len(ARRAYS) * n3, "bad EOS payload size")
    scalars = dict(zip(
        ("rdeltaS", "r1_S0", "r1_T0", "r1_Z0", "rho0", "r1_rho0"),
        values[:6], strict=True,
    ))
    coefficients = dict(zip(COEFFICIENTS, values[6:scalar_count], strict=True))
    arrays = {}
    for index, name in enumerate(ARRAYS):
        begin = scalar_count + index * n3
        arrays[name] = _xyz(values[begin:begin + n3])[2:-2, 2:-2, :ACTIVE_Z]
    return {"header": header, "scalars": scalars, "coefficients": coefficients,
            "arrays": arrays}


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict:
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.shape == expected.shape and actual.size, "empty or mismatched score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "non-finite score")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
    }


def validate(root: Path, *, plant: bool) -> dict:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "EOS gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    path = root / "oracle_rkstage2_eos_operands_kt00000001.bin"
    record = read_eos(path)
    oracle = record["arrays"]
    expected_scalars = {
        "rdeltaS": _ROQUET_EOS80["rdeltaS"],
        "r1_S0": _ROQUET_EOS80["r1_S0"],
        "r1_T0": _ROQUET_EOS80["r1_T0"],
        "r1_Z0": _ROQUET_EOS80["r1_Z0"],
        "rho0": 1026.0,
        "r1_rho0": 1.0 / 1026.0,
    }
    require(record["scalars"] == expected_scalars, "EOS scalar table differs")
    require(record["coefficients"] == {
        name: _ROQUET_EOS80[name] for name in COEFFICIENTS
    }, "EOS-80 coefficient table differs")

    operator = jax.jit(
        lambda t, s, depth, mask: nemo_roquet_density_anomaly_ratio(
            t, s, depth * 0.0, coeffs=_ROQUET_EOS80, rho0=1026.0,
            geometric_depth_m=depth, tmask=mask, return_intermediates=True,
        )
    )
    candidates = tuple(np.asarray(value) for value in operator(
        oracle["T"], oracle["S"], oracle["pdep"], oracle["ztm"]
    ))
    mask = oracle["ztm"] != 0.0
    rows = []
    for name, candidate in zip(ARRAYS, candidates, strict=True):
        trial = candidate.copy()
        if plant and name == "prd":
            index = tuple(np.argwhere(mask)[0])
            trial[index] = np.nextafter(trial[index], np.inf)
        rows.append({"field": name, **score(trial, oracle[name], mask)})
    if plant:
        planted = next(row for row in rows if row["field"] == "prd")
        require(planted["unequal"] > 0, "binding EOS plant did not fire")
        raise GateError(
            f"planted EOS cell rejected through scorer "
            f"({planted['unequal']}/{planted['count']})"
        )
    first = next((row["field"] for row in rows if row["status"] != "AT_BAR"), None)
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O3-EOS/stage2-eos_insitu",
        "owner": "GYRE_OWNER_SHARED_EOS",
        "result": "AT_BAR" if first is None else "DEBT",
        "first_over_bar_field": first,
        "rows": rows,
        "record": str(path),
        "record_sha256": sha256(path),
        "execution": {
            "backend": jax.default_backend(), "jax_disable_jit": False,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 owned wet T cells, 90x148x30",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.oracle_root, plant=args.plant)
    except (GateError, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

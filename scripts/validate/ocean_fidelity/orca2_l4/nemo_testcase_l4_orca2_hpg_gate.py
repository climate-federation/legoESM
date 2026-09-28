#!/usr/bin/env python3
"""Cellwise ORCA2 ``hpg_sco`` literal gate on NEMO's stage-2 operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    nemo_hpg_sco_literal_cgrid,
)

NX, NY, NZ = 94, 152, 31
ACTIVE_Z = NZ - 1
COMPONENTS = ("sum_u", "sum_v", "zhpi_u", "zhpi_v", "zuap_u", "zuap_v")


class GateError(RuntimeError):
    pass


class LocalMetricGrid(NamedTuple):
    dx_u: jax.Array
    dy_v: jax.Array


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


def _xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F").T


def read_operands(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L2_HPGOP_1", f"bad HPG operand magic {magic!r}")
    require(header == (1, 1, 3, NX, NY, NZ, 64), f"bad HPG operand header {header}")
    n2 = NX * NY
    require(values.size == 3 * n2 * NZ, "bad HPG operand payload size")
    slabs = {name: [] for name in ("rhd", "e3w", "gdept_z0")}
    cursor = 0
    for _ in range(NZ):
        for name in slabs:
            slabs[name].append(_xy(values[cursor:cursor + n2]))
            cursor += n2
    return {name: np.stack(parts, axis=-1) for name, parts in slabs.items()}


def read_literal(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L2_HPGLT_1", f"bad HPG literal magic {magic!r}")
    require(header == (1, 1, 3, 2, NX, NY, NZ, 64), f"bad HPG literal header {header}")
    n2, n3 = NX * NY, NX * NY * NZ
    require(values.size == 6 * n3 + 2 * n2, "bad HPG literal payload size")
    result = {
        name: _xyz(values[index * n3:(index + 1) * n3])
        for index, name in enumerate((
            "zhpi_u", "zhpi_v", "zuap_u", "zuap_v", "sum_u", "sum_v",
        ))
    }
    result["r1_e1u"] = _xy(values[6 * n3:6 * n3 + n2])
    result["r1_e2v"] = _xy(values[6 * n3 + n2:])
    return result


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
    require(jax.default_backend() == "cpu", "HPG gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    operand_path = root / "oracle_rkstage2_hpg_operands_kt00000001.bin"
    literal_path = root / "oracle_rkstage2_hpg_literal_kt00000001.bin"
    operands = read_operands(operand_path)
    literal = read_literal(literal_path)
    with Dataset(root / "mesh_mask_0000.nc") as dataset:
        umask = np.asarray(dataset["umask"][0, :ACTIVE_Z].data, dtype=bool)
        vmask = np.asarray(dataset["vmask"][0, :ACTIVE_Z].data, dtype=bool)
    umask = umask.transpose(1, 2, 0)[1:-1, 1:-1]
    vmask = vmask.transpose(1, 2, 0)[1:-1, 1:-1]

    # The canonical record intentionally zeroes MPI halos.  Keeping the full
    # 94x152 local arrays makes every rank-owned neighbour available except at
    # the outer canonical band; score the 88x146 stencil-valid owned core.
    r1_e1u = literal["r1_e1u"]
    r1_e2v = literal["r1_e2v"]
    dx_u = np.ones((NY, NX + 1), np.float64)
    dy_v = np.ones((NY + 1, NX), np.float64)
    dx_u[:, 1:] = np.divide(1.0, r1_e1u, out=np.ones_like(r1_e1u), where=r1_e1u != 0.0)
    dy_v[1:, :] = np.divide(1.0, r1_e2v, out=np.ones_like(r1_e2v), where=r1_e2v != 0.0)
    grid = LocalMetricGrid(jnp.asarray(dx_u), jnp.asarray(dy_v))

    values = tuple(np.asarray(value) for value in nemo_hpg_sco_literal_cgrid(
        jnp.asarray(operands["rhd"]),
        jnp.asarray(operands["e3w"]),
        jnp.asarray(operands["gdept_z0"]),
        grid, 9.80665, return_components=True,
    ))
    candidates = dict(zip(COMPONENTS, values, strict=True))
    rows = []
    for name in COMPONENTS:
        is_u = name.endswith("_u")
        native = candidates[name][:, 1:] if is_u else candidates[name][1:]
        candidate = native[3:-3, 3:-3, :ACTIVE_Z]
        oracle = literal[name][3:-3, 3:-3, :ACTIVE_Z]
        mask = umask if is_u else vmask
        trial = candidate.copy()
        if plant and name == "sum_u":
            index = tuple(np.argwhere(mask)[0])
            trial[index] = np.nextafter(trial[index], np.inf)
        rows.append({"field": name, **score(trial, oracle, mask)})
    if plant:
        planted = next(row for row in rows if row["field"] == "sum_u")
        require(planted["unequal"] > 0, "binding HPG plant did not fire")
        raise GateError(
            f"planted HPG cell rejected through scorer "
            f"({planted['unequal']}/{planted['count']})"
        )
    first = next((row["field"] for row in rows if row["status"] != "AT_BAR"), None)
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O4-HPG/stage2-hpg_sco",
        "owner": "GYRE_OWNER_SHARED_HPG",
        "result": "AT_BAR" if first is None else "DEBT",
        "first_over_bar_field": first,
        "rows": rows,
        "records": {
            operand_path.name: sha256(operand_path),
            literal_path.name: sha256(literal_path),
        },
        "execution": {
            "backend": jax.default_backend(), "jax_disable_jit": False,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 stencil-valid owned wet U/V core, 88x146x30",
            "halo_rule": "one owned-cell band excluded because canonical MPI halos are zero",
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

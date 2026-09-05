#!/usr/bin/env python3
"""Cellwise ORCA2 ``tra_qsr`` RGB boundary gate at kt=1, RK stage 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
from legoesm.ocean.physics.shortwave_penetration import (  # noqa: E402
    ShortwavePenetrationConfig,
    shortwave_penetration_rgb_tendency,
)

NX, NY, NZ = 94, 152, 31
OWN_X, OWN_Y, ACTIVE_Z = NX - 4, NY - 4, NZ - 1


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


def _full_xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F").T


def _owned_xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((OWN_X, OWN_Y), order="F").T


def _full_xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F").transpose(1, 0, 2)


def read_rgb(path: Path) -> dict[str, np.ndarray | int]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L4_CHL_001", "RGB record magic")
    require(header == (1, 1, 2, NX, NY, NZ, 1, 64), f"RGB header {header}")
    n2, ni = NX * NY, OWN_X * OWN_Y
    require(values.size == 2 * n2 + NZ + ni, "RGB payload size")
    return {
        "chl": _full_xy(values[:n2])[2:-2, 2:-2],
        "qsr_full_storage": values[n2:2 * n2],
        "qsr": _full_xy(values[n2:2 * n2])[2:-2, 2:-2],
        "gdepw_1d": values[2 * n2:2 * n2 + NZ],
        "r3t": _owned_xy(values[2 * n2 + NZ:]),
    }


def read_qsr(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L2_QSR___1", "QSR record magic")
    require(header == (1, 1, 3, 2, 3, NX, NY, NZ, 64), f"QSR header {header}")
    ni, n3 = OWN_X * OWN_Y, NX * NY * NZ
    require(values.size == ni + n3, "QSR payload size")
    return {
        # This inherited record writes the first A2D-sized storage segment of
        # the full qsr array, not a physically cropped owned rectangle.  It is
        # an integrity echo only; the RGB input frame carries the full field.
        "qsr_storage_prefix": values[:ni],
        "increment": _full_xyz(values[ni:])[2:-2, 2:-2, :ACTIVE_Z],
    }


def read_o1_qsr(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        require(handle.read(16).decode("ascii").rstrip() == "NEMO_L4_BLKIO_1",
                "O1 frame-0 magic")
        require(struct.unpack("=8i", handle.read(32)) ==
                (1, 1, 0, OWN_X, OWN_Y, 9, 0, 64), "O1 frame-0 header")
        handle.seek(9 * OWN_X * OWN_Y * 8, 1)
        require(handle.read(16).decode("ascii").rstrip() == "NEMO_L4_BLKIO_1",
                "O1 frame-1 magic")
        require(struct.unpack("=8i", handle.read(32)) ==
                (1, 1, 1, OWN_X, OWN_Y, 20, 0, 64), "O1 frame-1 header")
        values = np.fromfile(handle, np.float64)
    require(values.size == 20 * OWN_X * OWN_Y, "O1 frame-1 payload")
    # Frame-1 field 13 (zero based) is qsr.
    begin = 13 * OWN_X * OWN_Y
    return _owned_xy(values[begin:begin + OWN_X * OWN_Y])


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict:
    require(candidate.shape == oracle.shape == mask.shape, "score shape")
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "non-finite score")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    delta = np.abs(actual - expected)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(unequal.size),
        "max_abs": float(delta.max(initial=0.0)),
    }


def validate(root: Path, *, plant: bool) -> dict:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "RGB gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    rgb = read_rgb(root / "oracle_rgb_chl_kt00000001.bin")
    qsr = read_qsr(root / "oracle_qsr_stage3_kt00000001.bin")
    read_o1_qsr(root / "oracle_sbcblk_o1_kt00000001.bin")
    with Dataset(root / "mesh_mask_0000.nc") as dataset:
        wet = np.asarray(dataset["tmask"][0, :ACTIVE_Z].data, dtype=bool)
    wet = wet.transpose(1, 2, 0)
    require(np.array_equal(
        np.asarray(rgb["qsr_full_storage"])[:OWN_X * OWN_Y],
        qsr["qsr_storage_prefix"],
    ), "RGB/QSR qsr storage echo mismatch")
    dz_ref = np.diff(np.asarray(rgb["gdepw_1d"], np.float64))
    stretch = 1.0 + np.asarray(rgb["r3t"])[..., None]
    dz_live = dz_ref * stretch
    config = ShortwavePenetrationConfig(
        scheme="nemo_qsr_rgb",
        rgb_ir_fraction=0.58,
        rgb_ir_extinction_m=0.35,
        rgb_chl_profile="morel_berthon",
    )
    operator = jax.jit(
        lambda sw, chl, dz, mask: shortwave_penetration_rgb_tendency(
            sw, chl, dz, mask, config
        )
    )
    candidate = np.asarray(operator(rgb["qsr"], rgb["chl"], dz_live, wet)).copy()
    if plant:
        candidate = np.asarray(qsr["increment"]).copy()
        index = tuple(np.argwhere(wet)[0])
        candidate[index] = np.nextafter(candidate[index], np.inf)
    result = score(candidate, qsr["increment"], wet)
    if plant:
        require(result["unequal"] > 0, "binding RGB plant did not fire")
        raise GateError(
            f"planted RGB cell rejected through scorer ({result['unequal']}/{result['count']})"
        )
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O2-RGB/tra_qsr",
        "owner": "ORCA2_OWNER",
        "result": result,
        "oracle_operand_mode": "ORACLE_SUPPLIED_POST_SI3_QSR_FROM_RGB_FRAME",
        "surface_operand_chain": (
            "ORACLE_SUPPLIED_O1_FRAME_1_BULK_OUTPUTS_THEN_"
            "ORACLE_SUPPLIED_SI3_AGGREGATE_QSR"
        ),
        "record_sha256": {
            name: sha256(root / name) for name in (
                "oracle_sbcblk_o1_kt00000001.bin",
                "oracle_rgb_chl_kt00000001.bin",
                "oracle_qsr_stage3_kt00000001.bin",
            )
        },
        "execution": {
            "backend": jax.default_backend(),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 owned wet T cells, 90x148x30",
        },
        "downstream": "NOT_ENTERED_AFTER_FIRST_DEBT",
        "si3": "UNMEASURED_PENDING_ICE_MERGE_ORACLE_SUPPLIED_EXCHANGE",
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

#!/usr/bin/env python3
"""ORCA2 stage-1 zFu/zFv operand-substitution fidelity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    _nemo_metric_stage_transport,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402

NX, NY, NZ = 94, 152, 31
OWNED_NX, OWNED_NY = NX - 4, NY - 4


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


def _xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F")[2:-2, 2:-2].T


def _xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def read_record(path: Path) -> dict[str, np.ndarray | dict[str, int | str]]:
    """Validate and decode the source-adjacent rank-zero transport stream."""
    require(time_level_for_dump(path.name) == "now", "transport registry is not Kmm/now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    require(magic == "NEMO_L2_TRPOP_2", f"bad transport magic {magic!r}")
    require(header == (2, 1, 1, 1, NX, NY, NZ, 64), f"bad transport header {header}")
    n2, n3 = NX * NY, NX * NY * NZ
    require(values.size == 10 * n2 + 8 * n3, "bad derived transport payload size")
    require(np.isfinite(values).all(), "non-finite transport payload")
    result: dict[str, np.ndarray | dict[str, int | str]] = {}
    offset = 0
    for name, size, transform in (
        ("e2u", n2, _xy), ("e3u", n3, _xyz), ("uu", n3, _xyz),
        ("zub", n2, _xy), ("umask", n3, _xyz), ("zFu", n3, _xyz),
        ("e1v", n2, _xy), ("e3v", n3, _xyz), ("vv", n3, _xyz),
        ("zvb", n2, _xy), ("vmask", n3, _xyz), ("zFv", n3, _xyz),
        ("un_adv", n2, _xy), ("r1_hu", n2, _xy), ("uu_b", n2, _xy),
        ("vn_adv", n2, _xy), ("r1_hv", n2, _xy), ("vv_b", n2, _xy),
    ):
        result[name] = transform(values[offset:offset + size])
        offset += size
    require(offset == values.size, "transport schema walk did not reach EOF")
    result["header"] = {
        "version": 2, "kt": 1, "stage": 1, "Kmm": 1,
        "jpi": NX, "jpj": NY, "jpk": NZ, "bits": 64,
        "registry_level": "now",
    }
    return result


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.shape == expected.shape and actual.size, "empty transport score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite defined transport cell")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    ordered_actual = actual.view(np.int64) ^ ((actual.view(np.int64) >> 63) & 0x7fffffffffffffff)
    ordered_expected = expected.view(np.int64) ^ ((expected.view(np.int64) >> 63) & 0x7fffffffffffffff)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
        "max_ulp": int(np.abs(ordered_actual - ordered_expected).max(initial=0)),
    }


def validate(oracle_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "transport gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    path = oracle_root / "oracle_rkstage1_transport_operands_kt00000001.bin"
    require(path.is_file(), f"missing {path}")
    oracle = read_record(path)

    # The correction is the source statement immediately before the shared
    # production zF helper.  All inputs, including the six external-mode
    # quantities, come from NEMO's record: this is an operand-substitution arm.
    def source_program(metric, e3, velocity, correction, mask):
        corrected = nemo_source_round(
            velocity + nemo_source_round(correction[..., None] * mask))
        return _nemo_metric_stage_transport(metric, e3, corrected)

    program = jax.jit(source_program)
    rows: list[dict[str, object]] = []
    for face, names in (
        ("U", ("e2u", "e3u", "uu", "zub", "umask", "zFu")),
        ("V", ("e1v", "e3v", "vv", "zvb", "vmask", "zFv")),
    ):
        metric, e3, velocity, correction, mask, target = (
            np.asarray(oracle[name]) for name in names
        )
        # jpk is NEMO's all-zero dummy record; the source DO_3D writes jpkm1.
        e3, velocity, mask, target = (
            value[..., :-1] for value in (e3, velocity, mask, target)
        )
        live = mask != 0.0
        trial = np.asarray(program(
            jnp.asarray(metric), jnp.asarray(e3), jnp.asarray(velocity),
            jnp.asarray(correction), jnp.asarray(mask)))
        if plant and face == "U":
            # Bind the very same raw-bit scorer without borrowing a pre-existing
            # debt: start the control from the exact oracle payload.
            trial = target.copy()
            index = tuple(np.argwhere(live)[0])
            trial[index] = np.nextafter(trial[index], np.inf)
        rows.append({"face": face, **score(trial, target, live)})

    if plant:
        require(rows[0]["unequal"] == 1, "transport plant did not fire once")
        raise GateError(
            "planted stage-1 transport cell rejected through scorer "
            f"({rows[0]['unequal']}/{rows[0]['count']})"
        )
    first = next((row["face"] for row in rows if row["status"] != "AT_BAR"), None)
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O5-A/stage1-horizontal-zFu-zFv",
        "result": "AT_BAR" if first is None else "DEBT",
        "first_over_bar_face": first,
        "owner": (
            "CONFIRMED_SHARED_STAGE_TRANSPORT_AT_BAR" if first is None
            else "GYRE_OWNER_SHARED_STAGE_TRANSPORT"
        ),
        "rows": rows,
        "record": {"path": str(path), "sha256": sha256(path)},
        "operand_substitution": {
            "label": "ORACLE_SUPPLIED_EXTERNAL_MODE",
            "fields": ["un_adv", "vn_adv", "r1_hu", "r1_hv", "uu_b", "vv_b"],
            "certifies_external_mode": False,
        },
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 two-halo-stripped live U/V cells; jpkm1",
        },
        "source": {
            "program": "stprk3_stg.F90:265-280",
            "writer": "stprk3_stg.F90:283-310",
            "shared_helper": "ocean_model_latlon_cgrid.py:1301-1305",
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

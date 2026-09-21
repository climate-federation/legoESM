#!/usr/bin/env python3
"""Discriminate vector-math from scalar-libm H79 strength at ICE_RHEO step 8.

The oracle operand is a WRITE-only copy-run dump made immediately after
``icedyn_rhg_evp.F90:420-427`` in subcycle one.  The shipped NEMO tree is read
only; the probe executable and its dump live below the external run root.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import netCDF4
import numpy as np
from legoesm.ice.dynamics import _SI3_ICE_PRESENCE
from legoesm.ice.fidelity.nemo_rheo_testcase_recipe import build_ice_rheo_card
from legoesm.ocean.fidelity.provenance import worktree_stamp

from legoesm import constants

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
PROBE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/round10_subcycle_probe"
)
HERE = Path(__file__).resolve().parent
ORACLE_GATE = HERE / "nemo_si3_oracle_gate.py"
FRAME_STEP = 8
SUBCYCLE = 1
N_DUMP_FULL_FIELDS = 4
MESH_FIELDS = (
    "e1t",
    "e2t",
    "e1u",
    "e2u",
    "e1v",
    "e2v",
    "e1f",
    "e2f",
    "tmask",
    "umask",
    "vmask",
)
_HASH_BLOCK_BYTES = 1024 * 1024


class H79ProbeError(RuntimeError):
    """Fail-closed H79 discriminator error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H79ProbeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_oracle_gate():
    spec = importlib.util.spec_from_file_location("nemo_si3_oracle_gate_round10", ORACLE_GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_dump(path: Path) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    with path.open("rb") as stream:
        header = stream.read(struct.calcsize("=3i"))
        require(len(header) == struct.calcsize("=3i"), "truncated subcycle header")
        jpi, jpj, jter = struct.unpack("=3i", header)
        arrays = []
        for _ in range(N_DUMP_FULL_FIELDS):
            values = np.fromfile(stream, dtype=np.float64, count=jpi * jpj)
            require(values.size == jpi * jpj, "truncated full-grid subcycle field")
            arrays.append(values.reshape((jpi, jpj), order="F"))
    require(jter == SUBCYCLE, f"expected subcycle {SUBCYCLE}, got {jter}")
    return {"jpi": jpi, "jpj": jpj, "jter": jter}, dict(
        zip(("shear", "delta", "p_over_delta", "beta"), arrays, strict=True)
    )


def _scalar_libm_exp(values: np.ndarray) -> np.ndarray:
    libm = ctypes.CDLL("libm.so.6")
    function = libm.exp
    function.argtypes = (ctypes.c_double,)
    function.restype = ctypes.c_double
    result = np.empty_like(values)
    for index, value in enumerate(values.flat):
        result.flat[index] = function(float(value))
    return result


def _score(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    difference = np.abs(candidate - reference)
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    return {
        "bitwise_nonzero_over_n": f"{np.count_nonzero(difference)} / {difference.size}",
        "max_abs": float(difference[index]),
        "max_abs_index_xy": [int(value) for value in index],
    }


def run_probe(root: Path = ROOT, probe_root: Path = PROBE_ROOT) -> tuple[dict[str, object], int]:
    oracle_gate = _load_oracle_gate()
    frame_path = root / f"oracle_ice_step_entry_kt{FRAME_STEP:08d}.bin"
    mesh_path = root / "mesh_mask.nc"
    init_path = root / "output.init_ice.nc"
    dump_path = probe_root / f"round10_aevp_subcycle_{SUBCYCLE}.bin"
    for path in (frame_path, mesh_path, init_path, dump_path):
        require(path.is_file(), f"missing input: {path}")

    header, frame = oracle_gate.read_frame(frame_path)
    require(header["kt"] == FRAME_STEP, "entry-frame clock mismatch")
    require(header["storage_bits"] == np.dtype(np.float64).itemsize * 8, "frame is not fp64")
    with netCDF4.Dataset(mesh_path) as dataset:
        mesh = {name: np.asarray(dataset[name][:]) for name in MESH_FIELDS}
    with netCDF4.Dataset(init_path) as dataset:
        ocean_temperature_k = np.asarray(dataset["sst"][0]) + constants.T_freeze
    card = build_ice_rheo_card(frame, mesh, ocean_temperature_k)
    dump_header, dump = _read_dump(dump_path)
    require(
        (dump_header["jpi"], dump_header["jpj"])
        == (header["jpi"], header["jpj"]),
        "dump/frame geometry mismatch",
    )

    concentration = np.asarray(frame["a_i"][..., 0], dtype=np.float64)
    volume = np.asarray(frame["v_i"][..., 0], dtype=np.float64)
    config = card.dynamics_config
    concentration_deficit = 1.0 - concentration
    exponent = -config.strength_decay * concentration_deficit
    vector_strength = (
        config.strength_parameter_pa * volume * np.exp(exponent)
    )
    scalar_strength = (
        config.strength_parameter_pa * volume * _scalar_libm_exp(exponent)
    )
    mask = (concentration >= _SI3_ICE_PRESENCE).astype(np.float64)
    denominator = dump["delta"] + config.creep_limit_s_inv
    vector_p_over_delta = vector_strength / denominator * mask
    scalar_p_over_delta = scalar_strength / denominator * mask
    vector_row = _score(dump["p_over_delta"], vector_p_over_delta)
    scalar_row = _score(dump["p_over_delta"], scalar_p_over_delta)
    scalar_exact = scalar_row["max_abs"] == 0.0
    vector_binds = vector_row["max_abs"] != 0.0
    report = {
        "format": "nemo-si3-phase2-round10-h79-libm-probe-v1",
        "worktree": worktree_stamp(),
        "status": "BIT-EXACT" if scalar_exact and vector_binds else "DEBT",
        "execution": "CPU/fp64 host replay; NEMO operand from WRITE-only copy-run dump",
        "source": {
            "strength": "icedyn_rdgrft.F90:1048-1056; icedyn_rhg_evp.F90:420-427",
            "transcendental": (
                "scalar exp from libm.so.6, the library linked by the oracle executable"
            ),
        },
        "inputs": {
            str(path): _sha256(path) for path in (frame_path, mesh_path, init_path, dump_path)
        },
        "vector_np_exp_arm": vector_row,
        "scalar_libm_exp_arm": scalar_row,
        "predicates": {
            "vector_arm_binds": vector_binds,
            "scalar_arm_is_bit_exact": scalar_exact,
        },
    }
    return report, 0 if report["status"] == "BIT-EXACT" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--probe-root", type=Path, default=PROBE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, code = run_probe(args.root, args.probe_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

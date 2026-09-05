#!/usr/bin/env python3
"""Bit gate for the ORCA2 O1 Large & Yeager NCAR open-ocean bulk record.

The record and its semantics are handed off by Lane 4.  Frame 0 is the nine
already-mapped ``fld_read`` inputs.  Frame 1 is the NEMO ``blk_oce_1`` then
``blk_oce_2`` result.  This gate certifies the eighteen source-owned outputs;
the writer-only ``cd_du`` and ``qlwn`` slots are explicitly waived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from pathlib import Path

import jax
import jaxlib
import numpy as np
import xarray as xr

BAR = 1.0e-15
MAGIC = b"NEMO_L4_BLKIO_1 "
NX, NY = 90, 148
EXPECTED_SHA256 = "751b2d9181778e81f01bc5872d47100afc0fc3ad02c4ffcccc9613a0af2ad045"
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/runs/"
    "variant_icebergs_off_phase2g_o1canon_a_10step_np2"
)
DEFAULT_TWIN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/runs/"
    "variant_icebergs_off_phase2g_o1canon_b_10step_np2"
)
RECORD_NAME = "oracle_sbcblk_o1_kt00000001.bin"
ACCEPTED_RUNTIME = {
    "python": "3.13.0", "jax": "0.10.0", "jaxlib": "0.10.0", "numpy": "2.4.4",
}

INPUT_NAMES = (
    "wndi", "wndj", "tair", "humi", "qsr_down", "qlw_down",
    "precip_raw", "snow_raw", "slp",
)
OUTPUT_NAMES = (
    "theta_air", "q_air", "precip", "sst", "ssu", "ssv", "tsk", "ssq",
    "cd_du", "sensible", "latent", "evap", "qlwn", "qsr", "qns", "emp",
    "utau", "vtau", "taum", "wndm",
)
COVERAGE = {
    name: ("WAIVED", "writer-zeroed; inactive/source-unowned O1 slot")
    if name in {"cd_du", "qlwn"}
    else ("VERIFIED", "source-defined blk_oce_1/blk_oce_2 result")
    for name in OUTPUT_NAMES
}
SOURCES = {
    "input_preprocess": "sbcblk.F90:559-629",
    "blk_oce_1": "sbcblk.F90:720-988",
    "bulk_assembly": "sbcblk.F90:895-943",
    "blk_oce_2": "sbcblk.F90:991-1070",
    "ncar_iterations": "sbcblk_algo_ncar.F90:110-217",
    "neutral_coefficients": "sbcblk_algo_ncar.F90:244-290",
    "stability_functions": "sbcblk_algo_ncar.F90:312-363",
    "thermodynamics": "sbc_phy.F90:235-282,321-392,428-489,533-566,630-751",
    "longwave": "sbc_phy.F90:1004-1027",
}


class GateError(RuntimeError):
    """A schema, provenance, runtime, coverage, or numerical gate failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "jax": jax.__version__,
        "jaxlib": jaxlib.__version__, "numpy": np.__version__,
    }


def validate_runtime(*, require_bit_identity: bool) -> dict[str, object]:
    observed = runtime_versions()
    registered = observed == ACCEPTED_RUNTIME
    if require_bit_identity:
        require(registered, f"unregistered numeric runtime: {observed}; expected {ACCEPTED_RUNTIME}")
    return {
        "observed": observed,
        "bit_identity_reference": ACCEPTED_RUNTIME,
        "bit_identity_claim_valid": registered,
        "statement": "bit-exactness claims valid only under the registered Python/JAX/jaxlib/NumPy stack",
    }


def _read_frame(stream, expected_stage: int, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    require(stream.read(16) == MAGIC, f"O1 stage {expected_stage}: bad magic")
    raw_header = stream.read(32)
    require(len(raw_header) == 32, f"O1 stage {expected_stage}: short header")
    header = struct.unpack("=8i", raw_header)
    require(
        header == (1, 1, expected_stage, NX, NY, len(names), 0, 64),
        f"O1 stage {expected_stage}: invalid header {header}",
    )
    count = NX * NY
    result = {}
    for name in names:
        values = np.fromfile(stream, dtype=np.float64, count=count)
        require(values.size == count, f"O1 stage {expected_stage}.{name}: short payload")
        result[name] = values.reshape((NX, NY), order="F").T
    return result


def read_record(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    require(path.is_file(), f"missing O1 stream: {path}")
    require(sha256(path) == EXPECTED_SHA256, f"unregistered O1 sha256: {path}")
    with path.open("rb") as stream:
        inputs = _read_frame(stream, 0, INPUT_NAMES)
        outputs = _read_frame(stream, 1, OUTPUT_NAMES)
        require(stream.read(1) == b"", "unregistered trailing O1 bytes")
    return inputs, outputs


def read_wet_mask(path: Path) -> np.ndarray:
    require(path.is_file(), f"missing rank-0 mesh mask: {path}")
    with xr.open_dataset(path) as dataset:
        mask = np.asarray(dataset["tmask"].isel(time_counter=0, nav_lev=0), dtype=bool)
    require(mask.shape == (NY, NX), f"unexpected rank-0 tmask shape {mask.shape}")
    require(int(np.count_nonzero(mask)) == 8794, "rank-0 wet-cell register changed")
    return mask


def _row_scale_ulp(candidate: np.ndarray, oracle: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(candidate))), float(np.max(np.abs(oracle))), 1.0)
    return float(np.spacing(np.float64(scale)))


def score(candidate: np.ndarray, oracle: np.ndarray, field: str) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(candidate.shape == oracle.shape and candidate.ndim == 1, f"{field}: score shape")
    require(np.all(np.isfinite(candidate)) and np.all(np.isfinite(oracle)), f"{field}: nonfinite")
    absolute = np.abs(candidate - oracle)
    normalized = absolute / np.maximum(np.abs(oracle), 1.0)
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    over = normalized > BAR
    ulp = _row_scale_ulp(candidate, oracle)
    nz = np.abs(oracle) > 0.0
    relative = np.zeros_like(absolute)
    relative[nz] = absolute[nz] / np.abs(oracle[nz])
    return {
        "field": field,
        "bit_unequal_over_n": f"{int(np.count_nonzero(unequal))} / {candidate.size}",
        "non_bit_identical_count": int(np.count_nonzero(unequal)),
        "n": int(candidate.size),
        "over_bar_count": int(np.count_nonzero(over)),
        "max_normalized_error": float(np.max(normalized, initial=0.0)),
        "row_scale_ulp": ulp,
        "max_row_scale_ulp_error": float(np.max(absolute, initial=0.0) / ulp),
        "max_relative_error_nonzero_oracle": float(np.max(relative, initial=0.0)),
        "first_non_bit_index": int(np.flatnonzero(unequal)[0]) if np.any(unequal) else None,
        "first_over_bar_index": int(np.flatnonzero(over)[0]) if np.any(over) else None,
        "verdict": "AT_BAR" if not np.any(over) else "DEBT",
        "bit_verdict": "BIT_IDENTICAL" if not np.any(unequal) else "NON_BIT_IDENTICAL",
    }


def validate_coverage(coverage: dict[str, tuple[str, str]] = COVERAGE) -> None:
    require(set(coverage) == set(OUTPUT_NAMES), "incomplete O1 output coverage register")
    for name, (status, reason) in coverage.items():
        require(status in {"VERIFIED", "WAIVED"} and bool(reason), f"bad coverage row {name}")


def _predict(inputs: dict[str, np.ndarray], oracle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from legoesm.core.bulk_flux import nemo_ncar_ocean_bulk
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", f"CPU-only gate selected {jax.default_backend()}")
    compiled = jax.jit(nemo_ncar_ocean_bulk, static_argnames=("iterations",))
    result = compiled(
        inputs["wndi"], inputs["wndj"], inputs["tair"], inputs["humi"],
        oracle["sst"], inputs["slp"], inputs["qsr_down"], inputs["qlw_down"],
        inputs["precip_raw"], inputs["snow_raw"],
    )
    return {name: np.asarray(value) for name, value in result.items()}


def evaluate(
    root: Path = DEFAULT_ROOT,
    *,
    twin_root: Path = DEFAULT_TWIN,
    plant_field: str | None = None,
    require_bit_identity: bool = True,
) -> dict[str, object]:
    runtime = validate_runtime(require_bit_identity=require_bit_identity)
    validate_coverage()
    record = root / RECORD_NAME
    twin = twin_root / RECORD_NAME
    require(sha256(twin) == EXPECTED_SHA256, "O1 canonical twin hash differs")
    inputs, oracle = read_record(record)
    wet = read_wet_mask(root / "mesh_mask_0000.nc")
    predicted = _predict(inputs, oracle)
    verified = [name for name, (status, _) in COVERAGE.items() if status == "VERIFIED"]
    require(set(verified) <= set(predicted), "implementation omitted a verified O1 output")

    rows = []
    plants = []
    for name in verified:
        candidate = np.asarray(predicted[name][wet], dtype=np.float64).copy()
        target = np.asarray(oracle[name][wet], dtype=np.float64)
        if plant_field == name:
            candidate[0] = np.nextafter(candidate[0], np.float64(np.inf))
        row = score(candidate, target, name)
        rows.append(row)
        planted = np.asarray(predicted[name][wet], dtype=np.float64).copy()
        planted[0] = np.nextafter(planted[0], np.float64(np.inf))
        plant_row = score(planted, target, name)
        require(plant_row["non_bit_identical_count"] > 0, f"{name}: inert row plant")
        plants.append({"field": name, "status": "PASS_NONZERO", "exit_code": 1})

    non_bit = sum(int(row["non_bit_identical_count"]) for row in rows)
    over_bar = sum(int(row["over_bar_count"]) for row in rows)
    if plant_field is not None:
        require(plant_field in verified, f"plant field is not VERIFIED: {plant_field}")
    return {
        "gate": "ORCA2_O1_NCAR_BULK",
        "status": "AT_BAR" if over_bar == 0 else "DEBT",
        "bit_status": "BIT_IDENTICAL" if non_bit == 0 else "NON_BIT_IDENTICAL",
        "bar": BAR,
        "bit_unequal_over_n": f"{non_bit} / {len(verified) * int(np.count_nonzero(wet))}",
        "over_bar_over_n": f"{over_bar} / {len(verified) * int(np.count_nonzero(wet))}",
        "first_over_bar": next((row for row in rows if row["over_bar_count"]), None),
        "first_non_bit_operand": next((row for row in rows if row["non_bit_identical_count"]), None),
        "rows": rows,
        "coverage": {name: {"status": status, "reason": reason} for name, (status, reason) in COVERAGE.items()},
        "plants": plants,
        "plant_field": plant_field,
        "numeric_runtime": runtime,
        "execution": {"backend": jax.default_backend(), "dtype": "float64", "jit": True, "transcendentals": "libm"},
        "oracle": {
            "path": str(record), "sha256": sha256(record), "bytes": record.stat().st_size,
            "twin_path": str(twin), "twin_sha256": sha256(twin),
            "frame_semantics": {"0": list(INPUT_NAMES), "1": list(OUTPUT_NAMES)},
        },
        "sources": SOURCES,
        "handoff_before": {
            "label": "ORCA2 Lane-4 pre-round-18 diagnostic",
            "theta_air": "117 / 8794", "ssq": "3964 / 8794",
            "utau": "8794 / 8794", "vtau": "8794 / 8794",
            "sensible": "8794 / 8794", "latent": "8794 / 8794", "evap": "8794 / 8794",
        },
        "interpretation": (
            "All source-owned O1 rows are certified only when status is AT_BAR and "
            "bit_status is BIT_IDENTICAL; cd_du and qlwn remain waived."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--twin-root", type=Path, default=DEFAULT_TWIN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-field", choices=tuple(COVERAGE))
    parser.add_argument("--allow-unregistered-runtime", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(
            args.root, twin_root=args.twin_root, plant_field=args.plant_field,
            require_bit_identity=not args.allow_unregistered_runtime,
        )
    except GateError as exc:
        print(json.dumps({"gate": "ORCA2_O1_NCAR_BULK", "status": "ERROR", "error": str(exc)}, indent=2))
        return 2
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    return 0 if result["bit_status"] == "BIT_IDENTICAL" and args.plant_field is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

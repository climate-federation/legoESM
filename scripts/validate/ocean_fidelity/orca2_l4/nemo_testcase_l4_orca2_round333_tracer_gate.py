#!/usr/bin/env python3
"""Admit the passive, raw kt=2 stage-3 tracer source-order records."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round331_tke_gate as r331,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round332_source_order_gate as r332,
)

NX, NY, NZ, HALO = 94, 152, 31, 2
MAGIC = "NEMO_L4_R333TR"
HEADER_NAMES = (
    "version", "kt", "stage", "Kbb", "Kmm", "Krhs", "Kaa", "rank",
    "nx", "ny", "nz", "real_bits", "n_fields",
)
FIELDS = (
    "zero_T", "zero_S",
    "after_advection_T", "after_advection_S",
    "after_sbc_T", "after_sbc_S",
    "after_qsr_T", "after_qsr_S",
    "after_ldf_T", "after_ldf_S",
    "after_damping_T", "after_damping_S",
    "Kbb_T", "Kbb_S", "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
    "r3t_Kbb", "r3t_Kmm", "r3t_Kaa",
)
TWO_D_FIELDS = frozenset(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa"))
PLANTS = (
    "header", "truncation", "checksum", "stamp", "zero", "kmm",
    "time-level",
)
ZDF_REFERENCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/"
    "acquisition/orca1ice_zdfphy_kt1_np2"
)

GateError = r332.GateError
require = r332.require


def record_name(rank: int) -> str:
    return f"oracle_rktracer_stage3_kt00000002_r{rank:04d}.bin"


def rk3_levels_for_step_stage(kt: int, stage: int) -> tuple[int, int, int, int]:
    """Derive Kbb/Kmm/Krhs/Kaa from NEMO initialization and swaps.

    ``nemogcm.F90:396`` initializes Nbb/Nnn/Naa/Nrhs to 1/2/3/3.
    ``stprk3.F90:195-213`` supplies the stage arguments, swaps Nnn/Naa after
    stages 1 and 2, then swaps Nbb/Naa after every completed step.
    """

    require(kt >= 1, f"invalid RK3 step {kt}")
    require(stage in (1, 2, 3), f"invalid RK3 stage {stage}")
    nbb, nnn, naa, nrhs = 1, 2, 3, 3
    for step in range(1, kt + 1):
        levels = {
            1: (nbb, nbb, nrhs, naa),
            2: None,
            3: None,
        }
        if step == kt and stage == 1:
            return levels[1]
        nrhs, nnn, naa = nnn, naa, nnn
        levels[2] = (nbb, nnn, nrhs, naa)
        if step == kt and stage == 2:
            return levels[2]
        nrhs, nnn, naa = nnn, naa, nnn
        levels[3] = (nbb, nnn, nrhs, naa)
        if step == kt and stage == 3:
            return levels[3]
        nrhs, nbb, naa = nbb, naa, nbb
    raise AssertionError("unreachable")


def _take(stream: io.BytesIO, count: int, label: str) -> bytes:
    data = stream.read(count)
    require(len(data) == count, f"truncated {label}")
    return data


def read_stage3(
    path: Path,
    *,
    plant: str | None = None,
    expected_shape: tuple[int, int, int] = (NX, NY, NZ),
) -> dict[str, object]:
    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    stream = io.BytesIO(raw)
    magic = _take(stream, 16, "magic").decode("ascii").rstrip()
    values = struct.unpack(
        f"={len(HEADER_NAMES)}i",
        _take(stream, 4 * len(HEADER_NAMES), "header"),
    )
    header = dict(zip(HEADER_NAMES, values, strict=True))
    if plant == "header":
        header["stage"] = 2
    elif plant == "time-level":
        header["Kaa"] = 2 if header["Kaa"] != 2 else 3
    require(magic == MAGIC, f"bad magic {magic!r}")
    require(
        (header["version"], header["kt"], header["stage"], header["real_bits"])
        == (2, 2, 3, 64),
        "wrong version/step/stage/precision",
    )
    expected_levels = rk3_levels_for_step_stage(header["kt"], header["stage"])
    observed_levels = tuple(header[name] for name in ("Kbb", "Kmm", "Krhs", "Kaa"))
    require(
        observed_levels == expected_levels,
        f"wrong RK3 levels {observed_levels}; expected {expected_levels}",
    )
    require(header["rank"] in (0, 1), f"bad rank {header['rank']}")
    require((header["nx"], header["ny"], header["nz"]) == expected_shape,
            "record lacks exact jpi,jpj,jpk extents")
    require(header["n_fields"] == len(FIELDS), "wrong field count")

    fields: dict[str, np.ndarray] = {}
    checksums: dict[str, int] = {}
    order: list[str] = []
    for index in range(header["n_fields"]):
        name = _take(stream, 24, f"field {index} name").decode("ascii").rstrip()
        array_rank, n1, n2, n3, i0, j0, k0 = struct.unpack(
            "=7i", _take(stream, 28, f"{name} metadata")
        )
        expected_rank = 2 if name in TWO_D_FIELDS else 3
        field_shape = (
            expected_shape[0], expected_shape[1],
            1 if expected_rank == 2 else expected_shape[2],
        )
        require(
            (array_rank, n1, n2, n3, i0, j0, k0)
            == (expected_rank, *field_shape, 1, 1, 1),
            f"{name}: wrong rank, extent, or origin",
        )
        checksum, = struct.unpack("=Q", _take(stream, 8, f"{name} checksum"))
        payload = _take(stream, n1 * n2 * n3 * 8, f"{name} payload")
        array = np.frombuffer(payload, dtype="=f8").copy().reshape(
            field_shape, order="F"
        )
        if expected_rank == 2:
            array = array[:, :, 0]
        require(bool(np.isfinite(array).all()), f"{name}: non-finite payload")
        require(name not in fields, f"duplicate field {name}")
        fields[name], checksums[name] = array, checksum
        order.append(name)
    require(stream.read(1) == b"", "trailing bytes")
    require(tuple(order) == FIELDS, "field order differs from source order")
    return {
        "header": header,
        "fields": fields,
        "checksums": checksums,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _owned_xyz(values: np.ndarray) -> np.ndarray:
    return values[HALO:-HALO, HALO:-HALO].transpose(1, 0, 2)[..., :30]


def _comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    require(reference.shape == candidate.shape, "comparison shape moved")
    reference = np.ascontiguousarray(reference)
    candidate = np.ascontiguousarray(candidate)
    unequal = reference.view(np.uint64) != candidate.view(np.uint64)
    delta = candidate - reference
    return {
        "unequal": int(np.count_nonzero(unequal)),
        "count": int(reference.size),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
    }


def _full_wet_mask(root: Path, rank: int) -> np.ndarray:
    prior = r331.read_record(root / r331.record_name(rank))
    interior = prior["fields"]["tke_wmask"] != 0.0
    wet = np.zeros((NX, NY, NZ), dtype=bool)
    wet[HALO:-HALO, HALO:-HALO, : interior.shape[2]] = interior
    require(bool(np.any(wet)), f"rank {rank}: no wet tracer cells")
    return wet


def _signed_zero_census(record: dict[str, object]) -> dict[str, dict[str, int]]:
    result = {}
    for name, values in record["fields"].items():
        zeros = values == 0.0
        result[name] = {
            "negative_zero": int(np.count_nonzero(zeros & np.signbit(values))),
            "positive_zero": int(np.count_nonzero(zeros & ~np.signbit(values))),
        }
    return result


def run_gate(
    root: Path,
    *,
    expect_commit: str,
    zdf_reference: Path = ZDF_REFERENCE,
    plant: str | None = None,
) -> dict[str, object]:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    produced = (root / "producer_commit.txt").read_text(encoding="ascii").strip()
    if plant == "stamp":
        produced = "0" * 40
    require(produced == expect_commit, f"producer commit {produced} != {expect_commit}")
    inherited = r332.run_gate(root, zdf_reference, expect_commit)
    require(
        inherited["status"] == "PASS_R332_SOURCE_ORDER_ADMISSION",
        "round-332 inherited admission moved",
    )

    records = {}
    wet_masks = {}
    for rank in (0, 1):
        parser_plant = plant if rank == 0 and plant in (
            "header", "truncation", "time-level"
        ) else None
        record = read_stage3(root / record_name(rank), plant=parser_plant)
        require(record["header"]["rank"] == rank, "rank tag moved")
        wet = _full_wet_mask(root, rank)
        if rank == 0 and plant == "checksum":
            index = tuple(np.argwhere(wet)[0])
            record["fields"]["after_advection_T"][index] = np.nextafter(
                record["fields"]["after_advection_T"][index], np.inf
            )
        r331._verify_checksums(record)
        if rank == 0 and plant == "zero":
            index = tuple(np.argwhere(wet)[0])
            record["fields"]["zero_T"][index] = np.nextafter(0.0, 1.0)
        if rank == 0 and plant == "kmm":
            index = tuple(np.argwhere(wet)[0])
            value = record["fields"]["Kmm_T"][index]
            record["fields"]["Kmm_T"][index] = np.nextafter(value, np.inf)
        records[rank], wet_masks[rank] = record, wet

    for rank, record in records.items():
        for tracer in ("T", "S"):
            zeros = record["fields"][f"zero_{tracer}"]
            require(
                not bool(np.any(zeros[wet_masks[rank]] != 0.0)),
                f"rank {rank}: stage-3 {tracer} accumulator is not zero on wet cells",
            )

    stage2 = rung10.read_state_frame(
        root / "oracle_stage_kt00000002_s2.bin", kt=2, stage=2
    )
    stage3 = rung10.read_state_frame(
        root / "oracle_stage_kt00000002_s3.bin", kt=2, stage=3
    )
    rank0 = records[0]["fields"]
    kmm = {
        tracer: _comparison(stage2[tracer], _owned_xyz(rank0[f"Kmm_{tracer}"]))
        for tracer in ("T", "S")
    }
    require(
        all(row["unequal"] == 0 for row in kmm.values()),
        "recorded rank-0 Kmm does not align bitwise with kt=2 stage 2",
    )
    pre_fold = {
        tracer: _comparison(_owned_xyz(rank0[f"Kaa_{tracer}"]), stage3[tracer])
        for tracer in ("T", "S")
    }
    activity = {}
    for rank, record in records.items():
        for tracer in ("T", "S"):
            previous = record["fields"][f"zero_{tracer}"]
            for boundary in (
                "after_advection", "after_sbc", "after_qsr", "after_ldf",
                "after_damping",
            ):
                current = record["fields"][f"{boundary}_{tracer}"]
                activity[f"rank{rank}_{boundary}_{tracer}"] = _comparison(
                    previous, current
                )
                previous = current
    for rank in (0, 1):
        for tracer in ("T", "S"):
            require(
                activity[f"rank{rank}_after_advection_{tracer}"]["unequal"] > 0,
                f"rank {rank}: stage-3 {tracer} advection is inactive",
            )
        require(
            activity[f"rank{rank}_after_qsr_T"]["unequal"] > 0,
            f"rank {rank}: stage-3 solar source is inactive",
        )

    return {
        "format": "nemo-testcase-l4-orca2-round333-tracer-acquisition-v2",
        "status": "PASS_R333_TRACER_ACQUISITION",
        "producer_commit": produced,
        "round332_status": inherited["status"],
        "records": {record_name(rank): records[rank]["sha256"] for rank in (0, 1)},
        "recorded_boundaries": list(FIELDS),
        "kmm_vs_stage2_rank0": kmm,
        "pre_fold_kaa_vs_stage3_rank0": pre_fold,
        "boundary_activity": activity,
        "signed_zero_census": {
            str(rank): _signed_zero_census(record) for rank, record in records.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--zdf-reference", type=Path, default=ZDF_REFERENCE)
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = run_gate(
            args.record_dir,
            expect_commit=args.expect_commit,
            zdf_reference=args.zdf_reference,
            plant=args.plant,
        )
        require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, OSError, UnicodeError, ValueError, struct.error) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant else 'REFUSE'}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

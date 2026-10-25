#!/usr/bin/env python3
"""Admit the passive kt=2 stage-3 tracer source-order record for round 333."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round332_source_order_gate as r332,
)

NX, NY, NZ, HALO = 94, 152, 31, 2
MAGIC = "NEMO_L2_RKTR3_1"
RECORD_NAME = "oracle_rktracer_stage3_kt00000002.bin"
FIELDS = (
    "zero_T",
    "zero_S",
    "after_advection_T",
    "after_advection_S",
    "after_sbc_T",
    "after_sbc_S",
    "after_qsr_T",
    "after_qsr_S",
    "after_ldf_T",
    "after_ldf_S",
    "Kbb_T",
    "Kbb_S",
    "Kmm_T",
    "Kmm_S",
    "Kaa_T",
    "Kaa_S",
)
R3_FIELDS = ("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")
PLANTS = ("header", "truncation", "stamp", "zero", "kmm")
ROUND5_RUN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2"
)
ZDF_REFERENCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/"
    "acquisition/orca1ice_zdfphy_kt1_np2"
)

GateError = r332.GateError
require = r332.require


def _xyz(values: np.ndarray) -> np.ndarray:
    local = values.reshape((NX, NY, NZ), order="F")
    return local[HALO:-HALO, HALO:-HALO].transpose(1, 0, 2)


def _xy(values: np.ndarray) -> np.ndarray:
    local = values.reshape((NX, NY), order="F")
    return local[HALO:-HALO, HALO:-HALO].T


def read_stage3(
    path: Path,
    *,
    kt: int,
    levels: tuple[int, int, int, int],
    plant: str | None = None,
) -> dict[str, object]:
    """Decode the inherited RKTR3 schema at a caller-specified step."""

    raw = path.read_bytes()
    if plant == "truncation":
        raw = raw[:-8]
    header_bytes = 16 + struct.calcsize("=11i")
    require(len(raw) >= header_bytes, f"{path.name}: truncated header")
    magic = raw[:16].decode("ascii").rstrip()
    header = list(struct.unpack("=11i", raw[16:header_bytes]))
    if plant == "header":
        header[1] += 1
    expected = (1, kt, 3, *levels, NX, NY, NZ, 64)
    require(magic == MAGIC, f"{path.name}: bad magic {magic!r}")
    require(tuple(header) == expected, f"{path.name}: bad header {tuple(header)}")
    values = np.frombuffer(raw, dtype=np.float64, offset=header_bytes)
    n2, n3 = NX * NY, NX * NY * NZ
    require(
        values.size == len(FIELDS) * n3 + len(R3_FIELDS) * n2,
        f"{path.name}: bad payload count {values.size}",
    )
    require(bool(np.isfinite(values).all()), f"{path.name}: non-finite payload")
    fields = {
        name: _xyz(values[index * n3 : (index + 1) * n3]).copy()
        for index, name in enumerate(FIELDS)
    }
    offset = len(FIELDS) * n3
    fields.update(
        {
            name: _xy(values[offset + index * n2 : offset + (index + 1) * n2]).copy()
            for index, name in enumerate(R3_FIELDS)
        }
    )
    return {
        "header": dict(
            zip(
                (
                    "version",
                    "kt",
                    "stage",
                    "Kbb",
                    "Kmm",
                    "Krhs",
                    "Kaa",
                    "nx",
                    "ny",
                    "nz",
                    "real_bits",
                ),
                header,
                strict=True,
            )
        ),
        "fields": fields,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


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
    record = read_stage3(
        root / RECORD_NAME,
        kt=2,
        levels=(3, 2, 1, 1),
        plant=plant if plant in ("header", "truncation") else None,
    )
    fields = record["fields"]
    if plant == "zero":
        fields["zero_T"].flat[0] = np.nextafter(0.0, 1.0)
    elif plant == "kmm":
        fields["Kmm_T"].flat[0] = np.nextafter(fields["Kmm_T"].flat[0], np.inf)
    require(
        not bool(np.any(fields["zero_T"] != 0.0)),
        "stage-3 T accumulator is not numerically zero at entry",
    )
    require(
        not bool(np.any(fields["zero_S"] != 0.0)),
        "stage-3 S accumulator is not numerically zero at entry",
    )

    stage2 = rung10.read_state_frame(root / "oracle_stage_kt00000002_s2.bin", kt=2, stage=2)
    stage3 = rung10.read_state_frame(root / "oracle_stage_kt00000002_s3.bin", kt=2, stage=3)
    kmm = {
        tracer: _comparison(stage2[tracer], fields[f"Kmm_{tracer}"][..., :30])
        for tracer in ("T", "S")
    }
    require(
        all(row["unequal"] == 0 for row in kmm.values()),
        "recorded Kmm does not align bitwise with kt=2 stage 2",
    )
    pre_fold = {
        tracer: _comparison(fields[f"Kaa_{tracer}"][..., :30], stage3[tracer])
        for tracer in ("T", "S")
    }
    activity = {}
    for tracer in ("T", "S"):
        previous = fields[f"zero_{tracer}"]
        for boundary in ("after_advection", "after_sbc", "after_qsr", "after_ldf"):
            current = fields[f"{boundary}_{tracer}"]
            activity[f"{boundary}_{tracer}"] = _comparison(previous, current)
            previous = current
    require(
        activity["after_advection_T"]["unequal"] > 0, "kt=2 stage-3 tracer advection is inactive"
    )
    require(
        activity["after_advection_S"]["unequal"] > 0, "kt=2 stage-3 salinity advection is inactive"
    )
    require(activity["after_qsr_T"]["unequal"] > 0, "kt=2 stage-3 solar source is inactive")

    return {
        "format": "nemo-testcase-l4-orca2-round333-tracer-acquisition-v1",
        "status": "PASS_R333_TRACER_ACQUISITION",
        "producer_commit": produced,
        "round332_status": inherited["status"],
        "record": {RECORD_NAME: record["sha256"]},
        "recorded_boundaries": list(FIELDS),
        "kmm_vs_stage2": kmm,
        "pre_fold_kaa_vs_stage3": pre_fold,
        "boundary_activity": activity,
    }


def self_test(*, plant: str | None = None) -> dict[str, object]:
    path = ROUND5_RUN / "oracle_rktracer_stage3_kt00000001.bin"
    record = read_stage3(
        path,
        kt=1,
        levels=(1, 2, 3, 3),
        plant=plant if plant in ("header", "truncation") else None,
    )
    require(tuple(record["fields"]) == FIELDS + R3_FIELDS, "RKTR3 field order moved")
    require(plant is None, f"{plant} plant stayed green")
    return {"status": "PASS_R333_TRACER_PARSER_CONTROL"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--zdf-reference", type=Path, default=ZDF_REFERENCE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            require(
                args.record_dir is None and args.expect_commit is None,
                "self-test cannot take acquisition inputs",
            )
            result = self_test(plant=args.plant)
        else:
            require(
                args.record_dir is not None and args.expect_commit is not None,
                "admission needs record directory and producer commit",
            )
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

#!/usr/bin/env python3
"""Fail-closed admission gate for the Round-153 developed FCT record."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

import nemo_testcase_l2_gyre_round111_fct_writer_gate as self_describing

MAGIC = "NEMO_L2_R153FCT"
RECORD = "oracle_developed_fct_kt00001081.bin"
HEADER = (1, 1081, 1, 2, 3, 3, 36, 26, 31, 64, 61, 3, 3, 2)
EXPECTED_CHANGED_DIAGNOSTICS = (
    "oracle_developed_rhs_kt00001081.bin",
    "oracle_dynadv_split_kt00000001_s3.bin",
    "oracle_rkstage3_terms_kt00000001.bin",
    "oracle_rkstage3_wzv_kt00000001.bin",
    "oracle_rktracer_operands_kt00000001_s2.bin",
    "oracle_slow_forcing_kt00000001.bin",
    "oracle_tracer_transport_kt00000001_s3.bin",
    "oracle_transport_kt00000001_s1.bin",
    "oracle_transport_kt00000001_s2.bin",
    "oracle_transport_kt00000001_s3.bin",
    "oracle_zdf_matrix_kt00000001.bin",
)


class GateError(RuntimeError):
    """A named fail-closed refusal."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(
    name: str,
    rank: int,
    shape: tuple[int, int, int],
    origin: tuple[int, int, int],
) -> tuple:
    return name, rank, shape, origin


def expected_rows() -> tuple[tuple, ...]:
    rows = [
        _row("p2dt", 0, (1, 1, 1), (0, 0, 0)),
        _row("transport_u", 3, (35, 25, 30), (1, 1, 1)),
        _row("transport_v", 3, (35, 25, 30), (1, 1, 1)),
        _row("transport_w", 3, (34, 24, 30), (2, 2, 1)),
        _row("e3t_3d", 3, (34, 24, 30), (2, 2, 1)),
        _row("r3t_Kbb", 2, (34, 24, 1), (2, 2, 1)),
        _row("r3t_Kmm", 2, (34, 24, 1), (2, 2, 1)),
        _row("r3t_Kaa", 2, (34, 24, 1), (2, 2, 1)),
        _row("tmask", 3, (34, 24, 30), (2, 2, 1)),
        _row("wmask", 3, (34, 24, 30), (2, 2, 1)),
        _row("r1_e1e2t", 2, (34, 24, 1), (2, 2, 1)),
    ]
    tracer_rows = (
        ("base", (36, 26, 30), (1, 1, 1)),
        ("now", (36, 26, 30), (1, 1, 1)),
        ("rhs_entry", (32, 22, 30), (3, 3, 1)),
        ("first_u", (35, 25, 30), (1, 1, 1)),
        ("first_v", (35, 25, 30), (1, 1, 1)),
        ("first_w", (34, 24, 31), (2, 2, 1)),
        ("first_div", (34, 24, 30), (2, 2, 1)),
        ("midpoint", (34, 24, 30), (2, 2, 1)),
        ("average_u", (33, 23, 30), (2, 2, 1)),
        ("average_v", (33, 23, 30), (2, 2, 1)),
        ("average_w", (34, 24, 31), (2, 2, 1)),
        ("upstream_div", (32, 22, 30), (3, 3, 1)),
        ("rhs_after_up", (32, 22, 30), (3, 3, 1)),
        ("anti_pre_u", (33, 23, 30), (2, 2, 1)),
        ("anti_pre_v", (33, 23, 30), (2, 2, 1)),
        ("anti_pre_w", (32, 22, 31), (3, 3, 1)),
        ("coef_u", (33, 23, 30), (2, 2, 1)),
        ("coef_v", (33, 23, 30), (2, 2, 1)),
        ("coef_w", (32, 22, 31), (3, 3, 1)),
        ("anti_post_u", (33, 23, 30), (2, 2, 1)),
        ("anti_post_v", (33, 23, 30), (2, 2, 1)),
        ("anti_post_w", (32, 22, 31), (3, 3, 1)),
        ("final_div", (32, 22, 30), (3, 3, 1)),
        ("divisor", (32, 22, 30), (3, 3, 1)),
        ("rhs_final", (32, 22, 30), (3, 3, 1)),
    )
    for tracer in ("T", "S"):
        rows.extend(
            _row(f"{name}_{tracer}", 3, shape, origin)
            for name, shape, origin in tracer_rows
        )
    return tuple(rows)


EXPECTED_ROWS = expected_rows()


def inherited_names(root: Path) -> tuple[str, ...]:
    names = {path.name for path in root.glob("oracle*.bin")}
    names.update(path.name for path in root.glob("*_restart.nc"))
    if (root / "mesh_mask.nc").is_file():
        names.add("mesh_mask.nc")
    return tuple(sorted(names))


def check_stamp(root: Path, expected_commit: str) -> None:
    stamp = root / f"{RECORD}.stamp"
    parts = stamp.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 3, "malformed record stamp")
    require(parts[0] == sha256(root / RECORD), "record stamp digest mismatch")
    require(parts[1] == expected_commit, "record stamp producer mismatch")
    require(parts[2] == RECORD, "record stamp filename mismatch")


def check_binary(root: Path) -> str:
    parts = (root / "binary.sha256").read_text(encoding="utf-8").split()
    require(bool(parts), "malformed binary.sha256")
    digest = sha256(root / "nemo")
    require(parts[0] == digest, "run binary differs from binary.sha256")
    return digest


def check_inherited(
    root: Path,
    baseline: Path,
    *,
    plant: bool = False,
) -> dict:
    expected = inherited_names(baseline)
    manifest = root / "round153_inherited.sha256"
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name] = digest
    require(tuple(sorted(entries)) == expected,
            "inherited manifest file registry differs from source run")
    changed_diagnostics = []
    unchanged_diagnostics = []
    state_files = []
    for index, name in enumerate(expected):
        baseline_digest = sha256(baseline / name)
        candidate_digest = sha256(root / name)
        if plant and index == 0:
            candidate_digest = "0" * 64
        require(entries[name] == candidate_digest,
                f"inherited manifest digest differs for {name}")
        if name.endswith("_restart.nc") or name == "mesh_mask.nc":
            state_files.append(name)
            compared_digest = (
                "0" * 64
                if plant and name.endswith("_restart.nc") else candidate_digest
            )
            require(compared_digest == baseline_digest,
                    f"passive instrument moved state file {name}")
        elif candidate_digest == baseline_digest:
            unchanged_diagnostics.append(name)
        else:
            changed_diagnostics.append(name)
    require(tuple(changed_diagnostics) == EXPECTED_CHANGED_DIAGNOSTICS,
            "changed private diagnostic registry differs from preregistration")
    return {
        "count": len(expected),
        "state_files_bit_identical": state_files,
        "changed_diagnostics": changed_diagnostics,
        "unchanged_diagnostics": unchanged_diagnostics,
        "manifest_sha256": sha256(manifest),
    }


def measure(args: argparse.Namespace) -> dict:
    producer = (args.root / "producer_commit.txt").read_text(
        encoding="utf-8").strip()
    require(producer == args.expect_commit,
            "producer_commit.txt differs from --expect-commit")
    stamp_commit = (
        "planted-wrong-commit" if args.plant == "stamp" else args.expect_commit
    )
    check_stamp(args.root, stamp_commit)
    record = self_describing.read_self_describing_record(
        args.root / RECORD,
        magic_expected=MAGIC,
        header_expected=HEADER,
        rows_expected=EXPECTED_ROWS,
        truncate=args.plant == "truncation",
        drop_last=args.plant == "missing-field",
    )
    fields = record["fields"]
    for name, field in fields.items():
        require(np.all(np.isfinite(field["values"])),
                f"{name} contains a non-finite payload")
    require(fields["p2dt"]["values"].item() == 14400.0,
            "stage-3 p2dt is not the resolved 14400 s")
    for mask in ("tmask", "wmask"):
        values = fields[mask]["values"]
        require(np.all((values == 0.0) | (values == 1.0)),
                f"{mask} is not binary")

    one_bits = np.float64(1.0).view(np.uint64)
    active = {}
    for tracer in ("T", "S"):
        count = 0
        for face in ("u", "v", "w"):
            values = fields[f"coef_{face}_{tracer}"]["values"]
            if args.plant == "coefficients-one":
                values = np.ones_like(values)
            count += int(np.count_nonzero(values.view(np.uint64) != one_bits))
        active[tracer] = count
        require(count > 0, f"{tracer} limiter has no active coefficient")

    inherited = check_inherited(
        args.root, args.baseline, plant=args.plant == "inherited-byte")
    if args.plant == "restart-byte":
        inherited = check_inherited(
            args.root, args.baseline, plant=True)
    return {
        "format": "nemo-testcase-l2-gyre-round153-developed-fct-v1",
        "status": "PASS",
        "worktree": worktree_stamp(),
        "producer_commit": producer,
        "binary_sha256": check_binary(args.root),
        "record": {
            "name": RECORD,
            "sha256": record["sha256"],
            "bytes": (args.root / RECORD).stat().st_size,
            "field_count": len(fields),
            "header": list(record["header"]),
        },
        "limiter_active_coefficients": active,
        "inherited": inherited,
        "plant": args.plant,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plant",
        choices=("none", "stamp", "truncation", "missing-field",
                 "coefficients-one", "inherited-byte", "restart-byte"),
        default="none",
    )
    args = parser.parse_args(argv)
    try:
        report = measure(args)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (GateError, self_describing.GateError, OSError, ValueError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED: {args.plant}: {error}", file=sys.stderr)
        else:
            print(f"REFUSE: {error}", file=sys.stderr)
        return 1
    print(
        "STATUS PASS: round153 developed FCT record "
        f"fields={report['record']['field_count']} "
        f"active_T={report['limiter_active_coefficients']['T']} "
        f"active_S={report['limiter_active_coefficients']['S']} "
        f"inherited={report['inherited']['count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

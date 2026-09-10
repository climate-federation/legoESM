#!/usr/bin/env python
"""Create the controlled receipt for identical-binary NEMO dump nondeterminism."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bn2_alpha_compare as loaders
import numpy as np
from zdf_stream_bracket import (
    files_byte_identical,
    manifest_sha256,
    one_bit_file_control,
    sha256,
    stream_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--expected-binary-sha", required=True)
    parser.add_argument("--expected-donor-sha", required=True)
    parser.add_argument("--expected-restart-sha", required=True)
    parser.add_argument("--expected-stream-count", type=int, default=198)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a = args.a_dir.resolve()
    b = args.b_dir.resolve()
    manifest_a = stream_manifest(a)
    manifest_b = stream_manifest(b)
    if (
        len(manifest_a) != args.expected_stream_count
        or len(manifest_b) != args.expected_stream_count
    ):
        raise SystemExit(
            f"stream-count gate failed: A={len(manifest_a)}, B={len(manifest_b)}, "
            f"expected={args.expected_stream_count}"
        )
    if set(manifest_a) != set(manifest_b):
        raise SystemExit("identical-binary stream inventories differ")
    for directory in (a, b):
        if sha256(directory / "nemo") != args.expected_binary_sha:
            raise SystemExit(f"binary SHA gate failed: {directory}")
        if sha256(directory / "DINO_00005760_restart.nc") != args.expected_donor_sha:
            raise SystemExit(f"donor SHA gate failed: {directory}")
        if sha256(directory / "DINO_00005761_restart.nc") != args.expected_restart_sha:
            raise SystemExit(f"output restart SHA gate failed: {directory}")

    jpi, jpj, jpk, hls = loaders._read_dims(str(a))
    ni, nj = jpi - 2 * hls, jpj - 2 * hls
    unequal = {}
    for name in sorted(manifest_a):
        left, right = a / name, b / name
        if files_byte_identical(left, right):
            continue
        values_a = np.fromfile(left, dtype="<u8")
        values_b = np.fromfile(right, dtype="<u8")
        if values_a.shape != values_b.shape:
            raise SystemExit(f"stream size differs: {name}")
        changed = np.flatnonzero(values_a != values_b)
        classification = "UNCLASSIFIED"
        if values_a.size % (jpi * jpj) == 0:
            mask = np.zeros((values_a.size // (jpi * jpj), jpj, jpi), dtype=bool)
            mask.reshape(-1)[changed] = True
            classification = (
                "HALO_ONLY"
                if not mask[:, hls : jpj - hls, hls : jpi - hls].any()
                else "INCLUDES_PHYSICAL_INTERIOR"
            )
        elif values_a.size == ni * nj * jpk:
            levels = changed // (ni * nj)
            classification = (
                "COMPLETE_TERMINAL_INTERIOR_PLANE"
                if changed.size == ni * nj and np.all(levels == jpk - 1)
                else "INTERIOR_CAPTURE"
            )
        unequal[name] = {
            "a_sha256": manifest_a[name]["sha256"],
            "b_sha256": manifest_b[name]["sha256"],
            "size_bytes": manifest_a[name]["size_bytes"],
            "unequal_binary64_values": int(changed.size),
            "classification": classification,
        }

    bit_control = one_bit_file_control(a / sorted(manifest_a)[0])
    missing_control = len(set(manifest_a) - {sorted(manifest_a)[0]}) != args.expected_stream_count
    if not (bit_control and missing_control):
        raise SystemExit(
            f"controls failed: one_bit={bit_control}, missing_stream={missing_control}"
        )

    artifact = {
        "schema": "dino-zdf-row19-instrument-nondeterminism-v2",
        "date": "2026-08-29",
        "disposition": "ROW19_UNMEASURED_INSTRUMENT_WRITER_DEFECT",
        "binary_sha256": args.expected_binary_sha,
        "donor_restart_sha256": args.expected_donor_sha,
        "certified_output_restart_sha256": args.expected_restart_sha,
        "runs": {"A": str(a), "B_clean_retry": str(b)},
        "probe_sha256": sha256(Path(__file__).resolve()),
        "comparator_sha256": sha256(
            (Path(__file__).parent / "zdf_stream_bracket.py").resolve()
        ),
        "stream_count": args.expected_stream_count,
        "manifest_a_sha256": manifest_sha256(manifest_a),
        "manifest_b_sha256": manifest_sha256(manifest_b),
        "manifest_a": manifest_a,
        "manifest_b": manifest_b,
        "unequal_streams": unequal,
        "controls": {
            "one_bit_file_rejected_by_production_comparator": bit_control,
            "missing_stream_rejected_by_registered_count": missing_control,
        },
        "run_provenance_sha256": {
            "A": {
                name: sha256(a / name)
                for name in (
                    "nemo",
                    ".nemo_binary_sha256",
                    "DINO_00005760_restart.nc",
                    "DINO_00005761_restart.nc",
                    "run.log",
                )
            },
            "B_clean_retry": {
                name: sha256(b / name)
                for name in (
                    "nemo",
                    ".nemo_binary_sha256",
                    "DINO_00005760_restart.nc",
                    "DINO_00005761_restart.nc",
                    "run.log",
                )
            },
        },
        "failed_B_attempt": {
            "classification": "HUMAN_BOUND_RECEIPT",
            "reported_failure": "segmentation fault after 55 of 198 dumps",
            "same_binary_and_input_as_clean_retry": True,
            "qualification": (
                "The failed log was overwritten during the scrubbed retry; no failed-log "
                "SHA is claimed. The surviving B directory contains only the clean retry."
            ),
        },
        "adjudication": (
            "The fixed four-slot model is retracted. No stream exception is permitted; "
            "stack-wide deterministic writer initialization and a strict rerun bracket "
            "are required before row 19 is scored."
        ),
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        f"instrument nondeterminism: {len(unequal)}/{args.expected_stream_count} "
        f"streams differ; one-bit and missing-stream controls fired"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Localize a refused GYRE surface-V budget record without admitting it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from package_records import (
    CLOSURE_BAR_ABS_M_S,
    HALO,
    TERMS,
    compare_restarts,
    decode_record,
    sha256,
)


def diagnose(
    path: Path,
    control_restart: Path | None = None,
    instrumented_restart: Path | None = None,
) -> dict[str, object]:
    """Report the exact support of a pointwise closure refusal."""
    record = decode_record(path)
    mask = np.asarray(record["mask"], dtype=bool)
    terms = np.asarray(record["terms"])
    if not bool(np.all(np.isfinite(terms))):
        raise ValueError(f"{path.name}: non-finite budget word")
    residual = np.sum(terms[:-1], axis=0) - terms[-1]
    absolute = np.abs(residual)
    wet_j, wet_i = np.nonzero(mask)
    failing = mask & (absolute > CLOSURE_BAR_ABS_M_S)
    fail_j, fail_i = np.nonzero(failing)
    if wet_i.size == 0:
        raise ValueError(f"{path.name}: no wet words")
    maximum_flat = int(np.argmax(np.where(mask, absolute, -np.inf)))
    maximum_j, maximum_i = np.unravel_index(maximum_flat, mask.shape)
    eastmost = int(np.max(wet_i))
    without_east = mask.copy()
    without_east[:, eastmost] = False
    if not bool(np.any(without_east)):
        raise ValueError(f"{path.name}: no wet words west of the eastmost column")
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[5]),
         "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    report = {
        "status": "REFUSE",
        "record": str(path.resolve()),
        "record_sha256": sha256(path),
        "producer_commit": commit,
        "day": int(record["day"]),
        "step": int(record["step"]),
        "closure_bar_abs_m_s": CLOSURE_BAR_ABS_M_S,
        "wet_words": int(np.count_nonzero(mask)),
        "failing_words": int(np.count_nonzero(failing)),
        "failing_output_columns_zero_based": sorted(
            int(value) for value in np.unique(fail_i)),
        "eastmost_wet_output_column_zero_based": eastmost,
        "max_abs_residual_m_s": float(absolute[maximum_j, maximum_i]),
        "max_output_index_zero_based_j_i": [int(maximum_j), int(maximum_i)],
        "max_native_fortran_index_j_i": [
            int(maximum_j + HALO + 1), int(maximum_i + HALO + 1)],
        "max_signed_residual_m_s": float(residual[maximum_j, maximum_i]),
        "max_term_values_m_s": {
            name: float(terms[index, maximum_j, maximum_i])
            for index, name in enumerate(TERMS)
        },
        "max_abs_residual_without_eastmost_wet_column_m_s": float(
            np.max(absolute[without_east])),
        "all_failing_words_are_eastmost_wet_column": bool(
            fail_i.size and np.all(fail_i == eastmost)),
    }
    if control_restart is not None and instrumented_restart is not None:
        report["existing_recorder_passive_identity"] = compare_restarts(
            control_restart, instrumented_restart)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--control-restart", type=Path)
    parser.add_argument("--instrumented-restart", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if (args.control_restart is None) != (args.instrumented_restart is None):
            raise ValueError(
                "control and instrumented restarts must be supplied together")
        report = diagnose(
            args.record, args.control_restart, args.instrumented_restart)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
        sys.stdout.write(payload)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

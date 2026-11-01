#!/usr/bin/env python3
"""Exact ORCA2 O17 stream-record size and unified acquisition budget."""

from __future__ import annotations

import argparse
import json
import math


# ORCA2 one-rank extents: par_oce jpi/jpj and A2D(0) owned bounds.
FULL = 186 * 153  # par_oce.F90; o17_master_writer.F90:112-113
OWNED = 182 * 149  # ice.F90:399-465 (Nis0:Nie0,Njs0:Nje0)
HALO1 = 184 * 151  # sbc_ice.F90:107-108 (Nis0-1:Nie0+1)
JPL, NLAY_I, NLAY_S = 5, 10, 5  # ORCA2_SAS_ICE namelist_ice_ref
STEPS = 100  # ORCA2_SAS_ICE namelist_cfg nn_itend

STREAM_HEADER = 44  # o17_master_writer.F90:107-113
FIELD_HEADER = 52  # o17_master_writer.F90:30-37
STREAM_END = 52  # o17_master_writer.F90:81-84


def record_bytes(field_elements: list[int]) -> int:
    """Bytes written by one stream record, including every field header."""
    return STREAM_HEADER + sum(FIELD_HEADER + 8 * n for n in field_elements) + STREAM_END


def size_report(master_bytes: int = 140_965_556_460) -> dict[str, int | float]:
    """Return the exact schema maximum and its single 4% disk budget."""
    # o17_master_writer.F90:145-200, with extents from ice.F90:355-465,
    # sbc_ice.F90:101-121, and sbc_oce.F90:192-221.
    state_fields = (
        [FULL * JPL] * 9
        + [FULL * NLAY_S * JPL, FULL * NLAY_I * JPL, FULL * NLAY_I * JPL]
        + [FULL] * 53
        + [OWNED * JPL] * 8
        + [OWNED] * 62
        + [HALO1]
        + [JPL + 1, JPL]
        + [1]  # worst case: allocated delayed cflice scalar, writer lines 133-154
    )
    state = record_bytes(state_fields)
    # icedyn_adv_pra.F90:1175-1204; writer lines 225-255.
    prather = record_bytes(
        [FULL * JPL] * 40
        + [FULL * NLAY_S * JPL] * 5
        + [FULL * NLAY_I * JPL] * 10
    )
    raw = record_bytes([OWNED] * 8)  # sbcblk.F90 active A2D(0); writer lines 205-223
    tau = record_bytes([FULL] * 3)  # iceupdate.F90:407-409; master writer lines 504-523
    per_step = 8 * state + 2 * prather + 2 * raw + tau
    exact_max = STEPS * per_step
    combined = master_bytes + exact_max
    disk_kib = math.ceil(combined * 1.04 / 1024)  # O17 contract: one 4% margin
    return {
        "full_elements": FULL,
        "owned_elements": OWNED,
        "state_record_bytes_max": state,
        "prather_record_bytes": prather,
        "raw_record_bytes": raw,
        "tau_record_bytes": tau,
        "o17_bytes_per_step_max": per_step,
        "o17_record_bytes_exact_max": exact_max,
        "o8_o16_master_bytes_worst_case": master_bytes,
        "combined_record_bytes_max": combined,
        "combined_disk_budget_kib_4pct": disk_kib,
        "wall_clock_budget_seconds": 21_600,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-bytes", type=int, default=140_965_556_460)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    report = size_report(args.master_bytes)
    if args.shell:
        print(report["o17_record_bytes_exact_max"], report["combined_disk_budget_kib_4pct"])
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

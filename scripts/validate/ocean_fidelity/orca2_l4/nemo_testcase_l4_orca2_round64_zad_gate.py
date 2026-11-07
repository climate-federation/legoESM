#!/usr/bin/env python3
"""Replay ORCA2's compiled explicit ZAD statement from admitted rank records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l4_orca2_round62_vector_split_admission import (
    GateError,
    read_record,
    require,
)


RECORD_COMMIT = "1026f84027c730f4f61b70863e69c38c3ba9d9a3"


def replay_zad(arrays: dict[str, np.ndarray], header: dict[str, int]):
    """Literal scalar transcription of compiled ``dynzad.f90:102-137``."""
    u = arrays["uu_Kmm"]
    v = arrays["vv_Kmm"]
    ww = arrays["ww"]
    area = arrays["e1e2t"][:, :, 0]
    reciprocal_u = arrays["r1_e1e2u"][:, :, 0]
    reciprocal_v = arrays["r1_e1e2v"][:, :, 0]
    e3u = arrays["e3u_Kmm"]
    e3v = arrays["e3v_Kmm"]
    out_u = np.array(arrays["after_keg_u"], copy=True)
    out_v = np.array(arrays["after_keg_v"], copy=True)
    carry_u = np.zeros((header["jpi"], header["jpj"]), dtype=np.float64)
    carry_v = np.zeros((header["jpi"], header["jpj"]), dtype=np.float64)
    i0, i1 = header["ntsi"] - 1, header["ntei"] - 1
    j0, j1 = header["ntsj"] - 1, header["ntej"] - 1

    for k in range(header["jpk"] - 2):
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                zwf = np.float64(area[i, j] * ww[i, j, k + 1])
                zwfi = np.float64(area[i + 1, j] * ww[i + 1, j, k + 1])
                zwfj = np.float64(area[i, j + 1] * ww[i, j + 1, k + 1])
                flux_u = np.float64(zwfi + zwf)
                flux_v = np.float64(zwfj + zwf)
                delta_u = np.float64(u[i, j, k] - u[i, j, k + 1])
                delta_v = np.float64(v[i, j, k] - v[i, j, k + 1])
                next_u = np.float64(flux_u * delta_u)
                next_v = np.float64(flux_v * delta_v)
                scale_u = np.float64(np.float64(0.25) * reciprocal_u[i, j])
                scale_u = np.float64(scale_u / e3u[i, j, k])
                scale_v = np.float64(np.float64(0.25) * reciprocal_v[i, j])
                scale_v = np.float64(scale_v / e3v[i, j, k])
                pair_u = np.float64(carry_u[i, j] + next_u)
                pair_v = np.float64(carry_v[i, j] + next_v)
                out_u[i, j, k] = np.float64(
                    out_u[i, j, k] - np.float64(scale_u * pair_u)
                )
                out_v[i, j, k] = np.float64(
                    out_v[i, j, k] - np.float64(scale_v * pair_v)
                )
                carry_u[i, j] = next_u
                carry_v[i, j] = next_v

    k = header["jpkm1"] - 1
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            scale_u = np.float64(np.float64(0.25) * reciprocal_u[i, j])
            scale_u = np.float64(scale_u / e3u[i, j, k])
            scale_v = np.float64(np.float64(0.25) * reciprocal_v[i, j])
            scale_v = np.float64(scale_v / e3v[i, j, k])
            out_u[i, j, k] = np.float64(
                out_u[i, j, k] - np.float64(scale_u * carry_u[i, j])
            )
            out_v[i, j, k] = np.float64(
                out_v[i, j, k] - np.float64(scale_v * carry_v[i, j])
            )
    return out_u, out_v


def score(arrays: dict[str, np.ndarray], header: dict[str, int], *, plant=False):
    replay_u, replay_v = replay_zad(arrays, header)
    owned = (
        slice(header["ntsi"] - 1, header["ntei"]),
        slice(header["ntsj"] - 1, header["ntej"]),
        slice(0, header["jpkm1"]),
    )
    if plant:
        active = np.argwhere(arrays["umask"][owned] != 0.0)
        require(bool(active.size), "ZAD plant has no active U cell")
        local = tuple(active[0])
        index = tuple(part.start + item for part, item in zip(owned, local, strict=True))
        replay_u[index] = np.nextafter(replay_u[index], np.inf)
    rows = {}
    for component, replay, mask in (
        ("U", replay_u, arrays["umask"]),
        ("V", replay_v, arrays["vmask"]),
    ):
        expected = arrays[f"after_zad_{component.lower()}"][owned]
        actual = replay[owned]
        active = mask[owned] != 0.0
        unequal = np.logical_and(active, actual.view(np.uint64) != expected.view(np.uint64))
        rows[component] = {
            "active": int(np.count_nonzero(active)),
            "unequal": int(np.count_nonzero(unequal)),
            "max_abs": float(np.max(np.abs(actual[active] - expected[active]))),
            "dtype": str(actual.dtype),
        }
    return rows


def run(root: Path, *, expect_commit: str, plant=False) -> dict[str, object]:
    stamp = worktree_stamp()
    require(stamp["commit"].lower() == expect_commit.lower(), "worktree commit changed")
    producer = root / "producer_commit.txt"
    require(producer.is_file(), "missing producer commit")
    require(producer.read_text().strip() == RECORD_COMMIT, "record producer changed")
    paths = sorted(root.glob("oracle_vector_adv_split_kt00000001_s2_r*.bin"))
    require(len(paths) == 2, "expected exactly two rank records")
    totals = {
        component: {"active": 0, "unequal": 0, "max_abs": 0.0, "dtype": "float64"}
        for component in ("U", "V")
    }
    rank_rows = []
    for path in paths:
        record = read_record(path)
        rows = score(record["arrays"], record["header"], plant=plant)
        rank_rows.append({"rank": record["header"]["rank"], "rows": rows})
        for component in totals:
            totals[component]["active"] += rows[component]["active"]
            totals[component]["unequal"] += rows[component]["unequal"]
            totals[component]["max_abs"] = max(
                totals[component]["max_abs"], rows[component]["max_abs"]
            )
    exact = all(row["unequal"] == 0 for row in totals.values())
    if plant:
        require(not exact, "ZAD one-ULP plant stayed exact")
    return {
        "format": "nemo-testcase-l4-orca2-round64-zad-gate-v1",
        "claim_label": "given NEMO's recorded operands",
        "worktree": stamp,
        "record_producer_commit": RECORD_COMMIT,
        "execution_regime": "scalar fp64 source-order replay",
        "rank_rows": rank_rows,
        "totals": totals,
        "first_nonbit_statement": None if exact else "dyn_zad",
        "status": "PLANT_FIRED" if plant else (
            "AT_BAR_BIT_EXACT" if exact else "DEBT"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(
            args.root, expect_commit=args.expect_commit, plant=args.plant
        )
    except GateError as exc:
        if args.plant:
            print(f"STATUS PLANT-FIRED: {exc}")
            return 1
        raise
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print(f"STATUS {report['status']}")
    if args.plant:
        return 1
    return 0 if report["status"] == "AT_BAR_BIT_EXACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())

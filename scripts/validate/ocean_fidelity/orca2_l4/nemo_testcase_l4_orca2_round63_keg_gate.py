#!/usr/bin/env python3
"""Replay ORCA2's compiled C2 KEG statement from intact record fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l4_orca2_round63_vector_record_diagnosis import (
    RECORD_COMMIT,
    diagnose_record,
    require,
)

CAPTURE = frozenset(
    {
        "before_keg_u",
        "before_keg_v",
        "after_keg_u",
        "after_keg_v",
        "uu_Kmm",
        "vv_Kmm",
        "r1_e1u",
        "r1_e2v",
        "umask",
        "vmask",
    }
)


def replay_keg(arrays: dict[str, np.ndarray], header: dict[str, int]):
    """Literal source-order transcription of compiled dynkeg C2 loops."""
    u = arrays["uu_Kmm"]
    v = arrays["vv_Kmm"]
    out_u = np.array(arrays["before_keg_u"], copy=True)
    out_v = np.array(arrays["before_keg_v"], copy=True)
    hke = np.zeros((header["jpi"], header["jpj"]), dtype=np.float64)
    for k in range(header["jpkm1"]):
        for jj in range(header["ntsj"] - 1, header["ntej"] + 1):
            for ji in range(header["ntsi"] - 1, header["ntei"] + 1):
                zu = np.float64(u[ji - 1, jj, k] * u[ji - 1, jj, k])
                zu = np.float64(zu + u[ji, jj, k] * u[ji, jj, k])
                zv = np.float64(v[ji, jj - 1, k] * v[ji, jj - 1, k])
                zv = np.float64(zv + v[ji, jj, k] * v[ji, jj, k])
                hke[ji, jj] = np.float64(np.float64(0.25) * np.float64(zv + zu))
        for jj in range(header["ntsj"] - 1, header["ntej"]):
            for ji in range(header["ntsi"] - 1, header["ntei"]):
                grad_u = np.float64(hke[ji + 1, jj] - hke[ji, jj])
                grad_v = np.float64(hke[ji, jj + 1] - hke[ji, jj])
                out_u[ji, jj, k] = np.float64(
                    out_u[ji, jj, k] - np.float64(grad_u * arrays["r1_e1u"][ji, jj])
                )
                out_v[ji, jj, k] = np.float64(
                    out_v[ji, jj, k] - np.float64(grad_v * arrays["r1_e2v"][ji, jj])
                )
    return out_u, out_v


def score(arrays: dict[str, np.ndarray], header: dict[str, int], *, plant=False):
    replay_u, replay_v = replay_keg(arrays, header)
    owned = (
        slice(header["ntsi"] - 1, header["ntei"]),
        slice(header["ntsj"] - 1, header["ntej"]),
        slice(0, header["jpkm1"]),
    )
    if plant:
        active = np.argwhere(arrays["umask"][owned] != 0.0)
        require(bool(active.size), "KEG plant has no active U cell")
        local = tuple(active[0])
        index = tuple(part.start + item for part, item in zip(owned, local, strict=True))
        replay_u[index] = np.nextafter(replay_u[index], np.inf)
    rows = {}
    for component, replay, mask in (
        ("U", replay_u, arrays["umask"]),
        ("V", replay_v, arrays["vmask"]),
    ):
        expected = arrays[f"after_keg_{component.lower()}"][owned]
        actual = replay[owned]
        active = mask[owned] != 0.0
        unequal = np.logical_and(active, actual != expected)
        rows[component] = {
            "active": int(np.count_nonzero(active)),
            "unequal": int(np.count_nonzero(unequal)),
            "max_abs": float(np.max(np.abs(actual[active] - expected[active]))),
        }
    return rows


def run(root: Path, *, plant=False) -> dict[str, object]:
    producer = root / "producer_commit.txt"
    require(producer.is_file(), "missing producer commit")
    require(producer.read_text().strip() == RECORD_COMMIT, "producer changed")
    paths = sorted(root.glob("oracle_vector_adv_split_kt00000001_s2_r*.bin"))
    require(len(paths) == 2, "expected exactly two rank records")
    rank_rows = []
    totals = {
        component: {"active": 0, "unequal": 0, "max_abs": 0.0}
        for component in ("U", "V")
    }
    for path in paths:
        record = diagnose_record(path, expected_dims=(94, 152, 31, 30), capture=CAPTURE)
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
        require(not exact, "KEG plant stayed exact")
    else:
        require(exact, f"KEG is non-bit: {totals}")
    return {
        "format": "nemo-testcase-l4-orca2-round63-keg-gate-v1",
        "claim_label": "given NEMO's recorded operands",
        "worktree": worktree_stamp(),
        "producer_commit": RECORD_COMMIT,
        "rank_rows": rank_rows,
        "totals": totals,
        "status": "PLANT_FIRED" if plant else "AT_BAR_BIT_EXACT",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.root, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    print(f"STATUS {report['status']}")
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())

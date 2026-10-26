#!/usr/bin/env python3
"""Gate the admitted OVERFLOW kt=3 stage-2 VOR-to-UP3 boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l1_overflow_round50_pair_gate import (
    GateError,
    admit,
    read_record,
    require,
)
from nemo_testcase_l1_overflow_round51_pair_gate import (
    ORACLE_ROOT,
    PRODUCER_COMMIT,
    _nemo_owned,
    _score,
)
from nemo_testcase_phase3_trajectory_gate import expected_masks


COMPILED_ROOT = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo"
)
NAMELIST = COMPILED_ROOT.parents[2] / "EXP00/namelist_cfg"
MISSING_STREAMS = (
    "oracle_transport_kt00000003_s2.bin",
    "oracle_up3_internal_kt00000003_s2.bin",
)


def _require_executed_branch() -> dict:
    stage = (COMPILED_ROOT / "stprk3_stg.f90").read_text()
    dispatch = (COMPILED_ROOT / "dynadv.f90").read_text()
    up3 = (COMPILED_ROOT / "dynadv_up3.f90").read_text()
    namelist = NAMELIST.read_text()
    sentinels = {
        "stage_after_vor": "CALL r50_mom_uv( 'after_vor', uu, vv, Krhs )",
        "stage_dyn_adv": "CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )",
        "stage_after_adv": "CALL r50_mom_uv( 'after_adv', uu, vv, Krhs )",
        "dispatch_up3": "CALL dyn_adv_up3     ( kt       , Kbb, Kmm, puu, pvv, Krhs, pau, pav, paw )",
        "up3_horizontal_flux": "zFu_t(ji+1,jj  ) = (  zFu(ji,jj) + zFu(ji+1,jj  )  )",
        "up3_horizontal_rhs": "puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp",
        "up3_vertical_rhs": "puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zFu_t(ji,jj) - zzFu_kp1 )",
    }
    for name, sentinel in sentinels.items():
        source = dispatch if name == "dispatch_up3" else up3 if name.startswith("up3_") else stage
        require(sentinel in source, f"compiled sentinel missing: {name}")
    require("ln_dynadv_up3 = .true." in namelist,
            "running namelist does not select UP3")
    require("ln_dynadv_vec = .false." in namelist,
            "running namelist no longer selects flux form")
    return {
        "nemo_momentum_advection": "np_FLX_up3",
        "explicit_transport_arguments": ["zFu", "zFv", "zFw"],
        "compiled_root": str(COMPILED_ROOT),
        "sentinels": list(sentinels),
    }


def _boundary_rows(stage2: dict[str, np.ndarray], masks: dict[str, np.ndarray],
                   *, plant: bool) -> list[dict]:
    before_u = _nemo_owned(stage2["after_vor_u"])
    after_u = _nemo_owned(stage2["after_adv_u"])
    active_u = np.asarray(masks["u"], dtype=bool)
    require(before_u.shape == after_u.shape == active_u.shape,
            "UP3 u boundary shape drift")
    baseline_bits = before_u.view(np.uint64) != after_u.view(np.uint64)
    tested = after_u
    plant_at = None
    if plant:
        equal = active_u & ~baseline_bits
        require(equal.any(), "no equal active-u endpoint is available for plant")
        plant_at = tuple(int(i) for i in np.argwhere(equal)[0])
        tested = after_u.copy()
        tested[plant_at] = np.nextafter(tested[plant_at], np.float64(np.inf))
        require(tested[plant_at] != after_u[plant_at],
                "active-u endpoint plant did not move")
    u_row = _score("given.s2.up3.u", before_u, tested, active_u)
    u_row["baseline_n_unequal"] = int(np.count_nonzero(active_u & baseline_bits))
    u_row["plant_at"] = list(plant_at) if plant_at is not None else None
    v_row = _score(
        "given.s2.up3.v",
        _nemo_owned(stage2["after_vor_v"]),
        _nemo_owned(stage2["after_adv_v"]),
        masks["v"],
    )
    return [u_row, v_row]


def run(root: Path, expect_commit: str, plant: str | None) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "analysis worktree is dirty")
    require(stamp["commit"] == expect_commit,
            f"analysis commit {stamp['commit']} != {expect_commit}")
    admission = admit(root, PRODUCER_COMMIT, None)
    require(admission["status"] == "AT_BAR", "round-50 record is not admitted")
    record = read_record(
        root / "oracle_r50_momentum_kt00000003_s2.bin", "momentum", 2)
    card = build_nemo_testcase_card("OVERFLOW-zps")
    masks = expected_masks(card)
    rows = _boundary_rows(record["fields"], masks, plant=plant == "endpoint")
    u_row, v_row = rows
    if plant == "endpoint":
        require(u_row["n_unequal"] == u_row["baseline_n_unequal"] + 1,
                "endpoint plant did not add exactly one active-u refusal")
        status = "PLANTED_REFUSAL"
    else:
        require(v_row["status"] == "UNMEASURED_NO_ACTIVE_FACE",
                "OVERFLOW v-face classification drift")
        status = "ACQUISITION_NEEDED" if u_row["n_unequal"] else "AT_BAR"

    missing = [name for name in MISSING_STREAMS if not (root / name).exists()]
    fields = tuple(record["fields"])
    inventory_insufficient = (
        "after_vor_u" in fields and "after_adv_u" in fields and bool(missing))
    return {
        "format": "nemo-testcase-l1-overflow-round56-up3-boundary-v1",
        "status": status,
        "case": "OVERFLOW-zps",
        "kt": 3,
        "stage": 2,
        "claim_label": "given NEMO's recorded operands",
        "worktree": stamp,
        "precision": "recorded-fp64-bits",
        "record_root": str(root),
        "record_admission": admission["status"],
        "record_sha256": record["sha256"],
        "executed_branch": _require_executed_branch(),
        "boundary_rows": rows,
        "record_fields": list(fields),
        "missing_streams": missing,
        "inventory_insufficient": inventory_insufficient,
        "R56-P1": "CONFIRMED",
        "R56-P2": "CONFIRMED" if u_row["baseline_n_unequal"] else "REFUTED",
        "R56-P3": "CONFIRMED" if inventory_insufficient else "REFUTED",
        "R56-P4": "UNMEASURED_UNTIL_ACQUISITION_PREFLIGHT",
        "R56-P5": "UNMEASURED_UNTIL_FINAL_DIFF",
        "plant": plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=("endpoint",))
    args = parser.parse_args(argv)
    try:
        report = run(args.record_dir, args.expect_commit, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        if args.plant:
            return 2
        return 0 if report["status"] == "AT_BAR" else 1
    except (GateError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

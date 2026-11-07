#!/usr/bin/env python3
"""Replay OVERFLOW kt=3 stage-2 vorticity from the admitted record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
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
    _given_input_rows,
    _nemo_owned,
    _record_pair,
    _score,
)
from nemo_testcase_phase3_trajectory_gate import expected_masks

COMPILED_ROOT = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/"
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo"
)
NAMELIST = COMPILED_ROOT.parents[2] / "EXP00/namelist_cfg"


def _require_executed_branch(card) -> dict:
    """Bind the no-op replay to this record's compiled flux-form ENS arm."""
    dynvor = (COMPILED_ROOT / "dynvor.f90").read_text()
    namelist = NAMELIST.read_text()
    required_source = (
        "CASE( np_FLX_c2 , np_FLX_up3 )",
        "ntot = np_CME",
        "CASE( np_ENS )",
        "CALL vor_ens( kt, Kmm, ntot",
    )
    for sentinel in required_source:
        require(sentinel in dynvor, f"compiled dynvor sentinel missing: {sentinel}")
    for sentinel in (
        "ln_dynadv_up3 = .true.",
        "ln_dynvor_ens = .true.",
        "ln_dynvor_ene = .false.",
        "ln_dynvor_een = .false.",
    ):
        require(sentinel in namelist, f"executed namelist sentinel missing: {sentinel}")
    cfg = card.recipe.model_config
    require(cfg.momentum_advection == "flux_form",
            "OVERFLOW card no longer uses flux-form momentum advection")
    require(cfg.coriolis_scheme == "matsuno_split",
            "OVERFLOW card Coriolis program drift")
    f_t = np.asarray(card.recipe.grid.f_T)
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    wet_rows = int(np.count_nonzero(np.any(wet, axis=1)))
    require(np.array_equal(f_t, np.zeros_like(f_t)),
            "OVERFLOW rotation is no longer exactly zero")
    require(wet_rows == 1, "OVERFLOW no longer has exactly one wet row")
    return {
        "nemo_momentum_advection": "np_FLX_up3",
        "nemo_vorticity_scheme": "np_ENS",
        "nemo_vorticity_kind": "np_CME",
        "legoesm_momentum_advection": cfg.momentum_advection,
        "legoesm_coriolis_scheme": cfg.coriolis_scheme,
        "rotation_nonzero": int(np.count_nonzero(f_t)),
        "wet_rows": wet_rows,
    }


def _vorticity_rows(momentum: dict, masks: dict, *, plant: bool) -> list[dict]:
    """Score NEMO's VOR accumulator against the zero-addend replay."""
    stage2 = momentum[2]
    oracle_u = _nemo_owned(stage2["after_vor_u"])
    replay_u = _nemo_owned(stage2["after_hpg_u"])
    active_u = np.asarray(masks["u"], dtype=bool)
    require(oracle_u.shape == replay_u.shape == active_u.shape,
            "vorticity u replay shape drift")
    original_bits = replay_u.view(np.uint64)
    oracle_bits = oracle_u.view(np.uint64)
    baseline_unequal = active_u & (original_bits != oracle_bits)
    if plant:
        equal = active_u & ~baseline_unequal
        require(equal.any(), "no equal active u cell is available for the plant")
        replay_u = replay_u.copy()
        at = tuple(np.argwhere(equal)[0])
        replay_u[at] = np.nextafter(replay_u[at], np.float64(np.inf))
        require(replay_u[at] != _nemo_owned(stage2["after_hpg_u"])[at],
                "active-u plant did not move")
    u_row = _score(
        "given.s2.vor.u", oracle_u, replay_u, active_u)
    unequal = active_u & (replay_u.view(np.uint64) != oracle_bits)
    if unequal.any():
        first = tuple(int(i) for i in np.argwhere(unequal)[0])
        u_row["first_unequal"] = list(first)
    zero_delta = active_u & (oracle_u == 0.0) & (replay_u == 0.0)
    u_row["signed_zero_unequal"] = int(np.count_nonzero(
        zero_delta & (replay_u.view(np.uint64) != oracle_bits)))
    u_row["replay_positive_zero_oracle_negative_zero"] = int(np.count_nonzero(
        zero_delta & ~np.signbit(replay_u) & np.signbit(oracle_u)))
    u_row["replay_negative_zero_oracle_positive_zero"] = int(np.count_nonzero(
        zero_delta & np.signbit(replay_u) & ~np.signbit(oracle_u)))
    u_row["plant"] = plant
    u_row["baseline_n_unequal"] = int(np.count_nonzero(baseline_unequal))
    return [
        u_row,
        _score(
            "given.s2.vor.v",
            _nemo_owned(stage2["after_vor_v"]),
            _nemo_owned(stage2["after_hpg_v"]),
            masks["v"],
        ),
    ]


def run(root: Path, expect_commit: str, plant: str | None) -> dict:
    stamp = worktree_stamp()
    require(stamp["clean"], "producer worktree is dirty")
    require(stamp["commit"] == expect_commit,
            f"producer commit mismatch: {stamp['commit']} != {expect_commit}")
    require(jax.default_backend() == "cpu", "round-52 gate is CPU-only")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")

    admission = admit(root, PRODUCER_COMMIT, None)
    require(admission["status"] == "AT_BAR", "round-50 record is not admitted")
    momentum, tracer = _record_pair(root)
    stage1_record = read_record(
        root / "oracle_r50_momentum_kt00000003_s1.bin", "momentum", 1)
    stage2_record = read_record(
        root / "oracle_r50_momentum_kt00000003_s2.bin", "momentum", 2)
    stage1_header = stage1_record["header"]
    stage2_header = stage2_record["header"]
    require(stage1_header["Kaa"] == stage2_header["Kmm"],
            "stage-1 Kaa is not the recorded stage-2 Kmm slot")
    require(momentum[1]["postbar_kaa_u"].shape
            == momentum[2]["after_hpg_u"].shape,
            "stage-1 Kaa/stage-2 Kmm record extent drift")

    card = build_nemo_testcase_card("OVERFLOW-zps")
    branch = _require_executed_branch(card)
    masks = expected_masks(card)
    prerequisite = _given_input_rows(card, momentum, tracer, masks, None)
    hpg_u = next(row for row in prerequisite if row["name"] == "given.s2.hpg.u")
    hpg_v = next(row for row in prerequisite if row["name"] == "given.s2.hpg.v")
    require(hpg_u["exact"], "exact active-u HPG prerequisite no longer holds")
    require(hpg_v["status"] == "UNMEASURED_NO_ACTIVE_FACE",
            "v-face domain classification drift")

    rows = _vorticity_rows(momentum, masks, plant=plant == "vor_u")
    u_row, v_row = rows
    if plant == "vor_u":
        require(u_row["n_unequal"] == u_row["baseline_n_unequal"] + 1,
                "one-ULP active-u vorticity plant did not add one refusal")
        status = "PLANTED_REFUSAL"
    else:
        status = "AT_BAR" if (
            u_row["exact"] and v_row["status"] == "UNMEASURED_NO_ACTIVE_FACE"
        ) else "DEBT"

    return {
        "format": "nemo-testcase-l1-overflow-round52-vorticity-v1",
        "status": status,
        "case": "OVERFLOW-zps",
        "kt": 3,
        "stage": 2,
        "worktree": stamp,
        "precision": "cpu-fp64-libm-production-jit",
        "record_root": str(root),
        "record_admission": {
            "status": admission["status"],
            "producer_commit": admission["producer_commit"],
            "records": len(admission["records"]),
        },
        "executed_branch": branch,
        "stage_slot_mapping": {
            "source": "stage-1 postbar Kaa -> stage-2 Kmm",
            "stage1_Kaa": stage1_header["Kaa"],
            "stage2_Kmm": stage2_header["Kmm"],
        },
        "prerequisite_rows": prerequisite,
        "vorticity_rows": rows,
        "R52-P1": "CONFIRMED",
        "R52-P2": "CONFIRMED" if status == "AT_BAR" else "REFUTED",
        "R52-P3": "REFUTED_PREMISE_BUT_CONTROL_FIRES" if plant == "vor_u"
        else "NOT_RUN",
        "R52-P4": "UNMEASURED_UNTIL_FINAL_DIFF",
        "plant": plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=("vor_u",))
    args = parser.parse_args(argv)
    try:
        report = run(args.record_dir, args.expect_commit, args.plant)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(rendered, end="")
        return 2 if args.plant else (0 if report["status"] == "AT_BAR" else 1)
    except (GateError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "REFUSE", "reason": str(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

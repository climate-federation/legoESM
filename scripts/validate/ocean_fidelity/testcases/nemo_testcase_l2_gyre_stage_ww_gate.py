#!/usr/bin/env python3
"""Score GYRE's tracer-consumed RK3 stage ``ww`` against NEMO V2.

The older ``oracle_transport_*`` records precede ``tra_adv_trp`` in NEMO's
vector-invariant branch and therefore cannot score ``ww``.  This gate consumes
the Round-21 post-``tra_adv_trp`` records, runs the same production-JIT step
with a WRITE-only return seam, and reports both the production full-step clock
and the private one-variable NEMO stage-clock arm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

HERE = Path(__file__).resolve().parent
PHASE3_PATH = HERE / "nemo_testcase_l2_gyre_phase3_gate.py"
SPEC = importlib.util.spec_from_file_location("gyre_phase3_gate", PHASE3_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"cannot load {PHASE3_PATH}")
phase3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase3)

DEFAULT_ROOT = phase3.STAGE_WW_ROOT
STAGE_DT = {1: 4800.0, 2: 7200.0, 3: 14400.0}


def _host(value):
    result = jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf) if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
        value,
    )
    jax.clear_caches()
    return result


def run(root: Path, *, plant: bool = False) -> dict:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    phase3.require(jax.default_backend() == "cpu", "stage-W gate is CPU-only")
    phase3.require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    phase3.require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    phase3.require(get_policy() == policy, "fp64 scalar-libm policy is inactive")

    records = {}
    artifacts = {}
    for stage in (1, 2, 3):
        path = root / f"oracle_rkstage_ww_kt00000001_s{stage}.bin"
        phase3.require(path.is_file(), f"missing {path}")
        record = phase3.read_stage_ww(path, stage)
        phase3.require(
            record["rDt_s"] == STAGE_DT[stage],
            f"stage {stage}: rDt={record['rDt_s']} != {STAGE_DT[stage]}",
        )
        records[stage] = record
        artifacts[path.name] = phase3.sha256(path)

    card = build_nemo_testcase_card(phase3.CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    masks = phase3.expected_masks(card)
    freshwater, surface = phase3._surface_forcings(card, card.recipe.initial_state, 1)

    rows = []
    arm_rows = []
    for stage in (1, 2, 3):

        def candidate(stage_clock: bool):
            hooks = _NEMOWSRK3TestHooks(
                expose_tracer_transport_stage=stage,
                expose_tracer_transport_as_ww=True,
                source_stage_wzv_clock_arm=stage_clock,
            )
            result = LatLonCGridOceanModel(
                card.recipe.grid,
                card.recipe.z_coord,
                cfg,
                _nemo_ws_test_hooks=hooks,
            ).step(
                card.recipe.initial_state,
                dt=card.dt_s,
                freshwater=freshwater,
                surface_forcing=surface,
            )
            return np.asarray(_host(result).T.data)

        oracle = records[stage]["ww"][..., : card.recipe.z_coord.n_levels]
        faithful = phase3.score(
            f"{phase3.CASE}.kt1.stage{stage}.ww.production",
            oracle,
            candidate(False),
            masks["T"],
            plant=plant and stage == 1,
        )
        arm = phase3.score(
            f"{phase3.CASE}.kt1.stage{stage}.ww.nemo_stage_clock_arm",
            oracle,
            candidate(True),
            masks["T"],
        )
        faithful["clock_seconds"] = card.dt_s
        arm["clock_seconds"] = STAGE_DT[stage]
        rows.append(faithful)
        arm_rows.append(arm)

    if plant:
        phase3.require(
            rows[0]["status"] == "DEBT" and rows[0]["n_unequal"] >= 1,
            "stage-W planted cell did not fire",
        )
    failed = [row["name"] for row in rows if row["status"] != "AT-BAR"]
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-stage-ww-v1",
        "case": phase3.CASE,
        "status": "AT-BAR" if not failed else "DEBT",
        "failed_rows": failed,
        "production_rows": rows,
        "nemo_stage_clock_arm_rows": arm_rows,
        "scaling_before_owner": True,
        "owner": (
            "REFUTED_ORCA2_GATE_OR_OPERAND_PAIRING"
            if not failed
            else "GYRE_OWNER_SHARED_WZV_STAGE_CLOCK"
            if all(row["status"] == "AT-BAR" for row in arm_rows)
            else "UNMEASURED_STAGE_W_CLOCK_NOT_SOLE_OWNER"
        ),
        "execution": {
            "backend": jax.default_backend(),
            "production_jit": True,
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
        },
        "source": {
            "stage_clocks": "stprk3_stg.F90:118-124,173-178,217-222",
            "wzv_call": "traadv.F90:220-235",
            "qco_clock_operand": "sshwzv.F90:330-336",
        },
        "artifacts": artifacts,
        "plant": plant,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, plant=args.plant)
    except (phase3.GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

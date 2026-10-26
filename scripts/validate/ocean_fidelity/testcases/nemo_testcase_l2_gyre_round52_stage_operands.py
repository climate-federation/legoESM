#!/usr/bin/env python3
"""Round-52 live kt=1 WS-RK3 clock/LDF accumulator gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_entry,
    require,
    score,
)
from nemo_testcase_l2_gyre_round46_kt2_stage_gate import (
    _owned2,
    _owned3,
    read_stage,
)


ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/"
    "oracle_kt2_stage")
OLD_EXIT_MAX = {"u": 2.7478404751243857e-12,
                "v": 3.305560306813421e-12}


def _native(value, face):
    value = np.asarray(value)
    return value[:, 1:, :] if face == "u" else value[1:, :, :]


def _row(name, oracle, candidate, mask, *, plant=False):
    return score(name, oracle, candidate, mask, plant=plant)


def _accumulators(parts):
    """Materialize compiled HPG->VOR->KEG->ZAD->LDF barriers."""
    out = {}
    for face in ("u", "v"):
        vor = parts[f"after_vor_{face}"].data
        keg = jax.lax.optimization_barrier(vor + parts[f"keg_{face}"].data)
        zad = jax.lax.optimization_barrier(keg + parts[f"zad_{face}"].data)
        out[f"after_zad_{face}"] = zad
        out[f"after_adv_{face}"] = parts[f"after_adv_{face}"].data
        out[f"after_ldf_{face}"] = parts[f"after_ldf_{face}"].data
    return out


def run(root: Path, expect_commit: str, plant: str | None = None):
    stamp = worktree_stamp()
    expected = "0" * 40 if plant == "stamp" else expect_commit.lower()
    require(stamp["clean"], "producer worktree is dirty")
    require(stamp["commit"].lower() == expected,
            f"producer commit mismatch: {stamp['commit']} != {expected}")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT required")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")

    records = {stage: read_stage(
        root / f"oracle_momstage_kt00000001_s{stage}.bin")
        for stage in (1, 2, 3)}
    oracle_exit = read_entry(root / "oracle_step_entry_kt00000002.bin")
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    require((cfg.outer_integrator, cfg.momentum_time_integrator)
            == ("forward_euler", "rk3_ws"), "wrong GYRE stage program")
    state = card.recipe.initial_state
    freshwater, surface = _surface_forcings(card, state, 1)
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg).step(
            state, dt=card.dt_s, freshwater=freshwater,
            surface_forcing=surface)
    trace = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_live_stage_operands=True)).step(
                state, dt=card.dt_s, freshwater=freshwater,
                surface_forcing=surface)
    masks = expected_masks(card)
    ordinary_fields = lego_fields(ordinary)
    trace_fields = lego_fields(trace.state_after)
    observer = [_row(
        f"trace_noninterference.{field}", ordinary_fields[field],
        trace_fields[field], masks[field])
        for field in ("T", "S", "u", "v", "ssh")]
    require(all(row["n_unequal"] == 0 for row in observer),
            f"trace perturbed production output: {observer}")

    accum_rows = []
    operand_rows = []
    for stage in (2, 3):
        arrays = records[stage]["arrays"]
        parts = trace.operator_operands[stage - 1]
        got = jax.jit(_accumulators)(parts)
        for boundary in (("after_zad", "after_adv") if stage == 2 else
                         ("after_adv", "after_ldf")):
            for face in ("u", "v"):
                mask = _owned3(arrays[f"{face}mask"]) > 0.5
                candidate = _native(got[f"{boundary}_{face}"], face)
                reference = _owned3(arrays[f"{boundary}_{face}"])
                do_plant = (plant == "cell" and stage == 2
                            and boundary == "after_zad" and face == "u")
                accum_rows.append(_row(
                    f"GYRE-zco.kt1.s{stage}.{boundary}.{face}", reference,
                    candidate, mask, plant=do_plant))
        r1_dt = np.asarray([1.0 / float(
            parts["operand_zad_continuity_dt"])], dtype=np.float64)
        operand_rows.append(_row(
            f"GYRE-zco.kt1.s{stage}.zad.r1_Dt",
            np.asarray([arrays["r1_Dt"]]), r1_dt,
            np.ones(1, dtype=bool)))
        if stage == 3:
            tmask = _owned3(arrays["tmask"]) > 0.5
            operand_rows.append(_row(
                "GYRE-zco.kt1.s3.ldf.e3t_Kbb",
                _owned3(arrays["e3t_Kbb"]),
                np.asarray(parts["operand_h_k"]), tmask))

    exit_rows = []
    for field in ("u", "v", "ssh"):
        reference = oracle_exit[field]
        candidate = ordinary_fields[field]
        if field != "ssh":
            reference = reference[..., :candidate.shape[-1]]
        exit_rows.append(_row(
            f"GYRE-zco.kt1.exit.{field}", reference, candidate,
            masks[field]))
    exit_by_field = {
        row["name"].rsplit(".", 1)[-1]: row for row in exit_rows}
    clock_exact = all(row["n_unequal"] == 0 for row in operand_rows
                      if row["name"].endswith("r1_Dt"))
    exit_improved = all(
        exit_by_field[field]["absolute_max"] < OLD_EXIT_MAX[field]
        for field in ("u", "v"))
    if plant == "cell":
        require(any(row["absolute_max"] > 0.5 for row in accum_rows),
                "one-cell plant did not land")
    return {
        "format": "nemo-testcase-l2-gyre-round52-stage-operands-v1",
        "status": ("CONFIRMED" if clock_exact and exit_improved
                   and plant is None else "REFUTED"),
        "prediction": {"clock_exact": clock_exact,
                       "kt1_exit_strictly_improved": exit_improved},
        "accumulator_rows": accum_rows,
        "operand_rows": operand_rows,
        "kt1_exit_rows": exit_rows,
        "trace_noninterference": observer,
        "source": {
            "stage_clock": (
                "stprk3_stg.f90:195-200,239-244,329-335; "
                "sshwzv.f90:277-298"),
            "stage3_ldf": (
                "stprk3_stg.f90:690-712; dynldf_lev.f90:121-140"),
        },
        "worktree": stamp,
        "plant": plant,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=("cell", "stamp"))
    args = parser.parse_args(argv)
    report = run(args.root, args.expect_commit, args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())

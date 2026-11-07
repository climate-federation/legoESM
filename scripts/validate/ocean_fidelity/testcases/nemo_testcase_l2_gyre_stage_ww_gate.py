#!/usr/bin/env python3
"""Score GYRE's RK3 ``ww`` fields and stage-3 ZAD boundary.

The older ``oracle_transport_*`` records precede ``tra_adv_trp`` in NEMO's
vector-invariant branch and therefore cannot score tracer ``ww``.  This gate
consumes the Round-21 post-``tra_adv_trp`` records for that transport-form
field and the Round-41 pre-``dyn_adv`` record for the distinct velocity-form
field consumed by stage-3 ZAD.  The stage velocity comes from the existing
WRITE-only return seam; the committed gate then executes the shared production
WZV and vertical-momentum kernels on that operand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
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
DEFAULT_DYNADV_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round41/"
    "oracle_dynadv_split/oracle_dynadv_split_kt00000001_s3.bin")
STAGE_DT = {1: 4800.0, 2: 7200.0, 3: 14400.0}


def _host(value):
    result = jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf) if isinstance(leaf, (jax.Array, np.ndarray)) else leaf,
        value,
    )
    jax.clear_caches()
    return result


def run(root: Path, *, dynadv_record: Path = DEFAULT_DYNADV_RECORD,
        expect_commit: str, plant: str | bool | None = None) -> dict:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _bc_vertical_momentum_advection,
        nemo_qco_wzv_operands,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
    )
    import nemo_testcase_l2_gyre_round41_dynadv_split as round41

    plant_kind = "ww" if plant is True else plant

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    phase3.require(jax.default_backend() == "cpu", "stage-W gate is CPU-only")
    phase3.require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    phase3.require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    phase3.require(get_policy() == policy, "fp64 scalar-libm policy is inactive")
    stamp = worktree_stamp()
    expected = expect_commit.lower()
    if plant_kind == "stamp":
        expected = "0" * 40
    phase3.require(stamp["clean"] and stamp["untracked_count"] == 0,
                   f"dirty worktree is inadmissible: {stamp}")
    phase3.require(stamp["commit"].lower() == expected,
                   f"commit stamp mismatch: {stamp['commit']} != {expected}")

    phase3.require(dynadv_record.is_file(), f"missing {dynadv_record}")
    split = round41.read_split(dynadv_record)["arrays"]

    artifacts = {}
    tracer_path = root / "oracle_rkstage_ww_kt00000001_s3.bin"
    phase3.require(tracer_path.is_file(), f"missing {tracer_path}")
    tracer_record = phase3.read_stage_ww(tracer_path, 3)
    phase3.require(
        tracer_record["rDt_s"] == STAGE_DT[3],
        f"stage 3: rDt={tracer_record['rDt_s']} != {STAGE_DT[3]}",
    )
    artifacts[tracer_path.name] = phase3.sha256(tracer_path)

    card = build_nemo_testcase_card(phase3.CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    masks = phase3.expected_masks(card)
    freshwater, surface = phase3._surface_forcings(card, card.recipe.initial_state, 1)

    stage_state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_momentum_stage=2),
    ).step(card.recipe.initial_state, dt=card.dt_s,
           freshwater=freshwater, surface_forcing=surface)
    stage_state = _host(stage_state)
    tracer_state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_tracer_transport_stage=3,
            expose_tracer_transport_as_ww=True),
    ).step(card.recipe.initial_state, dt=card.dt_s,
           freshwater=freshwater, surface_forcing=surface)
    tracer_state = _host(tracer_state)
    tracer_oracle = tracer_record["ww"][..., :card.recipe.z_coord.n_levels]
    tracer_row = phase3.score(
        f"{phase3.CASE}.kt1.stage3.ww.tracer_transport_form",
        tracer_oracle, np.asarray(tracer_state.T.data), masks["T"],
        plant=plant_kind == "ww")
    tracer_row["clock_seconds"] = card.dt_s
    rows = [tracer_row]
    arm_rows = []
    eta_entry = jnp.asarray(card.recipe.initial_state.eta.data)
    eta_after = jnp.asarray(stage_state.eta.data)
    eta_half = jax.lax.optimization_barrier(
        jnp.asarray(0.5, dtype=eta_entry.dtype) * (eta_entry + eta_after))
    stage_u = jnp.asarray(stage_state.u.data)
    stage_v = jnp.asarray(stage_state.v.data)
    active = card.recipe.z_coord.is_active.astype(eta_entry.dtype)
    u_live, v_live = compute_face_masks_3d(
        card.recipe.z_coord.is_active, card.recipe.grid)
    u_live = u_live.astype(eta_entry.dtype)
    v_live = v_live.astype(eta_entry.dtype)

    def wzv(eta_before, eta_after_override):
        return nemo_qco_wzv_operands(
            eta_half, eta_before, stage_u, stage_v,
            card.recipe.grid, card.recipe.z_coord,
            u_live, v_live, active, card.dt_s,
            eta_after_override=eta_after_override)

    momentum_ww, _, _ = jax.jit(wzv)(eta_entry, eta_after)
    momentum_ww = np.asarray(momentum_ww)[..., :card.recipe.z_coord.n_levels]
    momentum_ww_oracle = round41._owned(split["ww"])
    momentum_ww_mask = round41._owned(split["wmask"]) > 0.5
    momentum_ww_row = phase3.score(
        f"{phase3.CASE}.kt1.stage3.ww.momentum_velocity_form",
        momentum_ww_oracle, momentum_ww, momentum_ww_mask,
        plant=plant_kind == "ww")

    if plant_kind == "ww":
        phase3.require(
            all(row["status"] == "DEBT" and row["n_unequal"] >= 1
                for row in (tracer_row, momentum_ww_row)),
            "stage-W planted cell did not fire",
        )

    resolved = {
        name: getattr(cfg, name) for name in (
            "momentum_advection", "ke_gradient_scheme",
            "vertical_momentum_scheme", "zad_qco_evaluation",
            "adaptive_implicit_vertadv")
    }
    if plant_kind == "arm":
        resolved["vertical_momentum_scheme"] = "planted_wrong_arm"
    phase3.require(
        resolved == {
            "momentum_advection": "vector_invariant",
            "ke_gradient_scheme": "c2",
            "vertical_momentum_scheme": "nemo_advective",
            "zad_qco_evaluation": "nemo_literal",
            "adaptive_implicit_vertadv": False,
        }, f"GYRE identity card resolved the wrong ZAD arm: {resolved}")

    # Reproduce the current production stage-3 operand source.  The stage
    # state has ``eta_before=None``, so tendencies() substitutes its live
    # Kmm eta for eta_before before calling the same shared WZV helper.
    model_ww, model_h_u, model_h_v = jax.jit(wzv)(eta_half, None)

    def production_zad():
        zeros_u = jnp.zeros_like(stage_u)
        zeros_v = jnp.zeros_like(stage_v)
        return _bc_vertical_momentum_advection(
            zeros_u, zeros_v, zeros_u, zeros_v, model_ww,
            model_h_u, model_h_v, u_live, v_live, card.recipe.grid,
            cfg.momentum_advection, None, cfg, diagnose_momentum=True,
            u_full=stage_u, v_full=stage_v)[2:]

    model_zad_u, model_zad_v = jax.jit(production_zad)()
    model_zad = {
        "u": np.asarray(model_zad_u)[:, 1:, :],
        "v": np.asarray(model_zad_v)[1:, :, :],
    }
    zad_maxima = {}
    for face in ("u", "v"):
        reference = round41._owned(
            split[f"after_zad_{face}"] - split[f"after_keg_{face}"])
        candidate = model_zad[face]
        mask = round41._owned(split[f"{face}mask"]) > 0.5
        if plant_kind == "zad" and face == "u":
            candidate = candidate.copy()
            candidate[tuple(np.argwhere(mask)[0])] += 1.0
        nemo_max = float(np.max(np.abs(reference[mask])))
        model_max = float(np.max(np.abs(candidate[mask])))
        zad_maxima[face] = {
            "model": model_max, "nemo": nemo_max,
            "model_over_nemo": model_max / nemo_max,
            "model_approximately_zero": model_max / nemo_max <= 1.0e-3,
        }

    model_wsd = np.zeros_like(round41._owned(split["wsd_effective"]))
    wsd_mask = round41._owned(split["wmask"]) > 0.5
    if plant_kind == "wsd":
        model_wsd[tuple(np.argwhere(wsd_mask)[0])] = 1.0
    wsd_row = phase3.score(
        f"{phase3.CASE}.kt1.stage3.effective_wsd",
        round41._owned(split["wsd_effective"]), model_wsd,
        wsd_mask)
    if plant_kind == "zad":
        phase3.require(zad_maxima["u"]["model"] > 0.5,
                       "zad planted violation did not fire")
    if plant_kind == "wsd":
        phase3.require(wsd_row["status"] == "DEBT",
                       "wsd planted violation did not fire")
    scored_rows = [*rows, momentum_ww_row, wsd_row]
    failed = [row["name"] for row in scored_rows
              if row["status"] != "AT-BAR"]
    return {
        "worktree": stamp,
        "format": "nemo-testcase-l2-gyre-stage-ww-zad-v2",
        "case": phase3.CASE,
        "status": "AT-BAR" if not failed else "DEBT",
        "failed_rows": failed,
        "production_rows": rows,
        "nemo_stage_clock_arm_rows": arm_rows,
        "momentum_stage3_ww_row": momentum_ww_row,
        "effective_wsd_row": wsd_row,
        "zad_contribution_max_abs": zad_maxima,
        "resolved_card": resolved,
        "scaling_before_owner": True,
        "owner": (
            "STAGE3_ZAD_PRESENT"
            if not any(value["model_approximately_zero"]
                       for value in zad_maxima.values())
            else "MISSING_OR_MISPLACED_STAGE3_ZAD"),
        "execution": {
            "backend": jax.default_backend(),
            "production_jit": True,
            "dtype": "float64",
            "transcendentals": get_policy().transcendentals,
        },
        "source": {
            "stage_clocks": "stprk3_stg.F90:118-124,173-178,217-222",
            "momentum_wzv_call": "stprk3_stg.f90:327-333",
            "momentum_consumer": "dynzad.f90:105-137",
            "shared_recurrence": "sshwzv.f90:271-298",
            "tracer_wzv_call": "traadv.F90:220-235",
            "first_nonbit_statement": (
                "ocean_pe_latlon_cgrid.py:4738-4750 substitutes live Kmm eta "
                "when eta_before is absent; NEMO passes Kbb explicitly"),
        },
        "artifacts": {
            **artifacts, dynadv_record.name: phase3.sha256(dynadv_record)},
        "plant": plant_kind,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dynadv-record", type=Path, default=DEFAULT_DYNADV_RECORD)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant", nargs="?", const="ww",
        choices=("ww", "wsd", "zad", "arm", "stamp"))
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, dynadv_record=args.dynadv_record,
                     expect_commit=args.expect_commit, plant=args.plant)
    except (phase3.GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    if args.plant:
        # Every planted violation exits nonzero after run() proves it fired.
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

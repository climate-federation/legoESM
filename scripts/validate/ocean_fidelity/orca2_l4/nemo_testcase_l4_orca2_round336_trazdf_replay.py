#!/usr/bin/env python3
"""Replay rung-10 kt=2 ``zdf_phy`` and ``tra_zdf`` statement boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round333_tracer_replay as outer,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round336_trazdf_gate as admission,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round336_trazdf_input_audit as input_audit,
)


class GateError(RuntimeError):
    """The admitted record cannot support the statement attribution."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


ROW_ORDER = (
    "zdf_tke_avt_copy",
    "zdf_runoff_avt_add",
    "zdf_evd_avt_replace",
    "zdf_ddm_avt_update",
    "zdf_ddm_avs_write",
    "zdf_iwm_avt_add",
    "zdf_iwm_avs_add",
    "trazdf_effective_temperature_K",
    "trazdf_lower",
    "trazdf_upper",
    "trazdf_diagonal",
    "trazdf_eliminated_diagonal",
    "trazdf_content_rhs_T",
    "trazdf_forward_T",
    "trazdf_back_substitution_T",
    "trazdf_Kaa_T_update",
)
PLANTS = ("row-order", "target-count", "first", "passivity", "record")


def _bits_score(candidate: np.ndarray, oracle: np.ndarray,
                support: np.ndarray) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    support = np.broadcast_to(np.asarray(support, dtype=bool), candidate.shape)
    require(candidate.shape == oracle.shape, f"score shape {candidate.shape} != {oracle.shape}")
    require(bool(np.any(support)), "score support is empty")
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    value_unequal = candidate != oracle
    selected_delta = (candidate - oracle)[support]
    changed_delta = np.abs((candidate - oracle)[value_unequal & support])
    return {
        "support": int(np.count_nonzero(support)),
        "bit_unequal": int(np.count_nonzero(unequal & support)),
        "value_unequal": int(np.count_nonzero(value_unequal & support)),
        "max_abs": float(np.max(changed_delta)) if changed_delta.size else 0.0,
        "rms": float(np.sqrt(np.mean(selected_delta * selected_delta))),
    }


def _global_full(records: list[dict], name: str, tracer: int | None = None) -> np.ndarray:
    slabs = []
    for record in records:
        value = record["fields"][name]
        owned = value[2:-2, 2:-2]
        if tracer is not None:
            owned = owned[..., tracer]
        owned = owned.transpose((1, 0) if owned.ndim == 2 else (1, 0, 2))
        slabs.append(owned)
    return np.concatenate(slabs, axis=1)


def _global_interior(records: list[dict], name: str) -> np.ndarray:
    slabs = [record["fields"][name].transpose(1, 0, 2) for record in records]
    return np.concatenate(slabs, axis=1)


def _target_faces(targets: np.ndarray) -> np.ndarray:
    """Interior W rows read by the registered T-cell matrix rows."""
    faces = np.zeros(targets.shape[:-1] + (targets.shape[-1] - 1,), dtype=bool)
    faces |= targets[..., :-1]
    faces |= targets[..., 1:]
    return faces


def _first(rows: dict[str, dict[str, dict[str, object]]], support: str) -> str | None:
    return next((name for name in ROW_ORDER
                 if rows[name][support]["value_unequal"]), None)


def classify(report: dict[str, object], *, plant: str | None = None) -> dict[str, object]:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    result = json.loads(json.dumps(report))
    if plant == "row-order":
        result["row_order"][0], result["row_order"][1] = (
            result["row_order"][1], result["row_order"][0])
    elif plant == "target-count":
        result["target_count_T"] = 17
    elif plant == "first":
        result["first_differing_target_statement"] = "zdf_tke_avt_copy"
    elif plant == "passivity":
        result["observer_passivity"]["bit_identical"] = False
    elif plant == "record":
        result["record_admission"] = "FAIL"

    require(result.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(result.get("row_order") == list(ROW_ORDER), "source row order moved")
    require(result.get("target_count_T") == 18, "temperature target census moved")
    require(result.get("record_admission") == "PASS_R336_TRAZDF_ACQUISITION",
            "round-336 record is not admitted")
    require(result.get("observer_passivity") == {
        "bit_identical": True, "changed_leaves": 0,
    }, "profile observer moved the ordinary trajectory")
    first = _first(result["rows"], "target")
    require(first == result.get("first_differing_target_statement"),
            "first target statement is not source ordered")
    require(first is not None, "statement walk found no registered departure")
    require(result["rows"]["trazdf_Kaa_T_update"]["target"]["value_unequal"] == 18,
            "the walk does not recover round 333's 18 temperature words")
    require(result["rows"]["trazdf_Kaa_T_update"]["target"]["max_abs"] > 0.0,
            "round-333 temperature maximum vanished")
    result["claim_label"] = "CONFIRMED"
    result["status"] = "PASS_R336_TRAZDF_STATEMENT_REPLAY"
    return result


def measure(label: str, deck_root: Path, record_root: Path,
            expect_commit: str) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
        _NEMOWSTracerZDFTrace,
    )
    from legoesm.ocean.physics.convection import enhanced_diffusion as evd_module
    from legoesm.ocean.physics.vertical_mixing import double_diffusion as ddm_module
    from legoesm.ocean.physics.vertical_mixing import internal_wave_mixing as iwm_module
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        nemo_ordered_tridiagonal_solve,
        nemo_tracer_tridiagonal,
    )

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-336 replay requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "replay requires production JIT on CPU")
    require(label in ("given_nemo_entry", "independent"), "unknown label")

    producer = (record_root / "producer_commit.txt").read_text().strip()
    admitted = admission.run_gate(record_root, expect_commit=producer)
    require(admitted["status"] == "PASS_R336_TRAZDF_ACQUISITION",
            "round-336 tra_zdf record is not admitted")
    records = [admission.read_record(record_root / admission.record_name(rank))
               for rank in (0, 1)]

    _, shipped = rung10.card_fields(deck_root)
    card = r329.selected_card(shipped, arm="recordice", rung="rung10")
    state0 = card.recipe.initial_state
    if label == "given_nemo_entry":
        entry = rung10.assemble_state_fields(record_root, 1, stage=None)
        state0 = state0._replace(
            eta=state0.eta.replace(data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))

    def model(*, trace: bool):
        hooks = _NEMOWSRK3TestHooks(tracer_zdf_trace=True) if trace else None
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            iwm_forcing=card.recipe.iwm_forcing, _nemo_ws_test_hooks=hooks)

    ordinary_model = model(trace=False)
    f1, s1, d1 = outer._surface(card, deck_root, record_root, 1)
    state1 = ordinary_model.step(
        state0, dt=card.dt_s, freshwater=f1, surface_forcing=s1, **d1)
    state1 = jax.tree.map(lambda value: value.block_until_ready(), state1)
    f2, s2, d2 = outer._surface(card, deck_root, record_root, 2)
    ordinary = ordinary_model.step(
        state1, dt=card.dt_s, freshwater=f2, surface_forcing=s2, **d2)
    ordinary = jax.tree.map(lambda value: value.block_until_ready(), ordinary)

    captures: dict[str, list[tuple[np.ndarray, ...]]] = {
        "evd": [], "ddm": [], "iwm": [],
    }

    def store(name, *values):
        captures[name].append(tuple(np.array(value, copy=True) for value in values))

    original_evd = evd_module.compose_evd_coefficient
    original_ddm = ddm_module.nemo_zdf_ddm
    original_iwm = iwm_module.nemo_zdf_iwm

    def observed_evd(base, enhancement, mode, *, convective):
        result = original_evd(base, enhancement, mode, convective=convective)
        if float(convective) == 100.0:  # heat; nn_evdm=0 leaves momentum alone
            jax.debug.callback(lambda *v: store("evd", *v), base, result, ordered=True)
        return result

    def observed_ddm(avt, avm, *args, **kwargs):
        result = original_ddm(avt, avm, *args, **kwargs)
        jax.debug.callback(
            lambda *v: store("ddm", *v), avt, result[0], result[1], ordered=True)
        return result

    def observed_iwm(avt, avs, avm, *args, **kwargs):
        result = original_iwm(avt, avs, avm, *args, **kwargs)
        jax.debug.callback(
            lambda *v: store("iwm", *v), avt, avs, result[0], result[1], ordered=True)
        return result

    evd_module.compose_evd_coefficient = observed_evd
    ddm_module.nemo_zdf_ddm = observed_ddm
    iwm_module.nemo_zdf_iwm = observed_iwm
    try:
        traced = model(trace=True).step(
            state1, dt=card.dt_s, freshwater=f2, surface_forcing=s2, **d2)
        traced = jax.tree.map(lambda value: value.block_until_ready(), traced)
    finally:
        evd_module.compose_evd_coefficient = original_evd
        ddm_module.nemo_zdf_ddm = original_ddm
        iwm_module.nemo_zdf_iwm = original_iwm
    require(isinstance(traced, _NEMOWSTracerZDFTrace),
            "production tracer trace has the wrong type")
    passive, changed = outer._tree_bit_identical(ordinary, traced.state_after)
    require(all(len(captures[name]) == 1 for name in captures),
            f"profile capture census moved: { {name: len(v) for name, v in captures.items()} }")

    oracle_stage3 = rung10.read_state_frame(
        record_root / "oracle_stage_kt00000002_s3.bin", kt=2, stage=3)
    target_rank0 = np.asarray(ordinary.T.data)[:, :90, :30] != oracle_stage3["T"]
    require(int(np.count_nonzero(target_rank0)) == 18,
            "registered temperature target set moved")
    targets = np.zeros_like(np.asarray(ordinary.T.data), dtype=bool)
    targets[:, :90, :30] = target_rank0
    target_faces = _target_faces(targets)

    wet = _global_full(records, "tmask")[..., :30] != 0.0
    wet_faces = wet[..., 1:]
    oracle_phase = {
        name: _global_interior(records, name)[..., 1:30]
        for name in admission.ZDFPHY_PHASE_FIELDS
    }
    oracle_rows = {
        name: _global_full(records, name, tracer=0)[..., :30]
        for name in ("lower", "diagonal", "upper", "eliminated", "content_rhs",
                     "forward", "backsub", "actual_Kaa")
    }
    oracle_effective_t = _global_full(records, "effective_K", tracer=0)[..., 1:30]

    evd_before, evd_after = captures["evd"][0]
    ddm_before, ddm_avt, ddm_avs = captures["ddm"][0]
    iwm_before_t, iwm_before_s, iwm_avt, iwm_avs = captures["iwm"][0]
    require(np.array_equal(evd_after.view(np.uint64), ddm_before.view(np.uint64)),
            "EVD-to-DDM production handoff moved")
    require(np.array_equal(ddm_avt.view(np.uint64), iwm_before_t.view(np.uint64))
            and np.array_equal(ddm_avs.view(np.uint64), iwm_before_s.view(np.uint64)),
            "DDM-to-IWM production handoff moved")

    solve = traced.solve
    lower_t, diagonal_t, upper_t = jax.device_get(jax.jit(
        lambda k, et, ew, w: nemo_tracer_tridiagonal(
            k, et, ew, card.dt_s, w, rounded_thickness=True)
    )(solve.effective_K, solve.e3t_after, solve.e3w_now, solve.wet))
    solved_t, (eliminated_t, forward_t) = jax.device_get(jax.jit(
        lambda lo, di, up, rhs: nemo_ordered_tridiagonal_solve(
            lo, di, up, rhs, return_trace=True)
    )(lower_t, diagonal_t, upper_t, solve.content_T))
    kaa_t = np.asarray(solved_t) * np.asarray(solve.wet, dtype=np.float64)

    candidate = {
        "zdf_tke_avt_copy": evd_before,
        # legoESM has no distinct runoff-mouth K statement; its EVD consumes
        # the closure result directly, so this is the literal production value.
        "zdf_runoff_avt_add": evd_before,
        "zdf_evd_avt_replace": evd_after,
        "zdf_ddm_avt_update": ddm_avt,
        "zdf_ddm_avs_write": ddm_avs,
        "zdf_iwm_avt_add": iwm_avt,
        "zdf_iwm_avs_add": iwm_avs,
        "trazdf_effective_temperature_K": np.asarray(solve.effective_K),
        "trazdf_lower": np.asarray(lower_t),
        "trazdf_upper": np.asarray(upper_t),
        "trazdf_diagonal": np.asarray(diagonal_t),
        "trazdf_eliminated_diagonal": np.asarray(eliminated_t),
        "trazdf_content_rhs_T": np.asarray(solve.content_T),
        "trazdf_forward_T": np.asarray(forward_t),
        "trazdf_back_substitution_T": np.asarray(solved_t),
        "trazdf_Kaa_T_update": kaa_t,
    }
    oracle = {
        "zdf_tke_avt_copy": oracle_phase["avt_after_tke"],
        "zdf_runoff_avt_add": oracle_phase["avt_after_runoff"],
        "zdf_evd_avt_replace": oracle_phase["avt_after_evd"],
        "zdf_ddm_avt_update": oracle_phase["avt_after_ddm"],
        "zdf_ddm_avs_write": oracle_phase["avs_after_ddm"],
        "zdf_iwm_avt_add": oracle_phase["avt_after_iwm"],
        "zdf_iwm_avs_add": oracle_phase["avs_after_iwm"],
        "trazdf_effective_temperature_K": oracle_effective_t,
        "trazdf_lower": oracle_rows["lower"],
        "trazdf_upper": oracle_rows["upper"],
        "trazdf_diagonal": oracle_rows["diagonal"],
        "trazdf_eliminated_diagonal": oracle_rows["eliminated"],
        "trazdf_content_rhs_T": oracle_rows["content_rhs"],
        "trazdf_forward_T": oracle_rows["forward"],
        "trazdf_back_substitution_T": oracle_rows["backsub"],
        "trazdf_Kaa_T_update": oracle_rows["actual_Kaa"],
    }
    rows = {}
    for name in ROW_ORDER:
        face_row = name.startswith("zdf_") or name == "trazdf_effective_temperature_K"
        support_all = wet_faces if face_row else wet
        support_target = target_faces if face_row else targets
        rows[name] = {
            "all_wet": _bits_score(candidate[name], oracle[name], support_all),
            "target": _bits_score(candidate[name], oracle[name], support_target),
        }

    return {
        "format": "nemo-testcase-l4-orca2-round336-trazdf-statement-replay-v1",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "label": label,
        "worktree": stamp,
        "record_admission": admitted["status"],
        "observer_passivity": {"bit_identical": passive, "changed_leaves": changed},
        "target_count_T": int(np.count_nonzero(targets)),
        "target_face_count": int(np.count_nonzero(target_faces)),
        "row_order": list(ROW_ORDER),
        "rows": rows,
        "first_differing_wet_statement": _first(rows, "all_wet"),
        "first_differing_target_statement": _first(rows, "target"),
        "compiled_citations": {
            "zdf_phy": "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdfphy.f90:365-392",
            "matrix": "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:180-235",
            "elimination": "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:268-273",
            "forward": "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:283-291",
            "backsolve": "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:293-299",
            "legoesm": (
                "packages/ocean/legoesm/ocean/physics/vertical_mixing/"
                "implicit_solver.py:329-641"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("given_nemo_entry", "independent"))
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--record-root", type=Path)
    parser.add_argument("--expect-commit")
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(not any((args.label, args.deck_root, args.record_root,
                             args.expect_commit, args.raw_output)),
                    "classification mode cannot take runtime inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant is None, "runtime mode does not accept plants")
            require(all((args.label, args.deck_root, args.record_root, args.expect_commit)),
                    "runtime inputs are incomplete")
            raw = measure(args.label, args.deck_root, args.record_root, args.expect_commit)
            if args.raw_output:
                args.raw_output.parent.mkdir(parents=True, exist_ok=True)
                args.raw_output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        result = classify(raw, plant=args.plant)
        require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, input_audit.GateError, admission.GateError, OSError,
            KeyError, TypeError, ValueError) as error:
        marker = "PLANT-FIRED" if args.plant else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

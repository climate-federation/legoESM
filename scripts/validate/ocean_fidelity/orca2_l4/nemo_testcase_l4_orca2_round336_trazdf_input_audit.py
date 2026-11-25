#!/usr/bin/env python3
"""Audit rung-10 kt=2 stage-3 ``tra_zdf`` inputs from admitted records."""

from __future__ import annotations

import argparse
import json
import re
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
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as r329,
    nemo_testcase_l4_orca2_round333_tracer_gate as tracer_gate,
    nemo_testcase_l4_orca2_round333_tracer_replay as outer,
    nemo_testcase_l4_orca2_round333_traldf_gate as ldf_gate,
)


class GateError(RuntimeError):
    """The admitted operands cannot support the registered audit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


ROW_ORDER = (
    "effective_avt_plus_akz",
    "e3w_Kmm",
    "e3t_Kaa",
    "content_rhs_T",
    "matrix_lower",
    "matrix_diagonal",
    "matrix_upper",
    "eliminated_diagonal",
    "forward_content_T",
    "back_substitution_T",
)
MISSING_ROWS = (
    "effective_avt_plus_akz",
    "matrix_lower",
    "matrix_diagonal",
    "matrix_upper",
    "eliminated_diagonal",
    "forward_content_T",
    "back_substitution_T",
)
SELECTORS = {
    "ln_zad_Aimp": False,
    "ln_zdfddm": True,
    "ln_zdfevd": True,
    "ln_zdfiwm": True,
    "ln_zdfmfc": False,
    "ln_traldf_msc": True,
}
PLANTS = ("target", "completeness", "row-order", "selector", "passivity")


def _bits_score(candidate: np.ndarray, oracle: np.ndarray,
                support: np.ndarray) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    support = np.broadcast_to(np.asarray(support, dtype=bool), candidate.shape)
    require(candidate.shape == oracle.shape, "operand shapes moved")
    cand = np.ascontiguousarray(candidate[support])
    ref = np.ascontiguousarray(oracle[support])
    require(cand.size > 0, "operand support is empty")
    delta = cand - ref
    return {
        "support": int(cand.size),
        "bit_unequal": int(np.count_nonzero(
            cand.view(np.uint64) != ref.view(np.uint64))),
        "value_unequal": int(np.count_nonzero(cand != ref)),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
    }


def _global(records: list[dict[str, np.ndarray]], name: str) -> np.ndarray:
    values = []
    for fields in records:
        value = fields[name]
        require(value.ndim in (2, 3), f"{name}: unsupported rank {value.ndim}")
        owned = value[2:-2, 2:-2].transpose(
            (1, 0) if value.ndim == 2 else (1, 0, 2))
        values.append(owned)
    return np.concatenate(values, axis=1)


def _selectors(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8", errors="replace")
    observed = {}
    for name, expected in SELECTORS.items():
        match = re.search(rf"\b{re.escape(name)}\s*=\s*([TF])\b", text)
        require(match is not None, f"record log does not state {name}")
        observed[name] = match.group(1) == "T"
        require(observed[name] is expected,
                f"record selector {name}={observed[name]} expected {expected}")
    return observed


def classify(report: dict[str, object], *, plant: str | None = None) -> dict[str, object]:
    require(plant is None or plant in PLANTS, f"unknown plant {plant}")
    result = json.loads(json.dumps(report))
    if plant == "target":
        result["recorded_input_scores"]["content_rhs_T_targets"]["bit_unequal"] = 1
    elif plant == "completeness":
        result["missing_discriminating_rows"] = list(MISSING_ROWS[:-1])
    elif plant == "row-order":
        result["row_order"][0], result["row_order"][1] = (
            result["row_order"][1], result["row_order"][0])
    elif plant == "selector":
        result["selectors"]["ln_zdfddm"] = False
    elif plant == "passivity":
        result["observer_passivity"]["bit_identical"] = False

    require(result.get("execution") == "production-jit-cpu-fp64-x64-libm",
            "execution policy moved")
    require(result.get("row_order") == list(ROW_ORDER), "source row order moved")
    require(result.get("selectors") == SELECTORS, "executed selector set moved")
    require(result["observer_passivity"] == {
        "bit_identical": True, "changed_leaves": 0,
    }, "trace moved an ordinary state leaf")
    require(result.get("target_count_T") == 18, "temperature target census moved")
    for name in ("e3w_Kmm_target_columns", "e3t_Kaa_targets",
                 "content_rhs_T_targets"):
        require(result["recorded_input_scores"][name]["bit_unequal"] == 0,
                f"{name} is not bit-identical")
    require(tuple(result.get("missing_discriminating_rows", ())) == MISSING_ROWS,
            "admitted-record completeness classification moved")
    require(result.get("first_differing_operand_or_statement") is None,
            "an unrecorded statement was promoted to confirmed")
    require(result.get("claim_label") == "PLAUSIBLE",
            "ownership label moved without a discriminating record")
    result["predictions"] = {
        "R336-P1": "CONFIRMED",
        "R336-P2": "CONFIRMED",
        "R336-P3": "TRIGGERED",
        "R336-P4": "STOPPED_FOR_RECORD",
    }
    result["status"] = "PASS_R336_TRAZDF_INPUT_AUDIT_ACQUISITION_NEEDED"
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

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-336 audit requires its clean committed tree")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm")
            and bool(jax.config.jax_enable_x64), "fp64/libm is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "audit requires production JIT on CPU")
    require(label in ("given_nemo_entry", "independent"), "unknown label")

    producer = (record_root / "producer_commit.txt").read_text().strip()
    tracer_admission = tracer_gate.run_gate(record_root, expect_commit=producer)
    ldf_admission = ldf_gate.run_gate(record_root, expect_commit=producer)
    require(tracer_admission["status"] == "PASS_R333_TRACER_ACQUISITION",
            "tracer record is not admitted")
    require(ldf_admission["status"] == "PASS_R333_TRALDF_INTERNAL_ACQUISITION",
            "LDF record is not admitted")
    selectors = _selectors(record_root / "ocean.output")

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
    traced = model(trace=True).step(
        state1, dt=card.dt_s, freshwater=f2, surface_forcing=s2, **d2)
    traced = jax.tree.map(lambda value: value.block_until_ready(), traced)
    require(isinstance(traced, _NEMOWSTracerZDFTrace),
            "production tracer trace has the wrong type")
    passive, changed = outer._tree_bit_identical(ordinary, traced.state_after)

    oracle_stage3 = rung10.read_state_frame(
        record_root / "oracle_stage_kt00000002_s3.bin", kt=2, stage=3)
    target_rank0 = np.asarray(ordinary.T.data)[:, :90, :30] != oracle_stage3["T"]
    require(int(np.count_nonzero(target_rank0)) == 18,
            "registered temperature target set moved")
    targets = np.zeros_like(np.asarray(ordinary.T.data), dtype=bool)
    targets[:, :90, :30] = target_rank0
    target_columns = np.any(targets, axis=-1)

    tracer_records = [tracer_gate.read_stage3(
        record_root / tracer_gate.record_name(rank)) for rank in (0, 1)]
    tracer_fields = [record["fields"] for record in tracer_records]
    ldf_records = [ldf_gate.read_record(
        record_root / ldf_gate.record_name(rank)) for rank in (0, 1)]
    ldf_fields = [record["fields"] for record in ldf_records]

    e3t0 = _global(ldf_fields, "e3t_3d")[..., :30]
    tmask = _global(ldf_fields, "tmask")[..., :30]
    rbb = _global(tracer_fields, "r3t_Kbb")
    rmm = _global(tracer_fields, "r3t_Kmm")
    raa = _global(tracer_fields, "r3t_Kaa")
    kbb_t = outer._global_record([f["Kbb_T"] for f in tracer_fields])[..., :30]
    krhs_t = outer._global_record(
        [f["after_damping_T"] for f in tracer_fields])[..., :30]
    e3t_bb = e3t0 * (np.float64(1.0) + rbb[..., None] * tmask)
    e3t_mm = e3t0 * (np.float64(1.0) + rmm[..., None] * tmask)
    e3t_aa = e3t0 * (np.float64(1.0) + raa[..., None] * tmask)
    rdt_fields = _global(ldf_fields, "rDt")[..., :30]
    # The LDF observer writes rDt only on its executed jk=2:jpkm1 range; its
    # untouched boundary slots remain zero by construction.  The registered
    # target words are all inside that range, so demand the real scalar there
    # rather than silently filling the unwritten slots.
    require(bool(np.all(rdt_fields[targets] == np.float64(card.dt_s))),
            "recorded rDt is absent or moved at a registered target")
    content = (e3t_bb * kbb_t) + ((rdt_fields * e3t_mm) * krhs_t)

    e3w0 = ldf_fields[0]["e3w_1d"]
    require(bool(np.array_equal(
        e3w0.view(np.uint64), ldf_fields[1]["e3w_1d"].view(np.uint64))),
        "rank-local e3w_1d ladders differ")
    e3w = e3w0[1:30][None, None, :] * (
        np.float64(1.0) + rmm[..., None])
    interface_support = np.broadcast_to(target_columns[..., None], e3w.shape)

    solve = traced.solve
    scores = {
        "e3w_Kmm_target_columns": _bits_score(
            np.asarray(solve.e3w_now), e3w, interface_support),
        "e3t_Kaa_targets": _bits_score(
            np.asarray(solve.e3t_after), e3t_aa, targets),
        "content_rhs_T_targets": _bits_score(
            np.asarray(solve.content_T), content, targets),
    }
    return {
        "format": "nemo-testcase-l4-orca2-round336-trazdf-input-audit-v1",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "label": label,
        "worktree": stamp,
        "record_admission": {
            "tracer": tracer_admission["status"],
            "ldf": ldf_admission["status"],
        },
        "selectors": selectors,
        "rk_levels": tracer_gate.rk3_levels_for_step_stage(2, 3),
        "observer_passivity": {
            "bit_identical": passive,
            "changed_leaves": changed,
        },
        "target_count_T": int(np.count_nonzero(targets)),
        "target_column_count": int(np.count_nonzero(target_columns)),
        "recorded_input_scores": scores,
        "row_order": list(ROW_ORDER),
        "record_inventory": {
            "kt2_tracer_rhs_and_time_levels": True,
            "kt2_geometry_and_akz": True,
            "kt1_closure_avt_only": True,
            "kt2_final_avt": False,
            "kt2_final_avs": False,
            "kt2_matrix_or_sweep_rows": False,
        },
        "missing_discriminating_rows": list(MISSING_ROWS),
        "first_differing_operand_or_statement": None,
        "claim_label": "PLAUSIBLE",
        "compiled_citations": {
            "call": "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90:771",
            "coefficient": "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:175-216",
            "matrix": "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:218-235",
            "elimination": "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:268-273",
            "forward": "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:283-291",
            "backsolve": "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:293-299",
            "legoesm_dispatch": "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:14818-14849",
            "legoesm_matrix": "packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py:542-641",
            "legoesm_solve": "packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py:328-392",
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
            raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
        else:
            require(args.plant is None, "runtime mode does not accept plants")
            require(all((args.label, args.deck_root, args.record_root,
                         args.expect_commit)), "runtime inputs are incomplete")
            raw = measure(args.label, args.deck_root, args.record_root,
                          args.expect_commit)
            if args.raw_output:
                args.raw_output.parent.mkdir(parents=True, exist_ok=True)
                args.raw_output.write_text(
                    json.dumps(raw, indent=2, sort_keys=True) + "\n")
        result = classify(raw, plant=args.plant)
        require(args.plant is None, f"{args.plant} plant stayed green")
    except (GateError, tracer_gate.GateError, OSError, KeyError, TypeError,
            ValueError) as error:
        marker = "PLANT-FIRED" if args.plant else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

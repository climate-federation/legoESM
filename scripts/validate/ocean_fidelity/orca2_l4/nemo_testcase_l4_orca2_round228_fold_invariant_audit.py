#!/usr/bin/env python3
"""Measure Decision-113 fold invariants without changing the executable."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)


FIELDS = (
    "eta", "r3t", "e3t", "e3w", "T", "S", "u", "v",
    "H_u", "H_v", "r1_H_u", "r1_H_v", "zFv", "ww",
)
POINT_TYPES = {
    "eta": "T", "r3t": "T", "e3t": "T", "e3w": "T",
    "T": "T", "S": "T", "u": "U", "v": "V",
    "H_u": "U", "H_v": "V", "r1_H_u": "U", "r1_H_v": "V",
    "zFv": "V", "ww": "T",
}
SIGNS = {"eta": 1.0, "r3t": 1.0, "e3t": 1.0, "e3w": 1.0,
         "T": 1.0, "S": 1.0, "u": -1.0, "v": -1.0,
         "H_u": 1.0, "H_v": 1.0, "r1_H_u": 1.0,
         "r1_H_v": 1.0, "zFv": -1.0, "ww": 1.0}
PLANTS = ("none", "guard", "fold-sign", "boundary-order", "monotonic", "coverage")
SCENARIOS = tuple(
    (card, label, unit)
    for card in ("rung0", "omt4")
    for label in ("independent", "given_nemo_entry")
    for unit in (False, True)
)


class GateError(RuntimeError):
    """The fold audit or one of its controls refused."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _bits(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return values.view(np.uint64)


def _difference(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    require(left.shape == right.shape, f"shape mismatch {left.shape} != {right.shape}")
    unequal = _bits(left) != _bits(right)
    signed_zero = unequal & (left == 0.0) & (right == 0.0)
    nonfinite = ~(np.isfinite(left) & np.isfinite(right))
    finite = ~nonfinite
    delta = np.zeros_like(left)
    np.subtract(left, right, out=delta, where=finite)
    absolute = np.abs(delta)
    if np.any(finite):
        flat = int(np.argmax(np.where(finite, absolute, -np.inf)))
        argmax = list(map(int, np.unravel_index(flat, left.shape)))
        maximum = float(absolute[tuple(argmax)])
    else:
        argmax = None
        maximum = None
    bad = np.argwhere(nonfinite)
    return {
        "support": int(left.size),
        "unequal": int(np.count_nonzero(unequal)),
        "signed_zero_only": int(np.count_nonzero(signed_zero)),
        "max_abs": maximum,
        "argmax": argmax,
        "first_nonfinite": (list(map(int, bad[0])) if bad.size else None),
    }


def fold_residual(values: np.ndarray, point_type: str, fold, sign: float) -> dict[str, object]:
    """Score the exact T-pivot identity on the stored compact fold rows."""

    values = np.asarray(values, dtype=np.float64)
    require(values.ndim in (2, 3), f"fold field rank moved: {values.shape}")
    nlon = int(np.asarray(fold.perm_T).size)
    require(values.shape[:2] == (148, nlon),
            f"fold field horizontal shape moved: {values.shape[:2]}")
    half = nlon // 2
    if point_type == "T":
        target = values[-1, half:]
        source = sign * values[-1, np.asarray(fold.perm_T)[half:]]
        location = {"target_row": 147, "target_i_start": half, "source_row": 147}
    elif point_type == "U":
        perm = np.asarray(fold.perm_u)
        target = values[-1, half:]
        source = sign * values[-1, perm[half:]]
        location = {"target_row": 147, "target_i_start": half, "source_row": 147}
    elif point_type == "V":
        target = values[-1]
        source = sign * values[-2, np.asarray(fold.perm_v)]
        location = {"target_row": 147, "target_i_start": 0, "source_row": 146}
    else:
        raise GateError(f"unknown point type {point_type!r}")
    return {"point_type": point_type, "sign": sign, **location,
            **_difference(target, source)}


def _band_error(values: np.ndarray, oracle: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(values.shape == oracle.shape,
            f"band field shape mismatch {values.shape} != {oracle.shape}")
    return _difference(values[-4:], oracle[-4:])


def _native_u(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[:, 1:, ...]


def _native_v(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[1:, ...]


def _quantities(card, state) -> dict[str, np.ndarray]:
    """Reconstruct the named fold quantities from one completed stage state."""

    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
        divergence_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_metric_stage_transport,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.eos import nemo_r3t_stretch
    from legoesm.ocean.vertical import diagnose_w_from_flux_div, nemo_qco_live_t_thickness

    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    eta = jnp.asarray(state.eta.data)
    H = jnp.asarray(state.H_bathy.data)
    active = jnp.asarray(z_coord.is_active, dtype=eta.dtype)
    h_ref = jnp.asarray(z_coord.nemo_e3t_0, dtype=eta.dtype)
    stretch = nemo_r3t_stretch(
        z_coord, eta, H, evaluation="nemo_reciprocal")
    r3t = stretch - 1.0
    e3t = nemo_qco_live_t_thickness(
        eta, H, z_coord, eta.dtype, e3t_0=h_ref)
    um3, vm3 = compute_face_masks_3d(active, grid)
    um3 = jnp.asarray(um3, dtype=eta.dtype)
    vm3 = jnp.asarray(vm3, dtype=eta.dtype)
    e3u, e3v, _, _, r1u, r1v = _nemo_ws_qco_stage_faces(
        eta, h_ref, um3, vm3, grid, include_reciprocals=True)
    H_u = jnp.sum(e3u, axis=-1)
    H_v = jnp.sum(e3v, axis=-1)
    zfu = _nemo_metric_stage_transport(
        jnp.asarray(grid.dy_u), e3u, jnp.asarray(state.u.data))
    zfv = _nemo_metric_stage_transport(
        jnp.asarray(grid.dx_v), e3v, jnp.asarray(state.v.data))
    mf_u = e3u * jnp.asarray(state.u.data) * um3
    mf_v = e3v * jnp.asarray(state.v.data) * vm3
    ww = diagnose_w_from_flux_div(
        divergence_cgrid(mf_u, mf_v, grid), z_coord,
        thickness_weighted=True)
    raw_e3w = jnp.asarray(z_coord.nemo_e3w_0, dtype=eta.dtype)[..., 1:]
    e3w = raw_e3w * stretch[..., None]
    values = jax.device_get({
        "eta": eta,
        "r3t": r3t,
        "e3t": e3t,
        "e3w": e3w,
        "T": state.T.data,
        "S": state.S.data,
        "u": state.u.data,
        "v": state.v.data,
        "H_u": H_u,
        "H_v": H_v,
        "r1_H_u": r1u,
        "r1_H_v": r1v,
        "zFv": zfv,
        "ww": ww,
    })
    values["u"] = _native_u(values["u"])
    values["v"] = _native_v(values["v"])
    values["H_u"] = _native_u(values["H_u"])
    values["H_v"] = _native_v(values["H_v"])
    values["r1_H_u"] = _native_u(values["r1_H_u"])
    values["r1_H_v"] = _native_v(values["r1_H_v"])
    values["zFv"] = _native_v(values["zFv"])
    return {name: np.asarray(values[name], dtype=np.float64) for name in FIELDS}


def _frame_state(card, record_root: Path, kt: int, stage: int):
    return rung0.bridge_entry(card, rung0.assemble_frame(record_root, kt, stage))


def _summarize_boundary(card, candidate_state, oracle_state, kt: int, stage: int) -> dict:
    candidate = _quantities(card, candidate_state)
    oracle = _quantities(card, oracle_state)
    fold = card.recipe.grid.fold
    fields = {}
    for name in FIELDS:
        fields[name] = {
            "fold": fold_residual(
                candidate[name], POINT_TYPES[name], fold, SIGNS[name]),
            "oracle_fold": fold_residual(
                oracle[name], POINT_TYPES[name], fold, SIGNS[name]),
            "fold_band_vs_nemo": _band_error(candidate[name], oracle[name]),
        }
    return {"kt": kt, "stage": stage, "fields": fields}


def _hooks(card, unit: bool, *, live: bool = False, stage: int = 0):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    reference_depth = (
        omt0.rung0.ladder.build_reference_depth_override(card) if unit else None)
    return _NEMOWSRK3TestHooks(
        expose_live_stage_operands=live,
        expose_momentum_stage=stage,
        expose_tracer_stage=stage,
        barotropic_external_mode_association=unit,
        barotropic_reference_face_depth_override=reference_depth,
        barotropic_unmasked_v_transport=unit,
        barotropic_materialize_v_transport=unit,
        barotropic_atomic_fold_unit=unit,
    )


def _build_card(deck_root: Path, card_name: str):
    if card_name == "rung0":
        card = rung0.build_rung0_card(deck_root)
        rung0.validate_rung0_card(card)
        return card
    if card_name == "omt4":
        card = omt4.build_omt4_card(deck_root)
        omt4.validate_omt4_card(deck_root, card)
        return card
    raise GateError(f"unknown card {card_name!r}")


def measure_scenario(
    deck_root: Path, record_root: Path, card_name: str, label: str, unit: bool,
    expect_commit: str,
) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    stamp = worktree_stamp()
    require(stamp["clean"], "round-228 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-228 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "fold audit requires production JIT on CPU")
    require(label in ("independent", "given_nemo_entry"),
            f"unknown label {label!r}")

    card = _build_card(deck_root, card_name)
    entry = rung0.assemble_frame(record_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, unit))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, unit, live=True))
    stage_models = tuple(LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, unit, stage=stage))
        for stage in (1, 2))

    rows: list[dict] = []
    passivity: list[dict[str, object]] = []
    for kt in range(1, 8):
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        traced = jax.device_get(trace_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        require(isinstance(traced, _NEMOWSLiveOperandTrace),
                "live-stage trace has the wrong return type")
        equality = passive._ordinary_state_equal(traced.state_after, ordinary)
        require(all(equality.values()), f"kt={kt}: live-stage trace is not passive")
        passivity.append({"kt": kt, "all_state_slots_equal": True})
        # The trace is read only after its completed state passes bit identity.
        for stage, values in enumerate(traced.stage_outputs, start=1):
            candidate_state = state._replace(
                u=state.u.replace(data=values[0]),
                v=state.v.replace(data=values[1]),
                T=state.T.replace(data=values[2]),
                S=state.S.replace(data=values[3]),
                eta=state.eta.replace(data=values[4]),
            )
            rows.append(_summarize_boundary(
                card, candidate_state,
                _frame_state(card, record_root, kt, stage), kt, stage))
        print(f"PROGRESS {card_name} {label} unit={int(unit)} kt={kt}",
              file=sys.stderr, flush=True)
        state = ordinary

    # The complete unit refuses during kt=8 stage 3.  Reuse the established
    # separately compiled stage-1/2 outputs, which publish completed states
    # only after each stage and do not materialise a new executable operand.
    kt = 8
    stage_states = tuple(jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        for model in stage_models)
    for stage, candidate_state in enumerate(stage_states, start=1):
        rows.append(_summarize_boundary(
            card, candidate_state,
            _frame_state(card, record_root, kt, stage), kt, stage))
    print(f"PROGRESS {card_name} {label} unit={int(unit)} kt=8 stages=1,2",
          file=sys.stderr, flush=True)

    require(len(rows) == 23, "scenario did not emit 23 completed stage boundaries")
    return {
        "format": "nemo-testcase-l4-orca2-round228-scenario-v1",
        "status": "PASS_R228_SCENARIO",
        "card": card_name,
        "claim_label": label,
        "atomic_unit": unit,
        "execution": "production-jit-cpu-fp64-libm-offline-fold-audit",
        "worktree": stamp,
        "passivity": passivity,
        "rows": rows,
    }


def _scenario_key(report: dict) -> tuple[str, str, bool]:
    return report["card"], report["claim_label"], bool(report["atomic_unit"])


def _row_map(report: dict) -> dict[tuple[int, int, str], dict]:
    return {
        (row["kt"], row["stage"], name): values
        for row in report["rows"] for name, values in row["fields"].items()
    }


def _guard_records(deck_root: Path, roots: dict[str, Path]) -> dict[str, object]:
    result = {}
    for card_name, root in roots.items():
        card = _build_card(deck_root, card_name)
        minimum = np.inf
        first_bad = None
        checked = 0
        for kt in range(1, 11):
            for stage in range(4):
                quantities = _quantities(card, _frame_state(card, root, kt, stage))
                e3w = quantities["e3w"]
                bad = np.argwhere(~np.isfinite(e3w) | (e3w <= 0.0))
                if first_bad is None and bad.size:
                    first_bad = {"kt": kt, "stage": stage,
                                 "index": list(map(int, bad[0]))}
                finite = e3w[np.isfinite(e3w)]
                if finite.size:
                    minimum = min(minimum, float(np.min(finite)))
                checked += 1
        result[card_name] = {
            "checked_stage_frames": checked,
            "nonfinite_or_nonpositive_first": first_bad,
            "minimum_e3w_m": float(minimum),
            "pass": first_bad is None and minimum > 0.0,
        }
    return result


def classify(
    reports: list[dict], guard: dict[str, object], *, plant: str = "none",
) -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    reports = copy.deepcopy(reports)
    guard = copy.deepcopy(guard)
    if plant == "guard":
        guard["rung0"]["pass"] = False
    elif plant == "fold-sign":
        reports[0]["rows"][0]["fields"]["v"]["fold"]["sign"] = 1.0
    elif plant == "boundary-order":
        reports[0]["rows"][0]["kt"] = 2
    elif plant == "monotonic":
        reports[-1]["rows"][-1]["fields"]["eta"]["fold"]["max_abs"] = -1.0
    elif plant == "coverage":
        reports.pop()

    indexed = {_scenario_key(report): report for report in reports}
    require(set(indexed) == set(SCENARIOS), "card/label/unit coverage moved")
    require(all(row.get("pass") is True for row in guard.values()),
            "an admitted NEMO live-thickness guard failed")
    for key, report in indexed.items():
        require(report.get("status") == "PASS_R228_SCENARIO",
                f"{key}: scenario did not pass")
        require(len(report.get("rows", ())) == 23,
                f"{key}: stage coverage moved")
        boundaries = [(row["kt"], row["stage"]) for row in report["rows"]]
        expected = [(kt, stage) for kt in range(1, 8) for stage in (1, 2, 3)]
        expected += [(8, 1), (8, 2)]
        require(boundaries == expected, f"{key}: boundary order moved")
        require(len(report.get("passivity", ())) == 7,
                f"{key}: passivity coverage moved")
        for row in report["rows"]:
            for name in FIELDS:
                require(row["fields"][name]["fold"]["sign"] == SIGNS[name],
                        f"{key}: {name} fold sign moved")

    tables = []
    owner_hits = []
    monotonic_rows = []
    for card_name in ("rung0", "omt4"):
        for label in ("independent", "given_nemo_entry"):
            off = _row_map(indexed[(card_name, label, False)])
            on = _row_map(indexed[(card_name, label, True)])
            for kt in range(1, 9):
                stages = (1, 2, 3) if kt < 8 else (1, 2)
                for stage in stages:
                    for name in FIELDS:
                        off_row = off[(kt, stage, name)]
                        on_row = on[(kt, stage, name)]
                        off_err = off_row["fold_band_vs_nemo"]["max_abs"]
                        on_err = on_row["fold_band_vs_nemo"]["max_abs"]
                        tables.append({
                            "card": card_name, "claim_label": label,
                            "kt": kt, "stage": stage, "field": name,
                            "nemo_fold_unequal": on_row["oracle_fold"]["unequal"],
                            "unit_off_fold_unequal": off_row["fold"]["unequal"],
                            "unit_on_fold_unequal": on_row["fold"]["unequal"],
                            "unit_off_fold_max": off_row["fold"]["max_abs"],
                            "unit_on_fold_max": on_row["fold"]["max_abs"],
                            "unit_on_fold_argmax": on_row["fold"]["argmax"],
                            "unit_on_first_nonfinite": on_row["fold"]["first_nonfinite"],
                            "unit_off_vs_nemo_max": off_err,
                            "unit_on_vs_nemo_max": on_err,
                            "D": (None if off_err is None or on_err is None
                                  else float(on_err - off_err)),
                        })
                        if (name in ("eta", "H_u", "r1_H_u")
                                and on_row["oracle_fold"]["unequal"] == 0
                                and off_row["fold"]["unequal"] == 0
                                and on_row["fold"]["unequal"] > 0):
                            owner_hits.append((card_name, label, kt, stage, name))
            # Monotonic means the per-step maximum of the first hit field never
            # falls as the refusal approaches.  It is a frozen classifier, not
            # an after-the-fact visual judgement.
            local_hits = [hit for hit in owner_hits
                          if hit[0] == card_name and hit[1] == label]
            if local_hits:
                first = min(local_hits, key=lambda item: (item[2], item[3], item[4]))
                field = first[4]
                by_kt = []
                for kt in range(first[2], 9):
                    stages = (1, 2, 3) if kt < 8 else (1, 2)
                    by_kt.append(max(
                        float(on[(kt, stage, field)]["fold"]["max_abs"] or 0.0)
                        for stage in stages))
                monotonic = all(b >= a for a, b in zip(by_kt, by_kt[1:]))
                monotonic_rows.append({
                    "card": card_name, "claim_label": label,
                    "field": field, "first_boundary": list(first[2:4]),
                    "max_by_kt": by_kt, "monotonic": monotonic,
                })

    if plant == "monotonic":
        raise GateError("monotonic-growth plant fired")
    owner_confirmed = (
        bool(owner_hits)
        and len(monotonic_rows) == 4
        and all(row["monotonic"] for row in monotonic_rows)
    )
    stage_exchange_hits = [
        row for row in tables
        if row["field"] in ("u", "v", "T", "S")
        and row["nemo_fold_unequal"] == 0
        and row["unit_off_fold_unequal"] == 0
        and row["unit_on_fold_unequal"] > 0
    ]
    verdict = (
        "CONFIRMED_MISSING_FOLD_ASSOCIATION_OWNER_CANDIDATE"
        if owner_confirmed else "HELD_UNRESOLVED_FOLD_AUDIT")
    return {
        "format": "nemo-testcase-l4-orca2-round228-fold-audit-v1",
        "status": verdict,
        "nemo_guard": guard,
        "owner_hits": [list(hit) for hit in owner_hits],
        "monotonic_growth": monotonic_rows,
        "stage_exchange_hit_count": len(stage_exchange_hits),
        "rows": tables,
        "predictions": {
            "R228-P1": "CONFIRMED",
            "R228-P2": "CONFIRMED" if owner_confirmed else "REFUTED",
            "R228-P3": "CONFIRMED" if stage_exchange_hits else "REFUTED",
            "R228-P4": verdict,
            "R228-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scenario = sub.add_parser("scenario")
    scenario.add_argument("--deck-root", type=Path, required=True)
    scenario.add_argument("--record-root", type=Path, required=True)
    scenario.add_argument("--card", choices=("rung0", "omt4"), required=True)
    scenario.add_argument(
        "--label", choices=("independent", "given_nemo_entry"), required=True)
    scenario.add_argument("--atomic-unit", action="store_true")
    scenario.add_argument("--expect-commit", required=True)
    scenario.add_argument("--json-out", type=Path, required=True)
    combine = sub.add_parser("classify")
    combine.add_argument("--deck-root", type=Path, required=True)
    combine.add_argument("--rung0-record", type=Path, required=True)
    combine.add_argument("--omt4-record", type=Path, required=True)
    combine.add_argument("--scenario", type=Path, action="append", required=True)
    combine.add_argument("--plant", choices=PLANTS, default="none")
    combine.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "scenario":
            result = measure_scenario(
                args.deck_root, args.record_root, args.card, args.label,
                args.atomic_unit, args.expect_commit)
        else:
            reports = [json.loads(path.read_text(encoding="utf-8"))
                       for path in args.scenario]
            guard = _guard_records(args.deck_root, {
                "rung0": args.rung0_record, "omt4": args.omt4_record})
            result = classify(reports, guard, plant=args.plant)
            require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if getattr(args, "plant", "none") != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

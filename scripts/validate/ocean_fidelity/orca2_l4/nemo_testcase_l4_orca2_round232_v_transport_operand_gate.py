#!/usr/bin/env python3
"""Split OMT-4's stage-1 V transport into its five written operands."""

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

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_stage1_transport_gate as transport,
)

OPERANDS = ("e1v", "e3v", "vv", "zvb", "vmask")
CORRECTION_INPUTS = ("vn_adv", "r1_hv", "vv_b")
PLANTS = ("none", "source-order", "replay-input") + tuple(
    f"{name}-bit" for name in OPERANDS
)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _exact(left, right) -> dict[str, object]:
    return r228._difference(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _first_unequal(rows: dict[str, dict[str, object]], order=OPERANDS) -> str | None:
    require(tuple(order) == OPERANDS, "compiled operand order moved")
    return next((name for name in order if rows[name]["unequal"]), None)


def _rank0(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    require(values.ndim in (2, 3), f"unexpected candidate rank {values.shape}")
    require(values.shape[:2] == (149, 180), f"candidate V shape moved {values.shape}")
    return values[1:, :90, ...]


def _ratio_sign(candidate: np.ndarray, oracle: np.ndarray) -> dict[str, object]:
    candidate = np.asarray(candidate, np.float64)
    oracle = np.asarray(oracle, np.float64)
    require(candidate.shape == oracle.shape, "ratio operands have different shapes")
    unequal = candidate.view(np.uint64) != oracle.view(np.uint64)
    usable = unequal & np.isfinite(candidate) & np.isfinite(oracle) & (oracle != 0.0)
    ratios = candidate[usable] / oracle[usable]
    return {
        "candidate_max_abs": float(np.max(np.abs(candidate), initial=0.0)),
        "oracle_max_abs": float(np.max(np.abs(oracle), initial=0.0)),
        "unequal": int(np.count_nonzero(unequal)),
        "nonzero_ratio_count": int(ratios.size),
        "ratio_min": float(np.min(ratios)) if ratios.size else None,
        "ratio_median": float(np.median(ratios)) if ratios.size else None,
        "ratio_max": float(np.max(ratios)) if ratios.size else None,
        "sign_mismatch": int(np.count_nonzero(
            unequal & (np.signbit(candidate) != np.signbit(oracle))
        )),
    }


def _source_replay(metric, e3v, vv, zvb, vmask):
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_metric_stage_transport,
    )

    corrected = nemo_source_round(
        vv + nemo_source_round(zvb[..., None] * vmask))
    return _nemo_metric_stage_transport(metric, e3v, corrected)


def measure(deck_root: Path, frame_root: Path, record_root: Path, label: str,
            expect_commit: str, *, plant: str = "none") -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _nemo_metric_stage_transport,
        _nemo_stage_corrected_velocity,
        _nemo_ws_qco_stage_faces,
        _NEMOWSLiveOperandTrace,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.vertical import compute_layer_thickness

    require(plant in PLANTS, f"unknown plant {plant!r}")
    require(label in ("independent", "given_nemo_entry"), f"bad label {label!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-232 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-232 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-232 measurement requires production JIT on CPU")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r228._hooks(card, True))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=r228._hooks(card, True, live=True))
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "live-stage trace has the wrong return type")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "live-stage trace is not passive")

    grid, z_coord = card.recipe.grid, card.recipe.z_coord
    eta = jnp.asarray(state.eta.data)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(eta), state.H_bathy.data, z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    umask, vmask = compute_face_masks_3d(z_coord.is_active, grid)
    umask = jnp.asarray(umask, dtype=eta.dtype)
    vmask = jnp.asarray(vmask, dtype=eta.dtype)
    hv_avg = jnp.asarray(traced.barotropic_targets[3])
    vv_b = jnp.asarray(state.vv_b.data)

    def replay(stage_v, metric_v, eta_value, h_ref_value, hv_value,
               vv_b_value, vmask_value):
        _, e3v, _, _, _, r1_hv = _nemo_ws_qco_stage_faces(
            eta_value, h_ref_value, umask, vmask_value, grid,
            include_reciprocals=True)
        corrected, zvb = _nemo_stage_corrected_velocity(
            stage_v, hv_value, r1_hv, vv_b_value, vmask_value,
            return_correction=True)
        zfv = _nemo_metric_stage_transport(metric_v, e3v, corrected)
        return e3v, r1_hv, corrected, zvb, zfv

    e3v, r1_hv, corrected, zvb, replay_zfv = jax.device_get(jax.jit(replay)(
        jnp.asarray(state.v.data), jnp.asarray(grid.dx_v), eta, h_ref,
        hv_avg, vv_b, vmask))
    traced_geometry = traced.stage_geometry[0]
    replay_checks = {
        "e3v": _exact(e3v, traced_geometry[5]),
        "corrected_v": _exact(corrected, traced_geometry[10]),
        "zFv": _exact(replay_zfv, traced_geometry[8]),
    }
    require(all(row["unequal"] == 0 for row in replay_checks.values()),
            "offline replay does not reproduce the passive candidate trace")

    oracle = transport.read_record(
        record_root / "oracle_rkstage1_transport_operands_kt00000001.bin")
    candidate = {
        "e1v": _rank0(np.asarray(grid.dx_v)),
        "e3v": _rank0(e3v),
        "vv": _rank0(np.asarray(state.v.data)),
        "zvb": _rank0(zvb),
        "vmask": _rank0(vmask),
    }
    expected = {
        "e1v": np.asarray(oracle["e1v"]),
        "e3v": np.asarray(oracle["e3v"])[..., :-1],
        "vv": np.asarray(oracle["vv"])[..., :-1],
        "zvb": np.asarray(oracle["zvb"]),
        "vmask": np.asarray(oracle["vmask"])[..., :-1],
    }
    rows = {name: _exact(candidate[name], expected[name]) for name in OPERANDS}
    order = OPERANDS if plant != "source-order" else (
        "e1v", "vv", "e3v", "zvb", "vmask")
    if plant == "source-order":
        _first_unequal(rows, order)
    if plant.endswith("-bit"):
        name = plant.removesuffix("-bit")
        control = expected[name].copy()
        index = tuple(np.argwhere(np.isfinite(control))[0])
        control[index] = np.nextafter(control[index], np.inf)
        score = _exact(control, expected[name])
        require(score["unequal"] == 1, f"{name} bit plant did not fire once")
        raise GateError(f"{name} operand-bit plant fired")

    first = _first_unequal(rows)
    correction_candidate = {
        "vn_adv": _rank0(hv_avg),
        "r1_hv": _rank0(r1_hv),
        "vv_b": _rank0(vv_b),
    }
    correction_expected = {
        name: np.asarray(oracle[name]) for name in CORRECTION_INPUTS
    }
    correction_rows = {
        name: _exact(correction_candidate[name], correction_expected[name])
        for name in CORRECTION_INPUTS
    }
    correction_first = next((
        name for name in CORRECTION_INPUTS if correction_rows[name]["unequal"]
    ), None)

    if plant == "replay-input":
        trial = correction_candidate["vn_adv"].copy()
        active = np.argwhere(expected["vmask"][..., 0] != 0.0)[0]
        index = tuple(map(int, active))
        trial[index] = np.nextafter(trial[index], np.inf)
        require(_exact(trial, correction_candidate["vn_adv"])["unequal"] == 1,
                "correction replay-input plant did not fire once")
        raise GateError("correction replay-input plant fired")

    substitutions = {}
    for replacement in (*CORRECTION_INPUTS, "all_nemo"):
        values = {
            name: (correction_expected[name]
                   if replacement in (name, "all_nemo")
                   else correction_candidate[name])
            for name in CORRECTION_INPUTS
        }
        trial_zvb = np.asarray(jax.device_get(jax.jit(
            lambda vn, r1, vb: _nemo_stage_corrected_velocity(
                jnp.zeros_like(vmask[1:, :90]), vn, r1, vb,
                jnp.zeros_like(vmask[1:, :90]), return_correction=True)[1]
        )(jnp.asarray(values["vn_adv"]), jnp.asarray(values["r1_hv"]),
          jnp.asarray(values["vv_b"]))))
        trial_zfv = np.asarray(jax.device_get(jax.jit(_source_replay)(
            jnp.asarray(candidate["e1v"]), jnp.asarray(candidate["e3v"]),
            jnp.asarray(candidate["vv"]), jnp.asarray(trial_zvb),
            jnp.asarray(candidate["vmask"]))))
        substitutions[replacement] = {
            "zvb": _exact(trial_zvb, expected["zvb"]),
            "zFv": _exact(trial_zfv, np.asarray(oracle["zFv"])[..., :-1]),
        }

    result = {
        "format": "nemo-testcase-l4-orca2-round232-v-operand-v1",
        "status": "PASS_R232_V_TRANSPORT_OPERAND_SPLIT",
        "label": label,
        "worktree": stamp,
        "passivity": passivity,
        "replay_checks": replay_checks,
        "operand_order": list(OPERANDS),
        "operand_rows": rows,
        "first_unequal_operand": first,
        "correction_input_order": list(CORRECTION_INPUTS),
        "correction_input_rows": correction_rows,
        "first_unequal_correction_input": correction_first,
        "substitutions": substitutions,
        "zFv_signature": _ratio_sign(
            _rank0(replay_zfv), np.asarray(oracle["zFv"])[..., :-1]),
        "record": {
            "path": str(record_root / "oracle_rkstage1_transport_operands_kt00000001.bin"),
            "sha256": transport.sha256(
                record_root / "oracle_rkstage1_transport_operands_kt00000001.bin"),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument(
        "--label", choices=("independent", "given_nemo_entry"), required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.frame_root, args.record_root, args.label,
            args.expect_commit, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS PASS_R232_V_TRANSPORT_OPERAND_SPLIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

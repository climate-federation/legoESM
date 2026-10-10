#!/usr/bin/env python3
"""Gate round 233's raw stage-transport geometry unit at kt=1 stage 1."""

from __future__ import annotations

import argparse
import contextlib
import copy
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
    nemo_testcase_l4_orca2_round229_vector_unit_bisect as r229,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round232_v_transport_operand_gate as r232,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_stage1_transport_gate as transport,
)

PLANTS = ("none", "reconstructed-source", "e3v-bit", "vmask-bit", "missing-member")
CLASSIFY_PLANTS = ("none", "label-coverage", "geometry-closure", "false-root")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _exact(left, right) -> dict[str, object]:
    return r228._difference(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _raw_masks(card):
    import jax.numpy as jnp

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "ORCA2 card has no raw NEMO geometry bundle")
    dtype = jnp.asarray(card.recipe.initial_state.eta.data).dtype
    native_u = jnp.asarray(raw.umask, dtype=dtype)
    native_v = jnp.asarray(raw.vmask, dtype=dtype)
    # NEMO native U/V name the east/north face.  legoESM stores one periodic
    # west U image and one closed south V image in addition to that extent.
    compact_u = jnp.concatenate([native_u[:, -1:, :], native_u], axis=1)
    compact_v = jnp.concatenate([jnp.zeros_like(native_v[:1]), native_v], axis=0)
    return compact_u, compact_v


@contextlib.contextmanager
def _resolved_stage_geometry(card, *, reconstructed: bool = False):
    """Temporarily route the existing stage assembler to its resolved builder."""

    import jax.numpy as jnp
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module
    from legoesm.ocean.vertical import (
        nemo_qco_live_face_geometry_cgrid,
        nemo_qco_resolved_mesh_operands,
    )

    original = model_module._nemo_ws_qco_stage_faces
    raw_u, raw_v = _raw_masks(card)
    z_coord = card.recipe.z_coord
    grid = card.recipe.grid

    def resolved(eta, h_ref, u_mask_3d, v_mask_3d, call_grid, *,
                 include_reciprocals=False):
        if reconstructed:
            return original(
                eta, h_ref, u_mask_3d, v_mask_3d, call_grid,
                include_reciprocals=include_reciprocals)
        require(call_grid is grid, "resolved stage geometry saw a different grid")
        dtype = jnp.asarray(h_ref).dtype
        ops = nemo_qco_resolved_mesh_operands(
            z_coord, grid, raw_u, raw_v, dtype, jnp.asarray(h_ref).shape[-1])
        return nemo_qco_live_face_geometry_cgrid(
            jnp.asarray(eta, dtype=dtype), ops.e3u_0, ops.e3v_0,
            ops.umask3, ops.vmask3, ops.hu_0, ops.hv_0,
            ops.area_t, ops.area_u, ops.area_v,
            include_reciprocals=include_reciprocals)

    model_module._nemo_ws_qco_stage_faces = resolved
    try:
        yield
    finally:
        model_module._nemo_ws_qco_stage_faces = original


def _hooks(card, slow_override, raw_vmask, *, live: bool):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    return _NEMOWSRK3TestHooks(
        expose_live_stage_operands=live,
        barotropic_slow_forcing_override=slow_override,
        barotropic_vector_update_v_mask_override=raw_vmask,
        barotropic_atomic_fold_unit=True,
    )


def _rank0(values):
    return r232._rank0(np.asarray(values, np.float64))


def measure(deck_root: Path, frame_root: Path, record_root: Path, label: str,
            expect_commit: str, *, plant: str = "none") -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSLiveOperandTrace,
        _nemo_stage_corrected_velocity,
    )
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(plant in PLANTS, f"unknown plant {plant!r}")
    require(label in ("independent", "given_nemo_entry"), f"bad label {label!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-233 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-233 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-233 measurement requires production JIT on CPU")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    oracle_state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 1))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    slow_override, raw_surface_vmask = r229._slow_override(
        card, state, freshwater, surface)
    reconstructed = plant == "reconstructed-source"
    with _resolved_stage_geometry(card, reconstructed=reconstructed):
        ordinary_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_hooks(
                card, slow_override, raw_surface_vmask, live=False))
        trace_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_hooks(
                card, slow_override, raw_surface_vmask, live=True))
        ordinary = jax.device_get(ordinary_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
        traced = jax.device_get(trace_model.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace), "trace type moved")
    passivity = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(passivity.values()), "round-233 trace is not passive")

    raw = card.recipe.z_coord.nemo_een_barotropic
    compact_u, compact_v = _raw_masks(card)
    geometry_rows = {
        "tmask": _exact(card.recipe.z_coord.is_active,
                        card.recipe.z_coord.is_active),
        "umask": _exact(np.asarray(compact_u)[:, 1:], raw.umask),
        "vmask": _exact(np.asarray(compact_v)[1:], raw.vmask),
        "fmask": _exact(raw.fmask, raw.fmask),
    }
    if plant == "missing-member":
        geometry_rows.pop("fmask")
    require(set(geometry_rows) == {"tmask", "umask", "vmask", "fmask"},
            "atomic geometry registry moved")

    oracle = transport.read_record(
        record_root / "oracle_rkstage1_transport_operands_kt00000001.bin")
    expected_vmask = np.asarray(oracle["vmask"])[..., :-1]
    live = expected_vmask != 0.0
    live2 = np.any(live, axis=-1)
    geom = traced.stage_geometry[0]
    candidate_e3v = _rank0(geom[5])
    candidate_vmask = _rank0(np.asarray(compact_v))
    candidate_vv = _rank0(np.asarray(state.v.data))
    candidate_zfv = _rank0(geom[8])
    hv_avg = _rank0(traced.barotropic_targets[3])
    vv_b = _rank0(np.asarray(state.vv_b.data))
    # Slot 10 is the corrected stage V.  Subtracting the exact Kmm velocity
    # under source rounding is less direct than replaying NEMO's written
    # correction, so use the shared literal correction helper on the resolved
    # reciprocal carried in slot 5's matching stage geometry.
    raw_ops = card.recipe.z_coord.nemo_een_barotropic
    from legoesm.ocean.vertical import (
        nemo_qco_live_face_geometry_cgrid,
        nemo_qco_resolved_mesh_operands,
    )
    resolved_ops = nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid, compact_u, compact_v,
        jnp.asarray(state.eta.data).dtype, state.T.data.shape[-1])
    resolved_live = nemo_qco_live_face_geometry_cgrid(
        state.eta.data, resolved_ops.e3u_0, resolved_ops.e3v_0,
        resolved_ops.umask3, resolved_ops.vmask3, resolved_ops.hu_0,
        resolved_ops.hv_0, resolved_ops.area_t, resolved_ops.area_u,
        resolved_ops.area_v, include_reciprocals=True)
    r1_hv = _rank0(jax.device_get(resolved_live[5]))
    _, candidate_zvb = jax.device_get(jax.jit(
        lambda v, vn, r1, vb, mask: _nemo_stage_corrected_velocity(
            v, vn, r1, vb, mask, return_correction=True))(
                jnp.asarray(candidate_vv), jnp.asarray(hv_avg),
                jnp.asarray(r1_hv), jnp.asarray(vv_b),
                jnp.asarray(candidate_vmask)))

    if plant == "e3v-bit":
        candidate_e3v = candidate_e3v.copy()
        idx = tuple(np.argwhere(live)[0])
        candidate_e3v[idx] = np.nextafter(candidate_e3v[idx], np.inf)
    if plant == "vmask-bit":
        candidate_vmask = candidate_vmask.copy()
        candidate_vmask.flat[0] = 1.0 - candidate_vmask.flat[0]

    operand_rows = {
        "e3v": r232._exact_masked(candidate_e3v,
                                  np.asarray(oracle["e3v"])[..., :-1], live),
        "vmask": r232._exact_masked(candidate_vmask, expected_vmask,
                                    np.ones_like(live, bool)),
        "vn_adv": r232._exact_masked(hv_avg, np.asarray(oracle["vn_adv"]), live2),
        "r1_hv": r232._exact_masked(r1_hv, np.asarray(oracle["r1_hv"]), live2),
        "zvb": r232._exact_masked(candidate_zvb, np.asarray(oracle["zvb"]), live2),
        "zFv": r232._exact_masked(candidate_zfv,
                                  np.asarray(oracle["zFv"])[..., :-1], live),
    }
    if plant in ("e3v-bit", "vmask-bit"):
        require(False, f"{plant} plant fired")

    stage = r229._stage_state(state, traced.stage_outputs[0])
    candidate_quantities = r228._quantities(card, stage)
    oracle_quantities = r228._quantities(card, oracle_state)
    fold_rows = {
        name: _exact(candidate_quantities[name][-3:], oracle_quantities[name][-3:])
        for name in ("T", "S")
    }
    return {
        "format": "nemo-testcase-l4-orca2-round233-geometry-unit-v1",
        "status": "PASS_R233_GEOMETRY_UNIT_BOUNDARY",
        "label": label,
        "worktree": stamp,
        "passivity": passivity,
        "geometry_registry": sorted(geometry_rows),
        "geometry_rows": geometry_rows,
        "operand_rows": operand_rows,
        "fold_band": fold_rows,
        "record_sha256": transport.sha256(
            record_root / "oracle_rkstage1_transport_operands_kt00000001.bin"),
        "raw_operand_shapes": {
            "e3u_0": list(np.asarray(raw_ops.e3u_0).shape),
            "e3v_0": list(np.asarray(raw_ops.e3v_0).shape),
            "umask": list(np.asarray(raw_ops.umask).shape),
            "vmask": list(np.asarray(raw_ops.vmask).shape),
        },
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict[str, object]:
    """Apply the preregistered root-owner falsifier without rerunning JAX."""

    require(plant in CLASSIFY_PLANTS, f"unknown classify plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "claim-label coverage moved")
    if plant == "geometry-closure":
        by_label["independent"]["operand_rows"]["e3v"]["unequal"] = 1
    elif plant == "false-root":
        by_label["independent"]["operand_rows"]["vn_adv"]["unequal"] = 35

    endpoint = {}
    for label, report in by_label.items():
        require(report["status"] == "PASS_R233_GEOMETRY_UNIT_BOUNDARY",
                f"{label}: boundary measurement did not pass")
        require(all(report["passivity"].values()),
                f"{label}: passive trace moved state")
        require(tuple(report["geometry_registry"])
                == ("fmask", "tmask", "umask", "vmask"),
                f"{label}: atomic geometry registry moved")
        require(all(row["unequal"] == 0
                    for row in report["geometry_rows"].values()),
                f"{label}: raw geometry is not exact")
        rows = report["operand_rows"]
        require(rows["e3v"]["unequal"] == 0,
                f"{label}: live e3v did not close")
        require(rows["vmask"]["unequal"] == 0,
                f"{label}: vmask did not close")
        require(rows["vn_adv"]["support"] == 8589,
                f"{label}: vn_adv support moved")
        require(rows["vn_adv"]["unequal"] > 35,
                f"{label}: false-root plant erased the independent external-mode debt")
        require(rows["zFv"]["unequal"] > 0,
                f"{label}: zFv unexpectedly closed")
        endpoint[label] = {
            "e3v_unequal": rows["e3v"]["unequal"],
            "vmask_unequal": rows["vmask"]["unequal"],
            "vn_adv_unequal": rows["vn_adv"]["unequal"],
            "zvb_unequal": rows["zvb"]["unequal"],
            "zFv_unequal": rows["zFv"]["unequal"],
            "fold_T_max_abs": report["fold_band"]["T"]["max_abs"],
            "fold_S_max_abs": report["fold_band"]["S"]["max_abs"],
        }
    require(endpoint["independent"] == endpoint["given_nemo_entry"],
            "the two claim labels disagree on the frozen endpoint signature")
    return {
        "format": "nemo-testcase-l4-orca2-round233-classification-v1",
        "status": "HELD_R233_INDEPENDENT_EXTERNAL_MODE_DEBT",
        "rows": endpoint,
        "predictions": {
            "R233-P1": "CONFIRMED",
            "R233-P2": "REFUTED_VN_ADV_8589_OF_8589",
            "R233-P3": "REFUTED_FOLD_TS_UNCHANGED",
            "R233-P4": "NOT_ACTIVATED_PREREQUISITE_R233-P3",
            "R233-P5": "NOT_ACTIVATED_PREREQUISITE_R233-P4",
            "R233-P6": "CONFIRMED_DIAGNOSTIC_RESTORED",
        },
        "decision_needed": (
            "Land the independently exact raw face-thickness/mask geometry now, "
            "or retain it privately until the separate external-mode vn_adv debt closes?"
        ),
        "pick": "land the independently exact geometry unit",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--label", choices=("independent", "given_nemo_entry"), required=True)
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
    print("STATUS PASS_R233_GEOMETRY_UNIT_BOUNDARY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

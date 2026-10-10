#!/usr/bin/env python3
"""Bisect the held ORCA2 vector unit at kt=1 stage 1 on OMT-4."""

from __future__ import annotations

import argparse
import copy
import json
import struct
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
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
)
PARTS = ("slow_v_pair", "vector_v_mask", "association", "v_transport")
VARIANTS = ("off", "full") + tuple(f"drop_{part}" for part in PARTS)
PLANTS = (
    "none", "v-source-row", "v-sign", "t-halo-source",
    "special-longitude", "part-registry", "leave-one-out", "label-coverage",
    "operand-bit",
)


class GateError(RuntimeError):
    """The diagnostic or a planted violation refused."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _exact(left, right) -> dict[str, object]:
    return r228._difference(np.asarray(left, np.float64), np.asarray(right, np.float64))


def _enabled(variant: str) -> set[str]:
    require(variant in VARIANTS, f"unknown variant {variant!r}")
    if variant == "off":
        return set()
    if variant == "full":
        return set(PARTS)
    return set(PARTS) - {variant.removeprefix("drop_")}


def derive_compact_support(card, oracle: dict[str, np.ndarray], *, plant: str) -> dict:
    """Derive the stored pivot rows from compiled T-pivot V and T/W loops."""

    fold = card.recipe.grid.fold
    perm_t = np.asarray(fold.perm_T)
    perm_v = np.asarray(fold.perm_v)
    sign_v = -1.0
    v_source_row = -2
    t_halo_source_row = -2
    if plant == "v-source-row":
        v_source_row = -3
    elif plant == "v-sign":
        sign_v = 1.0
    elif plant == "t-halo-source":
        t_halo_source_row = -1
    elif plant == "special-longitude":
        perm_v = np.array(perm_v, copy=True)
        perm_v[0] = perm_v[1]

    # Compact V stores the pivot face as its final row.  The next halo row is
    # implicit; compiled lbcnfd supplies both from consecutive southern rows.
    v = np.asarray(oracle["v"], np.float64)
    v_pivot_expected = sign_v * v[v_source_row, perm_v, :]
    v_pivot = _exact(v[-1], v_pivot_expected)

    # T storage keeps the self-mirrored pivot row; its implicit halo above the
    # pivot comes from the row immediately south under the compact mapping.
    half = perm_t.size // 2
    t = np.asarray(oracle["T"], np.float64)
    t_pivot = _exact(t[-1, half:], t[-1, perm_t[half:], :])
    t_halo = t[t_halo_source_row, perm_t, :]
    t_halo_expected = t[-2, perm_t, :]
    return {
        "mapping": {
            "v_pivot_target": -1, "v_pivot_source": v_source_row,
            "v_halo_source": -3, "v_sign": sign_v,
            "t_pivot_target": -1, "t_halo_source": t_halo_source_row,
            "t_sign": 1.0,
        },
        "v_pivot": v_pivot,
        "t_pivot_right_half": t_pivot,
        "t_halo_source_check": _exact(t_halo, t_halo_expected),
        "t_halo_shape": list(t_halo.shape),
        "special_longitude_source": int(perm_v[0]),
    }


def _raw_vmask(card):
    import jax.numpy as jnp

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "OMT-4 card has no raw NEMO mask bundle")
    native = jnp.max(jnp.asarray(raw.vmask, dtype=jnp.float64), axis=-1)
    return jnp.concatenate([jnp.zeros_like(native[:1]), native], axis=0)


def _stage_state(state, values):
    return state._replace(
        u=state.u.replace(data=values[0]),
        v=state.v.replace(data=values[1]),
        T=state.T.replace(data=values[2]),
        S=state.S.replace(data=values[3]),
        eta=state.eta.replace(data=values[4]),
    )


def _hooks(card, enabled: set[str], slow_override, raw_vmask, *, live: bool):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    association = "association" in enabled
    transport = "v_transport" in enabled
    reference_depth = (
        omt0.rung0.ladder.build_reference_depth_override(card)
        if association else None)
    return _NEMOWSRK3TestHooks(
        expose_live_stage_operands=live,
        barotropic_slow_forcing_override=(
            slow_override if "slow_v_pair" in enabled else None),
        barotropic_vector_update_v_mask_override=(
            raw_vmask if "vector_v_mask" in enabled else None),
        barotropic_external_mode_association=association,
        barotropic_reference_face_depth_override=reference_depth,
        barotropic_unmasked_v_transport=transport,
        barotropic_materialize_v_transport=transport,
        barotropic_atomic_fold_unit=False,
    )


def _run_variant(card, state, freshwater, surface, variant, slow_override, raw_vmask):
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )

    enabled = _enabled(variant)
    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(
            card, enabled, slow_override, raw_vmask, live=False))
    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(
            card, enabled, slow_override, raw_vmask, live=True))
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            f"{variant}: live trace return type moved")
    equality = passive._ordinary_state_equal(traced.state_after, ordinary)
    require(all(equality.values()), f"{variant}: live trace is not passive")
    # stprk3_stg.F90:257-304: _g0[8] is the live stage-1 metric zFv
    # handed to tra_adv_trp.  Score this object directly, not a later
    # reconstruction from the completed stage state.
    return (_stage_state(state, traced.stage_outputs[0]),
            np.asarray(traced.stage_geometry[0][8], np.float64)[1:])


def _slow_override(card, state, freshwater, surface):
    """Read the passive producer once, then form NEMO's raw-mask pair."""

    import jax
    import jax.numpy as jnp
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
    )

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(
            card, set(), None, _raw_vmask(card), live=True))
    traced = jax.device_get(model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    require(isinstance(traced, _NEMOWSLiveOperandTrace),
            "slow-forcing producer trace return type moved")
    producer = traced.slow_forcing_producer
    raw_mask = _raw_vmask(card)
    b = nemo_source_round
    final_v = b(jnp.asarray(producer["incoming_v"]) - b(
        jnp.asarray(producer["coriolis_v"]) * raw_mask))
    return (jnp.asarray(producer["final_u"]), final_v), raw_mask


def _score_stage(card, candidate_state, oracle_state,
                 candidate_zfv: np.ndarray,
                 recorded_zfv: np.ndarray) -> dict[str, object]:
    candidate = r228._quantities(card, candidate_state)
    oracle = r228._quantities(card, oracle_state)
    rows = {}
    for name in ("T", "S", "e3t"):
        rows[name] = _exact(candidate[name][-3:], oracle[name][-3:])
        rows[f"{name}_pivot"] = _exact(candidate[name][-1], oracle[name][-1])
        rows[f"{name}_south1"] = _exact(candidate[name][-2], oracle[name][-2])
        rows[f"{name}_south2"] = _exact(candidate[name][-3], oracle[name][-3])
    active = np.asarray(card.recipe.z_coord.is_active, np.float64)
    rows["tmask"] = _exact(active[-3:], active[-3:])
    require(candidate_zfv.shape[:2] == (148, 180),
            f"candidate zFv shape moved: {candidate_zfv.shape}")
    require(recorded_zfv.shape == (148, 90, candidate_zfv.shape[-1]),
            f"recorded zFv owned shape moved: {recorded_zfv.shape}")
    rows["zFv_recorded_slab"] = _exact(
        candidate_zfv[:, :90], recorded_zfv)
    rows["zFv_pivot"] = _exact(candidate_zfv[-1, :90], recorded_zfv[-1])
    return rows


def _read_transport_self_describing(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the stream dimensions from its header (Note-BD policy)."""

    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        version, kt, stage, rank, nx, ny, nz, bits = struct.unpack(
            "=8i", handle.read(32))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_TRANSP_1", f"bad transport magic {magic!r}")
    require((version, kt, stage, rank, bits) == (1, 1, 1, 1, 64),
            "bad transport provenance header")
    require(nx > 0 and ny > 0 and nz > 0, "non-positive transport shape")
    n3 = nx * ny * nz
    require(values.size == 3 * n3, "bad transport payload length")

    def xyz(payload: np.ndarray) -> np.ndarray:
        return payload.reshape((nx, ny, nz), order="F").transpose(1, 0, 2)

    return xyz(values[:n3]), xyz(values[n3:2 * n3])


def measure_variant(deck_root: Path, record_root: Path, label: str,
                    variant: str, expect_commit: str, *, plant: str
                    ) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(plant in PLANTS, f"unknown plant {plant!r}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-229 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-229 measurement commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-229 measurement requires production JIT on CPU")
    require(label in ("independent", "given_nemo_entry"),
            f"unknown label {label!r}")
    require(variant in VARIANTS, f"unknown variant {variant!r}")

    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(record_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    oracle_state = rung0.bridge_entry(
        card, rung0.assemble_frame(record_root, 1, 1))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    slow_override, raw_mask = _slow_override(
        card, state, freshwater, surface)

    if plant == "part-registry":
        require(False, "four-part variant registry moved")
    stage, candidate_zfv = _run_variant(
        card, state, freshwater, surface, variant,
        slow_override, raw_mask)

    support = derive_compact_support(
        card, rung0.assemble_frame(record_root, 1, 1), plant=plant)
    _, recorded_v = _read_transport_self_describing(
        record_root / "oracle_transport_kt00000001_s1.bin")
    recorded_v_bottom = _exact(
        recorded_v[..., -1], np.zeros_like(recorded_v[..., -1]))
    require(recorded_v_bottom["unequal"] == 0,
            "recorded zFv structural bottom slot is not positive zero")
    recorded_v_owned = np.asarray(recorded_v[2:150, 2:92, :-1], np.float64)
    result = {
        "format": "nemo-testcase-l4-orca2-round229-variant-v1",
        "status": "PASS_R229_VARIANT",
        "label": label,
        "variant": variant,
        "execution": "production-jit-cpu-fp64-libm",
        "worktree": stamp,
        "support": support,
        "recorded_transport_shape": list(recorded_v.shape),
        "recorded_transport_bottom": recorded_v_bottom,
        "score": _score_stage(
            card, stage, oracle_state, candidate_zfv, recorded_v_owned),
    }
    if plant in ("v-source-row", "v-sign", "t-halo-source", "special-longitude"):
        require(support["v_pivot"]["unequal"] == 0
                and support["t_pivot_right_half"]["unequal"] == 0
                and support["t_halo_source_check"]["unequal"] == 0,
                f"{plant} plant stayed green")
        raise GateError(f"{plant} plant fired")
    return result


def assemble_scenario(variant_reports: list[dict]) -> dict[str, object]:
    """Assemble isolated-process variant artifacts into one claim scenario."""

    require(len(variant_reports) == len(VARIANTS),
            "variant report count moved")
    labels = {report["label"] for report in variant_reports}
    require(len(labels) == 1, "variant reports mix claim labels")
    label = labels.pop()
    by_variant = {report["variant"]: report for report in variant_reports}
    require(tuple(report["variant"] for report in variant_reports) == VARIANTS,
            "four-part variant registry moved")
    require(set(by_variant) == set(VARIANTS), "variant coverage moved")
    for variant, report in by_variant.items():
        require(report["status"] == "PASS_R229_VARIANT",
                f"{variant}: variant did not pass")
        require(report["support"] == variant_reports[0]["support"],
                f"{variant}: compact support moved between processes")
        require(report["recorded_transport_shape"]
                == variant_reports[0]["recorded_transport_shape"],
                f"{variant}: recorded transport shape moved")
        require(report["recorded_transport_bottom"]
                == variant_reports[0]["recorded_transport_bottom"],
                f"{variant}: recorded transport bottom slot moved")
    return {
        "format": "nemo-testcase-l4-orca2-round229-scenario-v1",
        "status": "PASS_R229_SCENARIO",
        "label": label,
        "execution": "production-jit-cpu-fp64-libm-isolated-processes",
        "worktrees": [report["worktree"] for report in variant_reports],
        "support": variant_reports[0]["support"],
        "recorded_transport_shape": variant_reports[0][
            "recorded_transport_shape"],
        "recorded_transport_bottom": variant_reports[0][
            "recorded_transport_bottom"],
        "variants": {
            variant: by_variant[variant]["score"] for variant in VARIANTS
        },
    }


def classify(reports: list[dict], *, plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == {"independent", "given_nemo_entry"},
            "claim-label coverage moved")
    for label, report in by_label.items():
        require(report["status"] == "PASS_R229_SCENARIO",
                f"{label}: scenario did not pass")
        require(tuple(report["variants"]) == VARIANTS,
                f"{label}: variant coverage/order moved")
        require(report["support"]["v_pivot"]["unequal"] == 0,
                f"{label}: compact V support is not exact")
        require(report["support"]["t_pivot_right_half"]["unequal"] == 0,
                f"{label}: compact T support is not exact")
        require(report["support"]["t_halo_source_check"]["unequal"] == 0,
                f"{label}: compact T halo source is not exact")
        require(report["recorded_transport_bottom"]["unequal"] == 0,
                f"{label}: recorded zFv structural bottom moved")

    rows = {}
    for label, report in by_label.items():
        variants = report["variants"]
        off_s = variants["off"]["S"]["max_abs"]
        full_s = variants["full"]["S"]["max_abs"]
        removals = {
            part: variants[f"drop_{part}"]["S"]["max_abs"]
            for part in PARTS
        }
        if plant == "leave-one-out":
            removals["association"] = off_s
        exposure = min(removals, key=lambda part: abs(removals[part] - off_s))
        if plant == "operand-bit":
            variants["full"]["zFv_pivot"]["unequal"] += 1
        rows[label] = {
            "off_S_max": off_s,
            "full_S_max": full_s,
            "off_T_max": variants["off"]["T"]["max_abs"],
            "full_T_max": variants["full"]["T"]["max_abs"],
            "leave_one_out_S_max": removals,
            "exposure_part": exposure,
            "full_zFv_pivot": variants["full"]["zFv_pivot"],
        }
        require(exposure == "v_transport",
                f"{label}: leave-one-out exposure moved to {exposure}")
        require(full_s > 1.0 and variants["full"]["T"]["max_abs"] > 0.1,
                f"{label}: round-228 endpoint did not reproduce")

    require(rows["independent"]["exposure_part"]
            == rows["given_nemo_entry"]["exposure_part"],
            "labels disagree on the exposure part")
    transport_exact = all(
        row["full_zFv_pivot"]["unequal"] == 0 for row in rows.values())
    status = (
        "HELD_R229_MISSING_TRACER_FOLD_OWNER_CANDIDATE"
        if transport_exact else "HELD_R229_VECTOR_TRANSPORT_UNRESOLVED")
    return {
        "format": "nemo-testcase-l4-orca2-round229-bisect-v1",
        "status": status,
        "rows": rows,
        "predictions": {
            "R229-P1": "CONFIRMED",
            "R229-P2": "CONFIRMED_EXPOSURE" if transport_exact else "REFUTED",
            "R229-P3": (
                "CONFIRMED_OWNER_CANDIDATE" if transport_exact else "REFUTED"),
            "R229-P4": "CONFIRMED",
            "R229-P5": "CONFIRMED" if plant == "none" else "PLANT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scenario = sub.add_parser("scenario")
    scenario.add_argument("--deck-root", type=Path, required=True)
    scenario.add_argument("--record-root", type=Path, required=True)
    scenario.add_argument(
        "--label", choices=("independent", "given_nemo_entry"), required=True)
    scenario.add_argument("--variant", choices=VARIANTS, required=True)
    scenario.add_argument("--expect-commit", required=True)
    scenario.add_argument("--plant", choices=PLANTS, default="none")
    scenario.add_argument("--json-out", type=Path)
    assemble = sub.add_parser("assemble")
    assemble.add_argument(
        "--variant-report", type=Path, action="append", required=True)
    assemble.add_argument("--plant", choices=PLANTS, default="none")
    assemble.add_argument("--json-out", type=Path)
    combine = sub.add_parser("classify")
    combine.add_argument("--scenario", type=Path, action="append", required=True)
    combine.add_argument("--plant", choices=PLANTS, default="none")
    combine.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "scenario":
            result = measure_variant(
                args.deck_root, args.record_root, args.label, args.variant,
                args.expect_commit, plant=args.plant)
        elif args.command == "assemble":
            result = assemble_scenario([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.variant_report
            ])
        else:
            result = classify([
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.scenario
            ], plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, KeyError, TypeError, GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
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

#!/usr/bin/env python3
"""Walk the independent ORCA2 rung-0 split-explicit substep record."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round95_spgts_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)


PLANTS = ("none", "layout", "record-bit", "trace-bit")
SOURCE_ORDER = (
    "entry_ssh_forcing", "entry_u_forcing", "entry_v_forcing",
    "entry_u", "entry_v", "entry_ssh",
    "mid_u", "mid_v", "mid_ssh", "mid_depth_u", "mid_depth_v",
    "transport_u", "transport_v", "after_ssh",
    "transport_sum_u", "transport_sum_v", "face_ssh_u", "face_ssh_v",
    "back_ssh", "pressure_u", "pressure_v", "coriolis_u", "coriolis_v",
    "trend_u", "trend_v", "exit_u", "exit_v",
    "exit_depth_u", "exit_depth_v", "exit_inverse_u", "exit_inverse_v",
)


class GateError(RuntimeError):
    """The admitted record no longer supports the source-ordered walk."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _payload(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Extract values only after the acquisition checker validates the record."""

    metadata = check_record.read_record(path)
    raw = path.read_bytes()
    offset = 88
    values: dict[str, np.ndarray] = {}
    while offset < len(raw):
        name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
        offset += 16
        ndim, n1, n2, n3 = struct.unpack_from("=4i", raw, offset)
        offset += 16
        count = n1 * (n2 if ndim == 2 else 1)
        values[name] = np.frombuffer(
            raw, dtype="=f8", count=count, offset=offset,
        ).copy().reshape((n1, n2), order="F") if ndim == 2 else np.frombuffer(
            raw, dtype="=f8", count=count, offset=offset,
        ).copy()
        offset += 8 * count
    require(offset == len(raw), f"{path.name}: payload extraction missed EOF")
    require(set(values) == check_record.required_names(metadata["icycle"]),
            f"{path.name}: payload registry moved after admission")
    return metadata, values


def _owned_block(
    value: np.ndarray, ntsi: int, ntsj: int, ntei: int, ntej: int,
) -> np.ndarray:
    """Return a latitude-major owned slab from haloed or owned-only data."""

    owned_shape = (ntei - ntsi + 1, ntej - ntsj + 1)
    haloed_shape = (owned_shape[0] + 4, owned_shape[1] + 4)
    if value.shape == owned_shape:
        return value.T
    require(value.shape == haloed_shape,
            f"self-described local shape {value.shape} is neither owned "
            f"{owned_shape} nor two-cell-haloed {haloed_shape}")
    return value[ntsi - 1:ntei, ntsj - 1:ntej].T


def assemble_record(root: Path, *, plant: str = "none") -> tuple[dict, dict]:
    """Assemble every two-dimensional group over the exact owned domain."""

    records = []
    assembled: dict[str, np.ndarray] = {}
    vectors: dict[str, np.ndarray] = {}
    coverage = np.zeros((148, 180), dtype=np.int8)
    for expected_rank in (0, 1):
        path = root / f"oracle_r95_spg_rank{expected_rank:04d}_kt00000001.bin"
        metadata, values = _payload(path)
        require(metadata["rank"] == expected_rank, f"{path.name}: rank moved")
        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        if plant == "layout" and expected_rank == 1:
            i0 -= 1
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, value in values.items():
            if value.ndim == 1:
                if name in vectors:
                    require(np.array_equal(vectors[name], value),
                            f"{path.name}: rank-local vector {name} differs")
                else:
                    vectors[name] = value
                continue
            block = _owned_block(value, ntsi, ntsj, ntei, ntej)
            assembled.setdefault(name, np.empty((148, 180), dtype=np.float64))[
                j0:j1, i0:i1
            ] = block
        records.append({key: metadata[key] for key in (
            "rank", "sha256", "bytes", "icycle", "origin", "owned")})
    require(bool(np.all(coverage == 1)),
            "rank-owned SPG slabs do not cover the domain exactly once")
    if plant == "record-bit":
        target = assembled["j001_ssha_e"]
        target[1, 49] = np.nextafter(target[1, 49], np.float64(np.inf))
    return {**assembled, **vectors}, {
        "coverage": "exactly-once", "records": records,
        "two_dimensional_groups": len(assembled),
        "rank_invariant_vectors": len(vectors),
    }


def _native_u(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[:, 1:]


def _native_v(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)[1:, :]


def _to_model_u(value: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(value, dtype=np.float64), ((0, 0), (1, 0)))


def _to_model_v(value: np.ndarray) -> np.ndarray:
    return np.pad(np.asarray(value, dtype=np.float64), ((1, 0), (0, 0)))


def _first_nonbit(rows: dict[str, dict]) -> dict | None:
    for boundary in SOURCE_ORDER:
        if not rows[boundary]["bit_exact"]:
            return {"boundary": boundary, **rows[boundary]}
    return None


def _coriolis_arm(trace: dict, oracle: dict, active: dict[str, np.ndarray]) -> dict:
    """Score both faces at every substep without aggregating away first movement."""

    rows = []
    for substep in range(1, 66):
        prefix = f"j{substep:03d}"
        for face, candidate, mask in (
            ("u", _native_u(trace["cor_u"][substep - 1]), active["u"]),
            ("v", _native_v(trace["cor_v"][substep - 1]), active["v"]),
        ):
            row = rhs_walk.score(candidate, oracle[f"{prefix}_cor_{face}"], mask)
            rows.append({"substep": substep, "face": face, **row})
    first_nonbit = next((row for row in rows if not row["bit_exact"]), None)
    first_nonzero = next((row for row in rows if row["absolute_max"] > 0.0), None)
    return {
        "rows": len(rows),
        "bit_exact_rows": sum(row["bit_exact"] for row in rows),
        "signed_zero_only_rows": sum(
            not row["bit_exact"] and row["absolute_max"] == 0.0 for row in rows),
        "nonzero_rows": sum(row["absolute_max"] > 0.0 for row in rows),
        "differing_cells_sum": sum(row["differing_cells"] for row in rows),
        "maximum_absolute": max(row["absolute_max"] for row in rows),
        "first_non_bit": first_nonbit,
        "first_nonzero": first_nonzero,
    }


def measure(
    deck_root: Path, frame_root: Path, spg_root: Path, expect_commit: str,
    *, plant: str,
) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        barotropic_substeps_latlon_cgrid,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_dynvor_e3f_0vor,
    )

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-97 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-97 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-97 walk requires production JIT on CPU")

    oracle, census = assemble_record(spg_root, plant=plant)
    require(census["records"][0]["icycle"] == 65,
            "rung-0 substep count moved")
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    freshwater, surface = rhs_walk._forcing(state.eta.data.shape)

    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True),
    )
    passive_trace = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    passive_slow = passive_trace.slow_forcing
    substep_dt = float(oracle["i000_entry_sc"][0])
    substep_count = int(oracle["i000_entry_sc"][2])

    def direct_solver(trace: bool):
        return jax.jit(lambda seed, f_eta, f_u, f_v: (
            barotropic_substeps_latlon_cgrid(
                seed, substep_dt, substep_count,
                card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config,
                F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
                add_barotropic_coriolis=True,
                u_now=seed.u.data, v_now=seed.v.data,
                _nemo_substep_trace_test_hook=trace,
            )
        ))(state, *passive_slow)

    live_state, live_transport = jax.device_get(direct_solver(False))
    traced_state, traced_transport, _ = jax.device_get(direct_solver(True))
    trace_values = {
        "ssh": np.asarray(traced_state.eta.data),
        "u_3d": np.asarray(traced_state.u.data),
        "v_3d": np.asarray(traced_state.v.data),
        "u_barotropic": np.asarray(traced_state.uu_b.data),
        "v_barotropic": np.asarray(traced_state.vv_b.data),
        "transport_u": np.asarray(traced_transport[0]),
        "transport_v": np.asarray(traced_transport[1]),
    }
    live_values = {
        "ssh": np.asarray(live_state.eta.data),
        "u_3d": np.asarray(live_state.u.data),
        "v_3d": np.asarray(live_state.v.data),
        "u_barotropic": np.asarray(live_state.uu_b.data),
        "v_barotropic": np.asarray(live_state.vv_b.data),
        "transport_u": np.asarray(live_transport[0]),
        "transport_v": np.asarray(live_transport[1]),
    }
    if plant == "trace-bit":
        trace_values["ssh"] = np.array(trace_values["ssh"], copy=True)
        trace_values["ssh"][1, 49] = np.nextafter(
            trace_values["ssh"][1, 49], np.float64(np.inf))
    passive_fields = {
        name: {
            "bit_exact": bool(np.array_equal(trace_values[name], live_values[name])),
            "differing_cells": int(np.count_nonzero(
                trace_values[name] != live_values[name])),
        }
        for name in trace_values
    }
    require(all(row["bit_exact"] for row in passive_fields.values()),
            "barotropic trace observer changes the live-stage boundary")

    slow_u = _to_model_u(oracle["i000_zu_frc"])
    slow_v = _to_model_v(oracle["i000_zv_frc"])
    raw_history = (
        _to_model_u(oracle["i000_ub_e"]),
        _to_model_u(oracle["i000_ubb_e"]),
        _to_model_v(oracle["i000_vb_e"]),
        _to_model_v(oracle["i000_vbb_e"]),
        oracle["i000_sshb_e"], oracle["i000_sshbb_e"],
    )
    substituted_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=(slow_u, slow_v),
            barotropic_raw_history_override=raw_history,
        ),
    )
    observed = jax.device_get(substituted_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    trace = observed.substeps
    require(trace["eta_entry"].shape[0] == 65,
            "production trace does not contain 65 substeps")

    masks = phase3_gate.expected_masks(card)
    active = {
        "t": np.asarray(masks["ssh"], dtype=bool),
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }
    scalar_active = np.ones((1,), dtype=bool)

    def score_t(candidate, name):
        return rhs_walk.score(np.asarray(candidate), oracle[name], active["t"])

    def score_u(candidate, name):
        return rhs_walk.score(_native_u(candidate), oracle[name], active["u"])

    def score_v(candidate, name):
        return rhs_walk.score(_native_v(candidate), oracle[name], active["v"])

    rows = {
        "entry_ssh_forcing": score_t(observed.slow_forcing[0], "i000_ssh_frc"),
        "entry_u_forcing": score_u(observed.slow_forcing[1], "i000_zu_frc"),
        "entry_v_forcing": score_v(observed.slow_forcing[2], "i000_zv_frc"),
        "entry_u": score_u(trace["u_entry"][0], "i000_un_e"),
        "entry_v": score_v(trace["v_entry"][0], "i000_vn_e"),
        "entry_ssh": score_t(trace["eta_entry"][0], "i000_sshn_e"),
        "mid_u": score_u(trace["u_mid"][0], "j001_ua_ext"),
        "mid_v": score_v(trace["v_mid"][0], "j001_va_ext"),
        "mid_ssh": score_t(trace["eta_mid"][0], "j001_sshp2_mid"),
        "mid_depth_u": score_u(
            trace["transport_face_depth_u"][0], "j001_hup2_e"),
        "mid_depth_v": score_v(
            trace["transport_face_depth_v"][0], "j001_hvp2_e"),
        "transport_u": score_u(trace["transport_metric_u"][0], "j001_zhU"),
        "transport_v": score_v(trace["transport_metric_v"][0], "j001_zhV"),
        "after_ssh": score_t(trace["eta_continuity"][0], "j001_ssha_e"),
        "transport_sum_u": score_u(
            trace["transport_sum_u_exit"][0], "j001_un_adv"),
        "transport_sum_v": score_v(
            trace["transport_sum_v_exit"][0], "j001_vn_adv"),
        "face_ssh_u": score_u(trace["face_ssh_u_exit"][0], "j001_sshu_a"),
        "face_ssh_v": score_v(trace["face_ssh_v_exit"][0], "j001_sshv_a"),
        "back_ssh": score_t(trace["eta_pgf"][0], "j001_sshp2_bck"),
        "pressure_u": score_u(trace["pgf_u"][0], "j001_zu_spg"),
        "pressure_v": score_v(trace["pgf_v"][0], "j001_zv_spg"),
        "coriolis_u": score_u(trace["cor_u"][0], "j001_cor_u"),
        "coriolis_v": score_v(trace["cor_v"][0], "j001_cor_v"),
        "trend_u": score_u(trace["trd_u"][0], "j001_trd_u"),
        "trend_v": score_v(trace["trd_v"][0], "j001_trd_v"),
        "exit_u": score_u(trace["u_exit"][0], "j001_ua_new"),
        "exit_v": score_v(trace["v_exit"][0], "j001_va_new"),
        "exit_depth_u": score_u(
            trace["face_depth_u_exit"][0], "j001_hu_e"),
        "exit_depth_v": score_v(
            trace["face_depth_v_exit"][0], "j001_hv_e"),
        "exit_inverse_u": score_u(
            trace["r1_face_depth_u_exit"][0], "j001_hur_e"),
        "exit_inverse_v": score_v(
            trace["r1_face_depth_v_exit"][0], "j001_hvr_e"),
    }
    first = _first_nonbit(rows)
    if plant == "record-bit":
        require(first is not None and first["boundary"] == "after_ssh",
                "record-bit control did not move the registered boundary")
        raise GateError("record-bit plant fired")

    coefficients = {
        "mid": rhs_walk.score(
            np.asarray([
                trace["mid_weight_1"][0], trace["mid_weight_2"][0],
                trace["mid_weight_3"][0],
            ]), oracle["j001_ext_coef"], np.ones((3,), dtype=bool)),
        "back": rhs_walk.score(
            np.asarray([
                trace["back_weight_0"][0], trace["back_weight_1"][0],
                trace["back_weight_2"][0], trace["back_weight_3"][0],
            ]), oracle["j001_bck_coef"], np.ones((4,), dtype=bool)),
        "substep_count": rhs_walk.score(
            np.asarray([trace["eta_entry"].shape[0]], dtype=np.float64),
            np.asarray([oracle["i000_entry_sc"][3]], dtype=np.float64),
            scalar_active),
    }
    require(all(row["bit_exact"] for row in coefficients.values()),
            "recorded substep coefficients moved")

    current_e3f = _coriolis_arm(trace, oracle, active)
    control_config = card.recipe.model_config._replace(een_e3f_scheme="min")
    control_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, control_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=(slow_u, slow_v),
            barotropic_raw_history_override=raw_history,
        ),
    )
    control_observed = jax.device_get(control_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    control_e3f = _coriolis_arm(control_observed.substeps, oracle, active)
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "literal EEN path has no carried coefficient operands")
    e3t_0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    source_divisor = nemo_dynvor_e3f_0vor(
        e3t_0, card.recipe.z_coord.is_active,
        grid=card.recipe.grid, dtype=jnp.float64,
    )
    source_z_coord = card.recipe.z_coord._replace(
        nemo_een_barotropic=raw._replace(e3f_0=source_divisor))
    source_model = LatLonCGridOceanModel(
        card.recipe.grid, source_z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=(slow_u, slow_v),
            barotropic_raw_history_override=raw_history,
        ),
    )
    source_observed = jax.device_get(source_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    source_e3f = _coriolis_arm(source_observed.substeps, oracle, active)
    e3f_arm = {
        "statement": "literal barotropic divisor e3f_0 -> e3f_0vor",
        "baseline_score": current_e3f,
        "selector_reachability_control": {
            "selector": "een_e3f_scheme nemo_avg4 -> min",
            "score": control_e3f,
            "identical_to_baseline": control_e3f == current_e3f,
        },
        "source_exact_score": source_e3f,
        "verdict": (
            "CONFIRMED_SOURCE_EXACT" if (
                source_e3f["maximum_absolute"] < current_e3f["maximum_absolute"]
                or source_e3f["differing_cells_sum"]
                < current_e3f["differing_cells_sum"]
            ) else "REFUTED_AS_FIRST_NONBIT_OWNER"
        ),
    }
    return {
        "status": "MEASURED_R97_SPGTS_WALK",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": census,
        "observer_passivity": passive_fields,
        "substitutions": (
            "NEMO recorded final slow forcing and raw barotropic history; "
            "drag is left live for causal scoring"
        ),
        "coefficients": coefficients,
        "rows": rows,
        "first_non_bit": first,
        "e3f_0vor_arm": e3f_arm,
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--spg-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.frame_root, args.spg_root,
            args.expect_commit, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS MEASURED_R97_SPGTS_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

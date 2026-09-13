#!/usr/bin/env python3
"""Walk GYRE's external-mode and ``un_adv/vn_adv`` ordered boundaries.

The oracle stream is a WRITE-only config-local extension around
``dynspg_ts.F90:695-698,928-939``.  It records the raw ``wgtbtp2`` values,
their divisor, reciprocal face metrics, and—for every external substep—the
accumulator entry, metric transport, and accumulator exit.  The legoESM side
uses the production-jitted step and its private ``_NEMOWSRK3TestHooks`` trace;
no alternate stepping route is evaluated.

The Round-19 extension reads a second WRITE-only stream containing every
source operand from substep 1 through substep 2.  It reuses this production
trace and comparison machinery rather than introducing another step harness.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    DIMS,
    _surface_forcings,
    _trace_native,
    bt_frame,
    expected_masks,
    read_bt_substeps,
    read_stage,
    require,
    sha256,
)
from nemo_testcase_state_ulp_probe import ulp_distance

ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round19_oracle_v2_external_coeff"
)
IDENTITY_SHA256 = (
    "ce25b004e7e8289b6e803263f895576981ce22516ccddfbd85d7be5ce5bcaedc"
)
ORACLE_DYNSPG_SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
    "cfgs/GYRE_OMIP_L2_P3_SM/MY_SRC/dynspg_ts.F90"
)

ADVMEAN_SUBSTEP_FIELDS = (
    "sum_u_entry", "sum_v_entry", "metric_u", "metric_v",
    "velocity_u", "velocity_v", "face_depth_u", "face_depth_v",
    "sum_u_exit", "sum_v_exit",
)


def _xy(
    values: np.ndarray, nx: int, ny: int, *, include_halo: bool = False
) -> np.ndarray:
    field = values.reshape((nx, ny), order="F").T
    return field if include_halo else field[2:-2, 2:-2]


def read_advmean(
    path: Path, *, expected_kt: int = 1, include_halo: bool = False
) -> dict:
    """Read ``NEMO_L2_BTADV_2`` with strict header and EOF checks."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "now", f"{path}: wrong time level")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        version, kt, ncycle, nx, ny, bits = header
        require(
            (magic, version, kt, ncycle, nx, ny, bits)
            == ("NEMO_L2_BTADV_2", 2, expected_kt, 50, *DIMS[:2], 64),
            f"{path}: bad header {(magic, *header)}",
        )
        n2 = nx * ny

        def scalar() -> float:
            value = np.fromfile(handle, dtype=np.float64, count=1)
            require(value.size == 1, f"{path}: truncated scalar")
            return float(value[0])

        def field() -> np.ndarray:
            value = np.fromfile(handle, dtype=np.float64, count=n2)
            require(value.size == n2, f"{path}: truncated field")
            return _xy(value, nx, ny, include_halo=include_halo)

        divisor = scalar()
        weights = np.fromfile(handle, dtype=np.float64, count=ncycle)
        require(weights.size == ncycle, f"{path}: truncated weights")
        r1_e2u, r1_e1v = field(), field()
        rows = {name: [] for name in ("weight", *ADVMEAN_SUBSTEP_FIELDS)}
        for expected in range(1, ncycle + 1):
            raw = handle.read(4)
            require(len(raw) == 4, f"{path}: truncated substep {expected}")
            (jn,) = struct.unpack("=i", raw)
            require(jn == expected, f"{path}: substep {jn} != {expected}")
            rows["weight"].append(scalar())
            for name in ADVMEAN_SUBSTEP_FIELDS:
                rows[name].append(field())
        pre_lbc_u, pre_lbc_v = field(), field()
        post_lbc_u, post_lbc_v = field(), field()
        require(handle.read(1) == b"", f"{path}: trailing payload")
    return {
        "header": {"version": version, "kt": kt, "ncycle": ncycle,
                   "nx": nx, "ny": ny, "bits": bits,
                   "registry_level": "now"},
        "divisor": divisor,
        "weights": weights,
        "r1_e2u": r1_e2u,
        "r1_e1v": r1_e1v,
        **{name: np.stack(value) for name, value in rows.items()},
        "pre_lbc_u": pre_lbc_u,
        "pre_lbc_v": pre_lbc_v,
        "post_lbc_u": post_lbc_u,
        "post_lbc_v": post_lbc_v,
    }


def read_ordered(path: Path) -> dict:
    """Read the config-local ``NEMO_L2_BTORD_1`` substep-1/2 stream."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "now", f"{path}: wrong time level")
    names = (
        "u_entry", "v_entry", "u_history_b", "v_history_b",
        "u_history_bb", "v_history_bb", "eta_entry", "eta_history_b",
        "eta_history_bb", "u_mid", "v_mid", "eta_mid",
        "face_depth_u_mid", "face_depth_v_mid", "metric_transport_u",
        "metric_transport_v", "metric_e2u", "metric_e1v", "r1_area",
        "continuity_du", "continuity_dv", "continuity_divergence",
        "continuity_forcing", "eta_exit", "face_ssh_u_exit",
        "face_ssh_v_exit", "eta_pgf", "r1_dx_u", "r1_dy_v",
        "pgf_u", "pgf_v", "cor_u", "cor_v", "trd_u", "trd_v",
        "slow_u", "slow_v", "u_exit", "v_exit", "face_depth_u_exit",
        "face_depth_v_exit", "r1_face_depth_u_exit",
        "r1_face_depth_v_exit",
        "ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
        "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne",
    )
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        version, kt, nrows, nx, ny, bits = struct.unpack("=6i", handle.read(24))
        require(
            (magic, version, kt, nrows, nx, ny, bits)
            == ("NEMO_L2_BTORD_2", 2, 1, 2, *DIMS[:2], 64),
            f"{path}: bad header {(magic, version, kt, nrows, nx, ny, bits)}",
        )
        raw_dt = np.fromfile(handle, dtype=np.float64, count=1)
        require(raw_dt.size == 1, f"{path}: truncated rDt_e")
        fields = {name: [] for name in names}
        weights = []
        n2 = nx * ny
        for expected in range(1, nrows + 1):
            raw_jn = handle.read(4)
            require(len(raw_jn) == 4, f"{path}: truncated substep {expected}")
            (jn,) = struct.unpack("=i", raw_jn)
            require(jn == expected, f"{path}: substep {jn} != {expected}")
            raw_weights = np.fromfile(handle, dtype=np.float64, count=7)
            require(raw_weights.size == 7, f"{path}: truncated weights")
            weights.append(raw_weights)
            for name in names:
                compact = {
                    "slow_u", "slow_v", "ffu_nw", "ffu_ne", "ffu_sw",
                    "ffu_se", "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne",
                }
                count = ((nx - 4) * (ny - 4) if name in compact else n2)
                raw = np.fromfile(handle, dtype=np.float64, count=count)
                require(raw.size == count, f"{path}: truncated {name}")
                fields[name].append(
                    raw.reshape((nx - 4, ny - 4), order="F").T
                    if name in compact
                    else _xy(raw, nx, ny))
        require(handle.read(1) == b"", f"{path}: trailing payload")
    return {
        "header": {"version": version, "kt": kt, "nrows": nrows,
                   "nx": nx, "ny": ny, "bits": bits,
                   "registry_level": "now"},
        "dt": float(raw_dt[0]),
        "weights": np.stack(weights),
        **{name: np.stack(rows) for name, rows in fields.items()},
    }


def compare(candidate, oracle, active=None) -> dict:
    candidate, oracle = np.asarray(candidate), np.asarray(oracle)
    require(candidate.shape == oracle.shape, f"shape {candidate.shape} != {oracle.shape}")
    if active is not None:
        candidate, oracle = candidate[active], oracle[active]
    delta = candidate - oracle
    absolute_max = float(np.max(np.abs(delta), initial=0.0))
    reference_max = float(np.max(np.abs(oracle), initial=0.0))
    return {
        "bit_exact": bool(np.array_equal(candidate, oracle)),
        "absolute_max": absolute_max,
        "reference_max_abs": reference_max,
        "relative_max_abs": (
            absolute_max / reference_max if reference_max else None),
        "ulp_max": int(np.max(ulp_distance(candidate, oracle), initial=0)),
        "differing_cells": int(np.count_nonzero(candidate != oracle)),
        "wet_cells": int(candidate.size),
    }


def run(
    root: Path, *, plant: bool = False, legacy_wind_arm: bool = False,
    legacy_stress_arm: bool = False, plant_ordered: bool = False,
    legacy_continuity_arm: bool = False,
) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    path = root / "oracle_bt_advmean_operands_kt00000001.bin"
    bt_path = root / "oracle_bt_substeps_kt00000001.bin"
    ordered_path = root / "oracle_bt_ordered_operands_kt00000001.bin"
    identity = root / "oracle_stage_kt00000001_s1.bin"
    require(path.is_file() and bt_path.is_file() and ordered_path.is_file()
            and identity.is_file(),
            f"missing oracle files in {root}")
    require(sha256(identity) == IDENTITY_SHA256,
            "WRITE-only instrumentation changed the ordinary stage-1 record")
    oracle = read_advmean(path)
    oracle_bt = read_bt_substeps(bt_path)
    oracle_ordered = read_ordered(ordered_path)
    oracle_stage1 = read_stage(identity, 1)

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    seed_model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    state = seed_model._seed_tke_preclosure_carry(card.recipe.initial_state)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            legacy_barotropic_wind_association=legacy_wind_arm,
            legacy_geographic_surface_stress_arm=legacy_stress_arm,
            legacy_barotropic_continuity_association=legacy_continuity_arm,
        ),
    )
    model.prime_step_caches(state)
    trace = model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    trace = jax.tree_util.tree_map(
        lambda value: np.asarray(value) if isinstance(value, jax.Array) else value,
        trace,
    )

    masks = expected_masks(card)
    active_u, active_v = masks["u"][..., 0], masks["v"][..., 0]
    native = {}
    for name in (
        "transport_metric_u", "transport_metric_v",
        "transport_velocity_u", "transport_velocity_v",
        "transport_face_depth_u", "transport_face_depth_v",
        "transport_sum_u_entry", "transport_sum_v_entry",
        "transport_sum_u_exit", "transport_sum_v_exit",
    ):
        stagger_name = "u_entry" if "_u" in name else "v_entry"
        native[name] = _trace_native(trace.substeps[name], stagger_name)
    native["transport_weight"] = np.asarray(trace.substeps["transport_weight"])
    plant_location = None
    if plant:
        native["transport_sum_u_exit"] = native["transport_sum_u_exit"].copy()
        live = np.broadcast_to(active_u, native["transport_sum_u_exit"].shape) \
            & np.isfinite(native["transport_sum_u_exit"]) \
            & (native["transport_sum_u_exit"] != 0.0)
        require(np.any(live), "plant requires a nonzero live transport sum")
        index = tuple(np.argwhere(live)[0])
        native["transport_sum_u_exit"][index] = np.nextafter(
            native["transport_sum_u_exit"][index], np.inf)
        plant_location = {"substep": int(index[0] + 1), "j": int(index[1]),
                          "i": int(index[2]), "original_nonzero": True}

    weight_rows = {
        "header_weights": compare(oracle["weights"], oracle["weight"]),
        "legoesm_weights": compare(native["transport_weight"], oracle["weight"]),
    }

    # Round-19 source-order operand table.  Every candidate is taken from the
    # same production-JIT private trace as the longstanding advective-mean
    # walk; no replay replaces the production step.
    def trace_native(name, stagger=None):
        values = np.asarray(trace.substeps[name])[:2]
        return (_trace_native(values, f"{stagger}_entry")
                if stagger in ("u", "v") else values)

    candidate_ordered = {
        "u_entry": trace_native("u_entry", "u"),
        "v_entry": trace_native("v_entry", "v"),
        "u_history_b": trace_native("u_history_b", "u"),
        "v_history_b": trace_native("v_history_b", "v"),
        "u_history_bb": trace_native("u_history_bb", "u"),
        "v_history_bb": trace_native("v_history_bb", "v"),
        "eta_entry": trace_native("eta_entry"),
        "eta_history_b": trace_native("eta_history_b"),
        "eta_history_bb": trace_native("eta_history_bb"),
        "u_mid": trace_native("u_mid", "u"),
        "v_mid": trace_native("v_mid", "v"),
        "eta_mid": trace_native("eta_mid"),
        "face_depth_u_mid": trace_native("transport_face_depth_u", "u"),
        "face_depth_v_mid": trace_native("transport_face_depth_v", "v"),
        "metric_transport_u": trace_native("transport_metric_u", "u"),
        "metric_transport_v": trace_native("transport_metric_v", "v"),
        "metric_e2u": np.broadcast_to(
            np.asarray(card.recipe.grid.dy_u[:, 1:]), (2, *active_u.shape)),
        "metric_e1v": np.broadcast_to(
            np.asarray(card.recipe.grid.dx_v[1:, :]), (2, *active_v.shape)),
        "r1_area": np.broadcast_to(
            1.0 / np.asarray(card.recipe.grid.area), (2, *masks["ssh"].shape)),
        "continuity_du": trace_native("continuity_du"),
        "continuity_dv": trace_native("continuity_dv"),
        "continuity_divergence": trace_native("continuity_divergence"),
        "continuity_forcing": trace_native("continuity_forcing"),
        "eta_exit": trace_native("eta_exit"),
        "face_ssh_u_exit": trace_native("face_ssh_u_exit", "u"),
        "face_ssh_v_exit": trace_native("face_ssh_v_exit", "v"),
        "eta_pgf": trace_native("eta_pgf"),
        "r1_dx_u": np.broadcast_to(
            1.0 / np.asarray(card.recipe.grid.dx_u[:, 1:]), (2, *active_u.shape)),
        "r1_dy_v": np.broadcast_to(
            1.0 / np.asarray(card.recipe.grid.dy_v[1:, :]), (2, *active_v.shape)),
        "pgf_u": trace_native("pgf_u", "u"),
        "pgf_v": trace_native("pgf_v", "v"),
        "ffu_nw": trace_native("ffu_nw"),
        "ffu_ne": trace_native("ffu_ne"),
        "ffu_sw": trace_native("ffu_sw"),
        "ffu_se": trace_native("ffu_se"),
        "ffv_sw": trace_native("ffv_sw"),
        "ffv_se": trace_native("ffv_se"),
        "ffv_nw": trace_native("ffv_nw"),
        "ffv_ne": trace_native("ffv_ne"),
        "cor_u": trace_native("cor_u", "u"),
        "cor_v": trace_native("cor_v", "v"),
        "trd_u": trace_native("trd_u", "u"),
        "trd_v": trace_native("trd_v", "v"),
        "slow_u": trace_native("slow_u", "u"),
        "slow_v": trace_native("slow_v", "v"),
        "u_exit": trace_native("u_exit", "u"),
        "v_exit": trace_native("v_exit", "v"),
        "face_depth_u_exit": trace_native("face_depth_u_exit", "u"),
        "face_depth_v_exit": trace_native("face_depth_v_exit", "v"),
        "r1_face_depth_u_exit": trace_native("r1_face_depth_u_exit", "u"),
        "r1_face_depth_v_exit": trace_native("r1_face_depth_v_exit", "v"),
    }
    candidate_weights = np.stack([
        np.asarray(trace.substeps[f"mid_weight_{index}"])[:2]
        for index in (1, 2, 3)
    ] + [
        np.asarray(trace.substeps[f"back_weight_{index}"])[:2]
        for index in (0, 1, 2, 3)
    ], axis=-1)
    ordered_plant_location = None
    if plant_ordered:
        candidate_ordered["metric_e2u"] = np.array(
            candidate_ordered["metric_e2u"], copy=True)
        j, i = np.argwhere(active_u)[0]
        old = candidate_ordered["metric_e2u"][0, j, i]
        require(old != 0.0, "ordered plant requires a nonzero U metric")
        candidate_ordered["metric_e2u"][0, j, i] = np.nextafter(old, np.inf)
        ordered_plant_location = {"substep": 1, "boundary": "metric_e2u",
                                  "j": int(j), "i": int(i)}

    ordered_rows = []
    ordered_first_nonexact = None
    source_order = tuple(candidate_ordered)
    for jn in range(2):
        row = {"substep": jn + 1, "weights": compare(
            candidate_weights[jn], oracle_ordered["weights"][jn])}
        if ordered_first_nonexact is None and not row["weights"]["bit_exact"]:
            ordered_first_nonexact = {
                "substep": jn + 1, "boundary": "weights", **row["weights"]}
        for name in source_order:
            active = (active_u if ("_u" in name or name.startswith("u_")
                                   or name.startswith("ffu_")) else
                      active_v if ("_v" in name or name.startswith("v_")
                                   or name.startswith("ffv_")) else
                      masks["ssh"])
            result = compare(
                candidate_ordered[name][jn], oracle_ordered[name][jn], active)
            row[name] = result
            if ordered_first_nonexact is None and not result["bit_exact"]:
                ordered_first_nonexact = {
                    "substep": jn + 1, "boundary": name, **result}
        row["eta_unfloored_vs_oracle_exit"] = compare(
            trace_native("eta_unfloored")[jn],
            oracle_ordered["eta_exit"][jn], masks["ssh"])
        oracle_rhs = (oracle_ordered["continuity_forcing"][jn]
                      + oracle_ordered["continuity_divergence"][jn])
        oracle_increment = oracle_ordered["dt"] * oracle_rhs
        row["continuity_rhs_vs_oracle_literal"] = compare(
            trace_native("continuity_rhs")[jn], oracle_rhs, masks["ssh"])
        row["continuity_increment_vs_oracle_literal"] = compare(
            trace_native("continuity_increment")[jn],
            oracle_increment, masks["ssh"])
        ordered_rows.append(row)

    # Independent NumPy transcription of the NEMO statements, fed only oracle
    # operands.  It distinguishes a wrong formula/parser from a legoESM input
    # mismatch before any ownership label is emitted.
    literal_replays = []
    for jn in range(2):
        za1, za2, za3, _, _, _, _ = oracle_ordered["weights"][jn]
        replay_u_mid = ((za1 * oracle_ordered["u_entry"][jn]
                         + za2 * oracle_ordered["u_history_b"][jn])
                        + za3 * oracle_ordered["u_history_bb"][jn])
        replay_v_mid = ((za1 * oracle_ordered["v_entry"][jn]
                         + za2 * oracle_ordered["v_history_b"][jn])
                        + za3 * oracle_ordered["v_history_bb"][jn])
        replay_div = ((oracle_ordered["continuity_du"][jn]
                       + oracle_ordered["continuity_dv"][jn])
                      * oracle_ordered["r1_area"][jn])
        replay_eta = (
            oracle_ordered["eta_entry"][jn]
            - oracle_ordered["dt"] * (
                oracle_ordered["continuity_forcing"][jn] + replay_div)
        ) * masks["ssh"]
        literal_replays.append({
            "u_mid": compare(replay_u_mid, oracle_ordered["u_mid"][jn], active_u),
            "v_mid": compare(replay_v_mid, oracle_ordered["v_mid"][jn], active_v),
            "continuity_divergence": compare(
                replay_div, oracle_ordered["continuity_divergence"][jn], masks["ssh"]),
            "eta_exit": compare(replay_eta, oracle_ordered["eta_exit"][jn], masks["ssh"]),
        })
    # One-variable causal arm at the first velocity-producing boundary.  The
    # faithful reconstruction must reproduce the production trace before the
    # oracle slow-forcing operand alone is substituted.
    slow_forcing_arm = {}
    substep_dt = card.dt_s / cfg.barotropic.n_barotropic_substeps
    for face, active in (("u", active_u), ("v", active_v)):
        def reconstruct(slow):
            entry = _trace_native(
                bt_frame(trace.substeps, f"{face}_entry"), f"{face}_entry")[0]
            pgf = _trace_native(
                bt_frame(trace.substeps, f"pgf_{face}"), f"pgf_{face}")[0]
            trd = _trace_native(
                bt_frame(trace.substeps, f"trd_{face}"), f"trd_{face}")[0]
            # GYRE resolves ln_dynadv_vec=T, so dynspg_ts.F90:720-735 takes
            # the velocity-form update, not key_qcoTest_FluxForm.
            return (entry + substep_dt * (pgf + trd + slow)) * jnp.asarray(
                active, dtype=jnp.float64)

        candidate_slow = _trace_native(
            bt_frame(trace.substeps, f"slow_{face}"), f"slow_{face}")[0]
        faithful_reconstruction = np.asarray(jax.jit(reconstruct)(candidate_slow))
        oracle_slow_arm = np.asarray(jax.jit(reconstruct)(oracle_bt[f"slow_{face}"][0]))
        actual_exit = _trace_native(
            bt_frame(trace.substeps, f"{face}_exit"), f"{face}_exit")[0]
        reference_exit = oracle_bt[f"{face}_exit"][0]
        faithful_row = compare(actual_exit, reference_exit, active)
        arm_row = compare(oracle_slow_arm, reference_exit, active)
        movement = float(np.max(
            np.abs(oracle_slow_arm[active] - actual_exit[active]), initial=0.0))
        slow_forcing_arm[face] = {
            "reconstruction_vs_production": compare(
                faithful_reconstruction, actual_exit, active),
            "faithful_vs_oracle": faithful_row,
            "oracle_slow_only_vs_oracle": arm_row,
            "causal_movement": movement,
            "movement_over_faithful_residual": (
                movement / faithful_row["absolute_max"]
                if faithful_row["absolute_max"] else None
            ),
            "owner_label": (
                "CONFIRMED_CAUSAL_CONTRIBUTOR"
                if arm_row["absolute_max"] < faithful_row["absolute_max"]
                else "REFUTED"
            ),
        }
    external_rows = []
    external_first_nonexact = None
    external_order = (
        "eta_entry", "u_entry", "v_entry", "eta_mid", "u_mid", "v_mid",
        "eta_exit", "eta_pgf", "pgf_u", "pgf_v", "trd_u", "trd_v",
        "slow_u", "slow_v", "u_exit", "v_exit",
    )
    for jn in range(oracle["header"]["ncycle"]):
        values = {"substep": jn + 1, "rows": {}}
        for name in external_order:
            stagger = "u" if name.startswith("u_") or name.endswith("_u") else (
                "v" if name.startswith("v_") or name.endswith("_v") else "ssh")
            active = masks[stagger] if stagger == "ssh" else masks[stagger][..., 0]
            row = compare(
                _trace_native(bt_frame(trace.substeps, name), name)[jn],
                oracle_bt[name][jn], active)
            values["rows"][name] = row
            if external_first_nonexact is None and not row["bit_exact"]:
                external_first_nonexact = {
                    "substep": jn + 1, "boundary": name, **row,
                }
        external_rows.append(values)
    substeps = []
    first_nonexact = None
    for jn in range(oracle["header"]["ncycle"]):
        row = {"substep": jn + 1, "u": {}, "v": {}}
        for face, active, reciprocal in (
            ("u", active_u, oracle["r1_e2u"]),
            ("v", active_v, oracle["r1_e1v"]),
        ):
            entry = oracle[f"sum_{face}_entry"][jn]
            metric = oracle[f"metric_{face}"][jn]
            weight = oracle["weight"][jn]
            # Pure NumPy uses the same scalar libm-free IEEE statements and
            # gfortran's source association: ((weight*metric)*reciprocal), then add.
            oracle_input_increment = (weight * metric) * reciprocal
            oracle_input_exit = entry + oracle_input_increment
            checks = {
                "sum_entry": compare(
                    native[f"transport_sum_{face}_entry"][jn], entry, active),
                "eta_mid": compare(
                    _trace_native(bt_frame(trace.substeps, "eta_mid"), "eta_mid")[jn],
                    oracle_bt["eta_mid"][jn], masks["ssh"]),
                "velocity_mid": compare(
                    native[f"transport_velocity_{face}"][jn],
                    oracle[f"velocity_{face}"][jn], active),
                "face_depth": compare(
                    native[f"transport_face_depth_{face}"][jn],
                    oracle[f"face_depth_{face}"][jn], active),
                "metric_transport": compare(
                    native[f"transport_metric_{face}"][jn], metric, active),
                "oracle_input_literal_exit": compare(
                    oracle_input_exit, oracle[f"sum_{face}_exit"][jn], active),
                "sum_exit": compare(
                    native[f"transport_sum_{face}_exit"][jn],
                    oracle[f"sum_{face}_exit"][jn], active),
            }
            row[face] = checks
            if first_nonexact is None:
                for boundary in (
                    "sum_entry", "velocity_mid", "face_depth",
                    "metric_transport",
                    "oracle_input_literal_exit", "sum_exit",
                ):
                    if not checks[boundary]["bit_exact"]:
                        first_nonexact = {
                            "substep": jn + 1, "face": face,
                            "boundary": boundary, **checks[boundary],
                        }
                        break
        substeps.append(row)

    avg_u, avg_v = trace.transport_average
    avg_u = _trace_native(np.asarray(avg_u)[None, ...], "u_exit")[0]
    avg_v = _trace_native(np.asarray(avg_v)[None, ...], "v_exit")[0]
    final_rows = {
        "u_vs_pre_lbc": compare(avg_u, oracle["pre_lbc_u"], active_u),
        "u_vs_post_lbc": compare(avg_u, oracle["post_lbc_u"], active_u),
        "v_vs_pre_lbc": compare(avg_v, oracle["pre_lbc_v"], active_v),
        "v_vs_post_lbc": compare(avg_v, oracle["post_lbc_v"], active_v),
    }
    stage1_state = trace.state_after_barotropic
    stage1_kaa_rows = {
        "u": compare(
            np.asarray(stage1_state.u.data)[:, 1:, :],
            oracle_stage1["u"][..., :30], masks["u"][..., :30]),
        "v": compare(
            np.asarray(stage1_state.v.data)[1:, :, :],
            oracle_stage1["v"][..., :30], masks["v"][..., :30]),
    }
    oracle_literal_final_rows = {
        "u": compare(
            oracle["sum_u_exit"][-1] / oracle["divisor"],
            oracle["pre_lbc_u"], active_u),
        "v": compare(
            oracle["sum_v_exit"][-1] / oracle["divisor"],
            oracle["pre_lbc_v"], active_v),
    }
    all_exact = (
        all(value["bit_exact"] for value in weight_rows.values())
        and all(
            check["bit_exact"]
            for row in ordered_rows
            for check in row.values()
            if isinstance(check, dict) and "bit_exact" in check
        )
        and all(
            checks["bit_exact"]
            for row in substeps for face in ("u", "v")
            for checks in row[face].values()
        )
        and final_rows["u_vs_post_lbc"]["bit_exact"]
        and final_rows["v_vs_post_lbc"]["bit_exact"]
        and all(row["bit_exact"] for row in stage1_kaa_rows.values())
    )
    ordered_plant_control = {
        "requested": plant_ordered,
        "location": ordered_plant_location,
        "fires": (not plant_ordered) or (
            ordered_first_nonexact is not None
            and ordered_first_nonexact["substep"] == 1
            and ordered_first_nonexact["boundary"] == "metric_e2u"
        ),
    }
    if not ordered_plant_control["fires"]:
        raise AssertionError(
            "ordered one-ulp metric plant did not become first mismatch")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round14-advmean-v1",
        "status": "AT-BAR" if all_exact else "DEBT",
        "regime": "production-jit-cpu-fp64-x64",
        "plant": plant,
        "plant_ordered": plant_ordered,
        "legacy_wind_arm": legacy_wind_arm,
        "legacy_stress_arm": legacy_stress_arm,
        "legacy_continuity_arm": legacy_continuity_arm,
        "plant_location": plant_location,
        "oracle_root": str(root),
        "artifacts": {
            path.name: sha256(path), identity.name: sha256(identity),
            bt_path.name: sha256(bt_path), ordered_path.name: sha256(ordered_path),
            "dynspg_ts.F90": sha256(ORACLE_DYNSPG_SOURCE),
            "namelist_cfg": sha256(root / "namelist_cfg"),
            "output.namelist.dyn": sha256(root / "output.namelist.dyn"),
        },
        "oracle_header": oracle["header"],
        "weight_divisor": oracle["divisor"],
        "weight_rows": weight_rows,
        "slow_forcing_one_variable_arm": slow_forcing_arm,
        "external_first_non_bit_exact": external_first_nonexact,
        "all_50_external_frames_bit_exact": external_first_nonexact is None,
        "external_substeps": external_rows,
        "ordered_header": oracle_ordered["header"],
        "ordered_dt": oracle_ordered["dt"],
        "ordered_first_non_bit_exact": ordered_first_nonexact,
        "ordered_substeps": ordered_rows,
        "oracle_input_literal_replays": literal_replays,
        "ordered_plant_control": ordered_plant_control,
        "first_non_bit_exact": first_nonexact,
        "substeps": substeps,
        "final_rows": final_rows,
        "stage1_kaa_rows": stage1_kaa_rows,
        "oracle_input_literal_final_rows": oracle_literal_final_rows,
        "owner_label": (
            ("CONFIRMED_COMPLETE_ADVMEAN" if external_first_nonexact is None
             else "CONFIRMED_ADVMEAN_WITH_SUBSTEP_BIT_DEBT")
            if all_exact else (
                "CONFIRMED_FIRST_DIVERGENCE_UPSTREAM_SLOW_FORCING"
                if all(
                    row["owner_label"] == "CONFIRMED_CAUSAL_CONTRIBUTOR"
                    and row["reconstruction_vs_production"]["bit_exact"]
                    and row["oracle_slow_only_vs_oracle"]["ulp_max"] <= 2
                    for row in slow_forcing_arm.values()
                ) else "UNMEASURED_AFTER_FIRST_NONEXACT_BOUNDARY"
            )
        ),
        "scaling_before_owner": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--plant-ordered", action="store_true")
    parser.add_argument("--legacy-wind-arm", action="store_true")
    parser.add_argument("--legacy-stress-arm", action="store_true")
    parser.add_argument("--legacy-continuity-arm", action="store_true")
    args = parser.parse_args()
    report = run(
        args.oracle_root,
        plant=args.plant,
        legacy_wind_arm=args.legacy_wind_arm,
        legacy_stress_arm=args.legacy_stress_arm,
        plant_ordered=args.plant_ordered,
        legacy_continuity_arm=args.legacy_continuity_arm,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    first = report["first_non_bit_exact"]
    print(
        f"GYRE_ADVMEAN {report['status']}: first={first} "
        f"ordered_first={report['ordered_first_non_bit_exact']} "
        f"plant={report['plant']} plant_ordered={report['plant_ordered']}",
        file=sys.stderr)
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

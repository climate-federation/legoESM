#!/usr/bin/env python3
"""Walk GYRE's stage-1 ``un_adv/vn_adv`` weighted-transport boundary.

The oracle stream is a WRITE-only config-local extension around
``dynspg_ts.F90:695-698,928-939``.  It records the raw ``wgtbtp2`` values,
their divisor, reciprocal face metrics, and—for every external substep—the
accumulator entry, metric transport, and accumulator exit.  The legoESM side
uses the production-jitted step and its private ``_NEMOWSRK3TestHooks`` trace;
no alternate stepping route is evaluated.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    DIMS,
    _surface_forcings,
    _trace_native,
    bt_frame,
    expected_masks,
    read_bt_substeps,
    require,
    sha256,
)
from nemo_testcase_state_ulp_probe import ulp_distance


ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round14_oracle_advmean_v1"
)
IDENTITY_SHA256 = (
    "ce25b004e7e8289b6e803263f895576981ce22516ccddfbd85d7be5ce5bcaedc"
)
ORACLE_DYNSPG_SOURCE = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
    "cfgs/GYRE_OMIP_L2_P3_SM/MY_SRC/dynspg_ts.F90"
)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_advmean(path: Path) -> dict:
    """Read ``NEMO_L2_BTADV_2`` with strict header and EOF checks."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "now", f"{path}: wrong time level")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        version, kt, ncycle, nx, ny, bits = header
        require(
            (magic, version, kt, ncycle, nx, ny, bits)
            == ("NEMO_L2_BTADV_2", 2, 1, 50, *DIMS[:2], 64),
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
            return _xy(value, nx, ny)

        divisor = scalar()
        weights = np.fromfile(handle, dtype=np.float64, count=ncycle)
        require(weights.size == ncycle, f"{path}: truncated weights")
        r1_e2u, r1_e1v = field(), field()
        rows = {name: [] for name in (
            "weight", "sum_u_entry", "sum_v_entry", "metric_u", "metric_v",
            "velocity_u", "velocity_v", "face_depth_u", "face_depth_v",
            "sum_u_exit", "sum_v_exit",
        )}
        for expected in range(1, ncycle + 1):
            raw = handle.read(4)
            require(len(raw) == 4, f"{path}: truncated substep {expected}")
            (jn,) = struct.unpack("=i", raw)
            require(jn == expected, f"{path}: substep {jn} != {expected}")
            rows["weight"].append(scalar())
            for name in (
                "sum_u_entry", "sum_v_entry", "metric_u", "metric_v",
                "velocity_u", "velocity_v", "face_depth_u", "face_depth_v",
                "sum_u_exit", "sum_v_exit",
            ):
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


def compare(candidate, oracle, active=None) -> dict:
    candidate, oracle = np.asarray(candidate), np.asarray(oracle)
    require(candidate.shape == oracle.shape, f"shape {candidate.shape} != {oracle.shape}")
    if active is not None:
        candidate, oracle = candidate[active], oracle[active]
    delta = candidate - oracle
    return {
        "bit_exact": bool(np.array_equal(candidate, oracle)),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        "ulp_max": int(np.max(ulp_distance(candidate, oracle), initial=0)),
        "differing_cells": int(np.count_nonzero(candidate != oracle)),
    }


def run(root: Path, *, plant: bool = False) -> dict:
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
    identity = root / "oracle_stage_kt00000001_s1.bin"
    require(path.is_file() and bt_path.is_file() and identity.is_file(),
            f"missing oracle files in {root}")
    require(sha256(identity) == IDENTITY_SHA256,
            "WRITE-only instrumentation changed the ordinary stage-1 record")
    oracle = read_advmean(path)
    oracle_bt = read_bt_substeps(bt_path)

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    seed_model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    state = seed_model._seed_tke_preclosure_carry(card.recipe.initial_state)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True),
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
    if plant:
        native["transport_sum_u_exit"] = native["transport_sum_u_exit"].copy()
        live = active_u & np.isfinite(native["transport_sum_u_exit"][0]) \
            & (native["transport_sum_u_exit"][0] != 0.0)
        require(np.any(live), "plant requires a nonzero live substep-1 transport sum")
        index = tuple(np.argwhere(live)[0])
        native["transport_sum_u_exit"][(0, *index)] = np.nextafter(
            native["transport_sum_u_exit"][(0, *index)], np.inf)

    weight_rows = {
        "header_weights": compare(oracle["weights"], oracle["weight"]),
        "legoesm_weights": compare(native["transport_weight"], oracle["weight"]),
    }
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
            checks["bit_exact"]
            for row in substeps for face in ("u", "v")
            for checks in row[face].values()
        )
        and final_rows["u_vs_post_lbc"]["bit_exact"]
        and final_rows["v_vs_post_lbc"]["bit_exact"]
    )
    return {
        "format": "nemo-testcase-l2-gyre-round14-advmean-v1",
        "status": "AT-BAR" if all_exact else "DEBT",
        "regime": "production-jit-cpu-fp64-x64",
        "plant": plant,
        "oracle_root": str(root),
        "artifacts": {
            path.name: sha256(path), identity.name: sha256(identity),
            bt_path.name: sha256(bt_path),
            "dynspg_ts.F90": sha256(ORACLE_DYNSPG_SOURCE),
            "namelist_cfg": sha256(root / "namelist_cfg"),
            "output.namelist.dyn": sha256(root / "output.namelist.dyn"),
        },
        "oracle_header": oracle["header"],
        "weight_divisor": oracle["divisor"],
        "weight_rows": weight_rows,
        "slow_forcing_one_variable_arm": slow_forcing_arm,
        "external_first_non_bit_exact": external_first_nonexact,
        "external_substeps": external_rows,
        "first_non_bit_exact": first_nonexact,
        "substeps": substeps,
        "final_rows": final_rows,
        "oracle_input_literal_final_rows": oracle_literal_final_rows,
        "owner_label": (
            "CONFIRMED_COMPLETE_ADVMEAN" if all_exact else (
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
    args = parser.parse_args()
    report = run(args.oracle_root, plant=args.plant)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    first = report["first_non_bit_exact"]
    print(
        f"GYRE_ADVMEAN {report['status']}: first={first} "
        f"plant={report['plant']}", file=sys.stderr)
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

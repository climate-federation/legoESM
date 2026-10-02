#!/usr/bin/env python3
"""Walk independent ORCA2 rung-0 stage-1 momentum accumulators."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round92_rhs_acquisition import (
    check_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round83_slow_forcing_walk as round83,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_round84_rhs_walk as rhs_common,
)


BOUNDARIES = ("after_hpg", "after_ldf", "after_vor", "after_keg", "after_zad")
FACES = ("u", "v")
PLANTS = ("none", "layout", "bottom-slot", "trace-bit")


class GateError(RuntimeError):
    """The record or measurement no longer supports the registered claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def assemble_rhs(root: Path, *, plant: str = "none") -> tuple[dict[str, np.ndarray], dict]:
    """Assemble the two owned rank slabs and account for NEMO's jpk slot."""

    arrays = {
        name: np.empty((148, 180, 30), dtype=np.float64)
        for name in check_record.NAMES
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    bottom_nonzero = 0
    bottom_negative_zero = 0
    for expected_rank in (0, 1):
        path = root / f"oracle_r92_rhs_rank{expected_rank:04d}_kt00000001.bin"
        record = check_record.read_record(path, include_owned_values=True)
        require(record["rank"] == expected_rank, f"{path.name}: rank mismatch")
        nimpp, njmpp = record["origin"]
        ntsi, ntsj, ntei, ntej = record["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        if plant == "layout" and expected_rank == 1:
            i0 -= 1
        i1 = i0 + ntei - ntsi + 1
        j1 = j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: global placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, values in record["owned_values"].items():
            require(values.shape == (148, 90, 31),
                    f"{path.name}: {name} owned shape moved")
            bottom = np.array(values[..., 30], copy=True)
            if plant == "bottom-slot" and expected_rank == 0 and name == check_record.NAMES[0]:
                bottom[0, 0] = np.float64(1.0)
            bottom_nonzero += int(np.count_nonzero(bottom != 0.0))
            bottom_negative_zero += int(np.count_nonzero(np.signbit(bottom)))
            arrays[name][j0:j1, i0:i1] = values[..., :30]
    require(bool(np.all(coverage == 1)),
            "rank-owned RHS slabs do not cover the domain exactly once")
    require(bottom_nonzero == 0, "NEMO jpk accumulator slot is not exact zero")
    return arrays, {
        "coverage": "exactly-once",
        "jpk_values": 148 * 180 * len(check_record.NAMES),
        "jpk_nonzero": bottom_nonzero,
        "jpk_negative_zero": bottom_negative_zero,
    }


def score(candidate: np.ndarray, oracle: np.ndarray, active: np.ndarray) -> dict:
    """Strict bit score with full-domain and registered-wet accounting."""

    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(candidate.shape == oracle.shape == active.shape,
            "RHS comparison shape mismatch")
    require(bool(np.any(active)), "RHS comparison mask is empty")
    require(bool(np.isfinite(candidate).all() and np.isfinite(oracle).all()),
            "RHS comparison contains non-finite values")
    candidate_bits = np.ascontiguousarray(candidate).view(np.uint64)
    oracle_bits = np.ascontiguousarray(oracle).view(np.uint64)
    unequal = candidate_bits != oracle_bits
    wet_unequal = unequal & active
    delta = candidate - oracle
    masked_abs = np.where(active, np.abs(delta), -np.inf)
    at = tuple(map(int, np.unravel_index(np.argmax(masked_abs), delta.shape)))
    wet_delta = delta[active]
    return {
        "bit_exact": not bool(np.any(wet_unequal)),
        "differing_cells": int(np.count_nonzero(wet_unequal)),
        "wet_cells": int(np.count_nonzero(active)),
        "full_domain_differing_cells": int(np.count_nonzero(unequal)),
        "full_domain_cells": int(unequal.size),
        "absolute_max": float(np.max(np.abs(wet_delta), initial=0.0)),
        "rms": float(np.sqrt(np.mean(wet_delta * wet_delta))),
        "argmax_jik": list(at),
        "candidate_at_argmax": float(candidate[at]),
        "oracle_at_argmax": float(oracle[at]),
    }


def first_nonbit(rows: dict[str, dict[str, dict]]) -> dict | None:
    """Select by compiled boundary order and then U/V face order."""

    for boundary in BOUNDARIES:
        for face in FACES:
            row = rows[face][boundary]
            if not row["bit_exact"]:
                return {"boundary": boundary, "face": face, **row}
    return None


def _forcing(shape):
    import jax.numpy as jnp

    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    zero = jnp.zeros(shape, dtype=jnp.float64)
    freshwater = FreshwaterForcing(zero, zero, zero, zero, zero)
    surface = OceanSurfaceForcing(
        sw_down=zero, q_net=zero, tau_x=zero, tau_y=zero,
        freshwater=zero, salt_flux=zero, taum=zero,
        tau_i_native=zero, tau_j_native=zero,
    )
    return freshwater, surface


def measure(deck_root: Path, record_root: Path, expect_commit: str, *, plant: str) -> dict:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-93 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-93 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-93 walk requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    cfg = card.recipe.model_config
    require((cfg.momentum_time_integrator, cfg.tracer_time_integrator,
             cfg.momentum_advection, cfg.vorticity_scheme)
            == ("rk3_ws", "rk3_ws", "vector_invariant", "een_total"),
            "resolved rung-0 stage program moved")
    entry = rung0.assemble_frame(record_root, 1, 0)
    state = rung0.bridge_entry(card, entry)
    entry_rows = rung0.ladder.compare_fields(rung0.candidate_fields(state), entry)
    require(entry_rows["first_non_bit_field"] is None,
            "independent stage-0 entry is no longer bit exact")
    freshwater, surface = _forcing(entry["ssh"].shape)

    ordinary_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg)
    traced_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True),
    )
    ordinary = jax.device_get(ordinary_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    trace = jax.device_get(traced_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    traced_fields = rung0.candidate_fields(trace.state_after)
    if plant == "trace-bit":
        traced_fields["T"] = np.array(traced_fields["T"], copy=True)
        traced_fields["T"][0, 0, 0] = np.nextafter(
            traced_fields["T"][0, 0, 0], np.float64(np.inf))
    passivity = rung0.ladder.compare_fields(
        traced_fields, rung0.candidate_fields(ordinary))
    require(passivity["first_non_bit_field"] is None,
            "live operand trace changes the production trajectory")

    parts = trace.operator_operands[0]
    accumulated = jax.device_get(jax.jit(rhs_common.source_order_accumulators)(
        parts["hpg_u"].data, parts["hpg_v"].data,
        parts["ldf_u"].data, parts["ldf_v"].data,
        parts["vorticity_u"].data, parts["vorticity_v"].data,
        parts["keg_u"].data, parts["keg_v"].data,
        parts["zad_u"].data, parts["zad_v"].data,
    ))
    oracle, record_census = assemble_rhs(record_root, plant=plant)
    active = {
        "u": np.asarray(state.u_mask.data[:, 1:, :], dtype=bool),
        "v": np.asarray(state.v_mask.data[1:, :, :], dtype=bool),
    }
    live = {
        "u": {boundary: round83.native_u(accumulated[f"{boundary}_u"])
              for boundary in BOUNDARIES},
        "v": {boundary: round83.native_v(accumulated[f"{boundary}_v"])
              for boundary in BOUNDARIES},
    }
    rows = {
        face: {
            boundary: score(
                live[face][boundary], oracle[f"{boundary}_{face}"], active[face])
            for boundary in BOUNDARIES
        }
        for face in FACES
    }
    first = first_nonbit(rows)
    require(first is not None, "all five stage-1 RHS boundaries stayed bit exact")

    # Known-answer control independent of the measured residual: one ULP in an
    # otherwise exact active array must become exactly one differing wet cell.
    synthetic = np.zeros((2, 2, 2), dtype=np.float64)
    planted = synthetic.copy()
    planted[0, 0, 0] = np.nextafter(0.0, np.float64(np.inf))
    control = score(planted, synthetic, np.ones_like(synthetic, dtype=bool))
    require(control["differing_cells"] == 1 and not control["bit_exact"],
            "one-ULP RHS control did not fire")

    return {
        "status": "MEASURED_R93_RHS_WALK",
        "claim_label": "independent",
        "worktree": stamp,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "resolved_stage_program": {
            "momentum_integrator": cfg.momentum_time_integrator,
            "tracer_integrator": cfg.tracer_time_integrator,
            "momentum_advection": cfg.momentum_advection,
            "vorticity": cfg.vorticity_scheme,
        },
        "geometry_dtypes": {
            name: str(np.asarray(getattr(card.recipe.z_coord, name)).dtype)
            for name in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref", "h_partial")
        },
        "record": record_census,
        "stage0_entry": entry_rows,
        "trace_passivity": passivity,
        "source_order": list(BOUNDARIES),
        "rows": rows,
        "first_non_bit_statement": first,
        "one_ulp_control": control,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        result = measure(
            args.deck_root, args.record_root, args.expect_commit, plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, rung0.GateError, check_record.Refusal, OSError, ValueError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print("STATUS MEASURED_R93_RHS_WALK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

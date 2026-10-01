#!/usr/bin/env python3
"""Run the three preregistered ORCA2 month stability arms fail-closed."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round71_independent_month_gate as month,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round80_extremes_gate as extremes,
)

ARMS = ("landed", "old-backgrounds", "iwm-off")
PLANTS = ("none", "arm-label", "protocol", "failure-step", "failure-cell", "early-complete")
OLD_AVM = 1.2e-4
OLD_AVT = 1.2e-5


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _configured_model(card, arm: str):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    config = card.recipe.model_config
    vmix = config.physics.vertical_mixing
    if arm == "old-backgrounds":
        vmix = vmix._replace(
            tke=vmix.tke._replace(kappaM_min=OLD_AVM, kappaH_min=OLD_AVT))
    elif arm == "iwm-off":
        vmix = vmix._replace(
            iwm=vmix.iwm._replace(enabled=False, require_forcing_maps=False))
    elif arm != "landed":
        raise GateError(f"unknown arm {arm!r}")
    config = config._replace(
        physics=config.physics._replace(vertical_mixing=vmix))
    forcing = None if arm == "iwm-off" else card.recipe.iwm_forcing
    return config, LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, config,
        iwm_forcing=forcing)


def _first_bad(values: np.ndarray) -> tuple[list[int], float] | None:
    bad = ~np.isfinite(values)
    if not np.any(bad):
        return None
    index = tuple(int(x) for x in np.argwhere(bad)[0])
    return list(index), float(values[index])


def _location(card, vmix_root: Path, index: tuple[int, int, int], e3w_value: float,
              state) -> dict[str, object]:
    j, i, k = index
    z_coord = card.recipe.z_coord
    h_partial = np.asarray(z_coord.h_partial, dtype=np.float64)
    H = np.asarray(state.H_bathy.data, dtype=np.float64)
    eta = np.asarray(state.eta.data, dtype=np.float64)
    active = np.asarray(z_coord.is_active, dtype=bool)
    neighbours = []
    for jj, ii in ((j - 1, i), (j + 1, i), (j, (i - 1) % 180), (j, (i + 1) % 180)):
        if 0 <= jj < 148:
            neighbours.append(bool(active[jj, ii, min(k, active.shape[-1] - 1)]))
    record = extremes._global_record(vmix_root, 1)
    river = bool(record["rnfmsk"][j, i, 0] != 0.0)
    lat = float(np.rad2deg(np.asarray(card.recipe.grid.lat_T))[j, i])
    lon = float(np.rad2deg(np.asarray(card.recipe.grid.lon_T))[j, i])
    depth = float(H[j, i])
    raw_e3w = float(np.asarray(z_coord.nemo_e3w_0)[j, i, k + 1])
    implied_stretch = e3w_value / raw_e3w
    return {
        "index_jik": [j, i, k],
        "raw_e3w_m": e3w_value,
        "eta_m": float(eta[j, i]),
        "bathymetry_m": depth,
        "step_entry_stretch_1_plus_eta_over_H": float(1.0 + eta[j, i] / depth),
        "failing_stage_stretch_from_e3w_ratio": implied_stretch,
        "failing_stage_eta_implied_m": float((implied_stretch - 1.0) * depth),
        # Interior bn2 row k is NEMO jk=k+2, i.e. raw Python slot k+1.
        "raw_mesh_e3w0_m": raw_e3w,
        "latitude_deg": lat,
        "longitude_deg": lon,
        "fold_row": j == 147,
        "river_mouth": river,
        "shallow_shelf_le_200m": depth <= 200.0,
        "land_adjacent": not all(neighbours),
        "convection_site": "UNMEASURED_INVALID_GEOMETRY",
        "column_wet_levels": int(np.count_nonzero(h_partial[j, i] > 0.0)),
    }


def _install_e3w_failure_trace(trace: list[dict[str, object]]) -> None:
    """Wrap every loaded bn2 consumer with a scalar-only bad-e3w callback."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean import eos

    original = eos.compute_buoyancy_frequency_nemo_bn2

    def traced(*args, **kwargs):
        e3w = kwargs.get("e3w_int")
        if e3w is not None:
            values = jnp.asarray(e3w)
            valid = jnp.isfinite(values) & (values > 0.0)
            score = jnp.where(jnp.isfinite(values), values, -jnp.inf)
            minimum = jnp.min(score)
            flat_index = jnp.argmin(score)
            invalid_count = jnp.count_nonzero(~valid)

            def capture(bad, observed_minimum, observed_index, observed_count):
                if bool(bad) and not trace:
                    trace.append({
                        "value": float(observed_minimum),
                        "flat_index": int(observed_index),
                        "invalid_count": int(observed_count),
                        "shape": list(values.shape),
                    })

            jax.debug.callback(
                capture, ~jnp.all(valid), minimum, flat_index, invalid_count,
                ordered=True)
        return original(*args, **kwargs)

    # Several consumers bind the function at import time. Replace only exact
    # references to the original; no model arithmetic or return value changes.
    for module in tuple(sys.modules.values()):
        if module is not None and getattr(
                module, "compute_buoyancy_frequency_nemo_bn2", None) is original:
            setattr(module, "compute_buoyancy_frequency_nemo_bn2", traced)


def _prestep_failure(card, state, vmix_root: Path, step: int) -> dict[str, object] | None:
    from legoesm.ocean.eos import nemo_bn2_live_geometry

    fields = ladder._candidate_fields(state)
    for name in ladder.FIELD_ORDER:
        bad = _first_bad(np.asarray(fields[name]))
        if bad is not None:
            index, value = bad
            return {
                "step": step,
                "field": name,
                "kind": "nonfinite_step_entry_state",
                "index": index,
                "value": value,
            }
    _, _, e3w = nemo_bn2_live_geometry(
        card.recipe.z_coord, state.eta.data, state.H_bathy.data,
        r3t_evaluation="nemo_reciprocal")
    e3w_np = np.asarray(e3w, dtype=np.float64)
    valid = np.isfinite(e3w_np) & (e3w_np > 0.0)
    if bool(np.all(valid)):
        return None
    score = np.where(np.isfinite(e3w_np), e3w_np, -np.inf)
    index = tuple(int(x) for x in np.unravel_index(np.argmin(score), score.shape))
    return {
        "step": step,
        "field": "raw_mesh_e3w_int",
        "kind": "nonpositive_or_nonfinite_step_entry_geometry",
        "index": list(index),
        "value": float(e3w_np[index]),
        "invalid_count": int(np.count_nonzero(~valid)),
        "location": _location(card, vmix_root, index, float(e3w_np[index]), state),
    }


def measure(args) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    _, card = ladder.card_fields(args.deck_root)
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64) and not jax.config.jax_disable_jit,
            "execution is not production CPU JIT fp64")
    admission = extremes.vmix.run_gate(args.vmix_root)
    require(admission["status"] == "PASS", "vertical-mixing record is not admitted")
    failure_trace: list[dict[str, object]] = []
    # The old-background arm is preregistered to complete and therefore takes
    # the uninstrumented production path. The two expected failure arms use a
    # scalar-only observer which neither replaces nor clips the failing value.
    trace_failure = args.arm != "old-backgrounds"
    if trace_failure:
        _install_e3w_failure_trace(failure_trace)
    config, model = _configured_model(card, args.arm)
    vmix = config.physics.vertical_mixing
    state = card.recipe.initial_state
    started = time.time()
    consumed = 0
    failure = None
    for kt in range(1, month.STEPS + 1):
        print(f"ROUND81_ARM_START arm={args.arm} step={kt}/{month.STEPS}", flush=True)
        failure = (_prestep_failure(card, state, args.vmix_root, kt)
                   if trace_failure else None)
        if failure is not None:
            break
        fields = month.assemble_surface(args.surface_root, kt)
        consumed += 2
        freshwater, surface = ladder._surface_forcings(card, args.deck_root, fields, kt)
        try:
            state = model.step(
                state, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface)
        except Exception:
            if not failure_trace:
                raise
            observed = failure_trace[0]
            index = tuple(int(x) for x in np.unravel_index(
                int(observed["flat_index"]), tuple(observed["shape"])))
            failure = {
                "step": kt,
                "field": "raw_mesh_e3w_int",
                "kind": "nonpositive_or_nonfinite_rk_stage_geometry",
                "index": list(index),
                "value": float(observed["value"]),
                "invalid_count": int(observed["invalid_count"]),
                "location": _location(
                    card, args.vmix_root, index, float(observed["value"]), state),
            }
            break
        print(
            f"ROUND81_ARM_PROGRESS arm={args.arm} step={kt}/{month.STEPS} "
            f"wall_s={time.time() - started:.1f}", flush=True)

    completed = failure is None
    terminal = None
    if completed:
        oracle, restart = month.read_terminal_restart(
            args.month_root, args.restart_ledger, card)
        comparison = ladder.compare_fields(ladder._candidate_fields(state), oracle)
        terminal = {"comparison": comparison, "restart": restart}
    return {
        "format": "nemo-testcase-l4-orca2-round81-month-stability-v1",
        "claim_label": "independent",
        "arm": args.arm,
        "arm_config": {
            "iwm_enabled": bool(vmix.iwm.enabled),
            "iwm_require_forcing_maps": bool(vmix.iwm.require_forcing_maps),
            "kappaM_min": float(vmix.tke.kappaM_min),
            "kappaH_min": float(vmix.tke.kappaH_min),
        },
        "protocol": {
            "steps_required": month.STEPS,
            "dt_s": float(card.dt_s),
            "execution": "production-jit-cpu-fp64-x64-libm",
            "initial_mode": "card_own_state",
            "surface_root": str(args.surface_root),
            "month_root": str(args.month_root),
        },
        "steps_completed": month.STEPS if completed else int(failure["step"]) - 1,
        "surface_frames_consumed": consumed,
        "failure": failure,
        "terminal": terminal,
        "vertical_record_admission": admission["status"],
        "wall_seconds": time.time() - started,
        "worktree": worktree_stamp(),
    }


def classify(report: dict[str, object], plant: str = "none") -> dict[str, object]:
    require(plant in PLANTS, f"unknown plant {plant}")
    report = json.loads(json.dumps(report))
    if plant == "arm-label":
        report["arm"] = "unknown"
    elif plant == "protocol":
        report["protocol"]["dt_s"] = 1.0
    elif plant == "failure-step" and report["failure"] is not None:
        report["failure"]["step"] += 1
    elif plant == "failure-cell" and report["failure"] is not None:
        report["failure"]["index"] = [-1, -1, -1]
    elif plant == "early-complete":
        report["failure"] = None
        report["steps_completed"] = month.STEPS - 1

    require(report["claim_label"] == "independent", "claim label changed")
    require(report["arm"] in ARMS, "arm label changed")
    require(report["protocol"]["steps_required"] == month.STEPS
            and report["protocol"]["dt_s"] == 10800.0
            and report["protocol"]["initial_mode"] == "card_own_state",
            "month protocol changed")
    require(report["vertical_record_admission"] == "PASS",
            "vertical-mixing record admission changed")
    failure = report["failure"]
    if failure is None:
        require(report["steps_completed"] == month.STEPS,
                "arm claimed completion before step 240")
        require(report["terminal"] is not None, "complete arm has no terminal score")
        status = "PASS_ROUND81_ARM_COMPLETE"
    else:
        step = int(failure["step"])
        require(1 <= step <= month.STEPS, "failure step is outside the protocol")
        require(report["steps_completed"] == step - 1,
                "failure step and completed-step count disagree")
        index = failure["index"]
        require(len(index) == 3 and 0 <= index[0] < 148 and 0 <= index[1] < 180
                and 0 <= index[2] < 30, "failure cell is outside the ORCA2 ocean")
        require(failure["field"] == "raw_mesh_e3w_int",
                "failure is not the preregistered raw-mesh thickness class")
        require(not math.isfinite(float(failure["value"]))
                or float(failure["value"]) <= 0.0,
                "reported raw-mesh thickness is valid")
        status = "PASS_ROUND81_ARM_FAILURE"
    return {**report, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--surface-root", type=Path)
    parser.add_argument("--month-root", type=Path)
    parser.add_argument("--restart-ledger", type=Path)
    parser.add_argument("--vmix-root", type=Path)
    parser.add_argument("--classify-json", type=Path)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.classify_json:
            require(not any((args.arm, args.deck_root, args.surface_root,
                             args.month_root, args.restart_ledger, args.vmix_root)),
                    "--classify-json cannot be combined with run inputs")
            raw = json.loads(args.classify_json.read_text())
        else:
            require(args.plant == "none", "plants classify an existing report")
            require(all((args.arm, args.deck_root, args.surface_root, args.month_root,
                         args.restart_ledger, args.vmix_root)),
                    "run mode requires every input")
            raw = measure(args)
        result = classify(raw, args.plant)
    except (GateError, ladder.GateError, month.GateError, extremes.vmix.GateError,
            KeyError, OSError, TypeError, ValueError) as error:
        print(f"STATUS {'PLANT-FIRED' if args.plant != 'none' else 'REFUSE'}: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

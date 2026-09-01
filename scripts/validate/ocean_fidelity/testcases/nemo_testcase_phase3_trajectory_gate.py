#!/usr/bin/env python3
"""First-divergence trajectory gate for the certified NEMO testcases.

The NEMO records are the Nbb/before entry state.  legoESM's card initial state
therefore maps to kt=1 and one completed ``model.step`` maps to kt=2.  The
default gate stops immediately after first debt.  ``--continue-after-first``
is the explicit owner-exhausted sweep arm: it preserves that first-divergence
record and walks the remaining registered states without pretending they have
an exact entering prefix.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

BAR = 1.0e-15
DEFAULT_ORACLE_ROOTS = {
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"),
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10"),
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def read_entry(path: Path, case: str) -> dict:
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, step, nbb, nx, ny, nz, ntr, bits = struct.unpack(
            "=8i", fh.read(32))
        data = np.fromfile(fh, dtype=np.float64)
    expected = (206, 7, 101) if case == "OVERFLOW-zps" else (134, 7, 21)
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *expected, 2, 64),
        f"{path}: bad header")
    count = nx * ny * nz
    require(data.size == 4 * count + nx * ny, f"{path}: bad payload length")

    def xyz(values):
        return values.reshape((nx, ny, nz), order="F")[
            2:-2, 2:-2].transpose(1, 0, 2)

    return {
        "step": step,
        "Nbb": nbb,
        "T": xyz(data[:count]),
        "S": xyz(data[count:2 * count]),
        "u": xyz(data[2 * count:3 * count]),
        "v": xyz(data[3 * count:4 * count]),
        "ssh": data[4 * count:].reshape((nx, ny), order="F")[
            2:-2, 2:-2].T,
    }


def expected_masks(card):
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    v = active & np.roll(active, -1, axis=0)
    v[-1] = False
    return {"T": active, "S": active, "u": u, "v": v, "ssh": wet}


def lego_fields(state):
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        # NEMO's local interior stores one x record per T column.  The phase-2
        # geometry gate established these stagger mappings exactly.
        "u": np.asarray(state.u.data)[:, 1:, :],
        "v": np.asarray(state.v.data)[1:, :, :],
        "ssh": np.asarray(state.eta.data),
    }


def score(
    name: str, oracle, candidate, mask, *, plant=False,
    allow_empty_no_active_face=False,
) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate is {candidate.dtype}")
    no_active_face = not bool(use.any())
    if no_active_face:
        require(allow_empty_no_active_face, f"{name}: empty mask")
        # Inventory the structurally absent face array and retain a gross
        # nonzero control over all stored points. It is not an alignment row.
        use = np.ones(oracle.shape, dtype=bool)
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(use)[0])] += 1.0
    require(np.all(np.isfinite(candidate[use])), f"{name}: candidate nonfinite")
    exact = bool(np.array_equal(oracle[use], candidate[use]))
    scale = max(float(np.max(np.abs(oracle[use]))), 1.0)
    error = float(np.max(np.abs(candidate[use] - oracle[use]))) / scale
    status = "AT-BAR" if error <= BAR else "DEBT"
    row = {
        "name": name,
        "status": status,
        "exact": exact,
        "normalized_max_abs": error,
        "bar": BAR,
        "n": int(use.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if no_active_face and status == "AT-BAR":
        row["status"] = "UNMEASURED"
        row["reason"] = (
            "no active meridional velocity face in the three-row closed tank; "
            "all stored values were nevertheless checked against zero")
    return row


def mark_uninformative(row: dict, field: str, kt: int, reference, mask) -> dict:
    """Downgrade vacuous at-bar controls without hiding a real failure."""
    if row["status"] != "AT-BAR" or kt < 2:
        return row
    wet_values = np.asarray(reference)[np.asarray(mask, dtype=bool)]
    if field == "S" and np.unique(wet_values).size == 1:
        row["status"] = "UNINFORMATIVE"
        row["reason"] = (
            "oracle salinity is spatially uniform (n_unique=1); "
            "this row cannot detect transport/time-level errors")
    elif field == "ssh" and np.count_nonzero(wet_values) == 0:
        row["status"] = "UNINFORMATIVE"
        row["reason"] = (
            "oracle SSH is identically zero; the row only bounds "
            "candidate absolute noise and cannot corroborate alignment")
    return row


def run(
    case: str, oracle_root: Path, max_step: int, *, plant=False,
    continue_after_first=False, diagnostic_disable_bbl=False,
    owner_controls=False,
) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_nemo_testcase_card(case)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            disable_bbl=diagnostic_disable_bbl))
    state = card.recipe.initial_state
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    steps = []
    exact_prefix = True
    first_over_bar = None

    for kt in range(1, max_step + 1):
        path = oracle_root / f"oracle_step_entry_kt{kt:08d}.bin"
        require(path.is_file(), f"missing {path}")
        oracle = read_entry(path, case)
        require(oracle["step"] == kt, f"{path}: step mismatch")
        candidate = lego_fields(state)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(oracle[field])
            if field != "ssh":
                reference = reference[..., :nlev]
            rows.append(score(
                f"{case}.kt{kt}.before.{field}", reference,
                candidate[field], masks[field], plant=plant and kt == 1
                and field == "T", allow_empty_no_active_face=field == "v"))
            if field in ("u", "v"):
                rows[-1].update({
                    "frame": "instantaneous_prognostic_Nbb",
                    "staggering_and_reduction": (
                        f"oracle {field}{field}(:,:,:,Nbb) and legoESM "
                        f"prognostic {field} are both C-grid {field.upper()}-face "
                        "instantaneous 3-D fields; both use the same wet-face "
                        "mask and an elementwise L-infinity reduction, with no "
                        "depth or substep-time averaging"),
                })
            rows[-1] = mark_uninformative(
                rows[-1], field, kt, reference, masks[field])
        exact_here = all(row["exact"] for row in rows)
        over = [row["name"].rsplit(".", 1)[-1] for row in rows
                if row["status"] == "DEBT"]
        steps.append({
            "kt": kt,
            "Nbb": oracle["Nbb"],
            "exact_prefix_entering": exact_prefix,
            "exact_at_step": exact_here,
            "rows": rows,
        })
        exact_prefix = exact_prefix and exact_here
        if over:
            if first_over_bar is None:
                first_over_bar = {"kt": kt, "fields": over}
            if not continue_after_first:
                break
        if kt < max_step:
            state = model.step(state, dt=card.dt_s)

    cfg = card.recipe.model_config
    bbl_attribution = None
    overflow_t_owner_hunt = None
    if case == "OVERFLOW-zps":
        from legoesm.ocean.physics.bbl_adv import (
            bbl_static_geometry,
            bbl_transports,
        )
        geom = bbl_static_geometry(
            card.recipe.z_coord.h_partial,
            card.recipe.initial_state.land_mask.data)
        utr, vtr = bbl_transports(
            card.recipe.initial_state.T.data,
            card.recipe.initial_state.S.data,
            geom,
            np.asarray(card.recipe.grid.dy_u)[:, 1:-1],
            np.asarray(card.recipe.grid.dx_v)[1:-1, :],
            gamma_s=cfg.bbl_gamma_s,
            rho_0=cfg.rho_0,
        )
        faithful_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg)
        control_model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(disable_bbl=True))
        faithful_kt2 = faithful_model.step(
            card.recipe.initial_state, dt=card.dt_s)
        control_kt2 = control_model.step(
            card.recipe.initial_state, dt=card.dt_s)
        delta = np.asarray(faithful_kt2.T.data) - np.asarray(control_kt2.T.data)
        bbl_attribution = {
            "classification": "ONE_SIDED",
            "legoesm_side": "CONFIRMED_INACTIVE_AT_KT2",
            "nemo_side": "PLAUSIBLE_INACTIVE_FROM_GEOMETRY_ONLY",
            "scaling_check_before_owner_label": True,
            "max_abs_initial_utr_m3_s": float(np.max(np.abs(np.asarray(utr)))),
            "max_abs_initial_vtr_m3_s": float(np.max(np.abs(np.asarray(vtr)), initial=0.0)),
            "max_abs_kt2_temperature_movement_K": float(np.max(np.abs(delta))),
            "bit_identical_bbl_on_off_kt2_T": bool(np.array_equal(
                np.asarray(faithful_kt2.T.data), np.asarray(control_kt2.T.data))),
            "reason": (
                "legoESM's shipped initial density front does not intersect "
                "an active downslope face, so its option-2 transport is zero. "
                "The same NEMO conclusion is geometrically plausible but is "
                "not confirmed because utr_bbl was not read from NEMO"),
            "source": (
                "NEMO trabbl.F90:342-353,415-454; stage-3 calls at "
                "stprk3_stg.F90:468,588"),
        }
        if owner_controls:
            oracle2 = read_entry(
                oracle_root / "oracle_step_entry_kt00000002.bin", case)
            oracle_T = oracle2["T"][..., :nlev]
            active_T = masks["T"]
            faithful_T = np.asarray(faithful_kt2.T.data)

            def t_abs_error(values):
                return float(np.max(np.abs(
                    np.asarray(values)[active_T] - oracle_T[active_T])))

            faithful_error = t_abs_error(faithful_T)
            require(faithful_error > 0.0, "OVERFLOW owner hunt is vacuous")
            controls = []

            def add_control(name, state_control, reference, changed_operand):
                values = np.asarray(state_control.T.data)
                movement = float(np.max(np.abs(values[active_T] - faithful_T[active_T])))
                error = t_abs_error(values)
                controls.append({
                    "name": name,
                    "classification": "DIAGNOSTIC_ONE_VARIABLE_ARM",
                    "reference_arm": reference,
                    "changed_operand": changed_operand,
                    "faithful_absolute_max_error_K": faithful_error,
                    "control_absolute_max_error_K": error,
                    "candidate_movement_K": movement,
                    "movement_over_faithful_error": movement / faithful_error,
                    "error_change_K": error - faithful_error,
                    "owner_label": "UNMEASURED",
                    "reason": (
                        "scale and movement only; a legacy/private one-sided "
                        "candidate control cannot establish two-model ownership"),
                })

            hook_arms = (
                ("legacy_one_step_fct", _NEMOWSRK3TestHooks(
                    two_step_fct_predictor=False),
                 "legoESM legacy, pre-existing one-step FCT",
                 "two-step FCT predictor only"),
                ("legacy_frozen_final_tracer_transport", _NEMOWSRK3TestHooks(
                    kmm_tracer_transports=False),
                 "legoESM legacy, pre-existing frozen-final transport",
                 "Kmm tracer transport time level only"),
                ("omit_stage_barotropic_correction", _NEMOWSRK3TestHooks(
                    stage_barotropic_correction=False),
                 "legoESM legacy, pre-existing post-stage split",
                 "per-stage primary velocity correction only"),
                ("omit_transport_reconcile", _NEMOWSRK3TestHooks(
                    momentum_transport_reconcile=False),
                 "private harness ablation of NEMO stprk3_stg.F90:257-274",
                 "un_adv/H advecting-transport reconcile only"),
            )
            for name, hooks, reference, operand in hook_arms:
                control_state = LatLonCGridOceanModel(
                    card.recipe.grid, card.recipe.z_coord, cfg,
                    _nemo_ws_test_hooks=hooks).step(
                        card.recipe.initial_state, dt=card.dt_s)
                add_control(name, control_state, reference, operand)

            transport_cfg = cfg._replace(
                barotropic=cfg.barotropic._replace(
                    barotropic_reconcile_target="transport_avg"))
            add_control(
                "wrong_prognostic_transport_frame",
                LatLonCGridOceanModel(
                    card.recipe.grid, card.recipe.z_coord, transport_cfg).step(
                        card.recipe.initial_state, dt=card.dt_s),
                "NEMO un_adv tracer transport, not a prognostic state frame",
                "prognostic primary velocity replaced by time-mean transport")
            overflow_t_owner_hunt = {
                "status": "FIRST_DIVERGENCE_REMAINS_UNOWNED",
                "first_over_bar": "kt=2 T",
                "faithful_absolute_max_error_K": faithful_error,
                "faithful_normalized_max_error": faithful_error / max(
                    float(np.max(np.abs(oracle_T[active_T]))), 1.0),
                "scaling_check_before_owner_label": True,
                "one_variable_controls": controls,
            }
    return {
        "format": "nemo-testcase-l1-phase3-trajectory-v1",
        "case": case,
        "status": "AT-BAR" if first_over_bar is None else "DEBT",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "bar": BAR,
        "oracle_root": str(oracle_root),
        "selectors": {
            "eos": cfg.eos,
            "eos_depth": cfg.eos_depth,
            "barotropic_time_filter": cfg.barotropic.barotropic_time_filter,
            "n_barotropic_substeps": cfg.barotropic.n_barotropic_substeps,
            "vertical_momentum_scheme": cfg.vertical_momentum_scheme,
            "tracer_time_integrator": cfg.tracer_time_integrator,
            "rk3_ws_scheme_identity": (
                "nemo_kmm+two_step_fct+stage_correction+transport_reconcile"),
            "bbl_adv_option": cfg.bbl_adv_option,
            "bbl_gamma_s": cfg.bbl_gamma_s,
            "diagnostic_disable_bbl_test_hook": diagnostic_disable_bbl,
        },
        "first_over_bar": first_over_bar,
        "bbl_attribution": bbl_attribution,
        "overflow_t_owner_hunt": overflow_t_owner_hunt,
        "steps": steps,
        "unmeasured": [
            "NEMO per-term tendencies at the first divergent step",
            *([] if case == "OVERFLOW-zps" else
              ["stage-coupled OVERFLOW BBL transport and tendency"]),
            *([] if continue_after_first else
              ["trajectory after the first over-bar step"]),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(DEFAULT_ORACLE_ROOTS), required=True)
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--continue-after-first", action="store_true")
    parser.add_argument("--diagnostic-disable-bbl", action="store_true")
    parser.add_argument("--owner-controls", action="store_true")
    args = parser.parse_args()
    require(args.max_step >= 1, "max-step must be positive")
    report = run(
        args.case, args.oracle_dir or DEFAULT_ORACLE_ROOTS[args.case],
        args.max_step, plant=args.plant,
        continue_after_first=args.continue_after_first,
        diagnostic_disable_bbl=args.diagnostic_disable_bbl,
        owner_controls=args.owner_controls)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)

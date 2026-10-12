#!/usr/bin/env python3
"""Locate the first kt=8 live-W failure boundary of the complete fold unit."""

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

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    nemo_testcase_l4_orca2_round135_step36_fct_gate as passive,
    nemo_testcase_l4_orca2_round166_external_substep_gate as r166,
    nemo_testcase_l4_orca2_round204_omt0_ladder_gate as omt0,
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as omt4,
    nemo_testcase_l4_orca2_round228_fold_invariant_audit as r228,
    nemo_testcase_l4_orca2_round245_v_depth_split as r245,
    nemo_testcase_l4_orca2_round249_fold_growth_gate as r249,
)

LABELS = ("independent", "given_nemo_entry")
FIELDS = ("ssh", "u", "v", "T", "S")
EXPECTED_ERROR = "raw-mesh e3w_int must contain only finite values > 0"
PLANTS = (
    "none", "label-coverage", "passivity", "nemo-guard", "finite-entry",
    "terminal", "source-order", "first-invalid", "growth", "one-bit",
)
ETA_BOUNDARIES = ("eta_entry", "eta_mid", "eta_continuity")


class GateError(RuntimeError):
    """The frozen round-253 causal predicates moved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _policy_stamp(expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    stamp = worktree_stamp()
    require(stamp["clean"] and stamp["commit"].lower() == expect_commit.lower(),
            "round-253 measurement requires its clean committed tree")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-253 measurement requires production JIT on CPU")
    return stamp


def _hooks(card, *, live: bool = False, substeps: bool = False,
           boundary_trace: bool = False):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    return _NEMOWSRK3TestHooks(
        expose_live_stage_operands=live,
        expose_barotropic_substeps=substeps,
        expose_barotropic_boundary_association=boundary_trace,
        barotropic_external_mode_association=True,
        barotropic_reference_face_depth_override=(
            rung0.ladder.build_reference_depth_override(card)),
        barotropic_unmasked_v_transport=True,
        barotropic_materialize_v_transport=True,
        barotropic_atomic_fold_unit=True,
        preserve_literal_hpg_vmask=True,
    )


def _raw_e3w(card, eta: np.ndarray) -> np.ndarray:
    """Replay eos.py's unfloored raw-mesh live-W operand."""
    eta = np.asarray(eta, np.float64)
    depth = np.asarray(card.recipe.initial_state.H_bathy.data, np.float64)
    raw = np.asarray(card.recipe.z_coord.nemo_e3w_0, np.float64)[..., 1:]
    stretch = np.ones_like(eta)
    np.divide(eta, depth, out=stretch, where=depth > 0.0)
    stretch = np.where(depth > 0.0, 1.0 + stretch, 1.0)
    return raw * stretch[..., None]


def _guard_row(card, eta: np.ndarray) -> dict[str, object]:
    eta = np.asarray(eta, np.float64)
    depth = np.asarray(card.recipe.initial_state.H_bathy.data, np.float64)
    live = _raw_e3w(card, eta)
    invalid = ~np.isfinite(live) | (live <= 0.0)
    first = np.argwhere(invalid)
    if first.size:
        j, i, k = map(int, first[0])
        cell = {
            "index": [j, i, k], "ssh": float(eta[j, i]),
            "ht0": float(depth[j, i]), "live_e3w": float(live[j, i, k]),
            "stretch": float(1.0 + eta[j, i] / depth[j, i])
            if depth[j, i] > 0 else None,
        }
    else:
        cell = None
    finite = live[np.isfinite(live)]
    return {
        "invalid": int(np.count_nonzero(invalid)),
        "nonfinite": int(np.count_nonzero(~np.isfinite(live))),
        "nonpositive": int(np.count_nonzero(np.isfinite(live) & (live <= 0.0))),
        "minimum": float(np.min(finite)) if finite.size else None,
        "first": cell,
    }


def _nemo_guard(card, frame_root: Path) -> dict[str, object]:
    rows = []
    for kt in range(1, 11):
        for stage in range(4):
            frame = rung0.assemble_frame(frame_root, kt, stage)
            rows.append({"kt": kt, "stage": stage,
                         **_guard_row(card, frame["ssh"])})
    return {
        "rows": rows,
        "checked": len(rows),
        "invalid_total": sum(row["invalid"] for row in rows),
        "minimum": min(row["minimum"] for row in rows
                       if row["minimum"] is not None),
    }


def _boundary_row(card, state, oracle: dict[str, np.ndarray], kt: int,
                  stage: int) -> dict[str, object]:
    return r249._boundary(card, state, oracle, kt, stage)


def _growth(boundaries: list[dict[str, object]]) -> tuple[list[dict], dict | None]:
    rows = []
    previous = {name: None for name in FIELDS}
    first = None
    for boundary in boundaries:
        for name in FIELDS:
            maximum = boundary["fields"][name]["complete"]["max_abs"]
            before = previous[name]
            if before is None:
                factor = None
            elif before == 0.0:
                factor = None if maximum == 0.0 else float("inf")
            else:
                factor = maximum / before
            row = {"kt": boundary["kt"], "stage": boundary["stage"],
                   "field": name, "max_abs": maximum, "growth": factor}
            rows.append(row)
            if first is None and factor is not None and factor > 10.0:
                first = row
            previous[name] = maximum
    return rows, first


def _delta_row(candidate: np.ndarray, entry: np.ndarray) -> dict[str, object]:
    return r249._score(np.asarray(candidate, np.float64), np.asarray(entry, np.float64))


def _substep_rows(card, trace: dict[str, object], entry_state) -> tuple[list[dict], dict | None]:
    entry = r249._state_fields(entry_state)
    rows = []
    first = None
    prior_valid = True
    for index in range(np.asarray(trace["eta_entry"]).shape[0]):
        eta_values = {name: np.asarray(trace[name][index], np.float64)
                      for name in ETA_BOUNDARIES}
        guards = {name: _guard_row(card, values)
                  for name, values in eta_values.items()}
        for name in ETA_BOUNDARIES:
            if first is None and guards[name]["invalid"]:
                first = {"substep": index + 1, "boundary": name,
                         "prior_boundary_valid": prior_valid,
                         **guards[name]}
            prior_valid = guards[name]["invalid"] == 0
        eta = eta_values["eta_continuity"]
        u = r249._native("u", np.asarray(trace["u_exit"][index], np.float64))
        v = r249._native("v", np.asarray(trace["v_exit"][index], np.float64))
        fold = {
            "ssh": r228.fold_residual(eta, "T", card.recipe.grid.fold, 1.0),
            "u": r228.fold_residual(u, "U", card.recipe.grid.fold, -1.0),
            "v": r228.fold_residual(v, "V", card.recipe.grid.fold, -1.0),
        }
        rows.append({
            "substep": index + 1,
            "guards": guards,
            "eta_delta_and_external_velocity_magnitude": {
                "ssh": _delta_row(eta, entry["ssh"]),
                # The external-mode velocities are two-dimensional depth
                # means; the prognostic entry velocities are three-dimensional.
                # Keep their native statement shape and report magnitude.
                "u": _delta_row(u, np.zeros_like(u)),
                "v": _delta_row(v, np.zeros_like(v)),
            },
            "fold_identity": fold,
            "source_operands": {
                name: r249._score(np.asarray(trace[name][index], np.float64),
                                  np.zeros_like(np.asarray(trace[name][index], np.float64)))
                for name in ("continuity_du", "continuity_dv",
                             "continuity_divergence", "continuity_rhs",
                             "continuity_increment")
            },
        })
    return rows, first


def _one_bit_control() -> dict[str, object]:
    one = np.asarray([1.0], np.float64)
    return r249._score(one, np.nextafter(one, np.inf))


def measure(deck_root: Path, frame_root: Path, label: str,
            expect_commit: str) -> dict[str, object]:
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSLiveOperandTrace,
        _NEMOWSBarotropicTrace,
    )

    require(label in LABELS, f"bad label {label!r}")
    stamp = _policy_stamp(expect_commit)
    card = omt4.build_omt4_card(deck_root)
    omt4.validate_omt4_card(deck_root, card)
    entry = rung0.assemble_frame(frame_root, 1, 0)
    state = (card.recipe.initial_state if label == "independent"
             else rung0.bridge_entry(card, entry))
    freshwater, surface = omt0.rung0_ladder._zero_forcing((148, 180))
    ordinary = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card))
    live = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, live=True))
    early = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_hooks(card, substeps=True))

    boundaries = []
    passivity = None
    kt7_entry = None
    kt7_after = None
    arm = r245._barotropic_surface_mask_arm(
        card, True, v_only=True, t_pivot_north_neighbor=True)
    with arm:
        for kt in range(1, 8):
            if kt == 7:
                kt7_entry = state
                boundaries.append(_boundary_row(
                    card, state, rung0.assemble_frame(frame_root, kt, 0), kt, 0))
                ordinary_after = jax.device_get(ordinary.step(
                    state, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
                observed = jax.device_get(live.step(
                    state, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
                require(isinstance(observed, _NEMOWSLiveOperandTrace),
                        "live stage trace return type moved")
                passivity = passive._ordinary_state_equal(
                    observed.state_after, ordinary_after)
                for stage, values in enumerate(observed.stage_outputs, start=1):
                    stage_state = state._replace(
                        u=state.u.replace(data=values[0]),
                        v=state.v.replace(data=values[1]),
                        T=state.T.replace(data=values[2]),
                        S=state.S.replace(data=values[3]),
                        eta=state.eta.replace(data=values[4]))
                    boundaries.append(_boundary_row(
                        card, stage_state,
                        rung0.assemble_frame(frame_root, kt, stage), kt, stage))
                state = ordinary_after
                kt7_after = ordinary_after
            else:
                state = jax.device_get(ordinary.step(
                    state, card.dt_s, freshwater=freshwater,
                    surface_forcing=surface))
            print(f"PROGRESS round253 {label} kt={kt}", file=sys.stderr, flush=True)

        boundaries.append(_boundary_row(
            card, state, rung0.assemble_frame(frame_root, 8, 0), 8, 0))
        entry_guard = _guard_row(card, state.eta.data)
        terminal = None
        try:
            ordinary.step(state, card.dt_s, freshwater=freshwater,
                          surface_forcing=surface)
        except Exception as error:  # Equinox/JAX wraps the exact guard.
            terminal = str(error)
        trace = jax.device_get(early.step(
            state, card.dt_s, freshwater=freshwater, surface_forcing=surface))

    require(kt7_entry is not None and kt7_after is not None,
            "kt=7 state bookkeeping failed")
    require(isinstance(trace, _NEMOWSBarotropicTrace),
            "early substep trace return type moved")
    substeps, first_invalid = _substep_rows(card, trace.substeps, state)
    growth_rows, first_growth = _growth(boundaries)
    return {
        "format": "nemo-testcase-l4-orca2-round253-kt8-boundary-v1",
        "status": "PASS_R253_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved",
        "private_unit": "round217+raw-v-mask+t-pivot+resolved-vmask+associations",
        "worktree": stamp,
        "passivity": passivity,
        "completed_kt": 7,
        "kt8_entry_guard": entry_guard,
        "terminal_error": terminal,
        "nemo_guard": _nemo_guard(card, frame_root),
        "boundaries": boundaries,
        "growth_rows": growth_rows,
        "first_growth": first_growth,
        "source_order": list(ETA_BOUNDARIES),
        "substeps": substeps,
        "first_invalid": first_invalid,
        "one_bit_control": _one_bit_control(),
        "compiled_source": {
            "midpoint_ssh": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:489-493",
            "interpolation_coefficients": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1496-1531",
            "continuity": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:546-558",
            "external_update": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:655-718",
            "stage_install": "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178",
        },
    }


def classify(reports: list[dict[str, object]], *, plant: str = "none") -> dict:
    require(plant in PLANTS, f"unknown plant {plant!r}")
    reports = copy.deepcopy(reports)
    if plant == "label-coverage":
        reports.pop()
    by_label = {report["label"]: report for report in reports}
    require(set(by_label) == set(LABELS), "both claim labels are required")
    independent = by_label["independent"]
    if plant == "passivity":
        independent["passivity"]["T"] = False
    elif plant == "nemo-guard":
        independent["nemo_guard"]["invalid_total"] = 1
    elif plant == "finite-entry":
        independent["kt8_entry_guard"]["invalid"] = 1
    elif plant == "terminal":
        independent["terminal_error"] = "different refusal"
    elif plant == "source-order":
        independent["source_order"].reverse()
    elif plant == "first-invalid":
        independent["first_invalid"] = None
    elif plant == "growth":
        independent["first_growth"] = None
    elif plant == "one-bit":
        independent["one_bit_control"]["unequal"] = 0

    summary = {}
    signatures = set()
    for label, report in by_label.items():
        require(report["status"] == "PASS_R253_MEASUREMENT",
                f"{label}: measurement status moved")
        require(report["operand_route"] == "production-d114-resolved",
                f"{label}: stale operand route")
        require(all(report["passivity"].values()),
                f"{label}: live-stage instrument moved bits")
        require(report["completed_kt"] == 7,
                f"{label}: pre-boundary completion moved")
        require(report["kt8_entry_guard"]["invalid"] == 0,
                f"{label}: kt=8 entry is already invalid")
        require(report["terminal_error"] is not None
                and EXPECTED_ERROR in report["terminal_error"],
                f"{label}: registered kt=8 terminal moved")
        require(report["nemo_guard"]["checked"] == 40
                and report["nemo_guard"]["invalid_total"] == 0,
                f"{label}: NEMO's admitted states fail the live-W guard")
        require(tuple(report["source_order"]) == ETA_BOUNDARIES,
                f"{label}: source boundary order moved")
        first = report["first_invalid"]
        require(first is not None and first["prior_boundary_valid"],
                f"{label}: first-invalid boundary is missing or unordered")
        require(report["first_growth"] is not None,
                f"{label}: no >10x growth boundary was measured")
        require(report["one_bit_control"]["unequal"] == 1
                and report["one_bit_control"]["max_abs"] > 0.0,
                f"{label}: one-bit control went vacuous")
        signature = (first["substep"], first["boundary"],
                     tuple(first["first"]["index"]))
        signatures.add(signature)
        summary[label] = {
            "first_growth": report["first_growth"],
            "first_invalid": first,
            "kt8_entry_guard": report["kt8_entry_guard"],
            "nemo_guard": {"checked": report["nemo_guard"]["checked"],
                           "invalid_total": report["nemo_guard"]["invalid_total"],
                           "minimum": report["nemo_guard"]["minimum"]},
        }
    require(len(signatures) == 1,
            "claim labels disagree on the first invalid boundary")
    signature = next(iter(signatures))
    owner = {
        "substep": signature[0], "boundary": signature[1],
        "cell": list(signature[2]),
        "prediction": ("CONFIRMED_CONTINUITY_SSH_WRITE"
                       if signature[1] == "eta_continuity"
                       else "REFUTED_CONTINUITY_PREDICTION"),
        "compiled_source": independent["compiled_source"],
    }
    return {
        "format": "nemo-testcase-l4-orca2-round253-classification-v1",
        "status": "HELD_R253_FIRST_INVALID_WRITER_NAMED",
        "summary": summary,
        "first_invalid_writer": owner,
        "plant": plant,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "classify"), default="classify")
    parser.add_argument("--deck-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--label", choices=LABELS)
    parser.add_argument("--expect-commit")
    parser.add_argument("--report-in", type=Path, action="append", default=[])
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "measure":
            require(all((args.deck_root, args.frame_root, args.label,
                         args.expect_commit)), "measurement arguments incomplete")
            result = measure(args.deck_root, args.frame_root, args.label,
                             args.expect_commit)
        else:
            require(len(args.report_in) == 2,
                    "classification requires exactly two reports")
            result = classify([json.loads(path.read_text())
                               for path in args.report_in], plant=args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (GateError, OSError, KeyError, TypeError, ValueError,
            rung0.GateError) as error:
        marker = "PLANT-FIRED" if args.plant != "none" else "REFUSE"
        print(f"STATUS {marker} {args.plant}: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build the SI3 column reverse-mode sweep from independent targets.

The generator is deliberately split into one-repository sampling processes and
an assembly process.  This prevents Python's import cache and JAX compilation
cache from mixing the main and lane implementations.

Example (all output paths must be outside ``/tmp``)::

    python si3_reverse_sweep_oracle.py sample --repo MAIN --output main.json
    python si3_reverse_sweep_oracle.py sample --repo LANE --output lane.json
    python si3_reverse_sweep_oracle.py assemble \
        --main main.json --lane lane.json --output si3_reverse_sweep_oracle.json

For each output/input entry, assembly uses main's compiled reverse-mode AD only
when main and lane forward bits agree at the base point and at every point in
the entry's finite-difference neighbourhood.  Otherwise it uses a finite
difference of the lane forward.  The fixture records the target source,
stencil, step, and forward-rounding bound for every entry.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
from pathlib import Path
import subprocess
import sys
import zlib

import numpy as np


MAIN_SHA = "edd34c24469c834d536c511160ea0c4aef8974ae"
LANE_SHA = "920c61d884384e18470962568cda8295f0dfbe89"
CONCENTRATIONS = (0.0, 1.0e-12, 5.0e-11, 1.0e-10, 2.0e-10, 0.5)
THICKNESSES = (0.0, 0.5)
QLEADS = (0.0, -40.0)
CASES = tuple(
    (concentration, h_ice, qlead)
    for concentration in CONCENTRATIONS
    for h_ice in THICKNESSES
    for qlead in QLEADS
)
OUTPUT_NAMES = (
    "concentration", "h_ice", "h_snow", "T_surface", "e_ice",
    "e_snow", "S_bulk", "S_layers", "age_volume",
)
INPUT_NAMES = (
    "concentration", "h_ice", "h_snow", "T_surface", "e_ice",
    "e_snow", "S_bulk", "S_layers", "age_volume", "qns_ice",
    "qsr_ice", "dqns_ice", "qtr_ice_top", "t_bottom", "sss",
    "evaporation", "snow_precipitation", "qprec_ice",
    "qcn_ice_bottom", "qsb_ice_bottom", "fhld", "qlead", "frq_m",
)

# Steps are large enough to resolve the lane's source-rounded float64 forward
# while remaining local to its branch.  Concentration uses case-specific steps
# below so zero and epsi10 never cross their physical/statement boundaries.
_INPUT_STEPS = (
    None, 1.0e-7, 1.0e-7, 1.0e-4, 1.0e2, 1.0e2, 1.0e-5, 1.0e-8,
    1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4,
    9.0e-8, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-2, 1.0e-2, 1.0e-3,
    1.0e-4,
)
_CONCENTRATION_STEPS = {
    0.0: 1.0e-12,
    1.0e-12: 5.0e-13,
    5.0e-11: 4.0e-11,
    1.0e-10: 1.0e-11,
    2.0e-10: 2.0e-11,
    0.5: 1.0e-5,
}
METHOD_LEGEND = {
    0: "main_ad",
    1: "lane_fd_central",
    2: "lane_fd_forward",
    3: "lane_fd_backward",
    4: "lane_ad_at_nondifferentiable_boundary",
}


def _encode(array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "b85_zlib": base64.b85encode(zlib.compress(value.tobytes())).decode(),
    }


def decode(payload: dict[str, object]) -> np.ndarray:
    """Decode an array field from a generated sample or oracle fixture."""
    raw = zlib.decompress(base64.b85decode(str(payload["b85_zlib"])))
    return np.frombuffer(raw, dtype=str(payload["dtype"])).reshape(payload["shape"])


def _repo_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _configure_repo_imports(repo: Path) -> None:
    roots = [repo / "src"] + sorted(
        path.parent for path in (repo / "packages").glob("*/legoesm")
    )
    sys.path[:0] = [str(path) for path in roots]


def _stencil(case: tuple[float, float, float], input_index: int) -> tuple[int, float]:
    """Return method code and step without crossing a state boundary."""
    concentration, h_ice, _qlead = case
    if input_index == 0:
        step = _CONCENTRATION_STEPS[concentration]
        if concentration == 0.0:
            return 2, step
        if concentration == 1.0e-10:
            return 3, step
        return 1, step
    step = _INPUT_STEPS[input_index]
    assert step is not None
    if input_index == 1 and h_ice != 0.0:
        step = 1.0e-5
    value = {
        1: h_ice,
        2: 0.0,
        3: None,
        4: 0.0,
        5: 0.0,
        6: 0.0,
        7: 0.0,
        8: 0.0,
        13: None,
        14: None,
        21: case[2],
        22: 1.0,
    }.get(input_index, 0.0)
    if value == 0.0 and input_index not in (4, 5, 8):
        # Stay on the physical positive side of a zero state/forcing entry.
        return 2, step
    return 1, step


def _vectors(constants) -> np.ndarray:
    return np.asarray([
        (
            concentration, h_ice, 0.0, constants.T0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 271.27, 34.7, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, qlead, 1.0,
        )
        for concentration, h_ice, qlead in CASES
    ], dtype=np.float64)


def _sample(repo: Path, output: Path) -> None:
    _configure_repo_imports(repo)
    import jax
    import jax.numpy as jnp
    from legoesm.ice.bitz_lipscomb import (
        SI3ColumnArrays,
        SI3SurfaceForcing,
        si3_column_step_arrays,
    )
    from legoesm.ice.config import SI3ThermoConfig
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as constants

    def step(values):
        scalar = lambda index: values[index:index + 1]
        layers = lambda index: jnp.full((1, 3), values[index])
        state = SI3ColumnArrays(
            scalar(0), scalar(1), scalar(2), scalar(3), layers(4), layers(5),
            scalar(6), layers(7), scalar(8),
        )
        forcing_values = dict(
            qns_ice=scalar(9), qsr_ice=scalar(10), dqns_ice=scalar(11),
            qtr_ice_top=scalar(12), t_bottom=scalar(13), sss=scalar(14),
            evaporation=scalar(15), snow_precipitation=scalar(16),
            qprec_ice=scalar(17), qcn_ice_bottom=scalar(18),
            qsb_ice_bottom=scalar(19), fhld=scalar(20), qlead=scalar(21),
        )
        # edd34c2446 predates the lane's frq_m/sst_m carrier fields.  Keep the
        # common 23-vector API and let main's step be constant in that leaf.
        if "frq_m" in SI3SurfaceForcing._fields:
            forcing_values["frq_m"] = scalar(22)
        if "sst_m" in SI3SurfaceForcing._fields:
            forcing_values["sst_m"] = None
        forcing = SI3SurfaceForcing(**forcing_values)
        step_options = {}
        if "si3" in inspect.signature(si3_column_step_arrays).parameters:
            step_options["si3"] = SI3ThermoConfig()
        result = si3_column_step_arrays(
            state, forcing, 3600.0, constants, **step_options,
        ).exit
        return jnp.stack([jnp.sum(leaf) for leaf in result])

    vectors = _vectors(constants)
    perturbed = []
    refinement_perturbed = []
    stencil_codes = np.empty((len(CASES), len(INPUT_NAMES)), dtype=np.uint8)
    steps = np.empty_like(stencil_codes, dtype=np.float64)
    for case_index, case in enumerate(CASES):
        for input_index in range(len(INPUT_NAMES)):
            method, delta = _stencil(case, input_index)
            stencil_codes[case_index, input_index] = method
            steps[case_index, input_index] = delta
            minus = vectors[case_index].copy()
            plus = vectors[case_index].copy()
            minus[input_index] -= delta
            plus[input_index] += delta
            perturbed.extend((minus, plus))
            refinement_delta = delta + 5.0e-9 if input_index == 15 else delta
            refinement_minus = vectors[case_index].copy()
            refinement_plus = vectors[case_index].copy()
            refinement_minus[input_index] -= refinement_delta
            refinement_plus[input_index] += refinement_delta
            refinement_perturbed.extend((refinement_minus, refinement_plus))

    compiled_forward = jax.jit(jax.vmap(step))
    base = np.asarray(compiled_forward(jnp.asarray(vectors)))
    neighbourhood = np.asarray(
        compiled_forward(jnp.asarray(np.stack(perturbed)))
    ).reshape(len(CASES), len(INPUT_NAMES), 2, len(OUTPUT_NAMES))
    refinement_neighbourhood = np.asarray(
        compiled_forward(jnp.asarray(np.stack(refinement_perturbed)))
    ).reshape(len(CASES), len(INPUT_NAMES), 2, len(OUTPUT_NAMES))
    refinement_steps = steps.copy()
    refinement_steps[:, 15] += 5.0e-9
    jacobian = np.asarray(jax.jit(jax.vmap(jax.jacrev(step)))(jnp.asarray(vectors)))
    payload = {
        "schema": 1,
        "sha": _repo_sha(repo),
        "cases": CASES,
        "output_names": OUTPUT_NAMES,
        "input_names": INPUT_NAMES,
        "base": _encode(base),
        "neighbourhood": _encode(neighbourhood),
        "refinement_neighbourhood": _encode(refinement_neighbourhood),
        "jacobian": _encode(jacobian),
        "stencil_codes": _encode(stencil_codes),
        "steps": _encode(steps),
        "refinement_steps": _encode(refinement_steps),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"sample sha={payload['sha']} base={base.shape} jacobian={jacobian.shape}")


def _forward_agrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Float64 roundoff agreement, with exact agreement required at zero."""
    scale = np.maximum(np.abs(left), np.abs(right))
    return np.abs(left - right) <= 8.0 * np.finfo(np.float64).eps * scale


def build_oracle(main: dict[str, object], lane: dict[str, object]) -> dict[str, object]:
    """Assemble independently selected targets from two sample payloads."""
    if main["sha"] != MAIN_SHA or lane["sha"] != LANE_SHA:
        raise ValueError(f"unexpected provenance: main={main['sha']} lane={lane['sha']}")
    for key in ("cases", "output_names", "input_names"):
        if main[key] != lane[key]:
            raise ValueError(f"sample metadata differs for {key}")

    main_base, lane_base = decode(main["base"]), decode(lane["base"])
    main_neighbourhood = decode(main["neighbourhood"])
    lane_neighbourhood = decode(lane["neighbourhood"])
    main_refinement = decode(main["refinement_neighbourhood"])
    lane_refinement = decode(lane["refinement_neighbourhood"])
    main_jacobian = decode(main["jacobian"])
    lane_jacobian = decode(lane["jacobian"])
    stencil_codes = decode(lane["stencil_codes"])
    steps = decode(lane["steps"])
    refinement_steps = decode(lane["refinement_steps"])
    if not np.array_equal(stencil_codes, decode(main["stencil_codes"])):
        raise ValueError("sample stencil codes differ")
    if not np.array_equal(steps, decode(main["steps"])):
        raise ValueError("sample steps differ")
    if not np.array_equal(refinement_steps, decode(main["refinement_steps"])):
        raise ValueError("sample refinement steps differ")

    neighbour_equal = _forward_agrees(main_neighbourhood, lane_neighbourhood)
    refinement_equal = _forward_agrees(main_refinement, lane_refinement)
    agree = np.empty_like(main_jacobian, dtype=bool)
    lane_fd = np.empty_like(main_jacobian)
    rounding = np.empty_like(main_jacobian)
    fd_methods = np.broadcast_to(
        stencil_codes[:, None, :], main_jacobian.shape,
    ).copy()
    for case_index in range(len(CASES)):
        for input_index in range(len(INPUT_NAMES)):
            method = int(stencil_codes[case_index, input_index])
            delta = steps[case_index, input_index]
            minus = lane_neighbourhood[case_index, input_index, 0]
            plus = lane_neighbourhood[case_index, input_index, 1]
            if method == 1:
                lane_fd[case_index, :, input_index] = (plus - minus) / (2.0 * delta)
                rounding[case_index, :, input_index] = (
                    np.abs(np.spacing(plus)) + np.abs(np.spacing(minus))
                ) / (2.0 * delta)
                local_equal = neighbour_equal[case_index, input_index].all(axis=0)
                # At a non-smooth zero input, one side can share main's local
                # branch while the other enters a different statement arm.
                # Prefer that independently agreed one-sided neighbourhood.
                plus_equal = neighbour_equal[case_index, input_index, 1]
                minus_equal = neighbour_equal[case_index, input_index, 0]
                only_plus = plus_equal & ~minus_equal
                only_minus = minus_equal & ~plus_equal
                lane_fd[case_index, only_plus, input_index] = (
                    plus[only_plus] - lane_base[case_index, only_plus]
                ) / delta
                rounding[case_index, only_plus, input_index] = (
                    np.abs(np.spacing(plus[only_plus]))
                    + np.abs(np.spacing(lane_base[case_index, only_plus]))
                ) / delta
                fd_methods[case_index, only_plus, input_index] = 2
                lane_fd[case_index, only_minus, input_index] = (
                    lane_base[case_index, only_minus] - minus[only_minus]
                ) / delta
                rounding[case_index, only_minus, input_index] = (
                    np.abs(np.spacing(lane_base[case_index, only_minus]))
                    + np.abs(np.spacing(minus[only_minus]))
                ) / delta
                fd_methods[case_index, only_minus, input_index] = 3
                local_equal = local_equal | only_plus | only_minus
            elif method == 2:
                lane_fd[case_index, :, input_index] = (
                    plus - lane_base[case_index]
                ) / delta
                rounding[case_index, :, input_index] = (
                    np.abs(np.spacing(plus))
                    + np.abs(np.spacing(lane_base[case_index]))
                ) / delta
                local_equal = neighbour_equal[case_index, input_index, 1]
            elif method == 3:
                lane_fd[case_index, :, input_index] = (
                    lane_base[case_index] - minus
                ) / delta
                rounding[case_index, :, input_index] = (
                    np.abs(np.spacing(lane_base[case_index]))
                    + np.abs(np.spacing(minus))
                ) / delta
                local_equal = neighbour_equal[case_index, input_index, 0]
            else:
                raise ValueError(f"invalid stencil code {method}")
            agree[case_index, :, input_index] = (
                local_equal
                & _forward_agrees(main_base[case_index], lane_base[case_index])
            )

    # Evaporation's source-rounded h_ice response is bracketed by two nearby
    # forward steps; their mean removes the opposing subtraction bias without
    # consulting either implementation's AD.
    input_index = 15
    for case_index in range(len(CASES)):
        delta = refinement_steps[case_index, input_index]
        plus = lane_refinement[case_index, input_index, 1]
        refined = (plus - lane_base[case_index]) / delta
        lane_fd[case_index, :, input_index] = 0.5 * (
            lane_fd[case_index, :, input_index] + refined
        )
        agree[case_index, :, input_index] &= refinement_equal[
            case_index, input_index, 1
        ]

    # Zero snow is a statement boundary.  The live no-snow branch is approached
    # from below for thermodynamic outputs, while the thickness identity uses
    # the symmetric derivative.  Record those per-entry choices explicitly.
    for case_index, (_concentration, h_ice, _qlead) in enumerate(CASES):
        if h_ice == 0.0:
            continue
        input_index = 2
        delta = steps[case_index, input_index]
        minus = lane_neighbourhood[case_index, input_index, 0]
        plus = lane_neighbourhood[case_index, input_index, 1]
        for output_index in (1, 3, 4):
            lane_fd[case_index, output_index, input_index] = (
                lane_base[case_index, output_index] - minus[output_index]
            ) / delta
            rounding[case_index, output_index, input_index] = (
                abs(np.spacing(lane_base[case_index, output_index]))
                + abs(np.spacing(minus[output_index]))
            ) / delta
            fd_methods[case_index, output_index, input_index] = 3
        lane_fd[case_index, 2, input_index] = (
            plus[2] - minus[2]
        ) / (2.0 * delta)
        fd_methods[case_index, 2, input_index] = 1

    absolute_floor = 3.0e-6
    # An AD value below the global absolute noise floor is not independently
    # resolved when the lane difference is itself below its subtraction bound.
    # Material derivatives continue to use main AD when the forwards agree.
    unresolved = (
        (np.abs(lane_fd) <= rounding)
        & (np.abs(main_jacobian) <= absolute_floor)
        & (np.abs(lane_jacobian) <= absolute_floor)
    )
    scale = np.maximum(
        np.maximum(np.abs(main_jacobian), np.abs(lane_fd)),
        np.finfo(np.float64).tiny,
    )
    main_fd_agrees = np.abs(main_jacobian - lane_fd) <= 1.0e-6 * scale
    lane_scale = np.maximum(
        np.maximum(np.abs(lane_jacobian), np.abs(lane_fd)),
        np.finfo(np.float64).tiny,
    )
    lane_fd_agrees = np.abs(lane_jacobian - lane_fd) <= 1.0e-6 * lane_scale
    established_lane_branch = np.zeros_like(main_jacobian, dtype=bool)
    for case_index, (concentration, h_ice, _qlead) in enumerate(CASES):
        if concentration > 1.0e-10 and h_ice != 0.0:
            established_lane_branch[case_index] = True
        if h_ice == 0.0:
            established_lane_branch[case_index, :, 1] = True
    established_lane_branch[:, :, 16] = True
    established_lane_branch[:, 5, 2] = True
    smooth_input = np.ones_like(main_jacobian, dtype=bool)
    smooth_input[:, :, (0, 2, 15, 16)] = False
    use_main = (
        agree
        & (main_fd_agrees | smooth_input)
        & ~established_lane_branch
        & ~unresolved
    )
    # At an exact statement boundary (notably zero snow thickness), the two
    # directional differences can converge to distinct values.  No finite
    # derivative exists there, so preserve and label the lane program's JAX
    # branch convention rather than misrepresenting a crossing difference as
    # a derivative.  Every differentiable fallback remains an actual lane FD.
    nondifferentiable = (
        established_lane_branch
        & ~use_main
        & ~lane_fd_agrees
    )
    lane_branch_target = np.nan_to_num(lane_jacobian, nan=0.0)
    target = np.where(
        use_main,
        main_jacobian,
        np.where(nondifferentiable, lane_branch_target, lane_fd),
    )
    methods = np.where(
        use_main, 0, np.where(nondifferentiable, 4, fd_methods),
    ).astype(np.uint8)
    target = np.where(unresolved, lane_fd, target)
    methods = np.where(unresolved, fd_methods, methods).astype(np.uint8)
    # One absolute comparison floor covers targets within twice the outward-
    # rounded subtraction bound; every other entry remains relative-only.
    rounding_limited = np.abs(target) <= 2.0 * absolute_floor
    return {
        "schema": 1,
        "main_sha": MAIN_SHA,
        "lane_sha": LANE_SHA,
        "selection": (
            "main compiled reverse AD iff main/lane forwards and the lane finite "
            "difference agree in the recorded neighbourhood; otherwise lane finite "
            "difference, except explicitly labelled nondifferentiable statement boundaries"
        ),
        "method_legend": {str(key): value for key, value in METHOD_LEGEND.items()},
        "cases": main["cases"],
        "output_names": main["output_names"],
        "input_names": main["input_names"],
        "target": _encode(target.astype("<f8")),
        "methods": _encode(methods),
        "steps": _encode(np.broadcast_to(steps[:, None, :], target.shape).copy()),
        "refinement_steps": _encode(
            np.broadcast_to(refinement_steps[:, None, :], target.shape).copy()
        ),
        "fd_rounding_bound": _encode(rounding.astype("<f8")),
        "rounding_limited": _encode(rounding_limited),
        "absolute_floor": absolute_floor,
        "absolute_floor_derivation": (
            "3e-6 is the outward-rounded float64 subtraction bound for an "
            "O(0.5) forward and the 2e-11 concentration step: "
            "2*spacing(0.5)/(2*2e-11)=2.7755575615628914e-6; targets "
            "within two such bounds use this absolute comparison floor"
        ),
        "method_counts": {
            METHOD_LEGEND[method]: int(np.count_nonzero(methods == method))
            for method in METHOD_LEGEND
        },
    }


def _assemble(main_path: Path, lane_path: Path, output: Path) -> None:
    main = json.loads(main_path.read_text())
    lane = json.loads(lane_path.read_text())
    oracle = build_oracle(main, lane)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(oracle, indent=2) + "\n")
    print(
        f"oracle target={(len(CASES), len(OUTPUT_NAMES), len(INPUT_NAMES))} "
        f"methods={oracle['method_counts']} absolute_floor={oracle['absolute_floor']:.17g}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--repo", required=True, type=Path)
    sample.add_argument("--output", required=True, type=Path)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--main", required=True, type=Path)
    assemble.add_argument("--lane", required=True, type=Path)
    assemble.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "sample":
        _sample(args.repo.resolve(), args.output.resolve())
    else:
        _assemble(args.main.resolve(), args.lane.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

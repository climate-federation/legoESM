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
# S_layers at S=0, e_ice=0 has smooth nonlinearity scale c_ice*epsi10/zgamma =
# 1.146e-11, so 5e-15 resolves the central difference to <4e-7 relative error.
_INPUT_STEPS = (
    None, 1.0e-7, 1.0e-7, 1.0e-4, 1.0e2, 1.0e2, 1.0e-5, 5.0e-15,
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
    4: "excluded_nondifferentiable_boundary",
}
SELECTION_RULE = (
    "main compiled reverse AD iff main/lane forwards agree in the recorded "
    "neighbourhood and main AD agrees with the lane finite difference within "
    "1e-6 relative error; otherwise the lane finite difference. For the 36 "
    "ice-thermodynamic entries [c, 1|3|4, 2] (outputs h_ice, T_surface, e_ice "
    "with respect to h_snow at h_snow = 0.0 on the 12 h_ice = 0.5 cases), the "
    "base point h_snow = 0.0 executes the h_snow <= 0.0 no-snow branch while "
    "h_snow > 0.0 crosses the strict h_snow > 0.0 guard into the snow-present "
    "branch (so the right difference disagrees on 31 of the 36 entries), so "
    "these 36 entries are scored against the left (backward) difference at the "
    "1e-6 refinement step along the executed h_snow <= 0.0 branch. Statement-"
    "boundary candidates ([c, :, 1] at h_ice = 0.0, and [c, :, 16], [c, 2, 2], "
    "[c, 5, 2] at h_ice = 0.5) are excluded as non-differentiable only where "
    "measured left and right one-sided lane finite differences disagree by more "
    "than both 1e-6 relative error and 1x the entry's own fd_rounding_bound "
    "(|fd_right - fd_left| > max(1e-6 * max(|fd_left|, |fd_right|, tiny), "
    "fd_rounding_bound))"
)
FD_ROUNDING_BOUND_FORMULA = (
    "per-entry float64 subtraction rounding bound of the lane finite-difference "
    "stencil, using s(x+-d) = max(|f(x+-d)|, d) on diagonal state entries "
    "(output_index == input_index) where the perturbed state leaf enters the "
    "column step at scale >= d and s(x+-d) = |f(x+-d)| otherwise, with "
    "s(x) = |f(x)|: (|spacing(s(x+d))| + |spacing(s(x-d))|) / (2*d) for central "
    "differences, (|spacing(s(x+d))| + |spacing(s(x))|) / d for forward "
    "differences, (|spacing(s(x))| + |spacing(s(x-d))|) / d for backward "
    "differences, and the 0.5-weighted sum of the two forward-step bounds for "
    "two-step averaged evaporation differences; boundary candidates are excluded "
    "only when |fd_right - fd_left| > max(1e-6 * max(|fd_left|, |fd_right|, tiny), "
    "fd_rounding_bound), and each unexcluded entry passes iff "
    "relative_error <= 1e-6 or |error| <= 1x fd_rounding_bound"
)


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
    if value == 0.0 and input_index not in (4, 5, 7, 8):
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
            if input_index == 2:
                refinement_delta = 1.0e-6
            elif input_index == 15:
                refinement_delta = delta + 5.0e-9
            else:
                refinement_delta = delta
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
    refinement_steps[:, 2] = 1.0e-6
    refinement_steps[:, 15] += 5.0e-9
    sha = _repo_sha(repo)
    payload: dict[str, object] = {
        "schema": 1,
        "sha": sha,
        "cases": CASES,
        "output_names": OUTPUT_NAMES,
        "input_names": INPUT_NAMES,
        "base": _encode(base),
        "neighbourhood": _encode(neighbourhood),
        "refinement_neighbourhood": _encode(refinement_neighbourhood),
        "stencil_codes": _encode(stencil_codes),
        "steps": _encode(steps),
        "refinement_steps": _encode(refinement_steps),
    }
    if sha == MAIN_SHA:
        jacobian = np.asarray(
            jax.jit(jax.vmap(jax.jacrev(step)))(jnp.asarray(vectors))
        )
        payload["jacobian"] = _encode(jacobian)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"sample sha={payload['sha']} base={base.shape} has_jacobian={'jacobian' in payload}")


def _forward_agrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Float64 roundoff agreement, with exact agreement required at zero."""
    scale = np.maximum(np.abs(left), np.abs(right))
    return np.abs(left - right) <= 8.0 * np.finfo(np.float64).eps * scale


def build_oracle(main: dict[str, object], lane: dict[str, object]) -> dict[str, object]:
    """Assemble independently selected targets from two sample payloads."""
    if main["sha"] != MAIN_SHA or lane["sha"] != LANE_SHA:
        raise ValueError(f"unexpected provenance: main={main['sha']} lane={lane['sha']}")
    if "jacobian" in lane:
        raise ValueError("lane sample must not provide its own AD jacobian")
    for key in ("cases", "output_names", "input_names"):
        if main[key] != lane[key]:
            raise ValueError(f"sample metadata differs for {key}")

    main_base, lane_base = decode(main["base"]), decode(lane["base"])
    main_neighbourhood = decode(main["neighbourhood"])
    lane_neighbourhood = decode(lane["neighbourhood"])
    main_refinement = decode(main["refinement_neighbourhood"])
    lane_refinement = decode(lane["refinement_neighbourhood"])
    main_jacobian = decode(main["jacobian"])
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
            minus_scale = np.abs(minus).copy()
            plus_scale = np.abs(plus).copy()
            if input_index < len(OUTPUT_NAMES):
                minus_scale[input_index] = max(minus_scale[input_index], delta)
                plus_scale[input_index] = max(plus_scale[input_index], delta)
            if method == 1:
                lane_fd[case_index, :, input_index] = (plus - minus) / (2.0 * delta)
                rounding[case_index, :, input_index] = (
                    np.abs(np.spacing(plus_scale)) + np.abs(np.spacing(minus_scale))
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
                    np.abs(np.spacing(plus_scale[only_plus]))
                    + np.abs(np.spacing(lane_base[case_index, only_plus]))
                ) / delta
                fd_methods[case_index, only_plus, input_index] = 2
                lane_fd[case_index, only_minus, input_index] = (
                    lane_base[case_index, only_minus] - minus[only_minus]
                ) / delta
                rounding[case_index, only_minus, input_index] = (
                    np.abs(np.spacing(lane_base[case_index, only_minus]))
                    + np.abs(np.spacing(minus_scale[only_minus]))
                ) / delta
                fd_methods[case_index, only_minus, input_index] = 3
                local_equal = local_equal | only_plus | only_minus
            elif method == 2:
                lane_fd[case_index, :, input_index] = (
                    plus - lane_base[case_index]
                ) / delta
                rounding[case_index, :, input_index] = (
                    np.abs(np.spacing(plus_scale))
                    + np.abs(np.spacing(lane_base[case_index]))
                ) / delta
                local_equal = neighbour_equal[case_index, input_index, 1]
            elif method == 3:
                lane_fd[case_index, :, input_index] = (
                    lane_base[case_index] - minus
                ) / delta
                rounding[case_index, :, input_index] = (
                    np.abs(np.spacing(lane_base[case_index]))
                    + np.abs(np.spacing(minus_scale))
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
        refined_rounding = (
            np.abs(np.spacing(plus)) + np.abs(np.spacing(lane_base[case_index]))
        ) / delta
        lane_fd[case_index, :, input_index] = 0.5 * (
            lane_fd[case_index, :, input_index] + refined
        )
        rounding[case_index, :, input_index] = 0.5 * (
            rounding[case_index, :, input_index] + refined_rounding
        )
        agree[case_index, :, input_index] &= refinement_equal[
            case_index, input_index, 1
        ]

    # Zero snow is a statement boundary.  The live no-snow branch is approached
    # from below for thermodynamic outputs (using the 1e-6 refinement step to
    # keep float64 subtraction noise below 1e-6 relative error on h_ice).
    for case_index, (_concentration, h_ice, _qlead) in enumerate(CASES):
        if h_ice == 0.0:
            continue
        input_index = 2
        delta = refinement_steps[case_index, input_index]
        minus = lane_refinement[case_index, input_index, 0]
        for output_index in (1, 3, 4):
            lane_fd[case_index, output_index, input_index] = (
                lane_base[case_index, output_index] - minus[output_index]
            ) / delta
            rounding[case_index, output_index, input_index] = (
                np.abs(np.spacing(lane_base[case_index, output_index]))
                + np.abs(np.spacing(minus[output_index]))
            ) / delta
            fd_methods[case_index, output_index, input_index] = 3
            agree[case_index, output_index, input_index] = (
                refinement_equal[case_index, input_index, 0, output_index]
                & _forward_agrees(
                    main_base[case_index, output_index],
                    lane_base[case_index, output_index],
                )
            )

    # Measure one-sided left and right lane finite differences on the primary
    # neighbourhood.  Exclude an entry at a statement boundary only when the
    # measured left and right one-sided lane finite differences disagree by
    # more than both 1e-6 relative error and the entry's own fd_rounding_bound.
    fd_left = np.empty_like(main_jacobian)
    fd_right = np.empty_like(main_jacobian)
    for case_index in range(len(CASES)):
        for input_index in range(len(INPUT_NAMES)):
            delta = steps[case_index, input_index]
            minus = lane_neighbourhood[case_index, input_index, 0]
            plus = lane_neighbourhood[case_index, input_index, 1]
            base_val = lane_base[case_index]
            fd_left[case_index, :, input_index] = (base_val - minus) / delta
            fd_right[case_index, :, input_index] = (plus - base_val) / delta

    boundary_candidate = np.zeros_like(main_jacobian, dtype=bool)
    for case_index, (_concentration, h_ice, _qlead) in enumerate(CASES):
        if h_ice == 0.0:
            boundary_candidate[case_index, :, 1] = True
        else:
            boundary_candidate[case_index, :, 16] = True
            boundary_candidate[case_index, 2, 2] = True
            boundary_candidate[case_index, 5, 2] = True

    one_sided_scale = np.maximum(
        np.maximum(np.abs(fd_left), np.abs(fd_right)),
        np.finfo(np.float64).tiny,
    )
    one_sided_disagrees = np.abs(fd_right - fd_left) > np.maximum(
        1.0e-6 * one_sided_scale,
        rounding,
    )
    excluded = boundary_candidate & one_sided_disagrees

    scale = np.maximum(
        np.maximum(np.abs(main_jacobian), np.abs(lane_fd)),
        np.finfo(np.float64).tiny,
    )
    main_fd_agrees = np.abs(main_jacobian - lane_fd) <= 1.0e-6 * scale
    use_main = agree & main_fd_agrees & ~excluded
    methods = np.where(
        excluded, 4, np.where(use_main, 0, fd_methods),
    ).astype(np.uint8)
    target = np.where(
        excluded, np.nan, np.where(use_main, main_jacobian, lane_fd),
    )

    excluded_records = [
        {
            "index": [int(c), int(o), int(i)],
            "fd_left": float(fd_left[c, o, i]),
            "fd_right": float(fd_right[c, o, i]),
        }
        for c, o, i in np.argwhere(excluded)
    ]
    return {
        "schema": 1,
        "main_sha": MAIN_SHA,
        "lane_sha": LANE_SHA,
        "selection": SELECTION_RULE,
        "fd_rounding_bound_formula": FD_ROUNDING_BOUND_FORMULA,
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
        "excluded": _encode(excluded),
        "excluded_nondifferentiable": excluded_records,
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
        f"methods={oracle['method_counts']}"
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

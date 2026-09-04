#!/usr/bin/env python3
"""Walk GYRE's RK3 stage-1 slow-forcing construction in source order.

The oracle record is a WRITE-only, config-local extension of ``stp2d.F90``.
It captures the live RK3 path at ``:126-202``: the 3-D RHS operands, completed
reference-thickness depth mean, post-``dyn_drg_init`` value, wind operands,
and post-wind ``Ue_rhs/Ve_rhs``.  ``dynspg_ts.F90:280-300`` subsequently
copies these fields and removes the 2-D Coriolis trend.

The legoESM values come from the same production-jitted step through the
private ``_NEMOWSRK3TestHooks`` trace.  No eager or alternate step is used.
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
    expected_masks,
    require,
    sha256,
)
from nemo_testcase_state_ulp_probe import ulp_distance

DEFAULT_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round16_oracle_v2_slow_v2")


def _full_2d(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F").T


def _interior_2d(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx - 4, ny - 4), order="F").T


def _full_3d(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F").transpose(1, 0, 2)


def read_slow_forcing(path: Path) -> dict[str, object]:
    """Read ``NEMO_L2_SLOW_2`` with explicit size and EOF checks."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        version, kt, kbb, krhs, nx, ny, nz, bits = header
        sizes = struct.unpack("=7i", handle.read(28))
        expected_sizes = (nx * ny * nz,) * 6 + ((nx - 4) * (ny - 4),)
        require(
            (magic, *header, sizes) == ("NEMO_L2_SLOW_2", 2, 1, 1, 3, *DIMS, 64, expected_sizes),
            f"{path}: bad header {(magic, *header, sizes)}",
        )

        def field3() -> np.ndarray:
            values = np.fromfile(handle, dtype=np.float64, count=nx * ny * nz)
            require(values.size == nx * ny * nz, f"{path}: truncated 3-D field")
            return _full_3d(values, nx, ny, nz)[2:-2, 2:-2, : nz - 1]

        def full2() -> np.ndarray:
            values = np.fromfile(handle, dtype=np.float64, count=nx * ny)
            require(values.size == nx * ny, f"{path}: truncated full 2-D field")
            return _full_2d(values, nx, ny)

        def interior2() -> np.ndarray:
            n = (nx - 4) * (ny - 4)
            values = np.fromfile(handle, dtype=np.float64, count=n)
            require(values.size == n, f"{path}: truncated interior 2-D field")
            return _interior_2d(values, nx, ny)

        result = {
            "e3u": field3(),
            "krhs_u": field3(),
            "umask": field3(),
            "e3v": field3(),
            "krhs_v": field3(),
            "vmask": field3(),
            "depth_u": interior2(),
            "depth_v": interior2(),
            "r1_hu0": full2()[2:-2, 2:-2],
            "r1_hv0": full2()[2:-2, 2:-2],
            "post_drag_u": interior2(),
            "post_drag_v": interior2(),
            "cd_u": full2()[2:-2, 2:-2],
            "cd_v": full2()[2:-2, 2:-2],
        }
        scalar = np.fromfile(handle, dtype=np.float64, count=1)
        require(scalar.size == 1, f"{path}: truncated r1_rho0")
        result["r1_rho0"] = float(scalar[0])
        result["utau"] = full2()[2:-2, 2:-2]
        result["vtau"] = full2()[2:-2, 2:-2]
        result["r1_hu"] = full2()[2:-2, 2:-2]
        result["r1_hv"] = full2()[2:-2, 2:-2]
        result["post_wind_u"] = interior2()
        result["post_wind_v"] = interior2()
        require(handle.read(1) == b"", f"{path}: trailing payload")
    result["header"] = {
        "version": version,
        "kt": kt,
        "Kbb": kbb,
        "Krhs": krhs,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "bits": bits,
        "sizes": sizes,
    }
    return result


def compare(candidate, oracle, active) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    require(
        candidate.shape == oracle.shape == active.shape,
        f"comparison shape mismatch: {candidate.shape}, {oracle.shape}, {active.shape}",
    )
    delta = candidate[active] - oracle[active]
    return {
        "bit_exact": bool(np.array_equal(candidate[active], oracle[active])),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        "ulp_max": int(np.max(ulp_distance(candidate[active], oracle[active]), initial=0)),
        "differing_cells": int(np.count_nonzero(delta)),
    }


def _source_sum(e3, rhs, mask, reciprocal) -> np.ndarray:
    """Literal left-to-right transcription of the scalar-math Fortran SUM."""
    product = (e3 * rhs) * mask
    total = np.array(product[..., 0], copy=True)
    for level in range(1, product.shape[-1] - 1):
        total = total + product[..., level]
    return total * reciprocal


def _native_u(values) -> np.ndarray:
    return np.asarray(values)[:, 1:, :]


def _native_v(values) -> np.ndarray:
    return np.asarray(values)[1:, :, :]


def _native_u2(values) -> np.ndarray:
    return np.asarray(values)[:, 1:]


def _native_v2(values) -> np.ndarray:
    return np.asarray(values)[1:, :]


def run(root: Path, *, plant: bool) -> dict[str, object]:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    path = root / "oracle_slow_forcing_kt00000001.bin"
    require(path.is_file(), f"missing {path}")
    oracle = read_slow_forcing(path)
    card = build_nemo_testcase_card(CASE)
    config = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True
    )
    seed_model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, config)
    state = seed_model._seed_tke_preclosure_carry(card.recipe.initial_state)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True),
    )
    model.prime_step_caches(state)
    trace = model.step(state, card.dt_s, freshwater=freshwater, surface_forcing=surface)
    trace = jax.tree_util.tree_map(
        lambda value: np.asarray(value) if isinstance(value, jax.Array) else value,
        trace,
    )
    operands = trace.slow_forcing_operands
    masks = expected_masks(card)
    active_u3 = np.asarray(masks["u"], dtype=bool)
    active_v3 = np.asarray(masks["v"], dtype=bool)
    active_u2 = active_u3[..., 0]
    active_v2 = active_v3[..., 0]

    candidate = {
        "e3u": _native_u(operands["h_u"]),
        "krhs_u": _native_u(operands["du_dt"]),
        "umask": active_u3.astype(np.float64),
        "e3v": _native_v(operands["h_v"]),
        "krhs_v": _native_v(operands["dv_dt"]),
        "vmask": active_v3.astype(np.float64),
        "r1_hu0": 1.0 / _native_u2(operands["H_u"]),
        "r1_hv0": 1.0 / _native_v2(operands["H_v"]),
        "depth_u": _native_u2(operands["depth_u"]),
        "depth_v": _native_v2(operands["depth_v"]),
        "post_wind_u": _native_u2(operands["post_wind_u"]),
        "post_wind_v": _native_v2(operands["post_wind_v"]),
        "post_drag_u": _native_u2(operands["post_drag_u"]),
        "post_drag_v": _native_v2(operands["post_drag_v"]),
        "pre_external_u": _native_u2(operands["pre_external_u"]),
        "pre_external_v": _native_v2(operands["pre_external_v"]),
    }
    if plant:
        candidate["e3u"] = np.array(candidate["e3u"], copy=True)
        index = tuple(np.argwhere(active_u3)[0])
        candidate["e3u"][index] = np.nextafter(candidate["e3u"][index], np.inf)

    operand_rows = {}
    for face, active3, active2 in (
        ("u", active_u3, active_u2),
        ("v", active_v3, active_v2),
    ):
        operand_rows[face] = {
            "e3": compare(candidate[f"e3{face}"], oracle[f"e3{face}"], active3),
            "Krhs": compare(candidate[f"krhs_{face}"], oracle[f"krhs_{face}"], active3),
            "mask": compare(candidate[f"{face}mask"], oracle[f"{face}mask"], active3),
            "r1_h0": compare(candidate[f"r1_h{face}0"], oracle[f"r1_h{face}0"], active2),
        }

    source_replays = {}
    boundary_rows = {}
    arms = {}
    for face, active2 in (("u", active_u2), ("v", active_v2)):
        literal_oracle = _source_sum(
            oracle[f"e3{face}"],
            oracle[f"krhs_{face}"],
            oracle[f"{face}mask"],
            oracle[f"r1_h{face}0"],
        )
        literal_candidate = _source_sum(
            candidate[f"e3{face}"],
            candidate[f"krhs_{face}"],
            candidate[f"{face}mask"],
            candidate[f"r1_h{face}0"],
        )
        oracle_krhs_arm = _source_sum(
            candidate[f"e3{face}"],
            oracle[f"krhs_{face}"],
            candidate[f"{face}mask"],
            candidate[f"r1_h{face}0"],
        )
        source_replays[face] = {
            "oracle_operands_vs_oracle_depth_mean": compare(
                literal_oracle, oracle[f"depth_{face}"], active2
            ),
            "candidate_operands_vs_oracle_depth_mean": compare(
                literal_candidate, oracle[f"depth_{face}"], active2
            ),
            "oracle_Krhs_only_vs_oracle_depth_mean": compare(
                oracle_krhs_arm, oracle[f"depth_{face}"], active2
            ),
        }
        boundary_rows[face] = {
            "production_depth_mean": compare(
                candidate[f"depth_{face}"], oracle[f"depth_{face}"], active2
            ),
            # NEMO applies drag before wind. legoESM's trace applies the
            # identically-zero rest-state drag after wind, so its depth mean
            # is the source-equivalent post-drag value and its post-drag value
            # is the source-equivalent post-wind value.
            "post_drag": compare(
                candidate[f"depth_{face}"], oracle[f"post_drag_{face}"], active2
            ),
            "post_wind": compare(
                candidate[f"post_drag_{face}"], oracle[f"post_wind_{face}"], active2
            ),
            # At kt=1 from rest dynspg_ts.F90:280-300 removes an identically
            # zero 2-D Coriolis term, so NEMO's post-wind field is also the
            # pre-external-mode slow forcing.
            "pre_external": compare(
                candidate[f"pre_external_{face}"], oracle[f"post_wind_{face}"], active2
            ),
        }
        faithful = boundary_rows[face]["production_depth_mean"]
        arm = compare(literal_candidate, oracle[f"depth_{face}"], active2)
        movement = float(
            np.max(
                np.abs(literal_candidate[active2] - candidate[f"depth_{face}"][active2]),
                initial=0.0,
            )
        )
        arms[face] = {
            "association": {
                "changed_operands": ["depth-reduction association"],
                "faithful_residual": faithful["absolute_max"],
                "arm_residual": arm["absolute_max"],
                "causal_movement": movement,
                "movement_over_faithful_residual": (
                    movement / faithful["absolute_max"] if faithful["absolute_max"] else None
                ),
                "label": (
                    "CONFIRMED_CAUSAL_CONTRIBUTOR_NOT_OWNER"
                    if 0.0 < arm["absolute_max"] < faithful["absolute_max"]
                    else "REFUTED"
                ),
            },
            "Krhs": {
                "changed_operands": ["Krhs"],
                "literal_candidate_residual": arm["absolute_max"],
                "arm_residual": source_replays[face]["oracle_Krhs_only_vs_oracle_depth_mean"][
                    "absolute_max"
                ],
                "label": (
                    "BIT_EXACT_NO_ARM_NEEDED"
                    if operand_rows[face]["Krhs"]["bit_exact"]
                    else (
                        "CONFIRMED_UPSTREAM_OWNER"
                        if source_replays[face]["oracle_Krhs_only_vs_oracle_depth_mean"]["bit_exact"]
                        else "CAUSAL_CONTRIBUTOR"
                    )
                ),
            },
        }

    first = None
    order = ("e3", "Krhs", "mask", "r1_h0")
    for boundary in order:
        for face in ("u", "v"):
            row = operand_rows[face][boundary]
            if first is None and not row["bit_exact"]:
                first = {"boundary": boundary, "face": face, **row}
    if first is None:
        for boundary in (
            "production_depth_mean", "post_drag", "post_wind", "pre_external"
        ):
            for face in ("u", "v"):
                row = boundary_rows[face][boundary]
                if first is None and not row["bit_exact"]:
                    first = {"boundary": boundary, "face": face, **row}
    all_exact = first is None
    plant_control = {
        "requested": plant,
        "first_boundary": None if first is None else first["boundary"],
        "fires": (not plant) or (
            first is not None and first["boundary"] == "e3"
        ),
    }
    if not plant_control["fires"]:
        raise AssertionError("planted e3 operand did not become the first mismatch")

    return {
        "format": "nemo-testcase-l2-gyre-round16-slow-forcing-v1",
        "status": "AT-BAR" if all_exact else "DEBT",
        "regime": "production-jit/cpu/fp64/libm",
        "plant": plant,
        "oracle_root": str(root),
        "source_citations": {
            "three_dimensional_rhs": "stp2d.F90:126-171",
            "depth_mean": "stp2d.F90:177-181",
            "drag": "stp2d.F90:196",
            "wind": "stp2d.F90:198-202",
            "external_copy_and_coriolis_removal": "dynspg_ts.F90:280-300",
        },
        "artifacts": {
            path.name: sha256(path),
            "stp2d.F90": sha256(root / "stp2d.F90"),
            "nemo.exe": sha256(root / "nemo"),
        },
        "oracle_header": oracle["header"],
        "first_non_bit_exact_primitive": first,
        "primitive_operands": operand_rows,
        "source_replays": source_replays,
        "boundary_rows": boundary_rows,
        "one_variable_association_arms": arms,
        "owner_label": (
            "CONFIRMED_STAGE1_SLOW_FORCING_CHAIN"
            if all_exact else "UNMEASURED_AFTER_FIRST_NONEXACT_BOUNDARY"
        ),
        "scaling_before_owner": True,
        "plant_control": plant_control,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    report = run(args.oracle_root, plant=args.plant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"GYRE_SLOW_FORCING {report['status']}: "
        f"first={report['first_non_bit_exact_primitive']} plant={args.plant}",
        file=sys.stderr,
    )
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Source-ordered GYRE stage-1 tracer walk (production JIT, fp64)."""

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
    lego_fields,
    read_entry,
    require,
    sha256,
)
from nemo_testcase_state_ulp_probe import ulp_distance


ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round13_oracle_tracer_v1"
)
CONTROL_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round12_oracle_eos_v2"
)


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_stage1_tracer(path: Path) -> dict:
    """Read the WRITE-only stage-1 Krhs/update stream in source order."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "now", f"{path}: wrong registry level")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKTRA_1", f"{path}: bad magic {magic!r}")
    require(
        (version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
        == (1, 1, 1, 1, 1, 3, 3, *DIMS, 64),
        f"{path}: bad header {header}",
    )
    n3 = nx * ny * nz
    n2 = nx * ny
    require(values.size == 15 * n3 + 3 * n2, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    names = (
        "zero_T", "zero_S", "zFu", "zFv", "zFw",
        "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S",
        "Kbb_T", "Kbb_S", "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result = {
        name: _xyz(values[index * n3:(index + 1) * n3], nx, ny, nz)
        for index, name in enumerate(names)
    }
    offset = 15 * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        begin = offset + index * n2
        result[name] = _xy(values[begin:begin + n2], nx, ny)
    result["header"] = {
        "version": version, "kt": kt, "stage": stage, "Kbb": kbb,
        "Kmm": kmm, "Krhs": krhs, "Kaa": kaa, "bits": bits,
        "registry_level": "now",
    }
    return result


def _bits(value: np.float64) -> str:
    return f"0x{int(np.asarray(value, dtype=np.float64).view(np.uint64)):016x}"


def _cell_value(oracle, candidate, index) -> dict:
    ov = np.float64(oracle[index])
    cv = np.float64(candidate[index])
    return {
        "oracle": float(ov),
        "candidate": float(cv),
        "oracle_bits": _bits(ov),
        "candidate_bits": _bits(cv),
        "absolute_delta": float(cv - ov),
        "ulp": int(ulp_distance(np.asarray([cv]), np.asarray([ov]))[0]),
        "bit_exact": bool(cv == ov),
    }


def _run_boundary(card, cfg, boundary: str):
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    hooks = (
        _NEMOWSRK3TestHooks(expose_tracer_stage=1)
        if boundary == "after_update"
        else _NEMOWSRK3TestHooks(expose_tracer_stage1_boundary=boundary)
    )
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks,
    )
    state = model.step(
        card.recipe.initial_state, dt=card.dt_s,
        freshwater=freshwater, surface_forcing=surface,
    )
    host = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
        state,
    )
    jax.clear_caches()
    fields = lego_fields(host)
    return fields["T"], fields["S"]


def run(oracle_root: Path, control_root: Path, *, plant: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    tracer_path = oracle_root / "oracle_rktracer_operands_kt00000001_s1.bin"
    require(tracer_path.is_file(), f"missing {tracer_path}")
    oracle = read_stage1_tracer(tracer_path)
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)
    active = masks["T"]
    nlev = card.recipe.z_coord.n_levels

    candidate = {
        "after_advection": _run_boundary(card, cfg, "after_advection"),
        "after_sbc": _run_boundary(card, cfg, "after_sbc"),
        "after_update": _run_boundary(card, cfg, "after_update"),
    }
    candidate = {
        boundary: (np.asarray(pair[0])[..., :nlev], np.asarray(pair[1])[..., :nlev])
        for boundary, pair in candidate.items()
    }
    if plant:
        planted = candidate["after_update"][0].copy()
        oracle_update = oracle["Kaa_T"][..., :nlev]
        equal_wet = active & (planted == oracle_update)
        require(bool(equal_wet.any()), "plant has no bit-identical wet target")
        index = tuple(np.argwhere(equal_wet)[0])
        planted[index] = np.nextafter(planted[index], np.inf)
        candidate["after_update"] = (planted, candidate["after_update"][1])

    # The newly instrumented oracle must not perturb ordinary prognostics.
    bit_identity = {}
    for name in (
        "oracle_stage_kt00000001_s1.bin",
        "oracle_stage_kt00000001_s2.bin",
        "oracle_stage_kt00000001_s3.bin",
        "GYRE_OMIP_L2_P3_00000010_restart.nc",
    ):
        measured = oracle_root / name
        control = control_root / name
        require(measured.is_file() and control.is_file(), f"missing identity control {name}")
        measured_hash, control_hash = sha256(measured), sha256(control)
        require(measured_hash == control_hash, f"instrument changed {name}")
        bit_identity[name] = measured_hash

    entry = read_entry(oracle_root / "oracle_step_entry_kt00000001.bin")
    initial = {
        "T": np.asarray(card.recipe.initial_state.T.data)[..., :nlev],
        "S": np.asarray(card.recipe.initial_state.S.data)[..., :nlev],
    }
    initial_rows = {}
    for field in ("T", "S"):
        same = np.asarray(entry[field][..., :nlev]) == initial[field]
        initial_rows[field] = {
            "bit_exact": bool(np.all(same[active])),
            "differing_wet_cells": int(np.count_nonzero(~same & active)),
        }
        require(initial_rows[field]["bit_exact"], f"initial {field} is not bit-exact")

    field_index = {"T": 0, "S": 1}
    differing = []
    for field in ("T", "S"):
        oi = oracle[f"Kaa_{field}"][..., :nlev]
        ci = candidate["after_update"][field_index[field]]
        for j, i, k in np.argwhere((oi != ci) & active):
            differing.append((field, int(j), int(i), int(k)))
    require(len(differing) == (10 if plant else 9), f"expected {'10 planted' if plant else 'nine'} differing cells, got {len(differing)}")

    bottom_k = np.sum(active, axis=-1) - 1
    wet2 = np.any(active, axis=-1)
    cells = []
    boundaries = (
        ("zero_rhs", "zero"),
        ("after_advection", "after_advection"),
        ("after_sbc", "after_sbc"),
        ("after_update", "Kaa"),
    )
    for field, j, i, k in sorted(differing, key=lambda x: (x[1], x[2], x[3], x[0])):
        idx = (j, i, k)
        fi = field_index[field]
        values = {}
        first = None
        for boundary, oracle_prefix in boundaries:
            if boundary == "zero_rhs":
                ov = oracle[f"zero_{field}"][..., :nlev]
                cv = np.zeros_like(ov)
            elif boundary == "after_update":
                ov = oracle[f"Kaa_{field}"][..., :nlev]
                cv = candidate[boundary][fi]
            else:
                ov = oracle[f"{oracle_prefix}_{field}"][..., :nlev]
                cv = candidate[boundary][fi]
            values[boundary] = _cell_value(ov, cv, idx)
            if first is None and not values[boundary]["bit_exact"]:
                first = boundary
        coast = any(
            not wet2[jj, ii]
            for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1))
            if 0 <= jj < wet2.shape[0] and 0 <= ii < wet2.shape[1]
        )
        cells.append({
            "field": field,
            "nemo_ijk_1based": [i + 3, j + 3, k + 1],
            "legoesm_jik_0based": [j, i, k],
            "level": k + 1,
            "surface": k == 0,
            "bottom": k == int(bottom_k[j, i]),
            "coast_adjacent_at_T_level": coast,
            "differs_at_kt1_entry": False,
            "first_differing_boundary": first,
            "values": values,
        })

    aggregate = {}
    for boundary, oracle_prefix in boundaries[1:]:
        aggregate[boundary] = {}
        for field in ("T", "S"):
            fi = field_index[field]
            op = "Kaa" if boundary == "after_update" else oracle_prefix
            ov = oracle[f"{op}_{field}"][..., :nlev]
            cv = candidate[boundary][fi]
            use_o, use_c = ov[active], cv[active]
            aggregate[boundary][field] = {
                "absolute_max": float(np.max(np.abs(use_c - use_o))),
                "ulp_max": int(ulp_distance(use_c, use_o).max(initial=0)),
                "differing_wet_cells": int(np.count_nonzero(use_c != use_o)),
            }

    first_boundaries = sorted({cell["first_differing_boundary"] for cell in cells})
    report = {
        "format": "nemo-testcase-l2-gyre-round13-tracer-v1",
        "status": "AT-BAR" if not differing else "DEBT",
        "regime": "production-jit-cpu-fp64-x64",
        "oracle_root": str(oracle_root),
        "oracle_dump_sha256": sha256(tracer_path),
        "oracle_header": oracle["header"],
        "instrument_bit_identity": bit_identity,
        "initial_entry": initial_rows,
        "differing_cell_count": len(differing),
        "cells": cells,
        "aggregate": aggregate,
        "first_differing_boundaries": first_boundaries,
        "owner_label": "UNMEASURED_SCALING_REQUIRED",
        "source_dispositions": {
            "stage1_transcendentals": "REFUTED_BY_EXECUTED_SOURCE",
            "stage1_fct_limiter": "REFUTED_BY_EXECUTED_SOURCE",
            "tra_qsr_tra_ldf_tra_zdf": "STAGE3_ONLY",
            "tra_atf": "MLF_ONLY_NOT_CALLED_BY_STPRK3_STG",
        },
        "plant": plant,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--control-root", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, args.control_root, plant=args.plant)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

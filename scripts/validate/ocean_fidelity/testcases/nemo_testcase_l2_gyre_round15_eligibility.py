#!/usr/bin/env python3
"""Fail-closed scalar-libm eligibility gate for GYRE oracle V2.

The gate compares the two-band ``tra_qsr`` increment at its stage-3 Kmm
operand and every analytic ``usrdef_sbc`` field.  Both the historical native
XLA and requested scalar-libm precision policies are measured; only the latter
controls eligibility.  All comparisons are binary64 bit comparisons on the
owned 22x32 GYRE domain (and on active 3-D cells for the QSR increment).
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from nemo_testcase_l2_gyre_phase3_gate import read_qsr_stage3, read_tracer_stage3
from legoesm.ocean.fidelity.provenance import worktree_stamp


def _read_sbc(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, kbb, nx, ny, bits = header
    if (magic, version, kt, kbb, nx, ny, bits) != (
        "NEMO_L2_SBC___1", 1, 1, 1, 36, 26, 64,
    ):
        raise AssertionError(f"{path}: bad header {(magic, *header)}")
    interior, full = (nx - 4) * (ny - 4), nx * ny
    if values.size != 2 * interior + 3 * full:
        raise AssertionError(f"{path}: bad payload {values.size}")

    def inner(block: np.ndarray) -> np.ndarray:
        return block.reshape((nx - 4, ny - 4), order="F").T

    def owned(block: np.ndarray) -> np.ndarray:
        return block.reshape((nx, ny), order="F")[2:-2, 2:-2].T

    offset = 0
    qsr = inner(values[offset : offset + interior])
    offset += interior
    qns = inner(values[offset : offset + interior])
    offset += interior
    result = {"qsr": qsr, "qns": qns}
    for name in ("emp", "utau", "vtau"):
        result[name] = owned(values[offset : offset + full])
        offset += full
    return result


def _ordered_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = bits >> np.uint64(63)
    return np.where(sign != 0, ~bits, bits | np.uint64(1 << 63))


def _row(name: str, oracle, candidate, mask) -> dict[str, object]:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if oracle.shape != candidate.shape or oracle.shape != mask.shape:
        raise AssertionError(f"{name}: shape mismatch")
    differing = oracle.view(np.uint64) != candidate.view(np.uint64)
    ulps = np.abs(
        _ordered_bits(oracle).astype(np.int64)
        - _ordered_bits(candidate).astype(np.int64)
    )
    return {
        "name": name,
        "status": "BIT-EXACT" if not bool(np.any(differing & mask)) else "DEBT",
        "differing_cells": int(np.count_nonzero(differing & mask)),
        "total_differing_cells": int(np.count_nonzero(differing)),
        "masked_out_differing_cells": int(np.count_nonzero(differing & ~mask)),
        "max_abs": float(np.max(np.abs(candidate - oracle)[mask], initial=0.0)),
        "max_ulp": int(np.max(ulps[mask], initial=0)),
        "n": int(np.count_nonzero(mask)),
        "dtype": str(candidate.dtype),
    }


def _candidate(policy_name: str, qsr_record: dict, tracer_record: dict) -> dict:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.eos import (
        nemo_potential_temperature_from_conservative,
        nemo_source_round,
    )
    from legoesm.ocean.fidelity.nemo_recipe import nemo_gyre_qns
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_card,
        gyre_surface_boundary_condition,
    )
    from legoesm.ocean.physics.shortwave_penetration import (
        shortwave_penetration_tendency,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals=policy_name))
    jax.clear_caches()
    card = build_gyre_zco_card()
    config = card.recipe.model_config
    z_coord = card.recipe.z_coord

    qsr_increment = jax.jit(
        lambda surface_flux, stretch: shortwave_penetration_tendency(
            surface_flux,
            z_coord.dz_ref,
            z_coord.z_half_ref,
            stretch,
            config.physics.shortwave_penetration,
            rho_0=config.rho_0,
            c_sw=config.physics.constants.c_sw,
            z_half_stretch=stretch,
        )
    )(
        jnp.asarray(qsr_record["qsr"]),
        jnp.asarray(1.0 + tracer_record["r3t_Kmm"]),
    )

    def forcing_fields():
        sbc = gyre_surface_boundary_condition(card, card.dt_s)
        sst = card.recipe.initial_state.T.data[..., 0]
        sst_m = nemo_potential_temperature_from_conservative(
            sst, card.recipe.initial_state.S.data[..., 0])
        qns = nemo_gyre_qns(
            sst,
            sst_m,
            sbc.t_star_c,
            sbc.qsr_w_m2,
            sbc.emp_kg_m2_s,
        )
        return (sbc.qsr_w_m2, qns,
                sbc.emp_kg_m2_s, sbc.utau_pa, sbc.vtau_pa)

    qsr, qns, emp, utau, vtau = jax.jit(forcing_fields)()
    return {
        "qsr_increment": np.asarray(qsr_increment),
        "qsr_after": np.asarray(
            jax.jit(
                lambda before, increment: nemo_source_round(before + increment)
            )(
                jnp.asarray(tracer_record["after_sbc_T"][..., :30]),
                qsr_increment,
            )
        ),
        "qsr": np.asarray(qsr),
        "qns": np.asarray(qns),
        "emp": np.asarray(emp),
        "utau": np.asarray(utau),
        "vtau": np.asarray(vtau),
        "card": card,
    }


def _rule8_sweep(baseline_path: Path, current_path: Path) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    rows = []
    for baseline_step, current_step in zip(
        baseline["steps"], current["steps"], strict=True
    ):
        if baseline_step["kt"] != current_step["kt"]:
            raise AssertionError("V1/V2 sweep steps are not aligned")
        for before, after in zip(
            baseline_step["rows"], current_step["rows"], strict=True
        ):
            field = before["name"].split(".")[-1]
            if field != after["name"].split(".")[-1]:
                raise AssertionError("V1/V2 sweep fields are not aligned")
            old = float(before["absolute_max"])
            new = float(after["absolute_max"])
            direction = "IDENTICAL" if old == new else ("WORSE" if new > old else "BETTER")
            rows.append({
                "kt": baseline_step["kt"], "field": field,
                "v1_absolute_max": old, "v2_absolute_max": new,
                "ratio": None if old == 0.0 else new / old,
                "direction": direction,
            })
    return {
        "baseline": str(baseline_path), "current": str(current_path),
        "counts": {direction: sum(row["direction"] == direction for row in rows)
                   for direction in ("BETTER", "WORSE", "IDENTICAL")},
        "rows": rows,
    }


def _instrumentation_identity(
    baseline_root: Path, instrumented_root: Path,
) -> dict[str, object]:
    import xarray as xr

    histories = []
    for baseline_path in sorted(baseline_root.glob("*.nc")):
        if "restart" not in baseline_path.name and "_gr_" not in baseline_path.name:
            continue
        instrumented_path = instrumented_root / baseline_path.name
        with xr.open_dataset(baseline_path, decode_times=False) as baseline, \
                xr.open_dataset(instrumented_path, decode_times=False) as instrumented:
            differing = [
                name for name in sorted(set(baseline.variables) | set(instrumented.variables))
                if name not in baseline or name not in instrumented
                or baseline[name].shape != instrumented[name].shape
                or not np.array_equal(
                    baseline[name].values, instrumented[name].values,
                    equal_nan=True)
            ]
        histories.append({
            "file": baseline_path.name,
            "data_variables_identical": not differing,
            "differing_variables": differing,
        })
    if not histories or not all(row["data_variables_identical"] for row in histories):
        raise AssertionError("WRITE-only SBC instrument changed a production history field")

    baseline_records = {path.name: path for path in baseline_root.glob("oracle_*.bin")}
    instrumented_records = {
        path.name: path for path in instrumented_root.glob("oracle_*.bin")
        if path.name != "oracle_sbc_kt00000001.bin"
    }
    if baseline_records.keys() != instrumented_records.keys():
        raise AssertionError("ordinary oracle record inventory changed")
    different = [
        name for name in sorted(baseline_records)
        if baseline_records[name].read_bytes() != instrumented_records[name].read_bytes()
    ]
    return {
        "baseline_root": str(baseline_root),
        "instrumented_root": str(instrumented_root),
        "production_history": histories,
        "ordinary_record_raw_bytes": {
            "identical": len(baseline_records) - len(different),
            "total": len(baseline_records),
            "different_uninitialized_or_halo_records": different,
        },
    }


def run(
    oracle_root: Path,
    *,
    plant: bool = False,
    baseline_sweep: Path | None = None,
    current_sweep: Path | None = None,
    uninstrumented_root: Path | None = None,
) -> dict[str, object]:
    qsr_record = read_qsr_stage3(oracle_root / "oracle_qsr_stage3_kt00000001.bin")
    tracer_record = read_tracer_stage3(
        oracle_root / "oracle_rktracer_stage3_kt00000001.bin")
    sbc_record = _read_sbc(oracle_root / "oracle_sbc_kt00000001.bin")
    candidates = {
        name: _candidate(name, qsr_record, tracer_record)
        for name in ("native", "libm")
    }
    wet = np.asarray(candidates["libm"]["card"].recipe.land_mask) > 0.5
    active = (np.asarray(candidates["libm"]["card"].recipe.z_coord.is_active)
              & wet[..., None])
    rows = {}
    for policy_name, candidate in candidates.items():
        rows[policy_name] = [
            _row(
                "qsr_2BD_accumulation",
                tracer_record["after_qsr_T"][..., :30],
                candidate["qsr_after"],
                active,
            ),
            *[
                _row(f"usrdef_sbc.{field}", sbc_record[field], candidate[field],
                     wet)
                for field in ("qsr", "qns", "emp", "utau", "vtau")
            ],
        ]

    control_candidate = np.array(sbc_record["qsr"], copy=True)
    if plant:
        control_candidate[1, 1] = np.nextafter(control_candidate[1, 1], np.inf)
    control = _row("control.oracle_qsr_copy", sbc_record["qsr"],
                   control_candidate, np.ones_like(wet, dtype=bool))
    if (plant and control["status"] != "DEBT") or (
        not plant and control["status"] != "BIT-EXACT"
    ):
        raise AssertionError("eligibility planted control did not discriminate")

    eligible = all(row["status"] == "BIT-EXACT" for row in rows["libm"])
    result = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round16-eligibility-v2",
        "execution_regime": "production-jit/cpu/fp64",
        "oracle_root": str(oracle_root),
        "plant": plant,
        "control": control,
        "rows": rows,
        "status": "PASS" if eligible and not plant else "DEBT",
        "stop_required": not eligible,
    }
    if (baseline_sweep is None) != (current_sweep is None):
        raise ValueError("baseline_sweep and current_sweep must be supplied together")
    if baseline_sweep is not None and current_sweep is not None:
        result["rule8_v1_to_v2_sweep"] = _rule8_sweep(
            baseline_sweep, current_sweep)
    if uninstrumented_root is not None:
        result["instrumentation_identity"] = _instrumentation_identity(
            uninstrumented_root, oracle_root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--baseline-sweep", type=Path)
    parser.add_argument("--current-sweep", type=Path)
    parser.add_argument("--uninstrumented-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.oracle_root,
        plant=args.plant,
        baseline_sweep=args.baseline_sweep,
        current_sweep=args.current_sweep,
        uninstrumented_root=args.uninstrumented_root,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"ROUND15_ELIGIBILITY {result['status']} stop_required={result['stop_required']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

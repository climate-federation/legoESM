#!/usr/bin/env python3
"""Score Round 227's SMT-3 LDF internals through the production JIT step.

This is a thin extension of the Round-218 tracer-stage harness.  It reuses
that harness for the aggregate pre/post-LDF calibration and the shared
self-describing record parser for the new ldfslp/traldf_iso groups.  The
model's existing write-only ``tracer_ldf_diagnostics`` hook exposes the
values produced inside the ordinary production step; no isolated operator
closure is used for a claimed row.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nemo_testcase_l1_vortex"))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _strip2, _strip3, _u_full, _v_full, read_bt_frame,
)
from nemo_testcase_l1_vortex_smt_round218_tracer_walk import (  # noqa: E402
    run as run_stage_walk,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, read_entry, require,
)

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round227/"
    "oracle_vortex_smt3_ldf_internal")
CASE = "VORTEX_SMT3_VEC-zps"


def _checker():
    path = HERE / "nemo_testcase_l1_vortex" / "check_records.py"
    spec = importlib.util.spec_from_file_location("vortex_r228_checker", path)
    require(spec is not None and spec.loader is not None,
            "cannot load the shared self-describing parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _groups(path: Path) -> dict[str, np.ndarray]:
    """Parse and strip a group record using its own declared extents."""
    checker = _checker()
    parsed = checker.parse_record(path)
    values = checker._read_group_values(path)
    out = {}
    for name, value in values.items():
        shape = tuple(parsed["groups"][name]["shape"])
        flat = np.asarray(value).reshape(-1, order="F")
        out[name] = (_strip3(flat, *shape) if len(shape) == 3
                     else _strip2(flat, *shape))
    return out


def _row(name, oracle, candidate, mask, *, plant: str | None):
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape,
            f"{name}: shape mismatch oracle={oracle.shape}, "
            f"candidate={candidate.shape}, mask={use.shape}")
    planted = plant == name
    if planted:
        locations = np.argwhere(use)
        require(locations.size > 0, f"{name}: plant support is empty")
        where = tuple(locations[len(locations) // 2])
        candidate = candidate.copy()
        candidate[where] = np.nextafter(candidate[where], np.inf)
    unequal = (candidate != oracle) & use
    delta = np.abs(candidate - oracle)
    peak = float(np.max(np.abs(oracle[use]))) if np.any(use) else 0.0
    return {
        "name": name,
        "cells_unequal": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(delta[use])) if np.any(use) else 0.0,
        "relative_max_abs": ((float(np.max(delta[use])) / peak)
                             if peak else 0.0),
        "bit_exact": not bool(np.any(unequal)),
        "planted": planted,
        "shape": list(oracle.shape),
        "active_cells": int(np.count_nonzero(use)),
        "dtype": str(candidate.dtype),
        "execution_regime": "production_step_jit",
    }


def run(root: Path, *, plant: str | None = None,
        allow_dirty: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "this walk must run on CPU")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASE,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASE,
                        expect_interior=interior)
    frame = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin",
                          expect_step=1)
    external = (
        jnp.asarray(entry2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )
    seed = _seed_from_record(card.recipe.initial_state, entry1, nlev)
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        tracer_process_trace=(), tracer_ldf_diagnostics="slope")
    trace = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks).step(seed, dt=card.dt_s)
    diagnostics = trace.ldf_diagnostics
    require(isinstance(diagnostics, dict) and "slope" in diagnostics,
            "production tracer step returned no LDF slope diagnostics")
    slope_diag = diagnostics["slope"]

    slope = _groups(root / "oracle_ldf_slope_kt00000001.bin")
    iso = _groups(root / "oracle_ldf_iso_kt00000001.bin")
    for family, groups in (("slope", slope), ("iso", iso)):
        print(f"{family} record groups={len(groups)}")
        for name in sorted(groups):
            print(f"  {name}: shape={groups[name].shape} "
                  f"dtype={groups[name].dtype}")

    rows = []
    def add(name, reference, candidate, support):
        rows.append(_row(name, np.asarray(reference)[..., :nlev],
                         np.asarray(candidate)[..., :nlev], support,
                         plant=plant))

    # Compiled ldfslp order: inputs -> raw face gradients -> bounded vertical
    # gradients -> mixed-layer raw slopes -> the four Shapiro-filtered outputs.
    slope_rows = (
        ("slope.prd", "prd", "prd", "T"),
        ("slope.pn2", "pn2", "pn2", "T"),
        ("slope.e3u", "e3u_kmm", "e3u_live", "u"),
        ("slope.e3v", "e3v_kmm", "e3v_live", "v"),
        ("slope.raw_u", "raw_u", "zau", "u"),
        ("slope.raw_v", "raw_v", "zav", "v"),
        ("slope.bound_u", "bound_u", "zbu_limited", "u"),
        ("slope.bound_v", "bound_v", "zbv_limited", "v"),
        ("slope.hraw_u", "hraw_u", "zwz", "u"),
        ("slope.hraw_v", "hraw_v", "zww", "v"),
        ("slope.uslp", "uslp", "uslp", "u"),
        ("slope.vslp", "vslp", "vslp", "v"),
        ("slope.wslpi", "wslpi", "wslpi", "T"),
        ("slope.wslpj", "wslpj", "wslpj", "T"),
    )
    for name, recorded, live, support in slope_rows:
        add(name, slope[recorded], slope_diag[live], masks[support])

    # traldf_iso's compiled source order after its once-per-call A33 build.
    iso_rows = (
        ("iso.ah_wslp2", "ah_wslp2", "ah_wslp2", "T"),
        ("iso.akz", "akz", "akz", "T"),
        ("iso.dit", "dit", "dit", "u"),
        ("iso.djt", "djt", "djt", "v"),
        ("iso.dkt", "dkt", "dkt", "T"),
        ("iso.A11", "A11", "A11", "u"),
        ("iso.A22", "A22", "A22", "v"),
        ("iso.A13", "A13", "A13", "u"),
        ("iso.A23", "A23", "A23", "v"),
        ("iso.hmsku", "hmsku", "hmsku", "u"),
        ("iso.hmskv", "hmskv", "hmskv", "v"),
        ("iso.fu", "fu", "zfu", "u"),
        ("iso.fv", "fv", "zfv", "v"),
        ("iso.vmsku", "vmsku", "vmsku", "T"),
        ("iso.vmskv", "vmskv", "vmskv", "T"),
        ("iso.ahu_w", "ahu_w", "ahu_w", "T"),
        ("iso.ahv_w", "ahv_w", "ahv_w", "T"),
        ("iso.A31", "A31", "A31", "T"),
        ("iso.A32", "A32", "A32", "T"),
        ("iso.fw_lower", "fw_lower", "zfw_top", "T"),
        ("iso.fw_upper", "fw_upper", "zfw_kp1", "T"),
        ("iso.rhs_increment", "rhs_increment", "tendency", "T"),
    )
    for name, recorded, live, support in iso_rows:
        add(name, iso[recorded], diagnostics[live], masks[support])

    aggregate = run_stage_walk(
        root, "smt3", allow_dirty=allow_dirty, stage=3)
    aggregate_rows = {
        row["name"].split(".", 1)[1]: row for row in aggregate["rows"]}
    pre = aggregate_rows["stage3.pre_ldf.T"]
    post = aggregate_rows["stage3.post_ldf.T"]
    first = next((row for row in rows if not row["bit_exact"]), None)
    report = {
        "case": CASE, "legoesm_git_sha": sha, "oracle_root": str(root),
        "plant": plant, "rows": rows,
        "first_non_bit": None if first is None else first["name"],
        "aggregate_reproduction": {
            "pre_ldf_max_abs": pre["max_abs"],
            "post_ldf_max_abs": post["max_abs"],
            "additional_ldf_max_abs": post["max_abs"] - pre["max_abs"],
        },
        "status": "PLANT" if plant else "MEASURED",
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant")
    parser.add_argument("--clean-report", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_dir, plant=args.plant,
                     allow_dirty=args.allow_dirty)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    for row in report["rows"]:
        print(f"{row['name']:22s} bit={str(row['bit_exact']):5s} "
              f"cells={row['cells_unequal']:7d} "
              f"max_abs={row['max_abs']:.16e}")
    print(json.dumps(report["aggregate_reproduction"], sort_keys=True))
    print("first non-bit:", report["first_non_bit"])
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                               + "\n")
    if args.plant:
        require(args.clean_report is not None,
                "a plant requires --clean-report")
        clean = json.loads(args.clean_report.read_text())
        before = {row["name"]: row for row in clean["rows"]}
        planted = [row for row in report["rows"] if row["planted"]]
        require(len(planted) == 1, "plant did not select exactly one row")
        row = planted[0]
        visible = (row["name"] in before and
                   (row["cells_unequal"], row["max_abs"])
                   != (before[row["name"]]["cells_unequal"],
                       before[row["name"]]["max_abs"]))
        print(f"STATUS {'PLANT-FIRED' if visible else 'PLANT-MISSED'}")
        return 1 if visible else 0
    print("STATUS MEASURED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

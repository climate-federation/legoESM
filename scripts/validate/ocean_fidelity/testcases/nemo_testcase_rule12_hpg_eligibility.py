#!/usr/bin/env python3
"""Rule-12 eligibility row for the changed HPG/QCO operator on the lane-1 tanks.

Round 25 landed two source associations inside NEMO's hydrostatic pressure
gradient: the ``key_qco`` depth operand ``gdept_z0 = gdept_0*(1+r3t) - ssh``
materialised as two statements (``domzgr_substitute.h90:139,145``), and the
``nemo_sco`` acceleration carried straight into the momentum RHS instead of
through a pressure round trip (``dynhpg.F90:359,383``).  That change was shown
bit-exact given NEMO's own inputs on GYRE only.

Rule 12 requires the same exact-input row on EVERY card that executes the
changed operator, and ``nemo_testcase_recipe.py:92,274,914`` pins
``pgf_scheme='nemo_sco'`` for all of them.  This gate supplies the row for
``LOCK_EXCHANGE-zco`` and ``OVERFLOW-zps``.

Why the dumped stage-1 RHS *is* the HPG frame on these cards: NEMO's RK3
stage accumulates ``dyn_hpg``, ``dyn_vor`` and ``dyn_adv`` into a zeroed
``puu(:,:,:,Krhs)`` (``stprk3_stg.F90:309-334``).  ``dyn_vor`` and ``dyn_adv``
are bilinear in the velocity, and both lane-1 tanks start from rest, so at
``u = v = 0`` those two contribute exactly zero and the dumped
``oracle_rhs_kt00000001.bin`` frame is the hydrostatic pressure gradient
alone.  Both preconditions are asserted here, so the row cannot pass by
cancellation, and the two changed functions are counted as they execute so
the row cannot pass on a card that never runs them.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from nemo_testcase_phase3_stage_sweep_gate import (
    BAR,
    DIMS,
    ROOTS,
    GateError,
    _xy,
    _xyz,
    expected_masks,
    git_sha,
    require,
    score,
    sha256,
)


def read_rhs(path: Path, case: str) -> dict:
    """Read the stage-1 momentum RHS dump (``NEMO_L1_RHS___1``)."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, level, nx, ny, nz, bits = header
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic")
    require((version, nx, ny, nz, bits) == (1, *DIMS[case], 64),
            f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 2 * count, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    require((kt, level) == (1, 3), f"{path}: expected kt=1/Nrhs=3, got {kt}/{level}")
    return {"u": _xyz(values[:count], nx, ny, nz),
            "v": _xyz(values[count:], nx, ny, nz)}


def read_entry_full(path: Path, case: str) -> dict:
    """Read the kt=1 step-entry state INCLUDING v (the sweep reader drops it)."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nbb, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require((version, nx, ny, nz, ntr, bits) == (1, *DIMS[case], 2, 64),
            f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    require((kt, nbb) == (1, 1), f"{path}: expected kt=1/Nbb=1, got {kt}/{nbb}")
    return {
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count:2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count:3 * count], nx, ny, nz),
        "v": _xyz(values[3 * count:4 * count], nx, ny, nz),
        "ssh": _xy(values[4 * count:], nx, ny),
    }


def run(case: str, root: Path, *, plant: bool = False,
        allow_dirty: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe_module
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    legoesm_git_sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    card = build_nemo_testcase_card(case)
    cfg = card.recipe.model_config
    require(cfg.pgf_scheme == "nemo_sco",
            f"{case} does not execute the changed operator: "
            f"pgf_scheme={cfg.pgf_scheme!r}")
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    initial = card.recipe.initial_state

    rhs_path = root / "oracle_rhs_kt00000001.bin"
    entry_path = root / "oracle_step_entry_kt00000001.bin"
    require(rhs_path.is_file(), f"missing {rhs_path}")
    require(entry_path.is_file(), f"missing {entry_path}")
    rhs = read_rhs(rhs_path, case)
    entry = read_entry_full(entry_path, case)

    # Precondition A: legoESM consumes NEMO's own inputs, bit for bit.
    inputs = []
    for name, oracle_field, candidate in (
        ("T", entry["T"][..., :nlev], np.asarray(initial.T.data)),
        ("S", entry["S"][..., :nlev], np.asarray(initial.S.data)),
        ("u", entry["u"][..., :nlev],
         np.asarray(initial.u.data)[:, 1:, :]),
        ("ssh", entry["ssh"], np.asarray(initial.eta.data)),
    ):
        mask = masks["T"] if name in ("T", "S") else (
            masks["u"] if name == "u" else np.any(masks["T"], axis=-1))
        unequal = int(np.count_nonzero(
            candidate[mask] != oracle_field[mask]))
        inputs.append({
            "field": name, "unequal": unequal, "n": int(mask.sum()),
            "absolute_max": float(np.max(np.abs(
                candidate[mask] - oracle_field[mask]))),
        })
    require(all(row["unequal"] == 0 for row in inputs),
            f"{case}: legoESM does not consume NEMO's own kt=1 inputs: {inputs}")

    # Precondition B: NEMO's own entry velocity is IDENTICALLY zero, so the
    # dumped stage-1 Krhs carries no dyn_vor or dyn_adv contribution and the
    # row cannot pass by cancellation against a nonzero one.
    rest = {
        "oracle_entry_abs_max_u": float(np.max(np.abs(entry["u"]))),
        "oracle_entry_abs_max_v": float(np.max(np.abs(entry["v"]))),
    }
    require(rest["oracle_entry_abs_max_u"] == 0.0
            and rest["oracle_entry_abs_max_v"] == 0.0,
            f"{case}: NEMO's kt=1 entry is not at rest: {rest}; the dumped "
            "stage-1 RHS is then not the HPG frame and this row is void")

    # Precondition C: the two functions round 25 changed actually execute on
    # this card.  A row measured on a card that never calls them proves
    # nothing about the change's eligibility.
    calls = {"_nemo_qco_gdept_z0": 0,
             "_nemo_hpg_tendency_from_pressure_or_direct": 0}
    originals = {name: getattr(pe_module, name) for name in calls}

    def _counted(name):
        original = originals[name]

        def wrapper(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapper

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg)
    try:
        for name in calls:
            setattr(pe_module, name, _counted(name))
        tendency = model.tendencies(initial, dt=card.dt_s, momentum_only=True)
        candidate_u = np.asarray(tendency.du_dt.data)[:, 1:, :]
        candidate_v = np.asarray(tendency.dv_dt.data)[1:, :, :]
    finally:
        for name, original in originals.items():
            setattr(pe_module, name, original)
    require(all(count > 0 for count in calls.values()),
            f"{case}: the changed operator did not execute: {calls}")

    def _relabel(row: dict) -> dict:
        # score() stamps the Kaa-velocity frame of the trajectory gates.  This
        # row is a momentum TENDENCY frame, so the inherited label would be a
        # false claim about what is being compared.
        row["frame"] = "instantaneous_stage1_momentum_rhs"
        row["staggering_and_reduction"] = (
            "NEMO puu(:,:,:,Nrhs) after the stage-1 accumulation and "
            "legoESM's momentum tendency are both instantaneous 3-D C-grid "
            "face accelerations in m/s^2 at the same wet faces; elementwise "
            "L-infinity, no vertical, substep or time reduction. At rest the "
            "frame is dyn_hpg alone")
        return row

    rows = [_relabel(score(f"{case}.kt1.stage1.oracle_input_hpg.u",
                           rhs["u"][..., :nlev], candidate_u, masks["u"],
                           plant=plant))]

    # The V component: both tanks are single-wet-row channels, so the wet
    # V-face set is empty and a V row would compare masked zeros with masked
    # zeros.  WAIVED with the measured count rather than silently skipped.
    active = masks["T"]
    v_mask = active & np.roll(active, -1, axis=0)
    v_mask[-1] = False
    v_wet = int(v_mask.sum())
    if v_wet:
        rows.append(_relabel(score(f"{case}.kt1.stage1.oracle_input_hpg.v",
                                   rhs["v"][..., :nlev], candidate_v, v_mask)))
        v_disposition = "SCORED"
    else:
        v_disposition = (
            "WAIVED_STRUCTURALLY_ABSENT: this card has one wet j row, so it "
            "has no wet V face; a V row would compare masked zeros")

    status = "AT-BAR" if all(row["status"] == "AT-BAR" for row in rows) else "DEBT"
    exact = all(row["exact"] for row in rows)
    report = {
        "format": "nemo-testcase-rule12-hpg-eligibility-v1",
        "case": case,
        "status": status,
        "bit_exact_given_nemo_inputs": bool(exact and not plant),
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "transcendentals": "libm",
        "jax_backend": jax.default_backend(),
        "legoesm_git_sha": legoesm_git_sha,
        "pgf_scheme": cfg.pgf_scheme,
        "oracle_root": str(root),
        "artifacts": {rhs_path.name: sha256(rhs_path),
                      entry_path.name: sha256(entry_path)},
        "precondition_equal_inputs": inputs,
        "precondition_oracle_at_rest": rest,
        "precondition_changed_operator_executed": calls,
        "v_component": v_disposition,
        "v_wet_faces": v_wet,
        "nemo_source": [
            "stprk3_stg.F90:309-334 (dyn_hpg, dyn_vor, dyn_adv into Krhs)",
            "dynhpg.F90:359,383 (nemo_sco acceleration into the RHS)",
            "domzgr_substitute.h90:139,145 (key_qco gdept_z0)",
        ],
        "rows": rows,
        "planted_control": plant,
    }
    if plant:
        require(status == "DEBT" and rows[0]["absolute_max"] >= 0.5,
                "planted eligibility violation did not fire")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(ROOTS))
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stamp a dirty tree (never for a recorded row)")
    args = parser.parse_args(argv)
    root = args.oracle_root or ROOTS[args.case]
    try:
        report = run(args.case, root, plant=args.plant,
                     allow_dirty=args.allow_dirty)
    except GateError as error:
        print(json.dumps({"status": "GATE-ERROR", "error": str(error)},
                         indent=2))
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

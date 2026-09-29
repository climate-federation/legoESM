#!/usr/bin/env python3
"""Attribute the VORTEX card's kt=2 debt to a piece of NEMO's first step.

Round 2 scored the ladder and named nobody.  This walks it, in NEMO's own
execution order, by substituting ONE boundary at a time from NEMO's own record
of the inside of that step -- the barotropic frame written immediately after
the external solve, and the per-stage states written at each stage's pointer
boundary -- and scoring the kt=2 entry with the SAME normalized maximum the
ladder gate reports, so the arms and the ladder are directly comparable.

Every arm's residual is owned by whatever has NOT been substituted:

    0  nothing                          everything, the initial state included
    1  the kt=1 entry state             the whole first step
    2  + the external-solve handoff     the three stages
    3  + NEMO's stage-1 output          stages 2 and 3
    4  + NEMO's stage-2 output          stage 3 alone

Note BD: no record size or header tuple is predicted here.  Every record is
parsed from its own header and its payload is checked against the dimensions
that record declares.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    BAR, GateError, expected_masks, lego_fields, read_entry, require, score,
)

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round2")
_HALO = 2
CASE = "VORTEX-zco"


def _strip3(values, nx, ny, nz):
    return values.reshape((nx, ny, nz), order="F")[
        _HALO:-_HALO, _HALO:-_HALO].transpose(1, 0, 2)


def _strip2(values, nx, ny):
    return values.reshape((nx, ny), order="F")[_HALO:-_HALO, _HALO:-_HALO].T


def read_stage(path: Path, *, expect_step: int, expect_stage: int) -> dict:
    """One ``NEMO_L1_STAGE_1`` record, parsed from its own header."""
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        (version, step, stage, level, nx, ny, nz, ntr, bits) = struct.unpack(
            "=9i", fh.read(36))
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic {magic!r}")
    require((version, bits) == (1, 64), f"{path}: unsupported record format")
    require(step == expect_step and stage == expect_stage,
            f"{path}: header says step {step} stage {stage}, the walk asked "
            f"for step {expect_step} stage {expect_stage}")
    require(min(nx, ny, nz, ntr) > 0, f"{path}: nonpositive extent")
    cell = nx * ny * nz
    require(data.size == ntr * cell + 2 * cell + nx * ny,
            f"{path}: payload holds {data.size} doubles, but its own header "
            f"({nx}x{ny}x{nz}, {ntr} tracers) asks for "
            f"{ntr * cell + 2 * cell + nx * ny}")
    return {
        "stage": stage, "Kaa": level, "nz": nz,
        "T": _strip3(data[:cell], nx, ny, nz),
        "S": _strip3(data[cell:2 * cell], nx, ny, nz),
        "u": _strip3(data[ntr * cell:(ntr + 1) * cell], nx, ny, nz),
        "v": _strip3(data[(ntr + 1) * cell:(ntr + 2) * cell], nx, ny, nz),
        "ssh": _strip2(data[(ntr + 2) * cell:], nx, ny),
    }


def read_bt_frame(path: Path, *, expect_step: int) -> dict:
    """One ``NEMO_L1_BTFRM_1`` record, parsed from its own header."""
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, step, level, nx, ny, bits = struct.unpack("=6i", fh.read(24))
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == "NEMO_L1_BTFRM_1", f"{path}: bad magic {magic!r}")
    require((version, bits) == (1, 64), f"{path}: unsupported record format")
    require(step == expect_step, f"{path}: header says step {step}")
    require(min(nx, ny) > 0, f"{path}: nonpositive extent")
    require(data.size == 4 * nx * ny,
            f"{path}: payload holds {data.size} doubles, but its own header "
            f"({nx}x{ny}, four fields) asks for {4 * nx * ny}")
    names = ("uu_b", "vv_b", "un_adv", "vn_adv")
    plane = nx * ny
    out = {"Kaa": level}
    for index, name in enumerate(names):
        out[name] = _strip2(data[index * plane:(index + 1) * plane], nx, ny)
    return out


def _u_full(values):
    """Prepend the model's one redundant west face record."""
    values = np.asarray(values, dtype=np.float64)
    pad = np.zeros(values.shape[:1] + (1,) + values.shape[2:], dtype=np.float64)
    return np.concatenate([pad, values], axis=1)


def _v_full(values):
    values = np.asarray(values, dtype=np.float64)
    pad = np.zeros((1,) + values.shape[1:], dtype=np.float64)
    return np.concatenate([pad, values], axis=0)


def _seed_from_record(state, entry, nlev):
    import jax.numpy as jnp
    return state._replace(
        T=state.T.replace(data=jnp.asarray(entry["T"][..., :nlev])),
        S=state.S.replace(data=jnp.asarray(entry["S"][..., :nlev])),
        u=state.u.replace(data=jnp.asarray(_u_full(entry["u"][..., :nlev]))),
        v=state.v.replace(data=jnp.asarray(_v_full(entry["v"][..., :nlev]))),
        eta=state.eta.replace(data=jnp.asarray(entry["ssh"])),
    )


def run(root: Path, *, plant: bool = False, allow_dirty: bool = False) -> dict:
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
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")

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
    stages = {
        n: read_stage(root / f"oracle_stage_kt00000001_s{n}.bin",
                      expect_step=1, expect_stage=n)
        for n in (1, 2, 3)
    }
    for name, record in (("entry kt=1", entry1), ("entry kt=2", entry2),
                         *((f"stage {n}", stages[n]) for n in (1, 2, 3))):
        require(record["nz"] == nlev + card.dummy_bottom_records,
                f"{name}: {record['nz']} levels, the card executes {nlev} "
                f"plus {card.dummy_bottom_records} dummy bottom record(s)")

    # The external-solve handoff NEMO produced: the end-of-step height, which
    # this time-stepping program writes once in the external solve and the
    # stages never touch, and the barotropic velocity and time-mean transport
    # written immediately after that solve.
    external = (
        jnp.asarray(entry2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )

    def stage_entry(stage_number: int, source: dict):
        return (stage_number,
                jnp.asarray(_u_full(source["u"][..., :nlev])),
                jnp.asarray(_v_full(source["v"][..., :nlev])),
                jnp.asarray(source["T"][..., :nlev]),
                jnp.asarray(source["S"][..., :nlev]),
                jnp.asarray(source["ssh"]))

    arms = [
        ("0_card", False, None, None,
         "nothing substituted; the card as it ships"),
        ("1_nemo_entry", True, None, None,
         "the step-entry state is NEMO's kt=1 record"),
        ("2_nemo_external", True, external, None,
         "arm 1 plus NEMO's external-solve handoff"),
        ("3_nemo_stage1_out", True, external, stage_entry(2, stages[1]),
         "arm 2 plus NEMO's stage-1 output as stage 2's entry"),
        ("4_nemo_stage2_out", True, external, stage_entry(3, stages[2]),
         "arm 2 plus NEMO's stage-2 output as stage 3's entry"),
    ]

    report = {
        "case": CASE, "oracle_root": str(root), "bar": BAR,
        "legoesm_git_sha": sha, "plant": plant, "arms": [],
    }
    for name, seeded, baro, entry_override, description in arms:
        hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=baro,
            stage_entry_override=entry_override)
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        state = card.recipe.initial_state
        if seeded:
            state = _seed_from_record(state, entry1, nlev)
        after = model.step(state, dt=card.dt_s)
        candidate = lego_fields(after)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(entry2[field])
            if field != "ssh":
                reference = reference[..., :nlev]
            rows.append(score(
                f"{CASE}.walk.{name}.kt2.{field}", reference,
                candidate[field], masks[field],
                plant=plant and name == "0_card" and field == "u"))
        # Structure of the velocity residual, which is what discriminates a
        # depth-uniform owner (the barotropic correction adds one number per
        # column) from a vertical-difference owner (the implicit viscosity)
        # from a surface-weighted one (the free-surface stretching).
        umask3 = np.asarray(masks["u"], dtype=bool)
        residual = candidate["u"] - np.asarray(entry2["u"][..., :nlev])
        column_ok = umask3.all(axis=-1)
        structure = {}
        if column_ok.any():
            columns = residual[column_ok]
            spread = np.max(np.abs(columns - columns.mean(axis=-1,
                                                          keepdims=True)))
            structure = {
                "n_full_columns": int(column_ok.sum()),
                "max_abs_residual": float(np.max(np.abs(residual[umask3]))),
                "max_abs_column_mean": float(
                    np.max(np.abs(columns.mean(axis=-1)))),
                "max_abs_deviation_from_column_mean": float(spread),
                "per_level_max_abs": [
                    float(np.max(np.abs(residual[..., k][umask3[..., k]])))
                    if umask3[..., k].any() else 0.0
                    for k in range(nlev)],
            }
        report["arms"].append({
            "arm": name, "describes": description,
            "velocity_residual_structure": structure,
            "rows": rows,
            "kt2": {row["name"].rsplit(".", 1)[-1]: row["normalized_max_abs"]
                    for row in rows},
            "over_bar": [row["name"].rsplit(".", 1)[-1] for row in rows
                         if row["status"] == "DEBT"],
        })

    def velocity(arm):
        return max(arm["kt2"]["u"], arm["kt2"]["v"])

    by_name = {arm["arm"]: arm for arm in report["arms"]}
    report["attribution"] = {
        "initial_state": velocity(by_name["0_card"])
        - velocity(by_name["1_nemo_entry"]),
        "external_solve": velocity(by_name["1_nemo_entry"])
        - velocity(by_name["2_nemo_external"]),
        "stage_1": velocity(by_name["2_nemo_external"])
        - velocity(by_name["3_nemo_stage1_out"]),
        "stage_2": velocity(by_name["3_nemo_stage1_out"])
        - velocity(by_name["4_nemo_stage2_out"]),
        "stage_3_residual": velocity(by_name["4_nemo_stage2_out"]),
        "units": "normalized max abs velocity error at the kt=2 entry",
    }
    owner = max(
        (key for key in ("initial_state", "external_solve", "stage_1",
                         "stage_2", "stage_3_residual")),
        key=lambda key: report["attribution"][key])
    report["owner"] = owner
    report["status"] = "MEASURED"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true",
                        help="perturb one arm-0 velocity cell; MUST exit "
                             "non-zero, or this walk proves nothing")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_dir, plant=args.plant,
                     allow_dirty=args.allow_dirty)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    for arm in report["arms"]:
        print("{arm:22s} u={u:.6e} v={v:.6e} T={T:.6e} ssh={ssh:.6e}".format(
            arm=arm["arm"], **arm["kt2"]))
    print("attribution:", json.dumps(report["attribution"], sort_keys=True))
    print("owner:", report["owner"])
    if args.plant:
        planted = report["arms"][0]["kt2"]["u"]
        print(f"PLANT arm-0 velocity {planted:.6e}")
        return 1 if planted > 1.0e-3 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())

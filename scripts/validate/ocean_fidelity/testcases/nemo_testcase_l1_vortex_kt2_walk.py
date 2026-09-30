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

Round 4 adds three things the round-3 reviewers required, and one boundary
the same record already supports:

  * ``--case`` runs the SAME walk on either VORTEX card; the vector-EEN card
    leaves the bar at kt=2 by 126x the flux card's velocity error;
  * a plant per SUBSTITUTED ARM.  Round 3's plant perturbed arm 0's scoring
    only, so the walk passed even if both substitution hooks were inert.
    Each plant below perturbs the value actually handed to the model, so a
    dead hook makes the planted run identical to the clean one and the plant
    reports NOT VISIBLE and exits zero, which the gate refuses;
  * the STAGE-LOCAL boundary.  Each stage is run from NEMO's own recorded
    entry and its OUTPUT is scored against NEMO's record for that stage, so
    a stage that re-makes the same error every step is visible directly
    rather than through a difference of two kt=2 rows;
  * a VERDICT.  The walk first reproduces the ladder's own kt=2 velocity row
    from arm 0 (the instrument against a value already known) and refuses if
    it cannot, then refuses again if no boundary owns the residual.

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
    BAR, DEFAULT_ORACLE_ROOTS, GateError, expected_masks, lego_fields,
    read_entry, require, score,
)

_HALO = 2
# The kt=2 velocity row each card's own ladder reports, which arm 0 has to
# reproduce before any arm of this walk is believed.  Flux card: round-2
# receipt.  Vector card: round-3 receipt, section 4.
_LADDER_KT2_VELOCITY = {
    "VORTEX-zco": 1.136e-07,
    "VORTEX_VEC-zco": 1.432e-05,
}
_PLANTS = ("score", "entry", "external", "stage1", "stage2",
           "prestage_rhs")


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


def read_rhs(path: Path, *, expect_step: int) -> dict:
    """One ``NEMO_L1_RHS___1`` record, parsed from its own header.

    This is the COMPLETED three-dimensional momentum right-hand side NEMO
    holds immediately after ``stp_2D`` and before its depth reduction
    (stp2d.f90:176-180).  Under vector-invariant form it is the WHOLE of
    stage 1's momentum tendency -- NEMO does not recompute one
    (stprk3_stg.f90:310-316: "Vector Inv. Form : 1st stage 3D RHS already
    entirely computed in stp_2D").
    """
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, step, level, nx, ny, nz, bits = struct.unpack(
            "=7i", fh.read(28))
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == "NEMO_L1_RHS___1", f"{path}: bad magic {magic!r}")
    require((version, bits) == (1, 64), f"{path}: unsupported record format")
    require(step == expect_step, f"{path}: header says step {step}")
    require(min(nx, ny, nz) > 0, f"{path}: nonpositive extent")
    cell = nx * ny * nz
    require(data.size == 2 * cell,
            f"{path}: payload holds {data.size} doubles, but its own header "
            f"({nx}x{ny}x{nz}, two faces) asks for {2 * cell}")
    return {"nz": nz, "Krhs": level,
            "u": _strip3(data[:cell], nx, ny, nz),
            "v": _strip3(data[cell:], nx, ny, nz)}


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


def _bump(values, size=1.0e-3):
    """Perturb ONE substituted cell by an amount no rounding can imitate."""
    import jax.numpy as jnp
    values = jnp.asarray(values)
    return values.at[tuple(dim // 2 for dim in values.shape)].add(size)


def run(root: Path, *, case: str = "VORTEX-zco", plant: str | None = None,
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
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")

    require(plant in (None,) + _PLANTS,
            f"unknown plant {plant!r}; expected one of {_PLANTS}")
    require(case in _LADDER_KT2_VELOCITY,
            f"unknown case {case!r}; this walk knows "
            f"{sorted(_LADDER_KT2_VELOCITY)}")
    CASE = case
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
    rhs = read_rhs(root / "oracle_rhs_kt00000001.bin", expect_step=1)
    stages = {
        n: read_stage(root / f"oracle_stage_kt00000001_s{n}.bin",
                      expect_step=1, expect_stage=n)
        for n in (1, 2, 3)
    }
    for name, record in (("entry kt=1", entry1), ("entry kt=2", entry2),
                         ("pre-stage rhs", rhs),
                         *((f"stage {n}", stages[n]) for n in (1, 2, 3))):
        require(record["nz"] == nlev + card.dummy_bottom_records,
                f"{name}: {record['nz']} levels, the card executes {nlev} "
                f"plus {card.dummy_bottom_records} dummy bottom record(s)")

    # The external-solve handoff NEMO produced: the end-of-step height, which
    # this time-stepping program writes once in the external solve and the
    # stages never touch, and the barotropic velocity and time-mean transport
    # written immediately after that solve.
    planted_entry = entry1
    if plant == "entry":
        planted_entry = dict(entry1)
        planted_entry["u"] = np.asarray(entry1["u"]).copy()
        planted_entry["u"][tuple(d // 2 for d in planted_entry["u"].shape)] \
            += 1.0e-3

    _ssh = jnp.asarray(entry2["ssh"])
    _uu_b = jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0])
    if plant == "external":
        _uu_b = _bump(_uu_b)
    external = (
        _ssh, _uu_b,
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )

    def stage_entry(stage_number: int, source: dict):
        u_in = jnp.asarray(_u_full(source["u"][..., :nlev]))
        if plant == f"stage{stage_number - 1}":
            u_in = _bump(u_in)
        return (stage_number, u_in,
                jnp.asarray(_v_full(source["v"][..., :nlev])),
                jnp.asarray(source["T"][..., :nlev]),
                jnp.asarray(source["S"][..., :nlev]),
                jnp.asarray(source["ssh"]))

    _rhs_u = jnp.asarray(rhs["u"][..., :nlev])
    _rhs_v = jnp.asarray(rhs["v"][..., :nlev])
    if plant == "prestage_rhs":
        _rhs_u = _bump(_rhs_u, 1.0)
    prestage_rhs = (_rhs_u, _rhs_v)

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
        ("5_nemo_prestage_rhs", True, external, None,
         "arm 2 plus NEMO's completed pre-stage 3-D momentum RHS"),
        ("6_prestage_rhs_only", True, None, None,
         "arm 1 plus NEMO's pre-stage RHS, WITHOUT the external handoff: "
         "the external solve is forced by the depth average of that same "
         "right-hand side (stp2d.F90:176-180), so this arm says how much "
         "of the handoff's share the right-hand side already owns"),
    ]

    report = {
        "case": CASE, "oracle_root": str(root), "bar": BAR,
        "legoesm_git_sha": sha, "plant": plant, "arms": [],
        "stage_local": [],
    }

    def _model(hooks):
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)

    def _seeded_state(seeded):
        state = card.recipe.initial_state
        return _seed_from_record(state, planted_entry, nlev) if seeded \
            else state

    for name, seeded, baro, entry_override, description in arms:
        hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=baro,
            stage_entry_override=entry_override,
            slow_forcing_rhs_override=(
                prestage_rhs
                if name in ("5_nemo_prestage_rhs", "6_prestage_rhs_only")
                else None))
        after = _model(hooks).step(_seeded_state(seeded), dt=card.dt_s)
        candidate = lego_fields(after)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(entry2[field])
            if field != "ssh":
                reference = reference[..., :nlev]
            rows.append(score(
                f"{CASE}.walk.{name}.kt2.{field}", reference,
                candidate[field], masks[field],
                plant=plant == "score" and name == "0_card"
                and field == "u"))
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

    # ---- the stage-local boundary -------------------------------------
    # Every stage is handed NEMO's OWN entry for everything it consumes and
    # its output is scored against NEMO's record of that stage's output.  So
    # each row below is one stage's own contribution, with no cancellation
    # against the two it does not run.  Stage 3's output is the ordinary
    # returned state; stages 1 and 2 are exposed WRITE-only after the
    # compiled step has finished (ocean_model_latlon_cgrid.py:9285-9305).
    stage_sources = {1: None, 2: stage_entry(2, stages[1]),
                     3: stage_entry(3, stages[2])}
    for stage, with_rhs in ((1, False), (2, False), (3, False), (1, True)):
        hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            stage_entry_override=stage_sources[stage],
            expose_momentum_stage=0 if stage == 3 else stage,
            expose_tracer_stage=0 if stage == 3 else stage,
            slow_forcing_rhs_override=prestage_rhs if with_rhs else None)
        after = _model(hooks).step(_seeded_state(True), dt=card.dt_s)
        candidate = lego_fields(after)
        rows = []
        for field in ("T", "S", "u", "v", "ssh"):
            reference = np.asarray(stages[stage][field])
            if field != "ssh":
                reference = reference[..., :nlev]
            rows.append(score(
                f"{CASE}.walk.stage{stage}"
                + (".nemorhs" if with_rhs else "") + f".out.{field}",
                reference, candidate[field], masks[field]))
        report["stage_local"].append({
            "stage": stage,
            "nemo_prestage_rhs": with_rhs,
            "describes": (
                f"stage {stage} alone, from NEMO's own entry"
                + (" AND NEMO's pre-stage momentum RHS" if with_rhs else "")
                + f", scored against NEMO's stage-{stage} output record"),
            "rows": rows,
            "out": {row["name"].rsplit(".", 1)[-1]:
                    row["normalized_max_abs"] for row in rows},
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
        "prestage_rhs_residual": velocity(by_name["5_nemo_prestage_rhs"]),
        "prestage_rhs_only_residual": velocity(
            by_name["6_prestage_rhs_only"]),
        "units": "normalized max abs velocity error at the kt=2 entry",
    }
    report["stage_local_velocity"] = {
        f"stage_{row['stage']}" + ("_nemorhs" if row["nemo_prestage_rhs"]
                                   else ""):
        max(row["out"]["u"], row["out"]["v"])
        for row in report["stage_local"]
    }
    owner = max(
        (key for key in ("initial_state", "external_solve", "stage_1",
                         "stage_2", "stage_3_residual")),
        key=lambda key: report["attribution"][key])
    report["owner"] = owner

    # The verdict.  First the instrument against a number already known: arm
    # 0 is the card as it ships, so it must reproduce the ladder's own kt=2
    # velocity row.  Then the attribution itself.
    arm0 = velocity(by_name["0_card"])
    known = _LADDER_KT2_VELOCITY[CASE]
    report["arm0_vs_ladder"] = {
        "arm0_velocity": arm0, "ladder_kt2_velocity": known,
        "relative_difference": abs(arm0 - known) / known,
    }
    if plant is None and report["arm0_vs_ladder"]["relative_difference"] > 0.01:
        report["status"] = "INSTRUMENT_DISAGREES_WITH_THE_LADDER"
    elif max(report["stage_local_velocity"].values()) <= BAR:
        report["status"] = "UNATTRIBUTED"
    else:
        report["status"] = "MEASURED"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="VORTEX-zco",
                        choices=sorted(_LADDER_KT2_VELOCITY))
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=_PLANTS,
                        help="perturb one SUBSTITUTED value; the run MUST "
                             "exit non-zero, or that arm's hook is inert "
                             "and the walk proves nothing")
    parser.add_argument("--clean-report", type=Path,
                        help="a previously written unplanted report for the "
                             "SAME case, so a plant run need not repeat it")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    root = args.oracle_dir or DEFAULT_ORACLE_ROOTS[args.case]
    try:
        report = run(root, case=args.case, plant=args.plant,
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
    for row in report["stage_local"]:
        print("stage{stage}{tag:9s}   u={u:.6e} v={v:.6e} T={T:.6e} "
              "ssh={ssh:.6e}".format(
                  stage=row["stage"],
                  tag="_nemorhs" if row["nemo_prestage_rhs"] else "_out",
                  **row["out"]))
    print("attribution:", json.dumps(report["attribution"], sort_keys=True))
    print("stage-local:", json.dumps(report["stage_local_velocity"],
                                     sort_keys=True))
    print("owner:", report["owner"])
    print("arm0 vs ladder:", json.dumps(report["arm0_vs_ladder"],
                                        sort_keys=True))
    print("status:", report["status"])
    if args.plant:
        if args.clean_report:
            clean = json.loads(args.clean_report.read_text())
            if clean.get("case") != args.case or clean.get("plant"):
                print("REFUSE: --clean-report is not an unplanted report for "
                      f"{args.case}", file=sys.stderr)
                return 2
        else:
            clean = run(root, case=args.case, plant=None,
                        allow_dirty=args.allow_dirty)
        moved = plant_moved(args.plant, clean, report)
        print(f"PLANT {args.plant} "
              f"{'VISIBLE' if moved else 'NOT VISIBLE'}")
        return 1 if moved else 0
    return 0 if report["status"] == "MEASURED" else 3


def plant_moved(plant: str, clean: dict, planted: dict) -> bool:
    """Did the perturbation reach the arm whose substitution carries it?

    A plant that only moves arm 0 proves nothing about the substitution
    hooks, which is exactly what round 3 shipped.  Each plant below names the
    arm it must move, and that arm is one the substitution feeds.
    """
    watched = {
        "score": ("arm", "0_card"),
        "entry": ("arm", "1_nemo_entry"),
        "external": ("arm", "2_nemo_external"),
        "stage1": ("arm", "3_nemo_stage1_out"),
        "stage2": ("arm", "4_nemo_stage2_out"),
        "prestage_rhs": ("arm", "5_nemo_prestage_rhs"),
    }[plant]

    def _value(report):
        rows = {arm["arm"]: arm for arm in report["arms"]}
        arm = rows[watched[1]]
        return max(arm["kt2"]["u"], arm["kt2"]["v"])

    before, after = _value(clean), _value(planted)
    return after > max(10.0 * before, 1.0e-6)


if __name__ == "__main__":
    sys.exit(main())

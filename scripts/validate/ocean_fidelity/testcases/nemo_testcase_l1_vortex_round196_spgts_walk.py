#!/usr/bin/env python3
"""Production-JIT VORTEX walk of NEMO's split-explicit barotropic solve.

Round 195 named the external-mode solve (``dyn_spg_ts``) as the first non-bit
producer of the kt=2 velocity residual that survives round 194's held
candidate, but could not name a statement inside it: no record carried the
solve's SUBSTEP operands.  Round 196's acquisition writes them, one frame per
sub-time-step, and this walk compares them -- in NEMO's own execution order --
against the operands legoESM's production-jitted step materialises for the
same substep through the existing private ``expose_barotropic_substeps`` hook.

The comparison is a walk, not a score sheet: the FIRST boundary at which the
two disagree is the owner, and everything after it is downstream of that
disagreement and reported only for context.
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
    _seed_from_record, _u_full,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    BAR, GateError, expected_masks, read_entry, require, score,
)

CASE = "VORTEX_VEC-zco"
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round196/"
    "oracle_spgts_substeps")
_HALO = 2

# NEMO's execution order inside one sub-time-step of dyn_spg_ts, with the
# legoESM operand each compiled boundary produces.  (boundary, NEMO group,
# legoESM trace key, stagger).  The order is the compiled one; the walk stops
# at the first row that is not bit-equal.
SUBSTEP_ORDER = (
    ("entry.eta",      "sshn_e_eff", "eta_entry",               "t"),
    ("entry.u",        "un_e_eff",   "u_entry",                 "u"),
    ("entry.v",        "vn_e_eff",   "v_entry",                 "v"),
    ("mid.u",          "ua_ext",     "u_mid",                   "u"),
    ("mid.v",          "va_ext",     "v_mid",                   "v"),
    ("mid.eta",        "sshp2_mid",  "eta_mid",                 "t"),
    ("mid.hu",         "hup2_e",     "transport_face_depth_u",  "u"),
    ("mid.hv",         "hvp2_e",     "transport_face_depth_v",  "v"),
    ("flux.u",         "zhU",        "transport_metric_u",      "u"),
    ("flux.v",         "zhV",        "transport_metric_v",      "v"),
    ("ssha",           "ssha_e",     "eta_continuity",          "t"),
    ("bck.eta",        "sshp2_bck",  "eta_pgf",                 "t"),
    ("spg.u",          "zu_spg",     "pgf_u",                   "u"),
    ("spg.v",          "zv_spg",     "pgf_v",                   "v"),
    ("cor.u",          "cor_u",      "cor_u",                   "u"),
    ("cor.v",          "cor_v",      "cor_v",                   "v"),
    ("trd.u",          "trd_u",      "trd_u",                   "u"),
    ("trd.v",          "trd_v",      "trd_v",                   "v"),
    ("new.u",          "ua_new",     "u_exit",                  "u"),
    ("new.v",          "va_new",     "v_exit",                  "v"),
    ("depth.hu",       "hu_e",       "face_depth_u_exit",       "u"),
    ("depth.hv",       "hv_e",       "face_depth_v_exit",       "v"),
)
# The loop-entry operands, scored once.  A disagreement here puts the owner
# UPSTREAM of dyn_spg_ts (in the slow-forcing assembly), not inside it.
ENTRY_ORDER = (
    ("entry.ssh_frc", "ssh_frc", "continuity_forcing", "t"),
    ("entry.zu_frc",  "zu_frc",  "slow_u",             "u"),
    ("entry.zv_frc",  "zv_frc",  "slow_v",             "v"),
)
SCALAR_ORDER = (
    ("mid.coef1", "ext_coef", 0, "mid_weight_1"),
    ("mid.coef2", "ext_coef", 1, "mid_weight_2"),
    ("mid.coef3", "ext_coef", 2, "mid_weight_3"),
    ("bck.coef0", "bck_coef", 0, "back_weight_0"),
    ("bck.coef1", "bck_coef", 1, "back_weight_1"),
    ("bck.coef2", "bck_coef", 2, "back_weight_2"),
    ("bck.coef3", "bck_coef", 3, "back_weight_3"),
)


# The frame contents, re-exported for the unit controls so the synthetic
# record they build cannot drift from the one the checker requires.
_CHECKER_PATH = HERE / "nemo_testcase_l1_vortex" / "check_records.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location(
        "vortex_record_checker", _CHECKER_PATH)
    if spec is None or spec.loader is None:        # pragma: no cover
        raise GateError("cannot load the record checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHECKER = _load_checker_module()
ENTRY_NAMES_FOR_TEST = _CHECKER._SPGTS_ENTRY
SUBSTEP_NAMES_FOR_TEST = _CHECKER._SPGTS_SUBSTEP
EXIT_NAMES_FOR_TEST = _CHECKER._SPGTS_EXIT


def _checker():
    return _CHECKER


def read_spgts(root: Path, kt: int) -> tuple[dict, dict]:
    """Read one per-substep barotropic record through the acquisition's own
    self-describing parser.  Returns (header fields, {group name: array})."""
    checker = _checker()
    path = root / f"oracle_spgts_kt{kt:08d}.bin"
    parsed = checker.parse_record(path)
    require(parsed["magic"] == "NEMO_L1_SPGTS1", f"{path}: wrong record family")
    require(parsed["header"][1] == kt, f"{path}: header step is not {kt}")
    raw = path.read_bytes()
    offset = 16 + 4 * 15
    out: dict[str, np.ndarray] = {}
    for name, meta in parsed["groups"].items():
        offset += 32
        count = meta["doubles"]
        values = np.frombuffer(raw[offset:offset + 8 * count],
                               dtype=np.float64).copy()
        offset += 8 * count
        if meta["rank"] == 1:
            out[name] = values
            continue
        nx, ny = meta["shape"]
        plane = values.reshape((nx, ny), order="F")
        if (nx, ny) == (parsed["nx"], parsed["ny"]):
            plane = plane[_HALO:-_HALO, _HALO:-_HALO]
        out[name] = plane.T
    require(offset == len(raw), f"{path}: parser did not consume the record")
    meta = {"kt": parsed["header"][1], "icycle": parsed["icycle"],
            "jpi": parsed["nx"], "jpj": parsed["ny"], "jpk": parsed["nz"],
            "groups": len(parsed["groups"])}
    return meta, out


def _lego_plane(values, stagger):
    """One substep frame of a legoESM trace array, on NEMO's interior."""
    values = np.asarray(values, dtype=np.float64)
    if stagger == "u":
        return values[:, 1:]
    if stagger == "v":
        return values[1:, :]
    return values


def run(root: Path, *, kt: int = 1, allow_dirty: bool = False,
        plant: str | None = None, substeps: int | None = None) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    require(jax.default_backend() == "cpu", "the barotropic walk must run on CPU")

    valid = tuple(name for name, *_ in SUBSTEP_ORDER) \
        + tuple(name for name, *_ in ENTRY_ORDER)
    require(plant is None or plant in valid,
            f"unknown plant {plant!r}; expected one of {valid}")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks3 = expected_masks(card)
    masks = {"u": np.asarray(masks3["u"])[..., 0],
             "v": np.asarray(masks3["v"])[..., 0],
             "t": np.asarray(masks3["ssh"], dtype=bool)}
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry = read_entry(root / f"oracle_step_entry_kt{kt:08d}.bin", CASE,
                       expect_interior=interior)
    seed = _seed_from_record(card.recipe.initial_state, entry, nlev)

    meta, groups = read_spgts(root, kt)
    n_loop = meta["icycle"] if substeps is None else min(substeps, meta["icycle"])

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True))
    result = jax.device_get(model.step(seed, dt=card.dt_s))
    trace = {key: np.asarray(value) for key, value in result.substeps.items()}
    traced_loops = int(next(iter(trace.values())).shape[0])
    require(traced_loops == meta["icycle"],
            f"legoESM ran {traced_loops} barotropic substeps and NEMO's record "
            f"declares {meta['icycle']}; the two solves are not the same loop")

    # NEMO writes only ssha_e per substep; the swap at dynspg_ts.F90:816 makes
    # substep jn's sshn_e the previous substep's ssha_e (and the loop-entry
    # sshn_e at jn=1).  Same for un_e/vn_e from ua_new/va_new.  Deriving them
    # is not an assumption: it is the compiled swap, quoted.
    for jn in range(1, meta["icycle"] + 1):
        prev = f"j{jn - 1:03d}_"
        groups[f"j{jn:03d}_sshn_e_eff"] = (
            groups["i000_sshn_e"] if jn == 1 else groups[prev + "ssha_e"])
        groups[f"j{jn:03d}_un_e_eff"] = (
            groups["i000_un_e"] if jn == 1 else groups[prev + "ua_new"])
        groups[f"j{jn:03d}_vn_e_eff"] = (
            groups["i000_vn_e"] if jn == 1 else groups[prev + "va_new"])

    rows: list[dict] = []
    first = None

    def add(name, reference, candidate, stagger, substep):
        nonlocal first
        planted = plant == name
        use = masks[stagger]
        row = score(f"{CASE}.spgts.j{substep:03d}.{name}", reference,
                    candidate, use, plant=planted)
        active = np.asarray(use, dtype=bool)
        row["cells_unequal"] = int(np.count_nonzero(
            (candidate != reference)[active]))
        row["max_abs"] = float(np.max(np.abs(
            (candidate - reference)[active])))
        row["substep"] = substep
        row["boundary"] = name
        row["planted"] = planted
        row["bit_exact"] = row["cells_unequal"] == 0 and not planted
        rows.append(row)
        if first is None and not row["bit_exact"]:
            first = {"substep": substep, "boundary": name,
                     "cells_unequal": row["cells_unequal"],
                     "max_abs": row["max_abs"],
                     "normalized_max_abs": row["normalized_max_abs"],
                     "status": row["status"]}

    # 1. loop entry: the operands dyn_spg_ts is HANDED.
    for name, group, key, stagger in ENTRY_ORDER:
        add(name, groups[f"i000_{group}"], _lego_plane(trace[key][0], stagger),
            stagger, 0)

    # 2. the substep loop, in NEMO's compiled order.
    scalars = []
    for jn in range(1, n_loop + 1):
        prefix = f"j{jn:03d}_"
        for name, group, index, key in SCALAR_ORDER:
            nemo_value = float(groups[prefix + group][index])
            lego_value = float(np.asarray(trace[key])[jn - 1])
            scalars.append({"substep": jn, "name": name,
                            "nemo": nemo_value, "legoesm": lego_value,
                            "bit_exact": nemo_value == lego_value})
        for name, group, key, stagger in SUBSTEP_ORDER:
            add(name, groups[prefix + group],
                _lego_plane(trace[key][jn - 1], stagger), stagger, jn)

    # 3. the loop exit: the filtered quintuple the stages consume.
    exit_rows = []
    for name, group, key, stagger in (
            ("exit.uu_b", "uu_b_aa", "u_exit", "u"),
            ("exit.vv_b", "vv_b_aa", "v_exit", "v"),
            ("exit.ssh",  "ssh_aa",  "eta_exit", "t")):
        exit_rows.append({"name": name,
                          "nemo_max_abs": float(np.max(np.abs(
                              groups[f"o000_{group}"])))})

    bad_scalars = [s for s in scalars if not s["bit_exact"]]
    report = {
        "case": CASE, "kt": kt, "git_sha": sha, "oracle_root": str(root),
        "record": meta, "bar": BAR,
        "precision_policy": "fp64/libm", "jax_backend": jax.default_backend(),
        "execution_regime": "production_step_jit",
        "substeps_walked": n_loop,
        "plant": plant,
        "scalar_rows": len(scalars),
        "scalar_non_bit": bad_scalars[:8],
        "first_non_bit_boundary": first,
        "rows": rows,
        "exit_frame": exit_rows,
        "status": "PLANT-FIRED" if plant else ("DEBT" if first else "BIT"),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--kt", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=None,
                        help="walk only the first N substeps (the whole loop "
                             "by default)")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--plant", help="perturb one boundary; MUST exit 1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, kt=args.kt, allow_dirty=args.allow_dirty,
                     plant=args.plant, substeps=args.substeps)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    shown = 0
    for row in report["rows"]:
        if row["bit_exact"] and shown > 40:
            continue
        shown += 1
        print(f"{row['name']:<48} unequal={row['cells_unequal']:<6d} "
              f"max={row['max_abs']:.17e} exact={row['bit_exact']}")
    print("scalar_non_bit:", report["scalar_non_bit"])
    print("first_non_bit_boundary:", report["first_non_bit_boundary"])
    print("STATUS", report["status"])
    return 1 if report["status"] != "BIT" else 0


if __name__ == "__main__":
    sys.exit(main())

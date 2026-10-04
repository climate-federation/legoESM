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
    _seed_from_record, _u_full, _v_full,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    BAR, GateError, expected_masks, read_entry, require, score,
)

CASE = "VORTEX_VEC-zco"
# Round 215 / VORTEX_SMT round 5: the SAME walk on the seamount pair.  The
# routine under test is shared -- dyn_spg_ts is one compiled subroutine and
# legoESM has one barotropic solve -- so the card is a PARAMETER here rather
# than a second copy of this file.  Each card names the evidence root its own
# acquisition wrote; nothing is defaulted across cards.
SMT_CASES = {
    "VORTEX_SMT-zps": "VORTEX_SMT_R5_OMIP_L1_P3",
    "VORTEX_SMT_VEC-zps": "VORTEX_SMT_R5_VEC_R8_OMIP_L1_P3",
}
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


def require_case_matches_root(case: str, root: Path) -> None:
    """Refuse a card walked against the OTHER card's acquisition.

    This cannot be caught downstream: the step-entry record carries no case
    stamp, and the flat and seamount decks are the same 30 km grid, so their
    records are the same size and ``expect_interior`` passes on either.
    Both entry points (the walk and the conditioning probe) call it.
    """
    expected_dir = SMT_CASES.get(case)
    if expected_dir is not None:
        require(expected_dir in root.parts,
                f"{case} must be walked against its own acquisition: no "
                f"'{expected_dir}' component in --oracle-root {root}")
    else:
        require(not any(d in root.parts for d in SMT_CASES.values()),
                f"{case} is the flat card and --oracle-root {root} is a "
                "seamount acquisition")


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


def run(root: Path, *, case: str = CASE, kt: int = 1, allow_dirty: bool = False,
        plant: str | None = None, substeps: int | None = None,
        nemo_entry_forcing: bool = False,
        nemo_entry_velocity: bool = False,
        nemo_substep_coriolis: bool = False,
        nemo_substep_pgf: bool = False,
        nemo_depth_average: Path | None = None,
        dump_entry_coriolis: Path | None = None) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import (
        allow_dirty_stamps, git_sha, worktree_stamp,
    )

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    # An arm run with a HELD patch applied carries only "<sha>-dirty" without
    # this; the worktree stamp records the patch's own diff hash, which is
    # what pins the arm's numbers to the code that produced them.
    tree = worktree_stamp(allow_dirty=allow_dirty)
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

    require_case_matches_root(case, root)

    card = build_nemo_testcase_card(case)
    nlev = int(card.recipe.z_coord.n_levels)
    masks3 = expected_masks(card)
    masks = {"u": np.asarray(masks3["u"])[..., 0],
             "v": np.asarray(masks3["v"])[..., 0],
             "t": np.asarray(masks3["ssh"], dtype=bool)}
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry = read_entry(root / f"oracle_step_entry_kt{kt:08d}.bin", case,
                       expect_interior=interior)
    seed = _seed_from_record(card.recipe.initial_state, entry, nlev)

    meta, groups = read_spgts(root, kt)
    n_loop = meta["icycle"] if substeps is None else min(substeps, meta["icycle"])

    # Round 196's ONE VARIABLE for the ownership question: whether the loop
    # is handed NEMO's own recorded slow forcing (dynspg_ts.f90:362, after the
    # barotropic Coriolis subtraction -- exactly the boundary legoESM's
    # override lands on) or legoESM's own.  Everything else is production.
    override = None
    if nemo_entry_forcing:
        import jax.numpy as jnp
        override = (jnp.asarray(_u_full(groups["i000_zu_frc"][..., None])[..., 0]),
                    jnp.asarray(_v_full(groups["i000_zv_frc"][..., None])[..., 0]))
    if nemo_entry_velocity:
        # The other half of the loop-entry pair: replace the carried
        # barotropic velocity the loop starts from with NEMO's own recorded
        # un_e/vn_e.  Self-checking -- if the substitution does not bind,
        # the first substep's entry rows do not become bit-exact and the arm
        # says so instead of reporting a null.
        import jax.numpy as jnp
        seed = seed._replace(
            uu_b=seed.uu_b.replace(data=jnp.asarray(
                _u_full(groups["i000_un_e"][..., None])[..., 0])),
            vv_b=seed.vv_b.replace(data=jnp.asarray(
                _v_full(groups["i000_vn_e"][..., None])[..., 0])))
    # Round 197's ONE VARIABLE: a PER-SUBSTEP operand of the compiled loop.
    # The stacks are NEMO's own recorded frames in substep order, so the
    # override the scan reads at substep jn is the array NEMO's own
    # dyn_cor_2D (or ts_bck_interp pressure gradient) wrote at that substep,
    # not one frame reused for the whole window.
    def _stack(group_u, group_v):
        import jax.numpy as jnp
        us = np.stack([_u_full(groups[f"j{jn:03d}_{group_u}"][..., None])[..., 0]
                       for jn in range(1, meta["icycle"] + 1)])
        vs = np.stack([_v_full(groups[f"j{jn:03d}_{group_v}"][..., None])[..., 0]
                       for jn in range(1, meta["icycle"] + 1)])
        return jnp.asarray(us), jnp.asarray(vs)

    # Round 215's split arm: substitute NEMO's OWN depth average of the slow
    # forcing -- rebuilt by nemo_testcase_l1_vortex_round215_slow_forcing_split
    # from NEMO's recorded 3-D right-hand side and NEMO's own mesh operands --
    # at the boundary legoESM forms the same quantity, BEFORE the barotropic
    # Coriolis subtraction.  What survives is the subtraction's own share.
    depth_override = None
    if nemo_depth_average is not None:
        import jax.numpy as jnp
        _npz = np.load(nemo_depth_average)
        depth_override = (jnp.asarray(_npz["ue_rhs"]),
                          jnp.asarray(_npz["ve_rhs"]))
    # Round 216's ONE VARIABLE for the remainder round 215 left PLAUSIBLE:
    # legoESM's OWN loop-entry barotropic Coriolis subtraction, read out at
    # the boundary it is formed (dynspg_ts.f90:292's counterpart).  The model
    # already offers this boundary as a CALLABLE observer on the same hook
    # the forcing override uses, so nothing in the package changes; the probe
    # that compares it to NEMO's own is
    # nemo_testcase_l1_vortex_round216_entry_coriolis.
    _entry_dump: dict[str, np.ndarray] = {}
    if dump_entry_coriolis is not None:
        require(override is None,
                "the entry-Coriolis dump and the entry-forcing override share "
                "one hook; they cannot both be requested")

        def _observe(incoming_u, incoming_v, cor_u, cor_v,
                     umask, vmask, final_u, final_v):
            _entry_dump.update(
                incoming_u=np.asarray(incoming_u),
                incoming_v=np.asarray(incoming_v),
                cor_u=np.asarray(cor_u), cor_v=np.asarray(cor_v),
                umask=np.asarray(umask), vmask=np.asarray(vmask),
                final_u=np.asarray(final_u), final_v=np.asarray(final_v))

        override = _observe
    cor_override = _stack("cor_u", "cor_v") if nemo_substep_coriolis else None
    pgf_override = _stack("zu_spg", "zv_spg") if nemo_substep_pgf else None
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=override,
            slow_forcing_depth_override=depth_override,
            barotropic_substep_coriolis_override=cor_override,
            barotropic_substep_pgf_override=pgf_override))
    result = jax.device_get(model.step(seed, dt=card.dt_s))
    if dump_entry_coriolis is not None:
        # Self-checking: the observer must have FIRED and must have seen a
        # subtraction that is not a zero, or the arm reports nothing.
        require(bool(_entry_dump),
                "the loop-entry observer never fired; this card does not take "
                "the live barotropic-Coriolis-split path")
        require(float(np.max(np.abs(_entry_dump["cor_u"]))) > 0.0,
                "the loop-entry Coriolis subtraction is identically zero; a "
                "comparison against it would perturb a zero")
        np.savez(dump_entry_coriolis, **_entry_dump)
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
        row = score(f"{case}.spgts.j{substep:03d}.{name}", reference,
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

    # Per-CELL Coriolis check.  The receipt's trend-to-velocity ratio is a
    # ratio of two field maxima that need not sit at the same cell, so it is
    # only suggestive.  This is the same quantity cell by cell: at each
    # substep, over the faces where the entering velocity differs, the ratio
    # of the Coriolis-trend difference to the velocity difference.  A
    # faithful operator gives the Coriolis parameter; the measured median is
    # what decides whether the operator or its operand is in question.
    coriolis = []
    for jn in (1, 8, 16, 24, 28, 32, 40, n_loop):
        if jn > n_loop:
            continue
        prefix = f"j{jn:03d}_"
        du = np.abs(_lego_plane(trace["u_entry"][jn - 1], "u")
                    - groups[prefix + "un_e_eff"])
        dt_ = np.abs(_lego_plane(trace["cor_u"][jn - 1], "u")
                     - groups[prefix + "cor_u"])
        active = np.asarray(masks["u"], dtype=bool) & (du > 0.0)
        if not active.any():
            continue
        ratio = dt_[active] / du[active]
        coriolis.append({
            "substep": jn, "faces": int(active.sum()),
            "median_ratio": float(np.median(ratio)),
            "p95_ratio": float(np.percentile(ratio, 95)),
            "max_ratio": float(np.max(ratio)),
        })

    bad_scalars = [s for s in scalars if not s["bit_exact"]]
    report = {
        "case": case, "kt": kt, "git_sha": sha, "worktree": tree,
        # Which arm produced this artifact.  Without these an arm that
        # injected an operand is indistinguishable from one that did not.
        "arm_nemo_entry_forcing": bool(nemo_entry_forcing),
        "arm_nemo_entry_velocity": bool(nemo_entry_velocity),
        "arm_nemo_substep_coriolis": bool(nemo_substep_coriolis),
        "arm_nemo_substep_pgf": bool(nemo_substep_pgf),
        "arm_nemo_depth_average": (None if nemo_depth_average is None
                                   else str(nemo_depth_average)),
        "oracle_root": str(root),
        "record": meta, "bar": BAR,
        "precision_policy": "fp64/libm", "jax_backend": jax.default_backend(),
        "execution_regime": "production_step_jit",
        "substeps_walked": n_loop,
        "entry_forcing_arm": ("nemo_recorded" if nemo_entry_forcing
                              else "legoesm_production"),
        "entry_velocity_arm": ("nemo_recorded" if nemo_entry_velocity
                               else "legoesm_production"),
        "substep_coriolis_arm": ("nemo_recorded" if nemo_substep_coriolis
                                 else "legoesm_production"),
        "substep_pgf_arm": ("nemo_recorded" if nemo_substep_pgf
                            else "legoesm_production"),
        # The arms are self-checking: a substitution that does not BIND
        # reports NULL, not a result.  These are the rows the override is
        # supposed to make bit-exact by construction.
        "substitution_bound": {
            name: all(r["cells_unequal"] == 0
                      for r in rows if r["boundary"] == name
                      and r["substep"] >= 1)
            for name in ("cor.u", "cor.v", "spg.u", "spg.v")},
        "end_of_window": {
            r["boundary"]: {"max_abs": r["max_abs"],
                            "normalized_max_abs": r["normalized_max_abs"],
                            "cells_unequal": r["cells_unequal"]}
            for r in rows
            if r["substep"] == n_loop
            and r["boundary"] in ("new.u", "new.v", "ssha")},
        "plant": plant,
        "scalar_rows": len(scalars),
        "scalar_non_bit": bad_scalars[:8],
        "coriolis_per_cell_ratio": coriolis,
        "first_non_bit_boundary": first,
        "rows": rows,
        "exit_frame": exit_rows,
        "status": "PLANT-FIRED" if plant else ("DEBT" if first else "BIT"),
    }
    return report


def conditioning(root: Path, *, case: str = CASE, kt: int = 1,
                 allow_dirty: bool = False) -> dict:
    """How much does the solve amplify ONE last-bit change at its entry?

    A walk that finds every boundary inside the loop at the rounding floor
    while the loop's OUTPUT is 1e-08 has two readings: a statement inside the
    loop is wrong, or the loop amplifies what it is handed.  This arm settles
    it without NEMO: it perturbs legoESM's own barotropic entry velocity by
    exactly one unit in the last place at one wet face and re-runs the SAME
    production-jitted step, so the only difference between the two traces is
    that one ULP.  The per-substep response is the recurrence's own
    conditioning, measured rather than argued.
    """
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require_case_matches_root(case, root)
    card = build_nemo_testcase_card(case)
    nlev = int(card.recipe.z_coord.n_levels)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry = read_entry(root / f"oracle_step_entry_kt{kt:08d}.bin", case,
                       expect_interior=interior)
    seed = _seed_from_record(card.recipe.initial_state, entry, nlev)
    masks3 = expected_masks(card)
    use = np.asarray(masks3["u"])[..., 0]

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True))

    def trace_of(state):
        return {k: np.asarray(v) for k, v in
                jax.device_get(model.step(state, dt=card.dt_s)).substeps.items()}

    base = trace_of(seed)
    carried = np.asarray(seed.uu_b.data if hasattr(seed.uu_b, "data")
                         else seed.uu_b, dtype=np.float64).copy()
    # One ULP at EVERY wet face, not at one of them.  A single-cell probe
    # spreads more slowly than the real difference and so never reaches the
    # state the real difference is in when it starts to grow -- it would
    # prove nothing about the growth and would look like a refutation.  The
    # perturbation is field-wide and last-bit, exactly the shape of the
    # entry disagreement the walk measures.
    owned = carried[:, 1:]
    where = np.unravel_index(int(np.argmax(np.abs(np.where(use, owned, 0.0)))),
                             owned.shape)
    before = owned[where]
    bumped_owned = np.where(use, np.nextafter(owned, np.inf), owned)
    # ``owned`` is a VIEW into ``carried``; measure the perturbation BEFORE
    # writing it back, or the recorded size is identically zero and the
    # record of the control says it perturbed nothing.
    perturbation_max = float(np.max(np.abs(bumped_owned - owned)))
    carried[:, 1:] = bumped_owned
    import jax.numpy as jnp
    bumped = seed._replace(uu_b=seed.uu_b.replace(data=jnp.asarray(carried)))
    moved = trace_of(bumped)
    require(float(np.max(np.abs(
        np.asarray(bumped.uu_b.data) - np.asarray(seed.uu_b.data)))) > 0.0,
        "the one-ULP probe perturbed nothing; it would prove nothing")

    n_loop = int(base["u_exit"].shape[0])
    rows = []
    for jn in range(1, n_loop + 1):
        delta = np.abs(_lego_plane(moved["u_exit"][jn - 1], "u")
                       - _lego_plane(base["u_exit"][jn - 1], "u"))
        rows.append({"substep": jn,
                     "max_abs": float(np.max(delta[use])),
                     "cells_moved": int(np.count_nonzero(delta[use]))})
    first = next((r["max_abs"] for r in rows if r["max_abs"] > 0.0), 0.0)
    final = rows[-1]["max_abs"]
    return {
        "case": case, "kt": kt, "git_sha": sha, "arm": "one_ulp_entry_probe",
        "perturbed_faces": int(np.count_nonzero(use)),
        "largest_cell": [int(where[0]), int(where[1]) + 1],
        "perturbation_at_largest_cell": float(
            np.nextafter(before, np.inf) - before),
        "perturbation_max": perturbation_max,
        "substeps": n_loop,
        "first_responding_substep_max_abs": first,
        "final_substep_max_abs": final,
        "amplification": (final / first) if first > 0 else None,
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case", default=CASE,
                        choices=(CASE, *sorted(SMT_CASES)),
                        help="which card the walk runs; the seamount "
                             "cards need their own --oracle-root")
    parser.add_argument("--kt", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=None,
                        help="walk only the first N substeps (the whole loop "
                             "by default)")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--plant", help="perturb one boundary; MUST exit 1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nemo-entry-forcing", action="store_true",
                        help="one-variable arm: hand the loop NEMO's recorded "
                             "slow forcing instead of legoESM's own")
    parser.add_argument("--nemo-entry-velocity", action="store_true",
                        help="one-variable arm: start the loop from NEMO's "
                             "recorded barotropic entry velocity")
    parser.add_argument("--nemo-substep-coriolis", action="store_true",
                        help="one-variable arm: substitute NEMO's recorded "
                             "per-substep 2-D Coriolis trend inside the "
                             "compiled scan (dynspg_ts.f90:503)")
    parser.add_argument("--nemo-substep-pgf", action="store_true",
                        help="one-variable arm: substitute NEMO's recorded "
                             "per-substep surface pressure gradient "
                             "(dynspg_ts.f90:498)")
    parser.add_argument("--nemo-depth-average", type=Path,
                        help="one-variable arm: substitute NEMO's own depth "
                             "average of the slow forcing (stp2d.f90:178) "
                             "before the barotropic Coriolis subtraction")
    parser.add_argument("--dump-entry-coriolis", type=Path,
                        help="save legoESM's OWN loop-entry barotropic "
                             "Coriolis subtraction (and the forcing either "
                             "side of it) as an npz, for comparison against "
                             "NEMO's dynspg_ts.f90:292 trend")
    parser.add_argument("--one-ulp-entry-probe", action="store_true",
                        help="legoESM-vs-legoESM conditioning arm: perturb the "
                             "barotropic entry velocity by one ULP and report "
                             "the per-substep response")
    args = parser.parse_args(argv)
    if args.one_ulp_entry_probe:
        report = conditioning(args.oracle_root, case=args.case, kt=args.kt,
                              allow_dirty=args.allow_dirty)
        if args.output:
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n")
        for row in report["rows"]:
            print(f"j{row['substep']:03d} max={row['max_abs']:.17e} "
                  f"cells={row['cells_moved']}")
        print("amplification:", report["amplification"])
        return 0
    try:
        report = run(args.oracle_root, case=args.case, kt=args.kt,
                     allow_dirty=args.allow_dirty,
                     plant=args.plant, substeps=args.substeps,
                     nemo_entry_forcing=args.nemo_entry_forcing,
                     nemo_entry_velocity=args.nemo_entry_velocity,
                     nemo_substep_coriolis=args.nemo_substep_coriolis,
                     nemo_substep_pgf=args.nemo_substep_pgf,
                     nemo_depth_average=args.nemo_depth_average,
                     dump_entry_coriolis=args.dump_entry_coriolis)
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
    for row in report["coriolis_per_cell_ratio"]:
        print(f"coriolis j{row['substep']:03d} faces={row['faces']:<5d} "
              f"median={row['median_ratio']:.3e} p95={row['p95_ratio']:.3e} "
              f"max={row['max_ratio']:.3e}")
    print("substitution_bound:", report["substitution_bound"])
    for name, row in sorted(report["end_of_window"].items()):
        print(f"end_of_window {name:<8} unequal={row['cells_unequal']:<6d} "
              f"max={row['max_abs']:.17e} "
              f"norm={row['normalized_max_abs']:.17e}")
    print("first_non_bit_boundary:", report["first_non_bit_boundary"])
    print("STATUS", report["status"])
    return 1 if report["status"] != "BIT" else 0


if __name__ == "__main__":
    sys.exit(main())

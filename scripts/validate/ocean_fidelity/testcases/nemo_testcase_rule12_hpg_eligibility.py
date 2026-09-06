#!/usr/bin/env python3
"""Rule-12 exact-input row for the changed HPG/QCO operator on the lane-1 tanks.

Round 25 landed two source associations inside NEMO's hydrostatic pressure
gradient: the ``key_qco`` depth operand ``gdept_z0 = gdept_0*(1+r3t) - ssh``
materialised as two statements (``domzgr_substitute.h90:139,145``), and the
``nemo_sco`` acceleration carried straight into the momentum RHS instead of
through a pressure round trip (``dynhpg.F90:359,383``).  That change was shown
bit-exact given NEMO's own inputs on GYRE only.  Rule 12 requires the same
exact-input row on EVERY card that executes it, and
``nemo_testcase_recipe.py:92,274,914`` pins ``pgf_scheme='nemo_sco'`` for all
of them.  This gate supplies the row for ``LOCK_EXCHANGE-zco`` and
``OVERFLOW-zps``.

WHAT THE SCORED FRAME IS -- read from the PREPROCESSED source each card
COMPILES (``<config>/BLD/ppsrc/nemo/``), not from the shipped ``.F90``.
``oracle_rhs_kt00000001.bin`` is written by ``l1_dump_rhs`` at
``MY_SRC/stprk3.F90:206``, immediately after ``stp_2D`` at :204 and before
stage 1 at :215.  In ``stp2d.F90`` order the calls that can touch
``uu(:,:,:,Krhs)`` are ``:128`` ``dyn_hpg``, ``:131`` ``dyn_ldf``, ``:146``
``dyn_vor``, ``:172`` ``dyn_adv_up3``, ``:196`` ``dyn_drg_init`` and ``:281``
``dyn_spg_ts``; ``stp2d.F90`` itself contains zero direct ``uu``/``vv``
assignments.  At a rest start the frame is ``dyn_hpg`` ALONE, and every step
of that is asserted mechanically by :func:`assert_ppsrc_isolation` rather than
argued in prose:

* ``dyn_hpg`` ASSIGNS.  ``hpg_sco`` writes ``puu(ji,jj,jk,Krhs) = zhpi + zuap``
  (ppsrc ``dynhpg.f90``, "RK3 case: dyn_hpg always called first"), so whatever
  the RHS held before ``:128`` is overwritten and cannot reach the dump.
* ``dyn_ldf`` (``:131``) is called unconditionally and is inert only because
  both cards set ``ln_dynldf_OFF = .true.``  That is a NAMELIST fact; the gate
  asserts the legoESM equivalent instead of relying on it.
* ``dyn_vor`` (``:146``) accumulates ``pu_rhs + zuav*(zwz+zwz)`` where ``zuav``
  is built from ``zwx``/``zwy``, each a product with the entry velocity.  The
  gate asserts NEMO's dumped entry velocity is identically zero, so the added
  term is exactly ``0.0``.
* ``stp2d.F90``'s ``SELECT CASE( n_dynadv )`` has three arms.  ``dyn_keg``
  (``:163``) plus ``dyn_zad`` (``:165``) and ``dyn_adv_cen2`` (``:169``) DO
  write the 3-D RHS; ``ln_dynadv_up3 = .true.`` is what selects neither, and
  the gate now asserts that selection instead of leaving it unsaid.
* ``dyn_adv_up3`` (``:172``) is called WITH ``pUe``/``pVe``.  In the ppsrc every
  ``puu``/``pvv(...,Krhs)`` write sits in the ELSE of ``IF( PRESENT( pUe ) )``,
  so on a flux-form card the 3-D advection is not in this frame at all.
* ``dyn_drg_init`` (``:196``) declares ``puu, pvv`` ``INTENT(in)``.
* ``dyn_spg_ts`` (``:281``) writes NOTHING to ``puu``/``pvv``.  Both cards
  compile ``key_RK3``, and the ppsrc ``dynspg_ts.f90`` contains ZERO
  assignments to ``puu``/``pvv``: the depth-mean removal (shipped
  ``dynspg_ts.F90:344-345``) lives in the ``#else`` at ``:303`` and the
  barotropic add-back (shipped ``:938-975``) in the ``#else`` at ``:910`` --
  both the MLF arm, neither compiled here.

Round 26 read the SHIPPED ``dynspg_ts.F90`` and concluded the frame was a
composite of the pressure gradient and the mode split.  That is RETRACTED:
under ``key_RK3`` neither statement exists in the object code.  Its other
correction stands -- the frame is a ``stp_2D`` frame, not a stage-1 frame, and
the citation of ``stprk3_stg.F90:309-334`` was wrong -- so the row keeps the
name ``kt1.stp2d.momentum_rhs`` while the ISOLATION is full.
"""

from __future__ import annotations

import argparse
import json
import re
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
from legoesm.ocean.fidelity.provenance import (
    allow_dirty_stamps,
    scoped_allow_dirty,
    worktree_stamp,
)


# The NEMO build tree.  Each card's PREPROCESSED source -- the code the card
# actually compiles -- lives under ``<config>/BLD/ppsrc/nemo/``; the shipped
# ``src/OCE/**.F90`` still carries the branches the cpp keys removed.
NEMO_ROOT = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
PPSRC_CONFIG = {
    "LOCK_EXCHANGE-zco": "tests/LOCK_EXCHANGE_OMIP_L1_P3",
    "OVERFLOW-zps": "tests/OVERFLOW_OMIP_L1_P3",
}
_RHS_WRITE = re.compile(r"^\s*(?:puu|pvv)\s*\(")


def _routine(lines: list[str], name: str) -> tuple[int, int]:
    """Half-open [start, stop) line span of ``SUBROUTINE <name>``."""
    start = stop = None
    for index, line in enumerate(lines):
        if re.match(rf"^\s*SUBROUTINE\s+{name}\s*\(", line):
            start = index
        elif start is not None and re.match(
                rf"^\s*END\s+SUBROUTINE\s+{name}\s*$", line.rstrip()):
            stop = index
            break
    require(start is not None and stop is not None,
            f"SUBROUTINE {name} not found in the preprocessed source")
    return start, stop


def assert_ppsrc_isolation(case: str, nemo_root: Path = NEMO_ROOT,
                           config: str | None = None) -> dict:
    """Prove, from the code the card COMPILES, that the dumped kt=1 frame is
    ``dyn_hpg`` alone once the rest-state and ``ln_dynldf_OFF`` preconditions
    hold.  Every fact below is a grep over ``BLD/ppsrc/nemo/``; none is prose.

    All facts are evaluated and reported together rather than short-circuiting
    on the first.  MEASURED, not claimed: the planted control
    (``--plant-ppsrc cfgs/DINO``, an MLF build) fires THREE of them at once --
    the cpp keys, the ``dynspg_ts`` RHS writes, and the ``hpg_sco`` assignment
    (on MLF it accumulates).  The ``dyn_drg_init`` INTENT and the
    ``dyn_adv_up3`` guard pass on that build too, so those two have no
    non-vacuity control here and are stated as unproven by plant.
    """
    # ``config`` overrides the card's own build ONLY for the planted control.
    config = nemo_root / (config or PPSRC_CONFIG[case])
    ppsrc = config / "BLD/ppsrc/nemo"
    require(ppsrc.is_dir(), f"{case}: missing preprocessed source {ppsrc}")
    violations: list[str] = []

    # WHICH routines run is a namelist selection, and asserting the ppsrc
    # without it left "hpg_sco ASSIGNS" and "dyn_adv_up3 writes only 2-D"
    # resting on unstated facts.  stp2d.F90's SELECT CASE( n_dynadv ) also
    # offers dyn_keg (:163) + dyn_zad (:165) and dyn_adv_cen2 (:169), both of
    # which WOULD write the 3-D RHS; ln_dynadv_up3 is what excludes them.
    namelist = (config / "EXP00/namelist_cfg").read_text()
    selections = {
        name: bool(re.search(rf"^\s*{name}\s*=\s*\.true\.", namelist,
                             re.M))
        for name in ("ln_dynadv_up3", "ln_hpg_sco", "ln_dynldf_OFF",
                     "ln_dynadv_vec", "ln_dynadv_cen2")
    }
    keys = next(config.glob("cpp_*.fcm")).read_text().split()
    if not selections["ln_hpg_sco"]:
        violations.append(
            "ln_hpg_sco is not .true., so dyn_hpg does not run hpg_sco and "
            "the 'hpg_sco ASSIGNS' fact is about a routine this card may not "
            "call")
    if not selections["ln_dynadv_up3"] or selections["ln_dynadv_vec"] \
            or selections["ln_dynadv_cen2"]:
        violations.append(
            f"momentum advection is not the flux-form UP3 arm ({selections}); "
            "stp2d.F90's other SELECT CASE arms (dyn_keg + dyn_zad, "
            "dyn_adv_cen2) write the 3-D RHS and would be in this frame")
    if "key_RK3" not in keys:
        violations.append(
            f"does not compile key_RK3 ({keys}); the MLF arm of dynspg_ts "
            "would then write puu/pvv and the frame is a composite")

    # dyn_spg_ts: zero writes to the 3-D momentum RHS under key_RK3.
    spg = (ppsrc / "dynspg_ts.f90").read_text().splitlines()
    spg_writes = [index + 1 for index, line in enumerate(spg)
                  if _RHS_WRITE.match(line)]
    if spg_writes:
        violations.append(
            f"ppsrc dynspg_ts.f90 assigns puu/pvv at {spg_writes}; the "
            "depth-mean removal or the barotropic add-back compiled after all")

    # dyn_drg_init reads the 3-D velocity, never writes it.
    drg_start, drg_stop = _routine(spg, "dyn_drg_init")
    drg_intent = [line.strip() for line in spg[drg_start:drg_stop]
                  if "INTENT(in" in line
                  and ("puu, pvv" in line or "puu   , pvv" in line)]
    if not drg_intent:
        violations.append("dyn_drg_init does not declare puu/pvv INTENT(in)")

    # dyn_hpg (hpg_sco) ASSIGNS -- the RHS is overwritten, not accumulated.
    hpg = (ppsrc / "dynhpg.f90").read_text().splitlines()
    hpg_start, hpg_stop = _routine(hpg, "hpg_sco")
    hpg_writes = [line.strip() for line in hpg[hpg_start:hpg_stop]
                  if _RHS_WRITE.match(line)]
    if not hpg_writes or not all(
            re.match(r"^p[uv][uv]\([^)]*Krhs\)\s*=\s*zhp[ij]", line)
            for line in hpg_writes):
        violations.append(
            f"hpg_sco does not ASSIGN the momentum RHS: {hpg_writes}; a "
            "read-modify-write there would carry whatever preceded dyn_hpg")

    # dyn_adv_up3 with pUe present writes only the 2-D RHS: every 3-D write
    # sits in the ELSE of an IF( PRESENT( pUe ) ).  Fortran block-IF nesting
    # means a stack, not a flag.
    up3 = (ppsrc / "dynadv_up3.f90").read_text().splitlines()
    up3_start, up3_stop = _routine(up3, "dyn_adv_up3")
    guarded: list[int] = []
    unguarded: list[int] = []
    stack: list[list] = []
    for index in range(up3_start, up3_stop):
        line = up3[index]
        if line.lstrip().startswith("!"):        # commented-out code
            continue
        if _RHS_WRITE.match(line):
            in_pue_else = any(is_pue and in_else for is_pue, in_else in stack)
            (guarded if in_pue_else else unguarded).append(index + 1)
        if re.search(r"\bIF\s*\(.*\)\s*THEN\b", line):
            stack.append([
                bool(re.match(
                    r"^\s*IF\(\s*PRESENT\(\s*pUe\s*\)\s*\)\s*THEN", line)),
                False])
        elif re.match(r"^\s*ELSE\b", line) and stack:
            stack[-1][1] = True
        elif re.match(r"^\s*END\s*IF\b", line) and stack:
            stack.pop()
    require(not stack, f"{case}: unbalanced IF/ENDIF in dyn_adv_up3")
    if not guarded or unguarded:
        violations.append(
            f"dyn_adv_up3 writes the 3-D momentum RHS outside the "
            f"PRESENT(pUe) ELSE at lines {unguarded}; the dumped frame would "
            "then carry momentum advection")

    require(not violations,
            f"{case}: the dumped frame is NOT dyn_hpg alone -- "
            + "; ".join(violations))
    return {
        "ppsrc": str(ppsrc),
        "cpp_keys": keys,
        "namelist_selections": selections,
        "dynspg_ts_rhs_writes": spg_writes,
        "dyn_drg_init_intent": drg_intent,
        "hpg_sco_rhs_assignments": hpg_writes,
        "dyn_adv_up3_rhs_writes_in_present_pue_else": guarded,
        "dyn_adv_up3_rhs_writes_unguarded": unguarded,
    }


def bit_compare(oracle: np.ndarray, candidate: np.ndarray,
                mask: np.ndarray) -> dict:
    """Bit-level companion to the shared value scorer.

    ``score()`` reports ``exact`` from ``np.array_equal``, which is VALUE
    equality: ``-0.0 == 0.0``.  A campaign whose bar is bit equality cannot
    take that as its headline, so this counts raw bit patterns and separates
    the signed-zero population, which is the whole of the difference here.
    """
    a = np.asarray(candidate, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    b = np.asarray(oracle, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    unequal = a.view(np.uint64) != b.view(np.uint64)
    signed_zero = (a == 0.0) & (b == 0.0) & (np.signbit(a) != np.signbit(b))
    # NEMO's dyn_vor adds a term that is exactly 0.0 at a rest start, and
    # IEEE 754 gives -0.0 + 0.0 = +0.0, so NEMO carries no negative zero here
    # and legoESM does.  Adding that same zero is the discriminating check.
    healed = int(((a + 0.0).view(np.uint64) != b.view(np.uint64)).sum())
    return {
        "n": int(a.size),
        "bit_unequal": int(unequal.sum()),
        "value_unequal": int((a != b).sum()),
        "bit_unequal_that_are_signed_zero_only": int(signed_zero.sum()),
        "bit_unequal_after_adding_nemo_zero_vorticity_term": healed,
        "negative_zeros_candidate": int((np.signbit(a) & (a == 0.0)).sum()),
        "negative_zeros_oracle": int((np.signbit(b) & (b == 0.0)).sum()),
    }

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

    allow_dirty_stamps(allow_dirty)

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

    # Precondition Z: the frame's ISOLATION, proved against the code the card
    # compiles rather than the shipped source (round 26 read the latter and
    # mislabelled the frame a composite).
    isolation = assert_ppsrc_isolation(case)
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

    # Precondition B0: dyn_ldf is called UNCONDITIONALLY at stp2d.F90:129 and
    # is inert on these cards only because their namelist sets
    # ln_dynldf_OFF = .true.  That is a namelist fact, not a rest-state fact,
    # so it is asserted rather than assumed.
    lv = cfg.lateral_viscosity
    ldf_coefficients = {
        name: float(getattr(lv, name))
        for name in ("A_h", "A_h_merid", "A_h_floor", "B_h", "B_h_barotropic",
                     "C_smag", "C_smag_lap", "C_leith")
    }
    require(all(value == 0.0 for value in ldf_coefficients.values())
            and cfg.lateral_friction_scheme == "none",
            f"{case}: NEMO sets ln_dynldf_OFF=.true., but the card carries "
            f"lateral momentum diffusion {ldf_coefficients} / "
            f"lateral_friction_scheme={cfg.lateral_friction_scheme!r}; the "
            "scored frame would then hold a dyn_ldf contribution NEMO's does "
            "not")

    # Precondition B: NEMO's own entry velocity is IDENTICALLY zero, so the
    # dumped frame carries no dyn_vor contribution and the row cannot pass by
    # cancellation against a nonzero one.
    rest = {
        "oracle_entry_abs_max_u": float(np.max(np.abs(entry["u"]))),
        "oracle_entry_abs_max_v": float(np.max(np.abs(entry["v"]))),
    }
    require(rest["oracle_entry_abs_max_u"] == 0.0
            and rest["oracle_entry_abs_max_v"] == 0.0,
            f"{case}: NEMO's kt=1 entry is not at rest: {rest}; dyn_vor would "
            "then contribute and this row's composition statement is void")

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
        row["frame"] = "instantaneous_post_stp2d_momentum_rhs"
        row["staggering_and_reduction"] = (
            "NEMO puu(:,:,:,Nrhs) after stp_2D (MY_SRC/stprk3.F90:206, before "
            "stage 1) and legoESM's momentum tendency are both instantaneous "
            "3-D C-grid face accelerations in m/s^2 at the same wet faces; "
            "elementwise L-infinity, no vertical, substep or time reduction. "
            "At this rest start the frame is dyn_hpg ALONE: see "
            "precondition_ppsrc_isolation")
        return row

    rows = [_relabel(score(f"{case}.kt1.stp2d.momentum_rhs.u",
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
        rows.append(_relabel(score(f"{case}.kt1.stp2d.momentum_rhs.v",
                                   rhs["v"][..., :nlev], candidate_v, v_mask)))
        v_disposition = "SCORED"
    else:
        v_disposition = (
            "WAIVED_STRUCTURALLY_ABSENT: this card has one wet j row, so it "
            "has no wet V face; a V row would compare masked zeros")

    bits = bit_compare(rhs["u"][..., :nlev], candidate_u, masks["u"])
    exact = all(row["exact"] for row in rows)
    # The bar is BITS, and until round 28 this gate published
    # bit_exact_given_nemo_inputs: false while exiting 0 -- so a caller that
    # only checked the exit status read a bit-DEBT row as at the bar.  The
    # value-exact-but-bit-unequal case now gets its own status VALUE-AT-BAR,
    # which main() maps to a NONZERO exit exactly like DEBT.  Only bit
    # equality exits 0.
    if not all(row["status"] == "AT-BAR" for row in rows):
        status = "DEBT"
    elif exact and not plant and bits["bit_unequal"] == 0:
        status = "AT-BAR"
    else:
        status = "VALUE-AT-BAR"
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-rule12-hpg-eligibility-v1",
        "case": case,
        "status": status,
        # VALUE-exact is what score() measures; the bar is BITS, so both are
        # published and the headline is the weaker of the two.
        "value_exact_given_nemo_inputs": bool(exact and not plant),
        "bit_exact_given_nemo_inputs": bool(
            exact and not plant and bits["bit_unequal"] == 0),
        "bit_comparison": bits,
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
        "precondition_lateral_viscosity_off": ldf_coefficients,
        "precondition_changed_operator_executed": calls,
        "v_component": v_disposition,
        "v_wet_faces": v_wet,
        "nemo_source": [
            "MY_SRC/stprk3.F90:204,206 (dump is AFTER stp_2D, BEFORE stage 1)",
            "stp2d.F90:128 dyn_hpg, :131 dyn_ldf (ln_dynldf_OFF), :146 "
            "dyn_vor, :172 dyn_adv_up3 (pUe present), :196 dyn_drg_init, "
            ":281 dyn_spg_ts",
            "dynspg_ts.F90:344-345 depth-mean removal is inside the #else at "
            ":303 and :938-975 barotropic add-back inside the #else at :910; "
            "both are the MLF arm and neither compiles under key_RK3",
            "dynhpg.F90:359,383 (nemo_sco acceleration into the RHS)",
            "domzgr_substitute.h90:139,145 (key_qco gdept_z0)",
        ],
        "precondition_ppsrc_isolation": isolation,
        "frame_composition": (
            "dyn_hpg PLUS dyn_vor's exactly-zero accumulation. hpg_sco "
            "ASSIGNS over ntsi..ntei x jk=1..jpkm1 -- which IS the scored set "
            "(nn_hls=2, so the record reader's [2:-2] trim is the owned "
            "interior, and the reader takes levels :jpkm1) -- so nothing "
            "before stp2d.F90:128 reaches the comparison; dyn_ldf is inert by "
            "namelist (ln_dynldf_OFF); dyn_adv_up3 with pUe writes only the "
            "2-D RHS; dyn_drg_init takes puu/pvv INTENT(in); and the ppsrc "
            "dynspg_ts.f90 this card compiles has ZERO puu/pvv assignments. "
            "dyn_vor is NOT invisible: it adds exactly 0.0, and IEEE "
            "-0.0 + 0.0 = +0.0, so it normalises NEMO's negative zeros away "
            "while legoESM keeps its own. That is VALUE-invisible and "
            "BIT-visible; see bit_comparison. The earlier 'dyn_hpg ALONE' "
            "wording is retracted."
        ),
        "rows": rows,
        "planted_control": plant,
    }
    if plant:
        require(status == "DEBT" and rows[0]["absolute_max"] >= 0.5,
                "planted eligibility violation did not fire")
    return report


@scoped_allow_dirty
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(ROOTS))
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument(
        "--plant-ppsrc", metavar="CONFIG",
        help="run ONLY the ppsrc isolation assertion against another NEMO "
             "configuration (e.g. cfgs/DINO, which compiles the MLF arm); it "
             "MUST fail, which is what proves the assertion is not vacuous")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stamp a dirty tree (never for a recorded row)")
    args = parser.parse_args(argv)
    if args.plant_ppsrc:
        try:
            report = assert_ppsrc_isolation(args.case, config=args.plant_ppsrc)
        except GateError as error:
            print(json.dumps({"status": "PLANT-FIRED", "config":
                              args.plant_ppsrc, "error": str(error)}, indent=2))
            return 1
        print(json.dumps({"status": "PLANT-DID-NOT-FIRE",
                          "config": args.plant_ppsrc, "report": report},
                         indent=2, sort_keys=True))
        return 2
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

#!/usr/bin/env python
"""Every operand legoESM's tracer matrix reads, captured LIVE and scored.

Round 37 exonerated both halves of GYRE's kt=1 stage-3 tracer residual,
``3.1956659540810506e-11`` K on T: given NEMO's own dumped matrix and
right-hand side legoESM's ordered solve is bit-exact (0 of 21120), and
substituting NEMO's own right-hand side into the model moves the output only
``5.786374251651969e-16`` K.  What was left unmeasured is the MATRIX legoESM
BUILDS -- the operands at its own call site.

This gate captures them THROUGH THE MODEL'S OWN PATH.  It does not rebuild
them, and it does not read them from a config: it wraps the two functions the
step calls and emits ``jax.debug.callback`` on their arguments, so the arrays
scored here are the arrays the compiled step handed to the matrix.

    ocean_model_latlon_cgrid.py:10237 calls
    ``implicit_vertical_diffusion_ocean_tracer_pair_dispatch`` under a
    FUNCTION-SCOPE import (:10228), so a module-attribute patch is picked up
    at trace time.  ``K``, ``dz_after``, ``e3w_now``, ``wet``, ``dt``, the two
    content right-hand sides and ``implicit_w`` are exactly what
    ``nemo_tracer_tridiagonal`` (``implicit_solver.py:503``) consumes.

    ``LatLonCGridOceanModel._apply_implicit_vertical_mixing`` receives
    ``K33_iso``, the isoneutral vertical fold legoESM adds to the closure's
    diffusivity at :9797-9802.  NEMO carries no such term in ``avt``; its
    equivalent is ``ah_wslp2``, added at ``trazdf.f90:172-174``.

THREE CONTROLS, because a capture is not self-certifying (an independent
claim review broke the first version of each):

1. THE CALL COUNT.  ``_step_jitted`` is ``jax.jit(..., static_argnums=(0,))``,
   so its cache key is the MODEL INSTANCE.  A patch installed after an
   equivalent instance has compiled would be shadowed and the capture would
   silently return nothing.  Each arm builds a FRESH card and model, and the
   gate REFUSES a run in which either capture point did not fire exactly once.
2. INERTNESS.  The model's returned T and S must be bit-identical with and
   without the capture.  A debug callback gives an operand a second consumer,
   and round 37 measured that a second use is exactly what stops XLA
   contracting a product into a multiply-add.
3. THE ROUND TRIP.  Inertness alone cannot see a capture that reads a
   differently-lowered COPY of an operand while the solve keeps the original.
   So a third run REPLACES the dispatch's operands by the captured host
   arrays; bit-identical T and S prove the captured values ARE the consumed
   values.

WHAT THIS GATE CANNOT SEE, declared rather than discovered: it scores the
matrix's operands, not the right-hand side's (round 37 measured and exonerated
that), and not anything downstream of the solve.  The ``dz`` row's dry cells
are a KNOWN convention gap, not a measurement -- legoESM's thickness is
exactly 0.0 below the seafloor where NEMO's ``e3t_3d`` is the positive
reference thickness -- so every row is scored on WET cells and the dry-cell
census is reported beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from nemo_testcase_l2_gyre_phase3_gate import (  # noqa: E402
    CASE, _surface_forcings, read_stage, require, sha256,
)
from nemo_testcase_l2_gyre_round35_trazdf_matrix import (  # noqa: E402
    BAR, EXPECTED_DOMAIN, RecordError, _box, bit_row, read_trazdf_matrix,
)

RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
              "round37_oracle_trazdf_matrix/"
              "oracle_trazdf_matrix_kt00000001.bin")
# NEMO's own ssha, i.e. ssh(Kaa) at stage 3 (stprk3_stg.F90:224 recovers the
# stage-1 barotropic value).  Byte-identical in the round-29 and round-37
# roots, so the dz_owner arm reads it beside its own record.
STAGE3_RECORD = "oracle_stage_kt00000001_s3.bin"
# Every plantable operand, and the ROW its plant must move.  One map rather
# than two lists, so a renamed row cannot leave a plant pointing at nothing.
PLANT_ROW = {
    "K": "operand.K",
    "K33": "operand.K33_fold",
    "dz": "operand.dz_after",
    "e3w": "operand.e3w_now",
    "wet": "operand.wet",
}
PLANT_OPERANDS = tuple(PLANT_ROW)


def _oracle(rec: dict) -> dict[str, np.ndarray]:
    """NEMO's own arrays on the model's (lat, lon, level) axis order.

    ``_box`` returns the interior in NEMO's ``(i, j, k)``; the model's field
    is ``(lat, lon, level)``, i.e. ``(j, i, k)``.  The transpose is the SAME
    mapping ``nemo_testcase_l2_gyre_phase3_gate._xyz`` applies to every other
    row of this campaign -- reshape Fortran-order, take ``[2:-2, 2:-2]``,
    transpose ``(1, 0, 2)``.

    Level alignment: legoESM cell ``k`` is NEMO ``jk = k+1``, and legoESM
    interior face ``f`` is NEMO ``jk = f+2``, so the face arrays are the
    record's ``1:jpkm1`` slice.  That slice omits ``jk = jpk``, which NEMO's
    bottom ``zws`` does read (``trazdf.f90:444``); ``avt`` and ``ah_wslp2``
    are measured EXACTLY 0.0 there, reported below as ``bottom_face_zero``,
    so the omission is inert BY MEASUREMENT and not by argument.
    """
    jpkm1 = rec["header"]["jpkm1"]
    t = lambda a: np.ascontiguousarray(a.transpose(1, 0, 2))
    return {
        "K": t(_box(rec, "zwt_mix")[:, :, 1:jpkm1]),
        "avt": t(_box(rec, "avt")[:, :, 1:jpkm1]),
        "K33": t(_box(rec, "ah_wslp2")[:, :, 1:jpkm1]),
        "dz": t(_box(rec, "e3t_Kaa", jpkm1)),
        "e3w": t(_box(rec, "e3w_Kmm")[:, :, 1:jpkm1]),
        "wet": t(_box(rec, "tmask", jpkm1)),
        "zwi": t(_box(rec, "zwi")),
        "zwd": t(_box(rec, "zwd")),
        "zws": t(_box(rec, "zws")),
        "rDt": float(rec["arrays"]["rDt"]),
        "bottom_face_zero": {
            n: float(np.abs(_box(rec, n)[:, :, rec["header"]["jpk"] - 1]).max())
            for n in ("avt", "ah_wslp2")},
    }


def _model_step(*, capture: bool, k_override=None, operand_override=None,
                verify_against=None):
    """One production kt=1 GYRE step, fp64, production JIT.

    ``k_override`` replaces the matrix diffusivity the dispatch receives --
    the one-variable causal arm.  ``verify_against`` re-enters the step with
    the previously captured host arrays and compares them to the LIVE operands
    INSIDE the graph, which is control 3.  ``operand_override`` substitutes
    them instead; it is kept because it is what MEASURED that an output-level
    round trip is confounded (see ``run``).
    """
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    import legoesm.ocean.physics.vertical_mixing as vmix
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "certification requires production JIT; "
            "JAX_DISABLE_JIT is forbidden")

    caught: dict[str, list] = {}
    real_dispatch = vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch
    real_vmix = LatLonCGridOceanModel._apply_implicit_vertical_mixing

    def sink(name):
        return lambda value: caught.setdefault(name, []).append(
            np.array(value, dtype=np.float64))

    def wrapped_dispatch(f1, f2, c1, c2, K, dz_after, e3w_now, dt, wet, **kw):
        # implicit_w is a MATRIX operand (implicit_solver.py:557-566); GYRE's
        # oracle pins ln_zad_Aimp = F, so anything but None here means the
        # matrix carries a term NEMO's does not and no operand row could see
        # it.  A hard failure, never a row.
        require(kw.get("implicit_w") is None,
                "the tracer matrix received an implicit_w; GYRE's record "
                "pins ln_zad_Aimp = F")
        caught.setdefault("dtype", []).append(str(K.dtype))
        for label, value in (("K", K), ("dz", dz_after), ("e3w", e3w_now),
                             ("content_T", c1), ("content_S", c2)):
            jax.debug.callback(sink(label), value)
        jax.debug.callback(sink("wet"), jnp.asarray(wet, dtype=K.dtype))
        jax.debug.callback(sink("dt"), jnp.asarray(dt, dtype=K.dtype))
        if verify_against is not None:
            # CONTROL 3, IN THE GRAPH.  Comparing the captured array to the
            # LIVE one here proves the capture read the value the solve
            # consumed.  Doing it by SUBSTITUTING and comparing the step's
            # output does not: a host array lowers as a CONSTANT, which
            # changes fusion downstream, so that arm moves bits even when the
            # values are identical -- measured, and recorded in the report.
            for label, value in (("K", K), ("dz", dz_after),
                                 ("e3w", e3w_now),
                                 ("wet", jnp.asarray(wet, dtype=K.dtype))):
                ref = jnp.asarray(verify_against[label], dtype=K.dtype)
                jax.debug.callback(
                    sink(f"verify_{label}"),
                    jnp.sum(jnp.asarray(
                        jax.lax.bitcast_convert_type(value, jnp.uint64)
                        != jax.lax.bitcast_convert_type(ref, jnp.uint64),
                        dtype=jnp.float64)))
        if operand_override is not None:
            K = jnp.asarray(operand_override["K"], dtype=K.dtype)
            dz_after = jnp.asarray(operand_override["dz"], dtype=K.dtype)
            e3w_now = jnp.asarray(operand_override["e3w"], dtype=K.dtype)
            wet = jnp.asarray(operand_override["wet"], dtype=K.dtype) > 0.0
        elif k_override is not None:
            K = jnp.asarray(k_override, dtype=K.dtype)
        return real_dispatch(f1, f2, c1, c2, K, dz_after, e3w_now, dt, wet,
                             **kw)

    def wrapped_vmix(self, state, dt, surface_forcing, K_v_phys=None,
                     A_v_phys=None, K33_iso=None, **kw):
        if K33_iso is not None and not kw.get("return_K_profiles", False):
            jax.debug.callback(sink("K33"), K33_iso)
        return real_vmix(self, state, dt, surface_forcing, K_v_phys=K_v_phys,
                         A_v_phys=A_v_phys, K33_iso=K33_iso, **kw)

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    if capture:
        vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch = (
            wrapped_dispatch)
        LatLonCGridOceanModel._apply_implicit_vertical_mixing = wrapped_vmix
    try:
        freshwater, surface = _surface_forcings(
            card, card.recipe.initial_state, 1)
        state = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
        ).step(card.recipe.initial_state, dt=card.dt_s,
               freshwater=freshwater, surface_forcing=surface)
        out = (np.asarray(state.T.data), np.asarray(state.S.data),
               np.asarray(state.eta.data), np.asarray(state.H_bathy.data))
    finally:
        vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch = (
            real_dispatch)
        LatLonCGridOceanModel._apply_implicit_vertical_mixing = real_vmix
    if capture:
        # CONTROL 1 -- the jit cache key is the model instance, so a capture
        # that did not fire is indistinguishable from one that measured zero.
        for name in ("K", "K33", "dz", "e3w", "wet", "dt"):
            require(len(caught.get(name, ())) == 1,
                    f"capture point for {name!r} fired "
                    f"{len(caught.get(name, ()))} times, expected exactly 1")
        require(caught["dtype"][0] == "float64",
                f"the matrix works in {caught['dtype'][0]}, not float64")
    return out, {k: v[0] for k, v in caught.items() if k != "dtype"}


def _stage3_residual(k_override, *, oracle_root: Path, npz: Path) -> dict:
    """The kt=1 stage-3 output residual, with ONE operand substituted.

    This is the CAUSAL arm, and it is the mirror of round 37's right-hand-side
    substitution: rather than asking how big an operand difference is, it asks
    how much of the output the difference OWNS.  ``k_override`` replaces the
    matrix diffusivity the dispatch receives -- one variable, at the exact
    array the matrix reads -- and the stage-3 completion gate's own faithful
    measurement then scores the model against NEMO's own stage-3 dump.

    Reusing that gate rather than rebuilding the score is deliberate: the
    residual it prints is the number rounds 34 and 37 quote, so a movement
    measured here is a movement in THAT number and not in a lookalike.
    """
    import jax.numpy as jnp
    import legoesm.ocean.physics.vertical_mixing as vmix
    import nemo_testcase_l2_gyre_stage3_completion_gate as stage3

    calls = []
    real_dispatch = vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch

    def wrapped(f1, f2, c1, c2, K, dz_after, e3w_now, dt, wet, **kw):
        calls.append(1)
        if k_override is not None:
            K = jnp.asarray(k_override, dtype=K.dtype)
        return real_dispatch(f1, f2, c1, c2, K, dz_after, e3w_now, dt, wet,
                             **kw)

    vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch = wrapped
    try:
        report = stage3.run("faithful", npz, None, False, oracle_root)
    finally:
        vmix.implicit_vertical_diffusion_ocean_tracer_pair_dispatch = (
            real_dispatch)
    require(len(calls) == 1,
            f"the stage-3 gate called the tracer dispatch {len(calls)} times, "
            "expected exactly 1; a substitution that did not trace would "
            "report the faithful number and read as 'no movement'")

    def _find(node):
        if isinstance(node, dict):
            if "absolute_max" in node and "name" in node:
                yield node
            else:
                for value in node.values():
                    yield from _find(value)
        elif isinstance(node, list):
            for value in node:
                yield from _find(value)

    out = {}
    for row in _find(report):
        if row["name"].endswith("kt1.stage3.faithful.T"):
            out["T"] = float(row["absolute_max"])
        elif row["name"].endswith("kt1.stage3.faithful.S"):
            out["S"] = float(row["absolute_max"])
    require(set(out) == {"T", "S"},
            "the stage-3 gate did not report a faithful T and S row")
    return out


KT2_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                  "round38_oracle_trazdf_kt2/"
                  "oracle_trazdf_matrix_kt00000002.bin")


def _fold_removed(live: dict) -> np.ndarray:
    """legoESM's matrix diffusivity with legoESM's own fold subtracted out.

    The model builds ``K_v_cell = (K_v + K33) * interface_mask``
    (``ocean_model_latlon_cgrid.py``), so a plain ``K - K33`` is ``-K33`` at
    every MASKED interface -- a NEGATIVE diffusivity fed into the substitution
    solve.  An independent diff review found that; the masked interfaces keep
    the value the model gave them.
    """
    wet = np.asarray(live["wet"]) > 0.0
    face = wet[..., 1:] & wet[..., :-1]
    return np.where(face, live["K"] - live["K33"], live["K"])


def run_kt2_given_inputs(record: Path = KT2_RECORD,
                         *, mld_criterion: str | None = None) -> dict:
    """legoESM's isoneutral fold, GIVEN NEMO'S OWN BEFORE STATE, at kt = 2.

    ROUND 39, and it is the arm the round-38 acquisition exists for.  At
    ``kt = nit000`` NEMO's ``ah_wslp2`` is IDENTICALLY 0.0, so that record can
    only ask whether a candidate's fold is also exactly zero -- it cannot
    discriminate any TRANSCRIPTION of the slope formula.  The kt = 2 record
    can: its ``ah_wslp2`` is non-zero because step 1 created the horizontal
    structure its before state carries.

    WHAT THIS SCORES, and what it does NOT.  Round 39 moved WHICH STATE the
    slopes are built from; it did not touch the slope FORMULA.  So a residual
    here belongs to the slope transcription -- a DIFFERENT owner -- and is
    reported as such.  No value was predicted for this row before it was
    measured, deliberately (preregistration PR7).

    The before state is NEMO's own: ``T_Kbb_in``/``S_Kbb_in`` and the ssh
    implied by ``r3t_Kbb``, since ``r3t = ssh/ht_0`` (``domqco.F90:160``,
    ``dom_qco_r3c``) so ``ssh = r3t * H``.  Everything else -- grid, vertical
    coordinate, masks, GM/Redi configuration, timestep -- is the card's.
    """
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        compute_isoneutral_K33_latlon, static_kappa_redi_override)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    rec = read_trazdf_matrix(record, expect_kt=2)
    oracle = _oracle(rec)
    t = lambda a: np.ascontiguousarray(a.transpose(1, 0, 2))
    jpkm1 = rec["header"]["jpkm1"]
    T_bb = t(_box(rec, "T_Kbb_in", jpkm1))
    S_bb = t(_box(rec, "S_Kbb_in", jpkm1))
    r3t_bb = np.ascontiguousarray(_box(rec, "r3t_Kbb").transpose(1, 0))

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    init = card.recipe.initial_state
    H = np.asarray(init.H_bathy.data, dtype=np.float64)
    require(T_bb.shape == np.asarray(init.T.data).shape,
            f"the record's Kbb tracer box {T_bb.shape} is not the model's "
            f"{np.asarray(init.T.data).shape}")
    require(r3t_bb.shape == H.shape,
            f"the record's r3t box {r3t_bb.shape} is not the model's "
            f"{H.shape}")
    eta_bb = r3t_bb * H

    # THE MODEL'S OWN kappa OVERRIDES, not omitted and not assumed: the arm
    # must call the function the way ocean_model_latlon_cgrid.py:7323 calls
    # it.  Rule 10 -- what they RESOLVE to on this card is reported, not read
    # off the flag.
    kappa_t, kappa_v = static_kappa_redi_override(cfg.gm_redi, card.recipe.grid)
    # ROUND 40, item 2, ONE VARIABLE.  NEMO's nmln/hmlp come from zdf_mxl's
    # N2-INTEGRAL criterion (zdfmxl.F90:95-104), run on the before state at
    # stprk3.F90:165; this card resolves legoESM's 'rho_c', a
    # potential-density-difference test referenced to a CELL CENTRE.  The
    # exact NEMO criterion already exists behind mld_criterion='n2_integral'.
    # Nothing is selected here -- the arm reports what the swap does and the
    # card's default is untouched.
    gm_redi = cfg.gm_redi
    if mld_criterion is not None:
        gm_redi = gm_redi._replace(mld_criterion=mld_criterion)
    K33 = np.asarray(compute_isoneutral_K33_latlon(
        jnp.asarray(T_bb), jnp.asarray(S_bb), jnp.asarray(eta_bb),
        jnp.asarray(H), card.recipe.grid, card.recipe.z_coord, gm_redi,
        eos=cfg.eos, eos_linear=cfg.eos_linear,
        mask=jnp.asarray(init.land_mask.data),
        rho_0=cfg.constants.rho_0, g=cfg.constants.g,
        native_slope_eta=jnp.asarray(eta_bb),
        kappa_redi_override=kappa_t, kappa_redi_v_override=kappa_v,
        u_mask=jnp.asarray(init.u_mask.data),
        v_mask=jnp.asarray(init.v_mask.data), dt=card.dt_s,
        eos_depth=getattr(cfg, "eos_depth", "insitu")), dtype=np.float64)

    wet_cell = oracle["wet"] > 0.0
    # THE GATE'S OWN FACE MASK, not a lookalike: a face is wet only when the
    # cells on BOTH sides are.  The two agree on GYRE (0 of 17400 differ) and
    # diverge over topography, and this arm is meant to run on other cards.
    wet_face = wet_cell[..., 1:] & wet_cell[..., :-1]
    require(wet_face.shape == K33.shape,
            f"the wet-face mask {wet_face.shape} is not the fold's "
            f"{K33.shape}")
    row = _wet_row("kt2_given_inputs.K33_fold", oracle["K33"], K33, wet_face)
    # THE RECORD'S OWN PRECONDITION.  A kt=2 record whose ah_wslp2 is still
    # identically zero cannot discriminate anything, and a row scored against
    # it would read AT-BAR for a transcription that is arbitrarily wrong.
    nemo_max = float(np.abs(oracle["K33"]).max())
    return {
        "worktree": worktree_stamp(),
        "record": str(record),
        "kt": int(rec["header"]["kt"]),
        "mld_criterion_resolved": gm_redi.mld_criterion,
        "mld_criterion_is_the_card_default": mld_criterion is None,
        "discriminating": nemo_max > 0.0,
        "nemo_ah_wslp2_absmax": nemo_max,
        "lego_K33_absmax": float(np.abs(K33)[wet_face].max()),
        "before_state_wet_per_level_ptp_max": float(max(
            float(np.ptp(T_bb[..., k][wet_cell[..., k]]))
            if wet_cell[..., k].any() else 0.0
            for k in range(wet_cell.shape[-1]))),
        "row": row,
        "scored_wet_faces": int(wet_face.sum()),
        "kappa_redi_override_resolved": (
            "None" if kappa_t is None else str(np.asarray(kappa_t).shape)),
        "kappa_redi_v_override_resolved": (
            "None" if kappa_v is None else str(np.asarray(kappa_v).shape)),
        "owner_if_debt": ("the isoneutral SLOPE TRANSCRIPTION, not round 39's "
                          "placement: this arm feeds NEMO's own before state "
                          "to both sides, so any residual is the formula's"),
    }


def _dz_owner_verdict(rows: list[dict]) -> str:
    """The verdict is READ OFF the rows, never asserted beside them.

    A hardcoded "bit-exact" sentence survives a row going red, which is
    exactly the failure mode this campaign's gates exist to prevent.
    """
    by_name = {row["name"]: row for row in rows}

    def clean(name: str) -> bool:
        row = by_name.get(name)
        return bool(row) and row.get("bit_unequal") == 0

    ref = clean("dz_owner.reference_thickness")
    given = clean("dz_owner.stretch_given_nemo_ssh")
    path = clean("dz_owner.stretch_model_path")
    after = clean("dz_owner.dz_after")
    if not ref:
        return ("the REFERENCE thickness itself differs; nothing downstream "
                "of it can be attributed until that row is clean")
    if not given:
        return ("the stretch STATEMENT differs given NEMO's own ssh, so the "
                "owner is the statement and not the ssh")
    if not path:
        return ("the statement is bit-exact given NEMO's own ssh, so what "
                "remains on the model path is owned by legoESM's own "
                "ssh(Kaa) out of the stage-3 update")
    return ("reference thickness, stretch statement and the model path are "
            "all bit-exact"
            + ("; dz_after is too" if after else
               "; dz_after is NOT, so its owner is downstream of the stretch"))


def run_dz_owner(record: Path = RECORD) -> dict:
    """WHO OWNS the ``dz_after`` residual: the reference thickness, or the ssh?

    ``e3t(i,j,k,Kaa) = e3t_0(i,j,k) * (1 + r3t(i,j,Kaa))``
    (``domzgr_substitute.h90:139``) and ``r3t = ssh/ht_0``
    (``domqco.F90:160``, ``dom_qco_r3c``).  So a difference in ``dz_after`` is
    owned by exactly one of two things, and this arm says which:

    * the REFERENCE thickness ``e3t_0``, which is static geometry; or
    * the STRETCH, i.e. legoESM's own ``ssh(Kaa)`` out of the stage-3 update.

    The record's own consistency is checked first rather than assumed: NEMO's
    ``e3t_Kaa`` must equal ``e3t_0 * (1 + r3t_Kaa)`` on every level this
    campaign scores.
    """
    rec = read_trazdf_matrix(record)
    jpkm1 = rec["header"]["jpkm1"]
    t = lambda a: np.ascontiguousarray(a.transpose(1, 0, 2))
    e3t0 = t(_box(rec, "e3t_0", jpkm1))
    e3t_aa = t(_box(rec, "e3t_Kaa", jpkm1))
    r3t_aa = np.ascontiguousarray(_box(rec, "r3t_Kaa").transpose(1, 0))
    wet = t(_box(rec, "tmask", jpkm1)) > 0.0

    rebuilt = e3t0 * (1.0 + r3t_aa[..., None])
    record_consistent = int(np.count_nonzero(
        rebuilt.view(np.uint64) != e3t_aa.view(np.uint64)))

    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    (_, _, eta_aa, H), live = _model_step(capture=True)
    card = build_nemo_testcase_card(CASE)
    zc = card.recipe.z_coord
    ref = np.asarray(getattr(zc, "nemo_e3t_0"), dtype=np.float64)
    rows = [_wet_row("dz_owner.reference_thickness", e3t0, ref, wet)]
    # ROUND 40, Rule 10.  The row that used to sit here computed
    # ``eta/H`` IN THE GATE -- a division the model does not perform -- and
    # compared it against NEMO's ``r3t``, so it scored the gate's own
    # arithmetic on top of legoESM's ssh.  Both statements are scored
    # separately now, and each says what it owns.
    import jax.numpy as jnp

    from legoesm.ocean.vertical import compute_ocean_jacobian
    stage3 = read_stage(record.parent / STAGE3_RECORD, 3)
    nemo_ssha = np.asarray(stage3["ssh"], dtype=np.float64)
    one_plus_r3t = 1.0 + r3t_aa
    # (a) THE STATEMENT, given NEMO's own ssh: legoESM's shared stretch helper
    #     fed NEMO's ssh(Kaa) must reproduce NEMO's own 1 + r3t(Kaa), which is
    #     ``pssh * r1_ht_0`` (domqco.F90:209) with r1_ht_0 = 1/ht_0 on a wet
    #     column (domain.F90:158).
    lego_given = np.asarray(compute_ocean_jacobian(
        jnp.asarray(nemo_ssha), jnp.asarray(H), card.recipe.z_coord),
        dtype=np.float64)
    rows.append(_wet_row("dz_owner.stretch_given_nemo_ssh",
                         one_plus_r3t, lego_given, wet[..., 0]))
    # (b) THE SAME STATEMENT on the model's own path, which additionally
    #     carries legoESM's own ssh(Kaa) out of the stage-3 update.
    lego_path = np.asarray(compute_ocean_jacobian(
        jnp.asarray(eta_aa), jnp.asarray(H), card.recipe.z_coord),
        dtype=np.float64)
    rows.append(_wet_row("dz_owner.stretch_model_path",
                         one_plus_r3t, lego_path, wet[..., 0]))
    rows.append(_wet_row("dz_owner.dz_after", e3t_aa,
                         np.asarray(live["dz"]), wet))
    return {
        "worktree": worktree_stamp(),
        "record": str(record),
        "stage3_record": str(record.parent / STAGE3_RECORD),
        "stage3_record_sha256": sha256(record.parent / STAGE3_RECORD),
        "record_self_consistent_cells_unequal": record_consistent,
        "rows": rows,
        "owner": _dz_owner_verdict(rows),
    }


def run_knob_redundancy() -> dict:
    """Is ``slope_prd_geometry_stage='before_step'`` redundant with the fix?

    An independent claim review named this knob a BLOCKER: it already routes
    the slope density's tracers and Jacobian to the step-entry state on a
    forward-Euler card, so landing the placement fix could leave two
    mechanisms for one move.  The claim that they are REDUNDANT rather than
    competing is decided here by measurement, not by reading.
    """
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as M
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))

    def one(stage):
        card = build_nemo_testcase_card(CASE)
        cfg = card.recipe.model_config._replace(
            freshwater_closure="real_freshwater", fix_eta_drift=True)
        if stage is not None:
            cfg = cfg._replace(
                gm_redi=cfg.gm_redi._replace(slope_prd_geometry_stage=stage))
        freshwater, surface = _surface_forcings(
            card, card.recipe.initial_state, 1)
        state = M.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
        ).step(card.recipe.initial_state, dt=card.dt_s,
               freshwater=freshwater, surface_forcing=surface)
        return (np.asarray(state.T.data), np.asarray(state.S.data),
                cfg.gm_redi.slope_prd_geometry_stage)

    base_T, base_S, base_stage = one(None)
    knob_T, knob_S, knob_stage = one("before_step")
    require(base_stage != knob_stage,
            "the two arms resolved the same knob value, so this measures "
            "nothing")
    moved = {
        "T": int(np.count_nonzero(
            base_T.view(np.uint64) != knob_T.view(np.uint64))),
        "S": int(np.count_nonzero(
            base_S.view(np.uint64) != knob_S.view(np.uint64))),
    }
    return {
        "worktree": worktree_stamp(),
        "default_stage": base_stage, "arm_stage": knob_stage,
        "cells": int(base_T.size),
        "bits_moved": moved,
        "redundant": moved["T"] == 0 and moved["S"] == 0,
    }


def _wet_row(name, oracle, candidate, wet) -> dict:
    """One operand row, on WET cells only, at the campaign's exact bar.

    ``max_relative`` divides by the ORACLE's own magnitude where that is
    non-zero and by 1.0 where it is not, so on a row whose oracle is
    identically zero -- ``ah_wslp2`` is exactly that -- the reported number is
    an ABSOLUTE difference wearing a relative name.  Said here rather than
    left for a reader to infer from a suspiciously round figure.

    ``largest_difference_cell`` is the argmax of the masked difference, NOT
    the first differing cell in any order; it is named for what it is.
    """
    mask = np.asarray(wet, dtype=bool)
    row = bit_row(name, np.asarray(oracle)[mask], np.asarray(candidate)[mask])
    diff = np.abs(np.asarray(oracle) - np.asarray(candidate))
    scale = np.where(np.abs(oracle) > 0.0, np.abs(oracle), 1.0)
    rel = np.where(mask, diff / scale, 0.0)
    row["max_relative"] = float(rel.max())
    if row["bit_unequal"]:
        flat = np.argmax(np.where(mask, diff, -1.0))
        # argmax, unravel_index and ravel are all C order, so the index and
        # the pair below name the SAME cell.
        row["largest_difference_cell"] = [int(i) for i in
                                          np.unravel_index(flat, diff.shape)]
        row["largest_difference_pair"] = [
            float(np.ascontiguousarray(oracle).ravel()[flat]),
            float(np.ascontiguousarray(candidate).ravel()[flat])]
    row["dry_cells_unequal"] = int(np.count_nonzero(
        (np.asarray(oracle).view(np.uint64)
         != np.asarray(candidate).view(np.uint64)) & ~mask))
    # THE CANDIDATE'S OWN BYTES, so a plant can be SEEN on a row that is
    # already saturated.  Measured: a one-ulp plant on the largest |K| cell
    # leaves (bit_unequal, absolute_max, status) untouched, because 17383 of
    # 17400 cells already differ and the perturbation is 1.7e-18 against an
    # absmax of 9.66e-13.  A control that cannot see its own plant proves
    # nothing, and the count-and-max signature could not see it.
    row["candidate_sha256"] = hashlib.sha256(
        np.ascontiguousarray(candidate)[mask].tobytes()).hexdigest()
    return row


def run_substitution(record: Path, *, oracle_root: Path, npz: Path) -> dict:
    """PR3 and PR4: how much of the stage-3 output does the matrix K own?

    Three arms, each differing from the faithful run in ONE array:

    ``faithful``        legoESM's own K.
    ``nemo_K``          NEMO's own ``zwt_mix``, i.e. its ``avt`` plus its
                        identically-zero ``ah_wslp2``.
    ``nemo_avt_lego_fold``  NEMO's own ``avt`` plus legoESM's LIVE isoneutral
                        fold.  The difference between this arm and ``nemo_K``
                        is the FOLD and nothing else; the difference between
                        it and ``faithful`` is the closure and nothing else.
    """
    rec = read_trazdf_matrix(record)
    oracle = _oracle(rec)
    _, live = _model_step(capture=True)
    arms = {
        "faithful": None,
        # THE NULL SUBSTITUTION.  legoESM's OWN captured K, fed back as a host
        # array.  Its values are bit-identical to the live ones (the operand
        # gate's in-graph identity check proves it), so any movement here is
        # the CONSTANT LOWERING and not the substitution -- it is this arm's
        # noise floor, and the two arms below are only readable against it.
        "null_substitution": live["K"],
        "nemo_K": oracle["K"],
        "nemo_avt_lego_fold": oracle["avt"] + live["K33"],
        # ROUND 39, and it is the arm that PREDICTS the placement fix.
        # ``nemo_K`` removes the fold AND the legoESM-vs-NEMO ``avt``
        # difference, so its residual is a ceiling for a change that removes
        # only the fold -- an independent claim review measured that using it
        # as the prediction would fire the falsifier on a change that is
        # exactly right, because one ulp on every wet ``avt`` is worth up to
        # 1.16e-14 K on its own.  This arm is legoESM's OWN ``K`` with
        # legoESM's OWN fold subtracted back out, which is exactly what the
        # model produces once the slopes are built on the before state.
        "lego_avt_no_fold": _fold_removed(live),
    }
    residual = {name: _stage3_residual(k, oracle_root=oracle_root, npz=npz)
                for name, k in arms.items()}
    base = residual["faithful"]
    moved = {
        name: {tag: abs(residual[name][tag] - base[tag]) for tag in ("T", "S")}
        for name in arms if name != "faithful"}
    fold_only = {tag: abs(residual["nemo_avt_lego_fold"][tag]
                          - residual["nemo_K"][tag]) for tag in ("T", "S")}
    closure_only = {tag: abs(residual["nemo_avt_lego_fold"][tag]
                             - base[tag]) for tag in ("T", "S")}
    noise_floor = {tag: abs(residual["null_substitution"][tag] - base[tag])
                   for tag in ("T", "S")}
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round38-substitution-v1",
        "case": CASE, "record": str(record),
        "record_sha256": sha256(record),
        "oracle_root": str(oracle_root),
        "residual": residual,
        "moved_from_faithful": moved,
        "fold_only": fold_only,
        "closure_only": closure_only,
        "substitution_noise_floor": noise_floor,
        "owned_fraction_T": (
            moved["nemo_K"]["T"] / base["T"] if base["T"] else None),
    }


def run_time_level_discriminator() -> dict:
    """WHY is legoESM's isoneutral fold non-zero where NEMO's is exactly 0?

    Two mechanisms could produce it and they must not be collapsed:

    (a) THE CALL SITE.  NEMO computes the neutral slopes ONCE PER STEP, on the
        BEFORE state, OUTSIDE the stage loop -- ``stprk3.f90:175-178``,
        ``CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )`` -- and
        ``traldf_iso.f90:154`` then fills ``ah_wslp2`` from those slopes, which
        ``trazdf.f90:410`` reads at EVERY stage.  legoESM recomputes its K33
        INSIDE each stage from the stage's own tracers:
        ``ocean_model_latlon_cgrid.py:6969`` sets ``_T_gm_in = T_mid`` on this
        path (``_ldf_state`` is None outside ``_nemo_mlf_step``) and ``:7278``
        hands it to ``compute_isoneutral_K33_latlon``.
    (b) THE ARITHMETIC.  legoESM's slope chain might leave a rounding residue
        where NEMO's leaves an exact zero.

    This arm discriminates them by asking legoESM's OWN K33 function for its
    value on the STEP-ENTRY state -- the same state NEMO's ``ldf_slp`` reads.
    It is a property of that function, not a re-derivation of the model's
    number, and it is labelled as such.
    """
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_isoneutral_K33_latlon)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    state, cfg = card.recipe.initial_state, card.recipe.model_config
    T, S = np.asarray(state.T.data), np.asarray(state.S.data)
    wet = (np.asarray(state.land_mask.data) > 0.5)[..., None] & np.asarray(
        card.recipe.z_coord.is_active)
    spread = {
        tag: float(max(
            (np.ptp(field[:, :, k][wet[:, :, k]]) if wet[:, :, k].any()
             else 0.0) for k in range(field.shape[-1])))
        for tag, field in (("T", T), ("S", S))}
    k33 = np.asarray(compute_isoneutral_K33_latlon(
        jnp.asarray(T), jnp.asarray(S), state.eta.data, state.H_bathy.data,
        card.recipe.grid, card.recipe.z_coord, cfg.gm_redi,
        eos=cfg.eos, eos_linear=cfg.eos_linear, mask=state.land_mask.data,
        rho_0=cfg.constants.rho_0, g=cfg.constants.g, dt=card.dt_s,
        eos_depth=getattr(cfg, "eos_depth", "insitu")))
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round38-time-level-v1",
        "case": CASE,
        # Rule 10: the RESOLVED config, printed, not the library default.  An
        # earlier reading of this round took cfg.physics.lateral_mixing.gm_redi
        # -- a different object carrying implicit_K33=False -- and would have
        # concluded the fold was not even active.
        "resolved_gm_redi": {
            f: getattr(cfg.gm_redi, f, None) for f in (
                "implicit_K33", "slope_positions", "slope_scheme",
                "slope_limit", "slope_density", "msc_stabilize",
                "nemo_mld_slope_ramp", "nemo_slope_shapiro")},
        "step_entry_wet_per_level_ptp": spread,
        "k33_from_step_entry_state": {
            "absolute_max": float(np.abs(k33).max()),
            "nonzero_cells": int(np.count_nonzero(k33)),
            "n": int(k33.size)},
        "owner": ("THE CALL SITE" if float(np.abs(k33).max()) == 0.0
                  else "THE ARITHMETIC (or both)"),
    }


def run(record: Path, *, plant: str | None = None) -> dict:
    import jax
    rec = read_trazdf_matrix(record)
    header = rec["header"]
    require((header["jpi"], header["jpj"], header["jpk"]) == EXPECTED_DOMAIN,
            f"{record}: domain {(header['jpi'], header['jpj'], header['jpk'])}"
            f" is not this card's {EXPECTED_DOMAIN}")
    oracle = _oracle(rec)

    baseline, _ = _model_step(capture=False)
    captured_out, live = _model_step(capture=True)
    inert = {
        name: int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
        for name, a, b in (("T", baseline[0], captured_out[0]),
                           ("S", baseline[1], captured_out[1]))}
    # ROUND 39: A BIT COUNT WITHOUT A MAGNITUDE IS NOT A REPORTABLE NUMBER.
    # This control went from 0 to 1 moved T cell when the isoneutral fold
    # stopped being added, and "one bit moved" says nothing about whether the
    # capture perturbed the model by an ulp or by a metre.  A debug callback
    # gives an operand a second consumer, which is exactly what stops XLA
    # contracting a product into a multiply-add (round 37), so a change of one
    # or two ulp is the EXPECTED size and anything larger is a different
    # defect.  Both are now printed and neither is a passing status.
    inert_size = {}
    for name, a, b in (("T", baseline[0], captured_out[0]),
                       ("S", baseline[1], captured_out[1])):
        diff = np.abs(np.asarray(a, dtype=np.float64)
                      - np.asarray(b, dtype=np.float64))
        flat = int(np.argmax(diff))
        inert_size[name] = {
            "max_abs": float(diff.max()),
            "cell": [int(v) for v in np.unravel_index(flat, diff.shape)],
            "ulps_at_that_cell": (
                float(diff.max() / np.spacing(abs(float(
                    np.asarray(a).ravel()[flat]))))
                if diff.max() > 0.0 else 0.0),
        }
    # CONTROL 3 -- THE ROUND TRIP.  Inertness alone cannot see a capture that
    # reads a differently-lowered COPY of an operand while the solve keeps the
    # original; feeding the captured HOST arrays back in and getting bit
    # identical tracers is what proves the captured values ARE the consumed
    # ones.
    _, verified = _model_step(capture=True, verify_against=live)
    round_trip = {name.removeprefix("verify_"): int(value)
                  for name, value in verified.items()
                  if name.startswith("verify_")}
    require(set(round_trip) == {"K", "dz", "e3w", "wet"},
            "the in-graph identity check did not run on every operand")
    # THE CONFOUND, MEASURED RATHER THAN ASSERTED.  Substituting the captured
    # host arrays and comparing the STEP'S OUTPUT moves bits even though the
    # values are identical, because a host array lowers as a constant.  That
    # number is reported so the weaker control cannot be mistaken for this
    # one.
    substituted_out, _ = _model_step(capture=True, operand_override=live)
    output_substitution = {
        name: int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
        for name, a, b in (("T", baseline[0], substituted_out[0]),
                           ("S", baseline[1], substituted_out[1]))}
    unplanted = None
    if plant:
        # CONTROL 4 -- one ulp on ONE captured operand must MOVE ITS OWN ROW.
        #
        # "the gate exits non-zero" is NOT this control: three of the operand
        # rows are DEBT unplanted, so a plant that moved nothing would exit
        # non-zero too and read as passing.  The planted operand's row is
        # therefore scored BOTH ways and the two must differ.
        unplanted = dict(live)
        live = dict(live)
        target = live[plant].copy()
        idx = np.unravel_index(int(np.argmax(np.abs(target))), target.shape)
        target[idx] = np.nextafter(target[idx], np.inf)
        live[plant] = target

    wet_face = (oracle["wet"][..., 1:] > 0.0) & (oracle["wet"][..., :-1] > 0.0)
    wet_cell = oracle["wet"] > 0.0
    rows = [
        _wet_row("operand.K", oracle["K"], live["K"], wet_face),
        _wet_row("operand.K33_fold", oracle["K33"], live["K33"], wet_face),
        _wet_row("operand.dz_after", oracle["dz"], live["dz"], wet_cell),
        _wet_row("operand.e3w_now", oracle["e3w"], live["e3w"], wet_face),
        _wet_row("operand.wet", oracle["wet"], live["wet"], wet_cell),
        _wet_row("operand.dt", np.full((1,), oracle["rDt"]),
                 np.asarray(live["dt"]).reshape(1), np.ones(1, bool)),
    ]
    # The assembled diagonals, from the LIVE operands rather than the
    # record's: this is what the sweep actually factors.
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import nemo_tracer_tridiagonal
    lower, diagonal, upper = nemo_tracer_tridiagonal(
        jnp.asarray(live["K"]), jnp.asarray(live["dz"]),
        jnp.asarray(live["e3w"]), float(live["dt"]),
        jnp.asarray(live["wet"]) > 0.0, dtype=jnp.float64)
    for label, built in (("zwi", lower), ("zwd", diagonal), ("zws", upper)):
        rows.append(_wet_row(f"assembled_from_live.{label}", oracle[label],
                             np.asarray(built), wet_cell))

    # PR2's discriminator: is the K difference the FOLD, or the closure?
    fold_removed = live["K"] - live["K33"]
    attribution = {
        "max_abs_K33_live": float(np.abs(live["K33"])[wet_face].max()),
        "max_abs_ah_wslp2_nemo": float(np.abs(oracle["K33"]).max()),
        "max_abs_K_difference": float(
            np.abs(live["K"] - oracle["K"])[wet_face].max()),
        "max_abs_K_difference_with_fold_removed": float(
            np.abs(fold_removed - oracle["avt"])[wet_face].max()),
    }
    attribution["fold_dominates"] = bool(
        attribution["max_abs_K_difference_with_fold_removed"]
        < 0.5 * attribution["max_abs_K_difference"])

    plant_moved_its_row = None
    if plant:
        label = PLANT_ROW[plant]
        oracle_key = plant
        mask = wet_cell if plant in ("dz", "wet") else wet_face
        before = _wet_row(label, oracle[oracle_key], unplanted[plant], mask)
        after = next(r for r in rows if r["name"] == label)
        plant_moved_its_row = bool(
            (before["bit_unequal"], before["absolute_max"],
             before["status"], before["candidate_sha256"])
            != (after["bit_unequal"], after["absolute_max"],
                after["status"], after["candidate_sha256"]))

    status = "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT"
    if any(inert.values()) or any(round_trip.values()):
        status = "PERTURBED"
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round38-matrix-operands-v1",
        "case": CASE,
        "record": str(record), "record_sha256": sha256(record),
        "execution_regime": "production_jit", "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "scored_box": {"cells_cell": int(wet_cell.sum()),
                       "cells_face": int(wet_face.sum()),
                       "domain": list(EXPECTED_DOMAIN)},
        "capture_inert_bits_moved": inert,
        "capture_inert_magnitude": inert_size,
        "in_graph_identity_bits_unequal": round_trip,
        "output_substitution_bits_moved_CONFOUNDED": output_substitution,
        "bottom_face_zero": oracle["bottom_face_zero"],
        "dry_cell_thickness_convention": {
            "status": "REGISTERED, unchanged by this round",
            "lego_dry_thickness_max_abs": float(
                np.abs(live["dz"])[~wet_cell].max()),
            "nemo_dry_thickness_max_abs": float(
                np.abs(oracle["dz"])[~wet_cell].max()),
            "dry_cells": int((~wet_cell).sum()),
            "boundary": ("legoESM's h_partial is exactly 0.0 below the "
                         "seafloor where NEMO's e3t_3d is the positive "
                         "reference thickness (trazdf.f90:445 is written "
                         "unconditionally).  Closing it is a GEOMETRY change; "
                         "round 37 measured that removing the dry-diagonal "
                         "substitution that compensates for it makes "
                         "OVERFLOW's kt=2 tracers non-finite."),
        },
        "k_attribution": attribution,
        "rows": rows,
        "planted_operand": plant,
        "plant_moved_its_own_row": plant_moved_its_row,
        "status": status,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--plant", choices=PLANT_OPERANDS, default=None)
    parser.add_argument("--substitute", action="store_true",
                        help="run the CAUSAL arm instead of the operand table")
    parser.add_argument("--time-level", action="store_true",
                        help="discriminate the call site from the arithmetic")
    parser.add_argument("--dz-owner", action="store_true",
                        help="who owns the dz_after residual")
    parser.add_argument("--knob-redundancy", action="store_true",
                        help="is slope_prd_geometry_stage redundant here")
    parser.add_argument("--kt2-mld-arm", action="store_true",
                        help=("ONE-VARIABLE arm on the kt=2 fold: rerun it "
                              "with NEMO's own zdf_mxl N2-integral "
                              "mixed-layer criterion.  Measurement only; no "
                              "card default is changed."))
    parser.add_argument("--kt2-given-inputs", action="store_true",
                        help="score the fold against NEMO's kt=2 ah_wslp2, "
                             "given NEMO's own before state")
    parser.add_argument(
        "--oracle-root", type=Path,
        default=Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                     "round19_oracle_v2_external"))
    parser.add_argument("--npz", type=Path,
                        default=Path("/tmp/round38_substitution.npz"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.dz_owner:
        report = run_dz_owner(args.record)
        text = json.dumps(report, indent=1, sort_keys=True, default=str)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        print("RECORD-SELF-CONSISTENT cells unequal "
              f"{report['record_self_consistent_cells_unequal']}")
        for row in report["rows"]:
            print(f"{row['status']:<13}{row['name']:<34} "
                  f"unequal {row['bit_unequal']}/{row['n']} "
                  f"max {row['absolute_max']:.6g}")
        return 0 if all(r["status"] == "AT-BAR" for r in report["rows"]) else 1
    if args.knob_redundancy:
        report = run_knob_redundancy()
        text = json.dumps(report, indent=1, sort_keys=True, default=str)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        print(f"KNOB {report['default_stage']} vs {report['arm_stage']}: "
              f"bits moved {report['bits_moved']} of {report['cells']}; "
              f"REDUNDANT {report['redundant']}")
        return 0 if report["redundant"] else 1
    if args.kt2_mld_arm:
        base = run_kt2_given_inputs()
        arm = run_kt2_given_inputs(mld_criterion="n2_integral")
        base_rel = base["row"]["max_relative"]
        arm_rel = arm["row"]["max_relative"]
        moved = (abs(arm_rel - base_rel) / base_rel) if base_rel else None
        report = {
            "worktree": worktree_stamp(),
            "arm": "kt2 isoneutral fold, one variable: the mixed-layer "
                   "criterion.  NEMO's nmln/hmlp come from zdf_mxl's "
                   "N2-integral test (zdfmxl.F90:95-104); this card resolves "
                   "legoESM's 'rho_c'.  MEASUREMENT ONLY -- no card default "
                   "is changed by this arm.",
            "default": base, "n2_integral": arm,
            "relative_moved_fraction": moved,
        }
        text = json.dumps(report, indent=1, sort_keys=True, default=str)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        for label, rep in (("rho_c (card default)", base),
                           ("n2_integral (NEMO)  ", arm)):
            row = rep["row"]
            print(f"MLD-ARM {label} {row['status']:<12} "
                  f"unequal {row['bit_unequal']}/{rep['scored_wet_faces']} "
                  f"max {row['absolute_max']:.6g} rel {row['max_relative']:.6g} "
                  f"lego_absmax {rep['lego_K33_absmax']:.6g}")
        print(f"MLD-ARM relative moved by "
              f"{'n/a' if moved is None else f'{moved:.4g}'} "
              f"(NEMO ah_wslp2 absmax {base['nemo_ah_wslp2_absmax']:.6g})")
        # The preregistered falsifier: a move under 10 per cent REFUTES the
        # mixed-layer index as the ranked owner.  An arm that returns 0
        # whatever it measures cannot report its own refutation.
        report["falsifier"] = "relative moved by less than 0.10"
        report["falsified"] = (moved is None or moved < 0.10)
        if args.json:
            args.json.write_text(
                json.dumps(report, indent=1, sort_keys=True, default=str)
                + "\n")
        if report["falsified"]:
            print("MLD-ARM REFUTED: the mixed-layer criterion is not the "
                  "ranked owner")
            return 1
        return 0
    if args.kt2_given_inputs:
        report = run_kt2_given_inputs()
        text = json.dumps(report, indent=1, sort_keys=True, default=str)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        row = report["row"]
        print(f"KT2-RECORD-DISCRIMINATING {report['discriminating']} "
              f"NEMO ah_wslp2 absmax {report['nemo_ah_wslp2_absmax']:.17g}")
        print(f"KT2-GIVEN-INPUTS {row['status']} "
              f"unequal {row['bit_unequal']}/{report['scored_wet_faces']} "
              f"max {row['absolute_max']:.6g} rel {row['max_relative']:.6g}")
        if not report["discriminating"]:
            print("FINDING the kt=2 record cannot discriminate either")
            return 3
        return 0 if row["status"] == "AT-BAR" else 1
    if args.time_level:
        report = run_time_level_discriminator()
        text = json.dumps(report, indent=1, sort_keys=True, default=str)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        k = report["k33_from_step_entry_state"]
        print(f"STEP-ENTRY-PTP {report['step_entry_wet_per_level_ptp']}")
        print(f"K33-FROM-STEP-ENTRY absmax {k['absolute_max']:.17g} "
              f"nonzero {k['nonzero_cells']}/{k['n']}")
        print(f"OWNER {report['owner']}")
        return 0
    if args.substitute:
        report = run_substitution(args.record, oracle_root=args.oracle_root,
                                  npz=args.npz)
        text = json.dumps(report, indent=1, sort_keys=True, default=str)
        if args.json:
            args.json.write_text(text + "\n")
        print(text)
        for name, value in report["residual"].items():
            print(f"{name:<22} T {value['T']:.17g}  S {value['S']:.17g}")
        print(f"FOLD-ONLY T {report['fold_only']['T']:.6g} "
              f"S {report['fold_only']['S']:.6g}")
        print(f"CLOSURE-ONLY T {report['closure_only']['T']:.6g} "
              f"S {report['closure_only']['S']:.6g}")
        print(f"NOISE-FLOOR (null substitution) "
              f"T {report['substitution_noise_floor']['T']:.6g} "
              f"S {report['substitution_noise_floor']['S']:.6g}")
        print(f"OWNED-FRACTION-T {report['owned_fraction_T']}")
        return 0
    try:
        report = run(args.record, plant=args.plant)
    except RecordError as error:
        print(f"FAIL: {error}")
        return 2
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    for row in report["rows"]:
        cell = row.get("largest_difference_cell")
        print(f"{row['status']:<12} {row['name']:<34} "
              f"wet_unequal {row['bit_unequal']}/{row['n']} "
              f"max {row['absolute_max']:.6g} "
              f"rel {row['max_relative']:.6g} "
              f"dry_unequal {row['dry_cells_unequal']}"
              + (f" argmax {cell}" if cell else ""))
    a = report["k_attribution"]
    print(f"K-ATTRIBUTION difference {a['max_abs_K_difference']:.6g}; "
          f"with the fold removed "
          f"{a['max_abs_K_difference_with_fold_removed']:.6g}; "
          f"legoESM fold {a['max_abs_K33_live']:.6g}; "
          f"NEMO ah_wslp2 {a['max_abs_ah_wslp2_nemo']:.6g}; "
          f"fold_dominates {a['fold_dominates']}")
    print(f"CAPTURE-INERT {report['capture_inert_bits_moved']}  "
          f"IN-GRAPH-IDENTITY {report['in_graph_identity_bits_unequal']}")
    for tracer, size in report["capture_inert_magnitude"].items():
        print(f"CAPTURE-INERT-SIZE {tracer} max_abs {size['max_abs']:.6g} "
              f"at {size['cell']} = {size['ulps_at_that_cell']:.3g} ulp")
    print("OUTPUT-SUBSTITUTION (confounded by constant lowering, reported "
          f"not scored) {report['output_substitution_bits_moved_CONFOUNDED']}")
    print(f"STATUS {report['status']}")
    if args.plant:
        print(f"PLANT {args.plant} moved_its_own_row "
              f"{report['plant_moved_its_own_row']}")
        # THE VERDICT IS THE ROW, NOT THE EXIT STATUS.  Three operand rows are
        # DEBT unplanted, so "the gate exited non-zero" is satisfied by a
        # plant that did nothing at all.
        if not report["plant_moved_its_own_row"]:
            print("FAIL: the planted ulp did not move its own row; this "
                  "control proves nothing")
            return 3
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    sys.exit(main())

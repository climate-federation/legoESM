#!/usr/bin/env python3
"""Round-31 GYRE walk into ``dyn_zdf``: depth structure, and the pre-solve vector.

WHERE THIS PICKS UP.  Round 30 put the stage-3 momentum RHS entering
``dyn_zdf`` at ``2.0614443633579772e-16`` and GYRE's kt=1 stage-3 velocity is
still ``9.481924730527598e-07``, so the divergence is created between the two.
Three things sit in that gap: ``dyn_zdf``'s own explicit stage update
(``dynzdf.F90:121-122``), its implicit vertical solve
(``dynzdf.F90:182-195``, ``:296``, ``:328-330`` and the three recurrences),
and the barotropic correction that follows it
(``stprk3_stg.F90:437-446``).  This gate separates them.

``depth_profile`` is the free discriminator.  The barotropic correction adds a
2-D field UNIFORMLY down each column (``stprk3_stg.F90:444-445``), so if it
owned the error the error would be column-uniform; the implicit solve is
depth-structured, because the wind stress enters only the top cell
(``dynzdf.F90:328-330``) and the drag only the deepest wet one
(``dynzdf.F90:296``).  The metric is weighting-free: per column, the signed
error's peak-to-peak SPAN against its own largest magnitude, and a column
whose error is exactly zero is not scored at all.

``pre_solve`` scores the model's EXISTING pre-implicit exposure against
NEMO's explicit stage update.  Round 30 recorded that legoESM exposes no
pre-solve vector; that was wrong -- ``expose_pre_implicit_state`` publishes
``state_new`` immediately before ``_apply_implicit_vertical_mixing``
(``ocean_model_latlon_cgrid.py:7731-7733``), and it carries u and v.

THE ALIGNMENT, read before either was scored.  legoESM performs NEMO's
barotropic removal (``dynzdf.F90:148-150``) and its barotropic bottom-stress
addition (``dynzdf.F90:153-159``) INSIDE the implicit solver, under
``zdf_baroclinic_only`` / ``zdf_drag_in_matrix``
(``ocean_model_latlon_cgrid.py:9865-9876`` and ``:9991-9996``).  The record's
``uu_Kaa_pre`` is captured AFTER both.  So the model's capture point aligns
with ``dynzdf.F90:121-122`` and NOT with ``uu_Kaa_pre``, and the oracle side
of ``pre_solve`` is the explicit update reconstructed from the record's own
operands.

``calibrate`` is what makes that reconstruction quotable (Rule 1e): it walks
the record forward through ``:121-122``, ``:149-150`` and ``:156-159`` and
requires the result to equal the record's own ``uu_Kaa_pre`` bit for bit.  A
single unequal cell there means the alignment table is wrong and no number
from ``pre_solve`` may be used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    DIMS,
    _surface_forcings,
    _xyz,
    expected_masks,
    lego_fields,
    read_stage,
    require,
    score,
    sha256,
)
from nemo_testcase_l2_gyre_round29_zdf_matrix import read_zdf_matrix

ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round29_oracle_v2_zdf_matrix")
ZDF_MATRIX_RECORD = "oracle_zdf_matrix_kt00000001.bin"
STAGE3_RECORD = "oracle_stage_kt00000001_s3.bin"
ADMISSION_BASELINE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round19_oracle_v2_external")
# Column-uniformity threshold from the round-31 preregistration: a column is
# UNIFORM when its signed error spans less than this fraction of its own peak.
UNIFORM_TOL = 1.0e-2


def _fortran(rec: dict, name: str) -> np.ndarray:
    """One record array in NEMO's own ``(jpi, jpj, jpk)`` layout."""
    return np.asarray(rec["arrays"][name], dtype=np.float64)


def nemo_explicit_update(rec: dict, face: str) -> np.ndarray:
    """``dynzdf.F90:121-122`` on the record's own operands, Fortran layout.

    ``puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kbb) + rDt * puu(ji,jj,jk,Krhs) )
    * umask(ji,jj,jk)`` -- the ``ln_dynadv_vec`` arm, taken because
    ``ln_dynadv_vec = .true.`` (``GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:161``).
    Written in NEMO's own association order so the two evaluations agree bit
    for bit and not merely algebraically.
    """
    p = "uu" if face == "u" else "vv"
    mask = _fortran(rec, "umask" if face == "u" else "vmask")
    rdt = float(rec["arrays"]["rDt"])
    # NEMO writes levels 1..jpkm1 (DO_2Dik(0,0, 1,jpkm1,1)); this writes all
    # jpk, which agrees only where the mask is already zero above jpkm1.
    jpkm1 = int(rec["header"]["jpkm1"])
    require(not np.any(mask[..., jpkm1:] != 0.0),
            "the face mask is nonzero above jpkm1, where NEMO's explicit "
            "update does not run; the whole-array form would invent a value")
    return (_fortran(rec, f"{p}_Kbb_in") + rdt * _fortran(rec, f"{p}_Krhs_in")) * mask


def nemo_barotropic_removed(rec: dict, face: str, start=None) -> np.ndarray:
    """The explicit update after ``dynzdf.F90:149-150`` only.

    ``puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b(ji,jj,Kaa) ) *
    umask(ji,jj,jk)`` on levels ``1..jpkm1``, under
    ``IF( ln_drgimp .AND. ln_dynspg_ts )``, both true on this deck.  It is a
    separate boundary from :func:`nemo_pre_solve_vector` because legoESM adds
    the bottom stress inside its own solver even if it removes the barotropic
    mode earlier.
    """
    p = "uu" if face == "u" else "vv"
    mask = _fortran(rec, "umask" if face == "u" else "vmask")
    ub = np.asarray(rec["arrays"][f"{p}_b_Kaa"], dtype=np.float64)
    jpkm1 = int(rec["header"]["jpkm1"])
    out = (nemo_explicit_update(rec, face) if start is None
           else np.asarray(start, dtype=np.float64)).copy()
    out[..., :jpkm1] = (out[..., :jpkm1] - ub[..., None]) * mask[..., :jpkm1]
    return out


def nemo_pre_solve_vector(rec: dict, face: str, start=None) -> np.ndarray:
    """The explicit update plus ``dynzdf.F90:149-150`` and ``:156-159``.

    Both corrections run because ``ln_drgimp`` and ``ln_dynspg_ts`` are both
    true on this deck (``round19_oracle_v2_external/ocean.output:629`` and the
    split-explicit banner).  The barotropic removal is applied on levels
    ``1..jpkm1``; the bottom stress lands on ``mbku``/``mbkv`` only.
    """
    b = "uu_b_Kaa" if face == "u" else "vv_b_Kaa"
    e3 = _fortran(rec, "e3u_Kaa" if face == "u" else "e3v_Kaa")
    mbk = np.asarray(rec["arrays"]["mbku" if face == "u" else "mbkv"])
    ub = np.asarray(rec["arrays"][b], dtype=np.float64)
    cd = np.asarray(rec["arrays"]["rCdU_bot"], dtype=np.float64)
    zdt_2 = float(rec["arrays"]["rDt"]) * 0.5            # dynzdf.F90:97

    out = nemo_barotropic_removed(rec, face, start)   # :121-122 then :149-150
    # :153-159 -- add the bottom stress due to the barotropic component only.
    # rCdU_bot(ji+1,jj) for u; rCdU_bot(ji,jj+1) for v (dynzdf.F90:157,159).
    # np.roll WRAPS; NEMO reads a real neighbour, so the writer's tile must
    # end before the array does or the last column would take column 1's rate.
    h = rec["header"]
    require(h["ntei"] < h["jpi"] and h["ntej"] < h["jpj"],
            "the writer's tile reaches the array edge, where the +1 neighbour "
            "shift wraps instead of reading a real neighbour")
    shifted = np.roll(cd, -1, axis=0 if face == "u" else 1)
    nx, ny, _ = out.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    kk = mbk.astype(int) - 1                             # iku is 1-based
    ok = (kk >= 0) & (kk < out.shape[2])
    term = zdt_2 * (shifted + cd) * ub / e3[ii, jj, np.where(ok, kk, 0)]
    out[ii[ok], jj[ok], kk[ok]] = out[ii[ok], jj[ok], kk[ok]] + term[ok]
    return out


def _bit_rows(name: str, oracle: np.ndarray, candidate: np.ndarray,
              mask: np.ndarray) -> dict:
    """Bit comparison of two Fortran-layout record arrays on NEMO's own mask."""
    sel = np.asarray(mask, dtype=bool)
    o, c = oracle[sel], candidate[sel]
    return {
        "name": name,
        "n": int(sel.sum()),
        "n_unequal": int(np.count_nonzero(c.view(np.uint64) != o.view(np.uint64))),
        "max_abs": float(np.max(np.abs(c - o))) if sel.any() else 0.0,
        "exact": bool(np.array_equal(c, o)),
    }


def run_calibrate(oracle_root: Path, *, plant: bool = False) -> dict:
    """P4a: does the alignment table reproduce the record's own pre-solve vector?"""
    record = oracle_root / ZDF_MATRIX_RECORD
    rec = read_zdf_matrix(record)
    require(tuple(rec["header"][k] for k in ("jpi", "jpj", "jpk")) == DIMS,
            f"{record}: record dimensions are not GYRE's")
    h = rec["header"]
    # Only the writer's own tile is meaningful; everything outside it is halo.
    interior = np.zeros(DIMS, dtype=bool)
    interior[h["ntsi"] - 1:h["ntei"], h["ntsj"] - 1:h["ntej"], :] = True
    rows = []
    for face in ("u", "v"):
        built = nemo_pre_solve_vector(rec, face)
        if plant and face == "u":
            built = built.copy()
            built[h["ntsi"], h["ntsj"], 0] = np.nextafter(
                built[h["ntsi"], h["ntsj"], 0], np.inf)
        mask = _fortran(rec, "umask" if face == "u" else "vmask") > 0.5
        rows.append(_bit_rows(
            f"{CASE}.kt1.stage3.zdf_pre_solve_reconstruction.{face}",
            _fortran(rec, "uu_Kaa_pre" if face == "u" else "vv_Kaa_pre"),
            built, mask & interior))
    status = "AT-BAR" if all(r["exact"] for r in rows) else "DEBT"
    if plant:
        require(status == "DEBT", "the planted one-ulp move did not fire")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round31-zdf-walk-v1",
        "case": CASE,
        "mode": "calibrate",
        "claim": ("dynzdf.F90:121-122 + :149-150 + :156-159 on the record's "
                  "own operands reproduce uu_Kaa_pre/vv_Kaa_pre bit for bit"),
        "record": str(record),
        "record_sha256": sha256(record),
        "tile_bounds": {k: h[k] for k in ("ntsi", "ntei", "ntsj", "ntej")},
        "status": status,
        "rows": rows,
        "planted_control": plant,
    }


def _model_kt1(mode: str):
    """One production kt=1 step, fp64, production JIT."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "certification requires production JIT; JAX_DISABLE_JIT is forbidden")
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    hooks = (_NEMOWSRK3TestHooks(expose_pre_implicit_state=True)
             if mode == "pre_solve" else _NEMOWSRK3TestHooks())
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks,
    ).step(card.recipe.initial_state, dt=card.dt_s,
           freshwater=freshwater, surface_forcing=surface)
    state = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, state)
    return card, expected_masks(card), lego_fields(state), jax.default_backend()


def run_pre_solve(oracle_root: Path, *, plant: bool = False) -> dict:
    """P4b: the model's pre-implicit u/v against NEMO's explicit stage update."""
    record = oracle_root / ZDF_MATRIX_RECORD
    rec = read_zdf_matrix(record)
    card, masks, fields, backend = _model_kt1("pre_solve")
    nx, ny, nz = DIMS
    # The three boundaries dyn_zdf passes through before its matrix is built.
    # Which one the model's pre-implicit exposure IS is measured, not assumed.
    boundaries = {
        "A_explicit_update": (nemo_explicit_update, "dynzdf.F90:121-122"),
        "B_barotropic_removed": (nemo_barotropic_removed, "dynzdf.F90:149-150"),
        "C_pre_solve_vector": (nemo_pre_solve_vector, "dynzdf.F90:156-159"),
    }
    rows = []
    for label, (build, cite) in boundaries.items():
        for face in ("u", "v"):
            oracle = _xyz(build(rec, face).ravel(order="F"), nx, ny, nz)
            candidate = fields[face]
            row = score(
                f"{CASE}.kt1.stage3.zdf_entry_{label}.{face}",
                oracle[..., :candidate.shape[-1]], candidate, masks[face],
                plant=plant and face == "u" and label == "A_explicit_update")
            row["nemo_statement"] = cite
            # The candidate's own scale, next to the oracle's: a residual the
            # size of the field means the two sides are different quantities,
            # and that is a different finding from a diverging statement.
            row["candidate_max_abs"] = float(
                np.max(np.abs(candidate[masks[face]])))
            rows.append(row)
    best = min(rows, key=lambda r: r["absolute_max"])
    status = "AT-BAR" if all(
        r["status"] == "AT-BAR" for r in rows
        if r["name"].split(".")[-2] == best["name"].split(".")[-2]) else "DEBT"
    if plant:
        require(status == "DEBT" and rows[0]["absolute_max"] >= 0.9,
                "planted pre-solve violation did not fire")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round31-zdf-walk-v1",
        "case": CASE,
        "mode": "pre_solve",
        "boundary": ("all three of dyn_zdf's entry boundaries, scored in one "
                     "run: A the explicit stage update (dynzdf.F90:121-122), "
                     "B A after the barotropic removal (:149-150), C B after "
                     "the barotropic bottom-stress addition (:156-159), which "
                     "is the record's own uu_Kaa_pre"),
        "closest_boundary": best["name"],
        "closest_absolute_max": best["absolute_max"],
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": backend,
        "record": str(record),
        "record_sha256": sha256(record),
        "status": status,
        "rows": rows,
        "planted_control": plant,
    }


def run_rule12_stage_update(oracle_root: Path, *, plant: bool = False) -> dict:
    """Rule 12 for round 33's operator: NEMO's own inputs, legoESM's expression.

    The changed operator is the RK3 momentum stage update's VECTOR arm,
    ``dynzdf.F90:121-122``, which GYRE's compiled deck takes because it
    resolves ``ln_dynadv_vec = T`` with ``lk_linssh = .FALSE.``.  This drives
    legoESM's own ``rk3_stage_velocity_update`` -- imported from the
    production module the step function calls, not a copy -- with the record's
    ``uu_Kbb_in``, ``uu_Krhs_in``, ``rDt`` and ``umask``, and requires the
    result to equal :func:`nemo_explicit_update` BIT FOR BIT.  Not at the
    1e-15 bar: one multiply, one add and one multiply on identical operands in
    identical association order leaves no room for anything else.

    WHAT THIS CANNOT SEE, written down because the round-33 claim review
    measured it: the record's ``uu_Kbb_in`` is identically 0.0 -- GYRE is at
    rest at kt=1 -- so this arm exercises only ``(0 + rDt*rhs)*umask``.  A
    wrong ``velocity_before`` operand would pass it.  The non-vacuity arm for
    that half lives in
    ``tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round33_stage_arm.py``,
    which drives the same function with a NON-ZERO before-velocity against an
    independent transcription and plants two mutations on it.

    The report also carries the OPEN mask row (round-33 addendum 6): NEMO's
    ``umask`` from the record against the mask every GYRE gate scores on.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        rk3_stage_velocity_update,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    record = oracle_root / ZDF_MATRIX_RECORD
    rec = read_zdf_matrix(record)
    require(tuple(rec["header"][k] for k in ("jpi", "jpj", "jpk")) == DIMS,
            f"{record}: record dimensions are not GYRE's")
    h = rec["header"]
    rdt = float(rec["arrays"]["rDt"])
    interior = np.zeros(DIMS, dtype=bool)
    interior[h["ntsi"] - 1:h["ntei"], h["ntsj"] - 1:h["ntej"], :] = True
    nx, ny, nz = DIMS
    masks = expected_masks(build_nemo_testcase_card(CASE))
    rows, mask_rows = [], []
    for face in ("u", "v"):
        p_ = "uu" if face == "u" else "vv"
        mask = _fortran(rec, "umask" if face == "u" else "vmask")
        before = _fortran(rec, f"{p_}_Kbb_in")
        rhs = _fortran(rec, f"{p_}_Krhs_in")
        candidate = np.asarray(rk3_stage_velocity_update(
            before, rhs, rdt, mask, vector_form=True), dtype=np.float64)
        if plant and face == "u":
            candidate = candidate.copy()
            candidate[h["ntsi"], h["ntsj"], 0] = np.nextafter(
                candidate[h["ntsi"], h["ntsj"], 0], np.inf)
        scored = (mask > 0.5) & interior
        # ROW A is NOT a discharge and is labelled as such.  Both sides
        # evaluate the same expression on the same record operands, so it can
        # only fail if legoESM's helper is not the transcription this gate
        # carries.  Useful as a tripwire on the helper; worthless as evidence
        # against NEMO, because nothing on the "oracle" side is a NEMO OUTPUT.
        # The round-33 DIFF review found this before it reached a receipt.
        row = _bit_rows(
            f"{CASE}.kt1.stage3.stage_update_transcription_identity.{face}",
            nemo_explicit_update(rec, face), candidate, scored)
        row["nemo_statement"] = "dynzdf.F90:121-122"
        row["is_a_discharge"] = False
        row["why_not"] = ("both sides are the same expression on the same "
                          "record operands; no NEMO output array enters it")
        row["before_velocity_max_abs"] = float(np.max(np.abs(before)))
        rows.append(row)
        # ROW B IS the discharge.  It carries legoESM's helper output forward
        # through NEMO's own next two statements -- the barotropic removal at
        # dynzdf.F90:149-150 and the barotropic bottom stress at :156-159 --
        # and scores the result against uu_Kaa_pre, which IS a NEMO OUTPUT
        # ARRAY.  That composition is legitimate because ``--mode calibrate``
        # proves, on NEMO's own operands, that those two statements as
        # transcribed here reproduce uu_Kaa_pre bit for bit; the calibration
        # is re-run below and the gate RAISES rather than reporting if it
        # fails, so the composition can never be quoted uncalibrated.
        composed = nemo_pre_solve_vector(rec, face, start=candidate)
        row_b = _bit_rows(
            f"{CASE}.kt1.stage3.rule12_stage_update_composed.{face}",
            _fortran(rec, "uu_Kaa_pre" if face == "u" else "vv_Kaa_pre"),
            composed, scored)
        row_b["nemo_statement"] = ("dynzdf.F90:121-122 then :149-150 then "
                                   ":156-159, scored against uu_Kaa_pre")
        row_b["is_a_discharge"] = True
        rows.append(row_b)
        # OPEN row, addendum 6: is NEMO's umask the mask the gates score on?
        oracle_mask = _xyz(mask.ravel(order="F"), nx, ny, nz) > 0.5
        model_mask = masks[face]
        trimmed = oracle_mask[..., :model_mask.shape[-1]]
        mask_rows.append({
            "face": face,
            "nemo_wet_cells_in_scored_layout": int(trimmed.sum()),
            "legoesm_wet_cells": int(model_mask.sum()),
            "cells_where_they_disagree": int(
                np.count_nonzero(trimmed != model_mask)),
            "nemo_wet_above_the_models_deepest_level": int(
                oracle_mask[..., model_mask.shape[-1]:].sum()),
        })
    # Rule 1e: the composition may not be quoted unless it is calibrated on
    # NEMO's own operands FIRST.  This raises rather than reporting.
    calibration = run_calibrate(oracle_root)
    require(calibration["status"] == "AT-BAR",
            "the two follow-on statements do not reproduce NEMO's own "
            "uu_Kaa_pre bit for bit, so no composed number may be used")
    status = "AT-BAR" if all(r["exact"] for r in rows) else "DEBT"
    if plant:
        require(status == "DEBT", "the planted one-ulp move did not fire")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round31-zdf-walk-v1",
        "case": CASE,
        "mode": "rule12_stage_update",
        "calibration": {k: calibration[k] for k in ("status", "claim", "rows")},
        "claim": ("legoESM's own rk3_stage_velocity_update, given NEMO's "
                  "uu(Kbb), NEMO's uu(Krhs), NEMO's rDt and NEMO's umask, "
                  "carried forward through NEMO's own dynzdf.F90:149-150 and "
                  ":156-159, reproduces NEMO's OUTPUT ARRAY uu_Kaa_pre BIT "
                  "FOR BIT.  The bare :121-122 row beside it is a "
                  "transcription identity, NOT a discharge: no NEMO output "
                  "enters it"),
        "operator": ("legoesm.ocean.dynamics.ocean_model_latlon_cgrid."
                     "rk3_stage_velocity_update, imported from the production "
                     "module the step function calls"),
        "inputs_are_nemo": [
            f"{ZDF_MATRIX_RECORD}:uu_Kbb_in / vv_Kbb_in",
            f"{ZDF_MATRIX_RECORD}:uu_Krhs_in / vv_Krhs_in",
            f"{ZDF_MATRIX_RECORD}:rDt",
            f"{ZDF_MATRIX_RECORD}:umask / vmask",
        ],
        "blind_spot": ("uu_Kbb_in is identically zero on this card, so BOTH "
                       "rows drive only (0 + rDt*rhs)*umask and a wrong "
                       "velocity_before operand would pass either; the "
                       "non-zero before-velocity arm is in the round-33 unit "
                       "tests, with two plants on it.  NEMO holds no array at "
                       "the bare :121-122 boundary, which is why the "
                       "discharge is the COMPOSED row"),
        "open_mask_row": mask_rows,
        "record": str(record),
        "record_sha256": sha256(record),
        "rDt": rdt,
        "tile_bounds": {k: h[k] for k in ("ntsi", "ntei", "ntsj", "ntej")},
        "status": status,
        "rows": rows,
        "planted_control": plant,
    }


def _column_uniformity(err: np.ndarray, mask: np.ndarray) -> dict:
    """Is the signed error a column-uniform shift, or is it depth-structured?

    ``stprk3_stg.F90:444-445`` adds ``zub(ji,jj)*umask(ji,jj,jk)`` -- the SAME
    number at every level of a column.  So an error owned by an error in
    ``zub`` ITSELF is column-uniform.  The measure needs no weights (Rule 6):
    per column, the signed error's peak-to-peak SPAN against its own largest
    magnitude.

    Two ways this could have voted for the wrong owner, both closed here
    because a reviewer found them by running it.  A column whose error is
    exactly ZERO has span 0 and peak 0; counting it as "uniform" makes every
    already-correct column vote for the barotropic owner, which on a nearly
    exact field is every column.  Zero-error columns are reported as their own
    bucket and vote for nothing.  And a NaN column has ``peak > 0`` false, so
    it too used to read as uniform while ``nanmax`` hid it -- a blow-up would
    have become evidence.  NaN is fatal now.
    """
    counts = mask.sum(axis=-1)
    live = counts >= 2
    require(bool(np.all(np.isfinite(err[mask]))),
            "non-finite error in the depth profile; a NaN column reads as "
            "uniform and would vote for the barotropic owner")
    e = np.where(mask, err, np.nan)
    with np.errstate(invalid="ignore"):
        span = np.nanmax(e, axis=-1) - np.nanmin(e, axis=-1)
        peak = np.nanmax(np.abs(e), axis=-1)
        mean = np.nanmean(e, axis=-1)
    exact = live & (peak == 0.0)
    scored = live & (peak > 0.0)
    ratio = np.where(scored, span / np.where(scored, peak, 1.0), 0.0)
    uniform = scored & (ratio <= UNIFORM_TOL)
    if not scored.any():
        return {
            "columns_with_two_or_more_wet_levels": int(live.sum()),
            "columns_scored": 0,
            "columns_exact": int(exact.sum()),
            "columns_column_uniform_to_1e-2": 0,
            "fraction_column_uniform": None,
            "median_spread_over_peak": None,
            "max_abs_column_mean": None,
            "max_abs_error": 0.0 if mask.any() else None,
        }
    return {
        "columns_with_two_or_more_wet_levels": int(live.sum()),
        "columns_scored": int(scored.sum()),
        "columns_exact": int(exact.sum()),
        "columns_column_uniform_to_1e-2": int(uniform.sum()),
        "fraction_column_uniform": float(uniform.sum() / scored.sum()),
        "median_spread_over_peak": float(np.median(ratio[scored])),
        "max_abs_column_mean": float(np.max(np.abs(mean[scored]))),
        "max_abs_error": float(np.max(np.abs(e[mask]))),
    }


def verdict_for(fractions) -> str:
    """The preregistered verdict ladder, as one testable function.

    P3a: below 10 per cent column-uniform on BOTH faces exonerates the
    barotropic correction and leaves the implicit solve; P3b: 90 per cent or
    more names the correction; anything between, or a face that could not be
    scored at all, names nothing.
    """
    fractions = list(fractions)
    if not fractions or any(f is None for f in fractions):
        return "NO OWNER NAMED"
    if all(f >= 0.9 for f in fractions):
        return "BAROTROPIC-CORRECTION CANDIDATE"
    if all(f < 0.10 for f in fractions):
        return "IMPLICIT-SOLVE CANDIDATE"
    return "NO OWNER NAMED"


def run_depth_profile(oracle_root: Path, *, plant: bool = False) -> dict:
    """P3: bin the kt=1 stage-3 velocity error by model level."""
    stage = oracle_root / STAGE3_RECORD
    oracle = read_stage(stage, 3)
    card, masks, fields, backend = _model_kt1("depth_profile")
    rows, profiles, uniformity = [], {}, {}
    for face in ("u", "v"):
        candidate = fields[face]
        if plant:
            # A column-uniform plant on BOTH faces: what an error in zub would
            # look like.  Planting one face only would leave the verdict's AND
            # depending on the other face's REAL structure, i.e. on the
            # hypothesis under test -- the control would fail for the right
            # reason and blame itself.
            candidate = candidate + 1.0 * masks[face]
        mask = masks[face]
        rows.append(score(f"{CASE}.kt1.stage3.state.{face}",
                          oracle[face][..., :candidate.shape[-1]],
                          candidate, mask))
        err = candidate - oracle[face][..., :candidate.shape[-1]]
        per_level = []
        for k in range(err.shape[-1]):
            sel = mask[..., k]
            if not sel.any():
                per_level.append({"level": k + 1, "n": 0, "max_abs": None,
                                  "mean_abs": None, "mean_signed": None})
                continue
            per_level.append({
                "level": k + 1,
                "n": int(sel.sum()),
                "max_abs": float(np.max(np.abs(err[..., k][sel]))),
                "mean_abs": float(np.mean(np.abs(err[..., k][sel]))),
                "mean_signed": float(np.mean(err[..., k][sel])),
            })
        profiles[face] = per_level
        uniformity[face] = _column_uniformity(err, mask)
    verdict = verdict_for(
        u["fraction_column_uniform"] for u in uniformity.values())
    if plant:
        require(verdict == "BAROTROPIC-CORRECTION CANDIDATE",
                "the column-uniform plant did not read as column-uniform")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round31-zdf-walk-v1",
        "case": CASE,
        "mode": "depth_profile",
        "boundary": ("the kt=1 stage-3 state, i.e. after dyn_zdf and after the "
                     "barotropic correction at stprk3_stg.F90:437-446"),
        "discriminator": (
            "stprk3_stg.F90:444-445 adds zub(ji,jj)*umask(ji,jj,jk), the same "
            "number at every level of a column, so an error it owns is "
            "column-uniform; dynzdf.F90:328-330 puts the wind in the top cell "
            "only and :296 the drag in the deepest wet one, so an error the "
            "solve owns is depth-structured"),
        "uniform_tolerance": UNIFORM_TOL,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": backend,
        "record": str(stage),
        "record_sha256": sha256(stage),
        "status": "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT",
        "verdict": verdict,
        "rows": rows,
        "column_uniformity": uniformity,
        "level_profile": profiles,
        "planted_control": plant,
    }


def ordering_constant(rec: dict, face: str, interior: np.ndarray):
    """The column constant the two orderings differ by, and the drag factor.

    ``delta = mean(A) - uu_b(Kaa)``, the reference-weighted depth mean of
    NEMO's explicit update minus the barotropic velocity legoESM's stage
    correction substitutes for it, and ``drag`` is the factor
    ``dynzdf.F90:296`` puts on the deepest wet diagonal,
    ``zDt_2*|rCdU_bot(i+1,j)+rCdU_bot(i,j)|/e3u(iku,Kaa)``.  Returned
    separately from any verdict so both can be tested on operands with a known
    answer.
    """
    p_ = "uu" if face == "u" else "vv"
    jpkm1 = int(rec["header"]["jpkm1"])
    nz = int(rec["header"]["jpk"])
    A = nemo_explicit_update(rec, face)
    e3 = _fortran(rec, "e3u_Kaa" if face == "u" else "e3v_Kaa")
    mask = _fortran(rec, "umask" if face == "u" else "vmask")
    ub = np.asarray(rec["arrays"][f"{p_}_b_Kaa"], dtype=np.float64)
    cd = np.asarray(rec["arrays"]["rCdU_bot"], dtype=np.float64)
    mbk = np.asarray(rec["arrays"]["mbku" if face == "u" else "mbkv"]).astype(int)
    w = (e3 * mask)[..., :jpkm1]
    column = w.sum(axis=-1)
    wet = interior & (column > 0)
    mean_A = np.zeros_like(column)
    mean_A[wet] = (A[..., :jpkm1] * w).sum(axis=-1)[wet] / column[wet]
    delta = np.where(wet, mean_A - ub, 0.0)
    shifted = np.roll(cd, -1, axis=0 if face == "u" else 1)
    nx, ny = column.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    kk = np.clip(mbk - 1, 0, nz - 1)
    drag = np.where(
        wet,
        float(rec["arrays"]["rDt"]) * 0.5 * np.abs(shifted + cd) / e3[ii, jj, kk],
        0.0)
    return delta, drag, wet


def run_ordering_size(oracle_root: Path, *, plant: bool = False) -> dict:
    """P4g: size the column constant the two orderings differ by.

    legoESM applies NEMO's barotropic correction to the stage-3 velocity
    BEFORE the implicit solve and again after it; NEMO applies it once, AFTER
    (``stprk3_stg.F90:437-446``), and before the solve it instead subtracts
    ``uu_b`` (``dynzdf.F90:149-150``).  So the two solves receive vectors that
    differ by a column constant, and a constant is not invariant under this
    solve because the deepest wet diagonal carries the drag
    (``dynzdf.F90:296``).

    Record-only: no model runs here, so the number is the ORACLE's own
    arithmetic and it sizes an operand.  Sizing an operand is not an ablation
    and nothing lands on it (P4h).
    """
    record = oracle_root / ZDF_MATRIX_RECORD
    rec = read_zdf_matrix(record)
    h = rec["header"]
    interior = np.zeros(DIMS[:2], dtype=bool)
    interior[h["ntsi"] - 1:h["ntei"], h["ntsj"] - 1:h["ntej"]] = True
    rows = []
    for face in ("u", "v"):
        delta, drag, wet = ordering_constant(rec, face, interior)
        rows.append({
            "name": f"{CASE}.kt1.stage3.ordering_constant.{face}",
            "max_abs_delta": float(np.max(np.abs(delta[wet]))),
            "median_abs_delta": float(np.median(np.abs(delta[wet]))),
            "max_bottom_drag_factor": float(np.max(drag[wet])),
            "predicted_bottom_error": float(np.max(np.abs(delta[wet]) * drag[wet])),
            "n_columns": int(wet.sum()),
        })
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round31-zdf-walk-v1",
        "case": CASE,
        "mode": "ordering_size",
        "claim": ("the column constant legoESM's solve carries and NEMO's does "
                  "not, times the bottom-cell drag factor dyn_zdf puts on the "
                  "deepest diagonal (dynzdf.F90:296)"),
        "record": str(record),
        "record_sha256": sha256(record),
        "status": "MEASURED",
        "rows": rows,
        "planted_control": plant,
    }


def run_ordering_regression(oracle_root: Path, *, plant: bool = False) -> dict:
    """P4i: is the ordering term the owner COLUMN BY COLUMN, not at one peak?

    The independent review's reading, adopted here: NEMO's barotropic
    correction is NOT an additive constant.  ``stprk3_stg.F90:440`` builds
    ``zub = uu_b(Kaa) - SUM(e3u_0*uu(:,Kaa))*r1_hu_0`` from the SOLVE OUTPUT,
    so it is rank-1 -- a single-level error ``d`` at the deepest wet level
    leaves ``(1-f)*d`` there and ``-f*d`` at every level above, with
    ``f = e3(deepest)/column depth``.  Inverting that gives the pre-correction
    single-cell error from the measured post-correction one.

    Regressing it on the ordering term ``(mean(A) - uu_b) * drag`` -- the
    column constant legoESM's solve carries and NEMO's does not, times the
    drag factor ``dynzdf.F90:296`` puts on the deepest diagonal -- tests the
    mechanism on every column instead of at the maxima.
    """
    record = oracle_root / ZDF_MATRIX_RECORD
    rec = read_zdf_matrix(record)
    h = rec["header"]
    interior2d = np.zeros(DIMS[:2], dtype=bool)
    interior2d[h["ntsi"] - 1:h["ntei"], h["ntsj"] - 1:h["ntej"]] = True
    stage = oracle_root / STAGE3_RECORD
    oracle = read_stage(stage, 3)
    card, masks, fields, backend = _model_kt1("depth_profile")
    nx, ny, nz = DIMS
    rows = []
    for face in ("u", "v"):
        candidate = fields[face]
        mask = masks[face]
        err = candidate - oracle[face][..., :candidate.shape[-1]]
        # f from the MODEL's own geometry, printed rather than assumed.
        e3 = _xyz(_fortran(rec, "e3u_Kaa" if face == "u" else "e3v_Kaa")
                  .ravel(order="F"), nx, ny, nz)[..., :candidate.shape[-1]]
        thick = np.where(mask, e3, 0.0)
        depth = thick.sum(axis=-1)
        deepest = np.where(mask.any(axis=-1),
                           mask.shape[-1] - 1 - np.argmax(mask[..., ::-1], axis=-1),
                           0)
        i2, j2 = np.meshgrid(np.arange(err.shape[0]), np.arange(err.shape[1]),
                             indexing="ij")
        live = mask.any(axis=-1) & (depth > 0)
        f = np.zeros_like(depth)
        f[live] = thick[i2, j2, deepest][live] / depth[live]
        e_bottom = err[i2, j2, deepest]
        d_single = np.where(live, e_bottom / np.maximum(1.0 - f, 1e-30), 0.0)
        delta, drag, wet = ordering_constant(rec, face, interior2d)
        # both sides onto the scored interior layout
        term3 = _xyz(np.ascontiguousarray(
            np.broadcast_to((delta * drag)[..., None], DIMS)).ravel(order="F"),
            nx, ny, nz)
        term = term3[..., 0]
        sel = live & (term != 0.0)
        x, y = term[sel], d_single[sel]
        slope = float(np.dot(x, y) / np.dot(x, x))
        resid = y - slope * x
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else None
        if plant and face == "u":
            slope = 0.0
        rows.append({
            "name": f"{CASE}.kt1.stage3.ordering_regression.{face}",
            "n_columns": int(sel.sum()),
            "slope_through_origin": slope,
            "r_squared": r2,
            "f_deepest_over_depth_min": float(f[live].min()),
            "f_deepest_over_depth_max": float(f[live].max()),
            "max_abs_post_correction_error": float(np.max(np.abs(err[mask]))),
            "max_abs_single_cell_error": float(np.max(np.abs(d_single[live]))),
            "max_abs_ordering_term": float(np.max(np.abs(term[sel]))),
            "at_bar": bool(abs(slope - 1.0) <= 0.02 and (r2 or 0.0) >= 0.99),
        })
    verdict = ("ORDERING TERM CONFIRMED COLUMN-WISE"
               if all(r["at_bar"] for r in rows)
               else "NO OWNER NAMED")
    if plant:
        require(verdict == "NO OWNER NAMED",
                "a zeroed slope still read as confirmed")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round31-zdf-walk-v1",
        "case": CASE,
        "mode": "ordering_regression",
        "claim": ("the single-cell error at the deepest wet level, recovered "
                  "from the measured post-correction error by inverting the "
                  "rank-1 barotropic correction (stprk3_stg.F90:440,444-445), "
                  "against the ordering term legoESM's solve carries"),
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": backend,
        "record": str(record),
        "record_sha256": sha256(record),
        "stage_record_sha256": sha256(stage),
        "status": "MEASURED",
        "verdict": verdict,
        "rows": rows,
        "planted_control": plant,
    }


MODES = {"calibrate": run_calibrate, "pre_solve": run_pre_solve,
         "depth_profile": run_depth_profile, "ordering_size": run_ordering_size,
         "ordering_regression": run_ordering_regression,
         "rule12_stage_update": run_rule12_stage_update}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(MODES))
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    restart = args.oracle_root / "GYRE_OMIP_L2_P3_00000010_restart.nc"
    require(sha256(restart) == sha256(ADMISSION_BASELINE / restart.name),
            "the instrumented root's final restart moved; the records are not "
            "WRITE-only and no number from them may be quoted")
    report = MODES[args.mode](args.oracle_root, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    for row in report["rows"]:
        if "n_unequal" not in row:
            print(f"{'SIZE':<8} {row['name']:<52} " + "  ".join(
                f"{k} {v:.17g}" for k, v in row.items()
                if isinstance(v, float)))
            continue
        print(f"{row.get('status', 'BIT'):<8} {row['name']:<52} "
              f"unequal {row['n_unequal']}/{row['n']} "
              f"max {row.get('absolute_max', row.get('max_abs')):.17g}")
    print(f"STATUS {report['status']}"
          + (f"  VERDICT {report['verdict']}" if "verdict" in report else ""))
    if args.plant:
        return 1
    return 0 if report["status"] in ("AT-BAR", "MEASURED") else 1


if __name__ == "__main__":
    raise SystemExit(main())

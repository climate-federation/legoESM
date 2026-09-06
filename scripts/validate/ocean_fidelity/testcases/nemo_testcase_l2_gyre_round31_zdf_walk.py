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
error's spread about its own mean, against its own mean.

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
    return (_fortran(rec, f"{p}_Kbb_in") + rdt * _fortran(rec, f"{p}_Krhs_in")) * mask


def nemo_barotropic_removed(rec: dict, face: str) -> np.ndarray:
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
    out = nemo_explicit_update(rec, face).copy()
    out[..., :jpkm1] = (out[..., :jpkm1] - ub[..., None]) * mask[..., :jpkm1]
    return out


def nemo_pre_solve_vector(rec: dict, face: str) -> np.ndarray:
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

    out = nemo_barotropic_removed(rec, face)          # :121-122 then :149-150
    # :153-159 -- add the bottom stress due to the barotropic component only.
    # rCdU_bot(ji+1,jj) for u; rCdU_bot(ji,jj+1) for v (dynzdf.F90:157,159).
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


def _column_uniformity(err: np.ndarray, mask: np.ndarray) -> dict:
    """Is the signed error a column-uniform shift, or is it depth-structured?

    ``stprk3_stg.F90:444-445`` adds ``zub(ji,jj)*umask(ji,jj,jk)`` -- the SAME
    number at every level of a column.  So an error owned by that statement is
    column-uniform.  The measure needs no weights (Rule 6): per column, the
    signed error's peak-to-peak spread against its own largest magnitude.
    """
    counts = mask.sum(axis=-1)
    live = counts >= 2
    e = np.where(mask, err, np.nan)
    with np.errstate(invalid="ignore"):
        span = np.nanmax(e, axis=-1) - np.nanmin(e, axis=-1)
        peak = np.nanmax(np.abs(e), axis=-1)
        mean = np.nanmean(e, axis=-1)
    ratio = np.where(peak > 0, span / np.where(peak > 0, peak, 1.0), 0.0)
    uniform = live & (ratio <= UNIFORM_TOL)
    # A field with no multi-level column cannot discriminate anything, and a
    # reduction over an empty selection raises rather than saying so -- report
    # the census and refuse to name a fraction.
    if not live.any():
        return {
            "columns_with_two_or_more_wet_levels": 0,
            "columns_column_uniform_to_1e-2": 0,
            "fraction_column_uniform": None,
            "median_spread_over_peak": None,
            "max_abs_column_mean": None,
            "max_abs_error": (float(np.nanmax(np.abs(e[mask])))
                              if mask.any() else None),
        }
    return {
        "columns_with_two_or_more_wet_levels": int(live.sum()),
        "columns_column_uniform_to_1e-2": int(uniform.sum()),
        "fraction_column_uniform": float(uniform.sum() / live.sum()),
        "median_spread_over_peak": float(np.median(ratio[live])),
        "max_abs_column_mean": float(np.nanmax(np.abs(mean[live]))),
        "max_abs_error": float(np.nanmax(np.abs(e[mask]))),
    }


def run_depth_profile(oracle_root: Path, *, plant: bool = False) -> dict:
    """P3: bin the kt=1 stage-3 velocity error by model level."""
    stage = oracle_root / STAGE3_RECORD
    oracle = read_stage(stage, 3)
    card, masks, fields, backend = _model_kt1("depth_profile")
    rows, profiles, uniformity = [], {}, {}
    for face in ("u", "v"):
        candidate = fields[face]
        if plant and face == "u":
            candidate = candidate.copy()
            # a column-uniform plant: what the barotropic correction WOULD do.
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
    fractions = [u["fraction_column_uniform"] for u in uniformity.values()]
    verdict = (
        "NO OWNER NAMED"
        if any(f is None for f in fractions)
        else "BAROTROPIC-CORRECTION CANDIDATE"
        if all(f >= 0.9 for f in fractions)
        else "IMPLICIT-SOLVE CANDIDATE"
        if all(f < 0.10 for f in fractions)
        else "NO OWNER NAMED")
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


MODES = {"calibrate": run_calibrate, "pre_solve": run_pre_solve,
         "depth_profile": run_depth_profile}


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
        print(f"{row.get('status', 'BIT'):<8} {row['name']:<52} "
              f"unequal {row['n_unequal']}/{row['n']} "
              f"max {row.get('absolute_max', row.get('max_abs')):.17g}")
    print(f"STATUS {report['status']}"
          + (f"  VERDICT {report['verdict']}" if "verdict" in report else ""))
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

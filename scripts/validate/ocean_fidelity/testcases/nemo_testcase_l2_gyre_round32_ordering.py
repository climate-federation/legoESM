#!/usr/bin/env python3
"""Round-32 GYRE gate: the stage-3 ordering fix's discharge and its residual.

Two modes, both driven by NEMO's OWN record so neither needs the model to be
right about anything upstream:

``rule12_correction``
    Rule 12's question for the operator whose PLACEMENT this round moved.
    legoESM's own ``rk3_stage_barotropic_correction`` -- the exact expression
    the production step calls, imported, not retyped -- is handed NEMO's own
    ``uu_Kaa_out`` (the dyn_zdf output, before the correction), NEMO's own
    ``uu_b(Kaa)``, and NEMO's own ``e3u_0``/``hu_0`` from ``mesh_mask.nc``.
    Its output is scored against NEMO's own stage-3 state record.  That is
    "bit-exact given NEMO's inputs" asked literally.

``slope_model``
    Round 31 left an unexplained 0.27 per cent: regressing the ordering term
    ``(mean(A) - uu_b) * c`` on the back-derived single-cell error gave slope
    0.99726 rather than 1.0, identical on both faces to nine digits.  This
    mode asks whether that shortfall is the SOLVE's own response to the
    wrongly ordered column constant, using NEMO's own tridiagonal.

    The algebra, written out because the estimator follows from it.  The
    matrix satisfies ``M.1 = 1 + c.e_bot`` exactly -- the viscosity rows sum
    to one under zero-flux boundaries and ``dynzdf.F90:296`` adds ``c`` to the
    deepest diagonal -- so a column constant ``kappa`` in the solve input
    returns ``kappa(1 - c.M^-1 e_bot)``.  The post-solve correction removes
    the column mean, so with ``z = M^-1 e_bot`` the surviving error is
    ``-kappa.c.(z - mean_w(z))`` and the gate's back-derived single-cell error
    is ``x.(z_bot - mean_w(z))/(1-f)`` with ``x`` the regressed predictor.
    Two estimators are reported: the one this round PREREGISTERED, which
    approximates ``z`` by ``e_bot/(1+c)`` and therefore cannot produce a
    shortfall larger than ``c``; and the EXACT one, which solves NEMO's own
    tridiagonal.  Both are printed, and the preregistered one is scored
    against its registered window whatever it says.

The tridiagonal solver is calibrated before either number is used: it must
reproduce NEMO's own ``uu_Kaa_out`` from NEMO's own ``uu_Kaa_pre`` plus the
surface stress ``dynzdf.F90:329-330`` adds inside the recurrence.  A solver
that cannot reproduce the oracle's own solve cannot be used to model it.

Exit codes follow the campaign's shared convention (``ulp_move_gate``):
0 the gate passed, 1 the gate found debt OR a plant fired as it must, 2 a
control is BROKEN -- a plant that failed to fire, or a calibration that did
not hold.  A caller testing only ``!= 0`` cannot tell 1 from 2, so a plant run
is checked for exactly 1.

Blind spots, per Rule 2.  ``rule12_correction`` sees the correction's
arithmetic given exact inputs; it is blind to WHERE the production step calls
it, which is what actually changed -- that is the trajectory gate's job.
``slope_model`` is an OFFLINE model of a measured residual; it can only ever
support or refute an explanation, never establish that the residual is gone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    DIMS,
    _xyz,
    expected_masks,
    read_stage,
    require,
    score,
    sha256,
)
from nemo_testcase_l2_gyre_round29_zdf_matrix import read_zdf_matrix
from nemo_testcase_l2_gyre_round31_zdf_walk import (
    ADMISSION_BASELINE,
    ORACLE_ROOT,
    STAGE3_RECORD,
    ZDF_MATRIX_RECORD,
    ordering_constant,
)

# The round-31 regression's measured slopes, the thing slope_model explains.
OBSERVED_SLOPE = {
    "u": 0.99726006226993791,
    "v": 0.99726003050683432,
}
# Preregistered acceptance window on the PREDICTED shortfall, as a fraction of
# the observed shortfall (manifests/..._round32_preregister.json, item 3).
CONFIRM_FRACTION = 0.05
REFUTE_FRACTION = 0.20


def slope_verdict(misses) -> str:
    """The three-way ladder, as a FUNCTION so a test can exercise both ends.

    ``misses`` are per-face relative distances between a model's predicted
    slope shortfall and the observed one.  Round 31's diff review caught two
    tests that grepped a verdict string instead of running the ladder; this
    exists so that cannot repeat.
    """
    misses = list(misses)
    if not misses:
        raise ValueError("slope_verdict needs at least one face")
    if all(m <= CONFIRM_FRACTION for m in misses):
        return "SOLVE RESPONSE CONFIRMED"
    if all(m > REFUTE_FRACTION for m in misses):
        return "SECOND RESIDUAL"
    return "NO VERDICT"


def _interior2d(rec: dict) -> np.ndarray:
    h = rec["header"]
    interior = np.zeros(DIMS[:2], dtype=bool)
    interior[h["ntsi"] - 1:h["ntei"], h["ntsj"] - 1:h["ntej"]] = True
    return interior


def _to_scored_2d(field: np.ndarray) -> np.ndarray:
    """A record 2-D array (jpi, jpj) in the gates' scored (lat, lon) layout."""
    return field[2:-2, 2:-2].T


def _thomas(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
            rhs: np.ndarray) -> np.ndarray:
    """Tridiagonal solve, vectorised over the leading axes.

    NEMO's own recurrence shape (``dynzdf.F90:310-341``): forward elimination
    on the diagonal, forward substitution on the right-hand side, backward
    substitution.  Written here to be CALIBRATED against the oracle's own
    solve rather than trusted.
    """
    nk = diag.shape[-1]
    d = diag.astype(np.float64).copy()
    r = rhs.astype(np.float64).copy()
    for k in range(1, nk):
        factor = sub[..., k] / d[..., k - 1]
        d[..., k] = d[..., k] - factor * sup[..., k - 1]
        r[..., k] = r[..., k] - factor * r[..., k - 1]
    out = np.zeros_like(r)
    out[..., nk - 1] = r[..., nk - 1] / d[..., nk - 1]
    for k in range(nk - 2, -1, -1):
        out[..., k] = (r[..., k] - sup[..., k] * out[..., k + 1]) / d[..., k]
    return out


def _matrix(rec: dict, face: str):
    p = "u" if face == "u" else "v"
    sub = np.asarray(rec["arrays"][f"zwi_{p}"], dtype=np.float64)
    diag = np.asarray(rec["arrays"][f"zwd_{p}"], dtype=np.float64)
    sup = np.asarray(rec["arrays"][f"zws_{p}"], dtype=np.float64)
    return sub, diag, sup


def _calibrate_solver(rec: dict, face: str, wet: np.ndarray) -> dict:
    """The solver must reproduce NEMO's own solve before it models anything.

    ``uu_Kaa_pre`` is captured before the matrix build, so it is NOT the whole
    right-hand side: ``dynzdf.F90:329-330`` adds ``rDt*utauU/(e3u(1,Kaa)*rho0)``
    to the surface cell inside the recurrence.  That term is added here from
    the record's own ``utauU``/``rho0``/``e3u_Kaa``.
    """
    p_ = "uu" if face == "u" else "vv"
    tau = "utauU" if face == "u" else "vtauV"
    e3 = np.asarray(rec["arrays"]["e3u_Kaa" if face == "u" else "e3v_Kaa"],
                    dtype=np.float64)
    nk = _matrix(rec, face)[1].shape[-1]
    rhs = np.asarray(rec["arrays"][f"{p_}_Kaa_pre"], dtype=np.float64)[..., :nk]
    rhs = rhs.copy()
    mask = np.asarray(rec["arrays"]["umask" if face == "u" else "vmask"],
                      dtype=np.float64)
    rhs[..., 0] = rhs[..., 0] + (
        float(rec["arrays"]["rDt"])
        * np.asarray(rec["arrays"][tau], dtype=np.float64)
        / (e3[..., 0] * float(rec["arrays"]["rho0"])) * mask[..., 0])
    sub, diag, sup = _matrix(rec, face)
    got = _thomas(sub, diag, sup, rhs)
    ref = np.asarray(rec["arrays"][f"{p_}_Kaa_out"], dtype=np.float64)[..., :nk]
    sel = wet[..., None] & (mask[..., :nk] > 0)
    err = np.abs(got - ref)[sel]
    return {
        "name": f"{CASE}.kt1.stage3.solver_calibration.{face}",
        "n": int(sel.sum()),
        "max_abs": float(err.max()),
        "reference_max_abs": float(np.abs(ref[sel]).max()),
        "relative_max_abs": float(err.max() / np.abs(ref[sel]).max()),
    }


def run_slope_model(oracle_root: Path, *, plant: bool = False) -> dict:
    """Is round 31's 0.27 per cent the solve's own response to the constant?"""
    record = oracle_root / ZDF_MATRIX_RECORD
    rec = read_zdf_matrix(record)
    interior = _interior2d(rec)
    jpkm1 = int(rec["header"]["jpkm1"])
    rows = []
    calibration = []
    for face in ("u", "v"):
        delta, drag, wet = ordering_constant(rec, face, interior)
        calibration.append(_calibrate_solver(rec, face, wet))
        mask = np.asarray(rec["arrays"]["umask" if face == "u" else "vmask"],
                          dtype=np.float64)
        e3 = np.asarray(rec["arrays"]["e3u_Kaa" if face == "u" else "e3v_Kaa"],
                        dtype=np.float64)
        mbk = np.asarray(
            rec["arrays"]["mbku" if face == "u" else "mbkv"]).astype(int)
        w = (e3 * mask)[..., :jpkm1]
        depth = w.sum(axis=-1)
        safe = np.where(depth > 0, depth, 1.0)
        nx, ny = depth.shape
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        kk = np.clip(mbk - 1, 0, jpkm1 - 1)
        f = np.where(wet, w[ii, jj, kk] / safe, 0.0)
        # z = M^-1 e_bot, from NEMO's own tridiagonal.
        e_bot = np.zeros(w.shape, dtype=np.float64)
        e_bot[ii, jj, kk] = 1.0
        sub, diag, sup = _matrix(rec, face)
        z = _thomas(sub, diag, sup, e_bot)
        if plant and face == "u":
            # Control: a solver that returns the identity response makes the
            # exact estimator collapse onto slope 1 and must not confirm.
            z = e_bot
        z_bot = z[ii, jj, kk]
        mean_w_z = np.where(wet, (w * z).sum(axis=-1) / safe, 0.0)
        one_minus_f = np.maximum(1.0 - f, 1e-30)
        g_exact = np.where(wet, (z_bot - mean_w_z) / one_minus_f, 0.0)
        c = drag
        g_registered = np.where(wet, 1.0 / (1.0 + c), 0.0)
        x = np.where(wet, delta * c, 0.0)
        sel = wet & (x != 0.0)
        x2 = x[sel] ** 2
        slope_registered = float(np.sum(x2 * g_registered[sel]) / np.sum(x2))
        slope_exact = float(np.sum(x2 * g_exact[sel]) / np.sum(x2))
        observed = OBSERVED_SLOPE[face]
        shortfall_observed = 1.0 - observed
        rows.append({
            "name": f"{CASE}.kt1.stage3.slope_model.{face}",
            "n_columns": int(sel.sum()),
            "observed_slope_round31": observed,
            "observed_shortfall": shortfall_observed,
            "registered_model_slope": slope_registered,
            "registered_model_shortfall": 1.0 - slope_registered,
            "registered_model_relative_miss": abs(
                (1.0 - slope_registered) - shortfall_observed)
            / shortfall_observed,
            "exact_model_slope": slope_exact,
            "exact_model_shortfall": 1.0 - slope_exact,
            "exact_model_relative_miss": abs(
                (1.0 - slope_exact) - shortfall_observed) / shortfall_observed,
            "max_bottom_drag_factor": float(c[wet].max()),
            "max_bottom_offdiagonal": float(np.abs(sub[ii, jj, kk])[wet].max()),
        })
    require(all(r["max_abs"] == 0.0 or r["relative_max_abs"] < 1e-12
                for r in calibration),
            "the tridiagonal solver does not reproduce NEMO's own solve; "
            f"calibration {calibration}")

    def verdict_for(key: str) -> str:
        return slope_verdict(r[f"{key}_relative_miss"] for r in rows)

    registered_verdict = verdict_for("registered_model")
    exact_verdict = verdict_for("exact_model")
    if plant:
        require(exact_verdict != "SOLVE RESPONSE CONFIRMED",
                "an identity-response solver still confirmed the mechanism")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round32-ordering-v1",
        "case": CASE,
        "mode": "slope_model",
        "claim": ("round 31's 0.27 per cent slope shortfall is the implicit "
                  "solve's own response to the wrongly ordered column "
                  "constant (dynzdf.F90:296 plus the bottom off-diagonal), "
                  "not a second residual"),
        "confirm_fraction": CONFIRM_FRACTION,
        "refute_fraction": REFUTE_FRACTION,
        "record": str(record),
        "record_sha256": sha256(record),
        "solver_calibration": calibration,
        "status": "MEASURED",
        "registered_estimator_verdict": registered_verdict,
        "exact_estimator_verdict": exact_verdict,
        "rows": rows,
        "planted_control": plant,
    }


def run_rule12_correction(oracle_root: Path, *,
                          plant: bool = False) -> dict:
    """Rule 12 for the moved operator: NEMO's inputs, legoESM's expression."""
    from legoesm.ocean.dynamics.barotropic_common import (
        nemo_reference_depth_reciprocal,
        rk3_stage_barotropic_correction,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    reciprocal_provenance: dict[str, str] = {}
    record = oracle_root / ZDF_MATRIX_RECORD
    rec = read_zdf_matrix(record)
    stage = oracle_root / STAGE3_RECORD
    oracle = read_stage(stage, 3)
    mesh_path = ADMISSION_BASELINE / "mesh_mask.nc"
    mesh = xr.open_dataset(mesh_path, decode_times=False)
    nx, ny, nz = DIMS
    masks = expected_masks(build_nemo_testcase_card(CASE))
    jpkm1 = int(rec["header"]["jpkm1"])
    rows = []
    for face in ("u", "v"):
        p_ = "uu" if face == "u" else "vv"
        e3_0 = np.asarray(
            mesh["e3u_0" if face == "u" else "e3v_0"].values
        ).squeeze().transpose(1, 2, 0).astype(np.float64)
        mask3 = np.asarray(
            mesh["umask" if face == "u" else "vmask"].values
        ).squeeze().transpose(1, 2, 0).astype(np.float64)
        require(e3_0.shape == (ny - 4, nx - 4, nz),
                f"mesh_mask e3_0 is {e3_0.shape}, expected the scored layout")
        h_face = e3_0 * mask3
        # NOT NEMO's hu_0.  ROUND-33 CORRECTION: mesh_mask.nc carries no
        # hu_0/hv_0 and no r1_hu_0 -- only e3u_0/e3v_0 and the masks (39
        # variables, checked) -- so the divisor here is RECONSTRUCTED as
        # domain.F90:140-147 builds it, SUM(e3u_0*umask) over jk=1..jpkm1,
        # and NEMO's own reciprocal r1_hu_0 = ssumask/(hu_0 + 1 - ssumask)
        # (domain.F90:159) is never used.  Two association questions ride on
        # that and are OPEN rows at the 1e-15 bar, not closed ones: NEMO
        # accumulates hu_0 in an ascending jk loop where this uses pairwise
        # summation, and NEMO MULTIPLIES by a precomputed reciprocal where
        # this DIVIDES.
        depth = h_face.sum(axis=-1)
        wet2d = (mask3 > 0).any(axis=-1).astype(np.float64)
        # ROUND 36.  If the record carries NEMO's OWN reference geometry --
        # the round-33 instrument appends hu_0/hv_0 and r1_hu_0/r1_hv_0, and
        # the round-35 GYRE acquisition is the first GYRE run that has it --
        # the reciprocal is READ, with no floor and nothing rebuilt.
        # Otherwise it is built here exactly as domain.f90:213 builds it,
        # from the rebuilt depth, and the report says which was used.
        reciprocal_name = f"r1_h{face}_0"
        if reciprocal_name in rec["arrays"]:
            r1_depth = _to_scored_2d(np.asarray(
                rec["arrays"][reciprocal_name], dtype=np.float64))
            reciprocal_provenance[face] = (
                f"{ZDF_MATRIX_RECORD}:{reciprocal_name}, NEMO's own")
        else:
            r1_depth = np.asarray(nemo_reference_depth_reciprocal(
                np.maximum(depth, 1e-10), wet2d), dtype=np.float64)
            reciprocal_provenance[face] = (
                "REBUILT domain.f90:213 from SUM(e3u_0*umask); this record "
                "carries no r1_h_0")
        field = _xyz(
            np.asarray(rec["arrays"][f"{p_}_Kaa_out"],
                       dtype=np.float64).ravel(order="F"), nx, ny, nz)
        target = _to_scored_2d(
            np.asarray(rec["arrays"][f"{p_}_b_Kaa"], dtype=np.float64))
        # NEMO's DO_3D writes levels 1..jpkm1; the stage mask must not invent
        # a value at jpk, so it is refused rather than silently trimmed.
        require(not np.any(mask3[..., jpkm1:] != 0.0),
                "the reference face mask is wet above jpkm1, where "
                "stprk3_stg.F90:443 does not write")
        candidate = np.asarray(rk3_stage_barotropic_correction(
            field, target, h_face, r1_depth, mask3))
        if plant and face == "u":
            candidate = candidate + 1.0
        # The column mean is formed over every level NEMO's SUM covers; the
        # model's own ladder carries one level fewer, and only levels the
        # model holds can be scored.  Trimming AFTER the correction, never
        # before it, so the mean is not silently taken over a short column.
        nlev = masks[face].shape[-1]
        require(not np.any(mask3[..., nlev:] != 0.0),
                "the reference face mask is wet above the model's deepest "
                "level, so trimming would drop a scored cell")
        rows.append(score(
            f"{CASE}.kt1.stage3.rule12_correction.{face}",
            oracle[face][..., :nlev], candidate[..., :nlev], masks[face]))
        rows[-1]["nemo_statement"] = "stprk3_stg.F90:440,444-445"
    status = "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT"
    if plant:
        require(status == "DEBT", "a planted unit offset still read AT-BAR")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round32-ordering-v1",
        "case": CASE,
        "mode": "rule12_correction",
        "claim": ("legoESM's own rk3_stage_barotropic_correction, given "
                  "NEMO's dyn_zdf output, NEMO's uu_b(Kaa) and NEMO's e3u_0 "
                  "with the column depth RECONSTRUCTED from it, reproduces "
                  "NEMO's own stage-3 velocity AT the 1e-15 bar -- not bit "
                  "for bit; the rows carry exact=false"),
        "reciprocal_provenance": reciprocal_provenance,
        "divisor_provenance": (
            "mesh_mask.nc carries no hu_0 and no r1_hu_0, so the face "
            "thicknesses come from it and the RECIPROCAL comes from the "
            "record when the record has it -- reciprocal_provenance names "
            "the source per face.  ROUND 36: the operator now MULTIPLIES by "
            "that reciprocal (stprk3_stg.f90:522, domain.f90:213) where it "
            "used to divide by a rebuilt, floored depth.  The "
            "rebuilt-sum-versus-NEMO-hu_0 row stays OPEN."),
        "operator": ("legoesm.ocean.dynamics.barotropic_common."
                     "rk3_stage_barotropic_correction, imported from the "
                     "production module the step function calls"),
        "inputs_are_nemo": [
            f"{ZDF_MATRIX_RECORD}:uu_Kaa_out / vv_Kaa_out",
            f"{ZDF_MATRIX_RECORD}:uu_b_Kaa / vv_b_Kaa",
            "mesh_mask.nc:e3u_0 / e3v_0 / umask / vmask",
        ],
        "inputs_reconstructed_not_nemo": [
            "depth_ref = SUM(e3u_0*umask), rebuilt; NEMO's hu_0 / r1_hu_0 "
            "are not in mesh_mask.nc and are not read",
        ],
        "record": str(record),
        "record_sha256": sha256(record),
        "stage_record_sha256": sha256(stage),
        "mesh_mask": str(mesh_path),
        "status": status,
        "rows": rows,
        "planted_control": plant,
    }


MODES = {"slope_model": run_slope_model,
         "rule12_correction": run_rule12_correction}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(MODES))
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = MODES[args.mode](args.oracle_root, plant=args.plant)
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}")
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    for row in report["rows"]:
        print(f"{row['name']}")
        for key, value in row.items():
            if key != "name":
                print(f"    {key}: {value}")
    if report["mode"] == "slope_model":
        print(f"REGISTERED ESTIMATOR: {report['registered_estimator_verdict']}")
        print(f"EXACT ESTIMATOR:      {report['exact_estimator_verdict']}")
        return 1 if args.plant else 0
    print(f"STATUS {report['status']}")
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())

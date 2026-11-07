#!/usr/bin/env python
"""Round 222 / VORTEX_SMT round 10 -- the SMT-2 first-over-bar owner walk.

The rung's first over-bar row is kt=2 u.  Its owner is the BOTTOM-CELL
DIVISOR of NEMO's implicit linear drag, walked here one operand at a time
from the card's own resolved geometry -- no hand prediction, no model run.

NEMO, compiled source of the SMT-2 build
(tests/VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo):

  zdfdrg.f90  drg_init   pCd0 = rn_Cd0 * zmsk_boost  (zmsk_boost = ssmask,
                         ln_boost = .false.)
  zdfdrg.f90  zdf_drg_lin  pCdU = - pCd0 * rn_Uc0      [constant in time]
  dynzdf.f90:306  zwd(ji,iku) = zwd(ji,iku)
                   - zDt_2*( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) )
                   / ( e3u_3d(ji,jj,iku) * (1 + r3u(ji,jj,Kaa)*umask(ji,jj,iku)) )
  dynzdf.f90:473  the same statement on the V face with e3v_3d / r3v
  dynzdf.f90:166,168  the barotropic bottom-stress re-add, same divisor

``e3u_3d`` here is NEMO's OWN U-point scale factor, which over z partial
steps is the SHALLOWER neighbour's thickness (the min rule the card's
geometry gate already pins to 0 ULP), NOT a two-cell average.

legoESM divides the same statement by ``dz_u_open`` (
ocean_model_latlon_cgrid.py, the ``zdf_drag_in_matrix`` block), and
``dz_u_open = interp_cell_to_uface(dz_cell)`` is the plain two-cell AVERAGE
(operators_latlon_cgrid.py ``interp_cell_to_uface``: "Simple average of the
two cells sharing each lon face").  On a flat bottom the two agree exactly;
over a partial-cell bottom they do not, and this probe measures by how much.

Usage:
  nemo_testcase_l1_vortex_smt_round10_drag_divisor_walk.py [--case ...]
      [--output walk.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# rung-0 namelist_cfg:270 + namelist_ref:834/:835 -- the resolved linear rate.
RN_CD0 = 1.0e-3
RN_UC0 = 0.4
RDT_S = 2880.0          # namelist_cfg rn_Dt on the 30 km VORTEX deck


def walk(case: str) -> dict:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    card = build_nemo_testcase_card(case)
    z = card.recipe.z_coord
    e3t = np.asarray(z.h_partial)                     # (nlat, nlon, nlev)
    bl = np.asarray(z.bottom_level)

    # NEMO's U-point scale factor: the shallower neighbour (min rule).  Face
    # ji couples cells ji and ji+1.
    e3u_nemo = np.minimum(e3t, np.roll(e3t, -1, axis=1))
    # legoESM's drag divisor: the plain two-cell average at the same face.
    e3u_lego = 0.5 * (e3t + np.roll(e3t, -1, axis=1))
    bl_u = np.minimum(bl, np.roll(bl, -1, axis=1))

    nlev = e3t.shape[-1]
    lev = np.arange(nlev)
    isbot = (lev[None, None, :] == bl_u[..., None]) & (bl_u[..., None] >= 0)

    a = e3u_nemo[isbot]
    b = e3u_lego[isbot]
    diff = b - a
    nz = diff != 0.0
    # The per-step implicit damping the drag puts on the bottom cell is
    # 1/(1 + rDt*r/e3u) with r = rn_Cd0*rn_Uc0.  The two divisors therefore
    # give two different retained fractions; their gap is the statement's
    # own magnitude, independent of any model run.
    r = RN_CD0 * RN_UC0
    keep_nemo = 1.0 / (1.0 + RDT_S * r / a)
    keep_lego = 1.0 / (1.0 + RDT_S * r / b)
    out = {
        "case": case,
        "statement": "dynzdf.f90:306 / :473 divisor e3u_3d(iku) (min rule) "
                     "vs legoESM interp_cell_to_uface (two-cell average)",
        "rn_Cd0": RN_CD0, "rn_Uc0": RN_UC0, "rCdU_bot_m_s": -r,
        "rDt_s": RDT_S,
        "bottom_u_faces": int(isbot.sum()),
        "faces_where_divisors_differ": int(nz.sum()),
        "max_abs_divisor_diff_m": float(np.max(np.abs(diff))) if a.size else 0.0,
        "max_rel_divisor_diff": float(np.max(np.abs(diff) / a)) if a.size else 0.0,
        "min_e3u_nemo_m": float(a.min()) if a.size else 0.0,
        "min_e3u_lego_m": float(b.min()) if b.size else 0.0,
        "max_retained_fraction_gap": float(np.max(np.abs(keep_lego - keep_nemo)))
            if a.size else 0.0,
        "max_damping_ratio_lego_over_nemo": float(
            np.max((1.0 - keep_lego) / np.maximum(1.0 - keep_nemo, 1e-300)))
            if a.size else 0.0,
        "min_damping_ratio_lego_over_nemo": float(
            np.min((1.0 - keep_lego) / np.maximum(1.0 - keep_nemo, 1e-300)))
            if a.size else 0.0,
    }
    return out


def measured(case: str, smt2_npz: Path, smt1_npz: Path, kt: int = 2,
             field: str = "u", floor: float = 1e-7) -> dict:
    """The REALISED drag ratio, paired FACE BY FACE with the prediction.

    The two ladders start from the SAME recorded entry, so
    ``lego(SMT2) - lego(SMT1)`` is legoESM's own drag effect and
    ``oracle(SMT2) - oracle(SMT1)`` is NEMO's, on the same cells.  Their
    ratio is what the divisor statement predicts.  Both the predicted and
    the measured minimum are reported WITH THEIR CELL INDEX so the campaign's
    argmax rule can be applied: a prediction and a measurement that agree in
    value but sit on different faces are not the same statement.
    """
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    import importlib.util
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_traj_gate", here / "nemo_testcase_phase3_trajectory_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    card = build_nemo_testcase_card(case)
    mask = np.asarray(gate.expected_masks(card)[field])
    z2, z1 = np.load(smt2_npz), np.load(smt1_npz)

    def row(z, npz_path):
        """Find the row by NAME, through the registry the gate wrote beside it.

        The npz keys are NOT in the JSON's row order (the gate writes extra
        arrays between steps), so the row is identified by matching the
        named row's own ``normalized_max_abs`` -- never by index arithmetic.
        """
        reg = json.loads(
            Path(str(npz_path).replace(".residuals.npz", ".json")).read_text())
        case_name = reg["case"]
        want_name = f"{case_name}.kt{kt}.before.{field}"
        target = next(r["normalized_max_abs"] for st in reg["steps"]
                      for r in st["rows"] if r["name"] == want_name)
        n = int(mask.sum())
        hits = [k[:-len("_residual")] for k in z.files
                if k.endswith("_residual") and z[k].shape == (n,)
                and float(np.max(np.abs(z[k]))) == target]
        if len(hits) != 1:
            raise SystemExit(
                f"REFUSE: {want_name} matched {len(hits)} residual arrays")
        return hits[0]

    k2, k1 = row(z2, smt2_npz), row(z1, smt1_npz)

    def unflat(a):
        f = np.zeros(mask.shape)
        f[mask] = a
        return f

    nemo = unflat(z2[k2 + "_oracle"]) - unflat(z1[k1 + "_oracle"])
    lego = unflat(z2[k2 + "_candidate"]) - unflat(z1[k1 + "_candidate"])

    e3t = np.asarray(card.recipe.z_coord.h_partial)
    bl = np.asarray(card.recipe.z_coord.bottom_level)
    bl_mask = np.where(mask.any(-1), mask.sum(-1) - 1, -1)
    lev = np.arange(mask.shape[-1])
    isbot = (lev[None, None, :] == bl_mask[..., None]) & mask

    a = np.minimum(e3t, np.roll(e3t, -1, axis=1))          # NEMO e3u_3d
    b = 0.5 * (e3t + np.roll(e3t, -1, axis=1))             # legoESM divisor
    r = RN_CD0 * RN_UC0
    pred = ((1.0 - 1.0 / (1.0 + RDT_S * r / b))
            / (1.0 - 1.0 / (1.0 + RDT_S * r / a)))         # per cell

    sel = isbot & (np.abs(nemo) > floor)
    ratio = np.where(sel, lego / np.where(nemo == 0.0, 1.0, nemo), np.nan)
    flat = np.argmin(np.where(sel, ratio, np.inf))
    cell = np.unravel_index(flat, ratio.shape)
    rr = ratio[sel]
    return {
        "kt": kt, "field": field, "floor": floor,
        "cells_scored": int(sel.sum()),
        "measured_median": float(np.median(rr)),
        "measured_p05": float(np.percentile(rr, 5)),
        "measured_min": float(rr.min()),
        "measured_min_cell": [int(x) for x in cell],
        "predicted_at_measured_min_cell": float(pred[cell]),
        "predicted_min_over_bottom_faces": float(np.min(pred[isbot])),
        "nemo_effect_max_at_bottom": float(np.max(np.abs(nemo[isbot]))),
        "nemo_effect_max_above_bottom":
            float(np.max(np.abs(nemo[mask & ~isbot]))),
        "lego_effect_max_at_bottom": float(np.max(np.abs(lego[isbot]))),
        "residual_max_at_bottom":
            float(np.max(np.abs((lego - nemo)[isbot]))),
        "residual_max_above_bottom":
            float(np.max(np.abs((lego - nemo)[mask & ~isbot]))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="VORTEX_SMT2_VEC-zps")
    ap.add_argument("--smt2-residuals", type=Path)
    ap.add_argument("--smt1-residuals", type=Path)
    ap.add_argument("--also", nargs="*", default=["GYRE-zco"],
                    help="cards the statement must be inert on")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    res = {"primary": walk(args.case),
           "inert_check": [walk(c) for c in args.also]}
    if args.smt2_residuals and args.smt1_residuals:
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        res["measured"] = measured(
            args.case, args.smt2_residuals, args.smt1_residuals)
    text = json.dumps(res, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

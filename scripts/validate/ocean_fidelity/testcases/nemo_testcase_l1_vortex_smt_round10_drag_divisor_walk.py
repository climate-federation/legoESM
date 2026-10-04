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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="VORTEX_SMT2_VEC-zps")
    ap.add_argument("--also", nargs="*", default=["GYRE-zco"],
                    help="cards the statement must be inert on")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    res = {"primary": walk(args.case),
           "inert_check": [walk(c) for c in args.also]}
    text = json.dumps(res, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

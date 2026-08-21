#!/usr/bin/env python
"""A or B: is the layer height ALREADY wrong on entry to its update, or
made wrong by it?

WHERE THIS PICKS UP. The 1.3003e-07 m/s of ``w`` is corner-localised,
full size at the end of acoustic sub-step 1, does not scale with the
flow, and shares its spatial footprint with the layer height at that
sub-step. ``Riem_Solver3`` is strictly column-local
(nh_core.F90:42-206), so a corner-localised error out of it is a
statement about its INPUTS, and the only input carrying a horizontal
stencil is the height written by ``update_dz_d`` (nh_utils.F90:194-311),
halo-filled by the duo exchange before the loop. Every grid-metric
family has been transplanted or shown identical, and the four that
disagree do so only at cells no operator reads. Two hypotheses remain:

  A. the height differs BEFORE the update  ->  the halo FILL
  B. same before, differs after            ->  the update OPERATOR

THE HALOS ARE THE POINT. Every comparison in this campaign so far has
scored the compute window. The oracle now dumps ``zh`` over the FULL
PADDED BOX at both capture points, and this scores every region, corner
wedges separately from side strips, because the error is corner-local.

TWO INTERFACES, not one: the layer thickness spans ``zh(:,1)`` and
``zh(:,2)``, so a single interface cannot attach a height error to a
layer.

BOTH SIDES ARE READ, NOT REBUILT. The port's heights come from the
stepper's own ``stage_hook`` -- ``S_nh_before_update_dz_d`` and
``S_nh_after_update_dz_d``, emitted by the code the deck runs. Nothing
here re-implements the acoustic chain; a probe that did would be
measuring a different program.

CONTROLS, each fatal:
  1. the deck must be ``k_split=1``; the dumps carry ``n_map`` in their
     names and a second outer iteration would overwrite them, leaving
     the last iteration wearing the whole step's name;
  2. the IC face map, re-derived, at its floor;
  3. the two capture points must DIFFER, on both sides and at both
     interfaces -- if before equals after, one dump is not where it
     claims and every number below is one reading printed twice;
  4. the BEFORE state's compute-interior difference must be small. Both
     sides descend from identical initial conditions, so a wrong
     dihedral, a wrong face pairing, or a wrong assumption about the
     dump's index order shows up here as O(1) rather than as a halo
     story.

WHAT THIS DOES NOT COVER, stated: interfaces 1 and 2 only, so it cannot
say where in the column the damage is born; the first outer iteration
only (which is the deck's only one, per control 1); and halo cells
neither side's exchange writes are compared as stored -- each row prints
the oracle-side peak so a fill-value band is visible as one.

NO VERDICT IS PRINTED.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from full_step_oracle_parity import (          # noqa: E402
    DIHEDRAL,
    IC_CONTROL_MAX_REL,
    KM,
    N,
    NG,
    ORACLE_ROOT,
    build_port_ic,
    derive_face_map,
    load_oracle,
    port_window,
)
# The region split is the one the stage ladder already uses -- reused,
# not re-derived, so the two instruments cannot drift apart on what
# counts as a corner.
from compare_dyncore_stages import region_masks   # noqa: E402
from nh_preremap_w_parity import read_dump        # noqa: E402

HALO_ROOT = "/burg-archive/glab/users/pg2328/fv3_wsubstep/run_nh_1step"

#: Interfaces bracketing model layer 1; port level indices are 0-based.
LEVELS = {"zh1": 0, "zh2": 1}


def _read_h(run_dir, when, name, tile):
    """One ``hdump<n_map>_<when>_<name>_t<tile>.dat`` as (ni, nj).

    The shared reader stores (k, j, i) to match ``load_oracle``; the
    padded planes here are square, and the port side is handed the same
    (j, i) order, so both sides are transposed identically and the
    comparison never crosses the convention.
    """
    return read_dump(run_dir, 1, f"{when}_{name}", tile,
                     prefix="hdump")[0]


def require_k_split_one(run_dir: str) -> None:
    nml = os.path.join(run_dir, "input.nml")
    if not os.path.exists(nml):
        raise SystemExit(f"no input.nml under {run_dir}: cannot confirm "
                         f"the deck this probe assumes")
    ks = None
    with open(nml) as fh:
        for line in fh:
            bare = line.split("!")[0]
            if "=" in bare and bare.split("=")[0].strip().lower() == "k_split":
                try:
                    ks = int(bare.split("=")[1].strip().rstrip(","))
                except ValueError:
                    pass
    if ks is None:
        raise SystemExit(f"{nml}: k_split not found")
    if ks != 1:
        raise SystemExit(
            f"{nml}: k_split={ks}. The height dumps carry n_map in their "
            f"names, so a second outer iteration overwrites the first and "
            f"this probe would read the last iteration as the whole step.")
    print(f"deck: k_split={ks} (from {nml})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hdump", default=HALO_ROOT)
    ap.add_argument("--ic-run", default=f"{ORACLE_ROOT}/run_nh_zerostep_gfs")
    ap.add_argument("--dt", type=float, default=1920.0)
    ap.add_argument("--n-split", type=int, default=8)
    args = ap.parse_args(argv)

    require_k_split_one(args.hdump)

    from legoesm.core.fv3_native_acoustic_3d import acoustic_substep_3d
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_dynamics import p_var_nonhydrostatic
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    m_a = N + 2 * NG
    ctx["hs6"] = [np.zeros((m_a, m_a), dtype=np.float64) for _ in range(6)]
    ak, bk, ptop, _ks = set_eta_analytic(KM)

    orc_ic = load_oracle(args.ic_run, nh=True)
    state, _sphum = build_port_ic(ctx, ak, bk, nh=True)
    p_ic = port_window(state, ctx)

    # CONTROL 2: the face map at its own floor.
    _c, meta, perm, worst, _pf, _wo = derive_face_map(p_ic, orc_ic)
    print(f"IC face-map control: worst rel {worst:.3e} "
          f"(floor {IC_CONTROL_MAX_REL:.0e})")
    if worst > IC_CONTROL_MAX_REL:
        raise SystemExit("INSTRUMENT CONTROL FAILED: the IC does not "
                         "reproduce the oracle's.")

    # The port's own sub-step 1, with the stepper's stage hook. The two
    # heights come from the code the deck runs, by hook, not by replay
    # of a hand-built chain.
    grabbed: dict = {}

    def hook(name, *rest):
        # The stepper calls this hook with TWO arguments for the
        # hydrostatic stages and THREE (name, tile, payload) for the NH
        # ones. A fixed-arity hook raises on the first stage it does not
        # expect, so this takes both and keeps only what it asked for.
        if name in ("S_nh_before_update_dz_d", "S_nh_after_update_dz_d"):
            tile, payload = rest
            grabbed.setdefault(name, {})[tile] = np.array(payload, copy=True)

    press = [p_var_nonhydrostatic(f["delp"], f["delz"], f["pt"],
                                 ptop=ptop, akap=FV3_KAPPA,
                                 n=N, ng=NG, km=KM) for f in state]
    dt_sub = args.dt / args.n_split
    acoustic_substep_3d(ctx, state, dt_sub, KM, first_substep=True,
                        ptop=ptop, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                        remap_step=False, remap_follows=True,
                        hydrostatic=False, press_out=None,
                        stage_hook=hook)
    for need in ("S_nh_before_update_dz_d", "S_nh_after_update_dz_d"):
        if len(grabbed.get(need, {})) != 6:
            raise SystemExit(
                f"INSTRUMENT CONTROL FAILED: the stepper emitted "
                f"{len(grabbed.get(need, {}))} faces for {need}, not 6; "
                f"the capture is not where this probe assumes.")
    print("port: both capture points emitted on all six faces")

    # CONTROL 3: the two capture points must not be the same state.
    for side in ("port", "oracle"):
        for lvl, k in LEVELS.items():
            same = []
            for pf in range(6):
                if side == "port":
                    a = grabbed["S_nh_before_update_dz_d"][pf][:, :, k]
                    b = grabbed["S_nh_after_update_dz_d"][pf][:, :, k]
                else:
                    ot = perm[pf]
                    a = _read_h(args.hdump, "pre", lvl, ot + 1)
                    b = _read_h(args.hdump, "post", lvl, ot + 1)
                if float(np.abs(a - b).max()) == 0.0:
                    same.append(pf + 1)
            if same:
                raise SystemExit(
                    f"INSTRUMENT CONTROL FAILED: {side} {lvl} is identical "
                    f"before and after the update on faces {same}; one "
                    f"capture is not where it claims.")
    print("both sides: the two capture points hold different states")

    # THE MEASUREMENT.
    rows = []
    for when, key in (("pre", "S_nh_before_update_dz_d"),
                      ("post", "S_nh_after_update_dz_d")):
        for lvl, k in LEVELS.items():
            for pf in range(6):
                ot = perm[pf]
                _transposed, nm, _su, _sv = meta[pf][ot]
                ow = _read_h(args.hdump, when, lvl, ot + 1)
                pw = DIHEDRAL[nm](grabbed[key][pf][:, :, k])
                if pw.shape != ow.shape:
                    raise SystemExit(
                        f"{when} {lvl} face {pf + 1}: port {pw.shape} vs "
                        f"oracle {ow.shape} -- different domains")
                d = np.abs(pw - ow)
                masks = region_masks(d.shape[0], d.shape[1], width=NG)
                rows.append((when, lvl, pf + 1, ot + 1,
                             {n: (float(d[m].max()) if d[m].size else 0.0)
                              for n, m in masks.items()},
                             float(np.abs(ow).max())))

    print("\nzh, port vs oracle, FULL PADDED BOX "
          "(NG=%d rings; 'corner' = the diagonal wedges)" % NG)
    print(" when  level  face->tile   interior       edge        corner"
          "      oracle |peak|")
    for when, lvl, pf, ot, reg, peak in rows:
        print(f" {when:5s} {lvl:5s}  {pf}->{ot}       "
              f"{reg['interior']:.4e}  {reg['edge']:.4e}  "
              f"{reg['corner']:.4e}   {peak:.4e}")

    # CONTROL 4: identical ICs mean the BEFORE interior cannot be O(1).
    pre_int = max(r[4]["interior"] for r in rows if r[0] == "pre")
    pre_scale = max(r[5] for r in rows if r[0] == "pre")
    print(f"\nBEFORE-state compute-interior worst: {pre_int:.4e} "
          f"(scale {pre_scale:.4e})")
    if pre_int > 1.0e-04 * pre_scale:
        raise SystemExit(
            f"INSTRUMENT CONTROL FAILED: the BEFORE interior differs by "
            f"{pre_int:.4e} against a scale of {pre_scale:.4e}. Both sides "
            f"start from the same IC, so this is the mapping -- the "
            f"dihedral, the face pairing, or the dump's index order -- "
            f"not a halo finding.")
    print("\nNO VERDICT HERE. Read the two BEFORE rows against the two "
          "AFTER rows, per region: a difference already present in the "
          "BEFORE halo points at the exchange that fills it; one that "
          "appears only in AFTER points at the update itself. A "
          "difference in the BEFORE INTERIOR would mean neither, and "
          "control 4 above is what stops that being reported as either.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""The SOLAR row of the kt=1 residual, operator to operator: ``tra_qsr``.

WHY.  ``kt1_surface_gate.py`` closed the salt row exactly and left the
temperature row unequal on 296366 wet cells at ``max 2.4e-19`` K/s -- and the
SUB-SURFACE part of that row, which carries no restoring term at all, is
unequal on 288298 cells at the SAME magnitude.  A restoring statement cannot
produce a sub-surface residual, so what is left is the other NEMO routine that
writes the temperature RHS at the surface: ``tra_qsr``.

WHAT NEMO DOES, as COMPILED (``cfgs/DINO/BLD/ppsrc/nemo/traqsr.f90``), with
DINO's namelist resolved (``RUN_TRAJ/namelist_cfg:180-181``
``ln_qsr_2bd=.true.``, ``nn_chldta=0``; ``ln_qsr_rgb`` stays ``.false.`` from
``SHARED/namelist_ref:424``), so ``nqsr = np_2BD`` (``:1167``) and the routine
that runs is ``qsr_2BD`` (``:242`` -> ``:615``):

  ``:653-654``   zz0 = rn_abs*r1_rho0_rcp ;  zz1 = (1-rn_abs)*r1_rho0_rcp
  ``:658-660``   zatt = zz0*EXP(-(gdepw_3d(1)*(1+r3t(Kmm)))*r1_si0)
                      + zz1*EXP(-(gdepw_3d(1)*(1+r3t(Kmm)))*r1_si1)
  ``:664-672``   jk = 1..nk0      BOTH bands, zzatt *= wmask(jk+1)
  ``:676-683``   jk = nk0+1..nkV  VISIBLE BAND ONLY -- the IR term is gone
  ``:670,:681``  qsr_hc(jk) = qsr * ( zatt - zzatt )
  ``:261-265``   pts(Krhs) += z1_2*(qsr_hc_b + qsr_hc)
                              / (e3t_3d(jk)*(1+r3t(Kmm)*tmask(jk)))
                 for jk = 1..NKSR ONLY.  Levels below nksr get NOTHING.
  ``:214-231``   z1_2 = 1 and qsr_hc_b = 0 at ``kt == nit000`` with no
                 restart; 0.5 with the swapped ``qsr_hc_b`` afterwards.

``nk0`` and ``nkV`` are NOT namelist values: ``tra_qsr_init:1179,:1245`` derives
them from the mesh with ``qsr_ext_lev``.  NEMO printed them for this
configuration in its own ``ocean.output``::

    level of infrared extinction       = 2   ref depth = 20.593063338905267 m
    level of visible light extinction  = 22  ref depth = 635.29067417406986 m

and this gate ASSERTS the transcribed function reproduces that pair rather
than hardcoding it (Rule 10).

WHAT THIS GATE SCORES.  ``ttrd_qsr`` is the trend ``tra_qsr`` itself handed to
``trd_tra`` (``:275-278``), i.e. exactly the ``+=`` above, and ``qsr_hc_b`` in
the kt=1 restart is the kt=1 ``qsr_hc`` (``:293`` writes ``qsr_hc`` under the
name ``qsr_hc_b``).  So at kt=1 the two NEMO arrays are related by
``ttrd_qsr = qsr_hc_b / (e3t_3d*(1+r3t*tmask))`` with no free parameter, and
that identity is CHECKED here before either is used as a reference -- Rule 5:
a trend bucket is integrator bookkeeping until its closure is proved.

WHAT THIS GATE CANNOT SEE (Rule 2).  It scores the kernel's returned 3-D rate.
It never calls ``model.step``, so anything the step does with that rate -- the
vertical solve's damping of the level-0 increment, a factor planted at the
consumer -- is invisible here.  And it is a kt=1 gate: ``z1_2`` is 1 there, so
the TWO-STEP AVERAGE (``:229-231``, which needs a carried ``qsr_hc_b``) is
structurally unobservable.  That statement is sized by ``--kt2-dir``, off
NEMO's own two restarts, and is NOT landed.

Usage
-----
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
      python scripts/validate/ocean_fidelity/dino_1226/kt1_qsr_gate.py \\
        [--kt2-dir <the kt=2 trend record>] [--plant] [--oracle-self-test]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from rebuild_nemo_restart import rebuild                        # noqa: E402

DT = 2700.0
DEFAULT_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                   "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")
#: NEMO's own ocean.output for this configuration (both records print it).
NEMO_NK0, NEMO_NKV = 2, 22
NEMO_EXT_DEPTH_M = (20.593063338905267, 635.29067417406986)


def _row(name, lego, nemo, wet):
    d = np.abs(np.asarray(lego) - np.asarray(nemo))[wet]
    n = int((d != 0.0).sum())
    x, y = np.asarray(lego)[wet], np.asarray(nemo)[wet]
    den = float(y @ y)
    ratio = float(x @ y) / den if den else float("nan")
    print(f"  {name:30s}{int(wet.sum()):>9d}{n:>9d}{d.max():13.4e}"
          f"{float(np.sqrt(np.mean(d ** 2))):13.4e}"
          f"{float(np.sqrt(np.mean(y ** 2))):13.4e}{ratio:14.10f}  "
          f"{'AT BAR' if n == 0 else 'DEBT'}")
    return 0 if n == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart-glob", default=DEFAULT_RESTART)
    ap.add_argument("--kt2-dir", default=None)
    ap.add_argument("--ladder", default=None,
                    choices=("static", "nemo_live", "nemo_2bd"),
                    help="score a DIFFERENT DINOConfig.shortwave_penetration_"
                         "ladder than the card selects.  This is how the "
                         "before/after arms are measured with ONE instrument "
                         "(Rule 7); the card's own value is printed either "
                         "way and the cross-check against the applicator is "
                         "skipped when they differ, because then the kernel "
                         "call deliberately is NOT production.")
    ap.add_argument("--plant", action="store_true",
                    help="move ONE wet sub-surface cell of legoESM's solar "
                         "rate by 1 ulp; the gate MUST then fail")
    ap.add_argument("--oracle-self-test", action="store_true",
                    help="replace legoESM's rate by NEMO's own ttrd_qsr; "
                         "every row must read AT BAR")
    a = ap.parse_args()

    R = rebuild(a.restart_glob, ["ttrd_qsr", "qsr_hc_b", "fraqsr_1lev",
                                 "sshn", "sshb"])
    for k in ("ttrd_qsr", "qsr_hc_b"):
        if k not in R:
            raise SystemExit(f"the restart carries no {k}: ln_tra_trd off?")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    import jax.numpy as jnp
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig, nemo_qsr_ext_lev,
        shortwave_penetration_tendency)

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    g = ndm.nemo_dino_mesh()
    wet3 = np.asarray(g.tmask > 0.5)

    print(f"  precision: dz_ref {np.asarray(z.dz_ref).dtype}  z_half_ref "
          f"{np.asarray(z.z_half_ref).dtype}  tmask {np.asarray(g.tmask).dtype}")
    print(f"  shortwave_penetration_ladder = "
          f"{getattr(cfg, 'shortwave_penetration_ladder', None)!r}   "
          f"jerlov_water_type = {cfg.jerlov_water_type!r}   "
          f"rho_0 = {cfg.rho_0!r}  c_p = {cfg.c_p!r}")

    # ---- Rule 10: the derived extinction levels, PRINTED, against NEMO's own
    # ocean.output, before anything downstream uses them.
    # rDt = 2*rn_Dt for the MLF (domain.f90:309-310), which is what
    # tra_qsr_init sees; at rn_Dt the function returns nk0=1.
    nk0, nkv = nemo_qsr_ext_lev(z, np.asarray(g.tmask) > 0.5,
                                rdt=2.0 * DT, rho_0=cfg.rho_0,
                                c_sw=cfg.c_p)
    print(f"\n  nemo_qsr_ext_lev -> nk0 {nk0} (NEMO {NEMO_NK0}), "
          f"nkV {nkv} (NEMO {NEMO_NKV});  ref depths "
          f"{float(np.asarray(z.z_half_ref)[nk0]):.12f} / "
          f"{float(np.asarray(z.z_half_ref)[nkv]):.12f} m "
          f"(NEMO {NEMO_EXT_DEPTH_M[0]:.12f} / {NEMO_EXT_DEPTH_M[1]:.12f})")
    bad = 0
    if (nk0, nkv) != (NEMO_NK0, NEMO_NKV):
        print("  ^^ DISAGREES with NEMO's printed levels -- every row below "
              "is measured on the wrong band structure")
        bad += 1

    # ---- Rule 5: NEMO's own closure between the two arrays, before either
    # is used as a reference.  ttrd_qsr must BE qsr_hc / live e3t at kt=1.
    def O3(k):
        return np.nan_to_num(np.moveaxis(R[k], 0, -1))

    # THE TIME LEVEL (Rule 1d).  tra_qsr divides by ``e3t_3d*(1+r3t(Kmm))``
    # and at ``kt = nit000`` from rest Kmm IS the initial state, whose ssh is
    # identically zero -- so r3t = 0 and the divisor is the REFERENCE
    # thickness.  Using the restart's ``sshn`` instead (the ssh AFTER step 1)
    # is the wrong level and shows up as a 1.5e-10 closure residual; that was
    # this gate's first reading and it is wrong.  Asserted, not assumed.
    from legoesm.ocean.eos import nemo_r3t_stretch
    eta0 = np.asarray(state0.eta.data, dtype=np.float64)
    if np.abs(eta0).max() != 0.0:
        raise SystemExit(
            f"the card's initial ssh is not identically zero (max "
            f"{np.abs(eta0).max():.3e} m), so r3t(Kmm) at kt=1 is not 0 and "
            "this gate's divisor would be wrong")
    stretch = np.asarray(nemo_r3t_stretch(z, state0.eta.data,
                                          state0.H_bathy.data),
                         dtype=np.float64)
    dzr = np.asarray(z.dz_ref, dtype=np.float64)
    ze3t = dzr * stretch[..., None]
    closure = O3("qsr_hc_b") / np.where(ze3t > 0, ze3t, 1.0)
    print("\n  NEMO's OWN closure at kt=1 (traqsr.f90:261-265 with z1_2=1, "
          "qsr_hc_b=0):  ttrd_qsr  ==  qsr_hc / (e3t_3d*(1+r3t*tmask))")
    print(f"  {'row':30s}{'cells':>9s}{'!=':>9s}{'max|d|':>13s}{'rms':>13s}"
          f"{'NEMO rms':>13s}{'ratio':>14s}")
    # NOT GATING: this row measures NEMO against NEMO.  ttrd_qsr is a
    # difference of RHS accumulators and qsr_hc_b is the raw array, so the
    # residual here is the ORACLE's own cancellation floor -- counting it
    # against the model would make the gate unable to reach PASS even on the
    # self-test arm, which a diff reviewer demonstrated.
    _row("FLOOR closure: qsr_hc/e3t vs trd", closure, O3("ttrd_qsr"), wet3)

    # ---- legoESM's own solar kernel, called with the card's own operands.
    # THE FORCING (Rule 10): the card runs ``forcing_annual_cycle=True``, so
    # the applicator does NOT use the annual-mean ``Q_sr_2d`` array -- it
    # rebuilds Q_sr at this step's model time (dino.py:4632-4642).  Feeding
    # the annual mean here reads a 10% ratio error that is the GATE's, not the
    # model's, and that was this gate's first reading.  The reconstruction is
    # then CROSS-CHECKED bit-for-bit against the applicator's own returned
    # rate below, so it cannot drift from production silently.
    q_sr = forcing["Q_sr_2d"]
    if getattr(cfg, "forcing_annual_cycle", False):
        q_sr = jnp.broadcast_to(
            dm.dino_Q_sr_seasonal(forcing["lat_deg_1d"], DT, cfg)[:, None],
            q_sr.shape)
    card_ladder = getattr(cfg, "shortwave_penetration_ladder", "static")
    ladder = a.ladder or card_ladder
    if ladder != card_ladder:
        print(f"\n  ARM: scoring ladder {ladder!r}, the card selects "
              f"{card_ladder!r}")
    kw = {}
    if ladder == "static":
        zhs = None
    else:
        zhs = nemo_r3t_stretch(z, state0.eta.data, state0.H_bathy.data)
    if ladder == "nemo_2bd":
        kw["nemo_2bd_levels"] = (nk0, nkv)
        kw["cell_wet"] = jnp.asarray(wet3)
    dT = np.asarray(shortwave_penetration_tendency(
        sw_down=q_sr,
        z_coord_dz_ref=z.dz_ref, z_coord_z_half_ref=z.z_half_ref,
        jacobian=jnp.ones_like(state0.eta.data),
        config=ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type),
        rho_0=cfg.rho_0, c_sw=cfg.c_p, z_half_stretch=zhs, **kw),
        dtype=np.float64)

    # CROSS-CHECK: the sub-surface levels of the applicator's OWN returned
    # rate are pure solar (the restoring lands on level 0 only,
    # dino.py:4758-4759), so they must equal this kernel call EXACTLY.  If
    # they do not, the reconstruction above is not the production object and
    # nothing below is a fidelity measurement.
    if ladder != card_ladder:
        print("\n  CROSS-CHECK SKIPPED: this arm is not the card's ladder")
        _prod = None
    else:
        _prod = True
    if _prod is not None:
      _, _rate = dm.apply_dino_lat_lon_surface_forcing(
        state0, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
      _prod = np.asarray(_rate[0], dtype=np.float64)
      _cellmask = np.asarray(state0.land_mask.data, dtype=np.float64)[..., None]
      _d = np.abs(dT[..., 1:] * _cellmask - _prod[..., 1:])
      print(f"\n  CROSS-CHECK vs the applicator's own returned rate "
            f"(sub-surface, where it is pure solar): cells!= "
            f"{int((_d != 0).sum())}  max {_d.max():.4e}")
      if int((_d != 0).sum()) != 0:
          print("  ^^ the kernel call above is NOT what production runs; "
                "every row below is about the GATE")
          bad += 1

    if a.oracle_self_test:
        print("\nSELF-TEST: the legoESM side is REPLACED by NEMO's own "
              "ttrd_qsr, and the SCORED reference becomes that same array, so "
              "the arm is a true identity check and must read GATE PASS.  "
              "Scoring it against qsr_hc_b instead would only re-measure the "
              "oracle's own floor.")
        dT = O3("ttrd_qsr").copy()
        closure = O3("ttrd_qsr").copy()
    if a.plant:
        # THE PLANT MUST CREATE A NEW INEQUALITY.  The first version moved a
        # fixed index that was ALREADY unequal, so the gate's verdict and even
        # its printed cell counts were unchanged and the control proved
        # nothing -- a diff reviewer demonstrated it.  The cell is now CHOSEN
        # from the data: the first wet sub-surface cell where the two sides
        # currently agree bit for bit.
        _eq = (dT == closure) & wet3
        _eq[..., 0] = False
        _idx = np.argwhere(_eq)
        if not len(_idx):
            raise SystemExit("no wet sub-surface cell currently agrees, so a "
                             "1-ulp plant cannot create a NEW inequality here")
        j, i, k = (int(v) for v in _idx[0])
        dT = dT.copy()
        _before = int(((dT != closure) & wet3).sum())
        dT[j, i, k] = np.nextafter(dT[j, i, k], np.inf)
        _after = int(((dT != closure) & wet3).sum())
        print(f"\nPLANT ACTIVE: wet cell (j,i,k)=({j},{i},{k}) -- which "
              f"AGREED before -- moved 1 ulp.  Unequal cells {_before} -> "
              f"{_after}; the scored row MUST move and the gate MUST fail.")
        if _after != _before + 1:
            raise SystemExit("the plant did not create exactly one new "
                             "inequality; the control is broken")

    print("\nlegoESM's solar tendency vs NEMO's ttrd_qsr (kt=1)")
    print(f"  {'row':30s}{'cells':>9s}{'!=':>9s}{'max|d|':>13s}{'rms':>13s}"
          f"{'NEMO rms':>13s}{'ratio':>14s}")
    trd = O3("ttrd_qsr")
    # THE REFERENCE THAT IS THE OPERATOR (Rule 5).  ``ttrd_qsr`` is
    # ``pts(Krhs)_after - pts(Krhs)_before`` (traqsr.f90:206, :276), a
    # DIFFERENCE OF RHS ACCUMULATORS -- so it carries NEMO's own cancellation
    # noise, which the closure row above measures at max 8.5e-22 against
    # NEMO's other copy of the same quantity.  ``qsr_hc_b`` in the restart is
    # the RAW ``qsr_hc`` array (:293), with no subtraction, so
    # ``qsr_hc_b / ze3t`` is the operator's own output put through the
    # identical division.  THAT is the row a bit-for-bit claim may rest on;
    # the ttrd row below cannot go below NEMO's own floor.
    bad += _row("solar rate vs qsr_hc_b/e3t", dT, closure, wet3)
    _row("solar rate vs ttrd_qsr (NEMO floor 8.5e-22)", dT, trd, wet3)
    bad += _row("  level 0 only", dT[..., :1], trd[..., :1], wet3[..., :1])
    bad += _row(f"  levels 1..{nk0 - 1} (both bands)", dT[..., 1:nk0],
                trd[..., 1:nk0], wet3[..., 1:nk0])
    bad += _row(f"  levels {nk0}..{nkv - 1} (visible)", dT[..., nk0:nkv],
                trd[..., nk0:nkv], wet3[..., nk0:nkv])
    bad += _row(f"  levels {nkv}.. (NEMO zero)", dT[..., nkv:],
                trd[..., nkv:], wet3[..., nkv:])
    # the implied heat content, against NEMO's own stored array
    # NOT GATING for the same reason: multiplying the rate back by ze3t is
    # its own rounding step, so this row can never be cleaner than that.
    _row("FLOOR implied qsr_hc vs qsr_hc_b", dT * ze3t, O3("qsr_hc_b"), wet3)

    # ---- WHERE THE LAST ULPS LIVE.  A residual that is a CONSTANT RELATIVE
    # offset within a column is the surface flux Q_sr itself (one operand,
    # inherited by every level); one that varies with level is the profile's
    # own arithmetic (the exponentials).  Printed rather than asserted -- it
    # discriminates the two owners without a second probe.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(closure != 0.0, (dT - closure) / closure, np.nan)
    print("\n  relative residual vs qsr_hc_b/e3t, by level (nan = NEMO zero)")
    for k in (0, 1, 2, 5, 10, 21):
        v = rel[..., k][wet3[..., k]]
        v = v[np.isfinite(v)]
        if v.size:
            print(f"    k={k:<3d} median |rel| {np.median(np.abs(v)):10.3e}  "
                  f"max |rel| {np.abs(v).max():10.3e}  "
                  f"spread within a level {float(v.max() - v.min()):10.3e}")

    # ---- what NEMO's bands actually contain, so a zero row is not read as
    # agreement (Rule 3: know the reference's own magnitude).
    print("\n  NEMO's ttrd_qsr by band, wet rms")
    for lo, hi, tag in ((0, 1, "level 0"), (1, nk0, f"1..{nk0-1}"),
                        (nk0, nkv, f"{nk0}..{nkv-1}"), (nkv, 36, f"{nkv}..")):
        v = trd[..., lo:hi]
        m = wet3[..., lo:hi]
        print(f"    {tag:10s} rms {float(np.sqrt(np.mean(v[m] ** 2))):11.4e}  "
              f"nonzero {int((v[m] != 0).sum()):>8d} / {int(m.sum())}")

    # ---- the UNLANDED statement: the two-step average, sized off the record.
    if a.kt2_dir:
        k2 = os.path.join(a.kt2_dir, "DINO_00000002_restart_*.nc")
        if not glob.glob(k2):
            print(f"\n  TWO-STEP AVERAGE: UNMEASURED -- no tiles match {k2}")
            bad += 1
        else:
            R2 = rebuild(k2, ["qsr_hc_b"])
            q1 = O3("qsr_hc_b")
            q2 = np.nan_to_num(np.moveaxis(R2["qsr_hc_b"], 0, -1))
            # At kt=2 NEMO applies 0.5*(qsr_hc(kt=1) + qsr_hc(kt=2)); legoESM
            # applies qsr_hc(kt=2) alone.  The difference is 0.5*(q2 - q1).
            d = 0.5 * (q2 - q1) / np.where(ze3t > 0, ze3t, 1.0)
            n3 = int(wet3.sum())
            rms = float(np.sqrt(np.mean(d[wet3] ** 2)))
            pooled = float(np.sqrt(np.sum((d[wet3] * 2.0 * DT) ** 2) / n3))
            print("\n  SIZE of the UNLANDED two-step average "
                  "(traqsr.f90:229-231,:261-265), off NEMO's own qsr_hc_b at "
                  "kt=1 and kt=2 -- it needs a CARRIED qsr_hc_b and is not "
                  "taken here")
            print(f"    rate rms {rms:.4e} K/s   x rDt {rms * 2.0 * DT:.4e} K"
                  f"   pooled 3-D {pooled:.4e} K")

    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} rows)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

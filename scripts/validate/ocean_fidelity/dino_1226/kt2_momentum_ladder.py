#!/usr/bin/env python
"""NEMO's kt=2 MOMENTUM program on the standalone DINO card, operator by operator.

WHY THIS EXISTS.  The previous round ranked NEMO's own kt=2 momentum trend
buckets by magnitude and read the ranking as an OWNERSHIP claim -- "the u
residual is owned by vertical mixing first, then Coriolis, then the barotropic
pressure gradient".  Two things are wrong with that, and this gate measures
both rather than arguing them.

  1. ``utrd_zdf`` IS A RESIDUAL BUCKET.  NEMO does not diagnose the vertical
     diffusion trend; it diagnoses what is left of the step after every other
     bucket is subtracted::

         ztrdu = ( puu(Kaa) - puu(Kbb) )*r1_Dt - ztrdu
                                      cfgs/DINO/MY_SRC/dynzdf.F90:597

     and ``dyn_zdf`` has ALREADY removed the barotropic mode by then::

         puu(Kaa) = ( puu(Kaa) - puu_b(Kaa) ) * umask
                                      cfgs/DINO/MY_SRC/dynzdf.F90:167-169

     so, in NEMO's own instrumentation's words (``MY_SRC/trddump.F90:55-61``),
     ``true zdf trend = utrd_zdf + uu_b(Naa)*r1_Dt``.  Rule 5: a residual
     bucket can never be an attribution.

  2. ``utrd_spg`` IS NOT A PRESSURE GRADIENT.  It is the change in
     ``uu(Nrhs)`` across ``dyn_spg`` (``dynspg.F90:183-187``), and inside
     ``dyn_spg_ts`` that change is the depth-mean REMOVAL of the whole
     accumulated RHS (``MY_SRC/dynspg_ts.F90:350-353``) plus the split-explicit
     solver's own ``(uu_b(Kaa) - uu_b(Kbb))*r1_Dt`` re-injection (``:1125-1128``).
     A barotropic-mode substitution, not an operator increment.

  3. THE TREND RUN IS NOT THE PRODUCTION OPERATOR for vorticity.  With
     ``l_trddyn`` true, ``dyn_vor`` calls ``vor_een`` TWICE -- ``ncor`` then
     ``nrvm`` (``cfgs/DINO/MY_SRC/dynvor.F90:152,169``) -- where the
     un-instrumented model calls it ONCE with ``ntot`` (``:228``).  So
     ``utrd_pvo`` and ``utrd_rvo`` are two separately-rounded EEN passes and
     neither is comparable on its own to a single-pass ``(f+zeta)`` operator;
     only their SUM is, and only to the rounding of the split.

WHAT IT PRINTS.  Part A is oracle-only: the bucket inventory with each bucket
CLASSIFIED (clean / bookkeeping / residual, each with its citation), the
``utrd_zdf`` decomposition, and the ranking that survives it.  Part B runs one
kt=2 leap-frog step through legoESM's OWN path -- the same bridge
``kt2_leapfrog_gate.py`` uses -- and scores legoESM's per-operator momentum
tendencies against the buckets that Part A says are clean.

THE uu_b(Naa) IDENTITY USED IN PART A.  NEMO's restart does not carry
``uu_b``.  After ``mlf_baro_corr`` (``stpmlf.F90:753-765``) the column mean of
``uu(Naa)`` IS ``uu_b(Naa)`` by construction, so it is recovered from ``un``::

    uu_b = SUM_k( e3u(Kaa)*un*umask ) * r1_hu(Kaa)

and because ``e3u(Kaa) = e3u_0*(1+r3u(Kaa))`` with ``r3u`` depth-INDEPENDENT
(``domzgr_substitute.h90:127``, ``domqco.F90:166-167``), the stretch cancels
between the numerator and ``hu(Kaa)``: the reference ladder ``e3u_0`` gives the
same mean and no sea surface height is needed.  That cancellation is asserted,
not assumed -- ``--check-stretch`` recomputes the mean with a deliberately
perturbed stretch and reports the difference.

THE BAR IS EXACT (Rule 1b): zero cells unequal per scored row.  Rows that are
not at the bar say DEBT.  Rows whose oracle is identically zero say UNMEASURED,
never AT BAR.

Usage
-----
    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
        python scripts/validate/ocean_fidelity/dino_1226/kt2_momentum_ladder.py \
            --run-dir /data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends

NON-VACUITY, BUILT IN RATHER THAN BEHIND A FLAG.  Every scored Part-B row is
recomputed with legoESM's tendency scaled by 1 + 1e-4 and the CHANGE in its rms
is printed in its own column.  A row whose rms does not move is marked
PLANT-DEAD and fails: the number it prints does not depend on legoESM's
operator.  The size is not arbitrary -- a 1e-12 plant was measured to leave the
vorticity pair's rms unchanged to every printed digit, and two of these rows
are SATURATED (every wet cell already differs), so their cell count cannot move
either and the count is not a usable control there.
``--oracle-self-test`` replaces legoESM's side with the oracle bucket itself:
every SCORED row must then be AT BAR and the gate must exit zero, which proves
the comparison can pass at all.  Rows whose oracle is identically zero are
UNMEASURED in both modes and never AT BAR.
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
from rebuild_nemo_restart import rebuild                       # noqa: E402

RN_DT = 2700.0                   # namdom rn_Dt, cfgs/DINO/*/namelist_cfg:116
MESH = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/"
        "mesh_mask.nc")
CERTIFIED_KT1 = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/"
                 "DINO/RUN_FROMREST_KT1")

#: Every momentum bucket NEMO's DINO instrumentation can fill, with the line
#: that writes it and what it actually holds.  Rule 1: the disposition is
#: forced here, not left to whoever reads the table.
BUCKETS = {
    "keg": ("clean", "dynkeg.F90:163 -- dyn_keg increment to Nrhs"),
    "zad": ("clean", "dynzad.F90:124 -- dyn_zad increment to Nrhs"),
    "rvo": ("split", "MY_SRC/dynvor.F90:174 -- vor_een(nrvm), pass 2 of 2"),
    "pvo": ("split", "MY_SRC/dynvor.F90:157 -- vor_een(ncor), pass 1 of 2"),
    "ldf": ("clean", "MY_SRC/dynldf.F90:120 -- dynldf_lev_lap increment"),
    "hpg": ("clean", "MY_SRC/dynhpg.F90:127 -- hpg_sco increment"),
    "spg": ("bookkeeping",
            "dynspg.F90:187 -- DELTA of uu(Nrhs) across dyn_spg_ts: the "
            "depth-mean removal (MY_SRC/dynspg_ts.F90:350-353) plus the "
            "barotropic re-injection (:1125-1128), NOT a pressure gradient"),
    "zdf": ("residual",
            "MY_SRC/dynzdf.F90:597 -- (uu(Kaa)-uu(Kbb))*r1_Dt minus every "
            "other bucket, and uu_b(Kaa) was removed at :167-169"),
    "atf": ("clean", "MY_SRC/dynatf_qco.F90:277 -- but it moves Nnn, not Naa"),
    "tau": ("zero", "iom_put only on this card; ln_drgimp=.TRUE."),
    "bfr": ("zero", "ln_drgimp=.TRUE. -- the drag is on the zdf diagonal"),
    "bfri": ("zero", "ln_drgimp=.TRUE."),
}


def _stats(field, wet):
    a = np.asarray(field)[wet]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(a ** 2))), float(np.abs(a).max())


def _column_mean(field3, e3_0, mask3):
    """NEMO's ``uu_b`` from ``un``: the live-thickness column mean.

    ``field3``/``e3_0``/``mask3`` are ``(nlev, ny, nx)``.  See the module
    docstring for why the reference ladder is the right one here.
    """
    num = (e3_0 * field3 * mask3).sum(axis=0)
    den = (e3_0 * mask3).sum(axis=0)
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)


def part_a(R2, e3u0, e3v0, umask, vmask, check_stretch=False):
    """Oracle-only: classify every bucket, decompose the residual, re-rank."""
    rdt = 2.0 * RN_DT
    r1_dt = 1.0 / rdt
    uw, vw = umask > 0.5, vmask > 0.5
    print(f"\nPART A -- NEMO's own kt=2 momentum buckets (rDt = {rdt})")
    print(f"  {'bucket':10s}{'class':14s}{'u rms':>13s}{'u max':>13s}"
          f"{'v rms':>13s}{'v max':>13s}   what it holds")
    order = []
    for key, (cls, why) in BUCKETS.items():
        un, vn = f"utrd_{key}", f"vtrd_{key}"
        if un not in R2:
            continue
        ur, umx = _stats(R2[un], uw)
        vr, vmx = _stats(R2[vn], vw)
        print(f"  {key:10s}{cls:14s}{ur:13.4e}{umx:13.4e}{vr:13.4e}{vmx:13.4e}"
              f"   {why}")
        order.append((key, cls, ur, vr))

    uu_b = _column_mean(R2["un"], e3u0, umask)
    vv_b = _column_mean(R2["vn"], e3v0, vmask)
    if check_stretch:
        # The claim is that a depth-INDEPENDENT stretch cancels.  Perturb it
        # by 10% and show the mean does not move; a control that is a no-op
        # would prove nothing, so the unperturbed difference is printed too.
        pert = 1.0 + 0.10 * np.random.default_rng(0).random(uu_b.shape)
        uu_b_p = _column_mean(R2["un"], e3u0 * pert[None, :, :], umask)
        print(f"\n  STRETCH CONTROL: a depth-independent 10% column stretch "
              f"moves uu_b by max {np.abs(uu_b_p - uu_b)[uw[0]].max():.3e} "
              f"(it cancels); the stretch itself spans "
              f"{pert.min():.4f}..{pert.max():.4f}")

    bt_u = np.broadcast_to(uu_b * r1_dt, R2["utrd_zdf"].shape)
    bt_v = np.broadcast_to(vv_b * r1_dt, R2["vtrd_zdf"].shape)
    true_u, true_v = R2["utrd_zdf"] + bt_u, R2["vtrd_zdf"] + bt_v
    print("\n  THE RESIDUAL BUCKET, DECOMPOSED "
          "(MY_SRC/trddump.F90:60-61: true zdf = utrd_zdf + uu_b(Naa)*r1_Dt)")
    print(f"  {'term':26s}{'u rms':>13s}{'u max':>13s}{'v rms':>13s}"
          f"{'v max':>13s}")
    for name, fu, fv in (("?trd_zdf as dumped", R2["utrd_zdf"], R2["vtrd_zdf"]),
                         ("the barotropic mode it carries", bt_u, bt_v),
                         ("TRUE vertical-mixing trend", true_u, true_v)):
        ur, umx = _stats(fu, uw)
        vr, vmx = _stats(fv, vw)
        print(f"  {name:26s}{ur:13.4e}{umx:13.4e}{vr:13.4e}{vmx:13.4e}")

    zr_u, _ = _stats(true_u, uw)
    zr_v, _ = _stats(true_v, vw)
    print("\n  THE RANKING THAT SURVIVES, by rms, with the bookkeeping bucket "
          "dropped and the residual replaced by its true part")
    for comp, idx, tru in (("u", 0, zr_u), ("v", 1, zr_v)):
        rank = [(k, (ur, vr)[idx]) for k, c, ur, vr in order
                if c not in ("bookkeeping", "residual", "zero")]
        rank = [(k, v) for k, v in rank if np.isfinite(v) and v > 0.0]
        rank.append(("zdf(true)", tru))
        rank.sort(key=lambda kv: -kv[1])
        print(f"    {comp}: " + "  >  ".join(f"{k} {v:.3e}"
                                             for k, v in rank))
    print("    pvo and rvo are the TWO PASSES of one instrumented EEN call "
          "(MY_SRC/dynvor.F90:152,169); only their sum is comparable to a "
          "single-pass (f+zeta) operator.")
    return {"uu_b": uu_b, "vv_b": vv_b, "true_u": true_u, "true_v": true_v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="the DINO_KT2_TRENDS record directory")
    ap.add_argument("--kt1-dir", default=None)
    ap.add_argument("--part-a-only", action="store_true",
                    help="oracle-side classification and decomposition only; "
                         "runs no model and needs no GPU")
    ap.add_argument("--check-stretch", action="store_true",
                    help="run Part A's stretch-cancellation control")
    ap.add_argument("--oracle-self-test", action="store_true",
                    help="score the oracle against itself; every row MUST be "
                         "AT BAR and the gate MUST exit zero")
    a = ap.parse_args()

    import netCDF4 as nc
    k2 = os.path.join(a.run_dir, "DINO_00000002_restart_*.nc")
    if not glob.glob(k2):
        raise SystemExit(
            f"no tiles match {k2}.\nProduce the record first:\n"
            "  scripts/validate/ocean_fidelity/dino_1226/"
            "nemo_dino_kt2_trends/run.sh")
    trend_names = [f"{c}trd_{k}" for k in BUCKETS for c in ("u", "v")]
    R2 = rebuild(k2, ["un", "vn", "sshn"] + trend_names)
    missing = [n for n in trend_names if n not in R2]
    if missing:
        print(f"  buckets absent from the record: {missing}")

    m = nc.Dataset(MESH)
    e3u0 = np.squeeze(m.variables["e3u_0"][:]).astype(np.float64)
    e3v0 = np.squeeze(m.variables["e3v_0"][:]).astype(np.float64)
    umask = np.squeeze(m.variables["umask"][:]).astype(np.float64)
    vmask = np.squeeze(m.variables["vmask"][:]).astype(np.float64)
    m.close()

    A = part_a(R2, e3u0, e3v0, umask, vmask, check_stretch=a.check_stretch)
    if a.part_a_only:
        return 0

    return part_b(a, R2, A, umask, vmask)


def part_b(a, R2, A, umask, vmask) -> int:
    """legoESM's own kt=2 momentum operators, through the model's own step."""
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    import jax
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as mmod
    from legoesm.ocean.experiments import dino as dm

    kt1_dir = a.kt1_dir or CERTIFIED_KT1
    st, cfg, model, grid, z = _bridge(kt1_dir)

    rdt = 2.0 * RN_DT
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    st2, rate = dm.apply_dino_lat_lon_surface_forcing(
        st, forcing, z, cfg, RN_DT, t_seconds=rdt, return_rate=True)

    # THE SEAM.  ``latlon_cgrid_ocean_baroclinic_tendencies`` IS the function
    # the leap-frog step calls for its momentum RHS.  The spy returns the real
    # result UNCHANGED, so the trajectory is untouched, and separately calls
    # the SAME function object a second time with the SAME arguments plus
    # ``diagnose_momentum=True`` to obtain the per-operator split.  The split
    # is therefore taken on the trajectory's own operands with the
    # trajectory's own operator; the only gap is that it is a second
    # evaluation of a pure function, which is exact.
    #
    # THE ATTRIBUTE THAT HAS TO BE PATCHED is the one the MODEL module
    # resolves, not the defining module's: ocean_model_latlon_cgrid.py:63
    # binds the name at import, so rebinding it on ocean_pe_latlon_cgrid
    # captures NOTHING.  That is not a hypothetical -- the first version of
    # this gate patched the defining module and printed "0 RHS evaluations",
    # which is why the empty-capture branch below is a hard failure and not a
    # quiet zero.
    real = mmod.latlon_cgrid_ocean_baroclinic_tendencies
    captured: list = []

    def spy(*args, **kw):
        out = real(*args, **kw)
        if not kw.get("diagnose_momentum", False):
            kw2 = dict(kw)
            kw2["diagnose_momentum"] = True
            _, diag = real(*args, **kw2)
            # Record the OPERAND, not only the keywords: the leap-frog's two
            # RHS passes are distinguished by the STATE they are handed (the
            # Nnn advective pass and the Nbb dissipative pass), and the model
            # passes ldf_state positionally, so a keyword check cannot tell
            # them apart -- measured, both passes reported ldf_state=None.
            captured.append((diag, dict(kw), np.asarray(args[0].u.data)))
        return out

    mmod.latlon_cgrid_ocean_baroclinic_tendencies = spy
    try:
        with jax.disable_jit():
            model.step(st2, dt=RN_DT, surface_forcing=sf_step,
                       external_tracer_rate=rate)
    finally:
        mmod.latlon_cgrid_ocean_baroclinic_tendencies = real

    print(f"\nPART B -- legoESM's kt=2 momentum operators, captured through "
          f"model.step ({len(captured)} RHS evaluations)")
    if not captured:
        print("  the seam captured NOTHING: the leap-frog step did not reach "
              "latlon_cgrid_ocean_baroclinic_tendencies through this module "
              "attribute.  That is a harness failure, not a result.")
        return 1

    # The leap-frog makes TWO passes (the Nnn advective pass and the Nbb
    # dissipative pass).  The pass whose operands are the NOW level is the one
    # NEMO's dyn_adv/dyn_vor/dyn_hpg buckets are taken at (Kmm = Nnn):
    # MY_SRC/dynvor.F90:152, dynadv.F90:87, dynhpg.F90:117.  It is identified
    # by its ldf_state kwarg being absent, not by its position.
    u_now = np.asarray(st2.u.data)
    u_bb = np.asarray(st2.u_before.data)
    nnn, bbb = [], []
    for i, (diag_i, kw, u_in) in enumerate(captured):
        d_now = float(np.max(np.abs(u_in - u_now)))
        d_bb = float(np.max(np.abs(u_in - u_bb)))
        which = ("NOW (Nnn)" if d_now == 0.0 else
                 "BEFORE (Nbb)" if d_bb == 0.0 else "neither")
        print(f"  pass {i}: operand u is {which}  "
              f"(max|u_in - u_now| {d_now:.3e}, max|u_in - u_before| "
              f"{d_bb:.3e})  dt={kw.get('dt')}")
        if d_now == 0.0:
            nnn.append((i, diag_i))
        elif d_bb == 0.0:
            bbb.append((i, diag_i))
    if len(nnn) != 1:
        print(f"  could not identify ONE now-level pass ({len(nnn)} "
              "candidates); the mapping below would be a guess, so it is "
              "refused.  NEMO takes dyn_adv/dyn_vor/dyn_hpg at Kmm = Nnn "
              "(dynadv.F90:87, MY_SRC/dynvor.F90:152, dynhpg.F90:117), so "
              "the pass this gate scores has to be that one and nothing "
              "else.")
        return 1
    diag = nnn[0][1]
    diag_bb = bbb[0][1] if len(bbb) == 1 else None

    uw, vw = umask > 0.5, vmask > 0.5

    def _arr(x):
        return np.asarray(x.data if hasattr(x, "data") else x)

    # RULE 5 APPLIED TO OUR OWN SIDE.  A per-term split is only usable if the
    # terms sum to the total the model used.  ``total_u`` is captured at the
    # same point; if the split does not close on it, every row below is a
    # residual of unknown composition and the gate says so instead of
    # printing a table.
    for face in ("u", "v"):
        parts = [f for f in diag._fields
                 if f.endswith("_" + face) and not f.startswith("total")]
        ssum = sum(_arr(getattr(diag, f)) for f in parts)
        tot = _arr(getattr(diag, "total_" + face))
        cl = float(np.max(np.abs(ssum - tot)))
        rel = cl / max(float(np.max(np.abs(tot))), 1e-300)
        print(f"  CLOSURE {face}: max|sum(parts) - total_{face}| = {cl:.4e} "
              f"({rel:.2e} of max|total|)  over {len(parts)} slots")

    shapes = {f: np.asarray(getattr(diag, f)).shape for f in diag._fields}
    print("  captured diagnostic slots (Rule 10 -- printed, not assumed): "
          + ", ".join(f"{k}{v}" for k, v in shapes.items()))

    def lego(name, face, src=None):
        d = getattr(diag if src is None else src, name)
        d = np.asarray(d.data if hasattr(d, "data") else d)
        # legoESM carries one redundant face (the periodic image west of
        # cell 0, the closed southern wall); NEMO's un/vn are east/north.
        # Same inverse as nemo_dino_step1_gate.py:275-278.
        d = d[:, 1:, :] if face == "u" else d[1:, :, :]
        return np.moveaxis(d, -1, 0)               # -> (nlev, ny, nx)

    # THE NON-VACUITY CONTROL RUNS ON EVERY ROW, EVERY TIME, rather than
    # behind a flag nobody remembers to pass.  Each scored row is recomputed
    # with legoESM's tendency scaled by 1 + PLANT and the change in its rms is
    # printed; a row whose rms does NOT move is not reading legoESM's operator
    # and FAILS.  The size is 1e-4 and not 1e-12 deliberately: measured, a
    # 1e-12 plant on the vorticity pair moves the rms by 2e-19 against a
    # residual of 8e-12, i.e. below the printed precision -- and two of the
    # rows are SATURATED (every wet cell already differs), so their cell
    # count cannot move either.  A plant that cannot move the number it is
    # planted in proves nothing, which is the whole failure mode.
    PLANT = 1e-4
    rows = [
        # (label, legoESM diagnostic, NEMO bucket sum, note)
        ("keg+hpg", "KE_PGF_{f}", ("keg", "hpg"),
         "legoESM bundles the kinetic-energy gradient and the hydrostatic "
         "pressure gradient in one accumulator; NEMO's two buckets are summed "
         "to match, which is the only comparison this pair admits"),
        ("pvo+rvo", "vortcor_{f}", ("pvo", "rvo"),
         "een_total puts (f+zeta) in one triad; NEMO's instrumented run "
         "splits it into two passes, so only the sum is comparable"),
        ("zad", "vertadv_{f}", ("zad",),
         "vertical momentum advection"),
        ("ldf", "Ah_lap_{f}", ("ldf",),
         "scored on the BEFORE pass, because that is the level NEMO's "
         "dyn_ldf reads (MY_SRC/dynldf.F90 is called with Kbb) and the level "
         "this card's own leap-frog hands its dissipative pass.  The NOW "
         "pass ALSO fills this slot -- the diagnostics capture every term at "
         "its point of computation whether or not the scope override sums it "
         "-- so reading the now-level slot here would have scored a quantity "
         "the trajectory never adds"),
    ]
    print(f"\n  {'row':10s}{'cells!=':>10s}{'max':>13s}{'rms':>13s}"
          f"{'NEMO rms':>13s}{'plant drms':>13s}  verdict")
    bad = 0
    scored = at_bar = 0
    planted: dict[str, float] = {}
    for label, tmpl, keys, note in rows:
        for face, wet in (("u", uw), ("v", vw)):
            name = tmpl.format(f=face)
            if not hasattr(diag, name):
                print(f"  {label+'.'+face:10s}{'-':>10s}{'-':>13s}{'-':>13s}"
                      f"{'-':>13s}  UNMEASURED (no {name} slot)")
                bad += 1
                continue
            oracle = sum(R2[f"{face}trd_{k}"] for k in keys)
            orms, _ = _stats(oracle, wet)
            src = diag_bb if label == "ldf" else diag
            if src is None:
                print(f"  {label + '.' + face:10s}{'-':>10s}{'-':>13s}"
                      f"{'-':>13s}{'-':>13s}  UNMEASURED (no before pass)")
                bad += 1
                continue
            raw = oracle if a.oracle_self_test else lego(name, face, src)
            mine = raw
            d = np.abs(mine - oracle)[wet]
            d = d[np.isfinite(d)]
            n = int((d != 0.0).sum())
            dp = np.abs(raw * (1.0 + PLANT) - oracle)[wet]
            dp = dp[np.isfinite(dp)]
            if not np.isfinite(orms) or orms == 0.0:
                print(f"  {label+'.'+face:10s}{n:>10d}"
                      f"{(d.max() if d.size else np.nan):13.4e}"
                      f"{float(np.sqrt(np.mean(d**2))) if d.size else np.nan:13.4e}"
                      f"{orms:13.4e}  UNMEASURED (oracle identically zero)")
                bad += 1
                continue
            rms0 = float(np.sqrt(np.mean(d ** 2)))
            rms1 = float(np.sqrt(np.mean(dp ** 2))) if dp.size else float("nan")
            moved = np.isfinite(rms1) and rms1 != rms0
            verdict = "AT BAR" if n == 0 else "DEBT"
            if not moved:
                verdict += " /PLANT-DEAD"
            bad += (n != 0) or (not moved)
            scored += 1
            at_bar += (n == 0)
            planted[f"{label}.{face}"] = rms0
            print(f"  {label+'.'+face:10s}{n:>10d}{d.max():13.4e}"
                  f"{rms0:13.4e}{orms:13.4e}{rms1 - rms0:13.3e}"
                  f"  {verdict}")
    for label, _, _, note in rows:
        print(f"    {label}: {note}")

    print("\n  NOT SCORED HERE, and why -- Rule 1 forces the disposition")
    print("    spg : bookkeeping (see Part A); there is no legoESM operator "
          "whose increment it is")
    print("    zdf : a residual bucket; its true part needs the implicit "
          "solve's own before/after pair, which this seam does not hold")
    print("    atf : moves Nnn, not Naa; a different seam")
    if a.oracle_self_test:
        # The self-test proves the comparison CAN pass.  UNMEASURED rows
        # (oracle identically zero) are not scored either way, so they are
        # excluded from its verdict and still reported above.
        ok = scored > 0 and at_bar == scored
        print(f"\n  SELF-TEST: {at_bar} of {scored} scored rows AT BAR "
              f"-- {'PASS' if ok else 'FAILED, so no DEBT row below it means '
                                      'anything'}")
        print(f"\nGATE {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    print(f"\n  the 'plant drms' column is the built-in non-vacuity control: "
          f"legoESM's tendency scaled by 1 + {PLANT:g}.  A scored row whose "
          "drms is 0 is marked PLANT-DEAD and fails, because the number it "
          "prints does not depend on legoESM's operator at all.")
    print(f"\nGATE {'PASS' if bad == 0 else f'FAIL ({bad} rows)'}")
    return 0 if bad == 0 else 1


def _bridge(kt1_dir):
    """NEMO's kt=1 restart onto legoESM's Kbb/Kmm, as kt2_leapfrog_gate does."""
    import importlib.util as ilu
    import netCDF4  # noqa: F401  (imported for the same reason rebuild needs it)
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity.nemo_io import (
        NemoBeforeState, NemoState, read_nemo_mesh_mask)
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_nemo_to_legoesm_topo, bridge_before_state_topo)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    k1 = os.path.join(kt1_dir, "DINO_00000001_restart_*.nc")
    if not glob.glob(k1):
        raise SystemExit(f"no kt=1 tiles match {k1}")
    R1 = rebuild(k1, ["tn", "sn", "sshn", "un", "vn", "tb", "sb", "sshb",
                      "ub", "vb", "utau_b", "vtau_b", "en", "avm_k", "avt_k",
                      "dissl"])
    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    model = LatLonCGridOceanModel(grid, z, mc)
    g = read_nemo_mesh_mask(MESH, nn_hls=0)

    def O3(k):
        return np.moveaxis(R1[k], 0, -1)

    now = NemoState(T=O3("tn"), S=O3("sn"), u=O3("un"), v=O3("vn"),
                    ssh=R1["sshn"], rhd=None)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=cfg.omega)
    before = NemoBeforeState(T=O3("tb"), S=O3("sb"), u=O3("ub"), v=O3("vb"),
                             ssh=R1["sshb"], tau_x=R1.get("utau_b"),
                             tau_y=R1.get("vtau_b"))
    st = bridge_before_state_topo(br._replace(state=br.state), g, before,
                                  periodic_i=True)
    spec = ilu.spec_from_file_location(
        "kamm_twin_90d", os.path.join(_HERE, "kamm_twin_90d.py"))
    ktw = ilu.module_from_spec(spec)
    spec.loader.exec_module(ktw)
    carry = (getattr(cfg, "tke_preclosure_coeff_source", None)
             == "carried_previous_step")
    st = ktw.bridge_tke_from_restart(
        st, O3("en"), br.land_mask,
        restart_avm=(O3("avm_k") if carry else None),
        restart_avt=(O3("avt_k") if carry else None),
        restart_dissl=(O3("dissl") if carry else None))
    if st.u_before is None:
        raise SystemExit("the kt=1 restart did not carry the before levels; "
                         "this would be an Euler step, not the leap-frog one")
    return st, cfg, model, grid, z


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""NEMO's kt=2 LEAP-FROG step on the standalone DINO card.

Preregistered in ``PREREG_kt1_kmm_stretch_and_rdt.md`` (read it first).

WHY kt=2.  Three statements this branch transcribes are DEGENERATE on the
from-rest first step, so every number measured so far is silent about them:

  rDt         ``domain.f90:288`` sets ``rDt = 2*rn_Dt`` for the modified
              leap-frog; ``stpmlf.f90:131-133`` reduces it to ``rn_Dt`` while
              ``l_1st_euler`` and ``:617-620`` restores it at the END of that
              step.  At kt=1 a card that used ``rn_Dt`` everywhere would score
              identically.  ``traldf_iso.f90:829-830`` reads ``rDt`` and
              ``r1_Dt``, so this is where that statement can be wrong.

  ssh_atf     ``stpmlf.f90:425`` Asselin-filters ``ssh(:,:,Nnn)`` BEFORE
              ``tra_ldf`` at ``:504``, while ``r3t(:,:,Nnn)`` is not refreshed
              until ``:571``.  The filter is guarded ``IF(.NOT.l_1st_euler)``
              (``sshwzv.f90:443``), so at kt=1 the two "now" heights coincide
              and the ``Kmm``-stretch transcription is untested.

  akz         the stabilising correction fires on 0 of 342134 wet cells at
              kt=1, so ``traldf_iso_a33``'s own ``e3w`` and the implicit half
              of the explicit/implicit split are multiplied by nothing.

WHAT IT DOES.  Feeds legoESM NEMO's OWN kt=1 restart as the BEFORE (Kbb) and
NOW (Kmm) levels, runs ONE step through the model's own path, and scores the
after-state against NEMO's kt=2 restart -- plus the ``tra_ldf`` tendency
against NEMO's own ``ttrd_ldf``/``strd_ldf`` at kt=2.

THE BAR IS EXACT: zero cells unequal, per field (Rule 1b).  Rows that are not
at the bar say DEBT.

Usage
-----
    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
        python scripts/validate/ocean_fidelity/dino_1226/kt2_leapfrog_gate.py \\
            --run-dir /data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt2_trends

``--run-dir`` is what ``nemo_dino_kt2_trends/run.sh`` produced: it must hold
BOTH ``DINO_00000001_restart_*.nc`` (the before/now state) and
``DINO_00000002_restart_*.nc`` (the after state and the trends).

NON-VACUITY.  ``--plant-euler`` drops the before-level fields so the step takes
the EULER branch (``rdt = rn_Dt`` instead of ``2*rn_Dt``).  Every row must then
move; a gate whose rows survive that plant is not measuring a leap-frog step at
all, which is the one thing this record exists to test.
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

STATE_FIELDS = ("tn", "sn", "sshn", "un", "vn", "tb", "sb", "sshb", "ub", "vb")
TREND_FIELDS = ("ttrd_ldf", "strd_ldf", "ttrd_tot", "strd_tot")


def _score(name, lego, nemo, wet, step):
    """One row: cells unequal at the EXACT bar, plus the size of NEMO's own
    step for that field so the residual is read against the motion it is a
    fraction of (Rule 3)."""
    d = np.abs(np.asarray(lego) - np.asarray(nemo))[wet]
    n = int((d != 0.0).sum())
    rms = float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")
    print(f"  {name:14s}{n:>10d}{d.max() if d.size else np.nan:14.4e}"
          f"{rms:14.4e}{step:14.4e}  {'AT BAR' if n == 0 else 'DEBT'}")
    return 0 if n == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="the DINO_KT2_TRENDS record directory")
    ap.add_argument("--plant-euler", action="store_true",
                    help="drop the before levels so the step takes the EULER "
                         "branch (rdt = rn_Dt); every row MUST then move")
    a = ap.parse_args()

    k1 = os.path.join(a.run_dir, "DINO_00000001_restart_*.nc")
    k2 = os.path.join(a.run_dir, "DINO_00000002_restart_*.nc")
    for pat in (k1, k2):
        if not glob.glob(pat):
            raise SystemExit(
                f"no tiles match {pat}.\n"
                "Produce the record first:\n"
                "  scripts/validate/ocean_fidelity/dino_1226/"
                "nemo_dino_kt2_trends/run.sh")
    R1 = rebuild(k1, list(STATE_FIELDS))
    R2 = rebuild(k2, list(STATE_FIELDS) + list(TREND_FIELDS))

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    import jax
    import numpy as _np
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.fidelity.nemo_io import (
        read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before)
    from legoesm.ocean.fidelity.nemo_state_bridge import (
        bridge_nemo_to_legoesm_topo, bridge_before_state_topo)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gmmod

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    model = LatLonCGridOceanModel(grid, z, mc)

    # The BEFORE/NOW state is NEMO's own kt=1 restart, bridged the way every
    # other restart-seeded probe on this branch bridges it -- not rebuilt.
    mesh = os.path.join(a.run_dir, "mesh_mask.nc")
    if not os.path.exists(mesh):
        mesh = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                "RUN_FROMREST_KT1/mesh_mask.nc")
    g = read_nemo_mesh_mask(mesh, nn_hls=0)
    one = os.path.join(a.run_dir, "DINO_00000001_restart.nc")
    if not os.path.exists(one):
        one = sorted(glob.glob(k1))[0]
    now = read_nemo_restart(one, nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, now, periodic_i=True, full_step=True,
                                     omega=cfg.omega)
    before = read_nemo_restart_before(one, nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before,
                                  periodic_i=True)
    st_lf = st
    if a.plant_euler:
        print("EULER PLANT ACTIVE: this run scores the leap-frog step AND an "
              "arm with the before levels dropped (rdt = rn_Dt); every scored "
              "row MUST differ between them, or the gate is not measuring a "
              "leap-frog step at all.")

    # Rule 10: PRINT the timestep the step will actually use, do not infer it.
    euler = st_lf.u_before is None
    rdt = (1.0 if euler else 2.0) * RN_DT
    print(f"\nSTEP: euler_start={euler}  rdt={rdt}  "
          f"(NEMO kt=2: rDt = 2*rn_Dt = {2 * RN_DT})")
    if rdt != 2 * RN_DT:
        print("  ^^ the step is NOT the leap-frog step this record scores; "
              "the kt=1 restart did not carry the before levels")
        return 1
    st = st_lf

    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    st2, rate = dm.apply_dino_lat_lon_surface_forcing(
        st, forcing, z, cfg, RN_DT, t_seconds=2 * RN_DT, return_rate=True)

    real = gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid
    calls: list = []

    def spy(q, *aa, **kw):
        r = real(q, *aa, **kw)
        out = r[0] if (isinstance(r, tuple) and kw.get("return_bolus")) else r
        calls.append((_np.asarray(q), _np.asarray(out), (q,) + tuple(aa),
                      dict(kw)))
        return r

    gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = spy
    try:
        with jax.disable_jit():
            after = model.step(st2, dt=RN_DT, surface_forcing=sf_step,
                               external_tracer_rate=rate)
    finally:
        gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = real

    gmesh = ndm.nemo_dino_mesh()
    wet3 = gmesh.tmask > 0.5
    wet2 = wet3[..., 0]

    def O3(k, R):
        return _np.moveaxis(R[k], 0, -1)

    rows: dict[str, float] = {}
    print("\nSTATE at the end of kt=2 (legoESM vs NEMO's kt=2 restart)")
    print(f"  {'field':14s}{'cells!=':>10s}{'max':>14s}{'rms':>14s}"
          f"{'NEMO step':>14s}")
    bad = 0
    for lego_name, nemo_name, w in (("T", "tn", wet3), ("S", "sn", wet3),
                                    ("eta", "sshn", wet2),
                                    ("u", "un", wet3), ("v", "vn", wet3)):
        obj = getattr(after, lego_name).data
        nemo = O3(nemo_name, R2) if w is wet3 else R2[nemo_name]
        prev = O3(nemo_name, R1) if w is wet3 else R1[nemo_name]
        step = float(_np.abs(nemo - prev)[w].max())
        rows[lego_name] = float(
            _np.sqrt(_np.mean((_np.asarray(obj) - nemo)[w] ** 2)))
        bad += _score(lego_name, obj, nemo, w, step)

    print("\ntra_ldf at kt=2 (legoESM's own tendency vs NEMO's ttrd_ldf)")
    if not calls:
        print("  UNMEASURED: the isoneutral operator was never called")
        bad += 1
    else:
        for tag, key in (("T", "ttrd_ldf"), ("S", "strd_ldf")):
            if key not in R2:
                print(f"  UNMEASURED {key}: the kt=2 restart does not carry "
                      "it (was ln_tra_trd on?)")
                bad += 1
                continue
            # The operator is fed the BEFORE (Kbb) tracer, not the now one
            # (stpmlf.f90:504 passes Kbb=Nbb), so "nearest to T-now" is not
            # "NEMO's tra_ldf call".  Identify by an EXACT wet-masked match
            # against the Kbb field the step carries, and REFUSE when nothing
            # matches -- kt=1's own note records that scoring the wrong call
            # moved a ratio from 0.9967 to 5.53.
            src = _np.asarray(getattr(st2, tag + "_before").data
                              if getattr(st2, tag + "_before", None) is not None
                              else getattr(st2, tag).data)
            hits = [i for i in range(len(calls))
                    if _np.array_equal(calls[i][0][wet3], src[wet3])]
            print(f"  call identification [{tag}]: "
                  + "  ".join(f"call {i}: max|q-Kbb| "
                              f"{_np.abs(calls[i][0] - src)[wet3].max():.3e}"
                              for i in range(len(calls))))
            if len(hits) != 1:
                print(f"  ^^ {len(hits)} captured calls match the Kbb {tag} "
                      "exactly; the scored call is not identified and no row "
                      "below is about NEMO's tra_ldf")
                bad += 1
                continue
            idx = hits[0]
            tend = calls[idx][1]
            nemo = _np.nan_to_num(O3(key, R2))
            den = float(nemo[wet3] @ nemo[wet3])
            ratio = float(tend[wet3] @ nemo[wet3]) / den if den else float("nan")
            res = float(_np.sqrt(_np.mean((tend[wet3] - nemo[wet3]) ** 2)))
            n = int((tend[wet3] != nemo[wet3]).sum())
            print(f"  {tag} tra_ldf [call {idx}]  ratio {ratio:.9f}  "
                  f"res rms {res:.4e}  NEMO rms "
                  f"{float(_np.sqrt(_np.mean(nemo[wet3] ** 2))):.4e}  "
                  f"cells!= {n}/{int(wet3.sum())}  "
                  f"{'AT BAR' if n == 0 else 'DEBT'}")
            if n:
                bad += 1
        kw = calls[0][3]
        print(f"  the dt the operator received: {kw.get('dt')!r}  "
              f"(NEMO rDt at kt=2 = {2 * RN_DT})")
        if kw.get("dt") != 2 * RN_DT and not a.plant_euler:
            print("  ^^ the operator is NOT on NEMO's rDt "
                  "(traldf_iso.f90:829-830)")
            bad += 1

    # The akz census: this is the ONE thing the kt=1 record structurally could
    # not see, so it is printed whether or not it is at the bar.
    kwd = {**calls[0][3], "return_diagnostics": True,
           "return_operand_diagnostics": True}
    r = real(*calls[0][2], **kwd)
    # The DINO card resolves gm_bolus_advection='through_fct', so the captured
    # call carries return_bolus=True and the operator returns a THREE-tuple
    # when the diagnostics are also asked for (gm_redi_latlon_cgrid.py:2787).
    diags = r[-1]
    if not isinstance(diags, dict):
        print(f"\n  akz census FAILED: the operator returned "
              f"{type(r).__name__} of length {len(r)}, whose last member is "
              f"{type(diags).__name__}, not the diagnostics dict")
        bad += 1
    elif "zfw_operands" not in diags:
        print("\n  akz census FAILED: no 'zfw_operands' in the diagnostics -- "
              "that block exists only when msc_stabilize is on "
              "(gm_redi_latlon_cgrid.py:2764), so the stabilising correction "
              "is OFF while NEMO's namelist_cfg:267 turns it on")
        bad += 1
    else:
        akz = _np.asarray(diags["zfw_operands"]["akz"])
        w = wet3[..., :akz.shape[-1]]
        fired = int((akz[w] > 0.0).sum())
        tot = int(w.sum())
        print(f"\n  akz branch census at kt=2: the stabiliser fires on "
              f"{fired}/{tot} wet cells -- "
              + ("the A33 e3w (traldf_iso.f90:831-833) and the implicit half "
                 "of the split are now SCORED by the rows above"
                 if fired else
                 "still UNMEASURED: the branch is empty at kt=2 as well, so "
                 "those two operands remain multiplied by nothing"))

    if a.plant_euler:
        st_eu = st_lf._replace(u_before=None, v_before=None, T_before=None,
                               S_before=None, eta_before=None)
        st2e, ratee = dm.apply_dino_lat_lon_surface_forcing(
            st_eu, forcing, z, cfg, RN_DT, t_seconds=2 * RN_DT,
            return_rate=True)
        with jax.disable_jit():
            after_e = model.step(st2e, dt=RN_DT, surface_forcing=sf_step,
                                 external_tracer_rate=ratee)
        print("\nPLANT (euler arm): every scored row must MOVE")
        moved = 0
        for lego_name, nemo_name, w in (("T", "tn", wet3), ("S", "sn", wet3),
                                        ("eta", "sshn", wet2),
                                        ("u", "un", wet3), ("v", "vn", wet3)):
            nemo = O3(nemo_name, R2) if w is wet3 else R2[nemo_name]
            r = float(_np.sqrt(_np.mean(
                (_np.asarray(getattr(after_e, lego_name).data) - nemo)[w]
                ** 2)))
            ok = r != rows[lego_name]
            moved += int(ok)
            print(f"  {lego_name:6s} leap-frog rms {rows[lego_name]:.6e}  "
                  f"euler rms {r:.6e}  {'MOVED' if ok else 'DID NOT MOVE'}")
        if moved != len(rows):
            print("  ^^ a row that does not move between a leap-frog and an "
                  "Euler step is not measuring the timestep, and every rDt "
                  "row above rests on it")
            return 1
        print("  every row moved, as required")
        return 0
    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} rows)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

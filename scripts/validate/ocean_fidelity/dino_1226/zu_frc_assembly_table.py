#!/usr/bin/env python
"""#1455 sec-D JOB 2 -- the forcing-assembly read: ordered alignment table of
NEMO ``zu_frc`` (MY_SRC/dynspg_ts.F90) vs legoESM ``F_slow_{u,v}``
(ocean_model_latlon_cgrid.py), one row per statement, MATCH/DIFF, file:line.

The table itself is a READ (below, printed).  This probe QUANTIFIES the ONE
statement-level DIFF the read finds -- the depth-mean THICKNESS WEIGHTING:

  NEMO (key_qco, DINO): dynspg_ts.F90:337
      zu_frc = SUM_k( e3u_0 * puu(:,:,:,Krhs) * umask ) * r1_hu_0     [REST e3u_0]
  legoESM: ocean_model_latlon_cgrid.py:3465-3482
      h_k_pre = compute_layer_thickness(state.eta, ...)   [LIVE, from NOW eta]
      F_slow_u = SUM_k( du_dt * h_u_pre ) / H_u_pre        [LIVE h_u_pre]

NEMO weights the 3-D momentum trend's depth mean by the REST thickness e3u_0 /
r1_hu_0 (the key_qco branch, :337 -- verified: cpp_DINO.fcm has key_qco, and
namelist runs ln_bt_fw=.false.); legoESM weights by the LIVE column thickness
h_u_pre built from the NOW ssh.  Under key_qco NEMO deliberately uses the REST
metric here (the comment at :313 "e3. are substitute by 1D arrays").

MEASUREMENT: hand legoESM's OWN du_dt the two weightings and diff the resulting
depth means -- ISOLATES the thickness-weighting statement (same tendency both
sides).  Then compare that DIFF's magnitude & STRUCTURE to the JOB-1 per-face
residual (0.88 m^2/s max, wall-concentrated).  If they match -> the thickness
weighting owns the transport gap; if not -> the gap is elsewhere (drag/Coriolis
substep dynamics, see the CLEAN-table escalation).

fp64 via run_fp64.py; LEGOESM_NEMO_E3T=both; day-0 bit-identity; card printed.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS, os.path.dirname(_THIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hu_avg_perface_diff as J   # reuse mesh/loaders/report (JOB 1)


TABLE = """
================================================================================
JOB 2 ALIGNMENT TABLE -- zu_frc (NEMO) vs F_slow (legoESM), DINO card
  NEMO: MY_SRC/dynspg_ts.F90  #else (MLF, non-RK3) branch -- DINO runs this
        (key_qco defined; ln_bt_fw=.false.; ln_apr_dyn=.false.; ln_drgimp=.true.
         but dyn_drg_init's ln_isfcav/ln_drgice_imp=F so only BOTTOM drag fires)
  lego: ocean_model_latlon_cgrid.py _step_impl, card=nemo_dino_kamm_mlf
        (barotropic_forcing_centred=T, barotropic_drag_substep=T,
         barotropic_coriolis_split="live", barotropic_een_seed="nemo_kmm")
--------------------------------------------------------------------------------
 N | NEMO statement (file:line)                    | legoESM (file:line)          | verdict
--------------------------------------------------------------------------------
 1 | zu_frc = SUM_k e3u_0*puu(Krhs)*umask *r1_hu_0 | F_slow_u = SUM(du_dt*h_u_pre)| DIFF: NEMO
   |   (:337, key_qco REST thickness e3u_0/r1_hu_0)|   /H_u_pre  (:3465-3482,     |  REST e3u_0;
   |                                               |    LIVE h from NOW eta)      |  lego LIVE h  [Q1]
 2 | puu(Krhs) -= zu_frc  (baroclinic split, :351) | du_dt_pert = du_dt-F_slow    | MATCH (:3517)
 3 | dyn_cor_2D_init(Kmm); dyn_cor_2D(puu_b(Kmm))  | barotropic_coriolis_een_pre_ | MATCH: 2D EEN
   |   zu_frc -= zu_trd*ssumask  (:365-369,        |   step(...) Kmm coeffs;       |  Coriolis removed
   |    remove 2D Coriolis, re-applied live)       |   F_slow -= _cor_u_sub        |  Kmm, re-applied
   |                                               |   (:3946-3964)               |  live [nemo_kmm]
 4 | dyn_drg_init: pu_RHSi += r1_hu(Kmm)*r1_2*     | barotropic_drag_substep:     | MATCH (centred
   |  (rCdU_bot(i+1)+rCdU_bot(i))*(puu(ikbu,Kbb)   |  F_slow -= r_u_bt/H_u_pre*    |  Kbb residual,
   |  -puu_b(Kbb))  (ln_bt_fw=F -> Kbb, :51-60)    |  (u_bot-U_bar) w/ u_before    |  Kmm rate/H)
   |                                               |  (:3542-3564)                |  [Q4]
 5 | ln_apr_dyn: atmospheric-pressure grad add     | (none)                       | MATCH: both OFF
   |   (:407-421)  -- ln_apr_dyn=.FALSE. for DINO  |                              |  (ln_apr_dyn=F)
 6 | wind: zu_frc += r1_rho0*r1_2*(utau_b+utauU)   | surface_stress_implicit path | see #1460
   |  *r1_hu(Kmm)  (ln_bt_fw=F CENTRED, :443)      |  OR carried in du_dt         |  (tau_prev sign
   |                                               |                              |  bug, RESOLVED)
 7 | ssh_frc = r1_rho0*r1_2*(emp+emp_b) (:519 else)| F_slow_eta (freshwater)      | MATCH: emp=0
   |   -- emp==0 for DINO (no rnf/isf)             |  (:3784-3824)                |  DINO (=0)
--------------------------------------------------------------------------------
 EXCLUDED from zu_frc (NEMO), verified: ln_rnf=F, ln_isf=F, ln_bdy=F, key_agrif
 absent, ln_wd_dl_bc=F.  A2 biharmonic (lego B_h_barotropic) / slow_forcing_ab2:
 OFF on this card (default 0 / False) -> not traced.
================================================================================
"""


def main():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    print(TABLE)
    print(f"control dtype = {get_policy().control}  E3T={os.environ.get('LEGOESM_NEMO_E3T')!r}")

    m = J._mesh()
    # bridge one step and capture du_dt + h_k_pre the way _step_impl builds them.
    # Rather than re-plumb the model, reconstruct the two thickness weightings on
    # legoESM's OWN state (NOW eta) with a synthetic uniform du_dt AND with the
    # real 3-D tendency captured via a hook -- ISOLATE the weighting.
    import jax
    import multistep_replay as mr
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing, dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays)
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface, min_cell_to_vface)

    mr.IC_STEP = 230400
    jax.clear_caches()
    g, br, cfg, st0 = mr.build_replay_ic()
    print(f"\ncard: coriolis_split={getattr(cfg,'barotropic_coriolis_split',None)!r} "
          f"drag_substep={getattr(cfg,'barotropic_drag_substep',None)!r} "
          f"forcing_centred={getattr(cfg,'barotropic_forcing_centred',None)!r}")

    # capture legoESM's du_dt (the momentum tendency) via a hook on tendencies().
    cap = {}
    model = LatLonCGridOceanModel(br.geometry, br.z_coord,
                                  dino_lat_lon_model_config(br.geometry, cfg)[0])
    _orig_tend = model.tendencies

    def _tend_hook(*a, **k):
        t = _orig_tend(*a, **k)
        if "du_dt" not in cap and getattr(t, "du_dt", None) is not None:
            # first (stage-1) call is the one feeding F_slow
            jax.experimental.io_callback(
                lambda du, dv: cap.update(du_dt=np.asarray(du), dv_dt=np.asarray(dv)),
                None, t.du_dt.data, t.dv_dt.data)
        return t
    model.tendencies = _tend_hook

    # ALSO capture the FULLY-ASSEMBLED F_slow_u/F_slow_v handed to the
    # barotropic call (line 4022) -- the DIRECT input to NEMO's zu_frc.
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as omod
    _orig_baro = omod.barotropic_substeps_latlon_cgrid

    def _baro_hook(*a, **k):
        if "F_slow_u" not in cap and k.get("F_slow_u") is not None:
            jax.experimental.io_callback(
                lambda fu, fv: cap.update(F_slow_u=np.asarray(fu),
                                          F_slow_v=np.asarray(fv)),
                None, k["F_slow_u"], k["F_slow_v"])
        return _orig_baro(*a, **k)
    omod.barotropic_substeps_latlon_cgrid = _baro_hook

    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    DT = J.RN_DT
    st, rate = apply_dino_lat_lon_surface_forcing(
        st0, forcing, br.z_coord, cfg, DT, t_seconds=DT, return_rate=True)
    _ = model.step(st, DT, surface_forcing=None, external_tracer_rate=rate)
    omod.barotropic_substeps_latlon_cgrid = _orig_baro
    if "du_dt" not in cap:
        raise SystemExit("du_dt not captured")
    du_dt = cap["du_dt"]; dv_dt = cap["dv_dt"]
    print(f"captured du_dt shape={du_dt.shape} dtype={du_dt.dtype}")

    # LIVE weighting (legoESM :3465-3482): h from NOW eta
    eta_now = np.asarray(st0.eta.data)
    Hb = np.asarray(st0.H_bathy.data)
    h_k_live = np.asarray(compute_layer_thickness(
        st0.eta.data, st0.H_bathy.data, br.z_coord,
        min_water_column_m=model.config.min_water_column_m))
    h_u_live = np.asarray(min_cell_to_uface(h_k_live))
    h_v_live = np.asarray(min_cell_to_vface(h_k_live, br.geometry))

    # map du_dt to (nj,ni,jpk) at NEMO cell faces (drop the ni+1 west face)
    def _to_cell_u(a):
        return a[:, 1:, :] if a.shape[1] == J.NI + 1 else a
    def _to_cell_v(a):
        return a[1:, :, :] if a.shape[0] == J.NJ + 1 else a
    du = _to_cell_u(du_dt); dv = _to_cell_v(dv_dt)
    huL = _to_cell_u(h_u_live[..., None] if h_u_live.ndim == 2 else h_u_live)
    hvL = _to_cell_v(h_v_live[..., None] if h_v_live.ndim == 2 else h_v_live)
    huL = huL if huL.ndim == 3 else h_u_live
    # recompute h_u_live at cell faces cleanly
    huL = _to_cell_u(h_u_live); hvL = _to_cell_v(h_v_live)

    HuL = np.maximum(huL.sum(-1), 1e-10)
    HvL = np.maximum(hvL.sum(-1), 1e-10)
    Fu_live = (du * huL).sum(-1) / HuL * m["umask2"]
    Fv_live = (dv * hvL).sum(-1) / HvL * m["vmask2"]

    # REST weighting (NEMO :337): e3u_0 / r1_hu_0
    e3u0 = np.moveaxis(m["e3u0"], 0, -1) * np.moveaxis(m["umask"], 0, -1)  # (nj,ni,jpk)
    e3v0 = np.moveaxis(m["e3v0"], 0, -1) * np.moveaxis(m["vmask"], 0, -1)
    Hu0 = np.maximum(m["hu0"], 1e-10)
    Hv0 = np.maximum(m["hv0"], 1e-10)
    Fu_rest = (du * e3u0).sum(-1) / Hu0 * m["umask2"]
    Fv_rest = (dv * e3v0).sum(-1) / Hv0 * m["vmask2"]

    print("\n" + "=" * 74)
    print("[Q1] THICKNESS-WEIGHTING DIFF: F_slow_live (lego) - F_slow_rest (NEMO)")
    print("     same du_dt both sides -> isolates the e3u_0 vs h_u_pre statement")
    print("=" * 74)
    du_w = np.abs((Fu_live - Fu_rest)[m["umask2"]])
    dv_w = np.abs((Fv_live - Fv_rest)[m["vmask2"]])
    scu = np.abs(Fu_rest[m["umask2"]]); scv = np.abs(Fv_rest[m["vmask2"]])
    print(f"  u: |F_rest| p50={np.percentile(scu,50):.3e}  "
          f"DIFF max={du_w.max():.4e} p99.9={np.percentile(du_w,99.9):.4e} "
          f"p50={np.percentile(du_w,50):.4e} [m/s^2]")
    print(f"  v: |F_rest| p50={np.percentile(scv,50):.3e}  "
          f"DIFF max={dv_w.max():.4e} p99.9={np.percentile(dv_w,99.9):.4e} "
          f"p50={np.percentile(dv_w,50):.4e} [m/s^2]")
    # WHERE is the weighting DIFF (wall vs interior) -- compare to JOB-1 residual
    dwu = (Fu_live - Fu_rest) * m["umask2"]
    a = np.abs(dwu)
    west2 = a[:, :2][m["umask2"][:, :2]]; interior = a[2:-2, 2:-2][m["umask2"][2:-2, 2:-2]]
    print(f"  weighting DIFF structure (u): W-2col max={west2.max() if west2.size else 0:.3e} "
          f"interior max={interior.max() if interior.size else 0:.3e}")

    # convert the F_slow weighting DIFF to a transport-per-width DIFF to compare
    # with JOB-1's 0.88 m^2/s: the barotropic mode integrates F_slow over the
    # 2dt window into a velocity, x H -> transport.  Scale ~ F_diff * H * 2dt.
    F_to_transport = HuL * (2 * J.RN_DT)   # crude: forcing*H*window -> [m^2/s]
    trans_diff = np.abs((Fu_live - Fu_rest) * HuL * (2 * J.RN_DT))[m["umask2"]]
    print(f"\n  weighting DIFF as transport-per-width (F_diff*H_u*2dt): "
          f"max={trans_diff.max():.4e} p99.9={np.percentile(trans_diff,99.9):.4e} m^2/s")
    print(f"  (JOB-1 per-face residual was 0.88 m^2/s max, wall-concentrated)")
    print(f"  -> if this is MUCH smaller than 0.88, the thickness weighting does")
    print(f"     NOT own the transport gap; the DIFF lives in the substep DYNAMICS")
    print(f"     (drag/Coriolis/filter internals) -- CLEAN-table escalation.")

    # ==== DIRECT INPUT COMPARISON: lego F_slow_u (assembled) vs NEMO zu_frc ===
    # This is JOB-2's actual subject: the fully-assembled barotropic RHS forcing.
    # CAVEAT: lego du_dt and NEMO puu(Krhs) are different momentum operators, so
    # a nonzero diff mixes forcing-assembly with the 3-D momentum-operator gap.
    # But an assembled forcing that already carries the wall-concentrated
    # 0.88-m^2/s-equivalent structure would localise it to the INPUT; a clean
    # match localises it to the substep DYNAMICS.
    print("\n" + "=" * 74)
    print("[INPUT] lego assembled F_slow_u  vs  NEMO zu_frc dump (both m/s^2)")
    print("  (CAVEAT: different momentum operators feed each; see note)")
    print("=" * 74)
    if "F_slow_u" not in cap:
        print("  F_slow_u NOT captured (hook missed) -- skip")
        return 0
    Fu = cap["F_slow_u"]; Fv = cap["F_slow_v"]
    Fu_c = _to_cell_u(Fu[..., None])[..., 0] if Fu.ndim == 2 and Fu.shape[1] == J.NI + 1 else Fu
    if Fu.shape[1] == J.NI + 1:
        Fu_c = Fu[:, 1:]
    else:
        Fu_c = Fu
    Fv_c = Fv[1:, :] if Fv.shape[0] == J.NJ + 1 else Fv
    zu = J._load2d_full("spg_dump_zu_frc.bin") if False else _load_interior_52x199("spg_dump_zu_frc.bin")
    zv = _load_interior_52x199("spg_dump_zv_frc.bin")
    print(f"  |zu_frc| max={np.abs(zu).max():.3e} p50={np.percentile(np.abs(zu),50):.3e}  "
          f"|F_slow_u| max={np.abs(Fu_c).max():.3e}")
    du_in = np.abs((Fu_c - zu) * m["umask2"])[m["umask2"]]
    dv_in = np.abs((Fv_c - zv) * m["vmask2"])[m["vmask2"]]
    print(f"  u: DIFF max={du_in.max():.4e} p99.9={np.percentile(du_in,99.9):.4e} "
          f"p50={np.percentile(du_in,50):.4e} [m/s^2]")
    print(f"  v: DIFF max={dv_in.max():.4e} p99.9={np.percentile(dv_in,99.9):.4e} "
          f"p50={np.percentile(dv_in,50):.4e} [m/s^2]")
    # structure + transport-equivalent (F_diff * H * 2dt)
    dFu = (Fu_c - zu) * m["umask2"]
    a = np.abs(dFu)
    west2 = a[:, :2][m["umask2"][:, :2]]; inter = a[2:-2, 2:-2][m["umask2"][2:-2, 2:-2]]
    print(f"  u DIFF structure: W-2col max={west2.max() if west2.size else 0:.3e} "
          f"interior max={inter.max() if inter.size else 0:.3e}")
    tr = np.abs(dFu * HuL * (2 * J.RN_DT))[m["umask2"]]
    print(f"  u DIFF as transport (F_diff*H*2dt): max={tr.max():.4e} "
          f"p99.9={np.percentile(tr,99.9):.4e} m^2/s  (JOB-1 residual 0.88)")
    return 0


def _load_interior_52x199(fn):
    """Load a NO-HALO interior dump (Nis0:Nie0, Njs0:Nje0) = 52x199, already
    interior (NOT haloed).  Shape -> (nj=199, ni=52)."""
    a = np.fromfile(os.path.join(J.SEQDUMP, fn), dtype="<f8")
    assert a.size == J.NI * J.NJ, (fn, a.size, J.NI * J.NJ)
    return a.reshape(J.NJ, J.NI)


if __name__ == "__main__":
    raise SystemExit(main())

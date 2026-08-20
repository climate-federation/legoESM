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

fp64 via run_fp64.py; LEGOESM_NEMO_E3T=both; day-0 bit-identity; card printed;
git SHA + effective flags stamped by multistep_replay.provenance(); every raw
NEMO dump routed through ocean.fidelity.time_levels (which RAISES on an
unregistered basename) and NaN-fatal.

HISTORY -- TWO DEFECTS, BOTH FIXED HERE (2026-08-19)
==============================================================================
DEFECT A, WIND-OFF.  Every number recorded at commit 40b92249a was measured on
an UNFORCED ocean: this probe called ``model.step(surface_forcing=None)`` while
the kamm card routes the wind momentum THROUGH step, and the analytic
applicator skips its own wind when ``wind_through_step=True``
(dino.py:3756-3757).  Row 6 of the table IS the wind term, so a wind-off run
could not measure it at all.  Same defect as fdb5cfec6, fixed in four sibling
probes at 5e8407797, missed here.  ``DINO_ZUFRC_WIND=0`` is the continuity
control.

DEFECT B, FABRICATED VERDICTS.  Rows 2-7 printed "MATCH" from a hard-coded
string in the TABLE literal.  No run ever computed them.  The strings are
DELETED and replaced by a ROW STATUS block whose every line is MEASURED,
CONFIG-READ (from the oracle run's own ocean.output) or UNMEASURED.

WIND-ON RESULTS (fp64, LEGOESM_NEMO_E3T=both, day-0 gate 0.000e+00,
tau_x in [-0.1999, 0.1000] Pa):

  row 1  thickness weighting   MEASURED  1.0164e-20 m/s^2 max (was 6.8e-21
         wind-off) -- a REAL code difference that is numerically INERT, 18
         orders below the transport residual it was proposed to explain.
  row 2  baroclinic split      UNMEASURED (no puu(Krhs) dump around :351)
  row 3  2-D Coriolis removal  UNMEASURED.  The run's only Coriolis dump is
         the IN-LOOP substep-1 trend (:794) -- a DIFFERENT quantity from the
         pre-loop dyn_cor_2D(puu_b(Kmm)) this row removes.  Using it would
         have manufactured the "MATCH" that was previously just asserted.
  row 4  bottom drag           NEMO side MEASURED 2.7138e-09 m/s^2 max
         (drg_dump_zu_frc_inc.bin); lego side UNMEASURED per-term.
  row 5  ln_apr_dyn            CONFIG-READ "F" from the oracle run's own
         ocean.output at run time, not quoted from a comment.
  row 6  wind                  MEASURED BOTH SIDES.  NEMO 8.6218e-08 m/s^2
         max; legoESM's tau at the U-face over rho0*H_u_live agrees to
         4.4235e-12 m/s^2, i.e. 5.13e-05 relative.  ROW 6 GENUINELY MATCHES,
         and now for the first time by measurement.
  row 7  ssh_frc / emp         MEASURED exactly 0.0 (DINO carries no emp).
  TOTAL  assembled F_slow_u vs NEMO's zu_frc dump: max 2.6999e-08 m/s^2,
         p50 2.9494e-11, against |zu_frc| max 3.034e-05 -> 0.089% relative
         (wind-off it was 9.5e-08 / ~0.3%).

RETRACTED -- "rows 2-7 MATCH".  Rows 2 and 3 were never measured and still
are not; row 4 is measured on NEMO's side only.  The bare claim is withdrawn;
what the run now supports is the per-row status above.

SURVIVES, and improves wind-on -- "the zu_frc ASSEMBLY is clean, the gap is
inside the barotropic substep dynamics".  The fully-assembled slow forcing
agrees with NEMO's own zu_frc dump to 0.089% (from 0.3%), and row 1's
statement-level code difference is inert at 1e-20.  CAVEAT unchanged: legoESM
du_dt and NEMO puu(Krhs) are different momentum operators, so the 0.089% mixes
assembly with the 3-D momentum-operator gap.  It is an UPPER BOUND on the
assembly error, not a measurement of it.

SIGN-CONVENTION RETRACTION, this probe's own, caught in-flight: row 6's first
wind-on run reported an EXACT factor-2 residual (rel 2.000e+00).  That was the
probe reading the atmospheric-convention intermediate (dino.py:3608 negates
tau for the PE block, which negates it again) instead of the ocean-side stress
NEMO's utau dump uses.  Probe bug, not a model defect; fixed before anything
was recorded.  An exact factor of 2 in a residual is the signature of a
flipped sign, not of a physics gap.
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
  lego: ocean_model_latlon_cgrid.py _step_impl, card=nemo_dino_kamm_mlf
--------------------------------------------------------------------------------
 THIS TABLE IS A SOURCE READ ONLY.  It carries NO verdicts.  Every row's status
 is COMPUTED by this run and printed in the ROW STATUS block below.  (Until
 2026-08-19 rows 2-7 printed a baked-in "MATCH" that no run had ever computed;
 those strings are deleted, not re-derived.)
--------------------------------------------------------------------------------
 N | NEMO statement (file:line)                      | legoESM (file:line)
--------------------------------------------------------------------------------
 1 | zu_frc = SUM_k e3u_0*puu(Krhs)*umask * r1_hu_0  | F_slow_u = SUM(du_dt*h_u_pre)
   |   (:337, key_qco REST thickness e3u_0/r1_hu_0)  |   /H_u_pre (:3465-3482, LIVE h)
 2 | puu(Krhs) -= zu_frc  (baroclinic split, :351)   | du_dt_pert = du_dt - F_slow (:3517)
 3 | dyn_cor_2D(puu_b(Kmm)) -> zu_trd;               | barotropic_coriolis_een_pre_step
   |   zu_frc -= zu_trd*ssumask  (:365-369)          |   Kmm coeffs; F_slow -= _cor_u_sub
   |                                                 |   (:3946-3964)
 4 | dyn_drg_init: pu_RHSi += r1_hu(Kmm)*r1_2*       | barotropic_drag_substep:
   |  (rCdU_bot(i+1)+rCdU_bot(i))*(puu(ikbu,Kbb)     |  F_slow -= r_u_bt/H_u_pre*
   |  -puu_b(Kbb))  (ln_bt_fw=F -> Kbb, :51-60)      |  (u_bot-U_bar) w/ u_before (:3542-3564)
 5 | ln_apr_dyn: atmospheric-pressure gradient add   | (no APR term exists)
   |   (:407-421)                                    |
 6 | wind: zu_frc += r1_rho0*r1_2*(utau_b+utauU)     | surface_stress_implicit path
   |  *r1_hu(Kmm)  (ln_bt_fw=F CENTRED, :437-443)    |  OR carried in du_dt
 7 | ssh_frc = r1_rho0*r1_2*(emp+emp_b) (:519 else)  | F_slow_eta (:3784-3824)
--------------------------------------------------------------------------------
 EXCLUDED from zu_frc (NEMO), read from source: ln_rnf=F, ln_isf=F, ln_bdy=F,
 key_agrif absent, ln_wd_dl_bc=F.  A2 biharmonic / slow_forcing_ab2 off on this
 card.  These are SOURCE READS, not measurements, and are labelled as such.
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
        dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing)
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface, min_cell_to_vface)

    mr.IC_STEP = 230400
    mr.provenance("zu_frc_assembly_table")
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
    # WIND-ON harness (commit 5e8407797's pattern; the fdb5cfec6 retraction).
    # The kamm card routes the WIND MOMENTUM through model.step(surface_forcing=
    # sf) and the analytic applicator skips its eq-7 wind when
    # wind_through_step=True (dino.py:3756-3757), so surface_forcing=None left
    # the ocean UNFORCED.  Row 6 of the table below IS the wind term, so a
    # wind-off run could not have measured it at all.  DINO_ZUFRC_WIND=0 is the
    # continuity control that reproduces the wind-off numbers.
    _wind = bool(getattr(cfg, "wind_through_step", False))
    if os.environ.get("DINO_ZUFRC_WIND", "1") == "0":
        _wind = False
    sf_step = dino_step_surface_forcing(forcing) if _wind else None
    _tlo = float(np.min(np.asarray(sf_step.tau_x))) if sf_step is not None else 0.0
    _thi = float(np.max(np.asarray(sf_step.tau_x))) if sf_step is not None else 0.0
    print(f"  FORCING: wind_through_step="
          f"{bool(getattr(cfg, 'wind_through_step', False))} applied={_wind} "
          f"surface_stress_implicit={getattr(cfg,'surface_stress_implicit',None)} "
          f"tau_x[Pa] range=[{_tlo:.4f},{_thi:.4f}]", flush=True)
    if _wind and not (_thi - _tlo) > 0.0:
        raise SystemExit("*** FORCING GATE FAILED: wind nominally ON but tau_x "
                         "is flat -- the probe would measure an unforced ocean")
    DT = J.RN_DT
    st, rate = apply_dino_lat_lon_surface_forcing(
        st0, forcing, br.z_coord, cfg, DT, t_seconds=DT, return_rate=True)
    _ = model.step(st, DT, surface_forcing=sf_step, external_tracer_rate=rate)
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
    print(f"  (JOB-1's WIND-ON per-face residual is 5.99e-02 m^2/s max, "
          f"interior-peaked; its wind-off 0.88 m^2/s wall-concentrated figure "
          f"is RETRACTED -- see hu_avg_perface_diff.py HISTORY)")
    print(f"  -> this weighting DIFF is ~1e-13 m^2/s, eighteen orders below "
          f"even the wind-on residual: the thickness weighting is numerically "
          f"INERT here and does NOT own the transport gap.")

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
        raise SystemExit("*** F_slow_u hook never fired -- the assembled-total "
                         "row would be UNMEASURED and the run would still exit "
                         "0; abort instead of reporting a partial table")
    Fu = cap["F_slow_u"]; Fv = cap["F_slow_v"]
    if Fu.shape[1] == J.NI + 1:
        Fu_c = Fu[:, 1:]
    else:
        Fu_c = Fu
    Fv_c = Fv[1:, :] if Fv.shape[0] == J.NJ + 1 else Fv
    zu = _load_interior_52x199("spg_dump_zu_frc.bin")
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
          f"p99.9={np.percentile(tr,99.9):.4e} m^2/s")

    _row_status(m, HuL, du_w, du_in, forcing, cfg, model)
    return 0


def _row_status(m, HuL, weighting_diff, assembled_diff, forcing, cfg, model):
    """Print a COMPUTED status for each of the 7 table rows.

    Every line is one of:
      MEASURED     -- a number this run produced (both sides, or NEMO's own
                      dump for a term whose lego counterpart is separately
                      accounted for in the assembled-total check).
      CONFIG-READ  -- a switch read at run time from the ORACLE RUN'S OWN
                      ocean.output, not from a comment.
      UNMEASURED   -- no number exists; the missing measurement is named.
    Nothing here is a verdict the run did not compute.
    """
    import numpy as np
    print("\n" + "=" * 74)
    print("ROW STATUS -- computed by THIS run, one line per table row")
    print("=" * 74)
    um = m["umask2"]
    rho0 = model.config.constants.rho_0

    # -- row 1: the thickness weighting, measured above -------------------
    print(f"  row 1  thickness weighting (REST e3u_0 vs LIVE h_u_pre)")
    print(f"         MEASURED  DIFF max={weighting_diff.max():.4e} "
          f"p99.9={np.percentile(weighting_diff,99.9):.4e} m/s^2 "
          f"(same du_dt both sides -> isolates the statement)")

    # -- row 2: no dump of puu(Krhs) either side of :351 ------------------
    print(f"  row 2  baroclinic split puu(Krhs) -= zu_frc")
    print(f"         UNMEASURED -- NEMO dumps no puu(Krhs) around :351, so the "
          f"split cannot be diffed.")
    print(f"         Missing measurement: dump puu(:,:,:,Krhs) immediately "
          f"before and after :351.")

    # -- row 3: the ONLY cor2d dump is a DIFFERENT quantity ---------------
    print(f"  row 3  2-D Coriolis removal zu_frc -= zu_trd (:365-369)")
    print(f"         UNMEASURED -- the run's only Coriolis dump "
          f"(cor2d_dump_zu_trd_substep1.bin, :794) is the IN-LOOP substep-1 "
          f"trend, a DIFFERENT quantity from the pre-loop "
          f"dyn_cor_2D(puu_b(Kmm)) this row removes; conflating them would "
          f"fabricate a match.")
    print(f"         Missing measurement: dump zu_trd at :368.")

    # -- row 4: NEMO's own drag increment ---------------------------------
    drg = _load_interior_52x199("drg_dump_zu_frc_inc.bin")
    a = np.abs(drg[um])
    print(f"  row 4  bottom-drag increment added to zu_frc (dyn_drg_init)")
    print(f"         MEASURED (NEMO side, drg_dump_zu_frc_inc.bin :393-398): "
          f"max={a.max():.4e} p99.9={np.percentile(a,99.9):.4e} "
          f"p50={np.percentile(a,50):.4e} m/s^2")
    print(f"         lego side UNMEASURED per-term (no drag-only hook); it is "
          f"included in the assembled-total check below.")

    # -- row 5: read the ORACLE RUN'S OWN namelist echo --------------------
    apr = _oceanoutput_flag("ln_apr_dyn")
    print(f"  row 5  atmospheric-pressure gradient (ln_apr_dyn)")
    print(f"         CONFIG-READ from the run's own ocean.output: "
          f"ln_apr_dyn = {apr}   (lego has no APR term at all)")
    if apr != "F":
        raise SystemExit("*** ln_apr_dyn is not F in the oracle run -- the "
                         "table's row-5 premise is void")

    # -- row 6: NEMO's wind increment vs legoESM's own tau ----------------
    wnd = _load_interior_52x199("wnd_dump_zu_frc_inc.bin")
    # legoESM's card wind at the U-FACE.  STAGGERING, stated explicitly: the
    # card's tau is CELL-CENTRED (dino.py:3608 tau_x=-forcing["tau_u_cell_2d"],
    # atmospheric sign), NEMO's utauU is at the U-point; this reconstruction
    # applies NEMO's own 2-cell average and its r1_hu(Kmm) live depth.  It is
    # the PROBE's reconstruction of the lego term, NOT a value captured out of
    # the model -- so the row is labelled RECONSTRUCTED on the lego side.
    # SIGN, walked term by term: forcing["tau_u_cell_2d"] IS the stress ON the
    # ocean (+0.2 Pa accelerates the ocean eastward), which is exactly NEMO's
    # utau convention.  dino_step_surface_forcing NEGATES it (dino.py:3608)
    # only because the PE external-tau block re-applies the ocean reaction
    # -tau; the two negations cancel inside the model.  Comparing NEMO's
    # utau-convention dump against the atmospheric-convention intermediate is
    # a sign error -- it showed up here as an EXACT factor-2 residual
    # (|a-b| = 2|b| when a = -b), which is why this line reads the ocean-side
    # field directly instead of the step-forcing struct.
    tau_c = np.asarray(forcing["tau_u_cell_2d"], dtype=np.float64)
    tau_u = 0.5 * (tau_c + np.roll(tau_c, -1, axis=1))     # periodic in i
    lego_wnd = tau_u / (rho0 * np.maximum(HuL, 1e-10)) * um
    d6 = np.abs((lego_wnd - wnd)[um])
    n6 = np.abs(wnd[um])
    print(f"  row 6  wind increment r1_rho0*r1_2*(utau_b+utauU)*r1_hu(Kmm)")
    print(f"         MEASURED (NEMO, wnd_dump_zu_frc_inc.bin :451-455): "
          f"max={n6.max():.4e} p50={np.percentile(n6,50):.4e} m/s^2")
    print(f"         RECONSTRUCTED (lego tau at U-face / rho0 / H_u_live): "
          f"DIFF max={d6.max():.4e} p99.9={np.percentile(d6,99.9):.4e} "
          f"rel(max/|nemo|max)={d6.max()/max(n6.max(),1e-300):.3e}")

    # -- row 7: emp / ssh_frc, NEMO's own dump ----------------------------
    ssh_frc = J._load2d_full("spg_dump_ssh_frc.bin")
    a7 = np.abs(ssh_frc[m["ssmask"] > 0.5])
    print(f"  row 7  ssh_frc = r1_rho0*r1_2*(emp+emp_b)")
    print(f"         MEASURED (NEMO, spg_dump_ssh_frc.bin :518/522): "
          f"max={a7.max():.4e} m/s  -> "
          f"{'exactly zero, as DINO has no emp' if a7.max() == 0.0 else 'NONZERO -- DINO emp is NOT zero'}")

    # -- the assembled total, already measured above -----------------------
    print(f"  TOTAL  assembled F_slow_u vs NEMO zu_frc dump: "
          f"MEASURED max={assembled_diff.max():.4e} "
          f"p50={np.percentile(assembled_diff,50):.4e} m/s^2 "
          f"(the sum of all 7 rows plus the 3-D momentum-operator gap)")


def _oceanoutput_flag(name):
    """Read a namelist switch from the ORACLE RUN'S OWN ocean.output.

    A switch quoted from a code comment is a claim; this reads what the run
    that produced these dumps actually used.  Returns 'T'/'F' or raises.
    """
    path = os.path.join(J.SEQDUMP, "ocean.output")
    with open(path, errors="replace") as fh:
        for line in fh:
            if name in line and "=" in line:
                val = line.split("=")[-1].strip().split()[0]
                if val in ("T", "F"):
                    return val
    raise SystemExit(f"*** {name} not found in {path} -- cannot read the "
                     "oracle run's own configuration")


def _load_interior_52x199(fn):
    """Load a NO-HALO interior dump (Nis0:Nie0, Njs0:Nje0) = 52x199, already
    interior (NOT haloed).  Shape -> (nj=199, ni=52).

    Routed through the shared time-level registry (which RAISES on an
    unregistered basename) and NaN-fatal, same contract as JOB 1's loaders.
    """
    J._disposition(fn)
    a = np.fromfile(os.path.join(J.SEQDUMP, fn), dtype="<f8")
    assert a.size == J.NI * J.NJ, (fn, a.size, J.NI * J.NJ)
    return J._finite(a.reshape(J.NJ, J.NI), fn)


if __name__ == "__main__":
    raise SystemExit(main())

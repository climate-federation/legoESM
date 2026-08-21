#!/usr/bin/env python
"""#1455 sec-D PHASE 2 -- the decisive per-substep trajectory comparison.

The per-face probe (hu_avg_perface_diff.py) established that legoESM's
substep-mean transport ``Hu_avg`` carries a wall-concentrated ~0.88 m^2/s bias
vs BOTH NEMO references (entry transport and un_adv), with -2dt*div reproducing
the 4.2e-3 m/step eta injection to 3 sig figs.  The bias is INSIDE the 68-substep
loop of ``barotropic_substeps_latlon_cgrid``.

NEMO dumps the FULL per-substep trajectory for this exact step (kt=nit000=230401):
``substep_dump.bin`` (dynspg_ts.F90:926-940) holds, per jn=1..68,
``sshn_e ssha_e zsshp2_e un_e vn_e ua_e va_e``.  This probe captures legoESM's
matching per-substep fields and compares them, substep by substep, to localise
WHICH substep and WHICH term first grows the wall bias.

legoESM <-> NEMO map (barotropic_latlon_cgrid.py:750-1000 vs dynspg_ts.F90):
  * un_e (NEMO, substep-START velocity, jn)      <-> U_bar_c  (carry-in)
  * ua_e (NEMO, substep-END velocity, jn+1)      <-> U_bar_new
  * ua_e-mid (NEMO za1*un+za2*ub+za3*ubb, :645)  <-> U_mid    (:796)
  * zhup2_e (NEMO mid-step flux depth, :678-681) <-> H_u_flux (:819)
  * zhU/e2u = ua_e_mid*zhup2_e (NEMO flux, :700) <-> flux_u/u_mask (:825)
  * un_adv += wgtbtp2*zhU*r1_e2u (:736)          <-> Hu_sum += w_tr*flux_u (:829)

NOTE on the NEMO "ua_e" DUMPED field: substep_dump writes ua_e AFTER the
velocity update (dynspg_ts.F90:840-851 sets ua_e to the jn+1 velocity), i.e.
the DUMPED ua_e == the substep-END velocity == legoESM U_bar_new, NOT the
mid-step extrapolation.  The mid-step ua_e that feeds the flux (:645, :700) is
OVERWRITTEN by :840 before the dump.  So from substep_dump we can reconstruct
NEMO's mid-step velocity ourselves: ua_e_mid[jn] = za1*un_e[jn] + za2*un_e[jn-1]
+ za3*un_e[jn-2] (the AB3 on the dumped un_e history), and NEMO's flux
zhU/e2u = ua_e_mid * zhup2_e where zhup2_e uses the extrapolated ssh
zsshp2_e (also dumped, but as the BACK-interp AM4 eta at :772, NOT the mid-step
AB3 eta at :657 -- those are different zsshp2_e uses; see below).

CONTROLS (skill Rule 1c/1d/2/3):
  * fp64 (run_fp64 wraps or set here); LEGOESM_NEMO_E3T=both; dtypes printed.
  * day-0 bit-identity via build_replay_ic.
  * substep count reconcile: assert legoESM n_loop == NEMO icycle == 68.
  * PLANT: shift the NEMO trajectory by one substep -> the per-substep diff MUST
    blow up (the metric is not substep-shift-invariant).
  * card state printed.

RETRACTION POINTER (2026-08-19).  This file's premise cites the "~0.88 m^2/s
wall-concentrated transport bias" and the "~4.2e-3 m/step eta injection".  BOTH
WERE WIND-OFF ARTIFACTS and are RETRACTED: the probe that produced them ran with
surface_forcing=None on a card that threads the wind THROUGH model.step.  Wind-on
the transport residual is 5.99e-02 m^2/s and INTERIOR-peaked, and the eta
increment is 1.95e-04 m.  The associated claim that the transport diff "closes
the injection to 3 sig figs" is also retracted -- it compared against the wrong
reference.  See the HISTORY block in hu_avg_perface_diff.py before using any
number below.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEQDUMP = os.environ.get(
    "DINO_NEMO_RUN_SEQDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_Y20_1R")

HLS = 2


def _parse_nemo_trajectory():
    """Parse substep_dump.bin -> dict jn -> {field: (jpj,jpi) haloed}."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    time_level_for_dump("substep_dump.bin")  # force registry disposition
    f = os.path.join(SEQDUMP, "substep_dump.bin")
    with open(f, "rb") as fh:
        jpi, jpj, icy = (int(x) for x in np.fromfile(fh, dtype="<i4", count=3))
        n = jpi * jpj
        names = ["sshn_e", "ssha_e", "zsshp2_e", "un_e", "vn_e", "ua_e", "va_e"]
        subs = {}
        for _ in range(icy):
            jn = int(np.fromfile(fh, dtype="<i4", count=1)[0])
            # strip HLS halo -> interior (nj_int, ni_int) to match the haloless
            # mesh_mask (52x199) and the per-face probe's convention.
            subs[jn] = {nm: np.fromfile(fh, dtype="<f8", count=n)
                        .reshape(jpj, jpi)[HLS:-HLS, HLS:-HLS]
                        for nm in names}
    return jpi - 2 * HLS, jpj - 2 * HLS, icy, subs


# ponytail: capture legoESM's per-substep carries by teeing fori_loop -> scan
# (the DINO card uses differentiable_barotropic=False => fori_loop, which has no
# ys; converting to scan collects every intermediate carry, incl. the cumulative
# Hu_sum whose per-substep diff is w_tr*flux_u).  Done inline in main().


def main():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    print(f"control dtype = {get_policy().control}  "
          f"E3T={os.environ.get('LEGOESM_NEMO_E3T')!r}")

    jpi, jpj, icy, subs = _parse_nemo_trajectory()
    print(f"NEMO substep_dump: jpi={jpi} jpj={jpj} icycle={icy} "
          f"jns={min(subs)}..{max(subs)}")

    # ---- capture legoESM per-substep via a patched _run_substep_loop ----------
    import jax
    import jax.numpy as jnp
    import legoesm.ocean.dynamics.barotropic_latlon_cgrid as bmod

    _orig_run = bmod._run_substep_loop
    import multistep_replay as mr

    captured = {"carries": None}

    _orig_fori = jax.lax.fori_loop

    def _fori_capture(lower, upper, body, init, *args, **kw):
        # Convert to scan so we can collect every intermediate carry.
        def scan_body(carry, i):
            new = body(i, carry)
            return new, new
        idx = jnp.arange(lower, upper)
        final, stack = jax.lax.scan(scan_body, init, idx)
        captured["carries"] = stack
        return final

    # Only intercept the fori_loop INSIDE _run_substep_loop; guard by a flag.
    _intercept = {"on": False}

    def _run_wrap(*a, **k):
        _intercept["on"] = True
        # stash the loop's OWN arguments so the response-factor arm below can
        # re-run the IDENTICAL loop with one perturbed input and nothing else.
        captured["loop_args"] = (a, dict(k))
        try:
            jax.lax.fori_loop = _fori_capture
            return _orig_run(*a, **k)
        finally:
            jax.lax.fori_loop = _orig_fori
            _intercept["on"] = False
    bmod._run_substep_loop = _run_wrap

    # ALSO capture the F_slow_u forcing + the real substep dt_s the loop uses,
    # for the non-circularity control (verr - dt_s*ΔF).
    import inspect as _inspect
    _orig_baro = bmod.barotropic_substeps_latlon_cgrid
    _baro_sig = _inspect.signature(_orig_baro)

    def _baro_wrap(*a, **k):
        ba = _baro_sig.bind(*a, **k)
        ba.apply_defaults()
        captured["F_slow_u"] = np.asarray(ba.arguments["F_slow_u"])
        captured["dt_s"] = float(np.asarray(ba.arguments["dt_s"]))
        return _orig_baro(*a, **k)
    bmod.barotropic_substeps_latlon_cgrid = _baro_wrap

    # also need the module that CALLS fori_loop to see the patched jax.lax.
    # barotropic_latlon_cgrid uses `jax.lax.fori_loop` (attribute lookup at
    # call time), so patching jax.lax.fori_loop globally works; the wrap above
    # sets/restores it around the original run.

    # #1455 baro-substeps: the IC step is the ENV-selected one (multistep_replay
    # already reads DINO_1226_IC_STEP); the previous hard pin to 230400 silently
    # overrode it, so a day-180 SEQDUMP was replayed against the y20 restart
    # tiles and the probe aborted.  Keep the same default (230400) via the
    # module global, and STAMP what was actually used.
    mr.provenance('substep_traj_compare')
    jax.clear_caches()
    g, br, cfg, st0 = mr.build_replay_ic()
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing, dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    print(f"  card face_depth={mc.barotropic.barotropic_face_depth!r} "
          f"time_filter={mc.barotropic.barotropic_time_filter!r} "
          f"coriolis={mc.barotropic.barotropic_coriolis!r} "
          f"reconcile={mc.barotropic.barotropic_reconcile_target!r}")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    # #1455 retraction fix: route the WIND MOMENTUM through
    # model.step(surface_forcing=sf) as production does (run_dino.py:665-670,
    # 763-767); surface_forcing=None dropped the wind and fabricated the
    # F_slow residual. The wind ENTERS the substep loop's F_slow, so this
    # probe -- which compares the barotropic substep trajectory -- MUST carry
    # it or it measures the wrong F_slow.
    _wind = bool(getattr(cfg, "wind_through_step", False))
    sf_step = dino_step_surface_forcing(forcing) if _wind else None
    _tau_lo = float(np.min(np.asarray(sf_step.tau_x))) if sf_step is not None else 0.0
    _tau_hi = float(np.max(np.asarray(sf_step.tau_x))) if sf_step is not None else 0.0
    print(f"  FORCING: wind_through_step={_wind} "
          f"surface_stress_implicit={getattr(cfg, 'surface_stress_implicit', None)} "
          f"tau_x[Pa] range=[{_tau_lo:.4f},{_tau_hi:.4f}]")
    DT = 2700.0
    # #1455 PHASE 1 (the time axis) -- THE SEASONAL CLOCK, and it is a real
    # defect in the three commits before this one, not a refactor.  This card
    # sets forcing_annual_cycle=True (ln_ann_cyc), so the restoring target T*
    # and the solar Q_sr are functions of t_seconds.  The state being replayed
    # is NEMO step IC_STEP, i.e. ABSOLUTE time (IC_STEP+1)*DT; passing a bare
    # DT put legoESM's season at day 0.03 while NEMO's was at day 180+.  That
    # was harmless at the historical IC only by coincidence (230400 steps x
    # 2700 s = exactly 20 x 360 d, so relative == absolute -- the coincidence
    # multistep_replay's own IC_STEP note already flags), and it is NOT
    # harmless once this walk moves ACROSS the 90-day window: a season
    # mismatch that varies with the state would fake a time dependence in
    # exactly the quantity the walk is now measuring.
    # Set DINO_1226_T_SECONDS=2700 to restore the legacy bare-DT clock (the
    # A/B that quantifies what the earlier commits' numbers change by).
    _env_t = os.environ.get("DINO_1226_T_SECONDS")
    T_SECONDS = float(_env_t) if _env_t not in (None, "") else float(
        mr.IC_STEP + 1) * DT
    print(f"  seasonal clock: t_seconds={T_SECONDS:.1f} s "
          f"(day {T_SECONDS / 86400.0:.2f}); legacy bare-DT = {DT:.1f} s "
          f"(day {DT / 86400.0:.2f}); annual_cycle="
          f"{getattr(cfg, 'forcing_annual_cycle', None)}")
    st, rate = apply_dino_lat_lon_surface_forcing(
        st0, forcing, br.z_coord, cfg, DT, t_seconds=T_SECONDS,
        return_rate=True)
    # Run EAGER (disable_jit) so the fori->scan tee yields CONCRETE carries
    # across the jit boundary (fp64, one baroclinic step -- cheap).
    with jax.disable_jit():
        st = model.step(st, DT, surface_forcing=sf_step, external_tracer_rate=rate)

    if captured["carries"] is None:
        raise SystemExit("did not capture the substep carries (fori path?)")
    carries = captured["carries"]
    # carry layout (ab3 path): (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum,
    #   U_sum, V_sum, Ub, Ubb, Vb, Vbb, etab, etabb)
    eta_stk = np.asarray(carries[0])       # (n_loop, nj, nl)
    U_bar_stk = np.asarray(carries[1])     # (n_loop, nj, nl+1)
    Hu_sum_stk = np.asarray(carries[3])    # cumulative
    n_loop = eta_stk.shape[0]
    print(f"  captured n_loop={n_loop}  (NEMO icycle={icy})  "
          f"eta_stk dtype={eta_stk.dtype}")
    assert n_loop == icy, f"substep count mismatch: lego {n_loop} vs NEMO {icy}"

    # per-substep Hu increment = Hu_sum[j] - Hu_sum[j-1] = w_tr_j * flux_u_j
    Hu_inc = np.diff(Hu_sum_stk, axis=0, prepend=Hu_sum_stk[:1] * 0)
    # Compare the CUMULATIVE Hu_sum trajectory: at each substep, project both to
    # per-unit-width transport and measure the wall-column error growth.
    # NEMO cumulative un_adv up to substep j:
    #   un_adv_partial[j] = sum_{i<=j} wgtbtp2[i] * (ua_e_mid[i]*zhup2_e[i])
    # We reconstruct NEMO's per-substep flux zhU/e2u from the dumped trajectory.
    import netCDF4 as nc
    d = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    gg = lambda v: np.asarray(d.variables[v][0], dtype=np.float64)
    e3u0 = gg("e3u_0"); umask = gg("umask"); e1e2t = gg("e1t") * gg("e2t")
    e1u = gg("e1u"); e2u = gg("e2u"); e3t0 = gg("e3t_0"); tmask = gg("tmask")
    d.close()
    hu0_full = (e3u0 * umask).sum(0)  # (jpj,jpi) full haloed reference u-depth
    umask2 = (umask[0] > 0.5)
    r1_e1e2u = 1.0 / (e1u * e2u)

    # NEMO AB3 coeffs (dynspg_ts.F90:635-637), full-AB3 rows; ll_init ramp on jn<3
    ZA = (1.781105, -1.06221, 0.281105)

    def nemo_flux_perwidth(subs, jn):
        """Reconstruct NEMO zhU/e2u = ua_e_mid[jn]*zhup2_e[jn] per unit width."""
        un = subs[jn]["un_e"]                      # substep-start vel (jn)
        unm1 = subs[jn - 1]["un_e"] if jn - 1 in subs else un
        unm2 = subs[jn - 2]["un_e"] if jn - 2 in subs else un
        ll_init_row = jn < 3   # (jn<3).AND.ll_init -> za=(1,0,0)
        if ll_init_row:
            ua_mid = un
        else:
            ua_mid = ZA[0] * un + ZA[1] * unm1 + ZA[2] * unm2
        # zhup2_e uses the MID-step AB3 ssh zsshp2_e (dynspg_ts.F90:657), but the
        # DUMPED zsshp2_e is the BACK-interp AM4 eta (:772) -- a DIFFERENT field.
        # So reconstruct the mid-step ssh from the dumped sshn_e history too:
        sn = subs[jn]["sshn_e"]
        snm1 = subs[jn - 1]["sshn_e"] if jn - 1 in subs else sn
        snm2 = subs[jn - 2]["sshn_e"] if jn - 2 in subs else sn
        if ll_init_row:
            ssh_mid = sn
        else:
            ssh_mid = ZA[0] * sn + ZA[1] * snm1 + ZA[2] * snm2
        ssumask = (umask[0] > 0.5).astype(np.float64)
        # zhup2_e = hu_0 + 0.5*r1_e1e2u*(e1e2t[i]*ssh_mid[i]+e1e2t[i+1]*ssh_mid[i+1])*ssumask
        e1e2t_ip1 = np.roll(e1e2t, -1, axis=1)
        ssh_ip1 = np.roll(ssh_mid, -1, axis=1)
        zhup2 = hu0_full + 0.5 * r1_e1e2u * (
            e1e2t * ssh_mid + e1e2t_ip1 * ssh_ip1) * ssumask
        return ua_mid * zhup2 * (umask[0] > 0.5)   # per unit width [m^2/s]

    # Build NEMO's cumulative un_adv trajectory (per unit width), and the
    # per-substep wall error vs legoESM's cumulative Hu_sum trajectory.
    # legoESM weights:
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _compute_weights
    wf, wt, w_tr, nl2 = _compute_weights(mc, mc.barotropic.n_barotropic_substeps * 2,
                                         np.float64, substep_scale=2)
    w_tr = np.asarray(w_tr)   # (n_loop,) lego transport weights
    print(f"  lego n_loop from weights={nl2}  sum(w_tr)*n?={w_tr.sum():.6f}")

    # NEMO wgtbtp2 (secondary/transport weights), reconstructed from ts_wgt:
    nn_e = 23
    zwgt1 = np.zeros(3 * nn_e)
    jic = 2 * nn_e
    Kpit = 0
    for jn in range(1, 3 * nn_e + 1):
        if abs(jn - jic) / nn_e < 1.0:
            zwgt1[jn - 1] = 1.0
            Kpit = jn
    zwgt2 = np.zeros(3 * nn_e)
    for jn in range(1, Kpit + 1):
        for ji in range(jn, Kpit + 1):
            zwgt2[jn - 1] += zwgt1[ji - 1]
    r1_wgt2s = zwgt2.sum()
    wgtbtp2 = zwgt2[:Kpit] / r1_wgt2s   # normalised NEMO transport weights
    print(f"  NEMO Kpit(icycle)={Kpit}  wgtbtp2 nonzero={int((wgtbtp2>0).sum())} "
          f"sum={wgtbtp2.sum():.6f}")
    # code review #10: nn_e is hardcoded above but the namelist computes it at
    # runtime (ln_bt_auto=T, nn_e=30 is only the fallback).  It happens to be 23
    # for this grid+dt; assert it against the DUMP's own icycle so a different
    # dump set fails loudly instead of silently truncating the weights.
    assert Kpit == icy, (
        f"hardcoded nn_e={nn_e} gives Kpit={Kpit} but the dump says icycle={icy}")

    # INSTRUMENT VALIDATION (Rule: calibrate before trusting): the
    # nemo_flux_perwidth reconstruction, summed with wgtbtp2 over all 68
    # substeps, MUST reproduce NEMO's OWN spg_dump_un_adv_final.bin to roundoff
    # -- if it does not, the reconstruction (AB3 mid-step + zhup2_e) is wrong and
    # no diff claim stands.
    _un_adv_dump = np.fromfile(os.path.join(SEQDUMP, "spg_dump_un_adv_final.bin"),
                               dtype="<f8").reshape(jpj + 2 * HLS, jpi + 2 * HLS
                                                    )[HLS:-HLS, HLS:-HLS]
    _recon = np.zeros_like(hu0_full)
    for _j in range(1, Kpit + 1):
        _recon = _recon + wgtbtp2[_j - 1] * nemo_flux_perwidth(subs, _j)
    _rw = (umask[0] > 0.5)
    _rel = np.abs((_recon - _un_adv_dump)[_rw]).max() / max(
        np.abs(_un_adv_dump[_rw]).max(), 1e-300)
    print(f"\n  INSTRUMENT CHECK: reconstructed un_adv vs spg_dump_un_adv_final "
          f"rel={_rel:.3e} -> {'RECON VALID' if _rel < 1e-10 else 'RECON BROKEN'}")
    if _rel >= 1e-10:
        raise SystemExit("nemo_flux_perwidth reconstruction != un_adv dump; abort.")

    # cumulative NEMO un_adv[substep j] over wall vs interior; and final.
    un_adv_cum = np.zeros_like(hu0_full)
    # legoESM cumulative per-unit-width: Hu_sum_stk already cumulative w_tr*flux
    # (normalise not needed -- w_tr already /n).  Map lego east-face -> NEMO cell.
    def lego_Hu_cell(Hu_2d):
        # Hu_2d shape (nj, ni+1) -> east faces -> NEMO cell (nj, ni) interior.
        # jpi/jpj are already the INTERIOR dims (halo stripped in the parser).
        if Hu_2d.shape[1] == jpi + 1:
            return Hu_2d[:, 1:]           # (nj, ni) east faces == NEMO cell
        return Hu_2d

    # per-substep wall vs interior error of the CUMULATIVE transport
    print("\n  substep-by-substep CUMULATIVE Hu error (per unit width, m^2/s):")
    print("   jn |  wall(W col0-1)   interior     |  NEMO wall cum   lego wall cum")
    # haloless interior frame: per-face probe found the bias at cols 0,1 (W)
    # and 49,50,51 (E) -- the meridional-barrier seam columns.
    interior_sel = umask2.copy()
    interior_sel[:, :2] = False; interior_sel[:, -2:] = False
    interior_sel[:2, :] = False; interior_sel[-2:, :] = False
    wall_sel = np.zeros_like(umask2)
    for c in (0, 1, jpi - 1, jpi - 2, jpi - 3):     # W cols 0,1 ; E cols 49,50,51
        wall_sel[:, c] = umask2[:, c]
    rows = []
    for j in range(1, Kpit + 1):
        un_adv_cum = un_adv_cum + wgtbtp2[j - 1] * nemo_flux_perwidth(subs, j)
        lego_full = lego_Hu_cell(Hu_sum_stk[j - 1])
        diff = (lego_full - un_adv_cum) * (umask[0] > 0.5)
        wmax = np.abs(diff[wall_sel]).max() if wall_sel.any() else 0.0
        imax = np.abs(diff[interior_sel]).max() if interior_sel.any() else 0.0
        rows.append((j, wmax, imax,
                     np.abs(un_adv_cum[wall_sel]).max(),
                     np.abs(lego_full[wall_sel]).max()))
    for j, wmax, imax, nw, lw in rows:
        if j <= 6 or j % 8 == 0 or j >= Kpit - 3:
            print(f"   {j:2d} |  {wmax:.4e}   {imax:.4e}  |  {nw:.4e}   {lw:.4e}")

    # final wall bias -- should match the 0.88 m^2/s from hu_avg_perface_diff
    jn = Kpit
    lego_full = lego_Hu_cell(Hu_sum_stk[jn - 1])
    diff = (lego_full - un_adv_cum) * (umask[0] > 0.5)
    _fw = float(np.abs(diff[wall_sel]).max())
    _fi = float(np.abs(diff[interior_sel]).max())
    print(f"\n  FINAL (jn={jn}) wall max diff = {_fw:.4e}"
          f"  interior max = {_fi:.4e} m^2/s")
    print("  (compare hu_avg_perface_diff: wall 0.879, interior 0.589)")
    # jit-parity GATE (not an eyeballed print): the disable_jit fori->scan tee
    # must reproduce the independently-JITTED hu_avg_perface_diff wall/interior
    # bias -- if the tee perturbed the physics, these would diverge.  Assert,
    # don't just print (Rule: never trust a probe's own verdict).
    # #1455: the 0.879/0.589 reference values are WIND-OFF measurements
    # (hu_avg_perface_diff predates the retraction) -- with the wind now
    # correctly passed through model.step the wall bias collapses, so the pin
    # only applies when the wind is off (sf_step None).
    if sf_step is None:
        assert abs(_fw - 0.879) < 2e-3, (
            f"jit-parity BROKEN: final wall bias {_fw:.4e} != hu_avg_perface_diff "
            "0.879 -- the disable_jit tee perturbed the physics")
        assert abs(_fi - 0.589) < 2e-3, (
            f"jit-parity BROKEN: final interior bias {_fi:.4e} != 0.589")
    else:
        print("  (jit-parity pin vs hu_avg_perface_diff 0.879/0.589 SKIPPED: "
              "those are wind-off reference values; this run is wind-on)")

    # ===================================================================
    # DECOMPOSE: is the drift in the VELOCITY trajectory (U_bar vs un_e) or in
    # the FLUX DEPTH (H_u_flux vs zhup2_e)?  Compare legoESM's per-substep
    # U_bar (carry[1], the substep-END velocity == NEMO's dumped ua_e) directly
    # against NEMO's dumped un_e[jn+1] (== ua_e[jn], the substep-END velocity).
    # NEMO dumps un_e/ua_e AT the substep; ua_e[jn] is the jn+1 velocity, and
    # un_e[jn+1] == ua_e[jn] after the swap -- use un_e as the clean series.
    print("\n  VELOCITY trajectory: legoESM U_bar[jn] vs NEMO un_e[jn+1] "
          "(both substep-END, per-unit-vel m/s):")
    print("   jn |  wall max      interior max  |  |NEMO un|max")
    for j in (1, 2, 4, 8, 16, 24, 40, 56, 67):
        lego_u = np.asarray(U_bar_stk[j - 1])       # (nj, ni+1) end-of-substep-j
        lego_u_cell = lego_u[:, 1:] if lego_u.shape[1] == jpi + 1 else lego_u
        # NEMO substep-END velocity at jn == un_e[jn+1] (post-swap)
        nemo_u = subs[j + 1]["un_e"] if (j + 1) in subs else subs[j]["ua_e"]
        du = (lego_u_cell - nemo_u) * (umask[0] > 0.5)
        wmax = np.abs(du[wall_sel]).max()
        imax = np.abs(du[interior_sel]).max()
        print(f"   {j:2d} |  {wmax:.4e}   {imax:.4e}  |  {np.abs(nemo_u[umask2]).max():.4e}")

    # FLUX DEPTH: legoESM H_u_flux vs NEMO zhup2_e (mid-step).  We didn't stack
    # H_u_flux, but Hu_inc/w_tr = flux_u = H_u_flux*U_mid, and NEMO flux =
    # ua_e_mid*zhup2_e; the velocity comparison above isolates the U part, so a
    # residual in flux beyond U is the DEPTH part.  Report the RATIO of the
    # per-substep flux (lego/NEMO) at the wall to localise depth vs velocity.
    print("\n  PER-SUBSTEP FLUX ratio lego/NEMO at wall (isolates depth once "
          "velocity is matched):")
    print("   jn |  flux ratio(wall median)   lego fluxmax   NEMO fluxmax")
    for j in (2, 8, 24, 48, 68):
        lego_flux = np.asarray(Hu_inc[j - 1]) / max(w_tr[j - 1], 1e-300)
        lego_flux_cell = (lego_flux[:, 1:] if lego_flux.shape[1] == jpi + 1
                          else lego_flux)
        nemo_flux = nemo_flux_perwidth(subs, j)
        wm = wall_sel & (np.abs(nemo_flux) > 1e-6)
        ratio = np.median(lego_flux_cell[wm] / nemo_flux[wm]) if wm.any() else np.nan
        print(f"   {j:2d} |  {ratio:.6f}   {np.abs(lego_flux_cell[umask2]).max():.4e}"
              f"   {np.abs(nemo_flux[umask2]).max():.4e}")

    # ===================================================================
    # THE CAUSE: F_slow_u (lego's zu_frc-equiv) vs NEMO zu_frc, and the
    # NON-CIRCULARITY control  verr - dt_s*ΔF ~ 0  (proves the loop DYNAMICS
    # match; only the forcing input is biased).  spg_dump_zu_frc.bin is haloless
    # 52x199 already.
    zu_frc = np.fromfile(os.path.join(SEQDUMP, "spg_dump_zu_frc.bin"),
                         dtype="<f8").reshape(jpj, jpi)
    Fu = captured.get("F_slow_u")
    dt_s = captured.get("dt_s")
    if Fu is None or dt_s is None:
        raise SystemExit("did not capture F_slow_u/dt_s (baro hook missed?)")
    Fu_cell = Fu[:, 1:] if Fu.shape[1] == jpi + 1 else Fu
    dF = (Fu_cell - zu_frc) * (umask[0] > 0.5)
    wet = umask2
    print("\n  === THE CAUSE: F_slow_u(lego) vs NEMO zu_frc [m/s^2] ===")
    print(f"    dt_s used inside loop = {dt_s:.6f} s")
    print(f"    |ΔF| max={np.abs(dF[wet]).max():.4e} median={np.median(np.abs(dF[wet])):.4e}"
          f"  |zu_frc| max={np.abs(zu_frc[wet]).max():.4e}")
    print(f"    ΔF sign-coherence |mean|/rms = "
          f"{abs(dF[wet].mean())/max(dF[wet].std(),1e-300):.3f} (1=coherent bias, 0=noise)")
    # jn=1 vel err (independent of ΔF: from the carry trajectory)
    lego_u1 = np.asarray(U_bar_stk[0])
    lego_u1c = lego_u1[:, 1:] if lego_u1.shape[1] == jpi + 1 else lego_u1
    verr = (lego_u1c - subs[1]["ua_e"]) * (umask[0] > 0.5)
    resid = verr - dt_s * dF
    print(f"    dt_s*max|ΔF|    = {dt_s*np.abs(dF[wet]).max():.4e}   "
          f"jn=1 vel err max = {np.abs(verr[wet]).max():.4e}")
    print(f"    dt_s*median|ΔF| = {dt_s*np.median(np.abs(dF[wet])):.4e}   "
          f"jn=1 vel err med = {np.median(np.abs(verr[wet])):.4e}")
    print(f"    NON-CIRCULARITY resid (verr - dt_s*ΔF): max={np.abs(resid[wet]).max():.4e} "
          f"= {100*np.abs(resid[wet]).max()/max(np.abs(verr[wet]).max(),1e-300):.1f}% of verr "
          )

    # ===================================================================
    # #1455 baro-substeps: THE ACC DEPOSIT, SIGNED AND SECTION-INTEGRATED.
    #
    # Every number above is a MAXIMUM over faces.  The ACC gap is a SIGNED
    # SECTION INTEGRAL, and the two are not interchangeable: a max of 5e-2 m^2/s
    # says nothing about a deposit that only needs ~1e-8 m/s of coherent
    # section-mean velocity per step.  This block reduces the SAME per-substep
    # trajectory with the ACCEPTANCE GATE'S OWN metric
    #   ACC(i) = sum_j sum_k u(j,k,i) * e3t_1d(k) * e2u(j,i)   [per longitude]
    # (southern_term_torque_accum.py's acc_full weighting; rows 1..197, MEAN over
    # longitudes 2..-2 so the decomposition is LINEAR -- the gate itself takes the
    # median, which is not).  A depth-uniform barotropic increment dU contributes
    # dU * e2u * sum_k e3t_1d*umask, so the substep sum is exact, not a surrogate.
    #
    # BUDGET TARGET (what a suspect must be able to pay for): the accumulated
    # 90-day stage budget (c987db464) puts -0.8069 Sv of full-section barotropic
    # stage-sum difference over 2880 steps = -2.80e-4 Sv/step, which after the
    # slaved leap-frog halving is the -0.4014 Sv realized gap.  A per-step
    # barotropic deposit difference much smaller than 2.80e-4 Sv CANNOT own it.
    #
    # INSTRUMENT CONTROL (asserted, not printed): NEMO's own boxcar of the dumped
    # per-substep ua_e must reproduce spg_dump_puu_b_final.bin to roundoff -- if
    # it does not, the substep->deposit map is wrong and nothing below stands.
    print("\n  === ACC-METRIC DEPOSIT (gate weights, signed, Sv) ===")
    d2 = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    e3t_1d = np.asarray(d2.variables["e3t_1d"][0], dtype=np.float64)   # (jpk,)
    umask3 = np.asarray(d2.variables["umask"][0], dtype=np.float64)    # (jpk,jpj,jpi)
    e2u_2d = np.asarray(d2.variables["e2u"][0], dtype=np.float64)      # (jpj,jpi)
    d2.close()
    H1d = np.tensordot(e3t_1d, umask3, axes=(0, 0))          # (jpj,jpi) e3t_1d ladder
    acc_w = e2u_2d * H1d                                      # (jpj,jpi) [m^2] per m/s

    def acc_sv(dU_cell):
        """Section-integrated ACC transport of a depth-uniform u increment [Sv].

        Rows 1..197, MEAN over longitudes 2..-2.

        This is the BUDGET's linear mean surrogate (southern_term_torque_accum
        _met_full), NOT the acceptance gate's number -- the gate takes the
        MEDIAN over longitudes, which is not linear and so cannot be decomposed.
        Verified term-for-term against the budget's reducer by adversarial
        review: same e3t_1d ladder, same e2u, same longitude window, same rows
        (rows 0 and 198 carry zero wet u-faces)."""
        per_lon = (dU_cell * acc_w)[1:198, :].sum(axis=0)     # (jpi,)
        return float(per_lon[2:-2].mean()) / 1.0e6

    # NEMO's primary (velocity) boxcar weights, normalised exactly as ts_wgt does.
    wgtbtp1 = (zwgt1[:Kpit] / zwgt1[:Kpit].sum())
    # lego's own primary weights, from the SAME helper the model runs.
    wf_n = np.asarray(wf) / float(np.asarray(wt))
    print(f"    weights: max|lego_primary - NEMO_wgtbtp1| = "
          f"{np.abs(wf_n - wgtbtp1).max():.3e}   "
          f"max|lego_transport - NEMO_wgtbtp2| = {np.abs(w_tr - wgtbtp2).max():.3e}")

    # --- instrument control: rebuild NEMO's puu_b(Kaa) from its own substeps ---
    _puu_b = np.fromfile(os.path.join(SEQDUMP, "spg_dump_puu_b_final.bin"),
                         dtype="<f8").reshape(jpj + 2 * HLS, jpi + 2 * HLS
                                              )[HLS:-HLS, HLS:-HLS]
    nemo_Ubar_avg = np.zeros_like(_puu_b)
    for j in range(1, Kpit + 1):
        nemo_Ubar_avg = nemo_Ubar_avg + wgtbtp1[j - 1] * subs[j]["ua_e"]
    _wetu = (umask[0] > 0.5)
    _rel_pb = (np.abs((nemo_Ubar_avg - _puu_b)[_wetu]).max()
               / max(np.abs(_puu_b[_wetu]).max(), 1e-300))
    print(f"    INSTRUMENT CHECK: boxcar(dumped ua_e) vs spg_dump_puu_b_final "
          f"rel={_rel_pb:.3e}")
    assert _rel_pb < 1e-10, (
        "the substep->puu_b map is WRONG (rel %.3e): the per-substep deposit "
        "decomposition below cannot be trusted" % _rel_pb)

    # --- lego's own averaged velocity, from the captured carries ---------------
    def _cell(a2d):
        return a2d[:, 1:] if a2d.shape[1] == jpi + 1 else a2d

    lego_Ubar_avg = np.zeros_like(_puu_b)
    for j in range(1, Kpit + 1):
        lego_Ubar_avg = lego_Ubar_avg + wf_n[j - 1] * _cell(np.asarray(U_bar_stk[j - 1]))
    dU_avg = (lego_Ubar_avg - nemo_Ubar_avg) * _wetu
    _dep_vel = acc_sv(dU_avg)
    # transport-average route (NEMO dynspg_ts.F90:1170-1174 un_adv*r1_hu):
    _hu_now = np.maximum(hu0_full, 1e-30)
    dU_tr = (_cell(np.asarray(Hu_sum_stk[Kpit - 1])) - un_adv_cum) / _hu_now * _wetu
    _dep_tr = acc_sv(dU_tr)
    _TARGET = -2.80e-4     # Sv/step, the 90-day barotropic stage row / 2880
    print(f"    velocity-average deposit diff (lego - NEMO)  = {_dep_vel:+.4e} Sv/step"
          f"   = {100*_dep_vel/_TARGET:6.1f}% of the -2.80e-4 Sv/step budget row")
    print(f"    transport-average deposit diff (un_adv route) = {_dep_tr:+.4e} Sv/step"
          f"   = {100*_dep_tr/_TARGET:6.1f}%")

    # --- WHERE IN THE LOOP is it born: cumulative over the weighted substeps ---
    print("\n    per-substep ACC deposit, cumulative over the boxcar window [Sv]:")
    print("      jn | w_i        | this substep    | cumulative      | % of budget")
    _cum = 0.0
    _rows = []
    for j in range(1, Kpit + 1):
        _d = wf_n[j - 1] * acc_sv(
            (_cell(np.asarray(U_bar_stk[j - 1])) - subs[j]["ua_e"]) * _wetu)
        _cum += _d
        _rows.append((j, wf_n[j - 1], _d, _cum))
    for j, wi, dj, cj in _rows:
        if j <= 2 or (j >= 22 and j <= 26) or j % 8 == 0 or j >= Kpit - 2:
            print(f"      {j:2d} | {wi:.6f}   | {dj:+.4e}     | {cj:+.4e}     "
                  f"| {100*cj/_TARGET:6.1f}%")
    # fp64 summation-ORDER tolerance: the two sums differ only in the order the
    # 45 rows are added (per-substep here vs field-then-reduce above), so the
    # bound is ~n*eps on the partial sums, not on the tiny result.
    assert abs(_cum - _dep_vel) < 1e-9 * max(abs(_dep_vel), 1e-12), (
        f"substep decomposition does not close: {_cum} vs {_dep_vel}")
    print(f"    CLOSURE: sum of per-substep rows = {_cum:+.6e} == deposit "
          f"{_dep_vel:+.6e} Sv/step (rel {abs(_cum-_dep_vel)/abs(_dep_vel):.1e})")

    # --- DISCRIMINATOR: can the zu_frc residual alone explain the trajectory? --
    # If the loop DYNAMICS are faithful and only the frozen forcing differs, the
    # velocity error after jn substeps is jn*dt_s*dF (dF is held constant across
    # the whole loop on BOTH models).  Compare that prediction, in the SAME ACC
    # metric, against the measured per-substep error.  A prediction that lands
    # short means the loop AMPLIFIES rather than merely integrates.
    _acc_dF_step = acc_sv(dt_s * dF)
    print(f"\n    zu_frc residual, ACC metric: dt_s*dF = {_acc_dF_step:+.4e} Sv "
          f"per substep")
    print("      jn | predicted jn*dt_s*dF | measured err     | measured/predicted")
    for j in (1, 8, 24, 46, 68):
        _meas = acc_sv((_cell(np.asarray(U_bar_stk[j - 1])) - subs[j]["ua_e"]) * _wetu)
        _pred = j * _acc_dF_step
        _r = _meas / _pred if _pred != 0.0 else float("nan")
        print(f"      {j:2d} | {_pred:+.4e}          | {_meas:+.4e}      | {_r:8.3f}")

    # ===================================================================
    # RESPONSE FACTOR (physics review, measurement #4) -- the measurement that
    # decides whether the loop is exonerated or is itself the owner.
    #
    # The discriminator above compares the measured error to the prediction
    # jn*dt_s*dF, which assumes the loop's linear response to a STEADY forcing
    # residual is exactly 1 per substep.  Nobody measured that.  If the true
    # response is R, the forcing explains R*46*acc(dt_s*dF) of the deposit and
    # the REST is the loop's own, so "meas/pred = 0.81" is only "the loop is
    # clean" when R itself is 0.81.  Measure R directly: run the IDENTICAL loop
    # a second time with F_slow_u perturbed by +dF and nothing else changed.
    #
    #   R ~ 0.81  -> the shortfall IS the loop's linear response; residue ~ 0.
    #   R ~ 1.00  -> residue = -1.28e-3 Sv/step = 458% of the budget row, with
    #                the budget's own sign, and the loop is a prime suspect.
    print("\n  === RESPONSE FACTOR: the loop's own reply to a steady dF ===")
    _a_loop, _k_loop = captured["loop_args"]
    _Fu_prod = _k_loop["F_slow_u"]
    # dF lives on NEMO cells (nj,ni); lift it to lego's u-face frame (nj,ni+1).
    # lego face i+1 == NEMO u(i) (that is exactly what _cell inverts), and face 0
    # is the west face of cell 0, which under this grid's ZONAL PERIODICITY (the
    # channel: umask column 0 has 35 wet cells) is NEMO u(jpi-1).
    _pert = np.zeros((dF.shape[0], dF.shape[1] + 1), dtype=np.float64)
    _pert[:, 1:] = dF
    _pert[:, 0] = dF[:, -1]
    # the lift is ASSERTED, not assumed: round-tripping it through the same
    # _cell() the deposit uses must return dF bit-for-bit.
    assert np.array_equal(_cell(_pert), dF), "dF lift to the u-face frame is wrong"

    _k_pert = dict(_k_loop)
    _k_pert["F_slow_u"] = _Fu_prod + jnp.asarray(_pert, dtype=_Fu_prod.dtype)
    jax.lax.fori_loop = _fori_capture
    try:
        with jax.disable_jit():
            _orig_run(*_a_loop, **_k_pert)
    finally:
        jax.lax.fori_loop = _orig_fori
    U_bar_stk_p = np.asarray(captured["carries"][1])
    assert U_bar_stk_p.shape == U_bar_stk.shape

    _acc_dF = _acc_dF_step                      # acc_sv(dt_s*dF), one substep
    _jbar = float((wf_n * np.arange(1, Kpit + 1)).sum())   # weighted mean substep
    print(f"    weighted-mean substep index jbar = {_jbar:.3f}  "
          f"(uniform 1/45 over jn=24..68)")
    print("      jn | R(jn) = response / (jn*dt_s*dF)")
    for j in (1, 8, 24, 46, 68):
        _resp = acc_sv((_cell(np.asarray(U_bar_stk_p[j - 1]))
                        - _cell(np.asarray(U_bar_stk[j - 1]))) * _wetu)
        print(f"      {j:2d} | {_resp / (j * _acc_dF):8.4f}")
    _dU_resp = np.zeros_like(_puu_b)
    for j in range(1, Kpit + 1):
        _dU_resp = _dU_resp + wf_n[j - 1] * (
            _cell(np.asarray(U_bar_stk_p[j - 1]))
            - _cell(np.asarray(U_bar_stk[j - 1])))
    _R_dep = acc_sv(_dU_resp * _wetu) / (_jbar * _acc_dF)
    _forcing_part = _R_dep * _jbar * _acc_dF
    _residue = _dep_vel - _forcing_part
    print(f"    R_deposit = {_R_dep:.4f}   "
          f"(R=1 assumed by the naive discriminator)")
    print(f"    forcing-explained deposit = {_forcing_part:+.4e} Sv/step")
    print(f"    IN-LOOP RESIDUE           = {_residue:+.4e} Sv/step "
          f"= {100*_residue/_TARGET:.1f}% of the -2.80e-4 budget row")

    # --- free controls the physics review asked for ---------------------------
    # (i) legoESM-side deposit map: the hand-rebuilt boxcar must equal the model's
    #     OWN averaged velocity (carry 6 / w_total).  The NEMO side got two such
    #     asserts; this side had none.
    _U_sum_model = np.asarray(carries[6])[Kpit - 1]
    _lego_avg_model = _cell(_U_sum_model / float(np.asarray(wt)))
    _rel_lego = (np.abs((lego_Ubar_avg - _lego_avg_model)[_wetu]).max()
                 / max(np.abs(_lego_avg_model[_wetu]).max(), 1e-300))
    print(f"    INSTRUMENT CHECK (lego side): rebuilt boxcar vs model U_sum/w_total "
          f"rel={_rel_lego:.3e}")
    assert _rel_lego < 1e-12, "the legoESM deposit map is wrong"
    # (ii) PLANT on the NEW metric: shifting lego's trajectory one substep must
    #      move the deposit far more than the deposit itself, or the ACC
    #      reduction is shift-blind and proves nothing.
    _plant = np.zeros_like(_puu_b)
    for j in range(1, Kpit):
        _plant = _plant + wf_n[j - 1] * (
            _cell(np.asarray(U_bar_stk[j])) - subs[j]["ua_e"])
    _dep_plant = acc_sv(_plant * _wetu)
    print(f"    PLANT on the ACC deposit metric: {_dep_plant:+.4e} vs "
          f"{_dep_vel:+.4e} Sv/step -> ratio {abs(_dep_plant/_dep_vel):.2f}")
    print("    (ratio ~1 => the ACC deposit is substep-shift-INSENSITIVE: the "
          "per-substep table above DECOMPOSES the total, it does NOT localise "
          "a substep.  No localisation claim may rest on it.  The shift-\n"
          "     sensitive control for this file is the wall-max PLANT below, "
          "which is on the transport metric and does fire.)")

    # ===================================================================
    # FORCING SUBSTITUTION (physics review, measurement #5) -- the version with
    # NO linearity assumption.  Run legoESM's OWN loop on NEMO's OWN frozen
    # forcing (zu_frc / zv_frc / ssh_frc, all three dumped for this step) and
    # re-measure the deposit.  Whatever survives is IN-LOOP, by construction:
    # both models now integrate the same forcing from the same state with the
    # same weights over the same 68 substeps.
    #
    #   |deposit| collapses  -> the frozen forcing owns the difference.
    #   |deposit| survives   -> the loop owns it, and the response-factor split
    #                           above is confirmed without its u-only caveat.
    print("\n  === FORCING SUBSTITUTION: legoESM's loop on NEMO's zu/zv/ssh_frc ===")
    _zv_frc = np.fromfile(os.path.join(SEQDUMP, "spg_dump_zv_frc.bin"),
                          dtype="<f8").reshape(jpj, jpi)          # haloless
    _ssh_frc = np.fromfile(os.path.join(SEQDUMP, "spg_dump_ssh_frc.bin"),
                           dtype="<f8").reshape(jpj + 2 * HLS, jpi + 2 * HLS
                                                )[HLS:-HLS, HLS:-HLS]
    _Fv_prod = np.asarray(_k_loop["F_slow_v"])
    _Fe_prod = np.asarray(_k_loop["F_slow_eta"])

    # The v-face and cell mappings are CALIBRATED against a known answer rather
    # than assumed: legoESM's own F_slow_v already reproduces NEMO's zv_frc to
    # ~1e-6 relative under the CORRECT row alignment and to O(1) under the wrong
    # one, so the alignment is read off the data and then asserted.
    _vmask2 = np.asarray(_k_loop["v_mask"]) > 0.5
    def _v_resid(shift):
        cand = _Fv_prod[shift:shift + jpj]
        m = _vmask2[shift:shift + jpj]
        den = max(np.abs(_zv_frc[m]).max(), 1e-300)
        return float(np.abs((cand - _zv_frc)[m]).max() / den)
    _r1, _r0 = _v_resid(1), _v_resid(0)
    print(f"    v-face row alignment: resid(shift=1)={_r1:.3e}  "
          f"resid(shift=0)={_r0:.3e}")
    assert _r1 < 1e-3 and _r1 < 0.01 * _r0, (
        "v-face alignment not established from the data; substitution aborted")
    _wetc = np.asarray(_k_loop["mask"]) > 0.5
    _e_resid = float(np.abs((_Fe_prod - _ssh_frc)[_wetc]).max()
                     / max(np.abs(_ssh_frc[_wetc]).max(), 1e-300))
    print(f"    ssh cell alignment: resid={_e_resid:.3e}")
    assert _e_resid < 1e-3, "ssh_frc alignment not established; aborted"

    _Fu_sub = np.array(np.asarray(_Fu_prod))
    _Fu_sub[:, 1:] = zu_frc
    _Fu_sub[:, 0] = zu_frc[:, -1]        # zonal periodicity, established above
    _Fv_sub = np.array(_Fv_prod)
    _Fv_sub[1:1 + jpj] = _zv_frc
    _Fe_sub = np.array(_ssh_frc)
    _k_sub = dict(_k_loop)
    _k_sub["F_slow_u"] = jnp.asarray(_Fu_sub, dtype=_Fu_prod.dtype)
    _k_sub["F_slow_v"] = jnp.asarray(_Fv_sub, dtype=_Fu_prod.dtype)
    _k_sub["F_slow_eta"] = jnp.asarray(_Fe_sub, dtype=_Fu_prod.dtype)
    jax.lax.fori_loop = _fori_capture
    try:
        with jax.disable_jit():
            _orig_run(*_a_loop, **_k_sub)
    finally:
        jax.lax.fori_loop = _orig_fori
    U_bar_stk_s = np.asarray(captured["carries"][1])
    _lego_avg_sub = np.zeros_like(_puu_b)
    for j in range(1, Kpit + 1):
        _lego_avg_sub = _lego_avg_sub + wf_n[j - 1] * _cell(
            np.asarray(U_bar_stk_s[j - 1]))
    _dep_sub = acc_sv((_lego_avg_sub - nemo_Ubar_avg) * _wetu)
    print(f"    deposit with legoESM's own forcing = {_dep_vel:+.4e} Sv/step")
    print(f"    deposit with NEMO's frozen forcing = {_dep_sub:+.4e} Sv/step  "
          f"({100*_dep_sub/_dep_vel:.1f}% of it survives)")
    print(f"    -> IN-LOOP share (assumption-free) = {_dep_sub:+.4e} Sv/step "
          f"= {100*_dep_sub/_TARGET:.1f}% of the -2.80e-4 budget row")
    print(f"       FORCING share                   = {_dep_vel - _dep_sub:+.4e} "
          f"Sv/step")

    # PLANT: shift NEMO trajectory by one substep -> final diff must blow up
    un_adv_shift = np.zeros_like(hu0_full)
    for j in range(1, Kpit):
        un_adv_shift = un_adv_shift + wgtbtp2[j - 1] * nemo_flux_perwidth(subs, j + 1)
    diff_p = (lego_full - un_adv_shift) * (umask[0] > 0.5)
    print(f"  PLANT (NEMO shifted +1 substep): wall max = "
          f"{np.abs(diff_p[wall_sel]).max():.4e} "
          f"(shift-sensitivity ratio "
          f"{np.abs(diff_p[wall_sel]).max()/max(np.abs(diff[wall_sel]).max(),1e-300):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

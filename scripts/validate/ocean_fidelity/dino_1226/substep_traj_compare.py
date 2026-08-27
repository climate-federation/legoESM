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

    # #1455 PHASE-2: capture the INPUTS the barotropic drag coefficient is
    # built from.  legoESM's drag FORMULA is bit-exact with NEMO's dumped
    # rCdU_bot when fed NEMO's own now-level velocity (verified offline: 0.0
    # relative difference at the median, p99 AND max), so a runtime drag that
    # differs from NEMO's differs through its INPUT, not its arithmetic.  This
    # records which.
    _drg_cap = {}
    if os.environ.get("DINO_1455_DRAG_INPUTS"):
        # The caller imports this at FUNCTION scope, so the name must be
        # rebound on the SOURCE module, not on the caller's module object.
        import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as _pemod
        _orig_drg = _pemod.nemo_bottom_drag_rate_faces

        def _drg_spy(u, v, h_k, z_coord, config, grid):
            _drg_cap.setdefault("u", []).append(np.asarray(u))
            _drg_cap.setdefault("v", []).append(np.asarray(v))
            _drg_cap.setdefault("h_k", []).append(np.asarray(h_k))
            # WHICH SITE each call came from (#1455 sibling).  The five calls
            # were previously attributed by matching velocity fingerprints,
            # which is inference, not measurement -- and this campaign has been
            # burned by exactly that.  Record the enclosing function of the
            # caller (frame 1 = the call site inside ocean_model/barotropic).
            import traceback as _tb
            _st = _tb.extract_stack()
            _drg_cap.setdefault("caller", []).append(
                f"{os.path.basename(_st[-2].filename)}:{_st[-2].lineno}"
                f" in {_st[-2].name}")
            return _orig_drg(u, v, h_k, z_coord, config, grid)

        _pemod.nemo_bottom_drag_rate_faces = _drg_spy

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
    # than assumed: legoESM's own F_slow_v reproduces NEMO's zv_frc to ~7e-4
    # relative under the CORRECT row alignment and to O(1) under the wrong one
    # -- a discrimination of ~1500x -- so the alignment is read off the data and
    # then asserted.  (The figure quoted here used to be 1e-6, which this
    # probe's own output contradicts by ~700x; corrected 2026-08-26.  Note the
    # residual is NOT roundoff: the forcing assembly agrees at the median and
    # departs in a tail, which is a live question elsewhere on this card.)
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
    # Row 0 is outside NEMO's haloless array and keeps legoESM's own forcing.
    # For the substitution to be ONE VARIABLE that row must carry no wet v
    # face, or the arm silently mixes two forcings.  The u side handles its
    # edge explicitly by periodicity above; this one is asserted rather than
    # assumed (round-2 review: "probably harmless" is the wrong standard for a
    # one-variable claim).
    assert not bool(_vmask2[0].any()), (
        "row 0 of the v grid carries wet faces, so replacing rows 1.. only "
        "leaves legoESM's own forcing on a live row: the forcing substitution "
        "would not be a one-variable experiment")
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

    # =================================================================
    # THE MERIDIONAL (v-face) DEPOSIT.  Added 2026-08-26 after BOTH
    # adversarial reviews returned NO-SHIP on a u-only wall result for the same
    # reason: the scored walls are ZONAL walls (land to the north/south), so
    # the wall-NORMAL component there is v, and it is the convergence of the
    # wall-normal transport that sets the sea surface against a closed wall.
    # A u-only measurement at a zonal wall measures the TANGENTIAL component.
    #
    # Every input was already on disk and already parsed; only the carry index
    # changed.  The v row alignment is NOT assumed -- it is the shift=1 mapping
    # this file already calibrates against NEMO's own zv_frc above, and the two
    # reconstructions below are validated against NEMO's and legoESM's own
    # final fields exactly as the u side is.
    _v_bar_stk = np.asarray(carries[2])
    _v_bar_stk_s = np.asarray(captured["carries"][2])     # substitution run
    # The substitution run's u side had NO absolute check -- adversarial review
    # 2026-08-26 pointed out that _lego_avg_sub was the one reconstruction in
    # this file pinned by nothing, while its two siblings are pinned to 1e-10
    # and 1e-12.  Pin it the same way, against the substitution run's OWN
    # model-computed average, before anything downstream reads it.
    _u_sum_model_s = np.asarray(captured["carries"][6])[Kpit - 1]
    _lego_avg_model_s = _cell(_u_sum_model_s / float(np.asarray(wt)))
    _rel_lego_s = (np.abs((_lego_avg_sub - _lego_avg_model_s)[_wetu]).max()
                   / max(np.abs(_lego_avg_model_s[_wetu]).max(), 1e-300))
    print(f"    INSTRUMENT CHECK (lego side, SUBSTITUTED run): rebuilt boxcar "
          f"vs model U_sum/w_total rel={_rel_lego_s:.3e}")
    assert _rel_lego_s < 1e-12, "the substituted legoESM deposit map is wrong"

    def _cell_v(a2d):
        """lego v-arrays carry one extra row at the start (shift=1, calibrated
        against zv_frc above); NEMO's are haloless (jpj, jpi)."""
        return a2d[1:1 + jpj] if a2d.shape[0] == jpj + 1 else a2d

    _pvv_b = np.fromfile(os.path.join(SEQDUMP, "spg_dump_pvv_b_final.bin"),
                         dtype="<f8").reshape(jpj + 2 * HLS, jpi + 2 * HLS
                                              )[HLS:-HLS, HLS:-HLS]
    nemo_Vbar_avg = np.zeros_like(_pvv_b)
    for j in range(1, Kpit + 1):
        nemo_Vbar_avg = nemo_Vbar_avg + wgtbtp1[j - 1] * subs[j]["va_e"]
    _d4 = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    _vmask3 = np.asarray(_d4.variables["vmask"][0], dtype=np.float64)
    _e1v_2d = np.asarray(_d4.variables["e1v"][0], dtype=np.float64)
    _d4.close()
    _wetv = (_vmask3[0] > 0.5)
    _rel_pvb = (np.abs((nemo_Vbar_avg - _pvv_b)[_wetv]).max()
                / max(np.abs(_pvv_b[_wetv]).max(), 1e-300))
    print(f"    INSTRUMENT CHECK (v, NEMO): boxcar(dumped va_e) vs "
          f"spg_dump_pvv_b_final rel={_rel_pvb:.3e}")
    assert _rel_pvb < 1e-10, (
        f"the substep->pvv_b map is WRONG (rel {_rel_pvb:.3e}): the meridional "
        "deposit cannot be trusted")

    lego_Vbar_avg = np.zeros_like(_pvv_b)
    for j in range(1, Kpit + 1):
        lego_Vbar_avg = lego_Vbar_avg + wf_n[j - 1] * _cell_v(
            np.asarray(_v_bar_stk[j - 1]))
    _v_sum_model = np.asarray(carries[7])[Kpit - 1]
    _lego_vavg_model = _cell_v(_v_sum_model / float(np.asarray(wt)))
    _rel_lego_v = (np.abs((lego_Vbar_avg - _lego_vavg_model)[_wetv]).max()
                   / max(np.abs(_lego_vavg_model[_wetv]).max(), 1e-300))
    print(f"    INSTRUMENT CHECK (v, lego): rebuilt boxcar vs model "
          f"V_sum/w_total rel={_rel_lego_v:.3e}")
    assert _rel_lego_v < 1e-12, "the legoESM meridional deposit map is wrong"

    lego_Vbar_avg_sub = np.zeros_like(_pvv_b)
    for j in range(1, Kpit + 1):
        lego_Vbar_avg_sub = lego_Vbar_avg_sub + wf_n[j - 1] * _cell_v(
            np.asarray(_v_bar_stk_s[j - 1]))
    # ... and PIN IT.  Round-1 review found the u substituted run was the one
    # reconstruction constrained by nothing; round 2 found this one had the
    # same hole, on the side that carried the claim.  Same fix, same bar.
    _v_sum_model_s = np.asarray(captured["carries"][7])[Kpit - 1]
    _lego_vavg_model_s = _cell_v(_v_sum_model_s / float(np.asarray(wt)))
    _rel_lego_vs = (np.abs((lego_Vbar_avg_sub
                            - _lego_vavg_model_s)[_wetv]).max()
                    / max(np.abs(_lego_vavg_model_s[_wetv]).max(), 1e-300))
    print(f"    INSTRUMENT CHECK (v, lego, SUBSTITUTED run): rebuilt boxcar "
          f"vs model V_sum/w_total rel={_rel_lego_vs:.3e}")
    assert _rel_lego_vs < 1e-12, (
        "the substituted legoESM meridional deposit map is wrong")
    dV_avg = (lego_Vbar_avg - nemo_Vbar_avg) * _wetv
    _dV_sub = (lego_Vbar_avg_sub - nemo_Vbar_avg) * _wetv
    # v-face weight, mirroring acc_w = e2u * H1d on the u face.
    _h1d_v = np.tensordot(e3t_1d, _vmask3, axes=(0, 0))
    acc_w_v = _e1v_2d * _h1d_v
    print(f"    meridional deposit built: max|dV_avg|={np.abs(dV_avg).max():.3e}"
          f"  max|dV_sub|={np.abs(_dV_sub).max():.3e} m/s")

    # ===================================================================
    # #1455 PHASE-2: the CANDIDATE-ARRAY substitution, on TOP of the forcing
    # substitution.  The forcing substitution leaves a constant behind; the
    # pre-reduction map says that constant lives on rows 195-196, and lego's
    # own bottom-drag rate runs 1.7x/2.5x NEMO's on exactly those two rows
    # while matching it to ~7% everywhere else.  This block replaces lego's
    # drag rate with NEMO's OWN dumped rCdU_bot, averaged to faces by NEMO's
    # own dyn_drg_init rule, and re-measures.  If the drag owns the constant
    # it collapses; if it does not, it survives.
    if os.environ.get("DINO_1455_SUB_DRAG"):
        print("\n  === CANDIDATE SUBSTITUTION: NEMO's own rCdU_bot ===")
        if _k_loop.get("drag_r_u") is None:
            raise SystemExit("FATAL: barotropic drag is OFF on this card; the "
                             "substitution has nothing to replace")
        _rcd = np.fromfile(os.path.join(SEQDUMP, "drg_dump_rCdU_bot.bin"),
                           dtype="<f8").reshape(jpj + 2 * HLS, jpi + 2 * HLS
                                                )[HLS:-HLS, HLS:-HLS]
        # NEMO dyn_drg_init (dynspg_ts.F90:1614-1618), in lego's positive-r
        # convention (r = -pCdU >= 0):
        #   pCdU_u(ji,jj) = 1/2 ( rCdU_bot(ji+1,jj) + rCdU_bot(ji,jj) )
        #   pCdU_v(ji,jj) = 1/2 ( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) )
        _ru_n = np.zeros_like(_rcd)
        _ru_n[:, :-1] = -0.5 * (_rcd[:, :-1] + _rcd[:, 1:])
        _ru_n[:, -1] = -0.5 * (_rcd[:, -1] + _rcd[:, 0])     # zonal periodicity
        _rv_n = np.zeros_like(_rcd)
        _rv_n[:-1, :] = -0.5 * (_rcd[:-1, :] + _rcd[1:, :])
        _du_prod = np.asarray(_k_loop["drag_r_u"])
        _dv_prod = np.asarray(_k_loop["drag_r_v"])
        # ALIGNMENT, calibrated against a known answer rather than assumed: in
        # the deep interior BOTH models sit on the same rn_ke0 background floor,
        # so the two fields must already agree there to a few percent under the
        # CORRECT staggering and disagree by O(1) under a wrong one.
        _deep = (np.asarray(_k_loop["u_mask"])[:, 1:] > 0.5)
        _deep[190:, :] = False        # exclude the rows under test
        _deep[:10, :] = False
        # The gate is on the MEDIAN, not the max, and the reason matters: the
        # two fields agree to machine precision for the median cell (which is
        # what certifies the staggering) and differ by tens of percent in the
        # top few percent of cells -- which is the PHYSICS DIFFERENCE UNDER
        # TEST.  A max-gate here would refuse to run precisely because the
        # defect it exists to measure is present.  The tail is printed, not
        # asserted on.
        def _drag_resid(cand):
            return float(np.median((np.abs(_du_prod[:, 1:] - cand)
                                    / np.maximum(np.abs(cand), 1e-300))[_deep]))
        _med_u = _drag_resid(_ru_n)
        # A TWO-HYPOTHESIS DISCRIMINATOR, not a one-sided tolerance.  Both
        # reviews landed on this independently and they are right: in the deep
        # interior BOTH drag fields sit on the same rn_ke0 background floor, so
        # the field is very nearly CONSTANT there -- and a near-constant field
        # gives a small residual under ANY index shift.  A one-sided "median <
        # 1e-3" therefore certifies nothing, least of all the j-axis, which is
        # the axis this whole finding lives on.  The forcing substitution
        # sixty lines above already does this properly (resid(shift=1) vs
        # resid(shift=0), demanding the chosen one be 100x better); this is
        # that same test.
        _alts = {"j-shift +1": np.roll(_ru_n, 1, axis=0),
                 "j-shift -1": np.roll(_ru_n, -1, axis=0),
                 "i-shift +1": np.roll(_ru_n, 1, axis=1),
                 "no face average (T as-is)": -_rcd}
        _alt_res = {k: _drag_resid(v) for k, v in _alts.items()}
        _relf = (np.abs(_du_prod[:, 1:] - _ru_n)
                 / np.maximum(np.abs(_ru_n), 1e-300))[_deep]
        print(f"    u-face drag alignment (deep interior): median rel="
              f"{_med_u:.3e}  p95={np.percentile(_relf, 95):.3e}  "
              f"p99={np.percentile(_relf, 99):.3e}  max={_relf.max():.3e}")
        for _k, _v in _alt_res.items():
            print(f"      alternative {_k:26s}: median rel={_v:.3e}  "
                  f"({_v / max(_med_u, 1e-300):.1f}x worse)")
        _best_alt = min(_alt_res.values())
        # THE SEPARATION BAR, and its honest provenance: the first version
        # demanded 100x, copied from the forcing block sixty lines above
        # without checking that number was reachable for THIS field.  It is
        # not -- the correct mapping beats the best wrong one by 19x here, so
        # the run aborted.  The bar is 10x, which the measurement clears by a
        # factor of ~2 on its worst alternative and ~5 on its best.  Every
        # alternative's score is PRINTED above so a reader can judge the
        # separation rather than take the ratio on trust.  This is a bar set
        # to what a real discrimination supports; it is not a one-sided
        # tolerance, which is what the previous version was and what both
        # reviews rejected.
        assert _med_u < 1e-3 and _med_u * 10.0 < _best_alt, (
            "drag-face staggering NOT established: the chosen mapping scores "
            "%.3e and the best WRONG mapping scores %.3e (%.1fx), so the "
            "control cannot tell them apart and the substitution is aborted"
            % (_med_u, _best_alt, _best_alt / max(_med_u, 1e-300)))
        _dru_sub = np.array(_du_prod)
        _dru_sub[:, 1:] = _ru_n
        _dru_sub[:, 0] = _ru_n[:, -1]        # same wrap the forcing uses
        _drv_sub = np.array(_dv_prod)
        # _rv_n's last row is left at 0 by construction (the j+1 neighbour is
        # off-grid).  That is only harmless if the row is a WALL, so assert it
        # rather than trusting the comment.
        _vm_last = np.asarray(_k_loop["v_mask"])[1 + jpj - 1]
        assert float(np.abs(_vm_last).max()) < 0.5, (
            "the v-row the drag substitution leaves at zero is NOT a wall "
            "(max v_mask %.3e); the substituted arm would carry a fabricated "
            "zero drag on a wet row" % float(np.abs(_vm_last).max()))
        _drv_sub[1:1 + jpj] = _rv_n          # same row shift the forcing uses
        _k_drg = dict(_k_sub)
        _k_drg["drag_r_u"] = jnp.asarray(_dru_sub, dtype=_du_prod.dtype)
        _k_drg["drag_r_v"] = jnp.asarray(_drv_sub, dtype=_dv_prod.dtype)
        jax.lax.fori_loop = _fori_capture
        try:
            with jax.disable_jit():
                _orig_run(*_a_loop, **_k_drg)
        finally:
            jax.lax.fori_loop = _orig_fori
        _U_drg = np.asarray(captured["carries"][1])
        _lego_avg_drg = np.zeros_like(_puu_b)
        for j in range(1, Kpit + 1):
            _lego_avg_drg = _lego_avg_drg + wf_n[j - 1] * _cell(
                np.asarray(_U_drg[j - 1]))
        _dep_drg = acc_sv((_lego_avg_drg - nemo_Ubar_avg) * _wetu)
        _collapse = 100.0 * (1.0 - _dep_drg / _dep_sub)
        print(f"    in-loop deposit, lego's own drag  = {_dep_sub:+.4e} Sv/step")
        print(f"    in-loop deposit, NEMO's own drag  = {_dep_drg:+.4e} Sv/step")
        print(f"    COLLAPSE = {_collapse:.1f}%   (pre-registered: OWNER if "
              f">50%, REFUTED if <10%; point prediction 65%)")
        # The verdict text is BUILT from the measured value, never hardcoded.
        # An OVERSHOOT past 100% means the substitution pushed the deposit
        # through zero into the opposite sign.  That is over-correction, not
        # ownership, and without this band it would read as OWNER.
        _collapse_abs = 100.0 * (1.0 - abs(_dep_drg) / max(abs(_dep_sub), 1e-300))
        print(f"    magnitude collapse (sign-blind) = {_collapse_abs:.1f}%")
        _verdict = ("OVERSHOOT (over-correction, not ownership)"
                    if _collapse > 100.0 else
                    "OWNER" if _collapse > 50.0 else
                    "REFUTED" if _collapse < 10.0 else "INCONCLUSIVE")
        print(f"    VERDICT (from the measured collapse): {_verdict}")

    # The COMPOSABLE arm kwargs.  Each arm below applies its substitution to
    # `_k_arm` and reassigns it, so the JOINT arm (both substitutions at once)
    # is the same code path as either half alone rather than a third copy.
    _k_arm = _k_sub

    # ===================================================================
    # #1455 sec-D: THE PRE-REGISTERED V-FACE ZONAL METRIC ARM.
    #
    # THE CANDIDATE (docs/ocean/fidelity/dino_wall_fixed_bias.md sec 6).  NEMO
    # builds DINO's mesh isotropically and evaluates BOTH v-face scale factors
    # at its own v-point latitude `gphiv`.  legoESM's `nemo_isotropic`
    # convention fixes the v-face MERIDIONAL width but leaves the ZONAL one on
    # `R*cos(lat_v)*dlon` with `lat_v` the arithmetic MIDPOINT of the two
    # adjacent tracer rows.  On DINO's stretched meridional grid that midpoint
    # is not `gphiv`, so legoESM's v-face is zonally WIDER by up to 3.3e-05 --
    # the only horizontal metric off by more than roundoff.  It sits INSIDE the
    # barotropic loop, so the forcing substitution above does not remove it.
    #
    # THE ARM, as registered: substitute NEMO's OWN `e1v` for that one width,
    # rerun the SAME loop from the SAME entry state with the SAME (substituted)
    # forcing, and re-measure.  ONE VARIABLE -- one array.  It reaches three
    # in-loop consumers, all of which read the same width: the C-grid
    # continuity divergence's v-face length, the metric-complete EEN rotation
    # coefficient's `e1v` (the card runs `barotropic_coriolis='een_metric'`),
    # and the ssh-average face depth's `1/(e1v*e2v)`.  Feeding all three from
    # the substituted array is what makes it one variable rather than three.
    #
    # PRE-REGISTERED VERDICTS (dino_wall_fixed_bias.md sec 6, ranked test 1),
    # scored by the committed probe over the FIVE-state mean, not here:
    #   OWNER    if the state-constant wall-normal residual collapses > 50 %
    #   REFUTED  if it collapses < 10 %
    #   PARTIAL  in between.
    # This block writes the arm's own deposit maps; `baro_fixed_bias_wall_map.py`
    # is run UNCHANGED on them and its numbers carry the verdict.
    #
    # THE STAGGERING CONTROL, also registered: feed the UN-SHIFTED array.  Both
    # arms overwrite exactly the same rows, so the ONLY difference between them
    # is the row alignment.  The control passes when it is CLEARLY WORSE --
    # pre-registered here, before either arm was run, as a state-constant
    # wall-normal residual more than 50 % ABOVE the unsubstituted baseline's.
    # If the control does not fire, the loop is insensitive to this array and
    # NO verdict may be issued from the correct arm either.
    _vf_mode = os.environ.get("DINO_1455_SUB_VFACE", "")
    if _vf_mode:
        if _vf_mode not in ("nemo", "stagger"):
            raise SystemExit(
                f"DINO_1455_SUB_VFACE={_vf_mode!r}: expected 'nemo' (the "
                "registered arm) or 'stagger' (the registered staggering "
                "control).  A silent default here would run an unlabelled arm.")
        print(f"\n  === CANDIDATE SUBSTITUTION: NEMO's own e1v  [mode={_vf_mode}] ===")
        _grid_p = _k_sub["grid"]
        _dxv_p = np.asarray(_grid_p.dx_v, dtype=np.float64)
        if _dxv_p.ndim != 2 or _dxv_p.shape != (jpj + 1, jpi):
            raise SystemExit(
                f"FATAL: legoESM's stored dx_v is {_dxv_p.shape}, not the "
                f"({jpj + 1}, {jpi}) v-face frame this substitution assumes.")
        _dnc = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
        try:
            _mm = {_n: np.asarray(_dnc.variables[_n][0], dtype=np.float64)
                   for _n in ("e1v", "e2v", "e1u", "e2u", "e1t", "e2t")}
        finally:
            _dnc.close()
        _e1v_n = _mm["e1v"]
        if _e1v_n.shape != (jpj, jpi):
            raise SystemExit(f"FATAL: NEMO e1v is {_e1v_n.shape}, want "
                             f"({jpj}, {jpi})")
        # The alignment below is scored on COLUMN 0 alone.  That is sound only
        # if NEMO's e1v carries no zonal structure, so measure it rather than
        # assume it -- a column-0 score on an i-varying field would be blind to
        # exactly the half of the array it does not look at.
        _e1v_izonal = float(np.abs(_e1v_n - _e1v_n[:, :1]).max())
        if _e1v_izonal != 0.0:
            raise SystemExit(
                f"FATAL: NEMO's e1v varies along i (max spread "
                f"{_e1v_izonal:.3e}); the column-0 alignment score below would "
                "be blind to that structure.")
        # SECTION 6'S EXCLUSIVITY PREMISE, VERIFIED AT THE POINT OF USE rather
        # than relayed from prose.  The whole arm rests on "the v-face zonal
        # width is the ONLY horizontal metric off by more than roundoff": if a
        # second metric were also off, substituting one of them would be a
        # partial arm and the collapse would read LOW.  Note this also bounds
        # the mixed-metric worry -- the substitution leaves `area` on
        # legoESM's own value, which is only harmless because it IS NEMO's.
        _geom_p = _k_sub["grid"]
        _excl = {
            "dx_u vs e1u": (np.asarray(_geom_p.dx_u, np.float64)[:, :-1],
                            _mm["e1u"]),
            "dy_u vs e2u": (np.asarray(_geom_p.dy_u, np.float64)[:, :-1],
                            _mm["e2u"]),
            "dy_v vs e2v": (np.asarray(_geom_p.dy_v, np.float64)[1:jpj],
                            _mm["e2v"][0:jpj - 1]),
            "area vs e1t*e2t": (np.asarray(_geom_p.area, np.float64),
                                _mm["e1t"] * _mm["e2t"]),
        }
        for _nm, (_a, _b) in _excl.items():
            _rel = float(np.abs(_a - _b).max() / np.abs(_b).max())
            print(f"    metric exclusivity {_nm:18s}: max rel {_rel:.3e}")
            if _rel > 1e-12:
                raise SystemExit(
                    f"FATAL: {_nm} differs by {_rel:.3e}, so the v-face zonal "
                    "width is NOT the only horizontal metric off by more than "
                    "roundoff and substituting it alone is a PARTIAL arm whose "
                    "collapse would read low.")
        # ALIGNMENT, calibrated against a known answer rather than assumed --
        # the same two-hypothesis discriminator the drag arm uses.  legoESM's
        # own width must already agree with NEMO's to ~1e-5 under the CORRECT
        # row map (that residual IS the candidate) and disagree by O(1e-2)
        # under any shift, because one row of this stretched grid changes
        # cos(lat) by ~dlat*tan(lat) at 69.5 deg.  A one-sided tolerance would
        # certify nothing on a field this smooth.
        def _vf_resid(shift):
            j = np.arange(1, jpj + 1) + shift          # NEMO row for lego row j
            ok = (j >= 0) & (j < jpj) & (_dxv_p[1:jpj + 1, 0] > 0.0)
            cand = _e1v_n[np.clip(j, 0, jpj - 1), 0]
            return float(np.median(np.abs(_dxv_p[1:jpj + 1, 0][ok]
                                          - cand[ok]) / cand[ok]))
        _vf_scores = {sh: _vf_resid(sh) for sh in (-2, -1, 0, 1, 2)}
        _vf_best_wrong = min(v for sh, v in _vf_scores.items() if sh != -1)
        for _sh, _sc in sorted(_vf_scores.items()):
            print(f"    row map lego j <- NEMO j{_sh:+d}: median rel="
                  f"{_sc:.3e}"
                  + ("   <- the CORRECT map (asserted below)" if _sh == -1
                     else f"   ({_sc / max(_vf_scores[-1], 1e-300):.0f}x worse)"))
        # SystemExit, not assert: `python -O` deletes asserts, and this
        # discriminator is the only thing standing between the arm and a
        # silently mis-aligned array.  (The polar guards below already raise.)
        if not (_vf_scores[-1] < 1e-3
                and _vf_scores[-1] * 100.0 < _vf_best_wrong):
            raise SystemExit(
                "FATAL: v-face metric row alignment NOT established: the "
                "chosen map scores %.3e and the best WRONG map scores %.3e "
                "(%.1fx), so the control cannot tell them apart and the "
                "substitution is aborted"
                % (_vf_scores[-1], _vf_best_wrong,
                   _vf_best_wrong / max(_vf_scores[-1], 1e-300)))
        # The two polar v-rows are legoESM's closed-wall convention (exactly 0)
        # and have no NEMO counterpart in this map.  They are left untouched by
        # BOTH arms -- which is only one-variable-clean if they carry no wet
        # face, so assert it rather than trusting the convention.
        _vmask_full = np.asarray(_k_sub["v_mask"])
        for _jw in (0, jpj):
            if bool((np.abs(_vmask_full[_jw]) > 0.5).any()):
                raise SystemExit(
                    f"FATAL: v row {_jw} is left at legoESM's own metric by "
                    "this substitution but carries wet faces; the arm would "
                    "not be one variable.")
            if _dxv_p[_jw].max() != 0.0:
                raise SystemExit(
                    f"FATAL: v row {_jw} is not the pole-zeroed wall this "
                    f"substitution assumes (max {_dxv_p[_jw].max():.3e}).")
        _dxv_sub = np.array(_dxv_p)
        _shift = -1 if _vf_mode == "nemo" else 0
        _dxv_sub[1:jpj + 1] = _e1v_n[np.clip(np.arange(1, jpj + 1) + _shift,
                                             0, jpj - 1)]
        _dxv_sub[jpj] = _dxv_p[jpj]          # the north wall stays legoESM's 0
        _vf_touched = int((_dxv_sub != _dxv_p).sum())
        # PER-ROW, not against the basin maximum.  Dividing by the equatorial
        # width under-reports the wall rows by ~2.8x, and this is the one
        # printed number a reader checks against section 6's 3.3e-05.
        _vf_relmax = float((np.abs(_dxv_sub - _dxv_p)[1:jpj]
                            / _dxv_p[1:jpj]).max())
        print(f"    substituted rows 1..{jpj - 1} ({_vf_touched} cells); "
              f"max PER-ROW relative change {_vf_relmax:.3e}")
        # THE SUBSTITUTION.  `grid` covers every in-loop consumer that reads
        # the stored width; `een_pre` was built OUTSIDE the loop from the same
        # array, so it is replaced with the SAME array or the EEN consumer
        # would keep the old metric and the arm would not be one variable.
        _k_vfx = dict(_k_arm)
        _k_vfx["grid"] = _grid_p._replace(
            dx_v=jnp.asarray(_dxv_sub, dtype=_grid_p.dx_v.dtype))
        _een_p = _k_arm["een_pre"]
        if _een_p is None or "e1v" not in _een_p:
            raise SystemExit(
                "FATAL: the loop carries no metric-complete EEN pre-block, so "
                "the rotation-coefficient half of this candidate is not even "
                "active on this card and the arm would test something else.")
        if not np.array_equal(np.asarray(_een_p["e1v"], dtype=np.float64),
                              _dxv_p):
            raise SystemExit(
                "FATAL: the EEN pre-block's e1v is not the grid's own dx_v, so "
                "replacing both with one array is not one variable.")
        _k_vfx["een_pre"] = dict(_een_p)
        _k_vfx["een_pre"]["e1v"] = jnp.asarray(
            _dxv_sub, dtype=_een_p["e1v"].dtype)
        # ONE VARIABLE, and the gate that can actually FAIL.
        #
        # The three identity sweeps below are cheap regression tripwires for a
        # future edit, but on THIS code they are true by construction -- the
        # containers are shallow copies with one key reassigned -- so they are
        # NOT the one-variable evidence and are no longer described as it.
        # (Adversarial review, round 1: "three tautologies elevated to a
        # mechanical gate".)  The gate that can fail is the CONTENT SWEEP
        # underneath: every array reachable by the loop is compared, BY VALUE,
        # against legoESM's original width, and the set that matches must be
        # exactly the two the substitution replaced.  A THIRD match is a
        # consumer this arm MISSED -- which would shrink the perturbation and
        # bias the collapse toward REFUTED with nothing firing.
        for _kk in set(_k_vfx) | set(_k_arm):
            if _kk in ("grid", "een_pre"):
                continue
            if _k_vfx.get(_kk) is not _k_arm.get(_kk):
                raise SystemExit(f"FATAL: loop input {_kk!r} changed; the "
                                 "v-face arm is not one variable")
        for _f in _grid_p._fields:
            if _f == "dx_v":
                continue
            if getattr(_k_vfx["grid"], _f) is not getattr(_grid_p, _f):
                raise SystemExit(f"FATAL: grid field {_f!r} changed")
        for _kk in set(_k_vfx["een_pre"]) | set(_een_p):
            if _kk == "e1v":
                continue
            if _k_vfx["een_pre"].get(_kk) is not _een_p.get(_kk):
                raise SystemExit(f"FATAL: een_pre[{_kk!r}] changed")

        # REACHABILITY.  Each of the three consumers exists only under a card
        # setting, and a card that switched one off would silently shrink this
        # arm to two consumers.  Assert the settings instead of reading them
        # off a log afterwards.
        _cfg_bt = _k_sub["config"].barotropic
        from legoesm.grids.operators_latlon_cgrid import (
            reads_stored_vface_metric as _reads_stored)
        _reach = {
            "continuity divergence reads the STORED width":
                bool(_reads_stored(_grid_p)),
            "ssh-average face depth is active (barotropic_face_depth)":
                _cfg_bt.barotropic_face_depth == "nemo_ssh_avg",
            "EEN rotation coefficient is METRIC-COMPLETE":
                bool(_een_p.get("metric_complete", False)),
        }
        for _nm, _ok in _reach.items():
            print(f"    consumer reachable: {_nm}: {_ok}")
            if not _ok:
                raise SystemExit(
                    f"FATAL: {_nm} is FALSE on this card, so the substituted "
                    "array does not reach that consumer and this arm is a "
                    "partial perturbation whose collapse reads low.")

        # THE CONTENT SWEEP -- the gate that can fail.
        _matches = []
        for _kk, _vv in _k_vfx.items():
            if _kk in ("grid", "een_pre") or _vv is None:
                continue
            _av = np.asarray(_vv, dtype=np.float64) if getattr(
                _vv, "shape", None) == _dxv_p.shape else None
            if _av is not None and np.array_equal(_av, _dxv_p):
                _matches.append(f"loop kwarg {_kk}")
        for _f in _grid_p._fields:
            _vv = getattr(_grid_p, _f)
            if getattr(_vv, "shape", None) == _dxv_p.shape and np.array_equal(
                    np.asarray(_vv, dtype=np.float64), _dxv_p):
                _matches.append(f"grid.{_f}")
        for _kk, _vv in _een_p.items():
            if getattr(_vv, "shape", None) == _dxv_p.shape and np.array_equal(
                    np.asarray(_vv, dtype=np.float64), _dxv_p):
                _matches.append(f"een_pre[{_kk!r}]")
        print(f"    content sweep: arrays carrying legoESM's own v-face zonal "
              f"width = {sorted(_matches)}")
        if sorted(_matches) != ["een_pre['e1v']", "grid.dx_v"]:
            raise SystemExit(
                "FATAL: the arrays carrying legoESM's own v-face zonal width "
                f"are {sorted(_matches)}, not exactly the two this arm "
                "substitutes.  A third is a consumer the arm MISSED (partial "
                "perturbation, collapse biased low); a missing one means the "
                "substitution has nothing to replace.")
        _k_arm = _k_vfx
        jax.lax.fori_loop = _fori_capture
        try:
            with jax.disable_jit():
                _orig_run(*_a_loop, **_k_vfx)
        finally:
            jax.lax.fori_loop = _orig_fori
        _U_vfx = np.asarray(captured["carries"][1])
        _V_vfx = np.asarray(captured["carries"][2])
        _lego_avg_vfx = np.zeros_like(_puu_b)
        _lego_vavg_vfx = np.zeros_like(_pvv_b)
        for j in range(1, Kpit + 1):
            _lego_avg_vfx = _lego_avg_vfx + wf_n[j - 1] * _cell(
                np.asarray(_U_vfx[j - 1]))
            _lego_vavg_vfx = _lego_vavg_vfx + wf_n[j - 1] * _cell_v(
                np.asarray(_V_vfx[j - 1]))
        _dep_vfx = acc_sv((_lego_avg_vfx - nemo_Ubar_avg) * _wetu)
        _dV_vfx = (_lego_vavg_vfx - nemo_Vbar_avg) * _wetv
        # PER-STATE PREVIEW ONLY.  The registered quantity is the STATE-CONSTANT
        # (five-state mean) wall-normal residual and it is scored by the
        # committed probe on the maps this run writes, never here: a one-state
        # number cannot carry a verdict about a state-constant field.
        _wv_rows = [j for j in range(_wetv.shape[0]) if _wetv[j].any()]
        _j_s_v, _j_n_v = _wv_rows[0], _wv_rows[-1]
        def _rms_row(a, j):
            return float(np.sqrt((a[j][_wetv[j]] ** 2).mean()))
        print(f"    zonal (tangential) deposit: baseline {_dep_sub:+.4e} -> "
              f"arm {_dep_vfx:+.4e} Sv/step")
        for _tag, _jj in (("south", _j_s_v), ("north", _j_n_v)):
            _b, _a = _rms_row(_dV_sub, _jj), _rms_row(_dV_vfx, _jj)
            print(f"    PREVIEW (this state only) wall-normal RMS at the "
                  f"{_tag} wall row {_jj}: baseline {_b:.4e} -> arm {_a:.4e}  "
                  f"({100.0 * (1.0 - _a / max(_b, 1e-300)):+.1f}% change)")
        print("    (the REGISTERED collapse is the five-state mean, scored by "
              "baro_fixed_bias_wall_map.py on the maps written below)")
        # The maps this run writes are now the ARM's in-loop arrays, and they
        # say so in their own provenance so an arm map can never be read as a
        # baseline one.
        _lego_avg_sub = _lego_avg_vfx
        lego_Vbar_avg_sub = _lego_vavg_vfx
        _dV_sub = _dV_vfx
        _dep_sub = _dep_vfx
    # ===================================================================
    # #1455 next-action 1: THE PRE-REGISTERED VERTEX-CORIOLIS ARM.
    #
    # THE CANDIDATE (docs/ocean/fidelity/dino_campaign_synthesis.md sec 3
    # item 1; dino_wall_fixed_bias.md "The sibling error this arm uncovered").
    # legoESM builds the Coriolis parameter at the vertex as the AVERAGE of the
    # two adjacent tracer rows; NEMO evaluates it at its own f-point latitude.
    # Measured against NEMO's own dumped `ff_f` the gap is median -5.48e-05
    # over ALL mapped rows, and it peaks at the EQUATOR, which is exactly where
    # the v-face metric gap (this file's other arm) vanishes.  Their signed latitude profiles
    # correlate at +1.000 and they enter the SAME EEN rotation coefficient
    # `e1v * f` with OPPOSITE signs, so they PARTIALLY CANCEL: the v-face arm
    # removed the smaller of a cancelling pair.  That is the Rule-8 pattern
    # this campaign has recorded three times, and the response registered for
    # it is a JOINT arm, never a revert of the half already fixed.
    #
    # WHAT THIS ARM MEASURES DEPENDS ON WHEN IT WAS RUN, so read the stamp.
    # BEFORE the rotation-rate fix, this gap decomposed into a uniform
    # -1.59e-05 (a different Earth: the twin's geometry was built on legoESM's
    # rounded `constants.Omega` while NEMO's own ff_f inverts to
    # 7.292115083046e-05) and a latitude-varying -3.61e-05, summing to
    # -5.20e-05 -- the median over the |sin(phi)| >= 0.1 rows the split is
    # conditioned on, NOT the -5.48e-05 all-rows median quoted above.  Two
    # windows; they are not interchangeable.
    # AFTER the fix (one rate now reaches every site), the routing audit
    # reports the constant part as exactly 0.0 and the gap is PURE PLACEMENT,
    # so this arm is a single-variable convention arm.  An arm map produced
    # before that fix carries BOTH variables at once and must say so
    # (coriolis_omega_routing_audit.py, committed, stamps which).
    #
    # THE ARM: substitute NEMO's OWN `ff_f` for the ONE array the loop's
    # Coriolis reads, rerun the SAME loop from the SAME entry state with the
    # SAME (substituted) forcing, and re-measure.  Under this card
    # (`barotropic_coriolis="een_metric"`) the loop's Coriolis is
    # `een_barotropic_coriolis(..., een_pre)` and reads `een_pre["f_vtx"]`
    # ALONE; the `f_u`/`f_v` loop kwargs are on the dead `else` branch.  That
    # is not taken on trust -- the sentinel control below CORRUPTS them and
    # requires the loop's output to be bit-identical.
    #
    # PRE-REGISTERED VERDICTS.  Scored by `baro_fixed_bias_wall_map.py` over
    # the FIVE-state mean, never here, and REGISTERED ON THE JOINT ARM:
    #   OWNER    if the state-constant wall-normal residual collapses > 50 %
    #            both basin-wide AND on the northern lobe
    #   REFUTED  if it collapses < 10 %
    #   PARTIAL  in between.
    # The staggering control must be CLEARLY WORSE (> 50 % above baseline).
    _cor_mode = os.environ.get("DINO_1455_SUB_CORIOLIS", "")
    if _cor_mode:
        if _cor_mode not in ("nemo", "stagger"):
            raise SystemExit(
                f"DINO_1455_SUB_CORIOLIS={_cor_mode!r}: expected 'nemo' (the "
                "registered arm) or 'stagger' (the registered staggering "
                "control).  A silent default here would run an unlabelled arm.")
        print(f"\n  === CANDIDATE SUBSTITUTION: NEMO's own ff_f  "
              f"[mode={_cor_mode}] ===")
        _een_c = _k_arm["een_pre"]
        if _een_c is None or "f_vtx" not in _een_c:
            raise SystemExit(
                "FATAL: the loop carries no EEN pre-block, so the vertex "
                "Coriolis this arm substitutes is not the array the loop's "
                "Coriolis reads on this card and the arm would test nothing.")
        _fvtx_p = np.asarray(_een_c["f_vtx"], dtype=np.float64)
        if _fvtx_p.shape != (jpj + 1, jpi + 1):
            raise SystemExit(
                f"FATAL: legoESM's f_vtx is {_fvtx_p.shape}, not the "
                f"({jpj + 1}, {jpi + 1}) vertex frame this arm assumes.")
        _dnc_c = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
        try:
            _ff_f_n = np.asarray(_dnc_c.variables["ff_f"][0], dtype=np.float64)
            _gphiv_n = np.asarray(_dnc_c.variables["gphiv"][0],
                                  dtype=np.float64)
        finally:
            _dnc_c.close()
        if _ff_f_n.shape != (jpj, jpi):
            raise SystemExit(f"FATAL: NEMO ff_f is {_ff_f_n.shape}, want "
                             f"({jpj}, {jpi})")
        # The column-0 alignment score below is sound only if NEMO's ff_f
        # carries no zonal structure.  Measure it; do not assume it.
        _ff_izonal = float(np.abs(_ff_f_n - _ff_f_n[:, :1]).max())
        if _ff_izonal != 0.0:
            raise SystemExit(
                f"FATAL: NEMO's ff_f varies along i (max spread "
                f"{_ff_izonal:.3e}); the column-0 alignment score would be "
                "blind to that structure.")
        # WHICH EARTH THE SUBSTITUTED ARRAY CARRIES, recovered from the array
        # itself rather than quoted, so the arm's own stamp records what it
        # actually fed the loop.  The row spread doubles as the check that the
        # F-point latitude convention is the one assumed: a wrong latitude
        # cannot return a clean constant.
        _s_n = np.sin(np.deg2rad(_gphiv_n[:, 0]))
        _keep_n = np.abs(_s_n) >= 0.10
        _om_nemo_arr = _ff_f_n[_keep_n, 0] / (2.0 * _s_n[_keep_n])
        _om_lego_geom = float(_k_arm["grid"].omega)
        print(f"    NEMO's ff_f inverts to omega = "
              f"{np.median(_om_nemo_arr):.12e} "
              f"(row spread {float((_om_nemo_arr.max() - _om_nemo_arr.min()) / abs(np.median(_om_nemo_arr))):.2e} rel)")
        print(f"    legoESM's geometry was built on omega = "
              f"{_om_lego_geom:.12e}  (rel gap "
              f"{(_om_lego_geom - float(np.median(_om_nemo_arr))) / float(np.median(_om_nemo_arr)):+.4e})")
        # ALIGNMENT, calibrated against a known answer rather than assumed --
        # the same two-hypothesis discriminator the drag and v-face arms use.
        def _cor_resid(shift):
            j = np.arange(1, jpj + 1) + shift
            ok = (j >= 0) & (j < jpj)
            cand = _ff_f_n[np.clip(j, 0, jpj - 1), 0]
            a = _fvtx_p[1:jpj + 1, 0][ok]
            b = cand[ok]
            fine = np.abs(b) > 1e-6         # drop the equatorial row: a
            # relative score against f ~ 0 is unbounded and would swamp the
            # discriminator with one cell.
            return float(np.median(np.abs(a[fine] - b[fine])
                                   / np.abs(b[fine])))
        _cor_scores = {sh: _cor_resid(sh) for sh in (-2, -1, 0, 1, 2)}
        _cor_best_wrong = min(v for sh, v in _cor_scores.items() if sh != -1)
        for _sh, _sc in sorted(_cor_scores.items()):
            print(f"    row map lego j <- NEMO j{_sh:+d}: median rel="
                  f"{_sc:.3e}"
                  + ("   <- the CORRECT map (asserted below)" if _sh == -1
                     else f"   ({_sc / max(_cor_scores[-1], 1e-300):.0f}x worse)"))
        if not (_cor_scores[-1] < 1e-3
                and _cor_scores[-1] * 100.0 < _cor_best_wrong):
            raise SystemExit(
                "FATAL: vertex-Coriolis row alignment NOT established: the "
                "chosen map scores %.3e and the best WRONG map scores %.3e "
                "(%.1fx), so the control cannot tell them apart and the "
                "substitution is aborted"
                % (_cor_scores[-1], _cor_best_wrong,
                   _cor_best_wrong / max(_cor_scores[-1], 1e-300)))
        # The two polar v-rows have no NEMO counterpart under this map and are
        # left at legoESM's own value by BOTH arms.  That is one-variable-clean
        # only if they carry no wet face -- asserted, not assumed.
        # THE VERTEX mask, not the face mask.  `pv_flux_al81_partial_cell`
        # multiplies the triad by `vtx_mask`, so a vertex row that is dry on
        # v-FACES but live in the triad would slip past a face-mask check
        # (adversarial review, round 1).  It is in the same dict.
        _vtxm_c = np.asarray(_een_c["vtx_mask"])
        if bool((np.abs(_vtxm_c[0]) > 0.5).any()):
            print("    NOTE: south wall vertex row 0 is LIVE in the triad "
                  "mask and has no NEMO counterpart under this map, so it "
                  "stays at legoESM's own first-order tracer-row value -- a "
                  "floor this arm cannot reach. The reachability control "
                  "below measures whether the loop actually reads it.")
        # THE SUBSTITUTION, built the SAME way `vertex_coriolis` builds the
        # array it replaces (rows, then the periodic wrap column), so the two
        # differ in their VALUES and in nothing else.
        _fv_sub = np.array(_fvtx_p[:, :jpi])
        _shift_c = -1 if _cor_mode == "nemo" else 0
        _rows_c = np.arange(1, jpj + 1) + _shift_c
        if _rows_c.min() < 0 or _rows_c.max() > jpj - 1:
            # np.clip would silently DUPLICATE an end row instead of failing.
            _bad = _rows_c[(_rows_c < 0) | (_rows_c > jpj - 1)]
            print(f"    NOTE: shift {_shift_c:+d} maps {_bad.size} lego row(s) "
                  f"outside NEMO's [0,{jpj - 1}]; those rows are left at "
                  "legoESM's own value rather than clipped onto a duplicate.")
        _keep_c = (_rows_c >= 0) & (_rows_c <= jpj - 1)
        _fv_sub[1:jpj + 1][_keep_c] = _ff_f_n[_rows_c[_keep_c]]
        # THE END ROWS STAY AT legoESM's OWN VALUE, and what that costs is
        # MEASURED by the reachability control below rather than argued.
        #
        # Adversarial review (round 2) is right that these are the worst rows
        # in the array and wrong that leaving them is free-by-assumption.
        # legoESM's f_v carries the TRACER-row value at both ends rather than a
        # face value -- a FIRST-ORDER half-cell error, not the second-order
        # convention this arm is about.  Measured on the DINO mesh: -1.108e-03
        # at row jpj against a -2.50e-05 neighbour and a -5.48e-05 median, i.e.
        # 20x the median and 44x its own neighbour.  The south row 0 maps to
        # NEMO row -1 and has no counterpart at all; the north row jpj DOES
        # have one (NEMO row jpj-1), so leaving it is a choice, not a
        # necessity.
        #
        # It is left ANYWAY, for one reason and only if that reason holds: the
        # last wet v-face on this configuration is row 196, the AL81 triad at
        # v-face j reads vertex rows j and j+1, and so no wet face can reach
        # vertex row jpj=199.  That is a claim about reachability, and the
        # control below CORRUPTS both end rows by 1e3 and requires the carry
        # back bit-identical.  If it is not bit-identical this arm is partial
        # at its worst rows and must be re-run with them substituted.
        _fv_sub[jpj] = _fvtx_p[jpj, :jpi]     # the north wall stays legoESM's
        _fvtx_sub = np.concatenate([_fv_sub, _fv_sub[:, 0:1]], axis=1)
        _cor_touched = int((_fvtx_sub != _fvtx_p).sum())
        _fint = slice(1, jpj)
        _cor_relmax = float((np.abs(_fvtx_sub - _fvtx_p)[_fint]
                             / np.maximum(np.abs(_fvtx_p[_fint]), 1e-12)).max())
        _cor_relmed = float(np.median(
            np.abs(_fvtx_sub - _fvtx_p)[_fint]
            / np.maximum(np.abs(_fvtx_p[_fint]), 1e-12)))
        print(f"    substituted rows 1..{jpj - 1} ({_cor_touched} cells); "
              f"median PER-CELL relative change {_cor_relmed:.3e}, "
              f"max {_cor_relmax:.3e} (the max sits at the equator, where f "
              f"-> 0 and a RELATIVE change is unbounded)")
        _k_cor = dict(_k_arm)
        _k_cor["een_pre"] = dict(_een_c)
        _k_cor["een_pre"]["f_vtx"] = jnp.asarray(
            _fvtx_sub, dtype=_een_c["f_vtx"].dtype)

        # ONE VARIABLE.  Identity sweep first (a cheap regression tripwire,
        # true by construction here), then the CONTENT sweep that can fail.
        for _kk in set(_k_cor) | set(_k_arm):
            if _kk == "een_pre":
                continue
            if _k_cor.get(_kk) is not _k_arm.get(_kk):
                raise SystemExit(f"FATAL: loop input {_kk!r} changed; the "
                                 "Coriolis arm is not one variable")
        for _kk in set(_k_cor["een_pre"]) | set(_een_c):
            if _kk == "f_vtx":
                continue
            if _k_cor["een_pre"].get(_kk) is not _een_c.get(_kk):
                raise SystemExit(f"FATAL: een_pre[{_kk!r}] changed")
        # THE CONTENT SWEEP.  It compares the UN-WRAPPED form (the first jpi
        # columns), not the full vertex frame: `f_vtx` IS `grid.f_v` plus a
        # periodic wrap column, so a shape-exact filter is blind to the
        # narrower array carrying the identical values -- which is precisely
        # the consumer a missed-consumer check exists to find (adversarial
        # review, round 1).  Anything whose rows match and whose first jpi
        # columns equal legoESM's own vertex Coriolis is reported.
        _fvtx_core = _fvtx_p[:, :jpi]

        def _carries_lego_f(v):
            sh = getattr(v, "shape", None)
            if sh is None or len(sh) != 2 or sh[0] != _fvtx_p.shape[0]:
                return False
            if sh[1] < jpi:
                return False
            return np.array_equal(
                np.asarray(v, dtype=np.float64)[:, :jpi], _fvtx_core)

        _cmatches = []
        for _kk, _vv in _k_cor.items():
            if _kk == "een_pre" or _vv is None:
                continue
            if _carries_lego_f(_vv):
                _cmatches.append(f"loop kwarg {_kk}")
        for _f in _k_arm["grid"]._fields:
            if _carries_lego_f(getattr(_k_arm["grid"], _f)):
                _cmatches.append(f"grid.{_f}")
        for _kk, _vv in _een_c.items():
            if _carries_lego_f(_vv):
                _cmatches.append(f"een_pre[{_kk!r}]")
        print(f"    content sweep: arrays carrying legoESM's own vertex "
              f"Coriolis = {sorted(_cmatches)}")
        # THREE names are EXPECTED, and the widened sweep is what found the
        # third: `grid.f_v` is the same field one column narrower, and the
        # `f_v` loop kwarg IS that same array handed in separately.  Neither is
        # substituted, and the reachability control above -- not this list --
        # is what establishes the loop never reads either.  A FOURTH name is a
        # consumer this arm missed.  (The shape-exact version of this sweep saw
        # only the first name and would have reported "exactly one" while two
        # more sat in the same dict: adversarial review, round 1.)
        _EXPECTED_F_CARRIERS = ["een_pre['f_vtx']", "grid.f_v", "loop kwarg f_v"]
        if sorted(_cmatches) != sorted(_EXPECTED_F_CARRIERS):
            raise SystemExit(
                "FATAL: the arrays carrying legoESM's own vertex Coriolis are "
                f"{sorted(_cmatches)}, not {sorted(_EXPECTED_F_CARRIERS)}. "
                "An EXTRA name is a consumer the arm MISSED (partial "
                "perturbation, collapse biased low); a MISSING one means the "
                "substitution has nothing to replace.")

        # THE REACHABILITY CONTROL -- everything carrying a Coriolis value
        # that this arm does NOT substitute, corrupted at once, with the
        # loop's carry required back BIT-IDENTICAL.  Anything the loop
        # actually reads cannot survive a factor of 1e3.
        #
        # Three families, and each is here because leaving it out would make
        # the arm a PARTIAL perturbation whose collapse reads low with nothing
        # announcing it:
        #   * the `f_u`/`f_v` loop kwargs -- they still carry legoESM's
        #     rounded Earth, and they sit on the `else` of `elif een_pre is not
        #     None`.  Reading that branch is not proof it is dead.
        #   * the GRID's own `f_u`/`f_v`/`omega` -- adversarial review, round 1:
        #     the content sweep below compares only arrays shaped like f_vtx,
        #     so `grid.f_v` (the same values, one column narrower) is invisible
        #     to it.  A loop consumer rebuilding f from the grid would be
        #     missed entirely.
        #   * the SOUTH wall vertex row 0, which has no NEMO counterpart and is
        #     left at legoESM's own first-order tracer-row value.  If it is
        #     reachable, this arm has a floor it can never reach and must say
        #     so; if it is not, leaving it is free.
        #
        # A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL, so every array's
        # own magnitude is printed and asserted nonzero before it is scaled.
        _k_inert = dict(_k_arm)
        _g_in = _k_arm["grid"]
        _corrupt = {"f_u kwarg": np.abs(np.asarray(_k_arm["f_u"])).max(),
                    "f_v kwarg": np.abs(np.asarray(_k_arm["f_v"])).max(),
                    "grid.f_u": np.abs(np.asarray(_g_in.f_u)).max(),
                    "grid.f_v": np.abs(np.asarray(_g_in.f_v)).max(),
                    "grid.f_T": np.abs(np.asarray(_g_in.f_T)).max(),
                    "grid.omega": abs(float(_g_in.omega)),
                    "een_pre f_vtx row 0": np.abs(_fvtx_p[0]).max(),
                    "een_pre f_vtx row jpj": np.abs(_fvtx_p[jpj]).max()}
        for _nm, _mx in _corrupt.items():
            print(f"    reachability control corrupts {_nm:22s} max|.|="
                  f"{float(_mx):.4e}")
            if float(_mx) <= 0.0:
                raise SystemExit(
                    f"FATAL: {_nm} is identically zero, so scaling it is a "
                    "no-op and this control proves nothing about it.")
        _k_inert["f_u"] = _k_arm["f_u"] * 1.0e3
        _k_inert["f_v"] = _k_arm["f_v"] * 1.0e3
        _k_inert["grid"] = _g_in._replace(
            f_u=_g_in.f_u * 1.0e3, f_v=_g_in.f_v * 1.0e3,
            f_T=_g_in.f_T * 1.0e3, omega=float(_g_in.omega) * 1.0e3)
        _fvtx_row0 = np.array(_fvtx_p)
        _fvtx_row0[0] = _fvtx_p[0] * 1.0e3
        _fvtx_row0[jpj] = _fvtx_p[jpj] * 1.0e3
        _k_inert["een_pre"] = dict(_een_c)
        _k_inert["een_pre"]["f_vtx"] = jnp.asarray(
            _fvtx_row0, dtype=_een_c["f_vtx"].dtype)
        jax.lax.fori_loop = _fori_capture
        try:
            with jax.disable_jit():
                _orig_run(*_a_loop, **_k_inert)
        finally:
            jax.lax.fori_loop = _orig_fori
        _U_inert = np.asarray(captured["carries"][1])
        jax.lax.fori_loop = _fori_capture
        try:
            with jax.disable_jit():
                _orig_run(*_a_loop, **_k_arm)
        finally:
            jax.lax.fori_loop = _orig_fori
        _U_base_c = np.asarray(captured["carries"][1])
        _inert_gap = float(np.abs(_U_inert - _U_base_c).max())
        print(f"    reachability control: every unsubstituted Coriolis array "
              f"scaled by 1e3 -> max|dU_bar| = {_inert_gap:.3e} (must be 0.0)")
        if _inert_gap != 0.0:
            raise SystemExit(
                "FATAL: an unsubstituted Coriolis array is LIVE in the loop "
                f"(scaling them all by 1e3 moved the carry by {_inert_gap:.3e}"
                "), so substituting only een_pre['f_vtx'] is a PARTIAL "
                "perturbation and the collapse would read low.  Isolate which "
                "of the seven by re-running them one at a time, then substitute "
                "it too before anything is scored.")

        jax.lax.fori_loop = _fori_capture
        try:
            with jax.disable_jit():
                _orig_run(*_a_loop, **_k_cor)
        finally:
            jax.lax.fori_loop = _orig_fori
        _k_arm = _k_cor
        _U_cor = np.asarray(captured["carries"][1])
        _V_cor = np.asarray(captured["carries"][2])
        _lego_avg_cor = np.zeros_like(_puu_b)
        _lego_vavg_cor = np.zeros_like(_pvv_b)
        for j in range(1, Kpit + 1):
            _lego_avg_cor = _lego_avg_cor + wf_n[j - 1] * _cell(
                np.asarray(_U_cor[j - 1]))
            _lego_vavg_cor = _lego_vavg_cor + wf_n[j - 1] * _cell_v(
                np.asarray(_V_cor[j - 1]))
        _dep_cor = acc_sv((_lego_avg_cor - nemo_Ubar_avg) * _wetu)
        _dV_cor = (_lego_vavg_cor - nemo_Vbar_avg) * _wetv
        # PER-STATE PREVIEW ONLY.  The registered quantity is the five-state
        # mean and it is scored by the committed probe on the maps this run
        # writes, never here.
        _wv_rows_c = [j for j in range(_wetv.shape[0]) if _wetv[j].any()]
        _j_s_c, _j_n_c = _wv_rows_c[0], _wv_rows_c[-1]

        def _rms_row_c(a, j):
            return float(np.sqrt((a[j][_wetv[j]] ** 2).mean()))
        print(f"    zonal (tangential) deposit: entering {_dep_sub:+.4e} -> "
              f"arm {_dep_cor:+.4e} Sv/step")
        for _tag, _jj in (("south", _j_s_c), ("north", _j_n_c)):
            _b, _a = _rms_row_c(_dV_sub, _jj), _rms_row_c(_dV_cor, _jj)
            print(f"    PREVIEW (this state only) wall-normal RMS at the "
                  f"{_tag} wall row {_jj}: entering {_b:.4e} -> arm {_a:.4e}  "
                  f"({100.0 * (1.0 - _a / max(_b, 1e-300)):+.1f}% change)")
        print("    (the REGISTERED collapse is the five-state mean, scored by "
              "baro_fixed_bias_wall_map.py on the maps written below)")
        _lego_avg_sub = _lego_avg_cor
        lego_Vbar_avg_sub = _lego_vavg_cor
        _dV_sub = _dV_cor
        _dep_sub = _dep_cor

    # LABEL THE ARM.  When the v-face substitution is on, `_dep_sub` below is
    # the SUBSTITUTED-METRIC in-loop deposit, not the plain frozen-forcing one;
    # the four lines would otherwise report an arm number under the baseline's
    # name.  The line prefixes are unchanged because the walk driver parses
    # them; the arm is named after the number.
    _vf_tag = ("   [" + " + ".join(
        ([f"NEMO's own e1v, mode={_vf_mode}"] if _vf_mode else [])
        + ([f"NEMO's own ff_f, mode={_cor_mode}"] if _cor_mode else [])
    ) + "]") if (_vf_mode or _cor_mode) else ""
    print(f"    deposit with legoESM's own forcing = {_dep_vel:+.4e} Sv/step")
    print(f"    deposit with NEMO's frozen forcing = {_dep_sub:+.4e} Sv/step  "
          f"({100*_dep_sub/_dep_vel:.1f}% of it survives){_vf_tag}")
    print(f"    -> IN-LOOP share (assumption-free) = {_dep_sub:+.4e} Sv/step "
          f"= {100*_dep_sub/_TARGET:.1f}% of the -2.80e-4 budget row{_vf_tag}")
    print(f"       FORCING share                   = {_dep_vel - _dep_sub:+.4e} "
          f"Sv/step{_vf_tag}")

    # ===================================================================
    # #1455 PHASE-2: the PRE-REDUCTION deposit fields, written only when asked.
    #
    # acc_sv() collapses a (jpj,jpi) velocity increment to one number in two
    # steps -- a sum over rows 1..197 giving a per-longitude transport vector,
    # then a mean over longitudes 2..-2.  Both intermediates are needed by the
    # Phase-2 measurements and neither was ever persisted:
    #   * the per-longitude vector bounds the cancellation caveat (max|per_lon|
    #     against |mean|), which this campaign has carried unbounded since
    #     eb3f6d23d;
    #   * the 2-D map at two states discriminates a STATIC-ARRAY defect (pattern
    #     identical at both states) from a state-dependent reduction artifact;
    #   * dU_avg / dU_sub are the deposit-shaped barotropic velocity increments
    #     that the retention measurement injects into a free-running twin.
    #
    # Nothing here changes a printed number: the arrays saved are the exact
    # operands acc_sv() was already called on, and acc_sv() is re-run on them
    # below as a round-trip check that the saved fields reproduce the printed
    # deposits.
    # The loop's STATIC arrays, saved alongside the map when asked.  The forcing
    # substitution replaces F_slow_u/v/eta and NOTHING else, so whatever survives
    # it is carried by one of these -- they are the candidate list the in-loop
    # constant has to be named from.
    if os.environ.get("DINO_1455_DRAG_INPUTS"):
        _o = os.environ["DINO_1455_DRAG_INPUTS"]
        if not _drg_cap.get("u"):
            raise SystemExit(
                "FATAL: DINO_1455_DRAG_INPUTS was requested but the drag "
                "helper was never called -- the capture missed its target, "
                "which is exactly the failure mode this branch has hit before")
        os.makedirs(os.path.dirname(os.path.abspath(_o)), exist_ok=True)
        # EVERY call, not just the first.  The helper is called from several
        # stages and only ONE of them is the barotropic loop's; saving call 0
        # and assuming it was the right one produced a field that did not
        # reproduce the runtime coefficient, which is how this was noticed.
        _saved = {"n_calls": np.int64(len(_drg_cap["u"])),
                  "caller": np.array(_drg_cap.get("caller", []), dtype=object)}
        for _i in range(len(_drg_cap["u"])):
            _saved[f"u{_i}"] = _drg_cap["u"][_i]
            _saved[f"v{_i}"] = _drg_cap["v"][_i]
            _saved[f"h_k{_i}"] = _drg_cap["h_k"][_i]
        np.savez(_o, **_saved)
        print(f"    [drag inputs] wrote {_o}  calls={len(_drg_cap['u'])}  "
              f"u{_drg_cap['u'][0].shape}")

    _stat_out = os.environ.get("DINO_1455_STATIC_ARRAYS")
    if _stat_out:
        _stat = {}
        for _nm in ("H_bathy", "mask", "u_mask", "v_mask", "area",
                    "f_u", "f_v", "drag_r_u", "drag_r_v"):
            _v = _k_loop.get(_nm)
            if _v is not None:
                _stat[_nm] = np.asarray(_v)
        _stat["eta_entry"] = np.asarray(_a_loop[0])
        _stat["U_bar_entry"] = np.asarray(_a_loop[1])
        for _nm, _v in _stat.items():
            if not np.all(np.isfinite(_v)):
                raise SystemExit(f"FATAL: non-finite values in {_nm}")
        os.makedirs(os.path.dirname(os.path.abspath(_stat_out)), exist_ok=True)
        np.savez(_stat_out, **_stat)
        print(f"    [static arrays] wrote {_stat_out}: "
              + ", ".join(f"{k}{v.shape}" for k, v in _stat.items()))

    _map_out = os.environ.get("DINO_1455_DEPOSIT_MAP")
    if _map_out:
        _dU_sub = (_lego_avg_sub - nemo_Ubar_avg) * _wetu
        _pl_tot = (dU_avg * acc_w)[1:198, :].sum(axis=0)
        _pl_sub = (_dU_sub * acc_w)[1:198, :].sum(axis=0)
        # round-trip: the saved operands must reproduce the printed scalars
        _rt_tot, _rt_sub = acc_sv(dU_avg), acc_sv(_dU_sub)
        assert _rt_tot == _dep_vel and _rt_sub == _dep_sub, (
            "saved deposit operands do not reproduce the printed deposits "
            f"({_rt_tot!r} vs {_dep_vel!r}, {_rt_sub!r} vs {_dep_sub!r})")
        for _nm, _a in (("dU_avg", dU_avg), ("dU_sub", _dU_sub),
                        ("acc_w", acc_w), ("per_lon_total", _pl_tot),
                        ("per_lon_in_loop", _pl_sub),
                        ("lego_Ubar_avg", lego_Ubar_avg),
                        ("nemo_Ubar_avg", nemo_Ubar_avg),
                        ("lego_Ubar_avg_sub", _lego_avg_sub),
                        ("dV_avg", dV_avg), ("dV_sub", _dV_sub),
                        ("acc_w_v", acc_w_v),
                        ("lego_Vbar_avg", lego_Vbar_avg),
                        ("nemo_Vbar_avg", nemo_Vbar_avg),
                        ("lego_Vbar_avg_sub", lego_Vbar_avg_sub)):
            if not np.all(np.isfinite(_a)):
                raise SystemExit(f"FATAL: non-finite values in {_nm}")
        os.makedirs(os.path.dirname(os.path.abspath(_map_out)), exist_ok=True)
        np.savez(_map_out,
                 dU_avg=dU_avg, dU_sub=_dU_sub, acc_w=acc_w,
                 wetu=_wetu.astype(np.int8),
                 # The two SIDES, not only their difference.  A difference
                 # field answers "do the two disagree"; it cannot answer "which
                 # one carries the signal", and a two-step hunt needs the
                 # latter -- an alternating difference is produced equally by
                 # lego alternating and by NEMO alternating.
                 lego_Ubar_avg=lego_Ubar_avg, nemo_Ubar_avg=nemo_Ubar_avg,
                 lego_Ubar_avg_sub=_lego_avg_sub,
                 # the MERIDIONAL side: the wall-normal component at a zonal
                 # wall, which a u-only projection cannot see.
                 # legoESM's own primary boxcar weights, so a downstream
                 # projection can MEASURE this window's attenuation of a
                 # substep-alternating source instead of quoting a number.
                 wgt_primary=np.asarray(wf_n, dtype=np.float64),
                 dV_avg=dV_avg, dV_sub=_dV_sub, acc_w_v=acc_w_v,
                 wetv=_wetv.astype(np.int8),
                 lego_Vbar_avg=lego_Vbar_avg, nemo_Vbar_avg=nemo_Vbar_avg,
                 lego_Vbar_avg_sub=lego_Vbar_avg_sub,
                 per_lon_total=_pl_tot, per_lon_in_loop=_pl_sub,
                 dep_total=np.float64(_dep_vel), dep_in_loop=np.float64(_dep_sub),
                 ic_step=np.int64(mr.IC_STEP), seqdump=np.array(SEQDUMP),
                 # The v-face metric arm is stamped INTO the map, both
                 # as its own key and in the provenance string, so an
                 # arm map can never be read as a baseline one -- the
                 # two are well-formed arrays on an identical wet mask
                 # and no numerical guard could tell them apart.
                 vface_arm=np.array(os.environ.get(
                     "DINO_1455_SUB_VFACE", "") or "none"),
                 coriolis_arm=np.array(os.environ.get(
                     "DINO_1455_SUB_CORIOLIS", "") or "none"),
                 provenance=np.array(
                     mr.provenance("substep_deposit_map")
                     + " DINO_1455_SUB_VFACE=%r DINO_1455_SUB_CORIOLIS=%r" % (
                         os.environ.get("DINO_1455_SUB_VFACE") or None,
                         os.environ.get("DINO_1455_SUB_CORIOLIS") or None)))
        # Round-trip from the FILE, not from the in-memory operands: an
        # assertion that re-runs acc_sv() on the same objects it was just
        # called on is true by construction and proves nothing about what
        # landed on disk.
        _rb = np.load(_map_out)
        _rt2_tot = float((_rb["dU_avg"] * _rb["acc_w"])[1:198, :]
                         .sum(axis=0)[2:-2].mean()) / 1.0e6
        _rt2_sub = float((_rb["dU_sub"] * _rb["acc_w"])[1:198, :]
                         .sum(axis=0)[2:-2].mean()) / 1.0e6
        assert _rt2_tot == _dep_vel and _rt2_sub == _dep_sub, (
            "the SAVED FILE does not reproduce the printed deposits "
            f"({_rt2_tot!r} vs {_dep_vel!r}, {_rt2_sub!r} vs {_dep_sub!r})")
        # A ROUND-TRIP, NOT A CONTROL, and labelled as one after adversarial
        # review 2026-08-26 showed the earlier wording over-claimed it.  The
        # difference is DEFINED as (lego - nemo)*wet from these very operands,
        # so this equality is structural: it can catch a wrong array bound to a
        # keyword or a storage/dtype fault, and it can catch NOTHING about the
        # per-side ABSOLUTE levels that the side statistic reads.  Those are
        # constrained upstream instead, and each of the four now has its own
        # assert: nemo_Ubar_avg vs spg_dump_puu_b_final (1e-10), nemo_Vbar_avg
        # vs spg_dump_pvv_b_final (1e-10), and both legoESM boxcars vs the
        # model's own U_sum/V_sum (1e-12), plus the substituted run's.
        _wb = _rb["wetu"].astype(bool)
        for _nm, _a, _b in (
                ("dU_avg", (_rb["lego_Ubar_avg"] - _rb["nemo_Ubar_avg"]) * _wb,
                 _rb["dU_avg"]),
                ("dU_sub",
                 (_rb["lego_Ubar_avg_sub"] - _rb["nemo_Ubar_avg"]) * _wb,
                 _rb["dU_sub"]),
                ("dV_avg",
                 (_rb["lego_Vbar_avg"] - _rb["nemo_Vbar_avg"])
                 * _rb["wetv"].astype(bool), _rb["dV_avg"]),
                ("dV_sub",
                 (_rb["lego_Vbar_avg_sub"] - _rb["nemo_Vbar_avg"])
                 * _rb["wetv"].astype(bool), _rb["dV_sub"])):
            if not np.array_equal(_a, _b):
                raise SystemExit(
                    f"FATAL: the saved sides do not reconstruct {_nm} "
                    f"(max|d|={np.abs(_a - _b).max():.3e}); a per-side "
                    "two-step projection would name the wrong model.")
        # WHERE THE DEPOSIT LIVES IN LATITUDE, against the band the ACC
        # actually occupies.  The section reducer sums EVERY latitude and is
        # only a circumpolar-transport surrogate because the closed-basin rows
        # cancel; a wall-row error breaks exactly that cancellation.  So "how
        # much of this deposit is even inside the channel" is a first-class
        # number, not a diagnostic -- and it was never measured.
        import netCDF4 as _nc2
        _d3 = _nc2.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
        _lat_u = np.asarray(_d3.variables["gphiu"][0]).squeeze()
        _d3.close()
        _rows = np.arange(1, 198)
        _latc = _lat_u[1:198, jpi // 2]
        # the re-entrant channel: rows whose latitude band carries the ACC,
        # taken from the geometry rather than a literal (the southern rows
        # with no land between the meridional walls).
        _chan = (_latc >= -65.0) & (_latc <= -45.0)
        for _nm, _fld in (("total", dU_avg), ("in-loop", _dU_sub)):
            _C = (_fld * acc_w)[1:198, 2:-2] / 1.0e6 / float(
                acc_w[1:198, 2:-2].shape[1])
            _t = _C.sum()
            _in = _C[_chan].sum()
            _no = _C[_latc >= 66.0].sum()
            print(f"    [latitude] {_nm:8s} deposit {_t:+.4e} Sv | inside the "
                  f"ACC channel band (45S-65S) {_in:+.4e} "
                  f"({100 * _in / _t:5.1f}%) | north of 66N "
                  f"{_no:+.4e} ({100 * _no / _t:5.1f}%)")
        print(f"    [deposit map] wrote {_map_out}  "
              f"dU_avg{dU_avg.shape} per_lon{_pl_tot.shape}  "
              f"cancellation max|per_lon|/|mean| total="
              f"{np.abs(_pl_tot[2:-2]).max()/max(abs(_pl_tot[2:-2].mean()),1e-300):.3g}"
              f" in-loop="
              f"{np.abs(_pl_sub[2:-2]).max()/max(abs(_pl_sub[2:-2].mean()),1e-300):.3g}")

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

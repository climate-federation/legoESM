#!/usr/bin/env python
"""#1455 sec-D PHASE 1 -- decompose the F_slow_u vs NEMO zu_frc residual (ΔF) by
STAGE/TERM, to name WHICH constituent of the barotropic slow-mode forcing carries
the sign-coherent ~2%-median (max 9.5e-8 m/s^2) residual established by
substep_traj_compare.py.

CHAIN (all prior-confirmed): residual 3-D momentum-RHS gap -> ΔF=(F_slow_u - zu_frc)
-> x68 substeps -> 0.88 m^2/s wall transport bias -> 4.2e-3 m/step eta injection
-> the sec-D ACC anomaly.  substep_traj_compare.py proved the substep LOOP is
faithful (substep-1 bit-identical) and dt_s*ΔF reproduces the per-substep velocity
error to 4 s.f. -- so the whole bias lives in F_slow_u.  This probe opens F_slow_u.

NEMO zu_frc assembly (dynspg_ts.F90, DINO MLF key_qco -- see PHASE0 doc, read
from source):
  base  = SUM_k( e3u_0[k]*puu[k,Krhs]*umask[k] ) * r1_hu_0     (:337, REST wt)
          where puu(Krhs) at dyn_spg entry = adv+vor+ldf+hpg   (stpmlf :303-318)
  -cor  = - dyn_cor_2D(puu_b(Kmm)) * ssumask                   (:361-369)
  +drg  = dyn_drg_init increment (ln_drgimp=T)                 (:382-383)  [drg_dump]
  +wnd  = r1_rho0*0.5*(utau_b+utauU)*r1_hu(Kmm)                (:443)      [wnd_dump]
  ( ssh_ib inverse-barometer: ln_apr_dyn=F -> ZERO )

legoESM F_slow_u (ocean_model_latlon_cgrid.py:3480-3485): depth-mean of the full
3-D du_dt with REST thickness h_u_pre; wind is IN du_dt (top cell, surface_stress_
implicit=False); f*V removed later (coriolis_scheme=explicit_ab2); drag done
in-loop (barotropic_drag_substep=True) -- NOT in F_slow_u.

DECOMPOSITION (depth-mean is LINEAR, so F_slow_u = Σ_term depthmean_H(term_k)):
  lego per-term depth-means (via probe_latlon_cgrid + the model's own h_u_pre):
    wind        = depthmean(phys_u)                  <-> NEMO +wnd  (wnd_dump)
    adv+hpg+ke  = depthmean(pgf_ke_u + vertadv_u)    <-> NEMO base adv+hpg part
    vor         = depthmean(vortcor_u) [+coriolis_u] <-> NEMO base vor - cor2D removal
    ldf         = depthmean(ah_lap_u + bh_bilap_u)   <-> NEMO base ldf
    (botdrag_u/av_vert_u: NOT in F_slow_u for this card -- verified below == 0 share)

We do NOT need to reconstruct the 3-D dump_06 depth-mean by hand (halo-fragile).
Instead: the DIRECT test is ΔF vs each NEMO CONSTITUENT DUMP.  If ΔF is spatially
proportional to one dump (corr ~1, matched sign, matched magnitude), that term owns
it.  We also report each lego per-term depth-mean's own magnitude so the reader sees
which term is even large enough to carry a 9.5e-8 residual.

CONTROLS (skill 1c/1d/2/3/7): fp64; LEGOESM_NEMO_E3T=both; day-0 bit-identity via
build_replay_ic; card state printed; dtypes printed; reproduce the recorded ΔF
(max 9.49e-8, med 1.70e-8) BEFORE decomposing (Rule 1e); planted control per
constituent correlation (shuffle -> corr collapses).

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

_THIS = os.path.dirname(os.path.abspath(__file__))
for _p in (_THIS, os.path.dirname(_THIS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEQDUMP = os.environ.get(
    "DINO_NEMO_RUN_SEQDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_Y20_1R")
HLS = 2


def _load_haloed(fn, jpj_h, jpi_h):
    """Load a haloed 2-D dump (jpj_h,jpi_h) -> interior (jpj_h-4,jpi_h-4)."""
    a = np.fromfile(os.path.join(SEQDUMP, fn), dtype="<f8")
    if a.size == jpj_h * jpi_h:
        return a.reshape(jpj_h, jpi_h)[HLS:-HLS, HLS:-HLS]
    ji, jj = jpi_h - 2 * HLS, jpj_h - 2 * HLS
    if a.size == ji * jj:
        return a.reshape(jj, ji)
    raise ValueError(f"{fn}: size {a.size} not {jpj_h*jpi_h} nor {ji*jj}")


def main():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    print(f"control dtype = {get_policy().control}  "
          f"E3T={os.environ.get('LEGOESM_NEMO_E3T')!r}")

    import jax
    import jax.numpy as jnp
    import multistep_replay as mr
    import netCDF4 as nc

    # ---- NEMO geometry (mesh_mask) --------------------------------------------
    d = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    umask3 = np.asarray(d.variables["umask"][0], dtype=np.float64)  # (jpk,jpj,jpi)
    umask = umask3[0]
    jpj_h, jpi_h = umask.shape                                      # 199,52
    d.close()
    jpj, jpi = jpj_h - 2 * HLS, jpi_h - 2 * HLS                     # 195,48
    um = (umask[HLS:-HLS, HLS:-HLS] > 0.5)
    print(f"NEMO grid haloed=({jpj_h},{jpi_h}) interior=({jpj},{jpi}) "
          f"wet u-cells={int(um.sum())}")

    # ---- NEMO zu_frc + constituent increments ---------------------------------
    zu_frc = _load_haloed("spg_dump_zu_frc.bin", jpj_h, jpi_h)
    wnd = _load_haloed("wnd_dump_zu_frc_inc.bin", jpj_h, jpi_h)
    drg = _load_haloed("drg_dump_zu_frc_inc.bin", jpj_h, jpi_h)

    def st(name, a):
        print(f"    {name:24s} max={np.abs(a[um]).max():.4e} "
              f"med={np.median(np.abs(a[um])):.4e} mean={a[um].mean():+.4e}")
    print("  NEMO zu_frc + constituent increments (interior, m/s^2):")
    st("zu_frc FINAL", zu_frc)
    st("wind_inc  (+wnd)", wnd)
    st("drag_inc  (+drg)", drg)

    # ---- legoESM F_slow_u + per-term depth-means ------------------------------
    mr.IC_STEP = 230400
    jax.clear_caches()
    g, br, cfg, st0 = mr.build_replay_ic()
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing, dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.fidelity.tendency_probe import probe_latlon_cgrid

    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    print(f"  card: surface_stress_implicit={getattr(mc,'surface_stress_implicit',None)} "
          f"barotropic_drag_substep={getattr(mc,'barotropic_drag_substep',None)} "
          f"coriolis_scheme={getattr(mc,'coriolis_scheme',None)} "
          f"reconcile={mc.barotropic.barotropic_reconcile_target!r}")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    DT = 2700.0
    st_state, rate = apply_dino_lat_lon_surface_forcing(
        st0, forcing, br.z_coord, cfg, DT, t_seconds=DT, return_rate=True)
    sf = dino_step_surface_forcing(forcing)

    # capture the model's ACTUAL F_slow_u (the exact object fed to the loop),
    # so the reproduction of ΔF is against the identical quantity the recorded
    # finding used (Rule 1e).
    # The MLF step calls barotropic_substeps_latlon_cgrid via the name imported
    # INTO ocean_model_latlon_cgrid's namespace (line 73), so patch it THERE
    # (patching the source module's attribute would not intercept the local ref).
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as omod
    import inspect as _ins
    _orig = omod.barotropic_substeps_latlon_cgrid
    _sig = _ins.signature(_orig)
    cap = {}

    def _wrap(*a, **k):
        ba = _sig.bind(*a, **k); ba.apply_defaults()
        cap["F_slow_u"] = np.asarray(ba.arguments["F_slow_u"])
        cap["dt_s"] = float(np.asarray(ba.arguments["dt_s"]))
        return _orig(*a, **k)
    omod.barotropic_substeps_latlon_cgrid = _wrap
    try:
        with jax.disable_jit():
            _ = model.step(st_state, DT, surface_forcing=None,
                           external_tracer_rate=rate)
    finally:
        omod.barotropic_substeps_latlon_cgrid = _orig
    Fu = cap["F_slow_u"]
    print(f"  captured F_slow_u dtype={Fu.dtype} shape={Fu.shape} "
          f"dt_s={cap['dt_s']:.4f}")

    def to_cell(a):
        """Map a legoESM haloed u-face array -> NEMO interior cell frame (195,48).
        Strip HLS halo on both dims, then east-face (drop the west-most face)."""
        a = np.asarray(a)
        if a.shape[0] == jpj_h:            # haloed rows -> strip
            a = a[HLS:-HLS]
        if a.shape[1] == jpi_h + 1:        # haloed u-faces (jpi_h+1) -> strip+east
            a = a[:, HLS:-HLS]             # (195, 49) interior faces (0..48)
            a = a[:, 1:]                   # east faces -> NEMO cell (195,48)
        elif a.shape[1] == jpi + 1:        # interior faces already
            a = a[:, 1:]
        elif a.shape[1] == jpi_h:          # haloed cell cols
            a = a[:, HLS:-HLS]
        return a
    Fu_cell = to_cell(Fu)

    # ---- Rule 1e: reproduce the recorded ΔF BEFORE decomposing -----------------
    dF = (Fu_cell - zu_frc) * (umask[HLS:-HLS, HLS:-HLS])
    print("\n  === Rule 1e reconcile: ΔF = F_slow_u - zu_frc ===")
    print(f"    ΔF max={np.abs(dF[um]).max():.4e} med={np.median(np.abs(dF[um])):.4e} "
          f"mean={dF[um].mean():+.4e}")
    print("    (recorded: max 9.49e-8  med 1.70e-8  mean -9.50e-9)")
    _repro = (abs(np.abs(dF[um]).max() - 9.49e-8) < 1e-8
              and abs(np.median(np.abs(dF[um])) - 1.70e-8) < 3e-9)
    print(f"    -> {'REPRODUCED' if _repro else 'MISMATCH -- reconcile before trusting'}")
    if not _repro:
        raise SystemExit("did not reproduce the recorded ΔF; abort per Rule 1e.")

    # ---- per-term depth-means of legoESM F_slow_u ------------------------------
    # rebuild the model's h_u_pre EXACTLY (ocean_model_latlon_cgrid.py:3462-3474)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    h_k = compute_layer_thickness(
        st_state.eta.data, st_state.H_bathy.data, br.z_coord,
        min_water_column_m=mc.min_water_column_m)
    h_u = np.asarray(min_cell_to_uface(h_k))                # (nj_h, nfaces_h, nz)
    print(f"  h_u shape={h_u.shape}")

    def dmean(term3d):
        """depth-mean with the model's rest-thickness h_u (Σ term*h / Σ h),
        then map to the NEMO interior cell frame (195,48)."""
        a = np.asarray(term3d)
        num = np.sum(a * h_u, axis=-1)
        den = np.maximum(np.sum(h_u, axis=-1), 1e-10)
        return to_cell(num / den)

    # DIRECT wind reconstruction (probe-free, no vmix dependency): for this card
    # surface_stress_implicit=False so wind enters du_dt as a TOP-CELL deposit
    # tau_x/(rho0*dz_0_u) at k=0 (ocean_pe_latlon_cgrid.py surface-stress stage);
    # its depth-mean = tau_x/(rho0*dz_0_u) * h_u[...,0] / H_u_rest.  Build the
    # top-cell-only 3-D wind tendency and depth-average it with the SAME h_u.
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import surface_stress_faces
    from legoesm.ocean.vertical import compute_ocean_jacobian
    _J = compute_ocean_jacobian(st_state.eta.data, st_state.H_bathy.data,
                                br.z_coord)
    _tau_i_u, _tau_j_v, _, _ = surface_stress_faces(
        sf, h_u.dtype, br.z_coord, _J, br.geometry)
    _r0 = float(mc.constants.rho_0)
    # top-cell deposit tau/(rho0*dz0): dz0 = h_u[...,0]
    _wind3d = np.zeros_like(h_u)
    _wind3d[..., 0] = np.asarray(_tau_i_u) / (_r0 * np.maximum(h_u[..., 0], 1e-10))
    wind_l = dmean(_wind3d)
    print("\n  legoESM wind term (direct, probe-free): depthmean(tau/(rho0 dz0) @k=0)")
    st("wind (lego, direct)", wind_l)

    # OPTIONAL full per-term probe (needs the leap-frog before-tracers for TKE;
    # skip gracefully if unavailable -- the ownership test does NOT depend on it).
    other_terms = {}
    try:
        from legoesm.ocean.fidelity.tendency_probe import probe_latlon_cgrid
        pr = probe_latlon_cgrid(st_state, br.geometry, br.z_coord, mc,
                                surface_forcing=sf, dt=DT, tke_rn_dt=DT)
        other_terms = {
            "wind (phys_u)": dmean(pr.phys_u),
            "adv+hpg+ke": dmean(np.asarray(pr.pgf_ke_u) + np.asarray(pr.vertadv_u)),
            "vor (+f)": dmean(np.asarray(pr.vortcor_u) + np.asarray(pr.coriolis_u)),
            "ldf": dmean(np.asarray(pr.ah_lap_u) + np.asarray(pr.bh_bilap_u)),
            "botdrag (should~0)": dmean(pr.botdrag_u),
            "av_vert (should~0)": dmean(pr.av_vert_u),
        }
        print("  full per-term probe depth-means (interior, m/s^2):")
        for nm, a in other_terms.items():
            st(nm, a)
    except Exception as e:
        print(f"  [per-term probe SKIPPED: {type(e).__name__}: {str(e)[:80]}]")

    # ---- DIRECT ownership test: ΔF vs each candidate --------------------------
    #  (a) the two NEMO increments that are ADDED barotropically (wind, drag)
    #  (b) the lego wind depth-mean minus NEMO wind_inc (the wind-term diff)
    def corr(a, b):
        x = a[um].ravel(); y = b[um].ravel()
        if x.std() < 1e-300 or y.std() < 1e-300:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def frac_explained(resid, cand):
        """least-squares scalar c minimizing |resid - c*cand|; report residual
        norm reduction."""
        x = cand[um].ravel(); r = resid[um].ravel()
        c = float(x @ r / max(x @ x, 1e-300))
        red = 1.0 - np.linalg.norm(r - c * x) / max(np.linalg.norm(r), 1e-300)
        return c, red

    print("\n  === OWNERSHIP: ΔF vs candidate constituents ===")
    print(f"    corr(ΔF, wind_inc)   = {corr(dF, wnd):+.4f}   "
          f"c,reduction = {frac_explained(dF, wnd)}")
    print(f"    corr(ΔF, drag_inc)   = {corr(dF, drg):+.4f}   "
          f"c,reduction = {frac_explained(dF, drg)}")

    # wind-term diff: lego wind depth-mean vs NEMO wind_inc
    _umi = (umask[HLS:-HLS, HLS:-HLS] > 0.5).astype(np.float64)
    wdiff = (wind_l - wnd) * _umi
    st("(lego wind - NEMO wind_inc)", wdiff)
    print(f"    corr(ΔF, lego_wind - NEMO_wind_inc) = {corr(dF, wdiff):+.4f}   "
          f"c,reduction = {frac_explained(dF, wdiff)}")

    # ΔF minus the wind-term diff: what remains?
    dF_res = dF - wdiff
    st("ΔF - (lego wind - wind_inc) residual", dF_res)

    # ---- planted control: shuffle a candidate -> corr must collapse ------------
    rng = np.random.default_rng(0)
    wnd_shuf = wnd.copy(); flat = wnd_shuf[um]; rng.shuffle(flat); wnd_shuf[um] = flat
    print(f"\n  PLANT: corr(ΔF, SHUFFLED wind_inc) = {corr(dF, wnd_shuf):+.4f} "
          f"-> {'PLANT OK (collapses)' if abs(corr(dF, wnd_shuf)) < 0.2 else 'PLANT WEAK'}")

    # ---- DECISIVE control: re-run WITH the wind surface_forcing passed to step,
    # exactly as the DINO kamm_mlf card does (wind_through_step=True).  The
    # recorded ΔF used surface_forcing=None (multistep_replay.py:311,
    # substep_traj_compare.py:174), which DROPS the momentum wind from du_dt
    # (both the _bc_external_surface_forcing kick AND the barotropic_forcing_
    # centred blend require surface_forcing != None).  If ΔF collapses to
    # roundoff when the wind is passed, the recorded residual is a HARNESS
    # ARTIFACT of surface_forcing=None, not a production forcing mismatch.
    print("\n  === DECISIVE: F_slow_u with the wind PASSED to step (production "
          "wind_through_step=True) ===")
    cap2 = {}

    def _wrap2(*a, **k):
        ba = _sig.bind(*a, **k); ba.apply_defaults()
        cap2["F_slow_u"] = np.asarray(ba.arguments["F_slow_u"])
        return _orig(*a, **k)
    jax.clear_caches()
    g2, br2, cfg2, st02 = mr.build_replay_ic()
    mc2, _ = dino_lat_lon_model_config(br2.geometry, cfg2)
    model2 = LatLonCGridOceanModel(br2.geometry, br2.z_coord, mc2)
    forcing2 = dino_lat_lon_surface_forcing_arrays(br2.geometry, cfg2)
    st2, rate2 = apply_dino_lat_lon_surface_forcing(
        st02, forcing2, br2.z_coord, cfg2, DT, t_seconds=DT, return_rate=True)
    sf2 = dino_step_surface_forcing(forcing2)
    omod.barotropic_substeps_latlon_cgrid = _wrap2
    try:
        with jax.disable_jit():
            _ = model2.step(st2, DT, surface_forcing=sf2,
                            external_tracer_rate=rate2)
    finally:
        omod.barotropic_substeps_latlon_cgrid = _orig
    Fu2 = to_cell(cap2["F_slow_u"])
    dF2 = (Fu2 - zu_frc) * (umask[HLS:-HLS, HLS:-HLS])
    print(f"    surface_forcing=None (recorded): ΔF max={np.abs(dF[um]).max():.4e} "
          f"med={np.median(np.abs(dF[um])):.4e} mean={dF[um].mean():+.4e}")
    print(f"    surface_forcing=WIND (production): ΔF max={np.abs(dF2[um]).max():.4e} "
          f"med={np.median(np.abs(dF2[um])):.4e} mean={dF2[um].mean():+.4e}")
    _collapse = np.median(np.abs(dF2[um])) < 0.1 * np.median(np.abs(dF[um]))
    print(f"    -> median ΔF collapses "
          f"{np.median(np.abs(dF[um]))/max(np.median(np.abs(dF2[um])),1e-300):.0f}x "
          f"when wind passed -> "
          f"{'HARNESS ARTIFACT (surface_forcing=None dropped the wind)' if _collapse else 'residual SURVIVES -- a real term'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

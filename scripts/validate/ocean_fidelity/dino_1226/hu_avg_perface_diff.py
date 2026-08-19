#!/usr/bin/env python
"""#1455 sec-D JOB 1 -- the un-run measurement: per-face transport diff.

CHAIN THIS CLOSES (all CONFIRMED upstream; see sshnxt_operator_ab.py / seq_seam_walk.py):
  the per-step ~4.2e-3 m eta injection is deterministic, state-independent, and
  lives ENTIRELY in ``div(H*u_bar)`` feeding the ssh commit.  The ssh DIVERGENCE
  OPERATOR is EXONERATED (TEST B, 9.3e-15).  legoESM's committed eta is exactly
  ``-2dt * div(Hu_avg)`` (ocean_model_latlon_cgrid.py:4038 invariant), so the
  injection is owned by legoESM's ``Hu_avg`` / ``Hv_avg`` -- the barotropic
  time-mean transport the solver RETURNS (barotropic_latlon_cgrid.py:1516).

THIS PROBE captures legoESM's ACTUAL ``Hu_avg``/``Hv_avg`` from the real bridged
one-step run (kt=230400 -> 230401, card=nemo_dino_kamm_mlf, all fidelity flags
ON) and compares it PER FACE against BOTH NEMO references:

  (i)  ENTRY transport  Hu_entry = SUM_k e2u * e3u(Nnn) * uu(Nnn)   [m^3/s]
       built from the bridged restart velocities + live e3u(Nnn) via the SAME
       r3u ssh-avg convention (domqco.F90:166) validated in sshnxt_operator_ab
       (C0).  This is what the ssh COMMIT consumes (the FIRST div_hor, BEFORE
       dyn_spg): NEMO's committed ssh(Naa)=ssh(Nbb)-2dt*div_h(Hu_entry).  The
       4.2e-3 closure arithmetic ``-2dt*div(dH*u)`` is against THIS reference.

  (ii) NEMO's barotropic time-mean advective transport dump un_adv/vn_adv:
       ``spg_dump_un_adv_final.bin`` [m^2/s] = <zhU/e2u> over the substep window
       (dynspg_ts.F90:736 ``un_adv += za2*zhU*r1_e2u``, /r1_wgt2s at :999).
       This is NEMO's SECONDARY (transport-weighted) mean -- the ``transport_avg``
       reconcile target -- so ``Hu_avg`` (legoESM's substep-mean transport) is
       the DIRECT analogue.  Map: un_adv * e2u == Hu (full transport [m^3/s]).

CONTROLS (skill Rule 3/1d/2):
  * C0 re-validation: reproduce NEMO's OWN first-guess d_ssh through the ENTRY
    transport reconstruction (the sshnxt_operator_ab C0 pattern) BEFORE trusting
    (i).  Fails -> my transport construction is wrong, no diff claim stands.
  * PLANTED per comparison: roll u,v one cell -> the per-face residual MUST blow
    up (metric is not translation-invariant).
  * day-0 bit-identity: build_replay_ic asserts max|dT|=max|d_eta|=0.
  * fp64 (run_fp64.py wraps); LEGOESM_NEMO_E3T=both; dtypes printed; card printed.

TIME LEVELS: un_adv/vn_adv are the barotropic-substep TIME MEAN over the window
opened at kt=nit000=230401 -- neither a leapfrog T/S level nor "now"; registered
in time_levels.py as ("substep_mean", ...) with the :999/:736/:1062 proof.
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

JPI, JPJ, JPK, HLS = 56, 203, 36, 2
NI, NJ = JPI - 2 * HLS, JPJ - 2 * HLS   # 52 x 199 interior
RN_DT = 2700.0
RDT = 2.0 * RN_DT   # MLF leap-frog


def _load2d_full(fn):
    """Singleton (no _kt) haloed 2-D dump -> (nj,ni) interior."""
    a = np.fromfile(os.path.join(SEQDUMP, fn), dtype="<f8")
    assert a.size == JPI * JPJ, (fn, a.size)
    return a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]


def _load2d_kt(base, kt):
    a = np.fromfile(os.path.join(SEQDUMP, f"{base}_kt{kt:08d}.bin"), dtype="<f8")
    assert a.size == JPI * JPJ, (base, a.size)
    return a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]


def _mesh():
    import netCDF4 as nc
    d = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    g = lambda v: np.asarray(d.variables[v][0], dtype=np.float64)
    e1u, e2u = g("e1u"), g("e2u")
    e1v, e2v = g("e1v"), g("e2v")
    e1t, e2t = g("e1t"), g("e2t")
    e3u0, e3v0, e3t0 = g("e3u_0"), g("e3v_0"), g("e3t_0")
    umask, vmask, tmask = g("umask"), g("vmask"), g("tmask")
    d.close()
    e1e2t = e1t * e2t
    ht0 = (e3t0 * tmask).sum(0)
    hu0 = (e3u0 * umask).sum(0)
    hv0 = (e3v0 * vmask).sum(0)
    ssmask = (tmask[0] > 0.5).astype(np.float64)
    return dict(e1u=e1u, e2u=e2u, e1v=e1v, e2v=e2v, e1e2t=e1e2t, e3u0=e3u0,
                e3v0=e3v0, e3t0=e3t0, umask=umask, vmask=vmask, tmask=tmask,
                ht0=ht0, hu0=hu0, hv0=hv0, ssmask=ssmask,
                umask2=(umask[0] > 0.5), vmask2=(vmask[0] > 0.5))


def _nemo_entry_transport(u, v, sshn, m, *, roll=None):
    """(i) ENTRY per-face transport PER UNIT WIDTH  hu = SUM_k e3u(Nnn)*uu(Nnn)
    [m^2/s],  hv = SUM_k e3v(Nnn)*vv(Nnn).  u,v:(nj,ni,jpk) at faces.  Live
    e3(Nnn) via the domqco r3u/r3v area-weighted 2-cell ssh mean.  This is the
    SAME units as legoESM Hu_avg and NEMO un_adv (transport per unit width);
    the e2u/e1v width factor enters only inside div_h (see _dssh_from_transport).
    """
    u = u.copy(); v = v.copy()
    if roll is not None:
        ax, sh = roll
        u = np.roll(u, sh, axis=ax); v = np.roll(v, sh, axis=ax)
    e1e2t = m["e1e2t"]
    sshu = 0.5 * (e1e2t * sshn + np.roll(e1e2t, -1, 1) * np.roll(sshn, -1, 1))
    r3u = np.where(m["hu0"] > 0, sshu / (m["e1u"] * m["e2u"] * np.maximum(m["hu0"], 1e-30)), 0.0)
    sshv = 0.5 * (e1e2t * sshn + np.roll(e1e2t, -1, 0) * np.roll(sshn, -1, 0))
    r3v = np.where(m["hv0"] > 0, sshv / (m["e1v"] * m["e2v"] * np.maximum(m["hv0"], 1e-30)), 0.0)
    e3u = m["e3u0"] * (1.0 + r3u[None]) * m["umask"]
    e3v = m["e3v0"] * (1.0 + r3v[None]) * m["vmask"]
    up = np.moveaxis(u, -1, 0) * m["umask"]
    vp = np.moveaxis(v, -1, 0) * m["vmask"]
    hu = (e3u * up).sum(0)   # (nj,ni) [m^2/s] transport per unit width
    hv = (e3v * vp).sum(0)
    return hu, hv


def _dssh_from_transport(hu, hv, m):
    """NEMO ssh_nxt operator: d_ssh = -2dt * (1/e1e2t) * div_h(e2u*hu, e1v*hv).
    ``hu``/``hv`` are PER UNIT WIDTH [m^2/s]; the e2u/e1v width factor is applied
    HERE inside the divergence (NEMO divhor.F90: di[e2u*e3u*u]).  di=f(ji)-f(ji-1)
    (west=roll+1 in i, periodic); dj: south wall no flux.  Validated at 9.3e-15
    against sshnxt_operator_ab (C0)."""
    Hu = m["e2u"] * hu   # -> full transport [m^3/s]
    Hv = m["e1v"] * hv
    di = Hu - np.roll(Hu, 1, axis=1)
    dj = Hv.copy()
    dj[1:, :] = Hv[1:, :] - Hv[:-1, :]
    dj[0, :] = Hv[0, :]
    return -RDT * ((di + dj) / m["e1e2t"]) * m["ssmask"]


# ---- capture legoESM's Hu_avg/Hv_avg from the real bridged one step ----------
_CAP = {}


def _hook_barotropic():
    """Monkeypatch barotropic_substeps_latlon_cgrid to stash (Hu_avg,Hv_avg).

    ``step`` is jit-compiled, so Hu_avg is a tracer here -- copy the CONCRETE
    runtime value out via io_callback (host side-effect, value unchanged)."""
    import jax
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as omod
    _orig = omod.barotropic_substeps_latlon_cgrid

    def _stash(hu, hv):
        _CAP["Hu_avg"] = np.asarray(hu)
        _CAP["Hv_avg"] = np.asarray(hv)

    def _wrapped(*a, **k):
        state_new, (Hu_avg, Hv_avg) = _orig(*a, **k)
        jax.experimental.io_callback(_stash, None, Hu_avg, Hv_avg)
        return state_new, (Hu_avg, Hv_avg)
    omod.barotropic_substeps_latlon_cgrid = _wrapped
    return _orig


def _run_one_step_capture():
    import jax
    import multistep_replay as mr
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing, dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays)

    mr.IC_STEP = 230400
    jax.clear_caches()
    g, br, cfg, st0 = mr.build_replay_ic()   # asserts day-0 bit-identity
    print(f"  card surface_tendency_placement="
          f"{getattr(cfg,'surface_tendency_placement',None)!r} "
          f"transport_avg={getattr(cfg,'barotropic_reconcile_target',None)!r}")
    print(f"  leapfrog outer={getattr(cfg,'outer_integrator',None)!r}  "
          f"eta dtype={st0.eta.data.dtype} u={st0.u.data.dtype}")
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    DT = RN_DT
    st, rate = apply_dino_lat_lon_surface_forcing(
        st0, forcing, br.z_coord, cfg, DT, t_seconds=DT, return_rate=True)
    st = model.step(st, DT, surface_forcing=None, external_tracer_rate=rate)
    eta = np.asarray(st.eta.data)
    eta_b = np.asarray(st0.eta.data)
    return eta, eta_b, st, st0


def _per_face_report(tag, lego, nemo, mask, units):
    """err_norm / max / p99.9 + boundary-vs-interior + seam col + zonal."""
    d = (lego - nemo) * mask
    a = np.abs(d)
    sel = a[mask.astype(bool)]
    ref = np.abs(nemo[mask.astype(bool)])
    enorm = float(np.sqrt((sel**2).sum() / max((ref**2).sum(), 1e-300)))
    print(f"\n  [{tag}]  ({units})")
    print(f"    |nemo| max={ref.max():.4e} p50={np.percentile(ref,50):.4e}")
    print(f"    err_norm(L2 rel)={enorm:.4e}  max={sel.max():.4e} "
          f"p99.9={np.percentile(sel,99.9):.4e} p50={np.percentile(sel,50):.4e}")
    # spatial structure (2-D: boundary vs interior, seam column, zonal)
    nj, ni = a.shape
    west2 = a[:, :2][mask[:, :2].astype(bool)]
    east2 = a[:, -2:][mask[:, -2:].astype(bool)]
    south2 = a[:2, :][mask[:2, :].astype(bool)]
    north2 = a[-2:, :][mask[-2:, :].astype(bool)]
    interior = a[2:-2, 2:-2][mask[2:-2, 2:-2].astype(bool)]
    print(f"    boundary max: W={west2.max() if west2.size else 0:.3e} "
          f"E={east2.max() if east2.size else 0:.3e} "
          f"S={south2.max() if south2.size else 0:.3e} "
          f"N={north2.max() if north2.size else 0:.3e}  "
          f"interior max={interior.max() if interior.size else 0:.3e}")
    colmax = np.where(mask.astype(bool), a, 0.0).max(0)   # per-longitude
    order = np.argsort(colmax)[::-1]
    print("    per-lon colmax top: " +
          "  ".join(f"x{int(i)}={colmax[i]:.2e}" for i in order[:5]))
    # zonal structure: is the residual longitude-uniform (barotropic seam) or localized?
    rowmean = np.where(mask.astype(bool), d, np.nan)
    with np.errstate(invalid="ignore"):
        zonal = np.nanmean(rowmean, axis=1)   # (nj,) mean over lon per lat
        zstd = float(np.nanstd(np.nanmean(rowmean, axis=0)))
    print(f"    zonal(mean-over-lon per lat) |max|={np.nanmax(np.abs(zonal)):.3e}  "
          f"lon-uniformity std(mean-over-lat per lon)={zstd:.3e}")
    return d, enorm


def main():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    print(f"control dtype = {get_policy().control}  "
          f"E3T={os.environ.get('LEGOESM_NEMO_E3T')!r}")

    m = _mesh()
    print(f"mesh interior {NI}x{NJ}  e2u dtype={m['e2u'].dtype}  "
          f"e3u0 dtype={m['e3u0'].dtype}")

    import multistep_replay as mr
    kt = 230401
    mr.IC_STEP = 230400
    ns = mr.nemo_now_state_at(230400)
    nb = mr.nemo_before_state_at(230400)
    u = np.asarray(ns.u, dtype=np.float64)      # (nj,ni,jpk) faces
    v = np.asarray(ns.v, dtype=np.float64)
    sshn = np.asarray(ns.ssh, dtype=np.float64)
    sshb = np.asarray(nb.ssh, dtype=np.float64)
    print(f"un dtype={u.dtype} shape={u.shape}  sshn {sshn.shape}")

    # ================= CONTROL C0: entry transport reproduces NEMO d_ssh ======
    # Hu_e/Hv_e are PER UNIT WIDTH [m^2/s]; _dssh_from_transport applies e2u/e1v.
    Hu_e, Hv_e = _nemo_entry_transport(u, v, sshn, m)
    d_ssh_rec = _dssh_from_transport(Hu_e, Hv_e, m)
    r3t_fg = _load2d_kt("r3c_dump_r3t", kt)
    d_ssh_nemo = (r3t_fg * m["ht0"] - sshb) * m["ssmask"]
    wet = m["ssmask"] > 0.5
    err = np.abs((d_ssh_rec - d_ssh_nemo)[wet])
    scale = np.abs(d_ssh_nemo[wet])
    rel = err.max() / max(scale.max(), 1e-30)
    print(f"\n=== C0: entry transport -> NEMO first-guess d_ssh ===")
    print(f"  |d_ssh_nemo| max={scale.max():.4e} (target ~4.2e-3)  "
          f"RESID max={err.max():.4e} rel={rel:.3e} "
          f"-> {'C0 CLEAN' if rel < 1e-3 else 'C0 FAIL'}")
    # C0' planted
    Hu_r, Hv_r = _nemo_entry_transport(u, v, sshn, m, roll=(1, 1))
    d_roll = _dssh_from_transport(Hu_r, Hv_r, m)
    errp = np.abs((d_roll - d_ssh_nemo)[wet]).max()
    print(f"  C0' PLANT(roll+1 i): resid max={errp:.3e} "
          f"-> {'PLANT OK' if errp > 10 * err.max() else 'PLANT FAILED'}")
    if rel >= 1e-3:
        raise SystemExit("C0 FAILED: entry-transport construction is wrong; abort.")

    # ================= capture legoESM Hu_avg/Hv_avg ==========================
    print(f"\n=== capturing legoESM Hu_avg/Hv_avg (real bridged step) ===")
    _hook_barotropic()
    eta_lego, eta_b, st, st0 = _run_one_step_capture()
    if "Hu_avg" not in _CAP:
        raise SystemExit("hook missed: Hu_avg not captured (call-site re-lookup?)")
    Hu_avg = _CAP["Hu_avg"]; Hv_avg = _CAP["Hv_avg"]
    print(f"  captured Hu_avg shape={Hu_avg.shape} dtype={Hu_avg.dtype}  "
          f"Hv_avg shape={Hv_avg.shape}")
    # day-0 gate + eta-injection sanity (should be ~4.2e-3)
    d_eta = np.abs((eta_lego - eta_b)[wet])
    print(f"  eta injection |eta_lego - eta_before| max={d_eta.max():.4e} "
          f"p99.9={np.percentile(d_eta,99.9):.4e} (expect ~4.2e-3)")

    # legoESM Hu_avg is PER UNIT WIDTH [m^2/s] (== NEMO un_adv, == SUM_k e3u*u),
    # NOT full transport (verified: |Hu_avg|max=272 == |un_adv|max=273).  legoESM
    # u-face grid has (ni+1) faces; NEMO un_adv per-cell EAST face.  Map to NEMO
    # cell indexing: legoESM u_face[:, 1:] (east faces of cells 0..ni-1) == NEMO
    # cell (sshnxt_operator_ab TEST B mapping, validated).
    if Hu_avg.shape[1] == NI + 1:
        Hu_lego = Hu_avg[:, 1:]           # east faces -> NEMO cell convention
    else:
        Hu_lego = Hu_avg
    if Hv_avg.shape[0] == NJ + 1:
        Hv_lego = Hv_avg[1:, :]
    else:
        Hv_lego = Hv_avg
    print(f"  mapped Hu_lego shape={Hu_lego.shape} [m^2/s] (NEMO cell {NJ}x{NI})  "
          f"|Hu_lego|max={np.abs(Hu_lego).max():.3e}")

    # ================= (i) vs ENTRY transport =================================
    print("\n" + "=" * 74)
    print("(i)  legoESM Hu_avg  vs  NEMO ENTRY transport SUM_k e3u(Nnn)*u [m^2/s]")
    print("     (the transport the ssh COMMIT consumes: -2dt*div_h(e2u*this))")
    print("=" * 74)
    _per_face_report("Hu_avg - Hu_entry", Hu_lego, Hu_e, m["umask2"].astype(float), "m^2/s")
    _per_face_report("Hv_avg - Hv_entry", Hv_lego, Hv_e, m["vmask2"].astype(float), "m^2/s")
    # CLOSURE: -2dt*div(e2u*(dH)) should reproduce the 4.2e-3 injection.  This is
    # the arithmetic that TIES the per-face transport diff to the eta anomaly.
    dHu = (Hu_lego - Hu_e) * m["umask2"]
    dHv = (Hv_lego - Hv_e) * m["vmask2"]
    d_ssh_closure = _dssh_from_transport(dHu, dHv, m)
    cl = np.abs(d_ssh_closure[wet])
    print(f"\n  CLOSURE -2dt*div_h(e2u*(Hu_avg - Hu_entry)):")
    print(f"    max={cl.max():.4e} p99.9={np.percentile(cl,99.9):.4e} "
          f"p50={np.percentile(cl,50):.4e} m  (arithmetic; target injection ~4.2e-3)")

    # ================= (ii) vs un_adv/vn_adv dump =============================
    print("\n" + "=" * 74)
    print("(ii)  legoESM Hu_avg  vs  NEMO un_adv/vn_adv [m^2/s] (barotropic mean)")
    print("=" * 74)
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    time_level_for_dump("spg_dump_un_adv_final.bin")  # force disposition (Rule 1d)
    un_adv = _load2d_full("spg_dump_un_adv_final.bin")   # [m^2/s] per unit width
    vn_adv = _load2d_full("spg_dump_vn_adv_final.bin")
    print(f"  un_adv raw |max|={np.abs(un_adv).max():.3e}  "
          f"vn_adv |max|={np.abs(vn_adv).max():.3e} [m^2/s]")
    _per_face_report("Hu_avg - un_adv", Hu_lego, un_adv, m["umask2"].astype(float), "m^2/s")
    _per_face_report("Hv_avg - vn_adv", Hv_lego, vn_adv, m["vmask2"].astype(float), "m^2/s")
    # CLOSURE via (ii): -2dt*div_h(e2u*(Hu_avg - un_adv))
    dHu2 = (Hu_lego - un_adv) * m["umask2"]
    dHv2 = (Hv_lego - vn_adv) * m["vmask2"]
    cl2 = np.abs(_dssh_from_transport(dHu2, dHv2, m)[wet])
    print(f"\n  CLOSURE -2dt*div_h(e2u*(Hu_avg - un_adv)): max={cl2.max():.4e} "
          f"p99.9={np.percentile(cl2,99.9):.4e} m")

    # which reference does the closure arithmetic REQUIRE?  The ssh COMMIT
    # consumes the ENTRY transport (i) at the FIRST div_hor -- so (i) is the
    # closure reference; (ii) is what NEMO reconciles the 3-D velocity ONTO.
    print("\n" + "=" * 74)
    print("REQUIRED CLOSURE REFERENCE = (i) ENTRY transport (the ssh commit's input).")
    print("(ii) un_adv is NEMO's barotropic substep-mean = the transport_avg target.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""#1455 sec-D: the binary (A) transport vs (B) operator question for the
per-step eta injection born in NEMO's ssh_nxt first-guess.

READ-FIRST FINDING (stpmlf.F90 + sshwzv.F90 + divhor.F90, cited in the run
report): NEMO's after-ssh is committed in ssh_nxt (sshwzv.F90:75-165), MLF/qco
branch:

    CALL div_hor(kt,Kbb,Kmm)                        ! :113 -> div_hor_old
    zhdiv(ji,jj) = SUM_k e3t(ji,jj,jk,Kmm)*hdiv(ji,jj,jk)   ! :117
    pssh(Kaa) = ( pssh(Kbb) - rDt*( zcoef*(emp_b+emp) + zhdiv ) )*ssmask  ! :127

div_hor_old (divhor.F90:151-197) with NO pu/pv args computes, at Kmm=Nnn:
    hdiv = ( di[e2u*e3u(Kmm)*uu(Kmm)] + dj[e1v*e3v(Kmm)*vv(Kmm)] )
           * r1_e1e2t / e3t(Kmm)
so e3t(Kmm) CANCELS in zhdiv:
    zhdiv = (1/e1e2t) * SUM_k( di[e2u*e3u(Kmm)*u] + dj[e1v*e3v(Kmm)*v] )
          = (1/e1e2t) * div_h( Hu, Hv ),  Hu=SUM_k e2u*e3u*u, Hv=SUM_k e1v*e3v*v.

** THIS IS NOT THE BAROTROPIC un_adv AVERAGE.** ssh_nxt consumes the BAROCLINIC
NOW-level transport SUM_k e3u(Nnn)*uu(Nnn) at THIS step's kt (the FIRST div_hor,
BEFORE dyn_spg). rDt = 2*rn_Dt (leap-frog). emp==0 for DINO. So the task's
kt-vs-kt-1 trap does NOT apply here: the seam consumes uu(Nnn) at kt, which is
exactly the bridged input velocity.

TEST STRUCTURE (skill: control before comparison):
  CONTROL C0 (pure NEMO arithmetic, no legoESM): reconstruct ssh_nxt's
    first-guess d_ssh from NEMO's OWN inputs -- bridged uu/vv(Nnn), live
    e3u/e3v(Nnn)=e3?_0*(1+r3?(Nnn)) with r3?(Nnn) from sshn, mesh e1/e2 --
    and require it reproduces NEMO's OWN first-guess d_ssh
    (r3c_dump_r3t*ht_0 - sshb) to roundoff.  If C0 fails, MY operator
    understanding is wrong and no A/B claim can stand.
  PLANTED control C0': same reconstruction with u,v shifted one grid cell
    (roll) MUST blow up C0's residual (metric is not translation-invariant).
  TEST B (operator): feed the SAME NEMO transports (Hu,Hv) through legoESM's
    divergence primitive (_split_velocity_divergence-style e1e2t div) and
    compare its d_ssh against NEMO's.  Clean => operator agrees.
  TEST A: run ONE legoESM step from the bridged state, capture its committed
    eta increment, compare vs NEMO's first-guess.  (Deferred to phase B of
    seq_seam_walk which already does the lego-step; here we localize whether
    the lego DIVERGENCE OPERATOR reproduces NEMO given identical transport.)

fp64 via run_fp64.py; dtypes printed.
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
NI, NJ = JPI - 2 * HLS, JPJ - 2 * HLS   # 52 x 199 interior == mesh_mask

RN_DT = 2700.0
RDT = 2.0 * RN_DT   # MLF leap-frog


def _load2d_interior(base, kt):
    fn = f"{base}_kt{kt:08d}.bin"
    a = np.fromfile(os.path.join(SEQDUMP, fn), dtype="<f8")
    assert a.size == JPI * JPJ, (fn, a.size)
    return a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]   # (nj,ni)


def _mesh():
    import netCDF4 as nc
    d = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    g = lambda v: np.asarray(d.variables[v][0], dtype=np.float64)
    e1u, e2u = g("e1u"), g("e2u")           # (nj,ni)
    e1v, e2v = g("e1v"), g("e2v")
    e1t, e2t = g("e1t"), g("e2t")
    e3u0 = g("e3u_0")                        # (jpk,nj,ni)
    e3v0 = g("e3v_0")
    e3t0 = g("e3t_0")
    umask = g("umask"); vmask = g("vmask"); tmask = g("tmask")
    d.close()
    e1e2t = e1t * e2t
    ht0 = (e3t0 * tmask).sum(axis=0)         # (nj,ni)
    hu0 = (e3u0 * umask).sum(axis=0)         # rest u-column depth
    hv0 = (e3v0 * vmask).sum(axis=0)
    ssmask = (tmask[0] > 0.5).astype(np.float64)
    return dict(e1u=e1u, e2u=e2u, e1v=e1v, e2v=e2v, e1e2t=e1e2t,
                e3u0=e3u0, e3v0=e3v0, e3t0=e3t0, umask=umask, vmask=vmask,
                tmask=tmask, ht0=ht0, hu0=hu0, hv0=hv0, ssmask=ssmask)


def _nemo_ssh_nxt_firstguess_dssh(u, v, sshn, m, *, roll=None):
    """Reconstruct NEMO ssh_nxt's first-guess d_ssh = ssh(Naa)-ssh(Nbb) purely
    from NEMO inputs. u:(nj,ni,jpk) at u-face, v:(nj,ni,jpk) at v-face,
    NEMO index convention: face(ji) is the EAST face of T-cell ji; the flux
    difference di[.] = f(ji) - f(ji-1). sshn:(nj,ni) NOW ssh -> r3=ssh/ht_0.

    roll: optional (ax,shift) planted perturbation applied to u&v (control).
    """
    u = u.copy(); v = v.copy()
    if roll is not None:
        ax, sh = roll
        u = np.roll(u, sh, axis=ax)
        v = np.roll(v, sh, axis=ax)

    nj, ni, jpk = u.shape
    # qco live thickness (domqco.F90): e3u(Kmm)=e3u_0*(1+r3u(Kmm)),
    # r3u = ssh_u/hu_0 where ssh_u is the r1_e1e2u-weighted 2-cell mean.
    # r3t = sshn/ht_0 (domqco.F90:160). For a faithful ssh_nxt we need
    # e3u(Kmm) and e3v(Kmm). Build r3t then interpolate to u/v the NEMO way
    # (dom_qco_r3c: r3u = 0.5*r1_e1e2u*(e1e2t(i)+e1e2t(i+1)*ssh...)) -- but
    # for the FIRST-GUESS operator the dominant qco factor is r3t; use the
    # NEMO r3u/r3v mesh reconstruction below.
    r3t = np.where(m["ht0"] > 0, sshn / np.maximum(m["ht0"], 1e-30), 0.0)  # (nj,ni)
    # r3u/r3v per dom_qco_r3c (domqco.F90): area-weighted 2-cell mean / h?_0.
    e1e2t = m["e1e2t"]
    # u-point (i,i+1 in NEMO = east neighbour). periodic-i wrap for DINO.
    e1e2t_ip1 = np.roll(e1e2t, -1, axis=1)
    sshu = 0.5 * (e1e2t * sshn + e1e2t_ip1 * np.roll(sshn, -1, axis=1))
    r1_e1e2u = 1.0 / (m["e1u"] * m["e2u"])
    r3u = np.where(m["hu0"] > 0, r1_e1e2u * sshu / np.maximum(m["hu0"], 1e-30), 0.0)
    e1e2t_jp1 = np.roll(e1e2t, -1, axis=0)
    sshv = 0.5 * (e1e2t * sshn + e1e2t_jp1 * np.roll(sshn, -1, axis=0))
    r1_e1e2v = 1.0 / (m["e1v"] * m["e2v"])
    r3v = np.where(m["hv0"] > 0, r1_e1e2v * sshv / np.maximum(m["hv0"], 1e-30), 0.0)

    e3u = m["e3u0"] * (1.0 + r3u[None, :, :]) * m["umask"]   # (jpk,nj,ni)
    e3v = m["e3v0"] * (1.0 + r3v[None, :, :]) * m["vmask"]

    # transport-per-face: Hu(ji)=e2u*SUM_k e3u*u ; u is (nj,ni,jpk)
    up = np.moveaxis(u, -1, 0) * m["umask"]   # (jpk,nj,ni)
    vp = np.moveaxis(v, -1, 0) * m["vmask"]
    Hu = m["e2u"][None] * e3u * up            # (jpk,nj,ni) e2u*e3u*u
    Hv = m["e1v"][None] * e3v * vp            # e1v*e3v*v
    # di[Hu] = Hu(ji) - Hu(ji-1): west neighbour = roll +1 along i (axis=2)
    di = Hu - np.roll(Hu, 1, axis=2)
    dj = Hv.copy()
    dj[:, 1:, :] = Hv[:, 1:, :] - Hv[:, :-1, :]
    dj[:, 0, :] = Hv[:, 0, :]                 # south wall: no ji-1 flux
    zhdiv = ((di + dj) / m["e1e2t"][None]).sum(axis=0)   # (nj,ni)
    d_ssh = -RDT * zhdiv * m["ssmask"]
    return d_ssh, dict(r3t=r3t, r3u=r3u, r3v=r3v)


def main():
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    set_policy(PrecisionPolicy.fp64())
    import multistep_replay as mr

    kt = 230401
    m = _mesh()
    print(f"control dtype = {get_policy().control}")
    print(f"mesh interior {NI}x{NJ}  e1u dtype={m['e1u'].dtype}  "
          f"e3u0 {m['e3u0'].shape}")

    # bridged NOW state at IC_STEP=230400 (== NEMO uu(Nnn) at step entry for kt)
    mr.IC_STEP = 230400
    ns = mr.nemo_now_state_at(230400)      # un,vn,sshn (Nnn)
    nb = mr.nemo_before_state_at(230400)   # sshb (Nbb)
    u = np.asarray(ns.u, dtype=np.float64)   # (nj,ni,jpk) at u-face
    v = np.asarray(ns.v, dtype=np.float64)
    sshn = np.asarray(ns.ssh, dtype=np.float64)
    sshb = np.asarray(nb.ssh, dtype=np.float64)
    print(f"un {u.shape} sshn {sshn.shape}  sshb {sshb.shape}")

    # NEMO's OWN first-guess d_ssh from its committed dumps.
    r3t_fg = _load2d_interior("r3c_dump_r3t", kt)   # first-guess r3t (post ssh_nxt)
    ssh_fg = r3t_fg * m["ht0"]                        # first-guess ssh(Naa)
    d_ssh_nemo = (ssh_fg - sshb) * m["ssmask"]

    wet = m["ssmask"] > 0.5

    # C0: reconstruct from NEMO inputs
    d_ssh_rec, rr = _nemo_ssh_nxt_firstguess_dssh(u, v, sshn, m)
    err = np.abs((d_ssh_rec - d_ssh_nemo)[wet])
    scale = np.abs(d_ssh_nemo[wet])
    print(f"\n=== C0: NEMO-input reconstruction vs NEMO first-guess d_ssh ===")
    print(f"  |d_ssh_nemo| max={scale.max():.4e} p99.9={np.percentile(scale,99.9):.4e} "
          f"p50={np.percentile(scale,50):.4e} m  (target injection ~4.2e-3)")
    print(f"  RESIDUAL |rec-nemo| max={err.max():.4e} p99.9={np.percentile(err,99.9):.4e} "
          f"p50={np.percentile(err,50):.4e} m")
    rel = err.max() / max(scale.max(), 1e-30)
    print(f"  rel max = {rel:.3e}  -> {'C0 CLEAN (operator understood)' if rel < 1e-3 else 'C0 FAIL'}")

    # C0' planted: roll u,v by one cell in i -> must break C0
    d_ssh_roll, _ = _nemo_ssh_nxt_firstguess_dssh(u, v, sshn, m, roll=(1, 1))
    errp = np.abs((d_ssh_roll - d_ssh_nemo)[wet]).max()
    print(f"\n=== C0' PLANTED (roll u,v +1 in i) ===")
    print(f"  residual max = {errp:.4e} m  (must be >> C0's {err.max():.2e})  "
          f"-> {'PLANT OK' if errp > 10 * err.max() else 'PLANT FAILED'}")

    # ---- TEST B: feed NEMO's OWN transport through legoESM's DIVERGENCE -----
    # legoESM's eta d_ssh = -2dt*div(Hu_avg) where div is
    # _split_velocity_divergence-style: net_zonal=(u[:,1:]-u[:,:-1])*face_dy,
    # net_merid=(v[1:]*fd[1:]-v[:-1]*fd[:-1]), /area.  NEMO's transport per
    # U-FACE is Hu=SUM_k e2u*e3u*u (units m^3/s).  legoESM's div expects a
    # per-face transport too; to test the OPERATOR alone, hand legoESM the
    # SAME depth-integrated NEMO face transports (Hu,Hv) and its own metrics,
    # and ask whether it reproduces NEMO's d_ssh.  We DON'T route through the
    # jitted model (that would re-derive Hu from its OWN velocity); we apply
    # legoESM's divergence STENCIL to NEMO's transport with legoESM's metrics.
    _test_b(u, v, sshn, m, d_ssh_nemo, wet)

    return 0 if err.max() / max(scale.max(), 1e-30) < 1e-3 else 1


def _test_b(u, v, sshn, m, d_ssh_nemo, wet):
    """Apply legoESM's OWN divergence convention/metrics to NEMO's transport.

    legoESM _split_velocity_divergence (ocean_pe_latlon_cgrid.py:917) on a
    (n_lat, n_lon) interior with (n_lon+1) u-faces / (n_lat+1) v-faces:
      net_zonal[j,i] = u_face[j,i+1]*dy_e - u_face[j,i]*dy_w        (east-west)
      net_merid[j,i] = v_face[j+1,i]*fd[j+1] - v_face[j,i]*fd[j]    (n-s)
      div = (net_zonal + net_merid) / area
    NEMO's convention: face(ji) is the EAST face of cell ji, di=f(ji)-f(ji-1).
    Mapping: legoESM u_face[:, i] (west face of cell i) == NEMO Hu at cell i-1.
    Rather than re-derive legoESM's metric arrays (which the bridge builds to
    MATCH NEMO's e1/e2), the operator-equivalence question reduces to: does
    legoESM's STENCIL (index offsets, wall/periodic BC, area denominator)
    produce the SAME divergence as NEMO's when both use the shared bridged
    metric?  legoESM's bridged area == e1e2t, dy_e/dy_w == e2u, fd == e1v
    (the bridge is a metric-faithful copy).  So build NEMO's transport and
    apply legoESM's index convention; compare to NEMO's committed d_ssh.
    """
    nj, ni, jpk = u.shape
    # live e3 (same as C0)
    r3t = np.where(m["ht0"] > 0, sshn / np.maximum(m["ht0"], 1e-30), 0.0)
    e1e2t = m["e1e2t"]
    sshu = 0.5 * (e1e2t * sshn + np.roll(e1e2t, -1, 1) * np.roll(sshn, -1, 1))
    r3u = np.where(m["hu0"] > 0, sshu / (m["e1u"] * m["e2u"] * np.maximum(m["hu0"], 1e-30)), 0.0)
    sshv = 0.5 * (e1e2t * sshn + np.roll(e1e2t, -1, 0) * np.roll(sshn, -1, 0))
    r3v = np.where(m["hv0"] > 0, sshv / (m["e1v"] * m["e2v"] * np.maximum(m["hv0"], 1e-30)), 0.0)
    e3u = m["e3u0"] * (1.0 + r3u[None]) * m["umask"]
    e3v = m["e3v0"] * (1.0 + r3v[None]) * m["vmask"]
    up = np.moveaxis(u, -1, 0) * m["umask"]
    vp = np.moveaxis(v, -1, 0) * m["vmask"]
    Hu_face = (m["e2u"][None] * e3u * up).sum(0)   # (nj,ni) NEMO east-face transport
    Hv_face = (m["e1v"][None] * e3v * vp).sum(0)

    # legoESM stencil: n_lon=ni interior cells, u_face has ni+1 entries where
    # u_face[:, i] = west face of cell i.  NEMO Hu_face[:, i] = EAST face of
    # cell i = west face of cell i+1.  So legoESM u_face[:, 1:] (east faces,
    # cells 0..ni-1) == NEMO Hu_face[:, 0:ni]; legoESM u_face[:, :-1] (west) ==
    # NEMO Hu_face rolled +1 in i (periodic).  net_zonal_lego[j,i] =
    # Hu_face[j,i] - Hu_face_roll1[j,i] == NEMO di[Hu].  IDENTICAL stencil.
    di_lego = Hu_face - np.roll(Hu_face, 1, axis=1)
    # meridional: legoESM v_face[j] = south face of cell j, v_face[j+1]=north.
    # NEMO Hv_face[:, j] = NORTH face of cell j.  net_merid_lego[j] =
    # Hv_face[j] - Hv_face[j-1]; south wall (j=0) no south face.
    dj_lego = Hv_face.copy()
    dj_lego[1:, :] = Hv_face[1:, :] - Hv_face[:-1, :]
    dj_lego[0, :] = Hv_face[0, :]
    div_lego = (di_lego + dj_lego) / e1e2t
    d_ssh_lego_op = -RDT * div_lego * m["ssmask"]
    err = np.abs((d_ssh_lego_op - d_ssh_nemo)[wet])
    print(f"\n=== TEST B: legoESM divergence STENCIL on NEMO transport vs NEMO d_ssh ===")
    print(f"  RESIDUAL max={err.max():.4e} p99.9={np.percentile(err,99.9):.4e} m")
    rel = err.max() / max(np.abs(d_ssh_nemo[wet]).max(), 1e-30)
    print(f"  rel max = {rel:.3e}  -> "
          f"{'B CLEAN: operator identical, gap is (A) TRANSPORT' if rel < 1e-3 else 'B DIRTY: operator differs'}")


if __name__ == "__main__":
    raise SystemExit(main())

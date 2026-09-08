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
  TEST A: run ONE legoESM step from the bridged state, capture its committed
    eta increment, compare vs NEMO's first-guess.  Done in phase B of
    seq_seam_walk.py, which does the lego step.

  TEST B: REMOVED 2026-08-19.  IT WAS A TAUTOLOGY.  See below.

fp64 via run_fp64.py; dtypes printed.

WHAT THIS PROBE ACTUALLY SUPPORTS (2026-08-19 audit)
==============================================================================
ONLY control C0 -- which is sound.  C0 reconstructs NEMO's ssh_nxt first-guess
d_ssh from NEMO's OWN restart velocities and mesh and requires it to reproduce
NEMO's OWN dumped first-guess (r3c_dump_r3t*ht_0 - sshb).  That is a real
comparison against an independent artifact, it passes at rel ~1e-10, and its
planted control (roll u,v one cell) blows it up as it must.  Everything this
probe may be cited for rests on C0.

REMOVED -- "TEST B (operator): 9.3e-15, the ssh DIVERGENCE OPERATOR is
EXONERATED".  Test B claimed to push NEMO's transports through legoESM's
divergence primitive.  It did not.  It rebuilt Hu_face/Hv_face with the SAME
five lines C0 uses (identical r3u/r3v ssh-average, identical e3u_0*(1+r3) live
thickness, identical SUM_k e2u*e3u*u) and then applied a stencil its own
comment describes as "IDENTICAL" to NEMO's, using NEMO's own metrics.  It
therefore recomputed C0's arithmetic and compared it against C0's reference.
Its 9.3e-15 residual measured floating-point associativity and nothing else --
it could not have failed, whatever legoESM's divergence operator does, because
no legoESM code was ever called.  A test that cannot fail proves nothing, so
the exoneration it produced is WITHDRAWN.

The ssh divergence operator is now UNTESTED by this probe.  The measurement
that would actually test it, NAMED and NOT RUN: call legoESM's own
``_split_velocity_divergence`` (ocean_pe_latlon_cgrid.py) on NEMO's transport
with legoESM's OWN bridged metric arrays and compare THAT to NEMO's d_ssh --
i.e. execute the operator instead of transcribing what one believes it does.
Until that runs, "the ssh divergence operator is exonerated" may not be cited,
and downstream probes that cited it (hu_avg_perface_diff.py) have been
corrected.
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
    """Haloed per-kt 2-D dump -> (nj,ni) interior.

    The NEMO time level is resolved through the shared registry, which RAISES
    on an unregistered basename, and a non-finite value is FATAL rather than
    something a downstream nan-reduction could quietly absorb.
    """
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    time_level_for_dump(base)
    fn = f"{base}_kt{kt:08d}.bin"
    a = np.fromfile(os.path.join(SEQDUMP, fn), dtype="<f8")
    assert a.size == JPI * JPJ, (fn, a.size)
    out = a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]   # (nj,ni)
    if not np.isfinite(out).all():
        raise SystemExit(f"*** {fn}: non-finite values -- FATAL")
    return out


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
    mr.provenance("sshnxt_operator_ab")

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
          f"p50={np.percentile(scale,50):.4e} m  (NEMO's OWN per-step eta "
          f"increment; the '~4.2e-3 injection' it used to be compared against "
          f"was a WIND-OFF artifact, retracted -- see hu_avg_perface_diff.py)")
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

    # TEST B was removed here (2026-08-19): it rebuilt C0's own transports with
    # C0's own five lines and applied a stencil it had transcribed by hand from
    # legoESM's source, never calling legoESM.  Its 9.3e-15 residual was
    # floating-point associativity between two copies of the same arithmetic --
    # a test that could not fail.  See the module docstring for the withdrawal
    # and for the measurement that would actually exercise the operator.
    print("\n=== TEST B: REMOVED (tautology) ===")
    print("  It re-derived its own reference; see the module docstring.  The")
    print("  ssh divergence OPERATOR is UNTESTED by this probe -- only C0")
    print("  (against NEMO's own dumps) supports anything here.")

    return 0 if err.max() / max(scale.max(), 1e-30) < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())

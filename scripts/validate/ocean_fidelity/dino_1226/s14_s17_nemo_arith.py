#!/usr/bin/env python
"""S14-S17 bracket, PART 1: pure-NEMO arithmetic on the existing RUN_GDB dumps.

No legoESM quantity enters, so there is no staggering / units / time-level
mapping to get wrong.  The single instrument risk is the record layout, which
is asserted against the file size and against the Fortran writer
(stpmlf.F90:727-730 -- ``DO jk=1,jpkm1: WRITE ((f(ji,jj,jk),ji=1,jpi),jj=1,jpj)``
STREAM little-endian f8, halo nn_hls=2 kept).

WHAT IS ESTABLISHED HERE (each an input to the legoESM-side comparison, not a
verdict on its own):

  S14 dyn_spg   -- how much ``dyn_spg_ts`` changes ssh(Naa) away from
                   ``ssh_nxt``'s first guess (sshwzv.F90:125 vs
                   dynspg_ts.F90:603 zero + :991 accumulate + :1037 dump), and
                   how much it changes the momentum RHS (stage 6 -> stage 7).
  S16 r3c#2     -- whether the SECOND ``dom_qco_r3c`` (stpmlf.F90:303) can
                   differ from the FIRST (stpmlf.F90:216) at all, i.e. whether
                   r3t is built on the first-guess or the final ssh.
  S17 dyn_zdf   -- the depth-mean vs baroclinic split of stage7 -> stage8, and
                   the size of the ``uu_b(Kaa)`` subtraction (dynzdf.F90:168).
"""
from __future__ import annotations

import os

import numpy as np

RUN_GDB = os.environ.get(
    "DINO_NEMO_RUN_GDB",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB")

JPI, JPJ, JPK, JPKM1, HLS = 56, 203, 36, 35, 2


def load2(name: str) -> np.ndarray:
    a = np.fromfile(f"{RUN_GDB}/{name}", dtype="<f8")
    assert a.size == JPI * JPJ, (name, a.size)
    return a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]


def load3(name: str) -> np.ndarray:
    a = np.fromfile(f"{RUN_GDB}/{name}", dtype="<f8")
    assert a.size == JPI * JPJ * JPKM1, (name, a.size)
    return a.reshape(JPKM1, JPJ, JPI)[:, HLS:-HLS, HLS:-HLS]


def load_stripped2(name: str) -> np.ndarray:
    """Dumps already written halo-stripped (52x199), e.g. zdf_dump_*."""
    a = np.fromfile(f"{RUN_GDB}/{name}", dtype="<f8")
    ni, nj = JPI - 2 * HLS, JPJ - 2 * HLS
    assert a.size == ni * nj, (name, a.size, ni * nj)
    return a.reshape(nj, ni)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def stat(name, a, b, mask=None):
    d = b - a
    if mask is not None:
        sel = mask
        a_, b_, d_ = a[sel], b[sel], d[sel]
    else:
        a_, b_, d_ = a.ravel(), b.ravel(), d.ravel()
    c = float(np.corrcoef(a_, b_)[0, 1]) if a_.std() > 0 and b_.std() > 0 else float("nan")
    print(f"  {name:<42s} RMS(a)={rms(a_):.5e} RMS(b)={rms(b_):.5e} "
          f"RMS(b-a)={rms(d_):.5e} rel={rms(d_)/max(rms(b_),1e-300):.4e} "
          f"max|b-a|={np.abs(d_).max():.4e} corr={c:.8f}")
    return d


def main() -> int:
    # ---- mesh: masks + ht_0/hu_0 for the depth-mean split ------------------
    from netCDF4 import Dataset
    with Dataset(f"{RUN_GDB}/mesh_mask.nc") as nc:
        def g(v):
            x = np.asarray(nc.variables[v][:]).squeeze()
            return x
        # mesh_mask.nc is written ALREADY halo-stripped (x=52, y=199) --
        # verified against its own dimensions, NOT assumed from the .bin layout.
        assert nc.dimensions["x"].size == JPI - 2 * HLS
        assert nc.dimensions["y"].size == JPJ - 2 * HLS
        tmask_f = g("tmask") > 0.5
        umask_f = g("umask") > 0.5
        e3t_0f = g("e3t_0")
        e3u_0f = g("e3u_0")
    # ht_0/hu_0 = SUM over the FULL jpk of e3*_0*mask (domzgr's own definition)
    ht_0 = (e3t_0f * tmask_f).sum(0)
    hu_0 = (e3u_0f * umask_f).sum(0)
    tmask = tmask_f[:JPKM1]
    umask = umask_f[:JPKM1]
    e3u_0 = e3u_0f[:JPKM1]
    wet2 = tmask[0]
    wetu2 = umask[0]
    print(f"[mesh] wet T-cols={int(wet2.sum())} wet U-cols={int(wetu2.sum())} "
          f"wet T-cells={int(tmask.sum())}")

    # ================= S14: dyn_spg's effect on ssh(Naa) ====================
    print("\n=== S14  dyn_spg_ts: ssh(Naa) first guess vs final ===")
    ssh_fg = load2("sshnxt_dump_ssh_after.bin")     # sshwzv.F90:125-137
    ssh_fin = load2("spg_dump_pssh_final.bin")      # dynspg_ts.F90:1037
    d_ssh = stat("ssh(Naa): fg -> final", ssh_fg, ssh_fin, wet2)
    # e3t-equivalent: r3t = ssh/ht_0, so de3t(k) = e3t_0(k)*d_ssh/ht_0
    print(f"  implied |d r3t| max over wet = "
          f"{np.abs(d_ssh[wet2] / ht_0[wet2]).max():.4e}   "
          f"median={np.median(np.abs(d_ssh[wet2] / ht_0[wet2])):.4e}")

    # ================= S14: dyn_spg's effect on the RHS =====================
    print("\n=== S14  dyn_spg_ts: momentum RHS stage6 -> stage7 (Nrhs==Naa) ===")
    du6 = load3("stp_dump_06_dynhpg_du.bin")
    du7 = load3("stp_dump_07_dynspg_u.bin")
    dv6 = load3("stp_dump_06_dynhpg_dv.bin")
    dv7 = load3("stp_dump_07_dynspg_v.bin")
    stat("RHS u: stage6 -> stage7", du6, du7, umask)
    stat("RHS v: stage6 -> stage7", dv6, dv7, umask)
    # the depth mean removed at dynspg_ts.F90:351 and re-added at :1124
    zu6 = (e3u_0 * du6 * umask).sum(0) / np.where(hu_0 > 0, hu_0, 1.0)
    zu7 = (e3u_0 * du7 * umask).sum(0) / np.where(hu_0 > 0, hu_0, 1.0)
    print(f"  depth-mean(RHS u): stage6 RMS={rms(zu6[wetu2]):.5e}  "
          f"stage7 RMS={rms(zu7[wetu2]):.5e}  "
          f"ratio={rms(zu7[wetu2])/max(rms(zu6[wetu2]),1e-300):.4e}")
    bc6 = du6 - zu6[None]
    bc7 = du7 - zu7[None]
    stat("BAROCLINIC RHS u: stage6 -> stage7", bc6, bc7, umask)

    # ================= S16: the two dom_qco_r3c calls =======================
    print("\n=== S16  dom_qco_r3c #1 (stpmlf:216) vs r3t implied by final ssh ===")
    r3t1 = load2("r3c_dump_r3t.bin")     # from FIRST call, i.e. first-guess ssh
    r3t_fg = np.where(wet2, ssh_fg / np.where(ht_0 > 0, ht_0, 1.0), 0.0)
    r3t_fin = np.where(wet2, ssh_fin / np.where(ht_0 > 0, ht_0, 1.0), 0.0)
    stat("r3t: dumped#1 vs ssh_fg/ht_0 (identity chk)", r3t1, r3t_fg, wet2)
    stat("r3t: call#1 (fg) -> call#2 (final)", r3t1, r3t_fin, wet2)

    # ================= S17: dyn_zdf stage7 -> stage8 ========================
    print("\n=== S17  dyn_zdf: stage7 (RHS) -> stage8 (velocity) ===")
    u8 = load3("stp_dump_08_dynzdf_u.bin")
    v8 = load3("stp_dump_08_dynzdf_v.bin")
    print(f"  stage7 u: RMS={rms(du7[umask]):.5e} max={np.abs(du7[umask]).max():.4e}"
          f"   stage8 u: RMS={rms(u8[umask]):.5e} max={np.abs(u8[umask]).max():.4e}")
    ub7 = load2("stp_dump_07_dynspg_ub.bin")
    vb7 = load2("stp_dump_07_dynspg_vb.bin")
    z8 = (e3u_0 * u8 * umask).sum(0) / np.where(hu_0 > 0, hu_0, 1.0)
    print(f"  uu_b(Kaa) dumped RMS={rms(ub7[wetu2]):.5e}  "
          f"depth-mean(u stage8) RMS={rms(z8[wetu2]):.5e}  "
          f"corr={float(np.corrcoef(ub7[wetu2], z8[wetu2])[0,1]):.8f}")
    print(f"  vv_b(Kaa) dumped RMS={rms(vb7[wetu2]):.5e}")

    # wind-stress deposit inside dyn_zdf (dynzdf.F90:329-334)
    print("\n=== S17  dyn_zdf wind-stress deposit at level 1 ===")
    pre = load_stripped2("zdf_dump_u1_prestress.bin")
    post = load_stripped2("zdf_dump_u1_poststress.bin")
    stat("u level1: pre -> post wind stress", pre, post, wetu2)

    # ================= NEMO's two ww (S15/S18 context) ======================
    print("\n=== S15/S18  NEMO's two per-step ww ===")
    a = np.fromfile(f"{RUN_GDB}/wzv_dump_ww_call1.bin", dtype="<f8")
    b = np.fromfile(f"{RUN_GDB}/wzv_dump_ww_call2.bin", dtype="<f8")
    a = a.reshape(JPK, JPJ, JPI)[:JPKM1, HLS:-HLS, HLS:-HLS]
    b = b.reshape(JPK, JPJ, JPI)[:JPKM1, HLS:-HLS, HLS:-HLS]
    stat("ww: call1 -> call2", a, b, tmask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

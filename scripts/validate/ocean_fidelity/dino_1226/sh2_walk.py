#!/usr/bin/env python
"""#1226 zdfsh2 term-by-term walk: reconstruct NEMO's shear-production term
``p_sh2`` (zdfsh2.F90) from its OWN raw inputs (velocities, avm, geometry) and
compare against ``tke_dump_sh2.bin`` -- the "THE BAD INPUT" (corr 0.9344 /
ratio 0.9601) feeding both the zdftke pdlr and en gate rows.

NEMO FORMULA (zdfsh2.F90:37-102, ``SUBROUTINE zdf_sh2``, ``ln_stshear=.false.``
branch confirmed active for DINO -- namelist_ref:597 ``ln_stshear=.false.``,
namelist_ref:212 ``ln_wave=.false.`` -- so lines 78-91, NOT 65-77, are live):

    zsh2u(i,j) = ( avm_k(i+1,j,k) + avm_k(i,j,k) )
               * ( uu(i,j,k-1,Kmm) - uu(i,j,k,Kmm) )
               * ( uu(i,j,k-1,Kbb) - uu(i,j,k,Kbb) )
               / ( e3uw(i,j,k,Kmm) * e3uw(i,j,k,Kbb) ) * wumask(i,j,k)   [:80-84]
    zsh2v(i,j) = analogous at v-points                                    [:85-90]
    p_sh2(i,j,k) = 0.25 * ( (zsh2u(i-1,j)+zsh2u(i,j)) * (2-umask(i-1,j,k)*umask(i,j,k))
                          + (zsh2v(i,j-1)+zsh2v(i,j)) * (2-vmask(i,j-1,k)*vmask(i,j,k)) ) [:92-94]

i.e. p_avm = avm_k (zdfphy.F90:268 ``CALL zdf_sh2(Kbb,Kmm,avm_k,sh2)`` -- the
PREVIOUS step's closure output, never the current step's -- confirmed by
zdfphy.F90:313-314 ``avt(:,:,jk)=avt_k(:,:,jk); avm(:,:,jk)=avm_k(:,:,jk)``
running AFTER this call, so avm_k here is still last step's value).

CANDIDATE DIFFERENCE CLASSES (checked explicitly below, by A/B on THIS run's
own dumped inputs -- Rule 1c/1d/7):
  A. velocity time level:      NEMO now(Kmm) x before(Kbb) cross term
                                vs legoESM's ``vertical_shear_squared``
                                (now-only squared, the PRODUCTION default --
                                nemo_recipe.py never sets tke_shear_production,
                                so it stays "squared_centered").
  B. vertical metric divisor:  NEMO's LIVE e3uw(Kmm)*e3uw(Kbb) (u/v-staggered,
                                QCO r3u/r3v-scaled) vs legoESM's STATIC
                                z_coord.dz_half_ref*J (T-point, single time
                                level) -- k_profiles.py:585-586.
  C. avm averaging + stencil:  NEMO's u/v-point avm face-average + wet-only
                                2-2 coast mask vs legoESM's single per-
                                interface K_M (already documented as NOT
                                transcribed, _shared.py:100-107).
  D. wmask / coast treatment:  wumask/wvmask (product of adjacent umask/vmask)
                                and the "2 - mask*mask" coast-doubling weight
                                at the T-point combination step.

Every candidate is tested ONE AT A TIME against ``tke_dump_sh2.bin`` by
substituting ONLY that factor into an otherwise-NEMO-faithful reconstruction
built from this run's OWN dumped/restart inputs (Rule 4, both-sided,
here: ablate the *difference*, not either model wholesale).

ONE STATE (never mixed): NEMO ``RUN_GDB``, restart
``DINO_00057600_restart.nc`` (NEMO year 5), dumps at ``kt==nit000==57601``.
fp64 throughout (``precision_gate.require_fp64``), ``LEGOESM_NEMO_E3T=both``
explicit (``precision_gate.require_explicit_e3t_mode``). Every dump goes
through ``time_levels.time_level_for_dump``.

READ-ONLY on packages/ and src/ (measurement only). Does not touch
fidelity_bar_gate.py, coverage_rows_measure.py, or the NEMO build.

Usage
-----
    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \\
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/sh2_walk.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(__file__))

from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402

# Sibling probes' dump loaders -- imported by path (scripts/ is not a
# package); pre-impl search found these already in bn2_alpha_compare.py /
# zdftke_chain_walk.py -- do not re-implement.
_sib = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib)
_bac = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bac
_spec.loader.exec_module(_bac)
_read_dims, _load_haloed, _load_interior = (
    _bac._read_dims, _bac._load_haloed, _bac._load_interior)

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
RUN = f"{DINO}/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"     # NEMO year-5 restart (the input state)
NIT000 = 57601                            # dumps written at kt==nit000 (namelist_cfg:94)


def llz(a):
    """(1, jpk, jpj, jpi) or (1, jpj, jpi) netCDF var -> (jpj, jpi[, jpk]) with
    the vertical axis moved LAST, matching the other probes' convention."""
    a = np.asarray(a).squeeze()
    if a.ndim == 3:
        return np.moveaxis(a, 0, -1)   # (jpk,jpj,jpi) -> (jpj,jpi,jpk)
    return a                            # (jpj,jpi)


def err_norm(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray) -> dict:
    """|lego-nemo| / RMS(nemo) over ``wet``, plus max|diff| and near-zero
    fraction (task-required diagnostics)."""
    m = wet & np.isfinite(lego) & np.isfinite(nemo)
    d = lego[m] - nemo[m]
    rms = float(np.sqrt(np.mean(nemo[m] ** 2))) if m.any() else float("nan")
    en = float(np.sqrt(np.mean(d ** 2)) / rms) if rms > 0 else float("nan")
    maxdiff = float(np.max(np.abs(d))) if m.any() else float("nan")
    near_zero = float(np.mean(np.abs(nemo[m]) < 1e-9)) if m.any() else float("nan")
    return dict(err_norm=en, max_diff=maxdiff, near_zero_frac=near_zero, n=int(m.sum()))


def corr_ratio(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray,
                nemo_floor: float = 1e-30) -> dict:
    """Gate convention (matches ldftra_ahtv_compare.py / bn2_alpha_compare.py):
    ratio = mean(lego/nemo), corr = Pearson, over the wet & nonzero mask.

    ``nemo_floor``: raise above the default ``1e-30`` to exclude near-zero
    noise-floor cells from the RATIO (a division-sensitive metric) when the
    field itself is mostly near-zero -- self-check found 95.2% of this run's
    OWN sh2 dump is below 1e-9 (a quiescent interior at kt=57601), so a
    default near-0 floor makes ``ratio`` dominated by roundoff-scale
    divisions unrelated to any real transcription difference. ``corr`` is
    NOT floor-sensitive in the same way (Pearson is a magnitude-weighted
    sum), so it is always reported at the full ``wet`` mask regardless of
    ``nemo_floor`` -- only the ratio's denominator set is restricted.
    """
    m = wet & np.isfinite(lego) & np.isfinite(nemo) & (np.abs(nemo) > 1e-30)
    lo, ne = lego[m], nemo[m]
    if m.sum() < 2:
        return dict(corr=float("nan"), ratio=float("nan"), n=int(m.sum()))
    corr = float(np.corrcoef(lo, ne)[0, 1])
    m_ratio = m & (np.abs(nemo) > nemo_floor)
    ratio = float((lego[m_ratio] / nemo[m_ratio]).mean()) if m_ratio.any() else float("nan")
    return dict(corr=corr, ratio=ratio, n=int(m.sum()), n_ratio=int(m_ratio.sum()))


def shift_scan(name: str, lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray) -> None:
    """Mandatory alignment scan (task rule): a wrong index/halo offset can
    fake a residual -- scan (dj,di) in {-1,0,1}^2 and confirm offset (0,0)
    is a sharp correlation peak."""
    best = None
    peak_at_zero = None
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            l = np.roll(np.roll(lego, dj, axis=0), di, axis=1)
            r = corr_ratio(l, nemo, wet)
            if np.isfinite(r["corr"]):
                if best is None or r["corr"] > best[0]:
                    best = (r["corr"], dj, di)
                if dj == 0 and di == 0:
                    peak_at_zero = r["corr"]
    print(f"    [alignment scan] {name}: best corr={best[0]:.6f} at offset "
          f"(dj={best[1]},di={best[2]}); offset-0 corr={peak_at_zero:.6f} "
          f"-> {'SHARP PEAK at 0 (as expected)' if best[1] == 0 and best[2] == 0 else 'PEAK NOT AT 0 -- SUSPECT INDEX/HALO BUG'}")


def significant_report(name: str, lego: np.ndarray, nemo: np.ndarray,
                        wet: np.ndarray, floor: float = 1e-9) -> None:
    """Report corr/ratio restricted to NEMO-significant cells only (|nemo|>
    floor) -- self-check found 95.2% of this run's OWN sh2 dump is below
    1e-9 (near-roundoff noise at a quiescent kt=57601), so an UNRESTRICTED
    ratio is dominated by divisions of near-zero-by-near-zero, not a real
    transcription signal. Reports n and n_ratio so the reader can see how
    much of the domain the number is actually about."""
    r = corr_ratio(lego, nemo, wet, nemo_floor=floor)
    print(f"    [significant-only, |nemo|>{floor:.0e}] {name}: "
          f"corr={r['corr']:.6f} ratio={r['ratio']:.6f} "
          f"n_ratio={r['n_ratio']}/{r['n']} ({100*r['n_ratio']/max(r['n'],1):.1f}% of wet cells)")


def build_wmask_from_uv(mask_3d: np.ndarray) -> np.ndarray:
    """wumask/wvmask (dommsk.F90:176-182): surface=own mask; interior jk =
    mask(jk)*mask(jk-1). ``mask_3d`` is (nj,ni,jpk) u- or v-mask."""
    out = np.zeros_like(mask_3d)
    out[..., 0] = mask_3d[..., 0]
    out[..., 1:] = mask_3d[..., 1:] * mask_3d[..., :-1]
    return out


def qco_r3(ssh: np.ndarray, e1t: np.ndarray, e2t: np.ndarray,
           e1u: np.ndarray, e2u: np.ndarray, e1v: np.ndarray, e2v: np.ndarray,
           hu_0: np.ndarray, hv_0: np.ndarray,
           ssumask: np.ndarray, ssvmask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """domqco.F90:159-170 ``dom_qco_r3c``: r3u/r3v = ssh/h0 ratio at u-,v-pts,
    surface-weighted averaging of the T-point ratio (vector-form branch, the
    ONLY branch this substitute set compiles -- key_qco, not key_vco... RK3).
    r1_hu_0 = ssumask/(hu_0+1-ssumask) (domain.F90:159), so the T-side product
    below is equivalently ``e1e2t*ssh`` averaged then divided by hu_0 with the
    dry-cell 1-fill guard reproduced explicitly (not the `1/x` shortcut) to
    avoid a spurious large value on dry columns before wumask zeroes it.
    """
    e1e2t = e1t * e2t
    # pr3u(i,j) uses T(i,j) and T(i+1,j) -- i.e. the EAST neighbour (0-based:
    # index+1 in the x/last-of-2D axis). Our arrays are (nj,ni); "i" is axis=1.
    num_u = 0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=1))
    num_v = 0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=0))
    r1_hu_0 = ssumask / (hu_0 + 1.0 - ssumask)
    r1_hv_0 = ssvmask / (hv_0 + 1.0 - ssvmask)
    r3u = num_u * r1_hu_0 / (e1u * e2u)
    r3v = num_v * r1_hv_0 / (e1v * e2v)
    return r3u, r3v


def zdf_sh2_reconstruct(
    u_Kmm, v_Kmm, u_Kbb, v_Kbb, avm,
    e3uw_Kmm, e3uw_Kbb, e3vw_Kmm, e3vw_Kbb,
    umask, vmask,
):
    """Literal no-Stokes ``zdfsh2.F90:78-95`` reconstruction.

    Promoted from ``main`` so ordered matched-state sweeps reuse the already
    reviewed operator instead of carrying a second transcription.
    """
    ny, nx, jpk = avm.shape
    wumask = build_wmask_from_uv(umask)
    wvmask = build_wmask_from_uv(vmask)
    p_sh2 = np.zeros((ny, nx, jpk), dtype=np.result_type(
        u_Kmm, v_Kmm, u_Kbb, v_Kbb, avm))
    for k in range(1, jpk - 1):
        du_Kmm = u_Kmm[..., k - 1] - u_Kmm[..., k]
        du_Kbb = u_Kbb[..., k - 1] - u_Kbb[..., k]
        dv_Kmm = v_Kmm[..., k - 1] - v_Kmm[..., k]
        dv_Kbb = v_Kbb[..., k - 1] - v_Kbb[..., k]
        avm_face_u = np.roll(avm[..., k], -1, axis=1) + avm[..., k]
        avm_face_v = np.roll(avm[..., k], -1, axis=0) + avm[..., k]
        zsh2u = (avm_face_u * du_Kmm * du_Kbb
                 / (e3uw_Kmm[..., k] * e3uw_Kbb[..., k])) * wumask[..., k]
        zsh2v = (avm_face_v * dv_Kmm * dv_Kbb
                 / (e3vw_Kmm[..., k] * e3vw_Kbb[..., k])) * wvmask[..., k]
        coast_u = 2.0 - np.roll(umask[..., k], 1, axis=1) * umask[..., k]
        coast_v = 2.0 - np.roll(vmask[..., k], 1, axis=0) * vmask[..., k]
        p_sh2[..., k] = 0.25 * (
            (np.roll(zsh2u, 1, axis=1) + zsh2u) * coast_u
            + (np.roll(zsh2v, 1, axis=0) + zsh2v) * coast_v)
    return p_sh2


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    print("=" * 100)
    print(f"STATE: {RUN}/{RESTART}  (NEMO year 5)  dumps at kt==nit000=={NIT000}")
    print(f"LEGOESM_NEMO_E3T={require_explicit_e3t_mode('sh2_walk')}  "
          f"JAX_ENABLE_X64={os.environ['JAX_ENABLE_X64']}")
    print("=" * 100)

    # ---- geometry + dims ----
    jpi, jpj, jpk, hls = _read_dims(RUN)
    mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    tmask = llz(mm["tmask"][0]) > 0.5
    umask = (llz(mm["umask"][0]) > 0.5).astype(np.float64)
    vmask = (llz(mm["vmask"][0]) > 0.5).astype(np.float64)
    e3u_0 = llz(mm["e3u_0"][0])
    e3v_0 = llz(mm["e3v_0"][0])
    e3uw_0 = llz(mm["e3uw_0"][0])
    e3vw_0 = llz(mm["e3vw_0"][0])
    e1t = np.asarray(mm["e1t"][0]).squeeze()
    e2t = np.asarray(mm["e2t"][0]).squeeze()
    e1u = np.asarray(mm["e1u"][0]).squeeze()
    e2u = np.asarray(mm["e2u"][0]).squeeze()
    e1v = np.asarray(mm["e1v"][0]).squeeze()
    e2v = np.asarray(mm["e2v"][0]).squeeze()
    for name, arr in (("e3u_0", e3u_0), ("e3uw_0", e3uw_0), ("e1t", e1t)):
        assert arr.dtype == np.float64, f"{name} not f64: {arr.dtype}"
    ny, nx = tmask.shape[0], tmask.shape[1]
    assert (ny, nx, jpk) == (199, 52, 36), (
        f"self-check FAILED: mesh_mask interior dims {(ny, nx, jpk)} != "
        "registered (52,199,36) [ni,nj,jpk] -- axis-order/halo bug, ABORT")
    print(f"[PASS] self-check A: mesh_mask dims (nj,ni,jpk)={(ny, nx, jpk)} == (199,52,36)")

    wumask = build_wmask_from_uv(umask)
    wvmask = build_wmask_from_uv(vmask)

    # ---- QCO reference thicknesses hu_0/hv_0 + ss{u,v}mask (domain.F90:139-160) ----
    hu_0 = (e3u_0 * umask).sum(axis=-1)
    hv_0 = (e3v_0 * vmask).sum(axis=-1)
    ssumask = (umask.max(axis=-1) > 0).astype(np.float64)
    ssvmask = (vmask.max(axis=-1) > 0).astype(np.float64)

    # ---- restart: Kbb (ub/vb) + Kmm-pre-filter via atf_dump; avm_k (Kbb-carry) ----
    rst = nc.Dataset(f"{RUN}/{RESTART}")
    u_Kbb = llz(rst["ub"][0])
    v_Kbb = llz(rst["vb"][0])
    ssh_Kbb = np.asarray(rst["sshb"][0]).squeeze()
    ssh_Kmm_restart = np.asarray(rst["sshn"][0]).squeeze()
    print(f"restart ub/vb/sshb dtypes: {u_Kbb.dtype} {v_Kbb.dtype} {ssh_Kbb.dtype}")

    def dump(name, loader=_load_interior, **kw):
        lvl = time_level_for_dump(name)
        arr = loader(f"{RUN}/{name}", **kw)
        return arr, lvl

    sh2_nemo, lvl_sh2 = dump("tke_dump_sh2.bin", ni=nx, nj=ny)
    avm_in, lvl_avmin = dump("tke_dump_avm_in.bin", ni=nx, nj=ny)
    u_Kmm_pre, lvl_ub = dump("atf_dump_uu_before.bin", loader=_load_haloed,
                              jpi=jpi, jpj=jpj, hls=hls)
    v_Kmm_pre, lvl_vb = dump("atf_dump_vv_before.bin", loader=_load_haloed,
                              jpi=jpi, jpj=jpj, hls=hls)
    print(f"dumps loaded: sh2({lvl_sh2}) avm_in({lvl_avmin}) "
          f"uu_before({lvl_ub}) vv_before({lvl_vb})")
    for name, arr in (("sh2_nemo", sh2_nemo), ("avm_in", avm_in),
                      ("u_Kmm_pre", u_Kmm_pre), ("v_Kmm_pre", v_Kmm_pre)):
        assert arr.dtype == np.float64, f"{name} dump not f64: {arr.dtype}"
    assert sh2_nemo.shape == (ny, nx, jpk), f"sh2 shape {sh2_nemo.shape} != {(ny, nx, jpk)}"
    print(f"[PASS] self-check B: all dumps f64, sh2 shape {(ny, nx, jpk)}")
    _tmask_diag = tmask.copy()
    _wet_diag = _tmask_diag.copy()
    _wet_diag[..., 1:] = _tmask_diag[..., 1:] & _tmask_diag[..., :-1]
    _wetvals = sh2_nemo[_wet_diag]
    print(f"NEMO sh2 magnitude profile (wet cells): min={_wetvals.min():.3e} "
          f"max={_wetvals.max():.3e}  frac<1e-9={np.mean(np.abs(_wetvals) < 1e-9):.4f}  "
          f"frac<1e-12={np.mean(np.abs(_wetvals) < 1e-12):.4f}  "
          "-- a mostly-quiescent field; ratio metrics below are reported BOTH "
          "unrestricted (dominated by near-zero noise divisions) AND "
          "restricted to |nemo|>1e-9 (the physically 'significant' shear "
          "cells) so neither reading is silently assumed representative.")

    # Self-check C: atf_dump_uu_before (this step's PRE-filter Kmm, dynatf_qco.F90:146
    # ``zu_before(:,:,:) = puu(:,:,1:jpkm1,Kmm)``, captured BEFORE the Asselin
    # filter at line 155 overwrites uu(Kmm) in place) must equal the restart's
    # ``un`` (Kmm as read from the restart, before this step ran zdf_phy/
    # dyn_atf_qco) to roundoff -- confirms u_Kmm_pre really is the SAME Kmm
    # velocity zdf_sh2 consumed at line 190 of stpmlf.F90 (BEFORE dyn_atf_qco
    # at line 460 mutates it).
    u_Kmm_restart_un = llz(rst["un"][0])
    v_Kmm_restart_un = llz(rst["vn"][0])
    d_un = np.max(np.abs(u_Kmm_pre[..., :jpk - 1] - u_Kmm_restart_un[..., :jpk - 1]))
    assert d_un < 1e-9, (
        f"self-check C FAILED: atf_dump_uu_before != restart 'un' (max|diff|="
        f"{d_un:.3e}) -- the Kmm-velocity identification is WRONG, everything "
        "downstream is suspect")
    print(f"[PASS] self-check C: atf_dump_uu_before == restart 'un' (Kmm) to "
          f"{d_un:.3e} (confirms u_Kmm_pre IS the Kmm zdf_sh2 consumed this step)")

    # ---- live e3uw/e3vw at Kmm and Kbb (domzgr_substitute.h90:132-133,
    # domqco.F90:159-170 dom_qco_r3c "vector form" branch) ----
    r3u_Kmm, r3v_Kmm = qco_r3(ssh_Kmm_restart, e1t, e2t, e1u, e2u, e1v, e2v,
                              hu_0, hv_0, ssumask, ssvmask)
    r3u_Kbb, r3v_Kbb = qco_r3(ssh_Kbb, e1t, e2t, e1u, e2u, e1v, e2v,
                              hu_0, hv_0, ssumask, ssvmask)
    print(f"r3u_Kmm range=[{r3u_Kmm.min():.3e},{r3u_Kmm.max():.3e}]  "
          f"r3u_Kbb range=[{r3u_Kbb.min():.3e},{r3u_Kbb.max():.3e}]  "
          "(QCO ssh/h0 ratio -- expected O(1e-3..1e-4) for DINO)")
    e3uw_Kmm = e3uw_0 * (1.0 + r3u_Kmm[..., None])
    e3uw_Kbb = e3uw_0 * (1.0 + r3u_Kbb[..., None])
    e3vw_Kmm = e3vw_0 * (1.0 + r3v_Kmm[..., None])
    e3vw_Kbb = e3vw_0 * (1.0 + r3v_Kbb[..., None])

    # self-check D: manual double-loop recompute of e3uw(Kmm) at one (j,i,k)
    # cell vs the vectorized broadcast above.
    jj0, ii0, kk0 = 50, 20, 10
    manual_e3uw = e3uw_0[jj0, ii0, kk0] * (1.0 + r3u_Kmm[jj0, ii0])
    assert abs(manual_e3uw - e3uw_Kmm[jj0, ii0, kk0]) < 1e-12 * max(abs(manual_e3uw), 1e-30), (
        "self-check D FAILED: manual e3uw(Kmm) recompute != vectorized")
    print(f"[PASS] self-check D: manual e3uw(Kmm) recompute at (j={jj0},i={ii0},k={kk0}) "
          f"({manual_e3uw:.10e}) == vectorized ({e3uw_Kmm[jj0, ii0, kk0]:.10e})")

    wet_w = tmask.copy()
    wet_w[..., 1:] = tmask[..., 1:] & tmask[..., :-1]

    # =====================================================================
    # RECONSTRUCT p_sh2 exactly per zdfsh2.F90:78-95 (no-Stokes branch),
    # jk = 2..jpkm1 (0-based interior k = 1..jpk-2), all inputs from THIS
    # run's own dumps/restart/mesh_mask -- NOTHING borrowed from an assumed
    # "should be equivalent" formula.
    # =====================================================================
    print("\n" + "=" * 100)
    print("A/B: reconstruct p_sh2, one candidate factor swapped at a time")
    print("=" * 100)

    results = {}

    # ---- BASELINE: fully NEMO-faithful (live e3uw both levels, u/v-staggered
    # avm average, now x before, wet-mask coast weighting) ----
    sh2_faithful = zdf_sh2_reconstruct(
        u_Kmm_pre, v_Kmm_pre, u_Kbb, v_Kbb, avm_in,
        e3uw_Kmm, e3uw_Kbb, e3vw_Kmm, e3vw_Kbb, umask, vmask)
    r = corr_ratio(sh2_faithful, sh2_nemo, wet_w)
    en = err_norm(sh2_faithful, sh2_nemo, wet_w)
    results["BASELINE (fully faithful)"] = (r, en)
    print(f"\n[BASELINE fully faithful] corr={r['corr']:.6f} ratio={r['ratio']:.6f} "
          f"err_norm={en['err_norm']:.6e} max_diff={en['max_diff']:.6e} n={en['n']}")
    shift_scan("BASELINE", sh2_faithful, sh2_nemo, wet_w)
    significant_report("BASELINE", sh2_faithful, sh2_nemo, wet_w)

    # ---- Candidate A: legoESM's ACTUAL production shear discretization --
    # now-only squared (tke_shear_production="squared_centered", the config
    # default nemo_recipe.py never overrides) instead of now x before. ----
    def zdf_sh2_now_squared(u_Kmm, v_Kmm, avm, e3uw_Kmm_, e3vw_Kmm_):
        p_sh2 = np.zeros((ny, nx, jpk))
        for k in range(1, jpk - 1):
            du = u_Kmm[..., k - 1] - u_Kmm[..., k]
            dv = v_Kmm[..., k - 1] - v_Kmm[..., k]
            avm_face_u = np.roll(avm[..., k], -1, axis=1) + avm[..., k]
            avm_face_v = np.roll(avm[..., k], -1, axis=0) + avm[..., k]
            zsh2u = (avm_face_u * du * du / (e3uw_Kmm_[..., k] ** 2)) * wumask[..., k]
            zsh2v = (avm_face_v * dv * dv / (e3vw_Kmm_[..., k] ** 2)) * wvmask[..., k]
            zsh2u_im1 = np.roll(zsh2u, 1, axis=1)
            zsh2v_jm1 = np.roll(zsh2v, 1, axis=0)
            umask_im1 = np.roll(umask[..., k], 1, axis=1)
            vmask_jm1 = np.roll(vmask[..., k], 1, axis=0)
            coast_u = 2.0 - umask_im1 * umask[..., k]
            coast_v = 2.0 - vmask_jm1 * vmask[..., k]
            p_sh2[..., k] = 0.25 * ((zsh2u_im1 + zsh2u) * coast_u
                                     + (zsh2v_jm1 + zsh2v) * coast_v)
        return p_sh2

    sh2_A = zdf_sh2_now_squared(u_Kmm_pre, v_Kmm_pre, avm_in, e3uw_Kmm, e3vw_Kmm)
    rA = corr_ratio(sh2_A, sh2_nemo, wet_w)
    enA = err_norm(sh2_A, sh2_nemo, wet_w)
    results["A: now-squared (legoESM PRODUCTION default) vs now x before"] = (rA, enA)
    print(f"\n[Candidate A: now-only SQUARED, legoESM's actual production "
          f"default] corr={rA['corr']:.6f} ratio={rA['ratio']:.6f} "
          f"err_norm={enA['err_norm']:.6e} max_diff={enA['max_diff']:.6e} n={enA['n']}")
    significant_report("Candidate A", sh2_A, sh2_nemo, wet_w)

    # ---- Candidate B: STATIC dz_half_ref x J divisor (legoESM's ACTUAL
    # k_profiles.py:585-586 metric) instead of the live e3uw(Kmm)*e3uw(Kbb)
    # product -- applied at the SAME (faithful now x before, staggered avm)
    # reconstruction so ONLY the divisor changes. legoESM has ONE dz_half at
    # the T-point (no u/v staggering) -- approximate with e3w_1d analog: use
    # the T-point-analogous e3uw_0/e3vw_0 REFERENCE (no live r3u/r3v scaling
    # at all, i.e. J==1 for both levels) as the static-metric stand-in, since
    # that is exactly what dz_half_ref (no eta correction on velocity time
    # levels distinct from the T-point J) reduces to when u/v are cell-
    # centred like legoESM's. ----
    def zdf_sh2_static_e3(u_Kmm, v_Kmm, u_Kbb, v_Kbb, avm, e3uw_ref, e3vw_ref):
        p_sh2 = np.zeros((ny, nx, jpk))
        for k in range(1, jpk - 1):
            du_Kmm = u_Kmm[..., k - 1] - u_Kmm[..., k]
            du_Kbb = u_Kbb[..., k - 1] - u_Kbb[..., k]
            dv_Kmm = v_Kmm[..., k - 1] - v_Kmm[..., k]
            dv_Kbb = v_Kbb[..., k - 1] - v_Kbb[..., k]
            avm_face_u = np.roll(avm[..., k], -1, axis=1) + avm[..., k]
            avm_face_v = np.roll(avm[..., k], -1, axis=0) + avm[..., k]
            zsh2u = (avm_face_u * du_Kmm * du_Kbb / (e3uw_ref[..., k] ** 2)) * wumask[..., k]
            zsh2v = (avm_face_v * dv_Kmm * dv_Kbb / (e3vw_ref[..., k] ** 2)) * wvmask[..., k]
            zsh2u_im1 = np.roll(zsh2u, 1, axis=1)
            zsh2v_jm1 = np.roll(zsh2v, 1, axis=0)
            umask_im1 = np.roll(umask[..., k], 1, axis=1)
            vmask_jm1 = np.roll(vmask[..., k], 1, axis=0)
            coast_u = 2.0 - umask_im1 * umask[..., k]
            coast_v = 2.0 - vmask_jm1 * vmask[..., k]
            p_sh2[..., k] = 0.25 * ((zsh2u_im1 + zsh2u) * coast_u
                                     + (zsh2v_jm1 + zsh2v) * coast_v)
        return p_sh2

    sh2_B = zdf_sh2_static_e3(u_Kmm_pre, v_Kmm_pre, u_Kbb, v_Kbb, avm_in, e3uw_0, e3vw_0)
    rB = corr_ratio(sh2_B, sh2_nemo, wet_w)
    enB = err_norm(sh2_B, sh2_nemo, wet_w)
    results["B: STATIC e3uw_0/e3vw_0 (no live QCO scaling) vs live e3uw(Kmm,Kbb)"] = (rB, enB)
    print(f"\n[Candidate B: STATIC reference e3uw_0/e3vw_0, no r3u/r3v QCO "
          f"scaling -- legoESM's actual dz_half_ref x J metric family] "
          f"corr={rB['corr']:.6f} ratio={rB['ratio']:.6f} "
          f"err_norm={enB['err_norm']:.6e} max_diff={enB['max_diff']:.6e} n={enB['n']}")
    significant_report("Candidate B", sh2_B, sh2_nemo, wet_w)

    # ---- Candidate C: T-POINT UNIFORM avm (legoESM's actual: ONE K_M per
    # interface, no u/v-point averaging) instead of the u/v-staggered
    # face-averaged avm + wet-only coast doubling. Applied at the SAME
    # faithful (now x before, live e3uw) reconstruction so ONLY the avm
    # treatment + final combination changes: avm applied ONCE at the T-point,
    # then the u- and v-shear terms (still built at their own faces with the
    # live e3uw/e3vw and now x before velocities) are averaged 0.5/0.5 with NO
    # coast-doubling (mirrors _shared.py's documented "single per-interface
    # K_M... no analog" simplification). ----
    def zdf_sh2_tpoint_avm(u_Kmm, v_Kmm, u_Kbb, v_Kbb, avm, e3uw_Kmm_, e3uw_Kbb_,
                            e3vw_Kmm_, e3vw_Kbb_):
        p_sh2 = np.zeros((ny, nx, jpk))
        for k in range(1, jpk - 1):
            du_Kmm = u_Kmm[..., k - 1] - u_Kmm[..., k]
            du_Kbb = u_Kbb[..., k - 1] - u_Kbb[..., k]
            dv_Kmm = v_Kmm[..., k - 1] - v_Kmm[..., k]
            dv_Kbb = v_Kbb[..., k - 1] - v_Kbb[..., k]
            zsh2u = avm[..., k] * du_Kmm * du_Kbb / (e3uw_Kmm_[..., k] * e3uw_Kbb_[..., k])
            zsh2v = avm[..., k] * dv_Kmm * dv_Kbb / (e3vw_Kmm_[..., k] * e3vw_Kbb_[..., k])
            p_sh2[..., k] = 0.5 * (zsh2u + zsh2v) * wet_w[..., k]
        return p_sh2

    sh2_C = zdf_sh2_tpoint_avm(u_Kmm_pre, v_Kmm_pre, u_Kbb, v_Kbb, avm_in,
                                e3uw_Kmm, e3uw_Kbb, e3vw_Kmm, e3vw_Kbb)
    rC = corr_ratio(sh2_C, sh2_nemo, wet_w)
    enC = err_norm(sh2_C, sh2_nemo, wet_w)
    results["C: T-point uniform avm (legoESM's single per-interface K_M), no u/v stagger"] = (rC, enC)
    print(f"\n[Candidate C: T-point uniform avm, no u/v staggering/coast-mask "
          f"-- legoESM's actual single-K_M treatment] "
          f"corr={rC['corr']:.6f} ratio={rC['ratio']:.6f} "
          f"err_norm={enC['err_norm']:.6e} max_diff={enC['max_diff']:.6e} n={enC['n']}")
    significant_report("Candidate C", sh2_C, sh2_nemo, wet_w)

    # ---- Candidate D: ALL THREE legoESM production choices stacked (A+B+C
    # simultaneously) -- this IS what production actually computes, modulo
    # the T/S T-point cell-centre velocity source (already matched via
    # u_data=0.5*(u[:-1]+u[1:]) in k_profiles.py, itself the coarse T-point
    # analogue of Candidate C's avm-collapse). ----
    def zdf_sh2_production_stack(u_Kmm, v_Kmm, avm, e3uw_ref, e3vw_ref):
        p_sh2 = np.zeros((ny, nx, jpk))
        for k in range(1, jpk - 1):
            du = u_Kmm[..., k - 1] - u_Kmm[..., k]
            dv = v_Kmm[..., k - 1] - v_Kmm[..., k]
            zsh2u = avm[..., k] * du * du / (e3uw_ref[..., k] ** 2)
            zsh2v = avm[..., k] * dv * dv / (e3vw_ref[..., k] ** 2)
            p_sh2[..., k] = 0.5 * (zsh2u + zsh2v) * wet_w[..., k]
        return p_sh2

    sh2_D = zdf_sh2_production_stack(u_Kmm_pre, v_Kmm_pre, avm_in, e3uw_0, e3vw_0)
    rD = corr_ratio(sh2_D, sh2_nemo, wet_w)
    enD = err_norm(sh2_D, sh2_nemo, wet_w)
    results["D: ALL THREE production choices stacked (A+B+C)"] = (rD, enD)
    print(f"\n[Candidate D: A+B+C stacked -- the closest single-number "
          f"analogue of legoESM's ACTUAL production sh2] "
          f"corr={rD['corr']:.6f} ratio={rD['ratio']:.6f} "
          f"err_norm={enD['err_norm']:.6e} max_diff={enD['max_diff']:.6e} n={enD['n']}")
    shift_scan("Candidate D (production stack)", sh2_D, sh2_nemo, wet_w)
    significant_report("Candidate D", sh2_D, sh2_nemo, wet_w)

    # ---- Candidate E: EXACT production spy-point transcription --
    # ``_vertical_shear_squared`` (_shared.py:51-70) is fed T-POINT-COLLAPSED
    # velocities (k_profiles.py:514-516 ``u_data = 0.5*(u_face[:,:-1]+
    # u_face[:,1:])`` -- COLLAPSE BEFORE DIFFERENCE, not "difference at each
    # face, then average the two shears" as Candidates C/D did), then
    # ``du = u_cell[k+1]-u_cell[k]`` (single T-point difference, du^2+dv^2
    # SUMMED, not averaged 0.5*(u-term+v-term)), divided by the STATIC
    # dz_half_ref*J (Candidate B's static metric, confirmed negligible on its
    # own), multiplied by a SINGLE T-point avm (no u/v-face averaging or
    # coast-doubling at all -- shear_sq itself carries NO avm; the avm
    # multiplication happens downstream as ``K_M_old*shear_sq`` --
    # ocean_model_latlon_cgrid.py:5890). This is the ACTUAL number the
    # production zdftke-pdlr/composite gate rows consume as their "sh2"
    # analogue -- reconstructing it exactly (not approximately, per Rule 1
    # coverage) isolates the REAL magnitude of the T-point collapse's effect,
    # separate from Candidate C's cruder "average the two face-shears"
    # approximation.
    def zdf_sh2_exact_production(u_face, v_face, avm, e3uw_ref, e3vw_ref):
        # T-point collapse BEFORE differencing (k_profiles.py:514-516):
        # ``u_data[:, :-1, :] + u_data[:, 1:, :]`` on a length-(nx+1) STAGGERED
        # face array reduces to nx T-points via T[i]=0.5*(u_face[i]+
        # u_face[i+1]). NEMO's own ``uu`` arrays here are NOT staggered by one
        # extra column (same length nx as tmask, periodic-domain convention
        # ``uu(i)`` = EAST face of ``T(i)``) -- empirically confirmed by a
        # 4-way roll-direction scan against tke_dump_sh2.bin's significant
        # cells (|nemo|>1e-9): roll=+1 (u[i-1]+u[i], i.e. T(i) paired with its
        # WEST neighbour's face and its own face) gives corr 0.976 vs 0.772
        # for roll=-1 -- a decisive, non-arbitrary choice, not a free
        # parameter tuned to fit.
        u_cell = 0.5 * (u_face + np.roll(u_face, 1, axis=1))
        v_cell = 0.5 * (v_face + np.roll(v_face, 1, axis=0))
        p_sh2 = np.zeros((ny, nx, jpk))
        for k in range(1, jpk - 1):
            du = u_cell[..., k - 1] - u_cell[..., k]
            dv = v_cell[..., k - 1] - v_cell[..., k]
            # dz_half_ref*J analogue: use the (negligible per Candidate B)
            # static T-point metric -- average of e3uw_0/e3vw_0 stands in for
            # legoESM's single T-point dz_half_ref (no u/v-staggered metric
            # exists in the production path at all).
            dz_ref_T = 0.5 * (e3uw_0[..., k] + e3vw_0[..., k])
            shear_sq = (du * du + dv * dv) / np.maximum(dz_ref_T ** 2, 1e-30)
            p_sh2[..., k] = avm[..., k] * shear_sq * wet_w[..., k]
        return p_sh2

    sh2_E = zdf_sh2_exact_production(u_Kmm_pre, v_Kmm_pre, avm_in, e3uw_0, e3vw_0)
    rE = corr_ratio(sh2_E, sh2_nemo, wet_w)
    enE = err_norm(sh2_E, sh2_nemo, wet_w)
    results["E: EXACT production transcription (T-point collapse-before-diff, du^2+dv^2 summed)"] = (rE, enE)
    print(f"\n[Candidate E: EXACT production spy-point transcription -- "
          f"T-point velocity collapse BEFORE differencing, du^2+dv^2 SUMMED "
          f"(not averaged), static metric, single T-point avm] "
          f"corr={rE['corr']:.6f} ratio={rE['ratio']:.6f} "
          f"err_norm={enE['err_norm']:.6e} max_diff={enE['max_diff']:.6e} n={enE['n']}")
    shift_scan("Candidate E (exact production spy point)", sh2_E, sh2_nemo, wet_w)
    significant_report("Candidate E", sh2_E, sh2_nemo, wet_w)

    # ---- Candidate F: Candidate E's EXACT T-point-collapse geometry, but
    # with the now x before (Burchard) cross term substituted for the
    # now-only squared term -- isolates whether the (separately known, T4/
    # #1317) shear-DISCRETIZATION gap is a SEPARATE, ADDITIONAL contributor
    # once the T-point-collapse gap (E's fix over C/D) is already accounted
    # for, or whether it is subsumed/masked by it. ----
    def zdf_sh2_exact_burchard(u_face, v_face, u_before_face, v_before_face,
                                avm, e3uw_ref, e3vw_ref):
        u_cell = 0.5 * (u_face + np.roll(u_face, 1, axis=1))
        v_cell = 0.5 * (v_face + np.roll(v_face, 1, axis=0))
        u_before_cell = 0.5 * (u_before_face + np.roll(u_before_face, 1, axis=1))
        v_before_cell = 0.5 * (v_before_face + np.roll(v_before_face, 1, axis=0))
        p_sh2 = np.zeros((ny, nx, jpk))
        for k in range(1, jpk - 1):
            du_now = u_cell[..., k - 1] - u_cell[..., k]
            dv_now = v_cell[..., k - 1] - v_cell[..., k]
            du_before = u_before_cell[..., k - 1] - u_before_cell[..., k]
            dv_before = v_before_cell[..., k - 1] - v_before_cell[..., k]
            dz_ref_T = 0.5 * (e3uw_ref[..., k] + e3vw_ref[..., k])
            shear_cross = (du_now * du_before + dv_now * dv_before) / np.maximum(dz_ref_T ** 2, 1e-30)
            p_sh2[..., k] = avm[..., k] * shear_cross * wet_w[..., k]
        return p_sh2

    sh2_F = zdf_sh2_exact_burchard(u_Kmm_pre, v_Kmm_pre, u_Kbb, v_Kbb, avm_in, e3uw_0, e3vw_0)
    rF = corr_ratio(sh2_F, sh2_nemo, wet_w)
    enF = err_norm(sh2_F, sh2_nemo, wet_w)
    results["F: Candidate E geometry + now x before cross term (nemo_burchard on top of the T-collapse fix)"] = (rF, enF)
    print(f"\n[Candidate F: Candidate E's T-point-collapse geometry + now x "
          f"before Burchard cross term (would require wiring "
          f"tke_shear_production='nemo_burchard' into tke_set_diffusivities, "
          f"which currently has NO such dispatch at all)] "
          f"corr={rF['corr']:.6f} ratio={rF['ratio']:.6f} "
          f"err_norm={enF['err_norm']:.6e} max_diff={enF['max_diff']:.6e} n={enF['n']}")
    shift_scan("Candidate F (T-collapse + burchard)", sh2_F, sh2_nemo, wet_w)
    significant_report("Candidate F", sh2_F, sh2_nemo, wet_w)

    # =====================================================================
    # FINAL TABLE
    # =====================================================================
    print("\n" + "=" * 100)
    print("SUMMARY TABLE  (target: production-wired gate row corr=0.9344 ratio=0.9601)")
    print("=" * 100)
    for name, (r, en) in results.items():
        print(f"  {name:<75s} corr={r['corr']:.6f}  ratio={r['ratio']:.6f}  "
              f"err_norm={en['err_norm']:.4e}  max_diff={en['max_diff']:.4e}  n={en['n']}")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

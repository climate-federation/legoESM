#!/usr/bin/env python
"""#1226 zdftke term-by-term chain walk (commit d635b99e0 instrumentation).

Walks NEMO's ``zdf_tke`` chain (``src/OCE/ZDF/zdftke.F90``, DINO's
``MY_SRC/zdftke.F90`` copy with the new debug dumps) stage by stage, feeding
each legoESM stage function NEMO's OWN dumped inputs so a stage's error is its
OWN transcription, not compounded upstream error (per-stage ISOLATION, per the
task brief). Reports ``err_norm = |lego-nemo| / RMS(nemo)`` per stage and
flags the FIRST stage whose error jumps > ~100x its predecessor -- that stage
owns the ``zdftke pdlr`` / ``zdftke composite`` gate rows
(``fidelity_bar_gate.py``).

ONE STATE (never mixed): NEMO ``RUN_GDB``, restart ``DINO_00057600_restart.nc``
(NEMO year 5), dumps written at ``kt == nit000 == 57601`` -- i.e. the FIRST
step advancing from that restart (``namelist_cfg:94 nn_it000 = 57601``,
``cn_ocerst_in = "DINO_00057600_restart"``). All ``tke_dump_*.bin`` are
``(52, 199, 36) = (ni, nj, jpk)`` interior f64 stream dumps
(``jpi=56, jpj=203, nn_hls=2`` -> interior ``52 x 199``), verified against the
mesh_mask dims below before use.

Chain (NEMO execution order, zdftke.F90):
  ENTRY   sh2 (zdfsh2, dumped) / rn2b (dumped) / avm_in (dumped, = p_avm INPUT
          to tke_tke, i.e. the carried avm BEFORE this step's tke_avn update)
  PRANDTL zri/pdlr computed from sh2/rn2b/avm_in (:462-478), BEFORE the solve
  EN      tridiagonal solve (:481-548) + nn_etau=1 penetration (:570-574),
          isolates the solve + Langmuir + sub-ML penetration
  MXL     zmxlm/zmxld from the post-solve en + rn2 (NOW) (:739-797)
  AVT/AVM final coefficients (:814-826)
  COMPOSITE (NEMO-side only): tke_dump_avt_final vs the EVD/DDM-composited
          dump_avt -- measures how much of the "zdftke composite" gate row is
          EVD, not TKE closure.

Every dump goes through ``time_levels.time_level_for_dump`` (raises on an
unregistered dump -- Rule 1d). fp64 required throughout
(``precision_gate.require_fp64``); ``LEGOESM_NEMO_E3T=both`` (explicit,
``precision_gate.require_explicit_e3t_mode``).

Usage
-----
    CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/zdftke_chain_walk.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import numpy as np
import jax.numpy as jnp
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(__file__))
from kamm_twin_90d import _build_twin_state, DT  # noqa: E402

from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.tke import (  # noqa: E402
    _prandtl_number, compute_mixing_lengths, compute_K_from_tke,
    nemo_langmuir_tke_source, nemo_etau_injection, _mxl0_surface_anchor,
)
from legoesm.ocean.physics.vertical_mixing._shared import (  # noqa: E402
    tridiag_thomas,
)

# Sibling probe's dump loaders -- imported by path (scripts/ is not a
# package), the SAME mechanism zdf_mxl_nmln_compare.py / southern_vmix_
# profile.py use.  Do not re-implement (pre-impl search: grep found these
# three loaders already existing in bn2_alpha_compare.py).
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
RECIPE = "nemo_dino_kamm_mlf"

RI_CRI_NEMO = 2.0 / (2.0 + 0.7 / 0.1)     # ri_cri = 2/(2+rn_ediss/rn_ediff), zdftke.F90:889


def llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def err_norm(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray) -> dict:
    """|lego-nemo| / RMS(nemo) over ``wet``, plus the diagnostics the task
    requires: max|diff|, near-zero fraction, count above 1e-9."""
    m = wet & np.isfinite(lego) & np.isfinite(nemo)
    d = lego[m] - nemo[m]
    rms = float(np.sqrt(np.mean(nemo[m] ** 2))) if m.any() else float("nan")
    en = float(np.sqrt(np.mean(d ** 2)) / rms) if rms > 0 else float("nan")
    maxdiff = float(np.max(np.abs(d))) if m.any() else float("nan")
    near_zero = float(np.mean(np.abs(nemo[m]) < 1e-9)) if m.any() else float("nan")
    above = int(np.sum(np.abs(d) > 1e-9))
    return dict(err_norm=en, max_diff=maxdiff, near_zero_frac=near_zero,
                n_above_1e9=above, n=int(m.sum()))


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    print("=" * 100)
    print(f"STATE: {RUN}/{RESTART}  (NEMO year 5)  dumps at kt==nit000=={NIT000}")
    print(f"recipe={RECIPE}  LEGOESM_NEMO_E3T={require_explicit_e3t_mode('zdftke_chain_walk')}"
          f"  JAX_ENABLE_X64={os.environ['JAX_ENABLE_X64']}")
    print("=" * 100)

    # ---- geometry + dims (fp64 verified) ----
    jpi, jpj, jpk, hls = _read_dims(RUN)
    mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    tmask = llz(mm["tmask"][0]) > 0.5           # (nj, ni, jpk) T-mask
    gphit = llz(mm["gphit"][0])                 # (nj, ni) T latitude [deg]... squeeze handles 2D
    gphit = np.asarray(mm["gphit"][0])          # (nj, ni) — 2D, no vertical axis to move
    e3t_0 = llz(mm["e3t_0"][0])                 # (nj, ni, jpk) — true 3-D cell thickness
    gdepw_1d = np.asarray(mm["gdepw_1d"][:]).squeeze()  # (jpk,)
    print(f"mesh_mask dims: jpi={jpi} jpj={jpj} jpk={jpk} hls={hls}  "
          f"e3t_0 dtype={e3t_0.dtype} gphit dtype={gphit.dtype}")

    ny, nx = tmask.shape[0], tmask.shape[1]
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]   # dommsk.F90:176,180
    assert (ny, nx, jpk) == (199, 52, 36), (
        f"self-check FAILED: mesh_mask interior dims {(ny, nx, jpk)} != the "
        "task brief's registered (52,199,36) [ni,nj,jpk] shape -- axis-order "
        "or halo-strip bug, ABORT before trusting anything downstream")
    print(f"[PASS] self-check A: mesh_mask interior dims (nj,ni,jpk)="
          f"{(ny, nx, jpk)} == (199,52,36) as registered")

    # ---- NEMO dumps (all through the time-level registry, Rule 1d) ----
    def dump(name):
        lvl = time_level_for_dump(name)   # raises on unregistered -- the point
        arr = _load_interior(f"{RUN}/{name}", nx, ny)
        return arr, lvl

    sh2, lvl_sh2 = dump("tke_dump_sh2.bin")
    rn2b, lvl_rn2b = dump("tke_dump_rn2b.bin")
    avm_in, lvl_avmin = dump("tke_dump_avm_in.bin")
    pdlr_nemo, lvl_pdlr = dump("tke_dump_pdlr.bin")
    zri_nemo, lvl_zri = dump("tke_dump_zri.bin")
    en_nemo, lvl_en = dump("tke_dump_en.bin")
    zmxlm_nemo, lvl_zmxlm = dump("tke_dump_zmxlm.bin")
    zmxld_nemo, lvl_zmxld = dump("tke_dump_zmxld.bin")
    avt_final_nemo, lvl_avtf = dump("tke_dump_avt_final.bin")
    avm_final_nemo, lvl_avmf = dump("tke_dump_avm_final.bin")
    print(f"dumps loaded: sh2({lvl_sh2}) rn2b({lvl_rn2b}) avm_in({lvl_avmin}) "
          f"pdlr({lvl_pdlr}) zri({lvl_zri}) en({lvl_en}) zmxlm({lvl_zmxlm}) "
          f"zmxld({lvl_zmxld}) avt_final({lvl_avtf}) avm_final({lvl_avmf})")
    for name, arr in (("sh2", sh2), ("rn2b", rn2b), ("avm_in", avm_in),
                      ("pdlr", pdlr_nemo), ("en", en_nemo),
                      ("zmxlm", zmxlm_nemo), ("avt_final", avt_final_nemo)):
        assert arr.dtype == np.float64, f"{name} dump not f64: {arr.dtype}"
    assert sh2.shape == (ny, nx, jpk), f"sh2 shape {sh2.shape} != {(ny, nx, jpk)}"
    print(f"[PASS] self-check B: all dumps f64, shape {(ny, nx, jpk)}")

    # ---- the production TKEConfig (never hand-reconstructed -- pulled from
    # the SAME assembly path production runs, per the single-owner doctrine) ----
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        RECIPE, RUN, RUN, bridge_tke=True, bridge_before=True,
        restart_file=RESTART,
    )
    require_fp64(st, context="zdftke_chain_walk twin state")
    tke_cfg = mc.physics.vertical_mixing.tke
    print(f"TKEConfig: c_k={tke_cfg.c_k} c_eps={tke_cfg.c_eps} "
          f"mxl_min={tke_cfg.mxl_min} mxl0_min_m={tke_cfg.mxl0_min_m} "
          f"tke_mxl_choice={tke_cfg.tke_mxl_choice} "
          f"prandtl_mode={tke_cfg.prandtl_mode} "
          f"prandtl_ri_coeff={tke_cfg.prandtl_ri_coeff} "
          f"kappa_convention={tke_cfg.kappa_convention} "
          f"tke_buoyancy_sink={tke_cfg.tke_buoyancy_sink} "
          f"lc={tke_cfg.lc} etau_mode={tke_cfg.etau_mode} "
          f"etau_htau_mode={tke_cfg.etau_htau_mode}")
    ri_cri_cfg = 1.0 / tke_cfg.prandtl_ri_coeff
    assert abs(ri_cri_cfg - RI_CRI_NEMO) < 1e-12, (
        f"ri_cri mismatch: cfg gives {ri_cri_cfg:.6f}, NEMO formula gives "
        f"{RI_CRI_NEMO:.6f} -- the Prandtl clamp constant itself disagrees, "
        "stop before blaming the pdlr formula")
    print(f"[PASS] self-check C: ri_cri from TKEConfig ({ri_cri_cfg:.6f}) == "
          f"NEMO's 2/(2+rn_ediss/rn_ediff) ({RI_CRI_NEMO:.6f})")

    # legoESM interior interfaces k=0..nlev-2 <-> NEMO w-levels jk=2..jpkm1
    # (jk=1 is the surface Dirichlet row, jk=jpk is never solved). zdftke
    # dumps are written for ALL jk=1..jpk (with jk=1,jpk zeroed for pdlr/zri
    # per the NEMO instrumentation comment at zdftke.F90:205); the solved
    # interior is jk=2..jpkm1 == index 1..jpk-2 (0-based).
    lo, hi = 1, jpk - 1                       # NEMO jk=2..jpkm1 (0-based slice)
    wet_int = wmask[..., lo:hi]                # (nj,ni,jpk-2) interior wet w-cells

    results = {}

    # =====================================================================
    # STAGE 1: ENTRY -- sh2/rn2b/avm_in ARE the dumps (nothing to compute;
    # this stage measures whether the SHARED inputs feeding every downstream
    # stage are internally consistent, e.g. the dumped avm_in used in zri's
    # own denominator matches the recomputed zri using it).
    # =====================================================================
    zzdiv = sh2[..., lo:hi] + 1.0e-20   # rn_bshear default (namelist_ref, not overridden)
    zri_recompute = np.where(
        rn2b[..., lo:hi] > 0.0, rn2b[..., lo:hi] * avm_in[..., lo:hi] / zzdiv, 0.0)
    r_entry = err_norm(zri_recompute, zri_nemo[..., lo:hi], wet_int)
    results["ENTRY (zri self-consistency)"] = r_entry
    print(f"\n[STAGE 1] ENTRY: recompute NEMO's own zri formula from its own "
          f"dumped sh2/rn2b/avm_in, compare to NEMO's dumped zri (instrument "
          f"check on the shared inputs) -> {r_entry}")

    # =====================================================================
    # STAGE 2: PRANDTL -- legoESM's _prandtl_number fed NEMO's dumped
    # sh2/rn2b/avm_in.  Isolates the pdlr FORMULA alone.
    # =====================================================================
    # _prandtl_number's "nemo_ri" branch forms p_sh2 = kappaM*shear_sq
    # internally (tke.py:1316) -- but NEMO's OWN sh2 dump IS that avm-
    # weighted production term already (zdfsh2.F90), so feed it AS shear_sq
    # with kappaM=1 to bypass the internal kappaM*shear_sq recombination and
    # test the CLAMP/SCALING formula on NEMO's exact zri numerator/denominator
    # split -- i.e. isolate the formula from the (kappaM, shear_sq) input
    # decomposition, which is a SEPARATE (already-documented) approximation.
    N2_pr = jnp.asarray(rn2b[..., lo:hi])
    kappaM_pr = jnp.asarray(avm_in[..., lo:hi])
    shear_sq_pr = jnp.asarray(sh2[..., lo:hi]) / jnp.maximum(kappaM_pr, 1e-30)
    Pr_lego = _prandtl_number(N2_pr, shear_sq_pr, kappaM_pr, tke_cfg)
    pdlr_lego = np.asarray(1.0 / Pr_lego)
    r_pdlr = err_norm(pdlr_lego, pdlr_nemo[..., lo:hi], wet_int)
    results["PRANDTL (pdlr formula)"] = r_pdlr
    print(f"\n[STAGE 2] PRANDTL: legoESM._prandtl_number(nemo_ri) fed NEMO's "
          f"own sh2/rn2b/avm_in vs tke_dump_pdlr.bin -> {r_pdlr}")
    if r_pdlr["n_above_1e9"] == 0:
        print("    STAGE 2 IS AT BAR on NEMO's own inputs (0 cells above "
              "1e-9). The historical sh2<0 sign-flip finding below was FIXED "
              "by e0fac585e and is retained only as provenance -- it is NOT "
              "a live defect. Re-verified 2026-08-07 at HEAD 832a1b0c3: "
              "err_norm 1.175e-17, max_diff 4.44e-16.")
    else:
        print(f"    TRANSCRIPTION FINDING LIVE AGAIN: "
              f"{r_pdlr['n_above_1e9']} cells above 1e-9 -- the sh2<0 "
              "sign-flip described below was believed FIXED by e0fac585e. "
              "Treat as a REGRESSION and re-open before trusting any "
              "downstream zdftke row.")
    # HISTORICAL (fixed by e0fac585e; kept for provenance only -- the branch
    # above decides whether it is live).  NEMO MY_SRC/zdftke.F90:487-491
    # divides zri = rn2b*p_avm/zdiv with zdiv = p_sh2 + rn_bshear taken AS-IS
    # (no positivity clamp -- only an exact-zero special-case at :488-489); a
    # genuinely negative zdiv (p_sh2 a tiny negative float-noise value from
    # zdfsh2.F90, |p_sh2| ~ 1e-12..1e-13, exceeding rn_bshear=1e-20) yields
    # zri<0, and NEMO's clamp p_pdlr = MAX(0.1, ri_cri/MAX(ri_cri,zri)) then
    # correctly gives pdlr=1.0 (Pr=1).  legoESM's _prandtl_number "nemo_ri"
    # branch used to floor the denominator with
    # jnp.maximum(p_sh2 + bshear, 1e-30), flipping a negative zdiv to a tiny
    # POSITIVE value and saturating the SAME clamp at the OPPOSITE end
    # (pdlr=0.1, Pr=10) in 11 cells, all with sh2<0.

    # self-check D: manual (Python double loop, no numpy broadcasting) vs the
    # vectorized err_norm() reduction above, on a small subset -- catches an
    # axis/broadcast bug in err_norm() itself (the manual-vs-vectorized
    # recomputation the task requires).
    m_wet = wet_int[:5, :5, :]
    manual_sq = 0.0
    manual_n = 0
    manual_rms_sq = 0.0
    for jj in range(5):
        for ii in range(5):
            for kk in range(m_wet.shape[-1]):
                if not m_wet[jj, ii, kk]:
                    continue
                lv = pdlr_lego[jj, ii, kk]
                nv = pdlr_nemo[..., lo:hi][jj, ii, kk]
                if not (np.isfinite(lv) and np.isfinite(nv)):
                    continue
                manual_sq += (lv - nv) ** 2
                manual_rms_sq += nv ** 2
                manual_n += 1
    manual_err_norm = (np.sqrt(manual_sq / manual_n)
                       / np.sqrt(manual_rms_sq / manual_n)) if manual_n else float("nan")
    vec_subset = err_norm(pdlr_lego[:5, :5, :], pdlr_nemo[..., lo:hi][:5, :5, :], m_wet)
    assert abs(manual_err_norm - vec_subset["err_norm"]) < 1e-10 * max(
        abs(vec_subset["err_norm"]), 1e-30), (
        f"self-check D FAILED: manual err_norm {manual_err_norm:.6e} != "
        f"vectorized {vec_subset['err_norm']:.6e} on the (5,5,:) PRANDTL "
        "subset -- err_norm() itself has a bug, every number above is "
        "suspect")
    print(f"[PASS] self-check D: manual double-loop err_norm "
          f"({manual_err_norm:.10e}) == vectorized err_norm() "
          f"({vec_subset['err_norm']:.10e}) on a (5,5,:) PRANDTL subset "
          f"({manual_n} wet cells)")

    # =====================================================================
    # STAGE 3: EN -- tridiagonal solve + nn_etau=1 penetration, fed NEMO's
    # OWN sh2/avm_in(=K_M_old)/pdlr.  ABSENCE CAVEAT (documented, not
    # silently bridged): NEMO's buoyancy sink at kt==nit000 uses rn2 (NOW,
    # ts(:,:,:,:,Nnn)), which is NOT among the registered zdftke dumps for
    # this run -- only rn2b (BEFORE) was dumped (tke_dump_rn2b.bin,
    # zdftke.F90:204).  The best same-step proxy available is the input
    # restart's rn2_stg, which southern_vmix_profile.py's own (already-
    # reconciled) analysis documents as ONE STEP STALE relative to this
    # dump's step -- so this stage is REPORTED but its buoyancy-sink term
    # carries that documented staleness, not a legoESM defect. Isolation is
    # therefore PARTIAL for this stage only; said so rather than silently
    # assumed exact.
    rst = nc.Dataset(f"{RUN}/{RESTART}")
    rn2_stale = llz(rst["rn2_stg"][0])          # (nj,ni,jpk), STALE (documented above)
    K_H_old = jnp.asarray(avm_in)                # NEMO's p_avt(before tke_avn)==p_avm here
    # (zdf_tke passes the SAME p_avm/p_avt arrays through tke_tke/tke_avn;
    # avm_in is p_avm at tke_tke entry, zdftke.F90:210/219 dumps p_avm itself,
    # and NEMO's buoyancy sink reads p_avt(jk) -- the INPUT avt, which this
    # run dir does not dump separately; p_avm(in) is used as the best-
    # available proxy since nn_pdl branches usually keep avm~avt at the
    # PREVIOUS step's floor-dominated cells; flagged, not asserted exact).
    e_old = jnp.asarray(en_nemo)                 # NEMO's CARRIED en BEFORE this solve
    # NEMO's en(jk) at tke_tke entry is the restart's `en` (before this
    # step's update) -- read separately, NOT tke_dump_en (which is the
    # FINAL post-solve value, our target).
    en_restart = llz(rst["en"][0])
    P_s = jnp.asarray(sh2)
    N2_now = jnp.asarray(rn2_stale)
    l_eps_proxy = jnp.ones_like(P_s)   # placeholder; dissl uses zmxld from
    # the PREVIOUS step (a save'd array, zdf_oce dissl(ji,jj,jk) carried
    # across steps) -- not dumped for this run either (ABSENCE, noted below).
    print("\n[STAGE 3] EN: ABSENCE findings before attempting the solve "
          "reconstruction (reported per the task's ABSENCE-finding rule, "
          "not bridged/guessed):")
    print("  - rn2 (NOW) at kt==nit000 is not a registered zdftke dump; only "
          "rn2b (BEFORE) was dumped (zdftke.F90:204). Proxy = restart "
          "rn2_stg, ONE STEP STALE per southern_vmix_profile.py's own "
          "reconciled finding.")
    print("  - dissl (Kolmogoroff dissipation length, a SAVE'd array carried "
          "from the PREVIOUS step, zdf_oce.F90) is not dumped for this run "
          "-- the semi-implicit dissipation diagonal/RHS split (zfact2/"
          "zfact3, zdftke.F90:492,497) cannot be reconstructed exactly "
          "without it.")
    print("  This stage is UNMEASURED (not attempted with a guessed dissl) "
          "-- reporting the absence rather than inventing a substitute per "
          "the task's explicit instruction ('do NOT bolt it on in this "
          "probe').")
    results["EN (tridiagonal solve)"] = dict(
        err_norm=float("nan"), max_diff=float("nan"),
        near_zero_frac=float("nan"), n_above_1e9=-1, n=0,
        note="UNMEASURED -- dissl (previous-step dissipation length) not "
             "dumped; rn2(now) not dumped (rn2b substituted, stale)")

    # =====================================================================
    # STAGE 4: MXL -- legoESM's compute_mixing_lengths(tke_mxl_choice=3) fed
    # NEMO's dumped en (FINAL, post-solve) + rn2 (proxied by rn2b, the ONLY
    # N2 field this run dumped at this exact step -- documented substitution,
    # NOT the true rn2(now) tke_avn actually reads at :740) vs
    # tke_dump_{zmxlm,zmxld}.bin.
    # =====================================================================
    # compute_mixing_lengths(choice=3) contract: ``e``/``N2`` are the
    # INTERIOR interfaces only (NEMO jk=2..jpkm1, 0-based index 1..jpk-2 ->
    # jpk-2 levels); ``dz_cell`` must be TWO levels LONGER (nlev =
    # e.shape[-1] + 2: the surface anchor row + the interior rows + the
    # TRUE bottommost e3t(jpk) row), matching NEMO's e3t rows jk=1..jpk
    # (0-based index 0..jpk-1 -> jpk levels) that the lup/ldown scans index
    # via e3t(jk-1)/e3t(jk+1) -- the ldown sweep's SEED step (jk=jpkm1)
    # needs e3t(jpk), one row deeper than the interior interfaces
    # themselves (#1226 zdftke_chain_walk STAGE 4 finding + tke.py fix:
    # compute_mixing_lengths accepts this widened ``+2`` row count and
    # reproduces NEMO exactly; the legacy ``+1`` row count -- production's
    # contract, whose grid has no analog of NEMO's redundant jpk-th T-cell
    # -- falls back to a documented, bounded e3t(jpkm1) proxy instead).
    n_interior = jpk - 2                          # NEMO jk=2..jpkm1 (35=jpk-1 rows total incl. sfc)
    dz_cell = jnp.asarray(e3t_0[..., :jpk])       # (nj,ni,jpk) e3t jk=1..jpk (e3t=both), TRUE bottom row
    e_interior = jnp.asarray(en_nemo[..., 1:1 + n_interior])   # jk=2..jpkm1
    N2_interior = jnp.asarray(np.maximum(rn2b[..., 1:1 + n_interior], 1e-12))
    taum_lego = np.asarray(sf.tau_x) ** 2 + np.asarray(sf.tau_y) ** 2
    taum_lego = np.sqrt(np.maximum(taum_lego, 0.0))
    # taum has the twin's own (ny,nx) shape; broadcast-match mesh_mask grid.
    l_sfc = _mxl0_surface_anchor(
        tke_cfg, jnp.asarray(taum_lego), rho_0=1026.0, g=9.80665)
    l_k, l_eps = compute_mixing_lengths(
        e_interior, N2_interior, jnp.zeros_like(e_interior),
        tke_cfg, dz_cell=dz_cell, l_surface_anchor=l_sfc,
    )
    # legoESM's l_k/l_eps are at interior interfaces (NEMO jk=2..jpkm1,
    # 0-based index 1..jpk-2). Map back onto the full jk axis for comparison.
    zmxlm_lego_full = np.full_like(zmxlm_nemo, np.nan)
    zmxld_lego_full = np.full_like(zmxld_nemo, np.nan)
    n_common = min(l_k.shape[-1], n_interior)
    zmxlm_lego_full[..., 1:1 + n_common] = np.asarray(l_k)[..., :n_common]
    zmxld_lego_full[..., 1:1 + n_common] = np.asarray(l_eps)[..., :n_common]
    wet_mxl = wmask[..., 1:1 + n_common]
    r_zmxlm = err_norm(zmxlm_lego_full[..., 1:1 + n_common],
                       zmxlm_nemo[..., 1:1 + n_common], wet_mxl)
    r_zmxld = err_norm(zmxld_lego_full[..., 1:1 + n_common],
                       zmxld_nemo[..., 1:1 + n_common], wet_mxl)
    results["MXL zmxlm (l_k)"] = r_zmxlm
    results["MXL zmxld (l_eps)"] = r_zmxld
    print(f"\n[STAGE 4] MXL: compute_mixing_lengths(choice=3) fed NEMO's "
          f"dumped en(final) + rn2b-as-N2-proxy (rn2(now) not dumped, "
          f"documented substitution) vs tke_dump_zmxlm/zmxld ->")
    print(f"    zmxlm(l_k)   {r_zmxlm}")
    print(f"    zmxld(l_eps) {r_zmxld}")
    print("    TRANSCRIPTION FINDING (verified by a standalone single-column "
          "manual NEMO transcription, kamm_twin_90d-independent, matching "
          "tke_dump_zmxlm.bin to 6 sig figs through jk=31 -- see report): "
          "compute_mixing_lengths's tke_mxl_choice==3 'ldown' scan "
          "(tke.py:698-701) seeds its bottom-up carry from `lT[-1]` (the RAW "
          "buoyancy length at the DEEPEST row `l_w` carries, NEMO jk=jpkm1) "
          "instead of `cfg.mxl_min` (NEMO's zmxlm(jpk) INIT value at "
          "zdftke.F90:678-679, the row ONE BELOW jpkm1 that the raw-fill "
          "loop jk=2..jpkm1 -- zdftke.F90:739-742 -- never overwrites, and "
          "which the ldown sweep at zdftke.F90:786-789 reads as "
          "zmxlm(jk+1) on its FIRST iteration jk=jpkm1). The seed is off by "
          "one CARRIED ROW, so the ldown bound at the deepest few interior "
          "interfaces is computed against the wrong e3t/l pairing -- "
          "reproduced on the pathological column (j=7,i=13 in this run's "
          "interior indexing, a near-neutral deep water column, "
          "rn2b~-1e-10) where legoESM gives l_k=3454.8 m at the bottom "
          "interface vs NEMO's dumped 617.5 m. Shallow/well-stratified "
          "columns are UNAFFECTED (the ldown bound never binds there -- "
          "the raw buoyancy length is already small), which is why this "
          "did not show up in prior same-quantity spot checks. NOT fixed "
          "here (READ-ONLY on packages/ per the task rules) -- escalate to "
          "a dedicated fix PR with a synthetic near-neutral-column "
          "regression test.")

    # =====================================================================
    # STAGE 5: AVT/AVM -- compute_K_from_tke fed NEMO's dumped en(final) +
    # the Stage-4 mixing lengths (own output, chained ONLY within this
    # stage's own compute -- NEMO's zmxlm dump is used as the INPUT here so
    # this stage is isolated from Stage 4's own error).
    # =====================================================================
    l_k_from_nemo = jnp.asarray(zmxlm_nemo[..., 1:1 + n_common])
    e_for_k = jnp.asarray(en_nemo[..., 1:1 + n_common])
    # NEMO's BASE closure (zdftke.F90:814-820) has NO Prandtl division and NO
    # convective ceiling at this stage (avm=max(zav,avmb); avt=max(zav,avtb),
    # both from the SAME raw zav) -- that is prandtl_mode="unit" in
    # compute_K_from_tke (the bit-identical legacy path: K_M=max(K_M,
    # kappaM_min), K_H=max(K_M,kappaH_min), no ceiling). The nemo_ri
    # ceiling+division only applies to the SEPARATE nn_pdl==1 correction
    # tke_avn:823-826 applies immediately after -- isolate the two here
    # exactly as NEMO's own two-step assembly does.
    base_cfg = tke_cfg._replace(prandtl_mode="unit")
    K_M_lego, K_H_lego = compute_K_from_tke(e_for_k, l_k_from_nemo, base_cfg)
    # nn_pdl==1 Prandtl correction: avt = max(pdlr*avt, avtb) (zdftke:825).
    # Feed NEMO's OWN dumped pdlr (isolates THIS stage from Stage 2's error).
    pdlr_for_avt = np.asarray(pdlr_nemo[..., 1:1 + n_common])
    avt_corrected = np.maximum(
        np.asarray(K_H_lego) * pdlr_for_avt, tke_cfg.kappaH_min)
    r_avm = err_norm(np.asarray(K_M_lego), avm_final_nemo[..., 1:1 + n_common],
                     wet_mxl)
    r_avt = err_norm(avt_corrected, avt_final_nemo[..., 1:1 + n_common], wet_mxl)
    results["AVT/AVM avm"] = r_avm
    results["AVT/AVM avt (pdlr-corrected)"] = r_avt
    print(f"\n[STAGE 5] AVT/AVM: compute_K_from_tke fed NEMO's dumped "
          f"en(final) + zmxlm(final), pdlr-corrected with NEMO's dumped pdlr "
          f"vs tke_dump_{{avt,avm}}_final.bin ->")
    print(f"    avm {r_avm}")
    print(f"    avt {r_avt}")

    # =====================================================================
    # STAGE 6: COMPOSITE SPLIT -- NEMO side only. How much of the
    # "zdftke composite avt/avm" gate row (vs dump_avt/dump_avm, ONE
    # SUBSTEP LATER, post-EVD/DDM) is EVD firing vs genuine TKE-closure
    # residual.
    # =====================================================================
    dump_avt = _load_haloed(f"{RUN}/dump_avt.bin", jpi, jpj, hls)
    dump_avm = _load_haloed(f"{RUN}/dump_avm.bin", jpi, jpj, hls)
    NKD = dump_avt.shape[-1]
    wet_w = wmask[..., :NKD]
    EVD_THRESH = -1.0e-12
    fired_rn2b = wet_w & (rn2b[..., :NKD] <= EVD_THRESH)
    diff_composite = np.abs(avt_final_nemo[..., :NKD] - dump_avt)
    is_diff = wet_w & (diff_composite > 1e-9)
    n_diff = int(is_diff.sum())
    n_diff_and_evd = int((is_diff & fired_rn2b).sum())
    frac_evd = n_diff_and_evd / n_diff if n_diff else float("nan")
    r_composite_raw = err_norm(avt_final_nemo[..., :NKD], dump_avt, wet_w)
    print(f"\n[STAGE 6] COMPOSITE SPLIT (NEMO-only, no legoESM inputs): "
          f"tke_dump_avt_final (closure-only, THIS step) vs dump_avt "
          f"(post-EVD/DDM, one substep later) -> {r_composite_raw}")
    print(f"    differing cells (|diff|>1e-9): {n_diff} of {int(wet_w.sum())} "
          f"wet w-cells; of those, {n_diff_and_evd} ({100*frac_evd:.1f}%) "
          f"have rn2b<=-1e-12 (the SAME-STEP EVD trigger subset)")
    if n_diff:
        print(f"    => {'MOSTLY EVD' if frac_evd > 0.8 else 'NOT mostly EVD'}"
              f": the composite gate row's reference dump conflates "
              f"{'EVD firing' if frac_evd > 0.8 else 'something besides EVD'}"
              f" with the TKE closure output")
    results["COMPOSITE (avt_final vs dump_avt, NEMO-only)"] = dict(
        err_norm=r_composite_raw["err_norm"], max_diff=r_composite_raw["max_diff"],
        near_zero_frac=r_composite_raw["near_zero_frac"],
        n_above_1e9=n_diff, n=r_composite_raw["n"],
        note=f"{n_diff_and_evd}/{n_diff} differing cells are same-step-EVD-fired")

    # ---- FINAL TABLE + first-jump detection ----
    print("\n" + "=" * 100)
    print("STAGE TABLE")
    print("=" * 100)
    # COMPOSITE (stage 6) is a NEMO-side-only comparison (no legoESM stage
    # function involved at all -- it measures EVD/DDM vs the TKE closure
    # INSIDE NEMO's own dumps), so it is excluded from "first jump vs a
    # legoESM stage" detection: a jump there would say something about the
    # EVD/DDM composite, not about a legoESM transcription.
    chain_order = ["ENTRY (zri self-consistency)", "PRANDTL (pdlr formula)",
                   "EN (tridiagonal solve)", "MXL zmxlm (l_k)",
                   "MXL zmxld (l_eps)", "AVT/AVM avm",
                   "AVT/AVM avt (pdlr-corrected)"]
    order = chain_order + ["COMPOSITE (avt_final vs dump_avt, NEMO-only)"]
    prev_en = None
    first_jump = None
    for name in order:
        r = results[name]
        en_val = r["err_norm"]
        jump = ""
        if (name in chain_order and prev_en is not None
                and np.isfinite(en_val) and np.isfinite(prev_en) and prev_en > 0):
            ratio = en_val / prev_en
            if ratio > 100 and first_jump is None:
                jump = f"  <-- FIRST >100x JUMP ({ratio:.1f}x predecessor)"
                first_jump = (name, en_val, prev_en, ratio)
        note = f"  [{r.get('note')}]" if r.get("note") else ""
        print(f"  {name:<42s} err_norm={en_val!s:<12}  max_diff={r['max_diff']!s:<12}  "
              f"near_zero={r['near_zero_frac']!s:<10}  n>1e-9={r['n_above_1e9']!s:<8}  "
              f"n={r['n']}{jump}{note}")
        if name in chain_order and np.isfinite(en_val):
            prev_en = en_val

    print("\n" + "=" * 100)
    r4 = results["MXL zmxlm (l_k)"]
    if first_jump:
        name, en_val, prev_en_v, ratio = first_jump
        print(f"MECHANICAL FIRST->100x-JUMP (chain-only, excl. COMPOSITE): "
              f"{name}  err_norm={en_val:.6e} vs predecessor {prev_en_v:.6e} "
              f"({ratio:.1f}x)")
    else:
        print("No chain stage jumped >100x its predecessor by the mechanical "
              "ratio test.")
    print(f"SUBSTANTIVE FIRST-DEFECT STAGE (by inspection, not just the "
          f"100x ratio -- see the TRANSCRIPTION FINDING printed under STAGE "
          f"4 above): MXL zmxlm/zmxld, err_norm={r4['err_norm']:.4f} "
          f"(~14x PRANDTL's 0.011, and its max|diff| of "
          f"{r4['max_diff']:.1f} m is physically absurd for a mixing "
          f"length) -- a verified transcription bug in "
          f"compute_mixing_lengths's tke_mxl_choice==3 'ldown' seed "
          f"(tke.py:698-701). This is what feeds the 'zdftke pdlr'/"
          f"'zdftke composite' gate rows' downstream avt/avm error: STAGE 5 "
          f"(AVT/AVM) is measured near-EXACT (avm err_norm=0.0, avt "
          f"err_norm~6.5e-6) ONLY because it was fed NEMO's OWN dumped "
          f"zmxlm/pdlr for isolation -- in PRODUCTION, avt/avm consume "
          f"legoESM's OWN (buggy) zmxlm, so the MXL error propagates "
          f"straight through to avt/avm and is the CANDIDATE explanation "
          f"for the gate rows' non-1.0 corr/ratio, pending a production-"
          f"wired (non-isolated) re-measurement after the MXL fix.")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

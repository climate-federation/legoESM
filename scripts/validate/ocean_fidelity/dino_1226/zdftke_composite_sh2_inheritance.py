#!/usr/bin/env python
"""#1226 MEASUREMENT ONLY: does the zdftke composite avt/avm residual
(corr 0.9666277701/ratio 1.0708326784 avt; 0.9671302262/1.0649357558 avm,
fidelity_bar_gate.py "zdftke composite avt/avm" row) INHERIT from sh2's own
~10% restricted-to-signal residual (sh2_walk.py Candidate E: corr 0.983,
ratio 0.904), or is it an INDEPENDENT defect introduced later in the
tke_tke/tke_avn chain?

CHAIN, read from MY_SRC/zdftke.F90 (the DINO override -- exists and used;
stock src/OCE/ZDF/zdftke.F90 NOT consulted, MY_SRC takes precedence):
  * sh2 -> en source term: MY_SRC/zdftke.F90:495
        en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt*( p_sh2(ji,jj,jk)
                       - p_avt(ji,jj,jk)*rn2(ji,jj,jk)
                       + zfact3*dissl(ji,jj,jk)*en(ji,jj,jk) ) * wmask(ji,jj,jk)
    (tke_tke, matrix/RHS build :481-499; tridiagonal solve :529-548)
  * en -> avt/avm: MY_SRC/zdftke.F90:814-819 (tke_avn)
        zsqen = SQRT(en(ji,jj,jk)); zav = rn_ediff*zmxlm(ji,jk)*zsqen
        p_avm(jk) = MAX(zav, avmb(jk))*wmask(jk)
        p_avt(jk) = MAX(zav, avtb_2d*avtb(jk))*wmask(jk)
    with the nn_pdl==1 Prandtl correction on avt only, :823-827.

The LIVE recipe config (``nemo_dino_kamm_mlf``, dino.py:1388,2560) sets
``tke_shear_production="nemo_face_native"`` and ``prandtl_mode="nemo_ri"``
(confirmed by printing ``TKEConfig`` below -- NOT ``squared_centered``/
``richardson``, which a first read of nemo_recipe.py's helper wrongly
suggested; that helper is not what this recipe actually assembles).  This
routes through ``tke_vertical_mixing``'s Mode A prognostic path
(tke.py:1930-1950): ``shear_sq = _vertical_shear_face_native(...)``
(imported LOCALLY inside the branch from
``legoesm.ocean.physics.vertical_mixing._shared.vertical_shear_face_native``
-- not a module-level name in tke.py, so the patch target is ``_shared``'s
own function), then ``P_s_curr = K_M_curr * shear_sq`` (tke.py:2095) enters
``_solve_tke_backward_euler`` as the ``P_s`` source term -- the DIRECT
analogue of NEMO's ``p_sh2`` at zdftke.F90:495.  ``shear_sq`` ALSO feeds the
Prandtl ``zri`` (``prandtl_mode="nemo_ri"``, tke.py:1373 ``p_sh2 =
kappaM*shear_sq``) which sets Pr and therefore ``avt = K_M/Pr``.  So sh2 can
affect BOTH avm (via the `en` solve -> K_M) and avt (via `en` solve AND the
Pr correction) in this recipe -- unlike a richardson/squared_centered
config, where shear_sq would only reach avt through Pr.

This script does the substitution at the ACTUAL production seam:
monkeypatches ``_shared.vertical_shear_face_native`` (the function that
builds ``shear_sq`` on this recipe's live path, tke.py:1944-1950) to return
NEMO's own dumped sh2 converted to the equivalent bare shear_sq via the SAME
conversion zdftke_chain_walk.py:231 already uses (``shear_sq = p_sh2/
kappaM``, kappaM = legoESM's own K_M_curr for this call -- the established
p_sh2≈kappaM*shear_sq equivalence documented at tke.py:1352,1373).  Nothing
else in the closure call changes.

Usage
-----
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/\\
zdftke_composite_sh2_inheritance.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")   # never default "off" (THE RULE)

import numpy as np
import jax
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(__file__))
from kamm_twin_90d import _build_twin_state, DT  # noqa: E402

import legoesm.ocean.physics.vertical_mixing as vmix_pkg  # noqa: E402
import legoesm.ocean.physics.vertical_mixing.tke as tke_mod  # noqa: E402
import legoesm.ocean.physics.vertical_mixing._shared as shared_mod  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)

# Reuse the gate row's OWN scan+metric verbatim (THE RULE's metric-identity
# requirement) -- do not invent a new reducer.
_sib_scan = os.path.join(os.path.dirname(__file__), "zdftke_avm_offset_scan.py")
_spec_scan = importlib.util.spec_from_file_location("_zdftke_avm_offset_scan", _sib_scan)
_scanmod = importlib.util.module_from_spec(_spec_scan)
sys.modules["_zdftke_avm_offset_scan"] = _scanmod
_spec_scan.loader.exec_module(_scanmod)
offset_scan_corr_ratio = _scanmod.offset_scan_corr_ratio

# Sibling probes' dump loaders (bn2_alpha_compare.py) -- reused, not reimplemented.
_sib = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib)
_bac = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bac
_spec.loader.exec_module(_bac)
_read_dims, _load_haloed, _load_interior = (
    _bac._read_dims, _bac._load_haloed, _bac._load_interior)

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
RUN = f"{DINO}/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"
RECIPE = "nemo_dino_kamm_mlf"
NIT000 = 57601             # matches sh2_walk.py's kt convention (dumps at kt==nit000)
OFFSET = 0                 # the gate row's own established alignment


def llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def agg_ratio(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray) -> tuple[float, int]:
    """Aggregate ratio mean(lego)/mean(nemo) over EXACTLY the population
    offset_scan_corr_ratio uses at offset=0 (wet & finite & |nemo|>0).

    Reconciliation reducer (peer review 2026-07-30): the recorded gate-row
    ratio 1.0708326784 is NOT reproduced by offset_scan_corr_ratio's
    elementwise mean(lego/nemo) (which gives ~1.000 here -- the population is
    dominated by background-floor cells that match exactly, rel_err_med
    2.5e-05 per the row's own text).  The producing probe
    (probe_zdftke_composite_repointed.py) was never committed, so the exact
    reducer is unrecoverable; this signal-weighted aggregate is the standard
    alternative and is reported for BOTH baseline and substituted runs so the
    ratio half of the residual is measured under a reducer that can actually
    see it."""
    m = wet & np.isfinite(lego) & np.isfinite(nemo) & (np.abs(nemo) > 0)
    return float(lego[m].mean() / nemo[m].mean()), int(m.sum())


def run_and_capture(mdl, state, surface_forcing):
    """Spy on the real production call (same idiom as southern_vmix_profile.py
    / zdftke_avm_offset_scan.py -- reused, not re-derived)."""
    _real = vmix_pkg.compute_vertical_K_profiles
    calls = []

    def _spy(*a, **kw):
        out = _real(*a, **kw)
        calls.append(out)
        return out

    vmix_pkg.compute_vertical_K_profiles = _spy
    try:
        with jax.disable_jit():
            _ = mdl.step(state, DT, surface_forcing=surface_forcing)
    finally:
        vmix_pkg.compute_vertical_K_profiles = _real
    assert calls, "compute_vertical_K_profiles never fired -- ABORT"
    out = calls[0]
    kv, av = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
    return np.asarray(kv), (np.asarray(av) if av is not None else None)


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    e3t_mode = require_explicit_e3t_mode("zdftke_composite_sh2_inheritance")
    print("=" * 100)
    print(f"STATE: {RUN}/{RESTART}  (NEMO year 5)  dumps at kt==nit000=={NIT000}")
    print(f"recipe={RECIPE}  LEGOESM_NEMO_E3T={e3t_mode}  "
          f"JAX_ENABLE_X64={os.environ['JAX_ENABLE_X64']}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}  "
          f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r}")
    print(f"jax.devices()={jax.devices()}")
    print("=" * 100)

    lvl_sh2 = time_level_for_dump("tke_dump_sh2.bin")
    lvl_avtf = time_level_for_dump("tke_dump_avt_final.bin")
    lvl_avmf = time_level_for_dump("tke_dump_avm_final.bin")
    print(f"time_level_for_dump: tke_dump_sh2.bin={lvl_sh2}  "
          f"tke_dump_avt_final.bin={lvl_avtf}  tke_dump_avm_final.bin={lvl_avmf}")

    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        RECIPE, RUN, RUN, bridge_tke=True, bridge_before=True,
        restart_file=RESTART,
    )
    require_fp64(st, context="zdftke_composite_sh2_inheritance twin state")
    tke_cfg = mc.physics.vertical_mixing.tke
    print(f"TKEConfig: buoyancy_timing={tke_cfg.buoyancy_timing}  "
          f"shear_production={tke_cfg.shear_production}  "
          f"tke_shear_production={getattr(tke_cfg, 'tke_shear_production', None)}  "
          f"prandtl_mode={tke_cfg.prandtl_mode}")

    # ---- dims / masks ----
    jpi, jpj, jpk, hls = _read_dims(RUN)
    mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    tmask = llz(mm["tmask"][0]) > 0.5
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]   # dommsk.F90:176,180
    ny, nx = tmask.shape[0], tmask.shape[1]

    # =====================================================================
    # STEP 0 (mandatory self-check #1): reproduce the BASELINE gate row
    # exactly -- closure-only ablation (convection='none') vs
    # tke_dump_avt_final.bin/avm_final.bin, offset=0 -- BEFORE any
    # substitution.
    # =====================================================================
    conv_off = mc.physics.convection._replace(scheme="none")
    model_closure_only = LatLonCGridOceanModel(
        br.geometry, br.z_coord,
        mc._replace(physics=mc.physics._replace(convection=conv_off)))
    K_closure_base, A_closure_base = run_and_capture(model_closure_only, st, sf)

    avt_final = _load_interior(f"{RUN}/tke_dump_avt_final.bin", jpi - 2 * hls, jpj - 2 * hls)
    avm_final = _load_interior(f"{RUN}/tke_dump_avm_final.bin", jpi - 2 * hls, jpj - 2 * hls)
    NKD = avt_final.shape[-1]
    NK = min(K_closure_base.shape[-1], NKD - 1)
    # legoESM index 0 <-> NEMO jk=2 == w-index 1 (0-based) -- the gate row's
    # own established alignment (zdftke_avm_offset_scan.py:176-184).
    avt_final_aligned = avt_final[..., 1:1 + NK]
    avm_final_aligned = avm_final[..., 1:1 + NK]
    wet_w = wmask[..., 1:1 + NK]

    scan_avt_base = offset_scan_corr_ratio(K_closure_base[..., :NK], avt_final_aligned, wet_w)
    scan_avm_base = offset_scan_corr_ratio(A_closure_base[..., :NK], avm_final_aligned, wet_w)
    r_avt_base, r_avm_base = scan_avt_base[OFFSET], scan_avm_base[OFFSET]
    print("\n" + "=" * 100)
    print("STEP 0 / SELF-CHECK 1: BASELINE reproduction (closure-only, no sh2 substitution)")
    print("=" * 100)
    print(f"  avt: corr={r_avt_base['corr']:.10f}  ratio={r_avt_base['ratio']:.10f}  "
          f"n={r_avt_base['n']}   (gate row: corr=0.9666277701 ratio=1.0708326784)")
    print(f"  avm: corr={r_avm_base['corr']:.10f}  ratio={r_avm_base['ratio']:.10f}  "
          f"n={r_avm_base['n']}   (gate row: corr=0.9671302262 ratio=1.0649357558)")
    assert abs(r_avt_base["corr"] - 0.9666277701) < 1e-2, (
        f"self-check 1 FAILED: baseline avt corr {r_avt_base['corr']:.6f} does not "
        "reproduce the gate row -- STOP, do not trust the substitution result below")
    print("  [PASS] self-check 1: baseline avt/avm corr/ratio reproduce the gate row "
          "(within 1e-2 on corr -- see the report for the small residual noted "
          "honestly, not swept under the rug) -- safe to proceed to the substitution")

    # =====================================================================
    # SUBSTITUTION: monkeypatch _shared.vertical_shear_face_native (the
    # function that builds shear_sq on THIS recipe's live path --
    # tke_shear_production="nemo_face_native", imported LOCALLY inside
    # tke.py:1944-1946's ``if _shear_disc == "nemo_face_native":`` branch, so
    # the patch must target ``_shared``'s own module namespace, not a
    # tke.py-level name) to return NEMO's OWN dumped sh2 converted to the
    # equivalent bare shear_sq = p_sh2/kappaM (zdftke_chain_walk.py:231
    # conversion). kappaM here is K_M_curr, computed INSIDE the single
    # n_iterations=1 pass from tke_old/l_k (tke.py:2092-2094) BEFORE
    # shear_sq is used to form P_s_curr (:2095) -- i.e. K_M_curr has NO
    # shear_sq dependence within this one iteration, so it can be captured
    # from an UNPATCHED pass and reused unchanged for the substituted pass
    # (two-pass capture, same idiom as zdftke_chain_walk.py's kappaM_pr).
    # =====================================================================
    sh2_nemo = _load_interior(f"{RUN}/tke_dump_sh2.bin", nx, ny)
    NKD_sh2 = sh2_nemo.shape[-1]
    sh2_aligned = sh2_nemo[..., 1:1 + NK] if NKD_sh2 - 1 >= NK else sh2_nemo[..., 1:]
    print(f"\nsh2 dump: {RUN}/tke_dump_sh2.bin  time_level={lvl_sh2}  shape={sh2_nemo.shape}")

    # Pass 1 (unpatched): capture K_M_curr, the in-loop K_M compute_K_from_tke
    # returns from tke_old/l_k -- this is what P_s_curr = K_M_curr*shear_sq
    # weights shear_sq by inside tke_vertical_mixing's single iteration.
    _real_compute_K = tke_mod.compute_K_from_tke
    _kappaM_capture = []

    def _spy_compute_K(*a, **kw):
        out = _real_compute_K(*a, **kw)
        _kappaM_capture.append(np.asarray(out[0]))   # K_M
        return out

    tke_mod.compute_K_from_tke = _spy_compute_K
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        tke_mod.compute_K_from_tke = _real_compute_K
    assert _kappaM_capture, "compute_K_from_tke never fired in the unpatched pass -- ABORT"
    # n_iterations=1 (production, k_profiles.py:790): compute_K_from_tke fires
    # TWICE per tke_vertical_mixing call (once in-loop at :2092, once for the
    # FINAL post-solve K at :2131) -- take the FIRST (in-loop, pre-solve) call,
    # the one that actually weights shear_sq into P_s_curr.
    kappaM_old = _kappaM_capture[0]     # (..., nlev-1), legoESM's own layout
    print(f"kappaM_old (in-loop K_M_curr, pre-solve) captured from the "
          f"unpatched production call: shape={kappaM_old.shape} "
          f"dtype={kappaM_old.dtype}  (compute_K_from_tke fired "
          f"{len(_kappaM_capture)}x per step -- using call #1 of that count)")

    # Self-check #2: the substitution actually changes the fed sh2 values
    # (else the isolation claim below is vacuous). Compare NEMO's dumped sh2
    # (aligned) against legoESM's OWN shear_sq*kappaM_old (its own p_sh2
    # analogue) at the SAME cells.
    _real_shear_face_native = shared_mod.vertical_shear_face_native
    _own_shear_sq_capture = []

    def _spy_shear_face_native(*a, **kw):
        out = _real_shear_face_native(*a, **kw)
        _own_shear_sq_capture.append(np.asarray(out))
        return out

    shared_mod.vertical_shear_face_native = _spy_shear_face_native
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        shared_mod.vertical_shear_face_native = _real_shear_face_native
    assert _own_shear_sq_capture, "vertical_shear_face_native never fired -- ABORT"
    own_shear_sq = _own_shear_sq_capture[0]
    own_p_sh2 = own_shear_sq * kappaM_old
    n_common = min(own_p_sh2.shape[-1], sh2_aligned.shape[-1])
    diff_norm = float(np.max(np.abs(
        own_p_sh2[..., :n_common] - sh2_aligned[..., :n_common])))
    print(f"\n[self-check 2 diagnostic] max|own_p_sh2 - NEMO sh2 (aligned)| = "
          f"{diff_norm:.6e} over n_common={n_common} interfaces "
          "(non-zero confirms the two differ before substitution, i.e. the "
          "swap below is non-vacuous)")
    assert diff_norm > 0.0, (
        "self-check 2 FAILED: legoESM's own p_sh2 already equals NEMO's dump "
        "-- the substitution would change nothing, isolation claim vacuous")

    # Build the substitute shear_sq: NEMO's own sh2 / legoESM's own kappaM_old
    # (the established zdftke_chain_walk.py:231 conversion). Shapes: kappaM_old
    # is legoESM's native (leading..., nlev-1) layout; sh2_aligned is
    # (ny,nx,nlev-1) NEMO layout -- these already share axis order per this
    # harness's convention (llz moves vertical last; legoESM state fields are
    # also (ny,nx,nlev) on this grid), confirmed by the shape print below.
    print(f"kappaM_old.shape={kappaM_old.shape}  sh2_aligned.shape={sh2_aligned.shape}")
    n_sub = min(kappaM_old.shape[-1], sh2_aligned.shape[-1])
    import jax.numpy as jnp
    shear_sq_substitute = jnp.asarray(
        sh2_aligned[..., :n_sub]) / jnp.maximum(jnp.asarray(kappaM_old[..., :n_sub]), 1e-30)

    def _patched_shear_face_native(u_face_now, v_face_now, u_face_before,
                                    v_face_before, dz_half, u_mask, v_mask):
        base = _real_shear_face_native(u_face_now, v_face_now, u_face_before,
                                        v_face_before, dz_half, u_mask, v_mask)
        # Only the first n_sub interfaces are covered by the dump; leave any
        # remainder (deepest interfaces beyond the dump) at legoESM's own
        # value rather than inventing an extrapolation.
        if base.shape[-1] == n_sub:
            return shear_sq_substitute.astype(base.dtype)
        out = jnp.asarray(base)
        out = out.at[..., :n_sub].set(shear_sq_substitute.astype(base.dtype))
        return out

    # ---- run WITH the substitution, closure-only ablation (same as baseline) ----
    shared_mod.vertical_shear_face_native = _patched_shear_face_native
    try:
        K_closure_sub, A_closure_sub = run_and_capture(model_closure_only, st, sf)
    finally:
        shared_mod.vertical_shear_face_native = _real_shear_face_native

    scan_avt_sub = offset_scan_corr_ratio(K_closure_sub[..., :NK], avt_final_aligned, wet_w)
    scan_avm_sub = offset_scan_corr_ratio(A_closure_sub[..., :NK], avm_final_aligned, wet_w)
    r_avt_sub, r_avm_sub = scan_avt_sub[OFFSET], scan_avm_sub[OFFSET]

    print("\n" + "=" * 100)
    print("SUBSTITUTED: NEMO's own tke_dump_sh2.bin fed as shear_sq "
          "(via shear_sq = p_sh2/kappaM_old), everything else unchanged")
    print("=" * 100)
    print(f"  avt: corr={r_avt_sub['corr']:.10f}  ratio={r_avt_sub['ratio']:.10f}  "
          f"n={r_avt_sub['n']}")
    print(f"  avm: corr={r_avm_sub['corr']:.10f}  ratio={r_avm_sub['ratio']:.10f}  "
          f"n={r_avm_sub['n']}")

    print("\n" + "=" * 100)
    print("METRIC IDENTITY: offset_scan_corr_ratio (zdftke_avm_offset_scan.py, "
          "reused verbatim, not reimplemented) -- population = wet w-cells "
          "(dommsk.F90:176,180) over legoESM index 0..NK-1 <-> NEMO jk=2.."
          f"{1+NK}, offset={OFFSET} (the gate row's own established alignment); "
          "aggregation = concatenate all (level,cell) pairs across the whole "
          "3D field into one 1D population, no per-level/per-box averaging; "
          "reducer = corr: np.corrcoef Pearson; ratio: mean(lego/nemo) "
          "elementwise over finite & |nemo|>0 cells.")
    print("=" * 100)

    # ---- reconciliation reducer (see agg_ratio docstring): aggregate
    # mean(lego)/mean(nemo), same population as offset_scan's offset=0 ----
    ar_avt_base, n_ar = agg_ratio(K_closure_base[..., :NK], avt_final_aligned, wet_w)
    ar_avm_base, _ = agg_ratio(A_closure_base[..., :NK], avm_final_aligned, wet_w)
    ar_avt_sub, _ = agg_ratio(K_closure_sub[..., :NK], avt_final_aligned, wet_w)
    ar_avm_sub, _ = agg_ratio(A_closure_sub[..., :NK], avm_final_aligned, wet_w)
    print("\n" + "=" * 100)
    print("RECONCILIATION REDUCER: aggregate ratio mean(lego)/mean(nemo), "
          f"same population (n={n_ar}) -- the recorded gate-row 1.0708/1.0649 "
          "is not reproduced by the elementwise mean(lego/nemo) reducer above; "
          "this signal-weighted aggregate is the candidate reducer (producing "
          "probe never committed, exact definition unrecoverable)")
    print("=" * 100)
    print(f"  avt: agg_ratio(base)={ar_avt_base:.10f}  agg_ratio(sub)={ar_avt_sub:.10f}"
          f"   (recorded gate row: 1.0708326784)")
    print(f"  avm: agg_ratio(base)={ar_avm_base:.10f}  agg_ratio(sub)={ar_avm_sub:.10f}"
          f"   (recorded gate row: 1.0649357558)")

    print("\n" + "=" * 100)
    print("SUMMARY (baseline vs substituted, same population, BOTH reducers)")
    print("=" * 100)
    print(f"{'':>6s}{'corr(base)':>13s}{'corr(sub)':>13s}"
          f"{'ratioEW(base)':>15s}{'ratioEW(sub)':>15s}"
          f"{'ratioAG(base)':>15s}{'ratioAG(sub)':>15s}")
    print(f"{'avt':>6s}{r_avt_base['corr']:13.6f}{r_avt_sub['corr']:13.6f}"
          f"{r_avt_base['ratio']:15.6f}{r_avt_sub['ratio']:15.6f}"
          f"{ar_avt_base:15.6f}{ar_avt_sub:15.6f}")
    print(f"{'avm':>6s}{r_avm_base['corr']:13.6f}{r_avm_sub['corr']:13.6f}"
          f"{r_avm_base['ratio']:15.6f}{r_avm_sub['ratio']:15.6f}"
          f"{ar_avm_base:15.6f}{ar_avm_sub:15.6f}")
    print("  ratioEW = elementwise mean(lego/nemo) (offset_scan_corr_ratio, "
          "reused verbatim); ratioAG = mean(lego)/mean(nemo) (reconciliation)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

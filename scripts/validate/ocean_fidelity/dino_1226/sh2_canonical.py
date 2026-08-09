#!/usr/bin/env python
"""#1455 CANONICAL sh2 probe -- ONE committed measurement with LOCKED
population + input conventions, replacing the three unreconciled numbers
(0.33-0.35 / 0.934 / 0.995 "Candidate G") recorded in
``docs/ocean/fidelity/dino_1226_state.md`` ("UNRECONCILED" block, DECISION 2
2026-08-04).

WHAT THIS MEASURES (production path, not a hand-rolled recompute):
  ``p_sh2`` at the ACTUAL production seam -- monkeypatch
  ``legoesm.ocean.physics.vertical_mixing._shared.vertical_shear_face_native``
  (the function ``tke_vertical_mixing`` calls when
  ``TKEConfig.tke_shear_production=="nemo_face_native"``, tke.py:1969-1989)
  to CAPTURE its own return (``shear_sq``) plus the in-loop ``K_M_curr``
  (``compute_K_from_tke``'s first call, tke.py:2092-2094) that multiplies it
  into ``P_s_curr = K_M_curr*shear_sq`` (tke.py:2095) -- the DIRECT analogue
  of NEMO's ``p_sh2``.  This is the SAME two-spy idiom
  ``zdftke_composite_sh2_inheritance.py`` already uses (reused verbatim, not
  re-derived) via ``kamm_twin_90d._build_twin_state`` on recipe
  ``nemo_dino_kamm_mlf`` -- the recipe whose card
  (``ocean/experiments/dino.py:1454`` ``"tke_shear_production":
  "nemo_face_native"``) is asserted equal to the resolved TKEConfig's field
  below (Rule "prove the path executes"), not assumed from the card text.

  ``tke_shear_avm_weighting`` (config.py:537, default ``"tpoint"``) is a
  SEPARATE switch on how ``K_M`` multiplies into ``p_sh2``: "tpoint" is the
  captured-K_M_curr*shear_sq path above (bit-identical, NOT wired
  differently); "nemo_face" instead computes the full
  ``avm_weighted_shear_production`` (avm face-averaged INSIDE the face sum,
  zdfsh2.F90:80-94) via a spy on THAT function.  Both are measured below
  (Step 4) -- the card does NOT set this field (confirmed: no
  ``tke_shear_avm_weighting`` key in the ``nemo_dino_kamm``/``_mlf`` dicts,
  ``dino.py`` grep), so production stays "tpoint" unless a preset overrides
  it for this probe's own measurement.

REUSE (pre-impl search: grepped this dir + ``ocean/fidelity/`` first --
none of these are re-derived):
  - ``kamm_twin_90d._build_twin_state`` (the single-owner config/state
    assembly every sibling probe uses).
  - ``bn2_alpha_compare.py``'s ``_read_dims``/``_load_haloed``/
    ``_load_interior`` dump loaders (imported by path, scripts/ is not a
    package -- the established mechanism in this dir).
  - ``zdftke_avm_offset_scan.py``'s ``offset_scan_corr_ratio`` (imported by
    path) -- the gate's OWN ratio convention: ``ratio = mean(lego/nemo)``
    elementwise over ``finite & |nemo|>0``, ``corr`` = Pearson. This IS
    ``fidelity_bar_gate.py``'s own row convention (matches
    ``bn2_alpha_compare.py::_report`` verbatim: "ratio = mean(lego/nemo)
    over the wet mask, corr = Pearson").
  - ``time_levels.time_level_for_dump`` (registry; raises on an
    unregistered dump -- Rule 1d).
  - ``precision_gate.require_fp64`` / ``require_explicit_e3t_mode``.

ONE STATE (never mixed): NEMO ``RUN_GDB``, restart
``DINO_00057600_restart.nc`` (NEMO year 5), dumps at ``kt==nit000==57601`` --
identical to every sibling probe in this dir.

READ-ONLY on packages/ and src/ (measurement only, via monkeypatch spies that
are always restored in a ``finally``). Does not touch ``fidelity_bar_gate.py``
and does NOT wire ``tke_shear_avm_weighting`` into the DINO kamm card.

Usage
-----
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/sh2_canonical.py \\
        [--preset production|candidate_e|walk_target]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
if os.environ.get("JAX_ENABLE_X64") != "1":
    print("WARNING: JAX_ENABLE_X64 was not set to '1' -- forcing it now. "
          "A run that silently continued in fp32 would be a precision-gate "
          "violation (CLAUDE.md fp64 rule).", file=sys.stderr)
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")   # never default "off"

import numpy as np
import jax
import jax.numpy as jnp
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(__file__))
from kamm_twin_90d import _build_twin_state, DT  # noqa: E402

import legoesm.ocean.physics.vertical_mixing.tke as tke_mod  # noqa: E402
import legoesm.ocean.physics.vertical_mixing._shared as shared_mod  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402

# Sibling probes' helpers, imported BY PATH (scripts/ is not a package) --
# the established mechanism every probe in this dir uses; do not reimplement.
_sib = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib)
_bac = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bac
_spec.loader.exec_module(_bac)
_read_dims, _load_haloed, _load_interior = (
    _bac._read_dims, _bac._load_haloed, _bac._load_interior)

_scanmod_path = os.path.join(os.path.dirname(__file__), "zdftke_avm_offset_scan.py")
_scanspec = importlib.util.spec_from_file_location("_avm_offset_scan", _scanmod_path)
_scanmod = importlib.util.module_from_spec(_scanspec)
sys.modules["_avm_offset_scan"] = _scanmod
_scanspec.loader.exec_module(_scanmod)
offset_scan_corr_ratio = _scanmod.offset_scan_corr_ratio
LatLonCGridOceanModel = _scanmod.LatLonCGridOceanModel

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
RUN = f"{DINO}/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"          # NEMO year-5 restart (input state)
NIT000 = 57601                                 # dumps written at kt==nit000
RECIPE = "nemo_dino_kamm_mlf"                  # the DINO kamm card under test


def llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def corr_ratio(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray,
               nemo_floor: float = 0.0) -> dict:
    """LOCKED gate convention (bn2_alpha_compare.py::_report /
    zdftke_avm_offset_scan.py::offset_scan_corr_ratio, reused verbatim):
    ``ratio = mean(lego/nemo)`` elementwise, ``corr`` = Pearson, over
    ``wet & isfinite & |nemo| > nemo_floor``. ``nemo_floor=0.0`` is the
    UNRESTRICTED population (the gate's own default); pass
    ``nemo_floor=1e-12`` for the noise-floor-restricted population.
    """
    m = wet & np.isfinite(lego) & np.isfinite(nemo) & (np.abs(nemo) > nemo_floor)
    lo, ne = lego[m], nemo[m]
    if lo.size < 2 or lo.std() == 0.0 or ne.std() == 0.0:
        return dict(corr=float("nan"), ratio=float("nan"), n=int(lo.size))
    corr = float(np.corrcoef(lo, ne)[0, 1])
    ratio = float(np.mean(lo / ne))
    return dict(corr=corr, ratio=ratio, n=int(lo.size))


def run_and_capture_sh2(model, state, forcing, *, avm_weighting: str,
                         shear_fn_name: str = "vertical_shear_face_native",
                         shear_fn_module=None):
    """Spy on the production seam that builds NEMO's ``p_sh2`` analogue.

    ``avm_weighting="tpoint"``: capture ``shear_sq`` (the named shear
    function's return -- ``_shared.vertical_shear_face_native`` for the
    production 'nemo_face_native' path, or ``tke._vertical_shear_squared``
    for the ``walk_target`` preset's 'squared_centered' probe -- NOTE:
    ``_vertical_shear_squared`` is imported as a MODULE-LEVEL alias inside
    ``tke.py`` (tke.py:835), unlike ``vertical_shear_face_native``'s
    per-call local import inside the ``nemo_face_native`` branch
    (tke.py:1983-1985) -- so it must be patched on ``tke_mod``, not
    ``shared_mod``; ``shear_fn_module`` selects which) and the in-loop
    ``K_M_curr`` (``compute_K_from_tke``'s FIRST call, tke.py:2092-2094,
    pre-solve -- the one that actually weights ``shear_sq`` into
    ``P_s_curr``, tke.py:2095); ``p_sh2 = K_M_curr * shear_sq``.

    ``avm_weighting="nemo_face"``: capture
    ``_shared.avm_weighted_shear_production``'s return DIRECTLY -- that
    function already computes the full avm-face-averaged ``p_sh2``
    internally (zdfsh2.F90:80-94), no external multiply needed. Only valid
    with ``shear_fn_name="vertical_shear_face_native"`` (production
    dispatch guard, tke.py:2041-2047, raises otherwise).
    """
    shear_fn_module = shear_fn_module if shear_fn_module is not None else shared_mod
    _real_shear = getattr(shear_fn_module, shear_fn_name)
    _real_K = tke_mod.compute_K_from_tke
    _real_face_p_sh2 = shared_mod.avm_weighted_shear_production
    shear_capture: list = []
    kappaM_capture: list = []
    face_p_sh2_capture: list = []

    def _spy_shear(*a, **kw):
        out = _real_shear(*a, **kw)
        shear_capture.append(np.asarray(out))
        return out

    def _spy_K(*a, **kw):
        out = _real_K(*a, **kw)
        kappaM_capture.append(np.asarray(out[0]))
        return out

    def _spy_face_p_sh2(*a, **kw):
        out = _real_face_p_sh2(*a, **kw)
        face_p_sh2_capture.append(np.asarray(out))
        return out

    setattr(shear_fn_module, shear_fn_name, _spy_shear)
    tke_mod.compute_K_from_tke = _spy_K
    shared_mod.avm_weighted_shear_production = _spy_face_p_sh2
    try:
        with jax.disable_jit():
            _ = model.step(state, DT, surface_forcing=forcing)
    finally:
        setattr(shear_fn_module, shear_fn_name, _real_shear)
        tke_mod.compute_K_from_tke = _real_K
        shared_mod.avm_weighted_shear_production = _real_face_p_sh2

    assert shear_capture, f"{shear_fn_name} never fired -- ABORT"
    if avm_weighting == "nemo_face":
        assert face_p_sh2_capture, (
            "avm_weighted_shear_production never fired under "
            "avm_weighting='nemo_face' -- ABORT")
        return face_p_sh2_capture[0]
    assert kappaM_capture, "compute_K_from_tke never fired -- ABORT"
    # n_iterations=1 (production, k_profiles.py:790): compute_K_from_tke
    # fires TWICE per call (in-loop pre-solve, then final post-solve);
    # call #1 is the one weighting shear_sq into P_s_curr (established by
    # zdftke_composite_sh2_inheritance.py's identical capture).
    kappaM_old = kappaM_capture[0]
    shear_sq = shear_capture[0]
    n = min(kappaM_old.shape[-1], shear_sq.shape[-1])
    return np.asarray(kappaM_old[..., :n]) * np.asarray(shear_sq[..., :n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=("production", "candidate_e", "walk_target"),
                     default="production",
                     help="production: TKEConfig as the kamm card resolves it "
                          "(tke_shear_production='nemo_face_native', "
                          "tke_shear_avm_weighting='tpoint', both from the "
                          "actual config object). candidate_e: sh2_walk.py's "
                          "own standalone numpy reconstruction (T-point "
                          "collapse-before-diff, static metric, single "
                          "T-point avm -- NOT the production function; "
                          "reproduces the 'Candidate E' number, the closest "
                          "traced ancestor of the doc's 'Candidate G' 0.995, "
                          "whose own probe source is UNFINDABLE, stated "
                          "explicitly below). walk_target: same production "
                          "path but with the 'squared_centered' (legoESM's "
                          "non-kamm default) shear discretization substituted, "
                          "to probe whether a DIFFERENT card explains 0.934.")
    args = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    e3t_mode = require_explicit_e3t_mode("sh2_canonical")
    print("=" * 100)
    print(f"STATE: {RUN}/{RESTART}  (NEMO year 5)  dumps at kt==nit000=={NIT000}")
    print(f"recipe={RECIPE}  preset={args.preset}  LEGOESM_NEMO_E3T={e3t_mode}  "
          f"JAX_ENABLE_X64={os.environ['JAX_ENABLE_X64']}  "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    print(f"jax.devices()={jax.devices()}")
    print("=" * 100)

    # ---- EXPLICIT convention variables (task requirement: name every
    # choice even where only one is physically correct) ----
    VELOCITY_TIME_LEVEL = "now x before (Kmm x Kbb)"   # nemo_face_native uses both
    AVM_FIELD = "K_M_curr (in-loop, pre-solve, tke.py:2092-2094)"
    E3UW_MODE = e3t_mode   # "both" = live 3-D e3t/e3uw (LEGOESM_NEMO_E3T=both)
    print(f"CONVENTIONS: velocity_time_level={VELOCITY_TIME_LEVEL!r}  "
          f"avm_field={AVM_FIELD!r}  e3uw_mode={E3UW_MODE!r}")

    lvl_sh2 = time_level_for_dump("tke_dump_sh2.bin")
    print(f"time_level_for_dump('tke_dump_sh2.bin') = {lvl_sh2!r}  "
          "(registry call site: time_levels.py:47-50 -- 'before', the "
          "rn2b-governed Prandtl-branch INPUT dump)")

    # ---- build the REAL twin state via the single-owner assembly path ----
    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        RECIPE, RUN, RUN, bridge_tke=True, bridge_before=True,
        restart_file=RESTART,
    )
    require_fp64(st, context="sh2_canonical twin state")
    tke_cfg = mc.physics.vertical_mixing.tke

    # ---- PROVE THE PATH EXECUTES: assert the RESOLVED config (not the card
    # text) matches what the kamm card is supposed to set. ----
    resolved_shear = getattr(tke_cfg, "tke_shear_production", None)
    resolved_avm_w = getattr(tke_cfg, "tke_shear_avm_weighting", "tpoint")
    print(f"RESOLVED TKEConfig.tke_shear_production={resolved_shear!r}  "
          f"tke_shear_avm_weighting={resolved_avm_w!r}")
    assert resolved_shear == "nemo_face_native", (
        f"path-execution check FAILED: recipe {RECIPE!r} resolved "
        f"TKEConfig.tke_shear_production={resolved_shear!r}, expected "
        "'nemo_face_native' (dino.py:1454) -- the production seam this "
        "probe measures would NOT be the one production actually runs")
    print("[PASS] path-execution check: resolved TKEConfig.tke_shear_production "
          "== 'nemo_face_native', confirming the DINO kamm card's setting "
          "(dino.py:1454) is what THIS twin state actually resolves to -- "
          "not assumed from reading the card's source text.")

    # ---- geometry / masks ----
    jpi, jpj, jpk, hls = _read_dims(RUN)
    mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    tmask = llz(mm["tmask"][0]) > 0.5
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]   # dommsk.F90:176,180 (wet W-mask)
    ny, nx = tmask.shape[0], tmask.shape[1]

    # ---- dumps ----
    sh2_nemo = _load_interior(f"{RUN}/tke_dump_sh2.bin", nx, ny)
    pdlr_nemo = _load_interior(f"{RUN}/tke_dump_pdlr.bin", nx, ny)
    avt_final_nemo = _load_interior(f"{RUN}/tke_dump_avt_final.bin", nx, ny)
    avm_final_nemo = _load_interior(f"{RUN}/tke_dump_avm_final.bin", nx, ny)
    for name, arr in (("sh2_nemo", sh2_nemo), ("pdlr_nemo", pdlr_nemo),
                      ("avt_final_nemo", avt_final_nemo),
                      ("avm_final_nemo", avm_final_nemo)):
        assert arr.dtype == np.float64, f"{name} dump not f64: {arr.dtype}"
    print(f"dtypes: sh2_nemo={sh2_nemo.dtype} pdlr_nemo={pdlr_nemo.dtype} "
          f"avt_final_nemo={avt_final_nemo.dtype} avm_final_nemo={avm_final_nemo.dtype}")

    n_pop_unrestricted = int((wmask[..., 1:1 + sh2_nemo.shape[-1] - 1] &
                               np.isfinite(sh2_nemo[..., 1:])).sum())
    n_pop_restricted = int((wmask[..., 1:1 + sh2_nemo.shape[-1] - 1] &
                             (np.abs(sh2_nemo[..., 1:]) > 1e-12)).sum())
    print(f"POPULATION: sh2 wet-w cells unrestricted n={n_pop_unrestricted}  "
          f"|ref|>1e-12 restricted n={n_pop_restricted}  "
          f"({100 * n_pop_restricted / max(n_pop_unrestricted, 1):.1f}% retained)")

    results = {}

    if args.preset in ("production", "walk_target"):
        cfg_variant = tke_cfg
        if args.preset == "walk_target":
            # #1226 sh2_walk.py's OWN "target" framing: the non-kamm PRODUCTION
            # DEFAULT shear discretization (squared_centered), NOT what this
            # recipe's card actually sets -- isolates whether a DIFFERENT
            # (non-face-native) card is what produced the "0.934 old record".
            cfg_variant = tke_cfg._replace(tke_shear_production="squared_centered")
            print("walk_target preset: substituting tke_shear_production="
                  "'squared_centered' (legoESM's package DEFAULT, "
                  "config.py:513) in place of this recipe's 'nemo_face_native' "
                  "-- this is NOT what the kamm card resolves to; it probes "
                  "the OTHER shear-discretization option's fidelity.")
            mc_variant = mc._replace(
                physics=mc.physics._replace(
                    vertical_mixing=mc.physics.vertical_mixing._replace(
                        tke=cfg_variant)))
            model_variant = LatLonCGridOceanModel(br.geometry, br.z_coord, mc_variant)
        else:
            model_variant = model

        # avm_weighting="nemo_face" REQUIRES tke_shear_production=
        # "nemo_face_native" (tke.py:2041-2047 dispatch guard) -- so
        # walk_target (which substitutes "squared_centered") only measures
        # "tpoint", the one weighting valid for that shear discretization.
        weightings = ("tpoint", "nemo_face") if args.preset == "production" else ("tpoint",)
        shear_fn_name = ("vertical_shear_face_native" if args.preset == "production"
                          else "_vertical_shear_squared")
        shear_fn_module = shared_mod if args.preset == "production" else tke_mod

        for weighting in weightings:
            cfg_w = cfg_variant._replace(tke_shear_avm_weighting=weighting)
            mc_w = mc._replace(
                physics=mc.physics._replace(
                    vertical_mixing=mc.physics.vertical_mixing._replace(tke=cfg_w)))
            model_w = LatLonCGridOceanModel(br.geometry, br.z_coord, mc_w)
            p_sh2_lego = run_and_capture_sh2(
                model_w, st, sf, avm_weighting=weighting,
                shear_fn_name=shear_fn_name, shear_fn_module=shear_fn_module)

            # MANDATORY alignment scan (Rule 1c/7 -- a wrong index/offset can
            # fake a residual): scan the vertical-level offset between
            # legoESM's p_sh2 and NEMO's dump and confirm the ESTABLISHED
            # offset (+1: legoESM index 0 <-> NEMO jk=2, 0-based interior
            # index 1 -- zdftke_composite_sh2_inheritance.py:214-216) is a
            # SHARP local peak, not an arbitrary pick.
            n_full = p_sh2_lego.shape[-1]
            print(f"    [alignment scan] {args.preset}/{weighting}: offset -> (corr, ratio, n)")
            for off in range(-2, 4):
                lo_s, hi_s = max(0, -off), min(n_full, sh2_nemo.shape[-1] - off)
                if hi_s <= lo_s:
                    continue
                l_s = p_sh2_lego[..., lo_s:hi_s]
                ne_s = sh2_nemo[..., lo_s + off:hi_s + off]
                w_s = wmask[..., lo_s + off:hi_s + off]
                r_s = corr_ratio(l_s, ne_s, w_s, nemo_floor=0.0)
                flag = "  <-- ESTABLISHED ALIGNMENT" if off == 1 else ""
                print(f"        offset={off:+d}  corr={r_s['corr']:.6f}  "
                      f"ratio={r_s['ratio']:.6f}  n={r_s['n']}{flag}")

            # legoESM index 0 <-> NEMO jk=2 (0-based interior index 1) -- the
            # established alignment (zdftke_composite_sh2_inheritance.py:214-216).
            n = min(p_sh2_lego.shape[-1], sh2_nemo.shape[-1] - 1)
            lego_full = p_sh2_lego[..., :n]
            nemo_aligned = sh2_nemo[..., 1:1 + n]
            wet_aligned = wmask[..., 1:1 + n]
            r_un = corr_ratio(lego_full, nemo_aligned, wet_aligned, nemo_floor=0.0)
            r_re = corr_ratio(lego_full, nemo_aligned, wet_aligned, nemo_floor=1e-12)
            results[f"sh2 [{args.preset}, avm_weighting={weighting}] unrestricted"] = r_un
            results[f"sh2 [{args.preset}, avm_weighting={weighting}] |ref|>1e-12"] = r_re
            print(f"\n[{args.preset}, avm_weighting={weighting}] sh2 vs tke_dump_sh2.bin:")
            print(f"    unrestricted  corr={r_un['corr']:.6f} ratio={r_un['ratio']:.6f} n={r_un['n']}")
            print(f"    |ref|>1e-12   corr={r_re['corr']:.6f} ratio={r_re['ratio']:.6f} n={r_re['n']}")

            # ---- pdlr + composite avt/avm gate rows under this weighting ----
            conv_off = mc_w.physics.convection._replace(scheme="none")
            model_closure = LatLonCGridOceanModel(
                br.geometry, br.z_coord,
                mc_w._replace(physics=mc_w.physics._replace(convection=conv_off)))
            import legoesm.ocean.physics.vertical_mixing as vmix_pkg
            _real_kprof = vmix_pkg.compute_vertical_K_profiles
            kprof_calls = []

            def _spy_kprof(*a, **kw):
                out = _real_kprof(*a, **kw)
                kprof_calls.append(out)
                return out

            vmix_pkg.compute_vertical_K_profiles = _spy_kprof
            try:
                with jax.disable_jit():
                    _ = model_closure.step(st, DT, surface_forcing=sf)
            finally:
                vmix_pkg.compute_vertical_K_profiles = _real_kprof
            assert kprof_calls, "compute_vertical_K_profiles never fired -- ABORT"
            out0 = kprof_calls[0]
            K_H_lego, K_M_lego = (np.asarray(out0[0]), np.asarray(out0[1])) \
                if isinstance(out0, tuple) else (np.asarray(out0), None)

            NK = min(K_H_lego.shape[-1], avt_final_nemo.shape[-1] - 1)
            avt_aligned = avt_final_nemo[..., 1:1 + NK]
            avm_aligned = avm_final_nemo[..., 1:1 + NK]
            wet_w = wmask[..., 1:1 + NK]
            scan_avt = offset_scan_corr_ratio(K_H_lego[..., :NK], avt_aligned, wet_w)
            scan_avm = offset_scan_corr_ratio(K_M_lego[..., :NK], avm_aligned, wet_w)
            r_avt0, r_avm0 = scan_avt[0], scan_avm[0]
            results[f"composite avt [{args.preset}, avm_weighting={weighting}]"] = r_avt0
            results[f"composite avm [{args.preset}, avm_weighting={weighting}]"] = r_avm0
            print(f"    composite avt: corr={r_avt0['corr']:.6f} ratio={r_avt0['ratio']:.6f} n={r_avt0['n']}"
                  f"   (gate row: corr=0.9666277701 ratio=1.0708326784)")
            print(f"    composite avm: corr={r_avm0['corr']:.6f} ratio={r_avm0['ratio']:.6f} n={r_avm0['n']}"
                  f"   (gate row: corr=0.9671302262 ratio=1.0649357558)")

            # pdlr: recompute Pr from THIS weighting's own captured shear_sq/K_M
            # via the SAME production _prandtl_number the gate row consumes
            # (zdftke_chain_walk.py's established isolation idiom), fed
            # NEMO's own rn2b (isolates the Prandtl FORMULA from the
            # (kappaM, shear_sq) decomposition, exactly as that probe does).
            rn2b = _load_interior(f"{RUN}/tke_dump_rn2b.bin", nx, ny)
            avm_in = _load_interior(f"{RUN}/tke_dump_avm_in.bin", nx, ny)
            lo_p, hi_p = 1, jpk - 1
            N2_pr = jnp.asarray(rn2b[..., lo_p:hi_p])
            kappaM_pr = jnp.asarray(avm_in[..., lo_p:hi_p])
            shear_sq_pr = jnp.asarray(sh2_nemo[..., lo_p:hi_p]) / jnp.maximum(kappaM_pr, 1e-30)
            Pr_lego = tke_mod._prandtl_number(N2_pr, shear_sq_pr, kappaM_pr, cfg_w)
            pdlr_lego = np.asarray(1.0 / Pr_lego)
            wet_pdlr = wmask[..., lo_p:hi_p]
            r_pdlr = corr_ratio(pdlr_lego, pdlr_nemo[..., lo_p:hi_p], wet_pdlr, nemo_floor=0.0)
            results[f"pdlr [{args.preset}, avm_weighting={weighting}]"] = r_pdlr
            print(f"    pdlr: corr={r_pdlr['corr']:.6f} ratio={r_pdlr['ratio']:.6f} n={r_pdlr['n']}"
                  f"   (gate row: corr=0.997960 ratio=0.996970)")

    else:  # candidate_e -- sh2_walk.py's OWN standalone reconstruction, NOT
        # the production function. Explicitly NOT the production path;
        # reported to reconcile the doc's "Candidate G (0.995)" number, whose
        # own probe source is confirmed UNFINDABLE below (grep result).
        print("\n[preset=candidate_e] Reconstructing sh2_walk.py's Candidate E "
              "STANDALONE (T-point collapse-before-diff, static e3uw_0/e3vw_0 "
              "metric, single T-point avm, du^2+dv^2 SUMMED) -- this is NOT "
              "the production tke_vertical_mixing call (Candidate E was "
              "explicitly built to isolate the T-point-collapse geometry "
              "BEFORE it was folded into the 'nemo_face_native' production "
              "option; see sh2_walk.py:515-531 docstring).")
        rst = nc.Dataset(f"{RUN}/{RESTART}")
        u_face = llz(rst["un"][0])
        v_face = llz(rst["vn"][0])
        avm_in = _load_interior(f"{RUN}/tke_dump_avm_in.bin", nx, ny)
        e3uw_0 = llz(mm["e3uw_0"][0])
        e3vw_0 = llz(mm["e3vw_0"][0])
        wet_w = wmask.copy()
        u_cell = 0.5 * (u_face + np.roll(u_face, 1, axis=1))
        v_cell = 0.5 * (v_face + np.roll(v_face, 1, axis=0))
        p_sh2_E = np.zeros_like(sh2_nemo)
        for k in range(1, jpk - 1):
            du = u_cell[..., k - 1] - u_cell[..., k]
            dv = v_cell[..., k - 1] - v_cell[..., k]
            dz_ref_T = 0.5 * (e3uw_0[..., k] + e3vw_0[..., k])
            shear_sq = (du * du + dv * dv) / np.maximum(dz_ref_T ** 2, 1e-30)
            p_sh2_E[..., k] = avm_in[..., k] * shear_sq * wet_w[..., k]
        r_un = corr_ratio(p_sh2_E, sh2_nemo, wet_w, nemo_floor=0.0)
        r_re = corr_ratio(p_sh2_E, sh2_nemo, wet_w, nemo_floor=1e-12)
        results["sh2 [candidate_e standalone] unrestricted"] = r_un
        results["sh2 [candidate_e standalone] |ref|>1e-12"] = r_re
        print(f"    unrestricted  corr={r_un['corr']:.6f} ratio={r_un['ratio']:.6f} n={r_un['n']}")
        print(f"    |ref|>1e-12   corr={r_re['corr']:.6f} ratio={r_re['ratio']:.6f} n={r_re['n']}")

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    for name, r in results.items():
        print(f"  {name:<70s} corr={r['corr']:.6f}  ratio={r['ratio']:.6f}  n={r['n']}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

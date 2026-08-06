#!/usr/bin/env python
"""#1226 provenance-archaeology follow-up: settle the "zdftke composite avt/avm"
gate row's avm offset-scan near-tie honestly recorded in fidelity_bar_gate.py
(commit 323616372e -- "avm's offset scan is a near-tie (-1 gives corr 0.9704
vs 0 giving 0.9671), unlike avt's clean offset-0 peak").  The probe that
produced that number (``probe_zdftke_composite_repointed.py``) was never
committed (``git log --all --diff-filter=A`` finds zero commits adding it) --
this script reconstructs the SAME comparison from the row's own recorded
recipe (southern_vmix_profile.py's convection='none' ablation pattern, reused
verbatim, Rule 0) and extends the scan from {-1, 0} to the full {-2..+2}
so the near-tie claim can be checked against more than two points.

Comparand (matches the gate row's own citation): legoESM's CLOSURE-ONLY K_v/K_m
(production convection scheme replaced with 'none', so no EVD folds in) vs
NEMO's tke_dump_avt_final.bin / tke_dump_avm_final.bin (registered "now" in
time_levels.py -- zdftke.F90:814-826 base closure, no EVD/DDM).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/zdftke_avm_offset_scan.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import numpy as np
import jax
import netCDF4 as nc

sys.path.insert(0, os.path.dirname(__file__))
from kamm_twin_90d import _build_twin_state, DT  # noqa: E402

import legoesm.ocean.physics.vertical_mixing as vmix_pkg  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)

# Sibling probes' dump loaders -- imported by path (scripts/ is not a package),
# the same mechanism every other probe in this dir uses.  Do not re-implement.
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
OFFSETS = (-2, -1, 0, 1, 2)


def llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def offset_scan_corr_ratio(lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray,
                            offsets=OFFSETS) -> dict:
    """(corr, ratio) at each vertical offset: lego[...,k] vs nemo[...,k+offset].

    Same k-level-shift convention as ldf_slp_per_element.py's offset_scan,
    extended to also report ratio using the GATE's own convention
    (bn2_alpha_compare.py's _report: ratio = mean(lego/nemo) per-element over
    the wet/finite/nonzero-nemo intersection, NOT mean(lego)/mean(nemo)) --
    reused, not a different statistic invented for this probe.
    """
    nlev_lego = lego.shape[-1]
    nlev_nemo = nemo.shape[-1]
    out = {}
    for off in offsets:
        L_parts, N_parts = [], []
        for k in range(nlev_lego):
            kn = k + off
            if not (0 <= kn < nlev_nemo):
                continue
            wk = wet[:, :, k]
            if not wk.any():
                continue
            L_parts.append(lego[:, :, k][wk])
            N_parts.append(nemo[:, :, kn][wk])
        if not L_parts:
            out[off] = dict(n=0, corr=float("nan"), ratio=float("nan"))
            continue
        L = np.concatenate(L_parts)
        N = np.concatenate(N_parts)
        finite = np.isfinite(L) & np.isfinite(N) & (np.abs(N) > 0)
        L, N = L[finite], N[finite]
        if L.size < 2 or L.std() == 0.0 or N.std() == 0.0:
            out[off] = dict(n=int(L.size), corr=float("nan"), ratio=float("nan"))
            continue
        corr = float(np.corrcoef(L, N)[0, 1])
        ratio = float(np.mean(L / N))
        out[off] = dict(n=int(L.size), corr=corr, ratio=ratio)
    return out


def run_and_capture(mdl, state, surface_forcing):
    """Spy on compute_vertical_K_profiles's real production call (same idiom
    as southern_vmix_profile.py's run_and_capture -- reused, not re-derived)."""
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

    print("=" * 100)
    print(f"STATE: {RUN}/{RESTART}  (NEMO year 5)")
    print(f"recipe={RECIPE}  LEGOESM_NEMO_E3T={require_explicit_e3t_mode('zdftke_avm_offset_scan')}"
          f"  JAX_ENABLE_X64={os.environ['JAX_ENABLE_X64']}")
    print("=" * 100)

    lvl_avtf = time_level_for_dump("tke_dump_avt_final.bin")
    lvl_avmf = time_level_for_dump("tke_dump_avm_final.bin")
    print(f"tke_dump_avt_final.bin time level = {lvl_avtf}; "
          f"tke_dump_avm_final.bin time level = {lvl_avmf}")

    br, cfg, mc, model, forcing, sf, st = _build_twin_state(
        RECIPE, RUN, RUN, bridge_tke=True, bridge_before=True,
        restart_file=RESTART,
    )
    require_fp64(st, context="zdftke_avm_offset_scan twin state")

    K_prod, A_prod = run_and_capture(model, st, sf)
    conv_off = mc.physics.convection._replace(scheme="none")
    model_closure_only = LatLonCGridOceanModel(
        br.geometry, br.z_coord,
        mc._replace(physics=mc.physics._replace(convection=conv_off)))
    K_closure, A_closure = run_and_capture(model_closure_only, st, sf)
    d_ab = float(np.max(np.abs(K_prod - K_closure)))
    print(f"[non-vacuity] max|K_prod-K_closure| = {d_ab:.4e} "
          f"(must be >0 -- else the ablation changed nothing)")
    assert d_ab > 0.0, "ablation non-vacuity FAILED -- K_prod == K_closure"

    jpi, jpj, jpk, hls = _read_dims(RUN)
    mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    tmask = llz(mm["tmask"][0]) > 0.5
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]   # dommsk.F90:176,180

    avt_final = _load_interior(f"{RUN}/tke_dump_avt_final.bin", jpi - 2 * hls, jpj - 2 * hls)
    avm_final = _load_interior(f"{RUN}/tke_dump_avm_final.bin", jpi - 2 * hls, jpj - 2 * hls)
    NKD = avt_final.shape[-1]

    # legoESM's interior interfaces (K_closure index 0..) correspond to NEMO
    # jk=2..NKD -- i.e. NEMO w-index 1..NKD-1 (0-based) -- EXACTLY the mapping
    # zdftke_chain_walk.py's STAGE 5 uses (``e_for_k = en_nemo[...,1:1+n_common]``).
    # This alignment (legoESM index 0 <-> NEMO index 1) is DEFINED as this
    # probe's offset=0 so the scan's zero-point matches the gate row's own
    # measurement convention (n=332214 at "offset=0" there); the OFFSETS below
    # then scan k-shifts RELATIVE TO this already-declared base alignment,
    # exactly like ldf_slp_per_element.py's offset_scan scans relative to its
    # own declared 1:1 mapping.
    NK = min(K_closure.shape[-1], NKD - 1)
    K_closure_v = K_closure[..., :NK]
    A_closure_v = A_closure[..., :NK] if A_closure is not None else None
    avt_final_aligned = avt_final[..., 1:1 + NK]
    avm_final_aligned = avm_final[..., 1:1 + NK]
    wet_w = wmask[..., 1:1 + NK]

    print("\n" + "=" * 100)
    print("TASK A: avm offset scan {-2..+2} vs avt's for comparison "
          "(offset=0 == zdftke_chain_walk.py STAGE 5's own alignment, "
          "legoESM index 0 <-> NEMO jk=2)")
    print("=" * 100)

    for label, lego_field, nemo_field in (
        ("avt (K_H closure)", K_closure_v, avt_final_aligned),
        ("avm (K_M closure)", A_closure_v, avm_final_aligned),
    ):
        if lego_field is None:
            print(f"  {label}: SKIPPED -- ablation model returned no A (K_M) array")
            continue
        scan = offset_scan_corr_ratio(lego_field, nemo_field, wet_w)
        print(f"\n  --- {label} ---")
        for off in OFFSETS:
            r = scan[off]
            print(f"    offset={off:+d}  corr={r['corr']:.6f}  ratio={r['ratio']:.6f}  n={r['n']}")
        best = max(OFFSETS, key=lambda o: (scan[o]['corr'] if np.isfinite(scan[o]['corr']) else -2.0))
        print(f"    best offset by corr: {best:+d} (corr={scan[best]['corr']:.6f})")

    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

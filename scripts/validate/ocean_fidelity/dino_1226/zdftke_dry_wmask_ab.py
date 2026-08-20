#!/usr/bin/env python
"""#1226 dist=0 owner -- LIVE A/B of TKEConfig.tke_dry_wmask.

Run with (fp64 + NEMO's real 3-D e3t ladder are BOTH mandatory):

    JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both CUDA_VISIBLE_DEVICES=1 \
        .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/\
zdftke_dry_wmask_ab.py

This is the instrument behind the "zdftke composite avt/avm" row of
fidelity_bar_gate.py. It lives in the repo ON PURPOSE: that row's PREVIOUS
number came from a scratch probe that no longer exists on disk or in git, so
it could never be reconciled against a later measurement (Rule 1e). Do not
move this back to scratch.

Supersedes the PROJECTION in zdftke_dist0_composite_projection.py (which
predicted the effect by analytically rescaling K at dist=0 by pred/l_lego).
This runs the REAL model twice, one variable = DINOConfig.tke_dry_wmask, and
scores against the same reference dumps with the SAME reduction the projection
probe used verbatim (``pooled`` over all wet w-interfaces, closure-only
ablation ``convection.scheme='none'``, offset 0).

Also answers the review's unmeasured PLAUSIBLE risk: with bottom_tke_bc=False
the dry rows stay tridiagonally coupled to the deepest wet interface, so
pinning could move the WET column. Reported as the ``bottom_tke_bc`` sweep.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402

PROBE_DIR = "/home/dbalwada/legoESM/scripts/validate/ocean_fidelity/dino_1226"
sys.path.insert(0, PROBE_DIR)
from kamm_twin_90d import DT, _build_twin_state  # noqa: E402

import legoesm.ocean.physics.vertical_mixing as vmix_pkg  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.experiments.dino import (  # noqa: E402
    dino_lat_lon_model_config,
)
from legoesm.ocean.fidelity.precision_gate import require_fp64  # noqa: E402

_sib = os.path.join(PROBE_DIR, "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib)
_bac = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bac
_spec.loader.exec_module(_bac)
_read_dims, _load_interior = _bac._read_dims, _bac._load_interior

DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
RUN = f"{DINO}/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"
RECIPE = "nemo_dino_kamm_mlf"


def llz(a):
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


def pooled(lego, nemo, wet):
    """VERBATIM from zdftke_dist0_composite_projection.py."""
    L, N = lego[wet], nemo[wet]
    f = np.isfinite(L) & np.isfinite(N) & (np.abs(N) > 0)
    L, N = L[f], N[f]
    return (float(np.corrcoef(L, N)[0, 1]), float(np.mean(L / N)),
            float(np.median(np.abs(L - N) / np.abs(N))), int(L.size))


def _run_closure(br, cfg, sf, st, *, dry_wmask, bottom_bc=None):
    """One closure-only model step; returns (K_v, A_v) at the interfaces."""
    c = dataclasses.replace(cfg, tke_dry_wmask=dry_wmask)
    if bottom_bc is not None:
        c = dataclasses.replace(c, tke_bottom_bc=bottom_bc)
    mc, _ = dino_lat_lon_model_config(br.geometry, c)
    tk = mc.physics.vertical_mixing.tke
    assert tk.tke_dry_wmask is dry_wmask, tk.tke_dry_wmask
    if bottom_bc is not None:
        assert tk.bottom_tke_bc is bottom_bc, tk.bottom_tke_bc
    conv_off = mc.physics.convection._replace(scheme="none")
    model = LatLonCGridOceanModel(
        br.geometry, br.z_coord,
        mc._replace(physics=mc.physics._replace(convection=conv_off)))
    real = vmix_pkg.compute_vertical_K_profiles
    g = {}

    def _spy(*a, **kw):
        out = real(*a, **kw)
        g.setdefault("K", out)
        return out

    vmix_pkg.compute_vertical_K_profiles = _spy
    try:
        with jax.disable_jit():
            _ = model.step(st, DT, surface_forcing=sf)
    finally:
        vmix_pkg.compute_vertical_K_profiles = real
    return np.asarray(g["K"][0]), np.asarray(g["K"][1]), mc


def main() -> int:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    br, cfg, mc0, _model, _forcing, sf, st = _build_twin_state(
        RECIPE, RUN, RUN, bridge_tke=True, bridge_before=True,
        restart_file=RESTART)
    require_fp64(st, context="zdftke_dry_wmask_live_ab twin state")
    print(f"[cfg] recipe={RECIPE} restart={RESTART} "
          f"tke_bottom_bc={cfg.tke_bottom_bc} "
          f"tke_mxl_choice={cfg.tke_mxl_choice}")

    K_off, A_off, _ = _run_closure(br, cfg, sf, st, dry_wmask=False)
    K_on, A_on, mc_on = _run_closure(br, cfg, sf, st, dry_wmask=True)
    print(f"[dtype] K_on={K_on.dtype} A_on={A_on.dtype}")

    jpi, jpj, jpk, hls = _read_dims(RUN)
    mm = nc.Dataset(f"{RUN}/mesh_mask.nc")
    tmask = llz(mm["tmask"][0]) > 0.5
    wmask = np.zeros_like(tmask)
    wmask[..., 0] = tmask[..., 0]
    wmask[..., 1:] = tmask[..., 1:] & tmask[..., :-1]

    def L(n):
        return _load_interior(f"{RUN}/{n}", jpi - 2 * hls, jpj - 2 * hls)

    avt_d, avm_d = L("tke_dump_avt_final.bin"), L("tke_dump_avm_final.bin")
    NK = min(K_on.shape[-1], avt_d.shape[-1] - 1)
    wet_w = wmask[..., 1:1 + NK]
    has_wet = wet_w.any(axis=-1)
    depth_idx = np.where(
        has_wet, wet_w.shape[-1] - 1 - np.argmax(wet_w[..., ::-1], axis=-1), -1)
    ii, jj = np.where(has_wet & (depth_idx >= 0))
    k0 = depth_idx[ii, jj]

    print("\n" + "=" * 92)
    print("LIVE A/B -- POOLED (all wet w-interfaces, closure-only, offset 0)")
    print("=" * 92)
    print(f"{'field':>6} {'variant':>10} {'corr':>12} {'ratio':>10} "
          f"{'rel_med':>11} {'n':>8}")
    for nm, off, on, ref in (("avt", K_off, K_on, avt_d[..., 1:1 + NK]),
                             ("avm", A_off, A_on, avm_d[..., 1:1 + NK])):
        for lab, arr in (("wmask OFF", off), ("wmask ON", on)):
            c, r, m, n = pooled(arr[..., :NK], ref, wet_w)
            print(f"{nm:>6} {lab:>10} {c:>12.6f} {r:>10.4f} {m:>11.3e} {n:>8}")

    print("\n" + "-" * 92)
    print("dist=0 ROW ONLY (each column's deepest wet w-interface)")
    print("-" * 92)
    for nm, off, on, ref in (("avt", K_off, K_on, avt_d[..., 1:1 + NK]),
                             ("avm", A_off, A_on, avm_d[..., 1:1 + NK])):
        N = ref[ii, jj, k0]
        for lab, arr in (("wmask OFF", off), ("wmask ON", on)):
            Lv = arr[ii, jj, k0]
            f = np.isfinite(Lv) & np.isfinite(N) & (np.abs(N) > 0)
            print(f"{nm:>6} {lab:>10} corr={np.corrcoef(Lv[f], N[f])[0,1]:.6f} "
                  f"ratio={np.mean(Lv[f]/N[f]):.4f} "
                  f"rel_med={np.median(np.abs(Lv[f]-N[f])/np.abs(N[f])):.3e} "
                  f"n={int(f.sum())}")

    # ---- the review's unmeasured risk: does pinning move the WET column? ----
    print("\n" + "=" * 92)
    print("WET-COLUMN IMPACT (bottom_tke_bc sweep). With bottom_tke_bc=False "
          "the dry rows stay\ntridiagonally coupled to the deepest wet "
          "interface, so the pin can propagate upward.")
    print("=" * 92)
    sens = []
    for bbc in (False, True):
        Ko, Ao, _ = _run_closure(br, cfg, sf, st, dry_wmask=False,
                                 bottom_bc=bbc)
        Kn, An, _ = _run_closure(br, cfg, sf, st, dry_wmask=True,
                                 bottom_bc=bbc)
        # "wet, strictly ABOVE each column's deepest wet interface" -- the
        # rows that must NOT move if the fix is local to the seafloor.
        kk = np.arange(NK)[None, None, :]
        above = wet_w & (kk < depth_idx[..., None])
        d = np.abs(Kn[..., :NK] - Ko[..., :NK])
        rel = d[above] / np.maximum(np.abs(Ko[..., :NK][above]), 1e-30)
        at0 = np.abs(Kn[ii, jj, k0] - Ko[ii, jj, k0]) / np.maximum(
            np.abs(Ko[ii, jj, k0]), 1e-30)
        cA, rA, _, _ = pooled(Kn[..., :NK], avt_d[..., 1:1 + NK], wet_w)
        sens.append((bbc, Ko, Kn))
        print(f"  bottom_tke_bc={str(bbc):5s} | ABOVE-seafloor wet rows: "
              f"n={int(above.sum())} max|dK|/K={rel.max():.3e} "
              f"median={np.median(rel):.3e} | dist=0 max|dK|/K={at0.max():.3e}"
              f" | pooled avt ON corr={cA:.6f} ratio={rA:.4f}")
    (_, Ko_F, Kn_F), (_, Ko_T, Kn_T) = sens
    print(f"  DIRECT bottom-BC sensitivity, one variable (bottom_tke_bc T vs "
          f"F):\n    mask OFF: max|dK| = {np.abs(Ko_T - Ko_F).max():.3e}"
          f"    mask ON : max|dK| = {np.abs(Kn_T - Kn_F).max():.3e}"
          f"    (the mask's OWN max|dK| = {np.abs(Kn_F - Ko_F).max():.3e})")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

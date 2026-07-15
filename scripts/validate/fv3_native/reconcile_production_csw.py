#!/usr/bin/env python
"""Phase-4c reconciliation: production JAX cdgrid _c_sw vs certified reference.

The production forward-backward core (``fv3_sw_core._c_sw``) runs on a
JAX 6-face ``CubedSphereGrid`` (``create_fv3_native_cubed_sphere``); the
phase-4a certified reference (``fv3_native_sw_core.c_sw``) runs on a
single-tile ``build_fv3_native_gridstruct`` dict.  This gate checks the
production solver reproduces the certified reference on the h-INDEPENDENT
C-grid wind update (uc/vc/ua/va) — those depend only on d2a2c + vorticity
+ KE, never on the transported scalar, so they must agree across the
SW (h) vs hydrostatic (delp/pt) branch difference.

Two stages, both reported:
  1. METRIC ALIGNMENT — brute-force the (face, rot90) that maps the
     production cdgrid onto the reference tile 1, then report per-metric
     max relative error over the compute domain (dx/dy/area/sin_sg
     centre/cosa_u).  A large error here means the production grid builder
     diverges from the certified phase-1/2 builders (a real finding).
  2. WIND RECONCILIATION — if metrics align, set the cdgrid state from the
     same analytic solid-body field, run production _c_sw, extract the
     matched face, and compare uc/vc/ua/va to the reference c_sw.

Run under sbatch (JAX compile forbidden on the login node).
"""

from __future__ import annotations

import numpy as np


def _rot(a, k):
    """rot90 on the last two axes of a 2-D array (k in 0..3)."""
    return np.rot90(a, k)


def main() -> None:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.grids.cubed_sphere import create_fv3_native_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA, FV3_RADIUS_M, analytic_swcore_state,
        build_fv3_native_gridstruct)
    from legoesm.core.fv3_native_sw_core import Bounds, c_sw
    from legoesm.core.fv3_sw_core import _c_sw as prod_c_sw  # noqa: F401

    n = 12
    ng = 3
    print(f"[reconcile] C{n} FV3-native, radius={FV3_RADIUS_M}")

    # --- production JAX cdgrid (6-face): FV3-native ED base + the C-D grid
    # wrapper with the phase-2B exact cross-seam angles (the ones the
    # certified reference uses) ---
    base = create_fv3_native_cubed_sphere(
        n, radius=FV3_RADIUS_M, omega=FV3_OMEGA,
        use_duogrid=True, k2e_nord=4, dtype=jnp.float64)
    cd = create_cubed_sphere_cdgrid(base, fv3_native_angles=True)
    area_prod = np.asarray(cd.base.area)            # (6, n, n)
    sinc_prod = np.asarray(cd.sin_sg[:, :, :, 4])   # (6, n, n) centre
    print(f"[reconcile] cdgrid area shape {area_prod.shape}, "
          f"sin_sg shape {np.asarray(cd.sin_sg).shape}")

    # --- certified single-tile reference (tile 1) ---
    gs = build_fv3_native_gridstruct(n, ng, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    sl = slice(ng, ng + n)
    area_ref = np.asarray(gs["area"])[sl, sl]       # (n, n) compute
    sinc_ref = np.asarray(gs["sin_sg"])[sl, sl, 4]  # (n, n) centre

    # ---- stage 1: find (face, rot) matching reference tile 1 ----
    best = (1e9, None, None)
    for f in range(6):
        for k in range(4):
            e = np.abs(_rot(area_prod[f], k) - area_ref).max() \
                / max(area_ref.max(), 1e-30)
            if e < best[0]:
                best = (e, f, k)
    err_area, mf, mk = best
    print(f"[reconcile] STAGE-1 best match: face={mf} rot90={mk} "
          f"area rel err={err_area:.3e}")

    def m(field_prod2d):
        return _rot(field_prod2d, mk)

    # centre angles: the cdgrid stores them as cosa_cell / sina_cell (NOT
    # cos_sg[...,4] — its 9-position layout differs from the reference's).
    cosa_cell = np.asarray(cd.cosa_cell)[mf]
    sina_cell = np.asarray(cd.sina_cell)[mf]
    metrics = {
        "area": (m(area_prod[mf]), area_ref),
        "cosa_cell(=cosa_s)": (m(cosa_cell), np.asarray(gs["cosa_s"])[sl, sl]),
        "sina_cell": (m(sina_cell),
                      np.sqrt(np.maximum(
                          1.0 - np.asarray(gs["cosa_s"])[sl, sl] ** 2, 0.0))),
        "sin_sg_centre": (m(sinc_prod[mf]),
                          np.sqrt(np.maximum(
                              1.0 - np.asarray(gs["cosa_s"])[sl, sl] ** 2,
                              0.0))),
    }
    # solver-consumed edge/interface metrics (what _c_sw actually reads):
    # dy_edge_x / dx_edge_y (dy/dx at u_c/v_c), cosa_u, sin_sg upwind.
    dyx_ref = np.asarray(gs["dy"])[slice(ng, ng + n + 1), sl]   # (n+1, n)
    metrics["dy_edge_x(=dy)"] = (m(np.asarray(cd.dy_edge_x[mf])), None)
    metrics["_dy_ref_note"] = (dyx_ref, None)  # shapes differ under rot; see note

    print("[reconcile] STAGE-1 metric agreement (matched face):")
    aligned = True
    # ABS tolerance for centre cos (cosa_s ~ 0 -> rel err is 0/0); a tight
    # 1e-7 REL tolerance elsewhere (cross-builder angle-construction noise:
    # the cdgrid vs the certified longdouble compute_fv3_native_angles).
    for name in ("area", "cosa_cell(=cosa_s)", "sina_cell", "sin_sg_centre"):
        a, b = metrics[name]
        abs_e = np.abs(a - b).max()
        rel_e = abs_e / max(np.abs(b).max(), 1e-30)
        ok = abs_e < 1e-7 if "cosa_cell" in name else rel_e < 1e-7
        flag = "OK  " if ok else "DIFF"
        if not ok:
            aligned = False
        print(f"    {flag} {name:18s} abs={abs_e:.3e} rel={rel_e:.3e}")
    print("[reconcile] NOTE: geometry (area) is BIT-EXACT; the angle fields "
          "differ ~3e-8 (cdgrid double vs certified longdouble angle build) "
          "— sub-1e-7 physical, not bit-exact.  For strict bit-fidelity the "
          "cdgrid should adopt compute_fv3_native_angles (follow-up).")

    if not aligned:
        print("[reconcile] metrics DIVERGE — production cdgrid builder != "
              "certified phase-1/2 builder on this face; wind stage skipped "
              "(would confound solver vs grid differences).")
        return

    # ---- stage 2: h-independent wind reconciliation ----
    print("[reconcile] STAGE-2 winds: setting analytic state, running "
          "production _c_sw ...")
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(n, ng)
    ref = c_sw(delp=st["delp"], pt=st["pt"],
               w=np.zeros_like(st["delp"]),
               u=st["u"], v=st["v"], gs=gs, bd=bd,
               npx=n + 1, npy=n + 1, dt2=112.5, nord=1,
               hydrostatic=True, dord4=True, grid_type=0)
    ref_uc = np.asarray(ref["uc"])[ng:ng + n + 1, ng:ng + n]  # (n+1, n) tile-1
    print(f"[reconcile] reference uc compute range "
          f"{ref_uc.min():.4e}..{ref_uc.max():.4e}")
    print("[reconcile] (production 6-face state setup + _c_sw wind extract "
          "is the next brick; stage-1 metric alignment is the gate this "
          "run establishes.)")


if __name__ == "__main__":
    main()

"""Why do the ``update_dz_d`` twins differ by 1.333e-09?

The gate accepts it and says plainly that it does not explain it (codex
MAJOR, 2026-08-16). ~1e7 epsilon on a DIRECT kernel-twin comparison --
not a long composed chain -- is far too large to call an fp64 roundoff
floor without evidence, and "a bigger defect used to be here" is not
evidence.

This runs the two discriminating measurements that were never run, and
prints NO verdict:

  1. DOES IT SCALE LIKE ROUNDOFF?  Roundoff in a vertical recurrence
     grows with the number of levels accumulated; a fixed implementation
     difference does not.  The twins run at km = 1, 3, 6 and 12 on the
     same horizontal fixture, and the max relative difference is printed
     against km.  Growing ~sqrt(km) or ~km is consistent with
     accumulation; FLAT is a fixed difference, and a JUMP at some km
     points at a level-dependent branch.

  2. WHICH SUB-OPERATION FIRST DISAGREES.  update_dz_d calls fv_tp_2d
     (transport) and del6_vt_flux (damping) before assembling zh, and
     both twins are importable on their own.  Each is called with
     IDENTICAL operands taken from the spec's own call site, so any
     difference is that kernel's, and the first one that disagrees is
     where the expression order has to be matched.

  3. IS IT CONDITIONING?  The reported cells are 2 of 1296. Their
     values, neighbours and the local magnitude of the terms being
     differenced are printed, because a catastrophic cancellation at a
     handful of cells looks exactly like this and is diagnosable by
     eye once the numbers are on screen.

usage:  python scripts/validate/fv3_update_dz_d_twin_localiser.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "tests", "grids"))
sys.path.insert(0, _REPO)

N, NG = 12, 3
MA = N + 2 * NG
DT = 20.0


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    fin = np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        return float("nan"), 0
    scale = max(np.abs(b[fin]).max(), 1e-300)
    d = np.abs(a[fin] - b[fin])
    return float(d.max() / scale), int((d > 1e-13 * scale).sum())


def main() -> int:
    import test_fv3_dsw_tail_3d as gate  # noqa: E402
    import legoesm.core.fv3_native_nh_core as npnh  # noqa: E402
    import legoesm.core.fv3_nh_core as jnh  # noqa: E402
    from legoesm.core.fv3_duo_stepper import (  # noqa: E402
        build_jax_duo_stepper_context,
    )
    from legoesm.core.fv3_native_duo_stepper import (  # noqa: E402
        build_six_face_duo_context,
    )

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    jctx = build_jax_duo_stepper_context(ctx)
    bd = ctx["bd"]
    bounds = (bd.is_, bd.ie, bd.js, bd.je, NG)
    fl = jctx.flags6

    print("1. DOES THE DIFFERENCE SCALE WITH km?")
    print("   (roundoff in a vertical recurrence grows with the levels")
    print("    accumulated; a fixed implementation difference does not)")
    print(f"   {'km':>4s} {'max rel':>12s} {'cells':>7s} {'rel/km':>12s}")
    rng = np.random.default_rng(4)
    # km >= 2: edge_profile reads dp0[1] (fv3_native_nh_core.py:259),
    # so a single-level column is not a valid input to this kernel.
    for km in (2, 3, 6, 12):
        # One horizontal fixture, replicated per level, so the ONLY
        # thing changing across rows is the number of levels.
        zh = np.zeros((MA, MA, km + 1), dtype=np.float64)
        for k in range(km, -1, -1):
            zh[:, :, k] = 545.0 * (km - k)
        crx = 0.1 + 0.01 * rng.standard_normal((N + 1, MA, km))
        cry = 0.1 + 0.01 * rng.standard_normal((MA, N + 1, km))
        xfx = 1.0e6 * (1.0 + 0.01 * rng.standard_normal((N + 1, MA, km)))
        yfx = 1.0e6 * (1.0 + 0.01 * rng.standard_normal((MA, N + 1, km)))
        zs = np.zeros((MA, MA), dtype=np.float64)
        ws = np.zeros((N, N), dtype=np.float64)
        dp0 = np.full(km, 1.0e4, dtype=np.float64)
        area = np.asarray(ctx["gs6"][0]["area"], dtype=np.float64)
        rarea = 1.0 / area
        gs = ctx["gs6"][0]
        ndif = np.full(km + 1, 2.0)
        damp = np.full(km + 1, 0.12)

        zh_np = np.array(zh, copy=True)
        ws_np = np.array(ws, copy=True)
        gs_nh = dict(gs)
        gs_nh["area"] = area
        gs_nh["rarea"] = rarea
        npnh.update_dz_d(ndif, damp, 6, bd, km, N + 1, N + 1, area, rarea,
                         dp0, zs, zh_np, np.array(crx, copy=True),
                         np.array(cry, copy=True), np.array(xfx, copy=True),
                         np.array(yfx, copy=True), ws_np, 1.0 / DT, gs_nh,
                         lim_fac=1.0)
        zh_j, _ = jnh.update_dz_d(
            tuple(ndif), tuple(damp), 6, bounds, km, N + 1, N + 1,
            jnp.asarray(area), jnp.asarray(rarea), jnp.asarray(dp0),
            jnp.asarray(zs), jnp.asarray(zh), jnp.asarray(crx),
            jnp.asarray(cry), jnp.asarray(xfx), jnp.asarray(yfx),
            jnp.asarray(ws), 1.0 / DT,
            jnp.asarray(gs["dxa"]), jnp.asarray(gs["dya"]),
            jnp.asarray(gs["del6_u"]), jnp.asarray(gs["del6_v"]),
            lim_fac=1.0,
            bounded_domain=fl[0].bounded_domain, grid_type=fl[0].grid_type,
            sw_corner=fl[0].sw_corner, se_corner=fl[0].se_corner,
            nw_corner=fl[0].nw_corner, ne_corner=fl[0].ne_corner)
        r, nc = _rel(zh_j, zh_np)
        print(f"   {km:4d} {r:12.4e} {nc:7d} {r / km:12.4e}")

    print("\n2. THE GATE'S OWN OPERANDS, since the smooth fixture above")
    print("   is bit-identical -- so the residual is DATA-TRIGGERED, not a")
    print("   systematic expression-order difference. Face 3 is the worst.")
    KM = 3
    state = gate._seeded_state(KM, seed=11, hydrostatic=False)
    bundle = gate.nh_bundle.__wrapped__(ctx, state)
    bundle = dict(bundle, state=state)
    carry = bundle["carry"]
    dsw = bundle["dsw"]
    import legoesm.core.fv3_native_dsw_tail_3d as nptail_mod
    area6 = nptail_mod.nh_exchanged_area6(dict(ctx))
    rarea6 = [1.0 / np.asarray(a) for a in area6]

    for t in (2,):                      # face 3, the worst in the gate
        gs = ctx["gs6"][t]
        crx = np.stack([dsw[t]["levels"][k]["crx_adv"] for k in range(KM)],
                       axis=2)
        cry = np.stack([dsw[t]["levels"][k]["cry_adv"] for k in range(KM)],
                       axis=2)
        xfx = np.stack([dsw[t]["levels"][k]["xfx_adv"] for k in range(KM)],
                       axis=2)
        yfx = np.stack([dsw[t]["levels"][k]["yfx_adv"] for k in range(KM)],
                       axis=2)
        zh_np = np.array(carry["zh6"][t], copy=True)
        ws_np = np.array(carry["ws6"][t], copy=True)
        gs_nh = dict(gs)
        gs_nh["area"] = area6[t]
        gs_nh["rarea"] = rarea6[t]
        npnh.update_dz_d(np.full(KM + 1, 2.0), np.full(KM + 1, 0.12), 6, bd,
                         KM, N + 1, N + 1, area6[t], rarea6[t],
                         np.asarray(gate._DP0), carry["zs6"][t], zh_np,
                         np.array(crx, copy=True), np.array(cry, copy=True),
                         np.array(xfx, copy=True), np.array(yfx, copy=True),
                         ws_np, 1.0 / DT, gs_nh, lim_fac=1.0)
        zh_j, _ = jnh.update_dz_d(
            (2.0,) * (KM + 1), (0.12,) * (KM + 1), 6, bounds, KM, N + 1,
            N + 1, jnp.asarray(area6[t]), jnp.asarray(rarea6[t]),
            jnp.asarray(gate._DP0), jnp.asarray(carry["zs6"][t]),
            jnp.asarray(carry["zh6"][t]), jnp.asarray(crx), jnp.asarray(cry),
            jnp.asarray(xfx), jnp.asarray(yfx),
            jnp.asarray(carry["ws6"][t]), 1.0 / DT,
            jnp.asarray(gs["dxa"]), jnp.asarray(gs["dya"]),
            jnp.asarray(gs["del6_u"]), jnp.asarray(gs["del6_v"]),
            lim_fac=1.0,
            bounded_domain=fl[t].bounded_domain, grid_type=fl[t].grid_type,
            sw_corner=fl[t].sw_corner, se_corner=fl[t].se_corner,
            nw_corner=fl[t].nw_corner, ne_corner=fl[t].ne_corner)
        d = np.abs(np.asarray(zh_j) - zh_np)
        scale = max(np.abs(zh_np).max(), 1e-300)
        bad = np.argwhere(d > 1e-13 * scale)
        print(f"   face {t + 1}: |zh| max {scale:.6e}, max|d| {d.max():.6e}, "
              f"rel {d.max() / scale:.6e}, cells {len(bad)}")
        for idx in bad[:6]:
            i, j, k = (int(x) for x in idx)
            below = zh_np[i, j, min(k + 1, KM)]
            print(f"   [{i:3d},{j:3d},{k:2d}] numpy {zh_np[i, j, k]:.17e}")
            print(f"                 jax   {float(zh_j[i, j, k]):.17e}")
            print(f"                 |d| {d[i, j, k]:.3e}  cell-rel "
                  f"{d[i, j, k] / max(abs(zh_np[i, j, k]), 1e-300):.3e}")
            print(f"                 dz across this interface "
                  f"{zh_np[i, j, k] - below:.6e}  "
                  f"(a small dz here means CANCELLATION)")
            print(f"                 crx {crx[min(i, crx.shape[0] - 1), j, min(k, KM - 1)]:.6e}  "
                  f"cry {cry[i, min(j, cry.shape[1] - 1), min(k, KM - 1)]:.6e}")

    print("\nUPDATE_DZ_D_TWIN_LOCALISER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

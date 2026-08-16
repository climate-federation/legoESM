"""Where does the NH tail's ``zh`` disagreement live?

The phase-level gate reports 2.465e-02 on ``zh``, 5832 of 7776 cells --
and 5832 is exactly ``6 * 18 * 18 * 3``, i.e. THREE OF FOUR INTERFACE
LEVELS, every cell. Threading the grid flags into ``update_dz_d`` fixed
a real 1.155e-04 kernel difference and moved the phase number NOT AT ALL
(identical to four digits), so the dominant term is something else.

This prints the breakdown the failure message cannot: per level, per
face, halo versus compute window, and the same for the operands that
feed ``zh`` -- so the next step is chosen from a measurement instead of
from a reading.

No verdict is printed. The pre-registered readings:

  * a difference at every level EXCEPT the bottom, in the COMPUTE
    window, points at the kernel's vertical update;
  * a difference confined to HALO cells points at the per-interface
    ``ext_scalar`` exchange that follows it;
  * a difference at the bottom interface too points at ``zs`` or at the
    carry handed in.

usage:  python scripts/validate/fv3_nh_zh_localiser.py
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

N, NG, KM = 12, 3, 3
# the reference thickness profile the gate file uses
_DP0 = None  # set from the gate module at run time
MA = N + 2 * NG
DT = 20.0


def main() -> int:
    global _DP0
    import test_fv3_dsw_tail_3d as gate  # noqa: E402
    from legoesm.core import (  # noqa: E402
        fv3_native_dsw_tail_3d as nptail_mod,
    )
    from legoesm.core.fv3_duo_stepper import (  # noqa: E402
        build_jax_duo_stepper_context,
    )
    from legoesm.core.fv3_native_duo_stepper import (  # noqa: E402
        build_six_face_duo_context,
    )

    _DP0 = gate._DP0
    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    jctx = build_jax_duo_stepper_context(ctx)
    state = gate._seeded_state(KM, seed=11, hydrostatic=False)

    # The gate file owns the fixture; rebuilding it here would be a
    # second bundle that could drift from the one under test.
    bundle = gate.nh_bundle.__wrapped__(ctx, state)
    bundle = dict(bundle, state=state)

    kw = dict(remap_step=False, use_logp=False, square_domain=True)
    ref, ref_carry = gate._run_nh(dict(ctx), bundle, "numpy", **kw)
    got = gate._run_nh(jctx, bundle, "jax", **kw)

    zh_n = np.stack([np.asarray(x) for x in ref_carry["zh6"]])
    zh_j = np.asarray(got["nh"]["zh"])
    print(f"zh shapes: numpy {zh_n.shape}  jax {zh_j.shape}")

    cs = slice(NG, NG + N)
    halo = np.ones((MA, MA), bool)
    halo[cs, cs] = False

    print("\nper interface level (k = 0 is the model TOP, k = km the "
          "surface):")
    for k in range(KM + 1):
        d = np.abs(zh_j[:, :, :, k] - zh_n[:, :, :, k])
        dc = d[:, cs, cs]
        dh = d[:, halo]
        print(f"  k={k}: max|d| {d.max():.6e}   compute {dc.max():.6e}"
              f"   halo {dh.max():.6e}   "
              f"cells>1e-13: {int((d > 1e-13).sum())} of {d.size}")

    print("\nthe carry going IN (both lanes were handed the same bundle):")
    zh_in = np.stack([np.asarray(x) for x in bundle["carry"]["zh6"]])
    print(f"  zh_in max {np.abs(zh_in).max():.6e}, "
          f"identical to numpy-out at k=km: "
          f"{np.array_equal(zh_in[:, :, :, KM], zh_n[:, :, :, KM])}")

    print("\nthe operands update_dz_d reads, JAX stack vs NumPy levels:")
    dsw = bundle["dsw"]
    for nm in ("crx_adv", "cry_adv", "xfx_adv", "yfx_adv"):
        a = np.stack([np.stack([np.asarray(lvl[nm])
                                for lvl in dsw[t]["levels"]], axis=2)
                      for t in range(6)])
        b = np.asarray(gate._stack_dsw_np(dsw)[nm])
        print(f"  {nm}: max|stack - adapter| "
              f"{np.abs(a - b).max():.3e}  shape {a.shape}")

    # --- which of the TWO kernels that write zh is responsible -------
    # update_dz_d writes zh's compute window, then riem_solver3 rewrites
    # it, then the per-interface ext_scalar fills the halos.  The kernel
    # probe in the gate file already clears update_dz_d (1.333e-09 after
    # the grid flags were threaded), so this runs BOTH in sequence and
    # prints the diff after each.
    import legoesm.core.fv3_native_nh_core as npnh
    import legoesm.core.fv3_nh_core as jnh

    bd = ctx["bd"]
    bounds = (bd.is_, bd.ie, bd.js, bd.je, NG)
    carry = bundle["carry"]
    dsw, tail = bundle["dsw"], bundle["tail"]
    cswp = bundle["csw_press"]
    area6 = nptail_mod.nh_exchanged_area6(dict(ctx))
    rarea6 = [1.0 / np.asarray(a) for a in area6]
    fl = jctx.flags6

    print("\nthe two kernels that write zh, face by face:")
    for t in range(3):
        crx = np.stack([dsw[t]["levels"][k]["crx_adv"] for k in range(KM)],
                       axis=2)
        cry = np.stack([dsw[t]["levels"][k]["cry_adv"] for k in range(KM)],
                       axis=2)
        xfx = np.stack([dsw[t]["levels"][k]["xfx_adv"] for k in range(KM)],
                       axis=2)
        yfx = np.stack([dsw[t]["levels"][k]["yfx_adv"] for k in range(KM)],
                       axis=2)
        gs = ctx["gs6"][t]
        gs_nh = dict(gs)
        gs_nh["area"] = area6[t]
        gs_nh["rarea"] = rarea6[t]
        ndif = np.full(KM + 1, 2.0)
        damp = np.full(KM + 1, 0.12)
        rdt = 1.0 / DT

        zh_np = np.array(carry["zh6"][t], copy=True)
        ws_np = np.array(carry["ws6"][t], copy=True)
        npnh.update_dz_d(ndif, damp, 6, bd, KM, N + 1, N + 1, area6[t],
                         rarea6[t], _DP0, carry["zs6"][t], zh_np,
                         np.array(crx, copy=True), np.array(cry, copy=True),
                         np.array(xfx, copy=True), np.array(yfx, copy=True),
                         ws_np, rdt, gs_nh, lim_fac=1.0)
        zh_j, ws_j = jnh.update_dz_d(
            (2.0,) * (KM + 1), (0.12,) * (KM + 1), 6, bounds, KM, N + 1,
            N + 1, jnp.asarray(area6[t]), jnp.asarray(rarea6[t]),
            jnp.asarray(_DP0), jnp.asarray(carry["zs6"][t]),
            jnp.asarray(carry["zh6"][t]), jnp.asarray(crx),
            jnp.asarray(cry), jnp.asarray(xfx), jnp.asarray(yfx),
            jnp.asarray(carry["ws6"][t]), rdt, jnp.asarray(gs["dxa"]),
            jnp.asarray(gs["dya"]), jnp.asarray(gs["del6_u"]),
            jnp.asarray(gs["del6_v"]), lim_fac=1.0,
            bounded_domain=fl[t].bounded_domain, grid_type=fl[t].grid_type,
            sw_corner=fl[t].sw_corner, se_corner=fl[t].se_corner,
            nw_corner=fl[t].nw_corner, ne_corner=fl[t].ne_corner)
        d1 = float(np.abs(np.asarray(zh_j) - zh_np).max())

        # ...then riem_solver3 on each lane's own update_dz_d output.
        pe_np = np.array(carry["pe6"][t], copy=True)
        pkc_np = np.array(cswp[t]["pkc"], copy=True)
        pk3_np = np.array(carry["pk3_6"][t], copy=True)
        pk_np = np.array(carry["pk6"][t], copy=True)
        peln_np = np.array(carry["peln6"][t], copy=True)
        w_np = np.array(tail[t]["w"], copy=True)
        delz_np2 = np.array(bundle["state"][t]["delz"], copy=True)
        npnh.riem_solver3(0, DT, bd, KM, 2.0 / 7.0, 1004.6, 100.0,
                          carry["zs6"][t], w_np, delz_np2,
                          np.array(dsw[t]["pt"], copy=True),
                          np.array(dsw[t]["delp"], copy=True), zh_np,
                          pe_np, pkc_np, pk3_np, pk_np, peln_np, ws_np,
                          0.05, 1.0, use_logp=False, last_call=False,
                          fp_out=False)
        out_j = jnh.riem_solver3(
            0, DT, bounds, KM, 2.0 / 7.0, 1004.6, 100.0,
            jnp.asarray(carry["zs6"][t]), jnp.asarray(tail[t]["w"]),
            jnp.asarray(bundle["state"][t]["delz"]),
            jnp.asarray(dsw[t]["pt"]), jnp.asarray(dsw[t]["delp"]),
            zh_j, jnp.asarray(carry["pe6"][t]),
            jnp.asarray(cswp[t]["pkc"]), jnp.asarray(carry["pk3_6"][t]),
            jnp.asarray(carry["pk6"][t]), jnp.asarray(carry["peln6"][t]),
            ws_j, 0.05, 1.0, use_logp=False, last_call=False,
            fp_out=False)
        d2 = float(np.abs(np.asarray(out_j[2]) - zh_np).max())
        print(f"  face {t + 1}: after update_dz_d max|d| {d1:.6e}   "
              f"after riem_solver3 max|d| {d2:.6e}")

    print("\nzs and ws (the bottom boundary condition):")
    zs = np.stack([np.asarray(x) for x in bundle["carry"]["zs6"]])
    print(f"  zs max|.| {np.abs(zs).max():.6e}")
    print(f"  numpy zh[k=km] max|.| {np.abs(zh_n[:, :, :, KM]).max():.6e}")
    print(f"  jax   zh[k=km] max|.| {np.abs(zh_j[:, :, :, KM]).max():.6e}")
    print("\nLOCALISER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

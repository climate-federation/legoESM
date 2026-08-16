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
        fv3_dsw_tail_3d as jtail_mod,
    )
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

    # ⛔ THE ONE OPERAND THE DIRECT COMPARISON DOES NOT TEST.  Both
    # kernels agree to <=1.05e-04 when called with the NUMPY exchanged
    # area, while the phase differs by 2.590e+02 in the compute window --
    # so the phase must hand them something else, and `area` is the only
    # operand the phase COMPUTES rather than receives.
    a_j = np.asarray(jtail_mod.nh_exchanged_area6(jctx))
    a_n = np.stack([np.asarray(a) for a in area6])
    print("\nthe exchanged area, JAX vs NumPy (each lane's own):")
    print(f"  max|d| {np.abs(a_j - a_n).max():.6e}   "
          f"max|numpy| {np.abs(a_n).max():.6e}   "
          f"cells>1e-13: {int((np.abs(a_j - a_n) > 1e-13).sum())} "
          f"of {a_n.size}")
    sent_j = int(np.isclose(np.abs(a_j), 1.0e8, rtol=1e-12).sum())
    sent_n = int(np.isclose(np.abs(a_n), 1.0e8, rtol=1e-12).sum())
    print(f"  BIG_NUMBER sentinels surviving: jax {sent_j}  numpy {sent_n}")

    print("\nthe two kernels that write zh, face by face:")
    # ALL SIX faces, not three: the first version stopped at three
    # "enough to localise" and the trend across them was 0.0, 2.27e-13,
    # 2.85e-06 -- two decades per face, which makes the untested half
    # the likely home of the phase-level term.
    for t in range(6):
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
        # NOT `zh_j` / `ws_j`: those name the PHASE's outputs, which
        # every comparison below still needs. Rebinding them here made
        # the scale prints and the phase-vs-chain diff compare ONE
        # face's 3-D kernel result against the six-face 4-D arrays, and
        # that single shadow is the whole reason this hunt produced a
        # set of measurements that could not all be true.
        zh_k1, ws_k1 = jnh.update_dz_d(
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
        d1 = float(np.abs(np.asarray(zh_k1) - zh_np).max())

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
            zh_k1, jnp.asarray(carry["pe6"][t]),
            jnp.asarray(cswp[t]["pkc"]), jnp.asarray(carry["pk3_6"][t]),
            jnp.asarray(carry["pk6"][t]), jnp.asarray(carry["peln6"][t]),
            ws_k1, 0.05, 1.0, use_logp=False, last_call=False,
            fp_out=False)
        d2 = float(np.abs(np.asarray(out_j[2]) - zh_np).max())
        print(f"  face {t + 1}: after update_dz_d max|d| {d1:.6e}   "
              f"after riem_solver3 max|d| {d2:.6e}")

    # --- JAX PHASE vs JAX CHAIN, the technique that cracked module 4 --
    # Every component agrees (areas 0.0, operands 0.0, both kernels
    # <=1.05e-04) and the phase still differs by 2.590e+02, so the phase
    # is doing something the hand-composed chain is not.  This runs the
    # SAME JAX kernels in the phase's own order, per face, and compares
    # against what the phase returned.
    print("\nJAX phase vs the SAME JAX kernels composed by hand:")
    zh_chain = []
    for t in range(6):
        crx = np.stack([dsw[t]["levels"][k]["crx_adv"] for k in range(KM)],
                       axis=2)
        cry = np.stack([dsw[t]["levels"][k]["cry_adv"] for k in range(KM)],
                       axis=2)
        xfx = np.stack([dsw[t]["levels"][k]["xfx_adv"] for k in range(KM)],
                       axis=2)
        yfx = np.stack([dsw[t]["levels"][k]["yfx_adv"] for k in range(KM)],
                       axis=2)
        gs = ctx["gs6"][t]
        zh_t, ws_t = jnh.update_dz_d(
            (2.0,) * (KM + 1), (0.12,) * (KM + 1), 6, bounds, KM, N + 1,
            N + 1, jnp.asarray(area6[t]), jnp.asarray(rarea6[t]),
            jnp.asarray(_DP0), jnp.asarray(carry["zs6"][t]),
            jnp.asarray(carry["zh6"][t]), jnp.asarray(crx),
            jnp.asarray(cry), jnp.asarray(xfx), jnp.asarray(yfx),
            jnp.asarray(carry["ws6"][t]), 1.0 / DT,
            jnp.asarray(gs["dxa"]), jnp.asarray(gs["dya"]),
            jnp.asarray(gs["del6_u"]), jnp.asarray(gs["del6_v"]),
            lim_fac=1.0,
            bounded_domain=fl[t].bounded_domain, grid_type=fl[t].grid_type,
            sw_corner=fl[t].sw_corner, se_corner=fl[t].se_corner,
            nw_corner=fl[t].nw_corner, ne_corner=fl[t].ne_corner)
        out = jnh.riem_solver3(
            0, DT, bounds, KM, 2.0 / 7.0, 1004.6, 100.0,
            jnp.asarray(carry["zs6"][t]), jnp.asarray(tail[t]["w"]),
            jnp.asarray(bundle["state"][t]["delz"]),
            jnp.asarray(dsw[t]["pt"]), jnp.asarray(dsw[t]["delp"]),
            zh_t, jnp.asarray(carry["pe6"][t]),
            jnp.asarray(cswp[t]["pkc"]), jnp.asarray(carry["pk3_6"][t]),
            jnp.asarray(carry["pk6"][t]), jnp.asarray(carry["peln6"][t]),
            ws_t, 0.05, 1.0, use_logp=False, last_call=False,
            fp_out=False)
        zh_chain.append(np.asarray(out[2]))
    zh_chain = np.stack(zh_chain)
    # The chain has to include the EXCHANGE to be comparable: the phase
    # applies it per interface level after the kernels, and it rewrites
    # the compute ring as well as the halo (the first version of this
    # comparison omitted it and duly reported 4.213e+05, which measured
    # the exchange rather than any defect).
    zh_chain_j = jnp.asarray(zh_chain)
    for k in range(KM + 1):
        zh_chain_j = zh_chain_j.at[:, :, :, k].set(
            # ONE operator, named outright: the conditional this used to
            # carry could silently pick a different function than the
            # phase calls, which is not a thing a comparison may leave
            # open.  Signature read from source: (planes6, ctx).
            jtail_mod._ext_scalar_planes_6(zh_chain_j[:, :, :, k], jctx))
    # SHAPE GUARD, earned: a rebound name upstream made this line
    # compare a single face's 3-D kernel output against the six-face
    # 4-D stack, and numpy broadcast it into a number instead of
    # raising. Every comparison in this probe now states the shapes it
    # is comparing.
    if zh_j.shape != np.asarray(zh_chain_j).shape:
        raise SystemExit(
            f"REFUSING TO COMPARE: phase zh {zh_j.shape} vs chain "
            f"{np.asarray(zh_chain_j).shape} -- one of them is not what "
            f"its name says.")
    d_post = np.abs(zh_j - np.asarray(zh_chain_j))
    print(f"  after the exchange:  max|phase - chain| {d_post.max():.6e}"
          f"   compute {d_post[:, cs, cs].max():.6e}")

    # MAGNITUDES, not only differences.  Four measurements here cannot
    # all be true -- the two phases agree to 2.6e+02 while each differs
    # from the hand-composed chain by 4.2e+05 -- and the missing number
    # in every one of them is the SCALE.  If the chain is ~1e+05 where
    # the phases are ~1e+03 then the chain exploded and its "agreement"
    # with the NumPy kernels only says both hand-calls are equally
    # wrong; if all three are the same size, the disagreement is real
    # and lives in an operator the chain omits.  One print separates
    # them, and not having it is why this hunt ran as long as it did.
    # --- THE COMPARISON THAT NEEDED NO RECONSTRUCTION -----------------
    # Both lanes' OWN phases, on ONE shared bundle, reporting zh after
    # each of the two kernels that write it. The hand-built chain was a
    # second implementation; this is the phases themselves, so a
    # difference here is in the phase and nowhere else.
    print("\nstage-by-stage, both PHASES on the same bundle:")
    np_stage = {}

    def _hook(name, t, arr):
        np_stage.setdefault(name, {})[t] = np.array(arr, copy=True)

    _c2 = {k: [np.array(x, copy=True) for x in v] if isinstance(v, list) else v
           for k, v in bundle["carry"].items()}
    nptail_mod.dgrid_nh_pressure_phase_3d(
        dict(ctx), gate.deepcopy_faces(bundle["csw_press"]),
        gate.deepcopy_faces(bundle["dsw"]), gate.deepcopy_faces(bundle["tail"]),
        _c2, KM, dt=DT, ptop=100.0, akap=2.0 / 7.0, cp_air=1004.6,
        p_fac=0.05, a_imp=1.0, dp0=_DP0,
        delz6=[np.array(f["delz"], copy=True) for f in bundle["state"]],
        stage_hook=_hook, **kw)
    j_out = gate.jtail.dgrid_nh_pressure_phase_3d(
        jctx, gate.stack_np(bundle["csw_press"]),
        gate._stack_dsw_np(bundle["dsw"]), gate.stack_np(bundle["tail"]),
        gate._nh_stack_carry(bundle["carry"]), KM, dt=DT, ptop=100.0,
        akap=2.0 / 7.0, cp_air=1004.6, p_fac=0.05, a_imp=1.0,
        dp0=jnp.asarray(_DP0),
        delz=jnp.asarray(np.stack([np.asarray(f["delz"])
                                   for f in bundle["state"]])),
        return_stages=True, **kw)
    for sname in ("after_update_dz_d", "after_riem_solver3"):
        for t in range(6):
            a = np.asarray(j_out["stages"]["zh"][t][sname])
            b = np_stage[f"S_nh_{sname}"][t]
            if a.shape != b.shape:
                raise SystemExit(f"REFUSING: {sname} face {t} shapes "
                                 f"{a.shape} vs {b.shape}")
            d = np.abs(a - b)
            print(f"  {sname:22s} face {t + 1}: max|d| {d.max():.6e}"
                  f"   compute {d[cs, cs].max():.6e}")

    print("\nSCALES (the number missing from every comparison above):")
    for nm, arr in (("numpy phase", zh_n), ("jax phase", zh_j),
                    ("jax chain", np.asarray(zh_chain_j))):
        a = np.abs(arr)
        print(f"  {nm:11s} max {a.max():.6e}  compute max "
              f"{a[:, cs, cs].max():.6e}  mean {a.mean():.6e}")
    print("\nzs and ws (the bottom boundary condition):")
    zs = np.stack([np.asarray(x) for x in bundle["carry"]["zs6"]])
    print(f"  zs max|.| {np.abs(zs).max():.6e}")
    print(f"  numpy zh[k=km] max|.| {np.abs(zh_n[:, :, :, KM]).max():.6e}")
    print(f"  jax   zh[k=km] max|.| {np.abs(zh_j[:, :, :, KM]).max():.6e}")
    print("\nLOCALISER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "tests", "grids"))
sys.path.insert(0, _REPO)

N, NG, KM = 12, 3, 3
MA = N + 2 * NG
DT = 20.0


def main() -> int:
    import test_fv3_dsw_tail_3d as gate  # noqa: E402
    from legoesm.core.fv3_duo_stepper import (  # noqa: E402
        build_jax_duo_stepper_context,
    )
    from legoesm.core.fv3_native_duo_stepper import (  # noqa: E402
        build_six_face_duo_context,
    )

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

    print("\nzs and ws (the bottom boundary condition):")
    zs = np.stack([np.asarray(x) for x in bundle["carry"]["zs6"]])
    print(f"  zs max|.| {np.abs(zs).max():.6e}")
    print(f"  numpy zh[k=km] max|.| {np.abs(zh_n[:, :, :, KM]).max():.6e}")
    print(f"  jax   zh[k=km] max|.| {np.abs(zh_j[:, :, :, KM]).max():.6e}")
    print("\nLOCALISER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

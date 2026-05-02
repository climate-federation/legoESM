"""Iter-899 investigation script: empirically determine the actual
index convention of `_ppm_reconstruct_1d`'s production input strip.

**Background.**  The iter-892 boundary overrides at q_face[2,3,n+1,n+2]
were designed under the documented assumption that q[k]=q1(k-1) (i.e.,
halo=1 effective on the left).  But production extracts strips from
`_pad_halo_auto_h2(...)[:, :, 2:-2]`, where `_pad_halo_auto_h2` returns
shape (6, n+4, n+4) with halo=2 on each side.  This script verifies
empirically what production actually feeds the leaf.

**Method.**  Place a known scalar that varies as h[face, i, j] = i +
0.001*j on the interior, then pad with `_pad_halo_auto_h2`, and inspect
strip[face, k, j_interior] for k=0..n+3 on face 0.  The interior cell
i=0 (cube-edge) has value 0 + 0.001*j_int.  Whichever k satisfies
`strip[k] ≈ 0 + 0.001*j_int` reveals how many halo cells precede the
interior on the left.

**Finding (n=12, j=interior centre 6):**

    strip[ 0] = +10.0058   <- halo cell from neighbour face
    strip[ 1] = +11.0059   <- halo cell from neighbour face
    strip[ 2] =  +0.0060   <- INTERIOR cell 0 (cube-edge cell)
    strip[ 3] =  +1.0060
    ...
    strip[12] = +10.0060   <- INTERIOR cell n-1
    strip[13] = +11.0060   <- INTERIOR cell n-1 was at strip[12], so this is wait...

Wait, the strip on face 0's WEST edge gets halo from face 4 (per
CONNECTIVITY).  On face 4 the same scalar i+0.001*j gives values up
to i=11 in interior i.  After axis-swap and reverse for face 4 -> face 0
WEST, the halo cells receive face 4's NORTH edge column values.  Don't
read too much physics into the halo numbers — the IMPORTANT finding is
that strip[2] is interior cell 0, NOT strip[1].

**Conclusion.**  Production strip layout: q[0]=q1(-2), q[1]=q1(-1),
q[2]=q1(0), ..., q[n+1]=q1(n-1), q[n+2]=q1(n), q[n+3]=q1(n+1).  This
is **Hypothesis A** (full halo=2 per side), NOT the iter-892 docstring's
**Hypothesis B** (q[0]=q1(-1)).

**Implication.**  Iter-892's overrides at q_face[2,3,n+1,n+2] under
Hypothesis A correspond to Fortran al(0,1,n-1,n) — but iter-892's
formulas implement Fortran al(1,2,npx-1,npx)-style (i.e., the docstring's
intended targets).  This is a 1-cell shift bug that nonetheless improved
W2 v_ll_Linf from 0.159 to 0.132 in iter-893, presumably because the
iter-892 formulas (xt_L and c3/c2/c1 mirrors) are CLOSER to the truth
than the standard 4th-order edge extrapolation, even when placed at
the wrong q_face indices.

**iter-900+ work** (deferred): correct the q_face indices and add the
genuine al(0) override Fortran-faithfully (al(0) = c1*q1(-2) +
c2*q1(-1) + c3*q1(0); under Hypothesis A: c1*q_pad[2] + c2*q_pad[3] +
c3*q_pad[4]).  Risk: the corrected formulas may make W2 worse than
iter-892's empirically-good shift.  Measurement required.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.operators_cdgrid import _pad_halo_auto_h2


def main():
    n = 12
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    h = np.zeros((6, n, n))
    for face in range(6):
        for i in range(n):
            for j in range(n):
                h[face, i, j] = float(i) + 0.001 * float(j)
    h_jax = jnp.asarray(h)

    h_pad = _pad_halo_auto_h2(h_jax, cdgrid)
    print(f"h_pad shape = {h_pad.shape}")

    j_int = n // 2  # interior j = 6
    strip = np.asarray(h_pad)[0, :, j_int + 2]
    print(f"\nFace 0 strip at interior j={j_int} (i-axis k=0..{n+3}):")
    for k in range(n + 4):
        print(f"  strip[{k:>2}] = {strip[k]:>+9.5f}")

    target = 0.0 + 0.001 * j_int  # interior cell 0 expected value
    print(f"\nInterior cell i=0 expected value: {target:.5f}")

    found_at = None
    for k in range(4):
        if abs(strip[k] - target) < 1e-6:
            found_at = k
            break
    if found_at is None:
        print(f"WARNING: did not find interior cell i=0 in first 4 strip "
              f"slots.  Either the halo extrapolation is non-trivial or "
              f"this scalar field is not amenable to the test.")
        return

    print(f"Interior cell i=0 found at strip[{found_at}].")
    print(f"  -> strip layout: {found_at} halo cells precede interior.")
    if found_at == 2:
        print(f"  -> Hypothesis A confirmed: q[0]=q1(-2), q[1]=q1(-1), "
              f"q[2]=q1(0).  Production strip has 2 halo cells per side.")
    elif found_at == 1:
        print(f"  -> Hypothesis B confirmed: q[0]=q1(-1), q[1]=q1(0).  "
              f"Production strip has 1 halo cell on left.  This would "
              f"vindicate iter-892's docstring.")
    else:
        print(f"  -> Unexpected layout with {found_at} halo cells.")


if __name__ == "__main__":
    main()
